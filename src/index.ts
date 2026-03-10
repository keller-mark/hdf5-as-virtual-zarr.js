import type { Group, Dataset, Metadata, Filter } from "h5wasm";

/**
 * Hidden HDF5 attributes that should not be transferred to Zarr.
 * From h5netcdf.attrs.
 */
const HIDDEN_ATTRS = new Set([
  "REFERENCE_LIST",
  "CLASS",
  "DIMENSION_LIST",
  "NAME",
  "_Netcdf4Dimid",
  "_Netcdf4Coordinates",
  "_nc3_strict",
  "_NCProperties",
]);

/**
 * Zarr v2 array metadata (.zarray).
 */
interface ZarrArrayMeta {
  zarr_format: 2;
  shape: number[];
  chunks: number[];
  dtype: string;
  compressor: Record<string, unknown> | null;
  fill_value: unknown;
  order: "C" | "F";
  filters: Record<string, unknown>[] | null;
  dimension_separator?: string;
}

/**
 * Zarr v2 group metadata (.zgroup).
 */
interface ZarrGroupMeta {
  zarr_format: 2;
}

/**
 * Reference spec v1 output.
 */
export interface ReferenceSpec {
  version: 1;
  refs: Record<string, string | [string | null, number, number]>;
}

/**
 * Options for SingleHdf5ToZarr.
 */
export interface SingleHdf5ToZarrOptions {
  /**
   * URL of the HDF5 file, used for references.
   * If null, references will use null for the URL.
   */
  url?: string | null;
  /**
   * Include data chunks smaller than this value (in bytes)
   * directly in the output as base64. 0 or negative to disable.
   */
  inlineThreshold?: number;
}

/**
 * Convert HDF5 dtype metadata to Zarr dtype string.
 */
function metadataToZarrDtype(metadata: Metadata): string {
  const { type, size, littleEndian, signed, vlen } = metadata;
  const endian = littleEndian ? "<" : ">";

  // String types
  // type 3 = H5T_STRING
  if (type === 3) {
    if (vlen) {
      return "|O";
    }
    return `|S${size}`;
  }

  // Integer types
  // type 0 = H5T_INTEGER
  if (type === 0) {
    const intFmts: Record<number, string> = { 1: "b", 2: "h", 4: "i", 8: "q" };
    let fmt = intFmts[size];
    if (!fmt) {
      return `${endian}i${size}`;
    }
    if (!signed) {
      fmt = fmt.toUpperCase();
    }
    return `${endian}${fmt}`;
  }

  // Float types
  // type 1 = H5T_FLOAT
  if (type === 1) {
    const floatFmts: Record<number, string> = { 2: "e", 4: "f", 8: "d" };
    const fmt = floatFmts[size] || `f${size}`;
    return `${endian}${fmt}`;
  }

  // Enum types - treat as the base type
  // type 8 = H5T_ENUM
  if (type === 8) {
    const intFmts: Record<number, string> = { 1: "b", 2: "h", 4: "i", 8: "q" };
    let fmt = intFmts[size];
    if (!fmt) {
      return `${endian}i${size}`;
    }
    if (!signed) {
      fmt = fmt.toUpperCase();
    }
    return `${endian}${fmt}`;
  }

  // Compound types
  // type 6 = H5T_COMPOUND
  if (type === 6) {
    return "|V" + size;
  }

  return `${endian}V${size}`;
}

/**
 * Convert h5wasm filter info to Zarr compressor/filters metadata.
 */
function decodeFilters(
  filters: Filter[]
): { compressor: Record<string, unknown> | null; zarrFilters: Record<string, unknown>[] | null } {
  const zarrFilters: Record<string, unknown>[] = [];
  let compressor: Record<string, unknown> | null = null;

  for (const filter of filters) {
    if (filter.id === 1) {
      // H5Z_FILTER_DEFLATE (gzip)
      compressor = {
        id: "zlib",
        level: filter.cd_values?.[0] ?? 4,
      };
    } else if (filter.id === 2) {
      // H5Z_FILTER_SHUFFLE
      zarrFilters.push({
        id: "shuffle",
        elementsize: filter.cd_values?.[0] ?? 4,
      });
    } else if (filter.id === 32001) {
      // Blosc
      const blosccnames = [
        "blosclz",
        "lz4",
        "lz4hc",
        "snappy",
        "zlib",
        "zstd",
      ];
      const cd = filter.cd_values || [];
      compressor = {
        id: "blosc",
        cname: blosccnames[cd[4]] || "lz4",
        clevel: cd[3] || 5,
        shuffle: cd[5] || 0,
        blocksize: cd[2] || 0,
      };
    } else if (filter.id === 32015) {
      // Zstd
      compressor = {
        id: "zstd",
        level: filter.cd_values?.[0] ?? 3,
      };
    } else if (filter.id === 3) {
      // H5Z_FILTER_FLETCHER32 - checksum, not a compressor
      // Ignore for Zarr
    } else if (filter.id === 4) {
      // H5Z_FILTER_SZIP
      // Not well supported in numcodecs/Zarr
    }
  }

  return {
    compressor,
    zarrFilters: zarrFilters.length > 0 ? zarrFilters : null,
  };
}

/**
 * Encode a fill value for JSON serialization.
 */
function encodeFillValue(
  fillValue: unknown,
  dtype: string
): unknown {
  if (fillValue === null || fillValue === undefined) {
    return null;
  }

  if (typeof fillValue === "number") {
    if (Number.isNaN(fillValue)) return "NaN";
    if (fillValue === Infinity) return "Infinity";
    if (fillValue === -Infinity) return "-Infinity";
    // For integer types, ensure we return an integer
    if (dtype.match(/[bBhHiIqQ]/)) {
      return Math.round(fillValue);
    }
    return fillValue;
  }

  if (typeof fillValue === "bigint") {
    return Number(fillValue);
  }

  if (typeof fillValue === "string") {
    return fillValue;
  }

  return 0;
}

/**
 * Encode binary data as base64.
 */
function toBase64(data: Uint8Array): string {
  // Node.js environment
  if (typeof Buffer !== "undefined") {
    return "base64:" + Buffer.from(data).toString("base64");
  }
  // Browser environment
  let binary = "";
  for (let i = 0; i < data.length; i++) {
    binary += String.fromCharCode(data[i]);
  }
  return "base64:" + btoa(binary);
}

/**
 * Translate the content of one HDF5 file into Zarr metadata
 * following the Zarr References Specification v1.
 *
 * Ported from kerchunk's SingleHdf5ToZarr Python class.
 *
 * @see https://github.com/fsspec/kerchunk/blob/main/kerchunk/hdf.py
 */
export class SingleHdf5ToZarr {
  private h5File: InstanceType<typeof import("h5wasm").File>;
  private url: string | null;
  private inlineThreshold: number;
  private refs: Record<string, string | [string | null, number, number]>;

  /**
   * @param h5File - An opened h5wasm File object.
   * @param options - Configuration options.
   */
  constructor(
    h5File: InstanceType<typeof import("h5wasm").File>,
    options: SingleHdf5ToZarrOptions = {}
  ) {
    this.h5File = h5File;
    this.url = options.url ?? null;
    this.inlineThreshold = options.inlineThreshold ?? 500;
    this.refs = {};
  }

  /**
   * Translate the HDF5 file content into a Zarr reference spec.
   *
   * No large data is copied; only metadata and small inline data
   * are included in the output.
   */
  translate(): ReferenceSpec {
    this.refs = {};

    // Root group
    this.refs[".zgroup"] = JSON.stringify({ zarr_format: 2 } as ZarrGroupMeta);
    this.transferAttrs(this.h5File, "");

    // Walk all paths in the file
    const paths = this.h5File.paths();
    for (const path of paths) {
      const obj = this.h5File.get(path);
      if (!obj) continue;

      if ("keys" in obj && typeof (obj as Group).keys === "function") {
        this.processGroup(path, obj as Group);
      } else if ("metadata" in obj && "shape" in (obj as Dataset)) {
        this.processDataset(path, obj as Dataset);
      }
    }

    return {
      version: 1,
      refs: this.refs,
    };
  }

  /**
   * Process an HDF5 group into Zarr group metadata.
   */
  private processGroup(path: string, group: Group): void {
    this.refs[`${path}/.zgroup`] = JSON.stringify({
      zarr_format: 2,
    } as ZarrGroupMeta);
    this.transferAttrs(group, path);
  }

  /**
   * Process an HDF5 dataset into Zarr array metadata and chunk references.
   */
  private processDataset(path: string, dataset: Dataset): void {
    const metadata = dataset.metadata;
    const shape = metadata.shape;
    if (!shape) return; // null/scalar dataset with no shape

    const dtype = metadataToZarrDtype(metadata);
    const chunks = metadata.chunks || shape;
    const filters = dataset.filters;

    const { compressor, zarrFilters } = decodeFilters(filters);

    // Determine fill value
    let fillValue: unknown = null;
    if (dtype.startsWith("|S") || dtype.startsWith("|O")) {
      fillValue = "";
    } else if (dtype.includes("f") || dtype.includes("d") || dtype.includes("e")) {
      fillValue = 0.0;
    } else {
      fillValue = 0;
    }

    // Check for _FillValue attribute
    try {
      const fv = dataset.get_attribute("_FillValue", true);
      if (fv !== null && fv !== undefined) {
        fillValue = fv;
      }
    } catch {
      // No _FillValue attribute
    }

    fillValue = encodeFillValue(fillValue, dtype);

    const zarrMeta: ZarrArrayMeta = {
      zarr_format: 2,
      shape: [...shape],
      chunks: [...chunks],
      dtype,
      compressor,
      fill_value: fillValue,
      order: "C",
      filters: zarrFilters,
    };

    this.refs[`${path}/.zarray`] = JSON.stringify(zarrMeta);

    // Transfer attributes
    this.transferAttrs(dataset, path);

    // Add _ARRAY_DIMENSIONS attribute
    const dims = this.getArrayDims(dataset, path);
    const existingAttrsKey = `${path}/.zattrs`;
    const existingAttrs = this.refs[existingAttrsKey]
      ? JSON.parse(this.refs[existingAttrsKey] as string)
      : {};
    existingAttrs["_ARRAY_DIMENSIONS"] = dims;
    this.refs[existingAttrsKey] = JSON.stringify(existingAttrs);

    // Process data chunks
    this.processChunks(path, dataset, metadata, chunks);
  }

  /**
   * Get dimension names for a dataset.
   */
  private getArrayDims(dataset: Dataset, path: string): string[] {
    const shape = dataset.metadata.shape;
    if (!shape || shape.length === 0) return [];

    const dims: string[] = [];
    const rank = shape.length;

    // Try to get dimension labels
    try {
      const labels = dataset.get_dimension_labels();
      if (labels && labels.length === rank) {
        for (let i = 0; i < rank; i++) {
          if (labels[i]) {
            dims.push(labels[i]!);
          } else {
            dims.push(`phony_dim_${i}`);
          }
        }
        return dims;
      }
    } catch {
      // dimension labels not available
    }

    // Try to get attached dimension scales
    try {
      for (let i = 0; i < rank; i++) {
        const scales = dataset.get_attached_scales(i);
        if (scales && scales.length === 1) {
          // Remove leading slash from the scale path
          dims.push(scales[0].replace(/^\//, ""));
        } else {
          dims.push(`phony_dim_${i}`);
        }
      }
      return dims;
    } catch {
      // Dimension scales not available
    }

    // Fall back to phony dimension names
    for (let i = 0; i < rank; i++) {
      dims.push(`phony_dim_${i}`);
    }
    return dims;
  }

  /**
   * Process the data chunks for a dataset.
   * For small data: inline as base64.
   * For larger data: try to determine file offset and size, else inline.
   */
  private processChunks(
    path: string,
    dataset: Dataset,
    metadata: Metadata,
    chunks: number[]
  ): void {
    const shape = metadata.shape;
    if (!shape || shape.length === 0) return;

    // For vlen strings, always read and inline the data
    if (metadata.vlen || metadata.type === 3 /* H5T_STRING */) {
      this.inlineVlenData(path, dataset, shape, chunks);
      return;
    }

    // Calculate total size
    const totalElements = shape.reduce((a, b) => a * b, 1);
    const totalBytes = totalElements * metadata.size;

    // For small datasets or when inline threshold allows, inline the data
    if (totalBytes <= this.inlineThreshold || totalBytes === 0) {
      this.inlineDataset(path, dataset, shape, chunks, metadata);
      return;
    }

    // For chunked datasets, calculate chunk indices and create references.
    // Since h5wasm doesn't expose chunk offset/size API directly,
    // we read and inline each chunk.
    this.inlineChunkedDataset(path, dataset, shape, chunks, metadata);
  }

  /**
   * Inline vlen string data.
   */
  private inlineVlenData(
    path: string,
    dataset: Dataset,
    shape: number[],
    chunks: number[]
  ): void {
    try {
      const data = dataset.json_value;
      if (data === null || data === undefined) return;

      // For vlen strings, encode as JSON array and store inline
      const key = `${path}/${chunks.map(() => "0").join(".")}`;
      const jsonStr = JSON.stringify(data);
      this.refs[key] = jsonStr;

      // Update .zarray to use object dtype and appropriate filters
      const zarrMeta = JSON.parse(this.refs[`${path}/.zarray`] as string);
      zarrMeta.dtype = "|O";
      zarrMeta.filters = [{ id: "vlen-utf8" }];
      zarrMeta.compressor = null;
      this.refs[`${path}/.zarray`] = JSON.stringify(zarrMeta);
    } catch {
      // Cannot read vlen data
    }
  }

  /**
   * Inline a small dataset entirely.
   */
  private inlineDataset(
    path: string,
    dataset: Dataset,
    shape: number[],
    chunks: number[],
    metadata: Metadata
  ): void {
    try {
      const value = dataset.value;
      if (value === null || value === undefined) return;

      // Convert typed array to bytes
      if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
        const bytes = new Uint8Array(
          (value as unknown as { buffer: ArrayBuffer }).buffer,
          (value as unknown as { byteOffset: number }).byteOffset,
          (value as unknown as { byteLength: number }).byteLength
        );
        const key = `${path}/${chunks.map(() => "0").join(".")}`;
        this.refs[key] = toBase64(bytes);
      } else if (Array.isArray(value)) {
        // String arrays
        const key = `${path}/${chunks.map(() => "0").join(".")}`;
        this.refs[key] = JSON.stringify(value);
      }
    } catch {
      // Cannot read dataset
    }
  }

  /**
   * Read and inline a chunked dataset, chunk by chunk.
   */
  private inlineChunkedDataset(
    path: string,
    dataset: Dataset,
    shape: number[],
    chunks: number[],
    metadata: Metadata
  ): void {
    // Calculate the number of chunks along each dimension
    const nChunks = shape.map((s, i) => Math.ceil(s / chunks[i]));

    // Total number of chunks
    const totalChunks = nChunks.reduce((a, b) => a * b, 1);

    // Iterate through all chunk indices
    for (let chunkLinear = 0; chunkLinear < totalChunks; chunkLinear++) {
      // Convert linear index to multi-dimensional chunk index
      const chunkIndex: number[] = [];
      let remaining = chunkLinear;
      for (let d = nChunks.length - 1; d >= 0; d--) {
        chunkIndex.unshift(remaining % nChunks[d]);
        remaining = Math.floor(remaining / nChunks[d]);
      }

      // Calculate the slice ranges for this chunk
      const ranges: [number, number][] = chunkIndex.map((ci, d) => {
        const start = ci * chunks[d];
        const end = Math.min(start + chunks[d], shape[d]);
        return [start, end];
      });

      try {
        // Read the chunk data using h5wasm slice
        const sliceRanges = ranges.map(([start, end]) => [start, end] as [number, number]);
        const data = dataset.slice(sliceRanges);

        if (data === null || data === undefined) continue;

        const key = `${path}/${chunkIndex.join(".")}`;

        if (ArrayBuffer.isView(data) && !(data instanceof DataView)) {
          const actualSize = ranges.reduce((acc, [s, e]) => acc * (e - s), 1);
          const expectedSize = chunks.reduce((a, b) => a * b, 1);

          let bytes: Uint8Array;
          if (actualSize < expectedSize) {
            // Edge chunk: need to pad to full chunk size
            const fullBuffer = new ArrayBuffer(expectedSize * metadata.size);
            const fullView = new Uint8Array(fullBuffer);
            // Zero-fill is default for ArrayBuffer
            const sourceBytes = new Uint8Array(
              (data as unknown as { buffer: ArrayBuffer }).buffer,
              (data as unknown as { byteOffset: number }).byteOffset,
              (data as unknown as { byteLength: number }).byteLength
            );
            fullView.set(sourceBytes);
            bytes = fullView;
          } else {
            bytes = new Uint8Array(
              (data as unknown as { buffer: ArrayBuffer }).buffer,
              (data as unknown as { byteOffset: number }).byteOffset,
              (data as unknown as { byteLength: number }).byteLength
            );
          }
          this.refs[key] = toBase64(bytes);
        }
      } catch {
        // Skip chunks that can't be read
      }
    }
  }

  /**
   * Transfer attributes from an HDF5 object to the Zarr .zattrs.
   */
  private transferAttrs(
    h5obj: Group | Dataset | InstanceType<typeof import("h5wasm").File>,
    path: string
  ): void {
    const attrs: Record<string, unknown> = {};
    const attrNames = Object.keys(h5obj.attrs);

    for (const name of attrNames) {
      if (HIDDEN_ATTRS.has(name)) continue;
      if (name === "_FillValue") continue;

      try {
        const attr = h5obj.attrs[name];
        let value = attr.json_value;

        if (value === null || value === undefined) {
          value = "";
        }

        // Check if value is "DIMENSION_SCALE" and skip
        if (value === "DIMENSION_SCALE") continue;

        attrs[name] = value;
      } catch {
        // Skip attributes that can't be read
      }
    }

    if (Object.keys(attrs).length > 0) {
      const key = path ? `${path}/.zattrs` : ".zattrs";
      this.refs[key] = JSON.stringify(attrs);
    }
  }
}

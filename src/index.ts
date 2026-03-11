// @ts-nocheck - jsfive is plain JS without type declarations
import { File as Hdf5File, Dataset as Hdf5Dataset, Group as Hdf5Group } from "../jsfive/esm/high-level.js";
import { DataObjects } from "../jsfive/esm/dataobjects.js";
import { BTreeV1RawDataChunks } from "../jsfive/esm/btree.js";
import { struct } from "../jsfive/esm/core.js";

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
 * Zarr v2 consolidated metadata (.zmetadata).
 */
export interface ZarrConsolidatedMetadata {
  zarr_consolidated_format: 1;
  metadata: Record<string, Record<string, unknown>>;
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
 * Convert jsfive dtype string to Zarr-compatible dtype string.
 * jsfive produces numpy-style strings like "<f4", "<i4", etc.
 * For single-byte types, normalize to use "|" prefix.
 */
function jsfiveDtypeToZarr(dtype: string | string[]): string {
  if (Array.isArray(dtype)) {
    const dtypeClass = dtype[0];
    if (dtypeClass === "VLEN_STRING") return "|O";
    if (dtypeClass === "VLEN_SEQUENCE") return "|O";
    if (dtypeClass === "REFERENCE") return "|O";
    return "|O";
  }

  // Fixed-length string: "S123" -> "|S123"
  if (/^S\d+/.test(dtype)) {
    return "|" + dtype;
  }

  // Parse the dtype parts
  const match = dtype.match(/^([<>|])([iufS])(\d+)$/);
  if (!match) return dtype;

  const [, endian, typeChar, sizeStr] = match;
  const size = parseInt(sizeStr, 10);

  // Single-byte integers use "|" prefix
  if ((typeChar === "i" || typeChar === "u") && size === 1) {
    return `|${typeChar}${size}`;
  }

  return dtype;
}

/**
 * Convert jsfive filter pipeline to Zarr compressor/filters metadata.
 */
function decodeFilterPipeline(
  filterPipeline: Map<string, unknown>[] | null
): { compressor: Record<string, unknown> | null; zarrFilters: Record<string, unknown>[] | null } {
  const zarrFilters: Record<string, unknown>[] = [];
  let compressor: Record<string, unknown> | null = null;

  if (!filterPipeline) {
    return { compressor, zarrFilters: null };
  }

  for (const filterInfo of filterPipeline) {
    const filterId = filterInfo.get("filter_id") as number;
    const clientData = filterInfo.get("client_data") as number[] | undefined;

    if (filterId === 1) {
      // H5Z_FILTER_DEFLATE (gzip)
      compressor = {
        id: "zlib",
        level: clientData?.[0] ?? 4,
      };
    } else if (filterId === 2) {
      // H5Z_FILTER_SHUFFLE
      zarrFilters.push({
        id: "shuffle",
        elementsize: clientData?.[0] ?? 4,
      });
    } else if (filterId === 32001) {
      // Blosc
      const blosccnames = [
        "blosclz",
        "lz4",
        "lz4hc",
        "snappy",
        "zlib",
        "zstd",
      ];
      const cd = clientData || [];
      compressor = {
        id: "blosc",
        cname: blosccnames[cd[4]] || "lz4",
        clevel: cd[3] || 5,
        shuffle: cd[5] || 0,
        blocksize: cd[2] || 0,
      };
    } else if (filterId === 32015) {
      // Zstd
      compressor = {
        id: "zstd",
        level: clientData?.[0] ?? 3,
      };
    } else if (filterId === 3) {
      // H5Z_FILTER_FLETCHER32 - checksum, not a compressor
    } else if (filterId === 4) {
      // H5Z_FILTER_SZIP
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
    if (dtype.match(/[iu]\d/)) {
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
 * Inline binary data, matching kerchunk's encoding:
 * - If all bytes are < 128 (ASCII-safe), store as a raw character string
 * - Otherwise, encode as "base64:" + base64
 */
function inlineBytes(data: Uint8Array): string {
  let asciiSafe = true;
  for (let i = 0; i < data.length; i++) {
    if (data[i] >= 128) {
      asciiSafe = false;
      break;
    }
  }

  if (asciiSafe) {
    let result = "";
    for (let i = 0; i < data.length; i++) {
      result += String.fromCharCode(data[i]);
    }
    return result;
  }

  if (typeof Buffer !== "undefined") {
    return "base64:" + Buffer.from(data).toString("base64");
  }
  let binary = "";
  for (let i = 0; i < data.length; i++) {
    binary += String.fromCharCode(data[i]);
  }
  return "base64:" + btoa(binary);
}

/**
 * Encode an array of strings in the json2 codec format used by kerchunk.
 */
function encodeJson2(strings: string[], shape: number[]): string {
  const parts: unknown[] = [...strings, "|O", shape];
  return JSON.stringify(parts);
}

/**
 * The json2 filter metadata used by kerchunk for vlen string data.
 */
const JSON2_FILTER = {
  allow_nan: true,
  check_circular: true,
  encoding: "utf-8",
  ensure_ascii: true,
  id: "json2",
  indent: null,
  separators: [",", ":"],
  skipkeys: false,
  sort_keys: true,
  strict: true,
};

/**
 * JSON.stringify with keys sorted alphabetically at all levels.
 */
function jsonStringifySorted(obj: unknown): string {
  return JSON.stringify(obj, (_key, value) => {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const sorted: Record<string, unknown> = {};
      for (const k of Object.keys(value).sort()) {
        sorted[k] = (value as Record<string, unknown>)[k];
      }
      return sorted;
    }
    return value;
  });
}

/**
 * Serialize a ZarrArrayMeta to JSON, matching kerchunk's output.
 */
function serializeZarrayMeta(meta: ZarrArrayMeta): string {
  let json = jsonStringifySorted(meta);
  if (
    /[<>]f\d/.test(meta.dtype) &&
    typeof meta.fill_value === "number" &&
    Number.isFinite(meta.fill_value) &&
    meta.fill_value === Math.round(meta.fill_value)
  ) {
    const intStr = String(Math.round(meta.fill_value));
    json = json.replace(`"fill_value":${intStr}`, `"fill_value":${intStr}.0`);
  }
  return json;
}

const UNDEFINED_ADDRESS = 0xffffffffffff; // 2^48-1 (jsfive uses Number from 8-byte LE max)

/**
 * Get storage info for a dataset's DataObjects, returning chunk
 * byte offsets and sizes for file references.
 *
 * Based on jsfive's DataObjects._get_chunk_params and _get_data_message_properties.
 */
function getStorageInfo(dataobjects: any, fh: ArrayBuffer): {
  layoutClass: number;
  contiguousOffset?: number;
  contiguousSize?: number;
  chunkAddress?: number;
  chunkDims?: number;
  chunkShape?: number[];
} {
  const DATA_STORAGE_MSG_TYPE = 0x0008;
  const msg = dataobjects.find_msg_type(DATA_STORAGE_MSG_TYPE)[0];
  if (!msg) return { layoutClass: -1 };

  const msgOffset = msg.get("offset_to_message");
  const [version, arg1, arg2] = struct.unpack_from("<BBB", fh, msgOffset);

  let dims: number, layoutClass: number, propertyOffset: number;

  if (version === 1 || version === 2) {
    dims = arg1;
    layoutClass = arg2;
    propertyOffset = msgOffset + struct.calcsize("<BBB") + struct.calcsize("<BI");
  } else if (version === 3 || version === 4) {
    layoutClass = arg1;
    propertyOffset = msgOffset + struct.calcsize("<BB");
    dims = 0; // will be read below for chunked
  } else {
    return { layoutClass: -1 };
  }

  if (layoutClass === 1) {
    // Contiguous storage
    const dataOffset = struct.unpack_from("<Q", fh, propertyOffset)[0];
    if (dataOffset === UNDEFINED_ADDRESS || dataOffset >= Number.MAX_SAFE_INTEGER * 0.99) {
      return { layoutClass: 1 };
    }
    // For version 3, size follows the address
    let size: number;
    if (version === 3 || version === 4) {
      size = struct.unpack_from("<Q", fh, propertyOffset + 8)[0];
    } else {
      // For v1/v2, compute from shape * dtype size
      const shape = dataobjects.shape;
      const dtype = dataobjects.dtype;
      let itemSize = 0;
      if (typeof dtype === "string") {
        const m = dtype.match(/\d+$/);
        itemSize = m ? parseInt(m[0], 10) : 1;
      } else {
        itemSize = 8; // reference size
      }
      size = shape.reduce((a: number, b: number) => a * b, 1) * itemSize;
    }
    return { layoutClass: 1, contiguousOffset: dataOffset, contiguousSize: size };
  } else if (layoutClass === 2) {
    // Chunked storage
    let address: number, dataOffsetForChunks: number;
    if (version === 1 || version === 2) {
      address = struct.unpack_from("<Q", fh, propertyOffset)[0];
      dataOffsetForChunks = propertyOffset + struct.calcsize("<Q");
    } else {
      // version 3
      const result = struct.unpack_from("<BQ", fh, propertyOffset);
      dims = result[0];
      address = result[1];
      dataOffsetForChunks = propertyOffset + struct.calcsize("<BQ");
    }

    if (address === UNDEFINED_ADDRESS || address >= Number.MAX_SAFE_INTEGER * 0.99) {
      return { layoutClass: 2 };
    }

    const fmt = "<" + (dims - 1).toFixed() + "I";
    const chunkShape = struct.unpack_from(fmt, fh, dataOffsetForChunks);

    return {
      layoutClass: 2,
      chunkAddress: address,
      chunkDims: dims,
      chunkShape: Array.isArray(chunkShape) ? chunkShape : [chunkShape],
    };
  }

  return { layoutClass: layoutClass };
}

/**
 * Get chunk locations from a B-tree for chunked datasets.
 * Returns an array of { chunkIndex, offset, size } objects.
 */
function getChunkLocations(
  fh: ArrayBuffer,
  btreeAddress: number,
  chunkDims: number,
  dataShape: number[],
  chunkShape: number[]
): Array<{ chunkIndex: number[]; offset: number; size: number }> {
  const btree = new BTreeV1RawDataChunks(fh, btreeAddress, chunkDims);
  const results: Array<{ chunkIndex: number[]; offset: number; size: number }> = [];

  const leafNodes = btree.all_nodes.get(0);
  if (!leafNodes) return results;

  for (const node of leafNodes) {
    const nodeKeys = node.get("keys");
    const nodeAddresses = node.get("addresses");
    const nkeys = nodeKeys.length;

    for (let ik = 0; ik < nkeys; ik++) {
      const nodeKey = nodeKeys[ik];
      const addr = nodeAddresses[ik];
      const chunkOffset = nodeKey.get("chunk_offset");
      const chunkSize = nodeKey.get("chunk_size");

      // Convert chunk_offset (element offsets) to chunk indices
      const chunkIndex = chunkOffset.slice(0, -1).map((co: number, d: number) =>
        Math.floor(co / chunkShape[d])
      );

      // For 1D data, chunk_offset has 2 entries (dim + trailing 0)
      // chunkIndex should match the number of data dimensions
      while (chunkIndex.length < dataShape.length) {
        chunkIndex.push(0);
      }
      while (chunkIndex.length > dataShape.length) {
        chunkIndex.pop();
      }

      results.push({
        chunkIndex,
        offset: addr,
        size: chunkSize,
      });
    }
  }

  return results;
}

const ATTRIBUTE_MSG_TYPE = 0x000C;
const DATATYPE_ENUMERATED = 8;

/**
 * Get the raw HDF5 datatype class for each attribute in a DataObjects instance.
 * Returns a Map from attribute name to datatype class number.
 * Used to detect enum (boolean) attributes.
 */
function getAttrDatatypeClasses(dataobjects: any, fh: ArrayBuffer): Map<string, number> {
  const result = new Map<string, number>();
  const attrMsgs = dataobjects.find_msg_type(ATTRIBUTE_MSG_TYPE);

  for (const msg of attrMsgs) {
    let offset = msg.get("offset_to_message");
    const version = struct.unpack_from("<B", fh, offset)[0];

    let nameSize: number;
    let paddingMultiple: number;

    if (version === 1) {
      // ATTR_MSG_HEADER_V1: version(B) + reserved(B) + name_size(H) + datatype_size(H) + dataspace_size(H) = 8 bytes
      nameSize = struct.unpack_from("<H", fh, offset + 2)[0];
      paddingMultiple = 8;
      offset += 8; // skip header
    } else if (version === 3) {
      // ATTR_MSG_HEADER_V3: version(B) + flags(B) + name_size(H) + datatype_size(H) + dataspace_size(H) + encoding(B) = 9 bytes
      nameSize = struct.unpack_from("<H", fh, offset + 2)[0];
      paddingMultiple = 1;
      offset += 9; // skip header
    } else {
      continue;
    }

    // Read attribute name
    const nameBytes = struct.unpack_from("<" + nameSize.toFixed() + "s", fh, offset)[0];
    const name = nameBytes.replace(/\x00$/, "");

    // Skip past padded name
    const paddedNameSize = paddingMultiple > 1
      ? Math.ceil(nameSize / paddingMultiple) * paddingMultiple
      : nameSize;
    offset += paddedNameSize;

    // Read raw datatype class from first byte of datatype message
    const classAndVersion = struct.unpack_from("<B", fh, offset)[0];
    const datatypeClass = classAndVersion & 0x0f;

    result.set(name, datatypeClass);
  }

  return result;
}

/**
 * Translate the content of one HDF5 file into Zarr metadata
 * following the Zarr References Specification v1.
 *
 * Ported from kerchunk's SingleHdf5ToZarr Python class.
 * Uses jsfive for HDF5 parsing from an ArrayBuffer.
 */
export class SingleHdf5ToZarr {
  private h5File: any;
  private fh: ArrayBuffer;
  private url: string | null;
  private inlineThreshold: number;
  private refs: Record<string, string | [string | null, number, number]>;

  /**
   * @param buffer - The full HDF5 file as an ArrayBuffer.
   * @param options - Configuration options.
   */
  constructor(
    buffer: ArrayBuffer,
    options: SingleHdf5ToZarrOptions = {}
  ) {
    this.fh = buffer;
    this.h5File = new Hdf5File(buffer);
    this.url = options.url ?? null;
    this.inlineThreshold = options.inlineThreshold ?? 300;
    this.refs = {};
  }

  /**
   * Translate the HDF5 file content into a Zarr reference spec.
   */
  translate(): ReferenceSpec {
    this.refs = {};

    // Root group
    this.refs[".zgroup"] = jsonStringifySorted({ zarr_format: 2 } as ZarrGroupMeta);
    this.transferAttrs(this.h5File, "");

    // Walk all items recursively
    this.walkGroup(this.h5File, "");

    return {
      version: 1,
      refs: this.refs,
    };
  }

  /**
   * Recursively walk an HDF5 group, processing all children.
   */
  private walkGroup(group: any, parentPath: string): void {
    const keys = group.keys;
    for (const key of keys) {
      const childPath = parentPath ? `${parentPath}/${key}` : key;
      let child: any;
      try {
        child = group.get(key);
      } catch {
        continue;
      }
      if (!child) continue;

      if (child instanceof Hdf5Dataset) {
        this.processDataset(childPath, child);
      } else if (child instanceof Hdf5Group) {
        this.processGroup(childPath, child);
        this.walkGroup(child, childPath);
      }
    }
  }

  /**
   * Process an HDF5 group into Zarr group metadata.
   */
  private processGroup(path: string, group: any): void {
    this.refs[`${path}/.zgroup`] = jsonStringifySorted({
      zarr_format: 2,
    } as ZarrGroupMeta);
    this.transferAttrs(group, path);
  }

  /**
   * Process an HDF5 dataset into Zarr array metadata and chunk references.
   */
  private processDataset(path: string, dataset: any): void {
    const dataobjects = dataset._dataobjects;
    const shape = dataobjects.shape;
    if (!shape || shape.length === 0) {
      // Scalar dataset
      this.processScalarDataset(path, dataset, dataobjects);
      return;
    }

    const rawDtype = dataobjects.dtype;
    const isVlen = Array.isArray(rawDtype) &&
      (rawDtype[0] === "VLEN_STRING" || rawDtype[0] === "VLEN_SEQUENCE");
    const isFixedString = typeof rawDtype === "string" && /^S\d+/.test(rawDtype);

    const dtype = jsfiveDtypeToZarr(rawDtype);
    const chunks = dataobjects.chunks || [...shape];
    const filterPipeline = dataobjects.filter_pipeline;
    const { compressor, zarrFilters } = decodeFilterPipeline(filterPipeline);

    // Determine fill value
    let fillValue: unknown = null;
    if (dtype.startsWith("|S")) {
      fillValue = "";
    } else if (dtype === "|O") {
      fillValue = null;
    } else if (/[<>|]f\d/.test(dtype)) {
      fillValue = 0.0;
    } else {
      fillValue = 0;
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

    this.refs[`${path}/.zarray`] = serializeZarrayMeta(zarrMeta);

    // Transfer attributes
    this.transferAttrs(dataset, path);

    // Add _ARRAY_DIMENSIONS
    const dims = this.getArrayDims(dataset, path, shape);
    const existingAttrsKey = `${path}/.zattrs`;
    const existingAttrs = this.refs[existingAttrsKey]
      ? JSON.parse(this.refs[existingAttrsKey] as string)
      : {};
    existingAttrs["_ARRAY_DIMENSIONS"] = dims;
    this.refs[existingAttrsKey] = jsonStringifySorted(existingAttrs);

    // Process data: vlen strings get inlined, others get file references
    if (isVlen) {
      this.inlineVlenData(path, dataset, shape, chunks);
    } else if (isFixedString) {
      this.inlineFixedStringData(path, dataset, shape, chunks, rawDtype);
    } else {
      this.processChunkReferences(path, dataobjects, shape, chunks, dtype);
    }
  }

  /**
   * Process a scalar dataset (shape=[]).
   */
  private processScalarDataset(path: string, dataset: any, dataobjects: any): void {
    const rawDtype = dataobjects.dtype;
    const isVlen = Array.isArray(rawDtype) &&
      (rawDtype[0] === "VLEN_STRING" || rawDtype[0] === "VLEN_SEQUENCE");
    const dtype = jsfiveDtypeToZarr(rawDtype);

    let fillValue: unknown = null;
    if (dtype === "|O") fillValue = null;
    else if (/[<>|]f\d/.test(dtype)) fillValue = 0.0;
    else fillValue = 0;

    fillValue = encodeFillValue(fillValue, dtype);

    const zarrMeta: ZarrArrayMeta = {
      zarr_format: 2,
      shape: [],
      chunks: [],
      dtype,
      compressor: null,
      fill_value: fillValue,
      order: "C",
      filters: null,
    };

    if (isVlen) {
      zarrMeta.dtype = "|O";
      zarrMeta.fill_value = null;
      zarrMeta.filters = [JSON2_FILTER];
    }

    this.refs[`${path}/.zarray`] = serializeZarrayMeta(zarrMeta);
    this.transferAttrs(dataset, path);

    // Add _ARRAY_DIMENSIONS
    const existingAttrsKey = `${path}/.zattrs`;
    const existingAttrs = this.refs[existingAttrsKey]
      ? JSON.parse(this.refs[existingAttrsKey] as string)
      : {};
    existingAttrs["_ARRAY_DIMENSIONS"] = [];
    this.refs[existingAttrsKey] = jsonStringifySorted(existingAttrs);

    // Inline data
    if (isVlen) {
      try {
        const data = dataobjects.get_data();
        const strings: string[] = Array.isArray(data) ? data.map(String) : [String(data)];
        this.refs[`${path}/0`] = encodeJson2(strings, []);
      } catch {
        // Cannot read scalar vlen data
      }
    } else {
      try {
        const data = dataobjects.get_data();
        if (data !== null && data !== undefined) {
          const strings: string[] = Array.isArray(data) ? data.map(String) : [String(data)];
          this.refs[`${path}/0`] = inlineBytes(
            new Uint8Array(new Float64Array(strings.map(Number)).buffer)
          );
        }
      } catch {
        // Cannot read scalar data
      }
    }
  }

  /**
   * Inline vlen string data using json2 codec encoding.
   */
  private inlineVlenData(
    path: string,
    dataset: any,
    shape: number[],
    chunks: number[]
  ): void {
    try {
      const data = dataset._dataobjects.get_data();
      if (data === null || data === undefined) return;

      const strings: string[] = Array.isArray(data) ? data.map(String) : [String(data)];

      const key = `${path}/${chunks.map(() => "0").join(".")}`;
      this.refs[key] = encodeJson2(strings, shape);

      // Update .zarray for vlen string codec
      const zarrMeta = JSON.parse(this.refs[`${path}/.zarray`] as string);
      zarrMeta.dtype = "|O";
      zarrMeta.fill_value = null;
      zarrMeta.filters = [JSON2_FILTER];
      zarrMeta.compressor = null;
      this.refs[`${path}/.zarray`] = serializeZarrayMeta(zarrMeta);
    } catch {
      // Cannot read vlen data
    }
  }

  /**
   * Inline fixed-length string data.
   */
  private inlineFixedStringData(
    path: string,
    dataset: any,
    shape: number[],
    chunks: number[],
    rawDtype: string
  ): void {
    try {
      const data = dataset._dataobjects.get_data();
      if (data === null || data === undefined) return;

      const strings: string[] = Array.isArray(data) ? data.map(String) : [String(data)];
      const key = `${path}/${chunks.map(() => "0").join(".")}`;

      // For fixed-length strings, inline the raw bytes
      const sizeMatch = rawDtype.match(/S(\d+)/);
      const strSize = sizeMatch ? parseInt(sizeMatch[1], 10) : 1;
      const totalSize = strings.length * strSize;
      const buf = new Uint8Array(totalSize);
      const encoder = new TextEncoder();
      for (let i = 0; i < strings.length; i++) {
        const encoded = encoder.encode(strings[i]);
        buf.set(encoded.subarray(0, strSize), i * strSize);
      }
      this.refs[key] = inlineBytes(buf);
    } catch {
      // Cannot read fixed string data
    }
  }

  /**
   * Process chunk references for non-string datasets.
   * Uses file byte offset references for chunked and contiguous data.
   */
  private processChunkReferences(
    path: string,
    dataobjects: any,
    shape: number[],
    chunks: number[],
    dtype: string
  ): void {
    const storageInfo = getStorageInfo(dataobjects, this.fh);

    if (storageInfo.layoutClass === 1 && storageInfo.contiguousOffset !== undefined) {
      // Contiguous storage - single chunk reference
      const key = `${path}/${shape.map(() => "0").join(".")}`;
      const offset = storageInfo.contiguousOffset!;
      const size = storageInfo.contiguousSize!;

      if (size <= this.inlineThreshold) {
        const bytes = new Uint8Array(this.fh, offset, size);
        this.refs[key] = inlineBytes(new Uint8Array(bytes));
      } else {
        this.refs[key] = [this.url, offset, size];
      }
    } else if (storageInfo.layoutClass === 2 && storageInfo.chunkAddress !== undefined) {
      // Chunked storage - get chunk locations from B-tree
      const filterPipeline = dataobjects.filter_pipeline;
      const hasFilters = filterPipeline && filterPipeline.length > 0;

      // Item size from dtype
      let itemSize = 1;
      const sizeMatch = dtype.match(/\d+$/);
      if (sizeMatch) itemSize = parseInt(sizeMatch[0], 10);

      const chunkElements = chunks.reduce((a, b) => a * b, 1);
      const uncompressedChunkBytes = chunkElements * itemSize;

      // Check for fletcher32 (filter_id=3) to adjust sizes
      let hasF32 = false;
      if (filterPipeline) {
        for (const f of filterPipeline) {
          if (f.get("filter_id") === 3) {
            hasF32 = true;
            break;
          }
        }
      }

      const chunkLocations = getChunkLocations(
        this.fh,
        storageInfo.chunkAddress!,
        storageInfo.chunkDims!,
        shape,
        storageInfo.chunkShape!
      );

      for (const loc of chunkLocations) {
        const key = `${path}/${loc.chunkIndex.join(".")}`;
        let size = hasFilters ? loc.size : uncompressedChunkBytes;
        if (hasF32) {
          size -= 4; // Strip fletcher32 checksum
        }

        if (size <= this.inlineThreshold) {
          const bytes = new Uint8Array(this.fh.slice(loc.offset, loc.offset + size));
          this.refs[key] = inlineBytes(bytes);
        } else {
          this.refs[key] = [this.url, loc.offset, size];
        }
      }
    } else if (storageInfo.layoutClass === 1) {
      // Contiguous but no data written - empty dataset, no refs needed
    }
  }

  /**
   * Get dimension names for a dataset.
   */
  private getArrayDims(dataset: any, path: string, shape: number[]): string[] {
    if (!shape || shape.length === 0) return [];

    const rank = shape.length;
    const dims: string[] = [];

    // Check for DIMENSION_LIST attribute (links to dimension scales)
    try {
      const attrs = dataset._dataobjects.get_attributes();
      // jsfive doesn't have get_dimension_labels / get_attached_scales
      // Just fallback to phony dims for now
    } catch {
      // ignore
    }

    for (let i = 0; i < rank; i++) {
      dims.push(`phony_dim_${i}`);
    }
    return dims;
  }

  /**
   * Transfer attributes from an HDF5 object to the Zarr .zattrs.
   */
  private transferAttrs(h5obj: any, path: string): void {
    const attrs: Record<string, unknown> = {};
    let rawAttrs: Record<string, unknown>;
    try {
      rawAttrs = h5obj.attrs;
      if (!rawAttrs || typeof rawAttrs !== "object") return;
    } catch {
      return;
    }

    // Detect which attributes are HDF5 enum type (for boolean conversion)
    let enumAttrs = new Map<string, number>();
    try {
      const dataobjects = h5obj._dataobjects;
      if (dataobjects) {
        enumAttrs = getAttrDatatypeClasses(dataobjects, this.fh);
      }
    } catch {
      // ignore
    }

    for (const [name, value] of Object.entries(rawAttrs)) {
      if (HIDDEN_ATTRS.has(name)) continue;
      if (name === "_FillValue") continue;

      let v: unknown = value;

      // Handle bytes/strings
      if (v === null || v === undefined) {
        v = "";
      }
      if (v === "DIMENSION_SCALE") continue;

      // Convert HDF5 enum (boolean) attributes: 0 → false, 1 → true
      if (enumAttrs.get(name) === DATATYPE_ENUMERATED) {
        if (v === 0) v = false;
        else if (v === 1) v = true;
      }

      // Convert typed arrays to plain arrays
      if (ArrayBuffer.isView(v) && !(v instanceof DataView)) {
        v = Array.from(v as unknown as Iterable<number>);
      }

      // Convert single-element arrays to scalars
      if (Array.isArray(v) && v.length === 1) {
        v = v[0];
      }

      attrs[name] = v;
    }

    if (Object.keys(attrs).length > 0) {
      const key = path ? `${path}/.zattrs` : ".zattrs";
      this.refs[key] = jsonStringifySorted(attrs);
    }
  }
}

/**
 * Metadata key suffixes included in Zarr consolidated metadata.
 */
const METADATA_SUFFIXES = [".zgroup", ".zarray", ".zattrs"];

/**
 * Convert a Reference Spec JSON to Zarr v2 consolidated metadata (.zmetadata).
 */
export function refSpecToConsolidatedMetadata(
  refSpec: ReferenceSpec
): ZarrConsolidatedMetadata {
  const metadata: Record<string, Record<string, unknown>> = {};

  for (const [key, value] of Object.entries(refSpec.refs)) {
    if (METADATA_SUFFIXES.some((suffix) => key === suffix || key.endsWith(`/${suffix}`))) {
      if (typeof value === "string") {
        metadata[key] = JSON.parse(value);
      }
    }
  }

  return {
    zarr_consolidated_format: 1,
    metadata,
  };
}

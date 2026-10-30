import type { ReferenceSpec, ReferenceEntry, ZarrArrayMeta } from "./types.js";
import { jsonStringifySorted, decodeInline, inlineBytes, encodeVlenUtf8 } from "./encoding.js";

/**
 * Zarr v3 data type names for numpy-style v2 dtypes (without the byte order character).
 */
const V3_DATA_TYPES: Record<string, string> = {
  b1: "bool",
  i1: "int8",
  u1: "uint8",
  i2: "int16",
  u2: "uint16",
  i4: "int32",
  u4: "uint32",
  i8: "int64",
  u8: "uint64",
  f2: "float16",
  f4: "float32",
  f8: "float64",
};

const BLOSC_SHUFFLE = ["noshuffle", "shuffle", "bitshuffle"];

const V2_CHUNK_KEY = /^\d+(\.\d+)*$/;

type CodecMeta = { name: string; configuration?: Record<string, unknown> };

/**
 * Convert a v2 compressor or filter (numcodecs config) to a v3 bytes-to-bytes codec.
 */
function v2CodecToV3(codec: Record<string, unknown>, itemSize: number): CodecMeta {
  const { id, ...configuration } = codec;
  if (id === "blosc") {
    return {
      name: "blosc",
      configuration: {
        cname: configuration.cname,
        clevel: configuration.clevel,
        shuffle: BLOSC_SHUFFLE[configuration.shuffle as number] ?? "noshuffle",
        typesize: itemSize,
        blocksize: configuration.blocksize ?? 0,
      },
    };
  }
  if (id === "zstd") {
    return { name: "zstd", configuration: { level: configuration.level, checksum: false } };
  }
  // HDF5 deflate is a zlib stream (not gzip), and HDF5 shuffle is a bytes-to-bytes transform,
  // so both are expressed via the numcodecs namespace.
  return { name: `numcodecs.${id}`, configuration };
}

/**
 * Decode the strings held in a single inlined v2 string chunk.
 */
function decodeStringChunk(entry: ReferenceEntry, meta: ZarrArrayMeta): string[] {
  if (typeof entry !== "string") {
    throw new Error("String chunks must be inlined");
  }
  const bytes = decodeInline(entry);
  if (meta.dtype === "|O") {
    // json2 codec: [...items, "|O", shape]
    const parsed = JSON.parse(new TextDecoder().decode(bytes)) as unknown[];
    return parsed.slice(0, -2).map((v) => (v === null || v === undefined ? "" : String(v)));
  }
  // Fixed-length byte strings: null-padded items of N bytes each.
  const itemSize = parseInt(meta.dtype.slice(2), 10);
  const decoder = new TextDecoder();
  const strings: string[] = [];
  for (let i = 0; i + itemSize <= bytes.length; i += itemSize) {
    let end = i + itemSize;
    while (end > i && bytes[end - 1] === 0) end--;
    strings.push(decoder.decode(bytes.subarray(i, end)));
  }
  return strings;
}

/**
 * Convert a Zarr v2 Reference Spec (as produced by kerchunk / SingleHdf5ToZarr)
 * into an equivalent Reference Spec describing a Zarr v3 hierarchy.
 *
 * - `.zgroup` / `.zarray` / `.zattrs` are merged into a single `zarr.json` per node.
 * - `_ARRAY_DIMENSIONS` becomes `dimension_names`.
 * - Chunk keys use the v3 default encoding (`c/0/1` instead of `0.1`).
 * - String arrays (json2-encoded vlen strings and fixed-length byte strings)
 *   are re-encoded as a single `vlen-utf8` chunk with data type `string`.
 * - Arrays that cannot be represented in v3 (e.g. object arrays without json2) are dropped.
 */
export function refSpecV2ToV3(spec: ReferenceSpec): ReferenceSpec {
  const v2Refs = spec.refs;
  const refs: Record<string, ReferenceEntry> = {};

  const parseJson = (key: string): Record<string, unknown> | undefined => {
    const value = v2Refs[key];
    return typeof value === "string" ? JSON.parse(value) : undefined;
  };

  // Group chunk keys by their parent node path.
  const chunkKeysByNode = new Map<string, string[]>();
  for (const key of Object.keys(v2Refs)) {
    const idx = key.lastIndexOf("/");
    const nodePath = idx === -1 ? "" : key.slice(0, idx);
    const name = key.slice(idx + 1);
    if (V2_CHUNK_KEY.test(name)) {
      if (!chunkKeysByNode.has(nodePath)) chunkKeysByNode.set(nodePath, []);
      chunkKeysByNode.get(nodePath)!.push(name);
    }
  }

  for (const key of Object.keys(v2Refs)) {
    const isGroup = key === ".zgroup" || key.endsWith("/.zgroup");
    const isArray = key === ".zarray" || key.endsWith("/.zarray");
    if (!isGroup && !isArray) continue;

    const nodePath = key.slice(0, -".zgroup".length).replace(/\/$/, "");
    const prefix = nodePath ? `${nodePath}/` : "";
    const attributes = parseJson(`${prefix}.zattrs`) ?? {};

    if (isGroup) {
      refs[`${prefix}zarr.json`] = jsonStringifySorted({
        zarr_format: 3,
        node_type: "group",
        attributes,
      });
      continue;
    }

    const meta = parseJson(key) as unknown as ZarrArrayMeta;
    const { _ARRAY_DIMENSIONS: dims, ...arrayAttributes } = attributes;
    const chunkNames = chunkKeysByNode.get(nodePath) ?? [];
    const isJson2 = meta.dtype === "|O" && (meta.filters ?? []).some((f) => f.id === "json2");
    const isFixedString = /^\|S\d+$/.test(meta.dtype);

    if (meta.dtype === "|O" && !isJson2) {
      // Object arrays without a known string encoding cannot be represented.
      continue;
    }

    const v3Meta: Record<string, unknown> = {
      zarr_format: 3,
      node_type: "array",
      shape: meta.shape,
      chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
      attributes: arrayAttributes,
    };
    if (Array.isArray(dims) && dims.length === meta.shape.length) {
      v3Meta.dimension_names = dims;
    }

    // Map a v2 chunk name ("1.2") to its v3 key ("c/1/2"; "c" for scalars).
    const toV3ChunkKey = (name: string) =>
      meta.shape.length === 0 ? `${prefix}c` : `${prefix}c/${name.split(".").join("/")}`;

    if (isJson2 || isFixedString) {
      // Strings are inlined as one chunk covering the whole array.
      v3Meta.data_type = "string";
      v3Meta.chunk_grid = { name: "regular", configuration: { chunk_shape: meta.shape } };
      v3Meta.codecs = [{ name: "vlen-utf8" }];
      v3Meta.fill_value = "";
      for (const name of chunkNames) {
        const strings = decodeStringChunk(v2Refs[`${prefix}${name}`], meta);
        refs[toV3ChunkKey(name)] = inlineBytes(encodeVlenUtf8(strings));
      }
      refs[`${prefix}zarr.json`] = jsonStringifySorted(v3Meta);
      continue;
    }

    const match = meta.dtype.match(/^([<>|])([a-z]\d+)$/);
    const dataType = match ? V3_DATA_TYPES[match[2]] : undefined;
    if (!dataType) {
      continue;
    }
    const itemSize = parseInt(match![2].slice(1), 10);

    const codecs: CodecMeta[] = [
      itemSize > 1
        ? { name: "bytes", configuration: { endian: match![1] === ">" ? "big" : "little" } }
        : { name: "bytes" },
    ];
    for (const filter of meta.filters ?? []) {
      codecs.push(v2CodecToV3(filter, itemSize));
    }
    if (meta.compressor) {
      codecs.push(v2CodecToV3(meta.compressor, itemSize));
    }

    v3Meta.data_type = dataType;
    v3Meta.chunk_grid = { name: "regular", configuration: { chunk_shape: meta.chunks } };
    v3Meta.codecs = codecs;
    v3Meta.fill_value = meta.fill_value ?? (dataType === "bool" ? false : 0);
    refs[`${prefix}zarr.json`] = jsonStringifySorted(v3Meta);

    for (const name of chunkNames) {
      refs[toV3ChunkKey(name)] = v2Refs[`${prefix}${name}`];
    }
  }

  return { version: 1, refs };
}

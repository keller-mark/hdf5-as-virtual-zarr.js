# HdfStore Specification

**Status:** Draft
**Reference implementation:** `hdf5-as-virtual-zarr` (TypeScript), itself a port of `kerchunk.hdf.SingleHdf5ToZarr`

## 1. Introduction

An **HdfStore** is a read-only Zarr store that presents the contents of a single HDF5 file as a **Zarr v3** hierarchy, without converting or copying the file. Zarr metadata documents (`zarr.json`) are made up from the HDF5 object headers. Requests for chunk keys become byte-range reads against the original HDF5 file.

This document describes HdfStore without tying it to one language or Zarr library. A conforming implementation lets any Zarr v3 reader (zarr-python ≥ 3, zarrita, zarr-java, tensorstore, etc.) open an HDF5 file with its normal array and group APIs.

### 1.1 Design goals

1. **Partial reads only.** An implementation MUST NOT need the whole HDF5 file in memory. HDF5 files can be many gigabytes, and they are often read over HTTP. Every read of file bytes MUST be a bounded byte-range read.
2. **Metadata-proportional cost.** Opening a store SHOULD read only HDF5 structural metadata: the superblock, object headers, B-trees, and heaps. It SHOULD NOT read bulk array data. The only exceptions are small payloads that get *inlined* (§6.5) and string data (§6.6), which has no byte-range representation.
3. **Interoperability with kerchunk.** The virtual key/value map an HdfStore exposes is a valid [Zarr Reference Specification v1](https://fsspec.github.io/kerchunk/spec.html) `refs` object (the reference spec does not depend on the Zarr format version). An implementation MAY serialize it as a reference-spec JSON document. Its byte-range references SHOULD be identical to the ones `kerchunk.hdf.SingleHdf5ToZarr(...).translate()` produces for the same file. Kerchunk itself emits Zarr v2 metadata; Appendix B describes how the two map onto each other.

### 1.2 Terminology

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are to be read as described in RFC 2119.

| Term | Meaning |
|---|---|
| **Source** | The byte-addressable backing resource that holds the HDF5 file (local file, HTTP URL, object-store blob, in-memory buffer, ...). |
| **Range read** | `readRange(offset, length) -> bytes`. Returns exactly `length` bytes starting at absolute file offset `offset`. |
| **Reference map** | The mapping from Zarr keys to *reference entries* (§5) that the HdfStore serves. |
| **Zarr key** | A store key with no leading slash, e.g. `zarr.json`, `X/zarr.json`, `X/c/0/1`. |
| **Object path** | The `/`-separated HDF5 path of a group or dataset relative to the root, with no leading slash (root = empty string). |
| **Node prefix** | `""` for the root, otherwise the object path followed by `/`. Written `{p}` below, so `{p}zarr.json` is `zarr.json` at the root and `obs/zarr.json` for the group `obs`. |

## 2. Architecture

An HdfStore has two layers:

```
 Zarr client ──get(key)──▶ HdfStore ──lookup──▶ Reference map
                              │                     ▲
                              │                     │ built once at open time
                              │              HDF5 metadata translator
                              │                     │
                              └──readRange──▶ Source ◀┘
```

1. **Translator** (open time): walks the HDF5 file with range reads and builds the reference map (§4–§6).
2. **Reference store** (read time): answers `get(key)` from the reference map. It either returns inline bytes or does one range read against the source (§7).

The two layers are independent. An implementation MAY:
- persist the reference map (as reference-spec JSON) and reuse it later without running the translator again, or
- build the reference map lazily, translating each group or dataset the first time one of its keys is requested, as long as the observable results match eager translation.

## 3. Source interface

The translator and the reference store share one abstract source:

```
interface Source {
    readRange(offset: uint64, length: uint64) -> bytes   // REQUIRED
    readAll() -> bytes                                   // OPTIONAL
}
```

- `readRange` MUST return exactly `length` bytes, or fail. It MUST NOT return fewer bytes without signaling an error.
- Over HTTP, `readRange` maps to `GET` with `Range: bytes={offset}-{offset+length-1}`.
- Implementations SHOULD put a cache or coalescing layer in front of the source. Translation makes many small reads (often 1–64 bytes) at addresses that are close together. Block-aligned caching (for example, 64 KiB pages) greatly reduces round trips on high-latency sources. This is an optimization and does not affect correctness.
- Implementations MAY pass request options (headers, credentials, abort signals) from `get` calls through to `readRange`.

## 4. HDF5 parsing requirements

The translator MUST parse the HDF5 structures listed below. It MUST obtain every structure through `readRange` calls sized from information already decoded (header fields, message sizes, entry counts). It MUST NOT read speculatively to the end of the file.

All multi-byte integers in HDF5 metadata are little-endian.

### 4.1 Supported file features

| Feature | Requirement |
|---|---|
| Superblock versions 0, 2, 3 | MUST |
| Superblock version 1 | MAY (rejecting with a clear error is acceptable) |
| *Size of Offsets* = 8 and *Size of Lengths* = 8 | MUST. Other sizes MAY be rejected. |
| Userblock (superblock not at offset 0) | MAY. A conforming implementation MAY require the superblock at offset 0. |
| Object header v1 and v2, including continuation blocks | MUST |
| "Old-style" groups (Symbol Table message → v1 group B-tree + local heap) | MUST |
| "New-style" compact groups (Link messages in the object header) | MUST |
| "New-style" dense groups (Link Info message → v2 B-tree + fractal heap) | MUST |
| Hard links and soft links | MUST |
| External links | MAY skip |
| Compact and dense attribute storage | MUST |
| Data layout: contiguous (class 1) and chunked (class 2) with v1 chunk B-tree | MUST |
| Data layout: compact (class 0) | SHOULD (inline the bytes) |
| Chunk indexes from layout message v4 (fixed array, extensible array, v2 B-tree, single chunk) | SHOULD |
| Virtual datasets (layout class 3) | MAY skip |

If the translator hits an unsupported structure inside one object, it SHOULD skip that object (and log a warning) and keep translating the rest of the file instead of failing outright.

### 4.2 Bootstrap sequence

1. **Superblock.** Read a fixed-size prefix at offset 0 that covers the largest supported superblock layout. Check the 8-byte signature `\x89HDF\r\n\x1a\n`. The superblock version is the byte at offset 8.
2. **Root object header address.**
   - v0/v1: the root group symbol-table entry that immediately follows the superblock gives the object header address.
   - v2/v3: the *Root Group Object Header Address* field gives it.
3. **Object header.** Peek the first byte at the address:
   - `0x01` → object header v1 (16-byte prefix, then `object_header_size` bytes of messages).
   - `O` (start of `OHDR`) → object header v2. Read the fixed prefix, then the optional timestamp/attribute-phase fields that `flags` selects, then the size-of-chunk-0 field (1/2/4/8 bytes, also selected by `flags`).
   - Follow every **Object Header Continuation** message (type `0x0010`) and fetch each continuation block by its (address, length). For v2 headers, the continuation block starts with a 4-byte `OCHK` signature that MUST be skipped.
   - Collect every header message as (type, size, absolute offset of the message body).

Message types the translator uses:

| Type | Name | Used for |
|---|---|---|
| `0x0001` | Dataspace | Dataset shape. A dataset is any object that has this message. |
| `0x0002` | Link Info | Dense link storage |
| `0x0003` | Datatype | Dataset and attribute dtype |
| `0x0005` | Fill Value | (informational) |
| `0x0006` | Link | Compact link storage |
| `0x0008` | Data Layout | Storage class, addresses, chunk shape |
| `0x000B` | Filter Pipeline | Compression and filters |
| `0x000C` | Attribute | Compact attributes |
| `0x0010` | Continuation | More header blocks |
| `0x0011` | Symbol Table | Old-style group links |
| `0x0015` | Attribute Info | Dense attributes |

### 4.3 Traversal

Starting at the root group, the translator does a depth-first walk:

```
visit(group, path):
    emit group metadata for `path`                       (§6.1)
    for (name, target) in links(group):
        child_path = path == "" ? name : path + "/" + name
        if target is a soft link: resolve relative to the file root
        obj = open_object_header(target)
        if obj has a Dataspace message:  emit dataset for child_path  (§6.2)
        else:                            visit(obj, child_path)
```

- Child order SHOULD be name order for dense/indexed groups (or creation order when a creation-order index exists). Order does not affect the reference map.
- If a child fails to open or decode, the translator SHOULD skip it and continue.
- Implementations SHOULD track visited object addresses to avoid infinite loops on cyclic hard links. A hard link to an already-visited object MAY be emitted again under the new path (the same thing as an HDF5 hard-link alias) or skipped.


## 5. Reference map

The reference map is a dictionary from Zarr key (string, no leading `/`) to one of:

| Form | Meaning |
|---|---|
| `"<text>"` | Inline value. If it starts with `base64:`, the rest is standard base64 (RFC 4648 §4, with padding) and decodes to the value's bytes. Otherwise the value's bytes are the UTF-8 encoding of the string. |
| `[url, offset, length]` | Byte-range reference: `length` bytes starting at `offset` in the file at `url`. |
| `[url]` | Whole-file reference (not produced by the translator; accepted on read for spec compatibility). |

`url` MAY be `null`. A `null` URL means "the source this HdfStore was built from." Implementations that serialize the map for use elsewhere MAY write the real file URL instead.

When serialized, the map is wrapped as:

```json
{ "version": 1, "refs": { "<key>": <entry>, ... } }
```

The map contains exactly two kinds of keys:

- `{p}zarr.json`: one per group and per array (§6.1, §6.4).
- `{p}c/i0/i1/…`: chunk keys for arrays (§6.5). For 0-d arrays the chunk key is `{p}c`.

It MUST NOT contain Zarr v2 keys (`.zgroup`, `.zarray`, `.zattrs`, `.zmetadata`, or dot-separated chunk keys).

### 5.1 Inline byte encoding

To store raw bytes `b` inline:

- If every byte in `b` is `< 0x80`, store the string whose code points are exactly the bytes of `b`.
- Otherwise store `"base64:" + base64(b)`.

This matches kerchunk's encoding, and the decoding rule in §5 reverses it exactly.

### 5.2 Metadata document encoding

`zarr.json` values are JSON documents stored as inline strings. To keep output byte-for-byte reproducible across implementations, they SHOULD be serialized:

- with object keys sorted lexicographically at every nesting level,
- with no insignificant whitespace (separators `,` and `:`),
- with non-finite float fill values written as the JSON strings `"NaN"`, `"Infinity"`, `"-Infinity"` (Zarr v3 convention).

## 6. Translation rules

### 6.1 Groups

For every group (including the root), emit:

```
{p}zarr.json  →  {"attributes":{…},"node_type":"group","zarr_format":3}
```

`attributes` holds the group's transferred attributes (§6.8). It is `{}` if there are none.

### 6.2 Datasets: classification

Read the dataset's Dataspace (shape), Datatype, Data Layout, and Filter Pipeline messages, then classify:

| Class | Condition | Handling |
|---|---|---|
| **Scalar** | Dataspace rank 0 | §6.7 |
| **Compound** | Datatype class 6 | MUST skip (emit no keys). Structured data types are not part of Zarr v3 core. |
| **VLEN string** | Datatype class 9, type = string | §6.6 (inlined) |
| **VLEN sequence** | Datatype class 9, type = sequence | MUST skip |
| **Fixed-length string** | Datatype class 3 | §6.6 (inlined) |
| **Reference** | Datatype class 7 | MUST skip |
| **Numeric** | Classes 0 (fixed-point), 1 (floating-point), 8 (enum, mapped to its base type) | §6.3–§6.5 |

### 6.3 Data type mapping

| HDF5 datatype | Zarr v3 `data_type` | Item size |
|---|---|---|
| Fixed-point, signed, size 1 / 2 / 4 / 8 | `int8` / `int16` / `int32` / `int64` | 1 / 2 / 4 / 8 |
| Fixed-point, unsigned, size 1 / 2 / 4 / 8 | `uint8` / `uint16` / `uint32` / `uint64` | 1 / 2 / 4 / 8 |
| Floating-point, size 2 / 4 / 8 | `float16` / `float32` / `float64` | 2 / 4 / 8 |
| Enumerated | the mapping of its base type | |
| Fixed-length string | `string` (re-encoded, §6.6) | |
| Variable-length string | `string` (re-encoded, §6.6) | |

Byte order comes from bit 0 of class bit field 0 (`0` = little, `1` = big). It is expressed through the `bytes` codec (§6.4.1), not through the data type. For fixed-point, signedness comes from bit 3 of class bit field 0.

> **Booleans.** h5py stores `bool` as an enum of `int8` with members `FALSE=0` and `TRUE=1`. Following the rule above, such datasets map to `int8`. Implementations MAY instead map exactly that enum to `bool`. Readers can still tell booleans apart through format-level attributes such as AnnData's `encoding-type`.

### 6.4 Datasets: array `zarr.json`

For a numeric dataset at object path `p`, emit:

```json
{
  "attributes":         { … },
  "chunk_grid":         {"configuration":{"chunk_shape": <see below>},"name":"regular"},
  "chunk_key_encoding": {"configuration":{"separator":"/"},"name":"default"},
  "codecs":             [ <see §6.4.1> ],
  "data_type":          <see §6.3>,
  "dimension_names":    ["phony_dim_0", …, "phony_dim_{rank-1}"],
  "fill_value":         <see below>,
  "node_type":          "array",
  "shape":              <dataspace current dimensions>,
  "zarr_format":        3
}
```

**Chunk shape.** For chunked layouts (class 2), the layout message stores `rank + 1` dimension sizes. The last one is the element size in bytes and MUST be dropped. For contiguous and compact layouts, `chunk_shape` = `shape`.

**`fill_value`.** Use `0` for integers and `0.0` for floats. `fill_value` MUST NOT be `null` in Zarr v3. Implementations MAY instead use the HDF5 Fill Value message, or a `_FillValue` attribute if one exists.

**Dimension names.** `dimension_names` holds one `phony_dim_i` per dimension. Implementations MAY derive real names from HDF5 dimension scales (`DIMENSION_LIST`) as kerchunk does, but `phony_dim_i` is the conforming baseline. The xarray-on-Zarr-v2 attribute `_ARRAY_DIMENSIONS` MUST NOT appear in `attributes`, because v3 has a dedicated field for it.

**Attributes.** `attributes` holds the dataset's transferred attributes (§6.8), or `{}`.

#### 6.4.1 Filter pipeline → codecs

A v3 codec list is one optional array-to-array step, exactly one array-to-bytes codec, and then any number of bytes-to-bytes codecs, all in encode order. HDF5 filters operate on bytes, so every HDF5 filter becomes a bytes-to-bytes codec placed after `bytes`, in the same order as the HDF5 pipeline:

1. Always start with the array-to-bytes codec:
   - item size > 1: `{"name":"bytes","configuration":{"endian":"little" | "big"}}`
   - item size = 1: `{"name":"bytes"}`
2. Then, for each HDF5 filter in pipeline order:

| HDF5 filter id | Name | v3 codec appended |
|---|---|---|
| 2 | shuffle | `{"name":"numcodecs.shuffle","configuration":{"elementsize": cd[0] ?? itemsize}}` |
| 1 | deflate | `{"name":"numcodecs.zlib","configuration":{"level": cd[0] ?? 4}}` |
| 32001 | blosc | `{"name":"blosc","configuration":{"cname": CNAMES[cd[6]],"clevel": cd[4],"shuffle": SHUF[cd[5]],"typesize": itemsize,"blocksize": cd[3]}}` |
| 32015 | zstd | `{"name":"zstd","configuration":{"level": cd[0] ?? 3,"checksum":false}}` |
| 3 | fletcher32 | none (the checksum is stripped; see §6.5) |
| other | — | SHOULD skip the dataset (or emit metadata with a warning). Data decoded without the right codec would be silently wrong. |

Where `cd` is the filter's client-data array, `CNAMES = ["blosclz","lz4","lz4hc","snappy","zlib","zstd"]`, and `SHUF = ["noshuffle","shuffle","bitshuffle"]`.

Notes:
- HDF5 deflate writes a **zlib** stream (RFC 1950), not gzip (RFC 1952). So the v3 core `gzip` codec MUST NOT be used. `numcodecs.zlib` is the codec name registered by `numcodecs.zarr3` and by zarrita.
- HDF5 shuffle is a byte transposition over the encoded chunk. `numcodecs.shuffle` (bytes-to-bytes) matches it exactly.
- The HDF5 blosc filter's `cd_values` layout is `[filter_rev, blosc_ver, typesize, bufsize, clevel, shuffle, compressor]`. Blosc frames are self-describing, so decoders don't depend on these values. They are given for completeness.

### 6.5 Numeric datasets: chunk references

The Zarr chunk key for chunk grid index `(i0, i1, ..., i{r-1})` is `{p}c/i0/i1/…/i{r-1}` (the v3 `default` chunk key encoding with `/` separator).

**Contiguous layout (class 1).** Read the data address (8 bytes) from the layout message. For layout message v3+, also read the size (8 bytes). For v1/v2, compute size = `product(shape) × itemsize`.

- If the address is the undefined address (`0xFFFFFFFFFFFFFFFF`), the dataset was never written. Emit no chunk keys; readers will see `fill_value`.
- Otherwise emit a single chunk `{p}c/0/0/…/0` (one `0` per dimension).

**Chunked layout (class 2), v1 B-tree index.** Walk the v1 B-tree (node type 1) rooted at the layout message's B-tree address. Each node has a fixed 24-byte header (`TREE`, type, level, entries used, left/right sibling) followed by `entries_used` (key, child-pointer) pairs. A raw-data-chunk key is:

```
chunk_size  : uint32     // stored (filtered) size in bytes
filter_mask : uint32
offsets     : uint64[rank + 1]   // element offsets of the chunk's first element; last is always 0
```

Read internal nodes (level > 0) to find their children. For every key in a leaf (level 0):

- chunk grid index `i_d = offsets[d] / chunk_shape[d]` for `d < rank`
- byte offset = the child pointer
- byte length:
  - if the dataset has any filters: `chunk_size`
  - otherwise: `product(chunk_shape) × itemsize`
  - then, if fletcher32 (filter id 3) is in the pipeline: subtract 4 (the trailing checksum)

Chunks that are not in the B-tree are unallocated. Emit no keys for them.

If the `filter_mask` of a chunk has bits set (some filters were skipped for that chunk), the stored bytes do not match the dataset-level codec list. Implementations SHOULD inline such chunks after decoding them, or skip the dataset with a warning.

**Inlining small chunks.** If the computed byte length is ≤ `inline_threshold` (default **300** bytes; configurable; `0` disables), the translator SHOULD read those bytes and store them inline (§5.1) instead of as a `[url, offset, length]` triple. Otherwise emit:

```
{p}c/i0/…  →  [url, offset, length]
```

HDF5 always stores edge chunks at full chunk size, which matches the Zarr v3 regular chunk grid. No special handling is needed for partial edge chunks.

### 6.6 String datasets (inlined)

String data can't be served as raw byte ranges: VLEN strings live in the global heap, and fixed-length strings use HDF5 padding rules. The translator MUST therefore read the strings and inline them, re-encoded with the v3 `vlen-utf8` codec. Both VLEN and fixed-length strings produce the same array metadata:

```json
{
  "attributes":         { … },
  "chunk_grid":         {"configuration":{"chunk_shape": <shape>},"name":"regular"},
  "chunk_key_encoding": {"configuration":{"separator":"/"},"name":"default"},
  "codecs":             [{"name":"vlen-utf8"}],
  "data_type":          "string",
  "dimension_names":    [ … ],
  "fill_value":         "",
  "node_type":          "array",
  "shape":              <shape>,
  "zarr_format":        3
}
```

The array is stored as **a single chunk covering the whole array**: `chunk_shape` MUST equal `shape`, and the only chunk key is `{p}c/0/…/0`. HDF5 string compression filters are not carried over; they were already applied while decoding.

**Reading the strings.**

- *Variable-length (class 9, type = string):*
  1. Read every element's 16-byte VLEN descriptor (`uint32 length`, `uint64 global-heap collection address`, `uint32 object index`) from the dataset's storage. Storage is either contiguous, or every chunk in the B-tree in chunk-index order. If the chunks are filtered, decode them with the dataset's filter pipeline first.
  2. Resolve each descriptor against its **Global Heap Collection** (`GCOL`). Each collection SHOULD be fetched once and cached per dataset.
  3. Decode with the character set from the datatype message (0 = ASCII, 1 = UTF-8).
- *Fixed-length (class 3, size N):* read the raw `product(shape) × N` bytes, split them into N-byte items, strip trailing NUL (`0x00`) padding, and decode as UTF-8.

**Chunk encoding (`vlen-utf8`).** For strings `s_0 … s_{n-1}` in C order, `n = product(shape)`:

```
uint32 LE  n
repeat n times:
    uint32 LE  byte length of UTF-8(s_i)
    bytes      UTF-8(s_i)
```

Inline the result (§5.1). In practice this is almost always `base64:`, because the length prefixes usually contain bytes ≥ `0x80` or are not printable.

### 6.7 Scalar datasets

For a dataset with rank 0, emit array metadata with `"shape":[]`, `"chunk_shape":[]`, and `"dimension_names":[]`. Everything else follows §6.4 (numeric) or §6.6 (strings). The single chunk key is `{p}c`.

- Numeric scalar: codecs `[{"name":"bytes", …}]` with no compression. The chunk is the element's raw bytes, encoded in the dataset's own data type and byte order, inlined (§5.1).
- String scalar (VLEN or fixed): `vlen-utf8` with `n = 1`.

### 6.8 Attribute transfer

For each HDF5 object (root, groups, datasets), collect its attributes from both compact storage (Attribute messages, v1 or v3) and dense storage (Attribute Info → fractal heap + v2 B-tree). Then apply these rules in order:

1. **Drop hidden attributes:** `REFERENCE_LIST`, `CLASS`, `DIMENSION_LIST`, `NAME`, `_Netcdf4Dimid`, `_Netcdf4Coordinates`, `_nc3_strict`, `_NCProperties`, `_FillValue`.
2. A missing or undecodable value becomes `""`. Attributes whose datatype can't be decoded at all MAY be skipped.
3. Drop any attribute whose value is the string `"DIMENSION_SCALE"`.
4. **Booleans:** if the attribute's HDF5 datatype class is 8 (enumerated), map `0 → false` and `1 → true`. This follows h5py's boolean convention.
5. Convert numeric arrays to JSON arrays and decode string arrays (fixed or VLEN) to arrays of strings.
6. **Unwrap singletons:** an array with exactly one element becomes that element.
7. Non-finite floats are written as `"NaN"`, `"Infinity"`, `"-Infinity"`.

The result is the `attributes` object of the node's `zarr.json`.

## 7. Store read semantics

An HdfStore implements the read-only part of the host Zarr library's store interface:

```
get(key, options?) -> bytes | NOT_FOUND
```

1. **Normalize the key.** Strip one leading `/` if present (many libraries use absolute store paths like `/X/zarr.json`).
2. **Look it up** in the reference map. If the key is absent, return `NOT_FOUND` (`undefined`, `None`, `KeyError`, or HTTP 404 as the host library expects). Absent chunk keys are normal: Zarr readers fill those regions with `fill_value`. Requests for Zarr v2 keys (`.zgroup`, `.zarray`, …) MUST return `NOT_FOUND`, so that clients that probe both formats detect v3.
3. **String entry:** decode as in §5 and return the bytes.
4. **`[url, offset, length]` entry:** return `source.readRange(offset, length)`. A `null` URL refers to the store's own source. An implementation MAY support non-null URLs that point to other resources, but MUST NOT need them for files it translated itself.
5. **`[url]` entry:** return the whole referenced resource.

The store:
- MUST be read-only. Writes, deletes, and listing MAY be unsupported.
- SHOULD support `list` / `list_dir` / `exists` where the host library requires them (e.g. for group membership), by enumerating reference-map keys.
- MAY embed consolidated metadata in the root `zarr.json`, using the `consolidated_metadata` field that zarr-python writes (`{"kind":"inline","must_understand":false,"metadata":{<path>: <node zarr.json>}}`). This lets clients open the whole hierarchy with one request. Consolidated metadata is not part of the Zarr v3 core spec, so readers MUST still be able to open each node individually.
- SHOULD NOT implement `getRange` on top of virtual keys unless the host library needs it. Chunk references already map to one source range each.

### 7.1 Construction API

A conforming library SHOULD offer these entry points (names are illustrative):

```
HdfStore.fromSource(source, options?) -> HdfStore        // translate, then serve (v3)
HdfStore.fromReferenceSpec(spec, source) -> HdfStore     // serve a precomputed v3 map
translate(source, options?) -> ReferenceSpec             // translation only
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `url` | `null` | URL written into `[url, offset, length]` triples |
| `inline_threshold` | `300` | Max size in bytes of chunk payloads to inline. `0` disables inlining. |
| `zarr_format` | `3` for `HdfStore`; implementation-defined for `translate` | MAY be offered to emit the kerchunk-compatible v2 map (Appendix B) from `translate`. An HdfStore always serves v3. |

## 8. Conformance

An implementation conforms if, for any HDF5 file that uses only features marked MUST in §4.1:

1. For every non-skipped dataset `p`, opening `p` as a Zarr v3 array through the HdfStore and reading it in full gives the same values as `h5py.File(f)[p][()]` (strings compared after decoding to text).
2. For every group and dataset, the Zarr attributes equal the result of applying §6.8 to the HDF5 attributes, and array `dimension_names` follow §6.4.
3. Every `zarr.json` document is valid against the Zarr v3 core metadata schema.
4. Opening the store and reading any single numeric chunk never reads more of the source than the structural metadata plus that chunk's bytes.

Implementations SHOULD also check that every `[url, offset, length]` triple matches the corresponding triple from `kerchunk.hdf.SingleHdf5ToZarr(f, inline_threshold=300, vlen_encode="embed").translate()` under the key mapping in Appendix B. Representative fixtures: numeric contiguous, chunked + gzip + shuffle, fletcher32, VLEN strings, fixed-length strings, scalars, nested groups, dense attributes, and AnnData `.h5ad` files.

## 9. Security considerations

- All addresses and sizes come from untrusted file contents. Implementations MUST bounds-check them against sane limits (for example, refuse single metadata reads larger than a configurable maximum) to avoid unbounded memory allocation.
- B-tree, heap, and continuation traversal MUST guard against cycles and unbounded recursion.
- Inlining reads bulk data at open time. Implementations SHOULD cap the total bytes inlined per dataset (e.g. by refusing to inline string datasets above a configurable size) so a small-looking file can't force a huge download.

## Appendix A. Differences in the reference implementation

The TypeScript reference implementation differs from this spec in a few places. New implementations should follow the spec:

| Area | Spec | Reference implementation today |
|---|---|---|
| Compressed VLEN chunks | Decoded before reading descriptors | Read raw (assumes unfiltered) |
| Blosc client data indices | `cd[3]`=bufsize, `cd[4]`=clevel, `cd[5]`=shuffle, `cd[6]`=compressor | Uses `cd[2]`, `cd[3]`, `cd[5]`, `cd[4]` (harmless for decoding, since blosc frames are self-describing) |
| Unknown filters | Skip dataset / warn | Ignored silently (szip etc.) |
| `filter_mask` per chunk | Honored | Ignored |
| Fill value | Defaults (MAY use HDF5 fill value) | Defaults only |
| Layout message v4 chunk indexes, compact layout | SHOULD | Not supported |
| Shuffle default `elementsize` | dataset itemsize | `4` (HDF5 always stores the element size in `cd[0]`, so the default is never used in practice) |

## Appendix B. Relationship to kerchunk's Zarr v2 output

Kerchunk's `SingleHdf5ToZarr` emits a Zarr v2 reference map. The reference implementation builds that v2 map first and then converts it to v3 (`refSpecV2ToV3`). Any implementation can use the same mapping to check v3 output against kerchunk, or to offer v2 output as an option:

| Zarr v2 (kerchunk) | Zarr v3 (HdfStore) |
|---|---|
| `{p}.zgroup` + `{p}.zattrs` | `{p}zarr.json` with `node_type: "group"`, `attributes` = parsed `.zattrs` |
| `{p}.zarray` + `{p}.zattrs` | `{p}zarr.json` with `node_type: "array"` |
| `.zattrs._ARRAY_DIMENSIONS` | `dimension_names` (removed from `attributes`) |
| `dtype` `"<f4"`, `">i8"`, `"\|u1"`, … | `data_type` `float32`, `int64`, `uint8`, … plus `bytes` codec `endian` |
| `filters: [{"id":"shuffle",…}]` | `{"name":"numcodecs.shuffle",…}` after `bytes` |
| `compressor: {"id":"zlib",…}` | `{"name":"numcodecs.zlib",…}` last |
| `compressor: {"id":"blosc",…}` / `{"id":"zstd",…}` | v3 core `blosc` / `zstd` |
| `chunks` | `chunk_grid.configuration.chunk_shape` |
| `fill_value: null` (numeric) | `0` |
| `dtype "\|O"` + `json2` filter, chunk `[…items, "\|O", shape]` | `data_type: "string"`, `vlen-utf8`, `chunk_shape = shape` |
| `dtype "\|SN"`, raw N-byte items | `data_type: "string"`, `vlen-utf8`, `chunk_shape = shape` |
| `dtype "\|O"` without `json2` | dropped |
| chunk key `{p}i0.i1.…` | `{p}c/i0/i1/…` (byte-range entry unchanged) |
| scalar chunk key `{p}0` | `{p}c` |

# HdfStore Specification

**Status:** Draft
**Reference implementation:** `hdf5-as-virtual-zarr` (TypeScript), itself a port of `kerchunk.hdf.SingleHdf5ToZarr`

## 1. Introduction

An **HdfStore** is a read-only Zarr store that presents the contents of a single HDF5 file as a Zarr v2 hierarchy, without converting or copying the file. Zarr metadata documents (`.zgroup`, `.zarray`, `.zattrs`) are made up from the HDF5 object headers. Requests for chunk keys become byte-range reads against the original HDF5 file.

This document describes HdfStore without tying it to one language or Zarr library. A conforming implementation lets any Zarr v2 reader (zarr-python, zarrita, zarr-java, tensorstore, etc.) open an HDF5 file with its normal array and group APIs.

### 1.1 Design goals

1. **Partial reads only.** An implementation MUST NOT need the whole HDF5 file in memory. HDF5 files can be many gigabytes, and they are often read over HTTP. Every read of file bytes MUST be a bounded byte-range read.
2. **Metadata-proportional cost.** Opening a store SHOULD read only HDF5 structural metadata: the superblock, object headers, B-trees, and heaps. It SHOULD NOT read bulk array data. The only exceptions are small payloads that get *inlined* (§6.5) and variable-length strings (§6.6), which have no byte-range representation.
3. **Interoperability with kerchunk.** The virtual key/value map an HdfStore exposes is a valid [Zarr Reference Specification v1](https://fsspec.github.io/kerchunk/spec.html) `refs` object. An implementation MAY serialize it as a reference-spec JSON document, and the result SHOULD be equivalent to what `kerchunk.hdf.SingleHdf5ToZarr(...).translate()` produces for the same file.

### 1.2 Terminology

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are to be read as described in RFC 2119.

| Term | Meaning |
|---|---|
| **Source** | The byte-addressable backing resource that holds the HDF5 file (local file, HTTP URL, object-store blob, in-memory buffer, ...). |
| **Range read** | `readRange(offset, length) -> bytes`. Returns exactly `length` bytes starting at absolute file offset `offset`. |
| **Reference map** | The mapping from Zarr keys to *reference entries* (§5) that the HdfStore serves. |
| **Zarr key** | A store key with no leading slash, e.g. `X/.zarray`, `obs/index/0`, `.zgroup`. |
| **Object path** | The `/`-separated HDF5 path of a group or dataset relative to the root, with no leading slash (root = empty string). |

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

### 5.1 Inline byte encoding

To store raw bytes `b` inline:

- If every byte in `b` is `< 0x80`, store the string whose code points are exactly the bytes of `b`.
- Otherwise store `"base64:" + base64(b)`.

This matches kerchunk's encoding, and the decoding rule in §5 reverses it exactly.

### 5.2 Metadata document encoding

`.zgroup`, `.zarray`, and `.zattrs` values are JSON documents stored as inline strings. To keep output byte-for-byte reproducible across implementations, they SHOULD be serialized:

- with object keys sorted lexicographically at every nesting level,
- with no insignificant whitespace (separators `,` and `:`),
- with non-finite floats written as the JSON strings `"NaN"`, `"Infinity"`, `"-Infinity"` (Zarr v2 convention).

A `fill_value` that is a whole number on a floating-point dtype SHOULD be written with a trailing `.0` (e.g. `0.0`, not `0`) to match Python's `json` output.

## 6. Translation rules

### 6.1 Groups

For the root group, emit:

```
.zgroup  →  {"zarr_format":2}
```

For every other group at object path `p`:

```
p/.zgroup  →  {"zarr_format":2}
```

Then transfer the group's attributes (§6.8) to `.zattrs` / `p/.zattrs`.

### 6.2 Datasets: classification

Read the dataset's Dataspace (shape), Datatype, Data Layout, and Filter Pipeline messages, then classify:

| Class | Condition | Handling |
|---|---|---|
| **Scalar** | Dataspace rank 0 | §6.7 |
| **Compound** | Datatype class 6 | MUST skip (emit no keys). Zarr v2 structured dtypes are out of scope. |
| **VLEN string / sequence** | Datatype class 9 | §6.6 (inlined) |
| **Fixed-length string** | Datatype class 3 | §6.6 (inlined) |
| **Reference** | Datatype class 7 | SHOULD skip |
| **Numeric** | Classes 0 (fixed-point), 1 (floating-point), 8 (enum, mapped to its base type) | §6.3–§6.5 |

### 6.3 Datatype mapping

The Zarr v2 `dtype` is a NumPy type string:

| HDF5 datatype | Zarr `dtype` |
|---|---|
| Fixed-point, size N ∈ {2,4,8}, signed, little-endian | `<iN` |
| Fixed-point, size N ∈ {2,4,8}, unsigned, little-endian | `<uN` |
| Fixed-point, big-endian | `>iN` / `>uN` |
| Fixed-point, size 1 | `\|i1` / `\|u1` |
| Floating-point, size N ∈ {2,4,8} | `<fN` / `>fN` |
| Enumerated | the mapping of its base type |
| Fixed-length string, size N | `\|SN` |
| Variable-length string / sequence | `\|O` (object; see §6.6) |

Byte order comes from bit 0 of class bit field 0 (`0` = little, `1` = big). For fixed-point, signedness comes from bit 3 of class bit field 0.

### 6.4 Datasets: `.zarray`

For a non-scalar dataset at object path `p`, emit `p/.zarray`:

```json
{
  "chunks":      <chunk shape, or the dataset shape if not chunked>,
  "compressor":  <see §6.4.1>,
  "dtype":       <see §6.3>,
  "fill_value":  <see below>,
  "filters":     <see §6.4.1>,
  "order":       "C",
  "shape":       <dataspace current dimensions>,
  "zarr_format": 2
}
```

**Chunk shape.** For chunked layouts (class 2), the layout message stores `rank + 1` dimension sizes. The last one is the element size in bytes and MUST be dropped. For contiguous and compact layouts, `chunks` = `shape`.

**`fill_value`.** Use `""` for `|S*`, `null` for `|O`, `0.0` for floats, and `0` for integers. (Implementations MAY instead use the HDF5 Fill Value message, or a `_FillValue` attribute if one exists. kerchunk compatibility is defined against the defaults above.)

**Dimension names.** Also emit `p/.zattrs` with the transferred attributes (§6.8) plus:

```
"_ARRAY_DIMENSIONS": ["phony_dim_0", "phony_dim_1", ..., "phony_dim_{rank-1}"]
```

Implementations MAY derive real dimension names from HDF5 dimension scales (`DIMENSION_LIST`) as kerchunk does, but `phony_dim_i` is the conforming baseline.

#### 6.4.1 Filter pipeline → codecs

Process the HDF5 filter pipeline in order:

| HDF5 filter id | Name | Zarr output |
|---|---|---|
| 1 | deflate | `compressor = {"id":"zlib","level": cd[0] ?? 4}` |
| 2 | shuffle | append `{"id":"shuffle","elementsize": cd[0] ?? itemsize}` to `filters` |
| 3 | fletcher32 | none (the checksum is stripped; see §6.5) |
| 32001 | blosc | `compressor = {"id":"blosc","cname": CNAMES[cd[6]],"clevel": cd[4],"shuffle": cd[5],"blocksize": cd[3]}` where `CNAMES = ["blosclz","lz4","lz4hc","snappy","zlib","zstd"]` |
| 32015 | zstd | `compressor = {"id":"zstd","level": cd[0] ?? 3}` |
| other | — | SHOULD skip the dataset (or emit metadata with a warning). Data decoded without the right codec would be silently wrong. |

`cd` is the filter's client-data array. If no filters remain, `filters` is `null`. If there is no compressor, `compressor` is `null`.

> **Note on blosc client data.** The HDF5 blosc filter's `cd_values` layout is `[filter_rev, blosc_ver, typesize, bufsize, clevel, shuffle, compressor]`. Implementations MUST index into this layout as written above, and SHOULD check it against files produced by `hdf5plugin` / PyTables.

Zarr v2 applies `filters` in order on encode and in reverse on decode, with `compressor` applied last on encode. HDF5 applies its pipeline in order on write. So shuffle (applied before deflate in HDF5) correctly goes in `filters` and deflate in `compressor`.

### 6.5 Numeric datasets: chunk references

The Zarr chunk key for chunk grid index `(i0, i1, ..., i{r-1})` is `p/i0.i1.....i{r-1}` (dot separator, the Zarr v2 default).

**Contiguous layout (class 1).** Read the data address (8 bytes) from the layout message. For layout message v3+, also read the size (8 bytes). For v1/v2, compute size = `product(shape) × itemsize`.

- If the address is the undefined address (`0xFFFFFFFFFFFFFFFF`), the dataset was never written. Emit no chunk keys; readers will see `fill_value`.
- Otherwise emit a single chunk `p/0.0...0` (one `0` per dimension).

**Chunked layout (class 2), v1 B-tree index.** Walk the v1 B-tree (node type 1) rooted at the layout message's B-tree address. Each node has a fixed 24-byte header (`TREE`, type, level, entries used, left/right sibling) followed by `entries_used` (key, child-pointer) pairs. A raw-data-chunk key is:

```
chunk_size  : uint32     // stored (filtered) size in bytes
filter_mask : uint32
offsets     : uint64[rank + 1]   // element offsets of the chunk's first element; last is always 0
```

Read internal nodes (level > 0) to find their children. For every key in a leaf (level 0):

- chunk grid index `i_d = offsets[d] / chunks[d]` for `d < rank`
- byte offset = the child pointer
- byte length:
  - if the dataset has any filters: `chunk_size`
  - otherwise: `product(chunks) × itemsize`
  - then, if fletcher32 (filter id 3) is in the pipeline: subtract 4 (the trailing checksum)

Chunks that are not in the B-tree are unallocated. Emit no keys for them.

If the `filter_mask` of a chunk has bits set (some filters were skipped for that chunk), the stored bytes do not match the dataset-level codec list. Implementations SHOULD inline such chunks after decoding them, or skip the dataset with a warning.

**Inlining small chunks.** If the computed byte length is ≤ `inline_threshold` (default **300** bytes; configurable; `0` disables), the translator SHOULD read those bytes and store them inline (§5.1) instead of as a `[url, offset, length]` triple. Otherwise emit:

```
p/<chunk key>  →  [url, offset, length]
```

HDF5 always stores edge chunks at full chunk size, so no special handling is needed for partial edge chunks.

### 6.6 String datasets (inlined)

String data can't be served as raw byte ranges: VLEN strings live in the global heap, and fixed-length strings use HDF5 padding rules. The translator MUST therefore read and inline them. Inlined string datasets are stored as **a single chunk covering the whole array**: `chunks` MUST equal `shape`, and the only chunk key is `p/0.0...0`.

**Variable-length strings (class 9, type = string).**

1. Read every element's 16-byte VLEN descriptor (`uint32 length`, `uint64 global-heap collection address`, `uint32 object index`) from the dataset's storage (contiguous, or every chunk in the B-tree in key order).
2. Resolve each descriptor against its **Global Heap Collection** (`GCOL`). Each collection SHOULD be fetched once and cached per dataset.
3. Decode with the character set from the datatype message (0 = ASCII, 1 = UTF-8).
4. Rewrite `.zarray` with `"dtype":"|O"`, `"fill_value":null`, `"compressor":null`, and:

```json
"filters": [{"allow_nan":true,"check_circular":true,"encoding":"utf-8","ensure_ascii":true,
             "id":"json2","indent":null,"separators":[",",":"],"skipkeys":false,
             "sort_keys":true,"strict":true}]
```

5. Emit the chunk using numcodecs' `json2` encoding: a JSON array of all strings in C order, followed by the dtype string `"|O"`, followed by the shape array.

```
p/0  →  ["a","b","c","|O",[3]]
```

If the VLEN chunks are compressed, implementations MUST decode them with the dataset's filter pipeline before reading descriptors.

**Fixed-length strings (class 3, size N).** Read the raw `product(shape) × N` bytes, null-padding or truncating each element to N bytes, and inline them as one chunk (§5.1) with `"dtype":"|SN"` and `"fill_value":""`.

### 6.7 Scalar datasets

For a dataset with rank 0:

```
p/.zarray  →  {"chunks":[],"compressor":null,"dtype":<dtype>,"fill_value":<default>,
               "filters":null,"order":"C","shape":[],"zarr_format":2}
p/.zattrs  →  { ...attrs, "_ARRAY_DIMENSIONS": [] }
p/0        →  <inline value>
```

- Numeric scalar: inline the element's raw bytes, encoded in the dataset's own `dtype` (§5.1).
- VLEN string scalar: use the json2 rules from §6.6 with shape `[]`, e.g. `["hello","|O",[]]`.

### 6.8 Attribute transfer

For each HDF5 object (root, groups, datasets), collect its attributes from both compact storage (Attribute messages, v1 or v3) and dense storage (Attribute Info → fractal heap + v2 B-tree). Then apply these rules in order:

1. **Drop hidden attributes:** `REFERENCE_LIST`, `CLASS`, `DIMENSION_LIST`, `NAME`, `_Netcdf4Dimid`, `_Netcdf4Coordinates`, `_nc3_strict`, `_NCProperties`, `_FillValue`.
2. A missing or undecodable value becomes `""`. Attributes whose datatype can't be decoded at all MAY be skipped.
3. Drop any attribute whose value is the string `"DIMENSION_SCALE"`.
4. **Booleans:** if the attribute's HDF5 datatype class is 8 (enumerated), map `0 → false` and `1 → true`. This follows h5py's boolean convention.
5. Convert numeric arrays to JSON arrays and decode string arrays (fixed or VLEN) to arrays of strings.
6. **Unwrap singletons:** an array with exactly one element becomes that element.
7. Non-finite floats follow §5.2.

If at least one attribute remains, emit `.zattrs` (root) or `p/.zattrs` as a JSON object (§5.2). For datasets, `_ARRAY_DIMENSIONS` (§6.4) is merged into the same object.

## 7. Store read semantics

An HdfStore implements the read-only part of the host Zarr library's store interface:

```
get(key, options?) -> bytes | NOT_FOUND
```

1. **Normalize the key.** Strip one leading `/` if present (many libraries use absolute store paths like `/X/.zarray`).
2. **Look it up** in the reference map. If the key is absent, return `NOT_FOUND` (`undefined`, `None`, `KeyError`, or HTTP 404 as the host library expects). Absent chunk keys are normal: Zarr readers fill those regions with `fill_value`.
3. **String entry:** decode as in §5 and return the bytes.
4. **`[url, offset, length]` entry:** return `source.readRange(offset, length)`. A `null` URL refers to the store's own source. An implementation MAY support non-null URLs that point to other resources, but MUST NOT need them for files it translated itself.
5. **`[url]` entry:** return the whole referenced resource.

The store:
- MUST be read-only. Writes, deletes, and listing MAY be unsupported.
- SHOULD support `list`/`contains` where the host library requires it (e.g. for group membership), by enumerating reference-map keys.
- MAY additionally serve `.zmetadata` (Zarr v2 consolidated metadata) built from the map: `{"zarr_consolidated_format":1,"metadata":{<key>: <parsed JSON>}}` over every key that equals or ends in `/.zgroup`, `/.zarray`, or `/.zattrs`. This lets clients open the whole hierarchy with one request.
- SHOULD NOT implement `getRange` on top of virtual keys unless the host library needs it. Chunk references already map to one source range each.

### 7.1 Construction API

A conforming library SHOULD offer these entry points (names are illustrative):

```
HdfStore.fromSource(source, options?) -> HdfStore        // translate, then serve
HdfStore.fromReferenceSpec(spec, source) -> HdfStore     // serve a precomputed map
translate(source, options?) -> ReferenceSpec             // translation only
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `url` | `null` | URL written into `[url, offset, length]` triples |
| `inline_threshold` | `300` | Max size in bytes of chunk payloads to inline. `0` disables inlining. |

## 8. Conformance

An implementation conforms if, for any HDF5 file that uses only features marked MUST in §4.1:

1. For every non-skipped dataset `p`, opening `p` as a Zarr v2 array through the HdfStore and reading it in full gives the same values as `h5py.File(f)[p][()]`.
2. For every group and dataset, the Zarr attributes equal the result of applying §6.8 to the HDF5 attributes.
3. Opening the store and reading any single chunk never reads more of the source than the structural metadata plus that chunk's bytes.

Implementations SHOULD also check that the reference map's set of keys, `.zarray` documents, and byte-range triples match `kerchunk.hdf.SingleHdf5ToZarr(f, inline_threshold=300, vlen_encode="embed").translate()` for representative fixtures (numeric contiguous, chunked + gzip + shuffle, fletcher32, VLEN strings, fixed-length strings, scalars, nested groups, dense attributes, AnnData `.h5ad` files).

## 9. Security considerations

- All addresses and sizes come from untrusted file contents. Implementations MUST bounds-check them against sane limits (for example, refuse single metadata reads larger than a configurable maximum) to avoid unbounded memory allocation.
- B-tree, heap, and continuation traversal MUST guard against cycles and unbounded recursion.
- Inlining reads bulk data at open time. Implementations SHOULD cap the total bytes inlined per dataset (e.g. by refusing to inline VLEN string datasets above a configurable size) so a small-looking file can't force a huge download.

## Appendix A. Differences in the reference implementation

The TypeScript reference implementation (`src/single-hdf5-to-zarr.ts`) differs from this spec in a few places. New implementations should follow the spec:

| Area | Spec | Reference implementation today |
|---|---|---|
| Inlined string datasets | `chunks` = `shape` | Keeps the HDF5 chunk shape in `.zarray` while emitting one `0.0…0` chunk |
| Numeric scalar payload | Encoded in the dataset's dtype | Always encoded as float64 bytes |
| Compressed VLEN chunks | Decoded before reading descriptors | Read raw (assumes unfiltered) |
| Blosc client data indices | `cd[3]`=bufsize, `cd[4]`=clevel, `cd[5]`=shuffle, `cd[6]`=compressor | Uses `cd[2]`, `cd[3]`, `cd[5]`, `cd[4]` |
| Unknown filters | Skip dataset / warn | Ignored silently (szip etc.) |
| `filter_mask` per chunk | Honored | Ignored |
| Fill value | Defaults (MAY use HDF5 fill value) | Defaults only |
| Layout message v4 chunk indexes, compact layout | SHOULD | Not supported |
| Shuffle default `elementsize` | dataset itemsize | `4` |

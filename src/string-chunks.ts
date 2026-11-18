/**
 * On-demand reading of HDF5 string datasets as Zarr v3 `vlen-utf8` chunks.
 *
 * Variable-length strings are stored in HDF5 as an array of global heap IDs
 * (4-byte length, collection address, 4-byte object index) whose payloads live
 * in global heap collections ("GCOL") elsewhere in the file. Fixed-length strings
 * are stored inline as N-byte items. Neither can be referenced by byte offset in
 * a Zarr chunk manifest, so each Zarr chunk is computed when it is requested:
 *
 * - chunked layout: one Zarr chunk per HDF5 chunk (read via the chunk B-tree,
 *   with its filter pipeline undone in JS);
 * - contiguous / compact layout: "virtual" chunks of about
 *   {@link VIRTUAL_CHUNK_ELEMENTS} elements along the first dimension, each read
 *   with a single range request.
 *
 * Only the heap collections referenced by a chunk are fetched. Parsed collections
 * are kept in a {@link GlobalHeapCache} bounded by bytes; encoded chunks are not cached.
 */

import type { AsyncReadable } from "./types.js";
import { fetchRange, fetchRangeWithReadAhead } from "./types.js";
import type { DataObjects } from "./jsfive/dataobjects.js";
import { BTreeV1RawDataChunks } from "./jsfive/btree.js";
import { decodeChunk } from "./jsfive/filters.js";
import { UNDEFINED_ADDRESS } from "./jsfive/misc-low-level.js";

/** Elements per virtual chunk for contiguous / compact string datasets (1 MiB of 16-byte heap IDs). */
export const VIRTUAL_CHUNK_ELEMENTS = 65536;

/** Default byte budget of the global heap collection cache. */
export const DEFAULT_GLOBAL_HEAP_CACHE_BYTES = 32 * 1024 * 1024;

const DATA_STORAGE_MSG_TYPE = 0x0008;
/** Global heap collections are at least 4 KiB, so this read-ahead never crosses into another collection. */
const GLOBAL_HEAP_READ_AHEAD = 4096;
const GCOL_SIGNATURE = [0x47, 0x43, 0x4f, 0x4c]; // "GCOL"

/** Sizes of file addresses and lengths, from the superblock. */
export interface FileAddressing {
  offsetSize: number;
  lengthSize: number;
}

function readUint(view: DataView, offset: number, size: number): number {
  switch (size) {
    case 1:
      return view.getUint8(offset);
    case 2:
      return view.getUint16(offset, true);
    case 4:
      return view.getUint32(offset, true);
    case 8:
      return Number(view.getBigUint64(offset, true));
    default:
      throw new Error(`Unsupported HDF5 integer size: ${size}`);
  }
}

function isUndefinedAddress(address: number, size: number): boolean {
  return address === UNDEFINED_ADDRESS || address >= 2 ** (8 * size) - 1 || address >= Number.MAX_SAFE_INTEGER;
}

function product(values: number[]): number {
  return values.reduce((a, b) => a * b, 1);
}

// ────────── Global heap collections ──────────

/** A parsed global heap collection: its bytes plus an object index → (start, size) table. */
export class HeapCollection {
  readonly data: Uint8Array;
  readonly #starts: Float64Array;
  readonly #sizes: Float64Array;

  constructor(data: Uint8Array, lengthSize: number) {
    this.data = data;
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const objectHeaderSize = 8 + lengthSize;
    const indices: number[] = [];
    const starts: number[] = [];
    const sizes: number[] = [];
    let offset = 8 + lengthSize; // collection header
    let maxIndex = 0;
    while (offset + objectHeaderSize <= data.byteLength) {
      const index = view.getUint16(offset, true);
      if (index === 0) break; // free space: no more objects
      const size = readUint(view, offset + 8, lengthSize);
      const start = offset + objectHeaderSize;
      indices.push(index);
      starts.push(start);
      sizes.push(size);
      if (index > maxIndex) maxIndex = index;
      offset = start + Math.ceil(size / 8) * 8;
    }
    this.#starts = new Float64Array(maxIndex + 1).fill(-1);
    this.#sizes = new Float64Array(maxIndex + 1);
    for (let i = 0; i < indices.length; i++) {
      this.#starts[indices[i]] = starts[i];
      this.#sizes[indices[i]] = sizes[i];
    }
  }

  /** Approximate memory held by this collection. */
  get byteSize(): number {
    return this.data.byteLength + this.#starts.byteLength + this.#sizes.byteLength;
  }

  /** The bytes of a heap object (a view, not a copy), or undefined if absent. */
  object(index: number): Uint8Array | undefined {
    const start = index < this.#starts.length ? this.#starts[index] : -1;
    if (start < 0) return undefined;
    return this.data.subarray(start, start + this.#sizes[index]);
  }
}

async function readHeapCollection(
  source: AsyncReadable,
  address: number,
  lengthSize: number
): Promise<HeapCollection> {
  const headerSize = 8 + lengthSize;
  let data = new Uint8Array(
    await fetchRangeWithReadAhead(source, address, headerSize, GLOBAL_HEAP_READ_AHEAD)
  );
  if (!GCOL_SIGNATURE.every((b, i) => data[i] === b)) {
    throw new Error(`No global heap collection (GCOL) at address ${address}`);
  }
  const size = readUint(new DataView(data.buffer, data.byteOffset, data.byteLength), 8, lengthSize);
  if (data.byteLength > size) {
    data = data.slice(0, size);
  } else if (data.byteLength < size) {
    const rest = new Uint8Array(await fetchRange(source, address + data.byteLength, size - data.byteLength));
    const full = new Uint8Array(size);
    full.set(data);
    full.set(rest, data.byteLength);
    data = full;
  }
  return new HeapCollection(data, lengthSize);
}

/**
 * LRU cache of parsed global heap collections, bounded by total bytes.
 * Concurrent requests for the same collection share one read.
 * Callers hold references to the collections they are using, so eviction
 * never affects an in-progress chunk.
 */
export class GlobalHeapCache {
  readonly #source: AsyncReadable;
  readonly #lengthSize: number;
  readonly #maxBytes: number;
  #bytes = 0;
  /** Map iteration order is insertion order: oldest first. */
  readonly #entries = new Map<number, HeapCollection>();
  readonly #pending = new Map<number, Promise<HeapCollection>>();

  constructor(source: AsyncReadable, lengthSize: number, maxBytes: number) {
    this.#source = source;
    this.#lengthSize = lengthSize;
    this.#maxBytes = maxBytes;
  }

  /** Total bytes currently held by cached collections. */
  get bytes(): number {
    return this.#bytes;
  }

  get(address: number): Promise<HeapCollection> {
    const hit = this.#entries.get(address);
    if (hit) {
      // Move to the most recently used position.
      this.#entries.delete(address);
      this.#entries.set(address, hit);
      return Promise.resolve(hit);
    }
    let pending = this.#pending.get(address);
    if (!pending) {
      pending = readHeapCollection(this.#source, address, this.#lengthSize).finally(() => {
        this.#pending.delete(address);
      });
      pending.then((collection) => this.#insert(address, collection), () => {});
      this.#pending.set(address, pending);
    }
    return pending;
  }

  #insert(address: number, collection: HeapCollection) {
    const size = collection.byteSize;
    if (size > this.#maxBytes || this.#entries.has(address)) return;
    this.#entries.set(address, collection);
    this.#bytes += size;
    for (const [key, value] of this.#entries) {
      if (this.#bytes <= this.#maxBytes) break;
      this.#entries.delete(key);
      this.#bytes -= value.byteSize;
    }
  }
}

// ────────── vlen-utf8 encoding ──────────

/**
 * Encode byte strings with the Zarr v3 `vlen-utf8` codec layout: little-endian u32
 * element count, then for each element a u32 byte length followed by its bytes.
 * Missing elements (index >= items.length) are encoded as "".
 * Returns a fresh buffer (zarrita decodes from `bytes.buffer` at offset 0).
 */
export function encodeVlenUtf8Bytes(items: (Uint8Array | undefined)[], count: number): Uint8Array {
  let total = 4 + 4 * count;
  for (const item of items) if (item) total += item.byteLength;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, count, true);
  let pos = 4;
  for (let i = 0; i < count; i++) {
    const item = items[i];
    const length = item ? item.byteLength : 0;
    view.setUint32(pos, length, true);
    pos += 4;
    if (item) {
      out.set(item, pos);
      pos += length;
    }
  }
  return out;
}

// ────────── String array reader ──────────

type ElementType =
  | { kind: "vlen"; size: number }
  | { kind: "fixed"; size: number };

interface ChunkLocation {
  address: number;
  size: number;
  filterMask: number;
}

type Storage =
  | {
      kind: "chunked";
      btreeAddress: number | null;
      btreeDims: number;
      pipeline: Map<string, unknown>[] | null;
    }
  | { kind: "contiguous"; address: number | null }
  | { kind: "compact"; bytes: Uint8Array };

/** Whether a jsfive dtype is a string type handled by {@link StringArrayReader}. */
export function isStringDtype(dtype: unknown): boolean {
  return (
    (Array.isArray(dtype) && dtype[0] === "VLEN_STRING") ||
    (typeof dtype === "string" && /^S\d+$/.test(dtype))
  );
}

/**
 * Reads the chunks of one HDF5 string dataset as `vlen-utf8`-encoded Zarr chunks.
 * Holds only the metadata needed to locate chunks (and, for chunked layouts,
 * the chunk index once it has been read).
 */
export class StringArrayReader {
  readonly path: string;
  readonly shape: number[];
  /** Zarr chunk shape (HDF5 chunk shape, or the virtual chunk shape). */
  readonly chunkShape: number[];
  readonly #element: ElementType;
  readonly #storage: Storage;
  readonly #source: AsyncReadable;
  readonly #addressing: FileAddressing;
  readonly #heaps: GlobalHeapCache;
  #chunkIndex: Promise<Map<string, ChunkLocation>> | null = null;

  private constructor(
    path: string,
    shape: number[],
    chunkShape: number[],
    element: ElementType,
    storage: Storage,
    source: AsyncReadable,
    addressing: FileAddressing,
    heaps: GlobalHeapCache
  ) {
    this.path = path;
    this.shape = shape;
    this.chunkShape = chunkShape;
    this.#element = element;
    this.#storage = storage;
    this.#source = source;
    this.#addressing = addressing;
    this.#heaps = heaps;
  }

  /**
   * Create a reader for a string dataset, or return null if the dataset does
   * not hold strings. Only the (already loaded) object header is inspected.
   */
  static fromDataObjects(
    path: string,
    dataobjects: DataObjects,
    source: AsyncReadable,
    addressing: FileAddressing,
    heaps: GlobalHeapCache
  ): StringArrayReader | null {
    const dtype = dataobjects.dtype;
    if (!isStringDtype(dtype)) return null;
    const element: ElementType = Array.isArray(dtype)
      ? { kind: "vlen", size: 4 + addressing.offsetSize + 4 }
      : { kind: "fixed", size: parseInt((dtype as string).slice(1), 10) };
    const shape = [...(dataobjects.shape ?? [])];

    const msg = dataobjects.find_msg_type(DATA_STORAGE_MSG_TYPE)[0];
    if (!msg) throw new Error(`${path}: dataset has no data layout message`);
    const relOffset = (msg.get("offset_to_message") as number) - dataobjects.bufStart;
    const props = dataobjects._get_data_message_properties(relOffset);
    const buf = dataobjects.buf;
    const view = new DataView(buf);

    let storage: Storage;
    let chunkShape: number[];
    if (props.layout_class === 2) {
      dataobjects._get_chunk_params();
      if (dataobjects._chunks === null) {
        throw new Error(`${path}: unsupported chunk index (data layout message version ${props.version})`);
      }
      chunkShape = [...dataobjects._chunks];
      const address = dataobjects._chunk_address;
      storage = {
        kind: "chunked",
        btreeAddress:
          address === null || isUndefinedAddress(address, addressing.offsetSize) ? null : address,
        btreeDims: dataobjects._chunk_dims!,
        pipeline: dataobjects.filter_pipeline,
      };
    } else if (props.layout_class === 1) {
      const address = readUint(view, props.property_offset, addressing.offsetSize);
      storage = {
        kind: "contiguous",
        address: isUndefinedAddress(address, addressing.offsetSize) ? null : address,
      };
      chunkShape = virtualChunkShape(shape);
    } else if (props.layout_class === 0) {
      let sizeOffset: number;
      let sizeBytes: number;
      if (props.version === 1 || props.version === 2) {
        // Dimension sizes (4 bytes each), then a 4-byte compact data size.
        sizeOffset = props.property_offset + props.dims * 4;
        sizeBytes = 4;
      } else {
        sizeOffset = props.property_offset;
        sizeBytes = 2;
      }
      const size = readUint(view, sizeOffset, sizeBytes);
      const start = sizeOffset + sizeBytes;
      storage = { kind: "compact", bytes: new Uint8Array(buf.slice(start, start + size)) };
      chunkShape = virtualChunkShape(shape);
    } else {
      throw new Error(`${path}: unsupported data layout class ${props.layout_class}`);
    }

    return new StringArrayReader(path, shape, chunkShape, element, storage, source, addressing, heaps);
  }

  /**
   * The `vlen-utf8`-encoded chunk at the given chunk grid coordinates, or
   * undefined if the chunk is outside the grid or was never written
   * (in which case Zarr readers use the fill value, "").
   */
  async getChunk(coords: number[]): Promise<Uint8Array | undefined> {
    if (coords.length !== this.shape.length) return undefined;
    for (let d = 0; d < coords.length; d++) {
      const nChunks = Math.ceil(this.shape[d] / this.chunkShape[d]);
      if (!Number.isInteger(coords[d]) || coords[d] < 0 || coords[d] >= nChunks) return undefined;
    }
    const count = product(this.chunkShape);
    try {
      const raw = await this.#readChunkElements(coords, count);
      if (raw === undefined) return undefined;
      return await this.#encode(raw.bytes, raw.count, count);
    } catch (e) {
      throw new Error(`${this.path}: failed to read chunk ${coords.join("/")}: ${(e as Error).message}`, {
        cause: e,
      });
    }
  }

  /** Raw (unfiltered) element bytes of a chunk, and the number of elements they hold. */
  async #readChunkElements(
    coords: number[],
    maxCount: number
  ): Promise<{ bytes: Uint8Array; count: number } | undefined> {
    const storage = this.#storage;
    const elementSize = this.#element.size;
    if (storage.kind === "chunked") {
      const location = (await this.#getChunkIndex()).get(coords.join("/"));
      if (!location) return undefined;
      const raw = new Uint8Array(await fetchRange(this.#source, location.address, location.size));
      const bytes = await decodeChunk(raw, storage.pipeline, location.filterMask);
      return { bytes, count: Math.min(maxCount, Math.floor(bytes.byteLength / elementSize)) };
    }

    // Contiguous / compact: chunk i covers rows [i * rows, (i + 1) * rows) of the first dimension,
    // i.e. a contiguous run of elements in C order.
    const rowElements = product(this.shape.slice(1));
    const rows = this.shape.length === 0 ? 1 : this.chunkShape[0];
    const firstRow = this.shape.length === 0 ? 0 : coords[0] * rows;
    const totalRows = this.shape.length === 0 ? 1 : this.shape[0];
    const count = Math.min(rows, totalRows - firstRow) * rowElements;
    const start = firstRow * rowElements * elementSize;
    const length = count * elementSize;
    if (storage.kind === "compact") {
      return { bytes: storage.bytes.subarray(start, start + length), count };
    }
    if (storage.address === null || length === 0) return undefined;
    const bytes = new Uint8Array(await fetchRange(this.#source, storage.address + start, length));
    return { bytes, count };
  }

  #getChunkIndex(): Promise<Map<string, ChunkLocation>> {
    if (!this.#chunkIndex) {
      this.#chunkIndex = this.#readChunkIndex();
      this.#chunkIndex.catch(() => {
        this.#chunkIndex = null;
      });
    }
    return this.#chunkIndex;
  }

  async #readChunkIndex(): Promise<Map<string, ChunkLocation>> {
    const storage = this.#storage;
    const index = new Map<string, ChunkLocation>();
    if (storage.kind !== "chunked" || storage.btreeAddress === null) return index;
    const btree = await BTreeV1RawDataChunks.create(this.#source, storage.btreeAddress, storage.btreeDims);
    const rank = this.shape.length;
    for (const node of btree.all_nodes.get(0) ?? []) {
      for (let i = 0; i < node.keys.length; i++) {
        const key = node.keys[i];
        const coords = key.chunk_offset.slice(0, rank).map((offset, d) => Math.floor(offset / this.chunkShape[d]));
        index.set(coords.join("/"), {
          address: node.addresses[i],
          size: key.chunk_size,
          filterMask: key.filter_mask,
        });
      }
    }
    return index;
  }

  /** Encode `count` stored elements (padded with "" to `total`) as a vlen-utf8 chunk. */
  async #encode(bytes: Uint8Array, count: number, total: number): Promise<Uint8Array> {
    const items: (Uint8Array | undefined)[] = new Array(count);
    const size = this.#element.size;

    if (this.#element.kind === "fixed") {
      // numpy "S" semantics: trailing NUL bytes are stripped.
      for (let i = 0; i < count; i++) {
        let end = (i + 1) * size;
        const start = i * size;
        while (end > start && bytes[end - 1] === 0) end--;
        items[i] = bytes.subarray(start, end);
      }
      return encodeVlenUtf8Bytes(items, total);
    }

    // Variable length: parse global heap IDs, then fetch the referenced collections.
    const { offsetSize } = this.#addressing;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const lengths = new Uint32Array(count);
    const addresses = new Float64Array(count);
    const indices = new Uint32Array(count);
    const unique = new Set<number>();
    for (let i = 0; i < count; i++) {
      const o = i * size;
      const length = view.getUint32(o, true);
      const address = readUint(view, o + 4, offsetSize);
      lengths[i] = length;
      addresses[i] = address;
      indices[i] = view.getUint32(o + 4 + offsetSize, true);
      if (length > 0 && address !== 0 && !isUndefinedAddress(address, offsetSize)) unique.add(address);
    }
    const collections = new Map<number, HeapCollection>();
    await Promise.all(
      Array.from(unique, async (address) => {
        collections.set(address, await this.#heaps.get(address));
      })
    );
    for (let i = 0; i < count; i++) {
      if (lengths[i] === 0) continue; // empty string
      const collection = collections.get(addresses[i]);
      if (!collection) continue; // null reference: empty string
      const object = collection.object(indices[i]);
      if (!object) {
        throw new Error(`global heap object ${indices[i]} not found in collection at ${addresses[i]}`);
      }
      let item = object.subarray(0, Math.min(lengths[i], object.byteLength));
      // Like h5py, a string ends at its first NUL byte.
      const nul = item.indexOf(0);
      if (nul !== -1) item = item.subarray(0, nul);
      items[i] = item;
    }
    return encodeVlenUtf8Bytes(items, total);
  }
}

/**
 * Chunk shape for contiguous / compact string datasets: whole trailing dimensions,
 * and enough rows of the first dimension for about VIRTUAL_CHUNK_ELEMENTS elements.
 */
function virtualChunkShape(shape: number[]): number[] {
  if (shape.length === 0) return [];
  const rowElements = product(shape.slice(1));
  const rows = Math.max(1, Math.min(shape[0], Math.floor(VIRTUAL_CHUNK_ELEMENTS / Math.max(1, rowElements))));
  return [rows, ...shape.slice(1)].map((n) => Math.max(1, n));
}

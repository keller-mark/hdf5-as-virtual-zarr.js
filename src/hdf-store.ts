import type { AsyncReadable, ReferenceEntry } from "./types.js";
import { SingleHdf5ToZarr } from "./single-hdf5-to-zarr.js";
import {
  refSpecToConsolidatedMetadataV3,
  type ZarrV3ConsolidatedMetadata,
} from "./ref-spec-to-consolidated-metadata.js";
import { refSpecV2ToV3 } from "./ref-spec-v2-to-v3.js";
import { to_binary } from "./ref-store.js";
import { File as Hdf5File, Group as Hdf5Group, Dataset as Hdf5Dataset } from "./jsfive/high-level.js";
import {
  StringArrayReader,
  GlobalHeapCache,
  DEFAULT_GLOBAL_HEAP_CACHE_BYTES,
  type FileAddressing,
} from "./string-chunks.js";
import { jsonStringifySorted } from "./encoding.js";
import type { AbsolutePath, GetOptions } from "@zarrita/storage";
import QuickLRU from "quick-lru";

type Hdf5Node = Hdf5Group | Hdf5Dataset;
type Refs = Record<string, ReferenceEntry>;

export interface HdfStoreOptions {
  /** Maximum number of resolved HDF5 groups/datasets kept in memory (default 1000). */
  maxCachedNodes?: number;
  /** Maximum number of translated nodes (metadata or chunk references) kept in memory (default 500). */
  maxCachedRefs?: number;
  /**
   * Maximum total size in bytes of the parsed HDF5 global heap collections
   * (variable-length string payloads) kept in memory (default 32 MiB).
   */
  maxGlobalHeapCacheBytes?: number;
}

/**
 * Remove a rejected promise from an LRU cache (unless it has since been replaced),
 * so that a failed read can be retried.
 */
function evictOnReject<V>(cache: QuickLRU<string, Promise<V>>, key: string, promise: Promise<V>) {
  promise.catch(() => {
    if (cache.peek(key) === promise) cache.delete(key);
  });
}

/**
 * Exposes an HDF5 file as a (read-only) Zarr v3 hierarchy.
 *
 * Translation is lazy: opening the store only reads the superblock and the root
 * group. Each group / array is translated the first time one of its keys is
 * requested, and chunk references (B-tree walks) are only computed when a
 * chunk of that array is requested.
 *
 * String datasets (variable-length and fixed-length) are exposed as Zarr v3
 * arrays with `data_type: "string"` and the `vlen-utf8` codec. Their chunks are
 * built on demand: `get("<path>/c/<i>")` reads only HDF5 chunk `i` (or, for
 * contiguous / compact layouts, a virtual chunk of up to 65,536 elements) and
 * the global heap collections it references. Encoded chunks are not cached.
 *
 * The source is assumed to be wrapped with zarrita's `withRangeCoalescing`:
 * independent reads (sibling nodes, B-tree children, heap collections, chunks)
 * are issued concurrently so that the coalescer can merge them into few
 * requests. The store keeps no byte cache of its own.
 */
export class HdfStore implements AsyncReadable {
  #source: AsyncReadable;
  #converter: SingleHdf5ToZarr;
  /** The root group; kept outside the LRU caches so it is never evicted. */
  #root: Promise<Hdf5Node>;
  #nodes: QuickLRU<string, Promise<Hdf5Node | null>>;
  #nodeRefs: QuickLRU<string, Promise<Refs>>;
  /** Readers for string datasets (metadata and chunk index only, no string data). */
  #stringReaders: QuickLRU<string, Promise<StringArrayReader | null>>;
  #heaps: GlobalHeapCache;
  #addressing: FileAddressing;
  #consolidated: Promise<ZarrV3ConsolidatedMetadata> | null = null;

  private constructor(
    source: AsyncReadable,
    converter: SingleHdf5ToZarr,
    file: Hdf5File,
    options: HdfStoreOptions
  ) {
    this.#source = source;
    this.#converter = converter;
    this.#root = Promise.resolve(file);
    this.#nodes = new QuickLRU({ maxSize: options.maxCachedNodes ?? 1000 });
    this.#nodeRefs = new QuickLRU({ maxSize: options.maxCachedRefs ?? 500 });
    this.#stringReaders = new QuickLRU({ maxSize: options.maxCachedRefs ?? 500 });
    this.#addressing = { offsetSize: file.offset_size, lengthSize: file.length_size };
    this.#heaps = new GlobalHeapCache(
      source,
      file.length_size,
      options.maxGlobalHeapCacheBytes ?? DEFAULT_GLOBAL_HEAP_CACHE_BYTES
    );
  }

  /**
   * @param {AsyncReadable} hdfSource Assumed to be a zarrita Store,
   * with the HDF5 file located at the root key "/". In other words,
   * hdfSource.getRange("/", { offset, length }) should return the
   * corresponding byte range from the HDF5 file.
   * @returns {Promise<HdfStore>} An HdfStore instance which exposes the HDF5
   * file as a Zarr v3 hierarchy.
   * @param {HdfStoreOptions} options Sizes of the in-memory LRU caches.
   */
  static async fromStore(hdfSource: AsyncReadable, options: HdfStoreOptions = {}): Promise<HdfStore> {
    const converter = new SingleHdf5ToZarr(hdfSource, { url: null, zarrFormat: 2 });
    const file = await converter.open();
    return new HdfStore(hdfSource, converter, file, options);
  }

  async get(key: AbsolutePath, opts?: GetOptions): Promise<Uint8Array | undefined> {
    const path = key.replace(/^\/+/, "");
    const parts = path === "" ? [] : path.split("/");

    if (parts[parts.length - 1] === "zarr.json") {
      const nodePath = parts.slice(0, -1).join("/");
      const refs = await this.#getNodeRefs(nodePath, false);
      return this.#resolveRef(refs?.[path], opts);
    }

    // Chunk key: walk down the hierarchy until reaching an array;
    // the remaining segments are the chunk key (e.g. "c/0/1").
    let nodePath = "";
    for (const part of parts) {
      const node = await this.#resolveNode(nodePath);
      if (!node) return undefined;
      if (node instanceof Hdf5Dataset) break;
      nodePath = nodePath ? `${nodePath}/${part}` : part;
    }
    // v3 chunk keys use the default encoding: "c" (scalars) or "c/i/j/...".
    // Anything else (e.g. v2 probes like ".zarray") cannot be a chunk.
    const chunkKey = path.slice(nodePath.length + 1);
    if (
      nodePath === path ||
      !(chunkKey === "c" || chunkKey.startsWith("c/")) ||
      !((await this.#resolveNode(nodePath)) instanceof Hdf5Dataset)
    ) {
      return undefined;
    }
    const reader = await this.#getStringReader(nodePath);
    if (reader) {
      const coords = chunkKey === "c" ? [] : chunkKey.slice(2).split("/").map(Number);
      return reader.getChunk(coords);
    }
    const refs = await this.#getNodeRefs(nodePath, true);
    return this.#resolveRef(refs?.[path], opts);
  }

  /**
   * @returns The root zarr.json document with inline consolidated metadata
   * (zarr-python format) describing every group and array in the hierarchy.
   * Reads the metadata of every node in the file (but no chunk data); nodes
   * already opened through this store are not read again.
   */
  getConsolidatedMetadata(): Promise<ZarrV3ConsolidatedMetadata> {
    if (!this.#consolidated) {
      this.#consolidated = this.#collectMetadata("").then((parts) =>
        refSpecToConsolidatedMetadataV3({ version: 1, refs: Object.assign({}, ...parts) })
      );
      this.#consolidated.catch(() => {
        this.#consolidated = null;
      });
    }
    return this.#consolidated;
  }

  /**
   * Metadata refs for a node and all of its descendants, in depth-first order.
   * Sibling subtrees are visited concurrently.
   */
  async #collectMetadata(path: string): Promise<Refs[]> {
    const [node, own] = await Promise.all([
      this.#resolveNode(path),
      this.#getNodeRefs(path, false),
    ]);
    if (!(node instanceof Hdf5Group)) return [own];
    const children = await Promise.all(
      node.keys.map((key) => this.#collectMetadata(path ? `${path}/${key}` : key))
    );
    return [own, ...children.flat()];
  }

  /**
   * Resolve the HDF5 group or dataset at a hierarchy path (cached).
   * Returns null if there is no such node.
   */
  #resolveNode(path: string): Promise<Hdf5Node | null> {
    if (path === "") return this.#root;
    let node = this.#nodes.get(path);
    if (!node) {
      const idx = path.lastIndexOf("/");
      const parentPath = idx === -1 ? "" : path.slice(0, idx);
      const name = path.slice(idx + 1);
      node = this.#resolveNode(parentPath).then(async (parent) => {
        if (!(parent instanceof Hdf5Group) || !(name in parent._links)) return null;
        try {
          return await parent.get(name);
        } catch {
          return null;
        }
      });
      evictOnReject(this.#nodes, path, node);
      this.#nodes.set(path, node);
    }
    return node;
  }

  /**
   * Zarr v3 references for a single node (cached). Metadata-only refs contain
   * just the node's zarr.json; with `chunks`, its chunk references too.
   */
  #getNodeRefs(path: string, chunks: boolean): Promise<Refs> {
    const cacheKey = `${chunks ? "c" : "m"}:${path}`;
    let refs = this.#nodeRefs.get(cacheKey);
    if (!refs) {
      refs = (async () => {
        const node = await this.#resolveNode(path);
        if (!node) return {};
        const reader = await this.#getStringReader(path);
        // String chunks are computed on demand (see #getStringReader), never stored as refs.
        const v2Refs = await this.#converter.translateNode(path, node, { chunks: chunks && !reader });
        const v3Refs = refSpecV2ToV3({ version: 1, refs: v2Refs }).refs;
        const metaKey = path ? `${path}/zarr.json` : "zarr.json";
        if (reader && typeof v3Refs[metaKey] === "string") {
          const meta = JSON.parse(v3Refs[metaKey] as string);
          meta.chunk_grid = { name: "regular", configuration: { chunk_shape: reader.chunkShape } };
          v3Refs[metaKey] = jsonStringifySorted(meta);
        }
        return v3Refs;
      })();
      evictOnReject(this.#nodeRefs, cacheKey, refs);
      this.#nodeRefs.set(cacheKey, refs);
    }
    return refs;
  }

  /**
   * The on-demand chunk reader for a string dataset (cached), or null if the
   * node at `path` is not a string dataset.
   */
  #getStringReader(path: string): Promise<StringArrayReader | null> {
    let reader = this.#stringReaders.get(path);
    if (!reader) {
      reader = this.#resolveNode(path).then((node) => {
        if (!(node instanceof Hdf5Dataset)) return null;
        try {
          return StringArrayReader.fromDataObjects(
            path,
            node._dataobjects,
            this.#source,
            this.#addressing,
            this.#heaps
          );
        } catch {
          // Unsupported layout (e.g. a v4 chunk index): fall back to inlining
          // the whole dataset as a single chunk.
          return null;
        }
      });
      evictOnReject(this.#stringReaders, path, reader);
      this.#stringReaders.set(path, reader);
    }
    return reader;
  }

  async #resolveRef(ref: ReferenceEntry | undefined, opts?: GetOptions): Promise<Uint8Array | undefined> {
    if (ref === undefined) return undefined;
    if (typeof ref === "string") {
      return ref.startsWith("base64:")
        ? to_binary(ref.slice("base64:".length))
        : new TextEncoder().encode(ref);
    }
    const [, offset, length] = ref;
    return this.#source.getRange!("/", { offset, length }, opts);
  }
}

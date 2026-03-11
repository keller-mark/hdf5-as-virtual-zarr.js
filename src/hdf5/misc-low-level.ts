/**
 * Vendored and adapted from jsfive/esm/misc-low-level.js.
 * SuperBlock, Heap, SymbolTable, GlobalHeap, FractalHeap.
 *
 * All classes use async factory methods (static async create())
 * to read byte ranges via Source.fetch() instead of synchronous
 * slicing of a monolithic ArrayBuffer.
 */

import type { Source } from "../index.js";
import {
  _structure_size,
  _padded_size,
  _unpack_struct_from,
  struct,
  assert,
  _unpack_integer,
  bitSize,
} from "./core.js";

// ────────── format constants ──────────

const FORMAT_SIGNATURE = struct.unpack_from(
  "8s",
  new Uint8Array([137, 72, 68, 70, 13, 10, 26, 10]).buffer
)[0];

export const UNDEFINED_ADDRESS = struct.unpack_from(
  "<Q",
  new Uint8Array([255, 255, 255, 255, 255, 255, 255, 255]).buffer
)[0];

// Version 0 SUPERBLOCK
const SUPERBLOCK_V0 = new Map<string, string>([
  ["format_signature", "8s"],
  ["superblock_version", "B"],
  ["free_storage_version", "B"],
  ["root_group_version", "B"],
  ["reserved_0", "B"],
  ["shared_header_version", "B"],
  ["offset_size", "B"],
  ["length_size", "B"],
  ["reserved_1", "B"],
  ["group_leaf_node_k", "H"],
  ["group_internal_node_k", "H"],
  ["file_consistency_flags", "L"],
  ["base_address_lower", "Q"],
  ["free_space_address", "Q"],
  ["end_of_file_address", "Q"],
  ["driver_information_address", "Q"],
]);
const SUPERBLOCK_V0_SIZE = _structure_size(SUPERBLOCK_V0);

const SUPERBLOCK_V2_V3 = new Map<string, string>([
  ["format_signature", "8s"],
  ["superblock_version", "B"],
  ["offset_size", "B"],
  ["length_size", "B"],
  ["file_consistency_flags", "B"],
  ["base_address", "Q"],
  ["superblock_extension_address", "Q"],
  ["end_of_file_address", "Q"],
  ["root_group_address", "Q"],
  ["superblock_checksum", "I"],
]);
const SUPERBLOCK_V2_V3_SIZE = _structure_size(SUPERBLOCK_V2_V3);

const SYMBOL_TABLE_ENTRY = new Map<string, string>([
  ["link_name_offset", "Q"],
  ["object_header_address", "Q"],
  ["cache_type", "I"],
  ["reserved", "I"],
  ["scratch", "16s"],
]);
const SYMBOL_TABLE_ENTRY_SIZE = _structure_size(SYMBOL_TABLE_ENTRY);

const SYMBOL_TABLE_NODE = new Map<string, string>([
  ["signature", "4s"],
  ["version", "B"],
  ["reserved_0", "B"],
  ["symbols", "H"],
]);
const SYMBOL_TABLE_NODE_SIZE = _structure_size(SYMBOL_TABLE_NODE);

const LOCAL_HEAP = new Map<string, string>([
  ["signature", "4s"],
  ["version", "B"],
  ["reserved", "3s"],
  ["data_segment_size", "Q"],
  ["offset_to_free_list", "Q"],
  ["address_of_data_segment", "Q"],
]);

const GLOBAL_HEAP_HEADER = new Map<string, string>([
  ["signature", "4s"],
  ["version", "B"],
  ["reserved", "3s"],
  ["collection_size", "Q"],
]);
const GLOBAL_HEAP_HEADER_SIZE = _structure_size(GLOBAL_HEAP_HEADER);

const GLOBAL_HEAP_OBJECT = new Map<string, string>([
  ["object_index", "H"],
  ["reference_count", "H"],
  ["reserved", "I"],
  ["object_size", "Q"],
]);
const GLOBAL_HEAP_OBJECT_SIZE = _structure_size(GLOBAL_HEAP_OBJECT);

const FRACTAL_HEAP_HEADER = new Map<string, string>([
  ["signature", "4s"],
  ["version", "B"],
  ["object_index_size", "H"],
  ["filter_info_size", "H"],
  ["flags", "B"],
  ["max_managed_object_size", "I"],
  ["next_huge_object_index", "Q"],
  ["btree_address_huge_objects", "Q"],
  ["managed_freespace_size", "Q"],
  ["freespace_manager_address", "Q"],
  ["managed_space_size", "Q"],
  ["managed_alloc_size", "Q"],
  ["next_directblock_iterator_address", "Q"],
  ["managed_object_count", "Q"],
  ["huge_objects_total_size", "Q"],
  ["huge_object_count", "Q"],
  ["tiny_objects_total_size", "Q"],
  ["tiny_object_count", "Q"],
  ["table_width", "H"],
  ["starting_block_size", "Q"],
  ["maximum_direct_block_size", "Q"],
  ["log2_maximum_heap_size", "H"],
  ["indirect_starting_rows_count", "H"],
  ["root_block_address", "Q"],
  ["indirect_current_rows_count", "H"],
]);

// ────────── SuperBlock ──────────

export class SuperBlock {
  version: number;
  _contents: Map<string, any>;
  _end_of_sblock: number;
  _root_symbol_table: SymbolTable | null = null;

  private _source: Source;

  private constructor(
    source: Source,
    contents: Map<string, any>,
    version: number,
    end_of_sblock: number
  ) {
    this._source = source;
    this._contents = contents;
    this.version = version;
    this._end_of_sblock = end_of_sblock;
  }

  static async create(source: Source, offset: number): Promise<SuperBlock> {
    // Read enough bytes for the larger superblock variant
    const needed = Math.max(SUPERBLOCK_V0_SIZE, SUPERBLOCK_V2_V3_SIZE);
    const buf = await source.fetch(offset, needed);

    const version_hint = struct.unpack_from("<B", buf, 8)[0];
    let contents: Map<string, any>;
    let end_of_sblock: number;

    if (version_hint === 0) {
      contents = _unpack_struct_from(SUPERBLOCK_V0, buf, 0);
      end_of_sblock = offset + SUPERBLOCK_V0_SIZE;
    } else if (version_hint === 2 || version_hint === 3) {
      contents = _unpack_struct_from(SUPERBLOCK_V2_V3, buf, 0);
      end_of_sblock = offset + SUPERBLOCK_V2_V3_SIZE;
    } else {
      throw new Error("unsupported superblock version: " + version_hint);
    }

    if (contents.get("format_signature") !== FORMAT_SIGNATURE) {
      throw new Error("Incorrect file signature: " + contents.get("format_signature"));
    }
    if (contents.get("offset_size") !== 8 || contents.get("length_size") !== 8) {
      throw new Error("File uses non-64-bit addressing");
    }

    const version = contents.get("superblock_version");
    return new SuperBlock(source, contents, version, end_of_sblock);
  }

  async getOffsetToDataobjects(): Promise<number> {
    if (this.version === 0) {
      const sym_table = await SymbolTable.create(
        this._source,
        this._end_of_sblock,
        true
      );
      this._root_symbol_table = sym_table;
      return sym_table.group_offset!;
    } else if (this.version === 2 || this.version === 3) {
      return this._contents.get("root_group_address");
    } else {
      throw new Error("Not implemented version = " + this.version);
    }
  }
}

// ────────── Heap (local) ──────────

export class Heap {
  data: ArrayBuffer;
  private _contents: Map<string, any>;

  private constructor(contents: Map<string, any>, data: ArrayBuffer) {
    this._contents = contents;
    this.data = data;
  }

  static async create(source: Source, offset: number): Promise<Heap> {
    const headerSize = _structure_size(LOCAL_HEAP);
    const hBuf = await source.fetch(offset, headerSize);
    const local_heap = _unpack_struct_from(LOCAL_HEAP, hBuf, 0);
    assert(local_heap.get("signature") === "HEAP");
    assert(local_heap.get("version") === 0);

    const data_offset = local_heap.get("address_of_data_segment");
    const seg_size = local_heap.get("data_segment_size");
    const heap_data = await source.fetch(data_offset, seg_size);
    local_heap.set("heap_data", heap_data);

    return new Heap(local_heap, heap_data);
  }

  get_object_name(offset: number): string {
    const end = new Uint8Array(this.data).indexOf(0, offset);
    const name_size = end - offset;
    return struct.unpack_from("<" + name_size.toFixed() + "s", this.data, offset)[0];
  }
}

// ────────── SymbolTable ──────────

export class SymbolTable {
  entries: Map<string, any>[];
  group_offset?: number;
  private _contents: Map<string, any>;

  private constructor(
    contents: Map<string, any>,
    entries: Map<string, any>[],
    group_offset?: number
  ) {
    this._contents = contents;
    this.entries = entries;
    this.group_offset = group_offset;
  }

  static async create(
    source: Source,
    offset: number,
    root = false
  ): Promise<SymbolTable> {
    let node: Map<string, any>;
    let readOffset = offset;

    if (root) {
      node = new Map([["symbols", 1]]);
    } else {
      const nodeBuf = await source.fetch(offset, SYMBOL_TABLE_NODE_SIZE);
      node = _unpack_struct_from(SYMBOL_TABLE_NODE, nodeBuf, 0);
      if (node.get("signature") !== "SNOD") throw new Error("incorrect node type");
      readOffset = offset + SYMBOL_TABLE_NODE_SIZE;
    }

    const n_symbols = node.get("symbols");
    const entryBytes = n_symbols * SYMBOL_TABLE_ENTRY_SIZE;
    const entryBuf = await source.fetch(readOffset, entryBytes);

    const entries: Map<string, any>[] = [];
    let localOff = 0;
    for (let i = 0; i < n_symbols; i++) {
      entries.push(_unpack_struct_from(SYMBOL_TABLE_ENTRY, entryBuf, localOff));
      localOff += SYMBOL_TABLE_ENTRY_SIZE;
    }

    let group_offset: number | undefined;
    if (root) {
      group_offset = entries[0].get("object_header_address");
    }

    return new SymbolTable(node, entries, group_offset);
  }

  assign_name(heap: Heap): void {
    for (const entry of this.entries) {
      const offset = entry.get("link_name_offset");
      entry.set("link_name", heap.get_object_name(offset));
    }
  }

  get_links(heap: Heap): Record<string, any> {
    const links: Record<string, any> = {};
    for (const e of this.entries) {
      const cache_type = e.get("cache_type");
      const link_name = e.get("link_name");
      if (cache_type === 0 || cache_type === 1) {
        links[link_name] = e.get("object_header_address");
      } else if (cache_type === 2) {
        const scratch: string = e.get("scratch");
        const buf = new ArrayBuffer(4);
        const bufView = new Uint8Array(buf);
        for (let i = 0; i < 4; i++) {
          bufView[i] = scratch.charCodeAt(i);
        }
        const off = struct.unpack_from("<I", buf, 0)[0];
        links[link_name] = heap.get_object_name(off);
      }
    }
    return links;
  }
}

// ────────── GlobalHeap ──────────

export class GlobalHeap {
  heap_data: ArrayBuffer;
  private _header: Map<string, any>;
  private _objects: Map<number, ArrayBuffer> | null = null;

  private constructor(header: Map<string, any>, heap_data: ArrayBuffer) {
    this._header = header;
    this.heap_data = heap_data;
  }

  static async create(source: Source, offset: number): Promise<GlobalHeap> {
    const hBuf = await source.fetch(offset, GLOBAL_HEAP_HEADER_SIZE);
    const header = _unpack_struct_from(GLOBAL_HEAP_HEADER, hBuf, 0);

    const heap_data_size = header.get("collection_size") - GLOBAL_HEAP_HEADER_SIZE;
    const heap_data = await source.fetch(
      offset + GLOBAL_HEAP_HEADER_SIZE,
      heap_data_size
    );
    return new GlobalHeap(header, heap_data);
  }

  get objects(): Map<number, ArrayBuffer> {
    if (this._objects === null) {
      this._objects = new Map();
      let offset = 0;
      while (offset <= this.heap_data.byteLength - GLOBAL_HEAP_OBJECT_SIZE) {
        const info = _unpack_struct_from(GLOBAL_HEAP_OBJECT, this.heap_data, offset);
        if (info.get("object_index") === 0) break;
        offset += GLOBAL_HEAP_OBJECT_SIZE;
        const obj_data = this.heap_data.slice(offset, offset + info.get("object_size"));
        this._objects.set(info.get("object_index"), obj_data);
        offset += _padded_size(info.get("object_size"));
      }
    }
    return this._objects;
  }
}

// ────────── FractalHeap ──────────

export class FractalHeap {
  header: Map<string, any>;
  nobjects: number;
  managed: ArrayBuffer;

  private _managed_object_offset_size: number;
  private _managed_object_length_size: number;
  private _max_direct_nrows: number;
  private _indirect_nrows_sub: number;

  private direct_block_header: Map<string, string>;
  private direct_block_header_size: number;
  private indirect_block_header: Map<string, string>;
  private indirect_block_header_size: number;

  private constructor(
    header: Map<string, any>,
    managed: ArrayBuffer,
    direct_block_header: Map<string, string>,
    direct_block_header_size: number,
    indirect_block_header: Map<string, string>,
    indirect_block_header_size: number,
    managed_object_offset_size: number,
    managed_object_length_size: number,
    max_direct_nrows: number,
    indirect_nrows_sub: number
  ) {
    this.header = header;
    this.nobjects =
      header.get("managed_object_count") +
      header.get("huge_object_count") +
      header.get("tiny_object_count");
    this.managed = managed;
    this.direct_block_header = direct_block_header;
    this.direct_block_header_size = direct_block_header_size;
    this.indirect_block_header = indirect_block_header;
    this.indirect_block_header_size = indirect_block_header_size;
    this._managed_object_offset_size = managed_object_offset_size;
    this._managed_object_length_size = managed_object_length_size;
    this._max_direct_nrows = max_direct_nrows;
    this._indirect_nrows_sub = indirect_nrows_sub;
  }

  static async create(source: Source, offset: number): Promise<FractalHeap> {
    const headerSize = _structure_size(FRACTAL_HEAP_HEADER);
    const hBuf = await source.fetch(offset, headerSize);
    const header = _unpack_struct_from(FRACTAL_HEAP_HEADER, hBuf, 0);
    assert(header.get("signature") === "FRHP");
    assert(header.get("version") === 0);

    if (header.get("filter_info_size") > 0) {
      throw new Error("Filter info size not supported on FractalHeap");
    }
    if (header.get("btree_address_huge_objects") === UNDEFINED_ADDRESS) {
      header.set("btree_address_huge_objects", null);
    } else {
      throw new Error("Huge objects not implemented in FractalHeap");
    }
    if (header.get("root_block_address") === UNDEFINED_ADDRESS) {
      header.set("root_block_address", null);
    }

    const nbits = header.get("log2_maximum_heap_size");
    const block_offset_size = FractalHeap._min_size_nbits(nbits);

    const h = new Map<string, string>([
      ["signature", "4s"],
      ["version", "B"],
      ["heap_header_adddress", "Q"],
      ["block_offset", `${block_offset_size}B`],
    ]);
    const indirect_block_header = new Map(h);
    const indirect_block_header_size = _structure_size(h);
    if ((header.get("flags") & 2) === 2) {
      h.set("checksum", "I");
    }
    const direct_block_header = h;
    const direct_block_header_size = _structure_size(h);

    const maximum_dblock_size = header.get("maximum_direct_block_size");
    const managed_object_offset_size = FractalHeap._min_size_nbits(nbits);
    const value = Math.min(maximum_dblock_size, header.get("max_managed_object_size"));
    const managed_object_length_size = FractalHeap._min_size_integer(value);

    const start_block_size = header.get("starting_block_size");
    const table_width = header.get("table_width");
    if (!(start_block_size > 0)) {
      throw new Error("Starting block size == 0 not implemented");
    }

    const log2_maximum_dblock_size = Math.floor(Math.log2(maximum_dblock_size));
    assert(1n << BigInt(log2_maximum_dblock_size) === BigInt(maximum_dblock_size));

    const log2_start_block_size = Math.floor(Math.log2(start_block_size));
    assert(1n << BigInt(log2_start_block_size) === BigInt(start_block_size));

    const max_direct_nrows = log2_maximum_dblock_size - log2_start_block_size + 2;

    const log2_table_width = Math.floor(Math.log2(table_width));
    assert(1 << log2_table_width === table_width);
    const indirect_nrows_sub = log2_table_width + log2_start_block_size - 1;

    // Read managed data
    const managedParts: ArrayBuffer[] = [];
    const root_address = header.get("root_block_address");
    let nrows = 0;
    if (root_address != null) {
      nrows = header.get("indirect_current_rows_count");
    }

    // Helper context object for recursive reading
    const ctx = {
      source,
      header,
      direct_block_header,
      direct_block_header_size,
      indirect_block_header,
      indirect_block_header_size,
      max_direct_nrows,
      indirect_nrows_sub,
      table_width,
    };

    if (nrows > 0) {
      for await (const data of FractalHeap._iter_indirect_block(ctx, root_address, nrows)) {
        managedParts.push(data);
      }
    } else if (root_address != null) {
      const data = await FractalHeap._read_direct_block(ctx, root_address, start_block_size);
      managedParts.push(data);
    }

    const data_size = managedParts.reduce((p, c) => p + c.byteLength, 0);
    const combined = new Uint8Array(data_size);
    let moffset = 0;
    for (const m of managedParts) {
      combined.set(new Uint8Array(m), moffset);
      moffset += m.byteLength;
    }

    return new FractalHeap(
      header,
      combined.buffer,
      direct_block_header,
      direct_block_header_size,
      indirect_block_header,
      indirect_block_header_size,
      managed_object_offset_size,
      managed_object_length_size,
      max_direct_nrows,
      indirect_nrows_sub
    );
  }

  get_data(heapid: ArrayBuffer): ArrayBuffer {
    const firstbyte = struct.unpack_from("<B", heapid, 0)[0];
    const idtype = (firstbyte >> 4) & 3;
    const version = firstbyte >> 6;

    let data_offset = 1;
    if (idtype === 0) {
      // managed
      assert(version === 0);
      let nbytes = this._managed_object_offset_size;
      const offset = _unpack_integer(nbytes, heapid, data_offset);
      data_offset += nbytes;

      nbytes = this._managed_object_length_size;
      const size = _unpack_integer(nbytes, heapid, data_offset);

      return this.managed.slice(offset, offset + size);
    } else if (idtype === 1) {
      throw new Error("tiny objectID not supported in FractalHeap");
    } else if (idtype === 2) {
      throw new Error("huge objectID not supported in FractalHeap");
    } else {
      throw new Error("unknown objectID type in FractalHeap");
    }
  }

  private static _min_size_integer(integer: number): number {
    return FractalHeap._min_size_nbits(bitSize(integer));
  }

  private static _min_size_nbits(nbits: number): number {
    return Math.ceil(nbits / 8);
  }

  private static async _read_direct_block(
    ctx: any,
    offset: number,
    block_size: number
  ): Promise<ArrayBuffer> {
    const data = await ctx.source.fetch(offset, block_size);
    const header = _unpack_struct_from(ctx.direct_block_header, data, 0);
    assert(header.get("signature") === "FHDB");
    return data;
  }

  private static async *_iter_indirect_block(
    ctx: any,
    offset: number,
    nrows: number
  ): AsyncGenerator<ArrayBuffer> {
    const ibHdrBuf = await ctx.source.fetch(offset, ctx.indirect_block_header_size);
    const header = _unpack_struct_from(ctx.indirect_block_header, ibHdrBuf, 0);
    let readOff = offset + ctx.indirect_block_header_size;
    assert(header.get("signature") === "FHIB");

    const block_offset_bytes = header.get("block_offset");
    const block_offset = (block_offset_bytes as number[]).reduce(
      (p: number, c: number, i: number) => p + (c << (i * 8)),
      0
    );
    header.set("block_offset", block_offset);

    const [ndirect, nindirect] = FractalHeap._indirect_info(ctx, nrows);

    // Read addresses for direct blocks
    const addrBytes = (ndirect + nindirect) * 8;
    const addrBuf = await ctx.source.fetch(readOff, addrBytes);
    let addrOff = 0;

    const direct_blocks: [number, number][] = [];
    for (let i = 0; i < ndirect; i++) {
      const address = struct.unpack_from("<Q", addrBuf, addrOff)[0];
      addrOff += 8;
      if (address === UNDEFINED_ADDRESS) break;
      const block_size = FractalHeap._calc_block_size(ctx, i);
      direct_blocks.push([address, block_size]);
    }

    const indirect_blocks: [number, number][] = [];
    for (let i = ndirect; i < ndirect + nindirect; i++) {
      const address = struct.unpack_from("<Q", addrBuf, addrOff)[0];
      addrOff += 8;
      if (address === UNDEFINED_ADDRESS) break;
      const block_size = FractalHeap._calc_block_size(ctx, i);
      const nr = FractalHeap._iblock_nrows_from_block_size(ctx, block_size);
      indirect_blocks.push([address, nr]);
    }

    for (const [address, block_size] of direct_blocks) {
      yield await FractalHeap._read_direct_block(ctx, address, block_size);
    }
    for (const [address, nr] of indirect_blocks) {
      yield* FractalHeap._iter_indirect_block(ctx, address, nr);
    }
  }

  private static _calc_block_size(ctx: any, iblock: number): number {
    const row = Math.floor(iblock / ctx.table_width);
    return 2 ** Math.max(row - 1, 0) * ctx.header.get("starting_block_size");
  }

  private static _iblock_nrows_from_block_size(ctx: any, block_size: number): number {
    const log2_block_size = Math.floor(Math.log2(block_size));
    assert(2 ** log2_block_size === block_size);
    return log2_block_size - ctx.indirect_nrows_sub;
  }

  private static _indirect_info(ctx: any, nrows: number): [number, number] {
    const table_width = ctx.table_width;
    const nobjects = nrows * table_width;
    const ndirect_max = ctx.max_direct_nrows * table_width;
    let ndirect: number, nindirect: number;
    if (nrows <= ndirect_max) {
      ndirect = nobjects;
      nindirect = 0;
    } else {
      ndirect = ndirect_max;
      nindirect = nobjects - ndirect_max;
    }
    return [ndirect, nindirect];
  }
}

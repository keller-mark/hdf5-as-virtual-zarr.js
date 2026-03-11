/**
 * Vendored and adapted from jsfive/esm/dataobjects.js.
 * DataObjects - the HDF5 data object header parser.
 *
 * Key differences from jsfive:
 * - Uses async factory method DataObjects.create(source, offset)
 * - Partial reads via Source.fetch() instead of full file buffer
 * - get_links(), get_attributes(), get_data() are async
 * - buf/bufStart design: buf covers all message block ranges,
 *   bufStart = absolute offset where buf starts (= object header offset)
 */

import type { Source } from "../index.js";
import { DatatypeMessage } from "./datatype-msg.js";
import {
  _structure_size,
  _padded_size,
  _unpack_struct_from,
  struct,
  dtype_getter,
  DataView64,
  assert,
} from "./core.js";
import {
  BTreeV1Groups,
  BTreeV1RawDataChunks,
  BTreeV2GroupNames,
  BTreeV2GroupOrders,
} from "./btree.js";
import {
  Heap,
  SymbolTable,
  GlobalHeap,
  FractalHeap,
  UNDEFINED_ADDRESS,
} from "./misc-low-level.js";

// ────────── Message type constants ──────────

const DATASPACE_MSG_TYPE = 0x0001;
const LINK_INFO_MSG_TYPE = 0x0002;
const DATATYPE_MSG_TYPE = 0x0003;
const FILLVALUE_MSG_TYPE = 0x0005;
const LINK_MSG_TYPE = 0x0006;
const DATA_STORAGE_MSG_TYPE = 0x0008;
const DATA_STORAGE_FILTER_PIPELINE_MSG_TYPE = 0x000b;
const ATTRIBUTE_MSG_TYPE = 0x000c;
const OBJECT_CONTINUATION_MSG_TYPE = 0x0010;
const SYMBOL_TABLE_MSG_TYPE = 0x0011;
const ATTRIBUTE_INFO_MSG_TYPE = 0x0015;

// ────────── Structures ──────────

const OBJECT_HEADER_V1 = new Map<string, string>([
  ["version", "B"],
  ["reserved", "B"],
  ["total_header_messages", "H"],
  ["object_reference_count", "I"],
  ["object_header_size", "I"],
  ["padding", "I"],
]);
const OBJECT_HEADER_V1_SIZE = _structure_size(OBJECT_HEADER_V1);

const OBJECT_HEADER_V2 = new Map<string, string>([
  ["signature", "4s"],
  ["version", "B"],
  ["flags", "B"],
]);
const OBJECT_HEADER_V2_SIZE = _structure_size(OBJECT_HEADER_V2);

const HEADER_MSG_INFO_V1 = new Map<string, string>([
  ["type", "H"],
  ["size", "H"],
  ["flags", "B"],
  ["reserved", "3s"],
]);
const HEADER_MSG_INFO_V1_SIZE = _structure_size(HEADER_MSG_INFO_V1);

const HEADER_MSG_INFO_V2 = new Map<string, string>([
  ["type", "B"],
  ["size", "H"],
  ["flags", "B"],
]);
const HEADER_MSG_INFO_V2_SIZE = _structure_size(HEADER_MSG_INFO_V2);

const SYMBOL_TABLE_MSG = new Map<string, string>([
  ["btree_address", "Q"],
  ["heap_address", "Q"],
]);

const ATTR_MSG_HEADER_V1 = new Map<string, string>([
  ["version", "B"],
  ["reserved", "B"],
  ["name_size", "H"],
  ["datatype_size", "H"],
  ["dataspace_size", "H"],
]);
const ATTR_MSG_HEADER_V1_SIZE = _structure_size(ATTR_MSG_HEADER_V1);

const ATTR_MSG_HEADER_V3 = new Map<string, string>([
  ["version", "B"],
  ["flags", "B"],
  ["name_size", "H"],
  ["datatype_size", "H"],
  ["dataspace_size", "H"],
  ["character_set_encoding", "B"],
]);
const ATTR_MSG_HEADER_V3_SIZE = _structure_size(ATTR_MSG_HEADER_V3);

const DATASPACE_MSG_HEADER_V1 = new Map<string, string>([
  ["version", "B"],
  ["dimensionality", "B"],
  ["flags", "B"],
  ["reserved_0", "B"],
  ["reserved_1", "I"],
]);
const DATASPACE_MSG_HEADER_V1_SIZE = _structure_size(DATASPACE_MSG_HEADER_V1);

const DATASPACE_MSG_HEADER_V2 = new Map<string, string>([
  ["version", "B"],
  ["dimensionality", "B"],
  ["flags", "B"],
  ["type", "B"],
]);
const DATASPACE_MSG_HEADER_V2_SIZE = _structure_size(DATASPACE_MSG_HEADER_V2);

const FILLVAL_MSG_V1V2 = new Map<string, string>([
  ["version", "B"],
  ["space_allocation_time", "B"],
  ["fillvalue_write_time", "B"],
  ["fillvalue_defined", "B"],
]);
const FILLVAL_MSG_V1V2_SIZE = _structure_size(FILLVAL_MSG_V1V2);

const FILLVAL_MSG_V3 = new Map<string, string>([
  ["version", "B"],
  ["flags", "B"],
]);
const FILLVAL_MSG_V3_SIZE = _structure_size(FILLVAL_MSG_V3);

const FILTER_PIPELINE_DESCR_V1 = new Map<string, string>([
  ["filter_id", "H"],
  ["name_length", "H"],
  ["flags", "H"],
  ["client_data_values", "H"],
]);
const FILTER_PIPELINE_DESCR_V1_SIZE = _structure_size(FILTER_PIPELINE_DESCR_V1);

const GLOBAL_HEAP_ID = new Map<string, string>([
  ["collection_address", "Q"],
  ["object_index", "I"],
]);
const GLOBAL_HEAP_ID_SIZE = _structure_size(GLOBAL_HEAP_ID);

const LINK_INFO_MSG1 = new Map<string, string>([
  ["heap_address", "Q"],
  ["name_btree_address", "Q"],
]);

const LINK_INFO_MSG2 = new Map<string, string>([
  ["heap_address", "Q"],
  ["name_btree_address", "Q"],
  ["order_btree_address", "Q"],
]);

// ────────── DataObjects ──────────

export class DataObjects {
  buf: ArrayBuffer;
  bufStart: number;
  msgs: Map<string, any>[];
  msg_data: ArrayBuffer;
  offset: number;
  _global_heaps: Map<number, GlobalHeap>;
  _header: Map<string, any>;
  _filter_pipeline: Map<string, unknown>[] | null;
  _chunk_params_set: boolean;
  _chunks: number[] | null;
  _chunk_dims: number | null;
  _chunk_address: number | null;
  private _source: Source;

  private constructor(
    source: Source,
    buf: ArrayBuffer,
    bufStart: number,
    msgs: Map<string, any>[],
    msg_data: ArrayBuffer,
    offset: number,
    header: Map<string, any>
  ) {
    this._source = source;
    this.buf = buf;
    this.bufStart = bufStart;
    this.msgs = msgs;
    this.msg_data = msg_data;
    this.offset = offset;
    this._global_heaps = new Map();
    this._header = header;
    this._filter_pipeline = null;
    this._chunk_params_set = false;
    this._chunks = null;
    this._chunk_dims = null;
    this._chunk_address = null;
  }

  static async create(source: Source, offset: number): Promise<DataObjects> {
    const peekBuf = await source.fetch(offset, 1);
    const versionHint = struct.unpack_from("<B", peekBuf, 0)[0];

    let result: {
      buf: ArrayBuffer;
      bufStart: number;
      msgs: Map<string, any>[];
      msg_data: ArrayBuffer;
      header: Map<string, any>;
    };

    if (versionHint === 1) {
      result = await DataObjects._parseV1(source, offset);
    } else if (versionHint === "O".charCodeAt(0)) {
      result = await DataObjects._parseV2(source, offset);
    } else {
      throw new Error(`Unknown Data Object Header version hint: ${versionHint}`);
    }

    return new DataObjects(
      source,
      result.buf,
      result.bufStart,
      result.msgs,
      result.msg_data,
      offset,
      result.header
    );
  }

  // ────────── V1 parsing ──────────

  private static async _parseV1(
    source: Source,
    offset: number
  ): Promise<{
    buf: ArrayBuffer;
    bufStart: number;
    msgs: Map<string, any>[];
    msg_data: ArrayBuffer;
    header: Map<string, any>;
  }> {
    const headerBuf = await source.fetch(offset, OBJECT_HEADER_V1_SIZE);
    const header = _unpack_struct_from(OBJECT_HEADER_V1, headerBuf, 0);
    assert(header.get("version") === 1);

    const total_header_messages: number = header.get("total_header_messages");
    const first_block_size: number = header.get("object_header_size");
    const first_block_offset = offset + OBJECT_HEADER_V1_SIZE;

    // Collect all blocks (including continuation blocks)
    interface Block { absOffset: number; size: number; data: ArrayBuffer }
    const allBlocks: Block[] = [];
    const blockQueue: { absOffset: number; size: number }[] = [
      { absOffset: first_block_offset, size: first_block_size },
    ];

    while (blockQueue.length > 0) {
      const { absOffset, size } = blockQueue.shift()!;
      const data = await source.fetch(absOffset, size);
      allBlocks.push({ absOffset, size, data });

      // Scan messages to find continuation blocks
      let local_offset = 0;
      while (local_offset + HEADER_MSG_INFO_V1_SIZE <= size) {
        const msg = _unpack_struct_from(HEADER_MSG_INFO_V1, data, local_offset);
        const msg_type: number = msg.get("type");
        const msg_size: number = msg.get("size");
        const offset_in_block = local_offset + HEADER_MSG_INFO_V1_SIZE;

        if (msg_type === OBJECT_CONTINUATION_MSG_TYPE) {
          const [fh_off, cont_size] = struct.unpack_from("<QQ", data, offset_in_block);
          blockQueue.push({ absOffset: fh_off, size: cont_size });
        }

        if (msg_type === 0 && msg_size === 0) break; // NIL message, stop
        local_offset += HEADER_MSG_INFO_V1_SIZE + msg_size;
      }
    }

    // Build combined sparse buffer
    // bufStart = minimum of object header offset and all block offsets
    // (continuation blocks can appear before the object header in the file)
    let bufStart = offset;
    let maxEnd = first_block_offset + first_block_size;
    for (const b of allBlocks) {
      bufStart = Math.min(bufStart, b.absOffset);
      maxEnd = Math.max(maxEnd, b.absOffset + b.size);
    }
    const combined = new Uint8Array(maxEnd - bufStart);
    combined.set(new Uint8Array(headerBuf), offset - bufStart);
    for (const b of allBlocks) {
      combined.set(new Uint8Array(b.data), b.absOffset - bufStart);
    }
    const buf = combined.buffer;

    // Parse messages (second pass over combined buffer)
    const msgs: Map<string, any>[] = new Array(total_header_messages);
    let object_header_blocks: [number, number][] = [[first_block_offset, first_block_size]];
    // Add continuation blocks discovered during first pass
    for (let bi = 1; bi < allBlocks.length; bi++) {
      object_header_blocks.push([allBlocks[bi].absOffset, allBlocks[bi].size]);
    }

    let current_block = 0;
    let block_offset = object_header_blocks[0][0];
    let block_size = object_header_blocks[0][1];
    let local_offset = 0;

    for (let i = 0; i < total_header_messages; i++) {
      if (local_offset >= block_size) {
        current_block++;
        if (current_block >= object_header_blocks.length) break;
        [block_offset, block_size] = object_header_blocks[current_block];
        local_offset = 0;
      }

      const absPos = block_offset - bufStart + local_offset;
      const msg = _unpack_struct_from(HEADER_MSG_INFO_V1, buf, absPos);
      const offset_to_message = block_offset + local_offset + HEADER_MSG_INFO_V1_SIZE;
      msg.set("offset_to_message", offset_to_message);

      // Continuation blocks add more blocks (already handled above)
      const msg_size: number = msg.get("size");
      local_offset += HEADER_MSG_INFO_V1_SIZE + msg_size;
      msgs[i] = msg;
    }

    const msg_data = buf.slice(
      first_block_offset - bufStart,
      first_block_offset - bufStart + first_block_size
    );

    return { buf, bufStart, msgs: msgs.filter(Boolean), msg_data, header };
  }

  // ────────── V2 parsing ──────────

  private static async _parseV2(
    source: Source,
    offset: number
  ): Promise<{
    buf: ArrayBuffer;
    bufStart: number;
    msgs: Map<string, any>[];
    msg_data: ArrayBuffer;
    header: Map<string, any>;
  }> {
    // Fetch enough for the full V2 header (max 32 bytes)
    const peekSize = 64;
    const peekBuf = await source.fetch(offset, peekSize);

    const { header, creation_order_size, block_offset: first_block_offset } =
      DataObjects._parseV2Header(peekBuf, 0, offset);

    const first_block_size: number = header.get("size_of_chunk_0");

    // Collect all blocks
    interface Block { absOffset: number; size: number; data: ArrayBuffer }
    const allBlocks: Block[] = [];
    const blockQueue: { absOffset: number; size: number }[] = [
      { absOffset: first_block_offset, size: first_block_size },
    ];

    while (blockQueue.length > 0) {
      const { absOffset, size } = blockQueue.shift()!;
      let data: ArrayBuffer;
      // For small blocks that fit in the peek buffer, reuse it
      if (absOffset >= offset && absOffset + size <= offset + peekSize) {
        data = peekBuf.slice(absOffset - offset, absOffset - offset + size);
      } else {
        data = await source.fetch(absOffset, size);
      }
      allBlocks.push({ absOffset, size, data });

      // Scan messages to find continuation blocks
      let local_offset = 0;
      while (local_offset + HEADER_MSG_INFO_V2_SIZE <= size) {
        const relOff = local_offset;
        const msg_type = struct.unpack_from("<B", data, relOff)[0];
        const msg_size = struct.unpack_from("<H", data, relOff + 1)[0];
        const offset_in_data = relOff + HEADER_MSG_INFO_V2_SIZE + creation_order_size;

        if (msg_type === OBJECT_CONTINUATION_MSG_TYPE) {
          const [fh_off, cont_size] = struct.unpack_from("<QQ", data, offset_in_data);
          // Skip "OFHC" 4-byte signature in v2 continuation blocks
          blockQueue.push({ absOffset: fh_off + 4, size: cont_size - 4 });
        }

        if (msg_type === 0 && msg_size === 0) break;
        local_offset += HEADER_MSG_INFO_V2_SIZE + msg_size + creation_order_size;
        if (local_offset >= size - HEADER_MSG_INFO_V2_SIZE) break;
      }
    }

    // Build combined sparse buffer
    const bufStart = offset;
    let maxEnd = first_block_offset + first_block_size;
    for (const b of allBlocks) {
      maxEnd = Math.max(maxEnd, b.absOffset + b.size);
    }
    const combined = new Uint8Array(maxEnd - bufStart);
    // Copy the peek buffer (header)
    const copySize = Math.min(peekSize, combined.length);
    combined.set(new Uint8Array(peekBuf).subarray(0, copySize), 0);
    // Copy all blocks
    for (const b of allBlocks) {
      combined.set(new Uint8Array(b.data), b.absOffset - bufStart);
    }
    const buf = combined.buffer;

    // Parse messages (second pass)
    const msgs: Map<string, any>[] = [];
    const object_header_blocks: [number, number][] = [];
    for (const b of allBlocks) {
      object_header_blocks.push([b.absOffset, b.size]);
    }

    let current_block = 0;
    let block_offset = object_header_blocks[0][0];
    let block_size = object_header_blocks[0][1];
    let local_offset = 0;

    // eslint-disable-next-line no-constant-condition
    while (true) {
      if (local_offset >= block_size - HEADER_MSG_INFO_V2_SIZE) {
        current_block++;
        if (current_block >= object_header_blocks.length) break;
        [block_offset, block_size] = object_header_blocks[current_block];
        local_offset = 0;
      }

      const absPos = block_offset - bufStart + local_offset;
      const msg = _unpack_struct_from(HEADER_MSG_INFO_V2, buf, absPos);
      const offset_to_message =
        block_offset + local_offset + HEADER_MSG_INFO_V2_SIZE + creation_order_size;
      msg.set("offset_to_message", offset_to_message);

      const msg_size: number = msg.get("size");
      local_offset += HEADER_MSG_INFO_V2_SIZE + msg_size + creation_order_size;
      msgs.push(msg);

      if (local_offset >= block_size - HEADER_MSG_INFO_V2_SIZE) {
        current_block++;
        if (current_block >= object_header_blocks.length) break;
        [block_offset, block_size] = object_header_blocks[current_block];
        local_offset = 0;
      }
    }

    const msg_data = buf.slice(
      first_block_offset - bufStart,
      first_block_offset - bufStart + first_block_size
    );

    return { buf, bufStart, msgs, msg_data, header };
  }

  private static _parseV2Header(
    buf: ArrayBuffer,
    localOffset: number,
    absBase: number
  ): { header: Map<string, any>; creation_order_size: number; block_offset: number } {
    const header = _unpack_struct_from(OBJECT_HEADER_V2, buf, localOffset);
    assert(header.get("version") === 2);
    localOffset += OBJECT_HEADER_V2_SIZE;

    const flags: number = header.get("flags");
    const creation_order_size = (flags & 0b00000100) ? 2 : 0;

    assert((flags & 0b00010000) === 0);

    if (flags & 0b00100000) {
      // access/mod/change/birth times (4 uint32s)
      localOffset += 16;
    }

    const chunk_fmt = ["<B", "<H", "<I", "<Q"][flags & 0b00000011];
    header.set("size_of_chunk_0", struct.unpack_from(chunk_fmt, buf, localOffset)[0]);
    localOffset += struct.calcsize(chunk_fmt);

    const block_offset = absBase + localOffset;
    return { header, creation_order_size, block_offset };
  }

  // ────────── Core methods ──────────

  find_msg_type(msg_type: number): Map<string, any>[] {
    return this.msgs.filter((m) => m.get("type") === msg_type);
  }

  get is_dataset(): boolean {
    return this.find_msg_type(DATASPACE_MSG_TYPE).length > 0;
  }

  get dtype(): string | string[] {
    const msg = this.find_msg_type(DATATYPE_MSG_TYPE)[0];
    const msg_offset = msg.get("offset_to_message") as number;
    const relOffset = msg_offset - this.bufStart;
    return new DatatypeMessage(this.buf, relOffset).dtype;
  }

  get shape(): number[] {
    const msg = this.find_msg_type(DATASPACE_MSG_TYPE)[0];
    const msg_offset = msg.get("offset_to_message") as number;
    const relOffset = msg_offset - this.bufStart;
    return this._determine_data_shape(this.buf, relOffset);
  }

  get chunks(): number[] | null {
    this._get_chunk_params();
    return this._chunks;
  }

  get filter_pipeline(): Map<string, unknown>[] | null {
    if (this._filter_pipeline !== null) {
      return this._filter_pipeline;
    }

    const filter_msgs = this.find_msg_type(DATA_STORAGE_FILTER_PIPELINE_MSG_TYPE);
    if (!filter_msgs.length) {
      this._filter_pipeline = null;
      return null;
    }

    const absOffset = filter_msgs[0].get("offset_to_message") as number;
    let offset = absOffset - this.bufStart;
    const buf = this.buf;

    const [version, nfilters] = struct.unpack_from("<BB", buf, offset);
    offset += struct.calcsize("<BB");

    const filters: Map<string, unknown>[] = [];

    if (version === 1) {
      offset += struct.calcsize("<HI"); // reserved fields

      for (let _ = 0; _ < nfilters; _++) {
        const filter_info = _unpack_struct_from(FILTER_PIPELINE_DESCR_V1, buf, offset);
        offset += FILTER_PIPELINE_DESCR_V1_SIZE;

        const padded_name_length = _padded_size(filter_info.get("name_length") as number, 8);
        const fmt = "<" + padded_name_length.toFixed() + "s";
        const filter_name = struct.unpack_from(fmt, buf, offset)[0];
        filter_info.set("filter_name", filter_name);
        offset += padded_name_length;

        const client_data_values: number = filter_info.get("client_data_values");
        const client_data_fmt = "<" + client_data_values.toFixed() + "I";
        const client_data = struct.unpack_from(client_data_fmt, buf, offset);
        filter_info.set("client_data", client_data);
        offset += 4 * client_data_values;

        if (client_data_values % 2) {
          offset += 4; // padding for odd number of client data values
        }

        filters.push(filter_info);
      }
    } else if (version === 2) {
      for (let nf = 0; nf < nfilters; nf++) {
        const filter_info = new Map<string, unknown>();
        const filter_id = struct.unpack_from("<H", buf, offset)[0];
        offset += 2;
        filter_info.set("filter_id", filter_id);

        let name_length = 0;
        if (filter_id > 255) {
          name_length = struct.unpack_from("<H", buf, offset)[0];
          offset += 2;
        }

        const flags_val = struct.unpack_from("<H", buf, offset)[0];
        offset += 2;
        filter_info.set("optional", (flags_val & 1) > 0);

        const num_client_values = struct.unpack_from("<H", buf, offset)[0];
        offset += 2;

        if (name_length > 0) {
          const name = struct.unpack_from(`${name_length}s`, buf, offset)[0];
          filter_info.set("name", name);
          offset += name_length;
        }

        const client_values = struct.unpack_from(`<${num_client_values}i`, buf, offset);
        offset += 4 * num_client_values;
        filter_info.set("client_data", client_values);
        filter_info.set("client_data_values", num_client_values);
        filters.push(filter_info);
      }
    } else {
      throw new Error(`Filter pipeline version ${version} not supported`);
    }

    this._filter_pipeline = filters;
    return this._filter_pipeline;
  }

  get fillvalue(): any {
    const msg = this.find_msg_type(FILLVALUE_MSG_TYPE)[0];
    if (!msg) return 0;

    let absOffset = msg.get("offset_to_message") as number;
    let offset = absOffset - this.bufStart;
    const buf = this.buf;

    const version = struct.unpack_from("<B", buf, offset)[0];
    let info: Map<string, any>;
    let is_defined: boolean;

    if (version === 1 || version === 2) {
      info = _unpack_struct_from(FILLVAL_MSG_V1V2, buf, offset);
      offset += FILLVAL_MSG_V1V2_SIZE;
      is_defined = info.get("fillvalue_defined") !== 0;
    } else if (version === 3) {
      info = _unpack_struct_from(FILLVAL_MSG_V3, buf, offset);
      offset += FILLVAL_MSG_V3_SIZE;
      is_defined = (info.get("flags") & 0b00100000) !== 0;
    } else {
      throw new Error(`Unknown fillvalue message version: ${version}`);
    }

    let size = 0;
    if (is_defined) {
      size = struct.unpack_from("<I", buf, offset)[0];
      offset += 4;
    }

    if (size) {
      const dtype = this.dtype;
      if (typeof dtype === "string") {
        const [getter, big_endian, itemSize] = dtype_getter(dtype);
        const view = new DataView64(buf, 0);
        return (view as any)[getter](offset, !big_endian, itemSize);
      }
    }
    return 0;
  }

  _get_data_message_properties(
    msg_offset: number
  ): { version: number; dims: number; layout_class: number; property_offset: number } {
    const buf = this.buf;
    const [version, arg1, arg2] = struct.unpack_from("<BBB", buf, msg_offset);

    let dims: number;
    let layout_class: number;
    let property_offset: number;

    if (version === 1 || version === 2) {
      dims = arg1;
      layout_class = arg2;
      property_offset = msg_offset + struct.calcsize("<BBB") + struct.calcsize("<BI");
    } else if (version === 3 || version === 4) {
      layout_class = arg1;
      dims = 0;
      property_offset = msg_offset + struct.calcsize("<BB");
    } else {
      throw new Error(`Unknown data storage message version: ${version}`);
    }

    return { version, dims, layout_class, property_offset };
  }

  _get_chunk_params(): void {
    if (this._chunk_params_set) return;
    this._chunk_params_set = true;

    const msg = this.find_msg_type(DATA_STORAGE_MSG_TYPE)[0];
    if (!msg) return;

    const absOffset = msg.get("offset_to_message") as number;
    const relOffset = absOffset - this.bufStart;
    const { version, dims: msgDims, layout_class, property_offset } =
      this._get_data_message_properties(relOffset);

    if (layout_class !== 2) return; // not chunked

    const buf = this.buf;
    let address: number;
    let dims: number;
    let data_offset: number;

    if (version === 1 || version === 2) {
      address = struct.unpack_from("<Q", buf, property_offset)[0];
      data_offset = property_offset + struct.calcsize("<Q");
      dims = msgDims;
    } else if (version === 3) {
      [dims, address] = struct.unpack_from("<BQ", buf, property_offset);
      data_offset = property_offset + struct.calcsize("<BQ");
    } else {
      return;
    }

    const fmt = "<" + (dims - 1).toFixed() + "I";
    const chunk_shape = struct.unpack_from(fmt, buf, data_offset);
    this._chunks = chunk_shape as number[];
    this._chunk_dims = dims;
    this._chunk_address = address;
  }

  // ────────── Data shape helper ──────────

  _determine_data_shape(buf: ArrayBuffer, offset: number): number[] {
    const version = struct.unpack_from("<B", buf, offset)[0];
    let header: Map<string, any>;

    if (version === 1) {
      header = _unpack_struct_from(DATASPACE_MSG_HEADER_V1, buf, offset);
      assert(header.get("version") === 1);
      offset += DATASPACE_MSG_HEADER_V1_SIZE;
    } else if (version === 2) {
      header = _unpack_struct_from(DATASPACE_MSG_HEADER_V2, buf, offset);
      assert(header.get("version") === 2);
      offset += DATASPACE_MSG_HEADER_V2_SIZE;
    } else {
      throw new Error(`Unknown dataspace message version: ${version}`);
    }

    const ndims: number = header.get("dimensionality");
    if (ndims === 0) return [];

    // Read dimension sizes as uint64
    const dim_sizes = struct.unpack_from("<" + ndims.toFixed() + "Q", buf, offset);
    return dim_sizes as number[];
  }

  // ────────── Async: get_links ──────────

  async get_links(): Promise<Record<string, any>> {
    const links: Record<string, any> = {};

    for (const msg of this.msgs) {
      const msg_type: number = msg.get("type");

      if (msg_type === SYMBOL_TABLE_MSG_TYPE) {
        const newLinks = await this._iter_links_from_symbol_tables(msg);
        Object.assign(links, newLinks);
      } else if (msg_type === LINK_MSG_TYPE) {
        const absOffset = msg.get("offset_to_message") as number;
        const relOffset = absOffset - this.bufStart;
        const [, [name, address]] = this._decode_link_msg(this.buf, relOffset);
        links[name] = address;
      } else if (msg_type === LINK_INFO_MSG_TYPE) {
        const newLinks = await this._iter_link_from_link_info_msg(msg);
        Object.assign(links, newLinks);
      }
    }

    return links;
  }

  private async _iter_links_from_symbol_tables(
    sym_tbl_msg: Map<string, any>
  ): Promise<Record<string, any>> {
    assert(sym_tbl_msg.get("size") === 16);
    const absOffset = sym_tbl_msg.get("offset_to_message") as number;
    const relOffset = absOffset - this.bufStart;
    const data = _unpack_struct_from(SYMBOL_TABLE_MSG, this.buf, relOffset);
    return this._iter_links_btree_v1(data.get("btree_address"), data.get("heap_address"));
  }

  private async _iter_links_btree_v1(
    btree_address: number,
    heap_address: number
  ): Promise<Record<string, any>> {
    const btree = await BTreeV1Groups.create(this._source, btree_address);
    const heap = await Heap.create(this._source, heap_address);
    const links: Record<string, any> = {};

    for (const symbol_table_address of btree.symbol_table_addresses()) {
      const table = await SymbolTable.create(this._source, symbol_table_address);
      table.assign_name(heap);
      Object.assign(links, table.get_links(heap));
    }

    return links;
  }

  private async _iter_link_from_link_info_msg(
    info_msg: Map<string, any>
  ): Promise<Record<string, any>> {
    const absOffset = info_msg.get("offset_to_message") as number;
    const relOffset = absOffset - this.bufStart;
    const data = this._decode_link_info_msg(this.buf, relOffset);

    const heap_address: number | null = data.get("heap_address");
    const name_btree_address: number | null = data.get("name_btree_address");
    const order_btree_address: number | null = data.get("order_btree_address");

    if (name_btree_address !== null) {
      return this._iter_links_btree_v2(name_btree_address, order_btree_address, heap_address!);
    }

    return {};
  }

  private async _iter_links_btree_v2(
    name_btree_address: number,
    order_btree_address: number | null,
    heap_address: number
  ): Promise<Record<string, any>> {
    const heap = await FractalHeap.create(this._source, heap_address);
    const ordered = order_btree_address !== null;

    let btree: BTreeV2GroupNames | BTreeV2GroupOrders;
    if (ordered) {
      btree = await BTreeV2GroupOrders.create(this._source, order_btree_address!);
    } else {
      btree = await BTreeV2GroupNames.create(this._source, name_btree_address);
    }

    const items = new Map<any, [string, any]>();
    for (const record of btree.iter_records()) {
      const data = heap.get_data(record.get("heapid"));
      const [creationorder, item] = this._decode_link_msg(data, 0);
      const key = ordered ? creationorder : (item as [string, any])[0];
      items.set(key, item as [string, any]);
    }

    const sorted_keys = Array.from(items.keys()).sort();
    const links: Record<string, any> = {};
    for (const key of sorted_keys) {
      const [name, address] = items.get(key)!;
      links[name] = address;
    }
    return links;
  }

  private _decode_link_msg(
    data: ArrayBuffer,
    offset: number
  ): [any, [string, any]] {
    const [version, flags] = struct.unpack_from("<BB", data, offset);
    offset += 2;
    assert(version === 1);

    const size_of_length_of_link_name = 2 ** (flags & 3);
    const link_type_field_present = (flags & (1 << 3)) > 0;
    const link_name_character_set_field_present = (flags & (1 << 4)) > 0;
    const ordered = (flags & (1 << 2)) > 0;

    let link_type = 0;
    if (link_type_field_present) {
      link_type = struct.unpack_from("<B", data, offset)[0];
      offset += 1;
    }
    assert([0, 1].includes(link_type));

    let creationorder: any;
    if (ordered) {
      creationorder = struct.unpack_from("<Q", data, offset)[0];
      offset += 8;
    }

    let link_name_character_set = 0;
    if (link_name_character_set_field_present) {
      link_name_character_set = struct.unpack_from("<B", data, offset)[0];
      offset += 1;
    }

    const encoding = link_name_character_set === 0 ? "ascii" : "utf-8";
    const name_size_fmt = ["<B", "<H", "<I", "<Q"][flags & 3];
    const name_size = struct.unpack_from(name_size_fmt, data, offset)[0];
    offset += size_of_length_of_link_name;

    const name = new TextDecoder(encoding).decode(
      (data as ArrayBuffer).slice(offset, offset + name_size)
    );
    offset += name_size;

    let address: any;
    if (link_type === 0) {
      // hard link
      address = struct.unpack_from("<Q", data, offset)[0];
    } else if (link_type === 1) {
      // soft link
      const length_of_soft_link_value = struct.unpack_from("<H", data, offset)[0];
      offset += 2;
      address = new TextDecoder(encoding).decode(
        (data as ArrayBuffer).slice(offset, offset + length_of_soft_link_value)
      );
    }

    return [creationorder, [name, address]];
  }

  private _decode_link_info_msg(
    data: ArrayBuffer,
    offset: number
  ): Map<string, number | null> {
    const [version, flags] = struct.unpack_from("<BB", data, offset);
    assert(version === 0);
    offset += 2;

    if ((flags & 1) > 0) {
      // creation order present
      offset += 8;
    }

    const fmt = (flags & 2) > 0 ? LINK_INFO_MSG2 : LINK_INFO_MSG1;
    const link_info = _unpack_struct_from(fmt, data, offset);

    const output = new Map<string, number | null>();
    for (const [k, v] of link_info.entries()) {
      output.set(k, v === UNDEFINED_ADDRESS ? null : v);
    }
    // Ensure order_btree_address exists (may be absent in LINK_INFO_MSG1)
    if (!output.has("order_btree_address")) {
      output.set("order_btree_address", null);
    }
    return output;
  }

  // ────────── Async: get_attributes ──────────

  async get_attributes(): Promise<Record<string, any>> {
    const attrs: Record<string, any> = {};

    const attr_msgs = this.find_msg_type(ATTRIBUTE_MSG_TYPE);
    for (const msg of attr_msgs) {
      const absOffset = msg.get("offset_to_message") as number;
      const relOffset = absOffset - this.bufStart;
      try {
        const [name, value] = await this._unpack_attribute(relOffset);
        attrs[name] = value;
      } catch {
        // skip unreadable attributes
      }
    }

    // Handle ATTRIBUTE_INFO_MSG_TYPE (dense attribute storage)
    const attr_info_msgs = this.find_msg_type(ATTRIBUTE_INFO_MSG_TYPE);
    for (const info_msg of attr_info_msgs) {
      try {
        const newAttrs = await this._read_dense_attributes(info_msg);
        Object.assign(attrs, newAttrs);
      } catch {
        // skip if not supported
      }
    }

    return attrs;
  }

  private async _read_dense_attributes(
    info_msg: Map<string, any>
  ): Promise<Record<string, any>> {
    const absOffset = info_msg.get("offset_to_message") as number;
    let offset = absOffset - this.bufStart;
    const buf = this.buf;

    const [version, flags] = struct.unpack_from("<BB", buf, offset);
    if (version !== 0) return {};
    offset += 2;

    // If flags bit 0: max creation index (2 bytes)
    if (flags & 1) offset += 2;

    // fractal_heap_address (8 bytes), name_btree_address (8 bytes)
    const [fractal_heap_address, name_btree_address] = struct.unpack_from("<QQ", buf, offset);
    offset += 16;

    // If flags bit 1: order_btree_address (8 bytes)
    let order_btree_address: number | null = null;
    if (flags & 2) {
      order_btree_address = struct.unpack_from("<Q", buf, offset)[0];
      offset += 8;
    }

    if (fractal_heap_address === UNDEFINED_ADDRESS || name_btree_address === UNDEFINED_ADDRESS) {
      return {};
    }

    const heap = await FractalHeap.create(this._source, fractal_heap_address);
    const ordered = order_btree_address !== null && order_btree_address !== UNDEFINED_ADDRESS;

    let btree: BTreeV2GroupNames | BTreeV2GroupOrders;
    if (ordered) {
      btree = await BTreeV2GroupOrders.create(this._source, order_btree_address!);
    } else {
      btree = await BTreeV2GroupNames.create(this._source, name_btree_address);
    }

    const attrs: Record<string, any> = {};
    for (const record of btree.iter_records()) {
      try {
        const attr_data = heap.get_data(record.get("heapid"));
        const [name, value] = await this._unpack_attribute_from_buf(attr_data, 0);
        attrs[name] = value;
      } catch {
        // skip
      }
    }
    return attrs;
  }

  private async _unpack_attribute(relOffset: number): Promise<[string, any]> {
    return this._unpack_attribute_from_buf(this.buf, relOffset);
  }

  private async _unpack_attribute_from_buf(
    buf: ArrayBuffer,
    offset: number
  ): Promise<[string, any]> {
    const version = struct.unpack_from("<B", buf, offset)[0];

    let attr_map: Map<string, any>;
    let padding_multiple: number;

    if (version === 1) {
      attr_map = _unpack_struct_from(ATTR_MSG_HEADER_V1, buf, offset);
      assert(attr_map.get("version") === 1);
      offset += ATTR_MSG_HEADER_V1_SIZE;
      padding_multiple = 8;
    } else if (version === 3) {
      attr_map = _unpack_struct_from(ATTR_MSG_HEADER_V3, buf, offset);
      assert(attr_map.get("version") === 3);
      offset += ATTR_MSG_HEADER_V3_SIZE;
      padding_multiple = 1;
    } else {
      throw new Error(`Unsupported attribute message version: ${version}`);
    }

    // Read attribute name
    const name_size: number = attr_map.get("name_size");
    let name = struct.unpack_from("<" + name_size.toFixed() + "s", buf, offset)[0];
    name = name.replace(/\x00$/, "");
    offset += _padded_size(name_size, padding_multiple);

    // Read datatype
    let dtype: any;
    try {
      dtype = new DatatypeMessage(buf, offset).dtype;
    } catch {
      console.warn(`Attribute '${name}' type not implemented, set to null.`);
      return [name, null];
    }

    const datatype_size: number = attr_map.get("datatype_size");
    offset += _padded_size(datatype_size, padding_multiple);

    // Read dataspace (shape)
    const shape = this._determine_data_shape(buf, offset);
    const items = shape.reduce((a, b) => a * b, 1);
    const dataspace_size: number = attr_map.get("dataspace_size");
    offset += _padded_size(dataspace_size, padding_multiple);

    // Read value
    let value: any = await this._attr_value(dtype, buf, items, offset);

    if (shape.length === 0) {
      value = value[0];
    }

    return [name, value];
  }

  private async _attr_value(
    dtype: any,
    buf: ArrayBuffer,
    count: number,
    offset: number
  ): Promise<any[]> {
    const value: any[] = new Array(count);

    if (Array.isArray(dtype)) {
      const dtype_class = dtype[0];
      for (let i = 0; i < count; i++) {
        if (dtype_class === "VLEN_STRING") {
          const character_set = dtype[2];
          const [, vlen_data] = await this._vlen_size_and_data(buf, offset);
          const encoding = character_set === 0 ? "ascii" : "utf-8";
          value[i] = new TextDecoder(encoding).decode(vlen_data);
          offset += 16;
        } else if (dtype_class === "REFERENCE") {
          const address = struct.unpack_from("<Q", buf, offset);
          value[i] = address;
          offset += 8;
        } else if (dtype_class === "VLEN_SEQUENCE") {
          const base_dtype = dtype[1];
          const [vlen, vlen_data] = await this._vlen_size_and_data(buf, offset);
          value[i] = await this._attr_value(base_dtype, vlen_data, vlen, 0);
          offset += 16;
        } else {
          throw new Error("NotImplementedError: unsupported dtype class " + dtype_class);
        }
      }
    } else {
      const [getter, big_endian, size] = dtype_getter(dtype as string);
      const view = new DataView64(buf, 0);
      for (let i = 0; i < count; i++) {
        value[i] = (view as any)[getter](offset, !big_endian, size);
        offset += size;
      }
    }

    return value;
  }

  private async _vlen_size_and_data(
    buf: ArrayBuffer,
    offset: number
  ): Promise<[number, ArrayBuffer]> {
    const vlen_size = struct.unpack_from("<I", buf, offset)[0];
    const gheap_id = _unpack_struct_from(GLOBAL_HEAP_ID, buf, offset + 4);
    const gheap_address: number = gheap_id.get("collection_address");
    assert(gheap_address < Number.MAX_SAFE_INTEGER);

    let gheap: GlobalHeap;
    if (this._global_heaps.has(gheap_address)) {
      gheap = this._global_heaps.get(gheap_address)!;
    } else {
      gheap = await GlobalHeap.create(this._source, gheap_address);
      this._global_heaps.set(gheap_address, gheap);
    }

    const vlen_data = gheap.objects.get(gheap_id.get("object_index"));
    if (!vlen_data) {
      throw new Error(`Global heap object not found at index ${gheap_id.get("object_index")}`);
    }
    return [vlen_size, vlen_data];
  }

  // ────────── Async: get_data ──────────

  async get_data(): Promise<any[]> {
    const msg = this.find_msg_type(DATA_STORAGE_MSG_TYPE)[0];
    if (!msg) return [];

    const absOffset = msg.get("offset_to_message") as number;
    const relOffset = absOffset - this.bufStart;
    const { layout_class, property_offset } =
      this._get_data_message_properties(relOffset);

    if (layout_class === 0) {
      throw new Error("Compact storage of DataObject not implemented");
    } else if (layout_class === 1) {
      return this._get_contiguous_data(property_offset);
    } else if (layout_class === 2) {
      return this._get_chunked_data(relOffset);
    }

    return [];
  }

  private async _get_contiguous_data(property_offset: number): Promise<any[]> {
    const buf = this.buf;
    const data_offset: number = struct.unpack_from("<Q", buf, property_offset)[0];

    const shape = this.shape;
    const fullsize = shape.reduce((a, b) => a * b, 1);

    if (data_offset === UNDEFINED_ADDRESS) {
      return new Array(fullsize);
    }

    const dtype = this.dtype;

    if (!Array.isArray(dtype)) {
      if (/[<>=!@|]?(i|u|f|S)(\d*)/.test(dtype)) {
        const [item_getter, item_is_big_endian, item_size] = dtype_getter(dtype);
        const dataBuf = await this._source.fetch(data_offset, fullsize * item_size);
        const view = new DataView64(dataBuf, 0);
        const output: any[] = new Array(fullsize);
        for (let i = 0; i < fullsize; i++) {
          output[i] = (view as any)[item_getter](i * item_size, !item_is_big_endian, item_size);
        }
        return output;
      } else {
        throw new Error("Not implemented - no proper dtype defined: " + dtype);
      }
    } else {
      const dtype_class = dtype[0];

      if (dtype_class === "REFERENCE") {
        const size = dtype[1];
        if (size !== 8) {
          throw new Error("Unsupported Reference type size: " + size);
        }
        const dataBuf = await this._source.fetch(data_offset, fullsize * 8);
        return [dataBuf];
      } else if (dtype_class === "VLEN_STRING") {
        const character_set = dtype[2];
        const encoding = character_set === 0 ? "ascii" : "utf-8";
        const decoder = new TextDecoder(encoding);
        // Fetch all 16-byte vlen refs at once
        const refBuf = await this._source.fetch(data_offset, fullsize * 16);
        const output: string[] = new Array(fullsize);
        for (let i = 0; i < fullsize; i++) {
          const [, vlen_data] = await this._vlen_size_and_data(refBuf, i * 16);
          output[i] = decoder.decode(vlen_data);
        }
        return output;
      } else {
        throw new Error("Not implemented datatype class: " + dtype_class);
      }
    }
  }

  private async _get_chunked_data(msg_offset: number): Promise<any[]> {
    this._get_chunk_params();

    if (this._chunk_address === null || this._chunk_address === UNDEFINED_ADDRESS) {
      return [];
    }

    const dtype = this.dtype;
    if (!Array.isArray(dtype) || !/^VLEN/.test(dtype[0])) {
      // For non-VLEN chunked data, not needed for our use case
      // (processChunkReferences handles numeric chunked data via BTree directly)
      return [];
    }

    const dtype_class = dtype[0];
    const character_set = dtype[2];
    const encoding = character_set === 0 ? "ascii" : "utf-8";
    const decoder = new TextDecoder(encoding);

    const chunk_btree = await BTreeV1RawDataChunks.create(
      this._source,
      this._chunk_address,
      this._chunk_dims!
    );

    const output: string[] = [];

    const leafNodes = chunk_btree.all_nodes.get(0);
    if (!leafNodes) return output;

    for (const node of leafNodes) {
      for (let ik = 0; ik < node.keys.length; ik++) {
        const addr = node.addresses[ik];
        const chunk_size = node.keys[ik].chunk_size;

        // Fetch chunk data
        const chunkBuf = await this._source.fetch(addr, chunk_size);
        const nItems = chunk_size / 16; // each VLEN ref is 16 bytes

        for (let i = 0; i < nItems; i++) {
          try {
            const [, vlen_data] = await this._vlen_size_and_data(chunkBuf, i * 16);
            output.push(decoder.decode(vlen_data));
          } catch {
            // skip bad entries
          }
        }
      }
    }

    return output;
  }
}

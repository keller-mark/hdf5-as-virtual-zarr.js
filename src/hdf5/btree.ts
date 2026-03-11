/**
 * Vendored and adapted from jsfive/esm/btree.js.
 * BTreeV1Groups, BTreeV1RawDataChunks, BTreeV2GroupNames, BTreeV2GroupOrders.
 *
 * All classes use async factory methods with Source.fetch() for partial reads.
 * construct_data_from_chunks() and _filter_chunk() are omitted –
 * we only need chunk location metadata for the reference spec.
 */

import type { Source } from "../index.js";
import {
  _unpack_struct_from,
  _structure_size,
  struct,
  bitSize,
} from "./core.js";
import { UNDEFINED_ADDRESS } from "./misc-low-level.js";

// ────────── BTreeV1 common ──────────

const B_LINK_NODE_V1 = new Map<string, string>([
  ["signature", "4s"],
  ["node_type", "B"],
  ["node_level", "B"],
  ["entries_used", "H"],
  ["left_sibling", "Q"],
  ["right_sibling", "Q"],
]);
const B_LINK_NODE_V1_SIZE = _structure_size(B_LINK_NODE_V1);

async function _readV1NodeHeader(
  source: Source,
  offset: number,
  node_level: number | null
): Promise<[Map<string, any>, ArrayBuffer]> {
  // Read header + generous amount for keys/addresses that follow
  // We don't know exact size yet, so read header first
  const hBuf = await source.fetch(offset, B_LINK_NODE_V1_SIZE);
  const node = _unpack_struct_from(B_LINK_NODE_V1, hBuf, 0);
  if (node_level != null && node.get("node_level") !== node_level) {
    throw new Error("node level does not match");
  }
  return [node, hBuf];
}

interface BTreeV1Node {
  header: Map<string, any>;
  keys: any[];
  addresses: number[];
}

// ────────── BTreeV1Groups (type 0) ──────────

export class BTreeV1Groups {
  all_nodes: Map<number, BTreeV1Node[]>;
  depth: number;

  private constructor(all_nodes: Map<number, BTreeV1Node[]>, depth: number) {
    this.all_nodes = all_nodes;
    this.depth = depth;
  }

  static async create(source: Source, offset: number): Promise<BTreeV1Groups> {
    const all_nodes = new Map<number, BTreeV1Node[]>();

    async function readNode(
      off: number,
      node_level: number | null
    ): Promise<BTreeV1Node> {
      const [header] = await _readV1NodeHeader(source, off, node_level);
      let readOff = off + B_LINK_NODE_V1_SIZE;

      const entries_used = header.get("entries_used");
      // Each entry: key(Q=8) + address(Q=8) = 16 bytes, plus one extra key
      const dataSize = entries_used * 16 + 8;
      const dataBuf = await source.fetch(readOff, dataSize);
      let localOff = 0;

      const keys: number[] = [];
      const addresses: number[] = [];
      for (let i = 0; i < entries_used; i++) {
        keys.push(struct.unpack_from("<Q", dataBuf, localOff)[0]);
        localOff += 8;
        addresses.push(struct.unpack_from("<Q", dataBuf, localOff)[0]);
        localOff += 8;
      }
      // N+1 key
      keys.push(struct.unpack_from("<Q", dataBuf, localOff)[0]);

      return { header, keys, addresses };
    }

    // Read root
    const root = await readNode(offset, null);
    const depth = root.header.get("node_level");
    addNode(all_nodes, root, depth);

    // Read children
    let node_level = depth;
    while (node_level > 0) {
      const parents = all_nodes.get(node_level) || [];
      for (const parent of parents) {
        for (const child_addr of parent.addresses) {
          const child = await readNode(child_addr, node_level - 1);
          addNode(all_nodes, child, node_level - 1);
        }
      }
      node_level--;
    }

    return new BTreeV1Groups(all_nodes, depth);
  }

  symbol_table_addresses(): number[] {
    const all_address: number[] = [];
    const root_nodes = this.all_nodes.get(0) || [];
    for (const node of root_nodes) {
      all_address.push(...node.addresses);
    }
    return all_address;
  }
}

// ────────── BTreeV1RawDataChunks (type 1) ──────────

export interface ChunkKey {
  chunk_size: number;
  filter_mask: number;
  chunk_offset: number[];
}

export class BTreeV1RawDataChunks {
  all_nodes: Map<number, { header: Map<string, any>; keys: ChunkKey[]; addresses: number[] }[]>;
  depth: number;
  dims: number;

  private constructor(
    all_nodes: Map<number, any[]>,
    depth: number,
    dims: number
  ) {
    this.all_nodes = all_nodes;
    this.depth = depth;
    this.dims = dims;
  }

  static async create(
    source: Source,
    offset: number,
    dims: number
  ): Promise<BTreeV1RawDataChunks> {
    const all_nodes = new Map<number, any[]>();

    async function readNode(
      off: number,
      node_level: number | null
    ) {
      const [header] = await _readV1NodeHeader(source, off, node_level);
      let readOff = off + B_LINK_NODE_V1_SIZE;

      const entries_used = header.get("entries_used");
      // Each entry: chunk_size(I=4) + filter_mask(I=4) + dims*Q + address(Q=8)
      const fmt = "<" + dims.toFixed() + "Q";
      const fmt_size = struct.calcsize(fmt);
      const entrySize = 8 + fmt_size + 8; // II + dims*Q + Q
      const dataBuf = await source.fetch(readOff, entries_used * entrySize);
      let localOff = 0;

      const keys: ChunkKey[] = [];
      const addresses: number[] = [];
      for (let i = 0; i < entries_used; i++) {
        const [chunk_size, filter_mask] = struct.unpack_from("<II", dataBuf, localOff);
        localOff += 8;
        const chunk_offset = struct.unpack_from(fmt, dataBuf, localOff);
        localOff += fmt_size;
        const chunk_address = struct.unpack_from("<Q", dataBuf, localOff)[0];
        localOff += 8;

        keys.push({ chunk_size, filter_mask, chunk_offset });
        addresses.push(chunk_address);
      }

      return { header, keys, addresses };
    }

    // Read root
    const root = await readNode(offset, null);
    const depth = root.header.get("node_level");
    addNode(all_nodes, root, depth);

    // Read children
    let node_level = depth;
    while (node_level > 0) {
      const parents = all_nodes.get(node_level) || [];
      for (const parent of parents) {
        for (const child_addr of parent.addresses) {
          const child = await readNode(child_addr, node_level - 1);
          addNode(all_nodes, child, node_level - 1);
        }
      }
      node_level--;
    }

    return new BTreeV1RawDataChunks(all_nodes, depth, dims);
  }
}

// ────────── BTreeV2 common ──────────

const B_TREE_HEADER_V2 = new Map<string, string>([
  ["signature", "4s"],
  ["version", "B"],
  ["node_type", "B"],
  ["node_size", "I"],
  ["record_size", "H"],
  ["depth", "H"],
  ["split_percent", "B"],
  ["merge_percent", "B"],
  ["root_address", "Q"],
  ["root_nrecords", "H"],
  ["total_nrecords", "Q"],
]);

const B_LINK_NODE_V2 = new Map<string, string>([
  ["signature", "4s"],
  ["version", "B"],
  ["node_type", "B"],
]);
const B_LINK_NODE_V2_SIZE = _structure_size(B_LINK_NODE_V2);

function _required_bytes(integer: number): number {
  return Math.ceil(bitSize(integer) / 8);
}

function _int_format(bytelength: number): string {
  return ["<B", "<H", "<I", "<Q"][bytelength - 1];
}

function _nrecords_max(
  node_size: number,
  record_size: number,
  addr_size: number
): number {
  return Math.floor((node_size - 10 - addr_size) / (record_size + addr_size));
}

function _calculate_address_formats(
  header: Map<string, any>
): Map<number, [number, number, number, string, string, string]> {
  const node_size = header.get("node_size");
  const record_size = header.get("record_size");
  let nrecords_max = 0;
  let ntotalrecords_max = 0;
  const address_formats = new Map<
    number,
    [number, number, number, string, string, string]
  >();
  const max_depth = header.get("depth");

  for (let node_level = 0; node_level <= max_depth; node_level++) {
    let offset_fmt = "";
    let num1_fmt = "";
    let num2_fmt = "";
    let offset_size: number, num1_size: number, num2_size: number;

    if (node_level === 0) {
      offset_size = 0;
      num1_size = 0;
      num2_size = 0;
    } else if (node_level === 1) {
      offset_size = 8;
      offset_fmt = "<Q";
      num1_size = _required_bytes(nrecords_max);
      num1_fmt = _int_format(num1_size);
      num2_size = 0;
    } else {
      offset_size = 8;
      offset_fmt = "<Q";
      num1_size = _required_bytes(nrecords_max);
      num1_fmt = _int_format(num1_size);
      num2_size = _required_bytes(ntotalrecords_max);
      num2_fmt = _int_format(num2_size);
    }

    address_formats.set(node_level, [
      offset_size, num1_size, num2_size,
      offset_fmt, num1_fmt, num2_fmt,
    ]);

    if (node_level < max_depth) {
      const addr_size = offset_size + num1_size + num2_size;
      nrecords_max = _nrecords_max(node_size, record_size, addr_size);
      if (ntotalrecords_max > 0) {
        ntotalrecords_max *= nrecords_max;
      } else {
        ntotalrecords_max = nrecords_max;
      }
    }
  }

  return address_formats;
}

type RecordParser = (buf: ArrayBuffer, offset: number, size: number) => Map<string, any>;

async function _readBTreeV2(
  source: Source,
  offset: number,
  parseRecord: RecordParser
): Promise<{ all_nodes: Map<number, any[]>; header: Map<string, any> }> {
  // Read tree header
  const headerSize = _structure_size(B_TREE_HEADER_V2);
  const hBuf = await source.fetch(offset, headerSize);
  const header = _unpack_struct_from(B_TREE_HEADER_V2, hBuf, 0);
  const address_formats = _calculate_address_formats(header);
  const depth = header.get("depth");
  const record_size = header.get("record_size");

  const all_nodes = new Map<number, any[]>();

  async function readNode(
    address: [number, number, number],
    node_level: number
  ) {
    const [off, nrecords] = address;

    // Read node header + records + addresses
    // Estimate max bytes needed
    const addr_fmts = address_formats.get(node_level)!;
    const [offset_size, num1_size, num2_size] = addr_fmts;
    const addr_entry_size = offset_size + num1_size + num2_size;
    const maxBytes =
      B_LINK_NODE_V2_SIZE +
      nrecords * record_size +
      (node_level !== 0 ? (nrecords + 1) * addr_entry_size : 0) +
      4; // checksum
    const nodeBuf = await source.fetch(off, maxBytes);

    const node = _unpack_struct_from(B_LINK_NODE_V2, nodeBuf, 0);
    node.set("node_level", node_level);
    let localOff = B_LINK_NODE_V2_SIZE;

    const keys: Map<string, any>[] = [];
    for (let i = 0; i < nrecords; i++) {
      keys.push(parseRecord(nodeBuf, localOff, record_size));
      localOff += record_size;
    }

    const addresses: [number, number, number][] = [];
    if (node_level !== 0) {
      const [os, n1s, n2s, offset_fmt, num1_fmt, num2_fmt] = addr_fmts;
      for (let j = 0; j <= nrecords; j++) {
        const address_offset = struct.unpack_from(offset_fmt, nodeBuf, localOff)[0];
        localOff += os;
        const num1 = struct.unpack_from(num1_fmt, nodeBuf, localOff)[0];
        localOff += n1s;
        let num2 = num1;
        if (n2s > 0) {
          num2 = struct.unpack_from(num2_fmt, nodeBuf, localOff)[0];
          localOff += n2s;
        }
        addresses.push([address_offset, num1, num2]);
      }
    }

    return { header: node, keys, addresses };
  }

  // Read root
  const rootAddr: [number, number, number] = [
    header.get("root_address"),
    header.get("root_nrecords"),
    header.get("total_nrecords"),
  ];
  const root = await readNode(rootAddr, depth);
  addNode(all_nodes, root, depth);

  // Read children
  let node_level = depth;
  while (node_level > 0) {
    const parents = all_nodes.get(node_level) || [];
    for (const parent of parents) {
      for (const child_addr of parent.addresses) {
        const child = await readNode(child_addr, node_level - 1);
        addNode(all_nodes, child, node_level - 1);
      }
    }
    node_level--;
  }

  return { all_nodes, header };
}

// ────────── BTreeV2GroupNames (type 5) ──────────

export class BTreeV2GroupNames {
  all_nodes: Map<number, any[]>;
  header: Map<string, any>;

  private constructor(all_nodes: Map<number, any[]>, header: Map<string, any>) {
    this.all_nodes = all_nodes;
    this.header = header;
  }

  static async create(source: Source, offset: number): Promise<BTreeV2GroupNames> {
    const { all_nodes, header } = await _readBTreeV2(source, offset, (buf, off) => {
      const namehash = struct.unpack_from("<I", buf, off)[0];
      return new Map([
        ["namehash", namehash],
        ["heapid", buf.slice(off + 4, off + 4 + 7)],
      ]);
    });
    return new BTreeV2GroupNames(all_nodes, header);
  }

  *iter_records(): Generator<Map<string, any>> {
    for (const nodelist of this.all_nodes.values()) {
      for (const node of nodelist) {
        for (const key of node.keys) {
          yield key;
        }
      }
    }
  }
}

// ────────── BTreeV2GroupOrders (type 6) ──────────

export class BTreeV2GroupOrders {
  all_nodes: Map<number, any[]>;
  header: Map<string, any>;

  private constructor(all_nodes: Map<number, any[]>, header: Map<string, any>) {
    this.all_nodes = all_nodes;
    this.header = header;
  }

  static async create(source: Source, offset: number): Promise<BTreeV2GroupOrders> {
    const { all_nodes, header } = await _readBTreeV2(source, offset, (buf, off) => {
      const creationorder = struct.unpack_from("<Q", buf, off)[0];
      return new Map([
        ["creationorder", creationorder],
        ["heapid", buf.slice(off + 8, off + 8 + 7)],
      ]);
    });
    return new BTreeV2GroupOrders(all_nodes, header);
  }

  *iter_records(): Generator<Map<string, any>> {
    for (const nodelist of this.all_nodes.values()) {
      for (const node of nodelist) {
        for (const key of node.keys) {
          yield key;
        }
      }
    }
  }
}

// ────────── shared helper ──────────

function addNode(
  all_nodes: Map<number, any[]>,
  node: any,
  node_level: number
): void {
  const level = node.header?.get?.("node_level") ?? node_level;
  if (all_nodes.has(level)) {
    all_nodes.get(level)!.push(node);
  } else {
    all_nodes.set(level, [node]);
  }
}

/**
 * Vendored and adapted from jsfive/esm/datatype-msg.js.
 * Parses HDF5 datatype messages synchronously from a pre-fetched buffer.
 */

import { _structure_size, _unpack_struct_from } from "./core.js";

const DATATYPE_MSG = new Map<string, string>([
  ["class_and_version", "B"],
  ["class_bit_field_0", "B"],
  ["class_bit_field_1", "B"],
  ["class_bit_field_2", "B"],
  ["size", "I"],
]);
const DATATYPE_MSG_SIZE = _structure_size(DATATYPE_MSG);

const DATATYPE_FIXED_POINT = 0;
const DATATYPE_FLOATING_POINT = 1;
const DATATYPE_TIME = 2;
const DATATYPE_STRING = 3;
const DATATYPE_BITFIELD = 4;
const DATATYPE_OPAQUE = 5;
const DATATYPE_COMPOUND = 6;
const DATATYPE_REFERENCE = 7;
const DATATYPE_ENUMERATED = 8;
const DATATYPE_VARIABLE_LENGTH = 9;

export class DatatypeMessage {
  buf: ArrayBuffer;
  offset: number;
  dtype: any;

  constructor(buf: ArrayBuffer, offset: number) {
    this.buf = buf;
    this.offset = offset;
    this.dtype = this.determine_dtype();
  }

  determine_dtype(): any {
    const datatype_msg = _unpack_struct_from(DATATYPE_MSG, this.buf, this.offset);
    this.offset += DATATYPE_MSG_SIZE;
    const datatype_class = datatype_msg.get("class_and_version") & 0x0f;

    if (datatype_class === DATATYPE_FIXED_POINT) {
      return this._determine_dtype_fixed_point(datatype_msg);
    } else if (datatype_class === DATATYPE_FLOATING_POINT) {
      return this._determine_dtype_floating_point(datatype_msg);
    } else if (datatype_class === DATATYPE_TIME) {
      throw new Error("Time datatype class not supported.");
    } else if (datatype_class === DATATYPE_STRING) {
      return this._determine_dtype_string(datatype_msg);
    } else if (datatype_class === DATATYPE_BITFIELD) {
      throw new Error("Bitfield datatype class not supported.");
    } else if (datatype_class === DATATYPE_OPAQUE) {
      throw new Error("Opaque datatype class not supported.");
    } else if (datatype_class === DATATYPE_COMPOUND) {
      return this._determine_dtype_compound(datatype_msg);
    } else if (datatype_class === DATATYPE_REFERENCE) {
      return ["REFERENCE", datatype_msg.get("size")];
    } else if (datatype_class === DATATYPE_ENUMERATED) {
      return this.determine_dtype();
    } else if (datatype_class === DATATYPE_VARIABLE_LENGTH) {
      const vlen_type = this._determine_dtype_vlen(datatype_msg);
      if (vlen_type[0] === "VLEN_SEQUENCE") {
        const base_type = this.determine_dtype();
        return ["VLEN_SEQUENCE", base_type];
      }
      return vlen_type;
    } else {
      throw new Error("Invalid datatype class " + datatype_class);
    }
  }

  _determine_dtype_fixed_point(datatype_msg: Map<string, any>): string {
    const length_in_bytes = datatype_msg.get("size");
    if (![1, 2, 4, 8].includes(length_in_bytes)) {
      throw new Error("Unsupported datatype size");
    }
    const signed = datatype_msg.get("class_bit_field_0") & 0x08;
    const dtype_char = signed > 0 ? "i" : "u";
    const byte_order = datatype_msg.get("class_bit_field_0") & 0x01;
    const byte_order_char = byte_order === 0 ? "<" : ">";
    this.offset += 4; // 4-byte fixed-point property description
    return byte_order_char + dtype_char + length_in_bytes.toFixed();
  }

  _determine_dtype_floating_point(datatype_msg: Map<string, any>): string {
    const length_in_bytes = datatype_msg.get("size");
    if (![1, 2, 4, 8].includes(length_in_bytes)) {
      throw new Error("Unsupported datatype size");
    }
    const byte_order = datatype_msg.get("class_bit_field_0") & 0x01;
    const byte_order_char = byte_order === 0 ? "<" : ">";
    this.offset += 12; // 12-bytes floating-point property description
    return byte_order_char + "f" + length_in_bytes.toFixed();
  }

  _determine_dtype_string(datatype_msg: Map<string, any>): string {
    return "S" + datatype_msg.get("size").toFixed();
  }

  _determine_dtype_compound(datatype_msg: Map<string, any>): any[] {
    const version = (datatype_msg.get("class_and_version") >> 4) & 0x0f;
    const num_members = datatype_msg.get("class_bit_field_0") |
                       (datatype_msg.get("class_bit_field_1") << 8);
    const total_size = datatype_msg.get("size");

    // Return minimal descriptor if we can't parse members — processDataset will skip this anyway
    try {
      const members: Array<{name: string; dtype: any; offset: number}> = [];
      const view = new DataView(this.buf);

      for (let i = 0; i < num_members; i++) {
        // Read null-terminated name
        let nameEnd = this.offset;
        while (nameEnd < this.buf.byteLength && view.getUint8(nameEnd) !== 0) {
          nameEnd++;
        }
        const nameBytes = new Uint8Array(this.buf, this.offset, nameEnd - this.offset);
        const name = new TextDecoder().decode(nameBytes);

        let memberOffset: number;

        if (version === 1) {
          // Version 1: name padded to 8-byte boundary
          const nameLenWithNull = nameEnd - this.offset + 1;
          const paddedLen = Math.ceil(nameLenWithNull / 8) * 8;
          this.offset += paddedLen;

          // Read byte offset of member (4 bytes)
          memberOffset = view.getUint32(this.offset, true);
          this.offset += 4;

          // Skip: dimensionality (1) + reserved (3) + permutation (4) + reserved (4) + dim sizes (16) = 28 bytes
          this.offset += 28;
        } else if (version === 2) {
          // Version 2: null-terminated name, no padding
          this.offset = nameEnd + 1;

          // Read byte offset (4 bytes)
          memberOffset = view.getUint32(this.offset, true);
          this.offset += 4;
        } else {
          // Version 3: null-terminated name, variable-width offset
          this.offset = nameEnd + 1;

          const offsetBytes = Math.max(1, Math.ceil(Math.ceil(Math.log2(total_size + 1)) / 8));
          memberOffset = 0;
          for (let j = 0; j < offsetBytes; j++) {
            memberOffset |= view.getUint8(this.offset + j) << (j * 8);
          }
          this.offset += offsetBytes;
        }

        // Parse nested datatype (recursive) — updates this.offset
        const nestedMsg = new DatatypeMessage(this.buf, this.offset);
        this.offset = nestedMsg.offset;

        members.push({ name, dtype: nestedMsg.dtype, offset: memberOffset });
      }

      return ["COMPOUND", members, total_size];
    } catch {
      return ["COMPOUND", [], total_size];
    }
  }

  _determine_dtype_vlen(datatype_msg: Map<string, any>): any[] {
    const vlen_type = datatype_msg.get("class_bit_field_0") & 0x01;
    if (vlen_type !== 1) {
      return ["VLEN_SEQUENCE", 0, 0];
    }
    const padding_type = datatype_msg.get("class_bit_field_0") >> 4;
    const character_set = datatype_msg.get("class_bit_field_1") & 0x01;
    return ["VLEN_STRING", padding_type, character_set];
  }
}

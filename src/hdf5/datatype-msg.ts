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
      throw new Error("Compound datatype class not supported.");
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

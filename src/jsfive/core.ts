/**
 * Vendored and adapted from jsfive/esm/core.js.
 * Provides struct unpacking, DataView64, and helper utilities.
 * These are synchronous operations on pre-fetched ArrayBuffers.
 */

export function _unpack_struct_from(
  structure: Map<string, string>,
  buf: ArrayBuffer,
  offset = 0
): Map<string, any> {
  const output = new Map<string, any>();
  for (const [key, fmt] of structure.entries()) {
    let value: any = struct.unpack_from("<" + fmt, buf, offset);
    offset += struct.calcsize(fmt);
    if (value.length === 1) {
      value = value[0];
    }
    output.set(key, value);
  }
  return output;
}

export function _structure_size(structure: Map<string, string>): number {
  const fmt = "<" + Array.from(structure.values()).join("");
  return struct.calcsize(fmt);
}

export function _padded_size(size: number, padding_multiple = 8): number {
  return Math.ceil(size / padding_multiple) * padding_multiple;
}

const dtype_to_format: Record<string, string> = {
  u: "Uint",
  i: "Int",
  f: "Float",
};

export function dtype_getter(
  dtype_str: string
): [string, boolean, number] {
  const big_endian = struct._is_big_endian(dtype_str);
  let getter: string;
  let nbytes: number;
  if (/S/.test(dtype_str)) {
    getter = "getString";
    nbytes = (parseInt((dtype_str.match(/S(\d*)/) || [])[1] || "1", 10)) | 0;
  } else {
    const m = dtype_str.match(/[<>=!@]?(i|u|f)(\d*)/);
    if (!m) throw new Error("Invalid dtype: " + dtype_str);
    const fstr = m[1];
    const bytestr = m[2];
    nbytes = parseInt(bytestr || "4", 10);
    const nbits = nbytes * 8;
    getter = "get" + dtype_to_format[fstr] + nbits.toFixed();
  }
  return [getter, big_endian, nbytes];
}

export class Reference {
  address_of_reference: number;
  constructor(address_of_reference: number) {
    this.address_of_reference = address_of_reference;
  }
}

function isBigEndian(): boolean {
  const array = new Uint8Array(4);
  const view = new Uint32Array(array.buffer);
  return !((view[0] = 1) & array[0]);
}

function decodeFloat16(low: number, high: number): number {
  const sign = (high & 0b10000000) >> 7;
  const exponent = (high & 0b01111100) >> 2;
  const fraction = ((high & 0b00000011) << 8) + low;
  let magnitude: number;
  if (exponent === 0b11111) {
    magnitude = fraction === 0 ? Infinity : NaN;
  } else if (exponent === 0) {
    magnitude = 2 ** -14 * (fraction / 1024);
  } else {
    magnitude = 2 ** (exponent - 15) * (1 + fraction / 1024);
  }
  return sign ? -magnitude : magnitude;
}

export class DataView64 extends DataView<ArrayBuffer> {
  getFloat16(byteOffset: number, littleEndian: boolean): number {
    const bytes = [this.getUint8(byteOffset), this.getUint8(byteOffset + 1)];
    if (!littleEndian) bytes.reverse();
    return decodeFloat16(bytes[0], bytes[1]);
  }

  getUint64(byteOffset: number, littleEndian: boolean): number {
    const left = BigInt(this.getUint32(byteOffset, littleEndian));
    const right = BigInt(this.getUint32(byteOffset + 4, littleEndian));
    const combined = littleEndian
      ? left + (right << 32n)
      : (left << 32n) + right;
    return Number(combined);
  }

  getInt64(byteOffset: number, littleEndian: boolean): number {
    let low: number, high: number;
    if (littleEndian) {
      low = this.getUint32(byteOffset, true);
      high = this.getInt32(byteOffset + 4, true);
    } else {
      high = this.getInt32(byteOffset, false);
      low = this.getUint32(byteOffset + 4, false);
    }
    const combined = BigInt(low) + (BigInt(high) << 32n);
    return Number(combined);
  }

  getString(byteOffset: number, _littleEndian: boolean, length: number): string {
    const str_buffer = this.buffer.slice(byteOffset, byteOffset + length);
    return new TextDecoder().decode(str_buffer);
  }

  getVLENStruct(
    byteOffset: number,
    littleEndian: boolean,
    _length: number
  ): [number, number, number] {
    const item_size = this.getUint32(byteOffset, littleEndian);
    const collection_address = this.getUint64(byteOffset + 4, littleEndian);
    const object_index = this.getUint32(byteOffset + 12, littleEndian);
    return [item_size, collection_address, object_index];
  }
}

class Struct {
  big_endian: boolean;
  getters: Record<string, string>;
  byte_lengths: Record<string, number>;
  fmt_size_regex: string;

  constructor() {
    this.big_endian = isBigEndian();
    this.getters = {
      s: "getUint8",  b: "getInt8",  B: "getUint8",
      h: "getInt16",  H: "getUint16",
      i: "getInt32",  I: "getUint32",
      l: "getInt32",  L: "getUint32",
      q: "getInt64",  Q: "getUint64",
      e: "getFloat16", f: "getFloat32", d: "getFloat64",
    };
    this.byte_lengths = {
      s: 1, b: 1, B: 1, h: 2, H: 2, i: 4, I: 4,
      l: 4, L: 4, q: 8, Q: 8, e: 2, f: 4, d: 8,
    };
    const all_formats = Object.keys(this.byte_lengths).join("");
    this.fmt_size_regex = "(\\d*)([" + all_formats + "])";
  }

  calcsize(fmt: string): number {
    let size = 0;
    const regex = new RegExp(this.fmt_size_regex, "g");
    let match: RegExpExecArray | null;
    while ((match = regex.exec(fmt)) !== null) {
      const n = parseInt(match[1] || "1", 10);
      const f = match[2];
      size += n * this.byte_lengths[f];
    }
    return size;
  }

  _is_big_endian(fmt: string): boolean {
    if (/^</.test(fmt)) return false;
    if (/^(!|>)/.test(fmt)) return true;
    return this.big_endian;
  }

  unpack_from(fmt: string, buffer: ArrayBuffer, offset?: number): any[] {
    offset = Number(offset || 0);
    const view = new DataView64(buffer, 0);
    const output: any[] = [];
    const big_endian = this._is_big_endian(fmt);
    const regex = new RegExp(this.fmt_size_regex, "g");
    let match: RegExpExecArray | null;
    while ((match = regex.exec(fmt)) !== null) {
      const n = parseInt(match[1] || "1", 10);
      const f = match[2];
      const getter = this.getters[f];
      const size = this.byte_lengths[f];
      if (f === "s") {
        output.push(
          new TextDecoder().decode(buffer.slice(offset, offset + n))
        );
        offset += n;
      } else {
        for (let i = 0; i < n; i++) {
          output.push((view as any)[getter](offset, !big_endian));
          offset += size;
        }
      }
    }
    return output;
  }
}

export const struct = new Struct();

export function bitSize(integer: number): number {
  return integer.toString(2).length;
}

export function _unpack_integer(
  nbytes: number,
  fh: ArrayBuffer,
  offset = 0,
  littleEndian = true
): number {
  const bytes = new Uint8Array(fh.slice(offset, offset + nbytes));
  if (!littleEndian) bytes.reverse();
  return bytes.reduce(
    (acc, val, idx) => acc + (val << (idx * 8)),
    0
  );
}

export function assert(thing: any): void {
  if (!thing) {
    throw new Error("Assertion failed");
  }
}

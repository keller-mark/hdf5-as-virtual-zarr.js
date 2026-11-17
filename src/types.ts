import type { AsyncReadable } from "@zarrita/storage";

export type { AsyncReadable };

/**
 * Zarr v2 array metadata (.zarray).
 */
export interface ZarrArrayMeta {
  zarr_format: 2;
  shape: number[];
  chunks: number[];
  dtype: string;
  compressor: Record<string, unknown> | null;
  fill_value: unknown;
  order: "C" | "F";
  filters: Record<string, unknown>[] | null;
  dimension_separator?: string;
}

/**
 * Zarr v2 group metadata (.zgroup).
 */
export interface ZarrGroupMeta {
  zarr_format: 2;
}

/**
 * A single entry in a reference spec: inline data, or [url, offset, length].
 */
export type ReferenceEntry = string | [string | null, number, number];

/**
 * Reference spec v1 output.
 */
export interface ReferenceSpec {
  version: 1;
  refs: Record<string, ReferenceEntry>;
}

/**
 * Read a byte range from an AsyncReadable source, returning an ArrayBuffer.
 * Uses getRange("/", { offset, length }) from the zarrita AsyncReadable interface.
 */
export async function fetchRange(
  source: AsyncReadable,
  offset: number,
  length: number
): Promise<ArrayBuffer> {
  const result = await source.getRange!("/", { offset, length });
  if (!result) throw new Error(`fetchRange: no data at offset=${offset} length=${length}`);
  const copy = new Uint8Array(result.byteLength);
  copy.set(result);
  return copy.buffer;
}

const fileSizes = new WeakMap<AsyncReadable, number>();

/**
 * Record the size of the HDF5 file behind a source (from the superblock),
 * so that read-ahead requests never extend past the end of the file.
 */
export function setFileSize(source: AsyncReadable, size: number): void {
  fileSizes.set(source, size);
}

/**
 * Read at least `length` bytes at `offset`, speculatively reading up to
 * `readAhead` bytes (clamped to the end of the file, if known).
 *
 * Sources are assumed to be wrapped with zarrita's `withRangeCoalescing`,
 * which merges concurrent requests but does not cache: reading a little more
 * in one request avoids a dependent second round trip for adjacent bytes
 * (e.g. an object header followed by its messages). Callers must check the
 * returned length before using bytes beyond `length`.
 */
export function fetchRangeWithReadAhead(
  source: AsyncReadable,
  offset: number,
  length: number,
  readAhead: number
): Promise<ArrayBuffer> {
  const size = fileSizes.get(source);
  let want = length;
  if (size !== undefined && readAhead > length) {
    want = Math.max(length, Math.min(readAhead, size - offset));
  }
  return fetchRange(source, offset, want);
}

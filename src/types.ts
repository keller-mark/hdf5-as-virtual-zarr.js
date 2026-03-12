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
 * Reference spec v1 output.
 */
export interface ReferenceSpec {
  version: 1;
  refs: Record<string, string | [string | null, number, number]>;
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

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
 * Minimal Source interface compatible with @chunkd/source.
 * Accepts any object implementing fetch(offset, length?) for partial reads.
 */
export interface Source {
  type: string;
  url: URL;
  metadata?: { size?: number };
  head?(options?: { signal: AbortSignal }): Promise<{ size?: number }>;
  close?(): Promise<void>;
  fetch(offset: number, length?: number, options?: { signal: AbortSignal }): Promise<ArrayBuffer>;
}

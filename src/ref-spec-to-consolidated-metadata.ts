import type { ReferenceSpec } from "./types.js";

/**
 * Zarr v2 consolidated metadata (.zmetadata).
 */
export interface ZarrConsolidatedMetadata {
  zarr_consolidated_format: 1;
  metadata: Record<string, Record<string, unknown>>;
}


/**
 * Metadata key suffixes included in Zarr consolidated metadata.
 * TODO: support zarr v3 (zarr.json files)
 */
const METADATA_SUFFIXES = [".zgroup", ".zarray", ".zattrs"];

/**
 * Convert a Reference Spec JSON to Zarr v2 consolidated metadata (.zmetadata).
 */
export function refSpecToConsolidatedMetadata(
  refSpec: ReferenceSpec
): ZarrConsolidatedMetadata {
  const metadata: Record<string, Record<string, unknown>> = {};

  for (const [key, value] of Object.entries(refSpec.refs)) {
    if (METADATA_SUFFIXES.some((suffix) => key === suffix || key.endsWith(`/${suffix}`))) {
      if (typeof value === "string") {
        metadata[key] = JSON.parse(value);
      }
    }
  }

  return {
    zarr_consolidated_format: 1,
    metadata,
  };
}

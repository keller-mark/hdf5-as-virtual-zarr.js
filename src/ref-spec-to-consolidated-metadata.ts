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

/**
 * Zarr v3 root group metadata with inline consolidated metadata,
 * in the format written by zarr-python.
 */
export interface ZarrV3ConsolidatedMetadata {
  zarr_format: 3;
  node_type: "group";
  attributes?: Record<string, unknown>;
  consolidated_metadata: {
    kind: "inline";
    must_understand: false;
    metadata: Record<string, Record<string, unknown>>;
  };
}

/**
 * Convert a Zarr v3 Reference Spec JSON to a root zarr.json document
 * with inline consolidated metadata for every descendant node.
 */
export function refSpecToConsolidatedMetadataV3(
  refSpec: ReferenceSpec
): ZarrV3ConsolidatedMetadata {
  const rootValue = refSpec.refs["zarr.json"];
  if (typeof rootValue !== "string") {
    throw new Error("Reference spec does not contain a v3 root zarr.json");
  }
  const root = JSON.parse(rootValue);

  const metadata: Record<string, Record<string, unknown>> = {};
  for (const [key, value] of Object.entries(refSpec.refs)) {
    if (key.endsWith("/zarr.json") && typeof value === "string") {
      metadata[key.slice(0, -"/zarr.json".length)] = JSON.parse(value);
    }
  }

  return {
    ...root,
    consolidated_metadata: {
      kind: "inline",
      must_understand: false,
      metadata,
    },
  };
}

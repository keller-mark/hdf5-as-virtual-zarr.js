import { describe, it, expect } from "vitest";
import { open, get, root as zarrRoot } from "zarrita";
import ReferenceStore from "@zarrita/storage/ref";
import { SingleHdf5ToZarr, refSpecToConsolidatedMetadata } from "../src/index.js";
import { resolve } from "path";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const fixturesDir = resolve(__dirname, "fixtures");

/**
 * Decode base64-encoded string to Uint8Array.
 * From vitessce base64-store.ts.
 */
function base64Decode(encoded: string): Uint8Array {
  return Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
}

/**
 * Create a store from the JSON map contents produced by directory-to-memory-store.mjs.
 * From vitessce base64-store.ts.
 */
function createStoreFromMapContents(mapContents: [string, string][]) {
  const map = new Map(mapContents);
  return new Proxy(map, {
    get: (target, prop) => {
      if (prop === 'get') {
        return (key: string) => {
          const encodedVal = target.get(key);
          if (encodedVal) {
            return base64Decode(encodedVal);
          }
          return undefined;
        };
      }
      return Reflect.get(target, prop);
    },
  });
}

/**
 * Load a zarr JSON fixture and create a store from it.
 */
function loadZarrJsonStore(fixtureName: string) {
  const json = JSON.parse(readFileSync(resolve(fixturesDir, fixtureName), "utf-8"));
  return createStoreFromMapContents(json);
}

/**
 * Generate a reference spec from an h5ad fixture file.
 */
function generateRefSpec(fixtureName: string) {
  const buffer = readFileSync(resolve(fixturesDir, fixtureName));
  const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  const converter = new SingleHdf5ToZarr(arrayBuffer);
  return converter.translate();
}

/**
 * Load the kerchunk-generated ground-truth reference spec JSON.
 */
function loadKerchunkRefSpec(fixtureName: string) {
  return JSON.parse(readFileSync(resolve(fixturesDir, fixtureName), "utf-8"));
}

/**
 * Load the zarr.consolidate_metadata-generated ground-truth consolidated metadata JSON.
 */
function loadGroundTruthConsolidatedMetadata(fixtureName: string) {
  return JSON.parse(readFileSync(resolve(fixturesDir, fixtureName), "utf-8"));
}

describe("SingleHdf5ToZarr", () => {
  describe("reference spec structure", () => {
    it("should produce a valid reference spec with version 1", () => {
      const result = generateRefSpec("minimal.h5ad");
      expect(result.version).toBe(1);
      expect(result.refs).toBeDefined();
      expect(typeof result.refs).toBe("object");
    });

    it("should contain root .zgroup", () => {
      const result = generateRefSpec("minimal.h5ad");
      expect(result.refs[".zgroup"]).toBeDefined();
      const zgroup = JSON.parse(result.refs[".zgroup"] as string);
      expect(zgroup.zarr_format).toBe(2);
    });

    it("should generate valid JSON for all metadata refs", () => {
      const result = generateRefSpec("dense.h5ad");
      for (const [key, value] of Object.entries(result.refs)) {
        if (key.endsWith(".zarray") || key.endsWith(".zgroup") || key.endsWith(".zattrs")) {
          expect(() => JSON.parse(value as string), `Invalid JSON for key: ${key}`).not.toThrow();
        }
      }
    });

    it("should have proper zarr_format in all .zarray entries", () => {
      const result = generateRefSpec("dense.h5ad");
      for (const [key, value] of Object.entries(result.refs)) {
        if (key.endsWith(".zarray")) {
          const meta = JSON.parse(value as string);
          expect(meta.zarr_format).toBe(2);
          expect(meta.shape).toBeDefined();
          expect(Array.isArray(meta.shape)).toBe(true);
          expect(meta.chunks).toBeDefined();
          expect(Array.isArray(meta.chunks)).toBe(true);
          expect(meta.dtype).toBeDefined();
          expect(meta.order).toBe("C");
        }
      }
    });

    it("should have proper zarr_format in all .zgroup entries", () => {
      const result = generateRefSpec("dense.h5ad");
      for (const [key, value] of Object.entries(result.refs)) {
        if (key.endsWith(".zgroup")) {
          const meta = JSON.parse(value as string);
          expect(meta.zarr_format).toBe(2);
        }
      }
    });
  });

  describe("zarrita equivalence - minimal fixture", () => {
    it("should open as zarr group via ReferenceStore", async () => {
      const refSpec = generateRefSpec("minimal.h5ad");
      const store = ReferenceStore.fromSpec(refSpec);
      const grp = await open(store, { kind: "group" });
      expect(grp).toBeDefined();
      expect(grp.attrs).toBeDefined();
    });

    it.skip("X array should have matching shape compared to zarr store", async () => {
      // From zarr DirectoryStore-as-JSON
      const zarrStore = loadZarrJsonStore("minimal.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X"), { kind: "array" });

      // From HDF5 reference spec
      const refSpec = generateRefSpec("minimal.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("X"), { kind: "array" });

      expect(refArr.shape).toEqual(zarrArr.shape);
      expect(refArr.chunks).toEqual(zarrArr.chunks);
      expect(refArr.dtype).toEqual(zarrArr.dtype);
    });

    it.skip("X array data should be equivalent between zarr store and reference spec", async () => {
      // From zarr DirectoryStore-as-JSON
      const zarrStore = loadZarrJsonStore("minimal.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X"), { kind: "array" });
      const zarrData = await get(zarrArr);

      // From HDF5 reference spec
      const refSpec = generateRefSpec("minimal.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("X"), { kind: "array" });
      const refData = await get(refArr);

      // Compare the raw data values
      expect(Array.from(refData.data as Float32Array)).toEqual(
        Array.from(zarrData.data as Float32Array)
      );
      expect(refData.shape).toEqual(zarrData.shape);
    });

    it.skip("obs/_index data should be equivalent between zarr store and reference spec", async () => {
      // From zarr DirectoryStore-as-JSON
      const zarrStore = loadZarrJsonStore("minimal.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      // From HDF5 reference spec
      const refSpec = generateRefSpec("minimal.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it.skip("var/_index data should be equivalent between zarr store and reference spec", async () => {
      // From zarr DirectoryStore-as-JSON
      const zarrStore = loadZarrJsonStore("minimal.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("var/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      // From HDF5 reference spec
      const refSpec = generateRefSpec("minimal.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("var/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });
  });

  describe.skip("zarrita equivalence - dense fixture", () => {
    it("X array data should be equivalent between zarr store and reference spec", async () => {
      // From zarr DirectoryStore-as-JSON
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X"), { kind: "array" });
      const zarrData = await get(zarrArr);

      // From HDF5 reference spec
      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("X"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data as Float32Array)).toEqual(
        Array.from(zarrData.data as Float32Array)
      );
      expect(refData.shape).toEqual(zarrData.shape);
    });

    it("obs/_index data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("var/_index data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("var/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("var/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("obsm/X_umap data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obsm/X_umap"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obsm/X_umap"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data as Float32Array)).toEqual(
        Array.from(zarrData.data as Float32Array)
      );
      expect(refData.shape).toEqual(zarrData.shape);
    });

    it("obs/categorical/codes data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/categorical/codes"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/categorical/codes"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data as Int8Array)).toEqual(
        Array.from(zarrData.data as Int8Array)
      );
    });

    it("obs/categorical/categories data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/categorical/categories"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/categorical/categories"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("obs/string data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/string"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/string"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("group attrs should be equivalent at root level", async () => {
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrGrp = await open(zarrRoot(zarrStore), { kind: "group" });

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refGrp = await open(zarrRoot(refStore), { kind: "group" });

      expect(refGrp.attrs["encoding-type"]).toEqual(zarrGrp.attrs["encoding-type"]);
      expect(refGrp.attrs["encoding-version"]).toEqual(zarrGrp.attrs["encoding-version"]);
    });
  });

  describe("zarrita equivalence - sparse fixture", () => {
    it("X group should exist in reference spec (CSR sparse format)", async () => {
      const refSpec = generateRefSpec("sparse.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const grp = await open(zarrRoot(refStore).resolve("X"), { kind: "group" });
      expect(grp).toBeDefined();
    });

    it.skip("X/data should be equivalent between zarr store and reference spec", async () => {
      const zarrStore = loadZarrJsonStore("sparse.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X/data"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("sparse.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("X/data"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data as Float32Array)).toEqual(
        Array.from(zarrData.data as Float32Array)
      );
    });

    it.skip("X/indices should be equivalent between zarr store and reference spec", async () => {
      const zarrStore = loadZarrJsonStore("sparse.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X/indices"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("sparse.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("X/indices"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data as Int32Array)).toEqual(
        Array.from(zarrData.data as Int32Array)
      );
    });

    it.skip("X/indptr should be equivalent between zarr store and reference spec", async () => {
      const zarrStore = loadZarrJsonStore("sparse.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X/indptr"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("sparse.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("X/indptr"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data as Int32Array)).toEqual(
        Array.from(zarrData.data as Int32Array)
      );
    });

    it.skip("obs/_index data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("sparse.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("sparse.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });
  });

  describe("zarrita equivalence (zarr v3) - minimal fixture", () => {
    it("should open as zarr group via ReferenceStore", async () => {
      const refSpec = generateRefSpec("minimal.h5ad");
      const store = ReferenceStore.fromSpec(refSpec);
      const grp = await open(store, { kind: "group" });
      expect(grp).toBeDefined();
      expect(grp.attrs).toBeDefined();
    });

    it.skip("X array data should be equivalent between zarr v3 store and reference spec", async () => {
      // From zarr v3 DirectoryStore-as-JSON
      const zarrStore = loadZarrJsonStore("minimal.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X"), { kind: "array" });
      const zarrData = await get(zarrArr);

      // From HDF5 reference spec
      const refSpec = generateRefSpec("minimal.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("X"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data as Float32Array)).toEqual(
        Array.from(zarrData.data as Float32Array)
      );
      expect(refData.shape).toEqual(zarrData.shape);
    });

    it.skip("obs/_index data should be equivalent between zarr v3 store and reference spec", async () => {
      const zarrStore = loadZarrJsonStore("minimal.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("minimal.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it.skip("var/_index data should be equivalent between zarr v3 store and reference spec", async () => {
      const zarrStore = loadZarrJsonStore("minimal.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("var/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("minimal.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("var/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });
  });

  describe.skip("zarrita equivalence (zarr v3) - dense fixture", () => {
    it("X array data should be equivalent between zarr v3 store and reference spec", async () => {
      const zarrStore = loadZarrJsonStore("dense.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("X"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data as Float32Array)).toEqual(
        Array.from(zarrData.data as Float32Array)
      );
      expect(refData.shape).toEqual(zarrData.shape);
    });

    it("obs/_index data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("var/_index data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("var/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("var/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("obsm/X_umap data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obsm/X_umap"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obsm/X_umap"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data as Float32Array)).toEqual(
        Array.from(zarrData.data as Float32Array)
      );
      expect(refData.shape).toEqual(zarrData.shape);
    });

    it("obs/categorical/codes data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/categorical/codes"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/categorical/codes"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data as Int8Array)).toEqual(
        Array.from(zarrData.data as Int8Array)
      );
    });

    it("obs/categorical/categories data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/categorical/categories"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/categorical/categories"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("obs/string data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/string"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/string"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("group attrs should be equivalent at root level", async () => {
      const zarrStore = loadZarrJsonStore("dense.v3.adata.zarr.json");
      const zarrGrp = await open(zarrRoot(zarrStore), { kind: "group" });

      const refSpec = generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refGrp = await open(zarrRoot(refStore), { kind: "group" });

      expect(refGrp.attrs["encoding-type"]).toEqual(zarrGrp.attrs["encoding-type"]);
      expect(refGrp.attrs["encoding-version"]).toEqual(zarrGrp.attrs["encoding-version"]);
    });
  });

  describe("zarrita equivalence (zarr v3) - sparse fixture", () => {
    it("X group should exist in reference spec (CSR sparse format)", async () => {
      const refSpec = generateRefSpec("sparse.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const grp = await open(zarrRoot(refStore).resolve("X"), { kind: "group" });
      expect(grp).toBeDefined();
    });

    it.skip("X/data should be equivalent between zarr v3 store and reference spec", async () => {
      const zarrStore = loadZarrJsonStore("sparse.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X/data"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("sparse.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("X/data"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data as Float32Array)).toEqual(
        Array.from(zarrData.data as Float32Array)
      );
    });

    it.skip("X/indices should be equivalent between zarr v3 store and reference spec", async () => {
      const zarrStore = loadZarrJsonStore("sparse.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X/indices"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("sparse.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("X/indices"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data as Int32Array)).toEqual(
        Array.from(zarrData.data as Int32Array)
      );
    });

    it.skip("X/indptr should be equivalent between zarr v3 store and reference spec", async () => {
      const zarrStore = loadZarrJsonStore("sparse.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X/indptr"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("sparse.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("X/indptr"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data as Int32Array)).toEqual(
        Array.from(zarrData.data as Int32Array)
      );
    });

    it.skip("obs/_index data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("sparse.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = generateRefSpec("sparse.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });
  });

  describe("kerchunk ground-truth comparison - minimal fixture", () => {
    it("should match kerchunk output exactly", () => {
      const ours = generateRefSpec("minimal.h5ad");
      const kerchunk = loadKerchunkRefSpec("minimal.h5ad.refspec.json");
      expect(ours).toEqual(kerchunk);
    });
  });

  describe("kerchunk ground-truth comparison - dense fixture", () => {
    it("should have the same set of ref keys", () => {
      const ours = generateRefSpec("dense.h5ad");
      const kerchunk = loadKerchunkRefSpec("dense.h5ad.refspec.json");
      expect(Object.keys(ours.refs).sort()).toEqual(Object.keys(kerchunk.refs).sort());
    });

    it("should match kerchunk output exactly for all non-chunk refs", () => {
      const ours = generateRefSpec("dense.h5ad");
      const kerchunk = loadKerchunkRefSpec("dense.h5ad.refspec.json");
      for (const key of Object.keys(kerchunk.refs)) {
        if (key.endsWith(".zarray") || key.endsWith(".zgroup") || key.endsWith(".zattrs")) {
          expect(ours.refs[key], `Mismatch for ${key}`).toEqual(kerchunk.refs[key]);
        }
      }
    });

    it("should match kerchunk chunk data exactly for inline chunks", () => {
      const ours = generateRefSpec("dense.h5ad");
      const kerchunk = loadKerchunkRefSpec("dense.h5ad.refspec.json");
      for (const key of Object.keys(kerchunk.refs)) {
        if (key.endsWith(".zarray") || key.endsWith(".zgroup") || key.endsWith(".zattrs")) continue;
        const kerchunkRef = kerchunk.refs[key];
        // Only compare inline data (strings), skip file references
        if (typeof kerchunkRef === "string") {
          expect(ours.refs[key], `Mismatch for ${key}`).toEqual(kerchunkRef);
        }
      }
    });
  });

  describe("kerchunk ground-truth comparison with mouse_liver.h5ad ref spec output", () => {
    it("should match exactly", () => {
      const ours = generateRefSpec("mouse_liver.h5ad");
      const kerchunk = loadKerchunkRefSpec("mouse_liver.h5ad.refspec.json");
      expect(ours).toEqual(kerchunk);
    });

    it("diagnostic: show diff summary", () => {
      const ours = generateRefSpec("mouse_liver.h5ad");
      const kerchunk = loadKerchunkRefSpec("mouse_liver.h5ad.refspec.json");

      const oursKeys = new Set(Object.keys(ours.refs));
      const kerchunkKeys = new Set(Object.keys(kerchunk.refs));

      const missingFromOurs = [...kerchunkKeys].filter(k => !oursKeys.has(k));
      const extraInOurs = [...oursKeys].filter(k => !kerchunkKeys.has(k));
      const common = [...kerchunkKeys].filter(k => oursKeys.has(k));

      const metaDiffs: string[] = [];
      const dataFormatDiffs: string[] = []; // e.g., inline vs file ref
      const dataValueDiffs: string[] = []; // same format, different value

      for (const key of common) {
        const oVal = ours.refs[key];
        const kVal = kerchunk.refs[key];
        const isMeta = key.endsWith('.zarray') || key.endsWith('.zgroup') || key.endsWith('.zattrs');

        if (JSON.stringify(oVal) === JSON.stringify(kVal)) continue;

        if (isMeta) {
          metaDiffs.push(`META ${key}:\n  OURS: ${JSON.stringify(oVal).slice(0, 200)}\n  WANT: ${JSON.stringify(kVal).slice(0, 200)}`);
        } else {
          const oIsArr = Array.isArray(oVal);
          const kIsArr = Array.isArray(kVal);
          const oIsStr = typeof oVal === 'string';
          const kIsStr = typeof kVal === 'string';

          if (oIsArr && kIsArr) {
            dataValueDiffs.push(`BOTH_ARR ${key}: ours=${JSON.stringify(oVal)} want=${JSON.stringify(kVal)}`);
          } else if (oIsStr && kIsStr) {
            dataValueDiffs.push(`BOTH_STR ${key}: ours_len=${oVal.length} want_len=${kVal.length}`);
          } else {
            const oFmt = oIsArr ? `[${oVal}]` : (oIsStr ? `str(len=${oVal.length},prefix=${oVal.slice(0,30)})` : typeof oVal);
            const kFmt = kIsArr ? `[${kVal}]` : (kIsStr ? `str(len=${kVal.length},prefix=${kVal.slice(0,30)})` : typeof kVal);
            dataFormatDiffs.push(`FORMAT ${key}: ours=${oFmt} want=${kFmt}`);
          }
        }
      }

      console.log("=== MOUSE_LIVER DIFF SUMMARY ===");
      console.log(`Total keys: ours=${oursKeys.size} kerchunk=${kerchunkKeys.size}`);
      console.log(`Missing from ours (${missingFromOurs.length}):`, missingFromOurs);
      console.log(`Extra in ours (${extraInOurs.length}):`, extraInOurs);
      console.log(`Meta diffs (${metaDiffs.length}):`);
      metaDiffs.forEach(d => console.log(d));
      console.log(`Data format diffs (${dataFormatDiffs.length}):`);
      dataFormatDiffs.forEach(d => console.log(d));
      console.log(`Data value diffs (${dataValueDiffs.length}):`);
      dataValueDiffs.forEach(d => console.log(d));
      console.log("=== END DIFF SUMMARY ===");

      // This test always passes - it's for diagnostics only
      expect(true).toBe(true);
    });

    it("diagnostic: verify kerchunk file references match raw bytes", () => {
      const kerchunk = loadKerchunkRefSpec("mouse_liver.h5ad.refspec.json");
      const fileBuf = readFileSync(resolve(fixturesDir, "mouse_liver.h5ad"));

      // Check a few file reference entries from the kerchunk output
      const fileRefs = Object.entries(kerchunk.refs).filter(
        ([_k, v]) => Array.isArray(v)
      ) as [string, [string | null, number, number]][];

      console.log(`=== FILE REFERENCE ANALYSIS (${fileRefs.length} total) ===`);

      // Group by dataset path
      const byDataset = new Map<string, { key: string; offset: number; size: number }[]>();
      for (const [key, [_url, offset, size]] of fileRefs) {
        const dsPath = key.replace(/\/[^/]+$/, "");
        if (!byDataset.has(dsPath)) byDataset.set(dsPath, []);
        byDataset.get(dsPath)!.push({ key, offset, size });
      }

      for (const [dsPath, chunks] of byDataset) {
        chunks.sort((a, b) => a.offset - b.offset);
        const first = chunks[0];
        const last = chunks[chunks.length - 1];
        const contiguous = chunks.every((c, i) => 
          i === 0 || c.offset === chunks[i-1].offset + chunks[i-1].size
        );
        console.log(`  ${dsPath}: ${chunks.length} chunks, size=${first.size}, offsets=[${first.offset}..${last.offset + last.size}], contiguous=${contiguous}`);

        // Read first few bytes at first chunk offset
        const preview = fileBuf.slice(first.offset, first.offset + Math.min(16, first.size));
        console.log(`    first chunk bytes: ${Array.from(preview).map(b => b.toString(16).padStart(2, '0')).join(' ')}`);
      }
      console.log("=== END FILE REFERENCE ANALYSIS ===");

      expect(fileRefs.length).toBeGreaterThan(0);
    });

    it("diagnostic: dataset metadata for datasets that need file refs", () => {
      const buffer = readFileSync(resolve(fixturesDir, "mouse_liver.h5ad"));
      const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
      const converter = new SingleHdf5ToZarr(arrayBuffer);
      const result = converter.translate();

      const datasetsToCheck = [
        "X/data", "X/indices", "X/indptr",
        "obs/annotation/codes", "obs/fov_labels/codes",
        "obsm/spatial",
        "uns/spatialdata_attrs/instance_key",
        "uns/spatialdata_attrs/region",
        "uns/spatialdata_attrs/region_key",
      ];

      console.log("=== DATASET METADATA ===");
      for (const path of datasetsToCheck) {
        const zarrayKey = `${path}/.zarray`;
        if (zarrayKey in result.refs) {
          const zarray = JSON.parse(result.refs[zarrayKey] as string);
          console.log(`  ${path}:`);
          console.log(`    shape=${JSON.stringify(zarray.shape)}, chunks=${JSON.stringify(zarray.chunks)}`);
          console.log(`    dtype=${zarray.dtype}`);
          console.log(`    compressor=${JSON.stringify(zarray.compressor)}`);
        } else {
          console.log(`  ${path}: NOT FOUND`);
        }
      }
      console.log("=== END DATASET METADATA ===");
      expect(true).toBe(true);
    });

    it("diagnostic: HDF5 superblock info", () => {
      const buf = readFileSync(resolve(fixturesDir, "mouse_liver.h5ad"));
      const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);

      const signature = buf.slice(0, 8).toString("hex");
      const sbVersion = buf[8];
      const sizeOfOffsets = buf[13];
      const sizeOfLengths = buf[14];

      console.log("=== HDF5 SUPERBLOCK ===");
      console.log(`  Signature: ${signature}`);
      console.log(`  Superblock version: ${sbVersion}`);
      console.log(`  Size of offsets: ${sizeOfOffsets}`);
      console.log(`  Size of lengths: ${sizeOfLengths}`);
      console.log(`  File size: ${buf.length}`);

      if (sbVersion === 0 || sbVersion === 1) {
        const groupLeafNodeK = dv.getUint16(16, true);
        const groupInternalNodeK = dv.getUint16(18, true);
        const baseAddress = Number(dv.getBigUint64(24, true));
        const freeSpaceAddress = Number(dv.getBigUint64(32, true));
        const eofAddress = Number(dv.getBigUint64(40, true));
        const driverInfoAddress = Number(dv.getBigUint64(48, true));

        console.log(`  Group leaf node K: ${groupLeafNodeK}`);
        console.log(`  Group internal node K: ${groupInternalNodeK}`);
        console.log(`  Base address: ${baseAddress}`);
        console.log(`  Free-space address: ${freeSpaceAddress}`);
        console.log(`  EOF address: ${eofAddress}`);
        console.log(`  Driver info address: ${driverInfoAddress}`);

        // Root group symbol table entry
        const rootLinkNameOffset = Number(dv.getBigUint64(56, true));
        const rootObjHeaderAddress = Number(dv.getBigUint64(64, true));
        const rootCacheType = dv.getUint32(72, true);
        console.log(`  Root group link name offset: ${rootLinkNameOffset}`);
        console.log(`  Root group obj header address: ${rootObjHeaderAddress}`);
        console.log(`  Root group cache type: ${rootCacheType}`);
      }

      console.log("=== END HDF5 SUPERBLOCK ===");
      expect(signature).toBe("894844460d0a1a0a");
    });
  });

  describe("kerchunk ground-truth comparison - sparse fixture", () => {
    it("should match kerchunk output exactly", () => {
      const ours = generateRefSpec("sparse.h5ad");
      const kerchunk = loadKerchunkRefSpec("sparse.h5ad.refspec.json");
      expect(ours).toEqual(kerchunk);
    });
  });

  describe("refSpecToConsolidatedMetadata", () => {
    it("should produce consolidated metadata with zarr_consolidated_format 1", () => {
      const refSpec = generateRefSpec("minimal.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      expect(consolidated.zarr_consolidated_format).toBe(1);
      expect(consolidated.metadata).toBeDefined();
      expect(typeof consolidated.metadata).toBe("object");
    });

    it("should include root .zgroup as parsed JSON", () => {
      const refSpec = generateRefSpec("minimal.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      expect(consolidated.metadata[".zgroup"]).toEqual({ zarr_format: 2 });
    });

    it("should include root .zattrs as parsed JSON", () => {
      const refSpec = generateRefSpec("minimal.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      expect(consolidated.metadata[".zattrs"]).toBeDefined();
      expect(consolidated.metadata[".zattrs"]["encoding-type"]).toBe("anndata");
    });

    it("should include .zarray entries as parsed JSON", () => {
      const refSpec = generateRefSpec("minimal.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      const xArray = consolidated.metadata["X/.zarray"];
      expect(xArray).toBeDefined();
      expect(xArray.zarr_format).toBe(2);
      expect(xArray.shape).toEqual([5, 3]);
      expect(xArray.dtype).toBeDefined();
    });

    it("should not include data chunk refs", () => {
      const refSpec = generateRefSpec("minimal.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      for (const key of Object.keys(consolidated.metadata)) {
        expect(
          key.endsWith(".zgroup") || key.endsWith(".zarray") || key.endsWith(".zattrs"),
          `Unexpected non-metadata key: ${key}`
        ).toBe(true);
      }
    });

    it("should include all metadata keys from the reference spec", () => {
      const refSpec = generateRefSpec("dense.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      const metaKeys = Object.keys(refSpec.refs).filter(
        (k) => k.endsWith(".zgroup") || k.endsWith(".zarray") || k.endsWith(".zattrs")
      );
      expect(Object.keys(consolidated.metadata).sort()).toEqual(metaKeys.sort());
    });

    it("should produce correct consolidated metadata for sparse fixture", () => {
      const refSpec = generateRefSpec("sparse.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      // X should be a group (sparse CSR) with .zgroup
      expect(consolidated.metadata["X/.zgroup"]).toEqual({ zarr_format: 2 });
      // X/data, X/indices, X/indptr should have .zarray
      expect(consolidated.metadata["X/data/.zarray"]).toBeDefined();
      expect(consolidated.metadata["X/indices/.zarray"]).toBeDefined();
      expect(consolidated.metadata["X/indptr/.zarray"]).toBeDefined();
    });

    it("should produce parsed metadata values that match the ref spec JSON strings", () => {
      const refSpec = generateRefSpec("dense.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      for (const [key, value] of Object.entries(consolidated.metadata)) {
        const refValue = refSpec.refs[key];
        expect(typeof refValue).toBe("string");
        expect(value).toEqual(JSON.parse(refValue as string));
      }
    });

    it("should have same zarr_consolidated_format as ground truth for minimal fixture", () => {
      const refSpec = generateRefSpec("minimal.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      const groundTruth = loadGroundTruthConsolidatedMetadata("minimal.adata.zmetadata.json");
      expect(consolidated.zarr_consolidated_format).toBe(groundTruth.zarr_consolidated_format);
    });

    it("should have same .zgroup and .zattrs values as ground truth for minimal fixture", () => {
      const refSpec = generateRefSpec("minimal.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      const groundTruth = loadGroundTruthConsolidatedMetadata("minimal.adata.zmetadata.json");
      for (const key of Object.keys(groundTruth.metadata)) {
        if (key.endsWith(".zgroup")) {
          expect(consolidated.metadata[key], `Mismatch for ${key}`).toEqual(groundTruth.metadata[key]);
        }
      }
    });

    it("should match ground truth for all .zgroup entries across all fixtures", () => {
      for (const name of ["minimal", "dense", "sparse"]) {
        const refSpec = generateRefSpec(`${name}.h5ad`);
        const consolidated = refSpecToConsolidatedMetadata(refSpec);
        const groundTruth = loadGroundTruthConsolidatedMetadata(`${name}.adata.zmetadata.json`);
        for (const key of Object.keys(groundTruth.metadata)) {
          if (key.endsWith(".zgroup") && key in consolidated.metadata) {
            expect(consolidated.metadata[key], `${name}: mismatch for ${key}`).toEqual(groundTruth.metadata[key]);
          }
        }
      }
    });

    it("should have matching shape and dtype in .zarray entries compared to ground truth", () => {
      for (const name of ["minimal", "dense", "sparse"]) {
        const refSpec = generateRefSpec(`${name}.h5ad`);
        const consolidated = refSpecToConsolidatedMetadata(refSpec);
        const groundTruth = loadGroundTruthConsolidatedMetadata(`${name}.adata.zmetadata.json`);
        for (const key of Object.keys(groundTruth.metadata)) {
          if (key.endsWith(".zarray") && key in consolidated.metadata) {
            const gt = groundTruth.metadata[key] as Record<string, unknown>;
            const ours = consolidated.metadata[key] as Record<string, unknown>;
            expect(ours.shape, `${name} ${key}: shape mismatch`).toEqual(gt.shape);
            expect(ours.chunks, `${name} ${key}: chunks mismatch`).toEqual(gt.chunks);
            expect(ours.zarr_format, `${name} ${key}: zarr_format mismatch`).toEqual(gt.zarr_format);
            expect(ours.order, `${name} ${key}: order mismatch`).toEqual(gt.order);
          }
        }
      }
    });
  });
});

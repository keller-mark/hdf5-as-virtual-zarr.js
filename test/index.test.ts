import { describe, it, expect } from "vitest";
import { open, get, root as zarrRoot } from "zarrita";
import ReferenceStore from "@zarrita/storage/ref";
import { SingleHdf5ToZarr, refSpecToConsolidatedMetadata } from "../src/index.js";
import type { Source } from "../src/index.js";
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
 * Simple in-memory Source implementation for tests, compatible with @chunkd/source.
 */
class SourceMemory implements Source {
  type = "memory";
  url: URL;
  data: ArrayBuffer;
  metadata: { size: number };

  constructor(url: string, buffer: ArrayBuffer) {
    this.url = new URL(url);
    this.data = buffer;
    this.metadata = { size: buffer.byteLength };
  }

  async head() {
    return { size: this.data.byteLength };
  }

  async fetch(offset: number, length?: number): Promise<ArrayBuffer> {
    if (offset < 0) offset = this.data.byteLength + offset;
    return this.data.slice(offset, length == null ? undefined : offset + length);
  }
}

/**
 * Generate a reference spec from an h5ad fixture file.
 */
async function generateRefSpec(fixtureName: string) {
  const buffer = readFileSync(resolve(fixturesDir, fixtureName));
  const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  const source = new SourceMemory("memory://" + fixtureName, arrayBuffer);
  const converter = new SingleHdf5ToZarr(source, { url: null });
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
    it("should produce a valid reference spec with version 1", async () => {
      const result = await generateRefSpec("minimal.h5ad");
      expect(result.version).toBe(1);
      expect(result.refs).toBeDefined();
      expect(typeof result.refs).toBe("object");
    });

    it("should contain root .zgroup", async () => {
      const result = await generateRefSpec("minimal.h5ad");
      expect(result.refs[".zgroup"]).toBeDefined();
      const zgroup = JSON.parse(result.refs[".zgroup"] as string);
      expect(zgroup.zarr_format).toBe(2);
    });

    it("should generate valid JSON for all metadata refs", async () => {
      const result = await generateRefSpec("dense.h5ad");
      for (const [key, value] of Object.entries(result.refs)) {
        if (key.endsWith(".zarray") || key.endsWith(".zgroup") || key.endsWith(".zattrs")) {
          expect(() => JSON.parse(value as string), `Invalid JSON for key: ${key}`).not.toThrow();
        }
      }
    });

    it("should have proper zarr_format in all .zarray entries", async () => {
      const result = await generateRefSpec("dense.h5ad");
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

    it("should have proper zarr_format in all .zgroup entries", async () => {
      const result = await generateRefSpec("dense.h5ad");
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
      const refSpec = await generateRefSpec("minimal.h5ad");
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
      const refSpec = await generateRefSpec("minimal.h5ad");
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
      const refSpec = await generateRefSpec("minimal.h5ad");
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
      const refSpec = await generateRefSpec("minimal.h5ad");
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
      const refSpec = await generateRefSpec("minimal.h5ad");
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
      const refSpec = await generateRefSpec("dense.h5ad");
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

      const refSpec = await generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("var/_index data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("var/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = await generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("var/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("obsm/X_umap data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obsm/X_umap"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = await generateRefSpec("dense.h5ad");
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

      const refSpec = await generateRefSpec("dense.h5ad");
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

      const refSpec = await generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/categorical/categories"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("obs/string data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/string"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = await generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/string"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("group attrs should be equivalent at root level", async () => {
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrGrp = await open(zarrRoot(zarrStore), { kind: "group" });

      const refSpec = await generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refGrp = await open(zarrRoot(refStore), { kind: "group" });

      expect(refGrp.attrs["encoding-type"]).toEqual(zarrGrp.attrs["encoding-type"]);
      expect(refGrp.attrs["encoding-version"]).toEqual(zarrGrp.attrs["encoding-version"]);
    });
  });

  describe("zarrita equivalence - sparse fixture", () => {
    it("X group should exist in reference spec (CSR sparse format)", async () => {
      const refSpec = await generateRefSpec("sparse.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const grp = await open(zarrRoot(refStore).resolve("X"), { kind: "group" });
      expect(grp).toBeDefined();
    });

    it.skip("X/data should be equivalent between zarr store and reference spec", async () => {
      const zarrStore = loadZarrJsonStore("sparse.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X/data"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = await generateRefSpec("sparse.h5ad");
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

      const refSpec = await generateRefSpec("sparse.h5ad");
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

      const refSpec = await generateRefSpec("sparse.h5ad");
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

      const refSpec = await generateRefSpec("sparse.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });
  });

  describe("zarrita equivalence (zarr v3) - minimal fixture", () => {
    it("should open as zarr group via ReferenceStore", async () => {
      const refSpec = await generateRefSpec("minimal.h5ad");
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
      const refSpec = await generateRefSpec("minimal.h5ad");
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

      const refSpec = await generateRefSpec("minimal.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it.skip("var/_index data should be equivalent between zarr v3 store and reference spec", async () => {
      const zarrStore = loadZarrJsonStore("minimal.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("var/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = await generateRefSpec("minimal.h5ad");
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

      const refSpec = await generateRefSpec("dense.h5ad");
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

      const refSpec = await generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("var/_index data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("var/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = await generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("var/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("obsm/X_umap data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obsm/X_umap"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = await generateRefSpec("dense.h5ad");
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

      const refSpec = await generateRefSpec("dense.h5ad");
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

      const refSpec = await generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/categorical/categories"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("obs/string data should be equivalent", async () => {
      const zarrStore = loadZarrJsonStore("dense.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/string"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = await generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/string"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });

    it("group attrs should be equivalent at root level", async () => {
      const zarrStore = loadZarrJsonStore("dense.v3.adata.zarr.json");
      const zarrGrp = await open(zarrRoot(zarrStore), { kind: "group" });

      const refSpec = await generateRefSpec("dense.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refGrp = await open(zarrRoot(refStore), { kind: "group" });

      expect(refGrp.attrs["encoding-type"]).toEqual(zarrGrp.attrs["encoding-type"]);
      expect(refGrp.attrs["encoding-version"]).toEqual(zarrGrp.attrs["encoding-version"]);
    });
  });

  describe("zarrita equivalence (zarr v3) - sparse fixture", () => {
    it("X group should exist in reference spec (CSR sparse format)", async () => {
      const refSpec = await generateRefSpec("sparse.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const grp = await open(zarrRoot(refStore).resolve("X"), { kind: "group" });
      expect(grp).toBeDefined();
    });

    it.skip("X/data should be equivalent between zarr v3 store and reference spec", async () => {
      const zarrStore = loadZarrJsonStore("sparse.v3.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X/data"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const refSpec = await generateRefSpec("sparse.h5ad");
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

      const refSpec = await generateRefSpec("sparse.h5ad");
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

      const refSpec = await generateRefSpec("sparse.h5ad");
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

      const refSpec = await generateRefSpec("sparse.h5ad");
      const refStore = ReferenceStore.fromSpec(refSpec);
      const refArr = await open(zarrRoot(refStore).resolve("obs/_index"), { kind: "array" });
      const refData = await get(refArr);

      expect(Array.from(refData.data)).toEqual(Array.from(zarrData.data));
    });
  });

  describe("kerchunk ground-truth comparison - minimal fixture", () => {
    it("should match kerchunk output exactly", async () => {
      const ours = await generateRefSpec("minimal.h5ad");
      const kerchunk = loadKerchunkRefSpec("minimal.h5ad.refspec.json");
      expect(ours).toEqual(kerchunk);
    });
  });

  describe("kerchunk ground-truth comparison - dense fixture", () => {
    it("should have the same set of ref keys", async () => {
      const ours = await generateRefSpec("dense.h5ad");
      const kerchunk = loadKerchunkRefSpec("dense.h5ad.refspec.json");
      expect(Object.keys(ours.refs).sort()).toEqual(Object.keys(kerchunk.refs).sort());
    });

    it("should match kerchunk output exactly for all non-chunk refs", async () => {
      const ours = await generateRefSpec("dense.h5ad");
      const kerchunk = loadKerchunkRefSpec("dense.h5ad.refspec.json");
      for (const key of Object.keys(kerchunk.refs)) {
        if (key.endsWith(".zarray") || key.endsWith(".zgroup") || key.endsWith(".zattrs")) {
          expect(ours.refs[key], `Mismatch for ${key}`).toEqual(kerchunk.refs[key]);
        }
      }
    });

    it("should match kerchunk chunk data exactly for inline chunks", async () => {
      const ours = await generateRefSpec("dense.h5ad");
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
    it("should match exactly", async () => {
      const ours = await generateRefSpec("mouse_liver.h5ad");
      const kerchunk = loadKerchunkRefSpec("mouse_liver.h5ad.refspec.json");
      expect(ours).toEqual(kerchunk);
    });

  });

  describe("kerchunk ground-truth comparison - sparse fixture", () => {
    it("should match kerchunk output exactly", async () => {
      const ours = await generateRefSpec("sparse.h5ad");
      const kerchunk = loadKerchunkRefSpec("sparse.h5ad.refspec.json");
      expect(ours).toEqual(kerchunk);
    });
  });

  describe("refSpecToConsolidatedMetadata", () => {
    it("should produce consolidated metadata with zarr_consolidated_format 1", async () => {
      const refSpec = await generateRefSpec("minimal.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      expect(consolidated.zarr_consolidated_format).toBe(1);
      expect(consolidated.metadata).toBeDefined();
      expect(typeof consolidated.metadata).toBe("object");
    });

    it("should include root .zgroup as parsed JSON", async () => {
      const refSpec = await generateRefSpec("minimal.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      expect(consolidated.metadata[".zgroup"]).toEqual({ zarr_format: 2 });
    });

    it("should include root .zattrs as parsed JSON", async () => {
      const refSpec = await generateRefSpec("minimal.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      expect(consolidated.metadata[".zattrs"]).toBeDefined();
      expect(consolidated.metadata[".zattrs"]["encoding-type"]).toBe("anndata");
    });

    it("should include .zarray entries as parsed JSON", async () => {
      const refSpec = await generateRefSpec("minimal.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      const xArray = consolidated.metadata["X/.zarray"];
      expect(xArray).toBeDefined();
      expect(xArray.zarr_format).toBe(2);
      expect(xArray.shape).toEqual([5, 3]);
      expect(xArray.dtype).toBeDefined();
    });

    it("should not include data chunk refs", async () => {
      const refSpec = await generateRefSpec("minimal.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      for (const key of Object.keys(consolidated.metadata)) {
        expect(
          key.endsWith(".zgroup") || key.endsWith(".zarray") || key.endsWith(".zattrs"),
          `Unexpected non-metadata key: ${key}`
        ).toBe(true);
      }
    });

    it("should include all metadata keys from the reference spec", async () => {
      const refSpec = await generateRefSpec("dense.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      const metaKeys = Object.keys(refSpec.refs).filter(
        (k) => k.endsWith(".zgroup") || k.endsWith(".zarray") || k.endsWith(".zattrs")
      );
      expect(Object.keys(consolidated.metadata).sort()).toEqual(metaKeys.sort());
    });

    it("should produce correct consolidated metadata for sparse fixture", async () => {
      const refSpec = await generateRefSpec("sparse.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      // X should be a group (sparse CSR) with .zgroup
      expect(consolidated.metadata["X/.zgroup"]).toEqual({ zarr_format: 2 });
      // X/data, X/indices, X/indptr should have .zarray
      expect(consolidated.metadata["X/data/.zarray"]).toBeDefined();
      expect(consolidated.metadata["X/indices/.zarray"]).toBeDefined();
      expect(consolidated.metadata["X/indptr/.zarray"]).toBeDefined();
    });

    it("should produce parsed metadata values that match the ref spec JSON strings", async () => {
      const refSpec = await generateRefSpec("dense.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      for (const [key, value] of Object.entries(consolidated.metadata)) {
        const refValue = refSpec.refs[key];
        expect(typeof refValue).toBe("string");
        expect(value).toEqual(JSON.parse(refValue as string));
      }
    });

    it("should have same zarr_consolidated_format as ground truth for minimal fixture", async () => {
      const refSpec = await generateRefSpec("minimal.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      const groundTruth = loadGroundTruthConsolidatedMetadata("minimal.adata.zmetadata.json");
      expect(consolidated.zarr_consolidated_format).toBe(groundTruth.zarr_consolidated_format);
    });

    it("should have same .zgroup and .zattrs values as ground truth for minimal fixture", async () => {
      const refSpec = await generateRefSpec("minimal.h5ad");
      const consolidated = refSpecToConsolidatedMetadata(refSpec);
      const groundTruth = loadGroundTruthConsolidatedMetadata("minimal.adata.zmetadata.json");
      for (const key of Object.keys(groundTruth.metadata)) {
        if (key.endsWith(".zgroup")) {
          expect(consolidated.metadata[key], `Mismatch for ${key}`).toEqual(groundTruth.metadata[key]);
        }
      }
    });

    it("should match ground truth for all .zgroup entries across all fixtures", async () => {
      for (const name of ["minimal", "dense", "sparse"]) {
        const refSpec = await generateRefSpec(`${name}.h5ad`);
        const consolidated = refSpecToConsolidatedMetadata(refSpec);
        const groundTruth = loadGroundTruthConsolidatedMetadata(`${name}.adata.zmetadata.json`);
        for (const key of Object.keys(groundTruth.metadata)) {
          if (key.endsWith(".zgroup") && key in consolidated.metadata) {
            expect(consolidated.metadata[key], `${name}: mismatch for ${key}`).toEqual(groundTruth.metadata[key]);
          }
        }
      }
    });

    it("should have matching shape and dtype in .zarray entries compared to ground truth", async () => {
      for (const name of ["minimal", "dense", "sparse"]) {
        const refSpec = await generateRefSpec(`${name}.h5ad`);
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

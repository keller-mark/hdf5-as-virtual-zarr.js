import { describe, it, expect, beforeAll } from "vitest";
import * as h5wasm from "h5wasm/node";
import { open, get, root as zarrRoot } from "zarrita";
import ReferenceStore from "@zarrita/storage/ref";
import { SingleHdf5ToZarr } from "../src/index.js";
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
  const file = new h5wasm.File(resolve(fixturesDir, fixtureName), "r");
  try {
    const converter = new SingleHdf5ToZarr(file);
    return converter.translate();
  } finally {
    file.close();
  }
}

/**
 * Load the kerchunk-generated ground-truth reference spec JSON.
 */
function loadKerchunkRefSpec(fixtureName: string) {
  return JSON.parse(readFileSync(resolve(fixturesDir, fixtureName), "utf-8"));
}

beforeAll(async () => {
  await h5wasm.ready;
});

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

    it("X array should have matching shape compared to zarr store", async () => {
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

    it("X array data should be equivalent between zarr store and reference spec", async () => {
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

    it("obs/_index data should be equivalent between zarr store and reference spec", async () => {
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

    it("var/_index data should be equivalent between zarr store and reference spec", async () => {
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

  describe("zarrita equivalence - dense fixture", () => {
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

    it("X/data should be equivalent between zarr store and reference spec", async () => {
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

    it("X/indices should be equivalent between zarr store and reference spec", async () => {
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

    it("X/indptr should be equivalent between zarr store and reference spec", async () => {
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

    it("obs/_index data should be equivalent", async () => {
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

  describe("kerchunk ground-truth comparison - minimal fixture", () => {
    it("should have the same set of ref keys", () => {
      const ours = generateRefSpec("minimal.h5ad");
      const kerchunk = loadKerchunkRefSpec("minimal.h5ad.refspec.json");
      expect(Object.keys(ours.refs).sort()).toEqual(Object.keys(kerchunk.refs).sort());
    });

    it("should have matching .zgroup metadata for all groups", () => {
      const ours = generateRefSpec("minimal.h5ad");
      const kerchunk = loadKerchunkRefSpec("minimal.h5ad.refspec.json");
      for (const key of Object.keys(kerchunk.refs)) {
        if (key.endsWith(".zgroup")) {
          expect(JSON.parse(ours.refs[key] as string), `Mismatch for ${key}`).toEqual(
            JSON.parse(kerchunk.refs[key])
          );
        }
      }
    });

    it("should have matching .zarray metadata for all arrays", () => {
      const ours = generateRefSpec("minimal.h5ad");
      const kerchunk = loadKerchunkRefSpec("minimal.h5ad.refspec.json");
      for (const key of Object.keys(kerchunk.refs)) {
        if (key.endsWith(".zarray")) {
          const ourMeta = JSON.parse(ours.refs[key] as string);
          const kerchunkMeta = JSON.parse(kerchunk.refs[key]);
          expect(ourMeta.shape, `shape mismatch for ${key}`).toEqual(kerchunkMeta.shape);
          expect(ourMeta.chunks, `chunks mismatch for ${key}`).toEqual(kerchunkMeta.chunks);
          expect(ourMeta.dtype, `dtype mismatch for ${key}`).toEqual(kerchunkMeta.dtype);
          expect(ourMeta.fill_value, `fill_value mismatch for ${key}`).toEqual(kerchunkMeta.fill_value);
          expect(ourMeta.order, `order mismatch for ${key}`).toEqual(kerchunkMeta.order);
          expect(ourMeta.compressor, `compressor mismatch for ${key}`).toEqual(kerchunkMeta.compressor);
          // Filters for |O dtype differ (we use vlen-utf8, kerchunk uses json2) - both valid
          if (kerchunkMeta.dtype !== "|O") {
            expect(ourMeta.filters, `filters mismatch for ${key}`).toEqual(kerchunkMeta.filters);
          }
        }
      }
    });

    it("should have matching .zattrs metadata for all objects", () => {
      const ours = generateRefSpec("minimal.h5ad");
      const kerchunk = loadKerchunkRefSpec("minimal.h5ad.refspec.json");
      for (const key of Object.keys(kerchunk.refs)) {
        if (key.endsWith(".zattrs")) {
          expect(JSON.parse(ours.refs[key] as string), `Mismatch for ${key}`).toEqual(
            JSON.parse(kerchunk.refs[key])
          );
        }
      }
    });

    it("should have matching inline base64 data for numeric arrays", () => {
      const ours = generateRefSpec("minimal.h5ad");
      const kerchunk = loadKerchunkRefSpec("minimal.h5ad.refspec.json");
      // X/0.0 is base64-inlined in both kerchunk and our output
      expect(ours.refs["X/0.0"]).toEqual(kerchunk.refs["X/0.0"]);
    });
  });

  describe("kerchunk ground-truth comparison - dense fixture", () => {
    it("should have the same set of ref keys", () => {
      const ours = generateRefSpec("dense.h5ad");
      const kerchunk = loadKerchunkRefSpec("dense.h5ad.refspec.json");
      expect(Object.keys(ours.refs).sort()).toEqual(Object.keys(kerchunk.refs).sort());
    });

    it("should have matching .zgroup metadata for all groups", () => {
      const ours = generateRefSpec("dense.h5ad");
      const kerchunk = loadKerchunkRefSpec("dense.h5ad.refspec.json");
      for (const key of Object.keys(kerchunk.refs)) {
        if (key.endsWith(".zgroup")) {
          expect(JSON.parse(ours.refs[key] as string), `Mismatch for ${key}`).toEqual(
            JSON.parse(kerchunk.refs[key])
          );
        }
      }
    });

    it("should have matching .zarray metadata for all arrays", () => {
      const ours = generateRefSpec("dense.h5ad");
      const kerchunk = loadKerchunkRefSpec("dense.h5ad.refspec.json");
      for (const key of Object.keys(kerchunk.refs)) {
        if (key.endsWith(".zarray")) {
          const ourMeta = JSON.parse(ours.refs[key] as string);
          const kerchunkMeta = JSON.parse(kerchunk.refs[key]);
          expect(ourMeta.shape, `shape mismatch for ${key}`).toEqual(kerchunkMeta.shape);
          expect(ourMeta.chunks, `chunks mismatch for ${key}`).toEqual(kerchunkMeta.chunks);
          expect(ourMeta.dtype, `dtype mismatch for ${key}`).toEqual(kerchunkMeta.dtype);
          expect(ourMeta.fill_value, `fill_value mismatch for ${key}`).toEqual(kerchunkMeta.fill_value);
          expect(ourMeta.order, `order mismatch for ${key}`).toEqual(kerchunkMeta.order);
          expect(ourMeta.compressor, `compressor mismatch for ${key}`).toEqual(kerchunkMeta.compressor);
          // Filters for |O dtype differ (we use vlen-utf8, kerchunk uses json2) - both valid
          if (kerchunkMeta.dtype !== "|O") {
            expect(ourMeta.filters, `filters mismatch for ${key}`).toEqual(kerchunkMeta.filters);
          }
        }
      }
    });

    it("should have matching .zattrs metadata for all objects", () => {
      const ours = generateRefSpec("dense.h5ad");
      const kerchunk = loadKerchunkRefSpec("dense.h5ad.refspec.json");
      for (const key of Object.keys(kerchunk.refs)) {
        if (key.endsWith(".zattrs")) {
          expect(JSON.parse(ours.refs[key] as string), `Mismatch for ${key}`).toEqual(
            JSON.parse(kerchunk.refs[key])
          );
        }
      }
    });

    it("should have matching chunk references for numeric arrays", () => {
      const ours = generateRefSpec("dense.h5ad");
      const kerchunk = loadKerchunkRefSpec("dense.h5ad.refspec.json");
      for (const key of Object.keys(kerchunk.refs)) {
        if (key.endsWith(".zarray") || key.endsWith(".zgroup") || key.endsWith(".zattrs")) continue;
        const parts = key.split("/");
        const chunkPart = parts.pop()!;
        if (!/^[\d.]+$/.test(chunkPart)) continue;
        const arrayPath = parts.join("/");
        const zarrayKey = `${arrayPath}/.zarray`;
        const zarrayMeta = kerchunk.refs[zarrayKey];
        if (!zarrayMeta) continue;
        const meta = JSON.parse(zarrayMeta);
        if (meta.dtype === "|O") continue;
        // Both should have an entry for this chunk
        expect(ours.refs[key], `Missing chunk ${key}`).toBeDefined();
        // If kerchunk used an inline format (string), compare decoded bytes
        const kerchunkRef = kerchunk.refs[key];
        const ourRef = ours.refs[key];
        if (typeof kerchunkRef === "string" && typeof ourRef === "string") {
          // Both kerchunk and our output may use different inline encodings:
          // kerchunk uses raw binary strings, we use "base64:" prefix.
          // Decode both to bytes and compare.
          const ourBytes = typeof ourRef === "string" && ourRef.startsWith("base64:")
            ? Buffer.from(ourRef.slice(7), "base64")
            : Buffer.from(ourRef, "binary");
          const kerchunkBytes = typeof kerchunkRef === "string" && kerchunkRef.startsWith("base64:")
            ? Buffer.from(kerchunkRef.slice(7), "base64")
            : Buffer.from(kerchunkRef, "binary");
          expect(
            Array.from(ourBytes),
            `Byte mismatch for chunk ${key}`
          ).toEqual(Array.from(kerchunkBytes));
        }
        // If kerchunk used [url, offset, length], just verify our entry exists and is valid
        if (Array.isArray(kerchunkRef)) {
          expect(
            typeof ourRef === "string" || Array.isArray(ourRef),
            `Invalid chunk ref format for ${key}`
          ).toBe(true);
        }
      }
    });
  });

  describe("kerchunk ground-truth comparison - sparse fixture", () => {
    it("should have the same set of ref keys", () => {
      const ours = generateRefSpec("sparse.h5ad");
      const kerchunk = loadKerchunkRefSpec("sparse.h5ad.refspec.json");
      expect(Object.keys(ours.refs).sort()).toEqual(Object.keys(kerchunk.refs).sort());
    });

    it("should have matching .zgroup metadata for all groups", () => {
      const ours = generateRefSpec("sparse.h5ad");
      const kerchunk = loadKerchunkRefSpec("sparse.h5ad.refspec.json");
      for (const key of Object.keys(kerchunk.refs)) {
        if (key.endsWith(".zgroup")) {
          expect(JSON.parse(ours.refs[key] as string), `Mismatch for ${key}`).toEqual(
            JSON.parse(kerchunk.refs[key])
          );
        }
      }
    });

    it("should have matching .zarray metadata for all arrays", () => {
      const ours = generateRefSpec("sparse.h5ad");
      const kerchunk = loadKerchunkRefSpec("sparse.h5ad.refspec.json");
      for (const key of Object.keys(kerchunk.refs)) {
        if (key.endsWith(".zarray")) {
          const ourMeta = JSON.parse(ours.refs[key] as string);
          const kerchunkMeta = JSON.parse(kerchunk.refs[key]);
          expect(ourMeta.shape, `shape mismatch for ${key}`).toEqual(kerchunkMeta.shape);
          expect(ourMeta.chunks, `chunks mismatch for ${key}`).toEqual(kerchunkMeta.chunks);
          expect(ourMeta.dtype, `dtype mismatch for ${key}`).toEqual(kerchunkMeta.dtype);
          expect(ourMeta.fill_value, `fill_value mismatch for ${key}`).toEqual(kerchunkMeta.fill_value);
          expect(ourMeta.order, `order mismatch for ${key}`).toEqual(kerchunkMeta.order);
          expect(ourMeta.compressor, `compressor mismatch for ${key}`).toEqual(kerchunkMeta.compressor);
          // Filters for |O dtype differ (we use vlen-utf8, kerchunk uses json2) - both valid
          if (kerchunkMeta.dtype !== "|O") {
            expect(ourMeta.filters, `filters mismatch for ${key}`).toEqual(kerchunkMeta.filters);
          }
        }
      }
    });

    it("should have matching .zattrs metadata for all objects", () => {
      const ours = generateRefSpec("sparse.h5ad");
      const kerchunk = loadKerchunkRefSpec("sparse.h5ad.refspec.json");
      for (const key of Object.keys(kerchunk.refs)) {
        if (key.endsWith(".zattrs")) {
          expect(JSON.parse(ours.refs[key] as string), `Mismatch for ${key}`).toEqual(
            JSON.parse(kerchunk.refs[key])
          );
        }
      }
    });

    it("should have matching chunk references for numeric arrays", () => {
      const ours = generateRefSpec("sparse.h5ad");
      const kerchunk = loadKerchunkRefSpec("sparse.h5ad.refspec.json");
      for (const key of Object.keys(kerchunk.refs)) {
        if (key.endsWith(".zarray") || key.endsWith(".zgroup") || key.endsWith(".zattrs")) continue;
        const parts = key.split("/");
        const chunkPart = parts.pop()!;
        if (!/^[\d.]+$/.test(chunkPart)) continue;
        const arrayPath = parts.join("/");
        const zarrayKey = `${arrayPath}/.zarray`;
        const zarrayMeta = kerchunk.refs[zarrayKey];
        if (!zarrayMeta) continue;
        const meta = JSON.parse(zarrayMeta);
        if (meta.dtype === "|O") continue;
        // Both should have an entry for this chunk
        expect(ours.refs[key], `Missing chunk ${key}`).toBeDefined();
        // If kerchunk used an inline format (string), compare decoded bytes
        const kerchunkRef = kerchunk.refs[key];
        const ourRef = ours.refs[key];
        if (typeof kerchunkRef === "string" && typeof ourRef === "string") {
          // Both kerchunk and our output may use different inline encodings:
          // kerchunk uses raw binary strings, we use "base64:" prefix.
          // Decode both to bytes and compare.
          const ourBytes = typeof ourRef === "string" && ourRef.startsWith("base64:")
            ? Buffer.from(ourRef.slice(7), "base64")
            : Buffer.from(ourRef, "binary");
          const kerchunkBytes = typeof kerchunkRef === "string" && kerchunkRef.startsWith("base64:")
            ? Buffer.from(kerchunkRef.slice(7), "base64")
            : Buffer.from(kerchunkRef, "binary");
          expect(
            Array.from(ourBytes),
            `Byte mismatch for chunk ${key}`
          ).toEqual(Array.from(kerchunkBytes));
        }
        // If kerchunk used [url, offset, length], just verify our entry exists and is valid
        if (Array.isArray(kerchunkRef)) {
          expect(
            typeof ourRef === "string" || Array.isArray(ourRef),
            `Invalid chunk ref format for ${key}`
          ).toBe(true);
        }
      }
    });
  });
});

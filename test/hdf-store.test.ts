import { describe, it, expect } from "vitest";
import { open, get, root as zarrRoot, withConsolidatedMetadata } from "zarrita";
import { HdfStore } from "../src/hdf-store.js";
import { SingleHdf5ToZarr } from "../src/single-hdf5-to-zarr.js";
import { ComposableReferenceStore } from "../src/ref-store.js";
import type { AsyncReadable } from "../src/types.js";
import { resolve } from "path";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const fixturesDir = resolve(__dirname, "fixtures");

class SourceMemory implements AsyncReadable {
  data: ArrayBuffer;

  constructor(_url: string, buffer: ArrayBuffer) {
    this.data = buffer;
  }

  async get() {
    return new Uint8Array(this.data);
  }

  async getRange(_key: string, range: { offset: number; length: number } | { suffixLength: number }): Promise<Uint8Array> {
    let offset: number;
    let length: number;
    if ("suffixLength" in range) {
      offset = this.data.byteLength - range.suffixLength;
      length = range.suffixLength;
    } else {
      offset = range.offset;
      length = range.length;
    }
    return new Uint8Array(this.data, offset, length);
  }
}

function loadH5adSource(fixtureName: string): SourceMemory {
  const buffer = readFileSync(resolve(fixturesDir, fixtureName));
  const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  return new SourceMemory("memory://" + fixtureName, arrayBuffer);
}

function base64Decode(encoded: string): Uint8Array {
  return Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
}

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

function loadZarrJsonStore(fixtureName: string) {
  const json = JSON.parse(readFileSync(resolve(fixturesDir, fixtureName), "utf-8"));
  return createStoreFromMapContents(json);
}

describe("HdfStore", () => {
  describe("fromStore", () => {
    it("should return an HdfStore instance", async () => {
      const source = loadH5adSource("minimal.h5ad");
      const store = await HdfStore.fromStore(source);
      expect(store).toBeInstanceOf(HdfStore);
    });

    it("should return undefined for a missing key", async () => {
      const source = loadH5adSource("minimal.h5ad");
      const store = await HdfStore.fromStore(source);
      const result = await store.get("/nonexistent/key");
      expect(result).toBeUndefined();
    });

    it("should return Uint8Array for a known metadata key", async () => {
      const source = loadH5adSource("minimal.h5ad");
      const store = await HdfStore.fromStore(source);
      const result = await store.get("/zarr.json");
      expect(result).toBeInstanceOf(Uint8Array);
    });

    it("should return valid v3 group JSON for root zarr.json key", async () => {
      const source = loadH5adSource("minimal.h5ad");
      const store = await HdfStore.fromStore(source);
      const result = await store.get("/zarr.json");
      expect(result).toBeDefined();
      const parsed = JSON.parse(new TextDecoder().decode(result));
      expect(parsed.zarr_format).toBe(3);
      expect(parsed.node_type).toBe("group");
    });

    it("should not expose v2 metadata keys", async () => {
      const source = loadH5adSource("minimal.h5ad");
      const store = await HdfStore.fromStore(source);
      expect(await store.get("/.zgroup")).toBeUndefined();
      expect(await store.get("/X/.zarray")).toBeUndefined();
    });

    it("should return v3 array metadata with dimension names and v3 chunk keys", async () => {
      const source = loadH5adSource("dense.h5ad");
      const store = await HdfStore.fromStore(source);
      const result = await store.get("/X/zarr.json");
      const parsed = JSON.parse(new TextDecoder().decode(result));
      expect(parsed.zarr_format).toBe(3);
      expect(parsed.node_type).toBe("array");
      expect(parsed.data_type).toBe("float32");
      expect(parsed.dimension_names).toEqual(["phony_dim_0", "phony_dim_1"]);
      expect(parsed.attributes._ARRAY_DIMENSIONS).toBeUndefined();
      expect(parsed.chunk_key_encoding).toEqual({ name: "default", configuration: { separator: "/" } });
      expect(await store.get("/X/c/0/0")).toBeInstanceOf(Uint8Array);
      expect(await store.get("/X/0.0")).toBeUndefined();
    });

    it("should expose string arrays as v3 string data type with vlen-utf8", async () => {
      const source = loadH5adSource("dense.h5ad");
      const store = await HdfStore.fromStore(source);
      const result = await store.get("/obs/_index/zarr.json");
      const parsed = JSON.parse(new TextDecoder().decode(result));
      expect(parsed.data_type).toBe("string");
      expect(parsed.codecs).toEqual([{ name: "vlen-utf8" }]);
      expect(parsed.chunk_grid.configuration.chunk_shape).toEqual(parsed.shape);
    });
  });

  describe("getConsolidatedMetadata", () => {
    it("should return the root group with inline metadata for every node", async () => {
      const source = loadH5adSource("dense.h5ad");
      const store = await HdfStore.fromStore(source);
      const consolidated = store.getConsolidatedMetadata();
      const root = JSON.parse(new TextDecoder().decode(await store.get("/zarr.json")));
      expect(consolidated.zarr_format).toBe(3);
      expect(consolidated.node_type).toBe("group");
      expect(consolidated.attributes).toEqual(root.attributes);
      expect(consolidated.consolidated_metadata.kind).toBe("inline");
      expect(consolidated.consolidated_metadata.must_understand).toBe(false);
      const { metadata } = consolidated.consolidated_metadata;
      expect(metadata[""]).toBeUndefined();
      const xMeta = JSON.parse(new TextDecoder().decode(await store.get("/X/zarr.json")));
      expect(metadata["X"]).toEqual(xMeta);
      expect(metadata["obs"].node_type).toBe("group");
      expect(metadata["obs/_index"].node_type).toBe("array");
    });

    it("should be readable by zarrita withConsolidatedMetadata", async () => {
      const source = loadH5adSource("dense.h5ad");
      const store = await HdfStore.fromStore(source);
      const consolidated = store.getConsolidatedMetadata();
      const bytes = new TextEncoder().encode(JSON.stringify(consolidated));
      const rootOnly = {
        get: async (key: string) => (key === "/zarr.json" ? bytes : store.get(key as `/${string}`)),
      };
      const cStore = await withConsolidatedMetadata(rootOnly, { format: "v3" });
      expect(cStore.contents().map((c) => c.path)).toContain("/X");
    });
  });

  describe("zarrita equivalence - minimal fixture", () => {
    it("should open as zarr group via HdfStore", async () => {
      const source = loadH5adSource("minimal.h5ad");
      const store = await HdfStore.fromStore(source);
      const grp = await open(store, { kind: "group" });
      expect(grp).toBeDefined();
      expect(grp.attrs).toBeDefined();
    });

    it("X array shape should match zarr store", async () => {
      const zarrStore = loadZarrJsonStore("minimal.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X"), { kind: "array" });

      const source = loadH5adSource("minimal.h5ad");
      const store = await HdfStore.fromStore(source);
      const hdfArr = await open(zarrRoot(store).resolve("X"), { kind: "array" });

      expect(hdfArr.shape).toEqual(zarrArr.shape);
      expect(hdfArr.chunks).toEqual(zarrArr.chunks);
      expect(hdfArr.dtype).toEqual(zarrArr.dtype);
    });

    it("X array data should match zarr store", async () => {
      const zarrStore = loadZarrJsonStore("minimal.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const source = loadH5adSource("minimal.h5ad");
      const store = await HdfStore.fromStore(source);
      const hdfArr = await open(zarrRoot(store).resolve("X"), { kind: "array" });
      const hdfData = await get(hdfArr);

      expect(Array.from(hdfData.data as Float32Array)).toEqual(
        Array.from(zarrData.data as Float32Array)
      );
      expect(hdfData.shape).toEqual(zarrData.shape);
    });

    it("obs/_index data should match zarr store", async () => {
      const zarrStore = loadZarrJsonStore("minimal.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const source = loadH5adSource("minimal.h5ad");
      const store = await HdfStore.fromStore(source);
      const hdfArr = await open(zarrRoot(store).resolve("obs/_index"), { kind: "array" });
      const hdfData = await get(hdfArr);

      expect(Array.from(hdfData.data)).toEqual(Array.from(zarrData.data));
    });

    it("var/_index data should match zarr store", async () => {
      const zarrStore = loadZarrJsonStore("minimal.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("var/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const source = loadH5adSource("minimal.h5ad");
      const store = await HdfStore.fromStore(source);
      const hdfArr = await open(zarrRoot(store).resolve("var/_index"), { kind: "array" });
      const hdfData = await get(hdfArr);

      expect(Array.from(hdfData.data)).toEqual(Array.from(zarrData.data));
    });
  });

  describe("zarrita equivalence - dense fixture", () => {
    it("X array data should match zarr store", async () => {
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const source = loadH5adSource("dense.h5ad");
      const store = await HdfStore.fromStore(source);
      const hdfArr = await open(zarrRoot(store).resolve("X"), { kind: "array" });
      const hdfData = await get(hdfArr);

      expect(Array.from(hdfData.data as Float32Array)).toEqual(
        Array.from(zarrData.data as Float32Array)
      );
    });

    it("obs/_index data should match zarr store", async () => {
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/_index"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const source = loadH5adSource("dense.h5ad");
      const store = await HdfStore.fromStore(source);
      const hdfArr = await open(zarrRoot(store).resolve("obs/_index"), { kind: "array" });
      const hdfData = await get(hdfArr);

      expect(Array.from(hdfData.data)).toEqual(Array.from(zarrData.data));
    });

    it("obs/categorical/codes data should match zarr store", async () => {
      const zarrStore = loadZarrJsonStore("dense.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("obs/categorical/codes"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const source = loadH5adSource("dense.h5ad");
      const store = await HdfStore.fromStore(source);
      const hdfArr = await open(zarrRoot(store).resolve("obs/categorical/codes"), { kind: "array" });
      const hdfData = await get(hdfArr);

      expect(Array.from(hdfData.data as Int8Array)).toEqual(
        Array.from(zarrData.data as Int8Array)
      );
    });
  });

  describe("zarrita equivalence - sparse fixture", () => {
    it("X group should exist (CSR sparse format)", async () => {
      const source = loadH5adSource("sparse.h5ad");
      const store = await HdfStore.fromStore(source);
      const grp = await open(zarrRoot(store).resolve("X"), { kind: "group" });
      expect(grp).toBeDefined();
    });

    it("X/data should match zarr store", async () => {
      const zarrStore = loadZarrJsonStore("sparse.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X/data"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const source = loadH5adSource("sparse.h5ad");
      const store = await HdfStore.fromStore(source);
      const hdfArr = await open(zarrRoot(store).resolve("X/data"), { kind: "array" });
      const hdfData = await get(hdfArr);

      expect(Array.from(hdfData.data as Float32Array)).toEqual(
        Array.from(zarrData.data as Float32Array)
      );
    });

    it("X/indices should match zarr store", async () => {
      const zarrStore = loadZarrJsonStore("sparse.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X/indices"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const source = loadH5adSource("sparse.h5ad");
      const store = await HdfStore.fromStore(source);
      const hdfArr = await open(zarrRoot(store).resolve("X/indices"), { kind: "array" });
      const hdfData = await get(hdfArr);

      expect(Array.from(hdfData.data as Int32Array)).toEqual(
        Array.from(zarrData.data as Int32Array)
      );
    });

    it("X/indptr should match zarr store", async () => {
      const zarrStore = loadZarrJsonStore("sparse.adata.zarr.json");
      const zarrArr = await open(zarrRoot(zarrStore).resolve("X/indptr"), { kind: "array" });
      const zarrData = await get(zarrArr);

      const source = loadH5adSource("sparse.h5ad");
      const store = await HdfStore.fromStore(source);
      const hdfArr = await open(zarrRoot(store).resolve("X/indptr"), { kind: "array" });
      const hdfData = await get(hdfArr);

      expect(Array.from(hdfData.data as Int32Array)).toEqual(
        Array.from(zarrData.data as Int32Array)
      );
    });
  });

  describe("v3 store matches v2 reference spec", () => {
    const fixtures = [
      "minimal.h5ad",
      "dense.h5ad",
      "sparse.h5ad",
      "mouse_liver.h5ad",
      "secondary_analysis.subset.h5ad",
    ];
    for (const fixture of fixtures) {
      it(`every array in ${fixture} has the same data via v2 and v3`, async () => {
        const source = loadH5adSource(fixture);
        const v2Spec = await new SingleHdf5ToZarr(source, { url: null }).translate();
        const v2Store = ComposableReferenceStore.fromSpec(v2Spec as unknown as Record<string, unknown>, source);
        const v3Store = await HdfStore.fromStore(source);

        const arrayPaths = Object.keys(v2Spec.refs)
          .filter((k) => k.endsWith("/.zarray"))
          .map((k) => k.slice(0, -"/.zarray".length));
        expect(arrayPaths.length).toBeGreaterThan(0);

        for (const path of arrayPaths) {
          const v2Arr = await open(zarrRoot(v2Store).resolve(path), { kind: "array" });
          const v3Arr = await open(zarrRoot(v3Store).resolve(path), { kind: "array" });
          expect(v3Arr.shape, path).toEqual(v2Arr.shape);
          const v2Data = await get(v2Arr);
          const v3Data = await get(v3Arr);
          if (v2Arr.shape.length === 0) {
            // zarrita returns the bare value for 0-d arrays.
            expect(v3Data, path).toEqual(v2Data);
            continue;
          }
          expect(Array.from(v3Data.data as ArrayLike<unknown>), path).toEqual(
            Array.from(v2Data.data as ArrayLike<unknown>)
          );
        }
      });
    }
  });
});

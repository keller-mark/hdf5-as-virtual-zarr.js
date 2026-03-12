import { describe, it, expect } from "vitest";
import { open, get, root as zarrRoot } from "zarrita";
import { HdfStore } from "../src/hdf-store.js";
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
      const result = await store.get("/.zgroup");
      expect(result).toBeInstanceOf(Uint8Array);
    });

    it("should return valid JSON for .zgroup key", async () => {
      const source = loadH5adSource("minimal.h5ad");
      const store = await HdfStore.fromStore(source);
      const result = await store.get("/.zgroup");
      expect(result).toBeDefined();
      const text = new TextDecoder().decode(result);
      const parsed = JSON.parse(text);
      expect(parsed.zarr_format).toBe(2);
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
});

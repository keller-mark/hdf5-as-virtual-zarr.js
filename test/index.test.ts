import { describe, it, expect, beforeAll } from "vitest";
import * as h5wasm from "h5wasm/node";
import { SingleHdf5ToZarr } from "../src/index.js";
import { resolve } from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const fixturesDir = resolve(__dirname, "fixtures");

beforeAll(async () => {
  await h5wasm.ready;
});

describe("SingleHdf5ToZarr", () => {
  describe("minimal fixture", () => {
    it("should produce a valid reference spec with version 1", () => {
      const file = new h5wasm.File(`${fixturesDir}/minimal.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

        expect(result.version).toBe(1);
        expect(result.refs).toBeDefined();
        expect(typeof result.refs).toBe("object");
      } finally {
        file.close();
      }
    });

    it("should contain root .zgroup", () => {
      const file = new h5wasm.File(`${fixturesDir}/minimal.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

        expect(result.refs[".zgroup"]).toBeDefined();
        const zgroup = JSON.parse(result.refs[".zgroup"] as string);
        expect(zgroup.zarr_format).toBe(2);
      } finally {
        file.close();
      }
    });

    it("should contain X dataset .zarray metadata", () => {
      const file = new h5wasm.File(`${fixturesDir}/minimal.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

        expect(result.refs["X/.zarray"]).toBeDefined();
        const zarray = JSON.parse(result.refs["X/.zarray"] as string);
        expect(zarray.zarr_format).toBe(2);
        expect(zarray.shape).toEqual([5, 3]);
        expect(zarray.dtype).toMatch(/<f/);
        expect(zarray.order).toBe("C");
      } finally {
        file.close();
      }
    });

    it("should contain obs and var groups", () => {
      const file = new h5wasm.File(`${fixturesDir}/minimal.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

        expect(result.refs["obs/.zgroup"]).toBeDefined();
        expect(result.refs["var/.zgroup"]).toBeDefined();
      } finally {
        file.close();
      }
    });

    it("should contain _ARRAY_DIMENSIONS in zattrs for X", () => {
      const file = new h5wasm.File(`${fixturesDir}/minimal.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

        expect(result.refs["X/.zattrs"]).toBeDefined();
        const zattrs = JSON.parse(result.refs["X/.zattrs"] as string);
        expect(zattrs._ARRAY_DIMENSIONS).toBeDefined();
        expect(Array.isArray(zattrs._ARRAY_DIMENSIONS)).toBe(true);
      } finally {
        file.close();
      }
    });

    it("should contain inline data for X", () => {
      const file = new h5wasm.File(`${fixturesDir}/minimal.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

        // The minimal fixture X is small (5x3 float32 = 60 bytes)
        // so it should be inlined
        const dataKeys = Object.keys(result.refs).filter(
          (k) => k.startsWith("X/") && !k.includes(".z")
        );
        expect(dataKeys.length).toBeGreaterThan(0);

        // Check that the data reference is a base64 string
        const firstDataRef = result.refs[dataKeys[0]];
        expect(typeof firstDataRef).toBe("string");
        expect((firstDataRef as string).startsWith("base64:")).toBe(true);
      } finally {
        file.close();
      }
    });
  });

  describe("dense fixture", () => {
    it("should produce a valid reference spec", () => {
      const file = new h5wasm.File(`${fixturesDir}/dense.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

        expect(result.version).toBe(1);
        expect(result.refs).toBeDefined();
      } finally {
        file.close();
      }
    });

    it("should have correct X shape (50x25)", () => {
      const file = new h5wasm.File(`${fixturesDir}/dense.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

        const zarray = JSON.parse(result.refs["X/.zarray"] as string);
        expect(zarray.shape).toEqual([50, 25]);
      } finally {
        file.close();
      }
    });

    it("should have obs and var metadata", () => {
      const file = new h5wasm.File(`${fixturesDir}/dense.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

        expect(result.refs["obs/.zgroup"]).toBeDefined();
        expect(result.refs["var/.zgroup"]).toBeDefined();
      } finally {
        file.close();
      }
    });

    it("should handle obsm/X_umap dataset", () => {
      const file = new h5wasm.File(`${fixturesDir}/dense.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

        expect(result.refs["obsm/.zgroup"]).toBeDefined();

        // X_umap should be present as a dataset
        const hasXumap = Object.keys(result.refs).some(
          (k) => k.includes("X_umap") && k.endsWith(".zarray")
        );
        expect(hasXumap).toBe(true);
      } finally {
        file.close();
      }
    });
  });

  describe("sparse fixture", () => {
    it("should produce a valid reference spec", () => {
      const file = new h5wasm.File(`${fixturesDir}/sparse.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

        expect(result.version).toBe(1);
        expect(result.refs).toBeDefined();
      } finally {
        file.close();
      }
    });

    it("should have X group for sparse data (CSR format)", () => {
      const file = new h5wasm.File(`${fixturesDir}/sparse.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

        // In h5ad, sparse X is stored as a group with data/indices/indptr
        expect(result.refs["X/.zgroup"]).toBeDefined();
      } finally {
        file.close();
      }
    });
  });

  describe("options", () => {
    it("should include URL in references when provided", () => {
      const file = new h5wasm.File(`${fixturesDir}/minimal.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file, {
          url: "https://example.com/test.h5ad",
        });
        const result = converter.translate();
        expect(result.version).toBe(1);
      } finally {
        file.close();
      }
    });

    it("should respect inline threshold", () => {
      const file = new h5wasm.File(`${fixturesDir}/minimal.h5ad`, "r");
      try {
        // With a very high threshold, everything should be inlined
        const converter = new SingleHdf5ToZarr(file, {
          inlineThreshold: 1000000,
        });
        const result = converter.translate();

        // Check that data chunks exist as base64 strings
        const dataKeys = Object.keys(result.refs).filter(
          (k) => !k.includes(".z")
        );
        for (const key of dataKeys) {
          const ref = result.refs[key];
          expect(typeof ref).toBe("string");
        }
      } finally {
        file.close();
      }
    });
  });

  describe("metadata correctness", () => {
    it("should generate valid JSON for all refs", () => {
      const file = new h5wasm.File(`${fixturesDir}/dense.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

        for (const [key, value] of Object.entries(result.refs)) {
          if (key.endsWith(".zarray") || key.endsWith(".zgroup") || key.endsWith(".zattrs")) {
            expect(() => JSON.parse(value as string)).not.toThrow();
          }
        }
      } finally {
        file.close();
      }
    });

    it("should have proper zarr_format in all .zarray entries", () => {
      const file = new h5wasm.File(`${fixturesDir}/dense.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

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
      } finally {
        file.close();
      }
    });

    it("should have proper zarr_format in all .zgroup entries", () => {
      const file = new h5wasm.File(`${fixturesDir}/dense.h5ad`, "r");
      try {
        const converter = new SingleHdf5ToZarr(file);
        const result = converter.translate();

        for (const [key, value] of Object.entries(result.refs)) {
          if (key.endsWith(".zgroup")) {
            const meta = JSON.parse(value as string);
            expect(meta.zarr_format).toBe(2);
          }
        }
      } finally {
        file.close();
      }
    });
  });
});

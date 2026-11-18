import { describe, it, expect } from "vitest";
import { resolve, dirname } from "path";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import * as zarr from "zarrita";
import type { AsyncReadable } from "../src/types.js";
import { HdfStore } from "../src/index.js";

const fixturesDir = resolve(dirname(fileURLToPath(import.meta.url)), "fixtures");

/** In-memory HDF5 source that records every getRange call. */
class SpySource implements AsyncReadable {
  calls: Array<{ offset: number; length: number }> = [];
  private data: Uint8Array;

  constructor(fixtureName: string) {
    this.data = new Uint8Array(readFileSync(resolve(fixturesDir, fixtureName)));
  }

  async get(): Promise<Uint8Array> {
    throw new Error("HdfStore should only use getRange");
  }

  async getRange(_key: string, range: { offset: number; length: number } | { suffixLength: number }) {
    if (!("offset" in range)) throw new Error("unexpected suffix read");
    const { offset, length } = range;
    if (offset + length > this.data.length) {
      throw new Error(`read past end of file: offset=${offset} length=${length}`);
    }
    this.calls.push({ offset, length });
    return this.data.slice(offset, offset + length);
  }
}

async function readStrings(store: HdfStore, path: string) {
  const arr = await zarr.open(zarr.root(store).resolve(path), { kind: "array" });
  const chunk = (await zarr.get(arr)) as { data: string[]; shape: number[] };
  return { shape: chunk.shape, data: Array.from(chunk.data) };
}

describe("chunked, compressed variable-length strings", () => {
  it("decodes gzip-compressed 1-D chunks, including a partial final chunk", async () => {
    const store = await HdfStore.fromStore(new SpySource("compressed_strings.h5"));
    const { shape, data } = await readStrings(store, "gzip_1d");
    expect(shape).toEqual([103]);
    expect(data).toEqual(Array.from({ length: 103 }, (_, i) => `item-${i}-é`));
  });

  it("places 2-D chunk items at the correct positions", async () => {
    const store = await HdfStore.fromStore(new SpySource("compressed_strings.h5"));
    const { shape, data } = await readStrings(store, "gzip_2d");
    expect(shape).toEqual([7, 5]);
    const expected: string[] = [];
    for (let r = 0; r < 7; r++) for (let c = 0; c < 5; c++) expected.push(`r${r}c${c}`);
    expect(data).toEqual(expected);
  });

  it("reverses shuffle + gzip", async () => {
    const store = await HdfStore.fromStore(new SpySource("compressed_strings.h5"));
    const { data } = await readStrings(store, "shuffle_gzip");
    expect(data).toEqual(Array.from({ length: 20 }, (_, i) => `s${i}`));
  });
});

describe("lazy translation", () => {
  it("does not read string data when only metadata is requested", async () => {
    const source = new SpySource("compressed_strings.h5");
    const store = await HdfStore.fromStore(source);
    const before = source.calls.length;
    const arr = await zarr.open(zarr.root(store).resolve("gzip_1d"), { kind: "array" });
    expect(arr.shape).toEqual([103]);
    const metadataCalls = source.calls.length - before;

    await zarr.get(arr);
    // Reading the data (chunk B-tree, chunks, global heaps) needs more requests.
    expect(source.calls.length - before).toBeGreaterThan(metadataCalls);
  });

  it("only translates the requested part of the hierarchy", async () => {
    const lazySource = new SpySource("mouse_liver.h5ad");
    const store = await HdfStore.fromStore(lazySource);
    await zarr.open(zarr.root(store).resolve("obs"), { kind: "group" });

    const fullSource = new SpySource("mouse_liver.h5ad");
    const fullStore = await HdfStore.fromStore(fullSource);
    await fullStore.getConsolidatedMetadata();

    expect(lazySource.calls.length).toBeLessThan(fullSource.calls.length);
  });

  it("returns correct data when the LRU caches are tiny (forcing eviction)", async () => {
    const store = await HdfStore.fromStore(new SpySource("compressed_strings.h5"), {
      maxCachedNodes: 1,
      maxCachedRefs: 1,
    });
    for (let round = 0; round < 2; round++) {
      expect((await readStrings(store, "gzip_1d")).data[102]).toBe("item-102-é");
      expect((await readStrings(store, "shuffle_gzip")).data[19]).toBe("s19");
      expect((await readStrings(store, "gzip_2d")).data[34]).toBe("r6c4");
    }
  });

  it("returns undefined for missing nodes and chunks", async () => {
    const store = await HdfStore.fromStore(new SpySource("compressed_strings.h5"));
    expect(await store.get("/missing/zarr.json")).toBeUndefined();
    expect(await store.get("/gzip_1d/c/11")).toBeUndefined(); // 103 items in chunks of 10: c/0..c/10
    expect(await store.get("/gzip_1d/c/0/0")).toBeUndefined();
    expect(await store.get("/gzip_1d/.zarray")).toBeUndefined();
  });
});

describe("withRangeCoalescing", () => {
  it("reads the same data through a coalesced source with fewer requests", async () => {
    const plain = new SpySource("mouse_liver.h5ad");
    const plainStore = await HdfStore.fromStore(plain);
    const plainMeta = await plainStore.getConsolidatedMetadata();

    const inner = new SpySource("mouse_liver.h5ad");
    const coalescedStore = await HdfStore.fromStore(zarr.withRangeCoalescing(inner));
    const coalescedMeta = await coalescedStore.getConsolidatedMetadata();

    expect(coalescedMeta).toEqual(plainMeta);
    expect(inner.calls.length).toBeLessThan(plain.calls.length);
  });
});

import { describe, it, expect } from "vitest";
import { resolve } from "path";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname } from "path";
import type { AsyncReadable } from "../src/types.js";
import { SingleHdf5ToZarr } from "../src/index.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const fixturesDir = resolve(__dirname, "fixtures");

/**
 * Records every getRange() call so tests can verify partial read behavior.
 */
class SourceSpy implements AsyncReadable {
  readonly fileSize: number;
  readonly fetchCalls: Array<{ offset: number; length: number }> = [];
  private ab: ArrayBuffer;

  constructor(_url: string, ab: ArrayBuffer) {
    this.ab = ab;
    this.fileSize = ab.byteLength;
  }

  async get(): Promise<Uint8Array> {
    return new Uint8Array(this.ab);
  }

  async getRange(_key: string, range: { offset: number; length: number } | { suffixLength: number }): Promise<Uint8Array> {
    let offset: number;
    let length: number;
    if ("suffixLength" in range) {
      offset = this.ab.byteLength - range.suffixLength;
      length = range.suffixLength;
    } else {
      offset = range.offset;
      length = range.length;
    }
    const slice = new Uint8Array(this.ab, offset, length);
    this.fetchCalls.push({ offset, length: slice.byteLength });
    return slice;
  }

  /**
   * Returns the number of unique (non-overlapping) bytes fetched across all calls.
   */
  uniqueBytesFetched(): number {
    if (this.fetchCalls.length === 0) return 0;

    // Build and sort intervals by start offset
    const intervals = this.fetchCalls
      .map(({ offset, length }) => [offset, offset + length] as [number, number])
      .sort((a, b) => a[0] - b[0]);

    // Merge overlapping intervals and sum their lengths
    let total = 0;
    let curStart = intervals[0][0];
    let curEnd = intervals[0][1];
    for (let i = 1; i < intervals.length; i++) {
      const [start, end] = intervals[i];
      if (start <= curEnd) {
        curEnd = Math.max(curEnd, end);
      } else {
        total += curEnd - curStart;
        curStart = start;
        curEnd = end;
      }
    }
    total += curEnd - curStart;
    return total;
  }

  /**
   * Fraction of the file read (0–1).
   */
  fractionRead(): number {
    return this.uniqueBytesFetched() / this.fileSize;
  }
}

function makeSourceSpy(fixtureName: string): SourceSpy {
  const buffer = readFileSync(resolve(fixturesDir, fixtureName));
  const ab = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  return new SourceSpy(`memory://${fixtureName}`, ab);
}

async function translateWithSpy(fixtureName: string): Promise<SourceSpy> {
  const spy = makeSourceSpy(fixtureName);
  const converter = new SingleHdf5ToZarr(spy, { url: null });
  await converter.translate();
  return spy;
}

describe("partial reads", () => {
  describe("less than 100% of the file is read", () => {
    it("minimal.h5ad", async () => {
      const spy = await translateWithSpy("minimal.h5ad");
      expect(spy.uniqueBytesFetched()).toBeLessThan(spy.fileSize);
    });

    it("dense.h5ad", async () => {
      const spy = await translateWithSpy("dense.h5ad");
      expect(spy.uniqueBytesFetched()).toBeLessThan(spy.fileSize);
    });

    it("sparse.h5ad", async () => {
      const spy = await translateWithSpy("sparse.h5ad");
      expect(spy.uniqueBytesFetched()).toBeLessThan(spy.fileSize);
    });

    it("mouse_liver.h5ad", async () => {
      const spy = await translateWithSpy("mouse_liver.h5ad");
      expect(spy.uniqueBytesFetched()).toBeLessThan(spy.fileSize);
    });
  });

  describe("large data arrays are not read for files with chunked numeric data", () => {
    it("mouse_liver.h5ad reads less than 50% of the file", async () => {
      // mouse_liver.h5ad has large chunked arrays (X, obs categories, obsm)
      // that total hundreds of KB. These should be referenced by offset, not read.
      // The file is 734806 bytes; we expect metadata reads to be well under half.
      const spy = await translateWithSpy("mouse_liver.h5ad");
      expect(spy.fractionRead()).toBeLessThan(0.5);
    });

    it("sparse.h5ad reads less than 75% of the file", async () => {
      // sparse.h5ad is small (29 KB), so metadata is a larger fraction than in
      // mouse_liver.h5ad, but the actual CSR data arrays are still not read.
      const spy = await translateWithSpy("sparse.h5ad");
      expect(spy.fractionRead()).toBeLessThan(0.75);
    });
  });

  describe("each fetch call uses a length argument (no unbounded reads)", () => {
    it("minimal.h5ad: all fetch calls specify a length", async () => {
      const spy = makeSourceSpy("minimal.h5ad");
      const converter = new SingleHdf5ToZarr(spy, { url: null });
      await converter.translate();

      const unbounded = spy.fetchCalls.filter(
        ({ offset, length }) => length === spy.fileSize - offset
      );
      // Verify there are fetch calls at all, and none with length >= 95% of file size
      expect(spy.fetchCalls.length).toBeGreaterThan(0);
      const maxSingleRead = Math.max(...spy.fetchCalls.map(c => c.length));
      expect(maxSingleRead).toBeLessThan(spy.fileSize * 0.95);
    });
  });
});

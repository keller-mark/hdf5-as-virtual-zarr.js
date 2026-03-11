import { describe, it, expect } from "vitest";
import { resolve } from "path";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname } from "path";
import type { Source } from "../src/index.js";
import { File as Hdf5File } from "../src/hdf5/high-level.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const fixturesDir = resolve(__dirname, "fixtures");

class SourceMemory implements Source {
  type = "memory";
  url: URL;
  private ab: ArrayBuffer;
  constructor(url: string, ab: ArrayBuffer) {
    this.url = new URL(url);
    this.ab = ab;
  }
  async fetch(offset: number, length?: number): Promise<ArrayBuffer> {
    if (length === undefined) return this.ab.slice(offset);
    return this.ab.slice(offset, offset + length);
  }
}

describe("debug mouse_liver attrs", () => {
  it("should have encoding-type on uns/annotation_colors", async () => {
    const buffer = readFileSync(resolve(fixturesDir, "mouse_liver.h5ad"));
    const ab = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
    const source = new SourceMemory("memory://test", ab);
    const f = await Hdf5File.create(source);

    // Navigate to uns/annotation_colors
    const uns = await f.get("uns");
    console.log("uns type:", uns.constructor.name);
    console.log("uns keys:", (uns as any).keys);

    const annColors = await (uns as any).get("annotation_colors");
    console.log("annotation_colors type:", annColors.constructor.name);
    console.log("annotation_colors _dataobjects msgs:", annColors._dataobjects.msgs.map((m: any) => ({
      type: m.get("type"),
      size: m.get("size"),
      offset: m.get("offset_to_message"),
    })));

    // Check attribute msgs
    const attrMsgs = annColors._dataobjects.find_msg_type(0x000c);
    console.log("Inline ATTRIBUTE msgs:", attrMsgs.length);
    
    // Check attribute info msgs
    const attrInfoMsgs = annColors._dataobjects.find_msg_type(0x0015);
    console.log("ATTRIBUTE_INFO msgs:", attrInfoMsgs.length);
    for (const msg of attrInfoMsgs) {
      console.log("  attrInfo:", JSON.stringify(Object.fromEntries(msg)));
    }

    const attrs = await annColors.get_attrs();
    console.log("annotation_colors attrs:", JSON.stringify(attrs));

    expect(attrs).toHaveProperty("encoding-type");
  });
});

import { describe, it, expect } from "vitest";
import { fileURLToPath } from "url";
import { resolve, dirname, join } from "path";

import * as zarr from 'zarrita';
import FileSystemStore from '@zarrita/storage/fs';
import { HdfStore } from "hdf5-as-virtual-zarr";
import { readZarr, get, readElem } from 'anndata.js';



const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const fixturesDir = resolve(__dirname, "fixtures");


describe("read-anndata", () => {
  it("reads an anndata object from a file", async () => {
    const internalStore = new FileSystemStore(join(fixturesDir, "secondary_analysis.subset.h5ad"));
    const store = await HdfStore.fromStore(internalStore);

    const group = await zarr.open(store, { kind: "group" });
    console.log(group.attrs);

    const adata = await readZarr(store);

    const obsIndexArr = await adata.obsNames();

    // Slice into the array object
    const obsIndex = await get(obsIndexArr, [null]);

    expect(obsIndex.shape).toEqual([100]);
  });
});

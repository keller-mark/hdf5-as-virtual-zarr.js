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


/**
 * Error thrown when an assertion fails.
 */
export class AssertionError extends Error {
  /** @param message The error message. */
  constructor(message: string) {
    super(message);
    this.name = "AssertionError";
  }
}

/**
 * Make an assertion.
 *
 * Usage
 * @example
 * ```ts
 * const value: boolean = Math.random() <= 0.5;
 * assert(value, "value is greater than than 0.5!");
 * value // true
 * ```
 *
 * @param expr The expression to test.
 * @param msg The optional message to display if the assertion fails.
 * @throws an {@link Error} if `expression` is not truthy.
 *
 * @copyright Trevor Manz 2026
 * @license MIT
 * @see {@link https://github.com/manzt/manzt/blob/HASH/utils/assert_verbose.ts}
 */
export function assert(expr: unknown, msg = ""): asserts expr {
	if (!expr) throw new AssertionError(msg);
}



describe("read-anndata", () => {
  it("reads an anndata object from a file", async () => {
    const internalStore = new FileSystemStore(join(fixturesDir, "secondary_analysis.subset.h5ad"));
    const store = await HdfStore.fromStore(internalStore);

    const group = await zarr.open(store, { kind: "group" });
    console.log(group.attrs);

    expect(group.attrs).toEqual({
      'encoding-type': 'anndata',
      'encoding-version': '0.1.0'
    })

    const adata = await readZarr(store);

    const obsIndexArr = await adata.obsNames();

    expect(obsIndexArr.shape).toEqual([100]);

    // Slice into the array object
    const obsIndex = await get(obsIndexArr, [null]);

    expect(obsIndex.shape).toEqual([100]);
    expect(obsIndex.data.length).toEqual(100);

    expect(Array.from(obsIndex.data).slice(0, 4)).toEqual([
      'AAAAAAAAAAATAC', 'AAAAAAAAACGAAA', 'AAAAACCCCGGCCT', 'AAAAAGAGTGACCG'
    ]);

    const varIndex = await get(await adata.varNames(), [null]);
    expect(varIndex.shape).toEqual([50]);
    expect(varIndex.data.length).toEqual(50);
    expect(Array.from(varIndex.data).slice(0, 4)).toEqual([
      "ENSG00000000003.15",
      "ENSG00000000419.13",
      "ENSG00000000457.14",
      "ENSG00000000460.17",
    ]);

    const obsGroup = await adata.obs.axisRoot();
    expect(obsGroup.attrs).toEqual({
      _index: '_index',
      'column-order': [
        'n_genes',
        'n_counts',
        'leiden',
        'umap_density',
        'azimuth_label',
        'azimuth_id',
        'predicted_CLID',
        'predicted_label',
        'cl_match_type',
        'prediction_score'
      ],
      'encoding-type': 'dataframe',
      'encoding-version': '0.2.0',
    });

    const predictedLabel = await adata.obs.get("predicted_label");

    expect("codes" in predictedLabel).toEqual(true);

    const predictedLabelCodes = "codes" in predictedLabel ? predictedLabel.codes : null;
    const predictedLabelCats = "categories" in predictedLabel ? predictedLabel.categories : null

    expect(predictedLabelCodes?.shape).toEqual([100]);
    expect(predictedLabelCats?.shape).toEqual([16]);

    const nCountsArr = await adata.obs.get("n_counts");


    assert(!("codes" in nCountsArr), "Was unexpectedly categorical");
    assert(!("indptr" in nCountsArr), "Was unexpectedly sparse");
    assert(!("axisRoot" in nCountsArr), "Was unexpectedly axis_arrays");

    expect(nCountsArr.shape).toEqual([100]);

    console.log(nCountsArr.attrs);

    const nCounts = await get(nCountsArr, [null]);

    expect(nCounts.data.length).toEqual(100);

  });
});

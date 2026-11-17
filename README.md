# hdf5-as-virtual-zarr.js

[![NPM](https://img.shields.io/npm/v/hdf5-as-virtual-zarr.svg?color=black)](https://www.npmjs.com/package/hdf5-as-virtual-zarr)

Given an input HDF5 file URL, generate a Zarr References Specification JSON output.

This TypeScript implementation can run in a web browser, and generates output that is compatible with the [ReferenceStore](https://github.com/manzt/zarrita.js/blob/d511ea44551a62c2af64ba33672672e9286490da/packages/%40zarrita-storage/src/ref.ts#L42) from [zarrita.js](https://github.com/manzt/zarrita.js).

It supports large HDF5 files using range requests (fetching only the subset of bytes required). When possible, large arrays should not be accessed; only the minimal metadata should be accessed.

This is a port of [SingleHdf5ToZarr](https://github.com/fsspec/kerchunk/blob/main/kerchunk/hdf.py#L52) and its `translate` function from the `kerchunk` Python package.

## Usage

```ts
import { open, get, root, withRangeCoalescing } from "zarrita";
import { FetchStore } from "@zarrita/storage";
import { HdfStore } from "hdf5-as-virtual-zarr";

const internalStore = withRangeCoalescing(new FetchStore("https://example.com/data.h5"));
const hdfStore = await HdfStore.fromStore(internalStore);

const storeRoot = root(hdfStore);
const arr = await open(storeRoot.resolve("X"), { kind: "array" });
const arrData = await get(arr);
```

### Assumption: the source is wrapped with `withRangeCoalescing`

`HdfStore` assumes the store passed to `HdfStore.fromStore` is wrapped with zarrita's
[`withRangeCoalescing`](https://github.com/manzt/zarrita.js), as in the example above.
Reading HDF5 metadata involves many small range reads (object headers, B-tree nodes, heaps).
`HdfStore` issues independent reads concurrently (sibling groups and datasets, B-tree children,
global heap collections, chunks) so that `withRangeCoalescing` can merge reads made in the same
microtask into a few HTTP requests. `HdfStore` keeps no byte cache of its own, so it also reads
a little ahead (clamped to the end of the file) to avoid follow-up requests for adjacent bytes.
Without `withRangeCoalescing` the store still works, but it makes many more requests.

Translation is lazy:

- `HdfStore.fromStore` only reads the superblock and the root group.
- A group or array's metadata (`zarr.json`) is translated the first time it is requested.
- Chunk references (B-tree walks, inlined string data) are computed only when a chunk of that array is requested.
- `await hdfStore.getConsolidatedMetadata()` walks the metadata of the whole file, but reads no chunk data.

## Development

### Setup

Install PNPM and UV.

```sh
pnpm install
pnpm run generate-fixtures
```

### Test

```sh
pnpm run test
```

### Build

```sh
pnpm run build
```

### Preview demo site

```sh
pnpm run demo:build
pnpm run demo:preview
# Open http://localhost:4173/hdf5-as-virtual-zarr.js/
```

### Publish

```sh
pnpm publish --filter "hdf5-as-virtual-zarr"
```

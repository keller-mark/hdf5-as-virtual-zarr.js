# hdf5-as-virtual-zarr

[![NPM](https://img.shields.io/npm/v/hdf5-as-virtual-zarr.svg?color=black)](https://www.npmjs.com/package/hdf5-as-virtual-zarr)

Given an input HDF5 file URL, generate a Zarr References Specification JSON output.

This TypeScript implementation can run in a web browser, and generates output that is compatible with the [ReferenceStore](https://github.com/manzt/zarrita.js/blob/d511ea44551a62c2af64ba33672672e9286490da/packages/%40zarrita-storage/src/ref.ts#L42) from [zarrita.js](https://github.com/manzt/zarrita.js).

It supports large HDF5 files using range requests (fetching only the subset of bytes required). When possible, large arrays should not be accessed; only the minimal metadata should be accessed.

This is a port of [SingleHdf5ToZarr](https://github.com/fsspec/kerchunk/blob/main/kerchunk/hdf.py#L52) and its `translate` function from the `kerchunk` Python package.


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

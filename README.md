# hdf5-to-zarr.js

Given an input HDF5 file URL, generate a Zarr References Specification JSON output.

This will be a JavaScript/TypeScript implementation that can run in a web browser.

It should support large HDF5 files using range requests and/or streaming. When possible, large arrays should not be accessed; only the minimal metadata should be accessed.

Basically, we want to port into JavaScript: the `SingleHdf5ToZarr` and its `SingleHdf5ToZarr.translate` function from the `kerchunk` Python package.
- https://github.com/fsspec/kerchunk/blob/main/kerchunk/hdf.py#L52
- Example usage of `SingleHdf5ToZarr`: https://github.com/vitessce/vitessce-python/blob/main/src/vitessce/data_utils/anndata.py#L9

Tech stack:
- Typescript
- Use `h5wasm` NPM package for reading HDF5 files in the web browser
  - https://github.com/usnistgov/h5wasm
- Vitest for unit tests
- UV with script metadata for generating test fixtures.
  - Generate test fixtures using AnnData (the `.h5ad` file extension is actually just an HDF5 file). Other HDF5 files are out of scope, so we will ignore them during testing, although in theory our solution should support them just fine.
  - Example script that generates toy anndata objects and saves them to disk: https://github.com/ilan-gold/anndata.js/blob/main/test/fixtures/generate_fixture.py (this script just needs to be modified to call `adata.write_h5ad` when saving)

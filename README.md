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

## Test fixture generation

Test fixtures consist of `.h5ad` files (HDF5), `.adata.zarr` directories (Zarr DirectoryStore), and `.adata.zarr.json` files (JSON representation of the Zarr DirectoryStore for in-memory use with Zarrita).

### Prerequisites

- [UV](https://docs.astral.sh/uv/) (for running the Python fixture generator)
- [Node.js](https://nodejs.org/) (v18+)

### Steps

1. **Generate HDF5, Zarr, and kerchunk reference spec fixtures** using the Python script (UV resolves dependencies automatically via inline script metadata):

   ```bash
   uv run test/fixtures/generate_fixtures.py
   ```

   This creates for each fixture (minimal, dense, sparse):
   - `test/fixtures/{name}.h5ad` — HDF5/AnnData file
   - `test/fixtures/{name}.adata.zarr/` — Zarr v2 DirectoryStore
   - `test/fixtures/{name}.h5ad.refspec.json` — kerchunk ground-truth reference spec JSON

2. **Convert Zarr DirectoryStores to JSON** for use as in-memory stores in tests:

   ```bash
   node scripts/directory-to-memory-store.mjs test/fixtures/minimal.adata.zarr test/fixtures/minimal.adata.zarr.json
   node scripts/directory-to-memory-store.mjs test/fixtures/dense.adata.zarr test/fixtures/dense.adata.zarr.json
   node scripts/directory-to-memory-store.mjs test/fixtures/sparse.adata.zarr test/fixtures/sparse.adata.zarr.json
   ```

   Each JSON file contains `[[key, base64Value], ...]` pairs that can be loaded as an in-memory Zarrita store.

3. **Run tests**:

   ```bash
   pnpm test
   ```

> **Note:** The `.adata.zarr/` directories are excluded from git via `.gitignore`. Only the `.h5ad` files, `.adata.zarr.json` files, and `.h5ad.refspec.json` files are committed. To regenerate from scratch, run steps 1 and 2 above.

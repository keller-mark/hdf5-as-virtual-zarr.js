# hdf5-to-zarr.js

Given an input HDF5 file URL, generate a Zarr References Specification JSON output.

This will be a JavaScript/TypeScript implementation that can run in a web browser, compatible with the ReferenceStore from zarrita.js https://github.com/manzt/zarrita.js/blob/d511ea44551a62c2af64ba33672672e9286490da/packages/%40zarrita-storage/src/ref.ts#L42

It should support large HDF5 files using range requests and/or streaming. When possible, large arrays should not be accessed; only the minimal metadata should be accessed.

Basically, we want to port into JavaScript: the `SingleHdf5ToZarr` and its `SingleHdf5ToZarr.translate` function from the `kerchunk` Python package.
- https://github.com/fsspec/kerchunk/blob/main/kerchunk/hdf.py#L52

## Test fixture generation

Test fixtures consist of `.h5ad` files (HDF5), `.adata.zarr` directories (Zarr DirectoryStore), and `.adata.zarr.json` files (JSON representation of the Zarr DirectoryStore for in-memory use with Zarrita).



This is a Typescript port of the `SingleHdf5ToZarr` and its `SingleHdf5ToZarr.translate` method.

For your reference, the Python implementation can be found in `kerchunk/kerchunk/hdf.py` (present in this repo as a git submodule).

This implementation should use the logic from the jsfive NPM package as a reference (present in this repo as a git submodule):
- IMPORTANT: we always want to avoid loading the full HDF5 file, as it may be huge, multiple gigabytes. Jsfive assumes it always has the full file contents in an arrayBuffer. Its logic will need to be modified to support partial reads via range requests.
- jsfive provides internal utilities such as BTreeV1RawDataChunks to compute chunk addresses (byte offsets) and sizes/lengths. This logic will be important for creating the reference spec JSON.
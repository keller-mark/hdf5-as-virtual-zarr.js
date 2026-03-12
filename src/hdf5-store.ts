// TODO: Implement an Hdf5Store zarrita store.
// This should:
// - Accept an existing AsyncReadable source via its constructor. The root key "/" should correspond to an HDF5 file. (e.g., getRange("/", { offset: 0, length: 500 }) should return the first 500 bytes of the HDF5 file.)
// - The Hdf5Store itself should implement AsyncReadable.
// - Upon the first get or getRange call, the Hdf5Store should generate the Reference Spec JSON and internally create and cache a zarrita ReferenceStore.
// - Subsequent get or getRange calls should delegate to the internal ReferenceStore, which will use the original AsyncReadable source to fetch byte ranges as needed based on the Reference Spec.

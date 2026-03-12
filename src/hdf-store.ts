import { ComposableReferenceStore } from "./ref-store.js";
import type { AsyncReadable } from "./types.js";
import { SingleHdf5ToZarr } from "./single-hdf5-to-zarr.js";
import type { AbsolutePath, RangeQuery } from "@zarrita/storage";

export class HdfStore implements AsyncReadable {
  private refStore: ComposableReferenceStore;

  constructor(refStore: ComposableReferenceStore) {
    this.refStore = refStore;
  }

  async get(key: AbsolutePath, opts?: RequestInit): Promise<Uint8Array | undefined> {
    return this.refStore.get(key, opts);
  }

  // TODO: does it make sense to support getRange here?

  /**
   *
   * @param {AsyncReadable} hdfSource Assumed to be a zarrita Store,
   * with the HDF5 file located at the root key "/". In other words,
   * hdfSource.getRange("/", { offset, length }) should return the
   * corresponding byte range from the HDF5 file.
   * @returns {Promise<HdfStore>} An HdfStore instance which internally
   * uses a ReferenceStore and handles the Reference Specification JSON generation.
   */
  static async fromStore(hdfSource: AsyncReadable): Promise<HdfStore> {
    const converter = new SingleHdf5ToZarr(hdfSource, { url: null });
    const refSpec = await converter.translate();
    const refStore = ComposableReferenceStore.fromSpec(refSpec as unknown as Record<string, unknown>, hdfSource);
    return new HdfStore(refStore);
  }
}

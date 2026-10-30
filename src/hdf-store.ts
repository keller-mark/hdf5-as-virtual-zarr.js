import { ComposableReferenceStore } from "./ref-store.js";
import type { AsyncReadable, ReferenceSpec } from "./types.js";
import { SingleHdf5ToZarr } from "./single-hdf5-to-zarr.js";
import {
  refSpecToConsolidatedMetadataV3,
  type ZarrV3ConsolidatedMetadata,
} from "./ref-spec-to-consolidated-metadata.js";
import type { AbsolutePath, GetOptions, RangeQuery } from "@zarrita/storage";

export class HdfStore implements AsyncReadable {
  private refStore: ComposableReferenceStore;
  private refSpec: ReferenceSpec;

  constructor(refStore: ComposableReferenceStore, refSpec: ReferenceSpec) {
    this.refStore = refStore;
    this.refSpec = refSpec;
  }

  async get(key: AbsolutePath, opts?: GetOptions): Promise<Uint8Array | undefined> {
    return this.refStore.get(key, opts);
  }

  // TODO: does it make sense to support getRange here?

  /**
   * @returns The root zarr.json document with inline consolidated metadata
   * (zarr-python format) describing every group and array in the hierarchy.
   */
  getConsolidatedMetadata(): ZarrV3ConsolidatedMetadata {
    return refSpecToConsolidatedMetadataV3(this.refSpec);
  }

  /**
   *
   * @param {AsyncReadable} hdfSource Assumed to be a zarrita Store,
   * with the HDF5 file located at the root key "/". In other words,
   * hdfSource.getRange("/", { offset, length }) should return the
   * corresponding byte range from the HDF5 file.
   * @returns {Promise<HdfStore>} An HdfStore instance which exposes the HDF5
   * file as a Zarr v3 hierarchy. Internally uses a ReferenceStore and handles
   * the Reference Specification JSON generation.
   */
  static async fromStore(hdfSource: AsyncReadable): Promise<HdfStore> {
    const converter = new SingleHdf5ToZarr(hdfSource, { url: null, zarrFormat: 3 });
    const refSpec = await converter.translate();
    const refStore = ComposableReferenceStore.fromSpec(refSpec as unknown as Record<string, unknown>, hdfSource);
    return new HdfStore(refStore, refSpec);
  }
}

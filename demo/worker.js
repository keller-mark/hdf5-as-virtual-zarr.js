import h5wasm from "h5wasm";
import { SingleHdf5ToZarr } from "hdf5-to-zarr";

self.onmessage = async (e) => {
  try {
    const file = e.data;
    const { FS } = await h5wasm.ready;

    /* mount the File object via WORKERFS so h5wasm reads on demand */
    FS.mkdir("/work");
    FS.mount(FS.filesystems.WORKERFS, { files: [file] }, "/work");

    const h5File = new h5wasm.File("/work/" + file.name, "r");
    // Wrap h5File in a Source-compatible object
    const source = {
      type: "h5wasm",
      url: new URL("memory://" + file.name),
      async fetch(offset, length) {
        // Read from h5wasm file handle (this is a simplification;
        // a real implementation would use range reads)
        throw new Error("h5wasm Source.fetch not yet implemented");
      },
    };
    const converter = new SingleHdf5ToZarr(source, { url: null });
    const refSpec = await converter.translate();
    h5File.close();
    FS.unmount("/work");

    self.postMessage({ success: true, refSpec });
  } catch (err) {
    self.postMessage({ success: false, error: err.message });
  }
};

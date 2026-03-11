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
    const converter = new SingleHdf5ToZarr(h5File, { url: null });
    const refSpec = converter.translate();
    h5File.close();
    FS.unmount("/work");

    self.postMessage({ success: true, refSpec });
  } catch (err) {
    self.postMessage({ success: false, error: err.message });
  }
};

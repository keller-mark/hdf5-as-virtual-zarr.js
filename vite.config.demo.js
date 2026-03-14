import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  base: '/hdf5-as-virtual-zarr.js/',
  resolve: {
    alias: {
      "hdf5-as-virtual-zarr": resolve(import.meta.dirname, "src/index.ts"),
    },
  },
  build: {
    outDir: "demo-dist",
  },
});

import { defineConfig } from "vite";

// Bundles src/index.ts together with its dependencies into dist/bundled/,
// alongside the unbundled tsc output in dist/ (which keeps dependencies external).
export default defineConfig({
  build: {
    outDir: "dist/bundled",
    emptyOutDir: true,
    sourcemap: true,
    // No minification, so stack traces and debugging remain readable.
    minify: false,
    lib: {
      entry: "src/index.ts",
      formats: ["es"],
      fileName: () => "index.js",
    },
  },
});

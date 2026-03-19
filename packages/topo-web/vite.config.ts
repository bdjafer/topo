import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  root: ".",
  resolve: {
    alias: {
      // Point WASM imports to the topo-core pkg directory.
      "../../topo-core/pkg": resolve(__dirname, "../topo-core/pkg"),
    },
  },
  build: {
    outDir: "dist-site",
    rollupOptions: {
      input: resolve(__dirname, "index.html"),
    },
  },
  server: {
    fs: {
      // Allow serving WASM from the sibling topo-core package.
      allow: [resolve(__dirname, ".."), resolve(__dirname)],
    },
    headers: {
      // Required for WASM streaming compilation.
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
  assetsInclude: ["**/*.wasm"],
});

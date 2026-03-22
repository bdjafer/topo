import { defineConfig } from "vite";
import { resolve } from "path";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  root: ".",
  resolve: {
    alias: {
      // Point WASM imports to the topo-analyzer pkg directory.
      "../../topo-analyzer/pkg": resolve(__dirname, "../topo-analyzer/pkg"),
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
      // Allow serving WASM from the sibling topo-analyzer package.
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

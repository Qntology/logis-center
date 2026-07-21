import { defineConfig } from "vite";

export default defineConfig({
  // Use relative paths for assets so they work in Tauri's WebView
  base: "", 
  publicDir: "public",
  build: {
    outDir: "../dist/mobile",
    emptyOutDir: true,
    assetsInlineLimit: 0,
    rollupOptions: {
      output: {
        entryFileNames: `assets/[name].js`,
        chunkFileNames: `assets/[name].js`,
        assetFileNames: `assets/[name].[ext]`
      }
    }
  },
  server: {
    port: 1420,
    strictPort: true,
    host: true
  }
});
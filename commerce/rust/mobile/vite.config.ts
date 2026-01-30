import { defineConfig } from "vite";

export default defineConfig({
  // 상대 경로 빌드를 위해 base를 비웁니다.
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
    host: "0.0.0.0",
  },
});

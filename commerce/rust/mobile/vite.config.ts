import { defineConfig } from "vite";

export default defineConfig({
  // 루트를 mobile/ 로 설정 (기본값)
  base: "./",
  publicDir: "public",
  build: {
    outDir: "../dist/mobile",
    emptyOutDir: true,
    assetsInlineLimit: 0,
  },
  server: {
    port: 1420,
    strictPort: true,
    host: "0.0.0.0",
  },
});

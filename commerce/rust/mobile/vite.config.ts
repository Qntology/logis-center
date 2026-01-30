import { defineConfig } from "vite";

export default defineConfig({
  // 상대 경로(./)를 사용하여 안드로이드 assets 폴더에서 직접 로딩 가능하게 함
  base: "./",
  build: {
    outDir: "../dist/mobile",
    emptyOutDir: true,
    assetsInlineLimit: 0, // 폰트를 base64로 바꾸지 않고 파일을 유지함
  }
});
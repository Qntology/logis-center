import { defineConfig } from "vite";

export default defineConfig({
  clearScreen: false,
  root: "src", // index.html이 있는 폴더를 루트로 지정
  server: {
    port: 1421,
    strictPort: true,
    host: true,
  },
  build: {
    outDir: "../../dist/mobile", // 빌드 결과물을 프로젝트 공통 dist 폴더로 이동 (선택 사항)
    emptyOutDir: true,
  }
});
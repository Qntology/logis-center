import { defineConfig } from "vite";

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;

// https://vite.dev/config/
export default defineConfig(async () => ({

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  root: "src",
  publicDir: "../public", // 🌟 추가: src 바깥(rust/public)에 있는 라이브러리 파일들을 인식하여 빌드(dist) 시 복사합니다.
  base: "./", // 🌟 추가: 모든 에셋(JS, CSS, 정적파일)의 빌드 경로를 상대 경로로 강제 고정합니다.
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },
}));

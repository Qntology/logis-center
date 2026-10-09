import { defineConfig } from "vitest/config";

// Unit tests for the TypeScript frontend (src/lib, src/modes).
// Kept separate from vite.config.ts so the app build config (root: "src", fixed dev port) is untouched.
export default defineConfig({
  server: {
    watch: {
      // src-tauri/target and the mobile app are huge and irrelevant to these tests.
      ignored: ["**/src-tauri/**", "**/mobile/**", "**/dist/**"],
    },
  },
  test: {
    include: ["src/tests/**/*.test.ts"],
    environment: "happy-dom",
    setupFiles: ["src/tests/setup.ts"],
    restoreMocks: true,
  },
});

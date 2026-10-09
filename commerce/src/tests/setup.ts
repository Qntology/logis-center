// Global test setup, executed before every test file (vitest.config.ts -> setupFiles).
//
// index.html loads ethers / pako as classic <script> globals, and src/lib/utils.ts and
// src/modes/oauth.ts capture `window.ethers` / `window.pako` at *import* time, so the real
// vendored bundles must be on the global object before any test imports those modules.
import { beforeEach, vi } from "vitest";
// Vite `?raw` imports return the file contents as a string (works in vitest's DOM/web transform mode).
import ethersSource from "../ethers.umd.min.js?raw"; // ethers v6.6.2
import pakoSource from "../pako.js?raw"; // pako v2.1.0

// Indirect eval runs each UMD bundle as global code in this realm, so it attaches to globalThis.
(0, eval)(ethersSource);
(0, eval)(pakoSource);

const g = globalThis as any;
if (typeof window !== "undefined" && (window as any) !== g) {
  (window as any).ethers = g.ethers;
  (window as any).pako = g.pako;
}

// The production code logs heavily; keep the test output readable.
// (restoreMocks: true restores these spies before every test, so they are re-applied here.)
beforeEach(() => {
  vi.spyOn(console, "log").mockImplementation(() => {});
  vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
});

// 두 워커(원본 TS · Rust wasm)를 같은 조건으로: 가상 시계(setTimeout 이 시계를 밀고 바로 실행), 같은 씨앗의 Math.random.
let t = 1760000000000;
Date.now = () => t;
let seed = 20261118;
Math.random = () => {
  seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
  return (seed >>> 8) / 16777216;
};
const realSetTimeout = globalThis.setTimeout.bind(globalThis);
globalThis.setTimeout = (fn, ms, ...args) => {
  const d = Number(ms) || 0;
  t += Math.max(0, d);
  return realSetTimeout(fn, 0, ...args);
};
globalThis.__harness_clock = () => t;

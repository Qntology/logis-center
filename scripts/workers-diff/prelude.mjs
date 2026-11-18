// 두 워커(원본 TS · Rust wasm)를 같은 조건으로: 가상 시계(setTimeout 이 시계를 밀고 바로 실행), 같은 씨앗의 Math.random,
// 운영 Workers 와 같은 UTC 시간대.
let t = 1760000000000;
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

// 운영 Workers 의 지역 시간대는 UTC 다. Windows 의 workerd 는 TZ 환경 변수를 따르지 않고 PC 시간대(KST)를 쓰므로,
// 오프셋 없는 날짜 시각 글("2026-03-12T09:00:00")을 UTC 로 읽고 지역 시각 메서드를 UTC 메서드와 같게 만든다.
const RealDate = Date;
const NO_OFFSET = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?$/;
const asUtc = (a) => (typeof a === "string" && NO_OFFSET.test(a.trim()) ? a.trim() + "Z" : a);
class UTCDate extends RealDate {
  constructor(...args) {
    if (args.length === 0) super();
    else if (args.length === 1) super(asUtc(args[0]));
    else super(RealDate.UTC(...args));
  }
}
for (const part of ["FullYear", "Month", "Date", "Day", "Hours", "Minutes", "Seconds", "Milliseconds"]) {
  UTCDate.prototype[`get${part}`] = RealDate.prototype[`getUTC${part}`];
  if (part !== "Day") UTCDate.prototype[`set${part}`] = RealDate.prototype[`setUTC${part}`];
}
UTCDate.prototype.getTimezoneOffset = () => 0;
UTCDate.now = () => t;
UTCDate.parse = (s) => RealDate.parse(asUtc(s));
UTCDate.UTC = RealDate.UTC;
globalThis.Date = UTCDate;

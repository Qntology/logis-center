import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { crc32, hashId, parseItemData, parseStatus, randomHash, safeClone, time2text } from "../lib/utils";

const NOW = Date.UTC(2026, 9, 9, 12, 0, 0); // 2026-10-09T12:00:00Z
const gzipBytes = (s: string): Uint8Array => (window as any).pako.gzip(s);

describe("crc32 / randomHash", () => {
  it.each<[string, number]>([
    ["", 0],
    ["a", 3904355907],
    ["abc", 891568578],
    ["123456789", 0xcbf43926],
    ["hello", 907060870],
    ["hello world", 222957957],
    ["The quick brown fox jumps over the lazy dog", 1095738169],
    ["é", 198489425],
  ])("crc32(%j) = %d (standard CRC-32/IEEE)", (input, expected) => {
    expect(crc32(input)).toBe(expected);
  });

  it("throws RangeError for characters above 0xFF (also via randomHash)", () => {
    expect(() => crc32("한")).toThrow(RangeError);
    expect(() => randomHash("한")).toThrow(RangeError);
  });

  it.each([
    ["", "0"],
    ["abc", "qi8ge2"],
    ["123456789", "35v8e96"],
    ["hello", "r119k6"],
  ])("randomHash(%j) = %j (crc32 in base 32)", (input, expected) => {
    expect(randomHash(input)).toBe(expected);
  });

  it("randomHash() without an argument hashes a random number", () => {
    expect(randomHash()).toMatch(/^[0-9a-v]+$/);
  });
});

describe("hashId (real vendored ethers v6.6.2)", () => {
  it.each([
    ["logis.center", "0x45a2a96bf8ae28042073444f3ec1dd7acaa8a961"],
    ["example.com", "0x5134846d336b9828abe829c91bdaa2d07fea6a28"],
    ["example.co.kr", "0xb2e020367adcbfdf936a3266bb8ab4515fb0623b"],
    ["a@b.com", "0xf7932301cf309f69791a5e67be5a400b0c26106d"],
    ["", "0x7404358246d491ed2f9dac694f7dea789037424c"],
    ["0x0000000000000000000000000000000000000000", "0x4d9b9d714f7c42cacb353cf4861bb8bfd88a0420"],
  ])("hashId(%j) = %s", (input, expected) => {
    expect(hashId(input)).toBe(expected);
  });

  it("hashId(null/undefined) returns a fresh random lowercase address", () => {
    const a = hashId(undefined as unknown as string);
    const b = hashId(null as unknown as string);
    expect(a).toMatch(/^0x[0-9a-f]{40}$/);
    expect(b).toMatch(/^0x[0-9a-f]{40}$/);
    expect(a).not.toBe(b);
  });
});

describe("parseStatus (strict switch)", () => {
  it.each([
    [1, "progress"],
    [2, "stop"],
    [3, "cancel"],
    [4, "refund"],
    [5, "return"],
    [6, "error"],
    [7, "expire"],
    [8, "exchange"],
    [9, "complete"],
    [10, "draft"],
    [11, "show"],
    [12, "hide"],
  ])("maps code %d to %j", (code, name) => {
    expect(parseStatus(code)).toBe(name);
  });

  it.each<[number | string, string]>([
    ["1", "progress"],
    ["9", "complete"],
    ["  12 ", "hide"],
    ["complete", "complete"],
    ["abc", "abc"],
    [0, ""],
    [13, ""],
    ["", ""],
  ])("parseStatus(%j) -> %j", (input, expected) => {
    expect(parseStatus(input)).toBe(expected);
  });
});

describe("time2text (fake Date: 2026-10-09T12:00:00Z)", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(NOW);
  });
  afterEach(() => vi.useRealTimers());

  it.each([
    [0, "0 seconds"],
    [59, "59 seconds"],
    [60, "60 seconds"], // thresholds are strict (> 1 unit)
    [120, "2 minutes"],
    [3600, "60 minutes"],
    [7200, "2 hours"],
    [86400, "24 hours"],
    [2 * 86400, "2 days"],
    [30 * 86400, "30 days"],
    [60 * 86400, "2 months"],
    [365 * 86400, "12 months"],
    [2 * 365 * 86400, "2 years"],
  ])("%d s ago -> %j", (secondsAgo, expected) => {
    expect(time2text(NOW - secondsAgo * 1000)).toBe(expected);
  });

  it("crosses into the next unit just above each threshold", () => {
    expect(time2text(NOW - 61_000)).toMatch(/^1 minutes?$/);
    expect(time2text(NOW - 3_601_000)).toMatch(/^1 hours?$/);
    expect(time2text(NOW - 86_401_000)).toMatch(/^1 days?$/);
  });

  it("accepts ISO strings and Date objects", () => {
    expect(time2text("2026-10-07T12:00:00Z")).toBe("2 days");
    expect(time2text(new Date(NOW - 150_000))).toBe("2 minutes");
  });

  it.fails("BUG-UTIL-1 utils.ts:64-85 future timestamps render as negative seconds ('-60 seconds')", () => {
    expect(time2text(NOW + 60_000)).not.toMatch(/^-/);
  });

  it.fails("BUG-UTIL-2 utils.ts:65 numeric-string epoch ms ('1735689600000') is parsed as an invalid Date ('NaN seconds')", () => {
    expect(time2text("1735689600000")).toBe(time2text(1735689600000));
  });
});

describe("parseItemData", () => {
  it("returns {} for falsy input", () => {
    expect(parseItemData(null)).toEqual({});
    expect(parseItemData(undefined)).toEqual({});
    expect(parseItemData("")).toEqual({});
    expect(parseItemData(0)).toEqual({});
  });

  it("returns plain objects by reference (already parsed by serde)", () => {
    const obj = { a: 1 };
    expect(parseItemData(obj)).toBe(obj);
  });

  it("parses JSON strings and wraps non-JSON text as {text}", () => {
    expect(parseItemData('{"a":1}')).toEqual({ a: 1 });
    expect(parseItemData("hello")).toEqual({ text: "hello" });
  });

  it("gunzips number[] and Uint8Array payloads", () => {
    const gz = gzipBytes('{"a":1}');
    expect(parseItemData(Array.from(gz))).toEqual({ a: 1 });
    expect(parseItemData(Uint8Array.from(gz))).toEqual({ a: 1 });
  });

  it("returns {} for undecodable byte arrays and non-numeric arrays", () => {
    expect(parseItemData([1, 2, 3])).toEqual({});
    expect(parseItemData([])).toEqual({});
    expect(parseItemData(["a"])).toEqual({});
    expect(console.warn).toHaveBeenCalled();
  });

  it.fails("BUG-UTIL-3 utils.ts:128 an ArrayBuffer is returned untouched by the 'already an object' branch and never gunzipped", () => {
    const gz = Uint8Array.from(gzipBytes('{"a":1}'));
    expect(parseItemData(gz.buffer)).toEqual({ a: 1 });
  });

  it.fails("BUG-UTIL-4 utils.ts:136 parseItemData('null') returns null instead of an object (db.ts callers then throw)", () => {
    expect(parseItemData("null")).toEqual({});
  });
});

describe("safeClone", () => {
  it("deep-clones objects and arrays and replaces circular references with null", () => {
    const src: any = { x: 1, nested: { list: [1, { y: 2 }] } };
    src.self = src;
    const copy = safeClone(src) as any;
    expect(copy).toEqual({ x: 1, nested: { list: [1, { y: 2 }] }, self: null });
    expect(copy.nested).not.toBe(src.nested);
    expect(copy.nested.list[1]).not.toBe(src.nested.list[1]);
  });

  it("passes primitives through", () => {
    expect(safeClone(5)).toBe(5);
    expect(safeClone("s")).toBe("s");
    expect(safeClone(null)).toBe(null);
  });

  it.fails("BUG-UTIL-5 utils.ts:53 a repeated but non-circular reference is replaced with null on its second occurrence", () => {
    const shared = { v: 1 };
    expect(safeClone({ p: shared, q: shared })).toEqual({ p: { v: 1 }, q: { v: 1 } });
  });
});

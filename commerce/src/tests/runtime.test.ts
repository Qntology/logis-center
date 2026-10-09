import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  SYNC_BACKOFF_FACTOR,
  SYNC_BASE_INTERVAL_MS,
  SYNC_MAX_INTERVAL_MS,
  computeSyncInterval,
  decodeAnalyticBlob,
  getRootDomain,
  getSyncIntervalMs,
  resetSyncBackoff,
  updateSyncBackoff,
} from "../modes/runtime";

const utf8 = (s: string): number[] => Array.from(new TextEncoder().encode(s));
const gzip = (s: string): number[] => Array.from((window as any).pako.gzip(s) as Uint8Array);
const base64 = (bytes: number[]): string => btoa(String.fromCharCode(...bytes));

describe("getRootDomain", () => {
  it.each([
    ["www.example.com", "example.com"],
    ["example.com", "example.com"],
    ["a.b.c.example.com", "example.com"],
    ["shop.example.co.kr", "example.co.kr"],
    ["example.co.kr", "example.co.kr"],
    ["m.shop.example.com.sg", "example.com.sg"],
    ["commerce.logis.center", "logis.center"],
    ["co.kr", "co.kr"],
    ["localhost", "localhost"],
    ["", ""],
  ])("getRootDomain(%j) -> %j", (host, root) => {
    expect(getRootDomain(host)).toBe(root);
  });

  it.fails("BUG-RT-1 runtime.ts:206 two-part TLD check is a raw endsWith(), so www.costco.kr / www.telecom.my are not reduced", () => {
    expect(getRootDomain("www.costco.kr")).toBe("costco.kr");
    expect(getRootDomain("www.telecom.my")).toBe("telecom.my");
  });
});

describe("adaptive polling backoff", () => {
  beforeEach(() => resetSyncBackoff());

  it("exposes the documented constants", () => {
    expect([SYNC_BASE_INTERVAL_MS, SYNC_BACKOFF_FACTOR, SYNC_MAX_INTERVAL_MS]).toEqual([3000, 1.5, 30000]);
    expect(computeSyncInterval()).toBe(3000);
    expect(getSyncIntervalMs()).toBe(3000);
  });

  it("grows 1.5x per unchanged poll and caps at 30 s", () => {
    const seen: number[] = [];
    for (let i = 0; i < 7; i++) {
      updateSyncBackoff(false);
      seen.push(getSyncIntervalMs());
    }
    expect(seen).toEqual([4500, 6750, 10125, 15187.5, 22781.25, 30000, 30000]);
    expect(computeSyncInterval()).toBe(30000);
  });

  it("snaps back to the base interval on change and on resetSyncBackoff()", () => {
    updateSyncBackoff(false);
    updateSyncBackoff(false);
    updateSyncBackoff(true);
    expect(getSyncIntervalMs()).toBe(3000);
    expect(computeSyncInterval()).toBe(3000);

    updateSyncBackoff(false);
    resetSyncBackoff();
    expect(getSyncIntervalMs()).toBe(3000);
  });
});

describe("rt() / bindModeRuntime", () => {
  it("throws until a runtime is bound, then returns the bound object", async () => {
    vi.resetModules();
    const fresh = await import("../modes/runtime");
    expect(() => fresh.rt()).toThrow("[MODE] runtime");
    const fake = { marker: true } as any;
    fresh.bindModeRuntime(fake);
    expect(fresh.rt()).toBe(fake);
  });
});

describe("decodeAnalyticBlob", () => {
  it.each<[string, unknown, unknown]>([
    ["null", null, {}],
    ["undefined", undefined, {}],
    ["empty string", "", {}],
    ["a number", 42, {}],
    ["an empty array", [], {}],
    ["an empty object", {}, {}],
    ["a JSON object string", '{"a":1}', { a: 1 }],
    ["text that is neither JSON nor base64", "hello world", {}],
    ["a plain object (returned as is)", { foo: 1 }, { foo: 1 }],
    ["a {data:{...}} wrapper (unwrapped)", { data: { x: 1 }, type: "click" }, { x: 1 }],
  ])("decodes %s", (_label, input, expected) => {
    expect(decodeAnalyticBlob(input)).toEqual(expected);
  });

  it("decodes base64 UTF-8 JSON", () => {
    expect(decodeAnalyticBlob(base64(utf8('{"t":"한글"}')))).toEqual({ t: "한글" });
  });

  it("gunzips base64 gzip payloads larger than 50 bytes", () => {
    const payload = { action: "click", summary: "x".repeat(80) };
    const bytes = gzip(JSON.stringify(payload));
    expect(bytes.length).toBeGreaterThan(50);
    expect(decodeAnalyticBlob(base64(bytes))).toEqual(payload);
  });

  it("decodes byte arrays: gzip / plain number[], Node Buffer JSON shape, Uint8Array, numeric-key objects", () => {
    expect(decodeAnalyticBlob(gzip('{"a":1}'))).toEqual({ a: 1 });
    expect(decodeAnalyticBlob(utf8('{"a":1}'))).toEqual({ a: 1 });
    expect(decodeAnalyticBlob({ type: "Buffer", data: gzip('{"a":1}') })).toEqual({ a: 1 });
    expect(decodeAnalyticBlob(Uint8Array.from(gzip('{"a":1}')))).toEqual({ a: 1 });
    expect(decodeAnalyticBlob({ ...utf8('{"a":1}') })).toEqual({ a: 1 });
  });

  it("falls back to UTF-8 when window.pako is unavailable", () => {
    const saved = (window as any).pako;
    try {
      delete (window as any).pako;
      expect(decodeAnalyticBlob(utf8('{"a":1}'))).toEqual({ a: 1 });
    } finally {
      (window as any).pako = saved;
    }
  });

  it("fixture: gzip of '{\"a\":1}' is a small (<= 50 byte) payload", () => {
    expect(gzip('{"a":1}').length).toBeLessThanOrEqual(50);
  });

  it.fails("BUG-RT-2 runtime.ts:181 base64 gzip payloads of <= 50 bytes are never gunzipped (decoded as {})", () => {
    expect(decodeAnalyticBlob(base64(gzip('{"a":1}')))).toEqual({ a: 1 });
  });

  it.fails("BUG-RT-3 runtime.ts:159-160 a Uint8Array view is decoded from its whole underlying buffer, not the view", () => {
    const view = new TextEncoder().encode('xx{"a":1}').subarray(2);
    expect(decodeAnalyticBlob(view)).toEqual({ a: 1 });
  });

  it.fails("BUG-RT-4 runtime.ts:177 a JSON primitive string ('42') is returned as a number, not an object", () => {
    const decoded = decodeAnalyticBlob("42");
    expect(typeof decoded === "object" && decoded !== null && !Array.isArray(decoded)).toBe(true);
  });

  it.fails("BUG-RT-4 runtime.ts:177 the string 'null' decodes to null, not an object", () => {
    expect(decodeAnalyticBlob("null")).toEqual({});
  });
});

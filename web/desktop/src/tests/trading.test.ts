import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));

import { invoke } from "@tauri-apps/api/core";
import { bindModeRuntime, getSyncIntervalMs, resetSyncBackoff } from "../modes/runtime";
import {
  TRADING_API_HOST,
  pullTradingData,
  pushTradingData,
  resetTradingThrottle,
  syncTradingData,
  syncTradingInBackground,
  tradingApiFetch,
} from "../modes/trading";
import { makeDb, type FakeDb, type Row } from "./helpers/fakedb";
import { callsOf, makeRuntime } from "./helpers/runtime";

const invokeMock = vi.mocked(invoke) as unknown as ReturnType<typeof vi.fn>;
const NOW = Date.UTC(2026, 9, 9, 12, 0, 0); // 1791547200000
const KST = -9 * 3600 * 1000; // new Date().getTimezoneOffset() * 60_000 in UTC+9

let db: FakeDb;
let env: ReturnType<typeof makeRuntime>;

function setup(rows: Row[] = [], kv: Record<string, unknown> = {}) {
  db = makeDb({ items: rows });
  env = makeRuntime({
    session: { hash: "h", token: "t", team: "T" },
    appDb: db,
    timezoneOffset: KST,
    searchMode: "shipping",
    itemTombstones: ["dead"],
    kv,
  });
  bindModeRuntime(env.runtime);
}

/** Route invoke(): proxy_fetch GET -> `pull`, proxy_fetch POST -> `push(body)`, everything else -> "ok". */
function serve(handlers: { pull?: () => unknown; push?: (body: any) => unknown }) {
  invokeMock.mockImplementation(async (cmd: string, args: any) => {
    if (cmd !== "proxy_fetch") return "ok";
    if (args.method === "POST") return handlers.push ? handlers.push(args.body) : null;
    return handlers.pull ? handlers.pull() : { results: [] };
  });
}

const shippingRow = (id: string, stamp: number, extra: Row = {}): Row => ({
  id,
  type: "BL",
  mode: "shipping",
  created_at: stamp,
  updated_at: stamp,
  data: {},
  ...extra,
});

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(NOW);
  resetTradingThrottle();
  resetSyncBackoff();
  setup();
});
afterEach(() => vi.useRealTimers());

describe("tradingApiFetch", () => {
  it("GET: hash/token/to first, then non-empty query params (false and 0 are kept)", async () => {
    invokeMock.mockResolvedValue({ ok: 1 });
    expect(await tradingApiFetch({ since: 0, a: undefined, b: null, c: "", d: false, e: "x y" })).toEqual({ ok: 1 });
    const [cmd, args] = invokeMock.mock.calls[0];
    expect(cmd).toBe("proxy_fetch");
    expect(args.url).toBe(`${TRADING_API_HOST}/?hash=h&token=t&to=T&since=0&d=false&e=x+y`);
    expect(args.method).toBe("GET");
    expect(args.headers).toEqual({ "Content-Type": "application/json" });
    expect(args).not.toHaveProperty("body");
  });

  it("POST with gzip adds Content-Encoding and forwards the body", async () => {
    invokeMock.mockResolvedValue({});
    await tradingApiFetch({}, { method: "POST", body: { items: [] }, gzip: true });
    const [, args] = invokeMock.mock.calls[0];
    expect(args.url).toBe(`${TRADING_API_HOST}/?hash=h&token=t&to=T`);
    expect(args.headers).toEqual({ "Content-Type": "application/json", "Content-Encoding": "gzip" });
    expect(args.body).toEqual({ items: [] });
  });

  it("returns null without credentials and when invoke rejects", async () => {
    env.state.session = { hash: "h" };
    expect(await tradingApiFetch({})).toBeNull();
    expect(invokeMock).not.toHaveBeenCalled();

    env.state.session = { hash: "h", token: "t" };
    invokeMock.mockRejectedValue(new Error("offline"));
    expect(await tradingApiFetch({})).toBeNull();
  });
});

describe("pullTradingData", () => {
  it("pulls since the stored cursor, decodes data, applies the server envelope and advances the cursor", async () => {
    setup([], { trading_sync_cursor: "100" });
    const gz = Array.from((window as any).pako.gzip('{"summary":" s "}') as Uint8Array);
    serve({
      pull: () => ({
        results: [
          { id: "dead", type: "BL" }, // tombstoned
          { id: "n1", type: "BL", data: '{"text":"  hello  "}', created_at: "5", updated_at: 6, flag: "KR" },
          { id: "n2", type: "CI", data: { data: gz } },
          { type: "x" }, // no id
        ],
        cursor: 999,
      }),
    });

    expect(await pullTradingData()).toBe(2);

    // created_at is an upper bound: max(now, now - tzOffset) + 60 s
    expect(callsOf(invokeMock, "proxy_fetch")[0].url).toBe(
      `${TRADING_API_HOST}/?hash=h&token=t&to=T&since=100&created_at=${NOW - KST + 60_000}&limit=1000`,
    );
    const [upsert] = callsOf(invokeMock, "upsert_items");
    expect(upsert.items.map((i: Row) => ({ id: i.id, mode: i.mode, status: i.status, text: i.text, created_at: i.created_at, flag: i.flag }))).toEqual([
      { id: "n1", mode: "shipping", status: 9, text: "hello", created_at: 5, flag: "KR" },
      { id: "n2", mode: "shipping", status: 9, text: "s", created_at: 0, flag: "" },
    ]);
    expect(upsert.items[0].data).toMatchObject({ id: "n1", type: "BL", mode: "shipping", flag: "KR", created_at: 5, updated_at: 6 });
    expect(db.ids("items")).toEqual(["n1", "n2"]);
    expect(env.kv.get("trading_sync_cursor")).toBe("999");
  });

  it("stores the cursor even when no rows come back", async () => {
    serve({ pull: () => ({ results: [], cursor: 555 }) });
    expect(await pullTradingData()).toBe(0);
    expect(env.kv.get("trading_sync_cursor")).toBe("555");
    expect(callsOf(invokeMock, "upsert_items")).toHaveLength(0);
  });

  it.fails("BUG-TR-1 trading.ts:83-85 a row whose data is a JSON primitive ('42') throws and aborts the whole pull (cursor never advances)", async () => {
    serve({ pull: () => ({ results: [{ id: "p", type: "BL", data: "42" }, { id: "ok", type: "BL", data: '{"text":"fine"}' }], cursor: 7 }) });
    await pullTradingData();
    expect(env.kv.get("trading_sync_cursor")).toBe("7");
  });
});

describe("pushTradingData", () => {
  it("posts local shipping rows newer than the push cursor (oldest first) and advances the cursor", async () => {
    setup([shippingRow("b", 2, { flag: "KR", data: { digest: "dg" } }), shippingRow("a", 1), { id: "c", mode: "commerce", created_at: 9, data: {} }]);
    let posted: any = null;
    serve({ push: (body) => ((posted = body), { accepted: 2, skipped: 0, rejected: 0 }) });

    expect(await pushTradingData()).toBe(2);
    expect(posted.items).toEqual([
      { id: "a", type: "BL", flag: "", digest: "", created_at: 1, updated_at: 1, data: {} },
      { id: "b", type: "BL", flag: "KR", digest: "dg", created_at: 2, updated_at: 2, data: { digest: "dg" } },
    ]);
    const [, post] = invokeMock.mock.calls[0];
    expect(post.headers["Content-Encoding"]).toBe("gzip");
    expect(env.kv.get("trading_push_cursor")).toBe("2");
    expect(await pushTradingData()).toBe(0); // nothing newer than the cursor
  });

  it("adopts the server-confirmed envelope (ref/cc/bcc) back into LanceDB and Dexie", async () => {
    setup([shippingRow("a", 1, { ref: "", cc: "", bcc: "", data: { text: "T" } })]);
    serve({ push: () => ({ accepted: 1, results: [{ id: "a", type: "BL", ref: "R", cc: "C", bcc: "B", mode: "shipping", flag: "KR" }] }) });
    await pushTradingData();
    const [adopted] = callsOf(invokeMock, "upsert_items");
    expect(adopted.items).toHaveLength(1);
    expect(adopted.items[0]).toMatchObject({ id: "a", ref: "R", cc: "C", bcc: "B", text: "T", data: { ref: "R", cc: "C", bcc: "B", text: "T" } });
    expect(db.rows("items")[0]).toMatchObject({ ref: "R", cc: "C" });
  });

  it("returns 0 without a local DB", async () => {
    env.runtime.appDb = null;
    expect(await pushTradingData()).toBe(0);
    expect(invokeMock).not.toHaveBeenCalled();
  });

  it.fails("BUG-TR-2 trading.ts:146-170 rows sharing the batch-boundary timestamp are never pushed (strict > cursor)", async () => {
    setup(Array.from({ length: 201 }, (_, i) => shippingRow(`r${String(i).padStart(3, "0")}`, 1000)));
    const pushed = new Set<string>();
    serve({ push: (body) => (body.items.forEach((i: Row) => pushed.add(i.id)), { accepted: body.items.length, skipped: 0, rejected: 0 }) });
    await pushTradingData();
    await pushTradingData();
    expect(pushed.size).toBe(201);
  });

  it.fails("BUG-TR-3 trading.ts:167 one permanently rejected row freezes the push cursor, so later rows are never sent", async () => {
    setup(Array.from({ length: 201 }, (_, i) => shippingRow(`r${String(i).padStart(3, "0")}`, i + 1)));
    const accepted = new Set<string>();
    serve({
      push: (body) => {
        const ids: string[] = body.items.map((i: Row) => i.id);
        const ok = ids.filter((id) => id !== "r000");
        ok.forEach((id) => accepted.add(id));
        return { accepted: ok.length, skipped: 0, rejected: ids.length - ok.length };
      },
    });
    for (let i = 0; i < 3; i++) await pushTradingData();
    expect(accepted.has("r200")).toBe(true);
  });
});

describe("syncTradingData / syncTradingInBackground", () => {
  it("pushes then pulls; on changes re-renders the shipping list and resets the backoff", async () => {
    serve({ pull: () => ({ results: [{ id: "n1", type: "BL", data: "{}" }], cursor: 1 }) });
    await syncTradingData();
    const r = env.runtime;
    expect(r.renderNavigation).toHaveBeenCalledTimes(1);
    expect(r.loadMoreDocs).toHaveBeenCalledWith(false, true);
    expect(r.runLocalEmbeddingSync).toHaveBeenCalledTimes(1);
    expect(r.stopSpinner).toHaveBeenCalledTimes(1);
    expect(r.restoreSubmitButton).toHaveBeenCalledTimes(1);
    expect(getSyncIntervalMs()).toBe(3000);
  });

  it("backs off when nothing changed", async () => {
    serve({ pull: () => ({ results: [] }) });
    await syncTradingData();
    expect(env.runtime.renderNavigation).not.toHaveBeenCalled();
    expect(getSyncIntervalMs()).toBe(4500);
  });

  it("does nothing without a token", async () => {
    env.state.session = { hash: "h" };
    await syncTradingData();
    expect(invokeMock).not.toHaveBeenCalled();
    expect(env.runtime.stopSpinner).not.toHaveBeenCalled();
  });

  it("background sync is throttled to once per 30 s unless the throttle is reset", async () => {
    serve({ pull: () => ({ results: [] }) });
    await syncTradingInBackground();
    expect(env.runtime.stopSpinner).toHaveBeenCalledTimes(1);

    await syncTradingInBackground(); // same instant: throttled
    vi.setSystemTime(NOW + 29_000);
    await syncTradingInBackground(); // still inside the window
    expect(env.runtime.stopSpinner).toHaveBeenCalledTimes(1);

    vi.setSystemTime(NOW + 30_000);
    await syncTradingInBackground();
    expect(env.runtime.stopSpinner).toHaveBeenCalledTimes(2);

    resetTradingThrottle();
    await syncTradingInBackground();
    expect(env.runtime.stopSpinner).toHaveBeenCalledTimes(3);
  });
});

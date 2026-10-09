import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));

import { invoke } from "@tauri-apps/api/core";
import { bindModeRuntime, getSyncIntervalMs, resetSyncBackoff } from "../modes/runtime";
import { COMMERCE_API_HOST, syncCommerceData, syncCommerceInBackground } from "../modes/commerce";
import { makeDb, type FakeDb, type Row } from "./helpers/fakedb";
import { callsOf, makeRuntime, type FakeRuntimeOptions } from "./helpers/runtime";

const invokeMock = vi.mocked(invoke) as unknown as ReturnType<typeof vi.fn>;
const NOW = Date.UTC(2026, 9, 9, 12, 0, 0);
const DETECTED = "https://www.shop.example.co.kr/item?id=1";
const CC_EXAMPLE_CO_KR = "0xb2e020367adcbfdf936a3266bb8ab4515fb0623b"; // hashId("example.co.kr")
const CC_LOGIS_CENTER = "0x45a2a96bf8ae28042073444f3ec1dd7acaa8a961"; // hashId("logis.center")
const w = window as any;

let db: FakeDb;
let env: ReturnType<typeof makeRuntime>;

function setup(opts: FakeRuntimeOptions = {}, seed: Record<string, Row[]> = {}) {
  db = makeDb(seed);
  w.appDb = db; // lib/db.ts Select.* reads window.appDb
  env = makeRuntime({
    session: { hash: "h", token: "t", email: "me@x.com", flag: "KR" },
    appDb: db, // modes/commerce.ts writes through rt().appDb
    detectedUrl: DETECTED,
    itemTombstones: ["tomb"],
    currentTab: "list",
    ...opts,
  });
  bindModeRuntime(env.runtime);
}

/** The commerce worker answers `response`; Rust-side commands succeed with empty data. */
function respond(response: unknown) {
  invokeMock.mockImplementation(async (cmd: string) => {
    if (cmd === "proxy_fetch") return response;
    if (cmd === "get_known_users" || cmd === "get_known_pages") return [];
    return "ok";
  });
}

const workerParams = () => Object.fromEntries(new URL(callsOf(invokeMock, "proxy_fetch")[0].url).searchParams);
const upserted = (): Row[] => callsOf(invokeMock, "upsert_items")[0]?.items ?? [];
const base64Json = (value: unknown) => btoa(JSON.stringify(value));

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(NOW);
  resetSyncBackoff();
  document.body.innerHTML = "";
  setup();
});
afterEach(() => {
  vi.useRealTimers();
  delete w.appDb;
});

describe("syncCommerceData: request", () => {
  it("calls the commerce worker with session, sender, href and the root-domain cc", async () => {
    respond({ results: [] });
    await syncCommerceData();
    const [args] = callsOf(invokeMock, "proxy_fetch");
    const url = new URL(args.url);
    expect(url.origin + url.pathname).toBe(`${COMMERCE_API_HOST}/`);
    expect(args.method).toBe("GET");
    expect(workerParams()).toEqual({
      origin: COMMERCE_API_HOST,
      created_at: String(NOW),
      hash: "h",
      token: "t",
      href: DETECTED,
      sender: "me@x.com",
      cc: CC_EXAMPLE_CO_KR,
    });
    expect(env.runtime.stepQrSpinner).toHaveBeenCalled();
  });

  it("falls back to commerce.logis.center/tracking when the detected URL is localhost", async () => {
    setup({ detectedUrl: "http://localhost:1420/" });
    respond({ results: [] });
    await syncCommerceData();
    expect(workerParams()).toMatchObject({ href: "https://commerce.logis.center/tracking", cc: CC_LOGIS_CENTER });
  });

  it("keeps the sidebar context cc while a search tag is active", async () => {
    setup({ context: { cc: "CTX", bcc: "", ref: "" }, activeTags: [{ id: "type:goods", label: "#goods", type: "type", value: "goods" }] });
    respond({ results: [] });
    await syncCommerceData();
    expect(workerParams().cc).toBe("CTX");
  });

  it.fails(
    "BUG-COM-3 commerce.ts:63,271 (+8 other call sites) send `session_params`, but Rust proxy_fetch (lib.rs:4226, no rename_all) only reads camelCase `sessionParams`, so they are dropped",
    async () => {
      respond({ results: [] });
      await syncCommerceData();
      expect(callsOf(invokeMock, "proxy_fetch")[0]).toHaveProperty("sessionParams");
    },
  );
});

describe("syncCommerceData: storing server rows", () => {
  it("tags mode, recovers flag, absorbs root columns and preserves drafts before upsert_items; skips tombstones", async () => {
    respond({
      results: [
        { id: "b1", type: "BL", data: { title: "t" }, extra: "E", vessel: "V" },
        { id: "tomb", type: "goods", data: { title: "deleted locally" } },
        { id: "t1", type: "tracking", mode: "commerce", updated_at: 7, data: { title: "t", flag: "US" } },
      ],
    });
    await syncCommerceData();

    expect(upserted().map((r) => r.id)).toEqual(["b1", "t1"]);
    expect(upserted()[0]).toMatchObject({ mode: "shipping", flag: "KR", updated_at: 0, data: { title: "t", extra: "E", vessel: "V" } });
    expect(upserted()[1]).toMatchObject({ mode: "commerce", flag: "US", updated_at: 7 });
    expect(db.ids("items")).toEqual(["b1", "t1"]);
    expect(env.runtime.renderNavigation).toHaveBeenCalledTimes(1);
    expect(env.runtime.loadMoreDocs).toHaveBeenCalledWith(false, true);
    expect(env.runtime.runLocalEmbeddingSync).toHaveBeenCalledTimes(1);
    expect(getSyncIntervalMs()).toBe(3000);
  });

  it("routes members, page caches, talks and items to their own Dexie tables", async () => {
    respond({
      results: [
        { id: "m1", type: "member", data: { name: "N" } },
        { id: "p1", table: "pages", type: "tracking", data: { node: 1 } },
        { id: "k1", type: "talk", data: { text: "hi" } },
        { id: "g1", type: "goods", data: { title: "x" } },
      ],
    });
    await syncCommerceData();
    expect([db.ids("users"), db.ids("pages"), db.ids("talks"), db.ids("items")]).toEqual([["m1"], ["p1"], ["k1"], ["g1"]]);
  });

  it("only accepts rows newer than the rendered card or the local cache", async () => {
    document.body.innerHTML = '<div id="g1" data-updated-at="500"></div><div id="g2" data-updated-at="500"></div>';
    setup({}, { talks: [{ id: "k1", type: "talk", updated_at: 800 }] });
    respond({
      results: [
        { id: "g1", type: "goods", updated_at: 400, data: { title: "stale" } },
        { id: "g2", type: "goods", updated_at: 600, data: { title: "fresh" } },
        { id: "k1", type: "talk", updated_at: 700, data: { text: "stale" } },
        { id: "k2", type: "talk", updated_at: 1, data: { text: "new" } },
      ],
    });
    await syncCommerceData();
    expect(upserted().map((r) => r.id)).toEqual(["g2", "k2"]);
  });

  it("backs off and skips re-rendering when nothing changed", async () => {
    respond({ results: [] });
    await syncCommerceData();
    expect(callsOf(invokeMock, "upsert_items")).toHaveLength(0);
    expect(getSyncIntervalMs()).toBe(4500);
    expect(env.runtime.renderNavigation).not.toHaveBeenCalled();
    expect(env.runtime.stopSpinner).toHaveBeenCalledTimes(1);
    expect(env.runtime.restoreSubmitButton).toHaveBeenCalledTimes(1);
  });

  it("finishes cloud tasks the server no longer lists (after a 5 s grace period) and keeps running ones", async () => {
    const tasks = new Map<string, any>([
      ["search_1", { serverId: "srv1", kind: "search", createdAt: NOW - 10_000 }],
      ["task_2", { serverId: "gone", kind: "extract", createdAt: NOW - 10_000 }],
      ["task_3", { serverId: "", kind: "extract", createdAt: NOW - 1_000 }],
    ]);
    setup({ cloudPendingTasks: tasks });
    respond({ results: [{ table: "tasks", id: "srv1" }] });
    await syncCommerceData();
    expect(env.runtime.renderProgressToUI).toHaveBeenCalledWith({
      task_id: "search_1",
      category: "Cloud Queue",
      summary: "Processing on Logis Center...",
      spinner: "☁️",
    });
    expect(env.runtime.renderProgressToUI).toHaveBeenCalledWith({
      task_id: "task_2",
      category: "Done",
      summary: "Cloud AI extraction complete.",
      spinner: "✅",
    });
    expect([...tasks.keys()]).toEqual(["search_1", "task_3"]);
  });

  it("removes the pending-invite placeholder in Rust once the invitee appears as a member", async () => {
    setup({}, { users: [{ id: "pending_invite_1", type: "user", from: "0xme", to: "T", data: { email: "new@x.com", is_pending: true } }] });
    respond({ results: [{ id: "0xm1", type: "member", to: "T", data: { email: "new@x.com" } }] });
    await syncCommerceData();
    expect(callsOf(invokeMock, "delete_document")).toEqual([{ uuid: "pending_invite_1" }]);
  });

  it.fails("BUG-COM-2 commerce.ts:477-481 the pending-invite placeholder is deleted only in Rust; the Dexie users row (which Select.users reads first) survives", async () => {
    setup({}, { users: [{ id: "pending_invite_1", type: "user", from: "0xme", to: "T", data: { email: "new@x.com", is_pending: true } }] });
    respond({ results: [{ id: "0xm1", type: "member", to: "T", data: { email: "new@x.com" } }] });
    await syncCommerceData();
    expect(db.ids("users")).not.toContain("pending_invite_1");
  });

  it.fails("BUG-COM-1 commerce.ts:281-308,445 foreground sync has no base64 branch: JSON.parse throws after upsert_items, so Dexie is never updated", async () => {
    respond({ results: [{ id: "g1", type: "goods", created_at: 5, data: base64Json({ title: "x".repeat(60) }) }] });
    await syncCommerceData();
    expect(db.ids("items")).toContain("g1");
  });
});

describe("syncCommerceInBackground", () => {
  it("decodes base64 JSON string data before writing LanceDB and Dexie", async () => {
    const payload = { title: "x".repeat(60) };
    respond({ results: [{ id: "g1", type: "goods", created_at: 5, data: base64Json(payload) }] });
    await syncCommerceInBackground();
    expect(upserted()[0].data).toEqual(payload);
    expect(db.rows("items")[0]).toMatchObject({ id: "g1", mode: "commerce", data: payload });
    expect(env.runtime.loadMoreDocs).toHaveBeenCalledWith(false, true);
  });

  it("requires hash + email", async () => {
    setup({ session: { hash: "h", token: "t" } });
    await syncCommerceInBackground();
    expect(invokeMock).not.toHaveBeenCalled();
  });
});

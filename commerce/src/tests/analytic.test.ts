import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => true), open: vi.fn() }));

import { invoke } from "@tauri-apps/api/core";
import { bindModeRuntime } from "../modes/runtime";
import {
  ANALYTIC_API_HOST,
  extractAnalyticText,
  finalizeAnalyticBubbles,
  resetAnalyticThrottle,
  runAnalyticStructuring,
  syncAnalyticsData,
  syncAnalyticsInBackground,
} from "../modes/analytic";
import { makeDb, type FakeDb } from "./helpers/fakedb";
import { callsOf, makeRuntime, type FakeRuntimeOptions } from "./helpers/runtime";

const invokeMock = vi.mocked(invoke) as unknown as ReturnType<typeof vi.fn>;
const NOW = Date.UTC(2026, 9, 9, 12, 0, 0);
const SHOP_CC = "0xdf6b17e866098876f6e38cca1aad93cde91d6217"; // hashId("shop.com")
const CLICK_BCC = "0x2872a7c81be893bee2dddac334b6b25bbabfc206"; // hashId("click" + SHOP_CC)

let env: ReturnType<typeof makeRuntime>;
function setup(opts: FakeRuntimeOptions = {}) {
  env = makeRuntime({ session: { hash: "h", token: "t" }, searchMode: "analytic", currentTab: "list", ...opts });
  bindModeRuntime(env.runtime);
}
function serve(routes: Record<string, (args: any) => unknown>) {
  invokeMock.mockImplementation(async (cmd: string, args: any) => (routes[cmd] ? routes[cmd](args) : undefined));
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(NOW);
  resetAnalyticThrottle();
  setup();
});
afterEach(() => vi.useRealTimers());

describe("extractAnalyticText", () => {
  it.each<[unknown, string]>([
    [{ action: "  a ", summary: "s" }, "a"],
    [{ action: ["raw event"], summary: " s " }, "s"],
    [{ action: "   ", cross_action_flow: "f" }, "f"],
    [{ intent_evolution: "i", text: "t" }, "i"],
    [{ text: " t " }, "t"],
    [{}, ""],
    [null, ""],
    [undefined, ""],
  ])("extractAnalyticText(%j) -> %j", (parsed, expected) => {
    expect(extractAnalyticText(parsed)).toBe(expected);
  });
});

describe("runAnalyticStructuring", () => {
  it("structures pending events and stamps updated_at on the structured Dexie rows only", async () => {
    const db: FakeDb = makeDb({
      items: [
        { id: "x1", mode: "analytic", updated_at: 0, data: { summary: "S" } },
        { id: "x2", mode: "analytic", updated_at: 0, data: { action: ["raw"] } },
        { id: "x3", mode: "analytic", updated_at: 5, data: { summary: "S" } },
        { id: "x4", mode: "commerce", updated_at: 0, data: { summary: "S" } },
        { id: "x5", mode: "analytic", updated_at: 0, data: { action: "clicked" } },
      ],
    });
    setup({ appDb: db });
    invokeMock.mockResolvedValue({ processed: 2 });

    expect(await runAnalyticStructuring()).toBe(2);
    expect(invokeMock).toHaveBeenCalledWith("structure_pending_analytics", { limit: 20, devicePreference: null });
    const byId = Object.fromEntries(db.rows("items").map((r) => [r.id, r]));
    expect([byId.x1.updated_at, byId.x1.data.updated_at, byId.x5.updated_at]).toEqual([NOW, NOW, NOW]);
    expect([byId.x2.updated_at, byId.x3.updated_at, byId.x4.updated_at]).toEqual([0, 5, 0]);
    expect(env.runtime.renderNavigation).toHaveBeenCalledTimes(1);
    expect(env.runtime.loadMoreDocs).toHaveBeenCalledWith(false, true);
  });

  it("returns 0 without invoking while the app is busy, and 0 on backend errors", async () => {
    env.state.busy = true;
    expect(await runAnalyticStructuring()).toBe(0);
    expect(invokeMock).not.toHaveBeenCalled();

    env.state.busy = false;
    invokeMock.mockRejectedValue(new Error("gpu"));
    expect(await runAnalyticStructuring()).toBe(0);
    invokeMock.mockResolvedValue({ processed: 0, skipped: "searching" });
    expect(await runAnalyticStructuring()).toBe(0);
  });
});

describe("finalizeAnalyticBubbles", () => {
  it("marks unfinished analytic_sync bubbles as DONE and leaves finished / other bubbles alone", async () => {
    document.body.innerHTML = `
      <div class="chat-talks">
        <div class="chat-talk task-bubble" id="analytic_sync_1" data-task-id="analytic_sync_1" data-status="1">
          <div class="content">Structuring...</div>
          <div class="status-bar"><span class="active-spinner">*</span> PROCESSING</div>
        </div>
        <div class="chat-talk task-bubble" id="analytic_sync_2" data-status="10"><div class="content">queued</div></div>
        <div class="chat-talk task-bubble" id="analytic_sync_3" data-task-id="analytic_sync_3" data-status="9"><div class="content">old</div></div>
        <div class="chat-talk task-bubble" id="analytic_sync_4" data-task-id="analytic_sync_4" data-status="6"><div class="content">err</div></div>
        <div class="chat-talk task-bubble" id="search_1" data-task-id="search_1" data-status="1"><div class="content">search</div></div>
      </div>`;

    await finalizeAnalyticBubbles(3);

    const b1 = document.getElementById("analytic_sync_1")!;
    expect(b1.dataset.status).toBe("9");
    expect(b1.querySelector(".content")!.textContent).toBe("Analytic structuring complete (3 event(s)).");
    expect(b1.querySelector(".status-bar")!.textContent).toContain("DONE");
    expect(b1.querySelector(".active-spinner")).toBeNull();
    expect(document.getElementById("analytic_sync_2")!.dataset.status).toBe("9"); // matched by element id
    expect(document.getElementById("analytic_sync_3")!.querySelector(".content")!.textContent).toBe("old");
    expect(document.getElementById("analytic_sync_4")!.dataset.status).toBe("6");
    expect(document.getElementById("search_1")!.dataset.status).toBe("1");
  });

  it("is a no-op without a .chat-talks container", async () => {
    document.body.innerHTML = "";
    await expect(finalizeAnalyticBubbles(1)).resolves.toBeUndefined();
  });
});

describe("syncAnalyticsData", () => {
  it("pulls raw events per registered site with its OAuth credentials and stores them as analytic items", async () => {
    const db = makeDb();
    setup({ appDb: db, kv: { oauth_registered_sites: [{ host: "https://shop.com", client_id: "cid", client_secret: "sec" }] } });
    serve({
      proxy_fetch: () => ({
        results: [
          {
            id: "ev1",
            type: "click",
            created_at: 123,
            data: JSON.stringify({ action: ["click #buy"], summary: "User clicked buy" }),
          },
        ],
      }),
      upsert_items: () => "ok",
      structure_pending_analytics: () => ({ processed: 0 }),
    });

    await syncAnalyticsData();

    const [fetchArgs] = callsOf(invokeMock, "proxy_fetch");
    const url = new URL(fetchArgs.url);
    expect(url.origin).toBe(ANALYTIC_API_HOST);
    expect(Object.fromEntries(url.searchParams)).toEqual({
      origin: "https://console.logis.center",
      created_at: String(NOW + 60_000),
      hash: "h",
      token: "t",
      href: "https://shop.com/",
      cc: SHOP_CC,
      client_id: "cid",
      client_secret: "sec",
    });

    const [upsert] = callsOf(invokeMock, "upsert_items");
    expect(upsert.items).toHaveLength(1);
    expect(upsert.items[0]).toMatchObject({
      id: "ev1",
      type: "click",
      mode: "analytic",
      status: 9,
      cc: SHOP_CC,
      bcc: CLICK_BCC,
      created_at: 123,
      updated_at: 0, // raw (array) events stay un-structured
      text: "User clicked buy",
      masked_text: "User clicked buy",
      data: { origin: "https://shop.com", mode: "analytic", updated_at: 0, text: "User clicked buy" },
    });
    expect(db.ids("items")).toEqual(["ev1"]);
    expect(env.runtime.renderNavigation).toHaveBeenCalledTimes(1);
    expect(env.runtime.runLocalEmbeddingSync).toHaveBeenCalledTimes(1);
    expect(env.runtime.stopSpinner).toHaveBeenCalledTimes(1);
    expect(env.runtime.restoreSubmitButton).toHaveBeenCalledTimes(1);
  });

  it("skips sites without credentials, ignores localhost/*.logis.center, and never overwrites locally structured rows", async () => {
    const db = makeDb({
      items: [
        { id: "ev1", mode: "analytic", updated_at: 500, data: { origin: "https://other.com", summary: "structured" } },
        { id: "r2", mode: "analytic", updated_at: 1, data: { origin: "https://console.logis.center" } },
      ],
    });
    setup({
      appDb: db,
      detectedUrl: "http://localhost:1420/x",
      kv: { oauth_registered_sites: [{ host: "Shop.com", client_id: "cid" }] },
    });
    serve({
      proxy_fetch: () => ({ results: [{ id: "ev1", type: "click", updated_at: 900, data: "{}" }] }),
      structure_pending_analytics: () => ({ processed: 0 }),
    });

    await syncAnalyticsData();

    const fetches = callsOf(invokeMock, "proxy_fetch");
    expect(fetches).toHaveLength(1); // other.com has no API key; logis.center/localhost are never queried
    expect(new URL(fetches[0].url).searchParams.get("href")).toBe("https://shop.com/");
    expect(callsOf(invokeMock, "upsert_items")).toHaveLength(0);
    expect(db.rows("items").find((r) => r.id === "ev1")!.data.summary).toBe("structured");
    expect(vi.mocked(console.warn).mock.calls.some(([msg]) => String(msg).includes("'https://other.com'"))).toBe(true);
  });

  it("returns early without sites or without a session", async () => {
    await syncAnalyticsData();
    expect(callsOf(invokeMock, "proxy_fetch")).toHaveLength(0);
    expect(env.runtime.stopSpinner).toHaveBeenCalledTimes(1);

    env.state.session = { hash: "" };
    await syncAnalyticsData();
    expect(env.runtime.stopSpinner).toHaveBeenCalledTimes(1);
  });

  it("background sync is throttled to once per 30 s unless the throttle is reset", async () => {
    await syncAnalyticsInBackground();
    await syncAnalyticsInBackground();
    expect(env.runtime.stopSpinner).toHaveBeenCalledTimes(1);
    resetAnalyticThrottle();
    await syncAnalyticsInBackground();
    expect(env.runtime.stopSpinner).toHaveBeenCalledTimes(2);
  });
});

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => true), open: vi.fn() }));

import { invoke } from "@tauri-apps/api/core";
import { bindModeRuntime } from "../modes/runtime";
import {
  extractBalancedJson,
  fetchOAuthRegisteredSites,
  fetchOAuthSiteCount,
  fetchOAuthSitePaths,
  getOAuthCredentialForOrigin,
  normalizeOAuthHost,
  oauthApiFetch,
  parseOAuthApiResponse,
  renderOAuthSitesUI,
  submitOAuthRegistration,
} from "../modes/oauth";
import { makeRuntime } from "./helpers/runtime";

const invokeMock = vi.mocked(invoke) as unknown as ReturnType<typeof vi.fn>;
const NOW = Date.UTC(2026, 9, 9, 12, 0, 0);

// api.oauth.network answers with an HTML page that postMessage()s a JSON payload whose
// `cookies` field is itself a JSON string (double stringified).
const COOKIES = { email: "a@b.com", client_id: "0xme", "#length": "2", "#0": "0xA:0xB@shop.com", "#1": "bad" };
const PAYLOAD = {
  rows: [{ client_id: "0xabc", client_secret: "0xdef", Cc: "/p" }],
  cookies: JSON.stringify(COOKIES),
  count: 7,
  query: { x: 1 },
};
const page = (payload: object) =>
  `<html><body><script>parent.postMessage(JSON.stringify(${JSON.stringify(payload)}), "*");</script></body></html>`;
const NOT_AUTHED = { text: page({ rows: [], cookies: "{}" }) };
const EMPTY = { rows: [], cookies: {}, count: 0, query: {} };

let env: ReturnType<typeof makeRuntime>;
beforeEach(() => {
  env = makeRuntime({ session: { hash: "h", token: "t", email: "a@b.com" }, searchMode: "analytic" });
  bindModeRuntime(env.runtime);
});

describe("normalizeOAuthHost", () => {
  it.each([
    ["example.com", "https://example.com"],
    ["  Example.COM  ", "https://example.com"],
    ["HTTP://Example.COM/path?q=1", "https://example.com"],
    ["https://example.com:443", "https://example.com"],
    ["http://example.com:80", "https://example.com"],
    ["example.com:8080", "https://example.com:8080"],
    ["user:pass@example.com", "https://example.com"],
    ["www.example.com", "https://www.example.com"],
    ["한국.kr", "https://xn--3e0b707e.kr"],
  ])("normalizeOAuthHost(%j) -> %j", (raw, expected) => {
    expect(normalizeOAuthHost(raw)).toBe(expected);
  });

  it.each<[unknown]>([[""], ["   "], [null], [123], ["https://"], ["https://exa mple.com"], ["javascript:alert(1)"]])(
    "normalizeOAuthHost(%j) -> '' (invalid)",
    (raw) => {
      expect(normalizeOAuthHost(raw)).toBe("");
    },
  );
});

describe("extractBalancedJson", () => {
  it("returns the first balanced object, ignoring braces inside strings and escaped quotes", () => {
    const src = 'x {"a":"}{\\"","b":{"c":1}} tail {"z":1}';
    const json = extractBalancedJson(src, 0);
    expect(json).toBe('{"a":"}{\\"","b":{"c":1}}');
    expect(JSON.parse(json)).toEqual({ a: '}{"', b: { c: 1 } });
  });

  it("starts searching at fromIndex", () => {
    expect(extractBalancedJson('{"a":1} {"b":2}', 1)).toBe('{"b":2}');
  });

  it("returns '' when there is no object or it never closes", () => {
    expect(extractBalancedJson("no json here", 0)).toBe("");
    expect(extractBalancedJson('{"a":{"b":1}', 0)).toBe("");
  });
});

describe("parseOAuthApiResponse", () => {
  it("extracts the postMessage payload from {text: html} and parses double-stringified cookies", () => {
    expect(parseOAuthApiResponse({ text: page(PAYLOAD) })).toEqual({
      rows: PAYLOAD.rows,
      cookies: COOKIES,
      count: 7,
      query: { x: 1 },
    });
  });

  it("accepts the raw HTML string as well", () => {
    expect(parseOAuthApiResponse(page(PAYLOAD)).cookies).toEqual(COOKIES);
  });

  it("passes an already-parsed JSON response through", () => {
    expect(parseOAuthApiResponse({ rows: [], cookies: { a: 1 } })).toEqual({ rows: [], cookies: { a: 1 }, count: 0, query: {} });
  });

  it("returns an empty result for missing, unbalanced or non-JSON payloads", () => {
    expect(parseOAuthApiResponse(null)).toEqual(EMPTY);
    expect(parseOAuthApiResponse({ text: "no json here" })).toEqual(EMPTY);
    expect(parseOAuthApiResponse({ text: 'JSON.stringify({"rows": [1], "cookies": "x"' })).toEqual(EMPTY);
  });

  it("normalizes a non-array rows field and an unparsable cookie string", () => {
    expect(parseOAuthApiResponse('JSON.stringify({"rows":"x","cookies":"not json","count":0})')).toEqual(EMPTY);
  });
});

describe("oauthApiFetch", () => {
  it("GET: drops empty params, repeats array params, sends JSON content type + Referer", async () => {
    invokeMock.mockResolvedValue({ text: page(PAYLOAD) });
    const parsed = await oauthApiFetch({ hash: "h", arr: ["1", "2"], skip: "", nil: null, undef: undefined });
    expect(invokeMock).toHaveBeenCalledTimes(1);
    const [cmd, args] = invokeMock.mock.calls[0];
    expect(cmd).toBe("proxy_fetch");
    expect(args.url).toBe("https://api.oauth.network/?hash=h&arr=1&arr=2");
    expect(args.method).toBe("GET");
    expect(args.headers).toEqual({ "Content-Type": "application/json", Referer: "https://oauth.network/" });
    expect(args).not.toHaveProperty("body");
    expect(parsed.count).toBe(7);
  });

  it("POST with an empty query hits the bare host with a form-encoded body", async () => {
    invokeMock.mockResolvedValue({ rows: [] });
    await oauthApiFetch({}, { method: "POST", body: { host: "x" } });
    const [, args] = invokeMock.mock.calls[0];
    expect(args.url).toBe("https://api.oauth.network/");
    expect(args.method).toBe("POST");
    expect(args.headers["Content-Type"]).toBe("application/x-www-form-urlencoded");
    expect(args.body).toEqual({ host: "x" });
  });
});

describe("fetchOAuthRegisteredSites", () => {
  it("replaces kv oauth_registered_sites with the server list and stores the client address", async () => {
    invokeMock.mockResolvedValue({ text: page(PAYLOAD) });
    await fetchOAuthRegisteredSites();
    expect(invokeMock.mock.calls[0][1].url).toBe("https://api.oauth.network/?hash=h&token=t");
    expect(env.kv.get("oauth_client_address")).toBe("0xme");
    expect(env.kv.get("oauth_registered_sites")).toEqual([
      { host: "https://shop.com", client_id: "0xA", client_secret: "0xB", registered_at: expect.any(Number) },
    ]);
  });

  it("keeps the local list when the server session is not established", async () => {
    env.kv.set("oauth_registered_sites", ["keep"]);
    invokeMock.mockResolvedValue(NOT_AUTHED);
    await fetchOAuthRegisteredSites();
    expect(env.runtime.kvSet).not.toHaveBeenCalled();
    expect(env.kv.get("oauth_registered_sites")).toEqual(["keep"]);
  });

  it("does nothing without hash/token and swallows transport errors", async () => {
    env.state.session = { hash: "" };
    await fetchOAuthRegisteredSites();
    expect(invokeMock).not.toHaveBeenCalled();

    env.state.session = { hash: "h", token: "t" };
    invokeMock.mockRejectedValue(new Error("offline"));
    await expect(fetchOAuthRegisteredSites()).resolves.toBeUndefined();
    expect(env.runtime.kvSet).not.toHaveBeenCalled();
  });
});

describe("getOAuthCredentialForOrigin", () => {
  beforeEach(() => {
    env.kv.set("oauth_registered_sites", [
      { host: "https://Shop.com:8443", client_id: "id1", client_secret: null },
      { host: "https://shop.com", client_id: "" },
      { host: "https://shop.com", client_id: "id2", client_secret: "s2" },
    ]);
  });

  it.each([
    ["https://shop.com", { client_id: "id2", client_secret: "s2" }],
    ["http://SHOP.com/path", { client_id: "id2", client_secret: "s2" }],
    ["shop.com", { client_id: "id2", client_secret: "s2" }],
    ["shop.com:8443", { client_id: "id1", client_secret: "" }],
    ["https://www.shop.com", null],
    ["", null],
  ])("credential for %j -> %j", async (origin, expected) => {
    expect(await getOAuthCredentialForOrigin(origin)).toEqual(expected);
  });

  it("returns null when no sites are registered", async () => {
    env.kv.delete("oauth_registered_sites");
    expect(await getOAuthCredentialForOrigin("https://shop.com")).toBeNull();
  });
});

describe("site statistics", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(NOW);
  });
  afterEach(() => vi.useRealTimers());

  it("fetchOAuthSiteCount queries the [now - hours, now] window by referer", async () => {
    invokeMock.mockResolvedValue({ text: page(PAYLOAD) });
    expect(await fetchOAuthSiteCount("shop.com", 24)).toBe(7);
    expect(invokeMock.mock.calls[0][1].url).toBe(
      "https://api.oauth.network/?referer=https%3A%2F%2Fshop.com&id=%23LOG&cnt=true" +
        "&date=2026-10-08T12%3A00%3A00.000Z&date=2026-10-09T12%3A00%3A00.000Z",
    );
  });

  it("fetchOAuthSiteCount returns 0 for invalid hosts and transport errors", async () => {
    expect(await fetchOAuthSiteCount("", 1)).toBe(0);
    expect(invokeMock).not.toHaveBeenCalled();
    invokeMock.mockRejectedValue(new Error("offline"));
    expect(await fetchOAuthSiteCount("shop.com", 1)).toBe(0);
  });

  it("fetchOAuthSitePaths returns the non-empty distinct Cc values", async () => {
    invokeMock.mockResolvedValue({ text: page({ rows: [{ Cc: "/a" }, { Cc: "" }, { Cc: "/b" }], cookies: "{}" }) });
    expect(await fetchOAuthSitePaths("shop.com")).toEqual(["/a", "/b"]);
    expect(invokeMock.mock.calls[0][1].url).toBe("https://api.oauth.network/?referer=https%3A%2F%2Fshop.com&distinct=Cc&id=%23LOG");
    expect(await fetchOAuthSitePaths("")).toEqual([]);
  });
});

describe("submitOAuthRegistration", () => {
  it("requires a session and a valid host", async () => {
    env.state.session = { hash: "" };
    expect(await submitOAuthRegistration("shop.com")).toEqual({
      success: false,
      client_id: "",
      client_secret: "",
      error: "로그인이 필요합니다.",
      removed: false,
    });
    env.state.session = { hash: "h", token: "t" };
    const res = await submitOAuthRegistration("https://");
    expect(res.success).toBe(false);
    expect(res.error).toContain("도메인 형식");
    expect(invokeMock).not.toHaveBeenCalled();
  });

  it("returns the issued credentials and refreshes the site list on success", async () => {
    invokeMock.mockResolvedValue({ text: page(PAYLOAD) });
    expect(await submitOAuthRegistration("shop.com")).toEqual({
      success: true,
      client_id: "0xabc",
      client_secret: "0xdef",
      error: "",
      removed: false,
    });
    const [, post] = invokeMock.mock.calls[0];
    expect(post.method).toBe("POST");
    expect(post.body).toEqual({ host: "https://shop.com", hash: "h", token: "t" });
    expect(invokeMock).toHaveBeenCalledTimes(2); // POST + fetchOAuthRegisteredSites()
  });

  it("sends the existing credentials when removing a site", async () => {
    invokeMock.mockResolvedValue({ text: page(PAYLOAD) });
    const res = await submitOAuthRegistration("shop.com", { client_id: "c", client_secret: "s" });
    expect(res.removed).toBe(true);
    expect(invokeMock.mock.calls[0][1].body).toEqual({ host: "https://shop.com", hash: "h", token: "t", client_id: "c", client_secret: "s" });
  });

  it("explains the ownership meta tag (content = hashId(email)) when verification fails", async () => {
    invokeMock.mockResolvedValue(NOT_AUTHED);
    const res = await submitOAuthRegistration("shop.com");
    expect(res.success).toBe(false);
    expect(res.error).toContain('<meta name="oauth-network-verification" content="0xf7932301cf309f69791a5e67be5a400b0c26106d" />');
  });

  it("reports transport errors", async () => {
    invokeMock.mockRejectedValue(new Error("boom"));
    expect(await submitOAuthRegistration("shop.com")).toMatchObject({ success: false, error: "Error: boom", removed: false });
  });
});

describe("renderOAuthSitesUI", () => {
  let pageList: HTMLElement;
  beforeEach(() => {
    document.body.innerHTML = '<div id="nav-list-pages"></div>';
    pageList = document.getElementById("nav-list-pages")!;
    invokeMock.mockResolvedValue(NOT_AUTHED); // server refresh keeps the local kv list
  });

  it("renders one item per registered site and toggles its token panel", async () => {
    env.kv.set("oauth_registered_sites", [{ host: "https://shop.com", client_id: "0xA", client_secret: "0xB" }]);
    await renderOAuthSitesUI(pageList);
    const items = pageList.querySelectorAll(".oauth-site-item");
    expect(items).toHaveLength(1);
    expect(items[0].querySelector("span")!.textContent).toBe("shop.com");
    const tokens = pageList.querySelector<HTMLElement>(".oauth-site-tokens")!;
    const more = pageList.querySelector<HTMLButtonElement>(".btn-oauth-more")!;
    expect(tokens.style.display).toBe("none");
    more.click();
    expect(tokens.style.display).toBe("block");
    expect(more.textContent).toBe("fold");
  });

  it("renders nothing outside the analytic tab", async () => {
    env = makeRuntime({ session: { hash: "h", token: "t", email: "a@b.com" }, searchMode: "commerce" });
    bindModeRuntime(env.runtime);
    env.kv.set("oauth_registered_sites", [{ host: "https://shop.com", client_id: "0xA" }]);
    await renderOAuthSitesUI(pageList);
    expect(pageList.innerHTML).toBe("");
    expect(invokeMock).not.toHaveBeenCalled();
  });

  it.fails("BUG-OAUTH-1 oauth.ts:362-377 server-provided client_id/secret are inserted via innerHTML unescaped", async () => {
    env.kv.set("oauth_registered_sites", [{ host: "https://shop.com", client_id: '<b id="injected">x</b>', client_secret: "s" }]);
    await renderOAuthSitesUI(pageList);
    expect(pageList.querySelector(".oauth-site-item")).not.toBeNull();
    expect(pageList.querySelector("#injected")).toBeNull();
  });
});

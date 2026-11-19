// trading 차분 시험의 시나리오: D1 에 미리 넣을 행, S3 HEAD 가 돌려줄 메타, commerce 의 팀 응답, 차례로 보낼 요청.
// 요청: { method, query, headers, cookie, body(글 · 객체 · Buffer), gzip(true · 함수), cf(없으면 { country: "KR" }), noCf }
import { ethers } from "ethers";
import { gzipSync, deflateSync } from "node:zlib";

const T0 = 1760000000000; // prelude.mjs 의 가상 시계 시작
const hashId = (t) => ethers.computeAddress(ethers.hashMessage(t)).toLowerCase();
const CC = hashId("trading.logis.center");

// 사용자 A (S3 hash/token), 그 주소로 만든 본인 팀
const HASH_A = "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d";
const TOKEN_A = "tok-a";
const ADDR_A = "0x" + "ab".repeat(20);
const OWN_A = hashId(ADDR_A);
const TEAM_B = hashId("team-b");
const TEAM_C = hashId("team-c");
const meta = (email, address, token) => {
  const h = {};
  if (email !== undefined) h["x-amz-meta-email"] = email;
  if (address !== undefined) h["x-amz-meta-address"] = address;
  if (token !== undefined) h["x-amz-meta-token"] = token;
  return h;
};
const S3_A = { [HASH_A]: { status: 200, headers: meta("a@example.com", ADDR_A, TOKEN_A) } };
const AUTH = `hash=${HASH_A}; token=${TOKEN_A}`;
const JSON_CT = { "content-type": "application/json" };
const b64gz = (o) => gzipSync(Buffer.from(JSON.stringify(o))).toString("base64");
const arrgz = (o) => Array.from(gzipSync(Buffer.from(JSON.stringify(o))));

// 미리 넣을 행 (data 가 객체면 gzip BLOB). synced_at 을 주지 않으면 migrations/0002 가 옛 행에 채우는 값과 같게
const row = (id, over = {}) => {
  const r = {
    id, type: "BL", flag: "kr", from: ADDR_A, to: OWN_A, cc: CC, bcc: hashId("BL" + CC), ref: hashId(OWN_A + CC + "#R1"),
    mode: "shipping", digest: "d-" + id, data: { id, doc_number: id.toUpperCase() }, created_at: T0 - 50000, updated_at: 0, ...over,
  };
  if (!("synced_at" in over)) r.synced_at = Math.max(r.created_at ?? 0, r.updated_at ?? 0);
  return r;
};

// ── 앱(web/desktop/src/modes/trading.ts tradingApiFetch → apps/core proxy_fetch · reqwest)이 보내는 꼴 그대로 ──
//  GET  /?hash=&token=&to=<S.team>&since=<저장한 cursor>&created_at=<now+60s 상한>&limit=1000   Content-Type: application/json
//  POST /?hash=&token=&to=<S.team>   Content-Type: application/json · Content-Encoding: gzip, 본문은 gzip(키를 정렬한 JSON {items:[…]})
//  (session_params 는 null 이라 proxy_fetch 가 쿼리에 아무것도 더하지 않는다 · Cookie · Origin 없음 · DELETE 는 쓰지 않는다)
const sortKeys = (v) => (Array.isArray(v) ? v.map(sortKeys) : v && typeof v === "object" ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])])) : v);
const appQuery = (team, extra = {}) => {
  const sp = new URLSearchParams();
  sp.append("hash", HASH_A);
  sp.append("token", TOKEN_A);
  if (team) sp.append("to", team);
  for (const [k, v] of Object.entries(extra)) if (v !== undefined && v !== null && v !== "") sp.append(k, String(v));
  return sp.toString();
};
const APP_UPPER = T0 + 9 * 3600 * 1000 + 60000; // max(now, now - timezoneOffset) + 60 s (KST)
const appPull = (team, since) => ({ query: typeof since === "function" ? (prev) => appQuery(team, { since: since(prev), created_at: APP_UPPER, limit: 1000 }) : appQuery(team, { since, created_at: APP_UPPER, limit: 1000 }), headers: JSON_CT });
const appPush = (team, items) => ({
  method: "POST", query: appQuery(team), headers: { ...JSON_CT, "content-encoding": "gzip" }, gzip: true,
  // pushTradingData 의 payload: id · type · flag · digest · created_at · updated_at · data (compress_value 가 키를 정렬)
  body: sortKeys({ items: items.map((r) => ({ id: r.id, type: r.type, flag: r.flag || "", digest: r.digest || "", created_at: r.created_at || 0, updated_at: r.updated_at || 0, data: r.data || {} })) }),
});
const lastCursor = (prev) => prev[prev.length - 1].body.cursor;
const ZERO_TEAM = hashId("0x0000000000000000000000000000000000000000"); // main.ts: session.team 이 없을 때의 자리표
const ROWS = [
  row("bl-1", { created_at: T0 - 50000, updated_at: 0 }),
  row("ci-1", { type: "CI", bcc: hashId("CI" + CC), created_at: T0 - 40000, updated_at: T0 - 1000 }),
  row("po-1", { type: "PO", bcc: hashId("PO" + CC), ref: hashId(OWN_A + CC + "#PO1"), created_at: T0 - 30000, updated_at: T0 - 20000 }),
  row("old-1", { created_at: T0 - 900000, updated_at: T0 - 800000 }),
  row("future-1", { created_at: T0 + 120000, updated_at: 0 }),
  row("other-team", { to: TEAM_C }),
  row("other-mode", { mode: "commerce" }),
  row("nomode-1", { mode: null, digest: null, created_at: T0 - 20000 }),
  row("sd-1", { type: "shipping_doc", created_at: T0 - 10000 }),
];

// 보낼 무역 서류 (여러 data 꼴 · 허브 키 갈래 · 거부)
const ITEMS = () => [
  { id: " BL-NEW-1 ", type: "bl", data: { header: { doc_number: "bl 100", reference_po: "po-77" }, logistics: { vessel: "EVER" } }, created_at: T0 - 5000 },
  { id: "ci-new-1", doc_type: "Ci", data: b64gz({ items: [{ reference_bl: "ＢＬ－１００" }], doc_number: "ci-9" }), updated_at: "5" },
  { id: "pl-new-1", type: "PL", data: arrgz({ containers: [{ container_number: "MSCU 1234567" }] }), flag: " JP ", digest: " given-digest " },
  { id: "po-new-1", type: "PO", data: { doc_number: "PO-1", created_at: T0 - 7000, updated_at: null, flag: "CN" } },
  { id: "lc-new-1", type: "LC", data: { buffer: arrgz({ doc_number: "LC-5" }) } },
  { id: "bad-b64", type: "INS", data: "!!not base64!!" },
  { id: "bad-gz", type: "INS", data: Buffer.from("plain").toString("base64") },
  { id: "arr-data", type: "BK", data: [1, 2, 3] },
  { id: "num-id", type: "SWB", data: { reference_number: 4.5e21 } },
  { id: 12345, type: "SR", data: { contract_number: "계약-７７" }, created_at: "0", updated_at: "abc" },
  { id: "sd-new", type: "shipping_doc", data: {} },
  { id: "inv-1", type: "invoice", data: {} },
  { id: "nt-1", data: {} },
  { type: "BL", data: {} },
  null,
  "str-item",
  { id: "dup-1", type: "BL", data: { doc_number: "x" } },
  { id: "DUP-1", type: "BL", data: { doc_number: "y" } },
  { id: "kv", type: "HBL", data: { "2": "b", "10": "a", z: { tier: 1, score: 2, k: [{ time: 1 }] }, embed: [1] } },
];

export const SCENARIOS = [
  // ── OPTIONS · 다른 메서드 ──
  { name: "options_session", s3: S3_A, requests: [{ method: "OPTIONS", cookie: AUTH, headers: { origin: "https://app.logis.center" } }] },
  { name: "put_other_method", s3: S3_A, requests: [{ method: "PUT", cookie: AUTH, body: "x" }, { method: "HEAD", cookie: AUTH }] },

  // ── 자격증명 없음 (bootstrap PUT) ──
  { name: "get_html_bootstrap", requests: [{ headers: { "x-real-ip": "proxy 10.1.2.3, edge" } }] },
  { name: "get_json_unauth", requests: [{ headers: JSON_CT, cf: { country: "JP" } }, { headers: JSON_CT, cf: {} }, { headers: JSON_CT, cf: { country: "Constructor" } }] },
  { name: "post_unauth", requests: [{ method: "POST", body: ITEMS(), headers: JSON_CT }] },
  { name: "delete_unauth", requests: [{ method: "DELETE", query: "id=bl-1" }] },
  { name: "missing_cf", requests: [{ noCf: true, headers: JSON_CT }] },

  // ── 세션 ──
  { name: "get_html_session", s3: S3_A, requests: [{ cookie: AUTH + "; theme=dark; bad=a=b; =x; __proto__=p; 2=two" }] },
  { name: "session_query_creds", s3: S3_A, rows: ROWS, requests: [{ query: `hash=${HASH_A}&token=${TOKEN_A}`, headers: JSON_CT }] },
  { name: "session_hash_mismatch", s3: { ...S3_A, other: { status: 200, headers: meta("o@x.io", ADDR_A, "tok-o") } },
    requests: [{ cookie: AUTH, query: "hash=other&token=tok-o", headers: JSON_CT }] },
  { name: "session_wrong_token", s3: S3_A, requests: [{ cookie: `hash=${HASH_A}; token=nope`, headers: JSON_CT }] },
  { name: "session_null_token", s3: { [HASH_A]: { status: 200, headers: meta("a@example.com", ADDR_A, undefined) } }, rows: ROWS,
    requests: [{ cookie: `hash=${HASH_A}`, headers: JSON_CT }] },
  { name: "s3_404_bootstrap", requests: [{ cookie: AUTH, headers: JSON_CT }] },
  { name: "s3_500_retries", s3: { [HASH_A]: { status: 500 } }, requests: [{ cookie: AUTH, headers: JSON_CT }] },
  { name: "s3_403_plain_headers", s3: { [HASH_A]: { status: 403, headers: { email: "p@x.io", address: ADDR_A, token: TOKEN_A } } }, rows: ROWS,
    requests: [{ cookie: AUTH, headers: JSON_CT }] },
  { name: "s3_200_no_email", s3: { [HASH_A]: { status: 200, headers: meta(undefined, ADDR_A, TOKEN_A) } }, requests: [{ cookie: AUTH, headers: JSON_CT }] },
  { name: "cookie_team_injection", rows: ROWS, requests: [{ cookie: `address=x; team=${OWN_A}`, headers: JSON_CT }] },

  // ── 팀 위임 (commerce) · 캐시 ──
  { name: "commerce_member", s3: S3_A, rows: [...ROWS, row("b-1", { to: TEAM_B })],
    commerce: () => ({ body: { session: { team: TEAM_B.toUpperCase().replace("0X", "0x") } } }),
    requests: [{ cookie: AUTH, query: `to=${TEAM_B}`, headers: JSON_CT }, { cookie: AUTH, query: `to=${TEAM_B}`, headers: JSON_CT }] },
  { name: "commerce_mismatch", s3: S3_A, commerce: () => ({ body: { session: { team: TEAM_C } } }),
    requests: [{ cookie: AUTH, query: `to=${TEAM_B}`, headers: JSON_CT }, { cookie: AUTH, query: `to=${TEAM_B}`, headers: JSON_CT }] },
  { name: "commerce_invalid_json", s3: S3_A, commerce: () => ({ body: "<html>down</html>", status: 502 }),
    requests: [{ cookie: AUTH, query: `to=${TEAM_B}`, headers: JSON_CT }, { method: "POST", cookie: AUTH, query: `to=${TEAM_B}`, body: ITEMS().slice(0, 2), headers: JSON_CT }] },
  { name: "commerce_own_and_bad_to", s3: S3_A, rows: ROWS,
    requests: [{ cookie: AUTH, query: `to=${OWN_A.toUpperCase().replace("0X", "0x")}`, headers: JSON_CT }, { cookie: AUTH, query: "to=0x1234", headers: JSON_CT }] },
  { name: "commerce_null_token_probe", s3: { [HASH_A]: { status: 200, headers: meta("a@example.com", ADDR_A, undefined) } },
    commerce: (url) => ({ body: { session: { team: url.searchParams.get("token") === "undefined" ? TEAM_B : TEAM_C } } }),
    requests: [{ cookie: `hash=${HASH_A}`, query: `to=${TEAM_B}`, headers: JSON_CT }] },

  // ── GET 변경분 ──
  { name: "get_delta_default", s3: S3_A, rows: ROWS, requests: [{ cookie: AUTH, headers: JSON_CT }] },
  { name: "get_delta_since_created", s3: S3_A, rows: ROWS,
    requests: [{ cookie: AUTH, query: `since=${T0 - 25000}&created_at=${T0 - 15000}`, headers: JSON_CT },
      { cookie: AUTH, query: "since=abc&created_at=0x10", headers: JSON_CT }, { cookie: AUTH, query: "since=&created_at=", headers: JSON_CT }] },
  { name: "get_ref_filter", s3: S3_A, rows: ROWS,
    requests: [{ cookie: AUTH, query: `ref=${hashId(OWN_A + CC + "#PO1")}`, headers: JSON_CT },
      { cookie: AUTH, query: `ref=${hashId(OWN_A + CC + "#PO1").toUpperCase().replace("0X", "0x")}`, headers: JSON_CT },
      { cookie: AUTH, query: `ref=${ethers.getAddress(hashId(OWN_A + CC + "#R1"))}`, headers: JSON_CT }] },
  { name: "get_doc_type_filter", s3: S3_A, rows: ROWS,
    requests: [{ cookie: AUTH, query: "doc_type=ci", headers: JSON_CT }, { cookie: AUTH, query: "doc_type=Shipping_Doc&limit=1", headers: JSON_CT },
      { cookie: AUTH, query: "doc_type=%2542L", headers: JSON_CT }, { cookie: AUTH, query: "doc_type=%25", headers: JSON_CT },
      { cookie: AUTH, query: `doc_type=BL&ref=${hashId(OWN_A + CC + "#R1")}`, headers: JSON_CT }] },
  { name: "get_limit_variants", s3: S3_A, rows: ROWS,
    requests: ["limit=2", "limit=abc", "limit=", "limit=0x2", "limit=-5", "limit=1e3", "limit=Infinity", "limit=%201.9"].map((q) => ({ cookie: AUTH, query: q, headers: JSON_CT })) },
  { name: "get_data_variants", s3: S3_A,
    rows: [row("t-text", { data: "abc" }), row("t-num", { data: 3 }), row("t-neg", { data: -1 }), row("t-null", { data: null }), row("t-empty", { data: Buffer.alloc(0) }),
      row("t-numstr", { data: "2" }), row("t-type-ctor", { type: "constructor" }), row("t-type-proto", { type: "__proto__" }), row("t-type-null", { type: null })],
    requests: [{ cookie: AUTH, headers: JSON_CT }] },
  { name: "get_query_proto", s3: S3_A, rows: ROWS, requests: [{ cookie: AUTH, query: "__proto__=x&constructor=y&since=0&since=1", headers: JSON_CT }] },

  // ── POST ──
  { name: "post_plain_array", s3: S3_A, rows: ROWS, requests: [{ method: "POST", cookie: AUTH, body: ITEMS(), headers: JSON_CT }, { cookie: AUTH, headers: JSON_CT }] },
  { name: "post_gzip_items", s3: S3_A, requests: [{ method: "POST", cookie: AUTH, body: { items: ITEMS() }, gzip: true }] },
  { name: "post_zlib_and_bom", s3: S3_A,
    requests: [{ method: "POST", cookie: AUTH, body: ITEMS().slice(0, 3), gzip: (b) => deflateSync(b) },
      { method: "POST", cookie: AUTH, body: Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from(JSON.stringify(ITEMS().slice(3, 5)))]) },
      { method: "POST", cookie: AUTH, body: ITEMS().slice(5, 7), gzip: (b) => Buffer.concat([gzipSync(b), Buffer.from("junk")]) },
      { method: "POST", cookie: AUTH, body: ITEMS().slice(7, 9), gzip: (b) => Buffer.concat([gzipSync(b.subarray(0, 1)), gzipSync(b.subarray(1)), Buffer.from([0, 9])]) }] },
  { name: "post_unchanged_skip", s3: S3_A,
    requests: [{ method: "POST", cookie: AUTH, body: ITEMS() }, { method: "POST", cookie: AUTH, body: ITEMS() },
      { method: "POST", cookie: AUTH, body: ITEMS().map((x) => (x && typeof x === "object" && x.id === "po-new-1" ? { ...x, updated_at: 9 } : x)) },
      { cookie: AUTH, headers: JSON_CT }] },
  { name: "post_existing_rows", s3: S3_A, rows: ROWS,
    requests: [{ method: "POST", cookie: AUTH, body: [{ id: "bl-1", type: "BL", digest: "d-bl-1", data: { x: 1 } }, { id: "other-team", type: "BL", data: { x: 2 } },
      { id: "constructor", type: "BL", data: {} }, { id: "__proto__", type: "BL", data: {} }, { id: "constructor", type: "BL", data: { again: 1 } }] },
      { method: "POST", cookie: AUTH, body: [{ id: "constructor", type: "BL", data: {} }] }] },
  { name: "post_over_200", s3: S3_A,
    requests: [{ method: "POST", cookie: AUTH, body: Array.from({ length: 205 }, (_, i) => ({ id: `bulk-${i}`, type: i % 2 ? "CI" : "BL", data: { doc_number: `D${i % 7}` } })) },
      { method: "POST", cookie: AUTH, body: Array.from({ length: 205 }, (_, i) => ({ id: `bulk-${i}`, type: i % 2 ? "CI" : "BL", data: { doc_number: `D${i % 7}` } })) }] },
  { name: "post_bad_bodies", s3: S3_A,
    requests: [{ method: "POST", cookie: AUTH, body: "not json" }, { method: "POST", cookie: AUTH }, { method: "POST", cookie: AUTH, body: { items: "x" } },
      { method: "POST", cookie: AUTH, body: [] }, { method: "POST", cookie: AUTH, body: "5" }, { method: "POST", cookie: AUTH, body: [1, 2] }] },
  { name: "post_infinity_created_at", s3: S3_A, requests: [{ method: "POST", cookie: AUTH, body: [{ id: "inf-1", type: "BL", data: {}, created_at: "Infinity" }, { id: "ok-1", type: "BL", data: {} }] }] },
  // D1 배치 실패: 트리거로 한 문장을 막는다 (배치는 원자적이라 모두 안 들어감). id 를 그대로 두는 fix-trading-id-case 뒤에도
  // 같은 문장이 막히게 lower() 로
  { name: "post_batch_error", s3: S3_A, setup: ["CREATE TRIGGER no_boom BEFORE INSERT ON items WHEN lower(NEW.id) = 'boom' BEGIN SELECT RAISE(ABORT, 'boom blocked'); END"],
    requests: [{ method: "POST", cookie: AUTH, body: [{ id: "ok-1", type: "BL", data: {} }, { id: "BOOM", type: "BL", data: {} }, { id: "x", type: "memo" }] }] },
  { name: "delete_batch_error", s3: S3_A, rows: ROWS, setup: ["CREATE TRIGGER keep_bl BEFORE DELETE ON items WHEN OLD.id = 'bl-1' BEGIN SELECT RAISE(ABORT, 'keep bl-1'); END"],
    requests: [{ method: "DELETE", cookie: AUTH, body: ["ci-1", "bl-1"] }, { cookie: AUTH, headers: JSON_CT }] },

  // ── DELETE ──
  { name: "delete_variants", s3: S3_A, rows: ROWS,
    requests: [{ method: "DELETE", cookie: AUTH, query: "id=%20BL-1%20" },
      { method: "DELETE", cookie: AUTH, query: `ids=${encodeURIComponent(JSON.stringify(["CI-1", "po-1", "ci-1", 0, null, "other-team"]))}` },
      { method: "DELETE", cookie: AUTH, body: { ids: ["old-1", "future-1"] } },
      { method: "DELETE", cookie: AUTH, body: ["nomode-1"], gzip: true },
      { method: "DELETE", cookie: AUTH, query: "ids=%5Bbad", body: "garbage" },
      { method: "DELETE", cookie: AUTH },
      { method: "DELETE", cookie: AUTH, query: "id=%25E0%25A4%25A" },
      { cookie: AUTH, headers: JSON_CT }] },
  { name: "delete_over_200", s3: S3_A, rows: ROWS,
    requests: [{ method: "DELETE", cookie: AUTH, body: Array.from({ length: 230 }, (_, i) => (i === 210 ? "sd-1" : `x-${i}`)) }, { cookie: AUTH, headers: JSON_CT }] },

  // ══════════ 버그 수정 (docs/workers-rs.md "고친 버그", fixes/trading.mjs) ══════════

  // ── 앱이 보내는 꼴 그대로 (모두 받아들여야 한다) ──
  // 본인 팀 · 멤버 팀(commerce 확인) · 자리표 팀(hashId(0x0…0), commerce 가 본인 팀이라 답함)으로 push → pull
  { name: "app_own_team_cycle", s3: S3_A, rows: ROWS,
    requests: [appPull(OWN_A, 0),
      appPush(OWN_A, [{ id: "app-bl-1", type: "BL", created_at: T0 - 3000, updated_at: T0 - 2000, digest: "dg1", data: { header: { doc_number: "BL-9", reference_po: "PO-9" }, text: "t" } },
        { id: "app-ci-1", type: "CI", created_at: T0 - 3000, updated_at: 0, data: { doc_number: "CI-9", items: [{ reference_bl: "BL-9" }] } }]),
      appPull(OWN_A, lastCursor),
      appPush(OWN_A, [{ id: "app-bl-1", type: "BL", created_at: T0 - 3000, updated_at: T0 - 2000, digest: "dg1", data: { header: { doc_number: "BL-9", reference_po: "PO-9" }, text: "t" } }])] },
  { name: "app_member_team_cycle", s3: S3_A, rows: [row("b-1", { to: TEAM_B })],
    commerce: () => ({ body: { session: { team: TEAM_B } } }),
    requests: [appPull(TEAM_B, 0), appPush(TEAM_B, [{ id: "app-m-1", type: "PL", created_at: T0 - 1000, data: { containers: [{ container_number: "MSCU1" }] } }]), appPull(TEAM_B, lastCursor)] },
  { name: "app_placeholder_team", s3: S3_A, rows: ROWS,
    commerce: () => ({ body: { session: { team: OWN_A } } }),
    requests: [appPull(ZERO_TEAM, 0), appPush(ZERO_TEAM, [{ id: "app-z-1", type: "BL", data: {} }]), appPull(ZERO_TEAM, 0)] },

  // fix-trading-delta-cursor: 기기 B 가 당겨 간 뒤 기기 A 가 하루 전에 만든 서류를 처음 올린다 → B 의 다음 pull 이 받아야 한다
  // (원본: max(updated_at, created_at) 이 B 의 cursor 보다 옛 시각이라 영영 빠짐)
  { name: "delta_late_upload", s3: S3_A,
    requests: [appPull(OWN_A, 0),
      appPush(OWN_A, [{ id: "late-1", type: "BL", created_at: T0 - 86400000, updated_at: T0 - 86400000, data: { doc_number: "L1" } },
        { id: "late-2", type: "CI", created_at: T0 - 86400000, updated_at: 0, data: { doc_number: "L2" } }]),
      appPull(OWN_A, lastCursor)] },
  // fix-trading-delta-cursor: LIMIT 에 잘린 쪽 — 같은 synced_at 은 나누지 않고, 커서는 그 시각에서 이어진다 (원본: now 로 넘겨 남은 행을 잃음)
  { name: "delta_paging", s3: S3_A,
    rows: [row("p1", { created_at: T0 - 500000, synced_at: T0 - 500000 }), row("p2", { created_at: T0 - 100000, synced_at: T0 - 400000 }),
      row("p3", { created_at: T0 - 200000, synced_at: T0 - 400000 }), row("p4", { created_at: T0 - 300000, synced_at: T0 - 400000 }),
      row("p5", { created_at: T0 - 600000, synced_at: T0 - 300000 }), row("p6", { created_at: T0 + 500000, synced_at: T0 - 200000 })],
    requests: [{ cookie: AUTH, query: "since=0&limit=2", headers: JSON_CT },
      { cookie: AUTH, query: (prev) => `since=${lastCursor(prev)}&limit=2`, headers: JSON_CT },
      { cookie: AUTH, query: (prev) => `since=${lastCursor(prev)}&limit=2`, headers: JSON_CT }] },

  // fix-trading-cookie-identity: S3 세션 없이 address · team 쿠키만으로 쓰기 · 지우기 (원본: 남의 팀 OWN_A 에 쓰고 지움)
  { name: "cookie_identity_write", rows: ROWS,
    requests: [{ method: "POST", cookie: `address=0xattacker; team=${OWN_A}`, body: [{ id: "injected", type: "BL", data: {} }] },
      { method: "DELETE", cookie: `address=0xattacker; team=${OWN_A}; email=e`, query: "id=bl-1" },
      { cookie: AUTH + `; team=${TEAM_C}; role=owner`, headers: JSON_CT }] },

  // fix-trading-upsert-scope: 다른 팀 행의 id 로 올리기 (원본: TEAM_C 의 행을 덮어쓰고 "to" 를 OWN_A 로 가져감)
  { name: "post_takeover", s3: S3_A, rows: [row("victim", { to: TEAM_C, data: { secret: 1 } }), row("mine", {})],
    requests: [appPush(OWN_A, [{ id: "victim", type: "BL", data: { x: 1 } }, { id: "mine", type: "BL", data: { y: 2 } }]), appPull(OWN_A, 0)] },

  // fix-trading-html-escape: S3 가 200 을 주는 hash 에 태그 (token 은 틀림 → bootstrap 없이 그대로 반사)
  { name: "html_xss", s3: { [HASH_A]: S3_A[HASH_A], "<img src=x onerror=alert(1)>": { status: 200, headers: meta("x@x.io", ADDR_A, "other") } },
    requests: [{ query: `hash=${encodeURIComponent("<img src=x onerror=alert(1)>")}&token=nope` }, { cookie: AUTH }] },

  // fix-trading-setcookie-verified: 확인하지 않은 hash 로 쿠키 속성 끼워 넣기
  { name: "setcookie_injection", s3: { [`${HASH_A}; Domain=logis.center; Path=/`]: { status: 200, headers: meta("x@x.io", ADDR_A, "other") } },
    requests: [{ query: `hash=${encodeURIComponent(`${HASH_A}; Domain=logis.center; Path=/`)}&token=nope`, headers: JSON_CT }] },

  // fix-trading-cors · fix-trading-options-first
  { name: "cors_origins", s3: S3_A,
    requests: [{ method: "OPTIONS", headers: { origin: "https://commerce.logis.center", "access-control-request-method": "POST" } },
      { method: "OPTIONS", headers: { origin: "https://evil.example" } },
      { method: "OPTIONS" },
      { cookie: AUTH, headers: { ...JSON_CT, origin: "https://evil.example" } },
      { cookie: AUTH, headers: { ...JSON_CT, origin: "https://logis.center" } },
      { cookie: AUTH, headers: { ...JSON_CT, origin: "null" } }] },

  // fix-trading-id-case: 대소문자 · 앞뒤 공백이 든 id 는 앱이 쓰는 그대로 (원본: 소문자로 바꿔 내려줘 앱에 두 벌)
  { name: "id_case_roundtrip", s3: S3_A,
    requests: [appPush(OWN_A, [{ id: "Doc-ABC", type: "BL", data: {} }, { id: "doc-abc", type: "BL", data: { other: 1 } }, { id: " pad ", type: "CI", data: {} }, { id: "   ", type: "CI", data: {} }]),
      appPull(OWN_A, 0),
      { method: "DELETE", cookie: AUTH, query: "id=Doc-ABC" },
      { method: "DELETE", cookie: AUTH, body: [" pad "] },
      appPull(OWN_A, 0)] },

  // fix-trading-double-decode: '%' 가 든 id · 서식
  { name: "double_decode", s3: S3_A, rows: [row("50%off", {}), row("a%41", {})],
    requests: [{ method: "DELETE", cookie: AUTH, query: `id=${encodeURIComponent("50%off")}` },
      { method: "DELETE", cookie: AUTH, query: `ids=${encodeURIComponent(JSON.stringify(["a%41"]))}` },
      { cookie: AUTH, query: `doc_type=${encodeURIComponent("B%4C")}`, headers: JSON_CT },
      { cookie: AUTH, headers: JSON_CT }] },

  // fix-trading-relay-order: 정수 꼴 키 — 앱(정렬: "10" < "2")과 원본(for-in: 2 먼저)이 다른 허브를 골랐다
  { name: "relay_order", s3: S3_A,
    requests: [appPush(OWN_A, [{ id: "ro-1", type: "CI", data: { "2": { reference_bl: "TWO" }, "10": { reference_bl: "TEN" } } },
      { id: "ro-2", type: "PL", data: { "가": { container_number: "KOR1" }, z: [{ container_number: "Z1" }] } }])] },

  // fix-trading-team-mismatch: 팀을 옮김 (B 확인 → 캐시 B · 이제 C). 원본은 캐시가 다르면 5분 동안 본인 팀으로 강등
  { name: "team_moved_cache", s3: S3_A, rows: [row("c-1", { to: TEAM_C }), row("b-1", { to: TEAM_B })],
    commerce: (url, n) => ({ body: { session: { team: n === 0 ? TEAM_B : TEAM_C } } }),
    requests: [appPull(TEAM_B, 0), appPull(TEAM_C, 0), appPush(TEAM_C, [{ id: "c-2", type: "BL", data: {} }])] },
  // 자리표 팀인데 실제 팀은 C → 원본은 본인 팀에 써 C 의 서류가 개인 팀으로 갈라짐
  { name: "team_placeholder_other", s3: S3_A, rows: ROWS, commerce: () => ({ body: { session: { team: TEAM_C } } }),
    requests: [appPush(ZERO_TEAM, [{ id: "z-1", type: "BL", data: {} }]), appPull(ZERO_TEAM, 0)] },

  // fix-trading-nan-params: 앱 꼴의 pull 에 since 가 비었을 때 · Infinity 시각
  { name: "nan_params", s3: S3_A, rows: ROWS,
    requests: [{ cookie: AUTH, query: "since=Infinity&limit=Infinity", headers: JSON_CT },
      { method: "POST", cookie: AUTH, body: [{ id: "inf-u", type: "BL", data: {}, created_at: "-Infinity", updated_at: "Infinity" }] },
      { cookie: AUTH, query: "since=0", headers: JSON_CT }] },

  // fix-trading-preload-chunk: 앱 꼴 200건을 두 번 (두 번째는 모두 SKIP 이어야 한다)
  { name: "app_push_200_twice", s3: S3_A,
    requests: [appPush(OWN_A, Array.from({ length: 200 }, (_, i) => ({ id: `k-${i}`, type: "BL", created_at: T0 - 9000, updated_at: T0 - 9000, digest: `dg-${i}`, data: { doc_number: `K${i}` } }))),
      appPush(OWN_A, Array.from({ length: 200 }, (_, i) => ({ id: `k-${i}`, type: "BL", created_at: T0 - 9000, updated_at: T0 - 9000, digest: `dg-${i}`, data: { doc_number: `K${i}` } })))] },
];

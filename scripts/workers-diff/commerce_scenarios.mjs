// commerce 차분 시험의 시나리오: D1 행 · S3 객체(메타) · 차례로 보낼 요청들
import { ethers } from "ethers";
export const AWS = { aws_access_key_id: "AKIDTESTKEY", aws_secret_access_key: "testsecret", aws_region: "ap-northeast-2", aws_bucket: "logis-test" };

const T0 = 1760000000000;
const USER = "0x" + "a1".repeat(20);
const TEAM = "0x" + "b2".repeat(20);
const OTHER = "0x" + "c3".repeat(20);
const HASH = "d4".repeat(20);
const TOKEN = "e5".repeat(32);
const ORIGIN = "https://shop.example.com";
const HREF = "https://shop.example.com/goods/view?no=12";
const q = (o) => "/?" + Object.entries(o).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&");

// 로그인된 세션(S3 hash 객체의 메타)
// address/<ethers.computeAddress(hashMessage(email))> 는 잔액 — 없으면 POST 가 아무 응답 없이 418 로 떨어진다(원본 동작)
const WALLET = "0x218c3b402e1627e6578e394d8f7bd9eeaf161651";
const S3_NO_BALANCE = { [`hash/${HASH}`]: { email: "kim@example.com", token: TOKEN, address: USER, phone: "" } };
const S3_SESSION = { ...S3_NO_BALANCE, [`address/${WALLET}`]: { balance: "1000" } };
const COOKIES = { hash: HASH, token: TOKEN };

const ZONES = [
  { db: "commerce_logis_center", table: "zones", values: { id: "apac1", pool: 3 } },
  { db: "commerce_logis_center", table: "zones", values: { id: "enam1", pool: 0 } },
];
const TEAM_BASE = () => ({
  pages: {}, goods: { draft: 0, count: 2 }, order: { draft: 0, count: 1 }, event: { draft: 0, count: 0, coupon: 0 },
  tracking: { draft: 0, count: 0 }, sales: { draft: 0, count: 0 },
});
// 이미 있는 사용자 · 팀 (사용자 data.sender 가 있어야 POST · PUT 의 cookies.sender 가 선다)
const MEMBERS = (over = {}) => [
  { db: "commerce_logis_apac", table: "users", values: { id: USER, type: "user", from: USER, to: TEAM, cc: "", bcc: "", ref: "",
    data: { flag: "kr", name: "kim", title: "", region: "apac1", page_count: 0, favicon: null, team: "Kim Shop", sender: "서울특별시 강남구 1", ...over },
    created_at: T0 - 900000, updated_at: T0 - 900000 } },
  { db: "commerce_logis_apac", table: "users", values: { id: TEAM, type: "team", from: USER, to: TEAM, cc: "", bcc: "", ref: "",
    data: { name: "Kim Shop", base: TEAM_BASE(), address: ["서울특별시 강남구 1"] }, created_at: T0 - 900000, updated_at: T0 - 900000 } },
];
const PAGE = { db: "commerce_logis_apac", table: "pages", values: { id: "0x" + "f6".repeat(20), type: "goods", from: USER, to: TEAM, cc: "", bcc: "", ref: "",
  data: { type: "goods", origin: ORIGIN, link: "/goods/view?no=12" }, created_at: T0 - 600000, updated_at: T0 - 600000 } };
const ITEM = (id, over = {}) => ({ db: "commerce_logis_apac1_items", table: "items", values: { id, type: "goods", from: USER, to: TEAM, cc: "", bcc: "", ref: "",
  data: { title: "무거운 상자", price: 12000 }, created_at: T0 - 500000, updated_at: T0 - 400000, ...over } });
const TASK = (id, ref) => ({ db: "commerce_logis_apac1_items", table: "tasks", values: { id, type: "tracking", ref, data: { id, type: "tracking" }, created_at: T0 - 60000, updated_at: 0 } });

const JSON_GET = (over = {}) => ({ method: "GET", cookies: COOKIES, headers: { "Content-Type": "application/json" },
  path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000, ...over }) });

export const SCENARIOS = [
  // 익명: 새 지갑(무작위 hash · token) → S3 PUT, 셸 HTML
  { name: "anon_html", requests: [{ method: "GET", path: "/" }] },
  { name: "anon_options", requests: [{ method: "OPTIONS", path: "/" }] },
  // 익명 JSON: cookies.href 가 없어 new URL(undefined) 가 던진다 → 418
  { name: "anon_json", requests: [{ method: "GET", headers: { "Content-Type": "application/json" }, path: "/" }] },
  // 다른 나라(enam), Accept-Language · X-Real-IP
  { name: "anon_us", requests: [{ method: "GET", path: "/", cf: { country: "US", city: "Ashburn" }, headers: { "Accept-Language": "en-US,en;q=0.9", "X-Real-IP": "203.0.113.9" } }] },
  // 로그인 직후 첫 방문: 사용자 행이 없어 사용자 · 팀을 새로 만들고 zones.pool 을 올린다. 뒤 요청은 받은 쿠키로
  { name: "login_new_user", rows: ZONES, s3: S3_SESSION, requests: [
    { method: "GET", path: q({ hash: HASH, token: TOKEN }) },
    { method: "GET", jar: true, headers: { "Content-Type": "application/json" }, path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000 }) },
  ] },
  // 토큰이 틀림 → 로그인 안 됨
  { name: "login_bad_token", rows: ZONES, s3: S3_SESSION, requests: [{ method: "GET", path: q({ hash: HASH, token: "nope" }) }] },
  // 이미 있는 사용자: JSON 동기화
  { name: "member_json", rows: [...ZONES, ...MEMBERS(), PAGE, ITEM("0x" + "11".repeat(20)), ITEM("0x" + "12".repeat(20), { type: "order" })], s3: S3_SESSION,
    requests: [JSON_GET(), JSON_GET({ from: USER, to: TEAM, day: "3" })] },
  { name: "member_html", rows: [...ZONES, ...MEMBERS()], s3: S3_SESSION, requests: [{ method: "GET", cookies: COOKIES, path: q({ origin: ORIGIN, href: HREF }) }] },
  // 이름 · 팀 · 보내는 곳 바꾸기 (질의 값이 사용자 data 를 덮는다)
  { name: "member_rename", rows: [...ZONES, ...MEMBERS()], s3: S3_SESSION,
    requests: [{ method: "GET", cookies: COOKIES, path: q({ name: "lee", team: "Lee Shop", sender: "부산광역시 1" }) }] },
  // POST: 화면 글 → tasks 대기열
  { name: "post_text", rows: [...ZONES, ...MEMBERS()], s3: S3_SESSION, requests: [
    { method: "POST", cookies: COOKIES, headers: { "Content-Type": "text/html" }, gzip: encodeURIComponent("<html><body><h1>무거운 상자</h1><p>12,000원</p></body></html>"),
      path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000, from: USER, to: TEAM }) },
  ] },
  { name: "post_no_balance", rows: [...ZONES, ...MEMBERS()], s3: S3_NO_BALANCE, requests: [
    { method: "POST", cookies: COOKIES, headers: { "Content-Type": "text/html" }, gzip: "hello",
      path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000, from: USER, to: TEAM }) },
  ] },
  // 같은 글을 두 번: 두 번째는 tasks 에 이미 있어 그 행을 돌려준다. error 질의는 ON CONFLICT DO UPDATE
  { name: "post_repeat_error", rows: [...ZONES, ...MEMBERS()], s3: S3_SESSION, requests: [
    { method: "POST", cookies: COOKIES, headers: { "Content-Type": "text/plain" }, gzip: "same text", path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000, from: USER, to: TEAM, title: "제목" }) },
    { method: "POST", cookies: COOKIES, headers: { "Content-Type": "text/plain" }, gzip: "same text", path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000, from: USER, to: TEAM }) },
    { method: "POST", cookies: COOKIES, headers: { "Content-Type": "text/plain" }, gzip: "other text", path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000, from: USER, to: TEAM, error: "1" }) },
  ] },
  // 이미지 스캔(format=image/…) · 유료 사용자(이번 달 invoice 행)
  { name: "post_image_paid", rows: [...ZONES, ...MEMBERS(),
      { db: "commerce_logis_apac1_items", table: "items", values: { id: "inv1", type: "invoice", from: USER, to: "0x" + "9".repeat(40), cc: "", bcc: "", ref: "", data: { amount: 1 }, created_at: T0 - 1000, updated_at: 0 } }],
    s3: S3_SESSION, requests: [
    { method: "POST", cookies: COOKIES, headers: { "Content-Type": "image/png" }, gzip: "iVBORw0KGgo=", path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000, from: USER, to: TEAM, format: "image/png" }) },
  ] },
  // 클라우드 검색: { query, vector } — 데스크톱 앱이 보내는 꼴 (from=주소 · to=팀). 페이지가 없으면 안내 응답
  { name: "post_query_vector", rows: [...ZONES, ...MEMBERS(), PAGE], s3: S3_SESSION, requests: [
    { method: "POST", cookies: COOKIES, headers: { "Content-Type": "application/json" }, gzip: { query: "무거운 상자", vector: [0.1, 0.2] },
      path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000, from: USER, to: TEAM }) },
    { method: "POST", cookies: COOKIES, headers: { "Content-Type": "application/json" }, gzip: { query: "무거운 상자", vector: [0.1, 0.2] },
      path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000, from: TEAM, to: USER }) },
  ] },
  // 주소 배열(URL 목록) · 프롬프트 글
  { name: "post_url_list", rows: [...ZONES, ...MEMBERS()], s3: S3_SESSION, requests: [
    { method: "POST", cookies: COOKIES, headers: { "Content-Type": "application/json" }, gzip: ["https://shop.example.com/goods/view?no=1", "/goods/view?no=2", 7],
      path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000, from: TEAM, to: USER }) },
  ] },
  { name: "post_no_created_at", rows: [...ZONES, ...MEMBERS()], s3: S3_SESSION, requests: [
    { method: "POST", cookies: COOKIES, headers: { "Content-Type": "text/html" }, gzip: "hello", path: q({ origin: ORIGIN, href: HREF, from: USER, to: TEAM }) },
  ] },
  // PUT: 벡터 올리기 (Vectorize upsert)
  { name: "put_vector", rows: [...ZONES, ...MEMBERS()], s3: S3_SESSION, requests: [
    { method: "PUT", cookies: COOKIES, path: q({ type: "vector", origin: ORIGIN, href: HREF }),
      gzip: { items: [{ id: "v1", values: [0.1, 0.2, 0.3], table: "items", no: "12", type: "goods" }, { id: "v2", values: [] }, { values: [1] }, { id: "v3", values: [0.5, 0.5], table: "sales" }] } },
  ] },
  // PUT: 대화 글
  { name: "put_talk", rows: [...ZONES, ...MEMBERS()], s3: S3_SESSION, requests: [
    { method: "PUT", cookies: COOKIES, path: q({ origin: ORIGIN, href: HREF, from: USER, to: TEAM, text: "배송 언제 되나요?" }) },
  ] },
  // PUT: 멤버 초대(권한 주기) · 내보내기
  { name: "put_member_grant", rows: [...ZONES, ...MEMBERS(), ITEM("0x" + "11".repeat(20))], s3: S3_SESSION, requests: [
    { method: "PUT", cookies: COOKIES, path: q({ origin: ORIGIN, href: HREF, from: USER, to: OTHER, id: TEAM, cc: USER, ref: JSON.stringify(["0x" + "11".repeat(20)]) }) },
  ] },
  { name: "put_member_revoke", rows: [...ZONES, ...MEMBERS(), ITEM("0x" + "11".repeat(20), { to: OTHER, bcc: TEAM })], s3: S3_SESSION, requests: [
    { method: "PUT", cookies: COOKIES, path: q({ origin: ORIGIN, href: HREF, from: USER, to: OTHER, id: TEAM, cc: "0x" + "0".repeat(40), ref: JSON.stringify(["0x" + "11".repeat(20)]) }) },
  ] },
  // DELETE: 작업 지우기 · 로그아웃
  { name: "delete_crons", rows: [...ZONES, ...MEMBERS(), TASK("t1", "0x" + "77".repeat(20)), TASK("t2", "0x" + "78".repeat(20))], s3: S3_SESSION, requests: [
    { method: "DELETE", cookies: COOKIES, path: q({ type: "crons", ref: "0x" + "77".repeat(20) }) },
  ] },
  // 보안: hash · token 없이 남의 쿠키(address · team · sender · balance · cc)만 실어 보낸다 — 원본은 S3 검증 없이 그 값을 믿는다
  { name: "forged_cookie_delete", rows: [...ZONES, ...MEMBERS(), TASK("t1", "0x" + "77".repeat(20))], requests: [
    { method: "DELETE", cookies: { address: USER, team: TEAM }, path: q({ type: "crons", ref: "0x" + "77".repeat(20) }) },
  ] },
  { name: "forged_cookie_post", rows: [...ZONES, ...MEMBERS()], requests: [
    { method: "POST", cookies: { address: USER, team: TEAM, sender: "x", balance: "1000" }, headers: { "Content-Type": "text/plain" }, gzip: "forged",
      path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000, from: USER, to: TEAM }) },
  ] },
  { name: "forged_cookie_read", rows: [...ZONES, ...MEMBERS(), ITEM("0x" + "11".repeat(20))], requests: [
    { method: "GET", cookies: { address: USER, team: TEAM }, headers: { "Content-Type": "application/json" }, path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000 }) },
  ] },
  // 셸 HTML 은 cookies 를 그린다: team 쿠키만으로 남의 팀 정보(pages · 개수)가 실리는지
  { name: "forged_cookie_html", rows: [...ZONES, ...MEMBERS()], requests: [
    { method: "GET", cookies: { team: TEAM }, path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000 }) },
    { method: "GET", cookies: { address: USER, team: TEAM }, path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000 }) },
  ] },
  // 보안: 쿠키 cc 가 SQL 글에 그대로 붙는다
  { name: "forged_cookie_sql", rows: [...ZONES, ...MEMBERS(), ITEM("0x" + "11".repeat(20))], requests: [
    { method: "GET", cookies: { address: USER, team: TEAM, cc: "x' OR '1'='1" }, headers: { "Content-Type": "application/json" }, path: q({ href: HREF, created_at: T0 + 1000 }) },
  ] },
  { name: "delete_logout", rows: [...ZONES, ...MEMBERS()], s3: S3_SESSION, requests: [
    { method: "DELETE", cookies: COOKIES, path: "/" },
    { method: "GET", jar: true, path: "/" },
  ] },
];

// ---------------------------------------------------------------------------------------------------------------------
// 고친 버그(fix-*)마다의 시나리오 — 고친 원본 · Rust 는 SAME, ORIGINAL=raw(또는 COMMERCE_SKIP=<id>)면 그 수정이 닿는 곳에서 DIFF
const hashId = (text) => ethers.computeAddress(ethers.hashMessage(text)).toLowerCase();
const ZERO = "0x" + "0".repeat(40);
const CC = hashId("example.com"); // shop.example.com 의 cc
const HOME_ORIGIN = "https://commerce.logis.center";
const HOME_HREF = "https://commerce.logis.center/tracking";
const HASH2 = "a7".repeat(20);
const TOKEN2 = "b8".repeat(32);
const NEWADDR = "0x" + "e9".repeat(20);
const HASH3 = "c4".repeat(20);
const TOKEN3 = "d5".repeat(32);
const PAGE1 = hashId(CC + "/goods/view"); // HREF 의 pageId
const PAGE2 = "0x" + "5a".repeat(20);
const PAGE_ROW = (id, type, over = {}) => ({ db: "commerce_logis_apac1_items", table: "items", values: { id, type: "pages", from: USER, to: TEAM, cc: CC,
  bcc: hashId(type + CC), ref: "", data: { type, origin: ORIGIN, link: "/" + type + "/view" }, created_at: T0 - 600000, updated_at: T0 - 600000, ...over } });
const OTHER_USER = { db: "commerce_logis_apac", table: "users", values: { id: OTHER, type: "user", from: OTHER, to: TEAM, cc: "", bcc: "", ref: "",
  data: { flag: "kr", name: "lee", title: "", region: "apac1", page_count: 0, favicon: null, team: "Kim Shop", sender: "부산광역시 1" }, created_at: T0 - 800000, updated_at: T0 - 800000 } };
const S3_OTHER = { [`hash/${HASH3}`]: { email: "lee@example.com", token: TOKEN3, address: OTHER, phone: "" } };
const S3_INVITEE = { [`hash/${HASH2}`]: { email: "new@example.com", token: TOKEN2, address: NEWADDR, phone: "" } };
const LOGIS_TASK = (id, ref, from, over = {}) => ({ db: "commerce_logis_apac", table: "tasks", values: { id, type: "tracking", from, to: TEAM, cc: "", bcc: "", ref,
  data: { id, type: "tracking" }, created_at: T0 - 60000, updated_at: 0, ...over } });
const PUT = (over, cookies = COOKIES) => ({ method: "PUT", cookies, path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000, ...over }) });
// 앱(proxy_fetch)은 질의에 hash · token 을 한 번 더 붙인다
const APP_SESSION = `&hash=${HASH}&token=${TOKEN}`;
const APP_EXTRACT = (envelope, extra = {}) => ({ method: "POST", headers: { "Content-Type": "application/json" },
  gzip: { id: "task_1", from: USER, to: TEAM, cc: "", bcc: "", ref: "", link: "local", ...envelope },
  path: q({ from: USER, to: TEAM, ...extra, hash: HASH, token: TOKEN, href: "https://console.logis.center/" }) });

SCENARIOS.push(
  // fix-session-token: S3 hash 객체에 token 이 없으면 hash 쿠키만으로 로그인됐다
  { name: "session_no_token", rows: [...ZONES, ...MEMBERS()], s3: { [`hash/${HASH}`]: { email: "kim@example.com", address: USER } },
    requests: [{ method: "GET", cookies: { hash: HASH }, headers: { "Content-Type": "application/json" }, path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000 }) }] },
  // fix-zone-pool: 자리 남은 zone 이 없으면 사용자를 만들지 않는다 (원본은 region '' 사용자)
  { name: "login_zone_full", rows: [{ db: "commerce_logis_center", table: "zones", values: { id: "apac1", pool: 1000 } }], s3: S3_SESSION,
    requests: [{ method: "GET", path: q({ hash: HASH, token: TOKEN }) }] },
  // fix-region-fallback: 묶이지 않은 지역(US)의 로그인 — APAC 로 내려간다
  { name: "login_us", rows: ZONES, s3: S3_SESSION, requests: [{ method: "GET", cf: { country: "US", city: "Ashburn" }, path: q({ hash: HASH, token: TOKEN }) }] },
  // fix-coupon-count · fix-draft-count: base.coupon.count, 초안 ref 마다 센 수
  { name: "member_coupon_drafts", rows: [...ZONES, ...MEMBERS().map((r) => (r.values.type === "team"
      ? { ...r, values: { ...r.values, data: { ...r.values.data, base: { ...TEAM_BASE(), coupon: { draft: 0, count: 4 } } } } } : r)),
    ITEM("0x" + "21".repeat(20), { updated_at: 0, ref: "r1" }), ITEM("0x" + "22".repeat(20), { updated_at: 0, ref: "r1" }), ITEM("0x" + "23".repeat(20), { updated_at: 0, ref: "r2" })],
    s3: S3_SESSION, requests: [JSON_GET()] },
  // fix-tracking-index: 주문 상세(추적 번호 없음)의 tracking 행 — 원본은 앞에서 남은 item.index 로 "order" = undefined (418)
  { name: "detail_order_tracking", rows: [...ZONES, ...MEMBERS(),
      PAGE_ROW("0x" + "5b".repeat(20), "order", { bcc: hashId("order" + CC.toUpperCase()), data: { type: "order", origin: ORIGIN, link: "/order/view", detail: true } }),
      ITEM("0x" + "31".repeat(20), { type: "order", cc: CC, bcc: hashId("order" + CC.toUpperCase()), ref: hashId(TEAM + CC + "/order/view?no=5"), digest: "dg1" }),
      { db: "commerce_logis_apac1_sales", table: "sales", values: { id: "0x" + "31".repeat(20), type: "order", from: USER, to: TEAM, cc: CC, bcc: "", ref: "", data: { n: 1 },
        created_at: T0 - 300000, index: 7, tracking: 0 } },
      { db: "commerce_logis_apac1_tracking", table: "tracking", values: { id: "0x" + "41".repeat(20), type: "tracking", from: USER, to: TEAM, cc: CC, bcc: "", ref: "",
        data: { no: "123" }, created_at: T0 - 200000, order: 7 } }],
    s3: S3_SESSION, requests: [JSON_GET({ href: "https://shop.example.com/order/view?no=5" })] },
  // fix-crons-query: 글 · 겹친 id 가 섞인 crons — 원본은 crons 의 자리로 지워 엉뚱한 작업을 뺐다
  { name: "member_crons", rows: [...ZONES, ...MEMBERS(), LOGIS_TASK("0x" + "72".repeat(20), "r", USER)], s3: S3_SESSION,
    requests: [JSON_GET({ crons: JSON.stringify(["0x" + "71".repeat(20), "x", "0x" + "72".repeat(20), "0x" + "71".repeat(20)]) }),
      JSON_GET({ crons: JSON.stringify("0x" + "71".repeat(20)) })] },
  // fix-delete-tasks-db: 지역 DB 의 자기 작업만 지운다
  { name: "delete_own_tasks", rows: [...ZONES, ...MEMBERS(), LOGIS_TASK("t1", "0x" + "77".repeat(20), USER), LOGIS_TASK("t2", "0x" + "77".repeat(20), OTHER),
      LOGIS_TASK("t3", "0x" + "78".repeat(20), USER)], s3: S3_SESSION,
    requests: [{ method: "DELETE", cookies: COOKIES, path: q({ type: "crons", ref: "0x" + "77".repeat(20) }) }] },
  // fix-task-column: ?error= 로 같은 글을 다시 넣는다 (ON CONFLICT DO UPDATE)
  { name: "post_error_retry", rows: [...ZONES, ...MEMBERS()], s3: S3_SESSION, requests: [
    { method: "POST", cookies: COOKIES, headers: { "Content-Type": "text/plain" }, gzip: "same text", path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000, from: USER, to: TEAM }) },
    { method: "POST", cookies: COOKIES, headers: { "Content-Type": "text/plain" }, gzip: "same text", path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000, from: USER, to: TEAM, error: "1" }) },
  ] },
  // fix-app-cloud-search: 데스크톱 앱의 클라우드 검색 그대로 (세션은 질의, href 는 logis.center 홈)
  { name: "app_cloud_search", rows: [...ZONES, ...MEMBERS()], s3: S3_SESSION, requests: [
    { method: "POST", headers: { "Content-Type": "application/json" }, gzip: { query: " 무거운 상자 ", vector: [0.1, 0.2, 0.3] },
      path: q({ origin: HOME_ORIGIN, created_at: T0 + 1000, hash: HASH, token: TOKEN, href: HOME_HREF, from: USER, to: TEAM }) + APP_SESSION },
  ] },
  // fix-app-extraction: 앱의 추출 봉투 (created_at · origin 없음, href 는 proxy_fetch 의 기본값)
  { name: "app_extract_image", rows: [...ZONES, ...MEMBERS()], s3: S3_SESSION, requests: [
    APP_EXTRACT({ type: "image_extraction", body: "data:application/octet-stream;base64,iVBORw0KGgo=" }, { format: encodeURIComponent("image/png") }),
  ] },
  { name: "app_extract_html", rows: [...ZONES, ...MEMBERS()], s3: S3_SESSION, requests: [
    APP_EXTRACT({ type: "html_extraction", link: HREF, body: "<html><body><h1>무거운 상자</h1><p>12,000원</p></body></html>" }),
  ] },
  { name: "app_extract_document", rows: [...ZONES, ...MEMBERS()], s3: S3_SESSION, requests: [
    APP_EXTRACT({ type: "document_extraction", body: "data:application/pdf;base64,JVBERi0=" }),
  ] },
  // fix-app-relay: 데스크톱이 offer 를 올리고(세션) 모바일이 읽는다(10분), 틀린 token · 오래된 것 · 없는 것 · JSON 아님
  { name: "relay_roundtrip", s3: { ...S3_SESSION, [`relay/${HASH2}`]: { created_at: String(T0 - 700000), __body: '{"type":"offer","sdp":"old"}' } }, requests: [
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ type: "offer", sdp: "v=0\r\no=- 1 2 IN IP4 10.0.0.1" }), path: `/relay/${HASH}?hash=${HASH}&token=${TOKEN}` },
    { method: "GET", path: `/relay/${HASH}` },
    { method: "POST", headers: { "Content-Type": "application/json" }, body: '{"type":"offer","sdp":"evil"}', path: `/relay/${HASH}?token=nope` },
    { method: "POST", headers: { "Content-Type": "application/json" }, body: '{"type":"offer","sdp":"evil"}', path: `/relay/${HASH}` },
    { method: "POST", headers: { "Content-Type": "application/json" }, body: "[1,2]", path: `/relay/${HASH}?token=${TOKEN}` },
    { method: "GET", path: `/relay/${HASH2}` },
    { method: "GET", path: `/relay/${HASH3}/` },
    { method: "PUT", path: `/relay/${HASH}` },
  ] },
  // fix-app-invite: 팀장의 이메일 초대 (같은 이메일은 대소문자를 가리지 않는다), 이상한 ref 는 버린다
  { name: "invite_owner", rows: [...ZONES, ...MEMBERS(), PAGE_ROW(PAGE1, "goods"), PAGE_ROW(PAGE2, "order")], s3: S3_SESSION, requests: [
    PUT({ from: TEAM, to: USER, email: " New@Example.com ", ref: JSON.stringify([PAGE1, PAGE2, "junk", 7]) }),
  ] },
  // 팀장이 아닌 사람 · 틀린 이메일 · 남의 팀
  { name: "invite_rejected", rows: [...ZONES, ...MEMBERS(), OTHER_USER, PAGE_ROW(PAGE1, "goods")], s3: { ...S3_SESSION, ...S3_OTHER }, requests: [
    PUT({ from: TEAM, to: OTHER, email: "x@example.com", ref: JSON.stringify([PAGE1]) }, { hash: HASH3, token: TOKEN3 }),
    PUT({ from: TEAM, to: USER, email: "not-an-email", ref: JSON.stringify([PAGE1]) }),
    PUT({ from: OTHER, to: USER, email: "x@example.com", ref: JSON.stringify([PAGE1]) }),
  ] },
  // 다시 초대하면 refs 를 덮는다
  { name: "invite_reinvite", rows: [...ZONES, ...MEMBERS(), PAGE_ROW(PAGE1, "goods"), PAGE_ROW(PAGE2, "order")], s3: S3_SESSION, requests: [
    PUT({ from: TEAM, to: USER, email: "new@example.com", ref: JSON.stringify([PAGE1]) }),
    PUT({ from: TEAM, to: USER, email: "new@example.com", ref: JSON.stringify([PAGE2]) }),
  ] },
  // 초대 → 초대받은 사람의 첫 로그인(그 팀에 들어감 · 멤버 행 · 초대 지움) → 그 사람의 JSON GET (초대한 페이지 · 팀 아이템이 보인다)
  { name: "invite_login", rows: [...ZONES, ...MEMBERS(), PAGE_ROW(PAGE1, "goods"), PAGE_ROW(PAGE2, "order"), ITEM("0x" + "11".repeat(20), { cc: CC })],
    s3: { ...S3_SESSION, ...S3_INVITEE }, requests: [
    PUT({ from: TEAM, to: USER, email: "new@example.com", ref: JSON.stringify([PAGE1, PAGE2]) }),
    { method: "GET", path: q({ hash: HASH2, token: TOKEN2 }) },
    { method: "GET", jar: true, headers: { "Content-Type": "application/json" }, path: q({ origin: ORIGIN, href: HREF, created_at: T0 + 1000 }) },
  ] },
  // 주기 · 거두기 한 바퀴 (팀에 든 OTHER 에게 두 페이지, 하나를 거둠, 팀장은 거둘 수 없음)
  { name: "member_grant_revoke", rows: [...ZONES, ...MEMBERS(), OTHER_USER, PAGE_ROW(PAGE1, "goods"), PAGE_ROW(PAGE2, "order")], s3: { ...S3_SESSION, ...S3_OTHER }, requests: [
    PUT({ from: USER, to: OTHER, id: TEAM, cc: USER, ref: JSON.stringify([PAGE1, PAGE2]) }),
    PUT({ from: USER, to: OTHER, id: TEAM, cc: ZERO, ref: JSON.stringify([PAGE1]) }),
    PUT({ from: USER, to: USER, id: TEAM, cc: ZERO, ref: JSON.stringify([PAGE2]) }),
    PUT({ from: OTHER, to: OTHER, id: TEAM, cc: ZERO, ref: JSON.stringify([PAGE2]) }, { hash: HASH3, token: TOKEN3 }),
  ] },
);

// ---------------------------------------------------------------------------------------------------------------------
// fix-member-per-page: 팀장은 모든 페이지, 그 밖의 팀원은 페이지(bcc) · 사이트(cc) 멤버 행이 있는 페이지와 팀 공통 페이지(logis.center)만.
// 팀 base.pages 는 { [사이트 cc]: { [type]: { draft, count } } } — 팀장이 아니면 멤버인 것만 남는다
const HOME_CC = hashId("logis.center");
const OTHER_CC = hashId("other.com");
const PAGE_A = PAGE1; // HREF(/goods/view?no=12) 의 페이지
const PAGE_B = hashId(CC + "/order/view");
const HREF_B = "https://shop.example.com/order/view?no=3";
const PERPAGE_BASE = () => ({ ...TEAM_BASE(), pages: {
  [CC]: { goods: { draft: 0, count: 2 }, order: { draft: 0, count: 1 } },
  [OTHER_CC]: { goods: { draft: 1, count: 4 } },
  [HOME_CC]: { tracking: { draft: 0, count: 3 } },
} });
const PERPAGE_ROWS = (extra = []) => [...ZONES,
  ...MEMBERS().map((r) => (r.values.type === "team" ? { ...r, values: { ...r.values, data: { ...r.values.data, base: PERPAGE_BASE() } } } : r)),
  OTHER_USER, PAGE_ROW(PAGE_A, "goods"), PAGE_ROW(PAGE_B, "order"),
  ITEM("0x" + "11".repeat(20), { cc: CC }), ITEM("0x" + "12".repeat(20), { cc: CC, type: "order" }), ...extra];
const OTHER_COOKIES = { hash: HASH3, token: TOKEN3 };
const INVITEE_COOKIES = { hash: HASH2, token: TOKEN2 };
const S3_PERPAGE = { ...S3_SESSION, ...S3_OTHER, ...S3_INVITEE,
  [`address/${hashId("lee@example.com")}`]: { balance: "1000" }, [`address/${hashId("new@example.com")}`]: { balance: "1000" } };
const JGET = (cookies, over = {}) => ({ ...JSON_GET(over), cookies });
const POST_TEXT = (cookies, from, href, text, over = {}) => ({ method: "POST", cookies, headers: { "Content-Type": "text/html" },
  gzip: encodeURIComponent(`<html><body><h1>${text}</h1></body></html>`), path: q({ origin: ORIGIN, href, created_at: T0 + 1000, from, to: TEAM, ...over }) });
// 사이트 멤버 행 (bcc = 사이트 cc) — 초대 · 권한 주기는 만들지 않는 꼴이지만 1572 의 판정이 읽는다
const SITE_MEMBER = { db: "commerce_logis_apac1_items", table: "items", values: { id: hashId(OTHER + TEAM + CC), type: "member", from: OTHER, to: TEAM, cc: USER,
  bcc: CC, ref: "", data: { name: "lee" }, created_at: T0 - 100000, updated_at: T0 - 100000 } };

SCENARIOS.push(
  // 팀장: 페이지 B 도 멤버(동기화 · 전체 base.pages), POST · PUT 대화가 받아진다, 페이지 A 도
  { name: "perpage_owner", rows: PERPAGE_ROWS(), s3: S3_PERPAGE, requests: [
    JSON_GET({ href: HREF_B }),
    POST_TEXT(COOKIES, USER, HREF_B, "주문 B"),
    PUT({ href: HREF_B, from: USER, to: TEAM, text: "B 의 대화" }),
    JSON_GET(),
  ] },
  // 초대(refs [A]) → 첫 로그인 → A 는 멤버(동기화 · 걸러진 base.pages), B 는 아님(팀 아이템 없음) → POST · PUT 대화도 A 만
  { name: "perpage_invited", rows: PERPAGE_ROWS(), s3: S3_PERPAGE, requests: [
    PUT({ from: TEAM, to: USER, email: "new@example.com", ref: JSON.stringify([PAGE_A]) }),
    { method: "GET", path: q({ hash: HASH2, token: TOKEN2 }) },
    JGET(INVITEE_COOKIES),
    JGET(INVITEE_COOKIES, { href: HREF_B }),
    POST_TEXT(INVITEE_COOKIES, NEWADDR, HREF, "상품 A", { sender: "대구광역시 1" }),
    POST_TEXT(INVITEE_COOKIES, NEWADDR, HREF_B, "주문 B", { sender: "대구광역시 1" }),
    PUT({ from: NEWADDR, to: TEAM, text: "A 의 대화" }, INVITEE_COOKIES),
    PUT({ href: HREF_B, from: NEWADDR, to: TEAM, text: "B 의 대화" }, INVITEE_COOKIES),
  ] },
  // 멤버 행이 없는 옛 팀원: 앱의 클라우드 검색(홈 /tracking) · 이미지 추출(console)은 팀 공통이라 받아진다. 가게 페이지 A 는 막힌다
  { name: "perpage_app_old_member", rows: PERPAGE_ROWS(), s3: S3_PERPAGE, requests: [
    { method: "POST", headers: { "Content-Type": "application/json" }, gzip: { query: " 무거운 상자 ", vector: [0.1, 0.2, 0.3] },
      path: q({ origin: HOME_ORIGIN, created_at: T0 + 1000, hash: HASH3, token: TOKEN3, href: HOME_HREF, from: OTHER, to: TEAM }) + `&hash=${HASH3}&token=${TOKEN3}` },
    { method: "POST", headers: { "Content-Type": "application/json" },
      gzip: { id: "task_1", from: OTHER, to: TEAM, cc: "", bcc: "", ref: "", link: "local", type: "image_extraction", body: "data:application/octet-stream;base64,iVBORw0KGgo=" },
      path: q({ from: OTHER, to: TEAM, format: encodeURIComponent("image/png"), hash: HASH3, token: TOKEN3, href: "https://console.logis.center/" }) },
    JGET(OTHER_COOKIES),
    POST_TEXT(OTHER_COOKIES, OTHER, HREF, "상품 A"),
  ] },
  // 권한 주기(A) → A 멤버 → 거두기 → A 가 다시 막힌다
  { name: "perpage_revoke", rows: PERPAGE_ROWS(), s3: S3_PERPAGE, requests: [
    PUT({ from: USER, to: OTHER, id: TEAM, cc: USER, ref: JSON.stringify([PAGE_A]) }),
    JGET(OTHER_COOKIES),
    POST_TEXT(OTHER_COOKIES, OTHER, HREF, "상품 A"),
    PUT({ from: USER, to: OTHER, id: TEAM, cc: ZERO, ref: JSON.stringify([PAGE_A]) }),
    JGET(OTHER_COOKIES),
    POST_TEXT(OTHER_COOKIES, OTHER, HREF, "상품 A 다시"),
  ] },
  // 사이트 멤버 행(bcc = cc)은 그 사이트의 모든 페이지를 덮는다 (base.pages 는 그 cc 통째로 + 팀 공통)
  { name: "perpage_site_member", rows: PERPAGE_ROWS([SITE_MEMBER]), s3: S3_PERPAGE, requests: [
    JGET(OTHER_COOKIES),
    JGET(OTHER_COOKIES, { href: HREF_B }),
    POST_TEXT(OTHER_COOKIES, OTHER, HREF_B, "주문 B"),
  ] },
);

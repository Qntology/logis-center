// commerce 차분 시험의 시나리오: D1 행 · S3 객체(메타) · 차례로 보낼 요청들
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

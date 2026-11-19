// proxy 차분 시험의 시나리오: D1 에 넣을 행, apac 이 보낼 요청, LLM 가짜 응답
import { scanScenarios } from "./scan_scenarios.mjs";
const T0 = 1760000000000;

export const REQUEST = (over = {}) => ({
  now: T0, id: "T1", ref: "R1", region: "commerce_logis_apac",
  models: { deepinfra: 10000, cloudflare: 3000, "G1-gemini-flash-lite-latest": 4000 },
  limits: {},
  gemini_llm_api: "G1", gemini_llm_model: "gemini-flash-lite-latest", deepinfra: "DI",
  ...over,
});

const TEAM_DATA = () => ({
  base: {
    pages: {},
    goods: { price: { min: 1000, max: 90000 }, quantity: {}, width: {}, height: {}, length: {}, weight: { min: 1 }, shipping_fee: {}, shipping_duration: {}, sale_price: {}, supply_price: {}, low_stock_threshold: {}, discount: {}, min_order_amount: {}, max_discount_amount: {}, usage_limit: {}, usage_per: {}, started_at: {}, expired_at: {} },
    order: { price: {}, quantity: { max: 50 }, width: {}, height: {}, length: {}, weight: {}, shipping_fee: {}, shipping_duration: {}, sale_price: {}, supply_price: {}, low_stock_threshold: {}, discount: {}, min_order_amount: {}, max_discount_amount: {}, usage_limit: {}, usage_per: {}, started_at: {}, expired_at: {} },
  },
  address: ["서울특별시 강남구 테헤란로 1"],
});

const ROWS = (task, extra = []) => [
  { db: "commerce_logis_apac", table: "tasks", values: { id: task.id, type: task.type, ref: task.ref, data: task, created_at: T0 - 60000, updated_at: 0 } },
  { db: "commerce_logis_apac", table: "users", values: { id: "0xteam", type: "team", from: "0xowner", data: TEAM_DATA(), created_at: T0 - 900000, updated_at: 0 } },
  { db: "commerce_logis_apac", table: "users", values: { id: "0xowner", type: "user", data: { sender: "부산광역시 해운대구 1" }, created_at: T0 - 900000, updated_at: 0 } },
  ...extra,
];

const TASK = (over = {}) => ({
  id: "T1", type: "tracking", from: "0xowner", to: "0xteam", cc: "shop.example.com", bcc: "", ref: "R1",
  flag: "kr", zone: "apac1", rpm: 10, contentType: "text/plain", body: "", data: null, ...over,
});

const chat = (content) => ({ choices: [{ message: { content: typeof content === "string" ? content : JSON.stringify(content) } }] });
const gemini = (content) => ({ candidates: [{ content: { parts: [{ text: typeof content === "string" ? content : JSON.stringify(content) }] } }] });

// 프롬프트로 어느 호출인지 가려 그럴듯한 JSON 을 준다
const promptOf = (body) => {
  if (!body || typeof body !== "object") return "";
  if (body.messages) return body.messages.map((m) => (typeof m.content === "string" ? m.content : JSON.stringify(m.content))).join("\n");
  if (body.contents) return body.contents.map((c) => c.parts.map((p) => p.text || "").join("")).join("\n");
  return JSON.stringify(body);
};
const SEARCH_LLM = (over = {}) => ({ url, body }) => {
  const p = promptOf(body);
  const wrap = url.includes("deepinfra") ? chat : gemini;
  if (p.includes("# #date")) return wrap(over.contexts ?? { contexts: [
    { region: "kr", language: "Korean", type: "order", text: "여름 이벤트로 팔린 무거운 제품", status: "complete", substantial: "weight", find: "heavy",
      condition: { date: { gte: "2025-06-01T00:00:00", lte: "2025-08-31T23:59:59" }, price: { lte: 5000, currency: "KRW" } } },
    { region: "kr", language: "Korean", type: "review", text: "리뷰 남긴 고객 메시지", status: null, substantial: null, find: "many", condition: { date: { eq: null } } },
  ] });
  if (p.includes("Segment the natural language content")) return wrap(over.context ?? { context: [
    { language: "Korean", type: "sales", text: "올해 여름 이벤트로 판매된 제품" }, { language: "Korean", type: "goods", text: "무거운 제품" },
    { language: "Korean", type: "review", text: "리뷰 남긴 고객 메시지" }, { language: "Korean", type: "", text: "그 밖" },
  ] });
  if (p.includes("search :")) return wrap({ results: [{ id: "x" }], markdown: "## 결과\n- 무거운 제품 3건" });
  return wrap({});
};

const SALES = (n) => Array.from({ length: n }, (_, i) => ({ db: "commerce_logis_apac1_sales", table: "sales", values: {
  id: "S" + i, type: i % 2 ? "order" : "goods", from: "0xowner", to: "0xteam", cc: "shop.example.com", ref: "R" + i,
  data: { title: "상품 " + i, weight: 1000 + i * 700, sale_price: 1000 * (i + 1) }, created_at: 1752000000000 + i * 86400000,
  index: 1000 + i, status: 9, weight: 1000 + i * 700, sale_price: 1000 * (i + 1), quantity: i,
} }));

export const SCENARIOS = [
  { name: "empty_body", rows: [], request: null, llm: () => ({}) },
  { name: "no_task_row", rows: [], request: REQUEST(), llm: () => ({}) },
  {
    name: "missing_team",
    rows: [{ db: "commerce_logis_apac", table: "tasks", values: { id: "T1", ref: "R1", data: TASK(), created_at: T0 - 60000, updated_at: 0 } }],
    request: REQUEST(), llm: () => ({}),
  },
  {
    name: "image_label",
    rows: ROWS(TASK({ contentType: "image/png", body: "data:image/png;base64,iVBORw0KGgo=" })),
    request: REQUEST(),
    llm: ({ url }) => url.includes("deepinfra")
      ? chat({ tracking_number: "6O89-l234-5S78", recipient_match: true, barcodes: ["689123455878"], text: "강남구 수신 택배" })
      : gemini({}),
  },
  {
    name: "image_label_fenced",
    rows: ROWS(TASK({ contentType: "image/jpeg", body: "data:image/jpeg;base64,/9j/4AAQ" })),
    request: REQUEST(),
    llm: () => chat("```json\n{tracking_number: '5512 3300 9182', recipient_match: false, barcodes: [], text: '해운대 발송',}\n```"),
  },
  {
    name: "image_no_tracking",
    rows: ROWS(TASK({ contentType: "image/png", body: "data:image/png;base64,AA==" })),
    request: REQUEST(),
    llm: () => chat({ text: "not a label" }),
  },
  {
    name: "owner_unreadable",
    rows: ROWS(TASK({ contentType: "image/png", body: "data:image/png;base64,AA==" })).map((r) => (r.values.id === "0xowner" ? { ...r, unreadable: true } : r)),
    request: REQUEST(),
    llm: () => chat({ tracking_number: "1" }),
  },
  {
    name: "image_label_digest",
    rows: ROWS(TASK({ contentType: "image/png", body: "data:image/png;base64,iVBORw0KGgo=", digest: "d0", ref: "R1" }), SALES(3)),
    request: REQUEST(),
    llm: ({ url }) => url.includes("deepinfra")
      ? chat({ tracking_number: "1002", recipient_match: true, barcodes: ["1002"], text: "강남구 수신 택배", title: "택배 상자" })
      : gemini({}),
  },
  {
    name: "search_deepinfra",
    rows: ROWS(TASK({ type: "prompt", body: "올해 여름 이벤트로 판매된 제품중에서 무거운 제품으로 5000원 이하로 많이 팔린 제품 중에서 리뷰를 남긴 고객의 메세지도 보여줘" }), SALES(6)),
    request: REQUEST(), llm: SEARCH_LLM(),
  },
  {
    name: "search_gemini_only",
    rows: ROWS(TASK({ type: "prompt", body: "지난달 취소된 주문 보여줘" }), SALES(4)),
    request: REQUEST({ models: { deepinfra: 0, "G1-gemini-flash-lite-latest": 4000 } }), llm: SEARCH_LLM(),
  },
  {
    name: "search_dirty_json",
    rows: ROWS(TASK({ type: "prompt", body: "쿠폰 이벤트 현황" })),
    request: REQUEST(),
    llm: ({ url, body }) => { const p = promptOf(body); return p.includes("Segment the natural language content") ? chat("```json\n{context: [{language: 'Korean', type: 'coupon', text: '쿠폰 이벤트',},],}\n```") : SEARCH_LLM()({ url, body }); },
  },
  {
    name: "search_single_context",
    rows: ROWS(TASK({ type: "prompt", body: "재고 적은 상품" }), SALES(2)),
    request: REQUEST(),
    llm: SEARCH_LLM({ contexts: { type: "goods", condition: { quantity: { lte: 3 } } }, context: { context: [{ language: "Korean", type: "goods", text: "재고 적은 상품" }] } }),
  },
  {
    name: "search_unparseable",
    rows: ROWS(TASK({ type: "prompt", body: "???" })),
    request: REQUEST(), llm: () => chat("I cannot help with that."),
  },
  {
    name: "rpm_exhausted",
    rows: ROWS(TASK({ rpm: 0 })),
    request: REQUEST({ models: { deepinfra: 1, "shop.example.com": 0 } }),
    llm: () => ({}),
  },
  ...scanScenarios({ ROWS, TASK, REQUEST, extraRows: SALES(3) }),
];

// scenarios/<묶음>.mjs: export default (helpers) => [ 시나리오… ] — 버그 수정마다 그것을 드러내는 시나리오
{
  const fs = await import("node:fs");
  const dir = new URL("./scenarios/", import.meta.url);
  if (fs.existsSync(dir)) {
    const helpers = { T0, REQUEST, ROWS, TASK, TEAM_DATA, SALES, chat, gemini, promptOf, SEARCH_LLM };
    for (const f of fs.readdirSync(dir).filter((f) => f.endsWith(".mjs")).sort()) {
      const mod = await import(new URL(f, dir));
      SCENARIOS.push(...mod.default(helpers));
    }
  }
}

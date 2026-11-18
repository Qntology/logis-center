// 이미지 · 검색 갈래의 버그 수정(fixes/image_search.mjs)을 드러내는 시나리오 — 고치기 전 원본(ORIGINAL=raw)과는 달라야 한다.
// 드러낼 수 없는 것: fix-image-item-reset · fix-search-reset 의 paragraphs/contexts (요청 하나에 작업 하나라 앞 회차가 없음),
// fix-image-embeddings (CLIENT_EMBEDDING 이 참이라 죽은 갈래), fix-search-matches-length · fix-search-delete-index
// (가짜 Vectorize 가 늘 배열을 돌려주고, 요청마다 빈 저장소라 검색에서 맞는 벡터가 없음).
import { ethers } from "ethers";

// 원본의 hashId(text) · crc32(s) — 운송장 번호로 item.index · item.id 를 미리 계산해 맞는 행을 넣는다
const hashId = (text) => ethers.computeAddress(ethers.hashMessage(text)).toLowerCase();
function crc32(s) { var polynomial = 0x04C11DB7, initialValue = 0xFFFFFFFF, finalXORValue = 0xFFFFFFFF, crc = initialValue, table = [], i, j, c; function reverse(x, n) { var b = 0; while (n) { b = b * 2 + x % 2; x /= 2; x -= x % 1; n--; } return b; } for (i = 256; i >= 0; i--) { c = reverse(i, 32); for (j = 0; j < 8; j++) { c = ((c * 2) ^ (((c >>> 31) % 2) * polynomial)) >>> 0; } table[i] = reverse(c, 32); } for (i = 0; i < s.length; i++) { c = s.charCodeAt(i); if (c > 255) { throw new RangeError(); } j = (crc % 256) ^ c; crc = ((crc / 256) ^ table[j]) >>> 0; } return (crc ^ finalXORValue) >>> 0; }
/** 운송장 번호(숫자만) → { index, id } — 원본: item.index = crc32(hashId(type+team.id+no)), item.id = hashId(team.id+index) */
const labelKeys = (no, type = "tracking", team = "0xteam") => {
  const index = crc32(hashId(type + team + no));
  return { index, id: hashId(team + index) };
};

export default ({ T0, REQUEST, ROWS, TASK, SALES, chat, gemini, promptOf, SEARCH_LLM }) => {
  const IMAGE = { contentType: "image/png", body: "data:image/png;base64,iVBORw0KGgo=" };
  const label = (over = {}) => ({ tracking_number: "77001234", recipient_match: true, barcodes: [], text: "강남구 수신 택배", ...over });
  const K = labelKeys("77001234");

  // 두 문맥 모두 벡터 색인이 있는 sales 로 간다
  const TWO_CONTEXTS = { contexts: [
    { region: "kr", language: "Korean", type: "goods", text: "무거운 제품", status: null, substantial: "weight", find: "heavy", condition: { quantity: { lte: 3 } } },
    { region: "kr", language: "Korean", type: "order", text: "취소된 주문", status: "cancel", substantial: null, find: "many", condition: {} },
  ] };
  const searchRows = (body, extra = []) => ROWS(TASK({ type: "prompt", body }), [...SALES(4), ...extra]);

  return [
    // fix-image-gemini-config: Gemini 만 켜진 이미지 — 원본은 generationConfig: null 로 보내고 응답 뒤 던진다
    {
      name: "fix_image_gemini_only",
      rows: ROWS(TASK(IMAGE)),
      request: REQUEST({ models: { deepinfra: 0, cloudflare: 3000, "G1-gemini-flash-lite-latest": 4000 } }),
      llm: ({ url }) => (url.includes("generativelanguage") ? gemini(label({ tracking_number: "5512330", recipient_match: false, text: "해운대 발송" })) : chat({})),
    },
    // fix-image-items-bind: digest 없는 작업 + 같은 id 의 items 행(ref OLD) — 원본은 task.digest(undefined)를 묶어 던지고,
    // 고친 판은 item 의 칸을 묶고 ON CONFLICT 로 ref 까지 고친다
    {
      name: "fix_image_items_item_fields",
      rows: ROWS(TASK(IMAGE), [
        { db: "commerce_logis_apac1_items", table: "items", values: {
          id: K.id, type: "receiving", from: "0xowner", to: "0xteam", cc: "old.example.com", bcc: "", ref: "OLD", digest: "", data: { no: "77001234", time: T0 - 7200000 }, created_at: T0 - 7200000, updated_at: 0,
        } },
      ]),
      request: REQUEST(),
      llm: ({ url }) => (url.includes("deepinfra") ? chat(label({ title: "택배 상자" })) : gemini({})),
    },
    // fix-image-order-slot: 운송장 행 + 그 운송장을 가리키는 판매 행이 있으면 pages[cc].order 를 줄이는데, 칸이 없어 원본은 던진다
    {
      name: "fix_image_order_slot",
      rows: ROWS(TASK(IMAGE), [
        { db: "commerce_logis_apac1_sales", table: "sales", values: {
          id: "SO1", type: "order", from: "0xowner", to: "0xteam", cc: "shop.example.com", bcc: "", ref: "RS1",
          data: { title: "주문 상품", time: T0 - 86400000 }, created_at: T0 - 86400000, index: 5001, status: 1, tracking: K.index, quantity: 1,
        } },
        { db: "commerce_logis_apac1_tracking", table: "tracking", values: {
          id: K.id, type: "tracking", from: "0xowner", to: "0xteam", cc: "shop.example.com", bcc: "", ref: "RT1",
          data: { no: "77001234", time: T0 - 3600000 }, created_at: T0 - 3600000, index: K.index, status: 0, no: "77001234",
        } },
      ]),
      request: REQUEST(),
      llm: ({ url }) => (url.includes("deepinfra") ? chat(label()) : gemini({})),
    },
    // fix-search-reset (queryVector · generation): 두 문맥. 원본은 첫 문맥의 벡터를 다시 써 두 번째 Workers AI 호출이 없고,
    // DeepInfra 가 두 번째 문맥 답에서 바닥나고 Gemini 가 없으면 첫 문맥의 답을 다시 쓴다. (두 문맥이 모두 답을 내면 talks 에
    // 무작위 id 행이 둘 생겨 id 순 정렬이 실행마다 달라지므로, 두 번째 문맥은 답을 못 내게 한다)
    {
      name: "fix_search_generation_per_context",
      rows: searchRows("무거운 제품과 취소된 주문"),
      request: REQUEST({ models: { deepinfra: 3, cloudflare: 3000 }, gemini_llm_api: "" }),
      llm: SEARCH_LLM({ contexts: TWO_CONTEXTS }),
    },
    // fix-search-paragraphs-array: para2graph 가 글을 돌려준다 — 원본은 rowsTrim 이 던져 작업이 통째로 버려진다
    {
      name: "fix_search_paragraphs_string",
      rows: searchRows("무거운 제품"),
      request: REQUEST(),
      llm: SEARCH_LLM({ context: { context: "무거운 제품" } }),
    },
    // fix-search-primitive-results: graph2contexts 가 JSON 글 하나를 돌려준다 — 원본은 results.text = … 에서 던진다
    {
      name: "fix_search_contexts_string",
      rows: searchRows("여름 상품"),
      request: REQUEST(),
      llm: SEARCH_LLM({ contexts: JSON.stringify("여름 상품") }),
    },
    // fix-search-primitive-results: contexts 원소가 글 — 원본은 context.id = hashId() 에서 던지고, 고친 판은 유형 없는 문맥으로 건너뛴다
    {
      name: "fix_search_contexts_strings",
      rows: searchRows("여름 상품"),
      request: REQUEST(),
      llm: SEARCH_LLM({ contexts: { contexts: ["여름 상품", { type: "goods", text: "여름 상품", substantial: "sale_price", find: "few", condition: { sale_price: { gte: 1000 } } }] } }),
    },
    // fix-search-condition-guard: 문맥에 condition 이 없다 — 원본은 Object.keys(undefined) 에서 던진다
    {
      name: "fix_search_condition_missing",
      rows: searchRows("무거운 제품"),
      request: REQUEST(),
      llm: SEARCH_LLM({ contexts: { contexts: [{ type: "goods", text: "무거운 제품", substantial: "weight", find: "heavy" }] } }),
    },
    // fix-search-created-at: 날짜 조건(시작만) + now 뒤에 만들어진 판매 행 — 고친 판만 "created_at" < now 를 붙여 그 행을 뺀다
    {
      name: "fix_search_created_at_bound",
      rows: searchRows("6월 이후 상품", [
        { db: "commerce_logis_apac1_sales", table: "sales", values: {
          id: "SF", type: "goods", from: "0xowner", to: "0xteam", cc: "shop.example.com", ref: "RF",
          data: { title: "미래 상품", weight: 500, sale_price: 2000 }, created_at: T0 + 86400000, index: 2000, status: 9, weight: 500, sale_price: 2000, quantity: 1,
        } },
      ]),
      request: REQUEST(),
      llm: SEARCH_LLM({ contexts: { contexts: [{ type: "goods", text: "6월 이후 상품", find: "many", condition: { date: { gte: "2025-06-01T00:00:00" } } }] } }),
    },
  ];
};

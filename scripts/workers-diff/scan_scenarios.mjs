// 스캔 갈래 시나리오: SyntheticData 의 실제 화면 HTML + 목록 · 상세 분석 LLM 응답
import fs from "node:fs";
import path from "node:path";

const HTML_DIR = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")), "..", "..", "SyntheticData", "commerce", "output_html");
const html = (f) => fs.readFileSync(path.join(HTML_DIR, f), "utf8").replace(/^[\s\S]*?<body[^>]*>/i, "").replace(/<\/body>[\s\S]*$/i, "");
const names = (h) => [...h.matchAll(/<strong class="name">([^<]*)</g)].map((m) => m[1].trim());

const NL = String.fromCharCode(10);
const chat = (content) => ({ choices: [{ message: { content: typeof content === "string" ? content : JSON.stringify(content) } }] });
const gemini = (content) => ({ candidates: [{ content: { parts: [{ text: typeof content === "string" ? content : JSON.stringify(content) }] } }] });
const promptOf = (body) => {
  if (!body || typeof body !== "object") return "";
  if (body.messages) return body.messages.map((m) => (typeof m.content === "string" ? m.content : JSON.stringify(m.content))).join(NL);
  if (body.contents) return body.contents.map((c) => c.parts.map((p) => p.text || "").join("")).join(NL);
  return JSON.stringify(body);
};

// 목록 분석 응답 (list2json 꼴)
const listPage = (type, items, over = {}) => ({
  language: "Korean", type, item: "li.card", more: "a.edit", node: "ul.order-cards", next: "nav.pager a.next", text: `${type} 목록 ${items.length}건`, detail: false, items, ...over,
});
const goodsItems = (h) => names(h).slice(0, 5).map((t, i) => ({
  status: i % 3 ? "show" : "hide", link: `/admin/goods/edit?id=ES1${i}`, id: `ES1${i}`, title: t, sale_price: 12000 + i * 1500, supply_price: 9000 + i * 1000,
  currency: "KRW", quantity: 10 - i, tracking_number: i === 2 ? "5512-3300-9182" : "", registration_date: `2026-0${(i % 8) + 1}-1${i}T09:00:00`,
}));
const orderItems = (h) => names(h).slice(0, 4).map((t, i) => ({
  status: ["progress", "complete", "cancel", "refund"][i % 4], link: `/admin/orders/view?no=20260${i}`, id: `O20260${i}`, title: t, sale_price: 30000 + i,
  supply_price: 0, currency: "KRW", quantity: i + 1, tracking_number: i % 2 ? `6${i}89123455878` : "", registration_date: `2026-05-2${i}T11:30:00`,
}));
const trackingItems = (h) => names(h).slice(0, 3).map((t, i) => ({
  status: ["start", "progress", "stop"][i % 3], id: `TR${i}`, title: t, link: `/admin/delivery/${i}`, registration_date: `2026-06-0${i + 1}T08:00:00`,
}));
const couponItems = () => [0, 1].map((i) => ({ status: "progress", id: `CP${i}`, title: `여름 쿠폰 ${i}`, started_at: "2026-06-01T00:00:00", expired_at: "2026-08-31T23:59:59", registration_date: "2026-05-30T10:00:00" }));

// 상세 분석 응답 (item2json 꼴)
const detailGoods = (h) => ({
  node: "main.page", status: { value: "show", selector: "section.hero .status" }, title: { value: names(h)[0] || "상품", selector: "section.hero h1" },
  sale_price: { value: 15900, selector: "ul.spec li:nth-child(2) span.v" }, supply_price: { value: 11000, selector: "ul.spec li:nth-child(3) span.v" },
  currency: { value: "KRW", selector: "ul.spec li:nth-child(4) span.v" }, quantity: { value: 7, selector: "ul.spec li:nth-child(5) span.v" },
  weight: { value: 1200, selector: "ul.spec li:nth-child(6) span.v" }, registration_date: { value: "2026-04-16T10:01:59", selector: "time" },
});

export const scanLLM = (pick) => ({ url, body }) => {
  const p = promptOf(body);
  const wrap = url.includes("deepinfra") ? chat : gemini;
  if (p.includes("# selector :")) return wrap(pick.detail ?? {});
  if (p.includes("Analyze the provided Pug template")) return wrap(pick.list ?? {});
  if (p.includes("Converts and returns the JSON structure")) return wrap("요약: " + p.length);
  return wrap({});
};

export const scanScenarios = ({ ROWS, TASK, REQUEST, extraRows = [] }) => {
  const goodsList = html("alameda-goods-list-1.html");
  const goodsDetail = html("alameda-goods-detail-ES10037.html");
  const orderList = html("alameda-order-list-1.html");
  const trackingList = html("alameda-tracking-list-1.html");
  const couponList = html("alameda-coupon-list-1.html");
  const scanTask = (body, href, over = {}) => TASK({ type: "page", scan: true, body, href, link: href, contentType: "text/html", digest: "d", ...over });
  return [
    { name: "scan_goods_list", rows: ROWS(scanTask(goodsList, "https://alameda.example.com/admin/goods/list?page=1"), extraRows),
      request: REQUEST(), llm: scanLLM({ list: listPage("goods", goodsItems(goodsList)) }) },
    { name: "scan_order_list", rows: ROWS(scanTask(orderList, "https://alameda.example.com/admin/orders?page=1"), extraRows),
      request: REQUEST(), llm: scanLLM({ list: listPage("order", orderItems(orderList)) }) },
    { name: "scan_tracking_list", rows: ROWS(scanTask(trackingList, "https://alameda.example.com/admin/delivery"), extraRows),
      request: REQUEST(), llm: scanLLM({ list: listPage("tracking", trackingItems(trackingList)) }) },
    { name: "scan_coupon_list", rows: ROWS(scanTask(couponList, "https://alameda.example.com/admin/coupons"), extraRows),
      request: REQUEST(), llm: scanLLM({ list: listPage("coupon", couponItems()) }) },
    { name: "scan_goods_detail", rows: ROWS(scanTask(goodsDetail, "https://alameda.example.com/admin/goods/edit?id=ES10037", { detail: true }), extraRows),
      request: REQUEST(), llm: scanLLM({ list: listPage("goods", [], { detail: true }), detail: detailGoods(goodsDetail) }) },
    { name: "scan_list_then_detail_type", rows: ROWS(scanTask(goodsList, "https://alameda.example.com/admin/goods/list"), extraRows),
      request: REQUEST(), llm: scanLLM({ list: listPage("goods", []), detail: detailGoods(goodsList) }) },
    { name: "scan_gemini_only", rows: ROWS(scanTask(orderList, "https://alameda.example.com/admin/orders?page=2"), extraRows),
      request: REQUEST({ models: { deepinfra: 0, "G1-gemini-flash-lite-latest": 4000 } }), llm: scanLLM({ list: listPage("order", orderItems(orderList)) }) },
    { name: "scan_bad_url", rows: ROWS(scanTask(goodsList, "not a url"), extraRows), request: REQUEST(), llm: scanLLM({}) },
    { name: "scan_llm_garbage", rows: ROWS(scanTask(goodsList, "https://alameda.example.com/admin/goods/list"), extraRows),
      request: REQUEST(), llm: () => chat("Sorry.") },
  ];
};

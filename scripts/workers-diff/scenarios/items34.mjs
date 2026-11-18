// 아이템 반복 3 · 4 구역(원본 4556~5516) 시나리오: 관계(relates)에 이어진 행이 있을 때(relate_row) · 없을 때(초안 등록, fix-edge-draft).
// 원본 그대로(ORIGINAL=raw)면 이어진 행이 없는 첫 관계에서 env["commerce_logis_apac1_undefined"] 로 던지고(fix-edge-draft),
// to 가 아이템이면 객체인 edge.data 를 ungzip 하다 던져(fix-edge-data-ungzip) 스캔이 첫 아이템에서 멈춘다.
// 날짜는 오프셋을 붙인다: Windows 의 workerd 는 오프셋 없는 글을 호스트 시간대로 읽는다(Rust · Cloudflare 는 UTC).
import fs from "node:fs";
import path from "node:path";
import { ethers } from "ethers";
import { scanLLM } from "../scan_scenarios.mjs";

const HTML_DIR = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")), "..", "..", "..", "SyntheticData", "commerce", "output_html");
const html = (f) => fs.readFileSync(path.join(HTML_DIR, f), "utf8").replace(/^[\s\S]*?<body[^>]*>/i, "").replace(/<\/body>[\s\S]*$/i, "");

// 원본의 hashId · crc32 (아이템의 index · id · tracking 을 미리 계산해 맞는 행을 깐다)
const hashId = (text) => ethers.computeAddress(ethers.hashMessage(text)).toLowerCase();
function crc32(s) { var polynomial = 0x04C11DB7, initialValue = 0xFFFFFFFF, finalXORValue = 0xFFFFFFFF, crc = initialValue, table = [], i, j, c; function reverse(x, n) { var b = 0; while (n) { b = b * 2 + x % 2; x /= 2; x -= x % 1; n--; } return b; } for (i = 256; i >= 0; i--) { c = reverse(i, 32); for (j = 0; j < 8; j++) { c = ((c * 2) ^ (((c >>> 31) % 2) * polynomial)) >>> 0; } table[i] = reverse(c, 32); } for (i = 0; i < s.length; i++) { c = s.charCodeAt(i); if (c > 255) { throw new RangeError(); } j = (crc % 256) ^ c; crc = ((crc / 256) ^ table[j]) >>> 0; } return (crc ^ finalXORValue) >>> 0; }
const TEAM = "0xteam";
const CC = "shop.example.com";
const indexOf = (type, no) => crc32(hashId(type + TEAM + no));
const idOf = (index) => hashId(TEAM + index);
const trackingOf = (number) => crc32(hashId("tracking" + TEAM + number));

export default ({ T0, REQUEST, ROWS, TASK }) => {
  const orderList = html("alameda-order-list-1.html");
  const goodsList = html("alameda-goods-list-1.html");
  const scanTask = (body, href, over = {}) => TASK({ type: "page", scan: true, body, href, link: href, contentType: "text/html", digest: "d", ...over });
  const listPage = (type, items) => ({
    language: "Korean", type, item: "li.card", more: "a.edit", node: "ul.order-cards", next: "nav.pager a.next", text: `${type} 목록 ${items.length}건`, detail: false, items,
  });
  const row = (db, table, values) => ({ db: `commerce_logis_apac1_${db}`, table, values: { from: "0xowner", to: TEAM, cc: CC, ...values } });

  // 주문 두 건: 첫째는 운송장 · 이벤트가 있고(이어진 행이 있음), 둘째는 아무것도 없다(모든 관계가 초안)
  const orders = [
    { status: "progress", link: "/admin/orders/view?no=9001", id: "O9001", title: "여름 샌들", sale_price: 39000, supply_price: 0, currency: "krw", quantity: 2,
      tracking_number: "612345678901", event: 7001, registration_date: "2025-09-20T02:30:00Z" },
    { status: "complete", link: "/admin/orders/view?no=9002", id: "O9002", title: "린넨 셔츠", sale_price: 52000, supply_price: 0, currency: "KRW", quantity: 1,
      tracking_number: "", registration_date: "2025-09-21T02:30:00Z" },
  ];
  const o0 = { index: indexOf("order", "O9001"), tracking: trackingOf("612345678901") };
  o0.id = idOf(o0.index);

  // goods 관계(키 "order", sales 의 tracking 으로): to 가 아이템 → edge.data 는 이미 객체 (fix-edge-data-ungzip)
  const salesByTracking = row("sales", "sales", {
    id: "S_TRK", type: "order", ref: "R_TRK", data: { title: "기존 주문", weight: 1500, origin: "app" }, created_at: T0 - 86400000,
    index: 4242, status: 1, tracking: o0.tracking, width: 10, height: 20, length: 30, weight: 1500, size: "M", currency: "KRW", sale_price: 39000, quantity: 2, carrier: "CJ",
  });
  // 그 아이템의 items 행(updated_at 0): 찾은 갈래에서 draft → count 와 UPDATE items (fix-undefined-binds 5062)
  const itemRow = row("items", "items", { id: o0.id, type: "order", ref: "R_ITEM", digest: "", data: { id: o0.id, title: "여름 샌들" }, created_at: T0 - 3600000, updated_at: 0 });
  // event 표 행은 깔지 않는다: to 가 이벤트 행이면 event 문장이 들어가는데, 스키마의 event 표에 "number" 열이 없어 배치 전체가 실패한다(418)
  // coupon 관계: 다른 팀의 행이라 관계 질의(to = 팀)에는 안 잡히고, 초안 찾기(index · type 만)에는 잡힌다
  const couponOther = row("event", "event", {
    id: "CP_OTHER", type: "coupon", to: "0xother", ref: "R_CP", data: { title: "남의 쿠폰" }, created_at: T0 - 172800000, index: 7001, status: 1, discount: 5,
  });

  // 상품 목록: 각 상품의 order · tracking · coupon · event 관계
  const goods = [
    { status: "show", link: "/admin/goods/edit?id=G501", id: "G501", title: "캠핑 의자", sale_price: 45000, supply_price: 30000, currency: "KRW", quantity: 5, event: 8001,
      registration_date: "2025-08-01T00:00:00Z" },
    { status: "hide", link: "/admin/goods/edit?id=G502", id: "G502", title: "접이식 테이블", sale_price: 89000, supply_price: 61000, currency: "KRW", quantity: 0 },
  ];
  const g0 = { index: indexOf("goods", "G501") };
  // goods 의 order 관계(키 "order", sales 의 goods 열)가 덮이고 tracking 관계(키 "order", tracking 의 goods 열)가 남는다
  const trackingByGoods = row("tracking", "tracking", {
    id: "TRK_G", type: "order", ref: "R_TG", data: { title: "상품 배송", no: "5550001" }, created_at: T0 - 86400000, index: 5550001, goods: g0.index, status: 1, no: "5550001", weight: 800,
  });
  // coupon 관계는 이어진 행이 있다(from 이 아이템 → data 가 객체라 풀기 실패, fix-relate-data-reset · to 는 쿠폰 행 → 페이지 칸 없음, fix-edge-slots)
  const goodsCoupon = row("event", "event", {
    id: "CP8001", type: "coupon", ref: "R_CP8", data: { title: "가을 쿠폰" }, created_at: T0 - 86400000, index: 8001, status: 1, discount: 15,
  });

  return [
    { name: "items34_order_relates",
      rows: ROWS(scanTask(orderList, "https://alameda.example.com/admin/orders?page=3"), [salesByTracking, itemRow, couponOther]),
      request: REQUEST(), llm: scanLLM({ list: listPage("order", orders) }) },
    { name: "items34_order_drafts",
      rows: ROWS(scanTask(orderList, "https://alameda.example.com/admin/orders?page=4")),
      request: REQUEST(), llm: scanLLM({ list: listPage("order", orders.map((o) => ({ ...o, tracking_number: "", event: undefined }))) }) },
    { name: "items34_goods_relates",
      rows: ROWS(scanTask(goodsList, "https://alameda.example.com/admin/goods/list?page=7"), [trackingByGoods, goodsCoupon]),
      request: REQUEST(), llm: scanLLM({ list: listPage("goods", goods) }) },
  ];
};

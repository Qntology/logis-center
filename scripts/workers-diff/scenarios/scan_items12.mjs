// fixes/scan_items12.mjs 의 수정을 드러내는 시나리오 (스캔 갈래 · 아이템 반복 1 · 2 구역).
// 요청 하나는 작업 행 하나만 읽으므로(LIMIT 1) 앞 작업의 var 가 남는 경우는 만들 수 없다 — 같은 작업 안에서 남는 _page 만 보인다.
import { ethers } from "ethers";
import { scanLLM } from "../scan_scenarios.mjs";

// 원본 hashId(text)
const hashId = (text) => ethers.computeAddress(ethers.hashMessage(text)).toLowerCase();

const CC = "shop.example.com";
const ORIGIN = "https://shop.example.com";

const LIST_HTML = [
  '<ul class="order-cards">',
  '<li class="card"><a class="edit" href="/admin/goods/edit?id=ES10">편집</a><strong class="name">여름 셔츠</strong><span class="price">12000</span></li>',
  '<li class="card"><a class="edit" href="/admin/goods/edit?id=ES11">편집</a><strong class="name">겨울 코트</strong><span class="price">89000</span></li>',
  "</ul>",
  '<nav class="pager"><a class="next" href="?page=2">다음</a></nav>',
].join("");

const DETAIL_HTML = [
  '<main class="page"><section class="hero"><h1>여름 셔츠</h1><span class="status">show</span></section>',
  '<img class="main" src="a.png"><select class="size"><option selected>M</option></select>',
  '<ul class="spec"><li><span class="v">ES10</span></li><li><span class="v">15900</span></li></ul></main>',
].join("");

const listPage = (type, items, over = {}) => ({
  language: "Korean", type, item: "li.card", more: "a.edit", node: "ul.order-cards", next: "nav.pager a.next", text: `${type} 목록 ${items.length}건`, detail: false, items, ...over,
});

const goodsItem = (i, over = {}) => ({
  status: "show", link: `${ORIGIN}/admin/goods/edit?id=ES1${i}`, id: `ES1${i}`, title: i ? "겨울 코트" : "여름 셔츠", sale_price: 12000 + i * 77000,
  supply_price: 9000, currency: "KRW", quantity: 5, tracking_number: "", registration_date: "2026-04-1" + i + "T09:00:00Z", ...over,
});

const orderItem = (over = {}) => ({
  status: "progress", link: `${ORIGIN}/admin/orders/view?no=202601`, id: "O202601", title: "여름 셔츠 외 1건", sale_price: 30000,
  supply_price: 0, currency: "KRW", quantity: 1, tracking_number: "6O89-1234-5S78", registration_date: "2026-05-20T11:30:00Z", ...over,
});

const detailGoods = (over = {}) => ({
  node: "main.page", status: { value: "show", selector: "section.hero .status" }, title: { value: "여름 셔츠", selector: "section.hero h1" },
  id: { value: "ES10", selector: "ul.spec li:nth-child(1) span.v" }, sale_price: { value: 15900, selector: "ul.spec li:nth-child(2) span.v" },
  currency: { value: "KRW", selector: "ul.spec li:nth-child(2) span.v" }, ...over,
});

export default ({ T0, REQUEST, ROWS, TASK }) => {
  const scanTask = (body, href, over = {}) => TASK({ type: "page", scan: true, body, href, link: href, contentType: "text/html", digest: "d", origin: ORIGIN, ...over });
  const pageRow = (id, type, data, over = {}) => ({
    db: "commerce_logis_center", table: "pages",
    values: { id, type, from: "0xowner", to: "0xteam", cc: CC, bcc: "", ref: "", data, created_at: T0 - 900000, updated_at: 0, ...over },
  });

  // fix-scan-selectors-node: 캐시된 목록 페이지(pageId)가 맞지 않고 task.ref == pageId 라 대문자 경로 페이지를 본다 — 그 node 가 화면에 있으면
  // task.page 가 되어 저장되는 페이지 id 가 그쪽으로 바뀐다 (원본은 BLOB 의 .node 를 읽어 늘 던졌다)
  const listHref = `${ORIGIN}/admin/goods/list`;
  const listPageId = hashId(CC + "/admin/goods/list");
  const upperPageId = hashId(CC + "/ADMIN/GOODS/LIST");

  // fix-scan-reset (_page): 캐시 행(pageId)이 type 없이 data 가 NULL 이고 LLM 이 상세라고 판단 — 원본은 3348 에서 그 _page 가 남아
  // 3409 의 `_page.data.item` 에서 던지고 작업을 통째로 버렸다
  const detailHref = `${ORIGIN}/admin/goods/edit?id=ES10`;
  const editPageId = hashId(CC + "/admin/goods/edit");

  // fix-item-selectors-copy: ref 로 찾아지는 저장된 sales 행의 data 에 item · node 선택자가 있다 (아이템 반복이 item3 를 지나야 저장에 보인다)
  const storedRef = hashId("0xteam" + CC + "/admin/goods/edit?id=es10");

  return [
    {
      name: "s12_selectors_node",
      rows: ROWS(scanTask(LIST_HTML, listHref, { ref: listPageId }), [
        pageRow(listPageId, "goods", { node: "ul.not-here", item: "li.card" }),
        pageRow(upperPageId, "goods", { node: "ul.order-cards", item: "li.card" }),
      ]),
      request: REQUEST({ ref: listPageId }),
      llm: scanLLM({ list: listPage("goods", [goodsItem(0), goodsItem(1)]) }),
    },
    {
      name: "s12_stale_page_null_data",
      rows: ROWS(scanTask(DETAIL_HTML, detailHref), [pageRow(editPageId, "", null)]),
      request: REQUEST(),
      llm: scanLLM({ list: listPage("goods", [], { detail: true }), detail: detailGoods() }),
    },
    {
      // fix-scan-null-element (3376): 배열 값 속성의 null 원소
      name: "s12_detail_null_element",
      rows: ROWS(scanTask(DETAIL_HTML, detailHref, { detail: true })),
      request: REQUEST(),
      llm: scanLLM({ detail: detailGoods({ images: [null, { selector: "img.main", value: "a.png" }] }) }),
    },
    {
      // fix-scan-null-element (3386): 배열 원소 객체 안의 null 값
      name: "s12_detail_null_option",
      rows: ROWS(scanTask(DETAIL_HTML, detailHref, { detail: true })),
      request: REQUEST(),
      llm: scanLLM({ detail: detailGoods({ options: [{ color: null, size: { selector: "select.size option", value: "M" } }] }) }),
    },
    {
      // fix-item-semantic-reset: 상품 둘({} + G1)의 tracking 의미 요약 — deepinfra 는 목록 · 첫 상품에서 다 쓰고, 둘째 상품은
      // Gemini 로 넘어가야 한다 (원본은 앞 상품의 semantic 이 남아 Gemini 를 건너뛰었다)
      name: "s12_semantic_reset",
      rows: ROWS(scanTask(LIST_HTML, `${ORIGIN}/admin/orders`)),
      request: REQUEST({ models: { deepinfra: 2, cloudflare: 3000, "G1-gemini-flash-lite-latest": 4000 } }),
      llm: scanLLM({ list: listPage("order", [orderItem({ goods: [{ id: "G1", title: "여름 셔츠" }] })]) }),
    },
    {
      // fix-goods-unshift: goods 가 글이면 [ {}, "ABC" ] — 원본은 unshift 에서 던지고 글자마다 tracking 행(의미 요약 3번)
      name: "s12_goods_string",
      rows: ROWS(scanTask(LIST_HTML, `${ORIGIN}/admin/orders`)),
      request: REQUEST(),
      llm: scanLLM({ list: listPage("order", [orderItem({ goods: "ABC" })]) }),
    },
    {
      // fix-condition-indexof: 객체 condition — 원본은 indexOf 가 없어 던졌다. 글 condition 은 원래대로 표시
      name: "s12_condition_object",
      rows: ROWS(scanTask(LIST_HTML, listHref)),
      request: REQUEST(),
      llm: scanLLM({ list: listPage("goods", [goodsItem(0, { condition: { used: true } }), goodsItem(1, { condition: "used, refurbish" })]) }),
    },
    {
      name: "s12_condition_string",
      rows: ROWS(scanTask(LIST_HTML, listHref)),
      request: REQUEST(),
      llm: scanLLM({ list: listPage("goods", [goodsItem(0, { condition: "used, refurbish" })]) }),
    },
    {
      name: "s12_stored_item_selectors",
      rows: ROWS(scanTask(LIST_HTML, listHref), [{
        db: "commerce_logis_apac1_sales", table: "sales",
        values: { id: "S-stored", type: "goods", from: "0xowner", to: "0xteam", cc: CC, ref: storedRef,
          data: { no: "10", title: "여름 셔츠", item: "li.card", node: "ul.order-cards" }, created_at: T0 - 900000,
          index: 1, status: 11, sale_price: 11000, quantity: 3 },
      }]),
      request: REQUEST(),
      llm: scanLLM({ list: listPage("goods", [goodsItem(0)]) }),
    },
  ];
};

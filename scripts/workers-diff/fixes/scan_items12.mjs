// 스캔 갈래 (scan.rs) · 아이템 반복 1 · 2 구역 (item1.rs · item2.rs) · parseBody (pug.rs) 의 원본 버그 수정 — Rust 판과 같은 수정을 원본 TS 에.
// find 는 공백을 무시하고 맞추며 파일 전체에서 정확히 한 번 맞아야 한다(fixes.mjs).
// fix-sql-params(값을 SQL 글에 잇지 않고 묶기)는 Rust 판에만 있다: 시험 데이터에 작은따옴표가 없어 결과가 같다.
export default [
  // 1. var page · var before · var _page 는 작업(페이지)마다 undefined 에서 시작
  { id: "fix-scan-reset", worker: "proxy",
    find: "var page var pageType = ''",
    replace: "var page = undefined\n\n\t\t\t\t\t\t\t\t\tvar pageType = ''" },
  { id: "fix-scan-reset", worker: "proxy",
    find: "var before if(isDetail){ selectors.detail = true",
    replace: "var before = undefined\n\n\t\t\t\t\t\t\t\t\tif(isDetail){\n\t\t\t\t\t\t\t\t\t\tselectors.detail = true" },
  { id: "fix-scan-reset", worker: "proxy",
    find: "var _page if(results.length){ _page = results[0]",
    replace: "var _page = undefined\n\n\t\t\t\t\t\t\t\t\t\tif(results.length){\n\t\t\t\t\t\t\t\t\t\t\t_page = results[0]" },
  // 2. 풀지 않은 BLOB(_page.data) 대신 풀어 읽은 _selectors.node
  { id: "fix-scan-selectors-node", worker: "proxy",
    find: "document.querySelectorAll(_page.data.node)",
    replace: "document.querySelectorAll(_selectors.node)" },
  // 3. typeof null == "object" — null 원소는 건너뛴다
  { id: "fix-scan-null-element", worker: "proxy",
    find: 'if(typeof option == "object" && Object.keys(option).length){',
    replace: 'if(option && typeof option == "object" && Object.keys(option).length){' },
  { id: "fix-scan-null-element", worker: "proxy",
    find: '}else if(typeof opt == "object" && Object.keys(opt).length){',
    replace: '}else if(opt && typeof opt == "object" && Object.keys(opt).length){' },
  // 4. 상세면 detail = null (3342) — 객체일 때만 detail.page · detail.id
  { id: "fix-detail-null", worker: "proxy",
    find: "if(isDetail){ if(!detail.page){",
    replace: 'if(isDetail && detail && typeof detail == "object"){\n\t\t\t\t\t\t\t\t\t\t\t\t\tif(!detail.page){' },
  // 5. 새 아이템에 선택자가 없으면 저장된 item · node 를 쓴다 (원본 조건은 늘 거짓)
  { id: "fix-item-selectors-copy", worker: "proxy",
    find: "if(item.data.item && !_item.data.item){",
    replace: "if(!item.data.item && _item.data.item){" },
  // 6. mergeNode 는 새 객체 — items[i] 에도 다시 넣는다
  { id: "fix-items-writeback", worker: "proxy",
    find: "item = mergeNode(_item, item) var { results } = await env[`commerce_logis_${zoneRegion}_items`]",
    replace: "item = mergeNode(_item, item)\n\t\t\t\t\t\t\t\t\t\t\t\titems[i] = item\n\n\t\t\t\t\t\t\t\t\t\t\t\tvar { results } = await env[`commerce_logis_${zoneRegion}_items`]" },
  { id: "fix-items-writeback", worker: "proxy",
    find: "item = mergeNode(_item, item) }else{",
    replace: "item = mergeNode(_item, item)\n\t\t\t\t\t\t\t\t\t\t\t\titems[i] = item\n\n\t\t\t\t\t\t\t\t\t\t\t}else{" },
  { id: "fix-items-writeback", worker: "proxy",
    find: "item = mergeNode(_item, item) } } try{",
    replace: "item = mergeNode(_item, item)\n\t\t\t\t\t\t\t\t\t\t\t\t\titems[i] = item\n\t\t\t\t\t\t\t\t\t\t\t\t}\n\n\t\t\t\t\t\t\t\t\t\t\t}\n\n\t\t\t\t\t\t\t\t\t\t\ttry{" },
  // 7. var semantic · var embeddings 는 상품마다 undefined 에서 시작
  { id: "fix-item-semantic-reset", worker: "proxy",
    find: "payment_origin : tracking.payment_origin }) var semantic",
    replace: "payment_origin : tracking.payment_origin\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t})\n\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tvar semantic = undefined" },
  { id: "fix-item-semantic-reset", worker: "proxy",
    find: "console.log('[EMBED-PENDING] tracking relay delegated to Client App:', tracking.id) }else{ var embeddings",
    replace: "console.log('[EMBED-PENDING] tracking relay delegated to Client App:', tracking.id)\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t}else{\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tvar embeddings = undefined" },
  // 8. .tirm() 오타 (CLIENT_EMBEDDING 이 참이라 닿지 않는 갈래)
  { id: "fix-tirm", worker: "proxy",
    find: "'', semantic.tirm()) var $VectorizeVector: VectorizeVector[] = embeddings.map((values, i) => { return { id: item.id,",
    replace: "'', semantic.trim())\n\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tvar $VectorizeVector: VectorizeVector[] = embeddings.map((values, i) => {\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\treturn {\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tid: item.id," },
  // 9. goods 가 배열이 아니면 배열로 감싼 뒤(비었으면 빈 배열) 앞에 {} 하나
  { id: "fix-goods-unshift", worker: "proxy",
    find: 'try{ if(typeof goods.length != "undefined"){ goods.unshift({})',
    replace: 'try{\n\t\t\t\t\t\t\t\t\t\t\t\tif(!Array.isArray(goods)){\n\t\t\t\t\t\t\t\t\t\t\t\t\tgoods = goods ? [goods] : []\n\t\t\t\t\t\t\t\t\t\t\t\t}\n\n\t\t\t\t\t\t\t\t\t\t\t\tif(typeof goods.length != "undefined"){\n\t\t\t\t\t\t\t\t\t\t\t\t\tgoods.unshift({})' },
  // 10. condition 이 글 · 배열일 때만 indexOf
  { id: "fix-condition-indexof", worker: "proxy",
    find: "if(item.condition){ if(item.condition.indexOf('used') > -1){",
    replace: "if(item.condition && (typeof item.condition == \"string\" || Array.isArray(item.condition))){\n\t\t\t\t\t\t\t\t\t\t\t\tif(item.condition.indexOf('used') > -1){" },
  // 11. 관계 오류 글의 유형: try 앞에서 related[r], query[0].type 을 읽으면 그 값
  { id: "fix-relate-err-type", worker: "proxy",
    find: "try{ if(query.length){ var table = query[0].table var type = query[0].type",
    replace: "var relateType = related[r]\n\n\t\t\t\t\t\t\t\t\t\t\t\t\ttry{\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tif(query.length){\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tvar table = query[0].table\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tvar type = relateType = query[0].type" },
  { id: "fix-relate-err-type", worker: "proxy",
    find: "'tracking inner err'+type+err",
    replace: "'tracking inner err'+relateType+err" },
  // 12. parseBody: 인자 body 를 가리지 않고, page.selectors 를 쓴다 (부르는 곳은 없다)
  { id: "fix-parse-body", worker: "proxy",
    find: "function parseBody(body, page){ var body = ''",
    replace: "function parseBody(body, page){\n\tbody = body ? body : ''\n\n\tvar selectors = page.selectors" },
];

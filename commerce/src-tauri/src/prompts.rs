pub fn page_type_prompt() -> String { 
    r###"[TASK]
Based on the provided Pug template, identify the primary category.

[SCHEMA DEFINITIONS]
- type: The main category. Must be one of:
  - "order": Order list, Order history, Order details, Checkout success.
  - "goods": Product list, product detail.
  - "tracking": Shipment tracking status, delivery history.
  - "review": Product reviews, feedback list.
  - "coupon": Coupon list, discount events.
  - "event": Promotion pages, event announcements.
  - "": If none of the above match.
- language: ISO 639-1 language code.

[OUTPUT FORMAT]
{ "type": "String" }"###.to_string() 
}

pub fn extract_titles_prompt(page_type: &str) -> String {
    let (category_desc, titles_desc, title_desc) = match page_type {
        "goods" => ("product", "product titles", "product title"),
        "order" => ("product", "order product titles", "order product title"),
        "tracking" => ("product", "tracking product titles", "tracking product title"),
        "review" => ("title", "review titles", "review title"),
        "coupon" => ("title", "coupon titles", "coupon title"),
        "event" => ("title", "event titles", "event title"),
        _ => ("title", "titles", "title"),
    };

    let template = r###"[TASK]
Find all the {TITLES} from the following PUG/HTML content.

[SCHEMA DEFINITIONS]
{ {CATEGORY}: ["{TITLE}"] }

[OUTPUT FORMAT]
{ {CATEGORY}: [...] }

RETURN JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{CATEGORY}", category_desc)
            .replace("{TITLES}", titles_desc)
            .replace("{TITLE}", title_desc)
            .replace("{TYPE}", page_type)
}

// pub fn get_trade_doc_classification_prompt() -> String {
//     r###"Classify document type. Choose strictly from: PI, CI, BL, AWB, PL, CO, LC, TRACKING, Unknown. 
// Return JSON exactly like: {"doc_type": "BL"}
// NO EXPLANATION."###.to_string()
// }

pub fn get_trade_doc_classification_prompt() -> String {
    // 🌟 [CLASSIFIER v2 / 27 CODES]
    //  ── 왜 늘리는가 ──
    //   get_trade_doc_slice_config 는 27종 좌표를 갖고 있는데
    //   분류기가 9종만 낼 수 있어 19개 분기가 도달 불가였습니다.
    //   extract_shipping_conditions 도 16종을 조건으로 뽑으므로
    //   저장(9종) ↔ 조회(16종) 사이에 영구 공백이 있었습니다.
    //
    //  ── 오분류 완화 ──
    //   2B 비전 모델이 27갈래를 정확히 가르기는 어렵습니다.
    //   다만 slice_config 가 같은 그룹을 한 좌표로 묶어 두었으므로
    //   (CI|PI|SC / ED|ID|CINV / IC|WC|CA|PHYTO|HC|BEN_CERT / DGD|MSDS / POA|BIZ_LIC|INS)
    //   그룹 내 혼동은 추출 품질에 영향을 주지 않습니다.
    //   그래서 아래 프롬프트도 '그룹 → 코드' 순서로 제시합니다.
    r###"Classify this trade document. Return the single closest code.

[GROUPS]
1. Contract & Payment
   PO  = Purchase Order
   PI  = Proforma Invoice
   SC  = Sales Contract
   LC  = Letter of Credit
2. Shipping & Transport
   CI  = Commercial Invoice
   PL  = Packing List
   BL  = Bill of Lading
   AWB = Air Waybill
   SA  = Shipping Advice
   DO  = Delivery Order
   AN  = Arrival Notice
   BC  = Booking Confirmation
3. Customs
   ED   = Export Declaration
   ID   = Import Declaration
   CINV = Customs Invoice
   CO   = Certificate of Origin
4. Inspection & Certificates
   IC       = Inspection Certificate
   WC       = Weight Certificate
   CA       = Certificate of Analysis
   PHYTO    = Phytosanitary Certificate
   HC       = Health Certificate
   BEN_CERT = Beneficiary Certificate
5. Special & Legal
   DGD     = Dangerous Goods Declaration
   MSDS    = Material Safety Data Sheet
   POA     = Power of Attorney
   BIZ_LIC = Business License
   INS     = Insurance Policy
6. Parcel
   TRACKING = Courier label / parcel waybill

If none fit, return "Unknown".

[OUTPUT FORMAT]
{"doc_type": "BL"}

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###.to_string()
}

/// 🌟 [TRADE SCHEMA v2 / BASE + OVERLAY]
///  ── v1 의 결함 ──
///   시그니처가 `_doc_type` 이었습니다. 즉 27종 서식에 전부 같은 27개 필드를
///   물어봤습니다. L/C 의 tenor, DGD 의 un_number, CA 의 result_value 처럼
///   그 서식에만 존재하는 축은 추출 자체가 불가능했습니다.
///
///  ── 왜 bias.json 인가 ──
///   app-logis-center 의 get_category_schema 는 400줄짜리 doc_type 하드코딩입니다.
///   그대로 옮기면 새 서식마다 Rust 를 고치고 재빌드해야 합니다.
///   이 코드베이스가 이미 path_alias / multilingual_value_anchor / abstract_bridge 를
///   bias.json 으로 옮긴 것과 같은 이유로, 스키마도 데이터로 취급합니다.
///
///  ── 필드 이름 ──
///   base 는 extract_shipping_conditions(검색)와 '같은 이름' 을 씁니다.
///   그래야 저장과 조회가 alias 를 거치지 않고 바로 만납니다.
///   레거시 데이터는 path_alias 가 흡수합니다.
pub fn get_trade_category_schema(category: &str, doc_type: &str) -> String {
    use serde_json::Value;

    // ── 코드 폴백 base : bias.json 에 trade_schema 노드가 없어도 동작해야 합니다 ──
    fn fallback_base(category: &str) -> Vec<(&'static str, &'static str)> {
        match category {
            "header" => vec![
                ("doc_type",         "Document kind code {String}"),
                ("doc_number",       "Primary identifier (B/L No, Invoice No, PO No) {String}"),
                ("issue_date",       "Date of issue (YYYY-MM-DD) {String}"),
                ("reference_number", "Any other reference number printed {String}"),
            ],
            "parties" => vec![
                ("sender_name",       "Shipper, Seller, Exporter {String}"),
                ("sender_address",    "Address of sender {String}"),
                ("recipient_name",    "Consignee, Buyer, Importer {String}"),
                ("recipient_address", "Address of recipient {String}"),
                ("notify_party_name", "Notify party name {String}"),
            ],
            "logistics" => vec![
                ("vessel",         "Vessel name or Flight number {String}"),
                ("voyage_number",  "Voyage or flight leg number {String}"),
                ("pol",            "Port of Loading / Airport of Departure {String}"),
                ("pod",            "Port of Discharge / Airport of Destination {String}"),
                ("etd",            "Estimated time of departure {String}"),
                ("eta",            "Estimated time of arrival {String}"),
                ("transport_mode", "Sea, Air, Road, Rail {String}"),
            ],
            "conditions" => vec![
                ("incoterms",            "FOB, CIF, EXW, DDP, DAP {String}"),
                ("payment_terms",        "T/T, L/C, Net30 {String}"),
                ("freight_payment_term", "Freight Prepaid or Freight Collect {String}"),
            ],
            "financials" => vec![
                ("currency",        "ISO 4217 currency code {String}"),
                ("amount",          "Grand total amount {Number}"),
                ("amount_subtotal", "Subtotal before tax and charges {Number}"),
                ("amount_tax",      "Tax or VAT amount {Number}"),
            ],
            "cargo" => vec![
                ("package_count", "Total number of packages (NOT money) {Number}"),
                ("package_unit",  "Package unit (CTN, PLT, PKG) {String}"),
                ("weight_gross",  "Total gross weight {Number}"),
                ("weight_net",    "Total net weight {Number}"),
                ("volume",        "Total volume in CBM {Number}"),
                ("marks_numbers", "Marks and numbers {String}"),
            ],
            "items" => vec![
                ("description", "Description of goods {String}"),
                ("quantity",    "Line item quantity {Number}"),
                ("unit",        "Unit of measure {String}"),
                ("hs_code",     "HS code / tariff number {String}"),
                ("unit_price",  "Unit price {Number}"),
                ("total_price", "Line total {Number}"),
            ],
            "containers" => vec![
                ("container_number", "Container number (4 letters + 7 digits) {String}"),
                ("seal_number",      "Seal number {String}"),
                ("type_size",        "Size and type (20GP, 40HC) {String}"),
            ],
            _ => vec![],
        }
    }

    // ── bias.json 에서 { field: desc } 맵을 읽습니다 ──
    fn read_node(path: &[&str]) -> Option<serde_json::Map<String, Value>> {
        let mut cur: &Value = &crate::parsing::BIAS_DICT;
        for p in path {
            cur = cur.get(*p)?;
        }
        cur.as_object().cloned()
    }

    // 1) base
    let mut fields: Vec<(String, String)> = Vec::new();
    if let Some(obj) = read_node(&["trade_schema", "base", category]) {
        for (k, v) in obj {
            fields.push((k, v.as_str().unwrap_or("{String}").to_string()));
        }
    } else {
        for (k, d) in fallback_base(category) {
            fields.push((k.to_string(), d.to_string()));
        }
    }

    // 2) overlay : 이 서식에만 존재하는 축을 덧붙입니다.
    //    같은 이름이면 overlay 설명이 이깁니다(서식별 뉘앙스가 더 정확하므로).
    if let Some(obj) = read_node(&["trade_schema", "overlay", doc_type, category]) {
        for (k, v) in obj {
            let desc = v.as_str().unwrap_or("{String}").to_string();
            if let Some(slot) = fields.iter_mut().find(|(n, _)| n == &k) {
                slot.1 = desc;
            } else {
                fields.push((k, desc));
            }
        }
    }

    if fields.is_empty() {
        return format!(
            "RULES: Output JSON ONLY. MISSION: Extract data for category '{}'.\nSCHEMA:\n{{}}",
            category.to_uppercase()
        );
    }

    // 3) 렌더링 : items / containers 는 배열 스키마입니다.
    //    (merge_json_manual 이 items → line_items 로 매핑하므로 키 이름은 그대로 둡니다)
    let is_array = category == "items" || category == "containers";
    let body = fields
        .iter()
        .map(|(k, d)| format!("  \"{}\": \"{}\"", k, d.replace('"', "'")))
        .collect::<Vec<_>>()
        .join(",\n");

    let schema = if is_array {
        format!("[ {{\n{}\n}} ]", body)
    } else {
        format!("{{\n{}\n}}", body)
    };

    format!(
        "RULES: Follow comments strictly. Output JSON ONLY. Omit any field not visible in the image.\nMISSION: Extract data for category '{}' of a {} document.\nSCHEMA:\n{}",
        category.to_uppercase(),
        doc_type,
        schema
    )
}

// pub fn extract_shipping_conditions(query: &str, language: &str) -> String {
//     let template = r###"Task: Act as a deterministic shipping and trade logistics semantic parser.
// Extract the logistics filters from the natural language query into the JSON format.

// [SCHEMA DEFINITION]
// Extract the following tracking/trade properties if semantically present in the text:
// - "no": Tracking number, B/L number, Invoice number.
// - "status": Shipping status (draft, progress, return, complete, error).
// - "vessel": Vessel name, Flight No, or Carrier.
// - "pol": Port of Loading, Origin, Departure point.
// - "pod": Port of Discharge, Destination, Arrival point.
// - "sender_name": Shipper, Seller, or Exporter name.
// - "recipient_name": Consignee, Buyer, or Importer name.
// - "incoterms": Incoterms (e.g., FOB, CIF, EXW).
// - "weight": Cargo or gross weight.
// - "amount": Total financial amount or price.

// [TRANSFORMATION LOGIC]
// For EVERY extracted field, wrap it in an operator object:
// { "operator": "eq" | "gt" | "lt" | "gte" | "lte" | "contains", "value": <extracted_value> }
// - Use "contains" for text fields, names, ports, vessels.
// - Use "eq" for strict identifiers or status.

// [QUERY]
// {QUERY}

// [OUTPUT FORMAT]
// { "<property_name>": { "operator": "...", "value": "..." } }

// [ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;

//     template.replace("{QUERY}", query).replace("{LANGUAGE}", language)
// }

pub fn extract_shipping_conditions(query: &str, language: &str) -> String {
    // 🌟 [TRADING SCHEMA v2]
    //  app-logis-center 의 get_search_schema_definitions 가 정의하던 무역 축을 전량 흡수합니다.
    //  기존 10개 필드만으로는 '부킹번호로 찾아줘', '컨테이너 MSCU1234567',
    //  'ETA 다음주인 건' 같은 실무 질의가 통째로 조건 없이 넘어갔습니다.
    //
    //  ⚠️ 여기서 뽑힌 조건은 전부 Dexie(executeDexiePlan)가 data.* 경로로 실행합니다.
    //     LanceDB 는 봉투 스코프(mode/type/cc)만 담당하므로,
    //     이 목록에 필드를 추가해도 Rust 스키마나 SQL 을 고칠 필요가 전혀 없습니다.
    let template = r###"Task: Act as a deterministic shipping and trade logistics semantic parser.
Extract the logistics filters from the natural language query into the JSON format.

[SCHEMA DEFINITION]
Extract the following trade document properties if semantically present in the text:

# Document Identity
- "doc_type": Document kind (BL, AWB, CI, PI, PL, PO, SC, LC, CO, ED, ID, DO, AN, BC, DGD, MSDS).
- "doc_number": Primary document identifier (B/L No, AWB No, Invoice No, PO No, Contract No).
- "no": Tracking number, parcel number, or any generic reference number.
- "status": Shipping status (draft, progress, return, complete, error).
- "issue_date": Date the document was issued.
- "expiry_date": Expiry date (mainly L/C).

# Transport
- "vessel": Vessel name or Flight number.
- "voyage_number": Voyage or flight leg number.
- "pol": Port of Loading, Origin, Departure point.
- "pod": Port of Discharge, Destination, Arrival point.
- "place_receipt": Place of Receipt.
- "place_delivery": Place of Delivery.
- "etd": Estimated Time of Departure.
- "eta": Estimated Time of Arrival.
- "transport_mode": Sea, Air, Road, or Rail.

# Parties
- "sender_name": Shipper, Seller, Exporter, or Vendor name.
- "recipient_name": Consignee, Buyer, or Importer name.
- "notify_party_name": Notify Party name.

# Commercial Terms
- "incoterms": Incoterms code (FOB, CIF, EXW, DDP, DAP).
- "payment_terms": Payment condition (T/T, L/C, Net30).
- "freight_payment_term": Freight Prepaid or Freight Collect.
- "currency": ISO 4217 currency code.
- "amount": Total financial amount.
- "freight_amount": Freight charges only.
- "insurance_amount": Insurance charges only.
- "local_charges": Local handling charges.

# Cargo
- "container_number": Container number (4 letters + 7 digits).
- "seal_number": Seal number.
- "package_count": Number of packages or cartons.
- "weight_gross": Gross weight.
- "weight_net": Net weight.
- "volume": Volume in CBM.
- "hs_code": HS Code or tariff number.
- "marks_numbers": Shipping marks and numbers.

# Cross References
- "reference_invoice": Referenced commercial invoice number.
- "reference_lc": Referenced letter of credit number.
- "reference_booking": Referenced booking number.

[TRANSFORMATION LOGIC]
For EVERY extracted field, wrap it in an operator object:
{ "operator": "eq" | "gt" | "lt" | "gte" | "lte" | "contains", "value": <extracted_value> }
- Use "eq" for strict identifiers: doc_number, container_number, seal_number, hs_code, no, status, doc_type, incoterms, currency.
- Use "contains" for free text: names, ports, vessels, marks_numbers, payment_terms.
- Use "gte" / "lte" for date ranges and numeric ranges.
- Omit any field that is NOT explicitly present in the query. Never invent a value.

[QUERY]
{QUERY}

[OUTPUT FORMAT]
{ "<property_name>": { "operator": "...", "value": "..." } }

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{QUERY}", query).replace("{LANGUAGE}", language)
}

pub fn get_image_extraction_prompt(region: &str, language: &str, page_type: &str, address: &str) -> String {
    if page_type == "tracking" || page_type == "goods" {
        let template = r###"[TASK]
Convert the image to fit the structured JSON format. 

[CONTEXT]
Region: {REGION}
Recipient Address: {ADDRESS}
Current Language: {LANGUAGE}

[INSTRUCTION]
1. Extract the tracking_number or document number.
2. Set recipient_match to true if the label address matches the context address.
3. Extract all visible barcodes into an array.

[OUTPUT FORMAT]
{ "tracking_number": "string", "recipient_match": boolean, "barcodes": ["string"] }

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;
        template.replace("{REGION}", region).replace("{ADDRESS}", address).replace("{LANGUAGE}", language)
    } else {
        String::new()
    }
}

pub fn extract_table_structure_prompt(page_type: &str, item_selector: &str, pug_content: &str, reference_row: &str) -> String {
    let template = r###"[PUG CONTENT]
{PUG_CONTENT}

[Reference: Row Structure]
{REFERENCE_ROW}

[Instruction]
Locate the main table wrapper, its body container, and its corresponding header container within the [PUG CONTENT].

[Rules]
1. Tag Agnostic: Do NOT assume traditional <table> tags. The structure could be built using <div>, <ul>/<li>, or other semantic tags. Analyze logically.
2. Fill out the `table` selector FIRST to logically establish the common parent wrapper that encompasses both the header (thead) and the items (tbody).
3. The `tbody` selector is exactly "{ITEM_SELECTOR}". Return it as provided.
4. Provide the final exact CSS selector for the `thead` based on your analysis within that table wrapper.

[OUTPUT FORMAT]
{ "{TYPE}": { "tbody": { "selector": "{ITEM_SELECTOR}" }, "table": { "selector": "..." }, "thead": { "selector": "..." } } }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{TYPE}", page_type)
            .replace("{ITEM_SELECTOR}", item_selector)
            .replace("{PUG_CONTENT}", pug_content)
            .replace("{REFERENCE_ROW}", reference_row)
}

pub fn analytic_report_prompt() -> String {
    r###"[TASK]
You are a User Behavior Analysis Expert. Interpret raw HTML interactions to understand the user's specific intent and analyze the selection context within a list or a group of items.

Analyze the parallel arrays of 'Clicked HTML' (the selected element) and 'Related HTML' (the surrounding structure).
If 'Previous Analysis' is provided, use it to infer the user's behavioral flow and connect past actions with the current click.

[ANALYSIS GUIDELINES & CHAIN OF THOUGHT]
Fill out the JSON keys in the exact order specified below. Use 'analysis_*' keys to logically establish the context before finalizing the outputs.

1. analysis_target: Identify the primary entity name and its key attributes from the Clicked HTML.
2. analysis_surroundings: Identify the neighboring items or alternatives displayed in the Related HTML that were NOT selected.
3. action: Determine the specific user intent for clicking the item. Must explicitly include the primary entity name and key attributes. Output as a short verb phrase.
4. relate: Summarize the surrounding unselected items to capture the context of the choice. Do not summarize the clicked item itself in this field.
5. summary: Provide a detailed explanation of what the user aimed to accomplish on this page. Must explicitly reference the extracted primary entity and its key attributes.

[OUTPUT FORMAT]
{
    "actions": {
        "https://hostname.com/pathname?search=parameter": {
            "records": [
                {
                    "id": "...",
                    "analysis_target": "...",
                    "analysis_surroundings": "...",
                    "relate": [...],
                    "action": "..."
                }
            ],
            "summary": "..."
        }
    },
    "cross_action_flow": "...",
    "intent_evolution": "...",
    "consistent_preferences": "..."
}

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###.to_string()
}

// 🌟 [ANALYTIC SEMANTIC] 원시 outerHTML 을 PUG 로 변환하고 속성을 전부 제거한 뒤,
//    '태그 구조 + 화면 텍스트' 만 남은 상태에서 Qwen3.5 2B 가 의미를 요약합니다.
//    ── 왜 속성을 지우는가 ──
//     class="btn_prd_option_2 on" / id="cnt_capa_1" 같은 값은 사이트마다 제각각이라
//     LLM 이 그것을 '의미' 로 오인해 존재하지 않는 속성을 지어냅니다.
//     속성을 제거하면 남는 신호가 (a / li / td / 텍스트) 뿐이므로
//     모델은 화면에 실제로 인쇄된 문자열만 근거로 삼게 됩니다.
pub fn analytic_semantic_prompt(
    event_type: &str,
    link: &str,
    lang: &str,
    target_pug: &str,
    related_pug: &str,
) -> String {
    let template = r###"[TASK]
You are a User Behavior Analysis Expert. The raw HTML has already been converted into an ATTRIBUTE-FREE PUG tree, so ONLY the semantic tag structure and the visible text remain. Interpret that structure and describe what the user did.

[CONTEXT]
Event Type: {EVENT_TYPE}
Page: {LINK}
Output Language: {LANG}

[TARGET ELEMENT — the element the user actually interacted with]
{TARGET_PUG}

[SURROUNDING ELEMENTS — siblings shown next to it that the user did NOT choose]
{RELATED_PUG}

[RULES]
1. "action" MUST name the concrete entity in [TARGET ELEMENT] (product title, option value, menu label, typed value, price) EXACTLY as it is printed there. Never invent a name that is not printed.
2. "relate" describes only the NEIGHBOURING items in [SURROUNDING ELEMENTS]. Never describe the target itself here. Return an empty array when there is no sibling.
3. "summary" is ONE sentence explaining what the user was trying to accomplish on this page, and MUST reuse the same entity name used in "action".
4. Copy every proper noun, product name, code, price and number EXACTLY as printed. Do not translate, round, or reformat them.
5. If [TARGET ELEMENT] carries no readable text at all, return null for every key. A null answer is correct data; an invented one is corrupted data.

[OUTPUT FORMAT]
{ "action": String, "relate": [String], "summary": String }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON. /no_think"###;

    template
        .replace("{EVENT_TYPE}", event_type)
        .replace("{LINK}", link)
        .replace("{LANG}", lang)
        .replace("{TARGET_PUG}", target_pug)
        .replace("{RELATED_PUG}", related_pug)
}

// 🌟 [ANALYTIC FLOW] 같은 사용자·같은 페이지에서 구조화된 여러 행동을 묶어
//    흐름(cross_action_flow) / 의도 변화(intent_evolution) / 반복 성향(consistent_preferences)을
//    한 번에 합성합니다. analytics-logis-center 의 Cron 산출물과 동일한 3축입니다.
pub fn analytic_flow_prompt(lang: &str, records_json: &str) -> String {
    let template = r###"[TASK]
You are a User Behavior Analysis Expert. Below is a time-ordered list of ALREADY STRUCTURED user actions taken by ONE user. Synthesize them into a behavioural narrative.

[CONTEXT]
Output Language: {LANG}

[STRUCTURED ACTION RECORDS]
{RECORDS}

[RULES]
1. Use ONLY the facts present in [STRUCTURED ACTION RECORDS]. Never invent a page, product or option that is not listed.
2. Keep every proper noun, product name and number EXACTLY as printed in the records.
3. "cross_action_flow": describe the overall path in order (what was viewed, what was compared, what was chosen).
4. "intent_evolution": describe how the goal shifted from the first action to the last one.
5. "consistent_preferences": describe the attributes the user repeatedly gravitated toward. Return an empty string when nothing repeats.

[OUTPUT FORMAT]
{ "cross_action_flow": String, "intent_evolution": String, "consistent_preferences": String }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON. /no_think"###;

    template
        .replace("{LANG}", lang)
        .replace("{RECORDS}", records_json)
}

// 🌟 [ANALYTIC QUERY PARSER] parse_commerce_query 의 para2graph + extract_time_intent_prompt
//    구조를 분석(Analytic) 도메인에 맞춰 하나로 압축한 질의 파서입니다.
//    ── 왜 별도 파서인가 ──
//     commerce 는 sale_price / tracking_number 처럼 '컬럼 조건' 을 뽑아야 하지만,
//     analytic 의 저장 축은 action / summary / relate 세 개의 자유 서술 문장뿐입니다.
//     따라서 수치 조건 추출은 불필요하고, 실제로 필요한 것은
//       ① 기간(time_intent / season_intent)
//       ② 이벤트 종류(click / hover / change / report)
//       ③ 의미 키워드
//     세 가지입니다. 기간은 LLM 값을 그대로 쓰지 않고 Rust 가 다시 epoch 로 확정합니다.
pub fn analytic_query_prompt(query: &str, time_context: &str, lang: &str) -> String {
    let mut time_keys: Vec<String> = crate::parsing::BIAS_DICT
        .get("time_filters")
        .and_then(|v| v.as_object())
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_else(|| vec![
            "today".to_string(), "yesterday".to_string(),
            "this_month".to_string(), "last_month".to_string(),
            "this_year".to_string(), "last_year".to_string(),
            "recently".to_string()
        ]);
    time_keys.push("".to_string());
    let time_arr_str = serde_json::to_string(&time_keys).unwrap_or_else(|_| "[]".to_string());

    let mut season_keys: Vec<String> = crate::parsing::BIAS_DICT
        .get("season_filters")
        .and_then(|v| v.as_object())
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_else(|| vec![
            "spring".to_string(), "summer".to_string(),
            "autumn".to_string(), "winter".to_string()
        ]);
    season_keys.push("".to_string());
    let season_arr_str = serde_json::to_string(&season_keys).unwrap_or_else(|_| "[]".to_string());

    let template = r###"[TASK]
Act as a deterministic semantic parser for a USER BEHAVIOUR LOG search engine.
Split the natural language query into a period, an event kind, and the semantic keywords.

[SYSTEM TIME & LOCALE CONTEXT]
{TIME_CONTEXT}

[AVAILABLE TIME INTENTS]
{TIME_ARRAY}

[AVAILABLE SEASON INTENTS]
{SEASON_ARRAY}

[AVAILABLE EVENT TYPES]
- "click"  : the user pressed / selected something
- "hover"  : the user lingered on something without pressing it
- "change" : the user typed a value, picked a select option, toggled a checkbox
- "report" : a synthesized behavioural report over several actions
- ""       : the query does not restrict the event kind

[QUERY]
{QUERY}

[RULES]
1. "time_intent" MUST be chosen from [AVAILABLE TIME INTENTS]. Return "" when the query contains NO explicit temporal word. Never guess a period from context.
2. "season_intent" MUST be chosen from [AVAILABLE SEASON INTENTS]. Return "" when no season word is printed. A clothing name is NOT a season word.
3. "event_types" is an array chosen from [AVAILABLE EVENT TYPES]. Return an empty array when the query does not restrict the kind.
4. "keywords" holds the meaning-bearing chunks of the query in the ORIGINAL language, with every temporal word removed. Never include verbs such as "show me", "find", "tell me".
5. "target" is one short sentence, in {LANG}, restating what behaviour the user wants to see. It is used as the semantic search sentence.
6. "original_text" is the query copied character for character.

[OUTPUT FORMAT]
{ "original_text": String, "time_intent": String, "season_intent": String, "event_types": [String], "keywords": [String], "target": String }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template
        .replace("{TIME_CONTEXT}", time_context)
        .replace("{TIME_ARRAY}", &time_arr_str)
        .replace("{SEASON_ARRAY}", &season_arr_str)
        .replace("{QUERY}", query)
        .replace("{LANG}", lang)
}

// 🌟 [ANALYTIC REPORT] 벡터 검색으로 회수한 '구조화된 행동 기록' 목록을 근거로
//    사용자 질의에 답하는 리포트를 작성합니다. 결과는 JSON 이 아니라 마크다운 본문입니다.
pub fn analytic_report_answer_prompt(
    query: &str,
    time_context: &str,
    scope: &str,
    records_json: &str,
    lang: &str,
) -> String {
    let template = r###"[TASK]
You are a User Behavior Analyst. Answer the user's question using ONLY the retrieved behaviour records below.

[SYSTEM TIME & LOCALE CONTEXT]
{TIME_CONTEXT}

[SEARCH SCOPE]
{SCOPE}

[USER QUESTION]
{QUERY}

[RETRIEVED BEHAVIOUR RECORDS]
{RECORDS}

[RULES]
1. Every sentence you write MUST be supported by a record above. If the records do not answer the question, say so plainly instead of inventing an answer.
2. Copy product names, option values, prices and numbers EXACTLY as printed in the records.
3. Represent users as User A, User B, User C ... Never print the raw address / hash of a user.
4. Write in {LANG}.
5. Structure the answer as:
   - one short headline sentence that directly answers the question
   - a bullet list of the concrete supporting actions (what, where, when)
   - one closing sentence on the pattern or the recommended follow-up
6. Do NOT output JSON. Output plain readable text (markdown bullets are allowed).

[ACTION] WRITE THE REPORT ONLY. NO PREAMBLE. NO CODE FENCE. /no_think"###;

    template
        .replace("{TIME_CONTEXT}", time_context)
        .replace("{SCOPE}", scope)
        .replace("{QUERY}", query)
        .replace("{RECORDS}", records_json)
        .replace("{LANG}", lang)
}

pub fn is_detail_prompt(page_type: &str, title: &str, lang: &str) -> String {
    let (list_hints, form_hints) = crate::parsing::get_layout_prompt_hints(page_type, lang);

    let template = r###"[TASK]
Analyze the provided PUG/HTML content from top to bottom.

[ENTITY CONTEXT: {TYPE}]
Language Context: {LANGUAGE}
You are evaluating a page managing this specific domain entity. Use this context to conceptually understand the abstract structures:
- has_form: A property configuration interface. It features a large overarching form dedicated to inputting or updating the specific attributes of ONE primary entity.{FORM_HINTS}
- has_list: A catalog or inventory interface dedicated to displaying, filtering, or batch-processing multiple DIFFERENT primary entities.{LIST_HINTS}

[FORCED DOCUMENT SCANNING LOGIC]
Read the entire document from top to bottom, applying the following strict filters and evaluations:

1. IGNORE:
   - Strictly ignore global navigation, menus, headers, footers, aside, search, filter.
2. TARGET:
   - Focus purely on the main data payload where "{TYPE}", or actual items are listed.
3. EVALUATE:
   - You MUST evaluate the concluding elements at the very bottom of the main content area first. Check for the following:
     A. Does the page terminate with dataset navigation (pagination, "next/prev") or bulk-action execution elements?
     B. Does the main data area consist of a repeating multi-entity grid?
     C. Does the main data area contain an extensive configuration/input form (inputs, textareas, image uploads, save buttons) for a single entity?

[SCHEMA DEFINITIONS]
- {TYPE}:
    - has_header: Boolean. True if the document contains a header.
    - title: String. Default '{TITLE}'.
    - has_footer: Boolean. True if the document contains a footer.
    - language: String. Default '{LANGUAGE}'.
    - has_list: Boolean. True if the document contains a multi-entity grid, OR if the bottom of main content area has dataset navigation/bulk controls.
    - has_form: Boolean. True if the main data payload is heavily composed of data entry fields (text, select, radio, file uploads) dedicated to creating or updating a single entity.

[OUTPUT FORMAT]
{ "{TYPE}": { "has_header": Boolean, "title": String, "has_footer": Boolean, "language": String, "has_list": Boolean, "has_form": Boolean } }

JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;
    
    template.replace("{TYPE}", page_type)
        .replace("{TITLE}", title)
        .replace("{LANGUAGE}", lang)
        .replace("{FORM_HINTS}", &form_hints)
        .replace("{LIST_HINTS}", &list_hints)
}

pub fn para2graph(language: &str) -> String {
    let template = r###"Translate and convert the given natural language search query into English, then segment it into the specified JSON dataset structure.

[DOCUMENT SCANNING & STRICT SEGMENTATION LOGIC]
1. EXACT COPY: Copy the full original input into 'original_text' without changing anything.
2. TRANSLATE & TAGGED PIPE PLANNING: Translate the query into English. In the 'segmented_plan' field, prefix every translated segment with its assigned type tag in brackets, separated by pipes ('|'). Structure it strictly as '[tag1] english chunk1 | [tag2] english chunk2'.
3. MAXIMAL GROUPING: Group all contiguous words belonging to the same type into a SINGLE English segment. DO NOT split subjects from their numeric conditions. Break the segment ONLY when the type logically shifts.
4. STRICT ARRAY MAPPING: For EVERY tagged English segment in 'segmented_plan', create exactly one object in the 'context' array sequentially.

[SCHEMA DEFINITIONS]
- original_text: String. The exact, unaltered full natural language input.
- segmented_plan: String. Translated English text with '[type] english text | ' format inserted strictly at type boundaries.
- context:
  - 'text': String. The translated English chunk.
  - 'language': String. Default '{LANG}'.
  - 'type': String. Choose one:
    * 'order': Intent to measure sales performance or direct transactions. Triggers: conversion rate, sales volume, checkout, payment, cancellation, refund. (RULE: If the context measures buying success or revenue, classify as 'order' even if the word 'product' or 'item' is present).
    * 'goods': Intent to describe product catalog data, exposure, or traffic metrics. Triggers: page views, clicks, physical attributes, stock limits, unit prices. (RULE: Focuses on item specifications and customer traffic before the actual purchase).
    * 'tracking': Intent to manage logistics and fulfillment. Triggers: shipment status, dispatch, delivery duration, courier information.
    * 'review': Intent to analyze the voice of the customer. Triggers: feedback, ratings, reviews, CS messages, complaints.
    * 'coupon': Intent to manage specific discount vouchers. Triggers: coupon codes, issuance limits, discount amounts applied via coupons.
    * 'event': Intent to manage marketing campaigns or analyze broad operational trends. Triggers: promotions, exhibitions, seasonal sales, overarching managerial analysis requests.
    * '': If none logically apply.

[OUTPUT FORMAT]
{ "original_text": "String", "segmented_plan": "String", "context": [...] }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;
    template.replace("{LANG}", language)
}

pub fn extract_time_intent_prompt(text: &str, time_context: &str, first_choice: &str, first_score: f32, alternatives: &[(String, f32)]) -> String {
    let mut time_keys: Vec<String> = crate::parsing::BIAS_DICT
        .get("time_filters")
        .and_then(|v| v.as_object())
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_else(|| vec!["today".to_string(), "yesterday".to_string(), "this_month".to_string(), "last_month".to_string(), "this_year".to_string(), "last_year".to_string(), "recently".to_string()]);
    
    // 시간 의도가 아닐 경우를 대비해 빈 문자열 선택지를 추가합니다.
    time_keys.push("".to_string());

    let time_arr_str = serde_json::to_string(&time_keys).unwrap_or_else(|_| "[]".to_string());

    let mut cands_str = format!("- \"{}\" (Vector Score: {:.4})\n", first_choice, first_score);
    for (prop, score) in alternatives.iter() {
        cands_str.push_str(&format!("- \"{}\" (Vector Score: {:.4})\n", prop, score));
    }

    let template = r###"[TASK]
Analyze the given text and extract the exact relative time intent based on the Current Time Context.
You MUST strictly choose ONLY from the provided array. If none logically apply, return "". Do not invent any other values.

[SYSTEM TIME & LOCALE CONTEXT]
{TIME_CONTEXT}

[AVAILABLE TIME INTENTS]
{TIME_ARRAY}

[TEXT TO ANALYZE]
Text: "{TEXT}"

[CANDIDATE INTENTS]
{CANDIDATES}

[INSTRUCTIONS]
1. STRICT RULE: You MUST return "" (empty string) if the Text DOES NOT explicitly contain temporal words. Do NOT guess or infer time based on context.
2. Evaluate all [CANDIDATE INTENTS] equally. If one of them matches the explicit text perfectly, return it.
3. If none of the candidates match, but the text explicitly mentions time, choose the best fit from [AVAILABLE TIME INTENTS]. Otherwise, return "".

[OUTPUT FORMAT]
{ "time_intent": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{TIME_CONTEXT}", time_context)
            .replace("{TIME_ARRAY}", &time_arr_str)
            .replace("{TEXT}", text)
            .replace("{CANDIDATES}", &cands_str)
}

pub fn extract_season_intent_prompt(text: &str, first_choice: &str, first_score: f32, alternatives: &[(String, f32)]) -> String {
    let mut season_keys: Vec<String> = crate::parsing::BIAS_DICT
        .get("season_filters")
        .and_then(|v| v.as_object())
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_else(|| vec!["spring".to_string(), "summer".to_string(), "autumn".to_string(), "winter".to_string()]);
    
    // 계절 의도가 아닐 경우를 대비해 빈 문자열 선택지를 추가합니다.
    season_keys.push("".to_string());

    let season_arr_str = serde_json::to_string(&season_keys).unwrap_or_else(|_| "[]".to_string());

    let mut cands_str = format!("- \"{}\" (Vector Score: {:.4})\n", first_choice, first_score);
    for (prop, score) in alternatives.iter() {
        cands_str.push_str(&format!("- \"{}\" (Vector Score: {:.4})\n", prop, score));
    }

    let template = r###"[TASK]
Analyze the given text and extract the exact seasonal intent.
You MUST strictly choose ONLY from the provided array. If none logically apply, return "". Do not invent any other values.

[AVAILABLE SEASON INTENTS]
{SEASON_ARRAY}

[TEXT TO ANALYZE]
Text: "{TEXT}"

[CANDIDATE INTENTS]
{CANDIDATES}

[INSTRUCTIONS]
1. STRICT RULE: You MUST return "" (empty string) if the Text DOES NOT explicitly contain season-related words. Do NOT guess the season just because it's a specific clothing or item name.
2. Evaluate all [CANDIDATE INTENTS] equally. If one of them matches the explicit text perfectly, return it.
3. If none of the candidates match, but the text explicitly mentions a season, choose the best fit from [AVAILABLE SEASON INTENTS]. Otherwise, return "".

[OUTPUT FORMAT]
{ "season_intent": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{SEASON_ARRAY}", &season_arr_str)
            .replace("{TEXT}", text)
            .replace("{CANDIDATES}", &cands_str)
}

pub fn extract_numeric_conditions(current: &str, seg_type: &str, metrics_json: &str, vector_guide: &str, time_context: &str, lang: &str, value_type: &str) -> String {
    let (deterministic_time, _) = crate::parsing::get_deterministic_time_guide(vector_guide, lang);
    
    let final_time_context = if !deterministic_time.is_empty() {
        format!("{}\n{}", time_context, deterministic_time)
    } else {
        time_context.to_string()
    };

    let template = r###"[Task]
Act as a deterministic semantic parser.
You must extract, transform, and normalize numeric and property conditions from the natural language input into the strictly defined JSON output format.

[SYSTEM TIME & LOCALE CONTEXT]
{TIME_CONTEXT}
- If explicit exact dates are mentioned in the text, use them.

[DATABASE METRICS CONTEXT]
Metrics: {METRICS}
- CRITICAL RULES FOR FUZZY ADJECTIVES (Translate from any language):
  * If the query implies "many", "often", "popular", "best", or "high" without a specific number, you MUST IGNORE the Vector Guide's operator and map the operator to 'top' and set percent_total to 20.
  * If the query implies "few", "rarely", "unpopular", "worst", or "low" without a specific number, you MUST IGNORE the Vector Guide's operator and map the operator to 'bottom' and set percent_total to 20.
  * You MUST use the Metrics data to calculate the exact absolute threshold for these percentiles.

[VECTOR MATCHING GUIDE (HINT)]
The system has pre-calculated vector similarities for properties, operators, and metric types, including 1st choices and Alternatives.
- If the 1st choice operator/metric makes semantic sense, use it.
- If the 1st choice is wrong, consider the Alternatives provided.
- Metric Type gives a crucial hint about the data (date, time, price, discount, quantity, ratio). 
- If Metric Type is 'ratio', extract a percentage logic. If 'date', extract a date logic, etc.
- TEMPORAL & SEASON CORRECTION RULES:
  1. Vectors often hallucinate seasons. If the text explicitly contains a season word, IGNORE the Vector Guide's Season Intent and select the exact season yourself from the [LOCALE CALENDAR REFERENCE].
  2. If a Season is detected, check the Time Intent (or explicit time text):
     - If Time Intent implies the past, map the season to the PREVIOUS year's dates.
     - If Time Intent implies the present, map the season to the CURRENT year's dates.
     - Output BOTH 'started_at' (gte) and 'expired_at' (lte) to form a date range.
{GUIDE}

[SCHEMA DEFINITION]
Extract the following numeric/property conditions if semantically present in the text:
condition:
  - property: String.
  - is_percent: Boolean.
  - operator: String. 'gt' | 'gte' | 'lt' | 'lte' | 'eq' | 'contains' | 'top' | 'bottom'
  - percent_total: Number.
  - value: {VALUE_TYPE}.

[CURRENT CHUNK TO ANALYZE]
{CURRENT}

[OUTPUT FORMAT]
{ "condition": condition }

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{CURRENT}", current)
            .replace("{TYPE}", seg_type)
            .replace("{METRICS}", metrics_json)
            .replace("{GUIDE}", vector_guide)
            .replace("{TIME_CONTEXT}", &final_time_context)
            .replace("{VALUE_TYPE}", value_type)
}
pub fn extract_status_intent_prompt(current_text: &str, seg_type: &str, first_choice: &str, first_score: f32, alternatives: &[(String, f32)]) -> String {
    let status_options = match seg_type {
        "tracking" => r#"* 'draft': Shipment preparation or pending pickup.
* 'progress': Currently in transit or out for delivery.
* 'return': Returning to sender.
* 'complete': Successfully delivered to the recipient.
* '': If none logically apply."#,

        "goods" => r#"* 'draft': Product is being created, not yet published.
* 'show': Visible and available for sale on storefront.
* 'hide': Hidden from the storefront.
* 'progress': Currently being restocked or updated.
* 'stop': Sales temporarily suspended.
* 'cancel': Product discontinued or cancelled.
* 'refund': Related to refunded inventory.
* 'return': Related to returned inventory.
* 'exchange': Related to exchanged inventory.
* 'expire': Product expired.
* 'complete': Completely sold out or finished lifecycle.
* '': If none logically apply."#,

        "order" => r#"* 'draft': Pending payment or in cart.
* 'progress': Order processing or preparing for shipment.
* 'stop': Order on hold.
* 'cancel': Order cancelled before fulfillment.
* 'refund': Payment refunded.
* 'return': Items returned by customer.
* 'exchange': Items being exchanged.
* 'expire': Payment window expired.
* 'complete': Order fully fulfilled and closed.
* '': If none logically apply."#,

        "coupon" | "event" => r#"* 'show': Visible to customers.
* 'progress': Currently active and running.
* 'hide': Hidden from customers.
* 'stop': Temporarily paused.
* 'cancel': Terminated early.
* 'expire': Passed its expiration date.
* 'complete': Successfully finished its run.
* '': If none logically apply."#,

        "review" => r#"* 'progress': Under moderation or pending approval.
* 'stop': Blocked or suspended review.
* 'cancel': Deleted or withdrawn by user.
* 'refund': Associated with a refunded order.
* 'return': Associated with a returned order.
* 'exchange': Associated with an exchanged order.
* 'expire': Review period expired.
* 'complete': Published and visible.
* '': If none logically apply."#,

        _ => r#"* 'show': Visible state.
* 'progress': Active/Processing state.
* 'remove': Deleted state.
* 'hide': Hidden state.
* 'stop': Paused/Stopped state.
* 'cancel': Cancelled state.
* 'refund': Refunded state.
* 'return': Returned state.
* 'exchange': Exchanged state.
* 'expire': Expired state.
* 'complete': Finished/Completed state.
* '': If none logically apply."#,
    };

    let mut cands_str = format!("- \"{}\" (Vector Score: {:.4})\n", first_choice, first_score);
    for (prop, score) in alternatives.iter() {
        cands_str.push_str(&format!("- \"{}\" (Vector Score: {:.4})\n", prop, score));
    }

    let template = r###"[TASK]
Analyze the given text and extract the exact semantic intent for status.
You MUST strictly choose ONLY from the provided array. If none logically apply, return "". Do not invent any other values.

[SCHEMA DEFINITIONS]
- status: String. Choose one:
{STATUS_OPTIONS}

[TEXT TO ANALYZE]
Text: "{TEXT}"

[CANDIDATE INTENTS]
{CANDIDATES}

[INSTRUCTIONS]
1. Evaluate all [CANDIDATE INTENTS] equally. If one of them is semantically correct for this text, return it.
2. If none of the candidates match, but the text explicitly dictates a status state, choose a valid intent from the [SCHEMA DEFINITIONS] array.
3. Otherwise, return "".

[OUTPUT FORMAT]
{ "status": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template
        .replace("{STATUS_OPTIONS}", status_options)
        .replace("{TEXT}", current_text)
        .replace("{CANDIDATES}", &cands_str)
}

// 🌟 [추가] scheduler.rs 전용 3개 인자 레거시 함수
pub fn extract_status_intent_legacy_prompt(current_text: &str, seg_type: &str, vector_guide: &str) -> String {
    let status_options = match seg_type {
        "tracking" => r#"* 'draft': Shipment preparation or pending pickup.
* 'progress': Currently in transit or out for delivery.
* 'return': Returning to sender.
* 'complete': Successfully delivered to the recipient.
* '': If none logically apply."#,

        "goods" => r#"* 'draft': Product is being created, not yet published.
* 'show': Visible and available for sale on storefront.
* 'hide': Hidden from the storefront.
* 'progress': Currently being restocked or updated.
* 'stop': Sales temporarily suspended.
* 'cancel': Product discontinued or cancelled.
* 'refund': Related to refunded inventory.
* 'return': Related to returned inventory.
* 'exchange': Related to exchanged inventory.
* 'expire': Product expired.
* 'complete': Completely sold out or finished lifecycle.
* '': If none logically apply."#,

        "order" => r#"* 'draft': Pending payment or in cart.
* 'progress': Order processing or preparing for shipment.
* 'stop': Order on hold.
* 'cancel': Order cancelled before fulfillment.
* 'refund': Payment refunded.
* 'return': Items returned by customer.
* 'exchange': Items being exchanged.
* 'expire': Payment window expired.
* 'complete': Order fully fulfilled and closed.
* '': If none logically apply."#,

        "coupon" | "event" => r#"* 'show': Visible to customers.
* 'progress': Currently active and running.
* 'hide': Hidden from customers.
* 'stop': Temporarily paused.
* 'cancel': Terminated early.
* 'expire': Passed its expiration date.
* 'complete': Successfully finished its run.
* '': If none logically apply."#,

        "review" => r#"* 'progress': Under moderation or pending approval.
* 'stop': Blocked or suspended review.
* 'cancel': Deleted or withdrawn by user.
* 'refund': Associated with a refunded order.
* 'return': Associated with a returned order.
* 'exchange': Associated with an exchanged order.
* 'expire': Review period expired.
* 'complete': Published and visible.
* '': If none logically apply."#,

        _ => r#"* 'show': Visible state.
* 'progress': Active/Processing state.
* 'remove': Deleted state.
* 'hide': Hidden state.
* 'stop': Paused/Stopped state.
* 'cancel': Cancelled state.
* 'refund': Refunded state.
* 'return': Returned state.
* 'exchange': Exchanged state.
* 'expire': Expired state.
* 'complete': Finished/Completed state.
* '': If none logically apply."#,
    };

    let template = r###"[TASK]
Analyze the given text and extract the exact semantic intent for status.
You MUST strictly choose ONLY from the provided array and use the Vector Matching Guide. Do not invent any other values.

[VECTOR MATCHING GUIDE]
{VECTOR_GUIDE}

[TEXT TO ANALYZE]
{TEXT}

[SCHEMA DEFINITIONS]
- status: String. Choose one:
{STATUS_OPTIONS}
  * '': If none logically apply.

[OUTPUT FORMAT]
{ "status": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template
        .replace("{STATUS_OPTIONS}", status_options)
        .replace("{TEXT}", current_text)
        .replace("{VECTOR_GUIDE}", vector_guide)
}

pub fn extract_substantial_intent_prompt(current_text: &str, first_choice: &str, first_score: f32, alternatives: &[(String, f32)]) -> String {
    let mut cands_str = format!("- \"{}\" (Vector Score: {:.4})\n", first_choice, first_score);
    for (prop, score) in alternatives.iter() {
        cands_str.push_str(&format!("- \"{}\" (Vector Score: {:.4})\n", prop, score));
    }

    let template = r###"[TASK]
Analyze the given text and extract the exact semantic intent for substantial.
You MUST strictly choose ONLY from the provided array. If none logically apply, return "". Do not invent any other values.

[SCHEMA DEFINITIONS]
- substantial: String. Choose one:
  * 'size': Physical dimensions or volume.
  * 'weight': Mass or heaviness.
  * 'shipping_fee': Cost of delivery.
  * 'shipping_duration': Time taken for delivery.
  * 'sale_price': Final selling price to the customer.
  * 'supply_price': Wholesale or original cost.
  * 'low_stock_threshold': Minimum inventory alert level.
  * 'discount': Amount or percentage of price reduction.
  * 'min_order_amount': Minimum spend required to trigger an action.
  * 'max_discount_amount': Maximum cap for a discount.
  * 'usage_limit': Maximum number of times usable globally.
  * 'usage_per': Maximum number of times usable per user.
  * '': If none logically apply.

[TEXT TO ANALYZE]
Text: "{TEXT}"

[CANDIDATE INTENTS]
{CANDIDATES}

[INSTRUCTIONS]
1. Evaluate all [CANDIDATE INTENTS] equally. If one of them is semantically correct for this text, return it.
2. If none of the candidates match, but the text explicitly dictates a substantial state, choose a valid intent from the [SCHEMA DEFINITIONS] array.
3. Otherwise, return "".

[OUTPUT FORMAT]
{ "substantial": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{TEXT}", current_text)
            .replace("{CANDIDATES}", &cands_str)
}

pub fn extract_find_intent_prompt(current_text: &str, first_choice: &str, first_score: f32, alternatives: &[(String, f32)]) -> String {
    let mut cands_str = format!("- \"{}\" (Vector Score: {:.4})\n", first_choice, first_score);
    for (prop, score) in alternatives.iter() {
        cands_str.push_str(&format!("- \"{}\" (Vector Score: {:.4})\n", prop, score));
    }

    let template = r###"[TASK]
Analyze the given text and extract the exact semantic intent for find.
You MUST strictly choose ONLY from the provided array. If none logically apply, return "". Do not invent any other values.

[SCHEMA DEFINITIONS]
- find: String. Choose one:
  * 'many': High quantity, count, or volume.
  * 'few': Low quantity, count, or volume.
  * 'much': High financial value, price, or amount.
  * 'little': Low financial value, price, or amount.
  * 'heavy': High physical weight.
  * 'light': Low physical weight.
  * '': If none logically apply.

[TEXT TO ANALYZE]
Text: "{TEXT}"

[CANDIDATE INTENTS]
{CANDIDATES}

[INSTRUCTIONS]
1. Evaluate all [CANDIDATE INTENTS] equally. If one of them is semantically correct for this text, return it.
2. If none of the candidates match, but the text explicitly dictates a find state, choose a valid intent from the [SCHEMA DEFINITIONS] array.
3. Otherwise, return "".

[OUTPUT FORMAT]
{ "find": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{TEXT}", current_text)
            .replace("{CANDIDATES}", &cands_str)
}



pub fn extract_single_field_prompt(page_type: &str, field_name: &str, field_desc: &str, language: &str, metadata: &str, target_data: &str) -> String {
    let mut dynamic_output_keys = String::new();
    for key in field_name.split(',') {
        dynamic_output_keys.push_str(&format!("  \"{}\": <literal substring of [TARGET DATA], or null>,\n", key.trim()));
    }
    let dynamic_output_keys = dynamic_output_keys.trim_end_matches(",\n");

    // 🌟 기존 7번 SHAPE RULES 전체는 detect_field_format / value_matches_format 의
    //    사전 형식 게이트와 사후 [FORMAT REJECT] 검증이 코드로 강제하므로 프롬프트에서 제거했습니다.
    let template = r###"[TASK]
Copy ONE property set from [TARGET DATA]. You are a copier, not a writer.

[CONTEXT]
Page Type: {TYPE} / Output Language: {LANGUAGE}
Column labels (LABELS, never answers): {METADATA}

[SCHEMA DEFINITIONS]
{FIELDS}

[TARGET DATA]
{TARGET_DATA}

[RULES]
1. The answer MUST be an exact literal substring of [TARGET DATA]. Never translate, reformat, round, or re-type it.
2. Never answer with a column label, a format placeholder ("yyyy-MM-ddThh:mm:ss", "string", "...", "N/A"), or a value listed under [ALREADY CLAIMED VALUES].
3. NEVER answer with an HTML/PUG tag name.
4. If [VECTOR MATCH RESULT], [LINK CANDIDATES] or [DATE CANDIDATES] is given, the answer MUST come from it.
5. If nothing in [TARGET DATA] fits the schema, return null. null is correct data; a wrong value is corrupted data.

[OUTPUT FORMAT]
{ {DYNAMIC_KEYS} }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON. /no_think"###;

    template.replace("{TYPE}", page_type)
            .replace("{LANGUAGE}", language)
            .replace("{METADATA}", metadata)
            .replace("{FIELDS}", field_desc)
            .replace("{TARGET_DATA}", target_data)
            .replace("{DYNAMIC_KEYS}", &dynamic_output_keys)
}

// 🌟 [NEW] insight / summary / analysis 계열 합성 필드 전용 프롬프트.
// 기존 extract_single_field_prompt 는 "리터럴 복사" 지시라, 합성 필드에 쓰면
// LLM 이 어쩔 수 없이 셀 하나("89000", "본사")를 그대로 뱉게 됩니다.
// 또한 원문 언어(doc_lang)를 명시적으로 전달하여, 한국어 문서에서도
// 고유명사/코드/숫자는 원문 그대로 보존하고 문장만 영어로 합성하도록 강제합니다.
pub fn extract_synthesis_field_prompt(page_type: &str, field_name: &str, field_desc: &str, doc_lang: &str, source_data: &str) -> String {
    let mut dynamic_output_keys = String::new();
    for key in field_name.split(',') {
        dynamic_output_keys.push_str(&format!("  \"{}\": <one sentence written by you, or null>,\n", key.trim()));
    }
    let dynamic_output_keys = dynamic_output_keys.trim_end_matches(",\n");

    let template = r###"[TASK]
Write ONE analytic sentence for the requested field. This field is a SUMMARY you compose, not a value you copy.

[CONTEXT]
Page Type: {TYPE}
Source Document Language: {DOC_LANG}

[SCHEMA DEFINITIONS]
{FIELDS}

[SOURCE DATA]
{SOURCE_DATA}

[WRITING RULES]
1. Read ALL of [SOURCE DATA] before writing. NEVER answer with a single cell value such as a bare number, a status word, a person name, a branch name, or a column label.
2. The sentence MUST combine at least two different facts taken from [SOURCE DATA].
3. Do NOT invent facts. Only restate and connect what is present in [SOURCE DATA].
4. Keep every proper noun, product name, code, identifier, and number EXACTLY as written in [SOURCE DATA]. Never translate, transliterate, or reformat them, whatever the source language is.
5. Write the connecting sentence in English, while keeping the copied literals in their original script.
6. If [SOURCE DATA] has no usable content for this field, return null. A null summary is correct; a fabricated one is corrupted data.

[OUTPUT FORMAT]
{ {DYNAMIC_KEYS} }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON. /no_think"###;

    template.replace("{TYPE}", page_type)
            .replace("{DOC_LANG}", doc_lang)
            .replace("{FIELDS}", field_desc)
            .replace("{SOURCE_DATA}", source_data)
            .replace("{DYNAMIC_KEYS}", &dynamic_output_keys)
}

// 🌟 [NEW] 상태(status) 컨트롤의 CSS selector 를 찾는 전용 프롬프트.
// 코사인으로 '옵션 집합이 생애주기 상태인지'를 판정했는데 마진이 부족해 애매할 때만 호출됩니다.
// 값을 뱉게 하지 않고 '선택자'만 뱉게 하여, 실제 값은 우리가 selected 옵션에서 결정론적으로 읽습니다.
pub fn extract_status_selector_prompt(page_type: &str, lang: &str, candidates_json: &str) -> String {
    let template = r###"[TASK]
Pick the ONE form control that holds the CURRENT LIFECYCLE STATE of this single {TYPE} record.

[CONTEXT]
Page Type: {TYPE}
Document Language: {LANG}

[CANDIDATES]
Every entry below is a real <select> element found in this document.
- "selector": its exact CSS selector
- "role"    : the words attached to the control (its name / id / row header)
- "options" : every option label inside it
{CANDIDATES}

[RULES]
1. The correct control lists MUTUALLY EXCLUSIVE LIFECYCLE STATES of ONE record.
   Examples of lifecycle states: pending, preparing, in transit, delivered, completed,
   cancelled, returned, exchanged, refunded, expired, on hold.
2. It is NOT a control that lists organizations or catalogue values:
   couriers, delivery companies, banks, account numbers, card issuers, payment gateways,
   categories, brands, countries, quantities, dates, or addresses.
3. Judge by the OPTION SET, not by the control name. A control whose options are company
   names is never the state control, however its name reads.
4. "selector" MUST be copied character for character from [CANDIDATES]. Never invent one.
5. If no candidate lists lifecycle states, return null. null is correct data; a wrong
   selector silently corrupts every record.

[OUTPUT FORMAT]
{ "status_selector": <one selector copied verbatim from [CANDIDATES], or null> }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON. /no_think"###;

    template.replace("{TYPE}", page_type)
            .replace("{LANG}", lang)
            .replace("{CANDIDATES}", candidates_json)
}

pub fn verify_property_mapping_prompt(text: &str, property: &str) -> String {
    let template = r###"[TASK]
Given the text '{TEXT}' and the current property '{PROPERTY}', suggest the most accurate property name(s) from the schema.
If the current property is already correct, just return it in the array.

[OUTPUT FORMAT]
{"suggested_properties": ["String", "String"]}

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{TEXT}", text)
            .replace("{PROPERTY}", property)
}

pub fn verify_property_with_alternatives_prompt(
    text: &str,
    first_choice: &str,
    first_score: f32,
    alternatives: &[(String, f32)],
) -> String {
    let mut cands_str = format!("- \"{}\" (Vector Score: {:.4})\n", first_choice, first_score);
    for (prop, score) in alternatives.iter() {
        cands_str.push_str(&format!("- \"{}\" (Vector Score: {:.4})\n", prop, score));
    }

    let template = r###"[TASK]
Text: "{TEXT}"

[CANDIDATE PROPERTIES]
{CANDIDATES}

Instructions:
1. Evaluate all [CANDIDATE PROPERTIES] equally based on the text.
2. Return the best-fitting property as "suggested_property".
3. If none of the candidates are correct, suggest a completely different property from the schema.

[OUTPUT FORMAT]
{ "suggested_property": String }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{TEXT}", text)
            .replace("{CANDIDATES}", &cands_str)
}

pub fn verify_operator_mapping_prompt(text: &str, property: &str, operator: &str) -> String {
    let valid_ops: Vec<String> = crate::parsing::BIAS_DICT
        .get("operators")
        .and_then(|v| v.as_object())
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_else(|| vec![
            "eq".to_string(), "neq".to_string(), "gt".to_string(), "gte".to_string(), 
            "lt".to_string(), "lte".to_string(), "contains".to_string(), 
            "not_contains".to_string(), "top".to_string(), "bottom".to_string()
        ]);
    
    let valid_ops_str = valid_ops.join(", ");

    let template = r###"[TASK]
Given the text "{TEXT}", the property '{PROPERTY}' currently has the operator '{OPERATOR}'.
Suggest the most correct operator based on the context of the text.
If the current operator is already correct, just return it.
Valid operators: {VALID_OPS}

[OUTPUT FORMAT]
{ "suggested_operator": String }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{TEXT}", text)
            .replace("{PROPERTY}", property)
            .replace("{OPERATOR}", operator)
            .replace("{VALID_OPS}", &valid_ops_str)
}

pub fn verify_category_with_alternatives_prompt(
    text: &str,
    first_choice: &str,
    first_score: f32,
    alternatives: &[String],
) -> String {
    let mut cands_str = format!("- \"{}\" (Vector Score: {:.4})\n", first_choice, first_score);
    for prop in alternatives.iter() {
        if prop != first_choice {
            cands_str.push_str(&format!("- \"{}\"\n", prop));
        }
    }

    let template = r###"[TASK]
Text: "{TEXT}"

[CANDIDATE CATEGORIES]
{CANDIDATES}

Instructions:
1. Evaluate all [CANDIDATE CATEGORIES] equally based on the text.
2. Choose the category that best matches the text context.
3. Return the best-fitting category as "suggested_category".

[OUTPUT FORMAT]
{ "suggested_category": String }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{TEXT}", text)
            .replace("{CANDIDATES}", &cands_str)
}

pub fn transliteration_prompt(source_value: &str, target_language: &str) -> String {
    // 🌟 [SPECIAL CHAR PRE-STRIPPED] 호출부(build_transliteration_prompt)에서
    //    이미 특수문자가 공백으로 치환된 source_value 가 전달됩니다.
    //    여기서도 방어적으로 한 번 더 공백 정규화를 수행합니다.
    let cleaned = source_value
        .split_whitespace()
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    // 🌟 [WORD-KEY JSON] 각 단어를 독립 키로 갖는 객체 구조를 생성합니다.
    //    LLM 이 단어 단위로 독립 음차하므로 중간 맥락 끊김이 발생하지 않습니다.
    // 🌟 [FULL SOURCE FIRST] 최초 첫 번째 키로 특수문자 제거된 전체 SOURCE 를 배치하여
    //    LLM 이 전체 문맥을 먼저 파악한 뒤 단어별 음차를 수행하도록 합니다.
    let mut word_keys: Vec<String> = Vec::with_capacity(words.len() + 1);
    // word_keys.push(format!("\"{}\": String", cleaned));
    for w in &words {
        word_keys.push(format!("\"{}\": String", w));
    }
    let transliteration_obj = format!("{{ {} }}", word_keys.join(", "));
    let template = r###"[TASK]
You are a sound-based respelling engine.
write how that word sounds in the [TARGET LANGUAGE] writing system.

[TARGET LANGUAGE]
{TARGET_LANGUAGE}

[RULES]
- Digits inside a word must be copied exactly as they appear.

[OUTPUT FORMAT]
{ "language": "{TARGET_LANGUAGE}", "transcription": { "{SOURCE}" : String }, "transliteration": {TRANSLITERATION_OBJ} }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###;
    template.replace("{SOURCE}", &cleaned)
        .replace("{TARGET_LANGUAGE}", target_language)
        .replace("{TRANSLITERATION_OBJ}", &transliteration_obj)
}
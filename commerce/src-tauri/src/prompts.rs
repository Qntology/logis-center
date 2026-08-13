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

pub fn get_trade_doc_classification_prompt() -> String {
    r###"Classify document type. Choose strictly from: PI, CI, BL, AWB, PL, CO, LC, TRACKING, Unknown. 
Return JSON exactly like: {"doc_type": "BL"}
NO EXPLANATION."###.to_string()
}

pub fn get_trade_category_schema(category: &str, _doc_type: &str) -> String {
    let schema = match category {
        "header" => r#"{
  "document_type": "CLASSIFIED TYPE {String}",
  "document_number": "Primary Identifier (B/L No, Invoice No) {String}",
  "issue_date": "Date of Creation (YYYY-MM-DD) {String}",
  "reference_number": "Export Ref, Booking No, PO No {String}"
}"#,
        "parties" => r#"{
  "supplier_name": "Shipper, Seller, Exporter {String}",
  "supplier_address": "Address of Supplier {String}",
  "buyer_name": "Consignee, Buyer, Importer {String}",
  "buyer_address": "Address of Buyer {String}",
  "notify_party_name": "Notify Party Name {String}"
}"#,
        "logistics" => r#"{
  "vehicle_name": "Vessel Name, Flight No {String}",
  "voyage_number": "Voyage No {String}",
  "location_port_of_loading": "POL, Airport of Departure {String}",
  "location_port_of_discharge": "POD, Airport of Destination {String}"
}"#,
        "conditions" => r#"{
  "incoterms_code": "FOB, CIF, EXW, DDP {String}",
  "freight_payment_term": "Freight Prepaid, Freight Collect {String}"
}"#,
        "financials" => r#"{
  "currency_code": "Currency Symbol (USD, EUR) {String}",
  "amount_total": "Grand Total Amount {Number}"
}"#,
        "cargo" => r#"{
  "package_count": "Total Quantity (NOT Money) {Number}",
  "weight_gross": "Total Gross Weight {Number}",
  "volume_measurement": "Total Volume (CBM) {Number}",
  "marks_and_numbers": "Marks & Nos {String}"
}"#,
        "items" => r#"[ {
  "description": "Description of Goods {String}",
  "quantity": "Line Item Quantity {Number}",
  "hs_code": "HS Code / Tariff No {String}"
} ]"#,
        "containers" => r#"[ {
  "container_number": "Container No (4 char + 7 digit) {String}",
  "seal_number": "Seal No {String}",
  "type_description": "Type (20GP, 40HC) {String}"
} ]"#,
        _ => "{}"
    };

    format!("RULES: Follow comments strictly. Output JSON ONLY. MISSION: Extract data for category '{}'.\nSCHEMA:\n{}", category.to_uppercase(), schema)
}

pub fn extract_shipping_conditions(query: &str, language: &str) -> String {
    let template = r###"Task: Act as a deterministic shipping and trade logistics semantic parser.
Extract the logistics filters from the natural language query into the JSON format.

[SCHEMA DEFINITION]
Extract the following tracking/trade properties if semantically present in the text:
- "no": Tracking number, B/L number, Invoice number.
- "status": Shipping status (draft, progress, return, complete, error).
- "vessel": Vessel name, Flight No, or Carrier.
- "pol": Port of Loading, Origin, Departure point.
- "pod": Port of Discharge, Destination, Arrival point.
- "sender_name": Shipper, Seller, or Exporter name.
- "recipient_name": Consignee, Buyer, or Importer name.
- "incoterms": Incoterms (e.g., FOB, CIF, EXW).
- "weight": Cargo or gross weight.
- "amount": Total financial amount or price.

[TRANSFORMATION LOGIC]
For EVERY extracted field, wrap it in an operator object:
{ "operator": "eq" | "gt" | "lt" | "gte" | "lte" | "contains", "value": <extracted_value> }
- Use "contains" for text fields, names, ports, vessels.
- Use "eq" for strict identifiers or status.

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
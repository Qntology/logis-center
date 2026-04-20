use scraper::{Html, Node, Selector};
use ego_tree::NodeRef;
use regex::Regex;

#[derive(PartialEq, Clone, Copy)]
pub enum PugMode {
    StructureOnly,
    FullContent,
}

pub fn sanitize_llm_input(text: &str) -> String {
    // 1. Filter out non-printable and non-ASCII/Korean characters if they are broken
    // But we want to keep Korean. Let's filter out known problematic control codes.
    let cleaned: String = text.chars()
        .filter(|c| {
            let u = *c as u32;
            // Keep: standard ASCII, Korean Hangul Jamo/Syllables, Common Punctuation
            (u >= 32 && u <= 126) || // Basic ASCII
            (u >= 0xAC00 && u <= 0xD7A3) || // Hangul Syllables
            (u >= 0x1100 && u <= 0x11FF) || // Hangul Jamo
            (u >= 0x3130 && u <= 0x318F) || // Hangul Compatibility Jamo
            u == 10 || u == 13 || u == 9     // \n, \r, \t
        })
        .collect();

    // 2. Prevent internal special tokens from being interpreted
    cleaned.replace("<|", "< |").replace("|>", "| >")
}

pub fn pre_clean_html(html: &str) -> String {
    // 1. 주석 제거
    let re_comm = Regex::new(r"(?s)").unwrap();
    let html = re_comm.replace_all(html, "");

    // 2. 불필요한 태그 및 내부 콘텐츠 통째로 제거
    // JS filter list: script, style, link, noscript, iframe
    let re_tags = Regex::new(r"(?is)<(script|style|link|noscript|iframe)\b[^>]*>.*?</(script|style|link|noscript|iframe)>").unwrap();
    let html = re_tags.replace_all(&html, "");

    // 3. 단일 태그 및 불필요한 메타 태그 정리 (input은 제외하고 보존)
    let re_single = Regex::new(r"(?is)<(meta|link|br|hr|source)\b[^>]*>").unwrap();
    let clean = re_single.replace_all(&html, "");

    // 4. 연속된 줄바꿈 및 불필요한 공백 제거
    let re_whitespace = Regex::new(r"(?m)^\s*\n").unwrap();
    let clean = re_whitespace.replace_all(&clean, "");
    
    clean.trim().to_string()
}

pub fn convert_doc_to_clean_pug(document: &Html, mode: PugMode) -> String {
    let mut pug_output = String::new();
    pug_output.reserve(1024 * 50);
    
    // Discovery 모드(StructureOnly)일 때는 body 내부만 집중
    let mut found_body = false;
    for child in document.tree.root().children() {
        if let Some(element) = child.value().as_element() {
            if element.name() == "body" {
                generate_pug_lines(child, 0, &mut pug_output, &mode, &mut None);
                found_body = true;
                break;
            }
        }
    }
    if !found_body {
        for child in document.tree.root().children() {
            generate_pug_lines(child, 0, &mut pug_output, &mode, &mut None);
        }
    }
    sanitize_llm_input(&pug_output)
}
    
pub fn convert_to_clean_pug(html: &str, mode: PugMode) -> String {
    let document = Html::parse_document(html);
    convert_doc_to_clean_pug(&document, mode)
}

pub fn convert_doc_to_clean_pug_selector(document: &Html, selector_str: &str, mode: PugMode) -> String {
    let selector = match Selector::parse(selector_str) {
        Ok(s) => s,
        Err(_) => return String::new(),
    };
    let mut pug_output = String::new();
    pug_output.reserve(1024 * 5);
    let mut ctx = None;
    for node in document.tree.root().descendants() {
        if let Some(element_ref) = scraper::ElementRef::wrap(node) {
            if selector.matches(&element_ref) {
                 generate_pug_lines(node, 0, &mut pug_output, &mode, &mut ctx);
                 break;
            }
        }
    }
    pug_output
}

pub fn convert_to_clean_pug_selector(html: &str, selector_str: &str, mode: PugMode) -> String {
    let document = Html::parse_document(html);
    convert_doc_to_clean_pug_selector(&document, selector_str, mode)
}

#[derive(Default, Clone)]
pub struct TableContext {
    pub headers: Vec<Vec<String>>, // Row -> Col -> Title
    pub current_row_idx: usize,
    pub current_col_idx: usize,
    pub is_in_tbody: bool,
}

pub fn generate_pug_lines(node: NodeRef<scraper::Node>, indent_level: usize, output: &mut String, mode: &PugMode, ctx: &mut Option<TableContext>) {
    if indent_level > 50 { return; }
    let indent = "    ".repeat(indent_level);
    
    match node.value() {
        Node::Element(element) => {
            let tag_name = element.name().to_lowercase();

            // --- base64 이미지를 포함하는 img 태그 제외 ---
            if tag_name == "img" {
                if let Some(src) = element.attr("src") {
                    if src.contains("base64") {
                        return;
                    }
                }
            }

            // 불필요한 태그들을 만나면 건너뛰기
            if ["script", "style", "link", "noscript", "iframe"].contains(&tag_name.as_str()) {
                return;
            }

            // Context Management
            if tag_name == "tbody" { if let Some(c) = ctx.as_mut() { c.is_in_tbody = true; c.current_row_idx = 0; } }
            if tag_name == "tr" { if let Some(c) = ctx.as_mut() { c.current_col_idx = 0; } }

            // --- 허용된 속성만 Pug 문법으로 변환 ---
            let mut other_attributes = Vec::new();

            // ID 속성 처리
            if let Some(id) = element.id() {
                other_attributes.push(format!("id=\"{}\"", id));
            }

            // Class 속성 처리
            if let Some(classes) = element.attr("class") {
                if !classes.is_empty() {
                    other_attributes.push(format!("class=\"{}\"", classes));
                }
            }

            // Inject alt from headers for tbody cells
            if tag_name == "td" || tag_name == "th" {
                if let Some(c) = ctx.as_mut() {
                    if c.is_in_tbody && !c.headers.is_empty() {
                        let h_row = &c.headers[c.current_row_idx % c.headers.len()];
                        if let Some(title) = h_row.get(c.current_col_idx) {
                            if !title.is_empty() {
                                other_attributes.push(format!("alt=\"{}\"", title.replace("\"", "'")));
                            }
                        }
                    }
                }
            }

            // 필수 속성 정의
            let always_include = [
                "src", "href", "type", "name", "value", "placeholder", 
                "checked", "selected", "disabled", "readonly", "rows", "cols"
            ];

            for (name, value) in element.attrs() {
                if name == "id" || name == "class" || name == "alt" { continue; }

                if name.starts_with("data-") || always_include.contains(&name) {
                    if ["checked", "selected", "disabled", "readonly"].contains(&name) && (value.is_empty() || value == name) {
                        other_attributes.push(name.to_string());
                    } else if !value.is_empty() {
                        let safe_value = value.replace("\"", "'");
                        other_attributes.push(format!("{}=\"{}\"", name, safe_value));
                    }
                }
            }

            let mut attributes_string = String::new();
            if !other_attributes.is_empty() {
                attributes_string.push_str(&format!("[{}]", other_attributes.join(" ")));
            }

            // div 축약 로직 (JS Parity)
            let mut current_node = node;
            while let Some(current_el) = current_node.value().as_element() {
                if current_el.name().to_lowercase() != "div" { break; }
                
                let valid_children: Vec<_> = current_node.children().filter(|n| {
                    match n.value() {
                        Node::Element(_) => true,
                        Node::Text(t) => !t.trim().is_empty(),
                        _ => false
                    }
                }).collect();

                if valid_children.len() == 1 {
                    if let Some(child_el) = valid_children[0].value().as_element() {
                        if child_el.name().to_lowercase() == "div" {
                            current_node = valid_children[0];
                            continue;
                        }
                    }
                }
                break;
            }

            output.push_str(&format!("{}{}{}\n", indent, tag_name, attributes_string));

            if tag_name == "textarea" {
                let mut text_content = String::new();
                for child in current_node.children() {
                    if let Node::Text(t) = child.value() { text_content.push_str(t); }
                }
                if !text_content.trim().is_empty() {
                    for line in text_content.lines() {
                        let trimmed = line.trim();
                        if !trimmed.is_empty() {
                            output.push_str(&format!("{}    | {}\n", indent, trimmed));
                        }
                    }
                }
            } else {
                for child in current_node.children() {
                    generate_pug_lines(child, indent_level + 1, output, mode, ctx);
                }
            }

            // End of Tag Updates
            if tag_name == "tr" { if let Some(c) = ctx.as_mut() { if c.is_in_tbody { c.current_row_idx += 1; } } }
            if tag_name == "td" || tag_name == "th" { if let Some(c) = ctx.as_mut() { if c.is_in_tbody { c.current_col_idx += 1; } } }
            if tag_name == "tbody" { if let Some(c) = ctx.as_mut() { c.is_in_tbody = false; } }
        }
        Node::Text(text) => {
            if *mode == PugMode::FullContent {
                let text_content = text.trim();
                if !text_content.is_empty() {
                    output.push_str(&format!("{}| {}\n", indent, text_content.replace("\"", "'")));
                }
            }
        }
        _ => {}
    }
}

pub fn split_doc_to_pug_list(document: &Html, selector_str: &str, mode: PugMode) -> Vec<String> {
    split_doc_to_pug_list_advanced(document, selector_str, mode, None)
}

pub fn split_doc_to_pug_list_advanced(document: &Html, selector_str: &str, mode: PugMode, headers: Option<Vec<Vec<String>>>) -> Vec<String> {
    let selector = match Selector::parse(selector_str) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let mut pug_list = Vec::new();
    
    // rowspan으로 묶인 다음 행들을 병합하기 위한 버퍼와 카운터
    let mut skip_next_n_rows = 0;
    let mut combined_pug_buffer = String::new();

    for node in document.tree.root().descendants() {
        if let Some(element_ref) = scraper::ElementRef::wrap(node) {
            if selector.matches(&element_ref) {
                
                let mut pug_output = String::new();
                pug_output.reserve(2048);
                
                let mut ctx = headers.as_ref().map(|h| TableContext {
                    headers: h.clone(),
                    is_in_tbody: true,
                    ..Default::default()
                });
                
                // 현재 노드의 PUG 라인 생성
                generate_pug_lines(node, 0, &mut pug_output, &mode, &mut ctx);

                // 개별로 바로 push 하지 않고 rowspan 검사
                if !pug_output.trim().is_empty() {
                    // 현재 노드 내부에 rowspan 속성이 있는지 확인
                    let mut current_rowspan = 1;
                    if let Ok(td_selector) = scraper::Selector::parse("td, th") {
                        for cell in element_ref.select(&td_selector) {
                            if let Some(span_str) = cell.value().attr("rowspan") {
                                if let Ok(span) = span_str.parse::<usize>() {
                                    if span > current_rowspan {
                                        current_rowspan = span;
                                    }
                                }
                            }
                        }
                    }

                    if skip_next_n_rows > 0 {
                        // 이전 행에 rowspan이 있어서 현재 행을 합쳐야 하는 경우
                        combined_pug_buffer.push_str(&pug_output);
                        skip_next_n_rows -= 1;
                        
                        // 대기 중인 행을 다 합쳤다면 최종 리스트에 추가
                        if skip_next_n_rows == 0 {
                            pug_list.push(combined_pug_buffer.clone());
                            combined_pug_buffer.clear();
                        }
                    } else if current_rowspan > 1 {
                        // 새로운 rowspan 시작 지점
                        combined_pug_buffer.push_str(&pug_output);
                        skip_next_n_rows = current_rowspan - 1;
                    } else {
                        // 평범한 단일 행
                        pug_list.push(pug_output);
                    }
                }
            }
        }
    }
    
    // 혹시 버퍼에 남은 게 있다면 털어줌
    if !combined_pug_buffer.is_empty() {
        pug_list.push(combined_pug_buffer);
    }
    
    pug_list
}

pub fn extract_table_headers(html: &str, table_selector: &str) -> Vec<Vec<String>> {
    let document = Html::parse_document(html);
    let mut all_headers = Vec::new();
    
    if let Ok(sel) = Selector::parse(table_selector) {
        if let Some(first_match) = document.select(&sel).next() {
            let mut current = first_match.parent();
            while let Some(parent) = current {
                if let Some(el) = parent.value().as_element() {
                    if el.name() == "table" {
                        if let Some(table_ref) = scraper::ElementRef::wrap(parent) {
                            if let Ok(thead_sel) = Selector::parse("thead") {
                                if let Some(thead) = table_ref.select(&thead_sel).next() {
                                    if let Ok(tr_sel) = Selector::parse("tr") {
                                        for tr in thead.select(&tr_sel) {
                                            let mut row_headers = Vec::new();
                                            if let Ok(cell_sel) = Selector::parse("th, td") {
                                                for cell in tr.select(&cell_sel) {
                                                    row_headers.push(cell.text().collect::<Vec<_>>().join(" ").trim().to_string());
                                                }
                                            }
                                            if !row_headers.is_empty() { all_headers.push(row_headers); }
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                }
                current = parent.parent();
            }
        }
    }
    all_headers
}

pub fn split_html_to_pug_list(html: &str, selector_str: &str, mode: PugMode) -> Vec<String> {
    let document = Html::parse_document(html);
    split_doc_to_pug_list(&document, selector_str, mode)
}

pub fn page_type_prompt() -> String { r###"[TASK]
Based on the provided Pug template, identify the primary category of this webpage.

[SCHEMA DEFINITIONS]
- type: The main category. Must be one of:
  - "order": Order list, Order history, Order details, Checkout success.
  - "goods": Product list, product detail.
  - "tracking": Shipment tracking status, delivery history.
  - "review": Product reviews, feedback list.
  - "coupon": Coupon list, discount events.
  - "event": Promotion pages, event announcements.
  - "": If none of the above match.

[OUTPUT FORMAT]
{
    type: String
}"###.to_string() }

pub fn extract_titles_prompt(page_type: &str) -> String {
    let template = r###"[TASK]
Find all the {TYPE} titles from the following PUG/HTML content.

[SCHEMA DEFINITIONS]
- title: {TYPE} title {String}

[OUTPUT FORMAT]
{ "{TYPE}" : [ {"title" : String} ] }

NO EXPLANATION. NO THINKING. /no_think"###;
    template.replace("{TYPE}", page_type)
}

pub fn is_detail_prompt(page_type: &str) -> String {
    let template = r###"[TASK]
Analyze the provided Pug HTML content from top to bottom. Determine if the main content represents a "{TYPE} Detail/{TYPE} Edit Form/{TYPE} Manage Form" (true) or a "{TYPE} List/Index Page/Home Page/Dashboard Page" (false).
[ENTITY CONTEXT: {TYPE}]
You are evaluating a page managing this specific domain entity. Use this context to conceptually understand the abstract structures:
- Single Entity (Detail): A property configuration interface. It features a large overarching form dedicated to inputting or updating the specific attributes of ONE primary entity. (Minor sub-lists for options/variants do not make it a global list).
- Collection (List): A catalog or inventory interface dedicated to displaying, filtering, or batch-processing multiple DIFFERENT primary entities.

[FORCED DOCUMENT SCANNING LOGIC]
Read the entire document from top to bottom. You MUST evaluate the concluding elements at the very bottom of the main content area first.
Look past global navigation menus and overarching search/filter forms at the top. Focus purely on the main data payload and abstract structural signatures:
1. Does the page terminate with dataset navigation (pagination, "next/prev") or bulk-action execution elements?
2. Does the main data area consist of a repeating multi-entity grid?
3. Does the main data area contain an extensive configuration/input form (inputs, textareas, image uploads, save buttons) for a single entity?

[SCHEMA DEFINITIONS]
- has_{TYPE}_list: Boolean. True if the document contains a multi-entity grid, OR if the bottom of main content area has dataset navigation/bulk controls.
- has_{TYPE}_form: Boolean. True if the main data payload is heavily composed of data entry fields (text, select, radio, file uploads) dedicated to creating or updating a single entity.
- detail: Boolean. True ONLY if has_{TYPE}_list is false AND has_{TYPE}_form is true.

[OUTPUT FORMAT]
{
  "has_{TYPE}_list": Boolean,
  "has_{TYPE}_form": Boolean,
  "detail": Boolean
}

NO EXPLANATION. NO THINKING. /no_think"###;
    template.replace("{TYPE}", page_type)
}

// pub fn para2graph(language: &str) -> String {
//     let template = r###"convert the natural language content to fit the dataset JSON structure. no explanation.
//     {
//         "context" : [
//             {
//                 "language" : "{LANG}",
//                 "type": "sales" | "order" | "goods" | "tracking" | "view" | "review" | "coupon" | "event" | "",
//                 "text": "Segment the natural language content into single-type contexts"
//             }
//         ]
//     }"###;
//     template.replace("{LANG}", language)
// }

pub fn para2graph(language: &str) -> String {
    let template = r###"Convert the given natural language content into the specified JSON dataset structure by segmenting it into granular semantic chunks.

[DOCUMENT SCANNING & STRICT SEGMENTATION LOGIC]
1. EXACT COPY: Copy the full input sentence into 'original_text' without changing anything.
2. TAGGED PIPE PLANNING: In the 'segmented_plan' field, you MUST prefix every segment with its assigned category tag in brackets, followed by the exact substring, separated by pipes ("|"). Structure it strictly as "[tag1] chunk1 | [tag2] chunk2".
3. MAXIMAL GROUPING (CRITICAL): Group all contiguous words belonging to the same category into a SINGLE segment. DO NOT split subjects from their numeric conditions. Break the segment ONLY when the category logically shifts.
4. STRICT ARRAY MAPPING: For EVERY tagged segment in 'segmented_plan', create exactly one object in the 'context' array sequentially.
5. VERBATIM EXTRACTION: The 'text' field MUST be the exact substring from the plan, excluding the [tag] and | symbols. DO NOT translate, summarize, alter, or hallucinate any characters.

[SCHEMA DEFINITIONS]
- original_text: String. The exact, unaltered full natural language input.
- segmented_plan: String. The original text with "[type] text | " format inserted strictly at category boundaries.
- language: String. Default "{LANG}".
- categories: The main semantic category. Evaluate strictly based on e-commerce business intent. Must be exactly one of:
  * "order": Intent to measure sales performance or direct transactions. Triggers: conversion rate, sales volume, checkout, payment, cancellation, refund. (RULE: If the context measures buying success or revenue, classify as 'order' even if the word 'product' or 'item' is present).
  * "goods": Intent to describe product catalog data, exposure, or traffic metrics. Triggers: page views, clicks, physical attributes, stock limits, unit prices. (RULE: Focuses on item specifications and customer traffic before the actual purchase).
  * "tracking": Intent to manage logistics and fulfillment. Triggers: shipment status, dispatch, delivery duration, courier information.
  * "review": Intent to analyze the voice of the customer. Triggers: feedback, ratings, reviews, CS messages, complaints.
  * "coupon": Intent to manage specific discount vouchers. Triggers: coupon codes, issuance limits, discount amounts applied via coupons.
  * "event": Intent to manage marketing campaigns or analyze broad operational trends. Triggers: promotions, exhibitions, seasonal sales, overarching managerial analysis requests.
  * "": If none logically apply.

[OUTPUT FORMAT]
{
  "original_text": "String",
  "segmented_plan": "String",
  "context": [
    {
      "language": "String",
      "type": "String",
      "text": "String"
    } 
  ]
}

[ACTION] RETURN STRICTLY VALID JSON ONLY.
NO EXPLANATION. NO THINKING. /no_think"###;
    template.replace("{LANG}", language)
}

// --- File: src/parsing.rs ---

// ==============================================
// [Zone: src/parsing.rs 교체 코드]
// ==============================================
pub fn extract_numeric_conditions(current: &str, input: &str, seg_type: &str, metrics_json: &str) -> String {
    let template = r###"Task: Act as a deterministic semantic parser.
You must extract, transform, and normalize data from the natural language input into the strictly defined JSON output format based on the provided schema.

[DATABASE METRICS CONTEXT]
Use these current database min/max bounds to resolve any relative, proportional, or comparative queries into absolute numeric values.
Metrics: {METRICS}

[SCHEMA DEFINITION]
- operator: A string representing the comparison operator. Allowed values:
  * "gt": Strictly greater than
  * "gte": Greater than or equal to
  * "lt": Strictly less than
  * "lte": Less than or equal to
  * "eq": Exact match

[TRANSFORMATION LOGIC - MANDATORY EXECUTION]
1. ATTRIBUTE EXTRACTION: Identify the context of the numbers or comparative words in the text to determine the property type.
2. RELATIVE VALUE CALCULATION (CRITICAL): 
   - If the query contains relative conditions, percentages, or comparative adjectives, you MUST use the [DATABASE METRICS CONTEXT] to calculate the EXACT absolute numeric threshold. 
   - Do NOT output percentages or descriptive text in the `value` field. Always compute and output the final absolute number derived from the min/max metrics.
3. OPERATOR SELECTION: Map the semantic intent to "gt", "gte", "lt", "lte", or "eq".

[FULL QUERY CONTEXT]
{INPUT}

[CURRENT CHUNK TO ANALYZE]
{CURRENT}

[OUTPUT FORMAT]
{
  "condition": {
    "{TYPE}": {
      "is_percent": Boolean,
      "percent_total": is_percent === true ? 100 : 0,
      "value": "...",
      "operator": "..."
    }
  }
}

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{INPUT}", input)
            .replace("{CURRENT}", current)
            .replace("{TYPE}", seg_type)
            .replace("{METRICS}", metrics_json)
}
// ==============================================

pub fn graph2contexts(current_text: &str, seg_type: &str) -> String {
    let template = r###"Analyze the specific text segment and extract the logical attributes based on the defined schema.

[SCHEMA DEFINITIONS]
* status: Extract the exact status mentioned. Choose ONLY from: 'progress', 'stop', 'cancel', 'refund', 'return', 'exchange', 'expire', 'complete'. If no status is explicitly mentioned, return null.
* substantial: Extract specifically mentioned properties. Choose ONLY from: 'size', 'weight', 'shipping_fee', 'shipping_duration', 'sale_price', 'supply_price', 'low_stock_threshold', 'discount', 'min_order_amount', 'max_discount_amount', 'usage_limit', 'usage_per'. (CRITICAL: Return an array of strings ONLY if these exact words exist in the text. Do NOT blindly copy this list. If none exist, return null).
* find: Extract search intents. Choose ONLY from: 'many', 'few', 'much', 'little', 'heavy', 'light'. (CRITICAL: Return an array of strings ONLY if these exact words exist in the text. Do NOT blindly copy this list. If none exist, return null).

[CURRENT SEGMENT]
Category Type: {TYPE}
Text: {TEXT}

[OUTPUT FORMAT]
{
  "status": "...",
  "substantial": "...",
  "find": "..."
}

[ACTION] RETURN STRICTLY VALID JSON ONLY.
NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{TYPE}", seg_type).replace("{TEXT}", current_text)
}


pub fn item2json(page_type: &str, href: &str, language: &str) -> String {
    let schema = match page_type {
    "tracking" => r###"status:'draft' or 'progress' or 'return' or 'complete' or 'error',
id:tracking number | string,
title:tracking goods title | string,
sender_name:sender_name | string,
sender_address:sender_address | string,
sender_phone:sender_phone | string,
recipient_name:recipient_name | string,
recipient_address:recipient_address | string,
recipient_phone:recipient_phone | string,
width:Package width | number,
height:Package height | number,
length:Package length | number,
weight:Package weight | number,
carrier:carrier name translated into English | string,
shipping_fee:Shipping cost | number,
shipping_method:'standard' or 'express' or 'same_day' or 'pick_up' or 'freight' or 'prepaid',
shipping_duration:Estimated delivery days | number,
bundle_shipping:Allow combined shipping | string,
shipping_date:yyyy-MM-ddThh:mm:ss | string,
registration_date:yyyy-MM-ddThh:mm:ss | string,"###.to_string(),
    "goods" => r###"node:goods form container CSS selector,
code:product constant code | string,
link:'{HREF}',
id:Refer to the ID value from the link | string,
status:'draft' or 'show' or 'hide' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
payment_method:payment method | string,
bank:bank company name or '' | string,
card:card company name or '' | string,
model_name:product Model name | string,
brand_name:product Brand name | string,
condition:['new' or 'used' or 'lease' or 'rental' or 'refurbish'],
description:product Full description (HTML allowed) | string,
short_description:product short description | string,
tags:[{ tag : product keyword or tag | string }],
origin_country:product Country of origin/manufacture | string,
manufacturer:product Manufacturer name | string,
release_date:Product release date(yyyy-MM-ddThh:mm:ss) | string,
manufacture_date:product Date(yyyy-MM-ddThh:mm:ss) of manufacture | string,
expiration_date:product Expiration or use-by date(yyyy-MM-ddThh:mm:ss) | string,
gtin:product Global Trade Item Number | string,
mpn:product Manufacturer Part Number | string,
barcode:product Barcode value | string,
sale_price:product sale price | number,
supply_price:product supply price | number,
currency:ISO 4217 Currency Code | string,
compare_at_price:product Original price for showing discounts | number,
quantity:product Inventory quantity | number,
stock_keeping_unit:Stock Keeping Unit | string,
low_stock_threshold:product Low stock alert threshold | number,
unit:product Selling unit | string,
tax_included:product Whether tax | number,
tax_code:product Tax code for region-specific rules | string,
main_image_url:Main product image URL | string,
additional_image_url:additional product image URL | string,
video_url:product Promotional video URL | string,
carrier:product carrier name translated into English | string,
shipping_fee:product Shipping cost | number,
shipping_method:'standard' or 'express' or 'same_day' or 'pick_up' or 'freight' or 'prepaid',
shipping_duration:product Estimated delivery days | number,
bundle_shipping:product Allow combined shipping | string,
width:Package width(cm) | number,
height:Package height(cm) | number,
length:Package length(cm) | number,
weight:Package weight(kg) | number,
options:[
    {
        value:option name | string,
        inputs:[{
            value:option input value | string,
        }]
    }
],
additional_goods:[
    {
        value:URL includes a manage path, an administrative or edit route product Link | string,
    }
],
title:product name | string,
registration_date:yyyy-MM-ddThh:mm:ss | string,"###.replace("{HREF}", href),
        "order" => r###"node:order form container CSS selector,
link : '{HREF}',
id:Refer to the ID value from the link | string,
tracking_number:tracking number | string,
status:'draft' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
goods:[{
    title:{
        value:goods title | string,
    },
    link:{
        value:URL includes a manage path, an administrative or edit route goods Link | string,
    },
    id:{
        value:Refer to the product no value from the link or an attribute or input value | string,
    }
}],
sender_name:sender_name | string,
sender_address:sender_address, Filter the addresses to District-level and up | string,
sender_phone:sender_phone | string,
recipient_name:recipient_name | string,
recipient_address:recipient_address, Filter the addresses to District-level and up | string,
recipient_phone:recipient_phone | string,
bank:bank company name | string,
card:card company name | string,
order_date:order date | string,
payment_date:payment date or '' | string,
payment_method:'C.O.D.' or 'CARD' or 'BANK' or '' | string,
payment_origin:Payment Gateway Service Name or '' | string,
registration_date:yyyy-MM-ddThh:mm:ss | string,"###.replace("{HREF}", href),
    "coupon" | "event" => r###"node:{TYPE} container CSS selector,
link : '{HREF}',
id:Refer to the ID value from the link | string,
type:'percentage' or 'fixed_amount' or 'free_shipping' or '',
status:'draft' or 'progress' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error',
title:{TYPE} title | string, 
started_at:yyyy-MM-ddThh:mm:ss | string,
expired_at:yyyy-MM-ddThh:mm:ss | string,
code:{TYPE} code used at checkout | string,
discount:Discount value | number,
quantity:{TYPE} quantity | number,
usage_limit:Total usage limit for the coupon | number,
usage_per:Usage limit per customer | number,
new_customer_only:new customer only | boolean,
first_purchase_only:first purchase only | boolean,
min_order_amount:Minimum order amount required to apply coupon | number,
max_order_amount:Maximum order amount allowed to apply coupon | number,
max_discount_amount:Maximum discount limit allowed for the coupon | number,
region_restrictions:region restrictions | boolean,
number:contact phone number | string,
address:offline location address | string,
registration_date:yyyy-MM-ddThh:mm:ss | string,"###.replace("{TYPE}", page_type).replace("{HREF}", href),
    "review" => r###"node:review container CSS selector,
link : '{HREF}',
id:Refer to the ID value from the link | string,
status:'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
name:reviewer name | string,
title:reviewer item title | string, 
completed:order complete | boolean,
registration_date:yyyy-MM-ddThh:mm:ss | string,"###.replace("{HREF}", href),
        _ => "- id: Unique identifier.\n- title: General name or title.\n- status: Current state.".to_string()
    };

    let template = r###" 
[TASK]
Extract detailed information from the provided Pug template into a single structured JSON object.

[CONTEXT]
Page Type: {TYPE}

[SCHEMA DEFINITIONS]
{SCHEMA}

[EXTRACTION RULES]
1. Return ONLY valid JSON. No preamble, no postscript.
2. If a field is missing in the data, use null.
3. Normalize all dates to 'yyyy-MM-ddThh:mm:ss'.
4. Extract only numeric values for price, amount, weight, and dimensions.
5. Do NOT make up data. Only extract what is present in the Pug structure.

[OUTPUT FORMAT]
{
    field name: extracted value
}"###;

    template.replace("{TYPE}", page_type)
            .replace("{HREF}", href)
            .replace("{LANGUAGE}", language)
            .replace("{SCHEMA}", &schema)
}

pub fn list2json(page_type: &str, language: &str) -> String {
    let schema = match page_type {
    "order" | "goods" => r###"status:'show' or 'progress' or 'remove' or 'hide' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
link:URL includes a manage path, an administrative or edit route Link | string,
id:Refer to the ID value from the link or an attribute | string,
title:title | string, 
sale_price:sale price | number,
supply_price:supply price | number,
currency:ISO 4217 Currency Code | string,
quantity:item stock quantity | number,
tracking_number:Tracking Number or 운송장 번호 or 运单호 or 運單號 or 伝표번호 or Número de seguimiento or Numéro de suivi or Sendungsnummer or Ноमर 나кладной or Número de rastreamento or Numero di tracciamento or رقم التتبع or Số vận đơn or Nomor resi or หมายเลขติดตามพัสดุ | string,
registration_date:yyyy-MM-ddThh:mm:ss | string,"###.to_string(),
    "tracking" | "review" => r###"status:'start' or 'progress' or 'stop' or 'cancel' or 'return',
id:Refer to the ID value from the link or an attribute | string,
title:author and content | string, 
link:URL includes a manage path, an administrative or edit route Link | string,
registration_date:yyyy-MM-ddThh:mm:ss | string,"###.to_string(),
    "coupon" | "event" => r###"status:'show' or 'progress' or 'hide' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error',
id:Refer to the ID value from the link or an attribute | string,
title:type based item title, 
started_at:yyyy-MM-ddThh:mm:ss,
expired_at:yyyy-MM-ddThh:mm:ss,
registration_date:yyyy-MM-ddThh:mm:ss | string,"###.to_string(),
        _ => "id: ID\ntitle: Title\nstatus: Status".to_string()
    };

    let template = r###"[TASK]
Extract detailed information from the provided Pug template into a single structured JSON object.

[SCHEMA DEFINITIONS]
{SCHEMA}

[OUTPUT FORMAT]
Return ONLY a single valid JSON object. 
If a field is missing, use null. 
Normalize all dates to 'yyyy-MM-ddThh:mm:ss'. 
Extract only numeric values for price, quantity, amount.

[ACTION] RETURN JSON ONLY. 
NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{SCHEMA}", &schema)
}

/// Converts a JSON Value into a human-readable natural language narrative.
/// [STRICT ALIGNMENT] This logic perfectly synchronizes with every column in `parsing.rs`.
pub fn json_to_natural_language(value: &serde_json::Value) -> String {
    let mut output = String::new();
    
    if let Some(obj) = value.as_object() {
        if obj.len() == 1 && obj.contains_key("value") {
            return obj.get("value").unwrap().as_str().unwrap_or(&obj.get("value").unwrap().to_string()).to_string();
        }
    }

    if let serde_json::Value::Object(map) = value {
        let page_type = map.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let is_detail = map.get("detail").and_then(|v| v.as_bool()).unwrap_or(true);

        // 🌟 [CRITICAL FIX] 추출 스키마와 완벽하게 일치하도록 이름표 동기화 및 누락 항목 추가 완료!
        let keys: Vec<&str> = match page_type {
            "tracking" => {
                if is_detail {
                    vec!["status", "id", "title", "sender_name", "sender_address", "sender_phone", "recipient_name", "recipient_address", "recipient_phone", "width", "height", "length", "weight", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "bundle_shipping", "shipping_date", "registration_date"]
                } else {
                    vec!["status", "id", "title", "link", "registration_date"]
                }
            },
            "goods" => {
                if is_detail {
                    vec!["code", "link", "id", "status", "payment_method", "bank", "card", "model_name", "brand_name", "condition", "description", "short_description", "tags", "origin_country", "manufacturer", "release_date", "manufacture_date", "expiration_date", "gtin", "mpn", "barcode", "sale_price", "supply_price", "currency", "compare_at_price", "quantity", "stock_keeping_unit", "low_stock_threshold", "unit", "tax_included", "tax_code", "main_image_url", "additional_image_url", "video_url", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "bundle_shipping", "width", "height", "length", "weight", "options", "additional_goods", "title", "registration_date"]
                } else {
                    vec!["status", "link", "id", "title", "sale_price", "supply_price", "currency", "quantity", "tracking_number", "registration_date"]
                }
            },
            "order" => {
                if is_detail {
                    vec!["link", "id", "tracking_number", "status", "goods", "sender_name", "sender_address", "sender_phone", "recipient_name", "recipient_address", "recipient_phone", "bank", "card", "order_date", "payment_date", "payment_method", "payment_origin", "registration_date"]
                } else {
                    vec!["status", "link", "id", "title", "sale_price", "supply_price", "currency", "quantity", "tracking_number", "registration_date"]
                }
            },
            "coupon" | "event" => {
                if is_detail {
                    vec!["link", "id", "type", "status", "title", "started_at", "expired_at", "code", "discount", "quantity", "usage_limit", "usage_per", "new_customer_only", "first_purchase_only", "min_order_amount", "max_order_amount", "max_discount_amount", "region_restrictions", "number", "address", "registration_date"]
                } else {
                    vec!["status", "id", "title", "started_at", "expired_at", "registration_date"]
                }
            },
            "review" => {
                if is_detail {
                    vec!["link", "id", "status", "name", "title", "completed", "registration_date"]
                } else {
                    vec!["status", "id", "title", "link", "registration_date"]
                }
            },
            _ => map.keys().map(|s| s.as_str()).collect()
        };

        for key in keys {
            if let Some(v) = map.get(key) {
                if v.is_null() { continue; }
                let key_name = key.replace("_", " ");
                if v.is_array() {
                    let arr = v.as_array().unwrap();
                    let mut items = Vec::new();
                    for item in arr.iter().take(5) {
                        let sub = json_to_natural_language(item);
                        if !sub.is_empty() { items.push(sub); }
                    }
                    if !items.is_empty() {
                        output.push_str(&format!("{}: [{}]. ", key_name, items.join(", ")));   
                    }
                } else if v.is_object() {
                    let sub = json_to_natural_language(v);
                    if !sub.is_empty() {
                        output.push_str(&format!("{}: {}. ", key_name, sub));
                    }
                } else {
                    let s = match v {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Number(n) => n.to_string(),
                        serde_json::Value::Bool(b) => b.to_string(),
                        _ => String::new(),
                    };
                    if !s.is_empty() && s != "null" {
                        let s_clean = if s.len() > 400 { format!("{}...", &s[..400]) } else { s };
                        output.push_str(&format!("{}: {}. ", key_name, s_clean));
                    }
                }
            }
        }
    } else if let serde_json::Value::Array(arr) = value {
        for item in arr.iter().take(10) {
            let sub = json_to_natural_language(item);
            if !sub.is_empty() {
                output.push_str(&sub);
                output.push_str(" ");
            }
        }
    } else {
        output.push_str(&value.as_str().unwrap_or(&value.to_string()));
    }

    output.trim().to_string()
}

pub fn normalize_to_json_string(input: &str) -> String {
    let mut s = input.replace(&['\u{00A0}', '\u{200B}', '\u{202F}', '\u{FEFF}'][..], " ").trim().to_string();

    // 1. Backticks to quotes
    let re_backtick = Regex::new(r"`([\s\S]*?)`").unwrap();
    s = re_backtick.replace_all(&s, |caps: &regex::Captures| {
        format!("\"{}\"", caps[1].replace("\"", "\\\""))
    }).to_string();

    // 2. Key quotes correction (key: -> "key":)
    let re_keys = Regex::new(r"([{,])\s*([a-zA-Z0-9_]+)\s*:").unwrap();
    s = re_keys.replace_all(&s, "$1\"$2\":").to_string();

    // 3. Single quotes to double quotes for values
    let re_single_vals = Regex::new(r":\s*'([^']*)'").unwrap();
    s = re_single_vals.replace_all(&s, |caps: &regex::Captures| {
        format!(": \"{}\"", caps[1].replace("\"", "\\\""))
    }).to_string();

    // 4. CSS selector/nested quotes protection (Simplified for Rust regex)
    let re_nested = Regex::new(r#"="([^"]*)""#).unwrap();
    s = re_nested.replace_all(&s, "=\\\"$1\\\"").to_string();

    // 5. Trailing Comma removal
    let re_trailing = Regex::new(r",\s*([\]}])").unwrap();
    s = re_trailing.replace_all(&s, "$1").to_string();

    // 6. Force close braces
    let open_braces = s.matches('{').count();
    let close_braces = s.matches('}').count();
    if open_braces > close_braces {
        s.push_str(&"}".repeat(open_braces - close_braces));
    }

    s
}

pub fn parse_json_from_llm(text: &str) -> serde_json::Value {
    // [CLEANUP] Remove <think>...</think> tags if they exist
    let mut clean_text = text.to_string();
    if let Some(start_think) = clean_text.find("<think>") {
        if let Some(end_think) = clean_text.find("</think>") {
            if start_think < end_think {
                clean_text.replace_range(start_think..end_think + 8, "");
            }
        } else {
            clean_text.replace_range(start_think.., "");
        }
    }
    let clean_text = clean_text.trim();

    // 1. First attempt: Direct parse
    if let Ok(v) = serde_json::from_str(clean_text) { return v; }

    // 2. Extract JSON part and attempt direct parse
    let mut extracted = String::new();
    if let Some(start) = clean_text.find("{") {
        if let Some(end) = clean_text.rfind("}") {
            if start < end {
                extracted = clean_text[start..=end].to_string();
                if let Ok(v) = serde_json::from_str(&extracted) { return v; }
            }
        }
    } else if let Some(start) = clean_text.find("[") {
        if let Some(end) = clean_text.rfind("]") {
            if start < end {
                extracted = clean_text[start..=end].to_string();
                if let Ok(v) = serde_json::from_str(&extracted) { return v; }
            }
        }
    }

    // 3. Last attempt: Normalize/Repair then parse
    let to_repair = if extracted.is_empty() { clean_text } else { &extracted };
    let repaired = normalize_to_json_string(to_repair);
    
    if let Ok(v) = serde_json::from_str(&repaired) {
        return v;
    } else {
        // Final fallback: try extracting from repaired string again
        if let Some(start) = repaired.find("{") {
            if let Some(end) = repaired.rfind("}") {
                if let Ok(v) = serde_json::from_str(&repaired[start..=end]) { return v; }
            }
        }
    }

    println!("[Parsing] Warning: Failed to repair dirty JSON: {}", clean_text);
    serde_json::json!({})
}





pub fn extract_shipping_conditions(query: &str, language: &str) -> String {
    let template = r###"Task: Act as a deterministic shipping and logistics semantic parser.
Extract the logistics filters from the natural language query into the JSON format.

[SCHEMA DEFINITION]
Extract the following tracking properties if semantically present in the text:
- "no": Tracking number, waybill number, or reference identifier.
- "status": Shipping status (Allowed values: draft, progress, return, complete, error).
- "carrier": Courier, logistics carrier, vessel, or flight name.
- "shipping_method": Mode of transport or delivery service level.
- "sender_address": Origin location, port of loading, departure point, or sender address.
- "recipient_address": Destination location, port of discharge, arrival point, or recipient address.
- "shipping_date": Dispatch date, departure date, or shipped on board date.
- "delivery_date": Arrival date or delivered date.
- "weight": Cargo, package, or gross weight.
- "shipping_fee": Logistics cost, freight charge, or shipping fee.

[TRANSFORMATION LOGIC]
For EVERY extracted field, you MUST wrap it in an operator object:
{ "operator": "eq" | "gt" | "lt" | "gte" | "lte" | "contains", "value": <extracted_value> }

- Use "contains" for text fields, locations, and names.
- Use "eq" for exact matches like status or strict identifiers.
- Use "gt", "gte", "lt", "lte" for numeric values, amounts, weights, and dates.
- Universality Rule: Do NOT include descriptive text or units (like 'kg', '$', 'days') in the value. Extract and compute only the raw numbers or normalized strings.

[QUERY]
{QUERY}

[OUTPUT FORMAT]
{
  "<property_name>": { "operator": "...", "value": "..." }
}

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{QUERY}", query).replace("{LANGUAGE}", language)
}
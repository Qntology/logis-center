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
{ "{TYPE}" : [...] }

RETURN JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;
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
- {TYPE}:
  - has_list: Boolean. True if the document contains a multi-entity grid, OR if the bottom of main content area has dataset navigation/bulk controls.
  - has_form: Boolean. True if the main data payload is heavily composed of data entry fields (text, select, radio, file uploads) dedicated to creating or updating a single entity.
  - detail: Boolean. True ONLY if has_list is false AND has_form is true.

[OUTPUT FORMAT]
{
  "{TYPE}": {
    "has_list": Boolean,
    "has_form": Boolean,
    "detail": Boolean
  }
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
2. TAGGED PIPE PLANNING: In the 'segmented_plan' field, you MUST prefix every segment with its assigned type tag in brackets, followed by the exact substring, separated by pipes ('|'). Structure it strictly as '[tag1] chunk1 | [tag2] chunk2'.
3. MAXIMAL GROUPING (CRITICAL): Group all contiguous words belonging to the same type into a SINGLE segment. DO NOT split subjects from their numeric conditions. Break the segment ONLY when the type logically shifts.
4. STRICT ARRAY MAPPING: For EVERY tagged segment in 'segmented_plan', create exactly one object in the 'context' array sequentially.
5. VERBATIM EXTRACTION: The 'text' field MUST be the exact substring from the plan, excluding the [tag] and | symbols. DO NOT translate, summarize, alter, or hallucinate any characters.

[SCHEMA DEFINITIONS]
- original_text: String. The exact, unaltered full natural language input.
- segmented_plan: String. The original text with '[type] text | ' format inserted strictly at type boundaries.
- context:
  - text: String.
  - language: String. Default '{LANG}'.
  - type: The main semantic type. Evaluate strictly based on e-commerce business intent. Must be exactly one of:
    * 'order': Intent to measure sales performance or direct transactions. Triggers: conversion rate, sales volume, checkout, payment, cancellation, refund. (RULE: If the context measures buying success or revenue, classify as 'order' even if the word 'product' or 'item' is present).
    * 'goods': Intent to describe product catalog data, exposure, or traffic metrics. Triggers: page views, clicks, physical attributes, stock limits, unit prices. (RULE: Focuses on item specifications and customer traffic before the actual purchase).
    * 'tracking': Intent to manage logistics and fulfillment. Triggers: shipment status, dispatch, delivery duration, courier information.
    * 'review': Intent to analyze the voice of the customer. Triggers: feedback, ratings, reviews, CS messages, complaints.
    * 'coupon': Intent to manage specific discount vouchers. Triggers: coupon codes, issuance limits, discount amounts applied via coupons.
    * 'event': Intent to manage marketing campaigns or analyze broad operational trends. Triggers: promotions, exhibitions, seasonal sales, overarching managerial analysis requests.
    * '': If none logically apply.

[OUTPUT FORMAT]
{
  "original_text": "String",
  "segmented_plan": "String",
  "context": [...]
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
    let template = r###"[Task]
Act as a deterministic semantic parser.
You must extract, transform, and normalize data from the natural language input into the strictly defined JSON output format based on the provided schema.

[DATABASE METRICS CONTEXT]
Use these current database min/max bounds to resolve any relative, proportional, or comparative queries into absolute numeric values.
Metrics: {METRICS}

[SCHEMA DEFINITION]
- operator: A string representing the comparison operator. Allowed values:
  * 'gt': Strictly greater than
  * 'gte': Greater than or equal to
  * 'lt': Strictly less than
  * 'lte': Less than or equal to
  * 'eq': Exact match

[TRANSFORMATION LOGIC - MANDATORY EXECUTION]
1. ATTRIBUTE EXTRACTION: Identify the context of the numbers or comparative words in the text to determine the property type.
2. RELATIVE VALUE CALCULATION (CRITICAL): 
   - If the query contains relative conditions, percentages, or comparative adjectives, you MUST use the [DATABASE METRICS CONTEXT] to calculate the EXACT absolute numeric threshold. 
   - Do NOT output percentages or descriptive text in the `value` field. Always compute and output the final absolute number derived from the min/max metrics.
3. OPERATOR SELECTION: Map the semantic intent to 'gt', 'gte', 'lt', 'lte', or 'eq'.

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
    // 🌟 도메인(Type)별로 허용되는 상태(Status) 값을 완벽히 분리하여 매핑합니다.
    let status_options = match seg_type {
        "tracking" => "* 'draft'
  * 'progress'
  * 'return'
  * 'complete'
  * 'error'",
        "goods" => "* 'draft'
  * 'show'
  * 'hide'
  * 'progress'
  * 'stop'
  * 'cancel'
  * 'refund'
  * 'return'
  * 'exchange'
  * 'expire'
  * 'complete'
  * 'error'",
        "order" => "* 'draft'
  * 'progress'
  * 'stop'
  * 'cancel'
  * 'refund'
  * 'return'
  * 'exchange'
  * 'expire'
  * 'complete'
  * 'error'",
        "coupon" | "event" => "* 'show'
  * 'progress'
  * 'hide'
  * 'stop'
  * 'cancel'
  * 'expire'
  * 'complete'
  * 'error'",
        "review" => "* 'progress'
  * 'stop'
  * 'cancel'
  * 'refund'
  * 'return'
  * 'exchange'
  * 'expire'
  * 'complete'
  * 'error'",
        _ => "* 'show'
  * 'progress'
  * 'remove'
  * 'hide'
  * 'stop'
  * 'cancel'
  * 'refund'
  * 'return'
  * 'exchange'
  * 'expire'
  * 'complete'
  * 'error'"
    };

    let template = r###"Analyze the specific text segment and extract the logical attributes based on the defined schema.

[CURRENT SEGMENT]
{TEXT}

[SCHEMA DEFINITIONS]
{TYPE}:
  - status:
    {STATUS_OPTIONS}
  - substantial:
    * 'size'
    * 'weight'
    * 'shipping_fee'
    * 'shipping_duration'
    * 'sale_price'
    * 'supply_price'
    * 'low_stock_threshold'
    * 'discount'
    * 'min_order_amount'
    * 'max_discount_amount'
    * 'usage_limit'
    * 'usage_per'
  - find:
    * 'many'
    * 'few'
    * 'much'
    * 'little'
    * 'heavy'
    * 'light'

[OUTPUT FORMAT]
{
  "{TYPE}" : {
      "status": "...",
      "substantial": "...",
      "find": "..."
  }
}

[ACTION] RETURN STRICTLY VALID JSON ONLY.
NO EXPLANATION. NO THINKING. /no_think"###;

    // 🌟 {STATUS_OPTIONS} 를 먼저 치환한 뒤 {TYPE}을 치환합니다.
    template.replace("{STATUS_OPTIONS}", status_options)
            .replace("{TYPE}", seg_type)
            .replace("{TEXT}", current_text)
}


pub fn item2json(page_type: &str, href: &str, language: &str) -> String {
    let schema = match page_type {
    "tracking" => r###"- "{TYPE}":
    - "status":'draft' or 'progress' or 'return' or 'complete' or 'error' | string
    - "id":tracking number | string
    - "title":tracking goods title | string
    - "sender_name":sender_name | string
    - "sender_address":sender_address | string
    - "sender_phone":sender_phone | string
    - "recipient_name":recipient_name | string
    - "recipient_address":recipient_address | string
    - "recipient_phone":recipient_phone | string
    - "width":Package width | number
    - "height":Package height | number
    - "length":Package length | number
    - "weight":Package weight | number
    - "carrier":carrier name translated into English | string
    - "shipping_fee":Shipping cost | number
    - "shipping_method":'standard' or 'express' or 'same_day' or 'pick_up' or 'freight' or 'prepaid' | string
    - "shipping_duration":Estimated delivery days | number
    - "bundle_shipping":Allow combined shipping | string
    - "shipping_date":yyyy-MM-ddThh:mm:ss | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string"###.to_string(),
    "goods" => r###"- "{TYPE}":
    - "node":goods form container CSS selector | string
    - "code":product constant code | string
    - "link":'{HREF}' | string
    - "id":Refer to the ID value from the link | string
    - "status":'draft' or 'show' or 'hide' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error' | string
    - "payment_method":payment method | string
    - "bank":bank company name or '' | string
    - "card":card company name or '' | string
    - "model_name":product Model name | string
    - "brand_name":product Brand name | string
    - "condition":['new' or 'used' or 'lease' or 'rental' or 'refurbish']
    - "description":product Full description (HTML allowed) | string
    - "short_description":product short description | string
    - "tags":[{ tag : product keyword or tag | string }]
    - "origin_country":product Country of origin/manufacture | string
    - "manufacturer":product Manufacturer name | string
    - "release_date":Product release date(yyyy-MM-ddThh:mm:ss) | string
    - "manufacture_date":product Date(yyyy-MM-ddThh:mm:ss) of manufacture | string
    - "expiration_date":product Expiration or use-by date(yyyy-MM-ddThh:mm:ss) | string
    - "gtin":product Global Trade Item Number | string
    - "mpn":product Manufacturer Part Number | string
    - "barcode":product Barcode value | string
    - "sale_price":product sale price | number
    - "supply_price":product supply price | number
    - "currency":ISO 4217 Currency Code | string
    - "compare_at_price":product Original price for showing discounts | number
    - "quantity":product Inventory quantity | number
    - "stock_keeping_unit":Stock Keeping Unit | string
    - "low_stock_threshold":product Low stock alert threshold | number
    - "unit":product Selling unit | string
    - "tax_included":product Whether tax | number
    - "tax_code":product Tax code for region-specific rules | string
    - "main_image_url":Main product image URL | string
    - "additional_image_url":additional product image URL | string
    - "video_url":product Promotional video URL | string
    - "carrier":product carrier name translated into English | string
    - "shipping_fee":product Shipping cost | number
    - "shipping_method":'standard' or 'express' or 'same_day' or 'pick_up' or 'freight' or 'prepaid' | string
    - "shipping_duration":product Estimated delivery days | number
    - "bundle_shipping":product Allow combined shipping | string
    - "width":Package width(cm) | number
    - "height":Package height(cm) | number
    - "length":Package length(cm) | number
    - "weight":Package weight(kg) | number
    - "options":[ { value:option name | string, inputs:[{ value:option input value | string }] } ]
    - "additional_goods":[ { path:{ value:URL includes a manage path, an administrative or edit Link | string }, id:{ value:Refer to the product no value from the link or an attribute or input value | string }, link:{ value:Refer to the ID to find a URL that includes a manage link | string } } ]
    - "title":product name | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string"###.to_string(),
        "order" => r###"- "{TYPE}":
    - "node":order form container CSS selector | string
    - "link":'{HREF}' | string
    - "id":Refer to the ID value from the link | string
    - "tracking_number":tracking number | string
    - "status":'draft' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error' | string
    - "goods":[{ title:{ value:goods title | string }, path:{ value:URL includes a manage path, an administrative or edit Link | string }, id:{ value:Refer to the product no value from the link or an attribute or input value | string }, link:{ value:Refer to the ID to find a URL that includes a manage link | string } }]
    - "sender_name":sender_name | string
    - "sender_address":sender_address, Filter the addresses to District-level and up | string
    - "sender_phone":sender_phone | string
    - "recipient_name":recipient_name | string
    - "recipient_address":recipient_address, Filter the addresses to District-level and up | string
    - "recipient_phone":recipient_phone | string
    - "bank":bank company name | string
    - "card":card company name | string
    - "order_date":order date | string
    - "payment_date":payment date or '' | string
    - "payment_method":'C.O.D.' or 'CARD' or 'BANK' or '' | string
    - "payment_origin":Payment Gateway Service Name or '' | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string"###.to_string(),
    "coupon" | "event" => r###"- "{TYPE}":
    - "node":{TYPE} container CSS selector | string
    - "link":'{HREF}' | string
    - "id":Refer to the ID value from the link | string
    - "type":'percentage' or 'fixed_amount' or 'free_shipping' or '' | string
    - "status":'draft' or 'progress' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error' | string
    - "title":{TYPE} title | string
    - "started_at":yyyy-MM-ddThh:mm:ss | string
    - "expired_at":yyyy-MM-ddThh:mm:ss | string
    - "code":{TYPE} code used at checkout | string
    - "discount":Discount value | number
    - "quantity":{TYPE} quantity | number
    - "usage_limit":Total usage limit for the coupon | number
    - "usage_per":Usage limit per customer | number
    - "new_customer_only":new customer only | boolean
    - "first_purchase_only":first purchase only | boolean
    - "min_order_amount":Minimum order amount required to apply coupon | number
    - "max_order_amount":Maximum order amount allowed to apply coupon | number
    - "max_discount_amount":Maximum discount limit allowed for the coupon | number
    - "region_restrictions":region restrictions | boolean
    - "number":contact phone number | string
    - "address":offline location address | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string"###.to_string(),
    "review" => r###"- "{TYPE}":
    - "node":review container CSS selector | string
    - "link":'{HREF}' | string
    - "id":Refer to the ID value from the link | string
    - "status":'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error' | string
    - "name":reviewer name | string
    - "title":reviewer item title | string
    - "completed":order complete | boolean
    - "registration_date":yyyy-MM-ddThh:mm:ss | string"###.to_string(),
        _ => r###"- "{TYPE}":
    - "id": Unique identifier | string
    - "title": General name or title | string
    - "status": Current state | string"###.to_string()
    };

    let template = r###"[TASK]
Extract detailed information from the provided Pug template into a single structured JSON object.

[CONTEXT]
Page Type: {TYPE}
current Link: {HREF}

[SCHEMA DEFINITIONS]
{SCHEMA}

[EXTRACTION RULES]
1. Return ONLY valid JSON. No preamble, no postscript.
2. If a field is missing in the data, use null.
3. Normalize all dates to 'yyyy-MM-ddThh:mm:ss'.
4. Extract only numeric values for price, amount, weight, and dimensions.
5. Do NOT make up data. Only extract what is present in the Pug structure.

[OUTPUT FORMAT]
{...}

[ACTION] RETURN JSON ONLY. 
NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{SCHEMA}", &schema)
            .replace("{TYPE}", page_type)
            .replace("{HREF}", href)
            .replace("{LANGUAGE}", language)
}

pub fn list2json(page_type: &str, href: &str, language: &str, head_pug: &str, item_pug: &str) -> String {
    let schema = match page_type {
    "order" => r###"- "order":
    - "title":title | string
    - "path":URL includes a manage order path, an administrative or edit Link | string
    - "id":Refer to the ID value from the link or an attribute | string
    - "link":Refer to the ID to find a URL that includes a manage order link | string
    - "quantity":quantity | string
    - "sale_price":total price | number
    - "currency":ISO 4217 Currency Code | string
    - "tracking_number":Tracking Number or equivalent term in English | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string
    - "status":'show' or 'progress' or 'remove' or 'hide' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error' | string"###.to_string(),

    "goods" => r###"- "goods":
    - "title":title | string
    - "path":URL includes a manage goods path, an administrative or edit Link | string
    - "id":Refer to the ID value from the link or an attribute | string
    - "link":Refer to the ID to find a URL that includes a manage goods link | string
    - "quantity":quantity | string
    - "sale_price":total price | number
    - "currency":ISO 4217 Currency Code | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string
    - "status":'show' or 'progress' or 'remove' or 'hide' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error' | string"###.to_string(),
    
    "tracking" | "review" => r###"- "{TYPE}":
    - "title":author and content | string
    - "path":URL includes a manage {TYPE} path, an administrative or edit Link | string
    - "id":Refer to the ID value from the link or an attribute | string
    - "link":Refer to the ID to find a URL that includes a manage {TYPE} link | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string
    - "status":'start' or 'progress' or 'stop' or 'cancel' or 'return' | string"###.to_string(),
    
    "coupon" | "event" => r###"- "{TYPE}":
    - "title":type based item title | string
    - "path":URL includes a manage {TYPE} path, an administrative or edit Link | string
    - "id":Refer to the ID value from the link or an attribute | string
    - "link":Refer to the ID to find a URL that includes a manage {TYPE} link | string
    - "started_at":yyyy-MM-ddThh:mm:ss | string
    - "expired_at":yyyy-MM-ddThh:mm:ss | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string
    - "status":'show' or 'progress' or 'hide' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error' | string"###.to_string(),
    
        _ => r###"- "{TYPE}":
    - "title":title | string
    - "path":URL includes a manage {TYPE} path, an administrative or edit Link | string
    - "id":Refer to the ID value from the link or an attribute | string
    - "link":Refer to the ID to find a URL that includes a manage {TYPE} link | string
    - "status":'show' or 'progress' or 'remove' or 'hide' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error' | string"###.to_string()
    };

    // 🌟 [최종 반영] .txt 파일 구조와 동일하게 thead/tbody 태그 및 계층형 들여쓰기 적용
    let mut final_pug = String::new();

    // 1. 헤더 영역 (thead 태그 추가 및 내부 1단계 들여쓰기)
    if !head_pug.is_empty() {
        final_pug.push_str("thead\n");
        for line in head_pug.lines() {
            final_pug.push_str(&format!("    {}\n", line));
        }
    }

    // 2. 바디 영역 (tbody 태그 추가 및 내부 1단계 들여쓰기)
    if !item_pug.is_empty() {
        final_pug.push_str("tbody\n");
        for line in item_pug.lines() {
            final_pug.push_str(&format!("    {}\n", line));
        }
    }

    // 3. 전체 PUG를 프롬프트에 넣기 위해 일괄적으로 4칸 들여쓰기 적용
    let pug_content = final_pug.trim_end().lines()
        .map(|line| format!("    {}", line))
        .collect::<Vec<_>>()
        .join("\n");

    let template = r###"[TASK]
Extract detailed information from the provided Pug tbody into a single structured JSON object.

[PUG CONTENT]
{PUG_CONTENT}

[CONTEXT]
current Link: {HREF}
Language: {LANGUAGE}

[SCHEMA DEFINITIONS]
{SCHEMA}

[OUTPUT FORMAT]
{...}

[ACTION] RETURN JSON ONLY. 
NO EXPLANATION. NO THINKING. /no_think"###;

    // 🌟 {COMBINED_INPUT} 대신 {INPUT_SECTION}으로 정확히 치환합니다.
    template.replace("{SCHEMA}", &schema)
            .replace("{TYPE}", page_type)
            .replace("{HREF}", href)
            .replace("{LANGUAGE}", language)
            .replace("{PUG_CONTENT}", &pug_content)
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







pub fn get_trade_doc_classification_prompt() -> String {
    // TRACKING을 선택지에 명시적으로 추가
    r###"Classify document type. Choose strictly from: PI, CI, BL, AWB, PL, CO, LC, TRACKING, Unknown. 
Return JSON exactly like: {"doc_type": "BL"}
NO EXPLANATION."###.to_string()
}

pub fn get_trade_category_schema(category: &str, doc_type: &str) -> String {
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

// ==========================================
// [수정] 무역 문서(Trade Doc) 검색을 위한 Shipping Condition 업그레이드
// ==========================================
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
JSON ONLY. NO EXPLANATION. /no_think"###;

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
{ "tracking_number": "string", "recipient_match": boolean, "barcodes": ["string"] }"###;
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
Your task is to analyze the [Reference: Row Structure] and locate both its body container and its corresponding header container within the [PUG CONTENT].
Generate the exact CSS selectors for both containers.

[Rules]
1. Analyze Reference: Carefully examine the [Reference: Row Structure] to understand the context, structure, and attributes of a single data item/row.
2. Locate Body Container: Scan the [PUG CONTENT] to find the exact wrapper containing these data rows. Generate a precise CSS selector for this body container.
3. Locate Header Container: Identify the wrapper or container element that acts as the "Header" for this list (the element containing the column titles or labels). Generate a precise CSS selector for this header container.
4. Tag Agnostic: Do NOT assume the structure uses traditional <table>, <thead>, or <tbody> tags. It could be built using <div>, <ul>/<li>, or other semantic tags. Analyze the relationship logically.
5. Strict JSON Output: Output the result strictly in valid JSON format exactly matching the structure below. Do not include any other text, markdown formatting, or explanations.

[Expected Output Format]
{
  "{TYPE}" : {
    "table" : {
      "tbody" : {
        "selector" : "{ITEM_SELECTOR}"
      },
      "thead" : {
        "selector" : "..."
      }
    }
  }
}

[ACTION] RETURN JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{TYPE}", page_type)
            .replace("{ITEM_SELECTOR}", item_selector)
            .replace("{PUG_CONTENT}", pug_content)
            .replace("{REFERENCE_ROW}", reference_row)
}
use scraper::{Html, Node, Selector};
use ego_tree::NodeRef;
use regex::Regex;

#[derive(PartialEq, Clone, Copy)]
pub enum PugMode {
    StructureOnly,
    FullContent,
}

pub fn pre_clean_html(html: &str) -> String {
    // 1. 주석 제거
    let re_comm = Regex::new(r"(?s)<!--.*?-->").unwrap();
    let html = re_comm.replace_all(html, "");

    // 2. 분류/추출에 전혀 필요 없는 태그들 통째로 제거 (Proxy 로직 반영)
    // 역참조(\1)를 지원하지 않으므로 각 태그를 명시적으로 처리하거나 합쳐서 처리
    let re_tags = Regex::new(r"(?is)<(script|style|svg|noscript|iframe|head|meta|link|canvas)\b[^>]*>.*?</(script|style|svg|noscript|iframe|head|meta|link|canvas)>").unwrap();
    let html = re_tags.replace_all(&html, "");

    // 3. 닫는 태그가 없는 단일 태그들 정리
    let re_single = Regex::new(r"(?is)<(meta|link|br|hr|img)\b[^>]*>").unwrap();
    // 분류 단계에서 img는 맥락을 해칠 수 있으므로 제거하거나 최소화
    let clean = re_single.replace_all(&html, |caps: &regex::Captures| {
        let tag = caps.get(1).map(|m| m.as_str().to_lowercase()).unwrap_or_default();
        if tag == "img" { 
            String::new() 
        } else { 
            caps.get(0).map(|m| m.as_str().to_string()).unwrap_or_default()
        }
    });

    // 4. 연속된 줄바꿈 및 불필요한 공백 제거 (토큰 절약)
    let re_whitespace = Regex::new(r"(?m)^\s*\n").unwrap();
    let clean = re_whitespace.replace_all(&clean, "");
    
    clean.trim().to_string()
}

pub fn convert_doc_to_clean_pug(document: &Html, mode: PugMode) -> String {
    let mut pug_output = String::new();
    pug_output.reserve(1024 * 10); 
    let mut found_body = false;
    for child in document.tree.root().children() {
        if let Some(element) = child.value().as_element() {
            if element.name() == "html" {
                 for html_child in child.children() {
                     if let Some(body_el) = html_child.value().as_element() {
                         if body_el.name() == "body" {
                             generate_pug_lines(html_child, 0, &mut pug_output, &mode);
                             found_body = true;
                         }
                     }
                 }
            }
        }
    }
    if !found_body {
         generate_pug_lines(document.tree.root(), 0, &mut pug_output, &mode);
    }
    pug_output
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
    for node in document.tree.root().descendants() {
        if let Some(element_ref) = scraper::ElementRef::wrap(node) {
            if selector.matches(&element_ref) {
                 generate_pug_lines(node, 0, &mut pug_output, &mode);
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

fn generate_pug_lines(node: NodeRef<scraper::Node>, indent_level: usize, output: &mut String, mode: &PugMode) {
    if indent_level > 50 { return; }
    let indent = "    ".repeat(indent_level);
    match node.value() {
        Node::Element(element) => {
            let tag_name = element.name();
            if ["script", "style", "link", "noscript", "iframe", "svg", "path", "meta", "head", "symbol", "defs", "use"].contains(&tag_name) { return; }
            if tag_name == "img" {
                if let Some(src) = element.attr("src") {
                    if src.len() > 1000 || src.contains("base64") { return; }
                }
            }
            let mut line = String::with_capacity(64);
            line.push_str(&indent);
            line.push_str(tag_name);
            if let Some(id) = element.id() { 
                line.push('#');
                line.push_str(id);
            }
            for class in element.classes() { 
                line.push('.');
                line.push_str(class);
            }
            let mut attrs = Vec::new();
            match mode {
                PugMode::StructureOnly => {
                    if let Some(val) = element.attr("href") { 
                         let val = if val.len() > 200 { &val[..200] } else { val };
                         attrs.push(format!("href='{}'", val)); 
                    }
                },
                PugMode::FullContent => {
                    for (key, value) in element.attrs() {
                        if key == "id" || key == "class" { continue; }
                        if key.starts_with("data-") || ["src", "href", "type", "name", "value", "placeholder", "title", "alt", "checked", "selected", "disabled", "readonly", "rows", "cols", "action", "method"].contains(&key) {
                             if ["checked", "selected", "disabled", "readonly"].contains(&key) {
                                 attrs.push(key.to_string());
                             } else {
                                 let val_clean = value.replace("\"", "'" ).replace("\n", "");
                                 let val_trunc = if val_clean.len() > 300 { format!("{}\"...", &val_clean[..300]) } else { val_clean };
                                 attrs.push(format!(r#"{}='{}'"#, key, val_trunc));
                             }
                        }
                    }
                }
            }
            if !attrs.is_empty() { 
                line.push('(');
                line.push_str(&attrs.join(" "));
                line.push(')');
            }
            output.push_str(&line);
            output.push('\n');

            // --- [STRICT PARITY] Special Textarea Handling ---
            if tag_name == "textarea" {
                if let Some(val) = element.attr("value") {
                    for line in val.lines() {
                        output.push_str(&indent);
                        output.push_str("    | ");
                        output.push_str(line.trim());
                        output.push('\n');
                    }
                }
            }

            // --- Structural Compression for Classification ---
            let children: Vec<_> = node.children().collect();
            if *mode == PugMode::StructureOnly && children.len() > 5 {
                let mut last_tag = "";
                let mut last_classes = String::new();
                let mut repeat_count = 0;
                
                for (idx, child) in children.iter().enumerate() {
                    let is_repetitive = if let Some(el) = child.value().as_element() {
                        let current_classes = el.classes().collect::<Vec<_>>().join(".");
                        if el.name() == last_tag && current_classes == last_classes {
                            true
                        } else {
                            last_tag = el.name();
                            last_classes = current_classes;
                            false
                        }
                    } else { false };

                    if is_repetitive { repeat_count += 1; } else { repeat_count = 0; }

                    if repeat_count > 1 && idx < children.len() - 1 {
                        if repeat_count == 2 {
                            output.push_str(&indent);
                            output.push_str("    | ... (repetitive items compressed)\n");
                        }
                        continue; 
                    }
                    generate_pug_lines(*child, indent_level + 1, output, mode);
                }
            } else {
                for child in node.children() { generate_pug_lines(child, indent_level + 1, output, mode); }
            }
        },
        Node::Text(text) => {
            if *mode == PugMode::FullContent {
                let content = text.trim();
                if !content.is_empty() {
                    let content_trunc = if content.len() > 1000 { &content[..1000] } else { content };
                    for line in content_trunc.lines() {
                        let trimmed_line = line.trim();
                        if !trimmed_line.is_empty() {
                            output.push_str(&indent);
                            output.push_str("| ");
                            output.push_str(&trimmed_line.replace("\"", "'" ));
                            output.push('\n');
                        }
                    }
                }
            }
        },
        _ => { for child in node.children() { generate_pug_lines(child, indent_level, output, mode); } }
    }
}

pub fn split_doc_to_pug_list(document: &Html, selector_str: &str, mode: PugMode) -> Vec<String> {
    let selector = match Selector::parse(selector_str) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let mut pug_list = Vec::new();
    for node in document.tree.root().descendants() {
        if let Some(element_ref) = scraper::ElementRef::wrap(node) {
            if selector.matches(&element_ref) {
                 let mut pug_output = String::new();
                 pug_output.reserve(2048);
                 generate_pug_lines(node, 0, &mut pug_output, &mode);
                 if !pug_output.trim().is_empty() {
                     pug_list.push(pug_output);
                 }
            }
        }
    }
    pug_list
}

pub fn split_html_to_pug_list(html: &str, selector_str: &str, mode: PugMode) -> Vec<String> {
    let document = Html::parse_document(html);
    split_doc_to_pug_list(&document, selector_str, mode)
}

pub fn page_type_prompt(language: &str) -> String {
    let template = r###"
[STRICT ROLE]
You are a web structure classifier. Your ONLY job is to output a JSON object.

[CATEGORIES]
type:'order' or 'goods' or 'tracking' or 'review' or 'coupon' or 'event' or '',

[OUTPUT RULE]
- No conversation. No repetition of input. 
- Output ONLY: {"type": "category"}

[LANGUAGE]
{LANGUAGE}"###;
    template.replace("{LANGUAGE}", language)
}

pub fn page_selectors_prompt(page_type: &str, language: &str) -> String {
    let template = r###"
[TASK]
The page has been classified as '{TYPE}'.
Analyze the provided Pug template snippet and identify the structural CSS1 selectors required for data extraction.

[SCHEMA DEFINITIONS]
- item: type based item CSS1 selector excluding ads
- node: item parent list CSS1 selector excluding ads
- more: item URL includes a manage path, an administrative or edit route Link CSS1 selector
- next: list next button CSS1 selector
- detail: is a detail page or a detail form

[OUTPUT FORMAT]
Return valid JSON only. No explanation.
{{
    "item": "",
    "node": "",
    "more": "",
    "next": "",
    "detail": boolean
}}

[CONTEXT]
Language: {LANGUAGE}"###;
    template.replace("{TYPE}", page_type).replace("{LANGUAGE}", language)
}

pub fn para2graph(language: &str) -> String {
    format!(r###"convert the natural language content to fit the dataset JSON structure. no explanation.
	{{
		"context" : [
			{{
				"language" : "{}",
				"type": "sales" | "order" | "goods" | "tracking" | "view" | "review" | "coupon" | "event" | "",
				"text": "Segment the natural language content into single-type contexts"
			}}
		]
	}}"###, language)
}

pub fn graph2contexts(current: &str) -> String {
    format!(r###"convert the natural language content to fit the dataset JSON structure. no explanation.
	# #date : The date value is set by referencing both the natural language's implied time period and the region value against the current time ({}); it will be marked as null if a value is absent
	# #status : 'progress' | 'stop' | 'cancel' | 'refund' | 'return' | 'exchange' | 'expire' | 'complete' | 'error'
	# #substantial : 'size' | 'weight' | 'shipping_fee' | 'shipping_duration' | 'sale_price' | 'supply_price' | 'low_stock_threshold' | 'discount' | 'min_order_amount' | 'max_discount_amount' | 'usage_limit' | 'usage_per' | ''
	# #find : 'many' | 'few' | 'much' | 'little' | 'heavy' | 'light' | ''
    
    [TASK]
    Analyze EACH provided text segment and extract structured conditions. 
    
    [OUTPUT FORMAT]
    Return a JSON object containing a "context" array with one object per segment.
    {{
        "context": [
            {{
                "type": "string",
                "text": "the_original_segment_text",
                "status": "string or null",
                "substantial": "string or null",
                "find": "string or null",
                "condition" : {{
                    "date": {{ "eq": "yyyy-MM-ddThh:mm:ss", "lte": "yyyy-MM-ddThh:mm:ss", "gte": "yyyy-MM-ddThh:mm:ss" }},
                    "quantity": {{ "eq": number, "lte": number, "gte": number }},
                    "price": {{ "currency": "string", "eq": number, "lte": number, "gte": number }}
                }}
            }}
        ]
    }}"###, current)
}

pub fn item2json(page_type: &str, href: &str, language: &str) -> String {
    let schema = match page_type {
        "tracking" => r###"node:tracking form container CSS1 selector,
status:{
    value:'draft' or 'progress' or 'return' or 'complete' or 'error',
    selector:selector
},
id:{
    value:tracking number | string,
    selector:selector
},
title:{
    value:tracking goods title | string,
    selector:selector
} 
sender_name:{
    value:sender_name | string,
    selector:selector
},
sender_address:{
    value:sender_address | string,
    selector:selector
},
sender_phone:{
    value:sender_phone | string,
    selector:selector
},
recipient_name:{
    value:recipient_name | string,
    selector:selector
},
recipient_address:{
    value:recipient_address | string,
    selector:selector
},
recipient_phone:{
    value:recipient_phone | string,
    selector:selector
},
package_width:{
    value:Package width | number,
    selector:selector
},
package_height:{
    value:Package height | number,
    selector:selector
},
package_length:{
    value:Package length | number,
    selector:selector
},
package_weight:{
    value:Package weight | number,
    selector:selector
},
carrier:{
    value:carrier name translated into English | string,
    selector:selector
},
shipping_fee:{
    value:Shipping cost | number,
    selector:selector
},
shipping_method:{
    value:'standard' or 'express' or 'same_day' or 'pick_up' or 'freight' or 'prepaid',
    selector:selector
},
shipping_duration:{
    value:Estimated delivery days | number,
    selector:selector
},
bundle_shipping:{
    value:Allow combined shipping | string,
    selector:selector
},
shipping_date:{
    value:yyyy-MM-ddThh:mm:ss | string,
    selector:selector
},
registration_date:{
    value:yyyy-MM-ddThh:mm:ss | string,
    selector:selector
},"###,
        "goods" => r###"node:goods form container CSS1 selector,
code:{
    value:product constant code | string,
    selector:selector
},
link:'{HREF}',
id:{
    value:Refer to the ID value from the link or an attribute or input value | string,
    selector:selector
},
status:{
    value:'draft' or 'show' or 'hide' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
    selector:selector
},
payment_method:{
    value:payment method | string,
    selector:selector
},
bank:{
    value:bank company name or '' | string,
    selector:selector
},
card:{
    value:card company name or '' | string,
    selector:selector
},
model_name:{
    value:product Model name | string,
    selector:selector
},
brand_name:{
    value:product Brand name | string,
    selector:selector
},
condition:{
    value:['new' or 'used' or 'lease' or 'rental' or 'refurbish'],
    selector:selector
},
description:{
    value:product Full description (HTML allowed) | string,
    selector:selector
},
short_description:{
    value:product short description | string,
    selector:selector
},
tags:{
    value:[{ tag : product keyword or tag | string }],
    selector:selector
},
origin_country:{
    value:product Country of origin/manufacture | string,
    selector:selector
},
manufacturer:{
    value:product Manufacturer name | string,
    selector:selector
},
release_date:{
    value:Product release date(yyyy-MM-ddThh:mm:ss) | string,
    selector:selector
},
manufacture_date:{
    value:product Date(yyyy-MM-ddThh:mm:ss) of manufacture | string,
    selector:selector
},
expiration_date:{
    value:product Expiration or use-by date(yyyy-MM-ddThh:mm:ss) | string,
    selector:selector
},
gtin:{
    value:product Global Trade Item Number | string,
    selector:selector
},
mpn:{
    value:product Manufacturer Part Number | string,
    selector:selector
},
barcode:{
    value:product Barcode value | string,
    selector:selector
},
sale_price:{
    value:product sale price | number,
    selector:selector
},
supply_price:{
    value:product supply price | number,
    selector:selector
},
currency:{
    value:ISO 4217 Currency Code | string,
    selector:selector
},
compare_at_price:{
    value:product Original price for showing discounts | number,
    selector:selector
},
quantity:{
    value:product Inventory quantity | number,
    selector:selector
},
stock_keeping_unit:{
    value:Stock Keeping Unit | string,
    selector:selector
},
low_stock_threshold:{
    value:product Low stock alert threshold | number,
    selector:selector
},
unit:{
    value:product Selling unit | string,
    selector:selector
},
tax_included:{
    value:product Whether tax | number,
    selector:selector
},
tax_code:{
    value:product Tax code for region-specific rules | string,
    selector:selector
},
main_image_url:{
    value:Main product image URL | string,
    selector:selector
},
additional_image_url:{
    value:additional product image URL | string,
    selector:selector
},
video_url:{
    value:product Promotional video URL | string,
    selector:selector
},
carrier:{
    value:product carrier name translated into English | string,
    selector:selector
},
shipping_fee:{
    value:product Shipping cost | number,
    selector:selector
},
shipping_method:{
    value:'standard' or 'express' or 'same_day' or 'pick_up' or 'freight' or 'prepaid',
    selector:selector
},
shipping_duration:{
    value:product Estimated delivery days | number,
    selector:selector
},
bundle_shipping:{
    value:product Allow combined shipping | string,
    selector:selector
},
product_width:{
    value:Package width(cm) | number,
    selector:selector
},
product_height:{
    value:Package height(cm) | number,
    selector:selector
},
product_length:{
    value : Package length(cm) | number,
    selector:selector
},
product_weight:{
    value : Package weight(kg) | number,
    selector:selector
},
options:[
    {
        value:option name | string,
        selector:selector,
        inputs:[{
            value:option input value | string,
            selector:selector
        }]
    }
],
additional_goods:[
    {
        value:URL includes a manage path, an administrative or edit route product Link | string,
        selector:selector
    }
],
title:{
    value:product based title | string,
    selector:selector
},
registration_date:{
    value:yyyy-MM-ddThh:mm:ss | string,
    selector:selector
},"###,
        "order" => r###"node:order form container CSS1 selector,
link : '{HREF}',
id:{
    value:Refer to the ID value from the link or an attribute or input value | string,
    selector:selector
},
tracking_number:{
    value:tracking number | string,
    selector:selector
},
status:{
    value:'draft' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
    selector:selector
},
goods:[{
    title:{
        value:goods title | string,
        selector:selector
    },
    link:{
        value:URL includes a manage path, an administrative or edit route goods Link | string,
        selector:selector
    },
    id:{
        value:Refer to the product no value from the link or an attribute or input value | string,
        selector:selector
    }
}],
sender_name:{
    value:sender_name | string,
    selector:selector
},
sender_address:{
    value:sender_address, Filter the addresses to District-level and up | string,
    selector:selector
},
sender_phone:{
    value:sender_phone | string,
    selector:selector
},
recipient_name:{
    value:recipient_name | string,
    selector:selector
},
recipient_address:{
    value:recipient_address, Filter the addresses to District-level and up | string,
    selector:selector
},
recipient_phone:{
    value:recipient_phone | string,
    selector:selector
},
bank:{
    value:bank company name | string,
    selector:selector
},
card:{
    value:card company name | string,
    selector:selector
},
order_date:{
    value:order date | string,
    selector:selector
},
payment_date:{
    value:payment date or '' | string,
    selector:selector
},
payment_method:{
    value:'C.O.D.' or 'CARD' or 'BANK' or '' | string,
    selector:selector
},
payment_origin:{
    value:Payment Gateway Service Name or '' | string,
    selector:selector
},
registration_date:{
    value:yyyy-MM-ddThh:mm:ss | string,
    selector:selector
},"###,
        "coupon" | "event" => format!(r###"node:{} container CSS1 selector,
link : '{}',
id:{{
    value:Refer to the ID value from the link or an attribute or input value | string,
    selector:selector
}},
type:{{
    value:'percentage' or 'fixed_amount' or 'free_shipping' or '',
    selector:selector
}},
status:{{
    value:'draft' or 'progress' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error',
    selector:selector
}},
title:{{
    value:{} title | string, 
    selector:selector
}},
started_at:{{
    value:yyyy-MM-ddThh:mm:ss | string,
    selector:selector
}},
expired_at:{{
    value:yyyy-MM-ddThh:mm:ss | string,
    selector:selector
}},
code:{{
    value:{} code used at checkout | string,
    selector:selector
}},
discount:{{
    value:Discount value | number,
    selector:selector
}},
quantity:{{
    value:{} quantity | number
    selector:selector
}},
usage_limit:{{
    value:Total usage limit for the coupon | number,
    selector:selector
}},
usage_per:{{
    value:Usage limit per customer | number,
    selector:selector
}},
new_customer_only:{{
    value:new customer only | boolean
    selector:selector
}},
min_order_amount:{{
    value:Minimum order amount required to apply coupon | number,
    selector:selector
}},
max_discount_amount:{{
    value:Maximum discount limit allowed for the coupon | number,
    selector:selector
}},
region_restrictions:{{
    value:region restrictions | boolean,
    selector:selector
}},
registration_date:{{
    value:yyyy-MM-ddThh:mm:ss | string,
    selector:selector
}},"###, page_type, href, page_type, page_type, page_type),
        "review" => format!(r###"node:review container CSS1 selector,
link : '{}',
id:Refer to the ID value from the link or an attribute or input value | string,,
status:{{
    value:'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
    selector:selector
}},
name:{{
    value:reviewer name | string,
    selector:selector
}},
title:{{
    value:reviewer item title | string, 
    selector:selector
}},
completed:{{
    value:order complete | boolean,
    selector:selector
}},
registration_date:{{
    value:yyyy-MM-ddThh:mm:ss | string,
    selector:selector
}},"###, href),
        _ => "- id: Unique identifier.\n- title: General name or title.\n- status: Current state."
    };

    let template = r###" 
[TASK]
Extract detailed information from the provided Pug template into a single structured JSON object.

[CONTEXT]
Page Type: {TYPE}
Source URL: {HREF}

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
    "field_name": "extracted_value"
}"###;

    template.replace("{TYPE}", page_type)
            .replace("{HREF}", href)
            .replace("{LANGUAGE}", language)
            .replace("{SCHEMA}", schema)
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
tracking_number:Tracking Number or 운송장 번호 or 运단호 or 運單號 or 伝표번호 or Número de seguimiento or Numéro de suivi or Sendungsnummer or Но머 나кладной or Número de rastreamento or Numero di tracciamento or رقم التتبع or Số vận đơn or Nomor resi or หมายเลขติดตามพัสดุ | string,
registration_date:yyyy-MM-ddThh:mm:ss | string,"###,
        "tracking" | "review" => r###"status:'start' or 'progress' or 'stop' or 'cancel' or 'return',
id:Refer to the ID value from the link or an attribute | string,
title:author and content | string, 
link:URL includes a manage path, an administrative or edit route Link | string,
registration_date:yyyy-MM-ddThh:mm:ss | string,"###,
        "coupon" | "event" => r###"status:'show' or 'progress' or 'hide' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error',
id:Refer to the ID value from the link or an attribute | string,
title:type based item title, 
started_at:yyyy-MM-ddThh:mm:ss,
expired_at:yyyy-MM-ddThh:mm:ss,
registration_date:yyyy-MM-ddThh:mm:ss | string,"###,
        _ => "id: ID\ntitle: Title\nstatus: Status"
    };

    let template = r###" 
[TASK]
Extract a list of items from the provided Pug snippets into a JSON object matching the schema.

[CONTEXT]
Category: {TYPE}
Language: {LANGUAGE}

[SCHEMA DEFINITIONS]
{SCHEMA}

[EXTRACTION RULES]
1. Return a JSON object containing the fields defined in [SCHEMA DEFINITIONS].
2. The "items" field must be an array of objects.
3. If data for a field is missing, use null.
4. Extract only numeric values for prices and quantities.
5. Return ONLY valid JSON. No explanation.

[OUTPUT FORMAT]
{
    "type": "...",
    "items": [
        { "id": "...", "title": "...", "status": "..." }
    ]
}"###;

    template.replace("{TYPE}", page_type)
            .replace("{LANGUAGE}", language)
            .replace("{SCHEMA}", schema)
}

/// Converts a JSON Value into a human-readable natural language narrative.
/// [STRICT ALIGNMENT] This logic perfectly synchronizes with every column in `parsing.rs`.
pub fn json_to_natural_language(value: &serde_json::Value) -> String {
    let mut output = String::new();
    
    // Recursive handling for nested structures like { "value": "..." }
    if let Some(obj) = value.as_object() {
        if obj.len() == 1 && obj.contains_key("value") {
            return obj.get("value").unwrap().as_str().unwrap_or(&obj.get("value").unwrap().to_string()).to_string();
        }
    }

    if let serde_json::Value::Object(map) = value {
        let page_type = map.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let is_detail = map.get("detail").and_then(|v| v.as_bool()).unwrap_or(true);

        // Define EXACT columns from parsing.rs
        let keys: Vec<&str> = match page_type {
            "tracking" => {
                if is_detail {
                    vec!["status", "id", "title", "sender_name", "sender_address", "sender_phone", "recipient_name", "recipient_address", "recipient_phone", "package_width", "package_height", "package_length", "package_weight", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "bundle_shipping", "shipping_date", "registration_date"]
                } else {
                    vec!["status", "id", "title", "link", "registration_date"]
                }
            },
            "goods" => {
                if is_detail {
                    vec!["code", "link", "id", "status", "payment_method", "bank", "card", "model_name", "brand_name", "condition", "description", "short_description", "tags", "origin_country", "manufacturer", "release_date", "manufacture_date", "expiration_date", "gtin", "mpn", "barcode", "sale_price", "supply_price", "currency", "compare_at_price", "quantity", "stock_keeping_unit", "low_stock_threshold", "unit", "tax_included", "tax_code", "main_image_url", "additional_image_url", "video_url", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "bundle_shipping", "product_width", "product_height", "product_length", "product_weight", "options", "additional_goods", "title", "registration_date"]
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
                    vec!["link", "id", "type", "status", "title", "started_at", "expired_at", "code", "discount", "quantity", "usage_limit", "usage_per", "new_customer_only", "min_order_amount", "max_discount_amount", "region_restrictions", "registration_date"]
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

pub fn parse_json_from_llm(text: &str) -> serde_json::Value {
    if let Ok(v) = serde_json::from_str(text) { return v; }
    if let Some(start) = text.find("{") {
        if let Some(end) = text.rfind("}") {
            if start < end {
                if let Ok(v) = serde_json::from_str(&text[start..=end]) { return v; }
            }
        }
    }
    if let Some(start) = text.find("[") {
        if let Some(end) = text.rfind("]") {
            if start < end {
                if let Ok(v) = serde_json::from_str(&text[start..=end]) { return v; }
            }
        }
    }
    serde_json::json!({})
}

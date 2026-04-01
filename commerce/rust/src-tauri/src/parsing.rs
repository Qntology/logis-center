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

    let re_comm = Regex::new(r"(?s)<!--.*?-->").unwrap();

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

                generate_pug_lines(child, 0, &mut pug_output, &mode);

                found_body = true;

                break;

            }

        }

    }

        if !found_body {

            for child in document.tree.root().children() {

                generate_pug_lines(child, 0, &mut pug_output, &mode);

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



pub fn generate_pug_lines(node: NodeRef<scraper::Node>, indent_level: usize, output: &mut String, mode: &PugMode) {
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

            // --- 허용된 속성만 Pug 문법으로 변환 ---
            let mut attributes_string = String::new();
            let mut other_attributes = Vec::new();

            // ID 속성 처리
            if let Some(id) = element.id() {
                other_attributes.push(format!("id=\"{}\"", id));
            }

            // Class 속성 처리 [class="class1 class2"]
            if let Some(classes) = element.attr("class") {
                if !classes.is_empty() {
                    other_attributes.push(format!("class=\"{}\"", classes));
                }
            }

            // 필수 속성 정의
            let always_include = [
                "src", "href", "type", "name", "value", "placeholder", 
                "checked", "selected", "disabled", "readonly", "rows", "cols"
            ];

            for (name, value) in element.attrs() {
                if name == "id" || name == "class" { continue; }

                if name.starts_with("data-") || always_include.contains(&name) {
                    // Boolean 속성 처리
                    if ["checked", "selected", "disabled", "readonly"].contains(&name) && (value.is_empty() || value == name) {
                        other_attributes.push(name.to_string());
                    } else if !value.is_empty() {
                        let safe_value = value.replace("\"", "'");
                        other_attributes.push(format!("{}=\"{}\"", name, safe_value));
                    }
                }
            }

            // 브래킷으로 묶는 속성들 추가
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

            // 태그 이름과 변환된 속성 문자열을 함께 추가 (원래 노드 기준)
            output.push_str(&format!("{}{}{}\n", indent, tag_name, attributes_string));

            // textarea의 값 처리
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
                // 자식 노드 처리 (축약된 노드의 자식들 탐색)
                for child in current_node.children() {
                    generate_pug_lines(child, indent_level + 1, output, mode);
                }
            }
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
{ "order" : [ {"title" : String} ] }

[ACTION] RETURN JSON ONLY.
NO EXPLANATION. NO THINKING. /no_think"###;
    template.replace("{TYPE}", page_type)
}

pub fn page_selectors_prompt(page_type: &str) -> String {
    let template = r###"[TASK]
The page has been classified as '{TYPE}'.
Based on the provided Pug template, identify the structural CSS selectors for the MAIN data list.

[INSTRUCTION]
1. Exclude nav/header patterns navigation menus entirely (e.g., id/class containing gnb, lnb, tnb, header, footer, sidebar).
2. IDENTIFY the repeating sibling items that represent the actual {TYPE} records.
3. LOCATE the main content container.

[SCHEMA DEFINITIONS]
- item: The repeating child element. Match recurring item patterns. Exclude nav/header
- node: The direct parent wrapper of the recurring items. Exclude nav/header.
- detail: is a detail page or a detail form. Exclude nav/header

[OUTPUT FORMAT]
{
    item: String,
    node: String,
    detail: Boolean
}"###;
    template.replace("{TYPE}", page_type)
}

pub fn para2graph(language: &str) -> String {
    let template = r###"convert the natural language content to fit the dataset JSON structure. no explanation.
	{
		"context" : [
			{
				"language" : "{LANG}",
				"type": "sales" | "order" | "goods" | "tracking" | "view" | "review" | "coupon" | "event" | "",
				"text": "Segment the natural language content into single-type contexts"
			}
		]
	}"###;
    template.replace("{LANG}", language)
}

pub fn graph2contexts(current: &str) -> String {
    let template = r###"convert the natural language content to fit the dataset JSON structure. no explanation.
	# #date : The date value is set by referencing both the natural language's implied time period and the region value against the current time ({CURRENT}); it will be marked as null if a value is absent
	# #status : 'progress' | 'stop' | 'cancel' | 'refund' | 'return' | 'exchange' | 'expire' | 'complete' | 'error'
	# #substantial : 'size' | 'weight' | 'shipping_fee' | 'shipping_duration' | 'sale_price' | 'supply_price' | 'low_stock_threshold' | 'discount' | 'min_order_amount' | 'max_discount_amount' | 'usage_limit' | 'usage_per' | ''
	# #find : 'many' | 'few' | 'much' | 'little' | 'heavy' | 'light' | ''
    
    [TASK]
    Analyze EACH provided text segment and extract structured conditions. 
    
    [OUTPUT FORMAT]
    Return a JSON object containing a "context" array with one object per segment.
    {
        "context": [
            {
                "type": "string",
                "text": "the_original_segment_text",
                "status": "string or null",
                "substantial": "string or null",
                "find": "string or null",
                "condition" : {
                    "date": { "eq": "yyyy-MM-ddThh:mm:ss", "lte": "yyyy-MM-ddThh:mm:ss", "gte": "yyyy-MM-ddThh:mm:ss" },
                    "quantity": { "eq": number, "lte": number, "gte": number },
                    "price": { "currency": "string", "eq": number, "lte": number, "gte": number }
                }
            }
        ]
    }"###;
    template.replace("{CURRENT}", current)

    
}

pub fn item2json(page_type: &str, href: &str, language: &str) -> String {
    let schema = match page_type {
        "tracking" => r###"node:tracking form container CSS selector,
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
},"###.to_string(),
        "goods" => r###"node:goods form container CSS selector,
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
},"###.replace("{HREF}", href),
        "order" => r###"node:order form container CSS selector,
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
},"###.replace("{HREF}", href),
        "coupon" | "event" => r###"node:{TYPE} container CSS selector,
link : '{HREF}',
id:{
    value:Refer to the ID value from the link or an attribute or input value | string,
    selector:selector
},
type:{
    value:'percentage' or 'fixed_amount' or 'free_shipping' or '',
    selector:selector
},
status:{
    value:'draft' or 'progress' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error',
    selector:selector
},
title:{
    value:{TYPE} title | string, 
    selector:selector
},
started_at:{
    value:yyyy-MM-ddThh:mm:ss | string,
    selector:selector
},
expired_at:{
    value:yyyy-MM-ddThh:mm:ss | string,
    selector:selector
},
code:{
    value:{TYPE} code used at checkout | string,
    selector:selector
},
discount:{
    value:Discount value | number,
    selector:selector
},
quantity:{
    value:{TYPE} quantity | number
    selector:selector
},
usage_limit:{
    value:Total usage limit for the coupon | number,
    selector:selector
},
usage_per:{
    value:Usage limit per customer | number,
    selector:selector
},
new_customer_only:{
    value:new customer only | boolean
    selector:selector
},
min_order_amount:{
    value:Minimum order amount required to apply coupon | number,
    selector:selector
},
max_discount_amount:{
    value:Maximum discount limit allowed for the coupon | number,
    selector:selector
},
region_restrictions:{
    value:region restrictions | boolean,
    selector:selector
},
registration_date:{
    value:yyyy-MM-ddThh:mm:ss | string,
    selector:selector
},"###.replace("{TYPE}", page_type).replace("{HREF}", href),
        "review" => r###"node:review container CSS selector,
link : '{HREF}',
id:Refer to the ID value from the link or an attribute or input value | string,,
status:{
    value:'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
    selector:selector
},
name:{
    value:reviewer name | string,
    selector:selector
},
title:{
    value:reviewer item title | string, 
    selector:selector
},
completed:{
    value:order complete | boolean,
    selector:selector
},
registration_date:{
    value:yyyy-MM-ddThh:mm:ss | string,
    selector:selector
},"###.replace("{HREF}", href),
        _ => "- id: Unique identifier.\n- title: General name or title.\n- status: Current state.".to_string()
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
tracking_number:Tracking Number or 운송장 번호 or 运单号 or 運單號 or 伝票번호 or Número de seguimiento or Numéro de suivi or Sendungsnummer or Номер накладной or Número de rastreamento or Numero di tracciamento or رقم التتبع or Số vận đơn or Nomor resi or หมายเลขติดตามพัสดุ | string,
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

    let template = r###" 
[TASK]
Extract detailed information from the provided Pug template into a single structured JSON object.

[CONTEXT]
Category: {TYPE}
Language: {LANGUAGE}

[SCHEMA DEFINITIONS]
{SCHEMA}

[EXTRACTION RULES]
1. Return ONLY valid JSON. No preamble, no postscript.
2. If a field is missing in the data, use null.
3. Normalize all dates to 'yyyy-MM-ddThh:mm:ss'.
4. Extract only numeric values for price, quantity, amount.
5. Do NOT make up data. Only extract what is present in the Pug structure.

[OUTPUT FORMAT]
{
    "field_name": "extracted_value"
}"###;

    template.replace("{TYPE}", page_type)
            .replace("{LANGUAGE}", language)
            .replace("{SCHEMA}", &schema)
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

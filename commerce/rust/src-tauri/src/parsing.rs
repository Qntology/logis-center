use scraper::{Html, Node, Selector};
use ego_tree::NodeRef;
use regex::Regex;

#[derive(PartialEq, Clone, Copy)]
pub enum PugMode {
    StructureOnly,
    FullContent,
}

pub fn pre_clean_html(html: &str) -> String {
    let re = Regex::new(r"(?is)(<!--.*?-->)|(<script[^>]*>.*?</script>)|(<style[^>]*>.*?</style>)|(<svg[^>]*>.*?</svg>)|(<noscript[^>]*>.*?</noscript>)").unwrap();
    re.replace_all(html, "").to_string()
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
                    let content_trunc = if content.len() > 2000 { &content[..2000] } else { content };
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
[TASK]
Analyze the provided Pug template snippet and identify the primary category of this webpage.

[SCHEMA DEFINITIONS]
- type: The main category of the page content. Must be one of:
  - 'order': Order history, order details, checkout success.
  - 'goods': Product list, product detail, shopping cart.
  - 'tracking': Shipment tracking status, delivery history.
  - 'review': Product reviews, feedback list.
  - 'coupon': Coupon list, discount events.
  - 'event': Promotion pages, event announcements.
  - '': If none of the above match or the page is irrelevant (e.g., login, setting).

[OUTPUT FORMAT]
Return valid JSON only. No explanation.
{{
    "type": "category_string"
}}

[CONTEXT]
Language: {LANGUAGE}"###;
    template.replace("{LANGUAGE}", language)
}

pub fn page_selectors_prompt(page_type: &str, language: &str) -> String {
    let template = r###"
[TASK]
The page has been classified as '{TYPE}'.
Analyze the provided Pug template snippet and identify the structural CSS1 selectors required for data extraction.

[SCHEMA DEFINITIONS]
- item: CSS1 selector for individual items in a list (e.g., `tr`, `li`, `div.product-item`). Exclude header rows, ads, or pagination.
- node: CSS1 selector for the parent container that holds the list of items (e.g., `tbody`, `ul`, `div.grid`).
- detail: Boolean. Set to `true` if this is a single item detail page (e.g., order detail, product detail). Set to `false` if it is a list page.

[OUTPUT FORMAT]
Return valid JSON only. No explanation.
{{
    "item": "css_selector_string",
    "node": "css_selector_string",
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
        "tracking" => r###"- status: 'draft' | 'progress' | 'return' | 'complete' | 'error'
- id: tracking number | string
- title: tracking goods title | string
- sender_name: sender_name | string
- sender_address: sender_address | string
- sender_phone: sender_phone | string
- recipient_name: recipient_name | string
- recipient_address: recipient_address | string
- recipient_phone: recipient_phone | string
- package_width: Package width | number
- package_height: Package height | number
- package_length: Package length | number
- package_weight: Package weight | number
- carrier: carrier name translated into English | string
- shipping_fee: Shipping cost | number
- shipping_method: 'standard' | 'express' | 'same_day' | 'pick_up' | 'freight' | 'prepaid'
- shipping_duration: Estimated delivery days | number
- bundle_shipping: Allow combined shipping | string
- shipping_date: yyyy-MM-ddThh:mm:ss | string
- registration_date: yyyy-MM-ddThh:mm:ss | string"###,
        "goods" => r###"- node: goods form container CSS1 selector
- code: product constant code | string
- link: '{HREF}'
- id: Refer to the ID value from the link or an attribute or input value | string
- status: 'draft' | 'show' | 'hide' | 'progress' | 'stop' | 'cancel' | 'refund' | 'return' | 'exchange' | 'expire' | 'complete' | 'error'
- payment_method: payment method | string
- bank: bank company name | '' | string
- card: card company name | '' | string
- model_name: product Model name | string
- brand_name: product Brand name | string
- condition: ['new' | 'used' | 'lease' | 'rental' | 'refurbish']
- description: product Full description (HTML allowed) | string
- short_description: product short description | string
- tags: [{ tag : product keyword or tag | string }]
- origin_country: product Country of origin/manufacture | string
- manufacturer: product Manufacturer name | string
- release_date: Product release date(yyyy-MM-ddThh:mm:ss) | string
- manufacture_date: product Date(yyyy-MM-ddThh:mm:ss) of manufacture | string
- expiration_date: product Expiration or use-by date(yyyy-MM-ddThh:mm:ss) | string
- gtin: product Global Trade Item Number | string
- mpn: product Manufacturer Part Number | string
- barcode: product Barcode value | string
- sale_price: product sale price | number
- supply_price: product supply price | number
- currency: ISO 4217 Currency Code | string
- compare_at_price: product Original price for showing discounts | number
- quantity: product Inventory quantity | number
- stock_keeping_unit: Stock Keeping Unit | string
- low_stock_threshold: product Low stock alert threshold | number
- unit: product Selling unit | string
- tax_included: product Whether tax | number
- tax_code: product Tax code for region-specific rules | string
- main_image_url: Main product image URL | string
- additional_image_url: additional product image URL | string
- video_url: product Promotional video URL | string
- carrier: product carrier name translated into English | string
- shipping_fee: product Shipping cost | number
- shipping_method: 'standard' | 'express' | 'same_day' | 'pick_up' | 'freight' | 'prepaid'
- shipping_duration: product Estimated delivery days | number
- bundle_shipping: product Allow combined shipping | string
- product_width: Package width(cm) | number
- product_height: Package height(cm) | number
- product_length: Package length(cm) | number
- product_weight: Package weight(kg) | number
- options: [{ value:option name, inputs:[{ value:option input value }] }]
- additional_goods: [{ value:product Link }]
- title: product based title | string
- registration_date: yyyy-MM-ddThh:mm:ss | string"###,
        "order" => r###"- node: order form container CSS1 selector
- link: '{HREF}'
- id: Refer to the ID value from the link or an attribute or input value | string
- tracking_number: Tracking Number | 운송장 번호 | 运单호 | 運單號 | 伝표번호 | Número de seguimiento | Numéro de suivi | Sendungsnummer | Номер накладной | Número de rastreamento | Numero di tracciamento | رقم التتبع | Số vận đơn | Nomor resi | หมายเลขติดตามพัสดุ | string
- status: 'draft' | 'progress' | 'stop' | 'cancel' | 'refund' | 'return' | 'exchange' | 'expire' | 'complete' | 'error'
- goods: [{ title:{ value:goods title }, link:{ value:goods Link }, id:{ value:product no } }]
- sender_name: sender_name | string
- sender_address: sender_address, Filter the addresses to District-level and up | string
- sender_phone: sender_phone | string
- recipient_name: recipient_name | string
- recipient_address: recipient_address, Filter the addresses to District-level and up | string
- recipient_phone: recipient_phone | string
- bank: bank company name | string
- card: card company name | string
- order_date: order date | string
- payment_date: payment date or '' | string
- payment_method: 'C.O.D.' | 'CARD' | 'BANK' | '' | string
- payment_origin: Payment Gateway Service Name | '' | string
- registration_date: yyyy-MM-ddThh:mm:ss | string"###,
        "coupon" | "event" => r###"- node: {TYPE} container CSS1 selector
- link: '{HREF}'
- id: Refer to the ID value from the link or an attribute or input value | string
- type: 'percentage' | 'fixed_amount' | 'free_shipping' | ''
- status: 'draft' | 'progress' | 'stop' | 'cancel' | 'expire' | 'complete' | 'error'
- title: {TYPE} title | string
- started_at: yyyy-MM-ddThh:mm:ss | string
- expired_at: yyyy-MM-ddThh:mm:ss | string
- code: {TYPE} code used at checkout | string
- discount: Discount value | number
- quantity: {TYPE} quantity | number
- usage_limit: Total usage limit for the coupon | number
- usage_per: Usage limit per customer | number
- new_customer_only: new customer only | boolean
- min_order_amount: Minimum order amount required to apply coupon | number
- max_discount_amount: Maximum discount limit allowed for the coupon | number
- region_restrictions: region restrictions | boolean
- registration_date: yyyy-MM-ddThh:mm:ss | string"###,
        "review" => r###"- node: review container CSS1 selector
- link: '{HREF}'
- id: Refer to the ID value from the link or an attribute or input value | string
- status: 'progress' | 'stop' | 'cancel' | 'refund' | 'return' | 'exchange' | 'expire' | 'complete' | 'error'
- name: reviewer name | string
- title: reviewer item title | string
- completed: order complete | boolean
- registration_date: yyyy-MM-ddThh:mm:ss | string"###,
        _ => "- id: Unique identifier.\n- title: General name or title.\n- status: Current state."
    };

    let template = r###" 
[TASK]
Extract detailed information from the provided Pug template into a single structured JSON object.

[CONTEXT]
Page Type: {TYPE}
Source URL: {HREF}
Language: {LANGUAGE}

[SCHEMA DEFINITIONS]
{SCHEMA}

[EXTRACTION RULES]
1. Return ONLY valid JSON. No preamble, no postscript.
2. If a field is missing in the data, use null.
3. Normalize all dates to 'yyyy-MM-ddThh:mm:ss'.
4. Extract only numeric values for price, amount, weight, and dimensions.
5. Do NOT make up data. Only extract what is present in the Pug structure.

[OUTPUT FORMAT]
{{ 
    "field_name": "extracted_value"
}}"###;

    template.replace("{TYPE}", page_type)
            .replace("{HREF}", href)
            .replace("{LANGUAGE}", language)
            .replace("{SCHEMA}", schema)
}

pub fn list2json(page_type: &str, language: &str) -> String {
    let schema = match page_type {
        "order" | "goods" => r###"- item: type based item CSS1 selector excluding ads
- more: item URL includes a manage path, an administrative or edit route Link CSS1 selector
- node: item parent list CSS1 selector excluding ads
- next: list next button CSS1 selector
- status: 'show' | 'progress' | 'remove' | 'hide' | 'stop' | 'cancel' | 'refund' | 'return' | 'exchange' | 'expire' | 'complete' | 'error'
- id: Refer to the ID value from the link | an attribute | string
- title: title | string
- sale_price: sale price | number
- supply_price: supply price | number
- currency: ISO 4217 Currency Code | string
- quantity: item stock quantity | number
- tracking_number: Tracking Number | 운송장 번호 | 运单호 | 運單號 | 伝표번호 | Número de seguimiento | Numéro de suivi | Sendungsnummer | Номер накладной | Número de rastreamento | Numero di tracciamento | رقم التتبع | Số vận đơn | Nomor resi | หมายเลขติดตามพัสดุ | string
- link: URL includes a manage path, an administrative | edit route Link | string
- registration_date: yyyy-MM-ddThh:mm:ss | string"###,
        "tracking" | "review" => r###"- item: type based item CSS1 selector excluding ads
- more: item URL includes a manage path, an administrative or edit route Link CSS1 selector
- node: item parent list CSS1 selector excluding ads
- next: list next button CSS1 selector
- status: 'start' | 'progress' | 'stop' | 'cancel' | 'return'
- id: Refer to the ID value from the link | an attribute | string
- title: author and content | string
- link: URL includes a manage path, an administrative | edit route Link | string
- registration_date: yyyy-MM-ddThh:mm:ss | string"###,
        "coupon" | "event" => r###"- item: type based item CSS1 selector excluding ads
- more: item URL includes a manage path, an administrative or edit route Link CSS1 selector
- node: item parent list CSS1 selector excluding ads
- next: list next button CSS1 selector
- status: 'show' | 'progress' | 'hide' | 'stop' | 'cancel' | 'expire' | 'complete' | 'error'
- id: Refer to the ID value from the link | an attribute | string
- title: type based item title
- started_at: yyyy-MM-ddThh:mm:ss | string
- expired_at: yyyy-MM-ddThh:mm:ss | string
- registration_date: yyyy-MM-ddThh:mm:ss | string"###,
        _ => "- id: ID\n- title: Title\n- status: Status"
    };

    let template = r###" 
[TASK]
Extract a list of items from the provided Pug snippets into a JSON ARRAY.

[CONTEXT]
Category: {TYPE}
Language: {LANGUAGE}

[SCHEMA DEFINITIONS]
{SCHEMA}

[EXTRACTION RULES]
1. Return a JSON ARRAY [ {{...}}, {{...}} ] containing objects for each item found.
2. Ensure every object in the array follows the [SCHEMA DEFINITIONS].
3. If data for a field is missing for an item, use null.
4. Extract only numeric values for prices and quantities.
5. Return ONLY the JSON array. No explanation.

[OUTPUT FORMAT]
[
    {{ "id": "...", "title": "...", "status": "..." }}
]"###;

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

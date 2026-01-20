use scraper::{Html, Node, Selector};
use ego_tree::NodeRef;
use regex::Regex;

#[derive(PartialEq)]
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

            // --- Structural Compression for Classification (StructureOnly mode) ---
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

                    if is_repetitive {
                        repeat_count += 1;
                    } else {
                        repeat_count = 0;
                    }

                    // Keep first 2 of a repetitive sequence, and the very last child
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
{
    "type": "category_string"
}

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
{
    "item": "css_selector_string",
    "node": "css_selector_string",
    "detail": boolean
}

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

pub fn item2json(page_type: &str, href: &str, _language: &str) -> Vec<(String, String)> {
    let mut fields = Vec::new();
    match page_type {
        "tracking" => {
            fields.push(("status".to_string(), "'draft' | 'progress' | 'return' | 'complete' | 'error'".to_string()));
            fields.push(("id".to_string(), "tracking number | string".to_string()));
            fields.push(("title".to_string(), "tracking goods title | string".to_string()));
            fields.push(("sender_name".to_string(), "sender_name | string".to_string()));
            fields.push(("sender_address".to_string(), "sender_address | string".to_string()));
            fields.push(("sender_phone".to_string(), "sender_phone | string".to_string()));
            fields.push(("recipient_name".to_string(), "recipient_name | string".to_string()));
            fields.push(("recipient_address".to_string(), "recipient_address | string".to_string()));
            fields.push(("recipient_phone".to_string(), "recipient_phone | string".to_string()));
            fields.push(("package_width".to_string(), "Package width | number".to_string()));
            fields.push(("package_height".to_string(), "Package height | number".to_string()));
            fields.push(("package_length".to_string(), "Package length | number".to_string()));
            fields.push(("package_weight".to_string(), "Package weight | number".to_string()));
            fields.push(("carrier".to_string(), "carrier name translated into English | string".to_string()));
            fields.push(("shipping_fee".to_string(), "Shipping cost | number".to_string()));
            fields.push(("shipping_method".to_string(), "'standard' | 'express' | 'same_day' | 'pick_up' | 'freight' | 'prepaid'".to_string()));
            fields.push(("shipping_duration".to_string(), "Estimated delivery days | number".to_string()));
            fields.push(("bundle_shipping".to_string(), "Allow combined shipping | string".to_string()));
            fields.push(("shipping_date".to_string(), "yyyy-MM-ddThh:mm:ss | string".to_string()));
            fields.push(("registration_date".to_string(), "yyyy-MM-ddThh:mm:ss | string".to_string()));
        },
        "goods" => {
            fields.push(("node".to_string(), "goods form container CSS1 selector".to_string()));
            fields.push(("code".to_string(), "product constant code | string".to_string()));
            fields.push(("link".to_string(), format!("'{}'", href)));
            fields.push(("id".to_string(), "Refer to the ID value from the link or an attribute or input value | string".to_string()));
            fields.push(("status".to_string(), "'draft' | 'show' | 'hide' | 'progress' | 'stop' | 'cancel' | 'refund' | 'return' | 'exchange' | 'expire' | 'complete' | 'error'".to_string()));
            fields.push(("payment_method".to_string(), "payment method | string".to_string()));
            fields.push(("bank".to_string(), "bank company name | '' | string".to_string()));
            fields.push(("card".to_string(), "card company name | '' | string".to_string()));
            fields.push(("model_name".to_string(), "product Model name | string".to_string()));
            fields.push(("brand_name".to_string(), "product Brand name | string".to_string()));
            fields.push(("condition".to_string(), "['new' | 'used' | 'lease' | 'rental' | 'refurbish']".to_string()));
            fields.push(("description".to_string(), "product Full description (HTML allowed) | string".to_string()));
            fields.push(("short_description".to_string(), "product short description | string".to_string()));
            fields.push(("tags".to_string(), "[{ tag : product keyword or tag | string }]".to_string()));
            fields.push(("origin_country".to_string(), "product Country of origin/manufacture | string".to_string()));
            fields.push(("manufacturer".to_string(), "product Manufacturer name | string".to_string()));
            fields.push(("release_date".to_string(), "Product release date(yyyy-MM-ddThh:mm:ss) | string".to_string()));
            fields.push(("manufacture_date".to_string(), "product Date(yyyy-MM-ddThh:mm:ss) of manufacture | string".to_string()));
            fields.push(("expiration_date".to_string(), "product Expiration or use-by date(yyyy-MM-ddThh:mm:ss) | string".to_string()));
            fields.push(("gtin".to_string(), "product Global Trade Item Number | string".to_string()));
            fields.push(("mpn".to_string(), "product Manufacturer Part Number | string".to_string()));
            fields.push(("barcode".to_string(), "product Barcode value | string".to_string()));
            fields.push(("sale_price".to_string(), "product sale price | number".to_string()));
            fields.push(("supply_price".to_string(), "product supply price | number".to_string()));
            fields.push(("currency".to_string(), "ISO 4217 Currency Code | string".to_string()));
            fields.push(("compare_at_price".to_string(), "product Original price for showing discounts | number".to_string()));
            fields.push(("quantity".to_string(), "product Inventory quantity | number".to_string()));
            fields.push(("stock_keeping_unit".to_string(), "Stock Keeping Unit | string".to_string()));
            fields.push(("low_stock_threshold".to_string(), "product Low stock alert threshold | number".to_string()));
            fields.push(("unit".to_string(), "product Selling unit | string".to_string()));
            fields.push(("tax_included".to_string(), "product Whether tax | number".to_string()));
            fields.push(("tax_code".to_string(), "product Tax code for region-specific rules | string".to_string()));
            fields.push(("main_image_url".to_string(), "Main product image URL | string".to_string()));
            fields.push(("additional_image_url".to_string(), "additional product image URL | string".to_string()));
            fields.push(("video_url".to_string(), "product Promotional video URL | string".to_string()));
            fields.push(("carrier".to_string(), "product carrier name translated into English | string".to_string()));
            fields.push(("shipping_fee".to_string(), "product Shipping cost | number".to_string()));
            fields.push(("shipping_method".to_string(), "'standard' | 'express' | 'same_day' | 'pick_up' | 'freight' | 'prepaid'".to_string()));
            fields.push(("shipping_duration".to_string(), "product Estimated delivery days | number".to_string()));
            fields.push(("bundle_shipping".to_string(), "product Allow combined shipping | string".to_string()));
            fields.push(("product_width".to_string(), "Package width(cm) | number".to_string()));
            fields.push(("product_height".to_string(), "Package height(cm) | number".to_string()));
            fields.push(("product_length".to_string(), "Package length(cm) | number".to_string()));
            fields.push(("product_weight".to_string(), "Package weight(kg) | number".to_string()));
            fields.push(("options".to_string(), "[{ value:option name, inputs:[{ value:option input value }] }]".to_string()));
            fields.push(("additional_goods".to_string(), "[{ value:product Link }]".to_string()));
            fields.push(("title".to_string(), "product based title | string".to_string()));
            fields.push(("registration_date".to_string(), "yyyy-MM-ddThh:mm:ss | string".to_string()));
        },
        "order" => {
            fields.push(("node".to_string(), "order form container CSS1 selector".to_string()));
            fields.push(("link".to_string(), format!("'{}'", href)));
            fields.push(("id".to_string(), "Refer to the ID value from the link or an attribute or input value | string".to_string()));
            fields.push(("tracking_number".to_string(), r#"Tracking Number | 운송장 번호 | 运单号 | 運單號 | 伝票번호 | Número de seguimiento | Numéro de suivi | Sendungsnummer | Номер накладной | Número de rastreamento | Numero di tracciamento | رقم التتبع | Số vận đơn | Nomor resi | หมายเลขติดตามพัสดุ | string"#.to_string()));
            fields.push(("status".to_string(), "'draft' | 'progress' | 'stop' | 'cancel' | 'refund' | 'return' | 'exchange' | 'expire' | 'complete' | 'error'".to_string()));
            fields.push(("goods".to_string(), "[{ title:{ value:goods title }, link:{ value:goods Link }, id:{ value:product no } }]".to_string()));
            fields.push(("sender_name".to_string(), "sender_name | string".to_string()));
            fields.push(("sender_address".to_string(), "sender_address, Filter the addresses to District-level and up | string".to_string()));
            fields.push(("sender_phone".to_string(), "sender_phone | string".to_string()));
            fields.push(("recipient_name".to_string(), "recipient_name | string".to_string()));
            fields.push(("recipient_address".to_string(), "recipient_address, Filter the addresses to District-level and up | string".to_string()));
            fields.push(("recipient_phone".to_string(), "recipient_phone | string".to_string()));
            fields.push(("bank".to_string(), "bank company name | string".to_string()));
            fields.push(("card".to_string(), "card company name | string".to_string()));
            fields.push(("order_date".to_string(), "order date | string".to_string()));
            fields.push(("payment_date".to_string(), "payment date or '' | string".to_string()));
            fields.push(("payment_method".to_string(), "'C.O.D.' | 'CARD' | 'BANK' | '' | string".to_string()));
            fields.push(("payment_origin".to_string(), "Payment Gateway Service Name | '' | string".to_string()));
            fields.push(("registration_date".to_string(), "yyyy-MM-ddThh:mm:ss | string".to_string()));
        },
        "coupon" | "event" => {
            fields.push(("node".to_string(), format!("{} container CSS1 selector", page_type)));
            fields.push(("link".to_string(), format!("'{}'", href)));
            fields.push(("id".to_string(), "Refer to the ID value from the link or an attribute or input value | string".to_string()));
            fields.push(("type".to_string(), "'percentage' | 'fixed_amount' | 'free_shipping' | ''".to_string()));
            fields.push(("status".to_string(), "'draft' | 'progress' | 'stop' | 'cancel' | 'expire' | 'complete' | 'error'".to_string()));
            fields.push(("title".to_string(), format!("{} title | string", page_type)));
            fields.push(("started_at".to_string(), "yyyy-MM-ddThh:mm:ss | string".to_string()));
            fields.push(("expired_at".to_string(), "yyyy-MM-ddThh:mm:ss | string".to_string()));
            fields.push(("code".to_string(), format!("{} code used at checkout | string", page_type)));
            fields.push(("discount".to_string(), "Discount value | number".to_string()));
            fields.push(("quantity".to_string(), format!("{} quantity | number", page_type)));
            fields.push(("usage_limit".to_string(), "Total usage limit for the coupon | number".to_string()));
            fields.push(("usage_per".to_string(), "Usage limit per customer | number".to_string()));
            fields.push(("new_customer_only".to_string(), "new customer only | boolean".to_string()));
            fields.push(("min_order_amount".to_string(), "Minimum order amount required to apply coupon | number".to_string()));
            fields.push(("max_discount_amount".to_string(), "Maximum discount limit allowed for the coupon | number".to_string()));
            fields.push(("region_restrictions".to_string(), "region restrictions | boolean".to_string()));
            fields.push(("registration_date".to_string(), "yyyy-MM-ddThh:mm:ss | string".to_string()));
        },
        "review" => {
            fields.push(("node".to_string(), "review container CSS1 selector".to_string()));
            fields.push(("link".to_string(), format!("'{}'", href)));
            fields.push(("id".to_string(), "Refer to the ID value from the link or an attribute or input value | string".to_string()));
            fields.push(("status".to_string(), "'progress' | 'stop' | 'cancel' | 'refund' | 'return' | 'exchange' | 'expire' | 'complete' | 'error'".to_string()));
            fields.push(("name".to_string(), "reviewer name | string".to_string()));
            fields.push(("title".to_string(), "reviewer item title | string".to_string()));
            fields.push(("completed".to_string(), "order complete | boolean".to_string()));
            fields.push(("registration_date".to_string(), "yyyy-MM-ddThh:mm:ss | string".to_string()));
        },
        _ => {} 
    }
    fields
}

pub fn list2json(page_type: &str, language: &str) -> Vec<(String, String)> {
    let mut fields = Vec::new();
    fields.push(("language".to_string(), format!("'{}'", language)));
    fields.push(("type".to_string(), format!("'{}'", page_type)));
    fields.push(("item".to_string(), "type based item CSS1 selector excluding ads".to_string()));
    fields.push(("more".to_string(), "item URL includes a manage path, an administrative or edit route Link CSS1 selector".to_string()));
    fields.push(("node".to_string(), "item parent list CSS1 selector excluding ads".to_string()));
    fields.push(("next".to_string(), "list next button CSS1 selector".to_string()));
    fields.push(("text".to_string(), format!("summarize the contents of the items array in {}", language)));
    fields.push(("detail".to_string(), "is a detail page or a detail form | boolean".to_string()));
    match page_type {
        "tracking" | "review" => {
            fields.push(("status".to_string(), "'start' | 'progress' | 'stop' | 'cancel' | 'return'".to_string()));
            fields.push(("id".to_string(), "Refer to the ID value from the link | an attribute | string".to_string()));
            fields.push(("title".to_string(), "author and content | string".to_string()));
            fields.push(("link".to_string(), "URL includes a manage path, an administrative | edit route Link | string".to_string()));
            fields.push(("registration_date".to_string(), "yyyy-MM-ddThh:mm:ss | string".to_string()));
        },
        "order" | "goods" => {
            fields.push(("status".to_string(), "'show' | 'progress' | 'remove' | 'hide' | 'stop' | 'cancel' | 'refund' | 'return' | 'exchange' | 'expire' | 'complete' | 'error'".to_string()));
            fields.push(("link".to_string(), "URL includes a manage path, an administrative | edit route Link | string".to_string()));
            fields.push(("id".to_string(), "Refer to the ID value from the link | an attribute | string".to_string()));
            fields.push(("title".to_string(), "title | string".to_string()));
            fields.push(("sale_price".to_string(), "sale price | number".to_string()));
            fields.push(("supply_price".to_string(), "supply price | number".to_string()));
            fields.push(("currency".to_string(), "ISO 4217 Currency Code | string".to_string()));
            fields.push(("quantity".to_string(), "item stock quantity | number".to_string()));
            fields.push(("tracking_number".to_string(), "Tracking Number | 운송장 번호 | 运单号 | 運單號 | 伝票번호 | Número de seguimiento | Numéro de suivi | Sendungsnummer | Номер накладной | Número de rastreamento | Numero di tracciamento | رقم التتبع | Số vận đơn | Nomor resi | หมายเลขติดตามพัสดุ | string".to_string()));
            fields.push(("registration_date".to_string(), "yyyy-MM-ddThh:mm:ss | string".to_string()));
        },
        "coupon" | "event" => {
            fields.push(("status".to_string(), "'show' | 'progress' | 'hide' | 'stop' | 'cancel' | 'expire' | 'complete' | 'error'".to_string()));
            fields.push(("id".to_string(), "Refer to the ID value from the link | an attribute | string".to_string()));
            fields.push(("title".to_string(), "type based item title".to_string()));
            fields.push(("started_at".to_string(), "yyyy-MM-ddThh:mm:ss | string".to_string()));
            fields.push(("expired_at".to_string(), "yyyy-MM-ddThh:mm:ss | string".to_string()));
            fields.push(("registration_date".to_string(), "yyyy-MM-ddThh:mm:ss | string".to_string()));
        },
        _ => {} 
    }
    fields
}
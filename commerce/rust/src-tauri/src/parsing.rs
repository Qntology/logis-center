use scraper::{Html, Node, Selector};
use ego_tree::NodeRef;
use regex::Regex;

#[derive(PartialEq, Clone, Copy)]
pub enum PugMode {
    StructureOnly,
    FullContent,
}

pub fn sanitize_llm_input(text: &str) -> String {
    let cleaned: String = text.chars()
        .filter(|c| {
            let u = *c as u32;
            (u >= 32 && u <= 126) || (u >= 0xAC00 && u <= 0xD7A3) || (u >= 0x1100 && u <= 0x11FF) || (u >= 0x3130 && u <= 0x318F) || u == 10 || u == 13 || u == 9
        })
        .collect();
    cleaned.replace("<|", "< |").replace("|>", "| >")
}

pub fn pre_clean_html(html: &str) -> String {
    let re_comm = Regex::new(r"(?s)<!--.*?-->").unwrap();
    let html = re_comm.replace_all(html, "");
    let re_tags = Regex::new(r"(?is)<(script|style|link|noscript|iframe)\b[^>]*>.*?</(script|style|link|noscript|iframe)>").unwrap();
    let html = re_tags.replace_all(&html, "");
    let re_single = Regex::new(r"(?is)<(meta|link|br|hr|source)\b[^>]*>").unwrap();
    let clean = re_single.replace_all(&html, "");
    let re_whitespace = Regex::new(r"(?m)^\s*\n").unwrap();
    let clean = re_whitespace.replace_all(&clean, "");
    clean.trim().to_string()
}

pub fn generate_pug_lines(node: NodeRef<scraper::Node>, indent_level: usize, output: &mut String, mode: &PugMode) {
    if indent_level > 30 { return; }
    let indent = "  ".repeat(indent_level);
    
    match node.value() {
        Node::Element(element) => {
            let tag_name = element.name().to_lowercase();
            if ["script", "style", "link", "noscript", "iframe"].contains(&tag_name.as_str()) { return; }

            let mut attributes_string = String::new();
            if let Some(id) = element.id() { attributes_string.push_str(&format!("#{}", id)); }
            
            if let Some(classes) = element.attr("class") {
                if let Some(first_class) = classes.split_whitespace().next() {
                    attributes_string.push_str(&format!(".{}", first_class));
                }
            }

            let mut cur_node = node;
            if tag_name == "div" {
                let children: Vec<_> = cur_node.children().collect();
                if children.len() == 1 {
                    if let Some(child_el) = children[0].value().as_element() {
                        if child_el.name().to_lowercase() == "div" {
                            generate_pug_lines(children[0], indent_level, output, mode);
                            return;
                        }
                    }
                }
            }

            output.push_str(&format!("{}{}{}\n", indent, tag_name, attributes_string));

            for child in cur_node.children() {
                generate_pug_lines(child, indent_level + 1, output, mode);
            }
        }
        Node::Text(text) => {
            if *mode == PugMode::FullContent {
                let text_content = text.trim();
                if !text_content.is_empty() {
                    let s = text_content.replace("\"", "'");
                    let truncated = if s.len() > 200 { format!("{}...", &s[..200]) } else { s };
                    output.push_str(&format!("{}| {}\n", indent, truncated));
                }
            }
        }
        _ => {}
    }
}

pub fn convert_doc_to_clean_pug(document: &Html, mode: PugMode) -> String {
    let mut pug_output = String::new();
    pug_output.reserve(1024 * 20);
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

pub fn convert_to_clean_pug_selector(html: &str, selector_str: &str, mode: PugMode) -> String {
    let document = Html::parse_document(html);
    let selector = match Selector::parse(selector_str) { Ok(s) => s, Err(_) => return String::new() };
    let mut pug_output = String::new();
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

pub fn page_type_prompt() -> String { r###"[TASK]
Based on the PREVIOUSLY RECORDED Pug template content, identify the primary category of this webpage.

[INSTRUCTION]
You have already received the page data in chunks. Analyze the entire history and determine the page type. 

[SCHEMA DEFINITIONS]
- type: The main category of the page content. Must be one of:
  - 'order': Order history, order details, checkout success.
  - 'goods': Product list, product detail, shopping cart.
  - 'tracking': Shipment tracking status, delivery history.
  - 'review': Product reviews, feedback list.
  - 'coupon': Coupon list, discount events.
  - 'event': Promotion pages, event announcements.
  - '': If none of the above match or the page is irrelevant

[OUTPUT FORMAT]
Return valid JSON only. No explanation.
{
    "type": "..."
}"###.to_string() }

pub fn page_selectors_prompt(page_type: &str) -> String {
    let template = r###"[TASK]
The page has been classified as '{TYPE}'.
Based on the PREVIOUSLY RECORDED history, identify the structural CSS1 selectors required for data extraction.

[INSTRUCTION]
Analyze the page structure you have already learned and find the repeating patterns.

[SCHEMA DEFINITIONS]
- item: Common CSS1 selector for sibling items (e.g., `tr`, `li`). Match recurring patterns and exclude header, footer, ads, and pagination.
- node: Common CSS1 selector for the main container wrapping all list items (e.g., tbody, ul). Focus on the direct parent of recurring rows/items, excluding header, footer, ads, and pagination.
- detail: is a detail page or a detail form. Exclude header, footer, ads, pagination.

[OUTPUT FORMAT]
Return valid JSON only. No explanation.
{
    "item": "...",
    "node": "...",
    "detail": boolean
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

pub fn item2json(page_type: &str, href: &str, _language: &str) -> String {
    format!("[TASK]\nExtract {} from {} into structured JSON. Return only JSON.", page_type, href)
}

pub fn list2json(page_type: &str, _language: &str) -> String {
    format!("[TASK]\nExtract a list of {} items from the provided Pug snippets into a JSON array. Return only JSON.", page_type)
}

pub fn json_to_natural_language(value: &serde_json::Value) -> String {
    value.to_string()
}

pub fn parse_json_from_llm(text: &str) -> serde_json::Value {
    serde_json::from_str(text).unwrap_or(serde_json::json!({}))
}

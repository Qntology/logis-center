use scraper::{Html, Node, Selector};
use ego_tree::NodeRef;

#[derive(PartialEq)]
pub enum PugMode {
    StructureOnly,
    FullContent,
}

pub fn convert_to_clean_pug(html: &str, mode: PugMode) -> String {
    let document = Html::parse_document(html);
    let mut pug_output = String::new();
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

pub fn convert_to_clean_pug_selector(html: &str, selector_str: &str, mode: PugMode) -> String {
    let document = Html::parse_document(html);
    let selector = match Selector::parse(selector_str) {
        Ok(s) => s,
        Err(_) => return String::new(),
    };
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

fn generate_pug_lines(node: NodeRef<scraper::Node>, indent_level: usize, output: &mut String, mode: &PugMode) {
    let indent = "    ".repeat(indent_level);
    match node.value() {
        Node::Element(element) => {
            let tag_name = element.name();
            if ["script", "style", "link", "noscript", "iframe", "svg", "path", "meta", "head"].contains(&tag_name) { return; }
            if tag_name == "img" {
                if let Some(src) = element.attr("src") {
                    if src.contains("base64") { return; }
                }
            }
            let mut line = format!("{}{}", indent, tag_name);
            let mut attrs = Vec::new();
            if let Some(id) = element.id() { line.push_str(&format!("#{}", id)); }
            for class in element.classes() { line.push_str(&format!(".{}", class)); }
            match mode {
                PugMode::StructureOnly => {
                    if let Some(val) = element.attr("href") { attrs.push(format!("href='{}'", val)); }
                },
                PugMode::FullContent => {
                    for (key, value) in element.attrs() {
                        if key == "id" || key == "class" { continue; }
                        if key.starts_with("data-") || ["src", "href", "type", "name", "value", "placeholder", "title", "alt", "checked", "selected", "disabled", "readonly", "rows", "cols", "action", "method"].contains(&key) {
                             if ["checked", "selected", "disabled", "readonly"].contains(&key) {
                                 attrs.push(key.to_string());
                             } else {
                                 let safe_val = value.replace("\"", "'" ).replace("\n", "");
                                 attrs.push(format!(r#"{}='{}'"#, key, safe_val));
                             }
                        }
                    }
                }
            }
            if !attrs.is_empty() { line.push_str(&format!("({})", attrs.join(" "))); }
            output.push_str(&line);
            output.push('\n');
            for child in node.children() { generate_pug_lines(child, indent_level + 1, output, mode); }
        },
        Node::Text(text) => {
            if *mode == PugMode::FullContent {
                let content = text.trim();
                if !content.is_empty() {
                    for line in content.lines() {
                        let trimmed_line = line.trim();
                        if !trimmed_line.is_empty() {
                            output.push_str(&format!("{}| {}
", indent, trimmed_line.replace("\"", "'" )));
                        }
                    }
                }
            }
        },
        _ => {
            for child in node.children() { generate_pug_lines(child, indent_level, output, mode); }
        }
    }
}

pub fn split_html_to_pug_list(html: &str, selector_str: &str, mode: PugMode) -> Vec<String> {
    let document = Html::parse_document(html);
    let selector = match Selector::parse(selector_str) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let mut pug_list = Vec::new();
    for node in document.tree.root().descendants() {
        if let Some(element_ref) = scraper::ElementRef::wrap(node) {
            if selector.matches(&element_ref) {
                 let mut pug_output = String::new();
                 generate_pug_lines(node, 0, &mut pug_output, &mode);
                 if !pug_output.trim().is_empty() {
                     pug_list.push(pug_output);
                 }
            }
        }
    }
    pug_list
}

pub fn map_outline(language: &str) -> String {
    let template = r###"Analyze the provided Pug template and return it in the following JSON format, no explanation.
    # type : 'order' or 'goods' or 'tracking' or 'review' or 'coupon' or 'event' or ''
    # item : type based item CSS1 selector excluding ads
    # node : item parent list CSS1 selector excluding ads
    # detail : is a detail page or a detail form
    Return JSON:
    {
        "type": '',
        "item": '',
        "node": '',
        "detail": boolean,
    }
    Language: {LANGUAGE}"###;
    template.replace("{LANGUAGE}", language)
}

pub fn image2json(region: &str, language: &str, page_type: &str, address: &str) -> String {
    if page_type == "tracking" {
        let template = r###"convert the shipping label image to fit the dataset JSON structure. Return only the JSON structure result, no explanation.
		#region : {REGION}
		#recipient_address : {ADDRESS}
		#tracking_number is selected from the number that matches the barcode or QR code, among others, based on the #region, excluding the format of a national telephone or mobile phone number, or an order number.
		{
			tracking_number:tracking number or 운송장 번호 or 송장 번호 or 송장번호 or 등기 번호 or 등기번호 or 运单号 or 運單號 or 伝票番号 or Número de seguimiento or Numéro de suivi or Sendungsnummer or Номер накладной or Número de rastreamento or Numero di tracciamento or رقم التتبع or Số vận đơn or Nomor resi or หมายเลขติดตามพัสดุ | string,
			recipient_match : shipping label #recipient_address match. Ruled the same despite different floor levels | boolean,
			barcodes : [barcode number | string] | array,
			text : summarize the shipping label contents in {LANGUAGE}. Masking the address in the summary to District-level and up. Do not mention that information is masked or partially hidden | string,
		}"###;
        template.replace("{REGION}", region).replace("{ADDRESS}", address).replace("{LANGUAGE}", language)
    } else {
        String::new()
    }
}

pub fn para2graph(language: &str) -> String {
    let template = r###"convert the natural language content to fit the dataset JSON structure. no explanation.
	{
		"context": [
			{
				"language": "{LANGUAGE}",
				"type": "sales" or "order" or "goods" or "tracking" or "view" or "review" or "coupon" or "event" or "",
				"text": "Segment the natural language content into single-type contexts"
			}
		]
	}"###;
    template.replace("{LANGUAGE}", language)
}

pub fn graph2contexts(current: &str) -> String {
    let template = r###"convert the natural language content to fit the dataset JSON structure. no explanation.
	# #date : The date value is set by referencing both the natural language's implied time period and the region value against the current time ({CURRENT}); it will be marked as null if a value is absent
	# #status : 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error'
	# #substantial : 'size' or 'weight' or 'shipping_fee' or 'shipping_duration' or 'sale_price' or 'supply_price' or 'low_stock_threshold' or 'discount' or 'min_order_amount' or 'max_discount_amount' or 'usage_limit' or 'usage_per' or ''
	# #find : 'many' or 'few' or 'much' or 'little' or 'heavy' or 'light' or ''"###;
    template.replace("{CURRENT}", current)
}

pub fn item2json(page_type: &str, href: &str, language: &str) -> Vec<(String, String)> {
    let mut fields = Vec::new();
    let base_prompt = format!("Analyze the provided Pug template. Extract ONLY the '{{}}' field information based on the schema below.\nReturn valid JSON: {{ \"{{}}\": {{ ... }} }}.\nLanguage: {}", language);

    match page_type {
        "tracking" => {
            fields.push(("status".to_string(), r#"\n            status:{\n                value:'draft' or 'progress' or 'return' or 'complete' or 'error',\n                selector:selector\n            }"#.to_string()));
            fields.push(("id".to_string(), r#"\n            id:{\n                value:tracking number | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("title".to_string(), r#"\n            title:{\n                value:tracking goods title | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("sender_name".to_string(), r#"\n            sender_name:{\n                value:sender_name | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("sender_address".to_string(), r#"\n            sender_address:{\n                value:sender_address | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("sender_phone".to_string(), r#"\n            sender_phone:{\n                value:sender_phone | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("recipient_name".to_string(), r#"\n            recipient_name:{\n                value:recipient_name | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("recipient_address".to_string(), r#"\n            recipient_address:{\n                value:recipient_address | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("recipient_phone".to_string(), r#"\n            recipient_phone:{\n                value:recipient_phone | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("package_width".to_string(), r#"\n            package_width:{\n                value:Package width | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("package_height".to_string(), r#"\n            package_height:{\n                value:Package height | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("package_length".to_string(), r#"\n            package_length:{\n                value:Package length | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("package_weight".to_string(), r#"\n            package_weight:{\n                value:Package weight | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("carrier".to_string(), r#"\n            carrier:{\n                value:carrier name translated into English | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("shipping_fee".to_string(), r#"\n            shipping_fee:{\n                value:Shipping cost | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("shipping_method".to_string(), r#"\n            shipping_method:{\n                value:'standard' or 'express' or 'same_day' or 'pick_up' or 'freight' or 'prepaid',\n                selector:selector\n            }"#.to_string()));
            fields.push(("shipping_duration".to_string(), r#"\n            shipping_duration:{\n                value:Estimated delivery days | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("bundle_shipping".to_string(), r#"\n            bundle_shipping:{\n                value:Allow combined shipping | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("shipping_date".to_string(), r#"\n            shipping_date:{\n                value:yyyy-MM-ddThh:mm:ss | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("registration_date".to_string(), r#"\n            registration_date:{\n                value:yyyy-MM-ddThh:mm:ss | string,\n                selector:selector\n            }"#.to_string()));
        },
        "goods" => {
            fields.push(("node".to_string(), r#"node:goods form container CSS1 selector"#.to_string()));
            fields.push(("code".to_string(), r#"\n            code:{\n                value:product constant code | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("link".to_string(), format!("link:'{}'", href)));
            fields.push(("id".to_string(), r#"\n            id:{\n                value:Refer to the ID value from the link or an attribute or input value | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("status".to_string(), r#"\n            status:{\n                value:'draft' or 'show' or 'hide' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',\n                selector:selector\n            }"#.to_string()));
            fields.push(("payment_method".to_string(), r#"\n            payment_method:{\n                value:payment method | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("bank".to_string(), r#"\n            bank:{\n                value:bank company name or '' | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("card".to_string(), r#"\n            card:{\n                value:card company name or '' | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("model_name".to_string(), r#"\n            model_name:{\n                value:product Model name | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("brand_name".to_string(), r#"\n            brand_name:{\n                value:product Brand name | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("condition".to_string(), r#"\n            condition:{\n                value:['new' or 'used' or 'lease' or 'rental' or 'refurbish'],\n                selector:selector\n            }"#.to_string()));
            fields.push(("description".to_string(), r#"\n            description:{\n                value:product Full description (HTML allowed) | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("short_description".to_string(), r#"\n            short_description:{\n                value:product short description | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("tags".to_string(), r#"\n            tags:{\n                value:[{ tag : product keyword or tag | string }],\n                selector:selector\n            }"#.to_string()));
            fields.push(("origin_country".to_string(), r#"\n            origin_country:{\n                value:product Country of origin/manufacture | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("manufacturer".to_string(), r#"\n            manufacturer:{\n                value:product Manufacturer name | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("release_date".to_string(), r#"\n            release_date:{\n                value:Product release date(yyyy-MM-ddThh:mm:ss) | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("manufacture_date".to_string(), r#"\n            manufacture_date:{\n                value:product Date(yyyy-MM-ddThh:mm:ss) of manufacture | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("expiration_date".to_string(), r#"\n            expiration_date:{\n                value:product Expiration or use-by date(yyyy-MM-ddThh:mm:ss) | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("gtin".to_string(), r#"\n            gtin:{\n                value:product Global Trade Item Number | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("mpn".to_string(), r#"\n            mpn:{\n                value:product Manufacturer Part Number | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("barcode".to_string(), r#"\n            barcode:{\n                value:product Barcode value | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("sale_price".to_string(), r#"\n            sale_price:{\n                value:product sale price | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("supply_price".to_string(), r#"\n            supply_price:{\n                value:product supply price | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("currency".to_string(), r#"\n            currency:{\n                value:ISO 4217 Currency Code | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("compare_at_price".to_string(), r#"\n            compare_at_price:{\n                value:product Original price for showing discounts | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("quantity".to_string(), r#"\n            quantity:{\n                value:product Inventory quantity | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("stock_keeping_unit".to_string(), r#"\n            stock_keeping_unit:{\n                value:Stock Keeping Unit | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("low_stock_threshold".to_string(), r#"\n            low_stock_threshold:{\n                value:product Low stock alert threshold | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("unit".to_string(), r#"\n            unit:{\n                value:product Selling unit | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("tax_included".to_string(), r#"\n            tax_included:{\n                value:product Whether tax | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("tax_code".to_string(), r#"\n            tax_code:{\n                value:product Tax code for region-specific rules | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("main_image_url".to_string(), r#"\n            main_image_url:{\n                value:Main product image URL | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("additional_image_url".to_string(), r#"\n            additional_image_url:{\n                value:additional product image URL | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("video_url".to_string(), r#"\n            video_url:{\n                value:product Promotional video URL | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("carrier".to_string(), r#"\n            carrier:{\n                value:product carrier name translated into English | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("shipping_fee".to_string(), r#"\n            shipping_fee:{\n                value:product Shipping cost | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("shipping_method".to_string(), r#"\n            shipping_method:{\n                value:'standard' or 'express' or 'same_day' or 'pick_up' or 'freight' or 'prepaid',\n                selector:selector\n            }"#.to_string()));
            fields.push(("shipping_duration".to_string(), r#"\n            shipping_duration:{\n                value:product Estimated delivery days | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("bundle_shipping".to_string(), r#"\n            bundle_shipping:{\n                value:product Allow combined shipping | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("product_width".to_string(), r#"\n            product_width:{\n                value:Package width(cm) | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("product_height".to_string(), r#"\n            product_height:{\n                value:Package height(cm) | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("product_length".to_string(), r#"\n            product_length:{\n                value : Package length(cm) | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("product_weight".to_string(), r#"\n            product_weight:{\n                value : Package weight(kg) | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("options".to_string(), r#"\n            options:[\n                {\n                    value:option name | string,\n                    selector:selector,\n                    inputs:[{\n                        value:option input value | string,\n                        selector:selector\n                    }]\n                }\n            ]"#.to_string()));
            fields.push(("additional_goods".to_string(), r#"\n            additional_goods:[\n                {\n                    value:URL includes a manage path, an administrative or edit route product Link | string,\n                    selector:selector\n                }\n            ]"#.to_string()));
            fields.push(("title".to_string(), r#"\n            title:{\n                value:product based title | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("registration_date".to_string(), r#"\n            registration_date:{\n                value:yyyy-MM-ddThh:mm:ss | string,\n                selector:selector\n            }"#.to_string()));
        },
        "order" => {
            fields.push(("node".to_string(), r#"node:order form container CSS1 selector"#.to_string()));
            fields.push(("link".to_string(), format!("link : '{}'", href)));
            fields.push(("id".to_string(), r#"\n            id:{\n                value:Refer to the ID value from the link or an attribute or input value | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("tracking_number".to_string(), r#"\n            tracking_number:{\n                value:tracking number | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("status".to_string(), r#"\n            status:{\n                value:'draft' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',\n                selector:selector\n            }"#.to_string()));
            fields.push(("goods".to_string(), r#"\n            goods:[{\n                title:{\n                    value:goods title | string,\n                    selector:selector\n                },\n                link:{\n                    value:URL includes a manage path, an administrative or edit route goods Link | string,\n                    selector:selector\n                },\n                id:{\n                    value:Refer to the product no value from the link or an attribute or input value | string,\n                    selector:selector\n                }\n            }]"#.to_string()));
            fields.push(("sender_name".to_string(), r#"\n            sender_name:{\n                value:sender_name | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("sender_address".to_string(), r#"\n            sender_address:{\n                value:sender_address, Filter the addresses to District-level and up | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("sender_phone".to_string(), r#"\n            sender_phone:{\n                value:sender_phone | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("recipient_name".to_string(), r#"\n            recipient_name:{\n                value:recipient_name | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("recipient_address".to_string(), r#"\n            recipient_address:{\n                value:recipient_address, Filter the addresses to District-level and up | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("recipient_phone".to_string(), r#"\n            recipient_phone:{\n                value:recipient_phone | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("bank".to_string(), r#"\n            bank:{\n                value:bank company name | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("card".to_string(), r#"\n            card:{\n                value:card company name | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("order_date".to_string(), r#"\n            order_date:{\n                value:order date | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("payment_date".to_string(), r#"\n            payment_date:{\n                value:payment date or '' | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("payment_method".to_string(), r#"\n            payment_method:{\n                value:'C.O.D.' or 'CARD' or 'BANK' or '' | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("payment_origin".to_string(), r#"\n            payment_origin:{\n                value:Payment Gateway Service Name or '' | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("registration_date".to_string(), r#"\n            registration_date:{\n                value:yyyy-MM-ddThh:mm:ss | string,\n                selector:selector\n            }"#.to_string()));
        },
        "coupon" | "event" => {
            fields.push(("node".to_string(), format!("node:{} container CSS1 selector", page_type)));
            fields.push(("link".to_string(), format!("link : '{}'", href)));
            fields.push(("id".to_string(), r#"\n            id:{\n                value:Refer to the ID value from the link or an attribute or input value | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("type".to_string(), r#"\n            type:{\n                value:'percentage' or 'fixed_amount' or 'free_shipping' or '',\n                selector:selector\n            }"#.to_string()));
            fields.push(("status".to_string(), r#"\n            status:{\n                value:'draft' or 'progress' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error',\n                selector:selector\n            }"#.to_string()));
            fields.push(("title".to_string(), format!(r#"\n            title:{{ \n                value:{} title | string, \n                selector:selector\n            }}"#, page_type)));
            fields.push(("started_at".to_string(), r#"\n            started_at:{\n                value:yyyy-MM-ddThh:mm:ss | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("expired_at".to_string(), r#"\n            expired_at:{\n                value:yyyy-MM-ddThh:mm:ss | string,\n                selector:selector\n            }"#.to_string()));
            fields.push(("code".to_string(), format!(r#"\n            code:{{ \n                value:{} code used at checkout | string,\n                selector:selector\n            }}"#, page_type)));
            fields.push(("discount".to_string(), r#"\n            discount:{\n                value:Discount value | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("quantity".to_string(), format!(r#"\n            quantity:{{ \n                value:{} quantity | number\n                selector:selector\n            }}"#, page_type)));
            fields.push(("usage_limit".to_string(), r#"\n            usage_limit:{\n                value:Total usage limit for the coupon | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("usage_per".to_string(), r#"\n            usage_per:{\n                value:Usage limit per customer | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("new_customer_only".to_string(), r#"\n            new_customer_only:{\n                value:new customer only | boolean\n                selector:selector\n            }"#.to_string()));
            fields.push(("min_order_amount".to_string(), r#"\n            min_order_amount:{\n                value:Minimum order amount required to apply coupon | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("max_discount_amount".to_string(), r#"\n            max_discount_amount:{\n                value:Maximum discount limit allowed for the coupon | number,\n                selector:selector\n            }"#.to_string()));
            fields.push(("region_restrictions".to_string(), r#"\n            region_restrictions:{\n                value:region restrictions | boolean,\n                selector:selector\n            }"#.to_string()));
            fields.push(("registration_date".to_string(), r#"\n            registration_date:{\n                value:yyyy-MM-ddThh:mm:ss | string,\n                selector:selector\n            }"#.to_string()));
        },
        "review" => {
            fields.push(("node".to_string(), format!("node:{} container CSS1 selector", page_type)));
            fields.push(("link".to_string(), format!("link : '{}'", href)));
            fields.push(("id".to_string(), r#"id:Refer to the ID value from the link or an attribute or input value | string,"#.to_string()));
            fields.push(("status".to_string(), r#"\n            status:{\n                value:'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',\n                selector:selector\n            }"#.to_string()));
            fields.push(("name".to_string(), format!(r#"\n            name:{{ \n                value:{} name | string,\n                selector:selector\n            }}"#, page_type)));
            fields.push(("title".to_string(), format!(r#"\n            title:{{ \n                value:{} item title | string, \n                selector:selector\n            }}"#, page_type)));
            fields.push(("completed".to_string(), r#"\n            completed:{\n                value:order complete | boolean,\n                selector:selector\n            }"#.to_string()));
            fields.push(("registration_date".to_string(), r#"\n            registration_date:{\n                value:yyyy-MM-ddThh:mm:ss | string,\n                selector:selector\n            }"#.to_string()));
        },
        _ => {} // Do nothing for other page types
    }

    fields.into_iter().map(|(k, schema)| {
        (k.clone(), base_prompt.replace("{}", &k).replace("{{}}", &k).replace("{{}}", &schema))
    }).collect()
}

pub fn list2json(language: &str) -> String {
    let template = r###"Analyze the provided Pug template and return it in the following JSON format, no explanation.
    {
        language: '{LANGUAGE}',
        type:'order' or 'goods' or 'tracking' or 'review' or 'coupon' or 'event' or '',
        item:type based item CSS1 selector excluding ads,
        more:item URL includes a manage path, an administrative or edit route Link CSS1 selector,
        node:item parent list CSS1 selector excluding ads,
        next:list next button CSS1 selector,
        text:summarize the contents of the items array in {LANGUAGE},
        detail:is a detail page or a detail form | boolean,
        items: [
            if (type is 'tracking' or 'review') {
                status:'start' or 'progress' or 'stop' or 'cancel' or 'return',
                id:Refer to the ID value from the link or an attribute | string,
                title:author and content | string, 
                link:URL includes a manage path, an administrative or edit route Link | string,
                registration_date:yyyy-MM-ddThh:mm:ss | string,
            }
            if (type is 'order' or 'goods') {
                status:'show' or 'progress' or 'remove' or 'hide' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
                link:URL includes a manage path, an administrative or edit route Link | string,
                id:Refer to the ID value from the link or an attribute | string,
                title:title | string, 
                sale_price:sale price | number,
                supply_price:supply price | number,
                currency:ISO 4217 Currency Code | string,
                quantity:item stock quantity | number,
                tracking_number:Tracking Number or 운송장 번호 or 运单号 or 運單號 or 伝票番号 or Número de seguimiento or Numéro de suivi or Sendungsnummer or Номер накладной or Número de rastreamento or Numero di tracciamento or رقم التتبع or Số vận đơn or Nomor resi or หมายเลขติดตามพัสดุ | string,
                registration_date:yyyy-MM-ddThh:mm:ss | string,
            }
            if (type is 'coupon' or 'event') {
                status:'show' or 'progress' or 'hide' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error',
                id:Refer to the ID value from the link or an attribute | string,
                title:type based item title, 
                started_at:yyyy-MM-ddThh:mm:ss,
                expired_at:yyyy-MM-ddThh:mm:ss,
                registration_date:yyyy-MM-ddThh:mm:ss | string,
            }
        ] 
    } 
    "###;
    template.replace("{LANGUAGE}", language)
}

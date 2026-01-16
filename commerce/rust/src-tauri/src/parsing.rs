use scraper::{Html, Node, Selector};
use ego_tree::NodeRef;

#[derive(PartialEq)]
pub enum PugMode {
    StructureOnly, // 1단계: 구조 파악 (ID, Class, Href 만 유지)
    FullContent,   // 2단계: 데이터 추출 (모든 주요 속성 포함)
}

/// HTML 전체를 변환
pub fn convert_to_clean_pug(html: &str, mode: PugMode) -> String {
    let document = Html::parse_document(html);
    let mut pug_output = String::new();

    // 문서의 루트부터 순회 시작 (html -> body 탐색)
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
    
    // 만약 html/body 구조가 파싱되지 않았다면 루트부터 시도
    if !found_body {
         generate_pug_lines(document.tree.root(), 0, &mut pug_output, &mode);
    }

    pug_output
}

/// 특정 Selector에 해당하는 요소만 추출하여 변환 (Zoom-In)
pub fn convert_to_clean_pug_selector(html: &str, selector_str: &str, mode: PugMode) -> String {
    let document = Html::parse_document(html);
    let selector = match Selector::parse(selector_str) {
        Ok(s) => s,
        Err(_) => return String::new(), // Invalid selector
    };

    let mut pug_output = String::new();
    
    for node in document.tree.root().descendants() {
        if let Some(element_ref) = scraper::ElementRef::wrap(node) {
            if selector.matches(&element_ref) {
                 generate_pug_lines(node, 0, &mut pug_output, &mode);
                 break; // Process only the first match
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

            if ["script", "style", "link", "noscript", "iframe", "svg", "path", "meta", "head"].contains(&tag_name) {
                return;
            }
            
            if tag_name == "img" {
                if let Some(src) = element.attr("src") {
                    if src.contains("base64") { return; }
                }
            }

            let mut line = format!("{}{}", indent, tag_name);
            let mut attrs = Vec::new();

            if let Some(id) = element.id() {
                line.push_str(&format!("#{{}}", id));
            }

            for class in element.classes() {
                line.push_str(&format!(".{{}}", class));
            }

            match mode {
                PugMode::StructureOnly => {
                    if let Some(val) = element.attr("href") {
                        attrs.push(format!("href='{{}}'", val));
                    }
                },
                PugMode::FullContent => {
                    for (key, value) in element.attrs() {
                        if key == "id" || key == "class" { continue; }
                        
                        if key.starts_with("data-") || [
                            "src", "href", "type", "name", "value", "placeholder", "title", "alt",
                            "checked", "selected", "disabled", "readonly", "rows", "cols", "action", "method"
                        ].contains(&key) {
                             if ["checked", "selected", "disabled", "readonly"].contains(&key) {
                                 attrs.push(key.to_string());
                             } else {
                                 let safe_val = value.replace("\"", "'" ).replace("\n", "");
                                 attrs.push(format!(r#"{{}}='{{}}'"#, key, safe_val));
                             }
                        }
                    }
                }
            }

            if !attrs.is_empty() {
                line.push_str(&format!("({{}})", attrs.join(" ")));
            }

            output.push_str(&line);
            output.push('\n');

            for child in node.children() {
                generate_pug_lines(child, indent_level + 1, output, mode);
            }
        },
        Node::Text(text) => {
            if *mode == PugMode::FullContent {
                let content = text.trim();
                if !content.is_empty() {
                    for line in content.lines() {
                        let trimmed_line = line.trim();
                        if !trimmed_line.is_empty() {
                            output.push_str(&format!("{}| {{}}\n", indent, trimmed_line.replace("\"", "'" )))
                        }
                    }
                }
            }
        },
        _ => {
            for child in node.children() {
                generate_pug_lines(child, indent_level, output, mode);
            }
        }
    }
}

pub fn map_outline(language: &str) -> String {
    let template = r###"Analyze the provided Pug template and return it in the following JSON format, no explanation.
    #type : document category
    Return JSON:
    {
        "type":'order' or 'goods' or 'tracking' or 'review' or 'coupon' or 'event' or '',
        "item": type based item CSS1 selector excluding ads,
        "node": item parent list CSS1 selector excluding ads,
        "detail": is a detail page or a detail form | boolean,
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
        template.replace("{REGION}", region)
                .replace("{ADDRESS}", address)
                .replace("{LANGUAGE}", language)
    } else {
        String::new()
    }
}

pub fn para2graph(language: &str) -> String {
    let template = r###"convert the natural language content to fit the dataset JSON structure. no explanation.
	{
		context : [
			{
				language : "{LANGUAGE}",
				type:'sales' or 'order' or 'goods' or 'tracking' or 'view' or 'review' or 'coupon' or 'event' or '',
				text:Segment the natural language content into single-type contexts
			},...
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
    let base_prompt = format!("Analyze the provided Pug template. Extract ONLY the '{{}}' field information based on the schema below.\nReturn valid JSON: {{ \"{{}}\": {{ ... }} }}.\nLanguage: {{}}", language);

    match page_type {
        "tracking" => {
            fields.push(("status".to_string(), r#" 
            status:{
                value:'draft' or 'progress' or 'return' or 'complete' or 'error',
                selector:selector
            }"#.to_string()));
            
            fields.push(("id".to_string(), r#" 
            id:{
                value:tracking number | string,
                selector:selector
            }"#.to_string()));

            fields.push(("title".to_string(), r#" 
            title:{
                value:tracking goods title | string,
                selector:selector
            }"#.to_string()));

            fields.push(("sender_name".to_string(), r#" 
            sender_name:{
                value:sender_name | string,
                selector:selector
            }"#.to_string()));

            fields.push(("sender_address".to_string(), r#" 
            sender_address:{
                value:sender_address | string,
                selector:selector
            }"#.to_string()));

            fields.push(("sender_phone".to_string(), r#" 
            sender_phone:{
                value:sender_phone | string,
                selector:selector
            }"#.to_string()));

            fields.push(("recipient_name".to_string(), r#" 
            recipient_name:{
                value:recipient_name | string,
                selector:selector
            }"#.to_string()));

            fields.push(("recipient_address".to_string(), r#" 
            recipient_address:{
                value:recipient_address | string,
                selector:selector
            }"#.to_string()));

            fields.push(("recipient_phone".to_string(), r#" 
            recipient_phone:{
                value:recipient_phone | string,
                selector:selector
            }"#.to_string()));

            fields.push(("package_width".to_string(), r#" 
            package_width:{
                value:Package width | number,
                selector:selector
            }"#.to_string()));

            fields.push(("package_height".to_string(), r#" 
            package_height:{
                value:Package height | number,
                selector:selector
            }"#.to_string()));

            fields.push(("package_length".to_string(), r#" 
            package_length:{
                value:Package length | number,
                selector:selector
            }"#.to_string()));

            fields.push(("package_weight".to_string(), r#" 
            package_weight:{
                value:Package weight | number,
                selector:selector
            }"#.to_string()));

            fields.push(("carrier".to_string(), r#" 
            carrier:{
                value:carrier name translated into English | string,
                selector:selector
            }"#.to_string()));

            fields.push(("shipping_fee".to_string(), r#" 
            shipping_fee:{
                value:Shipping cost | number,
                selector:selector
            }"#.to_string()));

            fields.push(("shipping_method".to_string(), r#" 
            shipping_method:{
                value:'standard' or 'express' or 'same_day' or 'pick_up' or 'freight' or 'prepaid',
                selector:selector
            }"#.to_string()));

            fields.push(("shipping_duration".to_string(), r#" 
            shipping_duration:{
                value:Estimated delivery days | number,
                selector:selector
            }"#.to_string()));

            fields.push(("bundle_shipping".to_string(), r#" 
            bundle_shipping:{
                value:Allow combined shipping | string,
                selector:selector
            }"#.to_string()));

            fields.push(("shipping_date".to_string(), r#" 
            shipping_date:{
                value:yyyy-MM-ddThh:mm:ss | string,
                selector:selector
            }"#.to_string()));

            fields.push(("registration_date".to_string(), r#" 
            registration_date:{
                value:yyyy-MM-ddThh:mm:ss | string,
                selector:selector
            }"#.to_string()));
        },
        "goods" => {
            fields.push(("node".to_string(), "node:goods form container CSS1 selector".to_string()));
            
            fields.push(("code".to_string(), r#" 
            code:{
                value:product constant code | string,
                selector:selector
            }"#.to_string()));

            fields.push(("link".to_string(), format!("link:'{{}}'", href)));

            fields.push(("id".to_string(), r#" 
            id:{
                value:Refer to the ID value from the link or an attribute or input value | string,
                selector:selector
            }"#.to_string()));

            fields.push(("status".to_string(), r#" 
            status:{
                value:'draft' or 'show' or 'hide' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
                selector:selector
            }"#.to_string()));

            fields.push(("payment_method".to_string(), r#" 
            payment_method:{
                value:payment method | string,
                selector:selector
            }"#.to_string()));

            fields.push(("bank".to_string(), r#" 
            bank:{
                value:bank company name or '' | string,
                selector:selector
            }"#.to_string()));

            fields.push(("card".to_string(), r#" 
            card:{
                value:card company name or '' | string,
                selector:selector
            }"#.to_string()));

            fields.push(("model_name".to_string(), r#" 
            model_name:{
                value:product Model name | string,
                selector:selector
            }"#.to_string()));

            fields.push(("brand_name".to_string(), r#" 
            brand_name:{
                value:product Brand name | string,
                selector:selector
            }"#.to_string()));

            fields.push(("condition".to_string(), r#" 
            condition:{
                value:['new' or 'used' or 'lease' or 'rental' or 'refurbish'],
                selector:selector
            }"#.to_string()));

            fields.push(("description".to_string(), r#" 
            description:{
                value:product Full description (HTML allowed) | string,
                selector:selector
            }"#.to_string()));

            fields.push(("short_description".to_string(), r#" 
            short_description:{
                value:product short description | string,
                selector:selector
            }"#.to_string()));

            fields.push(("tags".to_string(), r#" 
            tags:{
                value:[{ tag : product keyword or tag | string }],
                selector:selector
            }"#.to_string()));

            fields.push(("origin_country".to_string(), r#" 
            origin_country:{
                value:product Country of origin/manufacture | string,
                selector:selector
            }"#.to_string()));

            fields.push(("manufacturer".to_string(), r#" 
            manufacturer:{
                value:product Manufacturer name | string,
                selector:selector
            }"#.to_string()));

            fields.push(("release_date".to_string(), r#" 
            release_date:{
                value:Product release date(yyyy-MM-ddThh:mm:ss) | string,
                selector:selector
            }"#.to_string()));

            fields.push(("manufacture_date".to_string(), r#" 
            manufacture_date:{
                value:product Date(yyyy-MM-ddThh:mm:ss) of manufacture | string,
                selector:selector
            }"#.to_string()));

            fields.push(("expiration_date".to_string(), r#" 
            expiration_date:{
                value:product Expiration or use-by date(yyyy-MM-ddThh:mm:ss) | string,
                selector:selector
            }"#.to_string()));

            fields.push(("gtin".to_string(), r#" 
            gtin:{
                value:product Global Trade Item Number | string,
                selector:selector
            }"#.to_string()));

            fields.push(("mpn".to_string(), r#" 
            mpn:{
                value:product Manufacturer Part Number | string,
                selector:selector
            }"#.to_string()));

            fields.push(("barcode".to_string(), r#" 
            barcode:{
                value:product Barcode value | string,
                selector:selector
            }"#.to_string()));

            fields.push(("sale_price".to_string(), r#" 
            sale_price:{
                value:product sale price | number,
                selector:selector
            }"#.to_string()));

            fields.push(("supply_price".to_string(), r#" 
            supply_price:{
                value:product supply price | number,
                selector:selector
            }"#.to_string()));

            fields.push(("currency".to_string(), r#" 
            currency:{
                value:ISO 4217 Currency Code | string,
                selector:selector
            }"#.to_string()));

            fields.push(("compare_at_price".to_string(), r#" 
            compare_at_price:{
                value:product Original price for showing discounts | number,
                selector:selector
            }"#.to_string()));

            fields.push(("quantity".to_string(), r#" 
            quantity:{
                value:product Inventory quantity | number,
                selector:selector
            }"#.to_string()));

            fields.push(("stock_keeping_unit".to_string(), r#" 
            stock_keeping_unit:{
                value:Stock Keeping Unit | string,
                selector:selector
            }"#.to_string()));

            fields.push(("low_stock_threshold".to_string(), r#" 
            low_stock_threshold:{
                value:product Low stock alert threshold | number,
                selector:selector
            }"#.to_string()));

            fields.push(("unit".to_string(), r#" 
            unit:{
                value:product Selling unit | string,
                selector:selector
            }"#.to_string()));

            fields.push(("tax_included".to_string(), r#" 
            tax_included:{
                value:product Whether tax | number,
                selector:selector
            }"#.to_string()));

            fields.push(("tax_code".to_string(), r#" 
            tax_code:{
                value:product Tax code for region-specific rules | string,
                selector:selector
            }"#.to_string()));

            fields.push(("main_image_url".to_string(), r#" 
            main_image_url:{
                value:Main product image URL | string,
                selector:selector
            }"#.to_string()));

            fields.push(("additional_image_url".to_string(), r#" 
            additional_image_url:{
                value:additional product image URL | string,
                selector:selector
            }"#.to_string()));

            fields.push(("video_url".to_string(), r#" 
            video_url:{
                value:product Promotional video URL | string,
                selector:selector
            }"#.to_string()));

            fields.push(("carrier".to_string(), r#" 
            carrier:{
                value:product carrier name translated into English | string,
                selector:selector
            }"#.to_string()));

            fields.push(("shipping_fee".to_string(), r#" 
            shipping_fee:{
                value:product Shipping cost | number,
                selector:selector
            }"#.to_string()));

            fields.push(("shipping_method".to_string(), r#" 
            shipping_method:{
                value:'standard' or 'express' or 'same_day' or 'pick_up' or 'freight' or 'prepaid',
                selector:selector
            }"#.to_string()));

            fields.push(("shipping_duration".to_string(), r#" 
            shipping_duration:{
                value:product Estimated delivery days | number,
                selector:selector
            }"#.to_string()));

            fields.push(("bundle_shipping".to_string(), r#" 
            bundle_shipping:{
                value:product Allow combined shipping | string,
                selector:selector
            }"#.to_string()));

            fields.push(("product_width".to_string(), r#" 
            product_width:{
                value:Package width(cm) | number,
                selector:selector
            }"#.to_string()));

            fields.push(("product_height".to_string(), r#" 
            product_height:{
                value:Package height(cm) | number,
                selector:selector
            }"#.to_string()));

            fields.push(("product_length".to_string(), r#" 
            product_length:{
                value : Package length(cm) | number,
                selector:selector
            }"#.to_string()));

            fields.push(("product_weight".to_string(), r#" 
            product_weight:{
                value : Package weight(kg) | number,
                selector:selector
            }"#.to_string()));

            fields.push(("options".to_string(), r#" 
            options:[
                {
                    value:option name | string,
                    selector:selector,
                    inputs:[{
                        value:option input value | string,
                        selector:selector
                    }]
                }
            ]"#.to_string()));

            fields.push(("additional_goods".to_string(), r#" 
            additional_goods:[
                {
                    value:URL includes a manage path, an administrative or edit route product Link | string,
                    selector:selector
                }
            ]"#.to_string()));

            fields.push(("title".to_string(), r#" 
            title:{
                value:product based title | string,
                selector:selector
            }"#.to_string()));

            fields.push(("registration_date".to_string(), r#" 
            registration_date:{
                value:yyyy-MM-ddThh:mm:ss | string,
                selector:selector
            }"#.to_string()));
        },
        "order" => {
            fields.push(("node".to_string(), "node:order form container CSS1 selector".to_string()));
            fields.push(("link".to_string(), format!("link : '{{}}'", href)));

            fields.push(("id".to_string(), r#" 
            id:{
                value:Refer to the ID value from the link or an attribute or input value | string,
                selector:selector
            }"#.to_string()));

            fields.push(("tracking_number".to_string(), r#" 
            tracking_number:{
                value:tracking number | string,
                selector:selector
            }"#.to_string()));

            fields.push(("status".to_string(), r#" 
            status:{
                value:'draft' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
                selector:selector
            }"#.to_string()));

            fields.push(("goods".to_string(), r#" 
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
            }]"#.to_string()));

            fields.push(("sender_name".to_string(), r#" 
            sender_name:{
                value:sender_name | string,
                selector:selector
            }"#.to_string()));

            fields.push(("sender_address".to_string(), r#" 
            sender_address:{
                value:sender_address, Filter the addresses to District-level and up | string,
                selector:selector
            }"#.to_string()));

            fields.push(("sender_phone".to_string(), r#" 
            sender_phone:{
                value:sender_phone | string,
                selector:selector
            }"#.to_string()));

            fields.push(("recipient_name".to_string(), r#" 
            recipient_name:{
                value:recipient_name | string,
                selector:selector
            }"#.to_string()));

            fields.push(("recipient_address".to_string(), r#" 
            recipient_address:{
                value:recipient_address, Filter the addresses to District-level and up | string,
                selector:selector
            }"#.to_string()));

            fields.push(("recipient_phone".to_string(), r#" 
            recipient_phone:{
                value:recipient_phone | string,
                selector:selector
            }"#.to_string()));

            fields.push(("bank".to_string(), r#" 
            bank:{
                value:bank company name | string,
                selector:selector
            }"#.to_string()));

            fields.push(("card".to_string(), r#" 
            card:{
                value:card company name | string,
                selector:selector
            }"#.to_string()));

            fields.push(("order_date".to_string(), r#" 
            order_date:{
                value:order date | string,
                selector:selector
            }"#.to_string()));

            fields.push(("payment_date".to_string(), r#" 
            payment_date:{
                value:payment date or '' | string,
                selector:selector
            }"#.to_string()));

            fields.push(("payment_method".to_string(), r#" 
            payment_method:{
                value:'C.O.D.' or 'CARD' or 'BANK' or '' | string,
                selector:selector
            }"#.to_string()));

            fields.push(("payment_origin".to_string(), r#" 
            payment_origin:{
                value:Payment Gateway Service Name or '' | string,
                selector:selector
            }"#.to_string()));

            fields.push(("registration_date".to_string(), r#" 
            registration_date:{
                value:yyyy-MM-ddThh:mm:ss | string,
                selector:selector
            }"#.to_string()));
        },
        "coupon" | "event" => {
            fields.push(("node".to_string(), format!("node:{{}} container CSS1 selector", page_type)));
            fields.push(("link".to_string(), format!("link : '{{}}'", href)));

            fields.push(("id".to_string(), r#" 
            id:{
                value:Refer to the ID value from the link or an attribute or input value | string,
                selector:selector
            }"#.to_string()));

            fields.push(("type".to_string(), r#" 
            type:{
                value:'percentage' or 'fixed_amount' or 'free_shipping' or '',
                selector:selector
            }"#.to_string()));

            fields.push(("status".to_string(), r#" 
            status:{
                value:'draft' or 'progress' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error',
                selector:selector
            }"#.to_string()));

            fields.push(("title".to_string(), format!(r#" 
            title:{{ 
                value:{{}} title | string, 
                selector:selector
            }}"#, page_type)));

            fields.push(("started_at".to_string(), r#" 
            started_at:{
                value:yyyy-MM-ddThh:mm:ss | string,
                selector:selector
            }"#.to_string()));

            fields.push(("expired_at".to_string(), r#" 
            expired_at:{
                value:yyyy-MM-ddThh:mm:ss | string,
                selector:selector
            }"#.to_string()));

            fields.push(("code".to_string(), format!(r#" 
            code:{{ 
                value:{{}} code used at checkout | string,
                selector:selector
            }}"#, page_type)));

            fields.push(("discount".to_string(), r#" 
            discount:{
                value:Discount value | number,
                selector:selector
            }"#.to_string()));

            fields.push(("quantity".to_string(), format!(r#" 
            quantity:{{ 
                value:{{}} quantity | number
                selector:selector
            }}"#, page_type)));

            fields.push(("usage_limit".to_string(), r#" 
            usage_limit:{
                value:Total usage limit for the coupon | number,
                selector:selector
            }"#.to_string()));

            fields.push(("usage_per".to_string(), r#" 
            usage_per:{
                value:Usage limit per customer | number,
                selector:selector
            }"#.to_string()));

            fields.push(("new_customer_only".to_string(), r#" 
            new_customer_only:{
                value:new customer only | boolean
                selector:selector
            }"#.to_string()));

            fields.push(("min_order_amount".to_string(), r#" 
            min_order_amount:{
                value:Minimum order amount required to apply coupon | number,
                selector:selector
            }"#.to_string()));

            fields.push(("max_discount_amount".to_string(), r#" 
            max_discount_amount:{
                value:Maximum discount limit allowed for the coupon | number,
                selector:selector
            }"#.to_string()));

            fields.push(("region_restrictions".to_string(), r#" 
            region_restrictions:{
                value:region restrictions | boolean,
                selector:selector
            }"#.to_string()));

            fields.push(("registration_date".to_string(), r#" 
            registration_date:{
                value:yyyy-MM-ddThh:mm:ss | string,
                selector:selector
            }"#.to_string()));
        },
        "review" => {
            fields.push(("node".to_string(), format!("node:{{}} container CSS1 selector", page_type)));
            fields.push(("link".to_string(), format!("link : '{{}}'", href)));
            fields.push(("id".to_string(), "id:Refer to the ID value from the link or an attribute or input value | string,".to_string()));

            fields.push(("status".to_string(), r#" 
            status:{
                value:'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
                selector:selector
            }"#.to_string()));

            fields.push(("name".to_string(), format!(r#" 
            name:{{ 
                value:{{}} name | string,
                selector:selector
            }}"#, page_type)));

            fields.push(("title".to_string(), format!(r#" 
            title:{{ 
                value:{{}} item title | string, 
                selector:selector
            }}"#, page_type)));

            fields.push(("completed".to_string(), r#" 
            completed:{
                value:order complete | boolean,
                selector:selector
            }"#.to_string()));

            fields.push(("registration_date".to_string(), r#" 
            registration_date:{
                value:yyyy-MM-ddThh:mm:ss | string,
                selector:selector
            }"#.to_string()));
        },
        _ => {} // Do nothing for other page types
    }

    // Wrap each field schema with the base instruction
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
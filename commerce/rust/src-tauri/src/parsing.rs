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
                line.push_str(&format!("#{}", id));
            }

            for class in element.classes() {
                line.push_str(&format!(".{}", class));
            }

            match mode {
                PugMode::StructureOnly => {
                    if let Some(val) = element.attr("href") {
                        attrs.push(format!("href='{}'", val));
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
                                 attrs.push(format!(r#"{}='{}'"#, key, safe_val));
                             }
                        }
                    }
                }
            }

            if !attrs.is_empty() {
                line.push_str(&format!("({})", attrs.join(" ")));
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
                            output.push_str(&format!("{}| {}
", indent, trimmed_line.replace("\"", "'" )));
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
			"tracking_number": "tracking number | string",
			"recipient_match": "shipping label #recipient_address match | boolean",
			"barcodes": ["barcode number | string"],
			"text": "summarize the shipping label contents in {LANGUAGE}. Masking the address in the summary to District-level and up."
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

pub fn item2json(page_type: &str, href: &str, language: &str) -> String {
    let schema = match page_type {
        "tracking" => r###" 
            node:tracking form container CSS1 selector,
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
            },
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
            },
        "###.to_string(),
        
        "goods" => {
            let t = r###" 
            node:goods form container CSS1 selector,
            code:{
                value:product constant code | string,
                selector:selector
            },
            link:'{LINK}',
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
            }
            "###;
            t.replace("{LINK}", href)
        },
        
        "order" => {
            let t = r###" 
            node:order form container CSS1 selector,
            link : '{LINK}',
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
            }
            "###;
            t.replace("{LINK}", href)
        },

        "coupon" | "event" => {
            let t = r###" 
            node:{TYPE} container CSS1 selector,
            link : '{LINK}',
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
            }
            "###;
            t.replace("{TYPE}", page_type).replace("{LINK}", href)
        },

        "review" => {
            let t = r###" 
            node:{TYPE} container CSS1 selector,
            link : '{LINK}',
            id:Refer to the ID value from the link or an attribute or input value | string,,
            status:{
                value:'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
                selector:selector
            },
            name:{
                value:{TYPE} name | string,
                selector:selector
            },
            title:{
                value:{TYPE} item title | string, 
                selector:selector
            },
            completed:{
                value:order complete | boolean,
                selector:selector
            },
            registration_date:{
                value:yyyy-MM-ddThh:mm:ss | string,
                selector:selector
            }
            "###;
            t.replace("{TYPE}", page_type).replace("{LINK}", href)
        },

        _ => "{}".to_string()
    };

    let template = r###" 
    Analyze the provided Pug template and return it in the following JSON format, no explanation.
    # selector : sibling value based CSS1 selector
    {language:'{LANGUAGE}', {SCHEMA}}
    "###;
    
    template.replace("{LANGUAGE}", language).replace("{SCHEMA}", &schema)
}

pub fn list2json(language: &str) -> String {
    let template = r###" 
    Analyze the provided Pug template and return it in the following JSON format, no explanation.
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

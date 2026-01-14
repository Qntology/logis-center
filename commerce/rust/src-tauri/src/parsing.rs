use scraper::{Html, Node};
use ego_tree::NodeRef;

pub enum PugMode {
    StructureOnly, // 1단계: 타입 식별용 (가볍게, 텍스트/링크 제거)
    FullContent,   // 2단계: 데이터 추출용 (텍스트/링크 포함)
}

/// HTML을 설정된 모드에 따라 Pug 포맷으로 변환합니다.
pub fn convert_to_clean_pug(html: &str, mode: PugMode) -> String {
    let document = Html::parse_document(html);
    let mut pug_output = String::new();

    for child in document.tree.root().children() {
        if let Some(element) = child.value().as_element() {
            if element.name() == "html" {
                 for html_child in child.children() {
                     if let Some(body_el) = html_child.value().as_element() {
                         if body_el.name() == "body" {
                             generate_pug_lines(html_child, 0, &mut pug_output, &mode);
                         }
                     }
                 }
            }
        }
    }
    
    if pug_output.is_empty() {
         generate_pug_lines(document.tree.root(), 0, &mut pug_output, &mode);
    }

    pug_output
}

fn generate_pug_lines(node: NodeRef<scraper::Node>, indent_level: usize, output: &mut String, mode: &PugMode) {
    let indent = "    ".repeat(indent_level);

    match node.value() {
        Node::Element(element) => {
            let tag_name = element.name();

            if ["script", "style", "noscript", "iframe", "svg", "path", "link", "meta"].contains(&tag_name) {
                return;
            }
            
            // 1단계(구조 파악)에서는 img 태그도 구조만 중요하므로 src는 굳이 필요 없을 수 있으나, 
            // 아이콘/로고 구분을 위해 남겨둘 수도 있음. 하지만 요청대로 1단계에선 최대한 가볍게 갑니다.
            if let PugMode::StructureOnly = mode {
                if tag_name == "img" {
                    // 1단계에선 이미지 태그 자체는 남기되 속성은 거의 날립니다.
                }
            } else {
                if tag_name == "img" {
                    if let Some(src) = element.attr("src") {
                        if src.starts_with("data:image") { return; }
                    }
                }
            }

            let mut line = format!("{}{}", indent, tag_name);
            let mut attrs = Vec::new();

            // ID
            if let Some(id) = element.id() {
                line.push_str(&format!("#{}", id));
            }

            // Class
            for class in element.classes() {
                line.push_str(&format!(".{}", class));
            }

            // Attributes
            match mode {
                PugMode::StructureOnly => {
                    // 1단계: type 식별에 도움되는 최소한의 속성만 허용
                    // input type 등은 폼 식별에 중요함
                    if let Some(val) = element.attr("type") {
                        attrs.push(format!("type='{}'", val));
                    }
                    // 그 외 href, src, data-*, value 등은 모두 제거하여 토큰 절약
                },
                PugMode::FullContent => {
                    // 2단계: 데이터 추출에 필요한 속성 포함
                    let allowed_attrs = ["href", "src", "type", "name", "value", "placeholder", "checked", "selected", "disabled", "readonly", "rows", "cols"];
                    
                    for (key, value) in element.attrs() {
                        if key == "id" || key == "class" { continue; }
                        
                        if allowed_attrs.contains(&key) {
                             if ["checked", "selected", "disabled", "readonly"].contains(&key) {
                                 attrs.push(key.to_string());
                             } else {
                                 attrs.push(format!("{}='{}'", key, value.replace("'", "\"").replace("\n", "")));
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
            match mode {
                PugMode::StructureOnly => {
                    // 1단계: 텍스트 노드 완전 제거 (구조만 보고 타입 파악)
                    // 단, 짧은 버튼 이름 등은 힌트가 될 수 있으나, 일단 요청대로 제거하여 "지도 맵"처럼 만듭니다.
                },
                PugMode::FullContent => {
                    let content = text.trim();
                    if !content.is_empty() {
                        // 여러 줄 텍스트 처리
                        for line in content.lines() {
                            let trimmed_line = line.trim();
                            if !trimmed_line.is_empty() {
                                output.push_str(&format!("{}| \{} 
", indent, trimmed_line.replace("'", "\"")));
                            }
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

// --- Schema Prompts (Ported from proxy/src/index.ts) ---

pub fn item2json(page_type: &str, href: &str, language: &str) -> String {
    let schema = match page_type {
        "tracking" => format!(r#" 
            node: tracking form container CSS1 selector,
            status:{{ value:'draft' or 'progress' or 'return' or 'complete' or 'error', selector:selector }},
            id:{{ value:tracking number | string, selector:selector }},
            title:{{ value:tracking goods title | string, selector:selector }},
            sender_name:{{ value:sender_name | string, selector:selector }},
            sender_address:{{ value:sender_address | string, selector:selector }},
            recipient_name:{{ value:recipient_name | string, selector:selector }},
            recipient_address:{{ value:recipient_address | string, selector:selector }},
            carrier:{{ value:carrier name translated into English | string, selector:selector }},
            shipping_fee:{{ value:Shipping cost | number, selector:selector }},
            shipping_date:{{ value:yyyy-MM-ddThh:mm:ss | string, selector:selector }},
        "#),
        "goods" => format!(r#" 
            node: goods form container CSS1 selector,
            link: '{}',
            id:{{ value:Product ID from link/attr/input | string, selector:selector }},
            status:{{ value:'draft' or 'show' or 'hide' or 'soldout', selector:selector }},
            title:{{ value:product title | string, selector:selector }},
            sale_price:{{ value:sale price | number, selector:selector }},
            supply_price:{{ value:supply price | number, selector:selector }},
            currency:{{ value:ISO 4217 Currency Code | string, selector:selector }},
            quantity:{{ value:Inventory quantity | number, selector:selector }},
            main_image_url:{{ value:Main product image URL | string, selector:selector }},
            description:{{ value:product description text | string, selector:selector }},
            options:[{{ value:option name | string, selector:selector, inputs:[{{ value:option input value | string, selector:selector }}] }}],
        "#, href),
        "order" => format!(r#" 
            node: order form container CSS1 selector,
            link: '{}',
            id:{{ value:Order ID | string, selector:selector }},
            tracking_number:{{ value:tracking number | string, selector:selector }},
            status:{{ value:'paid' or 'shipping' or 'complete' or 'cancel', selector:selector }},
            goods:[{{ title:{{ value:goods title | string, selector:selector }}, id:{{ value:product id | string, selector:selector }}, quantity:{{ value:count | number, selector:selector }} }}],
            recipient_name:{{ value:recipient name | string, selector:selector }},
            recipient_address:{{ value:recipient address | string, selector:selector }},
            recipient_phone:{{ value:recipient phone | string, selector:selector }},
            order_date:{{ value:order date | string, selector:selector }},
        "#, href),
        "event" | "coupon" => format!(r#" 
            node: event/coupon container CSS1 selector,
            link: '{}',
            title:{{ value:title | string, selector:selector }},
            started_at:{{ value:yyyy-MM-ddThh:mm:ss | string, selector:selector }},
            expired_at:{{ value:yyyy-MM-ddThh:mm:ss | string, selector:selector }},
            code:{{ value:coupon code | string, selector:selector }},
            discount:{{ value:discount amount/percent | number, selector:selector }},
        "#, href),
        _ => "{{}}".to_string()
    };

    format!("\n    Analyze the provided Pug template and return it in the following JSON format, no explanation.
    # selector : sibling value based CSS1 selector
    {{language:'{}', {}}}
    ", language, schema)
}

pub fn list2json(language: &str) -> String {
    format!("\n    Analyze the provided Pug template and return it in the following JSON format, no explanation.
    {{ 
        language: '{}',
        type: 'order' or 'goods' or 'tracking' or 'review' or 'coupon' or 'event' or '',
        item: type based item CSS1 selector excluding ads,
        node: item parent list CSS1 selector excluding ads,
        next: list next button CSS1 selector,
        detail: is a detail page or a detail form | boolean,
        items: [
            if (type is 'tracking') {{ status, id, title, link }}
            if (type is 'order' or 'goods') {{ status, link, id, title, sale_price, quantity, tracking_number }}
            if (type is 'coupon' or 'event') {{ status, id, title, started_at, expired_at }}
        ]
    }}
    ", language, language)
}
use once_cell::sync::Lazy;
use serde_json::Value;

pub static BIAS_DICT: Lazy<Value> = Lazy::new(|| {
    let json_str = include_str!("bias.json");
    serde_json::from_str(json_str).unwrap_or(serde_json::json!({}))
});

pub fn lang_code_of(lang: &str) -> String {
    let l = lang.trim().to_lowercase();
    if l.starts_with("zh-tw") || l.starts_with("zh-hk") || l.starts_with("zh-hant") {
        return "zh-tw".to_string();
    }
    let code: String = l.chars().take_while(|c| c.is_ascii_alphabetic()).take(2).collect();
    if code.chars().count() >= 2 { code } else { "en".to_string() }
}

pub fn get_localized_page_type(page_type: &str, lang: &str) -> String {
    // 🌟 zh-tw / zh-hk 번체 분기 포함, 바이트 슬라이싱 없이 안전하게 코드를 뽑습니다.
    let lang_code_owned = lang_code_of(lang);
    let lang_code = lang_code_owned.as_str();
    let matched_str = match lang_code {
        "ko" => match page_type { "order" => "주문", "goods" => "상품", "tracking" => "배송", "review" => "리뷰", "coupon" | "event" => "이벤트", _ => "문서" },
        "zh-tw" => match page_type { "order" => "訂單", "goods" => "商品", "tracking" => "物流", "review" => "評價", "coupon" | "event" => "活動", _ => "文件" },
        "zh" => match page_type { "order" => "订单", "goods" => "商品", "tracking" => "物流", "review" => "评价", "coupon" | "event" => "活动", _ => "文档" },
        "ja" => match page_type { "order" => "注文", "goods" => "商品", "tracking" => "配送", "review" => "レビュー", "coupon" | "event" => "イベント", _ => "ドキュメント" },
        "es" => match page_type { "order" => "pedido", "goods" => "producto", "tracking" => "seguimiento", "review" => "reseña", "coupon" | "event" => "evento", _ => "documento" },
        "pt" => match page_type { "order" => "pedido", "goods" => "produto", "tracking" => "rastreamento", "review" => "avaliação", "coupon" | "event" => "evento", _ => "documento" },
        "de" => match page_type { "order" => "bestellung", "goods" => "produkt", "tracking" => "sendungsverfolgung", "review" => "bewertung", "coupon" | "event" => "event", _ => "dokument" },
        "nl" => match page_type { "order" => "bestelling", "goods" => "product", "tracking" => "tracking", "review" => "beoordeling", "coupon" | "event" => "evenement", _ => "document" },
        "it" => match page_type { "order" => "ordine", "goods" => "prodotto", "tracking" => "tracciamento", "review" => "recensione", "coupon" | "event" => "evento", _ => "documento" },
        "id" | "ms" => match page_type { "order" => "pesanan", "goods" => "produk", "tracking" => "pelacakan", "review" => "ulasan", "coupon" | "event" => "acara", _ => "dokumen" },
        "vi" => match page_type { "order" => "đơn hàng", "goods" => "sản phẩm", "tracking" => "theo dõi", "review" => "đánh giá", "coupon" | "event" => "sự kiện", _ => "tài liệu" },
        "th" => match page_type { "order" => "คำสั่งซื้อ", "goods" => "สินค้า", "tracking" => "การติดตาม", "review" => "รีวิว", "coupon" | "event" => "กิจกรรม", _ => "เอกสาร" },
        "ar" => match page_type { "order" => "طلب", "goods" => "منتج", "tracking" => "تتبع", "review" => "مراجعة", "coupon" | "event" => "حدث", _ => "مستند" },
        "ta" => match page_type { "order" => "ஆர்டர்", "goods" => "தயாரிப்பு", "tracking" => "கண்காணிப்பு", "review" => "விமர்சனம்", "coupon" | "event" => "நிகழ்வு", _ => "ஆவணம்" },
        "te" => match page_type { "order" => "ఆర్డర్", "goods" => "ఉత్పత్తి", "tracking" => "ట్రాకింగ్", "review" => "సమీక్ష", "coupon" | "event" => "ఈవెంట్", _ => "పత్రం" },
        "kn" => match page_type { "order" => "ಆರడರ್", "goods" => "ಉತ್ಪನ್ನ", "tracking" => "ಟ್ರ್ಯಾಕಿಂಗ್", "review" => "ವಿಮರ್ಶೆ", "coupon" | "event" => "ಈವೆಂಟ್", _ => "ಡಾಕ್ಯುಮೆಂಟ್" },
        "gu" => match page_type { "order" => "ઓરડર", "goods" => "ઉત્પાદન", "tracking" => "ટ્રેકિંગ", "review" => "સમીક્ષા", "coupon" | "event" => "ઇવેન્ટ", _ => "દસ્તાવેજ" },
        _ => match page_type { "order" => "order", "goods" => "product", "tracking" => "tracking", "review" => "review", "coupon" | "event" => "event", _ => "document" },
    };
    // 최종 결과물에 단 한번만 .to_string()을 호출하여 타입 불일치 에러 완벽 해결
    matched_str.to_string()
}

pub fn get_layout_bias(page_type: &str, lang: &str) -> (String, String) {
    let mut bias = String::from("detail has_list has_form true false");
    let mut prejudice = String::new(); // 🌟 초기화
    let lang_code_owned = lang_code_of(lang);
    let lang_code = lang_code_owned.as_str();
    let localized_type = get_localized_page_type(page_type, lang);
    if let Some(localized_obj) = BIAS_DICT.get(lang_code).or_else(|| BIAS_DICT.get("en")).and_then(|l| l.get(page_type).or_else(|| l.get("default"))) {
        if let Some(l_list) = localized_obj.get("layout_list") {
            if let Some(b) = l_list.get("bias").and_then(|v| v.as_str()) {
                bias.push_str(" ");
                bias.push_str(&b.replace("{TYPE}", &localized_type));
            }
            if let Some(p) = l_list.get("prejudice").and_then(|v| v.as_str()) {
                prejudice.push_str(" ");
                prejudice.push_str(&p.replace("{TYPE}", &localized_type));
            }
        }
        if let Some(l_form) = localized_obj.get("layout_form") {
            if let Some(b) = l_form.get("bias").and_then(|v| v.as_str()) {
                bias.push_str(" ");
                bias.push_str(&b.replace("{TYPE}", &localized_type));
                bias.push_str(&format!(" {} input {} select {} textarea", localized_type, localized_type, localized_type));
            }
            if let Some(p) = l_form.get("prejudice").and_then(|v| v.as_str()) {
                prejudice.push_str(" ");
                prejudice.push_str(&p.replace("{TYPE}", &localized_type));
            }
        }
    } else {
        bias.push_str(&format!(" {} input {} select {} textarea", localized_type, localized_type, localized_type));
    }
    if prejudice.trim().is_empty() {
        prejudice = String::from("global navigation, menus, footers, aside, search, filter.");
    }
    (bias.trim().to_string(), prejudice.trim().to_string())
}

pub fn get_combinatorial_layout_bias(active_types: &[&str], lang: &str) -> (String, String, String) {
    let mut combined_list_bias = String::new();
    let mut combined_form_bias = String::new();
    let mut combined_prejudice = std::collections::HashSet::new();
    let lang_code_owned = lang_code_of(lang);
    let lang_code = lang_code_owned.as_str();
    // 1. 인입된 모든 활성 도메인 트랙을 순회하며 바이어스 융합 매트릭스 빌드
    for page_type in active_types {
        let localized_type = get_localized_page_type(page_type, lang);
        if let Some(localized_obj) = BIAS_DICT.get(lang_code).or_else(|| BIAS_DICT.get("en")).and_then(|l| l.get(*page_type).or_else(|| l.get("default"))) {
            // List 성격의 교차 키워드 누적 합산
            if let Some(l_list) = localized_obj.get("layout_list") {
                if let Some(b) = l_list.get("bias").and_then(|v| v.as_str()) {
                    if !combined_list_bias.is_empty() { combined_list_bias.push_str(" "); }
                    combined_list_bias.push_str(&b.replace("{TYPE}", &localized_type));
                }
                if let Some(p) = l_list.get("prejudice").and_then(|v| v.as_str()) {
                    combined_prejudice.insert(p.replace("{TYPE}", &localized_type));
                }
            }
            // Form 성격의 교차 키워드 누적 합산
            if let Some(l_form) = localized_obj.get("layout_form") {
                if let Some(b) = l_form.get("bias").and_then(|v| v.as_str()) {
                    if !combined_form_bias.is_empty() { combined_form_bias.push_str(" "); }
                    combined_form_bias.push_str(&b.replace("{TYPE}", &localized_type));
                }
                if let Some(p) = l_form.get("prejudice").and_then(|v| v.as_str()) {
                    combined_prejudice.insert(p.replace("{TYPE}", &localized_type));
                }
            }
        } else {
            // 폴백 키워드도 누적합 행렬에 부드럽게 통합 유도
            combined_list_bias.push_str(&format!(" {} list catalog grid repeating table rows items", localized_type));
            combined_form_bias.push_str(&format!(" {} detail form single input fields properties configuration", localized_type));
        }
    }
    // 2. 만약 활성 레이아웃이 비어있을 때를 대비한 기본 방어선 구축
    if combined_list_bias.trim().is_empty() {
        combined_list_bias = String::from("list catalog grid repeating multiple table rows items");
    }
    if combined_form_bias.trim().is_empty() {
        combined_form_bias = String::from("detail form single input fields properties configuration input select textarea");
    }
    // 3. 중복 제거된 배제(Prejudice) 단어 배열을 단일 텍스트 구문으로 압축 조립
    let mut prej_vec: Vec<String> = combined_prejudice.into_iter().collect();
    if prej_vec.is_empty() {
        prej_vec.push(String::from("global navigation, menus, footers, aside, search, guide, tip, filter."));
    }
    let final_prejudice = prej_vec.join(" ");
    (combined_list_bias.trim().to_string(), combined_form_bias.trim().to_string(), final_prejudice.trim().to_string())
}

pub fn get_page_type_full_bias(page_type: &str, lang: &str) -> String {
    let mut full_bias = String::from(page_type);
    let lang_code_owned = lang_code_of(lang);
    let lang_code = lang_code_owned.as_str();
    let localized_type = get_localized_page_type(page_type, lang);
    if let Some(localized_obj) = BIAS_DICT.get(lang_code).or_else(|| BIAS_DICT.get("en")).and_then(|l| l.get(page_type).or_else(|| l.get("default"))) {
        if let Some(obj) = localized_obj.as_object() {
            for (_, v) in obj {
                if let Some(b) = v.get("bias").and_then(|bv| bv.as_str()) {
                    full_bias.push_str(" ");
                    full_bias.push_str(&b.replace("{TYPE}", &localized_type));
                }
            }
        }
    }
    full_bias.trim().to_string()
}

/// 🌟 [PAGE TYPE CLASSIFICATION 전용] layout_list + layout_form + 로컬라이즈된 타입 이름만 사용하여
/// 필드 레벨 bias 노이즈(예: sender_name의 "테스트", goods 필드의 상품 예시값 등)를 원천 차단합니다.
pub fn get_page_type_classification_bias(page_type: &str, lang: &str) -> String {
    let lang_code_owned = lang_code_of(lang);
    let lang_code = lang_code_owned.as_str();
    let localized_type = get_localized_page_type(page_type, lang);
    let mut bias = String::from(page_type);
    bias.push_str(" ");
    bias.push_str(&localized_type);
    if let Some(localized_obj) = BIAS_DICT.get(lang_code).or_else(|| BIAS_DICT.get("en")).and_then(|l| l.get(page_type).or_else(|| l.get("default"))) {
        // 오직 layout_list, layout_form만 사용 (필드 레벨 bias는 분류 노이즈의 주범)
        for key in ["layout_list", "layout_form"] {
            if let Some(layout_obj) = localized_obj.get(key) {
                if let Some(b) = layout_obj.get("bias").and_then(|v| v.as_str()) {
                    bias.push_str(" ");
                    bias.push_str(&b.replace("{TYPE}", &localized_type));
                }
            }
        }
        // title bias의 시맨틱 키워드만 포함 (예시값 제외)
        if let Some(title_obj) = localized_obj.get("title") {
            if let Some(b) = title_obj.get("bias").and_then(|v| v.as_str()) {
                // 예시값(긴 문장)은 제거하고 쉼표 앞의 핵심 키워드만 추출
                let keywords: Vec<&str> = b.split(',').take(3).collect();
                for kw in keywords {
                    let kw_trimmed = kw.trim();
                    if kw_trimmed.len() < 20 {
                        bias.push_str(" ");
                        bias.push_str(kw_trimmed);
                    }
                }
            }
        }
        // status bias도 포함 (페이지의 상태 필터 텍스트가 타입 판별에 중요)
        if let Some(status_obj) = localized_obj.get("status") {
            if let Some(b) = status_obj.get("bias").and_then(|v| v.as_str()) {
                bias.push_str(" ");
                bias.push_str(&b.replace("{TYPE}", &localized_type));
            }
        }
    }
    bias.trim().to_string()
}

pub fn get_title_bias(page_type: &str, lang: &str) -> (String, String) {
    let lang_code_owned = lang_code_of(lang);
    let lang_code = lang_code_owned.as_str();
    let localized_type = get_localized_page_type(page_type, lang);
    let mut bias = String::from("title name product ");
    let mut prejudice = String::from("address location ");
    if let Some(localized_obj) = BIAS_DICT.get(lang_code).or_else(|| BIAS_DICT.get("en")).and_then(|l| l.get(page_type).or_else(|| l.get("default"))) {
        if let Some(t_obj) = localized_obj.get("title") {
            if let Some(b) = t_obj.get("bias").and_then(|v| v.as_str()) { bias = format!("{} {} ", bias, b.replace("{TYPE}", &localized_type)); }
            if let Some(p) = t_obj.get("prejudice").and_then(|v| v.as_str()) { prejudice = format!("{} {} ", prejudice, p.replace("{TYPE}", &localized_type)); }
        }
    }
    (bias, prejudice)
}

pub fn get_list_schema_fields(page_type: &str, _href: &str, lang: &str) -> Vec<(String, String, String, String)> {
    let mut fields = Vec::new();
    let lang_code_owned = lang_code_of(lang);
    let lang_code = lang_code_owned.as_str();
    let localized_type = get_localized_page_type(page_type, lang);
    let mut add = |key: &str, field_type: &str, en_bias: &str, en_prejudice: &str| {
        // 🌟 [핵심 변경] 콤마(,)를 기준으로 텍스트를 분리하여 모든 의미 단위(동의어)마다 독립적으로 영어 도메인(page_type)을 부착합니다.
        let inject_domain = |text: &str, domain: &str| -> String {
            if text.trim().is_empty() { return String::new(); }
            text.split(',')
                .map(|s| format!("{} {}", domain, s.trim()))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut final_bias = inject_domain(en_bias, page_type);
        let mut final_prejudice = inject_domain(en_prejudice, page_type);
        let mut semantic_desc = String::new();
        if let Some(localized_obj) = BIAS_DICT
            .get(lang_code)
            .and_then(|l| l.get(page_type).or_else(|| l.get("default")))
            .and_then(|p| p.get(key))
        {
            if let Some(semantic) = localized_obj.get("semantic").and_then(|v| v.as_str()) {
                semantic_desc = semantic.to_string();
            }
            if let Some(bias_str) = localized_obj.get("bias").and_then(|v| v.as_str()) {
                let localized_b = inject_domain(&bias_str.replace("{TYPE}", &localized_type), page_type);
                final_bias = if final_bias.is_empty() { localized_b } else { format!("{}, {}", final_bias, localized_b) };
            }
            if let Some(prejudice_str) = localized_obj.get("prejudice").and_then(|v| v.as_str()) {
                let localized_p = inject_domain(&prejudice_str.replace("{TYPE}", &localized_type), page_type);
                final_prejudice = if final_prejudice.is_empty() { localized_p } else { format!("{}, {}", final_prejudice, localized_p) };
            }
        }
        let final_desc = if key == "id,link" {
            "- \"link\": String. Detailed page URL.\n- \"id\": String. Refer to the ID value from the link.".to_string()
        } else if semantic_desc.is_empty() {
            format!("- \"{}\": {}.", key, field_type)
        } else {
            format!("- \"{}\": {}. {}", key, field_type, semantic_desc)
        };
        fields.push((key.to_string(), final_desc, final_bias.trim().to_string(), final_prejudice.trim().to_string()));
    };
    // 🌟 [비대칭 가중치 반영] bias.json의 insight 블록을 읽어 현재 page_type이 target_domain에 포함된 경우에만 스키마에 추가합니다.
    if let Some(insight_obj) = BIAS_DICT.get("insight").and_then(|v| v.as_object()) {
        for (insight_key, insight_val) in insight_obj {
            let target_domains = insight_val.get("target_domain").and_then(|v| v.as_array()).map(|arr| {
                arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>()
            }).unwrap_or_default();
            // target_domain이 비어있거나(전역), 현재 page_type이 포함된 경우에만 add
            if target_domains.is_empty() || target_domains.contains(&page_type) {
                let s_bias = insight_val.get("bias").and_then(|v| v.as_str()).unwrap_or("");
                let s_prej = insight_val.get("prejudice").and_then(|v| v.as_str()).unwrap_or("");
                add(insight_key, "String", s_bias, s_prej);
            }
        }
    }
    match page_type {
        "tracking" => {
            add("id,link", "", "id link tracking", "");
            add("status", "String", "status delivery", "");
            add("title", "String", "title product name", "");
            add("registration_date", "String", "date registration", "");
            add("sender_name", "String", "sender seller name", "");
            add("recipient_name", "String", "recipient buyer name", "");
            add("carrier", "String", "carrier courier", "");
        },
        "goods" => {
            add("id,link", "", "id link", "");
            add("code", "String", "code sku item", "");
            add("status", "String", "status condition", "");
            add("title", "String", "title name product", "");
            add("color", "String", "color hue shade tint", "");
            add("registration_date", "String", "date registration", "");
            add("sale_price", "Number", "sale price discount", "");
            add("supply_price", "Number", "supply price cost", "");
            add("currency", "String", "currency", "");
            add("quantity", "Number", "quantity inventory stock", "");
            add("stock_keeping_unit", "String", "sku code", "");
            add("main_image_url", "String", "main image thumbnail", "");
        },
        "order" => {
            add("id,link", "", "id link order number", "");
            add("tracking_number", "String", "tracking number", "");
            add("status", "String", "status state", "");
            add("registration_date", "String", "date registration", "");
            add("goods", "Array of Objects [ { title:String, link:String, id:String } ]", "goods product items", "");
            add("sender_name", "String", "sender buyer orderer name", "");
            add("recipient_name", "String", "recipient receiver name", "");
            add("payment_method", "String", "payment method type", "");
            add("payment_date", "String", "payment date", "");
        },
        "coupon" | "event" => {
            add("id,link", "", "id link", "");
            add("type", "String", "type", "");
            add("status", "String", "status state", "");
            add("title", "String", "title name", "");
            add("started_at", "String", "start date time", "");
            add("expired_at", "String", "expire end date time", "");
            add("code", "String", "code", "");
            add("discount", "Number", "discount", "");
            add("quantity", "Number", "quantity amount", "");
        },
        "review" => {
            add("id,link", "", "id link", "");
            add("status", "String", "status state", "");
            add("name", "String", "name reviewer author", "");
            add("title", "String", "title subject product", "");
            add("completed", "Boolean", "completed purchased", "");
            add("registration_date", "String", "date registration time", "");
        },
        _ => {
            add("id,link", "", "id link", "");
            add("title", "String", "title name", "");
            add("status", "String", "status state", "");
            add("registration_date", "String", "date registration", "");
        }
    }
    fields
}

pub fn get_vision_tracking_bias(lang: &str) -> (String, String) {
    let lang_code_owned = lang_code_of(lang);
    let lang_code = lang_code_owned.as_str();
    let mut bias = String::from("tracking number barcode awb ");
    // 🌟 [PREJUDICE RESTORE] 기존에는 이 변수를 만들고 한 번도 채우지 않아
    //    항상 빈 문자열이 반환되었습니다(unused_mut 경고 동반).
    //    ko.tracking 의 id,link / carrier prejudice 에는
    //    '배송상태, 상품명, 등록일, 주소, 무게' 같은 배제 어휘가 이미 있는데
    //    비전 운송장 판정에서 전혀 쓰이지 않았습니다.
    let mut prejudice = String::new();
    if let Some(localized_obj) = BIAS_DICT.get(lang_code).or_else(|| BIAS_DICT.get("en")).and_then(|l| l.get("tracking")) {
        if let Some(id_obj) = localized_obj.get("id,link") {
            if let Some(b) = id_obj.get("bias").and_then(|v| v.as_str()) { bias = format!("{} {} ", bias, b); }
            if let Some(p) = id_obj.get("prejudice").and_then(|v| v.as_str()) { prejudice = format!("{} {} ", prejudice, p); }
        }
        if let Some(c_obj) = localized_obj.get("carrier") {
            if let Some(b) = c_obj.get("bias").and_then(|v| v.as_str()) { bias = format!("{} {} ", bias, b); }
            if let Some(p) = c_obj.get("prejudice").and_then(|v| v.as_str()) { prejudice = format!("{} {} ", prejudice, p); }
        }
    }
    (bias, prejudice)
}

// 🌟 글로벌 언어를 한 번에 섞어 넣지 않고, 전달받은 언어 코드의 힌트만 생성합니다.
pub fn get_layout_prompt_hints(page_type: &str, lang: &str) -> (String, String) {
    let lang_code_owned = lang_code_of(lang);
    let lang_code = lang_code_owned.as_str();
    let localized_type = get_localized_page_type(page_type, lang);
    let mut list_words = String::new();
    let mut form_words = String::new();
    if let Some(localized_obj) = BIAS_DICT.get(lang_code).or_else(|| BIAS_DICT.get("en")).and_then(|l| l.get(page_type).or_else(|| l.get("default"))) {
        if let Some(l_list) = localized_obj.get("layout_list").and_then(|v| v.get("bias")).and_then(|v| v.as_str()) {
            list_words.push_str(&l_list.replace("{TYPE}", &localized_type));
        }
        if let Some(l_form) = localized_obj.get("layout_form").and_then(|v| v.get("bias")).and_then(|v| v.as_str()) {
            form_words.push_str(&l_form.replace("{TYPE}", &localized_type));
        }
    }
    let list_hints = if list_words.is_empty() { String::new() } else { format!("\n  (Related keywords in document: {})", list_words.trim()) };
    let form_hints = if form_words.is_empty() { String::new() } else { format!("\n  (Related keywords in document: {})", form_words.trim()) };
    (list_hints, form_hints)
}

pub fn get_multi_pass_contexts(page_type: &str, lang: &str) -> Vec<(String, String, String)> {
    let mut contexts = Vec::new();
    let lang_code_owned = lang_code_of(lang);
    let lang_code = lang_code_owned.as_str();
    let localized_type = get_localized_page_type(page_type, lang);
    let inject_domain = |text: &str| -> String {
        if text.trim().is_empty() { return String::new(); }
        text.split(',')
            .map(|s| format!("{} {} {}", page_type, localized_type, s.trim()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    // 1. 추상적 메타 검색 의도 (Core Intent) 강제 주입
    let core_intent = match page_type {
        "tracking" => "tracking, logistics, fulfillment, shipment, dispatch, delivery",
        "goods" => "goods, product, catalog, exposure, traffic, page views, clicks",
        "order" => "order, purchase, sales, conversion rate, volume, checkout, payment, refund",
        "coupon" | "event" => "event, promotion, campaign, exhibition, seasonal, discount, voucher",
        "review" => "review, feedback, rating, customer, complaint",
        _ => ""
    };
    if !core_intent.is_empty() {
        contexts.push((
            "core_search_intent".to_string(),
            inject_domain(core_intent),
            String::new()
        ));
    }
    // 🌟 [비대칭 가중치 반영] bias.json의 insight 블록을 순회하며 특정 도메인(page_type)에만 통계/분석 편향 점수를 폭발시킵니다.
    if let Some(insight_obj) = BIAS_DICT.get("insight").and_then(|v| v.as_object()) {
        for (insight_key, insight_val) in insight_obj {
            let target_domains = insight_val.get("target_domain").and_then(|v| v.as_array()).map(|arr| {
                arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>()
            }).unwrap_or_default();
            if target_domains.is_empty() || target_domains.contains(&page_type) {
                let s_bias = insight_val.get("bias").and_then(|v| v.as_str()).unwrap_or("");
                let s_prej = insight_val.get("prejudice").and_then(|v| v.as_str()).unwrap_or("");
                if !s_bias.is_empty() {
                    contexts.push((
                        insight_key.to_string(),
                        inject_domain(s_bias),
                        inject_domain(s_prej)
                    ));
                }
            }
        }
    }
    // 🌟 [CRITICAL FIX] "ignore" 도메인일 경우 bias.json 최상단의 데이터를 직접 가져옵니다.
    if page_type == "ignore" {
        if let Some(ignore_obj) = BIAS_DICT.get("ignore").and_then(|p| p.as_object()) {
            let s_bias = ignore_obj.get("bias").and_then(|v| v.as_str()).unwrap_or("");
            let s_prej = ignore_obj.get("prejudice").and_then(|v| v.as_str()).unwrap_or("");
            if !s_bias.is_empty() {
                contexts.push((
                    "ignore".to_string(),
                    s_bias.to_string(),
                    s_prej.to_string()
                ));
            }
        }
        return contexts;
    }
    // 2. bias.json 내부의 모든 Key(layout_list, layout_form, 세부 속성 전체)를 하드코딩 없이 동적 순회
    if let Some(localized_obj) = BIAS_DICT
        .get(lang_code)
        .or_else(|| BIAS_DICT.get("en"))
        .and_then(|l| l.get(page_type).or_else(|| l.get("default")))
        .and_then(|p| p.as_object())
    {
        for (key, val) in localized_obj {
            let mut final_bias = String::new();
            let mut final_prej = String::new();
            if let Some(b) = val.get("bias").and_then(|v| v.as_str()) {
                final_bias = inject_domain(&b.replace("{TYPE}", &localized_type));
            }
            if let Some(p) = val.get("prejudice").and_then(|v| v.as_str()) {
                final_prej = inject_domain(&p.replace("{TYPE}", &localized_type));
            }
            if !final_bias.is_empty() {
                contexts.push((key.to_string(), final_bias, final_prej));
            }
        }
    }
    contexts
}

// 🌟 [CRITICAL FIX] 4개의 반환값(String, String, String, String)으로 확장하여 prejudice를 스케줄러로 넘깁니다.
pub fn get_detail_schema_fields(page_type: &str, _href: &str, lang: &str) -> Vec<(String, String, String, String)> {
    let mut fields = Vec::new();
    let lang_code_owned = lang_code_of(lang);
    let lang_code = lang_code_owned.as_str();
    let localized_type = get_localized_page_type(page_type, lang);
    let mut add = |key: &str, field_type: &str, en_bias: &str, en_prejudice: &str| {
        // 🌟 [핵심 변경] 콤마(,)를 기준으로 텍스트를 분리하여 모든 의미 단위(동의어)마다 독립적으로 영어 도메인(page_type)을 부착합니다.
        let inject_domain = |text: &str, domain: &str| -> String {
            if text.trim().is_empty() { return String::new(); }
            text.split(',')
                .map(|s| format!("{} {}", domain, s.trim()))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut final_bias = inject_domain(en_bias, page_type);
        let mut final_prejudice = inject_domain(en_prejudice, page_type);
        let mut semantic_desc = String::new();
        if let Some(localized_obj) = BIAS_DICT
            .get(lang_code)
            .and_then(|l| l.get(page_type).or_else(|| l.get("default")))
            .and_then(|p| p.get(key))
        {
            if let Some(semantic) = localized_obj.get("semantic").and_then(|v| v.as_str()) {
                semantic_desc = semantic.to_string();
            }
            if let Some(bias_str) = localized_obj.get("bias").and_then(|v| v.as_str()) {
                let localized_b = inject_domain(&bias_str.replace("{TYPE}", &localized_type), page_type);
                final_bias = if final_bias.is_empty() { localized_b } else { format!("{}, {}", final_bias, localized_b) };
            }
            if let Some(prejudice_str) = localized_obj.get("prejudice").and_then(|v| v.as_str()) {
                let localized_p = inject_domain(&prejudice_str.replace("{TYPE}", &localized_type), page_type);
                final_prejudice = if final_prejudice.is_empty() { localized_p } else { format!("{}, {}", final_prejudice, localized_p) };
            }
        }
        let final_desc = if key == "id,link" {
            "- \"link\": String. Detailed page URL.\n- \"id\": String. Refer to the ID value from the link.".to_string()
        } else if semantic_desc.is_empty() {
            format!("- \"{}\": {}.", key, field_type)
        } else {
            format!("- \"{}\": {}. {}", key, field_type, semantic_desc)
        };
        fields.push((key.to_string(), final_desc, final_bias.trim().to_string(), final_prejudice.trim().to_string()));
    };
    // 🌟 [비대칭 가중치 반영] bias.json의 insight 블록을 읽어 현재 page_type이 target_domain에 포함된 경우에만 스키마에 추가합니다.
    if let Some(insight_obj) = BIAS_DICT.get("insight").and_then(|v| v.as_object()) {
        for (insight_key, insight_val) in insight_obj {
            let target_domains = insight_val.get("target_domain").and_then(|v| v.as_array()).map(|arr| {
                arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>()
            }).unwrap_or_default();
            // target_domain이 비어있거나(전역), 현재 page_type이 포함된 경우에만 add
            if target_domains.is_empty() || target_domains.contains(&page_type) {
                let s_bias = insight_val.get("bias").and_then(|v| v.as_str()).unwrap_or("");
                let s_prej = insight_val.get("prejudice").and_then(|v| v.as_str()).unwrap_or("");
                add(insight_key, "String", s_bias, s_prej);
            }
        }
    }
    match page_type {
        "tracking" => {
            add("id,link", "", "id link tracking", "");
            add("status", "String", "status delivery", "");
            add("title", "String", "title product name", "");
            add("registration_date", "String", "date registration", "");
            add("shipping_date", "String", "date shipping dispatch", "");
            add("sender_name", "String", "sender seller name", "");
            add("sender_address", "String", "sender address location", "");
            add("sender_phone", "String", "sender phone contact", "");
            add("recipient_name", "String", "recipient buyer name", "");
            add("recipient_address", "String", "recipient address location", "");
            add("recipient_phone", "String", "recipient phone contact", "");
            add("width", "Number", "width dimension", "");
            add("height", "Number", "height dimension", "");
            add("length", "Number", "length dimension", "");
            add("weight", "Number", "weight mass", "");
            add("carrier", "String", "carrier courier", "");
            add("shipping_fee", "Number", "shipping fee cost", "");
            add("shipping_method", "String", "shipping method", "");
            add("shipping_duration", "Number", "shipping duration days", "");
            add("bundle_shipping", "String", "bundle combined shipping", "");
        },
        "goods" => {
            add("id,link", "", "id link", "");
            add("code", "String", "code sku item", "");
            add("status", "String", "status condition", "");
            add("title", "String", "title name product", "");
            add("registration_date", "String", "date registration", "");
            add("payment_method", "String", "payment method", "");
            add("bank", "String", "bank account", "");
            add("card", "String", "card credit", "");
            add("model_name", "String", "model name", "");
            add("brand_name", "String", "brand name", "");
            add("condition", "String", "condition state", "");
            add("description", "String", "description detail", "");
            add("short_description", "String", "short description summary", "");
            add("tags", "Array of Strings", "tags keywords", "");
            add("color", "String", "color hue shade tint", "");
            add("origin_country", "String", "origin country", "");
            add("manufacturer", "String", "manufacturer", "");
            add("release_date", "String", "release date", "");
            add("manufacture_date", "String", "manufacture date", "");
            add("expired_at", "String", "expiration date", "");
            add("gtin", "String", "gtin barcode", "");
            add("mpn", "String", "mpn part number", "");
            add("barcode", "String", "barcode", "");
            add("sale_price", "Number", "sale price discount", "");
            add("supply_price", "Number", "supply price cost", "");
            add("currency", "String", "currency", "");
            add("compare_at_price", "Number", "compare original price", "");
            add("quantity", "Number", "quantity inventory stock", "");
            add("stock_keeping_unit", "String", "sku code", "");
            add("low_stock_threshold", "Number", "low stock threshold", "");
            add("unit", "String", "unit measure", "");
            add("tax_included", "Boolean", "tax vat", "");
            add("tax_code", "String", "tax code", "");
            add("main_image_url", "String", "main image thumbnail", "");
            add("additional_image_url", "String", "additional image", "");
            add("video_url", "String", "video url", "");
            add("carrier", "String", "carrier shipping", "");
            add("shipping_fee", "Number", "shipping fee cost", "");
            add("shipping_method", "String", "shipping method", "");
            add("shipping_duration", "Number", "shipping duration time", "");
            add("bundle_shipping", "String", "bundle combined shipping", "");
            add("options", "Array of Objects [ { value:String, inputs:[ { value:String } ] } ]", "options variations", "");
            add("additional_goods", "Array of Objects [ { link:String, id:String } ]", "additional goods related", "");
        },
        "order" => {
            add("id,link", "", "id link order number", "");
            add("tracking_number", "String", "tracking number", "");
            add("status", "String", "status state", "");
            add("registration_date", "String", "date registration", "");
            add("goods", "Array of Objects [ { title:String, link:String, id:String } ]", "goods product items", "");
            add("sender_name", "String", "sender buyer orderer name", "");
            add("sender_address", "String", "sender buyer address", "");
            add("sender_phone", "String", "sender buyer phone", "");
            add("recipient_name", "String", "recipient receiver name", "");
            add("recipient_address", "String", "recipient receiver address", "");
            add("recipient_phone", "String", "recipient receiver phone", "");
            add("bank", "String", "bank account", "");
            add("card", "String", "card credit", "");
            add("order_date", "String", "order date", "");
            add("payment_date", "String", "payment date", "");
            add("payment_method", "String", "payment method type", "");
            add("payment_origin", "String", "payment origin pg gateway", "");
        },
        "coupon" | "event" => {
            add("id,link", "", "id link", "");
            add("type", "String", "type", "");
            add("status", "String", "status state", "");
            add("title", "String", "title name", "");
            add("started_at", "String", "start date time", "");
            add("expired_at", "String", "expire end date time", "");
            add("code", "String", "code", "");
            add("discount", "Number", "discount", "");
            add("quantity", "Number", "quantity amount", "");
            add("usage_limit", "Number", "usage limit max", "");
            add("usage_per", "Number", "usage per customer user", "");
            add("new_customer_only", "Boolean", "new customer only", "");
            add("first_purchase_only", "Boolean", "first purchase only", "");
            add("min_order_amount", "Number", "minimum order amount", "");
            add("max_order_amount", "Number", "maximum order amount", "");
            add("max_discount_amount", "Number", "maximum discount amount", "");
            add("region_restrictions", "String", "region restriction area", "");
            add("number", "String", "number phone contact", "");
            add("address", "String", "address location", "");
            add("registration_date", "String", "date registration", "");
        },
        "review" => {
            add("id,link", "", "id link", "");
            add("status", "String", "status state", "");
            add("name", "String", "name reviewer author", "");
            add("title", "String", "title subject product", "");
            add("completed", "Boolean", "completed purchased", "");
            add("registration_date", "String", "date registration time", "");
        },
        "click" | "hover" | "change" | "report" => {
            add("id,link", "", "id link url address", "");
            add("action", "String",
                "user action, user intent, clicked item, selected option, entered value, chosen product, pressed button, picked menu",
                "identifier, code, url, link, date, price, quantity, status, address, phone number");
            add("summary", "String",
                "page goal, what the user tried to accomplish, purpose of the visit, task the user was performing",
                "identifier, code, url, link, date, price, quantity, status");
            add("relate", "Array of Strings",
                "neighbouring items, sibling options, alternatives not selected, surrounding list, competing choices",
                "identifier, code, url, link, date, price, status");
            add("cross_action_flow", "String",
                "overall behaviour flow, journey across pages, sequence of actions, path taken",
                "identifier, code, url, link, date, price, status");
            add("intent_evolution", "String",
                "how the goal changed over time, shifting objective, evolving purpose",
                "identifier, code, url, link, date, price, status");
            add("consistent_preferences", "String",
                "repeated preference, recurring choice, habitual attribute, favourite option",
                "identifier, code, url, link, date, price, status");
        },
        "shipping_doc"
        | "BL" | "AWB" | "CI" | "PI" | "PL" | "PO" | "SC" | "LC" | "CO"
        | "SA" | "DO" | "AN" | "BC" | "ED" | "ID" | "CINV"
        | "IC" | "WC" | "CA" | "PHYTO" | "HC" | "BEN_CERT"
        | "DGD" | "MSDS" | "POA" | "BIZ_LIC" | "INS" => {
            add("id,link", "", "id link document", "");
            add("doc_type", "String", "document type kind form", "");
            add("doc_number", "String", "document number identifier", "");
            add("no", "String", "tracking number reference", "");
            add("status", "String", "status state", "");
            add("issue_date", "String", "issue date", "");
            add("expiry_date", "String", "expiry date", "");
            add("sender_name", "String", "shipper seller exporter name", "");
            add("sender_address", "String", "shipper address", "");
            add("recipient_name", "String", "consignee buyer importer name", "");
            add("recipient_address", "String", "consignee address", "");
            add("notify_party_name", "String", "notify party name", "");
            add("vessel", "String", "vessel flight carrier", "");
            add("voyage_number", "String", "voyage flight leg number", "");
            add("pol", "String", "port of loading origin departure", "");
            add("pod", "String", "port of discharge destination arrival", "");
            add("place_receipt", "String", "place of receipt", "");
            add("place_delivery", "String", "place of delivery", "");
            add("etd", "String", "estimated time of departure", "");
            add("eta", "String", "estimated time of arrival", "");
            add("transport_mode", "String", "sea air road rail", "");
            add("incoterms", "String", "incoterms fob cif exw ddp dap", "");
            add("payment_terms", "String", "payment terms", "");
            add("freight_payment_term", "String", "freight prepaid collect", "");
            add("currency", "String", "currency", "");
            add("amount", "Number", "total amount", "");
            add("freight_amount", "Number", "freight charges", "");
            add("insurance_amount", "Number", "insurance charges", "");
            add("local_charges", "Number", "local handling charges", "");
            add("container_number", "String", "container number", "");
            add("seal_number", "String", "seal number", "");
            add("package_count", "Number", "package carton count", "");
            add("weight_gross", "Number", "gross weight", "");
            add("weight_net", "Number", "net weight", "");
            add("volume", "Number", "volume cbm measurement", "");
            add("hs_code", "String", "hs code tariff number", "");
            add("marks_numbers", "String", "shipping marks and numbers", "");
            add("reference_invoice", "String", "referenced invoice number", "");
            add("reference_lc", "String", "referenced letter of credit number", "");
            add("reference_booking", "String", "referenced booking number", "");
        },
        _ => {
            add("id,link", "", "id link", "");
            add("title", "String", "title name", "");
            add("status", "String", "status state", "");
            add("registration_date", "String", "date registration", "");
        }
    }
    fields
}
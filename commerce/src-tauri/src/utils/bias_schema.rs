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

pub const TRADE_DOC_TYPES: &[&str] = &[
    // ── 기존 27종 ──
    "BL", "AWB", "CI", "PI", "PL", "PO", "SC", "LC", "CO",
    "SA", "DO", "AN", "BC", "ED", "ID", "CINV",
    "IC", "WC", "CA", "PHYTO", "HC", "BEN_CERT",
    "DGD", "MSDS", "POA", "BIZ_LIC", "INS",
    // ── trade_schema.overlay 에는 있었으나 목록에 누락되었던 18종 ──
    "HBL",  // House Bill of Lading
    "FCR",  // Forwarder's Certificate of Receipt
    "POD",  // Proof of Delivery
    "LG",   // Letter of Guarantee
    "TR",   // Trust Receipt
    "LLC",  // Local Letter of Credit
    "TI",   // Tax Invoice
    "CP",   // Confirmation of Purchase
    "CM",   // Cargo Manifest
    "CCC",  // Customs Clearance Certificate
    "CNM",  // Certificate of Non-Manipulation
    "PC",   // Phytosanitary Certificate
    "COA",  // Certificate of Analysis
    "FI",   // Freight Invoice
    "CDR",  // Cargo Damage / Survey Report
    "ICF",  // Insurance Claim Form
    "SOA",  // Statement of Account
    "EL",   // Export License
    // ── 무역 공용 사전으로 처리 가능한 10종 ──
    //    개념이 갈리지 않으므로(SWB 의 pol 과 BL 의 pol 은 같은 개념)
    //    공용 노드를 그대로 쓰는 것이 정확도를 해치지 않습니다.
    "BE",   // Bill of Exchange
    "SR",   // Shipping Request
    "BK",   // Booking Confirmation
    "WR",   // Warehouse Receipt
    "CSI",  // Consular Invoice
    "SWB",  // Sea Waybill
    "IP",   // Insurance Policy
    "DN",   // Debit Note
    "CN",   // Credit Note
    "FC",   // Fumigation Certificate
];

/// 🌟 이 타입이 무역 서식인지 판정합니다.
///
///  ── 왜 대소문자를 무시하는가 ──
///   같은 문서가 저장 경로에 따라 두 형태로 들어옵니다.
///     · 로컬 추출 (scheduler/trading.rs → save_item) : 'BL'  (원형 보존)
///     · 서버 동기화 (lib.rs upsert_items)            : 'bl'  (강제 소문자)
///   upsert_items 의 `.trim().to_lowercase()` 가 원인이며, 그 한 줄 때문에
///   canonical_bias_type 이 'bl' 을 무역 서식으로 인식하지 못했습니다.
///
///  ── 실측 피해 ──
///   get_detail_schema_fields('bl') 이 무역 스키마를 로드하지 못해
///   index_item_chunks 가 조기 종료되고, 서버에서 내려온 무역 문서는
///   item_chunks 가 단 한 건도 생성되지 않았습니다.
///   (STAGE-4 필드 레벨 검색이 그 문서에 물리적으로 도달할 수 없습니다)
///
///  ── 왜 여기서 흡수하는가 ──
///   저장 경로를 통일해도(아래 수정 ②) 이미 소문자로 저장된 기존 행이 남습니다.
///   판정 함수가 두 형태를 모두 받아 주는 편이 마이그레이션과 무관하게 안전합니다.
pub fn is_trade_doc_type(page_type: &str) -> bool {
    let t = page_type.trim();
    if t.is_empty() { return false; }
    if t.eq_ignore_ascii_case("shipping_doc") { return true; }
    TRADE_DOC_TYPES.iter().any(|c| c.eq_ignore_ascii_case(t))
}

/// 🌟 무역 서식 코드를 저장 표준형(대문자)으로 정규화합니다.
///    bias.json 의 trade_schema.overlay 키가 대문자이므로 그것을 표준으로 삼습니다.
///    무역 서식이 아니면 None 을 돌려주어 호출부가 기존 소문자 규칙을 쓰게 합니다.
pub fn canonical_trade_doc_code(page_type: &str) -> Option<&'static str> {
    let t = page_type.trim();
    if t.is_empty() { return None; }
    TRADE_DOC_TYPES.iter().find(|c| c.eq_ignore_ascii_case(t)).copied()
}

pub fn canonical_bias_type(page_type: &str) -> &str {
    if is_trade_doc_type(page_type) {
        "shipping_doc"
    } else {
        page_type
    }
}

/// 🌟 [TRADE SCHEMA LOADER] bias.json 의 trade_schema 를 읽어
///    (category, field, description) 삼중항을 돌려줍니다.
///
///  ── 왜 코드가 아니라 데이터가 소유해야 하는가 ──
///   기존에는 이 목록이 get_detail_schema_fields 의 shipping_doc 분기에
///   `add(...)` 호출로 하드코딩되어 있었고, 같은 목록이 bias.json 의
///   trade_schema 에도 따로 존재했습니다.
///   두 목록이 어긋난 결과가 실측 실패였습니다.
///     · ⓑ에만 있던 items 6필드 → 비전 앵커 부재 → 표를 못 찾음
///     · ⓐ에만 있던 place_receipt → 히트맵 질량만 먹고 결과 기여 0
///   진실의 원천을 bias.json 하나로 고정하면 이 어긋남이 물리적으로 불가능해집니다.
///
///  ── 조건부 로드 ──
///   doc_type 은 STEP 2 에서 이미 확정되어 이 함수에 들어옵니다.
///   base 전체 + overlay[doc_type] 만 로드하므로,
///   CI 인보이스에 위험물(un_number) · 식물검역(botanical_name) 앵커가
///   올라가 히트맵을 오염시키는 일이 사라집니다.
///
///  ── 배열 카테고리 ──
///   items / containers / parties / charges 등은 원소 스키마입니다.
///   호출부(vision_encoder)가 카테고리 단위로 히트맵을 만들므로
///   여기서는 평평한 필드 목록으로 돌려주고, 배열 여부는
///   is_trade_array_category() 가 별도로 답합니다.
pub fn trade_schema_triples(doc_type: &str) -> Vec<(String, String, String)> {
    let mut out: Vec<(String, String, String)> = Vec::new();

    let node = match BIAS_DICT.get("trade_schema") {
        Some(n) => n,
        None => return out,
    };

    let mut absorb = |cat_obj: &Value, out: &mut Vec<(String, String, String)>| {
        let map = match cat_obj.as_object() { Some(m) => m, None => return };
        for (category, fields) in map {
            let fmap = match fields.as_object() { Some(f) => f, None => continue };
            for (field, desc) in fmap {
                let d = desc.as_str().unwrap_or("").to_string();
                // overlay 가 base 필드를 덮어쓰는 경우 설명만 교체합니다.
                if let Some(slot) = out.iter_mut()
                    .find(|(c, f, _)| c == category && f == field)
                {
                    if !d.trim().is_empty() { slot.2 = d; }
                    continue;
                }
                out.push((category.clone(), field.clone(), d));
            }
        }
    };

    if let Some(base) = node.get("base") {
        absorb(base, &mut out);
    }

    // 🌟 조건부 : 이 서식의 overlay 만 얹습니다.
    if let Some(ov) = node.get("overlay").and_then(|o| o.get(doc_type)) {
        absorb(ov, &mut out);
    }

    out
}

/// 🌟 [ALIAS-AWARE TRIPLES] trade_schema 삼중항에 '정준 필드 별칭' 을 앵커로 병합합니다.
///
///  ── 왜 여기서 하는가 ──
///   별칭은 스키마 필드가 아니라 '그 필드를 부르는 다른 이름' 입니다.
///   필드 목록에 넣으면 근거가 갈라지고, 프롬프트에 넣으면 모델이 중복 출력합니다.
///   앵커 문자열에 섞는 것이 유일하게 안전한 위치입니다.
pub fn canonical_trade_triples(doc_type: &str) -> Vec<(String, String, String)> {
    let mut triples = trade_schema_triples(doc_type);

    // 🌟 [ALIAS TABLE] 45종 예시 서식 대조에서 뽑은 '같은 개념의 다른 이름' 입니다.
    //    특정 서식에 종속되지 않는 표준 표기이므로 어휘 하드코딩이 아니라
    //    무역 서식 표기 규약의 목록입니다.
    const ALIAS: &[(&str, &str)] = &[
        ("doc_number",
         "invoice number, commercial invoice number, bill of lading number, B/L no, \
          air waybill number, AWB no, sea waybill number, house B/L number, HBL no, \
          purchase order number, PO no, proforma invoice number, PI no, \
          letter of credit number, L/C no, local L/C number, delivery order number, DO no, \
          booking number, shipping request number, arrival notice number, shipping advice number, \
          packing list number, certificate number, cert no, manifest number, \
          declaration number, export declaration no, import declaration no, license number, \
          policy number, claim number, statement number, tax invoice number, contract number, \
          warehouse receipt number, forwarder's certificate of receipt number, FCR no, \
          proof of delivery number, trust receipt number, bill of exchange number, \
          consular invoice number, debit note number, credit note number, \
          confirmation number, survey report number, visa number"),
        ("package_count",
         "number of packages, total packages, packages delivered, packages received, \
          number of pieces, total pieces, no. of pkgs, CTNS, PKGS"),
        ("weight_gross",
         "total gross weight, gross weight kg, G.W., cargo gross weight, \
          verified gross mass, VGM, chargeable weight"),
        ("weight_net",
         "total net weight, net weight kg, N.W., cargo net weight"),
        ("volume",
         "total measurement, measurement CBM, cubic meter, M3, measurement"),
        ("amount",
         "total amount, grand total, grand total amount, total invoice amount, \
          total charge, total charges, total amount due, total amount paid, invoice amount, \
          total invoice value, total FOB amount, total declared amount, \
          total authorized value, total claim amount, total credit amount, total debit amount"),
        ("amount_subtotal",
         "sub total, subtotal, total items amount, total supply amount"),
        ("amount_tax",
         "total VAT amount, VAT amount, tax amount, supply amount"),
        ("freight_payment_term",
         "freight term, freight prepaid, freight collect"),
        ("issue_date",
         "submission date, date of issue, issued on"),
    ];

    for (canon, extra) in ALIAS.iter() {
        if let Some(slot) = triples.iter_mut().find(|(_, f, _)| f == canon) {
            // 설명문 뒤에 별칭을 이어 붙입니다. 타입 마커는 trade_desc_to_type 이
            // 원문에서 읽으므로 순서가 바뀌어도 안전합니다.
            if !slot.2.contains(extra) {
                slot.2 = format!("{}, {}", slot.2.trim(), extra);
            }
        }
    }

    triples
}

/// bias.json 설명문에서 타입 마커를 떼어 앵커 문자열로 만듭니다.
fn trade_desc_to_anchor(desc: &str) -> String {
    let mut s = desc.to_string();
    for m in ["{String}", "{Number}", "{Boolean}", "{Array}"] {
        s = s.replace(m, " ");
    }
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// bias.json 설명문의 타입 마커를 스키마 타입 문자열로 변환합니다.
fn trade_desc_to_type(desc: &str) -> &'static str {
    if desc.contains("{Number}") { "Number" }
    else if desc.contains("{Boolean}") { "Boolean" }
    else if desc.contains("{Array}") { "Array of Strings" }
    else { "String" }
}

/// 🌟 이 카테고리가 배열(반복 행) 스키마인지 답합니다.
///
///  ── 판정 근거 ──
///   trade_schema 의 값 형태가 아니라 '개념이 반복되는가' 입니다.
///   items(품목 행) / containers(컨테이너 행) / parties(당사자 역할) /
///   charges(요금 항목) / test_results(시험 항목) 등은
///   한 문서에 여러 개가 인쇄되는 것이 정상입니다.
///   스칼라로 다루면 두 번째 이후가 통째로 소실됩니다.
///   (실측: 상품 표 2행 중 Shorts 행 소실)
pub fn is_trade_array_category(category: &str) -> bool {
    matches!(
        category,
        "items"
            | "containers"
            | "parties"
            | "charges"
            | "test_results"
            | "findings_and_damage"
            | "account_ledger"
            | "adjustments"
            | "packing_details"
            | "licensed_items"
            | "purchased_items"
    )
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

        let bias_type_key = canonical_bias_type(page_type);
        if let Some(localized_obj) = BIAS_DICT
            .get(lang_code)
            .and_then(|l| {
                l.get(page_type)
                    .or_else(|| if bias_type_key != page_type { l.get(bias_type_key) } else { None })
                    .or_else(|| l.get("default"))
            })
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
        // 🌟 [BIAS TYPE CANONICALIZE] 무역 서식 코드(BL/CI/PL...)는 bias.json 에
        //    개별 노드가 없으므로 공용 'shipping_doc' 노드로 접어서 조회합니다.
        let bias_type_key = canonical_bias_type(page_type);
        if let Some(localized_obj) = BIAS_DICT
            .get(lang_code)
            .and_then(|l| {
                l.get(page_type)
                    .or_else(|| if bias_type_key != page_type { l.get(bias_type_key) } else { None })
                    .or_else(|| l.get("default"))
            })
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
        | "DGD" | "MSDS" | "POA" | "BIZ_LIC" | "INS"
        | "HBL" | "FCR" | "POD" | "LG" | "TR" | "LLC" | "TI" | "CP" | "CM"
        | "CCC" | "CNM" | "PC" | "COA" | "FI" | "CDR" | "ICF" | "SOA" | "EL"
        | "BE" | "SR" | "BK" | "WR" | "CSI" | "SWB" | "IP" | "DN" | "CN" | "FC" => {
            // ── 봉투/시스템 축 (trade_schema 에는 없지만 저장 파이프라인이 요구) ──
            add("id,link", "", "id link document", "");
            add("status", "String", "status state", "");

            // ── trade_schema 에서 조건부 로드 ──
            let triples = canonical_trade_triples(page_type);
            if triples.is_empty() {
                // bias.json 에 trade_schema 노드가 없는 극단 상황의 최소 방어선입니다.
                add("doc_type", "String", "document type kind form", "");
                add("doc_number", "String", "document number identifier", "");
                add("issue_date", "String", "issue date", "");
            } else {
                let mut loaded_cats: Vec<String> = Vec::new();
                for (category, field, desc) in triples.iter() {
                    if !loaded_cats.iter().any(|c| c == category) {
                        loaded_cats.push(category.clone());
                    }
                    let en_anchor = trade_desc_to_anchor(desc);
                    let field_type = trade_desc_to_type(desc);
                    add(field, field_type, &en_anchor, "");
                }
                                println!(
                    "[SCHEMA] 🚢 '{}' 조건부 로드: 카테고리 {}개 | 필드 {}개 (base + overlay)",
                    page_type,
                    loaded_cats.len(),
                    triples.len()
                );
            }
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
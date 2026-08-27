use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QueryInfo {
    pub table: String,
    pub r#type: String,
    pub column: String,
    pub value: Value,
    pub status: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MergeInfo {
    pub update: Option<UpdateMerge>,
    pub upsert: Option<UpsertMerge>,
    pub from: String,
    pub to: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UpdateMerge {
    pub includes: Vec<String>,
    pub column: Option<String>,
    pub value: Option<Value>,
    pub foreign: Option<ForeignInfo>,
    pub from: String,
    pub to: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UpsertMerge {
    pub includes: Vec<String>,
    pub from: String,
    pub to: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ForeignInfo {
    pub from: String,
    pub to: String,
}

#[allow(dead_code)]

pub fn parse_status(status: &str) -> i32 {

    match status {

        "progress" => 1,

        "stop" => 2,

        "cancel" => 3,

        "refund" => 4,

        "return" => 5,

        "error" => 6,

        "expire" => 7,

        "exchange" => 8,

        "complete" => 9,

        "draft" => 10,

        "show" => 11,

        "hide" => 12,

        _ => 0,

    }

}



pub fn related(item_type: &str) -> Vec<&str> {
    let t = match item_type {
        "receiving" | "shipping" => "tracking",
        "sales" => "order",
        _ => item_type
    };
    match t {
        "goods" => vec!["order", "tracking", "coupon", "event"],
        "order" => vec!["goods", "tracking", "coupon", "event"],
        "tracking" => vec!["goods", "order", "coupon", "event"],
        "coupon" => vec!["goods", "event"],
        "event" => vec!["goods", "coupon"],
        "review" => vec!["goods", "coupon", "event"],
        _ => vec![],
    }
}

/// 🌟 [TRADE RELAY] 무역 서식 간 연결고리 규칙입니다.
/// Commerce의 relay()가 order↔tracking을 tracking_number로 연결하듯,
/// 무역 서식은 reference_invoice / reference_lc / reference_booking / container_number로 연결합니다.
///
/// 반환값: (연결 대상 서식 타입, 조회할 필드명, 현재 문서에서 가져올 값 필드명)
pub fn trade_relay_rules(doc_type: &str) -> Vec<(&'static str, &'static str, &'static str)> {
    match doc_type {
        // CI가 추출되면 → PL/BL/ED가 CI를 참조하는지 역방향 조회
        "CI" => vec![
            ("PL",  "reference_invoice", "doc_number"),
            ("BL",  "reference_invoice", "doc_number"),
            ("ED",  "reference_invoice", "doc_number"),
        ],
        // PL이 추출되면 → CI를 정방향 조회
        "PL" => vec![
            ("CI",  "doc_number", "reference_invoice"),
            ("BL",  "reference_invoice", "reference_invoice"),
        ],
        // BL이 추출되면 → CI/PL/BC를 조회
        "BL" => vec![
            ("CI",  "doc_number", "reference_invoice"),
            ("PL",  "reference_invoice", "reference_invoice"),
            ("BC",  "doc_number", "reference_booking"),
        ],
        // LC가 추출되면 → CI가 LC를 참조하는지 역방향 조회
        "LC" => vec![
            ("CI",  "reference_lc", "doc_number"),
        ],
        // BC가 추출되면 → BL이 BC를 참조하는지 역방향 조회
        "BC" => vec![
            ("BL",  "reference_booking", "doc_number"),
        ],
        // ED/ID가 추출되면 → CI를 정방향 조회
        "ED" | "ID" | "CINV" => vec![
            ("CI",  "doc_number", "reference_invoice"),
        ],
        // CO, SA, DO, AN 등 기타 서식
        "CO" | "SA" | "DO" | "AN" => vec![
            ("CI",  "doc_number", "reference_invoice"),
            ("BL",  "reference_invoice", "reference_invoice"),
        ],
        _ => vec![],
    }
}

/// 🌟 [TRADE RELATED TYPES] 관련 서식 타입 목록 (N:N 교차 검색용)
pub fn trade_related(doc_type: &str) -> Vec<&'static str> {
    match doc_type {
        "CI" => vec!["PL", "BL", "LC", "BC", "ED", "CO"],
        "PL" => vec!["CI", "BL", "ED"],
        "BL" => vec!["CI", "PL", "BC", "AN", "DO"],
        "LC" => vec!["CI", "BL"],
        "BC" => vec!["BL", "CI"],
        "ED" | "ID" | "CINV" => vec!["CI", "PL", "BL"],
        "CO" => vec!["CI", "BL"],
        "SA" | "DO" | "AN" => vec!["BL", "CI"],
        _ => vec![],
    }
}



// 🌟 [TRADING RELAY] 무역 서식 간 N:N 관계 정의.
//    commerce 의 related() 와 동일한 구조이지만,
//    무역 서식 코드(BL/AWB/CI/PI/PL/PO/SC/LC/CO 등)를 키로 사용합니다.
//
// 관계 규칙:
//   BL  → CI, PL       : reference_invoice / reference_booking
//   CI  → BL, PL, LC   : reference_invoice / reference_lc
//   PL  → BL, CI       : reference_invoice / reference_booking
//   PO  → PI, SC       : doc_number
//   PI  → PO, SC       : doc_number
//   SC  → PO, PI       : doc_number
//   LC  → CI           : reference_lc
//   CO  → CI           : reference_invoice
/// 🌟 [TRADING HUB] 45종 데이터셋의 참조 그래프는 4개 허브 키를 경유합니다.
///   PO  = 거래 시작점            (PO-99281A)
///   CI  = 물품 명세 / 대금 청구  (CI-2026-08001)
///   BL  = 화물 소유권 / 운송     (BL-55432219)
///   LC  = 대금 결제 보증         (LC-88492011)
/// 어떤 서식이든 이 4개는 항상 후보로 둡니다. 실제 연결 여부는
/// trading_relay_pair 가 돌려주는 참조 필드에 값이 있는지로 결정되므로,
/// 후보를 넓게 두어도 헛도는 쿼리가 생기지 않습니다.
pub const TRADE_HUB_TYPES: [&str; 4] = ["PO", "CI", "BL", "LC"];

pub fn related_trading(doc_type: &str) -> Vec<&'static str> {
    // ── ① 서식별 직속 상대 (허브 이외의 근접 관계) ──
    let direct: Vec<&'static str> = match doc_type {
        // 계약 · 결제
        "PO"      => vec!["PI", "SC", "EL", "CP", "LLC", "SOA"],
        "PI"      => vec!["SC", "EL"],
        "SC"      => vec!["PI", "EL"],
        "LC"      => vec!["LLC", "LG", "TR", "SOA"],
        "LLC"     => vec!["CP", "TI"],
        "CP"      => vec!["ED", "TI", "LLC"],

        // 선적 · 운송
        "CI"      => vec!["PL", "CINV", "CSI", "CO", "ED", "ID", "FI", "SOA"],
        "PL"      => vec!["ED", "ID", "WC", "CM"],
        "BL"      => vec!["HBL", "SWB", "PL", "DO", "AN", "BC", "CM", "FI", "LG", "TR", "CCC", "CDR"],
        "HBL"     => vec!["FCR", "BC"],
        "SWB"     => vec!["DO", "AN"],
        "AWB"     => vec!["PL", "DGD"],
        "BC"      => vec!["FI", "HBL"],
        "SA"      => vec!["PL"],
        "DO"      => vec!["AN", "POD", "LG"],
        "AN"      => vec!["DO", "FI"],
        "FCR"     => vec!["HBL"],
        "POD"     => vec!["DO", "CDR"],
        "CM"      => vec!["ED"],
        "FI"      => vec!["BC", "AN"],

        // 통관 · 신고
        "ED"      => vec!["PL", "CO", "CP", "CM", "EL"],
        "ID"      => vec!["PL", "CO", "CCC"],
        "CINV"    => vec!["CO"],
        "CO"      => vec!["CNM", "ED", "ID", "CCC"],
        "EL"      => vec!["SC", "PI", "ED"],
        "CCC"     => vec!["ID", "CO"],

        // 검사 · 증명
        "IC"      => vec!["COA", "WC"],
        "WC"      => vec!["PL", "IC"],
        "CA"      => vec!["IC"],
        "COA"     => vec!["IC"],
        "PHYTO"   => vec!["FC"],
        "PC"      => vec!["FC"],
        "HC"      => vec!["IC"],
        "BEN_CERT"=> vec![],
        "FC"      => vec!["PHYTO", "PC"],
        "CNM"     => vec!["CO"],

        // 특수 · 법무 · 금융
        "DGD"     => vec!["MSDS", "AWB"],
        "MSDS"    => vec!["DGD"],
        "POA"     => vec!["BIZ_LIC"],
        "BIZ_LIC" => vec!["POA"],
        "INS"     => vec!["IP", "CDR", "ICF"],
        "IP"      => vec!["CDR", "ICF"],
        "LG"      => vec!["TR", "DO"],
        "TR"      => vec!["LG"],
        "CDR"     => vec!["IP", "ICF", "POD"],
        "ICF"     => vec!["IP", "CDR", "SOA"],
        "SOA"     => vec!["DN", "CN", "ICF", "FI", "TI"],
        "DN"      => vec!["SOA"],
        "CN"      => vec!["SOA"],
        "TI"      => vec!["CP", "LLC", "SOA"],
        "CSI"     => vec!["CO"],

        _         => vec![],
    };

    // ── ② 허브 4종 병합 (자기 자신은 제외) ──
    let mut out: Vec<&'static str> = Vec::with_capacity(direct.len() + TRADE_HUB_TYPES.len());
    for d in direct {
        if d == doc_type { continue; }
        if !out.iter().any(|x| *x == d) { out.push(d); }
    }
    for h in TRADE_HUB_TYPES.iter() {
        if *h == doc_type { continue; }
        if !out.iter().any(|x| x == h) { out.push(*h); }
    }
    out
}

/// 🌟 [TRADE REFERENCE FIELD] 이 서식을 '다른 문서가 가리킬 때' 사용하는 참조 필드명입니다.
///  ── 계약 ──
///   BL 문서 안에 있는 "CI-2026-08001" 은 data.reference_invoice 에 담깁니다.
///   CI 문서 안에 있는 "BL-55432219" 은 data.reference_bl 에 담깁니다.
///   즉 필드명은 '가리켜지는 쪽' 의 서식으로 결정되며, 방향이 뒤집힐 여지가 없습니다.
///
///  ── 왜 함수 하나로 접는가 ──
///   45종 × 45종 = 2,025 조합을 손으로 적으면 서식이 하나 늘 때마다 90줄을 추가해야 합니다.
///   '가리켜지는 서식 → 필드명' 이라는 단방향 사전 하나면 조합이 자동으로 생성됩니다.
pub fn trade_reference_field_of(doc_type: &str) -> Option<&'static str> {
    let f = match doc_type {
        // ── 계약 · 결제 ──
        "PO"       => "reference_po",
        "PI"       => "reference_proforma",
        "SC"       => "reference_contract",
        "LC"       => "reference_lc",
        "LLC"      => "reference_local_lc",
        "CP"       => "reference_purchase_confirm",

        // ── 상거래 · 선적 ──
        "CI"       => "reference_invoice",
        "CINV"     => "reference_customs_invoice",
        "CSI"      => "reference_consular_invoice",
        "PL"       => "reference_packing",
        "BL"       => "reference_bl",
        "HBL"      => "reference_hbl",
        "SWB"      => "reference_swb",
        "AWB"      => "reference_awb",
        "BC" | "BK"=> "reference_booking",
        "SA"       => "reference_shipping_advice",
        "DO"       => "reference_do",
        "AN"       => "reference_arrival_notice",
        "FCR"      => "reference_fcr",
        "POD"      => "reference_pod",
        "CM"       => "reference_manifest",
        "FI"       => "reference_freight_invoice",

        // ── 통관 · 신고 ──
        "ED"       => "reference_export_decl",
        "ID"       => "reference_import_decl",
        "CO"       => "reference_origin",
        "EL"       => "reference_export_license",
        "CCC"      => "reference_customs_clearance",

        // ── 검사 · 증명 ──
        "IC"       => "reference_inspection",
        "WC"       => "reference_weight",
        "CA"       => "reference_analysis",
        "COA"      => "reference_analysis",
        "PHYTO"    => "reference_phyto",
        "PC"       => "reference_phyto",
        "HC"       => "reference_health",
        "BEN_CERT" => "reference_beneficiary",
        "FC"       => "reference_fumigation",
        "CNM"      => "reference_non_manipulation",

        // ── 특수 · 법무 · 금융 ──
        "DGD"      => "reference_dgd",
        "MSDS"     => "reference_msds",
        "POA"      => "reference_poa",
        "BIZ_LIC"  => "reference_biz_license",
        "INS"      => "reference_policy",
        "IP"       => "reference_policy",
        "LG"       => "reference_lg",
        "TR"       => "reference_tr",
        "CDR"      => "reference_survey",
        "ICF"      => "reference_claim",
        "SOA"      => "reference_statement",
        "DN"       => "reference_debit_note",
        "CN"       => "reference_credit_note",
        "TI"       => "reference_tax_invoice",

        _ => return None,
    };
    Some(f)
}

/// 🌟 [ALL REFERENCE FIELDS] 무역 문서가 가질 수 있는 모든 참조 축 목록입니다.
///  STEP C 정규화(FLATTEN)와 검색 조건 화이트리스트가 같은 목록을 공유해야
///  저장(정방향)과 조회(역방향)가 같은 이름 공간에서 만납니다.
pub const TRADE_REFERENCE_FIELDS: [&str; 48] = [
    "reference_po", "reference_proforma", "reference_contract", "reference_lc",
    "reference_local_lc", "reference_purchase_confirm",
    "reference_invoice", "reference_customs_invoice", "reference_consular_invoice",
    "reference_packing", "reference_bl", "reference_hbl", "reference_swb",
    "reference_awb", "reference_booking", "reference_shipping_advice",
    "reference_do", "reference_arrival_notice", "reference_fcr", "reference_pod",
    "reference_manifest", "reference_freight_invoice",
    "reference_export_decl", "reference_import_decl", "reference_origin",
    "reference_export_license", "reference_customs_clearance",
    "reference_inspection", "reference_weight", "reference_analysis",
    "reference_phyto", "reference_health", "reference_beneficiary",
    "reference_fumigation", "reference_non_manipulation",
    "reference_dgd", "reference_msds", "reference_poa", "reference_biz_license",
    "reference_policy", "reference_lg", "reference_tr",
    "reference_survey", "reference_claim",
    // 🌟 [FINANCE AXIS] Part 39 / 45 의 정산 계열입니다.
    //    trade_reference_field_of 가 이 4축을 이미 반환하는데 배열에서 빠져 있어,
    //    trade_condition_fields("reference") 순회에서 제외되어
    //    SOA / DN / CN / TI 질의가 조건 축을 찾지 못했습니다.
    "reference_statement", "reference_debit_note",
    "reference_credit_note", "reference_tax_invoice",
];

/// 🌟 [RELAY PAIR v2 / DIRECTION-FIXED]
///  반환값 계약:
///    .0 (mine_field)    = 내 문서 data 에서 '상대의 doc_number' 가 들어 있는 필드
///    .1 (foreign_field) = 상대 문서 data 에서 '내 doc_number' 가 들어 있는 필드
///
///  ── v1 의 결함 ──
///   ("CI","BL") 이 ("doc_number", "reference_invoice") 였습니다.
///   그러면 scheduler 가 crc32(hash("BL" + team + CI의 doc_number)) 를 만들어
///   BL 의 실제 index(= crc32(hash("BL" + team + BL의 doc_number))) 와
///   구조적으로 절대 일치할 수 없었습니다. (log: CI.rel_bl = 4100281351)
///
///  ── v2 ──
///   ("CI","BL") → ("reference_bl", "reference_invoice")
///   CI.reference_bl 에 담긴 "BL-55432219" 로 BL 의 index 를 정확히 재현합니다.
pub fn trading_relay_pair(from_type: &str, to_type: &str) -> Option<(&'static str, &'static str)> {
    if from_type == to_type { return None; }
    let mine = trade_reference_field_of(to_type)?;    // 상대를 가리키는 내 필드
    let foreign = trade_reference_field_of(from_type)?; // 나를 가리키는 상대 필드
    Some((mine, foreign))
}

// 🌟 [BACK-COMPAT] 기존 호출부(trading_relay_field)를 살려 둡니다.
//  '내 쪽 필드'만 반환하므로 v1 과 동일한 시그니처로 동작합니다.
pub fn trading_relay_field(from_type: &str, to_type: &str) -> Option<&'static str> {
    trading_relay_pair(from_type, to_type).map(|(mine, _)| mine)
}


pub fn trading_index_column(doc_type: &str) -> String {
    format!("rel_{}", doc_type.to_lowercase())
}

pub fn relay(foreign_type: &str, primary_item: &Value) -> Option<(Vec<QueryInfo>, MergeInfo)> {
    let mut primary_type = primary_item.get("type")?.as_str()?;

    if primary_type == "sales" { primary_type = "order"; }

    let f_type = if foreign_type == "receiving" || foreign_type == "shipping" { "tracking" } else { foreign_type };
    let mut queries = Vec::new();
    let get_val = |key: &str| -> Option<Value> { primary_item.get(key).cloned() };

    let sales_includes = vec![
        "event", "width", "height", "length", "weight", "size", "currency", 
        "cost_price", "sale_price", "discount", "quantity", "tracking", 
        "number", "carrier", "shipping_fee", "shipping_method", "shipping_duration", 
        "fulfillment_service", "stock_keeping_unit", "bundle_shipping", "used", 
        "lease", "rental", "refurbish", "tax_included", "release_date"
    ].into_iter().map(String::from).collect::<Vec<_>>();

    let (merge_from, merge_to) = (f_type.to_string(), primary_type.to_string());

    match (f_type, primary_type) {
        // --- Order as Primary ---
        ("goods", "order") => {
            if let Some(tracking) = get_val("tracking").or_else(|| get_val("tracking_number")) {
                queries.push(QueryInfo { r#type: primary_type.to_string(), table: "sales".to_string(), column: "tracking".to_string(), value: tracking, status: None });
                return Some((queries, MergeInfo { update: None, upsert: Some(UpsertMerge { includes: sales_includes, from: merge_from.clone(), to: merge_to.clone() }), from: merge_from, to: merge_to }));
            } else {

                let index_val = get_val("index")?;

                queries.push(QueryInfo { r#type: primary_type.to_string(), table: "sales".to_string(), column: "index".to_string(), value: index_val.clone(), status: None });

                return Some((queries, MergeInfo { upsert: None, update: Some(UpdateMerge { includes: sales_includes, column: Some("index".to_string()), value: Some(index_val), foreign: None, from: merge_from.clone(), to: merge_to.clone() }), from: merge_from, to: merge_to }));
            }
        },
        ("tracking", "order") => {
            let index_val = get_val("index")?;

            if get_val("tracking").is_some() || get_val("tracking_number").is_some() {
                queries.push(QueryInfo { r#type: f_type.to_string(), table: "tracking".to_string(), column: primary_type.to_string(), value: index_val.clone(), status: None });

                return Some((queries, MergeInfo { upsert: None, update: Some(UpdateMerge { 
                    includes: vec!["width", "height", "length", "weight"].into_iter().map(String::from).collect(), 
                    column: Some("index".to_string()), value: Some(index_val), 
                    foreign: Some(ForeignInfo { from: "index".to_string(), to: "tracking".to_string() }),
                    from: merge_to.clone(), to: merge_from.clone()
                }), from: merge_from, to: merge_to }));

            } else {
                queries.push(QueryInfo { r#type: f_type.to_string(), table: "tracking".to_string(), column: primary_type.to_string(), value: index_val.clone(), status: None });

                return Some((queries, MergeInfo { upsert: None, update: Some(UpdateMerge { 
                    includes: vec!["no", "goods", "event"].into_iter().map(String::from).collect(),
                    column: Some("index".to_string()), value: Some(index_val), 
                    foreign: Some(ForeignInfo { from: "index".to_string(), to: "tracking".to_string() }),
                    from: merge_from.clone(), to: merge_to.clone()
                }), from: merge_from, to: merge_to }));
            }
        },
        ("coupon" | "event", "order") => {
            let event_val = get_val("event")?;

            queries.push(QueryInfo { r#type: f_type.to_string(), table: "event".to_string(), column: "index".to_string(), value: event_val, status: None });

            return Some((queries, MergeInfo { upsert: None, update: Some(UpdateMerge {
                includes: vec!["discount".to_string()], column: Some("index".to_string()), value: Some(get_val("index")?), 
                foreign: None, from: merge_from.clone(), to: merge_to.clone() 
            }), from: merge_from, to: merge_to }));
        },
        ("order", "goods") => {
            let index_val = get_val("index")?;

            queries.push(QueryInfo { r#type: f_type.to_string(), table: "sales".to_string(), column: "goods".to_string(), value: index_val.clone(), status: None });

            return Some((queries, MergeInfo { upsert: None, update: Some(UpdateMerge { 
                includes: sales_includes, column: Some("goods".to_string()), value: Some(index_val), 
                foreign: None, from: merge_to.clone(), to: merge_from.clone() 
            }), from: merge_from, to: merge_to }));
        },
        ("tracking", "goods") => {
            queries.push(QueryInfo { r#type: "order".to_string(), table: "tracking".to_string(), column: "goods".to_string(), value: get_val("index")?, status: Some(0) });

            return Some((queries, MergeInfo { upsert: None, update: Some(UpdateMerge { 
                includes: vec!["width", "height", "length", "weight", "shipping_fee", "shipping_method", "shipping_duration", "bundle_shipping"].into_iter().map(String::from).collect(),
                column: None, value: None, foreign: None, from: merge_to.clone(), to: merge_from.clone() 
            }), from: merge_from, to: merge_to }));
        },
        ("coupon" | "event", "goods") => {
            let event_val = get_val("event")?;

            queries.push(QueryInfo { r#type: f_type.to_string(), table: "event".to_string(), column: "index".to_string(), value: event_val, status: None });

            return Some((queries, MergeInfo { upsert: None, update: Some(UpdateMerge { 
                includes: vec!["discount".to_string()], column: Some("index".to_string()), value: Some(get_val("index")?), 
                foreign: None, from: merge_from.clone(), to: merge_to.clone() 
            }), from: merge_from, to: merge_to }));
        },
        ("goods", "tracking") => {
             queries.push(QueryInfo { r#type: "order".to_string(), table: "sales".to_string(), column: "goods".to_string(), value: get_val("goods")?, status: Some(0) });

             return Some((queries, MergeInfo { upsert: None, update: Some(UpdateMerge { 
                includes: vec!["width", "height", "length", "weight", "shipping_fee", "shipping_method", "shipping_duration", "bundle_shipping"].into_iter().map(String::from).collect(),
                column: Some("index".to_string()), value: Some(get_val("index")?), 
                foreign: None, 
                from: merge_from.clone(), to: merge_to.clone() 
            }), from: merge_from, to: merge_to }));
        },
        ("order", "tracking") => {
            if let Some(goods_val) = get_val("goods") {

                queries.push(QueryInfo { r#type: f_type.to_string(), table: "sales".to_string(), column: "goods".to_string(), value: goods_val, status: None });

                return Some((queries, MergeInfo { upsert: None, update: Some(UpdateMerge { 
                    includes: vec!["width", "height", "length", "weight", "shipping_fee", "shipping_method", "shipping_duration", "bundle_shipping"].into_iter().map(String::from).collect(),
                    column: Some("tracking".to_string()), value: Some(get_val("index")?), 
                    foreign: Some(ForeignInfo { from: "index".to_string(), to: "tracking".to_string() }), 
                    from: merge_to.clone(), to: merge_from.clone() 
                }), from: merge_from, to: merge_to }));
            } else {
                queries.push(QueryInfo { r#type: f_type.to_string(), table: "tracking".to_string(), column: primary_type.to_string(), value: get_val("index")?, status: None });

                return Some((queries, MergeInfo { upsert: None, update: Some(UpdateMerge { 
                    includes: vec!["no", "order", "goods", "event"].into_iter().map(String::from).collect(),
                    column: Some("index".to_string()), value: Some(get_val("index")?), 
                    foreign: Some(ForeignInfo { from: "index".to_string(), to: "order".to_string() }), 
                    from: merge_from.clone(), to: merge_to.clone() 
                }), from: merge_from, to: merge_to }));
            }
        },
        ("goods", "coupon" | "event") => {
            queries.push(QueryInfo { r#type: f_type.to_string(), table: "sales".to_string(), column: "event".to_string(), value: get_val("index")?, status: None });

            return Some((queries, MergeInfo { upsert: None, update: None, from: merge_to.clone(), to: merge_from.clone() }));
        },
        ("order", "coupon" | "event") => {
             queries.push(QueryInfo { r#type: f_type.to_string(), table: "sales".to_string(), column: "event".to_string(), value: get_val("index")?, status: Some(0) });

             return Some((queries, MergeInfo { upsert: None, update: Some(UpdateMerge { 
                includes: vec!["discount".to_string()], column: Some("event".to_string()), value: Some(get_val("index")?), 
                foreign: None, from: merge_to.clone(), to: merge_from.clone() 
            }), from: merge_to.clone(), to: merge_from.clone() }));
        },
        ("event", "coupon") => {
            if let Some(event_val) = get_val("event") {
                queries.push(QueryInfo { r#type: f_type.to_string(), table: "event".to_string(), column: "index".to_string(), value: event_val, status: None });

                return Some((queries, MergeInfo { upsert: None, update: Some(UpdateMerge { 
                    includes: vec!["started_at", "expired_at", "phone", "address", "discount", "quantity", "usage_per", "usage_limit", "min_order_amount", "max_order_amount", "max_discount_amount", "new_customer_only", "first_purchase_only", "region_restrictions"].into_iter().map(String::from).collect(), 
                    column: Some("index".to_string()), value: Some(get_val("index")?), 
                    foreign: None, from: merge_from.clone(), to: merge_to.clone() 
                }), from: merge_from, to: merge_to }));
            }
            None
        },
        _ => None,
    }
}

// =====================================================================
// 🌟 [TRADE CONDITION BANK] 무역 검색 질의를 2뎁스로 좁히기 위한 앵커 뱅크
// ---------------------------------------------------------------------
//  ── 왜 필요한가 ──
//   기존 extract_shipping_conditions 는 44개 필드 + 변환 규칙을 한 프롬프트에
//   통째로 넣고 2B 모델에게 "알아서 골라라" 라고 시켰습니다.
//   scheduler.rs STEP A 가 27개 서식 코드를 한 번에 묻지 않고
//   '그룹 → 코드' 2뎁스로 좁히는 것과 정반대 구조입니다.
//
//  ── v3 구조 (STEP A 와 동일 계보) ──
//   Depth 1 : 질의 청크가 어느 '조건 카테고리' 인가          (7갈래)
//   Depth 2 : 그 카테고리 안에서 어느 '파라미터' 인가         (평균 6~13갈래)
//   Depth 3 : 마진이 부족할 때만, 그 카테고리 필드만 담은 소형 프롬프트로 LLM 1회
//
//  ── 채점 방식 ──
//   ai_utils::surprisal_dual_scores 를 그대로 재사용합니다.
//     surprisal = (max - μ_global)/σ_global - √(2 ln N)
//   뱅크 크기 편향(Cross References 44구 vs Parties 3구)이 제거되므로
//   구 개수가 많은 카테고리가 구조적으로 유리해지지 않습니다.
// =====================================================================

/// Depth 1 : 조건 카테고리 앵커.
///  편견(prejudice)은 별도 사전을 만들지 않고 '다른 카테고리의 bias' 를 그대로 씁니다.
///  (get_detail_schema_fields 가 다른 필드의 bias 를 편견으로 쓰는 것과 동일 원리)
pub const TRADE_CONDITION_CATEGORIES: [(&str, &str); 7] = [
    ("identity",
     "document kind, document type code, document number, bill of lading number, invoice number, purchase order number, contract number, tracking number, parcel number, reference number, document status, draft, in progress, completed, returned, error, issue date, date of issue, expiry date, validity date"),
    ("transport",
     "vessel name, mother vessel, ocean vessel, flight number, voyage number, voyage leg, port of loading, loading port, port of departure, port of discharge, discharge port, port of destination, place of receipt, place of delivery, estimated time of departure, estimated time of arrival, sailing date, arrival date, transport mode, sea freight, air freight, road, rail"),
    ("parties",
     "shipper, exporter, seller, supplier, vendor, consignor, consignee, importer, buyer, receiver, notify party, beneficiary, applicant, company name, trading partner"),
    ("terms",
     "incoterms, trade terms, price terms, delivery terms, FOB, CIF, EXW, DDP, DAP, CFR, CPT, CIP, payment terms, T/T, letter of credit payment, net 30, freight prepaid, freight collect, currency, ISO currency code, USD, EUR, JPY, KRW, total amount, invoice value, freight charges, insurance charges, local handling charges"),
    ("cargo",
     "container number, seal number, package count, carton count, pallet count, number of packages, gross weight, net weight, volume, measurement, cubic meter, CBM, HS code, tariff number, harmonized code, shipping marks, marks and numbers"),
    ("reference",
     "referenced invoice number, referenced bill of lading number, referenced purchase order number, referenced letter of credit number, referenced booking number, referenced contract number, referenced declaration number, referenced certificate number, referenced policy number, covering document, against document, relating to document, our reference, your reference"),
    ("hub",
     "trace everything related to this number, show every document under this number, all documents linked to, entire document set for, whole paperwork bundle, everything tied to this order, all paperwork for this shipment, full document chain"),
];

/// Depth 2 : 카테고리별 파라미터 (필드명, 프롬프트 설명, 앵커 구).
///  ── 설계 원칙 ──
///   ① 필드명은 저장(get_trade_category_schema) 과 동일해야 합니다.
///      그래야 저장과 조회가 alias 없이 바로 만납니다.
///   ② 앵커 구에는 값 예시를 포함시킵니다.
///      'BL-55432219' 같은 실제 번호가 질의에 그대로 등장하기 때문입니다.
pub fn trade_condition_fields(category: &str) -> Vec<(&'static str, &'static str, &'static str)> {
    match category {
        "identity" => vec![
            ("doc_type",    "Document kind code",
             "document type, document kind, bill of lading, air waybill, commercial invoice, packing list, purchase order, sales contract, letter of credit, certificate of origin, export declaration, import declaration, delivery order, arrival notice, booking confirmation"),
            ("doc_number",  "Primary identifier OF THE DOCUMENT ITSELF",
             "document number, doc no, our number, this document number, BL-55432219, CI-2026-08001, PO-99281A, LC-88492011"),
            ("no",          "Tracking number, parcel number, or generic reference number",
             "tracking number, parcel number, waybill number, generic number, 603145678912"),
            ("status",      "Document / shipping status",
             "status, draft, in progress, in transit, returned, completed, delivered, error, cancelled"),
            ("issue_date",  "Date the document was issued",
             "issue date, date of issue, issued on, drawn on, 2026-08-26"),
            ("expiry_date", "Expiry date (mainly L/C)",
             "expiry date, expiration, valid until, latest date, 2026-09-30"),
        ],
        "transport" => vec![
            ("vessel",         "Vessel name or Flight number",
             "vessel, vessel name, ocean vessel, mother vessel, flight number, OCEAN VOYAGER, MAERSK, MSC, HMM, EVERGREEN"),
            ("voyage_number",  "Voyage or flight leg number",
             "voyage number, voyage, flight leg, V.123E"),
            ("pol",            "Port of Loading, Origin, Departure point",
             "port of loading, loading port, departure port, origin, BUSAN, INCHEON, SHANGHAI, NINGBO, SINGAPORE"),
            ("pod",            "Port of Discharge, Destination, Arrival point",
             "port of discharge, discharge port, destination port, arrival port, LOS ANGELES, LONG BEACH, NEW YORK, ROTTERDAM"),
            ("place_receipt",  "Place of Receipt",
             "place of receipt, received at, pickup place"),
            ("place_delivery", "Place of Delivery",
             "place of delivery, final delivery place, door delivery"),
            ("etd",            "Estimated Time of Departure",
             "estimated time of departure, ETD, sailing date, departure date, on board date"),
            ("eta",            "Estimated Time of Arrival",
             "estimated time of arrival, ETA, arrival date, expected arrival"),
            ("transport_mode", "Sea, Air, Road, or Rail",
             "transport mode, by sea, by air, ocean freight, air freight, road, rail, multimodal"),
        ],
        "parties" => vec![
            ("sender_name",       "Shipper, Seller, Exporter, or Vendor name",
             "shipper, exporter, seller, supplier, consignor, vendor, beneficiary"),
            ("recipient_name",    "Consignee, Buyer, or Importer name",
             "consignee, importer, buyer, receiver, applicant, to order of"),
            ("notify_party_name", "Notify Party name",
             "notify party, notify, also notify"),
        ],
        "terms" => vec![
            ("incoterms",            "Incoterms code",
             "incoterms, trade terms, price terms, FOB, CIF, EXW, DDP, DAP, CFR, CPT, CIP, FCA, FAS, DPU"),
            ("payment_terms",        "Payment condition",
             "payment terms, T/T, telegraphic transfer, letter of credit, net 30, at sight, D/A, D/P"),
            ("freight_payment_term", "Freight Prepaid or Freight Collect",
             "freight prepaid, freight collect, prepaid, collect"),
            ("currency",             "ISO 4217 currency code",
             "currency, USD, EUR, JPY, CNY, KRW, GBP, dollars, euro"),
            ("amount",               "Total financial amount",
             "total amount, grand total, invoice value, total value, amount"),
            ("freight_amount",       "Freight charges only",
             "freight charges, ocean freight, air freight charge, freight amount"),
            ("insurance_amount",     "Insurance charges only",
             "insurance charges, insurance premium, insured amount"),
            ("local_charges",        "Local handling charges",
             "local charges, terminal handling charge, THC, documentation fee, handling charge"),
        ],
        "cargo" => vec![
            ("container_number", "Container number (4 letters + 7 digits)",
             "container number, container no, CNTR, PONU1234567, MSCU1234567"),
            ("seal_number",      "Seal number",
             "seal number, seal no, SEAL876543210"),
            ("package_count",    "Number of packages or cartons",
             "package count, number of packages, cartons, CTNS, PKGS, pallets, PLT"),
            ("weight_gross",     "Gross weight",
             "gross weight, G.W., total gross weight, KGS"),
            ("weight_net",       "Net weight",
             "net weight, N.W., total net weight"),
            ("volume",           "Volume in CBM",
             "volume, measurement, CBM, cubic meter, M3"),
            ("hs_code",          "HS Code or tariff number",
             "HS code, tariff number, harmonized code, HTS, 8543.70"),
            ("marks_numbers",    "Shipping marks and numbers",
             "marks and numbers, shipping marks, case marks, N/M"),
        ],
        "reference" => {
            let mut out: Vec<(&'static str, &'static str, &'static str)> = Vec::new();
            for f in TRADE_REFERENCE_FIELDS.iter() {
                out.push((f, "Referenced document number", trade_reference_anchor(f)));
            }
            out
        },
        "hub" => vec![
            ("hub_reference", "A document number to trace ACROSS every related document",
             "everything related to, all documents under, whole paperwork for, entire document chain of, PO-99281A, CI-2026-08001, BL-55432219, LC-88492011"),
        ],
        _ => vec![],
    }
}

/// Depth 2 보조 : 참조 축 하나의 앵커 구입니다.
///  값 예시(실제 데이터셋 번호)를 포함시켜야
///  '무역서류 CI-2026-08001' 같은 질의가 올바른 축으로 떨어집니다.
pub fn trade_reference_anchor(field: &str) -> &'static str {
    match field {
        "reference_po"                => "referenced purchase order number, against P/O, our P/O, order number, PO-99281A",
        "reference_proforma"          => "referenced proforma invoice number, against proforma, PI-2026-0801",
        "reference_contract"          => "referenced sales contract number, against contract, SC-2026-0802",
        "reference_lc"                => "referenced letter of credit number, against L/C, documentary credit number, LC-88492011",
        "reference_local_lc"          => "referenced local letter of credit number, LLC-2026-KR-0911",
        "reference_purchase_confirm"  => "referenced purchase confirmation number, CP-2026-KR-0419",
        "reference_invoice"           => "referenced commercial invoice number, against invoice, covering invoice, CI-2026-08001",
        "reference_customs_invoice"   => "referenced customs invoice number",
        "reference_consular_invoice"  => "referenced consular invoice number, CSI-2026-US-0827",
        "reference_packing"           => "referenced packing list number",
        "reference_bl"                => "referenced bill of lading number, against B/L, covering B/L, BL-55432219",
        "reference_hbl"               => "referenced house bill of lading number, HBL-55432219-01",
        "reference_swb"               => "referenced sea waybill number, SWB-55432219",
        "reference_awb"               => "referenced air waybill number, AWB-180-99281014",
        "reference_booking"           => "referenced booking number, against booking, BK-2026-0822",
        "reference_shipping_advice"   => "referenced shipping advice number",
        "reference_do"                => "referenced delivery order number, DO-SFO-20260911",
        "reference_arrival_notice"    => "referenced arrival notice number",
        "reference_fcr"               => "referenced forwarder cargo receipt number, FCR-2026-0827",
        "reference_pod"               => "referenced proof of delivery number, POD-SFO-20260912",
        "reference_manifest"          => "referenced cargo manifest number, CM-2026-0828",
        "reference_freight_invoice"   => "referenced freight invoice number, FI-2026-0828",
        "reference_export_decl"       => "referenced export declaration number, ED-2026-KR-77102",
        "reference_import_decl"       => "referenced import declaration number, ID-2026-US-99120",
        "reference_origin"            => "referenced certificate of origin number, CO-2026-KR-0801",
        "reference_export_license"    => "referenced export license number, EL-2026-KR-0815",
        "reference_customs_clearance" => "referenced customs clearance certificate number, CCC-2026-US-99120",
        "reference_inspection"        => "referenced inspection certificate number, IC-2026-0825",
        "reference_weight"            => "referenced weight certificate number, WC-2026-0826",
        "reference_analysis"          => "referenced certificate of analysis number, COA-2026-0824",
        "reference_phyto"             => "referenced phytosanitary certificate number, PC-2026-KR-0826",
        "reference_health"            => "referenced health certificate number",
        "reference_beneficiary"       => "referenced beneficiary certificate number",
        "reference_fumigation"        => "referenced fumigation certificate number, FC-2026-0825",
        "reference_non_manipulation"  => "referenced non manipulation certificate number, CNM-2026-SG-0902",
        "reference_dgd"               => "referenced dangerous goods declaration number, DGD-2026-0827",
        "reference_msds"              => "referenced material safety data sheet number",
        "reference_poa"               => "referenced power of attorney number",
        "reference_biz_license"       => "referenced business license number",
        "reference_policy"            => "referenced insurance policy number, IP-2026-08200",
        "reference_lg"                => "referenced letter of guarantee number, LG-SFO-20260909",
        "reference_tr"                => "referenced trust receipt number, TR-SFO-20260910",
        "reference_survey"            => "referenced cargo damage survey report number, CDR-2026-SFO-0912",
        "reference_claim"             => "referenced insurance claim number, ICF-2026-0914",
        "reference_statement"         => "referenced statement of account number, settlement statement, SOA-2026-0920",
        "reference_debit_note"        => "referenced debit note number, DN-2026-0912",
        "reference_credit_note"       => "referenced credit note number, CN-2026-0915",
        "reference_tax_invoice"       => "referenced tax invoice number, VAT invoice number, TI-2026-KR-0812",
        _                             => "referenced document number",
    }
}

/// 🌟 [TRADE OPERATOR HINT] 필드가 요구하는 기본 연산자입니다.
///  Depth 3 프롬프트에서 모델이 연산자를 창작하지 못하도록 미리 고정합니다.
///  ai_utils::detect_field_format 과 동일 계보의 결정론 판정입니다.
pub fn trade_default_operator(field: &str) -> &'static str {
    if field == "hub_reference" { return "contains"; }
    if field.starts_with("reference_") { return "eq"; }
    match field {
        "doc_number" | "no" | "status" | "doc_type"
        | "container_number" | "seal_number" | "hs_code"
        | "incoterms" | "currency" | "freight_payment_term" => "eq",

        "issue_date" | "expiry_date" | "etd" | "eta" => "gte",

        "amount" | "freight_amount" | "insurance_amount" | "local_charges"
        | "package_count" | "weight_gross" | "weight_net" | "volume" => "eq",

        _ => "contains",
    }
}

// =====================================================================
// 🌟 [DOC TYPE ANCHOR — 텍스트/비전 공용]
// ---------------------------------------------------------------------
//  ── 왜 여기로 옮기는가 ──
//   기존에는 scheduler.rs 의 process_trading_task 안에
//   지역 const TRADE_GROUPS / GROUP_CODES / fn trade_code_anchor 로 박혀 있었습니다.
//   그래서 비전 파이프라인(models/siglip2/vision_encoder.rs)이 같은 사전을
//   쓰려면 복제해야 했고, 서식이 하나 늘 때마다 두 곳을 고쳐야 했습니다.
//   판정 근거는 하나여야 하므로 logic.rs 로 승격합니다.
//
//  ── 사용처 ──
//   · scheduler.rs STEP A          : PUG 라인 임베딩 채점 (텍스트 트랙)
//   · siglip2/vision_encoder.rs    : 이미지 패치 임베딩 채점 (비전 트랙)
// =====================================================================

/// Depth 1 : 서식 그룹 앵커. 편견은 '다른 그룹의 bias' 를 그대로 씁니다.
pub const TRADE_GROUPS: [(&str, &str); 6] = [
    ("contract",  "purchase order, proforma invoice, sales contract, letter of credit, documentary credit, payment terms, contract number, buyer seller agreement, tenor at sight, issuing bank, advising bank, beneficiary, applicant, order confirmation, quotation"),
    ("shipping",  "commercial invoice, packing list, bill of lading, ocean bill of lading, air waybill, shipping advice, delivery order, arrival notice, booking confirmation, vessel voyage number, port of loading, port of discharge, place of receipt, place of delivery, container number, seal number, notify party, freight prepaid, freight collect, shipper and consignee, gross weight net weight measurement, carton quantity, marks and numbers, incoterms fob cif exw"),
    ("customs",   "export declaration, import declaration, customs invoice, certificate of origin, hs code, tariff classification, customs clearance, declaration number, customs value, duty and tax, chamber of commerce, country of origin"),
    ("inspection","inspection certificate, weight certificate, certificate of analysis, phytosanitary certificate, health certificate, beneficiary certificate, we hereby certify, test result, specification value, fumigation treatment, laboratory report, fit for human consumption, plant health"),
    ("legal",     "dangerous goods declaration, material safety data sheet, power of attorney, business license, insurance policy, un number, proper shipping name, packing group, hazard class, policy number, insured amount, premium, coverage all risks, attorney in fact, business registration number"),
    ("parcel",    "courier label, parcel waybill sticker, domestic courier service, home delivery parcel, door to door small package, delivery driver, barcode sticker label, parcel pickup, last mile delivery"),
];

/// Depth 2 : 그룹 소속 코드 목록.
pub const TRADE_GROUP_CODES: [(&str, &[&str]); 6] = [
    ("contract",   &["PO", "PI", "SC", "LC"]),
    ("shipping",   &["CI", "PL", "BL", "AWB", "SA", "DO", "AN", "BC"]),
    ("customs",    &["ED", "ID", "CINV", "CO"]),
    ("inspection", &["IC", "WC", "CA", "PHYTO", "HC", "BEN_CERT"]),
    ("legal",      &["DGD", "MSDS", "POA", "BIZ_LIC", "INS"]),
    ("parcel",     &["TRACKING"]),
];

/// 🌟 [VISION CHROME] 이미지에만 존재하는 시각 노이즈 앵커.
///  텍스트(PUG) 트랙에는 없던 축입니다.
///  로고 / 도장 / 서명 / 표 괘선 / 여백 / QR 은 문서 면적의 상당수를 차지하지만
///  어떤 스키마 필드의 값도 아닙니다. 모든 그룹·코드·카테고리의 공통 편견입니다.
pub const VISION_CHROME_ANCHOR: &str =
    "company logo, brand emblem, letterhead graphic, official round stamp, red seal, \
     handwritten signature, watermark, blank paper, empty margin, page border, table grid lines, \
     ruled lines, barcode stripes, QR code square, page number footer, printed form template, \
     decorative frame, background texture, scanned paper noise, staple hole, punch hole";

/// Depth 2 보조 : 서식 코드 하나의 앵커 구.
pub fn trade_code_anchor(code: &str) -> &'static str {
    match code {
        "PO"       => "purchase order, order confirmation, buyer issues to seller, order number, delivery date requested",
        "PI"       => "proforma invoice, quotation, preliminary invoice, offer to buyer before shipment",
        "SC"       => "sales contract, agreement between seller and buyer, contract terms and clauses",
        "LC"       => "letter of credit, documentary credit, issuing bank, beneficiary, tenor at sight, expiry date, advising bank",
        "CI"       => "commercial invoice, seller bills buyer, unit price, total amount, incoterms, invoice number",
        "PL"       => "packing list, carton details, gross weight, net weight, measurement, marks and numbers",
        "BL"       => "bill of lading, ocean carrier document, shipper consignee notify party, vessel voyage, port of loading, port of discharge, freight prepaid collect",
        "AWB"      => "air waybill, airline document, flight number, airport of departure, airport of destination, chargeable weight",
        "SA"       => "shipping advice, shipment notification to buyer, dispatch details",
        "DO"       => "delivery order, release cargo to consignee, pickup location, container release",
        "AN"       => "arrival notice, cargo arrival notification, local charges, free time, terminal",
        "BC"       => "booking confirmation, space booking with carrier, booking number, cut off time",
        "ED"       => "export declaration, customs export filing, declaration number, exporter, hs code",
        "ID"       => "import declaration, customs import filing, importer, duty, tax, hs code",
        "CINV"     => "customs invoice, invoice prepared for customs valuation",
        "CO"       => "certificate of origin, country of origin declaration, chamber of commerce stamp",
        "IC"       => "inspection certificate, quality inspection result, inspected by",
        "WC"       => "weight certificate, certified weight measurement",
        "CA"       => "certificate of analysis, laboratory test result, specification value",
        "PHYTO"    => "phytosanitary certificate, plant health, fumigation, treatment type",
        "HC"       => "health certificate, sanitary certificate, fit for human consumption",
        "BEN_CERT" => "beneficiary certificate, beneficiary statement, we hereby certify that",
        "DGD"      => "dangerous goods declaration, un number, proper shipping name, packing group, hazard class",
        "MSDS"     => "material safety data sheet, chemical hazard information, first aid measures",
        "POA"      => "power of attorney, authorization letter, attorney in fact",
        "BIZ_LIC"  => "business license, business registration certificate, company registration number",
        "INS"      => "insurance policy, marine cargo insurance, insured amount, premium, coverage all risks",
        "TRACKING" => "courier parcel label, tracking number barcode sticker, domestic courier service, home delivery small package, delivery driver route",
        _          => "trade document",
    }
}

/// 🌟 [FIELD → CATEGORY] 스키마 필드가 어느 추출 카테고리에 속하는지 판정합니다.
///
///  ── 왜 logic.rs 인가 ──
///   기존에는 scheduler.rs 의 process_trading_task 안에
///   지역 fn trade_field_category 로 박혀 있어서
///   비전 히트맵(카테고리 단위)이 같은 매핑을 쓸 수 없었습니다.
///   저장 스키마(get_trade_category_schema)와 히트맵 축이 어긋나면
///   크롭 영역과 추출 프롬프트가 서로 다른 필드를 가리키게 됩니다.
pub fn trade_field_category(field: &str) -> &'static str {
    if field.starts_with("reference_") {
        return "header";
    }
    match field {
        "doc_type" | "doc_number" | "issue_date" | "expiry_date"
            | "reference_number" | "no" | "status" => "header",
        "sender_name" | "sender_address" | "recipient_name"
            | "recipient_address" | "notify_party_name" => "parties",
        "vessel" | "voyage_number" | "pol" | "pod" | "place_receipt"
            | "place_delivery" | "etd" | "eta" | "transport_mode" => "logistics",
        "incoterms" | "payment_terms" | "freight_payment_term" => "conditions",
        "currency" | "amount" | "amount_subtotal" | "amount_tax"
            | "freight_amount" | "insurance_amount" | "local_charges" => "financials",
        "container_number" | "seal_number" | "package_count" | "package_unit"
            | "weight_gross" | "weight_net" | "volume" | "marks_numbers" => "cargo",
        "hs_code" => "items",
        _ => "",
    }
}

/// 🌟 [EXTRACTION CATEGORIES] 비전 크롭 + LLM 추출이 순회하는 카테고리 목록.
///
///  parsing.rs 의 get_trade_category_schema 가 소비하는 8개와 동일합니다.
///  items / containers 는 배열 스키마이며, 히트맵으로 위치를 찾은 뒤
///  그 영역 전체를 크롭해 표를 통째로 읽힙니다.
pub const TRADE_EXTRACTION_CATEGORIES: [&str; 8] = [
    "header", "parties", "logistics", "conditions",
    "financials", "cargo", "items", "containers",
];
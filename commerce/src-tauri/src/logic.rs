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
// 🌟 [DEPRECATED] trade_relay_rules 는 parsing.rs 의 plan_trade_relays 로 대체됩니다.
//    기존은 서식 코드마다 하드코딩된 (target, target_field, source_field) 튜플을
//    반환했는데, 필드 이름이 추출 결과의 실제 키와 어긋나면 릴레이가 성립하지 않았습니다.
//    (실측: "BL←doc_number(빈 키)" 가 4건 반복)
//
//    plan_trade_relays 는 역할 기반으로 릴레이 대상을 계산합니다.
//    같은 역할을 공유하면 서식 코드가 달라도 연결됩니다.
//    이 함수는 하위 호환을 위해 남겨두지만, 새 코드에서는 사용하지 마십시오.
#[deprecated(
    since = "relay-v4",
    note = "parsing.rs 의 plan_trade_relays 를 사용하십시오. 이 함수는 역할 기반이 아니라 서식 코드 하드코딩입니다."
)]
pub fn trade_relay_rules(doc_type: &str) -> Vec<(&'static str, &'static str, &'static str)> {
    match doc_type {
        "CI" => vec![
            ("PL",  "reference_invoice", "doc_number"),
            ("BL",  "reference_invoice", "doc_number"),
            ("ED",  "reference_invoice", "doc_number"),
        ],
        "PL" => vec![
            ("CI",  "doc_number", "reference_invoice"),
            ("BL",  "reference_invoice", "reference_invoice"),
        ],
        "BL" => vec![
            ("CI",  "doc_number", "reference_invoice"),
            ("PL",  "reference_invoice", "reference_invoice"),
            ("BC",  "doc_number", "reference_booking"),
        ],
        "LC" => vec![
            ("CI",  "reference_lc", "doc_number"),
        ],
        "BC" => vec![
            ("BL",  "reference_booking", "doc_number"),
        ],
        "ED" | "ID" | "CINV" => vec![
            ("CI",  "doc_number", "reference_invoice"),
        ],
        "CO" | "SA" | "DO" | "AN" => vec![
            ("CI",  "doc_number", "reference_invoice"),
            ("BL",  "reference_invoice", "reference_invoice"),
        ],
        "HBL" => vec![
            ("BL",  "reference_master_bl", "doc_number"),
            ("FCR", "reference_hbl", "doc_number"),
            ("CI",  "doc_number", "reference_invoice"),
        ],
        "SWB" => vec![
            ("CI",  "doc_number", "reference_invoice"),
            ("PL",  "reference_invoice", "reference_invoice"),
            ("DO",  "reference_swb", "doc_number"),
            ("AN",  "reference_swb", "doc_number"),
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
    // 🌟 [MISSING 10] 사용자 지적 '미포함 서식' 중
    //    related_trading 에 없던 10종을 추가합니다.
    //    BE / SR / BK / WR / CSI / SWB / IP / DN / CN / FC
    let direct: Vec<&'static str> = match doc_type {
        // 계약 · 결제
        "PO"      => vec!["PI", "SC", "EL", "CP", "LLC", "SOA"],
        "PI"      => vec!["SC", "EL"],
        "SC"      => vec!["PI", "EL"],
        "LC"      => vec!["LLC", "LG", "TR", "SOA"],
        "LLC"     => vec!["CP", "TI"],
        "CP"      => vec!["ED", "TI", "LLC"],
        "BE"      => vec!["LC", "LLC", "SOA"],
        // 선적 · 운송
        "CI"      => vec!["PL", "CINV", "CSI", "CO", "ED", "ID", "FI", "SOA"],
        "PL"      => vec!["ED", "ID", "WC", "CM"],
        "BL"      => vec!["HBL", "SWB", "PL", "DO", "AN", "BC", "CM", "FI", "LG", "TR", "CCC", "CDR"],
        "HBL"     => vec!["FCR", "BC"],
        "SWB"     => vec!["DO", "AN", "CI", "PL"],
        "AWB"     => vec!["PL", "DGD"],
        "BC"      => vec!["FI", "HBL", "BK"],
        "BK"      => vec!["BC", "FI", "BL"],
        "SR"      => vec!["BK", "BC"],
        "SA"      => vec!["PL"],
        "DO"      => vec!["AN", "POD", "LG"],
        "AN"      => vec!["DO", "FI"],
        "FCR"     => vec!["HBL"],
        "POD"     => vec!["DO", "CDR"],
        "CM"      => vec!["ED"],
        "FI"      => vec!["BC", "AN"],
        "WR"      => vec!["DO", "POD"],
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
        "IP"      => vec!["CDR", "ICF", "SOA"],
        "LG"      => vec!["TR", "DO"],
        "TR"      => vec!["LG"],
        "CDR"     => vec!["IP", "ICF", "SOA"],
        "ICF"     => vec!["IP", "CDR", "SOA"],
        "SOA"     => vec!["DN", "CN", "ICF", "FI", "TI"],
        "DN"      => vec!["SOA"],
        "CN"      => vec!["SOA"],
        "TI"      => vec!["CP", "LLC", "SOA"],
        "CSI"     => vec!["CO"],
        _         => vec![],
    };
    // ...

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
        "PO" => "reference_po",
        "PI" => "reference_proforma",
        "SC" => "reference_contract",
        "LC" => "reference_lc",
        "LLC" => "reference_local_lc",
        "CP" => "reference_purchase_confirm",
        "BE" => "reference_bill_of_exchange",
        "TR" => "reference_tr",
        "LG" => "reference_lg",
        "EL" => "reference_export_license",
        // ── 상거래 · 선적 ──
        "CI" => "reference_invoice",
        "CINV" => "reference_customs_invoice",
        "CSI" => "reference_consular_invoice",
        "PL" => "reference_packing",
        "BL" => "reference_bl",
        "HBL" => "reference_hbl",
        "SWB" => "reference_swb",
        "AWB" => "reference_awb",
        "BC" | "BK" => "reference_booking",
        "SA" => "reference_shipping_advice",
        "DO" => "reference_do",
        "AN" => "reference_arrival_notice",
        "FCR" => "reference_fcr",
        "POD" => "reference_pod",
        "CM" => "reference_manifest",
        "FI" => "reference_freight_invoice",
        "WR" => "reference_warehouse_receipt",
        "SR" => "reference_sr",
        // ── 통관 · 신고 ──
        "ED" => "reference_export_decl",
        "ID" => "reference_import_decl",
        "CO" => "reference_origin",
        "CCC" => "reference_customs_clearance",
        "CNM" => "reference_non_manipulation",
        // ── 검사 · 증명 ──
        "IC" => "reference_inspection",
        "WC" => "reference_weight",
        "CA" | "COA" => "reference_analysis",
        "PHYTO" | "PC" => "reference_phyto",
        "HC" => "reference_health",
        "BEN_CERT" => "reference_beneficiary",
        "FC" => "reference_fumigation",
        "CDR" => "reference_survey",
        // ── 특수 · 법무 · 금융 ──
        "DGD" => "reference_dgd",
        "MSDS" => "reference_msds",
        "POA" => "reference_poa",
        "BIZ_LIC" => "reference_biz_license",
        "INS" | "IP" => "reference_policy",
        "ICF" => "reference_claim",
        // ── 정산 ──
        "SOA" => "reference_statement",
        "DN" => "reference_debit_note",
        "CN" => "reference_credit_note",
        "TI" => "reference_tax_invoice",
        _ => return None,
    };
    Some(f)
}

/// 🌟 [DOC TYPE TO CODE] 문서 전체 이름을 코드로 변환합니다.
///  저장 시 `type_`은 전체 이름(예: "COMMERCIAL INVOICE")으로 설정되지만,
///  릴레이 검색 시 `target_type`은 코드(예: "CI", "BL")입니다.
///  타입 검증 시 이 둘을 매칭하기 위해 이 함수가 필요합니다.
pub fn doc_type_to_code(doc_type: &str) -> String {
    match doc_type.to_uppercase().as_str() {
        "COMMERCIAL INVOICE" => "CI".to_string(),
        "PROFORMA INVOICE" => "PI".to_string(),
        "PACKING LIST" => "PL".to_string(),
        "BILL OF LADING" => "BL".to_string(),
        "HOUSE BILL OF LADING" => "HBL".to_string(),
        "SEA WAYBILL" => "SWB".to_string(),
        "AIR WAYBILL" => "AWB".to_string(),
        "SHIPPING ADVICE" => "SA".to_string(),
        "DELIVERY ORDER" => "DO".to_string(),
        "ARRIVAL NOTICE" => "AN".to_string(),
        "BOOKING CONFIRMATION" => "BC".to_string(),
        "BOOKING NOTE" => "BK".to_string(),
        "SHIPPING REQUEST" => "SR".to_string(),
        "FREIGHT INVOICE" => "FI".to_string(),
        "FORWARDER CERTIFICATE OF RECEIPT" => "FCR".to_string(),
        "PROOF OF DELIVERY" => "POD".to_string(),
        "CARGO MANIFEST" => "CM".to_string(),
        "WAREHOUSE RECEIPT" => "WR".to_string(),
        "EXPORT DECLARATION" => "ED".to_string(),
        "IMPORT DECLARATION" => "ID".to_string(),
        "CUSTOMS INVOICE" => "CINV".to_string(),
        "CERTIFICATE OF ORIGIN" => "CO".to_string(),
        "CUSTOMS CLEARANCE CERTIFICATE" => "CCC".to_string(),
        "CERTIFICATE OF NON-MANIPULATION" => "CNM".to_string(),
        "CONSIGNMENT SUMMARY INVOICE" => "CSI".to_string(),
        "INSPECTION CERTIFICATE" => "IC".to_string(),
        "WEIGHT CERTIFICATE" => "WC".to_string(),
        "CERTIFICATE OF ANALYSIS" => "CA".to_string(),
        "PHYTOSANITARY CERTIFICATE" => "PHYTO".to_string(),
        "HEALTH CERTIFICATE" => "HC".to_string(),
        "BENEFICIARY CERTIFICATE" => "BEN_CERT".to_string(),
        "FUMIGATION CERTIFICATE" => "FC".to_string(),
        "CARGO DAMAGE SURVEY REPORT" => "CDR".to_string(),
        "DANGEROUS GOODS DECLARATION" => "DGD".to_string(),
        "MATERIAL SAFETY DATA SHEET" => "MSDS".to_string(),
        "POWER OF ATTORNEY" => "POA".to_string(),
        "BUSINESS LICENSE" => "BIZ_LIC".to_string(),
        "INSURANCE POLICY" => "INS".to_string(),
        "INSURANCE CLAIM FORM" => "ICF".to_string(),
        "PURCHASE ORDER" => "PO".to_string(),
        "SALES CONTRACT" => "SC".to_string(),
        "LETTER OF CREDIT" => "LC".to_string(),
        "LOCAL LETTER OF CREDIT" => "LLC".to_string(),
        "CONFIRMATION OF PURCHASE" => "CP".to_string(),
        "BILL OF EXCHANGE" => "BE".to_string(),
        "TRUST RECEIPT" => "TR".to_string(),
        "LETTER OF GUARANTEE" => "LG".to_string(),
        "EXPORT LICENSE" => "EL".to_string(),
        "STATEMENT OF ACCOUNT" => "SOA".to_string(),
        "DEBIT NOTE" => "DN".to_string(),
        "CREDIT NOTE" => "CN".to_string(),
        "TAX INVOICE" => "TI".to_string(),
        // 코드가 이미 코드인 경우 대문자화하여 그대로 반환
        "CI" | "PI" | "SC" | "LC" | "LLC" | "CP" | "BE" | "TR" | "LG" | "EL"
        | "PL" | "BL" | "HBL" | "SWB" | "AWB" | "SA" | "DO" | "AN"
        | "BC" | "BK" | "SR" | "FCR" | "POD" | "CM" | "FI" | "WR"
        | "ED" | "ID" | "CINV" | "CO" | "CCC" | "CNM" | "CSI"
        | "IC" | "WC" | "CA" | "COA" | "PHYTO" | "PC" | "HC" | "BEN_CERT" | "FC" | "CDR"
        | "DGD" | "MSDS" | "POA" | "BIZ_LIC" | "INS" | "IP" | "ICF"
        | "SOA" | "DN" | "CN" | "TI" => doc_type.to_uppercase(),
        // 🌟 [TITLE REVERSE LOOKUP] 위 match 에 없는 전문은 TRADE_DOC_TITLES 로 역조회합니다.
        //
        //  ── 왜 필요한가 ──
        //   같은 '코드 ↔ 전문' 사전이 이 함수와 TRADE_DOC_TITLES 두 벌로 존재해
        //   실제로 어긋나 있었습니다. 아래는 TRADE_DOC_TITLES 에는 있는데
        //   위 match 에는 없어 코드로 접히지 않던 전문입니다.
        //     "fumigation certificate"   → FC
        //     "purchase confirmation"    → CP   (match 는 "CONFIRMATION OF PURCHASE" 만 보유)
        //     "consignment summary invoice" 표기 불일치 계열
        //   코드로 접히지 않으면 entity_index / entity_bcc 가 전문으로 만들어져
        //   목록 필터와 릴레이가 통째로 어긋납니다.
        //   TITLE GATE 가 쓰는 사전과 저장이 쓰는 사전은 반드시 같아야 합니다.
        _ => {
            let upper = doc_type.to_uppercase();
            let norm = |s: &str| -> String {
                s.chars()
                    .map(|c| if c.is_alphanumeric() { c.to_ascii_uppercase() } else { ' ' })
                    .collect::<String>()
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            let key = norm(&upper);
            for (code, title) in TRADE_DOC_TITLES.iter() {
                if norm(title) == key {
                    return code.to_string();
                }
            }
            upper
        }
    }
}

/// 🌟 [ALL REFERENCE FIELDS] 무역 문서가 가질 수 있는 모든 참조 축 목록입니다.
///  STEP C 정규화(FLATTEN)와 검색 조건 화이트리스트가 같은 목록을 공유해야
///  저장(정방향)과 조회(역방향)가 같은 이름 공간에서 만납니다.
pub const TRADE_REFERENCE_FIELDS: [&str; 53] = [
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
    // 🌟 [SR / WR / BE 축] trade_reference_field_of 에 새로 추가한 3축입니다.
    //    이 배열과 그 함수의 반환값 집합은 반드시 같아야 합니다.
    //    (한쪽만 늘리면 저장은 되는데 조회 조건에는 안 잡히는 비대칭이 생깁니다)
    "reference_sr", "reference_warehouse_receipt", "reference_bill_of_exchange",
    // 🌟 [MASTER B/L] House → Master 방향은 reference_bl 로는 표현할 수 없습니다.
    //    HBL 문서가 자기 상위 M B/L 을 가리킬 때 쓰는 전용 축입니다.
    "reference_master_bl",
    // 🌟 [GENERIC] bias.json 의 trade_schema.base.header 에 reference_number 가
    //    이미 존재하는데 이 배열에 없어 조건 순회에서 빠져 있었습니다.
    "reference_number",
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
/// 🌟 [v3] 10갈래.
///
///  ── 왜 3개를 더하는가 ──
///   customs / inspection / settlement 세 축의 질의가 기존 7갈래 어디에도
///   자연스럽게 들어가지 않았습니다.
///     "관세 얼마 나왔어?"      → terms 의 amount 로 잘못 떨어짐
///     "검사 통과한 서류 찾아줘" → identity 의 status 로 잘못 떨어짐
///     "미결제 잔액 있는 거래"   → terms 의 amount 로 잘못 떨어짐
///   Depth 1 이 틀리면 Depth 2 는 그 카테고리 안에서만 고르므로 복구가 불가능합니다.
pub const TRADE_CONDITION_CATEGORIES: [(&str, &str); 10] = [
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
    ("customs",
     "customs declaration number, export declaration, import declaration, declaration date, clearance date, customs office code, customs clearance status, entry type, released by customs, duty rate, tariff rate, duty amount, dutiable value, tax base, customs value, entered value, personal customs clearance code, bonded warehouse, customs broker, port code"),
    ("inspection",
     "inspection certificate, inspection date, place of inspection, inspection result, pass or fail, certificate number, certified by, laboratory analysis, test result, specification value, treatment date, chemical used, dosage, exposure period, fumigation, heat treatment, cold treatment, ISPM 15 mark, weighing date, verified gross mass, survey report, damage findings, surveyor conclusion, phytosanitary, health certificate"),
    ("settlement",
     "statement of account, account ledger, transaction date, debit amount, credit amount, running balance, outstanding balance, ending balance, debit note, credit note, tax invoice, VAT amount, supply amount, reason for debit, reason for credit, payment status, unpaid, settled, due date, overdue, charge code, freight charge breakdown, terminal handling charge, documentation fee"),
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
        "customs" => vec![
            ("declaration_number", "Export or import declaration number",
             "declaration number, customs declaration no, export declaration number, import declaration number, ED-2026-KR-77102, ID-2026-US-99120"),
            ("declaration_date",   "Date the declaration was filed",
             "declaration date, filed on, date of declaration, lodged on"),
            ("clearance_date",     "Date customs released the cargo",
             "clearance date, released on, date of release, customs release date"),
            ("customs_office_code","Customs office code",
             "customs office code, office code, customs house code, port code"),
            ("customs_status",     "Customs clearance status",
             "customs status, cleared, pending, under inspection, released, held"),
            ("entry_type",         "Entry type",
             "entry type, consumption entry, warehouse entry, informal entry"),
            ("duty_rate",          "Tariff rate applied",
             "duty rate, tariff rate, rate of duty, percent duty"),
            ("duty_amount",        "Total duty assessed",
             "duty amount, total duty, duty paid, customs duty"),
            ("dutiable_value",     "Dutiable value or tax base",
             "dutiable value, tax base, customs value, entered value, assessable value"),
            ("pccc_number",        "Personal customs clearance code",
             "personal customs clearance code, PCCC, P번호, personal clearance number"),
        ],
        "inspection" => vec![
            ("certificate_number", "Certificate number of this inspection or test document",
             "certificate number, cert no, certificate of analysis number, inspection certificate number, IC-2026-0825, COA-2026-0824"),
            ("inspection_date",    "Date of inspection or survey",
             "inspection date, date of inspection, surveyed on, examined on"),
            ("inspection_place",   "Place of inspection",
             "place of inspection, inspection site, location of survey"),
            ("inspection_result",  "Result of inspection",
             "inspection result, pass, fail, conforms, does not conform, satisfactory, overall result"),
            ("treatment_date",     "Date of fumigation or treatment",
             "treatment date, fumigation date, date of treatment, treated on"),
            ("treatment_chemical", "Chemical used in treatment",
             "chemical used, fumigant, methyl bromide, phosphine, active ingredient, concentration"),
            ("weighing_date",      "Date of weighing",
             "weighing date, weighed on, date of weighing"),
            ("ispm15_mark",        "ISPM 15 mark on wooden packaging",
             "ISPM 15, ISPM15 mark, heat treated stamp, HT mark, wood packaging mark"),
        ],
        "settlement" => vec![
            ("transaction_date",   "Date of a ledger transaction",
             "transaction date, posted on, entry date, ledger date"),
            ("debit",              "Debit amount on a ledger line",
             "debit, debit amount, charged, owed"),
            ("credit",             "Credit amount on a ledger line",
             "credit, credit amount, credited, paid"),
            ("balance",            "Running or ending balance",
             "balance, running balance, ending balance, outstanding balance, closing balance"),
            ("account_status",     "Account status",
             "account status, open, closed, settled, outstanding, overdue"),
            ("payment_status",     "Payment status",
             "payment status, paid, unpaid, partially paid, pending payment"),
            ("due_date",           "Payment due date",
             "due date, payable by, payment due, net 30 due"),
            ("vat_type",           "VAT type on a tax invoice",
             "VAT type, taxable, zero rated, exempt, 과세, 영세, 면세"),
            ("charge_amount",      "Amount of an individual charge line",
             "charge amount, line charge, THC, terminal handling charge, documentation fee, handling charge"),
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
        "reference_sr"                => "referenced shipping request number, against S/R, booking request reference, SR-2026-0820",
        "reference_warehouse_receipt" => "referenced warehouse receipt number, godown receipt reference, WR-2026-0830",
        "reference_bill_of_exchange"  => "referenced bill of exchange number, draft number, against draft, BE-2026-0905",
        // 🌟 House B/L 이 자기 상위 Master B/L 을 가리키는 전용 축입니다.
        //    reference_bl 로 겸용하면 '이 문서가 가리키는 B/L' 과
        //    '이 문서의 상위 B/L' 이 같은 필드에 섞여 그래프 방향이 무너집니다.
        "reference_master_bl"         => "referenced master bill of lading number, master B/L, MBL no, ocean carrier B/L covering this house B/L, MBL-55432219",
        "reference_number"            => "reference number, our reference, your reference, ref no, generic reference printed on this document",
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
        // ── 코드·식별자 : 완전일치 ──
        "doc_number" | "no" | "status" | "doc_type"
        | "container_number" | "seal_number" | "hs_code"
        | "incoterms" | "currency" | "freight_payment_term"
        | "declaration_number" | "certificate_number" | "policy_number"
        | "claim_number" | "customs_office_code" | "pccc_number"
        | "charge_code" | "un_number" | "cas_number"
        | "fta_agreement_code" | "eccn" | "swift_code" | "account_number" => "eq",

        // ── 날짜 : 기준일 이후 ──
        //  질의가 "8월 이후 통관된 건" 처럼 하한을 뜻하는 경우가 압도적입니다.
        "issue_date" | "expiry_date" | "etd" | "eta"
        | "departure_date" | "arrival_date"
        | "declaration_date" | "clearance_date" | "release_date"
        | "inspection_date" | "treatment_date" | "weighing_date"
        | "claim_date" | "effective_date" | "transaction_date"
        | "due_date" | "maturity_date" | "valid_until"
        | "latest_shipment_date" | "cargo_closing_date"
        | "expected_ship_date" | "expected_delivery_date"
        | "estimated_shipment_date" | "date_received" | "closure_date" => "gte",

        // ── 수치 : 완전일치 ──
        //  🌟 [주의] 여기 있는 축은 '값이 정확히 얼마' 라는 질의를 전제합니다.
        //     '얼마 이상' 은 질의 청크에 비교 표현이 붙어 있고,
        //     ai_utils::split_numeric_and_comparator 가 연산자를 별도로 확정하므로
        //     이 기본값이 그 판정을 덮어쓰지 않습니다.
        "amount" | "amount_subtotal" | "amount_tax"
        | "freight_amount" | "insurance_amount" | "local_charges"
        | "package_count" | "weight_gross" | "weight_net" | "volume"
        | "chargeable_weight" | "exchange_rate"
        | "duty_rate" | "duty_amount" | "dutiable_value"
        | "insured_amount" | "premium" | "claim_amount"
        | "debit" | "credit" | "balance" | "charge_amount"
        | "usance_tenor_days" | "flash_point"
        | "unit_price" | "total_price" | "quantity" => "eq",

        // ── 그 외 자유 서술 : 부분일치 ──
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
/// 🌟 [TRADE GROUPS v2] 7갈래.
///
///  ── settlement 를 왜 새로 두는가 ──
///   SOA(거래명세서) / DN(차변표) / CN(대변표) / TI(세금계산서) / FI(운임인보이스)는
///   '거래가 끝난 뒤의 회계 정산' 이라는 뚜렷한 성격을 갖습니다.
///   contract 에 억지로 넣으면 '계약 조건' 앵커와 '차변/대변' 앵커가 한 그룹에서
///   서로를 희석해 Depth 1 판정이 흔들립니다.
pub const TRADE_GROUPS: [(&str, &str); 7] = [
    ("contract",  "purchase order, proforma invoice, sales contract, letter of credit, documentary credit, local letter of credit, purchase confirmation, bill of exchange, trust receipt, letter of guarantee, payment terms, contract number, buyer seller agreement, tenor at sight, usance, drawer drawee payee, issuing bank, advising bank, beneficiary, applicant, order confirmation, quotation, export license"),
    ("shipping",  "commercial invoice, packing list, bill of lading, ocean bill of lading, house bill of lading, sea waybill, air waybill, shipping request, booking confirmation, shipping advice, delivery order, arrival notice, proof of delivery, warehouse receipt, forwarder certificate of receipt, cargo manifest, freight invoice, vessel voyage number, flight number, port of loading, port of discharge, place of receipt, place of delivery, container number, seal number, notify party, freight prepaid, freight collect, shipper and consignee, gross weight net weight measurement, carton quantity, marks and numbers, incoterms fob cif exw"),
    ("customs",   "export declaration, import declaration, customs invoice, consular invoice, certificate of origin, non manipulation certificate, customs clearance certificate, hs code, tariff classification, customs clearance, declaration number, customs value, duty and tax, chamber of commerce, country of origin, consular visa, legalization"),
    ("inspection","inspection certificate, weight certificate, certificate of analysis, phytosanitary certificate, fumigation certificate, health certificate, beneficiary certificate, cargo damage survey report, we hereby certify, test result, specification value, fumigation treatment, laboratory report, fit for human consumption, plant health, verified gross mass, surveyor findings"),
    ("legal",     "dangerous goods declaration, material safety data sheet, power of attorney, business license, insurance policy, insurance claim form, un number, proper shipping name, packing group, hazard class, policy number, insured amount, premium, coverage all risks, attorney in fact, business registration number, claim amount, cause of loss"),
    ("settlement","statement of account, debit note, credit note, tax invoice, freight invoice, account ledger, opening balance, ending balance, transaction date, debit credit column, reason for debit, reason for credit, VAT amount, supply amount, remittance instructions, outstanding balance, due date, aging report"),
    ("parcel",    "courier label, parcel waybill sticker, domestic courier service, home delivery parcel, door to door small package, delivery driver, barcode sticker label, parcel pickup, last mile delivery"),
];

/// Depth 2 : 그룹 소속 코드 목록.
/// 🌟 [TRADE GROUP CODES v2] 27종 → 55종.
///
///  ── 무엇이 문제였나 ──
///   related_trading() 은 HBL / FCR / POD / LG / TR / LLC / TI / CP / CM /
///   CCC / CNM / PC / COA / FI / CDR / ICF / SOA / EL / BE / SR / BK / WR /
///   CSI / SWB / IP / DN / CN / FC 를 참조 그래프에 넣어 두었지만,
///   이 목록에 없으면 STEP 2 의 Depth 2 후보에 오르지 못합니다.
///   (실측 로그의 VISION CODE CANDIDATES 가 정확히 27개였습니다)
///   그 결과 House B/L 은 BL 로, 보험증권은 INS 로 오분류되고
///   trade_relay_rules 가 빈 vec 을 돌려줘 연결 그래프가 통째로 죽습니다.
pub const TRADE_GROUP_CODES: [(&str, &[&str]); 7] = [
    ("contract",   &["PO", "PI", "SC", "LC", "LLC", "CP", "BE", "TR", "LG", "EL"]),
    ("shipping",   &["CI", "PL", "BL", "HBL", "SWB", "AWB", "SA", "DO", "AN",
                     "BC", "BK", "SR", "FCR", "POD", "CM", "FI", "WR"]),
    ("customs",    &["ED", "ID", "CINV", "CO", "CCC", "CNM", "CSI"]),
    ("inspection", &["IC", "WC", "CA", "COA", "PHYTO", "PC", "HC",
                     "BEN_CERT", "FC", "CDR"]),
    ("legal",      &["DGD", "MSDS", "POA", "BIZ_LIC", "INS", "IP", "ICF"]),
    ("settlement", &["SOA", "DN", "CN", "TI"]),
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

/// 🌟 [TABLE STRUCTURE ANCHOR] 명세 표(line items / containers) 전용 시각 앵커.
///
///  ── 왜 필요한가 ──
///   items 카테고리의 스키마 필드는 `hs_code` 하나뿐입니다.
///   실측에서 items 히트맵의 최고 점수는 +1.3126 으로 7개 카테고리 중 최하위였고,
///   상품 표(y 550~634)를 전혀 잡지 못한 채 합계 블록을 크롭했습니다.
///   표는 '컬럼 헤더 나열 + 반복 행' 이라는 강한 시각 구조를 갖고 있으므로
///   그 구조 자체를 앵커로 세워야 위치가 잡힙니다.
///
///  ── 어휘가 아니라 구조인가 ──
///   여기 적힌 문구는 전 세계 무역 서식 표의 컬럼 헤더 표준 명칭입니다.
///   bias.json 의 필드 semantic 과 같은 계보이며, 특정 서식에 종속되지 않습니다.
pub const TRADE_TABLE_STRUCTURE_ANCHOR: &str =
    "description of goods, description of merchandise, commodity description, item description, \
     line item table, itemized list, goods table, product table, \
     quantity column, unit of measure, unit price column, unit value, total value column, \
     amount column, qty, pcs, unit weight, net weight column, \
     country of manufacture, country of origin column, hs code column, tariff code column, \
     repeating table rows, tabular data grid, column headers row, itemised breakdown";

/// 컨테이너 명세 표 전용 앵커.
pub const TRADE_CONTAINER_TABLE_ANCHOR: &str =
    "container list, container number column, seal number column, \
     container type size, number of packages column, gross weight column, measurement column, \
     container and seal table, equipment list";

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

        // 🌟 [NEW 28] TRADE_GROUP_CODES v2 로 후보에 오르게 된 서식들입니다.
        //    구버전은 이 코드들이 후보 목록에 없어 판정 자체가 불가능했고,
        //    설령 도달해도 이 함수가 기본값 "trade document" 를 돌려줘
        //    Depth 2 채점에서 모든 코드가 동점이 되었습니다.
        //    앵커 문구는 각 서식의 표준 표제·필수 기재사항입니다.

        // ── 계약 · 결제 ──
        "LLC"      => "local letter of credit, domestic letter of credit, internal L/C, applicant exporter, beneficiary supplier, local L/C amount in won",
        "CP"       => "confirmation of purchase, purchase confirmation certificate, supplier manufacturer, purchaser exporter, issuing agency, confirmation number",
        "BE"       => "bill of exchange, draft, drawer, drawee, payee, pay against this first bill of exchange, drawn under letter of credit, at sight of this draft",
        "TR"       => "trust receipt, entrustor bank, trustee importer, ownership clause, goods held in trust, maturity date, purpose of release",
        "LG"       => "letter of guarantee, shipping guarantee, we hereby guarantee, indemnify the carrier, original bill of lading not yet received, bank endorsement",
        "EL"       => "export license, export permit, licensed items, authorized value, license condition, ECCN, export control classification, issuing authority",

        // ── 선적 · 운송 ──
        "HBL"      => "house bill of lading, HBL, freight forwarder issued, master bill of lading reference, NVOCC, house B/L number",
        "SWB"      => "sea waybill, non negotiable waybill, express release, surrender bill, no original required, special instructions",
        "SR"       => "shipping request, booking request, S/R, request for space, forwarder carrier, shipping details requested",
        "BK"       => "booking confirmation, booking note, space confirmed, container allocation, cargo closing date, cut off time, booking number",
        "WR"       => "warehouse receipt, godown receipt, warehouse operator, depositor exporter, storage term, warehouse location, packages received, date received",
        "FCR"      => "forwarder certificate of receipt, FCR, forwarders cargo receipt, issuing forwarder, undertaking statement, received the goods described",
        "POD"      => "proof of delivery, delivery receipt, received by, delivery date, delivery status, carrier trucker, consignee recipient, packages delivered",
        "CM"       => "cargo manifest, ships manifest, shipping agent, flag state, total bills of lading, total containers, submission date, consignment summary",
        "FI"       => "freight invoice, carrier invoice, charge code, rate, total charges, due date, bill to shipper, terminal handling charge, documentation fee",

        // ── 통관 · 신고 ──
        "CCC"      => "customs clearance certificate, entry summary, entry type, release date, customs status, entered value, duty paid amount, customs authority",
        "CNM"      => "certificate of non manipulation, non manipulation certificate, transshipment customs authority, port of transshipment, arrival vessel, departure vessel, certification statement",
        "CSI"      => "consular invoice, consular visa, legalized by, notary agency, legalization fee, consulate stamp, visa number",

        // ── 검사 · 증명 ──
        "COA"      => "certificate of analysis, laboratory analysis report, test results table, specification and result, overall result pass, certified by, issuing laboratory",
        "PC"       => "phytosanitary certificate, plant quarantine certificate, botanical name, declared name of product, place of origin, declared port of entry, disinfestation treatment",
        "FC"       => "fumigation certificate, treatment certificate, chemical used, dosage, exposure period, minimum temperature, place of treatment, ISPM 15 mark",
        "CDR"      => "cargo damage report, survey report, surveying agency, findings and damage, nature of damage, cause of damage, estimated loss amount, surveyor conclusion, sound packages damaged packages",

        // ── 보험 · 청구 ──
        "IP"       => "insurance policy, marine insurance certificate, insurer, insured, sum insured, valuation basis, coverage conditions, claims payable at, effective date",
        "ICF"      => "insurance claim form, claim application, claimant, claim date, cause of loss, place of loss, damaged quantity, enclosed documents, reimbursement bank account",

        // ── 정산 ──
        "SOA"      => "statement of account, account ledger, opening balance, transaction date, debit column, credit column, ending balance, account status, closure date",
        "DN"       => "debit note, debit memo, reason for debit, charges, due date, payment instructions, remittance bank, total debit amount",
        "CN"       => "credit note, credit memo, reason for credit, adjustments, settlement instructions, total credit amount",
        "TI"       => "tax invoice, VAT invoice, supply amount, tax amount, VAT type, representative, business registration number, grand total in won",

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
    // ── ① 참조 축은 전부 header ──
    //    (구버전은 match 에도 "reference_number" 를 적어 두었지만
    //     이 return 이 먼저 실행되어 그 arm 은 도달 불가능한 죽은 코드였습니다)
    if field.starts_with("reference_") {
        return "header";
    }

    // ── ② 명시 매핑 ──
    match field {
        // ── header ──
        "doc_type" | "doc_number" | "issue_date" | "expiry_date"
            | "no" | "status" | "submission_date" => "header",

        // ── parties (스칼라 5축) ──
        //  🌟 [왜 배열로 바꾸지 않았는가]
        //   이 5개는 trade_condition_fields / bias.json ko.shipping_doc /
        //   search_bridge.multilingual_value_anchor / Dexie 인덱스 네 곳이
        //   이미 이름으로 참조합니다. 배열로 바꾸면 그 배선이 전부 끊깁니다.
        //   기존 5축은 스칼라로 두고, 나머지 역할은 other_parties 가 받습니다.
        "sender_name" | "sender_address" | "recipient_name"
            | "recipient_address" | "notify_party_name" => "parties",

        // ── other_parties (배열) ──
        //  🌟 45종 예시에서 확인된 역할은 42개입니다.
        //     (drawer / drawee / payee / carrier / insurer / issuing_bank /
        //      advising_bank / customs_broker / warehouse_operator / claimant …)
        //     개별 필드로 두면 issuing_bank·advising_bank·entrustor_bank 세 축이
        //     같은 '은행명' 영역을 놓고 경쟁해 근거가 3분할되고 전부 굶습니다.
        //     role 을 원소 필드로 두면 한 영역에서 여러 역할을 순서대로 읽어내고,
        //     우리가 예상하지 못한 역할까지 스키마 변경 없이 수용합니다.
        "party_role" | "party_name" | "party_address"
            | "party_contact" => "other_parties",

        // ── logistics ──
        //  🌟 place_receipt / place_delivery 를 되살렸습니다.
        //     Part 11 에서 "trade_schema 에 없다" 는 이유로 제거를 제안했는데,
        //     base v2 에 정식으로 추가했으므로 다시 축이 됩니다.
        //     B/L 과 FCR 의 필수 기재사항이라 빠뜨릴 수 없습니다.
        "vessel" | "voyage_number" | "flight_number" | "pol" | "pod"
            | "place_receipt" | "place_delivery" | "etd" | "eta"
            | "departure_date" | "arrival_date"
            | "transport_mode" | "means_of_conveyance"
            | "flag_state" | "terminal_of_discharge"
            | "port_of_transshipment" | "cargo_closing_date" => "logistics",

        // ── conditions ──
        "incoterms" | "payment_terms" | "freight_payment_term"
            | "partial_shipments" | "transshipment_allowed" | "latest_shipment_date"
            | "governing_law" | "arbitration" | "non_negotiable"
            | "temperature_control" | "storage_term" | "release_instruction"
            | "special_instructions" => "conditions",

        // ── financials ──
        "currency" | "amount" | "amount_subtotal" | "amount_tax"
            | "freight_amount" | "insurance_amount" | "local_charges"
            | "exchange_rate" | "bank_charges" | "usance_tenor_days" | "tenor"
            | "maturity_date" | "due_date" | "valid_until"
            | "remittance_reference" | "swift_code" | "account_number" => "financials",

        // ── cargo ──
        //  🌟 container_number / seal_number 를 여기서 뺐습니다.
        //     구버전이 이 둘을 cargo 로 보내는 바람에 containers 카테고리에
        //     배정되는 필드가 0개가 되어, 히트맵이 아예 생성되지 않았습니다.
        //     (실측 로그가 8개가 아니라 7개 카테고리만 출력한 직접 원인)
        "package_count" | "package_unit" | "weight_gross" | "weight_net"
            | "volume" | "marks_numbers" | "chargeable_weight"
            | "container_tare_weight" => "cargo",

        // ── items (배열) ──
        //  🌟 구버전은 hs_code 하나뿐이었습니다.
        //     bias.json 의 trade_schema.base.items 에는 6개가 멀쩡히 있는데
        //     이 함수가 5개를 버려서 items 히트맵이 단일 앵커로 붕괴했고,
        //     7개 카테고리 중 최하위(+1.3126)로 밀려 상품 표를 놓쳤습니다.
        //  🌟 item_ 접두 4축은 cargo 의 동명 필드(weight_net / package_count)와
        //     이름이 겹치지 않게 하려는 것입니다. 이 함수는 필드명 문자열만 받으므로
        //     겹치면 소속을 판별할 방법이 없어 한쪽이 앵커를 잃습니다.
        "description" | "item_code" | "quantity" | "unit" | "hs_code"
            | "country_of_manufacture" | "unit_price" | "total_price"
            | "item_net_weight" | "item_gross_weight"
            | "item_package_count" | "item_package_type" => "items",

        // ── containers (배열) ──
        "container_number" | "seal_number" | "type_size"
            | "container_package_count" | "container_gross_weight"
            | "container_measurement" => "containers",

        // ── customs (신규) ──
        "declaration_number" | "declaration_date" | "clearance_date"
            | "customs_office_code" | "duty_rate" | "duty_amount"
            | "dutiable_value" | "entry_type" | "customs_status"
            | "pccc_number" | "port_code" | "port_name"
            | "export_permission" => "customs",

        // ── inspection (신규) ──
        "inspection_date" | "inspection_place" | "inspection_body"
            | "inspection_result" | "inspection_scope" | "inspection_status"
            | "treatment_date" | "treatment_chemical" | "treatment_dosage"
            | "treatment_duration" | "treatment_temperature" | "treatment_type"
            | "certificate_number" | "test_summary" | "ispm15_mark"
            | "health_status" | "beneficiary_statement"
            | "weighing_date" | "weighing_location" => "inspection",

        // ── insurance (신규) ──
        "policy_number" | "insured_amount" | "premium" | "premium_amount"
            | "coverage_condition" | "coverage_type" | "valuation_basis"
            | "claims_payable_at" | "effective_date"
            | "claim_number" | "claim_date" | "cause_of_loss" | "place_of_loss" => "insurance",

        // ── hazmat (신규) ──
        "un_number" | "hazard_class" | "packing_group" | "proper_shipping_name"
            | "flash_point" | "cas_number" | "emergency_contact" => "hazmat",

        // ── origin (신규) ──
        "origin_criterion" | "fta_agreement_code" | "preference_indicator"
            | "origin_certificate_type" | "place_of_origin"
            | "botanical_name" | "country_of_origin" => "origin",

        // ── compliance (신규) ──
        "carbon_footprint" | "traceability_code" | "eccn"
            | "batch_lot_number" | "halal_kosher_cert_no"
            | "cites_permit_number" | "fda_ema_approval_no" => "compliance",

        // ── charges (배열, 신규) ──
        "charge_code" | "charge_description" | "charge_rate"
            | "charge_amount" => "charges",

        // ── settlement (신규) ──
        "transaction_date" | "debit" | "credit" | "balance"
            | "account_status" | "closure_date"
            | "reason_for_credit" | "reason_for_debit"
            | "vat_type" | "settlement_instructions"
            | "payment_instructions" | "payment_status" => "settlement",

        _ => trade_field_category_by_rule(field),
    }
}

// 🌟 [TRADE DOC TITLES] vision_encoder.rs 에 하드코딩되어 있던 서식 전문 사전을
//    logic.rs 로 이관합니다. 텍스트 트랙(scheduler STEP A)과 비전 트랙이
//    같은 사전을 쓰면 서식이 늘어도 수정 지점이 한 곳으로 고정됩니다.
//    전문(full name)을 키로 쓰는 이유: 'CI' 같은 접두어는 다른 서식에도
//    인쇄되지만(ED 의 reference_invoice 등), 전문은 헤더로 인쇄되는 서식 본인뿐입니다.
pub const TRADE_DOC_TITLES: &[(&str, &str)] = &[
    ("CI", "commercial invoice"),
    ("PI", "proforma invoice"),
    ("CINV", "customs invoice"),
    ("CSI", "consular invoice"),
    ("TI", "tax invoice"),
    ("FI", "freight invoice"),
    ("PL", "packing list"),
    ("BL", "bill of lading"),
    ("HBL", "house bill of lading"),
    ("SWB", "sea waybill"),
    ("AWB", "air waybill"),
    ("SA", "shipping advice"),
    ("DO", "delivery order"),
    ("AN", "arrival notice"),
    ("BC", "booking confirmation"),
    ("BK", "booking confirmation"),
    ("SR", "shipping request"),
    ("FCR", "forwarder certificate of receipt"),
    ("POD", "proof of delivery"),
    ("CM", "cargo manifest"),
    ("WR", "warehouse receipt"),
    ("ED", "export declaration"),
    ("ID", "import declaration"),
    ("CO", "certificate of origin"),
    ("CNM", "certificate of non manipulation"),
    ("CCC", "customs clearance certificate"),
    ("EL", "export license"),
    ("IC", "inspection certificate"),
    ("COA", "certificate of analysis"),
    ("CA", "certificate of analysis"),
    ("WC", "weight certificate"),
    ("PHYTO", "phytosanitary certificate"),
    ("PC", "phytosanitary certificate"),
    ("FC", "fumigation certificate"),
    ("HC", "health certificate"),
    ("BEN_CERT", "beneficiary certificate"),
    ("CDR", "cargo damage survey report"),
    ("DGD", "dangerous goods declaration"),
    ("MSDS", "material safety data sheet"),
    ("POA", "power of attorney"),
    ("BIZ_LIC", "business license"),
    ("INS", "insurance policy"),
    ("IP", "insurance policy"),
    ("ICF", "insurance claim form"),
    ("SOA", "statement of account"),
    ("DN", "debit note"),
    ("CN", "credit note"),
    ("PO", "purchase order"),
    ("SC", "sales contract"),
    ("LC", "letter of credit"),
    ("LLC", "local letter of credit"),
    ("CP", "purchase confirmation"),
    ("BE", "bill of exchange"),
    ("TR", "trust receipt"),
    ("LG", "letter of guarantee"),
    // 🌟 [COMMERCE/PARCEL AXIS] mode 감지용 추가 축.
    //    택배 라벨은 무역 서식과 달리 전문이 '라벨' 문구로 인쇄됩니다.
    //    동명 마진 0 자가거부 규칙 때문에 TRACKING 은 1개 항목만 둡니다.
    ("TRACKING", "tracking label shipping label parcel waybill"),
];

/// 🌟 [RULE FALLBACK] 명시 매핑에 없는 필드를 부분일치 규칙으로 라우팅합니다.
///
///  ── 왜 필요한가 ──
///   구버전은 `_ => ""` 였습니다. 그래서 bias.json 의 trade_schema 에 필드를
///   추가해도 이 함수를 함께 고치지 않으면 그 필드는 카테고리를 못 받고,
///   히트맵에서 통째로 사라졌습니다.
///   (실측: items 6필드 중 5개가 정확히 이 경로로 소멸)
///   규칙 폴백이 있으면 bias.json 만 고쳐도 새 필드가 자동으로 라우팅됩니다.
///
///  ── 순서가 의미를 갖습니다 ──
///   'duty_rate' 는 'rate'(financials) 와 'duty_'(customs) 양쪽에 걸립니다.
///   좁은 개념(customs / insurance / hazmat)을 먼저 검사해야 올바른 축으로 갑니다.
fn trade_field_category_by_rule(field: &str) -> &'static str {
    let f = field;

    // ── 좁은 축부터 ──
    if f.contains("hazard") || f.contains("packing_group") || f.contains("cas_")
        || f.contains("un_number") || f.contains("flash_point") {
        return "hazmat";
    }
    if f.contains("customs") || f.contains("duty") || f.contains("dutiable")
        || f.contains("declaration") || f.contains("clearance") || f.contains("tariff") {
        return "customs";
    }
    if f.contains("insur") || f.contains("polic") || f.contains("premium")
        || f.contains("claim") || f.contains("coverage") {
        return "insurance";
    }
    if f.contains("inspect") || f.contains("treatment") || f.contains("certificate")
        || f.contains("survey") || f.contains("weighing") || f.contains("test_") {
        return "inspection";
    }
    if f.contains("origin") || f.contains("fta_") || f.contains("preference") {
        return "origin";
    }

    // ── 넓은 축 ──
    if f.contains("port") || f.contains("vessel") || f.contains("flight")
        || f.contains("voyage") || f.contains("terminal") || f.contains("conveyance") {
        return "logistics";
    }
    if f.contains("weight") || f.contains("packages") || f.contains("pieces")
        || f.contains("volume") || f.contains("measurement") || f.contains("marks") {
        return "cargo";
    }
    if f.contains("amount") || f.contains("price") || f.contains("charge")
        || f.contains("currency") || f.contains("rate") || f.contains("bank")
        || f.contains("balance") {
        return "financials";
    }
    if f.ends_with("_name") || f.ends_with("_address") || f.contains("party")
        || f.contains("consignee") || f.contains("shipper") || f.contains("applicant") {
        return "parties";
    }
    if f.contains("date") || f.contains("_no") || f.contains("number") {
        return "header";
    }

    ""
}

/// 🌟 이 카테고리가 배열(반복 행) 스키마인지 답합니다.
///
///  ── 왜 이 함수가 logic.rs 에 있는가 ──
///   merge_extracted(model.rs) 의 배열 승격, get_trade_category_schema(prompts.rs) 의
///   `[ { ... } ]` 렌더링, plan_crops(vision_crop.rs) 의 TABLE UNION 이
///   전부 같은 판정을 필요로 합니다. 세 곳이 각자 문자열을 비교하면
///   카테고리 하나 추가할 때마다 셋이 어긋납니다.
pub fn is_trade_array_category(category: &str) -> bool {
    // 🌟 parties 는 스칼라 5축이므로 배열이 아닙니다.
    //    반복되는 당사자는 other_parties 가 받습니다. (22-7 주석 참조)
    matches!(
        category,
        "items" | "containers" | "other_parties" | "charges"
            | "test_results" | "findings_and_damage" | "account_ledger"
    )
}

/// 🌟 [EXTRACTION CATEGORIES] 비전 크롭 + LLM 추출이 순회하는 카테고리 목록.
///
///  parsing.rs 의 get_trade_category_schema 가 소비하는 8개와 동일합니다.
///  items / containers 는 배열 스키마이며, 히트맵으로 위치를 찾은 뒤
///  그 영역 전체를 크롭해 표를 통째로 읽힙니다.
/// 🌟 [EXTRACTION CATEGORIES v2]
///
///  ── 왜 8개에서 16개로 늘리는가 ──
///   45종 예시 대조에서 기존 8개 카테고리에 담을 수 없는 필드가
///   customs(통관) / inspection(검사) / insurance(보험) / settlement(정산) /
///   hazmat(위험물) / origin(원산지) / compliance(규제) / charges(요금명세)
///   여덟 갈래로 나타났습니다.
///   기존에는 이 필드들이 trade_field_category 에서 "" 를 받아 통째로 사라졌습니다.
///
///  ── 카테고리가 늘어도 호출 횟수는 서식마다 다릅니다 ──
///   히트맵은 '그 서식에 실제로 로드된 필드가 있는 카테고리' 에만 생성됩니다.
///   bias_schema 의 조건부 로드(base + overlay[doc_type])와 맞물려
///     · CI 인보이스 → header/parties/logistics/conditions/financials/cargo/items/containers = 8
///     · DGD        → 위 8 + hazmat = 9
///     · COA        → 위 8 + inspection = 9
///     · SOA        → header/parties/financials/settlement/charges = 5
///   처럼 자동으로 좁혀집니다. 16개가 항상 전부 도는 것이 아닙니다.
///
///  ⚠️ 배열 여부는 is_trade_array_category() 가 단독으로 답합니다.
///     이 배열의 순서는 크롭 추출 순서가 되므로,
///     식별자(header)를 먼저 확정해 ALREADY CLAIMED 로 뒤 카테고리를 보호합니다.
pub const TRADE_EXTRACTION_CATEGORIES: [&str; 17] = [
    // ── base (전 서식 공통) ──
    "header", "parties", "other_parties", "logistics", "conditions",
    "financials", "cargo", "items", "containers",
    // ── overlay (서식별 조건부) ──
    "customs", "inspection", "insurance", "settlement",
    "hazmat", "origin", "compliance", "charges",
];
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
                    .map(|c| if c.is_alphanumeric() { c } else { ' ' })
                    .collect::<String>()
                    .to_uppercase()
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            let key = norm(&upper);
            for (code, title) in trade_title_pairs().iter() {
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
    // 🌟 [ALIAS COLLAPSE GUARD]
    //
    //  ── 어떤 쌍이 걸리는가 ──
    //   INS ↔ IP      → 둘 다 reference_policy
    //   CA  ↔ COA     → 둘 다 reference_analysis
    //   PHYTO ↔ PC    → 둘 다 reference_phyto
    //   BC  ↔ BK      → 둘 다 reference_booking
    //   related_trading("INS") 이 IP 를 포함하므로 이 경로는 실제로 실행됩니다.
    //
    //  ── 왜 위험한가 ──
    //   mine == foreign 이면 상대 문서의 그 필드에 내 doc_number 를 덮어씁니다.
    //   상대가 원래 그 필드로 '나' 를 가리키고 있었다면 값이 자기 자신을 향하게 되고,
    //   다음 스캔에서 RELAY SELF-LOOP 판정으로 관계가 통째로 끊깁니다.
    //
    //  ── 왜 그냥 끊는가 ──
    //   두 코드는 같은 서식의 다른 표기입니다(보험증권 / 분석성적서 / 식물검역 / 부킹).
    //   별개의 두 문서가 아니므로 릴레이를 성립시킬 이유 자체가 없습니다.
    if mine == foreign { return None; }
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
pub const TRADE_CONDITION_CATEGORIES: [(&str, &str); 11] = [
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
    ("items",
     "description of goods, goods description, commodity, product name, article, item, line item, country of manufacture, country of origin, made in, manufactured in, unit price, price per unit, unit value, quantity, pieces, unit of measure, line total, total price per line, item code, model number, net weight per unit, gross weight per unit"),
];

pub const TRADE_CONDITION_CATEGORIES_ML: [(&str, &str); 11] = [
    ("identity",
     "Dokumentart, Dokumentnummer, Rechnungsnummer, Bestellnummer, Konnossementnummer, Vertragsnummer, Sendungsverfolgungsnummer, Dokumentstatus, Ausstellungsdatum, Ablaufdatum, \
      type de document, numéro de document, numéro de facture, numéro de commande, numéro de connaissement, numéro de contrat, numéro de suivi, statut du document, date d'émission, date d'expiration, \
      tipo de documento, número de documento, número de factura, número de pedido, número de conocimiento de embarque, número de contrato, número de seguimiento, estado del documento, fecha de emisión, fecha de vencimiento, \
      tipo di documento, numero documento, numero fattura, numero ordine, numero polizza di carico, numero contratto, numero di tracciamento, stato del documento, data di emissione, data di scadenza, \
      número do documento, número da fatura, número do pedido, número do conhecimento de embarque, número do contrato, número de rastreamento, situação do documento, data de emissão, data de validade, \
      documenttype, documentnummer, factuurnummer, ordernummer, cognossementnummer, contractnummer, trackingnummer, documentstatus, uitgiftedatum, vervaldatum, \
      typ dokumentu, číslo dokumentu, číslo faktury, číslo objednávky, číslo konosamentu, číslo smlouvy, sledovací číslo, stav dokumentu, datum vystavení, datum platnosti, \
      نوع المستند, رقم المستند, رقم الفاتورة, رقم أمر الشراء, رقم بوليصة الشحن, رقم العقد, رقم التتبع, حالة المستند, تاريخ الإصدار, تاريخ الانتهاء, \
      書類種別, 書類番号, 請求書番号, 注文番号, 船荷証券番号, 契約番号, 追跡番号, 書類の状態, 発行日, 有効期限, \
      单据类型, 单据编号, 发票号, 订单号, 提单号, 合同号, 追踪号, 单据状态, 签发日期, 有效期, \
      서류 종류, 문서번호, 송장번호, 주문번호, 선하증권번호, 계약번호, 추적번호, 문서 상태, 발행일, 만료일"),
    ("transport",
     "Schiffsname, Flugnummer, Reisenummer, Ladehafen, Löschhafen, Bestimmungshafen, Übernahmeort, Lieferort, voraussichtliche Abfahrt, voraussichtliche Ankunft, Transportart, Seefracht, Luftfracht, \
      nom du navire, numéro de vol, numéro de voyage, port de chargement, port de déchargement, port de destination, lieu de réception, lieu de livraison, date de départ prévue, date d'arrivée prévue, mode de transport, fret maritime, fret aérien, \
      nombre del buque, número de vuelo, número de viaje, puerto de carga, puerto de descarga, puerto de destino, lugar de recepción, lugar de entrega, fecha estimada de salida, fecha estimada de llegada, modo de transporte, flete marítimo, flete aéreo, \
      nome della nave, numero di volo, numero di viaggio, porto di carico, porto di scarico, porto di destinazione, luogo di presa in carico, luogo di consegna, data prevista di partenza, data prevista di arrivo, modalità di trasporto, trasporto marittimo, trasporto aereo, \
      nome do navio, número do voo, número da viagem, porto de embarque, porto de descarga, porto de destino, local de recebimento, local de entrega, data prevista de partida, data prevista de chegada, modo de transporte, frete marítimo, frete aéreo, \
      scheepsnaam, vluchtnummer, reisnummer, laadhaven, loshaven, bestemmingshaven, plaats van ontvangst, plaats van levering, verwachte vertrekdatum, verwachte aankomstdatum, vervoerswijze, zeevracht, luchtvracht, \
      název lodi, číslo letu, číslo plavby, přístav nakládky, přístav vykládky, přístav určení, místo převzetí, místo dodání, předpokládaný odjezd, předpokládaný příjezd, způsob dopravy, námořní přeprava, letecká přeprava, \
      اسم السفينة, رقم الرحلة الجوية, رقم الرحلة البحرية, ميناء الشحن, ميناء التفريغ, ميناء الوصول, مكان الاستلام, مكان التسليم, موعد المغادرة المتوقع, موعد الوصول المتوقع, وسيلة النقل, شحن بحري, شحن جوي, \
      船名, 便名, 航海番号, 積出港, 荷揚港, 仕向港, 受取地, 引渡地, 出港予定日, 入港予定日, 輸送手段, 海上輸送, 航空輸送, \
      航班号, 航次, 装货港, 卸货港, 目的港, 收货地, 交货地, 预计离港, 预计到港, 运输方式, 海运, 空运, \
      선박명, 항공편명, 항차, 선적항, 양하항, 목적항, 수취지, 인도지, 출항예정일, 도착예정일, 운송수단, 해상운송, 항공운송"),
    ("parties",
     "Versender, Exporteur, Verkäufer, Lieferant, Absender, Empfänger, Importeur, Käufer, Benachrichtigungsadresse, Begünstigter, Antragsteller, Firmenname, \
      expéditeur, exportateur, vendeur, fournisseur, destinataire, importateur, acheteur, partie à notifier, bénéficiaire, donneur d'ordre, raison sociale, \
      embarcador, exportador, vendedor, proveedor, consignatario, importador, comprador, parte a notificar, beneficiario, solicitante, razón social, \
      spedizioniere, esportatore, venditore, fornitore, destinatario, importatore, acquirente, parte da notificare, beneficiario, ordinante, ragione sociale, \
      fornecedor, consignatário, requerente, \
      verlader, exporteur, verkoper, leverancier, geadresseerde, importeur, koper, begunstigde, aanvrager, bedrijfsnaam, \
      odesílatel, vývozce, prodávající, dodavatel, příjemce, dovozce, kupující, oznamovací strana, oprávněný, žadatel, název společnosti, \
      الشاحن, المصدر, البائع, المورد, المرسل إليه, المستورد, المشتري, الجهة المخطرة, المستفيد, مقدم الطلب, اسم الشركة, \
      荷送人, 輸出者, 売主, 供給者, 荷受人, 輸入者, 買主, 着荷通知先, 受益者, 発行依頼人, 会社名, \
      发货人, 出口商, 卖方, 供应商, 收货人, 进口商, 买方, 通知方, 受益人, 申请人, 公司名称, \
      송하인, 수출자, 매도인, 공급자, 수하인, 수입자, 매수인, 통지처, 수익자, 개설의뢰인, 회사명"),
    ("terms",
     "Lieferbedingungen, Preisbedingungen, Zahlungsbedingungen, Akkreditiv, Fracht vorausbezahlt, Fracht unfrei, Währung, Gesamtbetrag, Rechnungswert, Frachtkosten, Versicherungskosten, \
      conditions de livraison, conditions de prix, conditions de paiement, lettre de crédit, fret prépayé, fret dû, devise, montant total, valeur de la facture, frais de fret, frais d'assurance, \
      condiciones de entrega, condiciones de precio, condiciones de pago, carta de crédito, flete prepagado, flete por cobrar, moneda, importe total, valor de la factura, gastos de flete, gastos de seguro, \
      termini di resa, condizioni di prezzo, termini di pagamento, lettera di credito, nolo prepagato, nolo assegnato, valuta, importo totale, valore della fattura, spese di nolo, spese di assicurazione, \
      condições de entrega, condições de preço, condições de pagamento, frete pré-pago, frete a pagar, moeda, valor total, valor da fatura, despesas de frete, despesas de seguro, \
      leveringsvoorwaarden, prijsvoorwaarden, betalingsvoorwaarden, kredietbrief, vracht vooruitbetaald, vracht te betalen, valuta, totaalbedrag, factuurwaarde, vrachtkosten, verzekeringskosten, \
      dodací podmínky, cenové podmínky, platební podmínky, akreditiv, přepravné předplaceno, přepravné k úhradě, měna, celková částka, hodnota faktury, náklady na přepravu, náklady na pojištění, \
      الإنكوتيرمز, شروط التسليم, شروط السعر, شروط الدفع, خطاب الاعتماد, أجرة الشحن مدفوعة مسبقاً, أجرة الشحن عند التسليم, العملة, المبلغ الإجمالي, قيمة الفاتورة, رسوم الشحن, رسوم التأمين, \
      インコタームズ, 引渡条件, 価格条件, 支払条件, 信用状, 運賃前払, 運賃着払, 通貨, 合計金額, 請求金額, 運賃, 保険料, \
      国际贸易术语, 交货条件, 价格条件, 付款条件, 信用证, 运费预付, 运费到付, 币种, 总金额, 发票金额, 运费, 保险费, \
      인코텀즈, 인도조건, 가격조건, 결제조건, 신용장, 운임선불, 운임후불, 통화, 총금액, 송장금액, 운임, 보험료"),
    ("cargo",
     "Containernummer, Plombennummer, Anzahl der Packstücke, Kartonanzahl, Palettenanzahl, Bruttogewicht, Nettogewicht, Volumen, Kubikmeter, Zolltarifnummer, Markierungen und Nummern, \
      numéro de conteneur, numéro de scellé, nombre de colis, nombre de cartons, nombre de palettes, poids brut, poids net, volume, mètre cube, code SH, marques et numéros, \
      número de contenedor, número de precinto, número de bultos, número de cajas, número de palés, peso bruto, peso neto, volumen, metro cúbico, código arancelario, marcas y números, \
      numero del container, numero del sigillo, numero di colli, numero di cartoni, numero di pallet, peso lordo, peso netto, metro cubo, codice doganale, marche e numeri, \
      número do contêiner, número do lacre, quantidade de volumes, quantidade de caixas, quantidade de paletes, peso líquido, código NCM, marcas e números, \
      containernummer, zegelnummer, aantal colli, aantal dozen, aantal pallets, brutogewicht, nettogewicht, kubieke meter, GS-code, merken en nummers, \
      číslo kontejneru, číslo plomby, počet balení, počet kartonů, počet palet, hrubá hmotnost, čistá hmotnost, objem, metr krychlový, kód HS, značky a čísla, \
      رقم الحاوية, رقم الختم, عدد الطرود, عدد الكراتين, عدد المنصات, الوزن الإجمالي, الوزن الصافي, الحجم, متر مكعب, رمز النظام المنسق, العلامات والأرقام, \
      コンテナ番号, シール番号, 梱包数, カートン数, パレット数, 総重量, 純重量, 容積, 立方メートル, HSコード, 荷印, \
      集装箱号, 铅封号, 件数, 箱数, 托盘数, 毛重, 净重, 体积, 立方米, HS编码, 唛头, \
      컨테이너번호, 봉인번호, 포장수량, 카톤수, 팔레트수, 총중량, 순중량, 용적, 입방미터, HS코드, 화인"),
    ("reference",
     "referenzierte Rechnungsnummer, referenzierte Konnossementnummer, referenzierte Bestellnummer, referenzierte Akkreditivnummer, referenzierte Buchungsnummer, referenzierte Vertragsnummer, unsere Referenz, Ihre Referenz, bezüglich Dokument, \
      numéro de facture référencé, numéro de connaissement référencé, numéro de commande référencé, numéro de lettre de crédit référencé, numéro de réservation référencé, numéro de contrat référencé, notre référence, votre référence, document concerné, \
      número de factura referenciado, número de conocimiento de embarque referenciado, número de pedido referenciado, número de carta de crédito referenciado, número de reserva referenciado, número de contrato referenciado, nuestra referencia, su referencia, documento relacionado, \
      numero fattura di riferimento, numero polizza di carico di riferimento, numero ordine di riferimento, numero lettera di credito di riferimento, numero prenotazione di riferimento, numero contratto di riferimento, nostro riferimento, vostro riferimento, documento correlato, \
      número da fatura referenciada, número do conhecimento de embarque referenciado, número do pedido referenciado, número da carta de crédito referenciada, número da reserva referenciada, número do contrato referenciado, nossa referência, sua referência, \
      gerefereerd factuurnummer, gerefereerd cognossementnummer, gerefereerd ordernummer, gerefereerd kredietbriefnummer, gerefereerd boekingsnummer, gerefereerd contractnummer, onze referentie, uw referentie, betreffend document, \
      odkazované číslo faktury, odkazované číslo konosamentu, odkazované číslo objednávky, odkazované číslo akreditivu, odkazované číslo rezervace, odkazované číslo smlouvy, naše značka, vaše značka, související dokument, \
      رقم الفاتورة المرجعي, رقم بوليصة الشحن المرجعي, رقم أمر الشراء المرجعي, رقم خطاب الاعتماد المرجعي, رقم الحجز المرجعي, رقم العقد المرجعي, مرجعنا, مرجعكم, المستند المرتبط, \
      参照請求書番号, 参照船荷証券番号, 参照注文番号, 参照信用状番号, 参照ブッキング番号, 参照契約番号, 当方参照番号, 貴社参照番号, 関連書類, \
      参考发票号, 参考提单号, 参考订单号, 参考信用证号, 参考订舱号, 参考合同号, 我方参考号, 贵方参考号, 相关单据, \
      참조 송장번호, 참조 선하증권번호, 참조 주문번호, 참조 신용장번호, 참조 부킹번호, 참조 계약번호, 당사 참조번호, 귀사 참조번호, 관련 문서"),
    ("hub",
     "alle Dokumente zu dieser Nummer, gesamte Dokumentenkette, alle Unterlagen dieser Sendung, alles was mit dieser Bestellung zusammenhängt, vollständiger Dokumentensatz, \
      tous les documents liés à ce numéro, chaîne documentaire complète, tous les documents de cette expédition, tout ce qui concerne cette commande, jeu complet de documents, \
      todos los documentos vinculados a este número, cadena documental completa, toda la documentación de este envío, todo lo relacionado con este pedido, juego completo de documentos, \
      tutti i documenti collegati a questo numero, catena documentale completa, tutti i documenti di questa spedizione, tutto ciò che riguarda questo ordine, set completo di documenti, \
      todos os documentos ligados a este número, toda a documentação deste embarque, tudo relacionado a este pedido, conjunto completo de documentos, \
      alle documenten gekoppeld aan dit nummer, volledige documentketen, alle documenten van deze zending, alles wat met deze order samenhangt, volledige documentenset, \
      všechny dokumenty spojené s tímto číslem, celý řetězec dokumentů, veškeré doklady této zásilky, vše související s touto objednávkou, kompletní sada dokumentů, \
      جميع المستندات المرتبطة بهذا الرقم, سلسلة المستندات الكاملة, كافة أوراق هذه الشحنة, كل ما يتعلق بهذا الطلب, مجموعة المستندات الكاملة, \
      この番号に関連する全ての書類, 書類チェーン全体, この出荷の全書類, この注文に関する全て, 書類一式, \
      与此编号相关的所有单据, 完整单据链, 本批货物的全部单据, 与此订单相关的一切, 全套单据, \
      이 번호와 관련된 모든 서류, 전체 문서 체인, 이 선적의 모든 서류, 이 주문과 연결된 전부, 서류 일체"),
    ("customs",
     "Zollanmeldungsnummer, Ausfuhranmeldung, Einfuhranmeldung, Anmeldedatum, Freigabedatum, Zollstellencode, Zollabfertigungsstatus, Zollsatz, Zollbetrag, Zollwert, Zollagent, Zolllager, \
      numéro de déclaration en douane, déclaration d'exportation, déclaration d'importation, date de déclaration, date de dédouanement, code du bureau de douane, statut du dédouanement, taux de droit, montant des droits, valeur en douane, commissionnaire en douane, entrepôt sous douane, \
      número de declaración aduanera, declaración de exportación, declaración de importación, fecha de declaración, fecha de despacho, código de la aduana, estado del despacho aduanero, tipo arancelario, importe de aranceles, valor en aduana, agente de aduanas, depósito aduanero, \
      numero di dichiarazione doganale, dichiarazione di esportazione, dichiarazione di importazione, data della dichiarazione, data di sdoganamento, codice ufficio doganale, stato dello sdoganamento, aliquota daziaria, importo dei dazi, valore in dogana, spedizioniere doganale, deposito doganale, \
      número da declaração aduaneira, declaração de exportação, declaração de importação, data da declaração, data do desembaraço, código da alfândega, situação do desembaraço, valor dos impostos, valor aduaneiro, despachante aduaneiro, armazém alfandegado, \
      aangiftenummer, uitvoeraangifte, invoeraangifte, aangiftedatum, datum van vrijgave, code douanekantoor, status inklaring, tarief invoerrecht, bedrag invoerrechten, douanewaarde, douane-expediteur, douane-entrepot, \
      číslo celního prohlášení, vývozní prohlášení, dovozní prohlášení, datum prohlášení, datum propuštění, kód celního úřadu, stav celního odbavení, celní sazba, výše cla, celní hodnota, celní deklarant, celní sklad, \
      رقم البيان الجمركي, بيان التصدير, بيان الاستيراد, تاريخ البيان, تاريخ الإفراج, رمز المكتب الجمركي, حالة التخليص الجمركي, نسبة الرسوم, مبلغ الرسوم الجمركية, القيمة الجمركية, المخلص الجمركي, مستودع جمركي, \
      通関申告番号, 輸出申告, 輸入申告, 申告日, 許可日, 税関コード, 通関状況, 関税率, 関税額, 課税価格, 通関業者, 保税倉庫, \
      报关单号, 出口申报, 进口申报, 申报日期, 放行日期, 海关代码, 清关状态, 关税税率, 关税金额, 完税价格, 报关行, 保税仓库, \
      수출입신고번호, 수출신고, 수입신고, 신고일, 통관일, 세관코드, 통관상태, 관세율, 관세액, 과세가격, 관세사, 보세창고"),
    ("inspection",
     "Inspektionszertifikat, Inspektionsdatum, Inspektionsort, Prüfergebnis, Zertifikatsnummer, Laboranalyse, Testergebnis, Begasung, Hitzebehandlung, Wiegedatum, Besichtigungsbericht, Schadensfeststellung, Pflanzengesundheitszeugnis, Gesundheitszeugnis, \
      certificat d'inspection, date d'inspection, lieu d'inspection, résultat de l'inspection, numéro de certificat, analyse en laboratoire, résultat d'essai, fumigation, traitement thermique, date de pesée, rapport d'expertise, constat de dommages, certificat phytosanitaire, certificat sanitaire, \
      certificado de inspección, fecha de inspección, lugar de inspección, resultado de la inspección, número de certificado, análisis de laboratorio, resultado de la prueba, fumigación, tratamiento térmico, fecha de pesaje, informe de peritaje, constatación de daños, certificado fitosanitario, certificado sanitario, \
      certificato di ispezione, data dell'ispezione, luogo dell'ispezione, esito dell'ispezione, numero del certificato, analisi di laboratorio, risultato del test, fumigazione, trattamento termico, data di pesatura, rapporto di perizia, accertamento dei danni, certificato fitosanitario, certificato sanitario, \
      certificado de inspeção, data da inspeção, local da inspeção, resultado da inspeção, número do certificado, análise laboratorial, resultado do teste, fumigação, data da pesagem, laudo de vistoria, constatação de danos, certificado fitossanitário, certificado sanitário, \
      inspectiecertificaat, inspectiedatum, plaats van inspectie, inspectieresultaat, certificaatnummer, laboratoriumanalyse, testresultaat, fumigatie, hittebehandeling, weegdatum, expertiserapport, schadevaststelling, fytosanitair certificaat, gezondheidscertificaat, \
      inspekční certifikát, datum inspekce, místo inspekce, výsledek inspekce, číslo certifikátu, laboratorní analýza, výsledek zkoušky, fumigace, tepelné ošetření, datum vážení, zpráva o průzkumu, zjištění škod, rostlinolékařské osvědčení, zdravotní osvědčení, \
      شهادة فحص, تاريخ الفحص, مكان الفحص, نتيجة الفحص, رقم الشهادة, تحليل مخبري, نتيجة الاختبار, تبخير, معالجة حرارية, تاريخ الوزن, تقرير المعاينة, تحديد الأضرار, شهادة صحة نباتية, شهادة صحية, \
      検査証明書, 検査日, 検査場所, 検査結果, 証明書番号, 検査分析, 試験結果, 燻蒸, 熱処理, 計量日, 鑑定報告書, 損害の所見, 植物検疫証明書, 衛生証明書, \
      检验证书, 检验日期, 检验地点, 检验结果, 证书编号, 实验室分析, 测试结果, 熏蒸, 热处理, 称重日期, 检验报告, 损害查定, 植物检疫证书, 卫生证书, \
      검사증명서, 검사일, 검사장소, 검사결과, 증명서번호, 실험실 분석, 시험결과, 훈증, 열처리, 계량일, 검정보고서, 손해 조사 결과, 식물검역증명서, 위생증명서"),
    ("settlement",
     "Kontoauszug, Kontobuch, Transaktionsdatum, Sollbetrag, Habenbetrag, laufender Saldo, offener Saldo, Endsaldo, Lastschrift, Gutschrift, Steuerrechnung, Umsatzsteuerbetrag, Zahlungsstatus, unbezahlt, beglichen, Fälligkeitsdatum, überfällig, \
      relevé de compte, grand livre, date de transaction, montant au débit, montant au crédit, solde courant, solde impayé, solde final, note de débit, note de crédit, facture fiscale, montant de la TVA, statut du paiement, impayé, réglé, date d'échéance, en retard, \
      estado de cuenta, libro mayor, fecha de transacción, importe al debe, importe al haber, saldo acumulado, saldo pendiente, saldo final, nota de débito, nota de crédito, factura fiscal, importe del IVA, estado del pago, impagado, liquidado, fecha de vencimiento, vencido, \
      estratto conto, libro mastro, data della transazione, importo a debito, importo a credito, saldo progressivo, saldo insoluto, saldo finale, nota di debito, nota di credito, fattura fiscale, importo IVA, stato del pagamento, non pagato, saldato, data di scadenza, scaduto, \
      extrato de conta, livro razão, data da transação, valor a débito, valor a crédito, saldo corrente, saldo em aberto, nota fiscal, valor do IVA, situação do pagamento, não pago, \
      rekeningoverzicht, grootboek, transactiedatum, debetbedrag, creditbedrag, lopend saldo, openstaand saldo, eindsaldo, debetnota, creditnota, belastingfactuur, btw-bedrag, betalingsstatus, onbetaald, voldaan, vervaldatum, achterstallig, \
      výpis z účtu, účetní kniha, datum transakce, částka má dáti, částka dal, průběžný zůstatek, nesplacený zůstatek, konečný zůstatek, vrubopis, dobropis, daňový doklad, částka DPH, stav platby, nezaplaceno, uhrazeno, datum splatnosti, po splatnosti, \
      كشف حساب, دفتر الأستاذ, تاريخ المعاملة, مبلغ مدين, مبلغ دائن, الرصيد الجاري, الرصيد المستحق, الرصيد الختامي, إشعار مدين, إشعار دائن, فاتورة ضريبية, مبلغ ضريبة القيمة المضافة, حالة الدفع, غير مدفوع, مسدد, تاريخ الاستحقاق, متأخر, \
      取引明細書, 元帳, 取引日, 借方金額, 貸方金額, 繰越残高, 未払残高, 期末残高, デビットノート, クレジットノート, 税務請求書, 消費税額, 支払状況, 未払, 決済済, 支払期日, 期限超過, \
      对账单, 账簿, 交易日期, 借方金额, 贷方金额, 累计余额, 未结余额, 期末余额, 借项通知单, 贷项通知单, 税务发票, 增值税额, 付款状态, 未付, 已结清, 到期日, 逾期, \
      거래명세서, 원장, 거래일, 차변금액, 대변금액, 누적잔액, 미결제잔액, 기말잔액, 차변표, 대변표, 세금계산서, 부가세액, 결제상태, 미납, 완납, 만기일, 연체"),
    ("items",
     "Warenbeschreibung, Warenbezeichnung, Handelsware, Produktname, Artikel, Herstellungsland, Ursprungsland, hergestellt in, Stückpreis, Menge, Maßeinheit, Positionssumme, Artikelnummer, Modellnummer, \
      description des marchandises, désignation des marchandises, marchandise, nom du produit, article, ligne d'article, pays de fabrication, pays d'origine, fabriqué en, prix unitaire, quantité, unité de mesure, total de la ligne, référence article, numéro de modèle, \
      descripción de la mercancía, denominación de la mercancía, producto, nombre del producto, artículo, línea de artículo, país de fabricación, país de origen, fabricado en, precio unitario, cantidad, unidad de medida, total de línea, código de artículo, número de modelo, \
      descrizione delle merci, denominazione delle merci, merce, nome del prodotto, articolo, riga articolo, paese di fabbricazione, paese di origine, fabbricato in, prezzo unitario, quantità, unità di misura, totale riga, codice articolo, numero di modello, \
      descrição das mercadorias, denominação da mercadoria, produto, nome do produto, artigo, linha do item, país de fabricação, país de origem, fabricado em, preço unitário, quantidade, total da linha, código do item, número do modelo, \
      goederenomschrijving, omschrijving van de goederen, handelswaar, productnaam, artikel, regelitem, land van vervaardiging, land van oorsprong, vervaardigd in, eenheidsprijs, hoeveelheid, meeteenheid, regeltotaal, artikelnummer, modelnummer, \
      popis zboží, označení zboží, komodita, název výrobku, položka, řádková položka, země výroby, země původu, vyrobeno v, jednotková cena, množství, měrná jednotka, celkem za řádek, kód položky, číslo modelu, \
      وصف البضاعة, بيان البضاعة, السلعة, اسم المنتج, الصنف, بند السطر, بلد الصنع, بلد المنشأ, صنع في, سعر الوحدة, الكمية, وحدة القياس, إجمالي السطر, رمز الصنف, رقم الطراز, \
      品名, 商品説明, 製品名, 品目, 明細行, 製造国, 原産国, 製造元, 単価, 数量, 単位, 明細合計, 品番, 型番, \
      货物描述, 货物名称, 商品, 产品名称, 品目, 明细行, 制造国, 原产国, 产地, 单价, 数量, 计量单位, 行合计, 货号, 型号, \
      품명, 상품 설명, 상품, 제품명, 품목, 명세행, 제조국, 원산지, 제조지, 단가, 수량, 단위, 라인합계, 품목코드, 모델번호"),
];

pub fn trade_condition_category_phrases(category: &str) -> Vec<String> {
    let en = TRADE_CONDITION_CATEGORIES
        .iter()
        .find(|(c, _)| *c == category)
        .map(|(_, raw)| *raw)
        .unwrap_or("");
    let ml = TRADE_CONDITION_CATEGORIES_ML
        .iter()
        .find(|(c, _)| *c == category)
        .map(|(_, raw)| *raw)
        .unwrap_or("");
    anchor_phrases(en, ml)
}

pub fn trade_condition_all_phrases() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (c, _) in TRADE_CONDITION_CATEGORIES.iter() {
        for p in trade_condition_category_phrases(c) {
            if out.iter().any(|e| e.eq_ignore_ascii_case(&p)) { continue; }
            out.push(p);
        }
    }
    out
}

pub fn merge_phrase_bank(
    phrases: &mut Vec<String>,
    weights: &mut Vec<f32>,
    extra: &[String],
    weight: f32,
) -> usize {
    while weights.len() < phrases.len() {
        weights.push(1.0);
    }
    let mut added = 0usize;
    for p in extra.iter() {
        let t = p.trim();
        if t.is_empty() { continue; }
        if phrases.iter().any(|e| e.eq_ignore_ascii_case(t)) { continue; }
        phrases.push(t.to_string());
        weights.push(weight);
        added += 1;
    }
    added
}

pub const TRADE_LABEL_SUPPLEMENT_ML: &[(&str, &str, &str)] = &[
    ("reference_po",
     "P/O No., PO No., PO number, purchase order number, purchase order no, order no, order number, your order, customer order number, order reference",
     "Bestellnummer, Best.-Nr., Ihre Bestellung, \
      numéro de commande, N° de commande, votre commande, \
      número de pedido, Nº de pedido, su pedido, \
      numero d'ordine, N. ordine, vostro ordine, \
      número do pedido, Nº do pedido, seu pedido, \
      ordernummer, bestelnummer, uw order, \
      číslo objednávky, č. objednávky, vaše objednávka, \
      رقم أمر الشراء, رقم الطلبية, طلبكم, \
      注文番号, 発注番号, 貴社注文番号, \
      订单号, 采购订单号, 贵方订单号, \
      주문번호, 발주번호, 귀사 주문번호"),
    ("reference_invoice",
     "invoice no, invoice number, commercial invoice no, commercial invoice number, invoice reference, against invoice, covering invoice, inv no",
     "Rechnungsnummer, Rechnungs-Nr., Handelsrechnungsnummer, \
      numéro de facture, N° de facture, numéro de facture commerciale, \
      número de factura, Nº de factura, número de factura comercial, \
      numero fattura, N. fattura, numero fattura commerciale, \
      número da fatura, Nº da fatura, número da fatura comercial, \
      factuurnummer, factuur nr., handelsfactuurnummer, \
      číslo faktury, č. faktury, číslo obchodní faktury, \
      رقم الفاتورة, رقم الفاتورة التجارية, \
      請求書番号, インボイス番号, 商業送り状番号, \
      发票号, 发票编号, 商业发票号, \
      송장번호, 인보이스 번호, 상업송장번호"),
    ("reference_bl",
     "B/L No., BL No., bill of lading number, bill of lading no, ocean B/L no, airwaybill / bill of lading, waybill number, transport document number, B/L reference",
     "Konnossementnummer, B/L-Nr., Frachtbriefnummer, Transportdokumentnummer, \
      numéro de connaissement, N° de connaissement, numéro de lettre de transport, numéro du document de transport, \
      número de conocimiento de embarque, Nº de B/L, número de carta de porte, número del documento de transporte, \
      numero polizza di carico, N. polizza di carico, numero lettera di vettura, numero documento di trasporto, \
      número do conhecimento de embarque, Nº do B/L, número da carta de porte, número do documento de transporte, \
      cognossementnummer, B/L-nummer, vrachtbriefnummer, nummer vervoersdocument, \
      číslo konosamentu, č. B/L, číslo nákladního listu, číslo přepravního dokladu, \
      رقم بوليصة الشحن, رقم سند الشحن, رقم وثيقة النقل, \
      船荷証券番号, B/L番号, 運送状番号, 輸送書類番号, \
      提单号, B/L号, 运单号, 运输单据号, \
      선하증권번호, B/L번호, 운송장번호, 운송서류번호"),
    ("reference_lc",
     "L/C No., LC No., letter of credit number, letter of credit no, documentary credit number, credit no, credit number, drawn under L/C, L/C reference",
     "Akkreditivnummer, L/C-Nr., Dokumentenakkreditiv-Nummer, \
      numéro de lettre de crédit, N° de L/C, numéro de crédit documentaire, \
      número de carta de crédito, Nº de L/C, número de crédito documentario, \
      numero lettera di credito, N. L/C, numero credito documentario, \
      número da carta de crédito, Nº da L/C, número do crédito documentário, \
      kredietbriefnummer, L/C-nummer, documentair kredietnummer, \
      číslo akreditivu, č. L/C, číslo dokumentárního akreditivu, \
      رقم خطاب الاعتماد, رقم الاعتماد المستندي, \
      信用状番号, L/C番号, 荷為替信用状番号, \
      信用证号, L/C号, 跟单信用证号, \
      신용장번호, L/C번호, 화환신용장번호"),
    ("reference_booking",
     "booking no, booking number, booking reference, BKG No., booking confirmation number, space booking number",
     "Buchungsnummer, Buchungs-Nr., Buchungsreferenz, \
      numéro de réservation, N° de booking, référence de réservation, \
      número de reserva, Nº de booking, referencia de reserva, \
      numero di prenotazione, N. booking, riferimento prenotazione, \
      número da reserva, Nº do booking, referência da reserva, \
      boekingsnummer, booking nr., boekingsreferentie, \
      číslo rezervace, č. bookingu, reference rezervace, \
      رقم الحجز, مرجع الحجز, \
      ブッキング番号, 予約番号, ブッキング照会番号, \
      订舱号, 订舱编号, 订舱参考号, \
      부킹번호, 예약번호, 부킹 참조번호"),
    ("reference_contract",
     "contract no, contract number, sales contract number, S/C No., agreement number, agreement no, contract reference",
     "Vertragsnummer, Vertrags-Nr., Kaufvertragsnummer, \
      numéro de contrat, N° de contrat, numéro de contrat de vente, \
      número de contrato, Nº de contrato, número de contrato de venta, \
      numero contratto, N. contratto, numero contratto di vendita, \
      número do contrato, Nº do contrato, número do contrato de venda, \
      contractnummer, contract nr., verkoopcontractnummer, \
      číslo smlouvy, č. smlouvy, číslo kupní smlouvy, \
      رقم العقد, رقم عقد البيع, \
      契約番号, 契約No., 売買契約番号, \
      合同号, 合同编号, 销售合同号, \
      계약번호, 계약 No., 매매계약번호"),
    ("reference_number",
     "reference no, ref no, reference number, export reference, export ref., our reference, our ref., your reference, your ref., job no, file no, case no, order reference, customer reference, shipper's reference",
     "Referenznummer, Ref.-Nr., Exportreferenz, unsere Referenz, Ihre Referenz, Aktenzeichen, \
      numéro de référence, N° de réf., référence export, notre référence, votre référence, numéro de dossier, \
      número de referencia, Nº de ref., referencia de exportación, nuestra referencia, su referencia, número de expediente, \
      numero di riferimento, N. rif., riferimento export, nostro riferimento, vostro riferimento, numero pratica, \
      número de referência, Nº de ref., referência de exportação, nossa referência, sua referência, número do processo, \
      referentienummer, ref. nr., exportreferentie, onze referentie, uw referentie, dossiernummer, \
      referenční číslo, ref. č., exportní reference, naše značka, vaše značka, číslo spisu, \
      الرقم المرجعي, رقم المرجع, مرجع التصدير, مرجعنا, مرجعكم, رقم الملف, \
      参照番号, 照会番号, 輸出参照番号, 当方参照, 貴社参照, 案件番号, \
      参考号, 参考编号, 出口参考号, 我方参考号, 贵方参考号, 案卷号, \
      참조번호, 조회번호, 수출 참조번호, 당사 참조번호, 귀사 참조번호, 사건번호"),
    ("reference_awb",
     "AWB No., air waybill number, air waybill no, airway bill number, airwaybill number, MAWB No., HAWB No.",
     "Luftfrachtbriefnummer, AWB-Nr., \
      numéro de lettre de transport aérien, N° de LTA, numéro AWB, \
      número de guía aérea, Nº de guía aérea, número AWB, \
      numero lettera di vettura aerea, N. LTA, numero AWB, \
      número do conhecimento aéreo, Nº do AWB, número da guia aérea, \
      luchtvrachtbriefnummer, AWB-nummer, \
      číslo leteckého nákladního listu, č. AWB, \
      رقم بوليصة الشحن الجوي, رقم AWB, \
      航空運送状番号, AWB番号, \
      空运单号, 航空运单号, AWB号, \
      항공운송장번호, AWB번호"),
    ("reference_hbl",
     "House B/L No., HBL No., house bill of lading number, house bill of lading no, forwarder's B/L number",
     "House-B/L-Nr., House-Konnossementnummer, Spediteurkonnossement-Nr., \
      numéro de House B/L, numéro de connaissement maison, N° de HBL, \
      número de House B/L, número de conocimiento de embarque hijo, Nº de HBL, \
      numero House B/L, numero polizza di carico house, N. HBL, \
      número do House B/L, número do conhecimento house, Nº do HBL, \
      House B/L-nummer, house cognossementnummer, HBL-nummer, \
      číslo House B/L, číslo house konosamentu, č. HBL, \
      رقم بوليصة الشحن الفرعية, رقم HBL, \
      ハウスB/L番号, HBL番号, ハウス船荷証券番号, \
      货代提单号, House提单号, HBL号, \
      하우스 B/L번호, HBL번호, 하우스 선하증권번호"),
    ("reference_master_bl",
     "Master B/L No., MBL No., master bill of lading number, master bill of lading no, ocean carrier B/L no",
     "Master-B/L-Nr., Master-Konnossementnummer, Reederei-Konnossement-Nr., \
      numéro de Master B/L, numéro de connaissement mère, N° de MBL, \
      número de Master B/L, número de conocimiento de embarque madre, Nº de MBL, \
      numero Master B/L, numero polizza di carico master, N. MBL, \
      número do Master B/L, número do conhecimento master, Nº do MBL, \
      Master B/L-nummer, master cognossementnummer, MBL-nummer, \
      číslo Master B/L, číslo master konosamentu, č. MBL, \
      رقم بوليصة الشحن الرئيسية, رقم MBL, \
      マスターB/L番号, MBL番号, マスター船荷証券番号, \
      船公司提单号, Master提单号, MBL号, \
      마스터 B/L번호, MBL번호, 마스터 선하증권번호"),
    ("reference_proforma",
     "proforma invoice no, proforma invoice number, PI No., pro forma invoice number, proforma no",
     "Proformarechnungsnummer, Proforma-Nr., PI-Nr., \
      numéro de facture pro forma, N° de pro forma, N° de PI, \
      número de factura proforma, Nº de proforma, Nº de PI, \
      numero fattura proforma, N. proforma, N. PI, \
      número da fatura pró-forma, Nº da pró-forma, Nº da PI, \
      proformafactuurnummer, proforma nr., PI-nummer, \
      číslo proforma faktury, č. proformy, č. PI, \
      رقم الفاتورة المبدئية, رقم PI, \
      プロフォーマインボイス番号, 見積送り状番号, PI番号, \
      形式发票号, PI号, 形式发票编号, \
      견적송장번호, 프로포마 인보이스 번호, PI번호"),
    ("reference_do",
     "delivery order no, delivery order number, D/O No., DO No., release order number",
     "Auslieferungsauftragsnummer, D/O-Nr., Freigabeauftragsnummer, \
      numéro de bon de livraison, N° de D/O, numéro d'ordre de livraison, \
      número de orden de entrega, Nº de D/O, número de orden de liberación, \
      numero ordine di consegna, N. D/O, numero ordine di svincolo, \
      número da ordem de entrega, Nº do D/O, número da ordem de liberação, \
      afleveringsordernummer, D/O-nummer, vrijgaveordernummer, \
      číslo dodacího příkazu, č. D/O, číslo příkazu k vydání, \
      رقم أمر التسليم, رقم D/O, رقم أمر الإفراج, \
      荷渡指図書番号, D/O番号, 引渡指図番号, \
      提货单号, D/O号, 放货单号, \
      화물인도지시서번호, D/O번호, 인도지시번호"),
    ("reference_export_decl",
     "export declaration no, export declaration number, ED No., export entry number, export permit number, customs export declaration no",
     "Ausfuhranmeldungsnummer, Ausfuhrerklärung-Nr., Ausfuhrgenehmigungsnummer, \
      numéro de déclaration d'exportation, N° de déclaration export, numéro de DAE, \
      número de declaración de exportación, Nº de DUA de exportación, número de despacho de exportación, \
      numero dichiarazione di esportazione, N. dichiarazione export, numero bolla doganale export, \
      número da declaração de exportação, Nº da DUE, número do despacho de exportação, \
      uitvoeraangiftenummer, nummer exportaangifte, exportvergunningnummer, \
      číslo vývozního prohlášení, č. vývozní deklarace, číslo vývozního povolení, \
      رقم بيان التصدير, رقم إقرار التصدير, رقم إذن التصدير, \
      輸出申告番号, 輸出許可番号, 輸出申告No., \
      出口报关单号, 出口申报号, 出口许可证号, \
      수출신고번호, 수출신고필증번호, 수출허가번호"),
    ("reference_import_decl",
     "import declaration no, import declaration number, ID No., import entry number, import permit number, customs import declaration no, entry no",
     "Einfuhranmeldungsnummer, Einfuhrerklärung-Nr., Einfuhrgenehmigungsnummer, \
      numéro de déclaration d'importation, N° de déclaration import, numéro de DAU, \
      número de declaración de importación, Nº de DUA de importación, número de despacho de importación, \
      numero dichiarazione di importazione, N. dichiarazione import, numero bolla doganale import, \
      número da declaração de importação, Nº da DI, número do despacho de importação, \
      invoeraangiftenummer, nummer importaangifte, importvergunningnummer, \
      číslo dovozního prohlášení, č. dovozní deklarace, číslo dovozního povolení, \
      رقم بيان الاستيراد, رقم إقرار الاستيراد, رقم إذن الاستيراد, \
      輸入申告番号, 輸入許可番号, 輸入申告No., \
      进口报关单号, 进口申报号, 进口许可证号, \
      수입신고번호, 수입신고필증번호, 수입허가번호"),
    ("reference_origin",
     "certificate of origin no, certificate of origin number, C/O No., CO No., origin certificate number",
     "Ursprungszeugnisnummer, UZ-Nr., Nummer des Ursprungszeugnisses, \
      numéro de certificat d'origine, N° de C/O, numéro du certificat d'origine, \
      número de certificado de origen, Nº de C/O, número del certificado de origen, \
      numero certificato di origine, N. C/O, numero del certificato di origine, \
      número do certificado de origem, Nº do C/O, número do certificado de origem, \
      nummer certificaat van oorsprong, C/O-nummer, oorsprongscertificaatnummer, \
      číslo osvědčení o původu, č. C/O, číslo certifikátu původu, \
      رقم شهادة المنشأ, رقم C/O, \
      原産地証明書番号, C/O番号, 原産地証明番号, \
      原产地证书号, C/O号, 原产地证编号, \
      원산지증명서번호, C/O번호, 원산지증명번호"),
    ("reference_policy",
     "insurance policy no, insurance policy number, policy no, policy number, certificate of insurance no, cover note number",
     "Versicherungspolicennummer, Policen-Nr., Versicherungsscheinnummer, \
      numéro de police d'assurance, N° de police, numéro de certificat d'assurance, \
      número de póliza de seguro, Nº de póliza, número de certificado de seguro, \
      numero polizza assicurativa, N. polizza, numero certificato di assicurazione, \
      número da apólice de seguro, Nº da apólice, número do certificado de seguro, \
      verzekeringspolisnummer, polisnummer, nummer verzekeringscertificaat, \
      číslo pojistné smlouvy, č. pojistky, číslo pojistného certifikátu, \
      رقم وثيقة التأمين, رقم البوليصة, رقم شهادة التأمين, \
      保険証券番号, ポリシー番号, 保険証明書番号, \
      保险单号, 保单号, 保险证明书号, \
      보험증권번호, 보험 증권 No., 보험증명서번호"),
    ("departure_date",
     "date of exportation, date of export, export date, exportation date, date of shipment, shipment date, date shipped, shipped on, dispatch date, date of dispatch, date of departure",
     "Ausfuhrdatum, Datum der Ausfuhr, Versanddatum, Verschiffungsdatum, Abfahrtsdatum, \
      date d'exportation, date d'expédition, date d'embarquement, date de départ, \
      fecha de exportación, fecha de embarque, fecha de envío, fecha de salida, \
      data di esportazione, data di spedizione, data di imbarco, data di partenza, \
      data de exportação, data de embarque, data de envio, data de partida, \
      uitvoerdatum, datum van uitvoer, verzenddatum, verschepingsdatum, \
      datum vývozu, datum odeslání, datum nalodění, datum odjezdu, \
      تاريخ التصدير, تاريخ الشحن, تاريخ الإرسال, تاريخ المغادرة, \
      輸出日, 出荷日, 船積日, 出港日, \
      出口日期, 发货日期, 装运日期, 离港日期, \
      수출일, 출하일, 선적일, 출항일"),
    ("arrival_date",
     "date of arrival, arrival date, arrived on, date arrived, discharge date, date of discharge",
     "Ankunftsdatum, Datum der Ankunft, Löschdatum, \
      date d'arrivée, date de déchargement, arrivé le, \
      fecha de llegada, fecha de arribo, fecha de descarga, \
      data di arrivo, data di scarico, arrivato il, \
      data de chegada, data de descarga, \
      aankomstdatum, datum van aankomst, losdatum, \
      datum příjezdu, datum příchodu, datum vykládky, \
      تاريخ الوصول, تاريخ التفريغ, \
      到着日, 入港日, 荷揚日, \
      到达日期, 到港日期, 卸货日期, \
      도착일, 입항일, 양하일"),
    ("place_delivery",
     "place of delivery, final destination, place of final delivery, delivery place, delivered to, final delivery point, ultimate destination, place of destination, door delivery address",
     "Lieferort, Auslieferungsort, Endbestimmungsort, endgültiger Bestimmungsort, \
      lieu de livraison, lieu de livraison finale, destination finale, lieu de destination, \
      lugar de entrega, lugar de entrega final, destino final, lugar de destino, \
      luogo di consegna, luogo di consegna finale, destinazione finale, luogo di destinazione, \
      local de entrega, local de entrega final, local de destino, \
      plaats van levering, plaats van aflevering, eindbestemming, plaats van bestemming, \
      místo dodání, místo konečného dodání, konečné místo určení, místo určení, \
      مكان التسليم, مكان التسليم النهائي, الوجهة النهائية, مكان الوصول, \
      引渡地, 最終引渡地, 最終仕向地, 配達先, \
      交货地, 最终交货地, 最终目的地, 送货地址, \
      인도지, 최종 인도지, 최종 목적지, 배송지"),
    ("country_of_destination",
     "country of destination, country of ultimate destination, ultimate destination country, destination country, final destination country, ship to country, country of final destination, importing country",
     "Bestimmungsland, endgültiges Bestimmungsland, Empfangsland, Einfuhrland, \
      pays de destination, pays de destination finale, pays destinataire, pays d'importation, \
      país de destino, país de destino final, país destinatario, país de importación, \
      paese di destinazione, paese di destinazione finale, paese destinatario, paese di importazione, \
      país destinatário, país de importação, \
      land van bestemming, land van eindbestemming, ontvangend land, land van invoer, \
      země určení, země konečného určení, země příjemce, země dovozu, \
      بلد الوصول, بلد المقصد النهائي, بلد الوجهة, بلد الاستيراد, \
      仕向国, 最終仕向国, 到着国, 輸入国, \
      目的国, 最终目的国, 到达国, 进口国, \
      목적국, 최종 목적국, 도착국, 수입국"),
    ("country_of_export",
     "country of export, country of exportation, exporting country, country of dispatch, country of departure, ship from country, country of consignment, country of shipment",
     "Ausfuhrland, Versendungsland, Abgangsland, Exportland, \
      pays d'exportation, pays d'expédition, pays de départ, pays de provenance, \
      país de exportación, país de expedición, país de salida, país de procedencia, \
      paese di esportazione, paese di spedizione, paese di partenza, paese di provenienza, \
      país de exportação, país de expedição, país de partida, país de procedência, \
      land van uitvoer, land van verzending, land van vertrek, exportland, \
      země vývozu, země odeslání, země odjezdu, vyvážející země, \
      بلد التصدير, بلد الإرسال, بلد المغادرة, بلد المصدر, \
      輸出国, 積出国, 出発国, 発送国, \
      出口国, 发货国, 起运国, 启运国, \
      수출국, 발송국, 출발국, 적출국"),
    ("reason_for_export",
     "reason for export, purpose of export, export reason, export purpose, purpose of shipment, reason for shipment, nature of transaction, purpose of transaction",
     "Grund der Ausfuhr, Ausfuhrgrund, Zweck der Ausfuhr, Art des Geschäfts, \
      motif de l'exportation, raison de l'exportation, objet de l'exportation, nature de la transaction, \
      motivo de la exportación, razón de la exportación, propósito de la exportación, naturaleza de la transacción, \
      motivo dell'esportazione, ragione dell'esportazione, scopo dell'esportazione, natura della transazione, \
      motivo da exportação, razão da exportação, finalidade da exportação, natureza da transação, \
      reden van uitvoer, doel van uitvoer, reden van verzending, aard van de transactie, \
      důvod vývozu, účel vývozu, důvod odeslání, povaha transakce, \
      سبب التصدير, غرض التصدير, سبب الشحن, طبيعة المعاملة, \
      輸出理由, 輸出目的, 出荷理由, 取引の性質, \
      出口原因, 出口目的, 发货原因, 交易性质, \
      수출 사유, 수출 목적, 출하 사유, 거래 성격"),
    ("sender_tax_number",
     "exporter VAT/EORI, exporter VAT number, exporter EORI number, exporter tax ID, shipper tax ID, shipper VAT number, seller VAT no, seller tax ID, VAT registration number of exporter, tax identification number of exporter, exporter's tax number, sender tax number",
     "USt-IdNr. des Exporteurs, EORI-Nummer des Exporteurs, Steuernummer des Versenders, USt-IdNr. des Verkäufers, \
      numéro de TVA de l'exportateur, numéro EORI de l'exportateur, identifiant fiscal de l'expéditeur, numéro de TVA du vendeur, \
      NIF del exportador, número de IVA del exportador, número EORI del exportador, identificación fiscal del remitente, \
      partita IVA dell'esportatore, numero EORI dell'esportatore, codice fiscale dello speditore, partita IVA del venditore, \
      NIF do exportador, número de IVA do exportador, número EORI do exportador, CNPJ do remetente, \
      btw-nummer van de exporteur, EORI-nummer van de exporteur, fiscaal nummer van de afzender, btw-nummer van de verkoper, \
      DIČ vývozce, číslo EORI vývozce, daňové číslo odesílatele, DIČ prodávajícího, \
      الرقم الضريبي للمصدر, رقم EORI للمصدر, الرقم الضريبي للشاحن, الرقم الضريبي للبائع, \
      輸出者VAT番号, 輸出者EORI番号, 荷送人の税番号, 売主の税務番号, \
      出口商VAT号, 出口商EORI号, 发货人税号, 卖方税号, \
      수출자 VAT번호, 수출자 EORI번호, 송하인 세금번호, 매도인 사업자번호"),
    ("recipient_tax_number",
     "consignee VAT/EORI, consignee VAT number, consignee EORI number, consignee tax ID, importer VAT no, importer EORI number, buyer tax ID, buyer VAT number, VAT registration number of consignee, tax identification number of importer, buyer's tax number, receiver tax number",
     "USt-IdNr. des Empfängers, EORI-Nummer des Empfängers, Steuernummer des Importeurs, USt-IdNr. des Käufers, \
      numéro de TVA du destinataire, numéro EORI du destinataire, identifiant fiscal de l'importateur, numéro de TVA de l'acheteur, \
      NIF del consignatario, número de IVA del consignatario, número EORI del consignatario, identificación fiscal del importador, \
      partita IVA del destinatario, numero EORI del destinatario, codice fiscale dell'importatore, partita IVA dell'acquirente, \
      NIF do consignatário, número de IVA do consignatário, número EORI do consignatário, CNPJ do importador, \
      btw-nummer van de geadresseerde, EORI-nummer van de geadresseerde, fiscaal nummer van de importeur, btw-nummer van de koper, \
      DIČ příjemce, číslo EORI příjemce, daňové číslo dovozce, DIČ kupujícího, \
      الرقم الضريبي للمرسل إليه, رقم EORI للمرسل إليه, الرقم الضريبي للمستورد, الرقم الضريبي للمشتري, \
      荷受人VAT番号, 荷受人EORI番号, 輸入者の税番号, 買主の税務番号, \
      收货人VAT号, 收货人EORI号, 进口商税号, 买方税号, \
      수하인 VAT번호, 수하인 EORI번호, 수입자 세금번호, 매수인 사업자번호"),
    ("signatory_name",
     "signatory name, name of signatory, signed by, authorized signatory, name of authorized signatory, authorized signature, signature of exporter, signature of shipper, name and signature, printed name, name of the person signing",
     "Name des Unterzeichners, unterzeichnet von, bevollmächtigter Unterzeichner, Unterschrift des Exporteurs, Name und Unterschrift, \
      nom du signataire, signé par, signataire autorisé, signature de l'exportateur, nom et signature, \
      nombre del firmante, firmado por, firmante autorizado, firma del exportador, nombre y firma, \
      nome del firmatario, firmato da, firmatario autorizzato, firma dell'esportatore, nome e firma, \
      nome do signatário, assinado por, signatário autorizado, assinatura do exportador, nome e assinatura, \
      naam ondertekenaar, ondertekend door, gemachtigde ondertekenaar, handtekening exporteur, naam en handtekening, \
      jméno podepisujícího, podepsal, oprávněný podepisující, podpis vývozce, jméno a podpis, \
      اسم الموقع, وقعه, الموقع المفوض, توقيع المصدر, الاسم والتوقيع, \
      署名者名, 署名者, 権限を有する署名者, 輸出者の署名, 氏名と署名, \
      签署人姓名, 签署人, 授权签署人, 出口商签字, 姓名与签字, \
      서명자 성명, 서명인, 권한 있는 서명자, 수출자 서명, 성명 및 서명"),
    ("party_name",
     "signatory company, company name, name of company, company, firm name, legal name, business name, organization name, name of the firm, entity name, corporate name",
     "unterzeichnendes Unternehmen, Firmenname, Name der Firma, Firmenbezeichnung, Name der Organisation, \
      société signataire, raison sociale, nom de la société, dénomination sociale, nom de l'organisation, \
      empresa firmante, razón social, nombre de la empresa, denominación social, nombre de la organización, \
      società firmataria, ragione sociale, nome della società, denominazione sociale, nome dell'organizzazione, \
      empresa signatária, nome da empresa, denominação social, nome da organização, \
      ondertekenend bedrijf, bedrijfsnaam, naam van het bedrijf, firmanaam, naam van de organisatie, \
      podepisující společnost, název společnosti, název firmy, obchodní firma, název organizace, \
      الشركة الموقعة, اسم الشركة, اسم المؤسسة, الاسم التجاري, اسم المنظمة, \
      署名会社, 会社名, 企業名, 商号, 組織名, \
      签署公司, 公司名称, 企业名称, 商号, 机构名称, \
      서명 회사, 회사명, 기업명, 상호, 조직명"),
    ("issue_date",
     "date of issue, issue date, issued on, invoice date, date of invoice, document date, date of document, dated, date issued",
     "Ausstellungsdatum, ausgestellt am, Rechnungsdatum, Belegdatum, Datum des Dokuments, \
      date d'émission, émis le, date de la facture, date du document, date d'établissement, \
      fecha de emisión, emitido el, fecha de la factura, fecha del documento, fecha de expedición, \
      data di emissione, emesso il, data della fattura, data del documento, data di rilascio, \
      data de emissão, emitido em, data da fatura, data do documento, \
      datum van uitgifte, uitgegeven op, factuurdatum, documentdatum, datum van afgifte, \
      datum vystavení, vystaveno dne, datum faktury, datum dokladu, datum vydání, \
      تاريخ الإصدار, صدر في, تاريخ الفاتورة, تاريخ المستند, تاريخ التحرير, \
      発行日, 発行日付, 請求書日付, 書類日付, 作成日, \
      签发日期, 开具日期, 发票日期, 单据日期, 出具日期, \
      발행일, 발행일자, 송장일자, 문서일자, 작성일"),
    ("country_of_manufacture",
     "country of manufacture, country of origin, made in, manufactured in, origin, origin country, manufacturing country, produced in, C/O",
     "Herstellungsland, Ursprungsland, hergestellt in, Produktionsland, \
      pays de fabrication, pays d'origine, fabriqué en, pays de production, \
      país de fabricación, país de origen, fabricado en, país de producción, \
      paese di fabbricazione, paese di origine, fabbricato in, paese di produzione, \
      país de fabricação, país de origem, fabricado em, país de produção, \
      land van vervaardiging, land van oorsprong, vervaardigd in, productieland, \
      země výroby, země původu, vyrobeno v, země produkce, \
      بلد الصنع, بلد المنشأ, صنع في, بلد الإنتاج, \
      製造国, 原産国, 製造元, 生産国, \
      制造国, 原产国, 产地, 生产国, \
      제조국, 원산지, 제조지, 생산국, 제조된, 제조, 에서 제조된"),
    ("description",
     "description of goods, goods description, description, item description, commodity description, product description, name of goods, article description, description of merchandise",
     "Warenbeschreibung, Warenbezeichnung, Artikelbeschreibung, Bezeichnung der Ware, \
      description des marchandises, désignation des marchandises, description de l'article, libellé, \
      descripción de la mercancía, denominación de la mercancía, descripción del artículo, concepto, \
      descrizione delle merci, denominazione delle merci, descrizione dell'articolo, descrizione, \
      descrição das mercadorias, denominação da mercadoria, descrição do item, \
      goederenomschrijving, omschrijving van de goederen, artikelomschrijving, omschrijving, \
      popis zboží, označení zboží, popis položky, popis, \
      وصف البضاعة, بيان البضاعة, وصف الصنف, الوصف, \
      品名, 商品説明, 品目説明, 商品名, \
      货物描述, 货物名称, 品名, 商品描述, \
      품명, 상품 설명, 품목 설명, 물품명"),
    ("amount",
     "invoice total, total invoice amount, total invoice value, grand total, total amount, total amount due, amount due, total payable, invoice amount, total sum",
     "Rechnungsbetrag, Gesamtbetrag, Rechnungssumme, Endbetrag, zu zahlender Betrag, \
      montant total, total de la facture, montant de la facture, montant à payer, total général, \
      importe total, total de la factura, importe de la factura, importe a pagar, total general, \
      importo totale, totale fattura, importo della fattura, importo da pagare, totale generale, \
      valor total, total da fatura, valor da fatura, valor a pagar, \
      totaalbedrag, factuurbedrag, factuurtotaal, te betalen bedrag, eindtotaal, \
      celková částka, fakturovaná částka, částka k úhradě, celkem k úhradě, \
      المبلغ الإجمالي, إجمالي الفاتورة, قيمة الفاتورة, المبلغ المستحق, \
      合計金額, 請求金額, 総額, インボイス合計, お支払金額, \
      总金额, 发票总额, 合计金额, 应付金额, \
      총 금액, 총금액, 총액, 합계 금액, 송장 총액, 인보이스 총액, 청구 금액"),
    ("amount_subtotal",
     "subtotal, sub-total, sub total, amount before tax, total before tax, net amount before tax, taxable amount, taxable value, amount excluding tax",
     "Zwischensumme, Nettobetrag, Betrag ohne MwSt., Summe netto, \
      sous-total, montant HT, total hors taxes, montant hors taxes, \
      base imponible, importe sin IVA, importe neto, \
      subtotale, imponibile, importo senza IVA, totale imponibile, \
      valor sem impostos, base de cálculo, valor antes de impostos, \
      subtotaal, bedrag exclusief btw, totaal excl. btw, \
      mezisoučet, základ daně, částka bez DPH, \
      المجموع الفرعي, المبلغ قبل الضريبة, المبلغ الخاضع للضريبة, \
      小計, 税抜金額, 課税対象額, \
      小计, 不含税金额, 计税金额, \
      소계, 공급가액, 세전 금액"),
    ("amount_tax",
     "tax amount, VAT amount, amount of VAT, total VAT, total tax, sales tax amount, GST amount, tax total, value added tax amount",
     "Steuerbetrag, Mehrwertsteuerbetrag, MwSt.-Betrag, Umsatzsteuerbetrag, \
      montant de la TVA, montant de la taxe, total TVA, \
      importe del IVA, cuota de IVA, importe del impuesto, \
      importo IVA, importo dell'imposta, totale IVA, \
      valor do IVA, valor do imposto, \
      btw-bedrag, belastingbedrag, totaal btw, \
      částka DPH, výše daně, DPH celkem, \
      مبلغ الضريبة, مبلغ ضريبة القيمة المضافة, \
      消費税額, 税額, \
      税额, 增值税额, \
      세액, 부가세액, 부가가치세액"),
    ("freight_amount",
     "freight, freight charge, freight charges, freight cost, freight amount, ocean freight charge, air freight charge, shipping charge, carriage charge, transport charge",
     "Frachtkosten, Frachtbetrag, Fracht, Transportkosten, \
      frais de transport, montant du fret, fret, coût du transport, \
      importe del flete, flete, costo del transporte, \
      spese di trasporto, importo del nolo, nolo, costo del trasporto, \
      valor do frete, frete, custo do transporte, \
      vrachtbedrag, vracht, \
      dopravné, přepravné, náklady na dopravu, \
      أجرة الشحن, تكلفة الشحن, رسوم النقل, \
      運賃, 輸送費, 海上運賃, 航空運賃, \
      运费, 海运费, 空运费, 运输费用, \
      운임, 운송료, 해상운임, 항공운임"),
    ("unit_price",
     "unit price, unit value, price per unit, price each, unit cost, rate per unit, price per piece",
     "Einzelpreis, Stückpreis, Preis pro Einheit, \
      prix unitaire, valeur unitaire, prix à l'unité, \
      precio unitario, valor unitario, precio por unidad, \
      prezzo unitario, valore unitario, prezzo per unità, \
      preço unitário, valor unitário, preço por unidade, \
      eenheidsprijs, prijs per stuk, stukprijs, \
      jednotková cena, cena za kus, cena za jednotku, \
      سعر الوحدة, قيمة الوحدة, السعر الإفرادي, \
      単価, 単位価格, 1個あたりの価格, \
      单价, 单位价格, 每件价格, \
      단가, 개당 가격, 품목별 단가, 단위 가격"),
    ("total_price",
     "line total, line amount, line value, total price, extended price, extended amount, amount per line, row total",
     "Gesamtpreis, Positionsbetrag, Zeilensumme, Positionswert, \
      prix total, montant de la ligne, total de la ligne, valeur de la ligne, \
      precio total, importe de línea, total de línea, valor de la línea, \
      prezzo totale, importo riga, totale riga, valore della riga, \
      preço total, valor da linha, total da linha, \
      totaalprijs, regelbedrag, regeltotaal, \
      celková cena, cena za řádek, částka za položku, \
      السعر الإجمالي للبند, مبلغ البند, قيمة السطر, \
      明細金額, 品目別金額, 行金額, \
      总价, 行金额, 明细金额, \
      품목별 금액, 행 금액, 라인 금액, 품목 합계액"),
    ("quantity",
     "quantity, qty, number of units, quantity shipped, quantity ordered, item quantity, quantity per line",
     "Menge, Stückzahl, Anzahl der Einheiten, Liefermenge, Bestellmenge, \
      quantité, qté, nombre d'unités, quantité expédiée, quantité commandée, \
      cantidad, cant., número de unidades, cantidad enviada, cantidad pedida, \
      quantità, q.tà, numero di unità, quantità spedita, quantità ordinata, \
      quantidade, qtd., quantidade enviada, quantidade pedida, \
      hoeveelheid, aantal eenheden, geleverde hoeveelheid, bestelde hoeveelheid, \
      množství, počet jednotek, dodané množství, objednané množství, \
      الكمية, عدد الوحدات, الكمية المشحونة, الكمية المطلوبة, \
      数量, 個数, 出荷数量, 注文数量, \
      个数, 发货数量, 订购数量, \
      수량, 개수, 선적 수량, 주문 수량"),
    ("weight_gross",
     "gross weight, G.W., GW, gross wt., total gross weight, gross mass",
     "Bruttogewicht, Gesamtbruttogewicht, Brutto-Gewicht, \
      poids brut, poids brut total, \
      peso bruto, peso bruto total, \
      peso lordo, peso lordo totale, \
      brutogewicht, totaal brutogewicht, \
      hrubá hmotnost, celková hrubá hmotnost, \
      الوزن القائم, الوزن الإجمالي, إجمالي الوزن القائم, \
      総重量, グロス重量, \
      毛重, 总毛重, \
      총중량, 그로스 중량"),
    ("weight_net",
     "net weight, N.W., NW, net wt., total net weight, net mass",
     "Nettogewicht, Gesamtnettogewicht, Netto-Gewicht, \
      poids net, poids net total, \
      peso neto, peso neto total, \
      peso netto, peso netto totale, \
      peso líquido, peso líquido total, \
      nettogewicht, totaal nettogewicht, \
      čistá hmotnost, celková čistá hmotnost, \
      الوزن الصافي, إجمالي الوزن الصافي, \
      正味重量, 純重量, ネット重量, \
      净重, 总净重, \
      순중량, 순 중량"),
    ("package_count",
     "total number of packages, total packages, total no. of packages, total no. of pkgs, total cartons, total number of cartons, packages in total, total package count",
     "Gesamtzahl der Packstücke, Packstücke insgesamt, Gesamtanzahl der Kartons, \
      nombre total de colis, total des colis, nombre total de cartons, \
      número total de bultos, total de bultos, número total de cajas, \
      numero totale di colli, totale colli, numero totale di cartoni, \
      número total de volumes, total de volumes, número total de caixas, \
      totaal aantal colli, colli in totaal, totaal aantal dozen, \
      celkový počet balení, balení celkem, celkový počet kartonů, \
      إجمالي عدد الطرود, العدد الإجمالي للطرود, إجمالي عدد الكراتين, \
      総梱包数, 総個口数, 総カートン数, \
      总件数, 总箱数, 包装总件数, \
      총 포장수, 총 포장 개수, 총 카톤수, 총 박스수"),
    ("volume",
     "volume, measurement, total volume, total measurement, CBM, cubic meters, cubic metres, M3",
     "Volumen, Rauminhalt, Gesamtvolumen, Kubikmeter, \
      volume total, cubage, mètres cubes, \
      volumen total, cubicaje, metros cúbicos, \
      volume totale, cubatura, metri cubi, \
      cubagem, \
      inhoud, totaal volume, kubieke meter, \
      objem, celkový objem, kubatura, metry krychlové, \
      الحجم, الحجم الإجمالي, متر مكعب, \
      容積, 総容積, 才数, 立方メートル, \
      体积, 总体积, 尺码, 立方米, \
      용적, 총 용적, 부피, 입방미터"),
];

pub fn trade_label_supplement(field: &str) -> Vec<String> {
    let key = field.trim();
    for (f, en, ml) in TRADE_LABEL_SUPPLEMENT_ML.iter() {
        if *f == key {
            return anchor_phrases(en, ml);
        }
    }
    Vec::new()
}

pub fn trade_label_supplement_fields() -> Vec<&'static str> {
    TRADE_LABEL_SUPPLEMENT_ML.iter().map(|(f, _, _)| *f).collect()
}

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
            ("signatory_name",    "Person who signed the document",
             "signatory name, signed by, authorized signatory, signer"),
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
        "items" => vec![
            ("description",            "Description of goods on a line item",
             "description of goods, goods description, commodity, product name, article, item description, nature of goods"),
            ("country_of_manufacture", "Country where the line item was manufactured",
             "country of manufacture, country of origin, made in, manufactured in, origin"),
            ("unit_price",             "Unit price of a line item",
             "unit price, unit value, price per unit, rate per unit"),
            ("quantity",               "Quantity of a line item",
             "quantity, qty, pieces, number of units"),
            ("total_price",            "Line total of a line item",
             "total price, total value of the line, line total, line amount, extended amount"),
            ("item_code",              "Item code or model number of a line item",
             "item code, model number, article number, part number, SKU"),
            ("item_net_weight",        "Net weight per unit of a line item",
             "net weight per unit, unit weight, net weight per item"),
            ("item_gross_weight",      "Gross weight per unit of a line item",
             "gross weight per unit, gross weight per item"),
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
        | "fta_agreement_code" | "eccn" | "swift_code" | "account_number"
        | "sender_tax_number" | "recipient_tax_number" => "eq",

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

pub const VISION_CHROME_ANCHOR: &str =
    "company logo, brand emblem, letterhead graphic, official round stamp, red seal, \
     handwritten signature, watermark, blank paper, empty margin, page border, table grid lines, \
     ruled lines, barcode stripes, QR code square, page number footer, printed form template, \
     decorative frame, background texture, scanned paper noise, staple hole, punch hole";

pub const VISION_CHROME_ANCHOR_ML: &str =
    "Firmenlogo, Briefkopf, Firmenstempel, Dienstsiegel, handschriftliche Unterschrift, Wasserzeichen, leerer Rand, Seitenrahmen, Tabellenlinien, Strichcode, QR-Code, Seitenzahl in der Fußzeile, Formularvorlage, Hintergrundmuster, Scanrauschen, Lochung, \
     logo de l'entreprise, en-tête de lettre, cachet officiel, sceau rouge, signature manuscrite, filigrane, marge vide, bordure de page, lignes du tableau, code-barres, code QR, numéro de page en pied de page, modèle de formulaire, texture de fond, bruit de numérisation, perforation, \
     logotipo de la empresa, membrete, sello oficial, sello rojo, firma manuscrita, marca de agua, margen en blanco, borde de página, líneas de la tabla, código de barras, código QR, número de página en el pie, plantilla de formulario, textura de fondo, ruido de escaneo, perforación, \
     logo aziendale, carta intestata, timbro ufficiale, sigillo rosso, firma autografa, filigrana, margine vuoto, bordo della pagina, righe della tabella, codice a barre, codice QR, numero di pagina a piè di pagina, modello di modulo, trama di sfondo, rumore di scansione, foro di perforazione, \
     logotipo da empresa, papel timbrado, carimbo oficial, selo vermelho, assinatura manuscrita, marca d'água, margem em branco, borda da página, linhas da tabela, código de barras, código QR, número de página no rodapé, modelo de formulário, textura de fundo, ruído de digitalização, furo de perfuração, \
     bedrijfslogo, briefhoofd, officiële stempel, rood zegel, handgeschreven handtekening, watermerk, lege marge, paginarand, tabellijnen, streepjescode, QR-code, paginanummer in de voettekst, formuliersjabloon, achtergrondtextuur, scanruis, perforatiegat, \
     firemní logo, hlavičkový papír, úřední razítko, červená pečeť, vlastnoruční podpis, vodoznak, prázdný okraj, okraj stránky, linky tabulky, čárový kód, QR kód, číslo stránky v zápatí, šablona formuláře, textura pozadí, šum skenování, děrovaný otvor, \
     شعار الشركة, ترويسة الرسالة, ختم رسمي, ختم أحمر, توقيع بخط اليد, علامة مائية, هامش فارغ, حدود الصفحة, خطوط الجدول, الباركود, رمز الاستجابة السريعة, رقم الصفحة في التذييل, قالب النموذج, نسيج الخلفية, ضوضاء المسح, ثقب التخريم, \
     会社ロゴ, レターヘッド, 社印, 角印, 丸印, 手書き署名, 透かし, 余白, ページ枠, 罫線, バーコード, QRコード, フッターのページ番号, 帳票テンプレート, 背景の地紋, スキャンノイズ, パンチ穴, \
     公司标志, 信头, 公章, 红色印章, 手写签名, 水印, 空白页边, 页面边框, 表格线, 条形码, 二维码, 页脚页码, 表单模板, 背景底纹, 扫描噪点, 打孔, \
     회사 로고, 레터헤드, 직인, 관인, 붉은 도장, 손글씨 서명, 워터마크, 빈 여백, 페이지 테두리, 표 괘선, 바코드, QR 코드, 바닥글 페이지 번호, 서식 템플릿, 배경 무늬, 스캔 잡티, 펀치 구멍";

pub fn vision_chrome_phrases() -> Vec<String> {
    anchor_phrases(VISION_CHROME_ANCHOR, VISION_CHROME_ANCHOR_ML)
}

/// 🌟 [COMMERCE QUERY ANCHOR] 질의가 '상품·주문·배송 조회' 를 뜻하는지 판정하는 개념 뱅크.
///  서식 전문 뱅크(all_trade_doc_titles)의 상대 편입니다. 질의 쪽 MODE REROUTE 가
///  두 뱅크의 자기 분포 초과분을 비교해 커머스 유지 / 서식 전환을 정합니다.
pub const COMMERCE_QUERY_ANCHOR: &str = "product, item, goods, sale price, discount, brand, product detail, product listing, shopping cart, checkout, order status, order history, product review, rating, seller, shop, store, delivery status, parcel tracking, tracking number, shipping fee, coupon, stock, sold out, wishlist, category, size, color, option";

pub const COMMERCE_QUERY_ANCHOR_ML: &str = "Produkt, Artikel, Verkaufspreis, Rabatt, Marke, Produktseite, Warenkorb, Kasse, Bestellstatus, Bewertung, Verkäufer, Shop, Lieferstatus, Sendungsverfolgung, Versandkosten, Gutschein, Lagerbestand, ausverkauft, Wunschliste, Kategorie, Größe, Farbe, \
producto, artículo, precio de venta, descuento, marca, página de producto, carrito, pago, estado del pedido, reseña, vendedor, tienda, estado de entrega, seguimiento del paquete, gastos de envío, cupón, existencias, agotado, lista de deseos, categoría, talla, color, \
produit, article, prix de vente, remise, marque, fiche produit, panier, paiement, statut de la commande, avis client, vendeur, boutique, statut de livraison, suivi de colis, frais de port, code promo, stock, épuisé, liste de souhaits, catégorie, taille, couleur, \
商品, アイテム, 販売価格, 割引, ブランド, 商品ページ, カート, 決済, 注文状況, レビュー, 出品者, ショップ, 配送状況, 荷物追跡, 送料, クーポン, 在庫, 売り切れ, ほしい物リスト, カテゴリー, サイズ, カラー, \
produto, item, preço de venda, desconto, marca, página do produto, carrinho, finalizar compra, status do pedido, avaliação, vendedor, loja, status da entrega, rastreamento de encomenda, frete, cupom, estoque, esgotado, lista de desejos, categoria, tamanho, cor, \
منتج, سلعة, سعر البيع, خصم, علامة تجارية, صفحة المنتج, سلة التسوق, الدفع, حالة الطلب, تقييم, بائع, متجر, حالة التوصيل, تتبع الشحنة, رسوم الشحن, قسيمة, المخزون, نفدت الكمية, قائمة الرغبات, فئة, مقاس, لون, \
produkt, zboží, prodejní cena, sleva, značka, stránka produktu, nákupní košík, pokladna, stav objednávky, recenze, prodejce, obchod, stav doručení, sledování zásilky, poštovné, kupón, skladem, vyprodáno, seznam přání, kategorie, velikost, barva, \
prodotto, articolo, prezzo di vendita, sconto, marca, scheda prodotto, carrello, cassa, stato dell'ordine, recensione, venditore, negozio, stato della consegna, tracciamento del pacco, spese di spedizione, coupon, disponibilità, esaurito, lista dei desideri, categoria, taglia, colore, \
상품, 제품, 판매가, 할인, 브랜드, 상품 상세, 장바구니, 결제, 주문 상태, 상품평, 리뷰, 판매자, 쇼핑몰, 배송 상태, 택배 조회, 배송비, 쿠폰, 재고, 품절, 위시리스트, 카테고리, 사이즈, 색상, \
product, artikel, verkoopprijs, korting, merk, productpagina, winkelwagen, afrekenen, bestelstatus, beoordeling, verkoper, webshop, bezorgstatus, pakket volgen, verzendkosten, kortingscode, voorraad, uitverkocht, verlanglijst, categorie, maat, kleur, \
商品, 货品, 售价, 折扣, 品牌, 商品详情, 购物车, 结算, 订单状态, 评价, 卖家, 店铺, 配送状态, 快递查询, 运费, 优惠券, 库存, 售罄, 心愿单, 分类, 尺码, 颜色";

/// 🌟 [SITE CHROME / 12 LANGUAGES] 사이트 껍데기 문장 — 내비게이션·헤더·푸터·관리자 메뉴·로그인·공지·저작권.
///  키는 언어 코드, 값은 쉼표로 이은 구입니다. scheduler.rs 의 prejudice_pair 가 영어 편견 문장 옆에
///  문서 언어 문장 하나를 더 세워 두 벡터 중 최댓값으로 잽니다. 영어는 scheduler.rs 의 기존 문장을 그대로 씁니다.
pub const SITE_CHROME_ML: &[(&str, &str)] = &[
    ("de", "globale Navigation, Menü, Hauptmenü, Kopfzeile, Fußzeile, Seitenleiste, Brotkrümelnavigation, Suchformular, Suchfilter, Seitennummerierung, Admin-Menü, Verwaltungsmenü, Schnellmenü, Untermenü, Kategoriemenü, Einstellungsmenü, Anmelden, Abmelden, Einstellungen, Meine Seite, Hinweis, Banner, Copyright, Dashboard, Verwaltungsseite, Administratorseite, Startseite, Willkommen, Seitenname"),
    ("es", "navegación global, menú, menú principal, encabezado, pie de página, barra lateral, ruta de navegación, formulario de búsqueda, filtro de búsqueda, paginación, menú de administración, menú rápido, submenú, menú de categorías, menú de configuración, iniciar sesión, cerrar sesión, configuración, mi página, aviso, banner, derechos de autor, panel de control, página de administración, página del administrador, inicio, bienvenido, nombre del sitio"),
    ("fr", "navigation globale, menu, menu principal, en-tête, pied de page, barre latérale, fil d'Ariane, formulaire de recherche, filtre de recherche, pagination, menu d'administration, menu rapide, sous-menu, menu des catégories, menu des paramètres, connexion, déconnexion, paramètres, mon compte, avis, bannière, droits d'auteur, tableau de bord, page d'administration, page administrateur, accueil, bienvenue, nom du site"),
    ("ja", "グローバルナビゲーション, メニュー, メインメニュー, ヘッダー, フッター, サイドバー, パンくずリスト, 検索フォーム, 検索フィルター, ページネーション, 管理メニュー, クイックメニュー, サブメニュー, カテゴリーメニュー, 設定メニュー, ログイン, ログアウト, 設定, マイページ, お知らせ, バナー, 著作権, ダッシュボード, 管理画面, 管理者ページ, ホーム, ようこそ, サイト名"),
    ("pt", "navegação global, menu, menu principal, cabeçalho, rodapé, barra lateral, trilha de navegação, formulário de pesquisa, filtro de pesquisa, paginação, menu de administração, menu rápido, submenu, menu de categorias, menu de configurações, entrar, sair, configurações, minha página, aviso, banner, direitos autorais, painel de controle, página de administração, página do administrador, início, bem-vindo, nome do site"),
    ("ar", "التنقل العام, القائمة, القائمة الرئيسية, الترويسة, التذييل, الشريط الجانبي, مسار التنقل, نموذج البحث, مرشح البحث, ترقيم الصفحات, قائمة الإدارة, القائمة السريعة, القائمة الفرعية, قائمة الفئات, قائمة الإعدادات, تسجيل الدخول, تسجيل الخروج, الإعدادات, صفحتي, إشعار, لافتة, حقوق النشر, لوحة التحكم, صفحة الإدارة, صفحة المسؤول, الصفحة الرئيسية, مرحبا, اسم الموقع"),
    ("cs", "globální navigace, menu, hlavní menu, záhlaví, zápatí, postranní panel, drobečková navigace, vyhledávací formulář, filtr vyhledávání, stránkování, administrační menu, rychlé menu, podmenu, menu kategorií, menu nastavení, přihlásit, odhlásit, nastavení, moje stránka, oznámení, banner, autorská práva, nástěnka, administrační stránka, stránka správce, domů, vítejte, název webu"),
    ("it", "navigazione globale, menu, menu principale, intestazione, piè di pagina, barra laterale, breadcrumb, modulo di ricerca, filtro di ricerca, paginazione, menu di amministrazione, menu rapido, sottomenu, menu categorie, menu impostazioni, accedi, esci, impostazioni, la mia pagina, avviso, banner, copyright, pannello di controllo, pagina di amministrazione, pagina amministratore, home, benvenuto, nome del sito"),
    ("ko", "전체 메뉴, 메뉴, 메인 메뉴, 헤더, 푸터, 사이드바, 경로 탐색, 검색 폼, 검색 필터, 페이지 이동, 관리자 메뉴, 퀵 메뉴, 서브 메뉴, 카테고리 메뉴, 설정 메뉴, 로그인, 로그아웃, 설정, 마이페이지, 공지사항, 배너, 저작권, 대시보드, 관리자 페이지, 관리 페이지, 홈, 환영합니다, 사이트명"),
    ("nl", "globale navigatie, menu, hoofdmenu, koptekst, voettekst, zijbalk, kruimelpad, zoekformulier, zoekfilter, paginering, beheermenu, snelmenu, submenu, categoriemenu, instellingenmenu, inloggen, uitloggen, instellingen, mijn pagina, mededeling, banner, auteursrecht, dashboard, beheerpagina, beheerderspagina, home, welkom, sitenaam"),
    ("zh", "全局导航, 菜单, 主菜单, 页眉, 页脚, 侧边栏, 面包屑导航, 搜索表单, 搜索筛选, 分页, 管理菜单, 快捷菜单, 子菜单, 分类菜单, 设置菜单, 登录, 退出登录, 设置, 我的页面, 公告, 横幅, 版权, 仪表盘, 管理页面, 管理员页面, 首页, 欢迎, 站点名称"),
];

/// 문서 언어의 사이트 껍데기 문장. 영어이거나 표에 없는 언어면 None 입니다.
pub fn site_chrome_sentence(lang: &str) -> Option<String> {
    let key = lang.trim().to_lowercase();
    let short = key.split(|c: char| c == '-' || c == '_').next().unwrap_or("");
    if short.is_empty() || short == "en" { return None; }
    SITE_CHROME_ML
        .iter()
        .find(|(l, _)| *l == short)
        .map(|(_, s)| s.to_string())
}

pub const UI_ACTION_ANCHOR: &str =
    "edit button, modify, update, delete, remove, copy, duplicate, register, add new, \
     save, cancel, confirm, submit, apply, reset, search button, view detail, go to detail, \
     open detail page, more, expand, manage, management, administration, row action, \
     action column, link button, print, download, export to excel, select all checkbox, \
     move, sort order input, quick edit, preview, share, send sms, send email";

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

pub const TRADE_TITLE_LABEL_ANCHOR: &str = "document type, kind of document, type of form, \
     name of this document, title of this document, document name, form name, \
     document code, form code, classification of this document";

pub const TRADE_REFERENCE_LABEL_ANCHOR: &str = "referenced document number, related document number, \
     reference number of another document, master document number, associated document, \
     payment terms, terms of payment, drawn under credit, issued under, \
     attached documents, required documents, enclosed documents, remark, note";

pub const TRADE_ITEM_ATTRIBUTE_ANCHOR: &str = "line item attribute, attribute of one product row, \
     item code, stock keeping unit, article number, product description, \
     quantity, unit of measure, unit price, line total, amount of this row, \
     table column header, row number in a list, subtotal, discount, total quantity";

pub const TRADE_ROW_MARKER_ANCHOR: &str = "row separator, table row marker, line item index, \
     item number in a list, section key, group key, metadata key, \
     continued from previous page, page break marker, list bullet";

pub const SITE_CHROME_ANCHOR: &str = "site name, shopping mall name, brand slogan, \
     administrator page, admin home, admin main menu, management menu, \
     dashboard, control panel, back office, console, \
     global navigation bar, breadcrumb, sidebar menu, footer, copyright notice, banner, \
     login, logout, sign in, sign out, my page, member management, \
     settings, configuration, preferences, \
     visitor counter, today visitors, yesterday visitors, total visitors, \
     software version number, welcome message, home, index page, \
     search form, filter form, page navigation, pagination";

pub fn trade_title_label_phrases() -> Vec<String> {
    anchor_phrases(TRADE_TITLE_LABEL_ANCHOR, TRADE_TITLE_LABEL_ANCHOR_ML)
}
pub fn trade_reference_label_phrases() -> Vec<String> {
    anchor_phrases(TRADE_REFERENCE_LABEL_ANCHOR, TRADE_REFERENCE_LABEL_ANCHOR_ML)
}
pub fn trade_item_attribute_phrases() -> Vec<String> {
    anchor_phrases(TRADE_ITEM_ATTRIBUTE_ANCHOR, TRADE_ITEM_ATTRIBUTE_ANCHOR_ML)
}
pub fn trade_row_marker_phrases() -> Vec<String> {
    anchor_phrases(TRADE_ROW_MARKER_ANCHOR, TRADE_ROW_MARKER_ANCHOR_ML)
}
pub fn site_chrome_phrases() -> Vec<String> {
    anchor_phrases(SITE_CHROME_ANCHOR, SITE_CHROME_ANCHOR_ML)
}
pub fn ui_action_phrases() -> Vec<String> {
    anchor_phrases(UI_ACTION_ANCHOR, UI_ACTION_ANCHOR_ML)
}

pub const TRADE_TITLE_LABEL_ANCHOR_ML: &str =
    "Dokumentart, Dokumenttyp, Art des Dokuments, Titel dieses Dokuments, Formularname, Dokumentcode, \
     type de document, nature du document, titre de ce document, nom du formulaire, code du document, \
     tipo de documento, clase de documento, título de este documento, nombre del formulario, código del documento, \
     tipo di documento, titolo di questo documento, nome del modulo, codice documento, \
     título deste documento, nome do formulário, código do documento, \
     documentsoort, documenttype, titel van dit document, formuliernaam, documentcode, \
     typ dokumentu, druh dokumentu, název tohoto dokumentu, název formuláře, kód dokumentu, \
     نوع المستند, صنف المستند, عنوان هذا المستند, اسم النموذج, رمز المستند, \
     書類の種類, 書類種別, この書類の名称, 様式名, 書類コード, \
     单据类型, 文件种类, 本单据名称, 表格名称, 单据代码, \
     서류 종류, 문서 유형, 이 문서의 제목, 양식명, 서류 코드";

pub const TRADE_REFERENCE_LABEL_ANCHOR_ML: &str =
    "referenzierte Dokumentnummer, zugehörige Dokumentnummer, Nummer eines anderen Dokuments, Hauptdokumentnummer, Zahlungsbedingungen, ausgestellt unter, beigefügte Unterlagen, Bemerkung, \
     numéro de document référencé, numéro de document associé, numéro d'un autre document, numéro du document principal, conditions de paiement, émis en vertu de, documents joints, remarque, \
     número de documento referenciado, número de documento relacionado, número de otro documento, número del documento principal, condiciones de pago, emitido bajo, documentos adjuntos, observación, \
     numero documento di riferimento, numero documento correlato, numero di un altro documento, numero documento principale, termini di pagamento, emesso ai sensi di, documenti allegati, osservazione, \
     número de outro documento, número do documento principal, condições de pagamento, emitido sob, documentos anexos, observação, \
     gerefereerd documentnummer, gerelateerd documentnummer, nummer van een ander document, hoofddocumentnummer, betalingsvoorwaarden, uitgegeven onder, bijgevoegde documenten, opmerking, \
     odkazované číslo dokumentu, číslo souvisejícího dokumentu, číslo jiného dokumentu, číslo hlavního dokumentu, platební podmínky, vystaveno na základě, přiložené doklady, poznámka, \
     رقم المستند المرجعي, رقم المستند المرتبط, رقم مستند آخر, رقم المستند الرئيسي, شروط الدفع, صادر بموجب, المستندات المرفقة, ملاحظة, \
     参照書類番号, 関連書類番号, 他の書類の番号, 主書類番号, 支払条件, に基づき発行, 添付書類, 備考, \
     参考单据号, 相关单据号, 其他单据编号, 主单据号, 付款条件, 依据签发, 附件, 备注, \
     참조 문서번호, 관련 문서번호, 다른 문서의 번호, 주 문서번호, 결제조건, 근거 발행, 첨부서류, 비고";

pub const TRADE_SELF_ID_LABEL_ANCHOR_ML: &str =
    "Dokumentnummer, Nummer dieses Dokuments, unter dem Titel gedruckte Nummer, Rechnungsnummer, Bestellnummer, Zertifikatsnummer, Anmeldenummer, Konnossementnummer, Frachtbriefnummer, Policennummer, Buchungsnummer, \
     numéro du document, numéro de ce document, numéro imprimé sous le titre, numéro de facture, numéro de commande, numéro de certificat, numéro de déclaration, numéro de connaissement, numéro de lettre de transport, numéro de police, numéro de réservation, \
     número del documento, número de este documento, número impreso bajo el título, número de factura, número de pedido, número de certificado, número de declaración, número de conocimiento de embarque, número de carta de porte, número de póliza, número de reserva, \
     numero del documento, numero di questo documento, numero stampato sotto il titolo, numero fattura, numero ordine, numero certificato, numero dichiarazione, numero polizza di carico, numero lettera di vettura, numero polizza, numero prenotazione, \
     número do documento, número deste documento, número impresso sob o título, número da fatura, número do pedido, número do certificado, número da declaração, número do conhecimento de embarque, número da carta de porte, número da apólice, número da reserva, \
     documentnummer, nummer van dit document, nummer onder de titel, factuurnummer, ordernummer, certificaatnummer, aangiftenummer, cognossementnummer, vrachtbriefnummer, polisnummer, boekingsnummer, \
     číslo dokumentu, číslo tohoto dokumentu, číslo vytištěné pod názvem, číslo faktury, číslo objednávky, číslo certifikátu, číslo prohlášení, číslo konosamentu, číslo nákladního listu, číslo pojistky, číslo rezervace, \
     رقم المستند, رقم هذا المستند, الرقم المطبوع تحت العنوان, رقم الفاتورة, رقم الطلب, رقم الشهادة, رقم الإقرار, رقم بوليصة الشحن, رقم وثيقة النقل, رقم وثيقة التأمين, رقم الحجز, \
     書類番号, この書類の番号, 表題の下に印字された番号, 請求書番号, 注文番号, 証明書番号, 申告番号, 船荷証券番号, 運送状番号, 保険証券番号, ブッキング番号, \
     单据编号, 本单据编号, 标题下方印制的编号, 发票号, 订单号, 证书编号, 申报编号, 提单号, 运单号, 保单号, 订舱号, \
     문서번호, 이 문서의 번호, 제목 아래 인쇄된 번호, 송장번호, 주문번호, 증명서번호, 신고번호, 선하증권번호, 운송장번호, 보험증권번호, 부킹번호";

pub const TRADE_ITEM_ATTRIBUTE_ANCHOR_ML: &str =
    "Artikelattribut, Artikelnummer, Warenbeschreibung, Menge, Maßeinheit, Stückpreis, Positionssumme, Spaltenüberschrift, Zwischensumme, Gesamtmenge, \
     attribut d'article, référence article, description des marchandises, quantité, unité de mesure, prix unitaire, total de ligne, en-tête de colonne, sous-total, quantité totale, \
     atributo de artículo, código de artículo, descripción de la mercancía, cantidad, unidad de medida, precio unitario, total de línea, encabezado de columna, subtotal, cantidad total, \
     attributo articolo, codice articolo, descrizione merce, quantità, unità di misura, prezzo unitario, totale riga, intestazione colonna, subtotale, quantità totale, \
     atributo do item, código do item, descrição da mercadoria, quantidade, unidade de medida, preço unitário, total da linha, cabeçalho da coluna, quantidade total, \
     artikelattribuut, artikelnummer, goederenomschrijving, hoeveelheid, meeteenheid, eenheidsprijs, regeltotaal, kolomkop, subtotaal, totale hoeveelheid, \
     atribut položky, kód položky, popis zboží, množství, měrná jednotka, jednotková cena, celkem za řádek, záhlaví sloupce, mezisoučet, celkové množství, \
     خاصية الصنف, رمز الصنف, وصف البضاعة, الكمية, وحدة القياس, سعر الوحدة, إجمالي البند, عنوان العمود, المجموع الفرعي, الكمية الإجمالية, \
     品目属性, 品番, 品名, 数量, 単位, 単価, 明細合計, 列見出し, 小計, 合計数量, \
     货号, 货物描述, 计量单位, 单价, 行合计, 列标题, 总数量, \
     품목 속성, 품목코드, 품명, 수량, 단위, 단가, 라인합계, 열 머리글, 소계, 총수량";

pub const TRADE_ROW_MARKER_ANCHOR_ML: &str =
    "Zeilentrenner, Positionsnummer in einer Liste, Abschnittsschlüssel, Fortsetzung von vorheriger Seite, Seitenumbruch, Aufzählungszeichen, \
     séparateur de ligne, numéro d'article dans une liste, clé de section, suite de la page précédente, saut de page, puce de liste, \
     separador de fila, número de artículo en una lista, clave de sección, continuación de la página anterior, salto de página, viñeta de lista, \
     separatore di riga, numero articolo in un elenco, chiave di sezione, continua dalla pagina precedente, interruzione di pagina, punto elenco, \
     separador de linha, número do item em uma lista, chave de seção, continuação da página anterior, quebra de página, marcador de lista, \
     rijscheiding, itemnummer in een lijst, sectiesleutel, vervolg van vorige pagina, pagina-einde, opsommingsteken, \
     oddělovač řádků, číslo položky v seznamu, klíč sekce, pokračování z předchozí strany, konec stránky, odrážka, \
     فاصل الصفوف, رقم البند في القائمة, مفتاح القسم, تابع من الصفحة السابقة, فاصل الصفحة, نقطة تعداد, \
     行区切り, リスト内の項目番号, セクションキー, 前ページからの続き, 改ページ, 箇条書き記号, \
     行分隔符, 列表中的项目编号, 章节键, 接上页, 分页符, 项目符号, \
     행 구분자, 목록 내 항목 번호, 섹션 키, 이전 페이지에서 계속, 페이지 나눔, 글머리 기호";

pub const SITE_CHROME_ANCHOR_ML: &str =
    "Website-Name, Shopname, Markenslogan, Administratorseite, Verwaltungsmenü, Dashboard, Navigationsleiste, Fußzeile, Anmelden, Abmelden, Einstellungen, Besucherzähler, Willkommensnachricht, Suchformular, Seitennummerierung, \
     nom du site, nom de la boutique, slogan de la marque, page d'administration, menu de gestion, tableau de bord, barre de navigation, pied de page, connexion, déconnexion, paramètres, compteur de visiteurs, message de bienvenue, formulaire de recherche, pagination, \
     nombre del sitio, nombre de la tienda, eslogan de la marca, página de administración, menú de gestión, panel de control, barra de navegación, pie de página, iniciar sesión, cerrar sesión, ajustes, contador de visitantes, mensaje de bienvenida, formulario de búsqueda, paginación, \
     nome del sito, nome del negozio, slogan del marchio, pagina di amministrazione, menu di gestione, cruscotto, barra di navigazione, piè di pagina, accedi, esci, impostazioni, contatore visitatori, messaggio di benvenuto, modulo di ricerca, paginazione, \
     nome do site, nome da loja, slogan da marca, página do administrador, menu de gestão, painel de controle, barra de navegação, rodapé, entrar, sair, configurações, mensagem de boas-vindas, formulário de pesquisa, \
     sitenaam, winkelnaam, merkslogan, beheerpagina, beheermenu, navigatiebalk, voettekst, inloggen, uitloggen, instellingen, bezoekersteller, welkomstbericht, zoekformulier, paginering, \
     název webu, název obchodu, slogan značky, stránka správce, menu správy, přehled, navigační lišta, zápatí, přihlásit se, odhlásit se, nastavení, počítadlo návštěv, uvítací zpráva, vyhledávací formulář, stránkování, \
     اسم الموقع, اسم المتجر, شعار العلامة التجارية, صفحة المسؤول, قائمة الإدارة, لوحة التحكم, شريط التنقل, التذييل, تسجيل الدخول, تسجيل الخروج, الإعدادات, عداد الزوار, رسالة ترحيب, نموذج البحث, ترقيم الصفحات, \
     サイト名, ショップ名, ブランドスローガン, 管理者ページ, 管理メニュー, ダッシュボード, ナビゲーションバー, フッター, ログイン, ログアウト, 設定, 訪問者カウンター, ようこそメッセージ, 検索フォーム, ページ送り, \
     网站名称, 商城名称, 品牌口号, 管理员页面, 管理菜单, 仪表盘, 导航栏, 页脚, 登录, 退出登录, 设置, 访客计数器, 欢迎信息, 搜索表单, 分页, \
     사이트명, 쇼핑몰명, 브랜드 슬로건, 관리자 페이지, 관리 메뉴, 대시보드, 내비게이션 바, 푸터, 로그인, 로그아웃, 설정, 방문자 카운터, 환영 메시지, 검색 폼, 페이지네이션";

pub const UI_ACTION_ANCHOR_ML: &str = "löschen, bearbeiten, ändern, Details anzeigen, mehr anzeigen, auswählen, bestätigen, abbrechen, speichern, anwenden, herunterladen, drucken, kopieren, teilen, in den Warenkorb, jetzt kaufen, bestellen, zur Kasse, schließen, öffnen, erweitern, einklappen, zurück, weiter, suchen, zurücksetzen, senden, hochladen, registrieren, \
eliminar, editar, modificar, ver detalles, ver más, seleccionar, confirmar, cancelar, guardar, aplicar, descargar, imprimir, copiar, compartir, añadir al carrito, comprar ahora, pedir, pagar, cerrar, abrir, expandir, contraer, anterior, siguiente, buscar, restablecer, enviar, subir, registrarse, \
supprimer, modifier, éditer, voir les détails, voir plus, sélectionner, confirmer, annuler, enregistrer, appliquer, télécharger, imprimer, copier, partager, ajouter au panier, acheter maintenant, commander, payer, fermer, ouvrir, développer, réduire, précédent, suivant, rechercher, réinitialiser, envoyer, téléverser, s'inscrire, \
削除, 編集, 修正, 詳細を見る, もっと見る, 選択, 確認, キャンセル, 保存, 適用, ダウンロード, 印刷, コピー, 共有, カートに入れる, 今すぐ購入, 注文する, 購入手続き, 閉じる, 開く, 展開, 折りたたむ, 前へ, 次へ, 検索, リセット, 送信, アップロード, 会員登録, \
excluir, editar, modificar, ver detalhes, ver mais, selecionar, confirmar, cancelar, salvar, aplicar, baixar, imprimir, copiar, compartilhar, adicionar ao carrinho, comprar agora, pedir, finalizar compra, fechar, abrir, expandir, recolher, anterior, próximo, buscar, redefinir, enviar, carregar, cadastrar, \
حذف, تعديل, تحرير, عرض التفاصيل, عرض المزيد, اختيار, تأكيد, إلغاء, حفظ, تطبيق, تنزيل, طباعة, نسخ, مشاركة, أضف إلى السلة, اشتر الآن, اطلب, الدفع, إغلاق, فتح, توسيع, طي, السابق, التالي, بحث, إعادة تعيين, إرسال, رفع, تسجيل, \
smazat, upravit, změnit, zobrazit detail, zobrazit více, vybrat, potvrdit, zrušit, uložit, použít, stáhnout, tisk, kopírovat, sdílet, přidat do košíku, koupit nyní, objednat, pokladna, zavřít, otevřít, rozbalit, sbalit, předchozí, další, hledat, obnovit, odeslat, nahrát, registrovat, \
elimina, modifica, aggiorna, vedi dettagli, mostra altro, seleziona, conferma, annulla, salva, applica, scarica, stampa, copia, condividi, aggiungi al carrello, acquista ora, ordina, procedi all'acquisto, chiudi, apri, espandi, comprimi, precedente, successivo, cerca, reimposta, invia, carica, registrati, \
삭제, 수정, 변경, 상세보기, 더보기, 선택, 확인, 취소, 저장, 적용, 다운로드, 인쇄, 복사, 공유, 장바구니 담기, 바로 구매, 주문하기, 결제하기, 닫기, 열기, 펼치기, 접기, 이전, 다음, 검색, 초기화, 전송, 업로드, 회원가입, \
verwijderen, bewerken, wijzigen, details bekijken, meer bekijken, selecteren, bevestigen, annuleren, opslaan, toepassen, downloaden, afdrukken, kopiëren, delen, in winkelwagen, nu kopen, bestellen, afrekenen, sluiten, openen, uitklappen, inklappen, vorige, volgende, zoeken, resetten, verzenden, uploaden, registreren, \
删除, 编辑, 修改, 查看详情, 查看更多, 选择, 确认, 取消, 保存, 应用, 下载, 打印, 复制, 分享, 加入购物车, 立即购买, 下单, 结算, 关闭, 打开, 展开, 收起, 上一页, 下一页, 搜索, 重置, 提交, 上传, 注册";

pub const DECLARATION_BOILERPLATE_ANCHOR: &str =
    "I declare all the information contained in this invoice to be true and correct, \
     I hereby certify that the above information is true and accurate, \
     we certify that this invoice is true and correct, the undersigned declares, \
     declaration of exporter, signed under penalty of perjury, \
     to the best of my knowledge and belief, we hereby confirm the accuracy of the above, attestation statement";

pub const DECLARATION_BOILERPLATE_ANCHOR_ML: &str =
    "Ich erkläre dass alle Angaben in dieser Rechnung wahr und richtig sind, Hiermit bestätigen wir die Richtigkeit der obigen Angaben, Erklärung des Ausführers, der Unterzeichner erklärt, nach bestem Wissen und Gewissen, \
     Je déclare que toutes les informations contenues dans cette facture sont exactes et véridiques, Nous certifions que cette facture est sincère et véritable, déclaration de l'exportateur, le soussigné déclare, en toute connaissance de cause, \
     Declaro que toda la información contenida en esta factura es verdadera y correcta, Certificamos que esta factura es verdadera y correcta, declaración del exportador, el abajo firmante declara, según mi leal saber y entender, \
     Dichiaro che tutte le informazioni contenute in questa fattura sono veritiere e corrette, Si certifica che la presente fattura è veritiera e corretta, dichiarazione dell'esportatore, il sottoscritto dichiara, per quanto a mia conoscenza, \
     Declaro que todas as informações contidas nesta fatura são verdadeiras e corretas, Certificamos que esta fatura é verdadeira e correta, declaração do exportador, o abaixo assinado declara, tanto quanto é do meu conhecimento, \
     Ik verklaar dat alle informatie in deze factuur waar en juist is, Wij verklaren dat deze factuur juist en volledig is, verklaring van de exporteur, ondergetekende verklaart, naar beste weten, \
     Prohlašuji že všechny údaje uvedené v této faktuře jsou pravdivé a správné, Potvrzujeme že tato faktura je pravdivá a správná, prohlášení vývozce, níže podepsaný prohlašuje, podle mého nejlepšího vědomí, \
     أقر بأن جميع المعلومات الواردة في هذه الفاتورة صحيحة وسليمة, نشهد بأن هذه الفاتورة صحيحة ودقيقة, إقرار المصدر, يقر الموقع أدناه, على حد علمي, \
     本請求書に記載された全ての情報が真実かつ正確であることを宣言します, 上記の内容が正確であることを証明します, 輸出者の申告, 署名者は以下のとおり宣言する, 私の知る限りにおいて, \
     本人声明本发票所载全部信息真实无误, 兹证明本发票内容真实准确, 出口商声明, 签署人特此声明, 据本人所知, \
     본 송장에 기재된 모든 정보가 사실이며 정확함을 선언합니다, 상기 내용이 사실임을 증명합니다, 수출자 신고서, 서명인은 다음과 같이 선언합니다, 본인이 아는 한";

pub const HANDLING_INSTRUCTION_ANCHOR: &str =
    "handle with care, fragile, this side up, keep dry, keep away from heat, \
     do not stack, do not drop, do not freeze, protect from moisture, \
     store in a cool dry place, keep refrigerated, temperature controlled, \
     partial shipment not allowed, transshipment not allowed, deliver before, \
     notify party on arrival, lift here, use no hooks, stack no more than";

pub const HANDLING_INSTRUCTION_ANCHOR_ML: &str =
    "Vorsicht zerbrechlich, oben, trocken halten, vor Hitze schützen, nicht stapeln, nicht stürzen, kühl und trocken lagern, vor Nässe schützen, Teillieferung nicht erlaubt, Umladung nicht erlaubt, \
     fragile manipuler avec soin, haut, tenir au sec, craint la chaleur, ne pas gerber, ne pas jeter, stocker au frais et au sec, protéger de l'humidité, expédition partielle non autorisée, transbordement non autorisé, \
     frágil manéjese con cuidado, este lado arriba, mantener seco, proteger del calor, no apilar, no dejar caer, almacenar en lugar fresco y seco, proteger de la humedad, embarque parcial no permitido, transbordo no permitido, \
     fragile maneggiare con cura, alto, tenere all'asciutto, proteggere dal calore, non impilare, non lasciare cadere, conservare in luogo fresco e asciutto, spedizione parziale non consentita, trasbordo non consentito, \
     frágil manuseie com cuidado, este lado para cima, manter seco, proteger do calor, não empilhar, não deixar cair, armazenar em local fresco e seco, embarque parcial não permitido, transbordo não permitido, \
     breekbaar voorzichtig behandelen, deze zijde boven, droog houden, uit de buurt van hitte houden, niet stapelen, niet laten vallen, koel en droog bewaren, deelzending niet toegestaan, overslag niet toegestaan, \
     křehké opatrně manipulovat, touto stranou nahoru, udržujte v suchu, chraňte před teplem, nestohovat, neházet, skladujte v chladu a suchu, částečná dodávka není povolena, překládka není povolena, \
     قابل للكسر يرجى الحذر, هذا الجانب لأعلى, يحفظ جافا, يحفظ بعيدا عن الحرارة, ممنوع التكديس, لا تسقط, يخزن في مكان بارد وجاف, الشحن الجزئي غير مسموح, إعادة الشحن غير مسموح, \
     取扱注意, われもの注意, 天地無用, 水濡れ厳禁, 直射日光を避ける, 積み重ね禁止, 落下厳禁, 冷暗所保管, 分割積み不可, 積み替え不可, \
     小心轻放, 易碎品, 此面向上, 保持干燥, 避免受热, 禁止堆叠, 严禁摔落, 阴凉干燥处存放, 不允许分批装运, 不允许转运, \
     취급주의, 파손주의, 천지무용, 습기엄금, 직사광선 피함, 적재금지, 낙하엄금, 서늘하고 건조한 곳 보관, 분할선적 불가, 환적 불가";

pub const TRADE_DOC_TITLES_ML: &[(&str, &str)] = &[
    ("BL", "Konnossement"), ("BL", "conocimiento de embarque"), ("BL", "connaissement"),
    ("BL", "polizza di carico"), ("BL", "conhecimento de embarque"), ("BL", "cognossement"),
    ("BL", "konosament"), ("BL", "بوليصة الشحن"), ("BL", "선하증권"), ("BL", "船荷証券"),
    ("BL", "提单"), ("BL", "提單"),
    ("AWB", "Luftfrachtbrief"), ("AWB", "guía aérea"), ("AWB", "lettre de transport aérien"),
    ("AWB", "lettera di vettura aerea"), ("AWB", "conhecimento aéreo"), ("AWB", "luchtvrachtbrief"),
    ("AWB", "letecký nákladní list"), ("AWB", "بوليصة الشحن الجوي"), ("AWB", "항공화물운송장"),
    ("AWB", "航空貨物運送状"), ("AWB", "航空运单"),
    ("CI", "Handelsrechnung"), ("CI", "factura comercial"), ("CI", "facture commerciale"),
    ("CI", "fattura commerciale"), ("CI", "fatura comercial"), ("CI", "handelsfactuur"),
    ("CI", "obchodní faktura"), ("CI", "فاتورة تجارية"), ("CI", "상업송장"), ("CI", "커머셜 인보이스"),
    ("CI", "商業送り状"), ("CI", "コマーシャルインボイス"), ("CI", "商业发票"),
    ("PL", "Packliste"), ("PL", "lista de empaque"), ("PL", "liste de colisage"),
    ("PL", "distinta di imballaggio"), ("PL", "lista de embalagem"), ("PL", "romaneio"),
    ("PL", "paklijst"), ("PL", "balicí list"), ("PL", "قائمة التعبئة"), ("PL", "포장명세서"),
    ("PL", "梱包明細書"), ("PL", "パッキングリスト"), ("PL", "装箱单"),
    ("PO", "Bestellung"), ("PO", "orden de compra"), ("PO", "ordem de compra"), ("PO", "bon de commande"),
    ("PO", "ordine di acquisto"), ("PO", "pedido de compra"), ("PO", "inkooporder"),
    ("PO", "objednávka"), ("PO", "أمر الشراء"), ("PO", "구매주문서"), ("PO", "발주서"),
    ("PO", "注文書"), ("PO", "発注書"), ("PO", "采购订单"),
    ("PI", "Proformarechnung"), ("PI", "factura proforma"), ("PI", "facture pro forma"),
    ("PI", "facture proforma"), ("PI", "fattura proforma"), ("PI", "fatura proforma"),
    ("PI", "proformafactuur"), ("PI", "proforma faktura"), ("PI", "فاتورة مبدئية"),
    ("PI", "견적송장"), ("PI", "프로포마 인보이스"), ("PI", "見積送り状"),
    ("PI", "プロフォーマインボイス"), ("PI", "形式发票"),
    ("SC", "Kaufvertrag"), ("SC", "contrato de compraventa"), ("SC", "contrat de vente"),
    ("SC", "contratto di vendita"), ("SC", "contrato de venda"), ("SC", "koopovereenkomst"),
    ("SC", "kupní smlouva"), ("SC", "عقد البيع"), ("SC", "매매계약서"), ("SC", "売買契約書"),
    ("SC", "销售合同"),
    ("LC", "Akkreditiv"), ("LC", "carta de crédito"), ("LC", "lettre de crédit"),
    ("LC", "lettera di credito"), ("LC", "kredietbrief"), ("LC", "akreditiv"),
    ("LC", "خطاب اعتماد"), ("LC", "신용장"), ("LC", "信用状"), ("LC", "信用证"),
    ("CO", "Ursprungszeugnis"), ("CO", "certificado de origen"), ("CO", "certificat d'origine"),
    ("CO", "certificato di origine"), ("CO", "certificado de origem"), ("CO", "certificaat van oorsprong"),
    ("CO", "osvědčení o původu"), ("CO", "شهادة المنشأ"), ("CO", "원산지증명"),
    ("CO", "原産地証明書"), ("CO", "原产地证书"),
    ("ED", "Ausfuhranmeldung"), ("ED", "declaración de exportación"), ("ED", "déclaration d'exportation"),
    ("ED", "dichiarazione di esportazione"), ("ED", "declaração de exportação"), ("ED", "uitvoeraangifte"),
    ("ED", "vývozní prohlášení"), ("ED", "بيان التصدير"), ("ED", "수출신고"),
    ("ED", "輸出申告"), ("ED", "出口报关"),
    ("ID", "Einfuhranmeldung"), ("ID", "declaración de importación"), ("ID", "déclaration d'importation"),
    ("ID", "dichiarazione di importazione"), ("ID", "declaração de importação"), ("ID", "invoeraangifte"),
    ("ID", "dovozní prohlášení"), ("ID", "بيان الاستيراد"), ("ID", "수입신고"),
    ("ID", "輸入申告"), ("ID", "进口报关"),
];

/// 🌟 [TRADE DOC TITLES / 12 LANGUAGES]
///  키: TRADE_DOC_TITLES 의 영문 전문 (대소문자·구두점·공백 무시)
///  값: 쉼표로 나눈 11개 언어 전문
///  ai_utils::all_trade_doc_titles 가 영문 전문으로 코드를 찾아 붙입니다.
///  키가 TRADE_DOC_TITLES 와 맞지 않는 항목은 [ML TABLE ORPHANS] 로그에 나타납니다.
pub const TRADE_DOC_TITLES_ML_FULL: &[(&str, &str)] = &[
    ("Commercial Invoice", "Handelsrechnung, Factura comercial, Facture commerciale, 商業送り状, コマーシャルインボイス, Fatura comercial, فاتورة تجارية, Obchodní faktura, Fattura commerciale, 상업송장, Handelsfactuur, 商业发票"),
    ("Proforma Invoice", "Proformarechnung, Factura proforma, Facture pro forma, プロフォーマインボイス, 見積送り状, Fatura pró-forma, فاتورة مبدئية, Proforma faktura, Fattura proforma, 견적송장, Pro-formafactuur, 形式发票"),
    ("Customs Invoice", "Zollrechnung, Factura de aduana, Facture douanière, 税関送り状, Fatura aduaneira, فاتورة جمركية, Celní faktura, Fattura doganale, 세관송장, Douanefactuur, 海关发票"),
    ("Packing List", "Packliste, Lista de empaque, Liste de colisage, 梱包明細書, パッキングリスト, Lista de embalagem, Romaneio, قائمة التعبئة, Balicí list, Lista di imballaggio, 포장명세서, Paklijst, 装箱单"),
    ("Bill of Lading", "Konnossement, Conocimiento de embarque, Connaissement, 船荷証券, Conhecimento de embarque, بوليصة الشحن, Konosament, Polizza di carico, 선하증권, Cognossement, 提单"),
    ("House Bill of Lading", "House-Konnossement, Conocimiento de embarque house, Connaissement house, ハウス船荷証券, Conhecimento de embarque house, بوليصة شحن فرعية, House konosament, Polizza di carico house, 하우스 선하증권, House cognossement, 货代提单"),
    ("Master Bill of Lading", "Master-Konnossement, Conocimiento de embarque master, Connaissement master, マスター船荷証券, Conhecimento de embarque master, بوليصة الشحن الرئيسية, Master konosament, Polizza di carico master, 마스터 선하증권, Master cognossement, 船东提单"),
    ("Sea Waybill", "Seefrachtbrief, Carta de porte marítimo, Lettre de transport maritime, 海上運送状, بيان الشحن البحري, Námořní nákladní list, Lettera di vettura marittima, 해상화물운송장, Zeevrachtbrief, 海运单"),
    ("Air Waybill", "Luftfrachtbrief, Guía aérea, Conocimiento aéreo, Lettre de transport aérien, 航空運送状, エアウェイビル, Conhecimento aéreo, بوليصة الشحن الجوي, Letecký nákladní list, Lettera di vettura aerea, 항공화물운송장, Luchtvrachtbrief, 航空运单"),
    ("House Air Waybill", "House-Luftfrachtbrief, Guía aérea house, Lettre de transport aérien house, ハウスエアウェイビル, Conhecimento aéreo house, بوليصة شحن جوي فرعية, House letecký nákladní list, Lettera di vettura aerea house, 하우스 항공화물운송장, House luchtvrachtbrief, 货代航空运单"),
    ("Master Air Waybill", "Master-Luftfrachtbrief, Guía aérea master, Lettre de transport aérien master, マスターエアウェイビル, Conhecimento aéreo master, بوليصة الشحن الجوي الرئيسية, Master letecký nákladní list, Lettera di vettura aerea master, 마스터 항공화물운송장, Master luchtvrachtbrief, 主航空运单"),
    ("Certificate of Origin", "Ursprungszeugnis, Certificado de origen, Certificat d'origine, 原産地証明書, Certificado de origem, شهادة المنشأ, Osvědčení o původu, Certificato di origine, 원산지증명서, Certificaat van oorsprong, 原产地证书"),
    ("Letter of Credit", "Akkreditiv, Carta de crédito, Lettre de crédit, Crédit documentaire, 信用状, خطاب اعتماد, Akreditiv, Lettera di credito, 신용장, Documentair krediet, 信用证"),
    ("Import Declaration", "Einfuhranmeldung, Declaración de importación, Déclaration d'importation, 輸入申告書, Declaração de importação, إقرار الاستيراد, Dovozní prohlášení, Dichiarazione di importazione, 수입신고서, Invoeraangifte, 进口报关单"),
    ("Export Declaration", "Ausfuhranmeldung, Declaración de exportación, Déclaration d'exportation, 輸出申告書, Declaração de exportação, إقرار التصدير, Vývozní prohlášení, Dichiarazione di esportazione, 수출신고서, Uitvoeraangifte, 出口报关单"),
    ("Purchase Order", "Bestellung, Kaufauftrag, Orden de compra, Bon de commande, 注文書, 発注書, Ordem de compra, Pedido de compra, أمر شراء, Objednávka, Ordine di acquisto, 구매주문서, 발주서, Inkooporder, 采购订单"),
    ("Delivery Order", "Lieferauftrag, Auslieferungsauftrag, Orden de entrega, Ordre de livraison, 荷渡指図書, Ordem de entrega, أمر التسليم, Dodací příkaz, Ordine di consegna, 화물인도지시서, Afleveringsorder, 提货单"),
    ("Arrival Notice", "Ankunftsanzeige, Aviso de llegada, Avis d'arrivée, 貨物到着案内, アライバルノーティス, Aviso de chegada, إشعار الوصول, Oznámení o příjezdu, Avviso di arrivo, 화물도착통지서, Aankomstbericht, 到货通知"),
    ("Booking Confirmation", "Buchungsbestätigung, Confirmación de reserva, Confirmation de réservation, ブッキング確認書, Confirmação de reserva, تأكيد الحجز, Potvrzení rezervace, Conferma di prenotazione, 부킹확인서, 선적예약확인서, Boekingsbevestiging, 订舱确认书"),
    ("Shipping Instruction", "Versandanweisung, Instrucciones de embarque, Instructions d'expédition, 船積指図書, Instruções de embarque, تعليمات الشحن, Přepravní instrukce, Istruzioni di spedizione, 선적지시서, Verschepingsinstructie, 装船指示"),
    ("Freight Invoice", "Frachtrechnung, Factura de flete, Facture de fret, 運賃請求書, Fatura de frete, فاتورة الشحن, Faktura za přepravu, Fattura di trasporto, 운임청구서, Vrachtfactuur, 运费发票"),
    ("Tax Invoice", "Steuerrechnung, Factura fiscal, Facture fiscale, 適格請求書, 税務請求書, Nota fiscal, فاتورة ضريبية, Daňový doklad, Fattura fiscale, 세금계산서, Btw-factuur, 税务发票"),
    ("Debit Note", "Belastungsanzeige, Lastschriftanzeige, Nota de débito, Note de débit, デビットノート, 借方票, إشعار مدين, Vrubopis, Nota di addebito, 차변전표, Debetnota, 借记单"),
    ("Credit Note", "Gutschrift, Nota de crédito, Note de crédit, Facture d'avoir, クレジットノート, 貸方票, إشعار دائن, Dobropis, Nota di credito, 대변전표, Creditnota, 贷记单"),
    ("Weight Certificate", "Gewichtsbescheinigung, Certificado de peso, Certificat de poids, 重量証明書, شهادة الوزن, Vážní list, Certificato di peso, 중량증명서, Gewichtscertificaat, 重量证明"),
    ("Verified Gross Mass", "Verifizierte Bruttomasse, Masa bruta verificada, Masse brute vérifiée, コンテナ総重量証明, Massa bruta verificada, الكتلة الإجمالية المتحقق منها, Ověřená hrubá hmotnost, Massa lorda verificata, 검증총중량, Geverifieerde brutomassa, 核实总重"),
    ("Dangerous Goods Declaration", "Gefahrguterklärung, Declaración de mercancías peligrosas, Déclaration de marchandises dangereuses, 危険物申告書, Declaração de mercadorias perigosas, إقرار البضائع الخطرة, Prohlášení o nebezpečném zboží, Dichiarazione merci pericolose, 위험물신고서, Verklaring gevaarlijke goederen, 危险品申报单"),
    ("Insurance Policy", "Versicherungspolice, Póliza de seguro, Police d'assurance, 保険証券, Apólice de seguro, بوليصة التأمين, Pojistná smlouva, Polizza di assicurazione, 보험증권, Verzekeringspolis, 保险单"),
    ("Insurance Certificate", "Versicherungszertifikat, Certificado de seguro, Certificat d'assurance, 保険証明書, شهادة التأمين, Pojistný certifikát, Certificato di assicurazione, 보험증명서, Verzekeringscertificaat, 保险证明"),
    ("Certificate of Analysis", "Analysenzertifikat, Certificado de análisis, Certificat d'analyse, 分析証明書, Certificado de análise, شهادة التحليل, Certifikát analýzy, Certificato di analisi, 성분분석증명서, Analysecertificaat, 分析证书"),
    ("Certificate of Conformity", "Konformitätsbescheinigung, Certificado de conformidad, Certificat de conformité, 適合証明書, Certificado de conformidade, شهادة المطابقة, Certifikát shody, Certificato di conformità, 적합성증명서, Conformiteitscertificaat, 合格证书"),
    ("Phytosanitary Certificate", "Pflanzengesundheitszeugnis, Certificado fitosanitario, Certificat phytosanitaire, 植物検疫証明書, Certificado fitossanitário, شهادة الصحة النباتية, Rostlinolékařské osvědčení, Certificato fitosanitario, 식물검역증명서, Fytosanitair certificaat, 植物检疫证书"),
    ("Health Certificate", "Gesundheitszeugnis, Certificado sanitario, Certificat sanitaire, 衛生証明書, Certificado sanitário, شهادة صحية, Zdravotní osvědčení, Certificato sanitario, 위생증명서, Gezondheidscertificaat, 卫生证书"),
    ("Fumigation Certificate", "Begasungszertifikat, Certificado de fumigación, Certificat de fumigation, 燻蒸証明書, Certificado de fumigação, شهادة التبخير, Certifikát o fumigaci, Certificato di fumigazione, 훈증증명서, Fumigatiecertificaat, 熏蒸证书"),
    ("Inspection Report", "Inspektionsbericht, Prüfbericht, Informe de inspección, Rapport d'inspection, 検査報告書, Relatório de inspeção, تقرير التفتيش, Inspekční zpráva, Rapporto di ispezione, 검사보고서, Inspectierapport, 检验报告"),
    ("Inspection Certificate", "Inspektionszertifikat, Certificado de inspección, Certificat d'inspection, 検査証明書, Certificado de inspeção, شهادة التفتيش, Inspekční certifikát, Certificato di ispezione, 검사증명서, Inspectiecertificaat, 检验证书"),
    ("Sales Contract", "Kaufvertrag, Contrato de compraventa, Contrat de vente, 売買契約書, Contrato de venda, عقد البيع, Kupní smlouva, Contratto di vendita, 매매계약서, Koopovereenkomst, 销售合同"),
    ("Mate's Receipt", "Bordempfangsschein, Recibo del primer oficial, Reçu de bord, 本船受取証, Recibo de bordo, إيصال الضابط الأول, Lodní potvrzenka, Ricevuta di bordo, 본선수취증, Stuurmansreçu, 大副收据"),
    ("Warehouse Receipt", "Lagerschein, Recibo de almacén, Récépissé d'entrepôt, 倉庫証券, Recibo de armazém, إيصال المستودع, Skladištní list, Ricevuta di magazzino, 창고증권, Opslagbewijs, 仓单"),
    ("Equipment Interchange Receipt", "Containerübergabeschein, Recibo de intercambio de equipo, Reçu d'échange d'équipement, 機器受渡証, Recibo de intercâmbio de equipamento, إيصال تبادل المعدات, Předávací protokol kontejneru, Ricevuta di interscambio container, 기기인수도증, Containerwisselbewijs, 设备交接单"),
    ("Proof of Delivery", "Liefernachweis, Zustellnachweis, Comprobante de entrega, Preuve de livraison, 配達証明, Comprovante de entrega, إثبات التسليم, Doklad o doručení, Prova di consegna, 배송완료증명, Afleverbewijs, 签收单"),
    ("CMR Consignment Note", "CMR-Frachtbrief, Carta de porte CMR, Lettre de voiture CMR, CMR運送状, بوليصة شحن CMR, Nákladní list CMR, Lettera di vettura CMR, 국제도로화물운송장, CMR-vrachtbrief, 国际公路运单"),
    ("Rail Waybill", "Eisenbahnfrachtbrief, Carta de porte ferroviario, Lettre de voiture ferroviaire, 鉄道運送状, Carta de porte ferroviário, بوليصة الشحن بالسكك الحديدية, Železniční nákladní list, Lettera di vettura ferroviaria, 철도화물운송장, Spoorvrachtbrief, 铁路运单"),
    ("Forwarder's Cargo Receipt", "Spediteurübernahmebescheinigung, Recibo de carga del transitario, Récépissé de transitaire, 貨物受取証, Recibo de carga do transitário, إيصال استلام البضائع من وكيل الشحن, Potvrzení zasílatele o převzetí, Ricevuta di carico dello spedizioniere, 운송주선인 화물수취증, Expediteursontvangstbewijs, 货代收货证明"),
    ("Bill of Exchange", "Wechsel, Letra de cambio, Lettre de change, 為替手形, Letra de câmbio, كمبيالة, Směnka, Cambiale, 환어음, Wisselbrief, 汇票"),
    ("Quotation", "Angebot, Cotización, Devis commercial, Offre de prix, 見積書, Cotação, Orçamento, عرض سعر, Cenová nabídka, Preventivo, 견적서, Offerte, 报价单"),
    ("Order Confirmation", "Auftragsbestätigung, Confirmación de pedido, Confirmation de commande, 注文確認書, Confirmação de pedido, تأكيد الطلب, Potvrzení objednávky, Conferma d'ordine, 주문확인서, Orderbevestiging, 订单确认"),
    ("Delivery Note", "Lieferschein, Albarán, Nota de entrega, Bon de livraison, 納品書, مذكرة التسليم, Dodací list, Bolla di consegna, Documento di trasporto, 납품서, Pakbon, 送货单"),
    ("Cargo Manifest", "Ladungsmanifest, Manifiesto de carga, Manifeste de cargaison, 積荷目録, Manifesto de carga, بيان الشحنة, Manifest nákladu, Manifesto di carico, 적하목록, Ladingmanifest, 载货清单"),
    ("Shipping Bill", "Ausfuhrschein, Póliza de exportación, Déclaration d'expédition, 船積申告書, Guia de exportação, بوليصة الشحن الجمركية, Vývozní celní doklad, Bolletta di esportazione, 선적신고서, Uitvoerdocument, 出口装运单"),
    ("Bill of Entry", "Einfuhrschein, Declaración de entrada, Déclaration d'entrée, 輸入申告, Declaração de entrada, بيان الدخول الجمركي, Celní prohlášení, Bolletta doganale, 수입통관신고서, Invoerdocument, 进口报关单"),
    ("Remittance Advice", "Zahlungsavis, Aviso de remesa, Avis de paiement, 送金通知書, Aviso de remessa, إشعار التحويل, Avízo o platbě, Avviso di pagamento, 송금통지서, Betalingsadvies, 汇款通知"),
    ("Payment Receipt", "Zahlungsbeleg, Quittung, Recibo de pago, Reçu de paiement, 領収書, Recibo de pagamento, إيصال الدفع, Doklad o zaplacení, Ricevuta di pagamento, 영수증, Betalingsbewijs, 付款收据"),
    ("Bank Guarantee", "Bankgarantie, Garantía bancaria, Garantie bancaire, 銀行保証状, Garantia bancária, ضمان بنكي, Bankovní záruka, Garanzia bancaria, 은행보증서, 银行保函"),
    ("Statement of Account", "Kontoauszug, Estado de cuenta, Relevé de compte, 取引明細書, Extrato de conta, كشف حساب, Výpis z účtu, Estratto conto, 거래명세서, Rekeningoverzicht, 对账单"),
    ("Survey Report", "Havariebericht, Schadensgutachten, Informe de peritaje, Rapport d'expertise, 鑑定報告書, Laudo de vistoria, تقرير المعاينة, Znalecký posudek, Perizia, 검정보고서, Expertiserapport, 鉴定报告"),
];

/// 🌟 [TRADE GROUPS / 12 LANGUAGES]
///  키: TRADE_GROUP_CODES 의 그룹명. 값: 그 그룹의 '개념' 구 (서식 전문이 아니라 서류 부류를 부르는 말).
///  그룹 뱅크의 서식 전문은 소속 코드의 전문(위 표 포함)에서 자동으로 모이므로 여기엔 부류 명칭만 둡니다.
pub const TRADE_GROUPS_ML: &[(&str, &str)] = &[
    ("shipping", "Transportdokument, Frachtpapiere, documento de transporte, documentos de embarque, document de transport, titre de transport, 運送書類, 船積書類, documento de transporte, documentos de embarque, مستندات النقل, مستندات الشحن, přepravní doklad, přepravní dokumenty, documento di trasporto, documenti di spedizione, 운송서류, 선적서류, vervoersdocument, transportdocumenten, 运输单据, 装运单据"),
    ("contract", "Vertrag, Bestellunterlagen, Handelsdokument, contrato, documentos de pedido, documento comercial, contrat, documents de commande, document commercial, 契約書, 注文書類, 商業書類, contrato, documentos de pedido, عقد, مستندات الطلب, مستندات تجارية, smlouva, objednávkové doklady, obchodní doklad, contratto, documenti d'ordine, documento commerciale, 계약서, 주문서류, 상업서류, contract, orderdocumenten, handelsdocument, 合同, 订单单据, 商业单据"),
    ("customs", "Zolldokument, Zollpapiere, documento aduanero, document douanier, 通関書類, 税関書類, documento aduaneiro, مستندات جمركية, celní doklad, documento doganale, 통관서류, douanedocument, 报关单据, 海关单证"),
    ("settlement", "Zahlungsdokument, Finanzdokument, documento de pago, documento financiero, document de paiement, document financier, 決済書類, 金融書類, documento de pagamento, documento financeiro, مستندات الدفع, مستندات مالية, platební doklad, finanční doklad, documento di pagamento, documento finanziario, 결제서류, 금융서류, betalingsdocument, financieel document, 结算单据, 金融单据"),
    ("legal", "Rechtsdokument, Versicherungsdokument, Erklärung, Lizenz, Vollmacht, documento legal, documento de seguro, declaración, licencia, poder notarial, document juridique, document d'assurance, déclaration, licence, procuration, 法的書類, 保険書類, 申告書, 許可証, 委任状, documento legal, documento de seguro, declaração, licença, procuração, مستند قانوني, مستندات التأمين, إقرار, ترخيص, توكيل, právní dokument, pojistný doklad, prohlášení, licence, plná moc, documento legale, documento assicurativo, dichiarazione, licenza, procura, 법률서류, 보험서류, 신고서, 허가증, 위임장, juridisch document, verzekeringsdocument, verklaring, vergunning, volmacht, 法律文件, 保险单据, 申报, 许可证, 委托书"),
    ("inspection", "Prüfzertifikat, Bescheinigung, Zeugnis, certificado, informe de inspección, certificat, attestation, 証明書, 検査書類, certificado, atestado, شهادة, مستندات الفحص, osvědčení, certifikát, certificato, attestato, 증명서, 검사서류, certificaat, keuringsdocument, 证书, 检验单据"),
    ("contract", "Vertrag, Bestellunterlagen, contrato, documentos de pedido, contrat, documents de commande, 契約書, 注文書類, contrato, documentos de pedido, عقد, مستندات الطلب, smlouva, objednávkové doklady, contratto, documenti d'ordine, 계약서, 주문서류, contract, orderdocumenten, 合同, 订单单据"),
];

pub fn anchor_phrases(en: &str, ml: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in [en, ml] {
        if raw.trim().is_empty() { continue; }
        for p in crate::utils::ai_utils::split_bias_phrases_full(raw) {
            let t = p.trim();
            if t.is_empty() { continue; }
            if out.iter().any(|e| e.eq_ignore_ascii_case(t)) { continue; }
            out.push(t.to_string());
        }
    }
    out
}

pub fn trade_title_pairs() -> Vec<(&'static str, &'static str)> {
    let mut out: Vec<(&'static str, &'static str)> = Vec::new();
    for (code, title) in TRADE_DOC_TITLES
        .iter()
        .chain(TRADE_DOC_TITLES_ML.iter())
    {
        let t = title.trim();
        if t.is_empty() { continue; }
        if out.iter().any(|(c, x)| *c == *code && x.eq_ignore_ascii_case(t)) { continue; }
        out.push((*code, t));
    }
    out
}

pub fn trade_title_bank_defs(
    cat: &str,
) -> (Vec<(String, String, String)>, Vec<(String, String, String)>) {
    let pairs = trade_title_pairs();
    let mut codes: Vec<&'static str> = Vec::new();
    for (c, _) in pairs.iter() {
        if !codes.iter().any(|x| x == c) { codes.push(*c); }
    }
    let mut bias: Vec<(String, String, String)> = Vec::new();
    let mut prej: Vec<(String, String, String)> = Vec::new();
    for code in codes.iter() {
        let own: Vec<&str> = pairs
            .iter()
            .filter(|(c, _)| c == code)
            .map(|(_, t)| *t)
            .collect();
        for t in own.iter() {
            bias.push((cat.to_string(), code.to_string(), t.to_string()));
        }
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (c, t) in pairs.iter() {
            if c == code { continue; }
            if own.iter().any(|o| o.eq_ignore_ascii_case(t)) { continue; }
            if seen.insert(t.to_lowercase()) {
                prej.push((cat.to_string(), code.to_string(), t.to_string()));
            }
        }
    }
    (bias, prej)
}

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
            | "recipient_address" | "notify_party_name"
            | "sender_tax_number" | "recipient_tax_number" => "parties",

        // ── other_parties (배열) ──
        //  🌟 45종 예시에서 확인된 역할은 42개입니다.
        //     (drawer / drawee / payee / carrier / insurer / issuing_bank /
        //      advising_bank / customs_broker / warehouse_operator / claimant …)
        //     개별 필드로 두면 issuing_bank·advising_bank·entrustor_bank 세 축이
        //     같은 '은행명' 영역을 놓고 경쟁해 근거가 3분할되고 전부 굶습니다.
        //     role 을 원소 필드로 두면 한 영역에서 여러 역할을 순서대로 읽어내고,
        //     우리가 예상하지 못한 역할까지 스키마 변경 없이 수용합니다.
        "party_role" | "party_name" | "party_address"
            | "party_contact" | "signatory_name" => "other_parties",

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
            | "port_of_transshipment" | "cargo_closing_date"
            | "country_of_export" | "country_of_destination" => "logistics",

        // ── conditions ──
        "incoterms" | "payment_terms" | "freight_payment_term"
            | "partial_shipments" | "transshipment_allowed" | "latest_shipment_date"
            | "governing_law" | "arbitration" | "non_negotiable"
            | "temperature_control" | "storage_term" | "release_instruction"
            | "special_instructions" | "reason_for_export" => "conditions",

        // ── financials ──
        //  🌟 [FINANCIAL ALIAS GUARD] CI overlay 의 'insurance' / 'freight_charge' /
        //     'grand_total_amount' / 'amount_krw' / 'total_amount_krw' / 'tax_rate' /
        //     'legalization_fee' / 'discount' 는 전부 금액 축입니다.
        //     그런데 명시 매핑에 없으면 규칙 폴백이 다음처럼 잡습니다.
        //       "insurance"        → f.contains("insur")   → insurance   (카테고리명과 충돌)
        //       "freight_charge"   → f.contains("charge")  → financials  (우연히 정답)
        //       "legalization_fee" → 어디에도 안 걸림       → ""          (전량 폐기)
        //     'insurance' 는 실측에서 insurance 버킷을 새로 만들어
        //     cargo 값이 그 안에 복제되는 오염을 유발했습니다.
        //     금액 축을 명시적으로 못박아 폴백이 개입할 여지 자체를 없앱니다.
        "currency" | "amount" | "amount_subtotal" | "amount_tax"
            | "freight_amount" | "insurance_amount" | "local_charges"
            | "exchange_rate" | "bank_charges" | "usance_tenor_days" | "tenor"
            | "maturity_date" | "due_date" | "valid_until"
            | "remittance_reference" | "swift_code" | "account_number"
            | "insurance" | "freight_charge" | "grand_total_amount"
            | "amount_krw" | "total_amount_krw" | "tax_rate"
            | "legalization_fee" | "discount" | "storage_fee" => "financials",

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

pub const TRADE_ENUM_VALUE_ANCHORS: &[(&str, &[(&str, &str)])] = &[
    ("currency", &[
        ("USD", "USD, US dollar, United States dollar"),
        ("EUR", "EUR, euro"),
        ("JPY", "JPY, Japanese yen"),
        ("CNY", "CNY, Chinese yuan renminbi"),
        ("KRW", "KRW, South Korean won"),
        ("GBP", "GBP, British pound sterling"),
    ]),
    ("incoterms", &[
        ("EXW", "EXW, ex works"),
        ("FCA", "FCA, free carrier"),
        ("FAS", "FAS, free alongside ship"),
        ("FOB", "FOB, free on board"),
        ("CFR", "CFR, cost and freight"),
        ("CIF", "CIF, cost insurance and freight"),
        ("CPT", "CPT, carriage paid to"),
        ("CIP", "CIP, carriage and insurance paid to"),
        ("DAP", "DAP, delivered at place"),
        ("DPU", "DPU, delivered at place unloaded"),
        ("DDP", "DDP, delivered duty paid"),
    ]),
];

pub const TRADE_ARRAY_CATEGORIES: &[&str] = &["items", "containers"];
pub const TRADE_IDENTITY_CATEGORY: &str = "header";
pub const TRADE_IDENTITY_FIELD: &str = "doc_number";

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
    ("TRACKING", "tracking label shipping label parcel waybill"),
];

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
    if f.contains("damage") || f.contains("finding") || f.contains("loss")
        || f.contains("affected") || f.contains("surveyor") {
        return "inspection";
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

pub fn is_trade_array_category(category: &str) -> bool {
    matches!(
        category,
        "items" | "containers" | "other_parties" | "charges"
            | "test_results" | "findings_and_damage" | "account_ledger"
    )
}

pub const TRADE_EXTRACTION_CATEGORIES: [&str; 20] = [
    // ── base (전 서식 공통) ──
    "header", "parties", "other_parties", "logistics", "conditions",
    "financials", "cargo", "items", "containers",
    // ── overlay (서식별 조건부) ──
    "customs", "inspection", "insurance", "settlement",
    "hazmat", "origin", "compliance", "charges",
    "test_results", "findings_and_damage", "account_ledger",
];
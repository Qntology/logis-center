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

// pub fn merge_node(target: &mut Value, source: &Value) {
//     if let (Value::Object(target_map), Value::Object(source_map)) = (target, source) {
//         for (key, source_value) in source_map {
//             let is_empty = source_value.is_null() || 
//                            (source_value.is_string() && source_value.as_str() == Some("")) ||
//                            (source_value.is_number() && source_value.as_f64() == Some(0.0));

//             if !is_empty {
//                 target_map.insert(key.clone(), source_value.clone());
//             }
//         }
//     }
// }

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
pub fn related_trading(doc_type: &str) -> Vec<&'static str> {
    match doc_type {
        "BL"  => vec!["CI", "PL"],
        "CI"  => vec!["BL", "PL", "LC"],
        "PL"  => vec!["BL", "CI"],
        "PO"  => vec!["PI", "SC"],
        "PI"  => vec!["PO", "SC"],
        "SC"  => vec!["PO", "PI"],
        "LC"  => vec!["CI"],
        "CO"  => vec!["CI"],
        "AWB" => vec!["CI", "PL"],
        "SA"  => vec!["BL", "CI"],
        "DO"  => vec!["BL", "AN"],
        "AN"  => vec!["DO", "BL"],
        "BC"  => vec!["BL", "CI"],
        "ED"  => vec!["CI", "PL"],
        "ID"  => vec!["CI", "PL"],
        "CINV"=> vec!["CI"],
        _     => vec![],
    }
}

// 🌟 [TRADING RELAY FIELD v2 / DIRECTIONAL PAIR]
//  ── 무엇이 문제였나 ──
//   v1 은 (from, to) 쌍에 대해 '단 하나의 필드명'만 돌려주었습니다.
//   그런데 실무 서식은 방향에 따라 참조 필드가 다릅니다.
//     BL.reference_invoice = CI.doc_number   (BL 쪽 필드는 reference_invoice)
//     CI.doc_number        = BL.reference_invoice
//   v1 은 양쪽 모두 "reference_invoice" 를 반환했기 때문에,
//   CI 문서에서 relay 를 돌 때 `CI.reference_invoice` 라는 존재하지 않는 값을 읽어
//   조회가 항상 0건이었습니다.
//
//  반환값: (내 문서에서 값을 읽어올 필드, 상대 문서에서 조회할 필드)
pub fn trading_relay_pair(from_type: &str, to_type: &str) -> Option<(&'static str, &'static str)> {
    match (from_type, to_type) {
        // ── 인보이스 참조 축 ──
        ("BL",  "CI")  => Some(("reference_invoice", "doc_number")),
        ("CI",  "BL")  => Some(("doc_number", "reference_invoice")),
        ("PL",  "CI")  => Some(("reference_invoice", "doc_number")),
        ("CI",  "PL")  => Some(("doc_number", "reference_invoice")),
        ("CO",  "CI")  => Some(("reference_invoice", "doc_number")),
        ("CI",  "CO")  => Some(("doc_number", "reference_invoice")),
        ("AWB", "CI")  => Some(("reference_invoice", "doc_number")),
        ("CI",  "AWB") => Some(("doc_number", "reference_invoice")),
        ("SA",  "CI")  => Some(("reference_invoice", "doc_number")),
        ("CI",  "SA")  => Some(("doc_number", "reference_invoice")),
        ("BC",  "CI")  => Some(("reference_invoice", "doc_number")),
        ("CI",  "BC")  => Some(("doc_number", "reference_invoice")),
        ("ED",  "CI")  => Some(("reference_invoice", "doc_number")),
        ("CI",  "ED")  => Some(("doc_number", "reference_invoice")),
        ("ID",  "CI")  => Some(("reference_invoice", "doc_number")),
        ("CI",  "ID")  => Some(("doc_number", "reference_invoice")),
        ("CINV","CI")  => Some(("reference_invoice", "doc_number")),
        ("CI",  "CINV")=> Some(("doc_number", "reference_invoice")),
        ("ED",  "PL")  => Some(("reference_invoice", "reference_invoice")),
        ("PL",  "ED")  => Some(("reference_invoice", "reference_invoice")),
        ("ID",  "PL")  => Some(("reference_invoice", "reference_invoice")),
        ("PL",  "ID")  => Some(("reference_invoice", "reference_invoice")),

        // ── 부킹 참조 축 ──
        ("BL",  "PL")  => Some(("reference_booking", "reference_booking")),
        ("PL",  "BL")  => Some(("reference_booking", "reference_booking")),
        ("AWB", "PL")  => Some(("reference_booking", "reference_booking")),
        ("PL",  "AWB") => Some(("reference_booking", "reference_booking")),
        ("BL",  "BC")  => Some(("reference_booking", "doc_number")),
        ("BC",  "BL")  => Some(("doc_number", "reference_booking")),
        ("SA",  "BL")  => Some(("reference_booking", "reference_booking")),
        ("BL",  "SA")  => Some(("reference_booking", "reference_booking")),
        ("DO",  "BL")  => Some(("reference_booking", "reference_booking")),
        ("BL",  "DO")  => Some(("reference_booking", "reference_booking")),
        ("DO",  "AN")  => Some(("reference_booking", "reference_booking")),
        ("AN",  "DO")  => Some(("reference_booking", "reference_booking")),
        ("AN",  "BL")  => Some(("reference_booking", "reference_booking")),
        ("BL",  "AN")  => Some(("reference_booking", "reference_booking")),

        // ── L/C 참조 축 ──
        ("CI",  "LC")  => Some(("reference_lc", "doc_number")),
        ("LC",  "CI")  => Some(("doc_number", "reference_lc")),

        // ── 계약 3종 상호 참조 ──
        ("PO",  "PI")  => Some(("doc_number", "doc_number")),
        ("PI",  "PO")  => Some(("doc_number", "doc_number")),
        ("PO",  "SC")  => Some(("doc_number", "doc_number")),
        ("SC",  "PO")  => Some(("doc_number", "doc_number")),
        ("PI",  "SC")  => Some(("doc_number", "doc_number")),
        ("SC",  "PI")  => Some(("doc_number", "doc_number")),

        _ => None,
    }
}

// 🌟 [BACK-COMPAT] 기존 호출부(trading_relay_field)를 살려 둡니다.
//  '내 쪽 필드'만 반환하므로 v1 과 동일하게 동작합니다.
pub fn trading_relay_field(from_type: &str, to_type: &str) -> Option<&'static str> {
    trading_relay_pair(from_type, to_type).map(|(mine, _)| mine)
}

// 🌟 [TRADING INDEX COLUMN] commerce 가 order↔tracking 을 'order'/'tracking' 이라는
//  숫자 index 컬럼으로 잇는 것과 동일한 구조를 무역 서식에 부여합니다.
//
//  commerce:
//     order.tracking  = crc32(hash_id("tracking" + team_id + tracking_number))
//     tracking.order  = crc32(hash_id("order"    + team_id + order_no))
//
//  trading (이 함수가 정의):
//     BL.ci = crc32(hash_id("CI" + team_id + 정규화된 CI doc_number))
//     CI.bl = crc32(hash_id("BL" + team_id + 정규화된 BL doc_number))
//
//  반환값은 data.* 인덱스 경로에 쓰일 소문자 컬럼명입니다.
//  (Dexie 의 'data.rel_ci' 인덱스와 1:1 대응)
pub fn trading_index_column(doc_type: &str) -> String {
    format!("rel_{}", doc_type.to_lowercase())
}

pub fn relay(foreign_type: &str, primary_item: &Value) -> Option<(Vec<QueryInfo>, MergeInfo)> {

    let mut primary_type = primary_item.get("type")?.as_str()?;

    

    // [STRICT PARITY] Handle type aliasing from server logic

    if primary_type == "sales" { primary_type = "order"; }

    let f_type = if foreign_type == "receiving" || foreign_type == "shipping" { "tracking" } else { foreign_type };

    

    let mut queries = Vec::new();

    

    let get_val = |key: &str| -> Option<Value> { primary_item.get(key).cloned() };

    

    // Common include fields for sales/goods merge

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



        // --- Goods as Primary ---

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



        // --- Tracking as Primary ---

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



        // --- Coupon/Event as Primary ---

        ("goods", "coupon" | "event") => {

             queries.push(QueryInfo { r#type: f_type.to_string(), table: "sales".to_string(), column: "event".to_string(), value: get_val("index")?, status: None });

             // No update/upsert info in original logic for this case, just return from/to

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

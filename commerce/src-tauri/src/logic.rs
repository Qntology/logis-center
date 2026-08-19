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

// 🌟 [TRADING RELAY FIELD] 두 무역 서식 사이를 연결하는 참조 필드명을 반환합니다.
//    related_trading() 이 '어떤 서식과 연결되는지'를 정의하고,
//    이 함수가 '어떤 필드로 연결되는지'를 정의합니다.
pub fn trading_relay_field(from_type: &str, to_type: &str) -> Option<&'static str> {
    match (from_type, to_type) {
        ("BL", "CI") | ("CI", "BL") => Some("reference_invoice"),
        ("BL", "PL") | ("PL", "BL") => Some("reference_booking"),
        ("CI", "PL") | ("PL", "CI") => Some("reference_invoice"),
        ("CI", "LC") | ("LC", "CI") => Some("reference_lc"),
        ("PO", "PI") | ("PI", "PO") => Some("doc_number"),
        ("PO", "SC") | ("SC", "PO") => Some("doc_number"),
        ("PI", "SC") | ("SC", "PI") => Some("doc_number"),
        ("CO", "CI") | ("CI", "CO") => Some("reference_invoice"),
        ("AWB", "CI") | ("CI", "AWB") => Some("reference_invoice"),
        ("AWB", "PL") | ("PL", "AWB") => Some("reference_booking"),
        ("SA", "BL") | ("BL", "SA") => Some("reference_booking"),
        ("SA", "CI") | ("CI", "SA") => Some("reference_invoice"),
        ("DO", "BL") | ("BL", "DO") => Some("reference_booking"),
        ("DO", "AN") | ("AN", "DO") => Some("reference_booking"),
        ("AN", "BL") | ("BL", "AN") => Some("reference_booking"),
        ("BC", "BL") | ("BL", "BC") => Some("reference_booking"),
        ("BC", "CI") | ("CI", "BC") => Some("reference_invoice"),
        ("ED", "CI") | ("CI", "ED") => Some("reference_invoice"),
        ("ED", "PL") | ("PL", "ED") => Some("reference_invoice"),
        ("ID", "CI") | ("CI", "ID") => Some("reference_invoice"),
        ("ID", "PL") | ("PL", "ID") => Some("reference_invoice"),
        ("CINV", "CI") | ("CI", "CINV") => Some("reference_invoice"),
        _ => None,
    }
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

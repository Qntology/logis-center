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

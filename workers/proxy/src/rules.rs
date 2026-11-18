//! 원본의 프롬프트 함수와 관계 규칙 — `image2json` · `para2graph` · `graph2contexts` · `list2json` · `item2json` · `context2results` ·
//! `semantic_prompt_system` · `Related` · `parseStatus` · `Relay` · `operators` · `paragraph2propertys` · `parseCondition` · `randomKey` ·
//! `hasUndefinedValue`. 프롬프트 글은 [`crate::templates`] 의 조각 그대로.

use logis_worker_common::date;
use logis_worker_common::jsv::{loose_eq, num_to_string, stringify_str, tpl, V};
use logis_worker_common::util::merge_node;
use logis_worker_common::{arr, console_log, obj, R};

use crate::templates::*;

fn is(v: &V, s: &str) -> bool {
    loose_eq(v, &V::from(s))
}

/// `image2json(region, language, type, address)` — tracking 이 아니면 undefined
pub fn image2json(region: &V, language: &V, type_: &V, address: &V) -> V {
    if is(type_, "tracking") {
        V::from(tpl(T128, &[region.clone(), V::from(stringify_str(address)), language.clone()]))
    } else {
        V::Undef
    }
}

pub fn para2graph(language: &V) -> V {
    V::from(tpl(T159, &[language.clone()]))
}

pub fn graph2contexts(current: &V) -> V {
    V::from(tpl(T273, &[current.clone()]))
}

pub fn list2json(language: &V) -> V {
    V::from(tpl(T282, &[language.clone()]))
}

/// `item2json(type, href)` — 다섯 갈래 밖이면 undefined
pub fn item2json(type_: &V, href: &V) -> V {
    let t = type_.clone();
    let h = href.clone();
    let s = if is(type_, "tracking") {
        tpl(T325, &[t.clone(), t])
    } else if is(type_, "goods") {
        tpl(T409, &[t, h])
    } else if is(type_, "order") {
        tpl(T607, &[t, h])
    } else if is(type_, "coupon") || is(type_, "event") {
        tpl(T690, &[t.clone(), h, t.clone(), t.clone(), t])
    } else if is(type_, "review") {
        tpl(T759, &[t.clone(), h, t.clone(), t])
    } else {
        return V::Undef;
    };
    V::from(s)
}

/// `context2results(context, results, language)` — context 가 undefined 면 원본처럼 던진다
pub fn context2results(context: &V, results: &V, language: &V) -> R<V> {
    let mut condition = V::from("");
    if context.g("condition")?.truthy() {
        condition = V::from(tpl(T794, &[V::from(stringify_str(&context.g("condition")?))]));
    }
    Ok(V::from(tpl(T797, &[context.g("text")?, condition, V::from(stringify_str(results)), language.clone()])))
}

pub fn semantic_prompt_system(language: &V) -> V {
    V::from(tpl(T815, &[language.clone()]))
}

/// `Related(type)`
pub fn related(type_: &V) -> V {
    let list: &[&str] = if is(type_, "goods") {
        &["order", "tracking", "coupon", "event"]
    } else if is(type_, "order") {
        &["goods", "tracking", "coupon", "event"]
    } else if is(type_, "tracking") {
        &["goods", "order", "coupon", "event"]
    } else if is(type_, "coupon") {
        &["goods", "event"]
    } else if is(type_, "event") {
        &["goods", "coupon"]
    } else if is(type_, "review") {
        &["goods", "coupon", "event"]
    } else {
        &[]
    };
    V::from(list.iter().map(|s| V::from(*s)).collect::<Vec<_>>())
}

/// `parseStatus(status)`
pub fn parse_status(status: &V) -> V {
    const STEPS: &[(&str, i32)] = &[
        ("progress", 1),
        ("stop", 2),
        ("cancel", 3),
        ("refund", 4),
        ("return", 5),
        ("error", 6),
        ("expire", 7),
        ("exchange", 8),
        ("complete", 9),
        ("draft", 10),
        ("show", 11),
        ("hide", 12),
    ];
    V::from(STEPS.iter().find(|(s, _)| is(status, s)).map_or(0, |(_, n)| *n))
}

fn strs(items: &[&str]) -> V {
    V::from(items.iter().map(|s| V::from(*s)).collect::<Vec<_>>())
}

const SALES_FIELDS: &[&str] = &[
    "event", "width", "height", "length", "weight", "size", "currency", "cost_price", "sale_price", "discount", "quantity", "tracking", "number",
    "carrier", "shipping_fee", "shipping_method", "shipping_duration", "fulfillment_service", "stock_keeping_unit", "bundle_shipping", "used",
    "lease", "rental", "refurbish", "tax_included", "release_date",
];
const SHIPPING_FIELDS: &[&str] = &["width", "height", "length", "weight", "shipping_fee", "shipping_method", "shipping_duration", "bundle_shipping"];
const EVENT_FIELDS: &[&str] = &[
    "started_at", "expired_at", "phone", "address", "discount", "quantity", "usage_per", "usage_limit", "min_order_amount", "max_order_amount",
    "max_discount_amount", "new_customer_only", "first_purchase_only", "region_restrictions",
];

/// `Relay(foreign, primary)` → `{ query, merge }` — primary 가 undefined 면 원본처럼 던진다
pub fn relay(foreign: &V, primary: &V) -> R<V> {
    let query = V::new_arr();
    let mut merge = V::new_obj();
    // 원본의 조건은 모두 `foreign == X && primary.type == Y` — foreign 이 어느 X 와도 다르면 primary 를 읽지 않는다(던지지 않는다)
    let known = ["goods", "order", "tracking", "coupon", "event"].iter().any(|x| is(foreign, x));
    let ptype = if known { primary.g("type")? } else { V::Undef };
    let p = |k: &str| primary.get(k);

    if is(foreign, "goods") && is(&ptype, "order") {
        if p("tracking").truthy() {
            query.push(obj! { "type" => ptype.clone(), "table" => "sales", "column" => "tracking", "value" => p("tracking") })?;
            merge = obj! { "upsert" => obj! { "includes" => strs(SALES_FIELDS), "from" => foreign.clone(), "to" => ptype.clone() } };
        } else {
            query.push(obj! { "type" => ptype.clone(), "table" => "sales", "column" => "index", "value" => p("index") })?;
            merge = obj! { "update" => obj! { "includes" => strs(SALES_FIELDS), "column" => "index", "value" => p("index"), "from" => foreign.clone(), "to" => ptype.clone() } };
        }
    } else if is(foreign, "tracking") && is(&ptype, "order") {
        if p("tracking").truthy() {
            query.push(obj! { "type" => foreign.clone(), "table" => "tracking", "column" => ptype.clone(), "value" => p("index") })?;
            merge = obj! { "update" => obj! {
                "includes" => strs(&["width", "height", "length", "weight"]),
                "column" => "index", "value" => p("index"),
                "foreign" => obj! { "from" => "index", "to" => "tracking" },
                "from" => ptype.clone(), "to" => foreign.clone(),
            } };
        } else {
            query.push(obj! { "type" => foreign.clone(), "table" => "tracking", "column" => ptype.clone(), "value" => p("index") })?;
            merge = obj! { "update" => obj! {
                "includes" => strs(&["no", "goods", "event"]),
                "column" => "index", "value" => p("index"),
                "foreign" => obj! { "from" => "index", "to" => "tracking" },
                "from" => foreign.clone(), "to" => ptype.clone(),
            } };
        }
    } else if (is(foreign, "coupon") || is(foreign, "event")) && is(&ptype, "order") {
        query.push(obj! { "type" => foreign.clone(), "table" => "event", "column" => "index", "value" => p("event") })?;
        merge = obj! { "update" => obj! { "includes" => strs(&["discount"]), "column" => "index", "value" => p("index"), "from" => foreign.clone(), "to" => ptype.clone() } };
    } else if is(foreign, "order") && is(&ptype, "goods") {
        query.push(obj! { "type" => foreign.clone(), "table" => "sales", "column" => "goods", "value" => p("index") })?;
        merge = obj! { "update" => obj! { "includes" => strs(SALES_FIELDS), "column" => "goods", "value" => p("index"), "from" => ptype.clone(), "to" => foreign.clone() } };
    } else if is(foreign, "tracking") && is(&ptype, "goods") {
        query.push(obj! { "type" => "order", "status" => 0, "table" => "tracking", "column" => "goods", "value" => p("index") })?;
        merge = obj! { "update" => obj! { "includes" => strs(SHIPPING_FIELDS), "from" => ptype.clone(), "to" => foreign.clone() } };
    } else if (is(foreign, "event") || is(foreign, "coupon")) && is(&ptype, "goods") {
        query.push(obj! { "type" => foreign.clone(), "table" => "event", "column" => "index", "value" => p("event") })?;
        merge = obj! { "update" => obj! { "includes" => strs(&["discount"]), "column" => "index", "value" => p("index"), "from" => foreign.clone(), "to" => ptype.clone() } };
    } else if is(foreign, "goods") && is(&ptype, "tracking") {
        query.push(obj! { "type" => "order", "status" => 0, "table" => "sales", "column" => "goods", "value" => p("goods") })?;
        merge = obj! { "update" => obj! { "includes" => strs(SHIPPING_FIELDS), "column" => "index", "value" => p("index"), "from" => foreign.clone(), "to" => ptype.clone() } };
    } else if is(foreign, "order") && is(&ptype, "tracking") {
        if p("goods").truthy() {
            query.push(obj! { "type" => foreign.clone(), "table" => "sales", "column" => "goods", "value" => p("goods") })?;
            merge = obj! { "update" => obj! {
                "includes" => strs(SHIPPING_FIELDS),
                "column" => "tracking", "value" => p("index"),
                "foreign" => obj! { "from" => "index", "to" => "tracking" },
                "from" => ptype.clone(), "to" => foreign.clone(),
            } };
        } else {
            query.push(obj! { "type" => foreign.clone(), "table" => "tracking", "column" => ptype.clone(), "value" => p("index") })?;
            merge = obj! { "update" => obj! {
                "includes" => strs(&["no", "order", "goods", "event"]),
                "column" => "index", "value" => p("index"),
                "foreign" => obj! { "from" => "index", "to" => "order" },
                "from" => foreign.clone(), "to" => ptype.clone(),
            } };
        }
    } else if (is(foreign, "event") || is(foreign, "coupon")) && is(&ptype, "tracking") {
        // 매칭이 아예 안되는 항목
    } else if (is(foreign, "goods") || is(foreign, "order")) && is(&ptype, "event") {
        query.push(obj! { "type" => foreign.clone(), "table" => "sales", "column" => "event", "value" => p("index") })?;
        merge = obj! { "update" => obj! { "includes" => strs(&["discount"]), "column" => "event", "value" => p("index"), "from" => ptype.clone(), "to" => foreign.clone() } };
    } else if is(foreign, "tracking") && is(&ptype, "event") {
        // 매칭이 아예 안되는 항목
    } else if is(foreign, "coupon") && is(&ptype, "event") {
        query.push(obj! { "type" => foreign.clone(), "table" => "event", "column" => "event", "value" => p("index") })?;
        merge = obj! { "update" => obj! { "includes" => strs(EVENT_FIELDS), "column" => "event", "value" => p("index"), "from" => ptype.clone(), "to" => foreign.clone() } };
    } else if is(foreign, "goods") && is(&ptype, "coupon") {
        query.push(obj! { "type" => foreign.clone(), "table" => "sales", "column" => "event", "value" => p("index") })?;
        merge = obj! { "from" => ptype.clone(), "to" => foreign.clone() };
    } else if is(foreign, "order") && is(&ptype, "coupon") {
        query.push(obj! { "type" => foreign.clone(), "status" => 0, "table" => "sales", "column" => "event", "value" => p("index") })?;
        merge = obj! { "update" => obj! { "includes" => strs(&["discount"]), "column" => "event", "value" => p("index"), "from" => ptype.clone(), "to" => foreign.clone() } };
    } else if is(foreign, "tracking") && is(&ptype, "coupon") {
        // 매칭이 아예 안되는 항목
    } else if is(foreign, "event") && is(&ptype, "coupon") {
        if !p("event").is_undef() {
            query.push(obj! { "type" => foreign.clone(), "table" => "event", "column" => "index", "value" => p("event") })?;
            merge = obj! { "update" => obj! { "includes" => strs(EVENT_FIELDS), "column" => "index", "value" => p("index"), "from" => foreign.clone(), "to" => ptype.clone() } };
        }
    }

    Ok(obj! { "query" => query, "merge" => merge })
}

/// `operators(property, current)`
pub fn operators(property: &V) -> V {
    if is(property, "date") {
        obj! { "eq" => "yyyy-MM-ddThh:mm:ss", "lte" => "yyyy-MM-ddThh:mm:ss", "gte" => "yyyy-MM-ddThh:mm:ss" }
    } else {
        obj! { "eq" => 0, "lte" => 0, "gte" => 0 }
    }
}

/// `paragraph2propertys(task, team, paragraph, current)` — 원본처럼 paragraph 를 고쳐서 돌려준다(유형이 없으면 undefined).
/// paragraph 에 쓰다 던지는 것은 바깥으로(원본의 try 밖), 안쪽 try 의 예외는 기록만
pub fn paragraph2propertys(task: &V, team: &V, paragraph: &V) -> R<V> {
    paragraph.s_("region", task.g("flag")?)?;
    paragraph.s_("status", V::from("#status"))?;
    paragraph.s_("orderBy", V::from("#orderBy"))?;
    paragraph.s_("find", V::from("#find"))?;
    paragraph.s_("substantial", V::from("#substantial"))?;
    paragraph.s_("condition", V::new_obj())?;

    let ptype = paragraph.g("type")?;
    let mut type_ = "";
    if is(&ptype, "tracking") {
        type_ = "tracking";
    } else if is(&ptype, "sales") {
        type_ = "sales";
        paragraph.s_("type", V::from("order"))?;
    } else if is(&ptype, "goods") || is(&ptype, "order") {
        type_ = "sales";
    } else if is(&ptype, "event") || is(&ptype, "coupon") {
        type_ = "event";
    }

    let mut out = paragraph.clone();
    let inner: R<()> = (|| {
        if !type_.is_empty() {
            let condition = paragraph.g("condition")?;
            if condition.g("date")?.truthy() {
                condition.s_("date", merge_node(&condition.g("date")?, &operators(&V::from("date"))))?;
            } else {
                condition.s_("date", operators(&V::from("date")))?;
            }
            for property in CONDITION_PROPERTYS {
                let condition = paragraph.g("condition")?;
                if !condition.g(property)?.truthy() {
                    condition.s_(property, V::new_obj())?;
                }
                let base = team.g("data")?.g("base")?.gk(&paragraph.g("type")?)?.g(property)?;
                if base.g("min")?.truthy() {
                    condition.g(property)?.s_("min", base.g("min")?)?;
                }
                if base.g("max")?.truthy() {
                    condition.g(property)?.s_("max", base.g("max")?)?;
                }
                if !condition.g(property)?.keys_checked()?.is_empty() {
                    condition.s_(property, merge_node(&condition.g(property)?, &operators(&V::from(*property))))?;
                    if *property == "price" {
                        condition.g(property)?.s_("currency", V::from(""))?;
                    }
                } else {
                    condition.del(property)?;
                }
            }
        } else {
            out = V::Undef;
        }
        Ok(())
    })();
    if let Err(err) = inner {
        console_log!("err", err);
    }
    Ok(out)
}

/// `parseCondition(obj, col, condition)`
///
/// 고침: 값과 열 이름을 SQL 글에 그대로 이어 붙였다(LLM 이 낸 값 · 키로 쿼리를 바꿀 수 있었음). 이제 글에 들어가는 값은
/// `Number()`(날짜는 `new Date().getTime()`) 뒤 유한한 수뿐이고(아니면 그 경계는 건너뜀), 열은 [`condition_column`] 의
/// 알려진 이름뿐이다(아니면 조건 없음 — 원본의 "모두 NaN" 갈래처럼 빈 글).
pub fn parse_condition(obj: &V, col: &V, condition: &V) -> R<V> {
    let is_date = is(col, "date");
    let val = |v: V| -> V {
        if is_date {
            V::Num(date::get_time(&v))
        } else {
            v
        }
    };
    let gte = val(obj.g("gte")?);
    let lte = val(obj.g("lte")?);
    let eq = val(obj.g("eq")?);
    let column = if is_date { V::from("created_at") } else { col.clone() };
    if !condition_column(&column.to_str()) {
        return Ok(V::from(""));
    }
    // 글에 넣을 경계: 원본처럼 참인 값만, 그리고 수로 바꿔 유한할 때만
    let bound = |raw: V| -> Option<V> {
        if !raw.truthy() {
            return None;
        }
        let n = val(raw).to_num();
        n.is_finite().then_some(V::Num(n))
    };
    let mut condition = condition.clone();
    if !gte.to_num().is_nan() || !lte.to_num().is_nan() || !eq.to_num().is_nan() {
        let add = |c: &V, s: String| logis_worker_common::add(c, &V::from(s));
        let (g, l, e) = (bound(obj.g("gte")?), bound(obj.g("lte")?), bound(obj.g("eq")?));
        if let (Some(g), Some(l)) = (&g, &l) {
            condition = add(&condition, tpl(T981, &[column.clone(), g.clone(), column.clone(), l.clone()]));
        } else if let Some(g) = g {
            condition = add(&condition, tpl(T983, &[column.clone(), g]));
        } else if let Some(l) = l {
            condition = add(&condition, tpl(T985, &[column.clone(), l]));
        } else if let Some(e) = e {
            condition = add(&condition, tpl(T987, &[column, e]));
        }
    } else {
        condition = V::from("");
    }
    Ok(condition)
}

/// 검색이 SQL 을 만드는 표의 열 (migrations/0001 의 스키마)
const TABLE_COLUMNS: &[(&str, &[&str])] = &[
    ("items", &["id", "type", "from", "to", "cc", "bcc", "ref", "digest", "data", "created_at", "updated_at"]),
    (
        "sales",
        &[
            "id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "started_at", "expired_at", "index", "event", "views", "goods",
            "status", "width", "height", "length", "weight", "size", "currency", "supply_price", "sale_price", "discount", "quantity", "tracking",
            "number", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "fulfillment_service", "stock_keeping_unit",
            "bundle_shipping", "used", "lease", "rental", "refurbish", "tax_included", "release_date",
        ],
    ),
    (
        "event",
        &[
            "id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "started_at", "expired_at", "index", "event", "phone", "address",
            "status", "code", "discount", "quantity", "usage_per", "usage_limit", "min_order_amount", "max_order_amount", "max_discount_amount",
            "new_customer_only", "first_purchase_only", "region_restrictions",
        ],
    ),
    (
        "tracking",
        &[
            "id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "index", "event", "goods", "order", "status", "no",
            "sender_address", "sender_phone", "recipient_address", "recipient_phone", "width", "height", "length", "weight", "carrier",
            "shipping_fee", "shipping_method", "shipping_duration", "shipping_date", "delivery_date", "order_date", "payment_date",
            "payment_method", "payment_origin", "payment_number", "bundle_shipping",
        ],
    ),
    ("talks", &["id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "updated_at"]),
];

/// `paragraph2propertys` 가 조건에 넣는 속성 (원본의 propertys 순서 그대로 — 표에 열이 없는 것도 있다: price 등)
const CONDITION_PROPERTYS: &[&str] = &[
    "price", "quantity", "width", "height", "length", "weight", "shipping_fee", "shipping_duration", "sale_price", "supply_price",
    "low_stock_threshold", "discount", "min_order_amount", "max_discount_amount", "usage_limit", "usage_per", "started_at", "expired_at",
];

/// 표 `table` 의 열 목록 (검색이 쓰는 표가 아니면 None)
pub fn table_columns(table: &str) -> Option<&'static [&'static str]> {
    TABLE_COLUMNS.iter().find(|(t, _)| *t == table).map(|(_, c)| *c)
}

/// 조건 글에 넣어도 되는 열 이름: 검색 표의 열이거나 `paragraph2propertys` 의 속성.
/// 이름은 따옴표 안에 들어가므로 이 목록 밖(LLM 이 지어낸 키)은 받지 않는다
pub fn condition_column(column: &str) -> bool {
    CONDITION_PROPERTYS.contains(&column) || TABLE_COLUMNS.iter().any(|(_, cols)| cols.contains(&column))
}

/// `randomKey()` — `parseInt(Math.random().toString().replace("0.",""))`
pub fn random_key() -> V {
    let s = num_to_string(logis_worker_common::platform::random());
    V::Num(logis_worker_common::jsv::parse_int(&s.replacen("0.", "", 1)))
}

/// `hasUndefinedValue(obj)`
pub fn has_undefined_value(obj: &V) -> bool {
    obj.keys().iter().any(|k| obj.get(k).is_undef())
}

#[allow(dead_code)]
fn _unused() -> V {
    arr![]
}

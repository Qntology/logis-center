//! 아이템 반복의 세 번째 구역 — 원본 4556~5169: 관계(`relates`)를 돌며 이어진 행을 합치고(import) 표 문장을 쌓는다.
//!
//! 안쪽 반복(`for type in relates` · `for d` · `for v`)의 `continue` 는 그 반복의 것이라 바깥 아이템 반복에는 늘 [`Flow::Next`].
//! 원본은 이 구역 안에 try 가 하나뿐(4633)이라 나머지 예외는 `?` 로 바깥(3069 의 try)에 샌다.

use logis_worker_common::gzip::{decode_utf8, gzip_json, ungzip};
use logis_worker_common::js::{self, Env};
use logis_worker_common::jsv::{add, js_trim, loose_eq, parse, parse_float, tpl, Throw};
use logis_worker_common::{console_log, obj, stringify_str, R, V};

use crate::ctx::Ctx;
use crate::flow::Flow;
use crate::handler::{dec, select};
use crate::rules::{parse_status, semantic_prompt_system};
use crate::templates::*;

/// 원본 1211: 임베딩은 늘 앱이 만든다 (서버는 `embed: 0` 표시만)
pub(crate) const CLIENT_EMBEDDING: bool = true;

/// 원본 1213 `markEmbedPending(obj)`
pub(crate) fn mark_embed_pending(obj: &V) {
    if !obj.truthy() || obj.type_of() != "object" {
        return;
    }
    obj.put("embed", V::Num(0.0));
}

/// `x ? x : d`
pub(crate) fn pick(x: V, d: V) -> V {
    if x.truthy() {
        x
    } else {
        d
    }
}

/// `parseFloat(x ? x : 0)`
pub(crate) fn pf(x: V) -> V {
    V::Num(if x.truthy() { parse_float(&x.to_str()) } else { 0.0 })
}

/// `x ? true : false`
fn yes(x: V) -> V {
    V::Bool(x.truthy())
}

/// `obj[key]++` · `obj[key]--` — 수로 더한다 (undefined 면 NaN)
pub(crate) fn bump(obj: &V, key: &str, delta: f64) -> R<()> {
    let n = obj.g(key)?.to_num();
    obj.s_(key, V::Num(n + delta))
}

/// `team.data.base.pages[cc][type]`
pub(crate) fn pages_of(team: &V, cc: &V, type_: &V) -> R<V> {
    team.g("data")?.g("base")?.g("pages")?.gk(cc)?.gk(type_)
}

/// `team.data.base[type]`
pub(crate) fn base_of(team: &V, type_: &V) -> R<V> {
    team.g("data")?.g("base")?.gk(type_)
}

/// `x.tirm()` — 원본 그대로: `trim` 오타라 늘 TypeError
pub(crate) fn tirm(x: &V, name: &str) -> Throw {
    if x.is_nullish() {
        Throw::type_error(format!("Cannot read properties of {} (reading 'tirm')", x.to_str()))
    } else {
        Throw::type_error(format!("{name}.tirm is not a function"))
    }
}

/// `statements[key].push(env[db].prepare(sql).bind(...params))` — 원본의 평가 순서: 칸 → `.push` 읽기 → prepare → 인자 → bind → push
pub(crate) fn push_bound(statements: &V, key: &str, env: &Env, db: &str, sql: &str, params: impl FnOnce() -> R<Vec<V>>) -> R<()> {
    let slot = statements.g(key)?;
    if slot.is_nullish() {
        return Err(Throw::type_error(format!("Cannot read properties of {} (reading 'push')", slot.to_str())));
    }
    let stmt = env.prepare(db, sql)?;
    let params = params()?;
    let bound = js::bind(&stmt, &params)?;
    slot.push(bound)
}

/// `env[name].prepare` 읽기 — 바인딩이 없으면 SQL 템플릿을 계산하기 전에 던진다 (원본의 평가 순서)
fn need_db(env: &Env, name: &str) -> R<()> {
    let db = env.get(name);
    if db.is_nullish() {
        return Err(Throw::type_error(format!("Cannot read properties of {} (reading 'prepare')", db.to_str())));
    }
    Ok(())
}

/// sales 의 41 개 값. head 는 앞 9 칸(id · from · to · cc · bcc · ref · data · created_at), e 는 나머지
pub(crate) fn sales_params(head: &V, type_: V, e: &V, status: V, currency: V) -> Vec<V> {
    let h = |k: &str| head.get(k);
    let f = |k: &str| e.get(k);
    vec![
        h("id"),
        type_,
        h("from"),
        h("to"),
        h("cc"),
        h("bcc"),
        h("ref"),
        h("data"),
        h("created_at"),
        pf(f("started_at")),
        pf(f("expired_at")),
        pf(f("index")),
        pf(f("event")),
        pf(f("views")),
        pf(f("goods")),
        status,
        pf(f("width")),
        pf(f("height")),
        pf(f("length")),
        pf(f("weight")),
        pick(f("size"), V::from("")),
        currency,
        pf(f("supply_price")),
        pf(f("sale_price")),
        pf(f("discount")),
        pf(f("quantity")),
        pf(f("tracking")),
        pick(f("number"), V::from("")),
        pick(f("carrier"), V::from("")),
        pf(f("shipping_fee")),
        pick(f("shipping_method"), V::from("")),
        pf(f("shipping_duration")),
        pick(f("fulfillment_service"), V::from("")),
        pick(f("stock_keeping_unit"), V::from("")),
        pf(f("bundle_shipping")),
        pf(f("used")),
        pf(f("lease")),
        pf(f("rental")),
        pf(f("refurbish")),
        pf(f("tax_included")),
        pf(f("release_date")),
    ]
}

/// tracking 의 35 개 값
pub(crate) fn tracking_params(head: &V, type_: V, e: &V, status: V) -> Vec<V> {
    let h = |k: &str| head.get(k);
    let f = |k: &str| e.get(k);
    let s = |k: &str| pick(e.get(k), V::from(""));
    vec![
        h("id"),
        type_,
        h("from"),
        h("to"),
        h("cc"),
        h("bcc"),
        h("ref"),
        h("data"),
        h("created_at"),
        f("index"),
        pick(f("event"), V::Num(0.0)),
        pick(f("goods"), V::Num(0.0)),
        pick(f("order"), V::Num(0.0)),
        status,
        s("no"),
        s("sender_address"),
        s("sender_phone"),
        s("recipient_address"),
        s("recipient_phone"),
        pf(f("width")),
        pf(f("height")),
        pf(f("length")),
        pf(f("weight")),
        pf(f("carrier")),
        pf(f("shipping_fee")),
        s("shipping_method"),
        pf(f("shipping_duration")),
        pf(f("shipping_date")),
        pf(f("delivery_date")),
        pf(f("order_date")),
        pf(f("payment_date")),
        s("payment_method"),
        s("payment_origin"),
        s("payment_number"),
        pf(f("bundle_shipping")),
    ]
}

/// event 의 27 개 값
pub(crate) fn event_params(head: &V, type_: V, e: &V, status: V) -> Vec<V> {
    let h = |k: &str| head.get(k);
    let f = |k: &str| e.get(k);
    let s = |k: &str| pick(e.get(k), V::from(""));
    vec![
        h("id"),
        type_,
        h("from"),
        h("to"),
        h("cc"),
        h("bcc"),
        h("ref"),
        h("data"),
        h("created_at"),
        pf(f("started_at")),
        pf(f("expired_at")),
        pf(f("index")),
        pf(f("event")),
        s("number"),
        s("address"),
        status,
        s("code"),
        pf(f("discount")),
        pf(f("quantity")),
        pf(f("usage_per")),
        pf(f("usage_limit")),
        pf(f("min_order_amount")),
        pf(f("max_order_amount")),
        pf(f("max_discount_amount")),
        pf(f("new_customer_only")),
        pf(f("first_purchase_only")),
        pf(f("region_restrictions")),
    ]
}

/// 4556~5169
pub async fn run(ctx: &mut Ctx, env: &Env) -> R<Flow> {
    if !ctx.relates.keys_checked()?.is_empty() {
        // for (var type in relates)
        for key in ctx.relates.keys() {
            ctx.type_ = V::from(key.clone());
            if ctx.relates.has(&key) {
                relate_one(ctx, env).await?;
            }
        }
    }

    console_log!("typeof item.vectorize", ctx.item.g("vectorize")?.type_of());
    Ok(Flow::Next)
}

/// 4562~5158: 관계 하나
async fn relate_one(ctx: &mut Ctx, env: &Env) -> R<()> {
    ctx.relate = ctx.relates.gk(&ctx.type_)?;
    ctx.query = ctx.relate.g("query")?;

    if !ctx.relate.truthy() || ctx.relate.g("rows")?.is_undef() {
        return Ok(());
    }
    // query[1] ? query[1].column : query[0].column (식마다 query[1] 을 다시 읽는다)
    ctx.column = if ctx.query.gi(1)?.truthy() { ctx.query.gi(1)?.g("column")? } else { ctx.query.gi(0)?.g("column")? };
    ctx.index = if ctx.query.gi(1)?.truthy() { ctx.query.gi(1)?.g("index")? } else { ctx.query.gi(0)?.g("index")? };

    if ctx.relate.g("rows")?.length()?.truthy() {
        ctx.d = V::Num(0.0);
        while ctx.d.to_num() < ctx.relate.g("rows")?.length()?.to_num() {
            relate_row(ctx, env).await?;
            ctx.d = V::Num(ctx.d.to_num() + 1.0);
        }
    } else {
        relate_empty(ctx, env).await?;
    }
    Ok(())
}

/// 4590~5075: 이어진 행 하나. 일찍 돌아가는 것이 원본의 `continue`(d 반복)
async fn relate_row(ctx: &mut Ctx, env: &Env) -> R<()> {
    ctx.row = ctx.relate.g("rows")?.gk(&ctx.d)?;
    ctx.nodeData = ctx.row.g("data")?;

    // 원본 그대로: `var merge` 는 값을 지우지 않는다 — update · upsert 가 없으면 앞 회차(또는 4471 Relay)의 merge 가 남는다
    if ctx.relate.g("merge")?.g("update")?.truthy() {
        ctx.merge = ctx.relate.g("merge")?.g("update")?;
    }
    if ctx.relate.g("merge")?.g("upsert")?.truthy() {
        ctx.merge = ctx.relate.g("merge")?.g("upsert")?;
    }
    if !ctx.merge.truthy() {
        return Ok(());
    }

    ctx.foreign = ctx.merge.g("foreign")?;
    ctx.node = V::new_obj();
    ctx.from = if loose_eq(&ctx.row.g("type")?, &ctx.merge.g("from")?) { ctx.item.clone() } else { ctx.row.clone() };
    ctx.to = if loose_eq(&ctx.row.g("type")?, &ctx.merge.g("to")?) { ctx.item.clone() } else { ctx.row.clone() };
    if loose_eq(&ctx.row.g("type")?, &ctx.merge.g("to")?) {
        ctx.to.s_("updated_at", ctx.updated_at.clone())?;
    }

    if ctx.merge.g("includes")?.truthy() && ctx.merge.g("includes")?.length()?.truthy() {
        ctx.v = V::Num(0.0);
        while ctx.v.to_num() < ctx.merge.g("includes")?.length()?.to_num() {
            ctx.include = ctx.merge.g("includes")?.gk(&ctx.v)?;
            let val = ctx.from.gk(&ctx.include)?;
            ctx.node.sk(&ctx.include, val)?;
            ctx.v = V::Num(ctx.v.to_num() + 1.0);
        }
    }

    // try { … } catch (err) { console.log('relate err', err) } — 실패하면 data 는 앞의 값 그대로
    let r: R<()> = (|| {
        let bytes = ungzip(&ctx.from.g("data")?)?;
        ctx.decompressedJsonString = V::from(decode_utf8(&bytes));
        ctx.data = parse(&ctx.decompressedJsonString)?;
        Ok(())
    })();
    if let Err(err) = r {
        console_log!("relate err", err);
    }

    if ctx.foreign.truthy() && ctx.foreign.g("from")?.truthy() && ctx.foreign.g("to")?.truthy() && ctx.from.gk(&ctx.foreign.g("to")?)?.truthy() {
        let key = ctx.foreign.g("from")?;
        let val = ctx.from.gk(&ctx.foreign.g("to")?)?;
        ctx.to.sk(&key, val)?;
    }

    ctx.edgeId = ctx.to.g("id")?;
    ctx.edgeType = ctx.to.g("type")?;
    ctx.edge = logis_worker_common::util::merge_node(&ctx.to, &ctx.node);
    ctx.edge.s_("id", ctx.edgeId.clone())?;
    ctx.edge.s_("type", ctx.edgeType.clone())?;

    if loose_eq(&ctx.to.g("id")?, &ctx.item.g("id")?) {
        ctx.item = ctx.edge.clone();
    }

    if loose_eq(&ctx.relate.g("type")?, &ctx.merge.g("from")?) {
        // import
        if loose_eq(&ctx.from.g("type")?, &V::from("goods")) && loose_eq(&ctx.to.g("type")?, &V::from("order")) {
            let edge = ctx.edge.clone();
            let packed = obj! {
                "id" => edge.get("id"),
                "no" => pick(edge.get("no"), V::from("")),
                "title" => edge.get("title"),
                "link" => edge.get("link"),
                "time" => ctx.now.clone(),
                "origin" => pick(ctx.data.g("origin")?, V::from("")),
                "data" => ctx.data.clone(),
            };
            ctx.arr = gzip_json(&packed)?;

            let row = ctx.row.clone();
            let r = |k: &str| row.get(k);
            ctx.metadata = obj! {
                "title" => pick(ctx.data.g("title")?, V::from("")),
                "size" => pick(r("size"), V::from("")),
                "currency" => pick(r("currency"), V::from("")),
                "carrier" => pick(r("carrier"), V::from("")),
                "shipping_fee" => yes(r("shipping_fee")),
                "shipping_method" => pick(r("shipping_method"), V::from("")),
                "fulfillment_service" => pick(r("fulfillment_service"), V::from("")),
                "stock_keeping_unit" => pick(r("stock_keeping_unit"), V::from("")),
                "bundle_shipping" => yes(r("bundle_shipping")),
                "used" => yes(r("used")),
                "lease" => yes(r("lease")),
                "rental" => yes(r("rental")),
                "refurbish" => yes(r("refurbish")),
                "tax_included" => yes(r("tax_included")),
            };
            ctx.content = V::from(stringify_str(&ctx.metadata));

            // 원본 그대로: `var semantic` 도 값을 지우지 않는다
            ctx.system = V::from(js_trim(&semantic_prompt_system(&ctx.language).to_str()).to_string());

            if ctx.models.g("deepinfra")?.truthy() {
                ctx.semantic = crate::llm::deepinfra(
                    &ctx.deepinfra,
                    &V::from("openai/gpt-oss-20b"),
                    &ctx.system,
                    &ctx.content,
                    &obj! { "temperature" => 1 },
                    &V::Undef,
                )
                .await?;
                ctx.models.s_("deepinfra", dec(&ctx.models.g("deepinfra")?))?;
            }

            if !ctx.semantic.truthy() && ctx.gemini_llm_api.truthy() {
                ctx.semantic = crate::llm::gemini(
                    &ctx.gemini_llm_api,
                    &ctx.gemini_llm_model,
                    &ctx.system,
                    &ctx.content,
                    &obj! { "temperature" => 1 },
                    &V::Undef,
                )
                .await?;
                let key = add(&add(&ctx.gemini_llm_api, &V::from("-")), &ctx.gemini_llm_model);
                ctx.models.sk(&key, dec(&ctx.models.gk(&key)?))?;
            }

            if !ctx.semantic.truthy() {
                ctx.fallback = V::from("semantic overflow");
                return Ok(());
            }

            let metadata = ctx.metadata.clone();
            metadata.s_("id", edge.get("id"))?;
            metadata.s_("type", edge.get("type"))?;
            metadata.s_("from", ctx.task.g("from")?)?;
            metadata.s_("to", ctx.task.g("to")?)?;
            metadata.s_("cc", ctx.task.g("cc")?)?;
            metadata.s_("bcc", edge.get("bcc"))?;
            metadata.s_("ref", edge.get("ref"))?;

            if CLIENT_EMBEDDING {
                mark_embed_pending(&edge);
                console_log!("[EMBED-PENDING] relay edge delegated to Client App:", edge.get("id"));
            } else {
                // `var embeddings` · `var $VectorizeVector` 도 값을 지우지 않는다
                if ctx.models.g("cloudflare")?.truthy() {
                    let out = env.ai_run("@cf/google/embeddinggemma-300m", &obj! { "text" => vec![ctx.semantic.clone()] }).await?;
                    ctx.embeddings = out.g("data")?;
                    ctx.s_VectorizeVector = V::from(vec![obj! {
                        "id" => edge.get("id"),
                        "values" => ctx.embeddings.gi(0)?,
                        "metadata" => metadata.clone(),
                    }]);
                    ctx.models.s_("cloudflare", dec(&ctx.models.g("cloudflare")?))?;
                }
                if !ctx.embeddings.truthy() && ctx.models.g("deepinfra")?.truthy() {
                    return Err(tirm(&ctx.semantic, "semantic"));
                }
                // 원본 그대로: 조건이 뒤집혀 있다 (임베딩이 있으면 건너뜀)
                if ctx.embeddings.truthy() {
                    ctx.fallback = V::from("embeddings overflow");
                    return Ok(());
                }
                let name = tpl(T4788, &[ctx.vectorRegion.clone(), ctx.type_.clone()]);
                env.vectorize_upsert(&name, &ctx.s_VectorizeVector).await?;
            }
        }
    } else {
        // export
    }

    // 원본 그대로: try 밖이라 to 가 item 이면(data 가 아직 객체) ungzip 이 던져 스캔 전체가 멈춘다
    let bytes = ungzip(&ctx.edge.g("data")?)?;
    ctx.decompressedJsonString = V::from(decode_utf8(&bytes));
    let parsed = parse(&ctx.decompressedJsonString)?;
    ctx.edge.s_("data", parsed)?;

    let stamp = if ctx.edge.g("data")?.g("item")?.truthy() { V::Num(0.0) } else { ctx.now.clone() };
    ctx.edge.s_("updated_at", stamp)?;

    let zone = ctx.zoneRegion.clone();
    let (edge, to) = (ctx.edge.clone(), ctx.to.clone());
    if loose_eq(&ctx.edgeType, &V::from("sales")) {
        let db = tpl(T4808, &[zone.clone()]);
        let edge_type = ctx.edgeType.clone();
        push_bound(&ctx.statements, &tpl(T4807, &[zone.clone()]), env, &db, T4808_1[0], || {
            Ok(sales_params(&to, edge_type, &edge, parse_status(&edge.get("status")), edge.get("currency")))
        })?;
    } else if loose_eq(&ctx.edgeType, &V::from("tracking")) {
        let db = tpl(T4900, &[zone.clone()]);
        push_bound(&ctx.statements, &tpl(T4899, &[zone.clone()]), env, &db, T4900_1[0], || {
            Ok(tracking_params(&to, to.get("type"), &edge, parse_status(&edge.get("status"))))
        })?;
    } else if loose_eq(&ctx.edgeType, &V::from("event")) {
        let db = tpl(T4980, &[zone.clone()]);
        let edge_type = ctx.edgeType.clone();
        push_bound(&ctx.statements, &tpl(T4979, &[zone.clone()]), env, &db, T4980_1[0], || {
            Ok(event_params(&to, edge_type, &edge, parse_status(&edge.get("status"))))
        })?;
    }

    let db = tpl(T5046, &[zone.clone()]);
    let sql = tpl(T5046_1, &[edge.get("id"), ctx.now.clone()]);
    ctx.results = select(env, &db, &sql).await?;

    if ctx.results.length()?.truthy() {
        ctx._item = ctx.results.gi(0)?;
        if !ctx._item.g("updated_at")?.truthy() && edge.get("updated_at").truthy() {
            bump(&pages_of(&ctx.team, &edge.g("cc")?, &edge.g("type")?)?, "draft", -1.0)?;
            bump(&pages_of(&ctx.team, &edge.g("cc")?, &edge.g("type")?)?, "count", 1.0)?;

            let db = tpl(T5058, &[zone.clone()]);
            let (now, row) = (ctx.now.clone(), ctx.row.clone());
            push_bound(&ctx.statements, &tpl(T5057, &[zone]), env, &db, T5058_1[0], || Ok(vec![now, row.g("id")?]))?;
        }
    } else {
        bump(&pages_of(&ctx.team, &edge.g("cc")?, &edge.g("type")?)?, "draft", 1.0)?;
        bump(&base_of(&ctx.team, &edge.g("type")?)?, "count", 1.0)?;
    }
    Ok(())
}

/// 5079~5156: 이어진 행이 없을 때. 원본 그대로: edgeType · edge 는 앞 회차(앞 아이템)의 값이고, 처음이면 undefined 라
/// `commerce_logis_…_undefined` 바인딩이 없어 던진다
async fn relate_empty(ctx: &mut Ctx, env: &Env) -> R<()> {
    let zone = ctx.zoneRegion.clone();
    let db = tpl(T5087, &[zone.clone(), ctx.edgeType.clone()]);
    need_db(env, &db)?;
    let sql = tpl(T5087_1, &[ctx.edgeType.clone(), ctx.edge.g("index")?, ctx.edge.g("type")?, ctx.now.clone()]);
    ctx.results = select(env, &db, &sql).await?;

    if ctx.results.length()?.truthy() {
        ctx._item = ctx.results.gi(0)?;
        if !ctx._item.g("updated_at")?.truthy() && ctx.updated_at.truthy() {
            let edge = ctx.edge.clone();
            bump(&pages_of(&ctx.team, &edge.g("cc")?, &edge.g("type")?)?, "draft", -1.0)?;
            bump(&pages_of(&ctx.team, &edge.g("cc")?, &edge.g("type")?)?, "count", 1.0)?;
            bump(&base_of(&ctx.team, &edge.g("type")?)?, "count", 1.0)?;

            let db = tpl(T5099, &[zone.clone()]);
            let (now, item) = (ctx.now.clone(), ctx.item.clone());
            push_bound(&ctx.statements, &tpl(T5098, &[zone]), env, &db, T5099_1[0], || Ok(vec![now, item.g("id")?]))?;
        }
    } else {
        let edge = ctx.edge.clone();
        bump(&pages_of(&ctx.team, &edge.g("cc")?, &edge.g("type")?)?, "draft", 1.0)?;
        bump(&base_of(&ctx.team, &edge.g("type")?)?, "count", 1.0)?;

        let item = ctx.item.clone();
        let packed = obj! {
            "id" => edge.g("id")?,
            "no" => pick(item.g("no")?, V::from("")),
            "title" => item.g("title")?,
            "link" => item.g("link")?,
            "time" => ctx.now.clone(),
            "data" => ctx.relate.clone(),
        };
        ctx.arr = gzip_json(&packed)?;

        let db = tpl(T5121, &[zone.clone()]);
        let (now, type_, arr) = (ctx.now.clone(), ctx.type_.clone(), ctx.arr.clone());
        push_bound(&ctx.statements, &tpl(T5120, &[zone]), env, &db, T5121_1[0], || {
            Ok(vec![
                edge.g("id")?,
                type_,
                item.g("from")?,
                item.g("to")?,
                item.g("cc")?,
                item.g("bcc")?,
                V::from(""),
                item.g("digest")?,
                arr,
                now,
                V::Num(0.0),
            ])
        })?;
    }
    Ok(())
}

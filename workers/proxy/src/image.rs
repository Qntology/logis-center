//! 운송장 이미지 (원본 2543~3025): 이미지를 LLM 으로 읽어 tracking · items · sales 문장을 쌓는다.

use logis_worker_common::gzip::{decode_utf8, gzip_json, ungzip};
use logis_worker_common::js::{self, Env};
use logis_worker_common::jsv::{less_than, parse_float, parse_str, tpl};
use logis_worker_common::util::{clean_number, crc32, digest, hash_id, merge_node, normalize_numeric_homoglyphs};
use logis_worker_common::{add, arr, console_log, loose_eq, obj, Throw, R, V};

use crate::ctx::Ctx;
use crate::flow::Flow;
use crate::handler::{dec, select};
use crate::llm;
use crate::rules::{image2json, parse_status};
use crate::templates::*;

/// 원본의 `CLIENT_EMBEDDING` (서버는 벡터를 만들지 않는다)
const CLIENT_EMBEDDING: bool = true;

fn is(v: &V, s: &str) -> bool {
    loose_eq(v, &V::from(s))
}

/// `x ? x : d`
fn or(v: V, d: impl Into<V>) -> V {
    if v.truthy() {
        v
    } else {
        d.into()
    }
}

/// `x ? parseFloat(x) : 0`
fn pf(v: V) -> V {
    if v.truthy() {
        V::Num(parse_float(&v.to_str()))
    } else {
        V::Num(0.0)
    }
}

/// `statements[key]` — push 의 인자보다 먼저 읽는다(없으면 원본처럼 TypeError)
fn slot(statements: &V, key: &str) -> R<V> {
    let s = statements.g(key)?;
    if s.is_nullish() {
        return Err(Throw::type_error(format!("Cannot read properties of {} (reading 'push')", s.to_str())));
    }
    Ok(s)
}

/// `models[key] -= 1`
fn models_dec(models: &V, key: &V) -> R<()> {
    let v = models.gk(key)?;
    models.sk(key, dec(&v))
}

/// `o.key++` · `o.key--` (o 가 undefined 면 원본처럼 던진다)
fn step(o: &V, key: &str, d: f64) -> R<()> {
    let v = o.g(key)?;
    o.s_(key, V::Num(v.to_num() + d))
}

/// `team.data.base.pages[task.cc]`
fn page_cc(team: &V, task: &V) -> R<V> {
    team.g("data")?.g("base")?.g("pages")?.gk(&task.g("cc")?)
}

/// `decompressedJsonString = …decode(ungzip(v)); JSON.parse(decompressedJsonString)`
fn unzip(ctx: &mut Ctx, v: &V) -> R<V> {
    let s = decode_utf8(&ungzip(v)?);
    ctx.decompressedJsonString = V::from(s.clone());
    parse_str(&s)
}

/// `markEmbedPending(obj)`
fn mark_embed_pending(o: &V) {
    if o.truthy() && o.type_of() == "object" {
        o.put("embed", V::Num(0.0));
    }
}

pub async fn run(ctx: &mut Ctx, env: &Env) -> R<Flow> {
    let task = ctx.task.clone();
    let team = ctx.team.clone();
    let talk = ctx.talk.clone();
    let models = ctx.models.clone();
    let now = ctx.now.clone();
    let zone = ctx.zoneRegion.clone();

    ctx.inlineData = obj! { "mimeType" => task.g("contentType")?, "data" => task.g("body")? };

    // var type = talk.type = task.type
    let t = task.g("type")?;
    talk.s_("type", t.clone())?;
    ctx.type_ = t;

    ctx.address = V::new_arr();
    if ctx.owner.g("data")?.truthy() && ctx.owner.g("data")?.g("sender")?.truthy() {
        ctx.address.push(ctx.owner.g("data")?.g("sender")?)?;
    }

    // 추후 꼭 반영해야함 (원본 주석: 프론트 백엔드 미개발)
    if team.g("data")?.truthy() && team.g("data")?.g("address")?.truthy() {
        ctx.a = V::Num(0.0);
        while less_than(&ctx.a, &team.g("data")?.g("address")?.length()?) == Some(true) {
            ctx.addr = team.g("data")?.g("address")?.gk(&ctx.a)?;
            ctx.address.push(ctx.addr.clone())?;
            ctx.a = V::Num(ctx.a.to_num() + 1.0);
        }
    }

    ctx.system = image2json(&task.g("flag")?, &ctx.language, &ctx.type_, &ctx.address);

    // 원본 그대로: `var item` 은 값을 지우지 않는다 — 두 모델이 다 꺼져 있으면 앞 회차의 item 이 남는다
    if models.g("deepinfra")?.truthy() {
        ctx.item = llm::deepinfra(&ctx.deepinfra, &V::from("Qwen/Qwen3.5-0.8B"), &ctx.system, &V::from(""), &V::Null, &ctx.inlineData).await?;
        models_dec(&models, &V::from("deepinfra"))?;
    }

    if !ctx.item.truthy() && ctx.gemini_llm_api.truthy() {
        // 원본 그대로: config 가 null 이라 Gemini 안의 config["response_mime_type"] 에서 던진다
        ctx.item = llm::gemini(&ctx.gemini_llm_api, &ctx.gemini_llm_model, &V::from(""), &ctx.system, &V::Null, &ctx.inlineData).await?;
        let key = add(&add(&ctx.gemini_llm_api, &V::from("-")), &ctx.gemini_llm_model);
        models_dec(&models, &key)?;
    }

    if !ctx.item.truthy() {
        ctx.fallback = V::from("overflow");
        return Ok(Flow::Continue);
    }

    if !ctx.item.g("tracking_number")?.truthy() {
        ctx.fallback = V::from("ShippingLabel Not Found");
        // 올바르지 않은 이미지 안내하기
        return Ok(Flow::Continue);
    }

    talk.s_("text", ctx.item.g("text")?)?;

    ctx.arr = gzip_json(&ctx.item)?;
    ctx.item.s_("data", ctx.arr.clone())?;

    ctx.item.s_("type", ctx.type_.clone())?;
    ctx.item.s_("no", normalize_numeric_homoglyphs(&ctx.item.g("tracking_number")?))?;
    ctx.item.s_("no", clean_number(&ctx.item.g("no")?)?)?;

    let key = add(&add(&ctx.item.g("type")?, &team.g("id")?), &ctx.item.g("no")?);
    ctx.item.s_("index", V::Num(crc32(&hash_id(&key)?)?))?;
    let key = add(&team.g("id")?, &ctx.item.g("index")?);
    ctx.item.s_("id", V::from(hash_id(&key)?))?;

    ctx.item.s_("digest", V::from(""))?;
    if ctx.item.g("title")?.truthy() {
        ctx.item.s_("digest", V::from(digest(&ctx.item.g("title")?)?))?;
    }

    ctx.item.s_("from", task.g("from")?)?;
    ctx.item.s_("to", task.g("to")?)?;
    ctx.item.s_("cc", task.g("cc")?)?; // commerce.logis.center 잡혀져 있음
    ctx.item.s_("bcc", task.g("bcc")?)?;
    ctx.item.s_("ref", task.g("ref")?)?;
    ctx.item.s_("created_at", now.clone())?;

    // var statusCode = item.status = item.status ? parseStatus(item.status) : 0
    let st = if ctx.item.g("status")?.truthy() { parse_status(&ctx.item.g("status")?) } else { V::Num(0.0) };
    ctx.item.s_("status", st.clone())?;
    ctx.statusCode = st;

    let sql = tpl(T2658_1, &[ctx.item.g("index")?, task.g("to")?, now.clone()]);
    ctx.results = select(env, &tpl(T2658, &[zone.clone()]), &sql).await?;
    ctx.sales = ctx.results.clone();

    if ctx.sales.length()?.truthy() {
        ctx._item = ctx.results.gi(0)?.deep_clone();

        // item.cc = task.cc = _item.cc (task 도 바뀐다)
        let cc = ctx._item.g("cc")?;
        task.s_("cc", cc.clone())?;
        ctx.item.s_("cc", cc)?;
        let r = ctx._item.g("ref")?;
        task.s_("ref", r.clone())?;
        ctx.item.s_("ref", r)?;

        for k in ["id", "type", "from", "to", "cc", "bcc", "data", "created_at", "ref"] {
            ctx._item.del(k)?;
        }

        if ctx.item.g("status")?.truthy() && ctx._item.g("status")?.truthy() {
            ctx._item.del("status")?;
        }

        ctx.item = merge_node(&ctx.item, &ctx._item);
    }

    if is(&ctx.type_, "tracking") {
        let sql = tpl(T2691_1, &[ctx.item.g("id")?, task.g("to")?, now.clone()]);
        ctx.results = select(env, &tpl(T2691, &[zone.clone()]), &sql).await?;

        let ty = ctx.type_.clone();
        let pages = team.g("data")?.g("base")?.g("pages")?;
        if !pages.gk(&task.g("cc")?)?.truthy() {
            team.g("data")?.g("base")?.g("pages")?.sk(&task.g("cc")?, V::new_obj())?;
        }
        if !page_cc(&team, &task)?.gk(&ty)?.truthy() {
            page_cc(&team, &task)?.sk(&ty, obj! { "draft" => 0, "count" => 0 })?;
        }
        if !team.g("data")?.g("base")?.gk(&ty)?.truthy() {
            team.g("data")?.g("base")?.sk(&ty, obj! { "draft" => 0, "count" => 0 })?;
        }

        if ctx.results.length()?.truthy() {
            ctx._tracking = ctx.results.gi(0)?;
            ctx._tracking.del("ref")?;

            if ctx.item.g("status")?.truthy() && ctx._tracking.g("status")?.truthy() {
                ctx._tracking.del("status")?;
            }

            ctx.item = merge_node(&ctx.item, &ctx._tracking);

            if ctx.sales.length()?.truthy() {
                ctx._order = ctx.sales.gi(0)?;
                ctx.item.s_("order", ctx._order.g("index")?)?;

                let s = slot(&ctx.statements, &tpl(T2730, &[zone.clone()]))?;
                let stmt = env.prepare(&tpl(T2731, &[zone.clone()]), T2731_1[0])?;
                let stmt = js::bind(&stmt, &[now.clone(), ctx.item.g("status")?, ctx.item.g("index")?, ctx._order.g("id")?])?;
                s.push(stmt)?;

                // 원본 그대로: pages[cc].order 가 없으면 여기서 던진다
                step(&page_cc(&team, &task)?.g("order")?, "draft", -1.0)?;
                step(&page_cc(&team, &task)?.g("order")?, "count", 1.0)?;

                step(&page_cc(&team, &task)?.g("tracking")?, "draft", -1.0)?;
                step(&page_cc(&team, &task)?.g("tracking")?, "count", 1.0)?;
            }
        } else if ctx.sales.length()?.truthy() {
            step(&page_cc(&team, &task)?.g("tracking")?, "draft", -1.0)?;
            step(&page_cc(&team, &task)?.g("tracking")?, "count", 1.0)?;
        } else {
            step(&page_cc(&team, &task)?.g("tracking")?, "draft", 1.0)?;
            step(&team.g("data")?.g("base")?.gk(&ctx.type_)?, "count", 1.0)?;
            // items 추가 해야함
        }

        ctx.item.s_("ref", task.g("ref")?)?;
        ctx.item.s_("created_at", now.clone())?;

        if ctx.sales.length()?.truthy() {
            ctx._item = ctx.sales.gi(0)?;
            let data = ctx._item.g("data")?;
            let parsed = unzip(ctx, &data)?;
            ctx._data = parsed;

            ctx.item.s_("order", ctx._item.g("index")?)?;

            let s = slot(&ctx.statements, &tpl(T2771, &[zone.clone()]))?;
            let stmt = env.prepare(&tpl(T2772, &[zone.clone()]), T2772_1[0])?;
            let it = ctx._item.clone();
            let stmt = js::bind(
                &stmt,
                &[
                    it.g("id")?,
                    it.g("type")?,
                    it.g("from")?,
                    it.g("to")?,
                    it.g("cc")?,
                    it.g("bcc")?,
                    ctx.item.g("id")?,
                    it.g("digest")?,
                    it.g("data")?,
                    ctx._data.g("time")?,
                    now.clone(),
                ],
            )?;
            s.push(stmt)?;
        } else {
            ctx.content = V::new_obj();
            for k in ["title", "sender_address", "recipient_address", "carrier", "shipping_method", "fulfillment_service"] {
                if ctx.item.g(k)?.truthy() {
                    ctx.content.s_(k, ctx.item.g(k)?)?;
                }
            }

            let t = if ctx.item.g("recipient_match")?.truthy() { "receiving" } else { "shipping" };
            ctx.item.s_("type", V::from(t))?;

            // 추후에 반품인지 아닌지 추가해야함
            ctx.arr = gzip_json(&obj! {
                "no" => or(ctx.item.g("no")?, ""),
                "type" => ctx.item.g("type")?,
                "text" => ctx.item.g("text")?,
                "link" => V::Null,
                "time" => now.clone(),
            })?;
            ctx.item.s_("data", ctx.arr.clone())?;

            ctx.metadata = obj! {
                "id" => ctx.item.g("id")?,
                "no" => or(ctx.item.g("no")?, ""),
                "type" => ctx.item.g("type")?,
                "from" => task.g("from")?,
                "to" => task.g("to")?,
                "cc" => task.g("cc")?,
                "bcc" => task.g("bcc")?,
                "ref" => or(task.g("ref")?, ""),
            };

            if CLIENT_EMBEDDING {
                mark_embed_pending(&ctx.item);
                console_log!("[EMBED-PENDING] tracking item delegated to Client App:", ctx.item.g("id")?);
            } else {
                // 원본 그대로: `var embeddings` 는 지우지 않는다 (CLIENT_EMBEDDING 이 참이라 지금은 안 탄다)
                if models.g("cloudflare")?.truthy() {
                    let r = env.ai_run("@cf/google/embeddinggemma-300m", &obj! { "text" => vec![ctx.item.g("text")?] }).await?;
                    ctx.embeddings = r.g("data")?;
                    ctx.s_VectorizeVector = arr![obj! {
                        "id" => ctx.item.g("id")?,
                        "values" => ctx.embeddings.gi(0)?,
                        "metadata" => ctx.metadata.clone(),
                    }];
                    models_dec(&models, &V::from("cloudflare"))?;
                }

                if !ctx.embeddings.truthy() && models.g("deepinfra")?.truthy() {
                    ctx.embeddings = llm::deepinfra(
                        &ctx.deepinfra,
                        &V::from("google/embeddinggemma-300m"),
                        &V::from(""),
                        &ctx.item.g("text")?,
                        &V::Undef,
                        &V::Undef,
                    )
                    .await?;
                    // embeddings.map((values, i) => ({ id: item.id, values, metadata }))
                    let list = map_items(&ctx.embeddings)?;
                    let mut out = Vec::with_capacity(list.len());
                    for values in list {
                        out.push(obj! { "id" => ctx.item.g("id")?, "values" => values, "metadata" => ctx.metadata.clone() });
                    }
                    ctx.s_VectorizeVector = V::from(out);
                    models_dec(&models, &V::from("deepinfra"))?;
                }

                if !ctx.embeddings.truthy() {
                    ctx.fallback = V::from("overflow");
                    return Ok(Flow::Continue);
                }

                env.vectorize_upsert(&tpl(T2908, &[ctx.vectorRegion.clone(), ctx.type_.clone()]), &ctx.s_VectorizeVector).await?;
            }

            // 원본 그대로: item 이 아니라 task 의 칸을 묶는다 (task.digest 는 undefined)
            let s = slot(&ctx.statements, &tpl(T2911, &[zone.clone()]))?;
            let stmt = env.prepare(&tpl(T2912, &[zone.clone()]), T2912_1[0])?;
            let stmt = js::bind(
                &stmt,
                &[
                    task.g("id")?,
                    task.g("type")?,
                    task.g("from")?,
                    task.g("to")?,
                    task.g("cc")?,
                    task.g("bcc")?,
                    task.g("ref")?,
                    task.g("digest")?,
                    ctx.arr.clone(),
                    now.clone(),
                    V::Num(0.0),
                ],
            )?;
            s.push(stmt)?;
        }

        let s = slot(&ctx.statements, &tpl(T2944, &[zone.clone()]))?;
        let stmt = env.prepare(&tpl(T2945, &[zone.clone()]), T2945_1[0])?;
        let it = ctx.item.clone();
        let params = [
            it.g("id")?,
            it.g("type")?,
            it.g("from")?,
            it.g("to")?,
            it.g("cc")?,
            it.g("bcc")?,
            it.g("ref")?,
            it.g("data")?,
            it.g("created_at")?,
            it.g("index")?,
            or(it.g("event")?, 0),
            or(it.g("goods")?, 0),
            or(it.g("order")?, 0),
            it.g("status")?,
            it.g("no")?,
            or(it.g("sender_address")?, ""),
            or(it.g("sender_phone")?, ""),
            or(it.g("recipient_address")?, ""),
            or(it.g("recipient_phone")?, ""),
            pf(it.g("width")?),
            pf(it.g("height")?),
            pf(it.g("length")?),
            pf(it.g("weight")?),
            pf(it.g("carrier")?),
            pf(it.g("shipping_fee")?),
            or(it.g("shipping_method")?, ""),
            pf(it.g("shipping_duration")?),
            pf(it.g("shipping_date")?),
            pf(it.g("delivery_date")?),
            pf(it.g("order_date")?),
            pf(it.g("payment_date")?),
            or(it.g("payment_method")?, ""),
            or(it.g("payment_origin")?, ""),
            or(it.g("payment_number")?, ""),
            pf(it.g("bundle_shipping")?),
        ];
        s.push(js::bind(&stmt, &params)?)?;
    }

    Ok(Flow::Next)
}

/// `x.map(...)` 의 원소들 — 배열이 아니면 원본처럼 TypeError
fn map_items(v: &V) -> R<Vec<V>> {
    match v {
        V::Arr(_) => Ok(v.items()),
        V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'map')", v.to_str()))),
        _ => Err(Throw::type_error("embeddings.map is not a function")),
    }
}

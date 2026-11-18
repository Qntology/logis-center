//! 아이템 반복의 마지막 구역 — 원본 5170~5516: 임베딩 대기 표시 · item.data 압축 · items 와 유형 표(sales · tracking · event) 문장.
//!
//! 이 구역에는 try 가 없어 예외는 모두 `?` 로 바깥(3069 의 try)에 샌다. 5228 의 `continue` 는 아이템 반복의 것([`Flow::Continue`]).

use logis_worker_common::gzip::gzip_json;
use logis_worker_common::js::Env;
use logis_worker_common::jsv::{loose_eq, stringify_checked, tpl};
use logis_worker_common::{console_log, obj, R, V};

use crate::ctx::Ctx;
use crate::flow::Flow;
use crate::handler::dec;
use crate::item3::{event_params, event_sql, map_vectors, mark_embed_pending, pick, push_bound, sales_params, tracking_params, trim_of, CLIENT_EMBEDDING};
use crate::templates::*;

/// 5170~5516
pub async fn run(ctx: &mut Ctx, env: &Env) -> R<Flow> {
    let item = ctx.item.clone();

    if item.g("semantic")?.truthy() && !item.g("vectorize")?.truthy() {
        ctx.metadata = obj! {
            "type" => item.g("type")?,
            "from" => item.g("from")?,
            "to" => item.g("to")?,
            "cc" => item.g("cc")?,
            "bcc" => item.g("bcc")?,
            "ref" => ctx.pageId.clone(),
        };

        if CLIENT_EMBEDDING {
            mark_embed_pending(&item);
            let data = item.g("data")?;
            if data.truthy() && data.type_of() == "object" {
                mark_embed_pending(&item.g("data")?);
            }
            console_log!("[EMBED-PENDING] item delegated to Client App:", item.g("id")?);
        } else {
            // 고침: `var embeddings` 가 앞 아이템의 값을 남겼다 — 아이템마다 비운다
            ctx.embeddings = V::Undef;
            if ctx.models.g("cloudflare")?.truthy() {
                let out = env.ai_run("@cf/google/embeddinggemma-300m", &obj! { "text" => vec![item.g("semantic")?] }).await?;
                ctx.embeddings = out.g("data")?;
                ctx.s_VectorizeVector = V::from(vec![obj! {
                    "id" => item.g("id")?,
                    "values" => ctx.embeddings.gi(0)?,
                    "metadata" => ctx.metadata.clone(),
                }]);
                ctx.models.s_("cloudflare", dec(&ctx.models.g("cloudflare")?))?;
            }
            if !ctx.embeddings.truthy() && ctx.models.g("deepinfra")?.truthy() {
                // 고침(fix-tirm): item.semantic.tirm() → item.semantic.trim()
                let content = trim_of(&item.g("semantic")?, "item.semantic")?;
                ctx.embeddings = crate::llm::deepinfra(
                    &ctx.deepinfra,
                    &V::from("google/embeddinggemma-300m"),
                    &V::from(""),
                    &content,
                    &V::Undef,
                    &V::Undef,
                )
                .await?;
                ctx.s_VectorizeVector = map_vectors(&ctx.embeddings, &item.g("id")?, &ctx.metadata)?;
                ctx.models.s_("deepinfra", dec(&ctx.models.g("deepinfra")?))?;
            }
            console_log!("typeof embeddings", ctx.embeddings.type_of());
            if !ctx.embeddings.truthy() {
                ctx.fallback = V::from("embeddings overflow");
                return Ok(Flow::Continue);
            }
            let name = tpl(T5233, &[ctx.vectorRegion.clone(), ctx.itemType.clone()]);
            env.vectorize_upsert(&name, &ctx.s_VectorizeVector).await?;
        }
    }

    // item.data.text = item.semantic · item.data.ref = page.ref (왼쪽 객체 → 오른쪽 값 → 쓰기)
    let target = item.g("data")?;
    let val = item.g("semantic")?;
    target.s_("text", val)?;
    let target = item.g("data")?;
    let val = ctx.page.g("ref")?;
    target.s_("ref", val)?;

    let shown = stringify_checked(&item)?.unwrap_or_else(|| "undefined".into());
    console_log!("item", shown);

    ctx.arr = gzip_json(&item.g("data")?)?;
    item.s_("data", ctx.arr.clone())?;

    let zone = ctx.zoneRegion.clone();
    let now = ctx.now.clone();
    let updated_at = ctx.updated_at.clone();
    let db = tpl(T5248, &[zone.clone()]);
    push_bound(&ctx.statements, &tpl(T5247, &[zone.clone()]), env, &db, T5248_1[0], || {
        let f = |k: &str| item.get(k);
        Ok(vec![
            f("id"),
            f("type"),
            f("from"),
            f("to"),
            f("cc"),
            f("bcc"),
            f("ref"),
            f("digest"),
            f("data"),
            now,
            updated_at,
        ])
    })?;

    if loose_eq(&ctx.itemType, &V::from("sales")) {
        let db = tpl(T5281, &[zone.clone()]);
        push_bound(&ctx.statements, &tpl(T5280, &[zone]), env, &db, T5281_1[0], || {
            Ok(sales_params(&item, item.get("type"), &item, item.get("status"), pick(item.get("currency"), V::from(""))))
        })?;
    } else if loose_eq(&ctx.itemType, &V::from("tracking")) {
        let db = tpl(T5373, &[zone.clone()]);
        push_bound(&ctx.statements, &tpl(T5372, &[zone]), env, &db, T5373_1[0], || {
            Ok(tracking_params(&item, item.get("type"), &item, item.get("status")))
        })?;
    } else if loose_eq(&ctx.itemType, &V::from("event")) {
        let db = tpl(T5453, &[zone.clone()]);
        push_bound(&ctx.statements, &tpl(T5452, &[zone]), env, &db, &event_sql(T5453_1[0]), || {
            Ok(event_params(&item, item.get("type"), &item, item.get("status")))
        })?;
    }

    Ok(Flow::Next)
}

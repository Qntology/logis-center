//! `export default { async fetch(request, env, ctx) }` — 원본 2354~6487 줄의 뼈대.
//!
//! 머리(2362~2541: 요청 풀기 · 작업 행 · 팀 · 소유자), 작업마다 세 갈래(이미지 [`crate::image`] · 스캔 [`crate::scan`] ·
//! 검색 [`crate::search`]), 꼬리(6326~6441: talks · tasks · users 문장), 바깥(6443~6487: lock 지우기 · 배치 · 응답 · 418).
//! 예외의 경계는 원본 try/catch 그대로: 작업 반복 안의 예외는 6443 의 catch 로(그 회차의 남은 일을 건너뜀), 그 밖은 418.

use logis_worker_common::gzip::{gzip_json, ungzip_json};
use logis_worker_common::js::{self, Env};
use logis_worker_common::jsv::{add, index_of, num_to_string, Throw};
use logis_worker_common::tables::{lookup, CENTER_REGION, LANGUAGE_CODE, LOGIS_REGION, TABLES};
use logis_worker_common::util::is_diff;
use logis_worker_common::{console_log, date, obj, stringify_str, R, V};

use crate::ctx::Ctx;
use crate::flow::Flow;
use crate::templates::*;

/// `statements[key].push(stmt)` — 칸이 없으면 원본처럼 TypeError
pub fn push_stmt(statements: &V, key: &str, stmt: V) -> R<()> {
    statements.get(key).push(stmt)
}

/// `env[db].prepare(sql).bind(...params)`
pub fn prepare_bind(env: &Env, db: &str, sql: &str, params: &[V]) -> R<V> {
    let stmt = env.prepare(db, sql)?;
    js::bind(&stmt, params)
}

/// `(await env[db].prepare(sql).all()).results`
pub async fn select(env: &Env, db: &str, sql: &str) -> R<V> {
    js::query(env, db, sql).await
}

/// `(await env[db].prepare(sql).bind(...params).all()).results` — 값은 SQL 글에 넣지 않고 묶는다
pub async fn select_bound(env: &Env, db: &str, sql: &str, params: &[V]) -> R<V> {
    js::query_bound(env, db, sql, params).await
}

/// 원본 2419 의 작업 찾기 — 값은 묶는다 (`id` = json.id, `ref` = json.ref, `created_at` < created_at)
const SQL_TASK: &str = "SELECT * FROM tasks WHERE \"id\" = ?1 AND \"ref\" = ?2 AND \"created_at\" < ?3 AND \"updated_at\" = 0 ORDER BY created_at ASC LIMIT 1";
/// 원본 2485 의 팀 찾기 — 값은 묶는다 (`id` = task.to, `created_at` < now)
const SQL_TEAM: &str = "SELECT * FROM users WHERE \"type\" = 'team' AND \"id\" = ?1 AND \"created_at\" < ?2 LIMIT 1";
/// 원본 2510 의 소유자 찾기 — 값은 묶는다 (`id` = team.from, `created_at` < now)
const SQL_OWNER: &str = "SELECT * FROM users WHERE \"type\" = 'user' AND \"id\" = ?1 AND \"created_at\" < ?2 LIMIT 1";

/// 따옴표 안에 이어 붙이던 값 — 원본이 글에 넣던 그대로(`${x}`)의 글로 묶는다
pub fn text(v: &V) -> V {
    V::from(v.to_str())
}

/// `x -= 1`
pub fn dec(v: &V) -> V {
    V::Num(v.to_num() - 1.0)
}

/// 요청 하나 — 원본과 같은 응답 (JSON 또는 418)
pub async fn fetch(body: Vec<u8>, env: Env) -> (u16, String, &'static str) {
    let mut ctx = Ctx::default();
    match outer(&body, &env, &mut ctx).await {
        Ok(Some(json)) => (200, json, "application/json"),
        Ok(None) => (418, "I'm a teapot!".into(), "text/html; charset=utf-8"),
        Err(err) => {
            console_log!("err", err);
            (418, "I'm a teapot!".into(), "text/html; charset=utf-8")
        }
    }
}

/// 2362~6477 (바깥 try 안)
async fn outer(buffer: &[u8], env: &Env, ctx: &mut Ctx) -> R<Option<String>> {
    if buffer.is_empty() {
        return Ok(None);
    }
    ctx.json = ungzip_json(&V::bytes(buffer.to_vec()))?;
    let json = ctx.json.clone();
    ctx.now = json.g("now")?;
    ctx.gemini_llm_api = json.g("gemini_llm_api")?;
    ctx.gemini_llm_model = json.g("gemini_llm_model")?;
    ctx.deepinfra = json.g("deepinfra")?;
    ctx.current = V::from(date::to_iso_string(date::get_time(&ctx.now))?);
    ctx.created_at = V::Num(ctx.now.to_num() - 10000.0);
    ctx.limits = json.g("limits")?;
    ctx.models = json.g("models")?;
    ctx.region = json.g("region")?;
    console_log!("region", ctx.region.clone());

    ctx.fallback = V::from("");
    ctx.statements = V::new_obj();
    ctx.statements.put(CENTER_REGION, V::new_arr());
    ctx.statements.sk(&ctx.region, V::new_arr())?;

    if let Err(err) = tasks(env, ctx).await {
        console_log!("err", err);
    }

    if ctx.region.truthy() {
        let region = ctx.region.to_key();
        let lock = env.prepare(&region, T6449[0])?;
        // statements[region].push(...) — 순서: 칸 읽기 → prepare → push
        let slot = ctx.statements.get(&region);
        slot.push(lock)?;
        for region in ctx.statements.keys() {
            let batch = ctx.statements.get(&region);
            ctx.batch = batch.clone();
            if batch.length()?.truthy() {
                let r = env.batch(&region, &batch).await?;
                ctx.results = r.get("results");
                ctx.success = r.get("success");
                ctx.error = r.get("error");
            }
        }
    }

    let mut results = V::new_obj();
    if ctx.limits.truthy() && ctx.models.truthy() {
        results = obj! { "models" => ctx.models.clone(), "limits" => ctx.limits.clone() };
    }
    ctx.results = results.clone();
    Ok(Some(stringify_str(&results)))
}

/// 2418~6442 (안쪽 try): 작업 행 읽기 · 작업마다 처리 · 꼬리 문장
async fn tasks(env: &Env, ctx: &mut Ctx) -> R<()> {
    let json = ctx.json.clone();
    // 고침: 값을 SQL 글에 이어 붙였다(작은따옴표가 든 값은 쿼리를 깨뜨림) — 묶는다
    let params = [text(&json.g("id")?), text(&json.g("ref")?), ctx.created_at.clone()];
    ctx.results = select_bound(env, &ctx.region.to_key(), SQL_TASK, &params).await?;
    console_log!("results.length", ctx.results.length()?);
    ctx.crons = ctx.results.deep_clone();

    if !ctx.crons.length()?.truthy() {
        return Ok(());
    }
    ctx.c = V::Num(0.0);
    while ctx.c.to_num() < ctx.crons.length()?.to_num() {
        match task(env, ctx).await? {
            Flow::Next | Flow::Continue => {}
            Flow::Break => break,
        }
        ctx.c = V::Num(ctx.c.to_num() + 1.0);
    }

    if !ctx.statements.keys().is_empty() {
        epilogue(env, ctx)?;
    }
    Ok(())
}

/// 작업 하나 (2427 반복의 몸통)
async fn task(env: &Env, ctx: &mut Ctx) -> R<Flow> {
    ctx.cron = ctx.crons.gk(&ctx.c)?;
    ctx.task = ungzip_json(&ctx.cron.g("data")?)?;
    let task = ctx.task.clone();
    ctx.talk = obj! {
        "id" => task.g("id")?,
        "type" => task.g("type")?,
        "from" => task.g("from")?,
        "to" => task.g("to")?,
        "cc" => task.g("cc")?,
        "bcc" => task.g("bcc")?,
        "ref" => task.g("ref")?,
        "data" => task.g("data")?,
        "created_at" => ctx.now.clone(),
        "updated_at" => ctx.now.clone(),
    };

    ctx.logisRegion = lookup(LOGIS_REGION, &task.g("flag")?);
    ctx.zoneRegion = task.g("zone")?;
    ctx.vectorRegion = add(&V::from("commerce-logis-"), &ctx.zoneRegion);
    ctx.language = lookup(LANGUAGE_CODE, &task.g("flag")?);
    if !ctx.statements.gk(&ctx.logisRegion)?.truthy() {
        ctx.statements.sk(&ctx.logisRegion, V::new_arr())?;
    }

    let cc = task.g("cc")?;
    if !ctx.models.gk(&cc)?.truthy() {
        ctx.models.sk(&cc, task.g("rpm")?)?;
    }
    if ctx.models.gk(&cc)?.truthy() {
        ctx.models.sk(&cc, dec(&ctx.models.gk(&cc)?))?;
    } else {
        ctx.fallback = V::from("models");
        return Ok(Flow::Continue);
    }

    ctx.t = V::Num(0.0);
    while (ctx.t.to_num() as usize) < TABLES.len() {
        ctx.table = V::from(TABLES[ctx.t.to_num() as usize]);
        let key = format!("commerce_logis_{}_{}", ctx.zoneRegion.to_str(), ctx.table.to_str());
        if !ctx.statements.g(&key)?.truthy() {
            ctx.statements.s_(&key, V::new_arr())?;
        }
        ctx.t = V::Num(ctx.t.to_num() + 1.0);
    }

    let from = task.g("from")?;
    if !ctx.limits.gk(&from)?.truthy() {
        ctx.limits.sk(&from, task.g("rpm")?)?;
    }

    // 고침: 값을 묶는다 (위와 같음)
    let params = [text(&task.g("to")?), ctx.now.clone()];
    ctx.results = select_bound(env, &ctx.logisRegion.to_key(), SQL_TEAM, &params).await?;
    ctx.team = ctx.results.gi(0)?;
    if ctx.team.g("data")?.truthy() {
        let data = ungzip_json(&ctx.team.g("data")?)?;
        ctx.team.s_("data", data)?;
    } else {
        ctx.team.s_("data", V::new_obj())?;
    }

    let limits_team = ctx.limits.g("team")?;
    if limits_team.truthy()
        && logis_worker_common::loose_eq(&limits_team.g("id")?, &ctx.team.g("id")?)
        && is_diff(&ctx.team.g("data")?, &limits_team.g("data")?)?
    {
        ctx.statements = V::new_obj();
        return Ok(Flow::Continue);
    }

    console_log!("team.data.base.pages", stringify_str(&ctx.team.g("data")?.g("base")?.g("pages")?));

    // 고침: 값을 묶는다 (위와 같음)
    let params = [text(&ctx.team.g("from")?), ctx.now.clone()];
    ctx.results = select_bound(env, &ctx.logisRegion.to_key(), SQL_OWNER, &params).await?;
    ctx.owner = ctx.results.gi(0)?;
    let owner_data = ungzip_json(&ctx.owner.g("data")?)?;
    ctx.owner.s_("data", owner_data)?;

    console_log!("task.contentType", task.g("contentType")?);

    let content_type = match task.g("contentType")? {
        V::Str(s) => s,
        other => return Err(Throw::type_error(format!("Cannot read properties of {} (reading 'indexOf')", other.to_str()))),
    };
    if index_of(&content_type, "image/") > -1 {
        crate::image::run(ctx, env).await
    } else if task.g("scan")?.truthy() {
        crate::scan::run(ctx, env).await
    } else {
        crate::search::run(ctx, env).await
    }
}

/// 6328~6441: 대체 대화 · 대화 · 작업 · 팀 문장
fn epilogue(env: &Env, ctx: &mut Ctx) -> R<()> {
    let zone = ctx.zoneRegion.to_str();
    let talks = format!("commerce_logis_{zone}_talks");
    let task = ctx.task.clone();
    let talk = ctx.talk.clone();
    let now = ctx.now.clone();

    if ctx.fallback.truthy() {
        console_log!("fallback", ctx.fallback.clone());
        let slot = ctx.statements.get(&talks);
        let stmt = prepare_bind(
            env,
            &talks,
            T6333_1[0],
            &[talk.g("id")?, V::from("prompt"), task.g("from")?, task.g("to")?, task.g("cc")?, task.g("bcc")?, talk.g("ref")?, V::Null, now.clone(), now.clone()],
        )?;
        slot.push(stmt)?;
    }

    talk.s_("data", V::Null)?;
    if talk.g("text")?.truthy() {
        talk.s_("data", gzip_json(&obj! { "text" => talk.g("text")? })?)?;
    }

    let slot = ctx.statements.get(&talks);
    let stmt = prepare_bind(
        env,
        &talks,
        T6374_1[0],
        &[talk.g("id")?, talk.g("type")?, talk.g("from")?, talk.g("to")?, talk.g("cc")?, talk.g("bcc")?, talk.g("ref")?, talk.g("data")?, now.clone(), now.clone()],
    )?;
    slot.push(stmt)?;

    let team = ctx.team.clone();
    let arr = gzip_json(&obj! { "id" => team.g("id")?, "flag" => task.g("flag")?, "team" => team.clone(), "created_at" => ctx.cron.g("created_at")? })?;
    let region = ctx.region.to_key();
    let slot = ctx.statements.get(&region);
    slot.push(prepare_bind(env, &region, T6416[0], &[arr, now.clone(), task.g("id")?])?)?;

    console_log!("team.data.base.pages", stringify_str(&team.g("data")?.g("base")?.g("pages")?));

    let arr = gzip_json(&team.g("data")?)?;
    let logis = ctx.logisRegion.to_key();
    let slot = ctx.statements.get(&logis);
    slot.push(prepare_bind(env, &logis, T6428[0], &[arr, now, team.g("id")?])?)?;

    let id = match task.g("id")? {
        V::Str(s) => s.to_uppercase(),
        other => return Err(Throw::type_error(format!("toUpperCase is not a function ({})", other.type_of()))),
    };
    ctx.limits.s_(&id, V::Bool(true))?;
    let _ = num_to_string;
    Ok(())
}

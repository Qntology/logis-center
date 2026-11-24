//! 자연어 검색 (원본 5682~6320): 글을 문맥으로 나누고 벡터 · SQL 로 찾은 행으로 LLM 이 답을 만들어 talks 에 쌓는다.

use logis_worker_common::gzip::{decode_utf8, gzip_json, ungzip};
use logis_worker_common::js::{self, Env};
use logis_worker_common::jsv::{index_of, less_than, parse_str, tpl};
use logis_worker_common::tables::TABLES;
use logis_worker_common::util::{hash_id, rows_trim};
use logis_worker_common::{add, arr, console_log, loose_eq, obj, stringify_str, Throw, R, V};

use crate::ctx::Ctx;
use crate::flow::Flow;
use crate::handler::{dec, select_bound, text};
use crate::llm;
use crate::rules::{context2results, graph2contexts, para2graph, paragraph2propertys, parse_condition, parse_status, table_columns};
use crate::templates::*;

fn is(v: &V, s: &str) -> bool {
    loose_eq(v, &V::from(s))
}

fn inc(v: &V) -> V {
    V::Num(v.to_num() + 1.0)
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

/// `decompressedJsonString = …decode(ungzip(v)); JSON.parse(decompressedJsonString)`
fn unzip(ctx: &mut Ctx, v: &V) -> R<V> {
    let s = decode_utf8(&ungzip(v)?);
    ctx.decompressedJsonString = V::from(s.clone());
    parse_str(&s)
}

/// 묶을 값을 넣고 그 자리표(`?n`)를 돌려준다
fn ph(params: &mut Vec<V>, v: V) -> String {
    params.push(v);
    format!("?{}", params.len())
}

/// SQL 의 표 이름 — 고침: LLM 이 낸 context.type 에서 오므로 검색 표(TABLES)만 받는다.
/// 원본에선 그 밖의 이름이면 `env[…]` 가 undefined 라 prepare 에서 던졌다(같은 try 가 받음)
fn search_table(type_: &V) -> R<String> {
    let t = type_.to_str();
    if TABLES.contains(&t.as_str()) {
        Ok(t)
    } else {
        Err(Throw::type_error("Cannot read properties of undefined (reading 'prepare')"))
    }
}

/// `[...x]` — 배열만 (원본에선 늘 배열)
fn spread(v: &V) -> R<Vec<V>> {
    match v {
        V::Arr(_) => Ok(v.items()),
        _ => Err(Throw::type_error(format!("{} is not iterable", v.to_str()))),
    }
}

/// `rowsTrim(rows)` (키 없이) = `rows.filter(row => row !== undefined)` — 배열이 아니면 원본처럼 던진다
fn rows_trim_all(rows: &V) -> R<V> {
    if !rows.is_arr() {
        return Err(Throw::type_error("rows.filter is not a function"));
    }
    rows_trim(rows, &V::Undef, &V::Undef)
}

pub async fn run(ctx: &mut Ctx, env: &Env) -> R<Flow> {
    let task = ctx.task.clone();
    let models = ctx.models.clone();
    let gemini_key = add(&add(&ctx.gemini_llm_api, &V::from("-")), &ctx.gemini_llm_model);

    // team.data.base['prompt'].count++ (원본 주석)

    // 고침: `var paragraphs` 가 값을 지우지 않아 앞 회차 값이 남았다 — 작업마다 비운다
    ctx.paragraphs = V::Undef;
    ctx.system = para2graph(&ctx.language);

    if models.g("deepinfra")?.truthy() {
        ctx.res = llm::deepinfra(&ctx.deepinfra, &V::from("openai/gpt-oss-20b"), &ctx.system, &task.g("body")?, &V::Undef, &V::Undef).await?;
        if ctx.res.truthy() {
            ctx.paragraphs = ctx.res.g("context")?;
        }
        models_dec(&models, &V::from("deepinfra"))?;
    }

    if !ctx.paragraphs.truthy() && ctx.gemini_llm_api.truthy() {
        ctx.res = llm::gemini(&ctx.gemini_llm_api, &ctx.gemini_llm_model, &ctx.system, &task.g("body")?, &V::Undef, &V::Undef).await?;
        if ctx.res.truthy() {
            ctx.paragraphs = ctx.res.g("context")?;
        }
        models_dec(&models, &gemini_key)?;
    }

    if !ctx.paragraphs.truthy() {
        ctx.fallback = V::from("para2graph overflow");
        return Ok(Flow::Continue);
    }

    console_log!("before paragraphs", stringify_str(&ctx.paragraphs));

    // try { … } catch(err) { console.log('paragraphs err', err) } — 안의 continue 는 작업 반복으로
    match paragraphs_step(ctx) {
        Ok(true) => return Ok(Flow::Continue),
        Ok(false) => {}
        Err(err) => console_log!("paragraphs err", err),
    }

    // 고침: 배열이 아닌 값(글 등)은 rowsTrim 이 던졌다 — 배열일 때만 거르고, 아니면 아래에서 빈 경우처럼 넘긴다
    if ctx.paragraphs.is_arr() && ctx.paragraphs.g("length")?.truthy() {
        ctx.paragraphs = rows_trim_all(&ctx.paragraphs)?;
    }

    console_log!("after paragraphs", stringify_str(&ctx.paragraphs));

    // 고침: length 가 없는 값(undefined == 0 은 거짓)이 그대로 지나갔다 — 배열이 아니면 빈 경우처럼
    if !ctx.paragraphs.is_arr() || loose_eq(&ctx.paragraphs.g("length")?, &V::Num(0.0)) {
        return Ok(Flow::Continue);
    }

    // 고침: `var contexts` 도 지우지 않았다 — 작업마다 비운다
    ctx.contexts = V::Undef;
    ctx.system = graph2contexts(&ctx.current);

    if models.g("deepinfra")?.truthy() {
        let user = tpl(T5775, &[V::from(stringify_str(&ctx.paragraphs))]);
        ctx.results = llm::deepinfra(&ctx.deepinfra, &V::from("openai/gpt-oss-20b"), &ctx.system, &V::from(user), &V::Undef, &V::Undef).await?;

        console_log!("contexts res", stringify_str(&ctx.results));

        if ctx.results.truthy() {
            if ctx.results.g("contexts")?.truthy() {
                ctx.contexts = ctx.results.g("contexts")?;
            } else if !ctx.results.keys_checked()?.is_empty() {
                // 고침: LLM 이 글 · 수를 돌려주면 results.text = … 가 (엄격 모드라) 던졌다 — 객체일 때만 쓴다
                if ctx.results.type_of() == "object" {
                    if !ctx.results.g("text")?.truthy() {
                        ctx.results.s_("text", task.g("body")?)?;
                    }

                    if !ctx.results.g("type")?.truthy() && loose_eq(&ctx.paragraphs.g("length")?, &V::Num(1.0)) {
                        ctx.results.s_("type", ctx.paragraphs.gi(0)?.g("type")?)?;
                    }
                }

                ctx.contexts = arr![ctx.results.clone()];
            }
        }

        models_dec(&models, &V::from("deepinfra"))?;
    }

    if !ctx.contexts.truthy() {
        ctx.fallback = V::from("graph2contexts overflow");
        return Ok(Flow::Continue);
    }

    console_log!("contexts", stringify_str(&ctx.contexts));

    ctx.augmented = V::from("");

    // 유료 회원 이전 컨텍스트 합치기는 원본에서 주석 처리됨

    if ctx.contexts.truthy() && ctx.contexts.g("length")?.truthy() {
        ctx.q = V::Num(0.0);
        while less_than(&ctx.q, &ctx.contexts.g("length")?) == Some(true) {
            // 몸통의 continue 는 q 반복으로 (Ok 로 돌아와 q 를 올린다)
            context_step(ctx, env).await?;
            ctx.q = inc(&ctx.q);
        }
    }

    Ok(Flow::Next)
}

/// 5741~5756 의 try 안. true 면 작업 반복의 continue
fn paragraphs_step(ctx: &mut Ctx) -> R<bool> {
    if ctx.paragraphs.g("length")?.truthy() {
        ctx.p = V::Num(0.0);
        while less_than(&ctx.p, &ctx.paragraphs.g("length")?) == Some(true) {
            ctx.paragraph = ctx.paragraphs.gk(&ctx.p)?;
            let v = paragraph2propertys(&ctx.task, &ctx.team, &ctx.paragraph)?;
            ctx.paragraphs.sk(&ctx.p, v)?;
            ctx.p = inc(&ctx.p);
        }
        Ok(false)
    } else {
        ctx.fallback = V::from("paragraph2propertys");
        // 올바르지 않은 대화요청입니다 리턴해야함
        Ok(true)
    }
}

/// 문맥 하나 (5879~6317). 일찍 Ok 로 돌아가는 곳이 원본의 q 반복 continue
async fn context_step(ctx: &mut Ctx, env: &Env) -> R<()> {
    let task = ctx.task.clone();
    let team = ctx.team.clone();
    let models = ctx.models.clone();
    let now = ctx.now.clone();
    let zone = ctx.zoneRegion.clone();

    ctx.context = ctx.contexts.gk(&ctx.q)?;
    let context = ctx.context.clone();

    // 고침: 문맥이 글 · 수면 context.id = … 가 던졌다 — 객체일 때만 쓴다(아래 !context.type 에서 건너뛴다)
    if context.type_of() == "object" && !context.is_null() {
        let id = hash_id(&V::Undef)?;
        context.s_("id", V::from(id))?;
    }

    if !context.g("type")?.truthy() {
        return Ok(());
    }

    if is(&context.g("date")?, "#date") {
        context.s_("date", V::from(""))?;
    }
    if is(&context.g("status")?, "#status") {
        context.s_("status", V::from(""))?;
    }
    if is(&context.g("substantial")?, "#substantial") {
        context.s_("substantial", V::from(""))?;
    }
    if is(&context.g("find")?, "#find") {
        context.s_("find", V::from(""))?;
    }

    ctx.type_ = context.g("type")?;

    let ct = context.g("type")?;
    if is(&ct, "sales") {
        ctx.type_ = V::from("sales");
        context.s_("type", V::from("order"))?;
    } else if is(&ct, "goods") || is(&context.g("type")?, "order") {
        ctx.type_ = V::from("sales");
    } else if is(&context.g("type")?, "event") || is(&context.g("type")?, "coupon") {
        ctx.type_ = V::from("event");
    }

    context.s_("by", V::from("created_at"))?;
    if context.g("substantial")?.truthy() {
        context.s_("by", context.g("substantial")?)?;
    }

    context.s_("sort", V::from("DESC"))?;
    if context.g("find")?.truthy() {
        let f = context.g("find")?;
        if is(&f, "light") || is(&f, "few") || is(&f, "little") {
            context.s_("sort", V::from("ASC"))?;
        }
    }

    ctx.query = obj! {
        "options" => obj! {
            "topK" => task.g("topK")?,
            "returnValues" => true,
            "returnMetadata" => "all",
            "filter" => obj! { "type" => context.g("type")?, "to" => team.g("id")? },
        },
    };

    // 고침: `var queryVector` 가 지워지지 않아 첫 문맥의 벡터가 다음 문맥 · 작업에도 쓰였다 — 문맥마다 비운다
    ctx.queryVector = V::Undef;

    console_log!("context.text", context.g("text")?);

    // Client App 이 보낸 질의 벡터를 먼저 쓴다 (Phase 4-1 D3: 384-dim granite-r2 유한 벡터 가드)
    const EXPECTED_EMBED_DIM: usize = 384;
    let qv = task.g("queryVector")?;
    if qv.truthy() && qv.is_arr() {
        let dim = qv.g("length")?.to_f64() as usize;
        let mut all_finite = dim == EXPECTED_EMBED_DIM;
        if all_finite {
            for idx in 0..dim {
                if !qv.gi(idx)?.to_f64().is_finite() {
                    all_finite = false;
                    break;
                }
            }
        }
        if all_finite {
            ctx.queryVector = qv;
            console_log!("[EMBED-CLIENT] Using client-side query vector. dim =", dim);
        } else {
            console_log!("[WORKER-EMBED-GUARD] Rejecting invalid client queryVector (dim =", dim, ", expected 384 finite)");
        }
    }

    // Phase 4-1 D3: Do NOT fall back to 768-dim `embeddinggemma-300m` (incompatible with 384-dim granite-97m-r2 space).
    if !ctx.queryVector.truthy() {
        console_log!("[WORKER-EMBED-GUARD] Missing/invalid 384-dim client queryVector -> overflow (skipping incompatible embeddinggemma-300m)");
        ctx.fallback = V::from("overflow");
        return Ok(());
    }

    // 고침: 값을 SQL 글에 이어 붙였다 — 묶는다(자리표 ?n 과 params). 조건 글의 나머지는 수 · 알려진 열 이름뿐
    let mut params: Vec<V> = Vec::new();
    ctx.condition = V::from(format!(" \"type\" = {} ", ph(&mut params, text(&context.g("type")?))));

    if context.g("status")?.truthy() {
        if is(&ctx.type_, "sales") {
            let s = context.g("status")?;
            if is(&s, "used") || is(&s, "lease") || is(&s, "rental") || is(&s, "refurbish") {
                // 열 이름 자리 — 위 네 이름만 들어온다
                ctx.condition = add(&ctx.condition, &V::from(tpl(T5999, &[context.g("status")?])));
            }
        } else {
            ctx.status = parse_status(&context.g("status")?);
            if ctx.status.truthy() {
                let p = ph(&mut params, ctx.status.clone());
                ctx.condition = add(&ctx.condition, &V::from(format!(" AND \"status\" = {p} ")));
            }
        }
    }

    ctx.temp = V::new_obj();

    // 고침: context.condition 이 없으면 Object.keys 가 던져 이 작업의 남은 일을 건너뛰었다 — 객체가 아니면 {} 로 본다
    let condition = context.g("condition")?;
    let condition = if condition.type_of() == "object" && !condition.is_null() { condition } else { V::new_obj() };
    if !condition.keys_checked()?.is_empty() {
        for key1 in context.g("condition")?.keys() {
            ctx.key1 = V::from(key1.clone());
            ctx.obj = context.g("condition")?.gk(&ctx.key1)?;

            if !context.g("condition")?.has(&key1) {
                continue;
            }
            // for-in 은 null · undefined 를 건너뛴다(keys() 가 빈 목록)
            for key2 in ctx.obj.keys() {
                ctx.key2 = V::from(key2.clone());
                ctx.value = ctx.obj.gk(&ctx.key2)?;

                if !ctx.value.truthy() || key2 == "min" || key2 == "max" {
                    continue;
                }
                if ctx.temp.gk(&ctx.key1)?.truthy() {
                    continue;
                }
                ctx.temp.sk(&ctx.key1, V::Bool(true))?;

                if ctx.value.to_num().is_nan() {
                    if key1 != "date" {
                        ctx.query.g("options")?.g("filter")?.sk(&ctx.key1, ctx.value.clone())?;
                    }

                    let c = parse_condition(&ctx.obj, &ctx.key1, &V::from(" AND "))?;
                    ctx.condition = add(&ctx.condition, &c);

                    if key1 == "price" && context.g("condition")?.g("currency")?.truthy() {
                        ctx.query.g("options")?.g("filter")?.s_("currency", context.g("condition")?.g("currency")?)?;
                    }
                } else {
                    let c = parse_condition(&ctx.value, &ctx.key1, &V::from(" AND "))?;
                    ctx.condition = add(&ctx.condition, &c);
                }
            }
        }
    }

    console_log!("before condition", ctx.condition.clone());

    console_log!("context.condition", stringify_str(&context.g("condition")?));

    console_log!("query.options", stringify_str(&ctx.query.g("options")?));
    console_log!(tpl(T6070, &[ctx.vectorRegion.clone(), ctx.type_.clone()]));

    let index = tpl(T6072, &[ctx.vectorRegion.clone(), ctx.type_.clone()]);
    let found = env.vectorize_query(&index, &ctx.queryVector, &ctx.query.g("options")?).await?;
    ctx.matches = found.g("matches")?;

    ctx.rag = obj! {
        "search" => obj! {
            "query" => context.g("condition")?,
            "sql" => obj! { "results" => V::new_arr() },
            "vector" => obj! { "results" => V::new_arr() },
        },
    };

    // 고침: matches 에 length 가 없으면 matches.length.toString() 이 던졌다 — 0 으로 본다
    let len = if ctx.matches.truthy() { ctx.matches.g("length")? } else { ctx.matches.clone() };
    let len = if len.truthy() { len } else { V::Num(0.0) };
    console_log!("matches.length", len.to_str());

    ctx.matches_condition = V::from("");

    // 고침: 값을 묶는다 — ?1 = team.id, ?2 = now, 그 뒤가 match.id 들
    let mut match_params: Vec<V> = vec![text(&team.g("id")?), now.clone()];
    if ctx.matches.truthy() && ctx.matches.g("length")?.truthy() {
        ctx.m = V::Num(0.0);
        while less_than(&ctx.m, &ctx.matches.g("length")?) == Some(true) {
            ctx.match_ = ctx.matches.gk(&ctx.m)?;

            if ctx.matches_condition.g("length")?.truthy() {
                ctx.matches_condition = add(&ctx.matches_condition, &V::from(" OR "));
            }

            let id = ph(&mut match_params, text(&ctx.match_.g("id")?));
            let part = format!("(\"id\" = {id} AND \"to\" = ?1 AND \"created_at\" < ?2)");
            ctx.matches_condition = add(&ctx.matches_condition, &V::from(part));
            ctx.m = inc(&ctx.m);
        }

        console_log!("matches_condition", ctx.matches_condition.clone());

        console_log!("type", ctx.type_.clone());

        if let Err(err) = vector_rows(ctx, env, &match_params).await {
            console_log!("matches_condition", err);
        }
    }

    ctx.orderBy = V::from("");

    if context.g("sort")?.truthy() && context.g("by")?.truthy() {
        ctx.orderBy = V::from(tpl(T6152, &[context.g("by")?, context.g("sort")?]));
    }

    console_log!("type", ctx.type_.clone());

    console_log!("after condition", ctx.condition.clone());

    if ctx.condition.truthy() {
        if let Err(err) = sql_rows(ctx, env, params).await {
            console_log!("text2sql err", err);
        }
    }

    ctx.system = V::from("Return the content related to the {search.text} value from the search results in a JSON structure.");

    let search = ctx.rag.g("search")?;
    let mut joined = spread(&search.g("sql")?.g("results")?)?;
    joined.extend(spread(&search.g("vector")?.g("results")?)?);
    ctx.content = context2results(&context, &V::from(joined), &ctx.language)?;

    // 고침: `var generation` 도 지우지 않아 두 모델이 다 꺼져 있으면 앞 문맥의 답이 다시 쓰였다 — 문맥마다 비운다
    ctx.generation = V::Undef;
    if models.g("deepinfra")?.truthy() {
        ctx.generation = llm::deepinfra(&ctx.deepinfra, &V::from("openai/gpt-oss-20b"), &ctx.system, &ctx.content, &V::Undef, &V::Undef).await?;
        models_dec(&models, &V::from("deepinfra"))?;
    }

    if !ctx.generation.truthy() && ctx.gemini_llm_api.truthy() {
        ctx.generation = llm::gemini(&ctx.gemini_llm_api, &ctx.gemini_llm_model, &ctx.system, &ctx.content, &V::Undef, &V::Undef).await?;
        let key = add(&add(&ctx.gemini_llm_api, &V::from("-")), &ctx.gemini_llm_model);
        models_dec(&models, &key)?;
    }

    if !ctx.generation.truthy() {
        ctx.fallback = V::from("overflow");
        return Ok(());
    }

    console_log!("generation", stringify_str(&ctx.generation));

    ctx.arr = gzip_json(&obj! {
        "id" => context.g("id")?,
        "markdown" => ctx.generation.g("markdown")?,
        "results" => ctx.generation.g("results")?,
        "search" => ctx.rag.g("search")?,
    })?;

    context.s_("data", ctx.arr.clone())?;

    let s = slot(&ctx.statements, &tpl(T6259, &[zone.clone()]))?;
    let stmt = env.prepare(&tpl(T6260, &[zone.clone()]), T6260_1[0])?;
    let stmt = js::bind(
        &stmt,
        &[
            task.g("id")?,
            context.g("type")?,
            team.g("id")?,
            task.g("bcc")?,
            task.g("cc")?,
            task.g("bcc")?,
            task.g("ref")?,
            context.g("data")?,
            V::Num(now.to_num() - 100.0),
            now.clone(),
        ],
    )?;
    s.push(stmt)?;

    let s = slot(&ctx.statements, &tpl(T6289, &[zone.clone()]))?;
    let stmt = env.prepare(&tpl(T6290, &[zone.clone()]), T6290_1[0])?;
    let stmt = js::bind(
        &stmt,
        &[
            context.g("id")?,
            context.g("type")?,
            team.g("id")?,
            task.g("bcc")?,
            task.g("cc")?,
            task.g("bcc")?,
            task.g("ref")?,
            context.g("data")?,
            now.clone(),
            now.clone(),
        ],
    )?;
    s.push(stmt)?;

    Ok(())
}

/// 6107~6145 의 try 안: 벡터로 찾은 id 들의 행
async fn vector_rows(ctx: &mut Ctx, env: &Env, params: &[V]) -> R<()> {
    let table = search_table(&ctx.type_)?;
    let db = tpl(T6108, &[ctx.zoneRegion.clone(), ctx.type_.clone()]);
    let sql = format!("SELECT * FROM {table} WHERE {} LIMIT 100", ctx.matches_condition.to_str());
    ctx.results = select_bound(env, &db, &sql, params).await?;

    if ctx.results.length()?.truthy() {
        ctx.r = V::Num(0.0);
        while less_than(&ctx.r, &ctx.results.length()?) == Some(true) {
            ctx.result = ctx.results.gk(&ctx.r)?;

            let blob = ctx.result.g("data")?;
            let parsed = unzip(ctx, &blob)?;
            ctx.data = parsed;

            if ctx.data.truthy() && !ctx.data.keys_checked()?.is_empty() {
                for name in ctx.data.keys() {
                    ctx.name = V::from(name.clone());
                    if ctx.data.has(&name) {
                        ctx.value = ctx.data.gk(&ctx.name)?;
                        ctx.result.sk(&ctx.name, ctx.value.clone())?;
                    }
                }
            }

            // 고침: r 이 아니라 함수 범위의 i 로 지웠다(엉뚱한 행을 고치거나 던져 rag.search.vector 가 비었다) — r 로
            for k in ["from", "to", "cc", "bcc", "ref", "data"] {
                ctx.results.gk(&ctx.r)?.del(k)?;
            }

            ctx.r = inc(&ctx.r);
        }

        ctx.rag.g("search")?.s_("vector", obj! { "results" => ctx.results.clone() })?;
    }
    Ok(())
}

/// 6160~6212 의 try 안: 조건 SQL 로 찾은 행
async fn sql_rows(ctx: &mut Ctx, env: &Env, mut params: Vec<V>) -> R<()> {
    let now = ctx.now.clone();
    let team = ctx.team.clone();

    // 고침: 조건 글에서 created_at 변수(수)를 찾아 사실상 붙지 않았다 — 열 이름 "created_at" 을 찾는다
    if index_of(&ctx.condition.to_str(), "\"created_at\"") > -1 {
        let p = ph(&mut params, now.clone());
        ctx.condition = add(&ctx.condition, &V::from(format!(" AND \"created_at\" < {p}")));
    }

    // 고침: 표 이름 · ORDER BY 의 열은 LLM 이 낸 값(context.type · context.substantial)이라 묶을 수 없다 — 알려진 이름만.
    // 원본에선 표에 없는 열이면 SQL 이 "no such column" 으로 던졌다(같은 try 가 받음)
    let table = search_table(&ctx.type_)?;
    if ctx.orderBy.truthy() {
        let by = ctx.context.g("by")?.to_str();
        let sort = ctx.context.g("sort")?.to_str();
        if !table_columns(&table).is_some_and(|cols| cols.contains(&by.as_str())) || !matches!(sort.as_str(), "ASC" | "DESC") {
            return Err(Throw::error(format!("no such column: {by}")));
        }
    }

    let to = ph(&mut params, text(&team.g("id")?));
    let sql = format!("SELECT * FROM {table} WHERE {} AND \"to\" = {to} {} LIMIT 300", ctx.condition.to_str(), ctx.orderBy.to_str());
    console_log!(sql.clone());

    let db = tpl(T6167, &[ctx.zoneRegion.clone(), ctx.type_.clone()]);
    ctx.results = select_bound(env, &db, &sql, &params).await?;

    console_log!("results.length", ctx.results.length()?);

    if ctx.results.length()?.truthy() {
        ctx.r = V::Num(0.0);
        while less_than(&ctx.r, &ctx.results.length()?) == Some(true) {
            ctx.result = ctx.results.gk(&ctx.r)?;

            let blob = ctx.result.g("data")?.deep_clone();
            let parsed = unzip(ctx, &blob)?;
            ctx.data = parsed;

            if ctx.data.truthy() {
                ctx.result.s_("data", V::new_obj())?;

                if !ctx.data.keys_checked()?.is_empty() {
                    for name in ctx.data.keys() {
                        ctx.name = V::from(name.clone());
                        if ctx.data.has(&name) {
                            ctx.value = ctx.data.gk(&ctx.name)?;

                            if !matches!(name.as_str(), "type" | "item" | "node" | "link" | "origin" | "detail") {
                                ctx.result.g("data")?.sk(&ctx.name, ctx.value.clone())?;
                            }
                        }
                    }
                }
            }

            let row = ctx.results.gk(&ctx.r)?;
            row.s_("data", ctx.result.g("data")?)?;

            for k in ["from", "to", "cc", "bcc", "ref"] {
                ctx.results.gk(&ctx.r)?.del(k)?;
            }

            ctx.r = inc(&ctx.r);
        }

        ctx.rag.g("search")?.s_("sql", obj! { "results" => ctx.results.clone() })?;
    }
    Ok(())
}

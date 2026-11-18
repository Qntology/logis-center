//! 자연어 검색 (원본 5682~6320): 글을 문맥으로 나누고 벡터 · SQL 로 찾은 행으로 LLM 이 답을 만들어 talks 에 쌓는다.

use logis_worker_common::gzip::{decode_utf8, gzip_json, ungzip};
use logis_worker_common::js::{self, Env};
use logis_worker_common::jsv::{index_of, less_than, parse_str, tpl};
use logis_worker_common::util::{hash_id, rows_trim};
use logis_worker_common::{add, arr, console_log, loose_eq, obj, stringify_str, Throw, R, V};

use crate::ctx::Ctx;
use crate::flow::Flow;
use crate::handler::{dec, select};
use crate::llm;
use crate::rules::{context2results, graph2contexts, para2graph, paragraph2propertys, parse_condition, parse_status};
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

    // 원본 그대로: `var paragraphs` 는 값을 지우지 않는다(앞 회차 값이 남는다)
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

    if ctx.paragraphs.g("length")?.truthy() {
        ctx.paragraphs = rows_trim_all(&ctx.paragraphs)?;
    }

    console_log!("after paragraphs", stringify_str(&ctx.paragraphs));

    // 원본 그대로: length 가 없는 값(undefined == 0 은 거짓)이면 그대로 지나간다
    if loose_eq(&ctx.paragraphs.g("length")?, &V::Num(0.0)) {
        return Ok(Flow::Continue);
    }

    // 원본 그대로: `var contexts` 도 지우지 않는다
    ctx.system = graph2contexts(&ctx.current);

    if models.g("deepinfra")?.truthy() {
        let user = tpl(T5775, &[V::from(stringify_str(&ctx.paragraphs))]);
        ctx.results = llm::deepinfra(&ctx.deepinfra, &V::from("openai/gpt-oss-20b"), &ctx.system, &V::from(user), &V::Undef, &V::Undef).await?;

        console_log!("contexts res", stringify_str(&ctx.results));

        if ctx.results.truthy() {
            if ctx.results.g("contexts")?.truthy() {
                ctx.contexts = ctx.results.g("contexts")?;
            } else if !ctx.results.keys_checked()?.is_empty() {
                if !ctx.results.g("text")?.truthy() {
                    ctx.results.s_("text", task.g("body")?)?;
                }

                if !ctx.results.g("type")?.truthy() && loose_eq(&ctx.paragraphs.g("length")?, &V::Num(1.0)) {
                    ctx.results.s_("type", ctx.paragraphs.gi(0)?.g("type")?)?;
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

    let id = hash_id(&V::Undef)?;
    context.s_("id", V::from(id))?;

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

    // 원본 그대로: `var queryVector` 는 지우지 않는다 — 앞 문맥(앞 회차)의 벡터가 다음 문맥에도 쓰인다

    console_log!("context.text", context.g("text")?);

    // Client App 이 보낸 질의 벡터를 먼저 쓴다
    let qv = task.g("queryVector")?;
    if qv.truthy() && qv.is_arr() && qv.g("length")?.truthy() {
        ctx.queryVector = task.g("queryVector")?;
        console_log!("[EMBED-CLIENT] Using client-side query vector. dim =", ctx.queryVector.g("length")?);
    }

    if !ctx.queryVector.truthy() && models.g("cloudflare")?.truthy() {
        ctx.embeddings = env.ai_run("@cf/google/embeddinggemma-300m", &obj! { "text" => vec![context.g("text")?] }).await?;
        ctx.queryVector = ctx.embeddings.g("data")?.gi(0)?;
        models_dec(&models, &V::from("cloudflare"))?;
    }

    if !ctx.queryVector.truthy() && models.g("deepinfra")?.truthy() {
        ctx.embeddings =
            llm::deepinfra(&ctx.deepinfra, &V::from("google/embeddinggemma-300m"), &V::from(""), &context.g("text")?, &V::Undef, &V::Undef).await?;
        ctx.queryVector = ctx.embeddings.clone();
        models_dec(&models, &V::from("deepinfra"))?;
    }

    if !ctx.queryVector.truthy() {
        ctx.fallback = V::from("overflow");
        return Ok(());
    }

    ctx.condition = V::from(tpl(T5994, &[context.g("type")?]));

    if context.g("status")?.truthy() {
        if is(&ctx.type_, "sales") {
            let s = context.g("status")?;
            if is(&s, "used") || is(&s, "lease") || is(&s, "rental") || is(&s, "refurbish") {
                ctx.condition = add(&ctx.condition, &V::from(tpl(T5999, &[context.g("status")?])));
            }
        } else {
            ctx.status = parse_status(&context.g("status")?);
            if ctx.status.truthy() {
                ctx.condition = add(&ctx.condition, &V::from(tpl(T6005, &[ctx.status.clone()])));
            }
        }
    }

    ctx.temp = V::new_obj();

    // 원본 그대로: context.condition 이 없으면 Object.keys 가 던져 이 작업의 남은 일을 건너뛴다
    if !context.g("condition")?.keys_checked()?.is_empty() {
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

    // matches.length.toString()
    let len = ctx.matches.length()?;
    if len.is_nullish() {
        return Err(Throw::type_error(format!("Cannot read properties of {} (reading 'toString')", len.to_str())));
    }
    console_log!("matches.length", len.to_str());

    ctx.matches_condition = V::from("");

    if ctx.matches.g("length")?.truthy() {
        ctx.m = V::Num(0.0);
        while less_than(&ctx.m, &ctx.matches.g("length")?) == Some(true) {
            ctx.match_ = ctx.matches.gk(&ctx.m)?;

            if ctx.matches_condition.g("length")?.truthy() {
                ctx.matches_condition = add(&ctx.matches_condition, &V::from(" OR "));
            }

            let part = tpl(T6100, &[ctx.match_.g("id")?, team.g("id")?, now.clone()]);
            ctx.matches_condition = add(&ctx.matches_condition, &V::from(part));
            ctx.m = inc(&ctx.m);
        }

        console_log!("matches_condition", ctx.matches_condition.clone());

        console_log!("type", ctx.type_.clone());

        if let Err(err) = vector_rows(ctx, env).await {
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
        if let Err(err) = sql_rows(ctx, env).await {
            console_log!("text2sql err", err);
        }
    }

    ctx.system = V::from("Return the content related to the {search.text} value from the search results in a JSON structure.");

    let search = ctx.rag.g("search")?;
    let mut joined = spread(&search.g("sql")?.g("results")?)?;
    joined.extend(spread(&search.g("vector")?.g("results")?)?);
    ctx.content = context2results(&context, &V::from(joined), &ctx.language)?;

    // 원본 그대로: `var generation` 도 지우지 않는다 — 두 모델이 다 꺼져 있으면 앞 문맥의 답이 다시 쓰인다
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
async fn vector_rows(ctx: &mut Ctx, env: &Env) -> R<()> {
    let db = tpl(T6108, &[ctx.zoneRegion.clone(), ctx.type_.clone()]);
    let sql = tpl(T6108_1, &[ctx.type_.clone(), ctx.matches_condition.clone()]);
    ctx.results = select(env, &db, &sql).await?;

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

            // 원본 그대로: r 이 아니라 함수 범위의 i 로 지운다 — i 가 undefined(또는 범위 밖)면 여기서 던져
            // catch 로 가고 rag.search.vector 는 채워지지 않는다
            for k in ["from", "to", "cc", "bcc", "ref", "data"] {
                ctx.results.gk(&ctx.i)?.del(k)?;
            }

            ctx.r = inc(&ctx.r);
        }

        ctx.rag.g("search")?.s_("vector", obj! { "results" => ctx.results.clone() })?;
    }
    Ok(())
}

/// 6160~6212 의 try 안: 조건 SQL 로 찾은 행
async fn sql_rows(ctx: &mut Ctx, env: &Env) -> R<()> {
    let now = ctx.now.clone();
    let team = ctx.team.clone();

    // 원본 그대로: 조건 글에 created_at 수가 들어 있을 때만 붙는다(사실상 붙지 않음)
    if index_of(&ctx.condition.to_str(), &ctx.created_at.to_str()) > -1 {
        ctx.condition = add(&ctx.condition, &V::from(tpl(T6162, &[now.clone()])));
    }

    console_log!(tpl(T6165, &[ctx.type_.clone(), ctx.condition.clone(), team.g("id")?, ctx.orderBy.clone()]));

    let db = tpl(T6167, &[ctx.zoneRegion.clone(), ctx.type_.clone()]);
    let sql = tpl(T6167_1, &[ctx.type_.clone(), ctx.condition.clone(), team.g("id")?, ctx.orderBy.clone()]);
    ctx.results = select(env, &db, &sql).await?;

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

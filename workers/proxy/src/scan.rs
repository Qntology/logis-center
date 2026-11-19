//! `}else if(task.scan){ … }` — 원본 3027~5680 줄: 화면 스캔.
//!
//! 캐시된 페이지(pages)로 상세 화면인지 가리고(`isMore`), 아니면 LLM 으로 목록 · 상세를 구조화해 pages · items 문장을 쌓은 뒤
//! 아이템마다 [`crate::item1`] ~ [`crate::item4`] 를 차례로 돌린다(원본 3631~5517 의 반복 몸통). 끝에 상세 · 작업 문장.
//! 3069 의 try 가 거의 전부를 감싼다: 안의 예외(아이템 구역 포함)는 5664 의 catch 로 가서 console 표에 남기고 다음으로 넘어간다.

use logis_worker_common::gzip::{decode_utf8, gzip_json, ungzip};
use logis_worker_common::js::{self, Env};
use logis_worker_common::jsv::{add, js_trim, less_than, loose_eq, parse_str, tpl, Throw};
use logis_worker_common::tables::CENTER_REGION;
use logis_worker_common::util::{hash_id, merge_node};
use logis_worker_common::{arr, console_log, obj, stringify_str, R, V};

use crate::ctx::Ctx;
use crate::dom::Document;
use crate::flow::Flow;
use crate::handler::dec;
use crate::rules::{item2json, list2json};
use crate::templates::*;

fn is(v: &V, s: &str) -> bool {
    loose_eq(v, &V::from(s))
}

/// `a || b`
fn or(a: V, b: V) -> V {
    if a.truthy() {
        a
    } else {
        b
    }
}

/// `x.toUpperCase()` — 문자열이 아니면 원본처럼 던진다
fn upper(v: &V) -> R<V> {
    match v {
        V::Str(s) => Ok(V::from(logis_worker_common::jsv::to_upper(s))),
        V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'toUpperCase')", v.to_str()))),
        _ => Err(Throw::type_error("toUpperCase is not a function")),
    }
}

/// `new TextDecoder('utf-8').decode(ungzip(x))`
fn ungzip_text(v: &V) -> R<V> {
    Ok(V::from(decode_utf8(&ungzip(v)?)))
}

/// `String(err)` — Error.prototype.toString (메시지가 비면 이름만)
fn err_str(err: &Throw) -> String {
    if err.message.is_empty() {
        err.name.to_string()
    } else {
        format!("{}: {}", err.name, err.message)
    }
}

/// linkedom 의 선택자 인자: 문자열이 아니면 `selectors.includes(':scope')` 에서 던진다
fn selector_text(sel: &V) -> R<String> {
    match sel {
        V::Str(s) => Ok(s.clone()),
        V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'includes')", sel.to_str()))),
        _ => Err(Throw::type_error("selectors.includes is not a function")),
    }
}

/// `document.querySelectorAll(sel).length`
fn qsa(doc: &Document, sel: &V) -> R<usize> {
    Ok(doc.query_selector_all(&selector_text(sel)?)?.len())
}

/// `document.querySelector(sel)` 가 있는지
fn qs(doc: &Document, sel: &V) -> R<bool> {
    Ok(doc.query_selector(&selector_text(sel)?)?.is_some())
}

/// 3084 · 3104 · 3150 · 3346 의 `SELECT * FROM pages WHERE "id" = '${id}' AND "created_at" < ${created_at} LIMIT 1`.
/// 고침: 값을 SQL 글에 잇지 않고 묶는다 (작은따옴표가 든 값이 쿼리를 깨뜨리지 않게)
const SQL_PAGE: &str = "SELECT * FROM pages WHERE \"id\" = ?1 AND \"created_at\" < ?2 LIMIT 1";

/// pages 에서 id 하나 (`results`)
async fn select_page(env: &Env, id: V, created_at: &V) -> R<V> {
    js::query_bound(env, CENTER_REGION, SQL_PAGE, &[id, created_at.clone()]).await
}

/// `statements[key].push(env[db].prepare(sql).bind(...args))` — 칸이 비면 prepare 전에 던진다(원본의 평가 순서)
fn push_bind(env: &Env, statements: &V, key: &str, db: &str, sql: &str, args: impl FnOnce() -> R<Vec<V>>) -> R<()> {
    let slot = statements.g(key)?;
    if slot.is_nullish() {
        return Err(Throw::type_error(format!("Cannot read properties of {} (reading 'push')", slot.to_str())));
    }
    let stmt = env.prepare(db, sql)?;
    let params = args()?;
    let bound = js::bind(&stmt, &params)?;
    slot.push(bound)
}

/// `isMore(_page, selectors)` — 원본 3029~3063. task 를 닫아 쓰고, 안의 var(bool · selector · document · $target)는 이 함수의 지역
fn is_more(task: &V, page: &V, selectors: &V) -> R<V> {
    let mut found = false;
    let mut selector = V::from("");

    if is(&page.g("type")?, "goods") {
        selector = V::from(tpl(T3035, &[selectors.g("title")?, selectors.g("sale_price")?]));
    } else if is(&page.g("type")?, "order") {
        selector = V::from(tpl(
            T3037,
            &[
                selectors.g("tracking_number")?,
                selectors.g("payment_method")?,
                selectors.g("payment_origin")?,
                selectors.g("bank")?,
                selectors.g("card")?,
            ],
        ));
    } else if is(&page.g("type")?, "tracking") {
        selector = V::from(tpl(T3039, &[selectors.g("title")?, selectors.g("id")?, selectors.g("shipping_method")?]));
    } else if is(&page.g("type")?, "event") {
        selector = V::from(tpl(T3041, &[selectors.g("title")?, selectors.g("started_at")?, selectors.g("expired_at")?]));
    } else if is(&page.g("type")?, "coupon") {
        selector = V::from(tpl(T3043, &[selectors.g("title")?, selectors.g("started_at")?, selectors.g("expired_at")?]));
    }

    if selector.truthy() {
        let r: R<()> = (|| {
            let document = Document::parse(&tpl(T3049, &[task.g("body")?]));
            if qsa(&document, &selector)? > 0 {
                found = true;
            }
            Ok(())
        })();
        if let Err(err) = r {
            console_log!("more page err", err);
        }
    }

    Ok(V::Bool(found))
}

/// 3232~3240 · 3249~3257: `if(page.node){ … isDetail = true }else{ page = null }`
fn keep_if_node(ctx: &mut Ctx, task: &V) -> R<()> {
    if ctx.page.g("node")?.truthy() {
        ctx.page.s_("type", ctx.pageType.clone())?;
        ctx.page.s_("link", task.g("link")?)?;
        ctx.isDetail = V::Bool(true);
    } else {
        ctx.page = V::Null;
    }
    Ok(())
}

/// `models[key] -= 1`
fn use_model(ctx: &Ctx, key: &str) -> R<()> {
    ctx.models.s_(key, dec(&ctx.models.g(key)?))
}

/// 3197 · 3216 의 Gemini 열쇠 `gemini_llm_api+'-'+gemini_llm_model`
fn gemini_key(ctx: &Ctx) -> String {
    add(&add(&ctx.gemini_llm_api, &V::from("-")), &ctx.gemini_llm_model).to_key()
}

/// 3190~3195 · 3208~3214: 목록 판단 결과 읽기
fn read_list(ctx: &mut Ctx) -> R<()> {
    ctx.pageType = ctx.page.g("type")?;
    if ctx.page.g("items")?.truthy() {
        ctx.pageLength = ctx.page.g("items")?.length()?;
    }
    ctx.isDetail = ctx.page.g("detail")?;
    Ok(())
}

pub async fn run(ctx: &mut Ctx, env: &Env) -> R<Flow> {
    // INSERT 백터 생성 INSERT
    ctx.isDetail = ctx.task.g("detail")?;

    console_log!("isDetail000", stringify_str(&ctx.isDetail));

    match body(ctx, env).await {
        Ok(flow) => Ok(flow),
        Err(err) => {
            // 5664~5679: catch — console 표에 남기는 쓰기의 예외는 바깥(6443)으로 샌다
            console_log!(format!("inner err {}", err_str(&err)));

            let stmt = env.prepare(CENTER_REGION, T5667[0])?;
            let stmt = js::bind(
                &stmt,
                &[V::from(hash_id(&V::Undef)?), ctx.task.g("bcc")?, V::from(format!("inner err{}", err_str(&err))), ctx.now.clone()],
            )?;
            js::run(&stmt).await?;
            Ok(Flow::Next)
        }
    }
}

/// 3069~5663 (try 안)
async fn body(ctx: &mut Ctx, env: &Env) -> R<Flow> {
    let task = ctx.task.clone();

    // 고침: `var page` 는 작업마다 undefined 에서 시작한다 (원본은 앞 작업의 page 가 남아 3201 · 3246 의 `!page` 가 그 값을 봤다)
    ctx.page = V::Undef;
    ctx.pageType = V::from("");
    ctx.pageLength = V::Num(0.0);

    console_log!("task.href", task.g("href")?);

    let href = task.g("href")?;
    let url = url::Url::parse(&href.to_str()).map_err(|_| Throw::type_error("Invalid URL string."))?;
    ctx.url = V::from(url.as_str());
    let pathname = V::from(url.path());

    let cc = if task.g("detail")?.truthy() { upper(&task.g("cc")?)? } else { task.g("cc")? };
    ctx.pageId = V::from(hash_id(&add(&cc, &pathname))?);

    console_log!("pageId", ctx.pageId.clone());

    ctx.results = select_page(env, ctx.pageId.clone(), &ctx.created_at).await?;

    ctx.pages = ctx.results.clone();

    if ctx.pages.length()?.truthy() {
        ctx._page = ctx.pages.gi(0)?;

        if ctx._page.g("type")?.truthy() {
            ctx.decompressedJsonString = ungzip_text(&ctx._page.g("data")?)?;

            ctx.selectors = parse_str(&ctx.decompressedJsonString.to_str())?;

            ctx.isDetail = is_more(&task, &ctx._page, &ctx.selectors)?;

            console_log!("_page isDetail", stringify_str(&ctx.isDetail));

            if ctx.isDetail.truthy() {
                ctx.pageType = ctx._page.g("type")?;
            } else if loose_eq(&task.g("ref")?, &ctx.pageId) {
                let id = hash_id(&add(&task.g("cc")?, &upper(&pathname)?))?;
                ctx.results = select_page(env, V::from(id), &ctx.created_at).await?;

                if ctx.results.length()?.truthy() {
                    ctx._page = ctx.results.gi(0)?;

                    ctx.decompressedJsonString = ungzip_text(&ctx._page.g("data")?)?;

                    ctx._selectors = parse_str(&ctx.decompressedJsonString.to_str())?;

                    ctx.isDetail = is_more(&task, &ctx._page, &ctx._selectors)?;

                    if ctx.isDetail.truthy() {
                        ctx.pageType = ctx._page.g("type")?;
                    }

                    let (page, node_selectors) = (ctx._page.clone(), ctx._selectors.clone());
                    let r: R<()> = (|| {
                        let document = Document::parse(&tpl(T3120, &[task.g("body")?]));

                        // 고침: 풀어 읽은 _selectors.node 를 쓴다 (원본의 _page.data 는 풀지 않은 BLOB 이라 .node 가 undefined → 늘 던졌다)
                        if qsa(&document, &node_selectors.g("node")?)? > 0 {
                            task.s_("page", page.clone())?;
                        }
                        Ok(())
                    })();
                    if let Err(err) = r {
                        console_log!("more page err", err);
                    }
                }
            }

            let selectors = ctx.selectors.clone();
            let r: R<()> = (|| {
                let document = Document::parse(&tpl(T3135, &[task.g("body")?]));

                if qs(&document, &selectors.g("node")?)? {
                    console_log!("selectors.node", selectors.g("node")?);
                    // task.body = document.querySelector(selectors.node).innerHTML
                }
                Ok(())
            })();
            if let Err(err) = r {
                console_log!("cache page err", err);
            }
        }
    }

    if !ctx.isDetail.truthy() && task.g("referrer")?.truthy() {
        ctx.results = select_page(env, task.g("referrer")?, &ctx.created_at).await?;

        if ctx.results.length()?.truthy() {
            ctx._page = ctx.results.gi(0)?;

            if ctx._page.g("type")?.truthy() {
                ctx.decompressedJsonString = ungzip_text(&ctx._page.g("data")?)?;

                ctx.selectors = parse_str(&ctx.decompressedJsonString.to_str())?;

                if ctx.selectors.g("node")?.truthy() {
                    ctx.isDetail = is_more(&task, &ctx._page, &ctx.selectors)?;

                    if ctx.isDetail.truthy() {
                        ctx.pageType = ctx._page.g("type")?;
                    }
                }
            }
        }
    }

    ctx.content = crate::pug::convert_html_to_clean_pug(&task.g("body")?);

    console_log!("content.length", ctx.content.length()?);

    console_log!("isDetail1", stringify_str(&ctx.isDetail));

    // 3180~3219: 목록인지 · 어떤 유형인지
    if !ctx.isDetail.truthy() {
        ctx.system = V::from(js_trim(&tpl(T3181, &[ctx.language.clone(), list2json(&ctx.language)])).to_string());

        if ctx.models.g("deepinfra")?.truthy() {
            ctx.page = crate::llm::deepinfra(&ctx.deepinfra, &V::from("openai/gpt-oss-20b"), &ctx.system, &ctx.content, &V::Undef, &V::Undef).await?;

            read_list(ctx)?;

            use_model(ctx, "deepinfra")?;
        }

        if !ctx.page.truthy() && ctx.gemini_llm_api.truthy() {
            let user = tpl(T3202, &[ctx.language.clone(), ctx.system.clone(), ctx.content.clone()]);
            ctx.page = crate::llm::gemini(&ctx.gemini_llm_api, &ctx.gemini_llm_model, &V::from(""), &V::from(user), &V::Undef, &V::Undef).await?;

            read_list(ctx)?;

            let key = gemini_key(ctx);
            use_model(ctx, &key)?;
        }
    }

    // 3222~3262: 상세(폼) 구조
    if (!ctx.isDetail.truthy() && !ctx.pageLength.truthy() && ctx.pageType.truthy()) || ctx.isDetail.truthy() {
        ctx.system = V::from(js_trim(&tpl(T3223, &[ctx.language.clone(), item2json(&ctx.pageType, &task.g("href")?)])).to_string());

        if ctx.models.g("deepinfra")?.truthy() {
            ctx.page = crate::llm::deepinfra(&ctx.deepinfra, &V::from("openai/gpt-oss-20b"), &ctx.system, &ctx.content, &V::Undef, &V::Undef).await?;

            keep_if_node(ctx, &task)?;

            use_model(ctx, "deepinfra")?;
        }

        if !ctx.page.truthy() && ctx.gemini_llm_api.truthy() {
            ctx.page = crate::llm::gemini(&ctx.gemini_llm_api, &ctx.gemini_llm_model, &ctx.system, &ctx.content, &V::Undef, &V::Undef).await?;

            keep_if_node(ctx, &task)?;

            let key = gemini_key(ctx);
            use_model(ctx, &key)?;
        }
    }

    if !ctx.page.truthy() {
        ctx.fallback = V::from("page overflow");

        return Ok(Flow::Continue);
    }

    let page = ctx.page.clone();
    page.s_("type", ctx.pageType.clone())?;
    page.s_("from", task.g("from")?)?;
    page.s_("to", task.g("to")?)?;
    page.s_("cc", task.g("cc")?)?;
    let cc = if ctx.isDetail.truthy() { upper(&task.g("cc")?)? } else { task.g("cc")? };
    page.s_("bcc", V::from(hash_id(&add(&page.g("type")?, &cc))?))?;

    if !ctx.team.g("data")?.g("base")?.g("pages")?.gk(&task.g("cc")?)?.truthy() {
        ctx.team.g("data")?.g("base")?.g("pages")?.sk(&task.g("cc")?, V::new_obj())?;
    }

    if page.g("type")?.truthy() {
        let by_cc = ctx.team.g("data")?.g("base")?.g("pages")?.gk(&task.g("cc")?)?;
        if !by_cc.gk(&page.g("type")?)?.truthy() {
            by_cc.sk(&page.g("type")?, obj! { "draft" => 0, "count" => 0 })?;
        }

        let base = ctx.team.g("data")?.g("base")?;
        if !base.gk(&page.g("type")?)?.truthy() {
            base.sk(&page.g("type")?, obj! { "draft" => 0, "count" => 0 })?;
        }
    }

    ctx.selectors = obj! {
        "type" => page.g("type")?,
        "text" => or(page.g("text")?, V::from("")),
        "node" => or(page.g("node")?, V::from("")),
        "list" => or(page.g("list")?, V::from("")),
        "item" => or(page.g("item")?, V::from("")),
        "more" => or(page.g("more")?, V::from("")),
        "next" => or(page.g("next")?, V::from("")),
        "link" => task.g("link")?,
        "time" => ctx.now.clone(),
        "origin" => if task.g("origin")?.truthy() { task.g("origin")? } else { V::from("") },
    };

    ctx.detail = obj! {
        "id" => hash_id(&add(&add(&ctx.pageType, &upper(&task.g("cc")?)?), &pathname))?,
        "type" => ctx.pageType.clone(),
        "from" => task.g("from")?,
        "to" => task.g("to")?,
        "cc" => task.g("cc")?,
        "bcc" => hash_id(&add(&page.g("type")?, &upper(&task.g("cc")?)?))?,
        "ref" => task.g("ref")?,
        "digest" => "",
        "data" => V::Null,
        "created_at" => ctx.now.clone(),
        "updated_at" => ctx.now.clone(),
    };

    console_log!("isDetail2", stringify_str(&ctx.isDetail));

    // 고침: `var before` 는 작업마다 undefined 에서 시작한다 (원본은 앞 작업의 before 가 남아 3451 이 그 값을 봤다)
    ctx.before = V::Undef;

    if ctx.isDetail.truthy() {
        ctx.selectors.s_("detail", V::Bool(true))?;

        let id = ctx.detail.g("id")?;
        ctx.pageId = id.clone();
        ctx.talk.s_("ref", id)?;

        ctx.detail = V::Null;

        console_log!("pageId", ctx.pageId.clone());

        ctx.results = select_page(env, ctx.pageId.clone(), &ctx.created_at).await?;

        // 고침: `var _page` 는 여기서 undefined 에서 시작한다 (원본은 3089 · 3107 · 3153 의 _page 가 남아 3409 가 그 값을 봤다 —
        // 그 행의 data 가 NULL 이면 `_page.data.item` 에서 던져 작업이 통째로 버려졌다)
        ctx._page = V::Undef;
        if ctx.results.length()?.truthy() {
            ctx._page = ctx.results.gi(0)?;

            ctx.decompressedJsonString = ungzip_text(&ctx._page.g("data")?)?;

            let data = parse_str(&ctx.decompressedJsonString.to_str())?;
            ctx._page.s_("data", data)?;

            ctx.before = ctx._page.g("data")?;
        }

        ctx.item = ctx.page.deep_clone();
        let item = ctx.item.clone();

        // for (var p in item) — 값만 바뀌고 키는 늘지 않아 처음 키 목록으로 돈다
        for p in item.keys() {
            ctx.p = V::from(p.clone());
            if !item.has(&p) {
                continue;
            }
            ctx.prop = item.g(&p)?;
            let prop = ctx.prop.clone();

            if !prop.truthy() {
                continue;
            }
            if !prop.g("selector")?.is_undef() {
                ctx.selectors.s_(&p, prop.g("selector")?)?;
                item.s_(&p, prop.g("value")?)?;
            } else if prop.g("length")?.is_num() {
                ctx.i = V::Num(0.0);
                while less_than(&ctx.i, &prop.g("length")?) == Some(true) {
                    ctx.option = prop.gk(&ctx.i)?;
                    let option = ctx.option.clone();

                    // 고침: null 원소는 건너뛴다 (typeof null == "object" 라 원본은 Object.keys(null) 에서 던졌다)
                    if option.truthy() && option.type_of() == "object" && !option.keys_checked()?.is_empty() {
                        for n in option.keys() {
                            ctx.n = V::from(n.clone());
                            if !option.has(&n) {
                                continue;
                            }
                            ctx.opt = option.g(&n)?;
                            let opt = ctx.opt.clone();

                            if n == "selector" {
                                ctx.selectors.s_(&tpl(T3382, &[V::from(p.clone())]), option.g("selector")?)?;

                                item.g(&p)?.sk(&ctx.i, option.g("value")?)?;
                            } else if opt.truthy() && opt.type_of() == "object" && !opt.keys_checked()?.is_empty() {
                                for o in opt.keys() {
                                    ctx.o = V::from(o.clone());
                                    if !opt.has(&o) {
                                        continue;
                                    }
                                    ctx.obj = opt.g(&o)?;

                                    if o == "selector" {
                                        ctx.selectors.s_(&tpl(T3392, &[V::from(p.clone()), V::from(n.clone())]), opt.g("selector")?)?;

                                        item.g(&p)?.gk(&ctx.i)?.s_(&n, opt.g("value")?)?;
                                    }
                                }
                            }
                        }
                    }
                    ctx.i = V::Num(ctx.i.to_num() + 1.0);
                }
            }
        }

        if ctx._page.truthy() && !ctx._page.g("data")?.g("item")?.truthy() {
            ctx.item.s_("ref", ctx._page.g("ref")?)?;
        }

        ctx.page.s_("items", arr![ctx.item.clone()])?;
    } else {
        ctx.arr = gzip_json(&obj! {
            "type" => ctx.page.g("type")?,
            "link" => task.g("link")?,
            "origin" => task.g("origin")?,
            "detail" => true,
            "node" => true,
            "item" => "",
        })?;

        ctx.detail.s_("data", ctx.arr.clone())?;
    }

    console_log!("pageId", ctx.pageId.clone());

    ctx.items = V::new_arr();

    if ctx.page.g("items")?.truthy() && ctx.page.g("items")?.length()?.truthy() {
        ctx.items = ctx.page.g("items")?;
    }

    ctx.talk.s_("text", ctx.page.g("text")?)?;

    // page.ref = task.referrer ? task.referrer : ''
    let page = ctx.page.clone();
    let ref_ = hash_id(&add(&add(&ctx.team.g("id")?, &task.g("cc")?), &page.g("link")?))?;
    page.s_("ref", V::from(ref_))?;

    page.s_("data", ctx.selectors.clone())?;

    if ctx.before.truthy() && ctx.pages.length()?.truthy() {
        ctx._page = ctx.pages.gi(0)?;

        ctx.after = ctx.selectors.clone();

        let (before, after, old, task) = (ctx.before.clone(), ctx.after.clone(), ctx._page.clone(), task.clone());
        let r: R<()> = (|| {
            let document = Document::parse(&tpl(T3457, &[task.g("body")?]));

            if !before.keys_checked()?.is_empty() {
                for selector in before.keys() {
                    ctx.selector = V::from(selector.clone());
                    // typeof selector == "string" 는 늘 참
                    if !before.has(&selector) {
                        continue;
                    }
                    let inner: R<()> = (|| {
                        let b = qsa(&document, &V::from(tpl(T3464, &[before.g(&selector)?])))?;
                        let a = qsa(&document, &V::from(tpl(T3465, &[after.g(&selector)?])))?;

                        let src = if b > 0 && a == 0 {
                            Some(&before)
                        } else if b == 0 && a > 0 {
                            Some(&after)
                        } else if b > 0 && a > 0 {
                            Some(if b < a { &before } else { &after })
                        } else {
                            None
                        };
                        if let Some(src) = src {
                            // _page.data[selector] = page.data[selector] = src[selector] + ''
                            // 원본 그대로: _page 는 pages[0] 이라 data 가 풀지 않은 BLOB 배열 — 이름 있는 속성은 남지 않는다
                            let old_data = old.g("data")?;
                            let value = add(&src.g(&selector)?, &V::from(""));
                            page.g("data")?.s_(&selector, value.clone())?;
                            old_data.s_(&selector, value)?;
                        }
                        Ok(())
                    })();
                    // console.log('selector err',err);
                    let _ = inner;
                }
            }
            Ok(())
        })();
        // console.log('_page.data err',err);
        let _ = r;

        // _page.data = before

        ctx.page = merge_node(&ctx._page, &ctx.page);

        ctx.selectors = ctx.page.g("data")?;
    }

    console_log!("page.data", stringify_str(&ctx.page.g("data")?));

    let page = ctx.page.clone();
    let id = if task.g("page")?.truthy() { task.g("page")?.g("id")? } else { ctx.pageId.clone() };
    page.s_("id", id)?;

    page.s_("digest", V::from(""))?;

    console_log!("page.id", page.g("id")?);

    console_log!("page", stringify_str(&page));

    ctx.arr = gzip_json(&page.g("data")?)?;

    page.s_("data", ctx.arr.clone())?;

    if page.g("type")?.truthy() {
        let now = ctx.now.clone();
        push_bind(env, &ctx.statements, CENTER_REGION, CENTER_REGION, T3520[0], || {
            Ok(vec![
                page.g("id")?,
                page.g("type")?,
                page.g("from")?,
                page.g("to")?,
                page.g("cc")?,
                page.g("bcc")?,
                page.g("ref")?,
                page.g("data")?,
                now.clone(),
                now.clone(),
            ])
        })?;

        let key = tpl(T3547, &[ctx.zoneRegion.clone()]);
        let db = tpl(T3548, &[ctx.zoneRegion.clone()]);
        push_bind(env, &ctx.statements, &key, &db, T3548_1[0], || {
            Ok(vec![
                page.g("id")?,
                V::from("pages"),
                page.g("from")?,
                page.g("to")?,
                page.g("cc")?,
                page.g("bcc")?,
                page.g("ref")?,
                page.g("digest")?,
                page.g("data")?,
                now.clone(),
                now.clone(),
            ])
        })?;
    }

    ctx.talk.s_("type", page.g("type")?)?;

    // items에서 시작 시간 · 최저가 최대가격 · 최저 수량 최대 수량 값 가져오기

    console_log!("진입전 items");

    if ctx.items.length()?.truthy() {
        console_log!("items", stringify_str(&ctx.items));

        // 주문이후의 절차는 주문번호로 매칭해야함 (원본 3599~3629 주석)

        // for(var i = 0; i < items.length; i++) — 몸통은 네 구역. continue 여도 i++ 은 돈다
        ctx.i = V::Num(0.0);
        while less_than(&ctx.i, &ctx.items.length()?) == Some(true) {
            let mut flow = crate::item1::run(ctx, env).await?;
            if flow == Flow::Next {
                flow = crate::item2::run(ctx, env).await?;
            }
            if flow == Flow::Next {
                flow = crate::item3::run(ctx, env).await?;
            }
            if flow == Flow::Next {
                flow = crate::item4::run(ctx, env).await?;
            }
            if flow == Flow::Break {
                break;
            }
            ctx.i = V::Num(ctx.i.to_num() + 1.0);
        }
    }

    let task = ctx.task.clone();
    let page = ctx.page.clone();
    let now = ctx.now.clone();

    if ctx.detail.truthy() && page.g("type")?.truthy() {
        let detail = ctx.detail.clone();
        push_bind(env, &ctx.statements, CENTER_REGION, CENTER_REGION, T5522[0], || {
            Ok(vec![
                detail.g("id")?,
                detail.g("type")?,
                detail.g("from")?,
                detail.g("to")?,
                detail.g("cc")?,
                detail.g("bcc")?,
                detail.g("ref")?,
                detail.g("data")?,
                now.clone(),
                now.clone(),
            ])
        })?;

        let key = tpl(T5549, &[ctx.zoneRegion.clone()]);
        let db = tpl(T5550, &[ctx.zoneRegion.clone()]);
        push_bind(env, &ctx.statements, &key, &db, T5550_1[0], || {
            Ok(vec![
                detail.g("id")?,
                V::from("pages"),
                detail.g("from")?,
                detail.g("to")?,
                detail.g("cc")?,
                detail.g("bcc")?,
                detail.g("ref")?,
                detail.g("digest")?,
                detail.g("data")?,
                now.clone(),
                now.clone(),
            ])
        })?;
    }

    task.s_("title", page.g("text")?)?; // Analyze the provided Pug template and return it in the following JSON format

    task.s_("semantic", page.g("text")?)?;

    if ctx.items.length()?.truthy() {
        // var type = talk.type = page.type
        let ptype = page.g("type")?;
        ctx.talk.s_("type", ptype.clone())?;
        ctx.type_ = ptype;

        if is(&page.g("type")?, "sales") {
            ctx.type_ = V::from("sales");
        } else if is(&page.g("type")?, "goods") || is(&page.g("type")?, "order") {
            ctx.type_ = V::from("sales");
        } else if is(&page.g("type")?, "event") || is(&page.g("type")?, "coupon") {
            ctx.type_ = V::from("event");
        }

        ctx.arr = gzip_json(&obj! {
            "no" => task.g("no")?,
            "title" => page.g("text")?,
            "semantic" => task.g("semantic")?,
        })?;

        task.s_("data", ctx.arr.clone())?;
    } else {
        // 정보가 없으면 클라이언트에서 정보를 찾지 못하였다고 안내하기 · 가능한 카테고리 안내 메세지 노출해야함

        ctx.talk.s_("type", V::from("empty"))?;

        task.s_("data", V::Null)?;
    }

    ctx.arr = gzip_json(&obj! {
        "type" => page.g("type")?,
        "text" => task.g("semantic")?,
        "link" => task.g("link")?,
        "time" => now.clone(),
        "origin" => if task.g("origin")?.truthy() { task.g("origin")? } else { V::from("") },
    })?;

    let key = tpl(T5630, &[ctx.zoneRegion.clone()]);
    let db = tpl(T5631, &[ctx.zoneRegion.clone()]);
    let arr = ctx.arr.clone();
    push_bind(env, &ctx.statements, &key, &db, T5631_1[0], || {
        Ok(vec![
            task.g("id")?,
            task.g("type")?,
            task.g("from")?,
            task.g("to")?,
            task.g("cc")?,
            task.g("bcc")?,
            task.g("ref")?,
            task.g("digest")?,
            arr,
            now.clone(),
            now.clone(),
        ])
    })?;

    Ok(Flow::Next)
}

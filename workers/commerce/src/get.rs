//! 원본 2455~2869 (GET: `Content-Type: application/json` 이면 동기화 JSON, 아니면 셸 HTML).
//!
//! JSON 의 BLOB 모양: D1 은 BLOB 을 숫자 배열로 돌려주므로 행의 `data` 는 `[31,139,…]` 이다. 그런데 원본이 `gzip(…, { to: 'arraybuffer' })`
//! 로 만든 값은 pako 2.2 에서 `to` 를 무시한 **Uint8Array** 라 `JSON.stringify` 가 `{"0":31,"1":139,…}` 로 쓴다(데스크톱 앱의 `bytesOf` 가
//! 이 모양을 읽는다). 이 응답에 닿는 그런 값은 home.data(1055) · `_page.data`(1360) · talk.data(2606) 이고, 모두 Uint8Array 다 —
//! ArrayBuffer(`.buffer`)는 D1 바인딩 · 세션의 user/team 객체에만 쓰이고 응답에 닿지 않는다(users 의 data 는 1124 에서 지운다,
//! cookies 에는 바이트가 없다). 그래서 응답을 쓸 때 [`V::Bytes`] 를 Uint8Array 꼴로 쓴다([`json_body`]).

use std::rc::Rc;

use logis_worker_common::gzip::{decode_utf8, gzip, ungzip};
use logis_worker_common::js::{self, Env};
use logis_worker_common::jsv::{parse_int, stringify_checked, to_lower, tpl, Throw};
use logis_worker_common::web::{headers_set, headers_vec, Body, Req, Resp};
use logis_worker_common::{console_log, eth, loose_eq, obj, parse_str, R, V};

use crate::ctx::Ctx;
use crate::flow::Step;
use crate::render::render_html;
use crate::templates::*;

/// `env[name].prepare(sql)` — JS 는 `env[name].prepare` 를 인자(SQL 템플릿)보다 먼저 평가하므로, 바인딩이 없으면
/// SQL 식(예: 2746 의 `item.index`)이 던지기 전에 `reading 'prepare'` 로 던진다
fn prepare(env: &Env, name: &str, sql: impl FnOnce() -> R<String>) -> R<V> {
    if env.get(name).is_nullish() {
        return env.prepare(name, "");
    }
    let sql = sql()?;
    env.prepare(name, &sql)
}

/// `var { results, success, error } = await stmt.all()`
async fn all_into(ctx: &mut Ctx, stmt: V) -> R<()> {
    let r = js::all(&stmt).await?;
    ctx.results = r.g("results")?;
    ctx.success = r.g("success")?;
    ctx.error = r.g("error")?;
    Ok(())
}

/// `i < results.length`
fn below_len(i: &V, arr: &V) -> R<bool> {
    Ok(i.to_num() < arr.length()?.to_num())
}

/// `i++`
fn inc(i: &V) -> V {
    V::Num(i.to_num() + 1.0)
}

/// `if(Object.keys(detail).length){ if(detail.id == item.id){ continue } }`
fn same_as_detail(detail: &V, item: &V) -> R<bool> {
    if !detail.keys_checked()?.is_empty() && loose_eq(&detail.g("id")?, &item.g("id")?) {
        return Ok(true);
    }
    Ok(false)
}

/// `conditions[k] ? conditions[k] : condition`
fn cond_for(ctx: &Ctx, k: &V) -> R<V> {
    let c = ctx.conditions.gk(k)?;
    // 원본 그대로: conditions 는 늘 {} 라 늘 condition 이다 (k 가 "constructor" 같은 Object.prototype 이름이면 JS 는 함수를 넣는데,
    // 이 모형에는 원형 사슬이 없어 condition 이 된다 — page.type · type 은 그런 값이 아니다)
    Ok(if c.truthy() { c } else { ctx.condition.clone() })
}

/// `new URL(s)` 의 쓰는 속성 (session.rs 의 url_obj 와 같은 꼴)
fn url_obj(u: &url::Url) -> V {
    let hostname = u.host_str().unwrap_or("").to_string();
    let host = match u.port() {
        Some(p) => format!("{hostname}:{p}"),
        None => hostname.clone(),
    };
    obj! {
        "href" => u.as_str(),
        "origin" => u.origin().ascii_serialization(),
        "protocol" => format!("{}:", u.scheme()),
        "host" => host,
        "hostname" => hostname,
        "pathname" => u.path(),
        "search" => url_search(u),
    }
}

/// `url.search` (빈 질의 `?` 는 "")
fn url_search(u: &url::Url) -> String {
    match u.query() {
        Some(q) if !q.is_empty() => format!("?{q}"),
        _ => String::new(),
    }
}

/// V8 `Array.prototype.sort(comparefn)` (TimSort). 원소가 64 개 미만이면 최소 run 길이가 n 이라 run 하나 —
/// CountAndMakeRun(내림차순 run 은 뒤집음) 뒤 BinaryInsertionSort 만 돈다. 비교가 NaN(created_at 이 없는 행)이어도
/// V8 과 같은 자리에 놓으려고 그 순서 그대로 옮겼다 (talks 는 최대 10 + 10 개라 병합 단계는 닿지 않는다).
/// node 의 sort 와 n ≤ 20 · created_at 에 undefined/null 이 섞인 무작위 20만 경우를 견줘 모두 같았다(같은 알고리즘의 JS 판)
fn v8_sort(v: &mut [V], comparefn: impl Fn(&V, &V) -> f64) {
    // SortCompareUserFn: 결과가 NaN 이면 +0
    let compare = |x: &V, y: &V| {
        let r = comparefn(x, y);
        if r.is_nan() { 0.0 } else { r }
    };
    let n = v.len();
    if n < 2 {
        return;
    }
    // CountAndMakeRun(0, n)
    let mut run = 2;
    let descending = compare(&v[1], &v[0]) < 0.0;
    let mut prev = v[1].clone();
    for idx in 2..n {
        let cur = v[idx].clone();
        let order = compare(&cur, &prev);
        if descending {
            if order >= 0.0 {
                break;
            }
        } else if order < 0.0 {
            break;
        }
        prev = cur;
        run += 1;
    }
    if descending {
        v[..run].reverse();
    }
    // BinaryInsertionSort(0, run, n) — run ≥ 2 라 start = run
    for start in run..n {
        let (mut left, mut right) = (0usize, start);
        let pivot = v[start].clone();
        while left < right {
            let mid = left + ((right - left) >> 1);
            if compare(&pivot, &v[mid]) < 0.0 {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        v[left..=start].rotate_right(1); // v[left] = pivot, 사이는 한 칸씩 뒤로
    }
}

/// 응답 JSON 용 사본: [`V::Bytes`] 를 Uint8Array 의 `JSON.stringify` 꼴(`{"0":31,"1":139,…}`)로 바꾼다 (머리 주석 참고).
/// 순환은 JSON.stringify 처럼 TypeError (조상 경로만 본다 — 같은 객체를 두 번 만나는 것은 괜찮다)
fn json_view(v: &V, stack: &mut Vec<*const ()>) -> R<V> {
    match v {
        V::Bytes(b) => {
            let out = V::new_obj();
            for (i, x) in b.borrow().iter().enumerate() {
                out.put(&i.to_string(), V::Num(*x as f64));
            }
            Ok(out)
        }
        V::Arr(a) => {
            let p = Rc::as_ptr(a) as *const ();
            if stack.contains(&p) {
                return Err(Throw::type_error("Converting circular structure to JSON"));
            }
            stack.push(p);
            let mut items = Vec::new();
            for x in v.items() {
                items.push(json_view(&x, stack)?);
            }
            stack.pop();
            Ok(V::from(items))
        }
        V::Obj(o) => {
            let p = Rc::as_ptr(o) as *const ();
            if stack.contains(&p) {
                return Err(Throw::type_error("Converting circular structure to JSON"));
            }
            stack.push(p);
            let out = V::new_obj();
            for (k, x) in v.entries() {
                out.put(&k, json_view(&x, stack)?);
            }
            stack.pop();
            Ok(out)
        }
        other => Ok(other.clone()),
    }
}

/// `JSON.stringify(x)` (응답 본문)
fn json_body(v: &V) -> R<String> {
    Ok(stringify_checked(&json_view(v, &mut Vec::new())?)?.unwrap_or_default())
}

/// `for(var i = 0; i < results.length; i++){ var item = results[i]; item.table = table; if(detail…) continue; items.push(item) }`
/// — 2502 · 2526 · 2707 · 2729 · 2750 · 2778 의 꼴 (셈 변수 i · item 은 함수 범위 var)
fn push_rows(ctx: &mut Ctx, table: V) -> R<()> {
    ctx.i = V::Num(0.0);
    while below_len(&ctx.i, &ctx.results)? {
        ctx.item = ctx.results.gk(&ctx.i)?;
        ctx.item.s_("table", table.clone())?;
        if same_as_detail(&ctx.detail, &ctx.item)? {
            ctx.i = inc(&ctx.i);
            continue;
        }
        ctx.items.push(ctx.item.clone())?;
        ctx.i = inc(&ctx.i);
    }
    Ok(())
}

pub async fn run(ctx: &mut Ctx, env: &Env, _req: &Req) -> R<Step> {
    // 2456
    if loose_eq(&ctx.contentType, &V::from("application/json")) {
        headers_set(&ctx.headers, "Content-Type", &ctx.contentType)?;
        let query = ctx.req.g("query")?;

        // 2459
        ctx.from = query.g("from")?;
        if eth::is_address(&query.g("from")?) {
            ctx.from = query.g("from")?;
        }

        ctx.to = V::from("");
        if eth::is_address(&query.g("to")?) {
            ctx.to = query.g("to")?;
        }

        ctx.ref_ = V::from("");
        if query.g("ref")?.truthy() && eth::is_address(&query.g("ref")?) {
            ctx.ref_ = query.g("ref")?;
        }

        // 2477 `!isNaN(req.query.day) ? parseInt(req.query.day) : 0`
        let q_day = query.g("day")?;
        ctx.day = if !q_day.to_num().is_nan() { V::Num(parse_int(&q_day.to_str())) } else { V::Num(0.0) };

        // 2479
        ctx.condition = V::from(tpl(T2481, &[ctx.created_at.clone()]));
        ctx.conditions = V::new_obj();

        // 2484 — cookies.href 가 URL 이 아니면 TypeError (바깥 try → 418)
        let u = url::Url::parse(&ctx.cookies.g("href")?.to_str()).map_err(|_| Throw::type_error("Invalid URL string."))?;
        ctx.url = url_obj(&u);
        ctx.link = V::from(to_lower(&(u.path().to_string() + &url_search(&u))));

        // 2490
        ctx.talks = V::new_arr();

        if ctx.member.truthy() && ctx.cookies.g("address")?.truthy() {
            member_rows(ctx, env).await?;
        }

        // 2853
        let cookies = ctx.cookies.clone();
        console_log!(T2853[0]);
        let type_or_all = if ctx.type_.truthy() { ctx.type_.clone() } else { V::from("all") };
        console_log!(tpl(T2854, &[cookies.g("cc")?, cookies.g("team")?, type_or_all]));
        console_log!(tpl(
            T2855,
            &[ctx.pages.length()?, ctx.tasks.length()?, ctx.talks.length()?, ctx.items.length()?, ctx.users.length()?]
        ));

        // 2857 `[...pages, ...tasks, ...talks, ...items, ...users, ...hints]`
        let mut results = Vec::new();
        for list in [&ctx.pages, &ctx.tasks, &ctx.talks, &ctx.items, &ctx.users, &ctx.hints] {
            if !list.is_arr() {
                return Err(Throw::type_error(format!("{} is not iterable", list.to_str())));
            }
            results.extend(list.items());
        }
        let body = json_body(&obj! { "results" => V::from(results), "session" => cookies })?;

        let mut r = Resp::new(200, Body::Text(body));
        r.headers = headers_vec(&ctx.headers);
        return Ok(Step::Return(r));
    }

    // 2863
    headers_set(&ctx.headers, "Content-Type", &V::from("text/html; charset=utf-8"))?;
    let mut r = Resp::new(200, Body::Text(render_html(&ctx.cookies, &V::new_arr())));
    r.headers = headers_vec(&ctx.headers);
    Ok(Step::Return(r))
}

/// 2492~2849 `if(member && cookies.address){ … }`
async fn member_rows(ctx: &mut Ctx, env: &Env) -> R<()> {
    let cookies = ctx.cookies.clone();
    let items_db = tpl(T2496, &[ctx.zoneRegion.clone()]);

    // 2494 relate update item
    let stmt = prepare(env, &items_db, || Ok(tpl(T2497, &[cookies.g("cc")?, cookies.g("team")?, ctx.condition.clone()])))?;
    all_into(ctx, stmt).await?;

    console_log!("results.length", ctx.results.length()?);

    if ctx.results.length()?.truthy() {
        push_rows(ctx, V::from("items"))?;
    }

    // 2518 recent update item
    let items_db = tpl(T2520, &[ctx.zoneRegion.clone()]);
    let stmt = prepare(env, &items_db, || Ok(tpl(T2521, &[cookies.g("cc")?, cookies.g("team")?, ctx.condition.clone()])))?;
    all_into(ctx, stmt).await?;

    if ctx.results.length()?.truthy() {
        push_rows(ctx, V::from("items"))?;
    }

    // 2540 recent draft item
    let items_db = tpl(T2542, &[ctx.zoneRegion.clone()]);
    let stmt = prepare(env, &items_db, || Ok(tpl(T2543, &[cookies.g("team")?, ctx.condition.clone()])))?;
    all_into(ctx, stmt).await?;

    if ctx.results.length()?.truthy() {
        ctx.i = V::Num(0.0);
        while below_len(&ctx.i, &ctx.results)? {
            ctx.draft = ctx.results.gk(&ctx.i)?;
            ctx.draft.s_("table", V::from("draft"))?;

            // 2552 `cookies.draft[draft.ref]++` — 원본 그대로: 세션이 cookies.draft 를 {} 로 둔 때만 돌고(없으면 TypeError → 418),
            // 처음 보는 ref 는 undefined++ 라 NaN (JSON 에서 null). 그 뒤로도 NaN + 1 = NaN
            let drafts = cookies.g("draft")?;
            let key = ctx.draft.g("ref")?;
            let old = drafts.gk(&key)?;
            drafts.sk(&key, V::Num(old.to_num() + 1.0))?;

            if same_as_detail(&ctx.detail, &ctx.draft)? {
                ctx.i = inc(&ctx.i);
                continue;
            }

            ctx.items.push(ctx.draft.clone())?;
            ctx.i = inc(&ctx.i);
        }
    }

    // 2564 — logisRegion 이 없으면 env["undefined"].prepare 로 TypeError
    let logis_db = ctx.logisRegion.to_key();
    let stmt = prepare(env, &logis_db, || Ok(tpl(T2566, &[cookies.g("address")?, cookies.g("cc")?, ctx.condition.clone()])))?;
    all_into(ctx, stmt).await?;

    if ctx.results.length()?.truthy() {
        ctx.i = V::Num(0.0);
        while below_len(&ctx.i, &ctx.results)? {
            ctx.task = ctx.results.gk(&ctx.i)?;
            ctx.task.s_("table", V::from("tasks"))?;
            ctx.task.s_("ref", V::from(""))?;
            ctx.tasks.push(ctx.task.clone())?;
            ctx.i = inc(&ctx.i);
        }
    }

    console_log!("[GET] type", ctx.type_);

    // 2584
    let talks_db = tpl(T2586, &[ctx.zoneRegion.clone()]);
    let stmt = prepare(env, &talks_db, || {
        let c = cond_for(ctx, &ctx.page.g("type")?)?;
        Ok(tpl(T2587, &[cookies.g("cc")?, ctx.itemId.clone(), c]))
    })?;
    all_into(ctx, stmt).await?;

    if ctx.results.length()?.truthy() {
        ctx.i = V::Num(0.0);
        while below_len(&ctx.i, &ctx.results)? {
            ctx.talk = ctx.results.gk(&ctx.i)?;
            ctx.talk.s_("table", V::from("talks"))?;

            // 2596 — 원본은 pako.ungzip(숫자 배열) (session.rs 와 같이 공용 ungzip 으로 푼다)
            ctx.decompressedJsonString = V::from(decode_utf8(&ungzip(&ctx.talk.g("data")?)?));
            ctx.talk_data = parse_str(&ctx.decompressedJsonString.to_str())?;

            if ctx.talk_data.truthy() && ctx.talk_data.g("search")?.truthy() {
                ctx.talk_data.del("search")?;
            }

            // 2606 `gzip(…, { to: 'arraybuffer' })` — pako 2.2 는 to 를 무시하고 Uint8Array 를 낸다 (응답에서 {"0":…} 꼴)
            let text = stringify_checked(&ctx.talk_data)?.unwrap_or_default();
            ctx.talk.s_("data", V::bytes(gzip(text.as_bytes())))?;

            ctx.talks.push(ctx.talk.clone())?;
            ctx.i = inc(&ctx.i);
        }

        // 2609
        let talks_db = tpl(T2611, &[ctx.zoneRegion.clone()]);
        let stmt = prepare(env, &talks_db, || Ok(tpl(T2612, &[cookies.g("cc")?, ctx.itemId.clone()])))?;
        all_into(ctx, stmt).await?;

        if ctx.results.length()?.truthy() {
            ctx.item = ctx.results.gi(0)?;
            // `for (const itemType in item)` — for-in 은 undefined 에서도 던지지 않는다
            for key in ctx.item.keys() {
                if ctx.item.has(&key) {
                    cookies.s_(&key, ctx.item.g(&key)?)?;
                }
            }
        }
    }

    // 2624
    let talks_db = tpl(T2626, &[ctx.zoneRegion.clone()]);
    let stmt = prepare(env, &talks_db, || {
        let c = cond_for(ctx, &ctx.page.g("type")?)?;
        Ok(tpl(T2627, &[cookies.g("team")?, cookies.g("address")?, c]))
    })?;
    all_into(ctx, stmt).await?;

    if ctx.results.length()?.truthy() {
        ctx.i = V::Num(0.0);
        while below_len(&ctx.i, &ctx.results)? {
            ctx.talk = ctx.results.gk(&ctx.i)?;
            ctx.talk.s_("table", V::from("talks"))?;
            ctx.talks.push(ctx.talk.clone())?;
            ctx.i = inc(&ctx.i);
        }
    }

    // 2638 `talks = talks.sort((a, b) => b.created_at - a.created_at)` (제자리 정렬, 같은 배열)
    if ctx.talks.length()?.truthy() {
        if let V::Arr(a) = &ctx.talks {
            let mut list = a.borrow().clone();
            v8_sort(&mut list, |a, b| b.get("created_at").to_num() - a.get("created_at").to_num());
            *a.borrow_mut() = list;
        }
    }

    console_log!("type", ctx.type_);
    console_log!("Object.keys(page).length", ctx.page.keys_checked()?.len());

    if ctx.type_.truthy() {
        // 2648
        {
            // 식은 왼쪽부터: page.bcc 를 page.type 보다 먼저 읽는다
            let bcc = ctx.page.g("bcc")?;
            let c = cond_for(ctx, &ctx.page.g("type")?)?;
            console_log!(tpl(T2650, &[ctx.type_.clone(), cookies.g("team")?, cookies.g("cc")?, bcc, c]));
        }

        if !ctx.detail.keys_checked()?.is_empty() {
            detail_rows(ctx, env).await?;
        } else {
            // 2767 list
            let db = tpl(T2770, &[ctx.zoneRegion.clone(), ctx.type_.clone()]);
            let stmt = prepare(env, &db, || {
                let bcc = ctx.page.g("bcc")?;
                let c = cond_for(ctx, &ctx.page.g("type")?)?;
                Ok(tpl(T2771, &[ctx.type_.clone(), cookies.g("team")?, cookies.g("cc")?, bcc, c]))
            })?;
            all_into(ctx, stmt).await?;

            console_log!("{type} results.length", ctx.results.length()?);

            if ctx.results.length()?.truthy() {
                push_rows(ctx, ctx.type_.clone())?;
            }

            console_log!("page.type", ctx.page.g("type")?);
            console_log!("type", ctx.type_);
            console_log!("zoneRegion", ctx.zoneRegion);
        }
    }
    Ok(())
}

/// 2652~2764 `if(Object.keys(detail).length){ … }` (type 이 있을 때)
async fn detail_rows(ctx: &mut Ctx, env: &Env) -> R<()> {
    let cookies = ctx.cookies.clone();

    // 2652 hints
    let items_db = tpl(T2654, &[ctx.zoneRegion.clone()]);
    let stmt = prepare(env, &items_db, || Ok(tpl(T2655, &[ctx.page.g("type")?, ctx.detail.g("digest")?, ctx.condition.clone()])))?;
    all_into(ctx, stmt).await?;

    if ctx.results.length()?.truthy() {
        ctx.hints_condition = V::from(" OR ");

        ctx.i = V::Num(0.0);
        while below_len(&ctx.i, &ctx.results)? {
            ctx.hint = ctx.results.gk(&ctx.i)?;
            let piece = tpl(T2664, &[ctx.page.g("type")?, ctx.hint.g("id")?, ctx.condition.clone()]);
            ctx.hints_condition = V::from(ctx.hints_condition.to_str() + &piece);
            ctx.i = inc(&ctx.i);
        }

        // 2667 `.replace(' OR ', '')` — 첫 번째만
        ctx.hints_condition = V::from(ctx.hints_condition.to_str().replacen(" OR ", "", 1));

        let db = tpl(T2669, &[ctx.zoneRegion.clone(), ctx.type_.clone()]);
        let stmt = prepare(env, &db, || Ok(tpl(T2670, &[ctx.type_.clone(), ctx.hints_condition.clone(), ctx.condition.clone()])))?;
        all_into(ctx, stmt).await?;

        if ctx.results.length()?.truthy() {
            // 원본 그대로: 다시 ' OR ' 로 두고 쓰지 않는다
            ctx.hints_condition = V::from(" OR ");

            ctx.i = V::Num(0.0);
            while below_len(&ctx.i, &ctx.results)? {
                ctx.hint = ctx.results.gk(&ctx.i)?;
                ctx.hint.s_("table", V::from("hints"))?;
                ctx.hints.push(ctx.hint.clone())?;
                ctx.i = inc(&ctx.i);
            }
        }
    }

    // 2685
    console_log!(tpl(T2687, &[ctx.type_.clone(), ctx.detail.g("id")?, ctx.condition.clone()]));
    let db = tpl(T2688, &[ctx.zoneRegion.clone(), ctx.type_.clone()]);
    let stmt = prepare(env, &db, || Ok(tpl(T2689, &[ctx.type_.clone(), ctx.detail.g("id")?, ctx.condition.clone()])))?;
    all_into(ctx, stmt).await?;

    console_log!("detail item results.length", ctx.results.length()?);

    if ctx.results.length()?.truthy() {
        ctx.s_item = ctx.results.gi(0)?;
        ctx.s_item.s_("table", ctx.type_.clone())?;
        ctx.items.push(ctx.s_item.clone())?;

        let detail_type = ctx.detail.g("type")?;
        if loose_eq(&detail_type, &V::from("goods")) {
            // 2700
            let db = tpl(T2702, &[ctx.zoneRegion.clone()]);
            let stmt = prepare(env, &db, || {
                let c = cond_for(ctx, &ctx.type_)?;
                Ok(tpl(T2703, &[ctx.s_item.g("index")?, cookies.g("cc")?, cookies.g("team")?, c]))
            })?;
            all_into(ctx, stmt).await?;

            if ctx.results.length()?.truthy() {
                push_rows(ctx, V::from("sales"))?;
            }
        } else if loose_eq(&detail_type, &V::from("order")) {
            if ctx.s_item.g("tracking")?.truthy() {
                // 2722
                let db = tpl(T2724, &[ctx.zoneRegion.clone()]);
                let stmt = prepare(env, &db, || {
                    let c = cond_for(ctx, &ctx.type_)?;
                    Ok(tpl(T2725, &[ctx.s_item.g("index")?, cookies.g("team")?, c]))
                })?;
                all_into(ctx, stmt).await?;

                if ctx.results.length()?.truthy() {
                    push_rows(ctx, V::from("tracking"))?;
                }
            } else {
                // 2743 — 원본 그대로: `$item.index` 가 아니라 앞에서 남은 `item`(위 반복의 마지막 행, 또는 2617 의
                // COUNT 행 { talks } — 그때는 "order" = undefined 로 D1 오류 → 418)의 index. item 이 undefined 면 TypeError
                let db = tpl(T2745, &[ctx.zoneRegion.clone()]);
                let stmt = prepare(env, &db, || {
                    let c = cond_for(ctx, &ctx.type_)?;
                    Ok(tpl(T2746, &[ctx.item.g("index")?, cookies.g("team")?, c]))
                })?;
                all_into(ctx, stmt).await?;

                if ctx.results.length()?.truthy() {
                    push_rows(ctx, V::from("tracking"))?;
                }
            }
        }
    }
    Ok(())
}

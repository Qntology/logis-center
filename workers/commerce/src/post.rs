//! 원본 1998~2454 (POST): 작업(task) 하나를 만들어 tasks 대기열(지역 DB)에 넣는다 — proxy · apac 이 그 행을 처리한다.
//!
//! 본문은 `Content-Encoding: gzip` 일 때만 읽는다(`request.arrayBuffer()` 한 번). 질의의 from · to 가 세션과 같으면 화면 HTML · 이미지
//! (2031~2114), 아니면 JSON: `{ query, vector }`(2125~2148) · URL 배열 → items 조회 응답(2150~2207) · 프롬프트 글(2208~2225).
//! 그 뒤 작업을 채우고(2245~2347) 잔액이 있으면 같은 id 의 작업이 있는지 보고(2357) 없으면 넣는다(2404).
//! 예외 경계는 원본 그대로 세 개의 try: referrer(2011) · 본문(2022) · 작업(2245, `console.error` 뒤 418 로 떨어진다).

use std::rc::Rc;

use logis_worker_common::gzip::{decode_utf8, gzip, ungzip};
use logis_worker_common::js::{self, Env};
use logis_worker_common::jsv::{index_of, js_trim, less_than, split, stringify_checked, to_lower, to_upper, tpl, Throw};
use logis_worker_common::util::hash_id;
use logis_worker_common::web::{headers_set, headers_vec, Body, Req, Resp};
use logis_worker_common::{add, arr, console_log, loose_eq, obj, parse_str, stringify_str, R, V};

use crate::ctx::Ctx;
use crate::flow::Step;
use crate::session::decode_uri_component;
use crate::templates::*;

/// `new URL(x)` — 인자는 `String(x)` (undefined 면 "undefined" 라 던진다)
fn new_url(v: &V) -> R<url::Url> {
    url::Url::parse(&v.to_str()).map_err(|_| Throw::type_error("Invalid URL string."))
}

/// `new URL(...)` 의 쓰는 속성만 담은 값 (session.rs 의 것과 같은 칸)
fn url_obj(u: &url::Url) -> V {
    let hostname = u.host_str().unwrap_or("").to_string();
    let host = match u.port() {
        Some(p) => format!("{hostname}:{p}"),
        None => hostname.clone(),
    };
    let search = match u.query() {
        Some(q) if !q.is_empty() => format!("?{q}"),
        _ => String::new(),
    };
    obj! {
        "href" => u.as_str(),
        "origin" => u.origin().ascii_serialization(),
        "protocol" => format!("{}:", u.scheme()),
        "host" => host,
        "hostname" => hostname,
        "pathname" => u.path(),
        "search" => search,
    }
}

/// `(url.pathname + url.search).toLowerCase()`
fn path_search_lower(url: &V) -> V {
    V::from(to_lower(&(url.get("pathname").to_str() + &url.get("search").to_str())))
}

/// `s.toUpperCase()` — 글이 아니면 원본처럼 TypeError
fn upper(v: &V, what: &str) -> R<String> {
    match v {
        V::Str(s) => Ok(to_upper(s)),
        V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'toUpperCase')", v.to_str()))),
        _ => Err(Throw::type_error(format!("{what}.toUpperCase is not a function"))),
    }
}

/// `s.split(sep)` — 글이 아니면 원본처럼 TypeError
fn split_v(v: &V, sep: &str, what: &str) -> R<V> {
    match v {
        V::Str(s) => Ok(V::from(split(s, sep).into_iter().map(V::from).collect::<Vec<_>>())),
        V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'split')", v.to_str()))),
        _ => Err(Throw::type_error(format!("{what}.split is not a function"))),
    }
}

/// `Object.keys(page).length` (undefined · null 이면 TypeError)
fn has_keys(v: &V) -> R<bool> {
    Ok(!v.keys_checked()?.is_empty())
}

/// JSON.stringify 앞에 Uint8Array 를 첨자 객체로 바꾼 사본. pako 의 `gzip(...)` 은 Uint8Array 를 돌려주고 JSON 에선
/// `{"0":31,"1":139,…}` 가 된다(ArrayBuffer 면 `{}`). V::Bytes 는 둘을 가르지 않고 `{}` 로 쓰이는데, 이 구역에서 문자열로 만드는
/// 값(작업 · 응답)에 든 바이트는 모두 Uint8Array 다 — `arr`(2388), 홈 페이지의 `data`(1055), 1363 의 `_page.data = arr`.
/// D1 의 BLOB 은 숫자 배열이라 그대로 둔다. 순환 참조는 JSON.stringify 처럼 TypeError
fn typed_view(v: &V, stack: &mut Vec<*const ()>) -> R<V> {
    match v {
        V::Bytes(b) => {
            let o = V::new_obj();
            for (i, x) in b.borrow().iter().enumerate() {
                o.put(&i.to_string(), V::Num(*x as f64));
            }
            Ok(o)
        }
        V::Arr(a) => {
            let p = Rc::as_ptr(a) as *const ();
            if stack.contains(&p) {
                return Err(Throw::type_error("Converting circular structure to JSON"));
            }
            stack.push(p);
            let items = a.borrow().clone();
            let mut out = Vec::with_capacity(items.len());
            for x in &items {
                out.push(typed_view(x, stack)?);
            }
            stack.pop();
            Ok(V::from(out))
        }
        V::Obj(o) => {
            let p = Rc::as_ptr(o) as *const ();
            if stack.contains(&p) {
                return Err(Throw::type_error("Converting circular structure to JSON"));
            }
            stack.push(p);
            let out = V::new_obj();
            for (k, x) in v.entries() {
                out.put(&k, typed_view(&x, stack)?);
            }
            stack.pop();
            Ok(out)
        }
        other => Ok(other.clone()),
    }
}

/// `JSON.stringify(x)` (바이트는 Uint8Array 로)
fn json(v: &V) -> R<String> {
    Ok(stringify_checked(&typed_view(v, &mut Vec::new())?)?.unwrap_or_else(|| "undefined".into()))
}

/// `new Response(JSON.stringify(body), { headers })`
fn respond(ctx: &Ctx, body: &V) -> R<Resp> {
    let mut r = Resp::new(200, Body::Text(json(body)?));
    r.headers = headers_vec(&ctx.headers);
    Ok(r)
}

/// `{ success: false, results: [], session: cookies }` — 질의를 하기 위한 정보(page)가 없음
fn no_page(ctx: &Ctx) -> R<Resp> {
    respond(ctx, &obj! { "success" => false, "results" => V::new_arr(), "session" => ctx.cookies.clone() })
}

/// 고침 (fix-app-extraction) 이 더한 `var appHref, appFormat` — 앱 봉투의 페이지 주소 · 이미지 형식 (없으면 undefined)
#[derive(Default)]
struct App {
    href: V,
    format: V,
}

/// 1998~2454
pub async fn run(ctx: &mut Ctx, env: &Env, req: &Req) -> R<Step> {
    let mut app = App::default();

    // 고침 (fix-app-extraction): 앱의 추출 요청에는 created_at 이 없다 — 없으면 지금 (원본은 아래 문에서 아무 응답 없이 418)
    if ctx.req.g("query")?.g("created_at")?.is_undef() {
        ctx.created_at = ctx.now.clone();
    }

    // 1999
    if !(ctx.cookies.g("sender")?.truthy() && ctx.created_at.truthy()) {
        // 2451 `else { }`
        return Ok(Step::Next);
    }

    // 고침 (fix-member-per-page): 이 페이지의 멤버(팀장 · 페이지 멤버 행 · 팀 공통 페이지 — resolve 1561)만 작업을 넣는다.
    // 원본은 POST 에서 member 를 보지 않았다 (화면 글 · 이미지 · { query, vector } · URL 목록 · 앱 추출 모두)
    if !ctx.member.truthy() {
        headers_set(&ctx.headers, "Content-Type", &V::from("application/json"))?;
        return Ok(Step::Return(respond(
            ctx,
            &obj! {
                "success" => false,
                "results" => V::new_arr(),
                "error" => "not a member of this page",
                "session" => ctx.cookies.clone(),
            },
        )?));
    }

    // 2000
    ctx.task = obj! {
        "cc" => ctx.cookies.g("cc")?,
        "ref" => "",
        "text" => "",
        "digest" => "",
    };
    let query = ctx.req.g("query")?;

    // 2011 — tracking 의 경우 별도 페이지 값
    let r: R<()> = (|| {
        let referrer = query.get("referrer");
        if referrer.truthy() {
            let u = new_url(&V::from(decode_uri_component(&referrer.to_str())?))?;
            ctx._url = url_obj(&u);
            let pathname = to_lower(&ctx._url.get("pathname").to_str());
            ctx.task.s_("referrer", V::from(hash_id(&add(&ctx.cookies.get("cc"), &V::from(pathname)))?))?;
        }
        Ok(())
    })();
    if let Err(err) = r {
        console_log!("task req.query.referrer err", err);
    }

    // 2021
    match body(ctx, env, req, &query, &mut app).await {
        Ok(Some(resp)) => return Ok(Step::Return(resp)),
        Ok(None) => {}
        Err(err) => console_log!("gzip or json err", err),
    }

    // 2237
    let task = ctx.task.clone();
    console_log!("task.from, task.to", task.get("from"), task.get("to"));
    console_log!("zoneRegion", ctx.zoneRegion);
    console_log!("typeof task.body", task.get("body").type_of());

    // 2245
    match enqueue(ctx, env, &query, &app).await {
        Ok(Some(resp)) => return Ok(Step::Return(resp)),
        Ok(None) => {}
        // 원본은 console.error
        Err(err) => console_log!("처리 중 오류 발생:", err),
    }
    Ok(Step::Next)
}

/// 2022~2232 (try 안 — 던지면 'gzip or json err' 로그 뒤 작업 구역으로)
async fn body(ctx: &mut Ctx, env: &Env, req: &Req, query: &V, app: &mut App) -> R<Option<Resp>> {
    if !loose_eq(&ctx.contentEncoding, &V::from("gzip")) {
        return Ok(None);
    }
    // 2023 `await request.arrayBuffer()`
    ctx.buffer = V::bytes(req.body.clone());

    console_log!("buffer.byteLength", ctx.buffer.g("byteLength")?);
    console_log!("req.query", stringify_str(&ctx.req.g("query")?));
    console_log!("contentType", ctx.contentType);

    if !ctx.buffer.g("byteLength")?.truthy() {
        return Ok(None);
    }

    let cookies = ctx.cookies.clone();
    let task = ctx.task.clone();

    // 고침 (fix-app-cloud-search · fix-app-extraction): 앱은 from = 주소 · to = 팀 으로 JSON 본문을 보낸다 — { query, vector } 는 검색 갈래(아래)로,
    // { type: '*_extraction', body, link, … } 봉투는 풀어서 이 갈래로 (원본은 둘 다 날 글로 받아 벡터를 버리고 봉투 JSON 을 작업 본문으로 넣었다)
    let content_type = if ctx.contentType.truthy() { ctx.contentType.to_str() } else { String::new() };
    let app_body = if index_of(&content_type, "application/json") > -1 {
        crate::fixes::app_json(&decode_utf8(&ungzip(&ctx.buffer)?))
    } else {
        None
    };
    let envelope = app_body.clone().filter(|b| {
        b.get("body").is_str() && ["image_extraction", "html_extraction", "document_extraction"].iter().any(|t| b.get("type").as_str() == Some(t))
    });
    let search = app_body.filter(|b| envelope.is_none() && (b.get("query").truthy() || b.get("vector").truthy()));

    // 2031 — 화면 HTML · 이미지 (원본 그대로: 둘 다 undefined 여도 `==` 라 같다고 본다)
    if search.is_none() && loose_eq(&query.get("from"), &cookies.get("address")) && loose_eq(&query.get("to"), &cookies.get("team")) {
        task.s_("from", cookies.get("address"))?;
        task.s_("to", cookies.get("team"))?;

        ctx.body = V::from(decode_utf8(&ungzip(&ctx.buffer)?));

        if let Some(envelope) = &envelope {
            if envelope.get("type").as_str() == Some("document_extraction") {
                // proxy 에는 문서 해석기가 없다 — 아무도 처리하지 못할 작업을 넣지 않는다
                headers_set(&ctx.headers, "Content-Type", &V::from("application/json"))?;
                return Ok(Some(respond(
                    ctx,
                    &obj! {
                        "success" => false,
                        "results" => V::new_arr(),
                        "error" => "document_extraction is not supported by the server",
                        "session" => cookies.clone(),
                    },
                )?));
            }

            // 봉투의 link 가 http(s) 주소면 그 페이지의 작업 (아니면 — 이미지의 "local" — 질의의 href)
            if let Ok(u) = url::Url::parse(&envelope.get("link").to_str()) {
                if u.scheme() == "http" || u.scheme() == "https" {
                    app.href = V::from(to_lower(u.as_str()));
                }
            }

            let page_href = if app.href.truthy() { app.href.clone() } else { cookies.get("href") };
            task.s_("cc", V::from(crate::fixes::cc_of_host(&crate::fixes::url_host(&new_url(&page_href)?))?))?;

            let text = envelope.get("body").to_str();
            if envelope.get("type").as_str() == Some("image_extraction") {
                // 질의 format 은 두 번 인코딩되어 온다 (image%2Fpng) — 풀어서 image/… 면 그것, 아니면 image/png. 본문은 data URL 의 base64 만
                let format = query.get("format");
                let mime = decode_uri_component(&if format.truthy() { format.to_str() } else { String::new() })?;
                app.format = V::from(if index_of(&mime, "image/") > -1 { mime } else { "image/png".to_string() });

                let base64 = match (text.starts_with("data:"), text.find(',')) {
                    (true, Some(comma)) => text[comma + 1..].to_string(),
                    _ => text,
                };
                task.s_("body", V::from(base64))?;
            } else {
                task.s_("body", V::from(text))?;
            }

            task.s_("scan", V::Bool(true))?;
        } else if query.get("format").truthy() {
            task.s_("body", ctx.body.clone())?;
            task.s_("scan", V::Bool(true))?;
        } else {
            // 원본 그대로: 여기서 URIError 면 from · to 만 채운 채 끝난다(body 가 없어 작업 구역은 건너뜀)
            task.s_("body", V::from(decode_uri_component(&ctx.body.to_str())?))?;
        }

        let content_type = if envelope.is_some() {
            if app.format.truthy() {
                app.format.clone()
            } else {
                V::from("text/html")
            }
        } else {
            ctx.contentType.clone()
        };
        task.s_("contentType", content_type)?;

        // 2052~2112 linkedom 선택자 추출은 주석 처리된 코드
        return Ok(None);
    }

    // 2116
    ctx.decompressedJsonString = V::from(decode_utf8(&ungzip(&ctx.buffer)?));
    ctx.data = parse_str(&ctx.decompressedJsonString.to_str())?;
    let data = ctx.data.clone();

    // 2125 [CLIENT-SIDE EMBEDDING] { query, vector }
    if data.truthy() && !data.is_arr() && data.type_of() == "object" && (data.get("query").truthy() || data.get("vector").truthy()) {
        // 고침 (fix-app-cloud-search): 앱의 검색은 logis.center 의 홈(/tracking)에서 온다. homeId 는 대문자 cc 로 만들어 pageId 와 늘 달라
        // page 가 비었다 — logis.center 의 요청이면 home 을 쓴다
        if !has_keys(&ctx.page)? && loose_eq(&cookies.get("cc"), &ctx.homeCc) {
            ctx.page = ctx.home.clone();
        }

        if has_keys(&ctx.page)? {
            task.s_("page", ctx.page.clone())?;
            task.s_("from", cookies.get("team"))?;
            task.s_("to", cookies.get("address"))?;
            let q = data.get("query");
            task.s_("body", if q.truthy() { V::from(js_trim(&q.to_str()).to_string()) } else { V::from("") })?;
            task.s_("scan", V::Null)?;
            // 고침 (fix-app-cloud-search): proxy 는 task.contentType.indexOf 로 갈래를 고른다 — 없으면 TypeError 로 작업을 버렸다
            task.s_("contentType", ctx.contentType.clone())?;

            let vector = data.get("vector");
            if vector.is_arr() && vector.g("length")?.truthy() {
                task.s_("queryVector", vector)?;
            }
        } else {
            return Ok(Some(no_page(ctx)?));
        }
    }

    // 2150 — 원본 그대로: else 가 아니라 위 갈래 뒤에도 본다(null 이면 여기서 TypeError, `length` 키가 있는 객체도 들어온다)
    if data.g("length")?.truthy() {
        if data.type_of() == "object" {
            return Ok(Some(lookup(ctx, env, &cookies).await?));
        } else if data.type_of() == "string" {
            if has_keys(&ctx.page)? {
                task.s_("page", ctx.page.clone())?;
                task.s_("from", cookies.get("team"))?;
                task.s_("to", cookies.get("address"))?;
                task.s_("body", V::from(js_trim(&data.to_str()).to_string()))?;
                task.s_("scan", V::Null)?;
                // 고침 (fix-app-cloud-search): 위와 같음
                task.s_("contentType", ctx.contentType.clone())?;
            } else {
                return Ok(Some(no_page(ctx)?));
            }
        }
    }
    Ok(None)
}

/// 2152~2206 — URL 목록의 items 행을 돌려준다
async fn lookup(ctx: &mut Ctx, env: &Env, cookies: &V) -> R<Resp> {
    ctx.list = ctx.data.clone();
    ctx.condition = V::from("");

    let u = new_url(&cookies.get("href"))?;
    ctx.url = url_obj(&u);
    ctx.link = path_search_lower(&ctx.url);
    ctx.ref_ = V::from(hash_id(&add(&add(&cookies.get("team"), &cookies.get("cc")), &ctx.link))?);

    // 2162
    ctx.i = V::Num(0.0);
    while less_than(&ctx.i, &ctx.list.g("length")?) == Some(true) {
        let r: R<()> = (|| {
            let u = new_url(&ctx.list.gk(&ctx.i)?)?;
            ctx.url = url_obj(&u);
            ctx.link = path_search_lower(&ctx.url);

            // 원본 그대로: ' OR ' 를 먼저 붙이므로 뒤의 식이 던지면 짝 없는 OR 가 남는다
            if ctx.condition.truthy() {
                ctx.condition = add(&ctx.condition, &V::from(" OR "));
            }

            // 고침 (fix-sql-bind): 원본은 팀을 큰따옴표(SQLite 는 먼저 열 이름으로 본다)로 이어 붙였다 → ?1 로 묶는다
            let bcc = hash_id(&add(&ctx.page.g("type")?, &V::from(upper(&cookies.g("cc")?, "cookies.cc")?)))?;
            let ref_ = hash_id(&add(&add(&cookies.get("team"), &cookies.get("cc")), &ctx.link))?;
            let clause = format!(
                r#"("to" = ?1 AND "bcc" = '{bcc}' AND "ref" = '{ref_}' AND "created_at" < {})"#,
                ctx.created_at.to_str()
            );
            ctx.condition = add(&ctx.condition, &V::from(clause));
            Ok(())
        })();
        // catch(err){} — 삼킨다
        let _ = r;
        ctx.i = V::Num(ctx.i.to_num() + 1.0);
    }

    ctx.success = V::Bool(false);
    ctx.results = V::new_arr();

    console_log!("condition", ctx.condition);

    if ctx.condition.truthy() {
        let db = tpl(T2186, &[ctx.zoneRegion.clone()]);
        let sql = tpl(T2187, &[ctx.condition.clone()]);
        let out = js::all(&js::bind(&env.prepare(&db, &sql)?, &[cookies.get("team")])?).await?;
        ctx.results = out.g("results")?;
        ctx.success = out.g("success")?;
        ctx.error = out.g("error")?;
    }

    headers_set(&ctx.headers, "Content-Type", &V::from("application/json"))?;

    respond(ctx, &obj! { "success" => ctx.success.clone(), "results" => ctx.results.clone(), "session" => cookies.clone() })
}

/// 2245~2448 (try 안 — 던지면 '처리 중 오류 발생:' 로그 뒤 418)
async fn enqueue(ctx: &mut Ctx, env: &Env, query: &V, app: &App) -> R<Option<Resp>> {
    let task = ctx.task.clone();
    if !(task.get("from").truthy() && task.get("to").truthy() && task.get("body").truthy()) {
        return Ok(None);
    }
    let cookies = ctx.cookies.clone();

    task.s_("no", V::from(""))?;

    // 2250 — 고침 (fix-app-extraction): 앱 봉투의 페이지 주소가 있으면 그것
    let page_href = if app.href.truthy() { app.href.clone() } else { cookies.get("href") };
    let u = new_url(&page_href)?;
    ctx.url = url_obj(&u);

    task.s_("origin", ctx.url.get("origin"))?;
    task.s_("link", path_search_lower(&ctx.url))?;
    task.s_("ref", V::from(hash_id(&add(&add(&cookies.get("team"), &task.get("cc")), &ctx.url.get("pathname")))?))?;
    task.s_("href", page_href)?;
    task.s_("type", V::from("prompt"))?;

    let title = query.get("title");
    task.s_("title", if title.truthy() { V::from(decode_uri_component(&title.to_str())?) } else { V::from("") })?;

    task.s_("topK", V::Num(50.0))?;
    task.s_("rpm", V::Num(3.0))?;
    task.s_("from", cookies.get("address"))?;
    task.s_("to", cookies.get("team"))?;
    task.s_("bcc", V::from(""))?;
    // 원본 그대로: 없던 키면 undefined 로 이 자리에 생긴다(JSON 에서는 빠지고, tracking 이면 이 자리에 값이 들어간다)
    task.s_("contentType", task.get("contentType"))?;

    // 2279 — 고침 (fix-app-extraction): 앱 이미지의 형식(appFormat)이 있으면 그것
    console_log!("req.query.format", query.get("format"));
    let format = if app.format.truthy() { app.format.clone() } else { query.get("format") };

    if format.truthy() {
        let Some(f) = format.as_str() else {
            return Err(Throw::type_error("taskFormat.indexOf is not a function"));
        };
        if index_of(f, "image/") > -1 {
            task.s_("type", V::from("tracking"))?;
            task.s_("bcc", V::from(hash_id(&add(&task.get("type"), &V::from(upper(&task.get("cc"), "task.cc")?)))?))?;
            task.s_("contentType", V::from(decode_uri_component(f)?))?;
        }
    }

    console_log!("task.type", task.get("type"));

    // 2293
    task.s_("flag", ctx.FLAG.clone())?;
    task.s_("score", V::Num(0.0))?;
    task.s_("amount", V::Num(0.0))?;
    task.s_("quantity", V::Num(0.0))?;
    task.s_("currency", V::from(""))?;
    task.s_("semantic", V::from(""))?;
    task.s_("zone", ctx.zoneRegion.clone())?;

    // 2315
    task.s_("id", V::from(hash_id(&add(&add(&cookies.get("team"), &task.get("cc")), &task.get("body")))?))?;

    // 2317
    ctx.current_date = split_v(&ctx.current, "T", "current")?;
    ctx.current_date = split_v(&ctx.current_date.gi(0)?, "-", "current_date[0]")?;
    ctx.year = ctx.current_date.gi(0)?;
    ctx.month = ctx.current_date.gi(1)?;
    ctx.s_to = V::from(hash_id(&add(&add(&ctx.year, &V::from("-")), &ctx.month))?);

    // 2326 — 고침 (fix-sql-bind)
    let db = tpl(T2326, &[ctx.zoneRegion.clone()]);
    let sql = format!(
        r#"SELECT * FROM items WHERE "type" = 'invoice' AND "from" = ?1 AND "to" = ?2 AND "created_at" < {} LIMIT 1"#,
        ctx.now.to_str()
    );
    ctx.results = js::query_bound(env, &db, &sql, &[cookies.get("address"), ctx.s_to.clone()]).await?;

    // 보강해야함 - 사용자 결제 혹은 팀결제 확인해서 진행하기
    if ctx.results.g("length")?.truthy() {
        // 유료 사용자
        task.s_("topK", V::Num(100.0))?;
        task.s_("rpm", V::Num(50.0))?;
    }

    task.s_("created_at", ctx.now.clone())?;
    task.s_("updated_at", V::Num(0.0))?;
    task.s_("detail", V::Bool(false))?;

    // 2340
    if has_keys(&ctx.page)? {
        // 원본 그대로: page.data 는 대개 풀지 않은 BLOB(숫자 배열 · Uint8Array)이라 detail 은 undefined
        task.s_("detail", ctx.page.g("data")?.g("detail")?)?;

        if task.get("detail").truthy() {
            let team = cookies.get("team");
            let cc = upper(&task.get("cc"), "task.cc")?;
            task.s_("ref", V::from(hash_id(&add(&add(&team, &V::from(cc)), &ctx.url.get("pathname")))?))?;
        }
    }

    // 2350 한도 초과인지 체크하기
    console_log!("cookies.balance", cookies.get("balance"));

    if !cookies.get("balance").truthy() {
        return Ok(None);
    }

    headers_set(&ctx.headers, "Content-Type", &V::from("application/json"))?;

    // 2357
    let region = ctx.logisRegion.to_key();
    // task.id 는 서버가 만든 해시라 글에 둔다
    let sql = tpl(T2357, &[task.get("id"), ctx.now.clone()]);
    let out = js::all(&env.prepare(&region, &sql)?).await?;
    ctx.results = out.g("results")?;
    ctx.success = out.g("success")?;

    // 고침 (fix-task-column): ?error= 는 다시 넣으라는 뜻 — 같은 id 의 작업이 있어도 아래 ON CONFLICT DO UPDATE 로 간다
    if ctx.results.g("length")?.truthy() && !query.get("error").truthy() {
        if loose_eq(&ctx.contentEncoding, &V::from("gzip")) {
            if has_keys(&ctx.page)? {
                let page = ctx.page.clone();
                page.s_("from", cookies.get("address"))?;
                page.s_("to", cookies.get("team"))?;
                page.s_("bcc", cookies.get("bcc"))?;
                page.s_("title", V::from(""))?;

                ctx.results = arr![page];
            }
        }
        // else: text 2 sql 진행 (원본도 비어 있음)

        return Ok(Some(respond(
            ctx,
            &obj! { "results" => ctx.results.clone(), "success" => ctx.success.clone(), "session" => cookies.clone() },
        )?));
    }

    // 2388 `gzip(new TextEncoder('utf-8').encode(JSON.stringify(task)), { to: 'arraybuffer' })` — Uint8Array (bind 는 arr.buffer)
    ctx.arr = V::bytes(gzip(json(&task)?.as_bytes()));

    ctx.onConflict = V::from("ON CONFLICT (id) DO NOTHING");

    if query.get("error").truthy() {
        // 고침 (fix-task-column): 원본은 tasks 에 없는 "task" 열을 써서 이 INSERT 가 늘 D1 오류(418)였다 → 진짜 열
        ctx.onConflict = V::from(ON_CONFLICT_RETRY);
    }

    // 2404 tasks 추가 작업
    let sql = tpl(T2404, &[ctx.onConflict.clone()]);
    let stmt = env.prepare(&region, &sql)?;
    let stmt = js::bind(
        &stmt,
        &[
            task.get("id"),
            task.get("type"),
            task.get("from"),
            task.get("to"),
            task.get("cc"),
            task.get("bcc"),
            task.get("ref"),
            ctx.arr.clone(),
            task.get("created_at"),
            task.get("updated_at"),
        ],
    )?;
    let out = js::run(&stmt).await?;
    ctx.results = out.g("results")?;
    ctx.success = out.g("success")?;
    ctx.error = out.g("error")?;

    // 2430 — 원본 그대로: ref 칸에 task.href, data 는 Uint8Array 라 JSON 에서 첨자 객체
    let row = obj! {
        "id" => task.get("id"),
        "table" => "tasks",
        "type" => task.get("type"),
        "from" => task.get("from"),
        "to" => task.get("to"),
        "cc" => task.get("cc"),
        "bcc" => task.get("bcc"),
        "ref" => task.get("href"),
        "data" => ctx.arr.clone(),
        "created_at" => task.get("created_at"),
        "updated_at" => task.get("updated_at"),
    };

    Ok(Some(respond(ctx, &obj! { "results" => arr![row], "success" => ctx.success.clone(), "session" => cookies.clone() })?))
}

/// 2393 의 ON CONFLICT — 고침 (fix-task-column): "task" 열 대신 tasks 의 진짜 열
const ON_CONFLICT_RETRY: &str = "ON CONFLICT (id) DO UPDATE SET
											\"cc\" = EXCLUDED.\"cc\",
											\"bcc\" = EXCLUDED.\"bcc\",
											\"ref\" = EXCLUDED.\"ref\",
											\"type\" = EXCLUDED.\"type\",
											\"from\" = EXCLUDED.\"from\",
											\"to\" = EXCLUDED.\"to\",
											\"data\" = EXCLUDED.\"data\",
											\"created_at\" = EXCLUDED.\"created_at\",
											\"updated_at\" = EXCLUDED.\"updated_at\"
										";

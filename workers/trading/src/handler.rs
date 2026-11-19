//! 원본 456~1079: fetch 처리기. 바깥 try(478~1074) 안에서 던진 것은 `fatal err` 로 남기고 418, 그 앞(AwsClient 생성)에서
//! 던진 것은 처리기 밖으로 나간다(`Err` → 진입점이 JS 예외로 던짐). 고친 것은 `// 고침 (fix-trading-…)` 과
//! docs/workers-rs.md 의 "고친 버그" 표, 원본에 덧대는 같은 수정은 scripts/workers-diff/fixes/trading.mjs.

use std::sync::OnceLock;

use logis_worker_common::aws::{self, AwsClient};
use logis_worker_common::gzip::decode_utf8;
use logis_worker_common::js::{self, Env, Js};
use logis_worker_common::jsv::{add, js_trim, loose_eq, parse_str, split, stringify_checked, to_lower, tpl, Throw, R, V};
use logis_worker_common::web::{headers_append, headers_new, headers_set, headers_vec, Body, Req, Resp};
use logis_worker_common::{console_log, eth, obj, platform};
use regex::Regex;
use wasm_bindgen::JsValue;

use crate::ctx::Ctx;
use crate::jsrt::{self, oget, oset};
use crate::logic::*;
use crate::team::resolve_team;
use crate::templates::*;

// ── 고친 SQL (templates.rs 는 원본에서 생성한 것이라 손대지 않는다 — 같은 글이 fixes/trading.mjs 에) ──

/// 643 (fix-trading-delta-cursor): created_at 상한 · max(updated_at, created_at) 대신 서버가 찍은 synced_at
const FIXED_WHERE: &str = "\"to\" = ?1 AND \"mode\" = ?2 AND \"synced_at\" > ?3";
/// 668 (fix-trading-delta-cursor): `SELECT * FROM items WHERE ${where} ORDER BY "synced_at" ASC, "id" ASC LIMIT ${limit}`
const FIXED_T668: &[&str] = &["SELECT * FROM items WHERE ", " ORDER BY \"synced_at\" ASC, \"id\" ASC LIMIT ", ""];
/// (fix-trading-delta-cursor) 꽉 찬 쪽의 마지막 시각에 남은 행: `… WHERE ${where} AND "synced_at" = ?${slot} AND "id" > ?${slot + 1} ORDER BY "id" ASC`
const FIXED_TAIL: &[&str] = &["SELECT * FROM items WHERE ", " AND \"synced_at\" = ?", " AND \"id\" > ?", " ORDER BY \"id\" ASC"];
/// 811 (fix-trading-upsert-scope · fix-trading-preload-chunk): 팀 조건 없이, "to" 를 같이
const FIXED_T811: &[&str] = &["SELECT \"id\", \"to\", \"digest\", \"ref\", \"updated_at\" FROM items WHERE \"id\" IN (", ")"];
/// 904 (fix-trading-upsert-scope · fix-trading-delta-cursor): 다른 팀 행은 덮어쓰지 않고, synced_at 을 같이 쓴다
const FIXED_UPSERT: &str = "INSERT INTO items (\"id\", \"type\", \"flag\", \"from\", \"to\", \"cc\", \"bcc\", \"ref\", \"mode\", \"digest\", \"data\", \"created_at\", \"updated_at\", \"synced_at\") \
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14) ON CONFLICT (id) DO UPDATE SET \
\"type\" = EXCLUDED.\"type\", \"flag\" = EXCLUDED.\"flag\", \"from\" = EXCLUDED.\"from\", \"to\" = EXCLUDED.\"to\", \"cc\" = EXCLUDED.\"cc\", \
\"bcc\" = EXCLUDED.\"bcc\", \"ref\" = EXCLUDED.\"ref\", \"mode\" = EXCLUDED.\"mode\", \"digest\" = EXCLUDED.\"digest\", \"data\" = EXCLUDED.\"data\", \
\"created_at\" = EXCLUDED.\"created_at\", \"updated_at\" = EXCLUDED.\"updated_at\", \"synced_at\" = EXCLUDED.\"synced_at\" \
WHERE items.\"to\" = EXCLUDED.\"to\"";
/// (fix-trading-upsert-scope) ${id}
const T_OTHER_TEAM: &[&str] = &["[TRADING] ⛔ id='", "' 은 다른 팀의 문서라 거부합니다."];

/// `new Response(string, { headers })` (Content-Type 이 없으면 `resp_into` 가 `text/plain;charset=UTF-8` 을 붙인다 — JS 와 같음)
fn respond(body: String, headers: &V) -> Resp {
    Resp { status: 200, headers: headers_vec(headers), body: Body::Text(body) }
}

/// `JSON.stringify(x)` (객체만 넘긴다)
fn json(v: &V) -> R<String> {
    Ok(stringify_checked(v)?.unwrap_or_else(|| "undefined".into()))
}

/// `request.cf.country` — cf 가 평범한 객체가 아니면(V::Ext) JS 로 읽는다
fn cf_get(cf: &V, key: &str) -> V {
    match cf.ext_ref::<Js>() {
        Some(j) => js::to_v(&js_sys::Reflect::get(&j.0, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)),
        None => cf.get(key),
    }
}

/// `Math.max(0, now - 1000)`
fn cursor(now: f64) -> V {
    V::Num(jsrt::math_max(0.0, now - 1000.0))
}

/// `String(x).trim().toLowerCase()`
fn trim_lower(v: &V) -> String {
    to_lower(js_trim(&v.to_str()))
}

pub async fn fetch(request: Req, env: Env) -> R<Resp> {
    let mut ctx = Ctx::default();
    // 458~463: new AwsClient(...) — aws4fetch 생성자가 던지는 것도 그대로 (try 밖)
    let s3 = AwsClient::try_new(env.get("aws_access_key_id"), env.get("aws_secret_access_key"), V::from("s3"), env.get("aws_region"))?;
    // 464: var { country } = request.cf
    // 고침 (fix-trading-cf-optional): 원본은 cf 가 없으면(로컬 개발 · 일부 내부 요청) try 밖에서 TypeError 로 처리기가
    // 통째로 던졌다 → `request.cf ? request.cf : {}` (country 가 없으면 아래에서 'us')
    ctx.country = if request.cf.is_nullish() { V::Undef } else { cf_get(&request.cf, "country") };
    ctx.ip = request.header("cf-connecting-ip");
    if !ctx.ip.truthy() {
        ctx.ip = request.header("X-Real-IP");
    }
    if ctx.ip.truthy() {
        // JS 의 \b · \d 는 ASCII
        static IP: OnceLock<Regex> = OnceLock::new();
        let re = IP.get_or_init(|| Regex::new(r"(?-u:\b)[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}(?-u:\b)").unwrap());
        let ip = ctx.ip.to_str();
        ctx.m = re.find(&ip).map_or(V::Null, |m| V::from(m.as_str()));
        if ctx.m.truthy() {
            ctx.ip = ctx.m.clone();
        }
    }
    ctx.FLAG = V::from(to_lower(&if ctx.country.truthy() { ctx.country.to_str() } else { "us".into() }));
    let lang = language_code(&ctx.FLAG);
    ctx.language = if lang.truthy() { lang } else { V::from("English") };

    match body(&mut ctx, &request, &env, &s3).await {
        Ok(resp) => return Ok(resp),
        Err(err) => console_log!("[TRADING] fatal err", err),
    }
    let mut r = Resp::new(418, Body::Text(T1075[0].into()));
    r.set("Content-Type", "text/html; charset=utf-8");
    Ok(r)
}

/// 478~1071 (바깥 try 의 몸통 — 모든 갈래가 응답을 돌려준다)
async fn body(ctx: &mut Ctx, request: &Req, env: &Env, s3: &AwsClient) -> R<Resp> {
    ctx.headers = headers_new();
    ctx.cookies = V::new_obj();
    ctx.cookiesStr = request.header("Cookie");
    ctx.contentType = request.header("Content-Type");
    ctx.contentEncoding = request.header("Content-Encoding");
    let cookies = ctx.cookies.clone();
    let headers = ctx.headers.clone();
    if ctx.cookiesStr.truthy() {
        for cookie in split(&ctx.cookiesStr.to_str(), ";") {
            let parts = split(&cookie, "=");
            // 고침 (fix-trading-cookie-identity): 원본은 Cookie 머리글의 모든 이름을 cookies 에 담아 address · team 쿠키만으로
            // 인증된 것처럼(S3 세션이 실패해도) 남의 팀을 읽고 쓰고 지울 수 있었다 → 자격증명인 hash · token 만 받는다
            if parts.len() == 2 && matches!(js_trim(&parts[0]), "hash" | "token") {
                oset(&cookies, js_trim(&parts[0]), V::from(js_trim(&parts[1])))?;
            }
        }
    }
    let request_url = url::Url::parse(&request.url).map_err(|_| Throw::type_error("Invalid URL string."))?;
    ctx.req = obj! { "method" => request.method.as_str(), "query" => obj! {} };
    let query = ctx.req.get("query");
    for (k, v) in request_url.query_pairs() {
        oset(&query, &k, V::from(v.into_owned()))?;
    }
    let q = |name: &str| oget(&query, name);

    let greeting = hello(&ctx.language);
    cookies.s_("hello", if greeting.truthy() { greeting } else { hello(&V::from("English")) })?;
    cookies.s_("flag", ctx.FLAG.clone())?;
    cookies.s_("mode", V::from(TRADING_MODE))?;
    // 🌟 [NO HREF] cc 는 고정 합성 도메인 하나뿐
    cookies.s_("cc", V::from(hash_id(&V::from(TRADING_HOST))?))?;
    cookies.s_("host", V::from(TRADING_HOST))?;

    ctx.now = V::Num(platform::now());
    let now = ctx.now.to_num();
    ctx.created_at = V::Num(0.0);
    if !jsrt::is_nan(&q("created_at")) {
        ctx.created_at = V::Num(jsrt::parse_int(&q("created_at")));
    }
    if !ctx.created_at.truthy() {
        ctx.created_at = V::Num(now + 60000.0);
    }

    // 고침 (fix-trading-options-first): 원본은 CORS 머리글과 OPTIONS 응답(605~613)이 세션 뒤에 있어, 쿠키 없는 preflight 마다
    // bootstrap 이 새 (hash, token) 을 S3 에 PUT 했다 → 세션보다 먼저 답한다
    ctx.requestOrigin = request.header("Origin");
    if !ctx.requestOrigin.truthy() {
        // Origin 없음 (앱의 reqwest 등 브라우저가 아닌 소비자) — 원본 그대로
        headers_set(&headers, "Access-Control-Allow-Origin", &V::from("*"))?;
        headers_set(&headers, "Access-Control-Allow-Credentials", &V::from("true"))?;
    } else if ALLOWED_ORIGINS.contains(&ctx.requestOrigin.to_str().as_str()) {
        // 고침 (fix-trading-cors): 원본은 아무 Origin 이나 되돌려 주며 Allow-Credentials 를 붙여, 어느 사이트든 방문자의
        // 쿠키로 응답(세션 · 서류)을 읽을 수 있었다 → 알려진 logis.center 출처만
        headers_set(&headers, "Access-Control-Allow-Origin", &ctx.requestOrigin)?;
        headers_set(&headers, "Access-Control-Allow-Credentials", &V::from("true"))?;
    }
    headers_set(&headers, "Access-Control-Allow-Methods", &V::from("GET, POST, DELETE, OPTIONS"))?;
    headers_set(&headers, "Access-Control-Allow-Headers", &V::from("Content-Type, Content-Encoding, Authorization, X-Requested-With"))?;

    if request.method == "OPTIONS" {
        return Ok(respond(String::new(), &headers));
    }

    // ── 인증 : S3 hash/token 계약 (517~567) ──
    if oget(&cookies, "hash").truthy() || (q("hash").truthy() && q("token").truthy()) {
        let r: R<()> = async {
            if oget(&cookies, "hash").truthy() && q("hash").truthy() && !loose_eq(&oget(&cookies, "hash"), &q("hash")) {
                cookies.s_("signature", V::from(""))?;
            }
            if q("hash").truthy() {
                cookies.s_("hash", q("hash"))?;
            }
            if q("token").truthy() {
                cookies.s_("token", q("token"))?;
            }
            let url = tpl(T528, &[env.get("aws_bucket"), env.get("aws_region"), oget(&cookies, "hash")]);
            let response = aws::fetch(s3, &url, "HEAD", &[], &[]).await?;
            ctx.response = V::ext(Js(response.clone()));
            let status = js::response_status(&response);
            if status == 200.0 {
                ctx.balance = V::Num(0.0);
            }
            ctx.email = js::response_header(&response, "email")?;
            ctx.address = js::response_header(&response, "address")?;
            ctx.token = js::response_header(&response, "token")?;
            if (200.0..=299.0).contains(&status) {
                ctx.headObject = V::new_obj();
                for (key, value) in js::response_headers(&response)? {
                    if key.starts_with("x-amz-meta-") {
                        oset(&ctx.headObject, &jsrt::replace_first(&key, "x-amz-meta-", ""), V::from(value))?;
                    }
                }
                if oget(&ctx.headObject, "email").truthy() {
                    ctx.email = oget(&ctx.headObject, "email");
                }
                if oget(&ctx.headObject, "address").truthy() {
                    ctx.address = oget(&ctx.headObject, "address");
                }
                if oget(&ctx.headObject, "token").truthy() {
                    ctx.token = oget(&ctx.headObject, "token");
                }
            }
            // 고침 (fix-trading-token-required): 원본은 S3 의 token 메타도 요청의 token 도 없으면 null == undefined 라 hash 만으로
            // 통과했고, commerce 에 token=undefined 로 팀을 물었다 → 둘 다 있고 같아야 한다
            let presented = oget(&cookies, "token");
            if ctx.email.truthy() && ctx.token.truthy() && presented.truthy() && loose_eq(&ctx.token, &presented) {
                cookies.s_("email", ctx.email.clone())?;
                cookies.s_("address", ctx.address.clone())?;
                // 🌟 팀 판정은 commerce Worker 에 위임
                ctx.resolved = resolve_team(env, &oget(&cookies, "hash"), &oget(&cookies, "token"), &ctx.address, &q("to")).await?;
                cookies.s_("team", ctx.resolved.g("team")?)?;
                cookies.s_("role", ctx.resolved.g("role")?)?;
                if loose_eq(&ctx.resolved.g("role")?, &V::from("denied")) {
                    console_log!(tpl(T556, &[ctx.address.clone()]));
                }
                // 고침 (fix-trading-setcookie-verified): 원본은 확인하지 않은 hash · token(쿼리 글 그대로)을 Set-Cookie 로 되돌려
                // `?hash=x;%20Domain=logis.center` 같은 값으로 쿠키 속성을 끼워 넣을 수 있었다 → 확인된 세션만 쿠키로
                headers_set(&headers, "Set-Cookie", &V::from(tpl(T559, &[oget(&cookies, "hash")])))?;
                headers_append(&headers, "Set-Cookie", &V::from(tpl(T560, &[oget(&cookies, "token")])))?;
                if oget(&cookies, "team").truthy() {
                    headers_append(&headers, "Set-Cookie", &V::from(tpl(T562, &[oget(&cookies, "team")])))?;
                }
            }
            Ok(())
        }
        .await;
        if let Err(err) = r {
            console_log!("[TRADING] session err", err);
        }
    }

    // ── bootstrap : 자격증명이 없으면 서버가 (hash, token) 쌍을 발급 (569~603) ──
    if ctx.balance.is_undef() {
        let (address, private_key) = eth::random_wallet();
        ctx.account = obj! { "address" => address.as_str(), "privateKey" => private_key.as_str() };
        ctx.hash = V::from(jsrt::replace_first(&to_lower(&address), "0x", ""));
        ctx.token = V::from(jsrt::replace_first(&to_lower(&private_key), "0x", ""));
        let r: R<()> = async {
            let url = tpl(T576, &[env.get("aws_bucket"), env.get("aws_region"), ctx.hash.clone()]);
            let meta: [(&str, V); 7] = [
                ("balance", V::from("1000")),
                ("host", V::from(TRADING_HOST)),
                ("token", ctx.token.clone()),
                ("email", V::from("")),
                ("ip", ctx.ip.clone()),
                ("phone", V::from("")),
                ("address", V::from("")),
            ];
            let mut put_headers: Vec<(String, String)> = vec![("Content-Type".into(), "text/plain".into())];
            for (k, v) in meta {
                // Headers 는 값을 String(v) 로 (ip 가 null 이면 "null")
                put_headers.push((tpl(T589, &[V::from(k)]), v.to_str()));
            }
            let signed = s3.sign(&url, "PUT", &put_headers, b"", platform::now())?;
            aws::send(&signed).await?;
            cookies.s_("hash", ctx.hash.clone())?;
            cookies.s_("token", ctx.token.clone())?;
            headers_set(&headers, "Set-Cookie", &V::from(tpl(T598, &[ctx.hash.clone()])))?;
            headers_append(&headers, "Set-Cookie", &V::from(tpl(T599, &[ctx.token.clone()])))?;
            Ok(())
        }
        .await;
        if let Err(err) = r {
            console_log!("[TRADING] bootstrap err", err);
        }
    }
    // 고침 (fix-trading-token-echo): 원본은 모든 응답의 session(과 GET 의 HTML)에 token 을 실어, HttpOnly 쿠키로 감춘
    // 비밀을 본문으로 되돌려 주었다 (CORS 반사와 겹치면 어느 사이트든 읽음). 쓰는 소비자가 없다 → 본문에서 뺀다
    cookies.del("token")?;

    let unauthenticated = !oget(&cookies, "address").truthy() || !oget(&cookies, "team").truthy();

    // ── GET : 서버 → 클라이언트 동기화 (618~736) ──
    if request.method == "GET" {
        if !loose_eq(&ctx.contentType, &V::from("application/json")) {
            headers_set(&headers, "Content-Type", &V::from("text/html; charset=utf-8"))?;
            // 고침 (fix-trading-html-escape): 원본은 JSON 을 그대로 <pre> 에 넣어 `?hash=<script>…` 가 그대로 실행됐다(반사 XSS)
            let pretty = jsrt::stringify_pretty(&cookies)?.map_or(V::Undef, |s| V::from(escape_html(&s)));
            return Ok(respond(T622[0].to_string() + &tpl(T623, &[pretty]), &headers));
        }
        headers_set(&headers, "Content-Type", &V::from("application/json"))?;
        // address · team 은 확인된 S3 세션에서만 온다 (fix-trading-cookie-identity)
        if unauthenticated {
            console_log!("[TRADING] ⚪ 미인증 요청. 빈 배열을 반환합니다.");
            return Ok(respond(json(&obj! { "results" => V::new_arr(), "session" => cookies.clone() })?, &headers));
        }
        // 🌟 [DELTA CURSOR]
        ctx.sinceRaw = q("since");
        ctx.since = V::Num(0.0);
        if !jsrt::is_nan(&ctx.sinceRaw) {
            ctx.since = V::Num(jsrt::parse_int(&ctx.sinceRaw));
        }
        // 고침 (fix-trading-nan-params): `since=` · `since=Infinity` 는 isNaN 을 지나 parseInt 가 NaN → 바인딩 NaN 으로 0건
        if ctx.since.to_num().is_nan() {
            ctx.since = V::Num(0.0);
        }
        // 고침 (fix-trading-delta-cursor): 원본은 클라이언트가 보낸 시각 max(updated_at, created_at) 을 서버 시계의 커서와
        // 견주어, 오프라인 편집 · 처음 올리는 옛 서류 · 시계가 늦은 기기의 서류처럼 '커서보다 옛 시각으로 나중에 들어온' 행을
        // 다른 기기가 영영 받지 못했다. created_at 상한도 시계가 빠른 기기의 서류를 커서 뒤로 흘려 보냈다
        // → 서버가 쓸 때 찍는 synced_at(migrations/0002) 하나로 고르고, 그 순서로 쪽을 나눈다 (아래 cursor)
        let mut where_ = FIXED_WHERE.to_string();
        ctx.binds = V::from(vec![oget(&cookies, "team"), V::from(TRADING_MODE), ctx.since.clone()]);
        let mut slot = 4.0;
        if q("ref").truthy() && eth::is_address(&q("ref")) {
            where_ += &tpl(T649, &[V::Num(slot)]);
            ctx.binds.push(V::from(to_lower(&q("ref").to_str())))?;
            slot += 1.0;
        }
        if q("doc_type").truthy() {
            // 고침 (fix-trading-double-decode): 쿼리는 URL 이 이미 풀었다. 원본은 한 번 더 decodeURIComponent 해
            // '%' 가 든 값이 URIError(418) 가 되거나 '%41' 이 'A' 로 바뀌었다
            ctx.wantType = V::from(normalize_doc_type(&q("doc_type")));
            where_ += &tpl(T655, &[V::Num(slot)]);
            ctx.binds.push(ctx.wantType.clone())?;
            slot += 1.0;
        }
        ctx.where_ = V::from(where_);
        ctx.limit = V::Num(1000.0);
        if !jsrt::is_nan(&q("limit")) {
            ctx.limit = V::Num(jsrt::math_max(1.0, jsrt::math_min(1000.0, jsrt::parse_int(&q("limit")))));
        }
        // 고침 (fix-trading-nan-params): `limit=` · `limit=Infinity` 는 "LIMIT NaN" 으로 질의가 실패해 빈 결과(커서도 없음)
        if ctx.limit.to_num().is_nan() {
            ctx.limit = V::Num(1000.0);
        }
        let mut results: Vec<V> = Vec::new();
        let mut next_cursor = jsrt::math_max(0.0, now - 1000.0);
        let r: R<()> = async {
            let stmt = env.prepare("DB", &tpl(FIXED_T668, &[ctx.where_.clone(), ctx.limit.clone()]))?;
            let stmt = js::bind(&stmt, &ctx.binds.items())?;
            ctx.q = js::all(&stmt).await?;
            let rows = ctx.q.get("results");
            results = if rows.truthy() { rows.items() } else { Vec::new() };
            // 고침 (fix-trading-delta-cursor): 원본은 LIMIT 에 잘려도 커서를 now 로 넘겨 남은 행을 잃었다. 쪽이 차면
            // 마지막 행의 synced_at 과 같은 나머지 행을 마저 붙이고(같은 시각을 둘로 나누지 않음) 커서를 그 시각으로 둔다
            if results.len() as f64 >= ctx.limit.to_num() {
                let last = results[results.len() - 1].clone();
                let last_stamp = last.get("synced_at").to_num();
                let stmt = env.prepare("DB", &tpl(FIXED_TAIL, &[ctx.where_.clone(), V::Num(slot), V::Num(slot + 1.0)]))?;
                let mut params = ctx.binds.items();
                params.push(V::Num(last_stamp));
                params.push(last.get("id"));
                let stmt = js::bind(&stmt, &params)?;
                let tail = js::all(&stmt).await?.get("results");
                if tail.truthy() {
                    results.extend(tail.items());
                }
                next_cursor = jsrt::math_min(next_cursor, last_stamp);
            }
            Ok(())
        }
        .await;
        if let Err(err) = r {
            console_log!("[TRADING] GET query err", err);
            return Ok(respond(json(&obj! { "results" => V::new_arr(), "session" => cookies.clone() })?, &headers));
        }
        // 🌟 [BASE64 CONTRACT] data 는 base64 문자열로
        let out = V::new_arr();
        for row in &results {
            let mut b64 = String::new();
            let r: R<()> = (|| {
                let data = row.get("data");
                if data.truthy() {
                    let bytes = if let V::Bytes(b) = &data {
                        b.borrow().clone()
                    } else {
                        let buffer = data.get("buffer");
                        jsrt::uint8array_new(if buffer.truthy() { &buffer } else { &data })?
                    };
                    b64 = bytes_to_base64(&bytes);
                }
                Ok(())
            })();
            if let Err(err) = r {
                console_log!("[TRADING] blob encode err", row.get("id"), err);
            }
            let mode = row.get("mode");
            let digest = row.get("digest");
            out.push(obj! {
                "id" => row.get("id"),
                "table" => "items",
                "type" => row.get("type"),
                "flag" => row.get("flag"),
                "from" => row.get("from"),
                "to" => row.get("to"),
                "cc" => row.get("cc"),
                "bcc" => row.get("bcc"),
                "ref" => row.get("ref"),
                "mode" => if mode.truthy() { mode } else { V::from(TRADING_MODE) },
                "digest" => if digest.truthy() { digest } else { V::from("") },
                "data" => b64,
                "created_at" => row.get("created_at"),
                "updated_at" => row.get("updated_at"),
            })?;
        }
        ctx.out = out.clone();
        // 고침 (fix-trading-proto-keys): Object.create(null) — 서식 이름이 '__proto__' 여도 보통 키 (로그만)
        ctx.brief = V::new_obj();
        for o in out.items() {
            let key = o.get("type").to_key();
            let prev = ctx.brief.get(&key);
            ctx.brief.s_(&key, add(&if prev.truthy() { prev } else { V::Num(0.0) }, &V::Num(1.0)))?;
        }
        console_log!(
            tpl(T721, &[oget(&cookies, "team"), ctx.since.clone(), ctx.created_at.clone()])
                + &tpl(T722, &[V::from(out.len()), V::from(json(&ctx.brief)?)])
        );
        return Ok(respond(json(&obj! { "results" => out, "cursor" => next_cursor, "session" => cookies.clone() })?, &headers));
    }

    // ── POST : 클라이언트 → 서버 동기화 (752~979) ──
    if request.method == "POST" {
        headers_set(&headers, "Content-Type", &V::from("application/json"))?;
        if unauthenticated {
            console_log!("[TRADING] ⚪ 미인증 POST. 저장하지 않습니다.");
            return Ok(respond(
                json(&obj! { "results" => V::new_arr(), "accepted" => 0, "skipped" => 0, "rejected" => 0, "session" => cookies.clone() })?,
                &headers,
            ));
        }
        // ── 본문 파싱 (gzip / 평문 모두 허용) ──
        ctx.payload = V::Null;
        let r: R<()> = (|| {
            if !request.body.is_empty() {
                let body_bytes = &request.body;
                let raw_text = match jsrt::pako_ungzip(body_bytes) {
                    Ok(b) => decode_utf8(&b),
                    Err(_) => decode_utf8(body_bytes),
                };
                ctx.payload = parse_str(&raw_text)?;
            }
            Ok(())
        })();
        if let Err(err) = r {
            console_log!("[TRADING] POST body parse err", err);
        }
        ctx.list = V::new_arr();
        if ctx.payload.is_arr() {
            ctx.list = ctx.payload.clone();
        } else if ctx.payload.truthy() && ctx.payload.get("items").is_arr() {
            ctx.list = ctx.payload.get("items");
        }
        if ctx.list.is_empty() {
            return Ok(respond(
                json(&obj! {
                    "results" => V::new_arr(), "accepted" => 0, "skipped" => 0, "rejected" => 0,
                    "cursor" => cursor(now), "session" => cookies.clone()
                })?,
                &headers,
            ));
        }
        // D1 배치 상한
        ctx.MAX_BATCH = V::Num(200.0);
        if ctx.list.len() > 200 {
            console_log!(tpl(T792, &[V::from(ctx.list.len()), ctx.MAX_BATCH.clone()]));
            ctx.list = V::from(ctx.list.items().into_iter().take(200).collect::<Vec<_>>());
        }
        let list = ctx.list.items();
        // ── digest 사전 조회 (SKIP GUARD 용) ──
        // 고침 (fix-trading-proto-keys): seenId · existing 은 Object.create(null) — 원본은 평범한 객체라 id 가 'constructor' 면
        // 이미 본 것으로 쳐 사전 조회에서 빠지고(늘 다시 씀), '__proto__' 행은 existing 의 프로토타입을 바꿨다
        ctx.seenId = V::new_obj();
        ctx.idList = V::new_arr();
        for item in &list {
            let id = item.get("id");
            // 고침 (fix-trading-id-case): 원본은 id 를 trim · 소문자로 바꿔 저장했다. 앱(store.rs upsert_item)은 id 를 그대로
            // 쓰므로 대문자가 든 id 는 내려받을 때 다른 문서가 되어(로컬 중복 · 묘비 · 봉투 반영 실패) 갈라졌다 → 보낸 그대로
            let rid = if item.truthy() && id.truthy() { id.to_str() } else { String::new() };
            if js_trim(&rid).is_empty() {
                continue;
            }
            if ctx.seenId.get(&rid).truthy() {
                continue;
            }
            ctx.seenId.s_(&rid, V::Bool(true))?;
            ctx.idList.push(V::from(rid))?;
        }
        ctx.existing = V::new_obj();
        // 고침 (fix-trading-preload-chunk): 원본은 1 + id 수만큼 바인딩해 100건을 넘으면 D1 이 거절했고(바인딩 상한 100),
        // 그 오류를 삼켜 SKIP GUARD 가 조용히 꺼졌다 → 99건씩 나눠 조회
        // 고침 (fix-trading-upsert-scope): 팀 조건 없이 id 로 찾아 "to" 를 같이 읽는다 (다른 팀의 행을 가려내려고)
        for chunk in ctx.idList.items().chunks(PRELOAD_CHUNK) {
            let r: R<()> = async {
                let ph = (0..chunk.len()).map(|n| format!("?{}", n + 1)).collect::<Vec<_>>().join(", ");
                ctx.ph = V::from(ph.as_str());
                let stmt = env.prepare("DB", &tpl(FIXED_T811, &[ctx.ph.clone()]))?;
                let stmt = js::bind(&stmt, chunk)?;
                ctx.q2 = js::all(&stmt).await?;
                let rows = ctx.q2.get("results");
                ctx.prevRows = if rows.truthy() { rows } else { V::new_arr() };
                for row in ctx.prevRows.items() {
                    ctx.existing.s_(&row.g("id")?.to_key(), row.clone())?;
                }
                Ok(())
            }
            .await;
            if let Err(err) = r {
                console_log!("[TRADING] digest preload err", err);
            }
        }
        // 고침 (fix-trading-delta-cursor): 문장은 반복 뒤에 synced_at 과 함께 묶는다 (아래)
        let mut pending: Vec<Vec<V>> = Vec::new();
        ctx.out = V::new_arr();
        let mut skipped = 0.0;
        let mut rejected = 0.0;
        for it in &list {
            if !it.truthy() || !it.get("id").truthy() {
                rejected += 1.0;
                continue;
            }
            let id = it.get("id").to_str();
            // 고침 (fix-trading-id-case): 빈 id(공백뿐)는 거부 — 원본은 trim 뒤 '' 를 id 로 저장했다
            if js_trim(&id).is_empty() {
                rejected += 1.0;
                continue;
            }
            let raw_type = it.get("type");
            let doc_type = normalize_doc_type(&if raw_type.truthy() { raw_type } else { it.get("doc_type") });
            // 🌟 무역 트랙 전용
            if !is_trade_doc_type(&V::from(doc_type.as_str())) {
                rejected += 1.0;
                console_log!(tpl(T835, &[V::from(id.as_str()), V::from(doc_type.as_str())]));
                continue;
            }
            // data 는 객체 / base64(gzip) / number[]
            let raw_data = it.get("data");
            let mut data = if raw_data.truthy() && raw_data.type_of() == "object" && !raw_data.is_arr() && !raw_data.get("buffer").truthy() {
                raw_data
            } else {
                decode_data(&raw_data)
            };
            if !data.truthy() || data.type_of() != "object" {
                data = V::new_obj();
            }
            // ── 봉투 재확정 ──
            let hub_key = resolve_hub_key(&V::from(doc_type.as_str()), &data, &V::from(id.as_str()));
            let ref_val = trading_ref(&oget(&cookies, "team"), &oget(&cookies, "cc"), &hub_key)?;
            let bcc_val = entity_bcc(&V::from(doc_type.as_str()), &oget(&cookies, "cc"))?;
            let it_flag = it.get("flag");
            let data_flag = data.get("flag");
            let flag_val = trim_lower(&if it_flag.truthy() {
                it_flag
            } else if data_flag.truthy() {
                data_flag
            } else {
                ctx.FLAG.clone()
            });
            let it_created = it.get("created_at");
            let mut created_at = if !it_created.is_nullish() { it_created } else { data.get("created_at") }.to_num();
            // 고침 (fix-trading-nan-params): Infinity 도 — 원본은 D1 이 NULL 로 저장해 그 행이 변경분에서 영영 빠졌다
            if created_at == 0.0 || !created_at.is_finite() {
                created_at = now;
            }
            // 🌟 [DRAFT PRESERVE] updated_at = 0 은 껍데기
            let it_updated = it.get("updated_at");
            let mut updated_at = if !it_updated.is_nullish() { it_updated } else { data.get("updated_at") }.to_num();
            if !updated_at.is_finite() {
                updated_at = 0.0;
            }
            // 🌟 [DIGEST] 클라이언트 값 우선
            let it_digest = it.get("digest");
            let mut digest_val = js_trim(&if it_digest.truthy() { it_digest.to_str() } else { String::new() }).to_string();
            if digest_val.is_empty() {
                digest_val = content_digest(&data);
            }
            let prev = ctx.existing.get(&id);
            // 고침 (fix-trading-upsert-scope): 원본의 ON CONFLICT (id) DO UPDATE 는 팀을 보지 않아, 다른 팀 행의 id 로 올리면
            // 그 행을 덮어쓰고 "to" 를 내 팀으로 바꿔 가져올 수 있었다 → 다른 팀의 id 는 거부 (문장에도 WHERE 로 한 번 더)
            if prev.truthy() && !loose_eq(&prev.get("to"), &oget(&cookies, "team")) {
                rejected += 1.0;
                console_log!(tpl(T_OTHER_TEAM, &[V::from(id.as_str())]));
                continue;
            }
            // ── SKIP GUARD ──
            if prev.truthy()
                && prev.get("digest").truthy()
                && prev.get("digest").to_str() == digest_val
                && prev.get("ref").to_str() == ref_val
                && prev.get("updated_at").to_num() == updated_at
            {
                skipped += 1.0;
                continue;
            }
            // ── 봉투를 data 안에도 ── (data 가 배열이면 이름 있는 속성은 JSON 에 나가지 않는다)
            data.s_("id", V::from(id.as_str()))?;
            data.s_("type", V::from(doc_type.as_str()))?;
            data.s_("mode", V::from(TRADING_MODE))?;
            data.s_("flag", V::from(flag_val.as_str()))?;
            data.s_("cc", oget(&cookies, "cc"))?;
            data.s_("bcc", V::from(bcc_val.as_str()))?;
            data.s_("ref", V::from(ref_val.as_str()))?;
            data.s_("from", oget(&cookies, "address"))?;
            data.s_("to", oget(&cookies, "team"))?;
            data.s_("digest", V::from(digest_val.as_str()))?;
            data.s_("created_at", V::Num(created_at))?;
            data.s_("updated_at", V::Num(updated_at))?;
            // blobArr.buffer 가 없으므로 blob = blobArr (ArrayBuffer)
            let blob = encode_data(&data)?;
            pending.push(vec![
                V::from(id.as_str()),
                V::from(doc_type.as_str()),
                V::from(flag_val.as_str()),
                oget(&cookies, "address"),
                oget(&cookies, "team"),
                oget(&cookies, "cc"),
                V::from(bcc_val.as_str()),
                V::from(ref_val.as_str()),
                V::from(TRADING_MODE),
                V::from(digest_val.as_str()),
                blob,
                V::Num(created_at),
                V::Num(updated_at),
            ]);
            // 🌟 서버가 확정한 봉투를 그대로 돌려준다
            ctx.out.push(obj! {
                "id" => id.as_str(),
                "table" => "items",
                "type" => doc_type.as_str(),
                "flag" => flag_val.as_str(),
                "from" => oget(&cookies, "address"),
                "to" => oget(&cookies, "team"),
                "cc" => oget(&cookies, "cc"),
                "bcc" => bcc_val.as_str(),
                "ref" => ref_val.as_str(),
                "mode" => TRADING_MODE,
                "digest" => digest_val.as_str(),
                "hub_key" => hub_key.as_str(),
                "created_at" => created_at,
                "updated_at" => updated_at,
            })?;
        }
        // 고침 (fix-trading-delta-cursor): synced_at 은 쓰기 바로 앞의 서버 시각 — 해시 · 사전 조회에 든 시간만큼 커서 여유(1초)를
        // 깎아 먹지 않게. 고침 (fix-trading-upsert-scope): DO UPDATE 에 WHERE items."to" = EXCLUDED."to"
        let synced_at = platform::now();
        ctx.statements = V::new_arr();
        for mut params in pending {
            params.push(V::Num(synced_at));
            let stmt = env.prepare("DB", FIXED_UPSERT)?;
            ctx.statements.push(js::bind(&stmt, &params)?)?;
        }
        if !ctx.statements.is_empty() {
            if let Err(err) = env.batch("DB", &ctx.statements).await {
                console_log!("[TRADING] POST batch err", err);
                return Ok(respond(
                    json(&obj! {
                        "results" => V::new_arr(), "accepted" => 0, "skipped" => skipped, "rejected" => rejected,
                        "error" => err.to_string(), "session" => cookies.clone()
                    })?,
                    &headers,
                ));
            }
        }
        // 고침 (fix-trading-proto-keys): Object.create(null) (키는 ref 라 실제로는 겹치지 않음)
        ctx.hubBrief = V::new_obj();
        for o in ctx.out.items() {
            let key = o.get("ref").to_key();
            let prev = ctx.hubBrief.get(&key);
            ctx.hubBrief.s_(&key, add(&if prev.truthy() { prev } else { V::Num(0.0) }, &V::Num(1.0)))?;
        }
        console_log!(
            tpl(T967, &[oget(&cookies, "team")])
                + &tpl(T968, &[V::from(ctx.out.len()), V::Num(skipped), V::Num(rejected)])
                + &tpl(T969, &[V::from(ctx.hubBrief.keys().len())])
        );
        let out = ctx.out.clone();
        return Ok(respond(
            json(&obj! {
                "results" => out.clone(),
                "accepted" => out.len(),
                "skipped" => skipped,
                "rejected" => rejected,
                "cursor" => cursor(now),
                "session" => cookies.clone(),
            })?,
            &headers,
        ));
    }

    // ── DELETE : 단건/다건 삭제 (989~1067) ──
    if request.method == "DELETE" {
        headers_set(&headers, "Content-Type", &V::from("application/json"))?;
        if unauthenticated {
            return Ok(respond(json(&obj! { "results" => V::new_arr(), "deleted" => 0, "session" => cookies.clone() })?, &headers));
        }
        let mut del_ids: Vec<String> = Vec::new();
        // 고침 (fix-trading-id-case): id 는 보낸 그대로 (원본은 trim · 소문자)
        let push_del_id = |del_ids: &mut Vec<String>, v: &V| {
            let s = if v.truthy() { v.to_str() } else { String::new() };
            if js_trim(&s).is_empty() || del_ids.contains(&s) {
                return;
            }
            del_ids.push(s);
        };
        if q("id").truthy() {
            // 고침 (fix-trading-double-decode): 쿼리는 이미 풀렸다 (원본은 '%' 가 든 id 로 URIError → 418)
            push_del_id(&mut del_ids, &q("id"));
        }
        if q("ids").truthy() {
            let r: R<()> = (|| {
                ctx.qArr = parse_str(&q("ids").to_str())?;
                if ctx.qArr.is_arr() {
                    for x in ctx.qArr.items() {
                        push_del_id(&mut del_ids, &x);
                    }
                }
                Ok(())
            })();
            if let Err(err) = r {
                console_log!("[TRADING] DELETE ids parse err", err);
            }
        }
        // URL 길이 제한을 피하려면 본문으로도
        let r: R<()> = (|| {
            if !request.body.is_empty() {
                let dbytes = &request.body;
                let dtext = match jsrt::pako_ungzip(dbytes) {
                    Ok(b) => decode_utf8(&b),
                    Err(_) => decode_utf8(dbytes),
                };
                ctx.dbody = parse_str(&dtext)?;
                ctx.dlist = if ctx.dbody.is_arr() {
                    ctx.dbody.clone()
                } else if ctx.dbody.truthy() && ctx.dbody.get("ids").truthy() {
                    ctx.dbody.get("ids")
                } else {
                    V::new_arr()
                };
                if ctx.dlist.is_arr() {
                    for x in ctx.dlist.items() {
                        push_del_id(&mut del_ids, &x);
                    }
                }
            }
            Ok(())
        })();
        // 본문이 없는 DELETE 가 정상 경로이므로 조용히 넘어간다
        let _ = r;
        if del_ids.is_empty() {
            return Ok(respond(json(&obj! { "results" => V::new_arr(), "deleted" => 0, "session" => cookies.clone() })?, &headers));
        }
        del_ids.truncate(200);
        ctx.delStatements = V::new_arr();
        for id in &del_ids {
            // 🌟 [TEAM SCOPE] "to" 조건이 유일한 권한 경계
            let stmt = env.prepare("DB", T1049[0])?;
            ctx.delStatements.push(js::bind(&stmt, &[V::from(id.as_str()), oget(&cookies, "team")])?)?;
        }
        if let Err(err) = env.batch("DB", &ctx.delStatements).await {
            console_log!("[TRADING] DELETE batch err", err);
            return Ok(respond(
                json(&obj! { "results" => V::new_arr(), "deleted" => 0, "error" => err.to_string(), "session" => cookies.clone() })?,
                &headers,
            ));
        }
        console_log!(tpl(T1061, &[oget(&cookies, "team"), V::from(del_ids.len())]));
        let results = V::from(del_ids.iter().map(|x| obj! { "id" => x.as_str(), "deleted" => true }).collect::<Vec<_>>());
        return Ok(respond(json(&obj! { "results" => results, "deleted" => del_ids.len(), "session" => cookies.clone() })?, &headers));
    }
    // 그 외 메서드는 지원하지 않는다
    Ok(respond(json(&obj! { "results" => V::new_arr(), "session" => cookies.clone() })?, &headers))
}

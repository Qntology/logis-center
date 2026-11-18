//! 원본 456~1079: fetch 처리기. 바깥 try(478~1074) 안에서 던진 것은 `fatal err` 로 남기고 418, 그 앞(AwsClient 생성 ·
//! `request.cf` 구조 분해)에서 던진 것은 처리기 밖으로 나간다(`Err` → 진입점이 JS 예외로 던짐).

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
    // 464: var { country } = request.cf — cf 가 없으면 try 밖에서 TypeError
    if request.cf.is_nullish() {
        return Err(Throw::type_error(format!(
            "Cannot destructure property 'country' of 'request.cf' as it is {}.",
            request.cf.to_str()
        )));
    }
    ctx.country = cf_get(&request.cf, "country");
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
            if parts.len() == 2 {
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
            // 원본 그대로: token 머리글도 token 쿠키도 없으면 null == undefined 라 hash 만으로 통과한다
            if ctx.email.truthy() && loose_eq(&ctx.token, &oget(&cookies, "token")) {
                cookies.s_("email", ctx.email.clone())?;
                cookies.s_("address", ctx.address.clone())?;
                // 🌟 팀 판정은 commerce Worker 에 위임
                ctx.resolved = resolve_team(env, &oget(&cookies, "hash"), &oget(&cookies, "token"), &ctx.address, &q("to")).await?;
                cookies.s_("team", ctx.resolved.g("team")?)?;
                cookies.s_("role", ctx.resolved.g("role")?)?;
                if loose_eq(&ctx.resolved.g("role")?, &V::from("denied")) {
                    console_log!(tpl(T556, &[ctx.address.clone()]));
                }
            }
            headers_set(&headers, "Set-Cookie", &V::from(tpl(T559, &[oget(&cookies, "hash")])))?;
            headers_append(&headers, "Set-Cookie", &V::from(tpl(T560, &[oget(&cookies, "token")])))?;
            if oget(&cookies, "team").truthy() {
                headers_append(&headers, "Set-Cookie", &V::from(tpl(T562, &[oget(&cookies, "team")])))?;
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

    ctx.requestOrigin = request.header("Origin");
    headers_set(&headers, "Access-Control-Allow-Origin", &if ctx.requestOrigin.truthy() { ctx.requestOrigin.clone() } else { V::from("*") })?;
    headers_set(&headers, "Access-Control-Allow-Credentials", &V::from("true"))?;
    headers_set(&headers, "Access-Control-Allow-Methods", &V::from("GET, POST, DELETE, OPTIONS"))?;
    headers_set(&headers, "Access-Control-Allow-Headers", &V::from("Content-Type, Content-Encoding, Authorization, X-Requested-With"))?;

    if request.method == "OPTIONS" {
        return Ok(respond(String::new(), &headers));
    }

    let unauthenticated = !oget(&cookies, "address").truthy() || !oget(&cookies, "team").truthy();

    // ── GET : 서버 → 클라이언트 동기화 (618~736) ──
    if request.method == "GET" {
        if !loose_eq(&ctx.contentType, &V::from("application/json")) {
            headers_set(&headers, "Content-Type", &V::from("text/html; charset=utf-8"))?;
            let pretty = jsrt::stringify_pretty(&cookies)?.map_or(V::Undef, V::from);
            return Ok(respond(T622[0].to_string() + &tpl(T623, &[pretty]), &headers));
        }
        headers_set(&headers, "Content-Type", &V::from("application/json"))?;
        // 원본 그대로: address · team 은 Cookie 머리글에서 온 값이어도 통과한다 (S3 세션이 실패해도 쿠키의 team 으로 읽음)
        if unauthenticated {
            console_log!("[TRADING] ⚪ 미인증 요청. 빈 배열을 반환합니다.");
            return Ok(respond(json(&obj! { "results" => V::new_arr(), "session" => cookies.clone() })?, &headers));
        }
        // 🌟 [DELTA CURSOR] max(updated_at, created_at) > since
        ctx.sinceRaw = q("since");
        ctx.since = V::Num(0.0);
        if !jsrt::is_nan(&ctx.sinceRaw) {
            ctx.since = V::Num(jsrt::parse_int(&ctx.sinceRaw));
        }
        let mut where_ = T643[0].to_string();
        ctx.binds = V::from(vec![oget(&cookies, "team"), V::from(TRADING_MODE), ctx.created_at.clone(), ctx.since.clone()]);
        let mut slot = 5.0;
        if q("ref").truthy() && eth::is_address(&q("ref")) {
            where_ += &tpl(T649, &[V::Num(slot)]);
            ctx.binds.push(V::from(to_lower(&q("ref").to_str())))?;
            slot += 1.0;
        }
        if q("doc_type").truthy() {
            // decodeURIComponent 의 URIError 는 바깥 try 까지 (418)
            ctx.wantType = V::from(normalize_doc_type(&V::from(jsrt::decode_uri_component(&q("doc_type").to_str())?)));
            where_ += &tpl(T655, &[V::Num(slot)]);
            ctx.binds.push(ctx.wantType.clone())?;
            slot += 1.0;
        }
        let _ = slot;
        ctx.where_ = V::from(where_);
        ctx.limit = V::Num(1000.0);
        if !jsrt::is_nan(&q("limit")) {
            // 원본 그대로: limit="" · "Infinity" 등은 NaN → "LIMIT NaN" 으로 질의 실패 → 빈 결과
            ctx.limit = V::Num(jsrt::math_max(1.0, jsrt::math_min(1000.0, jsrt::parse_int(&q("limit")))));
        }
        let mut results: Vec<V> = Vec::new();
        let r: R<()> = async {
            let stmt = env.prepare("DB", &tpl(T668, &[ctx.where_.clone(), ctx.limit.clone()]))?;
            let stmt = js::bind(&stmt, &ctx.binds.items())?;
            ctx.q = js::all(&stmt).await?;
            let rows = ctx.q.get("results");
            results = if rows.truthy() { rows.items() } else { Vec::new() };
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
        ctx.brief = V::new_obj();
        for o in out.items() {
            let key = o.get("type").to_key();
            let prev = oget(&ctx.brief, &key);
            oset(&ctx.brief, &key, add(&if prev.truthy() { prev } else { V::Num(0.0) }, &V::Num(1.0)))?;
        }
        console_log!(
            tpl(T721, &[oget(&cookies, "team"), ctx.since.clone(), ctx.created_at.clone()])
                + &tpl(T722, &[V::from(out.len()), V::from(json(&ctx.brief)?)])
        );
        return Ok(respond(json(&obj! { "results" => out, "cursor" => cursor(now), "session" => cookies.clone() })?, &headers));
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
        // ── digest 사전 조회 (SKIP GUARD 용, 쿼리 1회) ──
        ctx.seenId = V::new_obj();
        ctx.idList = V::new_arr();
        for item in &list {
            let id = item.get("id");
            let rid = if item.truthy() && id.truthy() { trim_lower(&id) } else { String::new() };
            if rid.is_empty() {
                continue;
            }
            // 원본 그대로: seenId 는 평범한 객체라 'constructor' · '__proto__' 는 이미 본 것으로 친다
            if oget(&ctx.seenId, &rid).truthy() {
                continue;
            }
            oset(&ctx.seenId, &rid, V::Bool(true))?;
            ctx.idList.push(V::from(rid))?;
        }
        ctx.existing = V::new_obj();
        if !ctx.idList.is_empty() {
            let r: R<()> = async {
                let ph = (0..ctx.idList.len()).map(|n| format!("?{}", n + 2)).collect::<Vec<_>>().join(", ");
                ctx.ph = V::from(ph.as_str());
                let stmt = env.prepare("DB", &tpl(T811, &[ctx.ph.clone()]))?;
                let mut params = vec![oget(&cookies, "team")];
                params.extend(ctx.idList.items());
                // 원본 그대로: 100건을 넘으면 D1 의 바인딩 상한(100)에 걸려 사전 조회가 실패하고 SKIP GUARD 가 꺼진다
                let stmt = js::bind(&stmt, &params)?;
                ctx.q2 = js::all(&stmt).await?;
                let rows = ctx.q2.get("results");
                ctx.prevRows = if rows.truthy() { rows } else { V::new_arr() };
                for row in ctx.prevRows.items() {
                    oset(&ctx.existing, &row.g("id")?.to_key(), row.clone())?;
                }
                Ok(())
            }
            .await;
            if let Err(err) = r {
                console_log!("[TRADING] digest preload err", err);
            }
        }
        ctx.statements = V::new_arr();
        ctx.out = V::new_arr();
        let mut skipped = 0.0;
        let mut rejected = 0.0;
        for it in &list {
            if !it.truthy() || !it.get("id").truthy() {
                rejected += 1.0;
                continue;
            }
            let id = trim_lower(&it.get("id"));
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
            if created_at == 0.0 || created_at.is_nan() {
                created_at = now;
            }
            // 🌟 [DRAFT PRESERVE] updated_at = 0 은 껍데기
            let it_updated = it.get("updated_at");
            let mut updated_at = if !it_updated.is_nullish() { it_updated } else { data.get("updated_at") }.to_num();
            if updated_at.is_nan() {
                updated_at = 0.0;
            }
            // 🌟 [DIGEST] 클라이언트 값 우선
            let it_digest = it.get("digest");
            let mut digest_val = js_trim(&if it_digest.truthy() { it_digest.to_str() } else { String::new() }).to_string();
            if digest_val.is_empty() {
                digest_val = content_digest(&data);
            }
            // ── SKIP GUARD ──
            let prev = oget(&ctx.existing, &id);
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
            let stmt = env.prepare("DB", T904[0])?;
            let stmt = js::bind(
                &stmt,
                &[
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
                ],
            )?;
            ctx.statements.push(stmt)?;
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
        ctx.hubBrief = V::new_obj();
        for o in ctx.out.items() {
            let key = o.get("ref").to_key();
            let prev = oget(&ctx.hubBrief, &key);
            oset(&ctx.hubBrief, &key, add(&if prev.truthy() { prev } else { V::Num(0.0) }, &V::Num(1.0)))?;
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
        let push_del_id = |del_ids: &mut Vec<String>, v: &V| {
            let s = trim_lower(&if v.truthy() { v.clone() } else { V::from("") });
            if s.is_empty() || del_ids.contains(&s) {
                return;
            }
            del_ids.push(s);
        };
        if q("id").truthy() {
            // decodeURIComponent 의 URIError 는 바깥 try 까지 (418)
            let id = jsrt::decode_uri_component(&q("id").to_str())?;
            push_del_id(&mut del_ids, &V::from(id));
        }
        if q("ids").truthy() {
            let r: R<()> = (|| {
                ctx.qArr = parse_str(&jsrt::decode_uri_component(&q("ids").to_str())?)?;
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

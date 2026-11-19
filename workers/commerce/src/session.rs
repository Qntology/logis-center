//! 원본 341~410 (try 밖: S3 클라이언트 · request.cf · ip · geoInfo · 지역) 와 411~965 (try 안: 머리글 · 쿠키 · 질의 · 언어 ·
//! S3 세션 · 사용자 · 팀 · 잔액 · 새 세션).
//!
//! `const s3 = new AwsClient({...})` 는 env 만으로 정해지는 값이라 Ctx 칸 대신 쓰는 곳마다 [`s3`] 로 다시 만든다.
//! 생성자가 던지는 자리(키가 없을 때)는 [`head`] 가 원본과 같은 때에 [`s3_checked`] 로 던진다.

use logis_worker_common::gzip::{decode_utf8, gzip_json, ungzip};
use logis_worker_common::js::{self, Env};
use logis_worker_common::jsv::{index_of, less_than, quote, to_lower, tpl, Throw};
use logis_worker_common::tables::{lookup, CENTER_REGION, HELLO, LANGUAGE_CODE, LOGIS_REGION, TWO_PART_DOMAINS};
use logis_worker_common::util::hash_id;
use logis_worker_common::web::{headers_append, headers_new, headers_set, Req};
use logis_worker_common::{add, aws, console_log, date, eth, loose_eq, obj, parse_str, platform, stringify_str, R, V};

use crate::ctx::Ctx;
use crate::flow::Step;
use crate::templates::*;

/// `new AwsClient({ accessKeyId: env.aws_access_key_id, secretAccessKey: env.aws_secret_access_key, service: 's3', region: env.aws_region })`
pub fn s3(env: &Env) -> aws::AwsClient {
    aws::AwsClient::new(env.get("aws_access_key_id"), env.get("aws_secret_access_key"), V::from("s3"), env.get("aws_region"))
}

/// [`s3`] + aws4fetch 1.0.20 생성자의 검사 (`accessKeyId == null` · `secretAccessKey == null` 이면 TypeError)
pub fn s3_checked(env: &Env) -> R<aws::AwsClient> {
    let client = s3(env);
    if client.access_key_id.is_nullish() {
        return Err(Throw::type_error("accessKeyId is a required option"));
    }
    if client.secret_access_key.is_nullish() {
        return Err(Throw::type_error("secretAccessKey is a required option"));
    }
    Ok(client)
}

/// `decodeURIComponent(s)` — 잘못된 `%` 꼴 · UTF-8 이면 `URIError: URI malformed`
pub fn decode_uri_component(s: &str) -> R<String> {
    let malformed = || Throw { name: "URIError", message: "URI malformed".into() };
    let hex_at = |b: &[u8], k: usize| -> Option<u8> {
        if k + 2 >= b.len() {
            return None;
        }
        let h = |c: u8| (c as char).to_digit(16);
        Some((h(b[k + 1])? * 16 + h(b[k + 2])?) as u8)
    };
    let b = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut k = 0;
    while k < b.len() {
        if b[k] != b'%' {
            out.push(b[k]);
            k += 1;
            continue;
        }
        let first = hex_at(b, k).ok_or_else(malformed)?;
        k += 3;
        if first & 0x80 == 0 {
            out.push(first);
            continue;
        }
        let n = first.leading_ones() as usize;
        if n == 1 || n > 4 {
            return Err(malformed());
        }
        let mut octets = vec![first];
        for _ in 1..n {
            if k >= b.len() || b[k] != b'%' {
                return Err(malformed());
            }
            let x = hex_at(b, k).ok_or_else(malformed)?;
            if x & 0xC0 != 0x80 {
                return Err(malformed());
            }
            octets.push(x);
            k += 3;
        }
        // 넘친 길이 · 대리 쌍 자리 · 0x10FFFF 넘음은 원본도 URIError
        std::str::from_utf8(&octets).map_err(|_| malformed())?;
        out.extend_from_slice(&octets);
    }
    String::from_utf8(out).map_err(|_| malformed())
}

/// `s.match(/\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/)` 의 `[0]` (없으면 None). JS 의 `\d` · `\b` 는 ASCII 라 바이트로 본다
fn match_ipv4(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let word = |i: usize| i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_');
    let boundary = |i: usize| (i > 0 && word(i - 1)) != word(i);
    // 욕심 많은 {1,3} 의 되돌아가기 순서(3, 2, 1) 그대로
    fn go(b: &[u8], pos: usize, group: usize, boundary: &dyn Fn(usize) -> bool) -> Option<usize> {
        for cnt in (1..=3).rev() {
            if pos + cnt > b.len() || !b[pos..pos + cnt].iter().all(u8::is_ascii_digit) {
                continue;
            }
            let next = pos + cnt;
            if group < 3 {
                if next < b.len() && b[next] == b'.' {
                    if let Some(end) = go(b, next + 1, group + 1, boundary) {
                        return Some(end);
                    }
                }
            } else if boundary(next) {
                return Some(next);
            }
        }
        None
    }
    (0..b.len()).filter(|&i| b[i].is_ascii_digit() && boundary(i)).find_map(|start| go(b, start, 0, &boundary).map(|end| s[start..end].to_string()))
}

/// `JSON.stringify(err)` — Error 객체는 열거되는 속성이 없어 "{}", 값을 던진 것(pako 의 글)은 그 글의 JSON
fn err_json(err: &Throw) -> V {
    if err.name.is_empty() {
        V::from(quote(&err.message))
    } else {
        V::from("{}")
    }
}

/// `s.toLowerCase()` — 문자열이 아니면 원본처럼 TypeError
fn lower(v: &V, what: &str) -> R<String> {
    match v {
        V::Str(s) => Ok(to_lower(s)),
        V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'toLowerCase')", v.to_str()))),
        _ => Err(Throw::type_error(format!("{what}.toLowerCase is not a function"))),
    }
}

/// `new URL(...)` 의 쓰는 속성만 담은 값 (`var url` — 뒤 구역은 다시 대입한 뒤에만 읽는다)
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

fn invalid_url() -> Throw {
    Throw::type_error("Invalid URL string.")
}

/// `response.headers.forEach((value, key) => { if (key.startsWith('x-amz-meta-')) { metaKey = …; headObject[metaKey] = value } })`
fn head_object(ctx: &mut Ctx, res: &wasm_bindgen::JsValue) -> R<()> {
    ctx.headObject = V::new_obj();
    for (key, value) in js::response_headers(res)? {
        if key.starts_with("x-amz-meta-") {
            ctx.metaKey = V::from(key.replacen("x-amz-meta-", "", 1));
            ctx.headObject.sk(&ctx.metaKey, V::from(value))?;
        }
    }
    Ok(())
}

/// 341~410 — 여기서 던지면 원본은 fetch 가 예외로 끝난다(try 밖)
pub async fn head(ctx: &mut Ctx, env: &Env, req: &Req) -> R<()> {
    // 346
    s3_checked(env)?;

    // 353~376 `var { city, … } = request.cf`
    if req.cf.is_nullish() {
        return Err(Throw::type_error(format!("Cannot destructure property 'city' of 'request.cf' as it is {}.", req.cf.to_str())));
    }
    let cf = &req.cf;
    ctx.city = cf.get("city");
    ctx.country = cf.get("country");
    ctx.countryRegion = cf.get("countryRegion");
    ctx.continent = cf.get("continent");
    ctx.latitude = cf.get("latitude");
    ctx.longitude = cf.get("longitude");
    ctx.region = cf.get("region");
    ctx.regionCode = cf.get("regionCode");
    ctx.timezone = cf.get("timezone");
    ctx.postalCode = cf.get("postalCode");
    ctx.asOrganization = cf.get("asOrganization");

    // 379
    ctx.ip = req.header("cf-connecting-ip");
    if !ctx.ip.truthy() {
        ctx.ip = req.header("X-Real-IP");
    }

    // 386
    ctx.geoInfo = obj! {
        "ip" => ctx.ip.clone(),
        "city" => ctx.city.clone(),
        "country" => ctx.country.clone(),
        "countryRegion" => ctx.countryRegion.clone(),
        "continent" => ctx.continent.clone(),
        "latitude" => ctx.latitude.clone(),
        "longitude" => ctx.longitude.clone(),
        "region" => ctx.region.clone(),
        "regionCode" => ctx.regionCode.clone(),
        "timezone" => ctx.timezone.clone(),
        "postalCode" => ctx.postalCode.clone(),
        "asOrganization" => ctx.asOrganization.clone(),
    };

    // 401 — country 가 없으면 TypeError (try 밖)
    ctx.FLAG = V::from(lower(&ctx.geoInfo.g("country")?, "geoInfo.country")?);
    ctx.logisRegion = lookup(LOGIS_REGION, &ctx.FLAG);

    // 고침 (fix-region-fallback): wrangler 는 APAC 지역 DB 만 묶는다 — 묶이지 않은 지역(US → commerce_logis_enam 등)은 모든 요청이 418 이었다
    if !ctx.logisRegion.truthy() || env.get(&ctx.logisRegion.to_key()).is_undef() {
        ctx.logisRegion = V::from("commerce_logis_apac");
    }
    ctx.zoneRegion = V::from("");
    ctx.language = lookup(LANGUAGE_CODE, &ctx.FLAG);

    console_log!("geoInfo", stringify_str(&ctx.geoInfo));
    Ok(())
}

/// 411~965
pub async fn run(ctx: &mut Ctx, env: &Env, req: &Req) -> R<Step> {
    // 412
    ctx.headers = headers_new();
    ctx.cookies = V::new_obj();
    ctx.cookiesStr = req.header("Cookie");
    ctx.contentType = req.header("Content-Type");
    ctx.contentEncoding = req.header("Content-Encoding");

    // 421
    if let V::Str(s) = &ctx.cookiesStr {
        for cookie in s.split(';') {
            let parts: Vec<&str> = cookie.split('=').collect();
            if parts.len() == 2 {
                let key = parts[0].trim_matches(logis_worker_common::jsv::is_js_space);
                let value = parts[1].trim_matches(logis_worker_common::jsv::is_js_space);
                // 고침 (fix-cookie-whitelist): 머리글에서 믿는 쿠키는 세션 키(hash · token)뿐 — address · team · sender · balance · cc … 는 세션이 다시 만든다
                if key == "hash" || key == "token" {
                    ctx.cookies.s_(key, V::from(value))?;
                }
            }
        }
    }

    // 432
    let request_url = url::Url::parse(&req.url).map_err(|_| invalid_url())?;
    let query = V::new_obj();
    ctx.req = obj! {
        "url" => req.url.as_str(),
        "host" => request_url.host_str().unwrap_or(""),
        "method" => req.method.as_str(),
        "query" => query.clone(),
        "body" => V::new_obj(),
    };
    // 443 `queryParams.forEach((value, key) => req.query[key] = value)` — 같은 키는 뒤 값, 자리는 처음 자리
    for (key, value) in request_url.query_pairs() {
        query.s_(&key, V::from(value.into_owned()))?;
    }

    // 451~465 `preferredLanguage` 는 let 이고 어디서도 읽지 않는다(던지는 자리도 없음)

    // 468
    ctx.redirect = V::from("https://commerce.logis.center/");

    let q_origin = query.get("origin");
    ctx.origin = V::from(decode_uri_component(&if q_origin.truthy() { q_origin.to_str() } else { String::new() })?);

    ctx.userAgent = req.header("User-Agent");

    // 475
    if ctx.ip.truthy() {
        ctx.m = match match_ipv4(&ctx.ip.to_str()) {
            Some(m) => V::from(vec![V::from(m)]),
            None => V::Null,
        };
        if ctx.m.truthy() {
            ctx.ip = ctx.m.gi(0)?;
        }
    }

    // 483 — language 가 표에 없으면 undefined
    ctx.cookies.s_("hello", lookup(HELLO, &ctx.language))?;
    ctx.cookies.s_("flag", ctx.FLAG.clone())?;

    console_log!("cookies", stringify_str(&ctx.cookies));

    console_log!("req.query.referrer", query.get("referrer"));
    let q_href = query.get("href");
    ctx.href = V::from(to_lower(&decode_uri_component(&if q_href.truthy() { q_href.to_str() } else { String::new() })?));

    // 492
    if ctx.origin.truthy() && ctx.href.truthy() {
        let r: R<()> = (|| {
            let href = ctx.href.to_str();
            let origin = ctx.origin.to_str();
            // 496 `var url;` 는 값을 지우지 않는다
            let u = match url::Url::parse(&href) {
                Ok(u) => u,
                Err(_) => {
                    let base = url::Url::parse(&origin).map_err(|_| invalid_url())?;
                    base.join(&href).map_err(|_| invalid_url())?
                }
            };
            ctx.url = url_obj(&u);

            if index_of(&href, &origin) > -1 || ctx.url.get("origin").to_str() == origin {
                ctx.currentHost = ctx.url.get("host");
                let current_host = ctx.currentHost.to_str();

                let host: Vec<String> = current_host.split('.').map(str::to_string).collect();
                ctx.host = V::from(host.iter().map(V::from).collect::<Vec<_>>());

                // 원본 그대로: endsWith 라 "xco.kr" 같은 이름도 두 부분 도메인으로 친다
                ctx.isTwoPartDomain = V::Bool(TWO_PART_DOMAINS.iter().any(|d| current_host.ends_with(d)));

                ctx.cc = V::from("");

                // 원본 그대로: 조각이 모자라면 host[-1] 등이 undefined 라 "undefined" 가 들어간다
                let part = |back: usize| -> String {
                    if host.len() >= back {
                        host[host.len() - back].clone()
                    } else {
                        "undefined".into()
                    }
                };
                if ctx.isTwoPartDomain.truthy() {
                    ctx.cc = V::from(part(3) + "." + &part(2) + "." + &part(1));
                } else {
                    ctx.cc = V::from(part(2) + "." + &part(1));
                }

                console_log!("cc", ctx.cc);

                ctx.cookies.s_("cc", V::from(hash_id(&ctx.cc)?))?;
            }
            Ok(())
        })();
        if let Err(err) = r {
            console_log!("err", err);
        }
    }

    // 531
    ctx.now = V::Num(platform::now());
    ctx.current = V::from(date::to_iso_string(ctx.now.to_num())?);

    // 535 `var balance` — 선언만(값을 지우지 않음)

    // 537
    let q_hash = query.get("hash");
    let q_token = query.get("token");
    if ctx.cookies.get("hash").truthy() || (q_hash.truthy() && q_token.truthy()) {
        let r = session(ctx, env, &query).await;
        if let Err(err) = r {
            console_log!("errrrr", err);
            ctx.cookies = obj! {
                "err" => err_json(&err),
                "logisRegion" => ctx.logisRegion.clone(),
                "zoneRegion" => ctx.zoneRegion.clone(),
            };
        }
    }

    console_log!("zoneRegion", ctx.zoneRegion);

    // 922
    if ctx.balance.is_undef() {
        let (address, private_key) = eth::random_wallet();
        ctx.account = obj! { "address" => address.as_str(), "privateKey" => private_key.as_str() };
        ctx.hash = V::from(to_lower(&address));
        ctx.hash = V::from(ctx.hash.to_str().replacen("0x", "", 1));

        ctx.token = V::from(to_lower(&private_key));
        ctx.token = V::from(ctx.token.to_str().replacen("0x", "", 1));

        let r: R<()> = async {
            let client = s3(env);
            let url = tpl(T932, &[env.get("aws_bucket"), env.get("aws_region"), ctx.hash.clone()]);
            let mut headers: Vec<(String, String)> = vec![("Content-Type".into(), "text/plain".into())];
            // 원본 그대로: ip 가 null 이면 Headers 가 "null" 로 바꿔 넣는다
            for (key, value) in [
                ("balance", V::from("1000")),
                ("host", V::from("logis.center")),
                ("token", ctx.token.clone()),
                ("email", V::from("")),
                ("ip", ctx.ip.clone()),
                ("phone", V::from("")),
                ("address", V::from("")),
            ] {
                headers.push((tpl(T945, &[V::from(key)]), value.to_str()));
            }
            let signed = client.sign(&url, "PUT", &headers, b"", platform::now())?;
            // 원본 그대로: 응답 상태를 보지 않는다(실패해도 새 hash · token 쿠키를 낸다)
            let res = aws::send(&signed).await?;
            ctx.response = V::ext(js::Js(res));

            ctx.cookies.s_("token", ctx.token.clone())?;
            ctx.cookies.s_("hash", ctx.hash.clone())?;

            headers_set(&ctx.headers, "Set-Cookie", &V::from(tpl(T959, &[ctx.hash.clone()])))?;
            headers_append(&ctx.headers, "Set-Cookie", &V::from(tpl(T960, &[ctx.token.clone()])))?;
            Ok(())
        }
        .await;
        if let Err(err) = r {
            ctx.cookies.s_("err", err_json(&err))?;
            console_log!("erraasd", err);
        }
    }

    Ok(Step::Next)
}

/// 538~913 (try 안 — 던지면 916 에서 cookies 를 오류 객체로 바꾼다)
async fn session(ctx: &mut Ctx, env: &Env, query: &V) -> R<()> {
    let cookies = ctx.cookies.clone();
    let q_hash = query.get("hash");
    let q_token = query.get("token");

    if cookies.get("hash").truthy() && q_hash.truthy() && !loose_eq(&cookies.get("hash"), &q_hash) {
        cookies.s_("signature", V::from(""))?;
    }

    if q_hash.truthy() {
        cookies.s_("hash", q_hash.clone())?;
    }

    if q_token.truthy() {
        cookies.s_("token", q_token.clone())?;
    }

    // 553
    let client = s3(env);
    let url = tpl(T553, &[env.get("aws_bucket"), env.get("aws_region"), cookies.get("hash")]);
    let res = aws::fetch(&client, &url, "HEAD", &[], &[]).await?;
    ctx.response = V::ext(js::Js(res.clone()));
    let status = js::response_status(&res);

    if status == 200.0 {
        ctx.balance = V::Num(0.0);
    }

    if js::response_header(&res, "vapid")?.truthy() {
        cookies.s_("vapid", V::Bool(true))?;
    }

    if js::response_header(&res, "subscription")?.truthy() {
        cookies.s_("subscription", V::Bool(true))?;
    }

    ctx.email = js::response_header(&res, "email")?;
    ctx.phone = js::response_header(&res, "phone")?;
    ctx.address = js::response_header(&res, "address")?;
    ctx.token = js::response_header(&res, "token")?;

    // 578 `response.ok`
    if (200.0..=299.0).contains(&status) {
        head_object(ctx, &res)?;

        if ctx.headObject.get("email").truthy() {
            ctx.email = ctx.headObject.get("email");
        }
        if ctx.headObject.get("phone").truthy() {
            ctx.phone = ctx.headObject.get("phone");
        }
        if ctx.headObject.get("address").truthy() {
            ctx.address = ctx.headObject.get("address");
        }
        if ctx.headObject.get("token").truthy() {
            ctx.token = ctx.headObject.get("token");
        }
    }

    // 607 — 고침 (fix-session-token): 원본은 `==` 라 S3 에 token 이 없고(null) 요청에도 없으면(undefined) 같다고 봐서,
    // token 없는 hash 객체는 hash 만으로 로그인됐다 → S3 의 token 이 있어야 한다
    if ctx.email.truthy() && ctx.token.truthy() && loose_eq(&ctx.token, &cookies.get("token")) {
        cookies.s_("email", ctx.email.clone())?;

        if ctx.phone.truthy() {
            cookies.s_("phone", ctx.phone.clone())?;
        }

        cookies.s_("address", ctx.address.clone())?;
        cookies.s_("draft", V::new_obj())?;
        cookies.s_("pages", V::new_obj())?;

        // 621 — logisRegion 이 없으면 env["undefined"].prepare 로 TypeError
        let db = ctx.logisRegion.to_key();
        // 고침 (fix-sql-bind): 값은 묶는다
        let sql = r#"SELECT * FROM users WHERE "type" = 'user' AND "id" = ?1 AND "from" = ?1 LIMIT 1"#;
        ctx.results = js::query_bound(env, &db, sql, &[cookies.get("address")]).await?;

        ctx.teamId = V::from(hash_id(&cookies.get("address"))?);

        if ctx.results.g("length")?.truthy() {
            existing_user(ctx, env, &cookies, query, &db).await?;
        } else {
            new_user(ctx, env, &cookies, &db).await?;
        }

        // 870
        let r: R<()> = async {
            ctx.clientAddress = V::from(hash_id(&ctx.email)?);

            let client = s3(env);
            let url = tpl(T874, &[env.get("aws_bucket"), env.get("aws_region"), ctx.clientAddress.clone()]);
            let res = aws::fetch(&client, &url, "HEAD", &[], &[]).await?;
            ctx.response = V::ext(js::Js(res.clone()));
            let status = js::response_status(&res);

            // 원본 그대로: 머리글이 없으면 null * 1 = 0
            if status == 200.0 {
                ctx.balance = V::Num(js::response_header(&res, "balance")?.to_num());
                cookies.s_("balance", ctx.balance.clone())?;
            }

            if (200.0..=299.0).contains(&status) {
                head_object(ctx, &res)?;

                if ctx.headObject.get("balance").truthy() {
                    ctx.balance = V::Num(ctx.headObject.get("balance").to_num());
                    cookies.s_("balance", ctx.balance.clone())?;
                }
            }
            Ok(())
        }
        .await;
        // 899 catch(err){} — 삼킨다
        let _ = r;
    } else {
        ctx.headObject = V::Undef;
    }

    // 908
    headers_set(&ctx.headers, "Set-Cookie", &V::from(tpl(T908, &[cookies.get("hash")])))?;
    headers_append(&ctx.headers, "Set-Cookie", &V::from(tpl(T909, &[cookies.get("token")])))?;

    if cookies.get("team").truthy() {
        headers_append(&ctx.headers, "Set-Cookie", &V::from(tpl(T912, &[cookies.get("team")])))?;
    }
    Ok(())
}

/// 625~686 — 있는 사용자: 이름 · 팀 · 보낸이 갱신 후 users 에 덮어쓴다
async fn existing_user(ctx: &mut Ctx, env: &Env, cookies: &V, query: &V, db: &str) -> R<()> {
    ctx.user = ctx.results.gi(0)?;
    let user = ctx.user.clone();

    cookies.s_("team", user.g("to")?)?;

    ctx.decompressedJsonString = V::from(decode_utf8(&ungzip(&user.g("data")?)?));
    ctx.data = parse_str(&ctx.decompressedJsonString.to_str())?;
    let data = ctx.data.clone();

    cookies.s_("name", data.g("name")?)?;

    ctx.FLAG = data.g("flag")?;
    ctx.zoneRegion = data.g("region")?;

    let q_team = query.get("team");
    ctx.team = if q_team.truthy() { V::from(decode_uri_component(&q_team.to_str())?) } else { data.g("team")? };
    let q_name = query.get("name");
    ctx.name = if q_name.truthy() { V::from(decode_uri_component(&q_name.to_str())?) } else { data.g("name")? };
    let q_sender = query.get("sender");
    ctx.sender = if q_sender.truthy() { V::from(decode_uri_component(&q_sender.to_str())?) } else { data.g("sender")? };

    // 646 `data.name = cookies.name = name` — 오른쪽(cookies) 이 먼저 들어간다
    if ctx.name.truthy() {
        cookies.s_("name", ctx.name.clone())?;
        data.s_("name", ctx.name.clone())?;
    }

    if ctx.team.truthy() {
        data.s_("team", ctx.team.clone())?;
    }

    if ctx.sender.truthy() {
        data.s_("sender", ctx.sender.clone())?;
    }

    if data.g("sender")?.truthy() {
        cookies.s_("sender", data.g("sender")?)?;
    }

    user.s_("cc", cookies.get("cc"))?;

    // 666 — 고침 (fix-link-hoist): 원본의 `link` 는 996 에서야 대입되는 var 라 여기선 undefined ("…undefined" 를 해시) → 996 과 같은 값
    let link = crate::fixes::page_link(&ctx.href.to_str(), &ctx.origin.to_str());
    let ref_text = add(&add(&cookies.get("team"), &cookies.get("cc")), &V::from(link));
    user.s_("ref", V::from(hash_id(&ref_text)?))?;

    // 668 `gzip(…, { to: 'arraybuffer' }).buffer`
    ctx.user_arr = gzip_json(&data)?;
    user.s_("data", ctx.user_arr.clone())?;

    // 672
    let stmt = js::bind(
        &env.prepare(db, T672[0])?,
        &[
            user.g("id")?,
            user.g("type")?,
            user.g("from")?,
            user.g("to")?,
            user.g("cc")?,
            user.g("bcc")?,
            user.g("ref")?,
            user.g("data")?,
            user.g("created_at")?,
            ctx.now.clone(),
        ],
    )?;
    let out = js::run(&stmt).await?;
    ctx.results = out.g("results")?;
    ctx.success = out.g("success")?;
    ctx.error = out.g("error")?;
    Ok(())
}

/// 688~868 — 새 사용자: 지역(zone) 고르기, 사용자 · 팀 행, zones.pool 올리기
async fn new_user(ctx: &mut Ctx, env: &Env, cookies: &V, db: &str) -> R<()> {
    // 689 (email 은 참인 글)
    let email = cookies.g("email")?;
    let Some(email_str) = email.as_str() else {
        return Err(Throw::type_error("cookies.email.split is not a function"));
    };
    ctx.name = V::from(email_str.split('@').next().unwrap_or("").to_string());

    // 691 (logisRegion 은 앞 질의가 성공했으니 글)
    let Some(logis_region) = ctx.logisRegion.as_str() else {
        return Err(Throw::type_error("logisRegion.split is not a function"));
    };
    ctx.domains = V::from(logis_region.split('_').map(V::from).collect::<Vec<_>>());
    ctx.regionCode = ctx.domains.gi(2)?;

    // 고침 (fix-app-invite): 초대받은 이메일이면 새 팀 대신 초대한 팀에 들어간다
    if let Some(invite) = find_invite(env, db, &email_str).await {
        return join_invite(ctx, env, cookies, db, &invite).await;
    }

    // 696
    ctx.results = js::query(env, CENTER_REGION, T696[0]).await?;

    // 700 `var pool;` 는 값을 지우지 않는다

    console_log!("regionCode", ctx.regionCode);

    // 705
    ctx.z = V::Num(0.0);
    while less_than(&ctx.z, &ctx.results.g("length")?) == Some(true) {
        ctx.zone = ctx.results.gk(&ctx.z)?;

        let r: R<()> = (|| {
            // `zone.id.split(/\d+/g)[0]` — 첫 숫자 앞까지
            let id = ctx.zone.g("id")?;
            let code = match &id {
                V::Str(s) => s.chars().take_while(|c| !c.is_ascii_digit()).collect::<String>(),
                V::Undef | V::Null => return Err(Throw::type_error(format!("Cannot read properties of {} (reading 'split')", id.to_str()))),
                _ => return Err(Throw::type_error("zone.id.split is not a function")),
            };
            ctx.zoneCode = V::from(code);

            if ctx.pool.is_undef()
                && loose_eq(&ctx.regionCode, &ctx.zoneCode)
                && less_than(&ctx.zone.g("pool")?, &V::Num(1000.0)) == Some(true)
            {
                ctx.zoneRegion = ctx.zone.g("id")?;
                ctx.pool = ctx.zone.g("pool")?;
            }
            Ok(())
        })();
        if let Err(err) = r {
            console_log!("zoneRegion err", err);
        }

        ctx.z = V::Num(ctx.z.to_num() + 1.0);
    }

    // 고침 (fix-zone-pool): 자리가 남은 zone 이 없으면 사용자를 만들지 않는다 (원본은 region '' 로 만들어 commerce_logis__items 를 찾다
    // 영영 깨졌다). 던지면 세션이 오류 쿠키로 바뀌어 로그인하지 않은 채로 남는다
    if ctx.pool.is_undef() {
        console_log!("no zone with room", ctx.regionCode);
        return Err(Throw::error(format!("no zone with room for {}", ctx.regionCode.to_str())));
    }

    // 725
    ctx.user_arr = gzip_json(&obj! {
        "flag" => ctx.FLAG.clone(),
        "name" => ctx.name.clone(),
        "title" => "",
        "region" => ctx.zoneRegion.clone(),
        "page_count" => 0,
        "favicon" => V::Null,
    })?;

    ctx.user = obj! {
        "id" => cookies.get("address"),
        "type" => "user",
        "from" => cookies.get("address"),
        "to" => if cookies.get("team").truthy() { cookies.get("team") } else { ctx.teamId.clone() },
        "cc" => "",
        "bcc" => "",
        "ref" => "",
        "data" => ctx.user_arr.clone(),
        "created_at" => ctx.now.clone(),
        "updated_at" => ctx.now.clone(),
    };

    // 748
    let counter = || obj! { "draft" => 0, "count" => 0 };
    ctx.base = obj! {
        "pages" => V::new_obj(),
        "goods" => counter(),
        "order" => counter(),
        "event" => counter(),
        "coupon" => counter(),
        "tracking" => counter(),
        "search" => counter(),
        "review" => counter(),
        "member" => counter(),
    };

    ctx.propertys = V::from(
        [
            "price", "quantity", "width", "height", "length", "weight", "shipping_fee", "shipping_duration", "sale_price", "supply_price",
            "low_stock_threshold", "discount", "min_order_amount", "max_discount_amount", "usage_limit", "usage_per", "started_at", "expired_at",
        ]
        .iter()
        .map(|s| V::from(*s))
        .collect::<Vec<_>>(),
    );

    // 786
    ctx.p = V::Num(0.0);
    while less_than(&ctx.p, &ctx.propertys.g("length")?) == Some(true) {
        ctx.property = ctx.propertys.gk(&ctx.p)?;

        for table in ctx.base.keys() {
            if ctx.base.has(&table) && table != "pages" {
                ctx.base.g(&table)?.sk(&ctx.property, obj! { "max" => 0, "min" => 0 })?;
            }
        }

        ctx.p = V::Num(ctx.p.to_num() + 1.0);
    }

    // 801
    ctx.team_arr = gzip_json(&obj! {
        "flag" => ctx.FLAG.clone(),
        "name" => add(&ctx.name, &V::from("'s team")),
        "title" => "",
        "region" => ctx.zoneRegion.clone(),
        "page_count" => 0,
        "favicon" => V::Null,
        "base" => ctx.base.clone(),
    })?;

    ctx.team = obj! {
        "id" => ctx.teamId.clone(),
        "type" => "team",
        "from" => cookies.get("address"),
        "to" => ctx.teamId.clone(),
        "cc" => "",
        "bcc" => "",
        "ref" => "",
        "data" => ctx.team_arr.clone(),
        "created_at" => ctx.now.clone(),
        "updated_at" => ctx.now.clone(),
    };

    cookies.s_("name", ctx.name.clone())?;
    cookies.s_("team", ctx.team.g("id")?)?;

    // 833
    ctx.statements = V::new_arr();

    let fields = |row: &V| -> R<Vec<V>> {
        Ok(vec![
            row.g("id")?,
            row.g("type")?,
            row.g("from")?,
            row.g("to")?,
            row.g("cc")?,
            row.g("bcc")?,
            row.g("ref")?,
            row.g("data")?,
            row.g("created_at")?,
            row.g("updated_at")?,
        ])
    };

    let team_stmt = js::bind(&env.prepare(db, T836[0])?, &fields(&ctx.team)?)?;
    ctx.statements.push(team_stmt)?;

    let user_stmt = js::bind(&env.prepare(db, T845[0])?, &fields(&ctx.user)?)?;
    ctx.statements.push(user_stmt)?;

    // 853 — catch(err){} 로 삼킨다
    let r: R<()> = async {
        env.batch(db, &ctx.statements).await?;

        // 맞는 zone 이 없으면 위(fix-zone-pool)에서 이미 던졌다
        let stmt = js::bind(&env.prepare(CENTER_REGION, T856[0])?, &[ctx.zoneRegion.clone(), add(&ctx.pool, &V::Num(1.0))])?;
        js::run(&stmt).await?;
        Ok(())
    }
    .await;
    let _ = r;
    Ok(())
}

/// fix-app-invite: `SELECT * FROM users WHERE "type" = 'invite' AND "cc" = ?1 ORDER BY created_at DESC LIMIT 1` (cc = hashId(소문자 이메일)).
/// data 를 풀어 zone 과 팀(to)이 있어야 초대로 본다. 던지면 원본 패치처럼 없는 것으로 ('invite err')
async fn find_invite(env: &Env, db: &str, email: &str) -> Option<V> {
    let r: R<Option<V>> = async {
        let cc = hash_id(&V::from(to_lower(email)))?;
        let sql = r#"SELECT * FROM users WHERE "type" = 'invite' AND "cc" = ?1 ORDER BY created_at DESC LIMIT 1"#;
        let results = js::query_bound(env, db, sql, &[V::from(cc)]).await?;
        if !results.g("length")?.truthy() {
            return Ok(None);
        }
        let invite = results.gi(0)?;
        let data = parse_str(&decode_utf8(&ungzip(&invite.g("data")?)?))?;
        invite.s_("data", data.clone())?;
        if !data.truthy() || !data.g("zone")?.truthy() || !invite.g("to")?.truthy() {
            return Ok(None);
        }
        Ok(Some(invite))
    }
    .await;
    match r {
        Ok(v) => v,
        Err(err) => {
            console_log!("invite err", err);
            None
        }
    }
}

/// fix-app-invite: 초대한 팀에 들어간다 — 사용자 행(to = 초대의 팀, region = 초대의 zone), 초대한 페이지마다 멤버 행, 초대 행 지우기.
/// 팀 행 · zones.pool 은 건드리지 않는다. 쓰기가 던지면 삼킨다('invite join err' — 다음 로그인에서 다시 한다)
async fn join_invite(ctx: &mut Ctx, env: &Env, cookies: &V, db: &str, invite: &V) -> R<()> {
    ctx.zoneRegion = invite.g("data")?.g("zone")?;

    ctx.user_arr = gzip_json(&obj! {
        "flag" => ctx.FLAG.clone(),
        "name" => ctx.name.clone(),
        "title" => "",
        "region" => ctx.zoneRegion.clone(),
        "page_count" => 0,
        "favicon" => V::Null,
    })?;

    ctx.user = obj! {
        "id" => cookies.get("address"),
        "type" => "user",
        "from" => cookies.get("address"),
        "to" => invite.g("to")?,
        "cc" => "",
        "bcc" => "",
        "ref" => "",
        "data" => ctx.user_arr.clone(),
        "created_at" => ctx.now.clone(),
        "updated_at" => ctx.now.clone(),
    };

    cookies.s_("name", ctx.name.clone())?;
    cookies.s_("team", invite.g("to")?)?;

    let r: R<()> = async {
        let refs = crate::fixes::address_list(&invite.g("data")?.g("refs")?, 50);
        let (statements, _rows) = crate::fixes::member_statements(
            env,
            &ctx.zoneRegion,
            &cookies.get("address"),
            &invite.g("to")?,
            &invite.g("from")?,
            &refs,
            &ctx.user_arr,
            &ctx.now,
        )
        .await?;
        if statements.g("length")?.truthy() {
            env.batch(&format!("commerce_logis_{}_items", ctx.zoneRegion.to_str()), &statements).await?;
        }

        let user = ctx.user.clone();
        let fields = [
            user.g("id")?,
            user.g("type")?,
            user.g("from")?,
            user.g("to")?,
            user.g("cc")?,
            user.g("bcc")?,
            user.g("ref")?,
            user.g("data")?,
            user.g("created_at")?,
            user.g("updated_at")?,
        ];
        let list = V::new_arr();
        list.push(js::bind(&env.prepare(db, T845[0])?, &fields)?)?;
        list.push(js::bind(&env.prepare(db, r#"DELETE FROM users WHERE "id" = ?1 AND "type" = 'invite'"#)?, &[invite.g("id")?])?)?;
        env.batch(db, &list).await?;
        Ok(())
    }
    .await;
    if let Err(err) = r {
        console_log!("invite join err", err);
    }
    Ok(())
}

//! 원본 966~1593 (try 안): CORS 머리글 · created_at · URL · referrer · 페이지 / 아이템 판별(center · zone DB) · `?crons=` ·
//! 쓰이지 않는 페이지 나무(1474~1558) · 멤버 판별. 이 구역에는 `return new Response(...)` 가 없어 늘 [`Step::Next`] 이고,
//! 던지면 처리기가 'err' 로그 뒤 418 로 끝낸다(원본과 같은 경계).

use logis_worker_common::gzip::{decode_utf8, gzip_json, ungzip};
use logis_worker_common::js::{self, Env};
use logis_worker_common::jsv::{js_trim_start, less_than, to_lower, to_upper, tpl, Throw};
use logis_worker_common::tables::CENTER_REGION;
use logis_worker_common::util::hash_id;
use logis_worker_common::web::{headers_set, Req};
use logis_worker_common::{add, console_log, eth, loose_eq, obj, parse_str, stringify_str, R, V};

use crate::ctx::Ctx;
use crate::flow::Step;
use crate::templates::*;

/// `ethers.ZeroAddress`
const ZERO_ADDRESS: &str = "0x0000000000000000000000000000000000000000";

/// `x ? x : ethers.ZeroAddress`
fn or_zero(v: V) -> V {
    if v.truthy() {
        v
    } else {
        V::from(ZERO_ADDRESS)
    }
}

fn invalid_url() -> Throw {
    Throw::type_error("Invalid URL string.")
}

/// `new URL(s)`
fn new_url(s: &str) -> R<url::Url> {
    url::Url::parse(s).map_err(|_| invalid_url())
}

/// `new URL(s, base)` — base 가 먼저 읽히고, 틀리면 던진다
fn new_url_base(s: &str, base: &str) -> R<url::Url> {
    let base = url::Url::parse(base).map_err(|_| invalid_url())?;
    base.join(s).map_err(|_| invalid_url())
}

/// `new URL(...)` 의 쓰는 속성만 담은 값 (session.rs 의 것과 같은 꼴 — 그쪽은 비공개라 여기 따로 둔다)
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

/// `s.toLowerCase()` — 문자열이 아니면 원본처럼 TypeError
fn lower(v: &V, what: &str) -> R<String> {
    match v {
        V::Str(s) => Ok(to_lower(s)),
        V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'toLowerCase')", v.to_str()))),
        _ => Err(Throw::type_error(format!("{what}.toLowerCase is not a function"))),
    }
}

/// `s.toUpperCase()` — 문자열이 아니면 원본처럼 TypeError
fn upper(v: &V, what: &str) -> R<String> {
    match v {
        V::Str(s) => Ok(to_upper(s)),
        V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'toUpperCase')", v.to_str()))),
        _ => Err(Throw::type_error(format!("{what}.toUpperCase is not a function"))),
    }
}

/// `parseInt(s)` (기수 없음): 앞 공백 · 부호, `0x`/`0X` 면 16진, 아니면 10진 — 숫자가 끊기는 곳까지, 없으면 NaN.
/// (공용 `jsv::parse_int` 는 `0x` 를 보지 않는다 — `?created_at=0x10` 은 원본에서 16)
fn parse_int(s: &str) -> f64 {
    let t = js_trim_start(s);
    let (neg, rest) = match t.chars().next() {
        Some('-') => (true, &t[1..]),
        Some('+') => (false, &t[1..]),
        _ => (false, t),
    };
    let (radix, body) = if rest.starts_with("0x") || rest.starts_with("0X") { (16, &rest[2..]) } else { (10, rest) };
    let digits: String = body.chars().take_while(|c| c.is_digit(radix)).collect();
    if digits.is_empty() {
        return f64::NAN;
    }
    let v = if radix == 10 {
        digits.parse::<f64>().unwrap_or(f64::NAN)
    } else {
        digits.chars().fold(0.0, |acc, c| acc * 16.0 + c.to_digit(16).unwrap_or(0) as f64)
    };
    if neg {
        -v
    } else {
        v
    }
}

/// 원본 `safeClone(obj)` (41) — 공용 `deep_clone` 과 달리 Uint8Array(`gzip` 결과)를 첨자 키 객체 `{"0":31,"1":139,…}` 로 바꾼다
/// (`typeof` 가 object 이고 배열이 아니라 `{}` 에 for-in 의 첨자 키를 옮긴다). 이미 본 객체는 null.
fn safe_clone(v: &V) -> V {
    fn go(v: &V, seen: &mut Vec<*const ()>) -> V {
        let ptr = match v {
            V::Arr(a) => std::rc::Rc::as_ptr(a) as *const (),
            V::Obj(o) => std::rc::Rc::as_ptr(o) as *const (),
            V::Bytes(b) => std::rc::Rc::as_ptr(b) as *const (),
            V::Ext(e) => std::rc::Rc::as_ptr(e) as *const (),
            other => return other.clone(),
        };
        if seen.contains(&ptr) {
            return V::Null;
        }
        seen.push(ptr);
        match v {
            V::Arr(_) => V::from(v.items().iter().map(|x| go(x, seen)).collect::<Vec<_>>()),
            V::Bytes(b) => {
                let out = V::new_obj();
                for (i, x) in b.borrow().iter().enumerate() {
                    out.put(&i.to_string(), V::Num(*x as f64));
                }
                out
            }
            // 원본 그대로: 바인딩 객체(Ext)는 열거되는 속성이 없어 `{}` (pages 에는 오지 않는다)
            V::Ext(_) => V::new_obj(),
            _ => {
                let out = V::new_obj();
                for (k, x) in v.entries() {
                    out.put(&k, go(&x, seen));
                }
                out
            }
        }
    }
    go(v, &mut Vec::new())
}

/// `{ ...x, children: c }` — 펼친 뒤 `children` 을 덮는다(이미 있으면 그 자리)
fn spread_children(x: &V, children: V) -> V {
    let out = V::new_obj();
    for (k, v) in x.entries() {
        out.put(&k, v);
    }
    out.put("children", children);
    out
}

/// 1174~1183 · 1195~1204 · 1287~1296 · 1309~1318 의 세 갈래 (`_page.type` → `type`, 맞는 갈래가 없으면 그대로)
fn page_kind(t: &V) -> Option<&'static str> {
    let is = |s: &str| loose_eq(t, &V::from(s));
    if is("tracking") {
        Some("tracking")
    } else if is("goods") || is("order") {
        Some("sales")
    } else if is("event") || is("coupon") {
        Some("event")
    } else {
        None
    }
}

fn set_kind(ctx: &mut Ctx) -> R<()> {
    if let Some(k) = page_kind(&ctx._page.g("type")?) {
        ctx.type_ = V::from(k);
    }
    Ok(())
}

/// 966~1593
pub async fn run(ctx: &mut Ctx, env: &Env, req: &Req) -> R<Step> {
    // 968
    headers_set(&ctx.headers, "Access-Control-Allow-Origin", &V::from("*"))?;
    // 고침 (fix-cors-methods): 서버가 처리하는 PUT · DELETE 도 (원본은 GET, POST, OPTIONS)
    headers_set(&ctx.headers, "Access-Control-Allow-Methods", &V::from("GET, POST, PUT, DELETE, OPTIONS"))?;
    headers_set(&ctx.headers, "Access-Control-Allow-Headers", &V::from("Content-Type, Content-Encoding, Authorization, X-Requested-With"))?;

    headers_set(&ctx.headers, "Content-Security-Policy", &V::from(tpl(T972, &[ctx.origin.clone()])))?;

    // 977
    console_log!("request.method", req.method.as_str());
    console_log!("contentEncoding", ctx.contentEncoding);

    // 980 — 원본 그대로: `isNaN` 은 Number(), 값은 parseInt 라 `?created_at=` (빈 글)이면 NaN 이 SQL 에 들어간다
    ctx.created_at = V::Num(0.0);

    let q_created_at = ctx.req.g("query")?.g("created_at")?;
    if !q_created_at.to_num().is_nan() {
        ctx.created_at = V::Num(parse_int(&q_created_at.to_str()));
    }

    // 987 `var url;` 는 값을 지우지 않는다. 두 번째 `new URL` 은 try 밖이라 던지면 418
    let href = ctx.href.to_str();
    let u = match new_url(&href) {
        Ok(u) => u,
        Err(_) => {
            let base = if ctx.origin.truthy() { ctx.origin.to_str() } else { "https://commerce.logis.center".to_string() };
            new_url_base(&href, &base)?
        }
    };
    ctx.url = url_obj(&u);

    ctx.pathname = V::from(to_lower(&ctx.url.g("pathname")?.to_str()));

    ctx.link = V::from(to_lower(&add(&ctx.url.g("pathname")?, &ctx.url.g("search")?).to_str()));

    // 999 `var referrer` — 선언만(값을 지우지 않음)

    // 1001
    let r: R<()> = async {
        let q_referrer = ctx.req.g("query")?.g("referrer")?;
        if q_referrer.truthy() {
            ctx._url = url_obj(&new_url(&q_referrer.to_str())?);

            let id = hash_id(&add(&ctx.cookies.g("cc")?, &V::from(to_lower(&ctx._url.g("pathname")?.to_str()))))?;
            let sql = tpl(T1005, &[V::from(id), ctx.created_at.clone()]);
            ctx.results = js::query(env, CENTER_REGION, &sql).await?;

            if ctx.results.g("length")?.truthy() {
                ctx.referrer = ctx.results.gi(0)?;
                ctx.referrer.s_("table", V::from("pages"))?;
            }
        }
        Ok(())
    }
    .await;
    if let Err(err) = r {
        console_log!("req.query.referrer err", err);
    }

    // 1020
    let q_id = ctx.req.g("query")?.g("id")?;
    ctx.pageId = if eth::is_address(&q_id) {
        V::from(lower(&q_id, "req.query.id")?)
    } else {
        V::from(hash_id(&add(&ctx.cookies.g("cc")?, &ctx.pathname))?)
    };

    ctx.itemId = V::from(hash_id(&add(&add(&ctx.cookies.g("team")?, &ctx.cookies.g("cc")?), &ctx.link))?);

    // 1026
    ctx.homeCc = V::from(hash_id(&V::from("logis.center"))?);

    ctx.homeId = V::from(hash_id(&V::from(to_upper(&ctx.homeCc.to_str()) + "/tracking"))?);

    ctx.homeBcc = V::from(hash_id(&V::from("tracking".to_string() + &to_upper(&ctx.homeCc.to_str())))?);

    ctx.homeRef = V::from(hash_id(&add(&add(&ctx.cookies.g("team")?, &ctx.homeCc), &V::from("/tracking")))?);

    // 1035
    ctx.home = obj! {
        "id" => ctx.homeId.clone(),
        "table" => "pages",
        "type" => "tracking",
        "from" => or_zero(ctx.cookies.g("address")?),
        "to" => or_zero(ctx.cookies.g("team")?),
        "cc" => ctx.homeCc.clone(),
        "bcc" => ctx.homeBcc.clone(),
        "ref" => ctx.homeRef.clone(),
        "data" => obj! {
            "type" => "tracking",
            "origin" => "https://logis.center",
            "link" => "/tracking",
            "item" => true,
            "node" => true,
        },
        "created_at" => ctx.now.clone(),
        "updated_at" => ctx.now.clone(),
    };

    // 1055 — pako 의 Uint8Array
    let home_data = gzip_json(&ctx.home.g("data")?)?;
    ctx.home.s_("data", home_data)?;

    // 1060
    ctx.tasks = V::new_arr();

    ctx.items = V::new_arr();

    ctx.hints = V::new_arr();

    ctx.detail = V::new_obj();

    ctx.pages = V::from(vec![ctx.home.clone()]);

    ctx.page = if loose_eq(&ctx.homeId, &ctx.pageId) { ctx.home.clone() } else { V::new_obj() };

    ctx.isDetail = V::Bool(false);

    ctx.temp = V::new_obj();

    ctx.type_ = V::from(if loose_eq(&ctx.homeId, &ctx.pageId) { "tracking" } else { "" });

    // 1082
    ctx.member = V::Bool(false);

    ctx.users = V::new_arr();

    ctx.admin = V::new_obj();

    // 1088 — logisRegion 이 없으면 env["undefined"].prepare 로 TypeError
    {
        let db = ctx.logisRegion.to_key();
        // 고침 (fix-sql-bind): 값은 묶는다 (수인 created_at 만 글에)
        let sql = format!(r#"SELECT * FROM users WHERE "to" = ?1 AND "created_at" < {} ORDER BY created_at DESC LIMIT 10"#, ctx.created_at.to_str());
        let stmt = env.prepare(&db, &sql)?;
        let out = js::all(&js::bind(&stmt, &[ctx.cookies.g("team")?])?).await?;
        ctx.results = out.g("results")?;
        ctx.success = out.g("success")?;
        ctx.error = out.g("error")?;
    }

    if ctx.results.g("length")?.truthy() {
        // 원본 그대로: `var user` · `var data` 라 세션의 user · data 를 이 행들로 덮는다
        ctx.i = V::Num(0.0);
        while less_than(&ctx.i, &ctx.results.g("length")?) == Some(true) {
            ctx.user = ctx.results.gk(&ctx.i)?;

            ctx.user.s_("table", V::from("users"))?;

            ctx.decompressedJsonString = V::from(decode_utf8(&ungzip(&ctx.user.g("data")?)?));

            ctx.data = parse_str(&ctx.decompressedJsonString.to_str())?;

            if ctx.data.g("name")?.truthy() {
                ctx.user.s_("name", ctx.data.g("name")?)?;
            }

            ctx.user.del("data")?;

            if loose_eq(&ctx.user.g("id")?, &ctx.user.g("to")?) {
                ctx.admin = ctx.user.clone();

                ctx.member = V::Bool(true);

                ctx.cookies.s_("pages", ctx.data.g("base")?.g("pages")?)?;

                ctx.cookies.s_("goods", ctx.data.g("base")?.g("goods")?.g("count")?)?;
                ctx.cookies.s_("order", ctx.data.g("base")?.g("order")?.g("count")?)?;
                ctx.cookies.s_("tracking", ctx.data.g("base")?.g("tracking")?.g("count")?)?;
                ctx.cookies.s_("event", ctx.data.g("base")?.g("event")?.g("count")?)?;
                // 고침 (fix-coupon-count): 원본은 event.coupon (없는 칸) — 팀 base 의 coupon 은 event 와 나란한 { draft, count }. 옛 팀은 0
                let coupon = ctx.data.g("base")?.g("coupon")?;
                ctx.cookies.s_("coupon", if coupon.truthy() { coupon.g("count")? } else { V::Num(0.0) })?;
            }

            ctx.users.push(ctx.user.clone())?;

            ctx.i = V::Num(ctx.i.to_num() + 1.0);
        }
    }

    // 1128
    ctx.cookies.s_("type", ctx.req.g("query")?.g("type")?)?;

    if ctx.cookies.g("address")?.truthy() {
        signed_in(ctx, env).await?;
    }

    // 1474 — 만든 tree 는 어디서도 읽지 않는다. 던지면 조용히 삼킨다(catch(err){})
    if ctx.pages.g("length")?.truthy() {
        let r = tree(ctx);
        let _ = r;
    }

    // 1561
    ctx.cookies.s_("bcc", V::from(""))?;

    if !ctx.page.keys_checked()?.is_empty() {
        let page_type = ctx.page.g("type")?;
        let cc = if ctx.isDetail.truthy() { V::from(upper(&ctx.cookies.g("cc")?, "cookies.cc")?) } else { ctx.cookies.g("cc")? };
        ctx.cookies.s_("bcc", V::from(hash_id(&add(&page_type, &cc))?))?;

        if !ctx.member.truthy() {
            let db = tpl(T1567, &[ctx.zoneRegion.clone()]);
            // 고침 (fix-sql-bind)
            let sql = r#"SELECT * FROM items WHERE "type" = 'member' AND "from" = ?1 AND "to" = ?2 AND "bcc" = ?3 LIMIT 1"#;
            ctx.results = js::query_bound(env, &db, sql, &[ctx.cookies.g("address")?, ctx.cookies.g("team")?, ctx.cookies.g("bcc")?]).await?;

            if ctx.results.g("length")?.truthy() {
                ctx.member = V::Bool(true);
            } else {
                let db = tpl(T1572, &[ctx.zoneRegion.clone()]);
                let sql = r#"SELECT * FROM items WHERE "type" = 'member' AND "from" = ?1 AND "to" = ?2 AND "bcc" = ?3 LIMIT 1"#;
                ctx.results = js::query_bound(env, &db, sql, &[ctx.cookies.g("address")?, ctx.cookies.g("team")?, ctx.cookies.g("cc")?]).await?;

                if ctx.results.g("length")?.truthy() {
                    ctx.member = V::Bool(true);
                }
            }
        }
    }

    // 1582
    ctx.cookies.s_("href", ctx.href.clone())?;

    console_log!("pageId", ctx.pageId);

    Ok(Step::Next)
}

/// 1130~1471 `if(cookies.address){ … }` — 아이템(detail) · 팀의 페이지 · cc 의 페이지 · center 의 페이지 · `?crons=`
async fn signed_in(ctx: &mut Ctx, env: &Env) -> R<()> {
    // 1132 — 원본 그대로: zoneRegion 이 '' 면 env["commerce_logis__items"]
    let items_db = tpl(T1132, &[ctx.zoneRegion.clone()]);
    // 고침 (fix-sql-bind)
    let sql = r#"SELECT * FROM items WHERE "to" = ?1 AND "ref" = ?2 LIMIT 1"#;
    ctx.results = js::query_bound(env, &items_db, sql, &[ctx.cookies.g("team")?, ctx.itemId.clone()]).await?;

    if ctx.results.g("length")?.truthy() {
        ctx.detail = ctx.results.gi(0)?;

        ctx.items.push(ctx.detail.clone())?;
    }

    console_log!("detail", stringify_str(&ctx.detail));

    // 1145
    let items_db = tpl(T1145, &[ctx.zoneRegion.clone()]);
    // 고침 (fix-sql-bind)
    let sql = r#"SELECT * FROM items WHERE "type" = 'pages' AND "to" = ?1 LIMIT 100"#;
    ctx.results = js::query_bound(env, &items_db, sql, &[ctx.cookies.g("team")?]).await?;

    console_log!("page results.length", ctx.results.g("length")?);

    if ctx.results.g("length")?.truthy() {
        ctx.i = V::Num(0.0);
        while less_than(&ctx.i, &ctx.results.g("length")?) == Some(true) {
            ctx._page = ctx.results.gk(&ctx.i)?;

            ctx.decompressedJsonString = V::from(decode_utf8(&ungzip(&ctx._page.g("data")?)?));

            ctx.page_data = parse_str(&ctx.decompressedJsonString.to_str())?;

            console_log!("page_data.type", ctx.page_data.g("type")?);

            ctx._page.s_("type", ctx.page_data.g("type")?)?;
            ctx._page.s_("from", or_zero(ctx.cookies.g("address")?))?;
            ctx._page.s_("to", or_zero(ctx.cookies.g("team")?))?;

            ctx._page.s_("table", V::from("pages"))?;

            ctx.temp.sk(&ctx._page.g("id")?, V::Bool(true))?;

            // 원본 그대로: `==` 라 detail 이 {} (bcc undefined) 이고 행의 bcc 가 NULL 이면 같다고 본다
            if loose_eq(&ctx.detail.g("bcc")?, &ctx._page.g("bcc")?) {
                ctx._page.s_("current", V::Bool(true))?;
                ctx.page = ctx._page.clone();

                set_kind(ctx)?;

                ctx.pageId = ctx.page.g("id")?;

                ctx.isDetail = V::Bool(true);
            }

            if ctx.detail.keys_checked()?.is_empty() && loose_eq(&ctx.pageId, &ctx._page.g("id")?) {
                ctx._page.s_("current", V::Bool(true))?;
                ctx.page = ctx._page.clone();

                set_kind(ctx)?;

                if !ctx.page_data.keys_checked()?.is_empty() {
                    ctx.isDetail = ctx.page_data.g("detail")?;
                }
            }

            ctx.pages.push(ctx._page.clone())?;

            ctx.i = V::Num(ctx.i.to_num() + 1.0);
        }
    }

    // 1225~1252 는 주석

    console_log!("Object.keys(page).length", ctx.page.keys_checked()?.len());

    // 1256
    if ctx.page.keys_checked()?.is_empty() {
        let items_db = tpl(T1257, &[ctx.zoneRegion.clone()]);
        // 고침 (fix-sql-bind)
        let sql = format!(r#"SELECT * FROM items WHERE "type" = 'pages' AND "cc" = ?1 AND "created_at" < {} LIMIT 100"#, ctx.created_at.to_str());
        ctx.results = js::query_bound(env, &items_db, &sql, &[ctx.cookies.g("cc")?]).await?;

        if ctx.results.g("length")?.truthy() {
            ctx.statements = V::new_arr();

            ctx.i = V::Num(0.0);
            while less_than(&ctx.i, &ctx.results.g("length")?) == Some(true) {
                ctx._page = ctx.results.gk(&ctx.i)?;

                if ctx.temp.gk(&ctx._page.g("id")?)?.truthy() {
                    // for 의 i++ 는 continue 에서도 돈다
                    ctx.i = V::Num(ctx.i.to_num() + 1.0);
                    continue;
                }

                ctx.temp.sk(&ctx._page.g("id")?, V::Bool(true))?;

                ctx.decompressedJsonString = V::from(decode_utf8(&ungzip(&ctx._page.g("data")?)?));

                ctx.page_data = parse_str(&ctx.decompressedJsonString.to_str())?;

                ctx.bool = V::Bool(loose_eq(&ctx.cookies.g("team")?, &ctx._page.g("to")?));

                ctx._page.s_("type", ctx.page_data.g("type")?)?;
                ctx._page.s_("from", or_zero(ctx.cookies.g("address")?))?;
                ctx._page.s_("to", or_zero(ctx.cookies.g("team")?))?;

                ctx._page.s_("table", V::from("pages"))?;

                if loose_eq(&ctx.detail.g("bcc")?, &ctx._page.g("bcc")?) {
                    ctx._page.s_("current", V::Bool(true))?;
                    ctx.page = ctx._page.clone();

                    set_kind(ctx)?;

                    ctx.pageId = ctx.page.g("id")?;

                    ctx.isDetail = V::Bool(true);
                }

                if ctx.detail.keys_checked()?.is_empty() && loose_eq(&ctx.pageId, &ctx._page.g("id")?) {
                    ctx._page.s_("current", V::Bool(true))?;
                    ctx.page = ctx._page.clone();

                    set_kind(ctx)?;

                    // 원본 그대로: page_data 가 아니라 _page 의 키를 센다(늘 참)
                    if !ctx._page.keys_checked()?.is_empty() {
                        ctx.isDetail = ctx.page_data.g("detail")?;
                    }
                }

                // 1326 — pako 의 Uint8Array (`.buffer` 도 같은 바이트)
                ctx.arr = gzip_json(&ctx.page_data)?;

                if !ctx.bool.truthy() {
                    let db = tpl(T1331, &[ctx.zoneRegion.clone()]);
                    let stmt = env.prepare(&db, T1331_1[0])?;
                    let stmt = js::bind(
                        &stmt,
                        &[
                            ctx._page.g("id")?,
                            V::from("pages"),
                            ctx._page.g("from")?,
                            ctx._page.g("to")?,
                            ctx._page.g("cc")?,
                            ctx._page.g("bcc")?,
                            ctx._page.g("ref")?,
                            ctx._page.g("digest")?,
                            ctx.arr.clone(),
                            ctx.now.clone(),
                            ctx.now.clone(),
                        ],
                    )?;
                    ctx.statements.push(stmt)?;
                }

                ctx._page.s_("data", ctx.arr.clone())?;

                ctx.pages.push(ctx._page.clone())?;

                ctx.i = V::Num(ctx.i.to_num() + 1.0);
            }

            if ctx.statements.g("length")?.truthy() {
                let db = tpl(T1371, &[ctx.zoneRegion.clone()]);
                env.batch(&db, &ctx.statements).await?;
            }
        } else {
            // 1374
            let sql = tpl(T1374, &[ctx.pageId.clone(), ctx.created_at.clone()]);
            ctx.results = js::query(env, CENTER_REGION, &sql).await?;

            if ctx.results.g("length")?.truthy() {
                ctx.page = ctx.results.gi(0)?;

                ctx.decompressedJsonString = V::from(decode_utf8(&ungzip(&ctx.page.g("data")?)?));

                ctx.page_data = parse_str(&ctx.decompressedJsonString.to_str())?;

                ctx.page.s_("current", V::Bool(true))?;

                ctx.page.s_("from", or_zero(ctx.cookies.g("address")?))?;
                ctx.page.s_("to", or_zero(ctx.cookies.g("team")?))?;
                ctx.page.s_("table", V::from("pages"))?;

                // 원본 그대로: 여기는 tracking 갈래가 없다
                let t = ctx.page.g("type")?;
                if loose_eq(&t, &V::from("goods")) || loose_eq(&t, &V::from("order")) {
                    ctx.type_ = V::from("sales");
                } else if loose_eq(&t, &V::from("event")) || loose_eq(&t, &V::from("coupon")) {
                    ctx.type_ = V::from("event");
                }

                // 원본 그대로: temp 에 true 가 아니라 page 객체를 넣는다
                ctx.temp.sk(&ctx.page.g("id")?, ctx.page.clone())?;

                ctx.pages.push(ctx.page.clone())?;

                if !ctx.page.keys_checked()?.is_empty() {
                    ctx.isDetail = ctx.page_data.g("detail")?;
                }
            }
        }
    }

    // 1414 — 고침 (fix-crons-query): 원본은 글 JSON 이면 글자마다 돌았고, 주소가 없으면 "WHERE  ORDER BY" 로 던졌고,
    // 지울 자리를 crons 의 자리(tasks 의 자리가 아님)로 골라 지울 때마다 뒤 자리가 당겨졌다 → 배열의 주소만(겹침 없이), 묶은 IN 질의, id 로 뺀다
    if ctx.req.g("query")?.g("crons")?.truthy() {
        let r: R<()> = async {
            ctx.crons = parse_str(&ctx.req.g("query")?.g("crons")?.to_str())?;

            if ctx.crons.is_arr() && ctx.crons.g("length")?.truthy() {
                let cron_ids = crate::fixes::address_list(&ctx.crons, 98);

                ctx.tasks = V::new_arr();

                for id in &cron_ids {
                    ctx.tasks.push(obj! {
                        "id" => id.as_str(),
                        "table" => "tasks",
                        "type" => "",
                        "from" => "",
                        "to" => "",
                        "cc" => "",
                        "bcc" => "",
                        "ref" => "",
                        "data" => V::Null,
                        "created_at" => ctx.now.clone(),
                        "updated_at" => ctx.now.clone(),
                    })?;
                }

                if !cron_ids.is_empty() {
                    let marks: Vec<String> = (0..cron_ids.len()).map(|n| format!("?{}", n + 3)).collect();
                    let sql = format!(r#"SELECT "id" FROM tasks WHERE "from" = ?1 AND "created_at" < ?2 AND "id" IN ({})"#, marks.join(", "));
                    let mut params = vec![ctx.cookies.g("address")?, ctx.now.clone()];
                    params.extend(cron_ids.iter().map(|id| V::from(id.as_str())));
                    let db = ctx.logisRegion.to_key();
                    let out = js::all(&js::bind(&env.prepare(&db, &sql)?, &params)?).await?;
                    ctx.results = out.g("results")?;
                    ctx.success = out.g("success")?;

                    let found: Vec<String> = ctx.results.items().iter().map(|row| row.get("id").to_str()).collect();
                    let kept: Vec<V> = ctx.tasks.items().into_iter().filter(|t| !found.contains(&t.get("id").to_str())).collect();
                    ctx.tasks = V::from(kept);
                }
            }
            Ok(())
        }
        .await;
        if let Err(err) = r {
            console_log!("tasks err", err);
        }
    }
    Ok(())
}

/// 1475~1553 (try 안) — 페이지 나무. 원본 그대로 만들고 버린다(var 칸만 남는다).
///
/// 원본의 `branchs` 는 `[]` 에 글 키를 단 배열이다. 이 모형의 배열은 이름 있는 속성을 담지 못해 객체로 둔다
/// (for-in 순서 — 첨자 꼴 키 오름차순, 그다음 넣은 순서 — 는 같다).
fn tree(ctx: &mut Ctx) -> R<()> {
    ctx.branchs = V::new_obj();

    // 원본 그대로: safeClone 이 home 등의 Uint8Array data 를 첨자 키 객체로 바꾼다
    ctx._pages = safe_clone(&ctx.pages);

    ctx.p = V::Num(0.0);
    while less_than(&ctx.p, &ctx._pages.g("length")?) == Some(true) {
        ctx._page = ctx._pages.gk(&ctx.p)?;

        ctx.decompressedJsonString = V::from(decode_utf8(&ungzip(&ctx._page.g("data")?)?));

        let data = parse_str(&ctx.decompressedJsonString.to_str())?;
        ctx._page.s_("data", data)?;

        ctx._url = url_obj(&new_url(&ctx._page.g("data")?.g("origin")?.to_str())?);

        if ctx._page.g("data")?.g("item")?.truthy() {
            let k = tpl(T1490, &[ctx._page.g("data")?.g("origin")?, ctx._page.g("type")?]);
            ctx.branchs.s_(&k, spread_children(&ctx._page, V::new_arr()))?;
        }

        let id = ctx._page.g("id")?;
        ctx.branchs.sk(&id, spread_children(&ctx._page, V::new_arr()))?;

        ctx.p = V::Num(ctx.p.to_num() + 1.0);
    }

    // 1497
    ctx._temp = V::new_obj();

    // for-in 은 시작할 때의 키만 돈다 (돌면서 더한 키는 보지 않음)
    for key in ctx.branchs.keys() {
        ctx.key = V::from(key.as_str());
        if !ctx.branchs.has(&key) {
            continue;
        }

        ctx._page = safe_clone(&ctx.branchs.gk(&ctx.key)?);

        if ctx._temp.gk(&ctx._page.g("id")?)?.truthy() {
            continue;
        }

        ctx._temp.sk(&ctx._page.g("id")?, V::Bool(true))?;

        let k = tpl(T1506, &[ctx._page.g("data")?.g("origin")?, ctx._page.g("type")?]);
        ctx.parent = ctx.branchs.g(&k)?;

        if ctx.parent.truthy() {
            if ctx._page.g("data")?.g("item")?.truthy() {
                ctx.children = safe_clone(&ctx.parent.g("children")?);

                let k = tpl(T1512, &[ctx._page.g("data")?.g("origin")?, ctx._page.g("type")?]);
                ctx.branchs.s_(&k, spread_children(&ctx._page, ctx.children.clone()))?;
            } else {
                let k = tpl(T1517, &[ctx._page.g("data")?.g("origin")?, ctx._page.g("type")?]);
                ctx.branchs.g(&k)?.g("children")?.push(ctx._page.clone())?;
            }
        } else if ctx._page.g("data")?.g("item")?.truthy() {
            let k = tpl(T1522, &[ctx._page.g("data")?.g("origin")?, ctx._page.g("type")?]);
            if !ctx.branchs.g(&k)?.truthy() {
                let k = tpl(T1523, &[ctx._page.g("data")?.g("origin")?, ctx._page.g("type")?]);
                ctx.branchs.s_(&k, spread_children(&ctx._page, V::new_arr()))?;
            }
        } else {
            // 원본 그대로: parent 가 없는데 그 자리의 children 에 넣으려다 TypeError (나무 전체가 멈춘다)
            let k = tpl(T1529, &[ctx._page.g("data")?.g("origin")?, ctx._page.g("type")?]);
            ctx.branchs.g(&k)?.g("children")?.push(ctx._page.clone())?;
        }
    }

    // 1541
    ctx.tree = V::new_arr();

    for key in ctx.branchs.keys() {
        ctx.key = V::from(key.as_str());
        if ctx.branchs.has(&key) && !eth::is_address(&ctx.key) {
            ctx._page = ctx.branchs.gk(&ctx.key)?;

            ctx.tree.push(ctx._page.clone())?;
        }
    }
    Ok(())
}

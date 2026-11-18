//! 원본 1594~1963 (PUT: 벡터 · 멤버 · 대화) · 1964~1997 (DELETE: 작업 · 로그아웃).
//!
//! 둘 다 411 의 try 안이다: 여기서 새는 예외는 처리기의 'err' 로그 뒤 418. 원본이 try/catch 로 삼키는 자리
//! (벡터 전체 · 표마다 upsert · refs 해석 · 배치)만 여기서 잡는다.

use logis_worker_common::gzip::{decode_utf8, gzip_json, ungzip};
use logis_worker_common::js::{self, Env};
use logis_worker_common::jsv::{less_than, stringify_checked, to_lower, to_upper, tpl, Throw};
use logis_worker_common::util::hash_id;
use logis_worker_common::web::{headers_set, headers_vec, Body, Req, Resp};
use logis_worker_common::{add, aws, console_log, eth, loose_eq, obj, parse, platform, R, V};

use crate::ctx::Ctx;
use crate::flow::Step;
use crate::session::{decode_uri_component, s3};
use crate::templates::*;

/// `ethers.ZeroAddress`
const ZERO_ADDRESS: &str = "0x0000000000000000000000000000000000000000";

/// `Object.prototype` 의 속성 이름 — 평범한 객체 `{}` 에서 물려받아 읽히는 값(모두 참: 함수 또는 `__proto__` 의 객체)
const OBJECT_PROTOTYPE_KEYS: [&str; 12] = [
    "constructor",
    "__defineGetter__",
    "__defineSetter__",
    "hasOwnProperty",
    "__lookupGetter__",
    "__lookupSetter__",
    "isPrototypeOf",
    "propertyIsEnumerable",
    "toString",
    "valueOf",
    "__proto__",
    "toLocaleString",
];

/// `a ? a : b`
fn or(a: V, b: V) -> V {
    if a.truthy() {
        a
    } else {
        b
    }
}

/// `x++` (수)
fn inc(v: &V) -> V {
    V::Num(v.to_num() + 1.0)
}

/// `s.toUpperCase()` — 문자열이 아니면 원본처럼 TypeError
fn upper(v: &V, what: &str) -> R<String> {
    match v {
        V::Str(s) => Ok(to_upper(s)),
        V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'toUpperCase')", v.to_str()))),
        _ => Err(Throw::type_error(format!("{what}.toUpperCase is not a function"))),
    }
}

/// `s.split("@")[0]` — 문자열이 아니면 원본처럼 TypeError
fn split_at_first(v: &V, what: &str) -> R<V> {
    match v {
        V::Str(s) => Ok(V::from(s.split('@').next().unwrap_or("").to_string())),
        V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'split')", v.to_str()))),
        _ => Err(Throw::type_error(format!("{what}.split is not a function"))),
    }
}

/// `new URL(s)` (기준 없음) — 실패하면 TypeError
fn new_url(s: &V) -> R<url::Url> {
    url::Url::parse(&s.to_str()).map_err(|_| Throw::type_error("Invalid URL string."))
}

/// `new URL(...)` 의 쓰는 속성만 담은 값 (`var url` — session.rs 의 url_obj 와 같은 꼴)
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

/// `new Response(JSON.stringify(body), { headers })` (200)
fn json_response(ctx: &Ctx, body: &V) -> R<Resp> {
    let text = stringify_checked(body)?.unwrap_or_default();
    let mut r = Resp::new(200, Body::Text(text));
    r.headers = headers_vec(&ctx.headers);
    Ok(r)
}

/// `var { results, success, error } = await …` — D1 결과의 세 칸을 var 에
fn take_result(ctx: &mut Ctx, out: &V) -> R<()> {
    ctx.results = out.g("results")?;
    ctx.success = out.g("success")?;
    ctx.error = out.g("error")?;
    Ok(())
}

/// 1594~1963
pub async fn put(ctx: &mut Ctx, env: &Env, req: &Req) -> R<Step> {
    let query = ctx.req.g("query")?;

    // 1608
    ctx.success = V::Bool(false);
    ctx.results = V::new_arr();

    // 1618
    if loose_eq(&query.g("type")?, &V::from("vector"))
        && ctx.cookies.g("address")?.truthy()
        && ctx.cookies.g("team")?.truthy()
        && ctx.zoneRegion.truthy()
    {
        if let Err(err) = vector(ctx, env, req).await {
            console_log!("vector put err", err);
        }

        headers_set(&ctx.headers, "Content-Type", &V::from("application/json"))?;

        let body = obj! {
            "results" => ctx.results.clone(),
            "success" => ctx.success.clone(),
            "session" => ctx.cookies.clone(),
        };
        return Ok(Step::Return(json_response(ctx, &body)?));
    }

    // 1710
    if ctx.cookies.g("sender")?.truthy() && eth::is_address(&query.g("from")?) && eth::is_address(&query.g("to")?) {
        // 1713 — JSON.parse 가 던지면 [] 그대로
        ctx.refs = V::new_arr();
        let q_ref = query.g("ref")?;
        if q_ref.truthy() {
            if let Ok(refs) = parse(&q_ref) {
                ctx.refs = refs;
            }
        }

        // 1723 — 원본 그대로: refs 가 null(`?ref=null`) 이면 refs.length 에서 TypeError (418)
        if eth::is_address(&query.g("id")?) && ctx.refs.g("length")?.truthy() {
            members(ctx, env, &query).await?;
        } else if query.g("text")?.truthy() {
            talk(ctx, env, &query).await?;
        }
    }

    // 1954
    headers_set(&ctx.headers, "Content-Type", &V::from("application/json"))?;

    let body = obj! {
        "results" => ctx.results.clone(),
        "success" => ctx.success.clone(),
        "session" => ctx.cookies.clone(),
    };
    Ok(Step::Return(json_response(ctx, &body)?))
}

/// 1619~1696 (try 안 — 던지면 'vector put err')
async fn vector(ctx: &mut Ctx, env: &Env, req: &Req) -> R<()> {
    ctx.buffer = V::bytes(req.body.clone());

    if !ctx.buffer.get("byteLength").truthy() {
        return Ok(());
    }

    ctx.decompressedJsonString = V::from(decode_utf8(&ungzip(&ctx.buffer)?));
    ctx.payload = parse(&ctx.decompressedJsonString)?;
    ctx.vectorRegion = add(&V::from("commerce-logis-"), &ctx.zoneRegion);

    // 1629 — payload 가 null 이면 payload.items 에서 TypeError
    ctx.list = if ctx.payload.is_arr() {
        ctx.payload.clone()
    } else {
        let items = ctx.payload.g("items")?;
        if items.truthy() {
            items
        } else {
            V::new_arr()
        }
    };

    ctx.groups = V::new_obj();

    // 1633
    ctx.v = V::Num(0.0);
    while less_than(&ctx.v, &ctx.list.g("length")?) == Some(true) {
        'body: {
            ctx.vec = ctx.list.gk(&ctx.v)?;
            let vec = ctx.vec.clone();

            if !vec.truthy() {
                break 'body;
            }
            if !vec.g("id")?.truthy() {
                break 'body;
            }
            let values = vec.g("values")?;
            if !values.is_arr() {
                break 'body;
            }
            if !values.g("length")?.truthy() {
                break 'body;
            }

            ctx.vectorTable = or(vec.g("table")?, V::from("items"));

            // 원본 그대로: groups 는 평범한 객체라 table 이 "constructor" · "toString" · "__proto__" 등이면 물려받은 값(참)을 읽어
            // [] 를 만들지 않고, 그 값의 push 가 없어 TypeError (벡터 전체가 'vector put err')
            let key = ctx.vectorTable.to_key();
            let inherited = !ctx.groups.has(&key) && OBJECT_PROTOTYPE_KEYS.contains(&key.as_str());

            if !inherited && !ctx.groups.gk(&ctx.vectorTable)?.truthy() {
                ctx.groups.sk(&ctx.vectorTable, V::new_arr())?;
            }

            let entry = obj! {
                "id" => vec.g("id")?,
                "values" => vec.g("values")?,
                "metadata" => obj! {
                    "id" => vec.g("id")?,
                    "no" => or(vec.g("no")?, V::from("")),
                    "type" => or(vec.g("type")?, V::from("")),
                    "from" => or(vec.g("from")?, ctx.cookies.g("address")?),
                    "to" => or(vec.g("to")?, ctx.cookies.g("team")?),
                    "cc" => or(vec.g("cc")?, ctx.cookies.g("cc")?),
                    "bcc" => or(vec.g("bcc")?, V::from("")),
                    "ref" => or(vec.g("ref")?, V::from("")),
                },
            };

            if inherited {
                return Err(Throw::type_error("groups[vectorTable].push is not a function"));
            }
            ctx.groups.gk(&ctx.vectorTable)?.push(entry)?;
        }
        ctx.v = inc(&ctx.v);
    }

    // 1674 `for (var vectorTable in groups)`
    for key in ctx.groups.keys() {
        ctx.vectorTable = V::from(key.clone());
        if !ctx.groups.has(&key) {
            continue;
        }
        let r: R<()> = async {
            let name = tpl(T1677, &[ctx.vectorRegion.clone(), ctx.vectorTable.clone()]);
            let group = ctx.groups.gk(&ctx.vectorTable)?;
            env.vectorize_upsert(&name, &group).await?;

            ctx.g = V::Num(0.0);
            while less_than(&ctx.g, &ctx.groups.gk(&ctx.vectorTable)?.g("length")?) == Some(true) {
                let item = ctx.groups.gk(&ctx.vectorTable)?.gk(&ctx.g)?;
                ctx.results.push(obj! {
                    "id" => item.g("id")?,
                    "table" => ctx.vectorTable.clone(),
                })?;
                ctx.g = inc(&ctx.g);
            }

            ctx.success = V::Bool(true);
            Ok(())
        }
        .await;
        if let Err(err) = r {
            console_log!("vector upsert err", ctx.vectorTable, err);
        }
    }

    console_log!("[EMBED-CLIENT] Received local vectors:", ctx.results.g("length")?);
    Ok(())
}

/// 1724~1889 — 멤버 내보내기(cc = ZeroAddress) · 허용(cc = 주소), 끝에 한 번 배치
async fn members(ctx: &mut Ctx, env: &Env, query: &V) -> R<()> {
    ctx.statements = V::new_arr();

    ctx.r = V::Num(0.0);
    while less_than(&ctx.r, &ctx.refs.g("length")?) == Some(true) {
        ctx.ref_ = ctx.refs.gk(&ctx.r)?;

        if loose_eq(&query.g("cc")?, &V::from(ZERO_ADDRESS)) {
            // 1729 나가기 · 내보내기
            if eth::is_address(&query.g("to")?) && loose_eq(&query.g("id")?, &ctx.cookies.g("team")?) {
                // 원본 그대로: team 은 세션의 var — 있는 사용자면 글(team.from 은 undefined, 늘 다르다), 없으면 undefined 라 TypeError (418)
                if !loose_eq(&ctx.team.g("from")?, &query.g("to")?) {
                    revoke(ctx, env, query).await?;
                }
            }
        } else if eth::is_address(&query.g("cc")?) {
            // 1761 허용
            if loose_eq(&query.g("from")?, &ctx.cookies.g("address")?) && loose_eq(&query.g("id")?, &ctx.cookies.g("team")?) {
                if !loose_eq(&ctx.team.g("from")?, &query.g("to")?) {
                    grant(ctx, env, query).await?;
                }
            }
            // 1853 `else if (from == cookies.team && to == cookies.address)` — 본문이 모두 주석(읽기만 하고 던지지 않음)
        }

        ctx.r = inc(&ctx.r);
    }

    // 1884 — catch(err){} 로 삼킨다
    let db = tpl(T1885, &[ctx.zoneRegion.clone()]);
    let _ = env.batch(&db, &ctx.statements).await;
    Ok(())
}

/// 1739~1757 — 권한 해제
async fn revoke(ctx: &mut Ctx, env: &Env, query: &V) -> R<()> {
    let db = tpl(T1739, &[ctx.zoneRegion.clone()]);
    let stmt = js::bind(
        &env.prepare(&db, T1740[0])?,
        &[
            query.g("id")?,
            V::from("member"),
            ctx.cookies.g("address")?,
            query.g("to")?,
            ctx.cookies.g("address")?,
        ],
    )?;
    // `var { results } = …` — success 는 그대로
    ctx.results = js::all(&stmt).await?.g("results")?;

    if ctx.results.g("length")?.truthy() {
        ctx.row = ctx.results.gi(0)?;

        // 원본 그대로: 응답의 results 에 든 행도 같이 바뀐다(같은 객체)
        ctx.row.s_("to", V::from(ZERO_ADDRESS))?;

        // 원본 그대로: `to` 는 SQLite 키워드(ID 대체 목록에 없음)라 이 문장은 배치에서 실패하고(1886 이 삼킴) 같은 배치의 다른 문장도
        // 함께 버려질 것이다 (D1 에서 시험하지 않음)
        let db = tpl(T1749, &[ctx.zoneRegion.clone()]);
        let stmt = js::bind(&env.prepare(&db, T1750[0])?, &[ctx.row.g("to")?, ctx.now.clone(), ctx.row.g("id")?])?;
        ctx.statements.push(stmt)?;
    }
    Ok(())
}

/// 1769~1849 — 권한 허용: 멤버 행을 지우고 refs 마다 다시 넣는다
async fn grant(ctx: &mut Ctx, env: &Env, query: &V) -> R<()> {
    let cookies = ctx.cookies.clone();

    ctx.memberId = V::from(hash_id(&add(&add(&cookies.g("address")?, &query.g("to")?), &cookies.g("team")?))?);

    // 1771 — 원본 그대로: 값을 SQL 에 이어 붙인다
    let db = tpl(T1771, &[ctx.zoneRegion.clone()]);
    let out = js::all(&env.prepare(&db, &tpl(T1772, &[ctx.memberId.clone()]))?).await?;
    take_result(ctx, &out)?;

    if ctx.results.g("length")?.truthy() {
        return Ok(());
    }

    // 1778
    let db = tpl(T1778, &[ctx.zoneRegion.clone()]);
    let sql = tpl(T1779, &[ctx.memberId.clone(), query.g("to")?, query.g("cc")?]);
    let out = js::all(&env.prepare(&db, &sql)?).await?;
    take_result(ctx, &out)?;

    if !ctx.results.g("length")?.truthy() {
        return Ok(());
    }

    // 1784
    ctx.row = ctx.results.gi(0)?;
    ctx.memberEmail = ctx.row.g("from")?;
    ctx.memberName = split_at_first(&ctx.memberEmail, "memberEmail")?;

    // 1789
    ctx.allowedRefs = V::new_arr();
    let r: R<()> = (|| {
        let q_ref = query.g("ref")?;
        if q_ref.truthy() {
            ctx.allowedRefs = parse(&q_ref)?;
        }
        Ok(())
    })();
    if let Err(err) = r {
        console_log!("ref parse error", err);
    }

    // 1800
    let db = tpl(T1800, &[ctx.zoneRegion.clone()]);
    let stmt = js::bind(
        &env.prepare(&db, T1800_1[0])?,
        &[cookies.g("address")?, ctx.memberEmail.clone(), cookies.g("team")?],
    )?;
    ctx.statements.push(stmt)?;

    // 1805 `gzip(…, { to: 'arraybuffer' })` 는 Uint8Array, 묶는 것은 그 `.buffer` — 여기서는 처음부터 ArrayBuffer(V::Bytes) 하나로 둔다
    ctx.memberDataArr = gzip_json(&obj! {
        "flag" => ctx.FLAG.clone(),
        "name" => ctx.memberName.clone(),
        "title" => "",
        "region" => V::Null,
        "page_count" => 0,
        "favicon" => V::Null,
    })?;

    // 1815 `allowedRefs.length > 0`
    if less_than(&V::Num(0.0), &ctx.allowedRefs.g("length")?) == Some(true) {
        // 원본 그대로: `var r` 라 바깥 refs 반복(1726)과 같은 셈 변수 — 이 반복이 끝나면 r = allowedRefs.length 라
        // 바깥 반복은 첫 회차 뒤 끝난다
        ctx.r = V::Num(0.0);
        while less_than(&ctx.r, &ctx.allowedRefs.g("length")?) == Some(true) {
            ctx.targetPageId = ctx.allowedRefs.gk(&ctx.r)?;

            let text = add(&add(&add(&cookies.g("address")?, &ctx.memberEmail), &cookies.g("team")?), &ctx.targetPageId);
            ctx.uniqueMemberRowId = V::from(hash_id(&text)?);

            let db = tpl(T1822, &[ctx.zoneRegion.clone()]);
            let stmt = js::bind(
                &env.prepare(&db, T1822_1[0])?,
                &[
                    ctx.uniqueMemberRowId.clone(),
                    cookies.g("address")?,
                    ctx.memberEmail.clone(),
                    cookies.g("team")?,
                    cookies.g("address")?,
                    ctx.targetPageId.clone(),
                    ctx.memberDataArr.clone(),
                    ctx.now.clone(),
                    ctx.now.clone(),
                ],
            )?;
            ctx.statements.push(stmt)?;

            ctx.r = inc(&ctx.r);
        }
    } else {
        // 1840
        let text = add(&add(&add(&cookies.g("address")?, &ctx.memberEmail), &cookies.g("team")?), &V::from("none"));
        ctx.baseId = V::from(hash_id(&text)?);

        let db = tpl(T1842, &[ctx.zoneRegion.clone()]);
        let stmt = js::bind(
            &env.prepare(&db, T1842_1[0])?,
            &[
                ctx.baseId.clone(),
                cookies.g("address")?,
                ctx.memberEmail.clone(),
                cookies.g("team")?,
                cookies.g("address")?,
                ctx.memberDataArr.clone(),
                ctx.now.clone(),
                ctx.now.clone(),
            ],
        )?;
        ctx.statements.push(stmt)?;
    }
    Ok(())
}

/// 1894~1948 — 대화 넣기 (try 없음: 던지면 418)
async fn talk(ctx: &mut Ctx, env: &Env, query: &V) -> R<()> {
    ctx.text = V::from(decode_uri_component(&query.g("text")?.to_str())?);

    // 1896 — cookies.href 가 절대 주소가 아니면 TypeError
    let u = new_url(&ctx.cookies.g("href")?)?;
    ctx.url = url_obj(&u);

    ctx.link = V::from(to_lower(&(ctx.url.get("pathname").to_str() + &ctx.url.get("search").to_str())));

    ctx.arr = gzip_json(&obj! {
        "text" => ctx.text.clone(),
        "link" => ctx.link.clone(),
        "origin" => ctx.url.get("origin"),
    })?;

    // 1907 — 객체 리터럴의 순서대로 (id 의 무작위 키가 먼저)
    let id = hash_id(&V::Undef)?;
    let from = query.g("from")?;
    let to = query.g("to")?;
    let cc = ctx.cookies.g("cc")?;
    let bcc = ctx.cookies.g("bcc")?;
    // 원본 그대로: isDetail 이면 cookies.cc 가 없을 때 toUpperCase 에서 TypeError
    let cc_part = if ctx.isDetail.truthy() { V::from(upper(&ctx.cookies.g("cc")?, "cookies.cc")?) } else { ctx.cookies.g("cc")? };
    let ref_ = hash_id(&add(&add(&ctx.cookies.g("team")?, &cc_part), &ctx.link))?;

    ctx.talk = obj! {
        "id" => id,
        "type" => "talk",
        "from" => from,
        "to" => to,
        "cc" => cc,
        "bcc" => bcc,
        "ref" => ref_,
        "data" => ctx.arr.clone(),
        "created_at" => ctx.now.clone(),
        "updated_at" => ctx.now.clone(),
    };

    // 1921
    let talk = ctx.talk.clone();
    let db = tpl(T1921, &[ctx.zoneRegion.clone()]);
    let stmt = js::bind(
        &env.prepare(&db, T1921_1[0])?,
        &[
            talk.g("id")?,
            talk.g("type")?,
            talk.g("from")?,
            talk.g("to")?,
            talk.g("cc")?,
            talk.g("bcc")?,
            talk.g("ref")?,
            talk.g("data")?,
            talk.g("created_at")?,
            talk.g("updated_at")?,
        ],
    )?;
    let out = js::run(&stmt).await?;
    take_result(ctx, &out)?;
    Ok(())
}

/// 1964~1997
pub async fn delete(ctx: &mut Ctx, env: &Env, _req: &Req) -> R<Step> {
    let query = ctx.req.g("query")?;

    if ctx.cookies.g("address")?.truthy() {
        if loose_eq(&query.g("type")?, &V::from("crons")) {
            let q_ref = query.g("ref")?;
            if eth::is_address(&q_ref) {
                ctx.ref_ = V::from(to_lower(&q_ref.to_str()));

                // 원본 그대로: tasks 를 items DB 에서 지우고(그 표가 없으면 D1 오류로 418), 값을 큰따옴표로 이어 붙인다
                // (SQLite 는 큰따옴표를 먼저 열 이름으로 본다). try 가 없어 run 이 던지면 418
                let db = tpl(T1970, &[ctx.zoneRegion.clone()]);
                let out = js::run(&env.prepare(&db, &tpl(T1971, &[ctx.ref_.clone()]))?).await?;
                take_result(ctx, &out)?;
            }
        } else {
            // 1975 로그아웃 — `fetch(await s3.sign(url, { method: 'DELETE' }))`: 서명만 aws4fetch, 보내기는 전역 fetch(다시 시도 없음)
            let client = s3(env);
            let url = tpl(T1976, &[env.get("aws_bucket"), env.get("aws_region"), ctx.cookies.g("hash")?]);
            let signed = client.sign(&url, "DELETE", &[], b"", platform::now())?;
            // 원본 그대로: 응답 상태를 보지 않는다
            let res = aws::send(&signed).await?;
            ctx.response = V::ext(js::Js(res));

            // `JSON.stringify(response)` — Response 는 열거되는 자기 속성이 없어 "{}"
            console_log!("response", "{}");

            ctx.cookies = V::new_obj();
        }
    }

    // 1988 — 원본 그대로: Content-Type 을 넣지 않는다
    let body = obj! {
        "success" => false,
        "results" => V::new_arr(),
        "session" => ctx.cookies.clone(),
    };
    Ok(Step::Return(json_response(ctx, &body)?))
}

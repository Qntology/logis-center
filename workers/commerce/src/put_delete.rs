//! 원본 1594~1963 (PUT: 벡터 · 멤버 · 대화) · 1964~1997 (DELETE: 작업 · 로그아웃).
//!
//! 둘 다 411 의 try 안이다: 여기서 새는 예외는 처리기의 'err' 로그 뒤 418. 원본이 try/catch 로 삼키는 자리
//! (벡터 전체 · 표마다 upsert · refs 해석 · 배치)만 여기서 잡는다.

use logis_worker_common::gzip::{decode_utf8, gzip_json, ungzip};
use logis_worker_common::js::{self, Env};
use logis_worker_common::jsv::{js_trim, less_than, stringify_checked, to_lower, to_upper, tpl, Throw};
use logis_worker_common::util::hash_id;
use logis_worker_common::web::{headers_set, headers_vec, Body, Req, Resp};
use logis_worker_common::{add, aws, console_log, eth, loose_eq, obj, parse, parse_str, platform, R, V};

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

    // 고침 (fix-app-invite): 이메일 초대 (앱 main.ts 2480~2518 — from = 팀, to = 자기 주소, email, ref = 페이지 id 배열, id 없음)
    if query.g("email")?.truthy() && !query.g("id")?.truthy() {
        return Ok(Step::Return(invite(ctx, env, &query).await?));
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
            // 고침 (fix-member-per-page): 이 페이지의 멤버가 아니면 대화를 넣지 않는다 (원본은 member 를 보지 않았다)
            if !ctx.member.truthy() {
                headers_set(&ctx.headers, "Content-Type", &V::from("application/json"))?;
                let body = obj! {
                    "results" => V::new_arr(),
                    "success" => false,
                    "error" => "not a member of this page",
                    "session" => ctx.cookies.clone(),
                };
                return Ok(Step::Return(json_response(ctx, &body)?));
            }
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

/// 1724~1889 — 고침 (fix-revoke-owner · fix-app-invite): 원본의 팀장 막기(`team.from != to`)는 세션의 팀 이름 글(있는 사용자) 또는
/// 팀 객체(새 사용자)를 읽어 듣지 않았고, 주기(from = 팀장 · to = 이메일 · cc = 팀)와 거두기(id = 팀 · bcc = 나)와 읽는 곳(1567: from = 멤버 ·
/// to = 팀 · bcc = 페이지 bcc)의 행 꼴이 서로 달랐다 → 팀장은 팀 행의 from([`crate::fixes::team_owner`]), 멤버 행은 초대 확정과 같은 꼴
/// ([`crate::fixes::member_statements`]). 거두기 = 팀장(내보내기) 또는 본인(나가기)이 refs 의 페이지 행을 지운다. 팀장 자신은 거둘 수 없다
async fn members(ctx: &mut Ctx, env: &Env, query: &V) -> R<()> {
    let logis_region = ctx.logisRegion.to_key();
    let team = ctx.cookies.g("team")?;
    let owner = crate::fixes::team_owner(env, &logis_region, &ctx.admin, &team).await?;

    // isAddress(to) 는 put() 이 이미 봤다
    let member = V::from(to_lower(&query.g("to")?.to_str()));
    let page_ids = crate::fixes::address_list(&ctx.refs, 50);

    ctx.statements = V::new_arr();
    let items_db = format!("commerce_logis_{}_items", ctx.zoneRegion.to_str());

    if loose_eq(&query.g("id")?, &team) && owner.truthy() && !loose_eq(&member, &owner) {
        let address = ctx.cookies.g("address")?;
        if loose_eq(&query.g("cc")?, &V::from(ZERO_ADDRESS)) {
            // 나가기(본인) · 내보내기(팀장)
            if loose_eq(&address, &owner) || loose_eq(&address, &member) {
                for id in &page_ids {
                    let stmt = js::bind(
                        &env.prepare(&items_db, r#"DELETE FROM items WHERE "type" = 'member' AND "from" = ?1 AND "to" = ?2 AND "ref" = ?3"#)?,
                        &[member.clone(), team.clone(), V::from(id.as_str())],
                    )?;
                    ctx.statements.push(stmt)?;
                    ctx.results.push(obj! {
                        "id" => id.as_str(),
                        "table" => "items",
                        "type" => "member",
                        "from" => member.clone(),
                        "to" => team.clone(),
                        "ref" => id.as_str(),
                    })?;
                }
            }
        } else if eth::is_address(&query.g("cc")?) {
            // 허용(팀장): 팀에 든 사용자에게만 — 그 멤버의 행을 지우고 refs 의 페이지마다 다시 넣는다
            if loose_eq(&query.g("from")?, &address) && loose_eq(&address, &owner) {
                let rows = js::query_bound(
                    env,
                    &logis_region,
                    r#"SELECT "id", "data" FROM users WHERE "id" = ?1 AND "type" = 'user' AND "to" = ?2 LIMIT 1"#,
                    &[member.clone(), team.clone()],
                )
                .await?;
                if rows.g("length")?.truthy() {
                    ctx.memberName = crate::fixes::member_name(&rows.gi(0)?.g("data")?);
                    ctx.memberDataArr = gzip_json(&obj! {
                        "flag" => ctx.FLAG.clone(),
                        "name" => ctx.memberName.clone(),
                        "title" => "",
                        "region" => ctx.zoneRegion.clone(),
                        "page_count" => 0,
                        "favicon" => V::Null,
                    })?;

                    let stmt = js::bind(
                        &env.prepare(&items_db, r#"DELETE FROM items WHERE "type" = 'member' AND "from" = ?1 AND "to" = ?2"#)?,
                        &[member.clone(), team.clone()],
                    )?;
                    ctx.statements.push(stmt)?;

                    let (statements, granted) = crate::fixes::member_statements(
                        env,
                        &ctx.zoneRegion,
                        &member,
                        &team,
                        &address,
                        &page_ids,
                        &ctx.memberDataArr,
                        &ctx.now,
                    )
                    .await?;
                    for stmt in statements.items() {
                        ctx.statements.push(stmt)?;
                    }
                    ctx.results = granted;
                }
            }
        }
    }

    if ctx.statements.g("length")?.truthy() {
        match env.batch(&items_db, &ctx.statements).await {
            Ok(_) => ctx.success = V::Bool(true),
            Err(err) => {
                console_log!("member batch err", err);
                ctx.results = V::new_arr();
            }
        }
    }
    Ok(())
}

/// 고침 (fix-app-invite): 팀장만 같은 지역 DB 의 users 에 type 'invite' 행을 넣는다 (다시 초대하면 덮어씀).
/// 행: id = hashId(팀 + 소문자 이메일), from = 초대한 사람, to = 팀, cc = hashId(소문자 이메일), data = gzip({ email, refs, zone, flag, name }).
/// 응답 results[0] = { id, table: 'users', type: 'invite', hook } — hook 은 앱의 기본값과 같은 `${hash}.logis.center@oauth.email`
async fn invite(ctx: &mut Ctx, env: &Env, query: &V) -> R<Resp> {
    let mut invited = V::Null;
    let mut error = "invite requires the signed-in team owner";
    let email = to_lower(js_trim(&query.g("email")?.to_str()));
    let cookies = ctx.cookies.clone();

    if !is_email(&email) {
        error = "invalid email";
    } else if cookies.g("address")?.truthy()
        && cookies.g("team")?.truthy()
        && ctx.zoneRegion.truthy()
        && loose_eq(&query.g("from")?, &cookies.g("team")?)
        && loose_eq(&query.g("to")?, &cookies.g("address")?)
    {
        let logis_region = ctx.logisRegion.to_key();
        let owner = crate::fixes::team_owner(env, &logis_region, &ctx.admin, &cookies.g("team")?).await?;
        if owner.truthy() && loose_eq(&owner, &cookies.g("address")?) {
            // JSON.parse(req.query.ref || '[]') 가 던지면 []
            let q_ref = query.g("ref")?;
            let refs_value = parse_str(&if q_ref.truthy() { q_ref.to_str() } else { "[]".to_string() }).unwrap_or_else(|_| V::new_arr());
            let refs = crate::fixes::address_list(&refs_value, 50);

            let id = V::from(hash_id(&add(&cookies.g("team")?, &V::from(email.as_str())))?);
            invited = obj! {
                "id" => id.clone(),
                "table" => "users",
                "type" => "invite",
                "hook" => format!("{}.logis.center@oauth.email", cookies.g("hash")?.to_str()),
            };

            let data = gzip_json(&obj! {
                "email" => email.as_str(),
                "refs" => V::from(refs.iter().map(|r| V::from(r.as_str())).collect::<Vec<_>>()),
                "zone" => ctx.zoneRegion.clone(),
                "flag" => ctx.FLAG.clone(),
                "name" => email.split('@').next().unwrap_or(""),
            })?;

            js::run_bound(
                env,
                &logis_region,
                SQL_INVITE_UPSERT,
                &[id, cookies.g("address")?, cookies.g("team")?, V::from(hash_id(&V::from(email.as_str()))?), data, ctx.now.clone()],
            )
            .await?;
        }
    }

    headers_set(&ctx.headers, "Content-Type", &V::from("application/json"))?;

    let body = if invited.truthy() {
        obj! { "results" => V::from(vec![invited]), "success" => true, "session" => cookies }
    } else {
        obj! { "results" => V::new_arr(), "success" => false, "error" => error, "session" => cookies }
    };
    json_response(ctx, &body)
}

const SQL_INVITE_UPSERT: &str = "
								INSERT INTO users (\"id\", \"type\", \"from\", \"to\", \"cc\", \"bcc\", \"ref\", \"data\", \"created_at\", \"updated_at\")
								VALUES (?1, 'invite', ?2, ?3, ?4, '', '', ?5, ?6, ?6) ON CONFLICT (id) DO UPDATE SET
									\"from\" = EXCLUDED.\"from\",
									\"to\" = EXCLUDED.\"to\",
									\"cc\" = EXCLUDED.\"cc\",
									\"data\" = EXCLUDED.\"data\",
									\"created_at\" = EXCLUDED.\"created_at\",
									\"updated_at\" = EXCLUDED.\"updated_at\"
							";

/// `/^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$/` (앱 main.ts 2464 의 검사를 소문자로)
fn is_email(s: &str) -> bool {
    let Some((local, domain)) = s.split_once('@') else {
        return false;
    };
    let local_ok = !local.is_empty() && local.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._%+-".contains(&b));
    // 도메인 = [a-z0-9.-]+ '.' [a-z]{2,} — 어떤 점 뒤가 끝까지 두 글자 넘는 영문이고 그 앞이 비지 않으면 맞는다 (정규식의 되돌아가기와 같은 결과)
    let domain_ok = domain.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'.' || b == b'-')
        && (1..domain.len()).any(|i| {
            domain.as_bytes()[i] == b'.' && domain.len() - i - 1 >= 2 && domain[i + 1..].bytes().all(|b| b.is_ascii_lowercase())
        });
    local_ok && domain_ok
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

                // 고침 (fix-delete-tasks-db): 원본은 tasks 를 zone items DB 에서 지우려 했고(작업은 지역 DB 에 있다) 값을 큰따옴표로 이어 붙였다
                // (SQLite 는 큰따옴표를 먼저 열 이름으로 본다) → 지역 DB 에서, 값은 묶고, 자기 작업(from = 나 · to = 내 팀)만. try 가 없어 던지면 418
                let db = ctx.logisRegion.to_key();
                let sql = r#"DELETE FROM tasks WHERE "ref" = ?1 AND "from" = ?2 AND "to" = ?3"#;
                let out = js::run_bound(env, &db, sql, &[ctx.ref_.clone(), ctx.cookies.g("address")?, ctx.cookies.g("team")?]).await?;
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

//! 고침(fix-*)이 더한 도우미 — 고친 원본(`scripts/workers-diff/fixes/commerce_patches.txt` 의 fix-helpers)과 같은 동작.
//! 원본에 없던 길(사전 요청 · /relay · 초대 · 멤버 행)이라 Ctx 의 var 대신 지역 값으로 쓴다.

use logis_worker_common::gzip::ungzip_json;
use logis_worker_common::js::{self, Env, Js};
use logis_worker_common::jsv::{num_to_string, stringify_str, u16_len, Throw};
use logis_worker_common::util::hash_id;
use logis_worker_common::web::{headers_new, headers_set, headers_vec, Body, Req, Resp};
use logis_worker_common::{add, aws, eth, obj, parse_str, platform, R, V};

use crate::session::s3;

/// `corsHeaders(h)` — fix-cors-methods · fix-preflight-session
pub fn cors_headers(h: &V) -> R<()> {
    headers_set(h, "Access-Control-Allow-Origin", &V::from("*"))?;
    headers_set(h, "Access-Control-Allow-Methods", &V::from("GET, POST, PUT, DELETE, OPTIONS"))?;
    headers_set(h, "Access-Control-Allow-Headers", &V::from("Content-Type, Content-Encoding, Authorization, X-Requested-With"))?;
    Ok(())
}

/// `appJson(text)` — 평범한 JSON 객체만 (배열 · null · 원시값 · 깨진 JSON 이면 None)
pub fn app_json(text: &str) -> Option<V> {
    match parse_str(text) {
        Ok(v @ V::Obj(_)) => Some(v),
        _ => None,
    }
}

/// `addressList(value, max)` — 배열이면 그 안의 주소 글만 (소문자, 겹침 없이, 최대 max 개)
pub fn address_list(value: &V, max: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if !value.is_arr() {
        return out;
    }
    for item in value.items() {
        if out.len() >= max {
            break;
        }
        let v = match &item {
            V::Str(s) => logis_worker_common::jsv::to_lower(s),
            _ => String::new(),
        };
        if eth::is_address(&V::from(v.as_str())) && !out.contains(&v) {
            out.push(v);
        }
    }
    out
}

/// 세션 505~517 의 cc 글 (두 부분 도메인이면 끝 세 조각, 아니면 끝 두 조각 — 모자란 조각은 "undefined")
pub fn domain_cc_text(current_host: &str) -> String {
    let host: Vec<&str> = current_host.split('.').collect();
    let part = |back: usize| -> String {
        if host.len() >= back {
            host[host.len() - back].to_string()
        } else {
            "undefined".into()
        }
    };
    // 원본 그대로: endsWith 라 "xco.kr" 같은 이름도 두 부분 도메인으로 친다
    if logis_worker_common::tables::TWO_PART_DOMAINS.iter().any(|d| current_host.ends_with(d)) {
        part(3) + "." + &part(2) + "." + &part(1)
    } else {
        part(2) + "." + &part(1)
    }
}

/// `ccOfHost(host)` = hashId(cc 글)
pub fn cc_of_host(current_host: &str) -> R<String> {
    hash_id(&V::from(domain_cc_text(current_host)))
}

/// `url.host` (포트가 있으면 붙인다)
pub fn url_host(u: &url::Url) -> String {
    let hostname = u.host_str().unwrap_or("").to_string();
    match u.port() {
        Some(p) => format!("{hostname}:{p}"),
        None => hostname,
    }
}

/// `pageLink(href, origin)` — fix-link-hoist: 996 의 link 와 같은 값, 실패하면 ""
pub fn page_link(href: &str, origin: &str) -> String {
    let u = match url::Url::parse(href) {
        Ok(u) => Some(u),
        Err(_) => {
            let base = if origin.is_empty() { "https://commerce.logis.center" } else { origin };
            url::Url::parse(base).ok().and_then(|b| b.join(href).ok())
        }
    };
    match u {
        Some(u) => {
            let search = match u.query() {
                Some(q) if !q.is_empty() => format!("?{q}"),
                _ => String::new(),
            };
            logis_worker_common::jsv::to_lower(&(u.path().to_string() + &search))
        }
        None => String::new(),
    }
}

/// `await memberPages(env, zone, address, team, pages, homeCc)` — fix-member-per-page: 팀장이 아닌 팀원의 `cookies.pages`.
/// 팀 base.pages 는 proxy 가 채우는 `{ [사이트 cc]: { [페이지 type]: { draft, count } } }` (페이지 id 가 아니라 사이트 cc · type 으로 묶임).
/// 그 사람의 멤버 행 bcc 로 거른다: cc 가 homeCc(팀 공통)이거나 사이트 멤버 행(bcc = cc)이 있으면 그 cc 통째로,
/// 아니면 페이지 멤버 행(bcc = hashId(type + cc), 상세 페이지면 hashId(type + 대문자 cc))이 있는 type 만. 남는 type 이 없는 cc 는 뺀다
pub async fn member_pages(env: &Env, zone: &V, address: &V, team: &V, pages: &V, home_cc: &V) -> R<V> {
    let db = format!("commerce_logis_{}_items", zone.to_str());
    let rows = js::query_bound(env, &db, r#"SELECT "bcc" FROM items WHERE "type" = 'member' AND "from" = ?1 AND "to" = ?2 LIMIT 1000"#, &[address.clone(), team.clone()]).await?;
    let bccs: Vec<String> = rows.items().iter().filter_map(|row| row.get("bcc").as_str().map(str::to_string)).collect();
    let has = |s: &str| bccs.iter().any(|b| b == s);

    let out = V::new_obj();
    for (cc, types) in pages.entries() {
        if cc == home_cc.to_str() || has(&cc) {
            out.put(&cc, types);
            continue;
        }
        let kept = V::new_obj();
        let mut any = false;
        if matches!(types, V::Obj(_)) {
            for (type_, slot) in types.entries() {
                let lower = hash_id(&V::from(type_.clone() + &cc))?;
                let upper = hash_id(&V::from(type_.clone() + &logis_worker_common::jsv::to_upper(&cc)))?;
                if has(&lower) || has(&upper) {
                    kept.put(&type_, slot);
                    any = true;
                }
            }
        }
        if any {
            out.put(&cc, kept);
        }
    }
    Ok(out)
}

/// `await teamOwner(env, logisRegion, admin, team)` — 팀 행(id == to == 팀)의 from
pub async fn team_owner(env: &Env, logis_region: &str, admin: &V, team: &V) -> R<V> {
    if admin.truthy() && admin.get("id").truthy() && logis_worker_common::loose_eq(&admin.get("id"), team) {
        return admin.g("from");
    }
    let results = js::query_bound(env, logis_region, r#"SELECT "from" FROM users WHERE "id" = ?1 AND "type" = 'team' LIMIT 1"#, &[team.clone()]).await?;
    if results.length()?.truthy() {
        results.gi(0)?.g("from")
    } else {
        Ok(V::Undef)
    }
}

const SQL_PAGE_BCC: &str = r#"SELECT "id", "bcc" FROM items WHERE "type" = 'pages' AND "id" = ?1 AND "to" = ?2 LIMIT 1"#;
const SQL_MEMBER_UPSERT: &str = "
			INSERT INTO items (\"id\", \"type\", \"from\", \"to\", \"cc\", \"bcc\", \"ref\", \"data\", \"created_at\", \"updated_at\")
			VALUES (?1, 'member', ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8) ON CONFLICT (id) DO UPDATE SET
				\"cc\" = EXCLUDED.\"cc\",
				\"ref\" = EXCLUDED.\"ref\",
				\"data\" = EXCLUDED.\"data\",
				\"updated_at\" = EXCLUDED.\"updated_at\"
		";

/// `await memberStatements(env, zone, member, team, granter, refs, data, now)` → (문장들, 응답 행들).
/// 멤버 행: type 'member', from = 멤버, to = 팀, cc = 준 사람, bcc = 페이지 행의 bcc, ref = 페이지 id, id = hashId(멤버 + 팀 + bcc)
#[allow(clippy::too_many_arguments)]
pub async fn member_statements(env: &Env, zone: &V, member: &V, team: &V, granter: &V, refs: &[String], data: &V, now: &V) -> R<(V, V)> {
    let db = format!("commerce_logis_{}_items", zone.to_str());
    let statements = V::new_arr();
    let rows = V::new_arr();
    for r in refs {
        let results = js::query_bound(env, &db, SQL_PAGE_BCC, &[V::from(r.as_str()), team.clone()]).await?;
        if !results.length()?.truthy() || !results.gi(0)?.g("bcc")?.truthy() {
            continue;
        }
        let page = results.gi(0)?;
        let id = V::from(hash_id(&add(&add(member, team), &page.g("bcc")?))?);
        if rows.items().iter().any(|row| logis_worker_common::loose_eq(&row.get("id"), &id)) {
            continue;
        }
        let stmt = js::bind(
            &env.prepare(&db, SQL_MEMBER_UPSERT)?,
            &[id.clone(), member.clone(), team.clone(), granter.clone(), page.g("bcc")?, page.g("id")?, data.clone(), now.clone()],
        )?;
        statements.push(stmt)?;
        rows.push(obj! {
            "id" => id,
            "table" => "items",
            "type" => "member",
            "from" => member.clone(),
            "to" => team.clone(),
            "cc" => granter.clone(),
            "bcc" => page.g("bcc")?,
            "ref" => page.g("id")?,
        })?;
    }
    Ok((statements, rows))
}

/// 멤버 users 행의 data.name (풀 수 없으면 "")
pub fn member_name(data: &V) -> V {
    match ungzip_json(data) {
        Ok(d) => {
            let name = d.get("name");
            if name.truthy() {
                name
            } else {
                V::from("")
            }
        }
        Err(_) => V::from(""),
    }
}

/// `/relay/{hash}` 의 hash (`^\/relay\/([0-9A-Za-z_-]{1,128})\/?$`)
pub fn relay_hash(path: &str) -> Option<String> {
    let rest = path.strip_prefix("/relay/")?;
    let rest = rest.strip_suffix('/').unwrap_or(rest);
    let ok = !rest.is_empty() && rest.len() <= 128 && rest.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    ok.then(|| rest.to_string())
}

fn relay_resp(status: u16, headers: &V, text: String) -> Resp {
    let mut r = Resp::new(status, Body::Text(text));
    r.headers = headers_vec(headers);
    r
}

/// `relayFetch(request, env, s3, hash)` — fix-app-relay
pub async fn relay(req: &Req, env: &Env, hash: &str) -> R<Resp> {
    let headers = headers_new();
    cors_headers(&headers)?;
    headers_set(&headers, "Content-Type", &V::from("application/json"))?;
    let client = s3(env);
    let bucket = env.get("aws_bucket").to_str();
    let region = env.get("aws_region").to_str();
    let key = format!("https://{bucket}.s3.{region}.amazonaws.com/relay/{hash}");

    if req.method == "POST" {
        // `new URL(request.url).searchParams.get('token')` — 첫 값
        let url = url::Url::parse(&req.url).map_err(|_| Throw::type_error("Invalid URL string."))?;
        let token = url.query_pairs().find(|(k, _)| k == "token").map(|(_, v)| v.into_owned());
        let Some(token) = token.filter(|t| !t.is_empty()) else {
            return Ok(relay_resp(401, &headers, "{}".into()));
        };
        let head = aws::fetch(&client, &format!("https://{bucket}.s3.{region}.amazonaws.com/hash/{hash}"), "HEAD", &[], &[]).await?;
        let status = js::response_status(&head);
        let meta_token = js::response_header(&head, "x-amz-meta-token")?;
        if !(200.0..=299.0).contains(&status) || !logis_worker_common::loose_eq(&meta_token, &V::from(token.as_str())) {
            return Ok(relay_resp(401, &headers, "{}".into()));
        }
        // `await request.text()` (UTF-8, 깨진 바이트는 U+FFFD)
        let text = String::from_utf8_lossy(&req.body).into_owned();
        let payload = app_json(&text);
        let Some(payload) = payload.filter(|_| u16_len(&text) <= 65536) else {
            return Ok(relay_resp(400, &headers, "{}".into()));
        };
        let put_headers: Vec<(String, String)> = vec![
            ("Content-Type".into(), "application/json".into()),
            ("x-amz-meta-created_at".into(), num_to_string(platform::now())),
        ];
        aws::fetch(&client, &key, "PUT", &put_headers, stringify_str(&payload).as_bytes()).await?;
        return Ok(relay_resp(200, &headers, stringify_str(&obj! { "success" => true })));
    }

    if req.method == "GET" {
        let res = aws::fetch(&client, &key, "GET", &[], &[]).await?;
        let status = js::response_status(&res);
        if (200.0..=299.0).contains(&status) {
            let created_at = js::response_header(&res, "x-amz-meta-created_at")?.to_num();
            if platform::now() - created_at < 600000.0 {
                let text = js::call_async(&V::ext(Js(res)), "text", &[]).await?;
                return Ok(relay_resp(200, &headers, text.to_str()));
            }
        }
        return Ok(relay_resp(404, &headers, "{}".into()));
    }

    Ok(relay_resp(405, &headers, "{}".into()))
}

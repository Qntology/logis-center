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

fn push_unique(list: &mut Vec<String>, s: String) {
    if !list.contains(&s) {
        list.push(s);
    }
}

const SQL_MEMBER_BCCS: &str = r#"SELECT "bcc" FROM items WHERE "type" = 'member' AND "from" = ?1 AND "to" = ?2 LIMIT 1000"#;
const SQL_TEAM_PAGES: &str = r#"SELECT "id", "cc", "bcc", "data" FROM items WHERE "type" = 'pages' AND "to" = ?1 LIMIT 1000"#;

/// 페이지 행에서 쓰는 칸 (data 는 gzip JSON — 풀 수 없으면 type · link 가 빈 글)
struct PageRow {
    id: Option<String>,
    cc: Option<String>,
    bcc: Option<String>,
    type_: String,
    link: String,
}

fn page_row(p: &V) -> PageRow {
    let data = ungzip_json(&p.get("data")).unwrap_or(V::Undef);
    let text = |v: V| v.as_str().map(str::to_string);
    PageRow {
        id: text(p.get("id")),
        cc: text(p.get("cc")),
        bcc: text(p.get("bcc")),
        type_: text(data.get("type")).unwrap_or_default(),
        link: text(data.get("link")).unwrap_or_default(),
    }
}

/// fix-member-sync-scope: 팀장이 아닌 멤버의 동기화 범위. 행은 `cc` 가 `ccs` 에 있거나(사이트 통째 — homeCc 와 사이트 멤버 행의 bcc = cc),
/// `bcc` 가 `bccs` 에 있거나(페이지 — 멤버 행의 bcc 와 그 페이지 type 의 목록 · 상세 두 꼴), `ref` 가 `refs` 에 있으면(페이지 bcc 가 없는 작업 ·
/// 대화 — 그 페이지 행의 id 와 `hashId(팀 + cc + pathname)` · `hashId(팀 + 대문자 cc + pathname)`) 범위 안이다
pub struct SyncScope {
    pub ccs: Vec<String>,
    pub bccs: Vec<String>,
    pub refs: Vec<String>,
}

/// `await memberSyncScope(env, zone, address, team, homeCc)`
pub async fn member_sync_scope(env: &Env, zone: &V, address: &V, team: &V, home_cc: &V) -> R<SyncScope> {
    let db = format!("commerce_logis_{}_items", zone.to_str());
    let rows = js::query_bound(env, &db, SQL_MEMBER_BCCS, &[address.clone(), team.clone()]).await?;
    let mut member: Vec<String> = Vec::new();
    for row in rows.items() {
        if let Some(b) = row.get("bcc").as_str() {
            if !b.is_empty() {
                push_unique(&mut member, b.to_string());
            }
        }
    }

    let mut ccs = vec![home_cc.to_str()];
    for b in &member {
        push_unique(&mut ccs, b.clone());
    }

    let pages: Vec<PageRow> = js::query_bound(env, &db, SQL_TEAM_PAGES, &[team.clone()]).await?.items().iter().map(page_row).collect();

    // 페이지 멤버 행의 bcc 는 그 페이지 행의 bcc(목록이면 hashId(type + cc), 상세면 hashId(type + 대문자 cc)). 아이템은 상세를 읽은 뒤
    // 대문자 꼴로 옮겨 가므로(proxy item1) 같은 type 의 두 꼴을 함께 둔다 — memberPages 가 base.pages 의 type 칸을 두 꼴로 보는 것과 같다
    let mut bccs = member.clone();
    for p in &pages {
        if let (Some(cc), Some(bcc)) = (&p.cc, &p.bcc) {
            if member.contains(bcc) && !p.type_.is_empty() {
                push_unique(&mut bccs, hash_id(&V::from(p.type_.clone() + cc))?);
                push_unique(&mut bccs, hash_id(&V::from(p.type_.clone() + &logis_worker_common::jsv::to_upper(cc)))?);
            }
        }
    }

    // 작업(prompt 의 bcc 는 '')과 그 대화는 ref = hashId(팀 + task.cc + pathname)(상세면 대문자 cc)로, 상세 대화는 ref = 상세 페이지 id 로 페이지에 딸린다
    let mut refs: Vec<String> = Vec::new();
    for p in &pages {
        let Some(bcc) = &p.bcc else { continue };
        if !bccs.contains(bcc) {
            continue;
        }
        if let Some(id) = &p.id {
            push_unique(&mut refs, id.clone());
        }
        if let Some(cc) = &p.cc {
            if !p.link.is_empty() {
                let path = logis_worker_common::jsv::to_lower(p.link.split('?').next().unwrap_or(""));
                push_unique(&mut refs, hash_id(&V::from(team.to_str() + cc + &path))?);
                push_unique(&mut refs, hash_id(&V::from(team.to_str() + &logis_worker_common::jsv::to_upper(cc) + &path))?);
            }
        }
    }

    Ok(SyncScope { ccs, bccs, refs })
}

/// `scopeSql(scope, n)` — 팀장(None)이면 빈 글. ?n · ?n+1 · ?n+2 는 [`scope_params`] 의 JSON 배열 셋 (바인딩 수 한도 · 빈 목록을 피하려고 json_each)
pub fn scope_sql(scope: Option<&SyncScope>, n: usize) -> String {
    match scope {
        Some(_) => format!(
            r#" AND ("cc" IN (SELECT value FROM json_each(?{n})) OR "bcc" IN (SELECT value FROM json_each(?{})) OR "ref" IN (SELECT value FROM json_each(?{})))"#,
            n + 1,
            n + 2
        ),
        None => String::new(),
    }
}

/// `scopeParams(scope)`
pub fn scope_params(scope: Option<&SyncScope>) -> Vec<V> {
    let list = |l: &[String]| V::from(stringify_str(&V::from(l.iter().map(|s| V::from(s.as_str())).collect::<Vec<_>>())));
    match scope {
        Some(s) => vec![list(&s.ccs), list(&s.bccs), list(&s.refs)],
        None => Vec::new(),
    }
}

/// fix-member-app-link: 팀 공통 페이지 = 서비스 자신의 호스트(데스크톱 앱이 href 로 보내는 commerce · console, 그리고 logis.center).
/// 다른 `*.logis.center` 는 homeCc 라도 팀 공통이 아니다
pub fn team_common_host(hostname: &str) -> bool {
    ["logis.center", "commerce.logis.center", "console.logis.center"].contains(&hostname)
}

/// `await appExtractionLink(request.clone(), contentEncoding, contentType)` — fix-member-app-link: 앱 추출 봉투(fix-app-extraction 이 푸는 꼴)의
/// link 가 http(s) 주소면 그 주소(소문자 — appHref 와 같은 값), 아니면 None
pub fn app_extraction_link(content_encoding: &V, content_type: &V, body: &[u8]) -> Option<String> {
    if !logis_worker_common::loose_eq(content_encoding, &V::from("gzip")) || body.is_empty() {
        return None;
    }
    let ct = if content_type.truthy() { content_type.to_str() } else { String::new() };
    if logis_worker_common::jsv::index_of(&ct, "application/json") < 0 {
        return None;
    }
    let bytes = logis_worker_common::gzip::ungzip(&V::bytes(body.to_vec())).ok()?;
    let b = app_json(&logis_worker_common::gzip::decode_utf8(&bytes))?;
    let is_envelope = b.get("body").is_str() && ["image_extraction", "html_extraction", "document_extraction"].iter().any(|t| b.get("type").as_str() == Some(t));
    if !is_envelope {
        return None;
    }
    let u = url::Url::parse(&b.get("link").to_str()).ok()?;
    (u.scheme() == "http" || u.scheme() == "https").then(|| logis_worker_common::jsv::to_lower(u.as_str()))
}

const SQL_LINK_DETAIL: &str = r#"SELECT "bcc" FROM items WHERE "to" = ?1 AND "ref" = ?2 LIMIT 1"#;
const SQL_LINK_TEAM_PAGES: &str = r#"SELECT * FROM items WHERE "type" = 'pages' AND "to" = ?1 LIMIT 100"#;
const SQL_LINK_CC_PAGES: &str = r#"SELECT * FROM items WHERE "type" = 'pages' AND "cc" = ?1 LIMIT 100"#;
const SQL_LINK_MEMBER: &str = r#"SELECT * FROM items WHERE "type" = 'member' AND "from" = ?1 AND "to" = ?2 AND "bcc" = ?3 LIMIT 1"#;

/// `await linkMember(env, zone, address, team, link)` — fix-member-app-link: 앱 추출이 작업을 넣을 페이지(봉투의 link)의 멤버인지,
/// resolve(1132 · 1145 · 1256 · 1561)와 같은 방법으로: 서비스 호스트면 팀 공통. 아니면 cc = 그 호스트의 cc, 그 주소의 아이템(ref = hashId(팀 + cc + link))이
/// 있으면 bcc 가 같은 페이지(상세), 없으면 id = hashId(cc + pathname) 인 페이지 — 팀 페이지에서, 없으면 그 cc 의 페이지에서. 페이지가 없으면 멤버가 아니다.
/// 페이지 bcc = hashId(type + cc)(상세면 대문자 cc), 멤버 행의 bcc 가 그것이거나 cc(사이트 멤버)면 멤버
pub async fn link_member(env: &Env, zone: &V, address: &V, team: &V, link: &str) -> R<bool> {
    let Ok(u) = url::Url::parse(link) else { return Ok(false) };
    if team_common_host(u.host_str().unwrap_or("")) {
        return Ok(true);
    }
    let db = format!("commerce_logis_{}_items", zone.to_str());
    let cc = cc_of_host(&url_host(&u))?;
    let search = match u.query() {
        Some(q) if !q.is_empty() => format!("?{q}"),
        _ => String::new(),
    };
    let pathname = logis_worker_common::jsv::to_lower(u.path());
    let page_id = hash_id(&V::from(cc.clone() + &pathname))?;
    let item_id = hash_id(&V::from(team.to_str() + &cc + &logis_worker_common::jsv::to_lower(&(u.path().to_string() + &search))))?;

    let detail = js::query_bound(env, &db, SQL_LINK_DETAIL, &[team.clone(), V::from(item_id)]).await?;
    let detail_bcc = if detail.length()?.truthy() { Some(detail.gi(0)?.get("bcc")) } else { None };

    let mut page: Option<(V, bool)> = None;
    for (sql, key) in [(SQL_LINK_TEAM_PAGES, team.clone()), (SQL_LINK_CC_PAGES, V::from(cc.as_str()))] {
        if page.is_some() {
            break;
        }
        for p in js::query_bound(env, &db, sql, &[key]).await?.items() {
            let data = ungzip_json(&p.get("data")).unwrap_or(V::Undef);
            match &detail_bcc {
                Some(bcc) => {
                    if logis_worker_common::loose_eq(bcc, &p.get("bcc")) {
                        page = Some((data.get("type"), true));
                    }
                }
                None => {
                    if logis_worker_common::loose_eq(&V::from(page_id.as_str()), &p.get("id")) {
                        page = Some((data.get("type"), data.get("detail").truthy()));
                    }
                }
            }
        }
    }
    let Some((page_type, is_detail)) = page else { return Ok(false) };

    let cc_part = if is_detail { logis_worker_common::jsv::to_upper(&cc) } else { cc.clone() };
    let bcc = hash_id(&add(&page_type, &V::from(cc_part)))?;
    for b in [bcc, cc] {
        if js::query_bound(env, &db, SQL_LINK_MEMBER, &[address.clone(), team.clone(), V::from(b)]).await?.length()?.truthy() {
            return Ok(true);
        }
    }
    Ok(false)
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

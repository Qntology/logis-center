//! 스캔 아이템 반복의 첫 구역 (원본 3632~4198): 아이템 꼴 갖추기(no · index · link · id · data) · 기존 행과 합치기 ·
//! 페이지 셈(draft · count) · goods · 상태 · 운송장 번호, 주문이면 상품마다 tracking 행(try 3863~4196).
//!
//! 이 구역과 [`crate::item2`] 가 같이 쓰는 작은 도우미도 여기 둔다.

use logis_worker_common::gzip::{decode_utf8, gzip_json, ungzip};
use logis_worker_common::js::{self, Env};
use logis_worker_common::jsv::{js_trim, less_than, parse_float, parse_int, stringify_checked, to_lower, to_upper, tpl, Throw};
use logis_worker_common::util::{clean_number, crc32, digest, hash_id, merge_node, normalize_numeric_homoglyphs};
use logis_worker_common::{add, arr, console_log, date, loose_eq, obj, parse, stringify_str, R, V};

use crate::ctx::Ctx;
use crate::flow::Flow;
use crate::handler::{dec, prepare_bind};
use crate::llm::{deepinfra, gemini};
use crate::rules::{parse_status, semantic_prompt_system};
use crate::templates::*;

/// 원본의 `CLIENT_EMBEDDING` (서버는 벡터를 만들지 않는다)
const CLIENT_EMBEDDING: bool = true;

// ───────────────────────── 도우미 (item2 도 씀) ─────────────────────────

/// `x == "s"`
pub(crate) fn is(v: &V, s: &str) -> bool {
    loose_eq(v, &V::from(s))
}

/// `a + b + c …` (왼쪽부터, 문자열이 끼면 이어 붙임)
pub(crate) fn cat(parts: &[V]) -> V {
    let mut acc = parts[0].clone();
    for p in &parts[1..] {
        acc = add(&acc, p);
    }
    acc
}

/// `x ? x : d`
pub(crate) fn or(v: V, d: impl Into<V>) -> V {
    if v.truthy() {
        v
    } else {
        d.into()
    }
}

/// `hashId(x)`
pub(crate) fn hid(v: &V) -> R<V> {
    Ok(V::from(hash_id(v)?))
}

/// `crc32(hashId(x))`
pub(crate) fn crc_hid(v: &V) -> R<V> {
    Ok(V::Num(crc32(&hash_id(v)?)?))
}

/// `x.toUpperCase()` — 문자열이 아니면 원본처럼 TypeError
pub(crate) fn upper(v: &V) -> R<V> {
    match v {
        V::Str(s) => Ok(V::Str(to_upper(s))),
        V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'toUpperCase')", v.to_str()))),
        _ => Err(Throw::type_error("toUpperCase is not a function")),
    }
}

/// `o[key]++` · `o[key]--` (ToNumeric 뒤 더하기, o 가 undefined 면 TypeError)
pub(crate) fn bump(o: &V, key: &str, delta: f64) -> R<()> {
    let old = o.g(key)?;
    o.s_(key, V::Num(old.to_num() + delta))
}

/// `models[key] -= 1`
pub(crate) fn dec_key(models: &V, key: &V) -> R<()> {
    let old = models.gk(key)?;
    models.sk(key, dec(&old))
}

/// `team.data.base.pages[task.cc][type]`
pub(crate) fn page_slot(team: &V, task: &V, type_: &V) -> R<V> {
    team.g("data")?.g("base")?.g("pages")?.gk(&task.g("cc")?)?.gk(type_)
}

/// `team.data.base[type]`
pub(crate) fn base_slot(team: &V, type_: &V) -> R<V> {
    team.g("data")?.g("base")?.gk(type_)
}

/// JS 관계 비교 `a < b` (객체 · 배열은 먼저 문자열로 — ToPrimitive)
pub(crate) fn lt(a: &V, b: &V) -> Option<bool> {
    let prim = |v: &V| match v {
        V::Arr(_) | V::Obj(_) | V::Bytes(_) | V::Ext(_) => V::Str(v.to_str()),
        o => o.clone(),
    };
    less_than(&prim(a), &prim(b))
}

/// `a <= b` (= !(b < a), NaN 이면 false)
pub(crate) fn le(a: &V, b: &V) -> bool {
    lt(b, a) == Some(false)
}

/// `a >= b` (= !(a < b), NaN 이면 false)
pub(crate) fn ge(a: &V, b: &V) -> bool {
    lt(a, b) == Some(false)
}

/// `'…' + err` — Error.prototype.toString (메시지가 비면 이름만)
pub(crate) fn err_str(err: &Throw) -> String {
    if err.message.is_empty() {
        err.name.to_string()
    } else {
        err.to_string()
    }
}

/// `new URL(s)` — 못 읽으면 TypeError
fn new_url(href: &V) -> R<url::Url> {
    url::Url::parse(&href.to_str()).map_err(|_| Throw::type_error("Invalid URL string."))
}

/// `url.search` (물음표뿐이면 빈 글)
fn search(u: &url::Url) -> String {
    match u.query() {
        Some(q) if !q.is_empty() => format!("?{q}"),
        _ => String::new(),
    }
}

/// `markEmbedPending(obj)` — 객체면 embed = 0
fn mark_embed_pending(o: &V) {
    if o.truthy() && o.type_of() == "object" {
        o.put("embed", V::Num(0.0));
    }
}

/// SQL 에 이어 붙여도 되는 표 이름 (값처럼 묶을 수 없다)
const KNOWN_TABLES: &[&str] = &["items", "sales", "event", "talks", "tracking", "pages", "users", "tasks"];

/// 고침: 유형 변수로 만든 표 이름은 아는 표일 때만 SQL 에 넣는다
pub(crate) fn table_name(v: &V) -> R<String> {
    match v {
        V::Str(s) if KNOWN_TABLES.contains(&s.as_str()) => Ok(s.clone()),
        _ => Err(Throw::type_error(format!("unknown table: {}", v.to_str()))),
    }
}

/// `semantic.trim()` — 문자열이 아니면 원본처럼 TypeError
fn trim_text(v: &V, name: &str) -> R<V> {
    match v {
        V::Str(s) => Ok(V::from(js_trim(s).to_string())),
        V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'trim')", v.to_str()))),
        _ => Err(Throw::type_error(format!("{name}.trim is not a function"))),
    }
}

/// `parseFloat(t[k] ? t[k] : 0)`
fn pf(t: &V, k: &str) -> R<V> {
    let v = or(t.g(k)?, 0);
    Ok(V::Num(parse_float(&v.to_str())))
}

// ───────────────────────── 구역 ─────────────────────────

/// 3632~4198
pub async fn run(ctx: &mut Ctx, env: &Env) -> R<Flow> {
    let task = ctx.task.clone();
    let team = ctx.team.clone();
    let now = ctx.now.clone();
    let zone = ctx.zoneRegion.clone();

    ctx.item = ctx.items.gk(&ctx.i)?;

    if !ctx.item.g("id")?.truthy() {
        return Ok(Flow::Continue);
    }
    if !ctx.page.g("type")?.truthy() {
        return Ok(Flow::Continue);
    }

    let item = ctx.item.clone();
    if ctx.isDetail.truthy() {
        item.s_("link", task.g("link")?)?;
    }
    item.s_("type", ctx.page.g("type")?)?;
    item.s_("no", clean_number(&V::from(item.g("id")?.to_str()))?)?;
    item.s_("index", crc_hid(&cat(&[item.g("type")?, team.g("id")?, item.g("no")?]))?)?;

    // try 3654~3674: 링크를 경로 + 검색어 소문자로, 상세면 detail.page · detail.id (예외는 삼킴)
    let _ = link_and_detail(ctx, &task);

    ctx.itemType = V::from("");
    let t = item.g("type")?;
    if is(&t, "tracking") {
        ctx.itemType = V::from("tracking");
    } else if is(&t, "goods") || is(&t, "order") {
        ctx.itemType = V::from("sales");
    } else if is(&t, "event") || is(&t, "coupon") {
        ctx.itemType = V::from("event");
    } else {
        return Ok(Flow::Continue);
    }

    let reg = item.g("registration_date")?;
    if reg.truthy() {
        console_log!("item.registration_date", reg.clone());
        item.s_("created_at", V::Num(date::get_time(&reg)))?;
    } else {
        item.s_("created_at", now.clone())?;
    }

    item.s_("updated_at", V::Num(0.0))?;
    ctx.updated_at = V::Num(0.0);
    if ctx.isDetail.truthy() {
        ctx.updated_at = now.clone();
        item.s_("updated_at", now.clone())?;
    }

    item.s_("id", hid(&cat(&[team.g("id")?, item.g("index")?]))?)?;
    item.s_("flag", task.g("flag")?)?;
    item.s_("from", task.g("from")?)?;
    item.s_("to", task.g("to")?)?;
    item.s_("cc", task.g("cc")?)?;
    let cc = if ctx.isDetail.truthy() { upper(&task.g("cc")?)? } else { task.g("cc")? };
    item.s_("bcc", hid(&cat(&[item.g("type")?, cc]))?)?;
    item.s_("ref", hid(&cat(&[team.g("id")?, task.g("cc")?, item.g("link")?]))?)?;

    let data = obj! {
        "id" => item.g("id")?,
        "no" => or(item.g("no")?, ""),
        "title" => or(item.g("title")?, ""),
        "link" => item.g("link")?,
        "time" => now.clone(),
        "origin" => or(task.g("origin")?, ""),
    };
    item.s_("data", data)?;

    item.s_("digest", V::from(""))?;
    if item.g("title")?.truthy() {
        item.s_("digest", V::from(digest(&item.g("title")?)?))?;
    }

    // 고침: 3741 · 3744 의 값은 SQL 글에 잇지 않고 묶는다 (표 이름은 아는 표만)
    let db = tpl(T3741, &[zone.clone(), ctx.itemType.clone()]);
    let table = table_name(&ctx.itemType)?;
    let sql = format!("SELECT * FROM {table} WHERE \"id\" = ?1 AND \"index\" = ?2 AND \"to\" = ?3 AND \"cc\" = ?4 AND \"created_at\" < ?5 LIMIT 1");
    ctx.results = js::query_bound(env, &db, &sql, &[item.g("id")?, item.g("index")?, task.g("to")?, task.g("cc")?, now.clone()]).await?;

    if loose_eq(&ctx.results.length()?, &V::Num(0.0)) {
        let db = tpl(T3744, &[zone.clone(), ctx.itemType.clone()]);
        let sql = format!("SELECT * FROM {table} WHERE \"to\" = ?1 AND \"cc\" = ?2 AND \"ref\" = ?3 AND \"created_at\" < ?4 LIMIT 1");
        ctx.results = js::query_bound(env, &db, &sql, &[task.g("to")?, task.g("cc")?, item.g("ref")?, now.clone()]).await?;
    }

    console_log!("add results.length", item.g("type")?, ctx.results.length()?);
    console_log!("item.updated_at", if item.g("updated_at")?.truthy() { "true" } else { "false" });
    console_log!(
        "team.data.base.pages[task.cc][item.type]",
        stringify_str(&team.g("data")?.g("base")?.g("pages")?.gk(&task.g("cc")?)?)
    );

    if ctx.results.length()?.truthy() {
        ctx._item = ctx.results.gi(0)?;

        // try 3758~3782
        if let Err(err) = absorb_stored(ctx, &task, &team) {
            console_log!("ungzip err", err, ctx._item.g("data")?.type_of());
            ctx._item.del("data")?;
        }

        if ctx.item.g("status")?.truthy() && ctx._item.g("status")?.truthy() {
            ctx._item.del("status")?;
        }

        ctx.item = merge_node(&ctx._item, &ctx.item);
        // 고침: mergeNode 는 새 객체를 만드니 items[i] 에도 다시 넣는다 (원본은 items[i] 가 옛 객체로 남았다)
        ctx.items.sk(&ctx.i, ctx.item.clone())?;

        // 고침: 값은 묶는다
        let db = tpl(T3793, &[zone.clone()]);
        let sql = "SELECT * FROM items WHERE \"id\" = ?1 AND \"to\" = ?2 AND \"cc\" = ?3 AND \"created_at\" < ?4 LIMIT 1";
        ctx.results = js::query_bound(env, &db, sql, &[ctx._item.g("id")?, task.g("to")?, task.g("cc")?, now.clone()]).await?;

        if ctx.results.length()?.truthy() {
            ctx._item = ctx.results.gi(0)?;

            console_log!("_item.updated_at", ctx._item.g("updated_at")?);
            console_log!("item.updated_at", ctx.updated_at.clone());

            if !ctx._item.g("updated_at")?.truthy() && ctx.updated_at.truthy() {
                let type_ = ctx.item.g("type")?;
                bump(&page_slot(&team, &task, &type_)?, "draft", -1.0)?;
                let type_ = ctx.item.g("type")?;
                bump(&page_slot(&team, &task, &type_)?, "count", 1.0)?;
            }
        }
    } else {
        let type_ = ctx.item.g("type")?;
        if ctx.updated_at.truthy() {
            bump(&page_slot(&team, &task, &type_)?, "count", 1.0)?;
        } else {
            bump(&page_slot(&team, &task, &type_)?, "draft", 1.0)?;
        }
        let type_ = ctx.item.g("type")?;
        bump(&base_slot(&team, &type_)?, "count", 1.0)?;
    }

    let item = ctx.item.clone();
    let goods = item.g("goods")?;
    ctx.goods = if goods.truthy() { goods.deep_clone() } else { V::new_arr() };
    item.del("goods")?;

    // try 3826~3834: 앞에 {} 하나 (주문 자체의 tracking 행).
    // 고침: 배열이 아니면 먼저 배열로 감싼다(비었으면 빈 배열) — 원본은 문자열이면 unshift 가 없어 던지고(기록만) 그 글자마다
    // tracking 행을 만들었고, length 없는 객체 하나는 [] 로 버렸다
    if !ctx.goods.is_arr() {
        ctx.goods = if ctx.goods.truthy() { arr![ctx.goods.clone()] } else { V::new_arr() };
    }
    if let V::Arr(a) = &ctx.goods {
        a.borrow_mut().insert(0, V::new_obj());
    }

    let currency = item.g("currency")?;
    item.s_("currency", if currency.truthy() { upper(&currency)? } else { V::from("") })?;

    let quantity = item.g("quantity")?;
    item.s_("quantity", if quantity.truthy() { V::Num(parse_int(&quantity.to_str())) } else { V::Num(0.0) })?;

    item.s_("semantic", item.g("title")?)?;
    item.s_("started_at", or(item.g("manufacture_date")?, 0))?;
    item.s_("expired_at", or(item.g("expiration_date")?, 0))?;

    let status = item.g("status")?;
    let code = if status.truthy() { parse_status(&status) } else { V::Num(0.0) };
    item.s_("status", code.clone())?;
    ctx.statusCode = code;

    let tracking_number = item.g("tracking_number")?;
    if tracking_number.truthy() {
        ctx.tracking_number = normalize_numeric_homoglyphs(&tracking_number);
        ctx.tracking_number = clean_number(&ctx.tracking_number)?;
        item.s_("tracking", crc_hid(&cat(&[V::from("tracking"), team.g("id")?, ctx.tracking_number.clone()]))?)?;
    }

    // try 3863~4196
    if let Err(err) = order_tracking(ctx, env).await {
        console_log!("err item.type == \"order\" && item.tracking_number", err);
    }

    Ok(Flow::Next)
}

/// try 3654~3674 의 몸통
fn link_and_detail(ctx: &mut Ctx, task: &V) -> R<()> {
    // 안쪽 try: item.link 그대로, 안 되면 task.origin 을 앞에 붙여서
    let link = ctx.item.g("link")?;
    if relink(ctx, &link).is_err() {
        let href = add(&task.g("origin")?, &ctx.item.g("link")?);
        relink(ctx, &href)?;
    }

    // 고침: detail 이 객체일 때만 (상세면 앞(3342)에서 detail = null 이라 원본은 여기서 던지고 삼켰다)
    if ctx.isDetail.truthy() && ctx.detail.truthy() && ctx.detail.type_of() == "object" {
        if !ctx.detail.g("page")?.truthy() {
            let pathname = ctx._url.ext_ref::<url::Url>().map_or(V::Undef, |u| V::from(u.path()));
            let id = hid(&cat(&[ctx.item.g("type")?, upper(&task.g("cc")?)?, pathname]))?;
            ctx.detail.s_("id", id.clone())?;
            ctx.detail.s_("page", id)?;
        }
    }
    Ok(())
}

/// `var _url = new URL(href); item.link = (_url.pathname + _url.search).toLowerCase()` — 못 읽으면 _url 은 그대로
fn relink(ctx: &mut Ctx, href: &V) -> R<()> {
    let u = new_url(href)?;
    let link = to_lower(&format!("{}{}", u.path(), search(&u)));
    ctx._url = V::ext(u);
    ctx.item.s_("link", V::from(link))
}

/// try 3758~3782 의 몸통: 저장된 행의 data 를 풀어 no · index · id 를 그쪽에 맞춘다
fn absorb_stored(ctx: &mut Ctx, task: &V, team: &V) -> R<()> {
    let stored = ctx._item.clone();
    let item = ctx.item.clone();

    ctx.decompressedJsonString = V::from(decode_utf8(&ungzip(&stored.g("data")?)?));
    stored.s_("data", parse(&ctx.decompressedJsonString)?)?;

    // item.no = item.data.no = _item.data.no
    let item_data = item.g("data")?;
    let no = stored.g("data")?.g("no")?;
    item_data.s_("no", no.clone())?;
    item.s_("no", no)?;

    item.s_("index", crc_hid(&cat(&[item.g("type")?, team.g("id")?, item.g("no")?]))?)?;
    item.s_("id", hid(&cat(&[team.g("id")?, item.g("index")?]))?)?;

    // 고침: 새 아이템에 선택자(item · node)가 없으면 저장된 것을 쓴다 (원본의 `item.data.item && !_item.data.item` 은
    // 방금 만든 item.data 에 item 이 없어 늘 거짓이었다)
    if !item.g("data")?.g("item")?.truthy() && stored.g("data")?.g("item")?.truthy() {
        item.g("data")?.s_("item", stored.g("data")?.g("item")?)?;
        item.g("data")?.s_("node", stored.g("data")?.g("node")?)?;
    }

    if stored.g("updated_at")?.truthy() {
        item.s_("bcc", hid(&cat(&[item.g("type")?, upper(&task.g("cc")?)?]))?)?;
        ctx.updated_at = ctx.now.clone();
        item.s_("updated_at", ctx.now.clone())?;
    }
    Ok(())
}

/// try 3863~4196 의 몸통: 주문에 운송장 번호가 있으면 상품마다 tracking 행
async fn order_tracking(ctx: &mut Ctx, env: &Env) -> R<()> {
    let item = ctx.item.clone();
    console_log!("item.type", item.g("type")?);

    if is(&item.g("type")?, "order") && ctx.goods.truthy() && ctx.goods.g("length")?.truthy() && item.g("tracking_number")?.truthy() {
        ctx.g = V::Num(0.0);
        while lt(&ctx.g, &ctx.goods.g("length")?) == Some(true) {
            // 원본의 continue 는 이 반복의 다음 상품으로 (g++)
            tracking_row(ctx, env).await?;
            ctx.g = V::Num(ctx.g.to_num() + 1.0);
        }
    }
    Ok(())
}

/// 상품 하나의 tracking 행 (3877 반복의 몸통). `Ok(())` 으로 일찍 끝나는 곳이 원본의 continue
async fn tracking_row(ctx: &mut Ctx, env: &Env) -> R<()> {
    let item = ctx.item.clone();
    let task = ctx.task.clone();
    let team = ctx.team.clone();
    let now = ctx.now.clone();
    let zone = ctx.zoneRegion.clone();

    ctx.good = ctx.goods.gk(&ctx.g)?.deep_clone();
    ctx.tracking = item.deep_clone();

    let tracking = ctx.tracking.clone();
    tracking.s_("type", V::from("tracking"))?;
    tracking.s_("no", ctx.tracking_number.clone())?;
    tracking.s_("index", item.g("tracking")?)?;

    let good = ctx.good.clone();
    if good.g("id")?.truthy() {
        ctx.no = clean_number(&V::from(good.g("id")?.to_str()))?;
        good.s_("no", ctx.no.clone())?;
        good.s_("index", crc_hid(&cat(&[V::from("goods"), team.g("id")?, good.g("no")?]))?)?;

        // 고침: 값은 묶는다
        let db = tpl(T3896, &[zone.clone()]);
        let sql = "SELECT * FROM sales WHERE \"type\" = 'goods' AND \"to\" = ?1 AND \"cc\" = ?2 AND \"created_at\" < ?3 LIMIT 1";
        ctx.results = js::query_bound(env, &db, sql, &[task.g("to")?, task.g("cc")?, now.clone()]).await?;
        if ctx.results.length()?.truthy() {
            tracking.s_("event", ctx.results.gi(0)?.g("event")?)?;
        }

        tracking.s_("goods", good.g("index")?)?;
        tracking.s_("id", hid(&cat(&[team.g("id")?, tracking.g("no")?, good.g("no")?]))?)?;
    } else {
        tracking.s_("id", hid(&cat(&[team.g("id")?, tracking.g("no")?]))?)?;
    }

    tracking.s_("digest", item.g("digest")?)?;
    tracking.s_("status", item.g("status")?)?;
    tracking.s_("order", item.g("index")?)?;
    for k in ["order_date", "payment_date", "payment_method", "payment_origin", "link", "sender_address", "sender_phone", "recipient_address", "recipient_phone"] {
        tracking.s_(k, item.g(k)?)?;
    }
    tracking.s_(
        "data",
        obj! {
            "id" => item.g("id")?,
            "link" => item.g("link")?,
            "time" => now.clone(),
            "origin" => or(task.g("origin")?, ""),
        },
    )?;

    let pages = team.g("data")?.g("base")?.g("pages")?;
    if !pages.gk(&task.g("cc")?)?.truthy() {
        team.g("data")?.g("base")?.g("pages")?.sk(&task.g("cc")?, V::new_obj())?;
    }
    let type_ = tracking.g("type")?;
    if !page_slot(&team, &task, &type_)?.truthy() {
        let type_ = tracking.g("type")?;
        team.g("data")?.g("base")?.g("pages")?.gk(&task.g("cc")?)?.sk(&type_, obj! { "draft" => 0, "count" => 0 })?;
    }

    // 고침: 값은 묶는다
    let db = tpl(T3950, &[zone.clone()]);
    let sql = "SELECT * FROM tracking WHERE \"index\" = ?1 AND \"to\" = ?2 AND \"created_at\" < ?3 LIMIT 1";
    ctx.results = js::query_bound(env, &db, sql, &[tracking.g("index")?, task.g("to")?, now.clone()]).await?;

    if ctx.results.length()?.truthy() {
        ctx._tracking = ctx.results.gi(0)?.deep_clone();
        let stored = ctx._tracking.clone();
        ctx.decompressedJsonString = V::from(decode_utf8(&ungzip(&stored.g("data")?)?));
        stored.s_("data", parse(&ctx.decompressedJsonString)?)?;
        stored.del("ref")?;
        ctx.tracking = merge_node(&stored, &ctx.tracking);
    } else {
        // 처음 저장할 때: 자연어로 바꾼 뒤 벡터로 (지금은 클라이언트가 임베딩)
        let t = ctx.tracking.clone();
        let content = obj! {
            "size" => or(t.g("size")?, ""),
            "currency" => or(t.g("currency")?, ""),
            "carrier" => or(t.g("carrier")?, ""),
            "shipping_fee" => t.g("shipping_fee")?.truthy(),
            "shipping_method" => or(t.g("shipping_method")?, ""),
            "fulfillment_service" => or(t.g("fulfillment_service")?, ""),
            "stock_keeping_unit" => or(t.g("stock_keeping_unit")?, ""),
            "bundle_shipping" => t.g("bundle_shipping")?.truthy(),
            "used" => t.g("used")?.truthy(),
            "lease" => t.g("lease")?.truthy(),
            "rental" => t.g("rental")?.truthy(),
            "refurbish" => t.g("refurbish")?.truthy(),
            "tax_included" => t.g("tax_included")?.truthy(),
            "sender_address" => t.g("sender_address")?,
            "recipient_address" => t.g("recipient_address")?,
            "payment_method" => t.g("payment_method")?,
            "payment_origin" => t.g("payment_origin")?,
        };
        ctx.content = V::from(stringify_checked(&content)?.unwrap_or_default());

        // 고침: `var semantic` 은 상품마다 undefined 에서 시작한다 (원본은 앞 상품의 값이 남아 `!semantic` 이 Gemini 대체를 건너뛰었다)
        ctx.semantic = V::Undef;
        ctx.system = V::from(js_trim(&semantic_prompt_system(&ctx.language).to_str()).to_string());

        if ctx.models.g("deepinfra")?.truthy() {
            ctx.semantic =
                deepinfra(&ctx.deepinfra, &V::from("openai/gpt-oss-20b"), &ctx.system, &ctx.content, &obj! { "temperature" => 1 }, &V::Undef)
                    .await?;
            dec_key(&ctx.models, &V::from("deepinfra"))?;
        }

        if !ctx.semantic.truthy() && ctx.gemini_llm_api.truthy() {
            ctx.semantic =
                gemini(&ctx.gemini_llm_api, &ctx.gemini_llm_model, &ctx.system, &ctx.content, &obj! { "temperature" => 1 }, &V::Undef).await?;
            let key = cat(&[ctx.gemini_llm_api.clone(), V::from("-"), ctx.gemini_llm_model.clone()]);
            dec_key(&ctx.models, &key)?;
        }

        if !ctx.semantic.truthy() {
            ctx.fallback = V::from("semantic overflow");
            return Ok(());
        }

        ctx.metadata = obj! {
            "type" => item.g("type")?,
            "from" => item.g("from")?,
            "to" => item.g("to")?,
            "cc" => item.g("cc")?,
            "bcc" => item.g("bcc")?,
            "ref" => ctx.pageId.clone(),
        };

        if CLIENT_EMBEDDING {
            mark_embed_pending(&ctx.tracking);

            let type_ = ctx.tracking.g("type")?;
            bump(&page_slot(&team, &task, &type_)?, "draft", 1.0)?;
            bump(&base_slot(&team, &item.g("type")?)?, "count", 1.0)?;

            console_log!("[EMBED-PENDING] tracking relay delegated to Client App:", ctx.tracking.g("id")?);
        } else {
            // 고침: `var embeddings` 도 상품마다 undefined 에서 시작한다
            ctx.embeddings = V::Undef;
            if ctx.models.g("cloudflare")?.truthy() {
                let r = env.ai_run("@cf/google/embeddinggemma-300m", &obj! { "text" => arr![ctx.semantic.clone()] }).await?;
                ctx.embeddings = r.g("data")?;
                ctx.s_VectorizeVector = arr![obj! {
                    "id" => item.g("id")?,
                    "values" => ctx.embeddings.gi(0)?,
                    "metadata" => ctx.metadata.clone(),
                }];
                dec_key(&ctx.models, &V::from("cloudflare"))?;
            }

            if !ctx.embeddings.truthy() && ctx.models.g("deepinfra")?.truthy() {
                // 고침: semantic.trim() (원본은 .tirm() 오타라 인자를 셀 때 TypeError 였다)
                let text = trim_text(&ctx.semantic, "semantic")?;
                ctx.embeddings = deepinfra(&ctx.deepinfra, &V::from("google/embeddinggemma-300m"), &V::from(""), &text, &V::Undef, &V::Undef).await?;
                // embeddings.map((values, i) => ({ id: item.id, values, metadata }))
                let list = match &ctx.embeddings {
                    V::Arr(a) => a.borrow().clone(),
                    V::Undef | V::Null => {
                        return Err(Throw::type_error(format!("Cannot read properties of {} (reading 'map')", ctx.embeddings.to_str())));
                    }
                    _ => return Err(Throw::type_error("embeddings.map is not a function")),
                };
                let mut vectors = Vec::with_capacity(list.len());
                for values in list {
                    vectors.push(obj! { "id" => item.g("id")?, "values" => values, "metadata" => ctx.metadata.clone() });
                }
                ctx.s_VectorizeVector = V::from(vectors);
                dec_key(&ctx.models, &V::from("deepinfra"))?;
            }

            if !ctx.embeddings.truthy() {
                ctx.fallback = V::from("embeddings overflow");
                return Ok(());
            }

            let type_ = ctx.tracking.g("type")?;
            bump(&page_slot(&team, &task, &type_)?, "draft", 1.0)?;
            bump(&base_slot(&team, &item.g("type")?)?, "count", 1.0)?;

            let index = tpl(T4071, &[ctx.vectorRegion.clone(), ctx.itemType.clone()]);
            env.vectorize_upsert(&index, &ctx.s_VectorizeVector).await?;
        }
    }

    let tracking = ctx.tracking.clone();
    ctx.arr = gzip_json(&tracking.g("data")?)?;

    console_log!("tracking", stringify_str(&tracking));

    let t = &tracking;
    let slot = ctx.statements.g(&tpl(T4079, &[zone.clone()]))?;
    let stmt = prepare_bind(
        env,
        &tpl(T4080, &[zone.clone()]),
        T4080_1[0],
        &[
            t.g("id")?,
            t.g("type")?,
            t.g("from")?,
            t.g("to")?,
            t.g("cc")?,
            t.g("bcc")?,
            t.g("ref")?,
            ctx.arr.clone(),
            t.g("created_at")?,
            t.g("index")?,
            or(t.g("event")?, 0),
            or(t.g("goods")?, 0),
            or(t.g("order")?, 0),
            t.g("status")?,
            or(t.g("no")?, ""),
            or(t.g("sender_address")?, ""),
            or(t.g("sender_phone")?, ""),
            or(t.g("recipient_address")?, ""),
            or(t.g("recipient_phone")?, ""),
            pf(t, "width")?,
            pf(t, "height")?,
            pf(t, "length")?,
            pf(t, "weight")?,
            pf(t, "carrier")?,
            pf(t, "shipping_fee")?,
            or(t.g("shipping_method")?, ""),
            pf(t, "shipping_duration")?,
            pf(t, "shipping_date")?,
            pf(t, "delivery_date")?,
            pf(t, "order_date")?,
            pf(t, "payment_date")?,
            or(t.g("payment_method")?, ""),
            or(t.g("payment_origin")?, ""),
            or(t.g("payment_number")?, ""),
            pf(t, "bundle_shipping")?,
        ],
    )?;
    slot.push(stmt)?;

    let slot = ctx.statements.g(&tpl(T4160, &[zone.clone()]))?;
    let stmt = prepare_bind(
        env,
        &tpl(T4161, &[zone]),
        T4161_1[0],
        &[
            t.g("id")?,
            t.g("type")?,
            t.g("from")?,
            t.g("to")?,
            t.g("cc")?,
            t.g("bcc")?,
            t.g("ref")?,
            t.g("digest")?,
            ctx.arr.clone(),
            now,
            ctx.updated_at.clone(),
        ],
    )?;
    slot.push(stmt)?;

    Ok(())
}

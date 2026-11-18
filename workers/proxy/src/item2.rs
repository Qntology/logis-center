//! 스캔 아이템 반복의 둘째 구역 (원본 4199~4555): condition 표시 · 저장된 행과 합치기 · 팀 범위(min · max, try 4255~4435) ·
//! 관련 유형마다 Relay 질의로 relates 모으기(try 4481~4546, 실패는 console 표에).

use logis_worker_common::js::{self, Env};
use logis_worker_common::jsv::{index_of, strict_eq, tpl, Throw};
use logis_worker_common::tables::CENTER_REGION;
use logis_worker_common::util::{hash_id, merge_node};
use logis_worker_common::{console_log, loose_eq, obj, R, V};

use crate::ctx::Ctx;
use crate::flow::Flow;
use crate::handler::{prepare_bind, select};
use crate::item1::{cat, err_str, ge, le, lt};
use crate::rules::{related, relay};
use crate::templates::*;

/// 팀 범위를 넓히는 값들 (원본 4256~4432 의 순서)
const RANGE: &[&str] = &[
    "price",
    "quantity",
    "width",
    "height",
    "length",
    "weight",
    "shipping_fee",
    "shipping_duration",
    "sale_price",
    "supply_price",
    "low_stock_threshold",
    "discount",
    "min_order_amount",
    "max_discount_amount",
    "usage_limit",
    "usage_per",
    "started_at",
    "expired_at",
];

/// `x.indexOf(needle)` — 문자열은 부분 글, 배열은 같은 원소(===). 그 밖은 원본처럼 TypeError
fn index_of_v(v: &V, needle: &str) -> R<f64> {
    match v {
        V::Str(s) => Ok(index_of(s, needle) as f64),
        V::Arr(a) => {
            let n = V::from(needle);
            Ok(a.borrow().iter().position(|x| strict_eq(x, &n)).map_or(-1.0, |p| p as f64))
        }
        V::Undef | V::Null => Err(Throw::type_error(format!("Cannot read properties of {} (reading 'indexOf')", v.to_str()))),
        _ => Err(Throw::type_error("item.condition.indexOf is not a function")),
    }
}

/// 4199~4554
pub async fn run(ctx: &mut Ctx, env: &Env) -> R<Flow> {
    let task = ctx.task.clone();
    let now = ctx.now.clone();
    let zone = ctx.zoneRegion.clone();

    let item = ctx.item.clone();
    if item.g("condition")?.truthy() {
        if index_of_v(&item.g("condition")?, "used")? > -1.0 {
            item.s_("used", V::Num(1.0))?;
        }
        if index_of_v(&item.g("condition")?, "lease")? > -1.0 {
            item.s_("lease", V::Num(2.0))?;
        }
        if index_of_v(&item.g("condition")?, "rental")? > -1.0 {
            item.s_("rental", V::Num(3.0))?;
        }
        if index_of_v(&item.g("condition")?, "refurbish")? > -1.0 {
            item.s_("refurbish", V::Num(4.0))?;
        }
    }

    let db = tpl(T4218, &[zone.clone(), ctx.itemType.clone()]);
    let sql = tpl(T4218_1, &[ctx.itemType.clone(), item.g("id")?, task.g("to")?, now.clone()]);
    ctx.results = select(env, &db, &sql).await?;

    if ctx.results.length()?.truthy() {
        absorb(ctx)?;
    } else {
        let db = tpl(T4235, &[zone.clone(), ctx.itemType.clone()]);
        let sql = tpl(T4235_1, &[ctx.itemType.clone(), item.g("index")?, task.g("to")?, task.g("cc")?, now.clone()]);
        ctx.results = select(env, &db, &sql).await?;

        if ctx.results.length()?.truthy() {
            absorb(ctx)?;
        }
    }

    // try 4255~4435
    if let Err(err) = widen_range(ctx) {
        console_log!("err team.data.base", err);
    }

    ctx.progress = V::new_obj();
    ctx.relates = V::new_obj();
    ctx.related = related(&ctx.item.g("type")?);

    if ctx.related.length()?.truthy() {
        ctx.r = V::Num(0.0);
        while lt(&ctx.r, &ctx.related.g("length")?) == Some(true) {
            let relay_out = relay(&ctx.related.gk(&ctx.r)?, &ctx.item)?;
            ctx.query = relay_out.g("query")?;
            ctx.merge = relay_out.g("merge")?;

            if ctx.query.truthy() && ctx.merge.truthy() {
                // try 4481~4546
                if let Err(err) = relate(ctx, env).await {
                    // 원본 그대로: type 은 앞에서 던졌으면 이전 회차의 값
                    let log = cat(&[V::from("tracking inner err"), ctx.type_.clone(), V::from(err_str(&err))]);
                    let stmt = prepare_bind(env, CENTER_REGION, T4534[0], &[V::from(hash_id(&V::Undef)?), task.g("bcc")?, log, now.clone()])?;
                    js::run(&stmt).await?;
                }
            }

            ctx.r = V::Num(ctx.r.to_num() + 1.0);
        }
    }

    console_log!("Object.keys(relates).length", ctx.relates.keys_checked()?.len());

    Ok(Flow::Next)
}

/// 저장된 행을 아이템 밑에 깐다: `delete _item.ref` · 새 상태가 있으면 옛 상태를 빼고 `item = mergeNode(_item, item)`
fn absorb(ctx: &mut Ctx) -> R<()> {
    ctx._item = ctx.results.gi(0)?;
    ctx._item.del("ref")?;

    if ctx.item.g("status")?.truthy() && ctx._item.g("status")?.truthy() {
        ctx._item.del("status")?;
    }

    // 원본 그대로: 새 객체로 바뀌어 items[i] 와는 갈라진다
    ctx.item = merge_node(&ctx._item, &ctx.item);
    Ok(())
}

/// try 4255~4435 의 몸통: 아이템 값으로 `team.data.base[item.type][f].min · max` 를 넓힌다 (중간에 던지면 나머지는 건너뜀)
fn widen_range(ctx: &mut Ctx) -> R<()> {
    let team = ctx.team.clone();
    let item = ctx.item.clone();
    let node = |f: &str| -> R<V> { team.g("data")?.g("base")?.gk(&item.g("type")?)?.g(f) };

    for f in RANGE {
        if le(&item.g(f)?, &node(f)?.g("min")?) {
            node(f)?.s_("min", item.g(f)?)?;
        }
        if ge(&item.g(f)?, &node(f)?.g("max")?) {
            node(f)?.s_("max", item.g(f)?)?;
        }
    }
    Ok(())
}

/// try 4481~4546 의 몸통: Relay 질의의 첫 줄로 관련 행을 찾아 relates[type] 에 둔다
async fn relate(ctx: &mut Ctx, env: &Env) -> R<()> {
    if !ctx.query.g("length")?.truthy() {
        return Ok(());
    }

    ctx.table = ctx.query.gi(0)?.g("table")?;
    ctx.type_ = ctx.query.gi(0)?.g("type")?;
    ctx.column = ctx.query.gi(0)?.g("column")?;
    ctx.column_value = ctx.query.gi(0)?.g("value")?;
    ctx.status = ctx.query.gi(0)?.g("status")?;

    let db = tpl(T4503, &[ctx.zoneRegion.clone(), ctx.table.clone()]);
    let stmt = env.prepare(&db, &tpl(T4504, &[ctx.table.clone(), ctx.column.clone()]))?;
    let stmt = js::bind(&stmt, &[ctx.type_.clone(), ctx.column_value.clone(), ctx.team.g("id")?, ctx.item.g("cc")?, ctx.now.clone()])?;
    ctx.results = js::all(&stmt).await?.g("results")?;

    if loose_eq(&ctx.results.length()?, &V::Num(0.0)) {
        // draft 상태 — 없으면 추가해야 함
        ctx.updated_at = V::Num(0.0);
    }

    let entry = obj! {
        "query" => ctx.query.clone(),
        "merge" => ctx.merge.clone(),
        "rows" => ctx.results.clone(),
        "type" => ctx.related.gk(&ctx.r)?,
    };
    ctx.relates.sk(&ctx.type_, entry)?;
    Ok(())
}

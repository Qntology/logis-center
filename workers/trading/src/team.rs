//! 원본 370~454: `resolveTeam` — 팀 판정을 commerce Worker 에 위임 (hash 단위 5분 캐시, `caches.default` 는 JS 그대로).

use js_sys::{Array, Function, Object, Reflect};
use logis_worker_common::js::{self, Env, Js};
use logis_worker_common::jsv::{loose_eq, stringify_str, to_lower, tpl, Throw, R, V};
use logis_worker_common::{console_log, eth, obj, platform};
use wasm_bindgen::{JsCast, JsValue};

use crate::jsrt;
use crate::logic::hash_id;
use crate::templates::*;

/// 370
pub const COMMERCE_HOST: &str = "https://commerce.logis.center";

fn global(name: &str) -> R<JsValue> {
    Reflect::get(&js_sys::global(), &JsValue::from_str(name)).map_err(js::to_throw)
}

/// `new <ctor>(...args)` (전역 생성자)
fn construct(ctor: &str, args: &[JsValue]) -> R<JsValue> {
    let f: Function = global(ctor)?.dyn_into().map_err(|_| Throw::type_error(format!("{ctor} is not a constructor")))?;
    let list = Array::new();
    for a in args {
        list.push(a);
    }
    Reflect::construct(&f, &list).map_err(js::to_throw)
}

fn plain(pairs: &[(&str, JsValue)]) -> JsValue {
    let o = Object::new();
    for (k, v) in pairs {
        let _ = Reflect::set(&o, &JsValue::from_str(k), v);
    }
    o.into()
}

/// 393~454: `resolveTeam(env, hash, token, address, requestedTo)` → `{ team, role }`
pub async fn resolve_team(_env: &Env, hash: &V, token: &V, address: &V, requested_to: &V) -> R<V> {
    let own = V::from(hash_id(address)?);
    if !requested_to.truthy() || !eth::is_address(requested_to) {
        return Ok(obj! { "team" => own, "role" => "owner" });
    }
    // isAddress 가 참이면 글이다
    let to = V::from(to_lower(&requested_to.to_str()));
    if loose_eq(&to, &own) {
        return Ok(obj! { "team" => own, "role" => "owner" });
    }
    let cache_key = construct("Request", &[JsValue::from_str(&tpl(T402, &[hash.clone()])), plain(&[("method", JsValue::from_str("GET"))])])?;
    let cache = V::ext(Js(Reflect::get(&global("caches")?, &JsValue::from_str("default")).map_err(js::to_throw)?));
    let r: R<Option<V>> = async {
        let hit = js::wait(js::call(&cache, "match", &[cache_key.clone()])?).await?;
        if !hit.is_undefined() && !hit.is_null() {
            let cached = js::response_json(&hit).await?;
            if cached.truthy() && cached.g("team")?.truthy() {
                if loose_eq(&cached.g("team")?, &to) {
                    return Ok(Some(obj! { "team" => to.clone(), "role" => "member" }));
                }
                // 고침 (fix-trading-team-mismatch): 확인된 팀이 본인 팀일 때만 본인 팀. 다른 팀이면 강등하지 않고
                // (팀을 옮겼을 수 있으니) 캐시를 버리고 commerce 에 다시 묻는다
                if loose_eq(&cached.g("team")?, &own) {
                    return Ok(Some(obj! { "team" => own.clone(), "role" => "owner" }));
                }
                console_log!(tpl(T412_FIXED, &[to.clone(), cached.g("team")?]));
            }
        }
        Ok(None)
    }
    .await;
    match r {
        Ok(Some(v)) => return Ok(v),
        Ok(None) => {}
        Err(err) => console_log!("[TRADING] team cache read err", err),
    }
    let mut resolved_team = V::from("");
    let r: R<()> = async {
        let probe = jsrt::search_params(&[
            ("origin", COMMERCE_HOST.to_string()),
            ("created_at", V::Num(platform::now()).to_str()),
            ("hash", hash.to_str()),
            ("token", token.to_str()),
            ("href", format!("{COMMERCE_HOST}/tracking")),
        ]);
        let url = tpl(T427, &[V::from(COMMERCE_HOST), V::from(probe)]);
        let res = js::fetch(&url, "GET", &[("Content-Type", "application/json")], js::Body::None).await?;
        let body = js::response_json(&res).await?;
        if body.truthy() && body.g("session")?.truthy() && body.g("session")?.g("team")?.truthy() {
            resolved_team = V::from(to_lower(&body.g("session")?.g("team")?.to_str()));
        }
        Ok(())
    }
    .await;
    if let Err(err) = r {
        console_log!("[TRADING] team delegate err", err);
    }
    if !resolved_team.truthy() {
        console_log!(tpl(T439, &[to.clone()]));
        return Ok(obj! { "team" => "", "role" => "denied" });
    }
    let r: R<()> = async {
        let body = stringify_str(&obj! { "team" => resolved_team.clone() });
        let headers = plain(&[("Content-Type", JsValue::from_str("application/json")), ("Cache-Control", JsValue::from_str("max-age=300"))]);
        let response = construct("Response", &[JsValue::from_str(&body), plain(&[("headers", headers)])])?;
        js::wait(js::call(&cache, "put", &[cache_key.clone(), response])?).await?;
        Ok(())
    }
    .await;
    if let Err(err) = r {
        console_log!("[TRADING] team cache write err", err);
    }
    if loose_eq(&resolved_team, &to) {
        let role = if loose_eq(&resolved_team, &own) { "owner" } else { "member" };
        return Ok(obj! { "team" => to, "role" => role });
    }
    // 고침 (fix-trading-team-mismatch): 원본은 요청 팀 != 확인된 팀이면 본인 팀으로 강등했다 — 다른 팀의 멤버가 올린
    // 서류가 개인 팀에 저장되는, 위 주석이 막으려던 바로 그 오염. 확인된 팀이 본인 팀이면 본인 팀, 아니면 거부
    if loose_eq(&resolved_team, &own) {
        return Ok(obj! { "team" => own, "role" => "owner" });
    }
    console_log!(tpl(T452_FIXED, &[to, resolved_team]));
    Ok(obj! { "team" => "", "role" => "denied" })
}

/// 고침 (fix-trading-team-mismatch) 의 로그 글
const T412_FIXED: &[&str] = &["[TRADING] (cache) 요청 팀 '", "' != 확인된 팀 '", "'. commerce 에 다시 묻습니다."];
const T452_FIXED: &[&str] = &["[TRADING] 🚫 요청 팀 '", "' != 확인된 팀 '", "'. 거부합니다."];

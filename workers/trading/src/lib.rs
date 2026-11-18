//! trading-logis-center — 원본(github.com/sungu-che/trading-logis-center 65684e6)을 같은 동작으로 옮긴 것.
//! 경로는 `/` 하나: 세션(S3 · commerce 로 팀 확인) 뒤 GET(변경분 받기) · POST(최대 200건 넣기) · DELETE · OPTIONS.
//!
//! - `logic`   원본 1~353 (상수 · stableStringify · contentDigest · decodeData · 서식 · 허브 키 · ref)
//! - `team`    원본 370~454 (resolveTeam: commerce 위임 · caches.default)
//! - `handler` 원본 456~1079 (fetch)
//! - `jsrt`    원본이 기대는 JS 내장 함수의 의미 (parseInt · decodeURIComponent · atob · Uint8Array · pako 2.2 ungzip …)

pub mod ctx;
pub mod flow;
pub mod jsrt;
pub mod logic;
pub mod templates;

#[cfg(target_arch = "wasm32")]
pub mod handler;
#[cfg(target_arch = "wasm32")]
pub mod team;

#[cfg(target_arch = "wasm32")]
mod worker_main {
    use logis_worker_common::js::{req_from, resp_into, Env};
    use logis_worker_common::Throw;
    use wasm_bindgen::prelude::*;
    use worker::worker_sys::web_sys;

    /// 원본에서 처리기 밖으로 새는 예외 → 같은 종류의 JS 예외 (런타임이 500 으로 끝낸다)
    fn to_js_error(t: &Throw) -> JsValue {
        match t.name {
            "" => JsValue::from_str(&t.message),
            "TypeError" => js_sys::TypeError::new(&t.message).into(),
            "RangeError" => js_sys::RangeError::new(&t.message).into(),
            "SyntaxError" => js_sys::SyntaxError::new(&t.message).into(),
            "URIError" => js_sys::UriError::new(&t.message).into(),
            _ => js_sys::Error::new(&t.message).into(),
        }
    }

    /// `#[event(fetch)]` 대신: 원본은 `request.cf` 가 없으면 처리기가 던진다(500 응답이 아니라 예외). 그 경계를 지키려고
    /// 오류를 응답으로 바꾸지 않고 프라미스를 거절한다
    #[wasm_bindgen(js_name = fetch)]
    pub async fn fetch(req: web_sys::Request, env: worker::Env, _ctx: worker::worker_sys::Context) -> Result<JsValue, JsValue> {
        logis_worker_common::platform::install_panic_hook();
        // request.method 그대로 (worker::Method 는 모르는 메서드를 바꾼다)
        let method = req.method();
        let mut r = req_from(worker::Request::from(req)).await;
        r.method = method;
        match crate::handler::fetch(r, Env(env)).await {
            Ok(resp) => {
                let resp = resp_into(resp).map_err(|e| JsValue::from(js_sys::Error::new(&e.to_string())))?;
                Ok(web_sys::Response::from(resp).into())
            }
            Err(t) => Err(to_js_error(&t)),
        }
    }
}

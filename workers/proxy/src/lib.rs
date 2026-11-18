//! proxy-commerce-logis-center — 원본 `proxy/src/index.ts` 를 같은 동작으로 옮긴 것 (workers-rs).
//!
//! apac 크론이 gzip JSON(`{ now, id, ref, region, models, limits, gemini_llm_api, gemini_llm_model, deepinfra }`)을 POST 하면
//! 그 작업 행을 읽어 세 갈래로 처리한다: 운송장 이미지([`image`]), 화면 스캔([`scan`]: 목록 · 상세 페이지를 LLM 으로 구조화하고
//! 아이템 · 관계를 D1 · Vectorize 에), 자연어 검색([`search`]). 응답은 `{ models, limits }` JSON, 요청이 비거나 깨지면 418.

pub mod ctx;
pub mod dom;
pub mod flow;
pub mod pug;
pub mod rules;
pub mod templates;

#[cfg(target_arch = "wasm32")]
pub mod handler;
#[cfg(target_arch = "wasm32")]
pub mod image;
#[cfg(target_arch = "wasm32")]
pub mod item1;
#[cfg(target_arch = "wasm32")]
pub mod item2;
#[cfg(target_arch = "wasm32")]
pub mod item3;
#[cfg(target_arch = "wasm32")]
pub mod item4;
#[cfg(target_arch = "wasm32")]
pub mod llm;
#[cfg(target_arch = "wasm32")]
pub mod scan;
#[cfg(target_arch = "wasm32")]
pub mod search;

#[cfg(target_arch = "wasm32")]
mod worker_main {
    use logis_worker_common::js::Env;
    use worker::{event, Context, Headers, Request, Response, Result};

    #[event(fetch)]
    pub async fn fetch(mut req: Request, env: worker::Env, _ctx: Context) -> Result<Response> {
        logis_worker_common::platform::install_panic_hook();
        let body = req.bytes().await.unwrap_or_default();
        let (status, text, content_type) = crate::handler::fetch(body, Env(env)).await;
        let headers = Headers::new();
        headers.set("Content-Type", content_type)?;
        Ok(Response::ok(text)?.with_status(status).with_headers(headers))
    }
}

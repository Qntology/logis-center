//! trading-logis-center — 원본(github.com/sungu-che/trading-logis-center 65684e6)을 같은 동작으로 옮긴 것.
//! 경로는 `/` 하나: 세션(S3 · commerce 로 팀 확인) 뒤 GET(변경분 받기) · POST(최대 200건 넣기) · DELETE · OPTIONS.

pub mod ctx;
pub mod flow;
pub mod templates;

#[cfg(target_arch = "wasm32")]
pub mod handler;

#[cfg(target_arch = "wasm32")]
mod worker_main {
    use logis_worker_common::js::{req_from, resp_into, Env};
    use worker::{event, Context, Request, Response, Result};

    #[event(fetch)]
    pub async fn fetch(req: Request, env: worker::Env, _ctx: Context) -> Result<Response> {
        logis_worker_common::platform::install_panic_hook();
        let req = req_from(req).await;
        resp_into(crate::handler::fetch(req, Env(env)).await)
    }
}

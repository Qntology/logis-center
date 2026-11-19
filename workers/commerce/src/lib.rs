//! commerce-logis-center — 원본(github.com/sungu-che/commerce-logis-center, HEAD fd1e991 에서 짝 없는 `}else if` 한 곳만 고친 것)을
//! 같은 동작으로 옮긴 것. 길 나눔은 경로가 아니라 메서드 + 질의 값이다: 모든 요청이 세션(S3 · 사용자 · 팀)과 페이지 · 아이템 판별을 지나고,
//! PUT(벡터 · 멤버 · 대화) · DELETE(작업 · 로그아웃) · POST(작업 넣기 — proxy · apac 이 처리하는 tasks 행) · GET(동기화 JSON · 셸 HTML).

pub mod ctx;
pub mod flow;
pub mod templates;

#[cfg(target_arch = "wasm32")]
pub mod handler;
#[cfg(target_arch = "wasm32")]
pub mod session;
#[cfg(target_arch = "wasm32")]
pub mod resolve;
#[cfg(target_arch = "wasm32")]
pub mod put_delete;
#[cfg(target_arch = "wasm32")]
pub mod post;
#[cfg(target_arch = "wasm32")]
pub mod get;
#[cfg(target_arch = "wasm32")]
pub mod fixes;
pub mod render;

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

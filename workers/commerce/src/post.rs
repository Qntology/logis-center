//! (옮기는 중) 원본 1998~2454 (POST: 화면 · 이미지 · {query,vector} · URL 배열 · 프롬프트 → tasks 대기열)

use logis_worker_common::js::Env;
use logis_worker_common::web::Req;
use logis_worker_common::R;

use crate::ctx::Ctx;
use crate::flow::Step;

pub async fn run(_ctx: &mut Ctx, _env: &Env, _req: &Req) -> R<Step> {
    Ok(Step::Next)
}

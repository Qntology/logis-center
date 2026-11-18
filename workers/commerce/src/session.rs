//! (옮기는 중) 원본 341~410 (try 밖: S3 클라이언트 · request.cf · ip · geoInfo · 지역) 와 411~965 (try 안: 머리글 · 쿠키 · 질의 · 언어 ·
//! S3 세션 · 사용자 · 팀 · 잔액 · 새 세션)

use logis_worker_common::js::Env;
use logis_worker_common::web::Req;
use logis_worker_common::R;

use crate::ctx::Ctx;
use crate::flow::Step;

/// 341~410 — 여기서 던지면 원본은 fetch 가 예외로 끝난다(try 밖)
pub async fn head(_ctx: &mut Ctx, _env: &Env, _req: &Req) -> R<()> {
    Ok(())
}

/// 411~965
pub async fn run(_ctx: &mut Ctx, _env: &Env, _req: &Req) -> R<Step> {
    Ok(Step::Next)
}

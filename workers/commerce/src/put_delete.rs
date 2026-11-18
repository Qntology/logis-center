//! (옮기는 중) 원본 1594~1963 (PUT: 벡터 · 멤버 · 대화) · 1964~1997 (DELETE: 작업 · 로그아웃)

use logis_worker_common::js::Env;
use logis_worker_common::web::Req;
use logis_worker_common::R;

use crate::ctx::Ctx;
use crate::flow::Step;

pub async fn put(_ctx: &mut Ctx, _env: &Env, _req: &Req) -> R<Step> {
    Ok(Step::Next)
}

pub async fn delete(_ctx: &mut Ctx, _env: &Env, _req: &Req) -> R<Step> {
    Ok(Step::Next)
}

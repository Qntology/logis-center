//! (옮기는 중) 원본 966~1593 (CORS 머리글 · created_at · 페이지 · 아이템 · ?crons · 멤버 판별)

use logis_worker_common::js::Env;
use logis_worker_common::web::Req;
use logis_worker_common::R;

use crate::ctx::Ctx;
use crate::flow::Step;

pub async fn run(_ctx: &mut Ctx, _env: &Env, _req: &Req) -> R<Step> {
    Ok(Step::Next)
}

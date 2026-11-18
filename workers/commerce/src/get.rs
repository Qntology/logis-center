//! (옮기는 중) 원본 2455~2869 (GET: JSON 동기화 · 셸 HTML)

use logis_worker_common::js::Env;
use logis_worker_common::web::Req;
use logis_worker_common::R;

use crate::ctx::Ctx;
use crate::flow::Step;

pub async fn run(_ctx: &mut Ctx, _env: &Env, _req: &Req) -> R<Step> {
    Ok(Step::Next)
}

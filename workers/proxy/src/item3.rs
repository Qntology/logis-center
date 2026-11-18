//! (옮기는 중)

use logis_worker_common::js::Env;
use logis_worker_common::R;

use crate::ctx::Ctx;
use crate::flow::Flow;

pub async fn run(_ctx: &mut Ctx, _env: &Env) -> R<Flow> {
    Ok(Flow::Next)
}

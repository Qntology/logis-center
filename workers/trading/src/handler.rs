//! (옮기는 중) 원본 457~1079

use logis_worker_common::js::Env;
use logis_worker_common::web::{Body, Req, Resp};

pub async fn fetch(_req: Req, _env: Env) -> Resp {
    let mut r = Resp::new(418, Body::Text("I'm a teapot!".into()));
    r.set("Content-Type", "text/html; charset=utf-8");
    r
}

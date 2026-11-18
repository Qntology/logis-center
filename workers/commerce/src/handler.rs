//! `export default { async fetch(request, env, ctx) }` — 원본 341~2883 의 뼈대.
//! 341~410 은 try 밖(던지면 fetch 가 예외로 끝남), 411~2874 는 try 안(던지면 'err' 로그 뒤 418), 메서드마다 구역으로 나눈다.

use logis_worker_common::js::Env;
use logis_worker_common::web::{headers_vec, Body, Req, Resp};
use logis_worker_common::{console_log, R};

use crate::ctx::Ctx;
use crate::flow::Step;

fn teapot() -> Resp {
    let mut r = Resp::new(418, Body::Text("I'm a teapot!".into()));
    r.set("Content-Type", "text/html; charset=utf-8");
    r
}

/// try 밖에서 던진 예외 — 런타임이 내는 500 과 같게
fn uncaught(err: logis_worker_common::Throw) -> Resp {
    console_log!("uncaught", err);
    Resp::new(500, Body::Text(format!("{err}")))
}

pub async fn fetch(req: Req, env: Env) -> Resp {
    let mut ctx = Ctx::default();
    if let Err(err) = crate::session::head(&mut ctx, &env, &req).await {
        return uncaught(err);
    }
    match body(&mut ctx, &env, &req).await {
        Ok(Some(resp)) => resp,
        Ok(None) => teapot(),
        Err(err) => {
            console_log!("err", err);
            teapot()
        }
    }
}

/// 411~2874 (try 안)
async fn body(ctx: &mut Ctx, env: &Env, req: &Req) -> R<Option<Resp>> {
    if let Step::Return(r) = crate::session::run(ctx, env, req).await? {
        return Ok(Some(r));
    }
    if let Step::Return(r) = crate::resolve::run(ctx, env, req).await? {
        return Ok(Some(r));
    }
    let step = match req.method.as_str() {
        "PUT" => crate::put_delete::put(ctx, env, req).await?,
        "DELETE" => crate::put_delete::delete(ctx, env, req).await?,
        "POST" => crate::post::run(ctx, env, req).await?,
        "GET" => crate::get::run(ctx, env, req).await?,
        "OPTIONS" => {
            // 2870~2873
            let mut r = Resp::new(200, Body::Text(String::new()));
            r.headers = headers_vec(&ctx.headers);
            Step::Return(r)
        }
        _ => Step::Next,
    };
    Ok(match step {
        Step::Return(r) => Some(r),
        Step::Next => None,
    })
}

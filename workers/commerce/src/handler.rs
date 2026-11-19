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
    // 346 `new AwsClient(...)` — 키가 없으면 여기서 던진다 (아래 두 갈래보다 먼저)
    if let Err(err) = crate::session::s3_checked(&env) {
        return uncaught(err);
    }

    // 고침 (fix-preflight-session): 브라우저 사전 요청(쿠키 없음)에 세션(새 지갑 · S3 PUT)을 돌리지 않고 CORS 머리글만으로 답한다
    if req.method == "OPTIONS" {
        let headers = logis_worker_common::web::headers_new();
        if let Err(err) = crate::fixes::cors_headers(&headers) {
            return uncaught(err);
        }
        let mut r = Resp::new(200, Body::Text(String::new()));
        r.headers = headers_vec(&headers);
        return r;
    }

    // 고침 (fix-app-relay): /relay/{hash} — 세션 앞에서 (GET 이 새 지갑을 만들지 않게)
    if let Some(hash) = url::Url::parse(&req.url).ok().and_then(|u| crate::fixes::relay_hash(u.path())) {
        return match crate::fixes::relay(&req, &env, &hash).await {
            Ok(r) => r,
            Err(err) => uncaught(err),
        };
    }

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
        // 2870~2873 의 OPTIONS 는 fix-preflight-session 이 fetch 맨 앞에서 답해 여기 닿지 않는다
        _ => Step::Next,
    };
    Ok(match step {
        Step::Return(r) => Some(r),
        Step::Next => None,
    })
}

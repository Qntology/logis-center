//! JS 바인딩을 원본과 같은 호출로 부르는 층 (wasm 전용).
//!
//! - `env[name]` → [`Env::get`]: 문자열(변수 · 비밀) · 수 · 참거짓은 그 값, 바인딩 객체는 [`V::Ext`]([`Js`]), 없으면 undefined.
//! - D1: [`Env::prepare`] · [`bind`] · [`all`] · [`run`] · [`Env::batch`] 는 진짜 `D1Database.prepare(...).bind(...).all()` 을 부른다.
//!   바인딩이 없으면 원본처럼 `Cannot read properties of undefined (reading 'prepare')`, SQL 오류는 D1 의 오류 그대로.
//! - Workers AI ([`Env::ai_run`]) · Vectorize ([`Env::vectorize_upsert`] · [`Env::vectorize_query`]) · `fetch` ([`fetch`]) 도 같다.
//! - D1 은 BLOB 을 숫자 배열로 돌려주므로 [`to_v`] 도 배열로 둔다(원본의 `ungzip(row.data)` 가 그것을 받는다).

use js_sys::{Array, ArrayBuffer, Function, Object, Promise, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

use crate::jsv::{Throw, R, V};

/// V 안에 든 JS 객체
pub struct Js(pub JsValue);

/// JS 값 → V
pub fn to_v(j: &JsValue) -> V {
    if j.is_undefined() {
        return V::Undef;
    }
    if j.is_null() {
        return V::Null;
    }
    if let Some(b) = j.as_bool() {
        return V::Bool(b);
    }
    if let Some(n) = j.as_f64() {
        return V::Num(n);
    }
    if let Some(s) = j.as_string() {
        return V::Str(s);
    }
    if Array::is_array(j) {
        let a: &Array = j.unchecked_ref();
        return V::from(a.iter().map(|x| to_v(&x)).collect::<Vec<_>>());
    }
    if j.is_instance_of::<ArrayBuffer>() {
        return V::bytes(Uint8Array::new(j).to_vec());
    }
    if ArrayBuffer::is_view(j) {
        if let Some(u) = j.dyn_ref::<Uint8Array>() {
            return V::bytes(u.to_vec());
        }
        // Float32Array 등: 숫자 배열로
        let len = Reflect::get(j, &"length".into()).ok().and_then(|x| x.as_f64()).unwrap_or(0.0) as u32;
        let items: Vec<V> = (0..len).map(|i| to_v(&Reflect::get_u32(j, i).unwrap_or(JsValue::UNDEFINED))).collect();
        return V::from(items);
    }
    if j.is_object() && !j.is_function() {
        let proto = Object::get_prototype_of(j);
        let plain = proto.is_null() || proto == Object::get_prototype_of(&Object::new());
        if plain {
            let out = V::new_obj();
            let o: &Object = j.unchecked_ref();
            for k in Object::keys(o).iter() {
                if let Some(key) = k.as_string() {
                    out.put(&key, to_v(&Reflect::get(j, &k).unwrap_or(JsValue::UNDEFINED)));
                }
            }
            return out;
        }
    }
    V::ext(Js(j.clone()))
}

/// V → JS 값 (객체는 키 순서대로, 바이트는 ArrayBuffer)
pub fn to_js(v: &V) -> JsValue {
    match v {
        V::Undef => JsValue::UNDEFINED,
        V::Null => JsValue::NULL,
        V::Bool(b) => JsValue::from_bool(*b),
        V::Num(n) => JsValue::from_f64(*n),
        V::Str(s) => JsValue::from_str(s),
        V::Arr(_) => {
            let a = Array::new();
            for x in v.items() {
                a.push(&to_js(&x));
            }
            a.into()
        }
        V::Obj(_) => {
            let o = Object::new();
            for (k, x) in v.entries() {
                let _ = Reflect::set(&o, &JsValue::from_str(&k), &to_js(&x));
            }
            o.into()
        }
        V::Bytes(b) => Uint8Array::from(&b.borrow()[..]).buffer().into(),
        V::Ext(_) => v.ext_ref::<Js>().map_or(JsValue::UNDEFINED, |j| j.0.clone()),
    }
}

/// JS 예외 → Throw
pub fn to_throw(e: JsValue) -> Throw {
    let name = Reflect::get(&e, &"name".into()).ok().and_then(|x| x.as_string());
    let message = Reflect::get(&e, &"message".into()).ok().and_then(|x| x.as_string()).or_else(|| e.as_string()).unwrap_or_default();
    let name: &'static str = match name.as_deref() {
        Some("TypeError") => "TypeError",
        Some("SyntaxError") => "SyntaxError",
        Some("RangeError") => "RangeError",
        Some("ReferenceError") => "ReferenceError",
        _ => "Error",
    };
    Throw { name, message }
}

/// `obj.method(...args)` — obj 가 undefined · null 이면 원본처럼 TypeError
pub fn call(obj: &V, method: &str, args: &[JsValue]) -> R<JsValue> {
    if obj.is_nullish() {
        return Err(Throw::type_error(format!("Cannot read properties of {} (reading '{method}')", obj.to_str())));
    }
    let target = to_js(obj);
    let f = Reflect::get(&target, &JsValue::from_str(method)).map_err(to_throw)?;
    let f: Function = f.dyn_into().map_err(|_| Throw::type_error(format!("{method} is not a function")))?;
    let list = Array::new();
    for a in args {
        list.push(a);
    }
    Reflect::apply(&f, &target, &list).map_err(to_throw)
}

/// `await promise`
pub async fn wait(p: JsValue) -> R<JsValue> {
    match p.dyn_into::<Promise>() {
        Ok(promise) => JsFuture::from(promise).await.map_err(to_throw),
        Err(v) => Ok(v),
    }
}

/// `await obj.method(...args)`
pub async fn call_async(obj: &V, method: &str, args: &[JsValue]) -> R<V> {
    let p = call(obj, method, args)?;
    Ok(to_v(&wait(p).await?))
}

/// 워커의 `env`
#[derive(Clone)]
pub struct Env(pub worker::Env);

impl Env {
    /// `env[name]`
    pub fn get(&self, name: &str) -> V {
        let j = Reflect::get(&self.0, &JsValue::from_str(name)).unwrap_or(JsValue::UNDEFINED);
        if j.is_undefined() || j.is_null() || j.as_string().is_some() || j.as_f64().is_some() || j.as_bool().is_some() {
            return to_v(&j);
        }
        V::ext(Js(j))
    }

    /// `env[name]` 의 계산된 키
    pub fn getk(&self, name: &V) -> V {
        self.get(&name.to_key())
    }

    /// `env[name].prepare(sql)`
    pub fn prepare(&self, name: &str, sql: &str) -> R<V> {
        let db = self.get(name);
        Ok(V::ext(Js(call(&db, "prepare", &[JsValue::from_str(sql)])?)))
    }

    /// `env[name].batch(statements)` — statements 는 준비된 문장의 배열
    pub async fn batch(&self, name: &str, statements: &V) -> R<V> {
        let db = self.get(name);
        call_async(&db, "batch", &[to_js(statements)]).await
    }

    /// `env.AI.run(model, input)`
    pub async fn ai_run(&self, model: &str, input: &V) -> R<V> {
        let ai = self.get("AI");
        call_async(&ai, "run", &[JsValue::from_str(model), to_js(input)]).await
    }

    /// `env[name].upsert(vectors)`
    pub async fn vectorize_upsert(&self, name: &str, vectors: &V) -> R<V> {
        let index = self.get(name);
        call_async(&index, "upsert", &[to_js(vectors)]).await
    }

    /// `env[name].query(vector, options)`
    pub async fn vectorize_query(&self, name: &str, vector: &V, options: &V) -> R<V> {
        let index = self.get(name);
        call_async(&index, "query", &[to_js(vector), to_js(options)]).await
    }
}

/// `stmt.bind(...params)` — 새 준비된 문장. 고침: undefined 는 null 로 묶는다 (D1 은 undefined 를
/// `D1_TYPE_ERROR` 로 거절해, 원본은 빠진 칸 하나 때문에 그 회차의 문장을 통째로 잃었다)
pub fn bind(stmt: &V, params: &[V]) -> R<V> {
    let args: Vec<JsValue> = params.iter().map(|p| if p.is_undef() { JsValue::NULL } else { to_js(p) }).collect();
    Ok(V::ext(Js(call(stmt, "bind", &args)?)))
}

/// `await stmt.all()` → `{ results, success, meta }`
pub async fn all(stmt: &V) -> R<V> {
    call_async(stmt, "all", &[]).await
}

/// `await stmt.run()`
pub async fn run(stmt: &V) -> R<V> {
    call_async(stmt, "run", &[]).await
}

/// `(await env[db].prepare(sql).all()).results` 줄임 — 원본의 `var { results } = await …all()`
pub async fn query(env: &Env, db: &str, sql: &str) -> R<V> {
    let stmt = env.prepare(db, sql)?;
    all(&stmt).await?.g("results")
}

/// `(await env[db].prepare(sql).bind(...params).all()).results` — 값은 SQL 글에 넣지 않고 묶는다
pub async fn query_bound(env: &Env, db: &str, sql: &str, params: &[V]) -> R<V> {
    let stmt = bind(&env.prepare(db, sql)?, params)?;
    all(&stmt).await?.g("results")
}

/// `await env[db].prepare(sql).bind(...params).run()`
pub async fn run_bound(env: &Env, db: &str, sql: &str, params: &[V]) -> R<V> {
    let stmt = bind(&env.prepare(db, sql)?, params)?;
    run(&stmt).await
}

/// fetch 의 요청 본문
pub enum Body {
    None,
    Text(String),
    Bytes(Vec<u8>),
}

/// `await fetch(url, { method, headers, body })` → (status, 응답 객체)
pub async fn fetch(url: &str, method: &str, headers: &[(&str, &str)], body: Body) -> R<JsValue> {
    let init = Object::new();
    let _ = Reflect::set(&init, &"method".into(), &JsValue::from_str(method));
    let h = Object::new();
    for (k, v) in headers {
        let _ = Reflect::set(&h, &JsValue::from_str(k), &JsValue::from_str(v));
    }
    let _ = Reflect::set(&init, &"headers".into(), &h);
    match body {
        Body::None => {}
        Body::Text(t) => {
            let _ = Reflect::set(&init, &"body".into(), &JsValue::from_str(&t));
        }
        Body::Bytes(b) => {
            let _ = Reflect::set(&init, &"body".into(), &Uint8Array::from(&b[..]).buffer());
        }
    }
    let global = js_sys::global();
    let f: Function = Reflect::get(&global, &"fetch".into()).map_err(to_throw)?.unchecked_into();
    let p = f.call2(&global, &JsValue::from_str(url), &init).map_err(to_throw)?;
    wait(p).await
}

/// `await res.json()`
pub async fn response_json(res: &JsValue) -> R<V> {
    let r = V::ext(Js(res.clone()));
    call_async(&r, "json", &[]).await
}

/// worker::Request → [`crate::web::Req`] (`request.cf` 는 JS 값 그대로)
pub async fn req_from(mut req: worker::Request) -> crate::web::Req {
    let method = req.method().to_string().to_uppercase();
    let url = req.url().map(|u| u.to_string()).unwrap_or_default();
    let headers: Vec<(String, String)> = req.headers().entries().map(|(k, v)| (k.to_ascii_lowercase(), v)).collect();
    let cf = Reflect::get(req.inner(), &JsValue::from_str("cf")).map(|v| to_v(&v)).unwrap_or(V::Undef);
    let body = req.bytes().await.unwrap_or_default();
    crate::web::Req { method, url, headers, cf, body }
}

/// [`crate::web::Resp`] → worker::Response (같은 이름의 머리글은 append 로 여럿).
/// JS 의 `new Response(문자열, {headers})` 처럼 글 본문에 Content-Type 이 없으면 `text/plain;charset=UTF-8` 을 붙인다
pub fn resp_into(resp: crate::web::Resp) -> worker::Result<worker::Response> {
    let headers = worker::Headers::new();
    for (k, v) in &resp.headers {
        headers.append(k, v)?;
    }
    if matches!(resp.body, crate::web::Body::Text(_)) && !headers.has("content-type")? {
        headers.set("content-type", "text/plain;charset=UTF-8")?;
    }
    let base = match resp.body {
        crate::web::Body::Text(t) => worker::Response::ok(t)?,
        crate::web::Body::Bytes(b) => worker::Response::from_bytes(b)?,
        crate::web::Body::Empty => worker::Response::empty()?,
    };
    Ok(base.with_status(resp.status).with_headers(headers))
}

/// `response.headers.get(name)` (없으면 null)
pub fn response_header(res: &JsValue, name: &str) -> R<V> {
    let headers = Reflect::get(res, &"headers".into()).map_err(to_throw)?;
    let h = V::ext(Js(headers));
    Ok(to_v(&call(&h, "get", &[JsValue::from_str(name)])?))
}

/// `response.headers.forEach((value, key) => …)` 의 순서 (이름 소문자 · 정렬 — Fetch Headers 의 반복 순서)
pub fn response_headers(res: &JsValue) -> R<Vec<(String, String)>> {
    let headers = Reflect::get(res, &"headers".into()).map_err(to_throw)?;
    let h = V::ext(Js(headers));
    let entries = call(&h, "entries", &[])?;
    let arr = js_sys::Array::from(&entries);
    Ok(arr
        .iter()
        .filter_map(|pair| {
            let p: &Array = pair.unchecked_ref();
            Some((p.get(0).as_string()?, p.get(1).as_string()?))
        })
        .collect())
}

/// `response.status`
pub fn response_status(res: &JsValue) -> f64 {
    Reflect::get(res, &"status".into()).ok().and_then(|v| v.as_f64()).unwrap_or(0.0)
}

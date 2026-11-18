//! 요청 · 응답 (commerce · trading 워커): 원본이 `request` 에서 읽는 것과 `new Response(...)` 로 만드는 것을 플랫폼과 떼어 둔 형.
//! 진입점(wasm)이 worker::Request 를 [`Req`] 로, [`Resp`] 를 worker::Response 로 바꾼다.

use crate::jsv::V;

/// 들어온 요청
#[derive(Clone, Default)]
pub struct Req {
    /// `request.method` (대문자)
    pub method: String,
    /// `request.url`
    pub url: String,
    /// 머리글 (이름은 소문자, 들어온 순서)
    pub headers: Vec<(String, String)>,
    /// `request.cf` (없으면 undefined — 원본의 구조 분해는 그때 던진다)
    pub cf: V,
    /// 본문 바이트
    pub body: Vec<u8>,
}

impl Req {
    /// `request.headers.get(name)` — 같은 이름이 여럿이면 ", " 로 잇는다(Fetch 규약). 없으면 null
    pub fn header(&self, name: &str) -> V {
        let name = name.to_ascii_lowercase();
        let values: Vec<&str> = self.headers.iter().filter(|(k, _)| *k == name).map(|(_, v)| v.as_str()).collect();
        if values.is_empty() {
            V::Null
        } else {
            V::from(values.join(", "))
        }
    }
}

/// 응답 본문
#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    Text(String),
    Bytes(Vec<u8>),
    Empty,
}

/// 내보낼 응답 (`new Response(body, { status, headers })`)
#[derive(Clone, Debug, PartialEq)]
pub struct Resp {
    pub status: u16,
    /// 머리글 (같은 이름을 여러 번 — Set-Cookie 등 — 넣을 수 있다)
    pub headers: Vec<(String, String)>,
    pub body: Body,
}

impl Resp {
    pub fn new(status: u16, body: Body) -> Resp {
        Resp { status, headers: Vec::new(), body }
    }
    /// `headers.set(name, value)` — 같은 이름을 모두 지우고 하나
    pub fn set(&mut self, name: &str, value: impl Into<String>) {
        let lower = name.to_ascii_lowercase();
        self.headers.retain(|(k, _)| k.to_ascii_lowercase() != lower);
        self.headers.push((name.to_string(), value.into()));
    }
    /// `headers.append(name, value)`
    pub fn append(&mut self, name: &str, value: impl Into<String>) {
        self.headers.push((name.to_string(), value.into()));
    }
}

/// `new Headers()` — V 안에 두는 머리글 목록 ([`headers_new`] 로 만든다)
pub struct HeaderList(pub std::cell::RefCell<Vec<(String, String)>>);

/// `new Headers()`
pub fn headers_new() -> V {
    V::ext(HeaderList(std::cell::RefCell::new(Vec::new())))
}

fn header_list(h: &V) -> crate::jsv::R<&HeaderList> {
    h.ext_ref::<HeaderList>()
        .ok_or_else(|| crate::jsv::Throw::type_error(format!("Cannot read properties of {} (reading 'set')", h.to_str())))
}

/// `headers.set(name, value)` (이름은 소문자로 — Fetch Headers 와 같음)
pub fn headers_set(h: &V, name: &str, value: &V) -> crate::jsv::R<()> {
    let list = header_list(h)?;
    let lower = name.to_ascii_lowercase();
    let mut l = list.0.borrow_mut();
    l.retain(|(k, _)| *k != lower);
    l.push((lower, value.to_str()));
    Ok(())
}

/// `headers.append(name, value)`
pub fn headers_append(h: &V, name: &str, value: &V) -> crate::jsv::R<()> {
    header_list(h)?.0.borrow_mut().push((name.to_ascii_lowercase(), value.to_str()));
    Ok(())
}

/// 응답에 넣을 머리글 (undefined 면 빈 목록)
pub fn headers_vec(h: &V) -> Vec<(String, String)> {
    h.ext_ref::<HeaderList>().map(|l| l.0.borrow().clone()).unwrap_or_default()
}

//! aws4fetch 1.0.20 의 `AwsClient` (S3 SigV4) — 원본과 같은 서명 · 같은 머리글 · 같은 다시 시도.
//! (옮기는 중: 몸통은 공용 층 담당이 채운다. 서명 · 머리글 순서는 aws4fetch 와 바이트 단위로 같아야 한다)

use crate::jsv::{Throw, R, V};

/// `new AwsClient({ accessKeyId, secretAccessKey, service, region })` — 값은 env 에서 읽은 그대로(undefined 일 수 있음)
#[derive(Clone)]
pub struct AwsClient {
    pub access_key_id: V,
    pub secret_access_key: V,
    pub service: V,
    pub region: V,
}

/// 서명된 요청 (`await aws.sign(url, init)` 이 돌려주는 Request 와 같은 내용)
#[derive(Clone, Debug, PartialEq)]
pub struct Signed {
    pub url: String,
    pub method: String,
    /// 보낼 머리글 (Authorization · X-Amz-Date · X-Amz-Content-Sha256 포함), aws4fetch 가 Headers 에 넣는 순서
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl AwsClient {
    pub fn new(access_key_id: V, secret_access_key: V, service: V, region: V) -> AwsClient {
        AwsClient { access_key_id, secret_access_key, service, region }
    }

    /// `aws.sign(url, { method, headers, body, aws: { service, region } })` — `now_ms` 는 서명 시각(Date.now)
    pub fn sign(&self, url: &str, method: &str, headers: &[(String, String)], body: &[u8], now_ms: f64) -> R<Signed> {
        let _ = (url, method, headers, body, now_ms);
        Err(Throw::error("AwsClient::sign is not implemented yet"))
    }
}

/// `await aws.fetch(url, init)` — 서명해 보내고, 5xx · 429 면 aws4fetch 처럼 다시 시도한다 (wasm 전용)
#[cfg(target_arch = "wasm32")]
pub async fn fetch(client: &AwsClient, url: &str, method: &str, headers: &[(String, String)], body: &[u8]) -> R<wasm_bindgen::JsValue> {
    let _ = (client, url, method, headers, body);
    Err(Throw::error("aws::fetch is not implemented yet"))
}

/// `await fetch(signed)` — 서명된 요청을 그대로 보낸다 (wasm 전용)
#[cfg(target_arch = "wasm32")]
pub async fn send(signed: &Signed) -> R<wasm_bindgen::JsValue> {
    let _ = signed;
    Err(Throw::error("aws::send is not implemented yet"))
}

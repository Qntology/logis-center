//! aws4fetch 1.0.20 의 `AwsClient` (S3 SigV4) — 원본과 같은 서명 · 같은 머리글 · 같은 다시 시도.
//!
//! `sign` 이 돌려주는 [`Signed`] 는 aws4fetch 가 만든 `Request` 와 같다: 주소는 WHATWG URL 직렬화, 방법은 Request 가 고친 꼴
//! (`put` → `PUT`), 머리글은 Fetch Headers 가 반복하는 순서(소문자 이름 정렬 · 같은 이름은 ", " 로 합침 · set-cookie 는 따로).
//! 머리글 검사(ByteString · 토큰 · 값)와 오류 글은 Node(undici) 의 것을 따른다. 대조는 `tests/aws_eth.rs`.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use crate::jsv::{is_js_space, strict_eq, Throw, R, V};

/// `new AwsClient({ accessKeyId, secretAccessKey, service, region })` — 값은 env 에서 읽은 그대로(undefined 일 수 있음)
#[derive(Clone)]
pub struct AwsClient {
    pub access_key_id: V,
    pub secret_access_key: V,
    pub service: V,
    pub region: V,
    /// `retries` (기본 10)
    pub retries: f64,
    /// `initRetryMs` (기본 50)
    pub init_retry_ms: f64,
}

/// 서명된 요청 (`await aws.sign(url, init)` 이 돌려주는 Request 와 같은 내용)
#[derive(Clone, Debug, PartialEq)]
pub struct Signed {
    pub url: String,
    pub method: String,
    /// 보낼 머리글 (Authorization · X-Amz-Date · X-Amz-Content-Sha256 포함), Request 의 Headers 가 반복하는 순서
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    /// 원본 요청에 본문이 있었나 (`body: ""` 도 본문)
    pub has_body: bool,
}

/// init.body 의 JS 값
#[derive(Clone, Debug, PartialEq)]
pub enum JsBody {
    /// 없음 (undefined · null)
    Undefined,
    /// 문자열 — Request 가 Content-Type `text/plain;charset=UTF-8` 을 덧붙인다(없을 때)
    Str(String),
    /// ArrayBuffer · Uint8Array
    Bytes(Vec<u8>),
}

impl JsBody {
    /// [`AwsClient::sign`] 의 바이트 본문: 비었으면 undefined(원본 HEAD · DELETE 처럼 본문 없음), 아니면 글(UTF-8) 또는 바이트
    pub fn from_bytes(body: &[u8]) -> JsBody {
        if body.is_empty() {
            JsBody::Undefined
        } else {
            match std::str::from_utf8(body) {
                Ok(s) => JsBody::Str(s.to_string()),
                Err(_) => JsBody::Bytes(body.to_vec()),
            }
        }
    }
    fn truthy(&self) -> bool {
        match self {
            JsBody::Undefined => false,
            JsBody::Str(s) => !s.is_empty(),
            JsBody::Bytes(_) => true,
        }
    }
    fn bytes(&self) -> &[u8] {
        match self {
            JsBody::Undefined => &[],
            JsBody::Str(s) => s.as_bytes(),
            JsBody::Bytes(b) => b,
        }
    }
}

/// `aws.sign(url, init)` 의 init. `aws_*` 는 `init.aws` 의 칸(Some = 키가 있음, 값이 undefined 여도 클라이언트 값을 덮는다)
#[derive(Clone, Debug)]
pub struct SignInit<'a> {
    /// 빈 글 = undefined (`method || (body ? 'POST' : 'GET')`)
    pub method: &'a str,
    /// `headers: { … }` 의 키 순서 그대로 (같은 이름은 append 로 합친다)
    pub headers: &'a [(String, String)],
    pub body: JsBody,
    pub aws_service: Option<V>,
    pub aws_region: Option<V>,
}

const UNSIGNABLE_HEADERS: [&str; 9] =
    ["authorization", "content-type", "content-length", "user-agent", "presigned-expires", "expect", "x-amzn-trace-id", "range", "connection"];

const HOST_SERVICES: [(&str, &str); 10] = [
    ("appstream2", "appstream"),
    ("cloudhsmv2", "cloudhsm"),
    ("email", "ses"),
    ("marketplace", "aws-marketplace"),
    ("mobile", "AWSMobileHubService"),
    ("pinpoint", "mobiletargeting"),
    ("queue", "sqs"),
    ("git-codecommit", "codecommit"),
    ("mturk-requester-sandbox", "mturk-requester"),
    ("personalize-runtime", "personalize"),
];

impl AwsClient {
    pub fn new(access_key_id: V, secret_access_key: V, service: V, region: V) -> AwsClient {
        AwsClient { access_key_id, secret_access_key, service, region, retries: 10.0, init_retry_ms: 50.0 }
    }

    /// 원본 생성자처럼 키가 undefined · null 이면 던진다 (`new AwsClient` 가 처리기 맨 앞에서 던지는 자리)
    pub fn try_new(access_key_id: V, secret_access_key: V, service: V, region: V) -> R<AwsClient> {
        let c = AwsClient::new(access_key_id, secret_access_key, service, region);
        c.check()?;
        Ok(c)
    }

    fn check(&self) -> R<()> {
        if self.access_key_id.is_nullish() {
            return Err(Throw::type_error("accessKeyId is a required option"));
        }
        if self.secret_access_key.is_nullish() {
            return Err(Throw::type_error("secretAccessKey is a required option"));
        }
        Ok(())
    }

    /// `aws.sign(url, { method, headers, body, aws: { service, region } })` — `now_ms` 는 서명 시각(Date.now).
    /// 본문은 [`JsBody::from_bytes`] 로 읽는다 (빈 본문 = undefined). `aws` 칸은 클라이언트 값과 같다고 본다(부르는 곳이 모두 그렇다)
    pub fn sign(&self, url: &str, method: &str, headers: &[(String, String)], body: &[u8], now_ms: f64) -> R<Signed> {
        let init = SignInit { method, headers, body: JsBody::from_bytes(body), aws_service: None, aws_region: None };
        self.sign_init(url, &init, now_ms)
    }

    /// `aws.sign(url, init)` 전부: AwsV4Signer 의 서명 뒤 `new Request(url, signed)`
    pub fn sign_init(&self, url: &str, init: &SignInit, now_ms: f64) -> R<Signed> {
        // Object.assign({ url }, init, this, init.aws)
        let service = init.aws_service.clone().unwrap_or_else(|| self.service.clone());
        let region = init.aws_region.clone().unwrap_or_else(|| self.region.clone());
        self.check()?;
        let access = self.access_key_id.to_str();
        let secret = self.secret_access_key.to_str();

        let method = if !init.method.is_empty() {
            init.method.to_string()
        } else if init.body.truthy() {
            "POST".to_string()
        } else {
            "GET".to_string()
        };
        let parsed = url::Url::parse(url).map_err(|_| Throw::type_error("Invalid URL string."))?;
        let mut headers = Headers::from_init(init.headers)?;

        let (guessed_service, guessed_region) =
            if !service.truthy() || !region.truthy() { guess_service_region(&parsed, &headers) } else { (String::new(), String::new()) };
        let service = if service.truthy() { service } else { V::Str(guessed_service) };
        let region = if region.truthy() {
            region
        } else if !guessed_region.is_empty() {
            V::Str(guessed_region)
        } else {
            V::from("us-east-1")
        };
        let is_s3 = strict_eq(&service, &V::from("s3"));
        let service = service.to_str();
        let region = region.to_str();

        // new Date().toISOString().replace(/[:-]|\.\d{3}/g, '')
        let iso = crate::date::to_iso_string(crate::date::time_clip(now_ms))?;
        let datetime = amz_datetime(&iso);

        headers.delete("host");
        if is_s3 && !headers.has("x-amz-content-sha256") {
            headers.set("X-Amz-Content-Sha256", "UNSIGNED-PAYLOAD")?;
        }
        headers.set("X-Amz-Date", &datetime)?;

        let mut signable: Vec<String> =
            std::iter::once("host".to_string()).chain(headers.keys()).filter(|h| !UNSIGNABLE_HEADERS.contains(&h.as_str())).collect();
        signable.sort();
        let signed_headers = signable.join(";");
        let host = url_host(&parsed);
        let canonical_headers = signable
            .iter()
            .map(|h| {
                let v = if h == "host" { host.clone() } else { collapse_js_space(&headers.get(h).unwrap_or_default()) };
                format!("{h}:{v}")
            })
            .collect::<Vec<_>>()
            .join("\n");
        let date = datetime.chars().take(8).collect::<String>();
        let credential = [date.as_str(), &region, &service, "aws4_request"].join("/");

        // 경로: s3 는 한 번 풀어(+ 는 공백) 다시 싼다 — 두 번 싸지 않는다. 그 밖은 // 를 줄이고 그대로 싼다
        let pathname = parsed.path();
        let path = if is_s3 {
            decode_uri_component(&pathname.replace('+', " ")).unwrap_or_else(|| pathname.to_string())
        } else {
            collapse_slashes(pathname)
        };
        let encoded_path = encode_rfc3986(&path, true);

        let mut seen = std::collections::HashSet::new();
        let mut pairs: Vec<(String, String)> = parsed
            .query_pairs()
            .filter(|(k, _)| !k.is_empty() && (!is_s3 || seen.insert(k.to_string())))
            .map(|(k, v)| (encode_rfc3986(&k, false), encode_rfc3986(&v, false)))
            .collect();
        pairs.sort();
        let encoded_search = pairs.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&");

        let body_hash = match headers.get("x-amz-content-sha256").filter(|h| !h.is_empty()) {
            Some(h) => h,
            None => hex(&Sha256::digest(init.body.bytes())),
        };
        let canonical =
            [crate::jsv::to_upper(&method), encoded_path, encoded_search, canonical_headers + "\n", signed_headers.clone(), body_hash].join("\n");
        let string_to_sign = ["AWS4-HMAC-SHA256".to_string(), datetime.clone(), credential.clone(), hex(&Sha256::digest(canonical.as_bytes()))].join("\n");
        let k_date = hmac(format!("AWS4{secret}").as_bytes(), &date);
        let k_region = hmac(&k_date, &region);
        let k_service = hmac(&k_region, &service);
        let k_credentials = hmac(&k_service, "aws4_request");
        let signature = hex(&hmac(&k_credentials, &string_to_sign));
        headers.set(
            "Authorization",
            &format!("AWS4-HMAC-SHA256 Credential={access}/{credential}, SignedHeaders={signed_headers}, Signature={signature}"),
        )?;

        // new Request(url, signed): 사용자 정보가 든 주소 금지, 방법 검사 · 고침, GET/HEAD 본문 금지, 글 본문의 Content-Type
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(Throw::type_error(format!("Request cannot be constructed from a URL that includes credentials: {}", parsed.as_str())));
        }
        let method = request_method(&method)?;
        let has_body = init.body != JsBody::Undefined;
        if has_body && (method == "GET" || method == "HEAD") {
            return Err(Throw::type_error("Request with GET/HEAD method cannot have body."));
        }
        if matches!(init.body, JsBody::Str(_)) && !headers.has("content-type") {
            headers.append("content-type", "text/plain;charset=UTF-8")?;
        }
        Ok(Signed { url: parsed.as_str().to_string(), method, headers: headers.entries(), body: init.body.bytes().to_vec(), has_body })
    }
}

/// `2026-11-19T12:34:56.789Z` → `20261119T123456Z`
fn amz_datetime(iso: &str) -> String {
    let c: Vec<char> = iso.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < c.len() {
        if c[i] == ':' || c[i] == '-' {
            i += 1;
        } else if c[i] == '.' && i + 3 < c.len() && c[i + 1..=i + 3].iter().all(|d| d.is_ascii_digit()) {
            i += 4;
        } else {
            out.push(c[i]);
            i += 1;
        }
    }
    out
}

fn hmac(key: &[u8], text: &str) -> Vec<u8> {
    let mut m = Hmac::<Sha256>::new_from_slice(key).expect("HMAC 는 아무 길이의 키를 받는다");
    m.update(text.as_bytes());
    m.finalize().into_bytes().to_vec()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `url.host` (기본 포트는 빠진다)
fn url_host(u: &url::Url) -> String {
    let host = u.host_str().unwrap_or("");
    match u.port() {
        Some(p) => format!("{host}:{p}"),
        None => host.to_string(),
    }
}

/// `.replace(/\s+/g, ' ')`
fn collapse_js_space(s: &str) -> String {
    let mut out = String::new();
    let mut in_space = false;
    for c in s.chars() {
        if is_js_space(c) {
            if !in_space {
                out.push(' ');
            }
            in_space = true;
        } else {
            out.push(c);
            in_space = false;
        }
    }
    out
}

/// `.replace(/\/+/g, '/')`
fn collapse_slashes(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c == '/' && out.ends_with('/') {
            continue;
        }
        out.push(c);
    }
    out
}

/// `decodeURIComponent(s)` — 잘못된 % · UTF-8 이면 None (URIError)
pub fn decode_uri_component(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let h = |x: u8| (x as char).to_digit(16);
            let hi = b.get(i + 1).and_then(|&x| h(x))?;
            let lo = b.get(i + 2).and_then(|&x| h(x))?;
            out.push((hi * 16 + lo) as u8);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// `encodeRfc3986(encodeURIComponent(s))` (keep_slash 면 `.replace(/%2F/g, '/')` 까지) — 남는 것은 A-Z a-z 0-9 - _ . ~
pub fn encode_rfc3986(s: &str, keep_slash: bool) -> String {
    let mut out = String::new();
    for &b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') || (keep_slash && b == b'/') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// aws4fetch 의 guessServiceRegion
fn guess_service_region(u: &url::Url, headers: &Headers) -> (String, String) {
    let hostname = u.host_str().unwrap_or("");
    let pathname = u.path();
    if hostname.ends_with(".on.aws") {
        let re = regex::Regex::new(r"^[^.]{1,63}\.lambda-url\.([^.]{1,63})\.on\.aws$").expect("정규식");
        return match re.captures(hostname) {
            Some(m) => ("lambda".into(), m.get(1).map_or("", |x| x.as_str()).into()),
            None => (String::new(), String::new()),
        };
    }
    if hostname.ends_with(".r2.cloudflarestorage.com") {
        return ("s3".into(), "auto".into());
    }
    if hostname.ends_with(".backblazeb2.com") {
        let re = regex::Regex::new(r"^(?:[^.]{1,63}\.)?s3\.([^.]{1,63})\.backblazeb2\.com$").expect("정규식");
        return match re.captures(hostname) {
            Some(m) => ("s3".into(), m.get(1).map_or("", |x| x.as_str()).into()),
            None => (String::new(), String::new()),
        };
    }
    let re = regex::Regex::new(r"([^.]{1,63})\.(?:([^.]{0,63})\.)?amazonaws\.com(?:\.cn)?$").expect("정규식");
    let host = hostname.replacen("dualstack.", "", 1);
    let m = re.captures(&host);
    let mut service = m.as_ref().and_then(|m| m.get(1)).map_or(String::new(), |x| x.as_str().to_string());
    // None = 짝이 없음(undefined · null)
    let mut region: Option<String> = m.as_ref().and_then(|m| m.get(2)).map(|x| x.as_str().to_string());
    let ends_dash_digit = |s: &str| {
        let b = s.as_bytes();
        b.len() >= 2 && b[b.len() - 2] == b'-' && b[b.len() - 1].is_ascii_digit()
    };
    if region.as_deref() == Some("us-gov") {
        region = Some("us-gov-west-1".into());
    } else if region.as_deref() == Some("s3") || region.as_deref() == Some("s3-accelerate") {
        region = Some("us-east-1".into());
        service = "s3".into();
    } else if service == "iot" {
        service = if hostname.starts_with("iot.") {
            "execute-api".into()
        } else if hostname.starts_with("data.jobs.iot.") {
            "iot-jobs-data".into()
        } else if pathname == "/mqtt" {
            "iotdevicegateway".into()
        } else {
            "iotdata".into()
        };
    } else if service == "autoscaling" {
        let target = headers.get("x-amz-target").unwrap_or_default();
        let prefix = target.split('.').next().unwrap_or("");
        if prefix == "AnyScaleFrontendService" {
            service = "application-autoscaling".into();
        } else if prefix == "AnyScaleScalingPlannerFrontendService" {
            service = "autoscaling-plans".into();
        }
    } else if region.is_none() && service.starts_with("s3-") {
        let r = &service[3..];
        let r = r.strip_prefix("fips-").or_else(|| r.strip_prefix("external-1")).unwrap_or(r);
        region = Some(r.to_string());
        service = "s3".into();
    } else if service.ends_with("-fips") {
        service.truncate(service.len() - 5);
    } else if region.as_deref().is_some_and(|r| !r.is_empty() && !ends_dash_digit(r)) && ends_dash_digit(&service) {
        let r = region.take().unwrap_or_default();
        region = Some(std::mem::replace(&mut service, r));
    }
    let service = HOST_SERVICES.iter().find(|(k, _)| *k == service).map_or(service.clone(), |(_, v)| v.to_string());
    (service, region.unwrap_or_default())
}

/// Request 의 방법: 토큰 검사 · CONNECT/TRACE/TRACK 금지 · 여섯 방법은 대문자로
fn request_method(m: &str) -> R<String> {
    if !is_token(m) {
        return Err(Throw::type_error(format!("'{m}' is not a valid HTTP method.")));
    }
    let upper = m.to_ascii_uppercase();
    if matches!(upper.as_str(), "CONNECT" | "TRACE" | "TRACK") {
        return Err(Throw::type_error(format!("'{m}' HTTP method is unsupported.")));
    }
    Ok(if matches!(upper.as_str(), "DELETE" | "GET" | "HEAD" | "OPTIONS" | "POST" | "PUT") { upper } else { m.to_string() })
}

fn is_token(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
}

/// Fetch `Headers` (이름은 소문자로, 값은 들어온 순서로 쌓음)
#[derive(Default)]
struct Headers {
    list: Vec<(String, Vec<String>)>,
}

impl Headers {
    /// `new Headers({ … })`: 레코드의 ByteString 변환을 모두 먼저, 그다음 append
    fn from_init(init: &[(String, String)]) -> R<Headers> {
        for (k, v) in init {
            byte_string(k)?;
            byte_string(v)?;
        }
        let mut h = Headers::default();
        for (k, v) in init {
            h.append(k, v)?;
        }
        Ok(h)
    }

    fn checked(name: &str, value: &str) -> R<(String, String)> {
        let value = value.trim_matches(|c| matches!(c, '\t' | '\n' | '\r' | ' '));
        if !is_token(name) {
            return Err(Throw::type_error(format!("Headers.append: \"{name}\" is an invalid header name.")));
        }
        if value.contains(['\0', '\n', '\r']) {
            return Err(Throw::type_error(format!("Headers.append: \"{value}\" is an invalid header value.")));
        }
        Ok((name.to_ascii_lowercase(), value.to_string()))
    }

    fn append(&mut self, name: &str, value: &str) -> R<()> {
        let (name, value) = Headers::checked(name, value)?;
        match self.list.iter_mut().find(|(k, _)| *k == name) {
            Some((_, vs)) => vs.push(value),
            None => self.list.push((name, vec![value])),
        }
        Ok(())
    }

    fn set(&mut self, name: &str, value: &str) -> R<()> {
        let (name, value) = Headers::checked(name, value)?;
        match self.list.iter_mut().find(|(k, _)| *k == name) {
            Some((_, vs)) => *vs = vec![value],
            None => self.list.push((name, vec![value])),
        }
        Ok(())
    }

    fn delete(&mut self, name: &str) {
        let name = name.to_ascii_lowercase();
        self.list.retain(|(k, _)| *k != name);
    }

    fn has(&self, name: &str) -> bool {
        let name = name.to_ascii_lowercase();
        self.list.iter().any(|(k, _)| *k == name)
    }

    fn get(&self, name: &str) -> Option<String> {
        let name = name.to_ascii_lowercase();
        self.list.iter().find(|(k, _)| *k == name).map(|(_, vs)| vs.join(", "))
    }

    /// 반복 순서: 이름 정렬, set-cookie 는 값마다 따로, 그 밖은 ", " 로 합침
    fn entries(&self) -> Vec<(String, String)> {
        let mut names: Vec<&(String, Vec<String>)> = self.list.iter().collect();
        names.sort_by(|a, b| a.0.cmp(&b.0));
        let mut out = Vec::new();
        for (k, vs) in names {
            if k == "set-cookie" {
                out.extend(vs.iter().map(|v| (k.clone(), v.clone())));
            } else {
                out.push((k.clone(), vs.join(", ")));
            }
        }
        out
    }

    fn keys(&self) -> Vec<String> {
        self.entries().into_iter().map(|(k, _)| k).collect()
    }
}

/// WebIDL ByteString: 255 를 넘는 UTF-16 단위가 있으면 TypeError
fn byte_string(s: &str) -> R<()> {
    let mut i = 0usize;
    for c in s.chars() {
        let mut units = [0u16; 2];
        let u = c.encode_utf16(&mut units);
        if u[0] > 255 {
            return Err(Throw::type_error(format!(
                "Cannot convert argument to a ByteString because the character at index {i} has a value of {} which is greater than 255.",
                u[0]
            )));
        }
        i += u.len();
    }
    Ok(())
}

/// `await aws.fetch(url, init)` — 서명해 보내고, 5xx · 429 면 aws4fetch 처럼 다시 시도한다 (wasm 전용)
#[cfg(target_arch = "wasm32")]
pub async fn fetch(client: &AwsClient, url: &str, method: &str, headers: &[(String, String)], body: &[u8]) -> R<wasm_bindgen::JsValue> {
    let init = SignInit { method, headers, body: JsBody::from_bytes(body), aws_service: None, aws_region: None };
    fetch_init(client, url, &init).await
}

/// [`fetch`] 의 init 전부. 시도마다 새로 서명한다(시각은 그때의 Date.now)
#[cfg(target_arch = "wasm32")]
pub async fn fetch_init(client: &AwsClient, url: &str, init: &SignInit<'_>) -> R<wasm_bindgen::JsValue> {
    use crate::platform;
    let mut i = 0.0f64;
    while i <= client.retries {
        let signed = client.sign_init(url, init, platform::now())?;
        let res = send(&signed).await?;
        if i == client.retries {
            return Ok(res);
        }
        let status = crate::js::response_status(&res);
        if status < 500.0 && status != 429.0 {
            return Ok(res);
        }
        platform::sleep(platform::random() * client.init_retry_ms * 2f64.powf(i)).await;
        i += 1.0;
    }
    Err(Throw::error("An unknown error occurred, ensure retries is not negative"))
}

/// `await fetch(signed)` — 서명된 요청을 그대로 보낸다 (wasm 전용).
/// 본문이 비었어도 원본에 본문이 있었거나 PUT · POST 면 빈 본문을 보낸다(Fetch 는 그때 Content-Length: 0 을 붙인다)
#[cfg(target_arch = "wasm32")]
pub async fn send(signed: &Signed) -> R<wasm_bindgen::JsValue> {
    use js_sys::{Array, Function, Object, Reflect, Uint8Array};
    use wasm_bindgen::{JsCast, JsValue};
    let init = Object::new();
    let _ = Reflect::set(&init, &"method".into(), &JsValue::from_str(&signed.method));
    let pairs = Array::new();
    for (k, v) in &signed.headers {
        pairs.push(&Array::of2(&JsValue::from_str(k), &JsValue::from_str(v)));
    }
    let _ = Reflect::set(&init, &"headers".into(), &pairs);
    let with_body = !signed.body.is_empty() || signed.has_body || signed.method == "PUT" || signed.method == "POST";
    if with_body {
        let _ = Reflect::set(&init, &"body".into(), &Uint8Array::from(&signed.body[..]).buffer());
    }
    let global = js_sys::global();
    let f: Function = Reflect::get(&global, &"fetch".into()).map_err(crate::js::to_throw)?.unchecked_into();
    let p = f.call2(&global, &JsValue::from_str(&signed.url), &init).map_err(crate::js::to_throw)?;
    crate::js::wait(p).await
}

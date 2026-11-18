//! aws4fetch 1.0.20 의 sign() · ethers 6 의 isAddress · getAddress · computeAddress 와 같은지
//! (aws_eth_vectors.json: scripts/workers-diff/aws_eth_vectors.mjs 가 Node 22 · undici Request 로 만든 것)

use logis_worker_common::aws::{AwsClient, JsBody, SignInit};
use logis_worker_common::eth::{address_from_private_key, get_address, is_address, random_wallet};
use logis_worker_common::{jsv, V};
use serde_json::Value;

fn vectors() -> Value {
    serde_json::from_str(include_str!("aws_eth_vectors.json")).unwrap()
}

fn b64(s: &str) -> Vec<u8> {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let (mut acc, mut n) = (0u32, 0);
    for c in s.bytes().filter(|&c| c != b'=') {
        acc = (acc << 6) | T.iter().position(|&x| x == c).unwrap() as u32;
        n += 6;
        if n >= 8 {
            n -= 8;
            out.push((acc >> n) as u8);
        }
    }
    out
}

/// 없는 칸 = undefined
fn field(o: &Value, k: &str) -> V {
    match o.get(k) {
        None => V::Undef,
        Some(v) => jsv::from_json(v.clone()),
    }
}

fn pairs(v: &Value) -> Vec<(String, String)> {
    v.as_array()
        .map(|a| a.iter().map(|p| (p[0].as_str().unwrap().to_string(), p[1].as_str().unwrap().to_string())).collect())
        .unwrap_or_default()
}

#[test]
fn sign_matches_aws4fetch() {
    let v = vectors();
    let cases = v["aws"].as_array().unwrap();
    let (mut signed, mut thrown) = (0, 0);
    for c in cases {
        let name = c["name"].as_str().unwrap();
        let cl = &c["client"];
        let args = (field(cl, "accessKeyId"), field(cl, "secretAccessKey"), field(cl, "service"), field(cl, "region"));
        let ctor = AwsClient::try_new(args.0.clone(), args.1.clone(), args.2.clone(), args.3.clone());
        if let Some(e) = c.get("ctor_throws") {
            let err = ctor.err().unwrap_or_else(|| panic!("{name}: constructor should throw"));
            assert_eq!(err.name, e["name"].as_str().unwrap(), "{name}");
            assert_eq!(err.message, e["message"].as_str().unwrap(), "{name}");
            // 생성자를 거치지 않아도 서명이 같은 글로 던진다 (AwsV4Signer 의 검사)
            let client = AwsClient::new(args.0, args.1, args.2, args.3);
            assert_eq!(client.sign(c["url"].as_str().unwrap(), "HEAD", &[], b"", 0.0).err().map(|e| e.message), Some(err.message), "{name}");
            thrown += 1;
            continue;
        }
        let client = ctor.unwrap_or_else(|e| panic!("{name}: {e}"));
        let init = &c["init"];
        let headers = pairs(&init["headers"]);
        let body = match init.get("body") {
            None => JsBody::Undefined,
            Some(b) if b.get("str").is_some() => JsBody::Str(b["str"].as_str().unwrap().to_string()),
            Some(b) => JsBody::Bytes(b64(b["b64"].as_str().unwrap())),
        };
        let aws = init.get("aws");
        let si = SignInit {
            method: init.get("method").and_then(|m| m.as_str()).unwrap_or(""),
            headers: &headers,
            body,
            aws_service: aws.and_then(|a| a.get("service")).map(|x| jsv::from_json(x.clone())),
            aws_region: aws.and_then(|a| a.get("region")).map(|x| jsv::from_json(x.clone())),
        };
        let got = client.sign_init(c["url"].as_str().unwrap(), &si, c["now"].as_f64().unwrap());
        if let Some(e) = c.get("throws") {
            let err = got.err().unwrap_or_else(|| panic!("{name}: should throw {e}"));
            assert_eq!(err.name, e["name"].as_str().unwrap(), "{name}");
            // URL 오류 글은 workerd 의 것("Invalid URL string.")을 쓴다 — Node 는 "Invalid URL"
            if err.message != "Invalid URL string." {
                assert_eq!(err.message, e["message"].as_str().unwrap(), "{name}");
            }
            thrown += 1;
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
        let out = &c["out"];
        assert_eq!(got.url, out["url"].as_str().unwrap(), "{name}: url");
        assert_eq!(got.method, out["method"].as_str().unwrap(), "{name}: method");
        assert_eq!(got.headers, pairs(&out["headers"]), "{name}: headers");
        assert_eq!(got.body, b64(out["body"].as_str().unwrap()), "{name}: body");
        signed += 1;
    }
    assert!(signed >= 80 && thrown >= 10, "{signed} signed, {thrown} thrown");
}

#[test]
fn byte_sign_matches_the_callers() {
    // 부르는 곳의 꼴: HEAD · DELETE 는 본문 없음, PUT 은 body "" (+ Content-Type) — 바이트 본문 sign 이 같은 결과를 낸다
    let v = vectors();
    let mut n = 0;
    for c in v["aws"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        if !(name.starts_with("commerce ") || name.starts_with("trading ") || name.starts_with("undefined region")) {
            continue;
        }
        let cl = &c["client"];
        let client = AwsClient::new(field(cl, "accessKeyId"), field(cl, "secretAccessKey"), field(cl, "service"), field(cl, "region"));
        let init = &c["init"];
        let got = client
            .sign(c["url"].as_str().unwrap(), init["method"].as_str().unwrap(), &pairs(&init["headers"]), b"", c["now"].as_f64().unwrap())
            .unwrap();
        assert_eq!(got.url, c["out"]["url"].as_str().unwrap(), "{name}");
        assert_eq!(got.method, c["out"]["method"].as_str().unwrap(), "{name}");
        assert_eq!(got.headers, pairs(&c["out"]["headers"]), "{name}");
        n += 1;
    }
    assert_eq!(n, 8);
}

#[test]
fn is_address_matches_ethers() {
    let v = vectors();
    let cases = v["is_address"].as_array().unwrap();
    for c in cases {
        let x = if c.get("undef").is_some() { V::Undef } else { jsv::from_json(c["v"].clone()) };
        assert_eq!(is_address(&x), c["ok"].as_bool().unwrap(), "{:?}", c["v"]);
    }
    assert!(cases.len() >= 100);
    for c in v["checksum"].as_array().unwrap() {
        assert_eq!(get_address(c["v"].as_str().unwrap()).as_deref(), c["address"].as_str(), "{}", c["v"]);
    }
}

fn unhex(s: &str) -> Vec<u8> {
    let s = s.trim_start_matches("0x");
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

#[test]
fn compute_address_matches_ethers() {
    let v = vectors();
    for c in v["compute_address"].as_array().unwrap() {
        assert_eq!(address_from_private_key(&unhex(c["key"].as_str().unwrap())).as_deref(), c["address"].as_str(), "{}", c["key"]);
    }
    // 범위 밖 키
    assert_eq!(address_from_private_key(&[0u8; 32]), None);
    assert_eq!(address_from_private_key(&[0xffu8; 32]), None);
}

#[test]
fn random_wallet_is_consistent() {
    let mut seen = std::collections::HashSet::new();
    for _ in 0..20 {
        let (address, key) = random_wallet();
        assert!(key.len() == 66 && key.starts_with("0x") && key[2..].bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)), "{key}");
        assert_eq!(address_from_private_key(&unhex(&key)).as_deref(), Some(address.as_str()));
        // 체크섬 대소문자 그대로 (getAddress 가 같은 글을 돌려준다)
        assert_eq!(get_address(&address).as_deref(), Some(address.as_str()));
        assert_eq!(get_address(&address.to_lowercase()).as_deref(), Some(address.as_str()));
        assert!(seen.insert(key));
    }
}

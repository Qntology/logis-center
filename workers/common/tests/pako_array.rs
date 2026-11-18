//! pako.ungzip(숫자 배열) 이 던지는 자리와 Rust ungzip 이 같은지 (pako_array.json: harness 의 pako_vectors.mjs 가 pako 2.2 로 만든 것)

use logis_worker_common::gzip::{pako_array_throws, ungzip};
use logis_worker_common::V;

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

#[test]
fn array_inputs_throw_where_pako_throws() {
    let cases: serde_json::Value = serde_json::from_str(include_str!("pako_array.json")).unwrap();
    let cases = cases.as_array().unwrap();
    let mut throws = 0;
    for (i, c) in cases.iter().enumerate() {
        let gz = b64(c["gz"].as_str().unwrap());
        let expect = c["throws"].as_bool().unwrap();
        assert_eq!(pako_array_throws(&gz), expect, "case {i} ({})", c["by"]);
        let as_array = V::from(gz.iter().map(|x| V::Num(*x as f64)).collect::<Vec<_>>());
        // 고친 뒤: 숫자 배열도 언제나 풀린다 (원본은 expect 인 경우에 던졌다)
        assert!(ungzip(&as_array).is_ok(), "case {i}");
        // ArrayBuffer · Uint8Array 로는 언제나 풀린다
        assert!(ungzip(&V::bytes(gz.clone())).is_ok(), "case {i}");
        throws += expect as usize;
    }
    assert!(throws > 0 && throws < cases.len(), "the corpus covers both outcomes");
}

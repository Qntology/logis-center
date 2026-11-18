//! pako 의 `gzip` · `ungzip` 와 `TextEncoder` · `TextDecoder`.
//!
//! 압축된 바이트는 pako 와 같지 않다(zlib 이식 vs miniz_oxide). 풀면 같은 글이고, 저장 · 비교하는 쪽(D1 BLOB, isDiff)은
//! 항상 풀어서 쓰므로 동작에는 차이가 없다.

use std::io::{Read, Write};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

use crate::jsv::{parse_str, stringify_checked, Throw, R, V};

/// `gzip(bytes)`
pub fn gzip(data: &[u8]) -> Vec<u8> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::new(6));
    enc.write_all(data).expect("in-memory gzip");
    enc.finish().expect("in-memory gzip")
}

/// `ungzip(x)` — x 는 ArrayBuffer · Uint8Array · 숫자 배열(D1 BLOB). 풀 수 없으면 pako 처럼 던진다
pub fn ungzip(v: &V) -> R<Vec<u8>> {
    let bytes = v.to_bytes().ok_or_else(|| Throw::error("invalid input"))?;
    let mut out = Vec::new();
    GzDecoder::new(&bytes[..])
        .read_to_end(&mut out)
        .map_err(|e| Throw::error(format!("{e}")))?;
    Ok(out)
}

/// `new TextDecoder('utf-8').decode(bytes)` (잘못된 바이트는 U+FFFD, 앞의 BOM 은 뺀다)
pub fn decode_utf8(bytes: &[u8]) -> String {
    let b = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    String::from_utf8_lossy(b).into_owned()
}

/// `JSON.parse(new TextDecoder('utf-8').decode(ungzip(v)))` — 원본이 수십 번 쓰는 꼴
pub fn ungzip_json(v: &V) -> R<V> {
    parse_str(&decode_utf8(&ungzip(v)?))
}

/// `gzip(new TextEncoder('utf-8').encode(JSON.stringify(x)), …).buffer` — D1 에 넣는 ArrayBuffer.
/// `JSON.stringify(undefined)` 는 undefined 라 TextEncoder 가 "undefined" 를 인코딩한다(원본 그대로)
pub fn gzip_json(v: &V) -> R<V> {
    let text = stringify_checked(v)?.unwrap_or_else(|| "undefined".into());
    Ok(V::bytes(gzip(text.as_bytes())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obj;

    #[test]
    fn round_trip_through_a_d1_number_array() {
        let x = obj! { "text" => "안녕", "n" => 1.5 };
        let V::Bytes(b) = gzip_json(&x).unwrap() else { panic!() };
        // D1 은 BLOB 을 숫자 배열로 돌려준다
        let as_array = V::from(b.borrow().iter().map(|x| V::Num(*x as f64)).collect::<Vec<_>>());
        let back = ungzip_json(&as_array).unwrap();
        assert_eq!(crate::jsv::stringify(&back).unwrap(), r#"{"text":"안녕","n":1.5}"#);
        assert!(ungzip(&V::from("not gzip")).is_err());
        assert_eq!(decode_utf8(&[0xEF, 0xBB, 0xBF, b'a']), "a");
    }
}

//! 원본이 기대는 JS 내장 함수의 의미 (원본 코드가 아니라 런타임의 동작):
//! `parseInt` · `isNaN` · `Math.min/max` 의 NaN · `decodeURIComponent` · `atob`/`btoa` · `new Uint8Array(x)` ·
//! `URLSearchParams#toString` · `JSON.stringify(x, null, 2)` · pako 2.2 의 `ungzip`(형식 자동 감지 · 여러 멤버) ·
//! 평범한 객체(`{}`)의 프로토타입 속성(`obj['constructor']`, `obj['__proto__'] = …`).

use logis_worker_common::jsv::{is_js_space, num_to_string, quote, to_uint8, Throw, R, V};

// ───────────────────────── 수 ─────────────────────────

/// `isNaN(x)` (ToNumber 뒤 NaN 인가)
pub fn is_nan(v: &V) -> bool {
    v.to_num().is_nan()
}

/// `parseInt(x)` (기수 없음): ToString → 앞 공백 무시 → 부호 → `0x`/`0X` 면 16진, 아니면 10진, 숫자가 끊기는 곳까지
pub fn parse_int(v: &V) -> f64 {
    let s = v.to_str();
    let t = s.trim_start_matches(is_js_space);
    let (neg, rest) = match t.chars().next() {
        Some('-') => (true, &t[1..]),
        Some('+') => (false, &t[1..]),
        _ => (false, t),
    };
    let (radix, body) = if rest.starts_with("0x") || rest.starts_with("0X") { (16u32, &rest[2..]) } else { (10u32, rest) };
    let digits: String = body.chars().take_while(|c| c.is_digit(radix)).collect();
    if digits.is_empty() {
        return f64::NAN;
    }
    let n = if radix == 10 {
        digits.parse::<f64>().unwrap_or(f64::NAN)
    } else {
        digits.chars().fold(0.0, |acc, c| acc * 16.0 + c.to_digit(16).unwrap() as f64)
    };
    if neg {
        -n
    } else {
        n
    }
}

/// `Math.min(a, b)` (NaN 이 하나라도 있으면 NaN, -0 < +0)
pub fn math_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b {
        return if a.is_sign_negative() { a } else { b };
    }
    if a < b {
        a
    } else {
        b
    }
}

/// `Math.max(a, b)`
pub fn math_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b {
        return if a.is_sign_positive() { a } else { b };
    }
    if a > b {
        a
    } else {
        b
    }
}

// ───────────────────────── 글 ─────────────────────────

/// `s.replace('a', 'b')` — 첫 번째만
pub fn replace_first(s: &str, from: &str, to: &str) -> String {
    s.replacen(from, to, 1)
}

/// `decodeURIComponent(s)` — 잘못된 `%` 나 UTF-8 이면 `URIError: URI malformed`
pub fn decode_uri_component(s: &str) -> R<String> {
    let malformed = || Throw { name: "URIError", message: "URI malformed".into() };
    let b = s.as_bytes();
    let hex = |i: usize| -> Option<u8> {
        let h = b.get(i + 1..i + 3)?;
        if !h.iter().all(u8::is_ascii_hexdigit) {
            return None;
        }
        u8::from_str_radix(std::str::from_utf8(h).ok()?, 16).ok()
    };
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'%' {
            // 이어지는 UTF-8 글자 하나를 그대로
            let ch_len = s[i..].chars().next().map_or(1, char::len_utf8);
            out.push_str(&s[i..i + ch_len]);
            i += ch_len;
            continue;
        }
        let first = hex(i).ok_or_else(malformed)?;
        i += 3;
        if first < 0x80 {
            out.push(first as char);
            continue;
        }
        let n = if first & 0xE0 == 0xC0 {
            2
        } else if first & 0xF0 == 0xE0 {
            3
        } else if first & 0xF8 == 0xF0 {
            4
        } else {
            return Err(malformed());
        };
        let mut octets = vec![first];
        for _ in 1..n {
            if i >= b.len() || b[i] != b'%' {
                return Err(malformed());
            }
            let x = hex(i).ok_or_else(malformed)?;
            if x & 0xC0 != 0x80 {
                return Err(malformed());
            }
            octets.push(x);
            i += 3;
        }
        // 넘치는 꼴(overlong) · 서로게이트 · 0x10FFFF 넘음은 from_utf8 이 거절한다 (JS 와 같음)
        out.push_str(std::str::from_utf8(&octets).map_err(|_| malformed())?);
    }
    Ok(out)
}

/// `URLSearchParams#toString` 의 한 칸 (application/x-www-form-urlencoded 직렬화)
pub fn form_urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b' ' => out.push('+'),
            b'*' | b'-' | b'.' | b'_' | b'0'..=b'9' | b'A'..=b'Z' | b'a'..=b'z' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// `new URLSearchParams()` 에 `append(k, String(v))` 한 것의 `toString()`
pub fn search_params(pairs: &[(&str, String)]) -> String {
    pairs.iter().map(|(k, v)| format!("{}={}", form_urlencode(k), form_urlencode(v))).collect::<Vec<_>>().join("&")
}

// ───────────────────────── base64 ─────────────────────────

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// `btoa(String.fromCharCode(...bytes))`
pub fn btoa_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { B64[n as usize & 63] as char } else { '=' });
    }
    out
}

/// `atob(s)` 의 글자 코드들 (forgiving-base64 decode: ASCII 공백을 빼고, 길이가 4의 배수면 끝의 `=` 한두 개를 빼고,
/// 나머지가 4로 나눠 1 이 남거나 알파벳 밖 글자가 있으면 `InvalidCharacterError`)
pub fn atob(s: &str) -> R<Vec<u8>> {
    let invalid = || Throw { name: "InvalidCharacterError", message: "atob() called with invalid base64-encoded data. (Only whitespace, '+', '/', alphanumeric ASCII, and up to two terminal '=' signs when the input data length is divisible by 4 are allowed.)".into() };
    let mut data: Vec<char> = s.chars().filter(|c| !matches!(c, '\t' | '\n' | '\u{0C}' | '\r' | ' ')).collect();
    if data.len() % 4 == 0 {
        if data.last() == Some(&'=') {
            data.pop();
            if data.last() == Some(&'=') {
                data.pop();
            }
        }
    }
    if data.len() % 4 == 1 {
        return Err(invalid());
    }
    let mut out = Vec::with_capacity(data.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits = 0;
    for c in data {
        let v = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '+' => 62,
            '/' => 63,
            _ => return Err(invalid()),
        };
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

// ───────────────────────── Uint8Array ─────────────────────────

/// 2^53 - 1
const MAX_SAFE: f64 = 9007199254740991.0;

fn alloc_zeros(len: f64) -> R<Vec<u8>> {
    // V8 의 형식 배열 최대 길이(2^32) 를 넘으면 RangeError
    if len > 4294967296.0 {
        return Err(Throw::range_error(format!("Invalid typed array length: {}", num_to_string(len))));
    }
    Ok(vec![0u8; len as usize])
}

/// `new Uint8Array(x)`: ArrayBuffer · 형식 배열은 그 바이트, 배열(반복 가능)은 원소의 ToUint8, 그 밖의 객체는 유사 배열
/// (`length` 와 첨자), 원시값은 ToIndex 길이만큼의 0 (음수 · 2^53 넘음은 RangeError)
pub fn uint8array_new(v: &V) -> R<Vec<u8>> {
    match v {
        V::Bytes(b) => Ok(b.borrow().clone()),
        V::Arr(a) => Ok(a.borrow().iter().map(|x| to_uint8(x.to_num())).collect()),
        V::Obj(_) | V::Ext(_) => {
            let n = v.get("length").to_num();
            let len = if n.is_nan() || n <= 0.0 { 0.0 } else { n.trunc().min(MAX_SAFE) };
            let mut out = alloc_zeros(len)?;
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = to_uint8(v.get(&i.to_string()).to_num());
            }
            Ok(out)
        }
        other => {
            // ToIndex: undefined → 0, 그 밖은 ToIntegerOrInfinity
            let n = if other.is_undef() { 0.0 } else { other.to_num() };
            let n = if n.is_nan() { 0.0 } else { n.trunc() };
            if !(0.0..=MAX_SAFE).contains(&n) {
                return Err(Throw::range_error(format!("Invalid typed array length: {}", num_to_string(n))));
            }
            alloc_zeros(n)
        }
    }
}

// ───────────────────────── JSON.stringify(x, null, 2) ─────────────────────────

fn pretty_into(v: &V, out: &mut String, indent: &str, stack: &mut Vec<*const ()>) -> R<bool> {
    match v {
        V::Arr(a) => {
            let p = std::rc::Rc::as_ptr(a) as *const ();
            if stack.contains(&p) {
                return Err(Throw::type_error("Converting circular structure to JSON"));
            }
            let items = a.borrow().clone();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(true);
            }
            stack.push(p);
            let inner = format!("{indent}  ");
            out.push('[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                out.push_str(&inner);
                if !pretty_into(x, out, &inner, stack)? {
                    out.push_str("null");
                }
            }
            out.push('\n');
            out.push_str(indent);
            out.push(']');
            stack.pop();
            Ok(true)
        }
        V::Obj(o) => {
            let p = std::rc::Rc::as_ptr(o) as *const ();
            if stack.contains(&p) {
                return Err(Throw::type_error("Converting circular structure to JSON"));
            }
            stack.push(p);
            let inner = format!("{indent}  ");
            let mut parts: Vec<String> = Vec::new();
            for k in v.keys() {
                let x = v.get(&k);
                let mut s = String::new();
                if pretty_into(&x, &mut s, &inner, stack)? {
                    parts.push(format!("{}: {}", quote(&k), s));
                }
            }
            stack.pop();
            if parts.is_empty() {
                out.push_str("{}");
            } else {
                out.push_str("{\n");
                out.push_str(&inner);
                out.push_str(&parts.join(&format!(",\n{inner}")));
                out.push('\n');
                out.push_str(indent);
                out.push('}');
            }
            Ok(true)
        }
        other => match logis_worker_common::jsv::stringify_checked(other)? {
            Some(s) => {
                out.push_str(&s);
                Ok(true)
            }
            None => Ok(false),
        },
    }
}

/// `JSON.stringify(v, null, 2)` (undefined 면 None)
pub fn stringify_pretty(v: &V) -> R<Option<String>> {
    let mut out = String::new();
    Ok(pretty_into(v, &mut out, "", &mut Vec::new())?.then_some(out))
}

// ───────────────────────── 평범한 객체의 프로토타입 ─────────────────────────

/// `Object.prototype` 의 속성 이름과 `String(그 값)` (함수는 `function x() { [native code] }`)
fn proto_value(key: &str) -> Option<V> {
    const FNS: &[&str] = &[
        "constructor", "toString", "toLocaleString", "valueOf", "hasOwnProperty", "isPrototypeOf", "propertyIsEnumerable",
        "__defineGetter__", "__defineSetter__", "__lookupGetter__", "__lookupSetter__",
    ];
    if key == "__proto__" {
        // Object.prototype 자체: 참인 객체, 그 속성(digest 등)은 undefined
        return Some(V::new_obj());
    }
    FNS.contains(&key).then(|| {
        let name = if key == "constructor" { "Object" } else { key };
        V::from(format!("function {name}() {{ [native code] }}"))
    })
}

/// `obj[key]` — `{}` 로 만든 객체: 자기 속성이 없으면 Object.prototype 의 것
pub fn oget(obj: &V, key: &str) -> V {
    if obj.has(key) {
        return obj.get(key);
    }
    proto_value(key).unwrap_or(V::Undef)
}

/// `obj[key] = val` — `__proto__` 에 원시값을 쓰면 아무 일도 없다 (객체를 쓰면 프로토타입이 바뀌지만 원본에서 그런 값은 오지 않는다)
pub fn oset(obj: &V, key: &str, val: V) -> R<()> {
    if key == "__proto__" && !obj.has(key) {
        return Ok(());
    }
    obj.s_(key, val)
}

// ───────────────────────── pako 2.2 ungzip ─────────────────────────

fn throw_str(msg: &str) -> Throw {
    // pako 는 Error 가 아니라 글을 던진다
    Throw::value(msg)
}

/// 날 deflate 를 풀고 (출력, 쓴 입력 바이트 수)
fn inflate_raw(input: &[u8]) -> R<(Vec<u8>, usize)> {
    use flate2::{Decompress, FlushDecompress, Status};
    let mut d = Decompress::new(false);
    let mut out: Vec<u8> = Vec::with_capacity(input.len() * 4 + 1024);
    loop {
        if out.capacity() - out.len() < 4096 {
            out.reserve(out.capacity().max(4096));
        }
        let in_pos = d.total_in() as usize;
        let out_before = d.total_out();
        // 고침 (fix-trading-inflate-grow, Rust 판만): Finish 로 부르면 처음 잡은 출력 칸(입력 × 4 + 1 KiB)이 차는 순간
        // 진전 없이 끝나 'buffer error' 가 되었다 — 4배 넘게 줄어든 gzip 본문(앱이 올리는 200건 JSON 이 그렇다)을 못 풀어
        // POST 가 빈 목록으로 끝났다. None 으로 부르고 칸을 늘려 가며 끝까지 푼다
        let status = d
            .decompress_vec(&input[in_pos.min(input.len())..], &mut out, FlushDecompress::None)
            .map_err(|_| throw_str("invalid stored block lengths"))?;
        if let Status::StreamEnd = status {
            return Ok((out, d.total_in() as usize));
        }
        if d.total_in() as usize == in_pos && d.total_out() == out_before {
            // 더 줄 입력이 없다: 잘린 스트림
            return Err(throw_str("buffer error"));
        }
    }
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = flate2::Crc::new();
    c.update(data);
    c.sum()
}

/// 멤버 하나 (HEAD 부터 DONE 까지). (출력, 쓴 바이트 수, gzip 멤버인가)
fn inflate_member(input: &[u8]) -> R<(Vec<u8>, usize, bool)> {
    let need = |n: usize| if input.len() < n { Err(throw_str("buffer error")) } else { Ok(()) };
    need(2)?;
    if input[0] == 0x1f && input[1] == 0x8b {
        // gzip (RFC 1952) — zlib inflate.c 의 FLAGS · TIME · OS · EXLEN · EXTRA · NAME · COMMENT · HCRC
        need(4)?;
        if input[2] != 8 {
            return Err(throw_str("unknown compression method"));
        }
        let flg = input[3];
        if flg & 0xe0 != 0 {
            return Err(throw_str("unknown header flags set"));
        }
        let mut p = 10;
        need(p)?;
        if flg & 0x04 != 0 {
            need(p + 2)?;
            let xlen = input[p] as usize | (input[p + 1] as usize) << 8;
            p += 2 + xlen;
            need(p)?;
        }
        for bit in [0x08u8, 0x10] {
            if flg & bit != 0 {
                loop {
                    need(p + 1)?;
                    p += 1;
                    if input[p - 1] == 0 {
                        break;
                    }
                }
            }
        }
        if flg & 0x02 != 0 {
            need(p + 2)?;
            let hcrc = input[p] as u32 | (input[p + 1] as u32) << 8;
            if hcrc != crc32(&input[..p]) & 0xffff {
                return Err(throw_str("header crc mismatch"));
            }
            p += 2;
        }
        let (out, used) = inflate_raw(&input[p..])?;
        p += used;
        need(p + 8)?;
        let crc = u32::from_le_bytes([input[p], input[p + 1], input[p + 2], input[p + 3]]);
        if crc != crc32(&out) {
            return Err(throw_str("incorrect data check"));
        }
        let isize = u32::from_le_bytes([input[p + 4], input[p + 5], input[p + 6], input[p + 7]]);
        if isize != out.len() as u32 {
            return Err(throw_str("incorrect length check"));
        }
        Ok((out, p + 8, true))
    } else {
        // zlib (RFC 1950)
        let (cmf, flg) = (input[0], input[1]);
        if ((cmf as u32) << 8 | flg as u32) % 31 != 0 {
            return Err(throw_str("incorrect header check"));
        }
        if cmf & 0x0f != 8 {
            return Err(throw_str("unknown compression method"));
        }
        if (cmf >> 4) + 8 > 15 {
            return Err(throw_str("invalid window size"));
        }
        if flg & 0x20 != 0 {
            // DICTID → Z_NEED_DICT (사전 없음)
            need(6)?;
            return Err(throw_str("need dictionary"));
        }
        let (out, used) = inflate_raw(&input[2..])?;
        let p = 2 + used;
        need(p + 4)?;
        let check = u32::from_be_bytes([input[p], input[p + 1], input[p + 2], input[p + 3]]);
        if check != adler32(&out) {
            return Err(throw_str("incorrect data check"));
        }
        Ok((out, p + 4, false))
    }
}

/// pako 2.2.0 `ungzip(bytes)` (= `inflate`, windowBits 15+32: gzip · zlib 자동 감지). gzip 멤버 뒤에 0 이 아닌 바이트가
/// 이어지면 다음 멤버로 풀고(실패하면 던짐), 0 이면 거기서 끝. zlib 뒤의 바이트는 무시한다. 잘리면 `buffer error`
pub fn pako_ungzip(input: &[u8]) -> R<Vec<u8>> {
    let mut out = Vec::new();
    let mut pos = 0;
    loop {
        let (chunk, used, gzip) = inflate_member(&input[pos..])?;
        out.extend_from_slice(&chunk);
        pos += used;
        if !(pos < input.len() && gzip && input[pos] != 0) {
            return Ok(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins() {
        assert_eq!(parse_int(&V::from(" 0x1A")), 26.0);
        assert_eq!(parse_int(&V::from("12.9e3")), 12.0);
        assert!(parse_int(&V::from("")).is_nan());
        assert!(math_max(1.0, math_min(1000.0, f64::NAN)).is_nan());
        assert_eq!(decode_uri_component("a%20%EA%B0%80+").unwrap(), "a 가+");
        assert!(decode_uri_component("%").is_err());
        assert!(decode_uri_component("%C0%80").is_err());
        assert!(decode_uri_component("%ED%A0%80").is_err());
        assert_eq!(btoa_bytes(&[0x1f, 0x8b, 8]), "H4sI");
        assert_eq!(btoa_bytes(&[1]), "AQ==");
        assert_eq!(atob(" H4s I\n").unwrap(), vec![0x1f, 0x8b, 8]);
        assert_eq!(atob("AQ").unwrap(), vec![1]);
        assert!(atob("A").is_err());
        assert!(atob("AQ=").is_err());
        assert_eq!(search_params(&[("o", "https://x.y/a b*~".into())]), "o=https%3A%2F%2Fx.y%2Fa+b*%7E");
        assert_eq!(uint8array_new(&V::from(3.7)).unwrap(), vec![0, 0, 0]);
        assert!(uint8array_new(&V::from(-1.0)).is_err());
        assert_eq!(uint8array_new(&V::from("abc")).unwrap(), Vec::<u8>::new());
        let o = logis_worker_common::obj! { "a" => "x", "b" => V::Null, "c" => logis_worker_common::obj!{}, "d" => V::Undef };
        assert_eq!(stringify_pretty(&o).unwrap().unwrap(), "{\n  \"a\": \"x\",\n  \"b\": null,\n  \"c\": {}\n}");
    }

    #[test]
    fn pako_formats() {
        use std::io::Write;
        let gz = |s: &[u8]| {
            let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            e.write_all(s).unwrap();
            e.finish().unwrap()
        };
        let zl = |s: &[u8]| {
            let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            e.write_all(s).unwrap();
            e.finish().unwrap()
        };
        assert_eq!(pako_ungzip(&gz(b"hello")).unwrap(), b"hello");
        assert_eq!(pako_ungzip(&zl(b"hello")).unwrap(), b"hello");
        let mut two = gz(b"ab");
        two.extend(gz(b"cd"));
        assert_eq!(pako_ungzip(&two).unwrap(), b"abcd");
        let mut pad = gz(b"ab");
        pad.extend([0, 1, 2]);
        assert_eq!(pako_ungzip(&pad).unwrap(), b"ab");
        let mut junk = gz(b"ab");
        junk.extend(b"xyz");
        assert!(pako_ungzip(&junk).is_err());
        let full = gz(b"hello world hello world");
        assert!(pako_ungzip(&full[..full.len() - 3]).is_err());
        assert!(pako_ungzip(b"[1,2]").is_err());
        assert!(pako_ungzip(b"").is_err());
        // fix-trading-inflate-grow: 4배 넘게 줄어든 본문 (처음 잡은 출력 칸을 넘는다)
        let big: Vec<u8> = (0..200).flat_map(|i| format!("{{\"id\":\"k-{i}\",\"type\":\"BL\",\"data\":{{\"doc_number\":\"K{i}\"}}}},").into_bytes()).collect();
        let packed = gz(&big);
        assert!(packed.len() * 4 + 1024 < big.len());
        assert_eq!(pako_ungzip(&packed).unwrap(), big);
        assert_eq!(pako_ungzip(&zl(&big)).unwrap(), big);
        let zeros = vec![0u8; 1 << 20];
        assert_eq!(pako_ungzip(&gz(&zeros)).unwrap(), zeros);
    }
}

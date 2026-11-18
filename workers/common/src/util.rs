//! 두 워커가 같이 쓰던 작은 함수들 — 원본 그대로(원본의 버릇까지). 버릇은 `원본 그대로:` 주석으로 남긴다.

use std::sync::OnceLock;

use regex::Regex;

use crate::jsv::{index_of, loose_eq, strict_eq, Throw, R, V};

impl Throw {
    pub fn reference_error(message: impl Into<String>) -> Throw {
        Throw { name: "ReferenceError", message: message.into() }
    }
}

/// `crc32(s)` — UTF-16 단위마다, 255 를 넘는 글자면 RangeError (원본 그대로)
pub fn crc32(s: &str) -> R<f64> {
    let mut crc: u32 = 0xFFFF_FFFF;
    for u in s.encode_utf16() {
        if u > 255 {
            return Err(Throw::range_error("Invalid code point"));
        }
        crc ^= u as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    Ok((!crc) as f64)
}

/// `hashId(text)` — `ethers.computeAddress(ethers.hashMessage(text)).toLowerCase()` (앱과 같은 logis-schema 구현).
/// undefined 면 무작위 개인 키 문자열로(원본: `ethers.Wallet.createRandom().privateKey`). 문자열이 아니면 ethers 처럼 던진다
pub fn hash_id(text: &V) -> R<String> {
    match text {
        V::Str(s) => Ok(logis_schema::hash::hash_id(s)),
        V::Undef => {
            let key: String = crate::platform::random_bytes(32).iter().map(|b| format!("{b:02x}")).collect();
            Ok(logis_schema::hash::hash_id(&format!("0x{key}")))
        }
        other => Err(Throw::type_error(format!("invalid BytesLike value (argument=\"value\", value={}, code=INVALID_ARGUMENT)", other.to_str()))),
    }
}

/// `hashId(String)` 줄임
pub fn hash_str(text: &str) -> String {
    logis_schema::hash::hash_id(text)
}

/// `Digest(text)`: 문장 부호 · 기호 · 구분자(\p{P}\p{S}\p{Z})를 지우고 소문자로 → hashId
pub fn digest(text: &V) -> R<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"[\p{P}\p{S}\p{Z}]").unwrap());
    let s = match text {
        V::Str(s) => s,
        V::Undef | V::Null => return Err(Throw::type_error(format!("Cannot read properties of {} (reading 'replace')", text.to_str()))),
        _ => return Err(Throw::type_error("text.replace is not a function")),
    };
    Ok(hash_str(&re.replace_all(s, "").to_lowercase()))
}

/// `normalizeNumericHomoglyphs(str)` — 문자열이 아니면 그대로
pub fn normalize_numeric_homoglyphs(v: &V) -> V {
    let V::Str(s) = v else { return v.clone() };
    V::Str(s.chars().map(homoglyph_digit).collect())
}

fn homoglyph_digit(c: char) -> char {
    match c {
        'O' | 'o' | 'Ο' | 'ο' | '○' | '〇' | '０' | 'Ｏ' | 'ｏ' | 'D' => '0',
        'I' | 'l' | '１' | 'Ｉ' | 'ｌ' | 'Ι' | 'ι' | '|' | 'ᛁ' | '!' => '1',
        'Z' | 'z' | '２' | 'Ƨ' | 'ᒿ' | 'Ƶ' | 'ᘔ' => '2',
        'Ɛ' | 'ɜ' | 'З' | 'з' | '３' | 'Ʒ' | 'Ƹ' => '3',
        'Ꮞ' | '４' | 'ɥ' => '4',
        'S' | 's' | '５' | 'ƽ' | '$' | '§' => '5',
        'b' | 'G' | 'Ꮾ' | '６' | 'б' => '6',
        'T' | '７' | 'т' | 'τ' => '7',
        'B' | 'Β' | 'β' | 'ß' | '８' | 'θ' | '∞' | '&' => '8',
        'g' | 'q' | '９' | 'ǵ' | 'ɡ' | 'P' => '9',
        other => other,
    }
}

/// `cleanNumber(str)` — 원본 그대로: `.` 이 있으면 `/./gi` 로 줄바꿈 아닌 모든 글자를 지운다
pub fn clean_number(v: &V) -> R<V> {
    let V::Str(s) = v else {
        return Err(Throw::type_error(format!("str.indexOf is not a function ({})", v.type_of())));
    };
    let mut s = s.clone();
    if index_of(&s, "-") > -1 {
        s = s.replace('-', "");
    }
    if index_of(&s, "_") > -1 {
        s = s.replace('_', "");
    }
    if index_of(&s, ".") > -1 {
        s = s.chars().filter(|c| matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}')).collect();
    }
    if index_of(&s, ",") > -1 {
        s = s.replace(',', "");
    }
    Ok(V::Str(s))
}

/// `{ ...obj }` 의 원본 키 (문자열은 글자 첨자, 원시값 · null · undefined 는 없음)
fn spread_keys(v: &V) -> Vec<String> {
    match v {
        V::Obj(_) | V::Arr(_) | V::Str(_) => v.keys(),
        _ => Vec::new(),
    }
}

/// `mergeNode(obj1, obj2)`: obj1 을 얕게 복사하고, obj2 의 값이 비어 있지 않으면(null · undefined · '' · 0 이 아니면) 덮어쓴다
pub fn merge_node(obj1: &V, obj2: &V) -> V {
    let merged = V::new_obj();
    for k in spread_keys(obj1) {
        merged.put(&k, obj1.get(&k));
    }
    if matches!(obj2, V::Obj(_) | V::Arr(_) | V::Str(_)) {
        for k in obj2.keys() {
            let v2 = obj2.get(&k);
            let empty = v2.is_nullish() || strict_eq(&v2, &V::from("")) || strict_eq(&v2, &V::Num(0.0));
            if !empty {
                merged.put(&k, v2);
            }
        }
    }
    merged
}

/// `isDiff(obj1, obj2)`. 원본은 키가 `data` 이면 `Buffer.isBuffer(...)` 를 부르는데, `nodejs_compat` 없는 Workers 에는
/// `Buffer` 가 없어 ReferenceError 가 난다 — 그대로 던진다
pub fn is_diff(a: &V, b: &V) -> R<bool> {
    if !a.truthy() && !b.truthy() {
        return Ok(false);
    }
    if !a.truthy() || !b.truthy() {
        return Ok(true);
    }
    let k1 = a.keys();
    let k2 = b.keys();
    if k1.len() != k2.len() {
        return Ok(true);
    }
    for key in k1 {
        if key == "data" {
            return Err(Throw::reference_error("Buffer is not defined"));
        }
        let (x, y) = (a.get(&key), b.get(&key));
        if x.type_of() == "object" && y.type_of() == "object" {
            if is_diff(&x, &y)? {
                return Ok(true);
            }
        } else if !strict_eq(&x, &y) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `rowsTrim(rows, key, value)`: rows[r][key] 가 있고 == value 인 행을 빼고 돌려준다
pub fn rows_trim(rows: &V, key: &V, value: &V) -> R<V> {
    if !key.is_undef() && !value.is_undef() {
        let k = key.to_key();
        for (r, row) in rows.items().iter().enumerate() {
            if row.truthy() {
                let x = row.g(&k)?;
                if x.truthy() && loose_eq(&x, value) {
                    rows.put(&r.to_string(), V::Undef);
                }
            }
        }
    }
    Ok(V::from(rows.items().into_iter().filter(|r| !r.is_undef()).collect::<Vec<_>>()))
}

/// `getZeroUTC(date, day)`: day 일 전의 UTC 자정 (Workers 의 지역 시각 = UTC)
pub fn get_zero_utc(t: f64, day: f64) -> f64 {
    let shifted = t - day.trunc() * 86_400_000.0;
    (shifted / 86_400_000.0).floor() * 86_400_000.0
}

/// `containsUnsupportedChars(str)`
pub fn contains_unsupported_chars(s: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r#"[^a-zA-Z0-9\s!@#$%^&*()_+\-=\[\]{}|;:'",.<>/?~`]"#).unwrap());
    // JS \s 와 regex 의 \s 는 거의 같다 (JS 는 \u{FEFF} 도 공백)
    re.is_match(&s.replace('\u{FEFF}', " "))
}

/// `Math.floor(Math.random() * n)`
pub fn random_below(n: u32) -> u32 {
    (crate::platform::random() * n as f64).floor() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obj;

    #[test]
    fn clean_number_keeps_the_original_quirks() {
        assert_eq!(clean_number(&V::from("12-34_5,6")).unwrap(), V::from("123456"));
        assert_eq!(clean_number(&V::from("1.234")).unwrap(), V::from(""), "/./gi removes everything");
        assert!(clean_number(&V::Undef).is_err());
    }

    #[test]
    fn merge_node_skips_empty_values() {
        let a = obj! { "x" => 1, "y" => "keep", "z" => 5 };
        let b = obj! { "y" => "", "z" => 0, "w" => V::Null, "q" => 7 };
        let m = merge_node(&a, &b);
        assert_eq!(crate::jsv::stringify(&m).unwrap(), r#"{"x":1,"y":"keep","z":5,"q":7}"#);
        assert_eq!(crate::jsv::stringify(&merge_node(&V::Undef, &a)).unwrap(), r#"{"x":1,"y":"keep","z":5}"#);
    }

    #[test]
    fn is_diff_matches_the_original() {
        let a = obj! { "base" => obj! { "n" => 1 } };
        let b = obj! { "base" => obj! { "n" => 1 } };
        assert_eq!(is_diff(&a, &b), Ok(false));
        b.get("base").put("n", V::from(2));
        assert_eq!(is_diff(&a, &b), Ok(true));
        assert_eq!(is_diff(&V::Undef, &V::Null), Ok(false));
        assert_eq!(is_diff(&obj! { "data" => 1 }, &obj! { "data" => 1 }).unwrap_err().name, "ReferenceError");
    }

    #[test]
    fn crc32_and_homoglyphs() {
        assert_eq!(crc32("hello").unwrap(), 907060870.0);
        assert!(crc32("가").is_err());
        assert_eq!(normalize_numeric_homoglyphs(&V::from("SO-l2B")), V::from("50-128"));
        assert_eq!(get_zero_utc(1_758_824_626_364.0, 1.0), 1_758_672_000_000.0);
    }
}

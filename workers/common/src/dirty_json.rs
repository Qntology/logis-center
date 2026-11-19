//! `normalizeToJsonString` · `parseDirtyJson` — LLM 이 낸 망가진 JSON 고치기 (원본 정규식 그대로, 4번째는 미리보기가 있어 fancy-regex)

use std::sync::OnceLock;

use fancy_regex::Regex as FancyRegex;
use regex::{Captures, Regex};

use crate::jsv::{js_trim, parse_str, V};

fn re(cell: &'static OnceLock<Regex>, pat: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pat).unwrap())
}

/// `normalizeToJsonString(input)`
pub fn normalize_to_json_string(input: &str) -> String {
    static SPACES: OnceLock<Regex> = OnceLock::new();
    static BACKTICK: OnceLock<Regex> = OnceLock::new();
    static KEYS: OnceLock<Regex> = OnceLock::new();
    static SINGLE: OnceLock<Regex> = OnceLock::new();
    static INNER: OnceLock<FancyRegex> = OnceLock::new();
    static ATTR: OnceLock<Regex> = OnceLock::new();
    static TRAILING: OnceLock<Regex> = OnceLock::new();

    let s = re(&SPACES, "[\u{00A0}\u{200B}\u{202F}\u{FEFF}]").replace_all(input, " ");
    let mut s = js_trim(&s).to_string();

    // 1. `...` → "..." (안의 " 는 \")
    s = re(&BACKTICK, r"`([\s\S]*?)`")
        .replace_all(&s, |c: &Captures| format!("\"{}\"", c[1].replace('"', "\\\"")))
        .into_owned();
    // 2. key: → "key":
    s = re(&KEYS, r"([{,])\s*([a-zA-Z0-9_]+)\s*:").replace_all(&s, "${1}\"${2}\":").into_owned();
    // 3. : 'value' → : "value"
    s = re(&SINGLE, r":\s*'([^']*)'")
        .replace_all(&s, |c: &Captures| format!(": \"{}\"", c[1].replace('"', "\\\"")))
        .into_owned();
    // 4. "node": "form[name="frm"]" 안쪽 따옴표 (원본 정규식 끝의 공백까지 그대로)
    let inner = INNER.get_or_init(|| FancyRegex::new(r#"(":\s*")([\s\S]*?)("(?=\s*[,}\]])) "#).unwrap());
    let attr = re(&ATTR, r#"="([^"]*)""#);
    let mut out = String::with_capacity(s.len());
    let mut last = 0;
    for m in inner.captures_iter(&s).flatten() {
        let whole = m.get(0).unwrap();
        out.push_str(&s[last..whole.start()]);
        let fixed = attr.replace_all(&m[2], "=\\\"${1}\\\"");
        out.push_str(&m[1]);
        out.push_str(&fixed);
        out.push_str(&m[3]);
        last = whole.end();
    }
    out.push_str(&s[last..]);
    s = out;
    // 5. 후행 쉼표
    s = re(&TRAILING, r",\s*([\]}])").replace_all(&s, "${1}").into_owned();
    // 6. 닫히지 않은 { 를 닫는다
    let open = s.matches('{').count();
    let close = s.matches('}').count();
    if open > close {
        s.push_str(&"}".repeat(open - close));
    }
    s
}

/// `parseDirtyJson(input)` — 그대로, 고쳐서, 안 되면 null
pub fn parse_dirty_json(input: &V) -> V {
    if !input.truthy() {
        return V::Null;
    }
    let text = input.to_str();
    if let Ok(v) = parse_str(&text) {
        return v;
    }
    match parse_str(&normalize_to_json_string(&text)) {
        Ok(v) => v,
        Err(e) => {
            crate::platform::log(&format!("🔧 수리 실패: {}", e.message));
            V::Null
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repairs_llm_json() {
        let v = parse_dirty_json(&V::from("{type: 'goods', items: [1,2,],}"));
        assert_eq!(crate::jsv::stringify(&v).unwrap(), r#"{"type":"goods","items":[1,2]}"#);
        let v = parse_dirty_json(&V::from("{a: `say \"hi\"`"));
        assert_eq!(crate::jsv::stringify(&v).unwrap(), r#"{"a":"say \"hi\""}"#);
        assert_eq!(parse_dirty_json(&V::from("")), V::Null);
        assert_eq!(parse_dirty_json(&V::from("nope")), V::Null);
    }
}

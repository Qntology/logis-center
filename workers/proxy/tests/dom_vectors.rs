//! linkedom 0.18.11 과 같은 트리인지: 하네스 `dom_vectors.mjs` 가 원본 `convertHtmlToCleanPug` · `querySelectorAll` 로 만든
//! 기댓값(`dom_vectors.json`)과 Pug 출력(바이트 단위) · 선택자 일치 수 · 첫 일치의 textContent · 예외 여부를 견준다.
//!
//! 긴 textContent(UTF-16 120 단위 초과)는 `{ n: UTF-16 길이, h: UTF-8 바이트의 FNV-1a 64 }` 로 적혀 있다.

use logis_worker_common::V;
use logis_worker_proxy::dom::Document;
use logis_worker_proxy::pug::convert_html_to_clean_pug;
use serde_json::Value;

fn fnv64(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// 기록과 같은 꼴로
fn text_record(t: &str) -> Value {
    let n = t.encode_utf16().count();
    if n <= 120 {
        Value::String(t.to_string())
    } else {
        serde_json::json!({ "n": n, "h": fnv64(t) })
    }
}

fn actual(doc: &Document, sel: &str) -> Value {
    match doc.query_selector_all(sel) {
        Ok(ids) => {
            let first = ids.first().map(|id| text_record(&doc.text_content(*id))).unwrap_or(Value::Null);
            serde_json::json!([ids.len(), first])
        }
        Err(e) => {
            assert_eq!(e.name, "SyntaxError");
            Value::Null
        }
    }
}

#[test]
fn dom_matches_linkedom() {
    // DOM_VECTORS 로 다른 벡터 파일(예: 더 많은 무작위 사례)을 줄 수 있다
    let path = std::env::var("DOM_VECTORS")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/tests/dom_vectors.json").to_string());
    let data: Value = serde_json::from_str(&std::fs::read_to_string(&path).expect("dom_vectors.json")).unwrap();
    let common: Vec<&str> = data["common"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
    let cases = data["cases"].as_array().unwrap();

    let (mut pug_ok, mut sel_ok, mut sel_total) = (0usize, 0usize, 0usize);
    let mut failures: Vec<String> = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let html = case["html"].as_str().unwrap();

        let pug = convert_html_to_clean_pug(&V::from(html)).to_str();
        let want = case["pug"].as_str().unwrap();
        if pug == want {
            pug_ok += 1;
        } else {
            let at = pug.lines().zip(want.lines()).position(|(a, b)| a != b).unwrap_or(pug.lines().count().min(want.lines().count()));
            failures.push(format!(
                "{name}: pug differs at line {at}\n  rust: {:?}\n  js:   {:?}",
                pug.lines().nth(at),
                want.lines().nth(at)
            ));
        }

        let doc = Document::parse(&format!("<html><body>{html}</body></html>"));
        let extra = case["extra"].as_array().unwrap();
        let pairs = common
            .iter()
            .copied()
            .zip(case["common"].as_array().unwrap())
            .chain(extra.iter().map(|e| (e[0].as_str().unwrap(), &e[1])));
        for (sel, want) in pairs {
            sel_total += 1;
            let got = actual(&doc, sel);
            if &got == want {
                sel_ok += 1;
            } else {
                failures.push(format!("{name}: selector {sel:?}\n  rust: {got}\n  js:   {want}"));
            }
        }
    }
    eprintln!("pug {pug_ok}/{} · selectors {sel_ok}/{sel_total}", cases.len());
    if !failures.is_empty() {
        for f in failures.iter().take(40) {
            eprintln!("{f}");
        }
        panic!("{} mismatches", failures.len());
    }
}

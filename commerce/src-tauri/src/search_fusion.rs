//! 🌟 [WHOLE QUERY FUSION] 질의 전체로 찾은 순위와 질의 조각으로 찾은 순위를 합칩니다.
//!
//!  ── 왜 필요한가 ──
//!   질의 해석은 문장을 조각내 유형과 조건을 붙입니다. 조각마다 유형이 흩어지면 정답 문서는 후보에 들어와도 순위가 밀립니다.
//!   합성 평가에서 의미 질의의 MRR 이 0.11 이었고, 같은 문서를 질의 전체 벡터로만 찾으면 0.72 였습니다.
//!   해석 결과는 그대로 두고, 질의 전체로 찾은 순위를 나란히 세워 해석이 틀려도 그 아래로 떨어지지 않게 합니다.
//!
//!  ── 왜 점수가 아니라 순위인가 ──
//!   트랙마다 눈금이 다릅니다(FTS 고정 +2.0, 벡터 1 - rank × 0.001, 청크 cos × 2.0 ~ 8.0).
//!   눈금을 맞추려면 트랙마다 상수가 필요하지만 순위는 그대로 더할 수 있습니다.

use serde_json::Value;
use std::collections::HashMap;

/// 순위 합산의 완충값. 작을수록 각 트랙의 상위권이 결과를 좌우합니다.
pub const FUSION_K: f32 = 20.0;
/// 글자가 그대로 일치한 근거는 벡터보다 틀릴 일이 적어 두 배로 셉니다.
pub const LITERAL_WEIGHT: f32 = 2.0;

/// 공백으로 나눈 질의 낱말. 앞뒤 구두점만 벗기므로 '20260625-100159' 같은 번호는 그대로 남습니다.
pub fn query_terms(query: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in query.split_whitespace() {
        let term = raw.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        if term.chars().count() < 2 || out.contains(&term) {
            continue;
        }
        // '329,800원' 은 저장값 329800 과 글자가 달라 숫자 부분을 함께 찾습니다.
        let digits: String = term.chars().take_while(|c| c.is_ascii_digit() || *c == ',').filter(|c| *c != ',').collect();
        let rest_has_digit = term.chars().skip_while(|c| c.is_ascii_digit() || *c == ',').any(|c| c.is_ascii_digit());
        if digits.len() >= 4 && digits != term && !rest_has_digit && !out.contains(&digits) {
            out.push(digits);
        }
        out.push(term);
    }
    out
}

/// 저장 JSON 의 값만 모읍니다. 키 이름까지 대조하면 'status' 같은 낱말이 모든 문서에 걸립니다.
pub fn value_text(data_json: &str) -> String {
    fn walk(v: &Value, out: &mut String) {
        match v {
            // 봉투의 해시 주소(cc · bcc · from · to)는 값이 아닙니다.
            Value::String(s) if !s.starts_with("0x") => {
                out.push_str(&s.to_lowercase());
                out.push('\n');
            }
            Value::Number(n) => {
                out.push_str(&n.to_string());
                out.push('\n');
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            Value::Object(o) => o.values().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let mut out = String::new();
    if let Ok(v) = serde_json::from_str::<Value>(data_json) {
        walk(&v, &mut out);
    }
    out
}

/// 질의 낱말이 값에 그대로 들어 있는 문서를 드문 낱말 순으로 세웁니다.
///   '이도윤 고객 주문' 에서는 '이도윤' 이 드물어 '주문' 만 맞는 문서보다 앞섭니다. 동점은 입력 순서를 지킵니다.
pub fn literal_ranking(terms: &[String], docs: &[(String, String)]) -> Vec<String> {
    let n = docs.len() as f32;
    let hits: Vec<Vec<bool>> = docs.iter().map(|(_, text)| terms.iter().map(|t| text.contains(t.as_str())).collect()).collect();
    let idf: Vec<f32> = (0..terms.len())
        .map(|t| {
            let df = hits.iter().filter(|h| h[t]).count() as f32;
            if df == 0.0 { 0.0 } else { ((n + 1.0) / (df + 0.5)).ln().max(0.0) }
        })
        .collect();
    let mut scored: Vec<(f32, usize)> = hits
        .iter()
        .enumerate()
        .map(|(i, h)| (h.iter().zip(idf.iter()).filter(|(hit, _)| **hit).map(|(_, w)| *w).sum::<f32>(), i))
        .filter(|(s, _)| *s > 0.0)
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(_, i)| docs[i].0.clone()).collect()
}

/// 🌟 [QUERY DOMAIN] 질의가 가리키는 문서 유형. 가리키는 낱말이 없으면 상품입니다.
///   주문 · 배송 문서는 상품명을 함께 담아 상품을 찾는 질의에 같이 걸립니다. 합성 평가에서 정답 유형만 놓고 재면 질의 전체 벡터의
///   MRR 이 0.47 에서 0.70 으로 오릅니다. 영문 낱말은 앞부분(orders), 한글은 합성어(주문번호) 를 고려해 포함 여부로 봅니다.
pub fn query_domains(query: &str, vocabulary: &[(String, Vec<String>)]) -> Vec<String> {
    let words: Vec<String> = query.split_whitespace().map(|w| w.to_lowercase()).collect();
    let mut out: Vec<String> = Vec::new();
    for (domain, terms) in vocabulary {
        let named = words.iter().any(|w| {
            terms.iter().any(|t| {
                let t = t.to_lowercase();
                if t.is_ascii() { w.starts_with(&t) } else { w.contains(&t) }
            })
        });
        if named {
            out.push(domain.clone());
        }
    }
    if out.is_empty() {
        out.push("goods".to_string());
    }
    out
}

/// 순위 목록을 (목록, 가중치) 로 받아 역순위 가중합으로 합칩니다. 동점은 id 순이라 실행마다 같습니다.
pub fn fuse(lists: &[(&[String], f32)], k: f32) -> Vec<(String, f32)> {
    let mut score: HashMap<&str, f32> = HashMap::new();
    for (list, weight) in lists {
        for (rank, id) in list.iter().enumerate() {
            *score.entry(id.as_str()).or_insert(0.0) += weight / (k + rank as f32 + 1.0);
        }
    }
    let mut out: Vec<(String, f32)> = score.into_iter().map(|(id, s)| (id.to_string(), s)).collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
    out
}

/// 🌟 [CONDITION GROUNDING] 조건 값이 이 행의 칸에 실제로 있는지 봅니다. 프론트엔드의 contains 와 같이 대소문자를 무시합니다.
pub fn field_contains(row: &Value, field: &str, value: &str) -> bool {
    let want = value.to_lowercase();
    match row.get(field) {
        Some(Value::String(s)) => s.to_lowercase().contains(&want),
        Some(Value::Number(n)) => n.to_string().contains(&want),
        Some(Value::Array(a)) => a.iter().any(|x| x.as_str().map_or(false, |s| s.to_lowercase().contains(&want))),
        _ => false,
    }
}

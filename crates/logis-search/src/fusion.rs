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
/// FTS / Sparse / Vision 트랙 기본 가중치 (Phase 4-2 D3).
pub const FTS_WEIGHT: f32 = 1.2;
pub const SPARSE_WEIGHT: f32 = 1.4;
pub const VISION_WEIGHT: f32 = 1.1;

/// 공백으로 나눈 질의 낱말 + 다국어(CJK 바이그램 · 아랍어 발음기호 제거 · 구조화 번호) 확장.
pub fn query_terms(query: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in query.split_whitespace() {
        let cleaned: String = logis_schema::text::normalize_digits_ascii(raw)
            .chars()
            .filter(|&c| {
                let u = c as u32;
                c != '\u{0640}' && !(0x064B..=0x065F).contains(&u) && u != 0x0670
            })
            .collect();
        let term = cleaned.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        if term.chars().count() < 2 || out.contains(&term) {
            continue;
        }
        // '329,800원' 은 저장값 329800 과 글자가 달라 숫자 부분을 함께 찾습니다.
        let digits: String = term.chars().take_while(|c| c.is_ascii_digit() || *c == ',').filter(|c| *c != ',').collect();
        let rest_has_digit = term.chars().skip_while(|c| c.is_ascii_digit() || *c == ',').any(|c| c.is_ascii_digit());
        if digits.len() >= 4 && digits != term && !rest_has_digit && !out.contains(&digits) {
            out.push(digits);
        }
        out.push(term.clone());
        // Add CJK 2-grams so unspaced Japanese/Chinese queries participate in literal ranking.
        let chars: Vec<char> = term.chars().collect();
        let has_cjk = chars.iter().any(|&c| {
            let u = c as u32;
            (0x4E00..=0x9FFF).contains(&u) || (0x3040..=0x30FF).contains(&u)
        });
        if has_cjk && chars.len() >= 3 {
            for w in chars.windows(2) {
                let bi: String = w.iter().collect();
                if !out.contains(&bi) {
                    out.push(bi);
                }
            }
        }
    }
    out
}

/// 모드 중립 다중 트랙 RRF 융합 입력 (commerce · shipping · all 공용, Phase 4-2 D3).
#[derive(Default)]
pub struct FusionTracks<'a> {
    pub pipeline: &'a [String],
    pub vector: &'a [String],
    pub chunks: &'a [String],
    pub literal: &'a [String],
    pub fts: &'a [String],
    pub sparse: &'a [String],
    pub typed_vector: &'a [String],
    pub typed_chunks: &'a [String],
    pub typed_literal: &'a [String],
    pub typed_sparse: &'a [String],
    pub vision: &'a [String],
}

pub fn fuse_query_tracks(t: &FusionTracks<'_>, k: f32) -> Vec<(String, f32)> {
    let mut lists: Vec<(&[String], f32)> = Vec::with_capacity(11);
    if !t.pipeline.is_empty() { lists.push((t.pipeline, 1.0)); }
    if !t.vector.is_empty() { lists.push((t.vector, 1.0)); }
    if !t.chunks.is_empty() { lists.push((t.chunks, 1.0)); }
    if !t.literal.is_empty() { lists.push((t.literal, LITERAL_WEIGHT)); }
    if !t.fts.is_empty() { lists.push((t.fts, FTS_WEIGHT)); }
    if !t.sparse.is_empty() { lists.push((t.sparse, SPARSE_WEIGHT)); }
    if !t.typed_vector.is_empty() { lists.push((t.typed_vector, 1.0)); }
    if !t.typed_chunks.is_empty() { lists.push((t.typed_chunks, 1.0)); }
    if !t.typed_literal.is_empty() { lists.push((t.typed_literal, 1.0)); }
    if !t.typed_sparse.is_empty() { lists.push((t.typed_sparse, 1.0)); }
    if !t.vision.is_empty() { lists.push((t.vision, VISION_WEIGHT)); }
    fuse(&lists, k)
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
            Value::Object(o) => logis_schema::json_order::sorted_values(o).for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let mut out = String::new();
    if let Ok(v) = serde_json::from_str::<Value>(data_json).map(logis_schema::json_order::sorted) {
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

/// Expands `query_terms` with `premium_discount_nms::normalize::normalize` noun lemmas / stems
/// so Korean particles & conjugated predicates recall literal & sparse search tracks cleanly.
pub fn query_terms_with_nms(query: &str) -> Vec<String> {
    let mut out = query_terms(query);
    let norm = premium_discount_nms::normalize::normalize(query);
    for w in norm.words {
        for cand in [w.surface, w.lemma, w.stem] {
            let clean = cand.trim().to_lowercase();
            if clean.chars().count() >= 2 && !out.contains(&clean) {
                out.push(clean);
            }
        }
    }
    out
}

/// Fuses NMS-normalized literal track with caller-provided vector/FTS/sparse/vision tracks via 4+ track RRF.
pub fn fuse_query_tracks_with_nms(
    query: &str,
    docs: &[(String, String)],
    tracks: FusionTracks<'_>,
    k: f32,
) -> Vec<(String, f32)> {
    let nms_terms = query_terms_with_nms(query);
    let nms_literal = literal_ranking(&nms_terms, docs);
    if tracks.literal.is_empty() && !nms_literal.is_empty() {
        let filled = FusionTracks { literal: &nms_literal, ..tracks };
        return fuse_query_tracks(&filled, k);
    }
    let mut base = fuse_query_tracks(&tracks, k);
    if nms_literal.is_empty() {
        return base;
    }
    let ranked_base: Vec<String> = base.iter().map(|(id, _)| id.clone()).collect();
    base = fuse(&[(&ranked_base, 1.0), (&nms_literal, LITERAL_WEIGHT)], k);
    base
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nms_normalized_terms_boost_korean_noun_recall_in_rrf() {
        let terms = query_terms_with_nms("컨테이너는 선적이 완료된 인보이스");
        assert!(terms.iter().any(|t| t == "컨테이너"));
        assert!(terms.iter().any(|t| t == "선적"));
        let docs = vec![
            ("doc-1".to_string(), "일반 텍스트".to_string()),
            ("doc-2".to_string(), "컨테이너 선적 인보이스 bl-901".to_string()),
        ];
        let fused = fuse_query_tracks_with_nms(
            "컨테이너는 선적이",
            &docs,
            FusionTracks {
                sparse: &["doc-2".to_string(), "doc-1".to_string()],
                fts: &["doc-1".to_string(), "doc-2".to_string()],
                ..Default::default()
            },
            FUSION_K,
        );
        assert_eq!(fused.first().map(|(id, _)| id.as_str()), Some("doc-2"));
    }
}

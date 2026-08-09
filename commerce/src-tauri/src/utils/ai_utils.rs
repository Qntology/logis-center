pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot_product / (norm_a * norm_b) }
}

// =====================================================================
// 🌟 [STANZA IMPROVEMENT] 언어 중립(Language-Agnostic) 형태·구문 판별 엔진
// ---------------------------------------------------------------------
// 판별 근거는 아래 4가지뿐이며, 전부 언어에 무관한 보편 자원입니다.
//   ① UD UPOS 태그셋      (vocab.json 의 pos.upos 배열 — 전 언어 공통 규격)
//   ② UD DEPREL 태그셋    (vocab.json 의 depparse.deprel 배열 — 전 언어 공통 규격)
//   ③ Stanza Lemma 출력   (표면형-원형 차이로 굴절/교착 여부를 모델이 직접 알려줌)
//   ④ 문자열 구조 규칙    (숫자 밀도 / 구분자 — 문자 체계와 무관)
// 특정 언어의 어휘·조사·어미 사전은 일절 사용하지 않습니다.
// =====================================================================

/// [PHASE 1] UD UPOS 중 개체명(PII)이 될 수 없는 태그 — 전 언어 공통
const UPOS_HARD_REJECT: &[&str] = &[
    "VERB", "AUX", "ADV", "ADP", "PART", "SCONJ", "CCONJ", "CONJ",
    "DET", "PRON", "INTJ", "PUNCT", "SYM",
];

/// [PHASE 1] 개체명 후보로 인정되는 핵심 체언 태그 — 전 언어 공통
const UPOS_STRONG_ENTITY: &[&str] = &["NOUN", "PROPN"];

/// [PHASE 1] 단독으로는 개체명 근거가 되지 못하고 보조 근거(구문/복합어)가 필요한 태그
const UPOS_WEAK_ENTITY: &[&str] = &["ADJ", "NUM", "X"];

/// UPOS 서브타입(`NOUN:xxx` 형태) 제거 후 상위 태그만 반환
fn upos_base(tag: &str) -> &str {
    match tag.find(':') {
        Some(i) => &tag[..i],
        None => tag,
    }
}

/// UD DEPREL 서브타입(`nsubj:pass`, `flat:name` 등) 제거 후 소문자 상위 레이블 반환
fn deprel_base(label: &str) -> String {
    let l = label.to_lowercase();
    match l.find(':') {
        Some(i) => l[..i].to_string(),
        None => l,
    }
}

/// [PHASE 2] UD 수식어·기능어 의존관계 → 개체명 기각 근거 (전 언어 공통)
fn is_modifier_deprel(label: &str) -> bool {
    matches!(
        deprel_base(label).as_str(),
        "acl" | "advcl" | "advmod" | "amod" | "aux" | "cop" | "case" | "mark"
            | "cc" | "det" | "discourse" | "expl" | "punct" | "dep"
    )
}

/// [PHASE 2] UD 체언 논항·복합어 의존관계 → 개체명 후보 근거 (전 언어 공통)
fn is_nominal_deprel(label: &str) -> bool {
    matches!(
        deprel_base(label).as_str(),
        "nsubj" | "obj" | "iobj" | "obl" | "nmod" | "flat" | "compound"
            | "appos" | "conj" | "root" | "vocative" | "list" | "nummod"
    )
}

/// [PHASE 4] 구분자 포함 여부 (전화·주민번호 등 식별번호의 보편적 구조 신호)
fn has_identifier_separator(word: &str) -> bool {
    word.chars().any(|c| matches!(c, '-' | '.' | '/' | '+' | '(' | ')' | ' ' | '_'))
}

/// [PHASE 4] 식별번호 후보가 실제 식별번호 '구조'인지 검증.
/// 언어별 단위 사전(도/명/원 등) 없이, 숫자 개수와 구분자 유무만으로 판정하므로
/// 어떤 문자 체계에서도 동일하게 동작합니다. ('38도','119에','24절기','12일' → 전부 기각)
fn is_valid_identifier_shape(word: &str, base_target: &str) -> bool {
    let digits = word.chars().filter(|c| c.is_ascii_digit()).count();
    let sep = has_identifier_separator(word);

    match base_target {
        // 이메일: 로컬파트@도메인 구조와 도메인 내 점(.) 존재를 요구
        "email" => {
            if let Some(at) = word.find('@') {
                let domain = &word[at + 1..];
                !word[..at].is_empty() && domain.contains('.') && domain.len() >= 4
            } else {
                false
            }
        }
        // 주민등록/사회보장번호류: 최소 8자리 이상 숫자. 구분자가 없으면 10자리 이상 요구
        "national_id" => (digits >= 8 && sep) || digits >= 10,
        // 연락처: 구분자 없는 순수 숫자열 9자리 이상, 또는 구분자 포함 7자리 이상
        "contact_number" => (digits >= 9) || (digits >= 7 && sep),
        _ => false,
    }
}

/// [PHASE 3] Stanza Lemma 출력과 표면형을 비교하여 '굴절/교착이 일어난 토큰'인지 판정.
/// 접두 일치(prefix) / 접미 일치(suffix) 양쪽을 모두 검사하므로
/// 교착어(한국어·일본어·터키어), 굴절어(러시아어·독일어), 고립어(중국어)를 함께 커버합니다.
fn is_inflected_surface(surface: &str, lemma: &str) -> bool {
    if lemma.trim().is_empty() { return false; }
    let s: String = surface.chars().filter(|c| c.is_alphanumeric()).collect();
    let l: String = lemma.chars().filter(|c| c.is_alphanumeric()).collect();
    if l.is_empty() || s.is_empty() { return false; }
    if s == l { return false; }
    let sc = s.chars().count();
    let lc = l.chars().count();
    // 원형이 표면형의 앞/뒤에 포함되며 길이가 더 짧다면 굴절이 발생한 것으로 간주
    (sc > lc) && (s.starts_with(&l) || s.ends_with(&l) || s.contains(&l))
}

/// [PHASE 3] Lemma 를 이용한 언어 중립 접미 절단.
/// 언어별 조사/어미 목록 대신 모델이 산출한 원형을 그대로 신뢰하여 잘라냅니다.
fn trim_surface_by_lemma(surface: &str, lemma: &str) -> Option<String> {
    if !is_inflected_surface(surface, lemma) { return None; }
    let l: String = lemma.chars().filter(|c| c.is_alphanumeric()).collect();
    if l.chars().count() < 2 { return None; }
    if surface.starts_with(&l) && surface.chars().count() > l.chars().count() {
        return Some(l);
    }
    if let Some(idx) = surface.rfind(&l) {
        // 원형이 표면형 뒤쪽에 붙은 형태(접두 굴절)는 원형만 남깁니다.
        if idx > 0 { return Some(l); }
    }
    None
}

/// 형태·구문 판별 결과
#[derive(Debug, Clone)]
pub struct MorphVerdict {
    pub accept: bool,
    pub reason: String,
}

impl MorphVerdict {
    fn ok(reason: &str) -> Self { Self { accept: true, reason: reason.to_string() } }
    fn no(reason: &str) -> Self { Self { accept: false, reason: reason.to_string() } }
}

/// [PHASE 1 + 2 + 4] 개체명 후보 최종 판정 (언어 중립).
/// - identifier 계열은 구조 검증으로 대체
/// - PROPN 단독 통과 금지: UD DEPREL 체언 관계 또는 복합 체언 구성일 때만 인정
/// - 전 토큰이 UD 수식어 관계이면 강제 기각
pub fn evaluate_entity_candidacy(
    surface: &str,
    words: &[String],
    tags: &[&str],
    deprels: Option<&[String]>,
    lemmas: Option<&[String]>,
    base_target: &str,
    is_sub_language: bool,
) -> MorphVerdict {
    // ── 0) 식별번호 계열: 형태소가 아니라 '구조'로 판정 (언어 무관) ──
    if matches!(base_target, "email" | "contact_number" | "national_id") {
        return if is_valid_identifier_shape(surface, base_target) {
            MorphVerdict::ok("IDENTIFIER-SHAPE 검증 통과")
        } else {
            MorphVerdict::no("식별번호 구조(숫자 밀도/구분자) 미충족")
        };
    }

    // ── 1) 유효 토큰 수집 (구두점·기호 제외) ──
    let mut core: Vec<usize> = Vec::new();
    for (i, w) in words.iter().enumerate() {
        if i >= tags.len() { break; }
        if w.chars().any(|c| c.is_alphanumeric()) {
            let t = upos_base(tags[i]);
            if t != "PUNCT" && t != "SYM" { core.push(i); }
        }
    }

    if core.is_empty() {
        // 메인 언어와 다른 표기(로마자 약어 등)는 태거 신뢰도가 낮으므로 최소 조건으로 구제
        let alnum = surface.chars().filter(|c| c.is_alphanumeric()).count();
        if is_sub_language && alnum >= 2 {
            return MorphVerdict::ok("SUB-LANGUAGE 구제 (태거 신뢰도 낮음)");
        }
        return MorphVerdict::no("유효 토큰 없음 (전부 구두점/기호)");
    }

    // ── 2) UPOS 1차 게이트 ──
    let mut has_strong = false;
    let mut has_weak = false;
    let mut all_hard_reject = true;
    for &i in &core {
        let t = upos_base(tags[i]);
        if UPOS_STRONG_ENTITY.contains(&t) { has_strong = true; }
        if UPOS_WEAK_ENTITY.contains(&t) { has_weak = true; }
        if !UPOS_HARD_REJECT.contains(&t) { all_hard_reject = false; }
    }

    if all_hard_reject {
        if is_sub_language {
            return MorphVerdict::ok("SUB-LANGUAGE 구제 (UPOS 기각 면제)");
        }
        return MorphVerdict::no("UD UPOS 전량 비체언(용언/부사/조사/접속사 등)");
    }

    // ── 3) UD DEPREL 교차 검증 (Phase 2 핵심) ──
    if let Some(rels) = deprels {
        let mut nominal_support = false;
        let mut modifier_hits = 0usize;
        let mut checked = 0usize;
        for &i in &core {
            if i >= rels.len() { continue; }
            checked += 1;
            if is_nominal_deprel(&rels[i]) { nominal_support = true; }
            else if is_modifier_deprel(&rels[i]) { modifier_hits += 1; }
        }
        if checked > 0 && !nominal_support && modifier_hits == checked {
            return MorphVerdict::no("UD DEPREL 전량 수식어 관계(acl/amod/advmod 등)");
        }
        // PROPN 단독 토큰은 구문상 체언 논항일 때만 인정 (PROPN 남발 차단)
        if core.len() == 1 && !has_weak {
            let i = core[0];
            let t = upos_base(tags[i]);
            if t == "PROPN" && i < rels.len() && !is_nominal_deprel(&rels[i]) {
                return MorphVerdict::no("단일 PROPN 이나 UD DEPREL 체언 근거 부재");
            }
        }
        if nominal_support {
            return MorphVerdict::ok("UD DEPREL 체언 논항 근거 확보");
        }
    }

    // ── 4) DEPREL 미확보 시 폴백: 체언 태그 + 비굴절 여부로 판단 ──
    if !has_strong {
        // 약체언(ADJ/NUM/X) 단독은 개체명 근거로 불충분
        if core.len() >= 2 {
            return MorphVerdict::ok("복합 구성 내 약체언 (보조 근거 인정)");
        }
        if is_sub_language {
            return MorphVerdict::ok("SUB-LANGUAGE 구제 (약체언 단독)");
        }
        return MorphVerdict::no("체언(NOUN/PROPN) 부재 — 약체언 단독은 개체명 불가");
    }

    // 단일 토큰이 굴절형이면(모델 원형 ≠ 표면형) 개체명보다 활용형일 가능성이 높음
    if core.len() == 1 {
        if let Some(lm) = lemmas {
            let i = core[0];
            if i < lm.len() && is_inflected_surface(&words[i], &lm[i]) {
                let t = upos_base(tags[i]);
                if t != "PROPN" {
                    return MorphVerdict::no("단일 굴절형 토큰 (Lemma 상이) — 활용형으로 판단");
                }
            }
        }
    }

    MorphVerdict::ok("UD UPOS 체언 근거 확보")
}

/// [PHASE 2] Depparse ONNX 세션을 실행하여 토큰별 UD DEPREL 레이블을 추출합니다.
/// preprocessor 와 session 을 분리 수신하여 StanzaPipeline 의 필드 단위 대여 충돌을 방지합니다.
pub fn run_depparse_deprels(
    preprocessor: &crate::stanza::StanzaPreprocessor,
    session: &mut onnxruntime::session::Session<'static>,
    words: &[&str],
    pos_ids: &[i64],
) -> Option<Vec<String>> {
    if preprocessor.deprel_vocab.is_empty() || words.is_empty() { return None; }
    if pos_ids.len() < words.len() { return None; }

    // 세션 가변 대여 이전에 사전을 복제하여 라이프타임 충돌을 원천 차단
    let deprel_vocab: Vec<String> = preprocessor.deprel_vocab.clone();

    let inputs = preprocessor
        .encode_to_tensor(words, session, Some(&pos_ids[..words.len()]), None)
        .ok()?;

    let outputs = session.run::<'_, '_, '_, i64, f32, _>(inputs).ok()?;
    if outputs.len() < 2 { return None; }

    let arc = &outputs[0];
    let rel = &outputs[1];
    let arc_shape = arc.shape();
    let rel_shape = rel.shape();
    if arc_shape.len() < 3 || rel_shape.len() < 4 { return None; }

    let seq = words.len().min(arc_shape[1] as usize).min(rel_shape[1] as usize);
    let head_dim = (arc_shape[2] as usize).min(rel_shape[2] as usize);
    let num_rel = rel_shape[3] as usize;
    if seq == 0 || head_dim == 0 || num_rel == 0 { return None; }

    let mut result = Vec::with_capacity(seq);
    for i in 0..seq {
        // 1) 최고 점수 head(지배소) 탐색
        let mut best_head = 0usize;
        let mut best_arc = std::f32::MIN;
        for h in 0..head_dim {
            let v = arc[[0, i, h]];
            if v > best_arc { best_arc = v; best_head = h; }
        }
        // 2) 해당 head 에 대한 최적 DEPREL 레이블 탐색
        let mut best_rel = 0usize;
        let mut best_rel_score = std::f32::MIN;
        for r in 0..num_rel {
            let v = rel[[0, i, best_head, r]];
            if v > best_rel_score { best_rel_score = v; best_rel = r; }
        }
        result.push(deprel_vocab.get(best_rel).cloned().unwrap_or_else(|| "dep".to_string()));
    }
    Some(result)
}

pub fn max_pool_sim(target: &[f32], phrase_embs: &Vec<Vec<f32>>) -> f32 {
    let mut best = 0.0f32;
    for pe in phrase_embs {
        let s = cosine_similarity(target, pe);
        if s > best { best = s; }
    }
    best
}

pub fn split_bias_phrases(raw: &str) -> Vec<String> {
    let mut v: Vec<String> = raw
        .split(|c: char| c == ',' || c == '\n' || c == '/' || c == '|')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let mut seen = std::collections::HashSet::new();
    v.retain(|p| seen.insert(p.clone()));
    if v.len() > 48 { v.truncate(48); }
    v
}

// 🌟 [PHRASE WEIGHTING] 바이어스 문자열을 구(phrase) 단위로 쪼개면서,
// "order 2026-03-15T14:16:35", "order 12345-67890" 같은 숫자 리터럴 예시에는 낮은 가중치를 부여합니다.
// 이 예시들은 '형식 힌트'일 뿐인데 Max-Pool 을 그대로 적용하면
// "수량 | 1", "상품금액 | 35000" 같은 무관한 숫자 라인을 강하게 끌어당겨 오매칭을 만듭니다.
pub fn split_bias_phrases_weighted(raw: &str) -> (Vec<String>, Vec<f32>) {
    let phrases = split_bias_phrases(raw);
    let mut weights = Vec::with_capacity(phrases.len());
    for p in &phrases {
        let compact: Vec<char> = p.chars().filter(|c| !c.is_whitespace()).collect();
        let total = compact.len().max(1);
        let digits = compact.iter().filter(|c| c.is_ascii_digit()).count();
        let ratio = digits as f32 / total as f32;

        if ratio >= 0.25 {
            // 숫자 비중이 높은 순수 예시 리터럴 (형식 힌트 전용)
            weights.push(0.80);
        } else if digits > 0 {
            weights.push(0.95);
        } else {
            // 의미 구(semantic phrase) - 실제 변별력의 원천
            weights.push(1.0);
        }
    }
    (phrases, weights)
}

// 🌟 [WEIGHTED MAX-POOL] 센트로이드(평균) 대신 구 단위 최대 유사도를 사용합니다.
// 거대한 콤마 나열 문자열 하나를 통째로 임베딩하면 모든 개념의 평균이 되어
// 어떤 라인과도 0.0x 수준의 무의미한 값만 나오는 문제를 원천 차단합니다.
pub fn weighted_max_pool_sim(target: &[f32], phrase_embs: &Vec<Vec<f32>>, weights: &Vec<f32>) -> f32 {
    let mut best = 0.0f32;
    for (i, pe) in phrase_embs.iter().enumerate() {
        if pe.iter().all(|&v| v == 0.0) { continue; }
        let w = weights.get(i).copied().unwrap_or(1.0);
        let s = cosine_similarity(target, pe) * w;
        if s > best { best = s; }
    }
    best
}

// 🌟 [UNCAPPED PHRASE SPLIT] split_bias_phrases 는 48개에서 잘라냅니다.
//    bias.json 의 color 뱅크는 50개 언어의 색상명이 수백 개 나열되어 있어
//    48개로 자르면 한국어/영어 이후의 언어(ベージュ, بيج, бежевый ...)가 통째로 소멸합니다.
//    다국어 검색이 목적이므로 속성 뱅크에는 절대 상한을 두지 않습니다.
pub fn split_bias_phrases_full(raw: &str) -> Vec<String> {
    let mut v: Vec<String> = raw
        .split(|c: char| c == ',' || c == '\n' || c == '/' || c == '|')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let mut seen = std::collections::HashSet::new();
    v.retain(|p| seen.insert(p.clone()));
    v
}

// 🌟 [UNCAPPED WEIGHTED SPLIT] split_bias_phrases_weighted 의 상한 제거판.
//    숫자 비중이 높은 순수 예시 리터럴("15000", "2026-03-15")은 형식 힌트일 뿐이므로
//    동일한 규칙으로 가중치만 낮춥니다. (새 상수 도입 아님 — 기존 규칙 재사용)
pub fn split_bias_phrases_weighted_full(raw: &str) -> (Vec<String>, Vec<f32>) {
    let phrases = split_bias_phrases_full(raw);
    let mut weights = Vec::with_capacity(phrases.len());
    for p in &phrases {
        let compact: Vec<char> = p.chars().filter(|c| !c.is_whitespace()).collect();
        let total = compact.len().max(1);
        let digits = compact.iter().filter(|c| c.is_ascii_digit()).count();
        let ratio = digits as f32 / total as f32;

        if ratio >= 0.25 {
            weights.push(0.80);
        } else if digits > 0 {
            weights.push(0.95);
        } else {
            weights.push(1.0);
        }
    }
    (phrases, weights)
}

// 🌟 [SEMANTIC ANCHOR] 필드의 '정체성 문구'(semantic)를 bias.json 에서 언어 중립으로 꺼냅니다.
//    ko.goods.title.semantic = "상품명, 의류명, 제품명, 품목명, 이름" 처럼
//    정답 변별 구가 bias 가 아니라 semantic 에만 존재하는 경우가 많은데
//    (로그의 '가디건' → title 이 정답인 근거는 '의류명' 단 하나입니다)
//    기존 파이프라인은 semantic 을 프롬프트 설명문으로만 쓰고 벡터 공간에는 올리지 않았습니다.
//    루트 전역 노드(color, metrics.*, operators.* ...)까지 깊이 무관 탐색으로 찾아냅니다.
pub fn semantic_anchor_text(doc_lang: &str, page_type: &str, field_name: &str) -> String {
    let dict: &serde_json::Value = &crate::parsing::BIAS_DICT;

    for lk in [doc_lang, "en", "ko"] {
        let lang_node = match dict.get(lk) { Some(v) => v, None => continue };
        if let Some(s) = lang_node
            .get(page_type)
            .and_then(|p| p.get(field_name))
            .and_then(|n| n.get("semantic"))
            .and_then(|v| v.as_str())
        {
            if !s.trim().is_empty() { return s.to_string(); }
        }
        if let Some(s) = lang_node
            .get("default")
            .and_then(|p| p.get(field_name))
            .and_then(|n| n.get("semantic"))
            .and_then(|v| v.as_str())
        {
            if !s.trim().is_empty() { return s.to_string(); }
        }
    }

    let mut stack: Vec<&serde_json::Value> = vec![dict];
    let mut hops = 0usize;
    while let Some(node) = stack.pop() {
        hops += 1;
        if hops > 8192 { break; }
        if let Some(obj) = node.as_object() {
            if let Some(child) = obj.get(field_name) {
                if let Some(s) = child.get("semantic").and_then(|v| v.as_str()) {
                    if !s.trim().is_empty() { return s.to_string(); }
                }
            }
            for (_, v) in obj {
                if v.is_object() { stack.push(v); }
            }
        }
    }

    humanize_url_token(field_name)
}

// 🌟 [CROSS-FIELD AMBIGUITY MASK] bias.json 을 수정하지 않고 런타임에서 무변별 구를 구조적으로 제거합니다.
//    ① 두 개 이상 필드의 bias 뱅크에 '문자 그대로 동일한 구'가 들어 있으면
//       그 구는 어떤 필드도 지목하지 못합니다.
//       (ko.goods 의 title / model_name / brand_name 이 "goods 상품명, goods 상품제목, goods 상품이름" 을
//        완전히 공유 → 로그의 '가디건 → brand_name' 오배정의 직접 원인)
//    ② 자기 필드의 prejudice 에 동일한 구가 존재하면 자기모순입니다.
//       (ko.goods.brand_name 은 bias 와 prejudice 양쪽에 "상품명" 을 갖고 있어 스스로 점수를 깎습니다)
//    문자열 집합 비교이므로 의미 판정(contains)이 아니라 순수 구조 판정이며 상수를 쓰지 않습니다.
pub fn cross_field_ambiguous_phrase_mask(
    bias_banks: &Vec<Vec<String>>,
    prejudice_banks: &Vec<Vec<String>>,
) -> Vec<Vec<bool>> {
    let mut counter: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for bank in bias_banks.iter() {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for p in bank.iter() {
            if seen.insert(p.as_str()) {
                *counter.entry(p.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut out: Vec<Vec<bool>> = Vec::with_capacity(bias_banks.len());
    for (i, bank) in bias_banks.iter().enumerate() {
        let empty: Vec<String> = Vec::new();
        let own_prej: &Vec<String> = prejudice_banks.get(i).unwrap_or(&empty);
        let mut keep: Vec<bool> = bank
            .iter()
            .map(|p| {
                let shared = counter.get(p).copied().unwrap_or(0) > 1;
                let self_contradiction = own_prej.iter().any(|q| q == p);
                !shared && !self_contradiction
            })
            .collect();
        // 전량 탈락 시 뱅크 소멸을 막기 위해 원본을 그대로 유지합니다.
        if keep.iter().all(|k| !*k) { keep = vec![true; bank.len()]; }
        out.push(keep);
    }
    out
}

// 🌟 [DETERMINISTIC CONDITION VALUE] 조건 값은 '벡터가 짚어준 원문 청크' 그 자체입니다.
//    0.6B 모델에게 값 복사를 맡기면 value 키를 통째로 누락시켜 조건이 증발합니다.
//    (로그: color 조건에 value 키가 없어 색상 필터 없이 FTS 가 실행됨)
//    형식이 확정적인 필드는 LLM 없이 코드가 직접 복사합니다.
pub fn deterministic_condition_value(chunks: &Vec<String>, numeric_only: bool) -> String {
    let joined = chunks
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if numeric_only {
        return joined.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
    }
    joined.split_whitespace().collect::<Vec<_>>().join(" ")
}

// 🌟 [QUERY FORMAT GATE] 자연어 질의 청크가 '이 속성의 값이 될 생김새'인지 배정 전에 검증합니다.
//    detect_field_format 이 이미 필드명에서 물리적 형식을 결정론적으로 판정하므로
//    다국어 어휘 리터럴을 단 하나도 쓰지 않고, 숫자 밀도 / 알파벳 존재 / 토큰 길이만 봅니다.
//    (로그: '가디건' 이 supply_price(Number) 2순위로 살아남아 LLM 후보 목록을 오염시켰습니다)
//    - Synthesis(insight/summary) : 합성 문장이라 DB 필터 조건이 될 수 없음
//    - Link                        : 조건 대상이 아님 (호출부에서 이미 제외하지만 방어적으로 차단)
pub fn query_chunk_matches_property(field_name: &str, chunk: &str) -> bool {
    let v = chunk.trim();
    if v.is_empty() { return false; }

    match detect_field_format(field_name) {
        FieldFormat::Synthesis => false,
        FieldFormat::Link => false,
        FieldFormat::Enum => true,
        FieldFormat::Numeric => v.chars().any(|c| c.is_ascii_digit()),
        FieldFormat::Date => v.chars().any(|c| c.is_ascii_digit()),
        FieldFormat::Phone => v.chars().filter(|c| c.is_ascii_digit()).count() >= 7,
        FieldFormat::TrackingCode => longest_code_token_len(v) >= 8,
        FieldFormat::Identifier => longest_code_token_len(v) >= 4,
        FieldFormat::Address => v.chars().any(|c| c.is_alphabetic()),
        FieldFormat::Text => v.chars().any(|c| c.is_alphabetic()),
    }
}

// 🌟 [MAX-COVERAGE GREEDY ASSIGN] 청크가 굶어 죽지 않는 1:1 배타 배정.
//    exclusive_assign_by_score 의 rival 은 '같은 라인에 대한 다른 필드의 최고 점수'입니다.
//    따라서 margin_threshold = 0.0 으로 호출하면
//        margin = own - max_{f'≠f} matrix[f'][l] >= 0  ⟺  own 이 그 라인의 argmax
//    가 되어, 각 라인은 자기 argmax 필드 하나에만 주장을 낼 수 있습니다.
//    그 필드를 더 높은 점수의 다른 라인이 가져가면 차선책으로 이동할 기회 없이 소멸합니다.
//    (로그: '가디건'/'무거운'/'제품중에서'/'제품으로'/'중에서'/'메세지도'/'보여줘' 가 전부 이 경로로 전멸.
//     특히 color 뱅크는 50개 언어 색상명 ~700구라 Max-Pool 이 구조적으로 부풀려져
//     무관한 청크의 argmax 를 독식하는 '흡수 싱크' 로 작동했습니다)
//    여기서는 margin 을 '정렬 기준'이 아니라 '보고용 지표'로만 쓰고,
//    유효한 모든 (필드 × 라인) 주장을 절대 점수 순으로 그리디 배정하여 커버리지를 최대화합니다.
//    matrix[field][line], 음수는 무효 칸. 반환값 = field_idx -> Option<(line_idx, own, margin)>
pub fn greedy_exclusive_assign(matrix: &Vec<Vec<f32>>) -> Vec<Option<(usize, f32, f32)>> {
    let field_count = matrix.len();
    let mut result: Vec<Option<(usize, f32, f32)>> = vec![None; field_count];
    if field_count == 0 { return result; }

    let mut line_count = 0usize;
    for row in matrix.iter() { if row.len() > line_count { line_count = row.len(); } }
    if line_count == 0 { return result; }

    let get = |f: usize, l: usize| -> f32 {
        matrix.get(f).and_then(|row| row.get(l)).copied().unwrap_or(-1.0)
    };

    // 라인별 2순위 점수 (margin 보고용). 무효 칸(-1.0)은 절대 포함되지 않습니다.
    let mut runner_up = vec![-1.0f32; line_count];
    for l in 0..line_count {
        let mut best = -1.0f32;
        let mut second = -1.0f32;
        for f in 0..field_count {
            let v = get(f, l);
            if v < 0.0 { continue; }
            if v > best { second = best; best = v; }
            else if v > second { second = v; }
        }
        runner_up[l] = second;
    }

    let mut claims: Vec<(usize, usize, f32)> = Vec::new();
    for f in 0..field_count {
        for l in 0..line_count {
            let own = get(f, l);
            if own < 0.0 { continue; }
            claims.push((f, l, own));
        }
    }
    claims.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut claimed_lines = vec![false; line_count];
    for (f, l, own) in claims {
        if result[f].is_some() { continue; }
        if claimed_lines[l] { continue; }
        let margin = if runner_up[l] < 0.0 { own } else { own - runner_up[l] };
        result[f] = Some((l, own, margin));
        claimed_lines[l] = true;
    }

    result
}

// 🌟 [SQL-EFFECTIVE FORMAT] 이 속성이 실제 SQL 필터를 바꾸는 '형식 확정' 필드인지 판정합니다.
//    lib.rs 의 convert_conditions_to_sql 이 물리 컬럼으로 매핑하는 것은
//    금액(amount) / 날짜(created_at, updated_at) / 송장(tracking_number LIKE) 계열뿐입니다.
//    문자열 속성(color/title/tags)은 SQL 을 전혀 바꾸지 않으므로
//    N:N 조합에서 별도 쿼리를 발행할 가치가 없고, 조건 완화 티어의 기준이 됩니다.
pub fn is_sql_effective_field(field_name: &str) -> bool {
    matches!(
        detect_field_format(field_name),
        FieldFormat::Date | FieldFormat::Numeric | FieldFormat::TrackingCode
    )
}

// 🌟 [LOCALIZED BIAS NODE] bias.json 의 {lang}.{page_type}.{field} 노드를 원본 그대로 꺼냅니다.
//    스케줄러가 쥐고 있던 bias_target 은 이미 한 덩어리로 합쳐진 짧은 문자열이라
//    콤마가 없어 구 분할이 1개로 끝났고, 그래서 로그의 MaxPoolSim 이 CentroidSim 과
//    소수점 4자리까지 완전히 동일했습니다(= Max-Pool 이 사실상 작동하지 않았음).
//    여기서는 원본 JSON 을 직접 읽어 다국어 구 뱅크를 복원합니다.
fn bias_node(doc_lang: &str, page_type: &str, field_name: &str) -> Option<serde_json::Value> {
    let dict: &serde_json::Value = &crate::parsing::BIAS_DICT;
    let lang_keys = [doc_lang, "en", "ko"];
    for lk in lang_keys {
        let lang_node = match dict.get(lk) { Some(v) => v, None => continue };
        if let Some(n) = lang_node.get(page_type).and_then(|p| p.get(field_name)) {
            return Some(n.clone());
        }
        if let Some(n) = lang_node.get("default").and_then(|p| p.get(field_name)) {
            let raw = n.to_string().replace("{TYPE}", page_type);
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) { return Some(v); }
        }
    }
    None
}

// 🌟 [VALUE EXAMPLE FILTER] bias 안에는 라벨(주문상태)과 값 예시(2026-03-15, 603145678912, 15)가
//    섞여 있습니다. 헤더는 라벨이므로 값 예시를 뱅크에 넣으면 "번호" 같은 헤더가
//    "12345-67890" 에 끌려가는 오탐이 생깁니다. 숫자 비중과 길이로 값 예시를 배제합니다.
pub fn is_value_example_phrase(p: &str) -> bool {
    let compact: Vec<char> = p.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.is_empty() { return true; }
    if compact.len() > 24 { return true; }
    let digits = compact.iter().filter(|c| c.is_ascii_digit()).count();
    digits * 4 >= compact.len()
}

// 🌟 [LABEL PHRASE BANK] semantic(가중치 1.00) + bias 중 비수치 구(가중치 0.92) 로
//    "컬럼 제목에 해당하는 구"만 모읍니다. 동일 문자열 구가 존재하면 코사인이 1.0 이 되므로
//    현재 로그의 +0.0006 ~ +0.0863 마진이 +0.25 이상으로 벌어집니다.
pub fn label_phrase_bank(doc_lang: &str, page_type: &str, field_name: &str) -> (Vec<String>, Vec<f32>) {
    let mut phrases: Vec<String> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    if let Some(node) = bias_node(doc_lang, page_type, field_name) {
        for (key, w) in [("semantic", 1.0f32), ("bias", 0.92f32)] {
            if let Some(raw) = node.get(key).and_then(|v| v.as_str()) {
                for p in split_bias_phrases(raw) {
                    if is_value_example_phrase(&p) { continue; }
                    if phrases.iter().any(|e| e == &p) { continue; }
                    phrases.push(p);
                    weights.push(w);
                }
            }
        }
    }
    if phrases.len() > 48 { phrases.truncate(48); weights.truncate(48); }
    (phrases, weights)
}

// 🌟 [PREJUDICE PHRASE BANK] bias.json 이 필드마다 손으로 써 둔 "이 컬럼이 절대 아닌 라벨" 목록입니다.
//    예) tracking_number.prejudice 에는 "주문번호" 가 리터럴로 들어 있어
//    헤더 '주문번호' 가 운송장번호로 오매핑되는 사고를 코사인 1.0 으로 즉시 차단합니다.
pub fn prejudice_phrase_bank(doc_lang: &str, page_type: &str, field_name: &str) -> Vec<String> {
    let mut phrases: Vec<String> = Vec::new();
    if let Some(node) = bias_node(doc_lang, page_type, field_name) {
        if let Some(raw) = node.get("prejudice").and_then(|v| v.as_str()) {
            for p in split_bias_phrases(raw) {
                if is_value_example_phrase(&p) { continue; }
                if phrases.iter().any(|e| e == &p) { continue; }
                phrases.push(p);
            }
        }
    }
    if phrases.len() > 64 { phrases.truncate(64); }
    phrases
}

// 🌟 [EXCLUSIVE ASSIGNMENT] (필드 × 라인) 유사도 행렬을 받아 상호 배타적 1:1 그리디 매칭을 수행합니다.
// - own    : 해당 필드 바이어스와의 weighted max-pool 유사도
// - rival  : 같은 라인을 노리는 다른 필드들 중 최고 유사도
// - margin : own - rival (경쟁 필드 대비 실제 우위)
//
// 기존 방식(필드마다 독립 argmax)은 "본사" 같은 한 라인을 여러 필드가 중복 점유했고,
// 절대 임계치가 없어 점수 0.0000 짜리 쓰레기 라인도 무조건 힌트로 주입되었습니다.
// 반환값 = field_idx -> Option<(line_idx, own, margin)> / None 이면 "힌트 없음(null 유도)"
pub fn exclusive_assign(
    matrix: &Vec<Vec<f32>>,
    abs_threshold: f32,
    margin_threshold: f32,
) -> Vec<Option<(usize, f32, f32)>> {
    let field_count = matrix.len();
    let mut result: Vec<Option<(usize, f32, f32)>> = vec![None; field_count];
    if field_count == 0 { return result; }

    let mut line_count = 0usize;
    for row in matrix.iter() {
        if row.len() > line_count { line_count = row.len(); }
    }
    if line_count == 0 { return result; }

    let get = |f: usize, l: usize| -> f32 {
        matrix.get(f).and_then(|row| row.get(l)).copied().unwrap_or(-1.0)
    };

    let mut claims: Vec<(usize, usize, f32, f32)> = Vec::new();
    for f in 0..field_count {
        for l in 0..line_count {
            let own = get(f, l);
            if own < abs_threshold { continue; }

            let mut rival = 0.0f32;
            for other in 0..field_count {
                if other == f { continue; }
                let s = get(other, l);
                if s > rival { rival = s; }
            }

            let margin = own - rival;
            if margin < margin_threshold { continue; }
            claims.push((f, l, own, margin));
        }
    }

    // 경쟁 우위(margin)가 큰 순서로, 동률이면 절대 유사도(own)가 큰 순서로 선점시킵니다.
    claims.sort_by(|a, b| {
        b.3.partial_cmp(&a.3)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut claimed_lines = vec![false; line_count];
    for (f, l, own, margin) in claims {
        if result[f].is_some() { continue; }
        if claimed_lines[l] { continue; }
        result[f] = Some((l, own, margin));
        claimed_lines[l] = true;
    }

    result
}

// 🌟 [SCORE-FIRST EXCLUSIVE ASSIGN]
// exclusive_assign 은 '경쟁 마진'이 큰 순서로 선점시키므로, 증거가 약하지만 경쟁자가 없는
// 라벨('판매자')이 증거가 압도적인 라벨('주문하신 분 이름', own 1.0)보다 먼저 필드를 채갑니다.
// 상세 페이지의 (라벨 → 필드) 매핑은 "가장 강한 증거부터 잠근다"가 옳으므로
// 절대 점수(own) 우선, 동률이면 마진 우선으로 정렬합니다.
pub fn exclusive_assign_by_score(
    matrix: &Vec<Vec<f32>>,
    abs_threshold: f32,
    margin_threshold: f32,
) -> Vec<Option<(usize, f32, f32)>> {
    let field_count = matrix.len();
    let mut result: Vec<Option<(usize, f32, f32)>> = vec![None; field_count];
    if field_count == 0 { return result; }

    let mut line_count = 0usize;
    for row in matrix.iter() {
        if row.len() > line_count { line_count = row.len(); }
    }
    if line_count == 0 { return result; }

    let get = |f: usize, l: usize| -> f32 {
        matrix.get(f).and_then(|row| row.get(l)).copied().unwrap_or(-1.0)
    };

    let mut claims: Vec<(usize, usize, f32, f32)> = Vec::new();
    for f in 0..field_count {
        for l in 0..line_count {
            let own = get(f, l);
            if own < abs_threshold { continue; }

            // 🌟 [RIVAL FIX] 후보 자격이 없는 칸(-1.0 같은 무효 표식)을 경쟁자로 세면
            //    margin = own - (-1.0) = own + 1.0 이 되어, 경쟁자가 아예 없는 쓰레기 후보가
            //    가장 강한 주장처럼 정렬됩니다. (로그의 '상품금액'→'bank' Margin +1.0407)
            //    abs_threshold 를 통과한 '실제 후보'만 경쟁자로 인정합니다.
            let mut rival = f32::MIN;
            for other in 0..field_count {
                if other == f { continue; }
                let s = get(other, l);
                if s < abs_threshold { continue; }
                if s > rival { rival = s; }
            }
            let rival = if rival == f32::MIN { abs_threshold } else { rival };

            let margin = own - rival;
            if margin < margin_threshold { continue; }
            claims.push((f, l, own, margin));
        }
    }

    claims.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut claimed_lines = vec![false; line_count];
    for (f, l, own, margin) in claims {
        if result[f].is_some() { continue; }
        if claimed_lines[l] { continue; }
        result[f] = Some((l, own, margin));
        claimed_lines[l] = true;
    }

    result
}

// 🌟 [SELF-POISON GUARD]
// bias.json 의 prejudice 는 "다른 필드 semantic 전부"로 기계 생성되어 있어서
// recipient_address.prejudice 안에 '받는사람' 이, sender_phone.prejudice 안에 '주문자' 가
// 들어가 있습니다. 그 결과 정답 라벨('받으시는 분 주소')이 자기 편견에 맞아 -0.1143 로 자멸합니다.
// 판정 규칙(문자열 비교가 아니라 순수 코사인):
//   편견 구 p 가 '자기 라벨 뱅크'를 경쟁 필드 라벨 뱅크보다 더 잘 설명하면,
//   그 p 는 이 필드의 편견이 될 자격이 없습니다.
pub fn self_poisoned_prejudice_mask(
    own_label_embs: &Vec<Vec<f32>>,
    prej_embs: &Vec<Vec<f32>>,
    all_label_embs: &[Vec<Vec<f32>>],
    self_index: usize,
) -> Vec<bool> {
    let mut mask = vec![false; prej_embs.len()];
    if own_label_embs.is_empty() { return mask; }
    for (pi, pe) in prej_embs.iter().enumerate() {
        if pe.iter().all(|&v| v == 0.0) { continue; }
        let own = max_pool_sim(pe, own_label_embs);
        let mut rival = 0.0f32;
        for (fi, bank) in all_label_embs.iter().enumerate() {
            if fi == self_index { continue; }
            if bank.is_empty() { continue; }
            let s = max_pool_sim(pe, bank);
            if s > rival { rival = s; }
        }
        if own >= rival { mask[pi] = true; }
    }
    mask
}

// 🌟 [SELECT GROUP] 상태(status)는 PUG 로는 절대 판정할 수 없습니다.
//    parsing.rs 의 generate_pug_lines 가 selected 가 아닌 option 을 전부 버리기 때문에
//    '배송완료 / 반품 / 교환' 이라는 열거 후보 집합 자체가 PUG 에서 소멸합니다.
//    따라서 원본 HTML 에서 select 컨트롤과 그 옵션 전체를 따로 수집합니다.
#[derive(Debug, Clone)]
pub struct SelectGroup {
    pub selector: String,     // 실제 CSS selector (LLM 이 복사할 원본)
    pub role_phrase: String,  // name/id 를 자연어화 + 같은 tr 의 th 라벨
    pub options: Vec<String>, // 모든 옵션 텍스트
    pub selected: String,     // selected 된 옵션 텍스트
}

pub fn collect_select_groups(html: &str) -> Vec<SelectGroup> {
    let doc = scraper::Html::parse_document(html);
    let sel_select = match scraper::Selector::parse("select") { Ok(s) => s, Err(_) => return Vec::new() };
    let sel_option = match scraper::Selector::parse("option") { Ok(s) => s, Err(_) => return Vec::new() };
    let sel_th = scraper::Selector::parse("th").ok();

    let mut out: Vec<SelectGroup> = Vec::new();
    for (idx, el) in doc.select(&sel_select).enumerate() {
        let name = el.value().attr("name").unwrap_or("").to_string();
        let id = el.value().attr("id").unwrap_or("").to_string();

        let selector = if !id.is_empty() {
            format!("select#{}", id)
        } else if !name.is_empty() {
            format!("select[name=\"{}\"]", name)
        } else {
            format!("select:nth-of-type({})", idx + 1)
        };

        let mut options: Vec<String> = Vec::new();
        let mut selected = String::new();
        for opt in el.select(&sel_option) {
            let txt = opt.text().collect::<Vec<_>>().join(" ")
                .split_whitespace().collect::<Vec<_>>().join(" ");
            if txt.is_empty() { continue; }
            if opt.value().attr("selected").is_some() && selected.is_empty() {
                selected = txt.clone();
            }
            if !options.iter().any(|o| o == &txt) { options.push(txt); }
        }
        if options.is_empty() { continue; }
        if selected.is_empty() { selected = options[0].clone(); }
        if options.len() > 40 { options.truncate(40); }

        // 역할 문구 : name/id 자연어화 + '같은 tr 안의 th' 라벨
        let mut role = humanize_url_token(&format!("{} {}", name, id));
        let mut cur = el.parent();
        let mut hops = 0usize;
        while let Some(p) = cur {
            hops += 1;
            if hops > 8 { break; }
            if let Some(pe) = p.value().as_element() {
                let tag = pe.name().to_lowercase();
                if tag == "tr" {
                    if let (Some(pref), Some(th_sel)) = (scraper::ElementRef::wrap(p), sel_th.as_ref()) {
                        if let Some(l) = pref.select(th_sel).next() {
                            let t = l.text().collect::<Vec<_>>().join(" ")
                                .split_whitespace().collect::<Vec<_>>().join(" ");
                            if !t.is_empty() { role = format!("{} {}", t, role).trim().to_string(); }
                        }
                    }
                    break;
                }
                if tag == "form" || tag == "body" { break; }
            }
            cur = p.parent();
        }
        if role.trim().is_empty() { role = "select control".to_string(); }

        out.push(SelectGroup { selector, role_phrase: role, options, selected });
    }
    if out.len() > 24 { out.truncate(24); }
    out
}

// 🌟 [STATUS CANONICAL BANK] 상태는 '취소' 라는 한국어 리터럴을 찾는 게 아니라,
//    bias.json 의 status_filters(영어 캐노니컬)와 코사인으로 대조합니다.
//    '배송완료' → complete, '반품' → return, '교환' → exchange 가 다국어 임베딩으로 연결됩니다.
pub fn enum_status_keys(page_type: &str) -> Vec<&'static str> {
    match page_type {
        "tracking" => vec!["draft", "progress", "return", "complete"],
        "goods" => vec!["draft", "show", "hide", "progress", "stop", "cancel", "refund", "return", "exchange", "expire", "complete"],
        "order" => vec!["draft", "progress", "stop", "cancel", "refund", "return", "exchange", "expire", "complete"],
        "coupon" | "event" => vec!["show", "progress", "hide", "stop", "cancel", "expire", "complete"],
        "review" => vec!["progress", "stop", "cancel", "refund", "return", "exchange", "expire", "complete"],
        _ => vec!["show", "progress", "remove", "hide", "stop", "cancel", "refund", "return", "exchange", "expire", "complete"],
    }
}

pub fn status_key_phrases(key: &str) -> Vec<String> {
    let mut v: Vec<String> = vec![key.to_string()];
    if let Some(node) = crate::parsing::BIAS_DICT.get("status_filters").and_then(|s| s.get(key)) {
        if let Some(b) = node.get("bias").and_then(|x| x.as_str()) {
            for p in split_bias_phrases(b) {
                if !v.iter().any(|e| e == &p) { v.push(p); }
            }
        }
    }
    if v.len() > 24 { v.truncate(24); }
    v
}

// 🌟 [FORMAT FAMILY] 스키마 필드가 물리적으로 어떤 "생김새"의 값을 가져야 하는지 분류합니다.
// 다국어 임베딩은 짧은 한국어 문자열끼리 기본 유사도가 0.5를 넘기 때문에
// ("번호" vs "운송장번호" = 0.67) 코사인 임계치만으로는 컬럼을 절대 분리할 수 없습니다.
// 유사도를 재기 "전에" 값의 형태부터 검증해야 오매칭이 원천 차단됩니다.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FieldFormat {
    Date,         // registration_date, payment_date, started_at, expired_at ...
    TrackingCode, // tracking_number, barcode, gtin, mpn
    Identifier,   // id, code, no, index, stock_keeping_unit
    Link,         // link, url
    Numeric,      // price, amount, quantity, weight, fee, discount ...
    Enum,         // status, payment_method, condition, currency, bank, card
    Synthesis,    // insight, summary, analysis (LLM 이 문장을 합성하는 필드)
    // 🌟 [NEW] 연락처는 '문자를 포함해야 한다'는 Text 규칙과 정반대의 생김새(숫자+하이픈)라
    //    Text 로 두면 'test3@gmail.com' 같은 이메일 셀이 형식 게이트를 그대로 통과합니다.
    Phone,        // sender_phone, recipient_phone, telephone, cellphone, number(연락처)
    // 🌟 [NEW] 주소는 반드시 2토큰 이상입니다. '우체국' / 'https://…' 같은 단일 토큰을 원천 차단합니다.
    Address,      // sender_address, recipient_address, address
    Text,         // title, name, description ...
}

pub fn detect_field_format(field_name: &str) -> FieldFormat {
    let lower = field_name.to_lowercase();
    let keys: Vec<String> = lower.split(',').map(|s| s.trim().to_string()).collect();
    let has = |k: &str| keys.iter().any(|x| x == k);

    if keys.iter().any(|k| k.contains("insight") || k.contains("summary") || k.contains("analysis")) {
        return FieldFormat::Synthesis;
    }
    if keys.iter().any(|k| k.contains("tracking_number") || k == "barcode" || k == "gtin" || k == "mpn") {
        return FieldFormat::TrackingCode;
    }
    if has("id") || has("code") || has("no") || has("index") || has("stock_keeping_unit") {
        return FieldFormat::Identifier;
    }
    if keys.iter().any(|k| k.contains("link") || k.contains("url")) {
        return FieldFormat::Link;
    }
    if keys.iter().any(|k| k.contains("date") || k.ends_with("_at")) {
        return FieldFormat::Date;
    }
    // 🌟 tracking_number 는 위에서 이미 반환되었으므로 여기의 "number" 는 순수 연락처입니다.
    if keys.iter().any(|k| {
        k.ends_with("phone") || k == "tel" || k == "telephone" || k == "mobile"
            || k == "cellphone" || k == "contact" || k == "number"
    }) {
        return FieldFormat::Phone;
    }
    if keys.iter().any(|k| k == "address" || k.ends_with("_address")) {
        return FieldFormat::Address;
    }
    if keys.iter().any(|k| {
        k.contains("status") || k.contains("payment_method") || k.contains("payment_origin")
            || k.contains("condition") || k.contains("currency") || k == "bank" || k == "card"
    }) {
        return FieldFormat::Enum;
    }
    if keys.iter().any(|k| {
        k.contains("price") || k.contains("amount") || k.contains("quantity") || k.contains("weight")
            || k == "width" || k == "height" || k == "length" || k.contains("fee")
            || k.contains("discount") || k.contains("usage_") || k.contains("threshold")
            || k.contains("duration")
    }) {
        return FieldFormat::Numeric;
    }
    FieldFormat::Text
}

// 🌟 "a-b-c" / "a/b/c" / "a.b.c" 형태의 실제 날짜 리터럴이 있는지 판정합니다.
// "615600", "9", "26031514155635" 같은 순수 숫자 덩어리는 날짜로 인정하지 않습니다.
pub fn has_date_literal(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    while i < n {
        if chars[i].is_ascii_digit() {
            let start1 = i;
            let mut j = i;
            while j < n && chars[j].is_ascii_digit() { j += 1; }
            let g1 = j - start1;

            if j < n && (chars[j] == '-' || chars[j] == '/' || chars[j] == '.') {
                let sep = chars[j];
                let mut k = j + 1;
                let start2 = k;
                while k < n && chars[k].is_ascii_digit() { k += 1; }
                let g2 = k - start2;

                if g2 >= 1 && k < n && chars[k] == sep {
                    let mut m = k + 1;
                    let start3 = m;
                    while m < n && chars[m].is_ascii_digit() { m += 1; }
                    let g3 = m - start3;
                    // 🌟 [DATE SHAPE GATE] 월(g2)·일(g3)은 물리적으로 최대 2자리.
                    //    "010-3333-3333"(g2=4, g3=4) 같은 전화번호를 날짜로 오인하는 것을 원천 차단.
                    if g3 >= 1 && g1 >= 2 && g1 <= 4 && g2 <= 2 && g3 <= 2 { return true; }
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    false
}

// 🌟 값 안에서 "숫자를 포함한 영숫자 토큰"의 최대 길이를 구합니다. (운송장/코드 판정용)
pub fn longest_code_token_len(s: &str) -> usize {
    let mut best = 0usize;
    for tok in s.split(|c: char| !c.is_alphanumeric()) {
        if !tok.chars().any(|c| c.is_ascii_digit()) { continue; }
        let l = tok.chars().count();
        if l > best { best = l; }
    }
    best
}

// 🌟 값이 실제로 URL(href) 안에 박혀 있는 식별자인지 판정합니다.
// id 의 정의는 "링크로 이어지는 키"이므로, href 풀에 없는 숫자는 id 가 아니라 code 입니다.
pub fn value_token_in_url_pool(value: &str, url_pool: &str) -> bool {
    if url_pool.trim().is_empty() { return false; }
    let pool = url_pool.to_lowercase();
    for tok in value.split(|c: char| !c.is_alphanumeric()) {
        if tok.chars().count() < 4 { continue; }
        if !tok.chars().any(|c| c.is_ascii_digit()) { continue; }
        if pool.contains(&tok.to_lowercase()) { return true; }
    }
    false
}

// 🌟 [MARKUP RESIDUE] 0.6B 모델은 [VECTOR MATCH RESULT] 라인을 복사할 때
//    "tr", "td | 364235" 처럼 구조 태그를 그대로 반환합니다.
//    Text 게이트는 알파벳 2자면 통과시키므로 "tr" 이 정상값으로 저장되어 버립니다.
pub fn is_bare_markup_token(value: &str) -> bool {
    const TAGS: [&str; 31] = [
        "html", "head", "body", "div", "span", "p", "a", "ul", "ol", "li", "dl", "dt", "dd",
        "table", "thead", "tbody", "tfoot", "tr", "td", "th", "form", "input", "select", "option",
        "textarea", "button", "label", "img", "section", "colgroup", "col",
    ];
    let v = value.trim().to_ascii_lowercase();
    if v.is_empty() { return true; }
    if TAGS.contains(&v.as_str()) { return true; }
    if v.contains('|') {
        let head = v.split('|').next().unwrap_or("").trim().to_string();
        let token = head.split(|c: char| c == '[' || c == ' ' || c == '(').next().unwrap_or("");
        if TAGS.contains(&token) { return true; }
    }
    false
}

// 🌟 [MARKUP STRIP] "td | 24120419364235" 처럼 태그 접두어가 붙어 돌아온 답변에서
//    실제 값 부분만 남깁니다. (파이프가 없거나 앞이 태그가 아니면 원문 그대로 보존)
pub fn strip_markup_prefix(value: &str) -> String {
    const TAGS: [&str; 31] = [
        "html", "head", "body", "div", "span", "p", "a", "ul", "ol", "li", "dl", "dt", "dd",
        "table", "thead", "tbody", "tfoot", "tr", "td", "th", "form", "input", "select", "option",
        "textarea", "button", "label", "img", "section", "colgroup", "col",
    ];
    let v = value.trim();
    if let Some(p) = v.find('|') {
        let head = v[..p].trim().to_ascii_lowercase();
        let token = head.split(|c: char| c == '[' || c == ' ' || c == '(').next().unwrap_or("");
        if TAGS.contains(&token) {
            return v[p + 1..].trim().to_string();
        }
    }
    v.to_string()
}

// 🌟 [DATE LITERAL] LLM 을 거치지 않고 벡터가 짚어준 라인에서 날짜 리터럴만 직접 뽑아냅니다.
pub fn extract_date_literal(s: &str) -> Option<String> {
    let re = regex::Regex::new(r"\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?").ok()?;
    re.find(s).map(|m| m.as_str().trim().to_string())
}

// 🌟 [PURE NUMERIC] 열거형(Enum)은 '상태/수단/기관명' 이므로 순수 금액·수량이 될 수 없습니다.
//    '615600원', '(-) 0원', '0' 처럼 숫자와 단위 한 글자로만 이루어진 값을 구조적으로 판별합니다.
//    (특정 통화 문자를 하드코딩하지 않고 '알파벳류 글자 수 <= 1' 로 일반화합니다)
pub fn is_pure_numeric_value(value: &str) -> bool {
    let v = value.trim();
    if v.is_empty() { return false; }
    let digits = v.chars().filter(|c| c.is_ascii_digit()).count();
    if digits == 0 { return false; }
    let letters = v.chars().filter(|c| c.is_alphabetic()).count();
    letters <= 1
}

pub fn value_matches_format(fmt: FieldFormat, value: &str) -> bool {
    let v = value.trim();
    if v.is_empty() { return false; }
    // 어떤 형식이든 구조 태그 잔재는 데이터가 아닙니다.
    if is_bare_markup_token(v) { return false; }
    match fmt {
        FieldFormat::Synthesis => true,
        FieldFormat::Enum => true,
        // 이름·상품명·제목은 문자가 반드시 있어야 합니다. "수량 | 1" 의 "1" 을 여기서 차단합니다.
        FieldFormat::Text => v.chars().any(|c| c.is_alphabetic()) && v.chars().count() >= 2,
        FieldFormat::Numeric => v.chars().any(|c| c.is_ascii_digit()),
        // "번호 | 9", "배송비 | 0" 을 여기서 차단합니다.
        FieldFormat::Date => has_date_literal(v),
        FieldFormat::Link => v.contains('/') || v.to_lowercase().starts_with("http"),
        // "번호 | 11" 을 여기서 차단합니다. 운송장은 최소 8자입니다.
        FieldFormat::TrackingCode => longest_code_token_len(v) >= 8,
        // Identifier 는 url_pool 대조가 본판정이므로 여기서는 최소 길이만 봅니다.
        FieldFormat::Identifier => longest_code_token_len(v) >= 4,
        // 🌟 [PHONE] 숫자 7자리 이상 + 전화번호에 물리적으로 허용되는 문자만.
        //    'test3@gmail.com'(문자·@ 포함) / '주문결제 내역'(숫자 0개) 이 여기서 전멸합니다.
        FieldFormat::Phone => {
            let digits = v.chars().filter(|c| c.is_ascii_digit()).count();
            if digits < 7 { return false; }
            v.chars().all(|c| c.is_ascii_digit() || c.is_whitespace() || "+-().,".contains(c))
        },
        // 🌟 [ADDRESS] 주소는 최소 2토큰 이상입니다.
        //    '우체국' / 'https://m.epost.go.kr/…' 같은 단일 토큰이 여기서 전멸합니다.
        FieldFormat::Address => {
            if v.split_whitespace().count() < 2 { return false; }
            if v.chars().count() < 6 { return false; }
            let lower = v.to_lowercase();
            if lower.starts_with("http") { return false; }
            v.chars().any(|c| c.is_alphabetic())
        },
    }
}

pub fn is_id_link_field(field_name: &str) -> bool {
    let lower = field_name.to_lowercase();
    let keys: Vec<&str> = lower.split(',').map(|s| s.trim()).collect();
    keys.contains(&"id") && keys.contains(&"link")
}

// 🌟 [DOUBLE CENTERING] (필드 × 라인) 원시 유사도에서 라인 고유 베이스라인과 필드 고유 베이스라인을 동시에 제거합니다.
// 원시값이 0.50~0.74 처럼 좁은 구간에 뭉쳐 있어도 상대적 우위가 선명하게 드러나
// 절대 임계치가 비로소 의미를 갖게 됩니다. -1.0 셀은 형식 게이트 탈락(무효)을 의미합니다.
pub fn double_center_matrix(raw: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let field_count = raw.len();
    if field_count == 0 { return Vec::new(); }
    let mut line_count = 0usize;
    for row in raw.iter() { if row.len() > line_count { line_count = row.len(); } }

    let mut out = vec![vec![-1.0f32; line_count]; field_count];
    if line_count == 0 { return out; }

    let mut line_sum = vec![0.0f32; line_count];
    let mut line_cnt = vec![0usize; line_count];
    let mut field_sum = vec![0.0f32; field_count];
    let mut field_cnt = vec![0usize; field_count];
    let mut global_sum = 0.0f32;
    let mut global_cnt = 0usize;

    for f in 0..field_count {
        for l in 0..raw[f].len() {
            let v = raw[f][l];
            if v < 0.0 { continue; }
            line_sum[l] += v; line_cnt[l] += 1;
            field_sum[f] += v; field_cnt[f] += 1;
            global_sum += v; global_cnt += 1;
        }
    }

    let global_mean = if global_cnt > 0 { global_sum / (global_cnt as f32) } else { 0.0 };
    let line_mean: Vec<f32> = (0..line_count)
        .map(|l| if line_cnt[l] > 0 { line_sum[l] / (line_cnt[l] as f32) } else { global_mean })
        .collect();
    let field_mean: Vec<f32> = (0..field_count)
        .map(|f| if field_cnt[f] > 0 { field_sum[f] / (field_cnt[f] as f32) } else { global_mean })
        .collect();

    for f in 0..field_count {
        for l in 0..raw[f].len() {
            let v = raw[f][l];
            if v < 0.0 { continue; }
            out[f][l] = v - line_mean[l] - field_mean[f] + global_mean;
        }
    }
    out
}

// 🌟 [URL SPLIT] href 를 (host, path, query) 로 분해합니다.
//    상대경로/프로토콜생략(//) 형태까지 그대로 받아냅니다.
//    이 분해가 있어야 "도메인 조각(cafe24)" 과 "쿼리값(18)" 을 구조적으로 구분할 수 있습니다.
pub fn split_href_parts(href: &str) -> (String, String, String) {
    let mut rest: &str = href.trim();
    let mut host = String::new();

    if let Some(pos) = rest.find("://") {
        rest = &rest[pos + 3..];
        let end = rest.find(|c: char| c == '/' || c == '?' || c == '#').unwrap_or(rest.len());
        host = rest[..end].to_string();
        rest = &rest[end..];
    } else if rest.starts_with("//") {
        let tmp = &rest[2..];
        let end = tmp.find(|c: char| c == '/' || c == '?' || c == '#').unwrap_or(tmp.len());
        host = tmp[..end].to_string();
        rest = &tmp[end..];
    }

    let no_frag = match rest.find('#') { Some(p) => &rest[..p], None => rest };
    match no_frag.find('?') {
        Some(p) => (host, no_frag[..p].to_string(), no_frag[p + 1..].to_string()),
        None => (host, no_frag.to_string(), String::new()),
    }
}

// 🌟 [HOST REGION] link 문자열에서 호스트 구간이 끝나는 바이트 인덱스를 돌려줍니다.
//    이 인덱스 이전에서 매칭된 조각으로는 절대 URL 패턴을 만들지 않습니다.
pub fn host_region_end(link: &str) -> usize {
    let total = link.len();
    if let Some(p) = link.find("://") {
        let after = p + 3;
        let rest = &link[after..];
        return rest.find(|c: char| c == '/' || c == '?' || c == '#').map(|e| after + e).unwrap_or(total);
    }
    if link.starts_with("//") {
        let rest = &link[2..];
        return rest.find(|c: char| c == '/' || c == '?' || c == '#').map(|e| 2 + e).unwrap_or(total);
    }
    0
}

// 🌟 [URL TOKEN HUMANIZE] "product_no" → "product no", "ProductRegister" → "product register".
//    임베딩 모델이 이해할 수 있는 자연어 구로 바꿔야 코사인 비교가 의미를 갖습니다.
pub fn humanize_url_token(raw: &str) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let mut out = String::new();
    for (i, ch) in chars.iter().enumerate() {
        if ch.is_alphanumeric() {
            let need_space = if i == 0 {
                false
            } else {
                let p = chars[i - 1];
                if p.is_alphanumeric() {
                    (p.is_lowercase() && ch.is_uppercase()) || (p.is_ascii_digit() != ch.is_ascii_digit())
                } else {
                    true
                }
            };
            if need_space && !out.is_empty() && !out.ends_with(' ') { out.push(' '); }
            for lc in ch.to_lowercase() { out.push(lc); }
        } else if !out.is_empty() && !out.ends_with(' ') {
            out.push(' ');
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

// 🌟 [ID/LINK CANDIDATE] href 안에서 식별자로 쓸 수 있는 토큰을,
//    "그 토큰이 URL 상에서 맡은 역할 문구" 와 함께 수집합니다.
//    role_phrase 가 코사인 채점 대상이며, token 은 문자열 포함검사가 아니라 구조 파싱으로 뽑힙니다.
#[derive(Debug, Clone)]
pub struct IdLinkCandidate {
    pub token: String,
    pub href: String,
    pub role_phrase: String,
    pub is_host_part: bool,
    pub prior: f32,
}

// 🌟 [HREF → CANDIDATE] 단일 href 를 (host / path / query) 로 분해해 식별자 후보를 뽑습니다.
//    collect_id_link_candidates(문서 내부 href) 와
//    collect_id_link_candidates_from_url(현재 추출 중인 페이지 주소) 가 공유합니다.
fn push_candidates_from_href(
    href: &str,
    out: &mut Vec<IdLinkCandidate>,
    seen: &mut std::collections::HashSet<String>,
) {
    let (host, path, query) = split_href_parts(href);
    let segs: Vec<String> = path.split('/')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let res_role = segs.last()
        .map(|s| humanize_url_token(s.split('.').next().unwrap_or(s.as_str())))
        .unwrap_or_default();

    // 1) 쿼리 파라미터 값 : 파라미터 키가 그대로 역할 문구가 됩니다. (product_no=18 → "product register product no")
    if !query.is_empty() {
        for pair in query.split('&') {
            let (k, v) = match pair.find('=') {
                Some(p) => (&pair[..p], &pair[p + 1..]),
                None => continue,
            };
            let val = v.trim();
            if val.is_empty() { continue; }
            if !val.chars().any(|c| c.is_ascii_digit()) { continue; }
            if val.chars().count() > 32 { continue; }

            let key_role = humanize_url_token(k);
            if key_role.is_empty() { continue; }
            let role = if key_role.split_whitespace().count() >= 2 || res_role.is_empty() {
                key_role
            } else {
                format!("{} {}", res_role, key_role).trim().to_string()
            };

            let dedup = format!("Q::{}::{}", val, role);
            if !seen.insert(dedup) { continue; }
            out.push(IdLinkCandidate {
                token: val.to_string(),
                href: href.to_string(),
                role_phrase: role,
                is_host_part: false,
                prior: 1.00,
            });
        }
    }

    // 2) 경로 세그먼트 : 직전 세그먼트들이 역할 문구가 됩니다. (/product/view/18 → "product view")
    for (i, seg) in segs.iter().enumerate() {
        let clean = seg.split('.').next().unwrap_or(seg.as_str());
        if clean.is_empty() { continue; }
        if !clean.chars().any(|c| c.is_ascii_digit()) { continue; }
        if clean.chars().count() > 32 { continue; }

        let mut ctx: Vec<String> = Vec::new();
        if i >= 2 { ctx.push(humanize_url_token(&segs[i - 2])); }
        if i >= 1 { ctx.push(humanize_url_token(&segs[i - 1])); }
        let joined = ctx.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");
        let role = if joined.is_empty() { "url path segment".to_string() } else { joined };

        let prior = if i + 1 == segs.len() { 0.92 } else { 0.80 };
        let dedup = format!("P::{}::{}", clean, role);
        if !seen.insert(dedup) { continue; }
        out.push(IdLinkCandidate {
            token: clean.to_string(),
            href: href.to_string(),
            role_phrase: role,
            is_host_part: false,
            prior,
        });
    }

    // 3) 호스트 조각 : 'cafe24' 같은 도메인 파편도 후보로 담되,
    //    도메인 역할 문구를 붙여 코사인이 스스로 떨어뜨리도록 만듭니다.
    for part in host.split('.') {
        if part.is_empty() { continue; }
        if !part.chars().any(|c| c.is_ascii_digit()) { continue; }
        let dedup = format!("H::{}", part);
        if !seen.insert(dedup) { continue; }
        out.push(IdLinkCandidate {
            token: part.to_string(),
            href: href.to_string(),
            role_phrase: "host name domain name website address server address".to_string(),
            is_host_part: true,
            prior: 0.05,
        });
    }
}

pub fn collect_id_link_candidates(lines: &[&str]) -> Vec<IdLinkCandidate> {
    let href_re = match regex::Regex::new(r#"href=["']([^"']+)["']"#) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    let mut hrefs: Vec<String> = Vec::new();
    for line in lines {
        for cap in href_re.captures_iter(line) {
            if let Some(m) = cap.get(1) {
                let v = m.as_str().trim().to_string();
                if v.is_empty() { continue; }
                let lower = v.to_ascii_lowercase();
                if lower.starts_with("javascript:") || lower.starts_with("mailto:") || lower.starts_with("tel:") { continue; }
                if lower == "#" || lower == "#none" { continue; }
                if !hrefs.contains(&v) { hrefs.push(v); }
            }
        }
    }
    if hrefs.is_empty() { return Vec::new(); }

    let mut out: Vec<IdLinkCandidate> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for href in &hrefs {
        push_candidates_from_href(href, &mut out, &mut seen);
    }

    if out.len() > 24 { out.truncate(24); }
    out
}

// 🌟 [PAGE-URL ID PRIORITY] "지금 추출 중인 주소(link)" 자체에서 먼저 식별자를 찾습니다.
//    상세페이지의 주문번호/상품번호는 대부분 문서 안의 a[href] 가 아니라
//    현재 URL 의 쿼리(od_id=24120419364235)에 실려 있습니다.
//    이걸 1순위로 두어야 외부 배송조회 URL(우체국) 쿼리가 id 로 승격되는 사고가 사라집니다.
pub fn collect_id_link_candidates_from_url(page_url: &str) -> Vec<IdLinkCandidate> {
    let mut out: Vec<IdLinkCandidate> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let u = page_url.trim();
    if u.is_empty() { return out; }
    push_candidates_from_href(u, &mut out, &mut seen);
    out.retain(|c| !c.is_host_part);
    if out.len() > 24 { out.truncate(24); }
    out
}

// 🌟 [LABELED TOKEN CANDIDATE] href 가 아예 없는 아이템의 소급 복구용 후보입니다.
//    핵심은 "값" 이 아니라 "그 값이 달린 컬럼 라벨" 을 코사인 채점 대상으로 삼는다는 점입니다.
//    ("상품코드 | P000000P" → 라벨 '상품코드' 를 id,link 라벨 뱅크와 비교)
#[derive(Debug, Clone)]
pub struct LabeledTokenCandidate {
    pub token: String,
    pub label_phrase: String,
}

fn is_structural_tag_label(label: &str) -> bool {
    let l = label.trim().to_lowercase();
    let tag = l.split(|c: char| c == '[' || c == ' ' || c == '(').next().unwrap_or("");
    ["td", "th", "tr", "div", "span", "p", "a", "li", "ul", "ol", "input", "table", "tbody", "thead", "label", "button", "textarea"]
        .contains(&tag)
}

pub fn collect_labeled_token_candidates(labeled_lines: &[String]) -> Vec<LabeledTokenCandidate> {
    let mut out: Vec<LabeledTokenCandidate> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for line in labeled_lines {
        let (label_raw, value) = match line.find('|') {
            Some(p) => (line[..p].trim(), line[p + 1..].trim()),
            None => continue,
        };
        if value.is_empty() { continue; }

        let label = if label_raw.is_empty() || is_structural_tag_label(label_raw) {
            "identifier code number".to_string()
        } else {
            label_raw.to_string()
        };

        for tok in value.split(|c: char| !c.is_alphanumeric()) {
            let n = tok.chars().count();
            if n < 2 || n > 32 { continue; }
            if !tok.chars().any(|c| c.is_ascii_digit()) { continue; }
            let dedup = format!("{}::{}", label, tok);
            if !seen.insert(dedup) { continue; }
            out.push(LabeledTokenCandidate { token: tok.to_string(), label_phrase: label.clone() });
        }
    }

    if out.len() > 48 { out.truncate(48); }
    out
}

// 🌟 [ID SHAPE] 확정된 식별자의 '생김새'(자릿수, 숫자전용 여부)를 학습해 둡니다.
//    goods 목록의 id 가 전부 '18','17','16' 처럼 2자리 숫자였다면
//    'P000000P' 를 URL 패턴에 대입하는 순간 링크가 깨지므로 아예 거부해야 합니다.
pub fn id_shape_signature(token: &str) -> (usize, bool) {
    let n = token.chars().count();
    let digits_only = !token.is_empty() && token.chars().all(|c| c.is_ascii_digit());
    (n, digits_only)
}

pub fn id_shape_allowed(token: &str, learned: &[(usize, bool)]) -> bool {
    if learned.is_empty() { return true; }
    let (n, digits_only) = id_shape_signature(token);

    let mut min_len = usize::MAX;
    let mut max_len = 0usize;
    let mut any_digits_only = false;
    let mut any_mixed = false;
    for (l, d) in learned {
        if *l < min_len { min_len = *l; }
        if *l > max_len { max_len = *l; }
        if *d { any_digits_only = true; } else { any_mixed = true; }
    }

    if digits_only && !any_digits_only { return false; }
    if !digits_only && !any_mixed { return false; }

    let lo = min_len.saturating_sub(2);
    let hi = max_len + 2;
    n >= lo && n <= hi
}

// 🌟 [HOST GUARD] 재구성된 링크가 기준 링크와 같은 호스트인지 검증합니다.
//    'https://breakbot.P000000P.com/...' 같은 도메인 변조를 여기서 최종 차단합니다.
pub fn same_host(a: &str, b: &str) -> bool {
    let (ha, _, _) = split_href_parts(a);
    let (hb, _, _) = split_href_parts(b);
    if ha.is_empty() || hb.is_empty() { return true; }
    ha.eq_ignore_ascii_case(&hb)
}

// 🌟 [DETERMINISTIC ID/LINK - LEGACY FALLBACK] 코사인 후보가 전부 탈락했을 때만 쓰이는 최후 보루입니다.
//    기존과 달리 '호스트 구간' 매칭은 무조건 무시하므로, 도메인 조각(cafe24)이 id 로 승격되는 경로가 사라집니다.
pub fn resolve_id_link_from_lines(lines: &[&str]) -> Option<(String, String)> {
    let href_re = regex::Regex::new(r#"href=["']([^"']+)["']"#).ok()?;

    let mut hrefs: Vec<String> = Vec::new();
    for line in lines {
        for cap in href_re.captures_iter(line) {
            if let Some(m) = cap.get(1) {
                let v = m.as_str().trim().to_string();
                if !v.is_empty() && !hrefs.contains(&v) { hrefs.push(v); }
            }
        }
    }
    if hrefs.is_empty() { return None; }

    let mut tokens: Vec<String> = Vec::new();
    for line in lines {
        let value = match line.find('|') {
            Some(p) => line[p + 1..].trim(),
            None => continue,
        };
        for tok in value.split(|c: char| !c.is_alphanumeric()) {
            if tok.chars().count() < 6 { continue; }
            if !tok.chars().any(|c| c.is_ascii_digit()) { continue; }
            let t = tok.to_string();
            if !tokens.contains(&t) { tokens.push(t); }
        }
    }
    if tokens.is_empty() { return None; }

    let mut best: Option<(String, String)> = None;
    for tok in &tokens {
        let lower_tok = tok.to_ascii_lowercase();
        for h in &hrefs {
            let lower_h = h.to_ascii_lowercase();
            let start = host_region_end(h);
            if start >= lower_h.len() { continue; }
            if !lower_h[start..].contains(&lower_tok) { continue; }

            let is_better = match &best {
                None => true,
                Some((bt, _)) => tok.chars().count() > bt.chars().count(),
            };
            if is_better { best = Some((tok.clone(), h.clone())); }
        }
    }
    best
}

// 🌟 [URL PATTERN EXTRACT] 성공적으로 확정된 id/link 쌍에서 URL 구조 패턴을 추출합니다.
// 예: id="18", link="https://host/disp/admin/shop1/product/ProductRegister?product_no=18"
//   → prefix="https://host/disp/admin/shop1/product/ProductRegister?product_no=", suffix=""
// [CRITICAL] 호스트(도메인) 구간에서 매칭된 조각으로는 절대 패턴을 만들지 않습니다.
//            이 가드가 없으면 prefix="https://breakbot." / suffix=".com/..." 같은 도메인 변조 패턴이 만들어집니다.
pub fn extract_url_pattern(id: &str, link: &str) -> Option<(String, String)> {
    if id.is_empty() || link.is_empty() { return None; }
    if !id.is_ascii() { return None; }

    let lower_link = link.to_ascii_lowercase();
    let lower_id = id.to_ascii_lowercase();

    let host_end = host_region_end(link);
    if host_end >= lower_link.len() { return None; }

    // 식별자는 URL 뒤쪽에 오는 것이 일반적이므로 path/query 구간의 '가장 오른쪽' 매칭을 채택합니다.
    let mut pos_opt: Option<usize> = None;
    let mut cursor = host_end;
    while cursor <= lower_link.len() {
        match lower_link[cursor..].find(&lower_id) {
            Some(rel) => {
                let abs = cursor + rel;
                pos_opt = Some(abs);
                cursor = abs + lower_id.len().max(1);
            },
            None => break,
        }
    }

    let pos = pos_opt?;
    let end = pos + id.len();
    if !link.is_char_boundary(pos) || !link.is_char_boundary(end) { return None; }

    let prefix = &link[..pos];
    let suffix = &link[end..];
    if prefix.is_empty() && suffix.is_empty() { return None; }
    Some((prefix.to_string(), suffix.to_string()))
}

// 🌟 [URL PATTERN APPLY] 추출된 패턴에 새 식별자를 대입하여 link를 생성합니다.
pub fn apply_url_pattern(prefix: &str, suffix: &str, new_id: &str) -> String {
    format!("{}{}{}", prefix, new_id, suffix)
}

// 🌟 [IDENTIFIER TOKEN SEARCH] PUG 라인 배열에서 식별자 후보 토큰을 탐색합니다.
// 조건: 파이프 뒤(value)에 위치, 8자 이상, 숫자 포함, 영숫자 토큰.
// 이제는 코사인 게이트가 모두 실패했을 때의 최후 폴백으로만 호출되며,
// 호출부에서 id_shape_allowed() 생김새 게이트를 반드시 통과해야 실제로 사용됩니다.
pub fn find_identifier_token_in_lines(lines: &[String]) -> Option<String> {
    for line in lines {
        let value = match line.find('|') {
            Some(p) => line[p + 1..].trim(),
            None => continue,
        };
        if value.is_empty() { continue; }
        for tok in value.split(|c: char| !c.is_alphanumeric()) {
            let char_count = tok.chars().count();
            if char_count < 8 { continue; }
            if !tok.chars().any(|c| c.is_ascii_digit()) { continue; }
            // 순수 알파벳 토큰은 제외 (숫자가 반드시 포함되어야 함)
            return Some(tok.to_string());
        }
    }
    None
}

// 🌟 [DEAD HREF] href 가 실제 페이지 이동이 아니라 자바스크립트 훅/레이어 토글인지 판정합니다.
//    '#', '#none', '#layerSnsShare', 'javascript:...' 는 전부 UI 액션이며 실데이터 링크가 아닙니다.
//    이 구분이 없으면 "a[href=\"#none\"] | SMS발송" 이 '링크 데이터'로 보호되어 컬럼을 오염시킵니다.
pub fn is_dead_href(href: &str) -> bool {
    let h = href.trim().to_ascii_lowercase();
    if h.is_empty() { return true; }
    if h.starts_with('#') { return true; }
    if h.starts_with("javascript:") { return true; }
    if h.starts_with("mailto:") || h.starts_with("tel:") { return true; }
    false
}

// 🌟 [REAL HREF] PUG 라인이 실제 상세 페이지로 이동하는 href 를 갖고 있으면 그 값을 돌려줍니다.
//    "a[href=\"/disp/.../ProductRegister?product_no=18\"] | 테스트상품" → Some(...)  ← 실데이터
//    "a[href=\"#none\"] | SMS발송"                                    → None        ← UI 액션
//    "li | 상품 상세보기"                                              → None        ← UI 액션(속성 유실)
pub fn line_real_href(line: &str) -> Option<String> {
    let re = match regex::Regex::new(r#"href=["']([^"']+)["']"#) {
        Ok(r) => r,
        Err(_) => return None,
    };
    for cap in re.captures_iter(line) {
        if let Some(m) = cap.get(1) {
            let v = m.as_str().trim();
            if !is_dead_href(v) { return Some(v.to_string()); }
        }
    }
    None
}

// 🌟 [MULTI VALUE FIELD] 값이 여러 개 이어붙어야 정상인 배열 성격의 필드인지 판정합니다.
//    옵션/태그/구성상품만 공백 병합을 허용하고, title/name 같은 단일 값 필드는
//    반드시 대표값 1개만 채택해야 합니다. (기존 무조건 Join 이 title 오염의 직접 원인)
//    🌟 address 는 우편번호/기본주소/상세주소/참고항목이 서로 다른 input 으로 쪼개져 있으므로
//    반드시 셀 전체를 병합해야 완전한 주소가 됩니다.
pub fn is_multi_value_field(field_name: &str) -> bool {
    let lower = field_name.to_lowercase();
    ["options", "tags", "goods", "additional_goods", "additional_image", "region_restrictions", "address"]
        .iter()
        .any(|k| lower.contains(k))
}

// 🌟 [PUG LINE ANATOMY] PUG 한 줄을 (들여쓰기, 태그명, 속성부, 값) 으로 안전 분해합니다.
//    line.find('|') 는 속성값 내부의 파이프를 먼저 잡아버립니다.
//    (예: option[selected value="우체국|https://m.epost.go.kr/..."] | 우체국)
//    → payment_origin 이 우체국 URL 을 뱉은 직접 원인이며, 여기서 원천 차단합니다.
pub fn pug_line_parts(line: &str) -> (usize, String, String, String) {
    let indent = line.chars().take_while(|c| c.is_whitespace()).count();
    let trimmed = line.trim();
    let chars: Vec<char> = trimmed.chars().collect();

    let mut depth = 0i32;
    let mut in_quote: Option<char> = None;
    let mut pipe_pos: Option<usize> = None;

    for (i, ch) in chars.iter().enumerate() {
        match in_quote {
            Some(q) => { if *ch == q { in_quote = None; } },
            None => {
                if *ch == '"' || *ch == '\'' { in_quote = Some(*ch); }
                else if *ch == '[' { depth += 1; }
                else if *ch == ']' { depth -= 1; }
                else if *ch == '|' && depth <= 0 { pipe_pos = Some(i); break; }
            }
        }
    }

    let (head, value) = match pipe_pos {
        Some(p) => (
            chars[..p].iter().collect::<String>().trim().to_string(),
            chars[p + 1..].iter().collect::<String>().trim().to_string(),
        ),
        None => (trimmed.to_string(), String::new()),
    };

    let tag = head.split(|c: char| c == '[' || c == ' ' || c == '(').next().unwrap_or("").to_lowercase();
    let attrs = match head.find('[') { Some(p) => head[p..].to_string(), None => String::new() };
    (indent, tag, attrs, value)
}

// 🌟 PUG 속성부에서 특정 속성값을 꺼냅니다. (구조 파싱 전용, 의미 판정에는 쓰지 않습니다)
pub fn pug_attr_string(attrs: &str, key: &str) -> Option<String> {
    let pat = format!("{}=\"", key);
    let start = attrs.find(&pat)? + pat.len();
    let rest = &attrs[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

pub fn pug_attr_number(attrs: &str, key: &str) -> Option<usize> {
    pug_attr_string(attrs, key).and_then(|v| v.trim().parse::<usize>().ok())
}

pub fn pug_attr_flag(attrs: &str, key: &str) -> bool {
    attrs.split(|c: char| c == '[' || c == ']' || c == ' ')
        .any(|t| t == key || t.starts_with(&format!("{}=", key)))
}

// 🌟 컬럼 '제목' 역할 태그 (값이 될 수 없는 태그)
pub fn is_label_role_tag(tag: &str) -> bool {
    matches!(tag, "th" | "label" | "legend" | "caption" | "dt"
        | "h1" | "h2" | "h3" | "h4" | "h5" | "h6")
}

// 🌟 값 라인이 될 수 없는 태그 (제목 + 순수 컨테이너)
pub fn is_non_value_role_tag(tag: &str) -> bool {
    if is_label_role_tag(tag) { return true; }
    matches!(tag, "tr" | "table" | "thead" | "tbody" | "tfoot"
        | "colgroup" | "col" | "form" | "button")
}

pub fn is_heading_tag(tag: &str) -> bool {
    matches!(tag, "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "legend" | "caption")
}

// 🌟 [STRUCTURAL LABEL-VALUE PAIR] 상세페이지의 라벨-값 결합 결과
#[derive(Debug, Clone)]
pub struct DetailPair {
    pub label: String,        // 예: "결제방법", "주문상태"
    pub section: String,      // 가장 가까운 상위 제목 (예: "주문하신 분")
    pub value: String,        // 대표값 (단일 값 필드용)
    pub value_all: String,    // 셀 전체 병합값 (주소 등 다중 값 필드용)
    pub primary_line: usize,  // 대표값이 위치한 라인 인덱스
    pub label_line: usize,
}

fn detail_block_end(lines: &[&str], parts: &[(usize, String, String, String)], start: usize) -> usize {
    let base = parts[start].0;
    let mut end = start;
    for j in (start + 1)..lines.len() {
        if lines[j].trim().is_empty() { continue; }
        if parts[j].0 <= base { break; }
        end = j;
    }
    end
}

fn detail_cell_label_text(
    lines: &[&str],
    parts: &[(usize, String, String, String)],
    start: usize,
    end: usize,
) -> String {
    if !parts[start].3.trim().is_empty() { return parts[start].3.trim().to_string(); }
    for j in (start + 1)..=end {
        if lines[j].trim().is_empty() { continue; }
        let t = parts[j].3.trim();
        if t.is_empty() { continue; }
        if parts[j].1 == "input" || parts[j].1 == "button" { continue; }
        return t.to_string();
    }
    String::new()
}

// 🌟 셀 하나의 값을 뽑습니다.
//    [DEPTH GATE] 같은 td 안이라도 '값을 지닌 라인 중 가장 얕은 깊이'에 있는 라인만
//    그 셀의 대표값/병합 대상입니다.
//      <td><a>상품명</a><div class="frm_info"><select>우체국</select><input 123456789><a>배송조회</a></div></td>
//    구조에서 기존 무조건 병합은 "상품명 우체국 123456789 배송조회" 라는 오염값을 만들었습니다.
//    반대로 주소 셀은 우편번호/기본주소/상세주소 input 이 전부 td 직속(동일 깊이)이라
//    병합이 그대로 유지됩니다.
//    rank 3 = 실제 이동 href, 2 = 폼 컨트롤 값(input/option), 1 = 일반 텍스트, 0 = 죽은 href.
//    동일 rank 면 더 긴 텍스트가 대표값이 됩니다. (리스트 경로의 REPRESENTATIVE 규칙과 동일)
fn detail_cell_value_text(
    lines: &[&str],
    parts: &[(usize, String, String, String)],
    start: usize,
    end: usize,
) -> (String, String, usize) {
    // 1차 : 값이 될 자격이 있는 라인만 추려 '최소 깊이'를 확정합니다.
    let mut candidates: Vec<usize> = Vec::new();
    let mut min_indent = usize::MAX;
    for j in start..=end {
        if lines[j].trim().is_empty() { continue; }
        let (indent, tag, attrs, text) = &parts[j];
        if text.trim().is_empty() { continue; }

        // 라벨/버튼/제목은 값이 아닙니다.
        if tag == "label" || tag == "button" || tag == "legend" || tag == "caption" { continue; }
        if tag == "th" && j != start { continue; }
        // 히든 인풋은 화면에 없는 내부 상태값이므로 셀 값 병합에서 제외합니다.
        if tag == "input" && pug_attr_string(attrs, "type").as_deref() == Some("hidden") { continue; }

        if *indent < min_indent { min_indent = *indent; }
        candidates.push(j);
    }
    if candidates.is_empty() { return (String::new(), String::new(), start); }

    let mut best_rank = -1i32;
    let mut best_text = String::new();
    let mut best_line = start;
    let mut joined: Vec<String> = Vec::new();

    for j in candidates {
        // 🌟 최소 깊이보다 깊은 라인 = 그 셀의 '부가 위젯'이므로 대표값에도 병합값에도 넣지 않습니다.
        if parts[j].0 > min_indent { continue; }

        let tag = &parts[j].1;
        let owned = parts[j].3.trim().to_string();

        let rank = if line_real_href(lines[j]).is_some() {
            3
        } else if tag == "input" || tag == "option" || tag == "select" || tag == "textarea" {
            2
        } else if lines[j].contains("href=") {
            0
        } else {
            1
        };

        if !joined.iter().any(|e| e == &owned) { joined.push(owned.clone()); }

        if rank > best_rank || (rank == best_rank && owned.chars().count() > best_text.chars().count()) {
            best_rank = rank;
            best_text = owned;
            best_line = j;
        }
    }

    (best_text, joined.join(" "), best_line)
}

// 🌟 [DETAIL PAIR EXTRACTOR]
//    (a) th[scope="row"] → 같은 tr 의 다음 td   : 폼형 key-value
//    (b) 전부 th 인 행    → 컬럼 헤더로 등록 후 후속 행의 같은 컬럼 td 에 부여
//    (c) input[placeholder] → placeholder 자체가 라벨
//    기존의 "직전 라벨 라인" 휴리스틱은 thead 의 th 끼리도 라벨-값으로 오인했기에 폐기합니다.
pub fn collect_detail_label_value_pairs(lines: &[&str]) -> Vec<DetailPair> {
    let n = lines.len();
    if n == 0 { return Vec::new(); }

    let parts: Vec<(usize, String, String, String)> =
        lines.iter().map(|l| pug_line_parts(l)).collect();

    // 가장 가까운 상위 제목(섹션) 계산
    let mut sections: Vec<String> = vec![String::new(); n];
    {
        let mut cur = String::new();
        for i in 0..n {
            if !lines[i].trim().is_empty() && is_heading_tag(&parts[i].1) && !parts[i].3.trim().is_empty() {
                cur = parts[i].3.trim().to_string();
            }
            sections[i] = cur.clone();
        }
    }

    // 가장 가까운 상위 table 라인
    let enclosing_table = |idx: usize| -> usize {
        let mut target_indent = parts[idx].0;
        for j in (0..idx).rev() {
            if lines[j].trim().is_empty() { continue; }
            let ind = parts[j].0;
            if ind < target_indent {
                if parts[j].1 == "table" { return j; }
                target_indent = ind;
            }
        }
        usize::MAX
    };

    let mut pairs: Vec<DetailPair> = Vec::new();
    let mut table_headers: std::collections::HashMap<usize, std::collections::HashMap<usize, String>> =
        std::collections::HashMap::new();

    for i in 0..n {
        if lines[i].trim().is_empty() { continue; }
        if parts[i].1 != "tr" { continue; }

        let tr_end = detail_block_end(lines, &parts, i);
        if tr_end <= i { continue; }

        let child_indent = {
            let mut ci = None;
            for j in (i + 1)..=tr_end {
                if lines[j].trim().is_empty() { continue; }
                if parts[j].0 > parts[i].0 { ci = Some(parts[j].0); break; }
            }
            match ci { Some(v) => v, None => continue }
        };

        let mut cells: Vec<(usize, String, usize, usize)> = Vec::new();
        let mut col_cursor = 0usize;
        for j in (i + 1)..=tr_end {
            if lines[j].trim().is_empty() { continue; }
            if parts[j].0 != child_indent { continue; }
            let tag = parts[j].1.clone();
            if tag != "td" && tag != "th" { continue; }
            let colspan = pug_attr_number(&parts[j].2, "colspan").unwrap_or(1).max(1);
            let cell_end = detail_block_end(lines, &parts, j).max(j);
            cells.push((j, tag, col_cursor, cell_end));
            col_cursor += colspan;
        }
        if cells.is_empty() { continue; }

        let table_id = enclosing_table(i);
        let all_th = cells.iter().all(|(_, t, _, _)| t == "th");

        if all_th {
            let map = table_headers.entry(table_id).or_insert_with(std::collections::HashMap::new);
            for (line_idx, _, col, cell_end) in &cells {
                let txt = detail_cell_label_text(lines, &parts, *line_idx, *cell_end);
                if txt.is_empty() { continue; }
                map.insert(*col, txt);
            }
            continue;
        }

        let mut pending_label: Option<(String, usize)> = None;
        for (line_idx, tag, col, cell_end) in &cells {
            if tag == "th" {
                let txt = detail_cell_label_text(lines, &parts, *line_idx, *cell_end);
                if !txt.is_empty() { pending_label = Some((txt, *line_idx)); }
                continue;
            }

            let (rep, all_v, prim) = detail_cell_value_text(lines, &parts, *line_idx, *cell_end);
            let (label, label_line) = if let Some((l, li)) = pending_label.clone() {
                (l, li)
            } else if let Some(m) = table_headers.get(&table_id) {
                match m.get(col) { Some(h) => (h.clone(), *line_idx), None => (String::new(), *line_idx) }
            } else {
                (String::new(), *line_idx)
            };
            pending_label = None;

            if label.trim().is_empty() || rep.trim().is_empty() { continue; }
            pairs.push(DetailPair {
                label: label.trim().to_string(),
                section: sections[label_line].clone(),
                value: rep,
                value_all: all_v,
                primary_line: prim,
                label_line,
            });
        }
    }

    // (c) placeholder 기반 보조 페어 : "개별 운송장번호" 처럼 th 가 없는 값의 유일한 라벨입니다.
    //     🌟 [PLACEHOLDER DEDUPE] 단, 그 라인을 이미 th 기반 라벨이 소유하고 있다면
    //     placeholder 는 '보조 안내문'일 뿐 라벨이 아닙니다.
    //     Line 165 는 th '가맹점 ID' 가 있는데 placeholder '없음' 이 중복 생성되어
    //     payment_origin ← '없음' → 값 'test1' 오염을 만들었고,
    //     Line 178 도 th '입금자명' 위에 '실 입금자명' 이 겹쳤습니다.
    //     반대로 Line 75 는 th 페어가 없으므로 placeholder 가 유일한 라벨이라 반드시 보존됩니다.
    let structural_lines: std::collections::HashSet<usize> =
        pairs.iter().map(|p| p.primary_line).collect();

    for i in 0..n {
        if lines[i].trim().is_empty() { continue; }
        if parts[i].1 != "input" && parts[i].1 != "textarea" { continue; }
        let ph = match pug_attr_string(&parts[i].2, "placeholder") { Some(v) => v, None => continue };
        if ph.trim().is_empty() { continue; }
        let v = parts[i].3.trim().to_string();
        if v.is_empty() { continue; }
        if structural_lines.contains(&i) { continue; }
        pairs.push(DetailPair {
            label: ph.trim().to_string(),
            section: sections[i].clone(),
            value: v.clone(),
            value_all: v,
            primary_line: i,
            label_line: i,
        });
    }

    pairs
}

pub fn extract_pug_context(lines: &[&str], target_idx: usize) -> String {
    if lines.is_empty() { return String::new(); }
    let mut parent_idx = target_idx;
    let target_indent = lines[target_idx].chars().take_while(|c| c.is_whitespace()).count();
    
    for i in (0..target_idx).rev() {
        let indent = lines[i].chars().take_while(|c| c.is_whitespace()).count();
        if indent < target_indent && !lines[i].trim().is_empty() {
            parent_idx = i;
            break;
        }
    }

    let parent_indent = lines[parent_idx].chars().take_while(|c| c.is_whitespace()).count();
    let mut context_lines = vec![lines[parent_idx]];
    
    for i in (parent_idx + 1)..lines.len() {
        if lines[i].trim().is_empty() { continue; }
        let indent = lines[i].chars().take_while(|c| c.is_whitespace()).count();
        if indent <= parent_indent {
            break;
        }
        context_lines.push(lines[i]);
    }
    
    context_lines.join("\n")
}
//! 필드별 라벨 · 선입견 구 뱅크 (bias.json 의 semantic · bias · prejudice, 12개 언어 순서). 예전 utils/ai_utils.rs 에서 옮김.

use logis_schema::text::split_bias_phrases;

// 🌟 [LOCALIZED BIAS NODE] bias.json 의 {lang}.{page_type}.{field} 노드를 원본 그대로 꺼냅니다.
//    스케줄러가 쥐고 있던 bias_target 은 이미 한 덩어리로 합쳐진 짧은 문자열이라
//    콤마가 없어 구 분할이 1개로 끝났고, 그래서 로그의 MaxPoolSim 이 CentroidSim 과
//    소수점 4자리까지 완전히 동일했습니다(= Max-Pool 이 사실상 작동하지 않았음).
//    여기서는 원본 JSON 을 직접 읽어 다국어 구 뱅크를 복원합니다.
pub fn bias_node(doc_lang: &str, page_type: &str, field_name: &str) -> Option<serde_json::Value> {
    let dict: &serde_json::Value = &crate::bias_schema::BIAS_DICT;
    let canon = crate::bias_schema::canonical_bias_type(page_type);
    let primary = crate::bias_schema::lang_code_of(doc_lang);
    // Phase 4-1 D3: `zh-tw` falls back to `zh` before `en` so Traditional Chinese doesn't drop Chinese bias nodes.
    let mut lang_keys: Vec<&str> = vec![primary.as_str()];
    if primary == "zh-tw" {
        lang_keys.push("zh");
    }
    lang_keys.push("en");
    lang_keys.push("ko");
    for lk in lang_keys {
        let lang_node = match dict.get(lk) { Some(v) => v, None => continue };
        if let Some(n) = lang_node.get(page_type).and_then(|p| p.get(field_name)) {
            return Some(n.clone());
        }
        if canon != page_type {
            if let Some(n) = lang_node.get(canon).and_then(|p| p.get(field_name)) {
                return Some(n.clone());
            }
        }
        if let Some(n) = lang_node.get("default").and_then(|p| p.get(field_name)) {
            let raw = logis_schema::json_order::to_string_sorted(n).replace("{TYPE}", page_type);
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw).map(logis_schema::json_order::sorted) { return Some(v); }
        }
    }
    None
}

// 🌟 [VALUE EXAMPLE FILTER] bias 안에는 라벨(주문상태)과 값 예시(2026-03-15, 603145678912, 15)가
//    섞여 있습니다. 헤더는 라벨이므로 값 예시를 뱅크에 넣으면 "번호" 같은 헤더가
//    "12345-67890" 에 끌려가는 오탐이 생깁니다. 숫자 비중과 길이로 값 예시를 배제합니다.
pub fn is_value_example_phrase(p: &str) -> bool {
    let norm = logis_schema::text::normalize_digits_ascii(p);
    let compact: Vec<char> = norm.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.is_empty() { return true; }
    // Phase 4-1 D3: 24 chars for CJK/Hangul/Kana scripts, 42 chars for Latin/Arabic/Cyrillic multi-word headers (fixes empty `cs` nodes).
    let has_cjk = compact.iter().any(|&c| {
        let u = c as u32;
        (0x3040..=0x30FF).contains(&u)
            || (0x3400..=0x9FFF).contains(&u)
            || (0xAC00..=0xD7AF).contains(&u)
    });
    let max_len = if has_cjk { 24 } else { 42 };
    if compact.len() > max_len { return true; }
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

pub const BANK_LANGS: [&str; 12] = ["en", "de", "es", "fr", "ja", "pt", "ar", "cs", "it", "ko", "nl", "zh"];

pub fn bank_lang_order(primary_lang: &str) -> Vec<String> {
    let dict: &serde_json::Value = &crate::bias_schema::BIAS_DICT;
    let mut order: Vec<String> = Vec::with_capacity(BANK_LANGS.len() + 1);
    let primary = crate::bias_schema::lang_code_of(primary_lang);
    if dict.get(primary.as_str()).is_some() {
        order.push(primary);
    }
    for l in BANK_LANGS.iter() {
        if dict.get(*l).is_none() { continue; }
        if !order.iter().any(|x| x.as_str() == *l) { order.push((*l).to_string()); }
    }
    if order.is_empty() {
        order.push("en".to_string());
    }
    order
}

pub fn bank_lang_coverage() -> (Vec<&'static str>, Vec<&'static str>) {
    let dict: &serde_json::Value = &crate::bias_schema::BIAS_DICT;
    let mut have: Vec<&'static str> = Vec::new();
    let mut miss: Vec<&'static str> = Vec::new();
    for l in BANK_LANGS.iter() {
        match dict.get(*l) {
            Some(_) => have.push(*l),
            None => miss.push(*l),
        }
    }
    (have, miss)
}

pub fn label_phrase_bank_multilingual(primary_lang: &str, page_type: &str, field_name: &str) -> (Vec<String>, Vec<f32>) {
    let mut phrases: Vec<String> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    for lk in bank_lang_order(primary_lang).into_iter() {
        let (ph, wt) = label_phrase_bank(&lk, page_type, field_name);
        for (p, w) in ph.into_iter().zip(wt.into_iter()) {
            if phrases.iter().any(|e| e == &p) { continue; }
            phrases.push(p);
            weights.push(w);
        }
    }
    (phrases, weights)
}

pub fn prejudice_phrase_bank_multilingual(primary_lang: &str, page_type: &str, field_name: &str) -> Vec<String> {
    let mut phrases: Vec<String> = Vec::new();
    for lk in bank_lang_order(primary_lang).into_iter() {
        for p in prejudice_phrase_bank(&lk, page_type, field_name) {
            if phrases.iter().any(|e| e == &p) { continue; }
            phrases.push(p);
        }
    }
    phrases
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

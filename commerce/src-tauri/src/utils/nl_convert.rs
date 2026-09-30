/// Converts a JSON Value into a human-readable natural language narrative.
/// [STRICT ALIGNMENT] This logic perfectly synchronizes with every column in `parsing.rs`.
pub fn json_to_natural_language(json_val: &serde_json::Value) -> String {
    let mut sentences = Vec::new();

    // 최상위 식별자(ID, Link 등)를 먼저 추출하여 문장 생성
    if let Some(obj) = json_val.as_object() {
        if let Some(id) = obj.get("id").or(obj.get("no")).and_then(|v| v.as_str()) {
            sentences.push(format!("The unique identifier is {}.", id));
        }
        if let Some(link) = obj.get("link").or(obj.get("path")).and_then(|v| v.as_str()) {
            sentences.push(format!("It can be accessed at {}.", link));
        }
    }

    fn parse_node(val: &serde_json::Value, context_name: &str, sentences: &mut Vec<String>) {
        match val {
            serde_json::Value::Object(map) => {
                let title = map.get("title").or(map.get("name")).and_then(|v| v.as_str()).unwrap_or("");
                let item_type = map.get("type").and_then(|v| v.as_str()).unwrap_or("");

                let mut intro = String::new();
                if !title.is_empty() {
                    let type_str = if item_type.is_empty() { "item" } else { item_type };
                    intro.push_str(&format!("This {} is titled '{}'.", type_str, title));
                } else if !context_name.is_empty() && context_name != "item" {
                    intro.push_str(&format!("Regarding {}.", context_name));
                }

                let intro_at = sentences.len();
                let intro_waits = title.is_empty();
                if !intro.is_empty() && !intro_waits && !sentences.contains(&intro) {
                    sentences.push(intro.clone());
                }

                for (key, v) in map {
                    if [
                        "title", "name", "type", "currency", "text", "masked_text",
                        "json_data", "data", "id", "index", "no", "link", "path",
                        "origin", "mode", "detail",
                        "updated_at", "created_at", "updated_at_ts", "created_at_ts",
                        "digest", "vector", "vision_vec", "from", "to", "cc", "bcc", "ref",
                        "is_masked", "tier", "score", "views",
                        "embed", "node", "item",
                        "table",
                        "doc_type",
                        "ledger", "relay_origin", "_relay_bound",
                    ].contains(&key.as_str()) { continue; }
                    if key.starts_with("rel_") { continue; }
                    if crate::store::ENVELOPE_COLUMNS.contains(&key.as_str()) { continue; }
                    if v.is_null() || (v.is_string() && v.as_str().unwrap_or("").trim().is_empty()) { continue; }
                    if crate::utils::canonical::is_relay_index_key(key)
                        && (!crate::utils::canonical::relay_value_is_content(v)
                            || map.iter().any(|(k2, v2)| v2 == v && k2.starts_with(&format!("{}_", key))))
                    {
                        continue;
                    }

                    let clean_key = match crate::utils::canonical::relay_companion_base(key) {
                        Some(base) => base.to_string(),
                        None => key.replace("_", " "),
                    };

                    if v.is_object() || v.is_array() {
                        parse_node(v, &clean_key, sentences);
                    } else {
                        let val_str = match v {
                            serde_json::Value::String(s) => s.clone(),
                            serde_json::Value::Number(n) => n.to_string(),
                            serde_json::Value::Bool(b) => b.to_string(),
                            _ => String::new(),
                        };

                        if ["sale_price", "supply_price", "price", "amount", "shipping_fee", "discount"].contains(&key.as_str()) {
                            let curr = map.get("currency").and_then(|c| c.as_str()).unwrap_or("");
                            let curr_str = if curr.is_empty() { String::new() } else { format!(" {}", curr) };
                            sentences.push(format!("The {} is {}{}.", clean_key, val_str, curr_str));
                        } else if key == "status" {
                            sentences.push(format!("It is currently in '{}' status.", val_str));
                        } else {
                            sentences.push(format!("Its {} is {}.", clean_key, val_str));
                        }
                    }
                }
                if !intro.is_empty()
                    && intro_waits
                    && sentences[intro_at..].iter().any(|x| !sentences[..intro_at].contains(x))
                    && !sentences[..intro_at].contains(&intro)
                {
                    sentences.insert(intro_at, intro);
                }
            },
            serde_json::Value::Array(arr) => {
                let mut arr_vals = Vec::new();
                for item in arr {
                    if item.is_string() || item.is_number() || item.is_boolean() {
                        arr_vals.push(item.as_str().map(|s| s.to_string()).unwrap_or_else(|| item.to_string()));
                    } else {
                        parse_node(item, context_name, sentences);
                    }
                }
                if !arr_vals.is_empty() {
                    sentences.push(format!("The {} includes: {}.", context_name, arr_vals.join(", ")));
                }
            },
            _ => {
                let val_str = val.as_str().map(|s| s.to_string()).unwrap_or_else(|| val.to_string());
                if !val_str.is_empty() {
                    sentences.push(format!("The {} is {}.", context_name, val_str));
                }
            }
        }
    }

    parse_node(json_val, "item", &mut sentences);

    let mut unique_sentences = Vec::new();
    for s in sentences {
        if !unique_sentences.contains(&s) { unique_sentences.push(s); }
    }

    unique_sentences.join(" ").replace("  ", " ").trim().to_string()
}

/// [PHASE A] json_to_natural_language() 출력을 문장/속성 단위 청크로 분할합니다.
///
/// 분할 규칙:
///   R1: 문장 경계(". ")를 1차 분할 기준으로 사용
///   R2: 단일 문장이 150자 초과 시 콤마/접속사 기준으로 2차 분할
///   R3: 분할된 각 청크는 최소 2 단어 이상 (1 단어 청크는 인접 청크에 병합)
///   R4: 식별자/URL/숫자 리터럴 포함 청크는 독립 청크로 보존
///   R5: JSON에 없는 상대어/형용사를 청크에 추가하지 않음
///
/// 반환값: Vec<(chunk_text, property_name)>
///   - chunk_text: 자연어 청크 원문 (마침표 제거)
///   - property_name: 해당 청크가 나타내는 JSON 필드명 (snake_case)
pub fn split_natural_language_to_chunks(text: &str) -> Vec<(String, String, bool)> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    // ── Step 1: 문장 경계 분할 ──
    // ". " (마침표 + 공백) 기준으로 1차 분할합니다.
    // json_to_natural_language() 의 출력은 이미 문장 단위로 구분되어 있으므로
    // 이 분할이 1:1 문장 대응이 됩니다.
    let raw_sentences: Vec<&str> = text
        .split(". ")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    // ── Step 2: 각 문장에서 속성명 추출 + 배열 분할 ──
    // 세 번째 요소 bool: true = JSON 구조 패턴에서 결정론적 확정 (전처리 경로)
    //                  false = 폴백/미분류 (검색 경로와 동일 취급)
    let mut chunks: Vec<(String, String, bool)> = Vec::new();

    for sentence in &raw_sentences {
        let mut s = sentence.trim().trim_end_matches('.').trim();
        if s.is_empty() {
            continue;
        }

        let mut peeled = 0usize;
        while let Some(rest) = s.strip_prefix("Regarding ") {
            let (name, tail) = match rest.find(|c: char| c == ',' || c == '.') {
                Some(p) => (rest[..p].trim(), rest[p + 1..].trim()),
                None => (rest.trim(), ""),
            };
            if !name.is_empty() {
                chunks.push((
                    format!("Regarding {}", name),
                    "context_intro".to_string(),
                    false,
                ));
            }
            peeled += 1;
            s = tail;
            if s.is_empty() {
                break;
            }
        }
        if peeled > 0 && s.is_empty() {
            continue;
        }

        // 패턴 1: "The unique identifier is {value}"
        // → property = "id", confirmed = true
        if let Some(rest) = s.strip_prefix("The unique identifier is ") {
            chunks.push((format!("The unique identifier is {}", rest), "id".to_string(), true));
            continue;
        }

        // 패턴 2: "It can be accessed at {value}"
        // → property = "link", confirmed = true
        if let Some(rest) = s.strip_prefix("It can be accessed at ") {
            chunks.push((format!("It can be accessed at {}", rest), "link".to_string(), true));
            continue;
        }

        // 패턴 3: "This {type} is titled '{title}'"
        // → property = "title", confirmed = true
        if s.contains("is titled '") {
            chunks.push((s.to_string(), "title".to_string(), true));
            continue;
        }

        // 패턴 4: "It is currently in '{value}' status"
        // → property = "status", confirmed = true
        if s.contains("is currently in '") && s.ends_with("status") {
            chunks.push((s.to_string(), "status".to_string(), true));
            continue;
        }

        // 패턴 5: "The {context} includes: {val1}, {val2}, ..."
        // → 배열 값을 개별 청크로 분할, property = context (snake_case 변환), confirmed = true
        if let Some(includes_pos) = s.find(" includes: ") {
            let context_part = &s[..includes_pos];
            let values_part = &s[includes_pos + " includes: ".len()..];

            // "The {context}" 에서 "The " 제거 후 snake_case 변환
            let context_raw = context_part.trim().strip_prefix("The ").unwrap_or(context_part.trim());
            let property = context_raw.to_lowercase().replace(' ', "_");

            // 콤마로 배열 값 분할
            let values: Vec<&str> = split_clause_commas(values_part)
                .into_iter()
                .map(|v| v.trim())
                .filter(|v| !v.is_empty())
                .collect();

            if values.len() == 1 {
                chunks.push((format!("{} includes {}", context_raw, values[0]), property, true));
            } else {
                for val in &values {
                    chunks.push((format!("{} includes {}", context_raw, val), property.clone(), true));
                }
            }
            continue;
        }

        // 패턴 6 은 위 [CONTEXT INTRO PEEL] 로 이관했습니다.
        // 도입부는 값이 없는 '문맥 표지' 일 뿐이므로 청크가 되어서는 안 되고,
        // 뒤따르는 값 문장을 삼켜서도 안 됩니다. 두 요구를 한 자리에서 처리합니다.

        // 패턴 7: "Its {key} is {value}"
        // → property = key (snake_case), confirmed = true
        if let Some(rest) = s.strip_prefix("Its ") {
            if let Some(is_pos) = rest.find(" is ") {
                let key_part = rest[..is_pos].trim();
                let val_part = rest[is_pos + " is ".len()..].trim();
                let property = key_part.to_lowercase().replace(' ', "_");
                chunks.push((format!("Its {} is {}", key_part, val_part), property, true));
                continue;
            }
        }

        // 패턴 8: "The {key} is {value} {currency}"
        // → property = key (snake_case), confirmed = true
        if let Some(rest) = s.strip_prefix("The ") {
            if let Some(is_pos) = rest.find(" is ") {
                let key_part = rest[..is_pos].trim();
                let val_part = rest[is_pos + " is ".len()..].trim();
                let property = key_part.to_lowercase().replace(' ', "_");
                chunks.push((format!("The {} is {}", key_part, val_part), property, true));
                continue;
            }
        }

        // 패턴 매칭 실패 시: 전체 문장을 "unclassified" 로 보관, confirmed = false
        chunks.push((s.to_string(), "unclassified".to_string(), false));
    }

    // ── Step 3: 150자 초과 문장 2차 분할 ──
    let mut expanded: Vec<(String, String, bool)> = Vec::new();

    /// [라벨 접두어 추출] "Its {key} is {value}" 또는 "The {key} is {value}" 패턴에서
    /// 라벨 부분("Its traffic insight is")을 추출합니다.
    /// 분할 시 후속 조각에 이 접두어를 복원하여 라벨 소실을 방지합니다.
    fn extract_label_prefix(chunk_text: &str) -> Option<String> {
        let s = chunk_text.trim();

        // "Its {key} is {value}" 패턴
        if let Some(rest) = s.strip_prefix("Its ") {
            if let Some(is_pos) = rest.find(" is ") {
                let prefix = format!("Its {} is", &rest[..is_pos]);
                return Some(prefix);
            }
        }

        // "The {key} is {value}" 패턴
        if let Some(rest) = s.strip_prefix("The ") {
            if let Some(is_pos) = rest.find(" is ") {
                let prefix = format!("The {} is", &rest[..is_pos]);
                return Some(prefix);
            }
        }

        // "It is currently in '{value}' status" 패턴
        if s.starts_with("It is currently in '") {
            return Some("It is currently in".to_string());
        }

        // "This {type} is titled '{value}'" 패턴
        if s.contains("is titled '") {
            if let Some(titled_pos) = s.find("is titled '") {
                let prefix = &s[..titled_pos + "is titled '".len()];
                return Some(prefix.trim_end_matches('\'').to_string());
            }
        }

        // "{context} includes: {values}" 패턴
        if let Some(includes_pos) = s.find(" includes: ") {
            let prefix = &s[..includes_pos + " includes: ".len()];
            return Some(prefix.to_string());
        }

        None
    }

    for (chunk_text, property, confirmed) in &chunks {
        let synthesis = crate::utils::ai_utils::detect_field_format(property)
            == crate::utils::ai_utils::FieldFormat::Synthesis;
        if chunk_text.chars().count() > 150 && !synthesis {
            let parts: Vec<&str> = split_clause_commas(chunk_text)
                .into_iter()
                .map(|p| p.trim())
                .filter(|p| !p.is_empty())
                .collect();

            if parts.len() > 1 {
                let label_prefix = extract_label_prefix(chunk_text);

                for (pi, part) in parts.iter().enumerate() {
                    if pi == 0 {
                        expanded.push((part.to_string(), property.clone(), *confirmed));
                        continue;
                    }
                    let restored = if let Some(prefix) = &label_prefix {
                        let clean_part = part
                            .trim_start_matches("and ")
                            .trim_start_matches("with ")
                            .trim_start_matches("but ");
                        format!("{} {}", prefix, clean_part)
                    } else {
                        part.to_string()
                    };
                    let sub_confirmed = *confirmed && label_prefix.is_some();
                    expanded.push((restored, property.clone(), sub_confirmed));
                }
                continue;
            }
        }
        expanded.push((chunk_text.clone(), property.clone(), *confirmed));
    }

    // ── Step 4: 1 단어 청크 병합 (R3) ──
    // 단어 수가 1개인 청크는 직전 청크에 병합합니다.
    // 단, 식별자/URL/숫자 리터럴 포함 청크(R4)는 독립 보존합니다.
    let mut merged: Vec<(String, String, bool)> = Vec::new();

    for (chunk_text, property, confirmed) in &expanded {
        let word_count = chunk_text.split_whitespace().count();
        let has_identifier = chunk_text.contains("http")
            || chunk_text.contains('/')
            || chunk_text.chars().any(|c| c.is_ascii_digit());

        if word_count < 2 && !has_identifier && !merged.is_empty() {
            // 직전 청크에 병합
            let last = merged.last_mut().unwrap();
            last.0.push(' ');
            last.0.push_str(chunk_text);
            // 병합 시: 양쪽 모두 confirmed 여야만 confirmed 유지.
            // 한쪽이라도 미확인이면 병합 결과가 불확실하므로 false.
            last.2 = last.2 && *confirmed;
        } else {
            merged.push((chunk_text.clone(), property.clone(), *confirmed));
        }
    }

    merged
}

/// [PHASE A - 테스트 헬퍼] 분할 결과를 디버그 로그로 출력합니다.
/// scheduler.rs 에서 인덱싱 파이프라인 호출 시 사용합니다.
/// confirmed=true 인 청크는 JSON 구조에서 결정론적으로 확정된 것입니다.
pub fn log_chunk_split_result(chunks: &[(String, String, bool)]) {
    let confirmed_count = chunks.iter().filter(|(_, _, c)| *c).count();
    let unconfirmed_count = chunks.len() - confirmed_count;
    println!(
        "  📝 [PHASE A] RAW-CHUNK 분할 결과: {}개 청크 (✓confirmed: {} | ?unconfirmed: {})",
        chunks.len(), confirmed_count, unconfirmed_count
    );
    for (i, (text, prop, confirmed)) in chunks.iter().enumerate() {
        let flag = if *confirmed { "✓" } else { "?" };
        println!("    [{}] {} property='{}' | text='{}'", i, flag, prop, text);
    }
}

pub fn split_clause_commas(s: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let mut out: Vec<&str> = Vec::new();
    let mut start = 0usize;
    for k in 0..chars.len() {
        let (bi, ch) = chars[k];
        if ch != ',' { continue; }
        let prev_digit = k > 0 && chars[k - 1].1.is_ascii_digit();
        let next_digit = chars.get(k + 1).map_or(false, |x| x.1.is_ascii_digit());
        if prev_digit && next_digit { continue; }
        out.push(&s[start..bi]);
        start = bi + ch.len_utf8();
    }
    out.push(&s[start..]);
    out
}

pub fn is_clause_fragment(part: &str) -> bool {
    let mut body = part.trim();
    for p in ["and ", "or ", "with ", "but "] {
        if let Some(rest) = body.strip_prefix(p) {
            body = rest.trim_start();
            break;
        }
    }
    let words: Vec<&str> = body.split_whitespace().collect();
    if words.len() < 3 {
        return true;
    }
    !words
        .iter()
        .any(|w| w.chars().filter(|c| c.is_alphabetic()).count() >= 2)
}

// =====================================================================
// 🌟 [PHASE B] 인덱싱 전용 청크 메타데이터 구조체 + 속성 태깅 강화 + NMS 중복 제거
// =====================================================================

/// [PHASE B] 인덱싱 파이프라인에서 LanceDB item_chunks 테이블에 저장할
/// 하나의 청크가 가져야 할 전체 메타데이터입니다.
///
/// 필드 설명:
///   - chunk_text:      자연어 청크 원문 (PHASE A 출력)
///   - property:        PLINKO 확정 속성명 (snake_case)
///   - property_format: detect_field_format() 결과 문자열 ("Numeric", "Text", "Enum" 등)
///   - bias_phrases:    bias.json 에서 해당 필드의 semantic + bias 구 목록
///   - prejudice_phrases: bias.json 에서 해당 필드의 prejudice 구 목록
///   - value_part:      청크에서 추출한 실제 값 부분 (파이프 뒤)
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    pub chunk_text: String,
    pub property: String,
    pub property_format: String,
    pub bias_phrases: Vec<String>,
    pub prejudice_phrases: Vec<String>,
    pub value_part: String,
    /// true = Phase A 패턴 매칭으로 JSON 구조에서 결정론적으로 확정된 청크.
    /// false = 패턴 매칭 실패 또는 검색 경로에서 유입된 청크.
    pub confirmed: bool,
}

/// [PHASE B] detect_field_format() 결과를 문자열로 변환합니다.
///
/// 🌟 [DUPLICATION REMOVED] 기존에는 ai_utils.rs::detect_field_format 의 판정 로직을
///    이 함수에 통째로 복제해 두었습니다. "nl_convert.rs 의 독립성" 이 명분이었지만,
///    실제로는 같은 규칙 두 벌이 서로 다르게 늙어가는 통로가 되었습니다.
///    (무역 필드 추가 시 ai_utils 만 고치면 인덱싱 경로가 조용히 구버전 판정을 유지)
///    이 함수의 계약은 '문자열 반환' 이지 '판정 로직 보유' 가 아니므로,
///    판정은 단일 진실의 원천에 위임하고 여기서는 이름만 붙입니다.
///    FieldFormat 의 변종과 반환 문자열은 1:1 대응이라 동작 변화가 없습니다.
pub fn field_format_to_string(field_name: &str) -> String {
    use crate::utils::ai_utils::{detect_field_format, FieldFormat};
    match detect_field_format(field_name) {
        FieldFormat::Synthesis => "Synthesis",
        FieldFormat::TrackingCode => "TrackingCode",
        FieldFormat::Identifier => "Identifier",
        FieldFormat::Link => "Link",
        FieldFormat::Date => "Date",
        FieldFormat::Phone => "Phone",
        FieldFormat::Address => "Address",
        FieldFormat::Enum => "Enum",
        FieldFormat::Numeric => "Numeric",
        FieldFormat::Text => "Text",
    }
    .to_string()
}

/// [PHASE B] bias.json 에서 해당 필드의 semantic + bias 구를 동적으로 읽어옵니다.
/// 다국어 하드코딩 없이 bias.json 의 기존 구조만 사용합니다.
///
/// 반환값: (bias_phrases, prejudice_phrases)
///   - bias_phrases: semantic 앵커 + bias 구를 split_bias_phrases_full 로 분할한 목록
///   - prejudice_phrases: prejudice 구를 분할한 목록
fn get_field_bias_phrases(doc_lang: &str, page_type: &str, field_name: &str) -> (Vec<String>, Vec<String>) {
    let canon = crate::utils::bias_schema::canonical_bias_type(page_type);
    let is_trade = crate::utils::bias_schema::is_trade_doc_type(page_type);

    let (mut bias_phrases, _weights) = if is_trade {
        crate::model::merge::owner_label_bank(doc_lang, canon, field_name)
    } else {
        crate::utils::ai_utils::label_phrase_bank_multilingual(doc_lang, canon, field_name)
    };

    let mut prejudice_phrases =
        crate::utils::ai_utils::prejudice_phrase_bank_multilingual(doc_lang, canon, field_name);

    if bias_phrases.is_empty() {
        let dict: &serde_json::Value = &crate::parsing::BIAS_DICT;
        let mut stack: Vec<&serde_json::Value> = vec![dict];
        let mut hops = 0usize;
        while let Some(node) = stack.pop() {
            hops += 1;
            if hops > 4096 { break; }
            if let Some(obj) = node.as_object() {
                if let Some(child) = obj.get(field_name) {
                    for key in ["semantic", "bias"] {
                        if let Some(b) = child.get(key).and_then(|v| v.as_str()) {
                            for p in crate::utils::ai_utils::split_bias_phrases_full(b) {
                                if !bias_phrases.iter().any(|e| e == &p) {
                                    bias_phrases.push(p);
                                }
                            }
                        }
                    }
                    if let Some(p) = child.get("prejudice").and_then(|v| v.as_str()) {
                        for ph in crate::utils::ai_utils::split_bias_phrases_full(p) {
                            if !prejudice_phrases.iter().any(|e| e == &ph) {
                                prejudice_phrases.push(ph);
                            }
                        }
                    }
                    if !bias_phrases.is_empty() { break; }
                }
                for (_, v) in obj {
                    if v.is_object() { stack.push(v); }
                }
            }
        }
    }

    (bias_phrases, prejudice_phrases)
}

/// [PHASE B] 청크 텍스트에서 "Its {key} is {value}" / "The {key} is {value}" 패턴의
/// 값 부분만 추출합니다. 형식 검증과 임베딩 시 값 부분을 별도로 사용하기 위한 것입니다.
///
/// 추출 규칙 (정규식 없이 구조 파싱):
///   "Its weight is 1.5"       → "1.5"
///   "The sale price is 29900 KRW" → "29900 KRW"
///   "It is currently in 'show' status" → "show"
///   "This goods is titled '테스트상품'" → "테스트상품"
///   "tags includes 가전"       → "가전"
///   매칭 실패 시 전체 텍스트 반환
pub fn extract_value_from_chunk(chunk_text: &str) -> String {
    let s = chunk_text.trim();

    // "Its {key} is {value}"
    if let Some(rest) = s.strip_prefix("Its ") {
        if let Some(is_pos) = rest.find(" is ") {
            return rest[is_pos + " is ".len()..].trim().to_string();
        }
    }

    // "The {key} is {value}"
    if let Some(rest) = s.strip_prefix("The ") {
        if let Some(is_pos) = rest.find(" is ") {
            return rest[is_pos + " is ".len()..].trim().to_string();
        }
    }

    // "It is currently in '{value}' status"
    if let Some(start) = s.find("in '") {
        if let Some(end) = s[start + 4..].find('\'') {
            return s[start + 4..start + 4 + end].to_string();
        }
    }

    // "This {type} is titled '{value}'"
    if let Some(start) = s.find("titled '") {
        if let Some(end) = s[start + 8..].find('\'') {
            return s[start + 8..start + 8 + end].to_string();
        }
    }

    // "{context} includes {value}"
    if let Some(inc_pos) = s.find(" includes ") {
        return s[inc_pos + " includes ".len()..].trim().to_string();
    }

    // "The unique identifier is {value}"
    if let Some(rest) = s.strip_prefix("The unique identifier is ") {
        return rest.trim().to_string();
    }

    // "It can be accessed at {value}"
    if let Some(rest) = s.strip_prefix("It can be accessed at ") {
        return rest.trim().to_string();
    }

    // 폴백: 전체 텍스트
    s.to_string()
}

/// [PHASE B] PHASE A 출력(Vec<(chunk_text, property)>)을 받아
/// 각 청크에 형식, bias/prejudice 구, 값 부분을 부여한 ChunkMetadata 배열을 반환합니다.
///
/// 이 함수는 LLM 을 호출하지 않으며, bias.json 동적 읽기 + 구조 파싱만으로 동작합니다.
///
/// # 인자
///   - raw_chunks: split_natural_language_to_chunks() 반환값
///   - doc_lang:   문서 언어 코드 ("ko", "en" 등)
///   - page_type:  도메인 타입 ("goods", "order", "tracking" 등)
///
/// # 반환
///   Vec<ChunkMetadata> — 각 청크의 완전 메타데이터
pub fn enrich_chunks_with_metadata(
    raw_chunks: &[(String, String, bool)],
    doc_lang: &str,
    page_type: &str,
) -> Vec<ChunkMetadata> {
    let mut results: Vec<ChunkMetadata> = Vec::with_capacity(raw_chunks.len());

    for (chunk_text, property, confirmed) in raw_chunks {
        let property_format = field_format_to_string(property);
        let (bias_phrases, prejudice_phrases) = get_field_bias_phrases(doc_lang, page_type, property);
        let value_part = extract_value_from_chunk(chunk_text);

        results.push(ChunkMetadata {
            chunk_text: chunk_text.clone(),
            property: property.clone(),
            property_format,
            bias_phrases,
            prejudice_phrases,
            value_part,
            confirmed: *confirmed,
        });
    }

    results
}

pub fn nms_battle_for_indexing(chunks: Vec<ChunkMetadata>) -> Vec<ChunkMetadata> {
    if chunks.len() <= 1 {
        return chunks;
    }

    let mut survivors: Vec<ChunkMetadata> = Vec::with_capacity(chunks.len());
    let mut removed_count = 0usize;

    for chunk in chunks {
        // ── D1: 완전 동일 텍스트 ──
        let is_exact_dup = survivors.iter().any(|s| s.chunk_text == chunk.chunk_text);
        if is_exact_dup {
            removed_count += 1;
            continue;
        }

        // ── D2: 동일 property + 동일 value_part ──
        let is_prop_val_dup = survivors.iter().any(|s| {
            s.property == chunk.property && s.value_part == chunk.value_part
        });
        if is_prop_val_dup {
            removed_count += 1;
            continue;
        }

        // ── D3: 동일 property + value_part 포함 관계 ──
        // confirmed 청크는 비confirmed 청크에 의해 절대 폐기되지 않음
        let same_prop_absorbed = survivors.iter().any(|s| {
            if s.property != chunk.property { return false; }
            // confirmed 보호: 기존 survivor가 confirmed이면 현재 청크가 폐기 대상
            if s.confirmed && !chunk.confirmed { return true; }
            // 현재 청크가 confirmed이면 기존 survivor를 폐기해야 하므로 여기서는 false
            if chunk.confirmed && !s.confirmed { return false; }
            // 둘 다 confirmed 또는 둘 다 비confirmed: value_part 포함 관계로 판정
            if s.value_part.is_empty() || chunk.value_part.is_empty() { return false; }
            s.value_part.contains(&chunk.value_part) || chunk.value_part.contains(&s.value_part)
        });
        if same_prop_absorbed {
            removed_count += 1;
            continue;
        }

        // ── D3 역방향: 기존 survivor 중 동일 property이면서 현재 청크에 포함되는
        //            비confirmed 청크가 있으면 폐기 ──
        let mut to_remove_prop: Vec<usize> = Vec::new();
        for (si, survivor) in survivors.iter().enumerate() {
            if survivor.property != chunk.property { continue; }
            if survivor.confirmed && !chunk.confirmed { continue; }
            if survivor.value_part.is_empty() || chunk.value_part.is_empty() { continue; }
            if chunk.value_part.contains(&survivor.value_part)
                && chunk.value_part != survivor.value_part
            {
                to_remove_prop.push(si);
            }
        }
        if !to_remove_prop.is_empty() {
            removed_count += to_remove_prop.len();
            to_remove_prop.sort_unstable();
            to_remove_prop.dedup();
            for &idx in to_remove_prop.iter().rev() {
                survivors.remove(idx);
            }
        }

        // ── D4: 텍스트 부분문자열 포함 (기존 규칙) ──
        // confirmed 보호: confirmed 청크는 비confirmed에 의해 폐기되지 않음
        let has_identifier = chunk.chunk_text.contains("http")
            || chunk.chunk_text.contains('/')
            || chunk.chunk_text.chars().any(|c| c.is_ascii_digit());

        if !has_identifier && !chunk.confirmed {
            let is_substring_of_existing = survivors.iter().any(|s| {
                s.chunk_text.contains(&chunk.chunk_text)
                    && s.chunk_text != chunk.chunk_text
            });
            if is_substring_of_existing {
                removed_count += 1;
                continue;
            }
        }

        // ── D4 역방향: 기존 survivor 중 현재 청크에 포함되는 짧은 비confirmed 청크 폐기 ──
        let mut to_remove_indices: Vec<usize> = Vec::new();
        for (si, survivor) in survivors.iter().enumerate() {
            if survivor.confirmed && !chunk.confirmed { continue; }
            let survivor_has_id = survivor.chunk_text.contains("http")
                || survivor.chunk_text.contains('/')
                || survivor.chunk_text.chars().any(|c| c.is_ascii_digit());
            if !survivor_has_id
                && chunk.chunk_text.contains(&survivor.chunk_text)
                && chunk.chunk_text != survivor.chunk_text
            {
                to_remove_indices.push(si);
            }
        }
        if !to_remove_indices.is_empty() {
            removed_count += to_remove_indices.len();
            to_remove_indices.sort_unstable();
            to_remove_indices.dedup();
            for &idx in to_remove_indices.iter().rev() {
                survivors.remove(idx);
            }
        }

        // ── D5: 동일 property 내 confirmed 청크 간 정보량 경쟁 ──
        // 현재 청크가 confirmed이고, 기존 survivor에 동일 property confirmed가 있으면
        // 텍스트 길이(정보량)가 더 긴 쪽만 생존
        let mut to_remove_d5: Vec<usize> = Vec::new();
        for (si, survivor) in survivors.iter().enumerate() {
            if survivor.property != chunk.property { continue; }
            if !survivor.confirmed || !chunk.confirmed { continue; }
            // 둘 다 confirmed + 동일 property: 정보량(텍스트 길이) 비교
            if survivor.chunk_text.chars().count() < chunk.chunk_text.chars().count() {
                to_remove_d5.push(si);
            }
        }
        if !to_remove_d5.is_empty() {
            removed_count += to_remove_d5.len();
            to_remove_d5.sort_unstable();
            to_remove_d5.dedup();
            for &idx in to_remove_d5.iter().rev() {
                survivors.remove(idx);
            }
        }

        survivors.push(chunk);
    }

    if removed_count > 0 {
        println!(
            "  ⚔️ [NMS BATTLE / INDEXING] {}개 중복 청크 제거 → {}개 생존",
            removed_count,
            survivors.len()
        );
    }

    survivors
}

/// [PHASE B - FORMAT GATE] 형식 검증 게이트입니다.
/// 청크의 value_part 가 해당 property_format 과 물리적으로 일치하는지 확인합니다.
/// 불일치 시 해당 청크를 "unclassified" 로 강등시킵니다.
///
/// 이 검증은 배정 전(행렬 구축 시점)에 수행하는 설계 원칙을 따릅니다.
/// LLM 호출 없이 순수 구조 판정만 사용합니다.
///
/// # 인자
///   - chunks: nms_battle_for_indexing() 반환값
///
/// # 반환
///   형식 검증 통과 후 Vec<ChunkMetadata> (불일치 청크는 property="unclassified" 로 변경)
/// [PHASE B - FORMAT GATE] 형식 검증 게이트입니다.
/// 청크의 value_part 가 해당 property_format 과 물리적으로 일치하는지 확인합니다.
/// 불일치 시 해당 청크를 "unclassified" 로 강등시킵니다.
///
/// 🌟 [CONFIRMED BYPASS] confirmed = true 인 청크는 JSON 구조에서 확정된 것이므로
///    형식 게이트를 통과하지 못해도 강등하지 않습니다.
///    (예: "Its sale_price is 12300" 에서 value_part 파싱이 미세하게 어긋나도
///     JSON 원본에서 sale_price = 12300 이 확정되어 있으므로 보호합니다)
///
/// 이 검증은 배정 전(행렬 구축 시점)에 수행하는 설계 원칙을 따릅니다.
/// LLM 호출 없이 순수 구조 판정만 사용합니다.
pub fn format_gate_for_indexing(mut chunks: Vec<ChunkMetadata>) -> Vec<ChunkMetadata> {
    let mut rejected_count = 0usize;
    let mut confirmed_bypass_count = 0usize;

    for chunk in chunks.iter_mut() {
        let val = chunk.value_part.trim();
        if val.is_empty() {
            continue;
        }

        let identifier_like = chunk.property_format == "Text"
            && crate::utils::ai_utils::query_value_format(&chunk.property)
                == crate::utils::ai_utils::FieldFormat::Identifier;
        let passes = match chunk.property_format.as_str() {
            "Numeric" => val.chars().any(|c| c.is_ascii_digit()),
            "Date" => {
                // 날짜 리터럴(YYYY-MM-DD 등) 또는 숫자 포함
                val.chars().any(|c| c.is_ascii_digit())
            },
            "Phone" => val.chars().filter(|c| c.is_ascii_digit()).count() >= 7,
            "TrackingCode" => {
                // 8자 이상 영숫자 토큰 존재
                val.split(|c: char| !c.is_alphanumeric())
                    .any(|tok| tok.chars().count() >= 8 && tok.chars().any(|c| c.is_ascii_digit()))
            },
            "Identifier" => {
                val.split(|c: char| !c.is_alphanumeric())
                    .any(|tok| tok.chars().count() >= 4 && tok.chars().any(|c| c.is_ascii_digit()))
            },
            "Link" => val.contains('/') || val.to_lowercase().starts_with("http"),
            "Enum" => true, // Enum 은 어떤 값이든 허용
            "Text" => {
                val.chars().any(|c| c.is_alphabetic())
                    || (identifier_like
                        && val
                            .split(|c: char| !c.is_alphanumeric())
                            .any(|tok| tok.chars().count() >= 4 && tok.chars().any(|c| c.is_ascii_digit())))
            },
            "Address" => val.chars().any(|c| c.is_alphabetic()) && val.split_whitespace().count() >= 2,
            "Synthesis" => true,
            _ => true,
        };

        if !passes {
            // 🌟 confirmed 청크는 형식 불일치여도 강등하지 않습니다.
            if chunk.confirmed {
                confirmed_bypass_count += 1;
                crate::utils::score_dynamics::record_field_seen(&chunk.property);
                crate::utils::score_dynamics::record_baseline("indexing.format_bypass", 1.0);
                println!(
                    "  🛡️ [FORMAT GATE BYPASS] '{}' (property='{}', format='{}') 형식 불일치이지만 JSON 구조 확정이므로 보호",
                    if chunk.chunk_text.chars().count() > 60 {
                        format!("{}...", chunk.chunk_text.chars().take(57).collect::<String>())
                    } else {
                        chunk.chunk_text.clone()
                    },
                    chunk.property,
                    chunk.property_format
                );
                continue;
            }
            crate::utils::score_dynamics::record_field_seen(&chunk.property);
            crate::utils::score_dynamics::record_field_reject(
                &chunk.property,
                crate::utils::score_dynamics::GateKind::Format,
            );
            println!(
                "  🚧 [FORMAT GATE / INDEXING] '{}' (property='{}', format='{}') 형식 불일치 → unclassified 강등",
                chunk.chunk_text, chunk.property, chunk.property_format
            );
            chunk.property = "unclassified".to_string();
            chunk.property_format = "Text".to_string();
            rejected_count += 1;
        }
    }

    // 🌟 [추가] 최종 요약 로그
    if rejected_count > 0 || confirmed_bypass_count > 0 {
        println!(
            "  🚧 [FORMAT GATE / INDEXING] 강등: {}건 | confirmed 보호: {}건",
            rejected_count, confirmed_bypass_count
        );
    }

    chunks
}

// =====================================================================
// 🌟 [SYNONYM EXPANSION] 음차(Transliteration) 별칭 생성 지원 함수
// ---------------------------------------------------------------------
// 목적:
//   다국어 임베딩은 'Knit' ↔ '니트' 를 의미 공간에서 연결하지만,
//   자유서술 값(상품명)은 bias.json 에 리터럴로 등재할 수 없어 표기 유사도가 0 입니다.
//   (log 실측: 'Cable Knit Cardigan' 저장 벡터 vs '니트 가디건' 질의 = 0.5631,
//    반면 '베이지'는 color 뱅크에 리터럴 등재되어 0.8582)
//   따라서 값 자체를 별칭 벡터로 물질화합니다.
//
// 설계 원칙 준수:
//   - 언어 이름/예시 문자열 하드코딩 없음.
//     1차 목표 표기 체계 = bias.json 에서 뽑은 '그 언어의 실제 문자 샘플'
//     2차 목표 표기 체계 = 원문 값 그 자체
//   - contains() 의미 판정 없음. 표기 체계 판정은 char::is_ascii_alphabetic 카운트뿐.
//   - 새 매직 상수 없음. 길이 상한은 split_natural_language_to_chunks 의 R2(150자) 재사용.
//   - LLM 은 '판정'이 아니라 '문자열 → 문자열 변환기'로만 사용.
// =====================================================================

/// [SYNONYM EXPANSION] ISO 언어 코드를 전체 언어 이름으로 변환합니다.
/// 프롬프트에 언어 코드가 아닌 전체 이름을 전달하여 LLM이 명확히 인식하도록 합니다.
pub fn lang_code_to_full_name(code: &str) -> String {
    match code {
        "ko" => "korean".to_string(),
        "en" => "english".to_string(),
        "ja" => "japanese".to_string(),
        "zh" | "zh-hans" | "zh-tw" | "zh-hk" => "chinese".to_string(),
        "fr" => "french".to_string(),
        "de" => "german".to_string(),
        "es" => "spanish".to_string(),
        "it" => "italian".to_string(),
        "pt" => "portuguese".to_string(),
        "nl" => "dutch".to_string(),
        "ru" => "russian".to_string(),
        "ar" => "arabic".to_string(),
        "th" => "thai".to_string(),
        "hi" => "hindi".to_string(),
        "bn" => "bengali".to_string(),
        "el" => "greek".to_string(),
        "he" => "hebrew".to_string(),
        "vi" => "vietnamese".to_string(),
        "tr" => "turkish".to_string(),
        "pl" => "polish".to_string(),
        "cs" => "czech".to_string(),
        "id" | "ms" => "indonesian".to_string(),
        _ => code.to_string(),
    }
}

/// [SYNONYM EXPANSION] any_ascii 기반 음차 시도.
/// 비라틴 → 라틴 방향일 때만 사용 가능합니다.
/// 성공 시 Some(result), 실패 시 None 반환.
pub fn try_any_ascii_transliteration(value: &str) -> Option<String> {
    let src = value.trim();
    if src.is_empty() { return None; }
    if is_latin_dominant(src) { return None; }
    let result = any_ascii::any_ascii(src);
    let result = result.trim().to_string();
    if result.is_empty() { return None; }
    if result.eq_ignore_ascii_case(src) { return None; }
    if is_latin_dominant(&result) == is_latin_dominant(src) { return None; }
    if result.chars().count() > 150 { return None; }
    Some(result)
}

/// [SYNONYM EXPANSION] 단어 목록에 대해 any_ascii 를 개별 시도합니다.
/// 모든 단어가 any_ascii 로 변환 가능하면 Some(전체 결과),
/// 하나라도 실패하면 None 을 반환하여 LLM 폴백을 유도합니다.
pub fn try_any_ascii_transliteration_words(words: &[String]) -> Option<String> {
    if words.is_empty() { return None; }
    let mut results: Vec<String> = Vec::with_capacity(words.len());
    for w in words {
        let src = w.trim();
        if src.is_empty() { continue; }
        if is_latin_dominant(src) {
            // 이미 라틴이면 그대로 유지
            results.push(src.to_string());
            continue;
        }
        let converted = any_ascii::any_ascii(src);
        let converted = converted.trim().to_string();
        if converted.is_empty() || converted.eq_ignore_ascii_case(src) {
            return None; // 변환 실패 → LLM 폴백
        }
        if !is_latin_dominant(&converted) {
            return None; // 표기 체계 반전 실패 → LLM 폴백
        }
        results.push(converted);
    }
    if results.is_empty() { return None; }
    Some(results.join(" "))
}

/// [SYNONYM EXPANSION] 값의 주 표기 체계가 라틴(ASCII 알파벳)인지 판정합니다.
/// 문자 클래스 카운트만 사용하므로 언어 사전이 전혀 필요 없습니다.
pub fn is_latin_dominant(value: &str) -> bool {
    let mut latin = 0usize;
    let mut other = 0usize;
    for c in value.chars() {
        if !c.is_alphabetic() { continue; }
        if c.is_ascii_alphabetic() { latin += 1; } else { other += 1; }
    }
    latin >= other
}

/// [SYNONYM EXPANSION] 이 청크가 음차 별칭 생성 대상인지 판정합니다.
///
/// 판정 규칙 (전부 결정론, 어휘 하드코딩 없음):
///   T1: property_format 이 '자유 서술 값'을 담는 형식이어야 합니다. (Text / Address)
///       Numeric / Date / Identifier / Link / Phone / TrackingCode / Enum 은
///       값이 숫자·코드·캐노니컬 키라 음차가 물리적으로 무의미합니다.
///       Synthesis 는 합성 문장이라 별칭 벡터가 노이즈만 늘리므로 제외합니다.
///   T2: 값에 '문자'가 하나라도 있어야 소리를 옮길 수 있습니다.
///   T3: 숫자 비율이 절반 이상이면 코드성 값이므로 제외합니다.
///   T4: 길이 상한은 R2 의 150자를 그대로 재사용합니다. (새 상수 도입 아님)
pub fn needs_transliteration(chunk: &ChunkMetadata) -> bool {
    match chunk.property_format.as_str() {
        "Text" | "Address" => {},
        _ => return false,
    }

    let v = chunk.value_part.trim();
    if v.is_empty() { return false; }

    let n = v.chars().count();
    if n < 2 { return false; }
    if n > 150 { return false; }

    if !v.chars().any(|c| c.is_alphabetic()) { return false; }

    let digits = v.chars().filter(|c| c.is_ascii_digit()).count();
    if digits * 2 >= n { return false; }

    true
}

/// [SYNONYM EXPANSION] 문서 언어의 '실제 문자 샘플'을 bias.json 에서 동적으로 확보합니다.
/// detect_document_language() 결과(doc_lang)를 그대로 받아
///   get_localized_page_type()  → 그 언어로 쓰인 도메인 명사
///   indexing_leaf_label()      → 그 언어로 쓰인 이 속성의 라벨
/// 두 조각을 이어붙입니다. 코드에는 어떤 언어 이름도 등장하지 않습니다.
pub fn native_script_sample(doc_lang: &str, page_type: &str, property: &str) -> String {
    let localized_type = crate::parsing::get_localized_page_type(page_type, doc_lang);
    let leaf = crate::utils::ai_utils::indexing_leaf_label(doc_lang, page_type, property);

    let mut s = String::new();
    let lt = localized_type.trim();
    if !lt.is_empty() { s.push_str(lt); }

    let lf = leaf.trim();
    if !lf.is_empty() && lf != lt {
        if !s.is_empty() { s.push(' '); }
        s.push_str(lf);
    }
    s
}

/// [SYNONYM EXPANSION] 로마자 표기 샘플입니다.
/// 스키마 속성명을 자연어화한 결과이므로 항상 ASCII 이며,
/// 특정 언어가 아니라 '문자 체계'만 지시합니다.
pub fn latin_script_sample(property: &str) -> String {
    let h = crate::utils::ai_utils::humanize_url_token(property);
    if h.trim().is_empty() { property.to_string() } else { h }
}

/// [SYNONYM EXPANSION] 1차 음차가 가능한지 판정합니다.
///   - 값이 라틴 우세  → 문서 언어가 비라틴이어야 음차 성립
///   - 값이 비라틴 우세 → 로마자는 항상 라틴이므로 항상 성립
///     (이 경우 any_ascii 로 처리 가능하므로 항상 true)
/// 반환값은 더 이상 샘플 문자열이 아니라 bool 입니다.
/// LLM 호출 여부를 결정하는 데만 사용하며, 프롬프트에는 전달하지 않습니다.
pub fn can_transliterate(
    value: &str,
    doc_lang: &str,
) -> bool {
    let src_latin = is_latin_dominant(value);
    if src_latin {
        // 라틴 원문 → 비라틴 문서 언어에서만 음차 가능
        let sample = native_script_sample(doc_lang, "", "");
        if sample.trim().is_empty() { return false; }
        !is_latin_dominant(&sample)
    } else {
        // 비라틴 원문 → 로마자(라틴)는 항상 성립
        // any_ascii 로 처리 가능 여부 확인 (실패해도 LLM 폴백이 있으므로 true)
        true
    }
}

pub fn strip_special_chars_for_transliteration(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn reglue_native_alias(source: &str, native: &str) -> String {
    let mut glue: Vec<bool> = Vec::new();
    let mut runs = 0usize;
    let mut in_run = false;
    let mut gap_has_space = false;
    for c in source.chars() {
        if c.is_alphanumeric() {
            if !in_run {
                if runs > 0 {
                    glue.push(!gap_has_space);
                }
                runs += 1;
                in_run = true;
            }
        } else {
            if in_run {
                in_run = false;
                gap_has_space = false;
            }
            if c.is_whitespace() {
                gap_has_space = true;
            }
        }
    }
    let parts: Vec<&str> = native.split_whitespace().collect();
    if runs < 2 || parts.len() != runs || !glue.iter().any(|g| *g) {
        return native.to_string();
    }
    let latin = |s: &str| s.chars().any(|c| c.is_ascii_alphabetic());
    let mut out = String::from(parts[0]);
    for i in 1..parts.len() {
        let glued = glue[i - 1] && !latin(parts[i - 1]) && !latin(parts[i]);
        if !glued {
            out.push(' ');
        }
        out.push_str(parts[i]);
    }
    out
}

pub fn split_words_by_script(source: &str) -> (Vec<String>, Vec<String>) {
    let cleaned = strip_special_chars_for_transliteration(source);
    let mut non_latin: Vec<String> = Vec::new();
    let mut latin: Vec<String> = Vec::new();
    for word in cleaned.split_whitespace() {
        if word.is_empty() { continue; }
        if is_latin_dominant(word) {
            latin.push(word.to_string());
        } else {
            non_latin.push(word.to_string());
        }
    }
    (non_latin, latin)
}

pub fn build_transliteration_prompt_for_words(words: &[String], target_language: &str) -> String {
    let full_lang = lang_code_to_full_name(target_language);
    let joined = words.join(" ");
    crate::prompts::transliteration_prompt(&joined, &full_lang)
}

const TRANSLIT_SCHEMA_KEYS: [&str; 2] = ["transcription", "transliteration"];

fn edit_distance(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur: Vec<usize> = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn key_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.trim().to_lowercase().chars().collect();
    let b: Vec<char> = b.trim().to_lowercase().chars().collect();
    edit_distance(&a, &b)
}

fn schema_value<'a>(parsed: &'a serde_json::Value, name: &str) -> Option<&'a serde_json::Value> {
    let map = parsed.as_object()?;
    if let Some(v) = map.get(name) {
        return Some(v);
    }
    let other = TRANSLIT_SCHEMA_KEYS.iter().copied().find(|k| *k != name).unwrap_or("");
    let mut best: Option<(usize, &String, &serde_json::Value)> = None;
    for (k, v) in map.iter() {
        if TRANSLIT_SCHEMA_KEYS.contains(&k.as_str()) || k.trim().eq_ignore_ascii_case("language") {
            continue;
        }
        let d = key_distance(k, name);
        if d > 3 || d >= key_distance(k, other) {
            continue;
        }
        if best.map_or(true, |(bd, _, _)| d < bd) {
            best = Some((d, k, v));
        }
    }
    let (d, k, v) = best?;
    println!(
        "    🩹 [TRANSLIT SCHEMA KEY] 응답 키 '{}' 를 '{}' 로 읽습니다 (철자 거리 {}). 키 철자만 틀리고 값은 정상인 응답을 '키 없음' 으로 버리면 그 값이 영구 음차 불가로 캐시됩니다.",
        k, name, d
    );
    Some(v)
}

fn word_key_value<'a>(map: &'a serde_json::Map<String, serde_json::Value>, word: &str) -> Option<&'a str> {
    let usable = |v: &'a serde_json::Value| v.as_str().filter(|s| !s.trim().is_empty());
    if let Some(s) = map.get(word).and_then(usable) {
        return Some(s);
    }
    let lw = word.trim().to_lowercase();
    if let Some(s) = map.iter().find(|(k, _)| k.trim().to_lowercase() == lw).and_then(|(_, v)| usable(v)) {
        return Some(s);
    }
    let n = word.trim().chars().count();
    let tol = if n >= 8 { 2 } else if n >= 4 { 1 } else { 0 };
    if tol == 0 {
        return None;
    }
    let mut hits: Vec<(usize, &str)> = map
        .iter()
        .filter_map(|(k, v)| usable(v).map(|s| (key_distance(k, word), s)))
        .filter(|(d, _)| *d <= tol)
        .collect();
    hits.sort_by_key(|(d, _)| *d);
    match hits.as_slice() {
        [(_, s)] => Some(*s),
        [(d0, s0), (d1, _), ..] if d0 < d1 => Some(*s0),
        _ => None,
    }
}

fn transcription_fallback(
    parsed: &serde_json::Value,
    joined_src: &str,
    word_map: &dyn Fn(&serde_json::Value) -> String,
) -> String {
    let tc = match schema_value(parsed, "transcription") {
        Some(v) => v,
        None => return String::new(),
    };
    if let Some(s) = tc.as_str() {
        return s.trim().to_string();
    }
    let map = match tc.as_object() {
        Some(m) => m,
        None => return String::new(),
    };
    if let Some(s) = word_key_value(map, joined_src) {
        return s.trim().to_string();
    }
    let mapped = word_map(tc);
    if !mapped.is_empty() && !mapped.eq_ignore_ascii_case(joined_src.trim()) {
        return mapped;
    }
    if map.len() == 1 && joined_src.split_whitespace().count() == 1 {
        if let Some(s) = map.values().next().and_then(|v| v.as_str()) {
            return s.trim().to_string();
        }
    }
    String::new()
}

fn record_schema_adherence(parsed: &serde_json::Value) {
    let both = TRANSLIT_SCHEMA_KEYS.iter().all(|k| parsed.get(*k).is_some());
    crate::utils::score_dynamics::record_baseline("indexing.translit_schema_both_keys", if both { 1.0 } else { 0.0 });
}

pub fn sanitize_transliteration_dual_for_words(raw: &str, words: &[String]) -> (String, String) {
    let parsed = crate::parsing::parse_json_from_llm(raw);
    record_schema_adherence(&parsed);
    let src_words: Vec<&str> = words.iter().map(|w| w.as_str()).collect();
    let extract_word_map = |obj: &serde_json::Value| -> String {
        let map = match obj.as_object() { Some(m) => m, None => return String::new() };
        let mut parts: Vec<String> = Vec::new();
        for w in &src_words {
            let cleaned: String = word_key_value(map, w)
                .map(|v| v.trim().chars().filter(|c| !matches!(c, '-' | '_')).collect())
                .unwrap_or_default();
            if cleaned.is_empty() {
                parts.push(w.to_string());
            } else {
                parts.push(cleaned);
            }
        }
        parts.join(" ")
    };
    let mut transliteration = String::new();
    if let Some(tr_obj) = schema_value(&parsed, "transliteration") {
        transliteration = extract_word_map(tr_obj);
    }
    if transliteration.is_empty() {
        if let Some(val) = schema_value(&parsed, "transliteration").and_then(|v| v.as_str()) {
            transliteration = val.trim().to_string();
        }
    }
    let joined_src = words.join(" ");
    if transliteration.is_empty() || transliteration.eq_ignore_ascii_case(&joined_src) {
        let from_transcription = transcription_fallback(&parsed, &joined_src, &extract_word_map);
        if !from_transcription.is_empty() && !from_transcription.eq_ignore_ascii_case(&joined_src) {
            println!(
                "    ↩️ [TRANSLIT KEY FALLBACK] 응답에 transliteration 이 비어 transcription 값을 씁니다: '{}' → '{}'",
                joined_src, from_transcription
            );
            transliteration = from_transcription;
        }
    }
    transliteration = transliteration.split_whitespace().collect::<Vec<_>>().join(" ");
    let src_joined = words.join(" ");
    if !transliteration.is_empty() && transliteration.eq_ignore_ascii_case(&src_joined) {
        transliteration = String::new();
    }
    let src_non_latin_count: usize = words.iter()
        .filter(|w| !is_latin_dominant(w))
        .count();
    if !transliteration.is_empty() {
        let result_non_latin = transliteration.chars()
            .filter(|c| c.is_alphabetic() && !c.is_ascii_alphabetic())
            .count();
        let script_flipped =
            (src_non_latin_count > 0 && result_non_latin < src_non_latin_count)
            || (src_non_latin_count == 0 && result_non_latin > 0);
        if !script_flipped && is_latin_dominant(&transliteration) == is_latin_dominant(&src_joined) {
            transliteration = String::new();
        } else if transliteration.chars().count() > 150 {
            transliteration = String::new();
        }
    }
    (String::new(), transliteration)
}

/// [LANGUAGE TRACK MERGE] 트랙별 결과를 원본 단어 순서로 병합합니다.
/// non_latin_results: 비라틴 단어들의 변환 결과 (원본 순서 유지)
/// latin_results: 라틴 단어들의 변환 결과 (원본 순서 유지)
/// 원본 소스의 단어 순서를 복원하여 하나의 문자열로 조립합니다.
pub fn merge_track_results(source: &str, non_latin_words: &[String], non_latin_result: &str, latin_words: &[String], latin_result: &str) -> String {
    let cleaned = strip_special_chars_for_transliteration(source);
    let original_words: Vec<&str> = cleaned.split_whitespace().collect();

    let non_latin_parts: Vec<&str> = non_latin_result.split_whitespace().collect();
    let latin_parts: Vec<&str> = latin_result.split_whitespace().collect();

    let mut nl_idx = 0usize;
    let mut la_idx = 0usize;
    let mut merged: Vec<String> = Vec::new();

    for word in &original_words {
        let is_latin = non_latin_words.iter().all(|w| w != word);
        if is_latin {
            if la_idx < latin_parts.len() {
                merged.push(latin_parts[la_idx].to_string());
                la_idx += 1;
            } else {
                merged.push(word.to_string());
            }
        } else {
            if nl_idx < non_latin_parts.len() {
                merged.push(non_latin_parts[nl_idx].to_string());
                nl_idx += 1;
            } else {
                merged.push(word.to_string());
            }
        }
    }

    merged.join(" ")
}

/// [SYNONYM EXPANSION] 음차 프롬프트를 만듭니다.
/// 목표 표기 체계를 '언어 이름'이 아니라 '실제 문자 샘플'로 지시하므로
/// 어떤 언어가 들어와도 동일한 프롬프트 한 벌로 동작합니다.
/// 프롬프트 본문은 prompts.rs 의 transliteration_prompt() 에 위임하며,
/// 출력 형식은 단어별 JSON ({ "transcription": {...}, "transliteration": {...} }) 입니다.
/// target_language 는 ISO 코드가 아닌 전체 언어 이름으로 변환되어 전달됩니다.
/// 특수문자는 공백으로 치환하여 순수 언어 토큰만 LLM 에 전달합니다.
pub fn build_transliteration_prompt(source_value: &str, target_language: &str) -> String {
    let full_lang = lang_code_to_full_name(target_language);
    let cleaned = strip_special_chars_for_transliteration(source_value);
    crate::prompts::transliteration_prompt(&cleaned, &full_lang)
}

/// [SYNONYM EXPANSION] LLM 응답에서 음차 결과를 추출하는 결정론 정화기입니다.
///
/// 응답은 단어별 JSON 형식으로 수신합니다:
/// { "language": "...", "transcription": { "word1": "...", ... }, "transliteration": { "word1": "...", ... } }
///
/// 반환값: (transcription_result, transliteration_result)
///   - 두 값이 다르면 둘 다 반환
///   - 두 값이 같으면 transcription 만 반환, transliteration 은 빈 문자열
///   - 게이트 실패 시 빈 문자열
///
/// 최종 게이트:
///   G1: 결과가 원문과 동일하면 폐기 (모델이 그대로 반복한 경우)
///   G2: 표기 체계가 뒤집히지 않았으면 폐기 (음차가 일어나지 않았다는 구조적 증거)
///   G3: 길이가 R2(150자)를 넘으면 폐기 (음차가 아니라 설명문)
pub fn sanitize_transliteration(raw: &str, source_value: &str) -> String {
    let (_, tr) = sanitize_transliteration_dual(raw, source_value);
    tr
}

/// [SYNONYM EXPANSION] LLM 응답에서 transcription 과 transliteration 을 모두 추출합니다.
///
/// 반환값: (transcription_result, transliteration_result)
///   - transcription 이 유효하고 transliteration 이 transcription 과 다르면 둘 다 반환
///   - 같으면 transcription 만 반환, transliteration 은 빈 문자열
///   - 둘 다 게이트 실패 시 둘 다 빈 문자열
pub fn sanitize_transliteration_dual(raw: &str, source_value: &str) -> (String, String) {
    let parsed = crate::parsing::parse_json_from_llm(raw);
    record_schema_adherence(&parsed);
    let src_clean = strip_special_chars_for_transliteration(source_value);
    let src_clean_ref = src_clean.as_str();
    let src_words: Vec<&str> = src_clean_ref.split_whitespace().collect();
    let extract_word_map = |obj: &serde_json::Value| -> String {
        let map = match obj.as_object() { Some(m) => m, None => return String::new() };
        let mut parts: Vec<String> = Vec::new();
        for w in &src_words {
            if let Some(v) = word_key_value(map, w) {
                let cleaned: String = v.trim().chars().filter(|c| !matches!(c, '-' | '_')).collect();
                if !cleaned.is_empty() {
                    parts.push(cleaned);
                }
            }
        }
        parts.join(" ")
    };
    let mut transliteration = String::new();
    if let Some(tr_obj) = schema_value(&parsed, "transliteration") {
        transliteration = extract_word_map(tr_obj);
    }
    if transliteration.is_empty() {
        if let Some(val) = schema_value(&parsed, "transliteration").and_then(|v| v.as_str()) {
            transliteration = val.trim().to_string();
        }
    }
    if transliteration.is_empty() || transliteration.eq_ignore_ascii_case(src_clean_ref) {
        let from_transcription = transcription_fallback(&parsed, src_clean_ref, &extract_word_map);
        if !from_transcription.is_empty() && !from_transcription.eq_ignore_ascii_case(src_clean_ref) {
            println!(
                "    ↩️ [TRANSLIT KEY FALLBACK] 응답에 transliteration 이 비어 transcription 값을 씁니다: '{}' → '{}'",
                src_clean_ref, from_transcription
            );
            transliteration = from_transcription;
        }
    }
    transliteration = transliteration.split_whitespace().collect::<Vec<_>>().join(" ");
    let src_non_latin = src_clean_ref
        .chars()
        .filter(|c| c.is_alphabetic() && !c.is_ascii_alphabetic())
        .count();
    let mut result_tr = transliteration.clone();
    if result_tr.is_empty() || result_tr.eq_ignore_ascii_case(src_clean_ref) {
        result_tr = String::new();
    } else {
        let result_tr_non_latin = result_tr
            .chars()
            .filter(|c| c.is_alphabetic() && !c.is_ascii_alphabetic())
            .count();
        let script_flipped =
            (src_non_latin > 0 && result_tr_non_latin < src_non_latin)
            || (src_non_latin == 0 && result_tr_non_latin > 0);
        if !script_flipped && is_latin_dominant(&result_tr) == is_latin_dominant(src_clean_ref) {
            result_tr = String::new();
        } else if result_tr.chars().count() > 150 {
            result_tr = String::new();
        }
    }
    (String::new(), result_tr)
}

/// [SYNONYM EXPANSION - MIXED SCRIPT DETECTION]
/// 음차 결과에서 하나의 단어 안에 비라틴 문자(한글 등)와 라틴 문자가
/// 혼용된 단어를 감지합니다.
/// 예: "시IELD" → 한글 '시' + 라틴 'IELD' 혼용 → 감지 대상
///
/// 판정 규칙:
///   M1: 단어 내 알파벳 문자 중 라틴과 비라틴이 '모두' 존재해야 혼용입니다.
///   M2: 순수 라틴("Pokemon") 또는 순수 비라틴("포켓몬")은 혼용이 아닙니다.
///   M3: 숫자/특수문자만 있는 단어는 판정에서 제외합니다.
pub fn find_mixed_script_words(text: &str) -> Vec<String> {
    let mut mixed: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        let has_latin = word.chars().any(|c| c.is_alphabetic() && c.is_ascii_alphabetic());
        let has_non_latin = word.chars().any(|c| c.is_alphabetic() && !c.is_ascii_alphabetic());
        if has_latin && has_non_latin {
            if !mixed.iter().any(|m| m == word) {
                mixed.push(word.to_string());
            }
        }
    }
    mixed
}

/// [SYNONYM EXPANSION - MIXED SCRIPT REPLACEMENT]
/// 혼용 단어를 재음차 결과로 교체합니다.
/// text 내 각 단어를 확인하여 replacements 에 포함된 혼용 단어만 교체합니다.
pub fn replace_mixed_words(text: &str, replacements: &[(String, String)]) -> String {
    let mut result: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        let mut replaced = false;
        for (old, new) in replacements {
            if word == old && !new.is_empty() {
                result.push(new.clone());
                replaced = true;
                break;
            }
        }
        if !replaced {
            result.push(word.to_string());
        }
    }
    result.join(" ")
}

/// [SYNONYM EXPANSION] 1차/2차 결과를 표기 체계 기준으로 슬롯에 배정합니다.
///   transliteration_native : 비라틴(문서 언어) 표기 별칭
///   transliteration_roman  : 라틴(로마자) 표기 별칭
/// 원문과 완전히 같은 후보는 검색 가치가 없으므로 배정하지 않습니다.
pub fn assign_transliterations(source_value: &str, stage1: &str, stage2: &str) -> (String, String) {
    let mut native = String::new();
    let mut roman = String::new();
    let src = source_value.trim();

    for cand in [stage1, stage2] {
        let c = cand.trim();
        if c.is_empty() { continue; }
        if c.eq_ignore_ascii_case(src) { continue; }

        if is_latin_dominant(c) {
            if roman.is_empty() { roman = c.to_string(); }
        } else if native.is_empty() {
            native = c.to_string();
        }
    }

    (native, roman)
}

pub const PHONETIC_PASS: f32 = 0.60;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PhoneClass {
    P,
    T,
    K,
    S,
    G,
    L,
    M,
    N,
}

fn phone_score(a: PhoneClass, b: PhoneClass) -> f32 {
    if a == b {
        return 1.0;
    }
    match (a, b) {
        (PhoneClass::G, PhoneClass::K) | (PhoneClass::K, PhoneClass::G) | (PhoneClass::G, PhoneClass::S) | (PhoneClass::S, PhoneClass::G) => 1.0,
        (PhoneClass::T, PhoneClass::S) | (PhoneClass::S, PhoneClass::T) | (PhoneClass::M, PhoneClass::N) | (PhoneClass::N, PhoneClass::M) => 0.5,
        _ => 0.0,
    }
}

fn phone_skeleton(romanized: &str) -> Vec<PhoneClass> {
    let mut out: Vec<PhoneClass> = Vec::new();
    for word in romanized.split_whitespace() {
        let c: Vec<char> = word.chars().collect();
        let mut last: Option<PhoneClass> = None;
        let mut i = 0usize;
        while i < c.len() {
            let x = c[i];
            let nx = c.get(i + 1).copied();
            let mut emitted: Vec<PhoneClass> = Vec::new();
            let mut step = 1usize;
            match (x, nx) {
                ('p', Some('h')) => { emitted.push(PhoneClass::P); step = 2; }
                ('t', Some('h')) => { emitted.push(PhoneClass::T); step = 2; }
                ('s', Some('h')) | ('c', Some('h')) => { emitted.push(PhoneClass::S); step = 2; }
                ('t', Some('s')) | ('t', Some('z')) | ('d', Some('z')) | ('d', Some('j')) => { emitted.push(PhoneClass::S); step = 2; }
                ('c', Some('k')) | ('q', Some('u')) | ('g', Some('h')) => { emitted.push(PhoneClass::K); step = 2; }
                ('n', Some('g')) => { emitted.push(PhoneClass::N); step = 2; }
                ('x', _) => { emitted.push(PhoneClass::K); emitted.push(PhoneClass::S); }
                ('c', Some(n)) if n == 'e' || n == 'i' || n == 'y' => emitted.push(PhoneClass::S),
                ('g', Some(n)) if n == 'e' || n == 'i' || n == 'y' => emitted.push(PhoneClass::G),
                ('r', n) if !n.map_or(false, |v| "aeiouy".contains(v)) => {}
                ('c', _) | ('k', _) | ('g', _) | ('q', _) => emitted.push(PhoneClass::K),
                ('b', _) | ('p', _) | ('f', _) | ('v', _) => emitted.push(PhoneClass::P),
                ('t', _) | ('d', _) => emitted.push(PhoneClass::T),
                ('s', _) | ('z', _) | ('j', _) => emitted.push(PhoneClass::S),
                ('l', _) | ('r', _) => emitted.push(PhoneClass::L),
                ('m', _) => emitted.push(PhoneClass::M),
                ('n', _) => emitted.push(PhoneClass::N),
                _ => {}
            }
            for p in emitted {
                if last != Some(p) {
                    out.push(p);
                }
                last = Some(p);
            }
            i += step;
        }
    }
    out
}

fn romanize_for_phonetics(text: &str) -> String {
    any_ascii::any_ascii(text)
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_lowercase() { c } else { ' ' })
        .collect()
}

const LETTER_NAMES: [&str; 26] = [
    "ei", "bi", "si", "di", "i", "ef", "ji", "eichi", "ai", "jei", "kei", "el", "em",
    "en", "ou", "pi", "kyu", "ar", "es", "ti", "yu", "bi", "deobeulyu", "eks", "wai", "ji",
];

fn letter_name_reading(src: &str) -> Option<String> {
    let mut changed = false;
    let words: Vec<String> = src
        .split_whitespace()
        .map(|w| {
            let letters: Vec<char> = w.chars().filter(|c| c.is_ascii_alphabetic()).collect();
            let acronym = (1..=5).contains(&letters.len())
                && letters.iter().all(|c| c.is_ascii_uppercase())
                && w.chars().all(|c| c.is_ascii_alphanumeric());
            if !acronym {
                return w.to_string();
            }
            changed = true;
            letters
                .iter()
                .map(|c| LETTER_NAMES[(*c as u8 - b'A') as usize])
                .collect::<Vec<_>>()
                .join("")
        })
        .collect();
    if changed { Some(words.join(" ")) } else { None }
}

pub const SPELLED_OUT_MIN: f32 = 0.90;

pub const TRANSLIT_GATE_REV: &str = "g2";

fn letter_reading_key(romanized: &str) -> String {
    let compact: String = romanized.split_whitespace().collect::<String>().replace("eu", "").replace("ou", "o");
    let mut out = String::new();
    for c in compact.chars() {
        let c = match c {
            'f' => 'p',
            'r' => 'l',
            'g' => 'k',
            other => other,
        };
        if out.ends_with(c) && !"aeiou".contains(c) {
            continue;
        }
        out.push(c);
    }
    out
}

fn char_similarity(a: &str, b: &str) -> f32 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let n = a.len().max(b.len());
    if n == 0 {
        return 0.0;
    }
    1.0 - edit_distance(&a, &b) as f32 / n as f32
}

pub fn spelled_out_similarity(latin_src: &str, native: &str) -> Option<f32> {
    let src_words: Vec<&str> = latin_src.split_whitespace().collect();
    let out_words: Vec<&str> = native.split_whitespace().collect();
    if src_words.is_empty() || out_words.is_empty() {
        return None;
    }
    let pairs: Vec<(&str, String)> = if src_words.len() == out_words.len() {
        src_words.iter().zip(out_words.iter()).map(|(s, o)| (*s, o.to_string())).collect()
    } else if src_words.len() == 1 {
        vec![(src_words[0], out_words.concat())]
    } else {
        return None;
    };
    let mut best: Option<f32> = None;
    for (word, heard_raw) in pairs {
        let letters: Vec<char> = word.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        let acronym = letters.len() <= 5 && letters.iter().all(|c| c.is_ascii_uppercase());
        if letters.len() < 3 || acronym || !letters.iter().any(|c| "aeiouyAEIOUY".contains(*c)) {
            continue;
        }
        let heard = letter_reading_key(&romanize_for_phonetics(&heard_raw));
        if heard.is_empty() {
            continue;
        }
        let mut spelled = String::new();
        for (k, c) in letters.iter().enumerate() {
            spelled.push_str(LETTER_NAMES[(c.to_ascii_uppercase() as u8 - b'A') as usize]);
            if k + 1 < 3 {
                continue;
            }
            let sim = char_similarity(&heard, &letter_reading_key(&spelled));
            if best.map_or(true, |b| sim > b) {
                best = Some(sim);
            }
        }
    }
    best
}

pub fn phonetic_similarity(a: &str, b: &str) -> Option<f32> {
    let sa = phone_skeleton(&romanize_for_phonetics(a));
    let sb = phone_skeleton(&romanize_for_phonetics(b));
    if sa.len().max(sb.len()) < 2 {
        return None;
    }
    let (n, m) = (sa.len(), sb.len());
    let mut dp = vec![vec![0.0f32; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            let diag = dp[i - 1][j - 1] + phone_score(sa[i - 1], sb[j - 1]);
            dp[i][j] = diag.max(dp[i - 1][j]).max(dp[i][j - 1]);
        }
    }
    Some(2.0 * dp[n][m] / (n + m) as f32)
}

pub fn phonetic_gate(latin_src: &str, native: &str) -> (bool, Option<f32>) {
    if native.trim().is_empty() || latin_src.trim().is_empty() {
        return (true, None);
    }
    let mut best = phonetic_similarity(latin_src, native);
    if let Some(alt) = letter_name_reading(latin_src) {
        if let Some(s) = phonetic_similarity(&alt, native) {
            best = Some(best.map_or(s, |b| b.max(s)));
        }
    }
    match best {
        Some(s) => (s >= PHONETIC_PASS, Some(s)),
        None => (true, None),
    }
}

pub fn gate_native_alias(latin_src: &str, native: String, tag: &str) -> String {
    if native.trim().is_empty() || is_latin_dominant(&native) {
        return native;
    }
    if let Some(sp) = spelled_out_similarity(latin_src, &native) {
        let spelled = sp >= SPELLED_OUT_MIN;
        crate::utils::score_dynamics::record_baseline("indexing.translit_spelled_out", if spelled { 1.0 } else { 0.0 });
        if spelled {
            crate::utils::score_dynamics::record_baseline("indexing.translit_phonetic_reject", 1.0);
            println!(
                "    🚫 [{} SPELLED-OUT REJECT] '{}' → '{}' | 원문 글자를 알파벳 이름으로 한 자씩 읽은 결과와 {:.2} 일치합니다 (기준 {:.2}). 발음할 수 있는 단어를 글자 이름으로 읽은 것은 음차가 아니라서 별칭으로 쓰지 않습니다. 대문자 5자 이하 약어(PC · BMW)는 글자 읽기가 정상 표기라 이 검사에서 뺍니다.",
                tag,
                latin_src,
                native,
                sp,
                SPELLED_OUT_MIN
            );
            return String::new();
        }
    }
    let (ok, sim) = phonetic_gate(latin_src, &native);
    if let Some(s) = sim {
        crate::utils::score_dynamics::record_baseline("indexing.translit_phonetic_sim", s);
        crate::utils::score_dynamics::record_baseline(
            "indexing.translit_phonetic_reject",
            if ok { 0.0 } else { 1.0 },
        );
    }
    if ok {
        return native;
    }
    println!(
        "    🚫 [{} PHONETIC REJECT] '{}' → '{}' | 자음 골격 유사도 {:.2} < {:.2} — 소리가 원문과 맞지 않는 음차라 별칭으로 쓰지 않습니다.",
        tag,
        latin_src,
        native,
        sim.unwrap_or(0.0),
        PHONETIC_PASS
    );
    String::new()
}

pub fn cached_translit_recheck(
    src: &str,
    cached_native: &str,
    doc_lang: &str,
    lang_engine: bool,
) -> Option<String> {
    let (_non_latin, latin) = split_words_by_script(src);
    if latin.is_empty() {
        return None;
    }
    if is_latin_dominant(&native_script_sample(doc_lang, "", "")) {
        return None;
    }
    let src_words: std::collections::HashSet<String> = strip_special_chars_for_transliteration(src)
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();
    let added: Vec<&str> = cached_native
        .split_whitespace()
        .filter(|w| !src_words.contains(&w.to_lowercase()))
        .collect();
    if added.is_empty() {
        return if lang_engine {
            Some("문서언어 별칭이 비어 있어 언어별 4B 로 한 번 더 시도합니다".to_string())
        } else {
            None
        };
    }
    if let Some(sp) = spelled_out_similarity(&latin.join(" "), &added.join(" ")).filter(|s| *s >= SPELLED_OUT_MIN) {
        return Some(format!(
            "캐시 별칭 '{}' 는 원문 글자를 알파벳 이름으로 한 자씩 읽은 것입니다 (일치 {:.2} ≥ {:.2})",
            added.join(" "),
            sp,
            SPELLED_OUT_MIN
        ));
    }
    let (ok, sim) = phonetic_gate(&latin.join(" "), &added.join(" "));
    if ok {
        None
    } else {
        Some(format!(
            "캐시 별칭 '{}' 의 자음 골격 유사도 {:.2} 가 {:.2} 미만입니다",
            added.join(" "),
            sim.unwrap_or(0.0),
            PHONETIC_PASS
        ))
    }
}

fn script_class(c: char) -> u8 {
    match c as u32 {
        0x0041..=0x005A | 0x0061..=0x007A | 0x00C0..=0x024F | 0x1E00..=0x1EFF => 1,
        0x1100..=0x11FF | 0x3130..=0x318F | 0xAC00..=0xD7AF => 2,
        0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => 3,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF => 4,
        0x0400..=0x04FF => 5,
        0x0E00..=0x0E7F => 6,
        0x0600..=0x06FF => 7,
        _ if c.is_alphabetic() => 8,
        _ => 0,
    }
}

fn script_set(text: &str) -> std::collections::BTreeSet<u8> {
    text.chars().map(script_class).filter(|s| *s != 0).collect()
}

fn display_exonym(name: &str) -> String {
    if name.chars().filter(|c| c.is_ascii_alphabetic()).count() <= 3 {
        return name.to_uppercase();
    }
    name.split_whitespace()
        .map(|w| {
            let mut cs = w.chars();
            match cs.next() {
                Some(f) => f.to_uppercase().collect::<String>() + cs.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn canonical_entity_alias(src: &str, doc_lang: &str) -> Option<(String, String, &'static str)> {
    let code = crate::utils::ai_utils::country_code_exact(src)?;
    let sample = native_script_sample(doc_lang, "", "");
    if sample.trim().is_empty() || is_latin_dominant(&sample) {
        return None;
    }
    let mut target = script_set(&sample);
    if target.contains(&3) {
        target.insert(4);
    }
    let (_, raw) = crate::utils::ai_utils::COUNTRY_NAMES_ML.iter().find(|(c, _)| *c == code)?;
    let key_of = |t: &str| -> String { t.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase() };
    let src_key = key_of(src);
    let mut native: Vec<&str> = Vec::new();
    let mut english: Option<&str> = None;
    for name in raw.split(',').map(|n| n.trim()).filter(|n| !n.is_empty()) {
        let set = script_set(name);
        if english.is_none() && name.is_ascii() && set.len() == 1 && set.contains(&1) {
            english = Some(name);
        }
        if native.len() < 2 && !set.is_empty() && set.is_subset(&target) && key_of(name) != src_key && !native.contains(&name) {
            native.push(name);
        }
    }
    let roman = if is_latin_dominant(src) { String::new() } else { english.map(display_exonym).unwrap_or_default() };
    let native = native.join(" ");
    if native.is_empty() && roman.is_empty() {
        return None;
    }
    Some((native, roman, code))
}

/// [PHASE B - 로그 헬퍼] 메타데이터 부여 + NMS + FORMAT GATE 전체 결과를 출력합니다.
/// confirmed(✓) / 비confirmed(?) 카운트 요약을 포함합니다.
pub fn log_enriched_chunks(chunks: &[ChunkMetadata]) {
    let confirmed_count = chunks.iter().filter(|c| c.confirmed).count();
    let unconfirmed_count = chunks.len() - confirmed_count;
    println!(
        "  📋 [PHASE B] 메타데이터 부여 완료: {}개 청크 (✓confirmed: {} | ?unconfirmed: {})",
        chunks.len(), confirmed_count, unconfirmed_count
    );
    for (i, c) in chunks.iter().enumerate() {
        let flag = if c.confirmed { "✓" } else { "?" };
        println!(
            "    [{}] {} property='{}' | format='{}' | bias={}구 | prej={}구 | value='{}' | text='{}'",
            i,
            flag,
            c.property,
            c.property_format,
            c.bias_phrases.len(),
            c.prejudice_phrases.len(),
            c.value_part,
            c.chunk_text
        );
    }
}

// =====================================================================
// 🌟 [PHASE C] 인덱싱 전용 PLINKO GAME — Sliding Window Cliff Detection
// =====================================================================
// 검색 경로(model.rs)의 PLINKO GAME 과 동일한 구조로 동작하되,
// 입력이 "사용자 질의" 대신 "json_to_natural_language() 출력 청크"입니다.
//
// 차이점:
//   - 검색 경로: 질의 단어를 하나씩 확장하며 필드 뱅크와 코사인 비교
//   - 인덱싱 경로: 청크 텍스트를 단어 단위로 확장하며 필드 뱅크와 코사인 비교
//   - FORMAT GATE 를 배정 '전' 에 적용 (설계 원칙 준수)
//   - 임베딩은 외부 클로저(embed_fn)로 주입받아 nl_convert.rs 의 독립성 유지
// =====================================================================

/// [PHASE C] PLINKO GAME 확정 결과 구조체입니다.
///
/// 필드 설명:
///   - chunk_text:    확정된 청크 텍스트 (Sliding Window 로 잘린 부분)
///   - property:      확정된 속성명 (snake_case)
///   - score:         확정 시점의 코사인 점수
///   - alternatives:  차순위 후보 (속성명, 점수) 최대 5개
///   - all_scores:    전 필드 코사인 점수 (배타 배정 행렬 구축용)
#[derive(Debug, Clone)]
pub struct PlinkoResult {
    pub chunk_text: String,
    pub property: String,
    pub score: f32,
    pub alternatives: Vec<(String, f32)>,
    pub all_scores: Vec<(String, f32)>,
}

/// [PHASE C] 인덱싱 전용 PLINKO GAME — Sliding Window Cliff Detection.
///
/// 검색 경로(model.rs)의 PLINKO GAME 과 **동일 알고리즘**으로 동작합니다.
/// 각 청크의 단어를 하나씩 확장하면서 필드 뱅크와 Max-Pool 코사인을 계산하고,
/// 점수가 하락(Cliff)하면 이전 청크를 해당 속성으로 확정합니다.
///
/// FORMAT GATE 는 배정 '전' 에 적용합니다:
///   - Numeric 필드: 숫자 포함 필수
///   - Date 필드: 숫자 포함 필수
///   - Phone 필드: 숫자 7자리 이상 필수
///   - Link 필드: '/' 또는 'http' 포함 필수
///   - 형식 불일치 시 해당 필드를 후보에서 제외
///
/// # 인자
///   - chunks:              nms_battle_for_indexing() 통과 청크 배열
///   - field_names:         필드명 배열 (예: ["id", "title", "sale_price", ...])
///   - field_phrase_embs:   필드별 bias 구 임베딩 뱅크
///   - field_phrase_weights: 필드별 구 가중치
///   - field_formats:       필드별 형식 문자열 ("Numeric", "Text", "Enum" 등)
///   - embed_fn:            텍스트 → 임베딩 벡터 변환 클로저 (비동기)
///
/// # 반환
///   Vec<PlinkoResult> — 각 청크의 확정 속성 + 점수 + 대안
/// [PHASE C] 인덱싱 전용 PLINKO GAME — 확인 모드 + Sliding Window Cliff Detection.
///
/// 검색 경로(model.rs)의 PLINKO GAME 과 **동일 알고리즘**으로 동작하되,
/// 전처리 경로에서는 Phase A 패턴 매칭으로 이미 확정된 청크(`confirmed = true`)에 대해
/// 슬라이딩 윈도우를 건너뛰고 기존 property 를 그대로 확정합니다.
///
/// 확인 모드 로직:
///   - confirmed = true  → PLINKO 슬라이딩 윈도우 건너뛰기.
///     기존 property 를 확정하고, 참고 점수(코사인)만 계산하여 로그에 남깁니다.
///     LLM 호출 없이, 매직 상수 없이, 구조적 사실(패턴 매칭 확정)만으로 결정합니다.
///   - confirmed = false → 기존 PLINKO 슬라이딩 윈도우 정상 수행.
///
/// FORMAT GATE 는 배정 '전' 에 적용합니다:
///   - Numeric 필드: 숫자 포함 필수
///   - Date 필드: 숫자 포함 필수
///   - Phone 필드: 숫자 7자리 이상 필수
///   - Link 필드: '/' 또는 'http' 포함 필수
///   - 형식 불일치 시 해당 필드를 후보에서 제외
///
/// # 인자
///   - chunks:              nms_battle_for_indexing() 통과 청크 배열
///   - field_names:         필드명 배열 (예: ["id", "title", "sale_price", ...])
///   - field_phrase_embs:   필드별 bias 구 임베딩 뱅크
///   - field_phrase_weights: 필드별 구 가중치
///   - field_formats:       필드별 형식 문자열 ("Numeric", "Text", "Enum" 등)
///   - embed_fn:            텍스트 → 임베딩 벡터 변환 클로저 (비동기)
///
/// # 반환
///   Vec<PlinkoResult> — 각 청크의 확정 속성 + 점수 + 대안
pub async fn plinko_game_for_indexing<F, Fut>(
    chunks: &[ChunkMetadata],
    field_names: &[String],
    field_phrase_embs: &[Vec<Vec<f32>>],
    field_phrase_weights: &[Vec<f32>],
    field_formats: &[String],
    embed_fn: F,
) -> Vec<PlinkoResult>
where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = Vec<f32>>,
{
    let mut results: Vec<PlinkoResult> = Vec::with_capacity(chunks.len());
    let mut confirm_count = 0usize;
    let mut discover_count = 0usize;

    for chunk_meta in chunks {
        let words: Vec<&str> = chunk_meta.chunk_text.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }

        // ── 확인 모드: Phase A 패턴 매칭으로 확정된 청크 ──
        if chunk_meta.confirmed {
            let prop_idx = field_names.iter().position(|f| f == &chunk_meta.property);

            // ── 임베딩 계산 (1회) ──
            let chunk_emb = embed_fn(chunk_meta.chunk_text.clone()).await;
            if chunk_emb.iter().all(|&v| v == 0.0) {
                crate::utils::score_dynamics::record_field_seen(&chunk_meta.property);
                crate::utils::score_dynamics::record_field_assigned(&chunk_meta.property, 0.0);
                crate::utils::score_dynamics::record_baseline("indexing.embed_fail", 1.0);
                results.push(PlinkoResult {
                    chunk_text: chunk_meta.chunk_text.clone(),
                    property: chunk_meta.property.clone(),
                    score: 1.0f32,
                    alternatives: Vec::new(),
                    all_scores: Vec::new(),
                });
                confirm_count += 1;
                continue;
            }

            // ── 전 필드 코사인 계산 (역방향 검증 핵심) ──
            let mut all_field_scores: Vec<(String, f32)> = Vec::with_capacity(field_names.len());
            for fi in 0..field_names.len() {
                if field_phrase_embs[fi].is_empty() { continue; }
                let s = crate::utils::ai_utils::weighted_max_pool_sim(
                    &chunk_emb,
                    &field_phrase_embs[fi],
                    &field_phrase_weights[fi],
                );
                all_field_scores.push((field_names[fi].clone(), s));
            }

            // ── argmax 판정 ──
            let origin_score = prop_idx
                .and_then(|pi| all_field_scores.iter().find(|(n, _)| n == &field_names[pi]))
                .map(|(_, s)| *s)
                .unwrap_or(0.0f32);

            let (argmax_prop, argmax_score) = all_field_scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(n, s)| (n.clone(), *s))
                .unwrap_or((String::new(), 0.0f32));

            // ── 차순위 후보 (상위 5개, origin 제외) ──
            let mut sorted_alts = all_field_scores.clone();
            sorted_alts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let alternatives: Vec<(String, f32)> = sorted_alts
                .iter()
                .filter(|(n, _)| n != &chunk_meta.property)
                .take(5)
                .cloned()
                .collect();

            // ── 역방향 검증 판정 ──
            let is_argmax = chunk_meta.property == argmax_prop;
            crate::utils::score_dynamics::record_field_seen(&chunk_meta.property);
            {
                let dist: Vec<f32> = all_field_scores.iter().map(|(_, s)| *s).collect();
                if dist.len() >= 2 {
                    crate::utils::score_dynamics::record_decay("indexing.confirm_row", &dist);
                }
            }
            if !is_argmax && origin_score < argmax_score {
                crate::utils::score_dynamics::record_near_miss(&chunk_meta.property);
                crate::utils::score_dynamics::record_confusion(
                    &chunk_meta.property,
                    &argmax_prop,
                    argmax_score - origin_score,
                );
                crate::utils::score_dynamics::record_baseline(
                    "indexing.confirm_flag_gap",
                    argmax_score - origin_score,
                );
                println!(
                    "  🚩 [CONFIRM FLAG] '{}' | origin='{}'({:.4}) < argmax='{}'({:.4}) | margin={:+.4}",
                    chunk_meta.chunk_text,
                    chunk_meta.property,
                    origin_score,
                    argmax_prop,
                    argmax_score,
                    origin_score - argmax_score
                );
            } else {
                let runner = alternatives.first().map(|(_, s)| *s).unwrap_or(origin_score);
                crate::utils::score_dynamics::record_field_assigned(
                    &chunk_meta.property,
                    origin_score - runner,
                );
            }
            results.push(PlinkoResult {
                chunk_text: chunk_meta.chunk_text.clone(),
                property: chunk_meta.property.clone(), // JSON 구조 우선: 변경하지 않음
                score: origin_score,
                alternatives,
                all_scores: Vec::new(), // 확인 모드에서는 배타 배정 불필요
            });
            confirm_count += 1;
            continue;
        }
        // ── 발견 모드: 미확정 청크에 대한 기존 PLINKO 슬라이딩 윈도우 ──
        discover_count += 1;
        // ── Sliding Window 상태 ──
        let mut current_window: Vec<&str> = Vec::new();
        let mut prev_max_score: f32 = -1.0;
        let mut best_prop: String = String::new();
        let mut prev_alternatives: Vec<(String, f32)> = Vec::new();
        let mut prev_all_scores: Vec<(String, f32)> = Vec::new();

        for word in &words {
            current_window.push(word);
            let test_text = current_window.join(" ");

            // 임베딩 계산 (외부 클로저)
            let test_emb = embed_fn(test_text.clone()).await;
            if test_emb.iter().all(|&v| v == 0.0) {
                continue;
            }

            // ── FORMAT GATE (배정 전) + Max-Pool 코사인 ──
            let mut candidates: Vec<(String, f32)> = Vec::with_capacity(field_names.len());
            for fi in 0..field_names.len() {
                let fmt = field_formats.get(fi).map(|s| s.as_str()).unwrap_or("Text");

                // FORMAT GATE: 값 형식 검증 (배정 전)
                let value_part = &chunk_meta.value_part;
                let passes = match fmt {
                    "Numeric" => value_part.chars().any(|c| c.is_ascii_digit()),
                    "Date" => value_part.chars().any(|c| c.is_ascii_digit()),
                    "Phone" => value_part.chars().filter(|c| c.is_ascii_digit()).count() >= 7,
                    "TrackingCode" => {
                        value_part.split(|c: char| !c.is_alphanumeric())
                            .any(|tok| tok.chars().count() >= 8 && tok.chars().any(|c| c.is_ascii_digit()))
                    },
                    "Identifier" => {
                        value_part.split(|c: char| !c.is_alphanumeric())
                            .any(|tok| tok.chars().count() >= 4 && tok.chars().any(|c| c.is_ascii_digit()))
                    },
                    "Link" => value_part.contains('/') || value_part.to_lowercase().starts_with("http"),
                    "Enum" => true,
                    "Text" => value_part.chars().any(|c| c.is_alphabetic()),
                    "Address" => value_part.chars().any(|c| c.is_alphabetic()) && value_part.split_whitespace().count() >= 2,
                    "Synthesis" => true,
                    _ => true,
                };

                if !passes {
                    continue;
                }

                // Max-Pool 코사인
                let own = crate::utils::ai_utils::weighted_max_pool_sim(
                    &test_emb,
                    &field_phrase_embs[fi],
                    &field_phrase_weights[fi],
                );
                candidates.push((field_names[fi].clone(), own));
            }

            if candidates.is_empty() {
                continue;
            }

            // 점수 내림차순 정렬
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let current_max = candidates[0].1;
            let current_best = candidates[0].0.clone();
            let current_alternatives: Vec<(String, f32)> = candidates.iter().skip(1).take(5).cloned().collect();

            // ── Cliff Detection: 점수 하락 시 이전 청크 확정 ──
            if current_max < prev_max_score && current_window.len() > 1 {
                // 이전 청크를 best_prop 으로 확정
                let confirmed_text = current_window[..current_window.len() - 1].join(" ");
                if !confirmed_text.trim().is_empty() && prev_max_score > 0.0 && !best_prop.is_empty() {
                    results.push(PlinkoResult {
                        chunk_text: confirmed_text,
                        property: best_prop.clone(),
                        score: prev_max_score,
                        alternatives: prev_alternatives.clone(),
                        all_scores: prev_all_scores.clone(),
                    });
                }

                // 새 창 시작: 현재 단어로 리셋
                current_window = vec![word];
                let reset_text = word.to_string();
                let reset_emb = embed_fn(reset_text.clone()).await;
                if !reset_emb.iter().all(|&v| v == 0.0) {
                    let mut reset_candidates: Vec<(String, f32)> = Vec::new();
                    for fi in 0..field_names.len() {
                        let fmt = field_formats.get(fi).map(|s| s.as_str()).unwrap_or("Text");
                        let value_part = &chunk_meta.value_part;
                        let passes = match fmt {
                            "Numeric" => value_part.chars().any(|c| c.is_ascii_digit()),
                            "Date" => value_part.chars().any(|c| c.is_ascii_digit()),
                            "Phone" => value_part.chars().filter(|c| c.is_ascii_digit()).count() >= 7,
                            "Link" => value_part.contains('/') || value_part.to_lowercase().starts_with("http"),
                            "Enum" => true,
                            "Text" => value_part.chars().any(|c| c.is_alphabetic()),
                            _ => true,
                        };
                        if !passes { continue; }
                        let own = crate::utils::ai_utils::weighted_max_pool_sim(
                            &reset_emb,
                            &field_phrase_embs[fi],
                            &field_phrase_weights[fi],
                        );
                        reset_candidates.push((field_names[fi].clone(), own));
                    }
                    reset_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    if !reset_candidates.is_empty() {
                        prev_max_score = reset_candidates[0].1;
                        best_prop = reset_candidates[0].0.clone();
                        prev_alternatives = reset_candidates.iter().skip(1).take(5).cloned().collect();
                        prev_all_scores = reset_candidates;
                    }
                }
            } else {
                prev_max_score = current_max;
                best_prop = current_best;
                prev_alternatives = current_alternatives;
                prev_all_scores = candidates;
            }
        }

        // ── 잔여 청크 처리 (Sweep) ──
        if !current_window.is_empty() && prev_max_score > 0.0 && !best_prop.is_empty() {
            let remaining_text = current_window.join(" ");
            if !remaining_text.trim().is_empty() {
                results.push(PlinkoResult {
                    chunk_text: remaining_text,
                    property: best_prop.clone(),
                    score: prev_max_score,
                    alternatives: prev_alternatives.clone(),
                    all_scores: prev_all_scores.clone(),
                });
            }
        }
    }

    // ── 확인/발견 모드 통계 로그 ──
    if confirm_count > 0 || discover_count > 0 {
        println!(
            "  🎯 [PLINKO MODE] 확인 모드: {}개 청크 (Phase A 확정 유지) | 발견 모드: {}개 청크 (슬라이딩 윈도우 수행)",
            confirm_count, discover_count
        );
    }

    results
}

// =====================================================================
// 🌟 [PHASE C-2] 인덱싱 전용 EXCLUSIVE ASSIGN — 배타 배정
// =====================================================================
// 검색 경로의 exclusive_assign_by_score() 와 동일한 그리디 배타 배정입니다.
// 한 청크는 하나의 속성만, 한 속성은 하나의 청크만 가질 수 있습니다.
//
// PLINKO 결과의 all_scores 로 행렬을 구축하고,
// double_center_matrix → exclusive_assign_by_score 순서로 처리합니다.
// =====================================================================

/// [PHASE C-2] PLINKO 결과를 받아 배타 배정을 수행합니다.
///
/// # 인자
///   - plinko_results: plinko_game_for_indexing() 반환값
///   - field_names:    필드명 배열 (PLINKO 에 전달한 것과 동일)
///
/// # 반환
///   Vec<PlinkoResult> — 배타 배정 후 최종 확정된 결과
///     (한 속성에 여러 청크가 배정된 경우, 점수가 높은 청크만 생존)
/// [PHASE C-2] PLINKO 결과를 받아 배타 배정을 수행합니다.
///
/// 배정 순서:
///   1단계 (confirmed 우선): all_scores 가 비어 있는 청크(= 확인 모드 통과)는
///     기존 property 를 그대로 확정합니다. 동일 property 에 이미 confirmed 가
///     배정되어 있으면 skip 하고 unclassified 로 밀려납니다.
///     이 청크들은 코사인 경쟁에 참여하지 않습니다.
///   2단계 (그리디 경쟁): 나머지 청크는 all_scores 로 행렬을 구축하고
///     double_center_matrix → exclusive_assign_by_score 로 배정합니다.
///
/// # 인자
///   - plinko_results: plinko_game_for_indexing() 반환값
///   - field_names:    필드명 배열 (PLINKO 에 전달한 것과 동일)
///
/// # 반환
///   Vec<PlinkoResult> — 배타 배정 후 최종 확정된 결과
pub fn exclusive_assign_for_indexing(
    plinko_results: Vec<PlinkoResult>,
    field_names: &[String],
) -> Vec<PlinkoResult> {
    if plinko_results.is_empty() {
        return plinko_results;
    }

    let chunk_count = plinko_results.len();
    let field_count = field_names.len();

    let mut final_results: Vec<PlinkoResult> = Vec::new();
    let mut assigned_chunks: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut claimed_fields: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // ── 1단계: confirmed 청크 우선 배정 ──
    // 🌟 [MULTI-VALUE PRESERVE] 검색 경로의 배타 배정은 "한 질의 조각 = 한 컬럼" 이라는
    //    '판정' 이지만, 인덱싱은 '저장' 입니다.
    //    goods 배열 안 여러 상품의 title, tags 다중 값, additional_image_url 처럼
    //    같은 property 가 여러 값을 갖는 것이 정상입니다.
    //    기존에는 2번째부터 unclassified 로 강등되어 PHASE D 필터에서 통째로 사라졌습니다.
    //    NMS 의 D1(동일 텍스트) / D2(동일 property+value) 가 이미 진짜 중복을 제거했으므로,
    //    여기서는 confirmed 청크를 절대 폐기하지 않고 그대로 보존합니다.
    let mut multi_value_kept = 0usize;
    for (ci, pr) in plinko_results.iter().enumerate() {
        if !pr.all_scores.is_empty() {
            continue; // 비confirmed 청크는 2단계에서 처리
        }

        let fi_opt = field_names.iter().position(|f| f == &pr.property);
        let fi = match fi_opt {
            Some(v) => v,
            None => {
                assigned_chunks.insert(ci);
                final_results.push(pr.clone());
                continue;
            }
        };

        if claimed_fields.contains(&fi) {
            multi_value_kept += 1;
            assigned_chunks.insert(ci);
            final_results.push(pr.clone());
            continue;
        }

        claimed_fields.insert(fi);
        assigned_chunks.insert(ci);
        final_results.push(pr.clone());
    }

    if multi_value_kept > 0 {
        println!(
            "  🧬 [MULTI-VALUE PRESERVE] 동일 property 의 추가 확정 청크 {}개를 강등 없이 보존했습니다.",
            multi_value_kept
        );
    }

    // ── 2단계: 비confirmed 청크 그리디 경쟁 ──
    let remaining_indices: Vec<usize> = (0..chunk_count)
        .filter(|ci| !assigned_chunks.contains(ci))
        .collect();

    if !remaining_indices.is_empty() {
        let remaining_count = remaining_indices.len();
        let mut matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; remaining_count]; field_count];

        for (ri, &ci) in remaining_indices.iter().enumerate() {
            let pr = &plinko_results[ci];
            for (fname, score) in &pr.all_scores {
                if let Some(fi) = field_names.iter().position(|f| f == fname) {
                    // 🌟 이미 confirmed 가 선점한 필드는 후보에서 제외
                    if claimed_fields.contains(&fi) {
                        continue;
                    }
                    matrix[fi][ri] = *score;
                }
            }
        }

        let centered = crate::utils::ai_utils::double_center_matrix(&matrix);
        let assign = crate::utils::ai_utils::exclusive_assign_by_score(&centered, 0.0, 0.0);

        for (fi, a) in assign.iter().enumerate() {
            if let Some((ri, own, _margin)) = a {
                let ci = remaining_indices[*ri];
                if assigned_chunks.contains(&ci) {
                    continue;
                }
                assigned_chunks.insert(ci);
                claimed_fields.insert(fi);

                let mut pr = plinko_results[ci].clone();
                pr.property = field_names[fi].clone();
                pr.score = *own;
                final_results.push(pr);
            }
        }
    }

    for (ci, pr) in plinko_results.iter().enumerate() {
        if !assigned_chunks.contains(&ci) {
            let mut unclassified = pr.clone();
            unclassified.property = "unclassified".to_string();
            final_results.push(unclassified);
        }
    }

    final_results
}

// =====================================================================
// 🌟 [PHASE C - 로그 헬퍼]
// =====================================================================

/// [PHASE C] PLINKO GAME + EXCLUSIVE ASSIGN 결과를 터미널에 출력합니다.
/// [PHASE C - 로그 헬퍼] PLINKO GAME + EXCLUSIVE ASSIGN 결과를 터미널에 출력합니다.
/// 확인 모드(all_scores 가 비어 있음)로 확정된 청크는 ✓ 마커로 표시합니다.
pub fn log_plinko_results(results: &[PlinkoResult]) {
    println!("  🎯 [PLINKO GAME / INDEXING] 확정 결과: {}개 청크", results.len());
    for (i, r) in results.iter().enumerate() {
        // ✓ = 확인 모드, ◇ = 발견 모드
        let mode_marker = if r.all_scores.is_empty() { "✓" } else { "◇" };
        let alt_str = if r.alternatives.is_empty() {
            String::from("-")
        } else {
            r.alternatives.iter()
                .map(|(name, score)| format!("{}({:.4})", name, score))
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!(
            "    [{}]{} property='{}' | score={:.4} | text='{}' | alts=[{}]",
            i, mode_marker, r.property, r.score, r.chunk_text, alt_str
        );
    }
}

/// [PHASE B+C - 통합 진입점] PHASE A 출력을 받아 PHASE B + PHASE C 전체 파이프라인을 순차 실행합니다.
/// 파이프라인 순서:
///   1. enrich_chunks_with_metadata()       — 형식 + bias/prejudice + 값 추출 (confirmed 전파)
///   2. nms_battle_for_indexing()           — property 기반 중복 제거 (confirmed 보호)
///   3. format_gate_for_indexing()          — 형식 검증 게이트 (배정 전)
///   4. plinko_game_for_indexing()          — 확인 모드 (confirmed) + Sliding Window (비confirmed)
///   5. exclusive_assign_for_indexing()     — confirmed 우선 + 그리디 배타 배정
///   6. PLINKO OVERRIDE 브릿지             — confirmed 청크는 OVERRIDE 불가
///
/// # 인자
///   - raw_chunks: split_natural_language_to_chunks() 반환값 (chunk_text, property, confirmed)
///   - doc_lang:   문서 언어 코드
///   - page_type:  도메인 타입
///   - field_names: 필드명 배열 (bias.json 에서 추출)
///   - field_phrase_embs: 필드별 bias 구 임베딩 뱅크
///   - field_phrase_weights: 필드별 구 가중치
///   - field_formats: 필드별 형식 문자열 배열
///   - embed_fn: 텍스트 → 임베딩 벡터 변환 클로저 (비동기)
///
/// # 반환
///   Vec<ChunkMetadata> — PHASE D(임베딩) + PHASE E(LanceDB 저장) 에 전달할 최종 배열
pub async fn run_phase_b_pipeline<F, Fut>(
    raw_chunks: &[(String, String, bool)],
    doc_lang: &str,
    page_type: &str,
    field_names: &[String],
    field_phrase_embs: &[Vec<Vec<f32>>],
    field_phrase_weights: &[Vec<f32>],
    field_formats: &[String],
    embed_fn: F,
) -> Vec<ChunkMetadata>
where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = Vec<f32>>,
{
    let _sds_index_guard = crate::utils::score_dynamics::indexing_guard();
    const SYSTEM_PROPERTIES: [&str; 17] = [
        "masked_text", "text", "unclassified", "context_intro", "json_data",
        "updated_at", "created_at", "digest", "index", "views",
        "embed", "node", "item", "detail",
        "table", "doc_type", "mode",
    ];
    const SYSTEM_HASH_PROPERTIES: [&str; 6] = ["id", "ref", "cc", "bcc", "from", "to"];
    let is_system_hash = |t: &str| -> bool {
        let w = t.trim_matches(|c: char| !c.is_ascii_alphanumeric());
        w.len() == 42
            && (w.starts_with("0x") || w.starts_with("0X"))
            && w[2..].chars().all(|c| c.is_ascii_hexdigit())
    };
    let is_envelope_key = |p: &str| -> bool {
        crate::store::ENVELOPE_COLUMNS.contains(&p) && !SYSTEM_HASH_PROPERTIES.contains(&p)
    };
    let mut hash_dropped = 0usize;
    let mut relay_dropped = 0usize;
    let filtered_chunks: Vec<&(String, String, bool)> = raw_chunks
        .iter()
        .filter(|(text, property, _)| {
            if SYSTEM_PROPERTIES.iter().any(|s| *s == property.as_str()) || is_envelope_key(property.as_str()) {
                return false;
            }
            if crate::utils::canonical::is_relay_index_key(property)
                && !crate::utils::canonical::relay_text_is_content(&extract_value_from_chunk(text))
            {
                relay_dropped += 1;
                return false;
            }
            if SYSTEM_HASH_PROPERTIES.iter().any(|s| *s == property.as_str())
                && text.split_whitespace().any(|t| is_system_hash(t))
            {
                hash_dropped += 1;
                return false;
            }
            true
        })
        .collect();
    crate::utils::score_dynamics::record_baseline("indexing.system_hash_drop", hash_dropped as f32);
    crate::utils::score_dynamics::record_baseline("indexing.relay_literal_drop", relay_dropped as f32);
    if relay_dropped > 0 {
        println!(
            "  🚫 [PHASE A FILTER / RELAY INDEX] 릴레이 인덱스 키(goods·order·tracking)에 숫자 값(crc32 인덱스)을 가진 청크 {}개를 뺍니다. 같은 키라도 글자가 들어 있는 값(주문 목록의 상품명 등)은 문서 내용이므로 남깁니다.",
            relay_dropped
        );
    }
    if hash_dropped > 0 {
        println!(
            "  🚫 [PHASE A FILTER / SYSTEM HASH] 저장소가 발급한 해시 식별자(0x + 16진 40자리)를 값으로 가진 청크 {}개를 뺍니다. 문서에 인쇄된 사실이 아니라서 검색이 만날 이유가 없고, 인덱싱 역검증에서 매번 doc_number 에 밀려 CONFIRM FLAG 와 혼동 사전(id,link ↔ doc_number)에 같은 잡음을 문서마다 남깁니다. 인쇄된 번호(상품 코드 등)는 이 모양이 아니므로 남습니다.",
            hash_dropped
        );
    }

    let removed_count = raw_chunks.len() - filtered_chunks.len();
    if removed_count > 0 {
        println!(
            "  🚫 [PHASE A FILTER] 자기참조/시스템 청크 {}개 인덱싱 대상에서 제외",
            removed_count
        );
    }

    let canonicalize = |p: &str| -> String {
        if field_names.iter().any(|f| f == p) {
            return p.to_string();
        }
        // 콤마 결합 키('id,link')의 구성 요소와 완전일치하면 그 결합 키로 승격
        for f in field_names.iter() {
            if f.split(',').any(|part| part.trim() == p) {
                return f.clone();
            }
        }
        p.to_string()
    };

    let mut canon_log: Vec<String> = Vec::new();
    let filtered_owned: Vec<(String, String, bool)> = filtered_chunks
        .into_iter()
        .map(|(t, p, c)| {
            let cp = canonicalize(p.as_str());
            if &cp != p && canon_log.len() < 8 {
                canon_log.push(format!("{}→{}", p, cp));
            }
            (t.clone(), cp, *c)
        })
        .collect();

    if !canon_log.is_empty() {
        println!(
            "  🔧 [SCHEMA CANONICALIZE] Phase A 속성명을 스키마 필드명으로 정규화: {:?}",
            canon_log
        );
    }

    // ── PHASE B ──
    // Step 1: 메타데이터 부여 (confirmed 플래그 전파)
    let enriched = enrich_chunks_with_metadata(&filtered_owned, doc_lang, page_type);

    // Step 2: NMS 중복 제거 (confirmed 보호 포함)
    let deduplicated = nms_battle_for_indexing(enriched);

    // Step 3: 형식 검증 게이트 (배정 전)
    let gated = format_gate_for_indexing(deduplicated);

    // ── PHASE C: PLINKO GAME ──
    // Step 4: 확인 모드(confirmed) + Sliding Window Cliff Detection(비confirmed)
    let plinko_results = plinko_game_for_indexing(
        &gated,
        field_names,
        field_phrase_embs,
        field_phrase_weights,
        field_formats,
        embed_fn,
    ).await;

    // Step 5: confirmed 우선 + 그리디 배타 배정
    let assigned = exclusive_assign_for_indexing(plinko_results, field_names);

    // Step 6: 로그 출력
    log_plinko_results(&assigned);

    // ── PHASE C → PHASE D 브릿지 ──
    // PLINKO 확정 결과를 ChunkMetadata 에 반영합니다.
    //
    // 🌟 [CONFIRMED PROTECTION] confirmed = true 인 청크는 Phase A 패턴 매칭에서
    //    JSON 구조로 확정된 것이므로, PLINKO 결과가 아무리 달라 보여도
    //    OVERRIDE 하지 않습니다. PLINKO 는 "확인"만 수행하며,
    //    결정론적 진실을 코사인 점수가 뒤집을 수 없습니다.
    //
    // 비confirmed 청크(unclassified 등)에 대해서만 PLINKO OVERRIDE 를 적용합니다.
    let mut final_chunks = gated;
    let mut override_count = 0usize;
    let mut protect_count = 0usize;

    for pr in &assigned {
        // 청크 텍스트로 매칭되는 ChunkMetadata 를 찾습니다.
        if let Some(chunk_meta) = final_chunks.iter_mut().find(|c| {
            c.chunk_text == pr.chunk_text
        }) {
            // 🌟 confirmed 청크는 PLINKO OVERRIDE 대상에서 제외합니다.
            if chunk_meta.confirmed {
                // PLINKO 가 다른 속성을 제안하더라도 무시합니다.
                if pr.property != "unclassified" && chunk_meta.property != pr.property {
                    protect_count += 1;
                    if protect_count <= 5 {
                        println!(
                            "  🛡️ [CONFIRMED PROTECT] '{}' | Phase A 확정 '{}' 유지 (PLINKO 제안 '{}' 무시, score: {:.4})",
                            pr.chunk_text, chunk_meta.property, pr.property, pr.score
                        );
                    }
                }
                continue;
            }

            // 비confirmed 청크: PLINKO 가 확정한 속성이 unclassified 가 아니면 덮어씁니다.
            if pr.property != "unclassified" {
                if chunk_meta.property != pr.property {
                    override_count += 1;
                    if override_count <= 10 {
                        println!(
                            "  🔄 [PLINKO OVERRIDE] '{}' | '{}' → '{}' (score: {:.4})",
                            pr.chunk_text, chunk_meta.property, pr.property, pr.score
                        );
                    }
                }
                chunk_meta.property = pr.property.clone();
                chunk_meta.property_format = field_format_to_string(&pr.property);
            }
        }
    }

    if protect_count > 0 || override_count > 0 {
        println!(
            "  📊 [BRIDGE SUMMARY] CONFIRMED 보호: {}건 | PLINKO OVERRIDE: {}건",
            protect_count, override_count
        );
    }

    final_chunks
}
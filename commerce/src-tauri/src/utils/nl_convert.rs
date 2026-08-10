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
                    intro.push_str(&format!("Regarding {},", context_name));
                }

                if !intro.is_empty() && !sentences.contains(&intro) {
                    sentences.push(intro);
                }

                for (key, v) in map {
                    // 이미 처리된 핵심 속성 및 시스템 변수는 스킵
                    if ["title", "name", "type", "currency", "text", "json_data", "data", "id", "index", "no", "link", "path", "origin", "mode", "detail"].contains(&key.as_str()) { continue; }
                    if v.is_null() || (v.is_string() && v.as_str().unwrap_or("").trim().is_empty()) { continue; }

                    let clean_key = key.replace("_", " ");

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
pub fn split_natural_language_to_chunks(text: &str) -> Vec<(String, String)> {
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
    let mut chunks: Vec<(String, String)> = Vec::new();

    for sentence in &raw_sentences {
        let s = sentence.trim().trim_end_matches('.').trim();
        if s.is_empty() {
            continue;
        }

        // 패턴 1: "The unique identifier is {value}"
        // → property = "id"
        if let Some(rest) = s.strip_prefix("The unique identifier is ") {
            chunks.push((format!("The unique identifier is {}", rest), "id".to_string()));
            continue;
        }

        // 패턴 2: "It can be accessed at {value}"
        // → property = "link"
        if let Some(rest) = s.strip_prefix("It can be accessed at ") {
            chunks.push((format!("It can be accessed at {}", rest), "link".to_string()));
            continue;
        }

        // 패턴 3: "This {type} is titled '{title}'"
        // → property = "title"
        if s.contains("is titled '") {
            chunks.push((s.to_string(), "title".to_string()));
            continue;
        }

        // 패턴 4: "It is currently in '{value}' status"
        // → property = "status"
        if s.contains("is currently in '") && s.ends_with("status") {
            chunks.push((s.to_string(), "status".to_string()));
            continue;
        }

        // 패턴 5: "The {context} includes: {val1}, {val2}, ..."
        // → 배열 값을 개별 청크로 분할, property = context (snake_case 변환)
        if let Some(includes_pos) = s.find(" includes: ") {
            let context_part = &s[..includes_pos];
            let values_part = &s[includes_pos + " includes: ".len()..];

            // "The {context}" 에서 "The " 제거 후 snake_case 변환
            let context_raw = context_part.trim().strip_prefix("The ").unwrap_or(context_part.trim());
            let property = context_raw.to_lowercase().replace(' ', "_");

            // 콤마로 배열 값 분할
            let values: Vec<&str> = values_part
                .split(',')
                .map(|v| v.trim())
                .filter(|v| !v.is_empty())
                .collect();

            if values.len() == 1 {
                chunks.push((format!("{} includes {}", context_raw, values[0]), property));
            } else {
                for val in &values {
                    chunks.push((format!("{} includes {}", context_raw, val), property.clone()));
                }
            }
            continue;
        }

        // 패턴 6: "Regarding {context},"
        // → 중첩 객체 컨텍스트 도입부, property = context
        if let Some(rest) = s.strip_prefix("Regarding ") {
            let context = rest.trim_end_matches(',').trim();
            let property = context.to_lowercase().replace(' ', "_");
            chunks.push((s.to_string(), property));
            continue;
        }

        // 패턴 7: "Its {key} is {value}"
        // → property = key (snake_case)
        if let Some(rest) = s.strip_prefix("Its ") {
            if let Some(is_pos) = rest.find(" is ") {
                let key_part = rest[..is_pos].trim();
                let val_part = rest[is_pos + " is ".len()..].trim();
                let property = key_part.to_lowercase().replace(' ', "_");
                chunks.push((format!("Its {} is {}", key_part, val_part), property));
                continue;
            }
        }

        // 패턴 8: "The {key} is {value} {currency}"
        // → property = key (snake_case)
        if let Some(rest) = s.strip_prefix("The ") {
            if let Some(is_pos) = rest.find(" is ") {
                let key_part = rest[..is_pos].trim();
                let val_part = rest[is_pos + " is ".len()..].trim();
                let property = key_part.to_lowercase().replace(' ', "_");
                chunks.push((format!("The {} is {}", key_part, val_part), property));
                continue;
            }
        }

        // 패턴 매칭 실패 시: 전체 문장을 "unclassified" 로 보관
        chunks.push((s.to_string(), "unclassified".to_string()));
    }

    // ── Step 3: 150자 초과 문장 2차 분할 ──
    let mut expanded: Vec<(String, String)> = Vec::new();
    for (chunk_text, property) in &chunks {
        if chunk_text.chars().count() > 150 {
            // 콤마 기준으로 분할 시도
            let parts: Vec<&str> = chunk_text
                .split(',')
                .map(|p| p.trim())
                .filter(|p| !p.is_empty())
                .collect();
            if parts.len() > 1 {
                for part in &parts {
                    expanded.push((part.to_string(), property.clone()));
                }
                continue;
            }
        }
        expanded.push((chunk_text.clone(), property.clone()));
    }

    // ── Step 4: 1 단어 청크 병합 (R3) ──
    // 단어 수가 1개인 청크는 직전 청크에 병합합니다.
    // 단, 식별자/URL/숫자 리터럴 포함 청크(R4)는 독립 보존합니다.
    let mut merged: Vec<(String, String)> = Vec::new();
    for (chunk_text, property) in &expanded {
        let word_count = chunk_text.split_whitespace().count();
        let has_identifier = chunk_text.contains("http")
            || chunk_text.contains('/')
            || chunk_text.chars().any(|c| c.is_ascii_digit());

        if word_count < 2 && !has_identifier && !merged.is_empty() {
            // 직전 청크에 병합
            let last = merged.last_mut().unwrap();
            last.0.push(' ');
            last.0.push_str(chunk_text);
        } else {
            merged.push((chunk_text.clone(), property.clone()));
        }
    }

    merged
}

/// [PHASE A - 테스트 헬퍼] 분할 결과를 디버그 로그로 출력합니다.
/// scheduler.rs 에서 인덱싱 파이프라인 호출 시 사용합니다.
pub fn log_chunk_split_result(chunks: &[(String, String)]) {
    println!("  📝 [PHASE A] RAW-CHUNK 분할 결과: {}개 청크", chunks.len());
    for (i, (text, prop)) in chunks.iter().enumerate() {
        println!("    [{}] property='{}' | text='{}'", i, prop, text);
    }
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
}

/// [PHASE B] detect_field_format() 결과를 문자열로 변환합니다.
/// ai_utils.rs 의 FieldFormat enum 을 직접 참조하지 않고
/// 문자열로 변환하여 nl_convert.rs 의 독립성을 유지합니다.
fn field_format_to_string(field_name: &str) -> String {
    let lower = field_name.to_lowercase();
    let keys: Vec<String> = lower.split(',').map(|s| s.trim().to_string()).collect();
    let has = |k: &str| keys.iter().any(|x| x == k);

    if keys.iter().any(|k| k.contains("insight") || k.contains("summary") || k.contains("analysis")) {
        return "Synthesis".to_string();
    }
    if keys.iter().any(|k| k.contains("tracking_number") || k == "barcode" || k == "gtin" || k == "mpn") {
        return "TrackingCode".to_string();
    }
    if has("id") || has("code") || has("no") || has("index") || has("stock_keeping_unit") {
        return "Identifier".to_string();
    }
    if keys.iter().any(|k| k.contains("link") || k.contains("url")) {
        return "Link".to_string();
    }
    if keys.iter().any(|k| k.contains("date") || k.ends_with("_at")) {
        return "Date".to_string();
    }
    if keys.iter().any(|k| {
        k.ends_with("phone") || k == "tel" || k == "telephone" || k == "mobile"
            || k == "cellphone" || k == "contact" || k == "number"
    }) {
        return "Phone".to_string();
    }
    if keys.iter().any(|k| k == "address" || k.ends_with("_address")) {
        return "Address".to_string();
    }
    if keys.iter().any(|k| {
        k.contains("status") || k.contains("payment_method") || k.contains("payment_origin")
            || k.contains("condition") || k.contains("currency") || k == "bank" || k == "card"
    }) {
        return "Enum".to_string();
    }
    if keys.iter().any(|k| {
        k.contains("price") || k.contains("amount") || k.contains("quantity") || k.contains("weight")
            || k == "width" || k == "height" || k == "length" || k.contains("fee")
            || k.contains("discount") || k.contains("usage_") || k.contains("threshold")
            || k.contains("duration")
    }) {
        return "Numeric".to_string();
    }
    "Text".to_string()
}

/// [PHASE B] bias.json 에서 해당 필드의 semantic + bias 구를 동적으로 읽어옵니다.
/// 다국어 하드코딩 없이 bias.json 의 기존 구조만 사용합니다.
///
/// 반환값: (bias_phrases, prejudice_phrases)
///   - bias_phrases: semantic 앵커 + bias 구를 split_bias_phrases_full 로 분할한 목록
///   - prejudice_phrases: prejudice 구를 분할한 목록
fn get_field_bias_phrases(doc_lang: &str, page_type: &str, field_name: &str) -> (Vec<String>, Vec<String>) {
    let dict: &serde_json::Value = &crate::parsing::BIAS_DICT;

    let mut bias_phrases: Vec<String> = Vec::new();
    let mut prejudice_phrases: Vec<String> = Vec::new();

    // 1. semantic 앵커 추출 (기존 semantic_anchor_text 로직 인라인)
    for lk in [doc_lang, "en", "ko"] {
        let lang_node = match dict.get(lk) { Some(v) => v, None => continue };
        if let Some(s) = lang_node
            .get(page_type)
            .and_then(|p| p.get(field_name))
            .and_then(|n| n.get("semantic"))
            .and_then(|v| v.as_str())
        {
            if !s.trim().is_empty() {
                for phrase in s.split(|c: char| c == ',' || c == '\n' || c == '/' || c == '|') {
                    let p = phrase.trim().to_string();
                    if !p.is_empty() && !bias_phrases.iter().any(|e| e == &p) {
                        bias_phrases.push(p);
                    }
                }
                break;
            }
        }
        if let Some(s) = lang_node
            .get("default")
            .and_then(|p| p.get(field_name))
            .and_then(|n| n.get("semantic"))
            .and_then(|v| v.as_str())
        {
            if !s.trim().is_empty() {
                for phrase in s.split(|c: char| c == ',' || c == '\n' || c == '/' || c == '|') {
                    let p = phrase.trim().to_string();
                    if !p.is_empty() && !bias_phrases.iter().any(|e| e == &p) {
                        bias_phrases.push(p);
                    }
                }
                break;
            }
        }
    }

    // 2. bias 구 추출
    for lk in [doc_lang, "en", "ko"] {
        let lang_node = match dict.get(lk) { Some(v) => v, None => continue };
        if let Some(b) = lang_node
            .get(page_type)
            .and_then(|p| p.get(field_name))
            .and_then(|n| n.get("bias"))
            .and_then(|v| v.as_str())
        {
            if !b.trim().is_empty() {
                for phrase in b.split(|c: char| c == ',' || c == '\n' || c == '/' || c == '|') {
                    let p = phrase.trim().to_string();
                    if !p.is_empty() && !bias_phrases.iter().any(|e| e == &p) {
                        bias_phrases.push(p);
                    }
                }
                break;
            }
        }
    }

    // 3. prejudice 구 추출
    for lk in [doc_lang, "en", "ko"] {
        let lang_node = match dict.get(lk) { Some(v) => v, None => continue };
        if let Some(p) = lang_node
            .get(page_type)
            .and_then(|pp| pp.get(field_name))
            .and_then(|n| n.get("prejudice"))
            .and_then(|v| v.as_str())
        {
            if !p.trim().is_empty() {
                for phrase in p.split(|c: char| c == ',' || c == '\n' || c == '/' || c == '|') {
                    let ph = phrase.trim().to_string();
                    if !ph.is_empty() && !prejudice_phrases.iter().any(|e| e == &ph) {
                        prejudice_phrases.push(ph);
                    }
                }
                break;
            }
        }
    }

    // 4. 루트 전역 노드 폴백 (color, metrics.*, operators.* 등)
    if bias_phrases.is_empty() {
        let mut stack: Vec<&serde_json::Value> = vec![dict];
        let mut hops = 0usize;
        while let Some(node) = stack.pop() {
            hops += 1;
            if hops > 4096 { break; }
            if let Some(obj) = node.as_object() {
                if let Some(child) = obj.get(field_name) {
                    if let Some(b) = child.get("bias").and_then(|v| v.as_str()) {
                        for phrase in b.split(|c: char| c == ',' || c == '\n' || c == '/' || c == '|') {
                            let p = phrase.trim().to_string();
                            if !p.is_empty() && !bias_phrases.iter().any(|e| e == &p) {
                                bias_phrases.push(p);
                            }
                        }
                    }
                    if let Some(p) = child.get("prejudice").and_then(|v| v.as_str()) {
                        for phrase in p.split(|c: char| c == ',' || c == '\n' || c == '/' || c == '|') {
                            let ph = phrase.trim().to_string();
                            if !ph.is_empty() && !prejudice_phrases.iter().any(|e| e == &ph) {
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

    // 5. bias_phrases 상한 (기존 split_bias_phrases 의 48개 상한과 동일)
    if bias_phrases.len() > 48 { bias_phrases.truncate(48); }
    if prejudice_phrases.len() > 48 { prejudice_phrases.truncate(48); }

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
fn extract_value_from_chunk(chunk_text: &str) -> String {
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
    raw_chunks: &[(String, String)],
    doc_lang: &str,
    page_type: &str,
) -> Vec<ChunkMetadata> {
    let mut results: Vec<ChunkMetadata> = Vec::with_capacity(raw_chunks.len());

    for (chunk_text, property) in raw_chunks {
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
        });
    }

    results
}

/// [PHASE B - NMS] 인덱싱 전용 중복 제거입니다.
///
/// 검색 파이프라인의 NMS BATTLE 은 임베딩 코사인으로 오버랩을 판정하지만,
/// 인덱싱 단계에서는 아직 임베딩이 생성되지 않았으므로
/// **텍스트 구조 기반** 중복 제거를 수행합니다.
///
/// 제거 규칙 (코사인 없이 결정론):
///   D1: 완전 동일 텍스트 → 하나만 보존
///   D2: 동일 property + 동일 value_part → 하나만 보존
///   D3: 한 청크의 텍스트가 다른 청크의 텍스트에 완전 포함(부분문자열) →
///       짧은 쪽 폐기 (단, 식별자/URL/숫자 리터럴 포함 청크는 독립 보존)
///
/// 이 규칙은 contains() 기반 '의미 판정'이 아니라
/// '동일 문자열 포함'이라는 구조적 사실만 사용하므로 설계 원칙을 위반하지 않습니다.
///
/// # 인자
///   - chunks: enrich_chunks_with_metadata() 반환값
///
/// # 반환
///   중복 제거 후 Vec<ChunkMetadata>
pub fn nms_battle_for_indexing(chunks: Vec<ChunkMetadata>) -> Vec<ChunkMetadata> {
    if chunks.len() <= 1 {
        return chunks;
    }

    let mut survivors: Vec<ChunkMetadata> = Vec::with_capacity(chunks.len());
    let mut removed_count = 0usize;

    for chunk in chunks {
        // D1: 완전 동일 텍스트 검사
        let is_exact_dup = survivors.iter().any(|s| s.chunk_text == chunk.chunk_text);
        if is_exact_dup {
            removed_count += 1;
            continue;
        }

        // D2: 동일 property + 동일 value_part 검사
        let is_prop_val_dup = survivors.iter().any(|s| {
            s.property == chunk.property && s.value_part == chunk.value_part
        });
        if is_prop_val_dup {
            removed_count += 1;
            continue;
        }

        // D3: 부분문자열 포함 검사 (짧은 쪽 폐기)
        //     단, 식별자/URL/숫자 리터럴 포함 청크는 독립 보존 (R4 규칙)
        let has_identifier = chunk.chunk_text.contains("http")
            || chunk.chunk_text.contains('/')
            || chunk.chunk_text.chars().any(|c| c.is_ascii_digit());

        if !has_identifier {
            let is_substring_of_existing = survivors.iter().any(|s| {
                s.chunk_text.contains(&chunk.chunk_text) && s.chunk_text != chunk.chunk_text
            });
            if is_substring_of_existing {
                removed_count += 1;
                continue;
            }
        }

        // 기존 survivor 중 현재 청크에 포함되는 짧은 청크가 있으면 폐기
        let mut to_remove_indices: Vec<usize> = Vec::new();
        for (si, survivor) in survivors.iter().enumerate() {
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
            // 뒤에서부터 제거하여 인덱스 시프트 방지
            to_remove_indices.sort_unstable();
            to_remove_indices.dedup();
            for &idx in to_remove_indices.iter().rev() {
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
pub fn format_gate_for_indexing(mut chunks: Vec<ChunkMetadata>) -> Vec<ChunkMetadata> {
    let mut rejected_count = 0usize;

    for chunk in chunks.iter_mut() {
        let val = chunk.value_part.trim();
        if val.is_empty() {
            continue;
        }

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
            "Text" => val.chars().any(|c| c.is_alphabetic()),
            "Address" => val.chars().any(|c| c.is_alphabetic()) && val.split_whitespace().count() >= 2,
            "Synthesis" => true,
            _ => true,
        };

        if !passes {
            println!(
                "  🚧 [FORMAT GATE / INDEXING] '{}' (property='{}', format='{}') 형식 불일치 → unclassified 강등",
                chunk.chunk_text, chunk.property, chunk.property_format
            );
            chunk.property = "unclassified".to_string();
            chunk.property_format = "Text".to_string();
            rejected_count += 1;
        }
    }

    if rejected_count > 0 {
        println!(
            "  🚧 [FORMAT GATE / INDEXING] 총 {}개 청크 형식 불일치 강등",
            rejected_count
        );
    }

    chunks
}

/// [PHASE B - 로그 헬퍼] 메타데이터 부여 + NMS + FORMAT GATE 전체 결과를 출력합니다.
pub fn log_enriched_chunks(chunks: &[ChunkMetadata]) {
    println!("  📋 [PHASE B] 메타데이터 부여 완료: {}개 청크", chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        println!(
            "    [{}] property='{}' | format='{}' | bias={}구 | prej={}구 | value='{}' | text='{}'",
            i,
            c.property,
            c.property_format,
            c.bias_phrases.len(),
            c.prejudice_phrases.len(),
            c.value_part,
            c.chunk_text
        );
    }
}

/// [PHASE B - 통합 진입점] PHASE A 출력을 받아 PHASE B 전체 파이프라인을 순차 실행합니다.
///
/// 호출 순서:
///   1. enrich_chunks_with_metadata()  — 형식 + bias/prejudice + 값 추출
///   2. nms_battle_for_indexing()      — 텍스트 레벨 중복 제거
///   3. format_gate_for_indexing()     — 형식 검증 게이트
///
/// # 인자
///   - raw_chunks: split_natural_language_to_chunks() 반환값
///   - doc_lang:   문서 언어 코드
///   - page_type:  도메인 타입
///
/// # 반환
///   Vec<ChunkMetadata> — PHASE C(임베딩) + PHASE D(LanceDB 저장) 에 전달할 최종 배열
pub fn run_phase_b_pipeline(
    raw_chunks: &[(String, String)],
    doc_lang: &str,
    page_type: &str,
) -> Vec<ChunkMetadata> {
    // Step 1: 메타데이터 부여
    let enriched = enrich_chunks_with_metadata(raw_chunks, doc_lang, page_type);

    // Step 2: NMS 중복 제거
    let deduplicated = nms_battle_for_indexing(enriched);

    // Step 3: 형식 검증 게이트
    let gated = format_gate_for_indexing(deduplicated);

    gated
}
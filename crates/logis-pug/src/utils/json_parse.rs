use regex::Regex;

pub fn sanitize_llm_input(text: &str) -> String {
    // 1. Filter out non-printable and non-ASCII/Korean characters if they are broken
    // But we want to keep Korean. Let's filter out known problematic control codes.
    let cleaned: String = text.chars()
        .filter(|c| {
            let u = *c as u32;
            // Keep: standard ASCII, Korean Hangul Jamo/Syllables, Common Punctuation
            (u >= 32 && u <= 126) || // Basic ASCII
            (u >= 0xAC00 && u <= 0xD7A3) || // Hangul Syllables
            (u >= 0x1100 && u <= 0x11FF) || // Hangul Jamo
            (u >= 0x3130 && u <= 0x318F) || // Hangul Compatibility Jamo
            u == 10 || u == 13 || u == 9     // \n, \r, \t
        })
        .collect();
    // 2. Prevent internal special tokens from being interpreted
    cleaned.replace("<|", "< |").replace("|>", "| >")
}

fn quote_bare_values(src: &str) -> String {
    let chars: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len() + 32);
    let mut in_string = false;
    let mut escape = false;
    let mut i = 0usize;
    let mut fixed: Vec<String> = Vec::new();

    while i < chars.len() {
        let c = chars[i];
        if escape {
            out.push(c);
            escape = false;
            i += 1;
            continue;
        }
        if in_string {
            out.push(c);
            if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            i += 1;
            continue;
        }
        if c != ':' {
            out.push(c);
            i += 1;
            continue;
        }

        out.push(':');
        i += 1;
        let mut ws = String::new();
        while i < chars.len()
            && (chars[i] == ' ' || chars[i] == '\t' || chars[i] == '\n' || chars[i] == '\r')
        {
            ws.push(chars[i]);
            i += 1;
        }
        out.push_str(&ws);
        if i >= chars.len() {
            break;
        }
        let first = chars[i];
        if first == '"' || first == '{' || first == '[' {
            continue;
        }

        let mut tok = String::new();
        while i < chars.len()
            && chars[i] != ','
            && chars[i] != '}'
            && chars[i] != ']'
            && chars[i] != '\n'
            && chars[i] != '\r'
        {
            tok.push(chars[i]);
            i += 1;
        }
        let trimmed = tok.trim_end().to_string();
        let tail: String = tok.chars().skip(trimmed.chars().count()).collect();
        let is_literal = matches!(trimmed.as_str(), "null" | "true" | "false");
        let numeric_shape = trimmed
            .chars()
            .next()
            .map(|c0| c0.is_ascii_digit() || c0 == '-' || c0 == '+')
            .unwrap_or(false);
        let is_number = numeric_shape && trimmed.parse::<f64>().is_ok();

        if trimmed.is_empty() || is_literal || is_number {
            out.push_str(&tok);
        } else {
            out.push('"');
            out.push_str(&trimmed.replace('\\', "\\\\").replace('"', "\\\""));
            out.push('"');
            out.push_str(&tail);
            if fixed.len() < 8 {
                fixed.push(trimmed.clone());
            }
        }
    }

    if !fixed.is_empty() {
        println!(
            "[Parsing] 🩹 [BARE VALUE QUOTED] 따옴표 없는 값 {}개를 문자열로 감쌌습니다: {:?}",
            fixed.len(),
            fixed
        );
    }
    out
}

pub fn normalize_to_json_string(input: &str) -> String {
    let mut s = input.replace(&['\u{00A0}', '\u{200B}', '\u{202F}', '\u{FEFF}'][..], " ").trim().to_string();
    // 🌟 [CRITICAL FIX] LLM이 흔히 생성하는 스마트 따옴표 및 전각 기호를 표준 ASCII 기호로 일괄 치환합니다.
    s = s.replace('\u{201C}', "\"")
         .replace('\u{201D}', "\"")
         .replace('\u{2018}', "'")
         .replace('\u{2019}', "'")
         .replace('\u{FF0C}', ",")
         .replace('\u{FF1A}', ":");
    // 🌟 [LLM ESCAPE FIX] Qwen 모델이 JSON 내부에서 "has_list\":false 처럼 큰따옴표 앞에 오염시킨 백슬래시(\)를 원천 제거합니다.
    s = s.replace("\\\"", "\"");
    // 1. Backticks to quotes
    let re_backtick = Regex::new(r"`([\s\S]*?)`").unwrap();
    s = re_backtick.replace_all(&s, |caps: &regex::Captures| {
        format!("\"{}\"", caps[1].replace("\"", "\\\""))
    }).to_string();
    // 2. Key quotes correction (key: -> "key":)
    // 🌟 [CRITICAL FIX] 줄 시작 지점(^), 중괄호({), 쉼표(,) 뒤에 나오는 Unquoted Key를 모두 확실히 잡아내어 따옴표로 감쌉니다.
    let re_keys = Regex::new(r"(?m)(^|[{,])\s*([a-zA-Z0-9_]+)\s*:").unwrap();
    s = re_keys.replace_all(&s, "$1\"$2\":").to_string();
    // 3. Single quotes to double quotes for values
    let re_single_vals = Regex::new(r":\s*'([^']*)'").unwrap();
    s = re_single_vals.replace_all(&s, |caps: &regex::Captures| {
        format!(": \"{}\"", caps[1].replace("\"", "\\\""))
    }).to_string();
    // 4. CSS selector/nested quotes protection (Simplified for Rust regex)
    let re_nested = Regex::new(r#"="([^"]*)""#).unwrap();
    s = re_nested.replace_all(&s, "=\\\"$1\\\"").to_string();
    s = quote_bare_values(&s);
    // 5. Trailing Comma removal
    let re_trailing = Regex::new(r",\s*([\]}])").unwrap();
    s = re_trailing.replace_all(&s, "$1").to_string();
    let re_artifact = Regex::new(r",?\s*\.\.\.\s*([\]}])").unwrap();
    s = re_artifact.replace_all(&s, "$1").to_string();
    let mut in_string = false;
    let mut escape = false;
    let mut stack = Vec::new();
    for c in s.chars() {
        if escape {
            escape = false;
        } else if c == '\\' {
            escape = true;
        } else if c == '"' {
            in_string = !in_string;
        } else if !in_string {
            match c {
                '{' => stack.push('}'),
                '[' => stack.push(']'),
                '}' | ']' => {
                    if stack.last() == Some(&c) {
                        stack.pop();
                    }
                }
                _ => {}
            }
        }
    }
    // 문자열이 끊겼을 경우 닫는 따옴표 추가 (끝부분 이스케이프 문자 방어)
    if in_string {
        if s.ends_with('\\') {
            s.pop();
        }
        s.push('"');
    }
    // 뒤에 쉼표가 남은 상태로 끊겼을 경우 쉼표 제거
    s = s.trim_end().to_string();
    if s.ends_with(',') {
        s.pop();
    }
    // 남은 괄호 스택을 순서대로 역전하여 전부 조립
    while let Some(c) = stack.pop() {
        s.push(c);
    }
    s
}

fn salvage_partial_json(src: &str) -> Option<serde_json::Value> {
    let chars: Vec<char> = src.chars().collect();
    let obj_at = chars.iter().position(|c| *c == '{');
    let arr_at = chars.iter().position(|c| *c == '[');
    let (start, is_obj) = match (obj_at, arr_at) {
        (Some(o), Some(a)) => {
            if o < a {
                (o, true)
            } else {
                (a, false)
            }
        }
        (Some(o), None) => (o, true),
        (None, Some(a)) => (a, false),
        (None, None) => return None,
    };

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    let mut closed = false;
    let mut seg = String::new();
    let mut segs: Vec<String> = Vec::new();

    for i in start..chars.len() {
        let c = chars[i];
        if escape {
            seg.push(c);
            escape = false;
            continue;
        }
        if in_string {
            seg.push(c);
            if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                seg.push(c);
            }
            '{' | '[' => {
                depth += 1;
                if depth > 1 {
                    seg.push(c);
                }
            }
            '}' | ']' => {
                depth -= 1;
                if depth >= 1 {
                    seg.push(c);
                }
                if depth <= 0 {
                    closed = true;
                }
            }
            ',' if depth == 1 => {
                if !seg.trim().is_empty() {
                    segs.push(seg.clone());
                }
                seg.clear();
            }
            _ => {
                seg.push(c);
            }
        }
        if closed {
            break;
        }
    }
    if !seg.trim().is_empty() {
        segs.push(seg);
    }
    if segs.is_empty() {
        return None;
    }

    if is_obj {
        let mut map = serde_json::Map::new();
        let mut dropped: Vec<String> = Vec::new();
        for s in segs.iter() {
            let candidate = format!("{{{}}}", s.trim());
            match serde_json::from_str::<serde_json::Value>(&candidate) {
                Ok(serde_json::Value::Object(o)) => {
                    for (k, v) in o {
                        map.insert(k, v);
                    }
                }
                _ => {
                    if dropped.len() < 8 {
                        dropped.push(s.trim().chars().take(60).collect());
                    }
                }
            }
        }
        if map.is_empty() {
            return None;
        }
        println!(
            "[Parsing] 🩹 [PAIR SALVAGE] 최상위 키 {}개를 보존하고 문법이 깨진 {}개만 버렸습니다: {:?}",
            map.len(),
            dropped.len(),
            dropped
        );
        return Some(serde_json::Value::Object(map));
    }

    let mut arr: Vec<serde_json::Value> = Vec::new();
    let mut dropped = 0usize;
    for s in segs.iter() {
        match serde_json::from_str::<serde_json::Value>(s.trim()) {
            Ok(v) => arr.push(v),
            Err(_) => dropped += 1,
        }
    }
    if arr.is_empty() {
        return None;
    }
    println!(
        "[Parsing] 🩹 [ELEMENT SALVAGE] 배열 원소 {}개를 보존하고 문법이 깨진 {}개만 버렸습니다.",
        arr.len(),
        dropped
    );
    Some(serde_json::Value::Array(arr))
}

pub fn parse_json_from_llm(text: &str) -> serde_json::Value {
    // [CLEANUP] Remove <think>...</think> tags if they exist
    let mut clean_text = text.to_string();
    if let Some(start_think) = clean_text.find("<think>") {
        if let Some(end_think) = clean_text.find("</think>") {
            if start_think < end_think {
                clean_text.replace_range(start_think..end_think + 8, " ");
            }
        } else {
            clean_text.replace_range(start_think.., " ");
        }
    }
    let clean_text = clean_text.trim();
    // 1. First attempt: Direct parse
    if let Ok(v) = serde_json::from_str(clean_text) { return v; }
    // 2. Extract JSON part and attempt direct parse
    let mut extracted = String::new();
    if let Some(start) = clean_text.find("{") {
        if let Some(end) = clean_text.rfind("}") {
            if start < end {
                let attempt = clean_text[start..=end].to_string();
                if let Ok(v) = serde_json::from_str(&attempt) { return v; }
            }
        }
        extracted = clean_text[start..].to_string();
    } else if let Some(start) = clean_text.find("[") {
        if let Some(end) = clean_text.rfind("]") {
            if start < end {
                let attempt = clean_text[start..=end].to_string();
                if let Ok(v) = serde_json::from_str(&attempt) { return v; }
            }
        }
        extracted = clean_text[start..].to_string();
    }
    // 3. Last attempt: Normalize/Repair then parse
    let to_repair = if extracted.is_empty() { clean_text } else { &extracted };
    let repaired = normalize_to_json_string(to_repair);
    if let Ok(v) = serde_json::from_str(&repaired) {
        println!("[Parsing] Success: JSON successfully repaired and parsed!");
        return v;
    } else {
        // Final fallback: try extracting from repaired string again
        if let Some(start) = repaired.find("{") {
            if let Some(end) = repaired.rfind("}") {
                if let Ok(v) = serde_json::from_str(&repaired[start..=end]) {
                    println!("[Parsing] Success: JSON successfully repaired and parsed on final fallback!");
                    return v;
                }
            }
        }
        if let Some(v) = salvage_partial_json(&repaired) {
            return v;
        }
        println!("[Parsing] Attempting aggressive character-by-character truncation repair...");
        let mut shrink_attempt = to_repair.to_string();
        let mut attempts = 0;
        // 시스템 랙(Freezing)을 방지하기 위해 최대 500번(글자)까지만 깎아내며 재시도합니다. (최소 5글자 유지)
        while shrink_attempt.len() > 5 && attempts < 500 {
            shrink_attempt.pop(); // 맨 끝 글자 하나 제거
            attempts += 1;
            let attempt_repaired = normalize_to_json_string(&shrink_attempt);
            if let Ok(v) = serde_json::from_str(&attempt_repaired) {
                println!("[Parsing] Success: JSON repaired by aggressive truncation after dropping {} characters!", attempts);
                return v;
            }
        }
    }
    println!("[Parsing] Warning: Failed to repair dirty JSON: {}", clean_text);
    serde_json::json!({})
}
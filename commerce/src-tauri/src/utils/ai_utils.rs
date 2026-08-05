pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot_product / (norm_a * norm_b) }
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
                    if g3 >= 1 && g1 >= 2 && g1 <= 4 { return true; }
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

pub fn value_matches_format(fmt: FieldFormat, value: &str) -> bool {
    let v = value.trim();
    if v.is_empty() { return false; }
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

// 🌟 [DETERMINISTIC ID/LINK] LLM 에게 id 와 link 를 동시에 물어보면
// "id 는 주문번호, link 는 아무 상세페이지 href" 처럼 서로 무관한 값이 짝지어져 나옵니다.
// href 문자열 안에 실제로 포함된 식별자 토큰만 채택하면 id 와 link 가 구조적으로 100% 일치합니다.
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
        let lower_tok = tok.to_lowercase();
        for h in &hrefs {
            if h.to_lowercase().contains(&lower_tok) {
                let is_better = match &best {
                    None => true,
                    Some((bt, _)) => tok.chars().count() > bt.chars().count(),
                };
                if is_better { best = Some((tok.clone(), h.clone())); }
            }
        }
    }
    best
}

// 🌟 [URL PATTERN EXTRACT] 성공적으로 확정된 id/link 쌍에서 URL 구조 패턴을 추출합니다.
// 예: id="26020315071105", link="/admin/pop_orderform.php?od_id=26031514155635"
//   → prefix="/admin/pop_orderform.php?od_id=", suffix=""
// 이 패턴을 사용하면 href가 없는 아이템에서도 식별자만 있으면 link를 역산할 수 있습니다.
pub fn extract_url_pattern(id: &str, link: &str) -> Option<(String, String)> {
    if id.is_empty() || link.is_empty() { return None; }
    let lower_link = link.to_lowercase();
    let lower_id = id.to_lowercase();
    if let Some(pos) = lower_link.find(&lower_id) {
        let prefix = &link[..pos];
        let suffix = &link[pos + id.len()..];
        // prefix가 비어있으면 패턴으로서 의미가 없음 (link 자체가 id인 경우)
        if prefix.is_empty() && suffix.is_empty() { return None; }
        Some((prefix.to_string(), suffix.to_string()))
    } else {
        None
    }
}

// 🌟 [URL PATTERN APPLY] 추출된 패턴에 새 식별자를 대입하여 link를 생성합니다.
pub fn apply_url_pattern(prefix: &str, suffix: &str, new_id: &str) -> String {
    format!("{}{}{}", prefix, new_id, suffix)
}

// 🌟 [IDENTIFIER TOKEN SEARCH] PUG 라인 배열에서 식별자 후보 토큰을 탐색합니다.
// 조건: 파이프 뒤(value)에 위치, 8자 이상, 숫자 포함, 영숫자 토큰.
// "주문번호 | 26031514155635" → "26031514155635" 추출.
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
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
                    href: href.clone(),
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
                href: href.clone(),
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
                href: href.clone(),
                role_phrase: "host name domain name website address server address".to_string(),
                is_host_part: true,
                prior: 0.05,
            });
        }
    }

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
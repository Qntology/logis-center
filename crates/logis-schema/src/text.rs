//! 언어 중립 글 도우미: 숫자 · 월 이름 · 구(phrase) 나누기. 예전 utils/ai_utils.rs 에서 옮김 (정규화 · 무역 규칙 · 사전이 함께 씀).

pub fn is_abbreviation_slash(chars: &[char], i: usize) -> bool {
    let mut left = 0usize;
    let mut left_alpha = false;
    let mut j = i;
    while j > 0 && chars[j - 1].is_ascii_alphanumeric() {
        left += 1;
        left_alpha |= chars[j - 1].is_ascii_alphabetic();
        j -= 1;
    }
    let mut right = 0usize;
    let mut right_alpha = false;
    let mut k = i + 1;
    while k < chars.len() && chars[k].is_ascii_alphanumeric() {
        right += 1;
        right_alpha |= chars[k].is_ascii_alphabetic();
        k += 1;
    }
    left_alpha && right_alpha && (1..=2).contains(&left) && (1..=2).contains(&right)
}

pub fn split_phrase_list(raw: &str) -> Vec<String> {
    let chars: Vec<char> = raw.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for (i, c) in chars.iter().enumerate() {
        let sep = match *c {
            ',' | '\n' | '|' => true,
            '/' => !is_abbreviation_slash(&chars, i),
            _ => false,
        };
        if sep {
            let t = cur.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
            cur.clear();
        } else {
            cur.push(*c);
        }
    }
    let t = cur.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
    out
}

pub fn split_bias_phrases(raw: &str) -> Vec<String> {
    let mut v: Vec<String> = split_phrase_list(raw);
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

// 🌟 [UNCAPPED PHRASE SPLIT] split_bias_phrases 는 48개에서 잘라냅니다.
//    bias.json 의 color 뱅크는 50개 언어의 색상명이 수백 개 나열되어 있어
//    48개로 자르면 한국어/영어 이후의 언어(ベージュ, بيج, бежевый ...)가 통째로 소멸합니다.
//    다국어 검색이 목적이므로 속성 뱅크에는 절대 상한을 두지 않습니다.
pub fn split_bias_phrases_full(raw: &str) -> Vec<String> {
    let mut v: Vec<String> = split_phrase_list(raw);
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

// 🌟 [DATE SHAPE] 숫자-구분자 날짜 외에 '월 이름이 섞인 날짜' 까지 구조로 판정합니다.
//
//  ── 실측 사고 ──
//   복구 창 8 이 "DATE" ↔ "Apr-19-2022" 쌍을 정확히 읽어 왔는데
//   그 창이 물은 날짜 축 7개가 전부 형식 게이트에서 탈락해 배정이 0건이 되었습니다.
//   has_date_literal 은 세 번째 숫자 그룹을 요구하므로 "Apr-19-2022" 는
//   ["19","2022"] 두 그룹에서 멈추고 false 를 돌려줍니다.
//   인쇄 원문은 정규화 이전이므로, 저장 직전 normalize_trading_data 가 ISO 로 바꾸기 전에
//   이 게이트를 통과해야 합니다. 통과하지 못하면 정규화할 값 자체가 생기지 않습니다.
//
//  ── 왜 월 이름 사전을 만들지 않는가 ──
//   월 이름은 언어마다 다르고, 사전을 코드에 적으면 언어가 늘 때마다 이 함수를 고쳐야 합니다.
//   대신 '토큰 구조' 만 봅니다: 4자리 연도가 정확히 하나, 나머지는 1~31 범위의
//   1~2자리 숫자이거나 3~12자 순수 알파벳 한 덩어리.
//   어떤 라틴 문자 언어의 월 이름도 이 모양을 벗어나지 않고,
//   금액("2000.00")·식별자("CI-43726")·중량("20KG")은 이 모양에 들어올 수 없습니다.
pub fn normalize_digits_ascii(s: &str) -> String {
    s.chars()
        .map(|c| {
            let base = match c {
                '\u{0660}'..='\u{0669}' => Some('\u{0660}'),
                '\u{06F0}'..='\u{06F9}' => Some('\u{06F0}'),
                '\u{FF10}'..='\u{FF19}' => Some('\u{FF10}'),
                _ => None,
            };
            match base {
                Some(b) => char::from_u32('0' as u32 + (c as u32 - b as u32)).unwrap_or(c),
                None => c,
            }
        })
        .collect()
}

pub const MONTH_NAMES_ML: [&str; 12] = [
    "january, jan, januar, jänner, enero, ene, janvier, janv, gennaio, gen, janeiro, januari, leden, ledna, يناير, 一月",
    "february, feb, februar, febrero, février, févr, fév, febbraio, fevereiro, fev, februari, únor, února, فبراير, شباط, 二月",
    "march, mar, märz, marzo, mars, março, maart, březen, března, مارس, آذار, 三月",
    "april, apr, abril, abr, avril, avr, aprile, duben, dubna, أبريل, إبريل, نيسان, 四月",
    "may, mai, mayo, maggio, mag, maio, mei, květen, května, مايو, أيار, 五月",
    "june, jun, juni, junio, juin, giugno, giu, junho, červen, června, يونيو, حزيران, 六月",
    "july, jul, juli, julio, juillet, juil, luglio, lug, julho, červenec, července, يوليو, تموز, 七月",
    "august, aug, agosto, ago, août, aout, augustus, srpen, srpna, أغسطس, آب, 八月",
    "september, sep, sept, septiembre, septembre, settembre, set, setembro, září, سبتمبر, أيلول, 九月",
    "october, oct, oktober, okt, octubre, octobre, ottobre, ott, outubro, out, říjen, října, أكتوبر, 十月",
    "november, nov, noviembre, novembre, novembro, listopad, listopadu, نوفمبر, 十一月",
    "december, dec, dezember, dez, diciembre, dic, décembre, déc, dicembre, dezembro, prosinec, prosince, ديسمبر, 十二月",
];

pub fn lower_alnum(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

pub fn short_tail_ok(rest: &str, max_chars: usize) -> bool {
    !rest.is_empty()
        && rest.chars().count() <= max_chars
        && !rest.chars().any(|c| c.is_ascii_alphabetic())
}

pub fn month_from_name(core: &str) -> Option<u32> {
    let norm = lower_alnum(core);
    if norm.is_empty() { return None; }
    for (mi, raw) in MONTH_NAMES_ML.iter().enumerate() {
        for p in raw.split(',') {
            let p = lower_alnum(p);
            if p.is_empty() { continue; }
            if norm == p { return Some(mi as u32 + 1); }
            let cjk = p.chars().any(|c| (c as u32) >= 0x2E80);
            if p.chars().count() < 3 && !cjk { continue; }
            if let Some(rest) = norm.strip_prefix(p.as_str()) {
                if short_tail_ok(rest, 3) {
                    return Some(mi as u32 + 1);
                }
            }
        }
    }
    None
}

pub fn numeric_run_count(value: &str) -> usize {
    value
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == ','))
        .filter(|r| r.chars().any(|c| c.is_ascii_digit()))
        .count()
}

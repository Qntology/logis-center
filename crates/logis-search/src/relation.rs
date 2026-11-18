//! 🌟 [RELATION] 질의 속 고정점에서 연결 축을 따라, 질의가 묻는 유형의 문서로 건너가는 관계 단계 (커머스 검색)
//!
//! 해석기는 질의를 유형 · 조건 덩어리로 나누지만 '주문 X 의 운송장' · 'X 고객이 주문한 상품' · '반품된 주문의 상품' 처럼
//! 조건이 걸린 유형(고정점)과 질의가 묻는 유형(목표)이 다른 질의를 하나로 잇지 못했습니다. v2 관계형 합성 데이터에서 이런 질의
//! (C1 · C3 · C8 · C9)는 12개 언어 모두 쇼핑몰 문서 전체가 보이고 정답이 아래로 묻혔습니다 (MRR 0.02 ~ 0.12).
//!
//!  1. 목표 유형 — 질의의 유형 낱말(search_bridge.domain_words, 12개 언어 어간) 중 머리 자리. 한국어 · 일본어 · 중국어는
//!     명사구의 머리가 뒤에 오므로 마지막, 그 밖은 처음 ('주문 X 의 운송장 번호' · 'tracking numbers for order X' 모두 운송장).
//!  2. 고정점 — 저장 문서의 값이 질의에 그대로 나온 것: 자기 식별자(자기 유형 별칭 링크 매개변수 · no · doc_number),
//!     사람 이름, 제목. 값 → 질의 방향으로 찾으므로 질의 언어와 무관합니다. 상태는 계획의 상태 조건이나 상태 낱말
//!     (status_filters)로, 목표가 아닌 유형 낱말이 가리키는 문서에 겁니다 ('반품된 주문의 상품' → 반품 상태 주문).
//!  3. 연결 — 고정점 → 목표: 정방향 연결 칸(order.goods · review.goods · tracking.order · event.goods[]), 역방향 참조
//!     (tracking.order = 주문 index · order.event = 쿠폰 index), 고정점을 가리키는 문서를 거친 두 단계(주문 ← 운송장 → 상품).
//!     질의에 나온 유형의 고정점('리뷰를 남긴' → 리뷰, '주문한' → 주문)과 식별자 고정점이 먼저이고, 취소 · 환불 · 반품 문서를
//!     거친 경로는 뒤입니다. 질의가 고정점의 기간을 말하면('X 기간에 팔린 상품') 행사 · 쿠폰의 유효 기간 안의 주문을 거칩니다.
//!  4. 목표 문서를 결과 맨 앞으로 올리고(결과에 없으면 다시 읽어 넣음), 고정점이 이미 풀어낸 계획 조건 — 목표가 아닌 유형
//!     계획의 조건, 식별자 · 상태가 조건으로 풀린 것 — 을 뺍니다. 기간처럼 목표 유형에 걸린 조건은 그대로 둡니다.
//! 기간 조인(relay_join)이 적용된 질의는 건드리지 않습니다.

use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

use logis_bias::entity::normalize_entity_key;
use logis_schema::canonical::relay_type_family;

/// 질의가 물을 수 있는 커머스 유형
pub const TYPES: [&str; 6] = ["goods", "order", "tracking", "review", "coupon", "event"];

/// 사람 이름 칸
pub const PERSON_KEYS: [&str; 10] = [
    "name", "sender_name", "recipient_name", "receiver_name", "buyer_name", "orderer_name", "customer_name", "member_name",
    "author_name", "writer_name",
];

/// 유형 낱말: 유형 이름 + search_bridge.domain_words (12개 언어 어간)
pub fn type_vocabulary() -> Vec<(&'static str, Vec<String>)> {
    let words = logis_pug::parsing::BIAS_DICT.get("search_bridge").and_then(|b| b.get("domain_words"));
    TYPES
        .iter()
        .map(|t| {
            let mut terms = vec![t.to_string()];
            if let Some(list) = words.and_then(|w| w.get(*t)).and_then(|v| v.as_array()) {
                terms.extend(list.iter().filter_map(|x| x.as_str()).map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()));
            }
            (*t, terms)
        })
        .collect()
}

/// 낱말 사이를 띄지 않고 붙여 쓰는 문자(한중일)와 접두 관사가 붙는 아랍 문자: 어간이 낱말 중간에 와도 됩니다.
pub fn glued_script(c: char) -> bool {
    matches!(c as u32, 0x0600..=0x06FF | 0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7A3)
}

/// 라틴 문자 낱말 경계 판정용 (한국어 조사 '의' 가 라틴 이름 뒤에 붙어도 경계로 봅니다)
fn latin_alnum(c: char) -> bool {
    c.is_ascii_alphanumeric() || (c.is_alphabetic() && (c as u32) < 0x0250)
}

fn arabic_script(c: char) -> bool {
    matches!(c as u32, 0x0600..=0x06FF)
}

/// 아랍어 낱말에서 앞에 붙는 접속사(و · ف) · 전치사(ب · ل · ك) · 관사(ال, ل 뒤에서는 لل)를 뗀 꼴들 (원형 포함).
/// 'لبوليصة' → بوليصة, 'الفاتورة' → فاتورة, 'بالفاتورة' → فاتورة, 'للشحن' → شحن. 남는 글자가 두 자 미만이면 떼지 않습니다.
/// 아랍어 서식 이름 · 상태어는 사전에 관사 없는 꼴('فاتورة تجارية' · 'مرتجع')로 있어, 질의의 관사형('الفاتورة التجارية' ·
/// 'المرتجعة')이 글자 일치로는 맞지 않았습니다 (v2 질의 R3 · C9 아랍어).
pub fn arabic_bare_forms(word: &str) -> Vec<String> {
    let mut out = vec![word.to_string()];
    for conj in ["", "و", "ف"] {
        for prep in ["", "ب", "ل", "ك"] {
            for art in ["", "ال"] {
                let p = if prep == "ل" && art == "ال" { format!("{}لل", conj) } else { format!("{}{}{}", conj, prep, art) };
                if p.is_empty() {
                    continue;
                }
                if let Some(rest) = word.strip_prefix(p.as_str()) {
                    if rest.chars().count() >= 2 && !out.iter().any(|x| x == rest) {
                        out.push(rest.to_string());
                    }
                }
            }
        }
    }
    out
}

/// 아랍어 질의 낱말이 서식 이름 낱말과 같은가: 질의 쪽은 붙은 접두를 떼어 보고, 이름 쪽은 관사만 뗍니다.
fn arabic_word_matches(query_word: &str, title_word: &str) -> bool {
    let bare = title_word.strip_prefix("ال").filter(|r| r.chars().count() >= 2).unwrap_or(title_word);
    arabic_bare_forms(query_word).iter().any(|f| f == title_word || f == bare)
}

/// 질의의 유형 낱말 (글자 위치, 유형). 따옴표 안(이름)은 보지 않습니다.
pub fn type_mentions(query: &str) -> Vec<(usize, &'static str)> {
    let text: Vec<char> = logis_bias::query_lang::strip_quoted(query).to_lowercase().chars().collect();
    let mut out: Vec<(usize, &'static str)> = Vec::new();
    for (ty, terms) in type_vocabulary() {
        for term in terms.iter() {
            let tc: Vec<char> = term.chars().collect();
            if tc.is_empty() {
                continue;
            }
            let glued = tc.iter().any(|c| glued_script(*c));
            let mut i = 0;
            while i + tc.len() <= text.len() {
                if text[i..i + tc.len()] == tc[..] && (glued || i == 0 || !latin_alnum(text[i - 1])) {
                    out.push((i, ty));
                    i += tc.len();
                } else {
                    i += 1;
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// 질의가 묻는 유형: 명사구 머리가 뒤에 오는 언어(ko · ja · zh)는 마지막 유형 낱말, 그 밖은 처음 유형 낱말
pub fn target_type(mentions: &[(usize, &'static str)], lang: &str) -> Option<&'static str> {
    let head_final = matches!(lang, "ko" | "ja" | "zh-hans");
    let m = if head_final { mentions.last() } else { mentions.first() };
    m.map(|(_, t)| *t)
}

pub fn text_of(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(x)) => x.trim().to_string(),
        Some(Value::Number(n)) => n.to_string(),
        _ => String::new(),
    }
}

pub fn norm_text(t: &str) -> String {
    t.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

/// 이름 · 제목으로 쓸 만한 값: 글자가 있고, 한중일 문자면 두 글자 이상, 그 밖은 네 글자 이상
pub fn name_like(v: &str) -> bool {
    let n = v.chars().count();
    v.chars().any(|c| c.is_alphabetic()) && if v.chars().any(glued_script) { n >= 2 } else { n >= 4 }
}

/// 질의(정규화)에 값이 통째로 나오는가. 라틴 문자 값은 앞뒤가 라틴 낱말 글자가 아니어야 합니다 (Lukas ⊄ Lukasz).
pub fn contains_value(query_norm: &str, value: &str) -> bool {
    if value.is_empty() {
        return false;
    }
    let glued = value.chars().any(glued_script);
    let mut start = 0;
    while let Some(off) = query_norm[start..].find(value) {
        let a = start + off;
        let b = a + value.len();
        let before = query_norm[..a].chars().last();
        let after = query_norm[b..].chars().next();
        if glued || (before.map_or(true, |c| !latin_alnum(c)) && after.map_or(true, |c| !latin_alnum(c))) {
            return true;
        }
        start = b;
    }
    false
}

/// 링크의 자기 유형 별칭 매개변수 값, 원문 그대로 (od_id · invoice · it_id · cp_id · ev_id · rv_id · goodsNo · orderNo …)
pub fn link_self_token(link: &str, page_type: &str) -> Option<String> {
    let aliases = logis_schema::logic::relay_type_aliases(page_type);
    let (_, q) = link.split_once('?')?;
    for pair in q.split('&') {
        if let Some((name, val)) = pair.split_once('=') {
            let n: String = name.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_lowercase();
            if !val.trim().is_empty() && aliases.iter().any(|a| n == *a || (n.starts_with(*a) && (n.ends_with("id") || n.ends_with("no")))) {
                return Some(val.trim().to_string());
            }
        }
    }
    None
}

/// 🌟 [OWN NUMBER] 커머스 문서의 자기 번호(주문번호 · 운송장 번호 · 상품 · 쿠폰 · 이벤트 · 리뷰 번호)를 data.no 에 원문 그대로 남깁니다.
/// 주문번호는 link 의 od_id 에만 있고(data.id 는 해시, no 는 빈 값) 번호로 주문을 찾는 조건이 걸 칸이 없었습니다
/// (v2 질의 C1 '주문 X 의 운송장' 12개 언어 MRR 0.02). 링크의 자기 유형 별칭 매개변수 값을 먼저, 없으면 식별 씨앗(자동 키 제외)을 씁니다.
/// 이미 no 가 있으면(무역 서식의 doc_number, 주문 → 운송장 다리의 운송장 번호) 그대로 둡니다.
pub fn expose_own_number(item: &mut Value, page_type: &str, fallback_raw: &str) {
    let fam = relay_type_family(page_type);
    if !matches!(fam.as_str(), "goods" | "order" | "tracking" | "event" | "review") {
        return;
    }
    let Some(o) = item.as_object_mut() else { return };
    if !text_of(o.get("no")).is_empty() {
        return;
    }
    let link = text_of(o.get("link"));
    let token = link_self_token(&link, page_type).or_else(|| {
        let f = fallback_raw.trim();
        (!f.is_empty() && !f.starts_with("AUTO-")).then(|| f.to_string())
    });
    if let Some(t) = token {
        o.insert("no".to_string(), json!(t));
    }
}

/// 문서의 자기 식별자 (정규화): 자기 유형 별칭 링크 매개변수 값(od_id · invoice · it_id · cp_id · ev_id · rv_id · goodsNo …),
/// no · doc_number, 운송장의 tracking_number, 쿠폰의 code. (주문이 들고 있는 운송장 번호처럼 남을 가리키는 값은 빼고)
pub fn own_ids(doc: &Value) -> Vec<String> {
    let ty = text_of(doc.get("type"));
    let fam = relay_type_family(&ty);
    let aliases = logis_schema::logic::relay_type_aliases(&ty);
    let mut out: Vec<String> = Vec::new();
    let mut push = |raw: &str| {
        let k = normalize_entity_key(raw);
        if k.chars().count() >= 3 && !out.contains(&k) {
            out.push(k);
        }
    };
    for k in ["no", "doc_number"] {
        let v = text_of(doc.get(k));
        if !v.is_empty() {
            push(&v);
        }
    }
    if fam == "tracking" {
        let v = text_of(doc.get("tracking_number"));
        if !v.is_empty() {
            push(&v);
        }
    }
    if ty == "coupon" {
        let v = text_of(doc.get("code"));
        if !v.is_empty() {
            push(&v);
        }
    }
    let link = text_of(doc.get("link"));
    if let Some((_, q)) = link.split_once('?') {
        for pair in q.split('&') {
            if let Some((name, val)) = pair.split_once('=') {
                let n: String = name.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_lowercase();
                if aliases.iter().any(|a| n == *a || (n.starts_with(*a) && (n.ends_with("id") || n.ends_with("no")))) {
                    push(val);
                }
            }
        }
    }
    out
}

/// 질의의 식별자 후보 (원문, 정규화): 숫자가 셋 이상 든 네 글자 이상 영숫자 토큰 ('20260413-302272의' → 20260413-302272)
pub fn query_id_tokens(query: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for raw in query.split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '/' | '.'))) {
        let t = raw.trim_matches(|c: char| matches!(c, '-' | '_' | '/' | '.'));
        if t.chars().count() >= 4 && t.chars().filter(|c| c.is_ascii_digit()).count() >= 3 {
            let n = normalize_entity_key(t);
            if !out.iter().any(|(_, x)| *x == n) {
                out.push((t.to_string(), n));
            }
        }
    }
    out
}

/// 질의의 상태 낱말 → 상태 코드 (status_filters 확정어 · 닫힌 꼬리 · 접두 어간; 아랍어는 붙은 관사 · 전치사를 뗀 꼴도)
pub fn query_status_codes(query: &str) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::new();
    for w in logis_bias::query_lang::strip_quoted(query).split_whitespace() {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric());
        if w.is_empty() {
            continue;
        }
        let forms = if w.chars().any(arabic_script) { arabic_bare_forms(w) } else { vec![w.to_string()] };
        let key = forms.iter().find_map(|f| {
            logis_pug::utils::ai_utils::exact_match_filter_key_tailed("status_filters", f)
                .or_else(|| logis_pug::utils::ai_utils::prefix_match_filter_stem("status_filters", f).map(|(k, _)| k))
        });
        if let Some(k) = key {
            let code = logis_schema::logic::parse_status(&k) as i64;
            if code != 0 && !out.contains(&code) {
                out.push(code);
            }
        }
    }
    out
}

pub fn status_code(doc: &Value) -> i64 {
    match doc.get("status") {
        Some(Value::Number(n)) => n.as_i64().unwrap_or(0),
        Some(Value::String(s)) => logis_schema::logic::parse_status(s.trim()) as i64,
        _ => 0,
    }
}

/// 취소 · 환불 · 반품 상태 (relay_ledger 의 기간 조인과 같은 규칙: 이런 거래는 '팔린 · 주문한' 것이 아닙니다)
pub fn void_status(code: i64) -> bool {
    code != 0 && ["cancel", "refund", "return"].iter().any(|s| logis_schema::logic::parse_status(s) as i64 == code)
}

/// 낱말 목록(search_bridge.<key>) 중 하나가 질의(따옴표 밖)에 나오는가. 한중일 · 한글은 글자 일치, 아랍어는 붙은 접두를
/// 뗀 낱말 일치, 그 밖은 낱말 첫머리부터 일치 (여러 낱말 구 'v průběhu' 도 그대로).
fn query_has_bridge_word(query: &str, key: &str) -> bool {
    let words: Vec<String> = logis_pug::parsing::BIAS_DICT
        .get("search_bridge")
        .and_then(|b| b.get(key))
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str()).map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default();
    let text = logis_bias::query_lang::strip_quoted(query).to_lowercase();
    let arabic_words: Vec<Vec<String>> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.chars().any(arabic_script))
        .map(arabic_bare_forms)
        .collect();
    words.iter().any(|w| {
        if w.chars().any(arabic_script) {
            return arabic_words.iter().any(|forms| forms.iter().any(|f| f == w));
        }
        if w.chars().any(glued_script) {
            return text.contains(w.as_str());
        }
        text.match_indices(w.as_str()).any(|(a, _)| text[..a].chars().last().map_or(true, |c| !latin_alnum(c)))
    })
}

/// 질의가 고정점의 기간을 말하는가 ('기간에' · during · während · pendant · durante · tijdens · během · خلال …, search_bridge.period_words)
pub fn mentions_period(query: &str) -> bool {
    query_has_bridge_word(query, "period_words")
}

fn date_of(v: Option<&Value>) -> Option<chrono::NaiveDate> {
    match v {
        Some(Value::String(t)) => chrono::NaiveDate::parse_from_str(&t.chars().take(10).collect::<String>(), "%Y-%m-%d").ok(),
        Some(Value::Number(n)) => n.as_i64().filter(|ms| *ms > 100_000_000_000).and_then(chrono::DateTime::from_timestamp_millis).map(|d| d.date_naive()),
        _ => None,
    }
}

/// 문서의 유효 기간 (행사 · 쿠폰의 started_at ~ expired_at)
pub fn validity_period(doc: &Value) -> Option<(chrono::NaiveDate, chrono::NaiveDate)> {
    let a = date_of(doc.get("started_at"))?;
    let b = date_of(doc.get("expired_at"))?;
    (a <= b).then_some((a, b))
}

pub fn cond_value_key(v: &Value) -> String {
    match v {
        Value::String(s) => normalize_entity_key(s),
        Value::Number(n) => normalize_entity_key(&n.to_string()),
        _ => String::new(),
    }
}

// ──────────────────────────────────────────────
// 무역 서식
// ──────────────────────────────────────────────

/// 서식 전체를 뜻하는 일반 이름 (documents · 서류 · 書類 · 单据 · Unterlagen …, search_bridge.document_words)
fn document_words() -> Vec<String> {
    logis_pug::parsing::BIAS_DICT
        .get("search_bridge")
        .and_then(|b| b.get("document_words"))
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str()).map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default()
}

/// 라틴 문자 낱말 한 쌍이 같은 낱말인가: 짧은 낱말은 완전 일치, 긴 낱말은 앞 글자 max(4, 길이 − 2)자 일치
/// (bills ↔ bill · polizze ↔ polizza · comerciais ↔ comercial · Konnossemente ↔ Konnossement)
fn word_matches(query_word: &str, title_word: &str) -> bool {
    let n = title_word.chars().count();
    if n <= 3 {
        return query_word == title_word;
    }
    let need = 4.max(n.saturating_sub(2));
    query_word.chars().zip(title_word.chars()).take_while(|(a, b)| a == b).count() >= need
}

/// 사전에 없는 서식 이름 복수형 (search_bridge.trade_title_plurals: 코드 → 이름들). 아랍어 서식 이름은 복수가 불규칙해
/// ('بوليصة' → 'بوالص' · 'فاتورة' → 'فواتير') 앞 글자 일치로 이을 수 없습니다 (v2 질의 R2 · R4 아랍어).
fn trade_title_plurals() -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    if let Some(m) = logis_pug::parsing::BIAS_DICT.get("search_bridge").and_then(|b| b.get("trade_title_plurals")).and_then(|v| v.as_object()) {
        for (code, list) in m {
            for t in list.as_array().into_iter().flatten().filter_map(|x| x.as_str()) {
                out.push((code.clone(), t.to_string()));
            }
        }
    }
    out
}

/// 질의의 서식 언급 (낱말 위치, 서식 코드들 — 일반 이름은 "*"). 따옴표 안과 식별자 토큰(BL-GB85095836)은 보지 않습니다.
/// 서식 이름은 12개 언어 서식 전문표(all_trade_doc_titles)와 복수형 표에서, 한중일 문자는 글자 일치, 아랍어는 붙은 접두를 뗀
/// 낱말 일치, 그 밖은 낱말 단위 앞 글자 일치입니다.
pub fn trade_mentions(query: &str) -> Vec<(usize, Vec<String>)> {
    let mut text = logis_bias::query_lang::strip_quoted(query).to_lowercase();
    for (raw, _) in query_id_tokens(query) {
        text = text.replace(&raw.to_lowercase(), " ");
    }
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !(c.is_alphanumeric() || c == '/')).to_string())
        .filter(|w| !w.is_empty())
        .collect();
    let mut titles: Vec<(String, Vec<String>)> = Vec::new();
    for (code, title) in logis_pug::utils::ai_utils::all_trade_doc_titles().into_iter().chain(trade_title_plurals()) {
        let t = title.trim().to_lowercase();
        if t.chars().count() < 2 {
            continue;
        }
        match titles.iter_mut().find(|(x, _)| *x == t) {
            Some((_, codes)) => {
                if !codes.contains(&code) {
                    codes.push(code);
                }
            }
            None => titles.push((t, vec![code])),
        }
    }
    for w in document_words() {
        titles.push((w, vec!["*".to_string()]));
    }
    let mut out: Vec<(usize, usize, Vec<String>)> = Vec::new();
    // 낱말 i 가 붙은 글 전체에서 시작하는 글자 위치 (띄어 쓴 한중일 서식 이름 '커머셜 인보이스' · '마스터 선하증권' 용)
    let joined = words.join(" ");
    let mut word_at: Vec<usize> = Vec::with_capacity(words.len());
    let mut at = 0;
    for w in words.iter() {
        word_at.push(at);
        at += w.len() + 1;
    }
    for (title, codes) in titles.iter() {
        if title.chars().any(arabic_script) {
            let tw: Vec<&str> = title.split_whitespace().collect();
            if tw.is_empty() || tw.len() > words.len() {
                continue;
            }
            for i in 0..=(words.len() - tw.len()) {
                if tw.iter().enumerate().all(|(k, t)| arabic_word_matches(&words[i + k], t)) {
                    out.push((i, title.chars().count(), codes.clone()));
                }
            }
            continue;
        }
        if title.chars().any(glued_script) {
            if title.contains(' ') {
                let mut start = 0;
                while let Some(off) = joined[start..].find(title.as_str()) {
                    let a = start + off;
                    let i = word_at.iter().rposition(|&p| p <= a).unwrap_or(0);
                    out.push((i, title.chars().count(), codes.clone()));
                    start = a + title.len();
                }
                continue;
            }
            for (i, w) in words.iter().enumerate() {
                if w.contains(title.as_str()) {
                    out.push((i, title.chars().count(), codes.clone()));
                }
            }
            continue;
        }
        let tw: Vec<&str> = title.split_whitespace().collect();
        if tw.is_empty() || tw.len() > words.len() {
            continue;
        }
        for i in 0..=(words.len() - tw.len()) {
            if tw.iter().enumerate().all(|(k, t)| word_matches(&words[i + k], t)) {
                out.push((i, title.chars().count(), codes.clone()));
            }
        }
    }
    // 같은 자리에서는 긴 이름을 (bill of lading > bill), 같은 길이면 일반 이름(documents)을 남기고 나머지는 버립니다.
    let star = |codes: &Vec<String>| codes.iter().any(|c| c == "*");
    out.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(star(&b.2).cmp(&star(&a.2))));
    let mut kept: Vec<(usize, Vec<String>)> = Vec::new();
    let mut last_pos: Option<usize> = None;
    for (pos, _, codes) in out.into_iter() {
        if last_pos == Some(pos) {
            continue;
        }
        last_pos = Some(pos);
        kept.push((pos, codes));
    }
    kept
}

/// 서식 질의의 목표 서식 코드 ("*" = 서식 전체): 커머스와 같은 머리 규칙
pub fn trade_target(query: &str, lang: &str) -> Option<Vec<String>> {
    let m = trade_mentions(query);
    let head_final = matches!(lang, "ko" | "ja" | "zh-hans");
    let pick = if head_final { m.last() } else { m.first() };
    match pick {
        Some((_, codes)) => Some(codes.clone()),
        // 서식 이름을 못 찾았어도 서식 번호가 있으면 그 번호에 걸린 서식 전체를 묻는 질의로 봅니다 (복수형이 불규칙한 아랍어 'بوالص الشحن' 등)
        None if !query_id_tokens(query).is_empty() => Some(vec!["*".to_string()]),
        None => None,
    }
}

pub fn upper_codes(codes: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in codes {
        for v in [c.to_uppercase(), c.to_lowercase()] {
            if !out.contains(&v) {
                out.push(v);
            }
        }
    }
    out
}

/// 서식 질의 전체 유형 목록 (봉투 유형 + 서식 전문표의 모든 코드, 대 · 소문자)
pub fn all_trade_types() -> Vec<String> {
    let mut codes: Vec<String> = vec!["tracking".into(), "TRACKING".into(), "receiving".into(), "Receiving".into(), "shipping".into(), "Shipping".into(), "shipping_doc".into()];
    let mut seen: Vec<String> = Vec::new();
    for (code, _) in logis_pug::utils::ai_utils::all_trade_doc_titles() {
        if !seen.contains(&code) {
            seen.push(code);
        }
    }
    codes.extend(upper_codes(&seen));
    codes
}

/// 🌟 [TRADE RELATION] 서식 질의: 목표 서식(머리 자리의 서식 이름, 일반 이름이면 전체)과 질의의 서식 번호에서
/// 번호를 자기 번호로 가진 서식(고정점) · 그 번호를 reference_* 칸에 적은 서식 · rel_* 연결로 이어진 서식을 목표로 모읍니다.
/// v2 질의 R2('PO X 의 B/L') · R4('L/C X 로 결제한 상업송장')는 번호 앞 글자(PO · LC)가 유형을 그 서식으로 좁혀, 저장소에 없는
/// 발주서 · 신용장만 찾다가 0건이었고(8 · 5개 언어), R3 일본어는 B/L 로 좁혀 상업송장을 놓쳤습니다.
/// 머리가 일반 이름(서류 · documents)이면 해석기가 좁힌 유형을 서식 전체로 되돌립니다 (R7 한국어: '선적된' → SR · ED · SA).
/// 음역 고정점의 자음 골격 일치도 하한. 질의 396건 탐침에서 참 음역(Felixstowe 넷)은 1.00, 무역 질의의 다른 조각 최고는 0.80 입니다.
pub const PHONETIC_ANCHOR_PASS: f32 = 0.90;
/// 음역 고정점에 쓰는 자음 골격 최소 길이 (조각 · 값 둘 다). 골격 2 ~ 3개는 우연히 같기 쉽습니다 ('운송' · '배송' ↔ Incheon · 부산 = 1.00).
pub const PHONETIC_ANCHOR_MIN_SKELETON: usize = 4;

/// 음역을 찾을 문자 종류 (라틴 · 숫자 · 기호는 None)
fn phonetic_script(c: char) -> Option<u8> {
    let u = c as u32;
    match u {
        0xAC00..=0xD7A3 => Some(1),
        0x30A0..=0x30FF => Some(2),
        0x3040..=0x309F => Some(3),
        0x3400..=0x9FFF | 0xF900..=0xFAFF => Some(4),
        0x0600..=0x06FF => Some(5),
        0x0400..=0x04FF => Some(6),
        _ if c.is_alphabetic() && u >= 0x0250 => Some(7),
        _ => None,
    }
}

/// 음역 고정점에 쓸 질의 조각: 라틴 문자가 아닌 글자 덩어리(문자 종류가 바뀌는 곳에서 끊음)의 2 ~ 10 글자 부분.
/// 붙여 쓴 질의('6月にフェリクストウから')와 조사가 붙은 어절('펠릭스토우에서')에서도 이름 조각이 나옵니다.
pub fn phonetic_spans(query: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for tok in query.split_whitespace() {
        let chars: Vec<char> = tok.chars().collect();
        let mut i = 0usize;
        while i < chars.len() {
            let Some(k) = phonetic_script(chars[i]) else {
                i += 1;
                continue;
            };
            let mut j = i;
            while j < chars.len() && phonetic_script(chars[j]) == Some(k) {
                j += 1;
            }
            let run = &chars[i..j];
            for a in 0..run.len() {
                for b in (a + 2)..=(run.len().min(a + 10)) {
                    let s: String = run[a..b].iter().collect();
                    if seen.insert(s.clone()) {
                        out.push(s);
                    }
                }
            }
            i = j;
        }
    }
    out
}

/// 라틴 문자가 글자의 과반인가
fn latin_dominant(v: &str) -> bool {
    let (mut latin, mut other) = (0usize, 0usize);
    for c in v.chars().filter(|c| c.is_alphabetic()) {
        if (c as u32) < 0x0250 {
            latin += 1;
        } else {
            other += 1;
        }
    }
    latin > other
}

/// 🌟 [PHONETIC ANCHOR] 라틴 문자 값(항구 · 당사자 · 선박 이름 첫 마디)이 질의 조각에 음역되어 나오는가: (조각, 자음 골격 일치도).
/// 값 · 조각 모두 골격이 [`PHONETIC_ANCHOR_MIN_SKELETON`] 개 이상이고 일치도(nl_convert::phonetic_similarity)가 [`PHONETIC_ANCHOR_PASS`] 이상인
/// 조각 중 가장 높은 것.
pub fn phonetic_value_hit(spans: &[String], value: &str) -> Option<(String, f32)> {
    if !latin_dominant(value) || logis_pug::nl_convert::phonetic_skeleton_len(value) < PHONETIC_ANCHOR_MIN_SKELETON {
        return None;
    }
    let mut best: Option<(String, f32)> = None;
    for s in spans.iter() {
        if logis_pug::nl_convert::phonetic_skeleton_len(s) < PHONETIC_ANCHOR_MIN_SKELETON {
            continue;
        }
        if let Some(sim) = logis_pug::nl_convert::phonetic_similarity(s, value) {
            if sim >= PHONETIC_ANCHOR_PASS && best.as_ref().map_or(true, |(_, b)| sim > *b) {
                best = Some((s.clone(), sim));
            }
        }
    }
    best
}

/// 무역 서식의 품목 이름들 (정규화): description · description_of_goods 를 ' / ' 로 나눈 것과 품목 행(items · line_items)의 설명.
/// 끝의 품목 번호 괄호('Handheld mini fan (KR20287)')는 뗍니다.
pub fn trade_item_names(doc: &Value) -> Vec<String> {
    let mut raw: Vec<String> = Vec::new();
    for k in ["description", "description_of_goods"] {
        raw.extend(text_of(doc.get(k)).split(" / ").map(String::from));
    }
    for arr in ["items", "line_items"] {
        for it in doc.get(arr).and_then(|v| v.as_array()).into_iter().flatten() {
            for k in ["description", "description_of_goods", "item_name"] {
                raw.push(text_of(it.get(k)));
            }
        }
    }
    let mut out: Vec<String> = Vec::new();
    for r in raw {
        let base = match (r.rfind('('), r.ends_with(')')) {
            (Some(p), true) => r[..p].trim().to_string(),
            _ => r.trim().to_string(),
        };
        let n = norm_text(&base);
        if !n.is_empty() && !out.contains(&n) {
            out.push(n);
        }
    }
    out
}


// =====================================================================
// 🌟 [CROSS-LINGUAL ANCHOR] 질의 언어로 번역된 제목 고정점
// ---------------------------------------------------------------------
//  값 고정점은 저장 값이 질의에 글자 그대로 나와야 했습니다. 상품 이름이 질의 언어로 번역되면('자외선 차단 양산' ·
//  'UVカット日傘' ↔ 'UV-Schutz-Sonnenschirm') 닿지 않았습니다 (v2 질의 A3 MRR 0.20 · C4 0.41).
//  글자 고정점이 아무것도 잇지 못했을 때만, 질의 조각(낱말 n-그램 · 붙여 쓰는 문자의 글자 n-그램)과 제목을 다국어 임베딩으로
//  비교합니다 (보정 코사인, models::embedding::calibrate_cos).
//   · 조각이 구조 낱말(유형 · 상태 · 행동 · 페이지 · 기간 · 동사 — bias 사전, [`structure_bank`])보다 제목에 더 가까울 때만 씁니다.
//     '장바구니 담기 기록' 같은 조각이 행동 버튼 글자가 비슷한 상품 제목으로 가지 않게.
//   · 범위(사이트)마다 1위 제목이 2위를 [`CROSS_ANCHOR_MARGIN`] 앞서고, 그 조각이 구조 낱말보다 [`CROSS_ANCHOR_BANK_GAP`] 앞서야 합니다.
//   · 범위가 여럿이면(사이트를 말하지 않은 행동 로그 질의) 전체 1위에서 [`CROSS_ANCHOR_BAND`] 안의 범위 1위만 받습니다. 사이트마다
//     따로 받으면 그 상품이 없는 쇼핑몰의 비슷한 상품('Mineral sunscreen')을 12개 언어 중 6개에서 받았습니다.
//  탐침(같은 임베딩 · 같은 규칙, 이번 브랜치 SyntheticData 세계): 평가 A3 12/12 맞음 · 0 틀림, 합성 A3 288건 맞음 155 · 틀림 13,
//  다른 행동 로그 질의 60건 틀림 0, 한 사이트 범위 C4 평가 22건 중 4건 맞음 · 0 틀림. 무역 품목(R6)은 규칙을 넘는 사례가 없어 쓰지 않습니다.
// =====================================================================

/// 범위 안 1위 제목이 2위를 앞서야 하는 보정 코사인 차
pub const CROSS_ANCHOR_MARGIN: f32 = 0.05;
/// 1위 제목을 고른 조각이 구조 낱말 은행보다 앞서야 하는 보정 코사인 차
pub const CROSS_ANCHOR_BANK_GAP: f32 = 0.05;
/// 범위가 여럿일 때 받는 범위 1위들의 폭 (전체 1위에서)
pub const CROSS_ANCHOR_BAND: f32 = 0.03;
/// 낱말 조각 최대 길이 (낱말 수)
const CROSS_SPAN_MAX_WORDS: usize = 5;
/// 붙여 쓰는 문자 조각 최대 길이 (글자 수)
const CROSS_SPAN_MAX_CHARS: usize = 10;

/// 낱말 경계를 공백으로 알 수 없는 문자 (가나 · 한자 · 타이 문자). 한글은 띄어 쓰므로 낱말 조각으로 봅니다.
fn cross_glued(c: char) -> bool {
    matches!(c as u32, 0x3040..=0x30FF | 0x3400..=0x9FFF | 0xF900..=0xFAFF | 0x0E00..=0x0E7F)
}

/// 값을 찾을 질의 조각: 앞뒤 기호를 뗀 낱말의 1 ~ 5-그램, 붙여 쓰는 문자 낱말(3글자 이상)은 2글자 ~ (길이 − 1, 최대 10) 글자 조각.
pub fn query_value_spans(query: &str) -> Vec<String> {
    let toks: Vec<String> = query
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|t| !t.is_empty())
        .collect();
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut add = |s: String| {
        if !s.is_empty() && seen.insert(s.clone()) {
            out.push(s);
        }
    };
    for n in 1..=CROSS_SPAN_MAX_WORDS.min(toks.len()) {
        for i in 0..=toks.len() - n {
            add(toks[i..i + n].join(" "));
        }
    }
    for t in toks.iter() {
        let chars: Vec<char> = t.chars().collect();
        if chars.len() >= 3 && chars.iter().any(|c| cross_glued(*c)) {
            for n in 2..=(chars.len() - 1).min(CROSS_SPAN_MAX_CHARS) {
                for i in 0..=chars.len() - n {
                    add(chars[i..i + n].iter().collect());
                }
            }
        }
    }
    out
}

/// 구조 낱말 은행: 값이 아닌 낱말들 — 유형 · 서식 · 기간 낱말(search_bridge), 행동 · 페이지 · 상태 · 기간 · 계절 · 실질 · 찾기 확정어,
/// 행동 어간, 동사 · 표현 구 (bias 사전, 12개 언어). 조각이 이것보다 제목에 가까워야 제목 조각으로 봅니다.
pub fn structure_bank() -> &'static [String] {
    static BANK: once_cell::sync::Lazy<Vec<String>> = once_cell::sync::Lazy::new(|| {
        let b = &*logis_pug::parsing::BIAS_DICT;
        let mut out: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut add = |p: &str| {
            let p = p.split_whitespace().collect::<Vec<_>>().join(" ");
            if !p.is_empty() && seen.insert(p.clone()) {
                out.push(p);
            }
        };
        let strs = |v: Option<&Value>| -> Vec<String> {
            v.and_then(|x| x.as_array()).into_iter().flatten().filter_map(|x| x.as_str().map(String::from)).collect()
        };
        let sb = b.get("search_bridge");
        if let Some(o) = sb.and_then(|s| s.get("domain_words")).and_then(|v| v.as_object()) {
            for (ty, ws) in o {
                add(ty);
                strs(Some(ws)).iter().for_each(|w| add(w));
            }
        }
        for key in ["document_words", "period_words"] {
            strs(sb.and_then(|s| s.get(key))).iter().for_each(|w| add(w));
        }
        if let Some(o) = sb.and_then(|s| s.get("trade_title_plurals")).and_then(|v| v.as_object()) {
            for ws in o.values() {
                match ws {
                    Value::Array(_) => strs(Some(ws)).iter().for_each(|w| add(w)),
                    Value::Object(m) => {
                        for v in m.values() {
                            match v {
                                Value::Array(_) => strs(Some(v)).iter().for_each(|w| add(w)),
                                Value::String(s) => add(s),
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        strs(sb.and_then(|s| s.get("sales_to_order")).and_then(|s| s.get("trigger_words"))).iter().for_each(|w| add(w));
        for cat in [
            "analytic_event_filters",
            "analytic_page_filters",
            "status_filters",
            "time_filters",
            "season_filters",
            "substantial_filters",
            "find_filters",
        ] {
            for ent in b.get(cat).and_then(|v| v.as_object()).into_iter().flat_map(|o| o.values()) {
                strs(ent.get("exact_match")).iter().for_each(|w| add(w));
            }
        }
        for ent in b.get("analytic_action_filters").and_then(|v| v.as_object()).into_iter().flat_map(|o| o.values()) {
            strs(ent.get("stems")).iter().for_each(|w| add(w));
        }
        for cat in ["verb", "expression"] {
            for ent in b.get(cat).and_then(|v| v.as_object()).into_iter().flat_map(|o| o.values()) {
                for s in ent.as_object().into_iter().flat_map(|o| o.values()).filter_map(|v| v.as_str()) {
                    s.split(',').for_each(|p| add(p));
                }
            }
        }
        out
    });
    &BANK
}

/// 받아들인 번역 제목 고정점: (범위, 후보 위치, 보정 코사인, 은행 여유, 범위 안 2위와의 차, 조각 위치)
#[derive(Debug, Clone, PartialEq)]
pub struct CrossAnchor {
    pub group: String,
    pub title: usize,
    pub cos: f32,
    pub bank_gap: f32,
    pub margin: f32,
    pub span: usize,
}

/// 조각 × 제목 보정 코사인(`sim[조각][제목]`)과 조각의 구조 낱말 최고 보정 코사인(`span_bank`)에서 받아들일 제목 고정점을 고릅니다.
/// `groups[제목]` 은 제목의 범위(사이트). 위 [CROSS-LINGUAL ANCHOR] 규칙 그대로입니다.
pub fn cross_anchor_pick(sim: &[Vec<f32>], span_bank: &[f32], groups: &[String]) -> Vec<CrossAnchor> {
    let mut names: Vec<&str> = groups.iter().map(|g| g.as_str()).collect();
    names.sort();
    names.dedup();
    let mut per: Vec<CrossAnchor> = Vec::new();
    for g in names {
        let cols: Vec<usize> = (0..groups.len()).filter(|&t| groups[t] == g).collect();
        if cols.len() < 2 {
            continue;
        }
        let elig: Vec<bool> = sim
            .iter()
            .zip(span_bank)
            .map(|(row, bank)| cols.iter().map(|&t| row[t]).fold(f32::MIN, f32::max) > *bank)
            .collect();
        if !elig.iter().any(|e| *e) {
            continue;
        }
        let best: Vec<(f32, usize)> = cols
            .iter()
            .map(|&t| (0..sim.len()).filter(|&s| elig[s]).fold((-1.0f32, 0usize), |a, s| if sim[s][t] > a.0 { (sim[s][t], s) } else { a }))
            .collect();
        let mut order: Vec<usize> = (0..cols.len()).collect();
        order.sort_by(|&a, &b| best[b].0.partial_cmp(&best[a].0).unwrap_or(std::cmp::Ordering::Equal));
        let (cos, span) = best[order[0]];
        per.push(CrossAnchor {
            group: g.to_string(),
            title: cols[order[0]],
            cos,
            bank_gap: cos - span_bank[span],
            margin: cos - best[order[1]].0.max(0.0),
            span,
        });
    }
    let b1 = per.iter().map(|a| a.cos).fold(f32::MIN, f32::max);
    per.into_iter()
        .filter(|a| a.cos >= b1 - CROSS_ANCHOR_BAND && a.bank_gap > CROSS_ANCHOR_BANK_GAP && a.margin > CROSS_ANCHOR_MARGIN)
        .collect()
}

/// 단위 벡터로 (0 벡터는 그대로)
pub fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 { v.iter().map(|x| x / n).collect() } else { v.to_vec() }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        s += a[i] * b[i];
    }
    s
}


/// 질의의 절대 기간 (오늘 기준; 'in September' · '9월' · 'from X to Y')
pub fn query_period(query: &str) -> Option<(chrono::NaiveDate, chrono::NaiveDate)> {
    let words: Vec<String> = logis_bias::query_lang::strip_quoted(query).split_whitespace().map(String::from).collect();
    let today = logis_bias::time_guide::now_local().date_naive();
    logis_pug::utils::ai_utils::exact_absolute_period(&words, today).map(|p| (p.start, p.end))
}

/// 문서의 날짜 칸들 (등록 · 주문 · 발송 · 배송 완료 · 발행일; 문자열 YYYY-MM-DD… 또는 epoch ms)
pub fn doc_dates(doc: &Value) -> Vec<chrono::NaiveDate> {
    ["registration_date", "order_date", "payment_date", "shipping_date", "shipped_at", "delivered_at", "issue_date"]
        .iter()
        .filter_map(|k| date_of(doc.get(*k)))
        .collect()
}

/// 목표 순서: 단계 → 질의 기간 안의 문서 먼저 → 원래 회수 순위 (목표끼리는 백엔드가 매긴 질의 일치 순서를 지킵니다;
/// v2 질의 C10 '9월에 X 고객에게 배송된 운송장' 은 고객의 운송장 전체가 임의 순서로 올라와 9월 것이 5위로 밀렸습니다)
pub fn order_targets(tiered: Vec<(usize, usize)>, docs: &[(String, Value)], results: &[Value], period: Option<(chrono::NaiveDate, chrono::NaiveDate)>) -> Vec<usize> {
    let pos: HashMap<&str, usize> =
        results.iter().enumerate().filter_map(|(i, r)| r.get("id").and_then(|v| v.as_str()).map(|s| (s, i))).collect();
    let mut best: HashMap<usize, usize> = HashMap::new();
    for (t, j) in tiered {
        let e = best.entry(j).or_insert(t);
        if t < *e {
            *e = t;
        }
    }
    let mut keyed: Vec<(usize, bool, usize, usize)> = best
        .into_iter()
        .map(|(j, t)| {
            let outside = period.map_or(false, |(a, b)| !doc_dates(&docs[j].1).iter().any(|d| *d >= a && *d <= b));
            (t, outside, pos.get(docs[j].0.as_str()).copied().unwrap_or(usize::MAX), j)
        })
        .collect();
    keyed.sort();
    keyed.into_iter().map(|(_, _, _, j)| j).collect()
}

#[derive(Debug, Default, Clone)]
pub struct RelationReport {
    pub target_type: String,
    pub anchors: usize,
    pub targets: usize,
    pub rescued: usize,
    pub consumed: usize,
}



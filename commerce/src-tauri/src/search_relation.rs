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

use crate::scheduler::normalize_entity_key;
use crate::store::VectorStore;
use crate::utils::canonical::{is_relay_placeholder, relay_edges, relay_key_for_type, relay_type_family};

/// 질의가 물을 수 있는 커머스 유형
pub const TYPES: [&str; 6] = ["goods", "order", "tracking", "review", "coupon", "event"];

/// 사람 이름 칸
const PERSON_KEYS: [&str; 10] = [
    "name", "sender_name", "recipient_name", "receiver_name", "buyer_name", "orderer_name", "customer_name", "member_name",
    "author_name", "writer_name",
];

/// 유형 낱말: 유형 이름 + search_bridge.domain_words (12개 언어 어간)
pub fn type_vocabulary() -> Vec<(&'static str, Vec<String>)> {
    let words = crate::parsing::BIAS_DICT.get("search_bridge").and_then(|b| b.get("domain_words"));
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
fn glued_script(c: char) -> bool {
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
    let text: Vec<char> = crate::stanza::strip_quoted(query).to_lowercase().chars().collect();
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

fn text_of(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(x)) => x.trim().to_string(),
        Some(Value::Number(n)) => n.to_string(),
        _ => String::new(),
    }
}

fn norm_text(t: &str) -> String {
    t.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

/// 이름 · 제목으로 쓸 만한 값: 글자가 있고, 한중일 문자면 두 글자 이상, 그 밖은 네 글자 이상
fn name_like(v: &str) -> bool {
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
    let aliases = crate::logic::relay_type_aliases(page_type);
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
    let aliases = crate::logic::relay_type_aliases(&ty);
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
    for w in crate::stanza::strip_quoted(query).split_whitespace() {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric());
        if w.is_empty() {
            continue;
        }
        let forms = if w.chars().any(arabic_script) { arabic_bare_forms(w) } else { vec![w.to_string()] };
        let key = forms.iter().find_map(|f| {
            crate::utils::ai_utils::exact_match_filter_key_tailed("status_filters", f)
                .or_else(|| crate::utils::ai_utils::prefix_match_filter_stem("status_filters", f).map(|(k, _)| k))
        });
        if let Some(k) = key {
            let code = crate::logic::parse_status(&k) as i64;
            if code != 0 && !out.contains(&code) {
                out.push(code);
            }
        }
    }
    out
}

fn status_code(doc: &Value) -> i64 {
    match doc.get("status") {
        Some(Value::Number(n)) => n.as_i64().unwrap_or(0),
        Some(Value::String(s)) => crate::logic::parse_status(s.trim()) as i64,
        _ => 0,
    }
}

/// 취소 · 환불 · 반품 상태 (relay_ledger 의 기간 조인과 같은 규칙: 이런 거래는 '팔린 · 주문한' 것이 아닙니다)
fn void_status(code: i64) -> bool {
    code != 0 && ["cancel", "refund", "return"].iter().any(|s| crate::logic::parse_status(s) as i64 == code)
}

/// 낱말 목록(search_bridge.<key>) 중 하나가 질의(따옴표 밖)에 나오는가. 한중일 · 한글은 글자 일치, 아랍어는 붙은 접두를
/// 뗀 낱말 일치, 그 밖은 낱말 첫머리부터 일치 (여러 낱말 구 'v průběhu' 도 그대로).
fn query_has_bridge_word(query: &str, key: &str) -> bool {
    let words: Vec<String> = crate::parsing::BIAS_DICT
        .get("search_bridge")
        .and_then(|b| b.get(key))
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str()).map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default();
    let text = crate::stanza::strip_quoted(query).to_lowercase();
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

fn cond_value_key(v: &Value) -> String {
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
    crate::parsing::BIAS_DICT
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
    if let Some(m) = crate::parsing::BIAS_DICT.get("search_bridge").and_then(|b| b.get("trade_title_plurals")).and_then(|v| v.as_object()) {
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
    let mut text = crate::stanza::strip_quoted(query).to_lowercase();
    for (raw, _) in query_id_tokens(query) {
        text = text.replace(&raw.to_lowercase(), " ");
    }
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !(c.is_alphanumeric() || c == '/')).to_string())
        .filter(|w| !w.is_empty())
        .collect();
    let mut titles: Vec<(String, Vec<String>)> = Vec::new();
    for (code, title) in crate::utils::ai_utils::all_trade_doc_titles().into_iter().chain(trade_title_plurals()) {
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

fn upper_codes(codes: &[String]) -> Vec<String> {
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
fn all_trade_types() -> Vec<String> {
    let mut codes: Vec<String> = vec!["tracking".into(), "TRACKING".into(), "receiving".into(), "Receiving".into(), "shipping".into(), "Shipping".into(), "shipping_doc".into()];
    let mut seen: Vec<String> = Vec::new();
    for (code, _) in crate::utils::ai_utils::all_trade_doc_titles() {
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
    if !latin_dominant(value) || crate::nl_convert::phonetic_skeleton_len(value) < PHONETIC_ANCHOR_MIN_SKELETON {
        return None;
    }
    let mut best: Option<(String, f32)> = None;
    for s in spans.iter() {
        if crate::nl_convert::phonetic_skeleton_len(s) < PHONETIC_ANCHOR_MIN_SKELETON {
            continue;
        }
        if let Some(sim) = crate::nl_convert::phonetic_similarity(s, value) {
            if sim >= PHONETIC_ANCHOR_PASS && best.as_ref().map_or(true, |(_, b)| sim > *b) {
                best = Some((s.clone(), sim));
            }
        }
    }
    best
}

pub async fn trade_relation_join(
    store: &VectorStore,
    results: &mut Vec<Value>,
    plans: &mut Vec<Value>,
    query: &str,
    language: &str,
    emit: &(dyn Fn(&str) + Send + Sync),
) -> RelationReport {
    let mut rep = RelationReport::default();
    let lang = crate::stanza::detect_query_lang(query, language).unwrap_or("en");
    let mut target = match trade_target(query, lang) {
        Some(t) => t,
        None => return rep,
    };
    let all_types = all_trade_types();
    let ids = query_id_tokens(query);
    let id_set: HashSet<&str> = ids.iter().map(|(_, n)| n.as_str()).collect();

    let rows = match store.get_all_items("items", 4000, 0, Some("mode = 'shipping'".to_string())).await {
        Ok(r) => r,
        Err(_) => return rep,
    };
    let mut docs: Vec<(String, Value)> = Vec::new();
    for r in rows.into_iter() {
        if let Ok(v) = serde_json::from_str::<Value>(&r.json_data) {
            if !is_relay_placeholder(&v) {
                docs.push((r.id, v));
            }
        }
    }
    let code_of = |d: &Value| -> String {
        let t = text_of(d.get("doc_type"));
        if t.is_empty() { text_of(d.get("type")).to_uppercase() } else { t.to_uppercase() }
    };
    // 목표 서식이 저장소에 한 건도 없으면(발주서 · 신용장은 번호로만 존재) 서식 전체로 봅니다.
    if !target.iter().any(|c| c == "*") && !docs.iter().any(|(_, d)| target.iter().any(|c| c.eq_ignore_ascii_case(&code_of(d)))) {
        target = vec!["*".to_string()];
    }
    let any = target.iter().any(|c| c == "*");
    rep.target_type = target.join("/");
    let wants = |d: &Value| any || target.iter().any(|c| c.eq_ignore_ascii_case(&code_of(d)));

    let mut tiered: Vec<(usize, usize)> = Vec::new();
    let mut anchors = 0usize;
    if !id_set.is_empty() {
        let mut by_index: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut referrers: HashMap<(String, u32), Vec<usize>> = HashMap::new();
        for (i, (_, d)) in docs.iter().enumerate() {
            if let Some(ix) = d.get("index").and_then(|v| v.as_u64()).filter(|n| *n > 0 && *n <= u64::from(u32::MAX)) {
                by_index.entry(ix as u32).or_default().push(i);
            }
            for (key, ix) in relay_edges(d) {
                referrers.entry((key, ix)).or_default().push(i);
            }
        }
        for (i, (_, d)) in docs.iter().enumerate() {
            let own = ["doc_number", "no"].iter().map(|k| normalize_entity_key(&text_of(d.get(*k)))).find(|k| id_set.contains(k.as_str()));
            if own.is_some() {
                anchors += 1;
                if wants(d) {
                    tiered.push((0, i));
                }
                // 정방향 rel_* · 역방향(이 서식을 rel_<코드> 로 가리키는 서식)
                for (_, ix) in relay_edges(d) {
                    for &j in by_index.get(&ix).into_iter().flatten() {
                        if wants(&docs[j].1) {
                            tiered.push((1, j));
                        }
                    }
                }
                if let Some(ix) = d.get("index").and_then(|v| v.as_u64()).filter(|n| *n > 0 && *n <= u64::from(u32::MAX)) {
                    let key = format!("rel_{}", code_of(d).to_lowercase());
                    for &j in referrers.get(&(key, ix as u32)).into_iter().flatten() {
                        if wants(&docs[j].1) {
                            tiered.push((1, j));
                        }
                    }
                }
                continue;
            }
            // 번호를 reference_* 칸에 적은 서식 (발주서 · 신용장처럼 자기 문서가 없는 번호도 여기서 잇습니다)
            let referenced = d
                .as_object()
                .map(|o| o.iter().any(|(k, v)| k.starts_with("reference_") && id_set.contains(normalize_entity_key(&text_of(Some(v))).as_str())))
                .unwrap_or(false);
            if referenced && wants(d) {
                tiered.push((1, i));
            }
        }
    }
    // 값 고정점: 항구 · 당사자 · 선박 이름이 질의에 그대로 나오는 서식 (쉼표 앞 첫 마디: 'Felixstowe, United Kingdom' → felixstowe)
    // v2 질의 R7('6월에 펠릭스토우에서 선적된 서류') · R5('株式会社ひなた工房 에서 보낸 B/L')는 번호가 없어 고정점이 없었습니다.
    // 🌟 [PHONETIC ANCHOR] 라틴 문자 값이 질의 언어로 음역되어 나오면(フェリクストウ · 费利克斯托 · فيليكستو ↔ Felixstowe) 자음 골격으로 잇습니다
    //    ([`phonetic_value_hit`]). v2 질의 R7 일본어 · 중국어 · 아랍어는 이 고정점이 없어 MRR 0.33 · 0.20 · 0.13 이었습니다.
    let query_norm = norm_text(query);
    let pspans = phonetic_spans(query);
    let mut phonetic: HashMap<String, Option<(String, f32)>> = HashMap::new();
    for (i, (_, d)) in docs.iter().enumerate() {
        let hit = ["pol", "pod", "sender_name", "recipient_name", "vessel"].iter().any(|k| {
            let v = text_of(d.get(*k));
            let head = norm_text(v.split(',').next().unwrap_or(""));
            if name_like(&head) && contains_value(&query_norm, &head) {
                return true;
            }
            if pspans.is_empty() || head.is_empty() {
                return false;
            }
            phonetic.entry(head.clone()).or_insert_with(|| phonetic_value_hit(&pspans, &head)).is_some()
        });
        if hit {
            anchors += 1;
            if wants(d) {
                tiered.push((2, i));
            }
        }
    }
    let heard: Vec<String> = phonetic
        .iter()
        .filter_map(|(head, h)| h.as_ref().map(|(span, s)| format!("'{}' ← '{}' ({:.2})", head, span, s)))
        .collect();
    if !heard.is_empty() {
        emit(&format!(
            "[AI-SEARCH] 🧭 [PHONETIC ANCHOR] 질의에 음역되어 나온 서류 값 {:?} (자음 골격 일치 ≥ {:.2}, 골격 {}개 이상)을 고정점으로 씁니다.",
            heard, PHONETIC_ANCHOR_PASS, PHONETIC_ANCHOR_MIN_SKELETON
        ));
    }
    let targets: Vec<usize> = order_targets(tiered, &docs, results, query_period(query));
    rep.anchors = anchors;
    rep.targets = targets.len();

    // ── 계획: 번호가 풀린 조건(doc_number · hub_reference · reference_* = 번호)을 빼고, 유형은 목표 서식으로 ──
    let mut consumed: Vec<String> = Vec::new();
    let mut widened = 0usize;
    for p in plans.iter_mut() {
        if !targets.is_empty() {
            if let Some(conds) = p.get_mut("conditions").and_then(|c| c.as_array_mut()) {
                conds.retain(|c| {
                    let v = cond_value_key(&c.get("value").cloned().unwrap_or(Value::Null));
                    let explained = !v.is_empty() && v.chars().count() >= 3 && id_set.iter().any(|t| t.contains(v.as_str()));
                    if explained {
                        consumed.push(format!("{} = {}", c.get("path").and_then(|x| x.as_str()).unwrap_or(""), v));
                    }
                    !explained
                });
            }
        }
        if let Some(o) = p.as_object_mut() {
            let types: Vec<String> = if any { all_types.clone() } else { upper_codes(&target) };
            let before = o.get("types").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
            if !targets.is_empty() || any {
                if before != types.len() {
                    widened += 1;
                }
                o.insert("types".to_string(), json!(types));
            }
            if !targets.is_empty() {
                o.insert(
                    "relation".to_string(),
                    json!({ "target_type": target, "target_ids": targets.iter().map(|&j| docs[j].0.clone()).collect::<Vec<_>>() }),
                );
            }
        }
    }
    rep.consumed = consumed.len();
    if targets.is_empty() {
        if widened > 0 {
            emit(&format!(
                "[AI-SEARCH] 🧭 [TRADE RELATION] 질의의 머리가 서식 전체를 뜻하는 이름이라 해석기가 좁힌 유형을 서식 전체({}종)로 되돌립니다.",
                all_types.len()
            ));
        }
        return rep;
    }

    // ── 결과: 목표 서식을 맨 앞으로 (결과에 없으면 다시 읽어 넣음) ──
    let top = results.iter().filter_map(|r| r.get("score").and_then(|v| v.as_f64())).fold(0.0f64, f64::max).max(1.0);
    let pos: HashMap<String, usize> =
        results.iter().enumerate().filter_map(|(i, r)| r.get("id").and_then(|v| v.as_str()).map(|s| (s.to_string(), i))).collect();
    let mut rescued = 0usize;
    for (k, &j) in targets.iter().enumerate() {
        let (id, d) = &docs[j];
        let score = top + 2.0 - (k as f64) * 1e-4;
        match pos.get(id) {
            Some(&r) => {
                if let Some(o) = results[r].as_object_mut() {
                    o.insert("score".to_string(), json!(score));
                    o.insert("relation".to_string(), json!("relation_target"));
                }
            }
            None => {
                results.push(json!({ "id": id, "text": d.to_string(), "score": score, "context_type": code_of(d), "doc_type": code_of(d), "relation": "relation_target" }));
                rescued += 1;
            }
        }
    }
    rep.rescued = rescued;
    results.sort_by(|a, b| {
        let sa = a.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sb = b.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    emit(&format!(
        "[AI-SEARCH] 🧭 [TRADE RELATION] 목표 서식 {:?} · 번호 {:?} → 고정 서식 {}건, 목표 서식 {}건을 맨 앞으로 (재회수 {}건). 번호가 풀린 계획 조건 {}건을 뺐습니다: {:?}",
        target,
        ids.iter().map(|(r, _)| r.as_str()).collect::<Vec<_>>(),
        anchors,
        targets.len(),
        rescued,
        consumed.len(),
        consumed.iter().take(6).collect::<Vec<_>>()
    ));
    rep
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
        let b = &*crate::parsing::BIAS_DICT;
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
fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 { v.iter().map(|x| x / n).collect() } else { v.to_vec() }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        s += a[i] * b[i];
    }
    s
}

/// 🌟 [CROSS-LINGUAL ANCHOR] 후보 (범위, 제목) 중 질의가 번역해 부른 것들: (같은 범위 · 같은 제목인 후보 위치들, 판정, 근거 조각).
/// 같은 제목이 여러 문서(목록 행 · 상세 페이지)에 있어도 한 후보로 겨룹니다 (안 그러면 1위와 2위가 같은 제목이라 차가 0).
/// 임베딩을 못 하면 빈 목록.
pub async fn cross_lingual_title_anchors(
    model: &crate::model::LogisModel,
    query: &str,
    candidates: &[(String, String)],
) -> Vec<(Vec<usize>, CrossAnchor, String)> {
    let spans = query_value_spans(query);
    let mut unique: Vec<(String, String)> = Vec::new();
    for (g, t) in candidates.iter() {
        if !unique.iter().any(|(ug, ut)| ug == g && ut == t) {
            unique.push((g.clone(), t.clone()));
        }
    }
    if unique.len() < 2 || spans.is_empty() {
        return Vec::new();
    }
    let Ok(bank) = model.get_embedding_batch_pinned(structure_bank().to_vec()).await else { return Vec::new() };
    let Ok(span_vecs) = model.get_embedding_batch(spans.clone()).await else { return Vec::new() };
    let Ok(title_vecs) = model.get_embedding_batch(unique.iter().map(|(_, t)| t.clone()).collect()).await else { return Vec::new() };
    let bank: Vec<Vec<f32>> = bank.iter().map(|v| unit(v)).collect();
    let titles: Vec<Vec<f32>> = title_vecs.iter().map(|v| unit(v)).collect();
    let cal = crate::models::embedding::calibrate_cos;
    let mut sim: Vec<Vec<f32>> = Vec::with_capacity(spans.len());
    let mut span_bank: Vec<f32> = Vec::with_capacity(spans.len());
    for v in span_vecs.iter() {
        let s = unit(v);
        sim.push(titles.iter().map(|t| cal(dot(&s, t))).collect());
        span_bank.push(cal(bank.iter().map(|b| dot(&s, b)).fold(f32::MIN, f32::max)));
    }
    let groups: Vec<String> = unique.iter().map(|(g, _)| g.clone()).collect();
    cross_anchor_pick(&sim, &span_bank, &groups)
        .into_iter()
        .map(|a| {
            let (g, t) = &unique[a.title];
            let members: Vec<usize> = (0..candidates.len()).filter(|&i| candidates[i].0 == *g && candidates[i].1 == *t).collect();
            let s = spans[a.span].clone();
            (members, a, s)
        })
        .collect()
}

/// 결과마다 기록 본문(data). 청크로만 올라온 결과는 text 가 청크 글이라('This touch is titled …') 저장소에서 id 로 읽습니다.
/// v2 질의 A4 한국어는 결과 55건 중 33건이 청크 결과였습니다 — 본문 없이는 방문자 · 링크를 볼 수 없습니다.
async fn result_docs(store: &VectorStore, results: &[Value]) -> Vec<Option<Value>> {
    let mut out: Vec<Option<Value>> = Vec::with_capacity(results.len());
    for r in results.iter() {
        let parsed = r
            .get("text")
            .and_then(|t| t.as_str())
            .and_then(|t| serde_json::from_str::<Value>(t).ok())
            .filter(|d| d.is_object());
        let doc = match parsed {
            Some(d) => Some(d),
            None => match r.get("id").and_then(|v| v.as_str()) {
                Some(id) => match store.get_item_by_id("items", id).await {
                    Ok(Some(d)) => serde_json::from_str::<Value>(&d.json_data).ok().map(|mut v| {
                        // 본문에 없는 봉투 칸(방문자 · 사이트)은 열에서 채웁니다.
                        if let Some(o) = v.as_object_mut() {
                            for (k, col) in [("from", &d.from), ("cc", &d.cc)] {
                                if o.get(k).map_or(true, |x| text_of(Some(x)).is_empty()) && !col.trim().is_empty() {
                                    o.insert(k.to_string(), json!(col));
                                }
                            }
                        }
                        v
                    }),
                    _ => None,
                },
                None => None,
            },
        };
        out.push(doc);
    }
    out
}

/// 링크 매개변수 값이 (사이트, 번호) 고정점인 행동 기록을 결과 맨 앞으로 올립니다. docs 는 results 와 같은 순서의 기록 본문. 올린 건수.
fn boost_linked_records(results: &mut Vec<Value>, docs: &[Option<Value>], ids: &HashSet<(String, String)>) -> usize {
    let top = results.iter().filter_map(|r| r.get("score").and_then(|v| v.as_f64())).fold(0.0f64, f64::max).max(1.0);
    let mut boosted = 0usize;
    for (r, doc) in results.iter_mut().zip(docs.iter()) {
        let Some(doc) = doc else { continue };
        let link = text_of(doc.get("link"));
        let site = text_of(doc.get("cc"));
        let hit = link.split_once('?').map_or(false, |(_, q)| {
            q.split('&').filter_map(|p| p.split_once('=')).any(|(_, v)| {
                let k = normalize_entity_key(v);
                ids.contains(&(site.clone(), k.clone())) || (site.is_empty() && ids.iter().any(|(_, x)| *x == k))
            })
        });
        if hit {
            if let Some(o) = r.as_object_mut() {
                let cur = o.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                o.insert("score".to_string(), json!(top + 1.0 + cur * 1e-3));
                o.insert("relation".to_string(), json!("relation_anchor"));
            }
            boosted += 1;
        }
    }
    if boosted > 0 {
        results.sort_by(|a, b| {
            let sa = a.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let sb = b.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    boosted
}

/// 🌟 [ANALYTIC ANCHOR] 행동 로그 질의에 커머스 문서의 제목(행사 · 쿠폰 · 상품 이름)이 그대로 나오면, 그 문서의 자기 번호
/// (ev_id · cp_id · it_id …)를 링크 매개변수로 가진 행동 기록(eventNo=EV718 · goodsNo=…)을 결과 맨 앞으로 올립니다.
/// v2 질의 A5('Summer getaway picks' 이벤트 페이지에서 쿠폰 받은 기록)는 같은 사이트의 다른 행사 쿠폰 기록이 앞에 섞였습니다.
/// 행동 기록과 커머스 문서는 같은 사이트 범위(cc)를 씁니다. 범위가 없으면 커머스 문서 전체에서 찾습니다.
/// 글자 그대로 나온 제목이 기록을 하나도 올리지 못하면 질의 언어로 번역된 제목을 찾습니다 ([CROSS-LINGUAL ANCHOR]).
pub async fn analytic_anchor_boost(
    store: &VectorStore,
    model: &crate::model::LogisModel,
    results: &mut Vec<Value>,
    query: &str,
    cc: &str,
    emit: &(dyn Fn(&str) + Send + Sync),
) -> usize {
    if results.is_empty() {
        return 0;
    }
    let mut filter = "mode = 'commerce'".to_string();
    if !cc.trim().is_empty() {
        filter.push_str(&format!(" AND `cc` = '{}'", cc.replace('\'', "''")));
    }
    let rows = match store.get_all_items("items", 4000, 0, Some(filter)).await {
        Ok(r) => r,
        Err(_) => return 0,
    };
    let query_norm = norm_text(query);
    // 번호는 사이트 안에서만 유일합니다 (네 쇼핑몰이 모두 EV718 을 씀): (사이트 cc, 번호) 쌍으로 맞춥니다.
    // v2 질의 A5 일본어는 copper 의 'Summer getaway picks'(EV718) 로 kranich · sorae 의 EV718 기록까지 올라와 1위를 내줬습니다.
    let mut ids: HashSet<(String, String)> = HashSet::new();
    let mut titles: Vec<String> = Vec::new();
    // 글자 그대로 나오지 않은 제목: (사이트, 원래 제목, 자기 번호들)
    let mut others: Vec<(String, String, Vec<String>)> = Vec::new();
    for r in rows.into_iter() {
        let Ok(d) = serde_json::from_str::<Value>(&r.json_data) else { continue };
        if !matches!(relay_type_family(&text_of(d.get("type"))).as_str(), "event" | "goods") || is_relay_placeholder(&d) {
            continue;
        }
        let raw = text_of(d.get("title"));
        let title = norm_text(&raw);
        let site = text_of(d.get("cc"));
        if name_like(&title) && contains_value(&query_norm, &title) {
            ids.extend(own_ids(&d).into_iter().map(|k| (site.clone(), k)));
            titles.push(title);
        } else if !raw.is_empty() {
            others.push((site, raw, own_ids(&d)));
        }
    }
    let docs = if ids.is_empty() && others.is_empty() { Vec::new() } else { result_docs(store, results).await };
    let boosted = if ids.is_empty() { 0 } else { boost_linked_records(results, &docs, &ids) };
    if boosted > 0 {
        emit(&format!(
            "[AI-SEARCH] 🧭 [ANALYTIC ANCHOR] 질의에 나온 커머스 제목 {:?} 의 번호 {:?} 를 같은 사이트 링크에 가진 행동 기록 {}건을 맨 앞으로 올렸습니다.",
            titles,
            ids.iter().take(6).map(|(_, k)| k.as_str()).collect::<Vec<_>>(),
            boosted
        ));
        return boosted;
    }
    // 🌟 [CROSS-LINGUAL ANCHOR] 글자 그대로 나온 제목이 없거나 그 상품의 기록이 없으면, 질의 언어로 번역해 부른 제목을 찾습니다.
    let cands: Vec<(String, String)> = others.iter().map(|(s, t, _)| (s.clone(), t.clone())).collect();
    let picked = cross_lingual_title_anchors(model, query, &cands).await;
    if picked.is_empty() {
        return 0;
    }
    let mut ids: HashSet<(String, String)> = HashSet::new();
    for (members, _, _) in picked.iter() {
        for &m in members.iter() {
            let (site, _, own) = &others[m];
            ids.extend(own.iter().map(|k| (site.clone(), k.clone())));
        }
    }
    let boosted = boost_linked_records(results, &docs, &ids);
    emit(&format!(
        "[AI-SEARCH] 🧭 [CROSS-LINGUAL ANCHOR] 질의 조각이 번역해 부른 커머스 제목 {:?} → 그 번호를 같은 사이트 링크에 가진 행동 기록 {}건을 맨 앞으로 올렸습니다.",
        picked
            .iter()
            .map(|(m, a, span)| format!("'{}' ← '{}' (보정 코사인 {:.3} · 2위와 {:+.3} · 구조 낱말과 {:+.3})", others[m[0]].1, span, a.cos, a.margin, a.bank_gap))
            .collect::<Vec<_>>(),
        boosted
    ));
    boosted
}

/// 🌟 [ANALYTIC NEGATION] 부정 조건의 방문 단위 여집합 (analytic.rs [ANALYTIC NEGATION] 설명).
/// 결과 기록 · 리포트의 방문자들의 행동 기록을 다시 읽어 방문으로 묶고, 부정된 일(주문 완료 · 제외 행동 · 제외 페이지)이 일어난
/// 방문의 기록 · 리포트를 결과에서 뺍니다. 페이지 · 행동으로 거른 결과에는 증거 기록(주문 완료 페이지)이 없으므로 다시 읽어야 합니다.
/// 다 빠지면 부정 판정이 틀렸을 수 있어 그대로 둡니다. 뺀 건수를 돌려줍니다.
pub async fn analytic_visit_complement(
    store: &VectorStore,
    results: &mut Vec<Value>,
    neg: &crate::analytic::NegatedConditions,
    cc: &str,
    emit: &(dyn Fn(&str) + Send + Sync),
) -> usize {
    if results.is_empty() || neg.is_empty() {
        return 0;
    }
    let from_of = |d: &Value| text_of(d.get("from")).trim().to_string();
    let docs: Vec<Option<Value>> = result_docs(store, results).await;
    let mut visitors: Vec<String> = docs.iter().flatten().map(|d| from_of(d)).filter(|s| !s.is_empty()).collect();
    visitors.sort();
    visitors.dedup();
    if visitors.is_empty() {
        return 0;
    }
    let mut filter = format!(
        "mode = 'analytic' AND `from` IN ({})",
        visitors.iter().map(|v| format!("'{}'", v.replace('\'', "''"))).collect::<Vec<_>>().join(", ")
    );
    if !cc.trim().is_empty() {
        filter.push_str(&format!(" AND `cc` = '{}'", cc.replace('\'', "''")));
    }
    let rows = match store.get_all_items("items", 20_000, 0, Some(filter)).await {
        Ok(r) => r,
        Err(e) => {
            emit(&format!("[AI-SEARCH] ⚠️ [ANALYTIC NEGATION] 방문자 기록을 읽지 못해 부정 조건을 적용하지 않습니다 ({}).", e));
            return 0;
        }
    };
    let records: Vec<(String, Value)> = rows
        .into_iter()
        .filter_map(|r| {
            let d = serde_json::from_str::<Value>(&r.json_data).ok()?;
            let from = from_of(&d);
            Some((if from.is_empty() { r.from.trim().to_string() } else { from }, d))
        })
        .collect();
    let visits = crate::analytic::evidence_visits(&records, neg);
    let drop: Vec<bool> = docs
        .iter()
        .map(|d| {
            d.as_ref().map_or(false, |d| {
                crate::analytic::record_window(d).map_or(false, |w| crate::analytic::in_evidence_visit(&from_of(d), w, &visits))
            })
        })
        .collect();
    let dropped = drop.iter().filter(|x| **x).count();
    if dropped == 0 || dropped == results.len() {
        emit(&format!(
            "[AI-SEARCH] 🚫 [ANALYTIC NEGATION] 부정 조건 의도 {:?} · 행동 {:?} · 페이지 {:?}: 방문자 {}명의 기록 {}건에서 그 일이 일어난 방문 {}개 — 결과 {}건 중 {}건이 그 방문의 것이라 {}.",
            neg.intents,
            neg.actions,
            neg.pages,
            visitors.len(),
            records.len(),
            visits.len(),
            results.len(),
            dropped,
            if dropped == 0 { "뺄 것이 없습니다" } else { "다 빠지면 판정이 틀렸을 수 있어 그대로 둡니다" }
        ));
        return 0;
    }
    let mut k = 0usize;
    results.retain(|_| {
        let keep = !drop[k];
        k += 1;
        keep
    });
    emit(&format!(
        "[AI-SEARCH] 🚫 [ANALYTIC NEGATION] 부정 조건 의도 {:?} · 행동 {:?} · 페이지 {:?}: 방문자 {}명의 기록 {}건을 {}분 간격으로 방문으로 묶어, 그 일이 일어난 방문 {}개의 기록 · 리포트 {}건을 결과에서 뺐습니다 (남은 {}건).",
        neg.intents,
        neg.actions,
        neg.pages,
        visitors.len(),
        records.len(),
        crate::analytic::VISIT_IDLE_MS / 60_000,
        visits.len(),
        dropped,
        results.len()
    ));
    dropped
}

/// 질의의 절대 기간 (오늘 기준; 'in September' · '9월' · 'from X to Y')
fn query_period(query: &str) -> Option<(chrono::NaiveDate, chrono::NaiveDate)> {
    let words: Vec<String> = crate::stanza::strip_quoted(query).split_whitespace().map(String::from).collect();
    let today = chrono::Local::now().date_naive();
    crate::utils::ai_utils::exact_absolute_period(&words, today).map(|p| (p.start, p.end))
}

/// 문서의 날짜 칸들 (등록 · 주문 · 발송 · 배송 완료 · 발행일; 문자열 YYYY-MM-DD… 또는 epoch ms)
fn doc_dates(doc: &Value) -> Vec<chrono::NaiveDate> {
    ["registration_date", "order_date", "payment_date", "shipping_date", "shipped_at", "delivered_at", "issue_date"]
        .iter()
        .filter_map(|k| date_of(doc.get(*k)))
        .collect()
}

/// 목표 순서: 단계 → 질의 기간 안의 문서 먼저 → 원래 회수 순위 (목표끼리는 백엔드가 매긴 질의 일치 순서를 지킵니다;
/// v2 질의 C10 '9월에 X 고객에게 배송된 운송장' 은 고객의 운송장 전체가 임의 순서로 올라와 9월 것이 5위로 밀렸습니다)
fn order_targets(tiered: Vec<(usize, usize)>, docs: &[(String, Value)], results: &[Value], period: Option<(chrono::NaiveDate, chrono::NaiveDate)>) -> Vec<usize> {
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

struct Anchor {
    doc: usize,
    kind: &'static str,
    matched: String,
    primary: bool,
}

pub async fn relation_join(
    store: &VectorStore,
    model: &crate::model::LogisModel,
    results: &mut Vec<Value>,
    plans: &mut Vec<Value>,
    query: &str,
    language: &str,
    cc: &str,
    emit: &(dyn Fn(&str) + Send + Sync),
) -> RelationReport {
    let mut rep = RelationReport::default();
    if plans.iter().any(|p| p.get("relay").and_then(|r| r.get("applied")).and_then(|v| v.as_bool()) == Some(true)) {
        return rep;
    }
    let lang = crate::stanza::detect_query_lang(query, language).unwrap_or("en");
    let mentions = type_mentions(query);
    let target = match target_type(&mentions, lang) {
        Some(t) => t,
        None => return rep,
    };
    let target_fam = relay_type_family(target);
    rep.target_type = target.to_string();
    let mentioned: HashSet<String> = mentions.iter().map(|(_, t)| relay_type_family(t)).collect();

    // ── 범위 안 커머스 문서 ──
    let mut filter = "mode = 'commerce'".to_string();
    if !cc.trim().is_empty() {
        filter.push_str(&format!(" AND `cc` = '{}'", cc.replace('\'', "''")));
    }
    let rows = match store.get_all_items("items", 4000, 0, Some(filter)).await {
        Ok(r) => r,
        Err(_) => return rep,
    };
    let mut docs: Vec<(String, Value)> = Vec::new();
    for r in rows.into_iter() {
        if let Ok(v) = serde_json::from_str::<Value>(&r.json_data) {
            if !is_relay_placeholder(&v) {
                docs.push((r.id, v));
            }
        }
    }
    let fam_of: Vec<String> = docs.iter().map(|(_, d)| relay_type_family(&text_of(d.get("type")))).collect();
    let mut by_index: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut referrers: HashMap<(String, u32), Vec<usize>> = HashMap::new();
    for (i, (_, d)) in docs.iter().enumerate() {
        if let Some(ix) = d.get("index").and_then(|v| v.as_u64()).filter(|n| *n > 0 && *n <= u64::from(u32::MAX)) {
            by_index.entry(ix as u32).or_default().push(i);
        }
        for (key, ix) in relay_edges(d) {
            referrers.entry((key, ix)).or_default().push(i);
        }
    }

    // ── 고정점 ──
    let query_norm = norm_text(query);
    let ids = query_id_tokens(query);
    let id_set: HashSet<&str> = ids.iter().map(|(_, n)| n.as_str()).collect();
    let mut anchors: Vec<Anchor> = Vec::new();
    for (i, (_, d)) in docs.iter().enumerate() {
        if !id_set.is_empty() {
            if let Some(hit) = own_ids(d).into_iter().find(|k| id_set.contains(k.as_str())) {
                anchors.push(Anchor { doc: i, kind: "id", matched: hit, primary: true });
                continue;
            }
        }
        let person = PERSON_KEYS
            .iter()
            .map(|k| norm_text(&text_of(d.get(*k))))
            .find(|v| name_like(v) && contains_value(&query_norm, v));
        if let Some(name) = person {
            anchors.push(Anchor { doc: i, kind: "person", matched: name, primary: false });
            continue;
        }
        if fam_of[i] != "order" {
            let title = norm_text(&text_of(d.get("title")));
            if name_like(&title) && contains_value(&query_norm, &title) {
                anchors.push(Anchor { doc: i, kind: "title", matched: title, primary: false });
            }
        }
    }
    // 상태: 계획의 상태 조건, 없으면 질의의 상태 낱말. 목표가 아닌 유형 낱말이 가리키는 문서에만 겁니다.
    let mut status_codes: Vec<i64> = Vec::new();
    for p in plans.iter() {
        for c in p.get("conditions").and_then(|v| v.as_array()).into_iter().flatten() {
            if c.get("path").and_then(|v| v.as_str()) == Some("data.status") && c.get("op").and_then(|v| v.as_str()) == Some("eq") {
                if let Some(code) = c.get("value").and_then(|v| v.as_i64()).filter(|c| *c != 0) {
                    if !status_codes.contains(&code) {
                        status_codes.push(code);
                    }
                }
            }
        }
    }
    if status_codes.is_empty() {
        status_codes = query_status_codes(query);
    }
    let status_fams: Vec<String> = mentioned.iter().filter(|f| **f != target_fam).cloned().collect();
    if !status_codes.is_empty() && !status_fams.is_empty() {
        for (i, (_, d)) in docs.iter().enumerate() {
            if status_fams.contains(&fam_of[i]) && status_codes.contains(&status_code(d)) && !anchors.iter().any(|a| a.doc == i) {
                anchors.push(Anchor { doc: i, kind: "status", matched: status_code(d).to_string(), primary: true });
            }
        }
    }
    // 🌟 [CROSS-LINGUAL ANCHOR] 번호 · 사람 · 제목 · 상태 고정점이 하나도 없으면, 질의 언어로 번역해 부른 상품 · 행사 · 쿠폰 제목을 찾습니다.
    //    v2 질의 C4('orders in April that included Fleece-lined zip hoodie' ↔ 저장 제목 '기모 후드집업').
    //    그 조각은 고정점이 풀어낸 값입니다: 제목 · 이름 칸 조건의 값이 조각과 겹치면 아래에서 뺍니다.
    let mut cross_spans: Vec<String> = Vec::new();
    if anchors.is_empty() {
        let cand_docs: Vec<usize> = (0..docs.len())
            .filter(|&i| matches!(fam_of[i].as_str(), "goods" | "event" | "coupon") && !text_of(docs[i].1.get("title")).is_empty())
            .collect();
        let cands: Vec<(String, String)> =
            cand_docs.iter().map(|&i| (text_of(docs[i].1.get("cc")), text_of(docs[i].1.get("title")))).collect();
        let picked = cross_lingual_title_anchors(model, query, &cands).await;
        if !picked.is_empty() {
            emit(&format!(
                "[AI-SEARCH] 🧭 [CROSS-LINGUAL ANCHOR] 질의 조각이 번역해 부른 제목 {:?} 을 고정점으로 씁니다.",
                picked
                    .iter()
                    .map(|(m, a, span)| format!("'{}' ← '{}' (보정 코사인 {:.3} · 2위와 {:+.3} · 구조 낱말과 {:+.3})", cands[m[0]].1, span, a.cos, a.margin, a.bank_gap))
                    .collect::<Vec<_>>()
            ));
        }
        for (members, _, span) in picked.into_iter() {
            for m in members {
                anchors.push(Anchor { doc: cand_docs[m], kind: "title", matched: norm_text(&cands[m].1), primary: false });
            }
            cross_spans.push(norm_text(&span));
        }
    }
    if anchors.is_empty() {
        return rep;
    }
    for a in anchors.iter_mut() {
        let f = &fam_of[a.doc];
        a.primary = a.primary || *f == target_fam || mentioned.contains(f);
    }
    rep.anchors = anchors.len();

    // ── 연결: 고정점 → 목표 (단계: 0 목표 유형 고정점, 1 한 단계, 2 고정점을 가리키는 문서를 거친 두 단계; 보조 고정점은 +3) ──
    // 취소 · 환불 · 반품 문서를 거친 경로는 한 단계 뒤로 둡니다 (질의가 그 상태를 묻지 않을 때). v2 질의 C3 프랑스어
    // 'Felix Wagner 가 주문한 상품' 은 취소된 주문의 상품이 1위였습니다 — relay_ledger 기간 조인과 같은 규칙입니다.
    let voided = |d: &Value| {
        let c = status_code(d);
        void_status(c) && !status_codes.contains(&c)
    };
    let mut tiered: Vec<(usize, usize)> = Vec::new();
    let add = |tiered: &mut Vec<(usize, usize)>, tier: usize, j: usize| {
        if fam_of[j] == target_fam {
            tiered.push((tier, j));
        }
    };
    for a in anchors.iter() {
        let base = if a.primary { 0 } else { 3 };
        let d = &docs[a.doc].1;
        if fam_of[a.doc] == target_fam {
            add(&mut tiered, base, a.doc);
            continue;
        }
        let pen = voided(d) as usize;
        // 정방향: 고정점의 연결 칸 → 목표
        for (key, ix) in relay_edges(d) {
            if relay_type_family(&key) == target_fam {
                for &j in by_index.get(&ix).into_iter().flatten() {
                    add(&mut tiered, base + 1 + pen, j);
                }
            }
        }
        // 역방향: 목표가 고정점을 가리킴 · 두 단계: 고정점을 가리키는 문서의 연결 칸 → 목표
        let own_ix = d.get("index").and_then(|v| v.as_u64()).filter(|n| *n > 0 && *n <= u64::from(u32::MAX)).map(|n| n as u32);
        if let (Some(key), Some(ix)) = (relay_key_for_type(&text_of(d.get("type"))), own_ix) {
            for &r in referrers.get(&(key.to_string(), ix)).into_iter().flatten() {
                if fam_of[r] == target_fam {
                    add(&mut tiered, base + 1 + pen, r);
                    continue;
                }
                if fam_of[r] == fam_of[a.doc] {
                    continue;
                }
                let pen2 = pen + voided(&docs[r].1) as usize;
                for (k2, ix2) in relay_edges(&docs[r].1) {
                    if relay_type_family(&k2) == target_fam {
                        for &j in by_index.get(&ix2).into_iter().flatten() {
                            add(&mut tiered, base + 2 + pen2, j);
                        }
                    }
                }
            }
        }
    }
    // ── 기간 다리: 질의가 고정점의 기간을 말하면('X 기간에 팔린 상품' · 'sold during the X promotion') 고정점(행사 · 쿠폰)의
    //    유효 기간 안에 날짜가 든 거래 문서(주문, search_bridge.sales_to_order)를 거쳐 목표로 건넙니다. 같은 단계 안에서는
    //    두 경로가 모두 닿은 목표 → 기간 경로만 닿은 목표 → 고정점이 직접 가리키기만 한 목표(행사 진열 상품) 순입니다.
    //    v2 질의 C7 은 진열 상품 다섯 중 기간에 안 팔린 하나가 절반의 언어에서 1위였습니다. ('期間限定クーポン X を使った注文' 처럼
    //    기간 낱말이 이름에 붙은 질의는 쿠폰을 쓴 주문이 두 경로 모두에 닿아 그대로 앞입니다.)
    let mut bridged: Vec<(usize, usize)> = Vec::new();
    let mut bridge_orders = 0usize;
    if mentions_period(query) {
        let via_fam = crate::parsing::BIAS_DICT
            .get("search_bridge")
            .and_then(|sb| sb.get("sales_to_order"))
            .and_then(|n| n.get("target_domain"))
            .and_then(|v| v.as_str())
            .map(relay_type_family)
            .unwrap_or_else(|| "order".to_string());
        for a in anchors.iter() {
            let Some((start, end)) = validity_period(&docs[a.doc].1) else { continue };
            if fam_of[a.doc] == via_fam {
                continue;
            }
            let base = if a.primary { 0 } else { 3 };
            for (j, (_, d)) in docs.iter().enumerate() {
                if fam_of[j] != via_fam || !doc_dates(d).iter().any(|x| *x >= start && *x <= end) {
                    continue;
                }
                bridge_orders += 1;
                let tier = base + 1 + voided(d) as usize;
                if fam_of[j] == target_fam {
                    bridged.push((tier, j));
                    continue;
                }
                for (key, ix) in relay_edges(d) {
                    if relay_type_family(&key) == target_fam {
                        for &g in by_index.get(&ix).into_iter().flatten() {
                            if fam_of[g] == target_fam {
                                bridged.push((tier, g));
                            }
                        }
                    }
                }
            }
        }
        if !bridged.is_empty() {
            let best = |list: &[(usize, usize)]| {
                let mut m: HashMap<usize, usize> = HashMap::new();
                for &(t, j) in list {
                    let e = m.entry(j).or_insert(t);
                    *e = (*e).min(t);
                }
                m
            };
            let direct = best(&tiered);
            let period = best(&bridged);
            let mut merged: Vec<(usize, usize)> = Vec::new();
            for (&j, &bt) in period.iter() {
                match direct.get(&j) {
                    Some(&dt) => merged.push((dt.min(bt) * 3, j)),
                    None => merged.push((bt * 3 + 1, j)),
                }
            }
            for (&j, &dt) in direct.iter() {
                if !period.contains_key(&j) {
                    merged.push((dt * 3 + 2, j));
                }
            }
            emit(&format!(
                "[AI-SEARCH] 🧭 [RELATION / PERIOD] 질의가 고정점의 기간을 말해 유효 기간 안의 거래 문서 {}건을 거친 목표 {}건을 잇습니다 (두 경로 모두 {}건 → 기간 경로만 {}건 → 직접 경로만 {}건).",
                bridge_orders,
                period.len(),
                period.keys().filter(|j| direct.contains_key(j)).count(),
                period.keys().filter(|j| !direct.contains_key(j)).count(),
                direct.keys().filter(|j| !period.contains_key(j)).count()
            ));
            tiered = merged;
        }
    }
    let targets: Vec<usize> = order_targets(tiered, &docs, results, query_period(query));
    if targets.is_empty() {
        emit(&format!(
            "[AI-SEARCH] ⚪ [RELATION] 목표 '{}' · 고정점 {}건(첫 {:?})에서 목표 유형으로 이어진 문서가 없어 결과를 바꾸지 않습니다.",
            target,
            anchors.len(),
            anchors.iter().take(3).map(|a| format!("{}:{}", a.kind, a.matched)).collect::<Vec<_>>()
        ));
        return rep;
    }
    rep.targets = targets.len();

    // ── 결과: 목표 → 주 고정점 → 나머지 ──
    let top = results.iter().filter_map(|r| r.get("score").and_then(|v| v.as_f64())).fold(0.0f64, f64::max).max(1.0);
    let pos: HashMap<String, usize> =
        results.iter().enumerate().filter_map(|(i, r)| r.get("id").and_then(|v| v.as_str()).map(|s| (s.to_string(), i))).collect();
    let primary_anchors: Vec<usize> = anchors.iter().filter(|a| a.primary && fam_of[a.doc] != target_fam).map(|a| a.doc).collect();
    let lift = |results: &mut Vec<Value>, j: usize, score: f64, relation: &str, rescued: &mut usize| {
        let (id, d) = &docs[j];
        match pos.get(id) {
            Some(&k) => {
                if let Some(o) = results[k].as_object_mut() {
                    let cur = o.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    if score > cur {
                        o.insert("score".to_string(), json!(score));
                        o.insert("relation".to_string(), json!(relation));
                    }
                }
            }
            None => {
                results.push(json!({
                    "id": id,
                    "text": d.to_string(),
                    "score": score,
                    "context_type": text_of(d.get("type")),
                    "doc_type": text_of(d.get("type")),
                    "relation": relation
                }));
                *rescued += 1;
            }
        }
    };
    let mut rescued = 0usize;
    for (k, &j) in targets.iter().enumerate() {
        lift(results, j, top + 2.0 - (k as f64) * 1e-4, "relation_target", &mut rescued);
    }
    for (k, &j) in primary_anchors.iter().take(20).enumerate() {
        if !targets.contains(&j) {
            lift(results, j, top + 1.0 - (k as f64) * 1e-4, "relation_anchor", &mut rescued);
        }
    }
    rep.rescued = rescued;
    results.sort_by(|a, b| {
        let sa = a.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let sb = b.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });

    // ── 계획: 고정점이 풀어낸 조건을 뺍니다 ──
    // 제목 · 이름 칸 조건의 값이 고정점 이름의 조각이면('준비전' ⊂ '여름 휴가 준비전') 그 값은 고정점에서 온 것입니다. v2 보류 질의
    // C6 프랑스어는 행사 이름 조각이 상품 제목 조건으로 남아 목표 상품이 모두 걸러지고 행사 문서만 보였습니다. 색 같은 다른 칸은 그대로 둡니다.
    let anchor_texts: HashSet<String> = anchors.iter().filter(|a| a.kind != "id").map(|a| a.matched.clone()).collect();
    let name_fragment = |path: &str, s: &str| {
        let v = norm_text(s);
        let long_enough = if v.chars().any(glued_script) { v.chars().count() >= 2 } else { v.chars().count() >= 3 };
        (path.ends_with("title") || path.ends_with("name"))
            && long_enough
            && (anchor_texts.iter().any(|a| contains_value(a, &v))
                || cross_spans.iter().any(|s| contains_value(s, &v) || contains_value(&v, s)))
    };
    let anchor_status: HashSet<i64> = anchors.iter().filter(|a| a.kind == "status").filter_map(|a| a.matched.parse::<i64>().ok()).collect();
    let mut consumed: Vec<String> = Vec::new();
    for p in plans.iter_mut() {
        let plan_fam = relay_type_family(p.get("type").and_then(|v| v.as_str()).unwrap_or(""));
        if let Some(conds) = p.get_mut("conditions").and_then(|c| c.as_array_mut()) {
            conds.retain(|c| {
                let value = c.get("value").cloned().unwrap_or(Value::Null);
                let path = c.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let explained = plan_fam != target_fam
                    || id_set.contains(cond_value_key(&value).as_str())
                    || value.as_str().map_or(false, |s| anchor_texts.contains(&norm_text(s)) || name_fragment(&path, s))
                    || (path == "data.status" && value.as_i64().map_or(false, |v| anchor_status.contains(&v)));
                if explained {
                    consumed.push(format!("{} {} {}", path, c.get("op").and_then(|v| v.as_str()).unwrap_or(""), value));
                }
                !explained
            });
        }
        if let Some(o) = p.as_object_mut() {
            o.insert(
                "relation".to_string(),
                json!({
                    "target_type": target,
                    "anchors": anchors.iter().take(12).map(|a| json!({ "id": docs[a.doc].0, "kind": a.kind, "matched": a.matched, "primary": a.primary })).collect::<Vec<_>>(),
                    "target_ids": targets.iter().map(|&j| docs[j].0.clone()).collect::<Vec<_>>()
                }),
            );
        }
    }
    rep.consumed = consumed.len();
    emit(&format!(
        "[AI-SEARCH] 🧭 [RELATION] 목표 '{}' ({} 머리 규칙, 유형 낱말 {:?}) · 고정점 {}건 {:?} → 목표 문서 {}건을 맨 앞으로 (재회수 {}건). 고정점이 풀어낸 계획 조건 {}건을 뺐습니다: {:?}",
        target,
        if matches!(lang, "ko" | "ja" | "zh-hans") { "끝" } else { "처음" },
        mentions.iter().map(|(_, t)| *t).collect::<Vec<_>>(),
        anchors.len(),
        anchors.iter().take(4).map(|a| format!("{}:{}{}", a.kind, a.matched, if a.primary { "" } else { "(보조)" })).collect::<Vec<_>>(),
        targets.len(),
        rescued,
        consumed.len(),
        consumed.iter().take(6).collect::<Vec<_>>()
    ));
    rep
}

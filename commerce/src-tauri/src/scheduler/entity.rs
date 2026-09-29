/// 🌟 [ENTITY KEY v5] 식별자 정규화. 커머스/무역 공통 단일 규칙입니다.
///
///  ── 왜 normalize_numeric_homoglyphs 를 무조건 쓰면 안 되는가 ──
///   그 함수는 커머스 주문번호(순수 숫자열)를 위해 s→5, o→0, i→1 치환을 수행합니다.
///   무역 서식 코드에 그대로 적용하면
///     SC-2026-0802  → 5C20260802
///     SWB-55432219  → 5WB55432219
///     SOA-2026-0920 → 50A20260920
///   처럼 S 로 시작하는 서식 전부가 변조됩니다.
///   (로그 실측: task_1787731795587 → ta5k1787731795587)
///
///  ── 규칙 ──
///   ① 구분자(-, _, ., ,, 공백, /)만 제거
///   ② 대문자로 통일 (BL-55432219 ↔ bl-55432219 를 같은 문서로 봅니다)
///   ③ 호모글리프 치환은 '알파벳이 하나도 없는 순수 숫자열' 일 때만 적용
pub fn normalize_entity_key(raw: &str) -> String {
    let stripped: String = raw
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_uppercase();
    if stripped.is_empty() {
        return stripped;
    }
    let has_alpha = stripped.chars().any(|c| c.is_alphabetic());
    if has_alpha {
        stripped
    } else {
        crate::utils::hash::normalize_numeric_homoglyphs(&stripped)
    }
}

/// 🌟 [ENTITY KEY v5] 문서 index. 이 식 외의 index 계산은 존재해서는 안 됩니다.
pub fn entity_index(type_: &str, team_id: &str, raw_no: &str) -> u32 {
    let clean = normalize_entity_key(raw_no);
    crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!(
        "{}{}{}",
        type_, team_id, clean
    )))
}

/// 🌟 [ENTITY KEY v5] 문서 id. 반드시 index 로부터만 유도합니다.
///  ── 왜 index 로부터인가 ──
///   릴레이는 '상대의 index 를 결정론으로 재현' 해서 상대를 찾습니다.
///   id 가 index 와 무관한 식으로 만들어지면, index 로는 찾았는데
///   저장은 다른 id 로 되어 같은 문서가 두 행이 됩니다.
pub fn entity_id(team_id: &str, index: u32) -> String {
    crate::utils::hash::hash_id(&format!("{}{}", team_id, index))
}

/// 🌟 [ENTITY KEY v5] 봉투 bcc. 타입별 목록 필터가 이 값으로 동작합니다.
pub fn entity_bcc(type_: &str, cc: &str) -> String {
    crate::utils::hash::hash_id(&format!("{}{}", type_, cc))
}

pub fn entity_seed(cc: &str, raw: &str) -> String {
    let t = raw.trim();
    let scope = cc.trim();
    if t.is_empty() || scope.is_empty() || crate::utils::hash::is_valid_relay_key(t) {
        return t.to_string();
    }
    format!("{}:{}", scope, t)
}

pub fn relay_type_key(type_: &str) -> String {
    let t = type_.trim();
    if let Some(code) = crate::utils::bias_schema::canonical_trade_doc_code(t) {
        return code.to_string();
    }
    let family = crate::utils::canonical::relay_type_family(t);
    match family.as_str() {
        "goods" | "order" | "tracking" => family,
        _ => t.to_lowercase(),
    }
}

pub fn relay_seed(type_key: &str, scope: &str, raw: &str) -> String {
    let t = raw.trim();
    if t.is_empty() {
        return String::new();
    }
    if crate::utils::bias_schema::is_trade_doc_type(type_key) {
        return t.to_string();
    }
    if type_key == "tracking" && crate::utils::hash::is_valid_relay_key(t) {
        return t.to_string();
    }
    let s = scope.trim();
    if s.is_empty() {
        t.to_string()
    } else {
        format!("{}:{}", s, t)
    }
}

pub fn entity_key_index(type_: &str, team_id: &str, scope: &str, raw: &str) -> u32 {
    let key = relay_type_key(type_);
    entity_index(&key, team_id, &relay_seed(&key, scope, raw))
}
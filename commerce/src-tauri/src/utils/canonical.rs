#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonKind {
    Identifier, // String 확정
    Numeric,    // Number 확정
    Boolean,    // 0|1 정수 확정
    Tags,       // 배열 확정 (멀티엔트리 인덱스)
    Free,       // 손대지 않음
}

const FORCE_ID: &[&str] = &[
    "id", "no", "digest",
];
const FORCE_NUM: &[&str] = &[
    "status", "views", "created_at", "updated_at",
    "index", "goods", "order", "tracking", "event",
];
const FORCE_BOOL: &[&str] = &[
    "detail", "node", "embed",
];

// ── ② 접미사 / 부분일치 규칙 : 새 필드는 여기에 자동으로 걸립니다 ──
const ID_SUFFIX: &[&str] = &[
    "_no", "_code", "_number", "_id", "_sku", "_barcode", "_gtin", "_mpn",
];
const ID_CONTAINS: &[&str] = &[
    "code", "barcode", "gtin", "mpn", "sku", "reference_", "container", "seal",
];

const NUM_PREFIX: &[&str] = &["rel_"];
const NUM_SUFFIX: &[&str] = &[
    "_price", "_amount", "_fee", "_rate", "_count", "_qty", "_at",
    "_weight", "_volume", "_duration", "_limit", "_threshold", "_charges",
    "_kg", "_cbm", "_m3", "_usd", "_krw", "_eur", "_jpy", "_cny", "_gbp",
];
const NUM_CONTAINS: &[&str] = &[
    "price", "amount", "quantity", "discount", "weight", "volume",
    "shipping_fee", "usage_", "threshold", "exchange_rate", "package_count",
    "local_charges", "number_of_",
    "packages", "pieces",
    "measurement", "premium", "duty_", "dutiable", "balance", "flash_point",
    "tare_weight", "chargeable",
];
const NUM_EXACT: &[&str] = &[
    "width", "height", "length",
    // 🌟 단독 명사형 수치 축
    "premium", "rate", "debit", "credit", "dosage",
];
const BOOL_PREFIX: &[&str] = &["is_", "has_", "allow_", "use_"];

const BOOL_SUFFIX: &[&str] = &["_only", "_included", "_allowed", "_match"];

const TRUTHY_WORDS: &[&str] = &[
    "yes", "y", "on", "o", "true", "allowed", "permitted", "included", "taxable", "applicable", "available", "enabled",
    "예", "네", "있음", "허용", "가능", "포함", "포함됨", "적용", "적용됨", "과세", "사용", "사용함",
    "はい", "あり", "有り", "有", "許可", "可", "可能", "課税", "含む", "税込", "対象",
    "是", "允许", "允許", "可以", "包含", "含税", "含稅", "启用", "啟用",
    "ja", "inklusive", "inkl", "enthalten", "erlaubt", "verfugbar", "aktiviert", "steuerpflichtig",
    "si", "incluido", "incluida", "permitido", "disponible", "activado", "gravado",
    "oui", "inclus", "incluse", "autorise", "active", "taxe",
    "sim", "disponivel", "ativado", "tributavel",
    "incluso", "inclusa", "consentito", "disponibile", "attivo", "attivato", "imponibile",
    "inbegrepen", "toegestaan", "beschikbaar", "ingeschakeld", "belast",
    "ano", "vcetne", "povoleno", "dostupne", "zapnuto", "aktivni",
    "نعم", "متاح", "مسموح", "مشمول", "مفعل",
];

pub fn truthy_word(raw: &str) -> bool {
    let t = fold_word(raw);
    !t.is_empty() && TRUTHY_WORDS.iter().any(|w| *w == t)
}

/// 🌟 필드 이름만으로 저장 타입을 판정합니다.
///    새 필드는 대부분 접미사 규칙에 자동으로 걸리므로 Rust 수정이 불필요합니다.
pub fn kind_of(key: &str) -> CanonKind {
    let k = key.to_lowercase();

    if k == "tags" { return CanonKind::Tags; }

    if FORCE_ID.iter().any(|x| *x == k) { return CanonKind::Identifier; }
    if FORCE_NUM.iter().any(|x| *x == k) { return CanonKind::Numeric; }
    if FORCE_BOOL.iter().any(|x| *x == k) { return CanonKind::Boolean; }

    if NUM_PREFIX.iter().any(|p| k.starts_with(p)) { return CanonKind::Numeric; }

    if BOOL_PREFIX.iter().any(|p| k.starts_with(p)) { return CanonKind::Boolean; }
    if BOOL_SUFFIX.iter().any(|s| k.ends_with(s)) { return CanonKind::Boolean; }

    if NUM_EXACT.iter().any(|x| *x == k) { return CanonKind::Numeric; }
    if NUM_SUFFIX.iter().any(|s| k.ends_with(s)) { return CanonKind::Numeric; }

    if ID_SUFFIX.iter().any(|s| k.ends_with(s)) { return CanonKind::Identifier; }
    if ID_CONTAINS.iter().any(|c| k.contains(c)) { return CanonKind::Identifier; }

    if NUM_CONTAINS.iter().any(|c| k.contains(c)) { return CanonKind::Numeric; }

    CanonKind::Free
}

pub fn iso_to_epoch_ms(t: &str) -> Option<i64> {
    let b = t.as_bytes();
    if t.len() < 10 || b.get(4) != Some(&b'-') || b.get(7) != Some(&b'-') {
        return None;
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(t, "%Y-%m-%dT%H:%M:%S") {
        return Some(dt.and_utc().timestamp_millis());
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(t, "%Y-%m-%d %H:%M:%S") {
        return Some(dt.and_utc().timestamp_millis());
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(&t[..10], "%Y-%m-%d") {
        return d.and_hms_opt(0, 0, 0).map(|x| x.and_utc().timestamp_millis());
    }
    None
}

pub const RELAY_INDEX_KEYS: &[&str] = &["goods", "order", "tracking", "event"];

pub fn is_relay_index_key(key: &str) -> bool {
    let k = key.trim().to_lowercase();
    RELAY_INDEX_KEYS.iter().any(|x| *x == k)
}

pub fn relay_text_is_content(s: &str) -> bool {
    let t = s.trim();
    !is_null_like(t) && t.chars().any(|c| c.is_alphabetic())
}

pub fn relay_value_is_content(v: &serde_json::Value) -> bool {
    v.as_str().map_or(false, relay_text_is_content)
}

pub const RELAY_IDENTITY_KEYS: &[&str] = &["id", "index"];

pub const RELAY_ZERO_EMPTY_KEYS: &[&str] = &[
    "goods", "order", "tracking", "event", "status", "index", "created_at", "updated_at",
    "width", "height", "length", "weight",
];

pub fn relay_value_is_placeholder(field: &str, v: &serde_json::Value) -> bool {
    let zero_empty = RELAY_ZERO_EMPTY_KEYS.iter().any(|k| *k == field);
    match v {
        serde_json::Value::Null => true,
        serde_json::Value::String(s) => {
            let t = s.trim();
            is_null_like(t) || (zero_empty && t.parse::<f64>().map_or(false, |x| x == 0.0))
        }
        serde_json::Value::Number(n) => zero_empty && n.as_f64() == Some(0.0),
        serde_json::Value::Array(a) => a.is_empty(),
        serde_json::Value::Object(o) => o.is_empty(),
        serde_json::Value::Bool(b) => zero_empty && !*b,
    }
}

pub const RELAY_LINK_KEYS: &[&str] = &["goods", "order", "tracking", "event"];

pub fn relay_key_is_empty(v: &serde_json::Value) -> bool {
    relay_value_is_placeholder("index", v)
}

pub fn relay_type_family(t: &str) -> String {
    match t.trim().to_lowercase().as_str() {
        "receiving" | "shipping" | "tracking" => "tracking".to_string(),
        "sales" | "order" => "order".to_string(),
        "coupon" | "event" => "event".to_string(),
        other => other.to_string(),
    }
}

pub fn relay_type_matches(expected: &str, found: &serde_json::Value) -> bool {
    match found.get("type").and_then(|v| v.as_str()) {
        Some(t) if !t.trim().is_empty() => relay_type_family(t) == relay_type_family(expected),
        _ => true,
    }
}

#[derive(Debug, Default, Clone)]
pub struct RelayWriteLog {
    pub written: Vec<String>,
    pub kept: Vec<String>,
}

pub fn relay_write(
    dst: &mut serde_json::Value,
    field: &str,
    val: serde_json::Value,
    overwrite: bool,
    log: &mut RelayWriteLog,
) -> bool {
    if relay_value_is_placeholder(field, &val) {
        return false;
    }
    let obj = match dst.as_object_mut() {
        Some(o) => o,
        None => return false,
    };
    let identity = RELAY_IDENTITY_KEYS.iter().any(|k| *k == field);
    let has_value = obj.get(field).map_or(false, |cur| !relay_value_is_placeholder(field, cur));
    let anchored = has_value && (RELAY_LINK_KEYS.iter().any(|k| *k == field) || !overwrite);
    if identity || anchored {
        if obj.get(field) != Some(&val) && !log.kept.iter().any(|f| f == field) {
            log.kept.push(field.to_string());
        }
        return false;
    }
    if obj.get(field) == Some(&val) {
        return false;
    }
    obj.insert(field.to_string(), val);
    if !log.written.iter().any(|f| f == field) {
        log.written.push(field.to_string());
    }
    true
}

pub fn relay_key_for_type(t: &str) -> Option<&'static str> {
    match relay_type_family(t).as_str() {
        "goods" => Some("goods"),
        "order" => Some("order"),
        "tracking" => Some("tracking"),
        "event" => Some("event"),
        _ => None,
    }
}

pub fn relay_ref_index(v: Option<&serde_json::Value>) -> Option<u32> {
    let n = v?.as_u64()?;
    if n == 0 || n > u64::from(u32::MAX) {
        None
    } else {
        Some(n as u32)
    }
}

pub fn relay_companion_base(key: &str) -> Option<&'static str> {
    let k = key.trim().to_lowercase();
    RELAY_LINK_KEYS
        .iter()
        .copied()
        .find(|base| k.len() == base.len() + 6 && k.starts_with(*base) && k.ends_with("_title"))
}

pub fn is_relay_placeholder(doc: &serde_json::Value) -> bool {
    let updated_zero = doc.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(0) == 0;
    let digest_empty = doc
        .get("digest")
        .and_then(|v| v.as_str())
        .map_or(true, |s| s.trim().is_empty());
    updated_zero && digest_empty
}

pub const LEDGER_KEY: &str = "ledger";
pub const RELAY_BOUND_KEY: &str = "_relay_bound";
pub const RELAY_ORIGIN_KEY: &str = "relay_origin";
pub const RELAY_TRANSIENT_KEYS: &[&str] = &[RELAY_BOUND_KEY, RELAY_ORIGIN_KEY];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LedgerPrior {
    Absent,
    Placeholder,
    Draft,
    Confirmed,
}

pub fn ledger_prior(doc: Option<&serde_json::Value>) -> LedgerPrior {
    let d = match doc {
        None => return LedgerPrior::Absent,
        Some(d) => d,
    };
    match d.get(LEDGER_KEY).and_then(|v| v.as_str()).map(|s| s.trim()) {
        Some("count") => return LedgerPrior::Confirmed,
        Some("draft") => return LedgerPrior::Draft,
        Some("placeholder") => return LedgerPrior::Placeholder,
        _ => {}
    }
    if d.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(0) > 0 {
        LedgerPrior::Confirmed
    } else if is_relay_placeholder(d) {
        LedgerPrior::Placeholder
    } else {
        LedgerPrior::Draft
    }
}

pub fn ledger_state(prior: LedgerPrior, confirm: bool) -> &'static str {
    if confirm || prior == LedgerPrior::Confirmed {
        "count"
    } else {
        "draft"
    }
}

pub fn ledger_delta(prior: LedgerPrior, confirm: bool) -> (i64, i64, i64) {
    match (prior, confirm) {
        (LedgerPrior::Absent, false) => (1, 0, 1),
        (LedgerPrior::Absent, true) => (0, 1, 1),
        (LedgerPrior::Placeholder, false) => (0, 0, 1),
        (LedgerPrior::Placeholder, true) => (-1, 1, 1),
        (LedgerPrior::Draft, false) => (0, 0, 0),
        (LedgerPrior::Draft, true) => (-1, 1, 0),
        (LedgerPrior::Confirmed, _) => (0, 0, 0),
    }
}

pub const LEDGER_PLACEHOLDER_DELTA: (i64, i64, i64) = (1, 0, 0);

pub fn relay_establishes(target_type: &str, by_type: &str) -> bool {
    let t = relay_type_family(target_type);
    let b = relay_type_family(by_type);
    if t.is_empty() || b.is_empty() || t == b {
        return false;
    }
    match t.as_str() {
        "goods" => b == "order" || b == "tracking",
        "order" => b == "goods" || b == "tracking",
        "tracking" => b == "order" || b == "goods",
        "event" => b == "goods" || b == "order",
        "review" => b == "goods",
        _ => true,
    }
}

pub fn relay_edge_target(key: &str) -> Option<String> {
    let k = key.trim();
    if let Some(code) = k.strip_prefix("rel_") {
        let c = code.trim();
        return if c.is_empty() { None } else { Some(c.to_uppercase()) };
    }
    if RELAY_LINK_KEYS.iter().any(|x| *x == k) {
        return Some(k.to_string());
    }
    None
}

fn push_edge(out: &mut Vec<(String, u32)>, key: &str, index: u32) {
    if !out.iter().any(|(k, i)| k == key && *i == index) {
        out.push((key.to_string(), index));
    }
}

pub fn relay_edges(doc: &serde_json::Value) -> Vec<(String, u32)> {
    let mut out: Vec<(String, u32)> = Vec::new();
    let obj = match doc.as_object() {
        Some(o) => o,
        None => return out,
    };
    let own_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let own_family = relay_type_family(own_type);
    let own_code = own_type.trim().to_uppercase();
    for key in RELAY_LINK_KEYS.iter() {
        if *key == own_family.as_str() {
            continue;
        }
        match obj.get(*key) {
            Some(serde_json::Value::Array(arr)) => {
                for el in arr.iter() {
                    if let Some(i) = relay_ref_index(el.get("index")) {
                        push_edge(&mut out, key, i);
                    }
                }
            }
            other => {
                if let Some(i) = relay_ref_index(other) {
                    push_edge(&mut out, key, i);
                }
            }
        }
    }
    for (k, v) in obj.iter() {
        let code = match k.strip_prefix("rel_") {
            Some(c) => c.trim().to_uppercase(),
            None => continue,
        };
        if code.is_empty() || code == own_code {
            continue;
        }
        if let Some(i) = relay_ref_index(Some(v)) {
            push_edge(&mut out, k, i);
        }
    }
    out
}

pub fn keep_relay_index(
    prior: &serde_json::Value,
    merged: &mut serde_json::Value,
    page_type: &str,
) -> Vec<(String, u32)> {
    let own = relay_type_family(page_type);
    let mut kept: Vec<(String, u32)> = Vec::new();
    let obj = match merged.as_object_mut() {
        Some(o) => o,
        None => return kept,
    };
    for key in RELAY_LINK_KEYS.iter() {
        if *key == own.as_str() {
            continue;
        }
        let idx = match relay_ref_index(prior.get(*key)) {
            Some(i) => i,
            None => continue,
        };
        let text = match obj.get(*key) {
            Some(serde_json::Value::Array(_)) | Some(serde_json::Value::Object(_)) => continue,
            Some(v) if relay_ref_index(Some(v)).is_some() => continue,
            Some(serde_json::Value::String(s)) if relay_text_is_content(s) => Some(s.trim().to_string()),
            _ => None,
        };
        if let Some(t) = text {
            let companion = format!("{}_title", key);
            let companion_empty = obj
                .get(&companion)
                .and_then(|v| v.as_str())
                .map_or(true, |s| s.trim().is_empty());
            if companion_empty {
                obj.insert(companion, serde_json::Value::String(t));
            }
        }
        obj.insert(key.to_string(), serde_json::Value::from(idx));
        kept.push((key.to_string(), idx));
    }
    kept
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueShape {
    Url,
    Code,
    Quantity,
    Text,
}

pub fn value_shape(key: &str, v: &serde_json::Value) -> Option<ValueShape> {
    let t = match v {
        serde_json::Value::Number(_) => {
            return Some(if kind_of(key) == CanonKind::Identifier { ValueShape::Code } else { ValueShape::Quantity });
        }
        serde_json::Value::String(s) => s.trim(),
        _ => return None,
    };
    if t.is_empty() || t.eq_ignore_ascii_case("null") || t.eq_ignore_ascii_case("n/a") {
        return None;
    }
    let digits = number_text(t);
    let spaced = t.chars().any(|c| c.is_whitespace());
    if (t.starts_with("http://") || t.starts_with("https://") || t.starts_with('/')) && !spaced {
        return Some(ValueShape::Url);
    }
    if kind_of(key) == CanonKind::Identifier {
        let code = !spaced && t.chars().count() >= 3 && digits.chars().any(|c| c.is_ascii_digit());
        return Some(if code { ValueShape::Code } else { ValueShape::Text });
    }
    if iso_to_epoch_ms(t).is_some() {
        return Some(ValueShape::Quantity);
    }
    let runs = crate::utils::ai_utils::numeric_run_count(&digits);
    let letters = t.chars().filter(|c| c.is_alphabetic()).count();
    if runs == 1 && letters <= 3 {
        return Some(ValueShape::Quantity);
    }
    Some(ValueShape::Text)
}

pub const SHAPE_GUARD_SKIP: &[&str] = &[
    "id", "index", "type", "mode", "status", "text", "masked_text", "digest", "ledger",
    "currency", "flag", "embed", "detail", "updated_at", "created_at",
];

pub fn keep_value_shapes(prior: &serde_json::Value, merged: &mut serde_json::Value) -> Vec<String> {
    let mut kept: Vec<String> = Vec::new();
    let (p, m) = match (prior.as_object(), merged.as_object_mut()) {
        (Some(p), Some(m)) => (p, m),
        _ => return kept,
    };
    for (k, pv) in p.iter() {
        let lk = k.to_lowercase();
        if SHAPE_GUARD_SKIP.iter().any(|s| *s == lk)
            || is_relay_index_key(&lk)
            || lk.starts_with("rel_")
            || lk.starts_with('_')
            || lk.contains("insight")
        {
            continue;
        }
        let before = match value_shape(&lk, pv) {
            Some(ValueShape::Text) | None => continue,
            Some(s) => s,
        };
        let after = match m.get(k).and_then(|mv| value_shape(&lk, mv)) {
            Some(s) => s,
            None => continue,
        };
        if after != before {
            m.insert(k.clone(), pv.clone());
            kept.push(k.clone());
        }
    }
    kept
}

pub fn is_latin_letter(c: char) -> bool {
    c.is_ascii_alphabetic()
        || (c.is_alphabetic() && matches!(c as u32, 0x00C0..=0x024F | 0x1E00..=0x1EFF))
}

pub fn is_latin_script_text(s: &str) -> bool {
    let mut letters = 0usize;
    for c in s.chars() {
        if !c.is_alphabetic() {
            continue;
        }
        if !is_latin_letter(c) {
            return false;
        }
        letters += 1;
    }
    letters > 0
}

pub fn fold_text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        let u = c as u32;
        if (0x064B..=0x065F).contains(&u) || matches!(u, 0x0640 | 0x0670 | 0x061C | 0x200E | 0x200F) {
            continue;
        }
        if !c.is_ascii() && is_latin_letter(c) {
            let mut buf = [0u8; 4];
            out.push_str(&any_ascii::any_ascii(c.encode_utf8(&mut buf)).to_lowercase());
        } else {
            out.extend(c.to_lowercase());
        }
    }
    out
}

fn fold_word(raw: &str) -> String {
    fold_text(raw.trim())
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_string()
}

const MISSING_WORDS: &[&str] = &[
    "null", "n/a", "n.a", "not available", "unknown", "undefined", "tbd", "tba",
    "미입력", "미정", "미등록", "정보없음", "정보 없음", "모름",
    "不明", "未定", "未入力", "未知", "不详", "不詳",
    "k.a", "keine angabe", "unbekannt", "n.v",
    "sin datos", "desconocido", "no disponible", "n/d",
    "inconnu", "non disponible", "n.c",
    "desconhecido", "nao disponivel",
    "sconosciuto", "non disponibile",
    "onbekend", "niet beschikbaar", "n.v.t",
    "neznamy", "neni k dispozici",
    "غير متوفر", "غير معروف",
];

const NONE_WORDS: &[&str] = &[
    "none", "nil", "없음", "해당없음", "해당 없음",
    "なし", "無し", "該当なし", "无", "無", "暂无", "暫無",
    "keine", "kein", "ninguno", "ninguna", "aucun", "aucune", "neant",
    "nenhum", "nenhuma", "nessuno", "nessuna", "geen", "zadny", "zadna", "zadne",
    "لا يوجد", "لا شيء",
];

const CATCH_ALL_WORDS: &[&str] = &[
    "기타", "기타 상품", "기타 재화", "기타재화", "그 외", "그외",
    "その他", "そのほか", "其他", "其它",
    "other", "others", "etc", "misc", "miscellaneous",
    "sonstige", "sonstiges", "andere", "anderes", "diverses", "verschiedenes",
    "otro", "otros", "otra", "otras", "varios", "varias",
    "autre", "autres", "divers",
    "outro", "outros", "outra", "outras", "diversos",
    "altro", "altri", "altra", "altre", "varie", "vari",
    "overig", "overige", "anders", "diversen",
    "ostatni", "jine", "ruzne",
    "أخرى", "اخرى", "غير ذلك", "متنوع", "متفرقات",
];

fn is_zero_date(t: &str) -> bool {
    t.chars().filter(|c| *c == '0').count() >= 6
        && t.chars().all(|c| c == '0' || matches!(c, '-' | '/' | '.' | ':' | ' ' | 'T'))
}

pub fn is_missing_marker(s: &str) -> bool {
    let raw = s.trim();
    if is_zero_date(raw) {
        return true;
    }
    let t = fold_word(raw);
    t.is_empty() || MISSING_WORDS.iter().any(|w| *w == t)
}

pub fn is_null_like(s: &str) -> bool {
    if is_missing_marker(s) {
        return true;
    }
    let t = fold_word(s);
    NONE_WORDS.iter().any(|w| *w == t)
}

pub fn is_catch_all_value(s: &str) -> bool {
    let t = fold_word(s);
    !t.is_empty() && CATCH_ALL_WORDS.iter().any(|w| *w == t)
}

const SEE_MARKERS: &[&str] = &[
    "참고", "참조", "확인", "기재", "표기", "표시",
    "参照", "参考", "確認", "详见", "詳見", "请见", "請見",
    "see", "refer", "siehe", "ver", "vea", "veja", "consulte", "consultar",
    "voir", "consulter", "consultez", "vedi", "vedere", "consultare", "zie", "raadpleeg", "viz",
    "انظر", "راجع",
];

const SEE_TARGETS: &[&str] = &[
    "페이지", "상세", "설명", "본문", "이미지", "첨부",
    "ページ", "詳細", "説明", "画像", "別紙", "页面", "頁面", "详情", "詳情", "描述", "附件", "页", "頁",
    "page", "description", "details", "detail", "listing", "image", "images", "attached", "attachment",
    "seite", "produktseite", "artikelseite", "detailseite", "beschreibung", "produktbeschreibung",
    "artikelbeschreibung", "bild", "anlage",
    "pagina", "descripcion", "detalles", "imagen", "anexo",
    "fiche", "descriptif", "annexe",
    "descricao", "detalhes", "imagem",
    "descrizione", "dettagli", "immagine", "scheda", "allegato",
    "productpagina", "beschrijving", "afbeelding", "bijlage",
    "stranka", "stranku", "popis", "obrazek", "priloha",
    "صفحة", "الصفحة", "وصف", "الوصف", "التفاصيل", "صورة", "مرفق", "المرفق",
];

const SEE_FILLERS: &[&str] = &[
    "상품", "제품", "해당", "별도", "바랍니다", "바람", "부탁드립니다", "해주세요", "주세요", "요망", "하세요",
    "를", "을", "의", "에서", "에",
    "商品", "製品", "产品", "產品", "の", "を", "ご", "下さい", "ください", "请", "請",
    "the", "product", "item", "for", "on", "our", "please", "more", "info", "information", "full", "to",
    "das", "die", "der", "den", "dem", "des", "bitte", "produkt", "produkts", "artikel", "zur", "zum",
    "el", "la", "los", "las", "de", "del", "en", "producto", "articulo", "por", "favor",
    "le", "les", "du", "produit", "article", "svp", "veuillez",
    "o", "a", "os", "as", "do", "da", "na", "no", "produto", "artigo",
    "il", "lo", "gli", "della", "dello", "di", "prodotto", "articolo",
    "het", "van", "op",
    "produktu", "zbozi", "prosim",
    "المنتج", "منتج", "يرجى", "من", "في",
];

fn spaced_token(tk: &str) -> bool {
    tk.chars().all(|c| c.is_ascii() || ('\u{0600}'..='\u{06FF}').contains(&c))
}

pub fn is_see_elsewhere_placeholder(s: &str) -> bool {
    let t = fold_text(s.trim());
    if t.is_empty() || t.chars().count() > 40 || t.chars().any(|c| c.is_ascii_digit()) {
        return false;
    }
    let toks: Vec<&str> = t
        .split(|c: char| !c.is_alphanumeric())
        .filter(|x| !x.is_empty())
        .collect();
    if toks.is_empty() || toks.len() > 6 {
        return false;
    }
    let mut marker = false;
    let mut target = false;
    for tk in toks.iter() {
        if spaced_token(tk) {
            if SEE_MARKERS.contains(tk) {
                marker = true;
            } else if SEE_TARGETS.contains(tk) {
                target = true;
            } else if !SEE_FILLERS.contains(tk) {
                return false;
            }
            continue;
        }
        let mut rest = tk.to_string();
        for m in SEE_MARKERS.iter().filter(|m| !spaced_token(m)) {
            if rest.contains(*m) {
                marker = true;
                rest = rest.replace(*m, "");
            }
        }
        for g in SEE_TARGETS.iter().filter(|g| !spaced_token(g)) {
            if rest.contains(*g) {
                target = true;
                rest = rest.replace(*g, "");
            }
        }
        for f in SEE_FILLERS.iter().filter(|f| !spaced_token(f)) {
            rest = rest.replace(*f, "");
        }
        if !rest.is_empty() {
            return false;
        }
    }
    marker && target
}

pub fn is_uninformative_text(s: &str) -> bool {
    let t = s.trim();
    t.is_empty() || is_missing_marker(t) || is_see_elsewhere_placeholder(t) || is_catch_all_value(t)
}

pub fn is_structural_key(lk: &str) -> bool {
    SHAPE_GUARD_SKIP.iter().any(|s| *s == lk)
        || is_relay_index_key(lk)
        || lk.starts_with("rel_")
        || lk.starts_with('_')
        || lk.contains("insight")
        || lk == "title"
        || lk == "link"
}

pub fn drop_placeholder_values(v: &mut serde_json::Value) -> Vec<String> {
    let mut dropped: Vec<String> = Vec::new();
    match v {
        serde_json::Value::Object(obj) => {
            let keys: Vec<String> = obj.keys().cloned().collect();
            for k in keys {
                if is_structural_key(&k.to_lowercase()) {
                    continue;
                }
                let (drop, nested) = match obj.get(&k) {
                    Some(serde_json::Value::String(s)) => {
                        let t = s.trim();
                        (!t.is_empty() && (is_missing_marker(t) || is_see_elsewhere_placeholder(t)), false)
                    }
                    Some(serde_json::Value::Object(_)) | Some(serde_json::Value::Array(_)) => (false, true),
                    _ => (false, false),
                };
                if drop {
                    obj.remove(&k);
                    dropped.push(k);
                } else if nested {
                    if let Some(child) = obj.get_mut(&k) {
                        for d in drop_placeholder_values(child) {
                            dropped.push(format!("{}.{}", k, d));
                        }
                    }
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for (i, it) in arr.iter_mut().enumerate() {
                for d in drop_placeholder_values(it) {
                    dropped.push(format!("[{}].{}", i, d));
                }
            }
        }
        _ => {}
    }
    dropped
}

pub fn keep_informative_values(prior: &serde_json::Value, merged: &mut serde_json::Value) -> Vec<String> {
    let mut kept: Vec<String> = Vec::new();
    let (p, m) = match (prior.as_object(), merged.as_object_mut()) {
        (Some(p), Some(m)) => (p, m),
        _ => return kept,
    };
    for (k, pv) in p.iter() {
        if is_structural_key(&k.to_lowercase()) {
            continue;
        }
        let prior_ok = match pv {
            serde_json::Value::String(s) => !is_uninformative_text(s),
            serde_json::Value::Number(_) => true,
            _ => false,
        };
        if !prior_ok {
            continue;
        }
        let lose = match m.get(k) {
            Some(serde_json::Value::String(ms)) => is_uninformative_text(ms),
            Some(serde_json::Value::Null) => true,
            _ => false,
        };
        if lose {
            m.insert(k.clone(), pv.clone());
            kept.push(k.clone());
        }
    }
    kept
}

fn timestamp_grain(s: &str) -> Option<(String, u8)> {
    use chrono::Timelike;
    let t = s.trim();
    let date = t.get(..10)?;
    chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").ok()?;
    let rest = t.get(10..)?.trim_start_matches(|c: char| c == 'T' || c == ' ');
    if rest.is_empty() {
        return Some((date.to_string(), 0));
    }
    let hms = rest.get(..8)?;
    let tail = rest.get(8..)?.trim();
    if !(tail.is_empty() || tail == "Z") {
        return None;
    }
    let tm = chrono::NaiveTime::parse_from_str(hms, "%H:%M:%S").ok()?;
    if tm.hour() == 0 && tm.minute() == 0 && tm.second() == 0 {
        Some((date.to_string(), 0))
    } else if tm.second() == 0 {
        Some((format!("{}T{}", date, hms.get(..5)?), 1))
    } else {
        Some((format!("{}T{}", date, hms), 2))
    }
}

fn grain_start_ms(prefix: &str, grain: u8) -> Option<i64> {
    let dt = match grain {
        0 => chrono::NaiveDate::parse_from_str(prefix, "%Y-%m-%d").ok()?.and_hms_opt(0, 0, 0)?,
        1 => chrono::NaiveDateTime::parse_from_str(&format!("{}:00", prefix), "%Y-%m-%dT%H:%M:%S").ok()?,
        _ => return None,
    };
    Some(dt.and_utc().timestamp_millis())
}

pub fn is_coarser_timestamp(incoming: &str, prior: &str) -> bool {
    match (timestamp_grain(incoming), timestamp_grain(prior)) {
        (Some((ip, ig)), Some((pp, pg))) => ig < pg && pp.starts_with(&ip),
        _ => false,
    }
}

fn finer_epoch_value(incoming: &str, prior_ms: i64) -> Option<serde_json::Value> {
    let (prefix, grain) = timestamp_grain(incoming)?;
    let start = grain_start_ms(&prefix, grain)?;
    let span: i64 = if grain == 0 { 86_400_000 } else { 60_000 };
    if prior_ms <= start || prior_ms >= start + span {
        return None;
    }
    if prior_ms % 1000 != 0 {
        return Some(serde_json::Value::from(prior_ms));
    }
    let dt = chrono::DateTime::from_timestamp_millis(prior_ms)?.naive_utc();
    Some(serde_json::Value::String(dt.format("%Y-%m-%dT%H:%M:%S").to_string()))
}

pub fn epoch_field_text(key: &str, n: &serde_json::Number) -> Option<String> {
    let k = key.trim().to_lowercase();
    if !k.ends_with("_at") || kind_of(&k) != CanonKind::Numeric {
        return None;
    }
    let ms = n.as_i64()?;
    if !(946_684_800_000..4_102_444_800_000).contains(&ms) {
        return None;
    }
    let dt = chrono::DateTime::from_timestamp_millis(ms)?.naive_utc();
    Some(dt.format("%Y-%m-%dT%H:%M:%S").to_string())
}

pub fn keep_finer_timestamps(prior: &serde_json::Value, merged: &mut serde_json::Value) -> Vec<String> {
    let mut kept: Vec<String> = Vec::new();
    let (p, m) = match (prior.as_object(), merged.as_object_mut()) {
        (Some(p), Some(m)) => (p, m),
        _ => return kept,
    };
    for (k, pv) in p.iter() {
        let incoming = match m.get(k).and_then(|mv| mv.as_str()) {
            Some(s) => s.trim().to_string(),
            None => continue,
        };
        let keep = match pv {
            serde_json::Value::String(ps) => {
                let ps = ps.trim();
                if ps != incoming && is_coarser_timestamp(&incoming, ps) {
                    Some(pv.clone())
                } else {
                    None
                }
            }
            serde_json::Value::Number(n) if kind_of(k) == CanonKind::Numeric && k != "created_at" && k != "updated_at" => {
                n.as_i64().and_then(|ms| finer_epoch_value(&incoming, ms))
            }
            _ => None,
        };
        if let Some(v) = keep {
            m.insert(k.clone(), v);
            kept.push(k.clone());
        }
    }
    kept
}

pub fn number_text(s: &str) -> String {
    let src: Vec<char> = crate::utils::ai_utils::normalize_digits_ascii(s)
        .chars()
        .map(|c| match c {
            '\u{066B}' | '\u{FF0E}' => '.',
            '\u{066C}' | '\u{FF0C}' => ',',
            '\u{2212}' | '\u{FF0D}' => '-',
            _ => c,
        })
        .collect();
    let mut out = String::with_capacity(src.len());
    let mut group = 0usize;
    for (i, &c) in src.iter().enumerate() {
        if c.is_ascii_digit() {
            group += 1;
            out.push(c);
            continue;
        }
        let spacer = matches!(c, ' ' | '\u{00A0}' | '\u{202F}' | '\u{2009}' | '\'' | '\u{2019}');
        if spacer && (1..=3).contains(&group) {
            let next3 = src.get(i + 1..i + 4).map_or(false, |w| w.iter().all(|d| d.is_ascii_digit()));
            let after = src.get(i + 4).map_or(true, |d| !d.is_ascii_digit());
            if next3 && after {
                group = 0;
                continue;
            }
        }
        group = 0;
        out.push(c);
    }
    out
}

fn first_number_run(chars: &[char]) -> Option<(usize, usize)> {
    let mut start = chars.iter().position(|c| c.is_ascii_digit())?;
    if start > 0
        && (chars[start - 1] == '.' || chars[start - 1] == ',')
        && (start == 1 || !chars[start - 2].is_alphanumeric())
    {
        start -= 1;
    }
    let mut end = start + 1;
    while end < chars.len() {
        let c = chars[end];
        let sep = (c == '.' || c == ',') && chars.get(end + 1).map_or(false, |n| n.is_ascii_digit());
        if c.is_ascii_digit() || sep {
            end += 1;
        } else {
            break;
        }
    }
    Some((start, end))
}

fn minus_before(chars: &[char], start: usize) -> bool {
    let mut j = start;
    while j > 0 && chars[j - 1] == ' ' {
        j -= 1;
    }
    j > 0 && chars[j - 1] == '-' && (j < 2 || !chars[j - 2].is_alphanumeric())
}

fn resolve_separators(run: &str, decimal_comma: bool) -> Option<f64> {
    let run = if run.starts_with('.') || run.starts_with(',') {
        format!("0{}", run)
    } else {
        run.to_string()
    };
    let grouped = |sep: char| -> bool {
        let parts: Vec<&str> = run.split(sep).collect();
        parts.len() > 1
            && (1..=3).contains(&parts[0].len())
            && !parts[0].starts_with('0')
            && parts[1..].iter().all(|p| p.len() == 3)
    };
    let dots = run.matches('.').count();
    let commas = run.matches(',').count();
    let plain = if dots > 0 && commas > 0 {
        let last_dot = run.rfind('.')?;
        let last_comma = run.rfind(',')?;
        if last_comma > last_dot {
            run.replace('.', "").replace(',', ".")
        } else {
            run.replace(',', "")
        }
    } else if commas > 1 {
        run.replace(',', "")
    } else if commas == 1 {
        if !decimal_comma && grouped(',') {
            run.replace(',', "")
        } else {
            run.replace(',', ".")
        }
    } else if dots > 1 {
        if grouped('.') {
            run.replace('.', "")
        } else {
            return None;
        }
    } else if dots == 1 && decimal_comma && grouped('.') {
        run.replace('.', "")
    } else {
        run.clone()
    };
    plain.parse::<f64>().ok().filter(|v| v.is_finite())
}

fn cjk_multiplier(c: char) -> Option<f64> {
    match c {
        '百' | '백' => Some(1e2),
        '千' | '천' => Some(1e3),
        '万' | '萬' | '만' => Some(1e4),
        '億' | '亿' | '억' => Some(1e8),
        _ => None,
    }
}

pub fn parse_number_run(s: &str, decimal_comma: bool) -> Option<f64> {
    let chars: Vec<char> = number_text(s).chars().collect();
    let (start, end) = first_number_run(&chars)?;
    let run: String = chars[start..end].iter().collect();
    let mut v = resolve_separators(&run, decimal_comma)?;
    let mut i = end;
    while let Some(m) = chars.get(i).and_then(|c| cjk_multiplier(*c)) {
        v *= m;
        i += 1;
    }
    Some(if minus_before(&chars, start) { -v } else { v })
}

pub fn normalize_number_lexeme(s: &str, decimal_comma: bool) -> Option<String> {
    let t = number_text(s);
    if crate::utils::ai_utils::numeric_run_count(&t) != 1 {
        return None;
    }
    let chars: Vec<char> = t.chars().collect();
    let (start, end) = first_number_run(&chars)?;
    let run: String = chars[start..end].iter().collect();
    let v = resolve_separators(&run, decimal_comma)?;
    let canon = if v.fract() == 0.0 && v.abs() < 9e15 { format!("{}", v as i64) } else { format!("{}", v) };
    if canon == run {
        return None;
    }
    let head: String = chars[..start].iter().collect();
    let tail: String = chars[end..].iter().collect();
    Some(format!("{}{}{}", head, canon, tail))
}

fn lang_base(lang: &str) -> String {
    lang.trim()
        .to_lowercase()
        .split(|c: char| c == '-' || c == '_')
        .next()
        .unwrap_or("")
        .to_string()
}

pub fn decimal_comma_lang(lang: &str) -> bool {
    matches!(lang_base(lang).as_str(), "de" | "es" | "fr" | "pt" | "it" | "nl" | "cs")
}

pub fn day_first_lang(lang: &str) -> bool {
    matches!(lang_base(lang).as_str(), "de" | "es" | "fr" | "pt" | "it" | "nl" | "cs" | "ar")
}

pub fn point_decimal_currency(code: &str) -> bool {
    matches!(
        code.trim().to_uppercase().as_str(),
        "USD" | "GBP" | "MXN" | "JPY" | "CNY" | "KRW" | "HKD" | "TWD" | "SGD" | "AUD" | "INR" | "CHF" | "NZD"
            | "ILS" | "THB" | "PHP" | "MYR"
    )
}

pub fn numeric_date_to_iso(s: &str, doc_lang: &str) -> Option<String> {
    let t = crate::utils::ai_utils::normalize_digits_ascii(s.trim());
    if t.is_empty() || is_zero_date(&t) {
        return None;
    }
    let mut runs: Vec<(u32, usize)> = Vec::new();
    let mut month_word: Option<u32> = None;
    let mut num = String::new();
    let mut word = String::new();
    for c in t.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_digit() {
            num.push(c);
        } else if !num.is_empty() {
            runs.push((num.parse().ok()?, num.len()));
            num.clear();
        }
        if c.is_alphabetic() {
            word.push(c);
        } else if !word.is_empty() {
            if let Some(m) = crate::utils::ai_utils::month_from_name(&word) {
                month_word = Some(m);
            }
            word.clear();
        }
    }
    if month_word.is_none() && runs.len() == 1 && runs[0].1 == 8 {
        let v = runs[0].0;
        let date = chrono::NaiveDate::from_ymd_opt((v / 10000) as i32, (v / 100) % 100, v % 100)?;
        return Some(format!("{}T00:00:00", date.format("%Y-%m-%d")));
    }
    let (year, month, day, time_from) = if let Some(m) = month_word {
        let a = *runs.first()?;
        let b = *runs.get(1)?;
        let (y, d) = if a.1 >= 3 || a.0 > 31 { (a.0, b.0) } else { (b.0, a.0) };
        (y, m, d, 2usize)
    } else {
        if runs.len() < 3 {
            return None;
        }
        let (a, b, c) = (runs[0], runs[1], runs[2]);
        let lang = lang_base(doc_lang);
        let cjk = matches!(lang.as_str(), "ko" | "ja" | "zh");
        let df_lang = day_first_lang(&lang);
        let dotted = t.contains('.') && !t.contains('/') && !t.contains('-');
        let dashed = t.contains('-') && !t.contains('/') && !t.contains('.');
        let (y, mo, d) = if a.1 >= 3 || a.0 > 31 {
            (a.0, b.0, c.0)
        } else if c.1 >= 3 || c.0 > 31 {
            let day_first = if a.0 > 12 {
                true
            } else if b.0 > 12 {
                false
            } else {
                dotted || df_lang
            };
            if day_first { (c.0, b.0, a.0) } else { (c.0, a.0, b.0) }
        } else if cjk || dashed || !(df_lang || lang == "en") {
            (a.0, b.0, c.0)
        } else if dotted || df_lang {
            (c.0, b.0, a.0)
        } else {
            (c.0, a.0, b.0)
        };
        (y, mo, d, 3usize)
    };
    let (mut year, mut month, mut day) = (year, month, day);
    if month > 12 && day <= 12 {
        std::mem::swap(&mut month, &mut day);
    }
    if year < 100 {
        year += if year > 50 { 1900 } else { 2000 };
    }
    let date = chrono::NaiveDate::from_ymd_opt(year as i32, month, day)?;
    let tail: Vec<u32> = runs.iter().skip(time_from).map(|(v, _)| *v).collect();
    let mut hour = tail.first().copied().filter(|h| *h <= 23).unwrap_or(0);
    if !tail.is_empty() && matches!(lang_base(doc_lang).as_str(), "en" | "ko" | "ja" | "zh") {
        let low = t.to_lowercase();
        let toks: Vec<&str> = low.split(|c: char| !c.is_alphanumeric()).collect();
        let pm = toks.contains(&"pm") || low.contains("p.m") || low.contains("오후") || low.contains("午後") || low.contains("下午");
        let am = toks.contains(&"am") || low.contains("a.m") || low.contains("오전") || low.contains("午前") || low.contains("上午");
        if pm && hour < 12 {
            hour += 12;
        } else if am && !pm && hour == 12 {
            hour = 0;
        }
    }
    let minute = tail.get(1).copied().filter(|m| *m <= 59).unwrap_or(0);
    let second = tail.get(2).copied().filter(|x| *x <= 59).unwrap_or(0);
    Some(format!("{}T{:02}:{:02}:{:02}", date.format("%Y-%m-%d"), hour, minute, second))
}

const SUPERLATIVE_TOKENS: &[&str] = &[
    "가장", "제일",
    "most", "least", "cheapest", "lowest", "highest", "priciest", "smallest", "largest", "biggest",
    "fewest", "heaviest", "lightest",
    "billigste", "billigsten", "gunstigste", "gunstigsten", "teuerste", "teuersten", "meiste", "meisten",
    "wenigste", "wenigsten", "hochste", "hochsten", "niedrigste", "niedrigsten", "grosste", "grossten",
    "kleinste", "kleinsten",
    "goedkoopste", "duurste", "meeste", "minste", "hoogste", "laagste", "grootste",
    "الأرخص", "الارخص", "الأغلى", "الاغلى", "الأكثر", "الاكثر", "الأقل", "الاقل", "الأعلى", "الاعلى",
    "الأدنى", "الادنى",
];

const SUPERLATIVE_PREFIXES: &[&str] = &[
    "최저", "최고가",
    "nejlevn", "nejdraz", "nejvic", "nejvets", "nejmen", "nejvys", "nejniz",
];

const SUPERLATIVE_SUBSTR: &[&str] = &[
    "最も", "一番", "いちばん", "最安", "最高値", "最安値",
    "最便宜", "最贵", "最貴", "最低", "最高", "最多", "最少", "最大", "最小",
];

const SUPERLATIVE_PAIRS: &[(&str, &str)] = &[
    ("el", "mas"), ("la", "mas"), ("los", "mas"), ("las", "mas"), ("lo", "mas"),
    ("el", "menos"), ("la", "menos"), ("los", "menos"), ("las", "menos"),
    ("le", "plus"), ("la", "plus"), ("les", "plus"), ("le", "moins"), ("la", "moins"), ("les", "moins"),
    ("il", "piu"), ("la", "piu"), ("i", "piu"), ("le", "piu"),
    ("il", "meno"), ("la", "meno"), ("i", "meno"), ("le", "meno"),
    ("o", "mais"), ("a", "mais"), ("os", "mais"), ("as", "mais"),
    ("o", "menos"), ("a", "menos"), ("os", "menos"), ("as", "menos"),
    ("het", "meest"), ("het", "minst"), ("am", "meisten"), ("am", "wenigsten"),
];

pub fn has_superlative_marker(query: &str) -> bool {
    let t = fold_text(query);
    if SUPERLATIVE_SUBSTR.iter().any(|m| t.contains(*m)) {
        return true;
    }
    let toks: Vec<&str> = t
        .split(|c: char| !c.is_alphanumeric())
        .filter(|x| !x.is_empty())
        .collect();
    if toks.iter().any(|tk| {
        SUPERLATIVE_TOKENS.contains(tk) || SUPERLATIVE_PREFIXES.iter().any(|p| tk.starts_with(*p))
    }) {
        return true;
    }
    toks.windows(2)
        .any(|w| SUPERLATIVE_PAIRS.iter().any(|(a, b)| w[0] == *a && w[1] == *b))
}
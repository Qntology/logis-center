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
    !t.is_empty()
        && !t.eq_ignore_ascii_case("null")
        && !t.eq_ignore_ascii_case("n/a")
        && t.chars().any(|c| c.is_alphabetic())
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
            t.is_empty()
                || t.eq_ignore_ascii_case("null")
                || t.eq_ignore_ascii_case("n/a")
                || (zero_empty && t.parse::<f64>().map_or(false, |x| x == 0.0))
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
    let spaced = t.chars().any(|c| c.is_whitespace());
    if (t.starts_with("http://") || t.starts_with("https://") || t.starts_with('/')) && !spaced {
        return Some(ValueShape::Url);
    }
    if kind_of(key) == CanonKind::Identifier {
        let code = !spaced && t.chars().count() >= 3 && t.chars().any(|c| c.is_ascii_digit());
        return Some(if code { ValueShape::Code } else { ValueShape::Text });
    }
    if iso_to_epoch_ms(t).is_some() {
        return Some(ValueShape::Quantity);
    }
    let runs = t
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == ','))
        .filter(|r| r.chars().any(|c| c.is_ascii_digit()))
        .count();
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
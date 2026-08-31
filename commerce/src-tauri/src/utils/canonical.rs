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
    "index", "goods", "order", "tracking",
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
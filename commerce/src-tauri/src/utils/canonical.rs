// =====================================================================
// 🌟 [CANONICAL CONTRACT]
//  data.* 값의 저장 타입을 확정하는 '단일 진실 공급원' 입니다.
//
//  ── 왜 별도 모듈인가 ──
//   기존에는 store.rs 의 canonicalize_data 와 find_item_by_property 두 곳에
//   ID_KEYS / NUM_KEYS / BOOL_KEYS 배열이 복제되어 있었고,
//   배열 길이까지 상수로 박혀 있어 필드 하나 추가에 4곳을 고쳐야 했습니다.
//
//  ── 왜 '이름 목록' 이 아니라 '규칙' 인가 ──
//   Dexie 확장 시 Rust 를 건드리지 않으려면, 새 필드를 이름으로 등록하는 대신
//   ① 접미사/부분일치 규칙 ② 값 형태 추론 으로 자동 판정해야 합니다.
//   그래야 'data.hs_code_2nd' 를 추가해도 Rust 재빌드가 필요 없습니다.
//
//  ⚠️ [PARITY] 이 파일의 판정 결과는 main.ts 의 canonicalizeData 와
//     비트 단위로 동일해야 합니다. 한쪽만 바뀌면 같은 값이
//     LanceDB 에는 Number, Dexie 에는 String 으로 저장되어
//     where('data.xxx').equals(...) 가 절반을 놓칩니다.
// =====================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonKind {
    Identifier, // String 확정
    Numeric,    // Number 확정
    Boolean,    // 0|1 정수 확정
    Tags,       // 배열 확정 (멀티엔트리 인덱스)
    Free,       // 손대지 않음
}

// ── ① 명시 예외 : 규칙만으로는 절대 판정 불가능한 이름 ──
//    이 목록은 '새 필드가 늘어난다고 커지지 않습니다'.
//    규칙과 충돌하는 기존 이름을 고정하기 위한 최소 집합입니다.
const FORCE_ID: &[&str] = &[
    "id", "no", "index", "goods", "order", "tracking", "digest",
];
const FORCE_NUM: &[&str] = &[
    "status", "views", "created_at", "updated_at",
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
const NUM_SUFFIX: &[&str] = &[
    "_price", "_amount", "_fee", "_rate", "_count", "_qty", "_at",
    "_weight", "_volume", "_duration", "_limit", "_threshold", "_charges",
];
const NUM_CONTAINS: &[&str] = &[
    "price", "amount", "quantity", "discount", "weight", "volume",
    "shipping_fee", "usage_", "threshold", "exchange_rate", "package_count",
    "local_charges", "number_of_",
];
const NUM_EXACT: &[&str] = &[
    "width", "height", "length",
];
const BOOL_PREFIX: &[&str] = &["is_", "has_", "allow_", "use_"];
const BOOL_SUFFIX: &[&str] = &["_only", "_included", "_allowed", "_match", "_shipping"];

/// 🌟 필드 이름만으로 저장 타입을 판정합니다.
///    새 필드는 대부분 접미사 규칙에 자동으로 걸리므로 Rust 수정이 불필요합니다.
pub fn kind_of(key: &str) -> CanonKind {
    let k = key.to_lowercase();

    if k == "tags" { return CanonKind::Tags; }

    if FORCE_ID.iter().any(|x| *x == k) { return CanonKind::Identifier; }
    if FORCE_NUM.iter().any(|x| *x == k) { return CanonKind::Numeric; }
    if FORCE_BOOL.iter().any(|x| *x == k) { return CanonKind::Boolean; }

    // 🌟 Boolean 을 먼저 검사합니다. 'bundle_shipping' 이 NUM_CONTAINS 의
    //    'shipping_fee' 와는 다르지만, '_shipping' 접미사로 잡혀야 하기 때문입니다.
    if BOOL_PREFIX.iter().any(|p| k.starts_with(p)) { return CanonKind::Boolean; }
    if BOOL_SUFFIX.iter().any(|s| k.ends_with(s)) { return CanonKind::Boolean; }

    if NUM_EXACT.iter().any(|x| *x == k) { return CanonKind::Numeric; }
    if NUM_SUFFIX.iter().any(|s| k.ends_with(s)) { return CanonKind::Numeric; }

    // 🌟 식별자를 수치보다 먼저 봅니다.
    //    'doc_number' 는 NUM_SUFFIX 의 '_number' 에도 걸리지만
    //    실제로는 'ABCD1234567' 같은 영숫자 혼합이므로 String 이어야 합니다.
    if ID_SUFFIX.iter().any(|s| k.ends_with(s)) { return CanonKind::Identifier; }
    if ID_CONTAINS.iter().any(|c| k.contains(c)) { return CanonKind::Identifier; }

    if NUM_CONTAINS.iter().any(|c| k.contains(c)) { return CanonKind::Numeric; }

    CanonKind::Free
}

/// 🌟 ISO 8601 문자열을 UTC epoch ms 로 환산합니다.
///    scheduler 의 normalize_data 가 started_at / expired_at 을
///    "2024-01-01T12:00:00" 형태로 만들기 때문에 반드시 필요합니다.
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
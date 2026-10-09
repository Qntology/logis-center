//! 텍스트 계층 검증 — json_parse / canonical / time_guide
//!
//! - json_parse   : LLM 이 망가뜨린 JSON 의 복구 규칙 (따옴표, 꼬리 쉼표, 잘린 괄호, 부분 구조 회수)
//! - canonical    : data.* 저장 타입 확정, 숫자/날짜 어휘 정규화, 릴레이 축(relay_*) 규칙, 원장(ledger) 상태기계
//! - time_guide   : 상대 기간 / 계절 / 벽시계 ms 경계
//!
//! 날짜 의존 테스트는 `today` 를 인자로 받는 함수만 고정값으로 단언하고,
//! 현재 시각을 쓰는 함수(resolve_intent / get_deterministic_time_guide)는 같은 함수로 기대값을 계산합니다.

use chrono::{Datelike, FixedOffset, NaiveDate};
use serde_json::{json, Value};
use tauri_app_lib::json_parse::{normalize_to_json_string, parse_json_from_llm};
use tauri_app_lib::time_guide::{
    anchor_exact_period, date_of_ms, day_start_ms, exact_with_season, get_deterministic_time_guide,
    intent_period, iso_bounds, lang_clock, operator_bounds_ms, period_ms, relative_period,
    resolve_intent, season_period, season_year, stored_wall_clock_ms, today_in, validity_condition,
    SeasonAnchor, RECENT_DAYS,
};
use tauri_app_lib::utils::canonical::{self as canon, RelayWriteLog};

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

// ═════════════════════════════ json_parse ═════════════════════════════

#[test]
fn normalize_to_json_string_repairs_common_llm_damage() {
    let cases = [
        // 따옴표 없는 키 + 작은따옴표 값 + 꼬리 쉼표
        ("{name: 'Kim', age: 30,}", r#"{"name": "Kim","age": 30}"#),
        // 잘린 배열/객체 → 괄호 스택 역순으로 닫기
        (r#"{"items": [{"a": 1}, {"b": 2"#, r#"{"items": [{"a": 1}, {"b": 2}]}"#),
        // 잘린 문자열 → 닫는 따옴표
        (r#"{"a": "hel"#, r#"{"a": "hel"}"#),
        // 따옴표 없는 값 → 문자열
        (r#"{"status": pending, "n": 5}"#, r#"{"status": "pending", "n": 5}"#),
        (r#"{"name": 홍길동}"#, r#"{"name": "홍길동"}"#),
        // "..." 생략 표식 제거
        (r#"{"list": [1, 2, ...]}"#, r#"{"list": [1, 2]}"#),
        // 스마트 따옴표 / 전각 기호
        ("{\u{201C}a\u{201D}: \u{201C}b\u{201D}}", r#"{"a": "b"}"#),
        ("{\"a\"\u{FF1A}1\u{FF0C}\"b\"\u{FF1A}2}", r#"{"a":1,"b":2}"#),
    ];
    for (input, expected) in cases {
        let out = normalize_to_json_string(input);
        assert_eq!(out, expected, "input {input:?}");
        assert!(serde_json::from_str::<Value>(&out).is_ok(), "repaired output must parse: {out}");
    }
}

#[test]
fn parse_json_from_llm_strips_think_tags_and_fences() {
    assert_eq!(
        parse_json_from_llm("<think>hmm {x}</think>\n```json\n{\"a\": 1}\n```"),
        json!({"a": 1})
    );
    // 닫히지 않은 <think> 는 그 뒤 전체를 버림
    assert_eq!(parse_json_from_llm("{\"a\":1}<think>unfinished"), json!({"a": 1}));
    // 문장 속 배열 / 스칼라
    assert_eq!(parse_json_from_llm("Here: [1, 2, 3] done"), json!([1, 2, 3]));
    assert_eq!(parse_json_from_llm("42"), json!(42));
    // 스마트 따옴표 한국어
    assert_eq!(
        parse_json_from_llm("{\u{201C}name\u{201D}: \u{201C}홍길동\u{201D}}"),
        json!({"name": "홍길동"})
    );
}

#[test]
fn parse_json_from_llm_salvages_or_falls_back_to_empty_object() {
    // 잘린 문자열
    assert_eq!(parse_json_from_llm(r#"{"a": 1, "b": "x"#), json!({"a": 1, "b": "x"}));
    // 문법이 깨진 쌍만 버리고 나머지 키 보존
    assert_eq!(
        parse_json_from_llm(r#"{"a": 1, "b": [1 2], "c": "x"}"#),
        json!({"a": 1, "c": "x"})
    );
    // 배열 원소 단위 회수
    assert_eq!(parse_json_from_llm("[1, 2, x y, 4]"), json!([1, 2, 4]));
    // 아무것도 없으면 빈 객체
    assert_eq!(parse_json_from_llm(""), json!({}));
    assert_eq!(parse_json_from_llm("nothing here"), json!({}));
}

#[test]
#[ignore = "BUG(B1): normalize_to_json_string strips every \\\" so valid escaped quotes break the string and the key is dropped"]
fn parse_json_from_llm_keeps_escaped_quotes() {
    // 꼬리 쉼표 때문에 복구 경로로 들어가는데, \" 제거로 "q" 값이 깨져 {"n":1} 만 남습니다.
    assert_eq!(
        parse_json_from_llm(r#"{"q": "He said \"hi\"", "n": 1,}"#),
        json!({"q": "He said \"hi\"", "n": 1})
    );
}

#[test]
fn sanitizers_drop_control_and_special_token_chars() {
    // json_parse 판: ASCII + 한글만 남기고 특수 토큰 경계를 깨뜨림
    assert_eq!(
        tauri_app_lib::json_parse::sanitize_llm_input("<|im_start|>안녕 é😀\tok"),
        "< |im_start| >안녕 \tok"
    );
    // parsing 판: 제어문자/zero-width/bidi/PUA 만 제거하고 다국어는 보존
    assert_eq!(
        tauri_app_lib::parsing::sanitize_llm_input(
            "a\u{200B}b\u{202E}c\u{E000}d\u{7}e\t한글 漢字 😀 <|end|>"
        ),
        "abcde\t한글 漢字 😀 < |end| >"
    );
}

// ═════════════════════════════ canonical : 타입 / 어휘 ═════════════════════════════

#[test]
fn kind_of_classifies_field_names() {
    use tauri_app_lib::utils::canonical::CanonKind::*;
    let cases = [
        ("tags", Tags),
        ("TAGS", Tags),
        ("id", Identifier),
        ("no", Identifier),
        ("digest", Identifier),
        ("order_no", Identifier),
        ("tracking_number", Identifier),
        ("hs_code", Identifier),
        ("reference_po", Identifier),
        ("barcode", Identifier),
        ("container_number", Identifier),
        ("discount_code", Identifier),
        ("status", Numeric),
        ("created_at", Numeric),
        ("goods", Numeric),
        ("rel_ci", Numeric),
        ("width", Numeric),
        ("rate", Numeric),
        ("sale_price", Numeric),
        ("package_count", Numeric),
        ("weight", Numeric),
        ("container_count", Numeric),
        ("detail", Boolean),
        ("is_active", Boolean),
        ("tax_included", Boolean),
        ("new_customer_only", Boolean),
        ("use_coupon", Boolean),
        ("title", Free),
        ("color", Free),
        ("currency", Free),
    ];
    for (key, expected) in cases {
        assert_eq!(canon::kind_of(key), expected, "kind_of({key:?})");
    }
}

#[test]
fn iso_to_epoch_ms_accepts_date_and_datetime_shapes() {
    assert_eq!(canon::iso_to_epoch_ms("2026-03-15T00:00:00"), Some(1773532800000));
    assert_eq!(canon::iso_to_epoch_ms("2026-03-15 12:00:00"), Some(1773576000000));
    assert_eq!(canon::iso_to_epoch_ms("2026-03-15T10:20:30"), Some(1773570030000));
    assert_eq!(canon::iso_to_epoch_ms("2026-03-15"), Some(1773532800000));
    for bad in ["2026/03/15", "short", "2026-13-01", ""] {
        assert_eq!(canon::iso_to_epoch_ms(bad), None, "{bad:?}");
    }
}

#[test]
#[ignore = "BUG(B2): iso_to_epoch_ms falls back to the date prefix when a zone suffix is present and silently drops the time"]
fn iso_to_epoch_ms_keeps_time_with_zone_suffix() {
    assert_eq!(canon::iso_to_epoch_ms("2026-03-15T10:20:30Z"), Some(1773570030000));
}

#[test]
#[ignore = "BUG(B2): iso_to_epoch_ms slices &t[..10] by bytes and panics when a multi-byte char straddles byte 10"]
fn iso_to_epoch_ms_does_not_panic_on_multibyte_input() {
    assert_eq!(canon::iso_to_epoch_ms("2026-03-1５"), None);
}

#[test]
fn numeric_date_to_iso_orders_fields_by_language() {
    let cases = [
        ("2026-03-15", "en", "2026-03-15T00:00:00"),
        ("15.03.2026", "de", "2026-03-15T00:00:00"),
        ("03/04/2026", "en", "2026-03-04T00:00:00"), // 미국식 월/일
        ("03/04/2026", "de", "2026-04-03T00:00:00"), // 일/월
        ("03.04.2026", "en", "2026-04-03T00:00:00"), // 점 구분은 일/월
        ("Apr-19-2022", "en", "2022-04-19T00:00:00"),
        ("19 April 2022 14:30", "en", "2022-04-19T14:30:00"),
        ("2026-03-15 2:30 PM", "en", "2026-03-15T14:30:00"),
        ("2026.3.15 오후 3시", "ko", "2026-03-15T15:00:00"),
        ("2026-03-15T10:20:30", "en", "2026-03-15T10:20:30"),
        ("20260315", "", "2026-03-15T00:00:00"),
        ("12/31/99", "en", "1999-12-31T00:00:00"),
        ("26-03-15", "ko", "2026-03-15T00:00:00"),
        ("05/06/07", "en", "2007-05-06T00:00:00"),
        ("05/06/07", "fr", "2007-06-05T00:00:00"),
        ("05/06/07", "ja", "2005-06-07T00:00:00"),
        ("2026-15-03", "en", "2026-03-15T00:00:00"), // 월>12 이면 일과 교환
        ("١٥/٠٣/٢٠٢٦", "ar", "2026-03-15T00:00:00"), // 아랍-인도 숫자
    ];
    for (input, lang, expected) in cases {
        assert_eq!(
            canon::numeric_date_to_iso(input, lang).as_deref(),
            Some(expected),
            "numeric_date_to_iso({input:?}, {lang:?})"
        );
    }
    for (input, lang) in [("0000-00-00", "en"), ("15/13/2026", "en"), ("not a date", "en"), ("2026-03", "en"), ("", "en")] {
        assert_eq!(canon::numeric_date_to_iso(input, lang), None, "{input:?}");
    }
}

#[test]
#[ignore = "BUG(B3): any trailing word that prefixes a month name ('set' → Sept, 'out' → Oct) overrides the numeric month"]
fn numeric_date_to_iso_ignores_ordinary_words() {
    assert_eq!(
        canon::numeric_date_to_iso("2026-03-15 set", "en").as_deref(),
        Some("2026-03-15T00:00:00")
    );
    assert_eq!(
        canon::numeric_date_to_iso("Shipped out 2026-03-15", "en").as_deref(),
        Some("2026-03-15T00:00:00")
    );
}

#[test]
fn parse_number_run_resolves_separators_and_cjk_multipliers() {
    let cases: [(&str, bool, Option<f64>); 16] = [
        ("1,234.56", false, Some(1234.56)),
        ("1.234,56", true, Some(1234.56)),
        ("1,5", true, Some(1.5)),
        ("1,500", false, Some(1500.0)), // 천 단위
        ("1,500", true, Some(1.5)),     // 소수 쉼표 언어
        ("0,500", false, Some(0.5)),
        ("1.234.567", false, Some(1234567.0)),
        ("1.2.3", false, None),
        ("-5", false, Some(-5.0)),
        ("A-5", false, Some(5.0)), // 식별자 하이픈은 부호가 아님
        ("3만", false, Some(30000.0)),
        ("1.5억원", false, Some(150000000.0)),
        (".5", false, Some(0.5)),
        ("abc", false, None),
        ("1 234,5 €", true, Some(1234.5)),
        ("1'000", false, Some(1000.0)),
    ];
    for (input, decimal_comma, expected) in cases {
        assert_eq!(canon::parse_number_run(input, decimal_comma), expected, "{input:?} dc={decimal_comma}");
    }
}

#[test]
fn number_text_and_normalize_number_lexeme() {
    assert_eq!(canon::number_text("1 234 567"), "1234567");
    assert_eq!(canon::number_text("12 34"), "12 34"); // 3자리 그룹이 아니면 공백 유지
    assert_eq!(canon::number_text("1'000"), "1000");
    assert_eq!(canon::number_text("١٢٣"), "123");
    assert_eq!(canon::number_text("１２３，４５６"), "123,456");

    assert_eq!(canon::normalize_number_lexeme("1,234원", false).as_deref(), Some("1234원"));
    assert_eq!(canon::normalize_number_lexeme("1234원", false), None); // 이미 정규형
    assert_eq!(canon::normalize_number_lexeme("12.50", false).as_deref(), Some("12.5"));
    assert_eq!(canon::normalize_number_lexeme("1.000,50 EUR", true).as_deref(), Some("1000.5 EUR"));
    assert_eq!(canon::normalize_number_lexeme("1,2 and 3", false), None); // 숫자 덩어리 2개

    assert!(canon::decimal_comma_lang("de-DE"));
    assert!(!canon::decimal_comma_lang("en"));
    assert!(canon::day_first_lang("ar"));
    assert!(!canon::decimal_comma_lang("ar"));
    assert!(canon::point_decimal_currency(" usd "));
    assert!(!canon::point_decimal_currency("EUR"));
}

#[test]
fn placeholder_and_truthy_word_classifiers() {
    for s in ["N/A", "0000-00-00", "미정", "Keine Angabe", "", "TBD"] {
        assert!(canon::is_missing_marker(s), "missing {s:?}");
    }
    assert!(!canon::is_missing_marker("none"));
    assert!(canon::is_null_like("none"));
    assert!(canon::is_null_like("없음"));
    assert!(!canon::is_null_like("ACME"));

    assert!(canon::is_catch_all_value("기타"));
    assert!(canon::is_catch_all_value("Others"));
    assert!(!canon::is_catch_all_value(""));

    for s in ["상세페이지 참조", "See product page", "상세 참조 바랍니다"] {
        assert!(canon::is_see_elsewhere_placeholder(s), "see-elsewhere {s:?}");
    }
    for s in ["See page 3", "Red", ""] {
        assert!(!canon::is_see_elsewhere_placeholder(s), "not see-elsewhere {s:?}");
    }

    for s in ["Ja", "verfügbar", "예", " YES! "] {
        assert!(canon::truthy_word(s), "truthy {s:?}");
    }
    assert!(!canon::truthy_word("no"));
    assert!(!canon::truthy_word(""));

    assert_eq!(canon::fold_text("Ünité"), "unite");
    assert_eq!(canon::fold_text("Straße"), "strasse");

    // 최상급 표지 (어휘 + 접두 + 관사쌍)
    for q in ["가장 싼 상품", "cheapest shirt", "最安値", "el más barato", "최저가"] {
        assert!(canon::has_superlative_marker(q), "{q:?}");
    }
    assert!(!canon::has_superlative_marker("blue shirt"));
}

// ═════════════════════════════ canonical : 릴레이 축 ═════════════════════════════

#[test]
fn relay_type_and_edge_helpers() {
    assert_eq!(canon::relay_type_family("receiving"), "tracking");
    assert_eq!(canon::relay_type_family("Sales"), "order");
    assert_eq!(canon::relay_type_family("coupon"), "event");
    assert_eq!(canon::relay_type_family(" BL "), "bl");

    assert!(canon::relay_type_matches("order", &json!({"type": "sales"})));
    assert!(!canon::relay_type_matches("order", &json!({"type": "goods"})));
    assert!(canon::relay_type_matches("order", &json!({})), "no type → no veto");

    assert!(canon::relay_establishes("goods", "order"));
    assert!(!canon::relay_establishes("goods", "goods"));
    assert!(canon::relay_establishes("review", "goods"));
    assert!(!canon::relay_establishes("review", "order"));
    assert!(!canon::relay_establishes("event", "tracking"));
    assert!(canon::relay_establishes("custom", "x"));
    assert!(!canon::relay_establishes("", "order"));

    assert_eq!(canon::relay_edge_target("rel_ci").as_deref(), Some("CI"));
    assert_eq!(canon::relay_edge_target(" rel_bl ").as_deref(), Some("BL"));
    assert_eq!(canon::relay_edge_target("rel_"), None);
    assert_eq!(canon::relay_edge_target("order").as_deref(), Some("order"));
    assert_eq!(canon::relay_edge_target("title"), None);

    assert_eq!(canon::relay_ref_index(Some(&json!(5))), Some(5));
    assert_eq!(canon::relay_ref_index(Some(&json!(4294967295u64))), Some(u32::MAX));
    for v in [json!(0), json!(4294967296u64), json!("5"), json!(-1), json!(null)] {
        assert_eq!(canon::relay_ref_index(Some(&v)), None, "{v}");
    }
    assert_eq!(canon::relay_ref_index(None), None);

    assert_eq!(canon::relay_companion_base("goods_title"), Some("goods"));
    assert_eq!(canon::relay_companion_base("Goods_Title"), Some("goods"));
    assert_eq!(canon::relay_companion_base("order_title"), Some("order"));
    assert_eq!(canon::relay_companion_base("goods_titles"), None);
}

#[test]
fn relay_write_respects_identity_anchor_and_overwrite_rules() {
    let mut log = RelayWriteLog::default();

    let mut dst = json!({});
    assert!(canon::relay_write(&mut dst, "title", json!("Hello"), false, &mut log));
    assert_eq!(dst, json!({"title": "Hello"}));
    assert_eq!(log.written, vec!["title".to_string()]);

    // 식별 축(id/index)은 절대 덮어쓰지 않음
    let mut dst = json!({"id": "a"});
    assert!(!canon::relay_write(&mut dst, "id", json!("b"), true, &mut log));
    assert_eq!(dst["id"], "a");
    assert_eq!(log.kept, vec!["id".to_string()]);

    // 링크 축은 값이 있으면 overwrite=true 여도 고정
    let mut dst = json!({"order": 5});
    assert!(!canon::relay_write(&mut dst, "order", json!(7), true, &mut log));
    assert_eq!(dst["order"], 5);

    // 플레이스홀더 값은 교체 대상
    let mut dst = json!({"title": "N/A"});
    assert!(canon::relay_write(&mut dst, "title", json!("Real"), false, &mut log));
    assert_eq!(dst["title"], "Real");

    // 일반 필드는 overwrite 플래그가 결정
    let mut dst = json!({"color": "red"});
    assert!(!canon::relay_write(&mut dst, "color", json!("blue"), false, &mut log));
    assert!(canon::relay_write(&mut dst, "color", json!("blue"), true, &mut log));
    assert_eq!(dst["color"], "blue");

    // 들어오는 값이 플레이스홀더면 쓰지 않음, 객체가 아닌 dst 도 거부
    let mut dst = json!({});
    assert!(!canon::relay_write(&mut dst, "color", json!(null), true, &mut log));
    assert!(!canon::relay_write(&mut dst, "color", json!("N/A"), true, &mut log));
    assert!(!canon::relay_write(&mut dst, "order", json!(0), true, &mut log));
    assert_eq!(dst, json!({}));
    let mut arr = json!([]);
    assert!(!canon::relay_write(&mut arr, "color", json!("x"), true, &mut log));
}

#[test]
fn ledger_prior_state_and_delta_table() {
    use tauri_app_lib::utils::canonical::LedgerPrior::*;
    assert_eq!(canon::ledger_prior(None), Absent);
    assert_eq!(canon::ledger_prior(Some(&json!({"ledger": "count"}))), Confirmed);
    assert_eq!(canon::ledger_prior(Some(&json!({"ledger": "draft"}))), Draft);
    assert_eq!(canon::ledger_prior(Some(&json!({"ledger": "placeholder"}))), Placeholder);
    assert_eq!(canon::ledger_prior(Some(&json!({"updated_at": 1}))), Confirmed);
    assert_eq!(canon::ledger_prior(Some(&json!({}))), Placeholder);
    assert_eq!(canon::ledger_prior(Some(&json!({"digest": "abc"}))), Draft);

    assert_eq!(canon::ledger_state(Draft, false), "draft");
    assert_eq!(canon::ledger_state(Draft, true), "count");
    assert_eq!(canon::ledger_state(Confirmed, false), "count");

    let table = [
        (Absent, false, (1, 0, 1)),
        (Absent, true, (0, 1, 1)),
        (Placeholder, false, (0, 0, 1)),
        (Placeholder, true, (-1, 1, 1)),
        (Draft, false, (0, 0, 0)),
        (Draft, true, (-1, 1, 0)),
        (Confirmed, false, (0, 0, 0)),
        (Confirmed, true, (0, 0, 0)),
    ];
    for (prior, confirm, expected) in table {
        assert_eq!(canon::ledger_delta(prior, confirm), expected, "{prior:?}/{confirm}");
    }
}

#[test]
fn relay_edges_and_keep_relay_index() {
    let doc = json!({
        "type": "order",
        "goods": [{"index": 3}, {"index": 3}, {"index": 4}],
        "tracking": 7,
        "rel_CI": 12,
        "rel_ORDER": 2,
        "event": 0
    });
    assert_eq!(
        canon::relay_edges(&doc),
        vec![
            ("goods".to_string(), 3),
            ("goods".to_string(), 4),
            ("tracking".to_string(), 7),
            ("rel_CI".to_string(), 12),
        ]
    );
    assert!(canon::relay_edges(&json!([1, 2])).is_empty());

    // 이전 index 를 복원하고, 덮어쓴 텍스트는 <key>_title 동반 필드로 보존
    let prior = json!({"order": 5});
    let mut merged = json!({"order": "ORD-1"});
    assert_eq!(canon::keep_relay_index(&prior, &mut merged, "goods"), vec![("order".to_string(), 5)]);
    assert_eq!(merged, json!({"order": 5, "order_title": "ORD-1"}));

    // 새 값이 이미 유효한 index 면 손대지 않음
    let mut merged = json!({"order": 9});
    assert!(canon::keep_relay_index(&prior, &mut merged, "goods").is_empty());
    assert_eq!(merged, json!({"order": 9}));
}

// ═════════════════════════════ canonical : 병합 가드 ═════════════════════════════

#[test]
fn value_shape_and_keep_value_shapes() {
    use tauri_app_lib::utils::canonical::ValueShape::*;
    assert_eq!(canon::value_shape("price", &json!(100)), Some(Quantity));
    assert_eq!(canon::value_shape("id", &json!(5)), Some(Code));
    assert_eq!(canon::value_shape("link", &json!("https://x.com/a")), Some(Url));
    assert_eq!(canon::value_shape("title", &json!("https://x.com a")), Some(Text));
    assert_eq!(canon::value_shape("order_no", &json!("ORD-123")), Some(Code));
    assert_eq!(canon::value_shape("order_no", &json!("N/A")), None);
    assert_eq!(canon::value_shape("created_at", &json!("2026-03-15")), Some(Quantity));
    assert_eq!(canon::value_shape("weight", &json!("12 kg")), Some(Quantity));
    assert_eq!(canon::value_shape("memo", &json!("call me at 3 or 4")), Some(Text));
    assert_eq!(canon::value_shape("flag", &json!(true)), None);

    // 수치였던 값이 문장으로 바뀌면 이전 값을 복원
    let prior = json!({"price": "1,200", "title": "Blue"});
    let mut merged = json!({"price": "Contact us", "title": "Shirt"});
    assert_eq!(canon::keep_value_shapes(&prior, &mut merged), vec!["price".to_string()]);
    assert_eq!(merged, json!({"price": "1,200", "title": "Shirt"}));
}

#[test]
fn placeholder_drop_and_informative_restore() {
    // drop_placeholder_values: 플레이스홀더 문자열을 지우고 경로를 보고
    let mut v = json!({
        "color": "N/A",
        "size": "M",
        "nested": {"memo": "상세페이지 참조", "x": "ok"},
        "items": [{"note": "TBD"}],
        "title": "N/A"
    });
    let dropped = canon::drop_placeholder_values(&mut v);
    assert_eq!(dropped, vec!["color", "items.[0].note", "nested.memo"]);
    // title 은 구조 키라 건드리지 않음
    assert_eq!(
        v,
        json!({"items": [{}], "nested": {"x": "ok"}, "size": "M", "title": "N/A"})
    );

    // keep_informative_values: 정보가 있던 값이 플레이스홀더/null 로 덮이면 복원
    let mut merged = json!({"color": "N/A"});
    assert_eq!(canon::keep_informative_values(&json!({"color": "Red"}), &mut merged), vec!["color"]);
    assert_eq!(merged, json!({"color": "Red"}));

    let mut merged = json!({"price": null});
    assert_eq!(canon::keep_informative_values(&json!({"price": 100}), &mut merged), vec!["price"]);
    assert_eq!(merged, json!({"price": 100}));

    // 구조 키(title)는 대상 아님
    let mut merged = json!({"title": "N/A"});
    assert!(canon::keep_informative_values(&json!({"title": "X"}), &mut merged).is_empty());
}

#[test]
fn timestamp_granularity_guards() {
    assert!(canon::is_coarser_timestamp("2026-03-15", "2026-03-15T10:20:00"));
    assert!(!canon::is_coarser_timestamp("2026-03-15T10:20:00", "2026-03-15"));
    assert!(!canon::is_coarser_timestamp("2026-03-16", "2026-03-15T10:20:00"));

    // 문자열 prior: 더 거친 날짜로 덮이면 원래 시각 복원
    let mut merged = json!({"etd": "2026-03-15"});
    assert_eq!(
        canon::keep_finer_timestamps(&json!({"etd": "2026-03-15T10:20:00"}), &mut merged),
        vec!["etd"]
    );
    assert_eq!(merged["etd"], "2026-03-15T10:20:00");

    // epoch prior: 같은 날 안의 ms 를 ISO 로 복원
    let mut merged = json!({"etd_at": "2026-03-15"});
    assert_eq!(
        canon::keep_finer_timestamps(&json!({"etd_at": 1773570030000i64}), &mut merged),
        vec!["etd_at"]
    );
    assert_eq!(merged["etd_at"], "2026-03-15T10:20:30");

    let n = |v: i64| serde_json::Number::from(v);
    assert_eq!(canon::epoch_field_text("created_at", &n(1773532800000)).as_deref(), Some("2026-03-15T00:00:00"));
    assert_eq!(canon::epoch_field_text("price", &n(1773532800000)), None);
    assert_eq!(canon::epoch_field_text("expired_at", &n(4102444800000)), None); // 2100-01-01 은 범위 밖(배타)
    assert_eq!(canon::epoch_field_text("updated_at", &n(946684799999)), None); // 2000 이전
}

// ═════════════════════════════ time_guide ═════════════════════════════

#[test]
fn relative_period_resolves_calendar_keys() {
    assert_eq!(relative_period("last_month", d(2026, 1, 15)), Some((d(2025, 12, 1), d(2025, 12, 31))));
    assert_eq!(relative_period("this_month", d(2024, 2, 10)), Some((d(2024, 2, 1), d(2024, 2, 29))));
    assert_eq!(relative_period("recently", d(2026, 10, 9)), Some((d(2026, 9, 9), d(2026, 10, 9))));
    assert_eq!(RECENT_DAYS, 30);
    assert_eq!(relative_period("yesterday", d(2026, 3, 1)), Some((d(2026, 2, 28), d(2026, 2, 28))));
    assert_eq!(relative_period("today", d(2026, 3, 1)), Some((d(2026, 3, 1), d(2026, 3, 1))));
    assert_eq!(relative_period("this_year", d(2026, 10, 9)), Some((d(2026, 1, 1), d(2026, 12, 31))));
    assert_eq!(relative_period("last_year", d(2026, 10, 9)), Some((d(2025, 1, 1), d(2025, 12, 31))));
    assert_eq!(relative_period("bogus", d(2026, 10, 9)), None);
}

#[test]
fn season_period_and_season_year() {
    // 북반구
    assert_eq!(season_period("winter", 2025, false), Some((d(2025, 12, 1), d(2026, 2, 28))));
    assert_eq!(season_period("winter", 2023, false), Some((d(2023, 12, 1), d(2024, 2, 29))));
    assert_eq!(season_period("summer", 2026, false), Some((d(2026, 6, 1), d(2026, 8, 31))));
    // 남반구는 6개월 이동
    assert_eq!(season_period("summer", 2026, true), Some((d(2026, 12, 1), d(2027, 2, 28))));
    assert_eq!(season_period("spring", 2026, true), Some((d(2026, 9, 1), d(2026, 11, 30))));
    assert_eq!(season_period("fall", 2026, false), None);

    let today = d(2026, 10, 9);
    assert_eq!(season_year("winter", "", today, false, SeasonAnchor::Current), Some(2026));
    assert_eq!(season_year("winter", "", today, false, SeasonAnchor::Latest), Some(2025));
    assert_eq!(season_year("autumn", "", today, false, SeasonAnchor::Latest), Some(2026));
    assert_eq!(season_year("summer", "last_year", today, false, SeasonAnchor::Current), Some(2025));
    assert_eq!(season_year("winter", "this_year", today, false, SeasonAnchor::Latest), Some(2026));
    assert_eq!(season_year("xx", "", today, false, SeasonAnchor::Current), None);
}

#[test]
fn intent_period_prefers_season_then_relative_key() {
    let today = d(2026, 10, 9);
    assert_eq!(
        intent_period("", "winter", today, false, SeasonAnchor::Latest),
        Some((d(2025, 12, 1), d(2026, 2, 28)))
    );
    assert_eq!(
        intent_period("this_month", "", today, false, SeasonAnchor::Latest),
        Some((d(2026, 10, 1), d(2026, 10, 31)))
    );
    // 남반구 1월은 진행 중인 여름(전년 12월 시작)
    assert_eq!(
        intent_period("", "summer", d(2026, 1, 15), true, SeasonAnchor::Current),
        Some((d(2025, 12, 1), d(2026, 2, 28)))
    );
    assert_eq!(intent_period("", "", today, false, SeasonAnchor::Current), None);
}

#[test]
fn exact_period_anchoring_and_season_narrowing() {
    let today = d(2026, 10, 9);
    // 연도 미기재 + 과거만 허용 + 미래 → 1년 전
    assert_eq!(
        anchor_exact_period(d(2026, 12, 1), d(2026, 12, 31), false, "", today, true),
        (d(2025, 12, 1), d(2025, 12, 31))
    );
    // last_year: 월 단위 구간은 월말까지 맞춤 (윤년 보정)
    assert_eq!(
        anchor_exact_period(d(2025, 2, 1), d(2025, 2, 28), false, "last_year", today, false),
        (d(2024, 2, 1), d(2024, 2, 29))
    );
    assert_eq!(
        anchor_exact_period(d(2024, 2, 1), d(2024, 2, 29), false, "last_year", today, false),
        (d(2023, 2, 1), d(2023, 2, 28))
    );
    // 월 중간 구간은 그대로 12개월 이동
    assert_eq!(
        anchor_exact_period(d(2026, 12, 15), d(2026, 12, 20), false, "", today, true),
        (d(2025, 12, 15), d(2025, 12, 20))
    );
    // 이동하지 않는 경우
    let p = (d(2026, 12, 1), d(2026, 12, 31));
    assert_eq!(anchor_exact_period(p.0, p.1, true, "last_year", today, true), p);
    assert_eq!(anchor_exact_period(p.0, p.1, false, "", today, false), p);
    assert_eq!(anchor_exact_period(p.0, p.1, false, "this_year", today, true), p);

    // exact_with_season: 연 단위 구간만 계절로 좁힘
    assert_eq!(
        exact_with_season(d(2026, 1, 1), d(2026, 12, 31), "year", "summer", false),
        (d(2026, 6, 1), d(2026, 8, 31), true)
    );
    assert_eq!(
        exact_with_season(d(2026, 3, 1), d(2026, 3, 31), "month", "summer", false),
        (d(2026, 3, 1), d(2026, 3, 31), false)
    );
    assert_eq!(
        exact_with_season(d(2026, 1, 1), d(2026, 12, 31), "year", "", false),
        (d(2026, 1, 1), d(2026, 12, 31), false)
    );
}

#[test]
fn wall_clock_bounds_and_validity_condition() {
    let (s, e) = (d(2026, 10, 1), d(2026, 10, 31));
    assert_eq!(
        iso_bounds(s, e),
        ("2026-10-01T00:00:00".to_string(), "2026-10-31T23:59:59".to_string())
    );
    assert_eq!(stored_wall_clock_ms(s, e), (1790812800000, 1793491199999));

    assert_eq!(
        Value::Object(validity_condition(s, e, "between")),
        json!({
            "started_at": {"operator": "lte", "value": 1793491199999i64},
            "expired_at": {"operator": "gte", "value": 1790812800000i64}
        })
    );
    assert_eq!(
        Value::Object(validity_condition(s, e, "gte")),
        json!({"expired_at": {"operator": "gte", "value": 1790812800000i64}})
    );
    assert_eq!(
        Value::Object(validity_condition(s, e, "lte")),
        json!({"started_at": {"operator": "lte", "value": 1793491199999i64}})
    );
}

#[test]
fn lang_clock_and_offset_period_math() {
    let east = |h: i32| FixedOffset::east_opt(h * 3600).unwrap();
    assert_eq!(lang_clock("ko"), (east(9), false));
    assert_eq!(lang_clock("pt-BR"), (east(-3), true));
    assert_eq!(lang_clock("en"), (east(0), false));
    assert_eq!(lang_clock("sw"), (east(3), true));

    let kst = east(9);
    let utc = east(0);
    let day = d(2026, 3, 15);
    assert_eq!(period_ms(&kst, day, day), (1773500400000, 1773586799999));
    assert_eq!(operator_bounds_ms(&kst, day, day, "gte"), (1773500400000, 0));
    assert_eq!(operator_bounds_ms(&kst, day, day, "lte"), (0, 1773586799999));
    assert_eq!(operator_bounds_ms(&kst, day, day, "between"), (1773500400000, 1773586799999));
    assert_eq!(day_start_ms(&utc, day), 1773532800000);
    assert_eq!(date_of_ms(&utc, 1773500400000), Some(d(2026, 3, 14)));
    assert_eq!(date_of_ms(&kst, 1773500400000), Some(d(2026, 3, 15)));
}

#[test]
fn resolve_intent_labels_follow_season_and_time_keys() {
    let p = resolve_intent("", "winter", "en", SeasonAnchor::Current).expect("winter");
    assert_eq!(p.label, "Season 'winter'");
    assert_eq!((p.start.month(), p.start.day()), (12, 1));
    assert_eq!(p.end.month(), 2);
    assert_eq!(p.end.year(), p.start.year() + 1);

    let p = resolve_intent("this_month", "", "ko", SeasonAnchor::Current).expect("this_month");
    assert_eq!(p.label, "Time intent 'this_month'");
    assert_eq!(p.start.day(), 1);
    assert_eq!(p.start.month(), p.end.month());
    assert_ne!(p.end.succ_opt().unwrap().month(), p.end.month(), "end is the last day of the month");

    let utc = lang_clock("en").0;
    let y0 = today_in(&utc).year();
    let p = resolve_intent("last_year", "summer", "en", SeasonAnchor::Current).expect("summer");
    let y1 = today_in(&utc).year();
    assert_eq!(p.label, "Season 'summer' of time intent 'last_year'");
    if y0 == y1 {
        assert_eq!((p.start, p.end), (d(y0 - 1, 6, 1), d(y0 - 1, 8, 31)));
    }

    assert!(resolve_intent("", "", "en", SeasonAnchor::Current).is_none());
}

#[test]
fn deterministic_time_guide_builds_override_and_condition() {
    let off = lang_clock("ko").0;
    let before = today_in(&off);
    let (guide, cond) = get_deterministic_time_guide("Time Intent [this_month]", "ko");
    let after = today_in(&off);

    if before == after {
        // 자정 경계에 걸리지 않았을 때만 정확값 비교
        let (s, e) = relative_period("this_month", before).unwrap();
        assert_eq!(
            guide,
            format!(
                "- [DETERMINISTIC OVERRIDE] Time intent 'this_month' detected ({} ~ {}). DO NOT extract date properties (like started_at, expired_at, date). The system will auto-inject them.",
                s, e
            )
        );
        assert_eq!(cond, Some(Value::Object(validity_condition(s, e, "between"))));
    }

    let (guide, cond) = get_deterministic_time_guide("Season Intent [ winter ]", "en");
    assert!(guide.contains("Season 'winter' detected"), "{guide}");
    let cond = cond.expect("season condition");
    assert_eq!(cond["started_at"]["operator"], "lte");
    assert_eq!(cond["expired_at"]["operator"], "gte");

    assert_eq!(get_deterministic_time_guide("", "en"), (String::new(), None));
    assert_eq!(get_deterministic_time_guide("Time Intent [bogus]", "en"), (String::new(), None));
}

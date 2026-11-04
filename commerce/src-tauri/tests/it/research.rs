//! 심층 리포트의 Map 단계 (model/research.rs): 저장된 문서 전체를 Rust 로 집계합니다. 모델 없음.

use serde_json::{json, Value};
use tauri_app_lib::model::research::{clip_chars, FieldSum, ResearchDigest, ResearchRow};
use tauri_app_lib::store::TradeDocument;

fn ms(y: i32, m: u32, d: u32) -> i64 {
    chrono::NaiveDate::from_ymd_opt(y, m, d).unwrap().and_hms_opt(9, 0, 0).unwrap().and_utc().timestamp_millis()
}

fn row(doc_type: &str, created_at: i64, updated_at: i64, data: Value) -> ResearchRow {
    ResearchRow { doc_type: doc_type.to_string(), created_at, updated_at, data }
}

#[test]
fn digest_counts_types_months_drafts_sums_and_categories() {
    let rows = vec![
        row("order", ms(2026, 9, 3), 5, json!({"amount": 12000, "quantity": "3", "status": "complete", "carrier": "CJ"})),
        row("order", ms(2026, 9, 20), 0, json!({"amount": "₩8,500", "quantity": 1, "status": "progress", "carrier": "CJ"})),
        row("order", ms(2026, 10, 1), 7, json!({"amount": "n/a", "status": "complete", "carrier": "한진"})),
        row("CI", ms(2026, 10, 2), 7, json!({"financials": {"amount": 78500.5, "currency": "USD"}, "cargo": {"package_count": 40}})),
        row("", 0, 7, Value::Null),
    ];
    let d = ResearchDigest::from_rows(&rows);

    assert_eq!(d.documents, 5);
    assert_eq!(d.drafts, 1);
    assert_eq!(d.by_type.get("order"), Some(&3));
    assert_eq!(d.by_type.get("unknown"), Some(&1), "빈 유형");
    assert_eq!(d.by_month.get("2026-09"), Some(&2));
    assert_eq!(d.by_month.get("2026-10"), Some(&2));
    assert_eq!(d.by_month.get("unknown"), Some(&1), "작성 시각 없음");

    assert_eq!(d.sums.get("order.amount"), Some(&FieldSum { sum: 20500.0, count: 2 }), "숫자 아닌 값은 건너뜁니다");
    assert_eq!(d.sums.get("order.quantity"), Some(&FieldSum { sum: 4.0, count: 2 }));
    assert_eq!(d.sums.get("CI.financials.amount"), Some(&FieldSum { sum: 78500.5, count: 1 }), "중첩 칸은 경로로");
    assert_eq!(d.sums.get("CI.cargo.package_count"), Some(&FieldSum { sum: 40.0, count: 1 }));

    assert_eq!(d.categories["status"].get("complete"), Some(&2));
    assert_eq!(d.categories["carrier"].get("CJ"), Some(&2));
    assert_eq!(d.categories["currency"].get("USD"), Some(&1));
}

#[test]
fn rendered_digest_quotes_the_numbers_and_stays_bounded() {
    let mut rows = Vec::new();
    for i in 0..40u32 {
        let doc_type = format!("type{:02}", i % 20);
        rows.push(row(&doc_type, ms(2024 + (i as i32 % 3), 1 + i % 12, 1), 1, json!({"amount": i, "status": format!("s{}", i % 9)})));
    }
    let d = ResearchDigest::from_rows(&rows);
    let text = d.render();
    assert!(text.contains("all 40 stored documents"), "{text}");
    assert!(text.contains("- drafts (not yet confirmed): 0"), "{text}");
    assert!(text.lines().count() <= 2 + 1 + 1 + 12 + d.categories.len(), "줄 수 상한: {text}");
    let types_line = text.lines().find(|l| l.starts_with("- documents by type:")).unwrap();
    assert_eq!(types_line.matches(", ").count(), 9, "유형은 상위 10개까지: {types_line}");
    let months_line = text.lines().find(|l| l.starts_with("- documents created per month:")).unwrap();
    assert_eq!(months_line.matches(", ").count(), 11, "월은 최근 12개까지: {months_line}");
    assert!(text.contains("- most frequent status: "), "{text}");
}

#[test]
fn rows_come_from_stored_documents() {
    let doc = TradeDocument {
        r#type: "goods".into(),
        json_data: r#"{"sale_price": 1500}"#.into(),
        created_at_ts: ms(2026, 11, 1),
        updated_at_ts: 0,
        ..Default::default()
    };
    let r = ResearchRow::from(&doc);
    assert_eq!((r.doc_type.as_str(), r.updated_at, r.data["sale_price"].as_i64()), ("goods", 0, Some(1500)));
    let broken = TradeDocument { json_data: "{".into(), ..Default::default() };
    assert_eq!(ResearchRow::from(&broken).data, Value::Null);
}

#[test]
fn clip_chars_cuts_on_character_boundaries() {
    let korean = "가".repeat(250);
    let clipped = clip_chars(&korean, 200);
    assert_eq!(clipped.chars().count(), 201, "200자 + 말줄임표");
    assert!(clipped.ends_with('…'));
    assert_eq!(clip_chars("짧은 글", 200), "짧은 글");
}

//! 모델 계층(`tauri_app_lib::model`) 의 순수 함수 검증 (T32, T34 중 공개 부분)
//!
//! - `model::merge`: 스키마 에코 판정, 인쇄값 동치, 조건값 추출, 요약문, LLM 결과 병합, 뱅크 중심.
//! - `model`: RAM 사용률 계산, 생성 모델 종류 번호.
//! 모델 가중치 · GPU · AppHandle 은 필요 없습니다. (`path_footprint_mb` 는 private 이라 제외)
//! `#[ignore = "BUG(Bn): …"]` 테스트는 '의도된 동작' 을 단언하며, 수정 전까지는 실패합니다.

use serde_json::{json, Map, Value};
use tauri_app_lib::model::merge::{
    bank_centroid, generate_rich_summary, is_schema_echo, merge_json_manual, printed_number,
    row_axis_aggregate_related, same_printed_token, same_printed_value, trade_resolve_condition_value,
};
use tauri_app_lib::model::{ram_used_pct, LogisModel, ModelSize};

fn obj(v: Value) -> Map<String, Value> {
    match v {
        Value::Object(m) => m,
        other => panic!("expected a JSON object, got {other}"),
    }
}

fn len_of(root: &Map<String, Value>, key: &str) -> usize {
    root.get(key).and_then(|v| v.as_array()).map_or(0, |a| a.len())
}

// ── T32 LLM 결과 정리 헬퍼 ───────────────────────────────────────

#[test]
fn schema_echo_placeholders() -> anyhow::Result<()> {
    for s in [
        "", "   ", "{String}", "<value>", "N/A", "null", "Null", "yyyy-mm-dd", "not specified", "해당 없음", "不明",
    ] {
        assert!(is_schema_echo(s), "{s:?} is a prompt placeholder");
    }
    for s in ["CI-1", "0", "ACME Corp", "2026-08-26", "BUSAN"] {
        assert!(!is_schema_echo(s), "{s:?} is a real value");
    }
    Ok(())
}

#[test]
fn printed_token_and_value_equivalence() -> anyhow::Result<()> {
    assert!(same_printed_token("CI-2026-08001", "ci2026 08001"));
    assert!(same_printed_token("BOX", "boxes"));
    assert!(same_printed_token("PC", "PCS"));
    assert!(!same_printed_token("", "x"));
    assert!(!same_printed_token("BL-1", "BL-2"));

    assert!(same_printed_value("1,000", "1000.0"));
    assert!(same_printed_value("12.50", "12.5"));
    assert!(!same_printed_value("abc", "xyz"));
    assert!(!same_printed_value("12.5", "12.6"));
    Ok(())
}

#[test]
#[ignore = "BUG(B24): the plural/vowel-swap rule equates identifiers that differ only in a final letter"]
fn printed_token_keeps_identifiers_distinct() -> anyhow::Result<()> {
    assert!(!same_printed_token("PO-1234A", "PO-1234E"));
    assert!(!same_printed_token("INV2024", "INV2024S"));
    Ok(())
}

#[test]
fn printed_number_extraction() -> anyhow::Result<()> {
    assert_eq!(printed_number("USD 1,234.50"), Some(1234.5));
    assert_eq!(printed_number("-12.5kg"), Some(-12.5));
    assert_eq!(printed_number("2024-01-15"), None);
    assert_eq!(printed_number("-"), None);
    assert_eq!(printed_number("N/A"), None);
    Ok(())
}

#[test]
fn trade_condition_value_picks_identifier() -> anyhow::Result<()> {
    assert_eq!(trade_resolve_condition_value("doc_number", "find bill BL-55432219, please"), "BL-55432219");
    // 구분자 없는 4자리 숫자는 식별자로 보지 않습니다 (6자 이상 필요).
    assert_eq!(trade_resolve_condition_value("no", "order 1234"), "");
    assert_eq!(trade_resolve_condition_value("reference_po", "against PO99281A"), "PO99281A");
    assert_eq!(trade_resolve_condition_value("hub_reference", "everything for PO-99281A"), "PO-99281A");
    assert_eq!(trade_resolve_condition_value("doc_number", "   "), "");
    Ok(())
}

#[test]
fn rich_summary_sentences() -> anyhow::Result<()> {
    let ci = json!({
        "header": {"document_number": "CI-1", "issue_date": "2026-08-26"},
        "parties": {"supplier_name": "ACME", "buyer_name": "null"},
        "financials": {"amount_total": 1500, "currency_code": "USD"},
        "line_items": [{"description": "Cotton T-Shirts"}, {"description": "Hat"}]
    });
    assert_eq!(
        generate_rich_summary("CI", &ci),
        "This is a Commercial Invoice document. Document number is CI-1. Issued on 2026-08-26. \
         Supplier/Shipper is ACME. Total amount is 1500 USD. Contains items: Cotton T-Shirts."
    );
    assert_eq!(
        generate_rich_summary("tracking", &json!({"tracking_number": "603145678912", "text": "Delivered"})),
        "This is a Shipping Label / Tracking Info document. The tracking number is 603145678912. Delivered"
    );
    assert_eq!(generate_rich_summary("XYZ", &json!({})), "This is a XYZ document.");
    assert_eq!(
        generate_rich_summary(
            "BL",
            &json!({"logistics": {
                "location_port_of_loading": "BUSAN",
                "location_port_of_discharge": "LONG BEACH",
                "transport_mode": "SEA"
            }})
        ),
        "This is a Bill of Lading document. Shipped from BUSAN to LONG BEACH. Transport mode is SEA."
    );
    assert_eq!(
        generate_rich_summary("PL", &json!({"parties": {"supplier_name": "ACME", "buyer_name": "Globex"}, "financials": {"amount_total": 0}})),
        "This is a Packing List document. Transaction involved ACME as the supplier/shipper and Globex as the buyer/consignee."
    );
    Ok(())
}

#[test]
fn merge_json_manual_header_skips_echo_and_null() -> anyhow::Result<()> {
    let mut root = obj(json!({"header": {}}));
    merge_json_manual(
        &mut root,
        "header",
        json!({"doc_number": "A", "issue_date": "2026-01-01", "seller": "{String}", "buyer": null}),
    );
    assert_eq!(Value::Object(root.clone()), json!({"header": {"doc_number": "A", "issue_date": "2026-01-01"}}));

    // 카테고리 이름으로 한 번 더 감싼 응답도 풀어서 병합합니다.
    merge_json_manual(&mut root, "header", json!({"header": {"doc_number": "B"}}));
    assert_eq!(root["header"]["doc_number"], "B");
    assert_eq!(root["header"]["issue_date"], "2026-01-01");
    Ok(())
}

#[test]
fn merge_json_manual_rows_and_containers() -> anyhow::Result<()> {
    // items → line_items, 단일 객체는 배열로 감쌉니다.
    let mut root = obj(json!({"line_items": []}));
    merge_json_manual(&mut root, "items", json!({"description": "Hat", "quantity": 1}));
    assert_eq!(len_of(&root, "line_items"), 1);
    // description 이 같은 행은 중복으로 보고 새 행만 추가
    merge_json_manual(
        &mut root,
        "items",
        json!({"items": [{"description": "Hat", "quantity": 1}, {"description": "Cap", "quantity": 2}]}),
    );
    assert_eq!(len_of(&root, "line_items"), 2);
    assert_eq!(root["line_items"][1]["description"], "Cap");
    // 값이 전부 null 인 객체는 아무것도 추가하지 않습니다.
    merge_json_manual(&mut root, "items", json!({"description": null}));
    assert_eq!(len_of(&root, "line_items"), 2);

    let mut boxes = obj(json!({"containers": []}));
    merge_json_manual(
        &mut boxes,
        "containers",
        json!([
            {"container_number": "MSCU1234567"},
            {"container_number": "MSCU1234567"},
            {"container_number": "TGHU7654321"}
        ]),
    );
    assert_eq!(len_of(&boxes, "containers"), 2);
    Ok(())
}

#[test]
#[ignore = "BUG(B17): rows without description compare \"\" == \"\" and are dropped as duplicates"]
fn merge_json_manual_keeps_rows_without_identity() -> anyhow::Result<()> {
    let mut root = obj(json!({"line_items": [{"quantity": 1}]}));
    merge_json_manual(&mut root, "items", json!({"items": [{"quantity": 2}]}));
    assert_eq!(len_of(&root, "line_items"), 2);
    Ok(())
}

#[test]
fn bank_centroid_is_normalized_mean_of_nonzero_rows() -> anyhow::Result<()> {
    let c = bank_centroid(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]]);
    assert_eq!(c.len(), 2);
    for v in c.iter() {
        assert!((v - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6, "{c:?}");
    }
    // 차원이 다른 행은 건너뜁니다.
    assert_eq!(bank_centroid(&[vec![2.0, 0.0, 0.0], vec![0.0, 1.0]]), vec![1.0, 0.0, 0.0]);
    assert!(bank_centroid(&[]).is_empty());
    assert!(bank_centroid(&[vec![0.0, 0.0]]).is_empty());
    Ok(())
}

#[test]
fn row_axis_aggregate_relation() -> anyhow::Result<()> {
    assert!(row_axis_aggregate_related("item_net_weight", "weight_net"));
    assert!(row_axis_aggregate_related("unit_price", "amount"));
    assert!(!row_axis_aggregate_related("hs_code", "doc_number"));
    assert!(!row_axis_aggregate_related("item_no", "doc_number"));
    Ok(())
}

// ── T34 RAM · 생성 모델 장부 ─────────────────────────────────────

#[test]
fn ram_and_generation_bookkeeping() -> anyhow::Result<()> {
    assert_eq!(ram_used_pct(25, 100), 75.0);
    assert_eq!(ram_used_pct(0, 100), 100.0);
    assert_eq!(ram_used_pct(0, 0), 0.0);
    assert_eq!(ram_used_pct(150, 100), 0.0, "avail > total saturates to 0");

    assert_eq!(LogisModel::gen_kind(ModelSize::Qwen), 0);
    assert_eq!(LogisModel::gen_kind(ModelSize::Qwen3), 1);
    assert_eq!(LogisModel::gen_kind(ModelSize::Qwen3_5), 2);
    assert_eq!(LogisModel::measured_generation_mb(99), 0, "unknown kind has no measurement");
    Ok(())
}

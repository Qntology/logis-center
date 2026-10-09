//! 업무 규칙(`tauri_app_lib::logic`) 검증 (T01–T08)
//!
//! 상태 코드, 커머스 릴레이(related / relay), 무역 참조 그래프(related_trading / trading_relay_pair),
//! 앵커 구 뱅크, 필드 카테고리, 통화 정규화를 다룹니다.
//! 전부 순수 함수라 DB·모델·AppHandle 이 필요 없습니다.
//! `#[ignore = "BUG(Bn): …"]` 테스트는 '의도된 동작' 을 단언하며, 수정 전까지는 실패합니다.

use std::collections::{BTreeSet, HashSet};

use serde_json::json;
use tauri_app_lib::bias_schema::TRADE_DOC_TYPES;
use tauri_app_lib::logic::{
    anchor_phrases, canonical_currency_code, doc_type_to_code, is_trade_array_category,
    merge_phrase_bank, parse_status, related, related_trading, relay, relay_type_aliases,
    site_chrome_sentence, status_name, trade_code_anchor, trade_condition_fields,
    trade_default_operator, trade_field_category, trade_label_supplement, trade_reference_anchor,
    trade_reference_field_of, trade_title_bank_defs, trade_title_pairs, trading_index_column,
    trading_relay_field, trading_relay_pair, TRADE_GROUP_CODES, TRADE_HUB_TYPES,
    TRADE_REFERENCE_FIELDS,
};

/// TRADE_GROUP_CODES 에서 택배(TRACKING)를 뺀 무역 서식 코드 (55종)
fn trade_codes() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for (_, codes) in TRADE_GROUP_CODES.iter() {
        for c in codes.iter() {
            if *c != "TRACKING" {
                out.push(*c);
            }
        }
    }
    out
}

// ── T01 상태 코드 ────────────────────────────────────────────────

#[test]
fn status_codes_roundtrip() -> anyhow::Result<()> {
    for code in 1..=12i64 {
        let name = status_name(code).expect("status name for 1..=12");
        assert_eq!(parse_status(name) as i64, code, "{name}");
    }
    assert_eq!(status_name(0), None);
    assert_eq!(status_name(13), None);
    assert_eq!(status_name(-1), None);
    assert_eq!(parse_status("draft"), 10);
    assert_eq!(parse_status("progress"), 1);
    assert_eq!(parse_status("complete"), 9);
    assert_eq!(parse_status("cancel"), 3);
    // "pending" 은 상태 어휘가 아닙니다 (대기 = draft = 10). 모르는 값은 0.
    assert_eq!(parse_status("pending"), 0);
    assert_eq!(parse_status(""), 0);
    Ok(())
}

#[test]
#[ignore = "BUG(B16): parse_status is case-sensitive ('Complete' → 0, 'CANCEL' → 0)"]
fn status_parse_is_case_insensitive() -> anyhow::Result<()> {
    assert_eq!(parse_status("Complete"), 9);
    assert_eq!(parse_status("CANCEL"), 3);
    Ok(())
}

// ── T02 무역 서식 표 정합성 ──────────────────────────────────────

#[test]
fn trade_code_tables_are_consistent() -> anyhow::Result<()> {
    let codes = trade_codes();
    assert_eq!(codes.len(), 55);
    let group_set: BTreeSet<&str> = codes.iter().copied().collect();
    assert_eq!(group_set.len(), 55, "a code appears in two groups");
    let doc_types: BTreeSet<&str> = TRADE_DOC_TYPES.iter().copied().collect();
    assert_eq!(group_set, doc_types, "TRADE_GROUP_CODES vs bias_schema::TRADE_DOC_TYPES");

    let pairs = trade_title_pairs();
    for code in codes.iter().copied() {
        assert_eq!(doc_type_to_code(code), code);
        let field = trade_reference_field_of(code)
            .unwrap_or_else(|| panic!("no reference field for {code}"));
        assert!(TRADE_REFERENCE_FIELDS.contains(&field), "{code}: {field}");
        assert_eq!(trade_field_category(field), "header", "{field}");
        assert_eq!(trade_default_operator(field), "eq", "{field}");
        assert_ne!(trade_code_anchor(code), "trade document", "{code} has only the generic anchor");
        assert!(pairs.iter().any(|(c, _)| *c == code), "{code} has no title");

        let rel = related_trading(code);
        assert!(!rel.contains(&code), "{code} relates to itself");
        let uniq: HashSet<&str> = rel.iter().copied().collect();
        assert_eq!(uniq.len(), rel.len(), "{code}: duplicates in {rel:?}");
        for hub in TRADE_HUB_TYPES.iter().copied().filter(|h| *h != code) {
            assert!(rel.contains(&hub), "{code} misses hub {hub}");
        }
    }
    Ok(())
}

#[test]
fn trade_reference_fields_are_unique_and_anchored() -> anyhow::Result<()> {
    assert_eq!(TRADE_REFERENCE_FIELDS.len(), 53);
    let set: BTreeSet<&str> = TRADE_REFERENCE_FIELDS.iter().copied().collect();
    assert_eq!(set.len(), TRADE_REFERENCE_FIELDS.len(), "duplicate reference field");
    for f in TRADE_REFERENCE_FIELDS.iter().copied() {
        assert!(f.starts_with("reference_"), "{f}");
        assert_ne!(trade_reference_anchor(f), "referenced document number", "{f} has only the generic anchor");
    }
    let cond = trade_condition_fields("reference");
    assert_eq!(cond.len(), 53);
    for (f, desc, anchor) in cond.iter() {
        assert_eq!(*desc, "Referenced document number");
        assert_eq!(*anchor, trade_reference_anchor(f));
    }
    assert!(trade_condition_fields("nope").is_empty());
    Ok(())
}

// ── T03 무역 릴레이 방향 ─────────────────────────────────────────

#[test]
fn trading_relay_pair_and_related_trading() -> anyhow::Result<()> {
    // (내 문서에서 상대를 가리키는 필드, 상대 문서에서 나를 가리키는 필드)
    assert_eq!(trading_relay_pair("CI", "BL"), Some(("reference_bl", "reference_invoice")));
    assert_eq!(trading_relay_pair("BL", "CI"), Some(("reference_invoice", "reference_bl")));
    // 같은 서식의 다른 표기(필드 공유)·자기 자신·미지 코드는 릴레이하지 않습니다.
    for (a, b) in [("INS", "IP"), ("CA", "COA"), ("BC", "BK"), ("PHYTO", "PC"), ("CI", "CI"), ("CI", "XX")] {
        assert_eq!(trading_relay_pair(a, b), None, "{a}->{b}");
    }
    assert_eq!(trading_relay_field("PO", "LC"), Some("reference_lc"));
    assert_eq!(trading_index_column("Bl"), "rel_bl");

    assert_eq!(related_trading("PO"), vec!["PI", "SC", "EL", "CP", "LLC", "SOA", "CI", "BL", "LC"]);
    assert_eq!(related_trading("LC"), vec!["LLC", "LG", "TR", "SOA", "PO", "CI", "BL"]);
    assert_eq!(related_trading("XYZ"), vec!["PO", "CI", "BL", "LC"]);
    Ok(())
}

// ── T04 서식 이름 → 코드 ─────────────────────────────────────────

#[test]
fn doc_type_to_code_maps_titles_and_codes() -> anyhow::Result<()> {
    assert_eq!(doc_type_to_code("commercial invoice"), "CI");
    assert_eq!(doc_type_to_code("Bill of Lading"), "BL");
    assert_eq!(doc_type_to_code("bl"), "BL");
    assert_eq!(doc_type_to_code("Purchase Confirmation"), "CP");
    assert_eq!(doc_type_to_code("fumigation-certificate"), "FC");
    assert_eq!(doc_type_to_code("Certificate of Non-Manipulation"), "CNM");
    assert_eq!(doc_type_to_code("Konnossement"), "BL");
    assert_eq!(doc_type_to_code("unknown doc"), "UNKNOWN DOC");
    Ok(())
}

#[test]
#[ignore = "BUG(B11): 'certificate_of_analysis' resolves to COA while 'Certificate of Analysis' resolves to CA"]
fn doc_type_to_code_is_separator_invariant() -> anyhow::Result<()> {
    assert_eq!(doc_type_to_code("certificate_of_analysis"), doc_type_to_code("Certificate of Analysis"));
    assert_eq!(doc_type_to_code("certificate-of-analysis"), doc_type_to_code("CERTIFICATE OF ANALYSIS"));
    Ok(())
}

// ── T05 필드 → 카테고리 ──────────────────────────────────────────

#[test]
fn trade_field_category_known_fields() -> anyhow::Result<()> {
    let cases = [
        ("reference_anything", "header"),
        ("doc_number", "header"),
        ("insurance", "financials"),
        ("container_number", "containers"),
        ("hs_code", "items"),
        ("vessel", "logistics"),
        ("charge_code", "charges"),
        ("hazard_level", "hazmat"),
        ("customs_broker", "customs"),
        ("total_weight_kg", "cargo"),
        ("buyer_name", "parties"),
        ("unit_cost", ""),
    ];
    for (field, want) in cases {
        assert_eq!(trade_field_category(field), want, "{field}");
    }
    for cat in ["items", "containers", "other_parties", "charges", "test_results", "findings_and_damage", "account_ledger"] {
        assert!(is_trade_array_category(cat), "{cat}");
    }
    assert!(!is_trade_array_category("header"));
    Ok(())
}

#[test]
#[ignore = "BUG(B10): substring rules send importer_/exporter_ (\"port\") to logistics and corporate_ (\"rate\") to financials"]
fn trade_field_category_party_names() -> anyhow::Result<()> {
    assert_eq!(trade_field_category("importer_name"), "parties");
    assert_eq!(trade_field_category("exporter_address"), "parties");
    assert_eq!(trade_field_category("corporate_name"), "parties");
    Ok(())
}

// ── T06 통화 · 기본 연산자 ───────────────────────────────────────

#[test]
fn currency_and_operator_tables() -> anyhow::Result<()> {
    for (raw, want) in [
        ("usd", "USD"),
        ("US$", "USD"),
        ("U.S. dollar", "USD"),
        ("€", "EUR"),
        ("yen", "JPY"),
        ("RMB", "CNY"),
        ("₩", "KRW"),
        ("£", "GBP"),
        ("Kč", "CZK"),
    ] {
        assert_eq!(canonical_currency_code(raw), Some(want), "{raw}");
    }
    for raw in ["Won", "", "   ", "XYZ", "dollar"] {
        assert_eq!(canonical_currency_code(raw), None, "{raw:?}");
    }
    for (field, op) in [
        ("hub_reference", "contains"),
        ("reference_bl", "eq"),
        ("doc_number", "eq"),
        ("etd", "gte"),
        ("issue_date", "gte"),
        ("amount", "eq"),
        ("unit_price", "eq"),
        ("vessel", "contains"),
        ("description", "contains"),
    ] {
        assert_eq!(trade_default_operator(field), op, "{field}");
    }
    Ok(())
}

// ── T07 앵커 구 뱅크 ─────────────────────────────────────────────

#[test]
fn anchor_phrases_split_and_dedupe() -> anyhow::Result<()> {
    // 쉼표·파이프·슬래시로 나누고, 대소문자 무시 중복은 첫 표기만 남깁니다.
    assert_eq!(anchor_phrases("a, b, A", "c | B / x"), vec!["a", "b", "c", "x"]);
    // 1~2 글자 약어 사이의 슬래시(T/T, B/L)는 구분자가 아닙니다.
    assert_eq!(anchor_phrases("T/T, B/L", ""), vec!["T/T", "B/L"]);
    assert!(anchor_phrases("", "  ").is_empty());
    Ok(())
}

#[test]
fn merge_phrase_bank_appends_unique() -> anyhow::Result<()> {
    let mut phrases = vec!["a".to_string(), "B".to_string()];
    let mut weights: Vec<f32> = Vec::new();
    let extra = vec!["b".to_string(), " c ".to_string(), String::new(), "C".to_string()];
    assert_eq!(merge_phrase_bank(&mut phrases, &mut weights, &extra, 0.5), 1);
    assert_eq!(phrases, vec!["a", "B", "c"]);
    assert_eq!(weights, vec![1.0f32, 1.0, 0.5]);
    Ok(())
}

#[test]
fn site_chrome_and_label_supplements() -> anyhow::Result<()> {
    assert_eq!(site_chrome_sentence("en-US"), None);
    assert_eq!(site_chrome_sentence(""), None);
    assert_eq!(site_chrome_sentence("xx"), None);
    assert!(site_chrome_sentence("de-DE").expect("de").starts_with("globale Navigation"));
    assert!(site_chrome_sentence("DE").is_some());
    assert!(site_chrome_sentence("ko_KR").expect("ko").starts_with("전체 메뉴"));

    let po = trade_label_supplement("reference_po");
    assert_eq!(po.first().map(String::as_str), Some("P/O No."));
    assert!(po.iter().any(|p| p == "Bestellnummer"));
    assert!(trade_label_supplement("nope").is_empty());
    Ok(())
}

#[test]
fn trade_title_bank_defs_exclude_own_titles() -> anyhow::Result<()> {
    let pairs = trade_title_pairs();
    let (bias, prej) = trade_title_bank_defs("t");
    assert!(!bias.is_empty() && !prej.is_empty());
    assert!(bias.iter().chain(prej.iter()).all(|(cat, _, _)| cat == "t"));
    assert!(bias.iter().any(|(_, c, t)| c == "BC" && t == "booking confirmation"));

    // 편견 구에는 그 코드 자신의 표제가 들어가면 안 됩니다 (BC/BK 처럼 표제를 공유해도).
    for (_, code, title) in prej.iter() {
        let own = pairs
            .iter()
            .any(|(c, x)| *c == code.as_str() && x.eq_ignore_ascii_case(title));
        assert!(!own, "{code}: prejudice contains its own title {title:?}");
    }
    assert!(!prej
        .iter()
        .any(|(_, c, t)| c == "BC" && t.eq_ignore_ascii_case("booking confirmation")));

    // 코드별로 대소문자 무시 중복이 없어야 합니다.
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for (_, code, title) in prej.iter() {
        assert!(seen.insert((code.clone(), title.to_lowercase())), "dup prejudice {code} {title:?}");
    }
    Ok(())
}

// ── T08 커머스 릴레이 ────────────────────────────────────────────

#[test]
fn commerce_related_and_aliases() -> anyhow::Result<()> {
    assert_eq!(related("sales"), vec!["goods", "tracking", "coupon", "event"]);
    assert_eq!(related("receiving"), vec!["goods", "order", "coupon", "event"]);
    assert_eq!(related("coupon"), vec!["goods", "event"]);
    assert!(related("member").is_empty());
    assert!(relay_type_aliases("shipping").contains(&"waybill"));
    assert_eq!(relay_type_aliases("coupon"), relay_type_aliases("event"));
    assert!(relay_type_aliases("nothing").is_empty());
    Ok(())
}

#[test]
fn relay_goods_to_sales_order_uses_tracking_number() -> anyhow::Result<()> {
    let (queries, merge) = relay("goods", &json!({"type": "sales", "tracking_number": "T1"}))
        .expect("relay plan");
    assert_eq!(queries.len(), 1);
    let q = &queries[0];
    assert_eq!(q.r#type, "order");
    assert_eq!(q.table, "sales");
    assert_eq!(q.column, "tracking");
    assert_eq!(q.value, json!("T1"));
    assert_eq!(q.status, None);
    assert!(merge.update.is_none());
    assert_eq!((merge.from.as_str(), merge.to.as_str()), ("goods", "order"));
    let up = merge.upsert.expect("upsert merge");
    assert_eq!(up.includes.len(), 26);
    assert!(up.includes.iter().any(|k| k == "sale_price"));
    assert_eq!((up.from.as_str(), up.to.as_str()), ("goods", "order"));

    // index 도 tracking 도 없으면 계획이 없습니다.
    assert!(relay("goods", &json!({"type": "order"})).is_none());
    assert!(relay("goods", &json!({})).is_none());
    Ok(())
}

#[test]
fn relay_shipping_to_order_and_order_to_coupon() -> anyhow::Result<()> {
    // foreign 'shipping' 은 tracking 으로 접힙니다.
    let (queries, merge) = relay("shipping", &json!({"type": "order", "index": 5})).expect("plan");
    let q = &queries[0];
    assert_eq!((q.r#type.as_str(), q.table.as_str(), q.column.as_str()), ("tracking", "tracking", "order"));
    assert_eq!(q.value, json!(5));
    assert_eq!((merge.from.as_str(), merge.to.as_str()), ("tracking", "order"));
    let upd = merge.update.expect("update merge");
    assert_eq!(upd.includes, vec!["no", "goods", "event"]);
    assert_eq!(upd.column.as_deref(), Some("index"));
    assert_eq!(upd.value, Some(json!(5)));
    assert_eq!((upd.from.as_str(), upd.to.as_str()), ("tracking", "order"));
    let foreign = upd.foreign.expect("foreign info");
    assert_eq!((foreign.from.as_str(), foreign.to.as_str()), ("index", "tracking"));

    // order → coupon : from/to 가 coupon → order 방향으로 뒤집혀 있습니다.
    let (queries, merge) = relay("order", &json!({"type": "coupon", "index": 7})).expect("plan");
    let q = &queries[0];
    assert_eq!((q.r#type.as_str(), q.table.as_str(), q.column.as_str()), ("order", "sales", "event"));
    assert_eq!(q.value, json!(7));
    assert_eq!(q.status, Some(0));
    assert_eq!((merge.from.as_str(), merge.to.as_str()), ("coupon", "order"));
    let upd = merge.update.expect("update merge");
    assert_eq!(upd.includes, vec!["discount"]);
    assert_eq!(upd.column.as_deref(), Some("event"));
    assert_eq!(upd.value, Some(json!(7)));
    Ok(())
}

#[test]
fn relay_goods_to_tracking_primary() -> anyhow::Result<()> {
    let (queries, merge) = relay("goods", &json!({"type": "tracking", "goods": 1, "index": 2})).expect("plan");
    let q = &queries[0];
    assert_eq!((q.r#type.as_str(), q.table.as_str(), q.column.as_str()), ("order", "sales", "goods"));
    assert_eq!(q.value, json!(1));
    assert_eq!(q.status, Some(0));
    let upd = merge.update.expect("update merge");
    assert_eq!(upd.includes.len(), 8);
    assert_eq!(upd.column.as_deref(), Some("index"));
    assert_eq!(upd.value, Some(json!(2)));
    assert_eq!((upd.from.as_str(), upd.to.as_str()), ("goods", "tracking"));
    Ok(())
}

#[test]
#[ignore = "BUG(B22): relay() folds a foreign 'shipping'/'receiving' into tracking but not a primary one"]
fn relay_accepts_shipping_as_primary_tracking() -> anyhow::Result<()> {
    assert!(relay("goods", &json!({"type": "shipping", "goods": 1, "index": 2})).is_some());
    assert!(relay("goods", &json!({"type": "receiving", "goods": 1, "index": 2})).is_some());
    Ok(())
}

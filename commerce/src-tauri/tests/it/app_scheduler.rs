//! 스케줄러 계층(`tauri_app_lib::scheduler`) 검증 (T13–T20)
//!
//! - 엔터티 키(entity_* / relay_*): `entity` 모듈은 private 이라 `scheduler` 의 pub use 재노출로만 접근합니다.
//! - 릴레이 원장(`scheduler::relay_ledger`): DB 가 필요 없는 순수 헬퍼만 다룹니다.
//! - 반복 구조 센서스(`scheduler::list_census`): 고정 HTML 로 그룹 · 점수 · 발췌 · 빈 표 판정을 확인합니다.
//! 기간/상태 헬퍼(iso_of · in_period · status_is_void …)는 private 이라 여기서 다루지 못합니다.
//! `#[ignore = "BUG(Bn): …"]` 테스트는 '의도된 동작' 을 단언하며, 수정 전까지는 실패합니다.

use serde_json::{json, Value};
use tauri_app_lib::scheduler::list_census::{self, harvest_titles};
use tauri_app_lib::scheduler::relay_ledger::{
    add_delta, bind_tracking_ref, goods_array_refs, mark_relay_bound, placeholder_origin_establishes,
    self_established, settle_relay_keys, BridgeOutcome, StatsDiff,
};
use tauri_app_lib::scheduler::{
    entity_bcc, entity_id, entity_index, entity_key_index, entity_seed, normalize_entity_key, relay_seed,
    relay_type_key,
};
use tauri_app_lib::utils::canonical::LedgerPrior;
use tauri_app_lib::utils::hash::hash_id;

const TEAM: &str = "0xteam000000000000000000000000000000000001";
const TEAM2: &str = "0xteam000000000000000000000000000000000002";

/// thead 1행 + tbody 3행 (4칸) 주문 표
const ORDERS: &str = concat!(
    r#"<html><body><table id="orders"><thead><tr><th>No</th><th>Item</th><th>Qty</th><th>Price</th></tr></thead>"#,
    r#"<tbody><tr><td>1</td><td>Shirt</td><td>2</td><td>10</td></tr>"#,
    r#"<tr><td>2</td><td>Pants</td><td>1</td><td>20</td></tr>"#,
    r#"<tr><td>3</td><td>Hat</td><td>5</td><td>5</td></tr></tbody></table></body></html>"#,
);

// ── T13 엔터티 키 ────────────────────────────────────────────────

#[test]
fn normalize_entity_key_strips_separators_and_folds_case() -> anyhow::Result<()> {
    for (raw, want) in [
        ("bl-5543 2219", "BL55432219"),
        // S 로 시작하는 서식 코드에 숫자 호모글리프 치환(S→5)을 하지 않습니다.
        ("SC-2026-0802", "SC20260802"),
        ("task_1787731795587", "TASK1787731795587"),
        // 순수 숫자열만 호모글리프(전각 숫자 포함)를 접습니다.
        ("１２３-４５６", "123456"),
        ("--", ""),
        ("", ""),
    ] {
        assert_eq!(normalize_entity_key(raw), want, "{raw:?}");
    }
    Ok(())
}

#[test]
fn relay_type_key_and_seed_rules() -> anyhow::Result<()> {
    for (t, want) in [
        ("bl", "BL"),
        (" CI ", "CI"),
        ("Sales", "order"),
        ("order", "order"),
        ("receiving", "tracking"),
        ("shipping", "tracking"),
        ("goods", "goods"),
        ("coupon", "coupon"),
        ("REVIEW", "review"),
    ] {
        assert_eq!(relay_type_key(t), want, "{t:?}");
    }

    // 무역 서식 · 유효 송장번호는 스코프 없이, 나머지는 '스코프:값'
    assert_eq!(relay_seed("BL", "a.com", " X "), "X");
    assert_eq!(relay_seed("order", "a.com", "X"), "a.com:X");
    assert_eq!(relay_seed("order", "", "X"), "X");
    assert_eq!(relay_seed("goods", "a", "  "), "");
    assert_eq!(relay_seed("tracking", "a.com", "603145678912"), "603145678912");
    assert_eq!(relay_seed("tracking", "a.com", "1234"), "a.com:1234");

    assert_eq!(entity_seed("shop.com", " 12345 "), "shop.com:12345");
    assert_eq!(entity_seed("shop.com", "603145678912"), "603145678912");
    assert_eq!(entity_seed("", "X-1"), "X-1");
    Ok(())
}

#[test]
fn entity_key_index_scoping_rules() -> anyhow::Result<()> {
    // 무역 서식: 스코프 · 구분자 · 대소문자 무관, entity_index 와 동일
    let ci = entity_key_index("ci", TEAM, "a.com", "CI-2026-08001");
    assert_eq!(ci, entity_key_index("CI", TEAM, "b.com", "ci 2026 08001"));
    assert_eq!(ci, entity_index("CI", TEAM, "CI-2026-08001"));

    // 커머스 주문: 사이트(스코프)마다 다른 문서, sales == order
    let order = entity_key_index("order", TEAM, "a.com", "123456");
    assert_ne!(order, entity_key_index("order", TEAM, "b.com", "123456"));
    assert_eq!(order, entity_key_index("sales", TEAM, "a.com", "123456"));

    // 유효한 송장번호는 스코프 무관(receiving == tracking), 짧은 번호는 스코프에 묶임
    assert_eq!(
        entity_key_index("receiving", TEAM, "a.com", "603145678912"),
        entity_key_index("tracking", TEAM, "b.com", "603145678912")
    );
    assert_ne!(
        entity_key_index("tracking", TEAM, "a.com", "1234"),
        entity_key_index("tracking", TEAM, "b.com", "1234")
    );

    // 팀이 다르면 다른 문서
    assert_ne!(ci, entity_key_index("CI", TEAM2, "a.com", "CI-2026-08001"));
    Ok(())
}

#[test]
fn entity_id_and_bcc_derive_from_hash_id() -> anyhow::Result<()> {
    let id = entity_id(TEAM, 7);
    assert_eq!(id, hash_id(&format!("{}{}", TEAM, 7)));
    assert_eq!(id.len(), 42);
    assert!(id.starts_with("0x"));
    assert_ne!(id, entity_id(TEAM, 8));
    assert_eq!(entity_bcc("goods", "shop.com"), hash_id("goodsshop.com"));
    Ok(())
}

#[test]
#[ignore = "BUG(B12): normalize_entity_key does not fold full-width alphanumerics (hash::normalize_identifier does)"]
fn entity_key_folds_full_width() -> anyhow::Result<()> {
    assert_eq!(normalize_entity_key("ＣＩ－４３７２６"), "CI43726");
    assert_eq!(entity_index("CI", TEAM, "ＣＩ－４３７２６"), entity_index("CI", TEAM, "CI-43726"));
    Ok(())
}

// ── T14 릴레이 원장 헬퍼 ─────────────────────────────────────────

#[test]
fn stats_delta_and_relay_bound_marks() -> anyhow::Result<()> {
    let mut stats = StatsDiff::new();
    add_delta(&mut stats, "goods", (1, 0, 1));
    add_delta(&mut stats, "goods", (0, 1, 0));
    add_delta(&mut stats, "", (1, 1, 1));
    add_delta(&mut stats, "  ", (1, 1, 1));
    add_delta(&mut stats, "order", (0, 0, 0));
    assert_eq!(stats.len(), 1, "{stats:?}");
    assert_eq!(stats["goods"], (1, 1, 1));

    let mut item = json!({});
    mark_relay_bound(&mut item, "tracking");
    mark_relay_bound(&mut item, "tracking");
    assert_eq!(item, json!({"_relay_bound": ["tracking"]}));
    mark_relay_bound(&mut item, "goods");
    assert_eq!(item, json!({"_relay_bound": ["tracking", "goods"]}));

    let mut broken = json!({"_relay_bound": "oops"});
    mark_relay_bound(&mut broken, "tracking");
    assert_eq!(broken, json!({"_relay_bound": ["tracking"]}));

    let mut scalar = json!(5);
    mark_relay_bound(&mut scalar, "tracking");
    assert_eq!(scalar, json!(5));
    Ok(())
}

#[test]
fn settle_relay_keys_moves_text_to_title_companions() -> anyhow::Result<()> {
    // 자기 타입 키(order)는 그대로, 숫자 index(tracking)도 그대로, 텍스트 링크 키는 *_title 로 이동
    let mut item = json!({"goods": "Blue Shirt", "order": "null", "tracking": 123, "event": " "});
    let moved = settle_relay_keys(&mut item, "order");
    assert_eq!(moved, ["goods"]);
    assert_eq!(item, json!({"order": "null", "tracking": 123, "goods_title": "Blue Shirt"}));

    // 이미 채워진 companion 은 덮어쓰지 않습니다.
    let mut kept = json!({"goods": "Hat", "goods_title": "Wool Hat"});
    assert_eq!(settle_relay_keys(&mut kept, "sales"), ["goods"]);
    assert_eq!(kept, json!({"goods_title": "Wool Hat"}));
    Ok(())
}

#[test]
fn ledger_establishment_rules() -> anyhow::Result<()> {
    let placeholder = |origin: Value| json!({"ledger": "placeholder", "relay_origin": origin});
    assert!(placeholder_origin_establishes("goods", Some(&placeholder(json!(["order"])))));
    assert!(placeholder_origin_establishes("goods", Some(&placeholder(json!(["review", "tracking"])))));
    assert!(!placeholder_origin_establishes("goods", Some(&placeholder(json!(["review"])))));
    assert!(!placeholder_origin_establishes("goods", Some(&placeholder(json!(["goods"])))));
    assert!(!placeholder_origin_establishes("goods", None));
    // 확정(count) 장부는 placeholder 가 아니므로 해당 없음
    let confirmed = json!({"ledger": "count", "relay_origin": ["order"]});
    assert!(!placeholder_origin_establishes("goods", Some(&confirmed)));

    let none = BridgeOutcome::default();
    assert!(!self_established(LedgerPrior::Draft, false, &none));
    assert!(!self_established(LedgerPrior::Placeholder, false, &none));
    assert!(self_established(LedgerPrior::Confirmed, false, &none));
    assert!(self_established(LedgerPrior::Draft, true, &none));
    let referenced = BridgeOutcome { referenced: true, ..Default::default() };
    assert!(self_established(LedgerPrior::Draft, false, &referenced));
    let outgoing = BridgeOutcome { establishing_out: 1, ..Default::default() };
    assert!(self_established(LedgerPrior::Absent, false, &outgoing));
    Ok(())
}

// ── T15 참조 바인딩 ──────────────────────────────────────────────

#[test]
fn bind_tracking_ref_only_for_order_pages() -> anyhow::Result<()> {
    let mut sale = json!({"tracking_number": "603-145-678-912"});
    let idx = bind_tracking_ref(&mut sale, "sales", TEAM, "shop.com");
    // 유효 송장번호는 스코프 무관 → 다른 사이트에서 계산해도 같은 index
    let want = entity_key_index("tracking", TEAM, "other.com", "603145678912");
    assert_eq!(idx, Some(want));
    assert_eq!(sale["tracking"], json!(want));
    assert_eq!(sale["_relay_bound"], json!(["tracking"]));

    let mut goods = json!({"tracking_number": "603-145-678-912"});
    assert_eq!(bind_tracking_ref(&mut goods, "goods", TEAM, "shop.com"), None);
    assert_eq!(goods, json!({"tracking_number": "603-145-678-912"}));

    // 송장번호가 없으면 기존 tracking index 를 그대로 씁니다 (0 은 미연결).
    assert_eq!(bind_tracking_ref(&mut json!({"tracking": 77}), "order", TEAM, "shop.com"), Some(77));
    assert_eq!(bind_tracking_ref(&mut json!({"tracking": 0}), "order", TEAM, "shop.com"), None);
    Ok(())
}

#[test]
fn goods_array_refs_index_each_element() -> anyhow::Result<()> {
    let mut item = json!({"goods": [{"id": "G-1001"}, {"no": "G-1001"}, {"id": ""}, {"x": 1}]});
    let refs = goods_array_refs(&mut item, TEAM, "c.com");
    let gi = entity_key_index("goods", TEAM, "c.com", "G-1001");
    assert_eq!(refs, vec![("goods".to_string(), gi)], "the same goods is referenced once");
    assert_eq!(item["goods"][0]["index"], json!(gi));
    assert_eq!(item["goods"][1]["index"], json!(gi));
    assert!(item["goods"][2].get("index").is_none());
    assert!(item["goods"][3].get("index").is_none());
    // 커머스 상품은 사이트 스코프에 묶입니다.
    assert_ne!(gi, entity_key_index("goods", TEAM, "d.com", "G-1001"));
    assert!(goods_array_refs(&mut json!({"goods": "Blue Shirt"}), TEAM, "c.com").is_empty());
    Ok(())
}

// ── T17 · T18 반복 구조 센서스 ───────────────────────────────────

#[test]
fn census_finds_order_rows() -> anyhow::Result<()> {
    let census = list_census::run(ORDERS);
    assert_eq!(census.groups.len(), 2, "{:?}", census.groups);

    let g0 = &census.groups[0];
    assert_eq!(g0.tag, "tr");
    assert_eq!(g0.members, 3);
    assert_eq!(g0.parent_sig, "table#orders");
    assert_eq!(g0.item_selector, "table#orders tr");
    assert!(g0.exact && !g0.form_like);
    assert_eq!(g0.avg_cells, 4.0);
    assert_eq!(g0.distinct_ratio, 1.0);
    assert_eq!(g0.score, 12.0);
    assert!((g0.mirror - 1.0 / 9.0).abs() < 1e-4, "mirror {}", g0.mirror);
    assert_eq!(g0.samples, ["1 Shirt 2 10", "2 Pants 1 20", "3 Hat 5 5"]);
    assert!(g0.structural_grade() && g0.data_grade() && !g0.mirror_copy());
    assert_eq!(
        g0.selector_json(),
        json!({"parent": "table#orders", "itemSelector": "table#orders tr", "matchCount": 3})
    );

    // 머리행 th 4개도 후보지만 칸이 1개라 구조급이 아닙니다.
    let g1 = &census.groups[1];
    assert_eq!((g1.tag.as_str(), g1.members), ("th", 4));
    assert_eq!(g1.score, 4.0);
    assert!(!g1.structural_grade());

    // 각 행 안의 td 묶음 3개는 tr 그룹과 겹쳐 억제됩니다.
    assert_eq!(census.suppressed, 3);
    assert_eq!(census.landmark_dropped, 0);
    assert!(census.empty_tables.is_empty());
    assert_eq!(census.data_grade_count(), 1);
    assert!(census.empty_verdict().is_none());

    let lines = census.report_lines();
    assert_eq!(lines.len(), 3, "{lines:?}");
    assert!(lines[0].contains("반복 구조 후보 2개"), "{}", lines[0]);

    // 내용 마진이 없으면 결정적 폴백을 내지 않고, 양수 마진이 붙으면 tr 그룹을 냅니다.
    assert!(census.decisive_fallback().is_none());
    let mut scored = census.clone();
    scored.groups[0].content_margin = Some(0.1);
    let fallback = scored.decisive_fallback().expect("dominant data-grade rows");
    assert_eq!(fallback.item_selector, "table#orders tr");
    Ok(())
}

#[test]
fn census_excerpt_anchor_and_shadow() -> anyhow::Result<()> {
    let census = list_census::run(ORDERS);

    let ex = census.title_excerpt(ORDERS).expect("dominant row group");
    assert!(ex.text.starts_with("[LIST ROWS] table#orders tr | rows 3 (first 3 shown) | "), "{}", ex.text);
    assert!(ex.text.contains("[HEADER] No | Item | Qty | Price\n"), "{}", ex.text);
    assert!(ex.text.contains("[ROW 1]\n1\nShirt\n2\n10\n[ROW 2]\n2\nPants\n1\n20\n"), "{}", ex.text);
    assert!(ex.text.contains("[ROW 3]\n3\nHat\n5\n"), "{}", ex.text);
    assert_eq!((ex.rows, ex.members, ex.header_cells), (3, 3, 4));
    assert_eq!(ex.selector, "table#orders tr");
    assert!((ex.dominance - 3.0).abs() < 1e-6, "dominance {}", ex.dominance);

    let titles: Vec<String> = ["shirt", "Hat", "x"].iter().map(|s| s.to_string()).collect();
    let anchored = census.anchor_titles(&titles).expect("titles appear in the rows");
    assert_eq!(anchored.item_selector, "table#orders tr");
    assert!(census.anchor_titles(&["socks".to_string()]).is_none());

    let (agree, top, rows, boa) = census
        .shadow_agreement(ORDERS, "table#orders tbody tr")
        .expect("a structural group exists");
    assert_eq!(agree, 1.0);
    assert_eq!(top.item_selector, "table#orders tr");
    assert_eq!((rows, boa), (3, 3));
    Ok(())
}

// ── T19 빈 표 · 랜드마크 · 폼 ────────────────────────────────────

#[test]
fn census_empty_table_landmark_and_form() -> anyhow::Result<()> {
    let empty = list_census::run(
        r#"<table class="list"><thead><tr><th>A</th><th>B</th><th>C</th></tr></thead><tbody><tr><td colspan="3">No data</td></tr></tbody></table>"#,
    );
    assert_eq!(empty.empty_tables.len(), 1);
    let t = &empty.empty_tables[0];
    assert_eq!(
        (t.selector.as_str(), t.header_cells, t.body_rows, t.notice.as_str()),
        ("table.list", 3, 1, "No data")
    );
    assert!(empty.groups.iter().all(|g| !g.structural_grade()));
    let verdict = empty.empty_verdict().expect("a header-only table is an empty list");
    assert_eq!(verdict.selector, "table.list");

    let nav = list_census::run("<nav><ul><li>Home</li><li>About</li><li>Contact</li></ul></nav>");
    assert!(nav.groups.is_empty(), "{:?}", nav.groups);
    assert_eq!(nav.landmark_dropped, 1);

    let form = list_census::run(
        r#"<table id="f"><tr><th>Name</th><td><input name="n"></td></tr><tr><th>Email</th><td><input name="e"></td></tr></table>"#,
    );
    assert_eq!(form.groups.len(), 1, "{:?}", form.groups);
    let g = &form.groups[0];
    assert_eq!((g.tag.as_str(), g.members), ("tr", 2));
    assert!(g.form_like, "rows led by <th> labels are a form");
    assert!(!g.structural_grade());
    Ok(())
}

// ── T20 제목 수확 ────────────────────────────────────────────────

#[test]
fn harvest_titles_prefers_parsed_json_then_raw_brackets() -> anyhow::Result<()> {
    // 선호 키 배열: 숫자뿐인 값과 중복은 버립니다.
    assert_eq!(
        harvest_titles(&json!({"order": ["Shirt", "Pants", "12,345", "Shirt"]}), ""),
        ["Shirt", "Pants"]
    );
    // 객체 배열: title > name > text > product > goods 순으로 첫 값
    assert_eq!(
        harvest_titles(&json!({"items": [{"name": "A"}, {"title": "B", "name": "X"}]}), ""),
        ["A", "B"]
    );
    // 키가 하나뿐인 문자열 객체
    assert_eq!(harvest_titles(&json!({"x": "Only"}), ""), ["Only"]);
    // 200자 초과 제목은 버립니다.
    let long = "a".repeat(201);
    assert_eq!(harvest_titles(&json!({"order": [long, "Ok"]}), ""), ["Ok"]);
    // 최상위 배열
    assert_eq!(harvest_titles(&json!(["P1", {"product": "P2"}]), ""), ["P1", "P2"]);

    // 파싱 실패(Null / 빈 객체)일 때만 원문의 첫 '[' 이후 문자열 리터럴을 줍습니다 (잘린 JSON 포함).
    assert_eq!(
        harvest_titles(&Value::Null, r#"{"titles": ["Alpha", "Be\"ta", "12.5""#),
        ["Alpha", "Be\"ta"]
    );
    assert_eq!(harvest_titles(&json!({}), r#"["Raw One", "Raw Two"]"#), ["Raw One", "Raw Two"]);
    assert!(harvest_titles(&json!({}), "no brackets").is_empty());
    assert_eq!(harvest_titles(&json!({"order": ["Shirt"]}), r#"["Other"]"#), ["Shirt"]);
    Ok(())
}

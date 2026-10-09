//! 스키마 계층 검증 — bias_schema / prompts / parsing(PUG·무역 릴레이) / nl_convert / ai_utils / pug_utils
//!               + openai_types serde / chat_template / tokenizer
//!
//! - bias.json(include_str!) 은 serde_json 기본(BTreeMap) 순서로 순회됩니다. 순서가 드러나는 단언은 그 순서를 따릅니다.
//! - nl_convert 는 score_dynamics::record_* 에 닿는 함수(sanitize_transliteration*, gate_native_alias,
//!   format_gate_for_indexing, plinko …)를 호출하지 않습니다.
//! - 파일이 필요한 테스트(chat_template / tokenizer)는 common::TempDir 안에서만 씁니다.

use crate::common::TempDir;
use chrono::NaiveDate;
use scraper::Html;
use serde_json::{json, Value};
use tauri_app_lib::bias_schema as bias;
use tauri_app_lib::chat_template::{get_template, ChatTemplate};
use tauri_app_lib::nl_convert as nl;
use tauri_app_lib::openai_types::{
    ChatCompletionParameters, ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPart,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent, ImageURL,
};
use tauri_app_lib::parsing::{self, PugMode};
use tauri_app_lib::prompts::get_trade_category_schema;
use tauri_app_lib::tokenizer::TokenizerModel;
use tauri_app_lib::utils::ai_utils::{self as ai, FieldFormat};
use tauri_app_lib::utils::hash::{hash_id, relay_index};
use tauri_app_lib::utils::pug_utils::{find_block_indices_in_pug, parse_pug_grid, HeaderGrid};

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn strings(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| s.to_string()).collect()
}

// ═════════════════════════════ bias_schema ═════════════════════════════

#[test]
fn trade_doc_registry_and_type_predicates() {
    let dict: &Value = &bias::BIAS_DICT;
    for key in ["trade_schema", "ko", "en", "insight", "search_bridge"] {
        assert!(dict.get(key).is_some(), "bias.json missing '{key}'");
    }

    // 55종, 중복 없음, 전부 overlay 를 가짐
    assert_eq!(bias::TRADE_DOC_TYPES.len(), 55);
    let uniq: std::collections::HashSet<&&str> = bias::TRADE_DOC_TYPES.iter().collect();
    assert_eq!(uniq.len(), 55);
    for code in bias::TRADE_DOC_TYPES {
        assert!(dict["trade_schema"]["overlay"][*code].is_object(), "overlay missing for {code}");
    }

    assert!(bias::is_trade_doc_type("BL"));
    assert!(bias::is_trade_doc_type("bl"));
    assert!(bias::is_trade_doc_type(" shipping_doc "));
    assert!(!bias::is_trade_doc_type("invoice"));
    assert!(!bias::is_trade_doc_type(""));
    assert_eq!(bias::canonical_trade_doc_code("bl"), Some("BL"));
    assert_eq!(bias::canonical_trade_doc_code("invoice"), None);
    assert_eq!(bias::canonical_bias_type("Ci"), "shipping_doc");
    assert_eq!(bias::canonical_bias_type("goods"), "goods");

    assert!(bias::is_trade_array_category("items"));
    assert!(bias::is_trade_array_category("other_parties"));
    assert!(!bias::is_trade_array_category("parties"));
    assert!(!bias::is_trade_array_category("Items"));

    assert!(bias::is_system_axis("id,link"));
    assert!(bias::is_system_axis("status"));
    assert!(bias::is_system_axis("doc_type"));
    assert!(!bias::is_system_axis("id"));
}

#[test]
fn lang_code_of_normalizes_tags_and_language_names() {
    let cases = [
        ("ko-KR", "ko"),
        ("en-US", "en"),
        ("zh-TW", "zh-tw"),
        ("zh-Hant", "zh-tw"),
        ("zh-hans", "zh"),
        ("Korean", "ko"),
        ("한국어", "ko"),
        ("malayalam", "ml"), // 'malay' 보다 먼저 매칭
        ("Chinese (Traditional)", "zh-tw"),
        ("pt-BR", "pt"),
        ("", "en"),
        ("x", "en"),
    ];
    for (input, expected) in cases {
        assert_eq!(bias::lang_code_of(input), expected, "lang_code_of({input:?})");
    }
    assert_eq!(bias::lang_names_of("es"), vec!["spanish", "espanol"]);
}

#[test]
#[ignore = "BUG(B24): 'Traditional Chinese' falls through to the 2-letter prefix and becomes 'tr' (Turkish)"]
fn lang_code_of_handles_traditional_chinese_word_order() {
    assert_eq!(bias::lang_code_of("Traditional Chinese"), "zh-tw");
}

#[test]
fn trade_schema_triples_load_base_plus_overlay() {
    let ci = bias::trade_schema_triples("CI");
    assert_eq!(ci.len(), 89);
    assert_eq!(bias::trade_schema_triples("BL").len(), 82);
    assert_eq!(bias::trade_schema_triples("ZZZ").len(), 81, "unknown code = base only");
    assert_eq!(
        ci[0],
        (
            "cargo".to_string(),
            "chargeable_weight".to_string(),
            "Chargeable weight for air freight {Number}".to_string()
        )
    );
    let doc_number = ci
        .iter()
        .find(|(c, f, _)| c == "header" && f == "doc_number")
        .expect("CI header.doc_number");
    // overlay 가 base 설명을 덮어씀
    assert_eq!(
        doc_number.2,
        "The INVOICE NUMBER of this commercial invoice printed under the title {String}"
    );

    // 모든 서식이 문서 기본키(header.doc_number)를 가짐
    for code in bias::TRADE_DOC_TYPES {
        let t = bias::trade_schema_triples(code);
        assert!(t.len() >= 81, "{code}");
        assert!(t.iter().any(|(c, f, _)| c == "header" && f == "doc_number"), "{code}");
    }

    // 별칭 병합판은 앵커 문자열에 표기 별칭을 덧붙임
    let bl = bias::canonical_trade_triples("BL");
    let dn = bl.iter().find(|(_, f, _)| f == "doc_number").expect("BL doc_number");
    assert!(dn.2.contains("B/L no"), "{}", dn.2);
}

#[test]
#[ignore = "BUG(B6): trade_schema_triples/get_detail_schema_fields are case-sensitive, so server-synced lowercase 'bl' loses its overlay and doc_number"]
fn lowercase_trade_codes_load_the_same_schema() {
    assert_eq!(bias::trade_schema_triples("bl").len(), 82);
    let fields = bias::get_detail_schema_fields("bl", "", "en");
    assert!(fields.iter().any(|f| f.0 == "doc_number"));
    assert_eq!(fields.len(), 84);
}

#[test]
fn schema_field_counts_per_page_type() {
    let counts = [
        ("CI", 91),
        ("BL", 84),
        ("goods", 46),
        ("order", 22),
        ("tracking", 21),
        ("review", 7),
        ("coupon", 21),
        ("event", 22),
        ("click", 7),
        ("foo", 4),
    ];
    for (page_type, n) in counts {
        assert_eq!(bias::get_detail_schema_fields(page_type, "", "en").len(), n, "detail({page_type})");
    }

    // insight 축은 target_domain 에 든 페이지 타입에만, BTreeMap 순서로 앞에 붙음
    let goods = bias::get_detail_schema_fields("goods", "", "ko");
    let keys: Vec<&str> = goods.iter().take(3).map(|f| f.0.as_str()).collect();
    assert_eq!(keys, vec!["general_insight", "traffic_insight", "id,link"]);
    assert_eq!(
        goods[2].1,
        "- \"link\": String. Detailed page URL.\n- \"id\": String. Refer to the ID value from the link."
    );

    let bl = bias::get_detail_schema_fields("BL", "", "en");
    assert_eq!(&bl[0].0, "id,link");
    assert_eq!(&bl[1].0, "status");
    let pc = bl.iter().find(|f| f.0 == "package_count").expect("BL package_count");
    assert!(pc.1.starts_with("- \"package_count\": Number."), "{}", pc.1);
    for code in ["CI", "AWB", "PO", "HBL", "FC"] {
        assert!(
            bias::get_detail_schema_fields(code, "", "en").iter().any(|f| f.0 == "doc_number"),
            "{code}"
        );
    }

    assert_eq!(bias::get_list_schema_fields("goods", "", "ko").len(), 16);
    assert_eq!(bias::get_list_schema_fields("tracking", "", "en").len(), 8);
}

#[test]
fn field_names_localized_types_and_layout_bias() {
    assert_eq!(bias::canonical_field_name("no"), "doc_number");
    assert_eq!(bias::canonical_field_name("gross_weight"), "weight_gross");
    assert_eq!(bias::canonical_field_name(" bl_number "), "doc_number");
    assert_eq!(bias::canonical_field_name("doc_number"), "doc_number");
    assert_eq!(bias::canonical_field_name("unknown_field_xyz"), "unknown_field_xyz");

    let cases = [
        ("order", "ko", "주문"),
        ("tracking", "ja", "配送"),
        ("event", "ko", "이벤트"),
        ("unknown", "zh-TW", "文件"),
        ("goods", "fr", "produit"),
        ("x", "en", "document"),
    ];
    for (page_type, lang, expected) in cases {
        assert_eq!(bias::get_localized_page_type(page_type, lang), expected, "{page_type}/{lang}");
    }

    let (b, p) = bias::get_layout_bias("goods", "ko");
    assert_eq!(
        b,
        "detail has_list has_form true false 상품목록, 상품리스트, 진열관리, 재고관리 상품상세, 상품폼, 상품입력, 상품수정, 상품등록, 상품수정, 옵션추가, 상세설명에디터, 썸네일등록, 연관상품설정, input, select, textarea 상품 input 상품 select 상품 textarea"
    );
    assert_eq!(p, "global navigation, menus, footers, aside. global navigation, menus, footers, aside.");

    let ignore = bias::get_multi_pass_contexts("ignore", "en");
    assert_eq!(ignore.len(), 1);
    assert_eq!(ignore[0].0, "ignore");
    assert!(!ignore[0].1.is_empty());

    let goods = bias::get_multi_pass_contexts("goods", "ko");
    assert_eq!(
        goods[0],
        (
            "core_search_intent".to_string(),
            "goods 상품 goods, goods 상품 product, goods 상품 catalog, goods 상품 exposure, goods 상품 traffic, goods 상품 page views, goods 상품 clicks".to_string(),
            String::new()
        )
    );
}

#[test]
#[ignore = "BUG(B25): the Kannada label for 'order' contains a Telugu letter (U+0C21) instead of Kannada U+0CA1"]
fn localized_page_type_kannada_uses_kannada_script() {
    let s = bias::get_localized_page_type("order", "kn");
    assert!(s.chars().all(|c| ('\u{0C80}'..='\u{0CFF}').contains(&c)), "{s:?}");
}

// ═════════════════════════════ prompts ═════════════════════════════

#[test]
fn trade_category_schema_prompt_contract() {
    let s = get_trade_category_schema("header", "CI");
    assert!(s.starts_with("RULES: Output JSON ONLY. Every value in SCHEMA is null on purpose"), "{s}");
    assert!(s.contains("MISSION: Extract data for category 'HEADER' of a CI document."));
    assert!(s.contains("IDENTITY RULE"));
    // 정의 블록과 값 블록이 분리되고 doc_number 가 첫 필드
    assert!(s.contains("[FIELD DEFINITIONS]\n- \"doc_number\" (String): "), "{s}");
    assert!(s.contains("SCHEMA:\n{\n  \"doc_number\": null"), "{s}");
    assert!(s.contains("\"reference_po\": null"));
    // 자기참조 축과 doc_type 은 스키마에서 제거
    assert!(!s.contains("\"reference_invoice\": null"));
    assert!(!s.contains("\"doc_type\": null"));

    let po = get_trade_category_schema("header", "PO");
    assert!(!po.contains("\"reference_po\": null"));
    assert!(po.contains("\"reference_invoice\": null"));

    // 표 카테고리는 원소 두 개짜리 배열 모양 + 표 규칙
    let items = get_trade_category_schema("items", "CI");
    assert!(items.contains("SCHEMA:\n[\n  {\n"), "{items}");
    assert!(items.contains("[TABLE RULES]"));
    assert!(items.contains("\"unit_price\": null"));

    // 필드가 없는 카테고리는 빈 스키마 (get_trade_doc_categories 가 이 문자열로 판정)
    assert_eq!(
        get_trade_category_schema("nonexistent_category", "CI"),
        "RULES: Output JSON ONLY. MISSION: Extract data for category 'NONEXISTENT_CATEGORY'.\nSCHEMA:\n{}"
    );
    assert!(get_trade_category_schema("charges", "CI").ends_with("SCHEMA:\n{}"));
    assert!(!get_trade_category_schema("charges", "AWB").ends_with("SCHEMA:\n{}"));

    let base = ["header", "parties", "other_parties", "logistics", "conditions", "financials", "cargo", "items", "containers"];
    assert_eq!(parsing::get_trade_doc_categories("CI"), base.to_vec());
    let mut awb = base.to_vec();
    awb.push("charges");
    assert_eq!(parsing::get_trade_doc_categories("AWB"), awb);
}

// ═════════════════════════════ parsing : 컬럼 / 표 ═════════════════════════════

#[test]
fn canonicalize_trade_column_and_label_echo() {
    let cases = [
        ("Description of Goods", "en", "description"),
        ("QTY", "en", "quantity"),
        ("Unit Weight (kg)", "en", "item_net_weight"), // 가장 긴 별칭 우선
        ("Amount (USD)", "en", "total_price"),
        ("Unit Price", "en", "unit_price"),
        ("Price", "en", "unit_price"),
        ("hs-code", "en", "hs_code"),
        ("N.W.", "en", "item_net_weight"),
        ("G.W.", "en", "item_gross_weight"),
        ("CTNS", "en", "item_package_count"),
        ("Model No.", "en", "item_code"),
        ("品名", "ko", ""), // 한자 별칭은 ko 문서에서 제외
        ("品名", "zh", "description"),
        ("수량", "ko", "quantity"),
        ("수량", "en", ""),
        ("", "en", ""),
        ("#", "en", ""),
    ];
    for (raw, lang, expected) in cases {
        assert_eq!(parsing::canonicalize_trade_column(raw, lang), expected, "{raw:?}/{lang}");
    }

    for (v, lang) in [("Invoice No.", "en"), ("Consignee VAT/EORI", "en"), ("Description", "en"), ("품명", "ko")] {
        assert!(parsing::is_printed_label_echo(v, lang), "{v:?} is a printed label");
    }
    for (v, lang) in [("품명", "en"), ("ACME Corp", "en"), ("", "en")] {
        assert!(!parsing::is_printed_label_echo(v, lang), "{v:?} is a value");
    }
}

#[test]
#[ignore = "BUG(B22): partial alias matching maps unrelated headers ('Exchange Rate' → unit_price via 'rate', 'Cooling Fan' → country_of_manufacture via 'coo')"]
fn canonicalize_trade_column_avoids_substring_false_positives() {
    assert_ne!(parsing::canonicalize_trade_column("Exchange Rate", ""), "unit_price");
    assert_ne!(parsing::canonicalize_trade_column("Cooling Fan", ""), "country_of_manufacture");
}

#[test]
fn build_table_row_contract_maps_printed_headers() {
    let headers = vec![strings(&["Description", "Qty", "Unit Price"])];
    assert_eq!(
        parsing::build_table_row_contract(&headers, "en"),
        "COLUMN MAP (printed header -> output field):\n   column 0 \"Description\" -> \"description\"\n   column 1 \"Qty\" -> \"quantity\"\n   column 2 \"Unit Price\" -> \"unit_price\"\nRULES:\n- Return a JSON ARRAY. One object per printed data row. Never collapse rows.\n- If the image shows 2 data rows, the array MUST have 2 elements.\n- Do not output header rows, subtotal rows, or total rows as elements.\n- Use null for a column that is not printed on that row.\nSCHEMA:\n[ { \"description\": null, \"quantity\": null, \"unit_price\": null } ]"
    );
    let empty: Vec<Vec<String>> = Vec::new();
    assert_eq!(parsing::build_table_row_contract(&empty, "en"), "");
}

#[test]
fn table_header_grid_labels_body_cells() {
    // 2단 헤더: rowspan/colspan 을 펼친 직사각 격자
    let html = r#"<table><thead><tr><th rowspan="2">Item</th><th colspan="2">Weight</th></tr><tr><th>Net</th><th>Gross</th></tr></thead><tbody><tr id="r1"><td>A</td><td>1</td><td>2</td></tr></tbody></table>"#;
    let doc = Html::parse_document(html);
    let (grid, pending) = parsing::extract_doc_table_headers_sync(&doc, "#r1", "en");
    assert_eq!(grid, vec![strings(&["Item", "Weight", "Weight"]), strings(&["Item", "Net", "Gross"])]);
    assert_eq!(
        pending,
        vec![
            (0, "Item".to_string()),
            (1, "Weight".to_string()),
            (2, "Net".to_string()),
            (3, "Gross".to_string())
        ]
    );
    let rows = parsing::split_doc_to_pug_list_advanced(&doc, "tbody tr", PugMode::DetailMode, Some(grid), None);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0],
        "tr\n    td[alt=\"Item\"] | A\n    td[alt=\"Weight Net\"] | 1\n    td[alt=\"Weight Gross\"] | 2\n"
    );

    // abbr 이 있으면 표기 대신 정식 명칭, 확정된 컬럼은 field= 로 함께 실림. 천 단위 쉼표 제거.
    let html = r#"<table><thead><tr><th>Description</th><th abbr="unit price">U/P</th></tr></thead><tbody><tr id="r"><td>Widget</td><td>1,200.50</td></tr></tbody></table>"#;
    let doc = Html::parse_document(html);
    let (grid, pending) = parsing::extract_doc_table_headers_sync(&doc, "#r", "en");
    assert_eq!(grid, vec![strings(&["Description", "unit price"])]);
    assert!(pending.is_empty());
    let rows = parsing::split_doc_to_pug_list_advanced(&doc, "tbody tr", PugMode::DetailMode, Some(grid), None);
    assert_eq!(
        rows,
        vec![
            "tr\n    td[alt=\"Description\" field=\"description\"] | Widget\n    td[alt=\"unit price\" field=\"unit_price\"] | 1200.50\n"
                .to_string()
        ]
    );

    // 선택자가 표 밖이면 빈 격자
    let (grid, pending) = parsing::extract_doc_table_headers_sync(&doc, "#missing", "en");
    assert!(grid.is_empty() && pending.is_empty());
}

#[test]
fn split_html_to_pug_list_merges_rowspan_rows() {
    let html = r#"<table><tr><td rowspan="2">A</td><td>1</td></tr><tr><td>2</td></tr><tr><td>B</td><td>3</td></tr></table>"#;
    let rows = parsing::split_html_to_pug_list(html, "tr", PugMode::FullContent);
    assert_eq!(
        rows,
        vec![
            "tr\n    td[rowspan=\"2\"] | A\n    td | 1\ntr\n    td | 2\n".to_string(),
            "tr\n    td | B\n    td | 3\n".to_string(),
        ]
    );
    assert!(parsing::split_html_to_pug_list(html, "[[bad selector", PugMode::FullContent).is_empty());
}

#[test]
fn pre_clean_html_and_clean_pug() {
    assert_eq!(
        parsing::pre_clean_html(
            "<!-- c --><div class=\"a\" style=\"x\" onclick=\"y\">Hi<script>alert(1)</script><br></div>"
        ),
        "<div class=\"a\">Hi</div>"
    );
    assert_eq!(parsing::pre_clean_html("<input type=\"text\" value=\"5\" />"), "<input type=\"text\" value=\"5\" />");
    assert_eq!(parsing::pre_clean_html("<img src=\"a.png\" alt=\"A\" width=\"10\">"), "<img src=\"a.png\" alt=\"A\">");

    let pug = parsing::convert_to_clean_pug(
        "<html><body><table><tr><th>Name</th><th>Price</th></tr><tr><td>Apple</td><td>1,200</td></tr></table></body></html>",
        PugMode::FullContent,
        None,
    );
    for needle in ["th | Name", "th | Price", "td | Apple", "td | 1200"] {
        assert!(pug.contains(needle), "missing {needle:?} in:\n{pug}");
    }
    assert!(!pug.contains("1,200"), "thousand separators are stripped:\n{pug}");
}

#[test]
#[ignore = "BUG(B12): pre_clean_html's attribute regex also matches words inside attribute values (aria-label=\"not selected\" → 'selected')"]
fn pre_clean_html_does_not_promote_words_inside_values() {
    assert_eq!(parsing::pre_clean_html("<span aria-label=\"not selected\">M</span>"), "<span>M</span>");
}

#[test]
#[ignore = "BUG(B23): convert_doc_to_clean_pug looks for <body> among the root's children, never finds it, and emits the whole <html> tree"]
fn convert_to_clean_pug_starts_at_body() {
    let pug = parsing::convert_to_clean_pug("<html><body><p>Hello</p></body></html>", PugMode::StructureOnly, None);
    assert_eq!(pug.lines().next(), Some("body"));
}

// ═════════════════════════════ parsing : 무역 릴레이 ═════════════════════════════

#[test]
fn trade_relay_keys_and_document_identity() {
    let data = json!({
        "doc_number": "CI-43726",
        "reference_po": "PO-99281A",
        "reference_bl": "BL55432219",
        "container_number": "MSKU1234567"
    });
    let keys = parsing::extract_trade_relay_keys_for(&data, "en", "CI");
    let roles: Vec<&str> = keys.iter().map(|k| k.role).collect();
    assert_eq!(roles, vec!["reference_invoice", "order", "transport", "container"]);

    // 자기 문서번호는 '남이 나를 부르는 축'(reference_invoice)으로 등록
    let k = &keys[0];
    assert_eq!(k.source_field, "doc_number");
    assert_eq!(k.search_field, "reference_invoice");
    assert_eq!(k.raw, "CI-43726");
    assert_eq!(k.normalized, "CI43726");
    assert_eq!(k.index, 1963561162);
    assert_eq!(k.id, "0x379a5d1b29644ed9beb9a512dcd39dd03db58800");

    assert_eq!((keys[1].source_field.as_str(), keys[1].search_field.as_str()), ("reference_po", "doc_number"));
    assert_eq!((keys[1].normalized.as_str(), keys[1].index), ("PO99281A", 3135199003));
    assert_eq!(keys[1].id, hash_id("orderPO99281A"));
    assert_eq!((keys[2].normalized.as_str(), keys[2].index), ("BL55432219", 3958018455));
    assert_eq!(keys[2].search_field, "doc_number");
    assert_eq!((keys[3].search_field.as_str(), keys[3].index), ("container_number", 2820017878));

    // 역방향 축은 서식별 (PO → reference_po)
    let po = parsing::extract_trade_relay_keys_for(&json!({"doc_number": "PO-99281A"}), "en", "PO");
    assert_eq!(po.len(), 1);
    assert_eq!((po[0].role, po[0].search_field.as_str()), ("reference_po", "reference_po"));

    // 인쇄 라벨 에코 / 플레이스홀더는 릴레이 키가 아님
    assert!(parsing::extract_trade_relay_keys_for(
        &json!({"doc_number": "Invoice No.", "reference_po": "N/A"}),
        "en",
        "CI"
    )
    .is_empty());

    // 문서 식별자: 문서번호 우선, 없으면 내용 지문
    assert_eq!(
        parsing::resolve_trade_doc_identity("CI", &json!({"doc_number": "CI-43726"}), "en"),
        ("CI-43726".to_string(), 1963561162, false)
    );
    assert_eq!(
        parsing::resolve_trade_doc_identity(
            "CI",
            &json!({"issue_date": "2026-03-15", "amount": 1200, "currency": "USD"}),
            "en"
        ),
        ("CI|issue_date=2026-03-15|amount=1200|currency=USD".to_string(), 2456121095, true)
    );
}

#[test]
#[ignore = "BUG(B7): resolve_trade_doc_identity ignores 'document_number' (accepted by extract_trade_relay_keys_for) and falls back to a content fingerprint"]
fn resolve_trade_doc_identity_accepts_document_number_alias() {
    let (key, index, fingerprint) =
        parsing::resolve_trade_doc_identity("CI", &json!({"document_number": "INV-2024-001"}), "en");
    assert!(!fingerprint, "got fingerprint {key:?}");
    assert_eq!(index, relay_index("INV-2024-001"));
}

#[test]
fn trade_hub_key_and_envelope() {
    // ① 허브 참조 ② 자기 자신이 허브 ③ 아무 참조 ④ fallback id
    assert_eq!(parsing::resolve_trade_hub_key("CI", &json!({"reference_po": "PO-1001", "doc_number": "CI-1"}), "fb"), "PO1001");
    assert_eq!(parsing::resolve_trade_hub_key("PO", &json!({"doc_number": "PO-1001"}), "fb"), "PO1001");
    assert_eq!(parsing::resolve_trade_hub_key("PL", &json!({"booking_number": "BK-5"}), "fb"), "BK5");
    assert_eq!(parsing::resolve_trade_hub_key("PL", &json!({"logistics": {"reference_bl": "BL-77001"}}), "fb"), "BL77001");
    assert_eq!(parsing::resolve_trade_hub_key("PL", &json!({}), "id-123"), "ID123");

    let data = json!({"reference_po": "PO-1001"});
    let (cc, bcc, r) = parsing::trading_envelope("team-1", "CI", &data, "fb");
    assert_eq!(cc, parsing::trading_cc());
    assert_eq!(bcc, hash_id(&format!("CI{cc}")));
    assert_eq!(r, parsing::trading_ref("team-1", &cc, "PO1001"));
    assert_eq!(r, hash_id(&format!("team-1{cc}#PO1001")));
}

#[test]
#[ignore = "BUG(B8): resolve_trade_hub_key accepts placeholder hub references ('N/A' → hub 'NA'), merging unrelated documents"]
fn resolve_trade_hub_key_skips_placeholder_references() {
    assert_eq!(
        parsing::resolve_trade_hub_key("CI", &json!({"reference_po": "N/A", "doc_number": "CI-9"}), "x"),
        "CI9"
    );
}

// ═════════════════════════════ nl_convert ═════════════════════════════

#[test]
fn json_to_natural_language_and_chunk_split() {
    let cases = [
        (
            json!({"title": "Blue Shirt", "type": "goods", "sale_price": 15000, "currency": "KRW", "color": "blue"}),
            "This goods is titled 'Blue Shirt'. Its color is blue. The sale price is 15000 KRW.",
        ),
        (json!({"shipping": {"carrier": "CJ"}}), "Regarding shipping. Its carrier is CJ."),
        (json!({"status": 9}), "It is currently in 'complete' status."),
        (json!({"status": 0}), ""),
        (json!({"tags": ["a", "b"]}), "The tags includes: a, b."),
        (
            json!({"id": "X1", "link": "https://a.com/1", "title": "T"}),
            "The unique identifier is X1. It can be accessed at https://a.com/1. This item is titled 'T'.",
        ),
        // 릴레이 index 는 숨기고 동반 제목을 기준 이름으로
        (json!({"order": 5, "order_title": "ORD-1"}), "Its order is ORD-1."),
        // 통화 기호는 통화 코드로 한 번만
        (json!({"sale_price": "15,000원", "currency": "KRW"}), "The sale price is 15,000 KRW."),
        // *_at epoch ms 는 ISO 로
        (json!({"etd_at": 1773532800000i64}), "Its etd at is 2026-03-15T00:00:00."),
    ];
    for (v, expected) in cases.iter() {
        assert_eq!(nl::json_to_natural_language(v), *expected, "{v}");
    }

    let chunk = |t: &str, p: &str, c: bool| (t.to_string(), p.to_string(), c);
    assert_eq!(
        nl::split_natural_language_to_chunks(cases[0].1),
        vec![
            chunk("This goods is titled 'Blue Shirt'", "title", true),
            chunk("Its color is blue", "color", true),
            chunk("The sale price is 15000 KRW", "sale_price", true),
        ]
    );
    assert_eq!(
        nl::split_natural_language_to_chunks(cases[1].1),
        vec![chunk("Regarding shipping", "context_intro", false), chunk("Its carrier is CJ", "carrier", true)]
    );
    assert_eq!(
        nl::split_natural_language_to_chunks(cases[4].1),
        vec![chunk("tags includes a", "tags", true), chunk("tags includes b", "tags", true)]
    );
    assert_eq!(
        nl::split_natural_language_to_chunks(cases[5].1),
        vec![
            chunk("The unique identifier is X1", "id", true),
            chunk("It can be accessed at https://a.com/1", "link", true),
            chunk("This item is titled 'T'", "title", true),
        ]
    );
    assert!(nl::split_natural_language_to_chunks("").is_empty());
}

#[test]
#[ignore = "BUG(B16): json_to_natural_language drops relay-key arrays of objects ({\"goods\":[{\"title\":..}]}) so item titles never reach the index"]
fn json_to_natural_language_keeps_relay_object_arrays() {
    assert!(nl::json_to_natural_language(&json!({"goods": [{"title": "Shirt"}]})).contains("Shirt"));
}

#[test]
fn chunk_value_clause_helpers_and_nms() {
    let cases = [
        ("Its weight is 1.5", "1.5"),
        ("The sale price is 29900 KRW", "29900 KRW"),
        ("It is currently in 'show' status", "show"),
        ("This goods is titled '테스트상품'", "테스트상품"),
        ("tags includes 가전", "가전"),
        ("random", "random"),
    ];
    for (chunk, expected) in cases {
        assert_eq!(nl::extract_value_from_chunk(chunk), expected, "{chunk:?}");
    }
    assert_eq!(nl::split_clause_commas("1,000, 2"), vec!["1,000", " 2"]);
    assert!(nl::is_clause_fragment("and red"));
    assert!(!nl::is_clause_fragment("the big red ball"));
    assert!(nl::is_clause_fragment("a b c"));

    let meta = |text: &str, prop: &str, value: &str, confirmed: bool| nl::ChunkMetadata {
        chunk_text: text.to_string(),
        property: prop.to_string(),
        property_format: "Text".to_string(),
        bias_phrases: Vec::new(),
        prejudice_phrases: Vec::new(),
        value_part: value.to_string(),
        confirmed,
    };
    // 완전 동일 텍스트
    let out = nl::nms_battle_for_indexing(vec![meta("Its color is red", "color", "red", true), meta("Its color is red", "color", "red", true)]);
    assert_eq!(out.len(), 1);
    // 서로 다른 속성은 공존
    let out = nl::nms_battle_for_indexing(vec![meta("Its color is red", "color", "red", true), meta("Its size is M", "size", "M", true)]);
    assert_eq!(out.len(), 2);
    // confirmed 청크는 비confirmed 청크에 밀리지 않음
    let out = nl::nms_battle_for_indexing(vec![
        meta("Its color is red", "color", "red", true),
        meta("red cotton", "color", "red cotton", false),
    ]);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].chunk_text, "Its color is red");
}

#[test]
fn transliteration_helpers() {
    assert_eq!(nl::try_any_ascii_transliteration("니트").as_deref(), Some("NiTeu"));
    assert_eq!(nl::try_any_ascii_transliteration("Knit"), None);
    assert_eq!(nl::try_any_ascii_transliteration("  "), None);
    assert!(nl::is_latin_dominant("Knit"));
    assert!(!nl::is_latin_dominant("니트"));
    assert_eq!(nl::strip_special_chars_for_transliteration("Knit-Cardigan (Blue)!"), "Knit Cardigan Blue");

    // 원문에서 붙어 있던(공백 없이 구분자로 이어진) 단어는 음차에서도 붙임
    assert_eq!(nl::reglue_native_alias("Knit-Cardigan", "니트 가디건"), "니트가디건");
    assert_eq!(nl::reglue_native_alias("Knit Cardigan", "니트 가디건"), "니트 가디건");

    assert!(nl::is_digit_word("1,500"));
    assert!(!nl::is_digit_word("15A"));
    assert!(!nl::is_digit_word(","));
    assert_eq!(nl::place_digit_words("iPhone 15 Pro", "아이폰 프로").as_deref(), Some("아이폰 15 프로"));
    assert_eq!(nl::place_digit_words("Knit", "니트"), None);
    assert_eq!(
        nl::split_words_by_script("니트 Cardigan 2024"),
        (strings(&["니트", "2024"]), strings(&["Cardigan"]))
    );

    assert_eq!(nl::assign_transliterations("Knit", "니트", "knit"), ("니트".to_string(), String::new()));
    assert_eq!(nl::assign_transliterations("니트", "NiTeu", "니트"), (String::new(), "NiTeu".to_string()));
    assert_eq!(nl::find_mixed_script_words("시IELD ok 포켓몬 시IELD"), strings(&["시IELD"]));
    assert_eq!(
        nl::replace_mixed_words("시IELD ok", &[("시IELD".to_string(), "실드".to_string())]),
        "실드 ok"
    );

    assert_eq!(nl::lang_code_to_full_name("zh-tw"), "chinese");
    assert_eq!(nl::lang_code_to_full_name("ms"), "indonesian");
    assert_eq!(nl::lang_code_to_full_name("xx"), "xx");
}

#[test]
fn phonetic_similarity_and_gate() {
    let close = |a: f32, b: f32| (a - b).abs() < 1e-5;
    assert!(close(nl::phonetic_similarity("Knit", "니트").unwrap(), 0.8));
    assert!(close(nl::phonetic_similarity("Cardigan", "가디건").unwrap(), 1.0));
    assert!(close(nl::phonetic_similarity("Samsung", "사과").unwrap(), 1.0 / 3.0));
    assert_eq!(nl::phonetic_similarity("a", "b"), None);

    // 약어는 알파벳 이름 읽기(피씨)로도 비교
    assert_eq!(nl::phonetic_gate("PC", "피씨"), (true, Some(1.0)));
    assert_eq!(nl::phonetic_gate("", "x"), (true, None));
    let (pass, score) = nl::phonetic_gate("Samsung", "사과");
    assert!(!pass);
    assert!(score.unwrap() < nl::PHONETIC_PASS);
}

// ═════════════════════════════ ai_utils ═════════════════════════════

#[test]
fn detect_field_format_families() {
    use tauri_app_lib::utils::ai_utils::FieldFormat::*;
    let cases = [
        ("sale_price", Numeric),
        ("quantity", Numeric),
        ("weight_gross", Numeric),
        ("package_count", Numeric),
        ("volume", Numeric),
        ("local_charges", Numeric),
        ("unit_price", Numeric),
        ("tracking_number", TrackingCode),
        ("barcode", TrackingCode),
        ("id", Identifier),
        ("id,link", Identifier),
        ("hs_code", Identifier),
        ("main_image_url", Link),
        ("registration_date", Date),
        ("started_at", Date),
        ("etd", Date),
        ("sender_phone", Phone),
        ("number", Phone),
        ("recipient_address", Address),
        ("status", Enum),
        ("incoterms", Enum),
        ("payment_terms", Enum),
        ("title", Text),
        ("doc_number", Text),
        ("traffic_insight", Synthesis),
    ];
    for (field, expected) in cases {
        assert_eq!(ai::detect_field_format(field), expected, "{field}");
    }
    // 검색 질의 쪽은 저장 타입이 식별자면 식별자로 승격
    assert_eq!(ai::query_value_format("doc_number"), Identifier);
    assert_eq!(ai::query_value_format("title"), Text);
    assert_eq!(ai::query_value_format("status"), Enum);
}

#[test]
#[ignore = "BUG(B20): detect_field_format('exchange_rate') is Text although canonical::kind_of stores it as Numeric, so numeric rates fail the format gate"]
fn detect_field_format_treats_rates_as_numeric() {
    assert_eq!(ai::detect_field_format("exchange_rate"), FieldFormat::Numeric);
}

#[test]
fn date_shape_detectors_and_month_names() {
    assert!(ai::has_date_literal("2026-03-15"));
    assert!(ai::has_date_literal("2026/3/5"));
    assert!(!ai::has_date_literal("010-3333-3333"));
    assert!(!ai::has_date_literal("1.5.3"));

    for s in ["Apr-19-2022", "2026年3月15日", "March 2026", "15.03.2026"] {
        assert!(ai::has_date_shape(s), "{s:?} is a date");
    }
    for s in ["2000.00", "CI-43726", "20KG", "15 Mar"] {
        assert!(!ai::has_date_shape(s), "{s:?} is not a date");
    }
    assert_eq!(
        ai::extract_date_literal("Shipped 2026-03-15 10:20:30 by DHL").as_deref(),
        Some("2026-03-15 10:20:30")
    );
    assert_eq!(ai::extract_date_literal("no date here"), None);

    let months = [
        ("Apr", Some(4)),
        ("March", Some(3)),
        ("mars", Some(3)),
        ("Mayo", Some(5)),
        ("Sept.", Some(9)),
        ("十二月", Some(12)),
        ("十一月", Some(11)),
        ("décembre", Some(12)),
        ("jan26", Some(1)),
        ("12月", None),
        ("Jan2026", None),
    ];
    for (name, expected) in months {
        assert_eq!(ai::month_from_name(name), expected, "{name:?}");
    }
    assert_eq!(ai::normalize_digits_ascii("٢٠٢٦"), "2026");
    assert_eq!(ai::normalize_digits_ascii("۱۲"), "12");
    assert_eq!(ai::normalize_digits_ascii("１２３"), "123");
}

#[test]
#[ignore = "BUG(B4): extract_date_literal only knows year-first shapes and cuts '15.03.2026' to '15.03.20'"]
fn extract_date_literal_keeps_day_first_dates_whole() {
    assert_eq!(ai::extract_date_literal("15.03.2026").as_deref(), Some("15.03.2026"));
}

#[test]
#[ignore = "BUG(B5): has_date_literal rejects day-first dates like '15.03.2026' (g3 must be <= 2 digits)"]
fn has_date_literal_accepts_day_first_dates() {
    assert!(ai::has_date_literal("15.03.2026"));
}

#[test]
fn currency_and_country_codes() {
    let currencies = [
        ("USD", Some("USD")),
        ("100 USD", Some("USD")),
        ("US$ 1,200", Some("USD")),
        ("1,200원", Some("KRW")),
        ("€50", Some("EUR")),
        ("krw", Some("KRW")),
        ("dollars", Some("USD")),
        ("EUR/USD", Some("EUR")),
        ("", None),
    ];
    for (raw, expected) in currencies {
        assert_eq!(ai::currency_code_of(raw), expected, "{raw:?}");
    }
    assert_eq!(ai::normalize_currency_value("", "ko"), "KRW");
    assert_eq!(ai::normalize_currency_value("null", "en"), "USD");
    assert_eq!(ai::normalize_currency_value("$", "ko"), "USD");
    assert_eq!(ai::normalize_currency_value("1,000", "ja"), "JPY");
    assert_eq!(ai::normalize_currency_value("50 CHF", "en"), "CHF");
    assert_eq!(ai::normalize_currency_value("rupees", "en"), "RUPEES");
    assert_eq!(ai::default_currency_for_lang("de"), "EUR");
    assert_eq!(ai::default_currency_for_lang("xx"), "USD");

    let countries = [
        ("Korea", Some("KR")),
        ("Made in Korea", Some("KR")),
        ("Republic of Korea", Some("KR")),
        ("일본산", Some("JP")),
        ("Indonesia", Some("ID")),
        ("인도네시아", Some("ID")),
        ("인도산", Some("IN")),
        ("Made in P.R.C.", Some("CN")),
        ("Indiana", None),
    ];
    for (raw, expected) in countries {
        assert_eq!(ai::country_code_of(raw), expected, "{raw:?}");
    }
    assert_eq!(ai::country_code_exact("Made in Korea"), None);
    assert_eq!(ai::country_code_exact("korea"), Some("KR"));
}

#[test]
#[ignore = "BUG(B27): currency_code_of does not recognise the yen/yuan sign '¥'"]
fn currency_code_of_recognises_yen_sign() {
    assert!(matches!(ai::currency_code_of("¥1000"), Some("JPY") | Some("CNY")));
}

#[test]
#[ignore = "BUG(B28): country_code_of('US') is None because only 'usa' / 'u.s.' spellings are listed"]
fn country_code_of_recognises_us_abbreviation() {
    assert_eq!(ai::country_code_of("US"), Some("US"));
}

#[test]
fn count_unit_and_comparator_split() {
    let pair = |a: &str, b: &str| Some((a.to_string(), b.to_string()));
    assert_eq!(ai::split_count_and_unit("10 CTNS"), pair("10", "CTNS"));
    assert_eq!(ai::split_count_and_unit("1,200 pcs"), pair("1200", "pcs"));
    assert_eq!(ai::split_count_and_unit("CTNS 10"), pair("10", "CTNS"));
    assert_eq!(ai::split_count_and_unit("(10) cartons"), pair("10", "cartons"));
    assert_eq!(ai::split_count_and_unit("12.5 KGS"), pair("12.5", "KGS"));
    for v in ["10", "10 x 20 boxes", "10 a b c d e", ""] {
        assert_eq!(ai::split_count_and_unit(v), None, "{v:?}");
    }

    assert_eq!(ai::split_numeric_and_comparator("5000원 이하로"), pair("5000", "이하로"));
    assert_eq!(ai::split_numeric_and_comparator("under $50"), pair("50", "under $"));
    assert_eq!(ai::split_numeric_and_comparator("1,000 won or less"), pair("1000", "won or less"));
    assert_eq!(ai::split_numeric_and_comparator("3.5kg 이상"), pair("3.5", "이상"));
    assert_eq!(ai::split_numeric_and_comparator("abc"), None);
}

#[test]
fn exclusive_assignment_matrices() {
    let close = |a: f32, b: f32| (a - b).abs() < 1e-5;

    let r = ai::greedy_exclusive_assign(&vec![vec![0.9, 0.1], vec![0.8, 0.7]]);
    let (l0, s0, m0) = r[0].expect("field 0");
    let (l1, s1, m1) = r[1].expect("field 1");
    assert_eq!((l0, l1), (0, 1));
    assert!(close(s0, 0.9) && close(m0, 0.1), "{s0} {m0}");
    assert!(close(s1, 0.7) && close(m1, 0.6), "{s1} {m1}");
    assert!(ai::greedy_exclusive_assign(&vec![]).is_empty());

    // 같은 행렬이라도 정렬 기준(점수 vs 마진)에 따라 배정이 다름
    let m = vec![vec![0.9, 0.5], vec![0.85, 0.1]];
    let by_score = ai::exclusive_assign_by_score(&m, 0.0, 0.0);
    let (l, s, mg) = by_score[0].expect("by score");
    assert_eq!(l, 0);
    assert!(close(s, 0.9) && close(mg, 0.05), "{s} {mg}");
    assert!(by_score[1].is_none());

    let by_margin = ai::exclusive_assign(&m, 0.0, 0.0);
    let (l, s, mg) = by_margin[0].expect("by margin");
    assert_eq!(l, 1);
    assert!(close(s, 0.5) && close(mg, 0.4), "{s} {mg}");
    assert!(by_margin[1].is_none());

    assert_eq!(
        ai::double_center_matrix(&vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
        vec![vec![0.5, -0.5], vec![-0.5, 0.5]]
    );
}

#[test]
fn url_helpers() {
    let parts = |h: &str, p: &str, q: &str| (h.to_string(), p.to_string(), q.to_string());
    assert_eq!(ai::split_href_parts("https://shop.com/goods/123?id=5#top"), parts("shop.com", "/goods/123", "id=5"));
    assert_eq!(ai::split_href_parts("//cdn.com/a.png"), parts("cdn.com", "/a.png", ""));
    assert_eq!(ai::split_href_parts("/goods/view?no=77"), parts("", "/goods/view", "no=77"));

    let (prefix, suffix) =
        ai::extract_url_pattern("12345", "https://shop.com/goods/view?no=12345&x=1").expect("pattern");
    assert_eq!((prefix.as_str(), suffix.as_str()), ("https://shop.com/goods/view?no=", "&x=1"));
    assert_eq!(ai::apply_url_pattern(&prefix, &suffix, "999"), "https://shop.com/goods/view?no=999&x=1");
    // 가장 오른쪽 출현을 식별자로
    assert_eq!(
        ai::extract_url_pattern("12", "https://a.com/12/item/12"),
        Some(("https://a.com/12/item/".to_string(), String::new()))
    );
    // 호스트 안의 문자열은 식별자가 아님
    assert_eq!(ai::extract_url_pattern("shop", "https://shop.com/"), None);

    assert_eq!(ai::humanize_url_token("mainImageURL"), "main image url");
    assert_eq!(ai::humanize_url_token("item2price"), "item 2 price");
    assert_eq!(ai::humanize_url_token("sale_price"), "sale price");
    assert_eq!(ai::humanize_url_token("--"), "");
    assert_eq!(ai::href_param_key("/view?goodsNo=123&amp;page=2", "123").as_deref(), Some("goodsno"));
    assert_eq!(ai::href_param_key("/view?page=2", "123"), None);
    assert_eq!(ai::href_resource_stem("https://shop.com/goods/detail.html?no=1"), "detail");
}

#[test]
fn value_format_gates() {
    use tauri_app_lib::utils::ai_utils::FieldFormat::*;
    let cases = [
        (Numeric, "1,200", true),
        (Numeric, "010-1234-5678", false), // 묶음 숫자(전화)
        (Numeric, "192.168.0.1", false),   // IP
        (Numeric, "2026-03-15", false),    // 날짜
        (Phone, "010-1234-5678", true),
        (Phone, "test3@gmail.com", false),
        (Date, "15.03.2026", true),
        (Text, "tr", false), // PUG 태그 잔재
        (Text, "A", false),
        (Identifier, "CI-43726", true),
        (TrackingCode, "1234567", false),
        (Link, "www.example.com", true),
        (Address, "Seoul", false),
        (Enum, "2026-03-15", false),
        (Synthesis, "anything", true),
    ];
    for (fmt, value, expected) in cases {
        assert_eq!(ai::value_matches_format(fmt, value), expected, "{fmt:?} {value:?}");
    }

    assert!(ai::is_ip_address_literal("192.168.0.1"));
    for v in ["256.1.1.1", "01.2.3.4", "1.2.3"] {
        assert!(!ai::is_ip_address_literal(v), "{v}");
    }
    assert!(ai::is_document_number_shaped("CI-43726"));
    assert!(ai::is_document_number_shaped("INV-2026-0315"));
    assert!(!ai::is_document_number_shaped("2026-03-15"));
    assert!(!ai::is_document_number_shaped("Apr-19-2022"));
}

#[test]
fn pug_line_parsing_helpers() {
    assert_eq!(
        ai::pug_line_parts("    td[alt=\"Qty\" field=\"quantity\"] | 10"),
        (4, "td".to_string(), "[alt=\"Qty\" field=\"quantity\"]".to_string(), "10".to_string())
    );
    // 따옴표 안의 파이프는 구분자가 아님
    assert_eq!(ai::pug_line_parts("a[href=\"x|y\"] | text").3, "text");
    assert_eq!(ai::pug_line_parts("tr"), (0, "tr".to_string(), String::new(), String::new()));

    assert_eq!(ai::pug_attr_string("[alt=\"Qty\" field=\"quantity\"]", "field").as_deref(), Some("quantity"));
    assert_eq!(ai::pug_attr_string("[alt=\"Qty\"]", "field"), None);
    assert!(ai::pug_attr_flag("[checked disabled]", "checked"));
    assert!(!ai::pug_attr_flag("[data-checked=\"1\"]", "checked"));
}

#[test]
#[ignore = "BUG(B11): pug_attr_string matches 'id=\"' inside 'data-id=\"' and returns the wrong attribute"]
fn pug_attr_string_matches_whole_attribute_names() {
    assert_eq!(ai::pug_attr_string("[data-id=\"5\" id=\"7\"]", "id").as_deref(), Some("7"));
}

#[test]
fn exact_absolute_period_and_operator_words() {
    let today = d(2026, 10, 9);
    let p = |words: &[&str]| ai::exact_absolute_period(&strings(words), today);

    let x = p(&["2026-03-15"]).expect("day");
    assert_eq!(
        (x.start, x.end, x.operator, x.granularity, x.year_explicit, x.range),
        (d(2026, 3, 15), d(2026, 3, 15), "between", "day", true, false)
    );
    assert_eq!(x.tokens, vec![0]);

    let x = p(&["2026-03"]).expect("year-month");
    assert_eq!((x.start, x.end, x.granularity), (d(2026, 3, 1), d(2026, 3, 31), "month"));

    let x = p(&["March", "2026"]).expect("month name + year");
    assert_eq!((x.start, x.end, x.granularity, x.year_explicit), (d(2026, 3, 1), d(2026, 3, 31), "month", true));
    assert_eq!(x.tokens, vec![0, 1]);

    let x = p(&["2026년", "3월"]).expect("korean units");
    assert_eq!((x.start, x.end, x.granularity), (d(2026, 3, 1), d(2026, 3, 31), "month"));

    // 연도 미기재는 today 의 연도, year_explicit=false
    let x = p(&["3월"]).expect("month only");
    assert_eq!((x.start, x.end, x.year_explicit), (d(2026, 3, 1), d(2026, 3, 31), false));

    // 앞쪽 연산어 / 꼬리 조사
    let x = p(&["after", "March", "2026"]).expect("after");
    assert_eq!((x.operator, x.range), ("gte", false));
    assert_eq!(x.tokens, vec![0, 1, 2]);
    let x = p(&["2026년", "3월부터"]).expect("부터");
    assert_eq!((x.operator, x.range), ("gte", false));
    let x = p(&["2026년", "3월부터", "5월까지"]).expect("부터~까지");
    assert_eq!(
        (x.start, x.end, x.operator, x.granularity, x.range),
        (d(2026, 3, 1), d(2026, 5, 31), "between", "month", true)
    );

    assert!(p(&["hello"]).is_none());

    assert_eq!(ai::comparator_exact(&strings(&["at", "least"])), Some("gte"));
    assert_eq!(ai::comparator_exact(&strings(&["이하"])), Some("lte"));
    assert_eq!(ai::comparator_exact(&strings(&["more", "than"])), Some("gt"));
    assert_eq!(ai::comparator_exact(&strings(&["less", "than"])), Some("lt"));
    assert_eq!(ai::comparator_exact(&strings(&["blue"])), None);
    assert_eq!(ai::time_unit_exact("년"), Some("year"));
    assert_eq!(ai::time_unit_exact("months"), Some("month"));
    assert_eq!(ai::time_unit_exact("dias"), Some("day"));
    assert_eq!(ai::time_unit_exact("yesterday"), None);
}

#[test]
fn filter_dictionaries_and_phrase_statistics() {
    assert_eq!(ai::exact_match_filter_key("time_filters", "오늘").as_deref(), Some("today"));
    assert_eq!(ai::exact_match_filter_key("time_filters", "This Month").as_deref(), Some("this_month"));
    assert_eq!(ai::exact_match_filter_key("season_filters", "여름").as_deref(), Some("summer"));
    assert_eq!(ai::exact_match_filter_key("analytic_event_filters", "클릭").as_deref(), Some("click"));
    assert_eq!(ai::exact_match_filter_key("no_such_category", "오늘"), None);
    // 교착어: 가장 긴 접두 어간
    assert_eq!(
        ai::prefix_match_filter_stem("time_filters", "올해는"),
        Some(("this_year".to_string(), "올해".to_string()))
    );

    assert_eq!(
        ai::split_bias_phrases("order no, B/L no, price/cost, order no"),
        strings(&["order no", "B/L no", "price", "cost"])
    );
    let fifty = (0..50).map(|i| format!("p{i}")).collect::<Vec<_>>().join(", ");
    assert_eq!(ai::split_bias_phrases(&fifty).len(), 48);
    assert_eq!(ai::split_bias_phrases_full(&fifty).len(), 50);
    let (phrases, weights) = ai::split_bias_phrases_weighted("2026-03-15, invoice no 1, name");
    assert_eq!(phrases, strings(&["2026-03-15", "invoice no 1", "name"]));
    assert_eq!(weights, vec![0.80, 0.95, 1.0]);

    assert_eq!(ai::cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
    assert_eq!(ai::cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    assert_eq!(ai::cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    assert_eq!(ai::cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);

    let close = |a: f32, b: f32| (a - b).abs() < 1e-3;
    assert_eq!(ai::gumbel_expected_z(1), 0.0);
    assert!(close(ai::gumbel_expected_z(2), 1.17741));
    assert!(close(ai::gumbel_expected_z(100), 3.03485));
    assert!(close(ai::gumbel_max_sd(2), 1.08930));
    assert!(close(ai::axis_snr(&[1.0, 0.5, 0.3, 0.1]).unwrap(), 4.28661));
    assert_eq!(ai::axis_snr(&[1.0, 0.5]), None);
    assert_eq!(ai::axis_snr(&[0.5, 0.5, 0.5]), None, "flat tail has no noise estimate");

    // 공유 구 / 자기모순 구는 마스킹, 전량 탈락 뱅크는 원본 유지
    let bias_banks = vec![strings(&["a", "s"]), strings(&["s", "b"])];
    let prej_banks = vec![Vec::new(), strings(&["b"])];
    assert_eq!(
        ai::cross_field_ambiguous_phrase_mask(&bias_banks, &prej_banks),
        vec![vec![true, false], vec![true, true]]
    );

    assert_eq!(ai::bank_lang_order("ja"), strings(&["en", "ko"]));
    assert_eq!(ai::bank_lang_order("ko"), strings(&["ko", "en"]));
    assert_eq!(ai::bank_lang_coverage().0, vec!["en", "ko"]);
}

#[test]
#[ignore = "BUG(B26): bias.json lists the same exact_match literal under two keys ('yaz' spring/summer, 'přejetí' hover/touch), so the lookup is ambiguous"]
fn exact_match_literals_are_unambiguous() {
    assert_eq!(ai::exact_match_filter_keys("season_filters", "yaz").len(), 1);
    assert_eq!(ai::exact_match_filter_keys("analytic_event_filters", "přejetí").len(), 1);
}

// ═════════════════════════════ pug_utils ═════════════════════════════

#[test]
fn pug_block_indices_and_header_grid() {
    let lines = vec!["html", "  body", "    div.a", "", "      p hello", "    div.b"];
    assert_eq!(find_block_indices_in_pug(lines.as_slice(), "div.a\n  p hello"), Some((2, 4)));
    assert_eq!(find_block_indices_in_pug(lines.as_slice(), "div.b"), Some((5, 5)));
    assert_eq!(find_block_indices_in_pug(lines.as_slice(), "div.c"), None);
    assert_eq!(find_block_indices_in_pug(lines.as_slice(), ""), None);

    let head = strings(&["tr", "  th[rowspan=\"2\"] | Item", "  th | Qty", "tr", "  th | Unit"]);
    let grid = HeaderGrid::new(parse_pug_grid(&head));
    assert_eq!(grid.rows, 2);
    assert_eq!(grid.column_count(), 2);
    assert_eq!(grid.column_label(0), "Item");
    assert_eq!(grid.column_label(1), "Qty > Unit");

    // 헤더와 같은 모양으로 교차 배치된 본문 행
    let body = strings(&["tr", "  td[rowspan=\"2\"] | Apple", "  td | 10", "tr", "  td | EA"]);
    let cells = parse_pug_grid(&body);
    assert_eq!(cells.len(), 3);
    assert_eq!((cells[2].row, cells[2].col, cells[2].text.as_str()), (1, 1, "EA"));
    assert!(grid.interleaved_with(&cells));
    assert_eq!(grid.cell_label(1, 1, 1, 1, true), "Unit");
    assert_eq!(grid.cell_label(0, 0, 1, 2, true), "Item");
    assert_eq!(grid.cell_label(1, 1, 1, 1, false), "Qty > Unit");
}

// ═════════════════════════════ openai_types / chat_template / tokenizer ═════════════════════════════

#[test]
fn openai_types_serde_shapes() -> anyhow::Result<()> {
    let m: ChatCompletionRequestMessage = serde_json::from_value(json!({"role": "user", "content": "hi"}))?;
    match &m {
        ChatCompletionRequestMessage::User(u) => match &u.content {
            ChatCompletionRequestUserMessageContent::Text(t) => assert_eq!(t, "hi"),
            other => panic!("expected text content, got {other:?}"),
        },
        other => panic!("expected user message, got {other:?}"),
    }
    // name 은 None 이면 직렬화에서 빠짐
    assert_eq!(serde_json::to_value(&m)?, json!({"role": "user", "content": "hi"}));

    let m: ChatCompletionRequestMessage = serde_json::from_value(json!({
        "role": "user",
        "content": [{"type": "text", "text": "a"}, {"type": "video_url", "video_url": {"url": "v.mp4"}}]
    }))?;
    match &m {
        ChatCompletionRequestMessage::User(u) => match &u.content {
            ChatCompletionRequestUserMessageContent::Array(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(&parts[0], ChatCompletionRequestMessageContentPart::Text(t) if t.text == "a"));
                assert!(matches!(&parts[1], ChatCompletionRequestMessageContentPart::VideoURL(v) if v.video_url.url == "v.mp4"));
            }
            other => panic!("expected array content, got {other:?}"),
        },
        other => panic!("expected user message, got {other:?}"),
    }

    let s: ChatCompletionRequestMessage = serde_json::from_value(json!({"role": "system", "content": "sys"}))?;
    assert!(matches!(s, ChatCompletionRequestMessage::System(ref x) if x.content == "sys"));

    let p: ChatCompletionParameters = serde_json::from_value(json!({
        "model": "qwen",
        "messages": [{"role": "assistant", "content": "ok"}],
        "temperature": 0.2
    }))?;
    assert_eq!(p.model, "qwen");
    assert_eq!(p.temperature, Some(0.2));
    assert!(p.tools.is_none() && p.max_tokens.is_none());
    assert!(matches!(&p.messages[0], ChatCompletionRequestMessage::Assistant(a) if a.content.as_deref() == Some("ok")));

    let text_part = ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText {
        text: "a".to_string(),
    });
    assert_eq!(serde_json::to_value(&text_part)?, json!({"type": "text", "text": "a"}));
    Ok(())
}

#[test]
#[ignore = "BUG(B14): rename_all = lowercase turns the ImageURL variant tag into 'imageurl', so OpenAI 'image_url' parts fail to deserialize"]
fn openai_image_url_part_deserializes() {
    let part = serde_json::from_value::<ChatCompletionRequestMessageContentPart>(json!({
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"}
    }));
    assert!(part.is_ok(), "{:?}", part.err());
}

const CHAT_TEMPLATE: &str =
    "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}";

#[test]
fn chat_template_renders_from_config_and_jinja_fallback() -> anyhow::Result<()> {
    let user_hi: ChatCompletionParameters =
        serde_json::from_value(json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}))?;

    // ① tokenizer_config.json 의 chat_template
    let dir = TempDir::new("it_chat_tpl_config");
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        serde_json::to_vec(&json!({"chat_template": CHAT_TEMPLATE}))?,
    )?;
    let tpl = ChatTemplate::init(dir.path().to_str().unwrap())?;
    assert_eq!(tpl.apply_chat_template(&user_hi)?, "user:hi\nassistant:");

    // ② 배열 content 는 문자열로 평탄화, 이미지 파트는 비전 플레이스홀더
    let parts = vec![
        ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { text: "a".to_string() }),
        ChatCompletionRequestMessageContentPart::ImageURL(ChatCompletionRequestMessageContentPartImage {
            image_url: ImageURL { url: "x.png".to_string(), detail: None },
        }),
        ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText { text: "b".to_string() }),
    ];
    let params = ChatCompletionParameters {
        messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(parts),
            name: None,
        })],
        ..Default::default()
    };
    assert_eq!(
        tpl.apply_chat_template(&params)?,
        "user:a<|vision_start|><|image_pad|><|vision_end|>\nb\nassistant:"
    );

    // ③ tokenizer_config.json 이 없으면 chat_template.jinja 로 폴백
    let dir2 = TempDir::new("it_chat_tpl_jinja");
    std::fs::write(dir2.path().join("chat_template.jinja"), CHAT_TEMPLATE)?;
    let tpl2 = ChatTemplate::init(dir2.path().to_str().unwrap())?;
    assert_eq!(tpl2.apply_chat_template(&user_hi)?, "user:hi\nassistant:");
    Ok(())
}

#[test]
fn chat_template_errors_and_python_method_rewrites() -> anyhow::Result<()> {
    let err = ChatTemplate::init("/nonexistent/logis_it_model").err().expect("missing model path");
    assert!(err.to_string().contains("model path not found"), "{err}");

    let empty = TempDir::new("it_chat_tpl_empty");
    let empty_path = empty.path().to_string_lossy().into_owned();
    let err = ChatTemplate::init(&empty_path).err().expect("no template files");
    assert!(err.to_string().contains("chat_template.jinja not found"), "{err}");
    let err = get_template(empty_path.clone()).err().expect("no tokenizer_config.json");
    assert!(err.to_string().contains("tokenizer_config.json not exists"), "{err}");

    let no_key = TempDir::new("it_chat_tpl_nokey");
    std::fs::write(no_key.path().join("tokenizer_config.json"), br#"{"bos_token": "<s>"}"#)?;
    let err = get_template(no_key.path().to_string_lossy().into_owned()).err().expect("missing chat_template key");
    assert!(err.to_string().contains("chat_template to str error"), "{err}");

    // 파이썬 문자열 메서드를 minijinja 테스트 문법으로 치환
    let py = TempDir::new("it_chat_tpl_python");
    std::fs::write(
        py.path().join("tokenizer_config.json"),
        serde_json::to_vec(&json!({
            "chat_template": "{% if x.startswith('a') %}A{% elif x.endswith('z') %}Z{% endif %}"
        }))?,
    )?;
    assert_eq!(
        get_template(py.path().to_string_lossy().into_owned())?,
        "{% if x is startingwith('a') %}A{% elif x is endingwith('z') %}Z{% endif %}"
    );
    Ok(())
}

#[test]
#[ignore = "BUG(B15): get_template unwraps the tokenizer_config.json parse result and panics instead of returning Err"]
fn get_template_returns_err_on_malformed_config() -> anyhow::Result<()> {
    let dir = TempDir::new("it_chat_tpl_malformed");
    std::fs::write(dir.path().join("tokenizer_config.json"), b"{not json")?;
    assert!(get_template(dir.path().to_string_lossy().into_owned()).is_err());
    Ok(())
}

/// tokenizers 0.22 직렬화 포맷의 최소 WordLevel 토크나이저 (공백 분리)
const WORDLEVEL_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "WhitespaceSplit"},
  "post_processor": null,
  "decoder": null,
  "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "hello": 1, "world": 2}, "unk_token": "[UNK]"}
}"#;

#[test]
fn tokenizer_init_errors_and_wordlevel_encoding() -> anyhow::Result<()> {
    let err = TokenizerModel::init("/nonexistent/logis_it_tok").err().expect("missing dir");
    assert!(err.to_string().contains("model path not found"), "{err}");

    let dir = TempDir::new("it_tokenizer_wordlevel");
    let path = dir.path().to_str().unwrap().to_string();
    let err = TokenizerModel::init(&path).err().expect("missing tokenizer.json");
    assert!(err.to_string().contains("tokenizer.json not found"), "{err}");

    std::fs::write(dir.path().join("tokenizer.json"), WORDLEVEL_TOKENIZER_JSON)?;
    let tok = TokenizerModel::init(&path)?;
    assert_eq!(tok.text_encode_vec("hello world".to_string(), false)?, vec![1, 2]);
    assert_eq!(tok.text_encode_vec("hello foo".to_string(), true)?, vec![1, 0], "unknown word → [UNK]");

    let t = tok.text_encode("world hello".to_string(), &candle_core::Device::Cpu)?;
    assert_eq!(t.dims(), &[1, 2]);
    assert_eq!(t.to_vec2::<u32>()?, vec![vec![2, 1]]);
    Ok(())
}

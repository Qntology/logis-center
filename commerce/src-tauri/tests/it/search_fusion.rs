use serde_json::json;
use tauri_app_lib::search_fusion::{field_contains, fuse, literal_ranking, query_domains, query_terms, value_text};

fn ids(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn query_terms_keep_numbers_with_inner_hyphens() {
    assert_eq!(query_terms("주문번호 20260625-100159 내역?"), ids(&["주문번호", "20260625-100159", "내역"]));
}

#[test]
fn query_terms_add_the_plain_number_of_a_formatted_amount() {
    assert_eq!(query_terms("329,800원 이상"), ids(&["329800", "329,800원", "이상"]));
}

#[test]
fn query_terms_drop_single_letters_and_repeats() {
    assert_eq!(query_terms("a Gift, a gift"), ids(&["gift"]));
}

#[test]
fn value_text_ignores_keys_and_envelope_hashes() {
    let text = value_text(r#"{"status":"paid","cc":"0xabc","title":"Red Coat","sale_price":1200,"text":{"note":"Soft"}}"#);
    assert!(text.contains("paid") && text.contains("red coat") && text.contains("1200") && text.contains("soft"));
    assert!(!text.contains("status") && !text.contains("0xabc"));
}

#[test]
fn literal_ranking_prefers_the_rare_term() {
    let docs = vec![
        ("a".to_string(), "주문".to_string()),
        ("b".to_string(), "주문".to_string()),
        ("c".to_string(), "주문 이도윤".to_string()),
        ("d".to_string(), "상품".to_string()),
    ];
    let ranked = literal_ranking(&ids(&["이도윤", "주문"]), &docs);
    assert_eq!(ranked[0], "c");
    assert!(!ranked.contains(&"d".to_string()), "맞는 낱말이 없는 문서는 순위에 없습니다");
}

#[test]
fn literal_ranking_keeps_input_order_on_ties() {
    let docs = vec![("x".to_string(), "coat".to_string()), ("y".to_string(), "coat".to_string()), ("z".to_string(), "shoe".to_string())];
    assert_eq!(literal_ranking(&ids(&["coat"]), &docs), ids(&["x", "y"]));
}

#[test]
fn fuse_rewards_agreement_between_tracks() {
    let a = ids(&["p", "q"]);
    let b = ids(&["q", "r"]);
    let out = fuse(&[(a.as_slice(), 1.0), (b.as_slice(), 1.0)], 20.0);
    assert_eq!(out[0].0, "q");
}

#[test]
fn fuse_is_deterministic_on_ties() {
    let a = ids(&["b"]);
    let c = ids(&["a"]);
    let out = fuse(&[(a.as_slice(), 1.0), (c.as_slice(), 1.0)], 20.0);
    assert_eq!(out.iter().map(|x| x.0.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
}

fn vocabulary() -> Vec<(String, Vec<String>)> {
    vec![
        ("order".to_string(), ids(&["order", "주문"])),
        ("tracking".to_string(), ids(&["shipment", "배송"])),
    ]
}

#[test]
fn query_domains_default_to_goods() {
    assert_eq!(query_domains("추운 날 몸을 따뜻하게 해 줄 물건", &vocabulary()), ids(&["goods"]));
}

#[test]
fn query_domains_read_compounds_and_plurals() {
    assert_eq!(query_domains("주문번호 20260625-100159 내역", &vocabulary()), ids(&["order"]));
    assert_eq!(query_domains("Orders from last week", &vocabulary()), ids(&["order"]));
    assert_eq!(query_domains("배송중 상태인 주문 목록", &vocabulary()), ids(&["order", "tracking"]));
}

#[test]
fn field_contains_matches_like_the_frontend_filter() {
    let row = json!({"color": "Sand Beige", "tags": ["Winter", "Sale"], "quantity": 12});
    assert!(field_contains(&row, "color", "sand"));
    assert!(field_contains(&row, "tags", "winter"));
    assert!(field_contains(&row, "quantity", "12"));
    assert!(!field_contains(&row, "color", "책상"));
    assert!(!field_contains(&row, "missing", "x"));
}

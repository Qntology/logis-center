//! 🌟 [ID/LINK SELF ALIAS] 목록 행의 자기 식별자 고르기 (모델 없음)
//!
//! 한 행에 주문 · 상품 · 운송장 · 쿠폰 링크가 함께 있고 코사인 게이트를 아무 후보도 통과하지 못하면, 예전에는 레거시 해석기가
//! 운송장(invoice) 링크를 주문의 자기 식별자로 잡았습니다. 이 페이지 유형의 별칭 매개변수(od_id)를 가진 링크를 먼저 골라야
//! 주문번호로 주문을 가리키는 배송 행과 연결이 맞습니다.

use tauri_app_lib::utils::ai_utils::{collect_id_link_candidates, self_alias_id_link};

const ORDER_ROW: &[&str] = &[
    r#"td a(href="https://sorae-market.test/admin/order/view?od_id=20260925-302130") 20260925-302130"#,
    r#"td a(href="https://sorae-market.test/admin/goods/edit?it_id=SR20656") 리넨 반팔 셔츠"#,
    r#"td a(href="https://sorae-market.test/admin/shipping/view?invoice=331773862277") 331773862277"#,
    r#"td a(href="https://sorae-market.test/admin/coupon/view?cp_id=CP5039") 재구매 고객 감사"#,
];

#[test]
fn order_row_picks_its_own_od_id_link() {
    let cands = collect_id_link_candidates(ORDER_ROW);
    assert!(cands.len() >= 4, "{cands:?}");
    let (token, href) = self_alias_id_link(&cands, "order").expect("주문 별칭 링크");
    assert_eq!(token, "20260925-302130");
    assert!(href.contains("od_id=20260925-302130"), "{href}");
}

#[test]
fn tracking_and_goods_rows_pick_their_own_links() {
    let cands = collect_id_link_candidates(ORDER_ROW);
    assert_eq!(self_alias_id_link(&cands, "tracking").map(|x| x.0).as_deref(), Some("331773862277"));
    assert_eq!(self_alias_id_link(&cands, "goods").map(|x| x.0), Some("SR20656".to_string()));
}

#[test]
fn ambiguous_or_missing_self_links_fall_through() {
    // 같은 유형의 다른 번호가 둘이면 고르지 않습니다(레거시 해석기로).
    let two = [
        r#"a(href="/admin/order/view?od_id=A-1") A-1"#,
        r#"a(href="/admin/order/view?od_id=B-2") B-2"#,
    ];
    assert!(self_alias_id_link(&collect_id_link_candidates(&two), "order").is_none());
    // 자기 유형 링크가 없으면 고르지 않습니다.
    let none = [r#"a(href="/admin/goods/edit?it_id=SR1") SR1"#];
    assert!(self_alias_id_link(&collect_id_link_candidates(&none), "order").is_none());
}

//! 🌟 [ANALYTIC PAGE / SITE] 행동 로그의 도메인 · 웹 경로 정의 (bias.json analytic_page_filters)
//!
//! link 경로 → 페이지 종류 판정과, 질의 문구 → 페이지 종류 해석을 12개 언어로 확인합니다. 모델 · 저장소 없음.

use tauri_app_lib::analytic::{page_kind_of, page_kinds_in_query, period_lexical_support, url_host_path};
use tauri_app_lib::stanza::quoted_token_flags;

#[test]
fn url_host_path_splits_host_and_query() {
    assert_eq!(url_host_path("https://Sorae-Market.test/goods/view?goodsNo=SR1"), ("sorae-market.test".into(), "/goods/view".into()));
    assert_eq!(url_host_path("https://a.test"), ("a.test".into(), "/".into()));
    assert_eq!(url_host_path("/order/complete?orderNo=1#top"), ("".into(), "/order/complete".into()));
}

#[test]
fn page_kind_takes_the_longest_path_prefix() {
    let k = |l: &str| page_kind_of(l);
    assert_eq!(k("https://a.test/goods/view?goodsNo=1").as_deref(), Some("goods_view"));
    assert_eq!(k("https://a.test/goods/list?cate=3").as_deref(), Some("goods_list"));
    assert_eq!(k("https://a.test/order/cart").as_deref(), Some("cart"));
    assert_eq!(k("https://a.test/order/checkout").as_deref(), Some("checkout"));
    assert_eq!(k("https://a.test/order/complete?orderNo=9").as_deref(), Some("order_complete"));
    assert_eq!(k("https://a.test/event/view?eventNo=EV1").as_deref(), Some("event_view"));
    assert_eq!(k("https://a.test/mypage/review/write?goodsNo=1").as_deref(), Some("review_write"));
    assert_eq!(k("https://a.test/mypage/orders").as_deref(), Some("mypage_orders"));
    assert_eq!(k("https://a.test/search?q=x").as_deref(), Some("search"));
    assert_eq!(k("https://a.test/").as_deref(), Some("home"));
    // 접두는 경로 조각 단위입니다: /cartoon 은 /cart 가 아닙니다.
    assert_eq!(k("https://a.test/cartoon"), None);
}

#[test]
fn page_phrases_in_twelve_languages() {
    let kinds = |q: &str| page_kinds_in_query(q).into_iter().map(|(k, _)| k).collect::<Vec<_>>();
    for q in [
        "소래마켓 상품 상세 페이지에서 옵션을 바꾼 기록",
        "option changes on product pages of Copperleaf Goods",
        "こもれび商店の商品詳細ページでオプションを変更した記録",
        "在こもれび商店商品详情页更改选项的记录",
        "Optionsänderungen auf Produktseiten von Kranichhof Markt",
        "changements d'option sur les fiches produit de X",
        "cambios de opción en las fichas de producto de X",
        "modifiche delle opzioni nelle pagine prodotto di X",
        "alterações de opção nas páginas de produto da X",
        "optiewijzigingen op productpagina's van X",
        "změny možností na stránkách produktů X",
        "تغييرات الخيارات في صفحات المنتجات على X",
    ] {
        assert_eq!(kinds(q), vec!["goods_view".to_string()], "{q}");
    }
    for q in [
        "X에서 결제 페이지까지 갔다가 결제하지 않은 기록",
        "visits on X that reached checkout but never paid",
        "Besuche bei X, die bis zur Kasse kamen, aber nicht bezahlt haben",
        "visites sur X arrivées au paiement sans commander",
        "návštěvy na X, které došly k pokladně, ale nezaplatily",
    ] {
        assert_eq!(kinds(q), vec!["checkout".to_string()], "{q}");
    }
    // 장바구니에 '담는' 행동은 상품 페이지에서 일어납니다. 장바구니 페이지 조건이 아닙니다.
    assert!(kinds("소래마켓에서 장바구니에 담은 기록").is_empty());
    assert!(kinds("add-to-cart actions on Copperleaf Goods").is_empty());
}

/// 기간(시간 · 계절) 뱅크 판정은 청크 낱말이 그 키의 구 낱말을 품을 때만 (변화형 포함). 벡터만 가까운 것은 기간이 아닙니다.
#[test]
fn period_needs_a_period_word_in_the_chunk() {
    let summer = ["summer", "summertime", "여름", "夏", "夏季", "Sommer", "été", "verano", "zomer", "léto", "الصيف"];
    let autumn = ["autumn", "fall", "가을", "秋", "Herbst", "automne", "otoño"];
    let this_year = ["this year", "올해", "今年", "dieses Jahr"];
    for chunk in ["여름에", "夏休みに買った", "last summer", "im Sommer", "zomerse", "v létě", "في الصيف", "Summers"] {
        assert!(period_lexical_support(chunk, &summer), "{chunk} → 여름");
    }
    // v2 행동 로그 질의에서 벡터로만 기간이 붙던 청크
    for chunk in ["actions on Copperleaf", "Copperleaf", "UV-blocking parasol", "车", "hat", "cupom"] {
        assert!(!period_lexical_support(chunk, &summer) && !period_lexical_support(chunk, &autumn), "{chunk}");
    }
    assert!(!period_lexical_support("bij", &this_year));
    assert!(!period_lexical_support("2026", &this_year), "숫자는 절대 기간 해석기 몫");
    assert!(period_lexical_support("올해는", &this_year));
}

/// 따옴표 안 낱말 표시: 이름('Summer getaway picks')은 기간 · 이벤트 청크로 올리지 않고 키워드로만 남깁니다.
#[test]
fn quoted_tokens_are_marked_in_order() {
    let toks = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let q = "coupon downloads on the 'Summer getaway picks' event page";
    let t = toks(&["coupon", "downloads", "on", "the", "'", "Summer", "getaway", "picks", "'", "event", "page"]);
    assert_eq!(
        quoted_token_flags(q, &t),
        vec![false, false, false, false, true, true, true, true, true, false, false]
    );
    let q = "「Summer getaway picks」のイベントページでクーポンを受け取った記録";
    let t = toks(&["「", "Summer", "getaway", "picks", "」", "の", "イベント", "ページ"]);
    assert_eq!(quoted_token_flags(q, &t), vec![true, true, true, true, true, false, false, false]);
    // 아포스트로피는 따옴표가 아닙니다
    let q = "coupons récupérés sur la page de l'opération « Summer getaway picks »";
    let t = toks(&["coupons", "l'", "opération", "«", "Summer", "»"]);
    assert_eq!(quoted_token_flags(q, &t), vec![false, false, false, true, true, true]);
}

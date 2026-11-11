//! 🌟 [ANALYTIC PAGE / SITE] 행동 로그의 도메인 · 웹 경로 정의 (bias.json analytic_page_filters)
//!
//! link 경로 → 페이지 종류 판정과, 질의 문구 → 페이지 종류 해석을 12개 언어로 확인합니다. 모델 · 저장소 없음.

use serde_json::{json, Value};
use tauri_app_lib::analytic::{
    evidence_visits, in_evidence_visit, negated_conditions, page_kind_of, page_kinds_in_query, period_lexical_support, record_window,
    url_host_path, NegatedConditions, VISIT_IDLE_MS,
};
use tauri_app_lib::stanza::quoted_token_flags;
use tauri_app_lib::utils::ai_utils::closed_tail_lemma;

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

/// 부정 조건 판정: 12개 언어 273문장(Stanza 토큰 포함)의 기준 판정과 같아야 합니다 (tests/fixtures/analytic/negation_cases.json).
/// 부정이 아닌 문장('결제하고 돌아오지 않은' · 'visitors who paid who did not review')도 들어 있습니다.
#[test]
fn negation_matches_the_reference_cases() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/analytic/negation_cases.json");
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(&path).expect("negation_cases.json")).expect("json");
    assert!(cases.len() >= 270, "고정 자료가 줄었습니다: {}", cases.len());
    let list = |v: &Value| -> Vec<String> { v.as_array().into_iter().flatten().filter_map(|x| x.as_str().map(String::from)).collect() };
    let mut wrong: Vec<String> = Vec::new();
    let mut negated = 0usize;
    for c in cases.iter() {
        let lang = c["lang"].as_str().unwrap_or("");
        let query = c["query"].as_str().unwrap_or("");
        let tokens: Vec<(String, String, String)> = c["tokens"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|t| (t[0].as_str().unwrap_or("").to_string(), t[1].as_str().unwrap_or("").to_string(), t[2].as_str().unwrap_or("").to_string()))
            .collect();
        let got = negated_conditions(query, lang, &tokens, &[]);
        let want = (list(&c["intents"]), list(&c["actions"]), list(&c["pages"]));
        if !want.0.is_empty() || !want.1.is_empty() || !want.2.is_empty() {
            negated += 1;
        }
        if (got.intents.clone(), got.actions.clone(), got.pages.clone()) != want {
            wrong.push(format!("[{lang}] {query}: 기대 {:?} · 실제 ({:?}, {:?}, {:?}) 근거 {:?}", want, got.intents, got.actions, got.pages, got.evidence));
        }
    }
    assert!(wrong.is_empty(), "{}건 다름 (부정 문장 {}건):\n{}", wrong.len(), negated, wrong.join("\n"));
    assert!(negated >= 100, "부정 문장이 너무 적습니다: {negated}");
}

/// 따옴표 안 낱말은 값이라 부정어로도 대상으로도 보지 않습니다 ('No Pay Day' 행사).
#[test]
fn negation_ignores_quoted_names() {
    let toks: Vec<(String, String, String)> = [("visits", "NOUN", "visit"), ("to", "ADP", "to"), ("'", "PUNCT", "'"), ("No", "DET", "no"), ("Pay", "PROPN", "Pay"), ("Day", "PROPN", "Day"), ("'", "PUNCT", "'")]
        .iter()
        .map(|(a, b, c)| (a.to_string(), b.to_string(), c.to_string()))
        .collect();
    let words: Vec<String> = toks.iter().map(|t| t.0.clone()).collect();
    let q = "visits to 'No Pay Day'";
    let quoted = quoted_token_flags(q, &words);
    assert!(negated_conditions(q, "en", &toks, &quoted).is_empty());
    assert!(!negated_conditions(q, "en", &toks, &[]).is_empty(), "따옴표 표시가 없으면 부정으로 읽힙니다 (위 판정이 표시 덕분인지 확인)");
}

fn ev(from: &str, at: i64, path: &str, page: &str, action: &str) -> (String, Value) {
    (from.to_string(), json!({ "from": from, "type": "click", "created_at": at, "link": format!("https://shop.test{path}"), "page": page, "action": action }))
}

/// 방문 단위 여집합: 결제를 마친 방문(주문 완료 페이지)의 기록만 빠지고, 같은 방문자가 37일 앞서 그만둔 방문은 남습니다.
/// 결제 버튼 기록은 증거가 아닙니다. 리포트는 묶은 방문의 시간 구간으로 판정합니다.
#[test]
fn visits_with_completion_evidence() {
    let day = 86_400_000i64;
    let t0 = 1_780_000_000_000i64;
    let records = vec![
        // A: 그만둔 방문 (장바구니 → 결제 페이지에서 결제수단만)
        ev("A", t0, "/order/cart", "cart", "주문하기"),
        ev("A", t0 + 50_000, "/order/checkout", "checkout", "결제수단"),
        // A: 37일 뒤 구매 방문
        ev("A", t0 + 37 * day, "/order/checkout", "checkout", "결제하기"),
        ev("A", t0 + 37 * day + 20_000, "/order/complete?orderNo=1", "order_complete", "주문 상세 보기"),
        // B: 결제 버튼까지 눌렀지만 완료 페이지가 없음 (결제 실패) → 부정된 일이 일어나지 않음
        ev("B", t0 + day, "/order/checkout", "checkout", "결제하기"),
    ];
    let neg = NegatedConditions { intents: vec!["purchase".into()], ..Default::default() };
    let visits = evidence_visits(&records, &neg);
    assert_eq!(visits, vec![("A".to_string(), t0 + 37 * day, t0 + 37 * day + 20_000)]);
    let window = |r: &(String, Value)| record_window(&r.1).unwrap();
    assert!(!in_evidence_visit("A", window(&records[1]), &visits), "그만둔 방문의 결제 페이지 기록은 남습니다");
    assert!(in_evidence_visit("A", window(&records[2]), &visits), "구매 방문의 결제 페이지 기록은 빠집니다");
    assert!(!in_evidence_visit("B", window(&records[4]), &visits), "결제 버튼만 누른 방문은 남습니다");
    // 리포트: 묶은 방문 구간이 구매 방문과 겹치면 빠지고, 그만둔 방문의 리포트는 남습니다
    let report = |s: i64, e: i64| json!({ "type": "report", "from": "A", "episode_started_at": s.to_string(), "episode_ended_at": e.to_string() });
    assert!(in_evidence_visit("A", record_window(&report(t0 + 37 * day, t0 + 37 * day + 20_000)).unwrap(), &visits));
    assert!(!in_evidence_visit("A", record_window(&report(t0, t0 + 50_000)).unwrap(), &visits));
    // 간격이 방문 만료(30분)를 넘으면 다른 방문입니다
    let split = vec![ev("C", t0, "/order/checkout", "checkout", ""), ev("C", t0 + VISIT_IDLE_MS + 1, "/order/complete", "order_complete", "")];
    assert_eq!(evidence_visits(&split, &neg), vec![("C".to_string(), t0 + VISIT_IDLE_MS + 1, t0 + VISIT_IDLE_MS + 1)]);
    // 제외 행동 · 제외 페이지도 증거입니다 ('쿠폰을 받지 않은' · '결제 페이지에 가지 않은')
    let neg_page = NegatedConditions { pages: vec!["checkout".into()], ..Default::default() };
    assert_eq!(evidence_visits(&records[4..5], &neg_page).len(), 1);
    let coupon = vec![ev("D", t0, "/goods/view?goodsNo=1", "goods_view", "쿠폰 받기")];
    let neg_action = NegatedConditions { actions: vec!["coupon".into()], ..Default::default() };
    assert_eq!(evidence_visits(&coupon, &neg_action).len(), 1);
}

/// Stanza 가 없을 때 원형 자리: 닫힌 조사 · 어미 표로 가장 많이 벗긴 어간 (부정 보조 용언 '않' 같은 한 글자 어간도)
#[test]
fn closed_tail_lemma_for_the_stanza_fallback() {
    assert_eq!(closed_tail_lemma("않은").as_deref(), Some("않"));
    assert_eq!(closed_tail_lemma("장바구니에").as_deref(), Some("장바구니"));
    assert_eq!(closed_tail_lemma("상품들을").as_deref(), Some("상품"));
    assert_eq!(closed_tail_lemma("決済ページで").as_deref(), Some("決済ページ"));
    assert_eq!(closed_tail_lemma("결제"), None);
}

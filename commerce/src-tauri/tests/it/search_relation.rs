//! 🌟 [RELATION] 고정점 → 목표 유형 관계 단계 · 서식 질의 근거 · 숫자 식별자 · 날짜 구간 · 행동 종류 (모델 · 저장소 없음)
//!
//! v2 관계형 합성 데이터의 12개 언어 질의에서 나온 실패를 규칙 단위로 고정합니다.

use chrono::NaiveDate;
use serde_json::json;
use tauri_app_lib::analytic::{action_kinds_in_query, doc_action_kinds, event_type_glued_keys};
use tauri_app_lib::search_relation::{
    arabic_bare_forms, contains_value, expose_own_number, link_self_token, mentions_period, own_ids, query_id_tokens, query_status_codes,
    target_type, trade_target, type_mentions, validity_period,
};
use tauri_app_lib::utils::ai_utils::{exact_absolute_period, record_id_token};

fn target(q: &str, lang: &str) -> Option<&'static str> {
    target_type(&type_mentions(q), lang)
}

/// 목표 유형은 명사구 머리: 한국어 · 일본어 · 중국어는 마지막 유형 낱말, 그 밖은 처음 유형 낱말
#[test]
fn target_type_follows_the_head_of_the_noun_phrase() {
    for (q, lang, want) in [
        ("주문 20260413-302272의 운송장 번호 알려줘", "ko", "tracking"),
        ("show the tracking numbers for order 20260413-302272", "en", "tracking"),
        ("注文20260413-302272の追跡番号を教えて", "ja", "tracking"),
        ("订单20260413-302272的快递单号是多少", "zh-hans", "tracking"),
        ("Sendungsnummern zur Bestellung 20260413-302272 anzeigen", "de", "tracking"),
        ("numéros de suivi de la commande 20260413-302272", "fr", "tracking"),
        ("números de seguimiento del pedido 20260413-302272", "es", "tracking"),
        ("أرقام تتبع الطلب 20260413-302272", "ar", "tracking"),
        ("Felix Wagner 고객이 주문한 상품", "ko", "goods"),
        ("products ordered by Felix Wagner", "en", "goods"),
        ("Felix Wagnerさんが注文した商品", "ja", "goods"),
        ("prodotti ordinati da Felix Wagner", "it", "goods"),
        ("이예준 고객이 리뷰를 남긴 상품", "ko", "goods"),
        ("반품된 주문의 상품", "ko", "goods"),
        ("products from returned orders", "en", "goods"),
        ("운송장 번호가 472339038575인 주문 찾아줘", "ko", "order"),
        ("which order was shipped with tracking number 472339038575?", "en", "order"),
        ("'Summer getaway coupon' 쿠폰을 사용한 주문", "ko", "order"),
        ("shipments delivered to 加藤 芽依 in September", "en", "tracking"),
    ] {
        assert_eq!(target(q, lang), Some(want), "{q}");
    }
    // 따옴표 안 이름의 유형 낱말은 보지 않습니다
    assert_eq!(target("products featured in the 'Order Week' promotion", "en"), Some("goods"));
    assert_eq!(target("hello there", "en"), None);
}

/// 저장 값이 질의에 통째로 나오는가 (라틴 이름은 낱말 경계, 한국어 조사가 붙어도 경계)
#[test]
fn stored_values_appear_in_the_query() {
    assert!(contains_value("products ordered by felix wagner", "felix wagner"));
    assert!(contains_value("felix wagner의 주문", "felix wagner"));
    assert!(!contains_value("products ordered by felix wagnerson", "felix wagner"));
    assert!(contains_value("이예준 고객이 리뷰를 남긴 상품", "이예준"));
    assert!(contains_value("shipments delivered to 加藤 芽依 in september", "加藤 芽依"));
}

/// 자기 식별자: 자기 유형 별칭 링크 매개변수 · no · 운송장 번호 (주문이 들고 있는 운송장 번호는 주문의 식별자가 아님)
#[test]
fn own_ids_and_order_numbers() {
    let order = json!({ "type": "order", "link": "/admin/order/view?od_id=20260413-302272", "tracking_number": "993186552339", "no": "" });
    assert_eq!(own_ids(&order), vec!["20260413302272".to_string()]);
    let tracking = json!({ "type": "tracking", "link": "/admin/shipping/view?invoice=902772346501", "tracking_number": "902772346501" });
    assert_eq!(own_ids(&tracking), vec!["902772346501".to_string()]);
    let coupon = json!({ "type": "coupon", "link": "/admin/coupon/view?cp_id=CP5052", "code": "KRHARVEST" });
    assert!(own_ids(&coupon).contains(&"CP5052".to_string()) && own_ids(&coupon).contains(&"KRHARVEST".to_string()));
    assert_eq!(link_self_token("/admin/goods/edit?it_id=kr20738", "goods").as_deref(), Some("kr20738"));
    assert_eq!(link_self_token("/order/complete?orderNo=20260921-300426", "order").as_deref(), Some("20260921-300426"));

    let mut item = json!({ "type": "order", "link": "/admin/order/view?od_id=20260501-300852", "no": "" });
    expose_own_number(&mut item, "order", "20260501300852");
    assert_eq!(item["no"], json!("20260501-300852"), "링크의 원문 번호");
    let mut trade = json!({ "type": "BL", "no": "BL-GB85095836" });
    expose_own_number(&mut trade, "BL", "x");
    assert_eq!(trade["no"], json!("BL-GB85095836"), "무역 서식은 건드리지 않음");

    let ids: Vec<String> = query_id_tokens("주문 20260413-302272의 운송장 번호 알려줘").into_iter().map(|(_, n)| n).collect();
    assert_eq!(ids, vec!["20260413302272".to_string()]);
}

/// 숫자 식별자는 수량 · 금액 · 날짜가 아닙니다 (날짜 · 금액 · 소수 · 단위 붙은 수는 식별자가 아님)
#[test]
fn record_id_tokens_are_not_quantities() {
    assert_eq!(record_id_token("20260413-302272의").as_deref(), Some("20260413-302272"));
    assert_eq!(record_id_token("472339038575인").as_deref(), Some("472339038575"));
    for v in ["2026-04-13", "20260413", "29900원", "46.05", "5개 이하", "500ml", "iPhone15", "1.234.567"] {
        assert_eq!(record_id_token(v), None, "{v}");
    }
}

/// 서식 질의의 목표 서식: 머리 자리의 서식 이름(복수형 · 굴절 허용), 일반 이름(서류 · documents)이면 전체
#[test]
fn trade_target_document_type() {
    let t = |q: &str, lang: &str| trade_target(q, lang).unwrap_or_default();
    assert!(t("bills of lading for purchase order PO-KR-86767", "en").contains(&"BL".to_string()));
    assert!(t("commercial invoices paid under L/C LC-DE7472990", "en").contains(&"CI".to_string()));
    assert!(t("신용장 LC-DE7472990 결제 건의 상업송장", "ko").contains(&"CI".to_string()));
    assert!(t("船荷証券BL-GB85095836の商業送り状", "ja").contains(&"CI".to_string()));
    assert_eq!(t("documents related to B/L BL-JP81652912", "en"), vec!["*".to_string()]);
    assert_eq!(t("B/L BL-JP81652912 관련 서류", "ko"), vec!["*".to_string()]);
    assert_eq!(t("6월에 펠릭스토우에서 선적된 서류", "ko"), vec!["*".to_string()]);
}

/// 서식 질의 조건 근거: 질의에 없는 값 · 날짜 단위가 붙은 수 · 식별자 속 수 · 다른 해의 날짜 · 없는 통화는 버리고, 글자 값은 힌트로
#[test]
fn trade_conditions_need_evidence_in_the_query() {
    let run = |q: &str, cond: serde_json::Value| -> serde_json::Value {
        let mut s = json!({ "context": [{ "type": "tracking", "condition": cond }] });
        tauri_app_lib::ground_trade_conditions(&mut s, q, &|_: &str| {});
        s["context"][0].clone()
    };
    let c = run(
        "documents for the shipment that left Felixstowe in June",
        json!({ "etd": { "operator": "gt", "value": "2019-06-30" }, "pol": { "operator": "eq", "value": "FELIXSTOWE" }, "vessel": { "operator": "eq", "value": "none" } }),
    );
    assert_eq!(c["condition"], json!({}));
    assert_eq!(c["hint"]["pol"]["value"], json!("FELIXSTOWE"));
    let c = run(
        "6月にフェリクストウから出港した貨物の書類",
        json!({ "currency": { "operator": "contains", "value": "JPY" }, "freight_amount": { "operator": "lte", "value": "6" } }),
    );
    assert_eq!(c["condition"], json!({}));
    let c = run(
        "信用状LC-DE7472990で決済した商業送り状",
        json!({ "doc_number": { "operator": "eq", "value": "LC-DE7472990" }, "freight_amount": { "operator": "eq", "value": "2990" } }),
    );
    assert_eq!(c["condition"], json!({ "doc_number": { "operator": "eq", "value": "LC-DE7472990" } }));
    let c = run(
        "6월에 펠릭스토우에서 선적된 서류",
        json!({ "issue_date": { "operator": "between", "value": format!("{}-06-01T00:00:00", chrono::Datelike::year(&chrono::Local::now())) } }),
    );
    assert!(c["condition"].get("issue_date").is_some(), "질의의 달에서 나온 올해 기간은 남김");
    let c = run("invoices in EUR over 5000", json!({ "currency": { "operator": "eq", "value": "EUR" }, "amount": { "operator": "gt", "value": "5000" } }));
    assert_eq!(c["condition"].as_object().map(|o| o.len()), Some(2));
}

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

/// 'from X to Y' 처럼 꼬리 연산자 없이 두 날짜 사이에 이음말이 끼는 언어도 한 구간 (v2 질의 A6 은 첫 날 하루만 기간이 되었음)
#[test]
fn date_ranges_with_connector_words() {
    let today = d(2026, 11, 9);
    for q in [
        "mobile taps on こもれび商店 from September 12, 2026 to October 2, 2026",
        "mobile Touch-Eingaben bei こもれび商店 vom 12. September 2026 bis 2. Oktober 2026",
        "appuis mobiles sur こもれび商店 du 12 septembre 2026 au 2 octobre 2026",
        "toques en el móvil en こもれび商店 del 12 de septiembre de 2026 al 2 de octubre de 2026",
        "tocchi da mobile su こもれび商店 dal 12 settembre 2026 al 2 ottobre 2026",
        "toques no celular na こもれび商店 de 12 de setembro de 2026 a 2 de outubro de 2026",
        "mobiele tikken bij こもれび商店 van 12 september 2026 tot 2 oktober 2026",
        "dotyky na mobilu na こもれび商店 od 12. září 2026 do 2. října 2026",
        "لمسات الجوال على こもれび商店 من 12 سبتمبر 2026 إلى 2 أكتوبر 2026",
    ] {
        let words: Vec<String> = q.split_whitespace().map(String::from).collect();
        let p = exact_absolute_period(&words, today).unwrap_or_else(|| panic!("기간 없음: {q}"));
        assert_eq!((p.start, p.end), (d(2026, 9, 12), d(2026, 10, 2)), "{q}");
    }
    // 목록 접속어는 구간이 아님 (첫 날짜만), 'between' 이 있으면 구간
    let w = |q: &str| q.split_whitespace().map(String::from).collect::<Vec<_>>();
    let p = exact_absolute_period(&w("orders on 2026-04-01 and 2026-04-03"), today).unwrap();
    assert_eq!((p.start, p.end), (d(2026, 4, 1), d(2026, 4, 1)));
    let p = exact_absolute_period(&w("orders between 2026-04-01 and 2026-04-03"), today).unwrap();
    assert_eq!((p.start, p.end), (d(2026, 4, 1), d(2026, 4, 3)));
}

/// 행동 로그의 버튼 행동: 질의 쪽(12개 언어 어간)과 기록 쪽(누른 버튼 글자)이 같은 종류로 만납니다. 페이지 문구는 행동이 아님
#[test]
fn analytic_action_kinds() {
    for q in [
        "Copperleaf Goods에서 장바구니에 담은 기록",
        "add-to-cart actions on Copperleaf Goods",
        "Copperleaf Goodsでカートに入れた記録",
        "Copperleaf Goods上加入购物车的记录",
        "Wann wurde bei Copperleaf Goods etwas in den Warenkorb gelegt?",
        "ajouts au panier sur Copperleaf Goods",
        "productos añadidos a la cesta en Copperleaf Goods",
        "aggiunte al carrello su Copperleaf Goods",
        "itens adicionados ao carrinho na Copperleaf Goods",
        "toevoegingen aan de winkelwagen bij Copperleaf Goods",
        "vložení do košíku na Copperleaf Goods",
        "عمليات الإضافة إلى السلة على Copperleaf Goods",
    ] {
        assert_eq!(action_kinds_in_query(q), vec!["add_to_cart".to_string()], "{q}");
    }
    assert!(action_kinds_in_query("coupon downloads on the 'Summer getaway picks' event page").contains(&"coupon".to_string()));
    assert!(action_kinds_in_query("option changes on product pages of こもれび商店").is_empty());
    for label in ["Add to cart", "장바구니 담기", "カートに入れる", "In den Warenkorb"] {
        assert_eq!(doc_action_kinds(&json!({ "action": label })), vec!["add_to_cart".to_string()], "{label}");
    }
    assert!(doc_action_kinds(&json!({ "action": "Proceed to checkout" })).is_empty());
    assert!(doc_action_kinds(&json!({ "action": "Charcoal" })).is_empty());
}

/// 숫자 없는 달 이름은 앞말(in · im · en · a · em · v · في) 뒤에서만 그 달 (영어 may · march 처럼 홀로 선 낱말은 아님)
#[test]
fn month_names_after_prepositions() {
    let today = d(2026, 11, 9);
    let w = |q: &str| q.split_whitespace().map(String::from).collect::<Vec<_>>();
    for q in [
        "shipments delivered to 加藤 芽依 in September",
        "Sendungen, die im September an 加藤 芽依 zugestellt wurden",
        "envíos entregados a 加藤 芽依 en septiembre",
        "spedizioni consegnate a 加藤 芽依 a settembre",
        "envios entregues a 加藤 芽依 em setembro",
        "zásilky doručené zákazníkovi 加藤 芽依 v září",
        "الشحنات التي سُلّمت إلى 加藤 芽依 في سبتمبر",
    ] {
        let p = exact_absolute_period(&w(q), today).unwrap_or_else(|| panic!("기간 없음: {q}"));
        assert_eq!((p.start, p.end), (d(2026, 9, 1), d(2026, 9, 30)), "{q}");
    }
    assert!(exact_absolute_period(&w("you may order now"), today).is_none());
}

/// 아랍어 서식 이름: 붙은 관사 · 전치사를 떼고 낱말로 맞추고, 불규칙 복수형(بوالص · فواتير)은 복수형 표로. 번호만 있으면 전체
#[test]
fn trade_target_arabic_and_number_only() {
    let t = |q: &str| trade_target(q, "ar").unwrap_or_default();
    assert!(t("بوالص الشحن الخاصة بأمر الشراء PO-KR-86767").contains(&"BL".to_string()));
    assert!(t("الفواتير التجارية المدفوعة بموجب خطاب الاعتماد LC-DE7472990").contains(&"CI".to_string()));
    assert!(t("الفاتورة التجارية لبوليصة الشحن BL-GB85095836").contains(&"CI".to_string()));
    assert_eq!(t("المستندات المتعلقة ببوليصة الشحن BL-JP81652912"), vec!["*".to_string()]);
    assert_eq!(trade_target("PO-KR-86767", "en"), Some(vec!["*".to_string()]));
    for (w, bare) in [("لبوليصة", "بوليصة"), ("الفاتورة", "فاتورة"), ("بالفاتورة", "فاتورة"), ("للشحن", "شحن"), ("والطلبات", "طلبات")] {
        assert!(arabic_bare_forms(w).contains(&bare.to_string()), "{w}");
    }
    assert_eq!(arabic_bare_forms("لا"), vec!["لا".to_string()], "남는 글자가 두 자 미만이면 떼지 않음");
}

/// 체코어 격변화(objednávce · zásilek · vrácených) · 아랍어 관사형 상태어(المرتجعة)
#[test]
fn czech_and_arabic_inflections() {
    assert_eq!(target("čísla zásilek k objednávce 20260413-302272", "cs"), Some("tracking"));
    assert_eq!(target("ke které objednávce patří číslo zásilky 472339038575?", "cs"), Some("order"));
    assert_eq!(target("zboží z vrácených objednávek", "cs"), Some("goods"));
    assert_eq!(target("منتجات الطلبات المرتجعة", "ar"), Some("goods"));
    let ret = tauri_app_lib::logic::parse_status("return") as i64;
    let cancel = tauri_app_lib::logic::parse_status("cancel") as i64;
    assert_eq!(query_status_codes("zboží z vrácených objednávek"), vec![ret]);
    assert_eq!(query_status_codes("منتجات الطلبات المرتجعة"), vec![ret]);
    assert_eq!(query_status_codes("zboží ze zrušených objednávek"), vec![cancel]);
    assert_eq!(query_status_codes("반품된 주문의 상품"), vec![ret]);
}

/// 질의가 행사 기간을 말하는가 (C7 '기간에 팔린') · 진열 상품만 묻는 질의(C6)는 아님
#[test]
fn period_words_and_validity() {
    for q in [
        "'Bereit für den Sommerurlaub' 기간에 팔린 상품",
        "products sold during the 'Bereit für den Sommerurlaub' promotion",
        "「Bereit für den Sommerurlaub」の期間中に売れた商品",
        "“Bereit für den Sommerurlaub”活动期间卖出的商品",
        "Artikel, die während der Aktion „Bereit für den Sommerurlaub“ verkauft wurden",
        "produits vendus pendant l'opération « Bereit für den Sommerurlaub »",
        "productos vendidos durante la promoción «Bereit für den Sommerurlaub»",
        "producten die tijdens de actie 'Bereit für den Sommerurlaub' zijn verkocht",
        "zboží prodané během akce „Bereit für den Sommerurlaub“",
        "المنتجات التي بيعت خلال عرض «Bereit für den Sommerurlaub»",
    ] {
        assert!(mentions_period(q), "{q}");
    }
    for q in [
        "products featured in the 'Bereit für den Sommerurlaub' promotion",
        "'Bereit für den Sommerurlaub' 기획전에 걸린 상품",
        "المنتجات المشمولة في عرض «Bereit für den Sommerurlaub»",
        "products featured in the 'during summer' promotion",
    ] {
        assert!(!mentions_period(q), "{q}");
    }
    let ev = json!({ "type": "event", "started_at": 1781481600000i64, "expired_at": 1783814400000i64 });
    assert_eq!(validity_period(&ev), Some((d(2026, 6, 15), d(2026, 7, 12))));
    assert_eq!(validity_period(&json!({ "type": "goods" })), None);
}

/// 띄어 쓰지 않는 일본어 · 중국어: 유형 확정어가 낱말 중간에 있어도 확정
#[test]
fn glued_event_words() {
    assert_eq!(event_type_glued_keys("こもれび商店の商品詳細ページでオプションを変更した記録"), vec!["change".to_string()]);
    assert_eq!(event_type_glued_keys("2026年9月12日から2026年10月2日までのこもれび商店のモバイルタッチ記録"), vec!["touch".to_string()]);
    assert_eq!(event_type_glued_keys("在こもれび商店商品详情页更改选项的记录"), vec!["change".to_string()]);
    assert!(event_type_glued_keys("UVカット日傘をカートに入れた記録").is_empty());
    assert!(event_type_glued_keys("option changes").is_empty());
}

/// 'taps' · 'appuis' · 'toques' 는 터치 (모바일 탭)
#[test]
fn tap_words_are_touch() {
    for w in ["taps", "appuis", "toques", "tocchi", "tikken", "タップ"] {
        assert_eq!(tauri_app_lib::analytic::event_type_exact_match(w).as_deref(), Some("touch"), "{w}");
    }
}

/// 번역 제목 고정점의 질의 조각: 낱말 1 ~ 5-그램과, 붙여 쓰는 문자 낱말의 글자 조각 (한글은 띄어 쓰므로 낱말 조각만)
#[test]
fn cross_lingual_query_spans() {
    use tauri_app_lib::search_relation::query_value_spans;
    let s = query_value_spans("자외선 차단 양산 장바구니 담기 기록");
    assert!(s.contains(&"자외선 차단 양산".to_string()) && s.contains(&"양산".to_string()));
    assert!(!s.contains(&"자외".to_string()), "한글은 글자 조각을 만들지 않습니다");
    let whole = "把防晒遮阳伞加入购物车的记录";
    let s = query_value_spans(whole);
    assert!(s.contains(&"防晒遮阳伞".to_string()) && s.contains(&whole.to_string()));
    assert!(s.iter().filter(|x| x.as_str() != whole).all(|x| x.chars().count() <= 10), "글자 조각은 10글자까지");
    let s = query_value_spans("who added UV-blocking parasol to the cart?");
    assert!(s.contains(&"UV-blocking parasol".to_string()) && s.contains(&"cart".to_string()), "앞뒤 기호를 뗍니다: {s:?}");
}

/// 번역 제목 고정점 규칙: 범위마다 1위가 2위를 앞서고 조각이 구조 낱말보다 제목에 가까워야 하며,
/// 범위가 여럿이면 전체 1위에서 폭 안의 범위 1위만 받습니다.
#[test]
fn cross_lingual_anchor_rule() {
    use tauri_app_lib::search_relation::{cross_anchor_pick, CROSS_ANCHOR_BAND};
    // 제목 0 · 1 은 사이트 k, 2 · 3 은 사이트 c. 조각 0 은 제목 0 을, 조각 1 은 제목 2 를 부릅니다.
    let groups: Vec<String> = ["k", "k", "c", "c"].iter().map(|s| s.to_string()).collect();
    let sim = vec![vec![0.76, 0.62, 0.70, 0.50], vec![0.40, 0.35, 0.69, 0.55]];
    let bank = vec![0.60, 0.60];
    let got = cross_anchor_pick(&sim, &bank, &groups);
    assert_eq!(got.len(), 1, "c 의 1위(0.70)는 전체 1위(0.76)에서 {} 밖이라 받지 않습니다: {got:?}", CROSS_ANCHOR_BAND);
    assert_eq!((got[0].group.as_str(), got[0].title, got[0].span), ("k", 0, 0));
    // 같은 상품이 두 사이트에 있으면(1위끼리 폭 안) 둘 다 받습니다
    let sim2 = vec![vec![0.76, 0.62, 0.75, 0.50]];
    assert_eq!(cross_anchor_pick(&sim2, &[0.60], &groups).len(), 2);
    // 2위와 차가 작으면 받지 않습니다
    let sim3 = vec![vec![0.76, 0.73, 0.40, 0.30]];
    assert!(cross_anchor_pick(&sim3, &[0.60], &groups).is_empty());
    // 조각이 구조 낱말('장바구니 담기')에 더 가까우면 제목 조각이 아닙니다
    let sim4 = vec![vec![0.76, 0.60, 0.40, 0.30]];
    assert!(cross_anchor_pick(&sim4, &[0.80], &groups).is_empty());
    assert!(cross_anchor_pick(&sim4, &[0.73], &groups).is_empty(), "은행 여유 0.03 < 0.05");
}

/// 품목 고정점의 품목 이름: description · description_of_goods 를 ' / ' 로 나누고 품목 행 설명을 더하며, 끝의 품목 번호 괄호는 뗍니다.
#[test]
fn trade_item_names_for_the_item_anchor() {
    use tauri_app_lib::search_relation::trade_item_names;
    let d = json!({
        "description": "USB-C-Multiport-Hub / Tragbares Kochgeschirr-Set",
        "items": [{ "description_of_goods": "Handheld mini fan (KR20287)" }, { "description": "USB-C-Multiport-Hub" }]
    });
    assert_eq!(trade_item_names(&d), vec!["usb-c-multiport-hub", "tragbares kochgeschirr-set", "handheld mini fan"]);
    let bl = json!({ "description": "세라믹 식칼" });
    let q = "세라믹 식칼 품목이 실린 선하증권";
    assert!(trade_item_names(&bl).iter().any(|n| contains_value(q, n)));
    assert!(!trade_item_names(&bl).iter().any(|n| contains_value("6월에 펠릭스토우에서 선적된 서류", n)));
}

/// 음역 고정점: 질의 언어로 음역된 항구 이름 ↔ 서류의 라틴 문자 값 (자음 골격), 짧은 골격의 우연한 일치는 받지 않습니다.
#[test]
fn phonetic_anchor_for_transliterated_values() {
    use tauri_app_lib::search_relation::{phonetic_spans, phonetic_value_hit};
    for q in [
        "6月にフェリクストウから出港した貨物の書類",
        "6月从费利克斯托装船的单据",
        "مستندات الشحنة التي غادرت فيليكستو في يونيو",
        "6월에 펠릭스토우에서 선적된 서류",
    ] {
        let hit = phonetic_value_hit(&phonetic_spans(q), "felixstowe");
        assert!(hit.as_ref().map_or(false, |(_, s)| *s >= 0.9), "{q}: {hit:?}");
    }
    // '운송' · '배송' 은 Incheon · 부산과 골격(3개)이 같지만 짧아서 보지 않습니다
    assert_eq!(phonetic_value_hit(&phonetic_spans("12월에 운송된 주문의 배송 상태"), "incheon"), None);
    // 라틴 문자 질의는 글자 고정점의 몫입니다
    assert!(phonetic_spans("documents that left Felixstowe in June").is_empty());
    // 다른 항구
    assert_eq!(phonetic_value_hit(&phonetic_spans("6月にフェリクストウから出港した貨物の書類"), "hamburg"), None);
}


//! Stanza 실행기 (src/stanza.rs, stanza-onnx-export format 2 변환본)
//!
//! - 의존 구문 디코딩: stanza 원본 `chuliu_edmonds_one_root` · `resolve_head_constraint_violations` 출력(tests/fixtures/stanza/decode_cases.json)과 같은 트리
//! - 질의 언어 판정(따옴표 · 문장 속 다른 문자 이름 제외) · 한국어 어간 · MWT 사전 대소문자 규칙
//! - (--ignored) 실제 변환본 12개 언어가 PyTorch Stanza 전체 파이프라인 출력(stanza-onnx-export 의 reference_outputs.json)과 같은지
//!     LOGIS_STANZA_DIR=<onnx_final 폴더> LOGIS_STANZA_REF=<reference_outputs.json> cargo test --test it stanza_runtime -- --ignored --nocapture

use serde_json::Value;
use std::collections::HashMap;
use tauri_app_lib::stanza::{
    chuliu_edmonds_one_root, detect_query_lang, dict_expansion, korean_stem, lang_code, resolve_head_constraint_violations,
    strip_quoted, StanzaLang,
};

fn cases() -> Value {
    serde_json::from_str(include_str!("../fixtures/stanza/decode_cases.json")).expect("decode_cases.json")
}

fn matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(f64::NEG_INFINITY)).collect())
        .collect()
}

fn ints(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
}

#[test]
fn chuliu_edmonds_matches_stanza() {
    let c = cases();
    for (i, case) in c["cle"].as_array().unwrap().iter().enumerate() {
        let tree = chuliu_edmonds_one_root(&matrix(&case["scores"]));
        assert_eq!(tree, ints(&case["tree"]), "사례 {i}");
        assert_eq!(tree.iter().skip(1).filter(|&&h| h == 0).count(), 1, "루트는 하나");
    }
}

#[test]
fn head_constraints_match_stanza() {
    let c = cases();
    let deprels: Vec<String> = c["deprels"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
    let groups: Vec<Option<&'static str>> = deprels
        .iter()
        .map(|d| match d.as_str() {
            "nsubj" => Some("subj"),
            "obj" => Some("obj"),
            _ => None,
        })
        .collect();
    let mut repaired = 0;
    for (i, case) in c["heads"].as_array().unwrap().iter().enumerate() {
        let scores = matrix(&case["scores"]);
        let labels: Vec<Vec<Vec<f64>>> = case["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| matrix(m))
            .collect();
        let plain = chuliu_edmonds_one_root(&scores);
        assert_eq!(plain, ints(&case["plain_tree"]), "사례 {i} 머리 제약 전");
        let (tree, raw) = resolve_head_constraint_violations(&scores, &labels, plain.clone(), &groups, 5);
        assert_eq!(tree, ints(&case["tree"]), "사례 {i} 트리");
        assert_eq!(raw, ints(&case["raw"]), "사례 {i} 관계");
        repaired += (tree != plain) as usize;
    }
    assert!(repaired > 0, "고쳐지는 사례가 하나는 있어야 이 시험이 의미가 있습니다");
}

#[test]
fn query_language_and_codes() {
    assert_eq!(detect_query_lang("5만원 이하 베이지 가디건", "english"), Some("ko"));
    assert_eq!(detect_query_lang("先週キャンセルされた注文", "korean"), Some("ja"));
    assert_eq!(detect_query_lang("货物的保险单", "korean"), Some("zh-hans"));
    assert_eq!(detect_query_lang("貨物の保険証券", "korean"), Some("ja"));
    assert_eq!(detect_query_lang("شهادة المنشأ للبضائع", "korean"), Some("ar"));
    // 그 언어에만 쓰는 글자 · 8개 라틴 언어로 좁힌 판정 (넓게 두면 라틴어 · 슬로바키아어로 판정되어 영어로 떨어졌음)
    assert_eq!(detect_query_lang("zrušené objednávky", "korean"), Some("cs"));
    assert_eq!(detect_query_lang("stornierte Bestellungen für Kunden", "korean"), Some("de"));
    assert_eq!(detect_query_lang("заказы", "korean"), None);
    assert_eq!(lang_code("czech"), Some("cs"));
    assert_eq!(lang_code("chinese"), Some("zh-hans"));
    assert_eq!(lang_code("russian"), None);
}

/// 데이터 값(행사 · 고객 · 쇼핑몰 · 수출자 이름)의 문자 체계가 질의 문장의 언어를 덮지 않는가 (v2 관계형 합성 데이터 질의).
/// 이전에는 영어 문장에 든 「夏休み準備セール」 · 加藤 芽依 · 이예준 때문에 질의가 일본어 · 중국어 · 한국어로 판정되어
/// 그 언어의 사전 · 변환본으로 풀렸습니다 (夏 → 여름 기간 조건 등).
#[test]
fn query_language_ignores_quoted_and_embedded_names() {
    let latin = |q: &str| {
        let l = detect_query_lang(q, "korean");
        assert!(matches!(l, Some("en" | "de" | "es" | "fr" | "it" | "pt" | "nl" | "cs")), "{q} → {l:?}");
    };
    // 따옴표로 묶은 이름
    assert_eq!(strip_quoted("products featured in the '夏休み準備セール' promotion"), "products featured in the   promotion");
    assert_eq!(strip_quoted("polizze di carico dell'ordine d'acquisto PO-KR-86767"), "polizze di carico dell'ordine d'acquisto PO-KR-86767");
    assert_eq!(strip_quoted("「夏休み準備セール」"), "「夏休み準備セール」", "따옴표 밖에 글자가 없으면 원문");
    latin("products featured in the '夏休み準備セール' promotion");
    latin("Artikel der Aktion „夏休み準備セール“");
    latin("produits de l'opération « 夏休み準備セール »");
    assert_eq!(detect_query_lang("“夏休み準備セール”活动的参与商品", "korean"), Some("zh-hans"));
    assert_eq!(detect_query_lang("'Bereit für den Sommerurlaub' 기간에 팔린 상품", "korean"), Some("ko"));
    assert_eq!(detect_query_lang("「Summer getaway picks」のイベントページでクーポンを受け取った記録", "korean"), Some("ja"));
    // 따옴표 없는 이름: 라틴 · 아랍 문장 속 한중일 이름 어절
    latin("products 이예준 wrote a review for");
    latin("shipments delivered to 加藤 芽依 in September");
    latin("bills of lading shipped by 株式会社ひなた工房");
    latin("prodotti recensiti da 이예준");
    assert_eq!(detect_query_lang("المنتجات التي كتب 이예준 مراجعة لها", "korean"), Some("ar"));
    // 한국어 · 일본어 · 중국어 문장 속 이름
    assert_eq!(detect_query_lang("株式会社ひなた工房에서 보낸 선하증권", "korean"), Some("ko"));
    assert_eq!(detect_query_lang("こもれび商店 상품 상세 페이지에서 옵션을 바꾼 기록", "korean"), Some("ko"));
    assert_eq!(detect_query_lang("Felix Wagner 고객이 주문한 상품", "english"), Some("ko"));
    assert_eq!(detect_query_lang("4월에 Fleece-lined zip hoodie 포함된 주문", "korean"), Some("ko"));
    assert_eq!(detect_query_lang("4月にFleece-lined zip hoodieを含む注文", "korean"), Some("ja"));
    assert_eq!(detect_query_lang("소래마켓で決済ページまで進んだのに支払わなかった記録", "korean"), Some("ja"));
    assert_eq!(detect_query_lang("注文書PO-KR-86767の船荷証券", "korean"), Some("ja"));
    assert_eq!(detect_query_lang("在こもれび商店商品详情页更改选项的记录", "korean"), Some("zh-hans"));
    assert_eq!(detect_query_lang("株式会社ひなた工房发运的提单", "korean"), Some("zh-hans"));
    assert_eq!(detect_query_lang("在소래마켓进入结算页面却没有付款的记录", "korean"), Some("zh-hans"));
    // 짧은 로망스어 문장(낱말 셋) 속 두 어절 한자 이름: 문장 낱말이 이름 어절보다 많으면 이름을 빼고 정합니다 (예전엔 중국어로 판정, v2 C8)
    assert_eq!(detect_query_lang("productos que 加藤 芽依 reseñó", "korean"), Some("es"));
    // 가나 없는 한자만: 일본어 신자체(気 · 圧)가 간체보다 많으면 일본어 (앱 언어가 일본어가 아니어도), 간체가 많으면 중국어
    assert_eq!(detect_query_lang("主蒸気圧力", "korean"), Some("ja"));
    assert_eq!(detect_query_lang("货物的保险单", "korean"), Some("zh-hans"));
    // 숫자가 든 번호의 로마자는 문장 언어가 아님 (아랍어 문장 + 서류 번호)
    assert_eq!(detect_query_lang("مستندات BL-DE-260911", "korean"), Some("ar"));
}

/// 배포 사전(full-bias)의 언어 구역 낱말 투표: whatlang 이 짧은 질의에서 틀리던 라틴 언어 질의 (v2 질의; fr → es, en → de,
/// de → en, es → pt, nl → fr · de, cs → pt 로 판정되던 것). 개발 사전(ko · en 구역만)에서는 투표하지 않습니다.
#[cfg(feature = "full-bias")]
#[test]
fn latin_queries_vote_with_dictionary_sections() {
    for (q, want) in [
        ("numéros de suivi de la commande 20260413-302272", "fr"),
        ("orders in April that included Fleece-lined zip hoodie", "en"),
        ("Artikel der Aktion „夏休み準備セール“", "de"),
        ("productos de pedidos devueltos", "es"),
        ("documenten bij cognossement BL-JP81652912", "nl"),
        ("obchodní faktura ke konosamentu BL-GB85095836", "cs"),
        ("met welk cognossement is Keramisch koksmes verscheept?", "nl"),
        // 근거 낱말이 쇼핑몰 이름 하나(goods)뿐이면 투표하지 않습니다
        ("aggiunte al carrello su Copperleaf Goods", "it"),
        // 표가 한 언어에 하나뿐이어도 그 언어 구역에만 있는 긴 소문자 낱말(prodotti)이면 그 언어, 이름 줄(Felix Wagner)은 세지 않음:
        // 예전엔 whatlang 이 pt 로 (v2 C3-kranich-es · C8-sorae-it · C8-komorebi-it)
        ("prodotti recensiti da 이예준", "it"),
        ("productos que pidió Felix Wagner", "es"),
        ("produtos que Felix Wagner encomendou", "pt"),
        ("prodotti recensiti da 加藤 芽依", "it"),
        // 이름 줄(Copperleaf Goods)을 빼면 영어 근거가 add 하나라 투표하지 않고 whatlang 이 정함
        ("add-to-cart actions on Copperleaf Goods", "en"),
    ] {
        assert_eq!(detect_query_lang(q, "korean"), Some(want), "{q}");
    }
}

/// 질의 목록(JSONL 의 id · query, `LOGIS_LANG_LIST`, 여러 파일은 `;`)마다 판정 언어를 찍고 id 끝 언어와 대조합니다 (v2 N:N 질의 전체 확인용).
#[test]
#[ignore = "진단 (LOGIS_LANG_LIST)"]
fn print_query_languages() {
    let Ok(list) = std::env::var("LOGIS_LANG_LIST") else {
        return;
    };
    let (mut ok, mut n) = (0, 0);
    for path in list.split(';').filter(|p| !p.trim().is_empty()) {
        for line in std::fs::read_to_string(path.trim()).unwrap().lines().filter(|l| !l.trim().is_empty()) {
            let v: Value = serde_json::from_str(line).unwrap();
            let (id, q) = (v["id"].as_str().unwrap_or(""), v["query"].as_str().unwrap_or(""));
            let want = tauri_app_lib::stanza::LANGS.iter().copied().filter(|l| id.ends_with(&format!("-{l}"))).max_by_key(|l| l.len());
            let got = detect_query_lang(q, "korean");
            n += 1;
            if got == want {
                ok += 1;
            }
            println!("@@LANG {id}\t{}\t{}\t{}\t{q}", want.unwrap_or("-"), got.unwrap_or("-"), if got == want { "ok" } else { "MISS" });
        }
    }
    println!("@@LANG 언어 판정 {ok}/{n}");
}

#[test]
fn korean_stems_and_mwt_casing() {
    assert_eq!(korean_stem("가디건+을", "ncn+jco"), "가디건");
    assert_eq!(korean_stem("취소+되+ㄴ", "ncpa+xsv+etm"), "취소");
    assert_eq!(korean_stem("5만+원", "nnc+nbu"), "5만원");
    assert_eq!(korean_stem("찾+아+주+어", "pvg+ecx+px+ef"), "찾");
    let d: HashMap<String, String> = [("del".to_string(), "de el".to_string()), ("zum".to_string(), "zu dem".to_string())].into_iter().collect();
    assert_eq!(dict_expansion(&d, "del").as_deref(), Some("de el"));
    assert_eq!(dict_expansion(&d, "DEL").as_deref(), Some("DE EL"));
    assert_eq!(dict_expansion(&d, "Zum").as_deref(), Some("Zu dem"));
    assert_eq!(dict_expansion(&d, "zUm"), None);
}

/// 검색 질의 단위 보기 (모델 필요): LOGIS_STANZA_QUERIES = [{"query": …}] JSON 파일. 질의마다 언어 · 단위 · 어절별 분석을 찍습니다.
#[test]
#[ignore = "LOGIS_STANZA_DIR · LOGIS_STANZA_QUERIES 필요"]
fn print_query_units() {
    let path = std::env::var("LOGIS_STANZA_QUERIES").expect("LOGIS_STANZA_QUERIES");
    let items: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut misaligned = 0;
    for it in items.as_array().unwrap() {
        let q = it["query"].as_str().unwrap();
        let lang = detect_query_lang(q, it.get("language").and_then(|v| v.as_str()).unwrap_or("korean")).unwrap_or("");
        match tauri_app_lib::stanza::analyze_query(q, lang) {
            Ok(a) => {
                misaligned += (!a.whitespace_units && !tauri_app_lib::stanza::lang_without_spaces(lang)) as usize;
                println!("[{lang}] {q}\n    {} ms · {} · {:?}", a.elapsed_ms, if a.whitespace_units { "공백 어절" } else { "Stanza 토큰" }, a.describe());
            }
            Err(e) => println!("[{lang}] {q}\n    오류 {e:#}"),
        }
    }
    assert_eq!(misaligned, 0, "공백 언어인데 어절에 맞추지 못한 질의");
}

/// 실제 변환본 · PyTorch 기준 출력 대조 (모델 필요)
#[test]
#[ignore = "LOGIS_STANZA_DIR · LOGIS_STANZA_REF 필요 (stanza-onnx-export onnx_final · reference_outputs.json)"]
fn real_models_match_pytorch_reference() {
    let root = std::path::PathBuf::from(std::env::var("LOGIS_STANZA_DIR").expect("LOGIS_STANZA_DIR"));
    let reference: Value = serde_json::from_str(&std::fs::read_to_string(std::env::var("LOGIS_STANZA_REF").expect("LOGIS_STANZA_REF")).unwrap()).unwrap();
    let only: Option<Vec<String>> = std::env::var("LOGIS_STANZA_LANGS").ok().map(|s| s.split(',').map(|x| x.to_string()).collect());
    let fields = ["upos", "xpos", "feats", "lemma", "head", "deprel"];
    let mut bad = Vec::new();
    for (lang, items) in reference.as_object().unwrap() {
        if only.as_ref().map_or(false, |o| !o.contains(lang)) {
            continue;
        }
        let t0 = std::time::Instant::now();
        let mut st = StanzaLang::load(&root.join(lang)).expect("변환본 열기");
        let (mut same, mut total) = (0usize, 0usize);
        for item in items.as_array().unwrap() {
            let text = item["text"].as_str().unwrap();
            let got = st.analyze(text).expect("analyze");
            let want = item["sentences"].as_array().unwrap();
            total += 1;
            let mut ok = got.len() == want.len();
            if ok {
                for (g, w) in got.iter().zip(want.iter()) {
                    let toks: Vec<&str> = w["tokens"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
                    let words: Vec<&str> = w["words"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
                    if g.tokens.iter().map(|s| s.as_str()).collect::<Vec<_>>() != toks
                        || g.words.iter().map(|x| x.text.as_str()).collect::<Vec<_>>() != words
                    {
                        ok = false;
                        break;
                    }
                    for f in fields {
                        let wv = w[f].as_array().unwrap();
                        for (k, gw) in g.words.iter().enumerate() {
                            let gv = match f {
                                "upos" => Value::from(gw.upos.clone()),
                                "xpos" => Value::from(gw.xpos.clone()),
                                "feats" => Value::from(gw.feats.clone()),
                                "lemma" => Value::from(gw.lemma.clone()),
                                "head" => Value::from(gw.head as u64),
                                _ => Value::from(gw.deprel.clone()),
                            };
                            if gv != wv[k] {
                                ok = false;
                                bad.push(format!("{lang} '{text}' {f} '{}': PyTorch {} · Rust {}", gw.text, wv[k], gv));
                            }
                        }
                    }
                }
            } else {
                bad.push(format!("{lang} '{text}': 문장 수 {} ≠ {}", got.len(), want.len()));
            }
            same += ok as usize;
        }
        println!("{lang}: {same}/{total} 원문 일치 ({:.1}s)", t0.elapsed().as_secs_f32());
    }
    for b in bad.iter().take(20) {
        println!("  ✗ {b}");
    }
    assert!(bad.is_empty(), "{}건 어긋남", bad.len());
}

/// 받는 주소: 저장소 루트(format 1)는 이미 배포된 앱 몫이고, 새 실행기는 main 의 format2/ 를 읽습니다.
#[test]
fn remote_base_is_format2_folder() {
    assert_eq!(
        tauri_app_lib::stanza::remote_base("zh-hans"),
        "https://huggingface.co/Qntology/stanza-zh-hans/resolve/main/format2"
    );
}

/// 올린 변환본 확인 (네트워크): 12개 언어 저장소의 format2/vocab.json 이 format 2 이고, 목록의 파일이 모두 받아지는지
#[test]
#[ignore = "네트워크 (huggingface.co)"]
fn hub_format2_manifests() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let client = reqwest::Client::new();
    for code in tauri_app_lib::stanza::LANGS {
        let base = tauri_app_lib::stanza::remote_base(code);
        let files = rt
            .block_on(tauri_app_lib::stanza::remote_manifest(&base))
            .unwrap_or_else(|e| panic!("{code}: {e:#}"));
        assert!(files.iter().any(|f| f == "tokenizer.onnx"), "{code}: {files:?}");
        for f in &files {
            let url = format!("{base}/{f}");
            let res = rt.block_on(client.head(&url).send()).unwrap_or_else(|e| panic!("{url}: {e}"));
            assert!(res.status().is_success() || res.status().is_redirection(), "{url}: {}", res.status());
        }
        println!("{code}: 파일 {}개", files.len());
    }
}

//! Stanza 실행기 (src/stanza.rs, stanza-onnx-export format 2 변환본)
//!
//! - 의존 구문 디코딩: stanza 원본 `chuliu_edmonds_one_root` · `resolve_head_constraint_violations` 출력(tests/fixtures/stanza/decode_cases.json)과 같은 트리
//! - 질의 언어 판정 · 한국어 어간 · MWT 사전 대소문자 규칙
//! - (--ignored) 실제 변환본 12개 언어가 PyTorch Stanza 전체 파이프라인 출력(stanza-onnx-export 의 reference_outputs.json)과 같은지
//!     LOGIS_STANZA_DIR=<onnx_final 폴더> LOGIS_STANZA_REF=<reference_outputs.json> cargo test --test it stanza_runtime -- --ignored --nocapture

use serde_json::Value;
use std::collections::HashMap;
use tauri_app_lib::stanza::{
    chuliu_edmonds_one_root, detect_query_lang, dict_expansion, korean_stem, lang_code, resolve_head_constraint_violations,
    StanzaLang,
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

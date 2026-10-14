//! 임베딩 처리량 · NMS 배틀 가속 검증 (STREAM A-5)
//!
//! - text_cache   : 전역 임베딩 캐시 — LRU 퇴출, 고정 계층, 0벡터 거부, 퍼지 후 유지, 공간 분리
//! - nvml         : 프로세스 전역 NVML 핸들 재사용 (init 1회, 이후 질의는 init/shutdown 없음)
//! - stage1       : RAW-CHUNK n-gram 프리패치 + 채점표가 기존 1건씩 루프와 비트 단위로 같은 점수·승자
//! - emb_bank     : 노름 캐시 코사인 / max-pool / surprisal 이 기존 함수와 비트 단위로 같음
//! - prefetch     : 필드 뱅크 프리패치 텍스트가 루프 요청을 모두 덮음
//! - GPU (ignore) : LOGIS_TEST_MODELS=1 이면 granite-embedding-97m-multilingual-r2 로
//!                  embed_many(배치) vs embed(per-text) 수치 대조와 처리량을 측정합니다.
//!                  cargo test --test it --no-run 후 실행 파일에
//!                  `app_embed -- --ignored --nocapture` 를 주어 돌립니다.

use std::collections::HashMap;
use std::sync::Arc;

use tauri_app_lib::models::embedding::text_cache::{
    is_cacheable, note_purge, space_id, TextEmbedCache, GRANITE_TEXT_CACHE,
};
use tauri_app_lib::utils::ai_utils::{
    cosine_similarity, max_pool_sim, max_pool_sim_bank, raw_chunk_spans, surprisal_dual_scores,
    surprisal_dual_scores_bank, weighted_max_pool_sim, weighted_max_pool_sim_bank, l2_norm_exact,
    LayoutScoreTable, NormBank, TripleBank, PAR_MIN_ROWS,
};

// ---------------------------------------------------------------------
//  결정론적 가짜 임베더 (텍스트 해시 → 384 차원, 일부 텍스트는 0벡터)
// ---------------------------------------------------------------------
fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn fake_embed(text: &str) -> Vec<f32> {
    let h = fnv(text);
    // 약 1/17 은 '실패' 처럼 0벡터를 돌려 0벡터 경로도 함께 검증합니다.
    if h % 17 == 3 {
        return vec![0.0; 384];
    }
    let mut x = h | 1;
    let mut v: Vec<f32> = (0..384)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    let n = l2_norm_exact(&v);
    for e in v.iter_mut() {
        *e /= n;
    }
    v
}

fn arc(v: Vec<f32>) -> Arc<[f32]> {
    Arc::from(v.into_boxed_slice())
}

// ---------------------------------------------------------------------
//  text_cache
// ---------------------------------------------------------------------
#[test]
fn text_cache_lru_evicts_oldest_but_keeps_pinned_and_recent() {
    let c = TextEmbedCache::new(10, 4);
    let sp = space_id(1, "t");
    // 고정 4건
    for i in 0..4 {
        assert!(c.insert(sp, format!("pin{}", i), arc(fake_embed(&format!("pin{}", i)).iter().map(|x| x + 0.5).collect()), true));
    }
    // 고정 상한(4)을 넘는 고정 요청은 LRU 로 들어갑니다.
    assert!(c.insert(sp, "pin_overflow".into(), arc(vec![1.0; 384]), true));
    assert!(!c.is_pinned(sp, "pin_overflow"));
    // LRU 9건 더 → LRU 10건 (상한 10, 아직 퇴출 없음)
    for i in 0..9 {
        c.insert(sp, format!("t{}", i), arc(vec![1.0 + i as f32; 384]), false);
    }
    assert_eq!(c.stats().lru, 10);
    assert_eq!(c.stats().evicted, 0);
    // t0 을 최근에 사용 → 퇴출 대상에서 밀려납니다.
    assert!(c.get(sp, "t0").is_some());
    // 11번째 LRU 엔트리 → 상한 초과 → 90%(9건)까지 오래된 것부터 퇴출 (2건)
    c.insert(sp, "t9".into(), arc(vec![3.0; 384]), false);
    let st = c.stats();
    assert_eq!(st.lru, 9, "{:?}", st);
    assert_eq!(st.pinned, 4);
    assert_eq!(st.evicted, 2);
    // 가장 오래된 두 건(pin_overflow, t1)이 퇴출되고, 최근 사용한 t0 과 고정 계층은 남습니다.
    assert!(!c.contains(sp, "pin_overflow"));
    assert!(!c.contains(sp, "t1"));
    assert!(c.contains(sp, "t0"));
    assert!(c.contains(sp, "t9"));
    for i in 0..4 {
        assert!(c.is_pinned(sp, &format!("pin{}", i)));
    }
    // LRU 계층만 비우기 → 고정 계층은 그대로
    let dropped = c.drop_lru_tier();
    assert_eq!(dropped, 9);
    assert_eq!(c.len(), 4);
}

#[test]
fn text_cache_never_stores_zero_or_nonfinite_vectors() {
    let c = TextEmbedCache::new(100, 100);
    let sp = space_id(2, "t");
    assert!(!is_cacheable(&[]));
    assert!(!is_cacheable(&vec![0.0; 384]));
    assert!(!is_cacheable(&vec![-0.0; 384]));
    let mut nan = vec![0.1; 384];
    nan[7] = f32::NAN;
    assert!(!is_cacheable(&nan));
    let mut inf = vec![0.1; 384];
    inf[3] = f32::INFINITY;
    assert!(!is_cacheable(&inf));
    let n = c.insert_many(
        sp,
        vec![
            ("zero".to_string(), arc(vec![0.0; 384])),
            ("nan".to_string(), arc(nan)),
            ("ok".to_string(), arc(vec![0.3; 384])),
        ],
        true,
    );
    assert_eq!(n, 1);
    assert!(!c.contains(sp, "zero"));
    assert!(!c.contains(sp, "nan"));
    assert!(c.contains(sp, "ok"));
    assert_eq!(c.stats().rejected, 2);
}

#[test]
fn text_cache_first_vector_wins_and_spaces_are_isolated() {
    let c = TextEmbedCache::new(100, 100);
    let a = space_id(3, "granite-r2/fwd-pertext/v1");
    let b = space_id(3, "granite-r2/fwd-batched/v1");
    assert_ne!(a, b);
    let v1 = vec![0.25f32; 384];
    let v2 = vec![0.5f32; 384];
    assert!(c.insert(a, "같은 텍스트".into(), arc(v1.clone()), false));
    // 같은 키 재삽입은 처음 벡터를 유지합니다.
    assert!(!c.insert(a, "같은 텍스트".into(), arc(v2.clone()), false));
    assert_eq!(&*c.get(a, "같은 텍스트").unwrap(), v1.as_slice());
    // 다른 공간(순전파 태그)에는 섞이지 않습니다.
    assert!(c.get(b, "같은 텍스트").is_none());
    // get_many(pin=true) 는 적중 엔트리를 고정 계층으로 올립니다.
    let got = c.get_many(a, &["같은 텍스트".to_string(), "없음".to_string()], true);
    assert!(got[0].is_some() && got[1].is_none());
    assert!(c.is_pinned(a, "같은 텍스트"));
    assert_eq!(c.forget_space(a), 1);
    assert!(c.is_empty());
}

#[test]
fn global_text_cache_survives_purge_paths() {
    // lifecycle.rs 의 실제 퍼지 경로(unload_embedding / deep_purge / relay)가 캐시를 비우지 않고
    // note_purge 만 부르는지 소스로 확인합니다(LogisModel 없이 돌릴 수 있는 범위의 검증).
    let lifecycle = include_str!("../../src/model/lifecycle.rs");
    for site in ["note_purge(\"unload_embedding\")", "note_purge(\"deep_purge\")", "note_purge(\"relay (image pipeline)\")"] {
        assert!(lifecycle.contains(site), "lifecycle.rs 에 {} 가 없습니다", site);
    }
    // 4곳: unload_embedding · deep_purge 2곳(6dc0245f 의 drop_embedding 분기 + 기존 전체 퍼지) · relay
    assert_eq!(lifecycle.matches("note_purge(").count(), 4, "퍼지 지점 수가 바뀌었습니다");
    for (no, line) in lifecycle.lines().enumerate() {
        let l = line.to_ascii_lowercase();
        let code = l.split("//").next().unwrap_or("");
        assert!(
            !(code.contains("cache") && code.contains(".clear()")),
            "lifecycle.rs:{} 에서 캐시를 비웁니다: {}", no + 1, line.trim()
        );
        assert!(
            !code.contains("drop_lru_tier") && !code.contains("forget_space("),
            "lifecycle.rs:{} 에서 전역 임베딩 캐시를 지웁니다: {}", no + 1, line.trim()
        );
    }

    // 다른 테스트와 겹치지 않는 고유 공간
    let sp = space_id(0xA5A5_0001, "it-purge");
    let v: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin() + 0.5).collect();
    assert!(is_cacheable(&v));
    GRANITE_TEXT_CACHE.insert(sp, "purge-survivor".into(), arc(v.clone()), true);
    // lifecycle.rs 의 세 퍼지 지점(unload_embedding / deep_purge / relay)이 부르는 함수
    note_purge("unload_embedding");
    note_purge("deep_purge");
    note_purge("relay (image pipeline)");
    let got = GRANITE_TEXT_CACHE.get(sp, "purge-survivor").expect("퍼지 후에도 남아 있어야 합니다");
    assert_eq!(&*got, v.as_slice());
    GRANITE_TEXT_CACHE.forget_space(sp);
}

// ---------------------------------------------------------------------
//  NVML 전역 핸들
// ---------------------------------------------------------------------
#[test]
fn nvml_handle_is_initialized_at_most_once() {
    use tauri_app_lib::utils::{gpu_mem_info, nvml_init_count};
    // 비교용: 기존 방식(질의마다 Nvml::init → 질의 → drop(shutdown + FreeLibrary)) 5회.
    // 전역 핸들이 열리기 전에 재야 DLL 적재·nvmlInit 비용이 그대로 드러납니다.
    let mut legacy_ms: Vec<f64> = Vec::new();
    for _ in 0..5 {
        let t = std::time::Instant::now();
        if let Ok(n) = nvml_wrapper::Nvml::init() {
            let _ = n.device_by_index(0).and_then(|d| d.memory_info());
        }
        legacy_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let first = std::time::Instant::now();
    let a = gpu_mem_info(0);
    let first_ms = first.elapsed().as_secs_f64() * 1000.0;
    let t = std::time::Instant::now();
    let mut ok = 0;
    for _ in 0..50 {
        if let Some((free, total)) = gpu_mem_info(0) {
            assert!(free <= total);
            ok += 1;
        }
    }
    let per_ms = t.elapsed().as_secs_f64() * 1000.0 / 50.0;
    let inits = nvml_init_count();
    // 전역 핸들이 열린 뒤에 기존 방식(init→drop)을 한 번 더 섞어도 핸들은 그대로여야 합니다(NVML 참조 계수).
    if a.is_some() {
        if let Ok(n) = nvml_wrapper::Nvml::init() {
            let _ = n.device_by_index(0).and_then(|d| d.memory_info());
        }
    }
    println!(
        "[NVML-GLOBAL] 첫 질의 {:.2}ms (init 포함) · 이후 50회 평균 {:.3}ms · 성공 {} · Nvml::init 횟수 {} · 결과 {:?} | 기존 방식(질의마다 init/shutdown) {:?} ms",
        first_ms, per_ms, ok, inits, a, legacy_ms.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>()
    );
    // 기존 방식 호출 뒤에도 전역 핸들은 그대로 동작해야 합니다.
    if a.is_some() {
        assert!(gpu_mem_info(0).is_some());
        assert_eq!(nvml_init_count(), inits);
    }
    // 핸들은 프로세스에서 한 번만 열립니다(NVIDIA 드라이버가 없으면 0).
    assert!(inits <= 1, "Nvml::init 이 {}회 호출되었습니다", inits);
    if a.is_some() {
        assert_eq!(ok, 50);
        // init/shutdown 이 없으면 질의 1회는 수 μs~수백 μs 입니다(기존 15~16ms).
        assert!(per_ms < 5.0, "질의당 {:.3}ms — 핸들이 재사용되지 않는 것 같습니다", per_ms);
    }
}

// ---------------------------------------------------------------------
//  STAGE-1 RAW-CHUNK 프리패치 동치 (가짜 임베더)
// ---------------------------------------------------------------------

/// query_commerce 와 같은 방식으로 레이아웃·앵커 뱅크를 만듭니다(가짜 임베더).
fn build_layout(lang: &str, categories: &[&str]) -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<f32>>) {
    let intent_anchors = [
        ("order", "manage order list, check order status"),
        ("goods", "product catalog data, items, find, search clothes"),
        ("tracking", "shipment status, tracking number, parcel"),
        ("review", "reviews, ratings, complaints"),
        ("coupon", "coupon codes, discount vouchers"),
        ("event", "marketing campaigns, promotions"),
        ("ignore", "ignore, stop, cancel"),
    ];
    let mut texts_to_embed = Vec::new();
    let mut emb_mappings = Vec::new();
    for (cat, text) in &intent_anchors {
        texts_to_embed.push(text.to_string());
        emb_mappings.push((cat.to_string(), "anchor".to_string()));
    }
    for cat in categories {
        let contexts = tauri_app_lib::utils::parsing::get_multi_pass_contexts(cat, lang);
        for (key, bias, prejudice) in contexts {
            texts_to_embed.push(bias);
            emb_mappings.push((cat.to_string(), format!("{}_bias", key)));
            let final_prej = if prejudice.trim().is_empty() { "random unrelated noise".to_string() } else { prejudice };
            texts_to_embed.push(final_prej);
            emb_mappings.push((cat.to_string(), format!("{}_prejudice", key)));
        }
    }
    let embedded: Vec<Vec<f32>> = texts_to_embed.iter().map(|t| fake_embed(t)).collect();
    let mut layout_embs = HashMap::new();
    let mut anchor_embs = HashMap::new();
    for (i, (cat, emb_type)) in emb_mappings.into_iter().enumerate() {
        if emb_type == "anchor" {
            anchor_embs.insert(cat, embedded[i].clone());
        } else {
            layout_embs.insert(format!("{}_{}", cat, emb_type), embedded[i].clone());
        }
    }
    (layout_embs, anchor_embs)
}

/// 기존(변경 전) RAW-CHUNK 루프를 그대로 옮긴 기준 구현: n-gram 마다 1건씩 임베딩.
fn stage1_reference(
    words: &[&str],
    categories: &[&str],
    lang: &str,
    layout_embs: &HashMap<String, Vec<f32>>,
    anchor_embs: &HashMap<String, Vec<f32>>,
    verb_emb: &[f32],
    calls: &mut usize,
) -> Vec<(usize, usize, String, HashMap<String, f32>)> {
    let mut raw_spans = Vec::new();
    for start in 0..words.len() {
        let max_end = words.len().min(start + 8);
        for end in (start + 2)..=max_end {
            let test_text = words[start..end].join(" ");
            *calls += 1;
            let test_emb = fake_embed(&test_text);
            let word_count = end - start;
            let v_sim = cosine_similarity(&test_emb, verb_emb);
            let beta = if word_count <= 2 { 0.05 } else { 0.10 };
            let verb_penalty = v_sim * beta;
            let penalty_weight = if word_count <= 2 { 0.3 } else { 0.7 };
            let mut scores = HashMap::new();
            for cat in categories {
                let contexts = tauri_app_lib::utils::parsing::get_multi_pass_contexts(cat, lang);
                let mut field_scores = Vec::new();
                for (key, _bias, _prejudice) in contexts {
                    let bias_emb = layout_embs.get(&format!("{}_{}_bias", cat, key)).cloned().unwrap_or(vec![0.0; 384]);
                    let prej_emb = layout_embs.get(&format!("{}_{}_prejudice", cat, key)).cloned().unwrap_or(vec![0.0; 384]);
                    let bias_score = cosine_similarity(&test_emb, &bias_emb);
                    let prej_score = cosine_similarity(&test_emb, &prej_emb);
                    let field_score = (bias_score - (prej_score * penalty_weight) - verb_penalty).max(0.0);
                    field_scores.push(field_score);
                }
                let anchor_score = if let Some(anchor_emb) = anchor_embs.get(*cat) {
                    cosine_similarity(&test_emb, anchor_emb).max(0.0)
                } else {
                    0.0
                };
                field_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let mut multi_pass_score = 0.0;
                let mut weight = 1.0;
                let max_pass = field_scores.len().min(5);
                for i in 0..max_pass {
                    multi_pass_score += field_scores[i] * weight;
                    weight *= 0.5;
                }
                multi_pass_score += anchor_score * 0.5;
                let word_count = end - start;
                let length_weight = 1.0 + ((word_count as f32 - 2.0) * 0.15);
                let weighted_base_score = multi_pass_score * length_weight;
                scores.insert(cat.to_string(), weighted_base_score);
            }
            raw_spans.push((start, end, test_text, scores));
        }
    }
    raw_spans
}

/// 기존 GAP BRIDGING 대결 점수 기준 구현.
fn gap_reference(emb: &[f32], cat: &str, lang: &str, layout_embs: &HashMap<String, Vec<f32>>) -> f32 {
    let contexts = tauri_app_lib::utils::parsing::get_multi_pass_contexts(cat, lang);
    let mut scores = Vec::new();
    for (key, _bias, _prej) in contexts {
        let bias_emb = layout_embs.get(&format!("{}_{}_bias", cat, key)).cloned().unwrap_or(vec![0.0; 384]);
        let prej_emb = layout_embs.get(&format!("{}_{}_prejudice", cat, key)).cloned().unwrap_or(vec![0.0; 384]);
        scores.push((cosine_similarity(emb, &bias_emb) - cosine_similarity(emb, &prej_emb)).max(0.0));
    }
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut s = 0.0;
    let mut weight = 1.0;
    for j in 0..scores.len().min(5) {
        s += scores[j] * weight;
        weight *= 0.5;
    }
    s
}

fn winner(scores: &HashMap<String, f32>) -> (String, f32) {
    // 카테고리 이름 순으로 정렬한 뒤 최대값(동률이면 이름이 앞선 쪽)으로 결정론적 승자를 구합니다.
    let mut v: Vec<(&String, &f32)> = scores.iter().collect();
    v.sort_by(|a, b| a.0.cmp(b.0));
    let mut best = (String::new(), f32::MIN);
    for (k, s) in v {
        if *s > best.1 {
            best = (k.clone(), *s);
        }
    }
    best
}

/// 같은 (가짜) 임베딩을 넣었을 때 RAW-CHUNK 스팬 목록 + LayoutScoreTable 채점이 기존의
/// n-gram 1건씩 루프와 비트 단위로 같은 점수·승자·갭 점수를 내는지 확인합니다.
/// (get_embedding_batch 와 get_embedding 의 벡터 동치는 여기서 보지 않습니다: per-text 모드에서
///  둘은 같은 embed() 를 거치며, 배치 순전파의 구성 무관성은 GPU 테스트가 확인합니다)
#[test]
fn stage1_score_table_matches_per_ngram_loop_bit_identical() {
    let categories = ["order", "goods", "tracking", "review", "coupon", "event", ""];
    let queries = [
        ("ko", "여름 원피스 중에서 할인율 20%에 속하지만 재고가 남은 상품 찾아줘 빨리"),
        ("ko", "지난달 배송 지연된 주문 송장 번호 알려줘"),
        ("en", "show me summer dresses under 50 dollars with good reviews and free shipping"),
        ("ja", "先月 配送 遅延 注文 一覧 を 見せて"),
    ];
    for (lang, query) in queries {
        let (layout_embs, anchor_embs) = build_layout(lang, &categories);
        let verb_emb = fake_embed(&format!("{} verb, predicate, idiom", lang));
        let words: Vec<&str> = query.split_whitespace().collect();

        let mut ref_calls = 0usize;
        let reference = stage1_reference(&words, &categories, lang, &layout_embs, &anchor_embs, &verb_emb, &mut ref_calls);

        // 새 경로: 스팬 목록 → 한 번의 배치(프리패치) → 채점표
        let spans = raw_chunk_spans(&words);
        let batch: Vec<Vec<f32>> = spans.iter().map(|(_, _, t)| fake_embed(t)).collect();
        let mut table = LayoutScoreTable::new();
        for cat in &categories {
            table.ensure_cat(cat, tauri_app_lib::utils::parsing::get_multi_pass_contexts(cat, lang), &layout_embs, &anchor_embs);
        }
        assert_eq!(spans.len(), reference.len(), "스팬 수");
        assert_eq!(ref_calls, spans.len(), "기존 루프의 1건 호출 수 = 배치 크기");
        let mut zero_spans = 0;
        for (i, (start, end, text)) in spans.iter().enumerate() {
            let (rs, re, rt, rscores) = &reference[i];
            assert_eq!((start, end, text), (rs, re, rt));
            if batch[i].iter().all(|&x| x == 0.0) {
                zero_spans += 1;
            }
            let new_scores: HashMap<String, f32> =
                table.raw_chunk_scores(&batch[i], &verb_emb, end - start).into_iter().collect();
            assert_eq!(new_scores.len(), rscores.len());
            for (cat, s) in rscores {
                let n = new_scores.get(cat).copied().expect("카테고리 누락");
                assert_eq!(n.to_bits(), s.to_bits(), "{} '{}' cat={} ref={} new={}", lang, text, cat, s, n);
            }
            assert_eq!(winner(&new_scores), winner(rscores));
            // GAP BRIDGING 점수도 같은 표로 비트 단위 동일
            for cat in &categories {
                let g_new = table.gap_score(&batch[i], cat).unwrap();
                let g_ref = gap_reference(&batch[i], cat, lang, &layout_embs);
                assert_eq!(g_new.to_bits(), g_ref.to_bits(), "gap {} {}", text, cat);
            }
        }
        println!(
            "[STAGE1-EQ] {} '{}' | 스팬 {}개 (0벡터 {}) | 1건 호출 {} → 배치 1회 | 점수·승자 비트 단위 일치",
            lang, query, spans.len(), zero_spans, ref_calls
        );
    }
}

// ---------------------------------------------------------------------
//  EMB-BANK 비트 동치
// ---------------------------------------------------------------------
fn bank(n: usize, tag: &str, zero_every: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| if zero_every > 0 && i % zero_every == 0 { vec![0.0; 384] } else { fake_embed(&format!("{}-{}", tag, i)) })
        .collect()
}

#[test]
fn emb_bank_scoring_is_bit_identical_to_legacy_helpers() {
    let queries: Vec<Vec<f32>> = (0..12).map(|i| fake_embed(&format!("q{}", i))).chain(std::iter::once(vec![0.0; 384])).collect();
    for &(n, zero_every) in &[(7usize, 3usize), (PAR_MIN_ROWS - 1, 11), (PAR_MIN_ROWS + 77, 9), (1115, 0)] {
        let b = bank(n, &format!("b{}", n), zero_every);
        let w: Vec<f32> = (0..n).map(|i| if i % 5 == 0 { 0.7 } else { 1.0 }).collect();
        let w_short: Vec<f32> = w.iter().take(n / 2).cloned().collect();
        let nb = NormBank::new(&b);
        for q in &queries {
            let qn = l2_norm_exact(q);
            assert_eq!(max_pool_sim(q, &b).to_bits(), max_pool_sim_bank(q, qn, &nb).to_bits());
            assert_eq!(weighted_max_pool_sim(q, &b, &w).to_bits(), weighted_max_pool_sim_bank(q, qn, &nb, &w).to_bits());
            assert_eq!(
                weighted_max_pool_sim(q, &b, &w_short).to_bits(),
                weighted_max_pool_sim_bank(q, qn, &nb, &w_short).to_bits()
            );
            for i in (0..n).step_by(37) {
                assert_eq!(cosine_similarity(q, &b[i]).to_bits(), nb.cos(i, q, qn).to_bits());
            }
        }
    }

    // surprisal_dual_scores vs 뱅크 버전 (필터 트리플 · 편견 트리플 · 스키마 뱅크)
    let mut fb: Vec<(String, String, Vec<f32>)> = Vec::new();
    for i in 0..470 {
        let cat = ["substantial_filters", "find_filters", "status_filters", "time_filters"][i % 4].to_string();
        let key = format!("k{}", i % 23);
        let e = if i % 31 == 0 { vec![0.0; 384] } else { fake_embed(&format!("fb{}", i)) };
        fb.push((cat, key, e));
    }
    let fp: Vec<(String, String, Vec<f32>)> = (0..120)
        .map(|i| (["substantial_filters", "time_filters"][i % 2].to_string(), format!("k{}", i % 23), fake_embed(&format!("fp{}", i))))
        .collect();
    let names: Vec<String> = (0..40).map(|i| format!("prop{}", i)).collect();
    let schema: Vec<Vec<Vec<f32>>> = (0..40).map(|i| bank(5 + (i * 7) % 60, &format!("s{}", i), 4)).collect();
    let skip: Vec<bool> = (0..40).map(|i| i % 9 == 0).collect();
    let fbank = TripleBank::new(&fb);
    let pbank = TripleBank::new(&fp);
    let sbanks: Vec<NormBank> = schema.iter().map(|b| NormBank::new(b)).collect();
    for q in &queries {
        let (f1, s1) = surprisal_dual_scores(q, &fb, &fp, &names, &schema, &skip);
        let (f2, s2) = surprisal_dual_scores_bank(q, &fbank, &pbank, &names, &sbanks, &skip);
        assert_eq!(f1.len(), f2.len());
        assert_eq!(s1.len(), s2.len());
        for (a, b) in f1.iter().zip(f2.iter()).chain(s1.iter().zip(s2.iter())) {
            assert_eq!((&a.category, &a.key, a.n), (&b.category, &b.key, b.n));
            assert_eq!(a.max_cos.to_bits(), b.max_cos.to_bits());
            assert_eq!(a.surprisal.to_bits(), b.surprisal.to_bits());
        }
        // filter_sim (0벡터 건너뛴 max)
        let mut m = 0.0f32;
        for (_, _, e) in fb.iter() {
            if e.iter().all(|&v| v == 0.0) { continue; }
            let s = cosine_similarity(q, e);
            if s > m { m = s; }
        }
        assert_eq!(m.to_bits(), fbank.max_nonzero(q, l2_norm_exact(q)).to_bits());
    }
}

// ---------------------------------------------------------------------
//  scheduler 필드 뱅크 프리패치 커버리지
// ---------------------------------------------------------------------
#[test]
fn field_bank_prefetch_covers_every_loop_request() {
    use tauri_app_lib::utils::ai_utils::{field_bank_prefetch_texts, label_bank_prefetch_texts, split_bias_phrases_weighted, prejudice_phrase_bank, label_phrase_bank};
    let fields = tauri_app_lib::utils::bias_schema::get_list_schema_fields("goods", "", "ko");
    assert!(!fields.is_empty());
    let pre: std::collections::HashSet<String> = field_bank_prefetch_texts(&fields, Some(("ko", "goods"))).into_iter().collect();
    // 루프가 실제로 요청하는 텍스트
    for (f_idx, (fname, _, bias_target, predefined_prej)) in fields.iter().enumerate() {
        assert!(pre.contains(bias_target));
        for p in split_bias_phrases_weighted(bias_target).0 {
            assert!(pre.contains(&p), "구 '{}' 누락", p);
        }
        for p in prejudice_phrase_bank("ko", "goods", fname) {
            assert!(pre.contains(&p));
        }
        let mut d = Vec::new();
        if !predefined_prej.trim().is_empty() { d.push(predefined_prej.clone()); }
        for (oi, (_, _, ob, _)) in fields.iter().enumerate() { if oi != f_idx { d.push(ob.clone()); } }
        assert!(pre.contains(&d.join(" , ")));
    }
    let lab: std::collections::HashSet<String> = label_bank_prefetch_texts(&fields, "ko", "goods").into_iter().collect();
    for (fname, _, _, _) in &fields {
        let (lp, _) = label_phrase_bank("ko", "goods", fname);
        if lp.is_empty() { continue; }
        for p in lp { assert!(lab.contains(&p)); }
        for p in prejudice_phrase_bank("ko", "goods", fname) { assert!(lab.contains(&p)); }
    }
}

// ---------------------------------------------------------------------
//  GPU: embed_many(배치) vs embed(per-text)  — LOGIS_TEST_MODELS=1
// ---------------------------------------------------------------------
/// `LOGIS_MODELS_DIR` 또는 OS 별 앱 데이터 경로의 granite 디렉터리 (tests/it/common.rs).
/// 목업(tests/fixtures/granite_mock)이면 디렉터리에 `MOCK` 파일이 있습니다.
fn granite_dir() -> Result<std::path::PathBuf, String> {
    crate::common::model_subdir("granite-embedding-97m-multilingual-r2", "model.safetensors")
}

/// ~300개의 길이가 섞인 텍스트: 짧은 뱅크 구, 질의형 n-gram, 긴 레이아웃 텍스트(>64, >512 토큰 포함).
fn mixed_texts() -> Vec<String> {
    use tauri_app_lib::utils::ai_utils::split_bias_phrases_full;
    let mut out: Vec<String> = Vec::new();
    // 1) 짧은 뱅크 구 (bias.json 의 필터·연산자 구)
    let dict = &*tauri_app_lib::utils::bias_schema::BIAS_DICT;
    for cat in ["operators", "time_filters", "season_filters", "status_filters", "find_filters"] {
        if let Some(obj) = dict.get(cat).and_then(|v| v.as_object()) {
            for (_, v) in obj {
                for f in ["bias", "semantic"] {
                    if let Some(s) = v.get(f).and_then(|x| x.as_str()) {
                        out.extend(split_bias_phrases_full(s));
                    }
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out.truncate(170);
    // 2) 질의형 n-gram
    let qs = [
        "여름 원피스 중에서 할인율 20%에 속하지만 재고가 남은 상품 찾아줘",
        "지난달 배송 지연된 주문 송장 번호 알려줘",
        "show me summer dresses under 50 dollars with good reviews",
    ];
    for q in qs {
        let w: Vec<&str> = q.split_whitespace().collect();
        for (_, _, t) in raw_chunk_spans(&w) {
            out.push(t);
        }
    }
    // 3) 긴 레이아웃 텍스트 (64~512 토큰 + 512 초과 청크 분할)
    for (cat, lang) in [("goods", "ko"), ("order", "en"), ("review", "ko"), ("tracking", "en")] {
        for (_, bias, prej) in tauri_app_lib::utils::parsing::get_multi_pass_contexts(cat, lang).into_iter().take(12) {
            out.push(bias);
            if !prej.trim().is_empty() {
                out.push(prej);
            }
        }
    }
    let long: String = out.iter().rev().take(40).cloned().collect::<Vec<_>>().join(", ");
    out.push(long);
    out.truncate(320);
    out
}

#[test]
#[ignore = "GPU + 로컬 모델 필요: LOGIS_TEST_MODELS=1"]
fn gpu_embed_many_matches_per_text_and_is_faster() {
    if std::env::var("LOGIS_TEST_MODELS").map(|v| v != "1").unwrap_or(true) {
        println!("LOGIS_TEST_MODELS=1 이 아니라 건너뜁니다.");
        return;
    }
    let dir = match granite_dir() {
        Ok(d) => d,
        Err(e) => return crate::common::skip_or_fail(&e),
    };
    let mock = dir.join("MOCK").exists();
    println!("[GPU-EQ] 모델: {} ({})", dir.display(), if mock { "목업 · 무작위 가중치" } else { "실모델" });
    // LOGIS_TEST_EMBED_DEVICE=cpu 이면 CPU 에서, 아니면 이 빌드의 GPU(CUDA/ROCm/Vulkan) 에서 봅니다.
    let device = if std::env::var("LOGIS_TEST_EMBED_DEVICE").map(|v| v.trim().eq_ignore_ascii_case("cpu")).unwrap_or(false) {
        candle_core::Device::Cpu
    } else {
        let Some(d) = tauri_app_lib::utils::new_gpu_device(0) else {
            println!("GPU 장치가 없어 건너뜁니다.");
            return;
        };
        d
    };
    use tauri_app_lib::models::embedding::EmbeddingModel;
    let t_load = std::time::Instant::now();
    let model = EmbeddingModel::new_with_device(&dir, &device).expect("load");
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    // 장치별 배치 순전파 기본값 (처음 판정할 때 [EMBED-BATCHED] 줄이 한 번 찍힙니다). 이 테스트는
    // embed_many / embed 를 직접 부르므로 LOGIS_EMBED_BATCHED 는 아래 수치 경로를 바꾸지 않습니다.
    println!(
        "[EMBED-BATCHED-TEST] 장치 {} · batched_enabled = {} · 캐시 공간 태그 {}",
        tauri_app_lib::models::embedding::text_cache::device_class(&device),
        model.batched_enabled(),
        tauri_app_lib::models::embedding::text_cache::forward_tag(&device)
    );
    let t_reload = std::time::Instant::now();
    let model2 = EmbeddingModel::new_with_device(&dir, &device).expect("reload");
    let reload_ms = t_reload.elapsed().as_secs_f64() * 1000.0;
    drop(model2);
    println!("[EMBED-TOKENIZER] 최초 로드 {:.0}ms · 재로드(토크나이저 재사용) {:.0}ms", load_ms, reload_ms);

    let texts = mixed_texts();
    let lens: Vec<usize> = texts.iter().map(|t| model.token_len(t).unwrap()).collect();
    let n_short = lens.iter().filter(|&&l| l <= 64).count();
    let n_long = lens.iter().filter(|&&l| l > 64 && l <= 512).count();
    let n_chunked = lens.iter().filter(|&&l| l > 512).count();
    println!(
        "[GPU-EQ] 텍스트 {}건 | ≤64토큰 {} · 65~512 {} · >512(청크 분할) {} | 토큰 중앙값 {} · 최대 {}",
        texts.len(), n_short, n_long, n_chunked,
        { let mut s = lens.clone(); s.sort(); s[s.len() / 2] },
        lens.iter().max().unwrap()
    );

    // 예열
    let _ = model.embed(&texts[0]).unwrap();
    let _ = model.embed_many(&texts[..8], &model.batch_opts()).unwrap();
    let _ = device.synchronize();

    // ── 기준: per-text embed() 순차 ──
    let t0 = std::time::Instant::now();
    let per_text: Vec<Vec<f32>> = texts.iter().map(|t| model.embed(t).unwrap()).collect();
    let per_text_s = t0.elapsed().as_secs_f64();
    // 기존 per-text 경로 자체의 재현성(같은 텍스트 두 번)
    let per_text2: Vec<Vec<f32>> = texts.iter().map(|t| model.embed(t).unwrap()).collect();

    // ── 기존 embed_batch (3스레드 × embed) ──
    let t1 = std::time::Instant::now();
    // (embed_batch 는 이제 CUDA 에서 1스레드이므로, 기존 3스레드 동작을 그대로 재현해 잽니다)
    let threaded: Vec<Vec<f32>> = {
        let chunk = (texts.len() + 2) / 3;
        let mut out = vec![Vec::new(); texts.len()];
        std::thread::scope(|s| {
            let hs: Vec<_> = texts
                .chunks(chunk)
                .enumerate()
                .map(|(ci, part)| {
                    let m = &model;
                    s.spawn(move || (ci * chunk, part.iter().map(|t| m.embed(t).unwrap()).collect::<Vec<_>>()))
                })
                .collect();
            for h in hs {
                let (st, v) = h.join().unwrap();
                for (k, e) in v.into_iter().enumerate() { out[st + k] = e; }
            }
        });
        let _ = device.synchronize();
        out
    };
    let threaded_s = t1.elapsed().as_secs_f64();

    // ── 배치 순전파 (2회째를 측정) ──
    let opts = model.batch_opts();
    let (batched_first, _) = model.embed_many_with_stats(&texts, &opts).unwrap();
    let t2 = std::time::Instant::now();
    let (batched, st) = model.embed_many_with_stats(&texts, &opts).unwrap();
    let batched_s = t2.elapsed().as_secs_f64();

    // ── 짧은 뱅크 구만 (가장 흔한 경우) ──
    let short_texts: Vec<String> = texts.iter().zip(lens.iter()).filter(|(_, &l)| l <= 64).map(|(t, _)| t.clone()).collect();
    let t3 = std::time::Instant::now();
    for t in &short_texts { let _ = model.embed(t).unwrap(); }
    let short_per_s = t3.elapsed().as_secs_f64();
    let t4 = std::time::Instant::now();
    let _ = model.embed_many(&short_texts, &opts).unwrap();
    let short_batch_s = t4.elapsed().as_secs_f64();
    // 단건 호출 지연(검색 질의 1건): 고정 모양은 1건이어도 B(L) 행을 계산합니다.
    let singles: Vec<&String> = short_texts.iter().step_by((short_texts.len() / 20).max(1)).take(20).collect();
    let t5 = std::time::Instant::now();
    for t in &singles { let _ = model.embed(t).unwrap(); }
    let single_per_ms = t5.elapsed().as_secs_f64() * 1000.0 / singles.len() as f64;
    let t6 = std::time::Instant::now();
    for t in &singles { let _ = model.embed_many(std::slice::from_ref(*t), &opts).unwrap(); }
    let single_batch_ms = t6.elapsed().as_secs_f64() * 1000.0 / singles.len() as f64;
    println!(
        "[GPU-EQ] 단건 호출(짧은 구 {}건 평균): per-text {:.1}ms · embed_many {:.1}ms (token_budget {})",
        singles.len(), single_per_ms, single_batch_ms, opts.token_budget
    );

    // ── 수치 대조 ──
    let mut min_cos = f32::MAX;
    let mut max_abs = 0f32;
    let mut identical = 0usize;
    let (mut min_cos_short, mut min_cos_long) = (f32::MAX, f32::MAX);
    let (mut ident_short, mut ident_long) = (0usize, 0usize);
    let mut worst = String::new();
    let mut thread_identical = 0usize;
    for i in 0..texts.len() {
        let a = &per_text[i];
        let b = &batched[i];
        let c = cosine_similarity(a, b);
        let d = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let same = a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
        if threaded[i].iter().zip(a.iter()).all(|(x, y)| x.to_bits() == y.to_bits()) { thread_identical += 1; }
        if same { identical += 1; }
        if lens[i] <= 64 {
            min_cos_short = min_cos_short.min(c);
            if same { ident_short += 1; }
        } else {
            min_cos_long = min_cos_long.min(c);
            if same { ident_long += 1; }
        }
        if c < min_cos {
            min_cos = c;
            worst = format!("{}토큰 '{}'", lens[i], texts[i].chars().take(40).collect::<String>());
        }
        max_abs = max_abs.max(d);
    }
    // 구성 무관성: 같은 텍스트는 단건 호출 · 다른 구성의 배치(역순 절반 · 표본끼리) 어디서 구해도
    // 전체 배치에서 구한 벡터와 비트 단위로 같아야 합니다(고정 모양 마이크로배치).
    let same_bits = |a: &Vec<f32>, b: &Vec<f32>| a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
    let mut single_identical = 0usize;
    let probe: Vec<usize> = (0..texts.len()).step_by(texts.len() / 12 + 1).collect();
    for &i in &probe {
        let s = model.embed_many(std::slice::from_ref(&texts[i]), &opts).unwrap();
        if same_bits(&s[0], &batched[i]) { single_identical += 1; }
    }
    let probe_texts: Vec<String> = probe.iter().map(|&i| texts[i].clone()).collect();
    let probe_batch = model.embed_many(&probe_texts, &opts).unwrap();
    let probe_identical = probe.iter().enumerate().filter(|(k, &i)| same_bits(&probe_batch[*k], &batched[i])).count();
    let sub_idx: Vec<usize> = (0..texts.len()).rev().step_by(2).collect();
    let sub_texts: Vec<String> = sub_idx.iter().map(|&i| texts[i].clone()).collect();
    let sub_batch = model.embed_many(&sub_texts, &opts).unwrap();
    let sub_identical = sub_idx.iter().enumerate().filter(|(k, &i)| same_bits(&sub_batch[*k], &batched[i])).count();

    let n = texts.len() as f64;
    println!(
        "[GPU-EQ] per-text 순차 {:.2}s ({:.0}건/s) | 기존 embed_batch 3스레드 {:.2}s ({:.0}건/s) | embed_many {:.3}s ({:.0}건/s) → per-text 대비 {:.1}배 · 3스레드 대비 {:.1}배",
        per_text_s, n / per_text_s, threaded_s, n / threaded_s, batched_s, n / batched_s, per_text_s / batched_s, threaded_s / batched_s
    );
    println!(
        "[GPU-EQ] 짧은 구(≤64토큰) {}건: per-text {:.2}s ({:.0}건/s) → embed_many {:.3}s ({:.0}건/s) = {:.1}배",
        short_texts.len(), short_per_s, short_texts.len() as f64 / short_per_s, short_batch_s, short_texts.len() as f64 / short_batch_s, short_per_s / short_batch_s
    );
    println!(
        "[GPU-EQ] 마이크로배치 {}개 (패딩 {}) | 실토큰 {} / 연산토큰 {} ({:.2}배)",
        st.micro_batches, st.padded_batches, st.real_tokens, st.computed_tokens, st.computed_tokens as f64 / st.real_tokens.max(1) as f64
    );
    println!(
        "[GPU-EQ] 수치: 최소 코사인 {:.7} (≤64토큰 {:.7} · >64토큰 {:.7}) · 최대 절대차 {:.3e} · 비트 동일 {}/{} (짧은 {}/{} · 긴 {}/{}) · 최악 {}",
        min_cos, min_cos_short, min_cos_long, max_abs, identical, texts.len(), ident_short, n_short, ident_long, texts.len() - n_short, worst
    );
    println!(
        "[GPU-EQ] 구성 무관성(전체 배치 대비 비트 동일): 단건 {}/{} · 표본끼리 배치 {}/{} · 역순 절반 배치 {}/{} | 기존 3스레드 embed_batch == 순차 embed 비트 동일 {}/{}",
        single_identical, probe.len(), probe_identical, probe.len(), sub_identical, sub_idx.len(), thread_identical, texts.len()
    );
    let bits_eq = |a: &Vec<f32>, b: &Vec<f32>| a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
    let per_text_repro = (0..texts.len()).filter(|&i| bits_eq(&per_text[i], &per_text2[i])).count();
    let batched_repro = (0..texts.len()).filter(|&i| bits_eq(&batched[i], &batched_first[i])).count();
    let thread_bad: Vec<f32> = (0..texts.len())
        .filter(|&i| !bits_eq(&threaded[i], &per_text[i]))
        .map(|i| cosine_similarity(&threaded[i], &per_text[i]))
        .collect();
    println!(
        "[GPU-EQ] 재현성: per-text 2회 비트 동일 {}/{} · embed_many 2회 비트 동일 {}/{} · 3스레드 불일치 {}건 코사인 {:?}",
        per_text_repro, texts.len(), batched_repro, texts.len(), thread_bad.len(), thread_bad
    );

    // ── F32(CPU) 기준 정확도: 두 BF16 경로가 '참값' 에서 얼마나 떨어져 있는가 ──
    let cpu = EmbeddingModel::new_with_device(&dir, &candle_core::Device::Cpu).expect("cpu load");
    let reference: Vec<Vec<f32>> = texts.iter().map(|t| cpu.embed(t).unwrap()).collect();
    let stats = |v: &Vec<Vec<f32>>, mask: &dyn Fn(usize) -> bool| -> (f64, f32) {
        let mut sum = 0f64;
        let mut mn = f32::MAX;
        let mut k = 0usize;
        for i in 0..texts.len() {
            if !mask(i) { continue; }
            let c = cosine_similarity(&v[i], &reference[i]);
            sum += c as f64;
            mn = mn.min(c);
            k += 1;
        }
        (sum / k.max(1) as f64, mn)
    };
    let short = |i: usize| lens[i] <= 64;
    let long = |i: usize| lens[i] > 64;
    let (pt_mean_s, pt_min_s) = stats(&per_text, &short);
    let (bt_mean_s, bt_min_s) = stats(&batched, &short);
    let (pt_mean_l, pt_min_l) = stats(&per_text, &long);
    let (bt_mean_l, bt_min_l) = stats(&batched, &long);
    println!(
        "[GPU-EQ] F32(CPU) 기준 코사인 — ≤64토큰: per-text 평균 {:.6} / 최소 {:.6} · embed_many 평균 {:.6} / 최소 {:.6} | >64토큰: per-text 평균 {:.6} / 최소 {:.6} · embed_many 평균 {:.6} / 최소 {:.6}",
        pt_mean_s, pt_min_s, bt_mean_s, bt_min_s, pt_mean_l, pt_min_l, bt_mean_l, bt_min_l
    );

    // ── 판정 일치: 질의형 n-gram 마다 가장 가까운 뱅크 구(1위)가 같은가 ──
    let bank_idx: Vec<usize> = (0..texts.len()).filter(|&i| i < 170).collect();
    let query_idx: Vec<usize> = (170..texts.len()).collect();
    let top1 = |v: &Vec<Vec<f32>>, q: usize| -> usize {
        let mut best = (usize::MAX, f32::MIN);
        for &b in &bank_idx {
            let s = cosine_similarity(&v[q], &v[b]);
            if s > best.1 { best = (b, s); }
        }
        best.0
    };
    let mut agree_pb = 0usize;
    let mut agree_pr = 0usize;
    let mut agree_br = 0usize;
    for &q in &query_idx {
        let (p, b, r) = (top1(&per_text, q), top1(&batched, q), top1(&reference, q));
        if p == b { agree_pb += 1; }
        if p == r { agree_pr += 1; }
        if b == r { agree_br += 1; }
    }
    println!(
        "[GPU-EQ] 판정 일치(질의 {}건 × 뱅크 {}구 1위): per-text↔embed_many {}/{} · per-text↔F32 {}/{} · embed_many↔F32 {}/{}",
        query_idx.len(), bank_idx.len(), agree_pb, query_idx.len(), agree_pr, query_idx.len(), agree_br, query_idx.len()
    );

    // CI 주석(annotation) 한 줄 요약 — .github/scripts/lib.sh publish_notices 가 이 줄을 우선 싣습니다.
    println!(
        "[EMBED-SUMMARY] {} · {} · fixed_shape={} · 전체 {:.2}배 · 짧은 구 {:.2}배 · 단건 per-text {:.1}ms / embed_many {:.1}ms · 비트 동일 {}/{} · 구성 무관 {}/{} {}/{} {}/{} · 재현 {}/{} · F32 평균 코사인 Δ(embed_many−per-text) ≤64 {:+.1e} >64 {:+.1e} · 판정 {}/{} (F32 {}/{})",
        if mock { "목업" } else { "실모델" },
        tauri_app_lib::models::embedding::text_cache::device_class(&device),
        opts.fixed_shape,
        per_text_s / batched_s,
        short_per_s / short_batch_s,
        single_per_ms,
        single_batch_ms,
        identical, texts.len(),
        single_identical, probe.len(), probe_identical, probe.len(), sub_identical, sub_idx.len(),
        batched_repro, texts.len(),
        bt_mean_s - pt_mean_s, bt_mean_l - pt_mean_l,
        agree_pb, query_idx.len(), agree_br, query_idx.len()
    );
    // LOGIS_EMBED_MEASURE_ONLY=1: 구성 무관성을 보장하지 않는 비교용 정책(예: Vulkan Auto 모드)의 측정 런 —
    // 무관성 수치는 위 요약 줄에 남기고 단언만 생략합니다(정확도 · 코사인 단언은 그대로).
    let measure_only = std::env::var("LOGIS_EMBED_MEASURE_ONLY").map(|v| v == "1").unwrap_or(false);
    if measure_only {
        println!("[GPU-EQ] 측정 전용 런(LOGIS_EMBED_MEASURE_ONLY=1): 구성 무관성 · 재현성 단언 생략");
    } else {
        assert_eq!(single_identical, probe.len(), "단건 호출 벡터가 배치 안의 벡터와 달라졌습니다(호출 구성 의존)");
        assert_eq!(probe_identical, probe.len(), "표본끼리 배치의 벡터가 전체 배치와 다릅니다(호출 구성 의존)");
        assert_eq!(sub_identical, sub_idx.len(), "역순 절반 배치의 벡터가 전체 배치와 다릅니다(호출 구성 의존)");
        assert_eq!(batched_repro, texts.len(), "같은 입력의 embed_many 는 실행마다 비트 단위로 같아야 합니다");
    }
    if device.is_cpu() && !opts.fixed_shape {
        // CPU 동적 배치(정확한 길이 묶음)는 per-text 와 같은 연산이라 모든 텍스트가 비트 동일해야 합니다.
        // (LOGIS_EMBED_FIXED_SHAPE=1 비교 런은 긴 텍스트를 패딩하므로 해당 없음 — macos-15 CPU 304/320)
        assert_eq!(identical, texts.len(), "CPU embed_many 가 per-text 와 비트 단위로 다릅니다");
    }
    assert!(min_cos > 0.995, "최소 코사인 {} — 배치 경로 수치가 허용 범위를 벗어났습니다", min_cos);
    // 배치 경로가 F32 참값에서 per-text 보다 눈에 띄게 멀어지면 안 됩니다.
    // 허용폭 2e-4 는 의도한 절충입니다: BF16 GEMM 은 모양마다 커널이 달라 배치 경로가 per-text 와
    // 비트 동일할 수 없고, 평균 코사인 차는 1e-4 수준(측정, CUDA)입니다. ROCm 도 기본 ON 이므로(models/embedding.rs)
    // AMD 에서는 이 테스트가 그 기본값의 실기기 근거입니다. 실패하면 LOGIS_EMBED_BATCHED=0 으로 되돌립니다.
    assert!(bt_mean_s >= pt_mean_s - 2e-4 && bt_mean_l >= pt_mean_l - 2e-4, "embed_many 의 F32 기준 정확도가 per-text 보다 낮습니다");
    assert!(batched.iter().all(|v| v.iter().any(|&x| x != 0.0)));
}

// ---------------------------------------------------------------------
//  판정 수준 일치: Stage-1 승자 · NMS 생존 스팬 · Plinko 슬롯 — per-text vs embed_many (LOGIS_TEST_MODELS=1)
// ---------------------------------------------------------------------

/// query_commerce 의 Stage-1(RAW-CHUNK 채점표 → 승자) + NMS(우선순위 → 점수 → 길이, 겹치면 탈락)를
/// 같은 규칙으로 돌려 (start, end, 승자 카테고리) 생존 목록을 돌려줍니다. `emb` 는 텍스트 → 벡터.
fn stage1_nms_decisions(
    lang: &str,
    query: &str,
    categories: &[&str],
    emb: &dyn Fn(&str) -> Vec<f32>,
) -> (Vec<(usize, usize, String)>, Vec<(usize, usize, String)>) {
    use tauri_app_lib::utils::ai_utils::{raw_chunk_spans, LayoutScoreTable};
    use tauri_app_lib::utils::parsing::get_multi_pass_contexts;
    // 레이아웃 뱅크 (build_layout 과 같은 키 규칙, 실제 임베더)
    let intent_anchors = [
        ("order", "manage order list, check order status"),
        ("goods", "product catalog data, items, find, search clothes"),
        ("tracking", "shipment status, tracking number, parcel"),
        ("review", "reviews, ratings, complaints"),
        ("coupon", "coupon codes, discount vouchers"),
        ("event", "marketing campaigns, promotions"),
    ];
    let mut anchor_embs = HashMap::new();
    for (cat, text) in intent_anchors {
        anchor_embs.insert(cat.to_string(), emb(text));
    }
    let mut layout_embs = HashMap::new();
    for cat in categories {
        for (key, bias, prej) in get_multi_pass_contexts(cat, lang) {
            let prej = if prej.trim().is_empty() { "random unrelated noise".to_string() } else { prej };
            layout_embs.insert(format!("{}_{}_bias", cat, key), emb(&bias));
            layout_embs.insert(format!("{}_{}_prejudice", cat, key), emb(&prej));
        }
    }
    let mut table = LayoutScoreTable::new();
    for cat in categories {
        table.ensure_cat(cat, get_multi_pass_contexts(cat, lang), &layout_embs, &anchor_embs);
    }
    let verb_emb = emb(&format!("{} verb, predicate, idiom", lang));
    let words: Vec<&str> = query.split_whitespace().collect();
    let mut winners = Vec::new();
    let mut evaluated: Vec<(usize, usize, String, String, f32)> = Vec::new();
    for (start, end, text) in raw_chunk_spans(&words) {
        let scores: HashMap<String, f32> = table.raw_chunk_scores(&emb(&text), &verb_emb, end - start).into_iter().collect();
        let (cat, s) = winner(&scores);
        winners.push((start, end, cat.clone()));
        if s > 0.0 {
            evaluated.push((start, end, text, cat, s));
        }
    }
    let priority = |c: &str| if matches!(c, "goods" | "order" | "tracking") { 1 } else { 2 };
    evaluated.sort_by(|a, b| {
        priority(&a.3)
            .cmp(&priority(&b.3))
            .then(b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal))
            .then(b.2.len().cmp(&a.2.len()))
    });
    let mut survivors: Vec<(usize, usize, String)> = Vec::new();
    for (s, e, _, cat, _) in evaluated {
        if !survivors.iter().any(|(ss, se, _)| s < *se && e > *ss) {
            survivors.push((s, e, cat));
        }
    }
    survivors.sort();
    (winners, survivors)
}

/// 인덱싱 PLINKO(nl_convert::plinko_game_for_indexing) 의 (청크, 확정 속성) 목록.
fn plinko_slots(emb: &(dyn Fn(&str) -> Vec<f32> + Sync)) -> Vec<(String, String)> {
    use tauri_app_lib::utils::ai_utils::split_bias_phrases_weighted;
    use tauri_app_lib::utils::nl_convert::{field_format_to_string, plinko_game_for_indexing, ChunkMetadata};
    let fields = tauri_app_lib::utils::bias_schema::get_list_schema_fields("goods", "", "ko");
    let names: Vec<String> = fields.iter().map(|f| f.0.clone()).collect();
    let formats: Vec<String> = names.iter().map(|n| field_format_to_string(n)).collect();
    let mut banks = Vec::new();
    let mut weights = Vec::new();
    for (_, _, bias_target, _) in &fields {
        let (phrases, w) = split_bias_phrases_weighted(bias_target);
        banks.push(phrases.iter().map(|p| emb(p)).collect::<Vec<_>>());
        weights.push(w);
    }
    let rows = [
        "여름 린넨 원피스 블루 M 사이즈 판매가 39,000원 할인율 20% 재고 12개",
        "무선 블루투스 이어폰 WE-PRO-2 정가 89,000원 무료배송 리뷰 4.8점 312개",
        "남성 반팔 티셔츠 화이트 L 상품코드 TS-2024-77 https://shop.example.com/p/77",
        "아동 운동화 230mm 핑크 품절 입고예정 다음주 월요일",
    ];
    let chunks: Vec<ChunkMetadata> = rows
        .iter()
        .map(|t| ChunkMetadata {
            chunk_text: t.to_string(),
            property: String::new(),
            property_format: String::new(),
            bias_phrases: Vec::new(),
            prejudice_phrases: Vec::new(),
            value_part: String::new(),
            confirmed: false,
        })
        .collect();
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let res = rt.block_on(plinko_game_for_indexing(&chunks, &names, &banks, &weights, &formats, |t: String| {
        std::future::ready(emb(&t))
    }));
    res.into_iter().map(|r| (r.chunk_text, r.property)).collect()
}

#[test]
#[ignore = "로컬 모델 필요: LOGIS_TEST_MODELS=1"]
fn embed_many_decisions_match_per_text() {
    if std::env::var("LOGIS_TEST_MODELS").map(|v| v != "1").unwrap_or(true) {
        println!("LOGIS_TEST_MODELS=1 이 아니라 건너뜁니다.");
        return;
    }
    let dir = match granite_dir() {
        Ok(d) => d,
        Err(e) => return crate::common::skip_or_fail(&e),
    };
    let mock = dir.join("MOCK").exists();
    let device = if std::env::var("LOGIS_TEST_EMBED_DEVICE").map(|v| v.trim().eq_ignore_ascii_case("cpu")).unwrap_or(false) {
        candle_core::Device::Cpu
    } else {
        match tauri_app_lib::utils::new_gpu_device(0) {
            Some(d) => d,
            None => {
                // 호스티드 러너처럼 GPU 가 없는 곳은 '못 함' 입니다(CI 는 측정 줄이 없으면 NOT RUN 으로 기록).
                println!("GPU 장치가 없어 건너뜁니다.");
                return;
            }
        }
    };
    use std::sync::Mutex;
    use tauri_app_lib::models::embedding::EmbeddingModel;
    let model = EmbeddingModel::new_with_device(&dir, &device).expect("load");
    let opts = model.batch_opts();
    // per-text: 텍스트마다 embed() (같은 텍스트는 같은 벡터라 메모해도 판정은 같습니다).
    // batched: 앱처럼 처음 보는 텍스트는 embed_many 로 구하고 캐시(첫 벡터 유지).
    let per_text_memo: Mutex<HashMap<String, Vec<f32>>> = Mutex::new(HashMap::new());
    let per_text = |t: &str| -> Vec<f32> {
        if let Some(v) = per_text_memo.lock().unwrap().get(t) {
            return v.clone();
        }
        let v = model.embed(t).unwrap();
        per_text_memo.lock().unwrap().insert(t.to_string(), v.clone());
        v
    };
    let cache: Mutex<HashMap<String, Vec<f32>>> = Mutex::new(HashMap::new());
    let batched = |t: &str| -> Vec<f32> {
        if let Some(v) = cache.lock().unwrap().get(t) {
            return v.clone();
        }
        let v = model.embed_many(&[t.to_string()], &opts).unwrap().pop().unwrap();
        cache.lock().unwrap().entry(t.to_string()).or_insert(v).clone()
    };
    let categories = ["order", "goods", "tracking", "review", "coupon", "event", ""];
    let queries = [
        ("ko", "여름 원피스 중에서 할인율 20%에 속하지만 재고가 남은 상품 찾아줘 빨리"),
        ("ko", "지난달 배송 지연된 주문 송장 번호 알려줘"),
        ("en", "show me summer dresses under 50 dollars with good reviews and free shipping"),
        ("ja", "先月 配送 遅延 注文 一覧 を 見せて"),
    ];
    let (mut w_same, mut w_total, mut nms_same, mut nms_total) = (0usize, 0usize, 0usize, 0usize);
    for (lang, q) in queries {
        // 배치 경로: 질의의 모든 스팬을 먼저 한 번에(앱의 RAW-CHUNK 프리패치와 같은 호출 구성) 넣습니다.
        let words: Vec<&str> = q.split_whitespace().collect();
        let spans: Vec<String> = tauri_app_lib::utils::ai_utils::raw_chunk_spans(&words).into_iter().map(|s| s.2).collect();
        for (t, v) in spans.iter().zip(model.embed_many(&spans, &opts).unwrap()) {
            cache.lock().unwrap().entry(t.clone()).or_insert(v);
        }
        let (wa, na) = stage1_nms_decisions(lang, q, &categories, &per_text);
        let (wb, nb) = stage1_nms_decisions(lang, q, &categories, &batched);
        w_total += wa.len();
        w_same += wa.iter().zip(wb.iter()).filter(|(a, b)| a == b).count();
        nms_total += 1;
        if na == nb {
            nms_same += 1;
        } else {
            println!("[DECISION-EQ] NMS 불일치 '{}': per-text {:?} vs embed_many {:?}", q, na, nb);
        }
    }
    let pa = plinko_slots(&per_text);
    let pb = plinko_slots(&batched);
    let p_same = pa.iter().zip(pb.iter()).filter(|(a, b)| a == b).count();
    if pa != pb {
        println!("[DECISION-EQ] Plinko per-text {:?}\n[DECISION-EQ] Plinko embed_many {:?}", pa, pb);
    }
    println!(
        "[DECISION-EQ] 모델 {} · 장치 {} · 정책 fixed_shape={} | Stage-1 승자 {}/{} · NMS 생존 목록 {}/{} 질의 · Plinko 슬롯 {} ({}/{} 청크)",
        if mock { "목업" } else { "실모델" },
        tauri_app_lib::models::embedding::text_cache::device_class(&device),
        opts.fixed_shape,
        w_same, w_total, nms_same, nms_total,
        if pa == pb { "일치" } else { "불일치" }, p_same, pa.len().max(pb.len())
    );
    if device.is_cpu() && !opts.fixed_shape {
        // CPU 동적 모드는 per-text 와 벡터가 비트 동일하므로 판정도 완전히 같아야 합니다.
        assert_eq!(w_same, w_total, "Stage-1 승자가 per-text 와 다릅니다");
        assert_eq!(nms_same, nms_total, "NMS 생존 목록이 per-text 와 다릅니다");
        assert_eq!(pa, pb, "Plinko 슬롯이 per-text 와 다릅니다");
    }
}

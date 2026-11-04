//! CLS → 예전 눈금 코사인 보정표를 다시 맞추는 절차 (models/embedding/calibration.rs)
//!
//! - CPU : 분위수 · 쌍 코사인 · 표 맞춤 · 지금 표의 단조성
//! - 실제 가중치 (ignore) : `refit_cls_to_legacy_cos` 가 granite 를 두 판본으로 올려 새 표를 Rust 소스로 찍고
//!   지금 표와의 최대 차이를 알려 줍니다. 표본은 `LOGIS_CALIB_TEXTS`(줄마다 텍스트 하나) 또는 bias.json 라벨 뱅크.
//!   `cargo test --test it -- refit_cls_to_legacy_cos --ignored --nocapture`

use tauri_app_lib::bias_schema::BIAS_DICT;
use tauri_app_lib::models::embedding::calibration::{
    bias_label_texts, fit_quantile_knots, max_knot_gap, pairwise_cosines, quantile_sorted, render_knots,
    CALIBRATION_PROBS,
};
use tauri_app_lib::models::embedding::{cls_to_legacy_knots, interp_knots};

fn assert_monotone(knots: &[(f32, f32)]) {
    assert_eq!(knots.first(), Some(&(-1.0, -1.0)), "{knots:?}");
    assert_eq!(knots.last(), Some(&(1.0, 1.0)), "{knots:?}");
    for w in knots.windows(2) {
        assert!(w[1].0 > w[0].0, "x 는 엄격히 증가해야 합니다: {w:?}");
        assert!(w[1].1 >= w[0].1, "y 는 줄지 않아야 합니다: {w:?}");
    }
}

/// 결정적인 의사 난수 (0..1).
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32) / (1u64 << 24) as f32
}

#[test]
fn quantile_matches_numpy_linear() {
    let s = [0.0, 1.0, 2.0, 3.0];
    assert_eq!(quantile_sorted(&s, 0.0), 0.0);
    assert_eq!(quantile_sorted(&s, 1.0), 3.0);
    assert_eq!(quantile_sorted(&s, 0.5), 1.5);
    assert!((quantile_sorted(&s, 1.0 / 3.0) - 1.0).abs() < 1e-6);
    assert!(quantile_sorted(&[], 0.5).is_nan());
}

#[test]
fn fit_recovers_a_known_monotone_map() {
    let mut seed = 7u64;
    let source: Vec<f32> = (0..20_000).map(|_| 0.6 + 0.4 * lcg(&mut seed)).collect();
    let target: Vec<f32> = source.iter().map(|x| 2.0 * x - 1.0).collect();
    let knots = fit_quantile_knots(source, target, &CALIBRATION_PROBS);
    assert_monotone(&knots);
    for &(x, y) in &knots[1..knots.len() - 1] {
        assert!((y - (2.0 * x - 1.0)).abs() < 1e-4, "({x}, {y}) 는 y = 2x − 1 위에 있어야 합니다");
    }
    // 두 판본이 같으면 표는 항등에 가깝습니다.
    let same: Vec<f32> = (0..5_000).map(|_| lcg(&mut seed) * 2.0 - 1.0).collect();
    let identity = fit_quantile_knots(same.clone(), same, &CALIBRATION_PROBS);
    for x in [-0.9f32, -0.3, 0.0, 0.4, 0.95] {
        assert!((interp_knots(x, &identity) - x).abs() < 1e-3, "{x}");
    }
}

#[test]
fn fit_drops_tied_knots_and_keeps_identity_at_one() {
    // 같은 값이 대부분이라 분위수가 겹칩니다. 겹친 칸은 버리고 1.0 은 1.0 으로 갑니다.
    let source = vec![0.8f32; 1_000].into_iter().chain([0.9, 1.0]).collect::<Vec<_>>();
    let target = vec![0.5f32; 1_000].into_iter().chain([0.7, 1.0]).collect::<Vec<_>>();
    let knots = fit_quantile_knots(source, target, &CALIBRATION_PROBS);
    assert_monotone(&knots);
    assert_eq!(interp_knots(1.0, &knots), 1.0);
    assert!(knots.len() < CALIBRATION_PROBS.len());
    // 표본이 없으면 항등 표입니다.
    assert_eq!(fit_quantile_knots(vec![], vec![f32::NAN], &CALIBRATION_PROBS), vec![(-1.0, -1.0), (1.0, 1.0)]);
}

#[test]
fn pairwise_cosines_counts_pairs_and_skips_zero_vectors() {
    let vectors = vec![vec![1.0, 0.0], vec![0.0, 2.0], vec![0.0, 0.0], vec![3.0, 3.0]];
    let mut cos = pairwise_cosines(&vectors);
    cos.sort_by(f32::total_cmp);
    let r = std::f32::consts::FRAC_1_SQRT_2;
    assert_eq!(cos.len(), 3, "0 벡터가 낀 쌍은 뺍니다");
    assert!((cos[0] - 0.0).abs() < 1e-6 && (cos[1] - r).abs() < 1e-6 && (cos[2] - r).abs() < 1e-6, "{cos:?}");
}

#[test]
fn current_table_is_monotone_and_renders() {
    let table = cls_to_legacy_knots();
    assert_eq!(table.len(), CALIBRATION_PROBS.len() + 2, "확률 25개 + 양 끝");
    assert_monotone(table);
    assert_eq!(max_knot_gap(table, table, 2_000), (0.0, -1.0));
    let src = render_knots("CLS_TO_LEGACY_COS", table);
    assert!(src.starts_with("const CLS_TO_LEGACY_COS: [(f32, f32); 27] = [\n"), "{src}");
    assert!(src.contains("    (0.7784, 0.5538),\n"), "{src}");
}

#[test]
fn label_bank_from_bias_is_deterministic_and_bounded() {
    let all = bias_label_texts(&BIAS_DICT, 0);
    assert!(all.len() > 500, "라벨 뱅크가 너무 작습니다: {}", all.len());
    assert!(all.windows(2).all(|w| w[0] < w[1]), "정렬 · 중복 없음");
    let picked = bias_label_texts(&BIAS_DICT, 300);
    assert_eq!(picked.len(), 300);
    assert_eq!(picked, bias_label_texts(&BIAS_DICT, 300));
    assert!(picked.iter().all(|t| all.binary_search(t).is_ok()));
}

#[test]
#[ignore = "실제 granite 가중치가 필요합니다 (LOGIS_MODELS_DIR 또는 앱 데이터 경로). --nocapture 로 새 표를 봅니다"]
fn refit_cls_to_legacy_cos() -> anyhow::Result<()> {
    use tauri_app_lib::models::embedding::{EmbeddingModel, ForwardMode};

    let dir = crate::common::model_subdir("granite-embedding-97m-multilingual-r2", "model.safetensors")
        .map_err(anyhow::Error::msg)?;
    let texts: Vec<String> = match std::env::var_os("LOGIS_CALIB_TEXTS").filter(|v| !v.is_empty()) {
        Some(path) => std::fs::read_to_string(&path)?
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(String::from)
            .collect(),
        None => bias_label_texts(&BIAS_DICT, 3_000),
    };
    anyhow::ensure!(texts.len() >= 100, "표본이 너무 적습니다: {}", texts.len());

    let device = candle_core::Device::Cpu;
    let mut cosines = Vec::new();
    for mode in [ForwardMode::Exact, ForwardMode::Legacy] {
        let started = std::time::Instant::now();
        let model = EmbeddingModel::new_with_device_and_mode(&dir, &device, mode)?;
        let vectors = texts.iter().map(|t| model.embed(t)).collect::<anyhow::Result<Vec<_>>>()?;
        let cos = pairwise_cosines(&vectors);
        let mean = cos.iter().map(|&c| c as f64).sum::<f64>() / cos.len() as f64;
        let sd = (cos.iter().map(|&c| (c as f64 - mean).powi(2)).sum::<f64>() / cos.len() as f64).sqrt();
        println!(
            "[CALIB] {:?}: 텍스트 {}개 · 쌍 {}개 · 평균 {:.3} · 표준편차 {:.3} ({:.1}s)",
            mode,
            texts.len(),
            cos.len(),
            mean,
            sd,
            started.elapsed().as_secs_f32()
        );
        cosines.push(cos);
    }
    let legacy = cosines.pop().expect("Legacy");
    let exact = cosines.pop().expect("Exact");
    let knots = fit_quantile_knots(exact, legacy, &CALIBRATION_PROBS);
    let (gap, at) = max_knot_gap(&knots, cls_to_legacy_knots(), 2_000);
    println!("{}", render_knots("CLS_TO_LEGACY_COS", &knots));
    println!("[CALIB] 지금 표와의 최대 차이 {:.4} (CLS 코사인 {:.3} 에서)", gap, at);
    assert_eq!(knots.first(), Some(&(-1.0, -1.0)));
    assert_eq!(knots.last(), Some(&(1.0, 1.0)));
    Ok(())
}

//! 🌟 [COS CALIBRATION FIT] CLS 코사인 → 예전 눈금 보정표를 다시 맞추는 절차
//!
//! `CLS_TO_LEGACY_COS` (models/embedding.rs) 는 같은 텍스트 쌍을 두 판본(예전 순전파 · CLS)으로 임베딩한 코사인의
//! 분위수를 짝지은 표입니다. 모델 가중치나 라벨 뱅크(bias.json)가 바뀌면 다시 맞춰야 한다고 적혀 있었지만 그 절차는
//! 코드에 없었습니다. 이 모듈이 그 절차이고, `tests/it/calibration.rs` 의 `refit_cls_to_legacy_cos`
//! (실제 가중치 · `--ignored`)가 새 표를 Rust 소스로 찍고 지금 표와의 차이를 알려 줍니다.

use serde_json::Value;

/// 지금 표를 만든 분위수 확률. 양 끝 (−1, −1) · (1, 1) 은 [`fit_quantile_knots`] 가 따로 더합니다.
pub const CALIBRATION_PROBS: [f64; 25] = [
    0.0, 1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05, //
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, //
    0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 0.99999, 1.0,
];

/// 정렬된 표본의 `p` 분위수 (numpy 기본값과 같은 선형 보간). 빈 표본이면 NaN.
pub fn quantile_sorted(sorted: &[f32], p: f64) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let h = (sorted.len() - 1) as f64 * p.clamp(0.0, 1.0);
    let (lo, hi) = (h.floor() as usize, h.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (h - lo as f64) as f32
}

/// 모든 쌍(i < j)의 코사인. 길이가 0 인 벡터가 낀 쌍은 뺍니다.
pub fn pairwise_cosines(vectors: &[Vec<f32>]) -> Vec<f32> {
    use rayon::prelude::*;
    let norms: Vec<f32> = vectors.iter().map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt()).collect();
    (0..vectors.len())
        .into_par_iter()
        .flat_map_iter(|i| {
            let (vi, ni, norms) = (&vectors[i], norms[i], &norms);
            ((i + 1)..vectors.len()).filter_map(move |j| {
                let denom = ni * norms[j];
                if denom <= 0.0 {
                    return None;
                }
                let dot: f32 = vi.iter().zip(&vectors[j]).map(|(a, b)| a * b).sum();
                Some((dot / denom).clamp(-1.0, 1.0))
            })
        })
        .collect()
}

/// 같은 텍스트 쌍의 두 판본 코사인에서 분위수 대응 표를 만듭니다 (`source` 의 p 분위수 → `target` 의 p 분위수).
///
/// 단조 함수라 순위 · 최근접 결과는 그대로이고, 각 문턱을 넘는 쌍의 비율이 `target` 판본과 같아집니다.
/// x 가 겹치는 칸은 첫 칸만 남기고(보간이 0 으로 나누지 않게), 양 끝 (−1, −1) · (1, 1) 을 더합니다.
/// 동일 텍스트의 코사인 1.0 은 언제나 1.0 으로 갑니다.
pub fn fit_quantile_knots(mut source: Vec<f32>, mut target: Vec<f32>, probs: &[f64]) -> Vec<(f32, f32)> {
    source.retain(|v| v.is_finite());
    target.retain(|v| v.is_finite());
    let mut knots = vec![(-1.0f32, -1.0f32)];
    if !source.is_empty() && !target.is_empty() {
        source.sort_by(f32::total_cmp);
        target.sort_by(f32::total_cmp);
        for &p in probs {
            let x = quantile_sorted(&source, p).clamp(-1.0, 1.0);
            let y = quantile_sorted(&target, p).clamp(-1.0, 1.0);
            let (last_x, last_y) = knots[knots.len() - 1];
            if x <= last_x {
                continue;
            }
            knots.push((x, y.max(last_y)));
        }
    }
    let last = knots.len() - 1;
    if knots[last].0 >= 1.0 {
        knots[last].1 = 1.0;
    } else {
        knots.push((1.0, 1.0));
    }
    knots
}

/// 표를 그대로 붙여 넣을 수 있는 Rust 상수 소스로 찍습니다.
pub fn render_knots(name: &str, knots: &[(f32, f32)]) -> String {
    let mut out = format!("const {}: [(f32, f32); {}] = [\n", name, knots.len());
    for (x, y) in knots {
        out.push_str(&format!("    ({:.4}, {:.4}),\n", x, y));
    }
    out.push_str("];\n");
    out
}

/// 두 표가 [−1, 1] 에서 가장 크게 다른 값과 그 위치 (`steps` 등분으로 잽니다).
pub fn max_knot_gap(a: &[(f32, f32)], b: &[(f32, f32)], steps: usize) -> (f32, f32) {
    let steps = steps.max(1);
    (0..=steps)
        .map(|i| {
            let x = -1.0 + 2.0 * i as f32 / steps as f32;
            ((super::interp_knots(x, a) - super::interp_knots(x, b)).abs(), x)
        })
        .fold((0.0, -1.0), |best, cur| if cur.0 > best.0 { cur } else { best })
}

/// 사전(bias.json)의 문자열 값을 겹치지 않게 모은 라벨 뱅크. 보정표를 다시 맞출 때의 기본 표본입니다.
/// 2자 미만이거나 200자를 넘는 값은 뺍니다. `limit` 를 넘으면 정렬 순서에서 고르게 솎습니다 (실행마다 같습니다).
pub fn bias_label_texts(dict: &Value, limit: usize) -> Vec<String> {
    fn walk(v: &Value, out: &mut std::collections::BTreeSet<String>) {
        match v {
            Value::String(s) => {
                let s = s.trim();
                let n = s.chars().count();
                if (2..=200).contains(&n) {
                    out.insert(s.to_string());
                }
            }
            Value::Array(items) => items.iter().for_each(|x| walk(x, out)),
            Value::Object(map) => map.values().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let mut set = std::collections::BTreeSet::new();
    walk(dict, &mut set);
    let all: Vec<String> = set.into_iter().collect();
    if limit == 0 || all.len() <= limit {
        return all;
    }
    (0..limit).map(|i| all[i * all.len() / limit].clone()).collect()
}

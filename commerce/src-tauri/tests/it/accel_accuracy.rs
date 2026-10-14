//! FlashAttention / TurboQuant 정확도 — teacher-forcing (docs/mac-verify.md 3-5, 목업 Qwen3)
//!
//! 0d3ef235 ('정확도 저하 이슈 있음') → f54732e9 디버깅 이후 앱 단에서 해결 여부를 확인한 기록이 없어,
//! 같은 토큰열을 FA on/off × TurboQuant KV on/off 로 teacher-forcing 하고 CPU F32 · FA off 기준과 비교합니다.
//!  - 모델: 시드 고정 무작위 가중치 Qwen3 (head_dim 64 · GQA 4/2 · 3층 · vocab 512) — **목업**.
//!  - 순서: 프리필 64 토큰(q_len > 1, FA 프리필 경로) → 1 토큰씩 디코드 96 회(FA 디코드 · TurboQuant 압축 KV).
//!  - 지표: 위치별 KLD(기준 ‖ 측정), top-1 일치율, max |Δlogit| (프리필 마지막 + 디코드 96 = 97 위치).
//!  - 장치: CPU + 이 빌드의 GPU 0번(`utils::new_gpu_device`). TurboQuant KV 는 CUDA/ROCm/Vulkan 에서만
//!    켜집니다(`turbo_kv_active`). Metal 의 FA 는 포크의 Metal sdpa 융합 커널(포크 패치 0004)이며, 첫 사용 때
//!    `accel_check` 자가 점검을 통과해야 켜집니다([ACCEL-CHECK] 줄).
//! 판정: FA 는 F32 에서 무손실이어야 하므로 같은 장치 FA off 대비 max |Δlogit| ≤ 1e-3, 기준 대비 top-1 ≥ 95%.
//! TurboQuant 는 손실 압축이라 측정값을 기록하고 느슨한 상한(평균 KLD ≤ 0.05 · top-1 ≥ 80%)만 봅니다.
#![cfg(any(feature = "flash-attn", feature = "turbo-quant"))]

use std::collections::HashMap;

use crate::common::{det, mock_qwen3, MOCK_QWEN3_VOCAB as VOCAB};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tauri_app_lib::models::qwen3::config::Qwen3Config;
use tauri_app_lib::models::qwen3::model::Qwen3Model;
use tauri_app_lib::utils::resources::KvResidency;

type R = anyhow::Result<()>;

const PREFILL: usize = 64;
const DECODE: usize = 96;

fn tokens() -> Vec<u32> {
    det(PREFILL + DECODE, 77).into_iter().map(|x| ((x.abs() * 1e4) as u32) % VOCAB as u32).collect()
}

/// teacher-forcing 로짓 DECODE 개: 프리필 마지막 위치 1 + 디코드 DECODE-1 스텝.
fn run(cfg: &Qwen3Config, w: &HashMap<String, Tensor>, dev: &Device, fa: bool, tq: bool) -> anyhow::Result<Vec<Vec<f32>>> {
    candle_core::accel::update_config(|c| {
        c.flash_attn.enabled = fa;
        c.turbo_quant.enabled = tq;
    });
    let ids = tokens();
    let mut m = Qwen3Model::new(cfg, VarBuilder::from_tensors(w.clone(), DType::F32, dev))?;
    m.set_kv_residency(KvResidency::Vram);
    let logits = |t: Tensor| -> anyhow::Result<Vec<f32>> {
        Ok(t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
    };
    let mut out = Vec::with_capacity(DECODE);
    let pre = Tensor::new(&ids[..PREFILL], dev)?.unsqueeze(0)?;
    out.push(logits(m.forward(Some(&pre), None, 0)?)?);
    for i in 0..DECODE - 1 {
        let tok = Tensor::new(&ids[PREFILL + i..PREFILL + i + 1], dev)?.unsqueeze(0)?;
        out.push(logits(m.forward(Some(&tok), None, PREFILL + i)?)?);
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy)]
struct Cmp {
    mean_kld: f64,
    max_kld: f64,
    top1: f64,
    max_abs: f32,
}

fn compare(reference: &[Vec<f32>], got: &[Vec<f32>]) -> Cmp {
    let softmax = |v: &[f32]| -> Vec<f64> {
        let mx = v.iter().fold(f32::NEG_INFINITY, |m, x| m.max(*x)) as f64;
        let e: Vec<f64> = v.iter().map(|x| (*x as f64 - mx).exp()).collect();
        let s: f64 = e.iter().sum();
        e.into_iter().map(|x| x / s).collect()
    };
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0;
    let (mut sum, mut mx, mut agree, mut max_abs) = (0f64, 0f64, 0usize, 0f32);
    for (r, g) in reference.iter().zip(got) {
        let (p, q) = (softmax(r), softmax(g));
        let kld: f64 = p.iter().zip(&q).map(|(p, q)| if *p > 0.0 { p * (p / q.max(1e-300)).ln() } else { 0.0 }).sum();
        sum += kld;
        mx = mx.max(kld);
        if argmax(r) == argmax(g) {
            agree += 1;
        }
        max_abs = r.iter().zip(g).fold(max_abs, |m, (a, b)| m.max((a - b).abs()));
    }
    let n = reference.len().max(1) as f64;
    Cmp { mean_kld: sum / n, max_kld: mx, top1: agree as f64 / n, max_abs }
}

/// 전역 accel 설정(FA · TurboQuant on/off)을 바꾸므로 병렬 suite 에서는 빠지고 단독으로 돌립니다
/// (`cargo test --test it --features …,flash-attn,turbo-quant -- accel_accuracy --ignored --nocapture --test-threads 1`).
#[test]
#[ignore]
fn flash_attn_and_turbo_quant_teacher_forcing_accuracy() -> R {
    let saved = candle_core::accel::config();
    let res = (|| -> R {
        let (cfg, w) = mock_qwen3(256, 4, 64, 3)?;
        let reference = run(&cfg, &w, &Device::Cpu, false, false)?;
        assert_eq!(reference.len(), DECODE);
        assert!(reference.iter().flatten().all(|x| x.is_finite()));
        let mut devices = vec![Device::Cpu];
        if let Some(d) = tauri_app_lib::utils::new_gpu_device(0) {
            devices.push(d);
        }
        for dev in &devices {
            let tq_active = tauri_app_lib::models::common::turbo_kv_active(dev);
            let label = format!("{:?}", dev.location());
            let base = run(&cfg, &w, dev, false, false)?;
            let mut results = Vec::new();
            for (fa, tq) in [(false, false), (true, false), (false, true), (true, true)] {
                let got = if (fa, tq) == (false, false) { base.clone() } else { run(&cfg, &w, dev, fa, tq)? };
                assert!(got.iter().flatten().all(|x| x.is_finite()), "{label} fa={fa} tq={tq}: non-finite logits");
                let vs_ref = compare(&reference, &got);
                let vs_base = compare(&base, &got);
                println!(
                    "[ACCEL-ACC] 목업 Qwen3 · {label} · FA {} · TurboQuant KV {}{} | CPU F32 기준: 평균 KLD {:.2e} · 최대 KLD {:.2e} · top-1 {:.1}% · max|Δlogit| {:.2e} | 같은 장치 FA/TQ off 대비 max|Δlogit| {:.2e} · top-1 {:.1}%",
                    if fa { "on" } else { "off" },
                    if tq { "on" } else { "off" },
                    if tq && !tq_active { " (이 장치에서는 비활성)" } else { "" },
                    vs_ref.mean_kld, vs_ref.max_kld, vs_ref.top1 * 100.0, vs_ref.max_abs, vs_base.max_abs, vs_base.top1 * 100.0
                );
                results.push(((fa, tq), vs_ref, vs_base));
            }
            for ((fa, tq), vs_ref, vs_base) in results {
                let lossy_tq = tq && tq_active;
                if !lossy_tq {
                    // FA(또는 비활성 TQ) 는 F32 에서 무손실이어야 합니다 (합산 순서 차이 수준).
                    if fa && !(dev.is_cuda() || dev.is_rocm()) {
                        assert!(vs_base.max_abs <= 1e-3, "{label}: FA on 이 FA off 와 {:.2e} 다릅니다", vs_base.max_abs);
                    }
                    assert!(vs_ref.top1 >= 0.95, "{label} fa={fa}: CPU F32 기준 top-1 {:.1}%", vs_ref.top1 * 100.0);
                } else {
                    assert!(
                        vs_ref.mean_kld <= 0.05 && vs_ref.top1 >= 0.80,
                        "{label} TurboQuant KV fa={fa}: 평균 KLD {:.2e} · top-1 {:.1}%",
                        vs_ref.mean_kld,
                        vs_ref.top1 * 100.0
                    );
                }
            }
        }
        Ok(())
    })();
    candle_core::accel::set_config(saved);
    res
}

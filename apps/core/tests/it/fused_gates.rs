//! Part 5 & Part 6 Cross-Repo QA Gates:
//! - Gate 5-A: `accel_check` 5-op fault injection (ReturnError / Panic / BadOutput) -> `catch_unwind` recovery & valid logits.
//! - Gate 5-B: Mock Qwen3 + Qwen4-Exp (GDN + QSA + SparseMoE) teacher-forcing parity (`LOGIS_FUSED_OPS=off` vs `on`) + `QSA` exact selection mask.
//! - Gate 1-C + Part 5 Commit 3: `TrimmedHead` + `RestRescueGuard` 100% lossless greedy parity, host SIMD rescue, `EmbedTable` byte-identity, and `greedy_with_shadow_top3` + `SmallVec` anchor matcher.

use std::sync::Arc;

use anyhow::Result;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;

use logis_app::models::common::accel_check::{
    diagnose, fused_op_ok, reset_fused_op_cache, set_fused_op_fault, set_fused_ops_override,
    FaultMode, FusedOpKind, FusedOpsOverride,
};
use logis_app::models::common::vocab_trim::{
    match_shadow_anchor_candidate, GreedyShadowStep, ShadowCandidate, TableSource, TrimmedHead,
};
use logis_app::models::qwen3::model::Qwen3Model;
use logis_app::models::qwen3_5::model::EmbedTable;
use logis_app::utils::resources::KvResidency;

use crate::common::{det, mock_qwen3, TempDir, MOCK_QWEN3_VOCAB};

struct AccelGuard;
impl Drop for AccelGuard {
    fn drop(&mut self) {
        for k in FusedOpKind::ALL {
            set_fused_op_fault(k, None);
        }
        set_fused_ops_override(None);
        reset_fused_op_cache();
    }
}

fn gpu_device_opt() -> Option<Device> {
    let d = logis_app::utils::new_gpu_device(0)?;
    if d.is_cuda() || d.is_rocm() || d.is_vulkan() {
        Some(d)
    } else {
        None
    }
}

#[test]
fn fused_ops_fault_injection_fallback() -> Result<()> {
    let _guard = AccelGuard;
    let require_gpu = std::env::var("LOGIS_REQUIRE_GPU").map(|v| v == "1").unwrap_or(false);
    let devices: Vec<Device> = match gpu_device_opt() {
        Some(g) => vec![Device::Cpu, g],
        None => {
            if require_gpu {
                anyhow::bail!("LOGIS_REQUIRE_GPU=1 set, but no CUDA/ROCm/Vulkan GPU available");
            }
            vec![Device::Cpu]
        }
    };

    for dev in &devices {
        let dt = if dev.is_cuda() || dev.is_rocm() { DType::BF16 } else { DType::F32 };
        for mode in [FaultMode::ReturnError, FaultMode::Panic, FaultMode::BadOutput] {
            for kind in FusedOpKind::ALL {
                reset_fused_op_cache();
                set_fused_ops_override(Some(FusedOpsOverride::AllOn));
                set_fused_op_fault(kind, Some(mode));

                // 1) Self-check MUST NOT abort; it must catch the error/panic and return false
                let ok = fused_op_ok(dev, kind, dt, 64);
                assert!(
                    !ok,
                    "fault-injected {kind:?} ({mode:?}) on {dev:?} must disable fused path"
                );

                // 2) Negative result must stay cached even after clearing the fault until cache reset
                set_fused_op_fault(kind, None);
                assert!(
                    !fused_op_ok(dev, kind, dt, 64),
                    "failed self-check for {kind:?} must remain cached as false"
                );
            }
        }
    }

    // 3) Run a full mock Qwen3 forward pass while ALL 5 fused ops are fault-injected (Panic)
    //    and verify logits are finite and match FusedOpsOverride::AllOff.
    let (cfg, w) = mock_qwen3(128, 4, 32, 2)?;
    let run_dev = gpu_device_opt().unwrap_or(Device::Cpu);
    let ids = Tensor::new(&[[1u32, 5, 9, 13, 21, 34]], &run_dev)?;

    reset_fused_op_cache();
    set_fused_ops_override(Some(FusedOpsOverride::AllOff));
    let mut m_off = Qwen3Model::new(&cfg, VarBuilder::from_tensors(w.clone(), DType::F32, &run_dev))?;
    m_off.set_kv_residency(KvResidency::Vram);
    let ref_logits = m_off
        .forward(Some(&ids), None, 0)?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    reset_fused_op_cache();
    set_fused_ops_override(Some(FusedOpsOverride::AllOn));
    for kind in FusedOpKind::ALL {
        set_fused_op_fault(kind, Some(FaultMode::Panic));
    }
    let mut m_fault = Qwen3Model::new(&cfg, VarBuilder::from_tensors(w, DType::F32, &run_dev))?;
    m_fault.set_kv_residency(KvResidency::Vram);
    let got_logits = m_fault
        .forward(Some(&ids), None, 0)?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    assert!(got_logits.iter().all(|x| x.is_finite()), "fallback logits must be finite");
    let max_diff = ref_logits
        .iter()
        .zip(&got_logits)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    println!(
        "[GATE 5-A] fault-injection fallback on {run_dev:?}: max|Δlogit| = {max_diff:.2e} (0 aborts)"
    );
    assert!(max_diff < 1e-4, "fault-injected fallback diverged from AllOff: {max_diff}");
    Ok(())
}

#[test]
fn qwen_family_fused_teacher_forcing_parity() -> Result<()> {
    let _guard = AccelGuard;
    let require_gpu = std::env::var("LOGIS_REQUIRE_GPU").map(|v| v == "1").unwrap_or(false);
    let dev = match gpu_device_opt() {
        Some(d) => d,
        None => {
            if require_gpu {
                anyhow::bail!("LOGIS_REQUIRE_GPU=1 set, but no CUDA/ROCm/Vulkan GPU available");
            }
            println!("[GATE 5-B] No CUDA/ROCm/Vulkan GPU built/available; running CPU diagnose check only.");
            let diag = diagnose(&Device::Cpu, DType::F32, 64);
            assert!(diag.contains("fused-ops:"), "diagnose missing fused-ops summary: {diag}");
            return Ok(());
        }
    };

    // 1) Diagnose reports all 5 fused ops cleanly on live GPU
    let diag_dt = if dev.is_vulkan() { DType::F32 } else { DType::BF16 };
    let diag = diagnose(&dev, diag_dt, 64);
    println!("[GATE 5-B] diagnose({dev:?}): {diag}");
    assert!(diag.contains("fused-ops:"), "diagnose missing fused-ops summary: {diag}");
    for kind in FusedOpKind::ALL {
        assert!(
            diag.contains(kind.token()) && !diag.contains(&format!("{}=ERR", kind.token())) && !diag.contains(&format!("{}=PANIC", kind.token())),
            "diagnose failed for {}: {diag}",
            kind.token()
        );
    }

    // 2) Mock Qwen3 teacher-forcing parity (prefill 32 + decode 32) + Mode A greedy parity
    let (cfg, w) = mock_qwen3(256, 4, 64, 3)?;
    let seq: Vec<u32> = det(64, 909)
        .into_iter()
        .map(|x| ((x.abs() * 1e4) as u32) % MOCK_QWEN3_VOCAB as u32)
        .collect();

    // Note: Qwen3 `eager_attention_forward` downcasts Q/K/V to BF16 on CUDA/ROCm, so a 1-ULP
    // F32 difference before a BF16 midpoint produces ~1e-2 logit delta across 3 layers while
    // keeping KLD ~1e-6 and top-1 == 100%. We verify mean KLD, top-1 agreement, and exact Mode-A greedy.
    let cases: &[(DType, f32, f64)] = if dev.is_vulkan() {
        &[(DType::F32, 5e-2f32, 1e-6f64)]
    } else {
        &[(DType::F32, 5e-2f32, 1e-6f64), (DType::BF16, 1.0f32, 1e-4f64)]
    };
    for &(model_dt, max_tol, kld_tol) in cases {
        let w_dt: std::collections::HashMap<String, Tensor> = w
            .iter()
            .map(|(k, v)| Ok((k.clone(), v.to_dtype(model_dt)?.to_device(&dev)?)))
            .collect::<Result<_>>()?;
        let run_qwen3 = |ov: FusedOpsOverride| -> Result<(Vec<Vec<f32>>, Vec<u32>)> {
            reset_fused_op_cache();
            set_fused_ops_override(Some(ov));
            let mut m = Qwen3Model::new(&cfg, VarBuilder::from_tensors(w_dt.clone(), model_dt, &dev))?;
            m.set_kv_residency(KvResidency::Vram);
            m.set_kv_exact(true);
            let mut all_logits = Vec::new();
            let mut greedy_ids = Vec::new();
            let pre = Tensor::new(&seq[..32], &dev)?.unsqueeze(0)?;
            let h0 = m.forward_last_hidden(Some(&pre), None, 0)?;
            greedy_ids.push(m.greedy_from_hidden(&h0)?);
            all_logits.push(
                m.logits_from_hidden(&h0)?
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
            for i in 0..32 {
                let tok = Tensor::new(&seq[32 + i..32 + i + 1], &dev)?.unsqueeze(0)?;
                let hi = m.forward_last_hidden(Some(&tok), None, 32 + i)?;
                greedy_ids.push(m.greedy_from_hidden(&hi)?);
                all_logits.push(
                    m.logits_from_hidden(&hi)?
                        .to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                );
            }
            Ok((all_logits, greedy_ids))
        };

        let (off_logits, off_greedy) = run_qwen3(FusedOpsOverride::AllOff)?;
        let (on_logits, on_greedy) = run_qwen3(FusedOpsOverride::AllOn)?;
        let softmax = |v: &[f32]| -> Vec<f64> {
            let mx = v.iter().fold(f32::NEG_INFINITY, |m, x| m.max(*x)) as f64;
            let e: Vec<f64> = v.iter().map(|x| (*x as f64 - mx).exp()).collect();
            let s: f64 = e.iter().sum();
            e.into_iter().map(|x| x / s).collect()
        };
        let mut max_abs = 0.0f32;
        let mut argmax_agree = 0usize;
        let mut kld_sum = 0.0f64;
        for (r, g) in off_logits.iter().zip(&on_logits) {
            let (p, q) = (softmax(r), softmax(g));
            kld_sum += p
                .iter()
                .zip(&q)
                .map(|(pv, qv)| if *pv > 0.0 { pv * (pv / qv.max(1e-300)).ln() } else { 0.0 })
                .sum::<f64>();
            let ra = r.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
            let ga = g.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
            if ra == ga {
                argmax_agree += 1;
            }
            for (a, b) in r.iter().zip(g) {
                max_abs = max_abs.max((a - b).abs());
            }
        }
        let mean_kld = kld_sum / off_logits.len().max(1) as f64;
        let greedy_exact = on_greedy.iter().zip(&on_logits).all(|(&gid, lv)| {
            let la = lv.iter().enumerate().fold((0u32, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i as u32, v) } else { b }).0;
            gid == la
        });
        println!(
            "[GATE 5-B] Qwen3 fused vs unfused ({dev:?} {model_dt:?}): mean_KLD={mean_kld:.2e}, max|Δlogit|={max_abs:.2e}, top-1={}/{}, Mode-A greedy exact={greedy_exact}",
            argmax_agree,
            off_logits.len()
        );
        assert!(greedy_exact, "Mode A fused_linear_argmax_cs must match full logits argmax 100% ({model_dt:?})");
        assert!(mean_kld <= kld_tol, "Qwen3 {model_dt:?} mean_KLD={mean_kld} > {kld_tol}");
        assert!(max_abs < max_tol, "Qwen3 {model_dt:?} fused vs unfused max|Δlogit|={max_abs} >= {max_tol}");
        assert!(argmax_agree * 100 >= off_logits.len() * 95, "Qwen3 {model_dt:?} top-1={argmax_agree}/{}", off_logits.len());
        assert_eq!(off_greedy.len(), on_greedy.len());
    }

    // 3) Qwen4-Exp tiny fixture (GDN + QSA + SparseMoE): fused vs unfused + QSA selection mask exact match
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4_exp_tiny");
    if fixture.join("config.json").exists() {
        let ids_t = Tensor::new(&[[1u32, 7, 13, 21, 5, 9, 17, 29]], &dev)?;
        let run_q4 = |ov: FusedOpsOverride| -> Result<(Vec<f32>, u32, Vec<String>)> {
            reset_fused_op_cache();
            set_fused_ops_override(Some(ov));
            let mut m = logis_app::models::qwen4_exp::model::Qwen4ExpForCausalLM::load(&fixture, DType::F32, &dev)?;
            for l in &mut m.layers {
                if let logis_app::models::qwen4_exp::model::Mixer::Qsa(a) = &mut l.mixer {
                    a.record_selection = true;
                }
            }
            let h = m.forward_last_hidden(&ids_t, 0)?;
            let gid = m.lm_head_greedy_token(&h)?;
            let logits = m.lm_head_forward(&h)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            let sels: Vec<String> = m
                .layers
                .iter()
                .filter_map(|l| match &l.mixer {
                    logis_app::models::qwen4_exp::model::Mixer::Qsa(a) => Some(format!("{:?}", a.last_selection)),
                    _ => None,
                })
                .collect();
            Ok((logits, gid, sels))
        };
        let (q4_off, q4_gid_off, sel_off) = run_q4(FusedOpsOverride::AllOff)?;
        let (q4_on, q4_gid_on, sel_on) = run_q4(FusedOpsOverride::AllOn)?;
        let q4_diff = q4_off.iter().zip(&q4_on).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        println!(
            "[GATE 5-B] Qwen4-Exp (GDN+QSA+MoE) fused vs unfused ({dev:?} F32): max|Δlogit|={q4_diff:.2e}, greedy={q4_gid_off}=={q4_gid_on}, QSA masks identical={}",
            sel_off == sel_on
        );
        assert!(q4_diff < 5e-4, "Qwen4-Exp F32 fused vs unfused max|Δlogit|={q4_diff}");
        assert_eq!(q4_gid_off, q4_gid_on, "Qwen4-Exp greedy token mismatch");
        assert_eq!(sel_off, sel_on, "QSA selection masks must be identical");
    }
    Ok(())
}

#[test]
fn trimmed_head_lossless_cs_guard_and_shadow_top3() -> Result<()> {
    let _guard = AccelGuard;
    let dev = gpu_device_opt().unwrap_or(Device::Cpu);
    let (vocab, cols, m) = (512usize, 128usize, 16usize);

    // Construct realistic Q8_0 embedding table with positive mean shift + adversarial REST spikes
    let mut w_data = det(vocab * cols, 701);
    for v in &mut w_data {
        *v = *v * 0.12 + 0.25;
    }
    let kept: Vec<u32> = (0..128u32).map(|i| i * 3).collect(); // 128 non-contiguous CORE rows
    let rest_winner = 311usize; // 311 % 3 != 0 -> in REST
    assert!(!kept.contains(&(rest_winner as u32)));
    for c in 0..cols {
        w_data[rest_winner * cols + c] = 0.25 + if (c / 8) % 2 == 0 { 0.15 } else { -0.15 };
    }
    let mut h_data = vec![0.0f32; m * cols];
    for r in 0..m {
        if r < m / 2 {
            let target = kept[r + 4] as usize;
            let phase = (r + 1) as f32 * 1.37;
            for c in 0..cols {
                let d = ((c as f32) * phase + 0.5).sin() * 0.44;
                w_data[target * cols + c] = 0.25 + d;
                h_data[r * cols + c] = d * 4.0;
            }
        } else {
            for c in 0..cols {
                h_data[r * cols + c] = (w_data[rest_winner * cols + c] - 0.25) * 6.0;
            }
        }
    }

    let w_t = Tensor::from_vec(w_data, (vocab, cols), &Device::Cpu)?;
    let q_full_cpu = QTensor::quantize(&w_t, GgmlDType::Q8_0)?;
    let raw_bytes = q_full_cpu.data()?.to_vec();

    let dir = TempDir::new("logis_trim_cs_guard");
    let path = dir.0.join("embd.q8");
    std::fs::write(&path, &raw_bytes)?;
    let map = Arc::new(unsafe { memmap2::Mmap::map(&std::fs::File::open(&path)?)? });
    let src = TableSource { map, start: 0, rows: vocab, cols, dtype: GgmlDType::Q8_0 };

    let head = TrimmedHead::build(&src, Arc::new(kept.clone()), "test:en".into(), &dev)?;
    let h_dev = Tensor::from_vec(h_data, (m, cols), &Device::Cpu)?.to_device(&dev)?;

    // Ground truth full-vocab Q8_0 argmax (computed on CPU from exact dequantized rows)
    let full_logits = QMatMul::from_qtensor(q_full_cpu)?
        .forward(&h_dev.to_device(&Device::Cpu)?)?
        .to_vec2::<f32>()?;
    let expected: Vec<u32> = full_logits
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .fold((0u32, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i as u32, v) } else { (bi, bv) }
                })
                .0
        })
        .collect();

    let (got_ids, rescued) = head.greedy_next_token_with_stats(&h_dev)?;
    println!(
        "[GATE VOCAB-TRIM] ({dev:?}) vocab={vocab} kept={} | greedy match={}/{} | rescued={rescued}/{m}",
        kept.len(),
        got_ids.iter().zip(&expected).filter(|(a, b)| a == b).count(),
        m
    );
    assert_eq!(got_ids, expected, "TrimmedHead + RestRescueGuard MUST match full-vocab greedy 100%");
    // Also exercise M <= 8 decode batch (triggers Quantized Mode A fused_quant_argmax_cs on ROCm)
    let (got_dec, rescued_dec) = head.greedy_next_token_with_stats(&h_dev.narrow(0, m / 2 - 2, 4)?)?;
    assert_eq!(got_dec, expected[m / 2 - 2..m / 2 + 2], "Quantized Mode A M=4 decode must match 100%");
    assert!(rescued_dec >= 2, "expected >= 2 REST rescues in M=4 mixed batch, got {rescued_dec}");
    for r in (m / 2)..m {
        assert_eq!(got_ids[r], rest_winner as u32, "adversarial REST row {r} must win rest_winner");
    }
    assert!(rescued >= m / 2 && rescued < m, "expected CORE bypass + REST rescue, got rescued={rescued}/{m}");

    // EmbedTable byte-identity check (unchanged contract)
    let table = EmbedTable::Mapped { src: src.clone(), device: dev.clone() };
    let sample_ids = Tensor::new(&[[kept[0], rest_winner as u32, kept[7], 1u32]], &dev)?;
    let emb_via_head = table.forward_with_head(&sample_ids, Some(&head))?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let emb_plain = table.forward(&sample_ids)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(emb_via_head, emb_plain, "EmbedTable::forward_with_head must be byte-identical");

    // Gate 1-C: greedy_with_shadow_top3 + SmallVec<[u8; 16]> subword anchor rescue
    let step_core = head.greedy_with_shadow_top3(&h_dev.narrow(0, 0, 1)?)?;
    assert_eq!(step_core.token_id, expected[0]);
    assert!(!step_core.rescued);
    assert!(step_core.top3[0].is_some() && step_core.top3[1].is_some() && step_core.top3[2].is_some());

    let step_rest = head.greedy_with_shadow_top3(&h_dev.narrow(0, m / 2, 1)?)?;
    assert_eq!(step_rest.token_id, rest_winner as u32);
    assert!(step_rest.rescued);

    // Build tiny WordLevel tokenizer for zero-alloc SmallVec shadow matcher test
    let mut vocab_map = serde_json::Map::new();
    for (i, tok) in ["[UNK]", "INV-001", "INV-002", "TOTAL"].iter().enumerate() {
        vocab_map.insert((*tok).into(), i.into());
    }
    let tok_json = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab_map, "unk_token": "[UNK]"}
    });
    let tok = tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&tok_json)?).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Low confidence (0.62 < 0.95) + alt prob 0.25 >= 0.08 + edit distance 0 vs 1 -> picks anchor token 2 ("INV-002")
    let low_conf_step = GreedyShadowStep {
        token_id: 1, // "INV-001"
        confidence: 0.62,
        top3: [
            Some(ShadowCandidate { token_id: 1, logit: 4.0, prob: 0.62 }),
            Some(ShadowCandidate { token_id: 2, logit: 3.1, prob: 0.25 }),
            Some(ShadowCandidate { token_id: 3, logit: 1.0, prob: 0.03 }),
        ],
        rescued: false,
    };
    assert_eq!(match_shadow_anchor_candidate(&low_conf_step, &tok, b"INV-002"), Some(2));
    // High confidence (>= 0.95) -> never overridden
    let high_conf_step = GreedyShadowStep { confidence: 0.97, ..low_conf_step.clone() };
    assert_eq!(match_shadow_anchor_candidate(&high_conf_step, &tok, b"INV-002"), None);
    // Low alt prob (< 0.08) -> ignored
    let low_alt_step = GreedyShadowStep {
        top3: [
            Some(ShadowCandidate { token_id: 1, logit: 4.0, prob: 0.85 }),
            Some(ShadowCandidate { token_id: 2, logit: 1.2, prob: 0.05 }),
            None,
        ],
        ..low_conf_step
    };
    assert_eq!(match_shadow_anchor_candidate(&low_alt_step, &tok, b"INV-002"), None);

    drop(head);
    drop(table);
    drop(src);
    Ok(())
}

/// Gate 4-B (`P4-F02` / `P4-F01` / `P4-O01`): Verifies `OffloadPrefetcher` performs real transfers
/// on both small (<4 MiB sync) and large (>=4 MiB async DMA on ROCm/CUDA) payloads with exact bit
/// parity against direct `to_device`.
#[test]
fn fused_gates_offload_prefetcher_parity() -> Result<()> {
    use logis_app::utils::prefetch::{OffloadPrefetcher, ASYNC_DMA_MIN_BYTES};
    let dev = gpu_device_opt().unwrap_or(Device::Cpu);
    // Test small sync path (<4 MiB) + large async-capable path (4 MiB on ROCm/CUDA)
    let cap = if dev.is_cuda() || dev.is_rocm() {
        ASYNC_DMA_MIN_BYTES
    } else {
        64 * 1024
    };
    let mut prefetcher = OffloadPrefetcher::new(&dev, cap)?;

    // 1) Small payload: must NOT be a silent no-op (P4-F02)
    let k_small = Tensor::from_vec(det(1 * 2 * 16 * 32, 701), (1, 2, 16, 32), &Device::Cpu)?;
    let v_small = Tensor::from_vec(det(1 * 2 * 16 * 32, 702), (1, 2, 16, 32), &Device::Cpu)?;
    let pf_small = prefetcher.prefetch_kv_pair(&k_small, &v_small)?;
    let (ks_dev, vs_dev) = pf_small.wait_and_take(&mut prefetcher)?;
    assert_eq!(
        ks_dev.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?,
        k_small.flatten_all()?.to_vec1::<f32>()?,
        "small prefetch K must match host bit-for-bit"
    );
    assert_eq!(
        vs_dev.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?,
        v_small.flatten_all()?.to_vec1::<f32>()?,
        "small prefetch V must match host bit-for-bit"
    );

    // 2) Large payload (>= 4 MiB on CUDA/ROCm): exercises pinned CopyStream async DMA path
    if dev.is_cuda() || dev.is_rocm() {
        let elems = ASYNC_DMA_MIN_BYTES / 4; // 4 MiB of f32
        let host_big: Vec<f32> = (0..elems).map(|i| ((i % 997) as f32 - 498.0) * 0.01).collect();
        let t_host = Tensor::from_vec(host_big.clone(), (1, 1, 1024, elems / 1024), &Device::Cpu)?;
        let pf = prefetcher.prefetch_tensor(&t_host)?;
        assert!(pf.is_async(), "expected >=4 MiB upload on {:?} to use async DMA", dev.location());
        let t_dev = pf.wait_and_take(&mut prefetcher)?;
        let got = t_dev.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(got.len(), host_big.len());
        assert_eq!(got[0], host_big[0]);
        assert_eq!(got[elems / 2], host_big[elems / 2]);
        assert_eq!(got[elems - 1], host_big[elems - 1]);
    }
    Ok(())
}

/// Verifies `accel_check::record_probe` rejects broken TurboQuant error (`> PROBE_BROKEN_REL`)
/// and switches back to plain KV cleanly (`P4-F06` / `P5-33`).
#[test]
fn fused_gates_turbo_probe_rejects_broken_error() -> Result<()> {
    use logis_app::models::common::accel_check::record_probe;
    // Use CPU device location so we don't poison live GPU state for other tests in the process.
    let ok_small = record_probe(&Device::Cpu, 0.03, "test-fp8");
    assert!(ok_small, "normal 3% FP8 probe error should keep TurboQuant active");
    let ok_broken = record_probe(&Device::Cpu, 0.85, "test-broken");
    assert!(!ok_broken, "85% error must trip PROBE_BROKEN_REL and return false for plain KV fallback");
    Ok(())
}

/// Verifies CUDA/ROCm `quant_embed_self_check`, `EmbedTable::Quant` (`QTensor::embedding`),
/// and W8A16 Quantized Mode A (`fused_quant_argmax_cs`) across all 5 formats & M=12 chunked prefill.
#[test]
fn fused_gates_cuda_quant_embed_and_mode_a() -> Result<()> {
    use candle_core::lm_head::{fused_linear_argmax_cs, fused_quant_argmax_cs};
    use candle_core::quantized::{GgmlDType, QTensor};
    use logis_app::models::qwen3_5::model::{quant_embed_self_check, EmbedTable};
    let Some(dev) = gpu_device_opt() else {
        println!("[GATE QUANT-EMBED] skipped: no GPU device");
        return Ok(());
    };
    if !(dev.is_cuda() || dev.is_rocm()) {
        println!("[GATE QUANT-EMBED] skipped on non-CUDA/ROCm device {:?}", dev.location());
        return Ok(());
    }
    let (v, k, m) = (97usize, 256usize, 12usize);
    let w_f32 = Tensor::from_vec(det(v * k, 811), (v, k), &Device::Cpu)?;
    let h_f32 = Tensor::from_vec(det(m * k, 812), (m, k), &Device::Cpu)?;
    let w_gpu = w_f32.to_device(&dev)?;
    let h_bf16 = h_f32.to_dtype(DType::BF16)?.to_device(&dev)?;

    // 1) Q8_0 quantized embedding self-check + EmbedTable::Quant lookup parity
    let q8 = QTensor::quantize(&w_gpu, GgmlDType::Q8_0)?;
    assert!(
        quant_embed_self_check(&dev, &q8),
        "quant_embed_self_check must pass on {:?}",
        dev.location()
    );
    let table = EmbedTable::Quant { q: std::sync::Arc::new(q8), hidden: k };
    let ids = Tensor::new(&[[0u32, 13, 47, 96]], &dev)?;
    let got_emb = table.forward(&ids)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let ref_q8 = QTensor::quantize(&w_f32, GgmlDType::Q8_0)?.dequantize(&Device::Cpu)?;
    let ref_emb = ref_q8
        .index_select(&Tensor::new(&[0u32, 13, 47, 96], &Device::Cpu)?, 0)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let max_diff = got_emb
        .iter()
        .zip(&ref_emb)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff <= 1e-4, "Q8_0 EmbedTable::Quant mismatch: max_diff={max_diff}");

    // 2) Quantized Mode A (M=12 chunked prefill + BF16) across all 5 W8A16 formats
    for qt in [
        GgmlDType::Q8_0,
        GgmlDType::Q4_0,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ] {
        let q_cpu = QTensor::quantize(&w_f32, qt)?;
        let w_dq = q_cpu.dequantize(&Device::Cpu)?;
        let ref_out = fused_linear_argmax_cs(&h_f32.to_dtype(DType::BF16)?.to_dtype(DType::F32)?, &w_dq, None, None, None)?;
        let ref_ids = ref_out.ids.to_vec1::<u32>()?;

        let q_dev = QTensor::quantize(&w_gpu, qt)?;
        let got = fused_quant_argmax_cs(&h_bf16, &q_dev, None, None, None)?;
        let (got_ids, _) = got.download_ids_and_mask()?;
        assert_eq!(got_ids, ref_ids, "Quantized Mode A M={m} BF16 mismatch for {qt:?}");
    }
    println!("[GATE QUANT-EMBED] ({dev:?}) Q8_0 embed + 5x W8A16 Quantized Mode A (M={m}, V={v}, BF16) OK");
    Ok(())
}

/// 🌟 [PRC-03 / PRC-13] CUDA K-quant(`Q4K`/`Q5K`/`Q6K`/`Q8_0`) 정확 W8A16 선형 경로와
/// `Device::cuda_compute_capability()` 기반 SM 아키텍처 조회를 검증합니다.
#[test]
fn fused_gates_cuda_kquant_exact_and_sm_arch() -> Result<()> {
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    use candle_nn::Module;
    use logis_app::models::common::gguf::{set_kquant_exact_override, QuantizedLinear};

    let Some(dev) = gpu_device_opt() else {
        println!("[GATE KQUANT-EXACT] skipped: no GPU device");
        return Ok(());
    };
    if !dev.is_cuda() {
        println!("[GATE KQUANT-EXACT] skipped on non-CUDA device {:?}", dev.location());
        return Ok(());
    }

    // 1) PRC-13: CUDA 드라이버 속성으로 SM 아키텍처 조회 (nvidia-smi 프로세스 기동 없음)
    let (maj, min) = dev
        .cuda_compute_capability()
        .expect("CUDA device must report compute capability");
    assert!(maj >= 7, "unexpected compute capability {maj}.{min}");
    let sm = logis_app::utils::get_gpu_sm_arch()?;
    let expected_sm = maj as f32 + (min as f32) * 0.1;
    assert!(
        (sm - expected_sm).abs() < 0.02,
        "get_gpu_sm_arch ({sm}) must match Device::cuda_compute_capability() ({expected_sm})"
    );

    // 2) PRC-03: Q8_0 / Q4K / Q5K / Q6K 에서 F32/F16/BF16 입력 정확 경로 검증 (m=1 decode + m=4)
    let (out_dim, in_dim) = (256usize, 512usize);
    let w_f32 = Tensor::from_vec(det(out_dim * in_dim, 911), (out_dim, in_dim), &Device::Cpu)?;
    let w_gpu = w_f32.to_device(&dev)?;
    let x_f32 = Tensor::from_vec(det(4 * in_dim, 912), (1, 4, in_dim), &Device::Cpu)?;

    set_kquant_exact_override(None);
    for qt in [GgmlDType::Q8_0, GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K] {
        let q_cpu = QTensor::quantize(&w_f32, qt)?;
        let w_dq = q_cpu.dequantize(&Device::Cpu)?;
        let q_dev = QTensor::quantize(&w_gpu, qt)?;
        let layer = QuantizedLinear::new(QMatMul::from_qtensor(q_dev)?, None);

        for (dt, tol) in [(DType::F32, 1e-5f32), (DType::F16, 2e-3), (DType::BF16, 1.5e-2)] {
            let x_q = x_f32.to_dtype(dt)?;
            let x_ref = x_q.to_dtype(DType::F32)?;
            let expected = x_ref.reshape((4, in_dim))?.matmul(&w_dq.t()?)?.reshape((1, 4, out_dim))?;
            let got = layer.forward(&x_q.to_device(&dev)?)?.to_device(&Device::Cpu)?;
            let rel = logis_app::models::common::accel_check::rel_err(&got, &expected)?;
            assert!(
                rel <= tol,
                "QuantizedLinear exact path {qt:?} {dt:?} rel_err {rel:.2e} > {tol:.0e}"
            );
        }
    }
    println!(
        "[GATE KQUANT-EXACT] ({dev:?} sm_{maj}{min}) Q8_0/Q4K/Q5K/Q6K x F32/F16/BF16 exact W8A16 OK"
    );
    Ok(())
}

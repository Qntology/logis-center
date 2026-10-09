//! Vulkan 백엔드 검증 (feature = "vulkan")
//!
//! 앱이 실제로 쓰는 연산/조합 패턴을 Vulkan 디바이스에서 실행해 CPU 결과와 대조합니다.
//!
//! 주의: 포크된 candle 은 소프트웨어 Vulkan(llvmpipe/lavapipe)이나 외장 GPU 에서
//! 기본값으로 "GPU 셰이더 끔(CPU 코드 + 매핑 메모리)" 경로를 탑니다.
//! 그래서 모든 비교는 `with_modes` 로 두 번 실행합니다.
//!   - native=false : 기본 경로 (CPU 폴백 + Vulkan 메모리)
//!   - native=true  : GPU 셰이더 강제 + 크기 임계값 0 (작은 텐서도 셰이더로 실행)
//! 셰이더 설정은 프로세스 전역이므로 Vulkan 테스트는 모두 `MODE_LOCK` 을 잡고 실행합니다.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::vulkan_backend::shaders;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::common::TempDir;
use tauri_app_lib::models::common::eager_attention_forward;
use tauri_app_lib::models::qwen3::config::Qwen3Config;
use tauri_app_lib::models::qwen3::model::Qwen3Model;
use tauri_app_lib::position_embed::rope::{apply_rotary_pos_emb, RoPE};
use tauri_app_lib::utils::crypto::{decrypt_data, encrypt_data, kv_block_plaintext};
use tauri_app_lib::utils::device_utils::{
    flush_gpu_memory_pool, get_dtype, get_gpu_device, get_optimal_device_config, gpu_count,
    gpu_mem_info, new_gpu_device,
};
use tauri_app_lib::utils::direct_loader::{load_kv_block, save_kv_block};
use tauri_app_lib::utils::get_logit_processor;
use tauri_app_lib::utils::resources::KvResidency;
use tauri_app_lib::utils::tensor_utils::prepare_causal_attention_mask;

type R = anyhow::Result<()>;

static MODE_LOCK: Mutex<()> = Mutex::new(());

const NO_DEVICE_HINT: &str = "no usable Vulkan device. The fork skips CPU-type devices by default; \
on a machine without a GPU run with CANDLE_VULKAN_ALLOW_CPU=1 (mesa llvmpipe/lavapipe)";

fn vk() -> Device {
    assert!(gpu_count() > 0, "{NO_DEVICE_HINT}");
    let d = get_gpu_device(0);
    assert!(d.is_vulkan(), "expected a Vulkan device, got {d:?}");
    d
}

/// native 커널 off / on(임계값 0) 두 모드로 `f` 를 실행합니다.
fn with_modes(f: impl Fn(&Device) -> R) -> R {
    let _guard = MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = vk();
    for native in [false, true] {
        shaders::set_native_override(Some(native));
        shaders::set_native_thresholds(Some(0), Some(0));
        let res = f(&dev);
        shaders::set_native_override(None);
        shaders::set_native_thresholds(None, None);
        res.map_err(|e| e.context(format!("native kernels: {native}")))?;
    }
    Ok(())
}

fn to_vec_f64(t: &Tensor) -> anyhow::Result<Vec<f64>> {
    Ok(t.to_device(&Device::Cpu)?
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?)
}

/// 원소별 |got-want| <= tol * max(|got|,|want|,1)
fn assert_close(got: &Tensor, want: &Tensor, tol: f64, what: &str) -> R {
    assert_eq!(got.dims(), want.dims(), "{what}: shape mismatch");
    let g = to_vec_f64(got)?;
    let w = to_vec_f64(want)?;
    for (i, (x, y)) in g.iter().zip(w.iter()).enumerate() {
        if x == y || (x.is_nan() && y.is_nan()) {
            continue;
        }
        let diff = (x - y).abs();
        let scale = x.abs().max(y.abs()).max(1.0);
        assert!(diff <= tol * scale, "{what}: mismatch at {i}: vulkan={x} cpu={y} (tol {tol})");
    }
    Ok(())
}

fn randn(shape: &[usize]) -> anyhow::Result<Tensor> {
    Ok(Tensor::randn(0f32, 1f32, shape, &Device::Cpu)?)
}

/// CPU 에서 계산한 결과와 같은 연산을 Vulkan 에서 계산한 결과를 비교합니다.
fn same_on_both(
    dev: &Device,
    inputs: &[&Tensor],
    tol: f64,
    what: &str,
    f: impl Fn(&[Tensor]) -> candle_core::Result<Tensor>,
) -> R {
    let cpu_in: Vec<Tensor> = inputs.iter().map(|t| (*t).clone()).collect();
    let vk_in: Vec<Tensor> = inputs.iter().map(|t| t.to_device(dev)).collect::<Result<_, _>>()?;
    let want = f(&cpu_in)?;
    let got = f(&vk_in)?;
    assert!(got.device().is_vulkan(), "{what}: result left the Vulkan device");
    assert_close(&got, &want, tol, what)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. 디바이스 탐색 / dtype 정책 (device_utils)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn device_discovery_and_dtype_policy() {
    let _guard = MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let n = gpu_count();
    assert!(n >= 1, "{NO_DEVICE_HINT}");
    assert!(new_gpu_device(0).expect("new_gpu_device(0)").is_vulkan());
    assert!(get_gpu_device(0).is_vulkan());

    let cfg = get_optimal_device_config();
    eprintln!("Vulkan device: {}", cfg.name);
    assert!(!cfg.is_cpu);
    assert_eq!(cfg.gpu_id, 0);
    assert!(cfg.name.starts_with("Vulkan (GPU 0: "), "unexpected name {}", cfg.name);

    // Vulkan 빌드에서는 모든 부동소수 설정을 F32 로 강제한다
    assert_eq!(get_dtype(None, "bfloat16"), DType::F32);
    assert_eq!(get_dtype(None, "float16"), DType::F32);
    assert_eq!(get_dtype(None, "float32"), DType::F32);
    assert_eq!(get_dtype(None, "int32"), DType::I64);
    assert_eq!(get_dtype(Some(DType::BF16), "float32"), DType::BF16);

    let (free, total) = gpu_mem_info(0).expect("gpu_mem_info(0)");
    assert!(total > 0 && free <= total, "free={free} total={total}");
    if n == 1 {
        assert!(gpu_mem_info(1).is_none());
    }
}

#[test]
fn transfer_roundtrip_keeps_values_for_app_dtypes() -> R {
    let _guard = MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = vk();
    let x = randn(&[3, 5, 7])?;
    for dt in [DType::F32, DType::F16, DType::BF16] {
        let c = x.to_dtype(dt)?;
        let back = c.to_device(&dev)?.to_device(&Device::Cpu)?;
        assert_close(&back, &c, 0.0, &format!("roundtrip {dt:?}"))?;
    }
    let ids = Tensor::new(&[0u32, 1, 151_935, 42], &Device::Cpu)?;
    assert_eq!(ids.to_device(&dev)?.to_vec1::<u32>()?, vec![0, 1, 151_935, 42]);
    let m = Tensor::new(&[0u8, 1, 1, 0], &Device::Cpu)?;
    assert_eq!(m.to_device(&dev)?.to_vec1::<u8>()?, vec![0, 1, 1, 0]);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. 모델에서 쓰는 matmul 패턴
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn matmul_patterns_used_by_models() -> R {
    let x = randn(&[2, 5, 64])?;
    let w = (randn(&[96, 64])? * 0.125)?;
    let b = randn(&[96])?;
    let q = randn(&[1, 4, 7, 16])?;
    let k = randn(&[1, 4, 9, 16])?;
    let k_kv = randn(&[1, 2, 9, 16])?;
    let pos = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((12, 1))?;
    let inv = randn(&[1, 8])?;
    let big = randn(&[4096, 64])?;
    let vecc = randn(&[64, 1])?;
    with_modes(|dev| {
        // Linear (x @ w^T + b)
        same_on_both(dev, &[&x, &w, &b], 1e-4, "linear", |t| {
            candle_nn::Linear::new(t[1].clone(), Some(t[2].clone())).forward(&t[0])
        })?;
        // 비연속 전치 K 와의 어텐션 스코어
        same_on_both(dev, &[&q, &k], 1e-4, "q@k^T (strided)", |t| {
            t[0].matmul(&t[1].transpose(2, 3)?)
        })?;
        // GQA-FOLD: Q 헤드 축을 접어서 K 헤드 수와 맞춘다
        same_on_both(dev, &[&q, &k_kv], 1e-4, "gqa folded scores", |t| {
            t[0].reshape((1, 2, 14, 16))?.matmul(&t[1].transpose(2, 3)?)
        })?;
        // RoPE: positions (len,1) @ inv_freq (1,d/2)  — K=1
        same_on_both(dev, &[&pos, &inv], 1e-5, "rope outer product", |t| t[0].matmul(&t[1]))?;
        // 배치 stride 0 (broadcast_as) 행렬곱
        same_on_both(dev, &[&inv], 1e-5, "batch-stride-0 matmul", |t| {
            let a = t[0].reshape((1, 8, 1))?.broadcast_as((3, 8, 1))?;
            let bb = t[0].narrow(1, 0, 4)?.reshape((1, 1, 4))?.broadcast_as((3, 1, 4))?;
            a.matmul(&bb)
        })?;
        // 어휘 청크 mat-vec (semantic prejudice 패턴)
        same_on_both(dev, &[&big, &vecc], 1e-4, "mat-vec", |t| t[0].matmul(&t[1]))?;
        Ok(())
    })
}

#[test]
fn quantized_matmul_matches_cpu_for_app_gguf_types() -> R {
    // GGUF 경로: Q8_0 (임베딩/ssm_beta·alpha), Q4K (대부분의 가중치), Q6K (출력 헤드)
    let w = (randn(&[64, 512])? * 0.05)?;
    let x = randn(&[1, 3, 512])?;
    with_modes(|dev| {
        for dt in [GgmlDType::Q8_0, GgmlDType::Q4K, GgmlDType::Q6K] {
            let mm_cpu = QMatMul::from_qtensor(QTensor::quantize(&w, dt)?)?;
            let mm_vk = QMatMul::from_qtensor(QTensor::quantize_onto(&w, dt, dev)?)?;
            let want = mm_cpu.forward(&x)?;
            let got = mm_vk.forward(&x.to_device(dev)?)?;
            assert!(got.device().is_vulkan(), "{dt:?}: result left the device");
            assert_close(&got, &want, 1e-3, &format!("{dt:?} qmatmul"))?;
            // 양자화 오차 자체도 합리적인 범위인지 (역양자화 가중치 기준)
            let deq = QTensor::quantize(&w, dt)?.dequantize(&Device::Cpu)?;
            let reference = x.broadcast_matmul(&deq.t()?)?;
            let r = to_vec_f64(&reference)?;
            let g = to_vec_f64(&got)?;
            let max = r.iter().fold(0f64, |m, v| m.max(v.abs()));
            let err = g.iter().zip(&r).fold(0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(err <= 5e-2 * max, "{dt:?}: err {err} vs max |y| {max}");
        }
        Ok(())
    })
}

#[test]
fn native_kernels_really_dispatch_on_the_device() -> R {
    // 셰이더가 실제로 실행됐는지 확인하는 진단:
    //  - native=false : CPU gemm 코드를 매핑 메모리에서 실행 → CPU 와 비트 단위로 동일
    //  - native=true  : GPU 셰이더(다른 합산 순서) → 일부 원소가 비트 단위로 달라야 한다
    let _guard = MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = vk();
    let a = randn(&[64, 512])?;
    let b = randn(&[512, 96])?;
    let want: Vec<f32> = a.matmul(&b)?.flatten_all()?.to_vec1()?;
    let run = |native: bool| -> anyhow::Result<Vec<f32>> {
        shaders::set_native_override(Some(native));
        shaders::set_native_thresholds(Some(0), Some(0));
        let r = a.to_device(&dev)?.matmul(&b.to_device(&dev)?)?.flatten_all()?.to_device(&Device::Cpu)?.to_vec1();
        shaders::set_native_override(None);
        shaders::set_native_thresholds(None, None);
        Ok(r?)
    };
    let off = run(false)?;
    let on = run(true)?;
    let bitwise_diff = |v: &[f32]| v.iter().zip(&want).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    eprintln!("bitwise differences vs CPU: native=false {}, native=true {} (of {})", bitwise_diff(&off), bitwise_diff(&on), want.len());
    assert_eq!(bitwise_diff(&off), 0, "default path should run the CPU gemm on mapped memory");
    assert!(bitwise_diff(&on) > 0, "native=true produced CPU-identical bits: shaders did not run");
    let max_err = on.iter().zip(&want).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(max_err < 1e-3, "shader matmul error {max_err}");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. candle-nn / 원소별 연산
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn nn_ops_used_by_models() -> R {
    let x = randn(&[2, 6, 96])?;
    let w = ((randn(&[96])? * 0.1)? + 1.0)?;
    let b = (randn(&[96])? * 0.1)?;
    let pos = (randn(&[2, 6, 96])?.abs()? + 0.1)?;
    with_modes(|dev| {
        same_on_both(dev, &[&x], 1e-5, "softmax_last_dim", |t| candle_nn::ops::softmax_last_dim(&t[0]))?;
        same_on_both(dev, &[&x], 1e-5, "softmax(dim=-1)", |t| candle_nn::ops::softmax(&t[0], D::Minus1))?;
        same_on_both(dev, &[&x], 1e-5, "softmax(dim=1)", |t| candle_nn::ops::softmax(&t[0], 1))?;
        same_on_both(dev, &[&x, &w], 1e-4, "rms_norm", |t| candle_nn::ops::rms_norm(&t[0], &t[1], 1e-6))?;
        same_on_both(dev, &[&x, &w], 1e-4, "RmsNorm module", |t| {
            candle_nn::RmsNorm::new(t[1].clone(), 1e-6).forward(&t[0])
        })?;
        same_on_both(dev, &[&x, &w, &b], 1e-4, "LayerNorm", |t| {
            candle_nn::LayerNorm::new(t[1].clone(), t[2].clone(), 1e-6).forward(&t[0])
        })?;
        same_on_both(dev, &[&x], 1e-5, "sigmoid", |t| candle_nn::ops::sigmoid(&t[0]))?;
        same_on_both(dev, &[&x], 1e-5, "silu", |t| t[0].silu())?;
        same_on_both(dev, &[&x], 1e-4, "gelu(tanh)", |t| t[0].gelu())?;
        same_on_both(dev, &[&x], 1e-4, "gelu_erf", |t| t[0].gelu_erf())?;
        same_on_both(dev, &[&x], 1e-5, "exp/cos/sin/affine", |t| {
            (t[0].exp()? + t[0].cos()? + t[0].sin()?)?.affine(0.5, -1.0)
        })?;
        same_on_both(dev, &[&pos], 1e-5, "sqrt/sqr/powf/log", |t| {
            t[0].sqrt()? + t[0].sqr()? + t[0].powf(1.7)? + t[0].log()?
        })?;
        same_on_both(dev, &[&x], 1e-5, "maximum floor (-10000)", |t| {
            let floor = Tensor::new(-0.5f32, t[0].device())?.broadcast_as(t[0].shape())?;
            t[0].maximum(&floor)
        })?;
        same_on_both(dev, &[&x], 1e-5, "softplus", |t| {
            // qwen3_5 gated delta net: ln(1+exp(x))
            (t[0].exp()? + 1.0)?.log()
        })?;
        same_on_both(dev, &[&x], 1e-4, "sum/max/mean keepdim", |t| {
            let s = t[0].sum_keepdim(D::Minus1)?;
            let m = t[0].max_keepdim(D::Minus1)?;
            let a = t[0].mean_keepdim(D::Minus1)?;
            Tensor::cat(&[&s, &m, &a], D::Minus1)
        })?;
        Ok(())
    })
}

/// f64 두 단계(평균 → 편차제곱) 기준 LayerNorm
fn layer_norm_reference(x: &Tensor, eps: f64) -> anyhow::Result<Tensor> {
    let x = x.to_device(&Device::Cpu)?.to_dtype(DType::F64)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(xc.broadcast_div(&(var + eps)?.sqrt()?)?.to_dtype(DType::F32)?)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> anyhow::Result<f64> {
    let (a, b) = (to_vec_f64(a)?, to_vec_f64(b)?);
    Ok(a.iter().zip(&b).fold(0f64, |m, (x, y)| m.max((x - y).abs())))
}

#[test]
fn layer_norm_zero_mean_matches_two_pass_reference() -> R {
    let x = randn(&[4, 16, 1152])?;
    let ones = Tensor::ones(1152, DType::F32, &Device::Cpu)?;
    let zeros = Tensor::zeros(1152, DType::F32, &Device::Cpu)?;
    let want = layer_norm_reference(&x, 1e-6)?;
    with_modes(|dev| {
        let ln = candle_nn::LayerNorm::new(ones.to_device(dev)?, zeros.to_device(dev)?, 1e-6);
        let got = ln.forward(&x.to_device(dev)?)?;
        let err = max_abs_diff(&got, &want)?;
        assert!(err < 1e-3, "layer_norm error {err}");
        Ok(())
    })
}

#[test]
#[ignore = "BUG(LN-1): candle-nn layer_norm uses one-pass E[x^2]-E[x]^2 variance; rows with a large mean (e.g. SigLIP2 / Qwen3-VL activations) lose precision on CPU and Vulkan"]
fn layer_norm_large_mean_matches_two_pass_reference() -> R {
    // 평균이 큰 행 (x = 1000 + N(0,1)) — 정상적인 두 단계 분산이면 오차가 작아야 한다
    let x = (randn(&[4, 16, 1152])? + 1000.0)?;
    let ones = Tensor::ones(1152, DType::F32, &Device::Cpu)?;
    let zeros = Tensor::zeros(1152, DType::F32, &Device::Cpu)?;
    let want = layer_norm_reference(&x, 1e-6)?;
    let cpu = candle_nn::LayerNorm::new(ones.clone(), zeros.clone(), 1e-6).forward(&x)?;
    let cpu_err = max_abs_diff(&cpu, &want)?;
    eprintln!("layer_norm(mean=1000) max abs error: cpu {cpu_err}");
    with_modes(|dev| {
        let ln = candle_nn::LayerNorm::new(ones.to_device(dev)?, zeros.to_device(dev)?, 1e-6);
        let got = ln.forward(&x.to_device(dev)?)?;
        let err = max_abs_diff(&got, &want)?;
        eprintln!("layer_norm(mean=1000) max abs error: vulkan {err}");
        assert!(err < 1e-2, "vulkan layer_norm error {err}");
        Ok(())
    })?;
    assert!(cpu_err < 1e-2, "cpu layer_norm error {cpu_err}");
    Ok(())
}

#[test]
fn layout_and_indexing_ops_used_by_models() -> R {
    let x = randn(&[2, 4, 6, 8])?;
    let ids = Tensor::new(&[3u32, 0, 5, 5, 1], &Device::Cpu)?;
    let gidx = Tensor::new(&[[1u32, 0], [2, 2], [0, 1]], &Device::Cpu)?;
    let g_src = randn(&[3, 4])?;
    let six = randn(&[1, 2, 3, 2, 3, 4])?;
    with_modes(|dev| {
        same_on_both(dev, &[&x], 0.0, "transpose+contiguous", |t| t[0].transpose(1, 2)?.contiguous())?;
        same_on_both(dev, &[&six], 0.0, "rank-6 permute", |t| t[0].permute((0, 1, 3, 2, 4, 5))?.contiguous())?;
        same_on_both(dev, &[&x], 0.0, "narrow/cat", |t| {
            let a = t[0].narrow(2, 1, 3)?;
            let b = t[0].narrow(2, 4, 2)?;
            Tensor::cat(&[&b, &a], 2)
        })?;
        same_on_both(dev, &[&x, &ids], 0.0, "index_select", |t| t[0].index_select(&t[1], 2))?;
        same_on_both(dev, &[&g_src, &gidx], 0.0, "gather", |t| t[0].narrow(1, 0, 2)?.contiguous()?.gather(&t[1], 0))?;
        same_on_both(dev, &[&x], 0.0, "repeat/flip/pad", |t| {
            let r = t[0].narrow(0, 0, 1)?.repeat((2, 1, 1, 1))?;
            let f = r.flip(&[3])?;
            f.pad_with_zeros(3, 2, 1)
        })?;
        same_on_both(dev, &[&x], 0.0, "where_cond(u8 mask)", |t| {
            let mask = Tensor::triu2(8, DType::F32, t[0].device())?.to_dtype(DType::U8)?.broadcast_as((2, 4, 8, 8))?;
            let xs = t[0].narrow(2, 0, 6)?.pad_with_zeros(2, 0, 2)?;
            let zero = Tensor::zeros((2, 4, 8, 8), DType::F32, t[0].device())?;
            mask.where_cond(&zero, &xs)
        })?;
        same_on_both(dev, &[&x], 1e-5, "cumsum / tril2 / eye", |t| {
            let c = t[0].cumsum(D::Minus1)?;
            let tri = Tensor::tril2(8, DType::F32, t[0].device())?;
            let eye = Tensor::eye(8, DType::F32, t[0].device())?;
            c.broadcast_add(&(tri + eye)?.sum_keepdim(0)?)
        })?;
        same_on_both(dev, &[&x], 0.0, "arange_step", |_t| Tensor::arange_step(0f32, 3f32, 0.25, _t[0].device()))?;
        Ok(())
    })
}

#[test]
fn argmax_and_sort_over_full_vocab() -> R {
    let logits = randn(&[151_936])?;
    let small = randn(&[2, 64])?;
    with_modes(|dev| {
        let l = logits.to_device(dev)?;
        let want = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
        assert_eq!(l.argmax(D::Minus1)?.to_scalar::<u32>()?, want, "argmax over 151936 vocab");
        // arg_sort 은 Vulkan 구현이 없어 CPU 폴백 — 결과만 같으면 된다
        let s_vk = small.to_device(dev)?.arg_sort_last_dim(false)?.to_device(&Device::Cpu)?;
        let s_cpu = small.arg_sort_last_dim(false)?;
        assert_eq!(s_vk.to_vec2::<u32>()?, s_cpu.to_vec2::<u32>()?);
        Ok(())
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. 앱 고유의 조합 연산 (마스크 / 어텐션 / RoPE / delta rule / 샘플링)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn causal_mask_matches_cpu_and_semantics() -> R {
    with_modes(|dev| {
        for (b, tgt, off) in [(1usize, 1usize, 0usize), (2, 5, 0), (2, 5, 3), (1, 7, 11)] {
            let want = prepare_causal_attention_mask(b, tgt, off, &Device::Cpu)?;
            let got = prepare_causal_attention_mask(b, tgt, off, dev)?;
            assert_eq!(got.dims(), &[b, 1, tgt, tgt + off]);
            assert_close(&got, &want, 0.0, &format!("mask b={b} tgt={tgt} off={off}"))?;
            // 의미 검증: query i 는 key j <= i+off 만 볼 수 있다
            let m = want.i((0, 0))?.to_vec2::<f32>()?;
            for (i, row) in m.iter().enumerate() {
                for (j, v) in row.iter().enumerate() {
                    if j <= i + off {
                        assert_eq!(*v, 0.0);
                    } else {
                        assert!(v.is_infinite() && *v < 0.0);
                    }
                }
            }
        }
        Ok(())
    })
}

/// repeat_kv + softmax(QK^T*s + mask) V 의 교과서 구현 → (b, q_len, q_heads, d)
fn naive_attention(q: &Tensor, k: &Tensor, v: &Tensor, groups: usize, mask: Option<&Tensor>, scale: f64) -> anyhow::Result<Tensor> {
    let rep = |t: &Tensor| -> candle_core::Result<Tensor> {
        let (b, h, s, d) = t.dims4()?;
        t.unsqueeze(2)?.expand((b, h, groups, s, d))?.reshape((b, h * groups, s, d))
    };
    let (k, v) = (rep(k)?, rep(v)?);
    let mut w = (q.matmul(&k.t()?)? * scale)?;
    if let Some(m) = mask {
        w = w.broadcast_add(m)?;
    }
    let p = candle_nn::ops::softmax_last_dim(&w)?;
    Ok(p.matmul(&v)?.transpose(1, 2)?.contiguous()?)
}

#[test]
fn eager_attention_matches_naive_reference_short_and_long_context() -> R {
    // kv <= 4096 : 단일 블록, kv > 4096 : 블록별 online-softmax 병합 경로
    let cases = [(1usize, 9usize, 1usize), (2, 9, 5), (1, 4100, 1), (2, 4100, 3)];
    let inputs: Vec<_> = cases
        .iter()
        .map(|&(groups, kv, ql)| {
            let q = randn(&[1, 4, ql, 16]).unwrap();
            let k = randn(&[1, 4 / groups, kv, 16]).unwrap();
            let v = randn(&[1, 4 / groups, kv, 16]).unwrap();
            (groups, kv, ql, q, k, v)
        })
        .collect();
    with_modes(|dev| {
        for (groups, kv, ql, q, k, v) in &inputs {
            let scale = 1.0 / 4.0;
            let mask_cpu = if *ql > 1 { Some(prepare_causal_attention_mask(1, *ql, kv - ql, &Device::Cpu)?) } else { None };
            let want = naive_attention(q, k, v, *groups, mask_cpu.as_ref(), scale)?;
            let cpu = eager_attention_forward(q, k, v, Some(*groups), mask_cpu.as_ref(), scale)?;
            let what = format!("groups={groups} kv={kv} q_len={ql}");
            assert_close(&cpu, &want, 1e-4, &format!("cpu eager vs naive {what}"))?;

            let mask_vk = mask_cpu.as_ref().map(|m| m.to_device(dev)).transpose()?;
            let got = eager_attention_forward(
                &q.to_device(dev)?,
                &k.to_device(dev)?,
                &v.to_device(dev)?,
                Some(*groups),
                mask_vk.as_ref(),
                scale,
            )?;
            assert!(got.device().is_vulkan());
            assert_close(&got, &want, 1e-4, &format!("vulkan eager vs naive {what}"))?;
        }
        Ok(())
    })
}

#[test]
fn rope_tables_and_rotation_match_cpu() -> R {
    let q = randn(&[1, 4, 5, 16])?;
    let k = randn(&[1, 2, 5, 16])?;
    with_modes(|dev| {
        let (cos_c, sin_c) = RoPE::new(16, 1_000_000.0, &Device::Cpu)?.forward(7, 5, &Device::Cpu)?;
        let (cos_v, sin_v) = RoPE::new(16, 1_000_000.0, dev)?.forward(7, 5, dev)?;
        assert_close(&cos_v, &cos_c, 1e-5, "rope cos")?;
        assert_close(&sin_v, &sin_c, 1e-5, "rope sin")?;

        let (qc, kc) = apply_rotary_pos_emb(&q, &k, &cos_c, &sin_c, false)?;
        let (qv, kv) = apply_rotary_pos_emb(&q.to_device(dev)?, &k.to_device(dev)?, &cos_v, &sin_v, false)?;
        assert_close(&qv, &qc, 1e-5, "rotated q")?;
        assert_close(&kv, &kc, 1e-5, "rotated k")?;

        // 위치 0 에서는 회전이 항등이어야 한다 (cos=1, sin=0)
        let (c0, s0) = RoPE::new(16, 1_000_000.0, dev)?.forward(0, 1, dev)?;
        let q0 = q.narrow(2, 0, 1)?.to_device(dev)?;
        let (r0, _) = apply_rotary_pos_emb(&q0, &q0, &c0, &s0, false)?;
        assert_close(&r0, &q0, 1e-6, "rope identity at position 0")?;
        Ok(())
    })
}

#[test]
fn delta_rule_recurrence_matches_reference_loop() -> R {
    // qwen3_5 gated delta net: Vulkan 은 chunk_tril_recurrence, 그 외는 slice_assign 루프
    let n = 16usize;
    let raw = (randn(&[1, 2, 3, n, n])? * 0.1)?;
    let lower = Tensor::tril2(n, DType::F32, &Device::Cpu)?
        .sub(&Tensor::eye(n, DType::F32, &Device::Cpu)?)?
        .broadcast_as(raw.shape())?;
    let attn = (raw * lower)?; // strictly lower triangular

    // 앱의 non-Vulkan 루프를 그대로 재현 (qwen3_5/model.rs)
    let mut want = attn.clone();
    let (d0, d1, d2, _, _) = want.dims5()?;
    for i in 1..n {
        let row = want.i((.., .., .., i, ..i))?.contiguous()?;
        let sub = want.i((.., .., .., ..i, ..i))?.contiguous()?;
        let attn_i = row.unsqueeze(D::Minus1)?.broadcast_mul(&sub)?.sum(D::Minus2)?.add(&row)?.unsqueeze(D::Minus2)?;
        want = want.slice_assign(&[(0..d0), (0..d1), (0..d2), (i..i + 1), (0..i)], &attn_i)?;
    }
    with_modes(|dev| {
        let got = candle_core::delta_rule::chunk_tril_recurrence(&attn.to_device(dev)?)?;
        assert!(got.device().is_vulkan());
        assert_close(&got, &want, 1e-5, "chunk_tril_recurrence")
    })
}

#[test]
fn logits_processor_sampling_is_device_independent() -> R {
    let logits = (randn(&[151_936])? * 3.0)?;
    with_modes(|dev| {
        let lv = logits.to_device(dev)?;
        // greedy
        let mut p = get_logit_processor(None, None, None, 1);
        let want = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
        assert_eq!(p.sample(&lv)?, want);
        // 온도 1e-8 은 greedy 로 취급
        let mut p = get_logit_processor(Some(1e-8), Some(0.9), Some(40), 1);
        assert_eq!(p.sample(&lv)?, want);
        // 같은 seed 면 디바이스와 무관하게 같은 토큰
        for (t, p_, k) in [(0.7f32, None, Some(50usize)), (0.9, Some(0.95f32), Some(40)), (1.0, Some(0.9), None)] {
            let mut a = get_logit_processor(Some(t), p_, k, 1234);
            let mut b = get_logit_processor(Some(t), p_, k, 1234);
            for _ in 0..5 {
                assert_eq!(a.sample(&lv)?, b.sample(&logits)?, "temp={t} top_p={p_:?} top_k={k:?}");
            }
        }
        Ok(())
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. 작은 랜덤 가중치 Qwen3 모델: CPU ↔ Vulkan, prefill+decode ↔ full forward
// ─────────────────────────────────────────────────────────────────────────────

fn tiny_qwen3() -> anyhow::Result<(Qwen3Config, HashMap<String, Tensor>)> {
    let cfg: Qwen3Config = serde_json::from_value(serde_json::json!({
        "hidden_act": "silu", "hidden_size": 64, "head_dim": 16, "intermediate_size": 128,
        "max_position_embeddings": 512, "num_attention_heads": 4, "num_hidden_layers": 2,
        "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0,
        "tie_word_embeddings": true, "vocab_size": 256
    }))?;
    let mut m = HashMap::new();
    let lin = |o: usize, i: usize| -> anyhow::Result<Tensor> { Ok((randn(&[o, i])? / (i as f64).sqrt())?) };
    let norm = |d: usize| -> anyhow::Result<Tensor> { Ok(((randn(&[d])? * 0.1)? + 1.0)?) };
    m.insert("model.embed_tokens.weight".to_string(), (randn(&[256, 64])? * 0.5)?);
    m.insert("model.norm.weight".to_string(), norm(64)?);
    for l in 0..2 {
        let p = format!("model.layers.{l}");
        m.insert(format!("{p}.self_attn.q_proj.weight"), lin(64, 64)?);
        m.insert(format!("{p}.self_attn.k_proj.weight"), lin(32, 64)?);
        m.insert(format!("{p}.self_attn.v_proj.weight"), lin(32, 64)?);
        m.insert(format!("{p}.self_attn.o_proj.weight"), lin(64, 64)?);
        m.insert(format!("{p}.self_attn.q_norm.weight"), norm(16)?);
        m.insert(format!("{p}.self_attn.k_norm.weight"), norm(16)?);
        m.insert(format!("{p}.mlp.gate_proj.weight"), lin(128, 64)?);
        m.insert(format!("{p}.mlp.up_proj.weight"), lin(128, 64)?);
        m.insert(format!("{p}.mlp.down_proj.weight"), lin(64, 128)?);
        m.insert(format!("{p}.input_layernorm.weight"), norm(64)?);
        m.insert(format!("{p}.post_attention_layernorm.weight"), norm(64)?);
    }
    Ok((cfg, m))
}

fn build(cfg: &Qwen3Config, w: &HashMap<String, Tensor>, dev: &Device) -> anyhow::Result<Qwen3Model> {
    Qwen3Model::new(cfg, VarBuilder::from_tensors(w.clone(), DType::F32, dev))
}

#[test]
fn tiny_qwen3_logits_match_cpu_with_kv_cache_and_ram_residency() -> R {
    let (cfg, w) = tiny_qwen3()?;
    let prompt = [3u32, 17, 42, 99, 5, 200, 7];
    let steps = [11u32, 250, 64];
    with_modes(|dev| {
        let ids = |v: &[u32], d: &Device| Tensor::new(v, d).and_then(|t| t.unsqueeze(0));

        // CPU 기준 (prefill + 3 decode)
        let mut cpu = build(&cfg, &w, &Device::Cpu)?;
        let mut cpu_logits = vec![cpu.forward(Some(&ids(&prompt, &Device::Cpu)?), None, 0)?];
        for (i, t) in steps.iter().enumerate() {
            cpu_logits.push(cpu.forward(Some(&ids(&[*t], &Device::Cpu)?), None, prompt.len() + i)?);
        }

        // Vulkan, KV 가 VRAM 에 상주 / RAM 으로 대피 두 경우 모두 같아야 한다
        for residency in [KvResidency::Vram, KvResidency::Ram] {
            let mut m = build(&cfg, &w, dev)?;
            m.set_kv_residency(residency);
            let got = m.forward(Some(&ids(&prompt, dev)?), None, 0)?;
            assert!(got.device().is_vulkan());
            assert_eq!(got.dims(), &[1, 1, 256]);
            assert_close(&got, &cpu_logits[0], 1e-3, &format!("{residency:?} prefill logits"))?;
            // 디코드 3 스텝 (가중치 GPU 미러는 2회 이상 사용 후 생성되므로 3회 이상 실행)
            for (i, t) in steps.iter().enumerate() {
                let got = m.forward(Some(&ids(&[*t], dev)?), None, prompt.len() + i)?;
                assert_close(&got, &cpu_logits[i + 1], 1e-3, &format!("{residency:?} decode step {i}"))?;
            }
        }

        // prefill 7 + decode 1 == 한 번에 8 토큰 forward 의 마지막 로짓
        let mut full = build(&cfg, &w, dev)?;
        let all: Vec<u32> = prompt.iter().chain(steps.iter().take(1)).copied().collect();
        let once = full.forward(Some(&ids(&all, dev)?), None, 0)?;
        assert_close(&once, &cpu_logits[1], 1e-3, "full forward vs prefill+decode")?;

        // KV 스냅샷 → 다른 모델로 복원 → 이어서 디코드
        let mut a = build(&cfg, &w, dev)?;
        a.forward(Some(&ids(&prompt, dev)?), None, 0)?;
        let snap = a.get_kv_cache();
        assert_eq!(snap.len(), 2);
        let (k0, _) = snap[0].as_ref().expect("layer0 kv");
        assert_eq!(k0.dims(), &[1, 2, prompt.len(), 16]);
        let mut b = build(&cfg, &w, dev)?;
        b.set_kv_cache(snap);
        let cont = b.forward(Some(&ids(&steps[..1], dev)?), None, prompt.len())?;
        assert_close(&cont, &cpu_logits[1], 1e-3, "decode after KV snapshot restore")?;
        Ok(())
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. KV 블록 SSD 오프로딩 왕복 (Vulkan 텐서 → safetensors → io_uring → Vulkan)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn kv_block_ssd_roundtrip_through_io_uring_restores_vulkan_tensors() -> R {
    let _guard = MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = vk();
    let dir = TempDir::new("kv_ssd");
    let (b_off, layer) = (1024usize, 3usize);
    let k = randn(&[1, 2, 64, 16])?.to_device(&dev)?;
    let v = randn(&[1, 2, 64, 16])?.to_device(&dev)?;

    // spawn_slot_worker 와 같은 키/형식으로 평문 safetensors 를 기록
    let prefix = format!("b{b_off}_l{layer}_");
    let shape: Vec<u32> = k.dims().iter().map(|&d| d as u32).collect();
    let mut ts: HashMap<String, Tensor> = HashMap::new();
    ts.insert(format!("{prefix}k_data"), k.to_device(&Device::Cpu)?);
    ts.insert(format!("{prefix}v_data"), v.to_device(&Device::Cpu)?);
    ts.insert(format!("{prefix}k_shape"), Tensor::new(shape.as_slice(), &Device::Cpu)?);
    let path = dir.path().join(format!("l{layer}.st"));
    candle_core::safetensors::save(&ts, &path)?;

    // 읽기: io_uring(load_kv_block) → 평문/암호문 판별 → safetensors 0.4 파싱
    let raw = load_kv_block(&path)?;
    assert_eq!(raw, std::fs::read(&path)?);
    let content = kv_block_plaintext(raw.clone());
    assert_eq!(content, raw, "plaintext block must be used as-is");
    let st = safetensors::SafeTensors::deserialize(&content)?;
    let kd = st.tensor(&format!("{prefix}k_data"))?;
    let vd = st.tensor(&format!("{prefix}v_data"))?;
    let sh = st.tensor(&format!("{prefix}k_shape"))?;
    let dims: Vec<usize> = sh.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize).collect();
    assert_eq!(dims, vec![1, 2, 64, 16]);
    assert_eq!(kd.dtype(), safetensors::Dtype::F32);
    let k2 = Tensor::from_raw_buffer(kd.data(), DType::F32, &dims, &Device::Cpu)?.to_device(&dev)?;
    let v2 = Tensor::from_raw_buffer(vd.data(), DType::F32, &dims, &Device::Cpu)?.to_device(&dev)?;
    assert_close(&k2, &k, 0.0, "restored K")?;
    assert_close(&v2, &v, 0.0, "restored V")?;

    // 암호화된 블록(SSM 스냅샷 형식)도 같은 함수로 복원된다
    let enc = encrypt_data(&raw)?;
    let enc_path = dir.path().join("enc.st");
    save_kv_block(&enc_path, &enc)?;
    assert_eq!(kv_block_plaintext(load_kv_block(&enc_path)?), raw);
    Ok(())
}

#[test]
fn decrypt_data_accepts_plaintext_which_is_why_kv_blocks_need_detection() -> R {
    // XOR 스트림이라 무결성 검증이 없다: 평문을 넣어도 Ok(쓰레기)를 돌려준다.
    // (수정 전 qwen3_5 는 decrypt_data(..).unwrap_or(raw) 를 써서 SSD 에서 다시 읽은 KV 가 전부 0 이 됐다)
    let mut ts = HashMap::new();
    ts.insert("x".to_string(), randn(&[4, 4])?);
    let dir = TempDir::new("kv_crypto");
    let p = dir.path().join("x.st");
    candle_core::safetensors::save(&ts, &p)?;
    let plain = std::fs::read(&p)?;
    let garbled = decrypt_data(&plain)?;
    assert_ne!(garbled, plain);
    assert!(safetensors::SafeTensors::deserialize(&garbled).is_err());
    assert_eq!(kv_block_plaintext(plain.clone()), plain);
    // 너무 짧은/깨진 입력은 그대로 돌려준다 (호출부에서 파싱 실패로 처리)
    assert_eq!(kv_block_plaintext(vec![1, 2, 3]), vec![1, 2, 3]);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. 메모리 풀
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn flush_gpu_memory_pool_releases_pooled_buffers() -> R {
    let _guard = MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dev = vk();
    {
        let a = Tensor::zeros((1024, 1024), DType::F32, &dev)?;
        let b = (a + 1.0)?;
        assert_eq!(b.sum_all()?.to_scalar::<f32>()?, (1024 * 1024) as f32);
    }
    flush_gpu_memory_pool(0);
    let vd = dev.as_vulkan_device().expect("vulkan device");
    assert_eq!(vd.pooled_bytes(), 0, "pool not trimmed after flush_gpu_memory_pool");
    Ok(())
}

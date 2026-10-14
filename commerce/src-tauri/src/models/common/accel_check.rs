//! 🌟 [ACCEL-CHECK] GPU 가속 커널(FlashAttention / TurboQuant) 자가 검증과 실측 정확도 감시.
//!
//! 1) 자가 검증 — 장치·dtype·head_dim 조합마다 처음 쓰기 직전에 한 번
//!    - FlashAttention: 실제 모델과 같은 모양(GQA, 청크 프리필, 디코딩/split-kv, 비전처럼 non-causal,
//!      bshd·bhsd 두 레이아웃)으로 fused 커널 결과를 CPU 의 일반 연산(matmul + softmax) 결과와 비교합니다.
//!    - TurboQuant: 앱이 쓰는 K/V 형식으로 GPU 양자화·복원 결과를 CPU 코덱과 비교하고,
//!      압축 캐시 위의 fused 어텐션을 CPU 에서 복원한 K/V 로 계산한 기준값과 비교합니다.
//!    - 틀리면 그 커널만 끕니다.
//!      · FlashAttention 실패 → 이 백엔드의 fused flash-attn 을 끄고, 앱은 기존 블록 어텐션을 씁니다.
//!      · TurboQuant 양자화 실패 → 이 장치에서 압축 KV 캐시를 쓰지 않습니다(기존 FP8 경로).
//!      · TurboQuant fused 어텐션만 실패 → 압축은 유지하고 "복원 후 flash-attn" 경로로 계산합니다.
//! 2) 실측 감시 — 장치마다 처음 [`PROBE_LIMIT`] 번
//!    압축 캐시가 원본 K/V 전체를 받는 순간(첫 프리필 청크, 디코딩 전환)에 압축 캐시로 계산한
//!    어텐션과 원본으로 계산한 어텐션을 비교해 실제 모델·실제 GPU 의 오차를 기록합니다.
//!    오차가 비정상적으로 크면([`PROBE_BROKEN_REL`]) 이 장치의 TurboQuant 를 끄고 원래 경로로 돌아갑니다.
//!
//! `LOGIS_ACCEL_CHECK=off` 이면 둘 다 건너뜁니다.

use anyhow::{Result, bail};
use candle_core::{DType, Device, DeviceLocation, Tensor};
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// 자가 검증·실측 감시를 끄는 환경 변수 (`off` / `0` / `false`)
pub const CHECK_ENV: &str = "LOGIS_ACCEL_CHECK";
/// 실측 감시: 압축 캐시 어텐션의 상대오차가 이 값을 넘으면 커널 오동작 수준으로 보고 TurboQuant 를 끕니다.
/// (참고: Qwen 류 합성 데이터에서 FP8 ≈ 4–6%, 4bit(NVFP4 / TurboQuant 4bit) ≈ 17–27%, 커널 오류는 보통 50% 이상)
pub const PROBE_BROKEN_REL: f32 = 0.5;
/// 실측 감시: 이 값을 넘으면 경고를 남깁니다. (4bit 형식은 보통 여기에 걸립니다)
pub const PROBE_WARN_REL: f32 = 0.12;
/// 장치마다 실측 감시를 하는 최대 횟수 (레이어 단위)
pub const PROBE_LIMIT: usize = 64;

/// FlashAttention 자가 검증 허용 오차 (상대 오차, 16bit 입력 / f32 입력)
const FA_TOL_HALF: f32 = 2.5e-2;
const FA_TOL_F32: f32 = 1e-2;
/// TurboQuant: GPU 로 압축한 행을 CPU 코덱으로 풀었을 때 CPU 압축과의 차이 (경계값 반올림 몇 개 허용)
#[cfg_attr(not(feature = "turbo-quant"), allow(dead_code))]
const TQ_QUANT_TOL: f32 = 2e-2;
/// TurboQuant: GPU 복원과 CPU 복원의 차이
#[cfg_attr(not(feature = "turbo-quant"), allow(dead_code))]
const TQ_DEQUANT_TOL: f32 = 1e-3;
/// TurboQuant: fused 어텐션과 CPU 기준값의 차이
const TQ_ATTN_TOL: f32 = 3e-2;

fn checks_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var(CHECK_ENV) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no" | "disable" | "disabled"
        ),
        Err(_) => true,
    })
}

#[derive(Default)]
struct ProbeStats {
    count: usize,
    sum: f64,
    max: f32,
    pending: usize,
    pending_sum: f64,
    pending_max: f32,
    label: String,
}

#[derive(Default)]
struct DeviceState {
    /// FlashAttention 자가 검증 결과: (dtype, head_dim) → 통과 여부
    fa: HashMap<(DType, usize), bool>,
    /// TurboQuant 자가 검증 결과: head_dim → 통과 여부
    tq: HashMap<usize, bool>,
    /// 이 장치에서 TurboQuant 를 끈 이유
    tq_off: Option<String>,
    probe: ProbeStats,
}

fn states() -> MutexGuard<'static, HashMap<DeviceLocation, DeviceState>> {
    static S: OnceLock<Mutex<HashMap<DeviceLocation, DeviceState>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

fn is_gpu(device: &Device) -> bool {
    matches!(device, Device::Cuda(_) | Device::Rocm(_) | Device::Vulkan(_) | Device::Metal(_))
}

fn backend_name(device: &Device) -> &'static str {
    match device {
        Device::Cuda(_) => "cuda",
        Device::Rocm(_) => "rocm",
        Device::Vulkan(_) => "vulkan",
        Device::Metal(_) => "metal",
        Device::Cpu => "cpu",
        _ => "other",
    }
}

fn arch(device: &Device) -> String {
    candle_core::accel::profile(device).arch
}

/// 모델들이 이 장치에서 어텐션을 계산하는 dtype (CUDA/ROCm 은 BF16, 그 외 F32)
pub fn model_dtype(device: &Device) -> DType {
    match device {
        Device::Cuda(_) | Device::Rocm(_) => DType::BF16,
        _ => DType::F32,
    }
}

fn disable_flash_attn_backend(device: &Device) {
    candle_core::accel::update_config(|c| match device {
        Device::Cuda(_) => c.flash_attn.backends.cuda = false,
        Device::Rocm(_) => c.flash_attn.backends.rocm = false,
        Device::Vulkan(_) => c.flash_attn.backends.vulkan = false,
        _ => {}
    });
}

/// TurboQuant GPU 커널을 끕니다. (남아 있는 압축 데이터는 CPU 코덱으로 정확하게 풀립니다)
fn disable_turbo_quant_kernels(device: &Device) {
    candle_core::accel::update_config(|c| match device {
        Device::Cuda(_) => c.turbo_quant.backends.cuda = false,
        Device::Rocm(_) => c.turbo_quant.backends.rocm = false,
        Device::Vulkan(_) => c.turbo_quant.backends.vulkan = false,
        _ => {}
    });
}

// ---------------------------------------------------------------------------
// 공개 API
// ---------------------------------------------------------------------------

/// FlashAttention fused 커널을 이 장치·dtype·head_dim 에서 써도 되는지.
///
/// fused 경로가 설정(`CANDLE_FLASH_ATTN=off` 등)이나 빌드로 꺼져 있으면 false 이고,
/// GPU 에서는 처음 한 번 자가 검증을 통과해야 true 입니다. false 이면 호출부는 기존 블록 어텐션을 씁니다.
pub fn flash_attn_ok(device: &Device, dtype: DType, head_dim: usize) -> bool {
    if !candle_core::accel::config().flash_attn_allowed(device) {
        return false;
    }
    if !is_gpu(device) || !checks_enabled() {
        return true;
    }
    let mut g = states();
    let st = g.entry(device.location()).or_default();
    if let Some(ok) = st.fa.get(&(dtype, head_dim)) {
        return *ok;
    }
    let res = std::panic::catch_unwind(AssertUnwindSafe(|| fa_check(device, dtype, head_dim)));
    let tol = if matches!(dtype, DType::F32) { FA_TOL_F32 } else { FA_TOL_HALF };
    let ok = match res {
        Ok(Ok((worst, name, cases))) if worst <= tol => {
            println!(
                "[ACCEL-CHECK] {} {} · flash-attn {:?} d{}: 통과 (최대 상대오차 {:.1e}, {}개 형태)",
                backend_name(device),
                arch(device),
                dtype,
                head_dim,
                worst,
                cases
            );
            let _ = name;
            true
        }
        Ok(Ok((worst, name, _))) => {
            println!(
                "[ACCEL-CHECK] ❌ {} {} · flash-attn {:?} d{}: 결과 불일치 ({name} 상대오차 {:.2e} > {:.0e}) → 이 백엔드의 fused flash-attn 을 끄고 기존 어텐션 경로를 씁니다",
                backend_name(device),
                arch(device),
                dtype,
                head_dim,
                worst,
                tol
            );
            false
        }
        Ok(Err(e)) => {
            println!(
                "[ACCEL-CHECK] ❌ {} {} · flash-attn {:?} d{}: 검증 실행 실패 ({e}) → fused flash-attn 을 끄고 기존 어텐션 경로를 씁니다",
                backend_name(device),
                arch(device),
                dtype,
                head_dim
            );
            false
        }
        Err(_) => {
            println!(
                "[ACCEL-CHECK] ❌ {} {} · flash-attn {:?} d{}: 검증 중 패닉 → fused flash-attn 을 끄고 기존 어텐션 경로를 씁니다",
                backend_name(device),
                arch(device),
                dtype,
                head_dim
            );
            false
        }
    };
    st.fa.insert((dtype, head_dim), ok);
    if !ok {
        disable_flash_attn_backend(device);
    }
    ok
}

/// TurboQuant 압축 KV 캐시를 이 장치·head_dim 에서 써도 되는지.
/// (turbo-quant 빌드·GPU·`CANDLE_TURBO_QUANT` 확인은 호출부 `turbo_kv_active` 가 합니다)
pub fn turbo_quant_ok(device: &Device, head_dim: usize) -> bool {
    if !is_gpu(device) || !checks_enabled() {
        return true;
    }
    let mut g = states();
    let st = g.entry(device.location()).or_default();
    if st.tq_off.is_some() {
        return false;
    }
    if let Some(ok) = st.tq.get(&head_dim) {
        return *ok;
    }
    let res = std::panic::catch_unwind(AssertUnwindSafe(|| tq_check(device, head_dim)));
    let label = super::kv_turbo::kv_formats_label(device);
    let ok = match res {
        Ok(Ok(r)) if r.quant_ok && r.attn_ok => {
            println!(
                "[ACCEL-CHECK] {} {} · turbo-quant {label} d{head_dim}: 통과 (압축 {:.1e} · 복원 {:.1e} · fused 어텐션 {:.1e})",
                backend_name(device),
                arch(device),
                r.quant_err,
                r.dequant_err,
                r.attn_err
            );
            true
        }
        Ok(Ok(r)) if r.quant_ok => {
            // 압축·복원은 맞고 fused 어텐션 커널만 틀림 → 복원 후 flash-attn 으로 계산
            candle_core::accel::update_config(|c| c.turbo_quant.fused_attention = false);
            println!(
                "[ACCEL-CHECK] ⚠️ {} {} · turbo-quant {label} d{head_dim}: fused 어텐션 불일치 ({:.2e} > {:.0e}) → 압축은 유지하고 복원 후 flash-attn 으로 계산합니다",
                backend_name(device),
                arch(device),
                r.attn_err,
                TQ_ATTN_TOL
            );
            true
        }
        Ok(Ok(r)) => {
            let reason = format!(
                "turbo-quant {label} d{head_dim} 압축 결과 불일치 (압축 {:.2e}, 복원 {:.2e})",
                r.quant_err, r.dequant_err
            );
            println!(
                "[ACCEL-CHECK] ❌ {} {} · {reason} → 이 장치에서 TurboQuant KV 캐시를 끄고 기존 FP8 경로를 씁니다",
                backend_name(device),
                arch(device)
            );
            st.tq_off = Some(reason);
            false
        }
        Ok(Err(e)) => {
            let reason = format!("turbo-quant {label} d{head_dim} 검증 실행 실패 ({e})");
            println!(
                "[ACCEL-CHECK] ❌ {} {} · {reason} → 이 장치에서 TurboQuant KV 캐시를 끄고 기존 FP8 경로를 씁니다",
                backend_name(device),
                arch(device)
            );
            st.tq_off = Some(reason);
            false
        }
        Err(_) => {
            let reason = format!("turbo-quant {label} d{head_dim} 검증 중 패닉");
            println!(
                "[ACCEL-CHECK] ❌ {} {} · {reason} → 이 장치에서 TurboQuant KV 캐시를 끄고 기존 FP8 경로를 씁니다",
                backend_name(device),
                arch(device)
            );
            st.tq_off = Some(reason);
            false
        }
    };
    st.tq.insert(head_dim, ok);
    if !ok {
        disable_turbo_quant_kernels(device);
    }
    ok
}

/// 실측 감시를 더 할지 (장치마다 [`PROBE_LIMIT`] 번까지)
pub fn probe_wanted(device: &Device) -> bool {
    if !is_gpu(device) || !checks_enabled() {
        return false;
    }
    let g = states();
    match g.get(&device.location()) {
        Some(st) => st.tq_off.is_none() && st.probe.count < PROBE_LIMIT,
        None => true,
    }
}

/// 실측 감시 결과(압축 캐시 어텐션 vs 원본 어텐션의 상대오차)를 기록합니다.
/// 커널 오동작 수준([`PROBE_BROKEN_REL`] 초과, NaN)이면 이 장치의 TurboQuant 를 끄고 false 를 돌려줍니다.
pub fn record_probe(device: &Device, rel: f32, label: &str) -> bool {
    let broken = !(rel <= PROBE_BROKEN_REL);
    let mut g = states();
    let st = g.entry(device.location()).or_default();
    let p = &mut st.probe;
    p.count += 1;
    p.pending += 1;
    let r = if rel.is_finite() { rel } else { f32::INFINITY };
    p.sum += r as f64;
    p.max = p.max.max(r);
    p.pending_sum += r as f64;
    p.pending_max = p.pending_max.max(r);
    p.label = label.to_string();
    if broken && st.tq_off.is_none() {
        let reason = format!(
            "실측 감시: 압축 캐시 어텐션 상대오차 {:.1}% ({label})",
            rel * 100.0
        );
        println!(
            "[ACCEL-CHECK] ❌ {} {} · {reason} > {:.0}% → 이 장치에서 TurboQuant KV 캐시를 끄고 원래 KV 경로로 돌아갑니다",
            backend_name(device),
            arch(device),
            PROBE_BROKEN_REL * 100.0
        );
        st.tq_off = Some(reason);
        drop(g);
        disable_turbo_quant_kernels(device);
    }
    !broken
}

/// 지난 보고 이후 쌓인 실측 감시 결과를 한 줄로 출력합니다. (모델 forward 끝에서 호출)
pub fn flush_probe_report(device: &Device) {
    if !is_gpu(device) {
        return;
    }
    let mut g = states();
    let Some(st) = g.get_mut(&device.location()) else {
        return;
    };
    let p = &mut st.probe;
    if p.pending == 0 {
        return;
    }
    let mean = p.pending_sum / p.pending as f64;
    let warn = if p.pending_max > PROBE_WARN_REL {
        " ⚠️ 오차가 큽니다. 4bit 형식(CANDLE_TURBO_QUANT_BITS 등)을 쓰고 있다면 지우고 기본 FP8 로 돌아가세요"
    } else {
        ""
    };
    println!(
        "[ACCEL-CHECK] TurboQuant 실측 오차 ({}): {}개 레이어 평균 {:.2}% · 최대 {:.2}% (원본 K/V 어텐션 대비){warn}",
        p.label,
        p.pending,
        mean * 100.0,
        p.pending_max * 100.0
    );
    p.pending = 0;
    p.pending_sum = 0.0;
    p.pending_max = 0.0;
}

/// 이 장치에서 TurboQuant 를 끈 이유 (켜져 있으면 None)
pub fn turbo_quant_off_reason(device: &Device) -> Option<String> {
    states().get(&device.location()).and_then(|st| st.tq_off.clone())
}

/// 진단용: 판정 캐시·전역 설정을 건드리지 않고 자가 검증만 실행해 결과를 한 줄로 돌려줍니다.
/// (CPU 장치에서도 실행됩니다. CPU 에서는 CPU 끼리 비교하므로 검증 코드 자체의 점검용입니다)
pub fn diagnose(device: &Device, dtype: DType, head_dim: usize) -> String {
    let fa = match fa_check(device, dtype, head_dim) {
        Ok((worst, name, cases)) => {
            format!("flash-attn {dtype:?} d{head_dim}: 최대 상대오차 {worst:.2e} ({name}, {cases}개 형태)")
        }
        Err(e) => format!("flash-attn {dtype:?} d{head_dim}: 실행 실패 ({e})"),
    };
    let tq = match tq_check(device, head_dim) {
        Ok(r) => format!(
            "turbo-quant {} d{head_dim}: 압축 {:.2e} · 복원 {:.2e} · fused 어텐션 {:.2e}",
            super::kv_turbo::kv_formats_label(device),
            r.quant_err,
            r.dequant_err,
            r.attn_err
        ),
        Err(e) => format!("turbo-quant d{head_dim}: 실행 실패 ({e})"),
    };
    format!("{fa} | {tq}")
}

/// `a` 와 기준값 `b` 의 상대 오차 ‖a-b‖/‖b‖ (f32, 어느 장치든). NaN/Inf 가 섞이면 Inf.
pub fn rel_err(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?.to_device(a.device())?;
    let d = (&a - &b)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let n = b.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let r = (d / n.max(1e-30)).sqrt();
    Ok(if r.is_finite() { r } else { f32::INFINITY })
}

// ---------------------------------------------------------------------------
// 검증 입력
// ---------------------------------------------------------------------------

/// 결정적인 정규분포 난수
fn randn(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    if s == 0 {
        s = 1;
    }
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (next(), next());
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect()
}

/// `(…, d)` CPU f32 텐서. `outliers` 이면 QK-norm 모델의 키처럼 몇몇 채널에 큰 값을 더합니다.
fn sample(shape: &[usize], seed: u64, scale: f32, outliers: bool) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let d = *shape.last().unwrap_or(&1);
    let mut v = randn(n, seed);
    for x in v.iter_mut() {
        *x *= scale;
    }
    if outliers && d >= 8 {
        let (c1, c2) = (1usize, d / 2 + 3);
        for row in v.chunks_mut(d) {
            row[c1] += 6.0;
            row[c2] -= 4.0;
        }
    }
    Ok(Tensor::from_vec(v, shape, &Device::Cpu)?)
}

// ---------------------------------------------------------------------------
// FlashAttention 자가 검증
// ---------------------------------------------------------------------------

/// 가장 나쁜 상대오차, 그 형태 이름, 검사한 형태 수
fn fa_check(device: &Device, dtype: DType, d: usize) -> Result<(f32, String, usize)> {
    use candle_nn::flash_attn::{
        FlashAttnConfig, flash_attn_bhsd, flash_attn_unfused, flash_attn_with_config,
    };
    // (이름, hq, hkv, lq, lk, causal, bshd 레이아웃)
    let cases: [(&str, usize, usize, usize, usize, bool, bool); 5] = [
        ("prefill 64x64 causal (bshd)", 8, 2, 64, 64, true, true),
        ("chunk 16x80 causal", 8, 4, 16, 80, true, false),
        ("decode 1x300", 8, 2, 1, 300, true, false),
        ("short 3x50 causal", 8, 4, 3, 50, true, false),
        ("vision 40x40 full", 4, 4, 40, 40, false, true),
    ];
    let scale = 1.0 / (d as f32).sqrt();
    let mut worst = 0f32;
    let mut worst_name = String::new();
    for (i, (name, hq, hkv, lq, lk, causal, bshd)) in cases.iter().enumerate() {
        let mut cfg = FlashAttnConfig::new(scale);
        if *causal {
            cfg = cfg.causal();
        }
        let seed = 0xFA00 + (i as u64) * 16 + d as u64;
        // (b, h, s, d) 입력. dtype 으로 반올림한 값을 기준 계산에도 그대로 씁니다.
        let q = sample(&[1, *hq, *lq, d], seed, 2.0, false)?.to_dtype(dtype)?;
        let k = sample(&[1, *hkv, *lk, d], seed + 1, 1.0, true)?.to_dtype(dtype)?;
        let v = sample(&[1, *hkv, *lk, d], seed + 2, 1.0, false)?.to_dtype(dtype)?;
        let bshd_of = |x: &Tensor| -> Result<Tensor> { Ok(x.transpose(1, 2)?.contiguous()?) };
        let reference = flash_attn_unfused(
            &bshd_of(&q.to_dtype(DType::F32)?)?,
            &bshd_of(&k.to_dtype(DType::F32)?)?,
            &bshd_of(&v.to_dtype(DType::F32)?)?,
            &cfg,
        )?;
        let fused = cfg.clone().fused(true);
        let got = if *bshd {
            flash_attn_with_config(
                &bshd_of(&q)?.to_device(device)?,
                &bshd_of(&k)?.to_device(device)?,
                &bshd_of(&v)?.to_device(device)?,
                &fused,
            )?
        } else {
            flash_attn_bhsd(&q.to_device(device)?, &k.to_device(device)?, &v.to_device(device)?, &fused)?
                .transpose(1, 2)?
        };
        let got = got.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.contiguous()?;
        if got.dims() != reference.dims() {
            bail!("{name}: 출력 모양 {:?} != {:?}", got.dims(), reference.dims());
        }
        let e = rel_err(&got, &reference)?;
        if !(e <= worst) {
            worst = e;
            worst_name = name.to_string();
        }
    }
    Ok((worst, worst_name, cases.len()))
}

// ---------------------------------------------------------------------------
// TurboQuant 자가 검증
// ---------------------------------------------------------------------------

struct TqReport {
    quant_ok: bool,
    attn_ok: bool,
    quant_err: f32,
    dequant_err: f32,
    attn_err: f32,
}

#[cfg(feature = "turbo-quant")]
fn tq_check(device: &Device, d: usize) -> Result<TqReport> {
    use candle_nn::flash_attn::{FlashAttnConfig, flash_attn_bhsd};
    use candle_nn::turbo_quant::{TurboKvCache, TurboQuantizer};
    let o = super::kv_turbo::kv_quant_options();
    let (kf, vf) = candle_core::accel::profile(device).turbo_formats(&o);
    let cpu = Device::Cpu;

    // 1) 압축·복원: GPU 커널 vs CPU 코덱. 행마다 크기를 0.05~6.4 배로 바꿔 스케일 처리를 확인합니다.
    let mut quant_err = 0f32;
    let mut dequant_err = 0f32;
    for (fi, (fmt, seed)) in [(kf, o.seed), (vf, o.seed.wrapping_add(2))].into_iter().enumerate() {
        let rows = 48usize;
        let x = sample(&[rows, d], 0x7100 + fi as u64 + d as u64, 1.0, fi == 0)?;
        let row_scale: Vec<f32> = (0..rows).map(|r| 0.05 * (1u32 << (r % 8)) as f32).collect();
        let x = x.broadcast_mul(&Tensor::from_vec(row_scale, (rows, 1), &cpu)?)?;
        let qg = TurboQuantizer::new(fmt, d, o.rotation, seed, device)?;
        let qc = TurboQuantizer::new(fmt, d, o.rotation, seed, &cpu)?;
        let packed_gpu = qg.quantize_rows(&x.to_device(device)?)?;
        let packed_cpu = qc.quantize_rows(&x)?;
        let pg_on_cpu = packed_gpu.to_device(&cpu)?;
        let deq_ref = qc.dequantize_rows(&packed_cpu)?;
        let deq_gpu_packed = qc.dequantize_rows(&pg_on_cpu)?;
        quant_err = quant_err.max(rel_err(&deq_gpu_packed, &deq_ref)?);
        let deq_gpu = qg.dequantize_rows(&packed_gpu)?.to_device(&cpu)?;
        dequant_err = dequant_err.max(rel_err(&deq_gpu, &deq_gpu_packed)?);
    }
    let quant_ok = quant_err <= TQ_QUANT_TOL && dequant_err <= TQ_DEQUANT_TOL;
    if !quant_ok {
        return Ok(TqReport { quant_ok, attn_ok: false, quant_err, dequant_err, attn_err: f32::NAN });
    }

    // 2) 압축 캐시 위 fused 어텐션 vs CPU 에서 복원한 K/V 로 계산한 기준값 (디코딩, 청크)
    let dt = model_dtype(device);
    let scale = 1.0 / (d as f32).sqrt();
    let mut attn_err = 0f32;
    for (ci, (hq, hkv, lq, lk)) in [(8usize, 4usize, 1usize, 300usize), (8, 4, 8, 100)].into_iter().enumerate() {
        let seed = 0x7A00 + ci as u64 * 8 + d as u64;
        let k = sample(&[1, hkv, lk, d], seed, 1.0, true)?.to_dtype(dt)?;
        let v = sample(&[1, hkv, lk, d], seed + 1, 1.0, false)?.to_dtype(dt)?;
        let q = sample(&[1, hq, lq, d], seed + 2, 2.0, false)?.to_dtype(dt)?;
        let mut cg = TurboKvCache::with_formats(kf, vf, d, o.rotation, o.seed, 64, device)?;
        let mut cc = TurboKvCache::with_formats(kf, vf, d, o.rotation, o.seed, 64, &cpu)?;
        // 두 번에 나눠 붙여 버퍼 확장·이어 붙이기도 함께 확인합니다.
        let cut = lk / 3;
        for (s, len) in [(0, cut), (cut, lk - cut)] {
            let ks = k.narrow(2, s, len)?.contiguous()?;
            let vs = v.narrow(2, s, len)?.contiguous()?;
            cg.append(&ks.to_device(device)?, &vs.to_device(device)?)?;
            cc.append(&ks.to_dtype(DType::F32)?, &vs.to_dtype(DType::F32)?)?;
        }
        let cfg = FlashAttnConfig::new(scale).causal();
        let got = cg.attention(&q.to_device(device)?, &cfg.clone().fused(true))?;
        let Some((kd, vd)) = cc.keys_values()? else {
            bail!("CPU 기준 캐시가 비어 있습니다");
        };
        let reference = flash_attn_bhsd(&q.to_dtype(DType::F32)?, &kd, &vd, &cfg.clone().fused(false))?;
        attn_err = attn_err.max(rel_err(&got.to_device(&cpu)?, &reference)?);
    }
    Ok(TqReport { quant_ok, attn_ok: attn_err <= TQ_ATTN_TOL, quant_err, dequant_err, attn_err })
}

#[cfg(not(feature = "turbo-quant"))]
fn tq_check(_device: &Device, _d: usize) -> Result<TqReport> {
    Ok(TqReport { quant_ok: true, attn_ok: true, quant_err: 0.0, dequant_err: 0.0, attn_err: 0.0 })
}

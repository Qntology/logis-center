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
/// Part 5 (Pack C): 융합 디코더·GDN·lm_head 커널 스위치 (`off` 또는 `-rope_kv,-silu_mul,...`)
pub const FUSED_OPS_ENV: &str = "LOGIS_FUSED_OPS";

/// Part 5 (Pack C): 자가 검증 및 자동 폴백 대상 융합 연산 5종.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusedOpKind {
    ResidualRmsNorm,
    SiluMul,
    RopeKvWrite,
    GatedDeltaRule,
    LmHeadArgmaxCs,
}

impl FusedOpKind {
    pub const ALL: [FusedOpKind; 5] = [
        FusedOpKind::ResidualRmsNorm,
        FusedOpKind::SiluMul,
        FusedOpKind::RopeKvWrite,
        FusedOpKind::GatedDeltaRule,
        FusedOpKind::LmHeadArgmaxCs,
    ];

    pub fn token(self) -> &'static str {
        match self {
            FusedOpKind::ResidualRmsNorm => "res_rmsnorm",
            FusedOpKind::SiluMul => "silu_mul",
            FusedOpKind::RopeKvWrite => "rope_kv",
            FusedOpKind::GatedDeltaRule => "gdn",
            FusedOpKind::LmHeadArgmaxCs => "lm_head",
        }
    }
}

/// Gate 5-A 테스트용 고장 주입 모드 (`set_fused_op_fault`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultMode {
    ReturnError,
    Panic,
    BadOutput,
}

/// 프로세스 내 테스트용 `LOGIS_FUSED_OPS` 강제 스위치.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedOpsOverride {
    AllOff,
    AllOn,
}
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
    /// Part 5: 융합 디코더/GDN/lm_head 5종 자가 검증 결과: (kind, dtype, dim) → 통과 여부
    fused_ops: HashMap<(FusedOpKind, DType, usize), bool>,
    probe: ProbeStats,
}

static FUSED_OPS_OVERRIDE: OnceLock<Mutex<Option<FusedOpsOverride>>> = OnceLock::new();
static FUSED_OP_FAULTS: OnceLock<Mutex<HashMap<FusedOpKind, FaultMode>>> = OnceLock::new();

fn fused_override_lock() -> MutexGuard<'static, Option<FusedOpsOverride>> {
    FUSED_OPS_OVERRIDE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

fn fault_map_lock() -> MutexGuard<'static, HashMap<FusedOpKind, FaultMode>> {
    FUSED_OP_FAULTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

/// 테스트용: `LOGIS_FUSED_OPS` 환경변수 대신 프로세스 내 강제값 설정 (`None` = 환경변수 사용).
pub fn set_fused_ops_override(v: Option<FusedOpsOverride>) {
    *fused_override_lock() = v;
}

/// Gate 5-A 테스트용: 특정 융합 커널 자가 검증에 고장을 주입합니다 (`None` = 해제).
pub fn set_fused_op_fault(kind: FusedOpKind, mode: Option<FaultMode>) {
    let mut g = fault_map_lock();
    match mode {
        Some(m) => {
            g.insert(kind, m);
        }
        None => {
            g.remove(&kind);
        }
    }
}

/// 테스트용: 융합 커널 자가 검증 캐시만 초기화합니다.
pub fn reset_fused_op_cache() {
    let mut g = states();
    for st in g.values_mut() {
        st.fused_ops.clear();
    }
}

fn fused_op_env_allowed(kind: FusedOpKind) -> bool {
    if let Some(ov) = *fused_override_lock() {
        return matches!(ov, FusedOpsOverride::AllOn);
    }
    static PARSED: OnceLock<(bool, Vec<String>)> = OnceLock::new();
    let (all_off, excluded) = PARSED.get_or_init(|| {
        let raw = std::env::var(FUSED_OPS_ENV).unwrap_or_default();
        let trimmed = raw.trim().to_ascii_lowercase();
        if matches!(trimmed.as_str(), "0" | "off" | "false" | "no" | "disable" | "disabled") {
            return (true, Vec::new());
        }
        let mut excl = Vec::new();
        for part in trimmed.split([',', ';', ' ']).map(str::trim).filter(|s| !s.is_empty()) {
            if let Some(rest) = part.strip_prefix('-') {
                excl.push(rest.to_string());
            }
        }
        (false, excl)
    });
    if *all_off {
        return false;
    }
    !excluded.iter().any(|e| e == kind.token())
}

/// CPU/Vulkan host-shared GDN 경로에서 `LOGIS_FUSED_OPS=off` 및 고장 주입을 일관되게 존중하기 위한 헬퍼.
pub fn fused_op_allowed_cpu(kind: FusedOpKind, device: &Device, dtype: DType, dim: usize) -> bool {
    if !fused_op_env_allowed(kind) {
        return false;
    }
    if fault_map_lock().contains_key(&kind) {
        return fused_op_ok(device, kind, dtype, dim);
    }
    true
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
        Device::Metal(_) => c.flash_attn.backends.metal = false,
        _ => {}
    });
}

/// TurboQuant GPU 커널을 끕니다. (남아 있는 압축 데이터는 CPU 코덱으로 정확하게 풀립니다)
fn disable_turbo_quant_kernels(device: &Device) {
    candle_core::accel::update_config(|c| match device {
        Device::Cuda(_) => c.turbo_quant.backends.cuda = false,
        Device::Rocm(_) => c.turbo_quant.backends.rocm = false,
        Device::Vulkan(_) => c.turbo_quant.backends.vulkan = false,
        Device::Metal(_) => c.turbo_quant.backends.metal = false,
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

/// Part 5 (Pack C): 융합 디코더/GDN/lm_head 커널을 이 장치·dtype·차원에서 써도 되는지.
///
/// - CUDA/ROCm/Vulkan GPU 에서 활성화됩니다 (GatedDeltaRule 은 Metal 포함; CPU 는 `embed-cpu-kernels`
///   패킹 경로를 유지 — 단 Gate 5-A 고장 주입 테스트가 활성화된 경우 CPU 에서도 자가 검증 + 폴백을 실행).
/// - `LOGIS_FUSED_OPS=off` 또는 `-<kind>` 로 비활성화할 수 있습니다.
/// - `(kind, dtype, dim)` 조합마다 최초 1회 CPU FP32 기준 대비 자가 검증을 `catch_unwind` 로 감싸
///   실행하며, 실패·패닉·오차 초과 시 해당 키를 영구 `false` 로 캐시해 기존 텐서 경로로 안전 폴백합니다.
pub fn fused_op_ok(device: &Device, kind: FusedOpKind, dtype: DType, dim: usize) -> bool {
    if !fused_op_env_allowed(kind) {
        return false;
    }
    let has_fault = fault_map_lock().contains_key(&kind);
    let supported_dev = match kind {
        FusedOpKind::GatedDeltaRule => {
            matches!(
                device,
                Device::Cuda(_) | Device::Rocm(_) | Device::Metal(_) | Device::Vulkan(_)
            )
        }
        _ => matches!(device, Device::Cuda(_) | Device::Rocm(_) | Device::Vulkan(_)),
    };
    if !supported_dev && !has_fault {
        return false;
    }
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return false;
    }
    if !checks_enabled() && !has_fault {
        return true;
    }

    let key = (kind, dtype, dim);
    {
        let g = states();
        if let Some(ok) = g.get(&device.location()).and_then(|st| st.fused_ops.get(&key)) {
            return *ok;
        }
    }

    let res = std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_fused_op(device, kind, dtype, dim)
    }));
    let tol = fused_op_tol(kind, dtype);
    let ok = match res {
        Ok(Ok(err)) if err <= tol => {
            println!(
                "[ACCEL-CHECK] {} {} · fused-op {} {:?} d{}: 통과 (최대 상대오차 {:.1e})",
                backend_name(device),
                arch(device),
                kind.token(),
                dtype,
                dim,
                err
            );
            true
        }
        Ok(Ok(err)) => {
            println!(
                "[ACCEL-CHECK] ❌ {} {} · fused-op {} {:?} d{}: 오차 초과 ({:.2e} > {:.0e}) → 기존 경로로 폴백합니다",
                backend_name(device),
                arch(device),
                kind.token(),
                dtype,
                dim,
                err,
                tol
            );
            false
        }
        Ok(Err(e)) => {
            println!(
                "[ACCEL-CHECK] ❌ {} {} · fused-op {} {:?} d{}: 검증 실행 실패 ({e}) → 기존 경로로 폴백합니다",
                backend_name(device),
                arch(device),
                kind.token(),
                dtype,
                dim
            );
            false
        }
        Err(_) => {
            println!(
                "[ACCEL-CHECK] ❌ {} {} · fused-op {} {:?} d{}: 검증 중 패닉 → 기존 경로로 폴백합니다",
                backend_name(device),
                arch(device),
                kind.token(),
                dtype,
                dim
            );
            false
        }
    };
    let mut g = states();
    g.entry(device.location()).or_default().fused_ops.insert(key, ok);
    ok
}

fn fused_op_tol(kind: FusedOpKind, dtype: DType) -> f32 {
    match (kind, dtype) {
        (FusedOpKind::LmHeadArgmaxCs, _) => 0.0, // argmax + CS rescue mask must be exact (0 mismatches)
        (_, DType::F32) => 1e-4,
        (_, DType::F16) => 1.5e-2,
        (_, DType::BF16) => 2.5e-2,
        _ => 1e-4,
    }
}

/// Returns all acceleration checks that fell back on `device` (for `ReflectionRecord`).
pub fn recorded_fallbacks(device: &Device) -> Vec<String> {
    let mut out = Vec::new();
    let g = states();
    if let Some(st) = g.get(&device.location()) {
        for ((dt, hd), &ok) in &st.fa {
            if !ok {
                out.push(format!("flash_attn:{dt:?}:d{hd}"));
            }
        }
        if let Some(reason) = &st.tq_off {
            out.push(format!("turbo_quant:{reason}"));
        }
        for ((kind, dt, dim), &ok) in &st.fused_ops {
            if !ok {
                out.push(format!("fused:{}:{dt:?}:d{dim}", kind.token()));
            }
        }
    }
    out.sort();
    out
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AccelProbeCache {
    pub binary_key: String,
    pub backend: String,
    pub arch: String,
    pub passed_ops: Vec<String>,
    pub failed_ops: Vec<String>,
}

fn accel_probe_cache_path() -> std::path::PathBuf {
    logis_schema::paths::get_app_dir()
        .join("cache")
        .join("accel_check_v1.json")
}

fn binary_fingerprint() -> String {
    let Ok(exe) = std::env::current_exe() else {
        return "unknown".into();
    };
    let Ok(meta) = std::fs::metadata(&exe) else {
        return exe.to_string_lossy().into_owned();
    };
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{}:{}:{}", exe.file_name().and_then(|s| s.to_str()).unwrap_or("bin"), meta.len(), mtime)
}

pub fn load_probe_cache(backend: &str, arch_name: &str) -> Option<AccelProbeCache> {
    let raw = std::fs::read_to_string(accel_probe_cache_path()).ok()?;
    let cached: AccelProbeCache = serde_json::from_str(&raw).ok()?;
    if cached.binary_key == binary_fingerprint()
        && cached.backend == backend
        && cached.arch == arch_name
    {
        Some(cached)
    } else {
        None
    }
}

pub fn save_probe_cache(cache: &AccelProbeCache) {
    let path = accel_probe_cache_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(cache) {
        let _ = std::fs::write(path, json);
    }
}

fn write_probe_inflight(backend: &str, gpu_id: usize, op: &str) {
    let path = logis_schema::paths::get_app_dir()
        .join("cache")
        .join(format!("probe_inflight_{backend}_{gpu_id}.json"));
    if let Some(p) = path.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    let _ = std::fs::write(path, format!("{{\"backend\":\"{backend}\",\"gpu_id\":{gpu_id},\"op\":\"{op}\"}}"));
}

fn clear_probe_inflight(backend: &str, gpu_id: usize) {
    let path = logis_schema::paths::get_app_dir()
        .join("cache")
        .join(format!("probe_inflight_{backend}_{gpu_id}.json"));
    let _ = std::fs::remove_file(path);
}

/// Subprocess probe entrypoint (`--logis-accel-probe <backend> <gpu_id>`).
/// Returns `Some(exit_code)` if the current process was launched as an isolated accel probe.
pub fn maybe_run_cli_probe(args: &[String]) -> Option<i32> {
    let pos = args.iter().position(|a| a == "--logis-accel-probe")?;
    let backend = args.get(pos + 1).map(String::as_str).unwrap_or("cpu");
    let gpu_id: usize = args
        .get(pos + 2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let dev_res: anyhow::Result<Device> = match backend {
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(gpu_id).map_err(Into::into),
        #[cfg(feature = "rocm")]
        "rocm" => crate::utils::new_gpu_device(gpu_id)
            .ok_or_else(|| anyhow::anyhow!("failed to open ROCm device {gpu_id}")),
        #[cfg(feature = "vulkan")]
        "vulkan" => {
            let d = Device::new_vulkan(gpu_id)?;
            if let Device::Vulkan(ref vd) = d {
                if vd.device_type() == "cpu" && std::env::var_os("LOGIS_REQUIRE_GPU").is_some() {
                    anyhow::bail!("software Vulkan adapter (type=cpu) rejected under LOGIS_REQUIRE_GPU=1");
                }
            }
            Ok(d)
        }
        #[cfg(feature = "metal")]
        "metal" => Device::new_metal(gpu_id).map_err(Into::into),
        other => Err(anyhow::anyhow!("unsupported backend '{other}' in this build")),
    };
    let dev = match dev_res {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[ACCEL-PROBE] ERROR opening {backend}:{gpu_id}: {e}");
            return Some(3);
        }
    };
    let dt = model_dtype(&dev);
    let mut passed_ops = Vec::new();
    let mut failed_ops = Vec::new();
    for kind in FusedOpKind::ALL {
        let token = format!("fused:{}", kind.token());
        write_probe_inflight(backend, gpu_id, &token);
        let tol = fused_op_tol(kind, dt);
        match check_fused_op(&dev, kind, dt, 128) {
            Ok(err) if err <= tol => passed_ops.push(token),
            _ => failed_ops.push(token),
        }
    }
    for &hd in &[64usize, 72, 128, 256] {
        let token = format!("flash_attn:{dt:?}:d{hd}");
        write_probe_inflight(backend, gpu_id, &token);
        if flash_attn_ok(&dev, dt, hd) {
            passed_ops.push(token);
        } else {
            failed_ops.push(token);
        }
    }
    clear_probe_inflight(backend, gpu_id);
    let cache = AccelProbeCache {
        binary_key: binary_fingerprint(),
        backend: format!("{}:{gpu_id}", backend_name(&dev)),
        arch: arch(&dev),
        passed_ops,
        failed_ops: failed_ops.clone(),
    };
    save_probe_cache(&cache);
    println!("[ACCEL-PROBE] backend={} passed={} failed={:?}", cache.backend, cache.passed_ops.len(), failed_ops);
    let exit_code = if !failed_ops.is_empty() && std::env::var_os("LOGIS_STRICT_ACCEL").is_some() {
        1
    } else {
        0
    };
    crate::utils::rocm_exit_now_if_used(exit_code as u32);
    Some(exit_code)
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
    let mut fused_parts = Vec::with_capacity(FusedOpKind::ALL.len());
    for kind in FusedOpKind::ALL {
        let d = match kind {
            FusedOpKind::RopeKvWrite | FusedOpKind::GatedDeltaRule => head_dim.max(32),
            _ => 128,
        };
        let s = match std::panic::catch_unwind(AssertUnwindSafe(|| check_fused_op(device, kind, dtype, d))) {
            Ok(Ok(e)) => format!("{}={:.1e}", kind.token(), e),
            Ok(Err(err)) => format!("{}=ERR({err})", kind.token()),
            Err(_) => format!("{}=PANIC", kind.token()),
        };
        fused_parts.push(s);
    }
    format!("{fa} | {tq} | fused-ops: {}", fused_parts.join(", "))
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
    // 🌟 [PRC-11] 긴 문맥 split-KV 디코드(1x4096) 경로까지 자가 검증에 포함합니다.
    let cases: [(&str, usize, usize, usize, usize, bool, bool); 6] = [
        ("prefill 64x64 causal (bshd)", 8, 2, 64, 64, true, true),
        ("chunk 16x80 causal", 8, 4, 16, 80, true, false),
        ("decode 1x300", 8, 2, 1, 300, true, false),
        ("decode 1x4096 split-kv", 8, 2, 1, 4096, true, false),
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

// ---------------------------------------------------------------------------
// Part 5 (Pack C): 융합 커널 5종 자가 검증
// ---------------------------------------------------------------------------

fn check_fused_op(device: &Device, kind: FusedOpKind, dtype: DType, dim: usize) -> Result<f32> {
    if let Some(mode) = fault_map_lock().get(&kind).copied() {
        match mode {
            FaultMode::ReturnError => bail!("injected fault (ReturnError) for {}", kind.token()),
            FaultMode::Panic => {
                // Under `panic = "abort"` (release profile), panicking would abort the process
                // instead of falling back cleanly; convert to Result::Err so fallback stays safe.
                if cfg!(panic = "abort") {
                    bail!("injected fault (Panic->Err under panic=abort) for {}", kind.token());
                } else {
                    panic!("injected fault (Panic) for {}", kind.token());
                }
            }
            FaultMode::BadOutput => return Ok(f32::INFINITY),
        }
    }
    match kind {
        FusedOpKind::ResidualRmsNorm => check_residual_rmsnorm(device, dtype, dim),
        FusedOpKind::SiluMul => check_silu_mul(device, dtype, dim),
        FusedOpKind::RopeKvWrite => check_rope_kv_write(device, dtype, dim),
        FusedOpKind::GatedDeltaRule => check_gated_delta_rule(device, dtype, dim),
        FusedOpKind::LmHeadArgmaxCs => check_lm_head_argmax_cs(device, dtype, dim),
    }
}

fn check_residual_rmsnorm(device: &Device, dtype: DType, dim: usize) -> Result<f32> {
    let d = dim.clamp(16, 512);
    let rows = 4usize;
    let x_cpu = sample(&[rows, d], 0x5101, 1.0, false)?.to_dtype(dtype)?;
    let dl_cpu = sample(&[rows, d], 0x5102, 0.5, false)?.to_dtype(dtype)?;
    let w_cpu = sample(&[d], 0x5103, 0.3, false)?;
    let x_dev = x_cpu.to_device(device)?;
    let dl_dev = dl_cpu.to_device(device)?;
    let w_dev = w_cpu.to_device(device)?;

    let mut worst = 0.0f32;
    for plus_one in [false, true] {
        let (ref_r, ref_y) = candle_core::decoder_ops::residual_rmsnorm(
            &x_cpu.to_dtype(DType::F32)?,
            Some(&dl_cpu.to_dtype(DType::F32)?),
            &w_cpu,
            1e-6,
            plus_one,
        )?;
        let (got_r, got_y) = candle_core::decoder_ops::residual_rmsnorm(
            &x_dev,
            Some(&dl_dev),
            &w_dev,
            1e-6,
            plus_one,
        )?;
        worst = worst.max(rel_err(&got_r.to_device(&Device::Cpu)?, &ref_r)?);
        worst = worst.max(rel_err(&got_y.to_device(&Device::Cpu)?, &ref_y)?);
    }
    Ok(worst)
}

fn check_silu_mul(device: &Device, dtype: DType, dim: usize) -> Result<f32> {
    let inter = dim.clamp(16, 512);
    let gu_cpu = sample(&[1, 3, 2 * inter], 0x5201, 1.0, false)?.to_dtype(dtype)?;
    let gu_dev = gu_cpu.to_device(device)?;
    // Zero-copy strided gate_up view 검증
    let got = candle_core::decoder_ops::silu_mul(
        &gu_dev.narrow(2, 0, inter)?,
        &gu_dev.narrow(2, inter, inter)?,
    )?;
    let gu_f32 = gu_cpu.to_dtype(DType::F32)?;
    let reference = candle_core::decoder_ops::silu_mul(
        &gu_f32.narrow(2, 0, inter)?,
        &gu_f32.narrow(2, inter, inter)?,
    )?;
    rel_err(&got.to_device(&Device::Cpu)?, &reference)
}

fn check_rope_kv_write(device: &Device, dtype: DType, dim: usize) -> Result<f32> {
    use candle_core::accel::decoder_ops::{HeadLayout, RopeMode};
    let hd = (dim.clamp(16, 256) / 2) * 2;
    let rot = ((hd / 2) / 2) * 2;
    let (b, hq, hkv, t, past, max_seq) = (1usize, 4usize, 2usize, 2usize, 3usize, 8usize);
    let q_cpu = sample(&[b, t, hq, hd], 0x5301, 1.0, false)?.to_dtype(dtype)?;
    let k_cpu = sample(&[b, t, hkv, hd], 0x5302, 1.0, false)?.to_dtype(dtype)?;
    let v_cpu = sample(&[b, t, hkv, hd], 0x5303, 1.0, false)?.to_dtype(dtype)?;
    let qnw = sample(&[hd], 0x5304, 0.2, false)?;
    let knw = sample(&[hd], 0x5305, 0.2, false)?;
    let cos = sample(&[t, rot / 2], 0x5306, 0.5, false)?;
    let sin = sample(&[t, rot / 2], 0x5307, 0.5, false)?;

    let kc_ref = Tensor::zeros((b, hkv, max_seq, hd), DType::F32, &Device::Cpu)?;
    let vc_ref = Tensor::zeros((b, hkv, max_seq, hd), DType::F32, &Device::Cpu)?;
    let q_ref = candle_core::decoder_ops::rope_kv_write(
        &q_cpu.to_dtype(DType::F32)?,
        &k_cpu.to_dtype(DType::F32)?,
        &v_cpu.to_dtype(DType::F32)?,
        HeadLayout::Bshd,
        HeadLayout::Bhsd,
        Some(&qnw),
        Some(&knw),
        &cos,
        &sin,
        &kc_ref,
        &vc_ref,
        past,
        1e-6,
        RopeMode::NeoX,
        true,
    )?;

    let kc_dev = Tensor::zeros((b, hkv, max_seq, hd), dtype, device)?;
    let vc_dev = Tensor::zeros((b, hkv, max_seq, hd), dtype, device)?;
    let q_got = candle_core::decoder_ops::rope_kv_write(
        &q_cpu.to_device(device)?,
        &k_cpu.to_device(device)?,
        &v_cpu.to_device(device)?,
        HeadLayout::Bshd,
        HeadLayout::Bhsd,
        Some(&qnw.to_device(device)?),
        Some(&knw.to_device(device)?),
        &cos.to_device(device)?,
        &sin.to_device(device)?,
        &kc_dev,
        &vc_dev,
        past,
        1e-6,
        RopeMode::NeoX,
        true,
    )?;

    let kc_host = kc_dev.to_device(&Device::Cpu)?;
    let vc_host = vc_dev.to_device(&Device::Cpu)?;
    // 슬롯 격리 확인: 0..past 및 past+t..max_seq 는 0 이어야 함
    let untouched_k = kc_host.narrow(2, 0, past)?.to_dtype(DType::F32)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    let untouched_k2 = kc_host.narrow(2, past + t, max_seq - past - t)?.to_dtype(DType::F32)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    if untouched_k != 0.0 || untouched_k2 != 0.0 {
        bail!("rope_kv_write corrupted untouched KV cache slots ({untouched_k}, {untouched_k2})");
    }
    let mut worst = rel_err(&q_got.to_device(&Device::Cpu)?, &q_ref)?;
    worst = worst.max(rel_err(&kc_host, &kc_ref)?);
    worst = worst.max(rel_err(&vc_host, &vc_ref)?);
    Ok(worst)
}

fn check_gated_delta_rule(device: &Device, dtype: DType, dim: usize) -> Result<f32> {
    use candle_core::delta_rule::{fused_gated_delta_rule, fused_gated_rms_norm, GdnConfig};
    let d = dim.clamp(16, 64);
    let cfg = GdnConfig::new(2, 4, d, d);
    let (b, t) = (1usize, 3usize);
    let c = cfg.conv_dim();
    let mixed = sample(&[b, t, c], 0x5401, 0.4, false)?.to_dtype(dtype)?;
    let cw = sample(&[c, cfg.conv_kernel], 0x5402, 0.2, false)?.to_dtype(dtype)?;
    let a = sample(&[b, t, cfg.num_v_heads], 0x5403, 0.3, false)?.to_dtype(dtype)?;
    let bv = sample(&[b, t, cfg.num_v_heads], 0x5404, 0.3, false)?.to_dtype(dtype)?;
    let neg_a = sample(&[cfg.num_v_heads], 0x5405, 0.1, false)?.abs()?.neg()?;
    let dt_bias = sample(&[cfg.num_v_heads], 0x5406, 0.1, false)?;

    let ref_o = fused_gated_delta_rule(
        &mixed.to_dtype(DType::F32)?,
        &cw.to_dtype(DType::F32)?,
        None,
        None,
        &a.to_dtype(DType::F32)?,
        &bv.to_dtype(DType::F32)?,
        &neg_a,
        &dt_bias,
        None,
        &cfg,
    )?;
    let got_o = fused_gated_delta_rule(
        &mixed.to_device(device)?,
        &cw.to_device(device)?,
        None,
        None,
        &a.to_device(device)?,
        &bv.to_device(device)?,
        &neg_a.to_device(device)?,
        &dt_bias.to_device(device)?,
        None,
        &cfg,
    )?;
    let mut worst = rel_err(&got_o.out.to_device(&Device::Cpu)?, &ref_o.out)?;
    worst = worst.max(rel_err(&got_o.conv_state.to_device(&Device::Cpu)?, &ref_o.conv_state)?);
    worst = worst.max(rel_err(&got_o.recurrent_state.to_device(&Device::Cpu)?, &ref_o.recurrent_state)?);

    // Gated RMSNorm 도 함께 검증
    let z = sample(&[b * t * cfg.num_v_heads, d], 0x5407, 0.5, false)?.to_dtype(dtype)?;
    let nw = sample(&[d], 0x5408, 0.2, false)?.to_dtype(dtype)?;
    let g_ref = fused_gated_rms_norm(
        &ref_o.out.reshape(((), d))?,
        &z.to_dtype(DType::F32)?,
        &nw.to_dtype(DType::F32)?,
        1e-6,
    )?;
    let g_got = fused_gated_rms_norm(
        &got_o.out.reshape(((), d))?,
        &z.to_device(device)?,
        &nw.to_device(device)?,
        1e-6,
    )?;
    worst = worst.max(rel_err(&g_got.to_device(&Device::Cpu)?, &g_ref)?);
    Ok(worst)
}

fn check_lm_head_argmax_cs(device: &Device, dtype: DType, dim: usize) -> Result<f32> {
    use candle_core::lm_head::{fused_linear_argmax_cs, logits_argmax_cs, CsGuardParams};
    let k = dim.clamp(32, 128);
    let (v_core, v_rest, m) = (64usize, 64usize, 4usize);
    let mut w_core_v = randn(v_core * k, 0x5501);
    for x in &mut w_core_v {
        *x = *x * 0.1 + 0.2;
    }
    let mut w_rest_v = randn(v_rest * k, 0x5502);
    for x in &mut w_rest_v {
        *x = *x * 0.1 + 0.2;
    }
    // 행 0,1: CORE 고신뢰 승자 (rescue=0), 행 2,3: REST 스파이크 승자 (rescue=1)
    let mut h_v = vec![0.0f32; m * k];
    for kk in 0..k {
        let spike_r = if (kk / 4) % 2 == 0 { 0.20 } else { -0.20 };
        w_rest_v[7 * k + kk] = 0.2 + spike_r;
        let spike_c = if kk % 2 == 0 { 0.45 } else { -0.45 };
        w_core_v[3 * k + kk] = 0.2 + spike_c;
        h_v[0 * k + kk] = spike_c * 4.0;
        h_v[1 * k + kk] = spike_c * 3.5;
        h_v[2 * k + kk] = spike_r * 5.0;
        h_v[3 * k + kk] = spike_r * 6.0;
    }

    let w_core_cpu = Tensor::from_vec(w_core_v, (v_core, k), &Device::Cpu)?.to_dtype(dtype)?;
    let w_rest_cpu = Tensor::from_vec(w_rest_v, (v_rest, k), &Device::Cpu)?.to_dtype(dtype)?;
    let h_cpu = Tensor::from_vec(h_v, (m, k), &Device::Cpu)?.to_dtype(dtype)?;
    let core_ids_v: Vec<u32> = (0..v_core as u32).map(|i| i * 2).collect();
    let core_ids_cpu = Tensor::from_vec(core_ids_v, (v_core,), &Device::Cpu)?;

    let guard_cpu = CsGuardParams::from_rest_weight(&w_rest_cpu, None, 1e-3)?;
    let guard_dev = CsGuardParams {
        mu_rest: guard_cpu.mu_rest.to_device(device)?,
        w_rest_bound: guard_cpu.w_rest_bound,
        b_rest_max: guard_cpu.b_rest_max,
    };

    let ref_a = fused_linear_argmax_cs(
        &h_cpu.to_dtype(DType::F32)?,
        &w_core_cpu.to_dtype(DType::F32)?,
        None,
        Some(&core_ids_cpu),
        Some(&guard_cpu),
    )?;
    let got_a = fused_linear_argmax_cs(
        &h_cpu.to_device(device)?,
        &w_core_cpu.to_device(device)?,
        None,
        Some(&core_ids_cpu.to_device(device)?),
        Some(&guard_dev),
    )?;

    let ref_ids = ref_a.ids.to_vec1::<u32>()?;
    let (got_ids, got_mask_opt) = got_a.download_ids_and_mask()?;
    let got_mask = got_mask_opt.ok_or_else(|| anyhow::anyhow!("lm_head Mode A missing rescue_mask"))?;
    if ref_ids != got_ids {
        bail!("lm_head Mode A ids mismatch: got {got_ids:?} != ref {ref_ids:?}");
    }
    if got_mask[2] != 1 || got_mask[3] != 1 {
        bail!("lm_head Mode A CS guard false negative on REST spike rows: {got_mask:?}");
    }

    // Quantized Mode A (CUDA / ROCm W8A16 fused_quant_argmax_cs, 0-byte logits)
    if (device.is_cuda() || device.is_rocm()) && k % 32 == 0 {
        use candle_core::lm_head::fused_quant_argmax_cs;
        use candle_core::quantized::{GgmlDType, QTensor};
        let q_cpu = QTensor::quantize(&w_core_cpu.to_dtype(DType::F32)?, GgmlDType::Q8_0)?;
        let q_dev = q_cpu.dequantize(&Device::Cpu)?;
        let q_ref = fused_linear_argmax_cs(
            &h_cpu.to_dtype(DType::F32)?,
            &q_dev,
            None,
            Some(&core_ids_cpu),
            Some(&guard_cpu),
        )?;
        let q_ref_ids = q_ref.ids.to_vec1::<u32>()?;
        let q_gpu = QTensor::quantize(&w_core_cpu.to_dtype(DType::F32)?.to_device(device)?, GgmlDType::Q8_0)?;
        let got_qa = fused_quant_argmax_cs(
            &h_cpu.to_device(device)?,
            &q_gpu,
            None,
            Some(&core_ids_cpu.to_device(device)?),
            Some(&guard_dev),
        )?;
        let (got_qa_ids, got_qa_mask_opt) = got_qa.download_ids_and_mask()?;
        let got_qa_mask = got_qa_mask_opt
            .ok_or_else(|| anyhow::anyhow!("lm_head Quantized Mode A missing rescue_mask"))?;
        if got_qa_ids != q_ref_ids {
            bail!("lm_head Quantized Mode A ids mismatch: got {got_qa_ids:?} != ref {q_ref_ids:?}");
        }
        if got_qa_mask[2] != 1 || got_qa_mask[3] != 1 {
            bail!("lm_head Quantized Mode A CS guard false negative: {got_qa_mask:?}");
        }
    }

    // Mode B (F32 sub_logits + dtype hidden)
    let sub_logits = h_cpu
        .to_dtype(DType::F32)?
        .matmul(&w_core_cpu.to_dtype(DType::F32)?.t()?)?
        .to_device(device)?;
    let got_b = logits_argmax_cs(
        &sub_logits,
        Some(&h_cpu.to_device(device)?),
        Some(&core_ids_cpu.to_device(device)?),
        Some(&guard_dev),
    )?;
    let (got_b_ids, got_b_mask_opt) = got_b.download_ids_and_mask()?;
    let got_b_mask = got_b_mask_opt.ok_or_else(|| anyhow::anyhow!("lm_head Mode B missing rescue_mask"))?;
    if got_b_ids != ref_ids {
        bail!("lm_head Mode B ids mismatch: got {got_b_ids:?} != ref {ref_ids:?}");
    }
    if got_b_mask[2] != 1 || got_b_mask[3] != 1 {
        bail!("lm_head Mode B CS guard false negative: {got_b_mask:?}");
    }
    Ok(0.0)
}

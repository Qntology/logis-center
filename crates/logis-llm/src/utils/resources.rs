use sysinfo::{System, RefreshKind, CpuRefreshKind, MemoryRefreshKind};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use once_cell::sync::Lazy;
use std::time::{Duration, Instant};
use tokio::time::sleep;

use anyhow::Result;



pub async fn wait_for_resources_settled(target_vram_mb: u64, target_ram_mb: u64, cancellation_token: Option<&Arc<AtomicBool>>, target_gpu_id: u32) -> Result<()> {
    let mut sys = System::new_with_specifics(
        RefreshKind::new().with_memory(MemoryRefreshKind::everything()),
    );
    let gpu_probe = crate::utils::GpuMemProbe::new();
    
    let target_vram_bytes = target_vram_mb * 1024 * 1024;
    let target_ram_bytes = target_ram_mb * 1024 * 1024;

    let mut last_vram = 0;
    let mut stable_ticks = 0;
    let mut last_report = std::time::Instant::now();
    let start_time = std::time::Instant::now();

    println!("[RESOURCE-WATCH] Monitoring recovery (Target VRAM > {}MB) on GPU {}...", target_vram_mb, target_gpu_id);

    loop {
        if let Some(token) = cancellation_token {
            if token.load(Ordering::Relaxed) {
                return Err(anyhow::anyhow!("Task cancelled during resource wait"));
            }
        }

        sys.refresh_memory(); 
        let current_ram = sys.available_memory();
        let mut current_vram = 0;
        let mut has_gpu = false;

        if let Some(free) = gpu_probe.free_bytes(target_gpu_id as usize) {
            current_vram = free;
            has_gpu = true;
        }

        let meets_vram = !has_gpu || current_vram >= target_vram_bytes;
        let meets_ram = current_ram >= target_ram_bytes;
        
        if meets_vram && meets_ram {
            break; // Perfect state reached
        }

        // [STABILITY-LOGIC] Even if below target, if memory release has stopped changing,
        // it means we've recovered all we can. Don't wait forever.
        let delta = if current_vram > last_vram { current_vram - last_vram } else { last_vram - current_vram };
        if delta < 10_000_000 { // Change < 10MB (more lenient)
            stable_ticks += 1;
        } else {
            stable_ticks = 0;
        }

        // [FAST-EXIT] If stable for 1.5 seconds OR we have at least 600MB free (enough for Embedding/0.6B)
        // This prevents being stuck at 0.7GB when target is 1.1GB.
        if (stable_ticks >= 3 && current_vram > 600_000_000) || current_vram > target_vram_bytes {
            println!("[RESOURCE-WATCH] Memory sufficient or stabilized. Proceeding with {:.2} GB free VRAM.", current_vram as f64 / 1e9);
            break;
        }

        if last_report.elapsed().as_secs() >= 2 { // Faster reporting
            println!("[RESOURCE-DIAG] Waiting... VRAM: {:.2} GB free (Target: {:.2} GB)", 
                current_vram as f64 / 1e9, target_vram_mb as f64 / 1024.0);
            last_report = std::time::Instant::now();
        }

        // Absolute maximum wait 10s (reduced from 20s)
        if start_time.elapsed().as_secs() > 10 {
            println!("[RESOURCE-WATCH] Timeout or sufficient VRAM reached. Proceeding.");
            break;
        }

        last_vram = current_vram;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Ok(())
}

// ============================================================================
// [KV RESIDENCY PLANNER]
// 디코딩 단계에서 KV Cache 를 어디에 둘지(VRAM / RAM / SSD) 단 1회 판정합니다.
// KV Cache 는 토큰이 진행될수록 단조 증가하므로, "지금 크기"가 아니라
// "마지막 토큰까지 자랐을 때의 최대 크기"를 기준으로 판정해야
// 디코딩 도중 OOM 이 터지는 사고를 원천 차단할 수 있습니다.
// ============================================================================

/// VRAM 상주 판정 시 남겨둘 안전 마진.
/// Attention score 행렬, logits(vocab 15만), cuBLAS 워크스페이스, 드라이버 예약분을 포괄합니다.
pub const KV_VRAM_SAFETY_MARGIN_BYTES: u64 = 640 * 1024 * 1024;

/// RAM 오프로딩 판정 시 남겨둘 안전 마진. OS/브라우저 동시 사용을 고려합니다.
pub const KV_RAM_SAFETY_MARGIN_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvResidency {
    /// VRAM 상주. 디코딩 중 SSD/RAM 왕복이 전혀 발생하지 않습니다.
    Vram,
    /// RAM 오프로딩. 매 토큰 필요한 레이어만 PCIe 로 올렸다 내립니다.
    Ram,
    /// SSD 오프로딩. 기존 경로 그대로 유지합니다.
    Ssd,
}

pub struct KvPlanInput<'a> {
    pub gpu_id: u32,
    pub is_cpu_mode: bool,
    /// KV 를 실제로 보관하는 레이어 수 (Qwen3.5 는 full_attention 레이어 수만 셉니다)
    pub num_kv_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// 보관 시 원소 1개당 바이트 수 (FP8=1, BF16/F16=2, F32=4)
    pub bytes_per_elem: usize,
    /// 현재 문맥 길이 + 앞으로 생성할 최대 토큰 수
    pub planned_tokens: usize,
    pub label: &'a str,
}

/// Context-free backend classification for runtime resource planning and reflection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeBackendKind {
    Cuda,
    Rocm,
    Vulkan,
    Metal,
    Cpu,
}

impl RuntimeBackendKind {
    pub fn from_device(dev: &candle_core::Device) -> Self {
        match dev {
            candle_core::Device::Cuda(_) => Self::Cuda,
            candle_core::Device::Rocm(_) => Self::Rocm,
            candle_core::Device::Vulkan(_) => Self::Vulkan,
            candle_core::Device::Metal(_) => Self::Metal,
            _ => Self::Cpu,
        }
    }
}

/// Primary GPU / runtime summary probed once without creating unused secondary CUDA contexts.
#[derive(Clone, Debug)]
pub struct RuntimeDeviceSummary {
    pub backend_kind: RuntimeBackendKind,
    pub runtime_ordinal: usize,
    pub nvml_index: Option<u32>,
    pub total_vram_bytes: Option<u64>,
    pub is_unified_memory: bool,
    pub is_software_adapter: bool,
}

/// Probe the runtime primary device summary once without opening extra GPU driver contexts.
pub fn probe_runtime_device_summary() -> &'static RuntimeDeviceSummary {
    static SUMMARY: std::sync::OnceLock<RuntimeDeviceSummary> = std::sync::OnceLock::new();
    SUMMARY.get_or_init(|| {
        #[cfg(feature = "rocm")]
        {
            let probe = crate::utils::GpuMemProbe::new();
            let count = probe.device_count();
            let mut best_ord = 0usize;
            let mut best_free = 0u64;
            let mut best_total = None;
            for i in 0..count {
                if let Some((free, total)) = probe.mem_info(i) {
                    if free >= best_free {
                        best_free = free;
                        best_ord = i;
                        best_total = Some(total);
                    }
                }
            }
            if count > 0 {
                return RuntimeDeviceSummary {
                    backend_kind: RuntimeBackendKind::Rocm,
                    runtime_ordinal: best_ord,
                    nvml_index: None,
                    total_vram_bytes: best_total,
                    is_unified_memory: false,
                    is_software_adapter: false,
                };
            }
        }
        #[cfg(feature = "cuda")]
        {
            if let Some(nvml) = crate::utils::device_utils::nvml_handle() {
                if let Ok(cnt) = nvml.device_count() {
                    let mut best_ord = 0usize;
                    let mut best_free = 0u64;
                    let mut best_total = None;
                    for i in 0..cnt {
                        if let Ok(d) = nvml.device_by_index(i) {
                            if let Ok(mem) = d.memory_info() {
                                if mem.free >= best_free {
                                    best_free = mem.free;
                                    best_ord = i as usize;
                                    best_total = Some(mem.total);
                                }
                            }
                        }
                    }
                    if cnt > 0 {
                        return RuntimeDeviceSummary {
                            backend_kind: RuntimeBackendKind::Cuda,
                            runtime_ordinal: best_ord,
                            nvml_index: Some(best_ord as u32),
                            total_vram_bytes: best_total,
                            is_unified_memory: false,
                            is_software_adapter: false,
                        };
                    }
                }
            }
        }
        #[cfg(feature = "vulkan")]
        {
            let probe = crate::utils::GpuMemProbe::new();
            let count = probe.device_count();
            if count > 0 {
                let total = probe.mem_info(0).map(|(_, t)| t);
                return RuntimeDeviceSummary {
                    backend_kind: RuntimeBackendKind::Vulkan,
                    runtime_ordinal: 0,
                    nvml_index: None,
                    total_vram_bytes: total,
                    is_unified_memory: false,
                    is_software_adapter: false,
                };
            }
        }
        #[cfg(feature = "metal")]
        {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            return RuntimeDeviceSummary {
                backend_kind: RuntimeBackendKind::Metal,
                runtime_ordinal: 0,
                nvml_index: None,
                total_vram_bytes: Some(sys.total_memory()),
                is_unified_memory: true,
                is_software_adapter: false,
            };
        }
        RuntimeDeviceSummary {
            backend_kind: RuntimeBackendKind::Cpu,
            runtime_ordinal: 0,
            nvml_index: None,
            total_vram_bytes: None,
            is_unified_memory: true,
            is_software_adapter: false,
        }
    })
}

/// NVML 장치 목록을 1회만 훑어 여유 VRAM 이 가장 큰 GPU 인덱스를 캐싱합니다.
/// device_utils::get_best_device_info 와 동일한 정책이지만 Device 객체를 생성하지 않아
/// 디코딩 경로에서 호출해도 CUDA 컨텍스트를 건드리지 않습니다.
static PRIMARY_GPU_ID: Lazy<u32> = Lazy::new(|| {
    let probe = crate::utils::GpuMemProbe::new();
    let mut best_id = 0u32;
    let mut max_free = 0u64;
    for i in 0..probe.device_count() {
        if let Some(free) = probe.free_bytes(i) {
            if free > max_free {
                max_free = free;
                best_id = i as u32;
            }
        }
    }
    best_id
});

pub fn primary_gpu_id() -> u32 {
    *PRIMARY_GPU_ID
}

/// 토큰 1개가 전 레이어에 걸쳐 차지하는 KV 바이트 수
pub fn kv_bytes_per_token(input: &KvPlanInput) -> u64 {
    (2u64)
        * (input.num_kv_heads as u64)
        * (input.head_dim as u64)
        * (input.bytes_per_elem as u64)
        * (input.num_kv_layers as u64)
}

/// 현재 여유 VRAM(bytes). GPU 가 없거나 NVML 실패 시 0 을 반환합니다.
pub fn free_vram_bytes(gpu_id: u32) -> u64 {
    crate::utils::gpu_mem_info(gpu_id as usize)
        .map(|(free, _)| free)
        .unwrap_or(0)
}

/// 현재 여유 RAM(bytes). 전역 `SYSTEM_MONITOR` 를 재사용해 매 호출 `System::new()` 생성 비용을 제거합니다.
pub fn free_ram_bytes() -> u64 {
    let mut sys = SYSTEM_MONITOR.lock().unwrap_or_else(|p| p.into_inner());
    sys.refresh_memory();
    sys.available_memory()
}

//// 디코딩 단계 KV Cache 배치 위치를 결정합니다.
/// 호출부가 순간치·이미 상주한 토큰을 알려 주지 않는 예전 서명입니다. 여유는 예전과 같은 640MB 이고,
/// 필요량은 `planned_tokens` 전체입니다. (`LOGIS_KV_PLAN=legacy` 이면 예전 판정 그대로)
pub fn plan_kv_residency(input: &KvPlanInput) -> KvResidency {
    plan_kv_residency_ex(input, &KvPlanExtra::default())
}

// ============================================================================
// 🌟 [KV-PLAN v2] 증분 가격 + 비례 여유 + 히스테리시스
// ----------------------------------------------------------------------------
//  ── 무엇이 문제였나 ──
//   (1) 이미 VRAM 에 만들어진 프롬프트 KV 를 free 에서 한 번 빼고 필요량으로 또 요구했습니다.
//   (2) 디코딩 여유 토큰을 실제 max_tokens(96~512) 가 아니라 고정 2048 로 잡았습니다.
//   (3) 여유가 고정 640MB(VRAM) / 2GB(RAM) 라 4GB 카드·7.5GB RAM 에서 48MB KV 가 SSD 로 갔습니다.
//   (4) 호출마다 순간 free 로만 판정해 이웃 호출 사이에 Vram / Ssd 가 뒤집혔습니다.
//  ── 해결 ──
//   필요량 = (앞으로 새로 생길 토큰 수) x 정확한 토큰당 바이트,
//   VRAM 여유 = max(96MB, 2 x 디코딩 순간치), RAM 여유 = min(2GB, 총 RAM 의 15%),
//   이전 판정보다 좋은 계층으로 올라갈 때만 max(64MB, 필요량 10%) 의 띠를 더 요구합니다.
// ============================================================================

/// VRAM 여유의 하한.
pub const KV_VRAM_MARGIN_MIN_BYTES: u64 = 96 * 1024 * 1024;
/// 순간치를 알려 주지 않는 호출부에 쓰는 순간치 (여유 = 2 x 이 값 = 예전과 같은 640MB).
pub const LEGACY_DECODE_TRANSIENT_BYTES: u64 = KV_VRAM_SAFETY_MARGIN_BYTES / 2;
/// 계층을 올릴 때 요구하는 추가 띠의 하한.
pub const KV_TIER_HYSTERESIS_MIN_BYTES: u64 = 64 * 1024 * 1024;

/// `LOGIS_KV_PLAN=legacy` 이면 예전 판정(전체 토큰 · 고정 여유 · 히스테리시스 없음)을 씁니다.
pub fn kv_plan_legacy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LOGIS_KV_PLAN")
            .map(|v| v.trim().eq_ignore_ascii_case("legacy"))
            .unwrap_or(false)
    })
}

/// VRAM 여유 = max(96MB, 2 x 디코딩 한 스텝 순간치).
pub fn kv_vram_margin_bytes(decode_transient_bytes: u64) -> u64 {
    KV_VRAM_MARGIN_MIN_BYTES.max(decode_transient_bytes.saturating_mul(2))
}

/// RAM 여유 = min(2GB, 총 RAM 의 15%).
pub fn kv_ram_margin_bytes(total_ram_bytes: u64) -> u64 {
    KV_RAM_SAFETY_MARGIN_BYTES.min(total_ram_bytes / 100 * 15)
}

/// 총 RAM(bytes), 프로세스당 1회만 읽습니다.
pub fn total_ram_bytes() -> u64 {
    static T: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        let mut sys = System::new();
        sys.refresh_memory();
        sys.total_memory()
    })
}

/// 압축 KV(TurboQuant 벡터별 FP8) 한 행(토큰 1 · 헤드 1 · K 또는 V)의 바이트 수.
/// 코드 d 바이트 + 행 머리(스케일 등) 2워드 = d + 8 → d=256 이면 264 B(264/256), d=128 이면 136 B(34/32).
/// 장치를 알면 `kv_turbo::default_meta(d, dev).bytes_per_token_head()` 가 정확한 값입니다.
pub fn tq_fp8_row_bytes(head_dim: usize) -> u64 {
    head_dim as u64 + 8
}

/// 압축 KV 토큰당 바이트 (전 레이어 K+V).
pub fn kv_bytes_per_token_compressed(num_kv_layers: usize, num_kv_heads: usize, head_dim: usize) -> u64 {
    2 * num_kv_layers as u64 * num_kv_heads as u64 * tq_fp8_row_bytes(head_dim)
}

/// 원본 KV 토큰당 바이트 (전 레이어 K+V, 원소 크기 `elem_bytes`).
pub fn kv_bytes_per_token_exact(num_kv_layers: usize, num_kv_heads: usize, head_dim: usize, elem_bytes: usize) -> u64 {
    2 * num_kv_layers as u64 * num_kv_heads as u64 * head_dim as u64 * elem_bytes as u64
}

/// 🌟 [KV-PLAN v2] 필요량 = 토큰당 바이트 x (계획 토큰 - 이미 VRAM 에 있는 토큰).
pub fn kv_need_bytes(per_token: u64, planned_tokens: usize, already_tokens: usize) -> u64 {
    per_token.saturating_mul(planned_tokens.saturating_sub(already_tokens) as u64)
}

/// [`plan_kv_residency_ex`] 의 추가 입력 (모두 선택).
#[derive(Debug, Clone, Copy, Default)]
pub struct KvPlanExtra {
    /// 이미 VRAM 에 만들어져 free 에서 빠져 있는 토큰 수 (필요량에서 뺍니다).
    pub already_tokens: usize,
    /// 디코딩 한 스텝의 활성값 순간치. 0 이면 모름 → 예전 640MB 여유.
    pub decode_transient_bytes: u64,
    /// 토큰당 바이트를 직접 줄 때 (압축 KV 행 머리 포함 등). None 이면 `kv_bytes_per_token(input)`.
    pub bytes_per_token: Option<u64>,
    /// 직전 판정 (히스테리시스).
    pub prev: Option<KvResidency>,
}

/// 🌟 [KV-PLAN] 순수 판정 함수 (테스트 가능). `vram_free` 가 None(측정 실패)이면 VRAM 을 고르지 않습니다.
/// 계층 선호: VRAM > RAM > SSD. 직전보다 좋은 계층으로 올라갈 때만 띠(max(64MB, 필요량 10%))를 더 요구합니다.
pub fn decide_kv_tier(
    need: u64,
    vram_free: Option<u64>,
    ram_free: u64,
    vram_margin: u64,
    ram_margin: u64,
    prev: Option<KvResidency>,
) -> KvResidency {
    let band = KV_TIER_HYSTERESIS_MIN_BYTES.max(need / 10);
    let rank = |r: KvResidency| match r {
        KvResidency::Vram => 2,
        KvResidency::Ram => 1,
        KvResidency::Ssd => 0,
    };
    let prev_rank = prev.map(rank);
    let fits = |free: u64, margin: u64, tier: KvResidency| {
        let extra = match prev_rank {
            Some(p) if rank(tier) > p => band,
            _ => 0,
        };
        free >= need.saturating_add(margin).saturating_add(extra)
    };
    if let Some(v) = vram_free {
        if v > 0 && fits(v, vram_margin, KvResidency::Vram) {
            return KvResidency::Vram;
        }
    }
    if fits(ram_free, ram_margin, KvResidency::Ram) {
        return KvResidency::Ram;
    }
    KvResidency::Ssd
}

/// 디코딩 단계 KV Cache 배치 위치 (증분 가격 · 비례 여유 · 히스테리시스).
pub fn plan_kv_residency_ex(input: &KvPlanInput, extra: &KvPlanExtra) -> KvResidency {
    if kv_plan_legacy() {
        return plan_kv_residency_legacy(input);
    }
    let per_token = extra.bytes_per_token.unwrap_or_else(|| kv_bytes_per_token(input));
    let new_tokens = input.planned_tokens.saturating_sub(extra.already_tokens);
    let need = kv_need_bytes(per_token, input.planned_tokens, extra.already_tokens);
    if input.is_cpu_mode {
        println!("[KV-PLAN] {} | CPU Mode → RAM 상주 (Need: {:.2} MB)", input.label, need as f64 / 1e6);
        return KvResidency::Ram;
    }
    let transient = if extra.decode_transient_bytes == 0 {
        LEGACY_DECODE_TRANSIENT_BYTES
    } else {
        extra.decode_transient_bytes
    };
    let vram_margin = kv_vram_margin_bytes(transient);
    let ram_margin = kv_ram_margin_bytes(total_ram_bytes());
    let vram_free = crate::utils::gpu_mem_info(input.gpu_id as usize).map(|(f, _)| f);
    let ram_free = free_ram_bytes();
    let decision = decide_kv_tier(need, vram_free, ram_free, vram_margin, ram_margin, extra.prev);
    println!(
        "[KV-PLAN] {} | Tokens: {} (이미 상주 {} · 새로 {}) | PerToken: {} B | Need: {:.2} MB | VRAM free: {} (margin {:.0} MB) | RAM free: {:.2} MB (margin {:.0} MB) | 직전 {:?} → {:?}",
        input.label,
        input.planned_tokens,
        extra.already_tokens.min(input.planned_tokens),
        new_tokens,
        per_token,
        need as f64 / 1e6,
        vram_free.map(|v| format!("{:.2} MB", v as f64 / 1e6)).unwrap_or_else(|| "?".to_string()),
        vram_margin as f64 / 1e6,
        ram_free as f64 / 1e6,
        ram_margin as f64 / 1e6,
        extra.prev,
        decision
    );
    decision
}

/// 예전 판정 (`LOGIS_KV_PLAN=legacy`).
pub fn plan_kv_residency_legacy(input: &KvPlanInput) -> KvResidency {
    let per_token = kv_bytes_per_token(input);
    let need = per_token.saturating_mul(input.planned_tokens as u64);

    // CPU 모드에서는 애초에 VRAM 이 없으므로 RAM 상주가 유일한 정답입니다.
    if input.is_cpu_mode {
        println!(
            "[KV-PLAN] {} | CPU Mode → RAM 상주 (Need: {:.2} MB)",
            input.label,
            need as f64 / 1e6
        );
        return KvResidency::Ram;
    }

    let vram_free = free_vram_bytes(input.gpu_id);
    let ram_free = free_ram_bytes();

    let vram_ok = vram_free > 0 && vram_free >= need.saturating_add(KV_VRAM_SAFETY_MARGIN_BYTES);
    let ram_ok = ram_free >= need.saturating_add(KV_RAM_SAFETY_MARGIN_BYTES);

    let decision = if vram_ok {
        KvResidency::Vram
    } else if ram_ok {
        KvResidency::Ram
    } else {
        KvResidency::Ssd
    };

    println!(
        "[KV-PLAN] {} | Tokens: {} | PerToken: {} B | Need: {:.2} MB | VRAM free: {:.2} MB (margin {:.0} MB) | RAM free: {:.2} MB (margin {:.0} MB) → {:?}",
        input.label,
        input.planned_tokens,
        per_token,
        need as f64 / 1e6,
        vram_free as f64 / 1e6,
        KV_VRAM_SAFETY_MARGIN_BYTES as f64 / 1e6,
        ram_free as f64 / 1e6,
        KV_RAM_SAFETY_MARGIN_BYTES as f64 / 1e6,
        decision
    );

    decision
}

// Global System Monitor Instance
static SYSTEM_MONITOR: Lazy<Arc<Mutex<System>>> = Lazy::new(|| {
    let mut sys = System::new_with_specifics(
        RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything())
    );
    // Initial refresh to get baseline
    sys.refresh_cpu();
    sys.refresh_memory();
    Arc::new(Mutex::new(sys))
});

#[derive(Debug, Clone)]
pub struct ThreadConfig {
    pub thread_count: usize,
    pub description: String,
}

pub fn set_current_thread_low_priority() {
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::System::Threading::*;
        unsafe {
            let handle = GetCurrentProcess();
            // BELOW_NORMAL_PRIORITY_CLASS (0x00004000)
            SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS);
            
            let thread_handle = GetCurrentThread();
            SetThreadPriority(thread_handle, THREAD_PRIORITY_BELOW_NORMAL);
        }
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        unsafe {
            // In Unix, priority is set via 'nice' value or setpriority.
            // 0 is normal, 19 is lowest. 10 is a good "below normal" value.
            // PRIO_PROCESS = 0
            libc::setpriority(libc::PRIO_PROCESS, 0, 10);
        }
    }
}

pub fn get_optimal_thread_config(is_cpu_mode: bool) -> ThreadConfig {
    let mut sys = SYSTEM_MONITOR.lock().unwrap();
    
    // 1. Refresh CPU stats
    sys.refresh_cpu();
    
    // 2. Calculate CPU Load
    let global_cpu_usage = sys.global_cpu_info().cpu_usage();
    let physical_cores = sys.physical_core_count().unwrap_or(4);
    
    // 3. Check GPU status
    let has_gpu = crate::utils::gpu_count() > 0 && !is_cpu_mode; // If forced CPU, treat as if no GPU for logging
    
    // --- ORGANIC DECISION LOGIC ---
    let (threads, mode) = if has_gpu {
        if global_cpu_usage > 60.0 {
            let safe_threads = (physical_cores / 2).max(2);
            (safe_threads, "GPU + Eco Mode (User Active)")
        } else {
            let fast_threads = (physical_cores as f64 * 0.9) as usize;
            (fast_threads.max(2), "GPU + Turbo Mode")
        }
    } else {
        if global_cpu_usage > 50.0 {
            let safe_threads = (physical_cores as f64 * 0.5) as usize;
            (safe_threads.max(1), "CPU Eco Mode (High Load)")
        } else {
            let fast_threads = physical_cores.saturating_sub(1).max(1);
            (fast_threads, "CPU Max Performance")
        }
    };

    ThreadConfig {
        thread_count: threads,
        description: format!("{} (Usage: {:.1}%)", mode, global_cpu_usage),
    }
}

/// Returns (RAM used in bytes, VRAM used in bytes)
pub fn get_memory_usage() -> (u64, u64) {
    let ram_used = {
        let mut sys = SYSTEM_MONITOR.lock().unwrap_or_else(|e| e.into_inner());
        sys.refresh_memory();
        sys.used_memory()
    };

    let mut vram_used = 0;
    let probe = crate::utils::GpuMemProbe::new();
    for i in 0..probe.device_count() {
        if let Some((free, total)) = probe.mem_info(i) {
            vram_used += total.saturating_sub(free);
        }
    }
    (ram_used, vram_used)
}

/// Waits until memory usage drops close to the baseline or timeout occurs.
/// This prevents OOM by ensuring the OS/Driver actually freed the resources.
/// 
/// * `baseline_ram`: RAM usage before model load (bytes)
/// * `baseline_vram`: VRAM usage before model load (bytes)
/// * `timeout_ms`: Max time to wait (e.g., 5000ms)
pub async fn wait_for_memory_release(baseline_ram: u64, baseline_vram: u64, timeout_ms: u64) {
    let start = Instant::now();
    // RAM tolerance is stricter because OS RAM management is more fluid.
    // We expect RAM to drop by at least 100MB to signal release start.
    let margin_ram = 100 * 1024 * 1024; 
    let margin_vram = 200 * 1024 * 1024;
    
    println!("[MEM-WATCH] Waiting for release... Baseline RAM: {:.2} GB, VRAM: {:.2} GB", 
        baseline_ram as f64 / 1e9, baseline_vram as f64 / 1e9);

    loop {
        let (curr_ram, curr_vram) = get_memory_usage();
        
        // Check if memory has started dropping
        let ram_dropped = curr_ram < baseline_ram.saturating_sub(margin_ram);
        let vram_dropped = curr_vram < baseline_vram.saturating_sub(margin_vram);

        if ram_dropped || vram_dropped {
            println!("[MEM-WATCH] ✅ Memory Drop Detected! RAM: {:.2} GB, VRAM: {:.2} GB. Took {}ms", 
                curr_ram as f64 / 1e9, curr_vram as f64 / 1e9, start.elapsed().as_millis());
            break;
        }

        if start.elapsed().as_millis() as u64 > timeout_ms {
            println!("[MEM-WATCH] ⚠️ Timeout. Proceeding with current RAM: {:.2} GB, VRAM: {:.2} GB", 
                curr_ram as f64 / 1e9, curr_vram as f64 / 1e9);
            break;
        }

        sleep(Duration::from_millis(150)).await; // Faster polling (0.15s)
    }
}

// ── RAM 압박 기준 · 스냅숏 · 힙 반환 (예전 앱 model.rs, 모델의 KV 계획과 앱 수명주기가 함께 씀) ──

/// 여유 RAM 이 이보다 적으면 압박으로 봅니다 (KV-PLAN 의 RAM margin 과 같은 값).
pub const RAM_PRESSURE_MARGIN_BYTES: u64 = 2 * 1024 * 1024 * 1024;

pub fn ram_snapshot() -> (u64, u64) {
    let mut sys = SYSTEM_MONITOR.lock().unwrap_or_else(|p| p.into_inner());
    sys.refresh_memory();
    (sys.available_memory(), sys.total_memory())
}

pub fn ram_used_pct(avail: u64, total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    (total.saturating_sub(avail) as f64 * 100.0 / total as f64) as f32
}

pub fn process_rss_bytes() -> Option<u64> {
    let pid = sysinfo::get_current_pid().ok()?;
    let mut sys = SYSTEM_MONITOR.lock().unwrap_or_else(|p| p.into_inner());
    sys.refresh_process_specifics(pid, sysinfo::ProcessRefreshKind::new());
    sys.process(pid).map(|p| p.memory())
}

pub fn ram_pressure_line(avail: u64, total: u64) -> String {
    let limit_pct = if total > 0 {
        (100.0 - RAM_PRESSURE_MARGIN_BYTES as f64 * 100.0 / total as f64).max(0.0)
    } else {
        0.0
    };
    format!(
        "RAM 사용 {:.1}% (여유 {}MB / 총 {}MB · 이 프로세스 작업 집합 {}) | 압박 기준: 여유 {}MB 미만 = 이 PC 에서 사용 {:.1}% 이상 (KV-PLAN 의 RAM margin 과 같은 값)",
        ram_used_pct(avail, total),
        avail / 1_000_000,
        total / 1_000_000,
        process_rss_bytes().map_or("-".to_string(), |b| format!("{}MB", b / 1_000_000)),
        RAM_PRESSURE_MARGIN_BYTES / 1_000_000,
        limit_pct
    )
}

/// 🌟 [HEAP-RELEASE] RAM 압박 시 메모리 반환.
/// 기본은 mimalloc 의 빈 힙 페이지만 OS 로 돌려줍니다(utils::release_heap). 예전처럼 작업 집합 전체를
/// 내쫓으면(SetProcessWorkingSetSizeEx) mmap 가중치·토크나이저 페이지가 곧바로 다시 읽혀 RAM 이 톱니처럼
/// 오르내리고 수 초씩 멈췄습니다. `LOGIS_MEM_TRIM=legacy` 이면 예전 동작입니다.
pub fn release_os_working_set() {
    if crate::utils::legacy_ws_trim_enabled() {
        crate::utils::force_working_set_trim();
    } else {
        crate::utils::release_heap();
    }
}

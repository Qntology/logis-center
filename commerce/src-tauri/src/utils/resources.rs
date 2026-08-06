use sysinfo::{System, RefreshKind, CpuRefreshKind, MemoryRefreshKind};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use once_cell::sync::Lazy;
use nvml_wrapper::Nvml;
use std::time::{Duration, Instant};
use tokio::time::sleep;

use anyhow::Result;



pub async fn wait_for_resources_settled(target_vram_mb: u64, target_ram_mb: u64, cancellation_token: Option<&Arc<AtomicBool>>, target_gpu_id: u32) -> Result<()> {
    use nvml_wrapper::Nvml;
    use sysinfo::System;
    
    let mut sys = System::new_all();
    let nvml = Nvml::init().ok();
    
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

        if let Some(ref nvml_inst) = nvml {
            if let Ok(dev) = nvml_inst.device_by_index(target_gpu_id) {
                if let Ok(mem) = dev.memory_info() {
                    current_vram = mem.free;
                    has_gpu = true;
                }
            }
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

/// NVML 장치 목록을 1회만 훑어 여유 VRAM 이 가장 큰 GPU 인덱스를 캐싱합니다.
/// device_utils::get_best_device_info 와 동일한 정책이지만 Device 객체를 생성하지 않아
/// 디코딩 경로에서 호출해도 CUDA 컨텍스트를 건드리지 않습니다.
static PRIMARY_GPU_ID: Lazy<u32> = Lazy::new(|| {
    if let Ok(nvml) = Nvml::init() {
        if let Ok(count) = nvml.device_count() {
            let mut best_id = 0u32;
            let mut max_free = 0u64;
            for i in 0..count {
                if let Ok(dev) = nvml.device_by_index(i) {
                    if let Ok(mem) = dev.memory_info() {
                        if mem.free > max_free {
                            max_free = mem.free;
                            best_id = i;
                        }
                    }
                }
            }
            return best_id;
        }
    }
    0
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
    if let Ok(nvml) = Nvml::init() {
        if let Ok(dev) = nvml.device_by_index(gpu_id) {
            if let Ok(mem) = dev.memory_info() {
                return mem.free;
            }
        }
    }
    0
}

/// 현재 여유 RAM(bytes).
pub fn free_ram_bytes() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.available_memory()
}

/// 디코딩 단계 KV Cache 배치 위치를 결정합니다.
pub fn plan_kv_residency(input: &KvPlanInput) -> KvResidency {
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
    let nvml = Nvml::init().ok();
    let has_gpu = nvml.is_some() && !is_cpu_mode; // If forced CPU, treat as if no GPU for logging
    
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
    let mut sys = SYSTEM_MONITOR.lock().unwrap();
    sys.refresh_memory();
    let ram_used = sys.used_memory(); 
    
    let mut vram_used = 0;
    if let Ok(nvml) = Nvml::init() {
        if let Ok(count) = nvml.device_count() {
            for i in 0..count {
                 if let Ok(dev) = nvml.device_by_index(i) {
                     if let Ok(mem) = dev.memory_info() {
                         vram_used += mem.used;
                     }
                 }
            }
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
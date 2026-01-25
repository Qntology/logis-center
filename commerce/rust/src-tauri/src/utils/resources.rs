use sysinfo::{System, RefreshKind, CpuRefreshKind, MemoryRefreshKind};
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use nvml_wrapper::Nvml;
use std::time::{Duration, Instant};
use tokio::time::sleep;

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

pub fn get_optimal_thread_config() -> ThreadConfig {
    let mut sys = SYSTEM_MONITOR.lock().unwrap();
    
    // 1. Refresh CPU stats
    sys.refresh_cpu();
    
    // 2. Calculate CPU Load
    let global_cpu_usage = sys.global_cpu_info().cpu_usage();
    let physical_cores = sys.physical_core_count().unwrap_or(4);
    
    // 3. Check GPU status
    let nvml = Nvml::init().ok();
    let has_gpu = nvml.is_some();
    
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
    let margin_ram = 1536 * 1024 * 1024; // 1.5GB tolerance (Windows memory management can be very slow)
    let margin_vram = 600 * 1024 * 1024; // 600MB tolerance
    
    println!("[MEM-WATCH] Waiting for release... Target RAM < {:.2} GB, VRAM < {:.2} GB", 
        (baseline_ram + margin_ram) as f64 / 1e9, (baseline_vram + margin_vram) as f64 / 1e9);

    loop {
        let (curr_ram, curr_vram) = get_memory_usage();
        
        let ram_ok = curr_ram <= (baseline_ram + margin_ram);
        let vram_ok = curr_vram <= (baseline_vram + margin_vram);

        if ram_ok && vram_ok {
            println!("[MEM-WATCH] ✅ Release Verified! RAM: {:.2} GB, VRAM: {:.2} GB. Took {}ms", 
                curr_ram as f64 / 1e9, curr_vram as f64 / 1e9, start.elapsed().as_millis());
            break;
        }

        if start.elapsed().as_millis() as u64 > timeout_ms {
            println!("[MEM-WATCH] ⚠️ Timeout waiting for memory release. Proceeding anyway. Current RAM: {:.2} GB, VRAM: {:.2} GB", 
                curr_ram as f64 / 1e9, curr_vram as f64 / 1e9);
            break;
        }

        sleep(Duration::from_millis(250)).await;
    }
}
use candle_core::{Device, DType};
use anyhow::Result;
#[cfg(feature = "cuda")]
use std::process::Command;
use nvml_wrapper::Nvml;
use once_cell::sync::Lazy;
use std::sync::Mutex;

// [FIX] 장치 번호별로 장치 객체를 캐싱하여 중복 생성 방지 (DeviceId 폭주 해결)
static DEVICE_CACHE: Lazy<Mutex<Vec<Option<Device>>>> = Lazy::new(|| Mutex::new(vec![None; 8]));

#[cfg(feature = "vulkan")]
static VULKAN_DEVICE_COUNT: Lazy<usize> =
    Lazy::new(|| candle_core::vulkan_backend::device_count().unwrap_or(0));

pub trait GpuDeviceExt {
    fn is_cuda_or_rocm(&self) -> bool;
}

impl GpuDeviceExt for Device {
    fn is_cuda_or_rocm(&self) -> bool {
        self.is_cuda() || self.is_rocm()
    }
}

pub fn new_gpu_device(id: usize) -> Option<Device> {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(id) {
        return Some(d);
    }
    #[cfg(feature = "rocm")]
    if let Ok(d) = Device::new_rocm(id) {
        return Some(d);
    }
    #[cfg(feature = "vulkan")]
    if let Ok(d) = Device::new_vulkan(id) {
        return Some(d);
    }
    let _ = id;
    None
}

pub struct GpuMemProbe {
    nvml: Option<Nvml>,
}

impl GpuMemProbe {
    pub fn new() -> Self {
        #[cfg(any(feature = "rocm", feature = "vulkan"))]
        let nvml = None;
        #[cfg(not(any(feature = "rocm", feature = "vulkan")))]
        let nvml = Nvml::init().ok();
        Self { nvml }
    }

    pub fn device_count(&self) -> usize {
        #[cfg(feature = "rocm")]
        {
            let n = candle_rocm::device_count().unwrap_or(0);
            if n > 0 {
                return n;
            }
        }
        #[cfg(feature = "vulkan")]
        {
            let n = *VULKAN_DEVICE_COUNT;
            if n > 0 {
                return n;
            }
        }
        self.nvml
            .as_ref()
            .and_then(|n| n.device_count().ok())
            .unwrap_or(0) as usize
    }

    pub fn mem_info(&self, gpu_id: usize) -> Option<(u64, u64)> {
        #[cfg(feature = "rocm")]
        if let Ok((free, total)) = candle_rocm::mem_info(gpu_id) {
            return Some((free as u64, total as u64));
        }
        #[cfg(feature = "vulkan")]
        if gpu_id == 0 && *VULKAN_DEVICE_COUNT > 0 {
            if let Device::Vulkan(d) = get_gpu_device(gpu_id) {
                if let Ok((free, total)) = d.mem_info() {
                    return Some((free as u64, total as u64));
                }
            }
        }
        let mem = self
            .nvml
            .as_ref()?
            .device_by_index(gpu_id as u32)
            .ok()?
            .memory_info()
            .ok()?;
        Some((mem.free, mem.total))
    }

    pub fn free_bytes(&self, gpu_id: usize) -> Option<u64> {
        self.mem_info(gpu_id).map(|(free, _)| free)
    }
}

pub fn gpu_count() -> usize {
    GpuMemProbe::new().device_count()
}

pub fn gpu_mem_info(gpu_id: usize) -> Option<(u64, u64)> {
    GpuMemProbe::new().mem_info(gpu_id)
}

pub fn flush_gpu_memory_pool(gpu_id: usize) {
    #[cfg(feature = "cuda")]
    {
        let _ = Device::new_cuda(gpu_id);
    }
    #[cfg(feature = "rocm")]
    {
        if let Device::Rocm(d) = get_gpu_device(gpu_id) {
            log_rocm_vram_report("before-release", &d);
            if let Err(e) = d.release_cached_resources() {
                println!("[VRAM-REPORT] release_cached_resources failed: {e}");
                let _ = d.trim_memory_pool();
            }
            log_rocm_vram_report("after-release", &d);
        }
    }
    #[cfg(feature = "vulkan")]
    {
        if let Device::Vulkan(d) = get_gpu_device(gpu_id) {
            log_vulkan_vram_report("before-release", &d);
            let _ = d.trim_memory_pool();
            log_vulkan_vram_report("after-release", &d);
        }
    }
    let _ = gpu_id;
}

/// Vulkan 전용: OS(작업 관리자) 기준 이 프로세스 VRAM·공유메모리 / 텐서 힙 사용량 / 이 프로세스 할당량(풀 대기분 포함).
#[cfg(feature = "vulkan")]
pub fn log_vulkan_vram_report(tag: &str, d: &candle_core::VulkanDevice) {
    let mb = |b: u64| b as f64 / 1048576.0;
    let os = crate::utils::os_vram::process_vram()
        .map(|v| format!("VRAM {:.0} MB / 공유 {:.0} MB", mb(v.local_usage), mb(v.nonlocal_usage)))
        .unwrap_or_else(|| "n/a".into());
    let (gpu_local, gpu_shared) = d.gpu_memory_bytes();
    if let Ok((free, total)) = d.mem_info() {
        println!(
            "[VRAM-REPORT] {tag} | OS(이 프로세스) {os} | GPU 메모리 사용 {:.0} MB (free {:.0} / total {:.0}) | 가중치·미러 VRAM {:.0} MB / 공유 {:.0} MB | 호스트 텐서 버퍼 {:.0} MB (풀 대기 {:.0} MB) | GPU 가중치 경로 {}",
            mb(total.saturating_sub(free) as u64), mb(free as u64), mb(total as u64),
            mb(gpu_local as u64), mb(gpu_shared as u64),
            mb(d.allocated_bytes() as u64), mb(d.pooled_bytes() as u64),
            if d.gpu_weights_enabled() { "ON" } else { "OFF" },
        );
    }
}

/// ROCm 전용: OS(작업 관리자) 기준 이 프로세스 VRAM / HIP 런타임이 보는 사용량 / 앱 텐서 / 풀을 한 줄로 남깁니다.
#[cfg(feature = "rocm")]
pub fn log_rocm_vram_report(tag: &str, d: &candle_core::RocmDevice) {
    let mb = |b: f64| b / 1048576.0;
    let os = crate::utils::os_vram::process_vram()
        .map(|v| format!("VRAM {:.0} MB / 공유 {:.0} MB", mb(v.local_usage as f64), mb(v.nonlocal_usage as f64)))
        .unwrap_or_else(|| "n/a".into());
    if let Ok(r) = d.memory_report() {
        let top = d
            .live_size_histogram(5)
            .iter()
            .map(|(b, c)| format!("{:.2}MB×{}", mb(*b as f64), c))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "[VRAM-REPORT] {tag} | OS(이 프로세스) {os} | HIP 사용 {:.0} MB (free {:.0} / total {:.0}) | 앱 텐서 {:.0} MB ({}건, peak {:.0}) | 풀 예약 {:.0} MB / 사용 {:.0} MB | 크기별 상위 [{top}]",
            mb((r.total - r.free) as f64), mb(r.free as f64), mb(r.total as f64),
            mb(r.live_bytes as f64), r.live_allocs, mb(r.peak_bytes as f64),
            mb(r.pool_reserved as f64), mb(r.pool_used as f64),
        );
    }
}

/// `device` 의 백엔드로 [VRAM-REPORT] 한 줄을 남깁니다 (gpu_id 를 모르는 호출부용).
pub fn log_vram_report_for(device: &Device, tag: &str) {
    match device {
        #[cfg(feature = "rocm")]
        Device::Rocm(d) => log_rocm_vram_report(tag, d),
        #[cfg(feature = "vulkan")]
        Device::Vulkan(d) => log_vulkan_vram_report(tag, d),
        _ => {
            let _ = tag;
        }
    }
}

/// 🌟 [POOL-TRIM] Vulkan: 재사용 대기 중인 호스트 버퍼를 드라이버에 돌려줍니다.
///
///  ── 왜 필요한가 (실측) ──
///   Vulkan 백엔드는 텐서를 호스트 가시 메모리(작업 관리자 '공유 GPU 메모리')에 두고,
///   해제된 버퍼를 최대 1GB 까지 재사용 풀에 남깁니다. 프리필이 만든 큰 임시 버퍼들이
///   그대로 풀에 남아 생성 직후에도 '풀 대기 993~1019 MB' 가 공유 메모리를 차지했습니다
///   (실사용 텐서는 28~34 MB). 디코드는 그 큰 크기를 다시 쓰지 않으므로
///   프리필 직후와 생성 직후에 풀을 비웁니다. 다음 프리필은 필요한 만큼 다시 할당합니다.
///   `LOGIS_VK_POOL_TRIM=0` 이면 끕니다.
pub fn trim_idle_gpu_pool(device: &Device, tag: &str) {
    match device {
        #[cfg(feature = "vulkan")]
        Device::Vulkan(d) => {
            let off = std::env::var("LOGIS_VK_POOL_TRIM")
                .map(|v| matches!(v.trim(), "0" | "false" | "off" | "no"))
                .unwrap_or(false);
            let pooled = d.pooled_bytes();
            if off || pooled < (64 << 20) {
                return;
            }
            let mb = |b: f64| b / 1048576.0;
            let os0 = crate::utils::os_vram::process_vram();
            let t0 = std::time::Instant::now();
            let _ = d.trim_memory_pool();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            match (os0, crate::utils::os_vram::process_vram()) {
                (Some(a), Some(b)) => println!(
                    "[POOL-TRIM] {tag} | 재사용 대기 호스트 버퍼 {:.0} MB 반환 ({:.1} ms) | OS 공유 {:.0} → {:.0} MB · VRAM {:.0} → {:.0} MB",
                    mb(pooled as f64), ms,
                    mb(a.nonlocal_usage as f64), mb(b.nonlocal_usage as f64),
                    mb(a.local_usage as f64), mb(b.local_usage as f64),
                ),
                _ => println!("[POOL-TRIM] {tag} | 재사용 대기 호스트 버퍼 {:.0} MB 반환 ({:.1} ms)", mb(pooled as f64), ms),
            }
        }
        // 🌟 [POOL-TRIM / ROCm] HIP 풀은 동기화 시점에 메모리를 돌려주도록 설정돼 있지만, 생성이
        //    끝난 뒤 다음 카테고리까지 동기화가 없어 프리필 임시 버퍼가 VRAM 에 남았습니다
        //    (실측 gen-end: 풀 예약 3,712 MB / 사용 2,391 MB). 같은 시점에 풀을 비웁니다.
        #[cfg(feature = "rocm")]
        Device::Rocm(d) => {
            let off = std::env::var("LOGIS_VK_POOL_TRIM")
                .map(|v| matches!(v.trim(), "0" | "false" | "off" | "no"))
                .unwrap_or(false);
            let Ok(r0) = d.memory_report() else { return };
            let idle = r0.pool_reserved.saturating_sub(r0.pool_used);
            if off || idle < (64 << 20) {
                return;
            }
            let mb = |b: f64| b / 1048576.0;
            let t0 = std::time::Instant::now();
            let _ = d.trim_memory_pool();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let os = crate::utils::os_vram::process_vram()
                .map(|v| format!(" · OS VRAM {:.0} MB", mb(v.local_usage as f64)))
                .unwrap_or_default();
            match d.memory_report() {
                Ok(r1) => println!(
                    "[POOL-TRIM] {tag} | HIP 풀 예약 {:.0} → {:.0} MB (사용 {:.0} MB) ({:.1} ms){os}",
                    mb(r0.pool_reserved as f64), mb(r1.pool_reserved as f64), mb(r1.pool_used as f64), ms
                ),
                Err(_) => println!("[POOL-TRIM] {tag} | HIP 풀 유휴 {:.0} MB 반환 ({:.1} ms){os}", mb(idle as f64), ms),
            }
        }
        _ => {
            let _ = (device, tag);
        }
    }
}

/// 🌟 [VK-PROFILE] `CANDLE_VULKAN_PROFILE=1` 일 때 Vulkan 백엔드의 연산별 시간표를 남기고 초기화합니다.
pub fn log_gpu_profile(device: &Device, tag: &str) {
    match device {
        #[cfg(feature = "vulkan")]
        Device::Vulkan(d) => {
            if let Some(r) = d.profile_report(30, true) {
                println!("[VK-PROFILE] {tag}\n{r}");
            }
        }
        _ => {
            let _ = (device, tag);
        }
    }
}

/// 🌟 [VK-PROFILE] 시간표 초기화 (생성 시작 시점).
pub fn reset_gpu_profile(device: &Device) {
    match device {
        #[cfg(feature = "vulkan")]
        Device::Vulkan(d) => d.profile_reset(),
        _ => {
            let _ = device;
        }
    }
}

/// 현재 백엔드로 [VRAM-REPORT] 한 줄을 남깁니다 (태스크 종료 직후 등 정리 전 상태 확인용).
pub fn log_vram_report(gpu_id: usize, tag: &str) {
    #[cfg(feature = "rocm")]
    {
        if let Device::Rocm(d) = get_gpu_device(gpu_id) {
            log_rocm_vram_report(tag, &d);
        }
    }
    #[cfg(feature = "vulkan")]
    {
        if let Device::Vulkan(d) = get_gpu_device(gpu_id) {
            log_vulkan_vram_report(tag, &d);
        }
    }
    let _ = (gpu_id, tag);
}

pub fn get_gpu_device(id: usize) -> Device {
    {
        let cache = DEVICE_CACHE.lock().unwrap();
        if id < cache.len() {
            if let Some(dev) = &cache[id] {
                return dev.clone();
            }
        }
    }

    #[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
    let dev = {
        println!("[CUDA/ROCm/Vulkan] 🚀 Attempting to Create Primary Context on GPU {}...", id);
        let d = new_gpu_device(id).unwrap_or(Device::Cpu);
        println!("[CUDA/ROCm/Vulkan] ✅ Primary Context Created on GPU {} ({:?}).", id, d);
        #[cfg(feature = "vulkan")]
        if let Device::Vulkan(v) = &d {
            // 어떤 GPU(외장/내장)가 선택됐고, 가중치를 VRAM·공유 메모리 중 어디에 둘 수 있는지 남깁니다.
            print!("[Vulkan] {}", v.memory_diagnostics());
        }
        d
    };

    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan"), feature = "metal"))]
    let dev = {
        println!("[Metal] 🚀 Initializing Metal Context on GPU {}...", id);
        Device::new_metal(id).unwrap_or(Device::Cpu)
    };

    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan"), not(feature = "metal")))]
    let dev = Device::Cpu;

    {
        let mut cache = DEVICE_CACHE.lock().unwrap();
        if id < cache.len() {
            cache[id] = Some(dev.clone());
        }
    }
    dev
}

pub fn get_cuda_device(id: usize) -> Device {
    get_gpu_device(id)
}

pub fn get_best_device_info() -> (Device, usize) {
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        let probe = GpuMemProbe::new();
        let count = probe.device_count();
        if count > 0 {
            let mut best_id = 0;
            let mut max_free = 0;

            println!("[GPU-CHECK] Found {} CUDA/ROCm device(s).", count);

            for i in 0..count {
                if let Some((free, _)) = probe.mem_info(i) {
                    let free_gb = free as f64 / 1e9;
                    println!("[GPU-CHECK] GPU {}: {:.2} GB Free VRAM", i, free_gb);
                    if free > max_free {
                        max_free = free;
                        best_id = i;
                    }
                }
            }

            if max_free > 0 {
                println!("[GPU-CHECK] Selecting GPU {} as the best device.", best_id);
                return (get_gpu_device(best_id), best_id);
            }
        }
        println!("[GPU-CHECK] VRAM query failed or no free VRAM. Defaulting to GPU 0.");
        return (get_gpu_device(0), 0);
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "vulkan"))]
    {
        let count = GpuMemProbe::new().device_count();
        if count == 0 {
            println!("[GPU-CHECK] No Vulkan device found. Falling back to CPU.");
            return (Device::Cpu, 0);
        }
        println!("[GPU-CHECK] Found {} Vulkan device(s). Selecting GPU 0.", count);
        return (get_gpu_device(0), 0);
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan"), feature = "metal"))]
    {
        return (get_gpu_device(0), 0);
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan"), not(feature = "metal")))]
    {
        (Device::Cpu, 0)
    }
}

pub fn get_best_device() -> Device {
    get_best_device_info().0
}

pub fn get_device(device: Option<&Device>) -> Device {
    match device {
        Some(d) => d.clone(),
        None => get_best_device()
    }
}

pub fn get_gpu_sm_arch() -> Result<f32> {
    #[cfg(feature = "cuda")]
    {
        let output = Command::new("nvidia-smi")
            .arg("--query-gpu=compute_cap")
            .arg("--format=csv,noheader")
            .output()
            .map_err(|e| anyhow::anyhow!(format!("Failed to execute nvidia-smi: {}", e)))?;
        if !output.status.success() {
            return Err(anyhow::anyhow!(format!(
                "nvidia-smi failed with status: {}
Error: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        let output_str = String::from_utf8_lossy(&output.stdout);
        let output_str = output_str.trim();
        
        let first_line = output_str.lines().next().unwrap_or("0.0");
        let sm_float = match first_line.parse::<f32>() {
            Ok(num) => num,
            Err(_) => {
                return Err(anyhow::anyhow!(format!(
                    "gpu sm arch: {} parse float32 error",
                    first_line
                )));
            }
        };
        Ok(sm_float)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(0.0)
    }
}

#[derive(Clone, Debug)]
pub struct DeviceConfig {
    pub device: Device,
    pub is_cpu: bool,
    pub classify_chunk_size: usize,
    pub extract_chunk_size: usize,
    pub name: String,
    pub gpu_id: usize,
}

pub fn get_optimal_device_config() -> DeviceConfig {
    let (device, gpu_id) = get_best_device_info();
    let is_cpu = device.is_cpu();
    
    let name = if device.is_cuda() {
        format!("CUDA (GPU {})", gpu_id)
    } else if device.is_rocm() {
        format!("ROCm (GPU {})", gpu_id)
    } else if device.is_vulkan() {
        let vk_name = device
            .as_vulkan_device()
            .map(|d| d.name().to_string())
            .unwrap_or_default();
        format!("Vulkan (GPU {}: {})", gpu_id, vk_name)
    } else if device.is_metal() {
        "Metal".to_string()
    } else {
        "CPU".to_string()
    };

    DeviceConfig {
        device,
        is_cpu,
        classify_chunk_size: 12_000, 
        extract_chunk_size: 12_000,
        name,
        gpu_id,
    }
}

pub fn get_dtype(dtype: Option<DType>, cfg_dtype: &str) -> DType {
    match dtype {
        Some(d) => d,
        None => {
            let is_cuda = cfg!(feature = "cuda");
            let is_rocm = cfg!(feature = "rocm");
            let is_metal = cfg!(feature = "metal");
            let is_vulkan = cfg!(feature = "vulkan");

            if is_vulkan && get_best_device().is_vulkan() {
                match cfg_dtype {
                    "float64" | "double" => DType::F64,
                    "uint8" => DType::U8,
                    "int8" | "int16" | "int32" | "int64" => DType::I64,
                    _ => DType::F32,
                }
            } else if (is_cuda || is_rocm || is_metal) && !get_best_device().is_cpu() {
                match cfg_dtype {
                    "float32" | "float" => DType::F32,
                    "float64" | "double" => DType::F64,
                    "float16" => DType::F16,
                    "bfloat16" => {
                        if is_cuda {
                            let arch = get_gpu_sm_arch();
                            match arch {
                                Err(_) => DType::F16,
                                Ok(a) => if a >= 8.0 { DType::BF16 } else { DType::F16 }
                            }
                        } else if is_rocm {
                            DType::BF16
                        } else {
                            DType::F16
                        }
                    }
                    "uint8" => DType::U8,
                    "int8" | "int16" | "int32" | "int64" => DType::I64,
                    _ => DType::F32,
                }
            } else {
                match cfg_dtype {
                    "float32" | "float" => DType::F32,
                    "float64" | "double" => DType::F64,
                    "float16" | "bfloat16" => DType::F16,
                    "uint8" => DType::U8,
                    "int8" | "int16" | "int32" | "int64" => DType::I64,
                    _ => DType::F32,
                }
            }
        }
    }
}
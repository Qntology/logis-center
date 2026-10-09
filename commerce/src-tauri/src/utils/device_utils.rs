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
            if d.release_cached_resources().is_err() {
                let _ = d.trim_memory_pool();
            }
        }
    }
    #[cfg(feature = "vulkan")]
    {
        if let Device::Vulkan(d) = get_gpu_device(gpu_id) {
            let _ = d.trim_memory_pool();
        }
    }
    let _ = gpu_id;
}

pub fn trim_idle_gpu_pool(device: &Device) {
    match device {
        #[cfg(feature = "vulkan")]
        Device::Vulkan(d) => {
            if d.pooled_bytes() >= (64 << 20) {
                let _ = d.trim_memory_pool();
            }
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(d) => {
            let _ = d.trim_memory_pool();
        }
        _ => {
            let _ = device;
        }
    }
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
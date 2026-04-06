pub mod img_utils;
pub mod tensor_utils;
pub mod hash;
pub mod paths;
pub mod compression;
pub mod resources;
pub mod direct_loader;
pub mod network;
pub mod crypto;

use candle_core::{Device, DType};
use anyhow::Result;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::process::Command;
use nvml_wrapper::Nvml;
use once_cell::sync::Lazy;
use std::sync::Mutex;

// [FIX] 장치 번호별로 장치 객체를 캐싱하여 중복 생성 방지 (DeviceId 폭주 해결)
static DEVICE_CACHE: Lazy<Mutex<Vec<Option<Device>>>> = Lazy::new(|| Mutex::new(vec![None; 8]));

pub fn get_gpu_device(id: usize) -> Device {
    {
        let mut cache = DEVICE_CACHE.lock().unwrap();
        if id < cache.len() {
            if let Some(dev) = &cache[id] {
                return dev.clone();
            }
        }
    }

    #[cfg(feature = "cuda")]
    let dev = {
        println!("[CUDA/ROCm] 🚀 Attempting to Create Primary Context on GPU {}...", id);
        let d = Device::new_cuda(id).unwrap_or(Device::Cpu);
        println!("[CUDA/ROCm] ✅ Primary Context Created on GPU {}.", id);
        d
    };

    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    let dev = {
        println!("[Metal] 🚀 Initializing Metal Context on GPU {}...", id);
        Device::new_metal(id).unwrap_or(Device::Cpu)
    };

    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
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
    // 1. 네이티브 백엔드 시도 (CUDA/ROCm)
    #[cfg(feature = "cuda")]
    {
        if let Ok(nvml) = Nvml::init() {
            if let Ok(count) = nvml.device_count() {
                let mut best_id = 0;
                let mut max_free = 0;
                
                println!("[GPU-CHECK] Found {} CUDA-capable device(s).", count);
                
                for i in 0..count {
                    if let Ok(device) = nvml.device_by_index(i) {
                        if let Ok(mem) = device.memory_info() {
                            let free_gb = mem.free as f64 / 1e9;
                            println!("[GPU-CHECK] GPU {}: {:.2} GB Free VRAM", i, free_gb);
                            if mem.free > max_free {
                                max_free = mem.free;
                                best_id = i;
                            }
                        }
                    }
                }
                
                if max_free > 0 {
                    println!("[GPU-CHECK] Selecting GPU {} as the best device.", best_id);
                    return (get_gpu_device(best_id as usize), best_id as usize);
                }
            }
        }
        println!("[GPU-CHECK] NVML failed or no free VRAM. Defaulting to GPU 0.");
        return (get_gpu_device(0), 0);
    }

    // 2. Mac 가속 시도 (Metal)
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        return (get_gpu_device(0), 0);
    }

    // 3. CPU 기본
    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
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
    
    let name = if cfg!(feature = "cuda") && !is_cpu {
        format!("CUDA/ROCm (GPU {})", gpu_id)
    } else if cfg!(feature = "metal") && !is_cpu {
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
            let is_metal = cfg!(feature = "metal");

            if (is_cuda || is_metal) && !get_best_device().is_cpu() {
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
                        } else {
                            DType::F16 // Metal 등은 우선 F16 권장
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

pub fn string_to_static_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

pub fn find_type_files(path: &str, extension_type: &str) -> Result<Vec<String>> {
    let mut files = Vec::new();

    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();

        if file_path.is_file() {
            if let Some(extension) = file_path.extension() {
                 if extension == extension_type {
                    files.push(file_path.to_string_lossy().to_string());
                 }
            }
        }
    }

    Ok(files)
}

pub fn get_logit_processor(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: u64,
) -> LogitsProcessor {
    let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
    match top_k {
        None => LogitsProcessor::new(
            seed,
            temperature.map(|temp| temp as f64),
            top_p.map(|tp| tp as f64),
        ),
        Some(k) => {
            let sampling = match temperature {
                None => Sampling::ArgMax,
                Some(temperature) => match top_p {
                    None => Sampling::TopK {
                        k,
                        temperature: temperature as f64,
                    },
                    Some(p) => Sampling::TopKThenTopP {
                        k,
                        p: p as f64,
                        temperature: temperature as f64,
                    },
                },
            };
            LogitsProcessor::from_sampling(seed, sampling)
        }
    }
}

pub fn round_by_factor(num: u32, factor: u32) -> u32 {
    let round = (num as f32 / factor as f32).round() as u32;
    round * factor
}

pub fn floor_by_factor(num: f32, factor: u32) -> u32 {
    let floor = (num / factor as f32).floor() as u32;
    floor * factor
}

pub fn ceil_by_factor(num: f32, factor: u32) -> u32 {
    let ceil = (num / factor as f32).ceil() as u32;
    ceil * factor
}

// --- GLOBAL EXTRACTION CONTROL ---

pub fn is_extraction_stopped() -> bool {
    paths::get_stop_signal_file().exists()
}

pub fn set_extraction_stop_signal(stopped: bool) {
    let file = paths::get_stop_signal_file();
    if stopped {
        let _ = std::fs::File::create(file);
    } else {
        let _ = std::fs::remove_file(file);
    }
}
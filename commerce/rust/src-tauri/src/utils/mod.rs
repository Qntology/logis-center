pub mod img_utils;
pub mod tensor_utils;
pub mod hash;
pub mod paths;
pub mod compression;
pub mod resources;

use candle_core::{Device, DType};
use anyhow::Result;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::process::Command;
use nvml_wrapper::Nvml;
use tokio::sync::OnceCell;

static CUDA_DEVICE: OnceCell<Device> = OnceCell::const_new();

pub async fn get_cuda_device(id: usize) -> Device {
    let dev = CUDA_DEVICE.get_or_init(|| async {
        println!("[CUDA] Initializing Persistent CUDA Device {}...", id);
        Device::new_cuda(id).unwrap_or(Device::Cpu)
    }).await;
    dev.clone()
}

pub fn get_best_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        // ... (This function is synchronous, we might need a sync wrapper or just use the logic from LogisModel)
        // For now, let's keep get_best_device but discourage its use in hot paths
        if let Ok(nvml) = Nvml::init() {
            if let Ok(count) = nvml.device_count() {
                let mut best_id = 0;
                let mut max_free = 0;
                println!("[DEVICE-SCAN] Found {} NVIDIA GPUs.", count);
                
                for i in 0..count {
                    if let Ok(device) = nvml.device_by_index(i) {
                        if let Ok(mem) = device.memory_info() {
                            println!("[DEVICE-SCAN] GPU {}: Free {:.2} GB / Total {:.2} GB", 
                                i, mem.free as f64 / 1e9, mem.total as f64 / 1e9);
                            if mem.free > max_free {
                                max_free = mem.free;
                                best_id = i;
                            }
                        }
                    }
                }
                
                if max_free > 0 {
                    println!("✅ [DEVICE-SELECT] Best GPU is ID {} with {:.2} GB Free.", best_id, max_free as f64 / 1e9);
                    return Device::new_cuda(best_id as usize).unwrap_or(Device::Cpu);
                }
            }
        }
        println!("[DEVICE-SCAN] NVML failed or no GPUs found. Trying default CUDA 0...");
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}

pub fn get_device(device: Option<&Device>) -> Device {
    match device {
        Some(d) => d.clone(),
        None => get_best_device()
    }
}

pub fn get_gpu_sm_arch() -> Result<f32> {
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
    
    // Multi-GPU 환경에서는 여러 줄이 나올 수 있으므로 첫 번째 줄만 파싱
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
    #[cfg(feature = "cuda")]
    {
        if let Ok(nvml) = Nvml::init() {
            if let Ok(count) = nvml.device_count() {
                let mut best_id = 0;
                let mut max_free = 0;
                
                for i in 0..count {
                    if let Ok(device) = nvml.device_by_index(i) {
                        if let Ok(mem) = device.memory_info() {
                            if mem.free > max_free {
                                max_free = mem.free;
                                best_id = i;
                            }
                        }
                    }
                }
                
                if max_free > 0 {
                    println!("🚀 [DEVICE-CONFIG] Selected GPU-{} with {:.2} GB Free.", best_id, max_free as f64 / 1e9);
                    return DeviceConfig {
                        device: Device::new_cuda(best_id as usize).unwrap_or(Device::Cpu),
                        is_cpu: false,
                        classify_chunk_size: 12_000, 
                        extract_chunk_size: 12_000,
                        name: format!("GPU-{}", best_id),
                        gpu_id: best_id as usize,
                    };
                }
            }
        }
    }

    // Fallback to CPU or if no CUDA
    DeviceConfig {
        device: Device::Cpu,
        is_cpu: true,
        classify_chunk_size: 12_000,  
        extract_chunk_size: 12_000,   
        name: "CPU".to_string(),
        gpu_id: 0,
    }
}

pub fn get_dtype(dtype: Option<DType>, cfg_dtype: &str) -> DType {
    match dtype {
        Some(d) => d,
        None => {
            #[cfg(feature = "cuda")]
            {
                match cfg_dtype {
                    "float32" | "float" => DType::F32,
                    "float64" | "double" => DType::F64,
                    "float16" => DType::F16,
                    "bfloat16" => {
                        let arch = get_gpu_sm_arch();
                        match arch {
                            Err(_) => DType::F16,
                            Ok(a) => {
                                if a >= 8.0 { DType::BF16 } else { DType::F16 }
                            }
                        }
                    }
                    "uint8" => DType::U8,
                    "int8" | "int16" | "int32" | "int64" => DType::I64,
                    _ => DType::F32,
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
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
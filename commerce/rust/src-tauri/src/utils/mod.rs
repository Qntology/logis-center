pub mod img_utils;
pub mod tensor_utils;
pub mod hash;
pub mod paths;
pub mod compression;
pub mod resources;
pub mod direct_loader;
pub mod config;
pub mod progress;
pub mod gptq;
pub mod image;
pub mod gguf_varbuilder;
pub mod logits_processor;

pub use config::{resolve_qwen3_hybrid_config, Qwen3HybridConfig};

#[macro_export]
macro_rules! serde_default {
    ($t:ty, $name:ident, $v:expr) => {
        fn $name() -> $t {
            $v
        }
    };
}

use candle_core::{Device, DType};
use anyhow::Result;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::process::Command;
use nvml_wrapper::Nvml;
use once_cell::sync::Lazy;
use std::sync::Mutex;

// [FIX] 장치 번호별로 장치 객체를 캐싱하여 중복 생성 방지 (DeviceId 폭주 해결)
static DEVICE_CACHE: Lazy<Mutex<Vec<Option<Device>>>> = Lazy::new(|| Mutex::new(vec![None; 8]));

pub fn get_cuda_device(id: usize) -> Device {
    let mut cache = DEVICE_CACHE.lock().unwrap();
    if id < cache.len() {
        if let Some(dev) = &cache[id] {
            return dev.clone();
        }
        println!("[CUDA] 🚀 Initializing Primary Context on GPU {}...", id);
        let dev = Device::new_cuda(id).unwrap_or(Device::Cpu);
        cache[id] = Some(dev.clone());
        dev
    } else {
        Device::new_cuda(id).unwrap_or(Device::Cpu)
    }
}

pub fn get_best_device() -> Device {
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
                    return get_cuda_device(best_id as usize);
                }
            }
        }
        get_cuda_device(0)
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
                        device: get_cuda_device(best_id as usize),
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

pub fn get_llama4_attn_scale(
    positions: &candle_core::Tensor,
    llama_4_scaling_beta: f64,
    original_max_position_embeddings: f64,
) -> candle_core::Result<candle_core::Tensor> {
    let div = (positions.to_dtype(DType::F32)? / original_max_position_embeddings)?;
    let floored = div.floor()?;

    let one = floored.ones_like()?; // tensor filled with 1.0
    let log_term = (one + floored)?.log()?;

    let scaling = (1f64 + (llama_4_scaling_beta * &log_term)?)?;
    scaling
        .unsqueeze(candle_core::D::Minus1)?
        .unsqueeze(0)?
        .unsqueeze(0)
}

pub fn module_path_matches_not_convert(module_path: &str, item: &str) -> bool {
    let module_path = module_path.trim_end_matches(".weight");
    let item = item.trim_end_matches(".weight");
    module_path == item
        || module_path.ends_with(item)
        || module_path.ends_with(&format!(".{item}"))
        || item.ends_with(module_path)
        || item.ends_with(&format!(".{module_path}"))
}

pub fn should_skip_fp8_for_module(module_path: &str, cfg: &crate::utils::config::QuantConfig) -> bool {
    if module_path.is_empty() || cfg.modules_to_not_convert.is_empty() {
        return false;
    }
    cfg.modules_to_not_convert
        .iter()
        .any(|item| module_path_matches_not_convert(module_path, item))
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
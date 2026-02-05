pub mod img_utils;
pub mod hash;
pub mod paths;
pub mod compression;
pub mod resources;

use anyhow::Result;
use std::process::Command;
use nvml_wrapper::Nvml;

#[derive(Clone, Debug)]
pub struct DeviceConfig {
    pub is_cpu: bool,
    pub classify_chunk_size: usize,
    pub extract_chunk_size: usize,
    pub name: String,
    pub gpu_id: usize,
}

pub fn get_optimal_device_config() -> DeviceConfig {
    // 1. Environment Override
    if let Ok(force_id_str) = std::env::var("LOGIS_FORCE_GPU") {
        if let Ok(force_id) = force_id_str.parse::<usize>() {
            println!("[DEVICE-SELECT] FORCING GPU-{} as requested.", force_id);
            return DeviceConfig { is_cpu: false, classify_chunk_size: 12_000, extract_chunk_size: 12_000, name: format!("GPU-{}", force_id), gpu_id: force_id };
        }
    }

    // 2. Intelligent Auto-Selection
    if let Ok(nvml) = Nvml::init() {
        if let Ok(count) = nvml.device_count() {
            println!("[DEVICE-SELECT] Total GPUs detected by NVML: {}", count);
            let mut candidates = Vec::new();
            for i in 0..count {
                if let Ok(device) = nvml.device_by_index(i) {
                    if let Ok(mem) = device.memory_info() {
                        let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
                        println!("[DEVICE-SELECT] Candidate GPU-{}: {} | Free: {:.2} GiB / Total: {:.2} GiB", 
                            i, name, mem.free as f64 / 1073741824.0, mem.total as f64 / 1073741824.0);
                        candidates.push((i, mem.total, mem.free, name));
                    }
                }
            }
            
            // Sort by FREE VRAM Descending (Priority to the card with more room)
            candidates.sort_by(|a, b| b.2.cmp(&a.2));

            if let Some(best) = candidates.first() {
                if best.1 > 1024 * 1024 * 1024 { // If it's a dedicated card (>1GB)
                    println!("[DEVICE-SELECT] Automatic: Picking {} (Free: {:.2} GiB, Total: {:.2} GiB) at index {}.", 
                        best.3, best.2 as f64 / 1073741824.0, best.1 as f64 / 1073741824.0, best.0);
                    
                    return DeviceConfig {
                        is_cpu: false,
                        classify_chunk_size: 12_000, 
                        extract_chunk_size: 12_000,
                        name: best.3.clone(),
                        gpu_id: best.0 as usize,
                    };
                }
            }
        }
    }

    println!("[DEVICE-SELECT] No suitable GPU found or insufficient VRAM. Falling back to CPU.");
    DeviceConfig {
        is_cpu: true,
        classify_chunk_size: 12_000,  
        extract_chunk_size: 12_000,   
        name: "CPU".to_string(),
        gpu_id: 0,
    }
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

pub fn string_to_static_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

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

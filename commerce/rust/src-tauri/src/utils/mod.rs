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
    if let Ok(nvml) = Nvml::init() {
        if let Ok(count) = nvml.device_count() {
            let mut best_id = 0;
            let mut max_total = 0;
            let mut best_free = 0;
            let mut found_gpu = false;
            
            for i in 0..count {
                if let Ok(device) = nvml.device_by_index(i) {
                    let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
                    if let Ok(mem) = device.memory_info() {
                        let total_gib = mem.total as f64 / 1073741824.0;
                        let free_gib = mem.free as f64 / 1073741824.0;
                        println!("[DEVICE-CHECK] GPU-{}: {} (Total: {:.2} GiB, Free: {:.2} GiB)", i, name, total_gib, free_gib);
                        
                        // [CRITICAL-FIX] Prioritize the card with HIGHEST TOTAL VRAM (performance card)
                        if mem.total > max_total {
                            max_total = mem.total;
                            best_free = mem.free;
                            best_id = i;
                            found_gpu = true;
                        }
                    }
                }
            }
            
            if found_gpu && best_free > 500 * 1024 * 1024 { // Ensure the best card has enough free space
                if let Ok(device) = nvml.device_by_index(best_id as u32) {
                    let name = device.name().unwrap_or_default();
                    println!("[DEVICE-SELECT] Choosing best-performance GPU-{} ({}) with {:.2} GiB free.", 
                        best_id, name, best_free as f64 / 1073741824.0);
                    return DeviceConfig {
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

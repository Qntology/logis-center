use crate::utils;
use anyhow::anyhow;
use crate::models::qwen::generate::QwenVLGenerateModel;
use crate::models::qwen3_5::generate::Qwen3_5GenerateModel;
use crate::models::embedding::EmbeddingModel;
use crate::openai_types::{
    ChatCompletionParameters,
    ChatCompletionRequestMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestMessageContentPart,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartImage,
    ImageURL,
};
use candle_core::{Device, DType};
use image::DynamicImage;
use serde_json::{Value, json, Map};
use std::sync::{Arc, atomic::AtomicBool};
use tauri::Emitter;
use std::io::Cursor;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use sysinfo::System;

pub struct Spinner {
    pub frames: Vec<&'static str>,
    pub interval: u64,
}

impl Spinner {
    pub fn dots() -> Self {
        Self {
            frames: vec!["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            interval: 80,
        }
    }
}

pub fn generate_rich_summary(doc_type: &str, data: &Value) -> String {
    let type_map = json!({
        "CI": "Commercial Invoice", "PI": "Proforma Invoice", "PL": "Packing List",
        "BL": "Bill of Lading", "AWB": "Air Waybill", "CO": "Certificate of Origin", "LC": "Letter of Credit",
        "tracking": "Shipping Label / Tracking Info"
    });
    
    let full_type = type_map.get(doc_type).and_then(|s| s.as_str()).unwrap_or(doc_type);
    let mut parts = vec![format!("This is a {} document.", full_type)];

    if let Some(h) = data.get("header") {
        if let Some(no) = h.get("document_number").and_then(|s| s.as_str()) {
            if no != "N/A" && !no.is_empty() {
                parts.push(format!("Document number is {}.", no));
            }
        }
        if let Some(date) = h.get("issue_date").and_then(|s| s.as_str()) {
            if date != "N/A" && !date.is_empty() {
                parts.push(format!("Issued on {}.", date));
            }
        }
    }

    if doc_type == "tracking" {
        if let Some(tn) = data.get("tracking_number").and_then(|s| s.as_str()) {
            parts.push(format!("The tracking number is {}.", tn));
        }
        if let Some(text) = data.get("text").and_then(|s| s.as_str()) {
            parts.push(text.to_string());
        }
    }

    if let Some(p) = data.get("parties") {
        let sup = p.get("supplier_name").and_then(|s| s.as_str());
        let buy = p.get("buyer_name").and_then(|s| s.as_str());
        
        let has_sup = sup.is_some() && sup.unwrap() != "N/A";
        let has_buy = buy.is_some() && buy.unwrap() != "N/A";

        if has_sup && has_buy {
            parts.push(format!("Transaction involved {} as the supplier/shipper and {} as the buyer/consignee.", sup.unwrap(), buy.unwrap()));
        } else if has_sup {
            parts.push(format!("Supplier/Shipper is {}.", sup.unwrap()));
        } else if has_buy {
            parts.push(format!("Buyer/Consignee is {}.", buy.unwrap()));
        }
    }

    if let Some(f) = data.get("financials") {
        if let Some(amt) = f.get("amount_total") {
             let amt_str = if amt.is_number() { amt.to_string() } else { amt.as_str().unwrap_or("0").to_string() };
             let curr = f.get("currency_code").and_then(|s| s.as_str()).unwrap_or("USD");
             if amt_str != "0" && amt_str != "0.0" {
                 parts.push(format!("Total amount is {} {}.", amt_str, curr));
             }
        }
    }

    if let Some(l) = data.get("logistics") {
        let pol = l.get("location_port_of_loading").and_then(|s| s.as_str());
        let pod = l.get("location_port_of_discharge").and_then(|s| s.as_str());
        
        if let (Some(o), Some(d)) = (pol, pod) {
            if o != "N/A" && d != "N/A" {
                parts.push(format!("Shipped from {} to {}.", o, d));
            }
        }
        
        if let Some(mode) = l.get("transport_mode").and_then(|s| s.as_str()) {
            parts.push(format!("Transport mode is {}.", mode));
        }
    }

    if let Some(items) = data.get("line_items").and_then(|v| v.as_array()) {
        let mut item_descs = Vec::new();
        for item in items.iter().take(5) {
            if let Some(d) = item.get("description").and_then(|s| s.as_str()) {
                if d.len() > 3 { item_descs.push(d); }
            }
        }
        if !item_descs.is_empty() {
            parts.push(format!("Contains items: {}.", item_descs.join(", ")));
        }
    }
    
    parts.join(" ")
}

use tokio::sync::Mutex as TokioMutex;
use std::time::{Duration, Instant};

use crate::models::qwen3::generate::Qwen3GenerateModel; // 🌟 Qwen3 텍스트 전용 로직 임포트

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelSize {
    Qwen,    // 0.6B for Ingestion (기존 Small)
    Qwen3,   // Qwen3 Text Model (기존 Large, /qwen3/ 로직 전용)
    Qwen3_5, // 0.8B Qwen 3.5 (Text Optimized)
}

#[derive(Clone)]
pub struct LogisModel {
    pub app_handle: tauri::AppHandle,
    pub generator: Arc<TokioMutex<Option<QwenVLGenerateModel>>>, 
    // 🌟 [복구 완료] 사용자님이 원하시던 오리지널 Qwen3 텍스트 전용 로직을 띄웁니다!
    pub qwen3_generator: Arc<TokioMutex<Option<Qwen3GenerateModel>>>, 
    pub qwen3_5_generator: Arc<TokioMutex<Option<Qwen3_5GenerateModel>>>,
    
    pub embedding_model: Arc<TokioMutex<Option<EmbeddingModel>>>,

    pub is_cpu_mode: bool, 
    pub is_disk_swap: bool,
    pub dual_mode_enabled: bool,
    
    // Config for Lazy Reloading
    qwen_model_path: String,      // 🌟 (기존 small_model_path 대신 이름 맞춤)
    qwen3_model_path: String,     // 🌟 Qwen3 모델 경로 추가
    qwen3_5_model_path: String,
    embedding_path: std::path::PathBuf,
    device_config: utils::DeviceConfig,
    max_tokens_limit: u32,
    _dtype: Option<DType>, 
    current_size: Arc<TokioMutex<Option<ModelSize>>>,
}

impl LogisModel {
    pub async fn unload_generator(&self) {
        let mut gen = self.generator.lock().await;
        *gen = None;
        let mut q3_gen = self.qwen3_generator.lock().await; 
        *q3_gen = None;
        let mut q35_gen = self.qwen3_5_generator.lock().await;
        *q35_gen = None;
        
        let mut size = self.current_size.lock().await;
        *size = None;
        println!("[MODEL] All generators (Active) destroyed."); 
    }

    pub async fn unload_embedding(&self) {
        let mut emb = self.embedding_model.lock().await;
        if emb.is_some() {
            *emb = None;
            println!("[MODEL] Embedding Model unloaded to free VRAM.");
        }
    }

    // --- [NEW] CPU System RAM Optimizer ---
    async fn optimize_system_ram(&self) {
        if !self.is_cpu_mode { return; }

        let mut sys = System::new();
        sys.refresh_memory();
        let free_ram = sys.total_memory().saturating_sub(sys.used_memory()); // Bytes in sysinfo 0.30

        // Threshold: 4GB
        if free_ram < 4 * 1024 * 1024 * 1024 {
            println!("[RAM-WATCH] Low System Memory ({:.2} GB). Flushing Working Set...", free_ram as f64 / 1024.0 / 1024.0 / 1024.0);
            
            #[cfg(target_os = "windows")]
            unsafe {
                use windows_sys::Win32::System::Threading::GetCurrentProcess;
                use windows_sys::Win32::System::Memory::SetProcessWorkingSetSizeEx;
                use windows_sys::Win32::System::Memory::QUOTA_LIMITS_HARDWS_MIN_DISABLE;
                use windows_sys::Win32::System::Memory::QUOTA_LIMITS_HARDWS_MAX_DISABLE;
                let process = GetCurrentProcess();
                let _ = SetProcessWorkingSetSizeEx(process, usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
            }
            
            tokio::time::sleep(Duration::from_millis(500)).await;
        } else {
            println!("[RAM-WATCH] Sufficient Memory ({:.2} GB). Skipping flush.", free_ram as f64 / 1024.0 / 1024.0 / 1024.0);
        }
    }

    /// [CLEANUP] Aggressive Factory Reset Purge (Reinforced with Diagnostics)
    pub async fn deep_purge_resources(&self) {
        println!("[DIAG-PURGE] Step 0: Waiting for background IO to finish...");
        crate::models::qwen::generate::wait_for_global_io().await; // [cite: 254]

        println!("[DIAG-PURGE] Step 1: Clearing ALL Generation Slots...");
        
        {
            let mut gen = self.generator.lock().await;
            if let Some(mut g) = gen.take() {
                println!("[DIAG-PURGE] Dropping Active Generator (0.6B)...");
                let _ = g.clear_kv_cache();
                let _ = g.qwen.drop_kv_storage(); 
                drop(g); 
            }
        }
        
        // 🌟 [신규] Qwen3 (텍스트 전용) 슬롯 해제 추가
        {
            let mut q3_gen = self.qwen3_generator.lock().await;
            if let Some(mut g) = q3_gen.take() {
                println!("[DIAG-PURGE] Dropping Qwen3 Generator...");
                g.clear_kv_cache(); // Qwen3 구조체에 구현된 캐시 클리어 호출
                drop(g);
            }
        }

        {
            let mut q35_gen = self.qwen3_5_generator.lock().await;
            if let Some(mut g) = q35_gen.take() {
                println!("[DIAG-PURGE] Dropping Qwen 3.5 Generator..."); //
                g.clear_kv_cache();
                drop(g);
            }
        }
        
        println!("[DIAG-PURGE] Step 2: Clearing Embedding Model...");
        {
            let mut emb = self.embedding_model.lock().await;
            if let Some(e) = emb.take() { 
                drop(e); 
            }
        }
        
        println!("[DIAG-PURGE] Step 3: Synchronizing CUDA Context...");
        if !self.is_cpu_mode {
            let dev = self.device_config.device.clone();
            let sync_res = tokio::time::timeout(Duration::from_secs(10), tokio::task::spawn_blocking(move || {
                if dev.is_cuda() { 
                    println!("[DIAG-PURGE] Executing dev.synchronize()...");
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        dev.synchronize()
                    }))
                } else { Ok(Ok(())) }
            })).await;
            
            match sync_res {
                Ok(Ok(Ok(Ok(_)))) => println!("[DIAG-PURGE] CUDA Synchronization Successful."),
                Ok(Ok(Ok(Err(e)))) => println!("[DIAG-PURGE] CUDA Sync Error: {:?}", e),
                Ok(Err(_)) => println!("[DIAG-PURGE] CUDA Sync Task Join Error."),
                Err(_) => println!("[DIAG-PURGE] CUDA Sync Timeout! Continuing purge."),
                _ => println!("[DIAG-PURGE] CUDA Sync Panicked or Failed."),
            }
        }

        println!("[DIAG-PURGE] Step 4: Flushing OS Memory...");
        #[cfg(target_os = "windows")]
        unsafe {
            use windows_sys::Win32::System::Threading::*;
            use windows_sys::Win32::System::Memory::*;
            let current_process = GetCurrentProcess();
            let _ = SetProcessWorkingSetSizeEx(current_process, usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
        }
        #[cfg(target_os = "linux")]
        unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
        #[cfg(target_os = "macos")]
        unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }

        println!("[DIAG-PURGE] Aggressive Purge Complete.");
        tokio::time::sleep(Duration::from_millis(300)).await;
    }

    // --- [NEW] VRAM Settlement Monitor (Smart Polling) ---
    async fn wait_for_vram_settle(&self, target_free_mb: u64, timeout_sec: u64, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<()> {
        if self.is_cpu_mode { return Ok(()); } 

        println!("[VRAM-WATCH] Monitoring VRAM (Target > {} MB)...", target_free_mb);
        let start = Instant::now();
        let target_bytes = target_free_mb * 1024 * 1024;
        let mut last_free = 0;
        let mut stable_ticks = 0;
        let mut increasing_ticks = 0;
        let mut has_flushed_ram = false;

        loop {
            // 1. Cancellation Check
            if let Some(token) = &cancel_token {
                if token.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err(anyhow::anyhow!("Task cancelled during VRAM wait"));
                }
            }

            // 2. Measure VRAM
            let mut current_free = 0;
            use nvml_wrapper::Nvml;
            if let Ok(nvml) = Nvml::init() {
                if let Ok(dev) = nvml.device_by_index(self.device_config.gpu_id as u32) {
                    if let Ok(mem) = dev.memory_info() {
                        current_free = mem.free;
                    }
                }
            }

            // [FAST-PATH] Immediate Success
            if current_free >= target_bytes {
                if stable_ticks >= 2 { // Confirm stability for 1 sec
                    println!("[VRAM-WATCH] Success! VRAM Secured: {:.2} GB", current_free as f64 / 1e9);
                    break;
                }
                stable_ticks += 1;
            } else {
                stable_ticks = 0;
            }

            // [ADAPTIVE-LOGIC] Analyze Trend (20MB sensitivity)
            if current_free > last_free + (20 * 1024 * 1024) { 
                increasing_ticks += 1;
                println!("[VRAM-WATCH] Reclaiming... ({:.2} GB -> {:.2} GB)", last_free as f64/1e9, current_free as f64/1e9);
            } else {
                increasing_ticks = 0;
            }

            // [ACTIVE-FLUSH] If stuck for > 1.5s, trigger OS RAM cleanup
            if start.elapsed().as_secs_f32() > 1.5 && !has_flushed_ram && current_free < target_bytes {
                println!("[VRAM-WATCH] Triggering Aggressive OS Working Set Trim...");
                #[cfg(target_os = "windows")]
                unsafe {
                    use windows_sys::Win32::System::Threading::GetCurrentProcess;
                    use windows_sys::Win32::System::Memory::SetProcessWorkingSetSizeEx;
                    use windows_sys::Win32::System::Memory::QUOTA_LIMITS_HARDWS_MIN_DISABLE;
                    use windows_sys::Win32::System::Memory::QUOTA_LIMITS_HARDWS_MAX_DISABLE;
                    let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
                }
                has_flushed_ram = true;
                tokio::time::sleep(Duration::from_millis(300)).await;
                continue;
            }

            // [TIMEOUT-HANDLER]
            if start.elapsed().as_secs() > timeout_sec {
                if increasing_ticks > 0 {
                    println!("[VRAM-WATCH] Timeout reached but memory is freeing up. Extending wait...");
                    increasing_ticks = 0;
                    continue; 
                }
                println!("[VRAM-WATCH] Timeout reached. Proceeding with {:.2} GB (Target: {:.2} GB)", current_free as f64/1e9, target_free_mb as f64/1024.0);
                break;
            }

            last_free = current_free;
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        Ok(())
    }

    // --- [NEW] SSD Bridge Operations ---
    pub async fn save_kv_snapshot(&self, task_id: &str, kv_name: Option<String>, offset: usize) -> anyhow::Result<String> {
        let current_size = *self.current_size.lock().await;
        let is_q35 = current_size == Some(ModelSize::Qwen3_5);
        let generator_arc = self.generator.clone();
        let task_id_str = task_id.to_string();
        
        tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let path = crate::utils::paths::get_kv_dir(None).join(format!("{}.safetensors", task_id_str));
            if is_q35 {
                // Qwen3.5는 자체 flush 매커니즘을 사용하므로 패스만 반환
                Ok(path.to_string_lossy().to_string())
            } else {
                let mut gen_guard = generator_arc.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    println!("[SSD-BRIDGE] Saving KV snapshot to {:?}", path);
                    gen.save_kv_to_disk(&path, kv_name.as_deref(), offset)?;
                    Ok(path.to_string_lossy().to_string())
                } else {
                    Err(anyhow::anyhow!("No active generator to save snapshot from"))
                }
            }
        }).await?
    }

    pub async fn truncate_kv_cache(&self, len: usize) -> anyhow::Result<()> {
        let current_size = *self.current_size.lock().await;
        let is_q35 = current_size == Some(ModelSize::Qwen3_5);
        let generator_arc = self.generator.clone();
        let q35_arc = self.qwen3_5_generator.clone();

        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            if is_q35 {
                let mut gen_guard = q35_arc.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    gen.qwen3_5.language_model.truncate_kv_cache(len).map_err(|e| anyhow::anyhow!("Truncate failed: {}", e))
                } else {
                    Ok(())
                }
            } else {
                let mut gen_guard = generator_arc.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    gen.truncate_kv_cache(len).map_err(|e| anyhow::anyhow!("Truncate failed: {}", e))
                } else {
                    Ok(())
                }
            }
        }).await?
    }

    pub async fn load_kv_snapshot(&self, task_id: &str, kv_name: Option<String>) -> anyhow::Result<()> {
        let current_size = *self.current_size.lock().await;
        let is_q35 = current_size == Some(ModelSize::Qwen3_5);
        
        let generator_arc = self.generator.clone();
        let q35_arc = self.qwen3_5_generator.clone();
        let task_id_str = task_id.to_string();
        let kv_name_str = kv_name.unwrap_or_else(|| "text".to_string());

        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let kv_root = crate::utils::paths::get_kv_dir(None).join(&task_id_str);
            let kv_type = kv_name_str.split('/').last().unwrap_or("text");
            let kv_type = if kv_type == "inference" || kv_type == "reference" || kv_type.is_empty() { "text" } else { kv_type };

            // 🌟 [핵심 픽스] 현재 모델이 Qwen 3.5(0.8B)라면 0.8B 방(q35_arc)에 스냅샷을 로드합니다!
            if is_q35 {
                let mut q35_guard = q35_arc.blocking_lock();
                if let Some(gen) = q35_guard.as_mut() {
                    let target_kv_name = format!("{}/inference/{}", task_id_str, kv_type);
                    let target_kv_name = if !crate::utils::paths::get_kv_dir(None).join(&target_kv_name).exists() {
                        format!("{}/reference/{}", task_id_str, kv_type)
                    } else { target_kv_name };
                    
                    println!("[SSD-BRIDGE] Restoring Qwen 3.5 Registry from {}", target_kv_name);
                    gen.qwen3_5.language_model.restore_kv_registry(&target_kv_name)?;
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("No active Qwen 3.5 generator to load snapshot into"))
                }
            } else {
                // 0.6B / 2B 로직은 그대로 유지
                let paths_to_try = vec![
                    kv_root.join("inference").join(kv_type),
                    kv_root.join("reference").join(kv_type),
                    kv_root.clone(),
                ];

                let mut target_path = None;
                for p in paths_to_try {
                    if p.exists() && std::fs::read_dir(&p).map(|mut d| d.next().is_some()).unwrap_or(false) {
                        target_path = Some(p);
                        break;
                    }
                }

                if let Some(p) = target_path {
                    let mut gen_guard = generator_arc.blocking_lock();
                    if let Some(gen) = gen_guard.as_mut() {
                        println!("[SSD-BRIDGE] Loading Directory-based KV snapshot from {:?}", p);
                        gen.load_kv_from_disk(&p, None)?;
                        Ok(())
                    } else {
                        Err(anyhow::anyhow!("No active generator to load snapshot into"))
                    }
                } else {
                    println!("[SSD-BRIDGE] No snapshot found for {} (Checked deep paths)", task_id_str);
                    Ok(())
                }
            }
        }).await?
    }

    // --- File: src/model.rs ---
    
    pub async fn secure_vram_relay(&self, target_size: ModelSize, task_id: Option<&str>, cancel_token: Option<Arc<AtomicBool>>, is_baking: bool, kv_name: Option<String>) -> anyhow::Result<()> {
        let start_time = Instant::now();
        
        println!("[RELAY] Performing Deep Purge before loading {:?} (Baking: {})...", target_size, is_baking);
        self.deep_purge_resources().await; //
        
        if !self.is_cpu_mode {
            tokio::time::sleep(Duration::from_millis(500)).await;
            self.wait_for_vram_settle(2000, 5, cancel_token.clone()).await?;
        }

        // 🌟 [핵심 변경] Enum 타입에 따라 완벽하게 독립된 로더를 타도록 분기
        match target_size {
            ModelSize::Qwen => {
                // 기존 0.6B VLM 로직 (Small)
                self.ensure_generator_ext(ModelSize::Qwen, false, is_baking).await?;
                if let Some(tid) = task_id {
                    self.load_kv_snapshot(tid, kv_name).await?;
                }
            },
            ModelSize::Qwen3 => {
                // 🌟 신규 0.8B 텍스트 전용 로직 (기존 Large 위치 대체)
                // Part 1에서 만든 ensure_qwen3()를 호출하여 /qwen3/ 로직만 타게 합니다.
                self.ensure_qwen3().await?;
            },
            ModelSize::Qwen3_5 => {
                // 0.8B Qwen 3.5 로직
                self.ensure_qwen3_5(false).await?; // 🌟 ModelSize::Qwen3_5 대신 false 로 변경
            }
        }

        println!("[RELAY] Transition to {:?} complete in {:.2}s", target_size, start_time.elapsed().as_secs_f32());
        Ok(())
    }

    // --- [NEW] Base Context Baking (One-time Heavy Lifting) ---
    pub async fn ingest_pug_to_ssd(&self, task_id: &str, pug_content: &str, cancel_token: Option<Arc<AtomicBool>>, kv_name: Option<String>) -> anyhow::Result<()> {
        let base_session = format!("{}_base", task_id);
        
        // 1. Load Small Model Isolated (Full layers, no baking)
        self.secure_vram_relay(ModelSize::Qwen, None, cancel_token.clone(), false, None).await?; // 🌟 Small -> Qwen

        // 2. Ingest PUG content
        {
            let prompt = format!("{}\n\n[SYSTEM] Analyze the document structure.", pug_content);
            let mut gen_guard = self.generator.lock().await;
            if let Some(gen) = gen_guard.as_mut() {
                // Just prefill, no generation needed for base context
                gen.prefill_chunk(prompt, cancel_token.clone(), None).await?;
            }
        }

        // 3. Save Base Snapshot
        self.save_kv_snapshot(&base_session, kv_name, 0).await?;
        
        // [FIX] 베이킹 직후 모델을 파괴하지 않고 그대로 유지하여 컨텍스트 오류를 방지합니다.
        // self.unload_generator().await; 제거됨
        
        Ok(())
    }

    // --- File: src/model.rs (LogisModel 내부) ---
    
    pub async fn ensure_qwen3(&self) -> anyhow::Result<()> {
        let needs_load = { self.qwen3_generator.lock().await.is_none() };
        if needs_load {
            println!("[MODEL] Loading Qwen3 Text Model (0.6B GGUF) exclusively via NATIVE /qwen3/ logic...");
            self.unload_generator().await;
            {
                *self.current_size.lock().await = Some(ModelSize::Qwen3);
            }
            
            let path = self.qwen3_model_path.clone();
            let dev = self.device_config.device.clone();
            let dtype = if self.is_cpu_mode { Some(candle_core::DType::F32) } else { Some(candle_core::DType::BF16) };
            
            // 🌟 방금 만든 init_from_gguf 를 호출합니다!
            let gen_result = tokio::task::spawn_blocking(move || -> anyhow::Result<Qwen3GenerateModel> {
                Qwen3GenerateModel::init_from_gguf(&path, Some(&dev), dtype)
            }).await?;

            match gen_result {
                Ok(gen) => {
                    println!("[MODEL] 🎉 Qwen3 (0.6B GGUF) Native Model loaded successfully!");
                    *self.qwen3_generator.lock().await = Some(gen);
                },
                Err(e) => {
                    println!("\n==================================================");
                    println!("🚨 [CRITICAL ERROR] 0.6B GGUF 로딩 실패!");
                    println!("원인: {:?}", e);
                    println!("==================================================\n");
                    return Err(e);
                }
            }
        }
        Ok(())
    }


    async fn load_generator_internal(&self, path: &str, shared_config_path: Option<&str>, force_text_only: bool) -> anyhow::Result<QwenVLGenerateModel> {
        println!("[MODEL] Loading Generator from {} (Text-Only: {})...", path, force_text_only);
        let dev = self.device_config.device.clone();
        let dev_id = self.device_config.gpu_id;

        let dtype = if self.device_config.is_cpu { Some(DType::F32) } else { Some(DType::BF16) };
        let limit = self.max_tokens_limit;
        let path_clone = path.to_string();
        let shared_path = shared_config_path.map(|s| s.to_string());

        let handle_clone = self.app_handle.clone();

        let is_disk_swap = self.is_disk_swap;

        let generator = tokio::task::spawn_blocking(move || {
            let kv_root = crate::utils::paths::get_kv_dir(Some(&handle_clone));
            // [CRITICAL] Use init_with_config to force shared settings (Config + Tokenizer)
            QwenVLGenerateModel::init_with_config(
                &path_clone, 
                shared_path.as_deref(), // Tokenizer path
                shared_path.as_deref(), // Config path
                Some(&dev), dev_id, Some(&dev), dev_id, dtype, Some(limit as usize),
                force_text_only,
                false, // [NEW] baking_only
                is_disk_swap, 
                kv_root
            )
        }).await??;
        
        Ok(generator)
    }

    pub async fn ensure_generator(&self, size: ModelSize) -> anyhow::Result<()> {
        self.ensure_generator_ext(size, false, false).await
    }

    pub async fn ensure_generator_ext(&self, size: ModelSize, force_text_only: bool, baking_only: bool) -> anyhow::Result<()> {
        if size == ModelSize::Qwen3_5 {
            return self.ensure_qwen3_5(false).await; // 🌟 ModelSize::Qwen3_5 대신 false 로 변경
        }
        if size == ModelSize::Qwen3 {
            return self.ensure_qwen3().await; 
        }

        // 오직 ModelSize::Qwen 만 이 아래 로직을 탐
        let mut current_size_guard = self.current_size.lock().await;
        let mut gen_guard = self.generator.lock().await;

        if *current_size_guard == Some(size) && gen_guard.is_some() && !baking_only {
            return Ok(());
        }

        println!("[LOAD] Fresh loading {:?} from disk...", size);
        let path = &self.qwen_model_path; 
        
        // 🌟 [핵심 픽스] Qwen 모델도 로딩 전에 미리 방주인 등록!
        {
            *self.current_size.lock().await = Some(size);
        }
        
        let target_device = self.device_config.device.clone();
        let is_disk_swap = self.is_disk_swap;
        let dev_id = self.device_config.gpu_id;
        let dtype = if target_device.is_cpu() { Some(candle_core::DType::F32) } else { Some(candle_core::DType::BF16) };
        let limit = self.max_tokens_limit;
        let path_clone = path.to_string();
        let handle_clone = self.app_handle.clone();

        let gen = tokio::task::spawn_blocking(move || {
            let kv_root = crate::utils::paths::get_kv_dir(Some(&handle_clone));
            QwenVLGenerateModel::init_with_config(
                &path_clone, None, None, // 공유 설정(shared_path) 제거
                Some(&target_device), dev_id, Some(&target_device), dev_id, dtype, Some(limit as usize),
                force_text_only, baking_only, is_disk_swap, kv_root // [cite: 298, 299, 300]
            )
        }).await??;

        *gen_guard = Some(gen);
        *current_size_guard = Some(size);
        
        Ok(())
    }

    pub async fn ensure_qwen3_5(&self, needs_vision: bool) -> anyhow::Result<()> {
        let needs_load = {
            let guard = self.qwen3_5_generator.lock().await;
            if let Some(gen) = guard.as_ref() {
                let is_large = gen.pre_processor.is_some();
                is_large != needs_vision // 🌟 wants_large 대신 needs_vision 직접 사용
            } else {
                true
            }
        };

        if needs_load {
            println!("[MODEL] Loading Qwen 3.5 Generator (0.8B) (Vision: {})...", needs_vision);
            self.unload_generator().await; 
            
            // 🌟 [핵심 픽스] 여기서도 로딩 전에 미리 방주인 등록!
            {
                *self.current_size.lock().await = Some(ModelSize::Qwen3_5);
            }
            
            let path = self.qwen3_5_model_path.clone();
            let dev = self.device_config.device.clone();
            
            let gen = tokio::task::spawn_blocking(move || {
                let gguf_files = crate::utils::find_type_files(&path, "gguf").unwrap_or_default();
                let model_gguf = gguf_files.iter().find(|f| !f.contains("mmproj")).cloned().ok_or_else(|| anyhow::anyhow!("No model GGUF found"))?;
                
                // 🌟 [수정]
                let mmproj_gguf = if needs_vision {
                    gguf_files.iter().find(|f| f.contains("mmproj")).cloned()
                } else {
                    None
                };
                
                Qwen3_5GenerateModel::init_from_gguf(&model_gguf, mmproj_gguf.as_deref(), Some(&dev))
            }).await??;
            
            let mut q35_gen_guard = self.qwen3_5_generator.lock().await;
            *q35_gen_guard = Some(gen);
            
            // 🌟 [CRITICAL FIX] 시스템 장부에 Qwen3.5가 켜졌음을 명시하여 스냅샷 미아 발생 방지!
            let mut current_size_guard = self.current_size.lock().await;
            *current_size_guard = Some(ModelSize::Qwen3_5);
        }
        Ok(())
    }

    pub async fn ensure_embedding(&self) -> anyhow::Result<()> {
        let current_size = { *self.current_size.lock().await };
        
        // [STRATEGY] High-priority exclusion logic
        match current_size {
            Some(ModelSize::Qwen3) => { // 🌟 Large -> Qwen3
                // If Qwen3 is active, Embedding must stay on CPU to avoid OOM
                println!("[MODEL] Qwen3 model active. Forcing Embedding to CPU to prevent swapping.");
            },
            Some(ModelSize::Qwen) | Some(ModelSize::Qwen3_5) => { // 🌟 Small -> Qwen
                // Qwen/Qwen3.5 and Embedding can coexist. 
                println!("[MODEL] Qwen/Qwen3.5 model active. Embedding will coexist.");
            },
            None => {
                // No generator active, safe to clean up any leftovers
                self.unload_generator().await;
            }
        }

        let mut emb_guard = self.embedding_model.lock().await;
        if emb_guard.is_none() {
            let self_clone = self.embedding_path.clone();
            
            // Determine target device: CPU if Qwen3 is active, else use default GPU
            let target_device = if current_size == Some(ModelSize::Qwen3) { // 🌟 Large -> Qwen3
                candle_core::Device::Cpu 
            } else { 
                self.device_config.device.clone() 
            };
            
            println!("[MODEL] Loading Embedding Model on {:?}...", if target_device.is_cpu() { "CPU" } else { "GPU" });
            
            let target_device_clone = target_device.clone();
            let emb = tokio::task::spawn_blocking(move || {
                EmbeddingModel::new_with_device(&self_clone, &target_device_clone)
            }).await??;
            
            *emb_guard = Some(emb);
        }
        Ok(())
    }

    pub async fn new(app_handle: tauri::AppHandle, device_preference: Option<&str>) -> anyhow::Result<Self> {
        // Default to true for SSD-Swap unless user explicitly wants pure CPU
        let is_disk_swap = match device_preference {
            Some("cpu") => false,
            _ => true,
        };
        
        println!("[MODEL-00] Initializing LogisModel (Preference: {:?}, DiskSwap: {})", device_preference, is_disk_swap);

        let mut config = utils::get_optimal_device_config();
        
        if device_preference == Some("cpu") {
            println!("⚠️ [MODEL] EXPLICIT CPU MODE FORCED by user/system preference.");
            config = utils::DeviceConfig {
                device: Device::Cpu,
                is_cpu: true,
                classify_chunk_size: 12000,
                extract_chunk_size: 12000,
                name: "CPU-Forced".to_string(),
                gpu_id: 0,
            };
        } else {
            // [STABILITY] Use persistent global CUDA device (Synchronous Singleton)
            let persistent_dev = utils::get_cuda_device(config.gpu_id);
            config.device = persistent_dev;
            println!("🚀 [MODEL] Running in default mode ({})", config.name);
        }

        let base_path = std::fs::canonicalize("src-tauri/models").or_else(|_| std::fs::canonicalize("models"))?;
        
        // [FIX] Normalize UNC paths for Windows to prevent "builder error" in model loaders
        let normalize_path = |path: std::path::PathBuf| -> String {
            let s = path.to_string_lossy().to_string();
            if s.starts_with(r"\\?\") {
                s[4..].to_string()
            } else {
                s
            }
        };

        let qwen_model_path = normalize_path(base_path.join("Qwen3-0.6B-Instruct-gguf")); 
        let qwen3_model_path = normalize_path(base_path.join("Qwen3-0.6B-Instruct-gguf")); 
        let qwen3_5_model_path = normalize_path(base_path.join("Qwen3.5-0.8B-Instruct-gguf"));
        let embedding_path = base_path.join("embeddinggemma-300m");

        let max_tokens_limit = 65536; 

        Ok(Self {
            app_handle,
            generator: Arc::new(TokioMutex::new(None)),
            qwen3_generator: Arc::new(TokioMutex::new(None)), // 🌟 추가
            qwen3_5_generator: Arc::new(TokioMutex::new(None)),
            embedding_model: Arc::new(TokioMutex::new(None)),
            is_cpu_mode: config.is_cpu,
            is_disk_swap,
            dual_mode_enabled: true, 
            qwen_model_path,    // 🌟 교체
            qwen3_model_path,   // 🌟 교체
            qwen3_5_model_path,
            embedding_path,
            device_config: config.clone(),
            max_tokens_limit: max_tokens_limit as u32,
            _dtype: None, 
            current_size: Arc::new(TokioMutex::new(None)),
        })
    }

    pub async fn extract_from_image(
        &self,
        task_id: String,
        image_path: String,
        language: String,
        app_handle: &tauri::AppHandle,
        cancel_token: Option<Arc<AtomicBool>>,
        store_mutex: &Arc<tokio::sync::Mutex<Option<crate::store::VectorStore>>>,
    ) -> anyhow::Result<()> {
        // [QWEN3.5-INTEGRATION] 이미지 추출이므로 비전(true) 모드로 호출합니다.
        self.ensure_qwen3_5(true).await?; // 🌟 ModelSize::Large 에서 true 로 변경

        if let Ok(img) = image::open(&image_path) {
            let dynamic_image = image::DynamicImage::ImageRgb8(img.to_rgb8());
            let prompt = get_image_extraction_prompt("kr", &language, "tracking", "");
            
            let result_str = self.chat_with_qwen3_5_image_spinner(
                "You are a highly precise document data extraction assistant.", // 🌟 System 주입
                &prompt,                                                        // 🌟 User 주입
                Some(dynamic_image),
                app_handle, 
                "extraction-progress", 
                json!({ "category": "Vision Analysis", "summary": "Analyzing image with Qwen 3.5..." }), 
                1024, 
                cancel_token.clone(), 
                Some(task_id.clone())
            ).await?;
            
            println!("\n=======================================");
            println!("[DEBUG-VISION] 🤖 AI Raw Response:\n{}", result_str);
            println!("=======================================\n");

            let extracted_data = crate::parsing::parse_json_from_llm(&result_str);
            
            let nl = crate::parsing::json_to_natural_language(&extracted_data);
            let item_digest = crate::utils::hash::digest(&nl);

            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                // Fixed identities for parity
                let from_addr = "0x0000000000000000000000000000000000000000";
                let team_id = crate::utils::hash::hash_id(from_addr); // Default team
                let hashed_cc = crate::utils::hash::hash_id("local.image");

                // [STRICT PARITY] Use stable hashing for image results
                let raw_no = extracted_data.get("tracking_number").and_then(|s| s.as_str()).unwrap_or(&task_id);
                let clean_no = crate::utils::hash::normalize_numeric_homoglyphs(raw_no).replace("-", "").replace("_", "");
                
                // index = crc32(hashId(type + team + no))
                let index_val = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("tracking{}{}", team_id, clean_no)));
                // id = hashId(team + index)
                let hashed_id = crate::utils::hash::hash_id(&format!("{}{}", team_id, index_val));
                
                // ref = hashId(team + cc + no)
                let ref_val = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, hashed_cc, clean_no));

                // 👇 데이터가 Object가 아닐 경우를 대비해 안전하게 감싸줍니다.
                let mut final_data = if extracted_data.is_object() {
                    extracted_data.clone()
                } else {
                    // JSON 객체가 아니면, AI의 원본 응답을 "raw_output"이라는 키에 담아 객체로 만듭니다.
                    json!({ "raw_output": extracted_data })
                };

                // 이제 final_data는 무조건 Object임이 보장되므로 unwrap()이 안전합니다.
                final_data.as_object_mut().unwrap().insert("index".to_string(), json!(index_val));
                final_data.as_object_mut().unwrap().insert("id".to_string(), json!(hashed_id));

                let _ = db.upsert_item(
                    "commerce_tracking", 
                    &hashed_id, 
                    "tracking", 
                    final_data, 
                    None,
                    Some(from_addr),
                    Some(&team_id),
                    Some(&hashed_cc),
                    Some(&crate::utils::hash::hash_id(&format!("tracking{}", hashed_cc))), // bcc
                    Some(&ref_val),
                    Some(&item_digest)
                ).await;
            }
            
            let _ = app_handle.emit("extraction-progress", json!({ 
               "task_id": task_id.clone(),
               "category": "Done", "summary": "Analysis Complete", "spinner": "✅", "data": extracted_data
            }));
            
            Ok(())
        } else {
            let _ = app_handle.emit("extraction-progress", json!({ 
               "task_id": task_id.clone(),
               "category": "Error", "summary": "Failed to load image file.", "spinner": "❌"
            }));
            Ok(())
        }
    }

    pub async fn chat_with_qwen3_5_image_spinner(
        &self, 
        system: &str,       
        user_input: &str,   
        image: Option<DynamicImage>,
        _app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        max_tokens: usize,
        cancellation_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>
    ) -> anyhow::Result<String> {
        // [VISION-DYNAMIC] 🌟 target_size 로직 삭제하고 바로 bool 전달
        self.ensure_qwen3_5(image.is_some()).await?;

        // [FIX] Inject task_id from session_id if it's a task reference
        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        // [LOG] Save to task history if task_id exists
        if let Some(task_id) = base_payload.get("task_id").and_then(|v| v.as_str()) {
            crate::scheduler::log_task_progress(_app_handle, task_id, &base_payload);
        }

        let mut q35_gen_guard = self.qwen3_5_generator.lock().await;
        let gen = q35_gen_guard.as_mut().ok_or_else(|| anyhow!("Qwen 3.5 Generator is unloaded"))?;
        
        let mut content_parts = Vec::new();
        
        if let Some(img) = image {
            let mut buf = Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            let b64 = BASE64_STANDARD.encode(buf.into_inner());
            let url = format!("data:image/png;base64,{}", b64);
            
            content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageURL { url, detail: None }
                }
            ));
        }

        // User Text 할당
        content_parts.push(ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: user_input.to_string() }
        ));

        // System 메시지 명시적 생성
        let system_message = ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage {
            content: system.to_string(),
            name: None,
        });

        // User 메시지 명시적 생성
        let user_message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        // 파라미터 세팅
        let params = ChatCompletionParameters {
            messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
            model: "qwen3.5".to_string(),
            max_tokens: Some(max_tokens as u32),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(
            params, 
            cancellation_token.clone(),
            session_id, // 🌟 SSD 저장 및 병합 캐시 활성화!
            Some("inference".to_string())
        ).await.map_err(|e| anyhow!("Qwen 3.5 Inference failed: {}", e))
    }

    pub fn is_cpu(&self) -> bool {
        self.is_cpu_mode
    }

    pub async fn chat(&self, system: &str, user_input: &str, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>, kv_name: Option<String>) -> anyhow::Result<String> {
        // [FIX] Default to Qwen (0.6B) for all chat tasks
        {
            let gen_guard = self.generator.lock().await;
            if gen_guard.is_none() {
                drop(gen_guard);
                self.ensure_generator(ModelSize::Qwen).await?; // 🌟 Small -> Qwen
            }
        }
        
        let _self_clone = self.generator.clone();
        let system_text = system.to_string();
        let user_text = user_input.to_string();
        let max_tok = self.max_tokens_limit;
        
        println!("[MODEL-CHAT] Sending Chat Request...");
        
        {
            let mut gen_guard = self.generator.lock().await;
            let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
            
            let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: system_text,
                name: None,
            });

            let content_parts = vec![
                ChatCompletionRequestMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText { text: user_text }
                )
            ];

            let user_message = ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Array(content_parts),
                name: None,
            };

            let params = ChatCompletionParameters {
                messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
                model: "qwen".to_string(),
                max_tokens: Some(max_tok),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            let response = gen.generate(params, cancel_token, session_id, kv_name).await.map_err(|e| anyhow!("Inference failed: {}", e))?;
            println!("[MODEL-CHAT] Raw Response: {}", response);
            Ok(response)
        }
    }

    pub async fn chat_with_spinner(
        &self, 
        system: &str, 
        user_input: &str,
        app_handle: &tauri::AppHandle,
        event_name: &str,
        base_payload: Value,
        max_tokens: usize,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
            content: system.to_string(),
            name: None,
        });

        let content_parts = vec![
            ChatCompletionRequestMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText { text: user_input.to_string() }
            )
        ];

        let user_message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        let params = ChatCompletionParameters {
            messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
            model: "qwen".to_string(),
            max_tokens: Some(max_tokens as u32),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };

        self.chat_params_with_spinner(params, app_handle, event_name, base_payload, cancel_token, session_id, kv_name).await
    }

    pub async fn chat_params_with_spinner(
        &self, 
        params: ChatCompletionParameters,
        app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        // [FIX] Ensure we stay on Qwen (0.6B).
        {
            let gen_guard = self.generator.lock().await;
            if gen_guard.is_none() {
                drop(gen_guard);
                self.ensure_generator(ModelSize::Qwen).await?; // 🌟 Small -> Qwen
            }
        }

        // [FIX] Inject task_id from session_id if it's a task reference
        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        // [FIX] Removed periodic UI emits from low-level model calls.
        // Higher-level scheduler will manage the initial and final UI states.
        // let _ = app_handle.emit(event_name, &base_payload);
        
        // [LOG] Save to task history if task_id exists
        if let Some(task_id) = base_payload.get("task_id").and_then(|v| v.as_str()) {
            crate::scheduler::log_task_progress(app_handle, task_id, &base_payload);
        }

        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        gen.generate(params, cancel_token, session_id, kv_name).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn chat_with_image_spinner(
        &self, 
        prompt: String, 
        image: Option<DynamicImage>,
        _app_handle: &tauri::AppHandle,
        _event_name: &str,
        _base_payload: Value,
        max_tokens: usize,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        // Ensure generator is loaded
        self.ensure_generator(ModelSize::Qwen).await?;

        // [FIX] Removed redundant emit. Only log the progress if needed.
        // let _ = app_handle.emit(event_name, base_payload);

        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        
        let mut content_parts = Vec::new();
        
        if let Some(img) = image {
            let mut buf = Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            let b64 = BASE64_STANDARD.encode(buf.into_inner());
            let url = format!("data:image/png;base64,{}", b64);
            
            content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageURL { url, detail: None }
                }
            ));
        }

        content_parts.push(ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: prompt }
        ));

        let message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        let params = ChatCompletionParameters {
            messages: vec![ChatCompletionRequestMessage::User(message)],
            model: "qwen".to_string(),
            max_tokens: Some(max_tokens as u32),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id, kv_name).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    async fn run_inference_text(&self, prompt: String, image: Option<DynamicImage>, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>, kv_name: Option<String>) -> anyhow::Result<String> {
        // [VISION-DYNAMIC]
        self.ensure_generator(ModelSize::Qwen).await?; // 🌟 무조건 Qwen으로 로드
        
        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        
        let mut content_parts = Vec::new();
        
        if let Some(img) = image {
            let mut buf = Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            let b64 = BASE64_STANDARD.encode(buf.into_inner());
            let url = format!("data:image/png;base64,{}", b64);
            
            content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageURL { url, detail: None }
                }
            ));
        }

        content_parts.push(ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: prompt }
        ));

        let message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        let params = ChatCompletionParameters {
            messages: vec![ChatCompletionRequestMessage::User(message)],
            model: "qwen".to_string(),
            max_tokens: Some(self.max_tokens_limit),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id, kv_name).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn run_inference_with_spinner(
        &self, 
        system: &str,       // 🌟 추가
        user_input: &str,   // 🌟 변경
        image: Option<DynamicImage>, 
        _app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        kv_name: Option<String>
    ) -> anyhow::Result<String> {
        // [VISION-DYNAMIC]
        self.ensure_generator(ModelSize::Qwen).await?;

        // [FIX] Inject task_id from session_id if it's a task reference
        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        // [LOG] Save to task history if task_id exists
        if let Some(task_id) = base_payload.get("task_id").and_then(|v| v.as_str()) {
            crate::scheduler::log_task_progress(_app_handle, task_id, &base_payload);
        }

        let max_tok = self.max_tokens_limit;
        
        let mut gen_guard = self.generator.lock().await;
        let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
        
        let mut content_parts = Vec::new();
        if let Some(img) = image {
            let mut buf = Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            let b64 = BASE64_STANDARD.encode(buf.into_inner());
            let url = format!("data:image/png;base64,{}", b64);
            
            content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageURL { url, detail: None }
                }
            ));
        }

        content_parts.push(ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: user_input.to_string() }
        ));

        let system_message = ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage {
            content: system.to_string(),
            name: None,
        });

        let user_message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        let params = ChatCompletionParameters {
            messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
            model: "qwen".to_string(),
            max_tokens: Some(max_tok),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id, kv_name).await.map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn process_image_full(&self, image_path: String, app_handle: &tauri::AppHandle, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<Value> {
        println!("[PROCESS] General image analysis for: {}", image_path);
        
        let full_img_raw = image::open(&image_path)?;
        let full_img_raw = DynamicImage::ImageRgb8(full_img_raw.to_rgb8());
        
        // Smart Resize for VRAM stability
        let master_img = full_img_raw.resize(1024, u32::MAX, image::imageops::FilterType::Triangle);
        
        let prompt = get_image_extraction_prompt("kr", "korean", "tracking", "");
        
        let response = self.run_inference_with_spinner(
            "You are a highly precise document data extraction assistant.", // 🌟 System 주입
            &prompt,                                                        // 🌟 User 주입
            Some(master_img),
            app_handle, 
            "extraction-progress", 
            json!({ "category": "Processing", "summary": "Analyzing document content..." }),
            cancel_token,
            None,
            None
        ).await?;

        println!("[PROCESS] Raw Response: {}", response);
        let extracted_data = crate::parsing::parse_json_from_llm(&response);
        
        Ok(extracted_data)
    }

    pub async fn get_embedding(&self, text: String) -> anyhow::Result<Vec<f32>> {
        // Ensure embedding model is loaded (and generator is unloaded)
        self.ensure_embedding().await?;

        let embedding_model_arc = self.embedding_model.clone();
        
        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f32>> {
            let guard = embedding_model_arc.blocking_lock();
            if let Some(model) = guard.as_ref() {
                model.embed(&text).map_err(|e| anyhow::anyhow!("Embedding error: {}", e))
            } else {
                // Fallback to zeros if model failed to load
                Ok(vec![0.0; 768])
            }
        }).await?
    }

    // [신규] Commerce 파이프라인: 2-Stage (0.6B para2graph -> 0.8B graph2contexts)
    // [신규] Commerce 파이프라인: 2-Stage (0.8B 단일 모델 연속 처리)
    pub async fn parse_commerce_query(&self, query: String, language: &str, metrics_json: &str) -> anyhow::Result<Value> {
        let current_time = chrono::Utc::now().to_rfc3339();

        // ----------------------------------------------------
        // Stage 1: 세그먼트 분할 (para2graph) - Qwen3 (0.6B) 사용 🌟
        // ----------------------------------------------------
        self.secure_vram_relay(crate::model::ModelSize::Qwen3, None, None, false, None).await?;

        let prompt1 = crate::parsing::para2graph(language);
        let task_question_1 = format!("{}\n\nQuery: {}", prompt1, query);
        
        let mut segments = serde_json::json!({});
        let max_retries = 2; 

        for attempt in 1..=max_retries {
            // 🌟 [CRITICAL FIX 1] 동기(Sync) 함수인 generate를 spawn_blocking으로 격리!
            // 이렇게 해야 LLM이 생각하는 10초 동안 프론트엔드의 #btn-settings 클릭(채팅 로드)이 멈추지 않습니다!
            let gen_arc = self.qwen3_generator.clone();
            let task_q = task_question_1.clone();
            
            let res1 = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
                let mut gen_guard = gen_arc.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    println!("[AI-SEARCH] 0.6B (Qwen3) Step 1 (Attempt {}/{}): Asking para2graph...", attempt, max_retries);
                    let params = crate::openai_types::ChatCompletionParameters {
                        messages: vec![
                            crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { 
                                content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(task_q),
                                name: None,
                            })
                        ],
                        model: "qwen3".to_string(),
                        max_tokens: Some(512),
                        temperature: Some(if attempt == 1 { 1.0 } else { 0.1 }), 
                        top_p: Some(0.9),
                        ..Default::default()
                    };
                    gen.generate(params).map_err(|e| anyhow::anyhow!("Qwen3 Inference failed: {}", e))
                } else {
                    Err(anyhow::anyhow!("Qwen3 Generator is missing"))
                }
            }).await??;

            segments = crate::parsing::parse_json_from_llm(&res1);

            let plan_str = segments.get("segmented_plan").and_then(|v| v.as_str()).unwrap_or("");
            let ctx_len = segments.get("context").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
            
            let intended_segments = if plan_str.is_empty() { 0 } else { plan_str.matches('|').count() + 1 };

            if attempt < max_retries && ctx_len == 1 && intended_segments > 1 {
                println!("⚠️ [STAGE 1] 할루시네이션 감지: Plan은 {}조각이지만 Context는 1개입니다. Temperature를 낮춰 재시도합니다...", intended_segments);
                continue;
            } else {
                break; 
            }
        }

        println!("\n✅ [STAGE 1: para2graph (0.6B)] 파싱 결과:");
        println!("{}", serde_json::to_string_pretty(&segments).unwrap_or_default());

        if let Some(ctx_arr) = segments.get_mut("context").and_then(|v| v.as_array_mut()) {
             ctx_arr.retain(|seg| {
                 let text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("").trim();
                 let seg_type = seg.get("type").and_then(|v| v.as_str()).unwrap_or("").trim();
                 !text.is_empty() && !seg_type.is_empty()
             });
        }

        // ----------------------------------------------------
        // Stage 1.5: 수치/연산자 추출 - Qwen3 (0.6B) 연속 사용
        // ----------------------------------------------------
        if let Some(ctx_arr) = segments.get_mut("context").and_then(|v| v.as_array_mut()) {
            let total_segments = ctx_arr.len();

            for (idx, seg) in ctx_arr.iter_mut().enumerate() {
                let current_text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let seg_type = seg.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();

                // 🌟 [CRITICAL FIX] 프롬프트 생성 함수에 metrics_json 을 전달!
                let prompt1_5 = crate::parsing::extract_numeric_conditions(&current_text, &query, &seg_type, metrics_json);
                
                let gen_arc = self.qwen3_generator.clone();
                let seg_type_clone = seg_type.clone();
                
                let res1_5 = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
                    let mut gen_guard = gen_arc.blocking_lock();
                    if let Some(gen) = gen_guard.as_mut() {
                        println!("[AI-SEARCH] Qwen3 (0.6B) Step 1.5 [{}/{}]: Extracting conditions for '{}'...", idx + 1, total_segments, seg_type_clone);
                        let params = crate::openai_types::ChatCompletionParameters {
                            messages: vec![
                                crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { 
                                    content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt1_5),
                                    name: None,
                                })
                            ],
                            model: "qwen3".to_string(),
                            max_tokens: Some(256),
                            temperature: Some(0.0), top_p: Some(0.01),
                            ..Default::default()
                        };
                        gen.generate(params).map_err(|e| anyhow::anyhow!("Qwen3 Inference failed: {}", e))
                    } else {
                        Err(anyhow::anyhow!("Qwen3 Generator is missing"))
                    }
                }).await??;

                let mut conditions_json = crate::parsing::parse_json_from_llm(&res1_5);
                
                if let Some(cond_wrapper) = conditions_json.get_mut("condition").and_then(|v| v.as_object_mut()) {
                    for (_, val_obj) in cond_wrapper.iter_mut() {
                        if let Some(is_pct) = val_obj.get("is_percent").and_then(|v| v.as_bool()) {
                            if is_pct {
                                if let Some(v_str) = val_obj.get("value").and_then(|v| v.as_str()) {
                                    let numeric_only: String = v_str.chars().filter(|c| c.is_digit(10) || *c == '.').collect();
                                    if !numeric_only.is_empty() {
                                        val_obj["value"] = json!(numeric_only);
                                    }
                                }
                            }
                        }
                    }
                }

                println!("🔍 [STAGE 1.5: {} 파싱 결과 (필터링 적용)]\n{}", seg_type, serde_json::to_string_pretty(&conditions_json).unwrap_or_default());

                if let Some(obj) = seg.as_object_mut() {
                    if let Some(cond_val) = conditions_json.get("condition") {
                        obj.insert("condition".to_string(), cond_val.clone());
                    } else {
                        obj.insert("condition".to_string(), conditions_json);
                    }
                }
            }
        }

        // ----------------------------------------------------
        // Stage 2: 조건 최종 병합 추출 (graph2contexts) - Qwen 3.5 (0.8B) 사용
        // ----------------------------------------------------
        // 🌟 [CRITICAL FIX] Qwen 3.5 로딩은 Stage 1.5가 무사히 다 끝난 바로 이 시점에 이루어져야 합니다!
        self.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, None, false, None).await?;

        if let Some(ctx_arr) = segments.get_mut("context").and_then(|v| v.as_array_mut()) {
            let total_segments = ctx_arr.len();

            for (idx, seg) in ctx_arr.iter_mut().enumerate() {
                tokio::task::yield_now().await; 

                let current_text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let seg_type = seg.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();

                let prompt2 = crate::parsing::graph2contexts(&current_text, &seg_type);
                
                let res2 = {
                    if let Some(gen) = self.qwen3_5_generator.lock().await.as_mut() {
                        println!("\n[AI-SEARCH] 0.8B (Qwen 3.5) Step 2 [{}/{}]: Extracting attributes for '{}'...", idx + 1, total_segments, seg_type);
                        let params = crate::openai_types::ChatCompletionParameters {
                            messages: vec![
                                crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { 
                                    content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt2),
                                    name: None,
                                })
                            ],
                            model: "qwen3.5".to_string(),
                            max_tokens: Some(256),
                            temperature: Some(0.1), top_p: Some(0.1), 
                            ..Default::default()
                        };
                        gen.generate(params, None, None, None).await?
                    } else {
                        return Err(anyhow::anyhow!("Qwen 3.5 Generator is missing"));
                    }
                };

                let attributes_json = crate::parsing::parse_json_from_llm(&res2);
                println!("🔍 [STAGE 2: {} 속성 파싱 결과]\n{}", seg_type, serde_json::to_string_pretty(&attributes_json).unwrap_or_default());

                if let Some(obj) = seg.as_object_mut() {
                    obj.insert("status".to_string(), attributes_json.get("status").cloned().unwrap_or(serde_json::Value::Null));
                    obj.insert("substantial".to_string(), attributes_json.get("substantial").cloned().unwrap_or(serde_json::Value::Null));
                    obj.insert("find".to_string(), attributes_json.get("find").cloned().unwrap_or(serde_json::Value::Null));
                }
            }
        }

        println!("\n🎯 [STAGE 2: graph2contexts (0.8B)] 최종 추출 결과 (모든 속성 및 조건 완벽 병합 완료):");
        println!("{}", serde_json::to_string_pretty(&segments).unwrap_or_default());
        println!("==================================================\n");

        Ok(segments)
    }

    // [신규] Shipping 파이프라인 (빠른 단일 처리)
    pub async fn parse_shipping_query(&self, query: String, _language: &str) -> anyhow::Result<Value> {
        let ctx = json!([{
            "type": "tracking",
            "text": query.clone(),
            "condition": { "date": null, "quantity": null, "price": null }
        }]);
        Ok(json!({ "context": ctx }))
    }

    // --- Ported from Python (search_engine.py) ---
    // --- Ported from Python (logic.py) ---
    pub async fn run_deep_research(&self, query: String, context_data: String, app_handle: &tauri::AppHandle, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<String> {
        let mut status_history = format!("### 🔍 Deep Research: '{}'\n\n", query);

        // 1. Context Gathering
        status_history.push_str("✅ Context gathered.\n\n");
        // [LOG-ONLY]
        crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

        // 2. Multi-step reasoning loop
        let steps = vec![
            "Analyzing relationships and implications...",
            "Evaluating cross-document consistency...",
            "Synthesizing final intelligence report..."
        ];

        for step in steps.iter() {
            status_history.push_str(&format!("**⏳ {}**\n", step));
            // [LOG-ONLY]
            crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

            let prompt = format!("Given this context: {}\n\nTask: {}\nQuery: {}\n\nProvide deep insight for this specific step.", context_data, step, query);
            
            // In a real implementation, we might want to stream this too, but for now we wait for the step result
            let step_result = self.run_inference_text(prompt, None, cancel_token.clone(), None, None).await?;
            
            let short_res = if step_result.len() > 200 { &step_result[..200] } else { &step_result };
            status_history.push_str(&format!("> {}...\n\n", short_res.replace("\n", " ")));
            // [LOG-ONLY]
            crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));
        }

        // 3. Final Report
        status_history.push_str("### 📊 Final Research Report\n\n");
        let final_prompt = format!("CONTEXT: {}\nQUERY: {}\n\nBased on the above steps, generate a comprehensive final trade intelligence report.", context_data, query);
        
        let report = self.run_inference_text(final_prompt, None, cancel_token, None, None).await?;
        status_history.push_str(&report);
        
        // [LOG-ONLY]
        crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

        Ok(report)
    }

    fn get_search_schema_definitions(&self, _doc_type: &str) -> String {
        r###"{ 
  "header.document_type": { "desc": "Type (Invoice, BL, AWB, PO, BC, AN, DO...)", "type": "String" },
  "header.document_number": { "desc": "ID, Doc No, Reference No", "type": "String" },
  "header.po_number": { "desc": "Purchase Order No (PO)", "type": "String" },
  "header.booking_number": { "desc": "Booking Reference No (BC)", "type": "String" },
  "header.an_number": { "desc": "Arrival Notice No (AN)", "type": "String" },
  "header.do_number": { "desc": "Delivery Order No (DO)", "type": "String" },
  "header.issue_date": { "desc": "Date (YYYY-MM-DD)", "type": "String" },
  
  "parties.supplier_name": { "desc": "Seller, Shipper, Exporter, Vendor", "type": "String" },
  "parties.buyer_name": { "desc": "Buyer, Consignee, Importer", "type": "String" },
  "parties.notify_party_name": { "desc": "Notify Party", "type": "String" },
  
  "financials.amount_total": { "desc": "Total Value/Amount", "type": "Number" },
  "financials.local_charges_total": { "desc": "Total Local Charges (AN)", "type": "Number" },
  
  "logistics.vehicle_name": { "desc": "Vessel Name, Flight No", "type": "String" },
  "logistics.location_port_of_loading": { "desc": "POL, Origin", "type": "String" },
  "logistics.location_port_of_discharge": { "desc": "POD, Destination", "type": "String" },
  "logistics.pickup_location": { "desc": "Pickup Location (DO)", "type": "String" },
  "logistics.etd": { "desc": "Estimated Departure", "type": "String" },
  "logistics.eta": { "desc": "Estimated Arrival", "type": "String" },
  
  "conditions.incoterms_code": { "desc": "Incoterms (FOB, CIF)", "type": "String" }
}"###.to_string()
    }
}

pub fn get_image_extraction_prompt(region: &str, language: &str, page_type: &str, address: &str) -> String {
    if page_type == "tracking" {
        let template = r###"[TASK]
Convert the shipping label image to fit the structured JSON format. 

[CONTEXT]
Region: {REGION}
Recipient Address: {ADDRESS}
Current Language: {LANGUAGE}

[INSTRUCTION]
1. Extract the tracking_number. It should be selected from numbers matching barcodes or QR codes, filtered by region, excluding telephone formats or order numbers.
2. Set recipient_match to true if the label address matches the context address (ignoring floor levels).
3. Extract all visible barcodes into an array.

[OUTPUT FORMAT]
{ "tracking_number": "string", "recipient_match": boolean, "barcodes": ["string"] }"###;
        template.replace("{REGION}", region).replace("{ADDRESS}", address).replace("{LANGUAGE}", language)
    } else {
        String::new()
    }
}

fn merge_json_manual(root: &mut Map<String, Value>, cat: &str, data: Value) {
    let target_key = if cat == "items" { "line_items" } else if cat == "containers" { "containers" } else { cat };
    
    // Some models might wrap the result in the category name or target_key
    let actual_data = if let Some(inner) = data.get(target_key) { inner.clone() } 
                      else if let Some(inner) = data.get(cat) { inner.clone() } 
                      else { data };

    if let Some(target) = root.get_mut(target_key) {
        if target.is_array() {
            let target_arr = target.as_array_mut().unwrap();
            if let Some(source_arr) = actual_data.as_array() {
                for new_item in source_arr {
                    // Check for duplicates in line_items/containers by description/number
                    let is_dup = if target_key == "line_items" {
                        let new_desc = new_item.get("description").and_then(|v| v.as_str()).unwrap_or("");
                        target_arr.iter().any(|ex| ex.get("description").and_then(|v| v.as_str()).unwrap_or("") == new_desc)
                    } else if target_key == "containers" {
                        let new_no = new_item.get("container_number").and_then(|v| v.as_str()).unwrap_or("");
                        target_arr.iter().any(|ex| ex.get("container_number").and_then(|v| v.as_str()).unwrap_or("") == new_no)
                    } else { false };

                    if !is_dup { target_arr.push(new_item.clone()); }
                }
            }
        } else if let Some(target_obj) = target.as_object_mut() {
            if let Some(source_obj) = actual_data.as_object() {
                for (k, v) in source_obj {
                    if !v.is_null() && v != "" && v != 0 { target_obj.insert(k.clone(), v.clone()); }
                }
            }
        }
    }
}
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
    pub device_config: utils::DeviceConfig,
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

    pub async fn ensure_generator(&self, size: ModelSize) -> anyhow::Result<()> {
        self.ensure_generator_ext(size, false, false).await
    }

    pub async fn ensure_generator_ext(&self, size: ModelSize, force_text_only: bool, baking_only: bool) -> anyhow::Result<()> {
        if size == ModelSize::Qwen3_5 {
            return self.ensure_qwen3_5(false).await; 
        }
        if size == ModelSize::Qwen3 {
            return self.ensure_qwen3().await; 
        }

        // 오직 ModelSize::Qwen 만 이 아래 로직을 탐
        let mut current_size_guard = self.current_size.lock().await; // 🌟 첫 번째 자물쇠 획득!
        let mut gen_guard = self.generator.lock().await;

        if *current_size_guard == Some(size) && gen_guard.is_some() && !baking_only {
            return Ok(());
        }

        println!("[LOAD] Fresh loading {:?} from disk...", size);
        let path = &self.qwen_model_path; 
        
        // 🌟 [CRITICAL FIX] 이중 자물쇠(Deadlock) 유발 코드 제거! 
        // 이미 가지고 있는 current_size_guard 에 직접 값을 할당합니다.
        *current_size_guard = Some(size); 
        
        let target_device = self.device_config.device.clone();
        let is_disk_swap = self.is_disk_swap;
        let dev_id = self.device_config.gpu_id;
        let dtype = if target_device.is_cpu() { Some(candle_core::DType::F32) } else { Some(candle_core::DType::BF16) };
        let limit = self.max_tokens_limit;
        let path_clone = path.to_string();
        let handle_clone = self.app_handle.clone();

        let gen = match tokio::time::timeout(
            std::time::Duration::from_secs(60), 
            tokio::task::spawn_blocking(move || {
                let kv_root = crate::utils::paths::get_kv_dir(Some(&handle_clone));
                QwenVLGenerateModel::init_with_config(
                    &path_clone, None, None,
                    Some(&target_device), dev_id, Some(&target_device), dev_id, dtype, Some(limit as usize),
                    force_text_only, baking_only, is_disk_swap, kv_root
                )
            })
        ).await {
            Ok(Ok(Ok(generator))) => generator,
            Ok(Ok(Err(e))) => {
                println!("🚨 [MODEL-ERROR] 모델 초기화 실패 (로직 에러): {:?}", e);
                return Err(e);
            },
            Ok(Err(e)) => {
                println!("🚨 [MODEL-ERROR] Spawn Blocking 실패 (스레드 에러): {:?}", e);
                return Err(e.into());
            },
            Err(_) => {
                println!("🚨 [CRITICAL] 60초 타임아웃 발생! 모델 로딩 내부에서 무한 대기에 빠졌습니다!");
                return Err(anyhow::anyhow!("Model Loading Timeout"));
            }
        };

        *gen_guard = Some(gen);
        // *current_size_guard = Some(size); // 위에서 미리 등록했으므로 생략 가능
        
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
        let mut emb_guard = self.embedding_model.lock().await;
        if emb_guard.is_none() {
            let self_clone = self.embedding_path.clone();
            
            // 🌟 [수정] 핸드오버 단계에서 이미 VRAM이 확보되었거나 모델이 충분히 가벼우므로, 
            // 강제 CPU(RAM) 우회 로직을 제거하고 항상 기본 디바이스(GPU)를 사용하도록 직결합니다.
            let target_device = self.device_config.device.clone();
            
            println!("[MODEL] Loading Embedding Model on {:?}...", if target_device.is_cpu() { "CPU" } else { "GPU" });
            
            let target_device_clone = target_device.clone();
            let emb = tokio::task::spawn_blocking(move || {
                EmbeddingModel::new_with_device(&self_clone, &target_device_clone)
            }).await??;
            
            *emb_guard = Some(emb);
        }
        Ok(())
    }

    // 🌟 [CRITICAL FIX] config.json의 물리적 텐서 크기와 실제 훈련된 Context Length를 완벽히 분리합니다.
    pub async fn truncate_pug_context(&self, pug: &str, is_detail: bool, margin_tokens: usize, bottom_drop_tokens: Option<usize>) -> String {
        let current_size = *self.current_size.lock().await;
        
        let max_context_length: usize = if is_detail { 60_000 } else { 9_000 };
        let tokenizer_path = &self.qwen_model_path;

        // 🌟 한도(최대 토큰)를 계산하고, 버릴 하단 토큰(bottom_drop_tokens)을 파서에 함께 전달합니다.
        let final_max = max_context_length.saturating_sub(margin_tokens);

        // 2. 이미 활성화된 제너레이터가 있다면 그 안에 탑재된 토크나이저를 즉시 재사용합니다.
        if let Some(gen) = self.qwen3_5_generator.lock().await.as_ref() {
            return crate::parsing::truncate_pug_by_tokens(pug, final_max, &gen.tokenizer, bottom_drop_tokens);
        }
        if let Some(gen) = self.qwen3_generator.lock().await.as_ref() {
            return crate::parsing::truncate_pug_by_tokens(pug, final_max, &gen.tokenizer, bottom_drop_tokens);
        }
        if let Some(gen) = self.generator.lock().await.as_ref() {
            return crate::parsing::truncate_pug_by_tokens(pug, final_max, &gen.tokenizer, bottom_drop_tokens);
        }

        // 3. 모델이 VRAM에 없을 경우, 디스크에서 가볍게 토크나이저만 읽어와서 정확한 토큰 수 기반으로 절단합니다.
        if let Ok(tokenizer) = crate::tokenizer::TokenizerModel::init(tokenizer_path) {
            crate::parsing::truncate_pug_by_tokens(pug, final_max, &tokenizer, bottom_drop_tokens)
        } else {
            pug.to_string()
        }
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

        let app_dir = crate::utils::get_app_dir();
        let base_path = app_dir.join("models");
        
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
        search_mode: String,
        app_handle: &tauri::AppHandle,
        cancel_token: Option<Arc<AtomicBool>>,
        store_mutex: &Arc<tokio::sync::Mutex<Option<crate::store::VectorStore>>>,
    ) -> anyhow::Result<()> {
        let app_handle_clone = app_handle.clone();
        let task_id_clone = task_id.clone();
        
        let emit_term = move |msg: &str| {
            println!("{}", msg);
            use tauri::Emitter;
            let _ = app_handle_clone.emit("task-console-log", serde_json::json!({"task_id": task_id_clone, "text": format!("{}\n", msg)}));
        };

        emit_term("\n=======================================");
        emit_term(&format!("[ENGINE] 🚀 Starting Image Extraction Pipeline for Task: {}", task_id));
        emit_term("[STAGE-1] Preparing VRAM and Loading Qwen3.5 (0.8B) Vision Model...");

        // 🌟 [CRITICAL FIX 1] 이미지 추출 5단계를 완벽하게 맞추기 위한 로딩 스텝(2단계) UI 추가!
        let payload_load = json!({ "task_id": task_id.clone(), "category": "Loading Model", "summary": "Initializing Vision Core...", "spinner": "⠋" });
        let _ = app_handle.emit("extraction-progress", &payload_load);
        crate::scheduler::log_task_progress(app_handle, &task_id, &payload_load);

        self.ensure_qwen3_5(true).await?; 

        if let Ok(img) = image::open(&image_path) {
            let dynamic_image = image::DynamicImage::ImageRgb8(img.to_rgb8());
            
            let is_trade_doc = search_mode == "shipping";
            let mut extracted_data = json!({});

            if is_trade_doc {
                emit_term("[STAGE-2] 🚢 Trade Document Mode: Initiating Classification...");
                
                // Step A: 문서 종류 1차 판별 (768px 축소 썸네일 사용)
                let class_img = dynamic_image.resize(768, 768, image::imageops::FilterType::Triangle);
                let class_prompt = crate::parsing::get_trade_doc_classification_prompt(); // (이 프롬프트 안에 TRACKING 추가됨)
                let type_res = self.chat_with_qwen3_5_image_spinner(
                    "You are a document classifier.", &class_prompt, Some(class_img), app_handle, "extraction-progress", 
                    json!({ "category": "Vision (Step 1/2)", "summary": "Identifying document type..." }), 128, cancel_token.clone(), Some(task_id.clone())
                ).await?;
                
                let detected_type = crate::parsing::parse_json_from_llm(&type_res)
                    .get("doc_type").and_then(|v| v.as_str()).unwrap_or("Unknown").to_string();
                emit_term(&format!("✅ Document identified as: **{}**", detected_type));

                // 🌟 [개선된 분기 포인트] TRACKING(운송장)으로 판별되면 무거운 Slice & Merge를 우회합니다!
                if detected_type == "TRACKING" {
                    emit_term("[STAGE-2] 📦 Fast-Tracking Parcel Label...");
                    
                    let prompt = crate::parsing::get_image_extraction_prompt("kr", &language, "tracking", "");
                    let result_str = self.chat_with_qwen3_5_image_spinner(
                        "You are a highly precise logistics data extraction assistant.", &prompt, Some(dynamic_image.clone()), app_handle, "extraction-progress", 
                        json!({ "category": "Vision Analysis", "summary": "Extracting Tracking Label data..." }), 512, cancel_token.clone(), Some(task_id.clone())
                    ).await?;
                    
                    extracted_data = crate::parsing::parse_json_from_llm(&result_str);
                    
                    // DB 저장 시 에러가 나지 않도록 doc_type 꼬리표를 강제로 달아줍니다.
                    if let Some(obj) = extracted_data.as_object_mut() {
                        obj.insert("doc_type".to_string(), json!("TRACKING"));
                    }
                    
                } else {
                    // 🌟 B/L, CI 등 밀도 높은 무역 문서일 경우 기존처럼 Slice & Merge 파이프라인을 탑니다.
                    emit_term("[STAGE-2] 🚢 Initiating Slice & Merge Pipeline...");
                    
                    // Step B: 판별된 문서에 따른 자르기(Slice) 미션 설정
                    let missions = match detected_type.as_str() {
                        "CI" | "PI" => vec![("header", 0.0, 0.20), ("parties", 0.0, 0.40), ("logistics", 0.20, 0.50), ("items", 0.30, 0.85), ("financials", 0.70, 0.95), ("conditions", 0.80, 1.0)],
                        "BL" => vec![("header", 0.0, 0.20), ("parties", 0.0, 0.60), ("logistics", 0.35, 0.65), ("cargo", 0.50, 0.90), ("conditions", 0.80, 1.0)],
                        "AWB" => vec![("header", 0.0, 0.15), ("parties", 0.0, 0.40), ("logistics", 0.10, 0.40), ("cargo", 0.30, 0.70), ("financials", 0.60, 0.90)],
                        _ => vec![("header", 0.0, 0.30), ("parties", 0.0, 0.50), ("items", 0.30, 0.80), ("conditions", 0.70, 1.0)],
                    };

                    let w = dynamic_image.width();
                    let h = dynamic_image.height();
                    let mut final_data_map = serde_json::Map::new();
                    
                    // 🌟 [CRITICAL FIX] Python 패리티: 병합을 위한 7대 기본 뼈대(Skeleton)를 무조건 미리 생성해야 합니다!
                    final_data_map.insert("header".to_string(), json!({"doc_type": detected_type}));
                    final_data_map.insert("parties".to_string(), json!({}));
                    final_data_map.insert("logistics".to_string(), json!({}));
                    final_data_map.insert("conditions".to_string(), json!({}));
                    final_data_map.insert("financials".to_string(), json!({}));
                    final_data_map.insert("cargo".to_string(), json!({}));
                    final_data_map.insert("line_items".to_string(), json!([]));
                    final_data_map.insert("containers".to_string(), json!([]));

                    // Step C: 구역별 분할 크롭 및 LLM 타격
                    for (idx, (cat, top, bot)) in missions.iter().enumerate() {
                        if cancel_token.as_ref().map_or(false, |t| t.load(std::sync::atomic::Ordering::Relaxed)) { return Err(anyhow!("Cancelled")); }
                        
                        let crop_y = (h as f32 * top) as u32;
                        let crop_h = (h as f32 * (bot - top)) as u32;
                        let img_slice = dynamic_image.crop_imm(0, crop_y, w, crop_h);
                        
                        let prompt = crate::parsing::get_trade_category_schema(cat, &detected_type);
                        let summary_msg = format!("Scanning {} ({}%)...", cat.to_uppercase(), (bot * 100.0) as i32);
                        
                        let tile_res = self.chat_with_qwen3_5_image_spinner(
                            "You are a highly precise document data extraction assistant.", &prompt, Some(img_slice), app_handle, "extraction-progress", 
                            json!({ "category": format!("Vision (Slice {}/{})", idx+1, missions.len()), "summary": summary_msg }), 1024, cancel_token.clone(), Some(task_id.clone())
                        ).await?;

                        let tile_json = crate::parsing::parse_json_from_llm(&tile_res);
                        
                        // 🌟 기존 병합 함수 호출 (이제 뼈대가 있으므로 정상적으로 채워집니다)
                        merge_json_manual(&mut final_data_map, cat, tile_json);
                    }
                    
                    extracted_data = Value::Object(final_data_map);
                }

            } else {
                // ============================================================
                // 🛒 [Commerce 모드] 커머스 라우팅 보완
                // ============================================================
                emit_term("[STAGE-2] 🛒 Commerce Mode: Analyzing Product/Label...");
                
                // 🌟 [개선] 기존에 무조건 "goods"(상품) 프롬프트를 먹이던 것을, 
                // 택배 운송장이 올라올 확률이 높으므로 바코드/송장 번호를 우선 추출하는 "tracking" 기반의 
                // 범용 커머스 프롬프트로 처리하도록 변경했습니다.
                let prompt = crate::parsing::get_image_extraction_prompt("kr", &language, "tracking", "");
                
                let result_str = self.chat_with_qwen3_5_image_spinner(
                    "You are a precise commerce and logistics extraction assistant.", &prompt, Some(dynamic_image.clone()), app_handle, "extraction-progress", 
                    json!({ "category": "Vision Analysis", "summary": "Analyzing commerce tracking/goods..." }), 1024, cancel_token.clone(), Some(task_id.clone())
                ).await?;
                
                extracted_data = crate::parsing::parse_json_from_llm(&result_str);
            }
            
            let mode_name = if is_trade_doc { "Trade Document" } else { "Commerce" };
            emit_term(&format!("[STAGE-2] Generating vision insights for {} mode...", mode_name));

            emit_term("\n=======================================");
            emit_term(&format!("[DEBUG-VISION] 🤖 AI Raw Response Extracted."));
            emit_term("=======================================\n");

            let nl = crate::parsing::json_to_natural_language(&extracted_data);
            
            // [PRIVACY] 무역 문서(BL, CI 등) 및 송장(Tracking)은 개인정보 밀집 구역이므로 반드시 마스킹을 적용합니다.
            // 커머스 상품(goods) 이미지인 경우에만 예외적으로 우회합니다.
            let doc_type = if is_trade_doc { 
                extracted_data.get("header")
                    .and_then(|h| h.get("doc_type"))
                    .and_then(|s| s.as_str())
                    .unwrap_or("shipping_doc") 
            } else { 
                "tracking" 
            };
            
            let masked_nl = nl.clone(); // 마스킹은 백엔드 push_data 단계에서 동적으로 수행됩니다.

            let item_digest = crate::utils::hash::digest(&nl);

            emit_term("[STAGE-3] Syncing extracted data to LanceDB...");

            // 🌟 [CRITICAL FIX 2] 5단계 마무리를 위한 저장 스텝(4단계) UI 추가!
            let payload_save = json!({ "task_id": task_id.clone(), "category": "Saving", "summary": "Syncing to database...", "spinner": "⠋" });
            let _ = app_handle.emit("extraction-progress", &payload_save);
            crate::scheduler::log_task_progress(app_handle, &task_id, &payload_save);

            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                let from_addr = "0x0000000000000000000000000000000000000000";
                let team_id = crate::utils::hash::hash_id(from_addr); 
                let hashed_cc = crate::utils::hash::hash_id(if is_trade_doc { "local.shipping" } else { "local.commerce" });

                // 식별자(ID) 추출 기준 분기
                let raw_no = if is_trade_doc {
                    extracted_data.get("document_number").and_then(|s| s.as_str()).unwrap_or(&task_id)
                } else {
                    extracted_data.get("tracking_number").and_then(|s| s.as_str()).unwrap_or(&task_id)
                };
                
                let clean_no = crate::utils::hash::normalize_numeric_homoglyphs(raw_no).replace("-", "").replace("_", "");
                
                // 🌟 [CRITICAL FIX] 프론트엔드 리스트(#doc-list)와 완벽 동기화하기 위해 "items" 테이블로 저장 위치를 강제 통합합니다!
                let table_name = "items"; 

                let index_val = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("{}{}", doc_type, clean_no)));
                let hashed_id = crate::utils::hash::hash_id(&format!("{}{}", team_id, index_val));
                let ref_val = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, hashed_cc, clean_no));

                let mut final_data = if extracted_data.is_object() { extracted_data.clone() } else { json!({ "raw_output": extracted_data }) };
                final_data.as_object_mut().unwrap().insert("index".to_string(), json!(index_val));
                final_data.as_object_mut().unwrap().insert("id".to_string(), json!(hashed_id));
                // 🌟 [CRITICAL FIX] 이미지 추출 결과에도 모드 필터를 위한 mode 값을 명시적으로 주입합니다.
                final_data.as_object_mut().unwrap().insert("mode".to_string(), json!(search_mode.clone()));
                final_data.as_object_mut().unwrap().insert("text".to_string(), json!(nl));
                final_data.as_object_mut().unwrap().insert("masked_text".to_string(), json!(masked_nl));

                // 🌟 [추가 보완] 무역 문서(Trade Doc)일 경우 Python처럼 핵심 컬럼 평탄화 (Flattening)
                if is_trade_doc {
                    let obj = final_data.as_object_mut().unwrap();
                    
                    // Header에서 날짜/문서번호 추출
                    if let Some(header) = extracted_data.get("header") {
                        obj.insert("issue_date".to_string(), header.get("issue_date").cloned().unwrap_or(json!("")));
                        obj.insert("no".to_string(), header.get("document_number").cloned().unwrap_or(json!("")));
                    }
                    // Parties에서 화주/수하인 추출
                    if let Some(parties) = extracted_data.get("parties") {
                        obj.insert("sender_name".to_string(), parties.get("supplier_name").cloned().unwrap_or(json!("")));
                        obj.insert("recipient_name".to_string(), parties.get("buyer_name").cloned().unwrap_or(json!("")));
                    }
                    // Logistics에서 선박/항구 추출
                    if let Some(logistics) = extracted_data.get("logistics") {
                        obj.insert("vessel".to_string(), logistics.get("vehicle_name").cloned().unwrap_or(json!("")));
                        obj.insert("pol".to_string(), logistics.get("location_port_of_loading").cloned().unwrap_or(json!("")));
                        obj.insert("pod".to_string(), logistics.get("location_port_of_discharge").cloned().unwrap_or(json!("")));
                    }
                    // Financials/Conditions 추출
                    if let Some(fin) = extracted_data.get("financials") {
                        obj.insert("amount".to_string(), fin.get("amount_total").cloned().unwrap_or(json!(0)));
                    }
                    if let Some(cond) = extracted_data.get("conditions") {
                        obj.insert("incoterms".to_string(), cond.get("incoterms_code").cloned().unwrap_or(json!("")));
                    }
                }
                
                let _ = db.upsert_item(
                    table_name, // 분기된 테이블 적용
                    &hashed_id, 
                    doc_type, 
                    final_data, 
                    None,
                    Some(from_addr),
                    Some(&team_id),
                    Some(&hashed_cc),
                    Some(&crate::utils::hash::hash_id(&format!("{}{}", doc_type, hashed_cc))),
                    Some(&ref_val),
                    Some(&item_digest)
                ).await;
                
                // 🌟 [CRITICAL FIX] 이미지 데이터 저장 직후, DB의 Task와 Message 상태도 9(DONE)로 완전히 굳혀버립니다!
                // 이 두 줄이 없어서 3초마다 UI가 이전 상태(1)를 DB에서 퍼와 덮어씌우고 있었습니다.
                let _ = db.update_task_status(&task_id, 9).await;
                let _ = db.update_message_status(&task_id, 9, Some("Extraction Complete")).await;
            }
            
            emit_term("[SUCCESS] Task Completed. Data saved.");
            
            let payload = json!({ 
               "task_id": task_id.clone(),
               "category": "Done", "summary": "Analysis Complete", "spinner": "✅", "data": extracted_data
            });
            
            // 🌟 [CRITICAL FIX] Done 상태를 파일에도 확실히 기록하여 상세페이지 복구 시 100% 출력되게 합니다!
            crate::scheduler::log_task_progress(app_handle, &task_id, &payload);
            
            crate::scheduler::notify_new_task();
            
            Ok(())
        } else {
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
            crate::scheduler::log_task_progress(_app_handle, task_id, &base_payload); // 기존 변수명이 app_handle이면 app_handle로 사용
        }
        
        // 🌟 [CRITICAL FIX] 화면에 실시간 진행률(퍼센트)을 쏘아 보내는 코드를 복구합니다!
        let _ = _app_handle.emit(_event_name, &base_payload); // 기존 변수명이 app_handle이면 app_handle, _event_name이면 _event_name 사용
        
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
    pub async fn parse_commerce_query(&self, task_id: &str, app_handle: &tauri::AppHandle, query: String, language: &str, metrics_json: &str, cancel_token: Arc<AtomicBool>) -> anyhow::Result<Value> {
        use tauri::Emitter;

        // 🌟 [신규] 터미널 로거 헬퍼 주입
        let emit_term = |msg: &str| {
            println!("{}", msg);
            let _ = app_handle.emit("task-console-log", json!({"task_id": task_id, "text": format!("{}\n", msg)}));
        };

        emit_term("[ENGINE] 🚀 Starting Commerce Search Pipeline...");
        
        // ----------------------------------------------------
        // Stage 1: 세그먼트 분할 (para2graph) - Qwen3 (0.6B) 사용
        // ----------------------------------------------------
        emit_term("[STAGE-1] Preparing VRAM and Loading Qwen3 (0.6B) Model...");
        let payload = json!({ "task_id": task_id, "category": "Stage 1", "summary": "Analyzing semantic intent...", "spinner": "⠋" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::scheduler::log_task_progress(app_handle, task_id, &payload);

        self.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancel_token.clone()), false, None).await?;
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        let prompt1 = crate::parsing::para2graph(language);
        let task_question_1 = format!("{}\n\nQuery: {}", prompt1, query);
        
        let mut segments = serde_json::json!({});
        let max_retries = 2; 

        for attempt in 1..=max_retries {
            if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            
            emit_term(&format!("[STAGE-1] Generating intent segment... (Attempt {}/{})", attempt, max_retries));
            
            let gen_arc = self.qwen3_generator.clone();
            let task_q = task_question_1.clone();
            let cancel_clone = cancel_token.clone();
            
            let res1 = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
                let mut gen_guard = gen_arc.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    let params = crate::openai_types::ChatCompletionParameters {
                        messages: vec![
                            crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { 
                                content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(task_q),
                                name: None,
                            })
                        ],
                        model: "qwen3".to_string(),
                        max_tokens: Some(256),
                        temperature: Some(1.0), 
                        // 🌟 [CRITICAL FIX] top_k(80) 효과를 온전히 보려면 top_p를 1.0으로 개방해야 합니다.
                        // (0.95일 경우 상위 5~10개 토큰 선에서 먼저 잘려버려 top_k 80이 무시되는 것처럼 보입니다)
                        top_p: Some(1.0), 
                        ..Default::default()
                    };
                    gen.generate(params, Some(cancel_clone)).map_err(|e| anyhow::anyhow!("Qwen3 Inference failed: {}", e))
                } else {
                    Err(anyhow::anyhow!("Qwen3 Generator is missing"))
                }
            }).await??;

            emit_term(&format!("[STAGE-1 RESULT]\n{}", res1)); // 🌟 AI가 뱉어낸 JSON 응답을 UI 터미널에 그대로 꽂아버립니다!
            
            segments = crate::parsing::parse_json_from_llm(&res1);
            let plan_str = segments.get("segmented_plan").and_then(|v| v.as_str()).unwrap_or("");
            let ctx_len = segments.get("context").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
            let intended_segments = if plan_str.is_empty() { 0 } else { plan_str.matches('|').count() + 1 };

            if attempt < max_retries && ctx_len == 1 && intended_segments > 1 { continue; } else { break; }
        }

        if let Some(ctx_arr) = segments.get_mut("context").and_then(|v| v.as_array_mut()) {
            // 🌟 [CRITICAL FIX] retain은 & 참조자만 전달하므로, 수정을 위한 루프를 별도로 분리합니다. (E0596 해결)
            for seg in ctx_arr.iter_mut() {
                let seg_type = seg.get("type").and_then(|v| v.as_str()).unwrap_or("").trim().to_lowercase();
                if let Some(obj) = seg.as_object_mut() {
                    obj.insert("type".to_string(), serde_json::json!(seg_type));
                }
            }
            
            // 그 다음 삭제 여부 판별 진행
            ctx_arr.retain(|seg| {
                let text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("").trim();
                let seg_type = seg.get("type").and_then(|v| v.as_str()).unwrap_or(""); // 이미 위에서 정리됨
                
                !text.is_empty() && !seg_type.is_empty()
            });
        }

        // 🌟 [CRITICAL FIX] 변수명 오타 수정: cancellation_token -> cancel_token (E0425 해결)
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        // ----------------------------------------------------
        // Stage 1.5: 수치/연산자 추출 - Qwen3 (0.6B) 연속 사용
        // ----------------------------------------------------
        if let Some(ctx_arr) = segments.get_mut("context").and_then(|v| v.as_array_mut()) {
            let total_segments = ctx_arr.len();

            for (idx, seg) in ctx_arr.iter_mut().enumerate() {
                // 🌟 루프 도중에도 취소 버튼을 누르면 즉시 멈춥니다!
                if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                let payload = json!({ "task_id": task_id, "category": format!("Stage 1.5 ({}/{})", idx+1, total_segments), "summary": "Extracting filter conditions...", "spinner": "⠋" });
                let _ = app_handle.emit("extraction-progress", &payload);
                crate::scheduler::log_task_progress(app_handle, task_id, &payload);

                let current_text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let seg_type = seg.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let prompt1_5 = crate::parsing::extract_numeric_conditions(&current_text, &query, &seg_type, metrics_json);
                
                let gen_arc = self.qwen3_generator.clone();
                let cancel_clone = cancel_token.clone();
                
                let res1_5 = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
                    let mut gen_guard = gen_arc.blocking_lock();
                    if let Some(gen) = gen_guard.as_mut() {
                        let params = crate::openai_types::ChatCompletionParameters {
                            messages: vec![
                                crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { 
                                    content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt1_5),
                                    name: None,
                                })
                            ],
                            model: "qwen3".to_string(), max_tokens: Some(256), temperature: Some(0.0), top_p: Some(0.01),
                            ..Default::default()
                        };
                        gen.generate(params, Some(cancel_clone)).map_err(|e| anyhow::anyhow!("Qwen3 Inference failed: {}", e))
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
                                    if !numeric_only.is_empty() { val_obj["value"] = json!(numeric_only); }
                                }
                            }
                        }
                    }
                }
                if let Some(obj) = seg.as_object_mut() {
                    if let Some(cond_val) = conditions_json.get("condition") { obj.insert("condition".to_string(), cond_val.clone()); } 
                    else { obj.insert("condition".to_string(), conditions_json); }
                }

                crate::models::qwen::generate::wait_for_global_io().await;
                
                // 🌟 [신규 추가] GPU 비동기 연산 찌꺼기 강제 동기화
                if !self.is_cpu_mode {
                    let dev = self.device_config.device.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        if dev.is_cuda() { let _ = dev.synchronize(); }
                    }).await;
                }

                #[cfg(target_os = "windows")]
                unsafe {
                    use windows_sys::Win32::System::Threading::GetCurrentProcess;
                    use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
                    let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
                }
                #[cfg(target_os = "linux")]
                unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
                #[cfg(target_os = "macos")]
                unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }
            }
        }

        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        // ----------------------------------------------------
        // Stage 2: 조건 최종 병합 추출 (graph2contexts) - Qwen 3.5 (0.8B) 사용
        // ----------------------------------------------------
        let payload = json!({ "task_id": task_id, "category": "Stage 2", "summary": "Switching to 0.8B model...", "spinner": "⠋" });
        let _ = app_handle.emit("extraction-progress", &payload);

        self.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancel_token.clone()), false, None).await?;
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        if let Some(ctx_arr) = segments.get_mut("context").and_then(|v| v.as_array_mut()) {
            let total_segments = ctx_arr.len();

            for (idx, seg) in ctx_arr.iter_mut().enumerate() {
                if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                let payload = json!({ "task_id": task_id, "category": format!("Stage 2 ({}/{})", idx+1, total_segments), "summary": "Extracting final attributes...", "spinner": "⠋" });
                let _ = app_handle.emit("extraction-progress", &payload);
                crate::scheduler::log_task_progress(app_handle, task_id, &payload);

                tokio::task::yield_now().await; 

                let current_text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let seg_type = seg.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let prompt2 = crate::parsing::graph2contexts(&current_text, &seg_type);
                
                let res2 = {
                    if let Some(gen) = self.qwen3_5_generator.lock().await.as_mut() {
                        let params = crate::openai_types::ChatCompletionParameters {
                            messages: vec![
                                crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { 
                                    content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt2),
                                    name: None,
                                })
                            ],
                            model: "qwen3.5".to_string(), max_tokens: Some(256), temperature: Some(0.1), top_p: Some(0.1), 
                            ..Default::default()
                        };
                        gen.generate(params, Some(cancel_token.clone()), None, None).await?
                    } else {
                        return Err(anyhow::anyhow!("Qwen 3.5 Generator is missing"));
                    }
                };

                let attributes_json = crate::parsing::parse_json_from_llm(&res2);
                if let Some(obj) = seg.as_object_mut() {
                    
                    // 🌟 [CRITICAL FIX] 변경된 중첩 JSON 구조에 맞게 seg_type 키 내부의 객체에서 속성들을 꺼내옵니다.
                    // 만약 LLM이 래퍼({TYPE})를 생략하고 바로 status, substantial을 뱉었을 경우를 대비해 Fallback(attributes_json 자체)도 적용합니다.
                    let inner_obj = attributes_json.get(&seg_type).unwrap_or(&attributes_json);

                    let status_val = inner_obj.get("status").cloned().unwrap_or(serde_json::Value::Null);
                    let substantial_val = inner_obj.get("substantial").cloned().unwrap_or(serde_json::Value::Null);
                    let find_val = inner_obj.get("find").cloned().unwrap_or(serde_json::Value::Null);

                    obj.insert("status".to_string(), status_val);
                    obj.insert("substantial".to_string(), substantial_val);
                    obj.insert("find".to_string(), find_val);
                }

                crate::models::qwen::generate::wait_for_global_io().await;
                
                // 🌟 [신규 추가] GPU 비동기 연산 찌꺼기 강제 동기화
                if !self.is_cpu_mode {
                    let dev = self.device_config.device.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        if dev.is_cuda() { let _ = dev.synchronize(); }
                    }).await;
                }

                #[cfg(target_os = "windows")]
                unsafe {
                    use windows_sys::Win32::System::Threading::GetCurrentProcess;
                    use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
                    let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
                }
                #[cfg(target_os = "linux")]
                unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
                #[cfg(target_os = "macos")]
                unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }
            }
        }

        let payload = json!({ "task_id": task_id, "category": "Done", "summary": "Analysis complete.", "spinner": "✅" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::scheduler::log_task_progress(app_handle, task_id, &payload);

        Ok(segments)
    }

    // [신규] Shipping 파이프라인 (빠른 단일 처리)
    pub async fn parse_shipping_query(&self, task_id: &str, app_handle: &tauri::AppHandle, query: String, language: &str, cancel_token: Arc<AtomicBool>) -> anyhow::Result<Value> {
        // 🌟 [CRITICAL FIX] 매크로 제거 후 비동기 우회 함수 장착!
        let app_handle_clone = app_handle.clone();
        let task_id_clone = task_id.to_string();
        let emit_term = move |msg: &str| {
            println!("{}", msg);
            let m = msg.to_string();
            let handle = app_handle_clone.clone();
            let tid = task_id_clone.clone();
            tokio::spawn(async move {
                use tauri::Emitter;
                let _ = handle.emit("task-console-log", serde_json::json!({"task_id": tid, "text": format!("{}\n", m)}));
            });
        };

        emit_term("\n=======================================");
        emit_term("[ENGINE] 🚀 Starting Shipping Search Pipeline...");

        let payload = json!({ "task_id": task_id, "category": "Shipping", "summary": "Extracting logistics filters...", "spinner": "⠋" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::scheduler::log_task_progress(app_handle, task_id, &payload);

        emit_term("[STAGE-1] Preparing VRAM and Loading Qwen3 (0.6B) Model...");
        self.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancel_token.clone()), false, None).await?;
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        emit_term(&format!("[STAGE-1] Extracting shipping filters from query: '{}'", query));
        let prompt = crate::parsing::extract_shipping_conditions(&query, language);
        let gen_arc = self.qwen3_generator.clone();
        let cancel_clone = cancel_token.clone();
        
        let res = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = gen_arc.blocking_lock();
            if let Some(gen) = gen_guard.as_mut() {
                let params = crate::openai_types::ChatCompletionParameters {
                    messages: vec![
                        crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { 
                            content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt),
                            name: None,
                        })
                    ],
                    model: "qwen3".to_string(), max_tokens: Some(256), temperature: Some(0.0), top_p: Some(0.01),
                    ..Default::default()
                };
                gen.generate(params, Some(cancel_clone)).map_err(|e| anyhow::anyhow!("Qwen3 Inference failed: {}", e))
            } else {
                Err(anyhow::anyhow!("Qwen3 Generator is missing"))
            }
        }).await??;

        // 🌟 추출된 결과를 터미널 화면에 꽂아줍니다!
        emit_term(&format!("[STAGE-1 RESULT]\n{}", res));

        let extracted_conditions = crate::parsing::parse_json_from_llm(&res);
        
        let payload = json!({ "task_id": task_id, "category": "Done", "summary": "Filter extraction complete.", "spinner": "✅" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::scheduler::log_task_progress(app_handle, task_id, &payload);

        let ctx = json!([{
            "type": "tracking",
            "text": query.clone(),
            "condition": extracted_conditions
        }]);

        emit_term("[SUCCESS] Shipping Search Pipeline Completed.");
        Ok(json!({ "context": ctx }))
    }

    // [신규] Analytic 파이프라인 (임시 Dummy 함수)
    pub async fn parse_analytic_query(&self, task_id: &str, app_handle: &tauri::AppHandle, query: String, language: &str, cancel_token: Arc<AtomicBool>) -> anyhow::Result<Value> {
        let app_handle_clone = app_handle.clone();
        let task_id_clone = task_id.to_string();
        let emit_term = move |msg: &str| {
            println!("{}", msg);
            let m = msg.to_string();
            let handle = app_handle_clone.clone();
            let tid = task_id_clone.clone();
            tokio::spawn(async move {
                use tauri::Emitter;
                let _ = handle.emit("task-console-log", serde_json::json!({"task_id": tid, "text": format!("{}\n", m)}));
            });
        };

        emit_term("\n=======================================");
        emit_term("[ENGINE] 🚀 Starting Analytic Search Pipeline (Draft Mode)...");

        // UI에 스피너 표기
        let payload = json!({ "task_id": task_id, "category": "Analytic", "summary": "Running mock analytics...", "spinner": "⠋" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::scheduler::log_task_progress(app_handle, task_id, &payload);

        // 🌟 취소 버튼 즉시 반응 대응
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        // [TODO] 향후 여기에 통계 분석 전용 프롬프트 및 LLM 추론 로직 (Graph2Metrics 등) 추가 예정
        tokio::time::sleep(std::time::Duration::from_millis(500)).await; // 임시 대기

        emit_term(&format!("[STAGE-1] Dummy parsing analytic intent from query: '{}'", query));
        
        // 검색 쿼리에 걸리도록 임시 컨텍스트(sales 등)를 뱉어냅니다.
        let ctx = json!([{
            "type": "sales", // 검색을 위한 기본 타겟 테이블 (임시)
            "text": query.clone(),
            "condition": {}
        }]);

        let payload_done = json!({ "task_id": task_id, "category": "Done", "summary": "Analytic processing complete (Dummy).", "spinner": "✅" });
        let _ = app_handle.emit("extraction-progress", &payload_done);
        crate::scheduler::log_task_progress(app_handle, task_id, &payload_done);

        emit_term("[SUCCESS] Analytic Search Pipeline Completed.");
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
            crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

            let prompt = format!("Given this context: {}\n\nTask: {}\nQuery: {}\n\nProvide deep insight for this specific step.", context_data, step, query);
            
            let step_result = self.run_inference_text(prompt, None, cancel_token.clone(), None, None).await?;
            
            let short_res = if step_result.len() > 200 { &step_result[..200] } else { &step_result };
            status_history.push_str(&format!("> {}...\n\n", short_res.replace("\n", " ")));
            crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

            crate::models::qwen::generate::wait_for_global_io().await;
            
            // 🌟 [신규 추가] GPU 비동기 연산 찌꺼기 강제 동기화
            if !self.is_cpu_mode {
                let dev = self.device_config.device.clone();
                let _ = tokio::task::spawn_blocking(move || {
                    if dev.is_cuda() { let _ = dev.synchronize(); }
                }).await;
            }
            
            #[cfg(target_os = "windows")]
            unsafe {
                use windows_sys::Win32::System::Threading::GetCurrentProcess;
                use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
                let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
            }
            #[cfg(target_os = "linux")]
            unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
            #[cfg(target_os = "macos")]
            unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }
            
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
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

//     fn get_search_schema_definitions(&self, _doc_type: &str) -> String {
//         r###"{ 
//   "header.document_type": { "desc": "Type (Invoice, BL, AWB, PO, BC, AN, DO...)", "type": "String" },
//   "header.document_number": { "desc": "ID, Doc No, Reference No", "type": "String" },
//   "header.po_number": { "desc": "Purchase Order No (PO)", "type": "String" },
//   "header.booking_number": { "desc": "Booking Reference No (BC)", "type": "String" },
//   "header.an_number": { "desc": "Arrival Notice No (AN)", "type": "String" },
//   "header.do_number": { "desc": "Delivery Order No (DO)", "type": "String" },
//   "header.issue_date": { "desc": "Date (YYYY-MM-DD)", "type": "String" },
  
//   "parties.supplier_name": { "desc": "Seller, Shipper, Exporter, Vendor", "type": "String" },
//   "parties.buyer_name": { "desc": "Buyer, Consignee, Importer", "type": "String" },
//   "parties.notify_party_name": { "desc": "Notify Party", "type": "String" },
  
//   "financials.amount_total": { "desc": "Total Value/Amount", "type": "Number" },
//   "financials.local_charges_total": { "desc": "Total Local Charges (AN)", "type": "Number" },
  
//   "logistics.vehicle_name": { "desc": "Vessel Name, Flight No", "type": "String" },
//   "logistics.location_port_of_loading": { "desc": "POL, Origin", "type": "String" },
//   "logistics.location_port_of_discharge": { "desc": "POD, Destination", "type": "String" },
//   "logistics.pickup_location": { "desc": "Pickup Location (DO)", "type": "String" },
//   "logistics.etd": { "desc": "Estimated Departure", "type": "String" },
//   "logistics.eta": { "desc": "Estimated Arrival", "type": "String" },
  
//   "conditions.incoterms_code": { "desc": "Incoterms (FOB, CIF)", "type": "String" }
// }"###.to_string()
//     }
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
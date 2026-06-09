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
    Qwen3_5, // 2B Qwen 3.5 (Text Optimized)
}

#[derive(Clone)]
pub struct LogisModel {
    pub app_handle: tauri::AppHandle,
    pub generator: Arc<TokioMutex<Option<QwenVLGenerateModel>>>, 
    pub qwen3_generator: Arc<TokioMutex<Option<Qwen3GenerateModel>>>, 
    pub qwen3_5_generator: Arc<TokioMutex<Option<Qwen3_5GenerateModel>>>,
    
    pub embedding_model: Arc<TokioMutex<Option<EmbeddingModel>>>,
    pub embedding_cache: Arc<TokioMutex<std::collections::HashMap<String, Vec<f32>>>>,

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
        
        println!("[DIAG-PURGE] Step 2: Clearing Embedding Model & Cache...");
        {
            let mut emb = self.embedding_model.lock().await;
            if let Some(e) = emb.take() { 
                drop(e); 
            }
            // 🌟 램 누수 방지를 위해 캐시도 깔끔하게 비워줍니다.
            let mut cache = self.embedding_cache.lock().await;
            cache.clear();
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

            // 🌟 [핵심 픽스] 현재 모델이 Qwen 3.5(2B)라면 2B 방(q35_arc)에 스냅샷을 로드합니다!
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

        // 🌟 [추가] 현재 로드된 모델이 목표와 같다면 로딩 과정을 건너뛰고 즉시 반환하여 VRAM 낭비 및 지연 방지
        {
            let current = self.current_size.lock().await;
            if *current == Some(target_size) {
                let is_loaded = match target_size {
                    ModelSize::Qwen => {
                        if let Some(gen) = self.generator.lock().await.as_ref() {
                            let is_baking_loaded = match &gen.qwen {
                                crate::models::qwen::generate::ModelVariant::QuantizedVL(m) => m.language_model.baking_only,
                                crate::models::qwen::generate::ModelVariant::QuantizedText(m) => m.language_model.baking_only,
                                _ => false,
                            };
                            // 🌟 [CRITICAL FIX] 베이킹 모드(LM Head 제거)로 떠있는데, 정상 추론이 필요하면 건너뛰지 않고 리로드!
                            if !is_baking && is_baking_loaded {
                                false
                            } else {
                                true
                            }
                        } else {
                            false
                        }
                    },
                    ModelSize::Qwen3 => self.qwen3_generator.lock().await.is_some(),
                    ModelSize::Qwen3_5 => self.qwen3_5_generator.lock().await.is_some(),
                };
                if is_loaded {
                    println!("[RELAY] {:?} is already loaded. Skipping purge/reload.", target_size);
                    return Ok(());
                } else {
                    // 🌟 [CRITICAL FIX] Background에서 로딩 중이라 객체(is_some)는 없지만, current_size는 target_size로 등록된 상태!
                    // 여기서 무식하게 deep_purge를 호출하면 백그라운드 로딩이 파괴되므로, 각 ensure 함수의 대기(Wait) 루프로 안전하게 진입시킵니다.
                    drop(current); // 락 해제
                    println!("[RELAY] {:?} is currently loading in background. Waiting for synchronization...", target_size);
                    match target_size {
                        ModelSize::Qwen => { self.ensure_generator_ext(ModelSize::Qwen, false, is_baking).await?; },
                        ModelSize::Qwen3 => { self.ensure_qwen3().await?; },
                        ModelSize::Qwen3_5 => { self.ensure_qwen3_5(false).await?; }
                    }
                    return Ok(());
                }
            }
        }
        
        println!("[RELAY] Performing Deep Purge before loading {:?} (Baking: {})...", target_size, is_baking);
        self.deep_purge_resources().await;
        
        if !self.is_cpu_mode {
            tokio::time::sleep(Duration::from_millis(500)).await;
            self.wait_for_vram_settle(2000, 5, cancel_token.clone()).await?;
        }

        match target_size {
            ModelSize::Qwen => {
                self.ensure_generator_ext(ModelSize::Qwen, false, is_baking).await?;
                if let Some(tid) = task_id {
                    self.load_kv_snapshot(tid, kv_name).await?;
                }
            },
            ModelSize::Qwen3 => {
                self.ensure_qwen3().await?;
            },
            ModelSize::Qwen3_5 => {
                self.ensure_qwen3_5(false).await?;
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
            // 🌟 [CRITICAL FIX] 백그라운드에서 이미 로딩이 시작되었는지 확인하고, 진행 중이라면 완료될 때까지 안전하게 대기합니다.
            {
                let size_guard = self.current_size.lock().await;
                if *size_guard == Some(ModelSize::Qwen3) {
                    drop(size_guard);
                    println!("[MODEL] Qwen3 is currently loading in background. Waiting for synchronization...");
                    loop {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        if self.qwen3_generator.lock().await.is_some() {
                            return Ok(());
                        }
                    }
                }
            }

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

        if *current_size_guard == Some(size) {
            if let Some(gen) = gen_guard.as_ref() {
                let is_baking_loaded = match &gen.qwen {
                    crate::models::qwen::generate::ModelVariant::QuantizedVL(m) => m.language_model.baking_only,
                    crate::models::qwen::generate::ModelVariant::QuantizedText(m) => m.language_model.baking_only,
                    _ => false,
                };
                // 🌟 [CRITICAL FIX] 현재 모델이 Baking(LM Head 부재) 상태인데 추론 요청이 오면 Fresh Loading 진행
                if !baking_only && is_baking_loaded {
                    // 통과하여 아래 리로드 로직(Fresh Loading) 실행
                } else {
                    return Ok(());
                }
            }
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
            println!("[MODEL] Loading Qwen 3.5 Generator (2B) (Vision: {})...", needs_vision);
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

    // 🌟 [신규 추가] 모델 파일 존재 여부만 가볍게 체크하는 함수 (메모리 로딩 안 함)
    // 🌟 [신규 추가] 모델 파일 존재 여부만 가볍게 체크하는 함수 (메모리 로딩 안 함)
    pub async fn check_embedding_downloaded(&self) -> anyhow::Result<()> {
        let weights_path = self.embedding_path.join("model.safetensors");
        if !weights_path.exists() {
            let err_msg = "Embedding model is missing. Please go to the Settings tab and download the required models.";
            println!("[MODEL] 🚨 {}", err_msg);
            use tauri::Emitter;
            let _ = self.app_handle.emit("app_error_alert", serde_json::json!({ "message": err_msg }));
            return Err(anyhow::anyhow!(err_msg));
        }
        Ok(())
    }

    pub async fn ensure_embedding(&self) -> anyhow::Result<()> {
        // 실제 메모리에 올리기 직전에 파일 존재 여부를 다시 한 번 방어합니다.
        self.check_embedding_downloaded().await?; 
        
        let mut emb_guard = self.embedding_model.lock().await;
        if emb_guard.is_none() {
            let self_clone = self.embedding_path.clone();
            
            // 🌟 CPU 강제 할당을 제거하고 시스템 설정(GPU)을 그대로 사용하여 초고속 VRAM 연산을 수행합니다.
            let target_device = self.device_config.device.clone(); 
            
            println!("[MODEL] Loading Embedding Model on {:?}...", target_device);
            
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
        let qwen3_5_model_path = normalize_path(base_path.join("Qwen3.5-2B-Instruct-gguf"));
        let embedding_path = base_path.join("granite-embedding-97m-multilingual-r2");

        let max_tokens_limit = 65536; 

        Ok(Self {
            app_handle,
            generator: Arc::new(TokioMutex::new(None)),
            qwen3_generator: Arc::new(TokioMutex::new(None)), // 🌟 추가
            qwen3_5_generator: Arc::new(TokioMutex::new(None)),
            embedding_model: Arc::new(TokioMutex::new(None)),
            embedding_cache: Arc::new(TokioMutex::new(std::collections::HashMap::new())), // 🌟 캐시 초기화
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
        emit_term("[STAGE-1] Preparing VRAM and Loading Qwen3.5 (2B) Vision Model...");

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
                    json!({ "category": "Vision (Step 1/2)", "summary": "Identifying document type..." }), 128, cancel_token.clone(), Some(task_id.clone()), None
                ).await?;
                
                let detected_type = crate::parsing::parse_json_from_llm(&type_res)
                    .get("doc_type").and_then(|v| v.as_str()).unwrap_or("Unknown").to_string();
                emit_term(&format!("✅ Document identified as: **{}**", detected_type));

                // 🌟 [개선된 분기 포인트] TRACKING(운송장)으로 판별되면 무거운 Slice & Merge를 우회합니다!
                if detected_type == "TRACKING" {
                    emit_term("[STAGE-2] 📦 Fast-Tracking Parcel Label...");
                    
                    let prompt = crate::parsing::get_image_extraction_prompt("kr", &language, "tracking", "");
                    let (_track_bias, track_prej) = crate::parsing::get_vision_tracking_bias(&language); // 🌟 Bias 호출
                    let result_str = self.chat_with_qwen3_5_image_spinner(
                        "You are a highly precise logistics data extraction assistant.", &prompt, Some(dynamic_image.clone()), app_handle, "extraction-progress", 
                        json!({ "category": "Vision Analysis", "summary": "Extracting Tracking Label data..." }), 512, cancel_token.clone(), Some(task_id.clone()), Some(&track_prej)
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
                            json!({ "category": format!("Vision (Slice {}/{})", idx+1, missions.len()), "summary": summary_msg }), 1024, cancel_token.clone(), Some(task_id.clone()), None
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
                let (_track_bias, track_prej) = crate::parsing::get_vision_tracking_bias(&language); // 🌟 Bias 호출
                
                let result_str = self.chat_with_qwen3_5_image_spinner(
                    "You are a precise commerce and logistics extraction assistant.", &prompt, Some(dynamic_image.clone()), app_handle, "extraction-progress", 
                    json!({ "category": "Vision Analysis", "summary": "Analyzing commerce tracking/goods..." }), 1024, cancel_token.clone(), Some(task_id.clone()), Some(&track_prej)
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
        session_id: Option<String>,
        semantic_prejudice: Option<&str>   // 🌟 추가
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
            Some("inference".to_string()),
            None, // 🌟 5번째 인자인 ignore_list 자리에 None을 명시적으로 추가합니다.
            semantic_prejudice  // 🌟 변경
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
            
            let response = gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))?;
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
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
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
        
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
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
        
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
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
        
        gen.generate(params, cancel_token, session_id, kv_name, None).await.map_err(|e| anyhow!("Inference failed: {}", e))
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
        // 🌟 1. 메모리 캐시부터 확인합니다 (중복된 텍스트면 GPU 연산 원천 차단)
        {
            let cache = self.embedding_cache.lock().await;
            if let Some(vector) = cache.get(&text) {
                return Ok(vector.clone());
            }
        }

        // Ensure embedding model is loaded (and generator is unloaded)
        self.ensure_embedding().await?;

        let embedding_model_arc = self.embedding_model.clone();
        let text_clone = text.clone();
        
        let vector = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f32>> {
            let guard = embedding_model_arc.blocking_lock();
            if let Some(model) = guard.as_ref() {
                model.embed(&text_clone).map_err(|e| anyhow::anyhow!("Embedding error: {}", e))
            } else {
                // Fallback to zeros if model failed to load
                Ok(vec![0.0; 384])
            }
        }).await??;

        // 🌟 2. 새로 연산된 벡터를 해시맵 캐시에 저장하여 다음 루프 때 재사용합니다.
        {
            let mut cache = self.embedding_cache.lock().await;
            cache.insert(text, vector.clone());
        }

        Ok(vector)
    }

    pub async fn get_embedding_batch(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.ensure_embedding().await?;
        let embedding_model_arc = self.embedding_model.clone();
        
        let vectors = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Vec<f32>>> {
            let guard = embedding_model_arc.blocking_lock();
            if let Some(model) = guard.as_ref() {
                model.embed_batch(&texts).map_err(|e| anyhow::anyhow!("Embedding error: {}", e))
            } else {
                Ok(vec![vec![0.0; 384]; texts.len()])
            }
        }).await??;

        Ok(vectors)
    }

    // [신규] Commerce 파이프라인: 2-Stage (0.6B para2graph -> 2B graph2contexts)
    // [신규] Commerce 파이프라인: 2-Stage (2B 단일 모델 연속 처리)
    pub async fn parse_commerce_query(&self, task_id: &str, app_handle: &tauri::AppHandle, query: String, language: &str, metrics_json: &str, cancel_token: Arc<AtomicBool>) -> anyhow::Result<Value> {
        use tauri::Emitter;

        // 🌟 [신규] 터미널 로거 헬퍼 주입
        let emit_term = |msg: &str| {
            println!("{}", msg);
            let _ = app_handle.emit("task-console-log", json!({"task_id": task_id, "text": format!("{}\n", msg)}));
        };

        emit_term("[ENGINE] 🚀 Starting Commerce Search Pipeline...");
        
        // ----------------------------------------------------
        // Stage 1: 세그먼트 분할 (Vector Cliff Detection) - Embedding 모델 사용
        // ----------------------------------------------------
        emit_term("[STAGE-1] Loading Embedding Model for Semantic Chunking...");
        let payload = json!({ "task_id": task_id, "category": "Stage 1", "summary": "Segmenting semantic intents...", "spinner": "⠋" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::scheduler::log_task_progress(app_handle, task_id, &payload);

        self.ensure_embedding().await?;
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        // 🌟 [CRITICAL FIX] 52개 글로벌 언어 및 유니코드 블록 완벽 감지 로직
        // 특수 문자가 감지되면 해당 언어(ru, ar, hi 등)를 강제 지정하고,
        // 알파벳(라틴) 기반이면 프론트엔드에서 전달받은 정확한 locale(language)을 신뢰하여 폴백(Fallback)합니다.
        let mut lang_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        let mut latin_count = 0;
        
        for c in query.chars() {
            let u = c as u32;
            if u >= 0xAC00 && u <= 0xD7A3 { *lang_counts.entry("ko").or_insert(0) += 1; } // Korean
            else if u >= 0x3040 && u <= 0x30FF { *lang_counts.entry("ja").or_insert(0) += 1; } // Japanese
            else if u >= 0x4E00 && u <= 0x9FFF { *lang_counts.entry("zh").or_insert(0) += 1; } // Chinese
            else if u >= 0x0400 && u <= 0x052F { *lang_counts.entry("ru").or_insert(0) += 1; } // Cyrillic (ru, bg, uk, kk, sr)
            else if u >= 0x0600 && u <= 0x06FF { *lang_counts.entry("ar").or_insert(0) += 1; } // Arabic (ar, fa, ur)
            else if u >= 0x0E00 && u <= 0x0E7F { *lang_counts.entry("th").or_insert(0) += 1; } // Thai
            else if u >= 0x0370 && u <= 0x03FF { *lang_counts.entry("el").or_insert(0) += 1; } // Greek
            else if u >= 0x0590 && u <= 0x05FF { *lang_counts.entry("he").or_insert(0) += 1; } // Hebrew
            else if u >= 0x0900 && u <= 0x097F { *lang_counts.entry("hi").or_insert(0) += 1; } // Devanagari (hi, mr)
            else if u >= 0x0980 && u <= 0x09FF { *lang_counts.entry("bn").or_insert(0) += 1; } // Bengali
            else if u >= 0x0C00 && u <= 0x0C7F { *lang_counts.entry("te").or_insert(0) += 1; } // Telugu
            else if u >= 0x1780 && u <= 0x17FF { *lang_counts.entry("km").or_insert(0) += 1; } // Khmer
            else if (u >= 0x0041 && u <= 0x005A) || (u >= 0x0061 && u <= 0x007A) || (u >= 0x00C0 && u <= 0x024F) || (u >= 0x1E00 && u <= 0x1EFF) { 
                latin_count += 1; // Latin-based (en, es, fr, de, pt, vi, nl, it, etc.)
            }
        }
        
        let query_lang = if !lang_counts.is_empty() {
            lang_counts.into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(lang, count)| if count > 1 { lang.to_string() } else { language.to_string() })
                .unwrap_or_else(|| language.to_string())
        } else if latin_count > 1 {
            // 알파벳 언어권인 경우 UI의 정확한 국가 언어 설정값을 승계합니다.
            language.to_string()
        } else {
            "en".to_string()
        };

        let categories = ["order", "goods", "tracking", "review", "coupon", "event"];
        let mut layout_embs = std::collections::HashMap::new();

        let mut texts_to_embed = Vec::new();
        let mut emb_mappings = Vec::new();

        // 🌟 [변경] Stage 1 멀티패스 전용 함수인 get_multi_pass_contexts를 호출하여
        // layout_list, layout_form 및 core_intent를 포함한 100% 모든 속성을 수집합니다.
        for cat in &categories {
            let contexts = crate::parsing::get_multi_pass_contexts(cat, &query_lang);
            
            for (key, bias, prejudice) in contexts {
                texts_to_embed.push(bias);
                emb_mappings.push((cat.to_string(), format!("{}_bias", key)));

                let final_prej = if prejudice.trim().is_empty() { "random unrelated noise".to_string() } else { prejudice };
                texts_to_embed.push(final_prej);
                emb_mappings.push((cat.to_string(), format!("{}_prejudice", key)));
            }
        }

        // 2. 단 한 번의 배치 호출로 모든 임베딩 벡터를 한 장바구니에 획득
        let embedded_texts = self.get_embedding_batch(texts_to_embed).await.unwrap_or_else(|_| vec![vec![0.0; 384]; emb_mappings.len()]);

        // 3. 획득한 벡터와 카테고리/키 값을 매칭하여 해시맵에 일괄 삽입
        for (i, (cat, emb_type)) in emb_mappings.into_iter().enumerate() {
            layout_embs.insert(format!("{}_{}", cat, emb_type), embedded_texts[i].clone());
        }

        fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
            let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot_product / (norm_a * norm_b) }
        }

        let words: Vec<&str> = query.split_whitespace().collect();
        let mut context_arr = Vec::new();

        // 🌟 [1차 패스] 최소 2단어 이상(2-gram)의 교차 윈도우 스팬 및 카테고리별 기본 점수 수집
        struct SpanData {
            start: usize,
            end: usize,
            text: String,
            scores: std::collections::HashMap<String, f32>,
        }
        let mut raw_spans = Vec::new();

        for start in 0..words.len() {
            let max_end = words.len().min(start + 8);
            
            // 🌟 [단어 수 제한] start + 2 로 설정하여 단일 단어(1단어)는 배제합니다.
            for end in (start + 2)..=max_end {
                if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                
                let test_text = words[start..end].join(" ");
                let test_emb = self.get_embedding(test_text.clone()).await.unwrap_or(vec![0.0; 384]);

                let mut scores = std::collections::HashMap::new();
                for cat in &categories {
                    let contexts = crate::parsing::get_multi_pass_contexts(cat, &query_lang);
                    let mut field_scores = Vec::new();

                    for (key, _bias, _prejudice) in contexts {
                        let bias_emb = layout_embs.get(&format!("{}_{}_bias", cat, key)).cloned().unwrap_or(vec![0.0; 384]);
                        let prej_emb = layout_embs.get(&format!("{}_{}_prejudice", cat, key)).cloned().unwrap_or(vec![0.0; 384]);
                        
                        let bias_score = cosine_similarity(&test_emb, &bias_emb);
                        let prej_score = cosine_similarity(&test_emb, &prej_emb);

                        let field_score = (bias_score - prej_score).max(0.0);
                        field_scores.push(field_score);
                    }

                    // 🌟 [멀티 패스 스코어 평가]
                    field_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    
                    // 🌟 [진정한 멀티패스 반영: 동적 감쇠 누적 합산 (Decaying Sum)]
                    // 상위 N개 점수를 동적으로 순회하며 가중치를 반감(1.0, 0.5, 0.25, 0.125...)시켜 합산합니다.
                    // 무한히 더해도 최대값이 수렴하므로, 필드 개수가 많은(예: 40개) 도메인이 
                    // 잡음(Noise)을 끌어모아 점수를 뻥튀기하는 현상을 수학적으로 완벽히 차단합니다.
                    let mut multi_pass_score = 0.0;
                    let mut weight = 1.0;
                    
                    // 🌟 상위 5개까지만 유의미한 멀티패스 공명으로 인정하여 합산합니다.
                    let max_pass = field_scores.len().min(5);
                    for i in 0..max_pass {
                        multi_pass_score += field_scores[i] * weight;
                        weight *= 0.5; // 다음 순위의 필드 점수는 반영 비율을 절반으로 깎습니다.
                    }
                    
                    // 🌟 [단어 개수 가중치 상향] 단어가 많이 합쳐질수록 문맥이 명확해지므로 길이에 비례하여 가중치를 부여합니다.
                    // 파편이 긴 문장을 잡아먹는 현상(NMS Battle 하극상)을 완벽히 막기 위해 가산점을 단어당 15%로 대폭 상향합니다.
                    let word_count = end - start;
                    let length_weight = 1.0 + ((word_count as f32 - 2.0) * 0.15); 
                    let weighted_base_score = multi_pass_score * length_weight;
                    
                    scores.insert(cat.to_string(), weighted_base_score);
                }
                raw_spans.push(SpanData { start, end, text: test_text, scores });
            }
        }

        // 🌟 [2차 패스] 앞뒤 교차 문장 점수 합산을 통한 최종 컨텍스트 점수 도출
        // 🌟 [2차 패스] 앞뒤 교차 문장 점수 합산 및 임시 목록 저장
        struct EvaluatedSpan {
            start: usize,
            end: usize,
            text: String,
            best_cat: String,
            context_score: f32,
            intersecting: Vec<String>,
            base_score: f32,
        }
        let mut evaluated_spans = Vec::new();

        for i in 0..raw_spans.len() {
            let target = &raw_spans[i];
            let mut contextual_scores: Vec<(String, f32)> = Vec::new();

            for cat in &categories {
                let base_score = *target.scores.get(*cat).unwrap_or(&0.0);
                
                let mut prev_bonus = 0.0;
                let mut next_bonus = 0.0;
                
                for j in 0..raw_spans.len() {
                    if i == j { continue; }
                    let other = &raw_spans[j];
                    let o_score = *other.scores.get(*cat).unwrap_or(&0.0);
                    
                    // 앞쪽 교차 문장: 시작점이 앞서면서 현재 문장과 겹침
                    if other.start < target.start && other.end > target.start {
                        if o_score > prev_bonus { prev_bonus = o_score; }
                    }
                    // 뒤쪽 교차 문장: 끝점이 뒤서면서 현재 문장과 겹침
                    if other.end > target.end && other.start < target.end {
                        if o_score > next_bonus { next_bonus = o_score; }
                    }
                }
                
                // 중심 점수에 앞뒤 교차 점수를 50% 가중치로 합산하여 자연스러운 의미 뭉치 우선순위 상향
                let final_context_score = base_score + (prev_bonus * 0.5) + (next_bonus * 0.5);
                contextual_scores.push((cat.to_string(), final_context_score));
            }

            // 최종 합산 점수 기준 내림차순 정렬
            contextual_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let current_best_cat = contextual_scores[0].0.clone();
            let current_max_contextual_score = contextual_scores[0].1;

            // 🌟 [커트라인 완전 해제] original_base_score 및 0.45 커트라인을 제거했습니다.
            // 최소한의 유사도(0.0 초과)만 있다면 모두 후보군으로 올리고, 길이와 문맥이 반영된 NMS 배틀을 통해 최강자만 살아남게 합니다.
            if current_max_contextual_score > 0.0 {
                let mut intersecting_categories: Vec<String> = Vec::new();
                for (cat_name, c_score) in &contextual_scores {
                    if *c_score >= current_max_contextual_score - 0.25 && *c_score > 0.0 {
                        intersecting_categories.push(cat_name.clone());
                    }
                }
                if intersecting_categories.is_empty() {
                    intersecting_categories.push(current_best_cat.clone());
                }

                // 🌟 유효 텍스트 후보군 출력 (Base Score 출력 제거)
                emit_term(&format!("  🟢 [CANDIDATE] '{}' -> Domain: {} (Context Score: {:.4})", target.text, current_best_cat, current_max_contextual_score));

                evaluated_spans.push(EvaluatedSpan {
                    start: target.start,
                    end: target.end,
                    text: target.text.clone(),
                    best_cat: current_best_cat,
                    context_score: current_max_contextual_score,
                    intersecting: intersecting_categories,
                    base_score: 0.0, // 구조체 호환성을 위해 0.0으로 고정
                });
            }
        }

        // 🌟 [3차 패스] 오버랩(교차) 충돌 해결 (길이 가중치가 포함된 최종 점수 기준 내림차순 정렬)
        evaluated_spans.sort_by(|a, b| b.context_score.partial_cmp(&a.context_score).unwrap_or(std::cmp::Ordering::Equal));

        let mut final_selected_spans: Vec<EvaluatedSpan> = Vec::new();

        emit_term("\n  ⚔️ [NMS BATTLE] Resolving Overlaps...");

        for span in evaluated_spans {
            let mut is_overlapped = false;
            let mut winner_text = String::new();

            // 이미 승리하여 선택된 상위 점수의 스팬들과 현재 스팬이 교차하는지 검사합니다.
            for selected in &final_selected_spans {
                let overlaps = span.start < selected.end && span.end > selected.start;
                
                if overlaps {
                    // 점수가 낮은 현재 스팬은 패배하여 탈락합니다.
                    is_overlapped = true;
                    winner_text = selected.text.clone();
                    break;
                }
            }

            if !is_overlapped {
                emit_term(&format!("    👑 [WINNER] '{}' (Score: {:.4}) survives.", span.text, span.context_score));
                final_selected_spans.push(span);
            } else {
                emit_term(&format!("    💀 [DEFEAT] '{}' is absorbed by higher score winner '{}'.", span.text, winner_text));
            }
        }

        // 프론트엔드로 보내기 위해 최종 생존한 문맥들을 원래 문장의 단어 순서대로 재정렬합니다.
        final_selected_spans.sort_by(|a, b| a.start.cmp(&b.start));

        // 🌟 [4차 패스] Gap Bridging & Score Battle (고아 단어 구출 및 양방향 흡수 대결)
        // NMS 배틀에서 탈락하여 붕 떠버린 단어(Gap)들을 양쪽 승자 문맥에 각각 붙여보고, 더 높은 멀티패스 점수를 내는 쪽이 흡수합니다.
        if !final_selected_spans.is_empty() {
            emit_term("\n  🌉 [GAP BRIDGING] Rescuing orphaned words via Score Battle...");
            
            let mut final_bounds: Vec<(usize, usize, String, f32, Vec<String>)> = final_selected_spans
                .into_iter()
                .map(|s| (s.start, s.end, s.best_cat, s.context_score, s.intersecting))
                .collect();

            // 1. 왼쪽 끝(Left Edge) 고아 단어 무조건 흡수 (예: 문장 맨 앞의 "여름")
            if final_bounds[0].0 > 0 {
                let gap_start = 0;
                let gap_end = final_bounds[0].0;
                let gap_text = words[gap_start..gap_end].join(" ");
                emit_term(&format!("    🛠️ [LEFT EDGE] '{}' is absorbed by '{}'", gap_text, words[final_bounds[0].0..final_bounds[0].1].join(" ")));
                final_bounds[0].0 = 0;
            }

            // 2. 중간(Gap) 고아 단어 양방향 점수 대결 흡수 (예: "20%에", "속하지만")
            for i in 0..(final_bounds.len() - 1) {
                let gap_start = final_bounds[i].1;
                let gap_end = final_bounds[i+1].0;

                if gap_start < gap_end {
                    let gap_text = words[gap_start..gap_end].join(" ");

                    // 대결 A: 왼쪽 승자가 흡수했을 때의 멀티패스 점수 계산
                    let left_cat = &final_bounds[i].2;
                    let left_test_text = words[final_bounds[i].0..gap_end].join(" ");
                    let left_emb = self.get_embedding(left_test_text.clone()).await.unwrap_or(vec![0.0; 384]);
                    let left_contexts = crate::parsing::get_multi_pass_contexts(left_cat, &query_lang);
                    
                    let mut left_scores = Vec::new();
                    for (key, _bias, _prej) in left_contexts {
                        let bias_emb = layout_embs.get(&format!("{}_{}_bias", left_cat, key)).cloned().unwrap_or(vec![0.0; 384]);
                        let prej_emb = layout_embs.get(&format!("{}_{}_prejudice", left_cat, key)).cloned().unwrap_or(vec![0.0; 384]);
                        left_scores.push((cosine_similarity(&left_emb, &bias_emb) - cosine_similarity(&left_emb, &prej_emb)).max(0.0));
                    }
                    left_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let mut left_score = 0.0;
                    let mut weight = 1.0;
                    for j in 0..left_scores.len().min(5) { left_score += left_scores[j] * weight; weight *= 0.5; }

                    // 대결 B: 오른쪽 승자가 흡수했을 때의 멀티패스 점수 계산
                    let right_cat = &final_bounds[i+1].2;
                    let right_test_text = words[gap_start..final_bounds[i+1].1].join(" ");
                    let right_emb = self.get_embedding(right_test_text.clone()).await.unwrap_or(vec![0.0; 384]);
                    let right_contexts = crate::parsing::get_multi_pass_contexts(right_cat, &query_lang);
                    
                    let mut right_scores = Vec::new();
                    for (key, _bias, _prej) in right_contexts {
                        let bias_emb = layout_embs.get(&format!("{}_{}_bias", right_cat, key)).cloned().unwrap_or(vec![0.0; 384]);
                        let prej_emb = layout_embs.get(&format!("{}_{}_prejudice", right_cat, key)).cloned().unwrap_or(vec![0.0; 384]);
                        right_scores.push((cosine_similarity(&right_emb, &bias_emb) - cosine_similarity(&right_emb, &prej_emb)).max(0.0));
                    }
                    right_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let mut right_score = 0.0;
                    let mut weight = 1.0;
                    for j in 0..right_scores.len().min(5) { right_score += right_scores[j] * weight; weight *= 0.5; }

                    // ⚔️ 대결 결과 판정 및 최종 흡수
                    if left_score >= right_score {
                        emit_term(&format!("    ⚔️ [GAP BATTLE] Gap '{}' -> LEFT WINS! (Left: {:.4} > Right: {:.4})", gap_text, left_score, right_score));
                        final_bounds[i].1 = gap_end; 
                        final_bounds[i].3 = left_score; // 점수 갱신
                    } else {
                        emit_term(&format!("    ⚔️ [GAP BATTLE] Gap '{}' -> RIGHT WINS! (Right: {:.4} > Left: {:.4})", gap_text, right_score, left_score));
                        final_bounds[i+1].0 = gap_start; 
                        final_bounds[i+1].3 = right_score; // 점수 갱신
                    }
                }
            }

            // 3. 오른쪽 끝(Right Edge) 고아 단어 무조건 흡수 (예: 문장 맨 끝의 "시급해")
            let last_idx = final_bounds.len() - 1;
            if final_bounds[last_idx].1 < words.len() {
                let gap_start = final_bounds[last_idx].1;
                let gap_end = words.len();
                let gap_text = words[gap_start..gap_end].join(" ");
                emit_term(&format!("    🛠️ [RIGHT EDGE] '{}' is absorbed by '{}'", gap_text, words[final_bounds[last_idx].0..final_bounds[last_idx].1].join(" ")));
                final_bounds[last_idx].1 = words.len();
            }

            // 4. 최종 조립된 결과를 배열에 삽입
            for (start, end, best_cat, context_score, intersecting) in final_bounds {
                let final_text = words[start..end].join(" ");
                emit_term(&format!("  📈 [CROSS MATCH FINAL] Intersection: {:?} -> '{}' (Context Score: {:.4})", intersecting, final_text, context_score));

                context_arr.push(json!({
                    "type": best_cat,
                    "types": intersecting,
                    "text": final_text,
                    "score": context_score
                }));
            }
        }

        let mut segments = json!({
            "original_text": query.clone(),
            "context": context_arr
        });
        
        // Stage 1 완료 후 최종 분할된 맥락 트리 전체를 출력합니다.
        emit_term("\n=======================================");
        emit_term("[STAGE-1 RESULT] 🧩 Semantic Chunking Complete:");
        emit_term(&serde_json::to_string_pretty(&segments).unwrap_or_default());
        emit_term("=======================================\n");

        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        // ----------------------------------------------------
        // Stage 2 & 3: Double Plinko Attribute/Operator Mapping & LLM Normalization
        // ----------------------------------------------------
        emit_term("[STAGE-2] Extracting attributes via Double Vector Plinko & LLM Normalization...");
        
        if let Some(ctx_arr) = segments.get_mut("context").and_then(|v| v.as_array_mut()) {
            let total_segments = ctx_arr.len();

            for (idx, seg) in ctx_arr.iter_mut().enumerate() {
                if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                let payload = json!({ "task_id": task_id, "category": format!("Stage 2 ({}/{})", idx+1, total_segments), "summary": "Mapping attributes...", "spinner": "⠋" });
                let _ = app_handle.emit("extraction-progress", &payload);
                crate::scheduler::log_task_progress(app_handle, task_id, &payload);

                let current_text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let seg_type = seg.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                
                // 🌟 [2차 분기] 세부 속성 매칭: 해당 도메인 타입의 Schema Field(Property Bias/Prej) 로드
                let fields = crate::parsing::get_detail_schema_fields(&seg_type, "", language);
                
                let mut prop_keys = Vec::new();
                let mut bias_texts = Vec::new();
                let mut prej_texts = Vec::new();
                
                for (key, _, bias, prej) in fields {
                    prop_keys.push(key);
                    bias_texts.push(bias);
                    prej_texts.push(if prej.trim().is_empty() { "random unrelated noise".to_string() } else { prej });
                }

                // 🌟 [3차 분기] 연산자 매칭: Operators Schema Field(Bias/Prej) 로드
                let op_keys = vec!["eq", "lte", "lt", "gte", "gt", "top", "bottom"];
                let mut op_bias_texts = Vec::new();
                let mut op_prej_texts = Vec::new();

                for op in &op_keys {
                    let mut b_text = String::new();
                    let mut p_text = String::new();
                    if let Some(op_obj) = crate::parsing::BIAS_DICT.get("operators").and_then(|o| o.get(*op)) {
                        if let Some(b) = op_obj.get("bias").and_then(|v| v.as_str()) { b_text = b.to_string(); }
                        if let Some(p) = op_obj.get("prejudice").and_then(|v| v.as_str()) { p_text = p.to_string(); }
                    }
                    op_bias_texts.push(b_text);
                    op_prej_texts.push(if p_text.trim().is_empty() { "random unrelated noise".to_string() } else { p_text });
                }

                // 🌟 [4차 분기] 수치/단위 메트릭(Metric Type) 매칭: metrics Schema Field 로드
                let metric_keys = vec!["date", "time", "price", "discount", "quantity", "ratio"];
                let mut metric_bias_texts = Vec::new();
                let mut metric_prej_texts = Vec::new();

                for metric in &metric_keys {
                    let mut b_text = String::new();
                    let mut p_text = String::new();
                    if let Some(m_obj) = crate::parsing::BIAS_DICT.get("metrics").and_then(|o| o.get(*metric)) {
                        if let Some(b) = m_obj.get("bias").and_then(|v| v.as_str()) { b_text = b.to_string(); }
                        if let Some(p) = m_obj.get("prejudice").and_then(|v| v.as_str()) { p_text = p.to_string(); }
                    }
                    metric_bias_texts.push(b_text);
                    metric_prej_texts.push(if p_text.trim().is_empty() { "random unrelated noise".to_string() } else { p_text });
                }

                // 🌟 [5차 분기-A] 상대적 시간 의도(Time Filters) 매칭 (bias.json 100% 의존)
                // 🌟 [5차 분기-A] 상대적 시간 의도(Time Filters) 매칭 (bias.json 100% 동적 추출)
                let time_keys: Vec<String> = crate::parsing::BIAS_DICT
                    .get("time_filters")
                    .and_then(|v| v.as_object())
                    .map(|obj| obj.keys().cloned().collect())
                    .unwrap_or_else(|| vec!["today".to_string(), "yesterday".to_string(), "this_month".to_string(), "last_month".to_string(), "this_year".to_string(), "last_year".to_string(), "recently".to_string()]);
                let mut time_bias_texts = Vec::new();
                let mut time_prej_texts = Vec::new();

                for tk in &time_keys {
                    // 🌟 [Option 2] 벡터 공간에서 다른 의도와 혼동되지 않도록 강력한 'Context Prefix'를 주입합니다.
                    let mut b_text = format!("Time context: {} period", tk); 
                    let mut p_text = format!("Time context: opposite not {}", tk);
                    if let Some(t_obj) = crate::parsing::BIAS_DICT.get("time_filters").and_then(|o| o.get(tk.clone())) {
                        if let Some(b) = t_obj.get("bias").and_then(|v| v.as_str()) { b_text = format!("Time context: {}", b); }
                        if let Some(p) = t_obj.get("prejudice").and_then(|v| v.as_str()) { p_text = format!("Time context: {}", p); }
                    }
                    time_bias_texts.push(b_text);
                    time_prej_texts.push(p_text);
                }

                // 🌟 [5차 분기-B] 계절 의도(Season Filters) 매칭 (bias.json 100% 의존)
                let season_keys: Vec<String> = crate::parsing::BIAS_DICT
                    .get("season_filters")
                    .and_then(|v| v.as_object())
                    .map(|obj| obj.keys().cloned().collect())
                    .unwrap_or_else(|| vec!["spring".to_string(), "summer".to_string(), "autumn".to_string(), "winter".to_string()]);
                let mut season_bias_texts = Vec::new();
                let mut season_prej_texts = Vec::new();
                let mut season_exact_matches: Vec<Vec<String>> = Vec::new(); // 🌟 신규: 다국어 Exact Match 캐시

                for sk in &season_keys {
                    // 🌟 [Option 2] 계절 데이터에도 강력한 도메인 접두사를 주입하여 다른 계절의 배제 단어들과 거리를 벌립니다.
                    let mut b_text = format!("Season context: {} weather", sk); 
                    let mut p_text = format!("Season context: not {}", sk);
                    let mut exact_words = Vec::new(); // 🌟 추가

                    if let Some(s_obj) = crate::parsing::BIAS_DICT.get("season_filters").and_then(|o| o.get(sk.clone())) {
                        if let Some(b) = s_obj.get("bias").and_then(|v| v.as_str()) { b_text = format!("Season context: {}", b); }
                        if let Some(p) = s_obj.get("prejudice").and_then(|v| v.as_str()) { p_text = format!("Season context: {}", p); }
                        
                        // 🌟 신규: exact_match 배열을 추출하여 소문자로 변환해 저장
                        if let Some(arr) = s_obj.get("exact_match").and_then(|v| v.as_array()) {
                            for v in arr {
                                if let Some(s) = v.as_str() {
                                    exact_words.push(s.to_lowercase());
                                }
                            }
                        }
                    }
                    season_bias_texts.push(b_text);
                    season_prej_texts.push(p_text);
                    season_exact_matches.push(exact_words); // 🌟 신규
                }
                
                // Batch Embedding (Bias & Prejudice) 동시 장전
                let bias_embs = self.get_embedding_batch(bias_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; prop_keys.len()]);
                let prej_embs = self.get_embedding_batch(prej_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; prop_keys.len()]);
                
                let op_bias_embs = self.get_embedding_batch(op_bias_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; op_keys.len()]);
                let op_prej_embs = self.get_embedding_batch(op_prej_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; op_keys.len()]);

                let metric_bias_embs = self.get_embedding_batch(metric_bias_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; metric_keys.len()]);
                let metric_prej_embs = self.get_embedding_batch(metric_prej_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; metric_keys.len()]);

                // 🌟 시간 및 계절 의도 벡터 일괄 연산
                let time_bias_embs = self.get_embedding_batch(time_bias_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; time_keys.len()]);
                let time_prej_embs = self.get_embedding_batch(time_prej_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; time_keys.len()]);

                let season_bias_embs = self.get_embedding_batch(season_bias_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; season_keys.len()]);
                let season_prej_embs = self.get_embedding_batch(season_prej_texts).await.unwrap_or_else(|_| vec![vec![0.0; 384]; season_keys.len()]);

                // Plinko Game: Sliding Window Cliff Detection over words
                let mut plinko_map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
                let words: Vec<&str> = current_text.split_whitespace().collect();
                
                let mut current_chunk = Vec::new();
                let mut prev_max_score = -1.0;
                let mut best_prop_for_chunk = String::new();

                for word in words {
                    let mut test_chunk = current_chunk.clone();
                    test_chunk.push(word);
                    let test_text = test_chunk.join(" ");
                    let test_emb = self.get_embedding(test_text).await.unwrap_or(vec![0.0; 384]);

                    let mut current_max = -1.0;
                    let mut current_best = String::new();

                    for i in 0..prop_keys.len() {
                        let b_score = cosine_similarity(&test_emb, &bias_embs[i]);
                        let p_score = cosine_similarity(&test_emb, &prej_embs[i]);
                        let score = b_score - p_score;
                        if score > current_max {
                            current_max = score;
                            current_best = prop_keys[i].clone();
                        }
                    }

                    // Score Drop (Cliff) = Cut & Drop into Slot
                    if current_max < prev_max_score && !current_chunk.is_empty() {
                        if prev_max_score > 0.10 && !best_prop_for_chunk.is_empty() {
                            plinko_map.entry(best_prop_for_chunk.clone()).or_default().push(current_chunk.join(" "));
                        }
                        
                        // Reset Window
                        current_chunk = vec![word];
                        let reset_emb = self.get_embedding(word.to_string()).await.unwrap_or(vec![0.0; 384]);
                        let mut r_max = -1.0;
                        let mut r_best = String::new();
                        for i in 0..prop_keys.len() {
                            let score = cosine_similarity(&reset_emb, &bias_embs[i]) - cosine_similarity(&reset_emb, &prej_embs[i]);
                            if score > r_max {
                                r_max = score;
                                r_best = prop_keys[i].clone();
                            }
                        }
                        prev_max_score = r_max;
                        best_prop_for_chunk = r_best;
                    } else {
                        current_chunk.push(word);
                        prev_max_score = current_max;
                        best_prop_for_chunk = current_best;
                    }
                }
                
                // Sweep remaining chunk
                if !current_chunk.is_empty() && prev_max_score > 0.10 && !best_prop_for_chunk.is_empty() {
                    plinko_map.entry(best_prop_for_chunk).or_default().push(current_chunk.join(" "));
                }

                // 🌟 [STAGE-1 계승] 시간(Time)과 계절(Season) 판별에 Stage 1의 '슬라이딩 윈도우 + NMS 교차 검증' 로직을 도입하여 정밀도를 극대화합니다!
                let temporal_words: Vec<&str> = current_text.split_whitespace().collect();
                
                #[derive(Clone)]
                struct TemporalSpan {
                    start: usize,
                    end: usize,
                    text: String,
                    best_intent: String,
                    group: String,
                    score: f32,
                }
                let mut temp_raw_spans = Vec::new();

                emit_term(&format!("  🔍 [PASS 1: SLIDING WINDOW] Analayzing chunks for '{}'", current_text));

                // 🌟 1차 패스: 슬라이딩 윈도우 (1~4단어 조합으로 쪼개서 타격)
                for start in 0..temporal_words.len() {
                    let max_end = temporal_words.len().min(start + 4);
                    for end in (start + 1)..=max_end {
                        let test_text = temporal_words[start..end].join(" ");
                        let test_emb = self.get_embedding(test_text.clone()).await.unwrap_or(vec![0.0; 384]);
                        
                        // 🌟 [CRITICAL FIX] 단 하나의 최고점만 뽑고 버리는 병목 현상을 완전히 제거했습니다.
                        // 한글 하드코딩 없이 bias.json의 모든 의도를 벡터 유사도(bias - prejudice)로 100% 평가하며,
                        // 임계값(-2.0)을 넘는 모든 후보를 로그에 출력하고 우선순위(NMS) 배틀에 전부 참전시킵니다.
                        let word_count = end - start;
                        let length_weight = 1.0 + ((word_count as f32 - 1.0) * 0.15); 

                        for i in 0..time_keys.len() {
                            let b_score = cosine_similarity(&test_emb, &time_bias_embs[i]);
                            let p_score = cosine_similarity(&test_emb, &time_prej_embs[i]);
                            
                            // 🌟 [Option 1] 단어 수(word_count)가 적을수록 문맥이 부족해 Prejudice(배제)에 과도하게 타격받는 현상을 방지합니다.
                            let p_weight = if word_count <= 2 { 0.3 } else { 0.7 };
                            let score = b_score - (p_score * p_weight);
                            
                            if score > 0.05 {
                                let weighted_time_score = score * length_weight;
                                emit_term(&format!("    🔹 [RAW-TIME] '{}' -> {} (Base: {:.4} * W: {:.2} = {:.4})", test_text, time_keys[i], score, length_weight, weighted_time_score));
                                temp_raw_spans.push(TemporalSpan { start, end, text: test_text.clone(), best_intent: time_keys[i].to_string(), group: "Time".to_string(), score: weighted_time_score });
                            }
                        }

                        for i in 0..season_keys.len() {
                            let b_score = cosine_similarity(&test_emb, &season_bias_embs[i]);
                            let p_score = cosine_similarity(&test_emb, &season_prej_embs[i]);
                            
                            // 🌟 [Option 1] '여름' 같은 짧은 단어가 'autumn'의 배제 단어에 포함되어 점수가 음수로 곤두박질치는 환각을 방지합니다.
                            let p_weight = if word_count <= 2 { 0.3 } else { 0.7 };
                            let mut score = b_score - (p_score * p_weight);
                            
                            // 🌟 [EXACT MATCH BOOST] 다국어 직접 매칭 (가중치 폭발)
                            let test_lower = test_text.to_lowercase();
                            let mut exact_hit = false;
                            for keyword in &season_exact_matches[i] {
                                if test_lower.contains(keyword) {
                                    exact_hit = true;
                                    break;
                                }
                            }
                            if exact_hit {
                                score += 0.4; // 🌟 직접 매칭 시 0.4의 고정 가산점 (NMS 배틀 무조건 압승)
                            }

                            if score > 0.05 {
                                let weighted_season_score = score * length_weight;
                                emit_term(&format!("    🔹 [RAW-SEASON] '{}' -> {} (Base: {:.4} * W: {:.2} = {:.4}){}", 
                                    test_text, season_keys[i], score, length_weight, weighted_season_score, if exact_hit { " [🔥 EXACT MATCH]" } else { "" }));
                                temp_raw_spans.push(TemporalSpan { start, end, text: test_text.clone(), best_intent: season_keys[i].to_string(), group: "Season".to_string(), score: weighted_season_score });
                            }
                        }
                    }
                }

                // 🌟 2차 패스: 앞뒤 교차 문장(Context) 점수 합산
                emit_term("  🔄 [PASS 2: CONTEXT ADJUSTMENT] Merging adjacent scores...");
                let mut temp_evaluated_spans = Vec::new();
                
                for i in 0..temp_raw_spans.len() {
                    let target = &temp_raw_spans[i];
                    let mut prev_bonus = 0.0;
                    let mut next_bonus = 0.0;

                    for j in 0..temp_raw_spans.len() {
                        if i == j { continue; }
                        let other = &temp_raw_spans[j];
                        // 🌟 [CRITICAL FIX] Time은 Time끼리, Season은 Season끼리만 문맥 보너스를 교환하도록 그룹 조건을 추가합니다.
                        if other.group == target.group && other.best_intent == target.best_intent {
                            if other.start < target.start && other.end > target.start && other.score > prev_bonus { prev_bonus = other.score; }
                            if other.end > target.end && other.start < target.end && other.score > next_bonus { next_bonus = other.score; }
                        }
                    }
                    
                    let final_context_score = target.score + (prev_bonus * 0.5) + (next_bonus * 0.5);
                    
                    emit_term(&format!("    🔸 [ADJUSTED-{}] '{}' -> {} (Score: {:.4} + Bonus: {:.4} = {:.4})", 
                        target.group.to_uppercase(), target.text, target.best_intent, target.score, (prev_bonus * 0.5) + (next_bonus * 0.5), final_context_score));

                    temp_evaluated_spans.push(TemporalSpan {
                        start: target.start, end: target.end, text: target.text.clone(),
                        best_intent: target.best_intent.clone(), group: target.group.clone(), score: final_context_score
                    });
                }

                // 🌟 3차 패스: NMS 오버랩(교차) 충돌 해결
                temp_evaluated_spans.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                let mut final_temporal_spans: Vec<TemporalSpan> = Vec::new();

                emit_term("  ⚔️ [PASS 3: NMS BATTLE] Resolving Overlaps...");
                for span in temp_evaluated_spans {
                    let mut is_overlapped = false;
                    for selected in &final_temporal_spans {
                        if span.start < selected.end && span.end > selected.start {
                            // Time과 Season이 서로 그룹이 다르면 공존 허용 (예: "올해 여름")
                            if span.group != selected.group {
                                continue;
                            }
                            is_overlapped = true;
                            break;
                        }
                    }
                    
                    if !is_overlapped {
                        emit_term(&format!("    👑 [WINNER] '{}' -> {} (Score: {:.4})", span.text, span.best_intent, span.score));
                        final_temporal_spans.push(span);
                    } else {
                        // 🌟 [LOGGING FIX] 패배하여 흡수(Absorbed)된 항목들도 원래의 점수를 로그에 함께 출력합니다.
                        emit_term(&format!("    💀 [DEFEAT] '{}' -> {} (Absorbed: {:.4})", span.text, span.best_intent, span.score));
                    }
                }

                let mut best_time_intent = String::from("none");
                let mut best_season_intent = String::from("none");
                let mut battle_logs = Vec::new();

                for span in &final_temporal_spans {
                    battle_logs.push(format!("'{}'->{}:{:.4}", span.text, span.best_intent, span.score));
                    if span.group == "Time" && best_time_intent == "none" { best_time_intent = span.best_intent.clone(); }
                    if span.group == "Season" && best_season_intent == "none" { best_season_intent = span.best_intent.clone(); }
                }

                if battle_logs.is_empty() {
                    battle_logs.push("No temporal intents found".to_string());
                }

                emit_term(&format!("  ✅ [GLOBAL TEMPORAL BATTLE (NMS)] {}", battle_logs.join(" | ")));

                let time_guide = if best_time_intent != "none" { format!(", Time Intent [{}]", best_time_intent) } else { "".to_string() };
                let season_guide = if best_season_intent != "none" { format!(", Season Intent [{}]", best_season_intent) } else { "".to_string() };

                // Formatting Plinko Fragments & Double Plinko for Operators
                let mut fragments_text = String::new();
                let mut prop_to_op: std::collections::HashMap<String, String> = std::collections::HashMap::new();

                for (k, v) in &plinko_map {
                    let combined_chunk = v.join(" | ");
                    
                    // 🌟 Double Plinko: 연산자(Operator), 메트릭(Metric Type) 판별 벡터 매칭 진행
                    let chunk_emb = self.get_embedding(combined_chunk.clone()).await.unwrap_or(vec![0.0; 384]);
                    
                    // 연산자 매칭
                    let mut best_op = "eq"; // Fallback
                    let mut best_op_score = 0.20; 
                    for i in 0..op_keys.len() {
                        let b_score = cosine_similarity(&chunk_emb, &op_bias_embs[i]);
                        let p_score = cosine_similarity(&chunk_emb, &op_prej_embs[i]);
                        let score = b_score - p_score;
                        if score > best_op_score {
                            best_op_score = score;
                            best_op = op_keys[i];
                        }
                    }

                    // 수치 유형(Metric Type) 매칭
                    let mut best_metric = "string"; // Fallback
                    let mut best_metric_score = 0.20;
                    for i in 0..metric_keys.len() {
                        let b_score = cosine_similarity(&chunk_emb, &metric_bias_embs[i]);
                        let p_score = cosine_similarity(&chunk_emb, &metric_prej_embs[i]);
                        let score = b_score - p_score;
                        if score > best_metric_score {
                            best_metric_score = score;
                            best_metric = metric_keys[i];
                        }
                    }

                    prop_to_op.insert(k.clone(), best_op.to_string());

                    // 🌟 전역(Global)에서 판별한 시간/계절 가이드를 각 청크에 동일하게 적용합니다.
                    fragments_text.push_str(&format!("Target Text: \"{}\" -> Vector Suggests: Property [{}], Operator [{}], Metric Type [{}]{}{}\n", combined_chunk, k, best_op, best_metric, time_guide, season_guide));
                }
                
                emit_term(&format!("  🎯 [PLINKO MAP (VECTOR GUIDE)] \n{}", fragments_text.trim()));

                // 🌟 [CRITICAL FIX] 시간/계절 판별 시 벡터 매칭의 환각을 완전히 차단하기 위해 LLM에게 직접 의도를 추출하도록 강제합니다.
                self.ensure_qwen3().await?;
                if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                // 🌟 시간 문맥(현재 시간)을 분리된 프롬프트에 제공하기 위해 미리 생성합니다.
                let now = chrono::Local::now();
                let time_context_for_intent = format!("Current Time: {}\nTimezone: {}\nLanguage: {}", now.format("%Y-%m-%dT%H:%M:%S"), now.format("%z"), language);

                let time_prompt = crate::parsing::extract_time_intent_prompt(&current_text, &time_context_for_intent);
                let season_prompt = crate::parsing::extract_season_intent_prompt(&current_text);
                
                let q3_for_temp = self.qwen3_generator.clone();
                let cancel_for_temp = cancel_token.clone();
                
                // 🌟 분리된 2개의 프롬프트를 단일 블로킹 스レッド 내에서 순차적으로 호출하여 오염을 막습니다.
                let (time_res_llm, season_res_llm) = tokio::task::spawn_blocking(move || -> anyhow::Result<(String, String)> {
                    let mut gen_guard = q3_for_temp.blocking_lock();
                    if let Some(gen) = gen_guard.as_mut() {
                        // 1. Time Intent 추출
                        let params_time = crate::openai_types::ChatCompletionParameters {
                            messages: vec![
                                crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { 
                                    content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(time_prompt),
                                    name: None,
                                })
                            ],
                            model: "qwen3".to_string(), max_tokens: Some(128), temperature: Some(0.0), top_p: Some(0.95),
                            ..Default::default()
                        };
                        let t_res = gen.generate(params_time, Some(cancel_for_temp.clone()), None, None).map_err(|e| anyhow::anyhow!("Qwen3 Time Inference failed: {}", e))?;

                        // 2. Season Intent 추출
                        let params_season = crate::openai_types::ChatCompletionParameters {
                            messages: vec![
                                crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { 
                                    content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(season_prompt),
                                    name: None,
                                })
                            ],
                            model: "qwen3".to_string(), max_tokens: Some(128), temperature: Some(0.0), top_p: Some(0.95),
                            ..Default::default()
                        };
                        let s_res = gen.generate(params_season, Some(cancel_for_temp), None, None).map_err(|e| anyhow::anyhow!("Qwen3 Season Inference failed: {}", e))?;

                        Ok((t_res, s_res))
                    } else {
                        Err(anyhow::anyhow!("Qwen3 Generator is missing"))
                    }
                }).await??;

                let time_json = crate::parsing::parse_json_from_llm(&time_res_llm);
                let season_json = crate::parsing::parse_json_from_llm(&season_res_llm);
                
                let mut llm_temporal_guide = String::new();
                if let Some(ti) = time_json.get("time_intent").and_then(|v| v.as_str()) {
                    if !ti.is_empty() && ti != "null" { llm_temporal_guide.push_str(&format!("Time Intent [{}] ", ti)); }
                }
                if let Some(si) = season_json.get("season_intent").and_then(|v| v.as_str()) {
                    if !si.is_empty() && si != "null" { llm_temporal_guide.push_str(&format!("Season Intent [{}]", si)); }
                }

                emit_term(&format!("  🤖 [LLM TIME INTENT]\n  {}", time_res_llm.trim()));
                emit_term(&format!("  🤖 [LLM SEASON INTENT]\n  {}", season_res_llm.trim()));

                // 🌟 LLM이 강제 선택한 의도를 바탕으로 최종 Deterministic Time Guide(달력 SQL 필터)를 획득합니다.
                let deterministic_guide_log = crate::parsing::get_deterministic_time_guide(&llm_temporal_guide, language);
                if !deterministic_guide_log.is_empty() {
                    emit_term(&format!("  ⏳ [DETERMINISTIC TIME GUIDE]\n  {}", deterministic_guide_log.replace("\n", "\n  ")));
                }

                // 5. LLM Normalization (Advanced Parser Mode with Vector Guide)
                if !fragments_text.is_empty() {
                    let now = chrono::Local::now();
                    let time_context = format!("Current Time: {}\nTimezone: {}\nLanguage: {}", now.format("%Y-%m-%dT%H:%M:%S"), now.format("%z"), language);

                    // 🌟 [CRITICAL FIX] 벡터 매칭 가이드와 LLM 시간 가이드를 병합하여 최종 조건 추출 프롬프트 호출
                    let combined_guide = format!("{}\n{}", fragments_text.trim(), llm_temporal_guide);
                    let prompt_final = crate::parsing::extract_numeric_conditions(&current_text, &seg_type, metrics_json, &combined_guide, &time_context, language);

                    let gen_arc = self.qwen3_generator.clone();
                    let cancel_clone = cancel_token.clone();
                    
                    let res_llm = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
                        let mut gen_guard = gen_arc.blocking_lock();
                        if let Some(gen) = gen_guard.as_mut() {
                            let params = crate::openai_types::ChatCompletionParameters {
                                messages: vec![
                                    crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage { 
                                        content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt_final),
                                        name: None,
                                    })
                                ],
                                model: "qwen3".to_string(), max_tokens: Some(256), temperature: Some(0.2), top_p: Some(0.95),
                                ..Default::default()
                            };
                            gen.generate(params, Some(cancel_clone), None, None).map_err(|e| anyhow::anyhow!("Qwen3 Inference failed: {}", e))
                        } else {
                            Err(anyhow::anyhow!("Qwen3 Generator is missing"))
                        }
                    }).await??;

                    emit_term(&format!("  🤖 [LLM RAW RESPONSE]\n{}", res_llm.trim()));
                    
                    let final_json = crate::parsing::parse_json_from_llm(&res_llm);
                    
                    emit_term(&format!("  ✅ [EXTRACTED DATA]\n{}", serde_json::to_string_pretty(&final_json).unwrap_or_default()));
                    
                    if let Some(obj) = seg.as_object_mut() {
                        if let Some(status_val) = final_json.get("status") {
                            obj.insert("status".to_string(), status_val.clone());
                        }
                        
                        // 🌟 LLM이 뽑아준 "값"과 Rust 메모리에 저장해둔 "연산자(operator)"를 여기서 최종 조립합니다.
                        if let Some(cond_val) = final_json.get("condition").and_then(|v| v.as_object()) {
                            let mut structured_cond = serde_json::Map::new();
                            for (k, val) in cond_val {
                                let op = prop_to_op.get(k).map(|s| s.as_str()).unwrap_or("eq");
                                structured_cond.insert(k.clone(), json!({
                                    "operator": op,
                                    "value": val.clone()
                                }));
                            }
                            obj.insert("condition".to_string(), json!(structured_cond.clone()));
                            
                            // 🌟 [추가] 벡터 매칭(연산자)과 LLM(추출 값)이 최종 병합된 결과 로그 출력
                            emit_term(&format!("  🚀 [FINAL MERGED CONDITION]\n{}", serde_json::to_string_pretty(&structured_cond).unwrap_or_default()));
                        } else {
                            obj.insert("condition".to_string(), json!({}));
                            emit_term("  🚀 [FINAL MERGED CONDITION]\n{}");
                        }
                    }
                } else {
                    if let Some(obj) = seg.as_object_mut() {
                        obj.insert("condition".to_string(), json!({}));
                    }
                }

                crate::models::qwen::generate::wait_for_global_io().await;
                
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

        // 🌟 [CRITICAL FIX] 분할된 컨텍스트(Segment)들의 추출 결과를 Rust 메모리에서 하나의 마스터 객체로 완벽히 취합(Merge)합니다.
        // 이를 통해 앞단에서 조각난 맥락(예: '구매 전환율이' + '1% 미만인')이 하나의 강력한 AND 조건으로 결합되어 DB 검색 시 정보 유실을 원천 차단합니다.
        if let Some(ctx_arr) = segments.get_mut("context").and_then(|v| v.as_array_mut()) {
            if ctx_arr.len() > 1 {
                emit_term("[STAGE-3] Merging all fragmented conditions into a single master context...");
                let mut master_condition = serde_json::Map::new();
                let mut master_text = String::new();
                let mut master_status = json!("");
                let mut master_type = String::new();

                for seg in ctx_arr.iter() {
                    // 첫 번째 유효한 도메인 타입을 마스터로 고정
                    if master_type.is_empty() {
                        master_type = seg.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    }
                    
                    // 텍스트는 띄어쓰기로 이어 붙임
                    let text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    if !text.is_empty() {
                        if !master_text.is_empty() { master_text.push_str(" "); }
                        master_text.push_str(text);
                    }
                    
                    // 상태값 병합 (첫 번째 유효값 우선)
                    if let Some(status) = seg.get("status") {
                        let s_str = status.as_str().unwrap_or("");
                        if !s_str.is_empty() && s_str != "null" && master_status.as_str().unwrap_or("").is_empty() {
                            master_status = status.clone();
                        }
                    }
                    
                    // 추출된 조건(Condition) 객체들 병합
                    if let Some(cond) = seg.get("condition").and_then(|v| v.as_object()) {
                        for (k, v) in cond {
                            // 값(value)이 비어있는 쓰레기 데이터는 무시하고, 유효한 값만 병합
                            let is_empty = match v.get("value") {
                                Some(serde_json::Value::String(s)) => s.trim().is_empty() || s == "null",
                                Some(serde_json::Value::Null) => true,
                                Some(serde_json::Value::Object(o)) => {
                                    o.get("value").and_then(|val| val.as_str()).map_or(false, |s| s.trim().is_empty() || s == "null")
                                },
                                _ => false,
                            };
                            
                            if !is_empty {
                                master_condition.insert(k.clone(), v.clone());
                            }
                        }
                    }
                }
                
                if master_type.is_empty() { master_type = "goods".to_string(); }

                // 덮어쓰기: 모든 세그먼트를 1개의 마스터 객체로 압축
                let master_ctx = json!({
                    "type": master_type,
                    "text": master_text,
                    "status": master_status,
                    "condition": master_condition
                });
                
                *ctx_arr = vec![master_ctx];
                
                emit_term(&format!("  ✅ [MASTER MERGED CONTEXT]\n{}", serde_json::to_string_pretty(&ctx_arr[0]).unwrap_or_default()));
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
                    model: "qwen3".to_string(), max_tokens: Some(256), temperature: Some(0.0), top_p: Some(0.95),
                    ..Default::default()
                };
                gen.generate(params, Some(cancel_clone), None, None).map_err(|e| anyhow::anyhow!("Qwen3 Inference failed: {}", e))
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
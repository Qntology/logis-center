use crate::utils;
use anyhow::anyhow;
use crate::models::qwen3vl::generate::Qwen3VLGenerateModel;
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelSize {
    Small, // 0.6B for Ingestion
    Large, // 2B-VL for Inference
}

#[derive(Clone)]
pub struct LogisModel {
    pub generator: Arc<TokioMutex<Option<Qwen3VLGenerateModel>>>, // Primary Active Slot (GPU)
    pub small_hibernation: Arc<TokioMutex<Option<Qwen3VLGenerateModel>>>, // 0.6B RAM Slot
    pub large_hibernation: Arc<TokioMutex<Option<Qwen3VLGenerateModel>>>, // 2B RAM Slot
    pub embedding_model: Arc<TokioMutex<Option<EmbeddingModel>>>,
    
    pub is_cpu_mode: bool, 
    pub dual_mode_enabled: bool,
    
    // Config for Lazy Reloading
    small_model_path: String,
    large_model_path: String,
    embedding_path: std::path::PathBuf,
    device_config: utils::DeviceConfig,
    max_tokens_limit: u32,
    dtype: Option<DType>, 
    current_size: Arc<TokioMutex<Option<ModelSize>>>,
}

impl LogisModel {
    pub async fn unload_generator(&self) {
        // Clear everything
        let mut gen = self.generator.lock().await;
        *gen = None;
        let mut s_hib = self.small_hibernation.lock().await;
        *s_hib = None;
        let mut l_hib = self.large_hibernation.lock().await;
        *l_hib = None;
        
        let mut size = self.current_size.lock().await;
        *size = None;
        println!("[MODEL] All generators (Active & Hibernated) destroyed.");
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

    /// [CLEANUP] Adaptive resource management based on system stress
    pub async fn deep_purge_resources(&self) {
        if !self.is_cpu_mode {
            let dev = self.device_config.device.clone();
            let _ = tokio::task::spawn_blocking(move || {
                // Ignore sync errors if context is already gone
                if dev.is_cuda() { let _ = dev.synchronize(); }
            }).await;
        }

        // 1. Clear Active Slots (Forced wait for lock to ensure reclamation)
        {
            let mut gen = self.generator.lock().await;
            *gen = None;
        }
        {
            let mut s_hib = self.small_hibernation.lock().await;
            *s_hib = None;
        }
        {
            let mut l_hib = self.large_hibernation.lock().await;
            *l_hib = None;
        }
        
        // 2. [CRITICAL-FIX] OS RAM/VRAM Flush
        #[cfg(target_os = "windows")]
        unsafe {
            use windows_sys::Win32::System::Threading::*;
            use windows_sys::Win32::System::Memory::*;
            let current_process = GetCurrentProcess();
            // 강제로 Working Set을 비워 OS가 VRAM/RAM을 즉시 수거하게 함
            let _ = SetProcessWorkingSetSizeEx(current_process, usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
        }
        
        println!("[MEMORY] Deep Purge Complete. VRAM/RAM released to OS.");
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
            if let Ok(nvml) = nvml_wrapper::Nvml::init() {
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

            // [ADAPTIVE-LOGIC] Analyze Trend
            if current_free > last_free + (50 * 1024 * 1024) { // Increased by > 50MB
                increasing_ticks += 1;
                if increasing_ticks > 0 {
                    println!("[VRAM-WATCH] Reclaiming... ({:.2} GB -> {:.2} GB)", last_free as f64/1e9, current_free as f64/1e9);
                }
            } else {
                increasing_ticks = 0;
            }

            // [ACTIVE-FLUSH] If stuck for > 1.5s, trigger OS RAM cleanup
            if start.elapsed().as_secs_f32() > 1.5 && !has_flushed_ram && current_free < target_bytes {
                println!("[VRAM-WATCH] Triggering OS Working Set Trim...");
                #[cfg(target_os = "windows")]
                unsafe {
                    use windows_sys::Win32::System::Threading::GetCurrentProcess;
                    use windows_sys::Win32::System::Memory::SetProcessWorkingSetSizeEx;
                    use windows_sys::Win32::System::Memory::QUOTA_LIMITS_HARDWS_MIN_DISABLE;
                    use windows_sys::Win32::System::Memory::QUOTA_LIMITS_HARDWS_MAX_DISABLE;
                    let process = GetCurrentProcess();
                    let _ = SetProcessWorkingSetSizeEx(process, usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
                }
                has_flushed_ram = true;
                // Give it a moment to reflect
                tokio::time::sleep(Duration::from_millis(500)).await;
                continue;
            }

            // [TIMEOUT-HANDLER]
            if start.elapsed().as_secs() > timeout_sec {
                // If memory is still actively freeing up, extend timeout dynamically
                if increasing_ticks > 0 {
                    println!("[VRAM-WATCH] Timeout reached but memory is freeing up. Extending wait...");
                    increasing_ticks = 0; // Reset to avoid infinite loop
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
    pub async fn save_kv_snapshot(&self, task_id: &str) -> anyhow::Result<String> {
        let generator_arc = self.generator.clone();
        let task_id_str = task_id.to_string();
        
        tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = generator_arc.blocking_lock();
            if let Some(gen) = gen_guard.as_mut() {
                let path = crate::utils::paths::get_kv_dir(None).join(task_id_str);
                println!("[SSD-BRIDGE] Saving KV snapshot to {:?}", path);
                gen.save_kv_to_disk(&path)?;
                Ok(path.to_string_lossy().to_string())
            } else {
                Err(anyhow::anyhow!("No active generator to save snapshot from"))
            }
        }).await?
    }

    pub async fn load_kv_snapshot(&self, task_id: &str) -> anyhow::Result<()> {
        let generator_arc = self.generator.clone();
        let task_id_str = task_id.to_string();

        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let mut gen_guard = generator_arc.blocking_lock();
            if let Some(gen) = gen_guard.as_mut() {
                let path = crate::utils::paths::get_kv_dir(None).join(task_id_str.clone());
                if path.exists() {
                    println!("[SSD-BRIDGE] Loading KV snapshot from {:?}", path);
                    gen.load_kv_from_disk(&path)?;
                    Ok(())
                } else {
                    println!("[SSD-BRIDGE] No snapshot found for {}", task_id_str);
                    Ok(())
                }
            } else {
                Err(anyhow::anyhow!("No active generator to load snapshot into"))
            }
        }).await?
    }

    /// [NEW] Secure VRAM/RAM Transition Logic (Isolation Protocol)
    pub async fn secure_vram_relay(&self, target_size: ModelSize, task_id: Option<&str>, cancel_token: Option<Arc<AtomicBool>>, is_baking: bool) -> anyhow::Result<()> {
        let start_time = Instant::now();
        
        // 1. [CLEANUP] 강력한 리소스 해제 및 OS 반환
        println!("[RELAY] Performing Deep Purge before loading {:?} (Baking: {})...", target_size, is_baking);
        self.deep_purge_resources().await;
        
        if !self.is_cpu_mode {
            // VRAM이 실제로 비워질 때까지 대기 (대기 시간 0.5s -> 1.0s 상향)
            tokio::time::sleep(Duration::from_millis(1000)).await;
                        // 타겟 VRAM 상향 (2000MB -> 2500MB)
                        self.wait_for_vram_settle(2500, 7, cancel_token.clone()).await?;
                    }
            
                    // 2. [LOAD] 새 모델 로드 (이제 VRAM이 최대치로 확보된 상태)
                    // [OPTIMIZATION] If transitioning to Large for a Relay (task_id present), skip Vision module
                    let text_only = target_size == ModelSize::Large && task_id.is_some() && !is_baking;
                    
                    // 로드 직전에 한 번 더 리소스 정리 시도 (안전 장치)
                    #[cfg(target_os = "windows")]
                    unsafe {
                        use windows_sys::Win32::System::Threading::GetCurrentProcess;
                        use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
                        let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
                    }
            
                    self.ensure_generator_ext(target_size, text_only, is_baking).await?;

        // 4. [RESTORE] 디스크 스냅샷 로드
        if let Some(tid) = task_id {
            self.load_kv_snapshot(tid).await?;
        }

        println!("[RELAY] Transition to {:?} complete in {:.2}s", target_size, start_time.elapsed().as_secs_f32());
        Ok(())
    }

    // --- [NEW] Base Context Baking (One-time Heavy Lifting) ---
    pub async fn ingest_pug_to_ssd(&self, task_id: &str, pug_content: &str, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<()> {
        let base_session = format!("{}_base", task_id);
        
        // 1. Load Small Model Isolated
        self.secure_vram_relay(ModelSize::Small, None, cancel_token.clone(), true).await?;

        // 2. Ingest PUG content
        {
            let gen_clone = self.generator.clone();
            let prompt = format!("{}\n\n[SYSTEM] Analyze the document structure.", pug_content);
            let token_clone = cancel_token.clone();
            
            // We use prefill_only via a manual chat construct or direct access if possible
            // Reusing chat_params_with_spinner for convenience but with empty generation
            
            let _ = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
                let mut gen_guard = gen_clone.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    // Just prefill, no generation needed for base context
                    gen.prefill_chunk(prompt, token_clone, None)?;
                }
                Ok(())
            }).await??;
        }

        // 3. Save Base Snapshot
        self.save_kv_snapshot(&base_session).await?;
        
        // 4. Unload immediately to free VRAM for 2B
        self.unload_generator().await;
        
        Ok(())
    }

    // --- [NEW] 2B Continuous Inference Helper ---
    pub async fn ensure_large_with_base(&self, task_id: &str, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<()> {
        let base_session = format!("{}_base", task_id);
        
        // Only load if not already loaded or if current loaded session is different?
        // secure_vram_relay checks size but not session content.
        // We force a relay if we are not Large. If we are Large, we assume we might need to reset or just continue.
        // For safety in this new flow, we can check if we need to load.
        
        {
            let current_size = *self.current_size.lock().await;
            if current_size == Some(ModelSize::Large) {
                // Already Large. Assuming context is preserved or we manage it.
                // But if we just switched from Small, we need to load base.
                // The safest is to rely on secure_vram_relay's logic:
                // If we pass the base_session as task_id, it will load that snapshot.
            }
        }

        self.secure_vram_relay(ModelSize::Large, Some(&base_session), cancel_token, false).await
    }

    async fn load_generator_internal(&self, path: &str, shared_config_path: Option<&str>, force_text_only: bool) -> anyhow::Result<Qwen3VLGenerateModel> {
        println!("[MODEL] Loading Generator from {} (Text-Only: {})...", path, force_text_only);
        let dev = self.device_config.device.clone();
        let dev_id = self.device_config.gpu_id;

        let dtype = if self.device_config.is_cpu { Some(DType::F32) } else { Some(DType::BF16) };
        let limit = self.max_tokens_limit;
        let path_clone = path.to_string();
        let shared_path = shared_config_path.map(|s| s.to_string());

        let generator = tokio::task::spawn_blocking(move || {
            // [CRITICAL] Use init_with_config to force shared settings (Config + Tokenizer)
            Qwen3VLGenerateModel::init_with_config(
                &path_clone, 
                shared_path.as_deref(), // Tokenizer path
                shared_path.as_deref(), // Config path
                Some(&dev), dev_id, Some(&dev), dev_id, dtype, Some(limit as usize),
                force_text_only,
                false
            )
        }).await??;
        
        Ok(generator)
    }

    pub async fn ensure_generator(&self, size: ModelSize) -> anyhow::Result<()> {
        self.ensure_generator_ext(size, false, false).await
    }

    pub async fn ensure_generator_ext(&self, size: ModelSize, force_text_only: bool, baking_only: bool) -> anyhow::Result<()> {
        let mut current_size_guard = self.current_size.lock().await;
        let mut gen_guard = self.generator.lock().await;
        let mut small_slot = self.small_hibernation.lock().await;
        let mut large_slot = self.large_hibernation.lock().await;

        if *current_size_guard == Some(size) && gen_guard.is_some() && !baking_only {
            return Ok(());
        }

        println!("[MODEL] Activating engine for size: {:?} (Text-Only: {}, Baking: {})...", size, force_text_only, baking_only);
        // ... (rest of the switching logic remains similar but uses the new loading)

        // 1. [SWITCH] If requested model is already in one of the slots, just move it to main
        let found_in_slot = match size {
            ModelSize::Small => {
                if let Some(m) = small_slot.take() {
                    if let Some(old_m) = gen_guard.take() {
                        if *current_size_guard == Some(ModelSize::Large) { *large_slot = Some(old_m); }
                        else if *current_size_guard == Some(ModelSize::Small) { *small_slot = Some(old_m); }
                    }
                    *gen_guard = Some(m);
                    *current_size_guard = Some(ModelSize::Small);
                    true
                } else { false }
            },
            ModelSize::Large => {
                if let Some(m) = large_slot.take() {
                    if let Some(old_m) = gen_guard.take() {
                        if *current_size_guard == Some(ModelSize::Small) { *small_slot = Some(old_m); }
                        else if *current_size_guard == Some(ModelSize::Large) { *large_slot = Some(old_m); }
                    }
                    *gen_guard = Some(m);
                    *current_size_guard = Some(ModelSize::Large);
                    true
                } else { false }
            }
        };

        if found_in_slot {
            println!("[MODEL] Switched to cached {:?} engine.", size);
            return Ok(());
        }

        // 2. [LOAD] Load from disk if not found in any slot
        println!("[LOAD] Fresh loading {:?} from disk...", size);
        let path = if size == ModelSize::Small { &self.small_model_path } else { &self.large_model_path };
        
        // [HYBRID-FIX] Use Large model's tokenizer for Small model to ensure token ID compatibility
        let tokenizer_path = if size == ModelSize::Small { Some(self.large_model_path.as_str()) } else { None };
        let config_path: Option<&str> = None; // Always use model's own config
        
        let mut target_device = self.device_config.device.clone();
        
        // [OOM-SAFETY] Small (0.6B) can stay on CPU if VRAM is tight to keep Large (2B) on GPU.
        if size == ModelSize::Small && target_device.is_cuda() {
            if let Ok(nvml_inst) = nvml_wrapper::Nvml::init() {
                if let Ok(dev) = nvml_inst.device_by_index(self.device_config.gpu_id as u32) {
                    if let Ok(mem) = dev.memory_info() {
                        if mem.free < 3_000_000_000 {
                            println!("[MODEL-CONFIG] Tight VRAM. Loading Small (0.6B) on CPU.");
                            target_device = Device::Cpu;
                        }
                    }
                }
            }
        }

        let dev_id = self.device_config.gpu_id;
        let dtype = if target_device.is_cpu() { Some(DType::F32) } else { Some(DType::BF16) };
        let limit = self.max_tokens_limit;
        let path_clone = path.to_string();
        let tok_path_clone = tokenizer_path.map(|s| s.to_string());
        let cfg_path_clone = config_path.map(|s| s.to_string());

        let gen = tokio::task::spawn_blocking(move || {
            Qwen3VLGenerateModel::init_with_config(
                &path_clone, 
                tok_path_clone.as_deref(), 
                cfg_path_clone.as_deref(), 
                Some(&target_device), dev_id, Some(&target_device), dev_id, dtype, Some(limit as usize),
                force_text_only,
                baking_only // [PASS-NEW]
            )
        }).await??;

        // Move current main to slot
        if let Some(old_m) = gen_guard.take() {
            if let Some(old_size) = *current_size_guard {
                match old_size {
                    ModelSize::Small => *small_slot = Some(old_m),
                    ModelSize::Large => *large_slot = Some(old_m),
                }
            }
        }

        *gen_guard = Some(gen);
        *current_size_guard = Some(size);
        
        Ok(())
    }

    pub async fn ensure_embedding(&self) -> anyhow::Result<()> {
        let current_size = { *self.current_size.lock().await };
        
        // [STRATEGY] High-priority exclusion logic
        match current_size {
            Some(ModelSize::Large) => {
                // If Large is active, Embedding must stay on CPU to avoid OOM
                println!("[MODEL] Large model active. Forcing Embedding to CPU to prevent swapping.");
            },
            Some(ModelSize::Small) => {
                // Small and Embedding can coexist. 
                println!("[MODEL] Small model active. Embedding and 0.6B will coexist.");
            },
            None => {
                // No generator active, safe to clean up any leftovers
                self.unload_generator().await;
            }
        }

        let mut emb_guard = self.embedding_model.lock().await;
        if emb_guard.is_none() {
            let self_clone = self.embedding_path.clone();
            
            // Determine target device: CPU if Large is active, else use default GPU
            let target_device = if current_size == Some(ModelSize::Large) { 
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

    pub async fn new(device_preference: Option<&str>) -> anyhow::Result<Self> {
        println!("[MODEL-00] Initializing LogisModel (Preference: {:?})", device_preference);

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

        let small_model_path = normalize_path(base_path.join("Qwen3-0.6B-Instruct-gguf"));
        let large_model_path = normalize_path(base_path.join("Qwen3-VL-2B-Hybrid-gguf"));
        let embedding_path = base_path.join("embeddinggemma-300m");

        let max_tokens_limit = 65536; 

        Ok(Self {
            generator: Arc::new(TokioMutex::new(None)),
            small_hibernation: Arc::new(TokioMutex::new(None)),
            large_hibernation: Arc::new(TokioMutex::new(None)),
            embedding_model: Arc::new(TokioMutex::new(None)),
            is_cpu_mode: config.is_cpu,
            dual_mode_enabled: true, 
            small_model_path,
            large_model_path,
            embedding_path,
            device_config: config.clone(),
            max_tokens_limit: max_tokens_limit as u32,
            dtype: None, 
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
        // [FIX] Do NOT force reload. Use current generator (Large should be already loaded via relay)
        {
            let gen_guard = self.generator.lock().await;
            if gen_guard.is_none() {
                drop(gen_guard);
                self.ensure_generator(ModelSize::Large).await?;
            }
        }

        if let Ok(img) = image::open(&image_path) {
            let dynamic_image = image::DynamicImage::ImageRgb8(img.to_rgb8());
            let prompt = get_image_extraction_prompt("kr", &language, "tracking", "");
            
            // [IMPORTANT] Pass task_id as session_id to continue from the baked relay snapshot
            let result_str = self.chat_with_image_spinner(
                prompt, 
                Some(dynamic_image), 
                app_handle, 
                "extraction-progress", 
                json!({ "category": "Vision Analysis", "summary": "Analyzing image with baked context..." }), 
                1024, 
                cancel_token.clone(), 
                Some(task_id.clone()) 
            ).await?;

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

                let mut final_data = extracted_data.clone();
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

    pub fn is_cpu(&self) -> bool {
        self.is_cpu_mode
    }

    pub async fn chat(&self, system: &str, user_input: &str, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>) -> anyhow::Result<String> {
        // [FIX] Use current generator if available, else default to Large
        {
            let gen_guard = self.generator.lock().await;
            if gen_guard.is_none() {
                drop(gen_guard);
                self.ensure_generator(ModelSize::Large).await?;
            }
        }
        
        let self_clone = self.generator.clone();
        let system_text = system.to_string();
        let user_text = user_input.to_string();
        let max_tok = self.max_tokens_limit;
        
        println!("[MODEL-CHAT] Sending Chat Request...");
        
        tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = self_clone.blocking_lock();
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
                model: "qwen3vl".to_string(),
                max_tokens: Some(max_tok),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            let response = gen.generate(params, cancel_token, session_id).map_err(|e| anyhow!("Inference failed: {}", e))?;
            println!("[MODEL-CHAT] Raw Response: {}", response);
            Ok(response)
        }).await?
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
        session_id: Option<String>
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
            model: "qwen3vl".to_string(),
            max_tokens: Some(max_tokens as u32),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };

        self.chat_params_with_spinner(params, app_handle, event_name, base_payload, cancel_token, session_id).await
    }

    pub async fn chat_params_with_spinner(
        &self, 
        params: ChatCompletionParameters,
        app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>
    ) -> anyhow::Result<String> {
        // [FIX] Do not force reload Large. Use current if available, else default to Large.
        {
            let gen_guard = self.generator.lock().await;
            if gen_guard.is_none() {
                drop(gen_guard);
                self.ensure_generator(ModelSize::Large).await?;
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

        let self_clone = self.generator.clone();
        
        let task = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = self_clone.blocking_lock();
            let gen = gen_guard.as_mut().ok_or_else(|| anyhow!("Generator is unloaded"))?;
            gen.generate(params, cancel_token, session_id).map_err(|e| anyhow!("Inference failed: {}", e))
        });

        task.await.map_err(|e| anyhow!("Task join error: {}", e))?
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
        session_id: Option<String>
    ) -> anyhow::Result<String> {
        // Ensure generator is loaded
        self.ensure_generator(ModelSize::Large).await?;

        // [FIX] Removed redundant emit. Only log the progress if needed.
        // let _ = app_handle.emit(event_name, base_payload);

        let self_clone = self.generator.clone();
        
        let task = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = self_clone.blocking_lock();
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
                model: "qwen3vl".to_string(),
                max_tokens: Some(max_tokens as u32),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            gen.generate(params, cancel_token, session_id).map_err(|e| anyhow!("Inference failed: {}", e))
        });

        task.await.map_err(|e| anyhow!("Task join error: {}", e))?
    }

    async fn run_inference_text(&self, prompt: String, image: Option<DynamicImage>, cancel_token: Option<Arc<AtomicBool>>, session_id: Option<String>) -> anyhow::Result<String> {
        self.ensure_generator(ModelSize::Large).await?;
        
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
            model: "qwen3vl".to_string(),
            max_tokens: Some(self.max_tokens_limit),
            temperature: Some(0.1),
            top_p: Some(0.9),
            ..Default::default()
        };
        
        gen.generate(params, cancel_token, session_id).map_err(|e| anyhow!("Inference failed: {}", e))
    }

    pub async fn run_inference_with_spinner(
        &self, 
        prompt: String, 
        image: Option<DynamicImage>, 
        _app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>
    ) -> anyhow::Result<String> {
        // Ensure generator is loaded
        self.ensure_generator(ModelSize::Large).await?;

        // [FIX] Inject task_id from session_id if it's a task reference
        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        // [FIX] Removed redundant emit. Only log the progress.
        // let _ = app_handle.emit(event_name, base_payload);

        let generator_arc = self.generator.clone();
        let max_tok = self.max_tokens_limit;
        
        // Spawn the heavy task using Tokio directly for standard behavior
        let task = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut gen_guard = generator_arc.blocking_lock();
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
                model: "qwen3vl".to_string(),
                max_tokens: Some(max_tok),
                temperature: Some(0.1),
                top_p: Some(0.9),
                ..Default::default()
            };
            
            gen.generate(params, cancel_token, session_id).map_err(|e| anyhow!("Inference failed: {}", e))
        });
        
        task.await.map_err(|e| anyhow!("Task join error: {}", e))?
    }

    pub async fn process_image_full(&self, image_path: String, app_handle: &tauri::AppHandle, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<Value> {
        println!("[PROCESS] General image analysis for: {}", image_path);
        
        let full_img_raw = image::open(&image_path)?;
        let full_img_raw = DynamicImage::ImageRgb8(full_img_raw.to_rgb8());
        
        // Smart Resize for VRAM stability
        let master_img = full_img_raw.resize(1024, u32::MAX, image::imageops::FilterType::Triangle);
        
        let prompt = get_image_extraction_prompt("kr", "korean", "tracking", "");
        
        let response = self.run_inference_with_spinner(
            prompt, 
            Some(master_img), 
            app_handle, 
            "extraction-progress", 
            json!({ "category": "Processing", "summary": "Analyzing document content..." }),
            cancel_token,
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

    pub async fn parse_query_structured(&self, query: String, language: &str) -> anyhow::Result<Value> {
        let current_time = chrono::Utc::now().to_rfc3339();
        
        // Stage 1: Segment query (para2graph) - Using persistent session for schema caching
        let prompt1 = crate::parsing::para2graph(language);
        let res1 = self.chat("", &format!("{}\n\nQuery: {}", prompt1, query), None, Some("system_search_p2g".to_string())).await?;
        let segments = crate::parsing::parse_json_from_llm(&res1);
        
        // Stage 2: Extract conditions for each segment (graph2contexts) in ONE BATCH
        let mut final_contexts = Vec::new();
        if let Some(ctx_arr) = segments.get("context").and_then(|v: &Value| v.as_array()) {
            // Combine all segments into one batch request
            let mut combined_segments = String::new();
            for (idx, seg) in ctx_arr.iter().enumerate() {
                let seg_text = seg.get("text").and_then(|v: &Value| v.as_str()).unwrap_or("");
                combined_segments.push_str(&format!("Segment #{}: {}\n", idx + 1, seg_text));
            }

            if !combined_segments.is_empty() {
                let prompt2 = crate::parsing::graph2contexts(&current_time);
                // Using persistent session for schema caching
                let res2 = self.chat("", &format!("{}\n\nInput Segments:\n{}", prompt2, combined_segments), None, Some("system_search_g2c".to_string())).await?;
                let mut batch_info = crate::parsing::parse_json_from_llm(&res2);
                
                // Process results and ensure type parity
                if let Some(res_arr) = batch_info.get_mut("context").and_then(|v: &mut Value| v.as_array_mut()) {
                    for (i, item) in res_arr.iter_mut().enumerate() {
                        // Match with original segment types if LLM lost them in batch
                        if let Some(original_seg) = ctx_arr.get(i) {
                            if item.get("type").is_none() || item.get("type").and_then(|v: &Value| v.as_str()) == Some("") {
                                if let Some(item_obj) = item.as_object_mut() {
                                    item_obj.insert("type".to_string(), original_seg.get("type").cloned().unwrap_or(json!("")));
                                }
                            }
                        }
                    }
                    final_contexts.extend(res_arr.clone());
                }
            }
        }
        
        Ok(json!({ "context": final_contexts }))
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
            let step_result = self.run_inference_text(prompt, None, cancel_token.clone(), None).await?;
            
            let short_res = if step_result.len() > 200 { &step_result[..200] } else { &step_result };
            status_history.push_str(&format!("> {}...\n\n", short_res.replace("\n", " ")));
            // [LOG-ONLY]
            crate::scheduler::log_task_progress(app_handle, "research", &json!({ "text": status_history }));
        }

        // 3. Final Report
        status_history.push_str("### 📊 Final Research Report\n\n");
        let final_prompt = format!("CONTEXT: {}\nQUERY: {}\n\nBased on the above steps, generate a comprehensive final trade intelligence report.", context_data, query);
        
        let report = self.run_inference_text(final_prompt, None, cancel_token, None).await?;
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
4. Provide a text summary in {LANGUAGE}, masking the address to District-level and up. Do not mention masking.

[OUTPUT FORMAT]
Return valid JSON only. No explanation.
{
    "tracking_number": "string",
    "recipient_match": boolean,
    "barcodes": ["string"],
    "text": "string"
}"###;
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
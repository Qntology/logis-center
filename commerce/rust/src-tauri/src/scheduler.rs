use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use crate::store::{VectorStore, Task};
use crate::logic;
use crate::utils;
use crate::parsing::{self, PugMode};
use crate::model::LogisModel;
use serde_json::{Value, json};
use anyhow::Result;
use tauri::Emitter;
use std::sync::atomic::{AtomicBool, Ordering};
use std::fs;
use std::path::PathBuf;

// --- [MEMORY OPTIMIZATION] Task Data Manager (RAII) ---
// Handles offloading of large text data to disk to prevent RAM/VRAM bloating.
// Automatically cleans up files when the task scope ends.
struct TaskDataManager {
    task_id: String,
    created_files: Vec<PathBuf>,
    app_handle: Option<tauri::AppHandle>,
}

impl TaskDataManager {
    fn new(task_id: &str, app_handle: Option<tauri::AppHandle>) -> Self {
        Self {
            task_id: task_id.to_string(),
            created_files: Vec::new(),
            app_handle,
        }
    }

    fn offload(&mut self, content: &str, suffix: &str) -> Result<PathBuf> {
        let dir = utils::paths::get_task_specific_dir(self.app_handle.as_ref(), &self.task_id);
        
        // [FIX] Use fixed filenames for intermediate steps to support resumption
        let filename = format!("{}.txt", suffix);
        let path = dir.join(filename);
        fs::write(&path, content)?;
        self.created_files.push(path.clone());
        Ok(path)
    }

    fn load(&self, path: &std::path::Path) -> Result<String> {
        Ok(fs::read_to_string(path)?)
    }

    fn get_path(&self, suffix: &str) -> PathBuf {
        let dir = utils::paths::get_task_specific_dir(self.app_handle.as_ref(), &self.task_id);
        dir.join(format!("{}.txt", suffix))
    }
}

impl Drop for TaskDataManager {
    fn drop(&mut self) {
        // [DEBUG] 디버깅 및 Resume을 위해 생성된 파일을 보존합니다.
        println!("[Cleanup] TaskDataManager dropping. Keeping {} files for debugging: {}", self.created_files.len(), self.task_id);
        for path in &self.created_files {
            if path.exists() {
                println!("[DEBUG] File available: {:?}", path);
            }
        }
        // KV 캐시는 재사용을 위해 디스크에 유지합니다.
    }
}

// Helper to chunk text, strictly respecting Pug line boundaries (\n)
fn chunk_text(text: &str, target_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();

    while start < text.len() {
        let mut end = (start + target_size).min(text.len());

        // [BACKTRACKING] If mid-line, move back until we find a newline
        if end < text.len() {
            let mut temp_end = end;
            while temp_end > start && bytes[temp_end] != b'\n' {
                temp_end -= 1;
            }
            
            // If found a newline, use it as the end
            if temp_end > start {
                end = temp_end + 1; // Include the \n
            } else {
                // No newline found in the whole chunk? Force char boundary to avoid hang
                while end < text.len() && !text.is_char_boundary(end) {
                    end += 1;
                }
            }
        } else {
            // Reached the end of string
            end = text.len();
        }

        let slice = &text[start..end];
        if !slice.trim().is_empty() {
            chunks.push(slice.to_string());
        }
        start = end;
    }
    chunks
}

// Deep merge for JSON objects
fn merge_json_results(target: &mut Value, source: &Value) {
    if let (Some(target_obj), Some(source_obj)) = (target.as_object_mut(), source.as_object()) {
        for (k, v) in source_obj {
            // If value is null or empty string/array, ignore
            if v.is_null() { continue; }
            if let Some(s) = v.as_str() { if s.is_empty() { continue; } }
            if let Some(a) = v.as_array() { if a.is_empty() { continue; } }
            
            // If target doesn't have it or target is empty, overwrite
            let should_update = match target_obj.get(k) {
                None => true,
                Some(tv) => {
                    tv.is_null() || 
                    (tv.is_string() && tv.as_str() == Some("")) ||
                    (tv.is_array() && tv.as_array().unwrap().is_empty())
                }
            };

            if should_update {
                target_obj.insert(k.clone(), v.clone());
            } else if let Some(target_inner) = target_obj.get_mut(k) {
                // If both are objects, recurse
                if target_inner.is_object() && v.is_object() {
                    merge_json_results(target_inner, v);
                }
                // Lists? Simply append for now (might duplicate, but safe for search)
                else if let (Some(ta), Some(sa)) = (target_inner.as_array_mut(), v.as_array()) {
                    ta.extend(sa.clone());
                }
            }
        }
    }
}

use tokio::sync::Notify;
use once_cell::sync::Lazy;

// [UI-SYNC] Instant notification system to wake up the worker
static UI_READY_SIGNAL: Lazy<Notify> = Lazy::new(|| Notify::new());
static TASK_QUEUED_SIGNAL: Lazy<Notify> = Lazy::new(|| Notify::new());
static UI_READY_FLAG: AtomicBool = AtomicBool::new(false);

pub fn mark_ui_ready() {
    UI_READY_FLAG.store(true, Ordering::SeqCst);
    UI_READY_SIGNAL.notify_waiters(); // Wake up any sleeping tasks instantly
    println!("[Scheduler] UI signaled ready. Background worker woke up.");
}

pub fn notify_new_task() {
    TASK_QUEUED_SIGNAL.notify_waiters();
}

pub async fn start_background_worker(
    store: Arc<Mutex<Option<VectorStore>>>,
    model: Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: Arc<AtomicBool>,
    app_handle: tauri::AppHandle,
) {
    println!("[Scheduler] Background worker waiting for UI Ready signal...");
    
    clear_all_temp_data(Some(&app_handle));

    // [NEW] 앱 시작 시 즉시 좀비 작업 정리 (잠금 획득 시도)
    {
        let store_clone = store.clone();
        tauri::async_runtime::spawn(async move {
            for _ in 0..10 { // 최대 5초간 대기하며 시도
                if let Ok(guard) = store_clone.try_lock() {
                    if let Some(db) = guard.as_ref() {
                        let _ = db.cleanup_zombie_tasks().await;
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        });
    }
    
    tokio::spawn(async move {
        // [EVENT-DRIVEN-WAIT] Zero CPU usage, zero delay. 
        // Wakes up exactly when mark_ui_ready is called.
        if !UI_READY_FLAG.load(Ordering::SeqCst) {
            UI_READY_SIGNAL.notified().await;
        }
        
        let mut delay_secs = 1;
        let mut current_device_pref: Option<String> = None;
        
        loop {
            // [CRITICAL] Global Stop Signal Check
            if crate::utils::is_extraction_stopped() {
                tokio::time::sleep(Duration::from_millis(500)).await;
                continue;
            }

            let mut pending_tasks = Vec::new();
            {
                let store_opt = store.lock().await;
                if let Some(db) = store_opt.as_ref() {
                    match db.get_pending_tasks(5).await {
                        Ok(tasks) => pending_tasks = tasks,
                        Err(e) => println!("[Scheduler] Failed to fetch tasks: {:?}", e),
                    }
                }
            }

            if pending_tasks.is_empty() {
                // [EVENT-DRIVEN] Wait for timeout OR new task signal
                tokio::select! {
                    _ = sleep(Duration::from_secs(delay_secs)) => {
                        delay_secs = (delay_secs + 1).min(10); 
                    }
                    _ = TASK_QUEUED_SIGNAL.notified() => {
                        delay_secs = 1;
                        println!("[Scheduler] New task signal received. Waking up immediately.");
                    }
                }
                continue;
            } else {
                delay_secs = 1;
            }

            for task in pending_tasks {
                if cancellation_token.load(Ordering::Relaxed) {
                    println!("[Scheduler] Cancellation detected before starting task {}, skipping batch.", task.id);
                    break;
                }

                println!("[Scheduler] Processing task: {}", task.id);
                
                {
                    let store_guard = store.lock().await;
                    if let Some(db) = store_guard.as_ref() {
                        let _ = db.update_task_status(&task.id, crate::logic::parse_status("progress")).await;
                    }
                }

                match process_task(task.clone(), &store, &model, &cancellation_token, &app_handle, current_device_pref.clone()).await {
                    Ok(_) => {
                        println!("[Scheduler] Task completed: {}", task.id);
                        {
                            let mut model_lock = model.lock().await;
                            if let Some(m) = model_lock.as_ref() {
                                m.deep_purge_resources().await;
                            }
                            // 모델 인스턴스 자체를 None으로 만들어 완전히 초기화 (다음 작업 시 필요하면 다시 로드)
                            *model_lock = None;
                        }
                        
                        let store_guard = store.lock().await;
                        if let Some(db) = store_guard.as_ref() {
                            let _ = db.update_task_status(&task.id, crate::logic::parse_status("complete")).await;
                        }
                        // [NEW] 성공 시에만 리소스 정리 및 모드 리셋
                        cleanup_task_resources(&task.id, Some(&app_handle));
                        current_device_pref = None; 
                    },
                    Err(e) => {
                        let err_msg = e.to_string();
                        println!("[Scheduler] Task failed: {:?}. Error: {}", task.id, err_msg);
                        
                        // [PERSISTENT-ERROR-LOG] 작업 디렉토리에 에러 사유 기록
                        let task_dir = utils::paths::get_task_specific_dir(Some(&app_handle), &task.id);
                        if !task_dir.exists() { let _ = std::fs::create_dir_all(&task_dir); }
                        let error_file = task_dir.join("error_reason.txt");
                        let _ = std::fs::write(&error_file, format!("Timestamp: {}\nError: {}\n", chrono::Utc::now(), err_msg));

                        // [CRITICAL-CLEANUP] 작업 실패 시 즉시 모델을 메모리에서 해제하여 다음 작업 대비
                        {
                            let mut model_lock: tokio::sync::MutexGuard<Option<LogisModel>> = model.lock().await;
                            if let Some(m) = model_lock.as_ref() {
                                println!("[Scheduler] Error detected. Performing emergency memory release...");
                                m.deep_purge_resources().await;
                            }
                            *model_lock = None;
                        }

                        if err_msg.contains("Task cancelled") {
                             println!("[Scheduler] Task cancelled: {}", task.id);
                             let store_guard = store.lock().await;
                             if let Some(db) = store_guard.as_ref() {
                                 let _ = db.update_task_status(&task.id, crate::logic::parse_status("cancel")).await;
                                 let _ = db.update_message_status(&task.id, crate::logic::parse_status("cancel"), Some("Cancelled by user")).await;
                             }
                             let _ = app_handle.emit("extraction-progress", json!({
                                "task_id": task.id,
                                "category": "Done", "summary": "Cancelled by user", "spinner": "🛑", "data": null 
                             }));
                             // [NEW] 취소 시 리소스 정리 및 모드 리셋
                             cleanup_task_resources(&task.id, Some(&app_handle));
                             current_device_pref = None;
                             break; 
                        } else {
                            println!("[Scheduler] Task failed: {:?}. Error: {}", task.id, err_msg);
                            
                            // [NEW] Automatic OOM Recovery Logic (Forcing SSD-Swap Mode)
                            if err_msg.contains("CUDA_ERROR_OUT_OF_MEMORY") || err_msg.contains("out of memory") {
                                println!("[Scheduler] OOM Detected! Activating SSD-Swap Mode for retry.");
                                {
                                    let mut model_lock: tokio::sync::MutexGuard<Option<LogisModel>> = model.lock().await;
                                    if let Some(m) = model_lock.as_ref() {
                                        let _ = m.deep_purge_resources().await;
                                    }
                                    *model_lock = None; 
                                }
                                
                                // Use gpu_disk_swap instead of cpu for better performance during retry
                                current_device_pref = Some("gpu_disk_swap".to_string());

                                log_task_progress(&app_handle, &task.id, &json!({
                                    "category": "Warning", "summary": "Memory pressure detected. Retrying with SSD-Swap Mode...", "spinner": "💾"
                                }));
                                
                                tokio::time::sleep(Duration::from_secs(2)).await;
                                continue; 
                            }

                            let store_guard = store.lock().await;                            
                            if let Some(db) = store_guard.as_ref() {
                                let _ = db.update_task_status(&task.id, crate::logic::parse_status("error")).await;
                                let _ = db.update_message_status(&task.id, crate::logic::parse_status("error"), Some(&format!("Error: {}", err_msg))).await;
                            }
                            
                            // [NEW] Explicitly notify UI of failure to stop spinner
                            let _ = app_handle.emit("extraction-progress", json!({
                                "task_id": task.id,
                                "category": "Error", "summary": format!("Failed: {}", err_msg), "spinner": "❌"
                            }));
                        }
                    }
                }
            }
            
            // Reset token after batch is done or broken, for the next poll
            cancellation_token.store(false, Ordering::SeqCst);
        }
    });
}

async fn process_task(
    task: Task,
    store_mutex: &Arc<Mutex<Option<VectorStore>>>,
    model_mutex: &Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    device_preference: Option<String>,
) -> Result<()> {
    let team_id = if !task.to.is_empty() { task.to.clone() } else { crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") };

    // [NEW] Ensure log directory exists at runtime using dynamic path 
    let pug_logs_dir = utils::paths::get_pug_logs_dir(Some(app_handle), &task.id);
    println!("[DEBUG] Pug logs directory: {:?}", pug_logs_dir);

    println!("[PROCESS] Task {} started processing.", task.id);

    // [KV-CHECK] Check if task specific KV exists
    let kv_path = utils::paths::get_kv_dir(Some(app_handle)).join(&task.id);
    if kv_path.exists() {
        println!("[PROCESS] Found existing KV cache for task {}. Ready to reuse.", task.id);
    }

    // [SPINNER-ACTIVATE] Ensure UI spinner is ON immediately upon task recovery/start
    let payload = json!({ 
        "task_id": task.id,
        "category": "Processing", "summary": "Starting extraction...", "spinner": "⠋" 
    });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // [MEMORY] Initialize Data Manager for this task scope with AppHandle for dynamic paths
    let mut data_manager = TaskDataManager::new(&task.id, Some(app_handle.clone()));

    let mut task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    // [FIX] 작업 유형에 따라 파일명을 자동으로 결정합니다.
    let kv_name = if task.r#type == "image_extraction" {
        Some("image".to_string())
    } else {
        Some("text".to_string())
    };
    
    // [FIX] Robust device preference parsing (supports both "cpu" string and true/false boolean)
    let task_device_pref = if let Some(v) = task_data.get("device_preference") {
        if v.as_str() == Some("cpu") || v.as_bool() == Some(true) {
            Some("cpu".to_string())
        } else {
            None
        }
    } else {
        None
    };
    let effective_device_pref = task_device_pref.as_deref().or(device_preference.as_deref());
    
    let language = "english"; 

    // [LOCK] Acquire Model Access
    let model = {
        println!("[Scheduler] 🛡️ Attempting to acquire Model Lock...");
        let mut model_lock = model_mutex.lock().await;
        println!("[Scheduler] ✅ Model Lock acquired.");
        
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        // [FIX] If current model doesn't match preference, unload it to force switch (CPU <-> GPU)
        if let Some(m) = model_lock.as_ref() {
            let wants_cpu = effective_device_pref == Some("cpu");
            if m.is_cpu_mode != wants_cpu {
                println!("[Scheduler] Device preference mismatch (Current CPU: {}, Wants CPU: {}). Reloading model...", m.is_cpu_mode, wants_cpu);
                m.deep_purge_resources().await;
                *model_lock = None;
            }
        }

        if model_lock.is_none() {
            println!("[Scheduler] Model not initialized. Starting LogisModel::new...");
            // [LOG-ONLY] No emit here to keep UI clean
            log_task_progress(app_handle, &task.id, &json!({ "category": "Loading Model", "summary": "Initializing AI Core..." }));
            
            match LogisModel::new(app_handle.clone(), effective_device_pref).await {
                Ok(m) => {
                    println!("[Scheduler] LogisModel::new successful.");
                    *model_lock = Some(m);
                },
                Err(e) => {
                    println!("[Scheduler] ❌ LogisModel::new failed: {}", e);
                    return Err(anyhow::anyhow!("Model Load Failed: {}", e));
                }
            }
        }
        model_lock.as_ref().unwrap().clone()
    };

    // --- Image Extraction Logic (Qwen 3.5 Pipeline) ---
    if task.r#type == "image_extraction" {
        let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("").to_string();
        if !image_path.is_empty() {
            println!("[Scheduler] Starting Image Extraction for {}", task.id);
            
            // [QWEN3.5] Log progress
            log_task_progress(app_handle, &task.id, &json!({ "category": "Vision", "summary": "Analyzing visual context with Qwen 3.5...", "spinner": "⠋" }));
            
            // ensure_qwen3_5 is called internally by extract_from_image
            model.extract_from_image(
                task.id.clone(),
                image_path,
                "korean".to_string(),
                app_handle,
                Some(cancellation_token.clone()),
                store_mutex,
            ).await?;
            
            return Ok(()); 
        }
    }

    let url = task_data.get("link").and_then(|s| s.as_str()).unwrap_or("").to_string();
    if url.is_empty() { return Ok(()); }

    // [RESUME-LOGIC] Check if PUG already exists
    let light_pug_path = data_manager.get_path("light_pug");
    let light_pug = if light_pug_path.exists() {
        println!("[PROCESS] Resuming from existing PUG file.");
        data_manager.load(&light_pug_path)?
    } else {
        // [MEMORY] Fetch and immediately offload Raw HTML
        // [FIX] Prefix with underscore to suppress unused variable warning
        let _raw_html_path = if let Some(raw_html) = task_data.get("html").and_then(|s| s.as_str()) {
            let p = data_manager.offload(raw_html, "raw_html")?;
            if let Some(obj) = task_data.as_object_mut() { obj.remove("html"); }
            p
        } else if !url.is_empty() {
            let response = reqwest::get(&url).await?;
            let bytes = response.bytes().await?;
            
            // [ENCODING-FIX] UTF-8 First Strategy
            let (decoded_utf8, _, malformed_utf8) = encoding_rs::UTF_8.decode(&bytes);
            let utf8_str = decoded_utf8.as_ref();
            
            // Check for explicit EUC-KR/CP949 markers in the UTF-8 decoded string
            let needs_euc = utf8_str.to_lowercase().contains("charset=euc-kr") || 
                            utf8_str.to_lowercase().contains("charset=\"euc-kr\"") ||
                            utf8_str.to_lowercase().contains("charset=cp949") ||
                            utf8_str.to_lowercase().contains("charset=ks_c_5601");

            let content = if needs_euc && malformed_utf8 {
                // Only use EUC-KR if it's explicitly requested AND UTF-8 decoding had issues
                let (decoded_euc, _, _) = encoding_rs::EUC_KR.decode(&bytes);
                decoded_euc.into_owned()
            } else {
                // Default to UTF-8 (Lossy fallback if needed)
                utf8_str.to_string()
            };
            
            data_manager.offload(&content, "raw_html")?
        } else {
            return Ok(());
        };
        
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        // 2. Clean & Pug Conversion
        let pug = {
            let raw_html_path = data_manager.get_path("raw_html");
            let clean_html_path = data_manager.get_path("clean_html");
            
            let clean = if clean_html_path.exists() {
                data_manager.load(&clean_html_path)?
            } else {
                let raw_content = data_manager.load(&raw_html_path)?;
                let c = parsing::pre_clean_html(&raw_content);
                data_manager.offload(&c, "clean_html")?;
                c
            };
            
            let p = parsing::convert_to_clean_pug(&clean, PugMode::FullContent);
            println!("[DEBUG-PUG] Generated PUG. Length: {}. Snippet: {}...", 
                p.len(), 
                if p.len() > 100 { &p[..100] } else { &p }.replace("\n", " ")
            );
            
            // [DEBUG-LOG] Save generated Pug
            let ts_nano = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
            let log_path = pug_logs_dir.join(format!("light_{}_{}.pug", task.id, ts_nano));
            let _ = std::fs::write(&log_path, &p);
            
            let _ = data_manager.offload(&p, "light_pug")?;
            p // Return String
        };
        pug
    };


    use crate::openai_types::{
        ChatCompletionRequestSystemMessage,
        ChatCompletionParameters, ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent
    };

    let mut page_type = String::new();
    let mut selector_info: serde_json::Value = json!({});

    let mut is_detail = false;

    // ==================================================================================
    // [ULTRA-OPTIMIZED PIPELINE]
    // Step 0: 0.6B Base Baking [System: PUG] -> Save task_id_base
    // Step 1: 0.6B Loads base -> Ask Classification [User: Task] -> Save task_id_step_a
    // Step 2: 0.6B Loads base -> Ask Selectors [User: Task] -> Save task_id_step_b
    // ==================================================================================

    let base_session_id = format!("{}_base", task.id);
    let base_session_id_35 = format!("{}_base_q35", task.id); // 🌟 0.8B 전용 세션 ID
    let system_content = format!("[PUG CONTENT]\n{}", light_pug);

    // --- STEP 0: BASE BAKING (공통 컨텍스트 딱 1번만 굽기) ---
    {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        
        let base_kv_path = utils::paths::get_kv_dir(Some(app_handle)).join(&base_session_id);
        if !base_kv_path.exists() {
            println!("[Scheduler] Baking Base PUG Context to SSD...");
            log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Reading document structure...", "spinner": "⠋" }));
            
            model.secure_vram_relay(crate::model::ModelSize::Small, None, Some(cancellation_token.clone()), false, kv_name.clone()).await?;
            
            let params = ChatCompletionParameters {
                messages: vec![ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: system_content.clone(),
                    name: None,
                })],
                model: "qwen".to_string(),
                ..Default::default()
            };

            if let Some(gen) = model.generator.lock().await.as_mut() {
                // System 메시지(PUG)만 1만 토큰을 읽어서 base_session_id 로 저장합니다.
                gen.prefill_only(params, Some(cancellation_token.clone()), Some(base_session_id.clone()), None, kv_name.clone()).await?;
            }
        }
    }

    // --- STEP A: CLASSIFICATION (분류) ---
    {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        println!("[Scheduler] Starting DISK BRIDGE RELAY (Load Base -> Classify)");
        
        log_task_progress(app_handle, &task.id, &json!({ "category": "Classification", "summary": "Determining page type...", "spinner": "⠋" }));

        let type_prompt = parsing::page_type_prompt();
        let task_question = format!("[TASK] Identify the page type.\n\n[INSTRUCTION]\n{}\n\n[ACTION] RETURN JSON ONLY.", type_prompt);
        let snapshot_id = format!("{}_step_a", task.id);
        
        if let Ok(content) = std::fs::read_to_string("tmp/index.json") {
            if let Ok(mut json_val) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(obj) = json_val.as_object_mut() {
                    obj.insert("step".to_string(), json!("Step A (Classification)"));
                    obj.insert("session_id".to_string(), json!(snapshot_id.clone()));
                    obj.insert("kv_path".to_string(), json!(kv_name.clone().unwrap_or_else(|| "tmp/kv/".to_string())));
                    let _ = std::fs::write("tmp/index.json", json_val.to_string());
                }
            }
        }

        {
            // [핵심] Step A가 아니라 '미리 구워둔 Base' 스냅샷을 불러옵니다!
            model.secure_vram_relay(crate::model::ModelSize::Small, Some(&base_session_id), Some(cancellation_token.clone()), false, kv_name.clone()).await?;

            let params = ChatCompletionParameters {
                messages: vec![
                    // Base 캐시와 토큰을 100% 일치시키기 위해 System 메시지를 그대로 넣습니다.
                    ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                        content: system_content.clone(),
                        name: None,
                    }),
                    // 질문은 User 메시지로 분리합니다. (이 부분 50토큰만 연산됨!)
                    ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                        content: ChatCompletionRequestUserMessageContent::Text(task_question.clone()),
                        name: None,
                    })
                ],
                model: "qwen".to_string(), 
                max_tokens: Some(16),
                temperature: Some(0.0), top_p: Some(0.01),
                ..Default::default()
            };

            if let Some(gen) = model.generator.lock().await.as_mut() {
                println!("[Scheduler] 0.6B Step A: Asking classification question...");
                let res = gen.generate(params, Some(cancellation_token.clone()), Some(snapshot_id.clone()), kv_name.clone()).await?;
                println!("[DEBUG-SCHED] Step A Raw Response: '{}'", res);
                
                let _ = data_manager.offload(&res, "step_a_res");
                let type_info = parsing::parse_json_from_llm(&res); 
                page_type = type_info.get("type").and_then(|s| s.as_str()).unwrap_or("").to_string();                
                if page_type.is_empty() {
                    page_type = match task.r#type.as_str() {
                        "image_extraction" => "tracking".to_string(),
                        _ => "unknown".to_string(),
                    };
                }
                println!("[Scheduler] Classified as: {}", page_type);
            }
        }
        
        if page_type.is_empty() || page_type == "unknown" { 
            model.deep_purge_resources().await;
            return Ok(()); 
        }
    }

    // --- STEP A-2: DETAIL CLASSIFICATION (디테일 페이지 여부 독립 판별) ---
    {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        println!("[Scheduler] Starting DISK BRIDGE RELAY (Load Base -> Is Detail)");
        
        log_task_progress(app_handle, &task.id, &json!({ "category": "Classification", "summary": "Determining if detail page...", "spinner": "⠋" }));

        let detail_prompt = parsing::is_detail_prompt(&page_type);
        // LLM이 지시사항을 잘 따르도록 래핑
        let task_question = format!("{}\n\n[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think", detail_prompt);
        let snapshot_id = format!("{}_step_a2", task.id);

        model.secure_vram_relay(crate::model::ModelSize::Small, Some(&base_session_id), Some(cancellation_token.clone()), false, kv_name.clone()).await?;

        let params = ChatCompletionParameters {
            messages: vec![
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: system_content.clone(),
                    name: None,
                }),
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                    content: ChatCompletionRequestUserMessageContent::Text(task_question),
                    name: None,
                })
            ],
            model: "qwen".to_string(), 
            max_tokens: Some(64), // true/false만 대답하므로 짧게 설정
            temperature: Some(0.0), top_p: Some(0.01),
            ..Default::default()
        };

        if let Some(gen) = model.generator.lock().await.as_mut() {
            println!("[Scheduler] 0.6B Step A-2: Asking detail classification...");
            let res = gen.generate(params, Some(cancellation_token.clone()), Some(snapshot_id.clone()), kv_name.clone()).await?;
            println!("[DEBUG-SCHED] Step A-2 Raw Response: '{}'", res);
            
            let detail_info = parsing::parse_json_from_llm(&res); 
            is_detail = detail_info.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
            println!("[Scheduler] Classified is_detail as: {}", is_detail);
        }
    }
                        
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    model.deep_purge_resources().await;
    wait_for_resources_settled(1200, 800, Some(cancellation_token)).await?;

    let mut extracted_data = json!({});

    // --- PHASE 2 Continue: Detail Extraction (If needed) --- 
    if !is_detail {
        // --- STEP B: SELECTORS (선택자 추출 - JS 기반 신규 로직) ---
        {
            use boa_engine::{Context, Source};
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            println!("[Scheduler] Starting JS-BASED SELECTOR ANALYSIS (LLM Titles -> Boa Engine)");
            
            log_task_progress(app_handle, &task.id, &json!({ "category": "Selector Search", "summary": "Analyzing DOM with JS engine...", "spinner": "⠋" }));

            // 1. LLM에게 상품명(titles) 추출 요청
            let title_prompt = parsing::extract_titles_prompt(&page_type);
            let task_question = format!("{}\n\n[ACTION] RETURN JSON ONLY.", title_prompt);
            let snapshot_id = format!("{}_step_b_titles", task.id);

            let mut titles = Vec::new();
            {
                // 🌟 None 넘기기
                model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, Some(&base_session_id), Some(cancellation_token.clone()), false, kv_name.clone()).await?;

                let params = ChatCompletionParameters {
                    messages: vec![
                        ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                            content: system_content.clone(),
                            name: None,
                        }),
                        ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                            content: ChatCompletionRequestUserMessageContent::Text(task_question.clone()),
                            name: None,
                        })
                    ],
                    model: "qwen3.5".to_string(), max_tokens: Some(512), temperature: Some(0.0), top_p: Some(0.01),
                    ..Default::default()
                };

                if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
                    println!("[JS-BRIDGE] 1. Requesting titles from LLM...");
                    // 🌟 generate_part 로 교체 및 base_session_id_35 넘기기
                    let res = gen.generate_part(&params, false, 0, None, Some(snapshot_id.clone()), kv_name.clone()).await?;
                    println!("[JS-BRIDGE] LLM Raw Response: '{}'", res.text);

                    let title_info = parsing::parse_json_from_llm(&res.text);
                    let items_opt = title_info.get("order")
                        .or(title_info.get("goods"))
                        .or(title_info.get("items"))
                        .or(title_info.get("titles"))
                        .or(title_info.get("products"))
                        .and_then(|v| v.as_array());

                    if let Some(items) = items_opt {
                        for item in items {
                            if let Some(t) = item.as_str() {
                                titles.push(t.to_string());
                            } else if let Some(t) = item.get("title").and_then(|v| v.as_str()) {
                                titles.push(t.to_string());
                            }
                        }
                    }
                    println!("[JS-BRIDGE] Titles extracted (Robust): {:?}", titles);
                }
            }

            if titles.is_empty() {
                 println!("[JS-BRIDGE] Warning: No titles extracted from LLM. Falling back to default.");
            }

            // 2. Boa Engine으로 DOM 분석
            {
                println!("[JS-BRIDGE] 2. Starting boa-engine for DOM analysis...");
                let mut context = Context::default();
                
                let clean_html = data_manager.load(&data_manager.get_path("clean_html"))?;
                let document = scraper::Html::parse_document(&clean_html);
                
                let mut nodes_json = Vec::new();
                let mut node_to_idx = std::collections::HashMap::new();

                // 1단계: 모든 노드 ID 매핑 (부모 참조 안정성 확보)
                for (idx, node) in document.tree.root().descendants().enumerate() {
                    node_to_idx.insert(node.id(), idx);
                }

                // 2단계: 노드 정보 수집 (Element 노드 중심)
                for (idx, node) in document.tree.root().descendants().enumerate() {
                    if let Some(el) = node.value().as_element() {
                        let parent_idx = node.parent().and_then(|p| node_to_idx.get(&p.id())).map(|&i| i as i32).unwrap_or(-1);
                        
                        let text: String = node.children()
                            .filter_map(|child| child.value().as_text().map(|t| t.to_string()))
                            .collect::<Vec<_>>()
                            .join(" ")
                            .trim()
                            .to_string();
                            
                        nodes_json.push(json!({
                            "index": idx,
                            "parentIndex": parent_idx,
                            "tagName": el.name().to_string(),
                            "id": el.id().unwrap_or("").to_string(),
                            "classes": el.attr("class").unwrap_or("").split_whitespace().collect::<Vec<_>>(),
                            "text": text
                        }));
                    } else {
                        nodes_json.push(json!(null));
                    }
                }
                
                let nodes_str = serde_json::to_string(&nodes_json)?;
                let titles_str = serde_json::to_string(&titles)?;

                let js_template = r##"
                    const nodes = NODES_PLACEHOLDER;
                    const titles = TITLES_PLACEHOLDER;
                    
                    function cleanClassList(classes, stripNumbers = false) {
                        if (!classes) return [];
                        const skip = ['active', 'selected', 'on', 'current', 'focus', 'hover', 'enabled', 'disabled'];
                        return classes
                            .filter(c => {
                                const lowerC = c.toLowerCase();
                                return !skip.includes(lowerC) && c.indexOf('__') === -1 && !/^[a-z0-9]{8,}$/.test(c);
                            })
                            .map(c => stripNumbers ? c.replace(/\d+$/, '') : c)
                            .sort();
                    }

                    function getSignature(node, includeId = true) {
                        if (!node || !node.tagName) return "";
                        let s = node.tagName;
                        if (includeId && node.id) s += "#" + node.id;
                        const cls = cleanClassList(node.classes);
                        if (cls.length > 0) {
                            s += "." + [...new Set(cls)].join(".");
                        }
                        return s;
                    }

                    function getChildren(pIdx) { 
                        return nodes.filter(n => n && n.parentIndex === pIdx); 
                    }

                    function calculateSimilarity(nodeA, nodeB) {
                        if (nodeA.tagName !== nodeB.tagName) return 0;
                        const clsA = cleanClassList(nodeA.classes, true);
                        const clsB = cleanClassList(nodeB.classes, true);
                        if (clsA.length === 0 && clsB.length === 0) return 100;
                        
                        let matchCount = 0;
                        clsA.forEach(c => { if (clsB.includes(c)) matchCount++; });
                        return clsA.length ? (matchCount / clsA.length) * 100 : 0;
                    }

                    function detect(tIdx) {
                        let cur = tIdx;
                        for (let i = 0; i < 15; i++) {
                            const node = nodes[cur];
                            if (!node) break;
                            
                            const pIdx = node.parentIndex;
                            if (pIdx === undefined || pIdx === -1) break;
                            
                            const parentNode = nodes[pIdx];
                            const siblings = getChildren(pIdx);
                            
                            const similarSiblings = siblings.filter(s => calculateSimilarity(node, s) >= 60);

                            if (similarSiblings.length >= 2) {
                                let finalParent = parentNode;
                                let walkIdx = pIdx;
                                for(let j=0; j<5; j++) {
                                    let gIdx = nodes[walkIdx] ? nodes[walkIdx].parentIndex : -1;
                                    if (gIdx !== -1 && nodes[gIdx]) {
                                        const grand = nodes[gIdx];
                                        if (grand.id || ["table", "ul", "ol", "nav"].includes(grand.tagName)) {
                                            finalParent = grand;
                                            if (grand.id || grand.tagName === "table") break;
                                        }
                                        walkIdx = gIdx;
                                    }
                                }

                                const parentSig = getSignature(finalParent, true);
                                const uniqueSigs = [];
                                similarSiblings.forEach(s => {
                                    const sig = getSignature(s, false);
                                    if (!uniqueSigs.includes(sig)) uniqueSigs.push(sig);
                                });

                                const fullSelector = uniqueSigs.map(sig => parentSig + " " + sig).join(", ");

                                return { 
                                    parent: parentSig, 
                                    itemSelector: fullSelector,
                                    matchCount: similarSiblings.length
                                };
                            }
                            cur = pIdx;
                        }
                        return null;
                    }

                    const findText = titles.length > 0 ? titles[0].toLowerCase().replace(/\s+/g, ' ') : "";
                    const matches = nodes.filter(n => n && n.text && n.text.toLowerCase().replace(/\s+/g, ' ').includes(findText));
                    
                    let res = { "parent": "body", "itemSelector": "div", "matchCount": matches.length };
                    if (matches.length > 0) {
                        const d = detect(matches[0].index);
                        if (d) { res.parent = d.parent; res.itemSelector = d.itemSelector; }
                    }
                    JSON.stringify(res);
                "##;

                let js_code = js_template
                    .replace("NODES_PLACEHOLDER", &nodes_str)
                    .replace("TITLES_PLACEHOLDER", &titles_str);

                match context.eval(Source::from_bytes(js_code.as_bytes())) {
                    Ok(val) => {
                        let res_str = val.as_string().unwrap().to_std_string_escaped();
                        println!("[JS-BRIDGE] Boa Final Result: {}", res_str);
                        selector_info = serde_json::from_str(&res_str).unwrap_or(json!({}));
                    },
                    Err(e) => {
                        println!("[JS-BRIDGE] Error executing JS: {:?}", e);
                    }
                }
            }
        }

        // [INTERMEDIATE PARITY LOGIC] Save Page Info
        let mut final_page_info = json!({ "type": page_type });
        if let Some(obj) = selector_info.as_object() {
            for (k, v) in obj { 
                final_page_info.as_object_mut().unwrap().insert(k.clone(), v.clone()); 
            }
        }
        
        // [PARITY] Store 'Page' Entity
        {
            // Acquire Store lock briefly
            let store = {
                let store_guard = store_mutex.lock().await;
                store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
            };
            
            // [FIX] Try to load the active origin and type from the shared JSON file (tmp/index.json)
            // This is the most reliable way to bridge the gap between automation and scheduler.
            let mut shared_origin = None;
            let mut shared_type = None;
            if let Ok(content) = std::fs::read_to_string("tmp/index.json") {
                if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(o) = json_val.get("origin").and_then(|v| v.as_str()) {
                        if let Ok(u) = url::Url::parse(o) {
                            shared_origin = Some(format!("{}://{}", u.scheme(), u.host_str().unwrap_or("localhost")));
                        }
                    }
                    if let Some(t) = json_val.get("type").and_then(|v| v.as_str()) {
                        if !t.is_empty() { shared_type = Some(t.to_string()); }
                    }
                }
            }

            let origin_str = task_data.get("origin").and_then(|s| s.as_str()).map(|s| s.to_string())
                .filter(|s| s != "http://localhost") // If it's already localhost, treat it as missing
                .or(shared_origin) // Fallback to the shared file
                .unwrap_or_else(|| {
                    // If all else fails, extract from the task.data_json URL
                    if let Ok(task_url) = url::Url::parse(&url) {
                        format!("{}://{}", task_url.scheme(), task_url.host_str().unwrap_or("localhost"))
                    } else {
                        "http://localhost".to_string()
                    }
                });

            // Use shared type if available and task type is missing or unknown
            if page_type.is_empty() || page_type == "unknown" {
                if let Some(st) = shared_type { page_type = st; }
            }
                
            let base_url = url::Url::parse(&origin_str).unwrap_or_else(|_| url::Url::parse("http://localhost").unwrap());
            let url_obj = base_url.join(&url).unwrap_or(base_url);
            let raw_path = url_obj.path();
            let page_id = crate::utils::hash::hash_id(&format!("{}{}", task.cc, raw_path)); 
            let cc_for_bcc = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
            let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_for_bcc));

            let mut page_data: serde_json::Value = selector_info.clone();
            if let Some(obj) = page_data.as_object_mut() {
                obj.insert("origin".to_string(), json!(format!("{}://{}", url_obj.scheme(), url_obj.host_str().unwrap_or(""))));
                obj.insert("link".to_string(), json!(url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str()));
                obj.insert("type".to_string(), json!(page_type));
            }

            let _ = store.upsert_item("pages", &page_id, "pages", page_data, None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(raw_path), None).await;
        }

        let item_selector = final_page_info.get("itemSelector")
            .or_else(|| final_page_info.get("item"))
            .and_then(|s| s.as_str())
            .unwrap_or("");
        let node_selector = final_page_info.get("node").and_then(|s| s.as_str()).unwrap_or("");
        
        let target_selector = if !node_selector.is_empty() && !item_selector.is_empty() && !item_selector.contains(",") {
            format!("{} {}", node_selector, item_selector) 
        } else if !item_selector.is_empty() { 
            item_selector.to_string() 
        } else { 
            node_selector.to_string() 
        };
        
        // [LIST MODE] 지능형 리스트 추출 (LLM 기반)
        let list_log = json!({ "category": "List Processing", "summary": "Extracting list data with LLM...", "spinner": "⠋" });
        log_task_progress(app_handle, &task.id, &list_log);

        let mut all_extracted_items = Vec::new();
        
        // 병합 대기열을 위한 변수들
        let mut pending_merge: Option<serde_json::Value> = None;
        let mut merge_countdown = 0;

        let pug_list = {
            let clean_html_path = data_manager.get_path("clean_html");
            let clean_content = data_manager.load(&clean_html_path)?;
            let headers = parsing::extract_table_headers(&clean_content, &target_selector);
            if !headers.is_empty() {
                println!("[Scheduler] Extracted {} header rows for 'alt' mapping.", headers.len());
            }
            let document = scraper::Html::parse_document(&clean_content);
            parsing::split_doc_to_pug_list_advanced(&document, &target_selector, PugMode::FullContent, if headers.is_empty() { None } else { Some(headers) })
        };

        if !pug_list.is_empty() {
            model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, Some(&base_session_id), Some(cancellation_token.clone()), false, Some("inference".to_string())).await?;

            for (idx, item_pug) in pug_list.iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { break; }
                
                let extraction_instruction = parsing::list2json(&page_type, language);
                let task_question = format!("[PUG CONTENT]\n{}\n\n{}", item_pug, extraction_instruction);
                
                // 🌟 [교체 구간 2-B] src/scheduler.rs 의 리스트 추출 루프 내부
                let res = if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
                    println!("[Scheduler] Qwen3.5 Extracting Item {}/{}...", idx + 1, pug_list.len());
                    
                    // 🌟 리스트 아이템별로 덮어쓰지 않도록 고유 세션 ID 생성
                    let item_session_id = format!("{}_list_item_{}", task.id, idx); 
                    
                    let params = ChatCompletionParameters {
                        messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                            content: ChatCompletionRequestUserMessageContent::Text(task_question),
                            name: None,
                        })],
                        model: "qwen3.5".to_string(), max_tokens: Some(512), temperature: Some(0.0), top_p: Some(0.01),
                        ..Default::default()
                    };
                    gen.generate(
                        params, 
                        Some(cancellation_token.clone()), 
                        Some(item_session_id), // 🌟 None 에서 생성한 고유 세션 ID로 변경
                        Some("inference".to_string())
                    ).await.map_err(|e| anyhow::anyhow!("Qwen 3.5 Inference failed: {}", e))
                } else {
                    Err(anyhow::anyhow!("Qwen 3.5 Generator not available"))
                };

                match res {
                    Ok(res_text) => {
                        let item_json = parsing::parse_json_from_llm(&res_text);
                        if !item_json.is_null() && (item_json.is_object() || item_json.is_array()) {
                            
                            // 단순 push 대신 후처리 병합 수행 (rowspan 처리)
                            let is_continuation = item_json.get("is_continuation").and_then(|v| v.as_bool()).unwrap_or(false);
                            
                            if is_continuation && pending_merge.is_some() {
                                let mut parent = pending_merge.take().unwrap();
                                crate::scheduler::merge_json_results(&mut parent, &item_json);
                                merge_countdown -= 1;
                                
                                if merge_countdown > 0 {
                                    pending_merge = Some(parent);
                                } else {
                                    all_extracted_items.push(parent);
                                }
                            } else {
                                let rowspan = item_json.get("rowspan_count").and_then(|v| v.as_u64()).unwrap_or(1);
                                if rowspan > 1 {
                                    pending_merge = Some(item_json);
                                    merge_countdown = rowspan - 1;
                                } else {
                                    // 기존에 대기 중이던 게 꼬였다면 일단 push하고 비움
                                    if let Some(stray) = pending_merge.take() {
                                        all_extracted_items.push(stray);
                                    }
                                    all_extracted_items.push(item_json);
                                }
                            }
                        }
                    },
                    Err(e) => println!("[Scheduler] Error extracting item {}: {:?}", idx, e),
                }
            }

            if let Some(last_item) = pending_merge.take() {
                all_extracted_items.push(last_item);
            }
        }
        extracted_data = json!({ "items": all_extracted_items, "type": page_type, "detail": false });

    } else {
        // [DETAIL MODE] Disk Bridge Relay
        println!("[Scheduler] Starting DISK BRIDGE RELAY for Details");
        
        let content_pug = {
            let clean_html_path = data_manager.get_path("clean_html");
            let clean_content = data_manager.load(&clean_html_path)?;
            parsing::convert_to_clean_pug(&clean_content, PugMode::FullContent)
        };

        if !content_pug.trim().is_empty() {
            let extraction_instruction = parsing::item2json(&page_type, &url, language);
            let snapshot_id = format!("{}_detail", task.id);

            // 1. [Large] Load & Generate (Direct Qwen3.5 0.8B-Layer Generation)
            {
                // 🌟 None 넘기기
                model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, Some(&base_session_id), Some(cancellation_token.clone()), false, kv_name.clone()).await?;

                let params = ChatCompletionParameters {
                    messages: vec![
                        ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                            content: system_content.clone(),
                            name: None,
                        }),
                        ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                            content: ChatCompletionRequestUserMessageContent::Text(format!(
                                "[TASK] {}\n\n[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think",
                                extraction_instruction
                            )),
                            name: None,
                        })
                    ],
                    model: "qwen3.5".to_string(), 
                    max_tokens: Some(1048), 
                    temperature: Some(0.0), 
                    top_p: Some(0.01),
                    ..Default::default()
                };

                // 🌟 [교체 구간 2-C] src/scheduler.rs 의 Detail Extraction 내부 (하단부)
                if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
                    println!("[Scheduler] Qwen3.5 Step C: Asking extraction question...");
                    log_task_progress(app_handle, &task.id, &json!({ "category": "Extraction", "summary": "Running Qwen 3.5 Inference..." }));
                    
                    // 🌟 generate_part 에 None 대신 Some(snapshot_id.clone()) 전달
                    let res = gen.generate_part(&params, false, 0, None, Some(snapshot_id.clone()), kv_name.clone()).await?;
                    
                    println!("[DEBUG-SCHED] Step C Raw Response: '{}'", res.text);

                    // [DEBUG] AI 응답 저장
                    let _ = data_manager.offload(&res.text, "step_c_res");

                    extracted_data = parsing::parse_json_from_llm(&res.text);
                } else {
                    println!("[Scheduler] ERROR: Qwen 3.5 generator is missing!");
                }
            }
        }
    }

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // --- PHASE 3: HANDOVER (Unload -> Load Embedding) ---
    {
        println!("[Scheduler] PHASE 3: Handover - Unloading, Preparing for Embedding...");
        log_task_progress(app_handle, &task.id, &json!({ "category": "Handover", "summary": "Switching to Embedding model..." }));
        
        // 1. Explicitly Unload to free VRAM for Embedding Model
        model.deep_purge_resources().await;
        
        // 2. Wait for VRAM to settle (Driver latency)
        wait_for_resources_settled(1200, 800, Some(cancellation_token)).await?;
    }

    // --- DB OPS & SIDE EFFECTS ---
    // Normalize Data
    if let Some(obj) = extracted_data.as_object_mut() {
        if obj.get("type").is_none() { obj.insert("type".to_string(), json!(page_type)); }
    }
    
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // [PARITY] ID Generation
    let id_val_raw = extracted_data.get("id").or_else(|| extracted_data.get("index")).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let clean_no = crate::utils::hash::normalize_numeric_homoglyphs(&id_val_raw).replace("-", "").replace("_", "");
    let index_val = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("{}{}{}", page_type, team_id, clean_no)));
    
    let generated_id = crate::utils::hash::hash_id(&format!("{}{}", team_id, index_val));

    if let Some(obj) = extracted_data.as_object_mut() {
        obj.insert("index".to_string(), json!(index_val));
        obj.insert("id".to_string(), json!(generated_id));
    }

    log_task_progress(app_handle, &task.id, &json!({ "category": "Saving", "summary": "Syncing to database..." }));

    // Re-acquire Store for final ops
    let store = {
        let store_guard = store_mutex.lock().await;
        store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
    };

    if page_type == "order" {
        if let Some(goods_arr) = extracted_data.get("goods").and_then(|v| v.as_array()) {
            let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
            for good in goods_arr {
                let g_no = good.get("id").or_else(|| good.get("no")).and_then(|v| v.as_str()).unwrap_or("");
                if !g_no.is_empty() {
                    let clean_g_no = crate::utils::hash::normalize_numeric_homoglyphs(g_no).replace("-", "").replace("_", "");
                    let tracking_id = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, index_val, clean_g_no));
                    let mut tracking_data = extracted_data.clone();
                    tracking_data.as_object_mut().unwrap().insert("type".to_string(), json!("tracking"));
                    
                    let _ = store.upsert_item(
                        "tracking", &tracking_id, "tracking", tracking_data, None,
                        Some(&task.from), Some(&team_id), Some(&task.cc),
                        Some(&crate::utils::hash::hash_id(&format!("tracking{}", cc_val))),
                        Some(&crate::utils::hash::hash_id(&format!("{}{}{}", team_id, task.cc, task.r#ref))),
                        None
                    ).await;
                }
            }
        }
    }

    let target_table = page_type.to_string();
    let text_to_embed = parsing::json_to_natural_language(&extracted_data);
    let item_digest = crate::utils::hash::digest(&text_to_embed); 
    let target_id = if !task.r#ref.is_empty() { task.r#ref.clone() } else { generated_id }; 
    
    let mut existing_vector = None;
    if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &target_id).await {
        if existing_item.digest == item_digest {
            existing_vector = Some(existing_item.vector);
        }
    }

    let vector = if let Some(v) = existing_vector {
        Some(v)
    } else {
        Some(model.get_embedding(text_to_embed).await?)
    };

    let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
    let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_val));
    let ref_val = task.r#ref.clone();

    let _ = store.upsert_item(
        &target_table, &target_id, &page_type, extracted_data.clone(), vector.clone(),
        Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)
    ).await;
    
    let _ = store.upsert_item(
        "items", &target_id, &page_type, extracted_data.clone(), vector,
        Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)
    ).await;

    // Final Status Update
    let _ = store.update_message_status(&task.id, logic::parse_status("complete"), Some("Extraction Complete")).await;

    let payload = json!({
        "task_id": task.id, "category": "Done", "summary": "Extraction complete.", "spinner": "✅",
        "data": if !is_detail { json!(null) } else { extracted_data }
    });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);
    
    println!("[PROCESS] Task {} completed. Handover to Embedding finished.", task.id);
    Ok(())
}

fn cleanup_task_resources(task_id: &str, app_handle: Option<&tauri::AppHandle>) {
    // [FIX] Only remove the specific directory for this task
    let _ = fs::remove_dir_all(utils::paths::get_task_specific_dir(app_handle, task_id));
}

// [3번 가속: PRE-FETCH] OS 페이지 캐시에 무게추 파일을 미리 로드함
fn pre_fetch_weights(path: &std::path::Path) -> Result<()> {
    use std::io::Read;
    println!("[PRE-FETCH] Warming up OS Page Cache for weights in: {:?}", path);
    if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let p = entry.path();
                    if p.extension().map_or(false, |ext| ext == "gguf" || ext == "safetensors") {
                        if let Ok(mut file) = std::fs::File::open(p) {
                            let mut buffer = [0u8; 1024 * 1024]; // 1MB buffer
                            // 파일 전체를 읽어서 OS가 램에 캐싱하도록 유도함
                            while let Ok(n) = file.read(&mut buffer) {
                                if n == 0 { break; }
                            }
                        }
                    }
                }
            }
        }
    }
    println!("[PRE-FETCH] Warm-up complete.");
    Ok(())
}

pub fn log_task_progress(app: &tauri::AppHandle, task_id: &str, payload: &serde_json::Value) {
    use std::io::Write;
    let log_path = crate::utils::paths::get_task_log_file(Some(app), task_id);
    
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path) 
    {
        let line = format!("{}\n", payload.to_string());
        let _ = file.write_all(line.as_bytes());
    }
}

fn clear_all_temp_data(app_handle: Option<&tauri::AppHandle>) {
    println!("[Cleanup] Clearing all temporary data directories...");
    utils::paths::cleanup_temp_dirs(app_handle);
}

async fn wait_for_resources_settled(target_vram_mb: u64, target_ram_mb: u64, cancellation_token: Option<&Arc<AtomicBool>>) -> Result<()> {
    use nvml_wrapper::Nvml;
    use sysinfo::System;
    
    let mut sys = System::new_all();
    let nvml = Nvml::init().ok();
    
    let target_vram_bytes = target_vram_mb * 1024 * 1024;
    let target_ram_bytes = target_ram_mb * 1024 * 1024;

    let mut last_vram = 0;
    let mut stable_ticks = 0;
    let mut last_report = std::time::Instant::now();
    let start_time = std::time::Instant::now();

    println!("[RESOURCE-WATCH] Monitoring recovery (Target VRAM > {}MB)...", target_vram_mb);

    loop {
        if let Some(token) = cancellation_token {
            if token.load(Ordering::Relaxed) {
                return Err(anyhow::anyhow!("Task cancelled during resource wait"));
            }
        }

        sys.refresh_memory(); 
        let current_ram = sys.available_memory();
        let mut current_vram = 0;
        let mut has_gpu = false;

        if let Some(ref nvml_inst) = nvml {
            if let Ok(count) = nvml_inst.device_count() {
                for i in 0..count {
                    if let Ok(dev) = nvml_inst.device_by_index(i) {
                        if let Ok(mem) = dev.memory_info() {
                            if mem.free > current_vram { current_vram = mem.free; }
                            has_gpu = true;
                        }
                    }
                }
            }
        }

        let meets_vram = !has_gpu || current_vram >= target_vram_bytes;
        let meets_ram = current_ram >= target_ram_bytes;
        
        if meets_vram && meets_ram {
            break; // Perfect state reached
        }

        // [STABILITY-LOGIC] Even if below target, if memory release has stopped changing,
        // it means we've recovered all we can. Don't wait forever.
        let delta = if current_vram > last_vram { current_vram - last_vram } else { last_vram - current_vram };
        if delta < 10_000_000 { // Change < 10MB (more lenient)
            stable_ticks += 1;
        } else {
            stable_ticks = 0;
        }

        // [FAST-EXIT] If stable for 1.5 seconds OR we have at least 600MB free (enough for Embedding/0.6B)
        // This prevents being stuck at 0.7GB when target is 1.1GB.
        if (stable_ticks >= 3 && current_vram > 600_000_000) || current_vram > target_vram_bytes {
            println!("[RESOURCE-WATCH] Memory sufficient or stabilized. Proceeding with {:.2} GB free VRAM.", current_vram as f64 / 1e9);
            break;
        }

        if last_report.elapsed().as_secs() >= 2 { // Faster reporting
            println!("[RESOURCE-DIAG] Waiting... VRAM: {:.2} GB free (Target: {:.2} GB)", 
                current_vram as f64 / 1e9, target_vram_mb as f64 / 1024.0);
            last_report = std::time::Instant::now();
        }

        // Absolute maximum wait 10s (reduced from 20s)
        if start_time.elapsed().as_secs() > 10 {
            println!("[RESOURCE-WATCH] Timeout or sufficient VRAM reached. Proceeding.");
            break;
        }

        last_vram = current_vram;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Ok(())
}
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
use once_cell::sync::OnceCell;

pub static PROGRESS_TX: OnceCell<tokio::sync::mpsc::UnboundedSender<serde_json::Value>> = OnceCell::new();

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
    
    let (ptx, mut prx) = tokio::sync::mpsc::unbounded_channel::<serde_json::Value>();
    let _ = PROGRESS_TX.set(ptx);
    let app_handle_prog = app_handle.clone();
    tokio::spawn(async move {
        use tauri::Emitter;
        while let Some(payload) = prx.recv().await {
            if let Ok(mut w) = crate::LATEST_PROGRESS_PAYLOAD.write() {
                *w = Some(payload.clone());
            }
            let _ = app_handle_prog.emit("extraction-progress", &payload);
        }
    });

    clear_all_temp_data(Some(&app_handle));

    {
        let store_clone = store.clone();
        tauri::async_runtime::spawn(async move {
            for _ in 0..10 { 
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
        if !UI_READY_FLAG.load(Ordering::SeqCst) {
            UI_READY_SIGNAL.notified().await;
        }
        
        let mut delay_secs = 1;
        let mut current_device_pref: Option<String> = None;
        // 🌟 [CRITICAL FIX] 무한 OOM 재시도를 막기 위한 재시도 장부 추가
        let mut oom_retry_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        
        loop {
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
                        let _ = db.update_message_status(&task.id, crate::logic::parse_status("progress"), Some("Processing...")).await;
                    }
                }

                match process_task(task.clone(), &store, &model, &cancellation_token, &app_handle, current_device_pref.clone()).await {
                    Ok(_) => {
                        println!("[Scheduler] Task completed: {}", task.id);
                        
                        // 🌟 [CRITICAL FIX] Task가 끝났을 때 메모리를 None으로 날려버리면 UI가 'Done' 상태를 읽기도 전에 증발해서 스피너가 무한루프를 돕니다.
                        // 상태를 9(Complete)로 업데이트하여 UI가 완료되었음을 확실히 인지하게 만듭니다.
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() { 
                            if let Some(task_val) = w.as_mut() {
                                if let Some(obj) = task_val.as_object_mut() {
                                    obj.insert("status".to_string(), serde_json::json!(9));
                                }
                            }
                        }
                        // LATEST_PROGRESS_PAYLOAD 는 지우지 않고 유지하여 프론트엔드가 마지막 "✅" 상태를 읽어갈 수 있게 둡니다.

                        {
                            let mut model_lock = model.lock().await;
                            if let Some(m) = model_lock.as_ref() {
                                m.deep_purge_resources().await;
                            }
                            *model_lock = None;
                        }
                        
                        let store_guard = store.lock().await;
                        if let Some(db) = store_guard.as_ref() {
                            let _ = db.update_task_status(&task.id, crate::logic::parse_status("complete")).await;
                            let _ = db.update_message_status(&task.id, crate::logic::parse_status("complete"), Some("Task Completed")).await;
                        }

                        cleanup_task_resources(&task.id, Some(&app_handle));
                        current_device_pref = None; 
                        oom_retry_map.remove(&task.id); // 성공 시 장부 삭제
                    },
                    Err(e) => {
                        let err_msg = e.to_string();
                        println!("[Scheduler] Task failed: {:?}. Error: {}", task.id, err_msg);
                        
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() { *w = None; }

                        let task_dir = utils::paths::get_task_specific_dir(Some(&app_handle), &task.id);
                        if !task_dir.exists() { let _ = std::fs::create_dir_all(&task_dir); }
                        let error_file = task_dir.join("error_reason.txt");
                        let _ = std::fs::write(&error_file, format!("Timestamp: {}\nError: {}\n", chrono::Utc::now(), err_msg));

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
                             cleanup_task_resources(&task.id, Some(&app_handle));
                             
                             current_device_pref = None;
                             continue;
                        } else if err_msg.contains("CUDA_ERROR_OUT_OF_MEMORY") || err_msg.contains("out of memory") {
                            let retries = oom_retry_map.entry(task.id.clone()).or_insert(0);
                            
                            if *retries == 0 {
                                *retries += 1;
                                println!("[Scheduler] OOM Detected! VRAM is purged. Retrying on GPU...");
                                current_device_pref = None;

                                // 🌟 [CRITICAL FIX 2] Warning 로그를 장부(파일)에 적지 않고 화면에만 즉시 쏩니다!
                                let payload = json!({
                                    "task_id": task.id,
                                    "category": "Warning", "summary": "Memory pressure detected. VRAM cleared. Retrying on GPU...", "spinner": "♻️"
                                });
                                let _ = app_handle.emit("extraction-progress", &payload);

                                // 🌟 그리고 파일을 삭제하여 다음 시작(Processing)이 100% 깨끗한 1번 스텝이 되게 만듭니다.
                                let log_path = crate::utils::paths::get_task_log_file(Some(&app_handle), &task.id);
                                let _ = std::fs::remove_file(&log_path);
                                
                                {
                                    let store_guard = store.lock().await;
                                    if let Some(db) = store_guard.as_ref() {
                                        let _ = db.update_task_status(&task.id, 10).await;
                                        let _ = db.update_message_status(&task.id, 10, Some("Retrying on GPU...")).await;
                                    }
                                }
                                
                                tokio::time::sleep(Duration::from_secs(2)).await;
                                continue; 
                            } else {
                                if task.r#type == "image_extraction" {
                                    let final_err = "High-resolution image exceeds VRAM capacity. Please try a smaller image.";
                                    println!("[Scheduler] GPU retry failed for Vision. Throwing error instead of freezing on CPU.");
                                    let store_guard = store.lock().await;                            
                                    if let Some(db) = store_guard.as_ref() {
                                        let _ = db.update_task_status(&task.id, crate::logic::parse_status("error")).await;
                                        let _ = db.update_message_status(&task.id, crate::logic::parse_status("error"), Some(&format!("Error: {}", final_err))).await;
                                    }
                                    let _ = app_handle.emit("extraction-progress", json!({
                                        "task_id": task.id,
                                        "category": "Error", "summary": final_err, "spinner": "❌"
                                    }));
                                    current_device_pref = None;
                                } else {
                                    println!("[Scheduler] OOM Detected twice! Activating CPU Mode for text task.");
                                    current_device_pref = Some("cpu".to_string());

                                    // 여기도 더러워진 로그 청소
                                    let log_path = crate::utils::paths::get_task_log_file(Some(&app_handle), &task.id);
                                    let _ = std::fs::remove_file(&log_path);

                                    log_task_progress(&app_handle, &task.id, &json!({
                                        "category": "Warning", "summary": "Memory pressure detected. Retrying with CPU Mode...", "spinner": "💾"
                                    }));
                                    
                                    {
                                        let store_guard = store.lock().await;
                                        if let Some(db) = store_guard.as_ref() {
                                            let _ = db.update_task_status(&task.id, 10).await;
                                            let _ = db.update_message_status(&task.id, 10, Some("Retrying in CPU Mode...")).await;
                                        }
                                    }
                                    
                                    tokio::time::sleep(Duration::from_secs(2)).await;
                                    continue;
                                }
                            }
                        } else {
                            let store_guard = store.lock().await;                            
                            if let Some(db) = store_guard.as_ref() {
                                let _ = db.update_task_status(&task.id, crate::logic::parse_status("error")).await;
                                let _ = db.update_message_status(&task.id, crate::logic::parse_status("error"), Some(&format!("Error: {}", err_msg))).await;
                            }
                            
                            let _ = app_handle.emit("extraction-progress", json!({
                                "task_id": task.id,
                                "category": "Error", "summary": format!("Failed: {}", err_msg), "spinner": "❌"
                            }));

                            current_device_pref = None;
                        }
                    }
                }
            }
            
            cancellation_token.store(false, Ordering::SeqCst);
            crate::utils::set_extraction_stop_signal(false); 
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
    
    // 🌟 [CRITICAL FIX] 비동기 채널과 스레드 꼬임으로 인한 로그 역전 현상 원천 차단!
    let app_handle_clone = app_handle.clone();
    let tid_clone = task.id.clone();
    let emit_term = move |msg: &str| {
        println!("{}", msg);
        use tauri::Emitter;
        let _ = app_handle_clone.emit("task-console-log", serde_json::json!({"task_id": tid_clone, "text": format!("{}\n", msg)}));
    };

    let team_id = if !task.to.is_empty() { task.to.clone() } else { crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") };

    let pug_logs_dir = utils::paths::get_pug_logs_dir(Some(app_handle), &task.id);
    emit_term("\n=======================================");
    emit_term(&format!("[PROCESS] ⚙️ Task {} started processing.", task.id));
    emit_term(&format!("[DEBUG] Pug logs directory: {:?}", pug_logs_dir));

    let kv_path = utils::paths::get_kv_dir(Some(app_handle)).join(&task.id);
    if kv_path.exists() {
        emit_term(&format!("[PROCESS] Found existing KV cache for task {}. Ready to reuse.", task.id));
    }

    // 🌟 [CRITICAL FIX 1] 프론트엔드가 1단계부터 총 스텝 수(분모)를 헷갈리지 않게 task_type을 강제 주입합니다!
    let payload = json!({ 
        "task_id": task.id,
        "task_type": task.r#type, 
        "category": "Processing", "summary": "Starting extraction...", "spinner": "⠋" 
    });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

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
        let search_mode = task_data.get("search_mode").and_then(|s| s.as_str()).unwrap_or("commerce").to_string();

        if !image_path.is_empty() {
            println!("[Scheduler] Starting Image Extraction for {}", task.id);
            
            // 🌟 [CRITICAL FIX 1] 이 녀석이 스텝을 5단계로 부풀리는 주범입니다! 과감히 삭제합니다.
            // log_task_progress(app_handle, &task.id, &json!({ "category": "Vision", "summary": "Analyzing visual context with Qwen 3.5...", "spinner": "⠋" }));
            
            model.extract_from_image(
                task.id.clone(),
                image_path,
                "korean".to_string(),
                search_mode, 
                app_handle,
                Some(cancellation_token.clone()),
                store_mutex,
            ).await?;
            
            return Ok(()); 
        }
    }

    let mut url = task_data.get("href")
        .or_else(|| task_data.get("link"))
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_string();

    let mut origin_candidate = task_data.get("origin")
        .or_else(|| task_data.get("domain"))
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_string();

    // 🌟 [CRITICAL FIX] 도메인이 누락되었거나 상대경로만 들어왔을 경우, 
    // 브라우저 자동화 모듈이 감지한 '진짜 현재 URL'을 강제로 끌어와서 복구합니다!
    if url.starts_with("/") || origin_candidate.is_empty() || origin_candidate.contains("localhost") {
        let state = crate::automation::LAST_DETECTED_STATE.lock().await;
        let real_url = state.url.clone();
        
        if let Ok(parsed) = url::Url::parse(&real_url) {
            let real_origin = format!("{}://{}", parsed.scheme(), parsed.host_str().unwrap_or("localhost"));
            
            if origin_candidate.is_empty() || origin_candidate.contains("localhost") {
                origin_candidate = real_origin.clone();
            }
            if url.starts_with("/") {
                url = format!("{}{}", real_origin, url);
            } else if url.is_empty() {
                url = real_url;
            }
        }
    }

    if url.starts_with("/") && !origin_candidate.is_empty() && !origin_candidate.contains("localhost") {
        let scheme = if origin_candidate.starts_with("http") { "" } else { "http://" };
        url = format!("{}{}{}", scheme, origin_candidate, url);
    }
    
    // 🌟 [CRITICAL FIX] 메모리 덮어쓰기 시 origin을 보존하여 뒤쪽 로직이 도메인을 알 수 있게 합니다.
    let active_task_json = json!({
        "id": task.id.clone(),
        "type": task.r#type.clone(),
        "link": url.clone(),
        "origin": origin_candidate.clone(),
        "status": 1, // Processing
        "created_at": task.created_at,
        "updated_at": chrono::Utc::now().timestamp_millis()
    });
    
    if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() {
        *w = Some(active_task_json.clone());
    }

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
    let mut skip_ai_analysis = false; // 🌟 [핵심] AI 분석 스킵 플래그 추가

    let raw_path = {
        let mut shared_origin = None;
        if let Ok(mem) = crate::ACTIVE_TASK_MEM.read() {
            if let Some(json_val) = mem.as_ref() {
                if let Some(o) = json_val.get("origin").and_then(|v| v.as_str()) {
                    if !o.is_empty() && !o.contains("localhost") {
                        let formatted = if o.starts_with("http") { o.to_string() } else { format!("http://{}", o) };
                        if let Ok(u) = url::Url::parse(&formatted) { 
                            shared_origin = Some(format!("{}://{}", u.scheme(), u.host_str().unwrap_or("localhost"))); 
                        }
                    }
                }
            }
        }
        
        let origin_str = task_data.get("origin")
            .or_else(|| task_data.get("domain"))
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .filter(|s| !s.contains("localhost"))
            .or(shared_origin)
            .unwrap_or_else(|| if let Ok(task_url) = url::Url::parse(&url) { format!("{}://{}", task_url.scheme(), task_url.host_str().unwrap_or("localhost")) } else { "http://localhost".to_string() });

        let base_url = url::Url::parse(&origin_str).unwrap_or_else(|_| url::Url::parse("http://localhost").unwrap());
        let url_obj = base_url.join(&url).unwrap_or(base_url);
        url_obj.path().to_string()
    };

    let page_id = crate::utils::hash::hash_id(&format!("{}{}", task.cc, raw_path));

    {
        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
            if let Ok(Some(page_doc)) = db.get_item_by_id("pages", &page_id).await {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&page_doc.json_data) {
                    let node_sel = val.get("node").and_then(|v| v.as_str()).unwrap_or("");
                    
                    // 🌟 유효성 검증: 캐시된 셀렉터가 현재 웹페이지 HTML에 아직 존재하는가?
                    let clean_html_path = data_manager.get_path("clean_html");
                    if let Ok(clean_content) = data_manager.load(&clean_html_path) {
                        let document = scraper::Html::parse_document(&clean_content);
                        if let Ok(sel) = scraper::Selector::parse(node_sel) {
                            if document.select(&sel).next().is_some() || node_sel == "body" || node_sel.is_empty() {
                                emit_term(&format!("[Scheduler] ⚡ CACHE HIT! Skipping AI Pre-processing for: {}", raw_path));
                                // 🌟 [CRITICAL FIX] trim() 과 to_lowercase() 추가
                                page_type = val.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").trim().to_lowercase(); 
                                is_detail = val.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
                                selector_info = val.clone();
                                skip_ai_analysis = true; // 스킵 활성화!
                                
                                log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Loaded page config from cache.", "spinner": "⚡" }));
                            } else {
                                emit_term("[Scheduler] Cache found but UI changed. Falling back to AI Analysis.");
                            }
                        }
                    }
                }
            }
        }
    }


    // ==================================================================================
    // [ULTRA-OPTIMIZED PIPELINE]
    // Step 0: 0.6B Base Baking [System: PUG] -> Save task_id_base
    // Step 1: 0.6B Loads base -> Ask Classification [User: Task] -> Save task_id_step_a
    // Step 2: 0.6B Loads base -> Ask Selectors [User: Task] -> Save task_id_step_b
    // ==================================================================================

    let base_session_id = format!("{}_base", task.id);
    let base_session_id_35 = format!("{}_base_q35", task.id); // 🌟 0.8B 전용 세션 ID
    let system_content = format!("[PUG CONTENT]\n{}", light_pug);

    // 🌟 [핵심 변경 1] 캐시 적중 시(skip_ai_analysis = true), 무거운 0.6B 분석을 통째로 건너뜁니다!
    if !skip_ai_analysis {
        // --- STEP 0: BASE BAKING (공통 컨텍스트 딱 1번만 굽기) ---
        {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            
            let base_kv_path = utils::paths::get_kv_dir(Some(app_handle)).join(&base_session_id);
            if !base_kv_path.exists() {
                println!("[Scheduler] Baking Base PUG Context to SSD...");
                log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Reading document structure...", "spinner": "⠋" }));
                
                // 🌟 [CRITICAL FIX] is_baking을 true로 전달하여 안 써도 되는 2GB짜리 비전(이미지) 모델 로딩을 강제 차단합니다! (로딩 속도 13초 -> 3초)
                model.secure_vram_relay(crate::model::ModelSize::Qwen, None, Some(cancellation_token.clone()), true, kv_name.clone()).await?;
                
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
            
            // 🌟 [성능 최적화] 파일 읽기/쓰기를 삭제하고 RAM 메모리에 직접 꽂아 넣습니다.
            if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() {
                if let Some(task_val) = w.as_mut() {
                    if let Some(obj) = task_val.as_object_mut() {
                        obj.insert("step".to_string(), json!("Step A (Classification)"));
                        obj.insert("session_id".to_string(), json!(snapshot_id.clone()));
                        obj.insert("kv_path".to_string(), json!(kv_name.clone().unwrap_or_else(|| "tmp/kv/".to_string())));
                    }
                }
            }

            {
                // [핵심] Step A가 아니라 '미리 구워둔 Base' 스냅샷을 불러옵니다!
                model.secure_vram_relay(crate::model::ModelSize::Qwen, Some(&base_session_id), Some(cancellation_token.clone()), false, kv_name.clone()).await?;

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
                    
                    // 🌟 [CRITICAL FIX] 공백 찌꺼기 완벽 제거
                    page_type = type_info.get("type").and_then(|s| s.as_str()).unwrap_or("").trim().to_lowercase();                
                    
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
            
            log_task_progress(app_handle, &task.id, &json!({ "category": "Classification", "summary": "Determining document...", "spinner": "⠋" }));

            let detail_prompt = parsing::is_detail_prompt(&page_type);
            // LLM이 지시사항을 잘 따르도록 래핑
            let task_question = format!("{}\n\n[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think", detail_prompt);
            
            let snapshot_id = format!("{}_step_a2", task.id); // 🌟 q35 접미사 제거

            // 🌟 [CRITICAL FIX] 다시 0.6B(Qwen)로 복구하고, 0.6B의 특권인 미리 구워둔 base_session_id를 전달하여 엄청난 속도 향상을 누립니다!
            model.secure_vram_relay(crate::model::ModelSize::Qwen, Some(&base_session_id), Some(cancellation_token.clone()), false, kv_name.clone()).await?;

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
                model: "qwen".to_string(), // 🌟 qwen 으로 복구
                max_tokens: Some(128),     // JSON 스키마가 길어졌으므로 토큰 길이는 128로 유지
                temperature: Some(0.0), top_p: Some(0.01),
                ..Default::default()
            };

            // 🌟 0.6B 제너레이터(generator)로 복구
            if let Some(gen) = model.generator.lock().await.as_mut() {
                println!("[Scheduler] 0.6B (Qwen) Step A-2: Asking detail classification...");
                let res = gen.generate(
                    params, 
                    Some(cancellation_token.clone()), 
                    Some(snapshot_id.clone()), 
                    kv_name.clone()
                ).await?;
                println!("[DEBUG-SCHED] Step A-2 Raw Response: '{}'", res);
                
                let detail_info = parsing::parse_json_from_llm(&res); 
                
                // 바뀐 프롬프트 스키마 형태 {"goods": {"detail": true}} 에 맞춘 파싱 로직 (그대로 유지)
                is_detail = detail_info
                    .get(&page_type)
                    .and_then(|v| v.get("detail"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                
                // (방어 로직) LLM이 가끔 depth를 무시하고 1차원에 바로 뱉을 경우 대비
                if !is_detail {
                    is_detail = detail_info.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
                }
                    
                println!("[Scheduler] Classified is_detail as: {}", is_detail);
            } else {
                println!("[Scheduler] ERROR: Qwen generator is missing!");
            }
        }
    } // 👈 🌟 [핵심 변경 1 끝] 0.6B 분석 블록 종료

                        
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    model.deep_purge_resources().await;
    wait_for_resources_settled(1200, 800, Some(cancellation_token)).await?;

    let mut extracted_data = json!({});

    // --- PHASE 2 Continue: Detail Extraction (If needed) --- 
    if !is_detail {
        // 🌟 [핵심 변경 2] 캐시가 없을 때만 자바스크립트 엔진(Boa)을 돌리고 DB에 저장합니다.
        if !skip_ai_analysis {
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
                    model.secure_vram_relay(crate::model::ModelSize::Qwen, Some(&base_session_id), Some(cancellation_token.clone()), false, kv_name.clone()).await?;

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
                        model: "qwen".to_string(), max_tokens: Some(256), temperature: Some(0.0), top_p: Some(0.01),
                        ..Default::default()
                    };

                    if let Some(gen) = model.generator.lock().await.as_mut() {
                        println!("[JS-BRIDGE] 1. Requesting titles from LLM (0.6B)...");
                        
                        // 0.6B 모델은 generate_part가 아닌 표준 `generate`를 사용하며, 
                        // 반환값도 구조체가 아닌 단순 String(res) 입니다.
                        let res = gen.generate(
                            params, 
                            Some(cancellation_token.clone()), 
                            Some(snapshot_id.clone()), 
                            kv_name.clone()
                        ).await?;
                        
                        println!("[JS-BRIDGE] LLM Raw Response: '{}'", res);

                        // res.text 가 아닌 res 를 그대로 파싱
                        let title_info = parsing::parse_json_from_llm(&res);
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
                            
                            // 🌟 [CRITICAL FIX] 직계 텍스트만 읽지 않고, <a>, <span> 등 자식 깊숙히 숨어있는 모든 텍스트를 재귀적으로 긁어옵니다!
                            let text = if let Some(element_ref) = scraper::ElementRef::wrap(node) {
                                element_ref.text().collect::<Vec<_>>().join(" ").trim().to_string()
                            } else {
                                String::new()
                            };
                                
                            // 🌟 [CRITICAL FIX 1] Rust에서 DOM의 colspan, rowspan 값을 추출하여 JS로 전달합니다!
                            nodes_json.push(json!({
                                "index": idx,
                                "parentIndex": parent_idx,
                                "tagName": el.name().to_string(),
                                "id": el.id().unwrap_or("").to_string(),
                                "classes": el.attr("class").unwrap_or("").split_whitespace().collect::<Vec<_>>(),
                                "text": text,
                                "colspan": el.attr("colspan").unwrap_or("1"),
                                "rowspan": el.attr("rowspan").unwrap_or("1")
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
                            
                            // 🌟 [CRITICAL FIX 2] TR(행) 비교 시, 하위 TD들의 colspan 총합 구조가 다르면 다른 아이템으로 취급합니다!
                            // (예: 일반 데이터 행 vs colspan=10 인 안내/합계 행)
                            if (nodeA.tagName === 'tr') {
                                const aKids = getChildren(nodeA.index).filter(n => n.tagName === 'td' || n.tagName === 'th');
                                const bKids = getChildren(nodeB.index).filter(n => n.tagName === 'td' || n.tagName === 'th');
                                
                                const aColspan = aKids.reduce((sum, k) => sum + parseInt(k.colspan || '1', 10), 0);
                                const bColspan = bKids.reduce((sum, k) => sum + parseInt(k.colspan || '1', 10), 0);
                                
                                // 두 행의 가로 칸 수(colspan 총합)가 2칸 이상 차이난다면 구조가 아예 다른 것입니다.
                                if (aColspan > 0 && bColspan > 0 && Math.abs(aColspan - bColspan) > 1) {
                                    return 0;
                                }
                            }
                            
                            // 🌟 [CRITICAL FIX 3] TD/TH(셀) 자체를 비교할 때 rowspan/colspan 이 다르면 감점
                            if (nodeA.tagName === 'td' || nodeA.tagName === 'th') {
                                if (nodeA.colspan !== nodeB.colspan || nodeA.rowspan !== nodeB.rowspan) return 0;
                            }

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
                                
                                if (node.tagName === "td" || node.tagName === "th") {
                                    // 🌟 [CRITICAL FIX 4] 현재 셀이 rowspan이나 colspan을 가지고 있다면, 
                                    // 이는 단일 항목이 아니라 복잡한 그리드의 부속품입니다. 묻지도 따지지도 않고 부모(tr)로 올라갑니다.
                                    if (parseInt(node.colspan || '1', 10) > 1 || parseInt(node.rowspan || '1', 10) > 1) {
                                        cur = pIdx;
                                        continue;
                                    }
                                    
                                    const pNode = nodes[pIdx];
                                    if (pNode && pNode.tagName === "tr") {
                                        const gpIdx = pNode.parentIndex; 
                                        if (gpIdx !== undefined && gpIdx !== -1) {
                                            const trSiblings = getChildren(gpIdx);
                                            const similarTrs = trSiblings.filter(s => calculateSimilarity(pNode, s) >= 60);
                                            
                                            // 부모(tr)가 유사한 구조의 다른 형제(tr)들을 여럿 거느리고 있다면 진짜 세로 리스트입니다.
                                            if (similarTrs.length >= 2) {
                                                cur = pIdx;
                                                continue;
                                            }
                                        }
                                    }
                                }

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

            // [PARITY] Store 'Page' Entity (이 구역도 스킵되지 않으면 DB에 저장)
            {
                // Acquire Store lock briefly
                let store = {
                    let store_guard = store_mutex.lock().await;
                    store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
                };
                
                let mut shared_origin = None;
                let mut shared_type = None;
                if let Ok(mem) = crate::ACTIVE_TASK_MEM.read() {
                    if let Some(json_val) = mem.as_ref() {
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

                let origin_str = task_data.get("origin")
                    .or_else(|| task_data.get("domain"))
                    .and_then(|s| s.as_str())
                    .map(|s| s.to_string())
                    .filter(|s| !s.contains("localhost")) 
                    .or(shared_origin) 
                    .unwrap_or_else(|| {
                        if let Ok(task_url) = url::Url::parse(&url) {
                            format!("{}://{}", task_url.scheme(), task_url.host_str().unwrap_or("localhost"))
                        } else {
                            "http://localhost".to_string()
                        }
                    });

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
                    
                    if let Some(item_sel) = selector_info.get("itemSelector") {
                        obj.insert("item".to_string(), item_sel.clone()); 
                    }
                    if let Some(parent_sel) = selector_info.get("parent") {
                        obj.insert("node".to_string(), parent_sel.clone()); 
                    }
                    obj.insert("detail".to_string(), json!(false));
                }

                // 🌟 [CRITICAL FIX] 페이지(pages)를 저장할 때 raw_path가 아닌, items와 동일한 해시값(task.ref)을 사용해 클릭 시 필터가 정확히 물리게 합니다!
                let ref_for_page = if !task.r#ref.is_empty() { &task.r#ref } else { raw_path };
                let _ = store.upsert_item("pages", &page_id, "pages", page_data, None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(ref_for_page), None).await;
            }
        } // 👈 🌟 [핵심 변경 2 끝] JS 선택자 분석 스킵 괄호 닫기!

        // 🌟 [핵심 변경 3] final_page_info 의존성을 제거하고 캐시된 selector_info를 직접 참조합니다.
        let item_selector = selector_info.get("itemSelector")
            .or_else(|| selector_info.get("item"))
            .and_then(|s| s.as_str())
            .unwrap_or("");
        let node_selector = selector_info.get("node").or_else(|| selector_info.get("parent")).and_then(|s| s.as_str()).unwrap_or("");
        
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
            
            // 🌟 [CRITICAL FIX] JS 엔진에서 넘어온 2차원 배열(Vec<Vec<String>>)을 완벽히 파싱합니다.
            let mut headers: Vec<Vec<String>> = selector_info.get("headers")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter().filter_map(|row_val| {
                        row_val.as_array().map(|r| r.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                    }).collect()
                })
                .unwrap_or_default();

            // 호환성 방어: 캐시에 남아있는 구형 1차원 배열 데이터가 있을 경우 2차원으로 래핑해줍니다.
            if !headers.is_empty() && headers[0].is_empty() {
                let flat_headers: Vec<String> = selector_info.get("headers")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                    .unwrap_or_default();
                if !flat_headers.is_empty() {
                    headers = vec![flat_headers];
                }
            }

            // Rust 백엔드의 순수 추출기(extract_table_headers)도 원래 2차원 배열을 반환하므로 그대로 수용합니다.
            if headers.is_empty() {
                let rust_headers = parsing::extract_table_headers(&clean_content, &target_selector);
                if !rust_headers.is_empty() {
                    headers = rust_headers; 
                }
            }

            if !headers.is_empty() {
                println!("[Scheduler] Extracted 2D header map (Rows: {}) for 'alt' mapping.", headers.len());
            }
            
            let document = scraper::Html::parse_document(&clean_content);
            
            parsing::split_doc_to_pug_list_advanced(
                &document, 
                &target_selector, 
                PugMode::FullContent, 
                if headers.is_empty() { None } else { Some(headers) } // vec![] 래핑 제거!
            )
        };

        if !pug_list.is_empty() {
            // 🌟 [CRITICAL FIX 2] 리스트 추출은 짧은 item_pug 조각만 독립적으로 읽으므로, 
            // 무겁고 호환되지 않는 Base 스냅샷 로딩을 원천 차단합니다.
            model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancellation_token.clone()), false, Some("inference".to_string())).await?;

            for (idx, item_pug) in pug_list.iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { break; }
                
                // 🌟 [CRITICAL FIX] 리스트 아이템 추출 진행률(%) 계산 및 UI/터미널 전송
                let percent = (((idx + 1) as f32 / pug_list.len() as f32) * 100.0) as i32;
                let summary_msg = format!("Extracting item data ({}%)...", percent);
                
                let payload = json!({ 
                    "task_id": task.id, 
                    "category": format!("List Extraction ({}/{})", idx + 1, pug_list.len()), 
                    "summary": summary_msg, 
                    "spinner": "⠋" 
                });
                // 🌟 [CRITICAL FIX] 상태 장부(log_task_progress)에 확실하게 기록해야 
                // AI가 디코딩 퍼센트를 올바른 "List Extraction" 카테고리로 쏩니다!
                log_task_progress(app_handle, &task.id, &payload);
                emit_term(&format!("[STAGE-3] {}", summary_msg));

                let extraction_instruction = parsing::list2json(&page_type, &url, language);
                let task_question = format!("[PUG CONTENT]\n{}\n\n{}", item_pug, extraction_instruction);
                
                // ==================================================================
                // 🌟 [DEBUG] 회원님 요청: 각 루프마다 LLM에 들어가는 100% 날것의 Context를 텍스트 파일로 박제합니다.
                // ==================================================================
                let debug_file_path = pug_logs_dir.join(format!("debug_context_item_{}.txt", idx + 1));
                let _ = std::fs::write(&debug_file_path, &task_question);
                println!("\n[DEBUG-CONTEXT] 📝 Item {}/{} 의 Context가 저장되었습니다: {:?}", idx + 1, pug_list.len(), debug_file_path);
                println!("[DEBUG-CONTEXT] 🔍 텍스트 길이: {} 글자", task_question.len());
                // ==================================================================
                
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
                        let mut parsed_json = parsing::parse_json_from_llm(&res_text);
                        
                        // 🌟 [CRITICAL FIX] LLM이 키값에 공백을 넣거나 대소문자를 섞었을 때(예: {"Goods ": ...})를 완벽히 방어합니다.
                        let mut target_key = page_type.clone();
                        if parsed_json.get(&target_key).is_none() {
                            if let Some(obj) = parsed_json.as_object() {
                                for k in obj.keys() {
                                    if k.trim().to_lowercase() == page_type {
                                        target_key = k.clone();
                                        break;
                                    }
                                }
                            }
                        }

                        // 찾아낸 가장 정확한 target_key로 껍데기(Outer Shell)를 벗겨냅니다.
                        let mut item_json = if let Some(inner) = parsed_json.get_mut(&target_key) {
                            inner.take() // 알맹이 적중 시 꺼냄
                        } else {
                            parsed_json // 방어 로직: LLM이 껍데기 없이 바로 뱉었을 경우 그대로 사용
                        };

                        if !item_json.is_null() && (item_json.is_object() || item_json.is_array()) {
                            
                            // (이전에 있던 {TYPE}_status 를 status로 롤백하던 코드는 이제 필요 없으므로 완전 삭제!)

                            if let Some(link_val) = item_json.get_mut("link") {
                                if let Some(relative_path) = link_val.as_str() {
                                    if let Ok(base_url) = url::Url::parse(&url) {
                                        if let Ok(absolute_url) = base_url.join(relative_path) {
                                            *link_val = json!(absolute_url.to_string());
                                        }
                                    }
                                }
                            }
                            
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


                // 1. 모델 내부에 쌓인 과거 문맥(KV 캐시) 명시적 파괴 및 GPU 파이프라인 강제 동기화
                if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
                    gen.clear_kv_cache();
                }
                
                // 🌟 [추가] GPU에 남아있는 비동기 연산 찌꺼기까지 완벽하게 털어내기
                if !model.is_cpu_mode {
                    let dev = model.device_config.device.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        if dev.is_cuda() { let _ = dev.synchronize(); }
                    }).await;
                }

                // 2. IO 작업이 꼬이지 않도록 대기
                crate::models::qwen::generate::wait_for_global_io().await;

                // 3. OS 커널 레벨에서 가비지 컬렉터(Garbage Collector)를 강제 호출하여 RAM 피크를 박살 냅니다.
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

                // 4. GPU와 OS가 메모리를 반환할 시간을 아주 짧게(0.1초) 줍니다.
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
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
                // 🌟 [CRITICAL FIX 3] 디테일 모드에서도 0.6B Base 스냅샷 로드 시도를 완벽히 끊어버립니다. 
                model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancellation_token.clone()), false, kv_name.clone()).await?;

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
                    
                    // 🌟 [CRITICAL FIX] 줄이 나뉘지 않도록 카테고리를 통합하고 문구를 부드럽게 이어줍니다.
                    let payload = json!({ "task_id": task.id, "category": "AI Inference", "summary": "Preparing AI engine...", "spinner": "⠋" });
                    let _ = app_handle.emit("extraction-progress", &payload);
                    emit_term("[STAGE-3] Preparing AI engine...");
                    
                    // 🌟 generate_part 에 None 대신 Some(snapshot_id.clone()) 전달
                    let res = gen.generate_part(&params, false, 0, None, Some(snapshot_id.clone()), kv_name.clone()).await?;
                    
                    println!("[DEBUG-SCHED] Step C Raw Response: '{}'", res.text);

                    // [DEBUG] AI 응답 저장
                    let _ = data_manager.offload(&res.text, "step_c_res");

                    let mut parsed_json = parsing::parse_json_from_llm(&res.text);
                    
                    // 🌟 [CRITICAL FIX] 디테일 모드에서도 키(Key) 공백/대소문자 오염 방어 로직 동일하게 적용
                    let mut target_key = page_type.clone();
                    if parsed_json.get(&target_key).is_none() {
                        if let Some(obj) = parsed_json.as_object() {
                            for k in obj.keys() {
                                if k.trim().to_lowercase() == page_type {
                                    target_key = k.clone();
                                    break;
                                }
                            }
                        }
                    }

                    extracted_data = if let Some(inner) = parsed_json.get_mut(&target_key) {
                        inner.take() // 알맹이 적중 시 꺼냄
                    } else {
                        parsed_json // 방어 로직
                    };

                } else {
                    return Err(anyhow::anyhow!("Qwen 3.5 Generator not available"));
                }
            }
        }
    }

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // --- PHASE 3: HANDOVER (Unload -> Load Embedding) ---
    {
        println!("[Scheduler] PHASE 3: Handover - Unloading, Preparing for Embedding...");
        // 🌟 스피너(⠋) 속성 추가
        log_task_progress(app_handle, &task.id, &json!({ "category": "Handover", "summary": "Switching to Embedding model...", "spinner": "⠋" }));
        
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
    let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
    let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_val));
    let ref_val = task.r#ref.clone();

    let mut items_to_process = Vec::new();
    let mut is_new_draft_global = true; 

    if is_detail {
        // 🌟 [DETAIL MODE] 단일 문서일 경우 기존처럼 통째로 저장합니다.
        let text_to_embed = parsing::json_to_natural_language(&extracted_data);
        let item_digest = crate::utils::hash::digest(&text_to_embed); 
        let target_id = if !task.r#ref.is_empty() { task.r#ref.clone() } else { generated_id.clone() }; 
        
        let mut existing_vector = None;
        if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &target_id).await {
            is_new_draft_global = false; 
            if existing_item.digest == item_digest {
                existing_vector = Some(existing_item.vector);
            }
        }

        let vector = if let Some(v) = existing_vector {
            Some(v)
        } else {
            Some(model.get_embedding(text_to_embed).await?)
        };

        let _ = store.upsert_item(
            &target_table, &target_id, &page_type, extracted_data.clone(), vector.clone(),
            Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)
        ).await;
        
        let _ = store.upsert_item(
            "items", &target_id, &page_type, extracted_data.clone(), vector,
            Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)
        ).await;

        items_to_process.push(extracted_data.clone());
        
    } else {
        // 🌟 [CRITICAL FIX] [LIST MODE] items 배열을 순회하며 낱개로 쪼개어 각각 DB에 독립된 문서로 저장합니다!
        if let Some(items) = extracted_data.get("items").and_then(|v| v.as_array()) {
            // 🌟 [CRITICAL FIX] 리스트 전처리 항목은 무조건 Draft(대기) 상태로 집계해야 프론트엔드에서 'Draft (13)' 처럼 정상 표기됩니다!
            is_new_draft_global = true; 
            
            for item_val in items.iter() {
                let mut single_item = item_val.clone();
                
                // 1. 개별 아이템만의 고유 식별자(ID) 생성 (상품번호나 링크 기반)
                let original_id = single_item.get("id").or_else(|| single_item.get("no")).and_then(|v| v.as_str())
                    .unwrap_or_else(|| single_item.get("link").and_then(|v| v.as_str()).unwrap_or(""));
                
                let clean_no = crate::utils::hash::normalize_numeric_homoglyphs(original_id).replace("-", "").replace("_", "");
                let index_val = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("{}{}{}", page_type, team_id, clean_no)));
                let hashed_item_id = if original_id.is_empty() {
                    crate::utils::hash::hash_id(&format!("{}{}", team_id, uuid::Uuid::new_v4()))
                } else {
                    crate::utils::hash::hash_id(&format!("{}{}", team_id, index_val))
                };

                // 2. 부모 페이지의 메타데이터(type, detail)를 개별 아이템에 주입
                if let Some(obj) = single_item.as_object_mut() {
                    obj.insert("type".to_string(), json!(page_type));
                    obj.insert("detail".to_string(), json!(false));
                    obj.insert("id".to_string(), json!(hashed_item_id.clone()));
                    obj.insert("index".to_string(), json!(index_val));
                }

                let text_to_embed = parsing::json_to_natural_language(&single_item);
                let item_digest = crate::utils::hash::digest(&text_to_embed);
                
                // 3. 중복 검사 및 벡터 임베딩 추출
                let mut existing_vector = None;
                if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &hashed_item_id).await {
                    if existing_item.digest == item_digest {
                        existing_vector = Some(existing_item.vector);
                    }
                }
                
                let vector = if let Some(v) = existing_vector {
                    Some(v)
                } else {
                    Some(model.get_embedding(text_to_embed).await?)
                };

                // 4. DB에 낱개 단위로 저장
                let _ = store.upsert_item(
                    &target_table, &hashed_item_id, &page_type, single_item.clone(), vector.clone(),
                    Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)
                ).await;
                
                let _ = store.upsert_item(
                    "items", &hashed_item_id, &page_type, single_item.clone(), vector,
                    Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)
                ).await;

                items_to_process.push(single_item);
            }
        }
    }

    if !items_to_process.is_empty() {
        // 🌟 추출된 낱개 데이터들로 통계(Base Metrics) 엔진 갱신!
        let _ = update_team_base_metrics(&store, &team_id, &task.cc, &page_type, &items_to_process, is_new_draft_global).await;
        println!("[PROCESS] Metrics Engine updated base statistics for {} items.", items_to_process.len());
    }

    // Final Status Update
    let _ = store.update_message_status(&task.id, logic::parse_status("complete"), Some("Extraction Complete")).await;

    // 🌟 [CRITICAL FIX] 불완전한 추출 데이터를 프론트로 직접 쏘지 않습니다.
    // 대신 프론트엔드가 이 'Done' 신호를 받고 내부적으로 app.fetch()를 트리거하여
    // DB에서 완벽하게 세팅된(id, ref, bcc 등) 데이터를 가져가도록 유도해야 합니다.
    let payload = json!({
        "task_id": task.id, 
        "category": "Done", 
        "summary": "Extraction complete. Updating list...", 
        "spinner": "✅",
        // data를 null로 보내어 프론트엔드가 기존에 그리던 캐시를 초기화하도록 합니다.
        "data": null 
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
    use tauri::Emitter;

    // 🌟 [CRITICAL FIX] 백엔드에서 쏘는 모든 익명 로그에 task_id를 강제 주입하여 프론트엔드 오작동 차단!
    let mut final_payload = payload.clone();
    if let Some(obj) = final_payload.as_object_mut() {
        obj.insert("task_id".to_string(), serde_json::json!(task_id));
    }

    let log_path = crate::utils::paths::get_task_log_file(Some(app), task_id);
    
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path) 
    {
        let line = format!("{}\n", final_payload.to_string());
        let _ = file.write_all(line.as_bytes());
    }

    if let Some(cat) = final_payload.get("category").and_then(|v| v.as_str()) {
        if let Ok(mut w) = crate::CURRENT_UI_CATEGORY.write() {
            *w = cat.to_string();
        }
    }
    if let Ok(mut w) = crate::LATEST_PROGRESS_PAYLOAD.write() {
        *w = Some(final_payload.clone());
    }

    let _ = app.emit("extraction-progress", &final_payload);
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


async fn update_team_base_metrics(
    store: &crate::store::VectorStore,
    team_id: &str,
    task_cc: &str,
    item_type: &str,
    items: &Vec<serde_json::Value>,
    is_new_draft: bool,
) -> anyhow::Result<()> {
    let (team_json_str, team_vector, t_from, t_to, t_cc, t_bcc, t_ref, t_digest) = match store.get_item_by_id("users", team_id).await {
        Ok(Some(doc)) => (doc.json_data, doc.vector, doc.from, doc.to, doc.cc, doc.bcc, doc.r#ref, doc.digest),
        _ => (
            json!({ "base": { "pages": {} } }).to_string(),
            vec![0.0; 768],
            "".to_string(), "".to_string(), "".to_string(), "".to_string(), "".to_string(), "".to_string()
        )
    };

    let mut team_data: serde_json::Value = serde_json::from_str(&team_json_str).unwrap_or(json!({ "base": { "pages": {} } }));

    // 🌟 [CRITICAL FIX] 블록({ })을 사용하여 가변 참조(Mutable Borrow)의 생명주기를 분리합니다!
    
    // --- [블록 1: 페이지별 통계 업데이트] ---
    {
        let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
        let pages = base.entry("pages").or_insert(json!({})).as_object_mut().unwrap();
        let cc_node = pages.entry(task_cc).or_insert(json!({})).as_object_mut().unwrap();
        let page_type_node = cc_node.entry(item_type).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();

        let items_count = items.len() as i64; // 🌟 [핵심] 실제 추출된 아이템 개수

        if is_new_draft {
            let draft = page_type_node.get("draft").and_then(|v| v.as_i64()).unwrap_or(0);
            page_type_node.insert("draft".to_string(), json!(draft + items_count)); // 1 대신 items_count 추가
        } else {
            let draft = page_type_node.get("draft").and_then(|v| v.as_i64()).unwrap_or(0);
            let count = page_type_node.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            if draft > 0 { page_type_node.insert("draft".to_string(), json!(draft - 1)); }
            page_type_node.insert("count".to_string(), json!(count + items_count)); // 1 대신 items_count 추가
        }
    } 

    // --- [블록 2: 글로벌 전체 통계 및 Min/Max 업데이트] ---
    {
        let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
        let global_type_node = base.entry(item_type).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();
        
        let items_count = items.len() as i64; // 🌟 [핵심] 실제 추출된 아이템 개수

        if !is_new_draft {
            let global_count = global_type_node.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            global_type_node.insert("count".to_string(), json!(global_count + items_count)); // 1 대신 items_count 추가
        }

        let properties = [
            "price", "quantity", "width", "height", "length", "weight", "shipping_fee", 
            "shipping_duration", "sale_price", "supply_price", "low_stock_threshold", 
            "discount", "min_order_amount", "max_discount_amount", "usage_limit", 
            "usage_per", "started_at", "expired_at"
        ];

        for item in items {
            for prop in properties.iter() {
                if let Some(val) = item.get(*prop) {
                    let num_val = if val.is_number() {
                        val.as_f64().unwrap_or(0.0)
                    } else if let Some(s) = val.as_str() {
                        s.parse::<f64>().unwrap_or(0.0)
                    } else {
                        continue;
                    };

                    if num_val == 0.0 && *prop != "started_at" && *prop != "expired_at" { continue; }

                    let prop_node = global_type_node.entry(*prop).or_insert(json!({ "min": 99999999999999.0, "max": -99999999999999.0 })).as_object_mut().unwrap();
                    
                    let current_min = prop_node.get("min").and_then(|v| v.as_f64()).unwrap_or(99999999999999.0);
                    let current_max = prop_node.get("max").and_then(|v| v.as_f64()).unwrap_or(-99999999999999.0);

                    if num_val < current_min { prop_node.insert("min".to_string(), json!(num_val)); }
                    if num_val > current_max { prop_node.insert("max".to_string(), json!(num_val)); }
                }
            }
        }
    } // 👈 여기서 두 번째 참조가 종료됩니다.

    // 5. Save back to DB
    let _ = store.upsert_item(
        "users", 
        team_id, 
        "team", 
        team_data, 
        Some(team_vector),
        Some(&t_from),
        Some(&t_to),
        Some(&t_cc),
        Some(&t_bcc),
        Some(&t_ref),
        Some(&t_digest)
    ).await;

    Ok(())
}
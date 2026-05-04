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

    // 🌟 [삭제] 이미 lib.rs 의 setup 블록에서 동기적으로 정리가 완료되었으므로, 
    // 여기서 다시 spawn 하여 불필요한 DB 락 경쟁을 일으킬 필요가 없습니다.
    
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
                        Ok(tasks) => {
                            // 🌟 [CRITICAL FIX] 스케줄러는 프론트엔드가 큐로 통제하는 검색 작업(ai_search)을 절대 훔쳐가지 않습니다!
                            pending_tasks = tasks.into_iter().filter(|t| t.r#type != "ai_search").collect();
                        },
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
                        // DB의 상태값만 안전하게 1(Processing)로 동기화합니다.
                        let _ = db.update_task_status(&task.id, 1).await;
                        let _ = db.update_message_status(&task.id, 1, Some("Processing...")).await;
                        
                        // 🌟 [CRITICAL FIX] 프론트엔드의 중복 대기열 방어 로직(check_active_task)이 정상 작동하도록 ref도 함께 저장합니다.
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() {
                            *w = Some(json!({ "id": task.id, "ref": task.r#ref, "status": 1 }));
                        }
                    }
                }

                match process_task(task.clone(), &store, &model, &cancellation_token, &app_handle, current_device_pref.clone()).await {
                    Ok(_) => {
                        println!("[Scheduler] Task completed: {}", task.id);
                        
                        // 🌟 [CRITICAL FIX] 메모리의 상태를 9(Complete)로 업데이트하여 UI가 완료되었음을 확실히 인지하게 만듭니다.
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() { 
                            if let Some(task_val) = w.as_mut() {
                                if let Some(obj) = task_val.as_object_mut() {
                                    obj.insert("status".to_string(), json!(9));
                                    obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                }
                            }
                        }

                        // 일정 시간 뒤에 메모리를 비워주거나, 다음 작업 시작 시 덮어씌워지도록 유지합니다.

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

                        current_device_pref = None; 
                        oom_retry_map.remove(&task.id); // 성공 시 장부 삭제
                    },
                    Err(e) => {
                        let err_msg = e.to_string();
                        println!("[Scheduler] Task failed: {:?}. Error: {}", task.id, err_msg);
                        
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() { *w = None; }

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

    emit_term("\n=======================================");
    emit_term(&format!("[PROCESS] ⚙️ Task {} started processing.", task.id));

    // 🌟 [추가] Analytic 작업일 경우 별도의 파이프라인으로 위임(Delegate)하여 처리합니다.
    if task.r#type == "analytic_extraction" {
        return crate::analytic::process_analytic_task(
            task, store_mutex, model_mutex, cancellation_token, app_handle, device_preference
        ).await;
    }

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
    
    // 🌟 [CRITICAL FIX] 메모리 덮어쓰기 시 origin과 ref를 보존하여 프론트엔드 중복 노출 방어 로직이 도메인을 알 수 있게 합니다.
    let active_task_json = json!({
        "id": task.id.clone(),
        "type": task.r#type.clone(),
        "link": url.clone(),
        "origin": origin_candidate.clone(),
        "ref": task.r#ref.clone(),
        "status": 1, 
        "created_at": task.created_at,
        "updated_at": chrono::Utc::now().timestamp_millis()
    });
    
    if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() {
        *w = Some(active_task_json.clone());
    }

    // 🌟 [CRITICAL FIX] URL이 없어서 처리가 불가능한 경우 조용히 성공 처리하지 않고 명시적 에러를 던져 UI의 스피너를 중단시킵니다.
    if url.is_empty() { 
        return Err(anyhow::anyhow!("Task missing target URL or unsupported type for background extraction.")); 
    }

    // [MEMORY] Fetch and process directly in memory
    let raw_html_content = if let Some(raw_html) = task_data.get("html").and_then(|s| s.as_str()) {
        let content = raw_html.to_string();
        if let Some(obj) = task_data.as_object_mut() { obj.remove("html"); }
        content
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

        if needs_euc && malformed_utf8 {
            // Only use EUC-KR if it's explicitly requested AND UTF-8 decoding had issues
            let (decoded_euc, _, _) = encoding_rs::EUC_KR.decode(&bytes);
            decoded_euc.into_owned()
        } else {
            // Default to UTF-8 (Lossy fallback if needed)
            utf8_str.to_string()
        }
    } else {
        return Ok(());
    };

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let clean_html_content = parsing::pre_clean_html(&raw_html_content);
    let raw_pug = parsing::convert_to_clean_pug(&clean_html_content, PugMode::FullContent);
    let light_pug = model.truncate_pug_context(&raw_pug, false, 2000, None).await;

    println!("[DEBUG-PUG] Generated PUG. Length: {}. Snippet: {}...", 
        light_pug.len(), 
        light_pug.chars().take(100).collect::<String>().replace("\n", " ")
    );


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
                    let item_sel = val.get("item").or_else(|| val.get("itemSelector")).and_then(|v| v.as_str()).unwrap_or("");
                    let head_sel = val.get("head").and_then(|v| v.as_str()).unwrap_or("");
                    
                    // 🌟 유효성 검증: 캐시된 부모(node), 아이템(item), 헤더(head) 셀렉터가 현재 웹페이지 구조에 온전히 존재하는가?
                    let clean_content = &clean_html_content;
                    {
                        let document = scraper::Html::parse_document(clean_content);
                        let mut is_valid = true;

                        if !node_sel.is_empty() && node_sel != "body" {
                            if let Ok(sel) = scraper::Selector::parse(node_sel) {
                                if document.select(&sel).next().is_none() { is_valid = false; }
                            } else { is_valid = false; }
                        }

                        if is_valid && !item_sel.is_empty() {
                            let target_sel_str = if !node_sel.is_empty() && !item_sel.contains(",") {
                                if item_sel.starts_with(node_sel) {
                                    item_sel.to_string()
                                } else {
                                    format!("{} {}", node_sel, item_sel)
                                }
                            } else {
                                item_sel.to_string()
                            };
                            
                            // 🌟 [CRITICAL FIX] E0597 해결: 에러 객체가 target_sel_str의 참조를 물고 늘어지지 않도록, 
                            // 즉시 boolean으로 매핑하여 Result 객체를 그 줄에서 완전 소멸시킵니다.
                            let is_match_found = scraper::Selector::parse(&target_sel_str)
                                .map(|sel| document.select(&sel).next().is_some())
                                .unwrap_or(false);
                                
                            if !is_match_found { is_valid = false; }
                        }

                        if is_valid && !head_sel.is_empty() && head_sel != "..." {
                            if let Ok(sel) = scraper::Selector::parse(head_sel) {
                                if document.select(&sel).next().is_none() { is_valid = false; }
                            } else { is_valid = false; }
                        }

                        if is_valid {
                            emit_term(&format!("[Scheduler] ⚡ CACHE HIT! Selectors validated. Skipping AI Pre-processing for: {}", raw_path));
                            page_type = val.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").trim().to_lowercase();
                            is_detail = val.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
                            selector_info = val.clone();
                            skip_ai_analysis = true; // 스킵 활성화!
                            
                            log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Loaded valid config from cache.", "spinner": "⚡" }));
                        } else {
                            emit_term("[Scheduler] Cache found but UI changed (Selector mismatch). Falling back to AI Analysis.");
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
                
                if let Some(gen) = model.generator.lock().await.as_mut() {
                    // 🌟 [CRITICAL FIX] ChatTemplate을 거치지 않고, 후속 질문과 100% 동일한 접두사(Prefix)를 생성하여 굽습니다!
                    // 이렇게 해야 f_ids[kv_len..] 슬라이싱 시 토큰이 엇갈려 환각(Hallucination)이 발생하는 것을 원천 차단할 수 있습니다.
                    let raw_system_prefix = format!("<|im_start|>system\n{}<|im_end|>\n", system_content);
                    
                    // System 메시지(PUG)만 1만 토큰을 읽어서 base_session_id 로 저장합니다.
                    gen.prefill_only(raw_system_prefix, Some(cancellation_token.clone()), Some(base_session_id.clone()), None, kv_name.clone()).await?;
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
                    temperature: Some(0.0), top_p: Some(0.95),
                    ..Default::default()
                };

                if let Some(gen) = model.generator.lock().await.as_mut() {
                    println!("[Scheduler] 0.6B Step A: Asking classification question...");
                    let res = gen.generate(params, Some(cancellation_token.clone()), Some(snapshot_id.clone()), kv_name.clone()).await?;
                    println!("[DEBUG-SCHED] Step A Raw Response: '{}'", res);
                    
                    let type_info = parsing::parse_json_from_llm(&res); 
                    
                    // 🌟 [CRITICAL FIX 복구] AI가 뱉어낸 값의 공백 및 대소문자 오염 방어!
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
                temperature: Some(0.0), top_p: Some(0.95),
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

                // println!("title_prompt {}", title_prompt);

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
                        model: "qwen".to_string(), max_tokens: Some(128), temperature: Some(0.0), top_p: Some(0.95),
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
                        
                        // 🌟 [CRITICAL FIX] 파싱 결과가 실패하여 빈 깡통({})이 반환되었다면 즉시 에러를 던져 작업을 중단합니다.
                        if title_info.as_object().map_or(true, |obj| obj.is_empty()) {
                            return Err(anyhow::anyhow!("LLM returned invalid or unparseable JSON response during title extraction."));
                        }

                        let items_opt = title_info.get("order")
                            .or(title_info.get("goods"))
                            .or(title_info.get("title"))
                            .or(title_info.get("titles"))
                            .or(title_info.get("product"))
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
                    // 🌟 [CRITICAL FIX] 쓸데없는 태그까지 전부 긁어오는 불상사를 막기 위해 폴백(Fallback)으로 진행하지 않고 확실하게 끊어냅니다.
                    return Err(anyhow::anyhow!("[JS-BRIDGE] No titles extracted from LLM. Aborting task to prevent invalid DOM fallback."));
                }

                // 2. Boa Engine으로 DOM 분석
                {
                    println!("[JS-BRIDGE] 2. Starting boa-engine for DOM analysis...");
                    let mut context = Context::default();
                    
                    let document = scraper::Html::parse_document(&clean_html_content);
                    
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
        } // 👈 🌟 [핵심 변경 2 끝] JS 선택자 분석 스킵 괄호 닫기!

        // 🌟 [CRITICAL FIX 1] E0425 해결: 타겟 선택자(target_selector)를 가장 먼저 정의합니다.
        let item_selector = selector_info.get("itemSelector")
            .or_else(|| selector_info.get("item"))
            .and_then(|s| s.as_str())
            .unwrap_or("");
        let node_selector = selector_info.get("node").or_else(|| selector_info.get("parent")).and_then(|s| s.as_str()).unwrap_or("");
        
        let target_selector = if !node_selector.is_empty() && !item_selector.is_empty() && !item_selector.contains(",") {
            if item_selector.starts_with(node_selector) {
                item_selector.to_string()
            } else {
                format!("{} {}", node_selector, item_selector) 
            }
        } else if !item_selector.is_empty() { 
            item_selector.to_string() 
        } else { 
            node_selector.to_string() 
        };

        let mut final_thead_selector = String::new();
        let mut cache_updated = false; // DB 업데이트가 필요한지 추적하는 플래그
        let mut thead_pug = String::new();

        // 1. 사용자 요청대로 'head' 키로 캐시된 선택자가 있는지 확인합니다.
        if let Some(sel) = selector_info.get("head").and_then(|v| v.as_str()) {
            if !sel.is_empty() && sel != "..." {
                final_thead_selector = sel.to_string();
                println!("[Scheduler] Using cached head selector: {}", final_thead_selector);
            }
        } 
        
        // 2. 캐시가 없거나 비어있는 경우 AI를 통해 테이블 헤더 구조를 분석합니다.
        if final_thead_selector.is_empty() {
            // Document를 다시 생성하여 안전하게 target_selector 기반으로 샘플 첫 행(ref_row)을 뽑아냅니다.
            let reference_row_for_thead = {
                let clean_content = &clean_html_content;
                let document = scraper::Html::parse_document(clean_content);
                if let Ok(sel) = scraper::Selector::parse(&target_selector) {
                    document.select(&sel).next().map(|first_match| {
                        let mut temp_pug = String::new();
                        crate::parsing::generate_pug_lines((*first_match).into(), 0, &mut temp_pug, &PugMode::FullContent, &mut None);
                        temp_pug.trim().to_string()
                    })
                } else { None }
            };

            if let Some(ref_row) = reference_row_for_thead {
                if !ref_row.is_empty() {
                    log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Analyzing table header structure...", "spinner": "⠋" }));
                    
                    // 🌟 [추가] ref_row의 텍스트 길이를 기반으로 대략적인 토큰을 산출하여 컨텍스트 사이즈를 예약하고 뒤에서 자릅니다.
                    let ref_row_context_size = ref_row.len() + 1000;
                    let thead_light_pug = model.truncate_pug_context(&raw_pug, false, 2000, Some(ref_row_context_size)).await;
                    let thead_prompt = crate::parsing::extract_table_structure_prompt(&page_type, &target_selector, &thead_light_pug, &ref_row);
                    let params = ChatCompletionParameters {
                        messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                            content: ChatCompletionRequestUserMessageContent::Text(thead_prompt),
                            name: None,
                        })],
                        model: "qwen3.5".to_string(),
                        max_tokens: Some(256), 
                        temperature: Some(0.0), 
                        top_p: Some(0.95),
                        ..Default::default()
                    };

                    model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancellation_token.clone()), false, kv_name.clone()).await?;

                    if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
                        if let Ok(res) = gen.generate(params, Some(cancellation_token.clone()), Some(format!("{}_step_thead", task.id)), kv_name.clone()).await {
                            let thead_json = crate::parsing::parse_json_from_llm(&res);
                            
                            // JSON 응답에서 page_type에 맞는 선택자 추출
                            let mut thead_val = thead_json.get(&page_type);
                            if thead_val.is_none() {
                                if let Some(obj) = thead_json.as_object() {
                                    for (k, v) in obj {
                                        if k.to_lowercase() == page_type.to_lowercase() { thead_val = Some(v); break; }
                                    }
                                }
                            }

                            final_thead_selector = thead_val
                                .and_then(|v| v.get("table"))
                                .and_then(|v| v.get("thead"))
                                .and_then(|v| v.get("selector"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("").to_string();
                            
                            if !final_thead_selector.is_empty() && final_thead_selector != "..." {
                                selector_info.as_object_mut().unwrap().insert("head".to_string(), json!(final_thead_selector.clone()));
                                println!("[Scheduler] AI determined head selector and cached: {}", final_thead_selector);
                                cache_updated = true; // 새로운 head를 찾았으므로 DB 업데이트 예약
                            }
                        }
                    }
                }
            }
        }

        // 3. 최종 결정된 selector를 사용하여 head PUG를 추출합니다.
        if !final_thead_selector.is_empty() && final_thead_selector != "..." {
            let clean_content = &clean_html_content;
            let doc = scraper::Html::parse_document(clean_content);
            if let Ok(tsel) = scraper::Selector::parse(&final_thead_selector) {
                if let Some(first_match) = doc.select(&tsel).next() {
                    // 🌟 [구조 보존 로직] 매칭된 요소가 th나 td일 경우, 다중 tr 계층 구조를 잃지 않기 위해 DOM 트리를 거슬러 올라가 최상위 thead(또는 tr) 블록 전체를 통째로 가져옵니다.
                    let mut target_node = first_match;
                    let mut current = target_node.parent();
                    
                    while let Some(parent) = current {
                        if let Some(el) = parent.value().as_element() {
                            let tag = el.name().to_lowercase();
                            if tag == "thead" || tag == "tr" {
                                if let Some(wrapped) = scraper::ElementRef::wrap(parent) {
                                    target_node = wrapped;
                                    // thead를 찾으면 가장 완벽한 다중 행 헤더 그룹이므로 즉시 탐색 종료
                                    if tag == "thead" { break; } 
                                }
                            }
                        }
                        current = parent.parent();
                    }
                    
                    let mut tpug = String::new();
                    crate::parsing::generate_pug_lines((*target_node).into(), 0, &mut tpug, &PugMode::FullContent, &mut None);
                    thead_pug = tpug.trim().to_string();
                    
                    if !thead_pug.is_empty() {
                        println!("[Scheduler] 🎉 thead_pug extraction successful ({} bytes)", thead_pug.len());
                    }
                }
            }
        }

        // 4. DB 저장을 head 추출 이후로 실행하여 head 정보를 포함한 selector_info를 영구 저장합니다.
        if !skip_ai_analysis || cache_updated {
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
                obj.insert("type".to_string(), json!(page_type.clone()));
                
                if let Some(item_sel) = selector_info.get("itemSelector") {
                    obj.insert("item".to_string(), item_sel.clone()); 
                }
                if let Some(parent_sel) = selector_info.get("parent") {
                    obj.insert("node".to_string(), parent_sel.clone()); 
                }
                obj.insert("detail".to_string(), json!(false));
            }

            let ref_for_page = if !task.r#ref.is_empty() { &task.r#ref } else { raw_path };
            let _ = store.upsert_item("pages", &page_id, "pages", page_data, None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(ref_for_page), None).await;
            println!("[Scheduler] Page cache updated in DB (including head selector).");
        }
        
        // [LIST MODE] 지능형 리스트 추출 (LLM 기반)
        let list_log = json!({ "category": "List Processing", "summary": "Extracting list data with LLM...", "spinner": "⠋" });
        log_task_progress(app_handle, &task.id, &list_log);

        let mut all_extracted_items = Vec::new();
        
        // 병합 대기열을 위한 변수들
        let mut pending_merge: Option<serde_json::Value> = None;
        let mut merge_countdown = 0;

        let pug_list = {
            let clean_content = &clean_html_content;
            let document = scraper::Html::parse_document(clean_content);
            
            // 🌟 5. alt 속성 주입을 위한 headers 수집을 완전히 폐기하고 None으로 PUG를 생성합니다.
            parsing::split_doc_to_pug_list_advanced(
                &document, 
                &target_selector, 
                PugMode::FullContent, 
                None
            )
        };

        if !pug_list.is_empty() {
            // 🌟 [CRITICAL FIX 2] 리스트 추출은 짧은 item_pug 조각만 독립적으로 읽으므로, 
            // 무겁고 호환되지 않는 Base 스냅샷 로딩을 원천 차단합니다.
            model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancellation_token.clone()), false, Some("inference".to_string())).await?;

            for (idx, item_pug) in pug_list.iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { break; }
                
                let percent = (((idx + 1) as f32 / pug_list.len() as f32) * 100.0) as i32;
                let summary_msg = format!("Extracting item data ({}%)...", percent);
                
                let payload = json!({ 
                    "task_id": task.id, 
                    "category": format!("List Extraction ({}/{})", idx + 1, pug_list.len()), 
                    "summary": summary_msg, 
                    "spinner": "⠋" 
                });
                log_task_progress(app_handle, &task.id, &payload);
                emit_term(&format!("[STAGE-3] {}", summary_msg));

                // 🌟 [CRITICAL FIX 3] E0061 해결: 인자 5개를 받도록 변경된 list2json 구조에 완벽하게 맞춥니다.
                let task_question = parsing::list2json(
                    &page_type, 
                    &url, 
                    language, 
                    &thead_pug, 
                    item_pug
                );
                
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
                        model: "qwen3.5".to_string(), max_tokens: Some(512), temperature: Some(0.0), top_p: Some(0.95),
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
                        
                        // 🌟 [CRITICAL FIX] LLM이 {"order": {...}} 형태로 껍데기를 씌워서 반환하므로,
                        // page_type(예: "order", "goods") 키를 찾아서 알맹이만 빼냅니다.
                        let mut item_json = if let Some(inner) = parsed_json.get_mut(&page_type) {
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
            let clean_content = &clean_html_content;
            let raw_pug = parsing::convert_to_clean_pug(clean_content, PugMode::FullContent);
            
            // 🌟 [CRITICAL FIX] 디테일 모드에서도 통일된 절단 로직을 호출하여 모델 부재 시의 누수를 막습니다.
            model.truncate_pug_context(&raw_pug, true, 2000, None).await
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
                    top_p: Some(0.95),
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

                    let mut parsed_json = parsing::parse_json_from_llm(&res.text);
                    
                    // 🌟 [CRITICAL FIX] LLM이 {"order": {...}} 형태로 껍데기를 씌워서 반환하므로,
                    // page_type(예: "order", "goods") 키를 찾아서 알맹이만 빼냅니다.
                    extracted_data = if let Some(inner) = parsed_json.get_mut(&page_type) {
                        inner.take() // 알맹이 적중 시 꺼냄
                    } else {
                        parsed_json // 방어 로직: LLM이 껍데기 없이 바로 뱉었을 경우 그대로 사용
                    };

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
        // 🌟 스피너(⠋) 속성 추가
        log_task_progress(app_handle, &task.id, &json!({ "category": "Handover", "summary": "Switching to Embedding model...", "spinner": "⠋" }));
        
        // 1. Explicitly Unload to free VRAM for Embedding Model
        model.deep_purge_resources().await;
        
        // 2. Wait for VRAM to settle (Driver latency)
        wait_for_resources_settled(1200, 800, Some(cancellation_token)).await?;
    }

    // --- DB OPS & SIDE EFFECTS ---
    // Normalize Data
    let normalize_data = |item: &mut serde_json::Value| {
        if let Some(obj) = item.as_object_mut() {
            if obj.get("type").is_none() { obj.insert("type".to_string(), json!(page_type.clone())); }
            
            // 통화 대문자 변환
            if let Some(c) = obj.get("currency").and_then(|v| v.as_str()) {
                obj.insert("currency".to_string(), json!(c.to_uppercase()));
            }
            
            // 수량 정수형 캐스팅
            if let Some(q) = obj.get("quantity").cloned() {
                let q_val = if q.is_number() { q.as_i64().unwrap_or(0) }
                            else if let Some(s) = q.as_str() { s.parse::<i64>().unwrap_or(0) }
                            else { 0 };
                obj.insert("quantity".to_string(), json!(q_val));
            }
            
            // 날짜 기본값(Fallback) 매핑
            if obj.get("started_at").is_none() || obj.get("started_at").unwrap().is_null() {
                if let Some(m) = obj.get("manufacture_date").cloned() { obj.insert("started_at".to_string(), m); }
            }
            if obj.get("expired_at").is_none() || obj.get("expired_at").unwrap().is_null() {
                if let Some(e) = obj.get("expiration_date").cloned() { obj.insert("expired_at".to_string(), e); }
            }
            
            // 상태(Condition) 텍스트의 정수형 플래그 매핑
            if let Some(cond) = obj.get("condition").and_then(|v| v.as_str()) {
                let cond_lower = cond.to_lowercase();
                if cond_lower.contains("used") { obj.insert("used".to_string(), json!(1)); }
                if cond_lower.contains("lease") { obj.insert("lease".to_string(), json!(2)); }
                if cond_lower.contains("rental") { obj.insert("rental".to_string(), json!(3)); }
                if cond_lower.contains("refurbish") { obj.insert("refurbish".to_string(), json!(4)); }
            }
        }
    };

    if is_detail {
        normalize_data(&mut extracted_data);
    } else {
        if let Some(items) = extracted_data.get_mut("items").and_then(|v| v.as_array_mut()) {
            for item in items.iter_mut() {
                normalize_data(item);
            }
        }
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
                    
                    // 🌟 [버그 수정] 부모(Order)의 송장번호와 자식(Goods)의 고유값을 결합하여 완벽한 Tracking 객체로 조립합니다.
                    let tracking_number = extracted_data.get("tracking_number").and_then(|v| v.as_str()).unwrap_or("");
                    let clean_tracking_no = crate::utils::hash::normalize_numeric_homoglyphs(tracking_number).replace("-", "").replace("_", "");
                    let tracking_index = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("tracking{}{}", team_id, clean_tracking_no)));
                    let goods_index = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("goods{}{}", team_id, clean_g_no)));
                    
                    let tracking_id = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, clean_tracking_no, clean_g_no));
                    let mut tracking_data = extracted_data.clone();
                    
                    if let Some(obj) = tracking_data.as_object_mut() {
                        obj.insert("type".to_string(), json!("tracking"));
                        obj.insert("no".to_string(), json!(clean_tracking_no));
                        obj.insert("index".to_string(), json!(tracking_index));
                        obj.insert("goods".to_string(), json!(goods_index));
                        obj.insert("order".to_string(), json!(index_val)); // 부모 오더 index 매핑
                    }
                    
                    // 🌟 배송 정보 분리 시 자연어 요약 및 임베딩 추출 로직 추가
                    let tracking_text = parsing::json_to_natural_language(&tracking_data);
                    let tracking_vector = model.get_embedding(tracking_text.clone()).await.unwrap_or(vec![0.0; 768]);
                    tracking_data.as_object_mut().unwrap().insert("text".to_string(), json!(tracking_text));
                    
                    let _ = store.upsert_item(
                        "tracking", &tracking_id, "tracking", tracking_data.clone(), Some(tracking_vector.clone()),
                        Some(&task.from), Some(&team_id), Some(&task.cc),
                        Some(&crate::utils::hash::hash_id(&format!("tracking{}", cc_val))),
                        Some(&crate::utils::hash::hash_id(&format!("{}{}{}", team_id, task.cc, task.r#ref))),
                        None
                    ).await;

                    // 🌟 공용 items 테이블에도 벡터와 함께 저장하여 통합 검색이 가능하게 합니다.
                    let _ = store.upsert_item(
                        "items", &tracking_id, "tracking", tracking_data, Some(tracking_vector),
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
    let mut page_draft_diff = 0i64;
    let mut page_count_diff = 0i64;
    let mut global_count_diff = 0i64;

    if is_detail {
        // 🌟 [DETAIL MODE] 단일 문서일 경우
        let text_to_embed = parsing::json_to_natural_language(&extracted_data);
        let item_digest = crate::utils::hash::digest(&text_to_embed); 
        let target_id = if !task.r#ref.is_empty() { task.r#ref.clone() } else { generated_id.clone() }; 
        
        let mut existing_vector = None;
        let mut is_new = true;
        let mut was_draft = false;

        if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &target_id).await {
            is_new = false;
            if existing_item.updated_at_ts == 0 {
                was_draft = true;
            }
            if existing_item.digest == item_digest {
                existing_vector = Some(existing_item.vector);
            }
        }

        // 🌟 [클라우드 패리티 일치] 상세(Detail) 페이지 수집 시: 
        // 새 항목이면 page_count++, 기존 Draft 항목이었다면 draft--, page_count++ 승급
        if is_new {
            page_count_diff += 1;
            global_count_diff += 1;
        } else if was_draft {
            page_draft_diff -= 1;
            page_count_diff += 1;
        }

        let vector = if let Some(v) = existing_vector {
            Some(v)
        } else {
            Some(model.get_embedding(text_to_embed).await?)
        };

        // 🌟 [누락 복구] 교차 업데이트 (Relay) 로직 실행 (단일 상세 항목)
        let related_types = crate::logic::related(&page_type);
        for foreign_type in related_types {
            if let Some((queries, merge_rule)) = crate::logic::relay(foreign_type, &extracted_data) {
                for q in queries {
                    match store.find_item_by_property(&q.table, &q.column, &q.value).await {
                        Ok(Some((foreign_id, mut foreign_data))) => {
                            let mut needs_update = false;

                            // 2. Update 속성 병합 (Import/Export)
                            if let Some(update) = &merge_rule.update {
                                for field in &update.includes {
                                    if update.from == page_type {
                                        if let Some(val) = extracted_data.get(field).cloned() {
                                            foreign_data.as_object_mut().unwrap().insert(field.clone(), val);
                                            needs_update = true;
                                        }
                                    } else if update.to == page_type {
                                        if let Some(val) = foreign_data.get(field).cloned() {
                                            extracted_data.as_object_mut().unwrap().insert(field.clone(), val);
                                        }
                                    }
                                }
                                if let Some(foreign_info) = &update.foreign {
                                    if update.from == page_type {
                                        if let Some(val) = extracted_data.get(&foreign_info.to).cloned() {
                                            foreign_data.as_object_mut().unwrap().insert(foreign_info.from.clone(), val);
                                            needs_update = true;
                                        }
                                    } else if update.to == page_type {
                                        if let Some(val) = foreign_data.get(&foreign_info.to).cloned() {
                                            extracted_data.as_object_mut().unwrap().insert(foreign_info.from.clone(), val);
                                        }
                                    }
                                }
                            }

                            // 3. Upsert 속성 병합
                            if let Some(upsert) = &merge_rule.upsert {
                                for field in &upsert.includes {
                                    if upsert.from == page_type {
                                        if let Some(val) = extracted_data.get(field).cloned() {
                                            foreign_data.as_object_mut().unwrap().insert(field.clone(), val);
                                            needs_update = true;
                                        }
                                    } else if upsert.to == page_type {
                                        if let Some(val) = foreign_data.get(field).cloned() {
                                            extracted_data.as_object_mut().unwrap().insert(field.clone(), val);
                                        }
                                    }
                                }
                            }

                            // 4. 연관 문서에 변경 사항이 있다면 벡터 재생성 후 DB 재저장
                            if needs_update {
                                let merged_text = parsing::json_to_natural_language(&foreign_data);
                                let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 768]);
                                foreign_data.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text));

                                let _ = store.upsert_item(
                                    &q.table, &foreign_id, foreign_type, foreign_data.clone(), Some(merged_vector.clone()),
                                    Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                ).await;
                                
                                let _ = store.upsert_item(
                                    "items", &foreign_id, foreign_type, foreign_data, Some(merged_vector),
                                    Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                ).await;
                            }
                        },
                        Ok(None) => {
                            // 🌟 연관 문서가 존재하지 않으면 Draft(가계정) 껍데기를 생성합니다.
                            let mut draft_data = json!({});
                            
                            // 🌟 [버그 수정] q.value가 Number(예: index 값)일 때 해시가 증발하는 현상을 방어합니다.
                            let val_str = match &q.value {
                                serde_json::Value::String(s) => s.clone(),
                                serde_json::Value::Number(n) => n.to_string(),
                                _ => q.value.to_string(),
                            };
                            let draft_id = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, foreign_type, val_str));
                            
                            if let Some(obj) = draft_data.as_object_mut() {
                                obj.insert("id".to_string(), json!(draft_id.clone()));
                                obj.insert("type".to_string(), json!(foreign_type));
                                obj.insert(q.column.clone(), q.value.clone());
                                obj.insert("updated_at".to_string(), json!(0)); // Draft 플래그
                            }

                            let _ = store.upsert_item(
                                &q.table, &draft_id, foreign_type, draft_data.clone(), None,
                                Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                            ).await;

                            let _ = store.upsert_item(
                                "items", &draft_id, foreign_type, draft_data, None,
                                Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                            ).await;
                        },
                        _ => {}
                    }
                }
            }
        }

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
        // 🌟 [LIST MODE] 리스트 배열 순회
        if let Some(items) = extracted_data.get("items").and_then(|v| v.as_array()) {
            for item_val in items.iter() {
                let mut single_item = item_val.clone();
                
                let original_id = single_item.get("id").or_else(|| single_item.get("no")).and_then(|v| v.as_str())
                    .unwrap_or_else(|| single_item.get("link").and_then(|v| v.as_str()).unwrap_or(""));
                
                let clean_no = crate::utils::hash::normalize_numeric_homoglyphs(original_id).replace("-", "").replace("_", "");
                let index_val = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("{}{}{}", page_type, team_id, clean_no)));
                let hashed_item_id = if original_id.is_empty() {
                    crate::utils::hash::hash_id(&format!("{}{}", team_id, uuid::Uuid::new_v4()))
                } else {
                    crate::utils::hash::hash_id(&format!("{}{}", team_id, index_val))
                };

                if let Some(obj) = single_item.as_object_mut() {
                    obj.insert("type".to_string(), json!(page_type));
                    obj.insert("detail".to_string(), json!(false));
                    obj.insert("id".to_string(), json!(hashed_item_id.clone()));
                    obj.insert("index".to_string(), json!(index_val));
                }

                let text_to_embed = parsing::json_to_natural_language(&single_item);
                let item_digest = crate::utils::hash::digest(&text_to_embed);
                
                let mut existing_vector = None;
                let mut is_new = true;

                if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &hashed_item_id).await {
                    is_new = false;
                    if existing_item.digest == item_digest {
                        existing_vector = Some(existing_item.vector);
                    }
                }

                // 🌟 [클라우드 패리티 일치] 리스트(List) 수집 시: 상세 데이터가 아니므로 무조건 draft++ 처리
                if is_new {
                    page_draft_diff += 1;
                    global_count_diff += 1;
                }
                
                let vector = if let Some(v) = existing_vector {
                    Some(v)
                } else {
                    Some(model.get_embedding(text_to_embed).await?)
                };

                // 🌟 [누락 복구] 교차 업데이트 (Relay) 로직 실행 (리스트 아이템)
                let related_types = crate::logic::related(&page_type);
                for foreign_type in related_types {
                    if let Some((queries, merge_rule)) = crate::logic::relay(foreign_type, &single_item) {
                        for q in queries {
                            match store.find_item_by_property(&q.table, &q.column, &q.value).await {
                                Ok(Some((foreign_id, mut foreign_data))) => {
                                    let mut needs_update = false;

                                    // 2. Update 속성 병합
                                    if let Some(update) = &merge_rule.update {
                                        for field in &update.includes {
                                            if update.from == page_type {
                                                if let Some(val) = single_item.get(field).cloned() {
                                                    foreign_data.as_object_mut().unwrap().insert(field.clone(), val);
                                                    needs_update = true;
                                                }
                                            } else if update.to == page_type {
                                                if let Some(val) = foreign_data.get(field).cloned() {
                                                    single_item.as_object_mut().unwrap().insert(field.clone(), val);
                                                }
                                            }
                                        }
                                        if let Some(foreign_info) = &update.foreign {
                                            if update.from == page_type {
                                                if let Some(val) = single_item.get(&foreign_info.to).cloned() {
                                                    foreign_data.as_object_mut().unwrap().insert(foreign_info.from.clone(), val);
                                                    needs_update = true;
                                                }
                                            } else if update.to == page_type {
                                                if let Some(val) = foreign_data.get(&foreign_info.to).cloned() {
                                                    single_item.as_object_mut().unwrap().insert(foreign_info.from.clone(), val);
                                                }
                                            }
                                        }
                                    }

                                    // 3. Upsert 속성 병합
                                    if let Some(upsert) = &merge_rule.upsert {
                                        for field in &upsert.includes {
                                            if upsert.from == page_type {
                                                if let Some(val) = single_item.get(field).cloned() {
                                                    foreign_data.as_object_mut().unwrap().insert(field.clone(), val);
                                                    needs_update = true;
                                                }
                                            } else if upsert.to == page_type {
                                                if let Some(val) = foreign_data.get(field).cloned() {
                                                    single_item.as_object_mut().unwrap().insert(field.clone(), val);
                                                }
                                            }
                                        }
                                    }

                                    // 4. 연관 문서에 변경 사항이 있다면 벡터 재생성 후 DB 재저장
                                    if needs_update {
                                        let merged_text = parsing::json_to_natural_language(&foreign_data);
                                        let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 768]);
                                        foreign_data.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text));

                                        let _ = store.upsert_item(
                                            &q.table, &foreign_id, foreign_type, foreign_data.clone(), Some(merged_vector.clone()),
                                            Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                        ).await;
                                        
                                        let _ = store.upsert_item(
                                            "items", &foreign_id, foreign_type, foreign_data, Some(merged_vector),
                                            Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                        ).await;
                                    }
                                },
                                Ok(None) => {
                                    // 🌟 연관 문서가 존재하지 않으면 Draft(가계정) 껍데기를 생성합니다.
                                    let mut draft_data = json!({});
                                    
                                    // 🌟 [버그 수정] q.value가 Number(예: index 값)일 때 해시가 증발하는 현상을 방어합니다.
                                    let val_str = match &q.value {
                                        serde_json::Value::String(s) => s.clone(),
                                        serde_json::Value::Number(n) => n.to_string(),
                                        _ => q.value.to_string(),
                                    };
                                    let draft_id = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, foreign_type, val_str));
                                    
                                    if let Some(obj) = draft_data.as_object_mut() {
                                        obj.insert("id".to_string(), json!(draft_id.clone()));
                                        obj.insert("type".to_string(), json!(foreign_type));
                                        obj.insert(q.column.clone(), q.value.clone());
                                        obj.insert("updated_at".to_string(), json!(0)); // Draft 플래그
                                    }

                                    let _ = store.upsert_item(
                                        &q.table, &draft_id, foreign_type, draft_data.clone(), None,
                                        Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                    ).await;

                                    let _ = store.upsert_item(
                                        "items", &draft_id, foreign_type, draft_data, None,
                                        Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                    ).await;
                                },
                                _ => {}
                            }
                        }
                    }
                }

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
        let _ = update_team_base_metrics(&store, &team_id, &task.cc, &page_type, &items_to_process, page_draft_diff, page_count_diff, global_count_diff).await;
        println!("[PROCESS] Metrics Engine updated base statistics for {} items. (Page Draft: {}, Page Count: {}, Global Count: {})", items_to_process.len(), page_draft_diff, page_count_diff, global_count_diff);
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
    page_draft_diff: i64,
    page_count_diff: i64,
    global_count_diff: i64,
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
    
    // --- [블록 1: 페이지별 통계 업데이트] ---
    {
        let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
        let pages = base.entry("pages").or_insert(json!({})).as_object_mut().unwrap();
        let cc_node = pages.entry(task_cc).or_insert(json!({})).as_object_mut().unwrap();
        let page_type_node = cc_node.entry(item_type).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();

        let current_draft = page_type_node.get("draft").and_then(|v| v.as_i64()).unwrap_or(0);
        let current_count = page_type_node.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
        
        page_type_node.insert("draft".to_string(), json!(0.max(current_draft + page_draft_diff)));
        page_type_node.insert("count".to_string(), json!(0.max(current_count + page_count_diff)));
    } 

    // --- [블록 2: 글로벌 전체 통계 및 Min/Max 업데이트] ---
    {
        let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
        let global_type_node = base.entry(item_type).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();
        
        let global_count = global_type_node.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
        
        // 🌟 [클라우드 패리티 완벽 일치] 클라우드(index.ts)에서는 글로벌 통계에 'draft'를 누적하지 않습니다! 오직 'count'(총 발생 건수)만 누적합니다.
        global_type_node.insert("count".to_string(), json!(0.max(global_count + global_count_diff)));

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

                    // 🌟 [버그 수정] 클라우드 JS 초기화 로직(0.0)과 호환되도록 기본값을 0.0으로 통일
                    let prop_node = global_type_node.entry(*prop).or_insert(json!({ "min": 0.0, "max": 0.0 })).as_object_mut().unwrap();
                    
                    let current_min = prop_node.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let current_max = prop_node.get("max").and_then(|v| v.as_f64()).unwrap_or(0.0);

                    // 🌟 [버그 수정] DB에서 가져온 current_min이 0.0(초기값)일 경우 조건 없이 무조건 첫 값을 덮어쓰도록 예외 처리
                    // 🌟 [비교 연산자 수정] JS와 동일하게 <=, >= 로 연산자 패리티 일치
                    if current_min == 0.0 || num_val <= current_min { prop_node.insert("min".to_string(), json!(num_val)); }
                    if current_max == 0.0 || num_val >= current_max { prop_node.insert("max".to_string(), json!(num_val)); }
                }
            }
        }
    } // 👈 여기서 두 번째 참조가 종료됩니다.

    // 🌟 [디버깅 로그 추가] 프론트엔드로 전달되기 전, DB에 최종 반영되는 base JSON 전체 구조를 예쁘게 출력합니다.
    if let Some(base_json) = team_data.get("base") {
        println!("\n[DEBUG-METRICS] 최종 반영된 Base JSON 값:\n{}", serde_json::to_string_pretty(base_json).unwrap_or_default());
    }

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
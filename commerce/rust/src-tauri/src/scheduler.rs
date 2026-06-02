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

fn merge_node(obj1: &Value, obj2: &Value) -> Value {
    let mut merged = obj1.clone();
    if let (Some(m_obj), Some(o2_obj)) = (merged.as_object_mut(), obj2.as_object()) {
        for (k, v) in o2_obj {
            let is_empty = match v {
                Value::Null => true,
                Value::String(s) => s.is_empty(),
                Value::Number(n) => n.as_f64().unwrap_or(0.0) == 0.0,
                _ => false,
            };
            if !is_empty {
                m_obj.insert(k.clone(), v.clone());
            }
        }
    }
    merged
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

    
    // 여기서 다시 spawn 하여 불필요한 DB 락 경쟁을 일으킬 필요가 없습니다.
    
    tokio::spawn(async move {
        if !UI_READY_FLAG.load(Ordering::SeqCst) {
            UI_READY_SIGNAL.notified().await;
        }
        
        let mut delay_secs = 1;
        let mut current_device_pref: Option<String> = None;
        
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
                        
                        
                        if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() {
                            *w = Some(json!({ "id": task.id, "ref": task.r#ref, "status": 1 }));
                        }
                    }
                }

                match process_task(task.clone(), &store, &model, &cancellation_token, &app_handle, current_device_pref.clone()).await {
                    Ok(_) => {
                        println!("[Scheduler] Task completed: {}", task.id);
                        
                        
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
                             
                             // 여기서 백엔드가 메시지를 다시 생성하거나 이벤트를 쏘면 UI가 좀비처럼 부활하므로 조용히 종료만 합니다.
                             current_device_pref = None;
                             continue;
                        } else if err_msg.contains("CUDA_ERROR_OUT_OF_MEMORY") || err_msg.contains("out of memory") {
                            let retries = oom_retry_map.entry(task.id.clone()).or_insert(0);
                            
                            if *retries == 0 {
                                *retries += 1;
                                println!("[Scheduler] OOM Detected! VRAM is purged. Retrying on GPU...");
                                current_device_pref = None;

                                
                                let payload = json!({
                                    "task_id": task.id,
                                    "category": "Warning", "summary": "Memory pressure detected. VRAM cleared. Retrying on GPU...", "spinner": "♻️"
                                });
                                let _ = app_handle.emit("extraction-progress", &payload);

                                
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
    
    
    let app_handle_clone = app_handle.clone();
    let tid_clone = task.id.clone();
    let emit_term = move |msg: &str| {
        println!("{}", msg);
        use tauri::Emitter;
        let _ = app_handle_clone.emit("task-console-log", serde_json::json!({"task_id": tid_clone, "text": format!("{}\n", msg)}));
    };

    
    let zero_addr = "0x0000000000000000000000000000000000000000";
    let from_addr = if task.from.is_empty() { zero_addr.to_string() } else { task.from.clone() };
    let team_id = if task.to.is_empty() || task.to == zero_addr { 
        crate::utils::hash::hash_id(&from_addr) 
    } else { 
        task.to.clone() 
    };

    emit_term("\n=======================================");
    emit_term(&format!("[PROCESS] ⚙️ Task {} started processing.", task.id));

    
    if task.r#type == "analytic_extraction" {
        return crate::analytic::process_analytic_task(
            task, store_mutex, model_mutex, cancellation_token, app_handle, device_preference
        ).await;
    }

    let kv_path = utils::paths::get_kv_dir(Some(app_handle)).join(&task.id);
    if kv_path.exists() {
        emit_term(&format!("[PROCESS] Found existing KV cache for task {}. Ready to reuse.", task.id));
    }

    
    let payload = json!({ 
        "task_id": task.id,
        "task_type": task.r#type, 
        "category": "Processing", "summary": "Starting extraction...", "spinner": "⠋" 
    });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let mut task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    
    
    let search_mode = task_data.get("search_mode").and_then(|s| s.as_str()).unwrap_or("commerce").to_string();

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

    
    // 브라우저 자동화 모듈이 감지한 '진짜 현재 활성화 탭 URL'을 강제로 끌어와서 완벽한 절대 주소로 병합(Join)합니다!
    {
        let state = crate::automation::LAST_DETECTED_STATE.lock().await;
        let active_tab_url = state.url.clone();
        
        if !active_tab_url.is_empty() {
            if let Ok(active_parsed) = url::Url::parse(&active_tab_url) {
                let active_origin = format!("{}://{}", active_parsed.scheme(), active_parsed.host_str().unwrap_or("localhost"));
                
                if origin_candidate.is_empty() || origin_candidate.contains("localhost") {
                    origin_candidate = active_origin;
                }
                
                if url.is_empty() {
                    url = active_tab_url;
                } else if !url.starts_with("http") {
                    
                    if let Ok(joined) = active_parsed.join(&url) {
                        url = joined.to_string();
                    }
                }
            }
        }
    }

    
    if !url.starts_with("http") && !origin_candidate.is_empty() && !origin_candidate.contains("localhost") {
        let scheme = if origin_candidate.starts_with("http") { "" } else { "http://" };
        let base_str = format!("{}{}", scheme, origin_candidate);
        if let Ok(base) = url::Url::parse(&base_str) {
            if let Ok(joined) = base.join(&url) {
                url = joined.to_string();
            }
        }
    }
    
    
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
    
    let mut raw_pug = parsing::convert_to_clean_pug(&clean_html_content, PugMode::NoAttributesMode, Some(&url));
    let mut light_pug = model.truncate_pug_context(&raw_pug, false, 2000, None).await;

    // 1. 정확한 토큰 수 측정을 위해 Tokenizer 로드 (파일 경로 탐색)
    let base_path = std::fs::canonicalize("src-tauri/models").or_else(|_| std::fs::canonicalize("models")).unwrap_or_default();
    let tokenizer_path = base_path.join("Qwen3-0.6B-Instruct-gguf").to_string_lossy().to_string();
    
    // 1. 모델이 실제로 받게 될 전체 서식을 먼저 만듭니다. (scheduler.rs 539라인 참고)
    let raw_system_prefix = format!("<|im_start|>system\n{}<|im_end|>\n", light_pug);

    // 2. 이 전체 문자열을 인코딩해야 [TEXT-PREFILL]과 100% 일치합니다.
    let mut token_count = raw_system_prefix.len() / 4; // 폴백용

    if let Ok(tokenizer) = crate::tokenizer::TokenizerModel::init(&tokenizer_path) {
        // light_pug가 아니라 서식이 포함된 raw_system_prefix를 넣습니다.
        token_count = tokenizer.text_encode_vec(raw_system_prefix.clone(), false)
            .map(|v| v.len())
            .unwrap_or(token_count);
    }

    // // 2. 실제 계산된 토큰 수가 3000 이하일 경우 FullContent 모드로 승급
    if token_count <= 6000 {
        println!("[Scheduler] Document is short enough ({} tokens). Upgrading to FullContent Mode...", token_count);
        raw_pug = parsing::convert_to_clean_pug(&clean_html_content, PugMode::FullContent, Some(&url));
        light_pug = model.truncate_pug_context(&raw_pug, true, 2000, None).await;
    }

    let base_model_size = if token_count > 60000 {
        crate::model::ModelSize::Qwen
    } else {
        crate::model::ModelSize::Qwen3
    };

    println!("[DEBUG-PUG] Generated PUG. Length: {}. Token Count: {}. Selected Model: {:?}. Snippet: {}...", 
        light_pug.len(), 
        token_count,
        base_model_size,
        light_pug.chars().take(100).collect::<String>().replace("\n", " ")
    );


    use crate::openai_types::{
        ChatCompletionRequestSystemMessage,
        ChatCompletionParameters, ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent
    };

    let mut page_type = String::new();
    let mut selector_info: serde_json::Value = json!({});
    
    let mut is_detail = task_data.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut skip_ai_analysis = false; 

    let (raw_path, url_obj) = {
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
        (url_obj.path().to_string(), url_obj)
    };

    
    let cc_for_hash = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
    let page_id = crate::utils::hash::hash_id(&format!("{}{}", cc_for_hash, raw_path));

    {
        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
            
            
            // 클라우드(aa.ts)는 원본 대소문자를 유지하여 저장하고, 로컬(main.ts)은 소문자로 변환하여 요청합니다.
            // 경로 비교 시 반드시 소문자로 통일하여 검색해야 100% 매칭됩니다!
            let link_val = (url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str()).to_lowercase();
            let path_only = url_obj.path().to_lowercase(); 
            
            let mut potential_caches = Vec::new();

            // 1. ID 기반 조회 (정확한 매칭 1차 수집)
            if let Ok(Some(page_doc)) = db.get_item_by_id("pages", &page_id).await {
                potential_caches.push(page_doc);
            } else if let Ok(Some(page_doc)) = db.get_item_by_id("items", &page_id).await {
                potential_caches.push(page_doc);
            }

            // 2. URL 경로 기반 역추적 조회 (대소문자 무시)
            let tables_to_check = ["pages", "items"];
            for tbl in tables_to_check {
                if let Ok(docs) = db.get_all_items(tbl, 1000, 0, None).await {
                    for doc in docs {
                        let json_lower = doc.json_data.to_lowercase();
                        if json_lower.contains(&link_val) || json_lower.contains(&path_only) {
                            if !potential_caches.iter().any(|c| c.id == doc.id) {
                                potential_caches.push(doc);
                            }
                        }
                    }
                }
            }

            // 3. 수집된 캐시 중 현재 DOM 구조와 가장 잘 맞는 캐시 선별
            let mut final_cache = None;

            for page_doc in potential_caches {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&page_doc.json_data) {
                    let cached_detail = val.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
                    let node_sel = val.get("node").or_else(|| val.get("parent")).and_then(|v| v.as_str()).unwrap_or("");
                    let item_sel = val.get("item").or_else(|| val.get("itemSelector")).and_then(|v| v.as_str()).unwrap_or("");

                    let target_sel_str = if !node_sel.is_empty() && !item_sel.is_empty() && !item_sel.contains(",") {
                        if item_sel.starts_with(node_sel) { item_sel.to_string() } else { format!("{} {}", node_sel, item_sel) }
                    } else if !item_sel.is_empty() { item_sel.to_string() } else { node_sel.to_string() };

                    
                    let target_sel_clean = target_sel_str.replace(">", " ");

                    if !cached_detail {
                        let mut is_dom_matched = false;
                        if !target_sel_clean.is_empty() {
                            let document = scraper::Html::parse_document(&clean_html_content);
                            is_dom_matched = scraper::Selector::parse(&target_sel_clean)
                                .map(|sel| document.select(&sel).next().is_some())
                                .unwrap_or(false);
                        }

                        if is_dom_matched {
                            // DOM까지 완벽 일치하는 리스트 캐시 -> 최우선 채택 및 탐색 종료
                            final_cache = Some((page_doc, val, false, target_sel_clean));
                            break;
                        } 
                        
                        // (빈 리스트일 가능성보다, 동일한 주소 체계를 가진 상세 페이지일 가능성이 99%이기 때문입니다.)
                    } else {
                        // Detail 캐시인 경우
                        if final_cache.is_none() {
                            final_cache = Some((page_doc, val, true, target_sel_clean));
                        }
                    }
                }
            }

            // 4. 최종 결정된 캐시 적용 및 파이프라인 패스
            if let Some((_page_doc, val, cached_detail, target_sel_str)) = final_cache {
                emit_term(&format!("[Scheduler] ⚡ CACHE HIT! Skipping AI Pre-processing for: {}", raw_path));
                page_type = val.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").trim().to_lowercase();
                
                
                is_detail = cached_detail; 
                
                selector_info = val.clone();
                selector_info.as_object_mut().unwrap().insert("final_target_selector".to_string(), json!(target_sel_str));
                skip_ai_analysis = true; 
                
                log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Loaded valid config from cache.", "spinner": "⚡" }));
            } else {
                emit_term("[Scheduler] Cache miss or elements not found in DOM. Falling back to AI Analysis.");
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
    let system_content = format!("[PUG CONTENT]\n{}", light_pug);

    
    if !skip_ai_analysis {
        // --- STEP 0: BASE BAKING (공통 컨텍스트 딱 1번만 굽기) ---
        if base_model_size == crate::model::ModelSize::Qwen {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            
            let base_kv_path = utils::paths::get_kv_dir(Some(app_handle)).join(&base_session_id);
            if !base_kv_path.exists() {
                println!("[Scheduler] Baking Base PUG Context to SSD...");
                log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Reading document structure...", "spinner": "⠋" }));
                
                
                model.secure_vram_relay(crate::model::ModelSize::Qwen, None, Some(cancellation_token.clone()), true, kv_name.clone()).await?;
                
                
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                if let Some(gen) = model.generator.lock().await.as_mut() {
                    
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
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

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
                    model: if base_model_size == crate::model::ModelSize::Qwen { "qwen".to_string() } else { "qwen3".to_string() }, 
                    max_tokens: Some(16),
                    temperature: Some(0.0), top_p: Some(0.95),
                    ..Default::default()
                };

                let res = if base_model_size == crate::model::ModelSize::Qwen {
                    // [핵심] Step A가 아니라 '미리 구워둔 Base' 스냅샷을 불러옵니다!
                    model.secure_vram_relay(crate::model::ModelSize::Qwen, Some(&base_session_id), Some(cancellation_token.clone()), false, kv_name.clone()).await?;

                    if let Some(gen) = model.generator.lock().await.as_mut() {
                        println!("[Scheduler] 0.6B Step A: Asking classification question...");
                        gen.generate(params, Some(cancellation_token.clone()), Some(snapshot_id.clone()), kv_name.clone()).await?
                    } else {
                        return Err(anyhow::anyhow!("Qwen generator missing"));
                    }
                } else {
                    model.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancellation_token.clone()), false, None).await?;
                    let q3_gen_arc = model.qwen3_generator.clone();
                    let cancel_clone = cancellation_token.clone();
                    tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
                        let mut gen_guard = q3_gen_arc.blocking_lock();
                        if let Some(gen) = gen_guard.as_mut() {
                            println!("[Scheduler] Qwen3 Step A: Asking classification question...");
                            gen.generate(params, Some(cancel_clone), None).map_err(|e| anyhow::anyhow!("Qwen3 failed: {}", e))
                        } else {
                            Err(anyhow::anyhow!("Qwen3 generator missing"))
                        }
                    }).await??
                };

                println!("[DEBUG-SCHED] Step A Raw Response: '{}'", res);
                
                let type_info = parsing::parse_json_from_llm(&res); 
                    
                page_type = type_info.get("type").and_then(|s| s.as_str()).unwrap_or("").trim().to_lowercase();                
                
                if page_type.is_empty() {
                    page_type = match task.r#type.as_str() {
                        "image_extraction" => "tracking".to_string(),
                        _ => "unknown".to_string(),
                    };
                }
                println!("[Scheduler] Classified as: {}", page_type);
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
            
            let snapshot_id = format!("{}_step_a2", task.id); 

            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

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
                model: if base_model_size == crate::model::ModelSize::Qwen { "qwen".to_string() } else { "qwen3".to_string() }, 
                max_tokens: Some(128),     // JSON 스키마가 길어졌으므로 토큰 길이는 128로 유지
                temperature: Some(0.0), top_p: Some(0.95),
                ..Default::default()
            };

            let res = if base_model_size == crate::model::ModelSize::Qwen {
                model.secure_vram_relay(crate::model::ModelSize::Qwen, Some(&base_session_id), Some(cancellation_token.clone()), false, kv_name.clone()).await?;
                if let Some(gen) = model.generator.lock().await.as_mut() {
                    println!("[Scheduler] 0.6B (Qwen) Step A-2: Asking detail classification...");
                    gen.generate(
                        params, 
                        Some(cancellation_token.clone()), 
                        Some(snapshot_id.clone()), 
                        kv_name.clone()
                    ).await?
                } else {
                    return Err(anyhow::anyhow!("Qwen generator missing"));
                }
            } else {
                model.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancellation_token.clone()), false, None).await?;
                let q3_gen_arc = model.qwen3_generator.clone();
                let cancel_clone = cancellation_token.clone();
                tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
                    let mut gen_guard = q3_gen_arc.blocking_lock();
                    if let Some(gen) = gen_guard.as_mut() {
                        println!("[Scheduler] Qwen3 Step A-2: Asking detail classification...");
                        gen.generate(params, Some(cancel_clone), None).map_err(|e| anyhow::anyhow!("Qwen3 failed: {}", e))
                    } else {
                        Err(anyhow::anyhow!("Qwen3 generator missing"))
                    }
                }).await??
            };
            
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
        }
    } // 👈 🌟 [핵심 변경 1 끝] 0.6B 분석 블록 종료

                        
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    model.deep_purge_resources().await;
    wait_for_resources_settled(1200, 800, Some(cancellation_token)).await?;

    let mut extracted_data = json!({});

    // --- PHASE 2 Continue: Detail Extraction (If needed) --- 
    if !is_detail {
        
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
                        model: if base_model_size == crate::model::ModelSize::Qwen { "qwen".to_string() } else { "qwen3".to_string() }, 
                        max_tokens: Some(128), temperature: Some(0.0), top_p: Some(0.95),
                        ..Default::default()
                    };

                    let res = if base_model_size == crate::model::ModelSize::Qwen {
                        model.secure_vram_relay(crate::model::ModelSize::Qwen, Some(&base_session_id), Some(cancellation_token.clone()), false, kv_name.clone()).await?;
                        if let Some(gen) = model.generator.lock().await.as_mut() {
                            println!("[JS-BRIDGE] 1. Requesting titles from LLM (0.6B)...");
                            
                            gen.generate(
                                params, 
                                Some(cancellation_token.clone()), 
                                Some(snapshot_id.clone()), 
                                kv_name.clone()
                            ).await?
                        } else {
                            return Err(anyhow::anyhow!("Qwen generator missing"));
                        }
                    } else {
                        model.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancellation_token.clone()), false, None).await?;
                        let q3_gen_arc = model.qwen3_generator.clone();
                        let cancel_clone = cancellation_token.clone();
                        tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
                            let mut gen_guard = q3_gen_arc.blocking_lock();
                            if let Some(gen) = gen_guard.as_mut() {
                                println!("[JS-BRIDGE] 1. Requesting titles from LLM (Qwen3)...");
                                gen.generate(params, Some(cancel_clone), None).map_err(|e| anyhow::anyhow!("Qwen3 failed: {}", e))
                            } else {
                                Err(anyhow::anyhow!("Qwen3 generator missing"))
                            }
                        }).await??
                    };
                    
                    println!("[JS-BRIDGE] LLM Raw Response: '{}'", res);

                    // res.text 가 아닌 res 를 그대로 파싱
                    let title_info = parsing::parse_json_from_llm(&res);
                        
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
                            let t_val = if let Some(t) = item.as_str() {
                                Some(t)
                            } else if let Some(t) = item.get("title").and_then(|v| v.as_str()) {
                                Some(t)
                            } else {
                                None
                            };
                            
                            if let Some(t) = t_val {
                                
                                let clean_t = t.replace(",", "").replace(".", "").trim().to_string();
                                let is_only_numbers = !clean_t.is_empty() && clean_t.chars().all(|c| c.is_ascii_digit());
                                
                                if !is_only_numbers {
                                    titles.push(t.to_string());
                                }
                            }
                        }
                    }
                    println!("[JS-BRIDGE] Titles extracted (Robust): {:?}", titles);
                }

                if titles.is_empty() {
                    
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

                        let matches = [];
                        for (let i = 0; i < titles.length; i++) {
                            let t = titles[i].toLowerCase().replace(/\s+/g, ' ');
                            let potentialMatches = [];
                            
                            // 깨진 문자(\uFFFD)가 포함되어 있는지 확인합니다.
                            if (t.includes('\uFFFD')) {
                                // 깨진 경우: 쪼개서 조각들로 유연하게 검색
                                let chunks = t.split(/[\uFFFD]+/).map(c => c.trim()).filter(c => c.length > 1);
                                if (chunks.length === 0) continue;
                                
                                potentialMatches = nodes.filter(n => {
                                    if (!n || !n.text) return false;
                                    let nText = n.text.toLowerCase().replace(/\s+/g, ' ');
                                    return chunks.every(chunk => nText.includes(chunk));
                                });
                            } else {
                                // 온전한 경우: 전체 문자열을 하나의 컬렉션으로 취급하여 정확하게 포함 여부 검색
                                potentialMatches = nodes.filter(n => {
                                    if (!n || !n.text) return false;
                                    let nText = n.text.toLowerCase().replace(/\s+/g, ' ');
                                    return nText.includes(t);
                                });
                            }
                            
                            if (potentialMatches.length > 0) {
                                // 부모 노드(body, tr 등)를 배제하고, 텍스트 길이가 가장 짧은(가장 타이트한) 진짜 제목 단일 노드만 추출합니다.
                                potentialMatches.sort((a, b) => a.text.length - b.text.length);
                                matches = [potentialMatches[0]];
                                break;
                            }
                        }
                        
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

        
        let target_selector = selector_info.get("final_target_selector")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                let item_selector = selector_info.get("itemSelector")
                    .or_else(|| selector_info.get("item"))
                    .and_then(|s| s.as_str())
                    .unwrap_or("");
                let node_selector = selector_info.get("node").or_else(|| selector_info.get("parent")).and_then(|s| s.as_str()).unwrap_or("");
                
                if !node_selector.is_empty() && !item_selector.is_empty() && !item_selector.contains(",") {
                    if item_selector.starts_with(node_selector) {
                        item_selector.to_string()
                    } else {
                        format!("{} {}", node_selector, item_selector) 
                    }
                } else if !item_selector.is_empty() { 
                    item_selector.to_string() 
                } else { 
                    node_selector.to_string() 
                }
            }).replace(">", " "); 
            
        emit_term(&format!("[Scheduler] Target Selector configured as: '{}'", target_selector));

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
                    
                    
                    let ref_row_context_size = ref_row.len() + 3000;
                    let full_pug = parsing::convert_to_clean_pug(&clean_html_content, PugMode::NoAttributesMode, Some(&url));
                    let thead_light_pug = model.truncate_pug_context(&full_pug, false, 2000, Some(ref_row_context_size)).await;

                    println!("ref_row: {}", ref_row);
                    
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

                            // 1. thead 선택자 추출 (Flat 구조에 맞추어 get("table") 제거)
                            final_thead_selector = thead_val
                                .and_then(|v| v.get("thead"))
                                .and_then(|v| v.get("selector"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("").to_string().replace(">", " "); 
                            
                            // 2. tbody와 thead를 감싸는 부모 wrapper(table) 선택자 추출
                            let final_table_selector = thead_val
                                .and_then(|v| v.get("table"))
                                .and_then(|v| v.get("selector"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("").to_string().replace(">", " ");

                            
                            if !final_thead_selector.is_empty() && final_thead_selector != "..." && !final_table_selector.is_empty() && final_table_selector != "..." {
                                if !final_thead_selector.contains(&final_table_selector) {
                                    let combined_sel = format!("{} {}", final_table_selector, final_thead_selector);
                                    let doc = scraper::Html::parse_document(&clean_html_content);
                                    
                                    // 컴파일 에러 해결: Result의 에러 객체가 참조를 붙잡지 않도록 즉시 boolean으로 변환 후 스코프를 닫습니다.
                                    let is_valid = scraper::Selector::parse(&combined_sel)
                                        .map(|parsed_sel| doc.select(&parsed_sel).next().is_some())
                                        .unwrap_or(false);

                                    if is_valid {
                                        final_thead_selector = combined_sel;
                                    }
                                }
                            }

                            if !final_thead_selector.is_empty() && final_thead_selector != "..." {
                                selector_info.as_object_mut().unwrap().insert("head".to_string(), json!(final_thead_selector.clone()));
                                println!("[Scheduler] AI determined head selector and cached: {}", final_thead_selector);
                                cache_updated = true; // 새로운 head를 찾았으므로 DB 업데이트 예약
                            }

                            
                            if !final_table_selector.is_empty() && !final_table_selector.contains("CSS selector") && final_table_selector != "..." {
                                selector_info.as_object_mut().unwrap().insert("wrapper".to_string(), json!(final_table_selector.clone()));
                                println!("[Scheduler] AI determined table wrapper selector and cached: {}", final_table_selector);
                                cache_updated = true;
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
                    
                    crate::parsing::generate_pug_lines((*target_node).into(), 0, &mut tpug, &PugMode::TheadMode, &mut None);
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
            let cc_for_hash = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
            let page_id = crate::utils::hash::hash_id(&format!("{}{}", cc_for_hash, raw_path)); 
            
            let cc_for_bcc = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
            let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_for_bcc));

            let ref_for_page = if !task.r#ref.is_empty() { &task.r#ref } else { raw_path };

            
            if !is_detail {
                let mut page_data: serde_json::Value = selector_info.clone();
                if let Some(obj) = page_data.as_object_mut() {
                    obj.insert("origin".to_string(), json!(format!("{}://{}", url_obj.scheme(), url_obj.host_str().unwrap_or(""))));
                    obj.insert("link".to_string(), json!(url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str()));
                    obj.insert("type".to_string(), json!(page_type.clone()));
                    
                    if let Some(item_sel) = selector_info.get("itemSelector") { obj.insert("item".to_string(), item_sel.clone()); }
                    if let Some(parent_sel) = selector_info.get("parent") { obj.insert("node".to_string(), parent_sel.clone()); }
                    obj.insert("detail".to_string(), json!(false));
                }

                
                let _ = store.upsert_item("pages", &page_id, &page_type, page_data.clone(), None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(ref_for_page), None).await;
                let _ = store.upsert_item("items", &page_id, "pages", page_data, None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(ref_for_page), None).await;
                
                println!("[Scheduler] Page cache updated in DB (including head selector).");

                
                let detail_page_id = crate::utils::hash::hash_id(&format!("{}{}{}", page_type, task.cc.to_uppercase(), raw_path));
                let detail_bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, task.cc.to_uppercase()));
                let detail_page_data = json!({
                    "origin": format!("{}://{}", url_obj.scheme(), url_obj.host_str().unwrap_or("")),
                    "link": url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str(),
                    "type": page_type.clone(),
                    "detail": true,
                    "node": true,
                    "item": ""
                });
                let _ = store.upsert_item("pages", &detail_page_id, &page_type, detail_page_data.clone(), None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&detail_bcc), Some(ref_for_page), None).await;
                let _ = store.upsert_item("items", &detail_page_id, "pages", detail_page_data, None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&detail_bcc), Some(ref_for_page), None).await;

            } else {
                let detail_page_id = crate::utils::hash::hash_id(&format!("{}{}{}", page_type, task.cc.to_uppercase(), raw_path));
                let detail_bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, task.cc.to_uppercase()));
                let detail_page_data = json!({
                    "origin": format!("{}://{}", url_obj.scheme(), url_obj.host_str().unwrap_or("")),
                    "link": url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str(),
                    "type": page_type.clone(),
                    "detail": true,
                    "node": true,
                    "item": ""
                });
                let _ = store.upsert_item("pages", &detail_page_id, &page_type, detail_page_data.clone(), None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&detail_bcc), Some(ref_for_page), None).await;
                let _ = store.upsert_item("items", &detail_page_id, "pages", detail_page_data, None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&detail_bcc), Some(ref_for_page), None).await;
            }
        }
        
        // [LIST MODE] 지능형 리스트 추출 (LLM 기반)
        let list_log = json!({ "category": "List Processing", "summary": "Extracting list data with LLM...", "spinner": "⠋" });
        log_task_progress(app_handle, &task.id, &list_log);

        let mut all_extracted_items = Vec::new();
        
        // 병합 대기열을 위한 변수들
        // let pending_merge: Option<serde_json::Value> = None;
        // let merge_countdown = 0;

        let mut pug_list = {
            let clean_content = &clean_html_content;
            let document = scraper::Html::parse_document(clean_content);
            
            
            parsing::split_doc_to_pug_list_advanced(
                &document, 
                &target_selector, 
                PugMode::ListMode, 
                None,
                Some(&url) 
            )
        };

        
        // 2순위: 속성이 없을 경우 thead의 tr 태그 개수로 폴백(Fallback)하여 완벽하게 묶어냅니다.
        let group_size = if !thead_pug.is_empty() {
            let mut max_span = 1;
            if let Ok(re) = regex::Regex::new(r#"(?:colspan|rowspan)="(\d+)""#) {
                for cap in re.captures_iter(&thead_pug) {
                    if let Ok(val) = cap[1].parse::<usize>() {
                        if val > max_span {
                            max_span = val;
                        }
                    }
                }
            }
            
            if max_span > 1 {
                max_span
            } else {
                thead_pug.lines().filter(|line| {
                    let s = line.trim_start();
                    s == "tr" || s.starts_with("tr[")
                }).count().max(1)
            }
        } else {
            1
        };

        if group_size > 1 && !pug_list.is_empty() {
            let mut grouped = Vec::new();
            for chunk in pug_list.chunks(group_size) {
                grouped.push(chunk.join("\n"));
            }
            pug_list = grouped;
            println!("[Scheduler] 🌟 Grouped multi-row items: {} rows per item. Total items reduced to {}.", group_size, pug_list.len());
        }

        if !pug_list.is_empty() {
            let total_items = pug_list.len();

            // ==========================================
            
            // ==========================================
            // 모델 가중치 변경(스위칭) 없이 가장 가벼운 모델인 Qwen3 하나만으로 전체 파이프라인을 관통하여 속도를 극대화합니다!
            model.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancellation_token.clone()), false, Some("inference".to_string())).await?;

            for (idx, item_pug) in pug_list.iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                
                let percent = (((idx as f32) / (total_items as f32)) * 100.0) as i32;
                let summary_msg = format!("Extracting item data ({}%)...", percent);
                
                let payload = json!({ 
                    "task_id": task.id, 
                    "category": format!("List Extraction ({}/{})", idx + 1, total_items), 
                    "summary": summary_msg, 
                    "spinner": "⠋" 
                });
                log_task_progress(app_handle, &task.id, &payload);
                emit_term(&format!("[STAGE-3] {}", summary_msg));

                let task_question_meta = parsing::list2json_meta(&page_type, &url, language, &thead_pug, item_pug);
                let task_question_info = parsing::list2json_info(&page_type, language, &thead_pug, item_pug);
                let task_question_data = parsing::list2json_data(&page_type, language, &thead_pug, item_pug);

                
                let q3_gen_meta = model.qwen3_generator.clone();
                let cancel_meta = cancellation_token.clone();
                let res_meta = tokio::task::spawn_blocking(move || {
                    let mut gen_guard = q3_gen_meta.blocking_lock();
                    if let Some(gen) = gen_guard.as_mut() {
                        println!("[Scheduler] Qwen3 Extracting Item Meta {}/{}...", idx + 1, total_items);
                        let params = ChatCompletionParameters {
                            messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                                content: ChatCompletionRequestUserMessageContent::Text(task_question_meta),
                                name: None,
                            })],
                            model: "qwen3".to_string(), max_tokens: Some(128), temperature: Some(0.0), top_p: Some(0.95),
                            ..Default::default()
                        };
                        gen.generate(params, Some(cancel_meta), None).map_err(|e| anyhow::anyhow!("Qwen 3 Meta failed: {}", e))
                    } else {
                        Err(anyhow::anyhow!("Qwen 3 Generator not available"))
                    }
                }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("Task join failed: {}", e)));

                
                let q3_gen_info = model.qwen3_generator.clone();
                let cancel_info = cancellation_token.clone();
                let res_info = tokio::task::spawn_blocking(move || {
                    let mut gen_guard = q3_gen_info.blocking_lock();
                    if let Some(gen) = gen_guard.as_mut() {
                        println!("[Scheduler] Qwen3 Extracting Item Info {}/{}...", idx + 1, total_items);
                        let params = ChatCompletionParameters {
                            messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                                content: ChatCompletionRequestUserMessageContent::Text(task_question_info),
                                name: None,
                            })],
                            model: "qwen3".to_string(), max_tokens: Some(128), temperature: Some(0.0), top_p: Some(0.95),
                            ..Default::default()
                        };
                        gen.generate(params, Some(cancel_info), None).map_err(|e| anyhow::anyhow!("Qwen 3 Info failed: {}", e))
                    } else {
                        Err(anyhow::anyhow!("Qwen 3 Generator not available"))
                    }
                }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("Task join failed: {}", e)));

                
                let q3_gen_data = model.qwen3_generator.clone();
                let cancel_data = cancellation_token.clone();
                let res_data = tokio::task::spawn_blocking(move || {
                    let mut gen_guard = q3_gen_data.blocking_lock();
                    if let Some(gen) = gen_guard.as_mut() {
                        println!("[Scheduler] Qwen3 Extracting Item Data {}/{}...", idx + 1, total_items);
                        let params = ChatCompletionParameters {
                            messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                                content: ChatCompletionRequestUserMessageContent::Text(task_question_data),
                                name: None,
                            })],
                            model: "qwen3".to_string(), max_tokens: Some(256), temperature: Some(0.0), top_p: Some(0.95),
                            ..Default::default()
                        };
                        gen.generate(params, Some(cancel_data), None).map_err(|e| anyhow::anyhow!("Qwen 3 Data failed: {}", e))
                    } else {
                        Err(anyhow::anyhow!("Qwen 3 Generator not available"))
                    }
                }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("Task join failed: {}", e)));

                match (res_meta, res_info, res_data) {
                    (Ok(res_m), Ok(res_i), Ok(res_d)) => {
                        let mut parsed_meta = parsing::parse_json_from_llm(&res_m);
                        let mut parsed_info = parsing::parse_json_from_llm(&res_i);
                        let mut parsed_data = parsing::parse_json_from_llm(&res_d);
                        
                        let mut item_meta = if let Some(inner) = parsed_meta.get_mut(&page_type) { inner.take() } else { parsed_meta };
                        let mut item_info = if let Some(inner) = parsed_info.get_mut(&page_type) { inner.take() } else { parsed_info };
                        let mut item_data = if let Some(inner) = parsed_data.get_mut(&page_type) { inner.take() } else { parsed_data };
                        
                        
                        if let (Some(m_obj), Some(i_obj)) = (item_meta.as_object_mut(), item_info.as_object_mut()) {
                            for (k, v) in i_obj {
                                m_obj.insert(k.clone(), v.clone());
                            }
                        }
                        if let (Some(m_obj), Some(d_obj)) = (item_meta.as_object_mut(), item_data.as_object_mut()) {
                            for (k, v) in d_obj {
                                m_obj.insert(k.clone(), v.clone());
                            }
                        }

                        
                        // 영문이 포함된 상품코드(P000000S)가 파괴되지 않도록 알파벳을 보존합니다!
                        if let Some(id_val) = item_meta.get("id").and_then(|v| v.as_str()) {
                            let extracted = if let Some(idx) = id_val.rfind('=') {
                                &id_val[idx + 1..]
                            } else {
                                id_val
                            };
                            
                            // aa.ts의 cleanNumber()와 동일하게 하이픈, 언더바, 온점, 쉼표만 제거
                            let clean_str = extracted.replace("-", "").replace("_", "").replace(".", "").replace(",", "");
                            if !clean_str.is_empty() {
                                item_meta.as_object_mut().unwrap().insert("id".to_string(), json!(clean_str.trim()));
                            }
                        }

                        let mut item_json = item_meta;

                        if !item_json.is_null() && (item_json.is_object() || item_json.is_array()) {
                            
                            if let Some(link_val) = item_json.get_mut("link") {
                                if let Some(relative_path) = link_val.as_str() {
                                    if let Ok(base_url) = url::Url::parse(&url) {
                                        if let Ok(absolute_url) = base_url.join(relative_path) {
                                            let path_query = format!("{}{}", absolute_url.path(), absolute_url.query().map(|q| format!("?{}", q)).unwrap_or_default());
                                            *link_val = json!(path_query.to_lowercase());
                                        }
                                    }
                                }
                            }
                            
                            all_extracted_items.push(item_json);
                        }
                    },
                    (Err(e), _, _) => println!("[Scheduler] Error extracting item meta: {:?}", e),
                    (_, Err(e), _) => println!("[Scheduler] Error extracting item info: {:?}", e),
                    (_, _, Err(e)) => println!("[Scheduler] Error extracting item data: {:?}", e),
                }

                let q3_clear_arc = model.qwen3_generator.clone();
                let _ = tokio::task::spawn_blocking(move || {
                    if let Some(gen) = q3_clear_arc.blocking_lock().as_mut() {
                        gen.clear_kv_cache();
                    }
                }).await;
                
                
                if !model.is_cpu_mode {
                    let dev = model.device_config.device.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        if dev.is_cuda() { let _ = dev.synchronize(); }
                    }).await;
                }

                // IO 작업 대기
                crate::models::qwen::generate::wait_for_global_io().await;

                // OS 커널 레벨에서 가비지 컬렉터 강제 호출
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

                tokio::time::sleep(std::time::Duration::from_millis(400)).await;
            }
        }
        extracted_data = json!({ "items": all_extracted_items, "type": page_type, "detail": false });

    } else {
        // [DETAIL MODE] Disk Bridge Relay
        println!("[Scheduler] Starting DISK BRIDGE RELAY for Details");
        
        let content_pug = {
            let clean_content = &clean_html_content;
            
            
            let full_pug = parsing::convert_to_clean_pug(clean_content, PugMode::DetailMode, Some(&url));
            
            
            model.truncate_pug_context(&full_pug, true, 2000, None).await
        };

        if !content_pug.trim().is_empty() {
            let extraction_instruction = parsing::item2json(&page_type, &url, language);
            let snapshot_id = format!("{}_detail", task.id);

            // 1. [Large] Load & Generate (Direct Qwen3.5 0.8B-Layer Generation)
            {
                
                model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancellation_token.clone()), false, kv_name.clone()).await?;

                
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                let params = ChatCompletionParameters {
                    messages: vec![
                        ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                            
                            content: format!("[PUG CONTENT]\n{}", content_pug),
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

                
                if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
                    println!("[Scheduler] Qwen3.5 Step C: Asking extraction question...");
                    
                    
                    let payload = json!({ "task_id": task.id, "category": "AI Inference", "summary": "Preparing AI engine...", "spinner": "⠋" });
                    let _ = app_handle.emit("extraction-progress", &payload);
                    emit_term("[STAGE-3] Preparing AI engine...");
                    
                    
                    let res = gen.generate_part(&params, false, 0, Some(cancellation_token.clone()), None, Some(snapshot_id.clone()), kv_name.clone()).await?;
                    
                    println!("[DEBUG-SCHED] Step C Raw Response: '{}'", res.text);

                    let mut parsed_json = parsing::parse_json_from_llm(&res.text);
                    
                    
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

    // --- DB OPS & SIDE EFFECTS ---
    
    let search_mode_str = search_mode.clone();
    let normalize_data = |item: &mut serde_json::Value| {
        if let Some(obj) = item.as_object_mut() {
            if obj.get("type").is_none() { obj.insert("type".to_string(), json!(page_type.clone())); }
            
            if obj.get("mode").is_none() { obj.insert("mode".to_string(), json!(search_mode_str.clone())); }
            
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
            
            
            let date_keys = [
                "registration_date", "order_date", "payment_date", "shipping_date", 
                "manufacture_date", "expiration_date", "release_date", "started_at", "expired_at"
            ];
            if let Ok(re_date) = regex::Regex::new(r"\d+") {
                for key in date_keys.iter() {
                    if let Some(date_val) = obj.get(*key).and_then(|v| v.as_str()) {
                        let s = date_val.trim();
                        if !s.is_empty() && s != "null" {
                            // 1. Unix Timestamp 감지 (순수 숫자 10자리 혹은 13자리)
                            if s.chars().all(char::is_numeric) && (s.len() == 10 || s.len() == 13) {
                                if let Ok(ts) = s.parse::<i64>() {
                                    let ts_ms = if s.len() == 10 { ts * 1000 } else { ts };
                                    if let Some(dt) = chrono::NaiveDateTime::from_timestamp_millis(ts_ms) {
                                        let iso_date = dt.format("%Y-%m-%dT%H:%M:%S").to_string();
                                        obj.insert(key.to_string(), json!(iso_date));
                                        continue;
                                    }
                                }
                            }

                            // 2. 이미 완벽한 ISO 포맷인 경우 스킵 (T 포함, 글자수 충분)
                            if s.contains('T') && s.len() >= 19 {
                                continue;
                            }

                            // 3. 다양한 형태의 문자열 분해 및 논리적 역추론 (MM/DD/YYYY, YY-MM-DD 등)
                            let nums: Vec<u32> = re_date.find_iter(s).filter_map(|m| m.as_str().parse().ok()).collect();
                            if nums.len() >= 3 {
                                let mut year = nums[0];
                                let mut month = nums[1];
                                let mut day = nums[2];

                                // MM/DD/YYYY 또는 DD/MM/YYYY 형태 대응 (마지막 숫자가 31을 초과하면 연도로 간주)
                                if day > 31 && year <= 31 {
                                    year = nums[2];
                                    day = nums[1]; // 월/일 판별은 모호하므로 순서 유지
                                    month = nums[0];
                                }

                                // 2자리 연도 보정 (예: 24 -> 2024, 99 -> 1999)
                                if year < 100 {
                                    year += if year > 50 { 1900 } else { 2000 };
                                }
                                
                                month = month.clamp(1, 12);
                                day = day.clamp(1, 31);
                                
                                let hour = if nums.len() > 3 { nums[3].clamp(0, 23) } else { 0 };
                                let minute = if nums.len() > 4 { nums[4].clamp(0, 59) } else { 0 };
                                let second = if nums.len() > 5 { nums[5].clamp(0, 59) } else { 0 };
                                
                                let iso_date = format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", year, month, day, hour, minute, second);
                                obj.insert(key.to_string(), json!(iso_date));
                            }
                        }
                    } else if let Some(date_num) = obj.get(*key).and_then(|v| v.as_i64()) {
                        // LLM이 문자열이 아닌 정수형(Unix Time)으로 뱉어냈을 경우 방어
                        let ts_ms = if date_num < 10_000_000_000 { date_num * 1000 } else { date_num };
                        if let Some(dt) = chrono::NaiveDateTime::from_timestamp_millis(ts_ms) {
                            let iso_date = dt.format("%Y-%m-%dT%H:%M:%S").to_string();
                            obj.insert(key.to_string(), json!(iso_date));
                        }
                    }
                }
            }

            // 날짜 기본값(Fallback) 매핑 (비어있는 경우도 확실히 체크)
            if obj.get("started_at").is_none() || obj.get("started_at").unwrap().is_null() || obj.get("started_at").unwrap().as_str() == Some("") {
                if let Some(m) = obj.get("manufacture_date").cloned() { obj.insert("started_at".to_string(), m); }
            }
            if obj.get("expired_at").is_none() || obj.get("expired_at").unwrap().is_null() || obj.get("expired_at").unwrap().as_str() == Some("") {
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

    
    {
        println!("[Scheduler] Generating natural language sentences for FTS/Vector matching and Privacy Masking...");
        
        // [PRIVACY] goods(상품) 타입은 개인정보가 없으므로 필터를 우회하여 속도를 최적화합니다.
        let should_mask = page_type != "goods";

        if is_detail {
            let original_lang_text = parsing::json_to_natural_language(&extracted_data);
            
            // [PRIVACY] AI를 통한 개인정보 마스킹 로직 주입 (조건부)
            let masked_lang_text = original_lang_text.clone(); // 마스킹은 Push 단계에서 동적 수행됨

            if let Some(obj) = extracted_data.as_object_mut() {
                obj.insert("text".to_string(), json!(original_lang_text));
                obj.insert("masked_text".to_string(), json!(masked_lang_text));
            }
        } else {
            if let Some(items) = extracted_data.get_mut("items").and_then(|v| v.as_array_mut()) {
                for item in items.iter_mut() {
                    let original_lang_text = parsing::json_to_natural_language(item);
                    
                    // [PRIVACY] AI를 통한 개인정보 마스킹 로직 주입 (조건부)
                    let masked_lang_text = original_lang_text.clone(); // 마스킹은 Push 단계에서 동적 수행됨

                    if let Some(obj) = item.as_object_mut() {
                        obj.insert("text".to_string(), json!(original_lang_text));
                        obj.insert("masked_text".to_string(), json!(masked_lang_text));
                    }
                }
            }
        }
    }

    // --- PHASE 3: HANDOVER (Unload Qwen -> Load Embedding) ---
    {
        println!("[Scheduler] PHASE 3: Handover - Unloading, Preparing for Embedding...");
        
        log_task_progress(app_handle, &task.id, &json!({ "category": "Handover", "summary": "Switching to Embedding model...", "spinner": "⠋" }));
        
        // 1. Explicitly Unload to free VRAM for Embedding Model
        model.deep_purge_resources().await;
        
        // 2. Wait for VRAM to settle (Driver latency)
        wait_for_resources_settled(1200, 800, Some(cancellation_token)).await?;
    }

    // [PARITY] ID Generation
    let id_val_raw = extracted_data.get("id")
        .or_else(|| extracted_data.get("no"))
        .or_else(|| extracted_data.get("code"))
        .or_else(|| extracted_data.get("tracking_number"))
        .or_else(|| extracted_data.get("index"))
        .and_then(|v| if v.is_number() { Some(v.to_string()) } else { v.as_str().map(|s| s.to_string()) })
        .unwrap_or_default();
    
    
    let clean_no = crate::utils::hash::normalize_numeric_homoglyphs(&id_val_raw)
        .replace("-", "").replace("_", "").replace(".", "").replace(",", "");
    
    let index_val = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("{}{}{}", page_type, team_id, clean_no)));
    let generated_id = crate::utils::hash::hash_id(&format!("{}{}", team_id, index_val));

    if let Some(obj) = extracted_data.as_object_mut() {
        obj.insert("index".to_string(), json!(index_val));
        obj.insert("id".to_string(), json!(generated_id.clone()));
        
        obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
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
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                let g_no = good.get("id").or_else(|| good.get("no")).and_then(|v| v.as_str()).unwrap_or("");
                if !g_no.is_empty() {
                    let clean_g_no = crate::utils::hash::normalize_numeric_homoglyphs(g_no).replace("-", "").replace("_", "");
                    
                    
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
                    
                    
                    let tracking_text = parsing::json_to_natural_language(&tracking_data);
                    let masked_tracking_text = tracking_text.clone(); // 마스킹은 Push 단계에서 동적 수행됨
                    let tracking_vector = model.get_embedding(tracking_text.clone()).await.unwrap_or(vec![0.0; 768]);
                    
                    tracking_data.as_object_mut().unwrap().insert("text".to_string(), json!(tracking_text));
                    tracking_data.as_object_mut().unwrap().insert("masked_text".to_string(), json!(masked_tracking_text));
                    
                    let _ = store.upsert_item(
                        "tracking", &tracking_id, "tracking", tracking_data.clone(), Some(tracking_vector.clone()),
                        Some(&task.from), Some(&team_id), Some(&task.cc),
                        Some(&crate::utils::hash::hash_id(&format!("tracking{}", cc_val))),
                        Some(&crate::utils::hash::hash_id(&format!("{}{}{}", team_id, task.cc, task.r#ref))),
                        None
                    ).await;

                    
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

    
    let target_table = match page_type.as_str() {
        "sales" | "goods" | "order" => "sales",
        "tracking" | "receiving" | "shipping" => "tracking",
        "event" | "coupon" => "event",
        "member" | "team" | "user" => "users",
        "talk" | "prompt" | "ai_search" => "talks",
        _ => "items",
    }.to_string();

    let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
    let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_val));
    let ref_val = task.r#ref.clone();

    let mut items_to_process = Vec::new();
    let mut stats_diff: std::collections::HashMap<String, (i64, i64, i64)> = std::collections::HashMap::new();

    if is_detail {
        
        // Phase 2.5에서 주입된 영문 FTS 키워드가 포함된 text 속성을 최우선으로 사용하여 벡터화합니다.
        let text_to_embed = extracted_data.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| parsing::json_to_natural_language(&extracted_data));
        let item_digest = crate::utils::hash::digest(&text_to_embed); 
        let mut target_id = generated_id.clone(); 
        
        let mut existing_vector = None;
        let mut is_new = true;
        let mut was_draft = false;

        
        if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &target_id).await {
            is_new = false;
            was_draft = if existing_item.updated_at_ts == 0 {
                true
            } else if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&existing_item.json_data) {
                !json_val.get("detail").and_then(|v| v.as_bool()).unwrap_or(true)
            } else {
                false
            };
            
            if existing_item.digest == item_digest {
                existing_vector = Some(existing_item.vector);
            }

            
            if let Ok(existing_json) = serde_json::from_str::<serde_json::Value>(&existing_item.json_data) {
                extracted_data = merge_node(&existing_json, &extracted_data);
            }
        } 
        
        else if !url.is_empty() {
            let normalized_link = if let Ok(parsed_url) = url::Url::parse(&url) {
                format!("{}{}", parsed_url.path(), parsed_url.query().map(|q| format!("?{}", q)).unwrap_or_default()).to_lowercase()
            } else {
                url.clone()
            };
            if let Ok(Some((found_id, json_val))) = store.find_item_by_property(&target_table, "link", &json!(normalized_link)).await {
                target_id = found_id.clone();
                is_new = false;
                
                if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &target_id).await {
                    was_draft = if existing_item.updated_at_ts == 0 {
                        true
                    } else if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&existing_item.json_data) {
                        !json_val.get("detail").and_then(|v| v.as_bool()).unwrap_or(true)
                    } else {
                        false
                    };
                    
                    if existing_item.digest == item_digest {
                        existing_vector = Some(existing_item.vector);
                    }
                }
                
                
                extracted_data = merge_node(&json_val, &extracted_data);
                if let Some(obj) = extracted_data.as_object_mut() {
                    obj.insert("id".to_string(), json!(target_id.clone()));
                }
            }
        }

        
        // 새 항목이면 pages: count++, global: count++
        // 기존 Draft 항목이었다면 pages: draft--, count++ 승급 (global은 이미 리스트에서 올렸으므로 변동 없음)
        if is_new {
            let e = stats_diff.entry(page_type.clone()).or_insert((0, 0, 0));
            e.1 += 1; // pages count++
            e.2 += 1; // global count++
        } else if was_draft {
            let e = stats_diff.entry(page_type.clone()).or_insert((0, 0, 0));
            e.0 -= 1; // pages draft--
            e.1 += 1; // pages count++
        }

        
        let vector = if let Some(v) = existing_vector {
            Some(v)
        } else {
            Some(model.get_embedding(text_to_embed).await?)
        };

        
        let related_types = crate::logic::related(&page_type);
        for foreign_type in related_types {
            if let Some((queries, merge_rule)) = crate::logic::relay(foreign_type, &extracted_data) {
                for q in queries {
                    match store.find_item_by_property(&q.table, &q.column, &q.value).await {
                        Ok(Some((foreign_id, mut foreign_data))) => {
                            let was_foreign_draft = foreign_data.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(0) == 0;
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
                                if was_foreign_draft && merge_rule.update.as_ref().map_or(false, |u| u.to == foreign_type) {
                                    let e = stats_diff.entry(foreign_type.to_string()).or_insert((0, 0, 0));
                                    e.0 -= 1; // pages draft--
                                    e.1 += 1; // pages count++
                                    
                                    e.2 += 1; // global count++
                                    foreign_data.as_object_mut().unwrap().insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                }
                                let merged_text = parsing::json_to_natural_language(&foreign_data);
                                let masked_merged_text = merged_text.clone(); // 마스킹은 Push 단계에서 동적 수행됨
                                let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 768]);
                                
                                foreign_data.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text));
                                foreign_data.as_object_mut().unwrap().insert("masked_text".to_string(), json!(masked_merged_text));

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
                            
                            let e = stats_diff.entry(foreign_type.to_string()).or_insert((0, 0, 0));
                            e.0 += 1; // pages draft++
                            e.2 += 1; // global count++

                            let mut draft_data = json!({});
                            
                            
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

        
        // 여기서 다시 덮어씌우는 과정을 생략하여 보호합니다.

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
        
        if let Some(items) = extracted_data.get("items").and_then(|v| v.as_array()) {
            for item_val in items.iter() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                let mut single_item = item_val.clone();
                
                
                let original_id = single_item.get("id")
                    .or_else(|| single_item.get("no"))
                    .or_else(|| single_item.get("code"))
                    .or_else(|| single_item.get("tracking_number"))
                    .or_else(|| single_item.get("index"))
                    .and_then(|v| if v.is_number() { Some(v.to_string()) } else { v.as_str().map(|s| s.to_string()) })
                    .unwrap_or_else(|| single_item.get("link").and_then(|v| v.as_str()).unwrap_or("").to_string());
                
                
                let clean_no = crate::utils::hash::normalize_numeric_homoglyphs(&original_id)
                    .replace("-", "").replace("_", "").replace(".", "").replace(",", "");
                
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
                    
                    obj.insert("updated_at".to_string(), json!(0));
                }

                // Phase 2.5에서 주입된 영문 FTS 키워드가 포함된 text 속성을 최우선으로 사용하여 벡터화합니다.
                let text_to_embed = single_item.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| parsing::json_to_natural_language(&single_item));
                let item_digest = crate::utils::hash::digest(&text_to_embed);
                
                let mut existing_vector = None;
                let mut is_new = true;
                // let is_fully_processed = false;

                if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &hashed_item_id).await {
                    is_new = false;
                    // 이미 상세 페이지에서 처리되어 updated_at이 0보다 큰지 확인
                    // if existing_item.updated_at_ts > 0 {
                    //     is_fully_processed = true;
                    // }
                    if existing_item.digest == item_digest {
                        existing_vector = Some(existing_item.vector);
                    }
                }

                
                // 새 항목이면 pages: draft++, global: count++
                if is_new {
                    let e = stats_diff.entry(page_type.clone()).or_insert((0, 0, 0));
                    e.0 += 1; // pages draft++
                    e.2 += 1; // global count++
                }
                
                let vector = if let Some(v) = existing_vector {
                    Some(v)
                } else {
                    Some(model.get_embedding(text_to_embed).await?)
                };

                
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
                                        let masked_merged_text = merged_text.clone(); // 마스킹은 Push 단계에서 동적 수행됨
                                        let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 768]);
                                        
                                        foreign_data.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text));
                                        foreign_data.as_object_mut().unwrap().insert("masked_text".to_string(), json!(masked_merged_text));

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
                                    
                                    let e = stats_diff.entry(foreign_type.to_string()).or_insert((0, 0, 0));
                                    e.0 += 1; // pages draft++
                                    e.2 += 1; // global count++

                                    let mut draft_data = json!({});
                                    
                                    
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

                
                // 여기서 다시 덮어씌우는 과정을 생략하여 보호합니다.

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
        let _ = update_team_base_metrics(&store, &team_id, &task.cc, &items_to_process, stats_diff.clone()).await;
        println!("[PROCESS] Metrics Engine updated base statistics for {} items. (Stats Diff: {:?})", items_to_process.len(), stats_diff);
    }

    // Final Status Update
    let _ = store.update_message_status(&task.id, logic::parse_status("complete"), Some("Extraction Complete")).await;

    
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

pub fn log_task_progress(app: &tauri::AppHandle, task_id: &str, payload: &serde_json::Value) {
    use std::io::Write;
    use tauri::Emitter;

    
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
    items: &Vec<serde_json::Value>,
    stats_diff: std::collections::HashMap<String, (i64, i64, i64)>,
) -> anyhow::Result<()> {
    let (team_json_str, team_vector, t_from, t_to, t_cc, t_bcc, t_ref, t_digest) = match store.get_item_by_id("users", team_id).await {
        Ok(Some(doc)) => (doc.json_data, doc.vector, doc.from, doc.to, doc.cc, doc.bcc, doc.r#ref, doc.digest),
        _ => (
            json!({ "base": { "pages": {} } }).to_string(),
            vec![0.0; 768],
            "".to_string(), "".to_string(), "".to_string(), "".to_string(), "".to_string(), "".to_string()
        )
    };

    
    let mut parsed_val: serde_json::Value = serde_json::from_str(&team_json_str).unwrap_or(json!({ "base": { "pages": {} } }));
    
    
    while let Some(inner_str) = parsed_val.get("json_data").and_then(|v| v.as_str()) {
        if let Ok(inner_obj) = serde_json::from_str(inner_str) {
            parsed_val = inner_obj;
        } else {
            break;
        }
    }
    let mut team_data = parsed_val;
    
    
    if let Some(obj) = team_data.as_object_mut() {
        obj.remove("json_data");
    }
    
    // --- [블록 1 & 2: 맵 순회로 모든 타입의 통계 업데이트] ---
    for (t_name, (pages_draft_diff, pages_count_diff, global_count_diff)) in stats_diff.iter() {
        // 페이지별 통계 업데이트
        {
            let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
            let pages = base.entry("pages").or_insert(json!({})).as_object_mut().unwrap();
            let cc_node = pages.entry(task_cc).or_insert(json!({})).as_object_mut().unwrap();
            let page_type_node = cc_node.entry(t_name).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();

            let current_draft = page_type_node.get("draft").and_then(|v| v.as_i64()).unwrap_or(0);
            let current_count = page_type_node.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            
            page_type_node.insert("draft".to_string(), json!(0.max(current_draft + pages_draft_diff)));
            page_type_node.insert("count".to_string(), json!(0.max(current_count + pages_count_diff)));
        } 

        // 글로벌 전체 통계 업데이트 (aa.ts와 동일하게 draft는 건드리지 않고 count만 누적)
        {
            let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
            let global_type_node = base.entry(t_name).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();
            
            let global_count = global_type_node.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            
            // 글로벌 draft는 클라우드 로직 상 사용되지 않으므로 보존하거나 건드리지 않습니다.
            global_type_node.insert("count".to_string(), json!(0.max(global_count + global_count_diff)));
        }
    }

    // Min/Max 업데이트는 items 내의 데이터에 한해서 진행
    {
        let properties = [
            "price", "quantity", "width", "height", "length", "weight", "shipping_fee", 
            "shipping_duration", "sale_price", "supply_price", "low_stock_threshold", 
            "discount", "min_order_amount", "max_discount_amount", "usage_limit", 
            "usage_per", "started_at", "expired_at"
        ];

        for item in items {
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
            let base = team_data.as_object_mut().unwrap().entry("base").or_insert(json!({ "pages": {} })).as_object_mut().unwrap();
            let global_type_node = base.entry(item_type).or_insert(json!({ "draft": 0, "count": 0 })).as_object_mut().unwrap();

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

                    
                    let prop_node = global_type_node.entry(*prop).or_insert(json!({ "min": 0.0, "max": 0.0 })).as_object_mut().unwrap();
                    
                    let current_min = prop_node.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let current_max = prop_node.get("max").and_then(|v| v.as_f64()).unwrap_or(0.0);

                    
                    
                    if current_min == 0.0 || num_val <= current_min { prop_node.insert("min".to_string(), json!(num_val)); }
                    if current_max == 0.0 || num_val >= current_max { prop_node.insert("max".to_string(), json!(num_val)); }
                }
            }
        }
    } // 👈 여기서 두 번째 참조가 종료됩니다.

    
    if let Some(base_json) = team_data.get("base") {
        println!("\n[DEBUG-METRICS] 최종 반영된 Base JSON 값:\n{}", serde_json::to_string_pretty(base_json).unwrap_or_default());
    }

    
    if let Some(obj) = team_data.as_object_mut() {
        obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
    }

    // 5. Save back to DB (digest 파라미터에 None을 전달하여 강제 쓰기를 유도합니다)
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
        None
    ).await;

    
    if let Ok(Some(saved_doc)) = store.get_item_by_id("users", team_id).await {
        println!("\n==================================================");
        println!("✅ [DB-VERIFY] DB에 통계(Team) 데이터가 100% 정상 저장되었습니다!");
        println!("- 타겟 ID: {}", saved_doc.id);
        println!("- 갱신된 Timestamp: {}", saved_doc.updated_at_ts);
        
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&saved_doc.json_data) {
            if let Some(base_stats) = parsed.get("base") {
                println!("- DB 내 실제 Base 통계:\n{}", serde_json::to_string_pretty(base_stats).unwrap_or_default());
            }
        }
        println!("==================================================\n");
    } else {
        println!("\n==================================================");
        println!("🚨 [DB-VERIFY] 치명적 오류: DB에 Team 데이터가 저장되지 않았습니다!");
        println!("==================================================\n");
    }

    Ok(())
}



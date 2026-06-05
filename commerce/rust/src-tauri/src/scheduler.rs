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

// 추출된 DOM 뭉치가 실제 원본 PUG의 몇 번째 줄부터 몇 번째 줄까지인지 정확하게 매핑합니다.
pub fn find_block_indices_in_pug<S: AsRef<str>>(full_lines: &[S], block_pug: &str) -> Option<(usize, usize)> {
    let b_lines: Vec<&str> = block_pug.lines().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if b_lines.is_empty() { return None; }
    
    for i in 0..full_lines.len() {
        if full_lines[i].as_ref().trim() == b_lines[0] {
            let mut match_count = 1;
            let mut j = i + 1;
            let mut k = 1;
            while j < full_lines.len() && k < b_lines.len() {
                if full_lines[j].as_ref().trim().is_empty() { j += 1; continue; }
                if full_lines[j].as_ref().trim() == b_lines[k] { match_count += 1; k += 1; } 
                else { break; }
                j += 1;
            }
            if match_count == b_lines.len() { return Some((i, j - 1)); }
        }
    }
    None
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
    let mut doc_lang = "en".to_string(); // 🌟 신규 다국어 감지 변수 추가

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

    // 🌟 [CRITICAL FIX] 텍스트 추출 시작 단계에서는 임베딩 모델의 "파일 다운로드 여부"만 가볍게 확인합니다.
    // 실제 텐서 메모리 로딩은 추후 AI가 분석을 끝내고 추출 단계(Stage 3)에 진입하여 모델이 '진짜 쓰일 때' 지연 로딩됩니다!
    if task.r#type != "image_extraction" && task.r#type != "analytic_extraction" {
        model.check_embedding_downloaded().await?;
    }

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
                    // 🌟 [CRITICAL FIX] 토큰 한도를 32로 늘려 언어 코드(ko)가 중간에 잘리는 현상을 완벽히 방지합니다.
                    max_tokens: Some(32),
                    temperature: Some(0.0), top_p: Some(0.95),
                    ..Default::default()
                };

                let res = if base_model_size == crate::model::ModelSize::Qwen {
                    // [핵심] Step A가 아니라 '미리 구워둔 Base' 스냅샷을 불러옵니다!
                    model.secure_vram_relay(crate::model::ModelSize::Qwen, Some(&base_session_id), Some(cancellation_token.clone()), false, kv_name.clone()).await?;

                    if let Some(gen) = model.generator.lock().await.as_mut() {
                        println!("[Scheduler] 0.6B Step A: Asking classification question...");
                        // 🎯 [SEMANTIC STEERING] 편향 주입 삭제
                        gen.generate(params, Some(cancellation_token.clone()), Some(snapshot_id.clone()), kv_name.clone(), None).await?
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
                            gen.generate(params, Some(cancel_clone), None, None).map_err(|e| anyhow::anyhow!("Qwen3 failed: {}", e))
                        } else {
                            Err(anyhow::anyhow!("Qwen3 generator missing"))
                        }
                    }).await??
                };

                println!("[DEBUG-SCHED] Step A Raw Response: '{}'", res);
                
                let type_info = parsing::parse_json_from_llm(&res); 
                
                if let Some(type_val) = type_info.get("type").and_then(|v| v.as_str()) {
                    page_type = type_val.to_lowercase();
                }
                if let Some(lang_val) = type_info.get("language").and_then(|v| v.as_str()) {
                    doc_lang = lang_val.trim().to_lowercase();
                }

                // 🌟 [CRITICAL FIX] bias.json (BIAS_DICT)에 해당 언어 코드가 등록되어 있지 않다면 심플하게 'ko'로 고정합니다.
                if crate::parsing::BIAS_DICT.get(doc_lang.as_str()).is_none() {
                    doc_lang = "ko".to_string();
                }

                println!("[Scheduler] Detected language in Step A: {}", doc_lang);
            }
            
            if page_type.is_empty() || page_type == "unknown" { 
                model.deep_purge_resources().await;
                return Ok(()); 
            }
        }

        // --- STEP A-2: DETAIL CLASSIFICATION (디테일 페이지 여부 판별) ---
        {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            println!("[Scheduler] Starting DISK BRIDGE RELAY (Load Base -> Is Detail)");

            // 🎯 [Track A: NOISE FILTER] Boa Engine을 통해 부모 엘리먼트를 찾고 뭉치 단위로 노이즈를 필터링합니다.
            let (list_bias, form_bias, layout_prejudice) = crate::parsing::get_separated_layout_bias(&page_type, &doc_lang);
            let prej_emb = model.get_embedding(layout_prejudice.clone()).await.unwrap_or(vec![0.0; 384]);
            let list_bias_emb = model.get_embedding(list_bias.clone()).await.unwrap_or(vec![0.0; 384]);
            let form_bias_emb = model.get_embedding(form_bias.clone()).await.unwrap_or(vec![0.0; 384]);
            
            // 🌟 [CRITICAL FIX] 소유권 에러(E0506)를 방지하기 위해 String 복제본으로 소유권을 가집니다.
            let pug_lines: Vec<String> = light_pug.lines().map(|s| s.to_string()).collect();
            let mut line_embeddings = Vec::new();
            
            // emit_term(&format!("\n[PRE-FILTER] Vectorizing context for Track A ({} lines)...", pug_lines.len()));
            for (line_idx, line) in pug_lines.iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                if line.trim().is_empty() {
                    line_embeddings.push(vec![0.0; 384]);
                    continue;
                }
                
                // 🌟 진행 상황을 터미널에 낱낱이 출력하여 멈춘 것처럼 보이지 않게 합니다.
                // emit_term(&format!("  [VECTORIZING] Track A Line {}/{} : {}", line_idx + 1, pug_lines.len(), line.trim()));
                
                let emb = model.get_embedding(line.to_string()).await.unwrap_or(vec![0.0; 384]);
                line_embeddings.push(emb);
            }

            // HTML 문서를 파싱하여 Nodes JSON 문자열을 바인딩하고, 스레드 안전성이 없는 scraper::Html 객체는 즉시 소멸하도록 생명주기를 블록 내로 한정합니다.
            let nodes_str = {
                let document_for_boa = scraper::Html::parse_document(&clean_html_content);
                let mut nodes_json = Vec::new();
                let mut node_to_idx = std::collections::HashMap::new();
                for (idx, node) in document_for_boa.tree.root().descendants().enumerate() {
                    node_to_idx.insert(node.id(), idx);
                }
                for (idx, node) in document_for_boa.tree.root().descendants().enumerate() {
                    if let Some(el) = node.value().as_element() {
                        let parent_idx = node.parent().and_then(|p| node_to_idx.get(&p.id())).map(|&i| i as i32).unwrap_or(-1);
                        let text: String = node.children()
                            .filter_map(|child| child.value().as_text().map(|t| t.to_string()))
                            .collect::<Vec<_>>().join(" ").trim().to_string();
                        nodes_json.push(serde_json::json!({
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
                        nodes_json.push(serde_json::json!(serde_json::Value::Null));
                    }
                }
                serde_json::to_string(&nodes_json).unwrap_or_default()
            };
            let js_template = get_boa_block_extractor_template(); // 🌟 Batch 처리용 템플릿 사용

            let mut wiped_indices = vec![false; pug_lines.len()];
            let mut processed_blocks = std::collections::HashSet::new();

            // 🌟 [최적화] 노이즈로 의심되는 텍스트 후보들을 먼저 싹 다 모읍니다.
            let mut track_a_candidates = Vec::new();
            let mut track_a_indices = Vec::new();

            for line_idx in 0..pug_lines.len() {
                if wiped_indices[line_idx] { continue; }
                let line = &pug_lines[line_idx];
                if line.trim().is_empty() { continue; }
                
                let line_prej_score = cosine_similarity(&prej_emb, &line_embeddings[line_idx]);
                
                if line_prej_score > 0.55 {
                    let text_part = if let Some(idx) = line.find('|') { line[idx + 1..].trim() } else { line.trim() };
                    if !text_part.is_empty() {
                        track_a_candidates.push(text_part.to_string());
                        track_a_indices.push(line_idx);
                    }
                }
            }

            // 🌟 [진행 로그 추가] 수집된 노이즈 후보군 개수를 명확히 보여줍니다.
            emit_term(&format!("  🔍 [Track A] Identified {} potential noise lines. Resolving DOM parents via Boa...", track_a_candidates.len()));

            // 🌟 [최적화] Boa Engine 1번만 켜서 전체 후보군의 부모 CSS Selector를 초고속으로 받아옵니다.
            let track_a_selectors: Vec<String> = {
                let target_len = track_a_candidates.len(); 
                let target_titles_str = serde_json::to_string(&track_a_candidates).unwrap_or_else(|_| "[]".to_string());
                let js_code = js_template
                    .replace("NODES_PLACEHOLDER", &nodes_str)
                    .replace("TARGET_TITLES_PLACEHOLDER", &target_titles_str);

                tokio::task::spawn_blocking(move || {
                    let mut context = boa_engine::Context::default();
                    if let Ok(val) = context.eval(boa_engine::Source::from_bytes(js_code.as_bytes())) {
                        if let Some(res_str) = val.as_string().map(|s| s.to_std_string_escaped()) {
                            if let Ok(arr) = serde_json::from_str::<Vec<String>>(&res_str) {
                                return arr;
                            }
                        }
                    }
                    vec![String::new(); target_len]
                }).await.unwrap_or_else(|_| vec![String::new(); target_len])
            };

            // 🌟 [결과 로그 추가] Boa Engine이 최종적으로 매칭해 낸 시맨틱 부모 뭉치의 개수를 출력합니다.
            let valid_selectors_count = track_a_selectors.iter().filter(|s| !s.is_empty()).count();
            emit_term(&format!("  📦 [Track A] Boa Engine successfully mapped {} structural parent blocks.", valid_selectors_count));

            // 반환받은 Selector들을 통해 VRAM 임베딩을 수행합니다.
            for (i, sel) in track_a_selectors.into_iter().enumerate() {
                if sel.is_empty() { continue; }
                let block_pug = crate::parsing::convert_to_clean_pug_selector(&clean_html_content, &sel, crate::parsing::PugMode::NoAttributesMode, None);
                
                if block_pug.is_empty() || processed_blocks.contains(&block_pug) { continue; }
                processed_blocks.insert(block_pug.clone());

                let block_emb = model.get_embedding(block_pug.clone()).await.unwrap_or(vec![0.0; 384]);
                
                let block_prej_score = cosine_similarity(&prej_emb, &block_emb);
                let block_list_score = cosine_similarity(&list_bias_emb, &block_emb);
                let block_form_score = cosine_similarity(&form_bias_emb, &block_emb);
                
                if block_prej_score > block_list_score && block_prej_score > block_form_score {
                    if let Some((start_idx, end_idx)) = find_block_indices_in_pug(&pug_lines, &block_pug) {
                        emit_term(&format!("  🚫 [NOISE BLOCK DELETED] Boa Engine Matched. Lines {}~{} (Prej: {:.4} > List: {:.4} & Form: {:.4})", start_idx + 1, end_idx + 1, block_prej_score, block_list_score, block_form_score));
                        for j in start_idx..=end_idx {
                            wiped_indices[j] = true;
                        }
                    }
                }
            }

            // Track A에 의해 청소된 결과물로 업데이트
            let mut filtered_light_pug = String::new();
            for (idx, line) in pug_lines.iter().enumerate() {
                if !wiped_indices[idx] { filtered_light_pug.push_str(line); }
                filtered_light_pug.push_str("\n");
            }
            light_pug = filtered_light_pug.trim_end().to_string();
            
            let system_content_a2 = format!("[PUG CONTENT]\n{}", light_pug);
            
            log_task_progress(app_handle, &task.id, &json!({ "category": "Classification", "summary": "Scoring DOM blocks to determine page type...", "spinner": "⠋" }));

            // 🎯 Track B & C: 상세/리스트 판별 (Boa Engine 일괄 처리 최적화)
            emit_term("\n[CLASSIFICATION] Track B & C Vector Matching (Batch DOM Blocks)...");
            
            let mut list_scores = Vec::new();
            let mut form_scores = Vec::new();

            for (i, emb) in line_embeddings.iter().enumerate() {
                if pug_lines[i].trim().is_empty() { continue; }
                list_scores.push((i, cosine_similarity(&list_bias_emb, emb)));
                form_scores.push((i, cosine_similarity(&form_bias_emb, emb)));
            }

            list_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            form_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // 🌟 [최적화] 표본 오차를 방지하기 위해 10개의 앵커(Top 5 리스트 + Top 5 폼)를 한 번에 묶습니다.
            let mut track_bc_candidates = Vec::new();
            let mut track_bc_indices = Vec::new();
            
            for (idx, _) in list_scores.iter().take(5) {
                let line = &pug_lines[*idx];
                let text = if let Some(p) = line.find('|') { line[p + 1..].trim() } else { line.trim() };
                track_bc_candidates.push(text.to_string());
                track_bc_indices.push(*idx);
            }
            for (idx, _) in form_scores.iter().take(5) {
                let line = &pug_lines[*idx];
                let text = if let Some(p) = line.find('|') { line[p + 1..].trim() } else { line.trim() };
                track_bc_candidates.push(text.to_string());
                track_bc_indices.push(*idx);
            }

            // Boa 엔진 1번 구동으로 10개의 선택자를 한 번에 추출합니다.
            let track_bc_selectors: Vec<String> = {
                let target_len = track_bc_candidates.len(); 
                let target_titles_str = serde_json::to_string(&track_bc_candidates).unwrap_or_else(|_| "[]".to_string());
                let js_code = js_template
                    .replace("NODES_PLACEHOLDER", &nodes_str)
                    .replace("TARGET_TITLES_PLACEHOLDER", &target_titles_str);

                tokio::task::spawn_blocking(move || {
                    let mut context = boa_engine::Context::default();
                    if let Ok(val) = context.eval(boa_engine::Source::from_bytes(js_code.as_bytes())) {
                        if let Some(res_str) = val.as_string().map(|s| s.to_std_string_escaped()) {
                            if let Ok(arr) = serde_json::from_str::<Vec<String>>(&res_str) {
                                return arr;
                            }
                        }
                    }
                    vec![String::new(); target_len]
                }).await.unwrap_or_else(|_| vec![String::new(); target_len])
            };

            // 🌟 [정량 로그 피드백 장착] 총 몇 개 중에서 몇 개의 의미 블록이 발굴되었는지 한눈에 보여줍니다.
            let valid_bc_count = track_bc_selectors.iter().filter(|s| !s.is_empty()).count();
            emit_term(&format!("  📦 [Track B & C] Boa Engine successfully mapped {}/{} structural processing blocks.", valid_bc_count, track_bc_candidates.len()));

            let mut total_list_score = 0.0;
            let mut processed_list_blocks = std::collections::HashSet::new();
            let mut total_form_score = 0.0;
            let mut processed_form_blocks = std::collections::HashSet::new();

            for (i, sel) in track_bc_selectors.into_iter().enumerate() {
                let is_list_track = i < 5;
                let track_name = if is_list_track { "TRACK B (LIST)" } else { "TRACK C (FORM)" };

                // 🌟 어떤 앵커 라인의 텍스트가 DOM 분석에 실패해서 버려졌는지 이유를 낱낱이 출력합니다.
                if sel.is_empty() { 
                    emit_term(&format!("  ⚠️ [{}] Anchor Line {} failed to resolve a valid structural parent block via DOM.", track_name, track_bc_indices[i] + 1));
                    continue; 
                }
                let block_pug = crate::parsing::convert_to_clean_pug_selector(&clean_html_content, &sel, crate::parsing::PugMode::NoAttributesMode, None);
                
                let is_list_track = i < 5;
                if is_list_track {
                    if block_pug.is_empty() || processed_list_blocks.contains(&block_pug) { continue; }
                    processed_list_blocks.insert(block_pug.clone());
                } else {
                    if block_pug.is_empty() || processed_form_blocks.contains(&block_pug) { continue; }
                    processed_form_blocks.insert(block_pug.clone());
                }

                let block_emb = model.get_embedding(block_pug).await.unwrap_or(vec![0.0; 384]);
                let b_prej_score = cosine_similarity(&prej_emb, &block_emb);
                
                if is_list_track {
                    let b_list_score = cosine_similarity(&list_bias_emb, &block_emb);
                    // 🌟 [CRITICAL FIX] 마이너스 점수는 노이즈이므로 0점 처리하여 진짜 뼈대의 점수를 깎아먹지 않게 합니다.
                    let final_score = (b_list_score - b_prej_score).max(0.0);
                    if final_score > 0.0 {
                        total_list_score += final_score;
                        emit_term(&format!("  📊 [TRACK B (LIST)] Anchor: {} | Bias: {:.4} | Prej: {:.4} | Sum: {:.4}", track_bc_indices[i] + 1, b_list_score, b_prej_score, final_score));
                    } else {
                        emit_term(&format!("  ⚠️ [TRACK B (LIST)] Anchor: {} Ignored (Prej {:.4} > Bias {:.4})", track_bc_indices[i] + 1, b_prej_score, b_list_score));
                    }
                } else {
                    let b_form_score = cosine_similarity(&form_bias_emb, &block_emb);
                    let final_score = (b_form_score - b_prej_score).max(0.0);
                    if final_score > 0.0 {
                        total_form_score += final_score;
                        emit_term(&format!("  📊 [TRACK C (FORM)] Anchor: {} | Bias: {:.4} | Prej: {:.4} | Sum: {:.4}", track_bc_indices[i] + 1, b_form_score, b_prej_score, final_score));
                    } else {
                        emit_term(&format!("  ⚠️ [TRACK C (FORM)] Anchor: {} Ignored (Prej {:.4} > Bias {:.4})", track_bc_indices[i] + 1, b_prej_score, b_form_score));
                    }
                }
            }

            // 3. 최종 판별 (Detail 여부 결정)
            is_detail = total_form_score > total_list_score;

            println!("[Scheduler] Classified is_detail as: {} (Total Form: {:.4}, Total List: {:.4})", is_detail, total_form_score, total_list_score);
            emit_term(&format!("  ✅ Determined Detail Page: {}", is_detail));
        } // 👈 🌟 [핵심 변경 1 끝] 0.6B 분석 블록 종료
    } // 🌟 [CRITICAL FIX] 누락된 if !skip_ai_analysis 블록 닫기 괄호를 복구합니다!

                        
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // 🌟 [CRITICAL FIX] 이어지는 추출 단계(Step B, List, Detail)에서 Qwen3를 그대로 재사용하므로, VRAM에서 강제로 모델을 내리지 않습니다.
    // secure_vram_relay 내부의 스마트 스위칭 로직이 모델 변경 여부를 감지하여 필요할 때만 교체합니다.
    // model.deep_purge_resources().await;
    
    // 🌟 [KV CACHE CLEAR] 모델 본체는 살려두고, 이전 단계 연산으로 팽창한 KV 캐시만 제거하여 VRAM을 확보합니다.
    {
        let q3_clear_arc = model.qwen3_generator.clone();
        let _ = tokio::task::spawn_blocking(move || {
            if let Some(gen) = q3_clear_arc.blocking_lock().as_mut() {
                gen.clear_kv_cache();
            }
        }).await;
        
        let gen_clear_arc = model.generator.clone();
        let _ = tokio::task::spawn_blocking(move || {
            if let Some(gen) = gen_clear_arc.blocking_lock().as_mut() {
                let _ = gen.clear_kv_cache();
            }
        }).await;

        if !model.is_cpu_mode {
            let dev = model.device_config.device.clone();
            let _ = tokio::task::spawn_blocking(move || {
                if dev.is_cuda() { let _ = dev.synchronize(); }
            }).await;
        }
    }
    
    // VRAM이 OS에서 완전히 반환될 때까지 잠시 대기합니다.
    wait_for_resources_settled(1200, 800, Some(&cancellation_token)).await?;

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
                            
                            // 🎯 [SEMANTIC STEERING] 상품 제목에 집중하도록 방향타 고정!
                            let (_title_bias, title_prej) = crate::parsing::get_title_bias(&page_type, &doc_lang);
                            gen.generate(
                                params, 
                                Some(cancellation_token.clone()), 
                                Some(snapshot_id.clone()), 
                                kv_name.clone(),
                                Some(&title_prej) 
                            ).await?
                        } else {
                            return Err(anyhow::anyhow!("Qwen generator missing"));
                        }
                    } else {
                        model.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancellation_token.clone()), false, None).await?;
                        let q3_gen_arc = model.qwen3_generator.clone();
                        let cancel_clone = cancellation_token.clone();
                        let (_title_bias, title_prej) = crate::parsing::get_title_bias(&page_type, &doc_lang);
                        tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
                            let mut gen_guard = q3_gen_arc.blocking_lock();
                            if let Some(gen) = gen_guard.as_mut() {
                                println!("[JS-BRIDGE] 1. Requesting titles from LLM (Qwen3)...");
                                gen.generate(params, Some(cancel_clone), None, Some(&title_prej)).map_err(|e| anyhow::anyhow!("Qwen3 failed: {}", e)) 
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

                    let js_template = get_boa_js_template();


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
                        // 🌟 [CRITICAL FIX] Qwen 3.5 생성기의 파라미터가 추가되었으므로 마지막 인자로 None(semantic_prejudice)을 추가로 전달합니다.
                        if let Ok(res) = gen.generate(params, Some(cancellation_token.clone()), Some(format!("{}_step_thead", task.id)), kv_name.clone(), None, None).await {
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

            // 리스트 전용 스키마 정의를 호출하여 핵심 필드만 개별 추출
            let fields = parsing::get_list_schema_fields(&page_type, &url, &doc_lang);
            let total_fields = fields.len();

            // 환각 검증을 위한 문서 타이틀 추출
            let doc_title = {
                let doc = scraper::Html::parse_document(&clean_html_content);
                if let Ok(sel) = scraper::Selector::parse("title") {
                    doc.select(&sel).next().map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string()).unwrap_or_default()
                } else {
                    String::new()
                }
            };

            // 모델 가중치 변경(스위칭) 없이 가장 가벼운 모델인 Qwen3 하나만으로 전체 파이프라인을 관통하여 속도를 극대화합니다!
            model.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancellation_token.clone()), false, Some("inference".to_string())).await?;

            // 🌟 [NOISE FILTER] bias.json의 layout_list.prejudice 값을 가져와 노이즈 필터링용 벡터를 생성합니다.
            let (_, layout_prejudice) = crate::parsing::get_layout_bias(&page_type, &doc_lang);
            let layout_prej_emb = model.get_embedding(layout_prejudice.clone()).await.unwrap_or(vec![0.0; 384]);

            // [최적화] thead는 모든 아이템에 공통으로 적용되므로 루프 바깥에서 단 한 번만 벡터화합니다.
            let mut thead_lines: Vec<String> = thead_pug.lines().map(|s| s.to_string()).collect();
            let mut thead_embeddings = Vec::new();
            if !thead_lines.is_empty() {
                emit_term(&format!("\n[PRE-PROCESSING] Vectorizing Table Header ({} lines)...", thead_lines.len()));
                for line_idx in 0..thead_lines.len() {
                    let line = thead_lines[line_idx].clone();
                    if line.trim().is_empty() {
                        thead_embeddings.push(vec![0.0; 384]);
                        continue;
                    }
                    let emb = match model.get_embedding(line.to_string()).await {
                        Ok(vector) => vector,
                        Err(_) => vec![0.0; 384],
                    };
                    
                    // 🌟 유사도 0.55 이상이면 PUG 텍스트를 비워버립니다.
                    let noise_score = cosine_similarity(&layout_prej_emb, &emb);
                    if noise_score > 0.55 {
                        emit_term(&format!("    🚫 [NOISE FILTERED] Header Line {} : {} (Score: {:.4})", line_idx + 1, line.trim(), noise_score));
                        thead_lines[line_idx] = String::new(); // 인덱스 보존을 위해 줄 내용만 삭제
                        thead_embeddings.push(vec![0.0; 384]);
                    } else {
                        thead_embeddings.push(emb);
                    }
                }
            }

            for (idx, item_pug) in pug_list.iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                
                let percent = (((idx as f32) / (total_items as f32)) * 100.0) as i32;
                let summary_msg = format!("Extracting item data ({}%)...", percent);
                
                let payload = json!({ 
                    "task_id": task.id, 
                    "category": format!("List Item {}/{}", idx + 1, total_items), 
                    "summary": summary_msg, 
                    "spinner": "⠋" 
                });
                log_task_progress(app_handle, &task.id, &payload);
                emit_term(&format!("\n[STAGE-3] Processing List Item {}/{} ...", idx + 1, total_items));

                // 헤더(Thead)와 개별 아이템(Item)의 PUG를 결합하여 검증 텍스트 생성 (나중의 본문 텍스트 검증용)
                let full_item_pug = format!("{}\n{}", thead_pug, item_pug);
                
                // [수정] 아이템 영역만 분리하여 가볍게 벡터화 진행 및 노이즈 필터링
                let mut item_lines: Vec<String> = item_pug.lines().map(|s| s.to_string()).collect();
                let mut item_embeddings = Vec::new();
                for line_idx in 0..item_lines.len() {
                    let line = item_lines[line_idx].clone();
                    if line.trim().is_empty() {
                        item_embeddings.push(vec![0.0; 384]);
                        continue;
                    }
                    
                    let emb = match model.get_embedding(line.to_string()).await {
                        Ok(vector) => vector,
                        Err(e) => {
                            emit_term(&format!("    🚨 [EMBEDDING ERROR] Failed to load or compute model: {}", e));
                            vec![0.0; 384]
                        }
                    };
                    
                    // 🌟 유사도 0.55 이상이면 PUG 텍스트를 비워버립니다.
                    let noise_score = cosine_similarity(&layout_prej_emb, &emb);
                    if noise_score > 0.55 {
                        emit_term(&format!("    🚫 [NOISE FILTERED] Item Line {}/{} : {} (Score: {:.4})", line_idx + 1, item_lines.len(), line.trim(), noise_score));
                        item_lines[line_idx] = String::new(); // 인덱스 보존을 위해 줄 내용만 삭제
                        item_embeddings.push(vec![0.0; 384]);
                    } else {
                        emit_term(&format!("    [VECTORIZING] Item Line {}/{} : {}", line_idx + 1, item_lines.len(), line.trim()));
                        item_embeddings.push(emb);
                    }
                }

                let mut item_val = json!({});
                let mut global_ignore_list: Vec<String> = Vec::new();
                
                // 🌟 String 벡터를 &str 배열로 참조 변환하여 이후 컨텍스트 추출기에 완벽 호환되게 연결합니다.
                let thead_lines_ref: Vec<&str> = thead_lines.iter().map(|s| s.as_str()).collect();
                let item_lines_ref: Vec<&str> = item_lines.iter().map(|s| s.as_str()).collect();

                // 상세 페이지와 완벽히 동일하게 필드별로 순회하며 개별 타격 추출
                for (f_idx, (field_name, field_desc, bias_target, prejudice_target)) in fields.clone().into_iter().enumerate() {
                    
                    // 2. get_list_schema_fields의 bias_target 값을 임베딩하여 인메모리 코사인 유사도 검색
                    let query_emb = model.get_embedding(bias_target.clone()).await.unwrap_or(vec![0.0; 384]);
                    
                    // Header 영역 독립 매칭
                    let mut best_thead_idx = 0;
                    let mut best_thead_score = -1.0;
                    for (i, emb) in thead_embeddings.iter().enumerate() {
                        if thead_lines_ref[i].trim().is_empty() { continue; }
                        let score = cosine_similarity(&query_emb, emb);
                        if score > best_thead_score {
                            best_thead_score = score;
                            best_thead_idx = i;
                        }
                    }
                    let matched_thead_pug = extract_pug_context(&thead_lines_ref, best_thead_idx);

                    // Item 본문 영역 독립 매칭
                    let mut best_item_idx = 0;
                    let mut best_item_score = -1.0;
                    for (i, emb) in item_embeddings.iter().enumerate() {
                        if item_lines_ref[i].trim().is_empty() { continue; }
                        let score = cosine_similarity(&query_emb, emb);
                        if score > best_item_score {
                            best_item_score = score;
                            best_item_idx = i;
                        }
                    }
                    let matched_item_pug = extract_pug_context(&item_lines_ref, best_item_idx);
                    
                    // 3. 찾은 Header 컨텍스트와 Item 컨텍스트를 하나로 결합
                    let targeted_pug = if matched_thead_pug.is_empty() {
                        matched_item_pug
                    } else {
                        format!("{}\n{}", matched_thead_pug, matched_item_pug)
                    };
                    
                    emit_term(&format!("    🎯 [MATCHED CONTEXT] Field: '{}' | Header Score: {:.4} | Item Score: {:.4}\n{}", field_name, best_thead_score, best_item_score, targeted_pug));
                    
                    let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                        content: format!("[PUG CONTENT]\n{}", targeted_pug),
                        name: None,
                    });
                    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                    
                    let f_percent = (((f_idx as f32) / (total_fields as f32)) * 100.0) as i32;
                    let f_summary_msg = format!("Extracting {} ({}%)...", field_name, f_percent);
                    
                    let payload = json!({ 
                        "task_id": task.id, 
                        "category": format!("List Item {}/{}", idx + 1, total_items), 
                        "summary": f_summary_msg, 
                        "spinner": "⠋" 
                    });
                    log_task_progress(app_handle, &task.id, &payload);
                    emit_term(&format!("  ▶ {}", f_summary_msg));

                    let task_question = parsing::extract_single_field_prompt(&page_type, &field_name, &field_desc, language, &doc_title);
                    
                    let mut ignore_list: Vec<String> = global_ignore_list.clone();
                    let mut miss_counter = 0;
                    
                    loop {
                        if cancellation_token.load(Ordering::Relaxed) { break; }

                        let q3_gen = model.qwen3_generator.clone();
                        let cancel_clone = cancellation_token.clone();
                        let sys_msg = system_message.clone();
                        
                        let field_name_clone = field_name.clone();
                        let bias_target_for_closure = bias_target.clone(); 
                        let prejudice_target_for_closure = prejudice_target.clone(); 
                        
                        let task_q = task_question.clone();
                        let ignore_list_clone = ignore_list.clone();
                        
                        let res = tokio::task::spawn_blocking(move || {
                            let mut gen_guard = q3_gen.blocking_lock();
                            if let Some(gen) = gen_guard.as_mut() {
                                let params = ChatCompletionParameters {
                                    messages: vec![
                                        sys_msg,
                                        ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                                            content: ChatCompletionRequestUserMessageContent::Text(task_q),
                                            name: None,
                                        })
                                    ],
                                    model: "qwen3".to_string(), max_tokens: Some(512), temperature: Some(0.0), top_p: Some(0.95),
                                    ..Default::default()
                                };
                                
                                let p_target = if prejudice_target_for_closure.is_empty() { None } else { Some(prejudice_target_for_closure.as_str()) };
                                
                                gen.generate(params, Some(cancel_clone), Some(&ignore_list_clone), p_target).map_err(|e| anyhow::anyhow!("Qwen 3 field extraction failed: {}", e))
                            } else {
                                Err(anyhow::anyhow!("Qwen 3 Generator not available"))
                            }
                        }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("Task join failed: {}", e)));

                        // 환각 캐시를 즉시 제거하여 다음 재시도 시 방어
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

                        match res {
                            Ok(res_text) => {
                                let mut parsed = parsing::parse_json_from_llm(&res_text);
                                let mut parsed_val = if let Some(inner) = parsed.get_mut(&page_type) { inner.take() } else { parsed };

                                let mut requires_retry = false;
                                let mut extracted_values_for_retry = Vec::new();
                                
                                let keys: Vec<&str> = field_name_clone.split(',').map(|s| s.trim()).collect();
                                let mut found_valid_value = false;

                                let skip_pug_match_fields = ["status", "payment_method", "payment_origin", "condition", "currency"];
                                let is_enum_field = skip_pug_match_fields.iter().any(|&f| field_name_clone.contains(f));

                                for k in &keys {
                                    if let Some(val) = parsed_val.get(*k) {
                                        let is_empty_val = match val {
                                            serde_json::Value::Null => true,
                                            serde_json::Value::String(s) => s.trim().is_empty() || s == "..." || s == "null",
                                            serde_json::Value::Array(a) => a.is_empty(),
                                            serde_json::Value::Object(o) => o.is_empty(),
                                            _ => false,
                                        };

                                        if !is_empty_val {
                                            found_valid_value = true;

                                            let extracted_str = if val.is_string() {
                                                val.as_str().unwrap_or("").trim().to_string()
                                            } else if val.is_number() {
                                                val.to_string()
                                            } else {
                                                String::new()
                                            };

                                            if !extracted_str.is_empty() && extracted_str != "..." && extracted_str != "null" {
                                                extracted_values_for_retry.push(extracted_str.clone());
                                                
                                                if !is_enum_field {
                                                    let is_iso_date = extracted_str.contains('T') && extracted_str.len() >= 19;
                                                    let is_url = extracted_str.starts_with("http") || extracted_str.starts_with('/');
                                                    let is_boolean_str = extracted_str == "true" || extracted_str == "false";
                                                    
                                                    if !is_iso_date && !is_url && !is_boolean_str {
                                                        let mut is_matched = full_item_pug.contains(&extracted_str) || doc_title.contains(&extracted_str);
                                                        
                                                        if !is_matched {
                                                            let digits_only: String = extracted_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                            if digits_only.len() >= 3 {
                                                                let pug_digits: String = full_item_pug.chars().filter(|c| c.is_ascii_digit()).collect();
                                                                if pug_digits.contains(&digits_only) {
                                                                    is_matched = true;
                                                                }
                                                            } else {
                                                                let extracted_lower = extracted_str.to_lowercase();
                                                                let pug_lower = full_item_pug.to_lowercase();
                                                                if pug_lower.contains(&extracted_lower) {
                                                                    is_matched = true;
                                                                }
                                                            }
                                                        }

                                                        if !is_matched {
                                                            requires_retry = true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                if !found_valid_value {
                                    requires_retry = true;
                                }

                                if requires_retry {
                                    miss_counter += 1;
                                    if miss_counter > 3 {
                                        emit_term(&format!("    ⏭️ Skipping field {} due to persistent hallucination or empty value.", field_name_clone));
                                        break; 
                                    }
                                    emit_term(&format!("    ⚠️ Hallucination or empty value detected for field {}. Retrying... ({}/3)", field_name_clone, miss_counter));
                                    for ex_str in extracted_values_for_retry {
                                        ignore_list.push(ex_str.clone());
                                        ignore_list.push(format!(" {}", ex_str));
                                        ignore_list.push(ex_str.to_lowercase());
                                    }
                                    if !found_valid_value {
                                        for k in &keys {
                                            ignore_list.push(format!("\"{}\": \"\"", k));
                                            ignore_list.push(format!("\"{}\":\"\"", k));
                                        }
                                    }
                                    continue;
                                }

                                let mut extracted_results = Vec::new();
                                for k in &keys {
                                    if let Some(val) = parsed_val.get(*k) {
                                        item_val.as_object_mut().unwrap().insert(k.to_string(), val.clone());
                                        extracted_results.push(format!("\"{}\": {}", k, val));
                                        
                                        let val_str = if val.is_string() { val.as_str().unwrap().trim().to_string() }
                                                      else if val.is_number() { val.to_string() }
                                                      else { String::new() };
                                        
                                        if val_str.len() >= 5 && val_str != "null" && val_str != "true" && val_str != "false" {
                                            if !global_ignore_list.contains(&val_str) {
                                                global_ignore_list.push(val_str.clone());
                                                global_ignore_list.push(format!(" {}", val_str));
                                                global_ignore_list.push(val_str.to_lowercase());
                                            }
                                        }
                                    }
                                }
                                
                                for ck in ["has_header", "has_footer", "title", "language"] {
                                    if let Some(val) = parsed_val.get(ck) {
                                        item_val.as_object_mut().unwrap().insert(ck.to_string(), val.clone());
                                    }
                                }

                                if !extracted_results.is_empty() {
                                    emit_term(&format!("    ✅ Extracted: {}", extracted_results.join(", ")));
                                } else {
                                    emit_term(&format!("    ✅ Extracted: (null or empty for {})", field_name_clone));
                                }
                                break;
                            },
                            Err(e) => {
                                println!("[Scheduler] Error extracting list item field {}: {:?}", field_name_clone, e);
                                break;
                            }
                        }
                    } // loop end
                } // fields for loop end

                // 포스트 프로세싱: id, link 등 병합
                if let Some(id_val) = item_val.get("id").and_then(|v| v.as_str()) {
                    let extracted = if let Some(idx) = id_val.rfind('=') {
                        &id_val[idx + 1..]
                    } else {
                        id_val
                    };
                    
                    let clean_str = extracted.replace("-", "").replace("_", "").replace(".", "").replace(",", "");
                    if !clean_str.is_empty() {
                        item_val.as_object_mut().unwrap().insert("id".to_string(), json!(clean_str.trim()));
                    }
                }

                if !item_val.is_null() && (item_val.is_object() || item_val.is_array()) {
                    if let Some(link_val) = item_val.get_mut("link") {
                        if let Some(relative_path) = link_val.as_str() {
                            if let Ok(base_url) = url::Url::parse(&url) {
                                if let Ok(absolute_url) = base_url.join(relative_path) {
                                    let path_query = format!("{}{}", absolute_url.path(), absolute_url.query().map(|q| format!("?{}", q)).unwrap_or_default());
                                    *link_val = json!(path_query.to_lowercase());
                                }
                            }
                        }
                    }
                    
                    emit_term(&format!("  ✅ Successfully Merged Extracted Item {}/{}: {}", idx + 1, total_items, serde_json::to_string(&item_val).unwrap_or_default()));
                    all_extracted_items.push(item_val);
                }
                
                crate::models::qwen::generate::wait_for_global_io().await;

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
            } // items for loop end
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
            // 모델 스위칭 딜레이 없이, 가장 가벼운 모델인 Qwen3 하나만으로 필드별 순차 추출(Loop)을 진행합니다!
            model.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancellation_token.clone()), false, Some("inference".to_string())).await?;

            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

            // 🌟 [NOISE FILTER] bias.json의 prejudice 값을 가져와 노이즈 필터링용 벡터를 생성합니다.
            let (_, layout_prejudice) = crate::parsing::get_layout_bias(&page_type, &doc_lang);
            let layout_prej_emb = model.get_embedding(layout_prejudice.clone()).await.unwrap_or(vec![0.0; 384]);

            // (이전 답변에서 생성된 parsing.rs의 get_detail_schema_fields를 호출합니다)
            let fields = parsing::get_detail_schema_fields(&page_type, &url, &doc_lang);
            let total_fields = fields.len();

            let payload = json!({ "task_id": task.id, "category": "AI Inference", "summary": format!("Extracting {} detail fields sequentially...", total_fields), "spinner": "⠋" });
            let _ = app_handle.emit("extraction-progress", &payload);
            emit_term(&format!("[STAGE-3] Extracting {} detailed fields individually...", total_fields));

            // 1. PUG 각 개별줄 내용을 임베딩 모델로 384차원 인메모리 벡터 저장 및 노이즈 필터링 적용
            let mut pug_lines: Vec<String> = content_pug.lines().map(|s| s.to_string()).collect();
            // 🌟 [CRITICAL FIX] 인덱스 정렬(Alignment)을 위해 미리 크기를 할당합니다.
            let mut line_embeddings = vec![vec![0.0; 384]; pug_lines.len()];
            for line_idx in 0..pug_lines.len() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                let line = &pug_lines[line_idx];
                if line.trim().is_empty() { continue; }

                // 🌟 진행 상황을 터미널에 낱낱이 출력하여 멈춘 것처럼 보이지 않게 합니다.
                emit_term(&format!("  [VECTORIZING] Stage-3 Line {}/{} : {}", line_idx + 1, pug_lines.len(), line.trim()));

                let emb = match model.get_embedding(line.to_string()).await {
                    Ok(vector) => vector,
                    Err(_) => vec![0.0; 384]
                };
                line_embeddings[line_idx] = emb;
            }

            // 🎯 Track A: Stage 3 Detail (Boa Engine 기반 부모 뭉치 단위 노이즈 삭제)
            let (list_bias, form_bias, _) = crate::parsing::get_separated_layout_bias(&page_type, &doc_lang);
            let list_bias_emb = model.get_embedding(list_bias.clone()).await.unwrap_or(vec![0.0; 384]);
            let form_bias_emb = model.get_embedding(form_bias.clone()).await.unwrap_or(vec![0.0; 384]);
            
            let mut wiped_indices = vec![false; pug_lines.len()];
            let mut processed_blocks = std::collections::HashSet::new();

            // Boa 엔진용 HTML 노드 JSON 생성 (Stage 3 컨텍스트 용)
            let nodes_str_detail = {
                let document_for_boa = scraper::Html::parse_document(&clean_html_content);
                let mut nodes_json = Vec::new();
                let mut node_to_idx = std::collections::HashMap::new();
                for (idx, node) in document_for_boa.tree.root().descendants().enumerate() {
                    node_to_idx.insert(node.id(), idx);
                }
                for (idx, node) in document_for_boa.tree.root().descendants().enumerate() {
                    if let Some(el) = node.value().as_element() {
                        let parent_idx = node.parent().and_then(|p| node_to_idx.get(&p.id())).map(|&i| i as i32).unwrap_or(-1);
                        let text: String = node.children()
                            .filter_map(|child| child.value().as_text().map(|t| t.to_string()))
                            .collect::<Vec<_>>().join(" ").trim().to_string();
                        nodes_json.push(serde_json::json!({
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
                        nodes_json.push(serde_json::json!(serde_json::Value::Null));
                    }
                }
                serde_json::to_string(&nodes_json).unwrap_or_default()
            };
            
            let js_template_detail = get_boa_block_extractor_template(); // 🌟 Batch 처리용 템플릿 사용

            let mut track_a_candidates = Vec::new();
            let mut track_a_indices = Vec::new();

            for line_idx in 0..pug_lines.len() {
                if wiped_indices[line_idx] { continue; }
                let line = &pug_lines[line_idx];
                if line.trim().is_empty() { continue; }
                
                let line_prej_score = cosine_similarity(&layout_prej_emb, &line_embeddings[line_idx]);
                
                if line_prej_score > 0.55 {
                    let text_part = if let Some(idx) = line.find('|') { line[idx + 1..].trim() } else { line.trim() };
                    if !text_part.is_empty() {
                        track_a_candidates.push(text_part.to_string());
                        track_a_indices.push(line_idx);
                    }
                }
            }

            // 🌟 Boa 엔진 1번으로 Stage 3 노이즈 후보군 일괄 처리
            let track_a_selectors: Vec<String> = {
                let target_len = track_a_candidates.len(); // 🌟 [CRITICAL FIX] 클로저 이동 전 길이 미리 추출
                let target_titles_str = serde_json::to_string(&track_a_candidates).unwrap_or_else(|_| "[]".to_string());
                let js_code = js_template_detail
                    .replace("NODES_PLACEHOLDER", &nodes_str_detail)
                    .replace("TARGET_TITLES_PLACEHOLDER", &target_titles_str);

                tokio::task::spawn_blocking(move || {
                    let mut context = boa_engine::Context::default();
                    if let Ok(val) = context.eval(boa_engine::Source::from_bytes(js_code.as_bytes())) {
                        if let Some(res_str) = val.as_string().map(|s| s.to_std_string_escaped()) {
                            if let Ok(arr) = serde_json::from_str::<Vec<String>>(&res_str) {
                                return arr;
                            }
                        }
                    }
                    vec![String::new(); target_len]
                }).await.unwrap_or_else(|_| vec![String::new(); target_len])
            };

            for (i, sel) in track_a_selectors.into_iter().enumerate() {
                if sel.is_empty() { continue; }
                // 🌟 Stage 3에서는 반드시 PugMode::DetailMode를 사용합니다.
                let block_pug = crate::parsing::convert_to_clean_pug_selector(&clean_html_content, &sel, crate::parsing::PugMode::DetailMode, None);
                
                if block_pug.is_empty() || processed_blocks.contains(&block_pug) { continue; }
                processed_blocks.insert(block_pug.clone());

                let block_emb = model.get_embedding(block_pug.clone()).await.unwrap_or(vec![0.0; 384]);
                
                let block_prej_score = cosine_similarity(&layout_prej_emb, &block_emb);
                let block_list_score = cosine_similarity(&list_bias_emb, &block_emb);
                let block_form_score = cosine_similarity(&form_bias_emb, &block_emb);
                
                if block_prej_score > block_list_score && block_prej_score > block_form_score {
                    if let Some((start_idx, end_idx)) = find_block_indices_in_pug(&pug_lines, &block_pug) {
                        emit_term(&format!("  🚫 [NOISE BLOCK DELETED] Boa Matched. Lines {}~{} (Prej: {:.4} > List: {:.4} & Form: {:.4})", start_idx + 1, end_idx + 1, block_prej_score, block_list_score, block_form_score));
                        for j in start_idx..=end_idx {
                            pug_lines[j] = String::new(); // 인덱스 보존을 위해 줄 내용만 삭제
                            wiped_indices[j] = true;
                        }
                    }
                }
            }

            for line_idx in 0..pug_lines.len() {
                if !wiped_indices[line_idx] && !pug_lines[line_idx].trim().is_empty() {
                    emit_term(&format!("  [FILTERED PUG] Line {} : {}", line_idx + 1, pug_lines[line_idx].trim()));
                }
            }
            
            let pug_lines_ref: Vec<&str> = pug_lines.iter().map(|s| s.as_str()).collect();

            // 문서 타이틀 추출 (환각 검증용)
            let doc_title = {
                let doc = scraper::Html::parse_document(&clean_html_content);
                if let Ok(sel) = scraper::Selector::parse("title") {
                    doc.select(&sel).next().map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string()).unwrap_or_default()
                } else {
                    String::new()
                }
            };

            let mut global_ignore_list: Vec<String> = Vec::new(); // 🌟 전역 무시 리스트 추가

            // 필드 단위로 하나씩 쪼개어 순차 추출 (병렬 처리 시의 VRAM 초과/컨텍스트 환각 방지)
            // 🌟 [CRITICAL FIX] prejudice_target 매개변수 추가
            for (idx, (field_name, field_desc, bias_target, prejudice_target)) in fields.into_iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                
                // 2. get_detail_schema_fields의 bias_target 값을 임베딩하여 인메모리 코사인 유사도 검색
                let query_emb = model.get_embedding(bias_target.clone()).await.unwrap_or(vec![0.0; 384]);
                let mut best_idx = 0;
                let mut best_score = -1.0;
                
                for (i, emb) in line_embeddings.iter().enumerate() {
                    if pug_lines_ref[i].trim().is_empty() { continue; }
                    let score = cosine_similarity(&query_emb, emb);
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                    }
                }
                
                // 3. 찾은 컨텍스트 블록으로 추론을 위한 시스템 메시지 동적 조립
                let targeted_pug = extract_pug_context(&pug_lines_ref, best_idx);
                
                emit_term(&format!("  🎯 [MATCHED CONTEXT] Field: '{}' | Score: {:.4}\n{}", field_name, best_score, targeted_pug));
                
                let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: format!("[PUG CONTENT]\n{}", targeted_pug),
                    name: None,
                });

                let percent = (((idx as f32) / (total_fields as f32)) * 100.0) as i32;
                let summary_msg = format!("Extracting {} ({}%)...", field_name, percent);
                
                let payload = json!({ 
                    "task_id": task.id, 
                    "category": format!("Detail Extraction ({}/{})", idx + 1, total_fields), 
                    "summary": summary_msg, 
                    "spinner": "⠋" 
                });
                log_task_progress(app_handle, &task.id, &payload);
                emit_term(&format!("[STAGE-3] {}", summary_msg));

                // 🌟 [CRITICAL FIX] 새롭게 추가된 title 파라미터 규격에 맞춰 &doc_title을 주입합니다.
                let task_question = parsing::extract_single_field_prompt(&page_type, &field_name, &field_desc, language, &doc_title);
                
                // 🌟 [BIAS SKIP LOGIC] 본문에 존재하지 않는 잘못된 추출값 기록용 리스트 및 카운터
                let mut ignore_list: Vec<String> = global_ignore_list.clone(); // 🌟 매 필드마다 전역 리스트를 복사하여 누적 시작
                let mut miss_counter = 0;
                
                loop {
                    if cancellation_token.load(Ordering::Relaxed) { break; }

                    let q3_gen = model.qwen3_generator.clone();
                    let cancel_clone = cancellation_token.clone();
                    let sys_msg = system_message.clone();
                    
                    let field_name_clone = field_name.clone();
                    let bias_target_for_closure = bias_target.clone(); // 🌟 다국어 Bias 할당
                    
                    // 🌟 [CRITICAL FIX] 다국어 Prejudice(배제) 타겟 클로저용 변수 생성
                    let prejudice_target_for_closure = prejudice_target.clone(); 
                    
                    let task_q = task_question.clone();
                    let ignore_list_clone = ignore_list.clone();
                    
                    let res = tokio::task::spawn_blocking(move || {
                        let mut gen_guard = q3_gen.blocking_lock();
                        if let Some(gen) = gen_guard.as_mut() {
                            let params = ChatCompletionParameters {
                                messages: vec![
                                    sys_msg,
                                    ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { 
                                        content: ChatCompletionRequestUserMessageContent::Text(task_q),
                                        name: None,
                                    })
                                ],
                                model: "qwen3".to_string(), max_tokens: Some(512), temperature: Some(0.0), top_p: Some(0.95),
                                ..Default::default()
                            };
                            
                            // 🌟 [CRITICAL FIX] Prejudice(오답 밀어내기) 파라미터를 Qwen3 모델에 전달합니다!
                            let p_target = if prejudice_target_for_closure.is_empty() { None } else { Some(prejudice_target_for_closure.as_str()) };
                            // let b_target = if bias_target_for_closure.is_empty() { None } else { Some(bias_target_for_closure.as_str()) };
                            
                            gen.generate(params, Some(cancel_clone), Some(&ignore_list_clone), p_target).map_err(|e| anyhow::anyhow!("Qwen 3 field extraction failed: {}", e))
                        } else {
                            Err(anyhow::anyhow!("Qwen 3 Generator not available"))
                        }
                    }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("Task join failed: {}", e)));

                    // 🌟 [개선 2] 한 번의 생성이 끝날 때마다 (성공/실패 무관하게) 즉시 KV 캐시를 초기화하여, 
                    // 다음 재시도 시 모델이 과거의 환각 데이터를 바탕으로 헛소리를 이어가는 것을 원천 차단합니다!
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

                    match res {
                        Ok(res_text) => {
                            let mut parsed = parsing::parse_json_from_llm(&res_text);
                            
                            // type 래퍼 제거 로직
                            let mut item_val = if let Some(inner) = parsed.get_mut(&page_type) { inner.take() } else { parsed };

                            // 🌟 [BIAS SKIP LOGIC] 추출된 값이 실제로 PUG 본문이나 제목에 존재하는지 검증
                            let mut requires_retry = false;
                            let mut extracted_values_for_retry = Vec::new();
                            
                            let keys: Vec<&str> = field_name_clone.split(',').map(|s| s.trim()).collect();
                            let mut found_valid_value = false;

                            // 🎯 [핵심 개선] ENUM 속성이나 영단어로 번역되어 출력될 수 있는 필드들은 본문 텍스트 완벽 매칭 검사에서 제외합니다.
                            let skip_pug_match_fields = ["status", "payment_method", "payment_origin", "condition", "currency"];
                            let is_enum_field = skip_pug_match_fields.iter().any(|&f| field_name_clone.contains(f));

                            for k in &keys {
                                if let Some(val) = item_val.get(*k) {
                                    let is_empty_val = match val {
                                        serde_json::Value::Null => true,
                                        serde_json::Value::String(s) => s.trim().is_empty() || s == "..." || s == "null",
                                        serde_json::Value::Array(a) => a.is_empty(),
                                        serde_json::Value::Object(o) => o.is_empty(),
                                        _ => false,
                                    };

                                    if !is_empty_val {
                                        found_valid_value = true;

                                        let extracted_str = if val.is_string() {
                                            val.as_str().unwrap_or("").trim().to_string()
                                        } else if val.is_number() {
                                            val.to_string() // 숫자형 데이터도 검증을 위해 문자열로 변환
                                        } else {
                                            String::new()
                                        };

                                        if !extracted_str.is_empty() && extracted_str != "..." && extracted_str != "null" {
                                            extracted_values_for_retry.push(extracted_str.clone());
                                            
                                            if !is_enum_field {
                                                let is_iso_date = extracted_str.contains('T') && extracted_str.len() >= 19;
                                                let is_url = extracted_str.starts_with("http") || extracted_str.starts_with('/');
                                                let is_boolean_str = extracted_str == "true" || extracted_str == "false";
                                                
                                                if !is_iso_date && !is_url && !is_boolean_str {
                                                    // 1차: 완벽한 텍스트 포함 여부 (문자열 일치)
                                                    let mut is_matched = content_pug.contains(&extracted_str) || doc_title.contains(&extracted_str);
                                                    
                                                    // 2차: 전화번호, 가격 등 포맷(하이픈, 콤마, 띄어쓰기) 차이로 인한 불일치 극복을 위해 순수 숫자만 추출하여 검증
                                                    if !is_matched {
                                                        let digits_only: String = extracted_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                        
                                                        // 숫자가 3자리 이상 포함된 데이터일 경우 숫자 연속성이 PUG에 존재하는지 확인
                                                        if digits_only.len() >= 3 {
                                                            let pug_digits: String = content_pug.chars().filter(|c| c.is_ascii_digit()).collect();
                                                            if pug_digits.contains(&digits_only) {
                                                                is_matched = true;
                                                            }
                                                        } else {
                                                            // 숫자가 아니거나 너무 짧은데 완벽 매칭이 안됐다면 대소문자 무시 검색
                                                            let extracted_lower = extracted_str.to_lowercase();
                                                            let pug_lower = content_pug.to_lowercase();
                                                            if pug_lower.contains(&extracted_lower) {
                                                                is_matched = true;
                                                            }
                                                        }
                                                    }

                                                    // 010-0356789 같은 환각 전화번호는 2차 검증(숫자 배열)에서도 걸러져서 확실히 재시도됩니다!
                                                    if !is_matched {
                                                        requires_retry = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // 🌟 [CRITICAL FIX] LLM이 요구한 키를 뱉지 않았거나, 빈 값("")을 뱉었을 경우에도 재시도 (최대 3회)
                            if !found_valid_value {
                                requires_retry = true;
                            }

                            if requires_retry {
                                miss_counter += 1;
                                if miss_counter > 3 {
                                    emit_term(&format!("  ⏭️ Skipping field {} due to persistent hallucination or empty value.", field_name_clone));
                                    break; 
                                }
                                emit_term(&format!("  ⚠️ Hallucination or empty value detected for field {}. Retrying... ({}/3)", field_name_clone, miss_counter));
                                for ex_str in extracted_values_for_retry {
                                    ignore_list.push(ex_str.clone());
                                    ignore_list.push(format!(" {}", ex_str));
                                    ignore_list.push(ex_str.to_lowercase());
                                }
                                // 🎯 빈 값을 뱉는 것을 억제하기 위해 JSON 빈 문자열 패턴을 ignore_list에 추가합니다.
                                if !found_valid_value {
                                    for k in &keys {
                                        ignore_list.push(format!("\"{}\": \"\"", k));
                                        ignore_list.push(format!("\"{}\":\"\"", k));
                                    }
                                }
                                continue;
                            }

                            // 검증 통과 후 저장 및 결과 로그 출력
                            let mut extracted_results = Vec::new();
                            for k in &keys {
                                if let Some(val) = item_val.get(*k) {
                                    extracted_data.as_object_mut().unwrap().insert(k.to_string(), val.clone());
                                    extracted_results.push(format!("\"{}\": {}", k, val));
                                    
                                    // 🌟 [전역 무시 리스트 업데이트] 성공적으로 찾은 값을 다른 필드에서 중복 추출하지 않도록 방어합니다.
                                    let val_str = if val.is_string() { val.as_str().unwrap().trim().to_string() }
                                                  else if val.is_number() { val.to_string() }
                                                  else { String::new() };
                                    
                                    // 이름이나 짧은 단어가 실수로 억제되는 것을 방지하기 위해 길이가 5 이상인 고유값만 추가합니다.
                                    if val_str.len() >= 5 && val_str != "null" && val_str != "true" && val_str != "false" {
                                        if !global_ignore_list.contains(&val_str) {
                                            global_ignore_list.push(val_str.clone());
                                            global_ignore_list.push(format!(" {}", val_str));
                                            global_ignore_list.push(val_str.to_lowercase());
                                        }
                                    }
                                }
                            }
                            
                            // 공통 속성(has_header, title 등)도 데이터에 병합하되, 로그에서는 생략하여 깔끔하게 유지합니다.
                            for ck in ["has_header", "has_footer", "title", "language"] {
                                if let Some(val) = item_val.get(ck) {
                                    extracted_data.as_object_mut().unwrap().insert(ck.to_string(), val.clone());
                                }
                            }

                            if !extracted_results.is_empty() {
                                emit_term(&format!("  ✅ Extracted: {}", extracted_results.join(", ")));
                            } else {
                                emit_term(&format!("  ✅ Extracted: (null or empty for {})", field_name_clone));
                            }
                            break; // 정상 추출 시 무한루프 탈출
                        },
                        Err(e) => {
                            println!("[Scheduler] Error extracting detail field {}: {:?}", field_name_clone, e);
                            break;
                        }
                    }
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
                                    if let Some(dt) = chrono::DateTime::from_timestamp_millis(ts_ms).map(|dt| dt.naive_utc()) {
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
                        if let Some(dt) = chrono::DateTime::from_timestamp_millis(ts_ms).map(|dt| dt.naive_utc()) {
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
                    let tracking_vector = model.get_embedding(tracking_text.clone()).await.unwrap_or(vec![0.0; 384]);
                    
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
                                let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 384]);
                                
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
                                        let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 384]);
                                        
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

pub fn get_boa_js_template() -> &'static str {
    r##"
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
    "##
}

// 🌟 [CRITICAL OPTIMIZATION] 여러 개의 텍스트 타겟을 한 번의 JS 엔진 구동으로 일괄(Batch) 역추적하여 반환하는 템플릿입니다.
pub fn get_boa_block_extractor_template() -> &'static str {
    r##"
        const nodes = NODES_PLACEHOLDER;
        const targetTitles = TARGET_TITLES_PLACEHOLDER;
        
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
            if (nodeA.tagName === 'tr') {
                const aKids = getChildren(nodeA.index).filter(n => n.tagName === 'td' || n.tagName === 'th');
                const bKids = getChildren(nodeB.index).filter(n => n.tagName === 'td' || n.tagName === 'th');
                const aColspan = aKids.reduce((sum, k) => sum + parseInt(k.colspan || '1', 10), 0);
                const bColspan = bKids.reduce((sum, k) => sum + parseInt(k.colspan || '1', 10), 0);
                if (aColspan > 0 && bColspan > 0 && Math.abs(aColspan - bColspan) > 1) { return 0; }
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
            let fallbackParent = null;
            for (let i = 0; i < 15; i++) {
                const node = nodes[cur];
                if (!node) break;
                const pIdx = node.parentIndex;
                if (pIdx === undefined || pIdx === -1) break;

                const parentNode = nodes[pIdx];
                if (parentNode && !fallbackParent) {
                    // 🌟 단독으로 배치된 Form 이나 고유 ID 클래스가 있는 시맨틱 컨테이너를 백업 부모로 동적 추적합니다.
                    if (parentNode.id || (parentNode.classes && parentNode.classes.length > 0) || ["form", "table", "nav", "tbody", "fieldset"].includes(parentNode.tagName)) {
                        fallbackParent = parentNode;
                    }
                }
                
                if (node.tagName === "td" || node.tagName === "th") {
                    if (parseInt(node.colspan || '1', 10) > 1 || parseInt(node.rowspan || '1', 10) > 1) {
                        cur = pIdx; continue;
                    }
                    const pNode = nodes[pIdx];
                    if (pNode && pNode.tagName === "tr") {
                        const gpIdx = pNode.parentIndex; 
                        if (gpIdx !== undefined && gpIdx !== -1) {
                            const trSiblings = getChildren(gpIdx);
                            const similarTrs = trSiblings.filter(s => calculateSimilarity(pNode, s) >= 60);
                            if (similarTrs.length >= 2) { cur = pIdx; continue; }
                        }
                    }
                }
                const siblings = getChildren(pIdx);
                const similarSiblings = siblings.filter(s => calculateSimilarity(node, s) >= 60);
                if (similarSiblings.length >= 2) {
                    let finalParent = parentNode;
                    let walkIdx = pIdx;
                    for(let j=0; j<5; j++) {
                        let gIdx = nodes[walkIdx] ? nodes[walkIdx].parentIndex : -1;
                        if (gIdx !== -1 && nodes[gIdx]) {
                            const grand = nodes[gIdx];
                            if (grand.id || ["table", "ul", "ol", "nav", "form"].includes(grand.tagName)) {
                                finalParent = grand;
                                if (grand.id || grand.tagName === "table" || grand.tagName === "form") break;
                            }
                            walkIdx = gIdx;
                        }
                    }
                    return getSignature(finalParent, true);
                }
                cur = pIdx;
            }
            // 🌟 형제 노드 반복 패턴이 없는 독자적인 레이아웃일 경우 추적 수집된 시맨틱 백업 부모 블록을 안전하게 반환합니다.
            if (fallbackParent) { return getSignature(fallbackParent, true); }
            return null;
        }

        let finalResults = [];
        for (let k = 0; k < targetTitles.length; k++) {
            let t = targetTitles[k].toLowerCase().replace(/\s+/g, ' ');
            let potentialMatches = [];
            if (t.includes('\uFFFD')) {
                let chunks = t.split(/[\uFFFD]+/).map(c => c.trim()).filter(c => c.length > 1);
                if (chunks.length > 0) {
                    potentialMatches = nodes.filter(n => {
                        if (!n || !n.text) return false;
                        let nText = n.text.toLowerCase().replace(/\s+/g, ' ');
                        return chunks.every(chunk => nText.includes(chunk));
                    });
                }
            } else {
                potentialMatches = nodes.filter(n => {
                    if (!n || !n.text) return false;
                    let nText = n.text.toLowerCase().replace(/\s+/g, ' ');
                    return nText.includes(t);
                });
            }
            
            let parentSel = "";
            if (potentialMatches.length > 0) {
                potentialMatches.sort((a, b) => a.text.length - b.text.length);
                const d = detect(potentialMatches[0].index);
                if (d && d !== "body") { parentSel = d; }
            }
            finalResults.push(parentSel);
        }
        JSON.stringify(finalResults);
    "##
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
            vec![0.0; 384],
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


// [IN-MEMORY VECTOR SEARCH] 코사인 유사도 계산 및 PUG 부모 컨텍스트 추출
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot_product / (norm_a * norm_b) }
}

fn extract_pug_context(lines: &[&str], target_idx: usize) -> String {
    if lines.is_empty() { return String::new(); }
    let mut parent_idx = target_idx;
    let target_indent = lines[target_idx].chars().take_while(|c| c.is_whitespace()).count();
    
    // 1. 위로 거슬러 올라가며 들여쓰기가 더 적은(부모) 노드를 찾습니다.
    for i in (0..target_idx).rev() {
        let indent = lines[i].chars().take_while(|c| c.is_whitespace()).count();
        if indent < target_indent && !lines[i].trim().is_empty() {
            parent_idx = i;
            break;
        }
    }

    let parent_indent = lines[parent_idx].chars().take_while(|c| c.is_whitespace()).count();
    let mut context_lines = vec![lines[parent_idx]];
    
    // 2. 부모 노드의 하위(자식) 노드들을 모두 긁어옵니다.
    for i in (parent_idx + 1)..lines.len() {
        if lines[i].trim().is_empty() { continue; }
        let indent = lines[i].chars().take_while(|c| c.is_whitespace()).count();
        // 다시 부모와 같거나 밖으로 나가는 들여쓰기를 만나면 블록 종료
        if indent <= parent_indent {
            break;
        }
        context_lines.push(lines[i]);
    }
    
    context_lines.join("\n")
}
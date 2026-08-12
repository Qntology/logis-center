use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use crate::store::{VectorStore, Task};
use crate::logic;
use crate::utils;
use crate::utils::parsing::{self, PugMode};
use crate::model::LogisModel;
use serde_json::{Value, json};
use anyhow::Result;
use tauri::Emitter;
use std::sync::atomic::{AtomicBool, Ordering};


use once_cell::sync::OnceCell;

use crate::utils::pug_utils::*;
use crate::js_templates::*;
use crate::utils::json_utils::merge_node;
use crate::utils::ai_utils::{cosine_similarity, extract_pug_context as other_extract_pug_context, max_pool_sim, split_bias_phrases, split_bias_phrases_weighted, weighted_max_pool_sim, exclusive_assign, exclusive_assign_by_score, self_poisoned_prejudice_mask, collect_select_groups, enum_status_keys, status_key_phrases, double_center_matrix, detect_field_format, value_matches_format, is_pure_numeric_value, value_token_in_url_pool, is_id_link_field, resolve_id_link_from_lines, extract_url_pattern, apply_url_pattern, find_identifier_token_in_lines, label_phrase_bank, prejudice_phrase_bank, collect_id_link_candidates, collect_id_link_candidates_from_url, collect_labeled_token_candidates, collect_detail_label_value_pairs, DetailPair, pug_line_parts, is_non_value_role_tag, pug_attr_flag, strip_markup_prefix, extract_date_literal, id_shape_signature, id_shape_allowed, same_host, line_real_href, is_multi_value_field, FieldFormat};
use crate::utils::logger::log_task_progress;

pub static PROGRESS_TX: OnceCell<tokio::sync::mpsc::UnboundedSender<serde_json::Value>> = OnceCell::new();
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
   
    tokio::spawn(async move {
        if !crate::utils::sync_utils::UI_READY_FLAG.load(std::sync::atomic::Ordering::SeqCst) {
            crate::utils::sync_utils::UI_READY_SIGNAL.notified().await;
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
                    _ = crate::utils::sync_utils::TASK_QUEUED_SIGNAL.notified() => {
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
                        oom_retry_map.remove(&task.id);
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

    let kv_name = if task.r#type == "image_extraction" {
        Some("image".to_string())
    } else {
        Some("text".to_string())
    };
 
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
    let mut doc_lang = "en".to_string();

    let model = {
        println!("[Scheduler] 🛡️ Attempting to acquire Model Lock...");
        let mut model_lock = model_mutex.lock().await;
        println!("[Scheduler] ✅ Model Lock acquired.");
        
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }


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

    if task.r#type != "image_extraction" && task.r#type != "analytic_extraction" {
        model.check_embedding_downloaded().await?;
    }

    if task.r#type == "image_extraction" {
        let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("").to_string();
        

        if !image_path.is_empty() {
            println!("[Scheduler] Starting Image Extraction for {}", task.id);

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

    let (mut url, mut origin_candidate) = crate::utils::url_utils::resolve_absolute_url(&task_data).await;

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

    let raw_html_content = if task.r#type == "document_extraction" {
        let file_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("");
        let ext = task_data.get("document_ext").and_then(|s| s.as_str()).unwrap_or("");
        
        let payload = json!({ 
            "task_id": task.id, 
            "category": "Document Parsing", 
            "summary": format!("Parsing {} file format...", ext.to_uppercase()), 
            "spinner": "📄" 
        });

        let _ = app_handle.emit("extraction-progress", &payload);

        let extracted_text = crate::parsers::extract_document_text(file_path).unwrap_or_else(|e| format!("Document Parsing Error: {}", e));

        let fake_html = extracted_text.lines()
            .map(|line| {
                let safe_line = line.replace("<", "&lt;").replace(">", "&gt;");
                format!("<div>{}</div>", safe_line)
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("<html><body>{}</body></html>", fake_html)
    } else if let Some(raw_html) = task_data.get("html").and_then(|s| s.as_str()) {
        let content = raw_html.to_string();
        if let Some(obj) = task_data.as_object_mut() { obj.remove("html"); }
        content
    } else if !url.is_empty() {
        let response = reqwest::get(&url).await?;
        let bytes = response.bytes().await?;

        let (decoded_utf8, _, malformed_utf8) = encoding_rs::UTF_8.decode(&bytes);
        let utf8_str = decoded_utf8.as_ref();

        let needs_euc = utf8_str.to_lowercase().contains("charset=euc-kr") || 
                        utf8_str.to_lowercase().contains("charset=\"euc-kr\"") ||
                        utf8_str.to_lowercase().contains("charset=cp949") ||
                        utf8_str.to_lowercase().contains("charset=ks_c_5601");

        if needs_euc && malformed_utf8 {

            let (decoded_euc, _, _) = encoding_rs::EUC_KR.decode(&bytes);
            decoded_euc.into_owned()
        } else {

            utf8_str.to_string()
        }
    } else {
        return Ok(());
    };

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let clean_html_content = parsing::pre_clean_html(&raw_html_content);
    
    let mut raw_pug = parsing::convert_to_clean_pug(&clean_html_content, PugMode::NoAttributesMode, Some(&url));
    let mut light_pug = model.truncate_pug_context(&raw_pug, false, 2000, None).await;

    let base_path = std::fs::canonicalize("src-tauri/models").or_else(|_| std::fs::canonicalize("models")).unwrap_or_default();
    let tokenizer_path = base_path.join("Qwen3-0.6B-Instruct-gguf").to_string_lossy().to_string();
 
    let raw_system_prefix = format!("<|im_start|>system\n{}<|im_end|>\n", light_pug);

    let mut token_count = raw_system_prefix.len() / 4;

    if let Ok(tokenizer) = crate::tokenizer::TokenizerModel::init(&tokenizer_path) {

        token_count = tokenizer.text_encode_vec(raw_system_prefix.clone(), false)
            .map(|v| v.len())
            .unwrap_or(token_count);
    }

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
            let link_val = (url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str()).to_lowercase();
            let path_only = url_obj.path().to_lowercase(); 
            
            let mut potential_caches = Vec::new();

            if let Ok(Some(page_doc)) = db.get_item_by_id("pages", &page_id).await {
                potential_caches.push(page_doc);
            } else if let Ok(Some(page_doc)) = db.get_item_by_id("items", &page_id).await {
                potential_caches.push(page_doc);
            }

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

                            final_cache = Some((page_doc, val, false, target_sel_clean));
                            break;
                        }

                    } else {
                        if final_cache.is_none() {
                            final_cache = Some((page_doc, val, true, target_sel_clean));
                        }
                    }
                }
            }


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

    let base_session_id = format!("{}_base", task.id);
    let system_content = format!("[PUG CONTENT]\n{}", light_pug);

    if !skip_ai_analysis {

        if base_model_size == crate::model::ModelSize::Qwen {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            
            let base_kv_path = utils::paths::get_kv_dir(Some(app_handle)).join(&base_session_id);
            if !base_kv_path.exists() {
                println!("[Scheduler] Baking Base PUG Context to SSD...");
                log_task_progress(app_handle, &task.id, &json!({ "category": "Preparation", "summary": "Reading document structure...", "spinner": "⠋" }));
                
                model.secure_vram_relay(crate::model::ModelSize::Qwen, None, Some(cancellation_token.clone()), true, kv_name.clone()).await?;
                
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                if let Some(gen) = model.generator.lock().await.as_mut() {
                    let raw_system_prefix = format!("<|im_start|>system\n{}<|im_end|>\n", system_content);

                    gen.prefill_only(raw_system_prefix, Some(cancellation_token.clone()), Some(base_session_id.clone()), None, kv_name.clone()).await?;
                }
            }

            model.deep_purge_resources().await;
        }

        let pug_lines: Vec<String> = light_pug.lines().map(|s| s.to_string()).collect();
        let mut line_embeddings = vec![vec![0.0; 384]; pug_lines.len()];
        let mut wiped_indices = vec![false; pug_lines.len()];

        let early_doc_title = {
            let doc = scraper::Html::parse_document(&clean_html_content);
            let mut t_val = if let Ok(sel) = scraper::Selector::parse("title") {
                doc.select(&sel).next().map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string()).unwrap_or_default()
            } else {
                String::new()
            };
            let mut heading_texts = Vec::new();
            if let Ok(sel_h1) = scraper::Selector::parse("h1") {
                for el in doc.select(&sel_h1) {
                    let txt = el.text().collect::<Vec<_>>().join(" ").trim().to_string();
                    if !txt.is_empty() { heading_texts.push(txt); }
                }
            }
            if let Ok(sel_h2) = scraper::Selector::parse("h2") {
                for el in doc.select(&sel_h2) {
                    let txt = el.text().collect::<Vec<_>>().join(" ").trim().to_string();
                    if !txt.is_empty() { heading_texts.push(txt); }
                }
            }

            if !heading_texts.is_empty() {
                if t_val.is_empty() || t_val.len() < 5 {
                    t_val = heading_texts.join(" | ");
                } else {
                    t_val = format!("{} | {}", t_val, heading_texts.join(" | "));
                }
            }
            t_val
        };
        let early_title_emb = if !early_doc_title.is_empty() {
            model.get_embedding(early_doc_title.clone()).await.unwrap_or(vec![0.0; 384])
        } else {
            vec![0.0; 384]
        };
        

        let mut filtered_light_pug = light_pug.clone();
        let mut line_col_positions: std::collections::HashMap<usize, (usize, usize)> = std::collections::HashMap::new();
        let mut is_table_structure = false;
        {
            let mut current_row: usize = 0;
            let mut current_col: usize = 0;
            let mut in_row = false;
            for (line_idx, line) in pug_lines.iter().enumerate() {
                let trimmed = line.trim();
                if trimmed.is_empty() { continue; }
                let tag_part = trimmed.split('|').next().unwrap_or("").trim();
                let tag_name = tag_part.split(|c: char| c == '[' || c == ' ' || c == '(').next().unwrap_or("").to_lowercase();
                if tag_name == "tr" {
                    is_table_structure = true;
                    if in_row { current_row += 1; }
                    current_col = 0;
                    in_row = true;
                } else if (tag_name == "td" || tag_name == "th") && in_row {

                    let mut colspan_val = 1;
                    if let Ok(re_cs) = regex::Regex::new(r#"colspan[=\\"]*(\d+)"#) {
                        if let Some(cap) = re_cs.captures(tag_part) {
                            colspan_val = cap[1].parse::<usize>().unwrap_or(1);
                        }
                    }

                    if let Some(pipe_idx) = trimmed.find('|') {
                        let txt = trimmed[pipe_idx + 1..].trim();
                        if !txt.is_empty() {
                            line_col_positions.insert(line_idx, (current_row, current_col));
                        }
                    }
                    current_col += colspan_val;
                } else if tag_name != "td" && tag_name != "th" && tag_name != "tr" && tag_name != "thead" && tag_name != "tbody" && tag_name != "table" {

                    if in_row && !["colgroup", "col", "caption"].contains(&tag_name.as_str()) {

                    }
                }
            }
        }

        let mut global_text_stats: std::collections::HashMap<String, (usize, Vec<(usize, usize, Option<(usize, usize)>)>)> = std::collections::HashMap::new();
        for (line_idx, line) in pug_lines.iter().enumerate() {
            if let Some(idx) = line.find('|') {
                let indent = line.chars().take_while(|c| c.is_whitespace()).count();
                let text_part = line[idx + 1..].trim();
                if !text_part.is_empty() && text_part.len() > 2 {
                    let col_pos = line_col_positions.get(&line_idx).cloned();
                    let entry = global_text_stats.entry(text_part.to_string()).or_insert((0, Vec::new()));
                    entry.0 += 1;
                    entry.1.push((line_idx, indent, col_pos));
                }
            }
        }

        let total_table_rows = if is_table_structure {
            let mut max_row = 0usize;
            for (_, &(r, _)) in &line_col_positions {
                if r > max_row { max_row = r; }
            }
            max_row + 1
        } else { 0 };

        let universal_prejudice = "global navigation, menus, footer, aside, search form, search filter.";
        let universal_prej_emb = model.get_embedding(universal_prejudice.to_string()).await.unwrap_or(vec![0.0; 384]);
        
        let mut global_boilerplate_texts = std::collections::HashSet::new();
        let re_numeric = regex::Regex::new(r"^\D*\d+[\d,\.]*\D*$").unwrap();
        let re_has_digit = regex::Regex::new(r"\d").unwrap();
        
        for (text, (count, occurrences)) in global_text_stats {
            if count >= 4 {

                let is_numeric_data = re_numeric.is_match(&text) || re_has_digit.is_match(&text);
                if is_numeric_data { continue; }

                if is_table_structure && total_table_rows >= 3 {
                    let mut col_hits: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                    let mut rows_with_this_text: std::collections::HashSet<usize> = std::collections::HashSet::new();
                    for (_, _, col_pos) in &occurrences {
                        if let Some((row_idx, col_idx)) = *col_pos {
                            *col_hits.entry(col_idx).or_insert(0) += 1;
                            rows_with_this_text.insert(row_idx);
                        }
                    }

                    let row_coverage = rows_with_this_text.len() as f64 / total_table_rows as f64;
                    let max_col_hit = col_hits.values().max().copied().unwrap_or(0);
                    let is_same_col_repeated = max_col_hit >= (total_table_rows as f64 * 0.7).ceil() as usize;
                    
                    if is_same_col_repeated && row_coverage >= 0.7 {

                        let has_link_or_event = occurrences.iter().any(|(line_idx, _, _)| {
                            let line = &pug_lines[*line_idx];
                            line.contains("href=") || line.contains("onclick") || line.contains("onsubmit") || line.contains("onchange") || line.contains("data-url")
                        });
                        if has_link_or_event {
                            emit_term(&format!("  🛡️ [TABLE-COL LINK PROTECT] 동일 컬럼 반복이지만 href/event 속성 포함 링크 데이터 보호: '{}' ({}회 발견)", text, count));
                            continue;
                        }

                        if text.len() < 10 {
                            global_boilerplate_texts.insert(text.clone());
                            emit_term(&format!("  🚫 [TABLE-COL DROP] 동일 컬럼({}회/{}rows) 반복 UI 탈락: '{}' ({}회 발견)", max_col_hit, total_table_rows, text, count));
                            continue;
                        } else {
                            emit_term(&format!("  🛡️ [TABLE-COL PROTECT] 동일 컬럼 반복이지만 긴 텍스트(데이터 추정): '{}' ({}회 발견)", text, count));
                            continue;
                        }
                    }
                }

                let mut is_contiguous = false;
                let mut is_dispersed = false;
                let mut is_same_depth = true;

                if occurrences.len() >= 2 {
                    let mut gaps = Vec::new();
                    let first_indent = occurrences[0].1;
                    for i in 1..occurrences.len() {
                        gaps.push(occurrences[i].0 - occurrences[i-1].0);
                        if occurrences[i].1 != first_indent {
                            is_same_depth = false;
                        }
                    }
                    let min_gap = *gaps.iter().min().unwrap_or(&0);
                    let max_gap = *gaps.iter().max().unwrap_or(&0);
                    if min_gap <= 3 && max_gap <= 5 {
                        is_contiguous = true;
                    } else if max_gap > 10 {
                        is_dispersed = true;
                    }

                    if is_same_depth && max_gap > 5 {

                        if is_table_structure {
                            let mut unique_cols: std::collections::HashSet<usize> = std::collections::HashSet::new();
                            for (_, _, col_pos) in &occurrences {
                                if let Some((_, col_idx)) = *col_pos {
                                    unique_cols.insert(col_idx);
                                }
                            }
                            if unique_cols.len() <= 1 && count >= 6 {

                                let has_link_or_event = occurrences.iter().any(|(line_idx, _, _)| {
                                    let line = &pug_lines[*line_idx];
                                    line.contains("href=") || line.contains("onclick") || line.contains("onsubmit") || line.contains("onchange") || line.contains("data-url")
                                });
                                if has_link_or_event {
                                    emit_term(&format!("  🛡️ [TABLE-SAME-COL LINK PROTECT] 단일 컬럼 반복이지만 href/event 속성 포함 링크 데이터 보호: '{}' ({}회, 컬럼 {:?})", text, count, unique_cols));
                                    continue;
                                }

                                global_boilerplate_texts.insert(text.clone());
                                emit_term(&format!("  🚫 [TABLE-SAME-COL DROP] 단일 컬럼 반복 탈락: '{}' ({}회, 컬럼 {:?})", text, count, unique_cols));
                                continue;
                            }
                        }                        emit_term(&format!("  🛡️ [GLOBAL PROTECT] 동일 구조(Depth) 내 분산 패턴 데이터 보호: '{}' ({}회 발견)", text, count));
                        continue;
                    } else if !is_same_depth && is_dispersed {
                        if text.len() < 20 {

                            let has_link_or_event = occurrences.iter().any(|(line_idx, _, _)| {
                                let line = &pug_lines[*line_idx];
                                line.contains("href=") || line.contains("onclick") || line.contains("onsubmit") || line.contains("onchange") || line.contains("data-url")
                            });
                            if has_link_or_event {
                                emit_term(&format!("  🛡️ [GLOBAL LINK PROTECT] 다중 구조 교차지만 href/event 속성 포함 링크 데이터 보호: '{}' ({}회 발견)", text, count));
                                continue;
                            }

                            let drop_text_emb = model.get_embedding(text.clone()).await.unwrap_or(vec![0.0f32; 384]);
                            let title_protect_sim = cosine_similarity(&early_title_emb, &drop_text_emb);
                            if title_protect_sim > 0.40 {
                                emit_term(&format!("  🛡️ [TITLE VECTOR PROTECT] 다중 구조 교차지만 타이틀 코사인 유사도({:.4})가 높아 도메인 시그널 보호: '{}' ({}회 발견)", title_protect_sim, text, count));
                                continue;
                            }
                            global_boilerplate_texts.insert(text.clone());
                            emit_term(&format!("  🚫 [GLOBAL DROP] 다중 구조(Depth) 교차 발견 노이즈 탈락: '{}' ({}회 발견)", text, count));
                            continue;
                        }
                    }
                }

                if is_contiguous {
                    if text.len() < 20 {
                        let has_link_or_event = occurrences.iter().any(|(line_idx, _, _)| {
                            let line = &pug_lines[*line_idx];
                            line.contains("href=") || line.contains("onclick") || line.contains("onsubmit") || line.contains("onchange") || line.contains("data-url")
                        });
                        if has_link_or_event {
                            emit_term(&format!("  🛡️ [GLOBAL LINK PROTECT] 뭉쳐있지만 href/event 속성 포함 링크 데이터 보호: '{}' ({}회 발견, 연속됨)", text, count));
                        } else {
                            global_boilerplate_texts.insert(text.clone());
                            emit_term(&format!("  🚫 [GLOBAL DROP] 뭉쳐있는 UI 노이즈 탈락: '{}' ({}회 발견, 연속됨)", text, count));
                        }
                    }
                } else if is_dispersed {
                    if text.len() >= 5 {
                        emit_term(&format!("  🛡️ [GLOBAL PROTECT] 분산된 데이터(상품명 추정) 보호: '{}' ({}회 발견, 간격 넓음)", text, count));
                    } else {
                        global_boilerplate_texts.insert(text.clone());
                        emit_term(&format!("  🚫 [GLOBAL DROP] 분산된 짧은 UI(버튼) 탈락: '{}' ({}회 발견)", text, count));
                    }
                } else {
                    if text.len() > 3 {
                        let text_emb = model.get_embedding(text.clone()).await.unwrap_or(vec![0.0f32; 384]);
                        let ui_noise_score = cosine_similarity(&universal_prej_emb, &text_emb);
                        if ui_noise_score > 0.35 {
                            global_boilerplate_texts.insert(text.clone());
                            emit_term(&format!("  🚫 [GLOBAL DROP] 판정 전 전역 중복 UI 탈락: '{}' ({}회 발견, NoiseScore: {:.4})", text, count, ui_noise_score));
                        }
                    }
                }
            }
        }

        for (i, line) in pug_lines.iter().enumerate() {
            if let Some(idx) = line.find('|') {
                let text_part = line[idx + 1..].trim();
                if global_boilerplate_texts.contains(text_part) {
                    wiped_indices[i] = true;
                }
            }
        }

        let mut texts_to_embed = Vec::new();
        let mut text_indices = Vec::new();
        
        for (line_idx, line) in pug_lines.iter().enumerate() {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

            if wiped_indices[line_idx] { continue; }
            
            let text_part = if let Some(idx) = line.find('|') { line[idx + 1..].trim() } else { "" };
            if !text_part.is_empty() {
                texts_to_embed.push(text_part.to_string());
                text_indices.push(line_idx);
            }
        }

        if !texts_to_embed.is_empty() {

            for (chunk_idx, text_chunk) in texts_to_embed.chunks(100).enumerate() {
                let start_idx = chunk_idx * 100;
                if let Ok(vectors) = model.get_embedding_batch(text_chunk.to_vec()).await {
                    for (i, vector) in vectors.into_iter().enumerate() {
                        let original_idx = text_indices[start_idx + i];
                        line_embeddings[original_idx] = vector;
                    }
                }
            }
        }

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

        {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            println!("[Scheduler] Starting PURE VECTOR DETERMINISTIC RELAY (Step A)");
            
            log_task_progress(app_handle, &task.id, &json!({ "category": "Classification", "summary": "Cleaning global noise layouts...", "spinner": "⠋" }));

            let js_template = get_boa_block_extractor_template();

            let mut pre_processed_blocks = std::collections::HashSet::new();
            let mut track_a_candidates = Vec::new();
            let mut seen_candidates = std::collections::HashSet::new();

            for line_idx in 0..pug_lines.len() {
                let text_part = if let Some(idx) = pug_lines[line_idx].find('|') { pug_lines[line_idx][idx + 1..].trim() } else { "" };
                if text_part.is_empty() { continue; }
                
                let line_prej_score = cosine_similarity(&universal_prej_emb, &line_embeddings[line_idx]);
                if line_prej_score > 0.55 {
                    if !seen_candidates.contains(text_part) {
                        seen_candidates.insert(text_part.to_string());
                        track_a_candidates.push(text_part.to_string());
                    }
                }
            }

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

            let track_a_pugs: Vec<(String, String)> = {
                let mut seen_selectors = std::collections::HashSet::new();
                let mut unique_sels = Vec::new();
                for sel in track_a_selectors {
                    if !sel.is_empty() && !seen_selectors.contains(&sel) {
                        seen_selectors.insert(sel.clone());
                        unique_sels.push(sel);
                    }
                }
                
                let html_clone = clean_html_content.clone();
                
                tokio::task::spawn_blocking(move || {
                    let mut results = Vec::new();
                    let num_threads = 8;
                    let chunk_size = (unique_sels.len() + num_threads - 1) / num_threads;
                    
                    if chunk_size > 0 {
                        std::thread::scope(|s| {
                            let mut handles = Vec::new();
                            for chunk in unique_sels.chunks(chunk_size) {
                                let chunk_owned = chunk.to_vec();
                                let html_ref = &html_clone;
                                

                                handles.push(s.spawn(move || {
                                    let doc = scraper::Html::parse_document(html_ref);
                                    let mut local_res = Vec::with_capacity(chunk_owned.len());
                                    for sel in chunk_owned {
                                        let block_pug = crate::parsing::convert_doc_to_clean_pug_selector(&doc, &sel, crate::parsing::PugMode::NoAttributesMode, None);
                                        local_res.push((sel, block_pug));
                                    }
                                    local_res
                                }));
                            }
                            for h in handles {
                                if let Ok(local_res) = h.join() {
                                    results.extend(local_res);
                                }
                            }
                        });
                    }
                    results
                }).await.unwrap_or_default()
            };

            let mut unique_pugs_to_embed = Vec::new();
            let mut track_a_pugs_clean = Vec::new();
            for (sel, block_pug) in track_a_pugs {
                if block_pug.is_empty() || pre_processed_blocks.contains(&block_pug) { continue; }
                pre_processed_blocks.insert(block_pug.clone());
                unique_pugs_to_embed.push(block_pug.clone());
                track_a_pugs_clean.push((sel, block_pug));
            }

            let mut block_embeddings_map = std::collections::HashMap::new();
            if !unique_pugs_to_embed.is_empty() {
                for chunk in unique_pugs_to_embed.chunks(100) {
                    if let Ok(vectors) = model.get_embedding_batch(chunk.to_vec()).await {
                        for (i, vector) in vectors.into_iter().enumerate() {
                            block_embeddings_map.insert(chunk[i].clone(), vector);
                        }
                    }
                }
            }

            for (sel, block_pug) in track_a_pugs_clean {
                let block_emb = block_embeddings_map.get(&block_pug).cloned().unwrap_or(vec![0.0; 384]);
                let block_prej_score = cosine_similarity(&universal_prej_emb, &block_emb);
                
                if block_prej_score > 0.50 {
                    if let Some((start_idx, end_idx)) = find_block_indices_in_pug(&pug_lines, &block_pug) {
                        emit_term(&format!("  🚫 [FRONT-CLEAN] Expunged Global Layout Block: '{}' (Lines {}~{})", sel, start_idx + 1, end_idx + 1));
                        for j in start_idx..=end_idx {
                            wiped_indices[j] = true;
                        }
                    }
                }
            }

            let mut pre_filtered_pug = String::new();
            for (idx, line) in pug_lines.iter().enumerate() {
                if !wiped_indices[idx] { pre_filtered_pug.push_str(line); }
                pre_filtered_pug.push_str("\n");
            }
            filtered_light_pug = pre_filtered_pug.trim_end().to_string();

            {
                let nav_prejudice_text = "global navigation, menus, header, footer, aside, sidebar, breadcrumb, search form, pagination, admin menu, top menu, quick menu, sub menu, depth menu, side navigation, left menu, right menu, navigation bar, submenu, category menu, management menu, settings menu, configuration menu";
                let nav_prej_emb = model.get_embedding(nav_prejudice_text.to_string()).await.unwrap_or(vec![0.0f32; 384]);

                let categories = ["order", "goods", "tracking", "review", "coupon", "event"];
                let mut category_embs = Vec::new();
                for cat in &categories {
                    let anchor_text = crate::parsing::get_page_type_classification_bias(cat, &doc_lang);
                    let anchor_emb = model.get_embedding(anchor_text).await.unwrap_or(vec![0.0; 384]);
                    category_embs.push(anchor_emb);
                }

                let mut nav_wiped_count = 0usize;
                let mut nav_domain_protected = 0usize;
                for (i, line) in pug_lines.iter().enumerate() {
                    if wiped_indices[i] { continue; }
                    let trimmed = line.trim();
                    if trimmed.is_empty() { continue; }
                    if !line_embeddings[i].iter().all(|&v| v == 0.0) {
                        let nav_score = cosine_similarity(&nav_prej_emb, &line_embeddings[i]);
                        if nav_score > 0.38 {
                            let title_line_sim = cosine_similarity(&early_title_emb, &line_embeddings[i]);

                            let mut max_domain_sim = 0.0;
                            for emb in &category_embs {
                                let sim = cosine_similarity(emb, &line_embeddings[i]);
                                if sim > max_domain_sim { max_domain_sim = sim; }
                            }

                            if (max_domain_sim > 0.30 && max_domain_sim >= nav_score * 0.85) || (title_line_sim > nav_score && title_line_sim > 0.40) {
                                nav_domain_protected += 1;
                                continue;
                            }
                            wiped_indices[i] = true;
                            nav_wiped_count += 1;
                        }
                    }
                }
                if nav_wiped_count > 0 || nav_domain_protected > 0 {
                    emit_term(&format!("  🚫 [STEP-A NAV PRE-FILTER] 페이지 분류 전 네비게이션/레이아웃 {}개 라인 사전 탈락 완료. (도메인/타이틀 벡터 보호: {}개)", nav_wiped_count, nav_domain_protected));
                    let mut re_filtered = String::new();
                    for (idx, line) in pug_lines.iter().enumerate() {
                        if !wiped_indices[idx] { re_filtered.push_str(line); }
                        re_filtered.push_str("\n");
                    }
                    filtered_light_pug = re_filtered.trim_end().to_string();
                }
            }

            doc_lang = crate::utils::lang_utils::detect_document_language(&filtered_light_pug);

            println!("[Scheduler] Deterministic Detected Language: {}", doc_lang);

            let title_candidates: Vec<String> = {
                let doc = scraper::Html::parse_document(&clean_html_content);
                let norm = |s: String| -> String {
                    s.split_whitespace().collect::<Vec<_>>().join(" ").trim().to_string()
                };
                let mut cands: Vec<String> = Vec::new();
                if let Ok(sel) = scraper::Selector::parse("title") {
                    if let Some(el) = doc.select(&sel).next() {
                        let t = norm(el.text().collect::<Vec<_>>().join(" "));
                        if !t.is_empty() { cands.push(t); }
                    }
                }
                for tag in ["h1", "h2", "h3", "legend", "caption"] {
                    if let Ok(sel_h) = scraper::Selector::parse(tag) {
                        for el in doc.select(&sel_h) {
                            let t = norm(el.text().collect::<Vec<_>>().join(" "));
                            if t.is_empty() || t.chars().count() > 60 { continue; }
                            if !cands.contains(&t) { cands.push(t); }
                        }
                    }
                }
                if cands.len() > 24 { cands.truncate(24); }
                cands
            };

            let mut doc_title = title_candidates.first().cloned().unwrap_or_default();
            let mut title_emb = vec![0.0f32; 384];

            let categories = ["order", "goods", "tracking", "review", "coupon", "event"];
            let mut best_type = "".to_string();
            let mut max_total_score = -1.0;
            let mut category_scores: Vec<(String, f32, f32, usize)> = Vec::new();
            let mut category_phrase_embs: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
            let mut category_title_only_embs: Vec<(String, Vec<f32>)> = Vec::new();
            for cat in &categories {
                let anchor_text = crate::parsing::get_page_type_classification_bias(cat, &doc_lang);
                let localized_type = crate::parsing::get_localized_page_type(cat, &doc_lang);
                let mut phrases: Vec<String> = anchor_text
                    .split(|c: char| c == ',' || c == '\n' || c == '/' || c == '|')
                    .flat_map(|seg| seg.split_whitespace())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                phrases.push(cat.to_string());
                phrases.push(localized_type.clone());
                phrases.push(format!("{} {}", cat, localized_type));
                let mut seen_phrase = std::collections::HashSet::new();
                phrases.retain(|p| seen_phrase.insert(p.clone()));
                if phrases.len() > 64 { phrases.truncate(64); }

                let phrase_embs = model
                    .get_embedding_batch(phrases.clone())
                    .await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; phrases.len()]);
                category_phrase_embs.push((cat.to_string(), phrase_embs));

                let title_only_bias = format!("{} {}", cat, localized_type);
                let title_only_emb = model.get_embedding(title_only_bias).await.unwrap_or(vec![0.0; 384]);
                category_title_only_embs.push((cat.to_string(), title_only_emb));
            }

            let chrome_prejudice_text = "admin page, administrator page, management page, admin home, admin main menu, main menu, dashboard, control panel, back office, console, site name, shopping mall, welcome, home, index, search, basic search, search form, filter, login, logout, settings, configuration, my page, notice, banner, footer, copyright";
            let chrome_prej_emb = model.get_embedding(chrome_prejudice_text.to_string()).await.unwrap_or(vec![0.0f32; 384]);

            {
                if !title_candidates.is_empty() {
                    let cand_embs = model
                        .get_embedding_batch(title_candidates.clone())
                        .await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; title_candidates.len()]);

                    let mut best_idx = 0usize;
                    let mut best_score = f32::MIN;
                    for (idx, cand) in title_candidates.iter().enumerate() {
                        let emb = &cand_embs[idx];
                        if emb.iter().all(|&v| v == 0.0) { continue; }
                        let mut sims: Vec<f32> = Vec::with_capacity(categories.len());
                        for ci in 0..categories.len() {
                            sims.push(max_pool_sim(emb, &category_phrase_embs[ci].1));
                        }
                        let mean_s: f32 = sims.iter().sum::<f32>() / (sims.len() as f32);
                        let max_s: f32 = sims.iter().cloned().fold(0.0f32, f32::max);
                        let domain_contrast = max_s - mean_s;
                        let chrome_sim = cosine_similarity(&chrome_prej_emb, emb);

                        let chrome_penalty = (chrome_sim - max_s * 0.85).max(0.0);
                        let cand_score = domain_contrast - chrome_penalty * 0.5;
                        emit_term(&format!(
                            "  🏷️ [TITLE CANDIDATE] '{}' | DomainMax: {:.4} | Contrast: {:+.4} | ChromeSim: {:.4} | Score: {:+.4}",
                            cand, max_s, domain_contrast, chrome_sim, cand_score
                        ));
                        if cand_score > best_score {
                            best_score = cand_score;
                            best_idx = idx;
                        }
                    }

                    doc_title = title_candidates[best_idx].clone();
                    title_emb = cand_embs[best_idx].clone();
                    emit_term(&format!("  👑 [TITLE ANCHOR SELECTED] '{}' (Score: {:+.4})", doc_title, best_score));
                }
            }

            let mut category_title_scores: std::collections::HashMap<String, (f32, f32)> = std::collections::HashMap::new();
            let mut category_title_raw: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
            {
                let mut raw_anchor: Vec<f32> = Vec::new();
                let mut raw_only: Vec<f32> = Vec::new();
                for (ci, cat) in categories.iter().enumerate() {
                    let a = if doc_title.is_empty() { 0.0 } else { max_pool_sim(&title_emb, &category_phrase_embs[ci].1) };
                    let o = if doc_title.is_empty() { 0.0 } else { cosine_similarity(&title_emb, &category_title_only_embs[ci].1).max(0.0) };
                    raw_anchor.push(a);
                    raw_only.push(o);
                    category_title_raw.insert(cat.to_string(), a);
                }
                let n = categories.len() as f32;
                let mean_a: f32 = raw_anchor.iter().sum::<f32>() / n;
                let mean_o: f32 = raw_only.iter().sum::<f32>() / n;
                for (ci, cat) in categories.iter().enumerate() {
                    category_title_scores.insert(cat.to_string(), (raw_anchor[ci] - mean_a, raw_only[ci] - mean_o));
                }
            }

            let mut category_line_scores: std::collections::HashMap<String, (f32, usize)> = std::collections::HashMap::new();
            for cat in &categories {
                category_line_scores.insert(cat.to_string(), (0.0, 0));
            }
            let mut ambiguous_lines = 0usize;
            let mut body_sim_pool: Vec<Vec<f32>> = vec![Vec::new(); categories.len()];

            for (i, emb) in line_embeddings.iter().enumerate() {

                if wiped_indices[i] { continue; }
                let text_part = if let Some(idx) = pug_lines[i].find('|') { pug_lines[i][idx + 1..].trim() } else { "" };
                if text_part.is_empty() { continue; }
                if emb.iter().all(|&v| v == 0.0) { continue; }

                let trimmed_line = pug_lines[i].trim();
                let tag_part = trimmed_line.split('|').next().unwrap_or("").trim().to_lowercase();
                let is_table_cell = tag_part.starts_with("td") || tag_part.starts_with("th");
                let weight = if is_table_cell { 1.5 } else { 1.0 };


                let sim_threshold = if is_table_cell { 0.30 } else { 0.38 };
                let margin_threshold = if is_table_cell { 0.015 } else { 0.030 };

                let mut sims: Vec<(usize, f32)> = Vec::with_capacity(categories.len());
                for (ci, (_, phrase_embs)) in category_phrase_embs.iter().enumerate() {
                    sims.push((ci, max_pool_sim(emb, phrase_embs)));
                }

                for (ci, s) in &sims {
                    body_sim_pool[*ci].push(*s);
                }
                let mean_sim: f32 = sims.iter().map(|(_, s)| *s).sum::<f32>() / (sims.len() as f32);

                let mut ordered = sims.clone();
                ordered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let (best_ci, best_sim) = ordered[0];
                let second_sim = ordered.get(1).map(|(_, s)| *s).unwrap_or(0.0);
                let margin = best_sim - second_sim;

                if best_sim < sim_threshold { continue; }
                if margin < margin_threshold {
                    ambiguous_lines += 1;
                    continue;
                }

                let contrast = best_sim - mean_sim;
                if contrast <= 0.0 { continue; }

                let entry = category_line_scores.get_mut(categories[best_ci]).unwrap();
                entry.0 += contrast * weight;
                entry.1 += 1;
            }
            if ambiguous_lines > 0 {
                emit_term(&format!("  ⚖️ [AMBIGUITY GATE] 카테고리 간 마진 부족으로 배제된 범용 라인: {}개", ambiguous_lines));
            }

            let body_consensus: Vec<f32> = {
                let mut raw: Vec<f32> = Vec::with_capacity(categories.len());
                for ci in 0..categories.len() {
                    let mut v = body_sim_pool[ci].clone();
                    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let k = v.len().min(10);
                    let avg = if k == 0 { 0.0 } else { v[..k].iter().sum::<f32>() / (k as f32) };
                    raw.push(avg);
                }
                let mean_b: f32 = if raw.is_empty() { 0.0 } else { raw.iter().sum::<f32>() / (raw.len() as f32) };
                for (ci, cat) in categories.iter().enumerate() {
                    emit_term(&format!("  🗳️ [BODY CONSENSUS] {} | Top10Mean: {:.4} | Contrast: {:+.4}", cat, raw[ci], raw[ci] - mean_b));
                }
                raw.iter().map(|v| v - mean_b).collect()
            };

            let title_probs: Vec<f32> = {
                let combined: Vec<f32> = categories.iter().map(|c| {
                    let (a, o) = category_title_scores.get(*c).copied().unwrap_or((0.0, 0.0));
                    a + o
                }).collect();
                let mx = combined.iter().cloned().fold(f32::MIN, f32::max);
                let temp = 0.05f32;
                let exps: Vec<f32> = combined.iter().map(|v| ((v - mx) / temp).exp()).collect();
                let sum_e: f32 = exps.iter().sum::<f32>().max(1e-6);
                exps.iter().map(|e| e / sum_e).collect()
            };

            let title_window_embs: Vec<Vec<f32>> = {
                let mut windows: Vec<String> = doc_title
                    .split(|c: char| c.is_whitespace() || c == '|' || c == '/' || c == '(' || c == ')' || c == '[' || c == ']' || c == '-' || c == ',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();

                let title_chars: Vec<char> = doc_title.chars().filter(|c| !c.is_whitespace()).collect();
                for w in 2..=4usize {
                    if title_chars.len() < w { break; }
                    for st in 0..=(title_chars.len() - w) {
                        windows.push(title_chars[st..st + w].iter().collect::<String>());
                    }
                }
                let mut seen_win = std::collections::HashSet::new();
                windows.retain(|w| seen_win.insert(w.clone()));
                if windows.len() > 48 { windows.truncate(48); }
                if windows.is_empty() {
                    Vec::new()
                } else {
                    let raw_embs = model.get_embedding_batch(windows.clone()).await.unwrap_or_else(|_| vec![vec![0.0; 384]; windows.len()]);

                    let mut gated: Vec<Vec<f32>> = Vec::with_capacity(raw_embs.len());
                    let mut dropped_win = 0usize;
                    for (wi, we) in raw_embs.into_iter().enumerate() {
                        if we.iter().all(|&v| v == 0.0) { continue; }
                        let chrome_s = cosine_similarity(&chrome_prej_emb, &we);
                        let mut dom_s = 0.0f32;
                        for ci in 0..categories.len() {
                            let s = max_pool_sim(&we, &category_phrase_embs[ci].1);
                            if s > dom_s { dom_s = s; }
                        }
                        if chrome_s >= dom_s * 0.85 {
                            dropped_win += 1;
                            if dropped_win <= 8 {
                                emit_term(&format!("  🚫 [CHROME WINDOW DROP] '{}' | ChromeSim: {:.4} >= DomainMax: {:.4} x 0.85", windows[wi], chrome_s, dom_s));
                            }
                            continue;
                        }
                        gated.push(we);
                    }
                    if dropped_win > 0 {
                        emit_term(&format!("  🚫 [CHROME WINDOW GATE] 껍데기 n-gram 윈도우 {}개 제외 (잔존: {}개)", dropped_win, gated.len()));
                    }
                    gated
                }
            };

            let title_window_contrast: Vec<f32> = {
                let mut raw: Vec<f32> = Vec::new();
                for ci in 0..categories.len() {
                    let mut mx = 0.0f32;
                    for we in &title_window_embs {
                        let s = max_pool_sim(we, &category_phrase_embs[ci].1);
                        if s > mx { mx = s; }
                    }
                    raw.push(mx);
                }
                let mean_w: f32 = if raw.is_empty() { 0.0 } else { raw.iter().sum::<f32>() / (raw.len() as f32) };
                raw.iter().map(|v| v - mean_w).collect()
            };

            let title_trust: f32 = {
                let mut dom_max = 0.0f32;
                for ci in 0..categories.len() {
                    let s = max_pool_sim(&title_emb, &category_phrase_embs[ci].1);
                    if s > dom_max { dom_max = s; }
                }
                let chrome_s = cosine_similarity(&chrome_prej_emb, &title_emb);

                let chrome_trust = ((dom_max - chrome_s) / 0.15).clamp(0.0, 1.0);

                let mut wc = title_window_contrast.clone();
                wc.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let peak_margin = (wc.get(0).copied().unwrap_or(0.0) - wc.get(1).copied().unwrap_or(0.0)).max(0.0);
                let margin_trust = ((peak_margin - 0.02) / 0.08).clamp(0.0, 1.0);
                let t = chrome_trust.min(margin_trust);
                emit_term(&format!(
                    "  🔒 [TITLE TRUST] DomainMax: {:.4} | ChromeSim: {:.4} | ChromeTrust: {:.3} | PeakMargin: {:.4} | MarginTrust: {:.3} → Trust: {:.3}",
                    dom_max, chrome_s, chrome_trust, peak_margin, margin_trust, t
                ));
                t
            };

            for (ci, cat) in categories.iter().enumerate() {
                let (title_contrast, title_only_contrast) = category_title_scores.get(*cat).copied().unwrap_or((0.0, 0.0));
                let title_raw = category_title_raw.get(*cat).copied().unwrap_or(0.0);
                let (line_total, contributing_lines) = category_line_scores.get(*cat).copied().unwrap_or((0.0, 0));

                let title_signal = ((title_contrast.max(0.0) * 15.0) + (title_only_contrast.max(0.0) * 12.0)) * title_trust;

                let mean_line_contrast = if contributing_lines > 0 {
                    line_total / (contributing_lines as f32)
                } else {
                    0.0
                };
                let coverage = if contributing_lines > 0 {
                    (((contributing_lines as f32) + 1.0).ln() / 4.0).min(1.2)
                } else {
                    0.0
                };

                let evidence_factor = if contributing_lines < 3 {
                    (contributing_lines as f32) / 3.0
                } else {
                    1.0
                };
                let line_signal = mean_line_contrast * 10.0 * coverage * evidence_factor;

                let body_contrast = body_consensus.get(ci).copied().unwrap_or(0.0);
                let body_signal = body_contrast.max(0.0) * 12.0;

                let title_prior_raw = 0.5 + 3.0 * title_probs[ci];
                let title_prior = 1.0 + (title_prior_raw - 1.0) * title_trust;

                let win_contrast = title_window_contrast.get(ci).copied().unwrap_or(0.0);
                let boost_raw = (1.0 + 6.0 * win_contrast.max(0.0)).min(2.5);
                let title_keyword_boost = 1.0 + (boost_raw - 1.0) * title_trust;
                emit_term(&format!("  🔤 [TITLE SOFT-CONTAINS] {} | WindowContrast: {:+.4} → raw {:.2}x × trust {:.3} → boost {:.2}x", cat, win_contrast, boost_raw, title_trust, title_keyword_boost));

                let normalized_score = (title_signal + line_signal + body_signal) * title_prior * title_keyword_boost;
                category_scores.push((cat.to_string(), normalized_score, title_raw, contributing_lines));
                if normalized_score > max_total_score {
                    max_total_score = normalized_score;
                    best_type = cat.to_string();
                }

                emit_term(&format!(
                    "  📐 [{}] TitleMaxPool: {:.4} | Contrast: {:+.4} | TitleP: {:.3} | Prior: {:.2}x | MeanLineContrast: {:.4} | Lines: {} | Coverage: {:.3} | BodyContrast: {:+.4} | TitleSig: {:.3} | LineSig: {:.3} | BodySig: {:.3}",
                    cat, title_raw, title_contrast, title_probs[ci], title_prior, mean_line_contrast, contributing_lines, coverage, body_contrast, title_signal, line_signal, body_signal
                ));
            }

            emit_term("\n[PAGE-TYPE CLASSIFICATION] === Per-Category Score Breakdown ===");
            emit_term(&format!("  Document Title: '{}'", doc_title));
            let mut sorted_scores = category_scores.clone();
            sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (cat, score, t_sim, line_cnt) in &sorted_scores {
                let marker = if *cat == best_type { "👑" } else { "  " };
                emit_term(&format!("  {} [{}] Normalized: {:.4} | TitleMaxPool: {:.4} | ContributingLines: {}", marker, cat, score, t_sim, line_cnt));
            }
            emit_term(&format!("  Anchor Bias Sample (winner '{}'): '{}'...", best_type, crate::parsing::get_page_type_classification_bias(&best_type, &doc_lang).chars().take(120).collect::<String>()));
            emit_term("[PAGE-TYPE CLASSIFICATION] ====================================\n");
            page_type = best_type;
            println!("[Scheduler] Deterministic Classified Page Type: {} (Max Score: {:.4})", page_type, max_total_score);

            if page_type.is_empty() { 
                return Ok(()); 
            }
        }

        {
            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
            println!("[Scheduler] Starting DISK BRIDGE RELAY (Load Base -> Is Detail)");
            let (list_bias, form_bias, layout_prejudice) = crate::parsing::get_combinatorial_layout_bias(&[&page_type], &doc_lang);
            let prej_emb: Vec<f32> = model.get_embedding(layout_prejudice.clone()).await.unwrap_or(vec![0.0f32; 384]);
            let list_bias_emb: Vec<f32> = model.get_embedding(list_bias.clone()).await.unwrap_or(vec![0.0f32; 384]);
            let form_bias_emb: Vec<f32> = model.get_embedding(form_bias.clone()).await.unwrap_or(vec![0.0f32; 384]);

            let list_phrases = split_bias_phrases(&list_bias);
            let form_phrases = split_bias_phrases(&form_bias);
            let list_phrase_embs: Vec<Vec<f32>> = if list_phrases.is_empty() {
                vec![list_bias_emb.clone()]
            } else {
                model.get_embedding_batch(list_phrases.clone()).await.unwrap_or_else(|_| vec![list_bias_emb.clone(); list_phrases.len()])
            };
            let form_phrase_embs: Vec<Vec<f32>> = if form_phrases.is_empty() {
                vec![form_bias_emb.clone()]
            } else {
                model.get_embedding_batch(form_phrases.clone()).await.unwrap_or_else(|_| vec![form_bias_emb.clone(); form_phrases.len()])
            };
            emit_term(&format!("  🧩 [LAYOUT ANCHOR SPLIT] ListPhrases: {} | FormPhrases: {}", list_phrase_embs.len(), form_phrase_embs.len()));

            let layout_chrome_text = "global navigation, menus, header, footer, sidebar, breadcrumb, admin main menu, main menu, admin page, administrator page, dashboard, control panel, site name, shopping mall, welcome, home, index, basic search, search form, search filter, login, logout, notice, banner, copyright";
            let nav_chrome_emb = model.get_embedding(layout_chrome_text.to_string()).await.unwrap_or(vec![0.0f32; 384]);

            {
                let nav_prejudice_text = "global navigation, menus, header, footer, aside, sidebar, breadcrumb, search form, pagination, admin menu, top menu, quick menu, sub menu, depth menu, side navigation, left menu, right menu, top bar, bottom bar, navigation bar, submenu, category menu, management menu, settings menu, configuration menu";
                let nav_prej_emb = model.get_embedding(nav_prejudice_text.to_string()).await.unwrap_or(vec![0.0f32; 384]);

                let domain_phrase_embs: Vec<Vec<f32>> = {
                    let anchor_text = crate::parsing::get_page_type_classification_bias(&page_type, &doc_lang);
                    let localized_type = crate::parsing::get_localized_page_type(&page_type, &doc_lang);
                    let mut phrases: Vec<String> = anchor_text
                        .split(|c: char| c == ',' || c == '\n' || c == '/' || c == '|')
                        .flat_map(|seg| seg.split_whitespace())
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    phrases.push(page_type.clone());
                    phrases.push(localized_type.clone());
                    let mut seen_p = std::collections::HashSet::new();
                    phrases.retain(|p| seen_p.insert(p.clone()));
                    if phrases.len() > 64 { phrases.truncate(64); }
                    if phrases.is_empty() {
                        Vec::new()
                    } else {
                        model.get_embedding_batch(phrases.clone()).await.unwrap_or_else(|_| vec![vec![0.0; 384]; phrases.len()])
                    }
                };

                let mut nav_wiped_count = 0usize;
                let mut nav_domain_protected = 0usize;
                for (i, line) in pug_lines.iter().enumerate() {
                    if wiped_indices[i] { continue; }
                    let trimmed = line.trim();
                    if trimmed.is_empty() { continue; }

                    if !line_embeddings[i].iter().all(|&v| v == 0.0) {
                        let nav_score = cosine_similarity(&nav_prej_emb, &line_embeddings[i]);
                        if nav_score > 0.38 {

                            let mut domain_sim = 0.0f32;
                            for pe in &domain_phrase_embs {
                                let s = cosine_similarity(pe, &line_embeddings[i]);
                                if s > domain_sim { domain_sim = s; }
                            }

                            let layout_sim = cosine_similarity(&list_bias_emb, &line_embeddings[i])
                                .max(cosine_similarity(&form_bias_emb, &line_embeddings[i]));

                            let title_line_sim = cosine_similarity(&early_title_emb, &line_embeddings[i]);

                            if (domain_sim > 0.30 && domain_sim >= nav_score * 0.85)
                                || (layout_sim >= nav_score * 0.85)
                                || (title_line_sim > nav_score && title_line_sim > 0.40)
                            {
                                nav_domain_protected += 1;
                                continue;
                            }
                            wiped_indices[i] = true;
                            nav_wiped_count += 1;
                        }
                    }
                }
                if nav_wiped_count > 0 || nav_domain_protected > 0 {
                    emit_term(&format!("  🚫 [NAV PRE-FILTER] Step A-2 진입 전 네비게이션/레이아웃 {}개 라인 사전 탈락 완료. (도메인/레이아웃/타이틀 벡터 보호: {}개)", nav_wiped_count, nav_domain_protected));
                }
            }

            let system_content_a2 = format!("[PUG CONTENT]\n{}", filtered_light_pug);
            log_task_progress(app_handle, &task.id, &json!({ "category": "Classification", "summary": "Scoring DOM blocks to determine page type...", "spinner": "⠋" }));
            emit_term("\n[CLASSIFICATION] Track B & C Vector Matching (Batch DOM Blocks)...");
            let mut list_scores = Vec::new();
            let mut form_scores = Vec::new();
            for (i, emb) in line_embeddings.iter().enumerate() {

                if wiped_indices[i] { continue; }
                let text_part = if let Some(idx) = pug_lines[i].find('|') { pug_lines[i][idx + 1..].trim() } else { "" };
                if text_part.is_empty() { continue; }

                let prej_score = cosine_similarity(&prej_emb, emb);
                let list_s = cosine_similarity(&list_bias_emb, emb);
                let form_s = cosine_similarity(&form_bias_emb, emb);
                if prej_score > list_s && prej_score > form_s && prej_score > 0.35 {
                    continue;
                }
                list_scores.push((i, list_s));
                form_scores.push((i, form_s));
            }

            list_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            form_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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

            let js_template = get_boa_block_extractor_template();

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

            let valid_bc_count = track_bc_selectors.iter().filter(|s| !s.is_empty()).count();
            emit_term(&format!("  📦 [Track B & C] Boa Engine successfully mapped {}/{} structural processing blocks.", valid_bc_count, track_bc_candidates.len()));

            let track_bc_pugs: Vec<(usize, String, String)> = {
                let html_clone = clean_html_content.clone();
                let selectors_with_idx: Vec<(usize, String)> = track_bc_selectors.into_iter().enumerate().collect();
                
                tokio::task::spawn_blocking(move || {
                    let mut seen_selectors = std::collections::HashSet::new();
                    let mut unique_tasks = Vec::new();
                    let mut fallback_results = Vec::new();
                    
                    for (i, sel) in selectors_with_idx {
                        if sel.is_empty() {
                            fallback_results.push((i, sel, String::new()));
                        } else if !seen_selectors.contains(&sel) {
                            seen_selectors.insert(sel.clone());
                            unique_tasks.push((i, sel));
                        } else {
                            fallback_results.push((i, sel, String::new()));
                        }
                    }

                    let mut results = Vec::new();
                    let num_threads = 8;
                    let chunk_size = (unique_tasks.len() + num_threads - 1) / num_threads;
                    
                    if chunk_size > 0 {
                        std::thread::scope(|s| {
                            let mut handles = Vec::new();
                            for chunk in unique_tasks.chunks(chunk_size) {
                                let chunk_owned = chunk.to_vec();
                                let html_ref = &html_clone;
                                handles.push(s.spawn(move || {
                                    let doc = scraper::Html::parse_document(html_ref);
                                    let mut local_res = Vec::with_capacity(chunk_owned.len());
                                    for (i, sel) in chunk_owned {
                                        let block_pug = crate::parsing::convert_doc_to_clean_pug_selector(&doc, &sel, crate::parsing::PugMode::NoAttributesMode, None);
                                        local_res.push((i, sel, block_pug));
                                    }
                                    local_res
                                }));
                            }
                            for h in handles {
                                if let Ok(local_res) = h.join() {
                                    results.extend(local_res);
                                }
                            }
                        });
                    }
                    results.extend(fallback_results);
                    results.sort_by_key(|k| k.0);
                    results
                }).await.unwrap_or_default()
            };

            let mut total_list_score = 0.0;
            let mut processed_list_blocks = std::collections::HashSet::new();
            let mut total_form_score = 0.0;
            let mut processed_form_blocks = std::collections::HashSet::new();

            let nav_block_prejudice_text = "global navigation, menus, header, footer, aside, sidebar, breadcrumb, search form, pagination, admin menu, top menu, quick menu, sub menu, depth menu, side navigation, left menu, right menu, navigation bar, submenu, category menu, management menu, settings menu, configuration menu, snb, gnb, nav, sidebar, side bar, left panel, right panel, quick links";
            let nav_block_prej_emb = model.get_embedding(nav_block_prejudice_text.to_string()).await.unwrap_or(vec![0.0f32; 384]);

            let mut unique_bc_pugs_to_embed = Vec::new();
            let mut track_bc_pugs_clean: Vec<(usize, String, String, f32)> = Vec::new();
            for (i, sel, block_pug) in track_bc_pugs {
                let is_list_track = i < 5;
                if sel.is_empty() { 
                    let track_name = if is_list_track { "TRACK B (LIST)" } else { "TRACK C (FORM)" };
                    emit_term(&format!("  ⚠️ [{}] Anchor Line {} failed to resolve a valid structural parent block via DOM.", track_name, track_bc_indices[i] + 1));
                    continue; 
                }

                let sel_naturalized: String = {
                    let lowered = sel.to_lowercase();
                    let mut out = String::new();
                    let mut prev_is_digit = false;
                    for ch in lowered.chars() {
                        if ch.is_alphanumeric() {
                            if prev_is_digit != ch.is_ascii_digit() && !out.is_empty() {
                                out.push(' ');
                            }
                            prev_is_digit = ch.is_ascii_digit();
                            out.push(ch);
                        } else {
                            if !out.ends_with(' ') { out.push(' '); }
                            prev_is_digit = false;
                        }
                    }
                    out.split_whitespace().collect::<Vec<_>>().join(" ")
                };

                let sel_emb = model.get_embedding(sel_naturalized.clone()).await.unwrap_or(vec![0.0f32; 384]);
                let sel_nav_score = cosine_similarity(&nav_block_prej_emb, &sel_emb);

                let sel_id_class_tokens: String = sel.to_lowercase()
                    .split(|c: char| c == ' ' || c == '>')
                    .flat_map(|part| {
                        let mut tokens = Vec::new();
                        if let Some(hash_pos) = part.find('#') {
                            let id_token: String = part[hash_pos+1..].chars().take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '-').collect();
                            if !id_token.is_empty() { tokens.push(id_token.replace('_', " ").replace('-', " ")); }
                        }
                        for class_part in part.split('.') {
                            let class_token: String = class_part.chars().take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '-').collect();
                            if !class_token.is_empty() && !class_token.contains('#') { tokens.push(class_token.replace('_', " ").replace('-', " ")); }
                        }
                        tokens
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                let mut sel_id_nav_score = 0.0f32;
                let mut sel_id_emb_opt: Option<Vec<f32>> = None;
                if !sel_id_class_tokens.is_empty() {
                    let sel_id_emb = model.get_embedding(sel_id_class_tokens.clone()).await.unwrap_or(vec![0.0f32; 384]);
                    sel_id_nav_score = cosine_similarity(&nav_block_prej_emb, &sel_id_emb);
                    sel_id_emb_opt = Some(sel_id_emb);
                }

                let effective_sel_nav_score = sel_nav_score.max(sel_id_nav_score);

                let sel_content_max = {
                    let mut m = cosine_similarity(&form_bias_emb, &sel_emb)
                        .max(cosine_similarity(&list_bias_emb, &sel_emb));
                    if let Some(idc) = &sel_id_emb_opt {
                        m = m.max(cosine_similarity(&form_bias_emb, idc))
                             .max(cosine_similarity(&list_bias_emb, idc));
                    }
                    m
                };

                let nav_dominance = if sel_content_max > 0.001 {
                    effective_sel_nav_score / sel_content_max
                } else {
                    f32::MAX
                };
                let track_name = if is_list_track { "TRACK B (LIST)" } else { "TRACK C (FORM)" };
                if effective_sel_nav_score > 0.35 && nav_dominance > 1.35 {
                    emit_term(&format!("  🚫 [NAV VECTOR SELECTOR DROP] {} Anchor Line {} selector '{}' NavScore: {:.4} (ID/Class: {:.4}) | ContentSim: {:.4} | Dominance: {:.2}x > 1.35. Excluded.", track_name, track_bc_indices[i] + 1, sel, sel_nav_score, sel_id_nav_score, sel_content_max, nav_dominance));
                    continue;
                } else if effective_sel_nav_score > 0.35 {
                    emit_term(&format!("  🛡️ [NAV SELECTOR SOFT-CARRY] {} Anchor Line {} selector '{}' NavScore: {:.4} (ID/Class: {:.4}) | ContentSim: {:.4} | Dominance: {:.2}x <= 1.35. 드롭 대신 블록 단계로 이월.", track_name, track_bc_indices[i] + 1, sel, sel_nav_score, sel_id_nav_score, sel_content_max, nav_dominance));
                }
                if is_list_track {
                    if block_pug.is_empty() || processed_list_blocks.contains(&block_pug) { continue; }
                    processed_list_blocks.insert(block_pug.clone());
                } else {
                    if block_pug.is_empty() || processed_form_blocks.contains(&block_pug) { continue; }
                    processed_form_blocks.insert(block_pug.clone());
                }
                unique_bc_pugs_to_embed.push(block_pug.clone());
                track_bc_pugs_clean.push((i, sel, block_pug, effective_sel_nav_score));
            }

            let mut bc_embeddings_map = std::collections::HashMap::new();
            if !unique_bc_pugs_to_embed.is_empty() {
                for chunk in unique_bc_pugs_to_embed.chunks(100) {
                    if let Ok(vectors) = model.get_embedding_batch(chunk.to_vec()).await {
                        for (i, vector) in vectors.into_iter().enumerate() {
                            bc_embeddings_map.insert(chunk[i].clone(), vector);
                        }
                    }
                }
            }

            for (i, sel, block_pug, sel_nav_carry) in track_bc_pugs_clean {
                let is_list_track = i < 5;
                let block_emb = bc_embeddings_map.get(&block_pug).cloned().unwrap_or(vec![0.0; 384]);

                let nav_block_score = cosine_similarity(&nav_block_prej_emb, &block_emb);

                if nav_block_score > 0.25 {

                    let block_form_sim = cosine_similarity(&form_bias_emb, &block_emb);
                    let block_list_sim = cosine_similarity(&list_bias_emb, &block_emb);
                    let block_content_max = block_form_sim.max(block_list_sim);
                    if block_content_max > nav_block_score * 0.85 {
                        let track_name = if is_list_track { "TRACK B (LIST)" } else { "TRACK C (FORM)" };
                        emit_term(&format!("  🛡️ [NAV BLOCK CONTENT PROTECT] {} Anchor: {} | Selector: '{}' | NavScore: {:.4} but ContentSim: {:.4} >= 85% of Nav. Protected.", track_name, track_bc_indices[i] + 1, sel, nav_block_score, block_content_max));
                    } else {
                        let track_name = if is_list_track { "TRACK B (LIST)" } else { "TRACK C (FORM)" };
                        emit_term(&format!("  🚫 [NAV VECTOR DROP] {} Anchor: {} | Selector: '{}' | NavScore: {:.4} > 0.25. Navigation block excluded.", track_name, track_bc_indices[i] + 1, sel, nav_block_score));
                        continue;
                    }
                }
                let mut b_prej_score = cosine_similarity(&prej_emb, &block_emb);


                if nav_block_score > 0.15 {
                    b_prej_score += nav_block_score * 0.5;
                }

                if sel_nav_carry > 0.35 {
                    b_prej_score += (sel_nav_carry - 0.35) * 0.5;
                }

                if is_list_track {
                    let sel_emb = model.get_embedding(sel.to_lowercase()).await.unwrap_or(vec![0.0f32; 384]);
                    let sel_list_sim = cosine_similarity(&list_bias_emb, &sel_emb);
                    if sel_list_sim > 0.30 {
                        b_prej_score *= 0.70; 
                    }
                    let b_list_score = cosine_similarity(&list_bias_emb, &block_emb);
                    let final_score = (b_list_score - b_prej_score).max(0.0);
                    if final_score > 0.0 {
                        total_list_score += final_score;
                        emit_term(&format!("  📊 [TRACK B (LIST)] Anchor: {} | Selector: '{}' | Bias: {:.4} | Prej: {:.4} | NavScore: {:.4} | Sum: {:.4}", track_bc_indices[i] + 1, sel, b_list_score, b_prej_score, nav_block_score, final_score));
                    } else {
                        emit_term(&format!("  ⚠️ [TRACK B (LIST)] Anchor: {} Ignored. Selector: '{}' (Prej {:.4} > Bias {:.4})", track_bc_indices[i] + 1, sel, b_prej_score, b_list_score));
                    }
                } else {
                    let sel_emb = model.get_embedding(sel.to_lowercase()).await.unwrap_or(vec![0.0f32; 384]);
                    let sel_form_sim = cosine_similarity(&form_bias_emb, &sel_emb);
                    if sel_form_sim > 0.30 {
                        b_prej_score *= 0.70;
                    }
                    let b_form_score = cosine_similarity(&form_bias_emb, &block_emb);
                    let final_score = (b_form_score - b_prej_score).max(0.0);
                    if final_score > 0.0 {
                        total_form_score += final_score;
                        emit_term(&format!("  📊 [TRACK C (FORM)] Anchor: {} | Selector: '{}' | Bias: {:.4} | Prej: {:.4} | NavScore: {:.4} | Sum: {:.4}", track_bc_indices[i] + 1, sel, b_form_score, b_prej_score, nav_block_score, final_score));
                    } else {
                        emit_term(&format!("  ⚠️ [TRACK C (FORM)] Anchor: {} Ignored. Selector: '{}' (Prej {:.4} > Bias {:.4})", track_bc_indices[i] + 1, sel, b_prej_score, b_form_score));
                    }
                }
            }

            let (heading_list_sim, heading_form_sim, heading_text) = {
                let heads: Vec<(usize, String)> = {
                    let doc = scraper::Html::parse_document(&clean_html_content);
                    let mut temp: Vec<(usize, String)> = Vec::new();
                    for (tier, tag) in ["h1", "h2"].iter().enumerate() {
                        if let Ok(sel_h) = scraper::Selector::parse(tag) {
                            for el in doc.select(&sel_h) {
                                let txt = el.text().collect::<Vec<_>>().join(" ").split_whitespace().collect::<Vec<_>>().join(" ");
                                if !txt.is_empty() && txt.chars().count() <= 60 { temp.push((tier, txt)); }
                            }
                        }
                    }
                    if temp.len() > 16 { temp.truncate(16); }
                    temp
                };

                if heads.is_empty() {
                    (0.0f32, 0.0f32, String::new())
                } else {
                    let head_texts: Vec<String> = heads.iter().map(|(_, t)| t.clone()).collect();
                    let head_embs = model.get_embedding_batch(head_texts.clone()).await.unwrap_or_else(|_| vec![vec![0.0; 384]; head_texts.len()]);

                    let mut best_tier = usize::MAX;
                    let mut best_gap = -1.0f32;
                    let mut sel_l = 0.0f32;
                    let mut sel_f = 0.0f32;
                    let mut sel_txt = String::new();
                    for (hi, he) in head_embs.iter().enumerate() {
                        if he.iter().all(|&v| v == 0.0) { continue; }
                        let tier = heads[hi].0;
                        let txt = &heads[hi].1;
                        let l = max_pool_sim(he, &list_phrase_embs);
                        let f = max_pool_sim(he, &form_phrase_embs);
                        let gap = (l - f).abs();
                        let chrome_s = cosine_similarity(&nav_chrome_emb, he);
                        let layout_max = l.max(f);
                        if chrome_s >= layout_max * 0.90 {
                            emit_term(&format!("  🚫 [HEADING CHROME DROP] '{}' (h{}) | ChromeSim: {:.4} >= LayoutMax: {:.4} x 0.90", txt, tier + 1, chrome_s, layout_max));
                            continue;
                        }
                        emit_term(&format!("  🧷 [HEADING CANDIDATE] '{}' (h{}) | ListMaxPool: {:.4} | FormMaxPool: {:.4} | Gap: {:+.4} | ChromeSim: {:.4}", txt, tier + 1, l, f, l - f, chrome_s));
                        if tier < best_tier || (tier == best_tier && gap > best_gap) {
                            best_tier = tier;
                            best_gap = gap;
                            sel_l = l;
                            sel_f = f;
                            sel_txt = txt.clone();
                        }
                    }
                    (sel_l, sel_f, sel_txt)
                }
            };

            let (periodicity_contrast, best_stride, periodicity_baseline) = {
                let mut content_idxs: Vec<usize> = Vec::new();
                for (i, line) in pug_lines.iter().enumerate() {
                    if wiped_indices[i] { continue; }
                    let text_part = if let Some(p) = line.find('|') { line[p + 1..].trim() } else { "" };
                    if text_part.is_empty() { continue; }
                    if line_embeddings[i].iter().all(|&v| v == 0.0) { continue; }
                    content_idxs.push(i);
                }
                let n = content_idxs.len();
                if n < 20 {
                    (0.0f32, 0usize, 0.0f32)
                } else {
                    let max_stride = (n / 3).min(40);
                    let mut stride_means: Vec<(usize, f32)> = Vec::new();
                    for stride in 2..=max_stride {
                        let mut sum = 0.0f32;
                        let mut cnt = 0usize;
                        for k in 0..(n - stride) {
                            let a = content_idxs[k];
                            let b = content_idxs[k + stride];
                            sum += cosine_similarity(&line_embeddings[a], &line_embeddings[b]);
                            cnt += 1;
                        }
                        if cnt >= 6 { stride_means.push((stride, sum / (cnt as f32))); }
                    }
                    if stride_means.is_empty() {
                        (0.0f32, 0usize, 0.0f32)
                    } else {

                        let base: f32 = stride_means.iter().map(|(_, m)| *m).sum::<f32>() / (stride_means.len() as f32);
                        let mut bs = 0usize;
                        let mut bm = -1.0f32;
                        for (s, m) in &stride_means {
                            if *s >= 5 && *m > bm { bm = *m; bs = *s; }
                        }
                        if bs == 0 { (0.0f32, 0usize, base) } else { ((bm - base).max(0.0), bs, base) }
                    }
                }
            };

            let (row_repeat_score, row_uniformity, row_baseline, row_dbg) = {
                let harvested: Vec<(Vec<String>, usize)> = {
                    let doc = scraper::Html::parse_document(&clean_html_content);
                    let mut out: Vec<(Vec<String>, usize)> = Vec::new();
                    if let (Ok(tbl_sel), Ok(tr_sel), Ok(cell_sel)) = (
                        scraper::Selector::parse("table"),
                        scraper::Selector::parse("tr"),
                        scraper::Selector::parse("td, th"),
                    ) {
                        for tbl in doc.select(&tbl_sel) {
                            let mut rows: Vec<String> = Vec::new();
                            let mut cell_counts: Vec<usize> = Vec::new();
                            for tr in tbl.select(&tr_sel) {
                                let txt = tr.text().collect::<Vec<_>>().join(" ").split_whitespace().collect::<Vec<_>>().join(" ");
                                if txt.chars().count() < 4 { continue; }
                                rows.push(txt.chars().take(400).collect::<String>());
                                cell_counts.push(tr.select(&cell_sel).count());
                            }
                            if rows.len() < 2 { continue; }
                            if rows.len() > 30 { rows.truncate(30); cell_counts.truncate(30); }
                            let mut freq: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                            for c in &cell_counts { *freq.entry(*c).or_insert(0) += 1; }
                            let modal_cells = freq.iter().max_by_key(|(_, v)| **v).map(|(k, _)| *k).unwrap_or(0);
                            out.push((rows, modal_cells));
                        }
                    }
                    if out.len() > 10 { out.truncate(10); }
                    out
                };

                if harvested.is_empty() {
                    (0.0f32, 0.0f32, 0.0f32, String::from("no table"))
                } else {
                    let mut all_embs: Vec<Vec<Vec<f32>>> = Vec::new();
                    for (rows, _) in &harvested {
                        let e = model.get_embedding_batch(rows.clone()).await.unwrap_or_else(|_| vec![vec![0.0; 384]; rows.len()]);
                        all_embs.push(e);
                    }

                    let mut best_uni = 0.0f32;
                    let mut best_ti: i32 = -1;
                    let mut best_row_stride = 0usize;
                    for (ti, (rows, modal_cells)) in harvested.iter().enumerate() {
                        if rows.len() < 3 || *modal_cells < 3 {
                            emit_term(&format!("  ⏭️ [ROW REPETITION SKIP] table[{}] rows:{} modalCells:{} (리스트 자격 미달)", ti, rows.len(), modal_cells));
                            continue;
                        }
                        let embs = &all_embs[ti];
                        for stride in 1..=3usize {
                            if embs.len() <= stride { break; }
                            let mut s = 0.0f32;
                            let mut c = 0usize;
                            for k in stride..embs.len() {
                                s += cosine_similarity(&embs[k - stride], &embs[k]);
                                c += 1;
                            }
                            if c < 2 { continue; }
                            let m = s / (c as f32);
                            if m > best_uni {
                                best_uni = m;
                                best_ti = ti as i32;
                                best_row_stride = stride;
                            }
                        }
                    }

                    if best_ti < 0 {
                        (0.0f32, 0.0f32, 0.0f32, String::from("no qualifying list table"))
                    } else {

                        let mut base_sum = 0.0f32;
                        let mut base_cnt = 0usize;
                        for a in 0..all_embs.len() {
                            for b in (a + 1)..all_embs.len() {
                                for ea in all_embs[a].iter().take(6) {
                                    for eb in all_embs[b].iter().take(6) {
                                        base_sum += cosine_similarity(ea, eb);
                                        base_cnt += 1;
                                    }
                                }
                            }
                        }

                        if base_cnt == 0 {
                            let embs = &all_embs[best_ti as usize];
                            let far = (embs.len() / 2).max(1);
                            for k in far..embs.len() {
                                base_sum += cosine_similarity(&embs[k - far], &embs[k]);
                                base_cnt += 1;
                            }
                        }
                        let baseline = if base_cnt > 0 { base_sum / (base_cnt as f32) } else { 0.0 };


                        let n_rows = all_embs[best_ti as usize].len() as f32;
                        let volume = (((n_rows - 2.0).max(0.0)).ln_1p() / 2.0).min(1.2);

                        let contrast = (best_uni - baseline).max(0.0);
                        let score = contrast * volume;
                        let dbg = format!("table[{}] rows:{} modalCells:{} rowStride:{} volume:{:.3}",
                            best_ti, n_rows as usize, harvested[best_ti as usize].1, best_row_stride, volume);
                        (score, best_uni, baseline, dbg)
                    }
                }
            };

            emit_term(&format!("  🧱 [ROW REPETITION] {} | Uniformity: {:.4} | Baseline: {:.4} | Contrast: {:+.4} | Score: {:.4}", row_dbg, row_uniformity, row_baseline, row_uniformity - row_baseline, row_repeat_score));

            let heading_gap = heading_list_sim - heading_form_sim;
            let heading_list_bonus = heading_gap.max(0.0) * 2.0;
            let heading_form_bonus = (-heading_gap).max(0.0) * 2.0;
            let periodicity_bonus = periodicity_contrast * 2.0;
            let row_repeat_bonus = row_repeat_score * 3.0;

            let list_measured = total_list_score > 0.0001;
            let form_measured = total_form_score > 0.0001;
            let track_damp = if list_measured != form_measured { 0.5f32 } else { 1.0f32 };
            let eff_list_track = total_list_score * track_damp;
            let eff_form_track = total_form_score * track_damp;
            if track_damp < 1.0 {
                emit_term(&format!("  ⚖️ [TRACK ASYMMETRY GUARD] 한쪽 트랙 측정 실패(List: {:.4} / Form: {:.4}). 양쪽 트랙 50% 감쇠 적용.", total_list_score, total_form_score));
            }

            let list_final = eff_list_track + heading_list_bonus + periodicity_bonus + row_repeat_bonus;
            let form_final = eff_form_track + heading_form_bonus;

            emit_term(&format!("  🧭 [HEADING VECTOR] '{}' | ListMaxPool: {:.4} | FormMaxPool: {:.4} | Gap: {:+.4}", heading_text, heading_list_sim, heading_form_sim, heading_gap));
            emit_term(&format!("  🔁 [PERIODICITY COSINE] BestStride: {} | PeakContrast: {:+.4} | Baseline: {:.4}", best_stride, periodicity_contrast, periodicity_baseline));
            emit_term(&format!("  🧮 [EVIDENCE SUM] ListFinal: {:.4} (track {:.4} + heading {:.4} + period {:.4} + rows {:.4}) | FormFinal: {:.4} (track {:.4} + heading {:.4})", list_final, eff_list_track, heading_list_bonus, periodicity_bonus, row_repeat_bonus, form_final, eff_form_track, heading_form_bonus));

            let decision_margin = (form_final - list_final).abs();
            if decision_margin < 0.02 {
                emit_term(&format!("  ⚠️ [LOW-CONFIDENCE FALLBACK] 판정 마진 {:.4} < 0.02. 전체 PUG 직접 임베딩 폴백 가동.", decision_margin));
                let fallback_pug_emb = model.get_embedding(filtered_light_pug.clone()).await.unwrap_or(vec![0.0f32; 384]);

                let fallback_form_sim = max_pool_sim(&fallback_pug_emb, &form_phrase_embs);
                let fallback_list_sim = max_pool_sim(&fallback_pug_emb, &list_phrase_embs);
                let fallback_prej_sim = cosine_similarity(&prej_emb, &fallback_pug_emb);
                let fallback_form_final = (fallback_form_sim - fallback_prej_sim).max(0.0) + heading_form_bonus;
                let fallback_list_final = (fallback_list_sim - fallback_prej_sim).max(0.0) + heading_list_bonus + periodicity_bonus + row_repeat_bonus;
                is_detail = fallback_form_final > fallback_list_final;
                emit_term(&format!("  📊 [FALLBACK SCORE] FormMaxPool: {:.4} | ListMaxPool: {:.4} | PrejSim: {:.4} | FormFinal: {:.4} | ListFinal: {:.4} | is_detail: {}", fallback_form_sim, fallback_list_sim, fallback_prej_sim, fallback_form_final, fallback_list_final, is_detail));
            } else {
                is_detail = form_final > list_final;
            }
            println!("[Scheduler] Classified is_detail as: {} (Form: {:.4}, List: {:.4})", is_detail, form_final, list_final);
            emit_term(&format!("  ✅ Determined Detail Page: {}", is_detail));
        }
    }

                        
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    model.deep_purge_resources().await;
 
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
 
    crate::utils::resources::wait_for_resources_settled(1200, 800, Some(&cancellation_token), model.device_config.gpu_id as u32).await?;

    let mut extracted_data = json!({});

    if !is_detail {
        
        if !skip_ai_analysis {

            {
                use boa_engine::{Context, Source};
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                println!("[Scheduler] Starting JS-BASED SELECTOR ANALYSIS (LLM Titles -> Boa Engine)");
                
                log_task_progress(app_handle, &task.id, &json!({ "category": "Selector Search", "summary": "Analyzing DOM with JS engine...", "spinner": "⠋" }));


                let title_prompt = parsing::extract_titles_prompt(&page_type);
                let task_question = format!("{}\n\n[ACTION] RETURN JSON ONLY.", title_prompt);
                let snapshot_id = format!("{}_step_b_titles", task.id);



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

                model.deep_purge_resources().await;

                if titles.is_empty() {
                    
                    return Err(anyhow::anyhow!("[JS-BRIDGE] No titles extracted from LLM. Aborting task to prevent invalid DOM fallback."));
                }

                {
                    println!("[JS-BRIDGE] 2. Starting boa-engine for DOM analysis...");
                    let mut context = Context::default();
                    
                    let document = scraper::Html::parse_document(&clean_html_content);
                    
                    let mut nodes_json = Vec::new();
                    let mut node_to_idx = std::collections::HashMap::new();

                    for (idx, node) in document.tree.root().descendants().enumerate() {
                        node_to_idx.insert(node.id(), idx);
                    }

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
        }

        
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
        let mut cache_updated = false;
        let mut thead_pug = String::new();


        if let Some(sel) = selector_info.get("head").and_then(|v| v.as_str()) {
            if !sel.is_empty() && sel != "..." {
                final_thead_selector = sel.to_string();
                println!("[Scheduler] Using cached head selector: {}", final_thead_selector);
            }
        } 
        

        if final_thead_selector.is_empty() {

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
                    
                    
                    let ref_row_context_size = ref_row.len() + 2000;
                    let full_pug = parsing::convert_to_clean_pug(&clean_html_content, PugMode::NoAttributesMode, Some(&url));
                    let thead_light_pug = model.truncate_pug_context(&full_pug, false, 0, Some(ref_row_context_size)).await;

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
                        if let Ok(res) = gen.generate(params, Some(cancellation_token.clone()), Some(format!("{}_step_thead", task.id)), kv_name.clone(), None, None).await {
                            let thead_json = crate::parsing::parse_json_from_llm(&res);
                            

                            let mut thead_val = thead_json.get(&page_type);
                            if thead_val.is_none() {
                                if let Some(obj) = thead_json.as_object() {
                                    for (k, v) in obj {
                                        if k.to_lowercase() == page_type.to_lowercase() { thead_val = Some(v); break; }
                                    }
                                }
                            }


                            final_thead_selector = thead_val
                                .and_then(|v| v.get("thead"))
                                .and_then(|v| v.get("selector"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("").to_string().replace(">", " "); 
                            

                            let final_table_selector = thead_val
                                .and_then(|v| v.get("table"))
                                .and_then(|v| v.get("selector"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("").to_string().replace(">", " ");

                            
                            if !final_thead_selector.is_empty() && final_thead_selector != "..." && !final_table_selector.is_empty() && final_table_selector != "..." {
                                if !final_thead_selector.contains(&final_table_selector) {
                                    let combined_sel = format!("{} {}", final_table_selector, final_thead_selector);
                                    let doc = scraper::Html::parse_document(&clean_html_content);
                                    

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
                                cache_updated = true;
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

        model.deep_purge_resources().await;


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


        let mut group_size = if !thead_pug.is_empty() {
            let mut max_span = 1;

            if let Ok(re) = regex::Regex::new(r#"rowspan="(\d+)""#) {
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

            let first_item_tr_count = pug_list.first()
                .map(|p| p.lines().filter(|l| {
                    let indent = l.chars().take_while(|c| c.is_whitespace()).count();
                    indent == 0 && (l.starts_with("tr") || l.starts_with("tr["))
                }).count())
                .unwrap_or(1);


            if first_item_tr_count >= group_size || first_item_tr_count > 1 {
                println!("[Scheduler] 🌟 Items are already grouped ({} trs per item). Skipping manual chunking.", first_item_tr_count);
                group_size = 1;
            } else {
                let mut grouped = Vec::new();
                for chunk in pug_list.chunks(group_size) {
                    grouped.push(chunk.join("\n"));
                }
                pug_list = grouped;
                println!("[Scheduler] 🌟 Grouped multi-row items: {} rows per item. Total items reduced to {}.", group_size, pug_list.len());
            }
        }

        if !pug_list.is_empty() {
            let total_items = pug_list.len();
            let mut text_frequency: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            // 🌟 [P0-1 SUBORDINATE] 같은 셀(td) 안에 "실제 상세페이지로 가는 href" 형제가 존재하는데
            //    자신은 링크가 없는 라인 = 그 컬럼의 대표값이 아닌 종속 라인(액션 버튼 / 옵션 나열).
            //    'li | 상품 상세보기' 와 'a[href=".../ProductRegister?product_no=18"] | 테스트상품' 이
            //    같은 td 에 있으므로 전자는 여기서 구조적으로 확정 탈락됩니다.
            let mut subordinate_texts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            // 🌟 [P0-2 DEAD ACTION] href 속성은 있으나 전부 '#', '#none', 'javascript:' 인 순수 UI 액션 버튼.
            //    'SMS발송', 'SNS공유', '주소복사' 가 여기에 해당합니다.
            let mut dead_action_texts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

            for item_pug in &pug_list {
                let mut seen_in_this_item = std::collections::HashSet::new();
                for line in item_pug.lines() {

                    if let Some(idx) = line.find('|') {
                        let text_part = line[idx + 1..].trim();
                        if !text_part.is_empty() && text_part.len() > 2 {
                            seen_in_this_item.insert(text_part.to_string());
                        }
                    }
                }
                for text in seen_in_this_item {
                    *text_frequency.entry(text).or_insert(0) += 1;
                }

                let cell_lines: Vec<String> = item_pug.lines().map(|s| s.to_string()).collect();
                let mut seen_sub = std::collections::HashSet::new();
                let mut seen_dead = std::collections::HashSet::new();

                for cell in parse_pug_grid(&cell_lines) {
                    let has_real_link = cell.line_indices.iter()
                        .any(|&li| line_real_href(&cell_lines[li]).is_some());
                    if !has_real_link { continue; }
                    for &li in &cell.line_indices {
                        if line_real_href(&cell_lines[li]).is_some() { continue; }
                        if let Some(p) = cell_lines[li].find('|') {
                            let t = cell_lines[li][p + 1..].trim();
                            if t.len() > 2 { seen_sub.insert(t.to_string()); }
                        }
                    }
                }

                for line in &cell_lines {
                    if !line.contains("href=") { continue; }
                    if line_real_href(line).is_some() { continue; }
                    if let Some(p) = line.find('|') {
                        let t = line[p + 1..].trim();
                        if t.len() > 2 { seen_dead.insert(t.to_string()); }
                    }
                }

                for t in seen_sub { *subordinate_texts.entry(t).or_insert(0) += 1; }
                for t in seen_dead { *dead_action_texts.entry(t).or_insert(0) += 1; }
            }

            let mut boilerplate_texts = std::collections::HashSet::new();

            let fields = parsing::get_list_schema_fields(&page_type, &url, &doc_lang);
            let total_fields = fields.len();

            let enum_guard_embs: Vec<Vec<f32>> = {
                let mut embs = Vec::new();
                for (fname, _, bias_target, _) in fields.iter() {
                    let is_enum_like = fname.contains("status")
                        || fname.contains("payment_method")
                        || fname.contains("payment_origin")
                        || fname.contains("condition")
                        || fname.contains("currency");
                    if is_enum_like {
                        let e = model.get_embedding(bias_target.clone()).await.unwrap_or(vec![0.0; 384]);
                        embs.push(e);
                    }
                }
                embs
            };

            if total_items >= 2 {
                let threshold = (total_items as f32 * 0.7).ceil() as usize; 
                
                let re_numeric = regex::Regex::new(r"^\D*\d+[\d,\.]*\D*$").unwrap();

                for (text, count) in text_frequency {
                    if count >= threshold {

                        let is_numeric_data = re_numeric.is_match(&text);
                        
                        if !is_numeric_data && text.len() > 3 {

                            // 🌟 [P0 ACTION GATE] 벡터 유사도를 재기 '전에' 구조적 사실부터 확정합니다.
                            //    (1) 같은 셀에 실제 이동 링크 형제가 있는데 자신은 링크가 없다 → 컬럼 대표값이 아님
                            //    (2) href 가 '#', '#none', 'javascript:' 뿐이다 → 순수 UI 액션 버튼
                            //    이 두 경우는 enum 유사도가 아무리 높아도 실데이터가 될 수 없습니다.
                            //    기존에는 '상품 상세보기'(0.5135) '쇼핑몰화면 진열보기'(0.5090) 가
                            //    ENUM VECTOR PROTECT 로 살아남아 title 을 오염시켰습니다.
                            let sub_hits = subordinate_texts.get(&text).copied().unwrap_or(0);
                            let dead_hits = dead_action_texts.get(&text).copied().unwrap_or(0);
                            if sub_hits >= threshold || dead_hits >= threshold {
                                boilerplate_texts.insert(text.clone());
                                emit_term(&format!("[Scheduler] 🚫 [ACTION LINE DROP] 구조적으로 UI 액션/종속 라인 확정 탈락: '{}' ({} / {} 아이템 | Subordinate: {} | DeadHref: {})", text, count, total_items, sub_hits, dead_hits));
                                continue;
                            }

                            let mut enum_sim = 0.0f32;
                            if !enum_guard_embs.is_empty() {
                                let t_emb = model.get_embedding(text.clone()).await.unwrap_or(vec![0.0f32; 384]);
                                for ge in &enum_guard_embs {
                                    let s = cosine_similarity(ge, &t_emb);
                                    if s > enum_sim { enum_sim = s; }
                                }
                            }

                            if enum_sim > 0.30 {
                                emit_term(&format!("[Scheduler] 🛡️ [ENUM VECTOR PROTECT] 반복되지만 enum 스키마 유사도({:.4})가 높아 실데이터로 보호: '{}' ({} / {} 아이템)", enum_sim, text, count, total_items));
                                continue;
                            }

                            boilerplate_texts.insert(text.clone());
                            emit_term(&format!("[Scheduler] 🚫 전역 중복 텍스트 사전 탈락(Drop): '{}' ({} / {} 아이템에서 발견, EnumSim: {:.4})", text, count, total_items, enum_sim));
                        }
                    }
                }
            }


            let doc_title = {
                let doc = scraper::Html::parse_document(&clean_html_content);
                let mut t_val = if let Ok(sel) = scraper::Selector::parse("title") {
                    doc.select(&sel).next().map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string()).unwrap_or_default()
                } else {
                    String::new()
                };
                
                if t_val.is_empty() || t_val.len() < 5 {
                    let mut heading_texts = Vec::new();
                    if let Ok(sel_h1) = scraper::Selector::parse("h1") {
                        for el in doc.select(&sel_h1) {
                            heading_texts.push(el.text().collect::<Vec<_>>().join(" ").trim().to_string());
                        }
                    }
                    if let Ok(sel_h2) = scraper::Selector::parse("h2") {
                        for el in doc.select(&sel_h2) {
                            heading_texts.push(el.text().collect::<Vec<_>>().join(" ").trim().to_string());
                        }
                    }
                    if !heading_texts.is_empty() {
                        t_val = heading_texts.join(" | ");
                    }
                }
                t_val
            };

            model.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancellation_token.clone()), false, Some("inference".to_string())).await?;

            let mut field_embeddings = Vec::new();
            // 🌟 [PHRASE-LEVEL BIAS BANK] 콤마 나열 문자열 전체를 하나로 임베딩하면 센트로이드가 되어
            //    변별력이 사라집니다(로그의 Item Score 0.0000 원인). 구 단위로 쪼개 Max-Pool 뱅크를 만듭니다.
            let mut field_phrase_embs: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut field_phrase_weights: Vec<Vec<f32>> = Vec::new();
            // 🌟 insight/summary/analysis 계열은 '단일 셀 복사'가 아니라 '문장 합성' 필드입니다.
            //    라인 벡터 매칭 대상에 넣는 것 자체가 "89000", "본사" 오염의 근원이므로 원천 제외합니다.
            let mut field_is_analytic: Vec<bool> = Vec::new();
            // 🌟 [FORMAT BANK] 각 필드가 요구하는 값의 물리적 생김새를 미리 확정해 둡니다.
            //    코사인 유사도를 재기 전에 이 게이트로 후보를 걸러야 "번호 | 11" → tracking_number 같은
            //    벡터 임계치로는 절대 못 잡는 오매칭이 원천 차단됩니다.
            let mut field_formats: Vec<FieldFormat> = Vec::new();

            for (f_idx, (fname, _, bias_target, predefined_prej)) in fields.iter().enumerate() {
                let bias_emb = model.get_embedding(bias_target.clone()).await.unwrap_or(vec![0.0; 384]);

                let (phrases, phrase_weights) = split_bias_phrases_weighted(bias_target);
                let p_embs = if phrases.is_empty() {
                    vec![bias_emb.clone()]
                } else {
                    model.get_embedding_batch(phrases.clone()).await.unwrap_or_else(|_| vec![bias_emb.clone(); phrases.len()])
                };
                let p_weights = if phrases.is_empty() { vec![1.0f32] } else { phrase_weights };
                field_phrase_embs.push(p_embs);
                field_phrase_weights.push(p_weights);

                let detected_fmt = detect_field_format(fname);
                field_formats.push(detected_fmt);
                emit_term(&format!("  📐 [FORMAT REGISTERED] '{}' → {:?}", fname, detected_fmt));

                let lower_fname = fname.to_lowercase();
                let is_analytic = lower_fname.contains("insight")
                    || lower_fname.contains("summary")
                    || lower_fname.contains("analysis");
                field_is_analytic.push(is_analytic);
                if is_analytic {
                    emit_term(&format!("  🧠 [SYNTHESIS FIELD REGISTERED] '{}' 는 벡터 라인 매칭에서 제외되고 전체 컨텍스트 요약 필드로 처리됩니다.", fname));
                }

                let mut dynamic_prej_texts = Vec::new();
                if !predefined_prej.trim().is_empty() {
                    dynamic_prej_texts.push(predefined_prej.clone());
                }
                for (other_idx, (_, _, other_bias, _)) in fields.iter().enumerate() {
                    if f_idx != other_idx {
                        dynamic_prej_texts.push(other_bias.clone());
                    }
                }
                let combined_prej = dynamic_prej_texts.join(" , ");
                let prej_emb = model.get_embedding(combined_prej.clone()).await.unwrap_or(vec![0.0; 384]);

                field_embeddings.push((bias_emb, prej_emb, combined_prej));
            }


            let (_, layout_prejudice) = crate::parsing::get_layout_bias(&page_type, &doc_lang);
            let layout_prej_emb = model.get_embedding(layout_prejudice.clone()).await.unwrap_or(vec![0.0; 384]);

            let mut thead_lines: Vec<String> = thead_pug.lines().map(|s| s.to_string()).collect();
            let mut thead_embeddings = vec![vec![0.0; 384]; thead_lines.len()];
            

            let thead_cells = parse_pug_grid(&thead_lines);
            let mut header_cols: std::collections::HashMap<usize, String> = std::collections::HashMap::new();

            for cell in &thead_cells {
                for c in cell.col..(cell.col + cell.colspan) {
                    let existing = header_cols.entry(c).or_insert(String::new());
                    if !existing.is_empty() && !cell.text.is_empty() {
                        existing.push_str(" > ");
                    }
                    if !cell.text.is_empty() {
                        existing.push_str(&cell.text);
                    }
                }
            }

            if !thead_lines.is_empty() {
                emit_term(&format!("\n[PRE-PROCESSING] Vectorizing Table Header ({} lines)...", thead_lines.len()));
                
                let mut texts_to_embed = Vec::new();
                let mut text_indices = Vec::new();
                
                for (line_idx, line) in thead_lines.iter().enumerate() {
                    if !line.trim().is_empty() {
                        texts_to_embed.push(line.to_string());
                        text_indices.push(line_idx);
                    }
                }
                
                if !texts_to_embed.is_empty() {
                    for (chunk_idx, text_chunk) in texts_to_embed.chunks(100).enumerate() {
                        let start_idx = chunk_idx * 100;
                        if let Ok(vectors) = model.get_embedding_batch(text_chunk.to_vec()).await {
                            for (i, vector) in vectors.into_iter().enumerate() {
                                let original_idx = text_indices[start_idx + i];
                                let emb = vector.clone();
                                let noise_score = cosine_similarity(&layout_prej_emb, &emb);
                                
                                let original_text = text_chunk[i].trim();
                                let has_digit = original_text.chars().any(|c| c.is_ascii_digit());
                                let is_short = original_text.len() <= 3;
                                

                                let is_structure_tag = original_text.starts_with("th") 
                                    || original_text.starts_with("td") 
                                    || original_text.starts_with("tr") 
                                    || original_text.starts_with("input")
                                    || original_text.starts_with("div");
                                

                                if noise_score > 0.6 && !has_digit && !is_short && !is_structure_tag {
                                    emit_term(&format!("    🚫 [NOISE FILTERED] Header Line {} : {} (Score: {:.4})", original_idx + 1, original_text, noise_score));
                                    thead_lines[original_idx] = String::new(); 
                                } else {
                                    thead_embeddings[original_idx] = emb;
                                }
                            }
                        }
                    }
                }
            }


            let mut unique_headers = Vec::new();
            for (_, h_text) in &header_cols {
                let clean_h = h_text.trim();
                if !clean_h.is_empty() && !unique_headers.contains(&clean_h.to_string()) {
                    unique_headers.push(clean_h.to_string());
                }
            }

            let mut header_to_field_map = std::collections::HashMap::new();

            if !unique_headers.is_empty() {
                // 🌟 [HEADER COSINE MAP] LLM 컬럼 매핑을 전면 폐기하고 코사인 2회로 확정합니다.
                //    기존 코드는 이미 한 덩어리로 합쳐진 bias_target("order status state")을 통째로
                //    임베딩해 비교했기 때문에 구가 1개뿐이었고, 그래서 로그의 MaxPoolSim 과
                //    CentroidSim 이 소수점 4자리까지 동일했으며 18개 헤더가 전부 탈락했습니다.
                //    여기서는 bias.json 의 {lang}.{type}.{field} 노드를 직접 읽어
                //    라벨 구 뱅크(semantic + 비수치 bias)와 편견 구 뱅크(prejudice)를 각각 임베딩하고
                //    score = 라벨MaxPool - 편견MaxPool 로 판정합니다.
                let header_embs: Vec<Vec<f32>> = model
                    .get_embedding_batch(unique_headers.clone())
                    .await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; unique_headers.len()]);

                let mut hdr_field_names: Vec<String> = Vec::new();
                let mut hdr_label_embs: Vec<Vec<Vec<f32>>> = Vec::new();
                let mut hdr_label_weights: Vec<Vec<f32>> = Vec::new();
                let mut hdr_prej_embs: Vec<Vec<Vec<f32>>> = Vec::new();

                for (fname, _, _, _) in &fields {
                    let (label_phrases, label_weights) = label_phrase_bank(&doc_lang, &page_type, fname);
                    if label_phrases.is_empty() { continue; }
                    let prej_phrases = prejudice_phrase_bank(&doc_lang, &page_type, fname);

                    let l_embs = model.get_embedding_batch(label_phrases.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; label_phrases.len()]);
                    let p_embs = if prej_phrases.is_empty() {
                        Vec::new()
                    } else {
                        model.get_embedding_batch(prej_phrases.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; prej_phrases.len()])
                    };

                    emit_term(&format!("  🏷️ [LABEL BANK] '{}' | 라벨 구 {}개 | 편견 구 {}개", fname, label_phrases.len(), p_embs.len()));
                    hdr_field_names.push(fname.clone());
                    hdr_label_embs.push(l_embs);
                    hdr_label_weights.push(label_weights);
                    hdr_prej_embs.push(p_embs);
                }

                let hdr_abs_floor = 0.62f32;
                let hdr_score_floor = 0.10f32;
                let hdr_margin = 0.03f32;

                let mut hdr_matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; unique_headers.len()]; hdr_field_names.len()];
                for f in 0..hdr_field_names.len() {
                    for h in 0..unique_headers.len() {
                        if header_embs[h].iter().all(|&v| v == 0.0) { continue; }
                        let own = weighted_max_pool_sim(&header_embs[h], &hdr_label_embs[f], &hdr_label_weights[f]);
                        if own < hdr_abs_floor { continue; }
                        let prej = if hdr_prej_embs[f].is_empty() { 0.0 } else { max_pool_sim(&header_embs[h], &hdr_prej_embs[f]) };
                        let score = own - prej;
                        if score < hdr_score_floor {
                            emit_term(&format!("    🚫 [HEADER PREJUDICE DROP] '{}' → '{}' | LabelMaxPool: {:.4} | PrejMaxPool: {:.4} | Score: {:+.4} < {:.2}", unique_headers[h], hdr_field_names[f], own, prej, score, hdr_score_floor));
                            continue;
                        }
                        hdr_matrix[f][h] = score;
                    }
                }

                let hdr_assign = exclusive_assign(&hdr_matrix, hdr_score_floor, hdr_margin);
                for (f, a) in hdr_assign.iter().enumerate() {
                    match a {
                        Some((h, score, margin)) => {
                            header_to_field_map.insert(unique_headers[*h].clone(), hdr_field_names[f].clone());
                            emit_term(&format!("    ✨ [HEADER COSINE MAP] Header '{}' → Field '{}' | Score: {:+.4} | Margin: {:+.4}", unique_headers[*h], hdr_field_names[f], score, margin));
                        },
                        None => {
                            emit_term(&format!("    ⚪ [HEADER UNMAPPED] Field '{}' | 확정 가능한 헤더 없음. 값 라인 벡터 매칭으로 폴백합니다.", hdr_field_names[f]));
                        }
                    }
                }
            }

            // 🌟 [ID/LINK COSINE BANK] href 후보의 "URL 상 역할 문구"와 소급 복구 후보의 "컬럼 라벨"을
            //    코사인으로 채점하기 위한 라벨/편견 뱅크입니다.
            //    문자열 포함(contains) 검사로는 도메인 조각 'cafe24' 와 쿼리값 '18' 을 절대 구분할 수 없습니다.
            let (idlink_label_phrases, idlink_label_weights) = label_phrase_bank(&doc_lang, &page_type, "id,link");
            let idlink_label_embs: Vec<Vec<f32>> = if idlink_label_phrases.is_empty() {
                Vec::new()
            } else {
                model.get_embedding_batch(idlink_label_phrases.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; idlink_label_phrases.len()])
            };

            let mut idlink_prej_phrases = prejudice_phrase_bank(&doc_lang, &page_type, "id,link");
            for extra in [
                "host name", "domain name", "website address", "server address",
                "cdn", "static asset", "image server", "protocol", "www",
                "file extension", "stylesheet", "script", "anchor", "javascript",
                "navigation menu", "layer popup",
            ] {
                let e = extra.to_string();
                if !idlink_prej_phrases.contains(&e) { idlink_prej_phrases.push(e); }
            }
            let idlink_prej_embs: Vec<Vec<f32>> = if idlink_prej_phrases.is_empty() {
                Vec::new()
            } else {
                model.get_embedding_batch(idlink_prej_phrases.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; idlink_prej_phrases.len()])
            };
            emit_term(&format!("  🔑 [ID/LINK COSINE BANK] 라벨 구 {}개 | 편견 구 {}개 준비 완료.", idlink_label_embs.len(), idlink_prej_embs.len()));

            // 🌟 [ID/LINK PATTERN TRACKER] 아이템 루프 중 id/link 확정 성공 시 URL 패턴과
            //    식별자 '생김새'(자릿수/숫자전용 여부), 기준 링크(호스트 검증용)를 함께 기억합니다.
            //    실패한 아이템의 원시 라인과 라벨 라인을 보관해 루프 종료 후 소급 복구에 사용합니다.
            let mut discovered_url_pattern: Option<(String, String)> = None; // (prefix, suffix)
            let mut pattern_reference_link: Option<String> = None;
            let mut confirmed_id_shapes: Vec<(usize, bool)> = Vec::new();
            let mut all_item_raw_lines: Vec<Vec<String>> = Vec::new();
            let mut all_item_labeled_lines: Vec<Vec<String>> = Vec::new();

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


                let full_item_pug = format!("{}\n{}", thead_pug, item_pug);
                

                let mut item_lines: Vec<String> = item_pug.lines().map(|s| s.to_string()).collect();
                

                for i in 0..item_lines.len() {
                    let line = &item_lines[i];
                    if let Some(idx) = line.find('|') {
                        let text_part = line[idx + 1..].trim();
                        if boilerplate_texts.contains(text_part) {

                            // 🌟 [P0] '#none' / '#' / '#layerSnsShare' / 'javascript:' 같은 죽은 href 는
                            //    데이터 링크가 아니라 UI 액션 훅이므로 보호 대상에서 제외합니다.
                            //    (기존 line.contains("href=") 는 'a[href="#none"] | SMS발송' 까지 보호했습니다)
                            let has_link_or_event = line_real_href(line).is_some() || line.contains("onclick") || line.contains("data-url");
                            if has_link_or_event {
                                emit_term(&format!("    🛡️ [DUPLICATE LINK PROTECT] Item Line {}/{} : {} (실제 이동 href/event 포함 데이터 보호)", i + 1, item_lines.len(), text_part));
                                continue;
                            }

                            emit_term(&format!("    🚫 [DUPLICATE FILTERED] Item Line {}/{} : {} (반복 UI 탈락)", i + 1, item_lines.len(), text_part));

                            item_lines[i] = format!("{} ", &line[..=idx]);
                        }
                    }
                }


                let item_cells = parse_pug_grid(&item_lines);
                let mut line_enriched_texts = vec![String::new(); item_lines.len()];
                // 🌟 [LINE OWNER] 헤더 코사인으로 확정된 컬럼이 어떤 라인을 소유하는지 기록합니다.
                //    이 기록이 있어야 (1) 정답 셀이 노이즈 필터에 삭제되지 않고
                //    (2) 다른 컬럼이 그 라인을 벡터로 선점하지 못합니다.
                let mut line_owner_field: Vec<Option<String>> = vec![None; item_lines.len()];
                
                for cell in &item_cells {
                    let h_text = header_cols.get(&cell.col).cloned().unwrap_or_default();
                    let owner = header_to_field_map.get(h_text.trim()).cloned();
                    for &line_idx in &cell.line_indices {
                        if let Some(o) = &owner {
                            line_owner_field[line_idx] = Some(o.clone());
                        }
                        let original_text = if let Some(p) = item_lines[line_idx].find('|') {
                            item_lines[line_idx][p + 1..].trim()
                        } else {
                            ""
                        };
                        if !original_text.is_empty() {
                            line_enriched_texts[line_idx] = if h_text.is_empty() {
                                original_text.to_string()
                            } else {
                                format!("{} | {}", h_text, original_text)
                            };
                        }
                    }
                }

                let mut item_embeddings = vec![vec![0.0; 384]; item_lines.len()];
                

                let mut texts_to_embed = Vec::new();
                let mut text_indices = Vec::new();
                
                for (line_idx, line) in item_lines.iter().enumerate() {
                    if !line.trim().is_empty() {
                        let enriched = &line_enriched_texts[line_idx];
                        let target_text = if enriched.is_empty() {
                            if let Some(p) = line.find('|') { line[p + 1..].trim() } else { "" }
                        } else {
                            enriched.as_str()
                        };

                        if !target_text.is_empty() {
                            texts_to_embed.push(target_text.to_string());
                            text_indices.push(line_idx);
                        }
                    }
                }
                
                if !texts_to_embed.is_empty() {
                    for (chunk_idx, text_chunk) in texts_to_embed.chunks(100).enumerate() {
                        let start_idx = chunk_idx * 100;
                        if let Ok(vectors) = model.get_embedding_batch(text_chunk.to_vec()).await {
                            for (i, vector) in vectors.into_iter().enumerate() {
                                let original_idx = text_indices[start_idx + i];
                                let emb = vector.clone();
                                let noise_score = cosine_similarity(&layout_prej_emb, &emb);
                                
                                let original_text = text_chunk[i].trim();
                                let has_digit = original_text.chars().any(|c| c.is_ascii_digit());
                                let is_short = original_text.len() <= 3;
                                

                                let is_structure_tag = original_text.starts_with("th") 
                                    || original_text.starts_with("td") 
                                    || original_text.starts_with("tr") 
                                    || original_text.starts_with("input")
                                    || original_text.starts_with("div");
                                

                                let is_header_owned = line_owner_field[original_idx].is_some();

                                if noise_score > 0.6 && !has_digit && !is_short && !is_structure_tag && !is_header_owned {
                                    emit_term(&format!("    🚫 [NOISE FILTERED] Item Line {}/{} : {} (Score: {:.4})", original_idx + 1, item_lines.len(), original_text, noise_score));
                                    item_lines[original_idx] = String::new(); 
                                } else {
                                    if noise_score > 0.6 && is_header_owned {
                                        emit_term(&format!("    🛡️ [HEADER OWNED PROTECT] Item Line {}/{} : {} (NoiseScore {:.4} 이지만 '{}' 컬럼으로 코사인 확정됨)", original_idx + 1, item_lines.len(), original_text, noise_score, line_owner_field[original_idx].clone().unwrap_or_default()));
                                    } else {
                                        emit_term(&format!("    [VECTORIZING] Item Line {}/{} : {}", original_idx + 1, item_lines.len(), original_text));
                                    }
                                    item_embeddings[original_idx] = emb;
                                }
                            }
                        }
                    }
                }


                let mut json_contexts = Vec::new();
                for (line_idx, line) in item_lines.iter().enumerate() {
                    if !line.trim().is_empty() {
                        let enriched = &line_enriched_texts[line_idx];
                        let target_text = if enriched.is_empty() {

                            if let Some(p) = line.find('|') { line[p + 1..].trim() } else { "" }
                        } else {
                            enriched.as_str()
                        };

                        if !target_text.is_empty() {
                            if let Some(idx) = target_text.find('|') {
                                json_contexts.push(json!({
                                    "metadata": target_text[..idx].trim(),
                                    "value": target_text[idx + 1..].trim()
                                }));
                            } else {
                                json_contexts.push(json!({
                                    "value": target_text.trim()
                                }));
                            }
                        }
                    }
                }
                let filtered_full_item_pug = serde_json::to_string_pretty(&json_contexts).unwrap_or_default();

                let mut item_val = json!({});
                let mut global_ignore_list: Vec<String> = Vec::new();
                

                let thead_lines_ref: Vec<&str> = thead_lines.iter().map(|s| s.as_str()).collect();
                let item_lines_ref: Vec<&str> = item_lines.iter().map(|s| s.as_str()).collect();


                // 🌟 [DEDUP] 아이템마다 동일한 필드 임베딩을 재계산하던 중복 블록을 제거했습니다.
                //    (아이템 11개 × 필드 11개 = 동일 임베딩 121회 재계산 → 속도 저하 + 결과 동일)
                //    리스트 루프 바깥에서 1회 구축한 field_embeddings / field_phrase_embs / field_phrase_weights /
                //    field_is_analytic 을 그대로 재사용합니다.

                // 🌟 [VALUE EXTRACTION] 형식 게이트는 반드시 "헤더가 아닌 값"만 검사해야 합니다.
                //    임베딩 대상은 "번호 | 11" 처럼 헤더가 붙은 문자열이지만,
                //    형식 판정 대상은 파이프 뒤의 "11" 하나뿐이어야 합니다.
                let line_values: Vec<String> = item_lines_ref.iter().map(|line| {
                    match line.find('|') {
                        Some(p) => line[p + 1..].trim().to_string(),
                        None => String::new(),
                    }
                }).collect();


                let mut pre_mapped_hints = Vec::new();
                

                let mut url_pool = String::new();
                if let Ok(href_re) = regex::Regex::new(r#"href=["']([^"']+)["']"#) {
                    for line in &item_lines_ref {
                        for cap in href_re.captures_iter(line) {
                            if let Some(m) = cap.get(1) {
                                url_pool.push_str(&m.as_str().to_lowercase());
                                url_pool.push_str(" ");
                            }
                        }
                    }
                }

                // 🌟 [HEADER OWNED ROUTING] 헤더 코사인으로 확정된 컬럼만 처리합니다.
                //    - enum 계열(status/payment_method 등)은 셀 값이 "취소"/"무통장" 이라 그대로 저장하면
                //      parse_status 가 인식하지 못하므로 LLM 우회 대신 "벡터 배정 강제"로 보냅니다.
                //    - id,link 는 결정론적 href 해석기 + URL 패턴 소급 복구가 이미 완성되어 있으므로
                //      선점만 걸어두고 값 주입은 하지 않습니다.
                //    - 나머지는 LLM 없이 셀 값을 그대로 확정합니다.
                let mut header_forced_assign: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                let mut header_owned_lines: std::collections::HashSet<usize> = std::collections::HashSet::new();
                let mut header_id_tokens: Vec<String> = Vec::new();

                for line_idx in 0..item_lines_ref.len() {
                    let owner_field = match &line_owner_field[line_idx] {
                        Some(o) => o.clone(),
                        None => continue,
                    };
                    if item_lines_ref[line_idx].trim().is_empty() { continue; }

                    let target_text = if !line_enriched_texts[line_idx].is_empty() { &line_enriched_texts[line_idx] } else { item_lines_ref[line_idx] };
                    let clean_text = if let Some(idx) = target_text.find('|') { target_text[idx + 1..].trim() } else { "" };
                    if clean_text.is_empty() || clean_text.chars().count() < 2 { continue; }

                    header_owned_lines.insert(line_idx);

                    if is_id_link_field(&owner_field) {
                        for tok in clean_text.split(|c: char| !c.is_alphanumeric()) {
                            if tok.chars().count() < 6 { continue; }
                            if !tok.chars().any(|c| c.is_ascii_digit()) { continue; }
                            if !header_id_tokens.iter().any(|t| t == tok) { header_id_tokens.push(tok.to_string()); }
                        }
                        emit_term(&format!("    🔑 [HEADER OWNED / ID COLUMN] Item Line {} 는 '{}' 컬럼입니다. 결정론적 ID/LINK 해석기에 위임하고 타 컬럼 선점을 차단합니다.", line_idx + 1, owner_field));
                        continue;
                    }

                    let lower_owner = owner_field.to_lowercase();
                    let needs_normalization = lower_owner.contains("status")
                        || lower_owner.contains("payment_method")
                        || lower_owner.contains("payment_origin")
                        || lower_owner.contains("condition")
                        || lower_owner.contains("currency");

                    if needs_normalization {
                        header_forced_assign.entry(owner_field.clone()).or_insert(line_idx);
                        emit_term(&format!("    🎯 [HEADER FORCED ASSIGN] '{}' ← Item Line {} (\"{}\") | enum 정규화가 필요해 값 우회 대신 벡터 배정을 확정합니다.", owner_field, line_idx + 1, clean_text));
                        continue;
                    }

                    // 🌟 [P1 REPRESENTATIVE VALUE] 같은 컬럼(td) 안에 액션 버튼·옵션 나열이 섞여 있어도
                    //    실제 상세페이지로 이어지는 링크를 가진 라인 하나만 대표값으로 채택합니다.
                    //    Rank 2 = 실제 이동 href 보유(진짜 상품명) / Rank 1 = 링크 없는 일반 셀
                    //    Rank 0 = 죽은 href(UI 액션). 동일 랭크면 더 긴 텍스트가 승리합니다.
                    //    기존의 무조건 공백 Join 이 'UI 버튼 + 옵션 + 상품명' 뭉침의 직접 원인이었습니다.
                    let raw_line = item_lines_ref[line_idx];
                    let line_rank: i32 = if line_real_href(raw_line).is_some() {
                        2
                    } else if raw_line.contains("href=") {
                        0
                    } else {
                        1
                    };

                    if let Some(existing) = pre_mapped_hints.iter_mut().find(|h: &&mut serde_json::Value| h.get("target_column").and_then(|v| v.as_str()) == Some(owner_field.as_str())) {
                        let prev = existing.get("extracted_value").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let prev_rank = existing.get("line_rank").and_then(|v| v.as_i64()).unwrap_or(1) as i32;

                        if is_multi_value_field(&owner_field) {
                            if !prev.is_empty() && prev != clean_text {
                                existing.as_object_mut().unwrap().insert("extracted_value".to_string(), json!(format!("{} {}", prev, clean_text)));
                            }
                        } else if prev.is_empty()
                            || line_rank > prev_rank
                            || (line_rank == prev_rank && clean_text.chars().count() > prev.chars().count())
                        {
                            existing.as_object_mut().unwrap().insert("extracted_value".to_string(), json!(clean_text));
                            existing.as_object_mut().unwrap().insert("line_rank".to_string(), json!(line_rank));
                            emit_term(&format!("    🥇 [REPRESENTATIVE SWAP] '{}' 대표값 교체: \"{}\" (Rank {}) → \"{}\" (Rank {})", owner_field, prev, prev_rank, clean_text, line_rank));
                        } else {
                            emit_term(&format!("    ⏭️ [SUBORDINATE SKIP] '{}' 는 이미 상위 랭크 대표값(\"{}\")을 확보하여 \"{}\" (Rank {}) 는 병합하지 않습니다.", owner_field, prev, clean_text, line_rank));
                        }
                    } else {
                        pre_mapped_hints.push(json!({
                            "target_column": owner_field.clone(),
                            "extracted_value": clean_text,
                            "line_rank": line_rank
                        }));
                    }
                    emit_term(&format!("    🔍 [FAST-PRE-MAP] Item Line {} mapped to '{}' (Rank {}) via Header cosine", line_idx + 1, owner_field, line_rank));
                }
                

                let pre_mapped_context = if !pre_mapped_hints.is_empty() {
                    // 🌟 line_rank 는 대표값 판정을 위한 내부 메타데이터이므로
                    //    LLM 에게 전달되는 [ALREADY CLAIMED VALUES] 컨텍스트에서는 제거합니다.
                    let clean_hints: Vec<serde_json::Value> = pre_mapped_hints.iter().map(|h| {
                        let mut c = h.clone();
                        if let Some(o) = c.as_object_mut() { o.remove("line_rank"); }
                        c
                    }).collect();
                    serde_json::to_string_pretty(&clean_hints).unwrap_or_default()
                } else {
                    String::new()
                };

                // 🌟 [COSINE ID/LINK RESOLVER]
                //  href 를 (host / path / query) 로 구조 분해해 식별자 후보를 뽑고,
                //  각 후보가 URL 상에서 맡은 '역할 문구'(예: "product register product no", "host name domain name")를
                //  id,link 라벨 뱅크와 코사인 비교해 승자를 고릅니다.
                //  이 방식이라야 도메인 조각 'cafe24'(6자+숫자) 가 식별자로 승격되는 사고가 원천 차단됩니다.
                let idlink_cands = collect_id_link_candidates(&item_lines_ref);
                let mut det_id_link: Option<(String, String)> = None;

                if !idlink_cands.is_empty() && !idlink_label_embs.is_empty() {
                    let role_texts: Vec<String> = idlink_cands.iter().map(|c| c.role_phrase.clone()).collect();
                    let role_embs = model.get_embedding_batch(role_texts.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; role_texts.len()]);

                    let mut best_score = f32::MIN;
                    let mut best_idx: Option<usize> = None;

                    for (ci, cand) in idlink_cands.iter().enumerate() {
                        let emb = &role_embs[ci];
                        if emb.iter().all(|&v| v == 0.0) { continue; }

                        let own = weighted_max_pool_sim(emb, &idlink_label_embs, &idlink_label_weights);
                        let prej = if idlink_prej_embs.is_empty() { 0.0 } else { max_pool_sim(emb, &idlink_prej_embs) };
                        let score = (own - prej) + 0.15 * (cand.prior - 1.0);

                        emit_term(&format!("      🧭 [ID/LINK CANDIDATE] '{}' ← 역할 '{}' | LabelMaxPool: {:.4} | PrejMaxPool: {:.4} | Prior: {:.2}{} | Score: {:+.4}",
                            cand.token, cand.role_phrase, own, prej, cand.prior,
                            if cand.is_host_part { " (host)" } else { "" }, score));

                        if own < 0.30 { continue; }
                        if score <= 0.0 { continue; }
                        if score > best_score { best_score = score; best_idx = Some(ci); }
                    }

                    if let Some(bi) = best_idx {
                        let c = &idlink_cands[bi];
                        det_id_link = Some((c.token.clone(), c.href.clone()));
                        emit_term(&format!("    🔑 [ID/LINK COSINE] 식별자 '{}' 확정 (역할 '{}', Score {:+.4}) → link '{}'", c.token, c.role_phrase, best_score, c.href));
                    }
                }

                if det_id_link.is_none() {
                    det_id_link = resolve_id_link_from_lines(&item_lines_ref);
                    if let Some((fid, flink)) = &det_id_link {
                        emit_term(&format!("    🔑 [ID/LINK FALLBACK] 코사인 게이트를 통과한 후보가 없어 레거시 해석기로 확정: '{}' → '{}'", fid, flink));
                    }
                }

                let mut det_consumed_lines: std::collections::HashSet<usize> = std::collections::HashSet::new();
                if let Some((det_id, det_link)) = &det_id_link {
                    // 부분 문자열 포함이 아니라 '토큰 완전일치'로 선점합니다.
                    // id 가 '18' 처럼 짧아졌을 때 '1800' 같은 무관한 셀까지 삼키는 것을 막습니다.
                    for (l, v) in line_values.iter().enumerate() {
                        if v.is_empty() { continue; }
                        let matched = v.split(|c: char| !c.is_alphanumeric())
                            .any(|tok| !tok.is_empty() && tok.eq_ignore_ascii_case(det_id.as_str()));
                        if matched { det_consumed_lines.insert(l); }
                    }
                    emit_term(&format!("    🔑 [ID/LINK DETERMINISTIC] 식별자 '{}' 가 href '{}' 안에 실제 존재함을 확인. 해당 라인은 다른 컬럼이 선점할 수 없습니다.", det_id, det_link));
                }

                // 🌟 헤더 코사인으로 주인이 확정된 라인은 다른 컬럼의 벡터 후보에서 제외합니다.
                for l in &header_owned_lines {
                    det_consumed_lines.insert(*l);
                }

                // 🌟 [ID SHADOW LINE] 체크박스 label 처럼 식별자 셀의 값을 그대로 복제한 라인이 존재합니다.
                //    (로그의 "1 | 주문번호 26031514155635" → recipient_name 오매칭 원인)
                //    의미 비교가 아니라 "같은 식별자 토큰의 복제본"이라는 구조적 사실로 소비시킵니다.
                if !header_id_tokens.is_empty() {
                    let mut shadow_hits = 0usize;
                    for (l, v) in line_values.iter().enumerate() {
                        if v.is_empty() || det_consumed_lines.contains(&l) { continue; }
                        let hit = v.split(|c: char| !c.is_alphanumeric())
                            .any(|tok| header_id_tokens.iter().any(|t| t.eq_ignore_ascii_case(tok)));
                        if hit {
                            det_consumed_lines.insert(l);
                            shadow_hits += 1;
                        }
                    }
                    if shadow_hits > 0 {
                        emit_term(&format!("    🧹 [ID SHADOW DROP] 식별자 컬럼 값을 복제한 라인 {}개를 다른 컬럼 후보에서 제외했습니다.", shadow_hits));
                    }
                }

                // 🌟 [EXCLUSIVE VECTOR ASSIGNMENT + FORMAT GATE + DOUBLE CENTERING]
                //  1) 형식 게이트 : 유사도를 재기 전에 값의 생김새부터 검증합니다. (핵심)
                //  2) 이중 센터링 : 라인/필드 고유 베이스라인을 제거해 0.5~0.7 에 뭉친 원시값을 변별 가능하게 만듭니다.
                //  3) 배타 배정   : 경쟁 마진이 큰 순서로 1:1 선점합니다.
                //  4) 유일후보 폴백 : 형식 통과 후보가 딱 하나면 마진과 무관하게 확정합니다.
                let (mut vector_assignment, vector_raw_matrix): (Vec<Option<(usize, f32, f32)>>, Vec<Vec<f32>>) = {
                    let line_count = item_lines_ref.len();
                    let field_count = field_phrase_embs.len();
                    let mut raw = vec![vec![-1.0f32; line_count]; field_count];

                    for f in 0..field_count {
                        if field_is_analytic[f] { continue; }
                        let fmt = field_formats[f];
                        for l in 0..line_count {
                            if item_lines_ref[l].trim().is_empty() { continue; }
                            if item_embeddings[l].iter().all(|&v| v == 0.0) { continue; }
                            if det_consumed_lines.contains(&l) { continue; }

                            let value = &line_values[l];
                            let format_ok = match fmt {
                                FieldFormat::Identifier | FieldFormat::Link => value_token_in_url_pool(value, &url_pool),
                                _ => value_matches_format(fmt, value),
                            };
                            if !format_ok { continue; }

                            raw[f][l] = weighted_max_pool_sim(
                                &item_embeddings[l],
                                &field_phrase_embs[f],
                                &field_phrase_weights[f],
                            );
                        }
                    }

                    let centered = double_center_matrix(&raw);
                    let mut assign = exclusive_assign(&centered, 0.0, 0.005);

                    let mut claimed = vec![false; line_count];
                    for a in assign.iter() {
                        if let Some((l, _, _)) = a { claimed[*l] = true; }
                    }
                    for f in 0..field_count {
                        if assign[f].is_some() { continue; }
                        if field_is_analytic[f] { continue; }
                        let cands: Vec<usize> = (0..line_count)
                            .filter(|&l| raw[f][l] >= 0.0 && !claimed[l])
                            .collect();
                        if cands.len() == 1 {
                            let l = cands[0];
                            assign[f] = Some((l, centered[f][l], 0.0));
                            claimed[l] = true;
                        }
                    }

                    (assign, raw)
                };

                // 🌟 [HEADER OVERRIDE] enum 계열은 헤더 코사인이 확정한 컬럼을 벡터 배정보다 우선합니다.
                //    로그의 'status ← "수량 | 1"', 'payment_method ← "총주문액 | 615600"' 이
                //    여기서 각각 '주문상태 | 취소', '결제방법 | 무통장' 으로 교체됩니다.
                for (f_i, (fname, _, _, _)) in fields.iter().enumerate() {
                    if let Some(l) = header_forced_assign.get(fname) {
                        let raw = vector_raw_matrix.get(f_i).and_then(|r| r.get(*l)).copied().unwrap_or(0.0).max(0.0);
                        vector_assignment[f_i] = Some((*l, raw, 0.0));
                        emit_term(&format!("    🧷 [HEADER OVERRIDE] '{}' 의 벡터 배정을 헤더 코사인 확정 컬럼(Line {})으로 교체했습니다.", fname, *l + 1));
                    }
                }

                for (f_i, (fname, _, _, _)) in fields.iter().enumerate() {
                    match vector_assignment[f_i] {
                        Some((l, contrast, margin)) => {
                            let shown = if line_enriched_texts[l].is_empty() {
                                item_lines_ref[l].trim()
                            } else {
                                line_enriched_texts[l].as_str()
                            };
                            emit_term(&format!("    🔗 [EXCLUSIVE ASSIGN] '{}' ({:?}) ← Line {} | RawSim: {:.4} | Contrast: {:+.4} | Margin: {:+.4} | \"{}\"", fname, field_formats[f_i], l + 1, vector_raw_matrix[f_i][l], contrast, margin, shown));
                        },
                        None => {
                            if !field_is_analytic[f_i] {
                                let cand_cnt = vector_raw_matrix[f_i].iter().filter(|&&v| v >= 0.0).count();
                                emit_term(&format!("    ⚪ [UNASSIGNED] '{}' ({:?}) | 형식 통과 후보 {}개 | 벡터 힌트 미주입", fname, field_formats[f_i], cand_cnt));
                            }
                        }
                    }
                }


                for (f_idx, (field_name, field_desc, bias_target, prejudice_target)) in fields.clone().into_iter().enumerate() {
                    

                    let keys: Vec<&str> = field_name.split(',').map(|s| s.trim()).collect();
                    let mut bypassed_values: Vec<(String, String)> = Vec::new();
                    for k in &keys {
                        for hint in &pre_mapped_hints {
                            if let Some(t_col) = hint.get("target_column").and_then(|v| v.as_str()) {
                                if t_col == *k {
                                    if let Some(e_val) = hint.get("extracted_value").and_then(|v| v.as_str()) {
                                        let clean_e_val = e_val.trim();
                                        if !clean_e_val.is_empty() {
                                            if let Some(existing) = bypassed_values.iter_mut().find(|(key, _)| key == *k) {
                                                if !existing.1.contains(clean_e_val) {
                                                    existing.1.push_str(" ");
                                                    existing.1.push_str(clean_e_val);
                                                }
                                            } else {
                                                bypassed_values.push((k.to_string(), clean_e_val.to_string()));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if !bypassed_values.is_empty() {
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

                        let shareable_field = ["name", "insight", "status", "payment_method", "date", "_at", "currency", "goods", "title"]
                            .iter().any(|f| field_name.contains(f));

                        let mut extracted_results = Vec::new();
                        for (k, val_str) in bypassed_values {
                            item_val.as_object_mut().unwrap().insert(k.clone(), json!(val_str));
                            extracted_results.push(format!("\"{}\": \"{}\"", k, val_str));
                            
                            if !shareable_field && val_str.len() >= 5 && val_str != "null" && val_str != "true" && val_str != "false" {
                                if !global_ignore_list.contains(&val_str) {
                                    global_ignore_list.push(val_str.clone());
                                    global_ignore_list.push(format!(" {}", val_str));
                                    global_ignore_list.push(val_str.to_lowercase());
                                }
                            }
                        }
                        emit_term(&format!("    ⚡ [PRE-MAP BYPASS] Successfully mapped without LLM: {}", extracted_results.join(", ")));
                        continue;
                    }

                    let field_format = field_formats[f_idx];

                    // 🌟 [ID/LINK BYPASS] href 안에 실제로 존재하는 토큰만 id 로, 그 href 를 link 로 확정합니다.
                    //    LLM 을 아예 호출하지 않으므로 id 와 link 가 어긋날 물리적 여지가 없습니다.
                    if is_id_link_field(&field_name) {
                        if let Some((det_id, det_link)) = det_id_link.clone() {
                            item_val.as_object_mut().unwrap().insert("id".to_string(), json!(det_id.clone()));
                            item_val.as_object_mut().unwrap().insert("link".to_string(), json!(det_link.clone()));
                            if !global_ignore_list.contains(&det_id) {
                                global_ignore_list.push(det_id.clone());
                                global_ignore_list.push(format!(" {}", det_id));
                                global_ignore_list.push(det_id.to_lowercase());
                            }
                            // 🌟 [URL PATTERN DISCOVERY] 성공한 id/link 쌍에서 URL 구조 패턴과 식별자 '생김새'를 함께 학습합니다.
                            //    extract_url_pattern 은 호스트(도메인) 구간 매칭을 거부하므로
                            //    prefix='https://breakbot.' / suffix='.com/...' 같은 도메인 변조 패턴이 만들어질 수 없습니다.
                            let id_shape = id_shape_signature(&det_id);
                            if !confirmed_id_shapes.contains(&id_shape) { confirmed_id_shapes.push(id_shape); }

                            if discovered_url_pattern.is_none() {
                                if let Some((prefix, suffix)) = extract_url_pattern(&det_id, &det_link) {
                                    discovered_url_pattern = Some((prefix.clone(), suffix.clone()));
                                    pattern_reference_link = Some(det_link.clone());
                                    emit_term(&format!("    📐 [URL PATTERN DISCOVERED] prefix: '{}' | suffix: '{}' | IdShape: (길이 {}, 숫자전용 {}) → 이후 실패 아이템에 소급 적용 가능", prefix, suffix, id_shape.0, id_shape.1));
                                } else {
                                    emit_term(&format!("    🚫 [URL PATTERN REJECTED] 식별자 '{}' 가 path/query 구간에서 발견되지 않아 패턴화를 거부했습니다. (link: {})", det_id, det_link));
                                }
                            }
                            emit_term(&format!("    ⚡ [ID/LINK BYPASS] LLM 없이 확정: \"id\": \"{}\", \"link\": \"{}\"", det_id, det_link));
                            continue;
                        }
                    }

                    // 🌟 아이템 라인은 위에서 이미 배타적으로 확정된 배정 결과만 사용합니다. (필드별 독립 argmax 폐기)
                    let (best_item_idx, best_item_contrast, best_item_margin, has_vector_match) = match vector_assignment[f_idx] {
                        Some((l, contrast, margin)) => (l, contrast, margin, true),
                        None => (0usize, 0.0f32, 0.0f32, false),
                    };
                    let best_item_raw = if has_vector_match { vector_raw_matrix[f_idx][best_item_idx] } else { 0.0f32 };

                    // 🌟 [STRICT FORMAT SKIP] 날짜·운송장·숫자·식별자·링크처럼 생김새가 확정적인 필드는
                    //    형식 게이트를 통과한 후보 셀이 이 아이템에 아예 없으면 LLM 을 호출하지 않고 비워둡니다.
                    //    "배송번호가 없으면 빈값이 맞다" / "잘못 등록될 바엔 추출하지 않는다" 를 그대로 구현합니다.
                    let strict_format_field = matches!(
                        field_format,
                        FieldFormat::Date | FieldFormat::TrackingCode | FieldFormat::Numeric | FieldFormat::Identifier | FieldFormat::Link
                    );
                    if !field_is_analytic[f_idx] && strict_format_field && !has_vector_match {
                        emit_term(&format!("    ⛔ [FORMAT SKIP] Field: '{}' ({:?}) | 형식에 맞는 후보 셀이 이 아이템에 존재하지 않습니다. LLM 호출 없이 빈 값으로 확정.", field_name, field_format));
                        continue;
                    }

                    let (_bias_emb, _prej_emb, dynamic_prej_str) = &field_embeddings[f_idx];

                    // 🌟 헤더(thead) 매칭도 센트로이드 대신 구 단위 Max-Pool + 경쟁 필드 대비 마진으로 계산합니다.
                    let mut best_thead_idx = 0usize;
                    let mut best_thead_score = -1.0f32;
                    let mut best_thead_own = 0.0f32;
                    for (i, emb) in thead_embeddings.iter().enumerate() {
                        if thead_lines_ref[i].trim().is_empty() { continue; }
                        if emb.iter().all(|&v| v == 0.0) { continue; }

                        let own = weighted_max_pool_sim(emb, &field_phrase_embs[f_idx], &field_phrase_weights[f_idx]);
                        let mut rival = 0.0f32;
                        for other_idx in 0..field_phrase_embs.len() {
                            if other_idx == f_idx { continue; }
                            let s = weighted_max_pool_sim(emb, &field_phrase_embs[other_idx], &field_phrase_weights[other_idx]);
                            if s > rival { rival = s; }
                        }
                        let final_score = own - rival;

                        if final_score > best_thead_score {
                            best_thead_score = final_score;
                            best_thead_idx = i;
                            best_thead_own = own;
                        }
                    }
                    let _ = best_thead_idx;

                    let targeted_pug = filtered_full_item_pug.clone();

                    if field_is_analytic[f_idx] {
                        emit_term(&format!("    🧠 [SYNTHESIS FIELD] Field: '{}' | 단일 라인 환원 불가 → 전체 아이템 컨텍스트 요약 모드 (HeaderOwn: {:.4})", field_name, best_thead_own));
                    } else if has_vector_match {
                        emit_term(&format!("    🎯 [MATCHED CONTEXT] Field: '{}' ({:?}) | Line: {} | RawSim: {:.4} | Contrast: {:+.4} | Margin: {:+.4}", field_name, field_format, best_item_idx + 1, best_item_raw, best_item_contrast, best_item_margin));
                    } else {
                        emit_term(&format!("    ⚠️ [NO CONFIDENT MATCH] Field: '{}' ({:?}) | 벡터 힌트 없이 전체 구조만 전달 (HeaderContrast: {:+.4})", field_name, field_format, best_thead_score));
                    }
                    
                    let mut final_context_str = format!("[JSON CONTEXT]\n{}", targeted_pug);

                    if field_name.contains("link") || field_name.contains("id") {
                        let mut link_cands: Vec<String> = Vec::new();
                        if let Ok(href_re) = regex::Regex::new(r#"href=["']([^"']+)["']"#) {
                            for line in &item_lines_ref {
                                for cap in href_re.captures_iter(line) {
                                    if let Some(m) = cap.get(1) {
                                        let v = m.as_str().trim().to_string();
                                        if !v.is_empty() && !link_cands.contains(&v) { link_cands.push(v); }
                                    }
                                }
                            }
                        }
                        if link_cands.is_empty() {
                            final_context_str.push_str("\n\n[LINK CANDIDATES]\n(none)\nThere is NO link in this item. You MUST return null for the link key.");
                        } else {
                            final_context_str.push_str(&format!("\n\n[LINK CANDIDATES]\n{}\nThe link value MUST be copied EXACTLY from this list. Never invent a URL.", link_cands.join("\n")));
                        }
                    }

                    if field_name.contains("date") || field_name.contains("_at") {
                        let mut date_cands: Vec<String> = Vec::new();
                        if let Ok(date_re) = regex::Regex::new(r"\d{2,4}[-/\.]\d{1,2}[-/\.]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?") {
                            for line in &item_lines_ref {
                                for m in date_re.find_iter(line) {
                                    let v = m.as_str().trim().to_string();
                                    if !date_cands.contains(&v) { date_cands.push(v); }
                                }
                            }
                        }
                        if date_cands.is_empty() {
                            final_context_str.push_str("\n\n[DATE CANDIDATES]\n(none)\nThere is NO date literal in this item. You MUST return null.");
                        } else {
                            final_context_str.push_str(&format!("\n\n[DATE CANDIDATES]\n{}\nThe answer MUST be one of these literals, copied character by character, or null.", date_cands.join("\n")));
                        }
                    }

                    if field_is_analytic[f_idx] {
                        // 🌟 합성 필드는 어떤 단일 셀도 정답이 아닙니다. 셀 복사 자체를 금지시킵니다.
                        final_context_str.push_str("\n\n[SYNTHESIS FIELD NOTICE]\nThis field is NOT a value to copy. Read the WHOLE [JSON CONTEXT] above and write ONE short sentence that summarizes it. Never return a single cell value such as a bare number, a status word, a person name, or a branch name. If [JSON CONTEXT] is empty, return null.");
                    } else if has_vector_match {
                        let matched_line = if line_enriched_texts[best_item_idx].is_empty() {
                            item_lines_ref[best_item_idx].trim()
                        } else {
                            line_enriched_texts[best_item_idx].as_str()
                        };
                        final_context_str.push_str(&format!("\n\n[VECTOR MATCH RESULT]\nThe format gate and the embedding model EXCLUSIVELY assigned this field to the single line below (RawSim {:.4}, Contrast {:+.4}, Margin {:+.4}). No other column may use this line.\nThe part BEFORE '|' is the column LABEL, the part AFTER '|' is the VALUE. Copy ONLY the value part, character for character. Do NOT copy the label. If that value does not fit the schema, return null.\n\"{}\"", best_item_raw, best_item_contrast, best_item_margin, matched_line));
                        if !pre_mapped_context.is_empty() {
                            final_context_str.push_str(&format!("\n\n[ALREADY CLAIMED VALUES]\nThese values are already assigned to OTHER columns. You MUST NOT return any of them for this field:\n{}", pre_mapped_context));
                        }
                    } else if !pre_mapped_context.is_empty() {
                        // 🌟 기존에는 이 블록이 최우선이라 "다른 컬럼의 값 목록"이 정답처럼 제시되어 오염되었습니다.
                        //    이제는 '이미 선점된 값 = 금지 목록'이라는 음(-)의 제약으로 뒤집어 전달합니다.
                        final_context_str.push_str(&format!("\n\n[ALREADY CLAIMED VALUES]\nThese values are already assigned to OTHER columns. You MUST NOT return any of them for this field. If nothing else in [JSON CONTEXT] fits this field, return null:\n{}", pre_mapped_context));
                    }

                    let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                        content: final_context_str,
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

                    let mut metadata_str = String::new();
                    let mut target_data_str = String::new();

                    for line in targeted_pug.lines() {
                        if let Some(idx) = line.find('|') {
                            metadata_str.push_str(line[..idx].trim());
                            metadata_str.push_str("\n");
                            target_data_str.push_str(line[idx + 1..].trim());
                            target_data_str.push_str("\n");
                        } else {
                            target_data_str.push_str(line.trim());
                            target_data_str.push_str("\n");
                        }
                    }

                    let metadata_str = metadata_str.trim();
                    let target_data_str = target_data_str.trim();

                    let task_question = if field_name.contains("status") {
                        parsing::extract_status_intent_legacy_prompt(&targeted_pug, &page_type, &bias_target)
                    } else if field_is_analytic[f_idx] {
                        // 🌟 "리터럴 복사" 지시를 받는 단일 필드 프롬프트로는 합성 필드를 절대 만들 수 없습니다.
                        parsing::extract_synthesis_field_prompt(&page_type, &field_name, &field_desc, &doc_lang, target_data_str)
                    } else {
                        parsing::extract_single_field_prompt(&page_type, &field_name, &field_desc, language, metadata_str, target_data_str)
                    };
                    
                    let mut ignore_list: Vec<String> = global_ignore_list.clone();
                    let mut miss_counter = 0;
                    
                    loop {
                        if cancellation_token.load(Ordering::Relaxed) { break; }

                        let q3_gen = model.qwen3_generator.clone();
                        let cancel_clone = cancellation_token.clone();
                        let sys_msg = system_message.clone();
                        
                        let field_name_clone = field_name.clone();
                        let bias_target_for_closure = bias_target.clone();
                        

                        let prejudice_target_for_closure = dynamic_prej_str.clone();
                        
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

                                // 🌟 [MARKUP STRIP] 리스트 경로도 동일하게 태그 접두어를 벗겨냅니다.
                                if let Some(obj) = parsed_val.as_object_mut() {
                                    let ks: Vec<String> = obj.keys().cloned().collect();
                                    for k in ks {
                                        let cleaned = match obj.get(&k) {
                                            Some(serde_json::Value::String(s)) => Some(strip_markup_prefix(s)),
                                            _ => None,
                                        };
                                        if let Some(c) = cleaned {
                                            obj.insert(k, json!(c));
                                        }
                                    }
                                }

                                let mut requires_retry = false;
                                let mut extracted_values_for_retry = Vec::new();
                                
                                let keys: Vec<&str> = field_name_clone.split(',').map(|s| s.trim()).collect();
                                let mut found_valid_value = false;

                                let skip_pug_match_fields = ["status", "payment_method", "payment_origin", "condition", "currency"];
                                // 🌟 insight/summary/analysis 계열은 '합성 문장'이라 PUG 원문에 리터럴로 존재할 수 없습니다.
                                //    문자열 포함 검사를 그대로 적용하면 정상 답변도 100% 환각 판정 → 3회 재시도 후 폐기됩니다.
                                let synthesis_fields = ["insight", "summary", "analysis"];
                                let field_name_lower = field_name_clone.to_lowercase();
                                let is_synthesis_field = synthesis_fields.iter().any(|&f| field_name_lower.contains(f));
                                let is_enum_field = is_synthesis_field || skip_pug_match_fields.iter().any(|&f| field_name_clone.contains(f));

                                let is_placeholder_str = |s: &str| -> bool {
                                    let t = s.trim();
                                    if t.is_empty() { return true; }
                                    let lower = t.to_lowercase();
                                    if ["...", "null", "string", "number", "boolean", "n/a", "none", "undefined"].contains(&lower.as_str()) { return true; }
                                    let compact: String = lower.chars().filter(|c| c.is_alphanumeric()).collect();
                                    if ["yyyymmddthhmmss", "yyyymmddhhmmss", "yyyymmdd", "hhmmss"].contains(&compact.as_str()) { return true; }
                                    let ymd_only = !lower.is_empty() && lower.chars().all(|c| "ymdhms-t:./ ".contains(c));
                                    if ymd_only && lower.chars().any(|c| c == 'y' || c == 'm' || c == 'd') { return true; }
                                    false
                                };

                                for k in &keys {
                                    if let Some(val) = parsed_val.get(*k) {
                                        let is_empty_val = match val {
                                            serde_json::Value::Null => true,
                                            serde_json::Value::String(s) => is_placeholder_str(s),
                                            serde_json::Value::Array(a) => a.is_empty(),
                                            serde_json::Value::Object(o) => o.is_empty(),
                                            _ => false,
                                        };

                                        if !is_empty_val {
                                            let extracted_str = if val.is_string() {
                                                val.as_str().unwrap_or("").trim().to_string()
                                            } else if val.is_number() {
                                                val.to_string()
                                            } else {
                                                String::new()
                                            };

                                            // 🌟 [POST FORMAT VALIDATION] 벡터가 정답 라인을 정확히 짚어줘도
                                            //    0.6B 모델은 "registration_date: 615600" 처럼 엉뚱한 셀을 복사합니다.
                                            //    형식이 확정적인 키는 반환값의 생김새를 다시 검증해 폐기합니다.
                                            //    🌟 Enum / Identifier 도 포함하여 마크업 잔재("tr","td")를 차단합니다.
                                            let key_fmt = detect_field_format(k);
                                            let strict_post = matches!(
                                                key_fmt,
                                                FieldFormat::Date | FieldFormat::TrackingCode | FieldFormat::Text
                                                    | FieldFormat::Numeric | FieldFormat::Enum | FieldFormat::Identifier
                                            );
                                            if strict_post && !extracted_str.is_empty() && !value_matches_format(key_fmt, &extracted_str) {
                                                emit_term(&format!("    🚫 [FORMAT REJECT] '{}' ({:?}) 에 형식 불일치 값 '{}' 반환. 폐기 후 재시도합니다.", k, key_fmt, extracted_str));
                                                requires_retry = true;
                                                extracted_values_for_retry.push(extracted_str.clone());
                                                continue;
                                            }

                                            found_valid_value = true;

                                            if !extracted_str.is_empty() && extracted_str != "..." && extracted_str != "null" {
                                                extracted_values_for_retry.push(extracted_str.clone());
                                                
                                                if !is_enum_field {
                                                    let is_iso_date = extracted_str.contains('T') && extracted_str.len() >= 19;
                                                    let is_url = extracted_str.starts_with("http") || extracted_str.starts_with('/');
                                                    let is_boolean_str = extracted_str == "true" || extracted_str == "false";
                                                    
                                                    if !is_iso_date && !is_url && !is_boolean_str {
                                                        let mut is_matched = doc_title.contains(&extracted_str);
                                                        
                                                        if !is_matched {
                                                            let extracted_lower = extracted_str.to_lowercase();
                                                            let digits_only: String = extracted_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                            
                                                            for ctx_val in &json_contexts {
                                                                if let Some(target_val_str) = ctx_val.get("value").and_then(|v| v.as_str()) {
                                                                    let target_lower = target_val_str.to_lowercase();
                                                                    
                                                                    if target_lower.contains(&extracted_lower) {
                                                                        if digits_only.len() > 0 && digits_only.len() < 3 && extracted_str.len() == digits_only.len() {
                                                                            let tokens: Vec<&str> = target_lower.split(|c: char| !c.is_alphanumeric()).collect();
                                                                            if tokens.contains(&extracted_lower.as_str()) {
                                                                                is_matched = true;
                                                                                break;
                                                                            }
                                                                        } else {
                                                                            is_matched = true;
                                                                            break;
                                                                        }
                                                                    }
                                                                    
                                                                    if !is_matched && digits_only.len() >= 3 {
                                                                        let target_digits: String = target_val_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                                        if target_digits.contains(&digits_only) {
                                                                            is_matched = true;
                                                                            break;
                                                                        }
                                                                    }
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

                                let shareable_field = ["name", "insight", "status", "payment_method", "date", "_at", "currency", "goods", "title"]
                                    .iter().any(|f| field_name_clone.contains(f));

                                let mut extracted_results = Vec::new();
                                for k in &keys {
                                    if let Some(val) = parsed_val.get(*k) {
                                        item_val.as_object_mut().unwrap().insert(k.to_string(), val.clone());
                                        extracted_results.push(format!("\"{}\": {}", k, val));
                                        
                                        let val_str = if val.is_string() { val.as_str().unwrap().trim().to_string() }
                                                      else if val.is_number() { val.to_string() }
                                                      else { String::new() };
                                        
                                        if !shareable_field && val_str.len() >= 5 && val_str != "null" && val_str != "true" && val_str != "false" {
                                            if !global_ignore_list.contains(&val_str) {
                                                global_ignore_list.push(val_str.clone());
                                                global_ignore_list.push(format!(" {}", val_str));
                                                global_ignore_list.push(val_str.to_lowercase());
                                            }
                                        }
                                    }
                                }
                                


                                for ck in ["has_header", "has_footer", "language"] {
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
                    }
                }


                let mut temp_id = item_val.get("id").and_then(|v| if v.is_string() { v.as_str().map(|s| s.to_string()) } else { Some(v.to_string()) }).unwrap_or_default();
                let mut temp_code = item_val.get("code").and_then(|v| if v.is_string() { v.as_str().map(|s| s.to_string()) } else { Some(v.to_string()) }).unwrap_or_default();
                

                if !temp_id.is_empty() || !temp_code.is_empty() {
                    let mut url_pool = String::new();
                    if let Ok(href_re) = regex::Regex::new(r#"href=["']([^"']+)["']"#) {
                        for line in &item_lines_ref {
                            for cap in href_re.captures_iter(line) {
                                if let Some(m) = cap.get(1) {
                                    url_pool.push_str(&m.as_str().to_lowercase());
                                    url_pool.push_str(" ");
                                }
                            }
                        }
                    }
                    
                    let id_in_url = !temp_id.is_empty() && url_pool.contains(&temp_id.to_lowercase());
                    let code_in_url = !temp_code.is_empty() && url_pool.contains(&temp_code.to_lowercase());

                    if !id_in_url && code_in_url {
                        let swap = temp_id.clone();
                        temp_id = temp_code.clone();
                        temp_code = swap;
                        emit_term("  🔄 [DEV-LOGIC] Swapped 'id' and 'code' based on URL presence in PUG.");
                    } else if !temp_id.is_empty() && !id_in_url {
                        if temp_code.is_empty() {
                            temp_code = temp_id.clone();
                        }
                        temp_id = String::new();
                        emit_term("  🔄 [DEV-LOGIC] Moved 'id' to 'code' because it was NOT found in any URL link.");
                    }
                }

                if !temp_id.is_empty() {
                    let extracted = if let Some(idx) = temp_id.rfind('=') {
                        &temp_id[idx + 1..]
                    } else {
                        &temp_id
                    };
                    let clean_str = extracted.replace("-", "").replace("_", "").replace(".", "").replace(",", "");
                    if !clean_str.is_empty() {
                        item_val.as_object_mut().unwrap().insert("id".to_string(), json!(clean_str.trim()));
                    } else {
                        item_val.as_object_mut().unwrap().remove("id");
                    }
                } else {
                    item_val.as_object_mut().unwrap().remove("id");
                }

                if !temp_code.is_empty() {
                    item_val.as_object_mut().unwrap().insert("code".to_string(), json!(temp_code.trim()));
                } else {
                    item_val.as_object_mut().unwrap().remove("code");
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
                    // 🌟 [RAW LINE ARCHIVE] 소급 복구 시 원시 라인에서 href 를 재탐색하기 위해 보관합니다.
                    all_item_raw_lines.push(item_lines.clone());
                    // 🌟 [LABELED LINE ARCHIVE] 소급 복구의 코사인 채점 대상은 '값'이 아니라 '그 값이 달린 컬럼 라벨'입니다.
                    //    (예: "상품코드 | P000000P" → 라벨 '상품코드' 를 id,link 라벨 뱅크와 코사인 비교)
                    let labeled_snapshot: Vec<String> = (0..item_lines.len()).map(|li| {
                        if !line_enriched_texts[li].is_empty() {
                            line_enriched_texts[li].clone()
                        } else {
                            item_lines[li].clone()
                        }
                    }).collect();
                    all_item_labeled_lines.push(labeled_snapshot);
                }
                
                crate::models::qwen::generate::wait_for_global_io().await;
            }

            // 🌟 [ID/LINK RETRY PASS - COSINE]
            //    1단계: 실패 아이템의 원시 라인에 href 가 남아 있으면, 그 href 후보를 다시 코사인으로 채점해 직접 복구합니다.
            //           (로그의 아이템 4·5·6·8·9·10 은 '바로구매 URL' 셀만 비어 있을 뿐
            //            상품명 셀의 a[href=".../ProductRegister?product_no=15"] 가 그대로 살아 있습니다.)
            //    2단계: href 자체가 없을 때만, '값'이 아니라 '그 값이 달린 컬럼 라벨'을
            //           id,link 라벨 뱅크와 코사인 비교해 가장 식별자다운 컬럼의 값만 채택하고 URL 패턴에 대입합니다.
            //    3단계: 학습된 식별자 생김새(자릿수/숫자전용 여부)와 다른 토큰, 그리고 호스트가 달라지는 재구성 링크는
            //           대입 자체를 거부합니다. 잘못된 링크를 만드는 것보다 빈 값이 안전합니다.
            {
                let total_extracted_items = all_extracted_items.len();
                let mut retry_count = 0usize;
                let mut reject_count = 0usize;

                for item_idx in 0..total_extracted_items {
                    let (has_id, has_link) = {
                        let iv = &all_extracted_items[item_idx];
                        (
                            iv.get("id").and_then(|v| v.as_str()).map_or(false, |s| !s.is_empty()),
                            iv.get("link").and_then(|v| v.as_str()).map_or(false, |s| !s.is_empty()),
                        )
                    };
                    if has_id && has_link { continue; }

                    let mut recovered: Option<(String, String, String)> = None; // (id, link, 사유)

                    // --- 1단계 : 원시 라인에 남아 있는 href 를 코사인으로 재채점 ---
                    if let Some(raw_lines) = all_item_raw_lines.get(item_idx) {
                        let raw_refs: Vec<&str> = raw_lines.iter().map(|s| s.as_str()).collect();
                        let cands = collect_id_link_candidates(&raw_refs);

                        if !cands.is_empty() && !idlink_label_embs.is_empty() {
                            let role_texts: Vec<String> = cands.iter().map(|c| c.role_phrase.clone()).collect();
                            let role_embs = model.get_embedding_batch(role_texts.clone()).await
                                .unwrap_or_else(|_| vec![vec![0.0; 384]; role_texts.len()]);

                            let mut best = f32::MIN;
                            let mut best_i: Option<usize> = None;
                            for (ci, c) in cands.iter().enumerate() {
                                let emb = &role_embs[ci];
                                if emb.iter().all(|&v| v == 0.0) { continue; }
                                let own = weighted_max_pool_sim(emb, &idlink_label_embs, &idlink_label_weights);
                                let prej = if idlink_prej_embs.is_empty() { 0.0 } else { max_pool_sim(emb, &idlink_prej_embs) };
                                let score = (own - prej) + 0.15 * (c.prior - 1.0);
                                emit_term(&format!("      🧭 [RETRY HREF CANDIDATE] Item {}/{}: '{}' ← 역할 '{}' | LabelMaxPool: {:.4} | PrejMaxPool: {:.4} | Score: {:+.4}",
                                    item_idx + 1, total_extracted_items, c.token, c.role_phrase, own, prej, score));
                                if own < 0.30 { continue; }
                                if score <= 0.0 { continue; }
                                if score > best { best = score; best_i = Some(ci); }
                            }

                            if let Some(bi) = best_i {
                                let c = &cands[bi];
                                recovered = Some((
                                    c.token.clone(),
                                    c.href.clone(),
                                    format!("href 코사인 재채점 (역할 '{}', Score {:+.4})", c.role_phrase, best),
                                ));
                            }
                        }
                    }

                    // --- 2단계 : href 가 아예 없을 때만 컬럼 라벨 코사인 + URL 패턴 대입 ---
                    if recovered.is_none() {
                        if let Some((ref pat_prefix, ref pat_suffix)) = discovered_url_pattern {
                            let labeled = all_item_labeled_lines.get(item_idx).cloned().unwrap_or_default();
                            let cands = collect_labeled_token_candidates(&labeled);

                            let mut chosen: Option<(String, String, f32)> = None; // (token, label, score)

                            if !cands.is_empty() && !idlink_label_embs.is_empty() {
                                let label_texts: Vec<String> = cands.iter().map(|c| c.label_phrase.clone()).collect();
                                let label_embs = model.get_embedding_batch(label_texts.clone()).await
                                    .unwrap_or_else(|_| vec![vec![0.0; 384]; label_texts.len()]);

                                for (ci, c) in cands.iter().enumerate() {
                                    if !id_shape_allowed(&c.token, &confirmed_id_shapes) {
                                        reject_count += 1;
                                        emit_term(&format!("      🚫 [SHAPE REJECT] Item {}/{}: 후보 '{}' 는 학습된 식별자 생김새와 달라 URL 대입을 거부했습니다.", item_idx + 1, total_extracted_items, c.token));
                                        continue;
                                    }

                                    let emb = &label_embs[ci];
                                    if emb.iter().all(|&v| v == 0.0) { continue; }
                                    let own = weighted_max_pool_sim(emb, &idlink_label_embs, &idlink_label_weights);
                                    let prej = if idlink_prej_embs.is_empty() { 0.0 } else { max_pool_sim(emb, &idlink_prej_embs) };
                                    let score = own - prej;

                                    emit_term(&format!("      🧭 [RECOVERY CANDIDATE] Item {}/{}: '{}' ← 라벨 '{}' | LabelMaxPool: {:.4} | PrejMaxPool: {:.4} | Score: {:+.4}",
                                        item_idx + 1, total_extracted_items, c.token, c.label_phrase, own, prej, score));

                                    if own < 0.40 { continue; }
                                    if score <= 0.05 { continue; }
                                    let better = chosen.as_ref().map(|(_, _, s)| score > *s).unwrap_or(true);
                                    if better { chosen = Some((c.token.clone(), c.label_phrase.clone(), score)); }
                                }
                            }

                            // 코사인 게이트를 전부 통과하지 못했을 때만 레거시 토큰 탐색을 시도하되,
                            // 생김새 게이트는 그대로 강제해 'P000000P' 같은 이형 토큰의 대입을 막습니다.
                            if chosen.is_none() {
                                if let Some(raw_lines) = all_item_raw_lines.get(item_idx) {
                                    if let Some(tok) = find_identifier_token_in_lines(raw_lines) {
                                        if id_shape_allowed(&tok, &confirmed_id_shapes) {
                                            chosen = Some((tok, "legacy token scan".to_string(), 0.0));
                                        } else {
                                            reject_count += 1;
                                            emit_term(&format!("      🚫 [SHAPE REJECT] Item {}/{}: 레거시 토큰 '{}' 도 학습된 식별자 생김새와 달라 폐기했습니다.", item_idx + 1, total_extracted_items, tok));
                                        }
                                    }
                                }
                            }

                            if let Some((tok, label, score)) = chosen {
                                let link = apply_url_pattern(pat_prefix, pat_suffix, &tok);
                                let host_ok = pattern_reference_link.as_ref()
                                    .map(|r| same_host(r, &link))
                                    .unwrap_or(true);
                                if host_ok {
                                    recovered = Some((tok, link, format!("라벨 코사인 (라벨 '{}', Score {:+.4})", label, score)));
                                } else {
                                    reject_count += 1;
                                    emit_term(&format!("      🚫 [HOST REJECT] Item {}/{}: 재구성 링크 '{}' 의 호스트가 기준 링크와 달라 폐기했습니다.", item_idx + 1, total_extracted_items, link));
                                }
                            }
                        }
                    }

                    // --- 3단계 : 모든 게이트를 통과한 값만 주입 ---
                    if let Some((found_id, constructed_link, reason)) = recovered {
                        if let Some(obj) = all_extracted_items[item_idx].as_object_mut() {
                            obj.insert("id".to_string(), json!(found_id.clone()));
                            obj.insert("link".to_string(), json!(constructed_link.clone()));
                        }
                        retry_count += 1;
                        emit_term(&format!("  🔄 [ID/LINK RETRY] Item {}/{}: {} → \"id\": \"{}\", \"link\": \"{}\"", item_idx + 1, total_extracted_items, reason, found_id, constructed_link));
                    } else {
                        emit_term(&format!("  ⚪ [ID/LINK RETRY SKIP] Item {}/{}: 코사인 게이트를 통과한 식별자 후보가 없어 id/link 를 비워 둡니다. (잘못된 링크보다 빈 값이 안전합니다)", item_idx + 1, total_extracted_items));
                    }
                }

                if retry_count > 0 || reject_count > 0 {
                    emit_term(&format!("  🔄 [ID/LINK RETRY SUMMARY] 복구 {}개 | 생김새·호스트 게이트 거부 {}개.", retry_count, reject_count));
                }
            }
        }

        extracted_data = json!({ "items": all_extracted_items, "type": page_type, "detail": false });

    } else {

        println!("[Scheduler] Starting DISK BRIDGE RELAY for Details");
        
        let content_pug = {
            let clean_content = &clean_html_content;
            let full_pug = parsing::convert_to_clean_pug(clean_content, PugMode::DetailMode, Some(&url));
            model.truncate_pug_context(&full_pug, true, 2000, None).await
        };

        if !content_pug.trim().is_empty() {

            model.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancellation_token.clone()), false, Some("inference".to_string())).await?;

            if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }


            let (_, layout_prejudice) = crate::parsing::get_layout_bias(&page_type, &doc_lang);
            let layout_prej_emb = model.get_embedding(layout_prejudice.clone()).await.unwrap_or(vec![0.0; 384]);


            let fields = parsing::get_detail_schema_fields(&page_type, &url, &doc_lang);
            let total_fields = fields.len();

            let payload = json!({ "task_id": task.id, "category": "AI Inference", "summary": format!("Extracting {} detail fields sequentially...", total_fields), "spinner": "⠋" });
            let _ = app_handle.emit("extraction-progress", &payload);
            emit_term(&format!("[STAGE-3] Extracting {} detailed fields individually...", total_fields));


            let mut pug_lines: Vec<String> = content_pug.lines().map(|s| s.to_string()).collect();

            // 🌟 [STRUCTURAL LABEL-VALUE PAIRS] th[scope="row"] → td, thead th[scope="col"] → tbody td,
            //    input[placeholder] → value 를 DOM 구조 그대로 결합합니다.
            //    기존 "직전 라벨 라인" 휴리스틱은 thead 의 th 끼리도 라벨-값으로 오인해
            //    goods="주문상태", sender_name="수량", bank="계좌번호" 를 만들었으므로 완전히 폐기합니다.
            let detail_pairs: Vec<DetailPair> = {
                let refs: Vec<&str> = pug_lines.iter().map(|s| s.as_str()).collect();
                collect_detail_label_value_pairs(&refs)
            };

            // 🌟 [ENRICHED EMBEDDING] 값 라인을 "라벨 | 값" 으로 임베딩합니다.
            //    태그 껍데기만 들어간 원시 라인 임베딩은 변별력이 없어
            //    status 가 '| 환불, 반품완료 후' 같은 안내 문구에 배정되던 원인이었습니다.
            let mut line_enriched_texts: Vec<String> = vec![String::new(); pug_lines.len()];
            for p in &detail_pairs {
                if p.primary_line < line_enriched_texts.len() && line_enriched_texts[p.primary_line].is_empty() {
                    line_enriched_texts[p.primary_line] = format!("{} | {}", p.label, p.value);
                }
            }
            for p in &detail_pairs {
                emit_term(&format!(
                    "  🧷 [DETAIL PAIR] Line {} | Section: '{}' | Label: '{}' | Value: '{}'",
                    p.primary_line + 1, p.section, p.label, p.value
                ));
            }

            let mut line_embeddings = vec![vec![0.0; 384]; pug_lines.len()];
            

            let mut texts_to_embed = Vec::new();
            let mut text_indices = Vec::new();
            
            for (line_idx, line) in pug_lines.iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
                if line.trim().is_empty() { continue; }
                let target = if line_enriched_texts[line_idx].is_empty() {
                    line.to_string()
                } else {
                    line_enriched_texts[line_idx].clone()
                };
                texts_to_embed.push(target);
                text_indices.push(line_idx);
            }
            
            if !texts_to_embed.is_empty() {
                for (chunk_idx, text_chunk) in texts_to_embed.chunks(100).enumerate() {
                    let start_idx = chunk_idx * 100;
                    if let Ok(vectors) = model.get_embedding_batch(text_chunk.to_vec()).await {
                        for (i, vector) in vectors.into_iter().enumerate() {
                            let original_idx = text_indices[start_idx + i];
                            emit_term(&format!("  [VECTORIZING] Stage-3 Line {}/{} : {}", original_idx + 1, pug_lines.len(), text_chunk[i].trim()));
                            line_embeddings[original_idx] = vector;
                        }
                    }
                }
            }


            let (list_bias, form_bias, _) = crate::parsing::get_combinatorial_layout_bias(&[&page_type], &doc_lang);
            let list_bias_emb: Vec<f32> = model.get_embedding(list_bias.clone()).await.unwrap_or(vec![0.0f32; 384]);
            let form_bias_emb: Vec<f32> = model.get_embedding(form_bias.clone()).await.unwrap_or(vec![0.0f32; 384]);
            
            let mut wiped_indices = vec![false; pug_lines.len()];
            let mut processed_blocks = std::collections::HashSet::new();


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
            
            let js_template_detail = get_boa_block_extractor_template();

            let mut track_a_candidates = Vec::new();
            let mut track_a_indices = Vec::new();
            let mut seen_detail_candidates = std::collections::HashSet::new();

            for line_idx in 0..pug_lines.len() {
                if wiped_indices[line_idx] { continue; }
                let line = &pug_lines[line_idx];
                if line.trim().is_empty() { continue; }
                
                let line_prej_score = cosine_similarity(&layout_prej_emb, &line_embeddings[line_idx]);
                
                if line_prej_score > 0.55 {
                    let text_part = if let Some(idx) = line.find('|') { line[idx + 1..].trim() } else { line.trim() };
                    if !text_part.is_empty() && !seen_detail_candidates.contains(text_part) {
                        seen_detail_candidates.insert(text_part.to_string());
                        track_a_candidates.push(text_part.to_string());
                        track_a_indices.push(line_idx);
                    }
                }
            }


            let track_a_selectors: Vec<String> = {
                let target_len = track_a_candidates.len();
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


            let stage3_pugs: Vec<String> = {
                let html_clone = clean_html_content.clone();
                let selectors = track_a_selectors.clone();
                
                tokio::task::spawn_blocking(move || {
                    let mut seen_stage3_sels = std::collections::HashSet::new();
                    let mut unique_sels = Vec::new();
                    for sel in selectors {
                        if sel.is_empty() { continue; }
                        if !seen_stage3_sels.contains(&sel) {
                            seen_stage3_sels.insert(sel.clone());
                            unique_sels.push(sel);
                        }
                    }

                    let mut results = Vec::new();
                    let num_threads = 8;
                    let chunk_size = (unique_sels.len() + num_threads - 1) / num_threads;
                    
                    if chunk_size > 0 {
                        std::thread::scope(|s| {
                            let mut handles = Vec::new();
                            for chunk in unique_sels.chunks(chunk_size) {
                                let chunk_owned = chunk.to_vec();
                                let html_ref = &html_clone;
                                handles.push(s.spawn(move || {
                                    let doc = scraper::Html::parse_document(html_ref);
                                    let mut local_res = Vec::with_capacity(chunk_owned.len());
                                    for sel in chunk_owned {
                                        local_res.push(crate::parsing::convert_doc_to_clean_pug_selector(&doc, &sel, crate::parsing::PugMode::DetailMode, None));
                                    }
                                    local_res
                                }));
                            }
                            for h in handles {
                                if let Ok(local_res) = h.join() {
                                    results.extend(local_res);
                                }
                            }
                        });
                    }
                    results
                }).await.unwrap_or_default()
            };


            let mut unique_stage3_pugs_to_embed = Vec::new();
            for block_pug in &stage3_pugs {
                if block_pug.is_empty() || processed_blocks.contains(block_pug) { continue; }
                processed_blocks.insert(block_pug.clone());
                unique_stage3_pugs_to_embed.push(block_pug.clone());
            }

            let mut stage3_embeddings_map = std::collections::HashMap::new();
            if !unique_stage3_pugs_to_embed.is_empty() {
                for chunk in unique_stage3_pugs_to_embed.chunks(100) {
                    if let Ok(vectors) = model.get_embedding_batch(chunk.to_vec()).await {
                        for (i, vector) in vectors.into_iter().enumerate() {
                            stage3_embeddings_map.insert(chunk[i].clone(), vector);
                        }
                    }
                }
            }

            for block_pug in stage3_pugs {
                if block_pug.is_empty() { continue; }
                let block_emb = stage3_embeddings_map.get(&block_pug).cloned().unwrap_or(vec![0.0; 384]);
                
                let block_prej_score = cosine_similarity(&layout_prej_emb, &block_emb);
                let block_list_score = cosine_similarity(&list_bias_emb, &block_emb);
                let block_form_score = cosine_similarity(&form_bias_emb, &block_emb);
                
                if block_prej_score > block_list_score && block_prej_score > block_form_score {
                    if let Some((start_idx, end_idx)) = find_block_indices_in_pug(&pug_lines, &block_pug) {
                        emit_term(&format!("  🚫 [NOISE BLOCK DELETED] Boa Matched. Lines {}~{} (Prej: {:.4} > List: {:.4} & Form: {:.4})", start_idx + 1, end_idx + 1, block_prej_score, block_list_score, block_form_score));
                        for j in start_idx..=end_idx {
                            pug_lines[j] = String::new();
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


            let doc_title = {
                let doc = scraper::Html::parse_document(&clean_html_content);
                let mut t_val = if let Ok(sel) = scraper::Selector::parse("title") {
                    doc.select(&sel).next().map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string()).unwrap_or_default()
                } else {
                    String::new()
                };
                

                if t_val.is_empty() || t_val.len() < 5 {
                    let mut heading_texts = Vec::new();
                    if let Ok(sel_h1) = scraper::Selector::parse("h1") {
                        for el in doc.select(&sel_h1) {
                            heading_texts.push(el.text().collect::<Vec<_>>().join(" ").trim().to_string());
                        }
                    }
                    if let Ok(sel_h2) = scraper::Selector::parse("h2") {
                        for el in doc.select(&sel_h2) {
                            heading_texts.push(el.text().collect::<Vec<_>>().join(" ").trim().to_string());
                        }
                    }
                    if !heading_texts.is_empty() {
                        t_val = heading_texts.join(" | ");
                    }
                }
                t_val
            };


            let mut field_embeddings = Vec::new();
            // 🌟 [PHRASE-LEVEL BIAS BANK] 리스트 경로와 동일하게 센트로이드 임베딩을 폐기하고
            //    구(phrase) 단위 Max-Pool 뱅크 + 합성 필드 플래그 + 형식 뱅크를 구축합니다.
            let mut field_phrase_embs: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut field_phrase_weights: Vec<Vec<f32>> = Vec::new();
            // 🌟 [PHRASE-LEVEL PREJUDICE BANK] Enum 폴백 판정에 쓸 구 단위 편견 뱅크입니다.
            let mut field_prej_phrase_embs: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut field_is_analytic: Vec<bool> = Vec::new();
            let mut field_formats: Vec<FieldFormat> = Vec::new();

            for (f_idx, (fname, _, bias_target, predefined_prej)) in fields.iter().enumerate() {
                let bias_emb = model.get_embedding(bias_target.clone()).await.unwrap_or(vec![0.0; 384]);

                let (phrases, phrase_weights) = split_bias_phrases_weighted(bias_target);
                let p_embs = if phrases.is_empty() {
                    vec![bias_emb.clone()]
                } else {
                    model.get_embedding_batch(phrases.clone()).await.unwrap_or_else(|_| vec![bias_emb.clone(); phrases.len()])
                };
                let p_weights = if phrases.is_empty() { vec![1.0f32] } else { phrase_weights };
                field_phrase_embs.push(p_embs);
                field_phrase_weights.push(p_weights);

                let prej_phrases = prejudice_phrase_bank(&doc_lang, &page_type, fname);
                let prej_p_embs = if prej_phrases.is_empty() {
                    Vec::new()
                } else {
                    model.get_embedding_batch(prej_phrases.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; prej_phrases.len()])
                };
                field_prej_phrase_embs.push(prej_p_embs);

                let detected_fmt = detect_field_format(fname);
                field_formats.push(detected_fmt);
                emit_term(&format!("  📐 [FORMAT REGISTERED] '{}' → {:?}", fname, detected_fmt));

                let lower_fname = fname.to_lowercase();
                let is_analytic = lower_fname.contains("insight")
                    || lower_fname.contains("summary")
                    || lower_fname.contains("analysis");
                field_is_analytic.push(is_analytic);
                if is_analytic {
                    emit_term(&format!("  🧠 [SYNTHESIS FIELD REGISTERED] '{}' 는 벡터 라인 매칭에서 제외되고 전체 컨텍스트 요약 필드로 처리됩니다.", fname));
                }

                let mut dynamic_prej_texts = Vec::new();
                if !predefined_prej.trim().is_empty() {
                    dynamic_prej_texts.push(predefined_prej.clone());
                }
                for (other_idx, (_, _, other_bias, _)) in fields.iter().enumerate() {
                    if f_idx != other_idx {
                        dynamic_prej_texts.push(other_bias.clone());
                    }
                }
                let combined_prej = dynamic_prej_texts.join(" , ");
                let prej_emb = model.get_embedding(combined_prej.clone()).await.unwrap_or(vec![0.0; 384]);

                field_embeddings.push((bias_emb, prej_emb, combined_prej));
            }


            let mut pre_mapped_hints = Vec::new();
            

            let mut url_pool = String::new();
            if let Ok(href_re) = regex::Regex::new(r#"href=["']([^"']+)["']"#) {
                for line in &pug_lines_ref {
                    for cap in href_re.captures_iter(line) {
                        if let Some(m) = cap.get(1) {
                            url_pool.push_str(&m.as_str().to_lowercase());
                            url_pool.push_str(" ");
                        }
                    }
                }
            }
            // 🌟 현재 추출 중인 주소 자체도 식별자 풀에 포함시킵니다. (od_id=24120419364235)
            url_pool.push_str(&url.to_lowercase());
            url_pool.push_str(" ");

            // 🌟 [LINE ANATOMY] 속성 내부 파이프에 속지 않는 안전 파서로 라인 구조를 확정합니다.
            let line_parts: Vec<(usize, String, String, String)> =
                pug_lines_ref.iter().map(|l| pug_line_parts(l)).collect();

            // 🌟 [VALUE EXTRACTION] 형식 게이트는 파이프 뒤의 실제 값만 검사해야 합니다.
            let line_values: Vec<String> = line_parts.iter().map(|p| p.3.clone()).collect();

            // 🌟 [ROLE GATE] th / label / legend / caption / h1~h6 / tr / table 은 '제목·컨테이너' 역할이므로
            //    어떤 필드의 값도 될 수 없습니다. (payment_method ← "th | 무통장 입금액" 오매칭 차단)
            let line_is_non_value: Vec<bool> = line_parts.iter().map(|p| is_non_value_role_tag(&p.1)).collect();
            // 🌟 select > option[selected] 는 '현재 선택된 값' 이므로 Enum 의 유일한 벡터 폴백 후보입니다.
            let line_is_selected_option: Vec<bool> = line_parts.iter()
                .map(|p| p.1 == "option" && pug_attr_flag(&p.2, "selected"))
                .collect();

            // 🌟 [ID/LINK COSINE BANK - DETAIL] 리스트 경로에만 있던 라벨/편견 구 뱅크를 디테일에도 구축합니다.
            let (idlink_label_phrases, idlink_label_weights) = label_phrase_bank(&doc_lang, &page_type, "id,link");
            let idlink_label_embs: Vec<Vec<f32>> = if idlink_label_phrases.is_empty() {
                Vec::new()
            } else {
                model.get_embedding_batch(idlink_label_phrases.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; idlink_label_phrases.len()])
            };
            let mut idlink_prej_phrases = prejudice_phrase_bank(&doc_lang, &page_type, "id,link");
            for extra in [
                "host name", "domain name", "website address", "server address",
                "cdn", "static asset", "image server", "protocol", "www",
                "file extension", "stylesheet", "script", "anchor", "javascript",
                "navigation menu", "layer popup", "delivery tracking service", "postal service",
            ] {
                let e = extra.to_string();
                if !idlink_prej_phrases.contains(&e) { idlink_prej_phrases.push(e); }
            }
            let idlink_prej_embs: Vec<Vec<f32>> = if idlink_prej_phrases.is_empty() {
                Vec::new()
            } else {
                model.get_embedding_batch(idlink_prej_phrases.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; idlink_prej_phrases.len()])
            };
            emit_term(&format!("  🔑 [ID/LINK COSINE BANK] 라벨 구 {}개 | 편견 구 {}개 준비 완료.", idlink_label_embs.len(), idlink_prej_embs.len()));

            let mut det_id_link: Option<(String, String)> = None;

            // 🌟 [1순위 · PAGE-URL ID PRIORITY] "지금 추출 중인 주소(link)" 안에 id 가 있는지 먼저 확인합니다.
            //    상세페이지의 주문번호는 문서 안 a[href] 가 아니라 현재 URL 쿼리(od_id=...)에 실려 있습니다.
            {
                let url_cands = collect_id_link_candidates_from_url(&url);
                if !url_cands.is_empty() && !idlink_label_embs.is_empty() {
                    let role_texts: Vec<String> = url_cands.iter().map(|c| c.role_phrase.clone()).collect();
                    let role_embs = model.get_embedding_batch(role_texts.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; role_texts.len()]);

                    let mut best = f32::MIN;
                    let mut best_i: Option<usize> = None;
                    for (ci, c) in url_cands.iter().enumerate() {
                        let emb = &role_embs[ci];
                        if emb.iter().all(|&v| v == 0.0) { continue; }
                        let own = weighted_max_pool_sim(emb, &idlink_label_embs, &idlink_label_weights);
                        let prej = if idlink_prej_embs.is_empty() { 0.0 } else { max_pool_sim(emb, &idlink_prej_embs) };
                        let score = (own - prej) + 0.15 * (c.prior - 1.0);
                        emit_term(&format!("      🧭 [PAGE-URL ID CANDIDATE] '{}' ← 역할 '{}' | LabelMaxPool: {:.4} | PrejMaxPool: {:.4} | Prior: {:.2} | Score: {:+.4}",
                            c.token, c.role_phrase, own, prej, c.prior, score));
                        if own < 0.30 { continue; }
                        if score <= 0.0 { continue; }
                        if score > best { best = score; best_i = Some(ci); }
                    }

                    if let Some(bi) = best_i {
                        let c = &url_cands[bi];
                        let page_link = match url::Url::parse(&url) {
                            Ok(u) => format!("{}{}", u.path(), u.query().map(|q| format!("?{}", q)).unwrap_or_default()).to_lowercase(),
                            Err(_) => url.clone(),
                        };
                        emit_term(&format!("  🔑 [PAGE-URL ID PRIORITY] 추출 주소에서 식별자 '{}' 확정 (역할 '{}', Score {:+.4}) → link '{}'",
                            c.token, c.role_phrase, best, page_link));
                        det_id_link = Some((c.token.clone(), page_link));
                    }
                }
            }

            // 🌟 [2순위 · 문서 내부 href] 단, 현재 페이지와 호스트가 다른 외부 링크는 후보에서 제외합니다.
            //    (우체국 배송조회 URL 의 sid1=123456789 가 id 로 승격되던 사고 차단)
            if det_id_link.is_none() && !idlink_label_embs.is_empty() {
                let cands: Vec<_> = collect_id_link_candidates(&pug_lines_ref)
                    .into_iter()
                    .filter(|c| {
                        if c.is_host_part { return false; }
                        if !same_host(&url, &c.href) {
                            return false;
                        }
                        true
                    })
                    .collect();

                if !cands.is_empty() {
                    let role_texts: Vec<String> = cands.iter().map(|c| c.role_phrase.clone()).collect();
                    let role_embs = model.get_embedding_batch(role_texts.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; role_texts.len()]);

                    let mut best = f32::MIN;
                    let mut best_i: Option<usize> = None;
                    for (ci, c) in cands.iter().enumerate() {
                        let emb = &role_embs[ci];
                        if emb.iter().all(|&v| v == 0.0) { continue; }
                        let own = weighted_max_pool_sim(emb, &idlink_label_embs, &idlink_label_weights);
                        let prej = if idlink_prej_embs.is_empty() { 0.0 } else { max_pool_sim(emb, &idlink_prej_embs) };
                        let score = (own - prej) + 0.15 * (c.prior - 1.0);
                        emit_term(&format!("      🧭 [ID/LINK CANDIDATE] '{}' ← 역할 '{}' | LabelMaxPool: {:.4} | PrejMaxPool: {:.4} | Score: {:+.4}",
                            c.token, c.role_phrase, own, prej, score));
                        if own < 0.30 { continue; }
                        if score <= 0.0 { continue; }
                        if score > best { best = score; best_i = Some(ci); }
                    }

                    if let Some(bi) = best_i {
                        let c = &cands[bi];
                        emit_term(&format!("  🔑 [ID/LINK COSINE] 식별자 '{}' 확정 (역할 '{}', Score {:+.4}) → link '{}'", c.token, c.role_phrase, best, c.href));
                        det_id_link = Some((c.token.clone(), c.href.clone()));
                    }
                }
            }

            // 🌟 [3순위 · 레거시 결정론 해석기] 위 두 게이트를 모두 통과하지 못했을 때만 동작합니다.
            if det_id_link.is_none() {
                det_id_link = resolve_id_link_from_lines(&pug_lines_ref);
                if let Some((fid, flink)) = &det_id_link {
                    emit_term(&format!("  🔑 [ID/LINK FALLBACK] 코사인 게이트 통과 후보가 없어 레거시 해석기로 확정: '{}' → '{}'", fid, flink));
                }
            }

            let mut det_consumed_lines: std::collections::HashSet<usize> = std::collections::HashSet::new();
            if let Some((det_id, _)) = &det_id_link {
                for (l, v) in line_values.iter().enumerate() {
                    if v.is_empty() { continue; }
                    let matched = v.split(|c: char| !c.is_alphanumeric())
                        .any(|tok| !tok.is_empty() && tok.eq_ignore_ascii_case(det_id.as_str()));
                    if matched { det_consumed_lines.insert(l); }
                }
            }

            // 🌟 [DETAIL PAIR COSINE MAP v2] 구조적으로 결합된 (라벨 → 값) 페어를
            //    ① 편견 자기오염 제거 ② 리프라벨/섹션 이중 행렬 ③ 이중 센터링 ④ 점수순 배타 배정
            //    네 단계로 확정합니다. 고정 임계치(0.55 / 0.10 / 0.03)는 전부 폐기합니다.
            //    폐기 근거: 다국어 임베딩의 한국어 짧은 라벨 코사인은 0.55~0.90 대역에 뭉쳐 있어
            //    '계좌번호→bank +0.0701', '개별 전자결제(PG)→payment_origin +0.0471',
            //    '주문하신 분 이름→sender_name -0.0015' 처럼 정답이 절대 임계치에 전멸했습니다.
            let mut header_forced_assign: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            let mut pair_owned_lines: std::collections::HashSet<usize> = std::collections::HashSet::new();
            // 🌟 [SCOPE FIX] FORMAT FAMILY SHARE 블록에서도 접근할 수 있도록 if 블록 바깥에 선언합니다.
            let mut pair_line_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

            if !detail_pairs.is_empty() {
                // 동일 라벨("이름", "핸드폰", "주소")이 여러 번 등장하면 섹션 제목을 접두어로 붙여 구분합니다.
                let mut label_count: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                for p in &detail_pairs { *label_count.entry(p.label.clone()).or_insert(0) += 1; }

                let mut pair_phrases: Vec<String> = Vec::with_capacity(detail_pairs.len());
                for p in &detail_pairs {
                    let dup = label_count.get(&p.label).copied().unwrap_or(0) > 1;
                    if dup && !p.section.trim().is_empty() {
                        pair_phrases.push(format!("{} {}", p.section.trim(), p.label));
                    } else {
                        pair_phrases.push(p.label.clone());
                    }
                }

                // 🌟 유일 키를 만들면서 '리프 라벨'과 '섹션'을 분리 보관합니다.
                //    Max-Pool 은 "주문하신 분 이름" 에서 공통 구 '이름' 만 뽑아
                //    sender_name 과 recipient_name 을 완전 동률로 만들어 버립니다.
                //    유일한 판별 신호인 섹션을 별도 행렬로 살려야 합니다.
                let mut unique_phrases: Vec<String> = Vec::new();
                let mut unique_leaf: Vec<String> = Vec::new();
                let mut unique_section: Vec<String> = Vec::new();
                for (pi, ph) in pair_phrases.iter().enumerate() {
                    if unique_phrases.iter().any(|e| e == ph) { continue; }
                    unique_phrases.push(ph.clone());
                    unique_leaf.push(detail_pairs[pi].label.clone());
                    unique_section.push(detail_pairs[pi].section.trim().to_string());
                }

                let leaf_embs: Vec<Vec<f32>> = model.get_embedding_batch(unique_leaf.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; unique_leaf.len()]);
                let section_texts: Vec<String> = unique_section.iter()
                    .map(|s| if s.is_empty() { " ".to_string() } else { s.clone() })
                    .collect();
                let section_embs: Vec<Vec<f32>> = model.get_embedding_batch(section_texts.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; section_texts.len()]);

                let mut d_field_names: Vec<String> = Vec::new();
                let mut d_label_embs: Vec<Vec<Vec<f32>>> = Vec::new();
                let mut d_label_weights: Vec<Vec<f32>> = Vec::new();
                let mut d_prej_raw: Vec<Vec<Vec<f32>>> = Vec::new();
                let mut d_prej_texts: Vec<Vec<String>> = Vec::new();

                for (fname, _, _, _) in &fields {
                    let (lp, lw) = label_phrase_bank(&doc_lang, &page_type, fname);
                    if lp.is_empty() { continue; }
                    let pp = prejudice_phrase_bank(&doc_lang, &page_type, fname);
                    let le = model.get_embedding_batch(lp.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; lp.len()]);
                    let pe = if pp.is_empty() {
                        Vec::new()
                    } else {
                        model.get_embedding_batch(pp.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; pp.len()])
                    };
                    d_field_names.push(fname.clone());
                    d_label_embs.push(le);
                    d_label_weights.push(lw);
                    d_prej_raw.push(pe);
                    d_prej_texts.push(pp);
                }

                // 🌟 ① [SELF-POISON GUARD] recipient_address.prejudice 의 '받는사람',
                //    sender_phone.prejudice 의 '주문자' 처럼 자기 자신을 가장 잘 설명하는
                //    편견 구를 코사인으로 색출해 제거합니다.
                let mut d_prej_embs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(d_field_names.len());
                for f in 0..d_field_names.len() {
                    let mask = self_poisoned_prejudice_mask(&d_label_embs[f], &d_prej_raw[f], &d_label_embs, f);
                    let mut kept: Vec<Vec<f32>> = Vec::new();
                    let mut dropped = 0usize;
                    for (pi, poisoned) in mask.iter().enumerate() {
                        if *poisoned {
                            dropped += 1;
                            if dropped <= 6 {
                                emit_term(&format!("    🧪 [SELF-POISON DROP] '{}' 의 편견 구 '{}' 는 경쟁 필드보다 자기 자신을 더 잘 설명하므로 편견 자격을 박탈합니다.",
                                    d_field_names[f], d_prej_texts[f].get(pi).cloned().unwrap_or_default()));
                            }
                        } else {
                            kept.push(d_prej_raw[f][pi].clone());
                        }
                    }
                    emit_term(&format!("  🏷️ [LABEL BANK] '{}' | 라벨 구 {}개 | 편견 구 {}개 (자기오염 {}개 제거)",
                        d_field_names[f], d_label_embs[f].len(), kept.len(), dropped));
                    d_prej_embs.push(kept);
                }

                // 🌟 ②-0 [PAIR VALUE PRE-BUILD] 각 유일 라벨이 실제로 들고 있는 값을 미리 합성합니다.
                //    행렬을 만들기 '전에' 값 형식을 검증해야, 형식이 맞지 않는 라벨이 필드를 선점한 뒤
                //    [DETAIL PAIR FORMAT REJECT] 로 버려지면서 그 필드가 통째로 빈값이 되는 사고를 막습니다.
                //    (로그: registration_date←'주문채널', sender_address←'주문하신 분 IP Address',
                //           payment_date←'전달 메세지' 가 전부 이 경로로 소멸했습니다)
                let mut phrase_single_value: Vec<String> = vec![String::new(); unique_phrases.len()];
                let mut phrase_multi_value: Vec<String> = vec![String::new(); unique_phrases.len()];
                for (pi, ph) in pair_phrases.iter().enumerate() {
                    let h = match unique_phrases.iter().position(|u| u == ph) { Some(v) => v, None => continue };
                    let p = &detail_pairs[pi];
                    if p.primary_line >= pug_lines_ref.len() { continue; }
                    if pug_lines_ref[p.primary_line].trim().is_empty() { continue; }
                    if phrase_single_value[h].is_empty() && !p.value.trim().is_empty() {
                        phrase_single_value[h] = p.value.clone();
                    }
                    let av = p.value_all.trim();
                    if !av.is_empty() && !phrase_multi_value[h].contains(av) {
                        if phrase_multi_value[h].is_empty() {
                            phrase_multi_value[h] = av.to_string();
                        } else {
                            phrase_multi_value[h].push(' ');
                            phrase_multi_value[h].push_str(av);
                        }
                    }
                }

                // 🌟 ② 리프 행렬 / 섹션 행렬을 각각 원시 코사인으로 채웁니다.
                //    편견은 '점수에서 빼는' 방식이 아니라 '경쟁 개념이 우세하면 후보 탈락'이라는
                //    상대 게이트로만 사용합니다. (절대 감점은 임계치 지옥을 다시 만듭니다)
                let pair_abs_floor = 0.50f32;
                let mut leaf_raw: Vec<Vec<f32>> = vec![vec![-1.0f32; unique_phrases.len()]; d_field_names.len()];
                let mut sec_raw: Vec<Vec<f32>> = vec![vec![-1.0f32; unique_phrases.len()]; d_field_names.len()];

                for f in 0..d_field_names.len() {
                    let f_fmt = detect_field_format(&d_field_names[f]);
                    let f_multi = is_multi_value_field(&d_field_names[f]);
                    // 생김새가 확정적인 형식은 행렬 진입 전에 값까지 검증합니다.
                    let f_strict = matches!(
                        f_fmt,
                        FieldFormat::Date | FieldFormat::TrackingCode | FieldFormat::Numeric
                            | FieldFormat::Phone | FieldFormat::Address | FieldFormat::Text
                    );

                    for h in 0..unique_phrases.len() {
                        if leaf_embs[h].iter().all(|&v| v == 0.0) { continue; }
                        let own = weighted_max_pool_sim(&leaf_embs[h], &d_label_embs[f], &d_label_weights[f]);
                        if own < pair_abs_floor { continue; }
                        let prej = if d_prej_embs[f].is_empty() { 0.0 } else { max_pool_sim(&leaf_embs[h], &d_prej_embs[f]) };
                        if prej >= own {
                            emit_term(&format!("    🚫 [PAIR PREJUDICE GATE] '{}' → '{}' | LabelMaxPool: {:.4} <= PrejMaxPool: {:.4}. 경쟁 개념이 우세하여 후보 제외.",
                                unique_phrases[h], d_field_names[f], own, prej));
                            continue;
                        }

                        // 🌟 ②-1 [PAIR VALUE FORMAT GATE] 라벨이 아무리 잘 맞아도
                        //    그 라벨이 들고 있는 값이 필드 형식과 다르면 후보가 될 수 없습니다.
                        let pair_val = if f_multi { &phrase_multi_value[h] } else { &phrase_single_value[h] };
                        if f_strict {
                            if pair_val.trim().is_empty() || !value_matches_format(f_fmt, pair_val) {
                                emit_term(&format!("    🚫 [PAIR VALUE FORMAT GATE] '{}' → '{}' ({:?}) | 값 \"{}\" 이 형식과 불일치하여 후보 제외.",
                                    unique_phrases[h], d_field_names[f], f_fmt, pair_val));
                                continue;
                            }
                        }
                        // 🌟 ②-2 [ENUM NUMERIC GATE] 열거형은 '상태/수단/기관명' 이므로
                        //    순수 금액·수량이 될 수 없습니다.
                        //    (로그: bank←'상품금액' 615600, payment_method←'무통장 입금액' 615600원)
                        if f_fmt == FieldFormat::Enum && is_pure_numeric_value(pair_val) {
                            emit_term(&format!("    🚫 [ENUM NUMERIC GATE] '{}' → '{}' | 값 \"{}\" 은 순수 수치이므로 열거형 후보가 될 수 없습니다.",
                                unique_phrases[h], d_field_names[f], pair_val));
                            continue;
                        }

                        leaf_raw[f][h] = own;

                        if unique_section[h].is_empty() { continue; }
                        if section_embs[h].iter().all(|&v| v == 0.0) { continue; }
                        sec_raw[f][h] = weighted_max_pool_sim(&section_embs[h], &d_label_embs[f], &d_label_weights[f]);
                    }
                }

                // 🌟 ③ [이중 센터링 폐기 + 섹션 라인 대비]
                //    이중 센터링은 밀집 행렬 전용 도구입니다. 편견/형식 게이트를 통과한 페어 행렬은
                //    밀도가 ~5% 라 '유효 후보가 1개뿐인 라벨'이 대부분이고, 그 경우
                //      centered = own - own - field_mean + global = global - field_mean
                //    으로 own 이 식에서 완전히 소거됩니다.
                //    → '계좌번호'(0.8817) 와 '상품금액'(0.7073) 의 점수가 비트 단위로 같아졌습니다.
                //    라벨 간 경쟁은 exclusive_assign_by_score 의 margin 이 이미 담당하므로
                //    센터링은 '동률을 깨는 섹션 항'에만 라인 단위로 제한 적용합니다.
                //    (sender_*/recipient_* 는 리프 라벨이 '이름/핸드폰/주소'로 완전 동률이라
                //     이 항이 유일한 판별 신호입니다)
                const SECTION_WEIGHT: f32 = 0.5f32;
                let mut d_matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; unique_phrases.len()]; d_field_names.len()];
                for h in 0..unique_phrases.len() {
                    let mut sec_sum = 0.0f32;
                    let mut sec_cnt = 0usize;
                    for f in 0..d_field_names.len() {
                        if leaf_raw[f][h] < 0.0 { continue; }
                        if sec_raw[f][h] < 0.0 { continue; }
                        sec_sum += sec_raw[f][h];
                        sec_cnt += 1;
                    }
                    let sec_mean = if sec_cnt > 0 { sec_sum / (sec_cnt as f32) } else { 0.0 };
                    for f in 0..d_field_names.len() {
                        if leaf_raw[f][h] < 0.0 { continue; }
                        // 경쟁 필드가 2개 이상일 때만 섹션 대비가 의미를 갖습니다.
                        let sec_term = if sec_cnt > 1 && sec_raw[f][h] >= 0.0 {
                            sec_raw[f][h] - sec_mean
                        } else {
                            0.0
                        };
                        d_matrix[f][h] = leaf_raw[f][h] + SECTION_WEIGHT * sec_term;
                    }
                }

                // 🌟 ④ 절대 점수(증거 강도) 우선 배타 배정.
                //    '계좌번호'(0.8817) 가 '상품금액'(0.7073) 보다, '주문하신 분 이름'(1.0) 이
                //    '판매자'(0.76) 보다 먼저 필드를 잠급니다.
                let d_assign = exclusive_assign_by_score(&d_matrix, 0.0, 0.0);
                // 🌟 [PAIR LINE MAP] FORMAT FAMILY SHARE가 PRE-MAP'd 필드의 실제 라인을
                //    소스로 사용할 수 있도록 field_name → primary_line 기록을 구축합니다.
                for (f, a) in d_assign.iter().enumerate() {
                    let (h, score, margin) = match a { Some(v) => *v, None => continue };
                    let owner = d_field_names[f].clone();

                    // id,link 는 '주소 우선' 결정론 해석기가 전담합니다.
                    if is_id_link_field(&owner) { continue; }

                    let mut targets: Vec<usize> = Vec::new();
                    for (pi, ph) in pair_phrases.iter().enumerate() {
                        if ph == &unique_phrases[h] { targets.push(pi); }
                    }
                    if targets.is_empty() { continue; }

                    let owner_fmt = detect_field_format(&owner);
                    let multi = is_multi_value_field(&owner);

                    let mut merged = String::new();
                    let mut primary = detail_pairs[targets[0]].primary_line;
                    for pi in &targets {
                        let p = &detail_pairs[*pi];
                        if p.primary_line >= pug_lines_ref.len() { continue; }
                        if pug_lines_ref[p.primary_line].trim().is_empty() { continue; }
                        let v = if multi { p.value_all.clone() } else { p.value.clone() };
                        if v.trim().is_empty() { continue; }
                        pair_owned_lines.insert(p.primary_line);
                        if merged.is_empty() {
                            merged = v;
                            primary = p.primary_line;
                        } else if multi && !merged.contains(&v) {
                            merged.push(' ');
                            merged.push_str(&v);
                        }
                    }
                    if merged.trim().is_empty() { continue; }

                    let lower_owner = owner.to_lowercase();
                    let needs_normalization = lower_owner.contains("status")
                        || lower_owner.contains("payment_method")
                        || lower_owner.contains("payment_origin")
                        || lower_owner.contains("condition")
                        || lower_owner.contains("currency");

                    if needs_normalization {
                        header_forced_assign.insert(owner.clone(), primary);
                        emit_term(&format!("    🎯 [DETAIL PAIR FORCED ASSIGN] '{}' ← Line {} (\"{}\") | Label '{}' | Score: {:+.4} | Margin: {:+.4} | enum 정규화가 필요해 값 우회 대신 벡터 배정을 확정합니다.",
                            owner, primary + 1, merged, unique_phrases[h], score, margin));
                        continue;
                    }

                    let fmt_ok = match owner_fmt {
                        FieldFormat::Identifier | FieldFormat::Link => true,
                        _ => value_matches_format(owner_fmt, &merged),
                    };
                    if !fmt_ok {
                        emit_term(&format!("    🚫 [DETAIL PAIR FORMAT REJECT] '{}' ({:?}) | 라벨 '{}' 의 값 '{}' 이 형식과 불일치하여 주입하지 않습니다.",
                            owner, owner_fmt, unique_phrases[h], merged));
                        continue;
                    }

                    pair_line_map.insert(owner.clone(), primary);
                    pre_mapped_hints.push(json!({
                        "target_column": owner.clone(),
                        "extracted_value": merged.clone()
                    }));
                    emit_term(&format!("    ✨ [DETAIL PAIR COSINE MAP] Label '{}' → Field '{}' | LeafRaw: {:.4} | SecRaw: {:.4} | Centered: {:+.4} | Margin: {:+.4} | Line {} | Value: \"{}\"",
                        unique_phrases[h], owner,
                        leaf_raw[f][h].max(0.0),
                        sec_raw[f][h].max(0.0),
                        score, margin, primary + 1, merged));
                }
            }

            // 구조적으로 주인이 확정된 라인은 다른 필드가 벡터로 선점할 수 없습니다.
            for l in &pair_owned_lines { det_consumed_lines.insert(*l); }

            // 🌟 [ENUM SELECT RESOLVER]
            //  상태는 '취소' 라는 한국어 리터럴 매칭이 아니라 아래 4단계로 확정합니다.
            //  ① 원본 HTML 에서 모든 <select> 를 수집합니다.
            //     (PUG 는 selected 아닌 option 을 버리므로 '반품/교환' 후보 집합이 소멸합니다)
            //  ② 각 select 의 "옵션 집합"을 bias.json status_filters(영어 캐노니컬) 뱅크와
            //     코사인 대조하고, 택배사/은행/카드 같은 '상태가 아닌 열거형' 뱅크와의 대비를 뺍니다.
            //  ③ 마진이 충분하면 selected 옵션 텍스트를 다시 캐노니컬 키로 환산합니다.
            //  ④ 마진이 부족해 애매할 때만 LLM(Qwen3.5)에게 CSS selector 를 묻고,
            //     값 자체는 우리가 그 selector 의 selected 옵션에서 결정론적으로 읽습니다.
            let mut enum_resolved: std::collections::HashMap<String, String> = std::collections::HashMap::new();
            {
                let select_groups = collect_select_groups(&clean_html_content);
                if select_groups.is_empty() {
                    emit_term("  ⚪ [ENUM SELECT] 문서에 <select> 컨트롤이 없어 상태 선택자 해석을 건너뜁니다.");
                } else {
                    let status_keys = enum_status_keys(&page_type);
                    let mut key_banks: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
                    for k in &status_keys {
                        let phrases = status_key_phrases(k);
                        let e = model.get_embedding_batch(phrases.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; phrases.len()]);
                        key_banks.push((k.to_string(), e));
                    }

                    // '상태가 아닌 열거형' 경쟁 뱅크 : 택배사 / 은행 / 카드 / PG / 안내문구
                    let rival_phrases: Vec<String> = {
                        let mut v: Vec<String> = vec![
                            "delivery company".to_string(), "courier company".to_string(),
                            "shipping carrier".to_string(), "postal service".to_string(),
                            "bank name".to_string(), "bank account number".to_string(),
                            "credit card company".to_string(), "payment gateway".to_string(),
                            "please select".to_string(), "choose an option".to_string(),
                            "category".to_string(), "brand".to_string(), "country".to_string(),
                        ];
                        for fname in ["carrier", "bank", "card", "payment_origin", "payment_method"] {
                            let (lp, _) = label_phrase_bank(&doc_lang, &page_type, fname);
                            for p in lp { if !v.iter().any(|e| e == &p) { v.push(p); } }
                        }
                        if v.len() > 64 { v.truncate(64); }
                        v
                    };
                    let rival_bank = model.get_embedding_batch(rival_phrases.clone()).await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; rival_phrases.len()]);

                    let mut scored: Vec<(usize, f32)> = Vec::new();
                    for (gi, g) in select_groups.iter().enumerate() {
                        let opt_embs = model.get_embedding_batch(g.options.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; g.options.len()]);
                        let mut s_sum = 0.0f32;
                        let mut r_sum = 0.0f32;
                        let mut cnt = 0usize;
                        for oe in &opt_embs {
                            if oe.iter().all(|&v| v == 0.0) { continue; }
                            let mut best_k = 0.0f32;
                            for (_, kb) in &key_banks {
                                let s = max_pool_sim(oe, kb);
                                if s > best_k { best_k = s; }
                            }
                            s_sum += best_k;
                            r_sum += max_pool_sim(oe, &rival_bank);
                            cnt += 1;
                        }
                        if cnt == 0 { continue; }
                        let s_mean = s_sum / (cnt as f32);
                        let r_mean = r_sum / (cnt as f32);

                        let role_emb = model.get_embedding(g.role_phrase.clone()).await.unwrap_or(vec![0.0; 384]);
                        let mut role_status = 0.0f32;
                        for (_, kb) in &key_banks {
                            let s = max_pool_sim(&role_emb, kb);
                            if s > role_status { role_status = s; }
                        }
                        let role_rival = max_pool_sim(&role_emb, &rival_bank);
                        let contrast = (s_mean - r_mean) + 0.5 * (role_status - role_rival);

                        emit_term(&format!("      🎛️ [SELECT CANDIDATE] '{}' | Role: '{}' | Options: {} | StatusMean: {:.4} | RivalMean: {:.4} | RoleΔ: {:+.4} | Contrast: {:+.4}",
                            g.selector, g.role_phrase, g.options.len(), s_mean, r_mean, role_status - role_rival, contrast));
                        scored.push((gi, contrast));
                    }

                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    let mut chosen: Option<usize> = None;
                    if let Some((gi, c1)) = scored.first().copied() {
                        let c2 = scored.get(1).map(|x| x.1).unwrap_or(f32::MIN);
                        let margin = if c2 == f32::MIN { c1 } else { c1 - c2 };
                        if c1 > 0.02 && margin > 0.02 {
                            chosen = Some(gi);
                            emit_term(&format!("  🎛️ [ENUM SELECT COSINE] 상태 컨트롤 확정: '{}' | Contrast: {:+.4} | Margin: {:+.4}",
                                select_groups[gi].selector, c1, margin));
                        } else {
                            emit_term(&format!("  ⚠️ [ENUM SELECT AMBIGUOUS] 최고 Contrast {:+.4} / Margin {:+.4} 로 코사인 확정 실패. LLM CSS selector 탐색으로 넘어갑니다.", c1, margin));
                        }
                    }

                    // ④ 코사인이 애매할 때만 LLM 에게 selector 를 묻습니다. (문서당 최대 1회)
                    if chosen.is_none() {
                        let catalogue: Vec<serde_json::Value> = select_groups.iter().map(|g| json!({
                            "selector": g.selector,
                            "role": g.role_phrase,
                            "options": g.options
                        })).collect();
                        let cat_str = serde_json::to_string_pretty(&catalogue).unwrap_or_default();
                        let sel_prompt = crate::parsing::extract_status_selector_prompt(&page_type, &doc_lang, &cat_str);

                        let params = ChatCompletionParameters {
                            messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                                content: ChatCompletionRequestUserMessageContent::Text(sel_prompt),
                                name: None,
                            })],
                            model: "qwen3.5".to_string(),
                            max_tokens: Some(128),
                            temperature: Some(0.0),
                            top_p: Some(0.95),
                            ..Default::default()
                        };

                        model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancellation_token.clone()), false, kv_name.clone()).await?;
                        let mut picked = String::new();
                        if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
                            if let Ok(res) = gen.generate(params, Some(cancellation_token.clone()), Some(format!("{}_status_selector", task.id)), kv_name.clone(), None, None).await {
                                let parsed = crate::parsing::parse_json_from_llm(&res);
                                picked = parsed.get("status_selector").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
                            }
                        }
                        model.deep_purge_resources().await;
                        model.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancellation_token.clone()), false, Some("inference".to_string())).await?;

                        if !picked.is_empty() && picked != "null" {
                            if let Some(pos) = select_groups.iter().position(|g| g.selector == picked) {
                                chosen = Some(pos);
                                emit_term(&format!("  🤖 [ENUM SELECT LLM] LLM 이 상태 컨트롤로 '{}' 를 지목했습니다.", picked));
                            } else {
                                emit_term(&format!("  🚫 [ENUM SELECT LLM REJECT] LLM 이 반환한 '{}' 는 실제 후보 목록에 없어 폐기합니다.", picked));
                            }
                        }
                    }

                    // ③ selected 옵션 텍스트 → 캐노니컬 키 환산
                    if let Some(gi) = chosen {
                        let g = &select_groups[gi];
                        let sel_emb = model.get_embedding(g.selected.clone()).await.unwrap_or(vec![0.0; 384]);
                        let mut best_key = String::new();
                        let mut best = f32::MIN;
                        let mut second = f32::MIN;
                        for (k, kb) in &key_banks {
                            let s = max_pool_sim(&sel_emb, kb);
                            emit_term(&format!("      🧭 [STATUS KEY] '{}' ← selected \"{}\" | MaxPool: {:.4}", k, g.selected, s));
                            if s > best { second = best; best = s; best_key = k.clone(); }
                            else if s > second { second = s; }
                        }
                        if !best_key.is_empty() && best > 0.35 && (best - second) > 0.01 {
                            enum_resolved.insert("status".to_string(), best_key.clone());
                            emit_term(&format!("  ✅ [ENUM SELECT RESOLVED] '{}' (selected: \"{}\") → status = '{}' | Top: {:.4} | Margin: {:+.4}",
                                g.selector, g.selected, best_key, best, best - second));
                        } else {
                            emit_term(&format!("  ⚠️ [ENUM SELECT UNRESOLVED] selected \"{}\" 의 캐노니컬 마진 부족 (Top {:.4} / 2nd {:.4}). 기존 경로로 위임합니다.",
                                g.selected, best, second));
                        }
                    }
                }
            }

            // 🌟 상태가 결정론적으로 확정되면 구조적 페어의 enum 강제 배정은 폐기합니다.
            if enum_resolved.contains_key("status") {
                header_forced_assign.remove("status");
            }

            let pre_mapped_context = if !pre_mapped_hints.is_empty() {
                serde_json::to_string_pretty(&pre_mapped_hints).unwrap_or_default()
            } else {
                String::new()
            };

            let mut global_ignore_list: Vec<String> = Vec::new();

            // 🌟 [EXCLUSIVE VECTOR ASSIGNMENT + ROLE GATE + FORMAT GATE + DOUBLE CENTERING]
            //    디테일 경로도 리스트와 동일하게 "역할 검증 → 형식 검증 → 이중 센터링 → 배타 배정 → 유일후보 폴백" 순으로 처리합니다.
            let (mut vector_assignment, vector_raw_matrix): (Vec<Option<(usize, f32, f32)>>, Vec<Vec<f32>>) = {
                let line_count = pug_lines_ref.len();
                let field_count = field_phrase_embs.len();
                let mut raw = vec![vec![-1.0f32; line_count]; field_count];

                for f in 0..field_count {
                    if field_is_analytic[f] { continue; }
                    // 🌟 [ID/LINK VECTOR SKIP] id,link 는 '추출 주소 우선' 결정론 해석기가 이미 확정했습니다.
                    //    벡터 배정을 남겨두면 운송장 input(Line 75) 같은 Identifier 형식 라인을 선점해
                    //    tracking_number 를 ⛔ FORMAT SKIP 으로 몰아냅니다.
                    if is_id_link_field(&fields[f].0) && det_id_link.is_some() { continue; }
                    let fmt = field_formats[f];
                    for l in 0..line_count {
                        if pug_lines_ref[l].trim().is_empty() { continue; }
                        if line_embeddings[l].iter().all(|&v| v == 0.0) { continue; }
                        if det_consumed_lines.contains(&l) { continue; }
                        // 🌟 제목/컨테이너 라인은 어떤 필드의 값도 될 수 없습니다.
                        if line_is_non_value[l] { continue; }
                        // 🌟 Enum 은 구조적 페어(강제 배정)로 확정하고, 벡터 폴백은 option[selected] 만 허용합니다.
                        //    ('| 환불, 반품완료 후' 같은 안내 문구가 status 로 배정되던 사고 차단)
                        if fmt == FieldFormat::Enum && !line_is_selected_option[l] { continue; }

                        let value = &line_values[l];
                        let format_ok = match fmt {
                            FieldFormat::Identifier | FieldFormat::Link => value_token_in_url_pool(value, &url_pool),
                            _ => value_matches_format(fmt, value),
                        };
                        if !format_ok { continue; }

                        let own = weighted_max_pool_sim(
                            &line_embeddings[l],
                            &field_phrase_embs[f],
                            &field_phrase_weights[f],
                        );

                        // 🌟 Enum 폴백은 편견 대비 절대 우위(0.15)를 넘어야만 후보로 인정합니다.
                        if fmt == FieldFormat::Enum {
                            let prej = if field_prej_phrase_embs[f].is_empty() {
                                0.0
                            } else {
                                max_pool_sim(&line_embeddings[l], &field_prej_phrase_embs[f])
                            };
                            if own - prej < 0.15 { continue; }
                        }

                        raw[f][l] = own;
                    }
                }

                let centered = double_center_matrix(&raw);
                let mut assign = exclusive_assign(&centered, 0.0, 0.005);

                let mut claimed = vec![false; line_count];
                for a in assign.iter() {
                    if let Some((l, _, _)) = a { claimed[*l] = true; }
                }
                for f in 0..field_count {
                    if assign[f].is_some() { continue; }
                    if field_is_analytic[f] { continue; }
                    let cands: Vec<usize> = (0..line_count)
                        .filter(|&l| raw[f][l] >= 0.0 && !claimed[l])
                        .collect();
                    if cands.len() == 1 {
                        let l = cands[0];
                        assign[f] = Some((l, centered[f][l], 0.0));
                        claimed[l] = true;
                    }
                }

                (assign, raw)
            };

            // 🌟 [PAIR OVERRIDE] enum 계열은 구조적 페어가 확정한 라인을 벡터 배정보다 우선합니다.
            for (f_i, (fname, _, _, _)) in fields.iter().enumerate() {
                if let Some(l) = header_forced_assign.get(fname) {
                    let raw = vector_raw_matrix.get(f_i).and_then(|r| r.get(*l)).copied().unwrap_or(0.0).max(0.0);
                    vector_assignment[f_i] = Some((*l, raw, 0.0));
                    emit_term(&format!("  🧷 [PAIR OVERRIDE] '{}' 의 벡터 배정을 구조적 페어 확정 라인(Line {})으로 교체했습니다.", fname, *l + 1));
                }
            }

            // 🌟 [FORMAT FAMILY SHARE] 같은 형식(FieldFormat)의 필드는 물리적으로 같은 셀을 가리킬 수 있습니다.
            //    order_date 와 registration_date 는 둘 다 '주문일시' 셀이 정답인데,
            //    1:1 배타 배정 때문에 한쪽이 반드시 빈값이 되고 ⛔ FORMAT SKIP 으로 폐기되었습니다.
            //    배정에서 밀려난 필드는 '같은 형식으로 이미 확정된 라인'을 공유하도록 허용합니다.
            {
                let mut shared = 0usize;
                for f in 0..vector_assignment.len() {
                    if vector_assignment[f].is_some() { continue; }
                    if field_is_analytic[f] { continue; }
                    if is_id_link_field(&fields[f].0) { continue; }
                    let fmt = field_formats[f];
                    if !matches!(fmt, FieldFormat::Date | FieldFormat::TrackingCode | FieldFormat::Numeric) { continue; }
                    let mut best_line: Option<usize> = None;
                    let mut best_raw = f32::MIN;
                    for other in 0..vector_assignment.len() {
                        if other == f { continue; }
                        if field_formats[other] != fmt { continue; }
                        // 🌟 [SOURCE RESOLUTION] PRE-MAP'd 필드는 DETAIL PAIR 라인을,
                        //    그 외에는 벡터 배정 라인을 소스로 사용합니다.
                        //    PRE-MAP이 있으면 벡터 배정은 무의미하므로(필드 루프에서 BYPASS됨)
                        //    벡터 배정 라인을 소스로 쓰면 오염됩니다.
                        let source_line: Option<usize> = if let Some(&pl) = pair_line_map.get(&fields[other].0) {
                            Some(pl)
                        } else if let Some((l, _, _)) = vector_assignment[other] {
                            Some(l)
                        } else {
                            None
                        };
                        if let Some(l) = source_line {
                            // 🌟 [VALUE FORMAT GATE] 공유 전 대상 필드 형식과 값의 생김새를 검증합니다.
                            //    "010-3333-3333"을 Date로 공유하는 사고를 여기서 최종 차단합니다.
                            if l < line_values.len() && !value_matches_format(fmt, &line_values[l]) {
                                continue;
                            }
                            let raw = vector_raw_matrix[f].get(l).copied().unwrap_or(0.0);
                            if raw > best_raw { best_raw = raw; best_line = Some(l); }
                        }
                    }
                    if let Some(l) = best_line {
                        vector_assignment[f] = Some((l, best_raw, 0.0));
                        shared += 1;
                        emit_term(&format!("  ♻️ [FORMAT FAMILY SHARE] '{}' ({:?}) ← Line {} | RawSim: {:.4} | 같은 형식 필드가 확정한 라인을 공유합니다.",
                            fields[f].0, fmt, l + 1, best_raw));
                    }
                }
                if shared > 0 {
                    emit_term(&format!("  ♻️ [FORMAT FAMILY SHARE] 총 {}개 필드가 동일 형식 라인을 공유했습니다.", shared));
                }
            }

            for (f_i, (fname, _, _, _)) in fields.iter().enumerate() {
                match vector_assignment[f_i] {
                    Some((l, contrast, margin)) => {
                        emit_term(&format!("  🔗 [EXCLUSIVE ASSIGN] '{}' ({:?}) ← Line {} | RawSim: {:.4} | Contrast: {:+.4} | Margin: {:+.4} | \"{}\"", fname, field_formats[f_i], l + 1, vector_raw_matrix[f_i][l], contrast, margin, pug_lines_ref[l].trim()));
                    },
                    None => {
                        if !field_is_analytic[f_i] {
                            let cand_cnt = vector_raw_matrix[f_i].iter().filter(|&&v| v >= 0.0).count();
                            emit_term(&format!("  ⚪ [UNASSIGNED] '{}' ({:?}) | 형식 통과 후보 {}개 | 벡터 힌트 미주입", fname, field_formats[f_i], cand_cnt));
                        }
                    }
                }
            }


            for (idx, (field_name, field_desc, bias_target, prejudice_target)) in fields.into_iter().enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

                // 🌟 [ENUM DETERMINISTIC BYPASS] select 옵션 집합 코사인(필요시 LLM selector)으로
                //    이미 캐노니컬 키가 확정된 열거형은 LLM 을 호출하지 않고 그대로 확정합니다.
                //    '취소' 라는 단어를 문자열로 찾는 로직은 여기서 완전히 사라집니다.
                if let Some(canon) = enum_resolved.get(&field_name).cloned() {
                    extracted_data.as_object_mut().unwrap().insert(field_name.clone(), json!(canon.clone()));
                    if !global_ignore_list.contains(&canon) {
                        global_ignore_list.push(canon.clone());
                        global_ignore_list.push(format!(" {}", canon));
                        global_ignore_list.push(canon.to_lowercase());
                    }
                    emit_term(&format!("  ⚡ [ENUM BYPASS] LLM 없이 확정: \"{}\": \"{}\"", field_name, canon));
                    continue;
                }

                let keys: Vec<&str> = field_name.split(',').map(|s| s.trim()).collect();
                let mut bypassed_values: Vec<(String, String)> = Vec::new();
                for k in &keys {
                    for hint in &pre_mapped_hints {
                        if let Some(t_col) = hint.get("target_column").and_then(|v| v.as_str()) {
                            if t_col == *k {
                                if let Some(e_val) = hint.get("extracted_value").and_then(|v| v.as_str()) {
                                    let clean_e_val = e_val.trim();
                                    if !clean_e_val.is_empty() {
                                        if let Some(existing) = bypassed_values.iter_mut().find(|(key, _)| key == *k) {
                                            if !existing.1.contains(clean_e_val) {
                                                existing.1.push_str(" ");
                                                existing.1.push_str(clean_e_val);
                                            }
                                        } else {
                                            bypassed_values.push((k.to_string(), clean_e_val.to_string()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if !bypassed_values.is_empty() {
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

                    let mut extracted_results = Vec::new();
                    for (k, val_str) in bypassed_values {
                        extracted_data.as_object_mut().unwrap().insert(k.clone(), json!(val_str));
                        extracted_results.push(format!("\"{}\": \"{}\"", k, val_str));
                        
                        if val_str.len() >= 5 && val_str != "null" && val_str != "true" && val_str != "false" {
                            if !global_ignore_list.contains(&val_str) {
                                global_ignore_list.push(val_str.clone());
                                global_ignore_list.push(format!(" {}", val_str));
                                global_ignore_list.push(val_str.to_lowercase());
                            }
                        }
                    }
                    emit_term(&format!("    ⚡ [PRE-MAP BYPASS] Successfully mapped without LLM: {}", extracted_results.join(", ")));
                    continue;
                }

                let field_format = field_formats[idx];

                // 🌟 [ID/LINK BYPASS] href 안에 실제로 존재하는 토큰만 id 로, 그 href 를 link 로 확정합니다.
                if is_id_link_field(&field_name) {
                    if let Some((det_id, det_link)) = det_id_link.clone() {
                        extracted_data.as_object_mut().unwrap().insert("id".to_string(), json!(det_id.clone()));
                        extracted_data.as_object_mut().unwrap().insert("link".to_string(), json!(det_link.clone()));
                        if !global_ignore_list.contains(&det_id) {
                            global_ignore_list.push(det_id.clone());
                            global_ignore_list.push(format!(" {}", det_id));
                            global_ignore_list.push(det_id.to_lowercase());
                        }
                        emit_term(&format!("  ⚡ [ID/LINK BYPASS] LLM 없이 확정: \"id\": \"{}\", \"link\": \"{}\"", det_id, det_link));
                        continue;
                    }
                }

                let (_bias_emb, _prej_emb, dynamic_prej_str) = &field_embeddings[idx];

                // 🌟 위에서 이미 배타적으로 확정된 배정 결과만 사용합니다. (필드별 독립 argmax 폐기)
                let (best_idx, best_contrast, best_margin, has_vector_match) = match vector_assignment[idx] {
                    Some((l, contrast, margin)) => (l, contrast, margin, true),
                    None => (0usize, 0.0f32, 0.0f32, false),
                };
                let best_raw = if has_vector_match { vector_raw_matrix[idx][best_idx] } else { 0.0f32 };

                // 🌟 [STRICT FORMAT SKIP] 형식이 확정적인 필드는 후보 셀이 없으면 LLM 호출 없이 비워둡니다.
                //    🌟 Enum 도 포함합니다. 구조적 페어도, option[selected] 폴백도 없으면
                //    쓰레기를 넣느니 비워 두는 것이 안전합니다. (card = "2323" 사고 차단)
                // 🌟 Phone 추가: 전화번호는 생김새가 100% 확정적이므로, 후보가 없으면
                //    이메일/네비링크를 억지로 물려 3회 환각시키느니 빈 값이 안전합니다.
                //    Address 는 다국어 단일토큰 주소가 존재할 수 있어 strict 에서 제외하고
                //    전체 컨텍스트 LLM 폴백을 남겨둡니다.
                let strict_format_field = matches!(
                    field_format,
                    FieldFormat::Date | FieldFormat::TrackingCode | FieldFormat::Numeric
                        | FieldFormat::Identifier | FieldFormat::Link | FieldFormat::Enum
                        | FieldFormat::Phone
                );
                if !field_is_analytic[idx] && strict_format_field && !has_vector_match {
                    emit_term(&format!("  ⛔ [FORMAT SKIP] Field: '{}' ({:?}) | 형식에 맞는 후보 셀이 문서에 존재하지 않습니다. LLM 호출 없이 빈 값으로 확정.", field_name, field_format));
                    continue;
                }

                // 🌟 [DATE REGEX BYPASS] 날짜는 생김새가 100% 확정적이므로 0.6B 모델에게 맡기지 않습니다.
                //    로그의 registration_date 3회 연속 환각("td | 04-19:36:51 (수)", "04-193651", "tr")을 원천 차단합니다.
                if !field_is_analytic[idx] && field_format == FieldFormat::Date && has_vector_match {
                    if let Some(date_literal) = extract_date_literal(&line_values[best_idx]) {
                        let keys: Vec<&str> = field_name.split(',').map(|s| s.trim()).collect();
                        let mut done = Vec::new();
                        for k in &keys {
                            extracted_data.as_object_mut().unwrap().insert(k.to_string(), json!(date_literal.clone()));
                            done.push(format!("\"{}\": \"{}\"", k, date_literal));
                        }
                        if !global_ignore_list.contains(&date_literal) {
                            global_ignore_list.push(date_literal.clone());
                            global_ignore_list.push(format!(" {}", date_literal));
                            global_ignore_list.push(date_literal.to_lowercase());
                        }
                        emit_term(&format!("  ⚡ [DATE REGEX BYPASS] LLM 없이 확정: {}", done.join(", ")));
                        continue;
                    }
                }

                // 🌟 [VALUE COPY BYPASS] 형식 게이트가 이미 '이 값은 이 형식이 맞다'고 판정한 라인을
                //    배타적으로 지목했다면, 0.6B 에게 '한 글자도 틀리지 말고 복사하라'고 부탁할 이유가 없습니다.
                //    로그 근거: recipient_phone 은 컨텍스트가 [td, "010-3333-3333"] 로 정답이었는데
                //    모델이 'td' → '10-3278' 을 반환해 3회 만에 폐기되었습니다.
                //    복사는 코드가 하고, LLM 은 Text/Enum/합성 필드에만 남깁니다.
                if !field_is_analytic[idx] && has_vector_match {
                    let copyable = matches!(
                        field_format,
                        FieldFormat::Phone | FieldFormat::Address | FieldFormat::TrackingCode | FieldFormat::Numeric
                    );
                    if copyable {
                        let raw_val = line_values[best_idx].trim().to_string();
                        if !raw_val.is_empty() && value_matches_format(field_format, &raw_val) {
                            let keys: Vec<&str> = field_name.split(',').map(|s| s.trim()).collect();
                            let mut done = Vec::new();
                            for k in &keys {
                                extracted_data.as_object_mut().unwrap().insert(k.to_string(), json!(raw_val.clone()));
                                done.push(format!("\"{}\": \"{}\"", k, raw_val));
                            }
                            if !global_ignore_list.contains(&raw_val) {
                                global_ignore_list.push(raw_val.clone());
                                global_ignore_list.push(format!(" {}", raw_val));
                                global_ignore_list.push(raw_val.to_lowercase());
                            }
                            emit_term(&format!("  ⚡ [VALUE COPY BYPASS] ({:?}) LLM 없이 Line {} 값 그대로 확정: {}",
                                field_format, best_idx + 1, done.join(", ")));
                            continue;
                        }
                    }
                }

                let targeted_pug = if field_is_analytic[idx] {
                    emit_term(&format!("  🧠 [SYNTHESIS FIELD] Field: '{}' | 단일 라인 환원 불가 → 전체 컨텍스트 요약 모드", field_name));
                    content_pug.clone()
                } else if !has_vector_match {
                    emit_term(&format!("  ⚠️ [NO CONFIDENT MATCH] Field: '{}' ({:?}) | 형식 통과 후보 부족 → 전체 컨텍스트만 사용하고 벡터 힌트는 주입하지 않습니다.", field_name, field_format));
                    content_pug.clone()
                } else {
                    emit_term(&format!("  🎯 [EXCLUSIVE MATCH] Field: '{}' ({:?}) | Line: {} | RawSim: {:.4} | Contrast: {:+.4} | Margin: {:+.4}", field_name, field_format, best_idx + 1, best_raw, best_contrast, best_margin));
                    other_extract_pug_context(&pug_lines_ref, best_idx)
                };

                let mut json_contexts = Vec::new();
                for line in targeted_pug.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() { continue; }
                    if let Some(idx) = trimmed.find('|') {
                        let meta = trimmed[..idx].trim();

                        let clean_meta = meta.split('[').next().unwrap_or(meta).trim();
                        json_contexts.push(json!({
                            "metadata": clean_meta,
                            "value": trimmed[idx + 1..].trim()
                        }));
                    } else {
                        json_contexts.push(json!({
                            "value": trimmed
                        }));
                    }
                }
                let targeted_json_context = serde_json::to_string_pretty(&json_contexts).unwrap_or_default();
                
                emit_term(&format!("  🎯 [MATCHED CONTEXT] Field: '{}' ({:?}) | RawSim: {:.4} | Contrast: {:+.4} | Margin: {:+.4}\n{}", field_name, field_format, best_raw, best_contrast, best_margin, targeted_json_context));

                let mut final_context_str = format!("[JSON CONTEXT]\n{}", targeted_json_context);
                if field_is_analytic[idx] {
                    final_context_str.push_str("\n\n[SYNTHESIS FIELD NOTICE]\nThis field is NOT a value to copy. Read the WHOLE [JSON CONTEXT] above and write ONE short sentence that summarizes it. Never return a single cell value such as a bare number, a status word, a person name, or a branch name. If [JSON CONTEXT] is empty, return null.");
                } else if has_vector_match {
                    let matched_line = pug_lines_ref[best_idx].trim();
                    final_context_str.push_str(&format!("\n\n[VECTOR MATCH RESULT]\nThe format gate and the embedding model EXCLUSIVELY assigned this field to the single line below (RawSim {:.4}, Contrast {:+.4}, Margin {:+.4}). No other column may use this line.\nThe part BEFORE '|' is the column LABEL, the part AFTER '|' is the VALUE. Copy ONLY the value part, character for character. Do NOT copy the label. If that value does not fit the schema, return null.\n\"{}\"", best_raw, best_contrast, best_margin, matched_line));
                    if !pre_mapped_context.is_empty() {
                        final_context_str.push_str(&format!("\n\n[ALREADY CLAIMED VALUES]\nThese values are already assigned to OTHER columns. You MUST NOT return any of them for this field:\n{}", pre_mapped_context));
                    }
                } else if !pre_mapped_context.is_empty() {
                    final_context_str.push_str(&format!("\n\n[ALREADY CLAIMED VALUES]\nThese values are already assigned to OTHER columns. You MUST NOT return any of them for this field. If nothing else in [JSON CONTEXT] fits this field, return null:\n{}", pre_mapped_context));
                }

                let system_message = ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: final_context_str,
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


                let mut metadata_str = String::new();
                let mut target_data_str = String::new();
                for line in targeted_pug.lines() {
                    if let Some(idx) = line.find('|') {
                        metadata_str.push_str(line[..idx].trim());
                        metadata_str.push_str("\n");
                        target_data_str.push_str(line[idx + 1..].trim());
                        target_data_str.push_str("\n");
                    } else {
                        target_data_str.push_str(line.trim());
                        target_data_str.push_str("\n");
                    }
                }
                let metadata_str = metadata_str.trim();
                let target_data_str = target_data_str.trim();

                let task_question = if field_name.contains("status") {
                    parsing::extract_status_intent_legacy_prompt(&targeted_pug, &page_type, &bias_target)
                } else if field_is_analytic[idx] {
                    parsing::extract_synthesis_field_prompt(&page_type, &field_name, &field_desc, &doc_lang, target_data_str)
                } else {
                    parsing::extract_single_field_prompt(&page_type, &field_name, &field_desc, language, metadata_str, target_data_str)
                };
                

                let mut ignore_list: Vec<String> = global_ignore_list.clone();
                let mut miss_counter = 0;
                
                loop {
                    if cancellation_token.load(Ordering::Relaxed) { break; }

                    let q3_gen = model.qwen3_generator.clone();
                    let cancel_clone = cancellation_token.clone();
                    let sys_msg = system_message.clone();
                    
                    let field_name_clone = field_name.clone();
                    let bias_target_for_closure = bias_target.clone(); 
                    let prejudice_target_for_closure = dynamic_prej_str.clone();
                    
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
                            

                            let mut item_val = if let Some(inner) = parsed.get_mut(&page_type) { inner.take() } else { parsed };

                            // 🌟 [MARKUP STRIP] "td | 24120419364235" 처럼 태그 접두어가 붙어 돌아온 답변에서
                            //    실제 값만 남깁니다. (0.6B 모델이 [VECTOR MATCH RESULT] 라인을 통째로 복사하는 습성 보정)
                            if let Some(obj) = item_val.as_object_mut() {
                                let ks: Vec<String> = obj.keys().cloned().collect();
                                for k in ks {
                                    let cleaned = match obj.get(&k) {
                                        Some(serde_json::Value::String(s)) => Some(strip_markup_prefix(s)),
                                        _ => None,
                                    };
                                    if let Some(c) = cleaned {
                                        obj.insert(k, json!(c));
                                    }
                                }
                            }

                            let mut requires_retry = false;
                            let mut extracted_values_for_retry = Vec::new();
                            
                            let keys: Vec<&str> = field_name_clone.split(',').map(|s| s.trim()).collect();
                            let mut found_valid_value = false;


                            let skip_pug_match_fields = ["status", "payment_method", "payment_origin", "condition", "currency"];
                            // 🌟 insight/summary/analysis 계열은 '합성 문장'이라 PUG 원문에 리터럴로 존재할 수 없습니다.
                            let synthesis_fields = ["insight", "summary", "analysis"];
                            let field_name_lower = field_name_clone.to_lowercase();
                            let is_synthesis_field = synthesis_fields.iter().any(|&f| field_name_lower.contains(f));
                            let is_enum_field = is_synthesis_field || skip_pug_match_fields.iter().any(|&f| field_name_clone.contains(f));

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
                                        let extracted_str = if val.is_string() {
                                            val.as_str().unwrap_or("").trim().to_string()
                                        } else if val.is_number() {
                                            val.to_string()
                                        } else {
                                            String::new()
                                        };

                                        // 🌟 [POST FORMAT VALIDATION] 형식이 확정적인 키는 반환값의 생김새를 재검증합니다.
                                        //    🌟 Enum 도 포함하여 "tr", "td" 같은 마크업 잔재를 즉시 폐기합니다.
                                        let key_fmt = detect_field_format(k);
                                        let strict_post = matches!(
                                            key_fmt,
                                            FieldFormat::Date | FieldFormat::TrackingCode | FieldFormat::Text
                                                | FieldFormat::Numeric | FieldFormat::Enum | FieldFormat::Identifier
                                                | FieldFormat::Phone | FieldFormat::Address
                                        );
                                        if strict_post && !extracted_str.is_empty() && !value_matches_format(key_fmt, &extracted_str) {
                                            emit_term(&format!("  🚫 [FORMAT REJECT] '{}' ({:?}) 에 형식 불일치 값 '{}' 반환. 폐기 후 재시도합니다.", k, key_fmt, extracted_str));
                                            requires_retry = true;
                                            extracted_values_for_retry.push(extracted_str.clone());
                                            continue;
                                        }

                                        found_valid_value = true;

                                        if !extracted_str.is_empty() && extracted_str != "..." && extracted_str != "null" {
                                            extracted_values_for_retry.push(extracted_str.clone());
                                            
                                            if !is_enum_field {
                                                let is_iso_date = extracted_str.contains('T') && extracted_str.len() >= 19;
                                                let is_url = extracted_str.starts_with("http") || extracted_str.starts_with('/');
                                                let is_boolean_str = extracted_str == "true" || extracted_str == "false";
                                                
                                                if !is_iso_date && !is_url && !is_boolean_str {
                                                    let mut is_matched = doc_title.contains(&extracted_str);
                                                    
                                                    if !is_matched {
                                                        let extracted_lower = extracted_str.to_lowercase();
                                                        let digits_only: String = extracted_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                        
                                                        for ctx_val in &json_contexts {
                                                            if let Some(target_val_str) = ctx_val.get("value").and_then(|v| v.as_str()) {
                                                                let target_lower = target_val_str.to_lowercase();
                                                                
                                                                if target_lower.contains(&extracted_lower) {
                                                                    if digits_only.len() > 0 && digits_only.len() < 3 && extracted_str.len() == digits_only.len() {
                                                                        let tokens: Vec<&str> = target_lower.split(|c: char| !c.is_alphanumeric()).collect();
                                                                        if tokens.contains(&extracted_lower.as_str()) {
                                                                            is_matched = true;
                                                                            break;
                                                                        }
                                                                    } else {
                                                                        is_matched = true;
                                                                        break;
                                                                    }
                                                                }
                                                                
                                                                if !is_matched && digits_only.len() >= 3 {
                                                                    let target_digits: String = target_val_str.chars().filter(|c| c.is_ascii_digit()).collect();
                                                                    if target_digits.contains(&digits_only) {
                                                                        is_matched = true;
                                                                        break;
                                                                    }
                                                                }
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
                                    emit_term(&format!("  ⏭️ Skipping field {} due to persistent hallucination or empty value.", field_name_clone));
                                    break; 
                                }
                                emit_term(&format!("  ⚠️ Hallucination or empty value detected for field {}. Retrying... ({}/3)", field_name_clone, miss_counter));
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
                                if let Some(val) = item_val.get(*k) {
                                    extracted_data.as_object_mut().unwrap().insert(k.to_string(), val.clone());
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
                            


                            for ck in ["has_header", "has_footer", "language"] {
                                if let Some(val) = item_val.get(ck) {
                                    extracted_data.as_object_mut().unwrap().insert(ck.to_string(), val.clone());
                                }
                            }

                            if !extracted_results.is_empty() {
                                emit_term(&format!("  ✅ Extracted: {}", extracted_results.join(", ")));
                            } else {
                                emit_term(&format!("  ✅ Extracted: (null or empty for {})", field_name_clone));
                            }
                            break;
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

    
    let search_mode_str = search_mode.clone();
    let doc_lang_str = doc_lang.clone();
    let normalize_data = |item: &mut serde_json::Value| {
        if let Some(obj) = item.as_object_mut() {
            if obj.get("type").is_none() { obj.insert("type".to_string(), json!(page_type.clone())); }
            
            if obj.get("mode").is_none() { obj.insert("mode".to_string(), json!(search_mode_str.clone())); }
            

            let currency_val = obj.get("currency").and_then(|v| v.as_str()).unwrap_or("").trim();
            if currency_val.is_empty() || currency_val == "null" {
                let default_currency = match doc_lang_str.as_str() {
                    "ko" => "KRW",
                    "ja" => "JPY",
                    "zh" | "zh-tw" | "zh-hk" | "zh-hans" => "CNY",
                    "de" | "fr" | "it" | "es" | "nl" | "pt" | "el" => "EUR",
                    "ru" => "RUB",
                    "th" => "THB",
                    "vi" => "VND",
                    "hi" | "bn" => "INR",
                    "en" | _ => "USD",
                };
                obj.insert("currency".to_string(), json!(default_currency));
            } else {
                obj.insert("currency".to_string(), json!(currency_val.to_uppercase()));
            }
            

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


                            if s.contains('T') && s.len() >= 19 {
                                continue;
                            }


                            let nums: Vec<u32> = re_date.find_iter(s).filter_map(|m| m.as_str().parse().ok()).collect();
                            if nums.len() >= 3 {
                                let mut year = nums[0];
                                let mut month = nums[1];
                                let mut day = nums[2];


                                if day > 31 && year <= 31 {
                                    year = nums[2];
                                    day = nums[1];
                                    month = nums[0];
                                }


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

                        let ts_ms = if date_num < 10_000_000_000 { date_num * 1000 } else { date_num };
                        if let Some(dt) = chrono::DateTime::from_timestamp_millis(ts_ms).map(|dt| dt.naive_utc()) {
                            let iso_date = dt.format("%Y-%m-%dT%H:%M:%S").to_string();
                            obj.insert(key.to_string(), json!(iso_date));
                        }
                    }
                }
            }


            if obj.get("started_at").is_none() || obj.get("started_at").unwrap().is_null() || obj.get("started_at").unwrap().as_str() == Some("") {
                if let Some(m) = obj.get("manufacture_date").cloned() { obj.insert("started_at".to_string(), m); }
            }
            if obj.get("expired_at").is_none() || obj.get("expired_at").unwrap().is_null() || obj.get("expired_at").unwrap().as_str() == Some("") {
                if let Some(e) = obj.get("expiration_date").cloned() { obj.insert("expired_at".to_string(), e); }
            }
            

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
        

        let should_mask = page_type != "goods";

        if is_detail {
            let original_lang_text = parsing::json_to_natural_language(&extracted_data);
            

            let masked_lang_text = original_lang_text.clone();

            if let Some(obj) = extracted_data.as_object_mut() {
                obj.insert("text".to_string(), json!(original_lang_text));
                obj.insert("masked_text".to_string(), json!(masked_lang_text));
            }
        } else {
            if let Some(items) = extracted_data.get_mut("items").and_then(|v| v.as_array_mut()) {
                for item in items.iter_mut() {
                    let original_lang_text = parsing::json_to_natural_language(item);
                    

                    let masked_lang_text = original_lang_text.clone();

                    if let Some(obj) = item.as_object_mut() {
                        obj.insert("text".to_string(), json!(original_lang_text));
                        obj.insert("masked_text".to_string(), json!(masked_lang_text));
                    }
                }
            }
        }
    }

    {
        println!("[Scheduler] PHASE 3: Handover - Unloading, Preparing for Embedding...");
        
        log_task_progress(app_handle, &task.id, &json!({ "category": "Handover", "summary": "Switching to Embedding model...", "spinner": "⠋" }));
        

        model.deep_purge_resources().await;
        
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        

        crate::utils::resources::wait_for_resources_settled(1200, 800, Some(cancellation_token), model.device_config.gpu_id as u32).await?;
    }

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
                        obj.insert("order".to_string(), json!(index_val));
                    }
                    
                    
                    let tracking_text = parsing::json_to_natural_language(&tracking_data);
                    let masked_tracking_text = tracking_text.clone();
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
        

        let text_to_embed = extracted_data.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| parsing::json_to_natural_language(&extracted_data));
        let item_digest = crate::utils::hash::digest(&text_to_embed); 
        let mut target_id = generated_id.clone(); 
        
        let mut existing_vector = None;
        let mut is_new = true;
        let mut was_draft = false;

        
        if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &target_id).await {
            is_new = false;
            // 🌟 [CRITICAL FIX] index.ts와 동일하게 "items" 테이블의 updated_at을 기준으로 draft 판정.
            // target_table(sales)의 updated_at_ts가 0이더라도, items 테이블에서 이미
            // updated_at이 설정되어 있으면(이전에 상세 스캔된 적 있음) draft가 아님.
            // 또한 update_team_base_metrics가 items_to_process를 스캔하여 draft/count를
            // 자동 계산하므로, 여기서 수동으로 stats_diff를 적용하면 이중 카운트 발생.
            let mut items_doc_updated_at: i64 = -1; // -1 = items 테이블에서 못 찾음
            if let Ok(Some(items_doc)) = store.get_item_by_id("items", &target_id).await {
                items_doc_updated_at = items_doc.updated_at_ts;
            }
            was_draft = if items_doc_updated_at == 0 {
                true // items 테이블의 updated_at이 0 → 아직 상세 스캔 안됨 (draft 상태)
            } else {
                false // items 테이블의 updated_at > 0 또는 항목 없음 → 이미 상세 스캔됨
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
                    let mut items_doc_updated_at: i64 = -1;
                    if let Ok(Some(items_doc)) = store.get_item_by_id("items", &target_id).await {
                        items_doc_updated_at = items_doc.updated_at_ts;
                    }
                    was_draft = if items_doc_updated_at == 0 {
                        true
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

        if is_new {
            let e = stats_diff.entry(page_type.clone()).or_insert((0, 0, 0));
            e.1 += 1;
            e.2 += 1;
        } else if was_draft {
            // 🌟 [CRITICAL FIX] update_team_base_metrics가 items_to_process를 스캔하여
            // updated_at > 0인 항목을 자동으로 count로 분류합니다.
            // 여기서 수동으로 draft--, count++를 적용하면 update_team_base_metrics의
            // 자동 분류와 겹쳐 이중 카운트가 발생합니다.
            // extracted_data의 updated_at을 now로 설정하는 것만으로 충분합니다.
            // (was_draft 판정은 updated_at 업데이트 트리거로만 사용)
            if let Some(obj) = extracted_data.as_object_mut() {
                obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
            }
        }

        let vector = if let Some(v) = existing_vector {
            Some(v)
        } else {
            Some(model.get_embedding(text_to_embed).await?)
        };

        // 🌟 [INDEX.TS PARITY] 릴레이 실행 전에 tracking_number로부터 tracking index를 사전 계산합니다.
        // index.ts에서는 item.tracking = crc32(hashId('tracking'+team.id+tracking_number))를
        // 릴레이 전에 설정하여 릴레이가 기존 항목을 정확히 찾을 수 있도록 합니다.
        if page_type == "order" {
            if let Some(tn_raw) = extracted_data.get("tracking_number").and_then(|v| v.as_str()) {
                if !tn_raw.trim().is_empty() {
                    let clean_tn_pre = crate::utils::hash::normalize_numeric_homoglyphs(tn_raw)
                        .replace("-", "").replace("_", "").replace(".", "").replace(",", "");
                    if !clean_tn_pre.is_empty() {
                        let tracking_index_pre = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("tracking{}{}", team_id, clean_tn_pre)));
                        if let Some(obj) = extracted_data.as_object_mut() {
                            obj.insert("tracking".to_string(), json!(tracking_index_pre));
                        }
                        emit_term(&format!("  🔑 [TRACKING INDEX PRE-COMPUTE] tracking_number '{}' → tracking index {} 사전 설정 완료.", clean_tn_pre, tracking_index_pre));
                    }
                }
            }
        }

        let related_types = crate::logic::related(&page_type);
        for foreign_type in related_types {
            if let Some((queries, merge_rule)) = crate::logic::relay(foreign_type, &extracted_data) {
                for q in queries {
                    match store.find_item_by_property(&q.table, &q.column, &q.value).await {
                        Ok(Some((foreign_id, mut foreign_data))) => {
                            let was_foreign_draft = foreign_data.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(0) == 0;
                            let mut needs_update = false;


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


                            if needs_update {
                                if was_foreign_draft && merge_rule.update.as_ref().map_or(false, |u| u.to == foreign_type) {
                                    let e = stats_diff.entry(foreign_type.to_string()).or_insert((0, 0, 0));
                                    e.0 -= 1;
                                    e.1 += 1;
                                    
                                    e.2 += 1;
                                    foreign_data.as_object_mut().unwrap().insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                }
                                let merged_text = parsing::json_to_natural_language(&foreign_data);
                                let masked_merged_text = merged_text.clone();
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
                            // 🌟 [DEDUP FIX] 상세 페이지 relay에서 기존 목록 전처리에서 이미 생성된
                            // goods/tracking이 있는지 "items" 테이블에서 교차 검색합니다.
                            let mut found_existing = false;
                            let val_str_for_search = match &q.value {
                                serde_json::Value::String(s) => s.clone(),
                                serde_json::Value::Number(n) => n.to_string(),
                                _ => q.value.to_string(),
                            };
                            if !val_str_for_search.is_empty() {
                                let cross_filter = format!("type = '{}' AND json_data LIKE '%{}%'", foreign_type, val_str_for_search.replace("'", "''"));
                                if let Ok(cross_results) = store.get_all_items("items", 1, 0, Some(cross_filter)).await {
                                    if !cross_results.is_empty() {
                                        found_existing = true;
                                        emit_term(&format!("  🔄 [RELAY DEDUP] 기존 {} 문서 발견 (값: '{}'). 새 draft 생성을 건너뜁니다.", foreign_type, val_str_for_search));
                                    }
                                }
                            }

                            // 🌟 [ORDER INDEX FALLBACK] goods/tracking relay가 tracking 컬럼으로 못 찾았을 때,
                            // order index로도 검색합니다. 목록 스캔에서 생성된 항목에는 tracking 값이 없지만
                            // order index는 있으므로 이 폴백으로 기존 항목을 찾을 수 있습니다.
                            // index.ts의 relay가 column: primary.type, value: primary.index로 쿼리하는 동작과 동일합니다.
                            if !found_existing && (foreign_type == "goods" || foreign_type == "tracking") {
                                if let Some(order_idx) = extracted_data.get("index") {
                                    let order_idx_str = match order_idx {
                                        serde_json::Value::Number(n) => n.to_string(),
                                        serde_json::Value::String(s) => s.clone(),
                                        _ => order_idx.to_string(),
                                    };
                                    let fallback_filter = format!("type = '{}' AND json_data LIKE '%\"order\":{}%'", foreign_type, order_idx_str);
                                    if let Ok(fallback_results) = store.get_all_items("items", 1, 0, Some(fallback_filter)).await {
                                        if !fallback_results.is_empty() {
                                            found_existing = true;
                                            emit_term(&format!("  🔄 [RELAY ORDER-INDEX FALLBACK] order index {}로 기존 {} 문서 발견. 새 draft 생성을 건너뜁니다.", order_idx_str, foreign_type));
                                        }
                                    }
                                }
                            }

                            if !found_existing {
                                let e = stats_diff.entry(foreign_type.to_string()).or_insert((0, 0, 0));
                                e.0 += 1;
                                e.2 += 1;

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
                                    obj.insert("updated_at".to_string(), json!(0));
                                }
                                let _ = store.upsert_item(
                                    &q.table, &draft_id, foreign_type, draft_data.clone(), None,
                                    Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                ).await;
                                let _ = store.upsert_item(
                                    "items", &draft_id, foreign_type, draft_data, None,
                                    Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                ).await;
                            }
                        },
                        _ => {}
                    }
                }
            }
        }


        if page_type == "order" {
            if let Some(tn_raw) = extracted_data.get("tracking_number").and_then(|v| v.as_str()) {
                if !tn_raw.trim().is_empty() {
                    let clean_tn = crate::utils::hash::normalize_numeric_homoglyphs(tn_raw)
                        .replace("-", "").replace("_", "");
                    if !clean_tn.is_empty() {
                        emit_term(&format!("  📦 [TRACKING RELAY] order 전처리에서 tracking_number '{}' 감지. tracking 테이블 역방향 쿼리 시작...", clean_tn));
                        match store.find_item_by_property("tracking", "tracking_number", &json!(clean_tn)).await {
                            Ok(Some((tracking_id, mut tracking_data))) => {

                                let was_foreign_draft = tracking_data.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(0) == 0;
                                let mut needs_update = false;

                                for field in ["width", "height", "length", "weight"] {
                                    if let Some(val) = extracted_data.get(field).cloned() {
                                        let existing = tracking_data.get(field).and_then(|v| v.as_f64()).unwrap_or(0.0);
                                        if existing == 0.0 {
                                            tracking_data.as_object_mut().unwrap().insert(field.to_string(), val);
                                            needs_update = true;
                                        }
                                    }
                                }

                                if let Some(order_index) = extracted_data.get("index") {
                                    if tracking_data.get("order").is_none() || tracking_data.get("order") == Some(&json!(0)) {
                                        tracking_data.as_object_mut().unwrap().insert("order".to_string(), order_index.clone());
                                        needs_update = true;
                                    }
                                }

                                if let Some(tracking_index) = tracking_data.get("index").cloned() {
                                    if extracted_data.get("tracking").is_none() || extracted_data.get("tracking") == Some(&json!(0)) {
                                        extracted_data.as_object_mut().unwrap().insert("tracking".to_string(), tracking_index);
                                    }
                                }
                                if needs_update {
                                    if was_foreign_draft {
                                        let e = stats_diff.entry("tracking".to_string()).or_insert((0, 0, 0));
                                        e.0 -= 1;
                                        e.1 += 1;
                                        e.2 += 1;
                                        tracking_data.as_object_mut().unwrap().insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                    }
                                    let merged_text = parsing::json_to_natural_language(&tracking_data);
                                    let masked_merged_text = merged_text.clone();
                                    let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 384]);
                                    tracking_data.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text));
                                    tracking_data.as_object_mut().unwrap().insert("masked_text".to_string(), json!(masked_merged_text));
                                    let _ = store.upsert_item(
                                        "tracking", &tracking_id, "tracking", tracking_data.clone(), Some(merged_vector.clone()),
                                        Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                    ).await;
                                    let _ = store.upsert_item(
                                        "items", &tracking_id, "tracking", tracking_data, Some(merged_vector),
                                        Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                    ).await;
                                    emit_term(&format!("  ✅ [TRACKING RELAY] 기존 tracking 문서 '{}'에 order.index 매핑 완료.", tracking_id));
                                }
                            },
                            Ok(None) => {
                                // 🌟 [DEDUP FIX] tracking_number로 items 테이블에서 기존 tracking 문서 검색
                                let mut found_existing_tracking = false;
                                let tracking_cross_filter = format!("type = 'tracking' AND json_data LIKE '%{}%'", clean_tn.replace("'", "''"));
                                if let Ok(tracking_cross) = store.get_all_items("items", 1, 0, Some(tracking_cross_filter)).await {
                                    if !tracking_cross.is_empty() {
                                        found_existing_tracking = true;
                                        let existing_tracking_id = &tracking_cross[0].id;
                                        // 기존 tracking 문서에 order index만 매핑
                                        if let Ok(Some(mut existing_data)) = store.get_item_by_id("tracking", existing_tracking_id).await {
                                            if let Ok(mut ej) = serde_json::from_str::<serde_json::Value>(&existing_data.json_data) {
                                                if ej.get("order").is_none() || ej.get("order") == Some(&json!(0)) {
                                                    if let Some(order_index) = extracted_data.get("index") {
                                                        ej.as_object_mut().unwrap().insert("order".to_string(), order_index.clone());
                                                    }
                                                    if let Some(tn_val) = extracted_data.get("tracking") {
                                                        ej.as_object_mut().unwrap().insert("tracking".to_string(), tn_val.clone());
                                                    }
                                                    ej.as_object_mut().unwrap().insert("tracking_number".to_string(), json!(clean_tn.clone()));
                                                    ej.as_object_mut().unwrap().insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                                    let merged_text = crate::parsing::json_to_natural_language(&ej);
                                                    let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 384]);
                                                    ej.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text));
                                                    ej.as_object_mut().unwrap().insert("masked_text".to_string(), json!(merged_text.clone()));
                                                    let _ = store.upsert_item(
                                                        "tracking", existing_tracking_id, "tracking", ej.clone(), Some(merged_vector.clone()),
                                                        Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                                    ).await;
                                                    let _ = store.upsert_item(
                                                        "items", existing_tracking_id, "tracking", ej.clone(), Some(merged_vector),
                                                        Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                                    ).await;
                                                }
                                                if let Some(tracking_index) = ej.get("index").cloned() {
                                                    extracted_data.as_object_mut().unwrap().insert("tracking".to_string(), tracking_index);
                                                }
                                            }
                                        }
                                        emit_term(&format!("  🔄 [TRACKING RELAY DEDUP] 기존 tracking 문서 '{}' 재사용 (tracking_number: {}). 새 draft 생성 건너뜀.", existing_tracking_id, clean_tn));
                                    }
                                }

                                // 🌟 [ORDER INDEX FALLBACK] tracking_number로 못 찾았으면 order index로도 검색합니다.
                                // 목록 스캔에서 생성된 tracking 항목에는 tracking_number가 없지만 order index는 있습니다.
                                // index.ts의 relay("tracking", "order")가 column: primary.type, value: primary.index로
                                // 쿼리하는 것과 동일한 로직입니다.
                                if !found_existing_tracking {
                                    if let Some(order_index_val) = extracted_data.get("index") {
                                        match store.find_item_by_property("tracking", "order", order_index_val).await {
                                            Ok(Some((fallback_tid, mut fallback_tdata))) => {
                                                found_existing_tracking = true;
                                                let was_fb_draft = fallback_tdata.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(0) == 0;
                                                if let Some(obj) = fallback_tdata.as_object_mut() {
                                                    obj.insert("tracking_number".to_string(), json!(clean_tn.clone()));
                                                    if let Some(tn_idx) = extracted_data.get("tracking") {
                                                        obj.insert("tracking".to_string(), tn_idx.clone());
                                                    }
                                                    obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                                }
                                                if was_fb_draft {
                                                    let e = stats_diff.entry("tracking".to_string()).or_insert((0, 0, 0));
                                                    e.0 -= 1;
                                                    e.1 += 1;
                                                }
                                                let merged_text = crate::parsing::json_to_natural_language(&fallback_tdata);
                                                let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 384]);
                                                fallback_tdata.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text));
                                                fallback_tdata.as_object_mut().unwrap().insert("masked_text".to_string(), json!(merged_text.clone()));
                                                let _ = store.upsert_item(
                                                    "tracking", &fallback_tid, "tracking", fallback_tdata.clone(), Some(merged_vector.clone()),
                                                    Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                                ).await;
                                                let _ = store.upsert_item(
                                                    "items", &fallback_tid, "tracking", fallback_tdata.clone(), Some(merged_vector),
                                                    Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                                ).await;
                                                if let Some(fb_tracking_index) = fallback_tdata.get("index").cloned() {
                                                    extracted_data.as_object_mut().unwrap().insert("tracking".to_string(), fb_tracking_index);
                                                }
                                                emit_term(&format!("  🔄 [TRACKING RELAY ORDER-INDEX FALLBACK] order index로 기존 tracking 문서 '{}' 발견. tracking_number '{}' 매핑 완료. 새 draft 생성 건너뜀.", fallback_tid, clean_tn));
                                            },
                                            _ => {}
                                        }
                                    }
                                }

                                if !found_existing_tracking {
                                    let e = stats_diff.entry("tracking".to_string()).or_insert((0, 0, 0));
                                    e.0 += 1;
                                    e.2 += 1;
                                    let tracking_index = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("tracking{}{}", team_id, clean_tn)));
                                    let draft_id = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, "tracking", clean_tn));
                                    let mut draft_data = json!({});
                                    if let Some(obj) = draft_data.as_object_mut() {
                                        obj.insert("id".to_string(), json!(draft_id.clone()));
                                        obj.insert("type".to_string(), json!("tracking"));
                                        obj.insert("tracking_number".to_string(), json!(clean_tn.clone()));
                                        obj.insert("index".to_string(), json!(tracking_index));
                                        if let Some(order_index) = extracted_data.get("index") {
                                            obj.insert("order".to_string(), order_index.clone());
                                        }
                                        obj.insert("updated_at".to_string(), json!(0));
                                    }
                                    extracted_data.as_object_mut().unwrap().insert("tracking".to_string(), json!(tracking_index));
                                    let _ = store.upsert_item(
                                        "tracking", &draft_id, "tracking", draft_data.clone(), None,
                                        Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                    ).await;
                                    let _ = store.upsert_item(
                                        "items", &draft_id, "tracking", draft_data, None,
                                        Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                    ).await;
                                    emit_term(&format!("  📝 [TRACKING RELAY] tracking draft '{}' 생성 (tracking_number: {}).", draft_id, clean_tn));
                                }
                            },
                            _ => {}
                        }
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

        // =====================================================================
        // 🌟 [PHASE A~D] 청크 단위 인덱싱 파이프라인 (상세 페이지)
        // ---------------------------------------------------------------------
        // 기존: json_to_natural_language() → 단일 긴 문장 → items.text 컬럼 1개
        // 변경: json_to_natural_language() → 문장 분할 → 속성 태깅 → 임베딩 → item_chunks 저장
        //
        // 이 블록이 있어야 검색 시 "무거운" ↔ "weight is 1.5" 코사인 매칭이 가능해집니다.
        // =====================================================================
        {
            let natural_text = crate::nl_convert::json_to_natural_language(&extracted_data);

            // PHASE A: 문장 단위 분할 + 로그 출력
            let raw_chunks = crate::nl_convert::split_natural_language_to_chunks(&natural_text);
            emit_term(&format!("  📝 [PHASE A] RAW-CHUNK 분할 결과: {}개 청크", raw_chunks.len()));
            for (ci, (ct, cp, confirmed)) in raw_chunks.iter().enumerate() {
                let flag = if *confirmed { "✓" } else { "?" };
                emit_term(&format!("    [{}] {} property='{}' | text='{}'", ci, flag, cp, ct));
            }

            if !raw_chunks.is_empty() {
                // ── 필드 뱅크 구축 (PLINKO GAME 입력) ──
                // bias.json 에서 필드명 + bias 구 임베딩 + 형식을 추출합니다.
                let fields = crate::parsing::get_detail_schema_fields(&page_type, &url, &doc_lang);
                let mut idx_field_names: Vec<String> = Vec::new();
                let mut idx_field_phrase_embs: Vec<Vec<Vec<f32>>> = Vec::new();
                let mut idx_field_phrase_weights: Vec<Vec<f32>> = Vec::new();
                let mut idx_field_formats: Vec<String> = Vec::new();

                for (fname, _, bias_target, _) in &fields {
                    // 🌟 [SYNTHESIS BANK INCLUDE] 역방향 PLINKO 는 확인 모드(경쟁 없음)이므로
                    //    insight 계열을 뱅크에서 빼도 배정이 달라지지 않습니다.
                    //    반대로 빼면 field_names 에 없어져 origin_score 가 항상 0.0000 이 되고
                    //    (log.txt: origin='general_insight'(0.0000), origin='traffic_insight'(0.0000))
                    //    CONFIRM FLAG 진단이 전량 오탐으로 오염됩니다.
                    //    format_gate 는 "Synthesis" 를 무조건 통과시키므로 부작용이 없습니다.
                    let lower_fname = fname.to_lowercase();
                    let _is_synthesis = lower_fname.contains("insight")
                        || lower_fname.contains("summary")
                        || lower_fname.contains("analysis");

                    let (mut phrases, mut weights) = crate::utils::ai_utils::split_bias_phrases_weighted_full(bias_target);

                    // 🌟 [ABSTRACT BRIDGE MERGE] 정방향은 '무거운' 을 search_bridge.abstract_bridge 로
                    //    substantial_filters.weight 에 라우팅합니다. 그런데 역방향 필드 뱅크에는
                    //    heavy / weighs a lot 계열 구가 하나도 없어서 그 의도를 받아줄 벡터가
                    //    DB 에 존재하지 않았습니다. 동일 bias.json 노드를 그대로 편입합니다.
                    let bridge_ph = crate::utils::ai_utils::abstract_bridge_field_phrases(fname);
                    if !bridge_ph.is_empty() {
                        emit_term(&format!("  🌉 [ABSTRACT BRIDGE MERGE] '{}' 뱅크에 추상 수식어 브릿지 구 {}개 편입", fname, bridge_ph.len()));
                    }
                    for p in bridge_ph {
                        if phrases.iter().any(|e| e == &p) { continue; }
                        phrases.push(p);
                        weights.push(1.0);
                    }

                    let phrase_embs = if phrases.is_empty() {
                        vec![vec![0.0f32; 384]]
                    } else {
                        model.get_embedding_batch(phrases.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; phrases.len()])
                    };

                    let fmt_str = {
                        let lower = fname.to_lowercase();
                        let keys: Vec<String> = lower.split(',').map(|s| s.trim().to_string()).collect();
                        let has = |k: &str| keys.iter().any(|x| x == k);

                        if keys.iter().any(|k| k.contains("insight") || k.contains("summary") || k.contains("analysis")) {
                            "Synthesis".to_string()
                        } else if keys.iter().any(|k| k.contains("tracking_number") || k == "barcode" || k == "gtin" || k == "mpn") {
                            "TrackingCode".to_string()
                        } else if has("id") || has("code") || has("no") || has("index") || has("stock_keeping_unit") {
                            "Identifier".to_string()
                        } else if keys.iter().any(|k| k.contains("link") || k.contains("url")) {
                            "Link".to_string()
                        } else if keys.iter().any(|k| k.contains("date") || k.ends_with("_at")) {
                            "Date".to_string()
                        } else if keys.iter().any(|k| {
                            k.ends_with("phone") || k == "tel" || k == "telephone" || k == "mobile"
                                || k == "cellphone" || k == "contact" || k == "number"
                        }) {
                            "Phone".to_string()
                        } else if keys.iter().any(|k| k == "address" || k.ends_with("_address")) {
                            "Address".to_string()
                        } else if keys.iter().any(|k| {
                            k.contains("status") || k.contains("payment_method") || k.contains("payment_origin")
                                || k.contains("condition") || k.contains("currency") || k == "bank" || k == "card"
                        }) {
                            "Enum".to_string()
                        } else if keys.iter().any(|k| {
                            k.contains("price") || k.contains("amount") || k.contains("quantity") || k.contains("weight")
                                || k == "width" || k == "height" || k == "length" || k.contains("fee")
                                || k.contains("discount") || k.contains("usage_") || k.contains("threshold")
                                || k.contains("duration")
                        }) {
                            "Numeric".to_string()
                        } else {
                            "Text".to_string()
                        }
                    };

                    idx_field_names.push(fname.clone());
                    idx_field_phrase_embs.push(phrase_embs);
                    idx_field_phrase_weights.push(weights);
                    idx_field_formats.push(fmt_str);
                }

                emit_term(&format!(
                    "  📐 [PHASE B+C] 필드 뱅크 구축 완료: {}개 필드 (PLINKO GAME 입력)",
                    idx_field_names.len()
                ));

                // ── PHASE B+C 통합 파이프라인 (비동기) ──
                // raw_chunks 는 Vec<(String, String, bool)> 타입으로,
                // confirmed 플래그가 PLINKO 확인 모드 / NMS 보호 / 배타 배정 우선순위에 사용됩니다.
                let model_for_embed = model.clone();
                let enriched_chunks = crate::nl_convert::run_phase_b_pipeline(
                    &raw_chunks,
                    &doc_lang,
                    &page_type,
                    &idx_field_names,
                    &idx_field_phrase_embs,
                    &idx_field_phrase_weights,
                    &idx_field_formats,
                    move |text: String| {
                        let m = model_for_embed.clone();
                        async move {
                            m.get_embedding(text).await.unwrap_or(vec![0.0; 384])
                        }
                    },
                ).await;
                crate::nl_convert::log_enriched_chunks(&enriched_chunks);

                if !enriched_chunks.is_empty() {
                    // ── PHASE D: 임베딩 생성 ──
                    let indexable_chunks: Vec<(usize, &crate::nl_convert::ChunkMetadata)> = enriched_chunks.iter()
                        .enumerate()
                        .filter(|(_, c)| c.property != "unclassified")
                        .collect();

                    let skipped_count = enriched_chunks.len() - indexable_chunks.len();
                    if skipped_count > 0 {
                        emit_term(&format!(
                            "  🚫 [PHASE D FILTER] unclassified 청크 {}개 인덱싱 제외",
                            skipped_count
                        ));
                    }

                    if indexable_chunks.is_empty() {
                        emit_term("  ⚠️ [PHASE D] 인덱싱 대상 청크가 없습니다. 건너뜁니다.");
                    } else {
                        let chunk_texts: Vec<String> = indexable_chunks.iter()
                            .map(|(_, c)| c.chunk_text.clone())
                            .collect();

                        let chunk_embs = model.get_embedding_batch(chunk_texts.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; chunk_texts.len()]);

                        // ── PHASE E: LanceDB item_chunks 테이블 저장 ──
                        let _ = store.delete_chunks_by_item(&target_id).await;

                        // 🌟 [MULTILINGUAL VALUE BLEND v3] 저장 벡터를 형식 인지 3중 합성으로 만듭니다.
                        //   ① chunk_text : "This goods is titled 'Cable Knit Cardigan'." (영어 자연문 원본)
                        //   ② anchor     : indexing_anchor_text() — 라벨 개념 센트로이드 + 다국어 값 도메인 축
                        //   ③ localized  : "{leaf_label} {value_part}" — 짧은 문서 언어 라벨 1개 + 실제 값
                        //
                        //   🌟 [v2 → v3 변경 이유]
                        //   v2 는 localized 를 "{anchor} {value}" 로 만들었는데, anchor 가
                        //   "상품명, 의류명, 제품명, 품목명, 이름, 상품제목, 상품이름, ..." 10~32구 블롭이라
                        //   값 'Cable Knit Cardigan' 이 30여 토큰 중 3토큰으로 희석되었습니다.
                        //
                        //   🌟 [Enum 그룹 교정] Enum 값은 'complete'/'show'/'hide' 같은
                        //   저카디널리티 캐노니컬 키이고 전 아이템에서 동일 문자열입니다.
                        //   값 지배 그룹에 넣으면 변별력 0인 청크가 한국어 라벨 축으로 상위를 독점합니다.
                        //   (new_log2.txt: property='status' 가 상위 10건 중 9건 점유)
                        //   따라서 Enum 은 라벨 지배 그룹으로 보냅니다.
                        //     Text/Address/Synthesis → 값이 자유 서술이라 의미를 나른다 → localized 지배
                        //     Enum/Numeric/Date/Identifier/Link/Phone/TrackingCode → 값은 리터럴 → 라벨 지배
                        let mut anchor_texts: Vec<String> = Vec::with_capacity(indexable_chunks.len());
                        let mut localized_texts: Vec<String> = Vec::with_capacity(indexable_chunks.len());
                        for (_, cm) in indexable_chunks.iter() {
                            let a = crate::utils::ai_utils::indexing_anchor_text(
                                &doc_lang, &page_type, &cm.property,
                            );
                            let leaf = crate::utils::ai_utils::indexing_leaf_label(
                                &doc_lang, &page_type, &cm.property,
                            );
                            let v = cm.value_part.trim();
                            let l = if v.is_empty() { leaf.clone() } else { format!("{} {}", leaf, v) };
                            anchor_texts.push(a);
                            localized_texts.push(l);
                        }
                        let anchor_embs = model.get_embedding_batch(anchor_texts.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; anchor_texts.len()]);
                        let localized_embs = model.get_embedding_batch(localized_texts.clone()).await
                            .unwrap_or_else(|_| vec![vec![0.0; 384]; localized_texts.len()]);

                        for (ei, (ci, chunk_meta)) in indexable_chunks.iter().enumerate() {
                            let chunk_id = format!("{}_{}", target_id, ci);

                            let chunk_vec = &chunk_embs[ei];
                            let anchor_emb = &anchor_embs[ei];
                            let localized_emb = &localized_embs[ei];

                            // 🌟 [FORMAT-AWARE WEIGHT] 형식이 값의 의미 밀도를 결정합니다.
                            let (w_chunk, w_anchor, w_local) = match chunk_meta.property_format.as_str() {
                                "Text" | "Address" | "Synthesis" => (0.25f32, 0.10f32, 0.65f32),
                                _ => (0.40f32, 0.30f32, 0.30f32),
                            };

                            let mut final_vec = vec![0.0f32; 384];
                            for d in 0..384 {
                                final_vec[d] = chunk_vec[d] * w_chunk
                                    + anchor_emb[d] * w_anchor
                                    + localized_emb[d] * w_local;
                            }
                            // L2 정규화
                            let norm: f32 = final_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                            if norm > 0.0 {
                                for d in 0..384 { final_vec[d] /= norm; }
                            }

                            let _ = store.upsert_chunk(
                                &chunk_id,
                                &target_id,
                                &page_type,
                                &chunk_meta.chunk_text,
                                &chunk_meta.property,
                                &chunk_meta.property_format,
                                &chunk_meta.value_part,
                                Some(final_vec),
                                Some(&task.cc),
                                Some(&bcc),
                                Some(&ref_val),
                                Some(&search_mode),
                            ).await;
                        }

                        emit_term(&format!(
                            "  🧩 [PHASE A~E] 청크 인덱싱 완료: item_id='{}' | 청크 {}개 (전체 {}개 중) | table='item_chunks'",
                            target_id, indexable_chunks.len(), enriched_chunks.len()
                        ));
                    }
                }
            }
        }
        // =====================================================================
        // 🌟 [PHASE A~D 종료]
        // =====================================================================
        
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


                let text_to_embed = single_item.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| parsing::json_to_natural_language(&single_item));
                let item_digest = crate::utils::hash::digest(&text_to_embed);
                
                let mut existing_vector = None;
                let mut is_new = true;


                if let Ok(Some(existing_item)) = store.get_item_by_id(&target_table, &hashed_item_id).await {
                    is_new = false;

                    if existing_item.digest == item_digest {
                        existing_vector = Some(existing_item.vector);
                    }
                }

                if is_new {
                    let e = stats_diff.entry(page_type.clone()).or_insert((0, 0, 0));
                    e.0 += 1;
                    e.2 += 1;
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


                                    if needs_update {
                                        let merged_text = parsing::json_to_natural_language(&foreign_data);
                                        let masked_merged_text = merged_text.clone();
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
                                    // 🌟 [DEDUP FIX] relay draft 생성 전, 동일한 foreign_type + 값이
                                    // 이미 items 테이블에 존재하는지 확인하여 중복 생성을 방지합니다.
                                    let mut found_existing = false;
                                    let val_str_for_search = match &q.value {
                                        serde_json::Value::String(s) => s.clone(),
                                        serde_json::Value::Number(n) => n.to_string(),
                                        _ => q.value.to_string(),
                                    };
                                    if !val_str_for_search.is_empty() {
                                        let cross_filter = format!("type = '{}' AND json_data LIKE '%{}%'", foreign_type, val_str_for_search.replace("'", "''"));
                                        if let Ok(cross_results) = store.get_all_items("items", 1, 0, Some(cross_filter)).await {
                                            if !cross_results.is_empty() {
                                                found_existing = true;
                                                emit_term(&format!("  🔄 [RELAY DEDUP] 기존 {} 문서 발견 (값: '{}'). 새 draft 생성을 건너뜁니다.", foreign_type, val_str_for_search));
                                            }
                                        }
                                    }

                                    // 🌟 [ORDER INDEX FALLBACK] goods/tracking relay가 못 찾았을 때 order index로도 검색
                                    if !found_existing && (foreign_type == "goods" || foreign_type == "tracking") {
                                        if let Some(order_idx) = single_item.get("index") {
                                            let order_idx_str = match order_idx {
                                                serde_json::Value::Number(n) => n.to_string(),
                                                serde_json::Value::String(s) => s.clone(),
                                                _ => order_idx.to_string(),
                                            };
                                            let fallback_filter = format!("type = '{}' AND json_data LIKE '%\"order\":{}%'", foreign_type, order_idx_str);
                                            if let Ok(fallback_results) = store.get_all_items("items", 1, 0, Some(fallback_filter)).await {
                                                if !fallback_results.is_empty() {
                                                    found_existing = true;
                                                    emit_term(&format!("  🔄 [RELAY ORDER-INDEX FALLBACK] order index {}로 기존 {} 문서 발견. 새 draft 생성을 건너뜁니다.", order_idx_str, foreign_type));
                                                }
                                            }
                                        }
                                    }

                                    if !found_existing {
                                        let e = stats_diff.entry(foreign_type.to_string()).or_insert((0, 0, 0));
                                        e.0 += 1;
                                        e.2 += 1;

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
                                            obj.insert("updated_at".to_string(), json!(0));
                                        }
                                        let _ = store.upsert_item(
                                            &q.table, &draft_id, foreign_type, draft_data.clone(), None,
                                            Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                        ).await;
                                        let _ = store.upsert_item(
                                            "items", &draft_id, foreign_type, draft_data, None,
                                            Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                        ).await;
                                    }
                                },
                                _ => {}
                            }
                        }
                    }
                }

                if page_type == "order" {
                    if let Some(tn_raw) = single_item.get("tracking_number").and_then(|v| v.as_str()) {
                        if !tn_raw.trim().is_empty() {
                            let clean_tn = crate::utils::hash::normalize_numeric_homoglyphs(tn_raw)
                                .replace("-", "").replace("_", "");
                            if !clean_tn.is_empty() {
                                emit_term(&format!("  📦 [TRACKING RELAY] order 리스트 아이템에서 tracking_number '{}' 감지. tracking 테이블 역방향 쿼리 시작...", clean_tn));
                                match store.find_item_by_property("tracking", "tracking_number", &json!(clean_tn)).await {
                                    Ok(Some((tracking_id, mut tracking_data))) => {

                                        let was_foreign_draft = tracking_data.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(0) == 0;
                                        let mut needs_update = false;

                                        for field in ["width", "height", "length", "weight"] {
                                            if let Some(val) = single_item.get(field).cloned() {
                                                let existing = tracking_data.get(field).and_then(|v| v.as_f64()).unwrap_or(0.0);
                                                if existing == 0.0 {
                                                    tracking_data.as_object_mut().unwrap().insert(field.to_string(), val);
                                                    needs_update = true;
                                                }
                                            }
                                        }

                                        if let Some(order_index) = single_item.get("index") {
                                            if tracking_data.get("order").is_none() || tracking_data.get("order") == Some(&json!(0)) {
                                                tracking_data.as_object_mut().unwrap().insert("order".to_string(), order_index.clone());
                                                needs_update = true;
                                            }
                                        }

                                        if let Some(tracking_index) = tracking_data.get("index").cloned() {
                                            if single_item.get("tracking").is_none() || single_item.get("tracking") == Some(&json!(0)) {
                                                single_item.as_object_mut().unwrap().insert("tracking".to_string(), tracking_index);
                                            }
                                        }
                                        if needs_update {
                                            if was_foreign_draft {
                                                let e = stats_diff.entry("tracking".to_string()).or_insert((0, 0, 0));
                                                e.0 -= 1;
                                                e.1 += 1;
                                                e.2 += 1;
                                                tracking_data.as_object_mut().unwrap().insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                            }
                                            let merged_text = parsing::json_to_natural_language(&tracking_data);
                                            let masked_merged_text = merged_text.clone();
                                            let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 384]);
                                            tracking_data.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text));
                                            tracking_data.as_object_mut().unwrap().insert("masked_text".to_string(), json!(masked_merged_text));
                                            let _ = store.upsert_item(
                                                "tracking", &tracking_id, "tracking", tracking_data.clone(), Some(merged_vector.clone()),
                                                Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                            ).await;
                                            let _ = store.upsert_item(
                                                "items", &tracking_id, "tracking", tracking_data, Some(merged_vector),
                                                Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                            ).await;
                                            emit_term(&format!("  ✅ [TRACKING RELAY] 기존 tracking 문서 '{}'에 order.index 매핑 완료.", tracking_id));
                                        }
                                    },
                                    Ok(None) => {
                                        // 🌟 [DEDUP FIX] tracking_number로 items 테이블에서 기존 tracking 문서 검색
                                        let mut found_existing_tracking = false;
                                        let tracking_cross_filter = format!("type = 'tracking' AND json_data LIKE '%{}%'", clean_tn.replace("'", "''"));
                                        if let Ok(tracking_cross) = store.get_all_items("items", 1, 0, Some(tracking_cross_filter)).await {
                                            if !tracking_cross.is_empty() {
                                                found_existing_tracking = true;
                                                let existing_tracking_id = &tracking_cross[0].id;
                                                if let Ok(Some(mut existing_data)) = store.get_item_by_id("tracking", existing_tracking_id).await {
                                                    if let Ok(mut ej) = serde_json::from_str::<serde_json::Value>(&existing_data.json_data) {
                                                        if ej.get("order").is_none() || ej.get("order") == Some(&json!(0)) {
                                                            if let Some(order_index) = single_item.get("index") {
                                                                ej.as_object_mut().unwrap().insert("order".to_string(), order_index.clone());
                                                            }
                                                            if let Some(tn_val) = single_item.get("tracking") {
                                                                ej.as_object_mut().unwrap().insert("tracking".to_string(), tn_val.clone());
                                                            }
                                                            ej.as_object_mut().unwrap().insert("tracking_number".to_string(), json!(clean_tn.clone()));
                                                            ej.as_object_mut().unwrap().insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                                            let merged_text = crate::parsing::json_to_natural_language(&ej);
                                                            let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 384]);
                                                            ej.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text));
                                                            ej.as_object_mut().unwrap().insert("masked_text".to_string(), json!(merged_text.clone()));
                                                            let _ = store.upsert_item(
                                                                "tracking", existing_tracking_id, "tracking", ej.clone(), Some(merged_vector.clone()),
                                                                Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                                            ).await;
                                                            let _ = store.upsert_item(
                                                                "items", existing_tracking_id, "tracking", ej.clone(), Some(merged_vector),
                                                                Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                                            ).await;
                                                        }
                                                        if let Some(tracking_index) = ej.get("index").cloned() {
                                                            single_item.as_object_mut().unwrap().insert("tracking".to_string(), tracking_index);
                                                        }
                                                    }
                                                }
                                                emit_term(&format!("  🔄 [TRACKING RELAY DEDUP] 기존 tracking 문서 '{}' 재사용 (tracking_number: {}). 새 draft 생성 건너뜀.", existing_tracking_id, clean_tn));
                                            }
                                        }

                                        // 🌟 [ORDER INDEX FALLBACK] tracking_number로 못 찾았으면 order index로도 검색합니다.
                                        if !found_existing_tracking {
                                            if let Some(order_index_val) = single_item.get("index") {
                                                match store.find_item_by_property("tracking", "order", order_index_val).await {
                                                    Ok(Some((fallback_tid, mut fallback_tdata))) => {
                                                        found_existing_tracking = true;
                                                        let was_fb_draft = fallback_tdata.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(0) == 0;
                                                        if let Some(obj) = fallback_tdata.as_object_mut() {
                                                            obj.insert("tracking_number".to_string(), json!(clean_tn.clone()));
                                                            if let Some(tn_idx) = single_item.get("tracking") {
                                                                obj.insert("tracking".to_string(), tn_idx.clone());
                                                            }
                                                            obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                                        }
                                                        if was_fb_draft {
                                                            let e = stats_diff.entry("tracking".to_string()).or_insert((0, 0, 0));
                                                            e.0 -= 1;
                                                            e.1 += 1;
                                                        }
                                                        let merged_text = crate::parsing::json_to_natural_language(&fallback_tdata);
                                                        let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 384]);
                                                        fallback_tdata.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text));
                                                        fallback_tdata.as_object_mut().unwrap().insert("masked_text".to_string(), json!(merged_text.clone()));
                                                        let _ = store.upsert_item(
                                                            "tracking", &fallback_tid, "tracking", fallback_tdata.clone(), Some(merged_vector.clone()),
                                                            Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                                        ).await;
                                                        let _ = store.upsert_item(
                                                            "items", &fallback_tid, "tracking", fallback_tdata.clone(), Some(merged_vector),
                                                            Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                                        ).await;
                                                        if let Some(fb_tracking_index) = fallback_tdata.get("index").cloned() {
                                                            single_item.as_object_mut().unwrap().insert("tracking".to_string(), fb_tracking_index);
                                                        }
                                                        emit_term(&format!("  🔄 [TRACKING RELAY ORDER-INDEX FALLBACK] order index로 기존 tracking 문서 '{}' 발견. tracking_number '{}' 매핑 완료. 새 draft 생성 건너뜀.", fallback_tid, clean_tn));
                                                    },
                                                    _ => {}
                                                }
                                            }
                                        }

                                        if !found_existing_tracking {
                                            let e = stats_diff.entry("tracking".to_string()).or_insert((0, 0, 0));
                                            e.0 += 1;
                                            e.2 += 1;
                                            let tracking_index = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("tracking{}{}", team_id, clean_tn)));
                                            let draft_id = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, "tracking", clean_tn));
                                            let mut draft_data = json!({});
                                            if let Some(obj) = draft_data.as_object_mut() {
                                                obj.insert("id".to_string(), json!(draft_id.clone()));
                                                obj.insert("type".to_string(), json!("tracking"));
                                                obj.insert("tracking_number".to_string(), json!(clean_tn.clone()));
                                                obj.insert("index".to_string(), json!(tracking_index));
                                                if let Some(order_index) = single_item.get("index") {
                                                    obj.insert("order".to_string(), order_index.clone());
                                                }
                                                obj.insert("updated_at".to_string(), json!(0));
                                            }
                                            single_item.as_object_mut().unwrap().insert("tracking".to_string(), json!(tracking_index));
                                            let _ = store.upsert_item(
                                                "tracking", &draft_id, "tracking", draft_data.clone(), None,
                                                Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                            ).await;
                                            let _ = store.upsert_item(
                                                "items", &draft_id, "tracking", draft_data, None,
                                                Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), None
                                            ).await;
                                            emit_term(&format!("  📝 [TRACKING RELAY] tracking draft '{}' 생성 (tracking_number: {}).", draft_id, clean_tn));
                                        }
                                    },
                                    _ => {}
                                }
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
                items_to_process.push(single_item.clone());

                // =====================================================================
                // 🌟 [PHASE A~D] 청크 단위 인덱싱 파이프라인 (리스트 페이지 개별 아이템)
                // =====================================================================
                {
                    let natural_text = crate::nl_convert::json_to_natural_language(&single_item);

                    // PHASE A: 문장 단위 분할 + 로그 출력
                    let raw_chunks = crate::nl_convert::split_natural_language_to_chunks(&natural_text);
                    emit_term(&format!("  📝 [PHASE A] RAW-CHUNK 분할 결과: {}개 청크", raw_chunks.len()));
                    for (ci, (ct, cp, confirmed)) in raw_chunks.iter().enumerate() {
                        let flag = if *confirmed { "✓" } else { "?" };
                        emit_term(&format!("    [{}] {} property='{}' | text='{}'", ci, flag, cp, ct));
                    }

                    if !raw_chunks.is_empty() {
                        // ── 필드 뱅크 구축 (PLINKO GAME 입력) ──
                        let fields = crate::parsing::get_list_schema_fields(&page_type, &url, &doc_lang);
                        let mut idx_field_names: Vec<String> = Vec::new();
                        let mut idx_field_phrase_embs: Vec<Vec<Vec<f32>>> = Vec::new();
                        let mut idx_field_phrase_weights: Vec<Vec<f32>> = Vec::new();
                        let mut idx_field_formats: Vec<String> = Vec::new();

                        for (fname, _, bias_target, _) in &fields {
                            // 🌟 [SYNTHESIS BANK INCLUDE] 상세 경로와 동일. 확인 모드이므로 배정 영향 없음.
                            let lower_fname = fname.to_lowercase();
                            let _is_synthesis = lower_fname.contains("insight")
                                || lower_fname.contains("summary")
                                || lower_fname.contains("analysis");

                            let (mut phrases, mut weights) = crate::utils::ai_utils::split_bias_phrases_weighted_full(bias_target);

                            // 🌟 [ABSTRACT BRIDGE MERGE] 리스트 경로도 동일하게 추상 수식어 브릿지를 편입합니다.
                            let bridge_ph = crate::utils::ai_utils::abstract_bridge_field_phrases(fname);
                            for p in bridge_ph {
                                if phrases.iter().any(|e| e == &p) { continue; }
                                phrases.push(p);
                                weights.push(1.0);
                            }

                            let phrase_embs = if phrases.is_empty() {
                                vec![vec![0.0f32; 384]]
                            } else {
                                model.get_embedding_batch(phrases.clone()).await
                                    .unwrap_or_else(|_| vec![vec![0.0; 384]; phrases.len()])
                            };

                            let fmt_str = {
                                let lower = fname.to_lowercase();
                                let keys: Vec<String> = lower.split(',').map(|s| s.trim().to_string()).collect();
                                let has = |k: &str| keys.iter().any(|x| x == k);

                                if keys.iter().any(|k| k.contains("insight") || k.contains("summary") || k.contains("analysis")) {
                                    "Synthesis".to_string()
                                } else if keys.iter().any(|k| k.contains("tracking_number") || k == "barcode" || k == "gtin" || k == "mpn") {
                                    "TrackingCode".to_string()
                                } else if has("id") || has("code") || has("no") || has("index") || has("stock_keeping_unit") {
                                    "Identifier".to_string()
                                } else if keys.iter().any(|k| k.contains("link") || k.contains("url")) {
                                    "Link".to_string()
                                } else if keys.iter().any(|k| k.contains("date") || k.ends_with("_at")) {
                                    "Date".to_string()
                                } else if keys.iter().any(|k| {
                                    k.ends_with("phone") || k == "tel" || k == "telephone" || k == "mobile"
                                        || k == "cellphone" || k == "contact" || k == "number"
                                }) {
                                    "Phone".to_string()
                                } else if keys.iter().any(|k| k == "address" || k.ends_with("_address")) {
                                    "Address".to_string()
                                } else if keys.iter().any(|k| {
                                    k.contains("status") || k.contains("payment_method") || k.contains("payment_origin")
                                        || k.contains("condition") || k.contains("currency") || k == "bank" || k == "card"
                                }) {
                                    "Enum".to_string()
                                } else if keys.iter().any(|k| {
                                    k.contains("price") || k.contains("amount") || k.contains("quantity") || k.contains("weight")
                                        || k == "width" || k == "height" || k == "length" || k.contains("fee")
                                        || k.contains("discount") || k.contains("usage_") || k.contains("threshold")
                                        || k.contains("duration")
                                }) {
                                    "Numeric".to_string()
                                } else {
                                    "Text".to_string()
                                }
                            };

                            idx_field_names.push(fname.clone());
                            idx_field_phrase_embs.push(phrase_embs);
                            idx_field_phrase_weights.push(weights);
                            idx_field_formats.push(fmt_str);
                        }

                        // ── PHASE B+C 통합 파이프라인 (비동기) ──
                        // raw_chunks: Vec<(String, String, bool)> — confirmed 플래그 포함
                        // 전처리 경로에서는 JSON 구조 패턴 매칭으로 확정된 청크가
                        // PLINKO 확인 모드에서 슬라이딩 윈도우를 건너뛰고 기존 property 를 유지합니다.
                        let model_for_embed = model.clone();
                        let enriched_chunks = crate::nl_convert::run_phase_b_pipeline(
                            &raw_chunks,
                            &doc_lang,
                            &page_type,
                            &idx_field_names,
                            &idx_field_phrase_embs,
                            &idx_field_phrase_weights,
                            &idx_field_formats,
                            move |text: String| {
                                let m = model_for_embed.clone();
                                async move {
                                    m.get_embedding(text).await.unwrap_or(vec![0.0; 384])
                                }
                            },
                        ).await;

                        if !enriched_chunks.is_empty() {
                            // ── PHASE D: 임베딩 생성 ──
                            let indexable_chunks: Vec<(usize, &crate::nl_convert::ChunkMetadata)> = enriched_chunks.iter()
                                .enumerate()
                                .filter(|(_, c)| c.property != "unclassified")
                                .collect();

                            let skipped_count = enriched_chunks.len() - indexable_chunks.len();
                            if skipped_count > 0 {
                                emit_term(&format!(
                                    "  🚫 [PHASE D FILTER] unclassified 청크 {}개 인덱싱 제외",
                                    skipped_count
                                ));
                            }

                            if indexable_chunks.is_empty() {
                                emit_term("  ⚠️ [PHASE D] 인덱싱 대상 청크가 없습니다. 건너뜁니다.");
                            } else {
                                let chunk_texts: Vec<String> = indexable_chunks.iter()
                                    .map(|(_, c)| c.chunk_text.clone())
                                    .collect();

                                let chunk_embs = model.get_embedding_batch(chunk_texts.clone()).await
                                    .unwrap_or_else(|_| vec![vec![0.0; 384]; chunk_texts.len()]);

                                // ── PHASE E: LanceDB item_chunks 테이블 저장 ──
                                let _ = store.delete_chunks_by_item(&hashed_item_id).await;

                                // 🌟 [MULTILINGUAL VALUE BLEND v3] 상세 경로와 동일한 형식 인지 3중 합성.
                                //    localized 를 "{leaf_label} {value}" 로 축약하여 값이 지배하게 하고,
                                //    Enum 은 라벨 지배 그룹으로 보내 저변별 청크의 상위 독점을 차단합니다.
                                let mut anchor_texts: Vec<String> = Vec::with_capacity(indexable_chunks.len());
                                let mut localized_texts: Vec<String> = Vec::with_capacity(indexable_chunks.len());
                                for (_, cm) in indexable_chunks.iter() {
                                    let a = crate::utils::ai_utils::indexing_anchor_text(
                                        &doc_lang, &page_type, &cm.property,
                                    );
                                    let leaf = crate::utils::ai_utils::indexing_leaf_label(
                                        &doc_lang, &page_type, &cm.property,
                                    );
                                    let v = cm.value_part.trim();
                                    let l = if v.is_empty() { leaf.clone() } else { format!("{} {}", leaf, v) };
                                    anchor_texts.push(a);
                                    localized_texts.push(l);
                                }
                                let anchor_embs = model.get_embedding_batch(anchor_texts.clone()).await
                                    .unwrap_or_else(|_| vec![vec![0.0; 384]; anchor_texts.len()]);
                                let localized_embs = model.get_embedding_batch(localized_texts.clone()).await
                                    .unwrap_or_else(|_| vec![vec![0.0; 384]; localized_texts.len()]);

                                for (ei, (ci, chunk_meta)) in indexable_chunks.iter().enumerate() {
                                    let chunk_id = format!("{}_{}", hashed_item_id, ci);

                                    let chunk_vec = &chunk_embs[ei];
                                    let anchor_emb = &anchor_embs[ei];
                                    let localized_emb = &localized_embs[ei];

                                    let (w_chunk, w_anchor, w_local) = match chunk_meta.property_format.as_str() {
                                        "Text" | "Address" | "Synthesis" => (0.25f32, 0.10f32, 0.65f32),
                                        _ => (0.40f32, 0.30f32, 0.30f32),
                                    };

                                    let mut final_vec = vec![0.0f32; 384];
                                    for d in 0..384 {
                                        final_vec[d] = chunk_vec[d] * w_chunk
                                            + anchor_emb[d] * w_anchor
                                            + localized_emb[d] * w_local;
                                    }
                                    let norm: f32 = final_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                                    if norm > 0.0 {
                                        for d in 0..384 { final_vec[d] /= norm; }
                                    }

                                    let _ = store.upsert_chunk(
                                        &chunk_id,
                                        &hashed_item_id,
                                        &page_type,
                                        &chunk_meta.chunk_text,
                                        &chunk_meta.property,
                                        &chunk_meta.property_format,
                                        &chunk_meta.value_part,
                                        Some(final_vec),
                                        Some(&task.cc),
                                        Some(&bcc),
                                        Some(&ref_val),
                                        Some(&search_mode),
                                    ).await;
                                }

                                emit_term(&format!(
                                    "  🧩 [PHASE A~E] 청크 인덱싱 완료: item_id='{}' | 청크 {}개 (전체 {}개 중)",
                                    hashed_item_id, indexable_chunks.len(), enriched_chunks.len()
                                ));
                            }
                        }
                    }
                }
                // =====================================================================
                // 🌟 [PHASE A~D 종료]
                // =====================================================================
            }
        }
    }

    if !items_to_process.is_empty() {
        let _ = crate::utils::metrics::update_team_base_metrics(&store, &team_id, &task.cc, &items_to_process, stats_diff.clone()).await;
        println!("[PROCESS] Metrics Engine updated base statistics for {} items. (Stats Diff: {:?})", items_to_process.len(), stats_diff);
    }

    let _ = store.update_message_status(&task.id, logic::parse_status("complete"), Some("Extraction Complete")).await;
 
    let payload = json!({
        "task_id": task.id, 
        "category": "Done", 
        "summary": "Extraction complete. Updating list...", 
        "spinner": "✅",

        "data": null 
    });
    
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);
    
    println!("[PROCESS] Task {} completed. Handover to Embedding finished.", task.id);
    Ok(())
}
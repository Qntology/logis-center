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
        let filename = format!("{}_{}_{}.txt", self.task_id, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_micros(), suffix);
        let path = dir.join(filename);
        fs::write(&path, content)?;
        self.created_files.push(path.clone());
        Ok(path)
    }

    fn load(&self, path: &std::path::Path) -> Result<String> {
        Ok(fs::read_to_string(path)?)
    }
}

impl Drop for TaskDataManager {
    fn drop(&mut self) {
        println!("[Cleanup] TaskDataManager dropping: {}", self.task_id);
    }
}

fn chunk_text(text: &str, target_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    while start < text.len() {
        let mut end = (start + target_size).min(text.len());
        if end < text.len() {
            let mut temp_end = end;
            while temp_end > start && bytes[temp_end] != b'\n' { temp_end -= 1; }
            if temp_end > start { end = temp_end + 1; }
            else { while end < text.len() && !text.is_char_boundary(end) { end += 1; } }
        } else { end = text.len(); }
        let slice = &text[start..end];
        if !slice.trim().is_empty() { chunks.push(slice.to_string()); }
        start = end;
    }
    chunks
}

fn merge_json_results(target: &mut Value, source: &Value) {
    if let (Some(target_obj), Some(source_obj)) = (target.as_object_mut(), source.as_object()) {
        for (k, v) in source_obj {
            if v.is_null() { continue; }
            if let Some(s) = v.as_str() { if s.is_empty() { continue; } }
            if let Some(a) = v.as_array() { if a.is_empty() { continue; } }
            
            let should_update = match target_obj.get(k) {
                None => true,
                Some(tv) => tv.is_null() || (tv.is_string() && tv.as_str() == Some("")) || (tv.is_array() && tv.as_array().unwrap().is_empty())
            };

            if should_update { target_obj.insert(k.clone(), v.clone()); }
            else if let Some(target_inner) = target_obj.get_mut(k) {
                if target_inner.is_object() && v.is_object() { merge_json_results(target_inner, v); }
                else if let (Some(ta), Some(sa)) = (target_inner.as_array_mut(), v.as_array()) { ta.extend(sa.clone()); }
            }
        }
    }
}

use tokio::sync::Notify;
use once_cell::sync::Lazy;

static UI_READY_SIGNAL: Lazy<Notify> = Lazy::new(|| Notify::new());
static UI_READY_FLAG: AtomicBool = AtomicBool::new(false);

pub fn mark_ui_ready() {
    UI_READY_FLAG.store(true, Ordering::SeqCst);
    UI_READY_SIGNAL.notify_waiters();
}

pub async fn start_background_worker(
    store: Arc<Mutex<Option<VectorStore>>>,
    model: Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: Arc<AtomicBool>,
    app_handle: tauri::AppHandle,
) {
    println!("[Scheduler] Background worker waiting for UI Ready signal...");
    clear_all_temp_data(Some(&app_handle));
    
    tokio::spawn(async move {
        if !UI_READY_FLAG.load(Ordering::SeqCst) { UI_READY_SIGNAL.notified().await; }
        let mut delay_secs = 1;
        loop {
            sleep(Duration::from_secs(delay_secs)).await;
            let mut pending_tasks = Vec::new();
            {
                let store_opt = store.lock().await;
                if let Some(db) = store_opt.as_ref() {
                    if let Ok(tasks) = db.get_pending_tasks(5).await { pending_tasks = tasks; }
                }
            }

            if pending_tasks.is_empty() {
                delay_secs = (delay_secs + 1).min(10);
                continue;
            } else { delay_secs = 1; }

            for task in pending_tasks {
                if cancellation_token.load(Ordering::Relaxed) { break; }
                println!("[Scheduler] Processing task: {}", task.id);
                {
                    let store_guard = store.lock().await;
                    if let Some(db) = store_guard.as_ref() {
                        let _ = db.update_task_status(&task.id, crate::logic::parse_status("progress")).await;
                    }
                }

                match process_task(task.clone(), &store, &model, &cancellation_token, &app_handle).await {
                    Ok(_) => {
                        println!("[Scheduler] Task completed: {}", task.id);
                        let store_guard = store.lock().await;
                        if let Some(db) = store_guard.as_ref() {
                            let _ = db.update_task_status(&task.id, crate::logic::parse_status("complete")).await;
                        }
                    },
                    Err(e) => {
                        let err_msg = e.to_string();
                        {
                            let mut model_lock = model.lock().await;
                            if let Some(m) = model_lock.as_ref() { m.unload_generator().await; }
                            *model_lock = None;
                            println!("[Scheduler] Error detected. Emergency release: {}", err_msg);
                        }

                        if err_msg.contains("Task cancelled") {
                             let store_guard = store.lock().await;
                             if let Some(db) = store_guard.as_ref() {
                                 let _ = db.update_task_status(&task.id, crate::logic::parse_status("cancel")).await;
                                 let _ = db.update_message_status(&task.id, crate::logic::parse_status("cancel"), Some("Cancelled by user")).await;
                             }
                             let _ = app_handle.emit("extraction-progress", json!({ "task_id": task.id, "category": "Done", "summary": "Cancelled by user", "spinner": "🛑" }));
                             break; 
                        } else {
                            println!("[Scheduler] Task failed: {}. Error: {}", task.id, err_msg);
                            if err_msg.contains("CUDA_ERROR") || err_msg.contains("out of memory") {
                                let mut model_lock = model.lock().await;
                                if let Some(m) = model_lock.as_ref() { m.unload_generator().await; }
                                *model_lock = None;
                                let _ = app_handle.emit("extraction-progress", json!({ "task_id": task.id, "category": "Error", "summary": "Memory Full", "spinner": "❌" }));
                            }
                            let store_guard = store.lock().await;
                            if let Some(db) = store_guard.as_ref() {
                                let _ = db.update_task_status(&task.id, crate::logic::parse_status("error")).await;
                                let _ = db.update_message_status(&task.id, crate::logic::parse_status("error"), Some(&format!("Error: {}", err_msg))).await;
                            }
                        }
                    }
                }
            }
            cancellation_token.store(false, Ordering::SeqCst);
        }
    });
}

async fn process_task(
    task: Task,
    store_mutex: &Arc<Mutex<Option<VectorStore>>>,
    model_mutex: &Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle
)
-> Result<()> {
    let pug_logs_dir = utils::paths::get_pug_logs_dir(Some(app_handle), &task.id);
    println!("[PROCESS] Task {} started.", task.id);

    cleanup_task_resources(&task.id, Some(app_handle));

    let large_model_path_hint = std::fs::canonicalize("src-tauri/models/Qwen3-VL-2B-Instruct-gguf").or_else(|_| std::fs::canonicalize("models/Qwen3-VL-2B-Instruct-gguf")).ok();
    if let Some(p) = large_model_path_hint { std::thread::spawn(move || { let _ = pre_fetch_weights(&p); }); }

    let mut data_manager = TaskDataManager::new(&task.id, Some(app_handle.clone()));
    let mut task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    let language = "english"; 

    if task.r#type == "image_extraction" {
        let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("").to_string();
        let mut model_lock = model_mutex.lock().await;
        if model_lock.is_none() {
            match LogisModel::new(None).await {
                Ok(m) => *model_lock = Some(m),
                Err(e) => return Err(anyhow::anyhow!("Model Load Failed: {}", e)),
            }
        }
        if let Some(model) = model_lock.as_ref() {
            let res = model.extract_from_image(task.id.clone(), image_path, language.to_string(), app_handle, Some(cancellation_token.clone()), store_mutex).await;
            drop(model_lock);
            return res;
        }
        return Ok(())
    }

    let url = task_data.get("link").and_then(|s| s.as_str()).unwrap_or("").to_string();
    if url.is_empty() { return Ok(()); }

    let raw_html_path = if let Some(raw_html) = task_data.get("html").and_then(|s| s.as_str()) {
        let p = data_manager.offload(raw_html, "raw_html")?;
        if let Some(obj) = task_data.as_object_mut() { obj.remove("html"); }
        p
    } else {
        let content = reqwest::get(&url).await?.text().await?;
        data_manager.offload(&content, "raw_html")?
    };
    
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let clean_html_path = {
        let raw_content = data_manager.load(&raw_html_path)?;
        let clean = parsing::pre_clean_html(&raw_content);
        data_manager.offload(&clean, "clean_html")?
    };
    
    let light_pug = {
        let clean_content = data_manager.load(&clean_html_path)?;
        parsing::convert_to_clean_pug(&clean_content, PugMode::FullContent)
    }; 
    
    let ts_nano = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let _ = std::fs::write(pug_logs_dir.join(format!("light_{}_{}.pug", task.id, ts_nano)), &light_pug);
    
    let store = {
        let store_guard = store_mutex.lock().await;
        store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
    };

    let model = {
        let mut model_lock = model_mutex.lock().await;
        if model_lock.is_none() {
            match LogisModel::new(None).await {
                Ok(m) => *model_lock = Some(m),
                Err(e) => return Err(anyhow::anyhow!("Model Load Failed: {}", e)),
            }
        }
        model_lock.as_ref().unwrap().clone()
    };

    use crate::openai_types::{ChatCompletionParameters, ChatCompletionRequestMessage, ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent};

    // --- STEP 1: BASE BAKING (0.6B) ---
    let _ = app_handle.emit("extraction-progress", json!({ "task_id": task.id, "category": "Analysis", "summary": "Baking context...", "spinner": "⠋" }));
    model.ingest_pug_to_ssd(&task.id, &light_pug, Some(cancellation_token.clone())).await?;
    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled after baking")); }

    // --- STEP 2: LOAD 2B WITH CONTEXT ---
    let _ = app_handle.emit("extraction-progress", json!({ "task_id": task.id, "category": "Loading", "summary": "Loading Reasoner...", "spinner": "⠋" }));
    model.ensure_large_with_base(&task.id, Some(cancellation_token.clone())).await?;
    
    // --- STEP 3: SEQUENTIAL INFERENCE ---
    let mut page_type = String::new();
    let mut selector_info = json!({});
    
    // Q1: Classification
    {
        let type_prompt = parsing::page_type_prompt();
        let params = ChatCompletionParameters {
            messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { content: ChatCompletionRequestUserMessageContent::Text(format!("TASK: {}\n\nACTION: JSON ONLY", type_prompt)), name: None })],
            model: "qwen3vl".to_string(), max_tokens: Some(128), temperature: Some(0.1), ..Default::default()
        };
        if let Some(gen) = model.generator.lock().await.as_mut() {
            let res = gen.generate(params, Some(cancellation_token.clone()), None)?;
            println!("[Scheduler] 2B Q1 Output: {}", res);
            let type_info = parsing::parse_json_from_llm(&res);
            page_type = type_info.get("type").and_then(|s| s.as_str()).unwrap_or("").to_string();
        }
    }
    
    if page_type.is_empty() || page_type == "unknown" { return Ok(()); }
    
    // Q2: Selector Identification
    {
        let selector_prompt = parsing::page_selectors_prompt(&page_type);
        let params = ChatCompletionParameters {
            messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { content: ChatCompletionRequestUserMessageContent::Text(format!("TASK: {}\n\nACTION: JSON ONLY", selector_prompt)), name: None })],
            model: "qwen3vl".to_string(), max_tokens: Some(512), temperature: Some(0.1), ..Default::default()
        };
        if let Some(gen) = model.generator.lock().await.as_mut() {
            let res = gen.generate(params, Some(cancellation_token.clone()), None)?;
            println!("[Scheduler] 2B Q2 Output: {}", res);
            selector_info = parsing::parse_json_from_llm(&res);
        }
    }

    let is_detail = selector_info.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
    
    // Page Learning Logic
    {
        let db = &store;
        let team_id = if task.to.is_empty() { crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") } else { task.to.clone() };
        let origin_str = task_data.get("origin").and_then(|s| s.as_str()).or_else(|| if let Ok(u) = url::Url::parse(&url) { Some(u.origin().ascii_serialization()).filter(|_| true) } else { None });
        let origin_final = origin_str.map(|s| s.to_string()).unwrap_or_else(|| "http://unknown.com".to_string());

        if let Ok(base_url) = url::Url::parse(&origin_final) {
            let url_obj = match url::Url::parse(&url) {
                Ok(parsed) => parsed,
                Err(url::ParseError::RelativeUrlWithoutBase) => base_url.join(&url).unwrap_or(base_url.clone()),
                Err(_) => base_url.clone(),
            };
            let raw_path = url_obj.path();
            let page_id = crate::utils::hash::hash_id(&format!("{}{}", task.cc, raw_path)); 
            let cc_for_bcc = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
            let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_for_bcc));

            let mut page_data = selector_info.clone();
            if let Some(obj) = page_data.as_object_mut() {
                obj.insert("origin".to_string(), json!(format!("{}://{}", url_obj.scheme(), url_obj.host_str().unwrap_or(""))));
                obj.insert("link".to_string(), json!(url_obj.path().to_string() + url_obj.query().map(|q| format!("?{}", q)).unwrap_or_default().as_str()));
                obj.insert("type".to_string(), json!(page_type));
            }
            let _ = db.upsert_item("pages", &page_id, "pages", page_data, None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(raw_path), None).await;
        }
    } 

    let page_info = {
        let mut pi = json!({ "type": page_type });
        if let Some(obj) = selector_info.as_object() { for (k, v) in obj { pi.as_object_mut().unwrap().insert(k.clone(), v.clone()); } }
        pi
    };
    let node_selector = page_info.get("node").and_then(|s| s.as_str()).unwrap_or("");
    let item_selector = page_info.get("item").and_then(|s| s.as_str()).unwrap_or("");
    let target_selector = if !node_selector.is_empty() && !item_selector.is_empty() {
        let mut combined = Vec::new();
        for node in node_selector.split(',').map(|s| s.trim()) { for item in item_selector.split(',').map(|s| s.trim()) { if !node.is_empty() && !item.is_empty() { combined.push(format!("{} {}", node, item)); } } }
        combined.join(", ")
    } else if !item_selector.is_empty() { item_selector.to_string() }
    else if !node_selector.is_empty() { node_selector.to_string() }
    else { "body".to_string() };

    let _ = app_handle.emit("extraction-progress", json!({ "task_id": task.id, "category": "Classification", "summary": format!("Type: {}, Detail: {}", page_type, is_detail), "spinner": "✅", "data": page_info }));

    let mut extracted_data = json!({});

    // --- STEP 4: EXTRACTION ---
    if !is_detail {
        let mut all_extracted_items = Vec::new();
        {
            let clean_content = data_manager.load(&clean_html_path)?;
            let document = scraper::Html::parse_document(&clean_content);
            let field_selectors = selector_info.get("selectors").and_then(|v| v.as_object());
            if let Ok(sel) = scraper::Selector::parse(&target_selector) {
                for item_node in document.select(&sel) {
                    if cancellation_token.load(Ordering::Relaxed) { break; }
                    let mut item_json = json!({});
                    let mut has_data = false;
                    if let Some(subs) = field_selectors {
                        for (field_name, sel_str) in subs {
                            if let Ok(sub_sel) = scraper::Selector::parse(sel_str.as_str().unwrap_or("")) {
                                if let Some(found_el) = item_node.select(&sub_sel).next() {
                                    let text = found_el.text().collect::<Vec<_>>().join("").trim().to_string();
                                    if !text.is_empty() { item_json.as_object_mut().unwrap().insert(field_name.clone(), json!(text)); has_data = true; }
                                }
                            }
                        }
                    } else {
                        let text = item_node.text().collect::<Vec<_>>().join(" ").trim().to_string();
                        if !text.is_empty() { item_json.as_object_mut().unwrap().insert("text".to_string(), json!(text)); has_data = true; }
                    }
                    if has_data { all_extracted_items.push(item_json); }
                }
            }
        }
        
        if selector_info.get("selectors").is_none() && !all_extracted_items.is_empty() {
            let mut refined_items = Vec::new();
            for batch in all_extracted_items.chunks(10) {
                if cancellation_token.load(Ordering::Relaxed) { break; }
                let batch_text: String = batch.iter().enumerate().map(|(i, item)| format!("Item {}:\n{}", i + 1, item.get("text").and_then(|s| s.as_str()).unwrap_or(""))).collect::<Vec<_>>().join("\n\n");
                let instruction = parsing::list2json(&page_type, &language);
                let params = ChatCompletionParameters {
                    messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { content: ChatCompletionRequestUserMessageContent::Text(format!("CONVERT THESE ITEMS TO JSON:\n{}\n\nTASK: {}\n\nACTION: JSON ONLY", batch_text, instruction)), name: None })],
                    model: "qwen3vl".to_string(), max_tokens: Some(2048), temperature: Some(0.1), ..Default::default()
                };
                if let Some(gen) = model.generator.lock().await.as_mut() {
                    if let Ok(res) = gen.generate(params, Some(cancellation_token.clone()), None) {
                        let parsed = parsing::parse_json_from_llm(&res);
                        if let Some(arr) = parsed.get("items").and_then(|v| v.as_array()).or_else(|| parsed.as_array()) {
                            for (i, refined) in arr.iter().enumerate() {
                                if i < batch.len() {
                                    let mut item = batch[i].clone();
                                    if let (Some(obj), Some(ref_obj)) = (item.as_object_mut(), refined.as_object()) {
                                        for (k, v) in ref_obj { obj.insert(k.clone(), v.clone()); }
                                    }
                                    refined_items.push(item);
                                }
                            }
                        }
                    }
                }
            }
            if !refined_items.is_empty() { all_extracted_items = refined_items; }
        }
        extracted_data = json!({ "items": all_extracted_items, "type": page_type, "detail": false });
    } else {
        let content_pug = {
            let clean_content = data_manager.load(&clean_html_path)?;
            let document = scraper::Html::parse_document(&clean_content);
            let mut pug_output = String::new();
            if let Ok(sel) = scraper::Selector::parse(&target_selector) {
                for node in document.tree.root().descendants() {
                    if let Some(element_ref) = scraper::ElementRef::wrap(node) {
                        if sel.matches(&element_ref) { parsing::generate_pug_lines(node, 0, &mut pug_output, &PugMode::FullContent); break; }
                    }
                }
            }
            parsing::sanitize_llm_input(&pug_output)
        };
        if !content_pug.trim().is_empty() {
            let extraction_instruction = parsing::item2json(page_type, &url, language);
            let params = ChatCompletionParameters {
                messages: vec![ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { content: ChatCompletionRequestUserMessageContent::Text(format!("CONTEXT:\n{}\n\nTASK: {}\n\nACTION: JSON ONLY", content_pug, extraction_instruction)), name: None })],
                model: "qwen3vl".to_string(), max_tokens: Some(2048), temperature: Some(0.1), ..Default::default()
            };
            if let Some(gen) = model.generator.lock().await.as_mut() {
                let res = gen.generate(params, Some(cancellation_token.clone()), None)?;
                println!("[Scheduler] 2B Q3 Output: {}", res);
                extracted_data = parsing::parse_json_from_llm(&res);
            }
        }
    }

    // Normalized & Save to DB
    if let Some(obj) = extracted_data.as_object_mut() { if obj.get("type").is_none() { obj.insert("type".to_string(), json!(page_type)); } }
    let mut normalized_data = json!({});
    if let Some(obj) = extracted_data.as_object() { for (k, v) in obj { let val = if let Some(io) = v.as_object() { io.get("value").cloned().unwrap_or(v.clone()) } else { v.clone() }; normalized_data.as_object_mut().unwrap().insert(k.clone(), val); } }
    
    let team_id = if !task.to.is_empty() { task.to.clone() } else { crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") };
    let item_no = normalized_data.get("id").or_else(|| normalized_data.get("index")).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let index_val = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("{}{}{}", page_type, team_id, crate::utils::hash::normalize_numeric_homoglyphs(&item_no).replace("-", "").replace("_", ""))));
    normalized_data.as_object_mut().unwrap().insert("index".to_string(), json!(index_val));
    let target_id = crate::utils::hash::hash_id(&format!("{}{}", team_id, index_val));
    normalized_data.as_object_mut().unwrap().insert("id".to_string(), json!(target_id));
    extracted_data = normalized_data;

    // Relay Logic
    if let Some((queries, merge_info)) = logic::relay(page_type, &extracted_data) {
        let db = &store;
        let mut target_data = extracted_data.clone();
        let mut final_target_id = target_id.clone();
        for query in queries {
            if let Ok(Some((id, existing_data))) = db.find_item_by_property(&query.table, &query.column, &query.value).await {
                logic::merge_node(&mut target_data, &existing_data);
                final_target_id = id;
                break;
            }
        }
        let text_to_embed = parsing::json_to_natural_language(&target_data);
        let item_digest = crate::utils::hash::digest(&text_to_embed);
        let vector = model.get_embedding(text_to_embed.chars().take(3000).collect()).await?;
        let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
        let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_val));
        let link = target_data.get("link").and_then(|v| v.as_str()).unwrap_or("");
        let ref_val = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, task.cc, link));
        let _ = db.upsert_item(&merge_info.to, &final_target_id, page_type, target_data.clone(), Some(vector.clone()), Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)).await;
        let _ = db.upsert_item("items", &final_target_id, page_type, target_data, Some(vector), Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)).await;
    } else {
        let text_to_embed = parsing::json_to_natural_language(&extracted_data);
        let item_digest = crate::utils::hash::digest(&text_to_embed);
        let vector = model.get_embedding(text_to_embed.chars().take(3000).collect()).await?;
        let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
        let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_val));
        let _ = db.upsert_item(&page_type, &target_id, page_type, extracted_data.clone(), Some(vector.clone()), Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&task.r#ref), Some(&item_digest)).await;
        let _ = db.upsert_item("items", &target_id, page_type, extracted_data, Some(vector), Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&task.r#ref), Some(&item_digest)).await;
    }

    let status_code = logic::parse_status("complete");
    let _ = store.update_message_status(&task.id, status_code, Some("Extraction Complete")).await;
    let _ = app_handle.emit("extraction-progress", json!({ "task_id": task.id, "category": "Done", "summary": "Extraction complete.", "spinner": "✅", "data": extracted_data }));
    model.unload_generator().await;
    Ok(())
}

fn cleanup_task_resources(task_id: &str, app_handle: Option<&tauri::AppHandle>) {
    let _ = fs::remove_dir_all(utils::paths::get_task_specific_dir(app_handle, task_id));
}

fn pre_fetch_weights(path: &std::path::Path) -> Result<()> {
    use std::io::Read;
    if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.filter_map(|e| e.ok()) {
                let p = entry.path();
                if p.extension().map_or(false, |ext| ext == "gguf" || ext == "safetensors") {
                    if let Ok(mut file) = std::fs::File::open(p) {
                        let mut buffer = [0u8; 1024 * 1024];
                        while let Ok(n) = file.read(&mut buffer) { if n == 0 { break; } }
                    }
                }
            }
        }
    }
    Ok(())
}

pub fn log_task_progress(app: &tauri::AppHandle, task_id: &str, payload: &serde_json::Value) {
    use std::io::Write;
    let log_path = crate::utils::paths::get_task_log_file(Some(app), task_id);
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(log_path) {
        let line = format!("{}\n", payload.to_string());
        let _ = file.write_all(line.as_bytes());
    }
}

fn clear_all_temp_data(app_handle: Option<&tauri::AppHandle>) {
    utils::paths::cleanup_temp_dirs(app_handle);
}

async fn wait_for_resources_settled(target_vram_mb: u64, target_ram_mb: u64, cancellation_token: Option<&Arc<AtomicBool>>) -> Result<()> {
    use nvml_wrapper::Nvml;
    use sysinfo::{System, SystemExt};
    let mut sys = System::new_all();
    let nvml = Nvml::init().ok();
    let target_vram_bytes = target_vram_mb * 1024 * 1024;
    let target_ram_bytes = target_ram_mb * 1024 * 1024;
    let mut last_vram = 0;
    let mut stable_ticks = 0;
    let start_time = std::time::Instant::now();

    loop {
        if let Some(token) = cancellation_token { if token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled during resource wait")); } }
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
        if (!has_gpu || current_vram >= target_vram_bytes) && current_ram >= target_ram_bytes { break; }
        let delta = if current_vram > last_vram { current_vram - last_vram } else { last_vram - current_vram };
        if delta < 5_000_000 { stable_ticks += 1; } else { stable_ticks = 0; }
        if (stable_ticks >= 6 && current_vram > 1_200_000_000) || start_time.elapsed().as_secs() > 20 { break; }
        last_vram = current_vram;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Ok(())
}
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use crate::store::{VectorStore, Task};
use crate::logic;
use crate::utils;
use crate::parsing::{self, PugMode};
use crate::model::{LogisModel, ModelSize};
use crate::models::qwen3vl::generate::Qwen3VLGenerateModel;
use crate::openai_types::ChatCompletionParameters;
use serde_json::{Value, json};
use anyhow::Result;
use tauri::{AppHandle, Manager, Emitter};
use std::sync::atomic::{AtomicBool, Ordering};
use std::fs;
use std::path::PathBuf;
use tokio::sync::Notify;

// --- [MEMORY OPTIMIZATION] Task Data Manager (RAII) ---
pub struct TaskDataManager {
    pub task_id: String,
    created_files: Vec<PathBuf>,
    app_handle: Option<tauri::AppHandle>,
}

impl TaskDataManager {
    pub fn new(task_id: &str, app_handle: Option<tauri::AppHandle>) -> Self {
        Self {
            task_id: task_id.to_string(),
            created_files: Vec::new(),
            app_handle,
        }
    }

    pub fn offload(&mut self, content: &str, suffix: &str) -> Result<PathBuf> {
        let dir = utils::paths::get_task_specific_dir(self.app_handle.as_ref(), &self.task_id);
        let _ = fs::create_dir_all(&dir);
        let filename = format!("{}_{}_{}.txt", self.task_id, chrono::Utc::now().timestamp_micros(), suffix);
        let path = dir.join(filename);
        fs::write(&path, content)?;
        self.created_files.push(path.clone());
        Ok(path)
    }

    pub fn load(&self, path: &std::path::Path) -> Result<String> {
        Ok(fs::read_to_string(path)?)
    }

    pub fn get_path(&self, suffix: &str) -> PathBuf {
        let dir = utils::paths::get_task_specific_dir(self.app_handle.as_ref(), &self.task_id);
        dir.join(format!("{}.txt", suffix))
    }
}

impl Drop for TaskDataManager {
    fn drop(&mut self) {
        println!("[Cleanup] TaskDataManager dropping for task: {}", self.task_id);
    }
}

// [RESTORED] Helper to chunk text
pub fn chunk_text(text: &str, target_size: usize) -> Vec<String> {
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
        }
        let slice = &text[start..end];
        if !slice.trim().is_empty() { chunks.push(slice.to_string()); }
        start = end;
    }
    chunks
}

// [RESTORED] Deep merge
pub fn merge_json_results(target: &mut Value, source: &Value) {
    if let (Some(target_obj), Some(source_obj)) = (target.as_object_mut(), source.as_object()) {
        for (k, v) in source_obj {
            if v.is_null() { continue; }
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

pub static UI_READY_SIGNAL: Notify = Notify::const_new();
pub static TASK_QUEUED_SIGNAL: Notify = Notify::const_new();
pub static UI_READY_FLAG: AtomicBool = AtomicBool::new(false);

pub fn mark_ui_ready() {
    UI_READY_FLAG.store(true, Ordering::SeqCst);
    UI_READY_SIGNAL.notify_waiters();
}

pub fn notify_new_task() {
    TASK_QUEUED_SIGNAL.notify_waiters();
}

pub fn log_task_progress(app_handle: &tauri::AppHandle, task_id: &str, payload: &Value) {
    let log_file = utils::paths::get_task_log_file(Some(app_handle), task_id);
    let _ = fs::create_dir_all(log_file.parent().unwrap());
    if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(&log_file) {
        use std::io::Write;
        let _ = writeln!(file, "{}", payload.to_string());
    }
    let _ = app_handle.emit("extraction-progress", payload);
}

pub async fn start_background_worker(
    store: Arc<Mutex<Option<VectorStore>>>,
    model: Arc<Mutex<Option<LogisModel>>>, 
    cancellation_token: Arc<AtomicBool>,
    app_handle: tauri::AppHandle,
) {
    println!("[Scheduler] Background worker started.");
    tokio::spawn(async move {
        if !UI_READY_FLAG.load(Ordering::SeqCst) { UI_READY_SIGNAL.notified().await; }
        let mut delay_secs = 1;
        let mut current_device_pref: Option<String> = None;
        loop {
            if crate::utils::is_extraction_stopped() { sleep(Duration::from_millis(500)).await; continue; }
            let mut pending_tasks = Vec::new();
            {
                let store_opt = store.lock().await;
                if let Some(db) = store_opt.as_ref() { if let Ok(tasks) = db.get_pending_tasks(5).await { pending_tasks = tasks; } }
            }
            if pending_tasks.is_empty() {
                tokio::select! {
                    _ = sleep(Duration::from_secs(delay_secs)) => { delay_secs = (delay_secs + 1).min(10); }
                    _ = TASK_QUEUED_SIGNAL.notified() => { delay_secs = 1; }
                }
                continue;
            } else { delay_secs = 1; }

            for task in pending_tasks {
                if cancellation_token.load(Ordering::Relaxed) { break; } 
                {
                    let store_guard = store.lock().await;
                    if let Some(db) = store_guard.as_ref() { let _ = db.update_task_status(&task.id, 1).await; }
                }
                match process_task(task.clone(), &store, &model, &cancellation_token, &app_handle, current_device_pref.clone()).await {
                    Ok(_) => {
                        let store_guard = store.lock().await;
                        if let Some(db) = store_guard.as_ref() { let _ = db.update_task_status(&task.id, 9).await; }
                        cleanup_task_resources(&task.id, Some(&app_handle));
                        current_device_pref = None;
                    },
                    Err(e) => {
                        let err_msg = e.to_string();
                        {
                            let mut model_lock = model.lock().await;
                            if let Some(m) = model_lock.as_ref() { m.unload_generator().await; }
                            *model_lock = None;
                        }
                        if err_msg.contains("Task cancelled") {
                            let store_guard = store.lock().await;
                            if let Some(db) = store_guard.as_ref() { let _ = db.update_task_status(&task.id, 3).await; }
                            cleanup_task_resources(&task.id, Some(&app_handle));
                            break;
                        } else {
                            if err_msg.contains("out of memory") || err_msg.contains("CUDA_ERROR_OUT_OF_MEMORY") {
                                current_device_pref = Some("cpu".to_string());
                                log_task_progress(&app_handle, &task.id, &json!({ "category": "Error", "summary": "OOM retry on CPU", "spinner": "⚠️" }));
                                sleep(Duration::from_secs(2)).await;
                                continue;
                            }
                            let store_guard = store.lock().await;
                            if let Some(db) = store_guard.as_ref() {
                                let _ = db.update_task_status(&task.id, 6).await;
                                let _ = db.update_message_status(&task.id, 6, Some(&format!("Error: {}", err_msg))).await;
                            }
                            log_task_progress(&app_handle, &task.id, &json!({ "category": "Error", "summary": format!("Error: {}", err_msg), "spinner": "❌" }));
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
    app_handle: &tauri::AppHandle,
    device_preference: Option<String>,
) -> Result<()> {
    let pug_logs_dir = utils::paths::get_pug_logs_dir(Some(app_handle), &task.id);
    let _ = fs::create_dir_all(&pug_logs_dir);
    
    log_task_progress(app_handle, &task.id, &json!({ "category": "Processing", "summary": "Starting high-speed extraction...", "spinner": "⠋" }));

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let mut data_manager = TaskDataManager::new(&task.id, Some(app_handle.clone()));
    let task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));

    let effective_device_pref = device_preference.or_else(|| {
        task_data.get("device_preference").and_then(|v| {
            if v.as_str() == Some("cpu") || v.as_bool() == Some(true) { Some("cpu".to_string()) } else { None }
        })
    });

    let url = task_data.get("link").and_then(|s| s.as_str()).unwrap_or("").to_string();
    let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("").to_string();

    let model = {
        let mut model_lock = model_mutex.lock().await;
        if model_lock.is_none() {
            match LogisModel::new(effective_device_pref.as_deref()).await {
                Ok(m) => *model_lock = Some(m),
                Err(e) => return Err(anyhow::anyhow!("Model Load Failed: {}", e)),
            }
        }
        model_lock.as_ref().unwrap().clone()
    };

    if !image_path.is_empty() {
        let language = task_data.get("language").and_then(|s| s.as_str()).unwrap_or("korean").to_string();
        return model.extract_from_image(task.id.clone(), task.r#ref.clone(), image_path, language, app_handle, Some(cancellation_token.clone()), store_mutex).await;
    }

    if url.is_empty() { return Ok(()); }

    let raw_html = if let Some(h) = task_data.get("html").and_then(|s| s.as_str()) {
        h.to_string()
    } else {
        reqwest::get(&url).await?.text().await?
    };
    
    let clean = parsing::pre_clean_html(&raw_html);
    let light_pug = parsing::convert_to_clean_pug(&clean, PugMode::FullContent);
    data_manager.offload(&light_pug, "light_pug")?;
    
    let ts_nano = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let _ = std::fs::write(pug_logs_dir.join(format!("light_{}_{}.pug", task.id, ts_nano)), &light_pug);

    log_task_progress(app_handle, &task.id, &json!({ "category": "Baking", "summary": "Baking HTML context on GPU...", "spinner": "⠋" }));
    model.bake_pug_context(&task.r#ref, &task.id, &light_pug, Some(cancellation_token.clone())).await?;

    let mut page_type = String::new();
    let mut selector_info = json!({});
    let mut is_detail = false;

    {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        let type_prompt = parsing::page_type_prompt();
        log_task_progress(app_handle, &task.id, &json!({ "category": "Classification", "summary": "Identifying page type...", "spinner": "⠋" }));
        let res = model.run_modular_inference(&task.r#ref, &task.id, "class", &type_prompt, "Classify this page.", Some(cancellation_token.clone())).await?;
        let type_info = parsing::parse_json_from_llm(&res);
        page_type = type_info.get("type").and_then(|s| s.as_str()).unwrap_or("unknown").to_string();
        if page_type == "unknown" || page_type.is_empty() { return Ok(()); }
        log_task_progress(app_handle, &task.id, &json!({ "category": "Classification", "summary": format!("Identified as {}", page_type), "spinner": "✅" }));
    }

    model.deep_purge_resources(false).await;
    wait_for_resources_settled(1200, 800, Some(cancellation_token)).await?;

    {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        let selector_prompt = parsing::page_selectors_prompt(&page_type);
        log_task_progress(app_handle, &task.id, &json!({ "category": "Selectors", "summary": "Analyzing structural selectors...", "spinner": "⠋" }));
        let res = model.run_modular_inference(&task.r#ref, &task.id, "sel", &selector_prompt, "Identify selectors.", Some(cancellation_token.clone())).await?;
        selector_info = parsing::parse_json_from_llm(&res);
        is_detail = selector_info.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
        log_task_progress(app_handle, &task.id, &json!({ "category": "Selectors", "summary": "Selectors identified.", "spinner": "✅", "data": selector_info }));
    }

    {
        let db_lock = store_mutex.lock().await;
        if let Some(db) = db_lock.as_ref() {
            let team_id = if task.to.is_empty() { crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") } else { task.to.clone() };
            if let Ok(url_obj) = url::Url::parse(&url) {
                let raw_path = url_obj.path();
                let page_id = crate::utils::hash::hash_id(&format!("{}{}", task.cc, raw_path)); 
                let cc_for_bcc = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
                let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_for_bcc));
                let mut page_data = selector_info.clone();
                if let Some(obj) = page_data.as_object_mut() {
                    obj.insert("origin".to_string(), json!(format!("{}://{}", url_obj.scheme(), url_obj.host_str().unwrap_or(""))));
                    obj.insert("link".to_string(), json!(url_obj.path()));
                    obj.insert("type".to_string(), json!(page_type));
                }
                let _ = db.upsert_item("pages", &page_id, "pages", page_data, None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(raw_path), None).await;
            }
        }
    }

    model.deep_purge_resources(false).await;
    wait_for_resources_settled(1200, 800, Some(cancellation_token)).await?;

    let mut all_extracted_items = Vec::new();
    let mut extracted_data = json!({});

        if !is_detail {

            log_task_progress(app_handle, &task.id, &json!({ "category": "Extraction", "summary": "Direct DOM extraction starting...", "spinner": "⠋" }));

            

            let target_selector = selector_info.get("item").and_then(|s| s.as_str()).unwrap_or("body").to_string();

            let field_selectors = selector_info.get("selectors").and_then(|v| v.as_object());

    

            // [FIX] Use a scoped block to ensure 'document' (!Send) is dropped before any .await

            {

                let document = scraper::Html::parse_document(&clean);

                if let Ok(sel) = scraper::Selector::parse(&target_selector) {

                    for item_node in document.select(&sel) {

                        let mut item_json = json!({});

                        let mut has_data = false;

                        if let Some(subs) = field_selectors {

                            for (field_name, sel_val) in subs {

                                if let Ok(sub_sel) = scraper::Selector::parse(sel_val.as_str().unwrap_or("")) {

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

                        if has_data {

                            if let Some(link_el) = item_node.select(&scraper::Selector::parse("a[href]").unwrap()).next() {

                                item_json.as_object_mut().unwrap().insert("link".to_string(), json!(link_el.value().attr("href").unwrap_or("")));

                            }

                            all_extracted_items.push(item_json);

                        }

                    }

                }

            } // 'document' dropped here

    

            if !all_extracted_items.is_empty() {

    
            log_task_progress(app_handle, &task.id, &json!({ "category": "Refinement", "summary": format!("Refining {} items with AI...", all_extracted_items.len()), "spinner": "⠋" }));
            let mut refined_items = Vec::new();
            for (batch_idx, batch) in all_extracted_items.chunks(10).enumerate() {
                if cancellation_token.load(Ordering::Relaxed) { break; }
                let batch_text: String = batch.iter().enumerate().map(|(i, it)| format!("Item {}:\n{}", i + 1, it.to_string())).collect::<Vec<_>>().join("\n\n");
                let ins = parsing::list2json(&page_type, "english");
                let res = model.run_modular_inference(&task.r#ref, &task.id, &format!("list_{}", batch_idx), &ins, &batch_text, Some(cancellation_token.clone())).await?;
                let parsed = parsing::parse_json_from_llm(&res);
                if let Some(arr) = parsed.as_array().or_else(|| parsed.get("items").and_then(|v| v.as_array())) {
                    for (i, refined) in arr.iter().enumerate() { if i < batch.len() { let mut m = batch[i].clone(); merge_json_results(&mut m, refined); refined_items.push(m); } }
                } else { refined_items.extend_from_slice(batch); }
            }
            all_extracted_items = refined_items;
        }
    } else {
        log_task_progress(app_handle, &task.id, &json!({ "category": "Extraction", "summary": "Extracting structured details...", "spinner": "⠋" }));
        let ins = parsing::item2json(&page_type, &url, "english");
        let res = model.run_modular_inference(&task.r#ref, &task.id, "detail", &ins, "Extract everything.", Some(cancellation_token.clone())).await?;
        extracted_data = parsing::parse_json_from_llm(&res);
        all_extracted_items.push(extracted_data.clone());
    }

    model.unload_generator().await;
    let store = { let g = store_mutex.lock().await; g.as_ref().ok_or_else(|| anyhow::anyhow!("DB Error"))?.clone() };
    let team_id = if task.to.is_empty() { crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") } else { task.to.clone() };

    // [NEW] Side-Effect Logic for Orders
    if page_type == "order" && is_detail {
        if let Some(goods_arr) = extracted_data.get("goods").and_then(|v| v.as_array()) {
            let cc_val = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
            for good in goods_arr {
                let g_no = good.get("id").or_else(|| good.get("no")).and_then(|v| v.as_str()).unwrap_or("");
                if !g_no.is_empty() {
                    let clean_g_no = crate::utils::hash::normalize_numeric_homoglyphs(g_no).replace("-", "").replace("_", "");
                    let g_index = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("goods{}{}", team_id, clean_g_no)));
                    let tracking_id = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, task.id, clean_g_no));
                    let mut tracking_data = extracted_data.clone();
                    if let Some(obj) = tracking_data.as_object_mut() {
                        obj.insert("type".to_string(), json!("tracking"));
                        obj.insert("goods".to_string(), json!(g_index));
                        obj.insert("order".to_string(), json!(crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("order{}{}", team_id, task.r#ref)))));
                    }
                    let _ = store.upsert_item("tracking", &tracking_id, "tracking", tracking_data, None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&crate::utils::hash::hash_id(&format!("tracking{}", cc_val))), Some(&crate::utils::hash::hash_id(&format!("{}{}{}", team_id, task.cc, task.r#ref))), None).await;
                }
            }
        }
    }

    if !is_detail {
        for mut item in all_extracted_items {
            if cancellation_token.load(Ordering::Relaxed) { break; }
            let item_id;
            let idx_val;
            let link;

            if let Some(obj) = item.as_object_mut() {
                if obj.get("type").is_none() { obj.insert("type".to_string(), json!(page_type)); }
                let id_raw = obj.get("id").or_else(|| obj.get("index")).and_then(|v| v.as_str()).unwrap_or("");
                let clean_no = crate::utils::hash::normalize_numeric_homoglyphs(id_raw).replace("-", "").replace("_", "");
                idx_val = crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("{}{}{}", page_type, team_id, clean_no)));
                item_id = crate::utils::hash::hash_id(&format!("{}{}", team_id, idx_val));
                obj.insert("index".to_string(), json!(idx_val));
                obj.insert("id".to_string(), json!(item_id));
                link = obj.get("link").and_then(|v| v.as_str()).unwrap_or("").to_string();
            } else { continue; }

            let mut final_item = item.clone();
            if let Some((queries, _merge_info)) = logic::relay(&page_type, &item) {
                for q in queries { 
                    if let Ok(Some((_ex_id, ex_data))) = store.find_item_by_property(&q.table, &q.column, &q.value).await { 
                        logic::merge_node(&mut final_item, &ex_data); 
                        break; 
                    } 
                }
            }

            let text_to_embed = parsing::json_to_natural_language(&final_item);
            let item_digest = crate::utils::hash::digest(&text_to_embed);
            let vector = model.get_embedding(text_to_embed).await?;
            
            // [STRICT PARITY] Re-generate BCC and REF exactly as server does
            let cc_for_hash = if is_detail { task.cc.to_uppercase() } else { task.cc.clone() };
            let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_for_hash));
            // For list items, ref_val must include the hashed link
            let ref_val = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, task.cc, link));

            let _ = store.upsert_item(&page_type, &item_id, &page_type, final_item.clone(), Some(vector.clone()), Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)).await;
            let _ = store.upsert_item("items", &item_id, &page_type, final_item, Some(vector), Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)).await;
        }
    } else {
        // [RESTORED] Direct Item Persistence for Detail Pages (Proxy Parity)
        let target_id = task.r#ref.clone();
        let mut final_data = extracted_data.clone();
        
        let mut text_to_embed = parsing::json_to_natural_language(&final_data);
        let item_digest = crate::utils::hash::digest(&text_to_embed);
        
        // Skip embedding if unchanged
        let mut existing_vector = None;
        if let Ok(Some(ex)) = store.get_item_by_id(&page_type, &target_id).await {
            if ex.digest == item_digest { 
                println!("[Scheduler] Content unchanged for {}. Reusing vector.", target_id);
                existing_vector = Some(ex.vector); 
            }
        }

        let vector = if let Some(v) = existing_vector { v } else { model.get_embedding(text_to_embed).await? };
        let cc_for_hash = task.cc.to_uppercase();
        let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_for_hash));
        // [FIX] For detail pages, ref_val IS the task.r#ref (URL hash)
        let ref_val = task.r#ref.clone();

        let _ = store.upsert_item(&page_type, &target_id, &page_type, final_data.clone(), Some(vector.clone()), Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)).await;
        let _ = store.upsert_item("items", &target_id, &page_type, final_data, Some(vector), Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)).await;
    }
    log_task_progress(app_handle, &task.id, &json!({ "category": "Done", "summary": "Extraction Complete", "spinner": "✅" }));
    Ok(())
}

fn cleanup_task_resources(task_id: &str, app_handle: Option<&tauri::AppHandle>) {
    let _ = fs::remove_dir_all(utils::paths::get_task_specific_dir(app_handle, task_id));
}

// [RESTORED] OS Page Cache Warm-up for high-speed model switching
pub fn pre_fetch_weights(path: &std::path::Path) -> Result<()> {
    use std::io::Read;
    println!("[PRE-FETCH] Warming up OS Page Cache for weights in: {:?}", path);
    if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().map_or(false, |ext| ext == "gguf" || ext == "safetensors") {
                    if let Ok(mut file) = std::fs::File::open(&p) {
                        let mut buffer = [0u8; 1024 * 1024]; 
                        while let Ok(n) = file.read(&mut buffer) {
                            if n == 0 { break; }
                        }
                    }
                }
            }
        }
    }
    println!("[PRE-FETCH] Warm-up complete.");
    Ok(())
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
        
        if meets_vram && meets_ram { break; }

        let delta = if current_vram > last_vram { current_vram - last_vram } else { last_vram - current_vram };
        if delta < 5_000_000 { stable_ticks += 1; } 
        else { stable_ticks = 0; }

        if stable_ticks >= 6 && current_vram > 1_200_000_000 {
            println!("[RESOURCE-WATCH] Memory stabilized. Proceeding with {:.2} GB free VRAM.", current_vram as f64 / 1e9);
            break;
        }

        if last_report.elapsed().as_secs() >= 5 {
            println!("[RESOURCE-DIAG] Waiting... VRAM: {:.2} GB free (Target: {:.2} GB)", 
                current_vram as f64 / 1e9, target_vram_mb as f64 / 1024.0);
            last_report = std::time::Instant::now();
        }

        if start_time.elapsed().as_secs() > 20 {
            println!("[RESOURCE-WATCH] Max wait reached. Proceeding anyway.");
            break;
        }

        last_vram = current_vram;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Ok(())
}

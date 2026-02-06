use std::sync::Arc;
use tokio::sync::{Mutex, Notify};
use tokio::time::{sleep, Duration};
use dashmap::DashMap;
use futures::future::{BoxFuture, Shared};
use futures::FutureExt;
use crate::store::{VectorStore, Task};
use crate::logic;
use crate::utils;
use crate::parsing::{self, PugMode};
use crate::model::{LogisModel, ModelSize};
use serde_json::{Value, json};
use anyhow::Result;
use tauri::Emitter;
use std::sync::atomic::{AtomicBool, Ordering};
use std::fs;
use std::path::PathBuf;

// --- [2026-REGISTRY] In-Flight Task Deduplication ---
type SharedBakeResult = Shared<BoxFuture<'static, Arc<Result<()>>>>;
type SharedInfResult = Shared<BoxFuture<'static, Arc<Result<String>>>>;

pub struct InFlightRegistry {
    pub baking: DashMap<String, SharedBakeResult>,
    pub inference: DashMap<String, SharedInfResult>,
}

lazy_static::lazy_static! {
    pub static ref REGISTRY: InFlightRegistry = InFlightRegistry {
        baking: DashMap::new(),
        inference: DashMap::new(),
    };
}

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
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_micros();
        let filename = format!("{}_{}_{}.txt", self.task_id, ts, suffix);
        let path = dir.join(filename);
        fs::write(&path, content)?;
        self.created_files.push(path.clone());
        Ok(path)
    }

    pub fn load(&self, path: &std::path::Path) -> Result<String> {
        Ok(fs::read_to_string(path)?)
    }
}

impl Drop for TaskDataManager {
    fn drop(&mut self) {
        println!("[Cleanup] TaskDataManager dropping for task: {}. Files persisted for debug.", self.task_id);
    }
}

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
    println!("[Scheduler] Background worker started with OOM recovery.");
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
                            if let Some(db) = store_guard.as_ref() {
                                let _ = db.update_task_status(&task.id, 3).await;
                                let _ = db.update_message_status(&task.id, 3, Some("Cancelled by user")).await;
                            }
                            let _ = app_handle.emit("extraction-progress", json!({ "task_id": task.id, "category": "Done", "summary": "Cancelled", "spinner": "🛑" }));
                            cleanup_task_resources(&task.id, Some(&app_handle));
                            break;
                        } else {
                            if err_msg.contains("out of memory") || err_msg.contains("CUDA_ERROR_OUT_OF_MEMORY") {
                                current_device_pref = Some("cpu".to_string());
                                log_task_progress(&app_handle, &task.id, &json!({ "category": "Error", "summary": "GPU OOM. Retrying on CPU.", "spinner": "⚠️" }));
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
    
    println!("[PROCESS] Task {} started.", task.id);
    cleanup_task_resources(&task.id, Some(app_handle));
    log_task_progress(app_handle, &task.id, &json!({ "category": "Processing", "summary": "Starting high-speed extraction...", "spinner": "⠋" }));

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    // [2026-CLEAN-START] Force clear existing temporary data for this address hash
    {
        let kv_dir = utils::paths::get_kv_dir(None, Some(&task.r#ref));
        if kv_dir.exists() {
            println!("[CLEAN-START] Purging existing KV cache for address: {}", task.r#ref);
            let _ = fs::remove_dir_all(&kv_dir);
        }
        let _ = fs::create_dir_all(&kv_dir);
    }

    let _data_manager = TaskDataManager::new(&task.id, Some(app_handle.clone()));
    let task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));

    let effective_device_pref = device_preference.or_else(|| {
        task_data.get("device_preference").and_then(|v| {
            if v.as_str() == Some("cpu") || v.as_bool() == Some(true) { Some("cpu".to_string()) } else { None }
        })
    });

    let url = task_data.get("link").and_then(|s| s.as_str()).unwrap_or("").to_string();
    let image_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("").to_string();

    // 1. Prepare Model
    let model = {
        let mut model_lock = model_mutex.lock().await;
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        if let Some(m) = model_lock.as_ref() {
            let wants_cpu = effective_device_pref.as_deref() == Some("cpu");
            if m.is_cpu_mode != wants_cpu {
                m.unload_generator().await;
                *model_lock = None;
                let _ = wait_for_resources_settled(1000, 500, cancellation_token).await;
            }
        }

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
    
    if light_pug.trim().len() < 10 { return Ok(()); }

    log_task_progress(app_handle, &task.id, &json!({ "category": "Baking", "summary": "Baking HTML context on GPU...", "spinner": "⠋" }));
    
    // [2026-STRICT-COMPATIBILITY] Always use LARGE model for baking site context.
    // Even in baking mode (Layer 0), the dimensions must match the final inference model.
    model.secure_vram_relay_ext(ModelSize::Large, None, Some(cancellation_token.clone()), false, true, Some(&task.r#ref)).await?;
    
    // [2026-PHYSICAL-DEDUPLICATION] Check registry before baking
    let bake_key = format!("bake_{}", task.r#ref);
    let bake_future = if let Some(existing) = REGISTRY.baking.get(&bake_key) {
        println!("[REGISTRY] Joining in-flight baking for address: {}", task.r#ref);
        existing.clone()
    } else {
        println!("[REGISTRY] Starting new baking for address: {}", task.r#ref);
        let model_arc = model.clone();
        let ref_id = task.r#ref.clone();
        let task_id = task.id.clone();
        let pug = light_pug.clone();
        let cancel = cancellation_token.clone();
        
        let new_bake = async move {
            // Internal call will use the Large model since we secured it above
            Arc::new(model_arc.bake_pug_context(&ref_id, &task_id, &pug, Some(cancel)).await)
        }.boxed().shared();
        
        REGISTRY.baking.insert(bake_key.clone(), new_bake.clone());
        new_bake
    };

    let _bake_result = (*bake_future.await).as_ref().map_err(|e| anyhow::anyhow!("{}", e))?;
    REGISTRY.baking.remove(&bake_key);
    
    println!("[PROCESS] Baking complete with 2B dimensions. Advancing to INTEGRATED analysis phase.");

    // --- INTEGRATED INFERENCE PHASE ---
    model.secure_vram_relay_ext(ModelSize::Large, None, Some(cancellation_token.clone()), false, false, Some(&task.r#ref)).await?;
    
    {
        let kv_dir = utils::paths::get_kv_dir(None, Some(&task.r#ref));
        let pug_base_path = kv_dir.join("shared_pug_base.safetensors");
        let gen_arc = model.generator.clone();
        tokio::task::spawn_blocking(move || {
            let mut gen_guard = gen_arc.blocking_lock();
            if let Some(gen) = gen_guard.as_mut() {
                gen.load_kv_stitched(&[pug_base_path])?;
                println!("[PROCESS] KV Cache Stitched. Offset: {} tokens.", gen.get_kv_len());
            }
            Ok::<(), anyhow::Error>(())
        }).await??;
    }

    let mut page_type = String::new();
    let mut selector_info = json!({});
    let mut is_detail = false;
    let mut all_extracted_items = Vec::new();

    // [2026-SPECULATIVE-INIT] Load 0.6B to CPU RAM for parallel drafting
    model.ensure_speculative_draftsman().await?;
    {
        let gen_arc = model.generator.clone();
        let draft_arc = model.speculative_draftsman.clone();
        let mut gen_guard = gen_arc.lock().await;
        if let Some(gen) = gen_guard.as_mut() {
            gen.speculative_draft = Some(draft_arc);
        }
    }

    // [PHASE 1 & 2] Integrated Analysis (Live Prompting over Stitched Base)
    {
        log_task_progress(app_handle, &task.id, &json!({ "category": "Analysis", "summary": "Identifying page type...", "spinner": "⠋" }));
        
        let type_prompt = parsing::page_type_prompt();

        // 1. Ensure 2B is loaded and stitched with ONLY the PUG base
        model.secure_vram_relay_ext(ModelSize::Large, None, Some(cancellation_token.clone()), false, true, Some(&task.r#ref)).await?;
        {
            let kv_dir = utils::paths::get_kv_dir(None, Some(&task.r#ref));
            let pug_base_path = kv_dir.join("shared_pug_base.safetensors");
            let gen_arc = model.generator.clone();
            tokio::task::spawn_blocking(move || {
                let mut gen_guard = gen_arc.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    gen.load_kv_stitched(&[pug_base_path])?;
                    println!("[PROCESS] KV Base Stitched for Analysis. Offset: {} tokens.", gen.get_kv_len());
                }
                Ok::<(), anyhow::Error>(())
            }).await??;
        }

        // 2. Inference: Pass system prompt LIVE (No-Bridge mode will handle the offset)
        let res = model.chat(&type_prompt, "", Some(cancellation_token.clone()), Some("integrated_analysis_nobridge".to_string())).await?;
        let type_info = parsing::parse_json_from_llm(&res);
        page_type = type_info.get("type").and_then(|s| s.as_str()).unwrap_or("unknown").to_string();
        
        if page_type == "unknown" || page_type.is_empty() { 
            println!("[PROCESS] identification failure. Response was: {}", res);
            return Ok(()); 
        }

        // 3. Selector Analysis (Live Prompting)
        log_task_progress(app_handle, &task.id, &json!({ "category": "Analysis", "summary": format!("Identified {}. Finding selectors...", page_type), "spinner": "⠋" }));
        let selector_prompt = parsing::page_selectors_prompt(&page_type);
        
        // No need to re-stitch if we are using the same PUG base, 
        // but to be safe and clean, we reset to the PUG base state for each new "question"
        {
            let kv_dir = utils::paths::get_kv_dir(None, Some(&task.r#ref));
            let pug_base_path = kv_dir.join("shared_pug_base.safetensors");
            let gen_arc = model.generator.clone();
            tokio::task::spawn_blocking(move || {
                let mut gen_guard = gen_arc.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    gen.load_kv_stitched(&[pug_base_path])?;
                }
                Ok::<(), anyhow::Error>(())
            }).await??;
        }

        let res_sel = model.chat(&selector_prompt, "", Some(cancellation_token.clone()), Some("integrated_analysis_nobridge".to_string())).await?;
        selector_info = parsing::parse_json_from_llm(&res_sel);
        
        is_detail = selector_info.get("detail").and_then(|v| v.as_bool()).unwrap_or(false);
        log_task_progress(app_handle, &task.id, &json!({ "category": "Analysis", "summary": format!("Identified {} (Detail: {})", page_type, is_detail), "spinner": "✅" }));
    }

    let store = { let g = store_mutex.lock().await; g.as_ref().ok_or_else(|| anyhow::anyhow!("DB Error"))?.clone() };

    // PHASE 3: Hybrid DOM Extraction (Rust Scraper)
    if !is_detail {
        log_task_progress(app_handle, &task.id, &json!({ "category": "Extraction", "summary": "Direct DOM extraction starting...", "spinner": "⠋" }));
        let target_selector = selector_info.get("item").and_then(|s| s.as_str()).unwrap_or("body").to_string();
        
        {
            let document = scraper::Html::parse_document(&clean);
            let field_selectors = selector_info.get("selectors").and_then(|v| v.as_object());
            
            let sel_res = scraper::Selector::parse(&target_selector);
            if let Ok(sel) = sel_res {
                for item_node in document.select(&sel) {
                    let mut item_json = json!({});
                    let mut has_data = false;
                    
                    if let Some(subs) = field_selectors {
                        for (field_name, sel_val) in subs {
                            let sub_sel_str = sel_val.as_str().unwrap_or("");
                            if let Ok(sub_sel) = scraper::Selector::parse(sub_sel_str) {
                                if let Some(found_el) = item_node.select(&sub_sel).next() {
                                    let text = found_el.text().collect::<Vec<_>>().join("").trim().to_string();
                                    if !text.is_empty() { 
                                        item_json.as_object_mut().unwrap().insert(field_name.clone(), json!(text)); 
                                        has_data = true; 
                                    }
                                } else {
                                    // [METICULOUS] Check next sibling if multi-row pattern detected
                                    if let Some(next_sibling) = item_node.next_sibling() {
                                        if let Some(sibling_el) = scraper::ElementRef::wrap(next_sibling) {
                                            if sibling_el.select(&sub_sel).next().is_some() {
                                                if let Some(found_el) = sibling_el.select(&sub_sel).next() {
                                                    let text = found_el.text().collect::<Vec<_>>().join("").trim().to_string();
                                                    item_json.as_object_mut().unwrap().insert(field_name.clone(), json!(text));
                                                    has_data = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        let text = item_node.text().collect::<Vec<_>>().join(" ").trim().to_string();
                        if !text.is_empty() { 
                            item_json.as_object_mut().unwrap().insert("text".to_string(), json!(text)); 
                            has_data = true; 
                        }
                    }

                    if has_data {
                        if let Some(link_el) = item_node.select(&scraper::Selector::parse("a[href]").unwrap()).next() {
                            item_json.as_object_mut().unwrap().insert("link".to_string(), json!(link_el.value().attr("href").unwrap_or("")));
                        }
                        all_extracted_items.push(item_json);
                    }
                }
            }
        }
    }

    // [METICULOUS-PROXY-PARITY] Upsert page info with correct grouping (bcc)
    {
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
            let _ = store.upsert_item("pages", &page_id, "pages", page_data, None, Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(raw_path), None).await;
        }
    }

    // PHASE 4: AI Refinement (Detail Target-Scan vs List Refine)
    let mut extracted_data = json!({});
    if is_detail {
        log_task_progress(app_handle, &task.id, &json!({ "category": "Extraction", "summary": "Baking detail extraction prompt...", "spinner": "⠋" }));
        
        let target_selector = selector_info.get("node").or(selector_info.get("item")).and_then(|v| v.as_str()).unwrap_or("body");
        let content_pug = {
            let document = scraper::Html::parse_document(&clean);
            let mut pug_output = String::new();
            if let Ok(selector) = scraper::Selector::parse(target_selector) {
                if let Some(element_ref) = document.select(&selector).next() {
                    let node_ref = element_ref.id();
                    parsing::generate_pug_lines(document.tree.get(node_ref).unwrap(), 0, &mut pug_output, &PugMode::FullContent);
                }
            }
            if pug_output.is_empty() { light_pug.clone() } else { pug_output }
        };

        let ins = parsing::item2json(&page_type, &url, "english");
        // 1. Bake System Prompt for Detail Extraction
        model.bake_system_prompt(&task.r#ref, "detail_extract", &ins, Some(cancellation_token.clone())).await?;

        // 2. Stitch and Inference
        {
            let kv_dir = utils::paths::get_kv_dir(None, Some(&task.r#ref));
            let pug_path = kv_dir.join("shared_pug_base.safetensors");
            let ins_path = kv_dir.join("shared_prompt_detail_extract.safetensors");
            let gen_arc = model.generator.clone();
            tokio::task::spawn_blocking(move || {
                let mut gen_guard = gen_arc.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    gen.load_kv_stitched(&[pug_path, ins_path])?;
                }
                Ok::<(), anyhow::Error>(())
            }).await??;
        }

        let query_text = format!("[TARGET CONTENT]\n{}\n\nACTION: SAVE", content_pug);
        let res = model.chat("", &query_text, Some(cancellation_token.clone()), Some("detail_extraction_nobridge".to_string())).await?;
        extracted_data = parsing::parse_json_from_llm(&res);
        all_extracted_items = vec![extracted_data.clone()];
    } else if !all_extracted_items.is_empty() {
        log_task_progress(app_handle, &task.id, &json!({ "category": "Refinement", "summary": "Baking refinement instruction...", "spinner": "⠋" }));
        
        let ins = parsing::list2json(&page_type, "english");
        // 1. Bake System Prompt for Batch Refinement
        model.bake_system_prompt(&task.r#ref, "batch_refine", &ins, Some(cancellation_token.clone())).await?;

        let mut refined_items = Vec::new();
        let batch_size = 10;
        
        for (batch_idx, batch) in all_extracted_items.chunks(batch_size).enumerate() {
            if cancellation_token.load(Ordering::Relaxed) { break; }
            
            let start_item = batch_idx * batch_size + 1;
            let end_item = std::cmp::min(start_item + batch.len() - 1, all_extracted_items.len());
            
            println!("[PROCESS] Refining batch {} (Items {}-{} of {})", batch_idx + 1, start_item, end_item, all_extracted_items.len());
            
            // 2. Stitch (Each batch needs a clean context start)
            {
                let kv_dir = utils::paths::get_kv_dir(None, Some(&task.r#ref));
                let pug_path = kv_dir.join("shared_pug_base.safetensors");
                let ins_path = kv_dir.join("shared_prompt_batch_refine.safetensors");
                let gen_arc = model.generator.clone();
                tokio::task::spawn_blocking(move || {
                    let mut gen_guard = gen_arc.blocking_lock();
                    if let Some(gen) = gen_guard.as_mut() {
                        gen.load_kv_stitched(&[pug_path, ins_path])?;
                    }
                    Ok::<(), anyhow::Error>(())
                }).await??;
            }

            let batch_text: String = batch.iter().enumerate()
                .map(|(i, it)| format!("Item {}:\n{}", i + 1, it.to_string()))
                .collect::<Vec<_>>().join("\n\n");
                
            let action_flag = if end_item == all_extracted_items.len() { "ACTION: SAVE" } else { "ACTION: INGEST" };
            let query_text = format!("{}\n\n{}", batch_text, action_flag);
            
            let res = model.chat("", &query_text, Some(cancellation_token.clone()), Some(format!("batch_refinement_{}", batch_idx))).await?;
            let parsed = parsing::parse_json_from_llm(&res);
            
            if let Some(arr) = parsed.as_array().or_else(|| parsed.get("items").and_then(|v| v.as_array())) {
                for (i, refined) in arr.iter().enumerate() {
                    if i < batch.len() {
                        let mut merged = batch[i].clone();
                        merge_json_results(&mut merged, refined);
                        refined_items.push(merged);
                    }
                }
            } else {
                refined_items.extend_from_slice(batch);
            }

            log_task_progress(app_handle, &task.id, &json!({ 
                "category": "Refinement", 
                "summary": format!("Processed items {}-{} of {}...", start_item, end_item, all_extracted_items.len()) 
            }));
        }
        all_extracted_items = refined_items;
    }

    // --- PHASE 4: PERSISTENCE & SYNC ---
    model.unload_generator().await;
    let team_id = if task.to.is_empty() { crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000") } else { task.to.clone() };

    let mut normalized_root = json!({});
    if let Some(obj) = extracted_data.as_object() {
        for (k, v) in obj {
            let val = if let Some(inner_obj) = v.as_object() {
                if let Some(value_field) = inner_obj.get("value") { value_field.clone() } else { v.clone() }
            } else { v.clone() };
            normalized_root.as_object_mut().unwrap().insert(k.clone(), val);
        }
    }
    extracted_data = normalized_root;

    // [INTEGRATED-RELAY] Database Side Effects: Order -> Tracking auto-generation
    if page_type == "order" && is_detail {
        if let Some(goods_arr) = extracted_data.get("goods").and_then(|v| v.as_array()) {
            let cc_val = task.cc.to_uppercase(); 
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

    for item_orig in &all_extracted_items {
        if cancellation_token.load(Ordering::Relaxed) { break; }
        
        let mut item = item_orig.clone();
        let mut normalized_item = json!({});
        if let Some(obj) = item.as_object() {
            for (k, v) in obj {
                let val = if let Some(inner) = v.as_object() {
                    if let Some(vf) = inner.get("value") { vf.clone() } else { v.clone() }
                } else { v.clone() };
                normalized_item.as_object_mut().unwrap().insert(k.clone(), val);
            }
        }
        item = normalized_item;

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
        if let Some((queries, _)) = logic::relay(&page_type, &item) {
            for q in queries { 
                if let Ok(Some((_, ex_data))) = store.find_item_by_property(&q.table, &q.column, &q.value).await { 
                    logic::merge_node(&mut final_item, &ex_data); 
                    break; 
                } 
            }
        }

        let text_to_embed = parsing::json_to_natural_language(&final_item);
        let item_digest = crate::utils::hash::digest(&text_to_embed);
        
        let mut existing_vector = None;
        if let Ok(Some(existing_item)) = store.get_item_by_id("items", &item_id).await {
            if existing_item.digest == item_digest {
                existing_vector = Some(existing_item.vector);
            }
        }

        let vector = if let Some(v) = existing_vector { v } else { model.get_embedding(text_to_embed).await? };

        let cc_for_hash = task.cc.clone(); 
        let bcc = crate::utils::hash::hash_id(&format!("{}{}", page_type, cc_for_hash));
        let ref_val = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, task.cc, link));

        let _ = store.upsert_item(&page_type, &item_id, &page_type, final_item.clone(), Some(vector.clone()), Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)).await;
        let _ = store.upsert_item("items", &item_id, &page_type, final_item, Some(vector), Some(&task.from), Some(&team_id), Some(&task.cc), Some(&bcc), Some(&ref_val), Some(&item_digest)).await;
    }

    let final_summary = format!("Extraction Complete ({} items saved)", all_extracted_items.len());
    let _ = store.update_message_status(&task.id, logic::parse_status("complete"), Some(&final_summary)).await;
    let payload = json!({ "task_id": task.id, "category": "Done", "summary": final_summary, "spinner": "✅", "data": extracted_data });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);
    
    println!("[PROCESS] Task {} completed. Model unloaded.", task.id);
    Ok(())
}

fn cleanup_task_resources(task_id: &str, app_handle: Option<&tauri::AppHandle>) {
    let _ = fs::remove_dir_all(utils::paths::get_task_specific_dir(app_handle, task_id));
}

pub fn pre_fetch_weights(path: &std::path::Path) -> Result<()> {
    use std::io::Read;
    use rayon::prelude::*;

    println!("[PRE-FETCH] Parallel Warm-up for weights in: {:?}", path);
    if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            let files: Vec<_> = entries.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().map_or(false, |ext| ext == "gguf" || ext == "safetensors"))
                .collect();

            // Parallelize file pre-fetching using Rayon
            files.par_iter().for_each(|p| {
                if let Ok(mut file) = std::fs::File::open(p) {
                    // Larger 8MB buffer for high-speed sequential read
                    let mut buffer = vec![0u8; 8 * 1024 * 1024]; 
                    while let Ok(n) = file.read(&mut buffer) { 
                        if n == 0 { break; } 
                    }
                }
            });
        }
    }
    println!("[PRE-FETCH] Warm-up complete.");
    Ok(())
}

async fn wait_for_resources_settled(target_vram_mb: u64, target_ram_mb: u64, cancellation_token: &Arc<AtomicBool>) -> Result<()> {
    use nvml_wrapper::Nvml;
    use sysinfo::System;
    let mut sys = System::new_all();
    let nvml = Nvml::init().ok();
    let target_vram_bytes = target_vram_mb * 1024 * 1024;
    let target_ram_bytes = target_ram_mb * 1024 * 1024;
    let start_time = std::time::Instant::now();
    loop {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        sys.refresh_memory(); 
        let current_ram = sys.available_memory();
        let mut current_vram = 0;
        let mut has_gpu = false;
        if let Some(ref nvml_inst) = nvml {
            if let Ok(count) = nvml_inst.device_count() {
                for i in 0..count { if let Ok(dev) = nvml_inst.device_by_index(i) { if let Ok(mem) = dev.memory_info() { if mem.free > current_vram { current_vram = mem.free; } has_gpu = true; } } }
            }
        }
        if (!has_gpu || current_vram >= target_vram_bytes) && current_ram >= target_ram_bytes { break; }
        if start_time.elapsed().as_secs() > 20 { break; }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Ok(())
}
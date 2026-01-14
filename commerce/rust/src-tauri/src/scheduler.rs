use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use crate::store::{VectorStore, Task};
use crate::logic;
use crate::parsing::{self, PugMode};
use crate::model::LogisModel;
use serde_json::{Value, json};
use anyhow::Result;

pub async fn start_background_worker(
    store: Arc<Mutex<Option<VectorStore>>>,
    model: Arc<Mutex<Option<LogisModel>>>,
) {
    println!("[Scheduler] Background worker started.");
    
    tokio::spawn(async move {
        loop {
            // 1. Sleep for interval
            sleep(Duration::from_secs(10)).await;
            
            // 2. Acquire Store Lock & Get Tasks
            let mut pending_tasks = Vec::new();
            {
                let store_opt = store.lock().await;
                if let Some(db) = store_opt.as_ref() {
                    match db.get_pending_tasks(5).await {
                        Ok(tasks) => pending_tasks = tasks,
                        Err(e) => println!("[Scheduler] Failed to fetch tasks: {{:?}}", e),
                    }
                }
            } // Release lock

            // 3. Process Tasks
            for task in pending_tasks {
                println!("[Scheduler] Processing task: {{}}", task.id);
                if let Err(e) = process_task(task.clone(), &store, &model).await {
                    println!("[Scheduler] Task failed: {{:?}}", e);
                    // Handle error state update here if needed
                }
            }
        }
    });
}

async fn process_task(
    task: Task, 
    store_mutex: &Arc<Mutex<Option<VectorStore>>>,
    model_mutex: &Arc<Mutex<Option<LogisModel>>>)
-> Result<()> {
    
    // Parse Task Data
    let task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    let url = task_data.get("link").and_then(|s| s.as_str()).unwrap_or("");
    let language = "english"; // Default, ideally detect from content or task

    if url.is_empty() {
        return Ok(())
    }

    // 1. Fetch Page Content (Using Reqwest for speed, headless browser for SPA if needed)
    // TODO: Switch to automation::run_browser_automation if HTML is empty/JS-heavy
    let html = reqwest::get(url).await?.text().await?;
    
    // =================================================================================
    // STEP 1: Classification (Structure Only) - "Map Outline"
    // =================================================================================
    
    // Convert to Light Pug (No text, no heavy attrs)
    let light_pug = parsing::convert_to_clean_pug(&html, PugMode::StructureOnly);
    
    // Classify Type
    let model_guard = model_mutex.lock().await;
    let page_info_str = if let Some(model) = model_guard.as_ref() {
        let system = r#" 
            Analyze the provided Pug structure (Ids, Classes, Tags).
            Identify the Page Type and if it is a List or Detail page.
            Return JSON: { "type": "goods"|"order"|"tracking"|"event"|"unknown", "view": "list"|"detail" }
        "#;
        // Truncate input for speed
        let input = light_pug.chars().take(3000).collect::<String>(); 
        model.chat(system, &input).await?
    } else {
        "{}".to_string()
    };
    drop(model_guard);
    
    let page_info: Value = serde_json::from_str(&page_info_str).unwrap_or(json!({"type": "unknown", "view": "unknown"}));
    let page_type = page_info.get("type").and_then(|s| s.as_str()).unwrap_or("unknown");
    let view_type = page_info.get("view").and_then(|s| s.as_str()).unwrap_or("unknown");

    println!("[Scheduler] Identified: {} / {{}}", page_type, view_type);

    if page_type == "unknown" {
        // Skip or mark error
        return Ok(());
    }

    // =================================================================================
    // STEP 2: Extraction (Full Content) - "Zoom In"
    // =================================================================================

    // Convert to Full Pug (With text, specific attrs)
    let content_pug = parsing::convert_to_clean_pug(&html, PugMode::FullContent);
    
    // Generate Schema Prompt based on Step 1 result
    let prompt = if view_type == "list" {
        parsing::list2json(language)
    } else {
        parsing::item2json(page_type, url, language)
    };

    let model_guard = model_mutex.lock().await;
    let extracted_json_str = if let Some(model) = model_guard.as_ref() {
         model.chat("Extract information matching the JSON structure.", &format!("{{}}\n\nData:\n{{}}", prompt, content_pug)).await?
    } else {
        "{}".to_string()
    };
    drop(model_guard);

    let mut extracted_data: Value = serde_json::from_str(&extracted_json_str).unwrap_or(json!({}));
    
    // Inject type if missing
    if extracted_data.get("type").is_none() {
        if let Some(obj) = extracted_data.as_object_mut() {
            obj.insert("type".to_string(), json!(page_type));
        }
    }

    // =================================================================================
    // STEP 3: Logic Relay, Merge & Save
    // =================================================================================

    if let Some((queries, _merge_info)) = logic::relay(page_type, &extracted_data) {
        println!("[Scheduler] Relay logic triggered. Queries: {{}}", queries.len());
        
        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
            let mut target_data = extracted_data.clone();
            let mut target_id = format!("{{}}_{{}}", page_type, chrono::Utc::now().timestamp_millis());
            
            let mut found_existing = false;
            
            for query in queries {
                if let Ok(Some((id, existing_data))) = db.find_item_by_property(&query.column, &query.value).await {
                    println!("[Scheduler] Found existing item: {{}}", id);
                    logic::merge_node(&mut target_data, &existing_data);
                    target_id = id;
                    found_existing = true;
                    break;
                }
            }
            
            if !found_existing {
                if let Some(id_val) = target_data.get("id").and_then(|v| v.as_str()) {
                    target_id = id_val.to_string();
                }
            }

            // Generate Vector
            let text_to_embed = target_data.to_string();
            
            drop(store_guard); 
            let model_guard = model_mutex.lock().await;
            let vector = if let Some(model) = model_guard.as_ref() {
                model.get_embedding(text_to_embed).await.ok()
            } else {
                None
            };
            drop(model_guard);
            
            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                println!("[Scheduler] Saving item: {{}}", target_id);
                let _ = db.upsert_item(&target_id, page_type, target_data, vector).await;
                let _ = db.update_task_status(&task.id, "done").await;
            }
        }
    } else {
        // Simple Save (No Relay)
        let store_guard = store_mutex.lock().await;
        if let Some(db) = store_guard.as_ref() {
             let id = extracted_data.get("id")
                .and_then(|s| s.as_str())
                .unwrap_or_else(|| task.id.as_str())
                .to_string();
                
             let _ = db.upsert_item(&id, page_type, extracted_data, None).await;
             let _ = db.update_task_status(&task.id, "done").await;
        }
    }
    
    Ok(())
}
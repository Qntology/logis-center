mod model;
mod store;
mod automation;
mod parsing;
mod logic;
mod scheduler;

pub mod models;
pub mod utils;
pub mod openai_types;
pub mod position_embed;
pub mod chat_template;
pub mod tokenizer;

use tauri::{State, Manager, Listener}; // Added Manager
use tokio::sync::Mutex as TokioMutex;
use model::LogisModel;
use store::{VectorStore, TradeDocument};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use serde_json::{Value, json};

pub struct AppState {
    pub model: Arc<TokioMutex<Option<LogisModel>>>,
    pub store: Arc<TokioMutex<Option<VectorStore>>>,
    pub cancellation_token: Arc<AtomicBool>,
}

#[tauri::command]
async fn stop_current_extraction(state: State<'_, AppState>) -> Result<String, String> {
    state.cancellation_token.store(true, Ordering::SeqCst);
    
    // 1. Update DB to clear active tasks so they reflect as 'Cancelled' in UI
    {
        let store_guard = state.store.lock().await;
        if let Some(db) = store_guard.as_ref() {
            if let Ok(active_tasks) = db.get_pending_tasks(50).await {
                for task in active_tasks {
                    let _ = db.update_task_status(&task.id, 3).await; // 3 = Cancelled
                    let _ = db.update_message_status(&task.id, 3, Some("Force stopped by user")).await;
                }
            }
        }
    }

    // 2. Force drop model and store to release VRAM/RAM immediately
    {
        let mut model_guard = state.model.lock().await;
        *model_guard = None;
    }
    
    {
        let mut store_guard = state.store.lock().await;
        *store_guard = None;
    }

    println!("[STOP] Cancellation signal sent, DB updated, and model/store dropped.");
    Ok("Stop signal sent and resources released.".to_string())
}

#[tauri::command]
async fn unload_model(state: State<'_, AppState>) -> Result<String, String> {
    {
        let mut model_guard = state.model.lock().await;
        *model_guard = None;
    }
    
    {
        let mut store_guard = state.store.lock().await;
        *store_guard = None;
    }

    println!("[UNLOAD] Model and Store explicitly dropped from memory.");
    Ok("Memory cleared.".to_string())
}

#[tauri::command]
async fn resize_window(app_handle: tauri::AppHandle, width: f64, height: f64) {
    if let Some(window) = app_handle.get_webview_window("main") {
        let _ = window.set_size(tauri::Size::Logical(tauri::LogicalSize { width, height }));
    }
}

#[tauri::command]
async fn start_drag(app_handle: tauri::AppHandle) {
    if let Some(window) = app_handle.get_webview_window("main") {
        let _ = window.start_dragging();
    }
}

#[tauri::command]
async fn move_to_center(app_handle: tauri::AppHandle) {
    if let Some(window) = app_handle.get_webview_window("main") {
        let _ = window.center();
    }
}

#[tauri::command]
async fn move_to_top_center(app_handle: tauri::AppHandle) {
    if let Some(window) = app_handle.get_webview_window("main") {
        if let Ok(Some(monitor)) = window.current_monitor() {
            let screen_size = monitor.size(); // PhysicalSize
            let scale_factor = monitor.scale_factor();
            let screen_width = screen_size.width as f64 / scale_factor;
            
            // Get current window size
            if let Ok(factor) = window.scale_factor() {
                if let Ok(size) = window.outer_size() {
                    let win_width = size.width as f64 / factor;
                    let new_x = (screen_width - win_width) / 2.0;
                    
                    let _ = window.set_position(tauri::Position::Logical(tauri::LogicalPosition {
                        x: new_x,
                        y: 0.0,
                    }));
                }
            }
        }
    }
}

#[tauri::command]
async fn launch_browser(
    app_handle: tauri::AppHandle,
    browser: String,
    url: String,
    script: String,
) -> Result<String, String> {
    automation::run_browser_automation(browser, url, script, app_handle)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn launch_best_browser(
    app_handle: tauri::AppHandle,
    url: String,
) -> Result<String, String> {
    let available = automation::get_available_browsers();
    // Priority: Chrome -> Edge -> Firefox
    let target = if available.iter().any(|b| b.name == "chrome") {
        "chrome"
    } else if available.iter().any(|b| b.name == "edge") {
        "edge"
    } else if available.iter().any(|b| b.name == "firefox") {
        "firefox"
    } else {
        return Err("No supported browser found.".to_string());
    };
    
    // Launch with default empty script
    automation::run_browser_automation(target.to_string(), url, "".to_string(), app_handle)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn check_available_browsers() -> Vec<automation::BrowserStatus> {
    automation::get_available_browsers()
}

// --- Helper to Generate Rich Summary (Moved to model.rs) ---

#[tauri::command]
async fn summarize_image(
    state: State<'_, AppState>,
    _app_handle: tauri::AppHandle,
    image_path: String,
) -> Result<String, String> {
    println!("[INVOKE-01] summarize_image (Queue Integration) for path: {}", image_path);

    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        let task_id = format!("img_{}", uuid::Uuid::new_v4().to_string());
        let now = chrono::Utc::now().timestamp_millis();
        
        let task_data = json!({
            "image_path": image_path,
            "id": task_id
        });

        let task = crate::store::Task {
            id: task_id.clone(),
            r#type: "image_extraction".to_string(),
            from_source: "manual_upload".to_string(),
            to_dest: "local".to_string(),
            cc: "".to_string(),
            bcc: "".to_string(),
            ref_id: "manual".to_string(),
            data_json: task_data.to_string(),
            created_at: now,
            updated_at: now,
            status: crate::logic::parse_status("pending"),
        };

        match db.add_task(task).await {
            Ok(_) => Ok(format!("Task {} queued successfully.", task_id)),
            Err(e) => Err(format!("Failed to queue image task: {}", e)),
        }
    } else {
        Err("Database not initialized.".to_string())
    }
}

#[tauri::command]
async fn search_documents(
    state: State<'_, AppState>,
    query: String,
    limit: usize,
    offset: usize,
) -> Result<Vec<(String, String, f32)>, String> {
    let mut store_guard = state.model.lock().await; // Using model lock or store lock based on current usage
    // Actually search_documents uses store, let's re-verify the lock usage from the file.
    // Based on the previous read_file of lib.rs:
    
    let mut store_guard = state.store.lock().await;
    if store_guard.is_none() {
        let db_path = "data/lancedb";
        match VectorStore::new(db_path).await {
            Ok(s) => { *store_guard = Some(s); },
            Err(e) => return Err(format!("Failed to load DB: {}", e)),
        }
    }
    
    let model_guard = state.model.lock().await;
    let query_vec = if let Some(model) = model_guard.as_ref() {
        model.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 768])
    } else {
        vec![0.0; 768]
    };

    if let Some(store) = store_guard.as_ref() {
        store.search_items("items", &query, query_vec, limit, None).await.map_err(|e| e.to_string())
    } else {
        Err("DB not initialized".to_string())
    }
}

// Helper to convert structured LLM conditions to SQL filter strings
fn convert_conditions_to_sql(ctx: &Value) -> Option<String> {
    let mut filters = Vec::new();
    
    // 1. Type Filter
    if let Some(t) = ctx.get("type").and_then(|v| v.as_str()) {
        if !t.is_empty() { filters.push(format!("type = '{}'", t)); }
    }

    // 2. Condition Filters (Price, Date, etc.)
    if let Some(cond) = ctx.get("condition") {
        if let Some(price) = cond.get("price") {
            if let Some(gte) = price.get("gte").and_then(|v| v.as_f64()) { filters.push(format!("amount >= {}", gte)); }
            if let Some(lte) = price.get("lte").and_then(|v| v.as_f64()) { filters.push(format!("amount <= {}", lte)); }
        }
    }

    if filters.is_empty() { None } else { Some(filters.join(" AND ")) }
}

#[tauri::command]
async fn get_all_documents(
    state: State<'_, AppState>,
    limit: usize,
    offset: usize,
) -> Result<Vec<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        let mut results = store.get_all_items("items", limit, offset).await.map_err(|e| e.to_string())?;
        
        // [DYNAMIC] Convert JSON to Natural Language for UI display only
        for doc in results.iter_mut() {
            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&doc.json_data) {
                doc.text = parsing::json_to_natural_language(&json_val);
            }
        }
        
        Ok(results)
    } else {
        Err("DB not initialized".to_string())
    }
}

#[tauri::command]
async fn get_document(
    state: State<'_, AppState>,
    uuid: String,
) -> Result<Option<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        let mut doc_opt = store.get_item_by_id("items", &uuid).await.map_err(|e| e.to_string())?;
        
        // [OPTIMIZED] Preserve high-quality summary from DB if available
        if let Some(ref mut doc) = doc_opt {
            if doc.text.is_empty() {
                if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&doc.json_data) {
                    doc.text = parsing::json_to_natural_language(&json_val);
                }
            }
        }
        
        Ok(doc_opt)
    } else {
        Err("DB not initialized".to_string())
    }
}

#[tauri::command]
async fn update_document(
    _state: State<'_, AppState>,
    _uuid: String,
    _json_data: String,
) -> Result<String, String> {
    Ok("Not implemented".to_string())
}

#[tauri::command]
async fn delete_document(
    _state: State<'_, AppState>,
    _uuid: String,
) -> Result<String, String> {
    Ok("Not implemented".to_string())
}

#[tauri::command]
async fn delete_documents(
    _state: State<'_, AppState>,
    _uuids: Vec<String>,
) -> Result<String, String> {
    Ok("Not implemented".to_string())
}

#[tauri::command]
async fn ai_search_complex(
    state: State<'_, AppState>,
    query: String,
    language: String,
) -> Result<Value, String> {
    let mut model_guard = state.model.lock().await;
    if model_guard.is_none() {
        if let Ok(m) = LogisModel::new(None).await { *model_guard = Some(m); }
        else { return Err("Failed to load model".to_string()); }
    }
    let model = model_guard.as_ref().unwrap();

    // 1. Structured Parse (Para2Graph -> Graph2Contexts)
    let structured_query = model.parse_query_structured(query.clone(), &language).await.map_err(|e| e.to_string())?;
    
    // 2. Perform searches for each segment
    let mut all_results = Vec::new();
    let mut store_guard = state.store.lock().await;
    if store_guard.is_none() {
        let db_path = "data/lancedb";
        if let Ok(s) = VectorStore::new(db_path).await {
            let _ = s.init_all_tables().await;
            *store_guard = Some(s);
        }
    }

    if let (Some(store), Some(ctx_arr)) = (store_guard.as_ref(), structured_query.get("context").and_then(|v| v.as_array())) {
        for ctx in ctx_arr {
            let text = ctx.get("text").and_then(|v| v.as_str()).unwrap_or("");
            if text.is_empty() { continue; }
            
            // [FIX] Convert extracted conditions to SQL filter
            let sql_filter = convert_conditions_to_sql(ctx);
            
            let emb = model.get_embedding(text.to_string()).await.unwrap_or(vec![0.0; 768]);
            // Now passing the filter to search_items
            if let Ok(results) = store.search_items("items", text, emb, 5, sql_filter).await {
                for (id, content, score) in results {
                    all_results.push(json!({
                        "id": id,
                        "text": content,
                        "score": score,
                        "context_type": ctx.get("type")
                    }));
                }
            }
        }
    }

    Ok(json!({
        "structured": structured_query,
        "results": all_results
    }))
}

#[tauri::command]
async fn check_query_intent(
    _state: State<'_, AppState>,
    _query: String,
) -> Result<String, String> {
    Ok("SEARCH".to_string())
}

#[tauri::command]
async fn deep_research_command(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
    query: String,
    _doc_id: Option<String>,
) -> Result<String, String> {
    let mut model_guard = state.model.lock().await;
    if model_guard.is_none() {
        if let Ok(m) = LogisModel::new(None).await {
            *model_guard = Some(m);
        } else {
            return Err("Failed to load model".to_string());
        }
    }
    let model = model_guard.as_ref().unwrap();

    // 1. Context Gathering
    let mut context_data = String::new();
    let mut store_guard = state.store.lock().await;
    
    if store_guard.is_none() {
        // Try init
        let db_path = "data/lancedb";
        let _ = std::fs::create_dir_all(db_path);
        if let Ok(s) = VectorStore::new(db_path).await {
            let _ = s.init_task_table().await;
            let _ = s.init_all_tables().await;
            *store_guard = Some(s);
        }
    }
    
    if let Some(store) = store_guard.as_ref() {
        // General search for context
        let emb = model.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 768]);
        if let Ok(results) = store.search_items("items", &query, emb, 3, None).await {
            let docs: Vec<String> = results.iter()
                .map(|(_, text, _)| format!("- {}", text))
                .collect();
            context_data = docs.join("\n");
        }
    }
    
    // 2. Run Deep Research
    model.run_deep_research(query, context_data, &app_handle, Some(state.cancellation_token.clone())).await.map_err(|e| e.to_string())
}

#[tauri::command]

async fn proxy_fetch(

    url: String,

    method: String,

    headers: std::collections::HashMap<String, String>,

    body: Option<Value>,

    session_params: Option<Value>, // { hash, token, cc }

) -> Result<Value, String> {

    let client = reqwest::Client::builder()
        .use_native_tls()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .build()
        .map_err(|e| e.to_string())?;



    let mut target_url = url::Url::parse(&url).map_err(|e| e.to_string())?;



    // [DETAIL 1] Inject Session into Query Params (Content.js logic)

    if let Some(sp) = session_params {

        let mut query = target_url.query_pairs_mut();

        if let Some(hash) = sp.get("hash").and_then(|v| v.as_str()) { query.append_pair("hash", hash); }

        if let Some(token) = sp.get("token").and_then(|v| v.as_str()) { query.append_pair("token", token); }

        if let Some(cc) = sp.get("cc").and_then(|v| v.as_str()) { query.append_pair("cc", cc); }

    }



    let mut req_builder = match method.to_uppercase().as_str() {

        "POST" => client.post(target_url),

        "PUT" => client.put(target_url),

        "DELETE" => client.delete(target_url),

        _ => client.get(target_url),

    };



    for (k, v) in headers { req_builder = req_builder.header(k, v); }
    if let Some(b) = body { req_builder = req_builder.json(&b); }

    let response = req_builder.send().await.map_err(|e| e.to_string())?;
    let status = response.status();
    
    // Read response as text first to handle non-JSON cases (HTML, error pages, etc.)
    let text_res = response.text().await.map_err(|e| e.to_string())?;

    let json_res: Value = match serde_json::from_str(&text_res) {
        Ok(v) => v,
        Err(_) => {
            // If it's not JSON but request was successful, wrap it or return as text
            if status.is_success() {
                json!({ "text": text_res })
            } else {
                return Err(format!("Server error {} (Not JSON): {}", status, text_res));
            }
        }
    };

    if !status.is_success() {
        return Err(format!("Server returned {}: {}", status, json_res));
    }

    Ok(json_res)
}





#[derive(serde::Deserialize)]
struct ActiveTaskQuery {
    cc: String,
    #[serde(alias = "refId")]
    ref_id: String,
}

#[tauri::command]
async fn check_active_task(state: State<'_, AppState>, payload: ActiveTaskQuery) -> Result<bool, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        db.has_active_task(&payload.cc, &payload.ref_id).await.map_err(|e| e.to_string())
    } else { Ok(false) }
}



#[tauri::command]
async fn initialize_hub(
    state: State<'_, AppState>,
    address: String,
    email: String,
    flag: String,
) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        match store.initialize_user_profiles(&address, &email, &flag).await {
            Ok(_) => Ok(format!("Hub initialized for address: {}", address)),
            Err(e) => Err(format!("Initialization failed: {}", e)),
        }
    } else {
        Err("Store not initialized".to_string())
    }
}

#[tauri::command]
async fn get_chat_messages(state: State<'_, AppState>) -> Result<Vec<Value>, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        db.get_all_messages(50).await.map_err(|e| e.to_string())
    } else { Ok(vec![]) }
}



#[tauri::command]
async fn get_known_pages(state: State<'_, AppState>) -> Result<Vec<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        store.get_all_items("pages", 20, 0).await.map_err(|e| e.to_string())
    } else { Ok(vec![]) }
}

#[tauri::command]
async fn get_known_users(state: State<'_, AppState>) -> Result<Vec<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        store.get_all_items("users", 20, 0).await.map_err(|e| e.to_string())
    } else { Ok(vec![]) }
}

#[tauri::command]
async fn set_login_state(
    state: State<'_, AppState>,
    is_logged_in: bool,
    token: Option<String>,
) -> Result<String, String> {

    let store_guard = state.store.lock().await;

    if let Some(store) = store_guard.as_ref() {

        let mut config = store.load_config();

        config.is_logged_in = is_logged_in;

        config.auth_token = token;

        

        match store.save_config(&config) {

            Ok(_) => Ok(format!("Login state set to: {}", is_logged_in)),

            Err(e) => Err(e.to_string()),

        }

    } else {

        Err("Store not initialized".to_string())

    }

}



#[tauri::command]
async fn extract_html_from_current_tab() -> Result<String, String> {
    automation::extract_html_from_current_tab().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_browser_status() -> Result<String, String> {
    let guard = automation::GLOBAL_BROWSER.lock().await;
    if guard.is_some() { 
        return Ok("running".to_string()); 
    }
    Ok("stopped".to_string())
}

#[tauri::command]
async fn get_active_tasks(state: State<'_, AppState>) -> Result<Vec<store::Task>, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        // Fetch tasks with status 10 (pending) or 1 (progress)
        db.get_pending_tasks(10).await.map_err(|e| e.to_string())
    } else {
        Ok(vec![])
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let model = Arc::new(TokioMutex::new(None));
    let store = Arc::new(TokioMutex::new(None));
    let cancellation_token = Arc::new(AtomicBool::new(false));

    tauri::Builder::default()
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .manage(AppState {
            model: model.clone(),
            store: store.clone(),
            cancellation_token: cancellation_token.clone(),
        })
        .setup(|app| {
            let setup_store = app.state::<AppState>().store.clone();
            tauri::async_runtime::spawn(async move {
                let mut store_guard = setup_store.lock().await;
                let db_path = "data/lancedb";
                let _ = std::fs::create_dir_all(db_path);
                if let Ok(s) = VectorStore::new(db_path).await {
                    println!("[Setup] VectorStore initialized.");
                    let _ = s.init_task_table().await;
                    let _ = s.init_all_tables().await;
                    *store_guard = Some(s);
                }
            });

            let scheduler_store = app.state::<AppState>().store.clone();
            let scheduler_model = app.state::<AppState>().model.clone();
            let scheduler_cancel = app.state::<AppState>().cancellation_token.clone();
            let scheduler_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                scheduler::start_background_worker(scheduler_store, scheduler_model, scheduler_cancel, scheduler_handle).await;
            });

            let auto_reconnect_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let _ = automation::try_reconnect_existing_browser(auto_reconnect_handle).await;
            });

            let event_store = app.state::<AppState>().store.clone();
            app.listen("new-task-from-browser", move |event| {
                if let Ok(payload_val) = serde_json::from_str::<serde_json::Value>(event.payload()) {
                    let store_clone = event_store.clone();
                    tauri::async_runtime::spawn(async move {
                        let store_guard = store_clone.lock().await;
                        if let Some(db) = store_guard.as_ref() {
                            let now = chrono::Utc::now().timestamp_millis();
                            let from_addr = payload_val.get("from").and_then(|v| v.as_str()).unwrap_or("0x0000000000000000000000000000000000000000").to_string();
                            let team_id = payload_val.get("to").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| crate::utils::hash::hash_id(&from_addr));
                            let task = crate::store::Task {
                                id: payload_val.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                                r#type: payload_val.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
                                from_source: from_addr, to_dest: team_id,
                                cc: payload_val.get("cc").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                                bcc: payload_val.get("bcc").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                                ref_id: payload_val.get("ref_id").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                                data_json: payload_val.to_string(), created_at: now, updated_at: now, status: 10,
                            };
                            let msg_content = format!("Task Started: {}", payload_val.get("link").and_then(|v| v.as_str()).unwrap_or("Unknown URL"));
                            let _ = db.add_message(&uuid::Uuid::new_v4().to_string(), "system_task", &msg_content, Some(&task.id), Some(1)).await;
                            let _ = db.add_task(task).await;
                        }
                    });
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            summarize_image, search_documents, get_all_documents, get_document, check_query_intent, deep_research_command, ai_search_complex,
            launch_browser, launch_best_browser, extract_html_from_current_tab, stop_current_extraction, check_available_browsers,
            resize_window, start_drag, move_to_top_center, set_login_state, check_active_task, get_chat_messages, proxy_fetch,
            get_known_pages, get_known_users, initialize_hub, get_browser_status, get_active_tasks, unload_model
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

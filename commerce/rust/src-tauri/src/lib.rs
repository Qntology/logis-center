pub mod _parsing;
pub mod automation;
pub mod logic;
pub mod model;
pub mod models;
pub mod openai_types;
pub mod parsing;
pub mod scheduler;
pub mod store;
pub mod tokenizer;
pub mod utils;
pub mod chat_template;

use tauri::{Manager, State, Listener};
use tokio::sync::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use serde_json::{json, Value};
use crate::model::LogisModel;
use crate::store::{VectorStore, TradeDocument};

pub struct AppState {
    pub store: Arc<Mutex<Option<VectorStore>>>,
    pub model: Arc<Mutex<Option<LogisModel>>>,
    pub cancellation_token: Arc<AtomicBool>,
}

#[tauri::command]
async fn stop_current_extraction(
    state: State<'_, AppState>,
    task_id: Option<String>
) -> Result<String, String> {
    state.cancellation_token.store(true, Ordering::SeqCst);
    crate::utils::set_extraction_stop_signal(true);
    
    if let Ok(store_guard) = state.store.try_lock() {
        if let Some(db) = store_guard.as_ref() {
            if let Some(ref id) = task_id {
                let _ = db.update_task_status(id, 3).await; // 3 = Cancel
                let _ = db.delete_message_by_task_id(id).await;
            } else {
                let _ = db.cleanup_zombie_tasks().await;
            }
        }
    }

    if let Ok(mut model_guard) = state.model.try_lock() {
        if let Some(m) = model_guard.as_ref() {
            m.unload_generator().await;
        }
        *model_guard = None;
    }

    Ok("Stop signal sent.".to_string())
}

#[tauri::command]
async fn delete_message(
    state: State<'_, AppState>,
    task_id: String
) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        db.delete_message_by_task_id(&task_id).await.map_err(|e| e.to_string())?;
        Ok(format!("Message for task {} deleted.", task_id))
    } else {
        Err("DB not initialized".to_string())
    }
}

#[tauri::command]
async fn unload_model(state: State<'_, AppState>) -> Result<String, String> {
    {
        let mut model_guard = state.model.lock().await;
        if let Some(m) = model_guard.as_ref() {
            m.unload_generator().await;
        }
        *model_guard = None;
    }
    
    {
        let mut store_guard = state.store.lock().await;
        *store_guard = None;
    }

    state.cancellation_token.store(false, Ordering::SeqCst);
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
async fn move_to_top_center(app_handle: tauri::AppHandle) {
    if let Some(window) = app_handle.get_webview_window("main") {
        if let Ok(Some(monitor)) = window.current_monitor() {
            let screen_size = monitor.size();
            let scale_factor = monitor.scale_factor();
            let screen_width = screen_size.width as f64 / scale_factor;
            
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
async fn set_ignore_cursor_events(app_handle: tauri::AppHandle, ignore: bool) {
    if let Some(window) = app_handle.get_webview_window("main") {
        let _ = window.set_ignore_cursor_events(ignore);
    }
}

#[tauri::command]
async fn launch_best_browser(
    app_handle: tauri::AppHandle,
    url: String,
) -> Result<String, String> {
    let available = automation::get_available_browsers();
    let target = if available.iter().any(|b| b.name == "chrome") {
        "chrome"
    } else if available.iter().any(|b| b.name == "edge") {
        "edge"
    } else if available.iter().any(|b| b.name == "firefox") {
        "firefox"
    } else {
        return Err("No supported browser found.".to_string());
    };
    
    automation::run_browser_automation(target.to_string(), url, "".to_string(), app_handle)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn check_available_browsers() -> Vec<automation::BrowserStatus> {
    automation::get_available_browsers()
}

#[tauri::command]
async fn proxy_fetch(
    url: String,
    method: String,
    headers: std::collections::HashMap<String, String>,
    body: Option<Value>,
    session_params: Option<Value>,
) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .use_native_tls()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .build()
        .map_err(|e| e.to_string())?;

    let mut target_url = url::Url::parse(&url).map_err(|e| e.to_string())?;

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

    for (k, v) in headers.iter() { req_builder = req_builder.header(k, v); }
    
    if let Some(b) = body { 
        if headers.get("Content-Encoding").map(|v| v.as_str()) == Some("gzip") {
            if let Ok(compressed) = crate::utils::compression::compress_value(&b) {
                req_builder = req_builder.body(compressed);
            } else {
                req_builder = req_builder.json(&b);
            }
        } else {
            req_builder = req_builder.json(&b); 
        }
    }

    let response = req_builder.send().await.map_err(|e| e.to_string())?;
    let status = response.status();
    let text_res = response.text().await.map_err(|e| e.to_string())?;

    let json_res: Value = match serde_json::from_str(&text_res) {
        Ok(v) => v,
        Err(_) => {
            if status.is_success() { json!({ "text": text_res }) } 
            else { return Err(format!("Server error {} (Not JSON): {}", status, text_res)); }
        }
    };

    Ok(json_res)
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
async fn get_active_tasks(state: State<'_, AppState>) -> Result<Vec<crate::store::Task>, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        db.get_pending_tasks(10).await.map_err(|e| e.to_string())
    } else {
        Ok(vec![])
    }
}

#[tauri::command]
async fn get_task_logs(app_handle: tauri::AppHandle, task_id: String) -> Result<Vec<Value>, String> {
    let log_path = crate::utils::paths::get_task_log_file(Some(&app_handle), &task_id);
    if !log_path.exists() { return Ok(vec![]); }
    let content = std::fs::read_to_string(log_path).map_err(|e| e.to_string())?;
    let logs = content.lines().filter_map(|line| serde_json::from_str(line).ok()).collect();
    Ok(logs)
}

#[tauri::command]
async fn mark_ui_ready(state: State<'_, AppState>) -> Result<Value, String> {
    scheduler::mark_ui_ready();
    let store_guard = state.store.lock().await;
    let mut tasks = Vec::new();
    let mut pages = Vec::new();
    if let Some(db) = store_guard.as_ref() {
        tasks = db.get_pending_tasks(10).await.unwrap_or_default();
        pages = db.get_all_items("pages", 50, 0, None).await.unwrap_or_default();
    }
    
    let browser_status = {
        let guard = automation::GLOBAL_BROWSER.lock().await;
        if guard.is_some() || automation::is_browser_reachable().await { "running".to_string() } else { "stopped".to_string() }
    };

    Ok(json!({
        "tasks": tasks,
        "pages": pages,
        "browser_status": browser_status,
    }))
}

#[tauri::command]
async fn check_gpu_availability() -> bool {
    let config = crate::utils::get_optimal_device_config();
    !config.is_cpu
}

#[tauri::command]
async fn get_all_documents(
    state: State<'_, AppState>,
    limit: usize,
    offset: usize,
    filter: Option<String>,
) -> Result<Vec<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        store.get_all_items("items", limit, offset, filter).await.map_err(|e| e.to_string())
    } else {
        Err("DB not initialized".to_string())
    }
}

#[tauri::command]
async fn get_chat_messages(
    state: State<'_, AppState>,
    limit: usize,
    offset: usize,
    filter: Option<String>,
) -> Result<Vec<Value>, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        db.get_all_messages(limit, offset, filter).await.map_err(|e| e.to_string())
    } else { Ok(vec![]) }
}

#[tauri::command]
async fn summarize_image(
    state: State<'_, AppState>,
    _app_handle: tauri::AppHandle,
    image_path: String,
) -> Result<String, String> {
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
            from: "manual_upload".to_string(),
            to: "local".to_string(),
            cc: "".to_string(),
            bcc: "".to_string(),
            r#ref: "manual".to_string(),
            data_json: task_data.to_string(),
            created_at: now,
            updated_at: now,
            status: 10, // pending
        };

        match db.add_task(task).await {
            Ok(_) => {
                crate::scheduler::notify_new_task();
                Ok(format!("Task {} queued successfully.", task_id))
            },
            Err(e) => Err(format!("Failed to queue image task: {}", e)),
        }
    } else {
        Err("Database not initialized.".to_string())
    }
}

#[tauri::command]
async fn extract_html_from_current_tab() -> Result<String, String> {
    automation::extract_html_from_current_tab().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_document(
    state: State<'_, AppState>,
    uuid: String,
) -> Result<Option<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        store.get_item_by_id("items", &uuid).await.map_err(|e| e.to_string())
    } else {
        Err("DB not initialized".to_string())
    }
}

#[tauri::command]
async fn delete_document(
    state: State<'_, AppState>,
    uuid: String,
) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        let tables = vec!["items", "sales", "tracking", "event", "users", "pages"];
        for table in tables { let _ = store.delete_item(table, &uuid).await; }
        Ok(format!("Document {} deleted.", uuid))
    } else {
        Err("DB not initialized".to_string())
    }
}

#[tauri::command]
async fn delete_documents(
    state: State<'_, AppState>,
    uuids: Vec<String>,
) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        let tables = vec!["items", "sales", "tracking", "event", "users", "pages"];
        for table in tables { let _ = store.delete_items(table, uuids.clone()).await; }
        Ok(format!("Deleted {} documents.", uuids.len()))
    } else {
        Err("DB not initialized".to_string())
    }
}

#[tauri::command]
async fn search_documents(
    state: State<'_, AppState>,
    query: String,
    limit: usize,
    _offset: usize,
    filter: Option<String>,
) -> Result<Vec<(String, String, f32)>, String> {
    let model_guard = state.model.lock().await;
    let query_vec = if let Some(model) = model_guard.as_ref() {
        model.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 768])
    } else { vec![0.0; 768] };

    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        store.search_items("items", &query, query_vec, limit, filter).await.map_err(|e| e.to_string())
    } else { Err("DB not initialized".to_string()) }
}

#[tauri::command]
async fn upsert_items(state: State<'_, AppState>, items: Vec<Value>) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        let mut count = 0;
        for item in items {
            let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let type_str = item.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
            if !id.is_empty() {
                let _ = db.upsert_item("items", &id, &type_str, item, None, None, None, None, None, None, None).await;
                count += 1;
            }
        }
        Ok(format!("Synced {} items", count))
    } else { Err("DB not initialized".to_string()) }
}

#[tauri::command]
async fn check_active_task(state: State<'_, AppState>, payload: Value) -> Result<bool, String> {
    let cc = payload.get("cc").and_then(|v| v.as_str()).unwrap_or("");
    let r#ref = payload.get("ref").and_then(|v| v.as_str()).unwrap_or("");
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        db.has_active_task(cc, r#ref).await.map_err(|e| e.to_string())
    } else { Ok(false) }
}

#[tauri::command]
async fn get_known_pages(state: State<'_, AppState>) -> Result<Vec<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        store.get_all_items("pages", 100, 0, None).await.map_err(|e| e.to_string())
    } else { Ok(vec![]) }
}

#[tauri::command]
async fn get_known_users(state: State<'_, AppState>) -> Result<Vec<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        store.get_all_items("users", 20, 0, None).await.map_err(|e| e.to_string())
    } else { Ok(vec![]) }
}

#[tauri::command]
async fn initialize_hub(state: State<'_, AppState>, address: String, email: String, flag: String) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        store.initialize_user_profiles(&address, &email, &flag).await.map_err(|e| e.to_string())?;
        Ok("Hub initialized".to_string())
    } else { Err("Store not initialized".to_string()) }
}

#[tauri::command]
async fn get_browser_status() -> Result<String, String> {
    let guard = automation::GLOBAL_BROWSER.lock().await;
    if guard.is_some() || automation::is_browser_reachable().await { Ok("running".to_string()) } else { Ok("stopped".to_string()) }
}

#[tauri::command]
async fn save_mobile_temp_file(app_handle: tauri::AppHandle, filename: String, data: Vec<u8>) -> Result<String, String> {
    let temp_dir = app_handle.path().app_cache_dir().map_err(|e| e.to_string())?.join("mobile_uploads");
    std::fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;
    let file_path = temp_dir.join(filename);
    std::fs::write(&file_path, data).map_err(|e| e.to_string())?;
    Ok(file_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn check_query_intent(_state: State<'_, AppState>, _query: String) -> Result<String, String> {
    Ok("SEARCH".to_string())
}

#[tauri::command]
async fn ai_search_complex(state: State<'_, AppState>, query: String, language: String, device_preference: Option<String>) -> Result<Value, String> {
    let mut model_guard = state.model.lock().await;
    if model_guard.is_none() { *model_guard = Some(LogisModel::new(device_preference.as_deref()).await.map_err(|e| e.to_string())?); }
    let model = model_guard.as_ref().unwrap();
    let structured = model.parse_query_structured(query.clone(), &language).await.map_err(|e| e.to_string())?;
    Ok(json!({ "structured": structured, "results": [] }))
}

#[tauri::command]
async fn deep_research_command(state: State<'_, AppState>, app_handle: tauri::AppHandle, query: String, _doc_id: Option<String>, device_preference: Option<String>) -> Result<String, String> {
    let mut model_guard = state.model.lock().await;
    if model_guard.is_none() { *model_guard = Some(LogisModel::new(device_preference.as_deref()).await.map_err(|e| e.to_string())?); }
    let model = model_guard.as_ref().unwrap();
    model.run_deep_research(query, "".to_string(), &app_handle, Some(state.cancellation_token.clone())).await.map_err(|e| e.to_string())
}

pub fn run() {
    let model = Arc::new(Mutex::new(None));
    let store = Arc::new(Mutex::new(None));
    let cancellation_token = Arc::new(AtomicBool::new(false));

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .manage(AppState {
            store: store.clone(),
            model: model.clone(),
            cancellation_token: cancellation_token.clone(),
        })
        .setup(|app| {
            let setup_store = app.state::<AppState>().store.clone();
            tauri::async_runtime::spawn(async move {
                let mut store_guard = setup_store.lock().await;
                let db_path = "data/lancedb";
                if let Ok(s) = VectorStore::new(db_path).await {
                    let _ = s.init_task_table().await;
                    let _ = s.init_all_tables().await;
                    let _ = s.cleanup_zombie_tasks().await;
                    *store_guard = Some(s);
                }
            });

            let sch_store = app.state::<AppState>().store.clone();
            let sch_model = app.state::<AppState>().model.clone();
            let sch_cancel = app.state::<AppState>().cancellation_token.clone();
            let sch_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                scheduler::start_background_worker(sch_store, sch_model, sch_cancel, sch_handle).await;
            });

            let auto_reconnect_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let _ = automation::try_reconnect_existing_browser(auto_reconnect_handle).await;
            });

            let event_store = app.state::<AppState>().store.clone();
            let event_cancel = app.state::<AppState>().cancellation_token.clone();
            app.listen("new-task-from-browser", move |event| {
                event_cancel.store(false, Ordering::SeqCst);
                crate::utils::set_extraction_stop_signal(false);
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
                                from: from_addr, to: team_id, cc: payload_val.get("cc").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                                bcc: payload_val.get("bcc").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                                r#ref: payload_val.get("ref").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                                data_json: payload_val.to_string(), created_at: now, updated_at: now, status: 10,
                            };
                            let _ = db.add_task(task).await;
                            crate::scheduler::notify_new_task();
                        }
                    });
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            stop_current_extraction, delete_message, unload_model, resize_window,
            start_drag, move_to_top_center, set_ignore_cursor_events, launch_best_browser,
            check_available_browsers, proxy_fetch, set_login_state, get_active_tasks,
            get_task_logs, mark_ui_ready, check_gpu_availability, get_all_documents,
            get_chat_messages, summarize_image, extract_html_from_current_tab,
            get_document, delete_document, delete_documents, search_documents,
            upsert_items, check_active_task, get_known_pages, get_known_users,
            initialize_hub, get_browser_status, save_mobile_temp_file,
            check_query_intent, ai_search_complex, deep_research_command
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
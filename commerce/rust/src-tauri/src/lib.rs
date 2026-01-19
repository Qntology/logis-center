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
use tokio::sync::Mutex;
use model::LogisModel;
use store::{VectorStore, TradeDocument};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use serde_json::{Value, json};

pub struct AppState {
    pub model: Arc<Mutex<Option<LogisModel>>>,
    pub store: Arc<Mutex<Option<VectorStore>>>,
    pub cancellation_token: Arc<AtomicBool>,
}

#[tauri::command]
async fn stop_current_extraction(state: State<'_, AppState>) -> Result<String, String> {
    state.cancellation_token.store(true, Ordering::SeqCst);
    
    // Force drop model to release VRAM/RAM
    let mut model_guard = state.model.lock().await;
    *model_guard = None;
    println!("[STOP] Model dropped and resources released.");

    Ok("Stop signal sent and resources released.".to_string())
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
            status: "pending".to_string(),
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
) -> Result<Vec<(String, String, f32)>, String> {
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
        store.search_items("commerce_items", &query, query_vec, 10).await.map_err(|e| e.to_string())
    } else {
        Err("DB not initialized".to_string())
    }
}

// Placeholder for missing CRUD - to be implemented in store.rs if needed
#[tauri::command]
async fn get_all_documents(
    _state: State<'_, AppState>,
    _limit: usize,
    _offset: usize,
) -> Result<Vec<TradeDocument>, String> {
    // Not implemented in VectorStore yet
    Ok(vec![])
}

#[tauri::command]
async fn get_document(
    _state: State<'_, AppState>,
    _uuid: String,
) -> Result<Option<TradeDocument>, String> {
    Ok(None)
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
            
            let emb = model.get_embedding(text.to_string()).await.unwrap_or(vec![0.0; 768]);
            if let Ok(results) = store.search_items("commerce_items", text, emb, 5).await {
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
    state: State<'_, AppState>,
    query: String,
) -> Result<String, String> {
    let mut model_guard = state.model.lock().await;
    if model_guard.is_none() {
        if let Ok(m) = LogisModel::new(None).await {
            *model_guard = Some(m);
        } else {
            return Err("Failed to load model".to_string());
        }
    }
    
    if let Some(model) = model_guard.as_ref() {
        model.parse_query_intent(query, Some(state.cancellation_token.clone())).await.map_err(|e| e.to_string())
    } else {
        Err("Model not initialized".to_string())
    }
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
        if let Ok(results) = store.search_items("commerce_items", emb, 3).await {
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]

pub fn run() {

    let model = Arc::new(Mutex::new(None));

    let store = Arc::new(Mutex::new(None));
    let cancellation_token = Arc::new(AtomicBool::new(false));



        let _model_clone = model.clone();



        let _store_clone = store.clone();



    

    let _store_server = store.clone();


        tauri::Builder::default()



    

        .plugin(tauri_plugin_fs::init())

        .plugin(tauri_plugin_shell::init())

                .manage(AppState {

                    model: model,

                    store: store,
                    cancellation_token: cancellation_token.clone(),

                })

                .setup(|app| {
                    let store = app.state::<AppState>().store.clone();
                    let _app_handle = app.handle().clone();

                    // Initialize Store on Startup
                    tauri::async_runtime::spawn(async move {
                        let mut store_guard = store.lock().await;
                        let db_path = "data/lancedb";
                        let _ = std::fs::create_dir_all(db_path);
                        
                        match VectorStore::new(db_path).await {
                            Ok(s) => {
                                println!("[Setup] VectorStore initialized.");
                                // Ensure tables exist
                                if let Err(e) = s.init_task_table().await {
                                    eprintln!("[Setup] Failed to init task table: {}", e);
                                }
                                if let Err(e) = s.init_all_tables().await {
                                    eprintln!("[Setup] Failed to init commerce tables: {}", e);
                                }
                                *store_guard = Some(s);
                            },
                            Err(e) => eprintln!("[Setup] Failed to init Vector Store: {}", e), 
                        }
                    });
                    
                    // Start Background Scheduler
                    let scheduler_store = app.state::<AppState>().store.clone();
                    let scheduler_model = app.state::<AppState>().model.clone();
                    let scheduler_cancel = app.state::<AppState>().cancellation_token.clone();
                    let scheduler_handle = app.handle().clone();

                    tauri::async_runtime::spawn(async move {
                        scheduler::start_background_worker(scheduler_store, scheduler_model, scheduler_cancel, scheduler_handle).await;
                    });
                    
                    let store_for_event = app.state::<AppState>().store.clone();

                    // Listen for events from the injected browser script
                    app.listen("new-task-from-browser", move |event| {
                        println!("[Event] Received 'new-task-from-browser'");
                        
                        if let Ok(payload_val) = serde_json::from_str::<serde_json::Value>(event.payload()) {
                            let store_clone = store_for_event.clone();
                            
                            tauri::async_runtime::spawn(async move {
                                let store_guard = store_clone.lock().await;
                                if let Some(db) = store_guard.as_ref() {
                                    let now = chrono::Utc::now().timestamp_millis();
                                    
                                    // Map JSON payload to Task struct
                                    let task = crate::store::Task {
                                        id: payload_val.get("id")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.to_string())
                                            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                                        r#type: payload_val.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
                                        from_source: "injected_script".to_string(),
                                        to_dest: "local".to_string(),
                                        cc: payload_val.get("cc").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                                        bcc: "".to_string(),
                                        ref_id: payload_val.get("ref_id").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                                        data_json: payload_val.to_string(),
                                        created_at: now,
                                        updated_at: now,
                                        status: "pending".to_string(),
                                    };
        
                                    match db.add_task(task).await {
                                        Ok(_) => println!("[Event] Task saved to DB successfully."),
                                        Err(e) => eprintln!("[Event] Failed to save task: {}", e),
                                    }
                                } else {
                                    eprintln!("[Event] Database not initialized.");
                                }
                            });
                        } else {
                            eprintln!("[Event] Failed to parse payload: {}", event.payload());
                        }
                    });

                    Ok(())
                })

                .plugin(tauri_plugin_opener::init())

        

        .plugin(tauri_plugin_dialog::init())

        .invoke_handler(tauri::generate_handler![

            summarize_image, 

            search_documents, 

            get_all_documents,

            get_document,

            update_document,

            delete_document,

            delete_documents,

            check_query_intent,

            deep_research_command,

            ai_search_complex,

            launch_browser,

            launch_best_browser,

            extract_html_from_current_tab,
            
            stop_current_extraction,

            check_available_browsers,

            resize_window,

            start_drag,

            move_to_top_center,

            set_login_state

        ])

        .run(tauri::generate_context!())

        .expect("error while running tauri application");

}

mod model;
mod store;
mod automation;
pub use utils::parsing;
pub use utils::bias_schema;
pub use utils::json_parse;
pub use utils::nl_convert;
pub use utils::time_guide;
mod logic;
mod scheduler;
pub mod analytic;
pub mod stanza;
pub mod js_templates;
pub mod prompts; // 🌟 새로 추가된 프롬프트 모듈 선언
pub mod parsers; // 🌟 문서 파서 모듈 선언
pub mod models;
pub mod utils;
pub mod position_embed;
pub mod openai_types;
pub mod chat_template;
pub mod tokenizer;

use tauri::{State, Manager, Listener, Emitter}; 
use tokio::sync::Mutex as TokioMutex;
use std::sync::RwLock; 
use once_cell::sync::Lazy; 
use model::LogisModel;
use store::{VectorStore, TradeDocument};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use serde_json::{Value, json};


static IS_SEARCHING: AtomicBool = AtomicBool::new(false);
// 브라우저가 실행되는 도중 상태를 방어하기 위한 락 추가
pub static IS_BROWSER_LAUNCHING: AtomicBool = AtomicBool::new(false);


pub static CURRENT_BROWSER_STATE: Lazy<RwLock<String>> = Lazy::new(|| RwLock::new("stopped".to_string()));
pub static LAST_BROWSER_STATE_CHANGE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

pub static ACTIVE_TASK_MEM: Lazy<RwLock<Option<Value>>> = Lazy::new(|| RwLock::new(None));
pub static CURRENT_UI_CATEGORY: Lazy<RwLock<String>> = Lazy::new(|| RwLock::new(String::from("Processing")));

pub static LATEST_PROGRESS_PAYLOAD: Lazy<RwLock<Option<Value>>> = Lazy::new(|| RwLock::new(None));

pub struct AppState {
    pub model: Arc<TokioMutex<Option<LogisModel>>>,
    pub store: Arc<TokioMutex<Option<VectorStore>>>,
    pub cancellation_token: Arc<AtomicBool>,
}

#[tauri::command]
async fn get_active_task_context() -> Result<Value, String> {
    let mut result = json!(null);
    
    if let Some(mem_task) = crate::ACTIVE_TASK_MEM.read().unwrap().clone() {
        if mem_task.get("id").is_some() { result = mem_task; }
    }

    if !result.is_null() {
        if let Ok(mem) = crate::LATEST_PROGRESS_PAYLOAD.read() {
            if let Some(latest) = mem.as_ref() {
                if result.get("id") == latest.get("task_id") {
                    if let Some(summary) = latest.get("summary") {
                        result.as_object_mut().unwrap().insert("summary".to_string(), summary.clone());
                    }
                    
                    result.as_object_mut().unwrap().insert("latest_payload".to_string(), latest.clone());
                }
            }
        }
        return Ok(result);
    }
    
    Ok(json!(null))
}

#[tauri::command]
async fn stop_current_extraction(
    state: State<'_, AppState>,
    task_id: Option<String>
) -> Result<String, String> {
    // 1. Set global stop signals (Atomic + File-based for persistence across threads)
    state.cancellation_token.store(true, Ordering::SeqCst);
    crate::utils::set_extraction_stop_signal(true);
    
    // 2. Clear from DB
    
    // lock().await를 사용하여 스케줄러의 DB 작업이 끝날 때까지 찰나를 기다린 후 100% 확실하게 지워버립니다.
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        if let Some(ref id) = task_id {
            let _ = db.update_task_status(id, crate::logic::parse_status("cancel")).await;
            let _ = db.delete_message_by_task_id(id).await;
            
            // [CLEANUP] 작업 취소 시 해당 세션의 무거운 KV 캐시와 임시 데이터 폴더를 즉각 삭제하여 디스크 용량을 확보합니다.
            let kv_dir = crate::utils::paths::get_kv_dir(None).join(id);
            let base_kv_dir = crate::utils::paths::get_kv_dir(None).join(format!("{}_base", id));
            let task_data_dir = crate::utils::paths::get_task_specific_dir(None, id);
            let pug_log_dir = crate::utils::paths::get_pug_logs_dir(None, id);

            let _ = std::fs::remove_dir_all(&kv_dir);
            let _ = std::fs::remove_dir_all(&base_kv_dir);
            let _ = std::fs::remove_dir_all(&task_data_dir);
            let _ = std::fs::remove_dir_all(&pug_log_dir);

            println!("[STOP] Task {} cleared from DB and temporary files deleted.", id);
        } else {
            
            let _ = db.cleanup_unfinished_tasks_on_startup().await;
            println!("[STOP] All pending tasks cleared from DB.");
        }
    }
    drop(store_guard); // 다음 단계(Model Clear) 진행을 위해 즉시 락 해제

    // 3. Try to clear model
    if let Ok(mut model_guard) = state.model.try_lock() {
        if let Some(m) = model_guard.as_ref() {
            m.deep_purge_resources().await; // 모델 메모리 및 VRAM 캐시도 확실하게 강제 파기
        }
        *model_guard = None;
    }

    
    // 백엔드의 현재 작업 캐시를 즉시 비워야 프론트엔드의 #btn-extract 버튼이 정상적으로 부활합니다.
    if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() {
        *w = None;
    }

    Ok("Stop signal sent and resources cleaned.".to_string())
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
    
    if IS_SEARCHING.load(Ordering::SeqCst) {
        println!("[UNLOAD] AI Search is active. Skipping unload to prevent deadlock.");
        return Ok("Search active. Memory kept.".to_string());
    }

    {
        let mut model_guard = state.model.lock().await;
        if let Some(m) = model_guard.as_ref() {
            m.deep_purge_resources().await;
        }
        *model_guard = None;
    }
    
    {
        let mut store_guard = state.store.lock().await;
        *store_guard = None;
    }

    state.cancellation_token.store(false, Ordering::SeqCst);

    println!("[UNLOAD] Model, Store and Cancellation flag cleared.");
    Ok("Memory cleared.".to_string())
}

// =====================================================================
// 🌟 [CLIENT-SIDE EMBEDDING] Cloud AI 모드에서도 임베딩은 무조건 로컬에서 수행합니다.
//    ① get_query_embedding      : 클라우드 검색용 질의 벡터를 로컬 모델로 생성
//    ② reindex_pending_embeddings : 클라우드가 내려준 아이템을 로컬에서 벡터화 + 청크 인덱싱
// =====================================================================
#[tauri::command]
async fn get_query_embedding(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
    text: String,
    device_preference: Option<String>,
) -> Result<Vec<f32>, String> {
    let model = {
        let mut model_guard = state.model.lock().await;

        if let Some(m) = model_guard.as_ref() {
            let wants_cpu = device_preference.as_deref() == Some("cpu");
            if m.is_cpu_mode != wants_cpu {
                m.deep_purge_resources().await;
                *model_guard = None;
            }
        }

        if model_guard.is_none() {
            match LogisModel::new(app_handle.clone(), device_preference.as_deref()).await {
                Ok(m) => { *model_guard = Some(m); },
                Err(e) => return Err(format!("Model load failed: {}", e)),
            }
        }

        model_guard.as_ref().unwrap().clone()
    };

    model.check_embedding_downloaded().await.map_err(|e| e.to_string())?;
    model.ensure_embedding().await.map_err(|e| e.to_string())?;

    let vec = model.get_embedding(text).await.map_err(|e| e.to_string())?;

    println!("[EMBED-LOCAL] Query embedding generated locally. dim = {}", vec.len());

    Ok(vec)
}

#[tauri::command]
async fn reindex_pending_embeddings(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
    limit: Option<usize>,
    device_preference: Option<String>,
    // 🌟 [MODE ROUTING] commerce / shipping / analytic 트랙을 구분합니다.
    //    analytic 은 console.logis.center, 나머지는 commerce.logis.center 로 벡터가 전송됩니다.
    mode: Option<String>,
) -> Result<Value, String> {
    // 진행 중인 로컬 작업이 있으면 VRAM 충돌 방지를 위해 즉시 반환합니다.
    if IS_SEARCHING.load(Ordering::SeqCst) {
        return Ok(json!({ "processed": 0, "vectors": [], "skipped": "searching" }));
    }
    if crate::ACTIVE_TASK_MEM.read().unwrap().is_some() {
        return Ok(json!({ "processed": 0, "vectors": [], "skipped": "busy" }));
    }

    let store_opt = {
        let mut store_guard = state.store.lock().await;
        if store_guard.is_none() {
            let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
            let _ = std::fs::create_dir_all(&db_path);
            if let Ok(s) = VectorStore::new(&db_path).await {
                let _ = s.init_all_tables().await;
                *store_guard = Some(s);
            }
        }
        store_guard.as_ref().cloned()
    };

    let store = match store_opt {
        Some(s) => s,
        None => return Err("DB not initialized".to_string()),
    };

    let model = {
        let mut model_guard = state.model.lock().await;

        if let Some(m) = model_guard.as_ref() {
            let wants_cpu = device_preference.as_deref() == Some("cpu");
            if m.is_cpu_mode != wants_cpu {
                m.deep_purge_resources().await;
                *model_guard = None;
            }
        }

        if model_guard.is_none() {
            match LogisModel::new(app_handle.clone(), device_preference.as_deref()).await {
                Ok(m) => { *model_guard = Some(m); },
                Err(e) => return Err(format!("Model load failed: {}", e)),
            }
        }

        model_guard.as_ref().unwrap().clone()
    };

    model.check_embedding_downloaded().await.map_err(|e| e.to_string())?;
    model.ensure_embedding().await.map_err(|e| e.to_string())?;

    let scan_limit = limit.unwrap_or(20);

    // 🌟 [MODE ROUTING] 트랙별로 격리된 문서만 스캔합니다.
    //    (analytic 트랙 문서는 syncAnalyticsData 가 mode='analytic' 으로 태깅해 저장합니다)
    let target_mode = mode.unwrap_or_else(|| "commerce".to_string());
    let mode_filter = format!("mode = '{}'", target_mode);

    let docs = store.get_all_items("items", 500, 0, Some(mode_filter)).await.map_err(|e| e.to_string())?;

    let mut vectors: Vec<Value> = Vec::new();
    let mut processed = 0usize;

    for doc in docs {
        if processed >= scan_limit { break; }
        if state.cancellation_token.load(Ordering::Relaxed) { break; }

        if doc.id.is_empty() { continue; }
        if doc.r#type == "pages" || doc.r#type == "talk" || doc.r#type == "prompt" || doc.r#type == "ai_search" {
            continue;
        }

        // 청크가 하나도 없다 = 로컬 임베딩이 아직 수행되지 않은 클라우드 동기화 아이템입니다.
        let chunk_count = store.count_chunks_by_item(&doc.id).await.unwrap_or(0);
        if chunk_count > 0 { continue; }

        let mut data: Value = serde_json::from_str(&doc.json_data).unwrap_or(json!({}));

        // 🌟 [ANALYTICS] Cron Worker 가 구조화한 'action'(사용자 의도 문장)이 벡터의 본체입니다.
        //    action 이 없으면 summary → cross_action_flow 순으로 폴백합니다.
        let analytic_text = data.get("action").and_then(|v| v.as_str()).unwrap_or("")
            .trim().to_string();
        let analytic_fallback = data.get("summary").and_then(|v| v.as_str())
            .or_else(|| data.get("cross_action_flow").and_then(|v| v.as_str()))
            .unwrap_or("").trim().to_string();

        let text = if !analytic_text.is_empty() {
            analytic_text
        } else if !doc.text.trim().is_empty() {
            doc.text.clone()
        } else if !analytic_fallback.is_empty() {
            analytic_fallback
        } else {
            crate::parsing::json_to_natural_language(&data)
        };

        if text.trim().is_empty() { continue; }

        let emb = model.get_embedding(text.clone()).await.map_err(|e| e.to_string())?;

        if let Some(o) = data.as_object_mut() {
            o.insert("text".to_string(), json!(text.clone()));
            if !o.contains_key("masked_text") {
                o.insert("masked_text".to_string(), json!(text.clone()));
            }
            o.insert("embed".to_string(), json!(1));
        }

        let target_table = match doc.r#type.as_str() {
            "sales" | "goods" | "order" => "sales",
            "tracking" | "receiving" | "shipping" => "tracking",
            "event" | "coupon" => "event",
            "member" | "team" | "user" => "users",
            // 🌟 [ANALYTICS] 사용자 행동 로그 / 리포트는 items 미러 테이블에만 존재합니다.
            "click" | "hover" | "change" | "report" | "question" | "answer" => "items",
            _ => "items",
        };

        let digest = crate::utils::hash::digest(&text);

        let _ = store.upsert_item(
            target_table, &doc.id, &doc.r#type, data.clone(), Some(emb.clone()),
            Some(&doc.from), Some(&doc.to), Some(&doc.cc), Some(&doc.bcc), Some(&doc.r#ref), Some(&digest)
        ).await;

        let _ = store.upsert_item(
            "items", &doc.id, &doc.r#type, data.clone(), Some(emb.clone()),
            Some(&doc.from), Some(&doc.to), Some(&doc.cc), Some(&doc.bcc), Some(&doc.r#ref), Some(&digest)
        ).await;

        let link = data.get("link").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let doc_lang = crate::utils::lang_utils::detect_document_language(&text);
        let mode = if doc.mode.trim().is_empty() { "commerce".to_string() } else { doc.mode.clone() };

        let cancel = state.cancellation_token.clone();
        let _ = crate::scheduler::index_item_chunks(
            &store, &model, &doc.id, &doc.r#type, &doc_lang, &data, true,
            &doc.cc, &doc.bcc, &doc.r#ref, &mode, &link, &cancel, &app_handle, "cloud_sync"
        ).await;

        vectors.push(json!({
            "id": doc.id,
            "table": target_table,
            "type": doc.r#type,
            "mode": target_mode,
            "no": data.get("no").and_then(|v| v.as_str()).unwrap_or(""),
            "from": doc.from,
            "to": doc.to,
            "cc": doc.cc,
            "bcc": doc.bcc,
            "ref": doc.r#ref,
            // 🌟 [ANALYTICS METADATA] console.logis.center Vectorize 메타데이터로 그대로 넘어갑니다.
            "summary": data.get("summary").and_then(|v| v.as_str()).unwrap_or(""),
            "action": data.get("action").and_then(|v| v.as_str()).unwrap_or(""),
            "relate": data.get("relate").cloned().unwrap_or(json!([])),
            "values": emb
        }));

        processed += 1;
    }

    if processed > 0 {
        println!("[EMBED-LOCAL] Cloud-synced items embedded locally: {} item(s). (mode: {})", processed, target_mode);
    }

    model.unload_embedding().await;

    Ok(json!({ "processed": processed, "vectors": vectors, "mode": target_mode }))
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
async fn set_ignore_cursor_events(app_handle: tauri::AppHandle, ignore: bool) {
    if let Some(window) = app_handle.get_webview_window("main") {
        let _ = window.set_ignore_cursor_events(ignore);
    }
}

#[tauri::command]
async fn launch_browser(
    state: State<'_, AppState>, 
    app_handle: tauri::AppHandle,
    browser: String,
    url: String,
    script: String,
) -> Result<String, String> {
    
    state.cancellation_token.store(false, Ordering::SeqCst);
    crate::utils::set_extraction_stop_signal(false);

    // 함수 진입 시점에 즉시 락을 걸어 get_browser_status가 stopped를 반환하지 못하게 함
    crate::IS_BROWSER_LAUNCHING.store(true, Ordering::SeqCst);
    {
        let mut current_state = crate::CURRENT_BROWSER_STATE.write().unwrap();
        *current_state = "running".to_string();
        crate::LAST_BROWSER_STATE_CHANGE.store(chrono::Utc::now().timestamp_millis(), Ordering::SeqCst);
    }
    
    let result = automation::run_browser_automation(browser, url, script, app_handle).await;

    // 🌟 [CRITICAL FIX] 500ms 단위의 느린 대기를 50ms 단위의 초고속 폴링으로 변경하여 브라우저가 준비되는 즉시 락을 해제합니다.
    for _ in 0..100 {
        if automation::is_browser_reachable().await {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    // 충분한 안정화 시간을 거친 후 런칭 플래그 해제
    crate::IS_BROWSER_LAUNCHING.store(false, Ordering::SeqCst);

    result.map_err(|e| e.to_string())
}

#[tauri::command]
async fn launch_best_browser(
    state: State<'_, AppState>, 
    app_handle: tauri::AppHandle,
    url: String,
) -> Result<String, String> {
    
    state.cancellation_token.store(false, Ordering::SeqCst);
    crate::utils::set_extraction_stop_signal(false);

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
    
    crate::IS_BROWSER_LAUNCHING.store(true, Ordering::SeqCst);
    
    let result = automation::run_browser_automation(target.to_string(), url, "".to_string(), app_handle).await;

    
    // 🌟 [CRITICAL FIX] 500ms 단위의 느린 대기를 50ms 단위의 초고속 폴링으로 변경하여 브라우저가 준비되는 즉시 락을 해제합니다.
    for _ in 0..100 {
        if automation::is_browser_reachable().await {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    crate::IS_BROWSER_LAUNCHING.store(false, Ordering::SeqCst);

    result.map_err(|e| e.to_string())
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
            from: "manual_upload".to_string(),
            to: "local".to_string(),
            cc: "".to_string(),
            bcc: "".to_string(),
            r#ref: "manual".to_string(),
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
    filter: Option<String>,
) -> Result<Vec<(String, String, f32)>, String> {
    
    println!("[DB-SEARCH] 텍스트 검색 요청 수신 (Query: '{}', Filter: {:?})", query, filter);

    let store_opt = {
        let mut store_guard = state.store.lock().await;
        if store_guard.is_none() {
            let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
            if let Ok(s) = VectorStore::new(&db_path).await {
                *store_guard = Some(s);
            }
        }
        store_guard.as_ref().cloned()
    }; 
    
    
    let query_vec = if !query.trim().is_empty() {
        
        let is_task_active = crate::ACTIVE_TASK_MEM.read().unwrap().is_some();
        if is_task_active {
            println!("[DB-SEARCH] Background task is active. Skipping embedding model load to prevent VRAM overflow.");
            vec![0.0; 384]
        } else {
            let model_opt = { state.model.lock().await.as_ref().cloned() }; 
            if let Some(model) = model_opt {
                model.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 384])
            } else {
                vec![0.0; 384]
            }
        }
    } else {
        vec![0.0; 384]
    };

    if let Some(store) = store_opt {
        
        let search_result = store.search_items("items", &query, query_vec, limit, offset, filter, false).await.map_err(|e| e.to_string());
        
        
        match &search_result {
            Ok(items) => {
                println!("[DB-SEARCH] 검색 완료: {}건 반환", items.len());
                for (i, (id, text, score)) in items.iter().enumerate() {
                    // 터미널 가독성을 위해 줄바꿈 문자를 공백으로 치환하여 한 줄로 출력합니다.
                    println!("  ↳ {}. ID: [{}] | Score: {:.4} | Text: {}", i + 1, id, score, text.replace("\n", " "));
                }
            },
            Err(e) => println!("[DB-SEARCH] 검색 실패: {}", e),
        }
        
        search_result
    } else {
        Err("DB not initialized".to_string())
    }
}

// Helper to convert structured LLM conditions to SQL filter strings
fn convert_conditions_to_sql(ctx: &Value) -> Option<String> {
    let mut filters = Vec::new();
    
    if let Some(t) = ctx.get("type").and_then(|v| v.as_str()) {
        if !t.is_empty() { filters.push(format!("type = '{}'", t)); }
    }

    if let Some(status) = ctx.get("status").and_then(|v| v.as_str()) {
        if !status.is_empty() && status != "null" {
            
            let status_int = crate::logic::parse_status(status);
            filters.push(format!("status = {}", status_int));
        }
    }

    if let Some(cond) = ctx.get("condition").and_then(|v| v.as_object()) {
        for (key, val_obj) in cond {

            // let valid_cols = [
            //     "amount", "status", "type", "created_at", "updated_at",
            //     "no", "carrier", "shipping_method", "sender_address", "recipient_address", 
            //     "shipping_date", "delivery_date", "weight",
            //     "vessel", "pol", "pod", "incoterms", "sender_name", "recipient_name", "issue_date",
            //     "started_at", "expired_at" // 🌟 [CRITICAL FIX] 이벤트/쿠폰 필터링용 날짜 컬럼 복구
            // ];
            
            // let mapped_key = match key.as_str() {
            //     "price" | "sale_price" | "discount" | "supply_price" | "order" | "goods" => "amount",
            //     "document_number" | "tracking_number" => "no",
            //     "supplier_name" | "shipper_name" => "sender_name",
            //     "buyer_name" | "consignee_name" => "recipient_name",
            //     "amount_total" | "total_amount" => "amount",
            //     "vehicle_name" | "flight_no" => "vessel",
            //     "location_port_of_loading" => "pol",
            //     "location_port_of_discharge" => "pod",
            //     "incoterms_code" => "incoterms",
            //     // 🌟 [CRITICAL FIX] 불필요한 가상 키워드를 완전히 제거하고 started_at과 expired_at으로 단일 통일시켰습니다.
            //     // coupon/event 도메인은 DB 고유의 started_at, expired_at 컬럼을 원본 보호하고 그 외 도메인은 전부 created_at 컬럼으로 연결합니다.
            //     "started_at" | "expired_at" => {
            //         let t = ctx.get("type").and_then(|v| v.as_str()).unwrap_or("");
            //         if t == "event" || t == "coupon" {
            //             if key == "started_at" { "started_at" } else { "expired_at" }
            //         } else {
            //             "created_at"
            //         }
            //     },
            //     k if valid_cols.contains(&k) => k,
            //     _ => "" 
            // };
            
            // 🌟 [CRITICAL FIX] LanceDB 스키마(store.rs)에 실제로 물리적으로 존재하는 컬럼만 명시해야 SQL 에러(Fallback)를 방지할 수 있습니다!
            // 가상 컬럼(weight, no, started_at 등)이 SQL에 포함되면 DataFusion 쿼리가 실패하여 모든 필터(5000원 등)가 통째로 초기화되는 치명적 버그 수정.
            // 🌟 [TRACKING FIX] tracking_number는 text 컬럼에 내장되어 있으므로 FTS 검색으로 처리합니다. SQL 필터에서는 무시합니다.
            let valid_cols = [
                "amount", "status", "type", "created_at", "updated_at", "mode", "is_masked"
            ];
            let mapped_key = match key.as_str() {
                "price" | "sale_price" | "discount" | "supply_price" | "order" | "goods" | "amount_total" | "total_amount" | "shipping_fee" => "amount",
                "started_at" | "shipping_date" | "issue_date" | "order_date" | "registration_date" | "release_date" | "manufacture_date" | "payment_date" => "created_at",
                "expired_at" | "delivery_date" => "updated_at",
                // 🌟 [TRACKING] tracking_number, no, code는 텍스트 기반 FTS 검색으로 처리하므로 SQL 필터에서는 스킵
                "tracking_number" | "no" | "code" | "carrier" => "",
                k if valid_cols.contains(&k) => k,
                _ => "" 
            };

            if mapped_key.is_empty() { continue; } // 유효하지 않은 컬럼은 무시하여 DB 크래시 방어

            if let Some(op_str) = val_obj.get("operator").and_then(|v| v.as_str()) {
                if let Some(val_val) = val_obj.get("value") {
                    // 🌟 [CRITICAL FIX] LLM이 "lt [Alts: lte, gte]" 처럼 쓰레기 값을 포함해서 주더라도 앞부분만 파싱하여 안전하게 추출 및 변환합니다.
                    let clean_op = op_str.trim().to_lowercase();
                    
                    // 🌟 [CRITICAL FIX] top, bottom, contains 연산자는 LanceDB 물리적 SQL 필터에서 지원하지 않으므로 무시하여 문법 에러(Crash)를 방지합니다. (UI에는 노출됨)
                    if clean_op == "top" || clean_op == "bottom" || clean_op == "contains" || clean_op == "not_contains" {
                        continue;
                    }

                    let operator = if clean_op.starts_with("gte") { ">=" }
                    else if clean_op.starts_with("gt") { ">" }
                    else if clean_op.starts_with("lte") { "<=" }
                    else if clean_op.starts_with("lt") { "<" }
                    else { "=" };
                    
                    let val_str = if val_val.is_number() {
                        val_val.to_string()
                    } else if let Some(s) = val_val.as_str() {
                        let numeric: String = s.chars().filter(|c| c.is_digit(10) || *c == '.').collect();
                        if numeric.is_empty() { continue; } else { numeric }
                    } else { continue; };

                    filters.push(format!("{} {} {}", mapped_key, operator, val_str));
                }
            }
        }
    }
    if filters.is_empty() { None } else { Some(filters.join(" AND ")) }
}

#[tauri::command]
async fn get_all_documents(
    state: State<'_, AppState>,
    limit: usize,
    offset: usize,
    filter: Option<String>,
) -> Result<Vec<TradeDocument>, String> {
    
    println!("[DB-FETCH] 리스트 불러오기 요청 수신 (Limit: {}, Filter: {:?})", limit, filter);

    let mut store_guard = state.store.lock().await; 
    
    
    // 프론트엔드가 데이터를 요청했는데 DB가 아직 없으면 즉시 여기서 로드합니다.
    if store_guard.is_none() {
        let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
        let _ = std::fs::create_dir_all(&db_path);
        if let Ok(s) = VectorStore::new(&db_path).await {
            let _ = s.init_all_tables().await;
            *store_guard = Some(s);
        } else {
            return Err("Failed to initialize LanceDB".to_string());
        }
    }

    if let Some(store) = store_guard.as_ref() {
        let mut results = store.get_all_items("items", limit, offset, filter).await.map_err(|e| e.to_string())?;
        
        // [DYNAMIC] Convert JSON to Natural Language for UI display only
        for doc in results.iter_mut() {
            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&doc.json_data) {
                doc.text = crate::parsing::json_to_natural_language(&json_val);
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
        let tables = vec!["items", "sales", "tracking", "event", "users", "pages"];
        
        // 1. Primary search: Exact ID match
        for table_name in tables.iter() {
            if let Ok(Some(mut doc)) = store.get_item_by_id(table_name, &uuid).await {
                if doc.text.is_empty() {
                    if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&doc.json_data) {
                        doc.text = parsing::json_to_natural_language(&json_val);
                    }
                }
                return Ok(Some(doc));
            }
        }

        // 2. Fallback search: If uuid is numeric or a hash, look inside the data JSON
        // This fixes cases where the top-level 'id' column was saved as an empty string.
        for table_name in tables {
            // Try matching against "id" field inside JSON
            if let Ok(Some((_found_id, json_val))) = store.find_item_by_property(table_name, "id", &json!(uuid)).await {
                if let Ok(doc) = store.get_item_by_id(table_name, "").await { // Get the row with empty ID
                    // Double check it's the right one by comparing json_data
                    if let Some(mut d) = doc {
                        if d.json_data == json_val.to_string() {
                            if d.text.is_empty() { d.text = parsing::json_to_natural_language(&json_val); }
                            return Ok(Some(d));
                        }
                    }
                }
            }
            
            // Try matching against "index" field inside JSON
            let index_query = uuid.parse::<i64>().map(|n| json!(n)).unwrap_or(json!(uuid));
            if let Ok(Some((_found_id, _json_val))) = store.find_item_by_property(table_name, "index", &index_query).await {
                // To be safe, we perform a broader search for any row where data contains the index
                // Since find_item_by_property already found it, we just need to reconstruct the TradeDocument
                if let Ok(all_docs) = store.get_all_items(table_name, 1000, 0, None).await {
                    for mut d in all_docs {
                        if d.json_data.contains(&uuid) {
                            if d.text.is_empty() {
                                if let Ok(jv) = serde_json::from_str::<Value>(&d.json_data) {
                                    d.text = parsing::json_to_natural_language(&jv);
                                }
                            }
                            return Ok(Some(d));
                        }
                    }
                }
            }
        }
        
        Ok(None)
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
    state: State<'_, AppState>,
    uuid: String,
) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        // [DETAIL] 'items' 테이블뿐만 아니라 다른 가능한 테이블에서도 삭제 시도
        let tables = vec!["items", "sales", "tracking", "event", "users", "pages"];
        for table in tables {
            let _ = store.delete_item(table, &uuid).await;
        }
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
        if uuids.is_empty() { return Ok("No documents to delete.".to_string()); }
        
        let tables = vec!["items", "sales", "tracking", "event", "users", "pages"];
        for table in tables {
            let _ = store.delete_items(table, uuids.clone()).await;
        }
        Ok(format!("Deleted {} documents.", uuids.len()))
    } else {
        Err("DB not initialized".to_string())
    }
}

#[tauri::command]
async fn ai_search_complex(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
    task_id: String,
    query: String,
    language: String,
    device_preference: Option<String>,
    search_mode: String,
    cc: String,
    bcc: String,
    ref_id: String,
) -> Result<Value, String> {
    
    // 백엔드의 무거운 LanceDB 조회 및 락 대기열 로직을 전면 철거했습니다! (속도 대폭 향상)
    
    let emit_term = |msg: &str| {
        println!("{}", msg);
        use tauri::Emitter;
        let _ = app_handle.emit("task-console-log", json!({"task_id": task_id, "text": format!("{}\n", msg)}));
    };

    emit_term("\n==================================================");
    emit_term("🚀 [AI-SEARCH] 프론트엔드 요청 수신 완료!");
    emit_term(&format!("   - Task ID: {}", task_id));
    emit_term(&format!("   - 검색어: {}", query));
    emit_term(&format!("   - 검색 모드: {}", search_mode));
    emit_term("==================================================\n");

    // 🌟 [CRITICAL FIX] 이전 작업 취소로 인해 굳어있던 취소 토큰(Cancellation Token)을 무조건 초기화하여
    // 스케줄러 큐를 타지 않고 직접 호출되는 검색 커맨드가 튕겨나가지 않도록 완벽히 방어합니다!
    state.cancellation_token.store(false, Ordering::SeqCst);
    crate::utils::set_extraction_stop_signal(false);

    let cancel_token = state.cancellation_token.clone();
    
    let store_opt = {
        let mut store_guard = state.store.lock().await;
        if store_guard.is_none() {
            let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
            if let Ok(s) = VectorStore::new(&db_path).await {
                let _ = s.init_all_tables().await;
                *store_guard = Some(s);
            }
        }
        store_guard.as_ref().cloned() 
    };

    
    if let Some(store) = store_opt.as_ref() {
        let now = chrono::Utc::now().timestamp_millis();
        
        // Task 객체는 1ms 뒤로 설정
        let task = crate::store::Task {
            id: task_id.clone(), 
            r#type: "ai_search".to_string(),
            from: "user".to_string(),
            to: "system".to_string(),
            cc: cc.clone(), bcc: bcc.clone(), r#ref: ref_id.clone(), 
            data_json: json!({"query": query.clone(), "mode": search_mode.clone()}).to_string(),
            created_at: now + 1, updated_at: now + 1,
            status: 10, 
        };
        let _ = store.add_task(task).await;
        
        let from_user = "user".to_string();
        let to_system = "system".to_string();

        // 사용자 질문 메시지: 기준 시간(now)에 저장
        let user_msg_id = format!("{}_query", task_id);
        let now_str = now.to_string();
        let _ = store.add_message(
            &user_msg_id, "user", &query, 
            Some(&task_id), Some(9), 
            Some(&cc), Some(&bcc), Some(&ref_id), 
            Some(&from_user), Some(&to_system), Some("talk"), 
            Some(&now_str)
        ).await;

        // 시스템 작업 메시지: 질문보다 확실히 나중에 보이도록 50ms 뒤에 저장 (프론트엔드 복구 로직과 정렬 대칭)
        let next_now_str = (now + 50).to_string();
        let _ = store.add_message(
            &task_id, "system_task", "Task Started: AI Search", 
            Some(&task_id), Some(10), 
            Some(&cc), Some(&bcc), Some(&ref_id), 
            Some(&to_system), Some(&from_user), Some("talk"), 
            Some(&next_now_str)
        ).await;
    }

    
    let payload_pending = json!({ 
        "task_id": task_id, 
        "category": "Pending", 
        "summary": "Waiting for AI Engine access...", 
        "spinner": "📥" 
    });
    let _ = app_handle.emit("extraction-progress", &payload_pending);
    
    emit_term("[QUEUE] Task queued. Waiting for Model Access...");
    
    let mut model_guard = state.model.lock().await;
    
    if cancel_token.load(Ordering::Relaxed) { 
        return Err("Task cancelled while waiting in queue".to_string()); 
    }

    emit_term("[QUEUE] AI Engine acquired. Starting process...");

    // [REMOVE] 백엔드 자체 검색 락 변수 조작 제거
    // 프론트엔드의 GlobalTaskManager가 이미 입구를 막고 있으므로 
    // 백엔드는 별도의 AtomicBool 락 없이 즉시 실행 로직에 집중합니다.

    {
        // 최소한의 동기화 정보만 업데이트 (UI 복구용)
        let mut mem_guard = crate::ACTIVE_TASK_MEM.write().unwrap();
        // let now = chrono::Utc::now().timestamp_millis();
        *mem_guard = Some(json!({
            "id": task_id.clone(),
            "status": 1
        }));
    }

    // 획득한 model_guard를 사용하여 모델 로드 또는 재사용
    let model = {
        if let Some(m) = model_guard.as_ref() {
            let wants_cpu = device_preference.as_deref() == Some("cpu");
            if m.is_cpu_mode != wants_cpu {
                m.deep_purge_resources().await;
                *model_guard = None;
            }
        }
        if model_guard.is_none() {
            if let Ok(m) = LogisModel::new(app_handle.clone(), device_preference.as_deref()).await { 
                *model_guard = Some(m);
            } else { 
                IS_SEARCHING.store(false, Ordering::SeqCst); 
                return Err("Failed to load model".to_string());
            }
        }
        model_guard.as_ref().unwrap().clone() 
    }; 

    // 🌟 [CRITICAL FIX] 백그라운드에서 Qwen3를 미리 로드하면 내부의 deep_purge_resources가
    // 메인 스레드가 사용 중인 임베딩 모델을 파괴하여 VRAM/메모리 충돌(panic 또는 0점 반환)을 유발합니다.
    // 따라서 병렬 꼼수를 제거하고, 필요한 시점에 순차적으로 모델을 로드하여 안정성을 100% 보장합니다.
    if !model.is_cpu_mode {
        emit_term("[VRAM-SETUP] GPU 모드 감지됨. 임베딩 모델 단독 로딩 개시...");
        let _ = model.ensure_embedding().await;
    }

    if let Some(store) = store_opt.as_ref() {
        let _ = store.update_task_status(&task_id, 1).await; // 1: Processing
        
        // 시스템 말풍선 텍스트만 깔끔하게 변경합니다.
        let _ = store.update_message_status(&task_id, 1, Some("Analyzing semantic intent...")).await;
    }

    // 화면의 찌꺼기를 날려버리는 트리거(Processing) 발송!
    let payload_start = json!({ 
        "task_id": task_id, 
        "category": "Processing", 
        "summary": "AI Engine ready. Starting search...", 
        "spinner": "⠋" 
    });
    let _ = app_handle.emit("extraction-progress", &payload_start);
    crate::utils::logger::log_task_progress(&app_handle, &task_id, &payload_start);

    
    let search_process = async {
        let mut all_results = Vec::new();
        
        let team_id = crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000"); 
        let mut metrics_json_str = "{}".to_string();
        
        if let Some(store) = store_opt.as_ref() {
            if let Ok(Some(doc)) = store.get_item_by_id("users", &team_id).await {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&doc.json_data) {
                    if let Some(base) = val.get("base") {
                        metrics_json_str = base.to_string();
                    }
                }
            }
        }

        
        let structured_query = match search_mode.as_str() {
            "shipping" => {
                model.parse_shipping_query(&task_id, &app_handle, query.clone(), &language, cancel_token.clone()).await.map_err(|e| e.to_string())?
            },
            "analytic" => {
                model.parse_analytic_query(&task_id, &app_handle, query.clone(), &language, cancel_token.clone()).await.map_err(|e| e.to_string())?
            },
            _ => { // default: commerce
                model.parse_commerce_query(&task_id, &app_handle, query.clone(), &language, &metrics_json_str, cancel_token.clone()).await.map_err(|e| e.to_string())?
            }
        };

        if let (Some(store), Some(ctx_arr)) = (store_opt.clone(), structured_query.get("context").and_then(|v| v.as_array())) {
            for ctx in ctx_arr {
                
                if cancel_token.load(Ordering::Relaxed) { 
                    return Err("Search cancelled by user".to_string()); 
                }

                tokio::task::yield_now().await;

                let text = ctx.get("text").and_then(|v| v.as_str()).unwrap_or("");
                if text.is_empty() { continue; }
                let raw_ctx_type = ctx.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
                // 🌟 [CRITICAL FIX] "ignore"로 분류된 명령어/분석 요청 청크는 DB 검색 단계에서 완전히 무시합니다!
                if raw_ctx_type == "ignore" {
                    continue;
                }

                // 🌟 [TABLE FALLBACK] STAGE-3 이 발행한 "{domain}_items" 폴백 컨텍스트는
                //    target_table match 에서 어떤 분기에도 걸리지 않아 items(전 타입 미러)로 조회됩니다.
                //    SQL 필터의 type 컬럼에는 원래 도메인 이름을 넣어야 하므로 접미사만 벗겨냅니다.
                //    이 경로가 있어야 scheduler 의 저장 테이블과 lib 의 조회 테이블이 어긋나도
                //    데이터가 존재하는 한 반드시 결과에 포함됩니다.
                let ctx_type: &str = match raw_ctx_type.strip_suffix("_items") {
                    Some(base) => {
                        println!("[AI-SEARCH] Table fallback context detected. Querying 'items' mirror with type = '{}'.", base);
                        base
                    },
                    None => raw_ctx_type,
                };
                let is_table_fallback = raw_ctx_type.ends_with("_items");
                // 🌟 [TRACKING EXACT MATCH] tracking_number는 LanceDB 물리적 컬럼이 아니므로(data JSON 내부 내장),
                // 백엔드 FTS/LIKE 검색을 수행하지 않습니다. 프론트엔드 Dexie DB의 eq 쿼리로 위임합니다.
                let search_text = text.to_string();
                let detected_tracking_number: Option<String> = {
                    let mut tn: Option<String> = None;
                    if let Some(cond_obj) = ctx.get("condition").and_then(|v| v.as_object()) {
                        if let Some(tn_cond) = cond_obj.get("tracking_number") {
                            if let Some(tn_val) = tn_cond.get("value").and_then(|v| v.as_str()) {
                                if !tn_val.is_empty() {
                                    tn = Some(tn_val.to_string());
                                    println!("[AI-SEARCH] Tracking number '{}' detected. Delegating to Dexie eq query (frontend).", tn_val);
                                }
                            }
                        }
                    }
                    tn
                };
                let target_table = if is_table_fallback {
                    // 🌟 items 는 scheduler 가 모든 타입을 이중 upsert 하는 미러 테이블입니다.
                    "items"
                } else {
                    match ctx_type {
                        "member" | "team" | "user" => "users",
                        "page" | "pages" => "pages",
                        "talk" => "talks",
                        "sales" | "goods" | "order" => "sales",
                        "tracking" | "shipping" | "receiving" => "tracking",
                        "event" | "coupon" => "event",
                        // 🌟 [TABLE MAPPING FIX] scheduler 의 저장 매핑은
                        //      "event" | "coupon" => "event",  나머지(review 포함) => "items"
                        //    입니다. 즉 review 는 items 에 저장되는데 여기서 event 를 조회하고 있어
                        //    데이터가 존재해도 물리적으로 항상 0건이 됩니다.
                        //    (로그: review A/FULL·B/NARROWED 두 티어 모두 Table: event / total_found: 0)
                        //    지금까지는 E/TABLE-FALLBACK 티어가 items 를 훑어 리콜을 겨우 보증했습니다.
                        "review" => "items",
                        _ => "items",
                    }
                };

                // 🌟 [FALLBACK SQL FIX] convert_conditions_to_sql 은 ctx["type"] 을 그대로 SQL 에 넣습니다.
                //    폴백 컨텍스트의 type 은 "review_items" 같은 라우팅 전용 이름이라
                //    DB 에 존재하지 않는 값이 되어 `type = 'review_items'` 로 나가고 항상 0건이 됩니다.
                //    (로그: [AI-SEARCH] Table fallback ... 직후 filter 가 "(type = 'review_items')" 로 출력됨)
                //    SQL 생성 시에는 접미사를 벗긴 원래 도메인 이름을 사용합니다.
                let sql_ctx = if is_table_fallback {
                    let mut c = ctx.clone();
                    if let Some(o) = c.as_object_mut() { o.insert("type".to_string(), json!(ctx_type)); }
                    c
                } else {
                    ctx.clone()
                };
                let sql_filter = convert_conditions_to_sql(&sql_ctx);
                let mode_filter = format!("mode = '{}'", search_mode);
                let final_sql_filter = match sql_filter {
                    Some(f) => Some(format!("({}) AND {}", f, mode_filter)),
                    None => Some(mode_filter.clone()),
                };
                // 🌟 [TRACKING EXACT MATCH] tracking_number가 감지되면 FTS를 비활성화하고,
                // text / json_data 컬럼에 SQL LIKE 필터를 걸어 해당 송장번호가 포함된 행만 정확히 조회합니다.
                let emb = model.get_embedding(search_text.clone()).await.unwrap_or(vec![0.0; 384]);
                let has_tracking = detected_tracking_number.is_some();
                let use_fts = !has_tracking;
                let tracking_sql = if let Some(ref tn) = detected_tracking_number {
                    let escaped_tn = tn.replace('\'', "''");
                    let like_clause = format!("(text LIKE '%{}%' OR json_data LIKE '%{}%')", escaped_tn, escaped_tn);
                    match final_sql_filter.clone() {
                        Some(f) => Some(format!("({}) AND {}", f, like_clause)),
                        None => Some(like_clause),
                    }
                } else {
                    final_sql_filter.clone()
                };
                let search_result = store.search_items(target_table, &search_text, emb.clone(), 5, 0, tracking_sql.clone(), use_fts).await;
                let final_results = match search_result {
                    Ok(res) => res,
                    Err(_) => {
                        // LIKE 필터가 DataFusion에서 실패하면 mode_filter만으로 폴백
                        store.search_items(target_table, &search_text, emb.clone(), 5, 0, Some(mode_filter.clone()), use_fts).await.unwrap_or_default()
                    }
                };
                // 🌟 [TRACKING EXACT MATCH] tracking_number는 LanceDB 물리적 컬럼이 아니므로(data JSON 내부 내장),
                // 백엔드 교차 검색을 수행하지 않습니다. 프론트엔드 Dexie DB의 eq 쿼리로 위임합니다.
                if let Some(ref tn) = detected_tracking_number {
                    println!("[AI-SEARCH] Tracking number '{}' cross-search delegated to Dexie eq query (frontend).", tn);
                }

                for (id, content, score) in final_results {
                    // 결과 배열 내 중복 방어
                    let is_dup = all_results.iter().any(|item: &serde_json::Value| item.get("id").and_then(|v| v.as_str()) == Some(&id));
                    if !is_dup {
                        all_results.push(json!({ "id": id.clone(), "text": content.clone(), "score": score, "context_type": ctx_type, "relation": "primary" }));
                    }

                    // 백엔드(LanceDB)에서의 N:N 양방향 교차 검색(FTS) 로직을 제거하고,
                    // 프론트엔드(Dexie DB)로 역할을 위임합니다.
                }

                // =====================================================================
                // 🌟 [STAGE-4] item_chunks 청크 레벨 코사인 유사도 매칭
                // ---------------------------------------------------------------------
                // 기존 search_items() 는 전체 문서 벡터 1개와 질의 벡터를 비교하므로
                // "무거운" ↔ "Its weight is 1.5" 같은 필드 레벨 매칭이 희석되어 0건이 됩니다.
                // STAGE-4 는 scheduler 가 사전 저장한 item_chunks 테이블의
                // 속성별 청크 임베딩과 코사인 유사도를 계산하여
                // 필드 레벨 매칭을 복원합니다.
                //
                // 매칭 전략:
                //   4-A: 질의 텍스트 임베딩으로 item_chunks 전체 코사인 검색
                //   4-B: PLINKO 조건(condition)에 확정된 속성이 있으면
                //        해당 속성 청크에 보너스 가중치 부여
                //   4-C: item_id 기준 그룹핑 후 기존 all_results 와 dedup 병합
                // =====================================================================
                {
                    // 4-A: item_chunks 테이블 코사인 검색
                    //      item_type 필터로 도메인을 좁히고, mode 필터로 검색 모드 일치 보장
                    let chunk_type_filter = format!("item_type = '{}' AND mode = '{}'", ctx_type, search_mode);
                    if is_table_fallback {
                        println!("[AI-SEARCH] 🧩 [STAGE-4] 폴백 컨텍스트 감지. item_type='{}' 로 청크 검색을 수행합니다.", ctx_type);
                    }

                    // 4-B: PLINKO 조건 기반 타겟 속성 목록 구축
                    //      ① condition 키 (PLINKO 가 확정한 속성)
                    //      ② substantial (추상 수식어가 지목한 물리 속성 — weight / sale_price ...)
                    //      substantial 은 LanceDB 물리 컬럼이 아니라 SQL 로는 절대 반영되지 않지만,
                    //      item_chunks 의 property 컬럼에는 그대로 존재하므로
                    //      '무거운 → weight' 의도를 여기서 실제 검색으로 회수합니다.
                    let mut condition_props: Vec<String> = ctx.get("condition")
                        .and_then(|v| v.as_object())
                        .map(|obj| obj.keys().cloned().collect())
                        .unwrap_or_default();

                    let substantial_prop = ctx.get("substantial")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if !substantial_prop.is_empty() && !condition_props.iter().any(|p| p == &substantial_prop) {
                        condition_props.push(substantial_prop.clone());
                    }

                    let mut chunk_results = store.search_chunks(
                        &emb,
                        10, // 상위 10개 청크 (item_id 그룹핑 전 오버페치)
                        Some(&chunk_type_filter),
                    ).await.unwrap_or_default();

                    // 4-C: 속성 타겟 청크 검색 (property 필터 고정)
                    //      전체 코사인 검색은 긴 질의 문장과 짧은 저장 청크 사이의 구조적 격차 때문에
                    //      정답 속성 청크가 상위 10개 밖으로 밀려날 수 있습니다.
                    //      확정된 속성마다 property 를 고정한 별도 쿼리를 던져 리콜을 보증합니다.
                    //
                    //      🌟 [ALIAS RECALL] 이 경로는 음차 별칭(_tn/_tr)이 살아 돌아오는 유일한 통로입니다.
                    //      별칭은 원본과 동일한 property 를 갖도록 저장되므로(STAGE-4B/4C 호환 목적),
                    //      limit 이 작으면 원본 청크만으로 창이 가득 차 별칭이 전멸합니다.
                    //      10개 상품 × (원본 + native + roman) = 30 행이 존재할 수 있으므로 창을 넓힙니다.
                    for target_prop in condition_props.iter() {
                        if target_prop.trim().is_empty() { continue; }
                        let escaped_prop = target_prop.replace('\'', "''");
                        let prop_filter = format!("{} AND property = '{}'", chunk_type_filter, escaped_prop);
                        let targeted = store.search_chunks(&emb, 12, Some(&prop_filter)).await.unwrap_or_default();
                        if targeted.is_empty() { continue; }
                        println!("[AI-SEARCH]   🎯 [STAGE-4C] property='{}' 타겟 청크 검색: {}건 추가 확보", target_prop, targeted.len());
                        for t in targeted {
                            if chunk_results.iter().any(|(cid, _, _, _, _, _)| cid == &t.0) { continue; }
                            chunk_results.push(t);
                        }
                    }

                    // 4-D: 조건이 하나도 없을 때의 값 청크 진입 보증
                    //      PLINKO 가 ACTION VERB 게이트 등으로 조건을 못 만들면 STAGE-4C 가 아예 돌지 않고,
                    //      전역 코사인 검색만 남습니다. 그런데 그 창을 저변별 청크(status/sale_price)가
                    //      독점하면 값이 담긴 title 청크가 후보에 진입조차 못 합니다.
                    //      (new_log2.txt: 조건 0개 → 상위 10건 전부 status/traffic_insight, title 0건)
                    //      스키마에서 '자유 서술 값을 담는 형식(Text)' 필드만 골라 타겟 검색을 겁니다.
                    //      필드 선정은 detect_field_format 의 결정론 판정만 사용하므로
                    //      다국어 어휘도, 필드명 하드코딩도 없습니다.
                    //
                    //      🌟 [ALIAS RECALL] 여기도 limit 3 은 치명적이었습니다.
                    //      store.rs 의 per_property_cap = max(2, 3/2) = 2 가 걸려
                    //      "title(16행)" 이 통째로 억제되었고(log 실측), 음차 별칭이 100% 소멸했습니다.
                    //      store.rs 가 property 고정 검색에서 캡을 해제했으므로,
                    //      여기서는 창 크기만 별칭 포함 규모로 넓혀 주면 됩니다.
                    if condition_props.is_empty() {
                        use crate::utils::ai_utils::FieldFormat;
                        let schema_fields = crate::parsing::get_detail_schema_fields(ctx_type, "", "en");
                        let mut probed = 0usize;
                        for (fname, _, _, _) in schema_fields.iter() {
                            if crate::utils::ai_utils::detect_field_format(fname) != FieldFormat::Text {
                                continue;
                            }
                            let escaped_prop = fname.replace('\'', "''");
                            let prop_filter = format!("{} AND property = '{}'", chunk_type_filter, escaped_prop);
                            let targeted = store.search_chunks(&emb, 12, Some(&prop_filter)).await.unwrap_or_default();
                            if targeted.is_empty() { continue; }
                            probed += targeted.len();
                            for t in targeted {
                                if chunk_results.iter().any(|(cid, _, _, _, _, _)| cid == &t.0) { continue; }
                                chunk_results.push(t);
                            }
                        }
                        if probed > 0 {
                            println!(
                                "[AI-SEARCH]   🧭 [STAGE-4D] 조건 부재 → Text 형식 필드 타겟 청크 검색: {}건 추가 확보 (값 청크 진입 보증)",
                                probed
                            );
                        }
                    }

                    if !chunk_results.is_empty() {
                        println!("[AI-SEARCH] 🧩 [STAGE-4] item_chunks 코사인 매칭: {}개 item 후보 발견 (ctx_type='{}')", chunk_results.len(), ctx_type);
                    }

                    for (chunk_id, item_id, chunk_text, property, group_score, best_cos) in chunk_results {
                        // 🌟 [SCALE FIX] 트랙 가중치는 반드시 '코사인(0~1)' 에만 곱합니다.
                        //    기존에는 search_chunks 가 돌려준 '그룹 합산 점수' 를 코사인이라 가정하고 곱해
                        //    (log 실측: score 3.0146 → 최종 10.5510) 상한 1.0 전제가 무너졌고,
                        //    질의와 무관해도 청크가 많이 살아남은 item 이 상위를 독점했습니다.
                        //    합산(group_score)은 '증거의 양' 이므로 별도의 완만한 보너스로만 반영합니다.
                        let raw_cosine = best_cos.clamp(0.0f32, 1.0f32);

                        // 🌟 [ALIAS FLAG] 이 매칭이 음차 별칭 청크에서 나왔는지 판정합니다.
                        //    scheduler 의 upsert_alias_chunks 가 chunk_id 에 _tn / _tr 접미어를 붙이므로
                        //    의미 판정이 아니라 '우리가 만든 접미어' 라는 구조적 사실입니다.
                        let is_alias = chunk_id.ends_with("_tn") || chunk_id.ends_with("_tr");

                        // 4-D: 3-Track 스케일 정합
                        //      store.rs 의 하이브리드 검색은 Column=3.0 / FTS=2.0 / Vector=1.0 가산 스케일입니다.
                        //      청크 매칭은 '본문 매칭' 이므로 FTS 트랙(2.0)과 동일 스케일로 환산하고,
                        //      PLINKO 가 확정한 속성과 일치하면 Column 트랙(3.0)을 추가로 얹습니다.
                        let mut score = raw_cosine * 2.0;
                        if !condition_props.is_empty() && condition_props.contains(&property) {
                            let column_track = raw_cosine * 3.0;
                            score += column_track;
                            println!("[AI-SEARCH]   🎯 [STAGE-4B] property='{}' PLINKO 조건 매칭 → Column 트랙 +{:.4} (cos {:.4}) → 최종 {:.4}", property, column_track, raw_cosine, score);
                        }

                        // 🌟 [STAGE-4X CROSS-LINGUAL VALUE BONUS]
                        //    "니트 가디건" ↔ "Cable Knit Cardigan" 처럼 값이 서로 다른 문자 체계일 때
                        //    FTS(ngram 문자열 포함)는 물리적으로 0건입니다.
                        //    이 경우 크로스링구얼 매칭이 가능한 트랙은 값 청크 코사인뿐이므로,
                        //    '값이 의미를 나르는 속성'에 FTS 결손분을 보전합니다.
                        {
                            use crate::utils::ai_utils::FieldFormat;
                            // 🌟 [ENUM EXCLUDE] Enum 값은 'complete' / 'show' / 'hide' 같은
                            //    저카디널리티 캐노니컬 키이며, 전 아이템에서 동일 문자열입니다.
                            //    자유 서술 값을 담는 Text / Address 만 크로스링구얼 보전 대상입니다.
                            let value_bearing = matches!(
                                crate::utils::ai_utils::detect_field_format(&property),
                                FieldFormat::Text | FieldFormat::Address
                            );
                            if value_bearing {
                                let cross_lingual_track = raw_cosine * 1.5;
                                score += cross_lingual_track;
                                println!(
                                    "[AI-SEARCH]   🌐 [STAGE-4X] property='{}' 자유서술 값 속성 → 크로스링구얼 트랙 +{:.4} (cos {:.4}) → 최종 {:.4}",
                                    property, cross_lingual_track, raw_cosine, score
                                );
                            }
                        }

                        // 🌟 [STAGE-4Y ALIAS TRACK] 음차 별칭이 매칭됐다는 것은
                        //    '문자 체계가 달라 FTS 가 물리적으로 0건인 상황에서, 발음 축으로 값이 일치했다'
                        //    는 뜻입니다. 이것이 바로 별칭을 저장한 목적이므로 전용 트랙을 부여합니다.
                        //    (원본 청크가 이미 확보한 점수를 대체하지 않고 가산만 하므로
                        //     별칭이 없는 item 이 손해를 보지 않습니다)
                        if is_alias {
                            let alias_track = raw_cosine * 1.5;
                            score += alias_track;
                            println!(
                                "[AI-SEARCH]   🔤 [STAGE-4Y] 음차 별칭 매칭 ({}) property='{}' | chunk='{}' → 별칭 트랙 +{:.4} (cos {:.4}) → 최종 {:.4}",
                                if chunk_id.ends_with("_tn") { "native" } else { "roman" },
                                property, chunk_text, alias_track, raw_cosine, score
                            );
                        }

                        // 🌟 [EVIDENCE VOLUME] 합산 점수는 '몇 개의 청크가 함께 반응했는가' 라는
                        //    보조 신호입니다. 로그 스케일로 완만하게만 반영하여
                        //    청크 수가 많은 item 이 코사인 우위를 뒤집지 못하게 합니다.
                        if group_score > raw_cosine {
                            let volume_bonus = ((group_score - raw_cosine).max(0.0) + 1.0).ln() * 0.25;
                            score += volume_bonus;
                        }

                        // 4-C: item_id 기준 기존 결과와 dedup
                        //      이미 all_results 에 동일 item_id 가 있으면
                        //      점수만 갱신하고 중복 삽입하지 않습니다.
                        let existing = all_results.iter_mut().find(|item| {
                            item.get("id").and_then(|v| v.as_str()) == Some(&item_id)
                        });

                        if let Some(existing_item) = existing {
                            // 기존 점수보다 높으면 갱신
                            let old_score = existing_item.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                            if score > old_score {
                                existing_item.as_object_mut().unwrap().insert("score".to_string(), json!(score));
                                existing_item.as_object_mut().unwrap().insert("chunk_match".to_string(), json!(true));
                                existing_item.as_object_mut().unwrap().insert("matched_property".to_string(), json!(property));
                                existing_item.as_object_mut().unwrap().insert("matched_chunk".to_string(), json!(chunk_text));
                                existing_item.as_object_mut().unwrap().insert("alias_match".to_string(), json!(is_alias));
                            }
                        } else {
                            // 신규 결과 삽입
                            // chunk_id 가 아닌 item_id 를 기준으로 삽입하여
                            // 동일 item 의 여러 청크가 중복 결과로 나타나지 않도록 합니다.
                            let is_item_dup = all_results.iter().any(|item| {
                                item.get("id").and_then(|v| v.as_str()) == Some(&item_id)
                            });
                            if !is_item_dup {
                                all_results.push(json!({
                                    "id": item_id,
                                    "text": chunk_text,
                                    "score": score,
                                    "context_type": ctx_type,
                                    "relation": "chunk_match",
                                    "chunk_id": chunk_id,
                                    "matched_property": property,
                                    "chunk_match": true,
                                    "alias_match": is_alias
                                }));
                            }
                        }
                    }
                }
                // =====================================================================
                // 🌟 [STAGE-4 종료]
                // =====================================================================
            }
        }

        // =====================================================================
        // 🌟 [STAGE-5] 결과 병합 & 스코어 랭킹 & 프론트엔드 응답 포맷 조립
        // ---------------------------------------------------------------------
        // STAGE-4 까지 수집된 all_results 에는 다음 두 종류의 결과가 혼재합니다.
        //   (A) search_items()  → 전체 문서 벡터 + FTS 매칭  (relation: "primary")
        //   (B) search_chunks() → 청크 레벨 코사인 매칭       (relation: "chunk_match")
        //
        // STAGE-5 의 역할:
        //   5-A: item_id 기준 dedup (동일 item 의 여러 청크가 중복 결과로 나타나지 않도록)
        //   5-B: 점수 정규화 (문서 점수와 청크 점수를 동일한 스케일로 병합)
        //   5-C: 매칭 속성(property) 메타데이터 주입 (프론트엔드 하이라이트용)
        //   5-D: 최종 랭킹 (내림차순) 후 limit 적용
        //   5-E: 프론트엔드 응답 JSON 포맷 확정
        // =====================================================================
        {
            // ── 5-A: item_id 기준 dedup ──
            // 동일 item_id 가 여러 번 등장하면 점수를 합산하고,
            // 가장 높은 청크 매칭 정보(property, chunk_text)를 대표로 남깁니다.
            let mut merged_map: std::collections::HashMap<String, serde_json::Value> =
                std::collections::HashMap::new();
            let mut merge_order: Vec<String> = Vec::new();

            for item in all_results.iter() {
                let item_id = item.get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if item_id.is_empty() { continue; }

                let score = item.get("score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                let is_chunk_match = item.get("chunk_match")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let matched_property = item.get("matched_property")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let matched_chunk = item.get("matched_chunk")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                if let Some(existing) = merged_map.get_mut(&item_id) {
                    // 기존 항목의 점수에 새 점수를 가산
                    let old_score = existing.get("score")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as f32;
                    let combined = old_score + score;
                    existing.as_object_mut().unwrap()
                        .insert("score".to_string(), json!(combined));

                    // 청크 매칭 정보가 더 구체적이면 대표 메타데이터로 교체
                    if is_chunk_match && !matched_property.is_empty() {
                        let old_prop = existing.get("matched_property")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if old_prop.is_empty() || score > old_score {
                            existing.as_object_mut().unwrap()
                                .insert("matched_property".to_string(), json!(matched_property));
                            existing.as_object_mut().unwrap()
                                .insert("matched_chunk".to_string(), json!(matched_chunk));
                            existing.as_object_mut().unwrap()
                                .insert("chunk_match".to_string(), json!(true));
                        }
                    }

                    // 매칭된 속성 목록 누적 (프론트엔드에서 다중 하이라이트 가능)
                    if is_chunk_match && !matched_property.is_empty() {
                        let props = existing.get("matched_properties")
                            .and_then(|v| v.as_array())
                            .cloned()
                            .unwrap_or_default();
                        let mut props_vec: Vec<Value> = props;
                        if !props_vec.iter().any(|p| p.as_str() == Some(&matched_property)) {
                            props_vec.push(json!(matched_property));
                        }
                        existing.as_object_mut().unwrap()
                            .insert("matched_properties".to_string(), json!(props_vec));
                    }
                } else {
                    // 신규 항목 삽입
                    let mut entry = item.clone();
                    if is_chunk_match && !matched_property.is_empty() {
                        entry.as_object_mut().unwrap()
                            .insert("matched_properties".to_string(), json!(vec![matched_property.clone()]));
                    }
                    merge_order.push(item_id.clone());
                    merged_map.insert(item_id, entry);
                }
            }

            // ── 5-B: 점수 정규화 ──
            // 문서 매칭(primary)과 청크 매칭(chunk_match)의 점수 스케일이 다를 수 있으므로
            // 최대 점수 기준으로 0.0~1.0 에 정규화합니다.
            // 단, 점수가 전부 0 이면 정규화를 건너뜁니다.
            let mut max_score: f32 = 0.0;
            for item in merged_map.values() {
                let s = item.get("score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                if s > max_score { max_score = s; }
            }

            let mut ranked_results: Vec<serde_json::Value> = Vec::new();
            for item_id in &merge_order {
                if let Some(mut item) = merged_map.remove(item_id) {
                    if max_score > 0.0 {
                        let raw = item.get("score")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0) as f32;
                        let normalized = raw / max_score;
                        item.as_object_mut().unwrap()
                            .insert("score".to_string(), json!(normalized));
                        item.as_object_mut().unwrap()
                            .insert("raw_score".to_string(), json!(raw));
                    }
                    ranked_results.push(item);
                }
            }

            // ── 5-D: 최종 랭킹 (점수 내림차순) ──
            ranked_results.sort_by(|a, b| {
                let sa = a.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let sb = b.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });

            // ── 5-E: 프론트엔드 응답 포맷 확정 ──
            // 프론트엔드(main.ts)가 소비하는 필드:
            //   id, text, score, context_type, relation,
            //   chunk_match(bool), matched_property, matched_chunk, matched_properties
            //
            // chunk_match = true 인 항목은 프론트엔드에서
            // "어떤 속성이 매칭되었는지" 를 배지로 표시할 수 있습니다.
            //
            // limit: 프론트엔드 렌더링 부하 방지를 위해 최대 20개 반환
            let final_limit = 20usize;
            if ranked_results.len() > final_limit {
                ranked_results.truncate(final_limit);
            }

            // ── 검색 통계 로그 출력 ──
            let chunk_match_count = ranked_results.iter()
                .filter(|r| r.get("chunk_match").and_then(|v| v.as_bool()).unwrap_or(false))
                .count();
            let primary_count = ranked_results.len() - chunk_match_count;

            println!("[AI-SEARCH] 📊 [STAGE-5] 결과 병합 완료: 총 {}건 (문서 매칭 {}건 + 청크 매칭 {}건)",
                ranked_results.len(), primary_count, chunk_match_count);

            for (rank, item) in ranked_results.iter().enumerate() {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                let score = item.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let prop = item.get("matched_property").and_then(|v| v.as_str()).unwrap_or("-");
                let is_chunk = item.get("chunk_match").and_then(|v| v.as_bool()).unwrap_or(false);
                let is_alias = item.get("alias_match").and_then(|v| v.as_bool()).unwrap_or(false);
                let ctx = item.get("context_type").and_then(|v| v.as_str()).unwrap_or("?");
                let matched_text = item.get("matched_chunk")
                    .or_else(|| item.get("text"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let brief: String = matched_text.chars().take(48).collect();
                println!("  [RANK {}] id={} | score={:.4} | type={} | match={} | property={} | matched='{}'",
                    rank + 1, id, score, ctx,
                    if is_chunk { if is_alias { "alias" } else { "chunk" } } else { "doc" },
                    prop, brief);
            }

            all_results = ranked_results;
        }
        // =====================================================================
        // 🌟 [STAGE-5 종료]
        // =====================================================================

        Ok(json!({ "structured": structured_query, "results": all_results }))
    }.await; 

    IS_SEARCHING.store(false, Ordering::SeqCst);
    
    if let Some(store) = store_opt.as_ref() {
        match &search_process {
            Ok(result_data) => { 
                let _ = store.update_task_status(&task_id, 9).await; 
                let _ = store.update_message_status(&task_id, 9, None).await;

                
                let payload_done = json!({ 
                    "task_id": task_id, 
                    "category": "Done", 
                    "summary": "AI Search Analysis Complete.", 
                    "spinner": "✅",
                    "data": result_data 
                });
                let _ = app_handle.emit("extraction-progress", &payload_done);
            },
            Err(e) => {
                let status_code = if e.contains("cancelled") { 3 } else { 6 };
                let _ = store.update_task_status(&task_id, status_code).await;
                
                // 🚨 [CRITICAL FIX] 에러 시에도 사용자의 쿼리를 덧붙이지 않고 깔끔하게 에러 사유만 표시합니다.
                let error_msg = format!("Task failed or cancelled: {}", e);
                let _ = store.update_message_status(&task_id, status_code, Some(&error_msg)).await;
            }
        }
    }

    
    {
        let mut mem_guard = crate::ACTIVE_TASK_MEM.write().unwrap();
        if let Some(mem) = mem_guard.as_ref() {
            if mem.get("id").and_then(|v| v.as_str()) == Some(task_id.as_str()) {
                *mem_guard = None;
            }
        }
    }

    model.deep_purge_resources().await; 
    
    
    drop(model_guard);
    IS_SEARCHING.store(false, Ordering::SeqCst);
    
    search_process
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
    device_preference: Option<String>,
) -> Result<String, String> {
    let mut model_guard = state.model.lock().await;

    // [FIX] Check if existing model matches preference
    if let Some(m) = model_guard.as_ref() {
        let wants_cpu = device_preference.as_deref() == Some("cpu");
        if m.is_cpu_mode != wants_cpu {
            println!("[DEEP-RESEARCH] Device preference mismatch. Reloading model...");
            m.deep_purge_resources().await;
            *model_guard = None;
        }
    }

    if model_guard.is_none() {
        if let Ok(m) = LogisModel::new(app_handle.clone(), device_preference.as_deref()).await {
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
        let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
        let _ = std::fs::create_dir_all(&db_path);
        if let Ok(s) = VectorStore::new(&db_path).await {
            let _ = s.init_task_table().await;
            let _ = s.init_all_tables().await;
            *store_guard = Some(s);
        }
    }
    
    if let Some(store) = store_guard.as_ref() {
        // General search for context
        let emb = model.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 384]);
        
        if let Ok(results) = store.search_items("items", &query, emb, 3, 0, None, false).await {
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



    for (k, v) in headers.iter() { req_builder = req_builder.header(k, v); }
    
    if let Some(b) = body { 
        if headers.get("Content-Encoding").map(|v| v.as_str()) == Some("gzip") {
            // [STRICT PARITY] Compress body if Gzip is requested
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
    r#ref: String,
}

#[tauri::command]
async fn check_active_task(
    _state: State<'_, AppState>,
    payload: ActiveTaskQuery,
) -> Result<bool, String> {
    
    if let Ok(mem_guard) = crate::ACTIVE_TASK_MEM.read() {
        if let Some(active) = mem_guard.as_ref() {
            let active_ref = active.get("ref").and_then(|v| v.as_str()).unwrap_or("");
            let status = active.get("status").and_then(|v| v.as_i64()).unwrap_or(0);
            
            
            // 완료된 작업(9)에 의해 추출 버튼이 영구적으로 숨겨지는 버그를 완벽히 막을 수 있습니다.
            if active_ref == payload.r#ref && (status == 1 || status == 10) {
                return Ok(true); // 현재 메모리에서 해당 페이지가 아직 처리 또는 대기 중임
            }
        }
    }
    Ok(false)
}

#[tauri::command]
async fn connect_with_seed(_target_ip: String, _seed: u64) -> Result<(), String> {
    // [DEPRECATED] UDP 방식은 더 이상 사용하지 않으며, 프론트엔드에서 
    // 직접 send_signal_offer(TCP)를 호출하도록 변경되었습니다.
    Ok(())
}

#[tauri::command]
async fn start_listener_command(app_handle: tauri::AppHandle, seed: u64) -> Result<(), String> {
    crate::utils::network::start_signal_listener(app_handle, seed);
    println!("Signal Listener started on port 9999 with seed: {}", seed);
    Ok(())
}

#[tauri::command]
async fn send_signal_offer(target_ip: String, seed: u64, sdp: String) -> Result<String, String> {
    crate::utils::network::send_signal_offer(target_ip, seed, sdp).await
}

#[tauri::command]
async fn submit_signal_answer(target_ip: String, sdp: String) -> Result<(), String> {
    crate::utils::network::submit_signal_answer(target_ip, sdp).await
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
async fn get_chat_messages(
    state: State<'_, AppState>,
    limit: usize,
    offset: usize,
    filter: Option<String>,
) -> Result<Vec<Value>, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        // 1. 일반 메시지 쿼리 (프론트엔드에서 요청한 limit, offset 적용)
        let mut messages = db.get_all_messages(limit, offset, filter.clone()).await.unwrap_or_default();
        
        
        // 진행 중(1)이거나 대기 중(10)인 활성 Task는 DB에서 한 번 더 쿼리하여 무조건 포함시킵니다!
        let active_filter = if let Some(ref f) = filter {
            format!("({}) AND status IN (1, 10)", f)
        } else {
            "status IN (1, 10)".to_string()
        };

        if let Ok(active_msgs) = db.get_all_messages(50, 0, Some(active_filter)).await {
            for active_msg in active_msgs {
                let active_id = active_msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                
                // 중복 방지: 이미 일반 쿼리(1번) 결과에 포함되어 있지 않은 녀석만 배열에 쏙 끼워 넣습니다.
                if !messages.iter().any(|m| m.get("id").and_then(|v| v.as_str()).unwrap_or("") == active_id) {
                    messages.push(active_msg);
                }
            }
        }
        
        Ok(messages)
    } else { 
        Ok(vec![]) 
    }
}


#[tauri::command]
async fn get_known_pages(state: State<'_, AppState>) -> Result<Vec<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        store.get_all_items("pages", 1000, 0, None).await.map_err(|e| e.to_string())
    } else { Ok(vec![]) }
}

#[tauri::command]
async fn get_known_users(state: State<'_, AppState>) -> Result<Vec<TradeDocument>, String> {
    let store_guard = state.store.lock().await;
    if let Some(store) = store_guard.as_ref() {
        let mut all_users = store.get_all_items("users", 50, 0, None).await.unwrap_or_default();
        
        
        if let Ok(team_docs) = store.get_all_items("users", 1, 0, Some("`type` = 'team'".to_string())).await {
            for t in team_docs {
                if !all_users.iter().any(|u| u.id == t.id) {
                    all_users.push(t);
                }
            }
        }
        Ok(all_users)
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
async fn get_browser_status() -> Result<Value, String> {
    let is_launching = crate::IS_BROWSER_LAUNCHING.load(std::sync::atomic::Ordering::SeqCst);
    
    // 1. 물리적 포트 응답 확인 및 메모리 가드 획득
    let reachable = automation::is_browser_reachable().await;
    let guard = automation::GLOBAL_BROWSER.lock().await; 
    
    
    // 브라우저 객체를 강제로 None 처리해버리는 치명적 버그 삭제. 실제 종료 정리는 백그라운드 핸들러가 수행함.
    
    // 2. 현재 브라우저의 물리적 실행 여부 판별
    let is_running = is_launching || guard.is_some() || reachable;
    let target_status = if is_running { "running" } else { "stopped" };

    // 3. 브라우저 상세 상태(URL, 권한 등) 추출
    // 에러 해결: LAST_DETECTED_STATE에서 안전하게 변수를 복사해옵니다.
    let (detected_url, is_client, is_admin) = {
        let state = automation::LAST_DETECTED_STATE.lock().await;
        (state.url.clone(), state.is_client, state.is_admin)
    };

    // 4. 즉각적인 상태 반영 (플리커링의 원인이었던 3초 지연 로직 철거)
    let status = target_status.to_string();
    {
        let mut current_state = crate::CURRENT_BROWSER_STATE.write().unwrap();
        *current_state = status.clone();
    }

    // 5. UI 버튼 숨김 여부 결정
    let hide_button = status == "running";

    Ok(json!({
        "status": status,
        "hide_button": hide_button,
        "url": detected_url,
        "is_client": is_client,
        "is_admin": is_admin
    }))
}

#[tauri::command]
async fn get_active_tasks(state: State<'_, AppState>) -> Result<Vec<store::Task>, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        let mut tasks = db.get_pending_tasks(10).await.unwrap_or_default();
        
        if let Ok(mut active) = db.get_processing_tasks(10).await {
            tasks.append(&mut active);
        }
        Ok(tasks)
    } else {
        Ok(vec![])
    }
}

#[tauri::command]
async fn get_task_logs(app_handle: tauri::AppHandle, task_id: String) -> Result<Vec<Value>, String> {
    let log_path = crate::utils::paths::get_task_log_file(Some(&app_handle), &task_id);
    
    
    // 메모리에 떠도는 최신 퍼센트를 강제로 끝에 끼워 넣으면 스텝 순서(stepMap)가 꼬입니다.
    if log_path.exists() {
        let content = std::fs::read_to_string(log_path).map_err(|e| e.to_string())?;
        Ok(content.lines().filter_map(|line| serde_json::from_str(line).ok()).collect())
    } else {
        Ok(vec![])
    }
}

#[tauri::command]
async fn upsert_items(state: State<'_, AppState>, items: Vec<Value>) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        let mut count = 0;
        for item in items {
            
            let id = item.get("id").and_then(|v| v.as_str())
                        .or_else(|| item.get("data").and_then(|d| d.get("id")).and_then(|v| v.as_str()))
                        .unwrap_or("").to_string();
            
            
            let type_str = item.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").trim().to_lowercase();

            
            println!("[DEBUG] Syncing item - ID: {}, Type: {}", id, type_str);
            
            
            let mut clean_item = item.clone();
            if let Some(obj) = clean_item.as_object_mut() {
                obj.insert("type".to_string(), serde_json::json!(type_str.clone()));
                
                // 🌟 [CRITICAL FIX] Dexie DB에서 최상단(Root)으로 꺼내졌던 스키마 컬럼들을 
                // 다시 Rust의 JSON 페이로드(내부 속성)로 안전하게 복원하여 LanceDB 컬럼에 매핑 시 누락되지 않도록 동기화합니다.
                if let Some(status) = item.get("status") { obj.insert("status".to_string(), status.clone()); }
                if let Some(amount) = item.get("amount") { obj.insert("amount".to_string(), amount.clone()); }
                if let Some(mode) = item.get("mode") { obj.insert("mode".to_string(), mode.clone()); }
                if let Some(is_masked) = item.get("is_masked") { obj.insert("is_masked".to_string(), is_masked.clone()); }
                if let Some(created_at) = item.get("created_at") { obj.insert("created_at".to_string(), created_at.clone()); }
                if let Some(updated_at) = item.get("updated_at") { obj.insert("updated_at".to_string(), updated_at.clone()); }
            }

            
            // item 안에 "data"가 객체로 존재한다면, 그 안의 알맹이를 최상위로 끌어올립니다.
            if type_str != "talk" && type_str != "prompt" && type_str != "ai_search" {
                if let Some(data_obj) = clean_item.get("data").and_then(|v| v.as_object()).cloned() {
                    if let Some(main_obj) = clean_item.as_object_mut() {
                        for (k, v) in data_obj {
                            main_obj.insert(k, v);
                        }
                        main_obj.remove("data"); // 기존 껍데기 data 제거
                    }
                }
            }
            
            
            if type_str == "talk" || type_str == "prompt" || type_str == "ai_search" {
                let text_val = clean_item.get("text")
                    .or_else(|| clean_item.get("query"))
                    .or_else(|| clean_item.get("data").and_then(|d| d.get("text")))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let link_val = clean_item.get("link")
                    .or_else(|| clean_item.get("data").and_then(|d| d.get("link")))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let origin_val = clean_item.get("origin")
                    .or_else(|| clean_item.get("data").and_then(|d| d.get("origin")))
                    .and_then(|v| v.as_str())
                    .unwrap_or("https://commerce.logis.center")
                    .to_string();

                // 기존 최상위 잔재 필드들을 깔끔하게 지웁니다.
                if let Some(obj) = clean_item.as_object_mut() {
                    obj.remove("text");
                    obj.remove("query");
                    obj.remove("link");
                    obj.remove("origin");
                    
                    // 프록시 서버(index.ts)와 동일하게 data 객체 안에 세 가지 필수 값을 몰아넣습니다.
                    obj.insert("data".to_string(), json!({
                        "text": text_val,
                        "link": link_val,
                        "origin": origin_val
                    }));
                }

                // 🌟 [CRITICAL FIX] 서버에서 받아온 talk/prompt 메시지는 items 테이블(upsert_item)이 아니라 
                // 반드시 messages 테이블(add_message)로 직접 인서트해야 화면의 get_chat_messages에서 정상 조회됩니다!
                let from_val = clean_item.get("from").and_then(|v| v.as_str()).unwrap_or("");
                let to_val = clean_item.get("to").and_then(|v| v.as_str()).unwrap_or("");
                let cc_val = clean_item.get("cc").and_then(|v| v.as_str()).unwrap_or("");
                let bcc_val = clean_item.get("bcc").and_then(|v| v.as_str()).unwrap_or("");
                let ref_val = clean_item.get("ref").and_then(|v| v.as_str()).unwrap_or("");
                let status_val = clean_item.get("status").and_then(|v| v.as_i64()).unwrap_or(9) as i32;
                let created_at_val = clean_item.get("created_at").map(|v| v.to_string()).unwrap_or_else(|| chrono::Utc::now().timestamp_millis().to_string());
                
                // Chrome.js처럼 role 구분 (프론트에서 address 비교로 재교정되므로 기본값 user 지정)
                let role_val = if type_str == "talk" { "user" } else { "system_task" };

                if !id.is_empty() {
                    let _ = db.add_message(
                        &id, role_val, &text_val, 
                        None, Some(status_val), 
                        Some(cc_val), Some(bcc_val), Some(ref_val), 
                        Some(from_val), Some(to_val), Some(type_str.as_str()), 
                        Some(created_at_val.as_str())
                    ).await;
                    count += 1;
                }
                continue; // messages 테이블에 저장했으므로 하단의 items/talks 테이블 저장 로직은 건너뜀
            }

            // Determine table based on cleaned type
            let final_table = match type_str.as_str() {
                "sales" | "goods" | "order" => "sales",
                "tracking" | "receiving" | "shipping" => "tracking",
                "event" | "coupon" => "event",
                "member" | "team" | "user" => "users",
                "talk" | "prompt" | "ai_search" => "talks", 
                "pages" | "page" => "pages", 
                _ => {
                    if clean_item.get("data").and_then(|d| d.get("origin")).is_some() {
                        "pages"
                    } else {
                        "items" 
                    }
                }
            };

            
            let from = item.get("from").and_then(|v| v.as_str());
            let to = item.get("to").and_then(|v| v.as_str());
            let cc = item.get("cc").and_then(|v| v.as_str());
            let bcc = item.get("bcc").and_then(|v| v.as_str());
            let r#ref = item.get("ref").and_then(|v| v.as_str());
            let digest = item.get("digest").and_then(|v| v.as_str());

            if !id.is_empty() {
                // 원본 item 대신 세탁된 clean_item을 DB에 밀어 넣습니다.
                let _ = db.upsert_item(final_table, &id, &type_str, clean_item, None, from, to, cc, bcc, r#ref, digest).await;
                count += 1;
            }
        }
        Ok(format!("Synced {} items", count))
    } else {
        Err("DB not initialized".to_string())
    }
}

#[derive(serde::Serialize)]
struct InitialSyncData {
    tasks: Vec<store::Task>,
    pages: Vec<store::TradeDocument>,
    users: Vec<store::TradeDocument>,
    items: Vec<store::TradeDocument>,
    browser_status: String,
    current_url: String,
    is_client: bool,
    is_admin: bool,
}

#[tauri::command]
async fn mark_ui_ready(state: State<'_, AppState>) -> Result<InitialSyncData, String> {
    crate::utils::sync_utils::mark_ui_ready();
    
    let store_guard = state.store.lock().await;
    let mut tasks = Vec::new();
    let mut pages = Vec::new();
    let mut users = Vec::new();
    let mut items = Vec::new();
    
    if let Some(db) = store_guard.as_ref() {
        let mut raw_tasks = db.get_pending_tasks(10).await.unwrap_or_default();
        
        if let Ok(mut active) = db.get_processing_tasks(10).await {
            raw_tasks.append(&mut active);
        }
        
        
        let mem_task_id = if let Ok(mem) = crate::ACTIVE_TASK_MEM.read() {
            mem.as_ref().and_then(|v| v.get("id")).and_then(|v| v.as_str()).unwrap_or("").to_string()
        } else { 
            "".to_string() 
        };

        for t in raw_tasks {
            
            if t.status == 1 && t.id != mem_task_id {
                let error_status = crate::logic::parse_status("error");
                println!("[DB-SYNC] Zombie task detected in DB: {}. Marking as ERROR.", t.id);
                let _ = db.update_task_status(&t.id, error_status).await;
                let _ = db.update_message_status(&t.id, error_status, Some("App closed unexpectedly. Task failed.")).await;
            } 
            
            else if t.status == 1 || t.status == 10 {
                tasks.push(t);
            }
        }

        
        pages = db.get_all_items("pages", 1000, 0, None).await.unwrap_or_default();
        
        
        users = db.get_all_items("users", 50, 0, None).await.unwrap_or_default();
        
        if let Ok(team_docs) = db.get_all_items("users", 1, 0, Some("`type` = 'team'".to_string())).await {
            for t in team_docs {
                if !users.iter().any(|u| u.id == t.id) {
                    users.push(t);
                }
            }
        }
        
        items = db.get_all_items("items", 50, 0, None).await.unwrap_or_default();
    }
    
    let browser_status = {
        let is_launching = crate::IS_BROWSER_LAUNCHING.load(std::sync::atomic::Ordering::SeqCst);
        let reachable = automation::is_browser_reachable().await;
        let guard = automation::GLOBAL_BROWSER.lock().await; 
        
        // 강제 메모리 해제 로직 제거
        
        let is_running = is_launching || guard.is_some() || reachable;
        let target_status = if is_running { "running" } else { "stopped" };

        // 지연 없이 물리적 상태를 즉시 동기화
        let mut current_state = crate::CURRENT_BROWSER_STATE.write().unwrap();
        *current_state = target_status.to_string();
        current_state.clone()
    };

    let (current_url, is_client, is_admin) = {
        let state = automation::LAST_DETECTED_STATE.lock().await;
        (state.url.clone(), state.is_client, state.is_admin)
    };

    Ok(InitialSyncData {
        tasks,
        pages,
        users,
        items,
        browser_status,
        current_url,
        is_client,
        is_admin,
    })
}

#[tauri::command]
async fn check_gpu_availability() -> serde_json::Value {
    let config = crate::utils::get_optimal_device_config();
    let mut vendor = "none";

    if !config.is_cpu {
        vendor = "amd"; // 기본적으로 CPU가 아니면 AMD(ROCm)로 가정

        // NVIDIA GPU 여부 판별 (nvml-wrapper 사용)
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        {
            if nvml_wrapper::Nvml::init().is_ok() {
                vendor = "nvidia";
            }
        }

        #[cfg(target_os = "macos")]
        {
            vendor = "apple";
        }
    }

    serde_json::json!({
        "has_gpu": !config.is_cpu,
        "vendor": vendor
    })
}

#[tauri::command]
async fn save_mobile_temp_file(
    app_handle: tauri::AppHandle,
    filename: String,
    data: Vec<u8>,
) -> Result<String, String> {
    let temp_dir = app_handle.path().app_cache_dir().map_err(|e| e.to_string())?.join("mobile_uploads");
    std::fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;
    
    let file_path = temp_dir.join(filename);
    std::fs::write(&file_path, data).map_err(|e| e.to_string())?;
    
    Ok(file_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn check_model_status() -> Result<serde_json::Value, String> {
    let app_dir = crate::utils::get_app_dir();
    let base_path = app_dir.join("models");

    // 🌟 [CRITICAL FIX] 특정 폴더 내에 10MB 이상의 GGUF 또는 SAFETENSORS 파일이 존재하는지 검사하도록 조건 확장
    let has_valid_model = |dir: &std::path::PathBuf| -> bool {
        if let Ok(entries) = std::fs::read_dir(dir) {
            entries.flatten().any(|e| {
                let is_model_ext = e.path().extension().map_or(false, |ext| ext == "gguf" || ext == "safetensors");
                is_model_ext && e.metadata().map(|m| m.len()).unwrap_or(0) > 10_000_000
            })
        } else {
            false
        }
    };
    
    // 🌟 [추가] Stanza 모델 체크용 함수 추가
    let has_valid_stanza = |dir: &std::path::PathBuf| -> bool {
        if let Ok(entries) = std::fs::read_dir(dir) {
            entries.flatten().any(|e| {
                e.path().extension().map_or(false, |ext| ext == "onnx") && e.metadata().map(|m| m.len()).unwrap_or(0) > 1_000_000
            })
        } else {
            false
        }
    };

    let qwen3_dir = base_path.join("Qwen3-0.6B-Instruct-gguf");
    let qwen3_5_dir = base_path.join("Qwen3.5-2B-Instruct-gguf");
    let granite_dir = base_path.join("granite-4.0-h-350m");
    let embed_dir = base_path.join("granite-embedding-97m-multilingual-r2");

    let mut status_map = serde_json::Map::new();
    status_map.insert("Qwen3".to_string(), serde_json::json!(has_valid_model(&qwen3_dir)));
    status_map.insert("Qwen3.5".to_string(), serde_json::json!(has_valid_model(&qwen3_5_dir)));
    status_map.insert("Granite".to_string(), serde_json::json!(has_valid_model(&granite_dir)));
    status_map.insert("Embedding".to_string(), serde_json::json!(has_valid_model(&embed_dir)));

    let supported_stanza_langs = [
        "korean", "english", "japanese", "chinese", "french", "german", "spanish", 
        "italian", "portuguese", "dutch", "russian", "arabic", "thai", "hindi", 
        "bengali", "greek", "hebrew", "vietnamese"
    ];

    for lang in supported_stanza_langs.iter() {
        let lang_code = match *lang {
            "korean" => "ko",
            "english" => "en",
            "japanese" => "ja",
            "chinese" => "zh-hans",
            "french" => "fr",
            "german" => "de",
            "spanish" => "es",
            "italian" => "it",
            "portuguese" => "pt",
            "dutch" => "nl",
            "russian" => "ru",
            "arabic" => "ar",
            "thai" => "th",
            "hindi" => "hi",
            "bengali" => "bn",
            "greek" => "el",
            "hebrew" => "he",
            "vietnamese" => "vi",
            _ => "en",
        };
        let stanza_dir = base_path.join(format!("stanza/{}", lang_code));
        status_map.insert(format!("stanza_{}", lang), serde_json::json!(has_valid_stanza(&stanza_dir)));
    }

    Ok(serde_json::Value::Object(status_map))
}

#[tauri::command]
async fn delete_all_models() -> Result<String, String> {
    let app_dir = crate::utils::get_app_dir();
    let models_dir = app_dir.join("models");
    if models_dir.exists() {
        std::fs::remove_dir_all(&models_dir).map_err(|e| e.to_string())?;
    }

    // 🌟 [VISION-CACHE] 모델이 사라지면 ViT 출력의 재현성도 보장할 수 없으므로
    //    캐시된 비전 임베딩을 함께 폐기합니다.
    crate::models::vision_cache::VISION_CACHE.clear_all();

    Ok("Deleted".to_string())
}

#[tauri::command]
async fn reset_lancedb(
    state: State<'_, AppState>,
) -> Result<String, String> {
    let store_guard = state.store.lock().await;
    if let Some(db) = store_guard.as_ref() {
        db.reset_database().await.map_err(|e| e.to_string())?;
        println!("[RESET] LanceDB fully reset completed.");
        Ok("LanceDB reset complete.".to_string())
    } else {
        // Store가 아직 초기화되지 않은 경우 직접 생성해서 리셋
        let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
        match VectorStore::new(&db_path).await {
            Ok(s) => {
                s.reset_database().await.map_err(|e| e.to_string())?;
                println!("[RESET] LanceDB fully reset completed (fresh connection).");
                Ok("LanceDB reset complete.".to_string())
            },
            Err(e) => Err(format!("Failed to connect to LanceDB for reset: {}", e)),
        }
    }
}

#[tauri::command]
async fn download_model(app_handle: tauri::AppHandle, model_name: String) -> Result<String, String> {
    let app_dir = crate::utils::get_app_dir();
    let app_dir_clone = app_dir.clone();
    
    tokio::task::spawn(async move {
        let base_path = app_dir_clone.join("models");

        // 🌟 [수정] stanza 다운로드 경로 동적 맵핑 (models/stanza/{lang} 구조 지원)
        let folder_name = if model_name.starts_with("stanza_") {
            let lang = model_name.replace("stanza_", "");
            let lang_code = match lang.as_str() {
                "korean" => "ko",
                "english" => "en",
                "japanese" => "ja",
                "chinese" => "zh-hans",
                "french" => "fr",
                "german" => "de",
                "spanish" => "es",
                "italian" => "it",
                "portuguese" => "pt",
                "dutch" => "nl",
                "russian" => "ru",
                "arabic" => "ar",
                "thai" => "th",
                "hindi" => "hi",
                "bengali" => "bn",
                "greek" => "el",
                "hebrew" => "he",
                "vietnamese" => "vi",
                _ => "en",
            };
            format!("stanza/{}", lang_code)
        } else {
            match model_name.as_str() {
                "Qwen3" => "Qwen3-0.6B-Instruct-gguf".to_string(),
                "Qwen3.5" => "Qwen3.5-2B-Instruct-gguf".to_string(),
                "Embedding" => "granite-embedding-97m-multilingual-r2".to_string(),
                "Granite" => "granite-4.0-h-350m".to_string(),
                _ => "unknown".to_string()
            }
        };

        let dir_path = base_path.join(folder_name);
        if let Err(e) = std::fs::create_dir_all(&dir_path) {
            let _ = app_handle.emit("download_error", serde_json::json!({"model": model_name, "error": e.to_string()}));
            return;
        }
        
        // 🌟 [수정] 문자열 타입(String)으로 변환하여 다국어 url을 생성할 수 있도록 확장
        let files_to_download: Vec<(String, String)> = if model_name.starts_with("stanza_") {
            let lang = model_name.replace("stanza_", "");
            let lang_code = match lang.as_str() {
                "korean" => "ko",
                "english" => "en",
                "japanese" => "ja",
                "chinese" => "zh-hans",
                "french" => "fr",
                "german" => "de",
                "spanish" => "es",
                "italian" => "it",
                "portuguese" => "pt",
                "dutch" => "nl",
                "russian" => "ru",
                "arabic" => "ar",
                "thai" => "th",
                "hindi" => "hi",
                "bengali" => "bn",
                "greek" => "el",
                "hebrew" => "he",
                "vietnamese" => "vi",
                _ => "en",
            };
            vec![
                (format!("https://huggingface.co/PopupLink/stanza-{}/resolve/main/vocab.json", lang_code), "vocab.json".to_string()),
                (format!("https://huggingface.co/PopupLink/stanza-{}/resolve/main/pos.onnx", lang_code), "pos.onnx".to_string()),
                (format!("https://huggingface.co/PopupLink/stanza-{}/resolve/main/tokenizer.onnx", lang_code), "tokenizer.onnx".to_string()),
                (format!("https://huggingface.co/PopupLink/stanza-{}/resolve/main/depparse.onnx", lang_code), "depparse.onnx".to_string()),
                (format!("https://huggingface.co/PopupLink/stanza-{}/resolve/main/lemma.onnx", lang_code), "lemma.onnx".to_string()),
            ]
        } else {
            match model_name.as_str() {
                "Qwen3" => vec![
                    ("https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf".to_string(), "Qwen3-0.6B-Q8_0.gguf".to_string())
                ],
                "Qwen3.5" => vec![
                    ("https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-BF16.gguf".to_string(), "mmproj-BF16.gguf".to_string()),
                    ("https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q8_0.gguf".to_string(), "Qwen3.5-2B-Q8_0.gguf".to_string())
                ],
                "Embedding" => vec![
                    ("https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2/resolve/main/model.safetensors".to_string(), "model.safetensors".to_string())
                ],
                "Granite" => vec![
                    ("https://huggingface.co/ibm-granite/granite-4.0-h-350m/resolve/main/model.safetensors".to_string(), "model.safetensors".to_string())
                ],
                _ => vec![]
            }
        };

        let total_files = files_to_download.len();
        let client = reqwest::Client::new();
        let mut has_error = false;

        for (file_idx, (url, filename)) in files_to_download.iter().enumerate() {
            let file_path = dir_path.join(filename);
            let tmp_path = dir_path.join(format!("{}.tmp", filename));
            
            let min_size = if filename.ends_with(".gguf") || filename.ends_with(".safetensors") { 10_000_000 } else { 0 };
            if file_path.exists() && std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0) > min_size {
                let percent = (((file_idx as f64 + 1.0) / total_files as f64) * 100.0) as u32;
                let _ = app_handle.emit("download_progress", serde_json::json!({"model": model_name, "percent": percent}));
                continue;
            }

            // 🌟 [추가] 다운로드 링크 터미널 로깅 및 UI 이벤트 발송
            println!("[DOWNLOAD] 다운로드 시작: {} (URL: {})", filename, url);
            let _ = app_handle.emit("download_progress", serde_json::json!({
                "model": model_name,
                "percent": 0,
                "message": format!("다운로드 중: {} (URL: {})", filename, url)
            }));

            // 🌟 [수정] url이 &String 이므로 역참조(*) 없이 그대로 전달합니다.
            match client.get(url).send().await {
                Ok(res) => {
                    if !res.status().is_success() {
                        let _ = app_handle.emit("download_error", serde_json::json!({"model": model_name, "error": format!("HTTP {}", res.status())}));
                        has_error = true;
                        break;
                    }
                    
                    let total_size = res.content_length().unwrap_or(0) as f64;
                    let mut downloaded = 0.0;
                    
                    if let Ok(mut file) = tokio::fs::File::create(&tmp_path).await {
                        use tokio::io::AsyncWriteExt;
                        use futures::StreamExt;
                        let mut stream = res.bytes_stream();
                        let mut write_error = false;
                        
                        while let Some(chunk_result) = stream.next().await {
                            match chunk_result {
                                Ok(chunk) => {
                                    if let Err(_) = file.write_all(&chunk).await {
                                        write_error = true; break;
                                    }
                                    downloaded += chunk.len() as f64;
                                    let file_progress = if total_size > 0.0 { downloaded / total_size } else { 0.0 };
                                    let percent = (((file_idx as f64 + file_progress) / total_files as f64) * 100.0) as u32;
                                    let _ = app_handle.emit("download_progress", serde_json::json!({"model": model_name, "percent": percent}));
                                },
                                Err(_) => { write_error = true; break; }
                            }
                        }
                        
                        if write_error {
                            let _ = std::fs::remove_file(&tmp_path);
                            has_error = true;
                            break;
                        } else {
                            let _ = std::fs::rename(&tmp_path, &file_path);
                        }
                    }
                },
                Err(_) => {
                    has_error = true;
                    break;
                }
            }
        }
        
        if !has_error {
            let _ = app_handle.emit("download_complete", serde_json::json!({"model": model_name}));
        }
    });

    Ok("Started".to_string())
}


#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let model = Arc::new(TokioMutex::new(None));
    let store = Arc::new(TokioMutex::new(None));
    let cancellation_token = Arc::new(AtomicBool::new(false));

    tauri::Builder::default()
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            model: model.clone(),
            store: store.clone(),
            cancellation_token: cancellation_token.clone(),
        })
        .setup(|app| {
            // [INIT] Copy model configs from local project source to AppData if they don't exist
            let app_dir = crate::utils::get_app_dir();
            let dest_models_dir = app_dir.join("models");
            
            let src_models_dir1 = std::env::current_dir().unwrap_or_default().join("models");
            let src_models_dir2 = std::env::current_dir().unwrap_or_default().join("src-tauri").join("models");
            
            let src_dir = if src_models_dir1.exists() {
                Some(src_models_dir1)
            } else if src_models_dir2.exists() {
                Some(src_models_dir2)
            } else {
                None
            };

            if let Some(src) = src_dir {
                println!("[Setup] Syncing model configs from {:?} to {:?}", src, dest_models_dir);
                let _ = crate::utils::paths::copy_model_configs(&src, &dest_models_dir);
            }

            // [INIT] AppData/tmp 내부의 kv, logs, task_data 디렉토리를 초기화하여 이전 실행의 찌꺼기 완벽 삭제
            crate::utils::paths::cleanup_temp_dirs(Some(app.handle()));

            // [INIT] KV Bake Worker (Immediate)
            crate::models::qwen::generate::init_bake_worker();

            // [FIX] Reset stop signals immediately on app startup
            let setup_cancel = app.state::<AppState>().cancellation_token.clone();
            setup_cancel.store(false, Ordering::SeqCst);
            crate::utils::set_extraction_stop_signal(false);

            let setup_store = app.state::<AppState>().store.clone();
            
            // spawn 대신 block_on 계열의 처리를 통해 순서를 보장합니다.
            tauri::async_runtime::block_on(async move {
                let mut store_guard = setup_store.lock().await;
                let db_path = crate::utils::get_app_dir().join("db").to_string_lossy().into_owned();
                let _ = std::fs::create_dir_all(&db_path);
                if let Ok(s) = VectorStore::new(&db_path).await {
                    println!("[Setup] VectorStore initialized. Recovering zombie records...");
                    let _ = s.init_task_table().await;
                    let _ = s.init_all_tables().await;
                    
                    
                    let _ = s.cleanup_unfinished_tasks_on_startup().await;
                    
                    let error_status = crate::logic::parse_status("error");
                    
                    if let Ok(processing_tasks) = s.get_processing_tasks(100).await {
                        for t in processing_tasks {
                            let _ = s.update_task_status(&t.id, error_status).await;
                            let _ = s.update_message_status(&t.id, error_status, Some("App closed unexpectedly. Task failed.")).await;
                        }
                    }
                    
                    if let Ok(pending_tasks) = s.get_pending_tasks(100).await {
                        for t in pending_tasks {
                            let _ = s.update_task_status(&t.id, error_status).await;
                            let _ = s.update_message_status(&t.id, error_status, Some("App closed unexpectedly. Task failed.")).await;
                        }
                    }
                    
                    *store_guard = Some(s);
                    println!("[Setup] Zombie cleanup complete. VectorStore is ready.");
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

            
            let status_monitor_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let mut last_status = String::new();
                loop {
                    let is_launching = crate::IS_BROWSER_LAUNCHING.load(std::sync::atomic::Ordering::SeqCst);
                    let reachable = automation::is_browser_reachable().await;
                    let guard = automation::GLOBAL_BROWSER.lock().await; 
                    
                    // 강제 메모리 해제 로직 제거
                    
                    let is_running = is_launching || guard.is_some() || reachable;
                    
                    // [수정] 런칭 중이거나 물리적으로 감지되었을 때는 무조건 running으로 고정합니다.
                    // 특히 is_launching이 true인 동안은 target_status가 절대로 stopped가 될 수 없습니다.
                    let target_status = if is_running { "running" } else { "stopped" };
                    
                    let current_status = {
                        let mut current_state = crate::CURRENT_BROWSER_STATE.write().unwrap();
                        *current_state = target_status.to_string();
                        current_state.clone()
                    };
                    
                    if current_status != last_status {                        
                        
                        let is_launching = crate::IS_BROWSER_LAUNCHING.load(std::sync::atomic::Ordering::SeqCst);
                        let (is_client, is_admin, url) = {
                            let state = automation::LAST_DETECTED_STATE.lock().await;
                            (state.is_client, state.is_admin, state.url.clone())
                        };
                        // [수정] 상태 모니터에서도 브라우저가 실행 중(running)이라면 새 탭(빈 URL)이든 아니든 무조건 버튼을 숨겨 
                        // 프론트엔드로 hide_button: false가 날아가 깜빡이는 현상을 원천 차단합니다.
                        let hide_button = current_status == "running";
                        
                        use tauri::Emitter;
                        // [수정] 이벤트 페이로드를 생성할 때, 런칭 중(is_launching)이라면 
                        // URL 감지 결과와 상관없이 hide_button을 무조건 true로 고정하여 발송합니다.
                        let payload = json!({
                            "status": current_status.clone(),
                            "hide_button": if is_launching { true } else { hide_button }
                        });
                        let _ = status_monitor_handle.emit("browser-status", payload);
                        last_status = current_status;
                    }
                    // 빠른 UI 복구를 위해 1초 간격으로 감시 속도 단축
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            });

            let event_store = app.state::<AppState>().store.clone();
            let event_cancel = app.state::<AppState>().cancellation_token.clone();
            
            let handle_for_event = app.handle().clone(); 

            app.listen("new-task-from-browser", move |event| {
                event_cancel.store(false, Ordering::SeqCst);
                crate::utils::set_extraction_stop_signal(false);
                let app_handle = handle_for_event.clone(); 

                if let Ok(payload_val) = serde_json::from_str::<serde_json::Value>(event.payload()) {
                    let store_clone = event_store.clone();
                    tauri::async_runtime::spawn(async move {
                        let store_guard = store_clone.lock().await;
                        if let Some(db) = store_guard.as_ref() {
                            let now = chrono::Utc::now().timestamp_millis();
                            
                            
                            let zero_addr = "0x0000000000000000000000000000000000000000";
                            let from_addr = payload_val.get("from").and_then(|v| v.as_str()).unwrap_or(zero_addr).to_string();
                            
                            let raw_to = payload_val.get("to").and_then(|v| v.as_str()).unwrap_or("");
                            let team_id = if raw_to.is_empty() || raw_to == zero_addr {
                                crate::utils::hash::hash_id(&from_addr)
                            } else {
                                raw_to.to_string()
                            };

                            let task = crate::store::Task {
                                id: payload_val.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                                r#type: payload_val.get("type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
                                from: from_addr, to: team_id,
                                cc: payload_val.get("cc").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                                bcc: payload_val.get("bcc").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                                r#ref: payload_val.get("ref").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                                data_json: payload_val.to_string(), created_at: now, updated_at: now, status: 10,
                            };
                            let msg_text = format!("Task Started: {}", payload_val.get("link").and_then(|v| v.as_str()).unwrap_or("Unknown URL"));
                            
                            let _ = db.add_message(
                                &task.id, "system_task", &msg_text, Some(&task.id), Some(10),
                                Some(&task.cc), Some(&task.bcc), Some(&task.r#ref),
                                Some(&task.from), Some(&task.to), Some("talk"), None
                            ).await;
                            
                            let _ = db.add_task(task.clone()).await;
                            
                            
                            let _ = app_handle.emit("task-db-registered", json!({
                                "task_id": task.id,
                                "status": task.status,
                                "created_at": task.created_at,
                                "text": msg_text
                            }));
                            crate::utils::sync_utils::notify_new_task();
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
            get_known_pages, get_known_users, initialize_hub, get_browser_status, get_active_tasks, unload_model, get_task_logs,
            upsert_items, set_ignore_cursor_events, mark_ui_ready, delete_document, delete_documents, delete_message, check_gpu_availability,
            save_mobile_temp_file, crate::utils::network::get_local_network_prefix, crate::utils::network::get_my_full_ip, connect_with_seed, start_listener_command, send_signal_offer, submit_signal_answer,
            get_active_task_context, check_model_status, download_model, delete_all_models, reset_lancedb,
            get_query_embedding, reindex_pending_embeddings
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                println!("[APP] Application exiting. Shutting down browser...");

                // 1. 전역 브라우저 상태를 즉시 stopped으로 고정
                if let Ok(mut state) = crate::CURRENT_BROWSER_STATE.write() {
                    *state = "stopped".to_string();
                }
                crate::IS_BROWSER_LAUNCHING.store(false, std::sync::atomic::Ordering::SeqCst);

                // 2. automation::shutdown_browser()로 명시적 close() 호출 후 인스턴스 제거
                let rt = tokio::runtime::Runtime::new();
                if let Ok(rt) = rt {
                    rt.block_on(async {
                        automation::shutdown_browser().await;
                    });
                }

                println!("[APP] Browser shutdown complete.");
            }
        });
}
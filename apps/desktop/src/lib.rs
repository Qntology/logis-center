// 🌟 [APP CORE] 모델 상주 · 저장소 · 업무 규칙 · 스케줄러 · 검색 도우미는 헤드리스 코어 crate(apps/core, logis-app)에 있습니다.
//    이 셸(Tauri 명령 · 창 · 브라우저 자동화)은 그 모듈들을 예전 이름으로 다시 내보내므로, 명령과 통합 테스트(tests/it)의
//    경로(crate::model · tauri_app_lib::store …)는 그대로입니다.
pub use logis_app::{
    js_templates, logic, model, models, parsers, prompts, scheduler, search_fusion, search_relation, stanza, store, utils,
};
pub use logis_app::{bias_schema, json_parse, nl_convert, parsing, score_dynamics, time_guide};
pub use logis_app::{chat_template, openai_types, position_embed, tokenizer};
// 검색 중 표시 · 앱 상태와 자연어 검색 본문(범위 필터 · dexie 계획 · 조건 그라운딩 · 질의 융합)은 코어에 있습니다.
pub use logis_app::{AppState, IS_SEARCHING};
pub use logis_app::query::ground_trade_conditions;
#[allow(unused_imports)]
use logis_app::query::*;
// 🌟 [COMMANDS] 명령 본문은 코어(logis_app::commands)에 있고, 아래 #[tauri::command] 는 상태 · Tauri 싱크를 넘기는 껍데기입니다.
//    Value 인자는 들어올 때, Value 결과는 나갈 때 키를 정렬합니다 (예전 serde_json 맵 순서, logis_app::json_order).
use logis_app::commands::ActiveTaskQuery;
mod automation;
// 🌟 [EVENT SINK] model · scheduler 는 tauri::AppHandle 대신 Sink 로 이벤트를 내보냅니다 (명령 진입점에서 sink::tauri 로 감쌈).
pub mod sink;

use tauri::{State, Manager, Listener, Emitter}; 
use tokio::sync::Mutex as TokioMutex;
use std::sync::RwLock; 
use once_cell::sync::Lazy; 
use store::{VectorStore, TradeDocument};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use serde_json::{Value, json};

// 브라우저가 실행되는 도중 상태를 방어하기 위한 락 추가
pub static IS_BROWSER_LAUNCHING: AtomicBool = AtomicBool::new(false);

pub static CURRENT_BROWSER_STATE: Lazy<RwLock<String>> = Lazy::new(|| RwLock::new("stopped".to_string()));
pub static LAST_BROWSER_STATE_CHANGE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

// 작업 문맥 · 화면 분류 · 마지막 진행 이벤트는 코어(스케줄러 · 로거 · 모델)가 쓰므로 logis-app 루트에 있습니다.
pub use logis_app::{ACTIVE_TASK_MEM, CURRENT_UI_CATEGORY, LATEST_PROGRESS_PAYLOAD};

#[tauri::command]
async fn get_active_task_context() -> Result<Value, String> {
    logis_app::commands::get_active_task_context().await.map(logis_app::json_order::sorted)
}

#[tauri::command]
async fn stop_current_extraction(state: State<'_, AppState>, task_id: Option<String>) -> Result<String, String> {
    logis_app::commands::stop_current_extraction(&state, task_id).await
}

#[tauri::command]
async fn delete_message(state: State<'_, AppState>, task_id: String) -> Result<String, String> {
    logis_app::commands::delete_message(&state, task_id).await
}

#[tauri::command]
async fn unload_model(state: State<'_, AppState>) -> Result<String, String> {
    logis_app::commands::unload_model(&state).await
}

#[tauri::command]
async fn translit_cache_respond(request_id: String, results: Vec<(String, String)>) -> Result<(), String> {
    logis_app::commands::translit_cache_respond(request_id, results).await
}

#[tauri::command]
async fn get_embedding_batch_for_translit(state: State<'_, AppState>, app_handle: tauri::AppHandle, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
    logis_app::commands::get_embedding_batch_for_translit(&state, crate::sink::tauri(&app_handle), texts).await
}

#[tauri::command]
async fn get_query_embedding(state: State<'_, AppState>, app_handle: tauri::AppHandle, text: String, device_preference: Option<String>) -> Result<Vec<f32>, String> {
    logis_app::commands::get_query_embedding(&state, crate::sink::tauri(&app_handle), text, device_preference).await
}

#[tauri::command]
async fn reindex_pending_embeddings(state: State<'_, AppState>, app_handle: tauri::AppHandle, limit: Option<usize>, device_preference: Option<String>, mode: Option<String>) -> Result<Value, String> {
    logis_app::commands::reindex_pending_embeddings(&state, crate::sink::tauri(&app_handle), limit, device_preference, mode).await.map(logis_app::json_order::sorted)
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
async fn summarize_image(state: State<'_, AppState>, _app_handle: tauri::AppHandle, image_path: String) -> Result<String, String> {
    logis_app::commands::summarize_image(&state, crate::sink::tauri(&_app_handle), image_path).await
}

#[tauri::command]
async fn search_documents(state: State<'_, AppState>, query: String, limit: usize, offset: usize, filter: Option<String>) -> Result<Vec<(String, String, f32)>, String> {
    logis_app::commands::search_documents(&state, query, limit, offset, filter).await
}

// =====================================================================
// 🌟 [QUERY ROUTER v4] 조건을 '스코프'와 '정밀 필터'로 물리적으로 분리합니다.
// ---------------------------------------------------------------------
//  기존 convert_conditions_to_sql 은 하나의 함수가 두 역할을 겸하면서
//  valid_cols 화이트리스트(7개) 밖의 조건을 전부 '버리고' 있었습니다.
//  그 손실을 메우려고 STAGE-3 이 A/FULL ~ E/TABLE-FALLBACK 5개 티어를 발행했고,
//  그래도 못 메운 부분은 tracking_sql 의 LIKE '%..%' 로 땜질했습니다.
//
//  v4 구조:
//    build_scope_filter → LanceDB : 봉투 컬럼만. 조건을 '절대' 버리지 않음(애초에 안 받음)
//    build_dexie_plan   → Dexie   : 도메인 조건 전량. 버려지는 조건이 0개
//
//  → 조건이 버려지지 않으므로 보험 티어(A~E)가 불필요해지고,
//    새 도메인 필드가 생겨도 이 두 함수는 수정할 필요가 없습니다.
// =====================================================================

#[tauri::command]
async fn get_all_documents(state: State<'_, AppState>, limit: usize, offset: usize, filter: Option<String>) -> Result<Vec<TradeDocument>, String> {
    logis_app::commands::get_all_documents(&state, limit, offset, filter).await
}

#[tauri::command]
async fn get_document(state: State<'_, AppState>, uuid: String) -> Result<Option<TradeDocument>, String> {
    logis_app::commands::get_document(&state, uuid).await
}

#[tauri::command]
async fn update_document(_state: State<'_, AppState>, _uuid: String, _json_data: String) -> Result<String, String> {
    logis_app::commands::update_document(&_state, _uuid, _json_data).await
}

#[tauri::command]
async fn delete_document(state: State<'_, AppState>, uuid: String) -> Result<String, String> {
    logis_app::commands::delete_document(&state, uuid).await
}

#[tauri::command]
async fn delete_documents(state: State<'_, AppState>, uuids: Vec<String>) -> Result<String, String> {
    logis_app::commands::delete_documents(&state, uuids).await
}

// 🌟 [QUERY] 자연어 검색 본문은 코어(logis_app::query::ai_search_complex)에 있습니다. 이 명령은 상태와 Tauri 싱크를 넘기는 껍데기입니다.
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
    logis_app::query::ai_search_complex(
        &state,
        crate::sink::tauri(&app_handle),
        task_id,
        query,
        language,
        device_preference,
        search_mode,
        cc,
        bcc,
        ref_id,
    )
    .await
    .map(logis_app::json_order::sorted)
}

#[tauri::command]
async fn check_query_intent(_state: State<'_, AppState>, _query: String) -> Result<String, String> {
    logis_app::commands::check_query_intent(&_state, _query).await
}

#[tauri::command]
async fn deep_research_command(state: State<'_, AppState>, app_handle: tauri::AppHandle, query: String, _doc_id: Option<String>, device_preference: Option<String>) -> Result<String, String> {
    logis_app::commands::deep_research_command(&state, crate::sink::tauri(&app_handle), query, _doc_id, device_preference).await
}

#[tauri::command]
async fn proxy_fetch(url: String, method: String, headers: std::collections::HashMap<String, String>, body: Option<Value>, session_params: Option<Value>) -> Result<Value, String> {
    logis_app::commands::proxy_fetch(url, method, headers, body.map(logis_app::json_order::sorted), session_params.map(logis_app::json_order::sorted)).await.map(logis_app::json_order::sorted)
}

#[tauri::command]
async fn check_active_task(_state: State<'_, AppState>, payload: ActiveTaskQuery) -> Result<bool, String> {
    logis_app::commands::check_active_task(&_state, payload).await
}

#[tauri::command]
async fn connect_with_seed(_target_ip: String, _seed: u64) -> Result<(), String> {
    // [DEPRECATED] UDP 방식은 더 이상 사용하지 않으며, 프론트엔드에서 
    // 직접 send_signal_offer(TCP)를 호출하도록 변경되었습니다.
    Ok(())
}

#[tauri::command]
async fn start_listener_command(app_handle: tauri::AppHandle, seed: u64) -> Result<(), String> {
    crate::utils::network::start_signal_listener(crate::sink::tauri(&app_handle), seed);
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
async fn initialize_hub(state: State<'_, AppState>, address: String, email: String, flag: String) -> Result<String, String> {
    logis_app::commands::initialize_hub(&state, address, email, flag).await
}

#[tauri::command]
async fn get_chat_messages(state: State<'_, AppState>, limit: usize, offset: usize, filter: Option<String>) -> Result<Vec<Value>, String> {
    logis_app::commands::get_chat_messages(&state, limit, offset, filter).await.map(|v| v.into_iter().map(logis_app::json_order::sorted).collect())
}

#[tauri::command]
async fn get_known_pages(state: State<'_, AppState>) -> Result<Vec<TradeDocument>, String> {
    logis_app::commands::get_known_pages(&state).await
}

#[tauri::command]
async fn get_known_users(state: State<'_, AppState>) -> Result<Vec<TradeDocument>, String> {
    logis_app::commands::get_known_users(&state).await
}

#[tauri::command]
async fn set_login_state(state: State<'_, AppState>, is_logged_in: bool, token: Option<String>) -> Result<String, String> {
    logis_app::commands::set_login_state(&state, is_logged_in, token).await
}

#[tauri::command]
async fn extract_html_from_current_tab() -> Result<String, String> {
    automation::extract_html_from_current_tab().await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_browser_status() -> Result<Value, String> {
    get_browser_status_body().await.map(logis_app::json_order::sorted)
}

async fn get_browser_status_body() -> Result<Value, String> {
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
    logis_app::commands::get_active_tasks(&state).await
}

#[tauri::command]
async fn get_task_logs(app_handle: tauri::AppHandle, task_id: String) -> Result<Vec<Value>, String> {
    logis_app::commands::get_task_logs(crate::sink::tauri(&app_handle), task_id).await.map(|v| v.into_iter().map(logis_app::json_order::sorted).collect())
}

#[tauri::command]
async fn upsert_items(state: State<'_, AppState>, items: Vec<Value>) -> Result<String, String> {
    logis_app::commands::upsert_items(&state, items.into_iter().map(logis_app::json_order::sorted).collect()).await
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
    logis_app::json_order::sorted(logis_app::commands::check_gpu_availability().await)
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
    check_model_status_body().await.map(logis_app::json_order::sorted)
}

async fn check_model_status_body() -> Result<serde_json::Value, String> {
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
    
    let qwen3_dir = base_path.join("Qwen3-0.6B-Instruct-gguf");
    let qwen3_5_dir = base_path.join("Qwen3.5-2B-Instruct-gguf");
    let granite_dir = base_path.join("granite-4.0-h-350m");
    let embed_dir = base_path.join("granite-embedding-97m-multilingual-r2");

    let siglip2_dir = base_path.join("siglip2-so400m-patch16-naflex");

    let mut status_map = serde_json::Map::new();
    status_map.insert("Qwen3".to_string(), serde_json::json!(has_valid_model(&qwen3_dir)));
    status_map.insert("Qwen3.5".to_string(), serde_json::json!(has_valid_model(&qwen3_5_dir)));
    status_map.insert("Granite".to_string(), serde_json::json!(has_valid_model(&granite_dir)));
    status_map.insert("Embedding".to_string(), serde_json::json!(has_valid_model(&embed_dir)));
    status_map.insert("SigLIP2".to_string(), serde_json::json!(has_valid_model(&siglip2_dir)));

    // 🌟 [STANZA] 변환본이 있는 12개 언어 (stanza-onnx-export format 2). vocab.json 과 그 파일 목록이 모두 있어야 설치로 봅니다.
    for (lang, code) in crate::stanza::UI_LANGS.iter() {
        let stanza_dir = base_path.join("stanza").join(code);
        status_map.insert(format!("stanza_{}", lang), serde_json::json!(crate::stanza::is_installed(&stanza_dir)));
    }

    Ok(serde_json::Value::Object(status_map))
}

#[tauri::command]
async fn get_lang_llm_status(language: Option<String>) -> Result<serde_json::Value, String> {
    logis_app::commands::get_lang_llm_status(language).await.map(logis_app::json_order::sorted)
}

#[tauri::command]
async fn delete_all_models() -> Result<String, String> {
    logis_app::commands::delete_all_models().await
}

#[tauri::command]
async fn reset_lancedb(state: State<'_, AppState>) -> Result<String, String> {
    logis_app::commands::reset_lancedb(&state).await
}

#[tauri::command]
async fn download_model(app_handle: tauri::AppHandle, model_name: String) -> Result<String, String> {
    logis_app::commands::download_model(crate::sink::tauri(&app_handle), model_name).await
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
fn app_context() -> tauri::Context {
    tauri::generate_context!()
}

// 로컬 네트워크 도우미는 코어(utils::network)에 있고, Tauri 명령 껍데기만 여기 둡니다 (명령 이름 · 인자는 예전 그대로).
#[tauri::command]
fn get_local_network_prefix() -> String {
    crate::utils::network::get_local_network_prefix()
}

#[tauri::command]
fn get_my_full_ip() -> String {
    crate::utils::network::get_my_full_ip()
}

pub fn run() {
    // 모델이 띄우는 비동기 작업(KV 슬롯 반환 · 굽기 · 지연 해제)도 예전처럼 Tauri 전역 런타임에서 돕니다.
    logis_llm::rt::set_spawner(|task| {
        tauri::async_runtime::spawn(task);
    });
    crate::model::lang_llm::register_loader(crate::models::qwen4_exp::registry::load_lang_runtime);
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
            // 저장소 루트에서 실행한 경우 (apps/desktop 에서 실행하면 위의 models)
            let src_models_dir2 = std::env::current_dir().unwrap_or_default().join("apps").join("desktop").join("models");
            
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
            crate::utils::paths::cleanup_temp_dirs();

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
                    // 🌟 [MODE MIGRATION] type 과 mode 가 어긋난 기존 문서를 1회 교정합니다.
                    //    대상이 0건이면 조회 1회로 끝나므로 매 시작마다 돌아도 부담이 없고,
                    //    교정된 문서는 embed 마커가 제거되어 reindex 가 자동으로 재인덱싱합니다.
                    let _ = s.migrate_mode_by_type().await;
                    // 🌟 [SDS LOAD] 점수 동역학 통계를 세션 시작 시 1회 불러옵니다.
                    //
                    //  ── 왜 여기인가 ──
                    //   VectorStore 초기화와 같은 블록이어야 '데이터 계층이 준비되는 시점'
                    //   이라는 의미가 코드 배치로 드러납니다.
                    //   이 시점에는 아직 로그인 전이라 팀이 ZERO_ADDRESS 기반일 수 있고,
                    //   initialize_hub 가 실제 팀을 확정하면 rebind_team 이 재바인딩합니다.
                    //
                    //  ── 파일이 없으면 ──
                    //   냉간 시작 로그만 남기고 빈 통계로 출발합니다.
                    //   ASE 는 관측 부족으로 None 을 돌려주므로 판정은 현행 그대로입니다.
                    let zero_team = crate::utils::hash::hash_id(
                        "0x0000000000000000000000000000000000000000"
                    );
                    crate::utils::score_dynamics::load(&zero_team);
                    
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
            let scheduler_handle = crate::sink::tauri(app.handle());
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
                        let _ = status_monitor_handle.emit("browser-status", logis_app::json_order::sorted(payload));
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

                if let Ok(payload_val) = serde_json::from_str::<serde_json::Value>(event.payload()).map(logis_app::json_order::sorted) {
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
                                data_json: logis_app::json_order::to_string_sorted(&payload_val), created_at: now, updated_at: now, status: 10,
                            };
                            let msg_text = format!("Task Started: {}", payload_val.get("link").and_then(|v| v.as_str()).unwrap_or("Unknown URL"));
                            let _ = db.add_message(
                                &task.id, "system_task", &msg_text, Some(&task.id), Some(10),
                                Some(&task.cc), Some(&task.bcc), Some(&task.r#ref),
                                Some(&task.from), Some(&task.to), Some("talk"), None
                            ).await;
                            let _ = db.add_task(task.clone()).await;
                            let _ = app_handle.emit("task-db-registered", logis_app::json_order::sorted(json!({
                                "task_id": task.id,
                                "status": task.status,
                                "created_at": task.created_at,
                                "text": msg_text
                            })));
                            crate::utils::sync_utils::notify_new_task();
                        } else {
                            // 🌟 [TASK DROP DIAGNOSTIC]
                            //  ── 무엇이 문제였나 ──
                            //   store 가 None 일 때 이 리스너는 로그 없이 태스크를 버렸습니다.
                            //   사용자는 이미지가 큐에 들어간 것처럼 보이지만,
                            //   실제로는 DB 에 아무것도 저장되지 않은 상태였습니다.
                            //   이 로그가 찍히면 reset_lancedb 의 재주입이 실패했다는 증거입니다.
                            let task_id = payload_val.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
                            let task_type = payload_val.get("type").and_then(|v| v.as_str()).unwrap_or("unknown");
                            println!(
                                "[TASK DROP] 🚨 new-task-from-browser 가 수신했으나 Store 가 None 입니다. \
                                 태스크 '{}' (type='{}') 가 DB 에 저장되지 않았습니다. \
                                 팩토리 리셋 후 reset_lancedb 가 Store 를 재주입했는지 확인하세요.",
                                task_id, task_type
                            );
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
            save_mobile_temp_file, get_local_network_prefix, get_my_full_ip, connect_with_seed, start_listener_command, send_signal_offer, submit_signal_answer,
            get_active_task_context, check_model_status, download_model, delete_all_models, reset_lancedb,
            get_query_embedding, reindex_pending_embeddings,
            translit_cache_respond, get_embedding_batch_for_translit,
            get_lang_llm_status
        ])
        .build(app_context())
        .expect("error while building tauri application")
        .run(|_app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                println!("[APP] Application exiting. Shutting down browser...");
                // 🌟 [SDS FLUSH] 종료 경로에서도 관측을 확정합니다.
                //    언로드를 거치지 않고 창을 닫는 경로가 존재하므로
                //    두 지점 모두에 flush 가 있어야 관측이 유실되지 않습니다.
                //    flush 는 dirty 일 때만 파일을 쓰므로 중복 호출이 무해합니다.
                println!("[APP] {}", crate::utils::score_dynamics::report());
                crate::utils::score_dynamics::flush();
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

                // 3. 🌟 [ROCM-EXIT] Windows ROCm: rocBLAS/HIP DLL 해제 중 멈춤을 피해 여기서 바로 끝냅니다.
                crate::utils::device_utils::rocm_exit_now_if_used(0);
            }
        });
}

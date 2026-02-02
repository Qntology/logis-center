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

use tauri::{Manager, State};
use tokio::sync::Mutex;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use serde_json::{json, Value};
use crate::model::LogisModel;
use crate::store::VectorStore;

pub struct AppState {
    pub store: Arc<Mutex<Option<VectorStore>>>,
    pub model: Arc<Mutex<Option<LogisModel>>>,
    pub cancellation_token: Arc<AtomicBool>,
}

#[tauri::command]
pub async fn search_documents_cmd(state: State<'_, AppState>, query: String) -> Result<Vec<Value>, String> {
    let mut model_guard = state.model.lock().await;
    if model_guard.is_none() { *model_guard = Some(LogisModel::new(None).await.map_err(|e| e.to_string())?); }
    let model = model_guard.as_ref().unwrap();
    let emb = model.get_embedding(query).await.map_err(|e| e.to_string())?;
    Ok(vec![json!({ "vector": emb })])
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .manage(AppState {
            store: Arc::new(Mutex::new(None)),
            model: Arc::new(Mutex::new(None)),
            cancellation_token: Arc::new(AtomicBool::new(false)),
        })
        .invoke_handler(tauri::generate_handler![search_documents_cmd])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
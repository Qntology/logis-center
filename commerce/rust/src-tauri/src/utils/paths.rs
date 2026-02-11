use std::path::PathBuf;
use tauri::{AppHandle, Manager};
use std::fs;

pub fn get_app_tmp_root(app: Option<&AppHandle>) -> PathBuf {
    // [TAURI-2.0-STANDARD] Use the new .path() API for data directory resolution
    if let Some(app_handle) = app {
        // Tauri 2.0 uses .path().app_local_data_dir()
        if let Ok(data_dir) = app_handle.path().app_local_data_dir() {
            if !data_dir.exists() {
                let _ = fs::create_dir_all(&data_dir);
            }
            return data_dir;
        }
    }

    // [FALLBACK] For CLI development or when handle is missing, use project-relative tmp folder
    let path = PathBuf::from("tmp");
    if !path.exists() {
        let _ = fs::create_dir_all(&path);
    }
    path
}

pub fn get_kv_dir(app: Option<&AppHandle>) -> PathBuf {
    let path = get_app_tmp_root(app).join("kv");
    if !path.exists() { let _ = fs::create_dir_all(&path); }
    path
}

pub fn get_task_data_dir(app: Option<&AppHandle>) -> PathBuf {
    let path = get_app_tmp_root(app).join("task_data");
    if !path.exists() { let _ = fs::create_dir_all(&path); }
    path
}

pub fn get_task_specific_dir(app: Option<&AppHandle>, task_id: &str) -> PathBuf {
    let path = get_task_data_dir(app).join(task_id);
    if !path.exists() { let _ = fs::create_dir_all(&path); }
    path
}

pub fn get_logs_dir(app: Option<&AppHandle>) -> PathBuf {
    let path = get_app_tmp_root(app).join("logs");
    if !path.exists() { let _ = fs::create_dir_all(&path); }
    path
}

pub fn get_pug_logs_dir(app: Option<&AppHandle>, task_id: &str) -> PathBuf {
    let path = get_logs_dir(app).join("pug").join(task_id);
    if !path.exists() { let _ = fs::create_dir_all(&path); }
    path
}

pub fn get_task_log_file(app: Option<&AppHandle>, task_id: &str) -> PathBuf {
    let path = get_logs_dir(app).join("tasks");
    if !path.exists() { let _ = fs::create_dir_all(&path); }
    path.join(format!("{}.jsonl", task_id))
}

pub fn get_stop_signal_file(app: Option<&AppHandle>) -> PathBuf {
    get_app_tmp_root(app).join("EXTRACTION_STOPPED")
}

/// Initialize all necessary directories
pub fn init_directories(app: Option<&AppHandle>) {
    let _ = get_kv_dir(app);
    let _ = get_task_data_dir(app);
    let _ = get_logs_dir(app);
}

/// Cleanup temporary directories (called on startup or shutdown)
/// [KEEP-KV] We do NOT remove the kv directory here to allow context reuse across sessions.
pub fn cleanup_temp_dirs(app: Option<&AppHandle>) {
    let data = get_task_data_dir(app);
    let logs = get_logs_dir(app);
    
    let _ = fs::remove_dir_all(&data);
    let _ = fs::remove_dir_all(&logs);
    
    let _ = fs::create_dir_all(&data);
    let _ = fs::create_dir_all(&logs);
}

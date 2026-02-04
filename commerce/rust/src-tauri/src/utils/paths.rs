use std::path::PathBuf;
use tauri::AppHandle;
use std::fs;

pub fn get_app_tmp_root(_app: Option<&AppHandle>) -> PathBuf {
    // [STRICT] All temporary files must be collected in the "tmp" folder in the project root
    let path = PathBuf::from("tmp");
    if !path.exists() {
        let _ = fs::create_dir_all(&path);
    }
    path
}

pub fn get_kv_dir(app: Option<&AppHandle>, sub_folder: Option<&str>) -> PathBuf {
    let mut path = get_app_tmp_root(app).join("kv");
    if let Some(sub) = sub_folder {
        path = path.join(sub);
    }
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

pub fn get_stop_signal_file() -> PathBuf {
    PathBuf::from("tmp").join("EXTRACTION_STOPPED")
}

/// Initialize all necessary directories
pub fn init_directories(app: Option<&AppHandle>) {
    let _ = get_kv_dir(app, None);
    let _ = get_task_data_dir(app);
    let _ = get_logs_dir(app);
}

/// Cleanup temporary directories (called on startup or shutdown)
pub fn cleanup_temp_dirs(app: Option<&AppHandle>) {
    let data = get_task_data_dir(app);
    let logs = get_logs_dir(app);
    
    let _ = fs::remove_dir_all(&data);
    let _ = fs::remove_dir_all(&logs);
    
    let _ = fs::create_dir_all(&data);
    let _ = fs::create_dir_all(&logs);
    // KV directory is preserved to keep baked cache
}

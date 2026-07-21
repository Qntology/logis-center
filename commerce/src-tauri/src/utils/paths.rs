use std::path::PathBuf;
use tauri::AppHandle;
use std::fs;

pub fn get_app_tmp_root(_app: Option<&AppHandle>) -> PathBuf {
    // [STRICT] All temporary files must be collected in the "tmp" folder in the project root
    let path = crate::utils::get_app_dir().join("tmp");
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

pub fn get_stop_signal_file() -> PathBuf {
    get_app_tmp_root(None).join("EXTRACTION_STOPPED")
}

/// Initialize all necessary directories
pub fn init_directories(app: Option<&AppHandle>) {
    let _ = get_kv_dir(app);
    let _ = get_task_data_dir(app);
    let _ = get_logs_dir(app);
}

/// Cleanup temporary directories (called on startup or shutdown)
pub fn cleanup_temp_dirs(app: Option<&AppHandle>) {
    let kv = get_kv_dir(app);
    let data = get_task_data_dir(app);
    let logs = get_logs_dir(app);
    
    let _ = fs::remove_dir_all(&kv);
    let _ = fs::remove_dir_all(&data);
    let _ = fs::remove_dir_all(&logs);
    
    let _ = fs::create_dir_all(&kv);
    let _ = fs::create_dir_all(&data);
    let _ = fs::create_dir_all(&logs);
}

/// Copy model configuration files (excluding .gguf) from source to destination
pub fn copy_model_configs(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    if !src.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let dest_path = dst.join(entry.file_name());
        
        if path.is_dir() {
            copy_model_configs(&path, &dest_path)?;
        } else {
            let file_name = entry.file_name().to_string_lossy().to_string();
            // .gguf 파일은 용량이 크므로 제외 (config, json, jinja, txt 등만 복사)
            if !file_name.ends_with(".gguf") && !dest_path.exists() {
                let _ = std::fs::copy(&path, &dest_path);
            }
        }
    }
    Ok(())
}

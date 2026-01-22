use std::path::{PathBuf, Path};
use tauri::{AppHandle, Manager};
use std::fs;

pub fn get_app_tmp_root(app: Option<&AppHandle>) -> PathBuf {
    if let Some(app) = app {
        // Production: Use OS-standard Cache directory
        // Windows: C:\Users\<User>\AppData\Local\<BundleId>\cache
        // macOS: /Users/<User>/Library/Caches/<BundleId>
        // Linux: /home/<User>/.cache/<BundleId>
        if let Ok(cache_dir) = app.path().app_cache_dir() {
            return cache_dir;
        }
    }
    
    // Development or Fallback: Use "tmp" folder in project root
    let dev_path = PathBuf::from("tmp");
    if !dev_path.exists() {
        let _ = fs::create_dir_all(&dev_path);
    }
    dev_path
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

pub fn get_logs_dir(app: Option<&AppHandle>) -> PathBuf {
    let path = get_app_tmp_root(app).join("logs");
    if !path.exists() { let _ = fs::create_dir_all(&path); }
    path
}

pub fn get_pug_logs_dir(app: Option<&AppHandle>) -> PathBuf {
    let path = get_logs_dir(app).join("pug");
    if !path.exists() { let _ = fs::create_dir_all(&path); }
    path
}

/// Initialize all necessary directories
pub fn init_directories(app: Option<&AppHandle>) {
    let _ = get_kv_dir(app);
    let _ = get_task_data_dir(app);
    let _ = get_pug_logs_dir(app);
}

/// Cleanup temporary directories (called on startup or shutdown)
pub fn cleanup_temp_dirs(app: Option<&AppHandle>) {
    let kv = get_kv_dir(app);
    let data = get_task_data_dir(app);
    
    let _ = fs::remove_dir_all(&kv);
    let _ = fs::remove_dir_all(&data);
    
    let _ = fs::create_dir_all(&kv);
    let _ = fs::create_dir_all(&data);
}

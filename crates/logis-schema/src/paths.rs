//! 앱 데이터 폴더와 그 아래 작업 폴더 (Tauri 없이 정함: LOGIS_APP_DIR 또는 OS 의 로컬 데이터 폴더/logis-center).
//! 예전 앱 utils/paths.rs 를 통째로 옮겼고, 모델 생성 루프가 보는 중단 신호(예전 utils/sync_utils.rs)도 여기 둡니다.

use std::fs;
use std::path::PathBuf;

/// 🌟 [misc_utils.rs에서 이동] 앱 데이터 루트 디렉토리 반환
///   LOGIS_APP_DIR 이 있으면 그 폴더를 씁니다. 합성 데이터 평가가 사용자의 DB · SDS 원장을 건드리지 않게 합니다.
pub fn get_app_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("LOGIS_APP_DIR").filter(|v| !v.is_empty()) {
        let path = std::path::PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&path);
        return path;
    }
    if let Some(mut path) = dirs::data_local_dir() {
        path.push("logis-center");
        let _ = std::fs::create_dir_all(&path);
        path
    } else {
        let path = std::path::PathBuf::from("logis-center-data");
        let _ = std::fs::create_dir_all(&path);
        path
    }
}

pub fn get_app_tmp_root() -> PathBuf {
    // [STRICT] All temporary files must be collected in the "tmp" folder in the project root
    let path = get_app_dir().join("tmp");
    if !path.exists() {
        let _ = fs::create_dir_all(&path);
    }
    path
}

pub fn get_kv_dir() -> PathBuf {
    let path = get_app_tmp_root().join("kv");
    if !path.exists() { let _ = fs::create_dir_all(&path); }
    path
}

pub fn get_task_data_dir() -> PathBuf {
    let path = get_app_tmp_root().join("task_data");
    if !path.exists() { let _ = fs::create_dir_all(&path); }
    path
}

pub fn get_task_specific_dir(task_id: &str) -> PathBuf {
    let path = get_task_data_dir().join(task_id);
    if !path.exists() { let _ = fs::create_dir_all(&path); }
    path
}

pub fn get_logs_dir() -> PathBuf {
    let path = get_app_tmp_root().join("logs");
    if !path.exists() { let _ = fs::create_dir_all(&path); }
    path
}

pub fn get_pug_logs_dir(task_id: &str) -> PathBuf {
    let path = get_logs_dir().join("pug").join(task_id);
    if !path.exists() { let _ = fs::create_dir_all(&path); }
    path
}

pub fn get_task_log_file(task_id: &str) -> PathBuf {
    let path = get_logs_dir().join("tasks");
    if !path.exists() { let _ = fs::create_dir_all(&path); }
    path.join(format!("{}.jsonl", task_id))
}

pub fn get_stop_signal_file() -> PathBuf {
    get_app_tmp_root().join("EXTRACTION_STOPPED")
}

// --- GLOBAL EXTRACTION CONTROL (예전 앱 utils/sync_utils.rs, 그 전엔 misc_utils.rs) ---
pub fn is_extraction_stopped() -> bool {
    get_stop_signal_file().exists()
}

pub fn set_extraction_stop_signal(stopped: bool) {
    let file = get_stop_signal_file();
    if stopped {
        let _ = std::fs::File::create(file);
    } else {
        let _ = std::fs::remove_file(file);
    }
}

/// Initialize all necessary directories
pub fn init_directories() {
    let _ = get_kv_dir();
    let _ = get_task_data_dir();
    let _ = get_logs_dir();
}

/// Cleanup temporary directories (called on startup or shutdown)
pub fn cleanup_temp_dirs() {
    let kv = get_kv_dir();
    let data = get_task_data_dir();
    let logs = get_logs_dir();

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

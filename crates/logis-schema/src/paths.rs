//! 앱 데이터 폴더 (Tauri 없이 정함: LOGIS_APP_DIR 또는 OS 의 로컬 데이터 폴더/logis-center). 예전 utils/paths.rs 에서 옮김.

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

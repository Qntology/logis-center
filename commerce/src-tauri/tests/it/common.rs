//! 통합 테스트 공용 헬퍼
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// 저장소 루트의 SyntheticData/trading 경로
pub fn synthetic_trading_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../SyntheticData/trading")
}

/// 테스트용 모델 루트 디렉터리 (`.../logis-center/models`).
///
/// 1. `LOGIS_MODELS_DIR` 가 있으면 그 경로 (목업 모델 · CI 캐시 등. tests/fixtures/granite_mock 참고)
/// 2. 없으면 앱과 같은 OS 별 데이터 경로(`utils::get_app_dir` 와 같은 `dirs::data_local_dir()`):
///    Windows `%LOCALAPPDATA%` · macOS `~/Library/Application Support` · Linux `$XDG_DATA_HOME` 또는 `~/.local/share`
pub fn models_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("LOGIS_MODELS_DIR").filter(|v| !v.is_empty()) {
        return PathBuf::from(p);
    }
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("logis-center-data"))
        .join("logis-center")
        .join("models")
}

/// `models_dir()/<name>` 가 있으면 그 경로, 없으면 어디를 봤는지 알려 주는 오류 문자열.
pub fn model_subdir(name: &str, must_contain: &str) -> Result<PathBuf, String> {
    let p = models_dir().join(name);
    if p.join(must_contain).exists() {
        Ok(p)
    } else {
        Err(format!("{} 가 없습니다 (LOGIS_MODELS_DIR 또는 OS 데이터 경로)", p.join(must_contain).display()))
    }
}

/// 테스트가 모델 없이 건너뛸 때: `LOGIS_REQUIRE_MODELS=1` 이면 실패로 처리합니다
/// (CI 에서 '모델이 없어 건너뜀' 이 통과로 세어지지 않게).
pub fn skip_or_fail(reason: &str) {
    if std::env::var("LOGIS_REQUIRE_MODELS").map(|v| v == "1").unwrap_or(false) {
        panic!("LOGIS_REQUIRE_MODELS=1 인데 건너뛰려 했습니다: {}", reason);
    }
    println!("{} — 건너뜁니다.", reason);
}

/// 합성 PDF 목록 (정렬)
pub fn synthetic_pdfs() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(synthetic_trading_root().join("output_pdfs"))
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().map_or(false, |e| e.eq_ignore_ascii_case("pdf")))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// 테스트마다 격리된 임시 디렉터리 (Drop 시 삭제)
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(name: &str) -> Self {
        let d = std::env::temp_dir().join(format!(
            "logis_it_{}_{}_{:?}",
            name,
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        TempDir(d)
    }
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// 결정적 바이트 패턴 (0 이 아닌 다양한 값이 섞이도록)
pub fn pattern(n: usize, salt: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i.wrapping_mul(31) ^ (i >> 8) ^ salt as usize) as u8)
        .collect()
}

/// 공백 정규화 + 소문자
pub fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

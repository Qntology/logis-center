//! 통합 테스트 공용 헬퍼
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// 저장소 루트의 SyntheticData/trading 경로
pub fn synthetic_trading_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../SyntheticData/trading")
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

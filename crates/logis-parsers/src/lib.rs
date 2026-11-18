//! logis-center 의 오프라인 문서 리더 crate (모델 · DB · Tauri 없음).
//!
//! - [`documents`]: Excel(xlsx · xls) · Word(docx) · HWP(hwp · hwpx) · CSV · PDF 를 글 · 쪽으로 (표는 `헤더: 값` 문맥)
//! - [`compression`]: JSON 을 gzip 으로 묶고 풀기
//!
//! 앱은 예전 경로(`crate::parsers` · `crate::utils::compression`)에서 재수출합니다.

pub mod compression;
pub mod documents;

pub use documents::*;

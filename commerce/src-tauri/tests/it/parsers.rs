//! 문서 파서 검증 — SyntheticData/trading (YAML → PDF 합성 무역서류 45종)
//!
//! input_yamls 의 스칼라 값을 정답(ground truth)으로 삼아
//! PDF 추출 텍스트에 값이 보존되는지 확인합니다.

use crate::common::{norm, synthetic_pdfs, synthetic_trading_root};
use tauri_app_lib::parsers::{extract_document_pages, extract_document_text};

/// PDF 생성기(batch_yaml_to_pdf.py)는 YAML 을 파이썬 값으로 읽은 뒤 str() 로 출력하므로
/// 따옴표 없는 실수 리터럴은 파이썬 표기로 바뀝니다 ("31.40" → "31.4", "0.00" → "0.0").
fn python_float_text(v: &str) -> String {
    let is_plain_float = v.contains('.')
        && v.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '-')
        && v.parse::<f64>().is_ok();
    if !is_plain_float {
        return v.to_string();
    }
    let s = format!("{}", v.parse::<f64>().unwrap());
    if s.contains('.') { s } else { format!("{s}.0") }
}

/// `key: value` / `- key: value` 줄에서 의미 있는 스칼라 값만 뽑습니다.
fn yaml_scalars(yaml: &str) -> Vec<String> {
    let re = regex::Regex::new(r"^\s*(?:-\s+)?[A-Za-z0-9_]+:\s+(.+?)\s*$").unwrap();
    yaml.lines()
        .filter_map(|l| {
            re.captures(l).map(|c| {
                let raw = &c[1];
                let quoted = raw.starts_with('\'') || raw.starts_with('"');
                let v = raw.trim_matches(|ch| ch == '\'' || ch == '"');
                if quoted { v.to_string() } else { python_float_text(v) }
            })
        })
        .filter(|v| v.len() >= 4 && !v.starts_with('|') && !v.starts_with('>') && !v.starts_with('['))
        .collect()
}

#[test]
fn all_45_synthetic_pdfs_present() {
    assert_eq!(synthetic_pdfs().len(), 45, "expected 45 synthetic trading PDFs");
}

#[test]
fn every_synthetic_pdf_extracts_pages_and_text() {
    for p in synthetic_pdfs() {
        let ps = p.to_str().unwrap();
        let pages = extract_document_pages(ps).unwrap_or_else(|e| panic!("{p:?}: {e}"));
        assert!(!pages.is_empty(), "{p:?}: no pages");
        assert!(pages.iter().any(|pg| pg.len() > 50), "{p:?}: no substantive page text");

        let whole = extract_document_text(ps).unwrap_or_else(|e| panic!("{p:?}: {e}"));
        assert!(norm(&whole).contains("document_type"), "{p:?}: whole-text lost header");
        assert!(norm(&pages.join(" ")).contains("document_type"), "{p:?}: paged-text lost header");
    }
}

#[test]
fn yaml_ground_truth_values_survive_pdf_extraction() {
    let mut worst = (1.0f64, String::new());
    let mut checked = 0;
    for p in synthetic_pdfs() {
        let stem = p.file_stem().unwrap().to_str().unwrap().to_string();
        let yaml_path = synthetic_trading_root().join("input_yamls").join(format!("{stem}.yaml"));
        let Ok(yaml) = std::fs::read_to_string(&yaml_path) else { continue };
        let values = yaml_scalars(&yaml);
        assert!(!values.is_empty(), "{stem}: no scalars parsed from yaml");

        let text = norm(&extract_document_pages(p.to_str().unwrap()).unwrap().join(" "));
        let missing: Vec<&String> = values.iter().filter(|v| !text.contains(&norm(v))).collect();
        let ratio = 1.0 - missing.len() as f64 / values.len() as f64;
        if ratio < worst.0 {
            worst = (ratio, format!("{stem} missing {:?}", &missing[..missing.len().min(5)]));
        }

        // 문서 종류 / 문서 코드 같은 핵심 식별자는 반드시 있어야 한다
        for key in ["document_type:", "document_code:"] {
            if let Some(l) = yaml.lines().find(|l| l.starts_with(key)) {
                let v = l[key.len()..].trim().trim_matches('\'');
                assert!(text.contains(&norm(v)), "{stem}: {key} '{v}' not found");
            }
        }
        checked += 1;
    }
    assert_eq!(checked, 45);
    assert!(worst.0 >= 0.95, "value recall too low: {:.3} ({})", worst.0, worst.1);
}

#[test]
fn missing_or_unknown_files_error() {
    assert!(extract_document_pages("/nonexistent/x.pdf").is_err());
}

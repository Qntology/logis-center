//! Tests for the production GGUF loader (`oxibonsai_model::gguf_loader`).
//!
//! All tests use either in-memory construction (no real GGUF file needed)
//! or deliberately non-existent paths to exercise error paths.

use oxibonsai_model::gguf_loader::{
    estimate_memory_bytes, fits_in_budget, load_tensor_metadata, validate_gguf_file, LoadConfig,
    LoadError, LoadStats, TensorChunkIter, TensorEntry,
};

// ─────────────────────────────────────────────────────────────────────────────
// TensorEntry — element_count
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tensor_entry_element_count() {
    let entry = TensorEntry {
        name: "test".to_string(),
        shape: vec![2, 3, 4],
        quant_type_id: 0,
        offset: 0,
        size_bytes: 96,
    };
    assert_eq!(entry.element_count(), 24, "2×3×4 should be 24 elements");
}

#[test]
fn tensor_entry_element_count_flat() {
    let entry = TensorEntry {
        name: "flat".to_string(),
        shape: vec![1024],
        quant_type_id: 0,
        offset: 0,
        size_bytes: 4096,
    };
    assert_eq!(entry.element_count(), 1024);
}

// ─────────────────────────────────────────────────────────────────────────────
// TensorEntry — quant_name
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tensor_entry_quant_name_f32() {
    let entry = TensorEntry {
        name: "w".to_string(),
        shape: vec![4],
        quant_type_id: 0, // F32
        offset: 0,
        size_bytes: 16,
    };
    assert_eq!(entry.quant_name(), "F32");
}

#[test]
fn tensor_entry_quant_name_q1_0_g128() {
    let entry = TensorEntry {
        name: "w".to_string(),
        shape: vec![128],
        quant_type_id: 41,
        offset: 0,
        size_bytes: 18,
    };
    assert_eq!(entry.quant_name(), "Q1_0_g128");
}

#[test]
fn tensor_entry_quant_name_unknown() {
    let entry = TensorEntry {
        name: "w".to_string(),
        shape: vec![4],
        quant_type_id: 9999,
        offset: 0,
        size_bytes: 4,
    };
    assert_eq!(entry.quant_name(), "UNKNOWN");
}

// ─────────────────────────────────────────────────────────────────────────────
// TensorEntry — is_known_quant
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tensor_entry_is_known_quant_q1() {
    let entry = TensorEntry {
        name: "w".to_string(),
        shape: vec![128],
        quant_type_id: 41,
        offset: 0,
        size_bytes: 18,
    };
    assert!(
        entry.is_known_quant(),
        "type id 41 (Q1_0_g128) should be known"
    );
}

#[test]
fn tensor_entry_is_known_quant_f16() {
    let entry = TensorEntry {
        name: "w".to_string(),
        shape: vec![4],
        quant_type_id: 1, // F16
        offset: 0,
        size_bytes: 8,
    };
    assert!(entry.is_known_quant());
}

#[test]
fn tensor_entry_is_unknown_quant() {
    let entry = TensorEntry {
        name: "w".to_string(),
        shape: vec![4],
        quant_type_id: 9999,
        offset: 0,
        size_bytes: 4,
    };
    assert!(!entry.is_known_quant(), "type id 9999 should be unknown");
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadConfig — defaults
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn load_config_default_values() {
    let cfg = LoadConfig::default();
    assert!(
        cfg.max_memory_bytes.is_none(),
        "default budget should be unlimited"
    );
    assert!(
        !cfg.validate_checksums,
        "checksum validation off by default"
    );
    assert!(
        cfg.allow_unknown_quant_types,
        "should allow unknown quant types by default"
    );
    assert!(
        cfg.streaming_chunk_size > 0,
        "streaming chunk size must be positive"
    );
    assert!(!cfg.strict_version, "strict version disabled by default");
}

#[test]
fn load_config_allow_unknown_quants_settable() {
    let cfg = LoadConfig {
        allow_unknown_quant_types: false,
        ..Default::default()
    };
    assert!(!cfg.allow_unknown_quant_types);
}

#[test]
fn load_config_memory_budget_settable() {
    let cfg = LoadConfig {
        max_memory_bytes: Some(1024 * 1024 * 1024),
        ..Default::default()
    };
    assert_eq!(cfg.max_memory_bytes, Some(1_073_741_824));
}

// ─────────────────────────────────────────────────────────────────────────────
// Functions with non-existent paths → I/O errors
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn estimate_memory_nonexistent_path() {
    let path = std::env::temp_dir().join("nonexistent_file_that_does_not_exist.gguf");
    let result = estimate_memory_bytes(&path);
    assert!(result.is_err(), "should error for non-existent file");
    assert!(
        matches!(result.unwrap_err(), LoadError::Io(_)),
        "error should be an I/O error"
    );
}

#[test]
fn fits_in_budget_nonexistent_path() {
    let path = std::env::temp_dir().join("no_such_file_oxibonsai.gguf");
    let result = fits_in_budget(&path, 1024 * 1024);
    assert!(result.is_err(), "should error for non-existent file");
    assert!(matches!(result.unwrap_err(), LoadError::Io(_)));
}

#[test]
fn load_tensor_metadata_nonexistent_path() {
    let path = std::env::temp_dir().join("oxibonsai_no_file.gguf");
    let result = load_tensor_metadata(&path);
    assert!(result.is_err(), "should error for non-existent file");
    assert!(matches!(result.unwrap_err(), LoadError::Io(_)));
}

#[test]
fn validate_gguf_nonexistent_path() {
    let path = std::env::temp_dir().join("validate_no_file.gguf");
    let result = validate_gguf_file(&path);
    assert!(result.is_err(), "should error for non-existent file");
    assert!(matches!(result.unwrap_err(), LoadError::Io(_)));
}

// ─────────────────────────────────────────────────────────────────────────────
// TensorChunkIter
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tensor_chunk_iter_basic() {
    // 10 bytes, chunk_size=3 → chunks of [3, 3, 3, 1] = 4 chunks
    let data: Vec<u8> = (0u8..10).collect();
    let iter = TensorChunkIter::new(data, 3);
    assert_eq!(iter.total_chunks(), 4, "ceil(10/3)=4 chunks expected");
}

#[test]
fn tensor_chunk_iter_exact_multiple() {
    // 9 bytes, chunk_size=3 → 3 full chunks
    let data: Vec<u8> = (0u8..9).collect();
    let iter = TensorChunkIter::new(data, 3);
    assert_eq!(iter.total_chunks(), 3);
}

#[test]
fn tensor_chunk_iter_total_chunks_matches_consumed() {
    let data: Vec<u8> = (0u8..100).collect();
    let expected_total = TensorChunkIter::new(data.clone(), 32).total_chunks();
    let consumed = TensorChunkIter::new(data, 32).count();
    assert_eq!(
        consumed, expected_total,
        "total_chunks should match actual iteration count"
    );
}

#[test]
fn tensor_chunk_iter_bytes_remaining_decreases() {
    let data: Vec<u8> = vec![0u8; 10];
    let mut iter = TensorChunkIter::new(data, 4);

    let r0 = iter.bytes_remaining();
    assert_eq!(r0, 10, "should start with 10 bytes remaining");

    iter.next().expect("first chunk should exist");
    let r1 = iter.bytes_remaining();
    assert!(r1 < r0, "bytes_remaining should decrease after first chunk");

    iter.next().expect("second chunk should exist");
    let r2 = iter.bytes_remaining();
    assert!(
        r2 < r1,
        "bytes_remaining should decrease after second chunk"
    );
}

#[test]
fn tensor_chunk_iter_chunk_contents_correct() {
    let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7];
    let mut iter = TensorChunkIter::new(data, 3);

    let first = iter.next().expect("first chunk");
    assert_eq!(first, vec![1, 2, 3]);

    let second = iter.next().expect("second chunk");
    assert_eq!(second, vec![4, 5, 6]);

    let third = iter.next().expect("third (partial) chunk");
    assert_eq!(third, vec![7]);

    assert!(iter.next().is_none(), "iterator should be exhausted");
}

#[test]
fn tensor_chunk_iter_single_chunk() {
    let data: Vec<u8> = vec![42u8; 5];
    let mut iter = TensorChunkIter::new(data, 100);
    assert_eq!(iter.total_chunks(), 1);
    let chunk = iter.next().expect("single chunk");
    assert_eq!(chunk.len(), 5);
    assert!(iter.next().is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadStats — constructibility
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn load_stats_default_constructible() {
    let stats = LoadStats::default();
    assert_eq!(stats.tensors_loaded, 0);
    assert_eq!(stats.bytes_loaded, 0);
    assert_eq!(stats.skipped_tensors, 0);
    assert_eq!(stats.load_time_ms, 0);
    assert_eq!(stats.peak_memory_bytes, 0);
    assert!(stats.validation_warnings.is_empty());
}

#[test]
fn load_stats_fields_settable() {
    let stats = LoadStats {
        tensors_loaded: 42,
        bytes_loaded: 1_000_000,
        skipped_tensors: 3,
        load_time_ms: 127,
        peak_memory_bytes: 512 * 1024 * 1024,
        validation_warnings: vec!["test warning".to_string()],
    };
    assert_eq!(stats.tensors_loaded, 42);
    assert_eq!(stats.skipped_tensors, 3);
    assert_eq!(stats.validation_warnings.len(), 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadError — display
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn load_error_budget_exceeded_displays() {
    let err = LoadError::MemoryBudgetExceeded {
        need: 2_000,
        budget: 1_000,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("2000"),
        "error message should include 'need' bytes"
    );
    assert!(
        msg.contains("1000"),
        "error message should include 'budget' bytes"
    );
}

#[test]
fn load_error_unsupported_version_displays() {
    let err = LoadError::UnsupportedVersion(99);
    let msg = err.to_string();
    assert!(
        msg.contains("99"),
        "error message should include the version number"
    );
}

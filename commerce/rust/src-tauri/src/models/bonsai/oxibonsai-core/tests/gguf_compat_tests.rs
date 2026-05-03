//! Integration tests for the GGUF forward-compatibility layer.
//!
//! Tests cover [`GgufVersion`], [`ExtendedQuantType`], [`GgufCompatReport`],
//! [`check_gguf_header`], and [`build_compat_report`].

use oxibonsai_core::{
    build_compat_report, check_gguf_header, CompatError, ExtendedQuantType, GgufCompatReport,
    GgufVersion,
};

// ─────────────────────────────────────────────────────────────────────────────
// Test 1 — GgufVersion::from_u32 with valid inputs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn gguf_version_from_u32_valid() {
    assert_eq!(GgufVersion::from_u32(1), Some(GgufVersion::V1));
    assert_eq!(GgufVersion::from_u32(2), Some(GgufVersion::V2));
    assert_eq!(GgufVersion::from_u32(3), Some(GgufVersion::V3));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2 — GgufVersion::from_u32 with invalid inputs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn gguf_version_from_u32_invalid() {
    assert_eq!(GgufVersion::from_u32(0), None);
    assert_eq!(GgufVersion::from_u32(99), None);
    assert_eq!(GgufVersion::from_u32(u32::MAX), None);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3 — GgufVersion ordering
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn gguf_version_ordering() {
    assert!(GgufVersion::V1 < GgufVersion::V2);
    assert!(GgufVersion::V2 < GgufVersion::V3);
    assert!(GgufVersion::V1 < GgufVersion::V3);
    assert_eq!(GgufVersion::V2, GgufVersion::V2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4 — F16 KV cache support by version
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn gguf_version_f16_kv_support() {
    assert!(
        !GgufVersion::V1.supports_f16_kv(),
        "V1 must NOT support F16 KV"
    );
    assert!(GgufVersion::V2.supports_f16_kv(), "V2 must support F16 KV");
    assert!(GgufVersion::V3.supports_f16_kv(), "V3 must support F16 KV");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5 — Aligned tensor support by version
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn gguf_version_aligned_tensor_support() {
    assert!(
        !GgufVersion::V1.supports_aligned_tensors(),
        "V1 must NOT support aligned tensors"
    );
    assert!(
        !GgufVersion::V2.supports_aligned_tensors(),
        "V2 must NOT support aligned tensors"
    );
    assert!(
        GgufVersion::V3.supports_aligned_tensors(),
        "V3 must support aligned tensors"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6 — ExtendedQuantType::from_u32 with known IDs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn extended_quant_from_u32_known() {
    assert_eq!(ExtendedQuantType::from_u32(0), ExtendedQuantType::F32);
    assert_eq!(ExtendedQuantType::from_u32(1), ExtendedQuantType::F16);
    assert_eq!(ExtendedQuantType::from_u32(2), ExtendedQuantType::Q4_0);
    assert_eq!(ExtendedQuantType::from_u32(3), ExtendedQuantType::Q4_1);
    assert_eq!(ExtendedQuantType::from_u32(6), ExtendedQuantType::Q5_0);
    assert_eq!(ExtendedQuantType::from_u32(7), ExtendedQuantType::Q5_1);
    assert_eq!(ExtendedQuantType::from_u32(8), ExtendedQuantType::Q8_0);
    assert_eq!(ExtendedQuantType::from_u32(9), ExtendedQuantType::Q8_1);
    assert_eq!(ExtendedQuantType::from_u32(10), ExtendedQuantType::Q2_K);
    assert_eq!(ExtendedQuantType::from_u32(11), ExtendedQuantType::Q3_K);
    assert_eq!(ExtendedQuantType::from_u32(12), ExtendedQuantType::Q4_K);
    assert_eq!(ExtendedQuantType::from_u32(13), ExtendedQuantType::Q5_K);
    assert_eq!(ExtendedQuantType::from_u32(14), ExtendedQuantType::Q6_K);
    assert_eq!(ExtendedQuantType::from_u32(15), ExtendedQuantType::Q8_K);
    assert_eq!(
        ExtendedQuantType::from_u32(41),
        ExtendedQuantType::Q1_0_G128
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7 — ExtendedQuantType::from_u32 with unknown IDs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn extended_quant_from_u32_unknown() {
    assert_eq!(
        ExtendedQuantType::from_u32(4),
        ExtendedQuantType::Unknown(4)
    );
    assert_eq!(
        ExtendedQuantType::from_u32(5),
        ExtendedQuantType::Unknown(5)
    );
    assert_eq!(
        ExtendedQuantType::from_u32(100),
        ExtendedQuantType::Unknown(100)
    );
    assert_eq!(
        ExtendedQuantType::from_u32(999_999),
        ExtendedQuantType::Unknown(999_999)
    );
    assert_eq!(
        ExtendedQuantType::from_u32(u32::MAX),
        ExtendedQuantType::Unknown(u32::MAX)
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8 — bits_per_weight values
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn extended_quant_bits_per_weight() {
    // Exact values
    assert_eq!(ExtendedQuantType::F32.bits_per_weight(), 32.0_f32);
    assert_eq!(ExtendedQuantType::F16.bits_per_weight(), 16.0_f32);

    // Q4_K ≈ 4.5 (within a small epsilon of the K-quant overhead)
    let q4k = ExtendedQuantType::Q4_K.bits_per_weight();
    assert!(
        (q4k - 4.5_f32).abs() < 0.1,
        "Q4_K bits_per_weight should be ~4.5, got {q4k}"
    );

    // Q1_0_G128 ≈ 1.125 (1 bit + 16-bit scale amortised over 128 weights)
    let q1 = ExtendedQuantType::Q1_0_G128.bits_per_weight();
    assert!(
        (q1 - 1.125_f32).abs() < 0.001,
        "Q1_0_G128 bits_per_weight should be 1.125, got {q1}"
    );

    // Q5_K ≈ 5.5
    let q5k = ExtendedQuantType::Q5_K.bits_per_weight();
    assert!(
        (q5k - 5.5_f32).abs() < 0.1,
        "Q5_K bits_per_weight should be ~5.5, got {q5k}"
    );

    // Unknown → 0.0
    assert_eq!(ExtendedQuantType::Unknown(42).bits_per_weight(), 0.0_f32);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9 — is_known()
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn extended_quant_is_known() {
    assert!(ExtendedQuantType::F32.is_known());
    assert!(ExtendedQuantType::F16.is_known());
    assert!(ExtendedQuantType::Q4_0.is_known());
    assert!(ExtendedQuantType::Q4_K.is_known());
    assert!(ExtendedQuantType::Q5_K.is_known());
    assert!(ExtendedQuantType::Q6_K.is_known());
    assert!(ExtendedQuantType::Q8_0.is_known());
    assert!(ExtendedQuantType::Q1_0_G128.is_known());

    assert!(!ExtendedQuantType::Unknown(0).is_known());
    assert!(!ExtendedQuantType::Unknown(999).is_known());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10 — name() returns non-empty strings for all variants
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn extended_quant_name() {
    let known_variants = [
        ExtendedQuantType::F32,
        ExtendedQuantType::F16,
        ExtendedQuantType::Q4_0,
        ExtendedQuantType::Q4_1,
        ExtendedQuantType::Q5_0,
        ExtendedQuantType::Q5_1,
        ExtendedQuantType::Q8_0,
        ExtendedQuantType::Q8_1,
        ExtendedQuantType::Q2_K,
        ExtendedQuantType::Q3_K,
        ExtendedQuantType::Q4_K,
        ExtendedQuantType::Q5_K,
        ExtendedQuantType::Q6_K,
        ExtendedQuantType::Q8_K,
        ExtendedQuantType::Q1_0_G128,
    ];

    for variant in &known_variants {
        let n = variant.name();
        assert!(!n.is_empty(), "name() must be non-empty for {:?}", variant);
    }

    // Unknown variant also has a non-empty name.
    assert!(!ExtendedQuantType::Unknown(77).name().is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 11 — GgufCompatReport::new creates a clean slate
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compat_report_new() {
    let report = GgufCompatReport::new(GgufVersion::V3, 10, 25);
    assert_eq!(report.version, GgufVersion::V3);
    assert_eq!(report.tensor_count, 10);
    assert_eq!(report.metadata_count, 25);
    assert!(
        report.warnings.is_empty(),
        "should start with zero warnings"
    );
    assert!(
        report.unknown_quant_types.is_empty(),
        "should start with no unknown quants"
    );
    assert!(report.is_loadable, "should start as loadable");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 12 — add_warning increments warning count
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compat_report_add_warning() {
    let mut report = GgufCompatReport::new(GgufVersion::V2, 5, 5);
    assert_eq!(report.warnings.len(), 0);

    report.add_warning("first warning");
    assert_eq!(report.warnings.len(), 1);

    report.add_warning("second warning");
    assert_eq!(report.warnings.len(), 2);

    assert_eq!(report.warnings[0], "first warning");
    assert_eq!(report.warnings[1], "second warning");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 13 — add_unknown_quant populates unknown_quant_types
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compat_report_add_unknown_quant() {
    let mut report = GgufCompatReport::new(GgufVersion::V3, 3, 3);
    assert!(report.unknown_quant_types.is_empty());

    report.add_unknown_quant(99);
    report.add_unknown_quant(200);

    assert_eq!(report.unknown_quant_types.len(), 2);
    assert!(report.unknown_quant_types.contains(&99));
    assert!(report.unknown_quant_types.contains(&200));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 14 — summary() returns a non-empty string
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compat_report_summary_nonempty() {
    let report = GgufCompatReport::new(GgufVersion::V3, 100, 30);
    let s = report.summary();
    assert!(!s.is_empty(), "summary must not be empty");
    // Should contain version and loadable status as a basic sanity check.
    assert!(s.contains("v3"), "summary should mention the version");
    assert!(s.contains("loadable"), "summary should mention loadability");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 15 — check_gguf_header with valid v2 header
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn check_gguf_header_valid_v2() {
    let mut bytes = Vec::with_capacity(8);
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&2u32.to_le_bytes());

    let result = check_gguf_header(&bytes);
    assert!(
        result.is_ok(),
        "v2 header should parse successfully: {:?}",
        result
    );
    assert_eq!(
        result.expect("GGUFv2 parse should succeed"),
        GgufVersion::V2
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 16 — check_gguf_header with valid v3 header
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn check_gguf_header_valid_v3() {
    let mut bytes = Vec::with_capacity(8);
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());

    let result = check_gguf_header(&bytes);
    assert!(
        result.is_ok(),
        "v3 header should parse successfully: {:?}",
        result
    );
    assert_eq!(
        result.expect("GGUFv3 parse should succeed"),
        GgufVersion::V3
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 17 — check_gguf_header with wrong magic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn check_gguf_header_invalid_magic() {
    let mut bytes = Vec::with_capacity(8);
    bytes.extend_from_slice(b"BAAD");
    bytes.extend_from_slice(&3u32.to_le_bytes());

    let result = check_gguf_header(&bytes);
    assert!(result.is_err(), "wrong magic should produce an error");
    match result.unwrap_err() {
        CompatError::InvalidMagic(magic) => {
            assert_eq!(magic, b"BAAD");
        }
        other => panic!("expected InvalidMagic, got: {other}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 18 — check_gguf_header with too-few bytes
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn check_gguf_header_truncated() {
    // Only 3 bytes — not enough even for the magic.
    let bytes = b"GGU";
    let result = check_gguf_header(bytes);
    assert!(result.is_err(), "truncated bytes should produce an error");
    match result.unwrap_err() {
        CompatError::TruncatedHeader { need, got } => {
            assert_eq!(need, 8);
            assert_eq!(got, 3);
        }
        other => panic!("expected TruncatedHeader, got: {other}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 19 — build_compat_report with all-known quants produces no warnings
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn build_compat_report_no_unknowns() {
    // Mix of known quant IDs: F32=0, F16=1, Q4_K=12, Q1_0_G128=41
    let quant_ids: &[u32] = &[0, 1, 12, 41];
    let report = build_compat_report(3, 4, 10, quant_ids);

    assert_eq!(report.version, GgufVersion::V3);
    assert_eq!(report.tensor_count, 4);
    assert_eq!(report.metadata_count, 10);
    assert!(
        report.unknown_quant_types.is_empty(),
        "should be no unknown quant types"
    );
    assert!(
        report.warnings.is_empty(),
        "should be no warnings when all quants are known"
    );
    assert!(
        report.is_loadable,
        "file should be loadable when all quants are known"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 20 — build_compat_report with unknown quant IDs triggers warning
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn build_compat_report_with_unknowns() {
    // Quant IDs 4, 5 are unassigned gaps in GGUF; 999 is a future format stub.
    let quant_ids: &[u32] = &[0, 1, 4, 5, 999];
    let report = build_compat_report(3, 5, 12, quant_ids);

    assert_eq!(report.tensor_count, 5);
    assert!(
        !report.unknown_quant_types.is_empty(),
        "should have recorded unknown quant types"
    );
    assert!(
        report.unknown_quant_types.contains(&4),
        "quant id 4 should be recorded as unknown"
    );
    assert!(
        report.unknown_quant_types.contains(&5),
        "quant id 5 should be recorded as unknown"
    );
    assert!(
        report.unknown_quant_types.contains(&999),
        "quant id 999 should be recorded as unknown"
    );

    // Because of unknown quants the file is considered un-loadable.
    assert!(
        !report.is_loadable,
        "file with unknown quant types must not be considered loadable"
    );

    // And a synthesised warning should have been appended by finalize().
    assert!(
        !report.warnings.is_empty(),
        "warnings should be non-empty after unknown quants"
    );
}

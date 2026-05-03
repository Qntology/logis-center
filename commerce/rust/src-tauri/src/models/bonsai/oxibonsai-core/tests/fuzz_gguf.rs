//! Fuzz-style tests for GGUF parser robustness using proptest.
//!
//! Ensures the parser handles adversarial and malformed input gracefully
//! (returning errors, never panicking or OOMing).

use proptest::prelude::*;

use oxibonsai_core::gguf::header::GgufHeader;
use oxibonsai_core::gguf::metadata::MetadataStore;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::TensorStore;
use oxibonsai_core::gguf::types::{GgufTensorType, GgufValueType};

// ── Helper: build a valid GGUF header ────────────────────────────────────

const GGUF_MAGIC: u32 = 0x4655_4747;

fn make_valid_header(version: u32, tensor_count: u64, metadata_kv_count: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&version.to_le_bytes());
    data.extend_from_slice(&tensor_count.to_le_bytes());
    data.extend_from_slice(&metadata_kv_count.to_le_bytes());
    data
}

// ── 1. Random byte sequences never panic ─────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn random_bytes_header_no_panic(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        // Should return Err, never panic
        let _ = GgufHeader::parse(&data, 0);
    }

    #[test]
    fn random_bytes_full_parse_no_panic(data in proptest::collection::vec(any::<u8>(), 0..512)) {
        // Full parser should also never panic on random data
        let _ = GgufFile::parse(&data);
    }

    #[test]
    fn random_bytes_metadata_no_panic(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        // Metadata parser with random data + random count
        let _ = MetadataStore::parse(&data, 0, 1);
        let _ = MetadataStore::parse(&data, 0, 0);
    }

    #[test]
    fn random_bytes_tensor_store_no_panic(data in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = TensorStore::parse(&data, 0, 1);
        let _ = TensorStore::parse(&data, 0, 0);
    }
}

// ── 2. Truncated data at various points returns errors ───────────────────

#[test]
fn truncated_header_empty() {
    let result = GgufHeader::parse(&[], 0);
    assert!(result.is_err(), "empty data should fail");
}

#[test]
fn truncated_header_partial_magic() {
    let data = GGUF_MAGIC.to_le_bytes();
    // Only 4 bytes: magic but no version
    let result = GgufHeader::parse(&data[..3], 0);
    assert!(result.is_err(), "3 bytes should fail");
}

#[test]
fn truncated_header_magic_only() {
    let data = GGUF_MAGIC.to_le_bytes();
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err(), "magic only (4 bytes) should fail");
}

#[test]
fn truncated_header_no_counts() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes()); // version
                                                 // Missing tensor_count and metadata_kv_count
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err(), "header without counts should fail");
}

#[test]
fn truncated_header_partial_tensor_count() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&[0u8; 4]); // partial u64 tensor_count
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err(), "partial tensor_count should fail");
}

// ── 3. Oversized metadata counts don't cause OOM ─────────────────────────

#[test]
fn oversized_metadata_count_returns_error() {
    // Header claims u64::MAX metadata entries, but only 24 bytes of header data
    let data = make_valid_header(3, 0, u64::MAX);
    let result = GgufFile::parse(&data);
    // Should fail because there's no actual metadata to parse, not OOM
    assert!(
        result.is_err(),
        "u64::MAX metadata count should fail gracefully"
    );
}

#[test]
fn oversized_tensor_count_returns_error() {
    // Header claims u64::MAX tensors, zero metadata
    let data = make_valid_header(3, u64::MAX, 0);
    let result = GgufFile::parse(&data);
    // Should fail since no tensor data follows
    assert!(
        result.is_err(),
        "u64::MAX tensor count should fail gracefully"
    );
}

#[test]
fn large_metadata_count_no_oom() {
    // Claim 1 billion metadata entries but provide no data
    let data = make_valid_header(3, 0, 1_000_000_000);
    let result = GgufFile::parse(&data);
    assert!(result.is_err(), "1B metadata entries should fail, not OOM");
}

// ── 4. Invalid UTF-8 in string metadata returns error ────────────────────

#[test]
fn invalid_utf8_string_metadata() {
    // Build a metadata entry with invalid UTF-8 bytes in the key
    let mut data = Vec::new();
    // String length = 4
    data.extend_from_slice(&4u64.to_le_bytes());
    // Invalid UTF-8 bytes (0xFF is never valid in UTF-8)
    data.extend_from_slice(&[0xFF, 0xFE, 0x80, 0x80]);
    // Value type = Uint32 (4)
    data.extend_from_slice(&4u32.to_le_bytes());
    // Value = 42
    data.extend_from_slice(&42u32.to_le_bytes());

    let result = MetadataStore::parse(&data, 0, 1);
    assert!(result.is_err(), "invalid UTF-8 key should return error");
}

#[test]
fn invalid_utf8_string_value() {
    // Valid key, but String value with invalid UTF-8
    let mut data = Vec::new();
    // Key: "test" (valid UTF-8)
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(b"test");
    // Value type = String (8)
    data.extend_from_slice(&8u32.to_le_bytes());
    // String length = 3
    data.extend_from_slice(&3u64.to_le_bytes());
    // Invalid UTF-8 bytes
    data.extend_from_slice(&[0xFF, 0xFE, 0xFD]);

    let result = MetadataStore::parse(&data, 0, 1);
    assert!(
        result.is_err(),
        "invalid UTF-8 string value should return error"
    );
}

// ── 5. Random tensor type IDs handled ────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn random_tensor_type_id_no_panic(id in any::<u32>()) {
        // Should return Ok for known types, Err for unknown, never panic
        let result = GgufTensorType::from_id(id);
        match id {
            0 | 1 | 2 | 3 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 30 | 41 => {
                assert!(result.is_ok(), "known type {id} should parse");
            }
            _ => {
                assert!(result.is_err(), "unknown type {id} should fail");
            }
        }
    }

    #[test]
    fn random_value_type_id_no_panic(id in any::<u32>()) {
        let result = GgufValueType::from_id(id);
        if id <= 12 {
            assert!(result.is_ok(), "value type {id} should parse");
        } else {
            assert!(result.is_err(), "value type {id} should fail");
        }
    }
}

// ── 6. Alignment calculations with random offsets ────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn alignment_never_decreases_offset(offset in 0usize..1_000_000, alignment in 1usize..1024) {
        let aligned = (offset + alignment - 1) & !(alignment - 1);
        prop_assert!(aligned >= offset, "aligned offset must be >= original");
    }

    #[test]
    fn alignment_result_is_multiple(offset in 0usize..1_000_000, alignment_exp in 0u32..10) {
        let alignment = 1usize << alignment_exp; // power of 2
        let aligned = (offset + alignment - 1) & !(alignment - 1);
        prop_assert_eq!(aligned % alignment, 0, "result must be multiple of alignment");
    }
}

// ── 7. Metadata with max u32/u64 values ──────────────────────────────────

#[test]
fn metadata_u32_max_value() {
    let mut data = Vec::new();
    // Key: "max_u32"
    let key = "max_u32";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    // Value type = Uint32 (4)
    data.extend_from_slice(&4u32.to_le_bytes());
    // Value = u32::MAX
    data.extend_from_slice(&u32::MAX.to_le_bytes());

    let (store, _) = MetadataStore::parse(&data, 0, 1).expect("max u32 value should parse");
    let val = store.get_u32("max_u32").expect("key should exist");
    assert_eq!(val, u32::MAX);
}

#[test]
fn metadata_u64_max_value() {
    let mut data = Vec::new();
    // Key: "max_u64"
    let key = "max_u64";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    // Value type = Uint64 (10)
    data.extend_from_slice(&10u32.to_le_bytes());
    // Value = u64::MAX
    data.extend_from_slice(&u64::MAX.to_le_bytes());

    let (store, _) = MetadataStore::parse(&data, 0, 1).expect("max u64 value should parse");
    let val = store.get_u64("max_u64").expect("key should exist");
    assert_eq!(val, u64::MAX);
}

#[test]
fn header_version_2_accepted() {
    let data = make_valid_header(2, 0, 0);
    let (header, _) = GgufHeader::parse(&data, 0).expect("version 2 should be accepted");
    assert_eq!(header.version, 2);
}

#[test]
fn header_version_3_accepted() {
    let data = make_valid_header(3, 0, 0);
    let (header, _) = GgufHeader::parse(&data, 0).expect("version 3 should be accepted");
    assert_eq!(header.version, 3);
}

#[test]
fn header_version_0_rejected() {
    let data = make_valid_header(0, 0, 0);
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err(), "version 0 should be rejected");
}

#[test]
fn header_version_1_rejected() {
    let data = make_valid_header(1, 0, 0);
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err(), "version 1 should be rejected");
}

#[test]
fn header_version_4_rejected() {
    let data = make_valid_header(4, 0, 0);
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err(), "version 4 should be rejected");
}

#[test]
fn empty_metadata_store_operations() {
    let store = MetadataStore::new();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
    assert!(store.get("any_key").is_none());
    assert!(store.get_string("any_key").is_err());
    assert!(store.get_u32("any_key").is_err());
    assert!(store.get_u64("any_key").is_err());
    assert!(store.get_f32("any_key").is_err());
    assert_eq!(store.get_u32_or("any_key", 99), 99);
    assert!(
        (store.get_f32_or("any_key", std::f32::consts::PI) - std::f32::consts::PI).abs()
            < f32::EPSILON
    );
}

#[test]
fn tensor_store_empty_operations() {
    let store = TensorStore::new();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
    assert!(store.get("any_name").is_none());
    assert!(store.require("any_name").is_err());
    assert!(store.sorted_names().is_empty());
    assert!(store.count_by_type().is_empty());
}

// ── Tensor type properties ───────────────────────────────────────────────

#[test]
fn all_known_tensor_types_have_properties() {
    let known_ids: &[u32] = &[
        0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 30, 35, 41, 42,
    ];
    for &id in known_ids {
        let ty = GgufTensorType::from_id(id).expect("known type should parse");
        assert!(ty.block_size() > 0, "block_size for {ty} must be > 0");
        assert!(ty.block_bytes() > 0, "block_bytes for {ty} must be > 0");
        assert!(!ty.name().is_empty(), "name for {ty} must not be empty");
    }
}

#[test]
fn q1_0_g128_is_only_one_bit() {
    let known_ids: &[u32] = &[
        0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 30, 35, 41, 42,
    ];
    for &id in known_ids {
        let ty = GgufTensorType::from_id(id).expect("known type should parse");
        if id == 41 {
            assert!(ty.is_one_bit(), "Q1_0_g128 should be one_bit");
        } else {
            assert!(!ty.is_one_bit(), "{ty} should not be one_bit");
        }
    }
}

// ── 8. Truncated-header proptest (varied byte counts) ────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Any slice shorter than a complete GGUF header (24 bytes) must return Err.
    #[test]
    fn prop_test_gguf_truncated_header(
        len in 0usize..24usize,
        fill in any::<u8>(),
    ) {
        let data = vec![fill; len];
        let result = GgufHeader::parse(&data, 0);
        prop_assert!(result.is_err(),
            "truncated header of {len} bytes should fail");
    }

    /// Supplying a wrong (non-GGUF) magic value must cause parse to fail.
    #[test]
    fn prop_test_gguf_random_magic(
        magic in any::<u32>().prop_filter("not GGUF magic", |&m| m != 0x4655_4747u32),
    ) {
        let mut data = Vec::with_capacity(24);
        data.extend_from_slice(&magic.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());         // version = 3
        data.extend_from_slice(&0u64.to_le_bytes());         // tensor_count = 0
        data.extend_from_slice(&0u64.to_le_bytes());         // metadata_kv_count = 0
        let result = GgufFile::parse(&data);
        prop_assert!(result.is_err(),
            "wrong magic {magic:#010x} should be rejected");
    }

    /// Valid GGUF magic with an implausibly large tensor count must return Err.
    #[test]
    fn prop_test_gguf_invalid_tensor_count(
        tensor_count in (1_000_000u64..=u64::MAX),
    ) {
        let data = make_valid_header(3, tensor_count, 0);
        let result = GgufFile::parse(&data);
        prop_assert!(result.is_err(),
            "giant tensor_count {tensor_count} should fail");
    }
}

/// Empty byte slice must return Err immediately without panicking.
#[test]
fn prop_test_gguf_empty_input() {
    let result = GgufFile::parse(&[]);
    assert!(result.is_err(), "empty input must return error");
    // Verify the header parser also rejects empty slices.
    let result2 = GgufHeader::parse(&[], 0);
    assert!(result2.is_err(), "GgufHeader must reject empty input");
}

/// Valid magic + version but random trailing body must return Err without panicking.
#[test]
fn prop_test_gguf_valid_magic_corrupted_body() {
    // Use a large claimed tensor/metadata count so the parser tries to read
    // more data than is available and returns an error.
    let mut data = Vec::new();
    data.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // correct magic
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&1_000u64.to_le_bytes()); // 1000 tensors (won't fit)
    data.extend_from_slice(&1_000u64.to_le_bytes()); // 1000 metadata entries
                                                     // Random trailing bytes that are nowhere near enough to satisfy the counts.
    data.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22, 0x33]);
    let result = GgufFile::parse(&data);
    assert!(result.is_err(), "corrupted body should return error");
}

// ── 9. Metadata string max-len handling ──────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// A metadata entry whose string key claims to be very long but provides
    /// insufficient data must return an error (not panic or allocate huge memory).
    #[test]
    fn prop_test_metadata_string_max_len(
        claimed_len in (1_000_000u64..=u64::MAX / 2),
    ) {
        let mut data = Vec::new();
        // Key length = claimed_len (far more than data provides).
        data.extend_from_slice(&claimed_len.to_le_bytes());
        // Only a few bytes of "key" data — far short of claimed_len.
        data.extend_from_slice(b"short");
        // Value type = Uint32, value = 0 (will never be reached).
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        let result = MetadataStore::parse(&data, 0, 1);
        prop_assert!(result.is_err(),
            "claimed string len {claimed_len} with only 5 bytes should fail");
    }

    /// A metadata array entry that claims a huge element count must return an
    /// error without allocating enormous memory (OOM prevention).
    #[test]
    fn prop_test_metadata_array_overflow(
        array_count in (1_000_000u64..=u64::MAX / 2),
    ) {
        let mut data = Vec::new();
        // Key: "arr"
        let key = b"arr";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key);
        // Value type = Array (9 in GGUF spec).
        data.extend_from_slice(&9u32.to_le_bytes());
        // Array element type = Uint32 (4).
        data.extend_from_slice(&4u32.to_le_bytes());
        // Element count = array_count (absurdly large).
        data.extend_from_slice(&array_count.to_le_bytes());
        // No actual elements follow — parser must detect truncation.

        let result = MetadataStore::parse(&data, 0, 1);
        prop_assert!(result.is_err(),
            "array with {array_count} claimed elements must fail, not OOM");
    }
}

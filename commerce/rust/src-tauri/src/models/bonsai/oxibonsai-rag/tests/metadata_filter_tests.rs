//! Metadata filter integration tests.

use std::collections::HashMap;

use oxibonsai_rag::metadata_filter::{MetadataFilter, MetadataValue};
use oxibonsai_rag::RagError;

fn md() -> HashMap<String, MetadataValue> {
    let mut m = HashMap::new();
    m.insert("lang".into(), MetadataValue::from("rust"));
    m.insert("year".into(), MetadataValue::from(2026_i64));
    m.insert("stable".into(), MetadataValue::from(true));
    m.insert("score".into(), MetadataValue::from(0.75_f64));
    m
}

// ── Equals ───────────────────────────────────────────────────────────────────

#[test]
fn equals_match() {
    assert!(MetadataFilter::eq("lang", "rust").matches(&md()));
}

#[test]
fn equals_no_match() {
    assert!(!MetadataFilter::eq("lang", "python").matches(&md()));
}

#[test]
fn equals_missing_key() {
    assert!(!MetadataFilter::eq("missing", "x").matches(&md()));
}

#[test]
fn equals_empty_map() {
    let empty: HashMap<String, MetadataValue> = HashMap::new();
    assert!(!MetadataFilter::eq("lang", "rust").matches(&empty));
}

// ── NotEquals ────────────────────────────────────────────────────────────────

#[test]
fn not_equals_match() {
    assert!(MetadataFilter::neq("lang", "python").matches(&md()));
}

#[test]
fn not_equals_no_match() {
    assert!(!MetadataFilter::neq("lang", "rust").matches(&md()));
}

#[test]
fn not_equals_missing_key_is_false() {
    // NotEquals requires the key to exist; missing key → no match.
    assert!(!MetadataFilter::neq("absent", "x").matches(&md()));
}

// ── In ───────────────────────────────────────────────────────────────────────

#[test]
fn in_match() {
    let f = MetadataFilter::In(
        "year".into(),
        vec![
            MetadataValue::from(2024_i64),
            MetadataValue::from(2025_i64),
            MetadataValue::from(2026_i64),
        ],
    );
    assert!(f.matches(&md()));
}

#[test]
fn in_no_match() {
    let f = MetadataFilter::In("year".into(), vec![MetadataValue::from(1999_i64)]);
    assert!(!f.matches(&md()));
}

#[test]
fn in_missing_key() {
    let f = MetadataFilter::In("missing".into(), vec![MetadataValue::from(1_i64)]);
    assert!(!f.matches(&md()));
}

#[test]
fn in_empty_values_rejected_by_validate() {
    let f = MetadataFilter::In("year".into(), vec![]);
    assert!(matches!(f.validate(), Err(RagError::InvalidFilter(_))));
}

// ── Exists ───────────────────────────────────────────────────────────────────

#[test]
fn exists_match() {
    assert!(MetadataFilter::exists("lang").matches(&md()));
}

#[test]
fn exists_no_match() {
    assert!(!MetadataFilter::exists("absent").matches(&md()));
}

#[test]
fn exists_empty_map() {
    let empty: HashMap<String, MetadataValue> = HashMap::new();
    assert!(!MetadataFilter::exists("lang").matches(&empty));
}

// ── All / Any composition ────────────────────────────────────────────────────

#[test]
fn all_empty_matches_everything() {
    let f = MetadataFilter::All(vec![]);
    assert!(f.matches(&md()));
    let empty: HashMap<String, MetadataValue> = HashMap::new();
    assert!(f.matches(&empty));
}

#[test]
fn any_empty_matches_nothing() {
    let f = MetadataFilter::Any(vec![]);
    assert!(!f.matches(&md()));
}

#[test]
fn all_nested_any() {
    let f = MetadataFilter::All(vec![
        MetadataFilter::eq("lang", "rust"),
        MetadataFilter::Any(vec![
            MetadataFilter::eq("year", 2025_i64),
            MetadataFilter::eq("year", 2026_i64),
        ]),
    ]);
    assert!(f.matches(&md()));
}

#[test]
fn any_nested_all() {
    let f = MetadataFilter::Any(vec![
        MetadataFilter::All(vec![
            MetadataFilter::eq("lang", "python"),
            MetadataFilter::eq("year", 2026_i64),
        ]),
        MetadataFilter::All(vec![
            MetadataFilter::eq("lang", "rust"),
            MetadataFilter::eq("year", 2026_i64),
        ]),
    ]);
    assert!(f.matches(&md()));
}

// ── Bool / Float leaves ──────────────────────────────────────────────────────

#[test]
fn equals_bool() {
    assert!(MetadataFilter::eq("stable", true).matches(&md()));
    assert!(!MetadataFilter::eq("stable", false).matches(&md()));
}

#[test]
fn equals_float_exact() {
    let f = MetadataFilter::Equals("score".into(), MetadataValue::from(0.75_f64));
    assert!(f.matches(&md()));
}

// ── Validation ───────────────────────────────────────────────────────────────

#[test]
fn empty_key_rejected() {
    let f = MetadataFilter::Equals(String::new(), MetadataValue::from("x"));
    assert!(matches!(f.validate(), Err(RagError::InvalidFilter(_))));
}

#[test]
fn empty_exists_key_rejected() {
    let f = MetadataFilter::Exists(String::new());
    assert!(matches!(f.validate(), Err(RagError::InvalidFilter(_))));
}

#[test]
fn deeply_nested_validation_propagates() {
    let f = MetadataFilter::All(vec![MetadataFilter::Any(vec![MetadataFilter::Equals(
        String::new(),
        MetadataValue::from("x"),
    )])]);
    assert!(matches!(f.validate(), Err(RagError::InvalidFilter(_))));
}

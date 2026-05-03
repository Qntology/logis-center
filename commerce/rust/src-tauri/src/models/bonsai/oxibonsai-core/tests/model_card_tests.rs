//! Integration tests for the GGUF model card extraction and rendering.

use oxibonsai_core::gguf::model_card::{extract_known_fields, extract_model_card, keys, ModelCard};
use std::collections::HashMap;

// ── Test helpers ─────────────────────────────────────────────────────────────

fn full_metadata() -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert(keys::MODEL_NAME.to_owned(), "Llama-3-8B".to_owned());
    m.insert(keys::ARCHITECTURE.to_owned(), "llama".to_owned());
    m.insert(keys::AUTHOR.to_owned(), "Meta".to_owned());
    m.insert(keys::LICENSE.to_owned(), "llama3".to_owned());
    m.insert(
        keys::DESCRIPTION.to_owned(),
        "An 8-billion-parameter language model.".to_owned(),
    );
    m.insert(keys::CONTEXT_LENGTH.to_owned(), "8192".to_owned());
    m.insert(keys::EMBEDDING_LENGTH.to_owned(), "4096".to_owned());
    m.insert(keys::NUM_LAYERS.to_owned(), "32".to_owned());
    m.insert(keys::NUM_HEADS.to_owned(), "32".to_owned());
    m.insert(keys::NUM_KV_HEADS.to_owned(), "8".to_owned());
    m.insert(keys::ROPE_FREQ_BASE.to_owned(), "500000.0".to_owned());
    m.insert(keys::VOCAB_SIZE.to_owned(), "128256".to_owned());
    m.insert(keys::QUANTIZATION.to_owned(), "Q4_K_M".to_owned());
    m.insert(keys::FILE_SIZE.to_owned(), "4815060000".to_owned());
    m
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// 1. A freshly constructed ModelCard is empty.
#[test]
fn model_card_new_empty() {
    let card = ModelCard::new();
    assert!(card.is_empty(), "new card must report is_empty() == true");
}

/// 2. populated_count() only counts typed fields, not extra_metadata.
#[test]
fn model_card_populated_count() {
    let mut card = ModelCard::new();
    assert_eq!(card.populated_count(), 0);

    card.model_name = Some("TestModel".to_owned());
    assert_eq!(card.populated_count(), 1);

    card.architecture = Some("llama".to_owned());
    card.num_layers = Some(32);
    card.num_heads = Some(32);
    assert_eq!(card.populated_count(), 4);

    // extra_metadata must not affect populated_count.
    card.extra_metadata
        .insert("some.key".to_owned(), "val".to_owned());
    assert_eq!(card.populated_count(), 4);
}

/// 3. to_markdown() returns a non-empty string even for an empty card.
#[test]
fn model_card_to_markdown_empty() {
    let card = ModelCard::new();
    let md = card.to_markdown();
    assert!(
        !md.is_empty(),
        "to_markdown() must return a non-empty string for an empty card"
    );
}

/// 4. to_markdown() contains the model name when set.
#[test]
fn model_card_to_markdown_with_name() {
    let mut card = ModelCard::new();
    card.model_name = Some("Qwen3-8B".to_owned());
    let md = card.to_markdown();
    assert!(
        md.contains("Qwen3-8B"),
        "markdown must contain the model name; got:\n{md}"
    );
}

/// 5. to_summary() returns a non-empty string for an empty card.
#[test]
fn model_card_to_summary_empty() {
    let card = ModelCard::new();
    let summary = card.to_summary();
    assert!(
        !summary.is_empty(),
        "to_summary() must not return an empty string"
    );
}

/// 6. to_summary() contains values for all set fields.
#[test]
fn model_card_to_summary_with_fields() {
    let metadata = full_metadata();
    let card = extract_model_card(&metadata);
    let summary = card.to_summary();
    assert!(
        summary.contains("Llama-3-8B"),
        "summary must contain model name"
    );
    assert!(
        summary.contains("llama"),
        "summary must contain architecture"
    );
    assert!(
        summary.contains("8192"),
        "summary must contain context length"
    );
    assert!(
        summary.contains("4096"),
        "summary must contain embedding length"
    );
}

/// 7. extract_model_card correctly parses general.name.
#[test]
fn extract_model_card_from_metadata() {
    let mut m = HashMap::new();
    m.insert(keys::MODEL_NAME.to_owned(), "TestModel-7B".to_owned());
    let card = extract_model_card(&m);
    assert_eq!(card.model_name.as_deref(), Some("TestModel-7B"));
}

/// 8. extract_model_card correctly parses general.architecture.
#[test]
fn extract_model_card_architecture() {
    let mut m = HashMap::new();
    m.insert(keys::ARCHITECTURE.to_owned(), "qwen3".to_owned());
    let card = extract_model_card(&m);
    assert_eq!(card.architecture.as_deref(), Some("qwen3"));
}

/// 9. extract_model_card correctly parses llm.context_length as u64.
#[test]
fn extract_model_card_context_length() {
    let mut m = HashMap::new();
    m.insert(keys::CONTEXT_LENGTH.to_owned(), "32768".to_owned());
    let card = extract_model_card(&m);
    assert_eq!(card.context_length, Some(32768u64));
}

/// 10. An empty metadata map produces an empty card.
#[test]
fn extract_model_card_empty_metadata() {
    let card = extract_model_card(&HashMap::new());
    assert!(
        card.is_empty(),
        "empty metadata must yield an empty ModelCard"
    );
}

/// 11. extract_known_fields returns only keys defined in the keys module.
#[test]
fn extract_known_fields_identifies_known() {
    let mut m = full_metadata();
    m.insert("myapp.custom.key".to_owned(), "custom_value".to_owned());
    m.insert("another.unknown".to_owned(), "something".to_owned());

    let known = extract_known_fields(&m);

    assert!(
        known.contains_key(keys::MODEL_NAME),
        "must contain general.name"
    );
    assert!(
        known.contains_key(keys::ARCHITECTURE),
        "must contain general.architecture"
    );
    assert!(
        known.contains_key(keys::CONTEXT_LENGTH),
        "must contain llm.context_length"
    );
    assert!(
        !known.contains_key("myapp.custom.key"),
        "must NOT contain custom keys"
    );
    assert!(
        !known.contains_key("another.unknown"),
        "must NOT contain unknown keys"
    );
}

/// 12. estimated_param_count() returns Some when embedding_length and
///     num_layers are both set.
#[test]
fn model_card_estimated_param_count() {
    let mut card = ModelCard::new();
    card.embedding_length = Some(4096);
    card.num_layers = Some(32);
    card.num_heads = Some(32);
    card.num_kv_heads = Some(8);
    card.vocab_size = Some(128_256);

    let est = card.estimated_param_count();
    assert!(est.is_some(), "estimated_param_count must return Some");
    let b = est.expect("checked above");
    // A 4096/32-layer GQA model is roughly 8B — allow a generous window.
    assert!(
        b > 2.0 && b < 30.0,
        "estimate {b:.2}B is outside plausible 2–30 B range"
    );
}

/// 13. to_markdown() starts with a level-1 Markdown heading.
#[test]
fn model_card_markdown_has_header() {
    let card = ModelCard::new();
    let md = card.to_markdown();
    assert!(
        md.starts_with("# "),
        "markdown must start with a level-1 heading; got: {:?}",
        &md[..md.len().min(40)]
    );
}

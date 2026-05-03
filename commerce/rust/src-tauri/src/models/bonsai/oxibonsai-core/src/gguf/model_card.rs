//! GGUF model card: extract and render model information from GGUF metadata.
//!
//! This module provides structured extraction of well-known GGUF metadata fields
//! and renders them as a human-readable markdown model card or plain text summary.

use std::collections::HashMap;

// ── Well-known GGUF metadata key names ──────────────────────────────────────

/// Well-known GGUF metadata key names.
pub mod keys {
    pub const MODEL_NAME: &str = "general.name";
    pub const ARCHITECTURE: &str = "general.architecture";
    pub const AUTHOR: &str = "general.author";
    pub const LICENSE: &str = "general.license";
    pub const DESCRIPTION: &str = "general.description";
    pub const CONTEXT_LENGTH: &str = "llm.context_length";
    pub const EMBEDDING_LENGTH: &str = "llm.embedding_length";
    pub const NUM_LAYERS: &str = "llm.block_count";
    pub const NUM_HEADS: &str = "llm.attention.head_count";
    pub const NUM_KV_HEADS: &str = "llm.attention.head_count_kv";
    pub const ROPE_FREQ_BASE: &str = "llm.rope.freq_base";
    pub const VOCAB_SIZE: &str = "tokenizer.ggml.tokens_count";
    pub const QUANTIZATION: &str = "general.quantization_version";
    pub const FILE_SIZE: &str = "general.file_size";
    pub const PARAMETER_COUNT: &str = "general.parameter_count";
}

// ── Markdown rendering helpers ───────────────────────────────────────────────

/// Markdown rendering helpers.
mod render {
    /// Render a markdown heading at the given level (1–6).
    pub fn heading(level: u8, text: &str) -> String {
        let level = level.clamp(1, 6);
        let hashes = "#".repeat(level as usize);
        format!("{hashes} {text}\n")
    }

    /// Render a `**label**: value` field line.
    pub fn field(label: &str, value: &str) -> String {
        format!("- **{label}**: {value}\n")
    }

    /// Render a markdown table row with pipe-separated cells.
    pub fn table_row(cells: &[&str]) -> String {
        let inner = cells.join(" | ");
        format!("| {inner} |\n")
    }

    /// Wrap text in backticks for inline code.
    pub fn code(text: &str) -> String {
        format!("`{text}`")
    }

    /// Wrap text in double-asterisks for bold.
    #[allow(dead_code)]
    pub fn bold(text: &str) -> String {
        format!("**{text}**")
    }
}

// ── ModelCard ────────────────────────────────────────────────────────────────

/// Structured information extracted from GGUF metadata.
///
/// All fields are optional — a file may not include every piece of metadata.
/// Use [`extract_model_card`] to populate this from a raw metadata map.
#[derive(Debug, Clone, Default)]
pub struct ModelCard {
    /// Human-readable model name (e.g. `"Llama-3-8B"`).
    pub model_name: Option<String>,
    /// Architecture identifier (e.g. `"llama"`, `"qwen3"`).
    pub architecture: Option<String>,
    /// Author or organisation that produced the model.
    pub author: Option<String>,
    /// SPDX license identifier or URL (e.g. `"apache-2.0"`).
    pub license: Option<String>,
    /// Free-text description of the model.
    pub description: Option<String>,
    /// Maximum context window in tokens.
    pub context_length: Option<u64>,
    /// Hidden-state / embedding dimension.
    pub embedding_length: Option<u64>,
    /// Number of transformer blocks (layers).
    pub num_layers: Option<u64>,
    /// Number of attention heads.
    pub num_heads: Option<u64>,
    /// Number of key-value heads (GQA).
    pub num_kv_heads: Option<u64>,
    /// RoPE frequency base.
    pub rope_freq_base: Option<f64>,
    /// Vocabulary size.
    pub vocab_size: Option<u64>,
    /// Parameter count expressed in billions (e.g. `7.0` for a 7B model).
    pub parameter_count_billions: Option<f64>,
    /// Quantization scheme string (e.g. `"Q4_K_M"`).
    pub quantization: Option<String>,
    /// Size of the GGUF file on disk, in bytes.
    pub file_size_bytes: Option<u64>,
    /// Any metadata key-value pairs not covered by the typed fields above.
    pub extra_metadata: HashMap<String, String>,
}

impl ModelCard {
    /// Create an empty `ModelCard`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Render the card as a Markdown document.
    ///
    /// The output always begins with a level-1 heading so that the caller can
    /// detect a non-trivial render with `starts_with("# ")`.
    pub fn to_markdown(&self) -> String {
        let mut out = String::with_capacity(1024);

        // Title
        let title = self.model_name.as_deref().unwrap_or("Unknown Model");
        out.push_str(&render::heading(1, title));
        out.push('\n');

        // ── Model Information ──
        out.push_str(&render::heading(2, "Model Information"));
        if let Some(ref v) = self.architecture {
            out.push_str(&render::field("Architecture", v));
        }
        if let Some(ref v) = self.author {
            out.push_str(&render::field("Author", v));
        }
        if let Some(ref v) = self.license {
            out.push_str(&render::field("License", v));
        }
        if let Some(ref v) = self.description {
            out.push_str(&render::field("Description", v));
        }
        if let Some(ref v) = self.quantization {
            out.push_str(&render::field("Quantization", &render::code(v)));
        }
        if let Some(v) = self.file_size_bytes {
            let gb = v as f64 / (1024.0 * 1024.0 * 1024.0);
            out.push_str(&render::field(
                "File Size",
                &format!("{v} bytes ({gb:.2} GB)"),
            ));
        }
        out.push('\n');

        // ── Architecture Details ──
        out.push_str(&render::heading(2, "Architecture Details"));

        // Table header
        out.push_str(&render::table_row(&["Parameter", "Value"]));
        out.push_str(&render::table_row(&["---", "---"]));

        let param_count_str;
        let param_billions = self
            .parameter_count_billions
            .or_else(|| self.estimated_param_count());
        if let Some(b) = param_billions {
            param_count_str = format!("{b:.2}B");
            out.push_str(&render::table_row(&["Parameter Count", &param_count_str]));
        }

        let ctx_str;
        if let Some(v) = self.context_length {
            ctx_str = v.to_string();
            out.push_str(&render::table_row(&["Context Length", &ctx_str]));
        }

        let embed_str;
        if let Some(v) = self.embedding_length {
            embed_str = v.to_string();
            out.push_str(&render::table_row(&["Embedding Length", &embed_str]));
        }

        let layers_str;
        if let Some(v) = self.num_layers {
            layers_str = v.to_string();
            out.push_str(&render::table_row(&["Layers", &layers_str]));
        }

        let heads_str;
        if let Some(v) = self.num_heads {
            heads_str = v.to_string();
            out.push_str(&render::table_row(&["Attention Heads", &heads_str]));
        }

        let kv_heads_str;
        if let Some(v) = self.num_kv_heads {
            kv_heads_str = v.to_string();
            out.push_str(&render::table_row(&["KV Heads", &kv_heads_str]));
        }

        let rope_str;
        if let Some(v) = self.rope_freq_base {
            rope_str = format!("{v:.1}");
            out.push_str(&render::table_row(&["RoPE Freq Base", &rope_str]));
        }

        let vocab_str;
        if let Some(v) = self.vocab_size {
            vocab_str = v.to_string();
            out.push_str(&render::table_row(&["Vocab Size", &vocab_str]));
        }

        out.push('\n');

        // ── Extra Metadata ──
        if !self.extra_metadata.is_empty() {
            out.push_str(&render::heading(2, "Additional Metadata"));
            let mut sorted: Vec<(&String, &String)> = self.extra_metadata.iter().collect();
            sorted.sort_by_key(|(k, _)| k.as_str());
            for (k, v) in sorted {
                out.push_str(&render::field(k, v));
            }
            out.push('\n');
        }

        out
    }

    /// Render a compact plain-text summary — one line per populated field.
    pub fn to_summary(&self) -> String {
        let mut lines: Vec<String> = Vec::new();

        macro_rules! push_opt_str {
            ($label:expr, $field:expr) => {
                if let Some(ref v) = $field {
                    lines.push(format!("{}: {}", $label, v));
                }
            };
        }
        macro_rules! push_opt_num {
            ($label:expr, $field:expr) => {
                if let Some(v) = $field {
                    lines.push(format!("{}: {}", $label, v));
                }
            };
        }

        push_opt_str!("Model", self.model_name);
        push_opt_str!("Architecture", self.architecture);
        push_opt_str!("Author", self.author);
        push_opt_str!("License", self.license);
        push_opt_str!("Description", self.description);
        push_opt_str!("Quantization", self.quantization);
        push_opt_num!("Context Length", self.context_length);
        push_opt_num!("Embedding Length", self.embedding_length);
        push_opt_num!("Layers", self.num_layers);
        push_opt_num!("Attention Heads", self.num_heads);
        push_opt_num!("KV Heads", self.num_kv_heads);

        if let Some(v) = self.rope_freq_base {
            lines.push(format!("RoPE Freq Base: {v:.1}"));
        }
        push_opt_num!("Vocab Size", self.vocab_size);

        let param_billions = self
            .parameter_count_billions
            .or_else(|| self.estimated_param_count());
        if let Some(b) = param_billions {
            lines.push(format!("Parameter Count: {b:.2}B"));
        }

        if let Some(v) = self.file_size_bytes {
            lines.push(format!("File Size: {v} bytes"));
        }

        if lines.is_empty() {
            lines.push("(no metadata available)".to_owned());
        }

        lines.join("\n")
    }

    /// Returns `true` if no structured fields and no extra metadata are set.
    pub fn is_empty(&self) -> bool {
        self.model_name.is_none()
            && self.architecture.is_none()
            && self.author.is_none()
            && self.license.is_none()
            && self.description.is_none()
            && self.context_length.is_none()
            && self.embedding_length.is_none()
            && self.num_layers.is_none()
            && self.num_heads.is_none()
            && self.num_kv_heads.is_none()
            && self.rope_freq_base.is_none()
            && self.vocab_size.is_none()
            && self.parameter_count_billions.is_none()
            && self.quantization.is_none()
            && self.file_size_bytes.is_none()
            && self.extra_metadata.is_empty()
    }

    /// Returns the number of typed (non-`extra_metadata`) fields that are `Some`.
    pub fn populated_count(&self) -> usize {
        let mut count = 0usize;
        if self.model_name.is_some() {
            count += 1;
        }
        if self.architecture.is_some() {
            count += 1;
        }
        if self.author.is_some() {
            count += 1;
        }
        if self.license.is_some() {
            count += 1;
        }
        if self.description.is_some() {
            count += 1;
        }
        if self.context_length.is_some() {
            count += 1;
        }
        if self.embedding_length.is_some() {
            count += 1;
        }
        if self.num_layers.is_some() {
            count += 1;
        }
        if self.num_heads.is_some() {
            count += 1;
        }
        if self.num_kv_heads.is_some() {
            count += 1;
        }
        if self.rope_freq_base.is_some() {
            count += 1;
        }
        if self.vocab_size.is_some() {
            count += 1;
        }
        if self.parameter_count_billions.is_some() {
            count += 1;
        }
        if self.quantization.is_some() {
            count += 1;
        }
        if self.file_size_bytes.is_some() {
            count += 1;
        }
        count
    }

    /// Estimate the parameter count (in billions) from known architecture dimensions
    /// when [`ModelCard::parameter_count_billions`] is not explicitly set.
    ///
    /// The approximation is based on the dominant transformer weight matrices:
    ///
    /// ```text
    /// Per-layer:
    ///   Q  projection: embed × (num_heads × head_dim) ≈ embed²
    ///   K  projection: embed × (kv_heads × head_dim)  ≈ embed × embed * kv_ratio
    ///   V  projection: same as K
    ///   O  projection: embed²
    ///   FFN (up + gate + down): 3 × embed × ffn_dim   ≈ 3 × embed × 2.67 × embed
    ///       (common ratio is ~8/3 ≈ 2.667)
    ///
    /// Embedding table: vocab_size × embed (shared with lm_head)
    /// ```
    ///
    /// Returns `None` when neither `embedding_length` nor `num_layers` is known.
    pub fn estimated_param_count(&self) -> Option<f64> {
        let embed = self.embedding_length? as f64;
        let layers = self.num_layers? as f64;

        // Attention parameters per layer.
        // Q: embed × embed (full)
        let q_params = embed * embed;
        // K+V: if kv_heads known, scale; otherwise assume MHA (same as Q)
        let kv_ratio = if let (Some(kv_h), Some(h)) = (self.num_kv_heads, self.num_heads) {
            if h > 0 {
                kv_h as f64 / h as f64
            } else {
                1.0
            }
        } else {
            1.0
        };
        let kv_params = 2.0 * embed * embed * kv_ratio;
        // O projection: embed × embed
        let o_params = embed * embed;

        // FFN parameters per layer (SwiGLU with 8/3 expansion).
        let ffn_dim = (embed * 8.0 / 3.0).ceil();
        let ffn_params = 3.0 * embed * ffn_dim;

        let per_layer = q_params + kv_params + o_params + ffn_params;
        let total_transformer = layers * per_layer;

        // Embedding table (vocab → embed); shared with lm_head → count once.
        let embed_table = self.vocab_size.unwrap_or(32_000) as f64 * embed;

        let total = total_transformer + embed_table;
        Some(total / 1e9)
    }
}

// ── Public extraction API ────────────────────────────────────────────────────

/// Extract a [`ModelCard`] from a flat `key → value` string metadata map.
///
/// Numeric fields are parsed from their string representations.  Unrecognised
/// keys that do not contain `"."` in the value are stored in
/// [`ModelCard::extra_metadata`] only when they are not already one of the
/// well-known keys processed into typed fields.
pub fn extract_model_card(metadata: &HashMap<String, String>) -> ModelCard {
    let mut card = ModelCard::new();

    // String fields
    card.model_name = metadata.get(keys::MODEL_NAME).cloned();
    card.architecture = metadata.get(keys::ARCHITECTURE).cloned();
    card.author = metadata.get(keys::AUTHOR).cloned();
    card.license = metadata.get(keys::LICENSE).cloned();
    card.description = metadata.get(keys::DESCRIPTION).cloned();
    card.quantization = metadata.get(keys::QUANTIZATION).cloned();

    // u64 fields
    card.context_length = parse_u64(metadata, keys::CONTEXT_LENGTH);
    card.embedding_length = parse_u64(metadata, keys::EMBEDDING_LENGTH);
    card.num_layers = parse_u64(metadata, keys::NUM_LAYERS);
    card.num_heads = parse_u64(metadata, keys::NUM_HEADS);
    card.num_kv_heads = parse_u64(metadata, keys::NUM_KV_HEADS);
    card.vocab_size = parse_u64(metadata, keys::VOCAB_SIZE);
    card.file_size_bytes = parse_u64(metadata, keys::FILE_SIZE);

    // f64 fields
    card.rope_freq_base = parse_f64(metadata, keys::ROPE_FREQ_BASE);

    // Parameter count may be stored as a raw integer (e.g. 7_000_000_000).
    if let Some(raw) = parse_u64(metadata, keys::PARAMETER_COUNT) {
        card.parameter_count_billions = Some(raw as f64 / 1e9);
    } else if let Some(raw) = parse_f64(metadata, keys::PARAMETER_COUNT) {
        // Already expressed as a decimal (rare, but possible).
        card.parameter_count_billions = Some(raw);
    }

    // Collect all unrecognised keys into extra_metadata.
    let known = known_key_set();
    for (k, v) in metadata {
        if !known.contains(k.as_str()) {
            card.extra_metadata.insert(k.clone(), v.clone());
        }
    }

    card
}

/// Return a map containing only the recognised GGUF metadata fields (those
/// whose keys appear in the [`keys`] module) that are present in `metadata`.
pub fn extract_known_fields(metadata: &HashMap<String, String>) -> HashMap<String, String> {
    let known = known_key_set();
    metadata
        .iter()
        .filter(|(k, _)| known.contains(k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// The complete set of key strings defined in the [`keys`] module.
fn known_key_set() -> std::collections::HashSet<&'static str> {
    [
        keys::MODEL_NAME,
        keys::ARCHITECTURE,
        keys::AUTHOR,
        keys::LICENSE,
        keys::DESCRIPTION,
        keys::CONTEXT_LENGTH,
        keys::EMBEDDING_LENGTH,
        keys::NUM_LAYERS,
        keys::NUM_HEADS,
        keys::NUM_KV_HEADS,
        keys::ROPE_FREQ_BASE,
        keys::VOCAB_SIZE,
        keys::QUANTIZATION,
        keys::FILE_SIZE,
        keys::PARAMETER_COUNT,
    ]
    .into_iter()
    .collect()
}

/// Parse a `u64` from the given key in the metadata map.
fn parse_u64(metadata: &HashMap<String, String>, key: &str) -> Option<u64> {
    metadata.get(key)?.trim().parse::<u64>().ok()
}

/// Parse an `f64` from the given key in the metadata map.
fn parse_f64(metadata: &HashMap<String, String>, key: &str) -> Option<f64> {
    metadata.get(key)?.trim().parse::<f64>().ok()
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert(keys::MODEL_NAME.to_owned(), "TestModel-7B".to_owned());
        m.insert(keys::ARCHITECTURE.to_owned(), "llama".to_owned());
        m.insert(keys::CONTEXT_LENGTH.to_owned(), "4096".to_owned());
        m.insert(keys::EMBEDDING_LENGTH.to_owned(), "4096".to_owned());
        m.insert(keys::NUM_LAYERS.to_owned(), "32".to_owned());
        m.insert(keys::NUM_HEADS.to_owned(), "32".to_owned());
        m.insert(keys::VOCAB_SIZE.to_owned(), "32000".to_owned());
        m
    }

    #[test]
    fn model_card_new_is_empty() {
        let card = ModelCard::new();
        assert!(card.is_empty());
    }

    #[test]
    fn populated_count_tracks_set_fields() {
        let mut card = ModelCard::new();
        assert_eq!(card.populated_count(), 0);
        card.model_name = Some("X".to_owned());
        assert_eq!(card.populated_count(), 1);
        card.architecture = Some("llama".to_owned());
        assert_eq!(card.populated_count(), 2);
        card.num_layers = Some(32);
        assert_eq!(card.populated_count(), 3);
    }

    #[test]
    fn markdown_empty_card_is_nonempty_string() {
        let card = ModelCard::new();
        let md = card.to_markdown();
        assert!(!md.is_empty());
    }

    #[test]
    fn markdown_contains_model_name() {
        let mut card = ModelCard::new();
        card.model_name = Some("Llama-3-8B".to_owned());
        let md = card.to_markdown();
        assert!(
            md.contains("Llama-3-8B"),
            "markdown must contain the model name"
        );
    }

    #[test]
    fn markdown_starts_with_heading() {
        let card = ModelCard::new();
        assert!(card.to_markdown().starts_with("# "));
    }

    #[test]
    fn summary_empty_card_is_nonempty() {
        let card = ModelCard::new();
        let s = card.to_summary();
        assert!(!s.is_empty());
    }

    #[test]
    fn summary_contains_known_fields() {
        let metadata = sample_metadata();
        let card = extract_model_card(&metadata);
        let s = card.to_summary();
        assert!(s.contains("TestModel-7B"));
        assert!(s.contains("llama"));
        assert!(s.contains("4096"));
    }

    #[test]
    fn extract_model_card_parses_name() {
        let metadata = sample_metadata();
        let card = extract_model_card(&metadata);
        assert_eq!(card.model_name.as_deref(), Some("TestModel-7B"));
    }

    #[test]
    fn extract_model_card_parses_architecture() {
        let metadata = sample_metadata();
        let card = extract_model_card(&metadata);
        assert_eq!(card.architecture.as_deref(), Some("llama"));
    }

    #[test]
    fn extract_model_card_parses_context_length() {
        let metadata = sample_metadata();
        let card = extract_model_card(&metadata);
        assert_eq!(card.context_length, Some(4096));
    }

    #[test]
    fn extract_model_card_empty_metadata_gives_empty_card() {
        let card = extract_model_card(&HashMap::new());
        assert!(card.is_empty());
    }

    #[test]
    fn extract_known_fields_identifies_known() {
        let mut metadata = sample_metadata();
        metadata.insert("unknown.custom.key".to_owned(), "value".to_owned());
        let known = extract_known_fields(&metadata);
        assert!(known.contains_key(keys::MODEL_NAME));
        assert!(known.contains_key(keys::ARCHITECTURE));
        assert!(!known.contains_key("unknown.custom.key"));
    }

    #[test]
    fn estimated_param_count_returns_some_when_dims_known() {
        let mut card = ModelCard::new();
        card.embedding_length = Some(4096);
        card.num_layers = Some(32);
        card.num_heads = Some(32);
        card.num_kv_heads = Some(32);
        card.vocab_size = Some(32_000);
        let est = card.estimated_param_count();
        assert!(est.is_some(), "must return Some when dims are known");
        let b = est.expect("checked above");
        // A 4096/32-layer model is roughly 7B — allow a wide range.
        assert!(
            b > 1.0 && b < 50.0,
            "estimate {b:.2}B out of plausible range"
        );
    }
}

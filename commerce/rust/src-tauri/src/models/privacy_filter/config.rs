/// Model configuration parsed from HuggingFace `config.json`.

use std::collections::HashMap;
use std::path::Path;

// ── ModelConfig ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub sliding_window: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default)]
    pub pad_token_id: Option<usize>,
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub id2label: Option<HashMap<String, String>>,
    #[serde(default)]
    pub label2id: Option<HashMap<String, usize>>,
}

fn default_attention_bias() -> bool {
    true
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_factor")]
    pub factor: f64,
    #[serde(default = "default_beta_fast")]
    pub beta_fast: f64,
    #[serde(default = "default_beta_slow")]
    pub beta_slow: f64,
    #[serde(default = "default_original_max_pos")]
    pub original_max_position_embeddings: usize,
    #[serde(default)]
    pub truncate: bool,
}

fn default_rope_type() -> String { "yarn".to_string() }
fn default_rope_theta() -> f64 { 150_000.0 }
fn default_factor() -> f64 { 32.0 }
fn default_beta_fast() -> f64 { 32.0 }
fn default_beta_slow() -> f64 { 1.0 }
fn default_original_max_pos() -> usize { 4096 }

impl ModelConfig {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Self = serde_json::from_str(&text)?;
        Ok(cfg)
    }

    /// Number of query heads per KV head group.
    pub fn num_key_value_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Total number of output labels (33 for BIOES over 8 categories + O).
    pub fn num_labels(&self) -> usize {
        if let Some(ref id2l) = self.id2label {
            id2l.len()
        } else {
            33
        }
    }
}

// ── ViterbiConfig ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ViterbiConfig {
    pub transition_bias_background_stay: f64,
    pub transition_bias_background_to_start: f64,
    pub transition_bias_inside_to_continue: f64,
    pub transition_bias_inside_to_end: f64,
    pub transition_bias_end_to_background: f64,
    pub transition_bias_end_to_start: f64,
}

impl Default for ViterbiConfig {
    fn default() -> Self {
        Self {
            transition_bias_background_stay: 0.0,
            transition_bias_background_to_start: 0.0,
            transition_bias_inside_to_continue: 0.0,
            transition_bias_inside_to_end: 0.0,
            transition_bias_end_to_background: 0.0,
            transition_bias_end_to_start: 0.0,
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct ViterbiCalibrationFile {
    operating_points: HashMap<String, ViterbiOperatingPoint>,
}

#[derive(Debug, serde::Deserialize)]
struct ViterbiOperatingPoint {
    biases: ViterbiBiases,
}

#[derive(Debug, serde::Deserialize)]
struct ViterbiBiases {
    transition_bias_background_stay: f64,
    transition_bias_background_to_start: f64,
    transition_bias_inside_to_continue: f64,
    transition_bias_inside_to_end: f64,
    transition_bias_end_to_background: f64,
    transition_bias_end_to_start: f64,
}

impl ViterbiConfig {
    pub fn from_file(path: &Path, operating_point: &str) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cal: ViterbiCalibrationFile = serde_json::from_str(&text)?;
        let op = cal.operating_points.get(operating_point)
            .ok_or_else(|| anyhow::anyhow!("operating point '{}' not found", operating_point))?;
        Ok(Self {
            transition_bias_background_stay: op.biases.transition_bias_background_stay,
            transition_bias_background_to_start: op.biases.transition_bias_background_to_start,
            transition_bias_inside_to_continue: op.biases.transition_bias_inside_to_continue,
            transition_bias_inside_to_end: op.biases.transition_bias_inside_to_end,
            transition_bias_end_to_background: op.biases.transition_bias_end_to_background,
            transition_bias_end_to_start: op.biases.transition_bias_end_to_start,
        })
    }
}

// ── Label helpers ────────────────────────────────────────────────────────────

/// The 8 span categories in order.
pub const SPAN_LABELS: &[&str] = &[
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
];

/// BIOES tag prefixes.
pub const BIOES_PREFIXES: &[&str] = &["B", "I", "E", "S"];

/// Build the full 33-label list: O, B-account_number, I-account_number, E-account_number, S-account_number, ...
pub fn build_label_list() -> Vec<String> {
    let mut labels = vec!["O".to_string()];
    for &cat in SPAN_LABELS {
        for &prefix in BIOES_PREFIXES {
            labels.push(format!("{}-{}", prefix, cat));
        }
    }
    labels
}

/// Get the span category from a label index (0 = O, 1-4 = account_number, etc.)
pub fn label_to_category(label_idx: usize) -> Option<&'static str> {
    if label_idx == 0 {
        return None; // O (background)
    }
    let cat_idx = (label_idx - 1) / 4;
    SPAN_LABELS.get(cat_idx).copied()
}

/// Get the BIOES prefix for a label index.
pub fn label_to_prefix(label_idx: usize) -> Option<&'static str> {
    if label_idx == 0 {
        return Some("O");
    }
    let prefix_idx = (label_idx - 1) % 4;
    BIOES_PREFIXES.get(prefix_idx).copied()
}

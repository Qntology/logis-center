//! WASM-compatible inference API.
//!
//! Provides a JSON-in / JSON-out interface for WASM hosts (wasmtime, wasmer,
//! browser environments, etc.). No `wasm-bindgen` required — this module works
//! with any WASM runtime that can call exported functions with string arguments.
//!
//! ## Request format
//!
//! ```json
//! {
//!   "hidden_size": 4096,
//!   "num_layers": 32,
//!   "num_attention_heads": 32,
//!   "num_kv_heads": 8,
//!   "intermediate_size": 14336,
//!   "vocab_size": 151936,
//!   "max_context_length": 32768,
//!   "rms_norm_eps": 1e-6,
//!   "rope_theta": 1000000.0,
//!   "head_dim": 128,
//!   "prompt_tokens": [151644, 872, 151645],
//!   "max_tokens": 32,
//!   "temperature": 0.7,
//!   "top_k": 40,
//!   "top_p": 0.9,
//!   "seed": 42
//! }
//! ```
//!
//! ## Response format (success)
//!
//! ```json
//! { "tokens": [1234, 5678, ...], "error": null }
//! ```
//!
//! ## Response format (error)
//!
//! ```json
//! { "tokens": [], "error": "description of the error" }
//! ```

use crate::engine::InferenceEngine;
use crate::sampling::SamplingParams;
use oxibonsai_core::config::Qwen3Config;

// ─── Request / Response types ────────────────────────────────────────────────

/// JSON request for WASM inference.
#[derive(serde::Deserialize, Debug)]
struct WasmInferenceRequest {
    // ── Model architecture (matches Qwen3Config fields) ──────────────────
    hidden_size: usize,
    num_layers: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_context_length: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    head_dim: usize,

    // ── Inference parameters ──────────────────────────────────────────────
    /// Prompt as a list of token IDs.
    prompt_tokens: Vec<u32>,
    /// Maximum number of tokens to generate.
    max_tokens: usize,

    // ── Sampling parameters (all optional with sensible defaults) ─────────
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default = "default_seed")]
    seed: u64,
}

fn default_temperature() -> f32 {
    0.7
}
fn default_top_k() -> usize {
    40
}
fn default_top_p() -> f32 {
    0.9
}
fn default_seed() -> u64 {
    42
}

/// JSON response from WASM inference.
#[derive(serde::Serialize, Debug)]
struct WasmInferenceResponse {
    /// Generated token IDs (empty on error).
    tokens: Vec<u32>,
    /// Error message, or `null` on success.
    error: Option<String>,
}

impl WasmInferenceResponse {
    fn success(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            error: None,
        }
    }

    fn error(msg: impl Into<String>) -> Self {
        Self {
            tokens: vec![],
            error: Some(msg.into()),
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Run inference from a JSON request string, returning a JSON response string.
///
/// This is the primary entry point for WASM hosts. It is synchronous and
/// self-contained: creates an engine, runs generation, and returns results.
///
/// All configuration is passed via JSON — no file I/O or mmap required,
/// making this fully compatible with wasm32-unknown-unknown.
///
/// # Example (Rust)
///
/// ```no_run
/// let req = r#"{
///   "hidden_size": 256, "num_layers": 2, "num_attention_heads": 4,
///   "num_kv_heads": 2, "intermediate_size": 512, "vocab_size": 1024,
///   "max_context_length": 512, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
///   "head_dim": 64, "prompt_tokens": [1, 2, 3], "max_tokens": 5
/// }"#;
/// let resp = oxibonsai_runtime::wasm_api::generate_json(req);
/// // resp is a JSON string like {"tokens":[...],"error":null}
/// ```
pub fn generate_json(request_json: &str) -> String {
    let response = match run_inference(request_json) {
        Ok(tokens) => WasmInferenceResponse::success(tokens),
        Err(e) => WasmInferenceResponse::error(e),
    };

    match serde_json::to_string(&response) {
        Ok(s) => s,
        Err(e) => format!(r#"{{"tokens":[],"error":"failed to serialize response: {e}"}}"#),
    }
}

/// Run inference, returning generated tokens or an error description.
fn run_inference(request_json: &str) -> Result<Vec<u32>, String> {
    let req: WasmInferenceRequest =
        serde_json::from_str(request_json).map_err(|e| format!("invalid request JSON: {e}"))?;

    let config = Qwen3Config {
        hidden_size: req.hidden_size,
        num_layers: req.num_layers,
        num_attention_heads: req.num_attention_heads,
        num_kv_heads: req.num_kv_heads,
        intermediate_size: req.intermediate_size,
        vocab_size: req.vocab_size,
        max_context_length: req.max_context_length,
        rms_norm_eps: req.rms_norm_eps,
        rope_freq_base: req.rope_theta,
        head_dim: req.head_dim,
        architecture: "qwen3".to_string(),
        model_name: "bonsai".to_string(),
    };

    let sampling = SamplingParams {
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        repetition_penalty: 1.1,
        ..SamplingParams::default()
    };

    let mut engine = InferenceEngine::new(config, sampling, req.seed);

    engine
        .generate(&req.prompt_tokens, req.max_tokens)
        .map_err(|e| format!("inference error: {e}"))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config_json(prompt_tokens: &[u32], max_tokens: usize) -> String {
        let tokens_json = serde_json::to_string(prompt_tokens).expect("serialize tokens");
        format!(
            r#"{{
              "hidden_size": 256,
              "num_layers": 2,
              "num_attention_heads": 4,
              "num_kv_heads": 2,
              "intermediate_size": 512,
              "vocab_size": 1024,
              "max_context_length": 128,
              "rms_norm_eps": 1e-6,
              "rope_theta": 10000.0,
              "head_dim": 64,
              "prompt_tokens": {tokens_json},
              "max_tokens": {max_tokens}
            }}"#
        )
    }

    #[test]
    fn generate_json_empty_prompt_returns_empty_tokens() {
        let req = tiny_config_json(&[], 5);
        let resp_str = generate_json(&req);
        let resp: serde_json::Value = serde_json::from_str(&resp_str).expect("valid JSON response");
        assert!(resp["error"].is_null(), "expected no error, got: {resp}");
        let tokens = resp["tokens"].as_array().expect("tokens array");
        assert!(tokens.is_empty(), "empty prompt should yield no tokens");
    }

    #[test]
    fn generate_json_invalid_json_returns_error() {
        let resp_str = generate_json("this is not json");
        let resp: serde_json::Value =
            serde_json::from_str(&resp_str).expect("response should be valid JSON");
        assert!(
            !resp["error"].is_null(),
            "invalid input should produce an error"
        );
    }

    #[test]
    fn generate_json_missing_required_field_returns_error() {
        // Missing hidden_size
        let req = r#"{"num_layers": 2, "prompt_tokens": [1], "max_tokens": 1}"#;
        let resp_str = generate_json(req);
        let resp: serde_json::Value =
            serde_json::from_str(&resp_str).expect("response should be valid JSON");
        assert!(
            !resp["error"].is_null(),
            "missing fields should produce an error"
        );
    }

    #[test]
    fn response_serialization_success() {
        let r = WasmInferenceResponse::success(vec![1, 2, 3]);
        let s = serde_json::to_string(&r).expect("serialize");
        assert!(s.contains("\"tokens\":[1,2,3]"));
        assert!(s.contains("\"error\":null"));
    }

    #[test]
    fn response_serialization_error() {
        let r = WasmInferenceResponse::error("something went wrong");
        let s = serde_json::to_string(&r).expect("serialize");
        assert!(s.contains("\"tokens\":[]"));
        assert!(s.contains("\"error\":\"something went wrong\""));
    }
}

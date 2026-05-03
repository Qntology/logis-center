//! Classify ONNX initializers and MatMulNBits nodes by the Qwen3 tensor role
//! they represent, and emit the matching GGUF tensor name.
//!
//! HuggingFace's ONNX export for Qwen3 uses a few different naming
//! conventions depending on the exporter. OxiBonsai's importer currently
//! handles two:
//!
//! * Classic HF convention (as used by the safetensors export):
//!   `model.layers.0.self_attn.q_proj.weight`,
//!   `model.layers.0.self_attn.q_norm.weight`,
//!   `model.norm.weight`, etc.
//! * The `onnx-community` convention emitted by transformers.js-style ONNX
//!   exports, where per-layer sub-modules are named `attn` instead of
//!   `self_attn`, Q/K-norms live under `attn.q_norm.layernorm.weight`, and
//!   the final model norm is re-parented as
//!   `model.layers.{num_hidden_layers}.final_norm_layernorm.weight`.
//!
//! MatMulNBits input initializers in the `onnx-community` export use
//! underscored names with `_quant` / `_scales` / `_zp` suffixes, e.g.
//! `model_layers_0_attn_q_proj_MatMul_weight_quant`. The importer does
//! *not* rely on those suffixes to wire up tensors — the authoritative
//! source is the MatMulNBits **node name**
//! (e.g. `/model/layers.0/attn/q_proj/MatMul_Quant`). The initializer-name
//! suffixes are retained only as informational classifications for debugging.
//!
//! Four classification families are used:
//!
//! * [`OnnxRole::EmbeddingFp`] — `model.embed_tokens.weight` stored as
//!   full-precision f32/f16/bf16 (unused by the `onnx-community` export,
//!   kept for other conventions).
//! * [`OnnxRole::NormFp`] — RMS-norm scale tensors stored in f32/f16/bf16.
//! * [`OnnxRole::MatMulPacked`] / [`OnnxRole::MatMulScales`] /
//!   [`OnnxRole::MatMulZeroPoints`] — the three MatMulNBits inputs (for
//!   either the old `_quantized` / `_scales` / `_zero_points` style or the
//!   new `_quant` / `_scales` / `_zp` style). Looked up through the
//!   MatMulNBits node rather than by name; classification is informational.
//! * [`OnnxRole::LmHeadFp`] — `lm_head.weight` stored full-precision
//!   (present only when `tie_word_embeddings = false` on some exports).

use super::error::OnnxImportError;

/// The role an ONNX initializer plays in a Qwen3 model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnnxRole {
    /// Full-precision embedding table `model.embed_tokens.weight`.
    EmbeddingFp,
    /// Full-precision LM head `lm_head.weight`.
    LmHeadFp,
    /// Full-precision RMSNorm scale (either global or per-layer).
    NormFp {
        /// Destination GGUF tensor name (e.g. `"output_norm.weight"` or
        /// `"blk.0.attn_norm.weight"`).
        gguf_name: String,
    },
    /// Packed 2-bit MatMulNBits weight tensor (`*_quantized` or `*_quant`).
    MatMulPacked {
        /// Suspected base name without the recognised suffix.
        base: String,
    },
    /// MatMulNBits scales tensor (`*_scales`).
    MatMulScales {
        /// Suspected base name without the recognised suffix.
        base: String,
    },
    /// MatMulNBits zero-points tensor (`*_zero_points` or `*_zp`).
    MatMulZeroPoints {
        /// Suspected base name without the recognised suffix.
        base: String,
    },
}

/// Classify an initializer by its name, taking the model's
/// `num_hidden_layers` into account so the phantom final-norm layer
/// (index == `num_hidden_layers`) can be distinguished from real per-layer
/// norms.
///
/// Returns `None` when the name is unrecognised (e.g. opset-version
/// constants, KV-cache scratch buffers, cos/sin RoPE caches).
pub fn classify_initializer(name: &str, num_hidden_layers: usize) -> Option<OnnxRole> {
    // ── Top-level FP tensors (classic HF names) ──────────────────────
    if name == "model.embed_tokens.weight" {
        return Some(OnnxRole::EmbeddingFp);
    }
    if name == "lm_head.weight" {
        return Some(OnnxRole::LmHeadFp);
    }
    if name == "model.norm.weight" {
        return Some(OnnxRole::NormFp {
            gguf_name: "output_norm.weight".to_string(),
        });
    }

    // ── Per-layer norm tensors ───────────────────────────────────────
    if let Some(gguf_name) = layer_norm_gguf_name(name, num_hidden_layers) {
        return Some(OnnxRole::NormFp { gguf_name });
    }

    // ── MatMulNBits packed inputs (weight/scales/zero-points) ────────
    // Accept both the classic HF suffixes and the onnx-community short
    // suffixes. Order matters: `_zero_points` must be checked before
    // `_zp` only for correctness clarity (no prefix overlap exists, but we
    // still prefer the longer, more specific suffix first).
    if let Some(base) = name
        .strip_suffix("_quantized")
        .or_else(|| name.strip_suffix("_quant"))
    {
        return Some(OnnxRole::MatMulPacked {
            base: base.to_string(),
        });
    }
    if let Some(base) = name.strip_suffix("_scales") {
        return Some(OnnxRole::MatMulScales {
            base: base.to_string(),
        });
    }
    if let Some(base) = name
        .strip_suffix("_zero_points")
        .or_else(|| name.strip_suffix("_zp"))
    {
        return Some(OnnxRole::MatMulZeroPoints {
            base: base.to_string(),
        });
    }

    None
}

/// Return the GGUF name for a per-layer norm, or `None` if `name` is not a
/// per-layer norm.
///
/// Handles:
///   * `model.layers.N.input_layernorm.weight` → `blk.N.attn_norm.weight`
///   * `model.layers.N.post_attention_layernorm.weight` → `blk.N.ffn_norm.weight`
///   * `model.layers.N.self_attn.q_norm.weight` → `blk.N.attn_q_norm.weight` (HF)
///   * `model.layers.N.self_attn.k_norm.weight` → `blk.N.attn_k_norm.weight` (HF)
///   * `model.layers.N.attn.q_norm.layernorm.weight` → `blk.N.attn_q_norm.weight` (onnx-community)
///   * `model.layers.N.attn.k_norm.layernorm.weight` → `blk.N.attn_k_norm.weight` (onnx-community)
///   * `model.layers.{num_hidden_layers}.final_norm_layernorm.weight` →
///     `output_norm.weight` (onnx-community phantom final layer)
fn layer_norm_gguf_name(name: &str, num_hidden_layers: usize) -> Option<String> {
    let rest = name.strip_prefix("model.layers.")?;
    let dot_pos = rest.find('.')?;
    let (layer_str, suffix_with_dot) = rest.split_at(dot_pos);
    let layer_idx: usize = layer_str.parse().ok()?;
    let suffix = suffix_with_dot.strip_prefix('.')?;

    // The phantom "layer" == num_hidden_layers holds the final model norm.
    if layer_idx == num_hidden_layers && suffix == "final_norm_layernorm.weight" {
        return Some("output_norm.weight".to_string());
    }

    if layer_idx >= num_hidden_layers {
        return None;
    }

    let gguf_suffix = match suffix {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "attn.q_norm.layernorm.weight" => "attn_q_norm.weight",
        "attn.k_norm.layernorm.weight" => "attn_k_norm.weight",
        _ => return None,
    };
    Some(format!("blk.{layer_idx}.{gguf_suffix}"))
}

/// Convert a MatMulNBits **node name** to its target GGUF tensor name.
///
/// Expected input shapes (examples):
///   * `/model/layers.0/attn/q_proj/MatMul_Quant`
///   * `/model/layers.27/mlp/down_proj/MatMul_Quant`
///   * `/lm_head/MatMul_Quant`
///
/// Returns an `OnnxImportError::Other` if the node name does not match any
/// recognised layout.
pub fn matmul_node_to_gguf(node_name: &str) -> Result<String, OnnxImportError> {
    // Accept a trailing `/MatMul_Quant` or `/MatMul` marker (different
    // exporters use slightly different suffixes).
    let trimmed = node_name.trim_start_matches('/');
    let parts: Vec<&str> = trimmed.split('/').collect();

    // `lm_head/MatMul_Quant` (or `lm_head/MatMul`) → output.weight
    if parts.len() >= 2 && parts[0] == "lm_head" && is_matmul_marker(parts[1]) {
        return Ok("output.weight".to_string());
    }

    // `model/layers.N/{attn|mlp}/<proj>/MatMul_Quant`
    if parts.len() >= 5 && parts[0] == "model" && is_matmul_marker(parts[4]) {
        let layer_idx = parts[1]
            .strip_prefix("layers.")
            .ok_or_else(|| {
                OnnxImportError::Other(format!(
                    "MatMulNBits node '{node_name}' parts[1] '{}' does not start with 'layers.'",
                    parts[1]
                ))
            })?
            .parse::<usize>()
            .map_err(|e| {
                OnnxImportError::Other(format!(
                    "MatMulNBits node '{node_name}' has unparseable layer index '{}': {e}",
                    parts[1]
                ))
            })?;

        let group = parts[2];
        let proj = parts[3];

        let gguf_suffix = match (group, proj) {
            ("attn", "q_proj") => "attn_q.weight",
            ("attn", "k_proj") => "attn_k.weight",
            ("attn", "v_proj") => "attn_v.weight",
            ("attn", "o_proj") => "attn_output.weight",
            ("self_attn", "q_proj") => "attn_q.weight",
            ("self_attn", "k_proj") => "attn_k.weight",
            ("self_attn", "v_proj") => "attn_v.weight",
            ("self_attn", "o_proj") => "attn_output.weight",
            ("mlp", "gate_proj") => "ffn_gate.weight",
            ("mlp", "up_proj") => "ffn_up.weight",
            ("mlp", "down_proj") => "ffn_down.weight",
            _ => {
                return Err(OnnxImportError::Other(format!(
                    "MatMulNBits node '{node_name}' has unknown (group, projection) = ('{group}', '{proj}')"
                )));
            }
        };

        return Ok(format!("blk.{layer_idx}.{gguf_suffix}"));
    }

    Err(OnnxImportError::Other(format!(
        "MatMulNBits node '{node_name}' does not match any recognised layout (expected /model/layers.N/{{attn|mlp}}/<proj>/MatMul_Quant or /lm_head/MatMul_Quant)"
    )))
}

/// Return true if a node-path segment looks like a MatMul marker suffix
/// (`MatMul`, `MatMul_Quant`, `MatMul_Q`, etc.). We accept the common
/// variants rather than force a single form.
fn is_matmul_marker(segment: &str) -> bool {
    segment == "MatMul_Quant" || segment == "MatMul" || segment.starts_with("MatMul_")
}

#[cfg(test)]
mod tests {
    use super::*;

    const NUM_LAYERS: usize = 28;

    #[test]
    fn classify_top_level_hf() {
        assert_eq!(
            classify_initializer("model.embed_tokens.weight", NUM_LAYERS),
            Some(OnnxRole::EmbeddingFp)
        );
        assert_eq!(
            classify_initializer("lm_head.weight", NUM_LAYERS),
            Some(OnnxRole::LmHeadFp)
        );
        assert_eq!(
            classify_initializer("model.norm.weight", NUM_LAYERS),
            Some(OnnxRole::NormFp {
                gguf_name: "output_norm.weight".to_string()
            })
        );
    }

    #[test]
    fn classify_onnx_community_final_norm() {
        // onnx-community phantom final layer (index == num_hidden_layers).
        assert_eq!(
            classify_initializer("model.layers.28.final_norm_layernorm.weight", NUM_LAYERS),
            Some(OnnxRole::NormFp {
                gguf_name: "output_norm.weight".to_string()
            })
        );
        // With a different num_hidden_layers, the same name is unrecognised.
        assert_eq!(
            classify_initializer("model.layers.28.final_norm_layernorm.weight", 24),
            None
        );
    }

    #[test]
    fn classify_per_layer_norms_hf() {
        assert_eq!(
            classify_initializer("model.layers.3.input_layernorm.weight", NUM_LAYERS),
            Some(OnnxRole::NormFp {
                gguf_name: "blk.3.attn_norm.weight".to_string()
            })
        );
        assert_eq!(
            classify_initializer(
                "model.layers.27.post_attention_layernorm.weight",
                NUM_LAYERS
            ),
            Some(OnnxRole::NormFp {
                gguf_name: "blk.27.ffn_norm.weight".to_string()
            })
        );
        assert_eq!(
            classify_initializer("model.layers.5.self_attn.q_norm.weight", NUM_LAYERS),
            Some(OnnxRole::NormFp {
                gguf_name: "blk.5.attn_q_norm.weight".to_string()
            })
        );
        assert_eq!(
            classify_initializer("model.layers.5.self_attn.k_norm.weight", NUM_LAYERS),
            Some(OnnxRole::NormFp {
                gguf_name: "blk.5.attn_k_norm.weight".to_string()
            })
        );
    }

    #[test]
    fn classify_per_layer_norms_onnx_community() {
        // onnx-community: attn.q_norm.layernorm.weight (note extra ".layernorm.")
        assert_eq!(
            classify_initializer("model.layers.0.attn.q_norm.layernorm.weight", NUM_LAYERS),
            Some(OnnxRole::NormFp {
                gguf_name: "blk.0.attn_q_norm.weight".to_string()
            })
        );
        assert_eq!(
            classify_initializer("model.layers.13.attn.k_norm.layernorm.weight", NUM_LAYERS),
            Some(OnnxRole::NormFp {
                gguf_name: "blk.13.attn_k_norm.weight".to_string()
            })
        );
    }

    #[test]
    fn classify_matmul_nbits_suffixes_classic_hf() {
        let base = "model.layers.0.self_attn.q_proj.weight";
        assert_eq!(
            classify_initializer(&format!("{base}_quantized"), NUM_LAYERS),
            Some(OnnxRole::MatMulPacked {
                base: base.to_string()
            })
        );
        assert_eq!(
            classify_initializer(&format!("{base}_scales"), NUM_LAYERS),
            Some(OnnxRole::MatMulScales {
                base: base.to_string()
            })
        );
        assert_eq!(
            classify_initializer(&format!("{base}_zero_points"), NUM_LAYERS),
            Some(OnnxRole::MatMulZeroPoints {
                base: base.to_string()
            })
        );
    }

    #[test]
    fn classify_matmul_nbits_suffixes_onnx_community() {
        let base = "model_layers_0_attn_q_proj_MatMul_weight";
        assert_eq!(
            classify_initializer(&format!("{base}_quant"), NUM_LAYERS),
            Some(OnnxRole::MatMulPacked {
                base: base.to_string()
            })
        );
        assert_eq!(
            classify_initializer(&format!("{base}_scales"), NUM_LAYERS),
            Some(OnnxRole::MatMulScales {
                base: base.to_string()
            })
        );
        assert_eq!(
            classify_initializer(&format!("{base}_zp"), NUM_LAYERS),
            Some(OnnxRole::MatMulZeroPoints {
                base: base.to_string()
            })
        );
    }

    #[test]
    fn matmul_node_to_gguf_onnx_community_names() {
        assert_eq!(
            matmul_node_to_gguf("/model/layers.0/attn/q_proj/MatMul_Quant")
                .expect("valid onnx-community q_proj node should map to blk.0.attn_q.weight"),
            "blk.0.attn_q.weight"
        );
        assert_eq!(
            matmul_node_to_gguf("/model/layers.12/mlp/down_proj/MatMul_Quant")
                .expect("valid onnx-community down_proj node should map to blk.12.ffn_down.weight"),
            "blk.12.ffn_down.weight"
        );
        assert_eq!(
            matmul_node_to_gguf("/model/layers.27/attn/o_proj/MatMul_Quant")
                .expect("valid onnx-community o_proj node should map to blk.27.attn_output.weight"),
            "blk.27.attn_output.weight"
        );
        assert_eq!(
            matmul_node_to_gguf("/lm_head/MatMul_Quant")
                .expect("valid lm_head MatMul_Quant node should map to output.weight"),
            "output.weight"
        );
    }

    #[test]
    fn matmul_node_to_gguf_accepts_classic_self_attn_path() {
        // Some exporters keep `self_attn` in the node path.
        assert_eq!(
            matmul_node_to_gguf("/model/layers.5/self_attn/q_proj/MatMul_Quant")
                .expect("self_attn variant q_proj node should map to blk.5.attn_q.weight"),
            "blk.5.attn_q.weight"
        );
    }

    #[test]
    fn matmul_node_to_gguf_accepts_plain_matmul_suffix() {
        // `/lm_head/MatMul` without the `_Quant` suffix.
        assert_eq!(
            matmul_node_to_gguf("/lm_head/MatMul")
                .expect("plain MatMul suffix on lm_head should map to output.weight"),
            "output.weight"
        );
        assert_eq!(
            matmul_node_to_gguf("/model/layers.0/attn/q_proj/MatMul")
                .expect("plain MatMul suffix on q_proj node should map to blk.0.attn_q.weight"),
            "blk.0.attn_q.weight"
        );
    }

    #[test]
    fn matmul_node_to_gguf_rejects_unknown_layout() {
        let err = matmul_node_to_gguf("/foo/bar/MatMul_Quant").unwrap_err();
        match err {
            OnnxImportError::Other(msg) => assert!(msg.contains("does not match")),
            _ => panic!("expected Other, got {:?}", err),
        }

        let err =
            matmul_node_to_gguf("/model/layers.0/attn/unknown_proj/MatMul_Quant").unwrap_err();
        match err {
            OnnxImportError::Other(msg) => assert!(msg.contains("unknown")),
            _ => panic!("expected Other, got {:?}", err),
        }
    }

    #[test]
    fn classify_unknown_name_is_none() {
        assert_eq!(classify_initializer("opset_version", NUM_LAYERS), None);
        assert_eq!(
            classify_initializer("past_key_values.0.key", NUM_LAYERS),
            None
        );
        assert_eq!(classify_initializer("cos_cache", NUM_LAYERS), None);
        assert_eq!(classify_initializer("sin_cache", NUM_LAYERS), None);
    }
}

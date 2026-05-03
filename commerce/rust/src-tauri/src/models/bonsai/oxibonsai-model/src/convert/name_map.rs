//! HuggingFace safetensors → GGUF tensor name mapping for Qwen3 architectures.
//!
//! All norm tensors (those whose GGUF name ends in `_norm.weight`) are stored
//! as FP32.  All other weights use the requested quantisation format (e.g.
//! TQ2_0_g128).

/// Result of mapping a single HuggingFace tensor name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameMapResult {
    /// Target GGUF tensor name (e.g. `"blk.0.attn_q.weight"`).
    pub gguf_name: String,
    /// Whether this tensor should be stored as FP32 (norm weights).
    pub is_norm: bool,
}

/// Map a HuggingFace safetensors tensor name to its GGUF counterpart.
///
/// Returns `None` for tensors that should be skipped (e.g. unknown names or
/// purely HF bookkeeping tensors).
pub fn hf_to_gguf_name(hf_name: &str) -> Option<NameMapResult> {
    // ── Top-level tensors ────────────────────────────────────────────────────
    if hf_name == "model.embed_tokens.weight" {
        return Some(NameMapResult {
            gguf_name: "token_embd.weight".to_string(),
            is_norm: false,
        });
    }
    if hf_name == "model.norm.weight" {
        return Some(NameMapResult {
            gguf_name: "output_norm.weight".to_string(),
            is_norm: true,
        });
    }
    if hf_name == "lm_head.weight" {
        return Some(NameMapResult {
            gguf_name: "output.weight".to_string(),
            is_norm: false,
        });
    }

    // ── Per-layer tensors: model.layers.N.<suffix> ───────────────────────────
    let layer_prefix = "model.layers.";
    if let Some(rest) = hf_name.strip_prefix(layer_prefix) {
        // Split at the first '.' after the layer index digit(s)
        let dot_pos = rest.find('.')?;
        let (layer_str, suffix_with_dot) = rest.split_at(dot_pos);
        let layer_idx: usize = layer_str.parse().ok()?;
        let suffix = suffix_with_dot.strip_prefix('.')?;

        let (gguf_suffix, is_norm) = match suffix {
            "input_layernorm.weight" => ("attn_norm.weight", true),
            "post_attention_layernorm.weight" => ("ffn_norm.weight", true),
            "self_attn.q_proj.weight" => ("attn_q.weight", false),
            "self_attn.k_proj.weight" => ("attn_k.weight", false),
            "self_attn.v_proj.weight" => ("attn_v.weight", false),
            "self_attn.o_proj.weight" => ("attn_output.weight", false),
            "mlp.gate_proj.weight" => ("ffn_gate.weight", false),
            "mlp.up_proj.weight" => ("ffn_up.weight", false),
            "mlp.down_proj.weight" => ("ffn_down.weight", false),
            "self_attn.q_norm.weight" => ("attn_q_norm.weight", true),
            "self_attn.k_norm.weight" => ("attn_k_norm.weight", true),
            _ => return None,
        };

        return Some(NameMapResult {
            gguf_name: format!("blk.{layer_idx}.{gguf_suffix}"),
            is_norm,
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_name_maps_correctly() {
        // Top-level embeddings
        let r = hf_to_gguf_name("model.embed_tokens.weight").expect("known tensor name must map");
        assert_eq!(r.gguf_name, "token_embd.weight");
        assert!(!r.is_norm);

        // lm_head
        let r = hf_to_gguf_name("lm_head.weight").expect("known tensor name must map");
        assert_eq!(r.gguf_name, "output.weight");
        assert!(!r.is_norm);

        // model.norm (FP32 norm)
        let r = hf_to_gguf_name("model.norm.weight").expect("known tensor name must map");
        assert_eq!(r.gguf_name, "output_norm.weight");
        assert!(r.is_norm);

        // Per-layer attention Q projection
        let r = hf_to_gguf_name("model.layers.0.self_attn.q_proj.weight")
            .expect("known tensor name must map");
        assert_eq!(r.gguf_name, "blk.0.attn_q.weight");
        assert!(!r.is_norm);

        // Per-layer FFN gate projection
        let r = hf_to_gguf_name("model.layers.27.mlp.gate_proj.weight")
            .expect("known tensor name must map");
        assert_eq!(r.gguf_name, "blk.27.ffn_gate.weight");
        assert!(!r.is_norm);

        // Per-layer input layernorm (FP32)
        let r = hf_to_gguf_name("model.layers.3.input_layernorm.weight")
            .expect("known tensor name must map");
        assert_eq!(r.gguf_name, "blk.3.attn_norm.weight");
        assert!(r.is_norm);

        // Per-layer attention K norm (FP32)
        let r = hf_to_gguf_name("model.layers.5.self_attn.k_norm.weight")
            .expect("known tensor name must map");
        assert_eq!(r.gguf_name, "blk.5.attn_k_norm.weight");
        assert!(r.is_norm);

        // Unknown tensor should map to None
        assert!(hf_to_gguf_name("some.unknown.tensor").is_none());
    }
}

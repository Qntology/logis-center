// src/utils/config.rs
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MoEConfig {
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: Option<usize>,
    pub num_experts: Option<usize>,
    pub mlp_only_layers: Option<Vec<usize>>,
    pub decoder_sparse_step: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: Option<f64>,
    pub first_k_dense_replace: Option<usize>,
    pub n_shared_experts: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RopeScalingValue {
    Bool(bool),
    Number(f64),
    NumberArray(Vec<f64>),
    String(String),
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct QuantConfig {
    pub quant_method: String,
    #[serde(default)]
    pub bits: usize,
    #[serde(default)]
    pub group_size: i32,
    pub sym: Option<bool>,
    pub desc_act: Option<bool>,
    pub checkpoint_format: Option<String>,
    pub fmt: Option<String>,
    pub weight_block_size: Option<Vec<usize>>,
    #[serde(default, alias = "ignore")]
    pub modules_to_not_convert: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    pub architectures: Option<Vec<String>>,
    pub head_dim: Option<usize>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub max_model_len: Option<usize>,
    #[serde(default, alias = "ffn_hidden_size", alias = "feed_forward_length")]
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: Option<usize>,
    pub rope_theta: Option<f64>,
    pub attention_bias: Option<bool>,
    pub qkv_bias: Option<bool>,
    pub attn_output_gate: Option<bool>,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub tie_word_embeddings: Option<bool>,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<EosTokenId>,
    pub use_sliding_window: Option<bool>,
    pub sliding_window: Option<usize>,
    pub max_window_layers: Option<usize>,
    pub partial_rotary_factor: Option<f32>,
    pub hidden_act: candle_nn::Activation,
    pub rope_scaling: Option<HashMap<String, RopeScalingValue>>,
    pub quant: Option<String>,
    pub moe_cfg: Option<MoEConfig>,
    pub fp8_kvcache: Option<bool>,
    pub quantization_config: Option<QuantConfig>,
    pub is_multi_model: Option<bool>,
    pub extra_config_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Qwen3HybridRawConfig {
    #[serde(alias = "layers_block_type")]
    pub layers_block_type: Option<Vec<String>>,
    #[serde(alias = "linear_conv_kernel_dim")]
    pub conv_kernel_size: Option<usize>,
    pub full_attention_interval: Option<usize>,
    pub linear_num_heads: Option<usize>,
    #[serde(alias = "linear_num_key_heads")]
    pub linear_num_key_heads: Option<usize>,
    #[serde(alias = "linear_num_value_heads")]
    pub linear_num_value_heads: Option<usize>,
    pub linear_num_key_value_heads: Option<usize>,
    pub linear_key_head_dim: Option<usize>,
    pub linear_value_head_dim: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct Qwen3HybridConfig {
    pub layer_types: Vec<String>,
    pub conv_kernel_size: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
}

pub fn is_qwen3_hybrid_arch_name(arch: &str) -> bool {
    matches!(
        arch,
        "Qwen3_5ForCausalLM"
            | "Qwen3_5MoeForCausalLM"
            | "Qwen3NextForCausalLM"
            | "Qwen3_5ForConditionalGeneration"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3NextForConditionalGeneration"
    )
}

fn is_qwen3_hybrid_arch(config: &Config) -> bool {
    let arch = config.architectures.as_ref().and_then(|a| a.first());
    arch.map(|a| is_qwen3_hybrid_arch_name(a)).unwrap_or(false)
}

fn qwen3_hybrid_raw_from_extra_config(config: &Config) -> Option<Qwen3HybridRawConfig> {
    if !is_qwen3_hybrid_arch(config) {
        return None;
    }
    let extra = config.extra_config_json.as_ref()?;
    let root = serde_json::from_str::<serde_json::Value>(extra).ok()?;
    let cfg = root.get("text_config").cloned().unwrap_or(root);
    serde_json::from_value::<Qwen3HybridRawConfig>(cfg).ok()
}

pub fn resolve_qwen3_hybrid_config(config: &Config) -> Qwen3HybridConfig {
    let raw_cfg = qwen3_hybrid_raw_from_extra_config(config).unwrap_or_default();

    let mut layer_types = if let Some(layer_types) = raw_cfg.layers_block_type {
        layer_types
    } else if let Some(interval) = raw_cfg.full_attention_interval {
        if interval > 0 {
            (0..config.num_hidden_layers)
                .map(|idx| {
                    if (idx + 1) % interval == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect::<Vec<_>>()
        } else {
            vec!["full_attention".to_string(); config.num_hidden_layers]
        }
    } else {
        vec!["full_attention".to_string(); config.num_hidden_layers]
    };

    for layer_type in layer_types.iter_mut() {
        if layer_type == "attention" {
            *layer_type = "full_attention".to_string();
        }
    }
    if layer_types.len() != config.num_hidden_layers {
        layer_types = vec!["full_attention".to_string(); config.num_hidden_layers];
    }

    let num_v_heads = raw_cfg
        .linear_num_value_heads
        .or(raw_cfg.linear_num_heads)
        .unwrap_or(config.num_attention_heads);
    let num_k_heads = raw_cfg
        .linear_num_key_heads
        .or(raw_cfg.linear_num_key_value_heads)
        .unwrap_or(num_v_heads);
    let key_head_dim = raw_cfg.linear_key_head_dim.unwrap_or(
        config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads),
    );
    let value_head_dim = raw_cfg.linear_value_head_dim.unwrap_or(key_head_dim);

    Qwen3HybridConfig {
        layer_types,
        conv_kernel_size: raw_cfg.conv_kernel_size.unwrap_or(4),
        num_v_heads,
        num_k_heads,
        key_head_dim,
        value_head_dim,
    }
}

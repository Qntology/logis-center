use candle_nn::Activation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Default)]
pub struct RopeParameters {
    #[serde(default)]
    pub mrope_interleaved: bool,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub rope_type: String,
    #[serde(default)]
    pub rope_theta: f32,
    #[serde(default)]
    pub partial_rotary_factor: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Qwen3_5TextConfig {
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default)]
    pub attn_output_gate: bool,
    pub dtype: String,
    pub eos_token_id: u32,
    #[serde(default)]
    pub full_attention_interval: usize,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    #[serde(default)]
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub linear_conv_kernel_dim: usize,
    #[serde(default)]
    pub linear_key_head_dim: usize,
    #[serde(default)]
    pub linear_num_key_heads: usize,
    #[serde(default)]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub linear_value_head_dim: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    pub use_cache: bool,
    pub vocab_size: usize,
    #[serde(default)]
    pub mamba_ssm_dtype: String,
    pub rope_parameters: RopeParameters,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Default)]
pub struct Qwen3VLVisionConfig {
    #[serde(default)]
    pub depth: usize,
    #[serde(default)]
    pub embed_dim: usize,
    #[serde(default)]
    pub hidden_size: usize,
    #[serde(default)]
    pub num_heads: usize,
    #[serde(default)]
    pub patch_size: usize,
    #[serde(default)]
    pub spatial_merge_size: usize,
    #[serde(default)]
    pub temporal_patch_size: usize,
    #[serde(default)]
    pub window_size: usize,
    #[serde(default)]
    pub out_hidden_size: usize,
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Qwen3_5Config {
    pub text_config: Qwen3_5TextConfig,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub image_token_id: u32,
    #[serde(default)]
    pub video_token_id: u32,
    #[serde(default)]
    pub vision_start_token_id: u32,
    #[serde(default)]
    pub vision_end_token_id: u32,
    #[serde(default)]
    pub vision_config: Option<Qwen3VLVisionConfig>,
}

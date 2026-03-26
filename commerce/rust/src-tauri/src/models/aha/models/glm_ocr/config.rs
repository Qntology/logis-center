use candle_nn::Activation;
use serde::Deserialize;

use crate::models::qwen3vl::config::Size;

/// Vision encoder configuration for GLM-OCR.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct GlmOcrVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub hidden_act: Activation,
    pub attention_bias: bool,
    pub num_heads: usize,
    /// Number of input image channels. Default: 3
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub rms_norm_eps: f64,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    pub intermediate_size: usize,
    pub initializer_range: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

fn default_in_channels() -> usize {
    3
}

fn default_rope_theta() -> f32 {
    10000.0
}

#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct GlmOcrRopeParameters {
    pub rope_type: String,
    pub mrope_section: Vec<usize>,
    pub partial_rotary_factor: f32,
    pub rope_theta: f32,
}

/// Text decoder configuration for GLM-OCR.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct GlmOcrTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: Activation,
    pub use_cache: bool,
    pub rope_parameters: GlmOcrRopeParameters,
    pub eos_token_id: Vec<u32>,
    pub dtype: String,
}

/// Top-level configuration for GLM-OCR multimodal model.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmOcrConfig {
    pub model_type: String,
    pub vision_config: GlmOcrVisionConfig,
    pub text_config: GlmOcrTextConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub image_start_token_id: u32,
    pub image_end_token_id: u32,
    pub video_start_token_id: u32,
    pub video_end_token_id: u32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmOcrGenerationConfig {
    pub pad_token_id: u32,
    pub do_sample: bool,
    pub eos_token_id: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmOcrPreprocessorConfig {
    pub size: Size,
    pub do_rescale: bool,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
}

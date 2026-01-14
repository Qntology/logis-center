use candle_nn::Activation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HunYuanVLConfig {
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub attention_head_dim: usize,
    pub bos_token_id: u32,
    pub eod_token_id: u32,
    pub eos_token_id: u32,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub image_start_token_id: u32,
    pub image_end_token_id: u32,
    pub image_token_id: u32,
    pub image_newline_token_id: u32,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub mlp_bias: bool,
    pub norm_type: String,
    pub num_attention_heads: usize,
    pub num_experts: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub org_vocab_size: usize,
    pub pad_id: i32,
    pub pad_token_id: i32,
    pub pretraining_tp: i32,
    pub rms_norm_eps: f64,
    pub rope_scaling: HunYuanVLRopeScaling,
    pub rope_theta: f64,
    pub routed_scaling_factor: f64,
    pub sep_token_id: u32,
    pub text_end_id: u32,
    pub text_start_id: u32,
    pub tie_word_embeddings: bool,
    pub dtype: String,
    pub use_cache: bool,
    pub use_qk_norm: bool,
    pub use_cla: bool,
    pub vision_config: HunYuanVLVisionConfig,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HunYuanVLRopeScaling {
    pub alpha: f64,
    pub beta_fast: i32,
    pub beta_slow: i32,
    pub factor: f64,
    pub mscale: f64,
    pub mscale_all_dim: f64,
    #[serde(rename = "type")]
    pub type_field: String,
    pub xdrope_section: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HunYuanVLVisionConfig {
    pub add_patchemb_bias: bool,
    pub attention_dropout: f64,
    pub cat_extra_token: i32,
    pub hidden_act: Activation,
    pub hidden_dropout: f64,
    pub hidden_size: usize,
    pub img_max_token_num: usize,
    pub intermediate_size: usize,
    pub interpolate_mode: String,
    pub max_image_size: usize,
    pub max_vit_seq_len: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub num_hidden_layers: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub rms_norm_eps: f64,
    pub spatial_merge_size: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct HunyuanOCRGenerationConfig {
    pub bos_token_id: usize,
    pub pad_token_id: usize,
    pub do_sample: bool,
    pub eos_token_id: Vec<usize>,
    pub top_p: f32,
    pub top_k: usize,
    pub temperature: f32,
    pub repetition_penalty: f32,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct HunyuanOCRPreprocessorConfig {
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub patch_size: usize,
    pub resample: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
}

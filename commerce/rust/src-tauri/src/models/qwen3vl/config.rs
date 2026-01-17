use serde::Deserialize;
use candle_nn::Activation;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3VLGenerationConfig {
    pub bos_token_id: u32,
    pub eos_token_id: Vec<i64>, 
    pub pad_token_id: u32,
    pub repetition_penalty: f32,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub transformers_version: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3VLVisionConfig {
    pub depth: usize,
    pub embed_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub in_channels: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_position_embeddings: usize, 
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub deepstack_visual_indexes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RopeScaling {
    pub mrope_section: Vec<usize>, 
    pub rope_type: String,
    pub r#type: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3VLTextConfig {
    pub architectural: String, 
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub max_window_layers: usize,
    pub model_type: String, 
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: RopeScaling,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub use_sliding_window: bool,
    pub vocab_size: usize,
    pub dtype: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3VLConfig {
    pub architectural: String, 
    pub auto_map: std::collections::HashMap<String, String>,
    pub hidden_size: usize,
    pub image_token_id: usize,
    pub model_type: String, 
    pub text_config: Qwen3VLTextConfig,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub video_token_id: usize,
    pub vision_config: Qwen3VLVisionConfig,
    pub vision_start_token_id: usize,
    pub vision_token_id: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Size {
    pub longest_edge: usize,
    pub shortest_edge: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PreprocessorConfig {
    pub do_convert_rgb: bool,
    pub do_normalize: bool,
    pub do_pad: bool,
    pub do_resize: bool,
    pub do_rescale: bool,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub max_pixels: usize,
    pub min_pixels: usize,
    pub rescale_factor: f64,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub size: Size,
}

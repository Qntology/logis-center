use serde::Deserialize;
use candle_nn::Activation;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct QwenVLGenerationConfig {
    pub bos_token_id: u32,
    pub eos_token_id: serde_json::Value, 
    pub pad_token_id: u32,
    pub repetition_penalty: f32,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub transformers_version: String,
}

impl Default for QwenVLGenerationConfig {
    fn default() -> Self {
        Self {
            bos_token_id: 151643,
            eos_token_id: serde_json::json!([151643, 151645]),
            pad_token_id: 151643,
            repetition_penalty: 1.0,
            temperature: 0.1,
            top_k: 9,
            top_p: 0.1,
            transformers_version: "4.45.0".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct QwenVLVisionConfig {
    pub depth: usize,
    pub embed_dim: Option<usize>,
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
    pub mrope_interleaved: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct QwenVLTextConfig {
    pub architectural: Option<String>, 
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: u32,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub max_window_layers: Option<usize>,
    pub model_type: String, 
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: Option<RopeScaling>,
    pub rope_theta: f32,
    pub sliding_window: Option<usize>,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub use_sliding_window: Option<bool>,
    pub vocab_size: usize,
    pub dtype: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct QwenVLConfig {
    pub architectures: Option<Vec<String>>, 
    pub auto_map: Option<std::collections::HashMap<String, String>>,
    pub hidden_size: Option<usize>,
    pub image_token_id: Option<usize>,
    pub model_type: String, 
    pub text_config: Option<QwenVLTextConfig>,
    pub tie_word_embeddings: bool,
    pub torch_dtype: Option<String>,
    pub transformers_version: String,
    pub video_token_id: Option<usize>,
    pub vision_config: Option<QwenVLVisionConfig>,
    pub vision_start_token_id: Option<usize>,
    pub vision_end_token_id: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Size {
    pub longest_edge: usize,
    pub shortest_edge: usize,
}

impl Default for Size {
    fn default() -> Self {
        Self {
            longest_edge: 1344,
            shortest_edge: 224,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PreprocessorConfig {
    pub do_convert_rgb: Option<bool>,
    pub do_normalize: Option<bool>,
    pub do_pad: Option<bool>,
    pub do_resize: Option<bool>,
    pub do_rescale: Option<bool>,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub max_pixels: Option<usize>,
    pub min_pixels: Option<usize>,
    pub rescale_factor: Option<f64>,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub size: Size,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            do_convert_rgb: Some(true),
            do_normalize: Some(true),
            do_pad: Some(true),
            do_resize: Some(true),
            do_rescale: Some(true),
            image_mean: vec![0.48145466, 0.4578275, 0.40821073],
            image_std: vec![0.26862954, 0.26130258, 0.2757771],
            max_pixels: Some(12845056),
            min_pixels: Some(3136),
            rescale_factor: Some(0.00392156862745098),
            patch_size: 14,
            merge_size: 2,
            temporal_patch_size: 2,
            size: Size::default(),
        }
    }
}
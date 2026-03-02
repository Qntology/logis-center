use serde::Deserialize;
use candle_nn::Activation;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct Qwen3VLGenerationConfig {
    pub bos_token_id: u32,
    pub eos_token_id: serde_json::Value, 
    pub pad_token_id: u32,
    pub repetition_penalty: f32,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub transformers_version: String,
}

impl Default for Qwen3VLGenerationConfig {
    fn default() -> Self {
        Self {
            bos_token_id: 151643,
            eos_token_id: serde_json::json!([248044, 151643, 151645]),
            pad_token_id: 151643,
            repetition_penalty: 1.0,
            temperature: 0.9,
            top_k: 9,
            top_p: 0.9,
            transformers_version: "4.57.0".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3VLVisionConfig {
    pub depth: usize,
    pub hidden_act: String, // gelu_pytorch_tanh 등 대응을 위해 String으로 유지 후 수동 변환
    pub hidden_size: usize,
    pub in_channels: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub model_type: String,
    pub num_heads: usize,
    pub num_position_embeddings: usize, 
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub deepstack_visual_indexes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RopeParameters {
    pub mrope_interleaved: bool,
    pub mrope_section: Vec<usize>, 
    pub rope_type: String,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3VLTextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub attn_output_gate: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: u32,
    pub full_attention_interval: usize,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>, // linear_attention, full_attention 구분
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub max_position_embeddings: usize,
    pub model_type: String, 
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub vocab_size: usize,
    pub mamba_ssm_dtype: String,
    pub dtype: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3VLConfig {
    pub architectures: Option<Vec<String>>, 
    pub auto_map: Option<std::collections::HashMap<String, String>>,
    pub hidden_size: Option<usize>,
    pub image_token_id: Option<usize>,
    pub model_type: String, 
    pub text_config: Option<Qwen3VLTextConfig>,
    pub tie_word_embeddings: bool,
    pub transformers_version: String,
    pub video_token_id: Option<usize>,
    pub vision_config: Option<Qwen3VLVisionConfig>,
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
pub struct VideoPreprocessorConfig {
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub merge_size: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub size: Size,
    pub processor_class: String,
    pub video_processor_type: String,
}

impl Default for VideoPreprocessorConfig {
    fn default() -> Self {
        Self {
            image_mean: vec![0.5, 0.5, 0.5],
            image_std: vec![0.5, 0.5, 0.5],
            merge_size: 2,
            patch_size: 16,
            temporal_patch_size: 2,
            size: Size { longest_edge: 25165824, shortest_edge: 4096 },
            processor_class: "Qwen3VLProcessor".to_string(),
            video_processor_type: "Qwen3VLVideoProcessor".to_string(),
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

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub size: PreprocessorSize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorSize {
    pub longest_edge: usize,
    pub shortest_edge: usize,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            patch_size: 14,
            temporal_patch_size: 2,
            merge_size: 2,
            image_mean: vec![0.48145466, 0.4578275, 0.40821073],
            image_std: vec![0.26862954, 0.26130258, 0.27577711],
            size: PreprocessorSize { longest_edge: 1344, shortest_edge: 28 },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3VLConfig {
    pub architectures: Option<Vec<String>>,
    pub auto_map: Option<serde_json::Value>,
    pub hidden_size: Option<usize>,
    pub image_token_id: Option<usize>,
    pub model_type: String,
    pub text_config: Option<Qwen3VLTextConfig>,
    pub tie_word_embeddings: bool,
    pub torch_dtype: Option<String>,
    pub transformers_version: String,
    pub video_token_id: Option<usize>,
    pub vision_config: Option<Qwen3VLVisionConfig>,
    pub vision_start_token_id: Option<usize>,
    pub vision_end_token_id: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3VLTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub dtype: Option<String>,
    pub rope_scaling: Option<RopeScaling>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub mrope_section: Vec<usize>,
    pub rope_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3VLVisionConfig {
    pub depth: usize,
    pub embed_dim: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub mlp_ratio: f32,
    pub num_heads: usize,
    pub out_features: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Qwen3VLGenerationConfig {
    pub eos_token_id: serde_json::Value,
    pub bos_token_id: Option<usize>,
    pub pad_token_id: Option<usize>,
}

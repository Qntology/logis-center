#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct DeepseekV2Config {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub first_k_dense_replace: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub kv_lora_rank: Option<usize>,
    pub lm_head: bool,
    pub max_position_embeddings: usize,
    pub moe_intermediate_size: usize,
    #[serde(default = "default_moe_layer_freq")]
    pub moe_layer_freq: usize,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f64,
    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,
    #[serde(default = "default_aux_loss_alpha")]
    pub aux_loss_alpha: f32,
    #[serde(default = "default_true")]
    pub seq_aux: bool,
    #[serde(default = "default_false")]
    pub norm_topk_prob: bool,
    pub n_group: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_attention_heads: usize,
    pub num_experts_per_tok: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub q_lora_rank: Option<usize>,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub rm_head: bool,
    pub topk_group: usize,
    pub topk_method: String,
    pub torch_dtype: String,
    pub use_mla: bool,
    pub v_head_dim: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
}

fn default_moe_layer_freq() -> usize {
    1
}
fn default_routed_scaling_factor() -> f64 {
    1.0
}
fn default_scoring_func() -> String {
    "softmax".to_string()
}
fn default_aux_loss_alpha() -> f32 {
    0.001
}
fn default_true() -> bool {
    true
}
fn default_false() -> bool {
    false
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct ProjectorConfig {
    pub input_dim: usize,
    pub model_type: String,
    pub n_embed: usize,
    pub projector_type: String,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct ClipL14_224 {
    pub heads: usize,
    pub image_size: usize,
    pub layers: usize,
    pub patch_size: usize,
    pub width: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct SamVitB {
    pub downsample_channels: Vec<usize>,
    pub global_attn_indexes: Vec<usize>,
    pub heads: usize,
    pub layers: usize,
    pub width: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Width {
    #[serde(rename = "clip-l-14-224")]
    pub clip_l_14_224: ClipL14_224,
    pub sam_vit_b: SamVitB,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct DeepseekOCRVisionConfig {
    pub image_size: usize,
    pub mlp_ratio: f32,
    pub width: Width,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct DeepseekOCRConfig {
    pub language_config: DeepseekV2Config,
    pub projector_config: ProjectorConfig,
    pub torch_dtype: String,
    pub vision_config: DeepseekOCRVisionConfig,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub first_k_dense_replace: u32,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub kv_lora_rank: Option<usize>,
    pub lm_head: bool,
    pub max_position_embeddings: usize,
    pub moe_intermediate_size: usize,
    pub n_group: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_attention_heads: usize,
    pub num_experts_per_tok: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub q_lora_rank: Option<usize>,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub rm_head: bool,
    pub topk_group: usize,
    pub topk_method: String,
    pub use_mla: bool,
    pub v_head_dim: usize,
    pub vocab_size: usize,
}

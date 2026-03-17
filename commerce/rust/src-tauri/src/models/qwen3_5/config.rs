use candle_nn::Activation;
use serde::Deserialize;

// [FIX] qwen3vl의 VisionConfig를 그대로 재사용하여 코드 중복을 막습니다.
use crate::models::qwen3vl::config::Qwen3VLVisionConfig;

// ============================================================================
// [GENERATION] Qwen 3.5 전용 생성(디코딩) 파라미터 구조체 추가
// ============================================================================
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct Qwen3_5GenerationConfig {
    pub bos_token_id: u32,
    pub eos_token_id: serde_json::Value, // 배열([151643, 151645])이나 단일 숫자 모두 대응
    pub pad_token_id: u32,
    pub repetition_penalty: f32,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub transformers_version: String,
}

impl Default for Qwen3_5GenerationConfig {
    fn default() -> Self {
        Self {
            bos_token_id: 151643,
            eos_token_id: serde_json::json!([151643, 151645]),
            pad_token_id: 151643,
            repetition_penalty: 1.05,
            temperature: 0.4,
            top_k: 20,
            top_p: 0.95,
            transformers_version: "4.45.0".to_string(),
        }
    }
}

// ============================================================================
// [ROPE] 회전 위치 임베딩 파라미터
// ============================================================================
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RopeParameters {
    pub mrope_interleaved: bool,
    pub mrope_section: Vec<usize>,
    pub rope_type: String,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

// ============================================================================
// [TEXT CONFIG] Qwen 3.5 텍스트 백본 설정
// ============================================================================
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3_5TextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub dtype: String,
    pub eos_token_id: u32,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub use_cache: bool,
    pub vocab_size: usize,
    pub rope_parameters: RopeParameters,
    
    // [CRITICAL FIX] Qwen3.5 특수 레이어 및 Mamba/DeltaNet 필드들을 모두 Option으로 감싸
    // 일반 GGUF 모델이나 순수 텍스트 모델 로드 시 발생하는 Serde Panic을 원천 차단!
    pub layer_types: Vec<String>,
    pub attn_output_gate: Option<bool>,
    pub full_attention_interval: Option<usize>,
    pub linear_conv_kernel_dim: Option<usize>,
    pub linear_key_head_dim: Option<usize>,
    pub linear_num_key_heads: Option<usize>,
    pub linear_num_value_heads: Option<usize>,
    pub linear_value_head_dim: Option<usize>,
    pub mlp_only_layers: Option<Vec<usize>>,
    pub mtp_num_hidden_layers: Option<usize>,
    pub mtp_use_dedicated_embeddings: Option<bool>,
    pub tie_word_embeddings: Option<bool>,
    pub mamba_ssm_dtype: Option<String>,
}

// ============================================================================
// [ROOT CONFIG] Qwen 3.5 마스터 설정
// ============================================================================
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3_5Config {
    pub text_config: Qwen3_5TextConfig,
    pub tie_word_embeddings: bool,
    
    // [CRITICAL FIX] 비전(Vision) 관련 토큰과 설정들도 Option으로 감싸서 
    // Qwen3.5 순수 텍스트(LLM) 모델을 로드할 때도 에러 없이 호환되도록 수정!
    pub image_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
    pub vision_end_token_id: Option<u32>,
    pub vision_start_token_id: Option<u32>,
    pub vision_config: Option<Qwen3VLVisionConfig>,
}
use candle_nn::Activation;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3Config {
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f64,
    pub bos_token_id: Option<u32>, // 🌟 파일에 없어도 에러 안 나게 Option 처리
    pub eos_token_id: Option<u32>, // 🌟 Option 처리
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    #[serde(default)]
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub max_window_layers: Option<usize>, // 🌟 생략 잦은 필드 Option 처리
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_torch_dtype")]
    pub torch_dtype: String,
    #[serde(default)]
    pub use_cache: bool,
    #[serde(default)]
    pub use_sliding_window: bool,
    pub vocab_size: usize,
}

// 🌟 JSON에 해당 값이 없을 때 들어갈 기본값 함수들
fn default_rope_theta() -> f32 { 10000.0 }
fn default_torch_dtype() -> String { "float16".to_string() }

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Qwen3GenerationConfig {
    pub bos_token_id: usize,
    pub pad_token_id: usize,
    pub do_sample: bool,
    pub eos_token_id: Vec<usize>,
    pub top_p: f32,
    pub top_k: usize,
    pub temperature: f32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
}

fn default_repetition_penalty() -> f32 {
    1.2
}


impl Default for Qwen3GenerationConfig {
    fn default() -> Self {
        Self {
            bos_token_id: 151643,
            pad_token_id: 151643,
            do_sample: false,
            eos_token_id: vec![151643, 151645], 
            top_p: 0.95,
            top_k: 80,
            temperature: 1.0,
            repetition_penalty: 1.2,
        }
    }
}
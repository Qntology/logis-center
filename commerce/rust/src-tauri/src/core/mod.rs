#[macro_export]
macro_rules! def_broadcast_message_to_runners {
    (
        $vis:vis,
        $fn_name:ident,
        $thread_fn_name:ident,
        ($($arg_name:ident: $arg_type:ty),*),
        $msg_variant:path,
        ($($msg_arg:expr),*),
        $resp_variant:path,
        $ret_type:ty
    ) => {
        $vis fn $fn_name(&self, $($arg_name: $arg_type),*) -> candle_core::Result<$ret_type> {
            match &*self.runners.read() {
                RunnerType::Thread(r) => r.$thread_fn_name($($arg_name),*),
                RunnerType::Process(streams) => {
                    let mut streams_guard = self.runners.write();
                    if let RunnerType::Process(ref mut streams_vec) = *streams_guard {
                        // Pass the whole Vec to send_local
                        crate::runner::send_local(streams_vec, &$msg_variant($($msg_arg),*), false)?;
                        
                        let mut values = Vec::new();
                        for stream in streams_vec.iter_mut() {
                            match crate::runner::receive_local(stream, false)? {
                                $resp_variant(v) => values.push(v),
                                _ => candle_core::bail!("Unexpected response from runner"),
                            }
                        }
                        if values.is_empty() {
                            candle_core::bail!("No responses from runners");
                        }
                        let first_val = values.pop().unwrap();
                        Ok(first_val)
                    } else {
                        candle_core::bail!("Invalid runner state");
                    }
                }
            }
        }
    };
}

pub mod block_manager;
pub mod engine;
pub mod prefix_cache;
pub mod runner;
pub mod scheduler;
pub mod sequence;

#[cfg(feature = "python")]
use pyo3::pyclass;

#[cfg(feature = "python")]
#[pyclass]
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct GenerationOutput {
    #[pyo3(get)]
    pub seq_id: usize,
    #[pyo3(get)]
    pub prompt_length: usize,
    #[pyo3(get)]
    pub prompt_start_time: usize,
    #[pyo3(get)]
    pub decode_start_time: usize,
    #[pyo3(get)]
    pub decode_finish_time: usize,
    #[pyo3(get)]
    pub decoded_length: usize,
    #[pyo3(get)]
    pub decode_output: String,
    #[pyo3(get)]
    pub stop_sequence: Option<String>,
}

#[cfg(not(feature = "python"))]
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct GenerationOutput {
    pub seq_id: usize,
    pub prompt_length: usize,
    pub prompt_start_time: usize,
    pub decode_start_time: usize,
    pub decode_finish_time: usize,
    pub decoded_length: usize,
    pub decode_output: String,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum EmbeddingStrategy {
    LastToken,
    AllTokens,
    Mean,
    Last,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct UsageResponse {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub token_used: u32,
    pub max_model_len: u32,
    pub used_kvcache_tokens: u32,
    pub total_kv_cache_tokens: u32,
    pub swap_used: f32,
    pub total_swap_memory: f32,
    pub session_status: String,
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        println!("[INFO] {}", format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        eprintln!("[WARN] {}", format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        eprintln!("[ERROR] {}", format!($($arg)*))
    };
}

pub trait DecodeStreamTrait: Send + Sync {
    fn step(&mut self, id: u32) -> candle_core::Result<Option<String>>;
}

pub type DecodeStreamType = Box<dyn DecodeStreamTrait>;

pub struct StreamWithTokenizer<M, N, PT, PP, D>
where
    M: tokenizers::Model + Send + Sync + 'static,
    N: tokenizers::Normalizer + Send + Sync + 'static,
    PT: tokenizers::PreTokenizer + Send + Sync + 'static,
    PP: tokenizers::PostProcessor + Send + Sync + 'static,
    D: tokenizers::Decoder + Send + Sync + 'static,
{
    pub _tokenizer: Box<tokenizers::TokenizerImpl<M, N, PT, PP, D>>,
    pub stream: tokenizers::DecodeStream<'static, M, N, PT, PP, D>,
}

impl<M, N, PT, PP, D> DecodeStreamTrait for StreamWithTokenizer<M, N, PT, PP, D>
where
    M: tokenizers::Model + Send + Sync + 'static,
    N: tokenizers::Normalizer + Send + Sync + 'static,
    PT: tokenizers::PreTokenizer + Send + Sync + 'static,
    PP: tokenizers::PostProcessor + Send + Sync + 'static,
    D: tokenizers::Decoder + Send + Sync + 'static,
{
    fn step(&mut self, id: u32) -> candle_core::Result<Option<String>> {
        self.stream.step(id).map_err(candle_core::Error::msg)
    }
}

pub struct ChatCompletionLogger;
impl ChatCompletionLogger {
    pub fn log_stream_token(&self, _token: &str) {}
}

#[derive(Clone, Debug)]
pub struct ToolConfig {
    pub start_token_ids: std::collections::HashSet<u32>,
    pub end_token_ids: std::collections::HashSet<u32>,
    pub start_token_str: String,
    pub end_token_str: String,
}

impl ToolConfig {
    pub fn for_model_type(model_type: &crate::utils::config::ModelType) -> Self {
        use std::collections::HashSet;
        use crate::utils::config::ModelType;
        let mut start_ids = HashSet::new();
        let mut end_ids = HashSet::new();

        match model_type {
            ModelType::LLaMa => {
                start_ids.insert(128010); // <|python_tag|>
                end_ids.insert(128008); // <|eom_id|>
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<|python_tag|>".to_string(),
                    end_token_str: "<|eom_id|>".to_string(),
                }
            }
            ModelType::Qwen3
            | ModelType::Qwen3MoE
            | ModelType::Qwen3_5
            | ModelType::Qwen3_5MoE
            | ModelType::Qwen3VL => {
                start_ids.insert(151657); // <tool_call>
                end_ids.insert(151658); // </tool_call>
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<tool_call>".to_string(),
                    end_token_str: "</tool_call>".to_string(),
                }
            }
            _ => {
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "".to_string(),
                    end_token_str: "".to_string(),
                }
            }
        }
    }
}

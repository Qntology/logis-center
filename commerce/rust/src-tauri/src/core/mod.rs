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
        $return_ty:ty
    ) => {
        $vis fn $fn_name(&self, $($arg_name: $arg_type),*) -> candle_core::Result<$return_ty>
        where
            $return_ty: std::fmt::Debug + Send,
        {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            use interprocess::local_socket::traits::Stream;
            use crate::runner::{send_local, receive_local, RunnerType, LocalStream};

            match &mut *self.runners.write() {
                RunnerType::Thread(model_runner) => {
                    // Thread Mode: Call the method directly.
                    model_runner.$thread_fn_name($($arg_name),*)
                }
                RunnerType::Process(ref mut runner_streams) => {
                    // Process Mode: Broadcast to all subprocess runners.
                    let cloned_streams: Vec<LocalStream> = runner_streams
                        .iter_mut()
                        .map(|s: &mut LocalStream| s.try_clone().expect("Failed to clone runner stream"))
                        .collect();

                    // Use Rayon for parallel broadcast
                    let all_results: candle_core::Result<Vec<$return_ty>> = cloned_streams
                        .into_par_iter()
                        .map(|mut stream: LocalStream| {
                            // Send the message
                            send_local(
                                &mut vec![stream.try_clone().map_err(candle_core::Error::msg)?],
                                &$msg_variant($($msg_arg),*),
                                false,
                            ).map_err(candle_core::Error::msg)?;

                            // Wait for the response
                            let response = receive_local(&mut stream, false).map_err(candle_core::Error::msg)?;
                            match response {
                                // Match on the expected response containing the value
                                $resp_variant(value) => {
                                    Ok(value)
                                }
                                other => {
                                    candle_core::bail!("Unexpected response for {}: {:?}", stringify!($fn_name), other)
                                }
                            }
                        })
                        .collect();

                    // Check that all ranks returned the same value
                    match all_results {
                        Ok(mut values) => {
                            let values: Vec<$return_ty> = values; // Explicit type annotation
                            if values.is_empty() {
                                candle_core::bail!("No values received from runners for {}", stringify!($fn_name));
                            }
                            // Pop first element to return
                            let first_val = values.into_iter().next().unwrap();
                            Ok(first_val)
                        }
                        Err(e) => Err(e),
                    }
                }
            }
        }
    };
}

pub mod block_manager;
pub mod chat;
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

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq)]
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

pub type DecodeStreamType = Box<dyn DecodeStreamTrait + Send + Sync>;

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
    pub fn log_prompt(&self, _prompt: &str) {}
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

    pub fn has_start_tokens(&self) -> bool {
        !self.start_token_ids.is_empty()
    }

    pub fn has_end_tokens(&self) -> bool {
        !self.end_token_ids.is_empty()
    }

    pub fn validate_with_tokenizer(&mut self, tokenizer: &tokenizers::Tokenizer, model_type: &crate::utils::config::ModelType) {
        use std::collections::HashSet;
        if self.has_start_tokens()
            && !Self::matches_single_token(tokenizer, &self.start_token_str, &self.start_token_ids)
        {
            if Self::try_rebind_single_token_id(
                tokenizer,
                &self.start_token_str,
                &mut self.start_token_ids,
            ) {
                crate::log_warn!(
                    "Tool start token IDs corrected from tokenizer for model {:?}: {:?}",
                    model_type,
                    self.start_token_ids
                );
            } else {
                crate::log_warn!(
                    "Tool start token IDs not supported by tokenizer for model {:?}, falling back to text matching",
                    model_type
                );
                self.start_token_ids.clear();
            }
        }

        if self.has_end_tokens()
            && !Self::matches_single_token(tokenizer, &self.end_token_str, &self.end_token_ids)
        {
            if Self::try_rebind_single_token_id(
                tokenizer,
                &self.end_token_str,
                &mut self.end_token_ids,
            ) {
                crate::log_warn!(
                    "Tool end token IDs corrected from tokenizer for model {:?}: {:?}",
                    model_type,
                    self.end_token_ids
                );
            } else {
                crate::log_warn!(
                    "Tool end token IDs not supported by tokenizer for model {:?}, falling back to text matching",
                    model_type
                );
                self.end_token_ids.clear();
            }
        }
    }

    pub fn tool_call_end_ids(&self, tokenizer: &tokenizers::Tokenizer) -> Vec<u32> {
        let mut tool_call_end_ids: Vec<u32> = Vec::new();
        let mut used_special = false;
        if self.has_end_tokens() {
            let mut use_special = true;
            if !self.end_token_str.is_empty() {
                if let Ok(encoded) = tokenizer.encode(self.end_token_str.as_str(), false) {
                    let ids = encoded.get_ids();
                    if ids.len() != 1 || !self.end_token_ids.contains(&ids[0]) {
                        use_special = false;
                    }
                } else {
                    use_special = false;
                }
            }
            if use_special {
                tool_call_end_ids.extend(self.end_token_ids.iter().copied());
                used_special = true;
            }
        }
        if !used_special && !self.end_token_str.is_empty() && self.end_token_str.starts_with('<') {
            if let Ok(encoded) = tokenizer.encode(self.end_token_str.as_str(), false) {
                let ids = encoded.get_ids();
                if ids.len() == 1 {
                    tool_call_end_ids.push(ids[0]);
                }
            }
        }
        tool_call_end_ids
    }

    pub fn tool_call_start_ids(&self, tokenizer: &tokenizers::Tokenizer) -> Vec<u32> {
        let mut tool_call_start_ids: Vec<u32> = Vec::new();
        let mut used_special = false;
        if self.has_start_tokens() {
            let mut use_special = true;
            if !self.start_token_str.is_empty() {
                if let Ok(encoded) = tokenizer.encode(self.start_token_str.as_str(), false) {
                    let ids = encoded.get_ids();
                    if ids.len() != 1 || !self.start_token_ids.contains(&ids[0]) {
                        use_special = false;
                    }
                } else {
                    use_special = false;
                }
            }
            if use_special {
                tool_call_start_ids.extend(self.start_token_ids.iter().copied());
                used_special = true;
            }
        }
        if !used_special
            && !self.start_token_str.is_empty()
            && self.start_token_str.starts_with('<')
        {
            if let Ok(encoded) = tokenizer.encode(self.start_token_str.as_str(), false) {
                let ids = encoded.get_ids();
                if ids.len() == 1 {
                    tool_call_start_ids.push(ids[0]);
                }
            }
        }
        tool_call_start_ids
    }

    fn matches_single_token(tokenizer: &tokenizers::Tokenizer, text: &str, token_ids: &std::collections::HashSet<u32>) -> bool {
        if text.is_empty() {
            return false;
        }
        match tokenizer.encode(text, false) {
            Ok(encoded) => {
                let ids = encoded.get_ids();
                ids.len() == 1 && token_ids.contains(&ids[0])
            }
            Err(_) => false,
        }
    }

    fn try_rebind_single_token_id(
        tokenizer: &tokenizers::Tokenizer,
        text: &str,
        token_ids: &mut std::collections::HashSet<u32>,
    ) -> bool {
        if text.is_empty() {
            return false;
        }
        if let Ok(encoded) = tokenizer.encode(text, false) {
            let ids = encoded.get_ids();
            if ids.len() == 1 {
                token_ids.clear();
                token_ids.insert(ids[0]);
                return true;
            }
        }
        if let Some(id) = tokenizer.get_vocab(true).get(text).copied() {
            token_ids.clear();
            token_ids.insert(id);
            return true;
        }
        false
    }
}

const REASONING_MARKERS: &[(&str, &str)] = &[
    ("<think>", "</think>"),
    ("<|think|>", "<|/think|>"),
    ("[THINK]", "[/THINK]"),
    ("<thought>", "</thought>"),
];

pub fn detect_prefilled_reasoning_end_marker(prompt: &str) -> Option<String> {
    let trimmed = prompt.trim_end();
    for &(start, end) in REASONING_MARKERS {
        if trimmed.ends_with(start) {
            return Some(end.to_string());
        }
    }
    None
}

#[macro_export]
macro_rules! build_model {
    ($model_type:expr, $vb:expr, $comm:expr, $config:expr, $dtype:expr, $is_rope_i:expr, $device:expr, $reporter:expr,
        { $( $variant:ident => $ctor:ident ),+ $(,)? }
    ) => {{
        use crate::utils::config::ModelType;
        use crate::core::runner::Model;
        use std::sync::Arc;
        match $model_type {
            $( ModelType::$variant => Ok::<Model, candle_core::Error>(Model::$variant(Arc::new($ctor::new(
                $vb,
                $comm.clone(),
                $config,
                $dtype,
                $is_rope_i,
                $device,
                Arc::clone(&$reporter),
            )?))), )+
            _ => {
                candle_core::bail!("Unsupported model type: {:?}", $model_type);
            }
        }
    }};
}

#[macro_export]
macro_rules! model_call {
    ($model:expr, $method:ident,
        ($input_ids:expr, $positions:expr, $kv:expr, $input_metadata:expr),
        { $( $variant:ident => $extra:expr ),+ $(,)? }
        $(, $fallback:expr )?
    ) => {{
        use crate::core::runner::Model;
        match $model {
            $( Model::$variant(model) => model.$method($input_ids, $positions, $kv, $input_metadata, $extra), )+
            $( _ => $fallback, )?
        }
    }};
}

#[cfg(all(feature = "cuda", feature = "graph"))]
#[macro_export]
macro_rules! graph_extra_arg {
    (EmbedInputs, $embeded_inputs:ident) => {
        $embeded_inputs
    };
    (NoneArg, $embeded_inputs:ident) => {
        None
    };
}

#[cfg(all(feature = "cuda", feature = "graph"))]
#[macro_export]
macro_rules! graph_wrapper {
    ($model:expr, $device:expr,
        { $( $variant:ident => $arg:tt ),+ $(,)? }
    ) => {{
        use crate::core::runner::Model;
        use std::sync::Arc;
        use crate::runner::CudaGraphWrapper;
        use crate::utils::kvcache_allocator::InputMetadata;
        use candle_core::Tensor;
        use crate::runner::ModelFn;

        match $model {
            $( Model::$variant(m) => {
                let model_arc = Arc::clone(m);
                let closure = move |input_ids: &Tensor,
                                    positions: &Tensor,
                                    kv_caches: Option<&Vec<(Tensor, Tensor)>>,
                                    input_metadata: &InputMetadata,
                                    embeded_inputs: bool| {
                    model_arc.forward(
                        input_ids,
                        positions,
                        kv_caches,
                        input_metadata,
                        crate::graph_extra_arg!($arg, embeded_inputs),
                    )
                };
                let boxed_closure: Box<ModelFn> = Box::new(closure);
                CudaGraphWrapper::new(boxed_closure, $device.as_cuda_device()?.clone().into())
            }, )+
        }
    }};
}

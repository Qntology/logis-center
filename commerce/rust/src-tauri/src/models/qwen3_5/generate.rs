use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, D};
use crate::models::qwen3_5::{Qwen3_5ForCausalLM};
use crate::models::layers::VarBuilderX;
use crate::models::layers::distributed::Comm;
use crate::models::attention::InputMetadata;
use crate::tokenizer::TokenizerModel;
use crate::chat_template::ChatTemplate;
use crate::openai_types::ChatCompletionParameters;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use std::sync::Arc;
use std::rc::Rc;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::models::common::get_logit_processor;

pub struct Qwen3_5GenerateModel {
    pub qwen3_5: Qwen3_5ForCausalLM,
    pub tokenizer: TokenizerModel,
    pub chat_template: ChatTemplate,
    pub device: Device,
    pub config: Config,
    pub kv_caches: Vec<Option<(Tensor, Tensor)>>,
    pub block_tables: Tensor,
    pub eos_token_id: u32,
}

impl Qwen3_5GenerateModel {
    pub fn init(model_path: &str, device: Option<&Device>, dtype: Option<DType>, _use_relay: bool) -> Result<Self> {
        let dev = device.unwrap_or(&Device::Cpu).clone();
        let dtype = dtype.unwrap_or(DType::F16);
        
        let config_path = std::path::Path::new(model_path).join("config.json");
        let config_str = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;
        
        let tokenizer = TokenizerModel::init(model_path)?;
        let chat_template = ChatTemplate::init(model_path)?;
        
        let mut weight_files = vec![];
        for entry in std::fs::read_dir(model_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "st" || ext == "safetensors") {
                weight_files.push(path);
            }
        }
        
        let vb = VarBuilderX::new(&weight_files, dtype, &dev)?;
        let comm = Rc::new(Comm::new_single());
        let progress = Arc::new(RwLock::new(Box::new(crate::utils::progress::NoProgress) as Box<dyn ProgressLike>));
        
        let qwen3_5 = Qwen3_5ForCausalLM::new(&vb, comm, &config, dtype, false, &dev, progress)?;
        
        // Allocate KV cache for Attention layers
        let block_size = 16;
        let max_seq_len = 2048;
        let num_blocks = (max_seq_len + block_size - 1) / block_size;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.hidden_size / config.num_attention_heads;
        let x = match dtype {
            DType::F32 => 4 / 4, // dummy
            _ => 2 / 2,
        };
        // For PagedAttention, x is actually related to the data type size or specific packing.
        // vLLM uses x = 16 / size_of::<T>() for some kernels.
        // Looking at paged_attention.rs: (num_blocks, num_kv_heads, head_size / x, block_size, x)
        // Usually x = 1 for simple cases.
        let x = 1;

        let mut kv_caches = Vec::new();
        let hybrid = crate::utils::resolve_qwen3_hybrid_config(&config);
        
        for layer_type in &hybrid.layer_types {
            if layer_type == "full_attention" {
                let k_cache = Tensor::zeros((num_blocks, num_kv_heads, head_dim / x, block_size, x), dtype, &dev)?;
                let v_cache = Tensor::zeros((num_blocks, num_kv_heads, head_dim, block_size), dtype, &dev)?;
                kv_caches.push(Some((k_cache, v_cache)));
            } else {
                kv_caches.push(None);
            }
        }

        // Single sequence block table: [1, max_blocks]
        let block_indices: Vec<u32> = (0..num_blocks as u32).collect();
        let block_tables = Tensor::from_vec(block_indices, (1, num_blocks), &dev)?;

        Ok(Self {
            qwen3_5,
            tokenizer,
            chat_template,
            device: dev,
            config,
            kv_caches,
            block_tables,
            eos_token_id: 151643, // Default Qwen EOS, should be read from config
        })
    }

    pub async fn generate(
        &mut self,
        params: ChatCompletionParameters,
        cancel_flag: Option<Arc<AtomicBool>>,
        _session_id: Option<String>,
        _kv_name: Option<String>
    ) -> Result<String> {
        let seed = params.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(
            params.temperature.map(|t| t as f32),
            params.top_p.map(|p| p as f32),
            None,
            seed
        );

        let prompt = self.chat_template.apply_chat_template(&params)?;
        let mut tokens = self.tokenizer.text_encode_vec(prompt, true)?;
        
        let mut gen_text = String::new();
        let max_tokens = params.max_tokens.unwrap_or(128) as usize;
        
        self.qwen3_5.reset_mamba_cache()?;
        // Reset KV cache context lens (implicitly done by InputMetadata)

        for step in 0..max_tokens {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { break; }
            }

            let is_prefill = step == 0;
            let input_ids = if is_prefill {
                Tensor::from_vec(tokens.clone(), (tokens.len(),), &self.device)?
            } else {
                Tensor::from_vec(vec![*tokens.last().unwrap()], (1,), &self.device)?
            };

            let seq_len = tokens.len();
            let positions = if is_prefill {
                Tensor::arange(0u32, seq_len as u32, &self.device)?
            } else {
                Tensor::from_vec(vec![(seq_len - 1) as u32], (1,), &self.device)?
            };

            // Prepare InputMetadata
            let block_size = 16;
            let slot_mapping = if is_prefill {
                let mut mapping = Vec::new();
                for i in 0..seq_len {
                    mapping.push(i as i64);
                }
                Tensor::from_vec(mapping, (seq_len,), &self.device)?
            } else {
                Tensor::from_vec(vec![(seq_len - 1) as i64], (1,), &self.device)?
            };

            let metadata = InputMetadata {
                is_prefill,
                sequence_ids: Some(vec![0]),
                mamba_slot_mapping: None, // Will be resolved from sequence_ids
                slot_mapping: slot_mapping.clone(),
                block_tables: Some(self.block_tables.clone()),
                context_lens: Some(Tensor::from_vec(vec![seq_len as u32], (1,), &self.device)?),
                cu_seqlens_q: if is_prefill {
                    Some(Tensor::from_vec(vec![0u32, seq_len as u32], (2,), &self.device)?)
                } else {
                    None
                },
                cu_seqlens_k: None,
                max_seqlen_q: if is_prefill { seq_len } else { 1 },
                max_seqlen_k: seq_len,
                max_context_len: seq_len,
                disable_flash_attn: Some(true),
                seqlens: Some(vec![seq_len as u32]),
                flashinfer_metadata: None,
            };

            let kv_caches_refs: Vec<(Tensor, Tensor)> = self.kv_caches.iter()
                .filter_map(|c| c.as_ref().cloned())
                .collect();

            let logits = self.qwen3_5.forward(
                &input_ids,
                &positions,
                Some(&kv_caches_refs),
                &metadata,
                false
            )?;

            let next_token = logit_processor.sample(&logits.to_dtype(DType::F32)?)?;
            if next_token == self.eos_token_id {
                break;
            }

            tokens.push(next_token);
            let piece = self.tokenizer.token_decode(vec![next_token])?;
            gen_text.push_str(&piece);
            print!("{}", piece);
            std::io::Write::flush(&mut std::io::stdout())?;
        }

        Ok(gen_text)
    }
}

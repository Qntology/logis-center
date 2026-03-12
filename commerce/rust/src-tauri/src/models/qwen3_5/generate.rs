use anyhow::{Result, anyhow};
use serde_json;
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
use crate::utils::logits_processor::get_logit_processor;

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
        // Force BF16 for original models if supported by CUDA
        let dtype = dtype.unwrap_or(DType::BF16);
        
        let config_path = std::path::Path::new(model_path).join("config.json");
        let config_str = std::fs::read_to_string(config_path)?;
        let config_value: serde_json::Value = serde_json::from_str(&config_str)?;
        let mut config: Config = if let Some(text_config) = config_value.get("text_config") {
            let mut cfg: Config = serde_json::from_value(text_config.clone())?;
            if cfg.architectures.is_none() {
                cfg.architectures = config_value.get("architectures").and_then(|a| serde_json::from_value(a.clone()).ok());
            }
            cfg
        } else {
            serde_json::from_str(&config_str)?
        };

        // Propagate nested rope_parameters to root fields for compatibility with ScalingRotaryEmbedding
        if let Some(ref rp) = config.rope_parameters {
            if config.rope_theta.is_none() {
                config.rope_theta = rp.rope_theta;
            }
            if config.partial_rotary_factor.is_none() {
                config.partial_rotary_factor = rp.partial_rotary_factor;
            }
        }
        
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
        let comm = Rc::new(Comm::default());
        let progress = Arc::new(RwLock::new(Box::new(crate::utils::progress::NoProgress) as Box<dyn ProgressLike>));
        
        let qwen3_5 = Qwen3_5ForCausalLM::new_with_prefix(&vb, comm, &config, dtype, false, &dev, progress, Some("model.language_model.".to_string()))?;
        
        // Allocate KV cache for Attention layers
        let block_size = 16;
        let max_seq_len = 2048;
        let num_blocks = (max_seq_len + block_size - 1) / block_size;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim.unwrap_or(config.hidden_size / config.num_attention_heads);
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

        let eos_id = match config.eos_token_id {
            Some(crate::utils::config::EosTokenId::Single(id)) => id,
            Some(crate::utils::config::EosTokenId::Multiple(ref ids)) => ids[0],
            None => 151643, // Fallback
        };

        Ok(Self {
            qwen3_5,
            tokenizer,
            chat_template,
            device: dev,
            config,
            kv_caches,
            block_tables,
            eos_token_id: eos_id,
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
        let prompt_len = tokens.len();
        
        let mut gen_text = String::new();
        let max_tokens = params.max_tokens.unwrap_or(128) as usize;
        
        self.qwen3_5.reset_mamba_cache()?;
        // Ensure mamba slot 0 is assigned to sequence 0
        self.qwen3_5.ensure_mamba_slots_for_sequences(&[0])?;

        for step in 0..max_tokens {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { break; }
            }

            let is_prefill = step == 0;
            let current_seq_len = tokens.len();
            
            let input_ids = if is_prefill {
                Tensor::from_vec(tokens.clone(), (current_seq_len,), &self.device)?
            } else {
                Tensor::from_vec(vec![*tokens.last().unwrap()], (1,), &self.device)?
            };

            let positions = if is_prefill {
                Tensor::arange(0u32, current_seq_len as u32, &self.device)?
            } else {
                Tensor::from_vec(vec![(current_seq_len - 1) as u32], (1,), &self.device)?
            };

            // slot_mapping:
            // For Mamba/GDN layers, this MUST be the persistent slot index for the sequence.
            // For sequence 0, we use slot 0 always.
            let slot_mapping = if is_prefill {
                // In prefill, slot_mapping should match tokens length
                vec![0i64; current_seq_len]
            } else {
                // In decode, slot_mapping should match tokens length (1)
                vec![0i64; 1]
            };
            let slot_mapping_tensor = Tensor::from_vec(slot_mapping, (input_ids.dim(0)?,), &self.device)?;

            // context_lens: Total length including the current new token
            let context_lens = Tensor::from_vec(vec![current_seq_len as u32], (1,), &self.device)?;

            let metadata = InputMetadata {
                is_prefill,
                sequence_ids: Some(vec![0]),
                mamba_slot_mapping: Some(slot_mapping_tensor.clone()), // Fix: Provide explicit slot mapping
                slot_mapping: slot_mapping_tensor,
                block_tables: Some(self.block_tables.clone()),
                context_lens: Some(context_lens),
                cu_seqlens_q: if is_prefill {
                    Some(Tensor::from_vec(vec![0u32, current_seq_len as u32], (2,), &self.device)?)
                } else {
                    Some(Tensor::from_vec(vec![0u32, 1u32], (2,), &self.device)?)
                },
                cu_seqlens_k: None,
                max_seqlen_q: if is_prefill { current_seq_len } else { 1 },
                max_seqlen_k: current_seq_len,
                max_context_len: current_seq_len,
                disable_flash_attn: Some(true),
                seqlens: Some(vec![current_seq_len as u32]),
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

            // Sample from the logits returned by the model. 
            // The model's forward already performs index_select to return only the last token's logits [1, vocab_size].
            let next_tokens = logit_processor.sample(&logits.to_dtype(DType::F32)?, &None)?;
            let next_token = next_tokens[0];
            
            if next_token == self.eos_token_id {
                break;
            }

            tokens.push(next_token);
            let piece = self.tokenizer.token_decode(vec![next_token])?;
            gen_text.push_str(&piece);
            print!("{}", piece);
            std::io::Write::flush(&mut std::io::stdout())?;
            
            // Check for assistant turn end in Qwen
            if piece.contains("<|im_end|>") || piece.contains("<|endoftext|>") {
                break;
            }
        }

        Ok(gen_text)
    }
}

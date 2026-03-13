use anyhow::{Result, anyhow};
use serde_json;
use candle_core::{DType, Device, Tensor, D, IndexOp};
use crate::models::qwen3_5::{Qwen3_5ForCausalLM};
use crate::models::layers::VarBuilderX;
use crate::models::layers::distributed::Comm;
use attention_rs::InputMetadata;
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
        
        config.extra_config_json = Some(config_str);

        if let Some(ref rp) = config.rope_parameters {
            if rp.rope_theta.is_some() { config.rope_theta = rp.rope_theta; }
            if rp.partial_rotary_factor.is_some() { config.partial_rotary_factor = rp.partial_rotary_factor; }
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
        
        let is_interleaved = config.rope_parameters.as_ref().and_then(|p| p.mrope_interleaved).unwrap_or(false);
        let qwen3_5 = Qwen3_5ForCausalLM::new_with_prefix(&vb, comm, &config, dtype, is_interleaved, &dev, progress, Some("model.language_model.".to_string()))?;
        
        let block_size = 16;
        let max_seq_len = 2048;
        let num_blocks = (max_seq_len + block_size - 1) / block_size;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim.unwrap_or(config.hidden_size / config.num_attention_heads);
        let x = 16 / dtype.size_in_bytes();

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

        let block_indices: Vec<u32> = (0..num_blocks as u32).collect();
        let block_tables = Tensor::from_vec(block_indices, (1, num_blocks), &dev)?;

        let eos_id = match config.eos_token_id {
            Some(crate::utils::config::EosTokenId::Single(id)) => id,
            Some(crate::utils::config::EosTokenId::Multiple(ref ids)) => ids[0],
            None => 151643,
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
        
        let mut gen_text = String::new();
        let max_tokens = params.max_tokens.unwrap_or(128) as usize;
        
        self.qwen3_5.reset_mamba_cache()?;
        // Official vllm.rs expects slots to be allocated per sequence
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
                Tensor::arange(0u32, current_seq_len as u32, &self.device)?.to_dtype(DType::I64)?
            } else {
                Tensor::from_vec(vec![(current_seq_len - 1) as u32], (1,), &self.device)?.to_dtype(DType::I64)?
            };

            // CRITICAL: mamba_slot_mapping should be [0] (size 1) for sequence index 0
            // Even in prefill, it's used to identify which sequence's state slot to update.
            let mamba_slot_mapping = vec![0i64; 1];
            
            let attn_slot_mapping = if is_prefill {
                (0..current_seq_len as i64).collect::<Vec<_>>()
            } else {
                vec![(current_seq_len - 1) as i64]
            };
            
            let mamba_slot_mapping_tensor = Tensor::from_vec(mamba_slot_mapping, (1,), &self.device)?;
            let attn_slot_mapping_tensor = Tensor::from_vec(attn_slot_mapping, (input_ids.dim(0)?,), &self.device)?;

            let metadata = InputMetadata {
                is_prefill,
                sequence_ids: Some(vec![0]),
                mamba_slot_mapping: Some(mamba_slot_mapping_tensor),
                slot_mapping: attn_slot_mapping_tensor,
                block_tables: Some(self.block_tables.clone()),
                context_lens: Some(Tensor::from_vec(vec![current_seq_len as u32], (1,), &self.device)?),
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

            let logits = logits.reshape((1, ()))?;

            let next_tokens = logit_processor.sample(&logits.to_dtype(DType::F32)?, &None)?;
            let next_token = next_tokens[0];
            
            let piece = self.tokenizer.token_decode(vec![next_token])?;
            // println!("[Step {}] Token ID: {}, Piece: '{}'", step, next_token, piece);

            if next_token == self.eos_token_id {
                break;
            }

            tokens.push(next_token);
            gen_text.push_str(&piece);
            print!("{}", piece);
            std::io::Write::flush(&mut std::io::stdout())?;
            
            if piece.contains("<|im_end|>") || piece.contains("<|endoftext|>") {
                break;
            }
        }

        Ok(gen_text)
    }
}

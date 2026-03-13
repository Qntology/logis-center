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
use crate::utils::config::{Config, SamplingParams};
use crate::utils::progress::ProgressLike;
use std::sync::Arc;
use std::rc::Rc;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::utils::logits_processor::get_logit_processor;

use crate::utils::downloader::ModelPaths;
use std::path::PathBuf;

pub struct Qwen3_5GenerateModel {
    pub qwen3_5: Qwen3_5ForCausalLM,
    pub tokenizer: TokenizerModel,
    pub chat_template: ChatTemplate,
    pub device: Device,
    pub config: Config,
    pub kv_caches: Vec<Option<(Tensor, Tensor)>>,
    pub block_tables: Vec<u32>,
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

        // Official vllm-rs uses rope_scaling instead of rope_parameters
        if let Some(ref rs) = config.rope_scaling {
            if let Some(v) = rs.get("rope_theta") {
                if let Some(theta) = v.as_f64() { config.rope_theta = Some(theta); }
            }
            if let Some(v) = rs.get("partial_rotary_factor") {
                if let Some(factor) = v.as_f64() { config.partial_rotary_factor = Some(factor as f32); }
            }
        }
        
        let tokenizer = TokenizerModel::init(model_path)?;
        let chat_template = ChatTemplate::init(model_path)?;
        
        let model_path_buf = PathBuf::from(model_path);
        let mut weight_files = vec![];
        for entry in std::fs::read_dir(model_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "st" || ext == "safetensors") {
                weight_files.push(path);
            }
        }
        
        let model_pathes = ModelPaths {
            tokenizer_filename: model_path_buf.join("tokenizer.json"),
            tokenizer_config_filename: model_path_buf.join("tokenizer_config.json"),
            config_filename: model_path_buf.join("config.json"),
            generation_config_filename: model_path_buf.join("generation_config.json"),
            filenames: weight_files,
            chat_template_filename: None,
        };
        
        let vb = VarBuilderX::new(&model_pathes, false, dtype, &dev)?;
        let comm = Rc::new(Comm::default());
        let progress = Arc::new(RwLock::new(Box::new(crate::utils::progress::NoProgress) as Box<dyn ProgressLike>));
        
        let is_interleaved = false; // Forced to false to match vllm-rs baseline
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

        // Use sequential block allocation for simple local runner
        let block_tables = (0..num_blocks as u32).collect::<Vec<_>>();

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
        let sampling_params = params.to_sampling_params();
        let mut logit_processor = get_logit_processor(seed, &sampling_params);

        let prompt = self.chat_template.apply_chat_template(&params)?;
        println!("📝 Full Prompt (len={}):\n---\n{}\n---", prompt.len(), prompt);
        let mut tokens = self.tokenizer.text_encode_vec(prompt, false)?;
        println!("🔢 Tokenized prompt (len={}): {:?}", tokens.len(), tokens);
        for (i, t) in tokens.iter().enumerate() {
            println!("  Token[{}]: {}", i, t);
        }
        
        let mut gen_text = String::new();
        let max_tokens = params.max_tokens.unwrap_or(128) as usize;
        
        self.qwen3_5.reset_mamba_cache()?;
        // IMPORTANT: Let the model handle sequence slot allocation via internal MambaCache
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
            println!("[Step {}] input_ids shape: {:?}", step, input_ids.shape());

            let positions = if is_prefill {
                Tensor::arange(0u32, current_seq_len as u32, &self.device)?.to_dtype(DType::I64)?
            } else {
                Tensor::from_vec(vec![(current_seq_len - 1) as u32], (1,), &self.device)?.to_dtype(DType::I64)?
            };

            // Calculate attn_slot_mapping using block_table logic (Official runner style)
            let mut slot_mapping = Vec::new();
            let block_size = 16;
            if is_prefill {
                for i in 0..current_seq_len {
                    let block_idx = i / block_size;
                    let block_offset = i % block_size;
                    let physical_idx = (self.block_tables[block_idx] * block_size as u32) as i64 + block_offset as i64;
                    slot_mapping.push(physical_idx);
                }
            } else {
                let i = current_seq_len - 1;
                let block_idx = i / block_size;
                let block_offset = i % block_size;
                let physical_idx = (self.block_tables[block_idx] * block_size as u32) as i64 + block_offset as i64;
                slot_mapping.push(physical_idx);
            }
            let attn_slot_mapping_tensor = Tensor::from_vec(slot_mapping, (input_ids.dim(0)?,), &self.device)?;

            let block_tables_tensor = Tensor::from_vec(self.block_tables.clone(), (1, self.block_tables.len()), &self.device)?;

            let metadata = InputMetadata {
                is_prefill,
                sequence_ids: Some(vec![0]),
                mamba_slot_mapping: None, // Let resolve_seq_slots handle it via sequence_ids 
                slot_mapping: attn_slot_mapping_tensor,
                block_tables: Some(block_tables_tensor),
                context_lens: Some(Tensor::from_vec(vec![current_seq_len as u32], (1,), &self.device)?),
                cu_seqlens_q: if is_prefill {
                    Some(Tensor::from_vec(vec![0u32, current_seq_len as u32], (2,), &self.device)?)
                } else {
                    None
                },
                cu_seqlens_k: if is_prefill {
                    Some(Tensor::from_vec(vec![0u32, current_seq_len as u32], (2,), &self.device)?)
                } else {
                    None
                },
                max_seqlen_q: if is_prefill { current_seq_len } else { 0 },
                max_seqlen_k: if is_prefill { current_seq_len } else { 0 },
                max_context_len: current_seq_len,
                disable_flash_attn: None, // Use default (auto)
                seqlens: if is_prefill { Some(vec![current_seq_len as u32]) } else { None },
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

            // IMPORTANT: forward already selects the last token's logits
            let logits = logits.reshape((1, ()))?;

            let next_tokens = logit_processor.sample(&logits.to_dtype(DType::F32)?, &None)?;
            let next_token = next_tokens[0];
            
            let piece = self.tokenizer.token_decode(vec![next_token])?;

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

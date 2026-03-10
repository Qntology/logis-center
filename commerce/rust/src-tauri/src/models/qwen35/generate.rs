use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{
    chat_template::ChatTemplate,
    tokenizer::TokenizerModel,
    models::qwen35::{
        config::Qwen3_5Config,
        quantized_model::{QuantizedQwen3_5Model, QuantizedRegistry, DiskStateManager, LayerContext},
    },
    openai_types::ChatCompletionParameters,
    utils::{
        get_logit_processor,
    },
};

pub struct Qwen3_5GenerateModel {
    pub qwen3_5: QuantizedQwen3_5Model,
    pub tokenizer: TokenizerModel,
    pub device: Device,
    pub registry: QuantizedRegistry,
    pub chat_template: ChatTemplate,
    pub eos_token_id: u32,
}

impl Qwen3_5GenerateModel {
    pub fn init(
        model_path: &str,
        device: Option<&Device>,
        _dtype: Option<DType>,
        _prefill_bake: bool,
    ) -> Result<Self> {
        let model_path = std::path::Path::new(model_path);
        let config_path = model_path.join("config.json");
        let config: Qwen3_5Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        
        let tokenizer = TokenizerModel::init(model_path.to_str().unwrap())?;

        let dev = match device {
            Some(d) => d.clone(),
            None => Device::Cpu,
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[] as &[&str], DType::F16, &dev)? };
        let qwen3_5 = QuantizedQwen3_5Model::new(vb, config)?;

        let mut model_files = vec![];
        for entry in std::fs::read_dir(model_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                model_files.push(path.to_str().unwrap().to_string());
            }
        }
        let registry = QuantizedRegistry::new(&model_files)?;

        let chat_template = ChatTemplate::init(model_path.to_str().unwrap())?;

        Ok(Self {
            qwen3_5,
            tokenizer,
            device: dev,
            registry,
            chat_template,
            eos_token_id: 151643,
        })
    }

    pub async fn generate(
        &mut self, 
        mes: ChatCompletionParameters, 
        cancel_flag: Option<Arc<AtomicBool>>,
        _session_id: Option<String>,
        _kv_name: Option<String>
    ) -> Result<String> {
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(
            mes.temperature.map(|t| t as f32), 
            mes.top_p.map(|p| p as f32), 
            None, 
            seed
        );
        
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let full_input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
            
        let full_seq_len = full_input_ids.dim(1)?;
        
        let mut seqlen_offset = 0;
        let chunk_size = 64; 
        let mut last_logits = None;

        self.qwen3_5.clear_cache();
        let mut disk_manager = DiskStateManager::new(self.qwen3_5.model.layers.len())?;

        // 0.8B default hyperparams for Relay I/O
        let num_layers = self.qwen3_5.model.layers.len();
        let num_kv_heads = 4; 
        let head_dim = 128;   
        let hidden_size = 1024; 

        println!("[MODEL-QWEN35] Starting HYBRID Prefill (True Layer-by-Layer Relay)...");

        // ====================================================================
        // [PHASE 1] PREFILL: Layer-by-Layer (VRAM-safe)
        // ====================================================================
        while seqlen_offset < full_seq_len {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { break; }
            }

            let remaining = full_seq_len - seqlen_offset;
            let current_chunk = remaining.min(chunk_size);
            let input_ids = full_input_ids.narrow(1, seqlen_offset, current_chunk)?;

            // 1. Embeddings
            let mut h = self.qwen3_5.model.embed.forward(&input_ids, &self.registry)?;
            
            // RoPE
            let pos_vec: Vec<u32> = (seqlen_offset as u32..(seqlen_offset + current_chunk) as u32).collect();
            let pos = Tensor::from_vec(pos_vec, (1, current_chunk), &self.device)?.unsqueeze(0)?.broadcast_as((3, input_ids.dim(0)?, current_chunk))?.contiguous()?;
            let (cos, sin) = self.qwen3_5.model.rotary.forward(&pos, DType::F16, self.qwen3_5.model.mrope.clone())?;
            
            // Mask
            let mask = if current_chunk <= 1 && seqlen_offset == 0 { None } else { 
                Some(crate::utils::tensor_utils::prepare_causal_attention_mask(input_ids.dim(0)?, current_chunk, seqlen_offset, &self.device)?.to_dtype(DType::F16)?) 
            };

            // 2. Layer Relay
            for i in 0..num_layers {
                if seqlen_offset > 0 {
                   let lt = if self.qwen3_5.model.layers[i].self_attn.is_some() { "attention" } else { "linear_attention" };
                   if lt == "attention" {
                       let s_k = candle_core::Shape::from((1, num_kv_heads, seqlen_offset, head_dim));
                       let s_v = s_k.clone();
                       if let Ok(LayerContext::Attention { k, v }) = disk_manager.load_layer_context(i, lt, &s_k, &s_v, &self.device) {
                           self.qwen3_5.model.layers[i].self_attn.as_mut().unwrap().kv_cache = Some((k, v));
                       }
                   } else {
                       let s_state = candle_core::Shape::from((1, 1, hidden_size)); 
                       if let Ok(LayerContext::DeltaNet { state }) = disk_manager.load_layer_context(i, lt, &s_state, &s_state, &self.device) {
                           self.qwen3_5.model.layers[i].linear_attn.as_mut().unwrap().delta_state = Some(state.to_device(&self.device)?);
                       }
                   }
                }

                h = self.qwen3_5.model.layers[i].forward(&h, &cos, &sin, mask.as_ref(), &self.registry)?;

                // Offload & Clear VRAM
                if let Some(ref mut sa) = self.qwen3_5.model.layers[i].self_attn {
                    if let Some((k, v)) = sa.kv_cache.take() { 
                        disk_manager.save_layer_context(i, &LayerContext::Attention { k, v })?;
                    }
                }
                if let Some(ref mut la) = self.qwen3_5.model.layers[i].linear_attn {
                    if let Some(st) = la.delta_state.take() {
                        disk_manager.save_layer_context(i, &LayerContext::DeltaNet { state: st })?;
                    }
                }

                #[cfg(feature = "cuda")]
                self.device.synchronize()?; 
            }

            // Head
            let last_h = self.qwen3_5.model.norm.forward(&h, &self.registry)?;
            let logits = self.qwen3_5.head.forward(&last_h.narrow(1, current_chunk - 1, 1)?, &self.registry)?;
            
            seqlen_offset += current_chunk;
            last_logits = Some(logits);

            println!("[PREFILL-RELAY] Processed {}/{} tokens...", seqlen_offset, full_seq_len);
        }

        // ====================================================================
        // [PHASE 2] TRANSITION
        // ====================================================================
        println!("[MODEL-QWEN35] Prefill complete. Loading weights to VRAM for Decode...");
        self.qwen3_5.load_all_to_vram(&self.registry, &self.device)?;

        // ====================================================================
        // [PHASE 3] DECODING: Streaming Relay
        // ====================================================================
        println!("[MODEL-QWEN35] Starting Streaming Decoding...");
        let mut logits = last_logits.ok_or(anyhow::anyhow!("No logits generated"))?;
        let mut gen_text = String::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);

        for _i in 0..sample_len {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { break; }
            }

            let current_logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&current_logits)?;
            
            if next_token == self.eos_token_id { break; }

            let piece = self.tokenizer.token_decode(vec![next_token])?;
            gen_text.push_str(&piece);

            let input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?.to_dtype(DType::U32)?;
            
            let mut h = self.qwen3_5.model.embed.forward(&input_ids, &self.registry)?; 
            let pos_vec: Vec<u32> = vec![seqlen_offset as u32];
            let pos = Tensor::from_vec(pos_vec, (1, 1), &self.device)?.unsqueeze(0)?.broadcast_as((3, 1, 1))?.contiguous()?;
            let (cos, sin) = self.qwen3_5.model.rotary.forward(&pos, DType::F16, self.qwen3_5.model.mrope.clone())?;

            for i in 0..num_layers {
                let lt = if self.qwen3_5.model.layers[i].self_attn.is_some() { "attention" } else { "linear_attention" };
                
                // SSD -> VRAM Load
                if lt == "attention" {
                    let s_k = candle_core::Shape::from((1, num_kv_heads, seqlen_offset, head_dim));
                    let s_v = s_k.clone();
                    if let Ok(LayerContext::Attention { k, v }) = disk_manager.load_layer_context(i, lt, &s_k, &s_v, &self.device) {
                        self.qwen3_5.model.layers[i].self_attn.as_mut().unwrap().kv_cache = Some((k, v));
                    }
                } else {
                    let s_state = candle_core::Shape::from((1, 1, hidden_size)); 
                    if let Ok(LayerContext::DeltaNet { state }) = disk_manager.load_layer_context(i, lt, &s_state, &s_state, &self.device) {
                        self.qwen3_5.model.layers[i].linear_attn.as_mut().unwrap().delta_state = Some(state.to_device(&self.device)?);
                    }
                }

                h = self.qwen3_5.model.layers[i].forward(&h, &cos, &sin, None, &self.registry)?;

                // VRAM -> SSD Save & Clear
                if let Some(ref mut sa) = self.qwen3_5.model.layers[i].self_attn {
                    if let Some((k, v)) = sa.kv_cache.take() { 
                        disk_manager.save_layer_context(i, &LayerContext::Attention { k, v })?;
                    }
                }
                if let Some(ref mut la) = self.qwen3_5.model.layers[i].linear_attn {
                    if let Some(st) = la.delta_state.take() {
                        disk_manager.save_layer_context(i, &LayerContext::DeltaNet { state: st })?;
                    }
                }

                #[cfg(feature = "cuda")]
                self.device.synchronize()?; 
            }

            let last_h = self.qwen3_5.model.norm.forward(&h, &self.registry)?;
            logits = self.qwen3_5.head.forward(&last_h, &self.registry)?;
            
            seqlen_offset += 1;
        }

        self.qwen3_5.clear_cache();
        Ok(gen_text)
    }
}

pub fn init_bake_worker() {
    println!("[MODEL-QWEN35] Bake worker initialized for text-only prefill.");
}

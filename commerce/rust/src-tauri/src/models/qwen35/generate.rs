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
        quantized_model::{QuantizedQwen3_5Model, QuantizedRegistry, KVStateManager, LayerContext},
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

        let mut model_files = vec![];
        for entry in std::fs::read_dir(model_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                model_files.push(path.to_str().unwrap().to_string());
            }
        }

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::F16, &dev)? };
        let qwen3_5 = QuantizedQwen3_5Model::new(vb, config)?;

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
        // Use memory-based KV cache for performance
        let mut kv_manager = KVStateManager::new(self.qwen3_5.model.layers.len(), true)?;

        let num_layers = self.qwen3_5.model.layers.len();

        println!("[MODEL-QWEN35] Starting FULL-RELAY Prefill (Layer-by-Layer weights & cache)...");

        // ====================================================================
        // [PHASE 1] PREFILL: Layer-by-Layer Relay (VRAM minimum)
        // ====================================================================
        while seqlen_offset < full_seq_len {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { break; }
            }

            let remaining = full_seq_len - seqlen_offset;
            let current_chunk = remaining.min(chunk_size);
            let input_ids = full_input_ids.narrow(1, seqlen_offset, current_chunk)?;

            // 1. Embeddings (Dynamic Load/Drop)
            self.qwen3_5.model.embed.load_to_vram(&self.registry, &self.device)?;
            let mut h = self.qwen3_5.model.embed.forward(&input_ids, &self.registry)?;
            self.qwen3_5.model.embed.clear_vram();
            
            // RoPE & Mask
            let pos_vec: Vec<u32> = (seqlen_offset as u32..(seqlen_offset + current_chunk) as u32).collect();
            let pos = Tensor::from_vec(pos_vec, (1, current_chunk), &self.device)?.unsqueeze(0)?.broadcast_as((3, input_ids.dim(0)?, current_chunk))?.contiguous()?;
            let (cos, sin) = self.qwen3_5.model.rotary.forward(&pos, DType::F16, self.qwen3_5.model.mrope.clone())?;
            let mask = if current_chunk <= 1 && seqlen_offset == 0 { None } else { 
                Some(crate::utils::tensor_utils::prepare_causal_attention_mask(input_ids.dim(0)?, current_chunk, seqlen_offset, &self.device)?.to_dtype(DType::F16)?) 
            };

            // 2. Layer Relay
            for i in 0..num_layers {
                let layer = &mut self.qwen3_5.model.layers[i];
                
                // [WEIGHTS UP]
                layer.load_to_vram(&self.registry, &self.device)?;

                // [KV LOAD]
                if seqlen_offset > 0 {
                   if let Some(ref mut sa) = layer.self_attn {
                       let s_k = candle_core::Shape::from((1, sa.nkv, seqlen_offset, sa.hd));
                       let s_v = s_k.clone();
                       if let Ok(LayerContext::Attention { k, v }) = kv_manager.load_layer_context(i, "attention", &s_k, &s_v, &self.device) {
                           sa.kv_cache = Some((k, v));
                       }
                   } else if let Some(ref mut la) = layer.linear_attn {
                       let s_state = candle_core::Shape::from((1, la.nv, la.dk, la.dv)); 
                       if let Ok(LayerContext::DeltaNet { state }) = kv_manager.load_layer_context(i, "linear_attention", &s_state, &s_state, &self.device) {
                           la.recurrent_state_cache = Some(state.to_device(&self.device)?);
                       }
                   }
                }

                // [COMPUTE]
                h = layer.forward(&h, &cos, &sin, mask.as_ref(), &self.registry)?;

                // [KV SAVE]
                if let Some(ref mut sa) = layer.self_attn {
                    if let Some((k, v)) = sa.kv_cache.take() { 
                        kv_manager.save_layer_context(i, LayerContext::Attention { k, v })?;
                    }
                }
                if let Some(ref mut la) = layer.linear_attn {
                    if let Some(st) = la.recurrent_state_cache.take() {
                        kv_manager.save_layer_context(i, LayerContext::DeltaNet { state: st })?;
                    }
                }

                // [WEIGHTS DOWN]
                layer.clear_vram();

                #[cfg(feature = "cuda")]
                self.device.synchronize()?; 
                
                if i % 8 == 0 || i == num_layers - 1 {
                    println!("[VRAM-RELAY] Prefill Chunk {}/{} - Layer {}/{} Weights(↑↓) Cache(MEM)", seqlen_offset + current_chunk, full_seq_len, i+1, num_layers);
                }
            }

            // Head (Dynamic Load/Drop)
            self.qwen3_5.model.norm.load_to_vram(&self.registry, &self.device)?;
            let last_h = self.qwen3_5.model.norm.forward(&h, &self.registry)?;
            self.qwen3_5.model.norm.clear_vram();

            self.qwen3_5.head.load_to_vram(&self.registry, &self.device)?;
            let logits = self.qwen3_5.head.forward(&last_h.narrow(1, current_chunk - 1, 1)?, &self.registry)?;
            self.qwen3_5.head.clear_vram();
            
            seqlen_offset += current_chunk;
            last_logits = Some(logits);
        }

        // ====================================================================
        // [PHASE 2] TRANSITION: Load ALL to VRAM for FAST Decoding
        // ====================================================================
        println!("[MODEL-QWEN35] Prefill complete. Loading ALL weights to VRAM for fast Decoding...");
        self.qwen3_5.load_all_to_vram(&self.registry, &self.device)?;

        // ====================================================================
        // [PHASE 3] DECODING: High Speed (Persistent Weights in VRAM)
        // ====================================================================
        let mut logits = last_logits.ok_or(anyhow::anyhow!("No logits generated"))?;
        let mut gen_text = String::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);

        for step in 0..sample_len {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { break; }
            }

            let current_logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            
            // [DEBUG] Top-5 Tokens
            if step < 3 {
                let logits_v = current_logits.to_vec1::<f32>()?;
                let mut indexed: Vec<(usize, f32)> = logits_v.into_iter().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                println!("\n[DEBUG-TOKEN] Step {}: ", step);
                for i in 0..5 {
                    let token_text = self.tokenizer.token_decode(vec![indexed[i].0 as u32]).unwrap_or_default();
                    println!("  - Top {}: ID={}, Prob={:.4}, Text='{}'", i+1, indexed[i].0, indexed[i].1, token_text);
                }
            }

            let next_token = logit_processor.sample(&current_logits)?;
            
            if next_token == self.eos_token_id { break; }

            let piece = self.tokenizer.token_decode(vec![next_token])?;
            print!("{}", piece); // Stream to console
            gen_text.push_str(&piece);

            let input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?.to_dtype(DType::U32)?;
            
            // Embeddings (Already in VRAM)
            let mut h = self.qwen3_5.model.embed.forward(&input_ids, &self.registry)?; 

            let pos_vec: Vec<u32> = vec![seqlen_offset as u32];
            let pos = Tensor::from_vec(pos_vec, (1, 1), &self.device)?.unsqueeze(0)?.broadcast_as((3, 1, 1))?.contiguous()?;
            let (cos, sin) = self.qwen3_5.model.rotary.forward(&pos, DType::F16, self.qwen3_5.model.mrope.clone())?;

            for i in 0..num_layers {
                let layer = &mut self.qwen3_5.model.layers[i];

                // [KV LOAD] - From Memory Cache
                if let Some(ref mut sa) = layer.self_attn {
                    let s_k = candle_core::Shape::from((1, sa.nkv, seqlen_offset, sa.hd));
                    let s_v = s_k.clone();
                    if let Ok(LayerContext::Attention { k, v }) = kv_manager.load_layer_context(i, "attention", &s_k, &s_v, &self.device) {
                        sa.kv_cache = Some((k, v));
                    }
                } else if let Some(ref mut la) = layer.linear_attn {
                    let s_state = candle_core::Shape::from((1, la.nv, la.dk, la.dv)); 
                    if let Ok(LayerContext::DeltaNet { state }) = kv_manager.load_layer_context(i, "linear_attention", &s_state, &s_state, &self.device) {
                        la.recurrent_state_cache = Some(state.to_device(&self.device)?);
                    }
                }

                // [COMPUTE]
                h = layer.forward(&h, &cos, &sin, None, &self.registry)?;

                // [KV SAVE] - To Memory Cache
                if let Some(ref mut sa) = layer.self_attn {
                    if let Some((k, v)) = sa.kv_cache.take() { 
                        kv_manager.save_layer_context(i, LayerContext::Attention { k, v })?;
                    }
                }
                if let Some(ref mut la) = layer.linear_attn {
                    if let Some(st) = la.recurrent_state_cache.take() {
                        kv_manager.save_layer_context(i, LayerContext::DeltaNet { state: st })?;
                    }
                }
            }

            // Norm & Head (Already in VRAM)
            let last_h = self.qwen3_5.model.norm.forward(&h, &self.registry)?;
            logits = self.qwen3_5.head.forward(&last_h, &self.registry)?;
            
            seqlen_offset += 1;
        }

        self.qwen3_5.clear_cache();
        println!("\n[MODEL-QWEN35] Generation complete.");
        Ok(gen_text)
    }
}

pub fn init_bake_worker() {
    println!("[MODEL-QWEN35] Bake worker initialized for text-only prefill.");
}

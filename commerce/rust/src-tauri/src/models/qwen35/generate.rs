use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use crate::models::qwen35::quantized_model::{QuantizedQwen3_5Model, QuantizedRegistry, LayerContext};
use crate::tokenizer::TokenizerModel;
use crate::chat_template::ChatTemplate;
use crate::openai_types::ChatCompletionParameters;
use crate::models::qwen35::config::Qwen3_5Config;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::models::common::get_logit_processor;

pub struct Qwen3_5GenerateModel {
    pub qwen3_5: QuantizedQwen3_5Model,
    pub tokenizer: TokenizerModel,
    pub chat_template: ChatTemplate,
    pub device: Device,
    pub registry: QuantizedRegistry,
    pub eos_token_id: u32,
    // [FIX] Persistent cache storage for RELAY mode
    pub layer_caches: Vec<Option<LayerContext>>,
}

impl Qwen3_5GenerateModel {
    pub fn init(model_path: &str, device: Option<&Device>, _dtype: Option<DType>, use_relay: bool) -> Result<Self> {
        let dev = device.unwrap_or(&Device::Cpu).clone();
        println!("[MODEL-QWEN35] Initializing Hybrid Engine ({} Mode)...", if use_relay { "RELAY" } else { "FULL" });
        
        let config_path = std::path::Path::new(model_path).join("config.json");
        let config_str = std::fs::read_to_string(config_path)?;
        let config: Qwen3_5Config = serde_json::from_str(&config_str)?;
        
        let tokenizer = TokenizerModel::init(model_path)?;
        let chat_template = ChatTemplate::init(model_path)?;
        
        let mut model_files = vec![];
        for entry in std::fs::read_dir(model_path)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".st") || name.ends_with(".safetensors") {
                model_files.push(entry.path().to_str().unwrap().to_string());
            }
        }
        println!("[MODEL-QWEN35] Loading {} model files...", model_files.len());
        let registry = QuantizedRegistry::new(&model_files)?;
        
        let map = std::collections::HashMap::new();
        let vb = candle_nn::VarBuilder::from_tensors(map, DType::F16, &dev);
        let qwen3_5 = QuantizedQwen3_5Model::new(vb, config.clone())?;
        
        Ok(Self {
            qwen3_5,
            tokenizer,
            chat_template,
            device: dev,
            registry,
            eos_token_id: 248046, 
            layer_caches: vec![None; config.text_config.num_hidden_layers],
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
            Some(50),
            seed
        );

        self.qwen3_5.clear_cache();
        self.layer_caches.iter_mut().for_each(|c| *c = None);

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let token_ids = self.tokenizer.text_encode_vec(mes_render, true)?;
        let full_input_ids = Tensor::from_slice(&token_ids, (1, token_ids.len()), &self.device)?.to_dtype(DType::U32)?;
        let full_seq_len = full_input_ids.dim(1)?;

        println!("[MODEL-QWEN35] Starting FULL-RELAY Prefill...");
        let mut last_logits = None;
        
        // Chunked prefill for memory efficiency
        for i in 0..full_seq_len {
            let input_id = full_input_ids.narrow(1, i, 1)?;
            let mut x = self.qwen3_5.model.embed.forward(&input_id, &self.registry)?;
            
            let pos_ids = Tensor::new(&[i as u32], &self.device)?.reshape((1, 1))?.broadcast_as((3, 1, 1))?;
            let (cos, sin) = self.qwen3_5.model.rotary.forward(&pos_ids, DType::F16, self.qwen3_5.model.mrope.clone(), self.qwen3_5.model.mrope_interleaved)?;

            for (l_idx, layer) in self.qwen3_5.model.layers.iter_mut().enumerate() {
                layer.load_to_vram(&self.registry, &self.device)?;
                
                // [RESTORE CACHE]
                if let Some(ctx) = self.layer_caches[l_idx].take() {
                    match ctx {
                        LayerContext::Attention { k, v } => if let Some(ref mut sa) = layer.self_attn { sa.kv_cache = Some((k, v)); },
                        LayerContext::DeltaNet { state, conv } => if let Some(ref mut la) = layer.linear_attn { la.recurrent_state_cache = Some(state); la.conv_state_cache = Some(conv); },
                    }
                }

                x = layer.forward(&x, &cos, &sin, None, &self.registry)?;
                
                // [SAVE CACHE]
                if let Some(ref sa) = layer.self_attn { if let Some(ref kv) = sa.kv_cache { self.layer_caches[l_idx] = Some(LayerContext::Attention { k: kv.0.clone(), v: kv.1.clone() }); } }
                if let Some(ref la) = layer.linear_attn { 
                    if let (Some(ref s), Some(ref c)) = (&la.recurrent_state_cache, &la.conv_state_cache) {
                        self.layer_caches[l_idx] = Some(LayerContext::DeltaNet { state: s.clone(), conv: c.clone() });
                    }
                }

                layer.clear_vram();
            }
            
            if i == full_seq_len - 1 {
                self.qwen3_5.model.norm.load_to_vram(&self.registry, &self.device)?;
                let out = self.qwen3_5.model.norm.forward(&x, &self.registry)?;
                self.qwen3_5.model.norm.clear_vram();
                
                self.qwen3_5.head.load_to_vram(&self.registry, &self.device)?;
                let mut logits = self.qwen3_5.head.forward(&out, &self.registry)?;
                if self.qwen3_5.tie { logits = logits.affine(1.0 / self.qwen3_5.hidden_size.sqrt(), 0.0)?; }
                last_logits = Some(logits);
                self.qwen3_5.head.clear_vram();
            }
        }

        let mut logits = last_logits.ok_or(anyhow!("No logits generated"))?;
        let mut gen_text = String::new();
        let sample_len = mes.max_tokens.unwrap_or(100) as usize;
        let mut all_tokens = token_ids.clone();
        let repetition_penalty = 1.2f32;

        for step in 0..sample_len {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { break; } }

            let mut current_logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            if repetition_penalty != 1.0 {
                let mut logits_vec = current_logits.to_vec1::<f32>()?;
                for &token in all_tokens.iter() {
                    let id = token as usize;
                    if id < logits_vec.len() {
                        if logits_vec[id] > 0.0 { logits_vec[id] /= repetition_penalty; }
                        else { logits_vec[id] *= repetition_penalty; }
                    }
                }
                current_logits = Tensor::from_vec(logits_vec, current_logits.dims(), current_logits.device())?;
            }

            let next_token = logit_processor.sample(&current_logits)?;
            if next_token == self.eos_token_id { break; }

            let piece = self.tokenizer.token_decode(vec![next_token])?;
            print!("{}", piece);
            std::io::Write::flush(&mut std::io::stdout())?;
            gen_text.push_str(&piece);
            all_tokens.push(next_token);

            // Decoding Step
            let input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?.to_dtype(DType::U32)?;
            let mut x = self.qwen3_5.model.embed.forward(&input_ids, &self.registry)?;
            let pos_ids = Tensor::new(&[(full_seq_len + step) as u32], &self.device)?.reshape((1, 1))?.broadcast_as((3, 1, 1))?;
            let (cos, sin) = self.qwen3_5.model.rotary.forward(&pos_ids, DType::F16, self.qwen3_5.model.mrope.clone(), self.qwen3_5.model.mrope_interleaved)?;

            for (l_idx, layer) in self.qwen3_5.model.layers.iter_mut().enumerate() {
                layer.load_to_vram(&self.registry, &self.device)?;
                
                // [RESTORE CACHE]
                if let Some(ctx) = self.layer_caches[l_idx].take() {
                    match ctx {
                        LayerContext::Attention { k, v } => if let Some(ref mut sa) = layer.self_attn { sa.kv_cache = Some((k, v)); },
                        LayerContext::DeltaNet { state, conv } => if let Some(ref mut la) = layer.linear_attn { la.recurrent_state_cache = Some(state); la.conv_state_cache = Some(conv); },
                    }
                }

                x = layer.forward(&x, &cos, &sin, None, &self.registry)?;
                
                // [SAVE CACHE]
                if let Some(ref sa) = layer.self_attn { if let Some(ref kv) = sa.kv_cache { self.layer_caches[l_idx] = Some(LayerContext::Attention { k: kv.0.clone(), v: kv.1.clone() }); } }
                if let Some(ref la) = layer.linear_attn { 
                    if let (Some(ref s), Some(ref c)) = (&la.recurrent_state_cache, &la.conv_state_cache) {
                        self.layer_caches[l_idx] = Some(LayerContext::DeltaNet { state: s.clone(), conv: c.clone() });
                    }
                }

                layer.clear_vram();
            }
            
            self.qwen3_5.model.norm.load_to_vram(&self.registry, &self.device)?;
            let out = self.qwen3_5.model.norm.forward(&x, &self.registry)?;
            self.qwen3_5.model.norm.clear_vram();
            self.qwen3_5.head.load_to_vram(&self.registry, &self.device)?;
            let mut next_logits = self.qwen3_5.head.forward(&out, &self.registry)?;
            if self.qwen3_5.tie { next_logits = next_logits.affine(1.0 / self.qwen3_5.hidden_size.sqrt(), 0.0)?; }
            logits = next_logits;
            self.qwen3_5.head.clear_vram();
        }

        println!("\n[MODEL-QWEN35] Generation complete.");
        Ok(gen_text)
    }
}

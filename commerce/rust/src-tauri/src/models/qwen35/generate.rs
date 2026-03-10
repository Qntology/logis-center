use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{
    chat_template::ChatTemplate,
    models::qwen35::{config::Qwen3_5Config, quantized_model::QuantizedQwen3_5Model},
    tokenizer::TokenizerModel,
    utils::{
        find_type_files, get_device, get_dtype, get_logit_processor,
    },
    openai_types::ChatCompletionParameters,
};

pub struct Qwen3_5GenerateModel {
    pub chat_template: ChatTemplate,
    pub tokenizer: TokenizerModel,
    pub qwen3_5: QuantizedQwen3_5Model,
    pub registry: crate::models::qwen35::quantized_model::QuantizedRegistry,
    pub device: Device,
    pub eos_token_id: u32,
    pub model_name: String,
}

impl Qwen3_5GenerateModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>, _force_text_only: bool) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = std::path::Path::new(path).join("config.json");
        let cfg: Qwen3_5Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.text_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        
        let mut model_list = find_type_files(path, "st")?;
        model_list.retain(|f| !f.contains("vision.st"));

        // [DYNAMIC REGISTRY] Use Mmap-based Registry for extreme VRAM/RAM efficiency
        let registry = crate::models::qwen35::quantized_model::QuantizedRegistry::new(&model_list)?;

        let vb_dummy = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let qwen3_5 = QuantizedQwen3_5Model::new(vb_dummy, cfg.clone())?;

        println!("[MODEL-QWEN35] Quantized dynamic registry loaded. Ready for Layer Cycling.");

        Ok(Self {
            chat_template,
            tokenizer,
            qwen3_5,
            registry,
            device,
            eos_token_id: cfg.text_config.eos_token_id,
            model_name: "qwen3.5".to_string(),
        })
    }

    pub async fn generate(
        &mut self, 
        mes: ChatCompletionParameters, 
        cancel_flag: Option<Arc<AtomicBool>>
    ) -> Result<String> {
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(
            mes.temperature.map(|t| t as f32), 
            mes.top_p.map(|p| p as f32), 
            None, 
            seed
        );
        
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let full_input_ids = self.tokenizer.text_encode(mes_render, &self.device)?
            .to_dtype(DType::U32)?;
            
        let full_seq_len = full_input_ids.dim(1)?;
        
        let mut seqlen_offset = 0;
        let chunk_size = 512;
        let mut last_logits = None;

        self.qwen3_5.clear_cache();
        // [HYBRID] Initialize Disk Manager for Relay Prefill
        let mut disk_manager = crate::models::qwen35::quantized_model::DiskStateManager::new(self.qwen3_5.model.layers.len())?;

        println!("[MODEL-QWEN35] Starting HYBRID Prefill (Relay mode with SSD offloading)...");

        while seqlen_offset < full_seq_len {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { break; }
            }

            let remaining = full_seq_len - seqlen_offset;
            let current_chunk = remaining.min(chunk_size);
            let input_ids = full_input_ids.narrow(1, seqlen_offset, current_chunk)?
                .to_dtype(DType::U32)?;

            let logits = self.qwen3_5.forward(&input_ids, seqlen_offset, &self.registry, None)?;
            
            // [CRITICAL] Offload each layer's context to SSD and clear VRAM immediately
            for (i, layer) in self.qwen3_5.model.layers.iter_mut().enumerate() {
                if let Some(ref mut sa) = layer.self_attn {
                    if let Some((k, v)) = &sa.kv_cache {
                        disk_manager.save_layer_context(i, &crate::models::qwen35::quantized_model::LayerContext::Attention { k: k.clone(), v: v.clone() })?;
                        sa.kv_cache = None; // Free VRAM
                    }
                }
                if let Some(ref mut la) = layer.linear_attn {
                    if let Some(st) = &la.delta_state {
                        disk_manager.save_layer_context(i, &crate::models::qwen35::quantized_model::LayerContext::DeltaNet { state: st.clone() })?;
                        la.delta_state = None; // Free VRAM
                    }
                }
            }

            seqlen_offset += current_chunk;
            last_logits = Some(logits);

            if full_seq_len > chunk_size {
                println!("[PREFILL-RELAY] Processed {}/{} tokens...", seqlen_offset, full_seq_len);
            }
        }

        // [TRANSITION] Load all weights to VRAM for fast decoding
        println!("[MODEL-QWEN35] Prefill complete. Transitioning to FULL VRAM mode for Decoding...");
        self.qwen3_5.load_all_to_vram(&self.registry, &self.device)?;

        // [TRANSITION] Load all saved context from SSD back to VRAM
        let config_path = std::path::Path::new("models/Qwen3.5-0.8B-Split").join("config.json");
        let cfg: Qwen3_5Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let text_cfg = &cfg.text_config;

        for (i, layer) in self.qwen3_5.model.layers.iter_mut().enumerate() {
            let lt = if layer.self_attn.is_some() { "attention" } else { "linear_attention" };
            
            if lt == "attention" {
                let s_k = candle_core::Shape::from((1, text_cfg.num_key_value_heads, seqlen_offset, text_cfg.head_dim));
                let s_v = candle_core::Shape::from((1, text_cfg.num_key_value_heads, seqlen_offset, text_cfg.head_dim));
                match disk_manager.load_layer_context(i, lt, &s_k, &s_v, &self.device)? {
                    crate::models::qwen35::quantized_model::LayerContext::Attention { k, v } => {
                        layer.self_attn.as_mut().unwrap().kv_cache = Some((k, v));
                    },
                    _ => unreachable!(),
                }
            } else {
                // DeltaNet state shape
                let s_state = candle_core::Shape::from((1, 1, text_cfg.hidden_size)); 
                match disk_manager.load_layer_context(i, lt, &s_state, &s_state, &self.device)? {
                    crate::models::qwen35::quantized_model::LayerContext::DeltaNet { state } => {
                        layer.linear_attn.as_mut().unwrap().delta_state = Some(state);
                    },
                    _ => unreachable!(),
                }
            }
        }

        let mut logits = last_logits.ok_or(anyhow::anyhow!("No logits generated"))?;
        let mut gen_text = String::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);

        for _i in 0..sample_len {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { break; }
            }

            let current_logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&current_logits)?;
            
            if next_token == self.eos_token_id {
                break;
            }

            let piece = self.tokenizer.token_decode(vec![next_token])?;
            gen_text.push_str(&piece);

            let input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?
                .to_dtype(DType::U32)?;
            logits = self.qwen3_5.forward(&input_ids, seqlen_offset, &self.registry, None)?;
            seqlen_offset += 1;
        }

        self.qwen3_5.clear_cache();
        Ok(gen_text)
    }
}

pub fn init_bake_worker() {
    println!("[MODEL-QWEN35] Bake worker initialized for text-only prefill.");
}


use anyhow::{Result, anyhow};
use candle_core::{quantized::gguf_file, DType, Device, Tensor, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::utils::apply_repeat_penalty;

use crate::{
    chat_template::ChatTemplate,
    models::{
        qwen3vl::{
            config::{Qwen3VLConfig, Qwen3VLGenerationConfig},
            model::Qwen3VLModel,
            quantized_model::{QuantizedQwen3VLModel, QuantizedQwen3TextModel},
            processor::Qwen3VLProcessor,
        },
    },
    tokenizer::TokenizerModel,
    utils::{
        find_type_files,
        get_device,
        get_dtype,
        get_logit_processor,
    },
    openai_types::ChatCompletionParameters,
};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::fs;
use std::path::Path;

#[derive(Clone)]
pub enum ModelVariant {
    Standard(crate::models::qwen3vl::model::Qwen3VLModel),
    QuantizedVL(crate::models::qwen3vl::quantized_model::QuantizedQwen3VLModel),
    QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel),
}

impl ModelVariant {
    pub fn forward(&mut self, input_ids: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, _video_pixel_values: Option<&Tensor>, _video_grid_thw: Option<&Tensor>, cache_position: Option<&Tensor>, seqlen_offset: usize) -> Result<Tensor> {
        match self {
            Self::Standard(m) => m.forward(input_ids, pixel_values, image_grid_thw, _video_pixel_values, _video_grid_thw, cache_position, seqlen_offset),
            Self::QuantizedVL(m) => m.forward(input_ids, pixel_values, image_grid_thw, _video_pixel_values, _video_grid_thw, cache_position, seqlen_offset),      
            Self::QuantizedText(m) => m.forward(input_ids, cache_position, seqlen_offset),
        }
    }
    pub fn rebalance_layers(&mut self, _device_id: usize) -> Result<()> { Ok(()) }
    pub fn drop_kv_storage(&mut self) -> Result<()> {
        match self {
            Self::Standard(_) => Ok(()),
            Self::QuantizedVL(m) => { m.language_model.clear_kv_cache(); Ok(()) },
            Self::QuantizedText(m) => { m.language_model.clear_kv_cache(); Ok(()) },
        }
    }
    pub fn inject_kv_bitkv(&mut self, _k_anchors: &[Tensor], _k_packed: &[Tensor], _k_scales: &[Tensor], _v_anchors: &[Tensor], _v_packed: &[Tensor], _v_scales: &[Tensor], _original_shape: &[usize]) -> Result<()> { Ok(()) }
    pub fn is_cpu(&self) -> bool {
        match self {
            Self::Standard(m) => m.device().is_cpu(),
            Self::QuantizedVL(m) => m.language_model.is_forced_cpu,
            Self::QuantizedText(m) => m.language_model.is_forced_cpu,
        }
    }
}

pub struct Qwen3VLGenerateModel {
    pub chat_template: ChatTemplate, pub tokenizer: TokenizerModel, pub pre_processor: Qwen3VLProcessor,
    pub qwen3_vl: ModelVariant, pub text_device: Device, pub vision_device: Device,
    pub eos_token_id1: u32, pub eos_token_id2: u32, pub generation_config: Qwen3VLGenerationConfig,
    pub model_name: String, pub hard_token_limit: Option<usize>,
}

impl Qwen3VLGenerateModel {
    pub fn init(path: &str, text_device: Option<&Device>, text_device_id: usize, vision_device: Option<&Device>, vision_device_id: usize, dtype: Option<DType>, hard_token_limit: Option<usize>, force_text_only: bool, baking_only: bool, high_fidelity: bool) -> Result<Self> {
        Self::init_with_config(path, None, None, text_device, text_device_id, vision_device, vision_device_id, dtype, hard_token_limit, force_text_only, baking_only, high_fidelity)
    }
    pub fn init_with_tokenizer(path: &str, tokenizer_path: Option<&str>, text_device: Option<&Device>, text_device_id: usize, vision_device: Option<&Device>, vision_device_id: usize, dtype: Option<DType>, hard_token_limit: Option<usize>, force_text_only: bool, baking_only: bool, high_fidelity: bool) -> Result<Self> {
        Self::init_with_config(path, tokenizer_path, None, text_device, text_device_id, vision_device, vision_device_id, dtype, hard_token_limit, force_text_only, baking_only, high_fidelity) 
    }
    pub fn init_with_config(path: &str, tokenizer_path: Option<&str>, config_path: Option<&str>, text_device: Option<&Device>, text_device_id: usize, vision_device: Option<&Device>, vision_device_id: usize, dtype: Option<DType>, hard_token_limit: Option<usize>, force_text_only: bool, baking_only: bool, _high_fidelity: bool) -> Result<Self> {
        let path = if let Some(s) = path.strip_prefix(r"\\\\") { s } else { path };
        let tok_path = tokenizer_path.unwrap_or(path);
        let cfg_path = config_path.unwrap_or(path);
        let chat_template = ChatTemplate::init(tok_path)?;
        let tokenizer = TokenizerModel::init(tok_path)?;
        let raw_config: serde_json::Value = serde_json::from_slice(&std::fs::read(std::path::Path::new(cfg_path).join("config.json"))?)?;
        let mut cfg: Qwen3VLConfig = if raw_config.get("text_config").is_some() { serde_json::from_value(raw_config)? } else {
            let mut tc: crate::models::qwen3vl::config::Qwen3VLTextConfig = serde_json::from_value(raw_config.clone())?;
            if let Some(h) = raw_config.get("head_dim").and_then(|v| v.as_u64()) { tc.head_dim = h as usize; }
            Qwen3VLConfig { text_config: Some(tc), model_type: "qwen2".to_string(), ..Default::default() }
        };
        if path.contains("0.6B") { if let Some(ref mut tc) = cfg.text_config { if tc.hidden_size == 2048 { tc.hidden_size = 1024; tc.intermediate_size = 2816; } } }
        let text_dev = get_device(text_device); let vision_dev = get_device(vision_device);
        let dtype = get_dtype(dtype, cfg.text_config.as_ref().and_then(|tc| tc.dtype.as_deref()).unwrap_or("float16"));
        let st_files = find_type_files(path, "safetensors")?;
        let gguf_files = find_type_files(path, "gguf")?;
        let model_path = if baking_only { st_files.iter().find(|f| f.contains("ANCHOR_IQ0.safetensors")).cloned() } else { st_files.iter().find(|f| f.contains("model-BITSERIAL_ALL.safetensors")).cloned() };
        if let Some(st_path) = model_path {
            let is_anchor = st_path.contains("ANCHOR_IQ0");
            let mut m_cfg = cfg.clone(); if baking_only || is_anchor { if let Some(ref mut tc) = m_cfg.text_config { tc.num_hidden_layers = 1; } }
            let mut merged_data = crate::models::qwen3vl::quantized_model::load_tensors_from_true_iq0(Path::new(&st_path), &text_dev, dtype, baking_only)?;
            if !force_text_only {
                let mmproj_st = st_files.iter().find(|f| f.contains("mmproj-BITSERIAL_ALL.safetensors"));
                if let Some(mm_path) = mmproj_st {
                    let vision_data = crate::models::qwen3vl::quantized_model::load_tensors_from_true_iq0(Path::new(mm_path), &vision_dev, dtype, baking_only)?;
                    for (k, v) in vision_data { merged_data.insert(k, v); }
                }
            }
            let vb = VarBuilder::from_tensors(merged_data, dtype, &text_dev);
            let mut model = Qwen3VLModel::new_ext(m_cfg, vb, force_text_only)?;
            if baking_only || is_anchor { model.set_baking(true); }
            return Ok(Self { chat_template, tokenizer, pre_processor: Qwen3VLProcessor::new(tok_path, &vision_dev, dtype)?, qwen3_vl: ModelVariant::Standard(model), text_device: text_dev, vision_device: vision_dev, eos_token_id1: 151643, eos_token_id2: 151645, generation_config: Qwen3VLGenerationConfig::default(), model_name: if is_anchor { "qwen3-anchor" } else { "qwen3-bitserial-full" }.to_string(), hard_token_limit });
        }
        let kv_reserve = 512 * 1024 * 1024;
        let qwen3_vl = if !gguf_files.is_empty() {
            let model_p = if !baking_only { gguf_files.iter().find(|f| f.contains("Qwen3VL-2B") && f.contains("Q4_K_M")).cloned() } else { gguf_files.iter().find(|f| f.contains("Qwen3VL-2B") && f.contains("IQ1_S")).cloned() };
            let main = model_p.or_else(|| if !gguf_files.is_empty() { Some(gguf_files[0].clone()) } else { None }).ok_or(anyhow!("No GGUF found"))?;
            let file = std::fs::File::open(&main)?;
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let mut cursor = std::io::Cursor::new(&mmap[..]);
            let content = gguf_file::Content::read(&mut cursor)?;
            let mmproj_p = if !force_text_only { gguf_files.iter().find(|f| f.contains("mmproj") && (if baking_only { f.contains("IQ1_S") } else { f.contains("Q8_0") })).cloned() } else { None };
            if let Some(mmproj) = mmproj_p {
                let mm_file = std::fs::File::open(&mmproj)?;
                let mm_mmap = unsafe { memmap2::MmapOptions::new().map(&mm_file)? };
                let mut mm_cursor = std::io::Cursor::new(&mm_mmap[..]);
                let mm_content = gguf_file::Content::read(&mut mm_cursor)?;
                ModelVariant::QuantizedVL(QuantizedQwen3VLModel::new_with_mmap(&cfg, &content, Some(Arc::new(mmap)), &mm_content, Some(Arc::new(mm_mmap)), &text_dev, text_device_id, &vision_dev, vision_device_id, dtype, kv_reserve, baking_only, force_text_only)?)
            } else {
                ModelVariant::QuantizedText(QuantizedQwen3TextModel::new_with_mmap(&cfg, &content, Some(Arc::new(mmap)), &text_dev, text_device_id, dtype, kv_reserve, baking_only, path.contains("0.6B"))?)
            }
        } else { return Err(anyhow!("No model files found")); };
        Ok(Self { chat_template, tokenizer, pre_processor: Qwen3VLProcessor::new(tok_path, &vision_dev, dtype)?, qwen3_vl, text_device: text_dev, vision_device: vision_dev, eos_token_id1: 151643, eos_token_id2: 151645, generation_config: Qwen3VLGenerationConfig::default(), model_name: "qwen3-gguf".to_string(), hard_token_limit })
    }

    pub fn prefill_text_only(&mut self, text: &str, cancel_token: Option<Arc<AtomicBool>>, _relay_target: Option<&mut Qwen3VLGenerateModel>, auto_save_path: Option<&std::path::Path>) -> Result<()> {
        let token_ids = self.tokenizer.text_encode_vec(text.to_string(), false)?;
        let total_tokens = token_ids.len(); let chunk_size = 512; let mut current_pos = 0;
        while current_pos < total_tokens {
            if let Some(token) = &cancel_token { if token.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            let end = (current_pos + chunk_size).min(total_tokens);
            let chunk_ids = Tensor::from_vec(token_ids[current_pos..end].to_vec(), (1, end - current_pos), &self.text_device)?;
            let chunk_pos = Tensor::arange(current_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?;
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos)?;
            if let Some(path) = auto_save_path { if !path.exists() { let _ = fs::create_dir_all(path); } self.save_kv_to_disk(path)?; }
            current_pos = end;
        }
        if auto_save_path.is_some() { let _ = self.qwen3_vl.drop_kv_storage(); }
        Ok(())
    }

    pub fn prefill_only(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let full_input_ids_vec = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_tokens = full_input_ids_vec.len(); let mut current_pos = self.get_kv_len(); let chunk_size = 512;
        while current_pos < total_tokens {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            let end = (current_pos + chunk_size).min(total_tokens);
            if end <= current_pos { break; }
            let chunk_ids = Tensor::from_vec(full_input_ids_vec[current_pos..end].to_vec(), (1, end - current_pos), &self.text_device)?;
            let chunk_pos = Tensor::arange(current_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?;
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos)?;
            if let Some(ref sid) = session_id { let path = crate::utils::paths::get_kv_dir(None).join(sid); if !path.exists() { let _ = fs::create_dir_all(&path); } self.save_kv_to_disk(&path)?; }
            current_pos = end;
        }
        Ok(current_pos)
    }

    pub fn prefill_chunk(&mut self, text: String, cancel_flag: Option<Arc<AtomicBool>>, _relay_target: Option<&mut Qwen3VLGenerateModel>, session_id: Option<String>) -> Result<usize> {
        let chunk = self.tokenizer.text_encode_vec(text, false)?;
        let chunk_size = chunk.len(); let current_pos = self.get_kv_len();
        let chunk_ids = Tensor::from_vec(chunk.to_vec(), (1, chunk_size), &self.text_device)?;
        let chunk_pos = Tensor::arange(current_pos as u32, (current_pos + chunk_size) as u32, &self.text_device)?.unsqueeze(0)?;
        if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
        self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos)?;
        if let Some(sid) = session_id { let path = crate::utils::paths::get_kv_dir(None).join(sid); if !path.exists() { let _ = fs::create_dir_all(&path); } self.save_kv_to_disk(&path)?; }
        Ok(chunk_size)
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>) -> Result<String> {
        let mut logit_processor = get_logit_processor(mes.temperature.map(|t| t as f32), mes.top_p.map(|p| p as f32), Some(40), mes.seed.unwrap_or(34562) as u64);
        let mut seqlen_offset = self.get_kv_len();
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut input = self.pre_processor.process_info(&mes, &mes_render)?;
        let full_input_ids_vec = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_tokens = full_input_ids_vec.len();
        
        if seqlen_offset == 0 { 
            if let Some(sid) = &session_id { 
                let path = crate::utils::paths::get_kv_dir(None).join(sid); 
                if path.exists() { 
                    self.load_kv_from_disk(&path)?; 
                    seqlen_offset = self.get_kv_len();
                    if seqlen_offset > 0 {
                        println!("[RESUME] Loaded KV cache via Bridge. Offset: {}", seqlen_offset);
                    }
                } 
            } 
        }

        // [FIX] local_pos should track where we are in the CURRENT prompt.
        // If we resumed from a snapshot, we should skip tokens already in the cache.
        let mut local_pos = seqlen_offset.min(total_tokens);
        let chunk_size = if self.text_device.is_cpu() { 1024 } else { 256 };

        while local_pos < total_tokens {
            let remaining = total_tokens - local_pos; 
            if remaining <= 1 && generated_text.is_empty() { break; } // Leave last token for generation loop
            
            let mut c_size = remaining.min(chunk_size); 
            if local_pos + c_size >= total_tokens && generated_text.is_empty() { 
                c_size = (total_tokens - local_pos).saturating_sub(1); 
            }
            if c_size == 0 { break; }

            let chunk_ids = Tensor::from_vec(full_input_ids_vec[local_pos..local_pos + c_size].to_vec(), (1, c_size), &self.text_device)?;
            let chunk_pos = Tensor::arange(seqlen_offset as u32, (seqlen_offset + c_size) as u32, &self.text_device)?.unsqueeze(0)?;
            
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), seqlen_offset)?;
            
            local_pos += c_size; 
            seqlen_offset += c_size;
        }
        let mut all_ids = full_input_ids_vec.clone(); 
        let mut pixel_values = input.pixel_values.take(); 
        let image_grid_thw = input.image_grid_thw.take();

        for _i in 0..mes.max_tokens.unwrap_or(2048) {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            
            // [FIX] Correctly determine the next token to feed. 
            // If we just finished prefilling, feed the very last token of the prompt.
            let input_ids = if generated_text.is_empty() {
                let start = local_pos.min(total_tokens.saturating_sub(1));
                Tensor::new(&full_input_ids_vec[start..total_tokens], &self.text_device)?.unsqueeze(0)?
            } else {
                Tensor::new(vec![*all_ids.last().unwrap()], &self.text_device)?.unsqueeze(0)?
            };

            let seq_len = input_ids.dim(1)?;
            let chunk_pos = Tensor::arange(seqlen_offset as u32, (seqlen_offset + seq_len) as u32, &self.text_device)?.unsqueeze(0)?;
            
            let logits = self.qwen3_vl.forward(&input_ids, pixel_values.as_ref(), image_grid_thw.as_ref(), None, None, Some(&chunk_pos), seqlen_offset)?;
            
            let logits = logits.squeeze(0)?; 
            let mut logits = logits.i(logits.dim(0)? - 1)?.to_dtype(DType::F32)?;
            
            logits = apply_repeat_penalty(&logits, 1.1, if all_ids.len() > 512 { &all_ids[all_ids.len()-512..] } else { &all_ids[..] })?;
            let next_id = logit_processor.sample(&logits)?;
            
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            
            all_ids.push(next_id); 
            let decoded = self.tokenizer.token_decode(vec![next_id])?;
            generated_text.push_str(&decoded);
            
            // Optional progress logging
            // if _i % 10 == 0 { println!("[GEN] Progress: {}", generated_text); }

            if _i > 0 && _i % 512 == 0 && !self.qwen3_vl.is_cpu() { let _ = self.qwen3_vl.rebalance_layers(0); }
            
            seqlen_offset += seq_len; 
            local_pos = total_tokens; // After first token, we only feed the last generated ID
            pixel_values = None;
        }
        Ok(generated_text)
    }

    pub fn get_kv_len(&self) -> usize { match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.get_kv_len(), ModelVariant::QuantizedText(m) => m.language_model.get_kv_len(), _ => 0 } }
    pub fn get_current_kv(&self) -> (Vec<Tensor>, Vec<Tensor>) {
        let mut ks: Vec<Tensor> = vec![]; let mut vs: Vec<Tensor> = vec![];
        match &self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => { for l in &m.language_model.layers { if let Some((k, v)) = &l.self_attn.kv_cache { ks.push(k.clone()); vs.push(v.clone()); } } },
            ModelVariant::QuantizedText(m) => { for l in &m.language_model.layers { if let Some((k, v)) = &l.self_attn.kv_cache { ks.push(k.clone()); vs.push(v.clone()); } } },
            _ => {} 
        }
        (ks, vs)
    }
    pub fn inject_kv_bitkv(&mut self, k_a: &[Tensor], k_p: &[Tensor], k_s: &[Tensor], v_a: &[Tensor], v_p: &[Tensor], v_s: &[Tensor], os: &[usize]) -> Result<()> { self.qwen3_vl.inject_kv_bitkv(k_a, k_p, k_s, v_a, v_p, v_s, os) }
    pub fn save_kv_to_disk(&mut self, p: &Path) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.save_kv_cache(p, false, 1024),
            ModelVariant::QuantizedText(m) => m.save_kv_cache(p, false, 1024),
            ModelVariant::Standard(m) => m.save_kv_cache(p),
        }
    }
    pub fn load_kv_from_disk(&mut self, p: &Path) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.load_kv_cache(p, &self.text_device, 0, 128),
            ModelVariant::QuantizedText(m) => m.load_kv_cache(p, &self.text_device, 0, 128),
            ModelVariant::Standard(m) => m.load_kv_cache(p, &self.text_device),
        }
    }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.to_device(d)?, ModelVariant::QuantizedText(m) => m.to_device(d)?, _ => {} }; self.text_device = d.clone(); self.vision_device = d.clone(); Ok(()) }
    pub fn clear_kv_cache(&mut self) { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.clear_kv_cache(), ModelVariant::QuantizedText(m) => m.clear_kv_cache(), _ => {} } }
}

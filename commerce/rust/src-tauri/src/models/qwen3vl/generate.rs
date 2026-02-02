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
        find_type_files, get_device, get_dtype, get_logit_processor,
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
            Self::Standard(m) => m.forward(input_ids, pixel_values, image_grid_thw, _video_pixel_values, image_grid_thw, cache_position, seqlen_offset),
            Self::QuantizedVL(m) => m.forward(input_ids, pixel_values, image_grid_thw, _video_pixel_values, _video_grid_thw, cache_position, seqlen_offset),      
            Self::QuantizedText(m) => m.forward(input_ids, cache_position, seqlen_offset),
        }
    }
    pub fn set_baking(&mut self, b: bool) { if let Self::Standard(m) = self { m.set_baking(b); } }
    pub fn clear_kv_cache(&mut self) { match self {
            Self::Standard(m) => m.clear_kv_cache(),
            Self::QuantizedVL(m) => m.clear_kv_cache(),
            Self::QuantizedText(m) => m.clear_kv_cache()
        } 
    }
    pub fn is_cpu(&self) -> bool { match self {
            Self::Standard(m) => m.device().is_cpu(),
            Self::QuantizedVL(m) => m.language_model.is_forced_cpu,
            Self::QuantizedText(m) => m.language_model.is_forced_cpu
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
    pub fn init_with_config(path: &str, tokenizer_path: Option<&str>, config_path: Option<&str>, text_device: Option<&Device>, text_device_id: usize, vision_device: Option<&Device>, _vision_device_id: usize, dtype: Option<DType>, hard_token_limit: Option<usize>, force_text_only: bool, baking_only: bool, _high_fidelity: bool) -> Result<Self> {
        let path = if let Some(s) = path.strip_prefix(r"\\?\") { s } else { path };
        let chat_template = ChatTemplate::init(tokenizer_path.unwrap_or(path))?;
        let tokenizer = TokenizerModel::init(tokenizer_path.unwrap_or(path))?;
        let config_file = std::path::Path::new(config_path.unwrap_or(path)).join("config.json");
        let raw_config: serde_json::Value = serde_json::from_slice(&std::fs::read(config_file)?)?;
        let cfg: Qwen3VLConfig = serde_json::from_value(raw_config)?;
        let text_dev = get_device(text_device); let vision_dev = get_device(vision_device); let dt = get_dtype(dtype, "float16");
        
        let st_files = find_type_files(path, "safetensors")?;
        if let Some(st_path) = st_files.iter().find(|f| f.contains("BITSERIAL")) {
            let merged_data = crate::models::qwen3vl::quantized_model::load_tensors_from_true_iq0(Path::new(st_path), &text_dev, dt, baking_only)?;
            let vb = VarBuilder::from_tensors(merged_data.clone(), dt, &text_dev);
            // CRITICAL: Pass Some(merged_data) to Standard model
            let model = Qwen3VLModel::new_ext(cfg, vb, Some(merged_data), force_text_only, baking_only)?;
            return Ok(Self { chat_template, tokenizer, pre_processor: Qwen3VLProcessor::new(path, &vision_dev, dt)?, qwen3_vl: ModelVariant::Standard(model), text_device: text_dev, vision_device: vision_dev, eos_token_id1: 151643, eos_token_id2: 151645, generation_config: Qwen3VLGenerationConfig::default(), model_name: "qwen3-bitserial".to_string(), hard_token_limit });
        }
        let gguf_files = find_type_files(path, "gguf")?;
        let qwen3_vl = if !gguf_files.is_empty() {
            let mmap = Arc::new(unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&gguf_files[0])?)? });
            ModelVariant::QuantizedText(QuantizedQwen3TextModel::new_with_mmap(&cfg, &gguf_file::Content::read(&mut std::io::Cursor::new(&mmap[..]))?, Some(mmap), &text_dev, text_device_id, dt, 0, baking_only, true)?)
        } else { return Err(anyhow!("No files")); };
        Ok(Self { chat_template, tokenizer, pre_processor: Qwen3VLProcessor::new(path, &vision_dev, dt)?, qwen3_vl, text_device: text_dev, vision_device: vision_dev, eos_token_id1: 151643, eos_token_id2: 151645, generation_config: Qwen3VLGenerationConfig::default(), model_name: "qwen3-gguf".to_string(), hard_token_limit })
    }

    pub fn prefill_text_only(&mut self, text: &str, cancel_token: Option<Arc<AtomicBool>>, _relay_target: Option<&mut Qwen3VLGenerateModel>, auto_save_path: Option<&std::path::Path>) -> Result<()> {
        let token_ids = self.tokenizer.text_encode_vec(text.to_string(), false)?;
        let total_tokens = token_ids.len(); let mut current_pos = 0;
        while current_pos < total_tokens {
            if let Some(token) = &cancel_token { if token.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            let end = (current_pos + 256).min(total_tokens);
            let chunk_ids = Tensor::from_vec(token_ids[current_pos..end].to_vec(), (1, end - current_pos), &self.text_device)?;
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, None, current_pos)?;
            if let Some(path) = auto_save_path { self.save_kv_to_disk(path)?; }
            current_pos = end;
        } Ok(())
    }

    pub fn prefill_chunk(&mut self, text: String, cancel_flag: Option<Arc<AtomicBool>>, _relay_target: Option<&mut Qwen3VLGenerateModel>, session_id: Option<String>) -> Result<usize> {
        let tokens = self.tokenizer.text_encode_vec(text, false)?; let sl = tokens.len(); let offset = self.get_kv_len();
        if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
        let ids = Tensor::from_vec(tokens, (1, sl), &self.text_device)?;
        self.qwen3_vl.forward(&ids, None, None, None, None, None, offset)?;
        if let Some(sid) = session_id { let path = crate::utils::paths::get_kv_dir(None).join(sid); if !path.exists() { fs::create_dir_all(&path)?; } self.save_kv_to_disk(&path)?; }
        Ok(sl)
    }

    pub fn prefill_only(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let input = self.pre_processor.process_info(&mes, &self.chat_template.apply_chat_template(&mes)?)?;
        let full_ids = self.tokenizer.text_encode_vec(input.replace_text, false)?; let total = full_ids.len(); let mut pos = self.get_kv_len();
        while pos < total {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            let end = (pos + 256).min(total); if end <= pos { break; }
            let ids = Tensor::from_vec(full_ids[pos..end].to_vec(), (1, end - pos), &self.text_device)?;
            self.qwen3_vl.forward(&ids, None, None, None, None, None, pos)?;
            if let Some(ref sid) = session_id { let path = crate::utils::paths::get_kv_dir(None).join(sid); if !path.exists() { fs::create_dir_all(&path)?; } self.save_kv_to_disk(&path)?; }
            pos = end;
        } Ok(pos)
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>) -> Result<String> {
        let mut logit_processor = get_logit_processor(mes.temperature.map(|t| t as f32), mes.top_p.map(|p| p as f32), Some(40), mes.seed.unwrap_or(34562) as u64);
        let mut offset = self.get_kv_len(); let mut generated = String::new();
        let input = self.pre_processor.process_info(&mes, &self.chat_template.apply_chat_template(&mes)?)?;
        let full_ids = self.tokenizer.text_encode_vec(input.replace_text, false)?; let total = full_ids.len();
        if offset == 0 { if let Some(sid) = &session_id { let p = crate::utils::paths::get_kv_dir(None).join(sid); if p.exists() { self.load_kv_from_disk(&p)?; offset = self.get_kv_len(); } } }
        let mut pos = offset.min(total);
        while pos < total {
            let rem = total - pos; if rem <= 1 && generated.is_empty() { break; }
            let c_size = if pos + rem >= total && generated.is_empty() { rem.saturating_sub(1) } else { rem.min(256) }; if c_size == 0 { break; }
            let ids = Tensor::from_vec(full_ids[pos..pos+c_size].to_vec(), (1, c_size), &self.text_device)?;
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            self.qwen3_vl.forward(&ids, None, None, None, None, None, pos)?;
            pos += c_size; offset += c_size;
        }
        let mut all_ids = full_ids.clone();
        for _i in 0..mes.max_tokens.unwrap_or(2048) {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            let ids = if generated.is_empty() { Tensor::new(&full_ids[pos.min(total.saturating_sub(1))..total], &self.text_device)?.unsqueeze(0)? } else { Tensor::new(vec![*all_ids.last().unwrap()], &self.text_device)?.unsqueeze(0)? };
            let logits = self.qwen3_vl.forward(&ids, None, None, None, None, None, offset)?.squeeze(0)?;
            let mut logits = logits.i(logits.dim(0)? - 1)?.to_dtype(DType::F32)?;
            logits = apply_repeat_penalty(&logits, 1.1, if all_ids.len() > 512 { &all_ids[all_ids.len()-512..] } else { &all_ids[..] })?;
            let next_id = logit_processor.sample(&logits)?; if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            all_ids.push(next_id); generated.push_str(&self.tokenizer.token_decode(vec![next_id])?);
            offset += ids.dim(1)?; pos = total;
        } Ok(generated)
    }

    pub fn get_kv_len(&self) -> usize { match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.get_kv_len(), ModelVariant::QuantizedText(m) => m.language_model.get_kv_len(), ModelVariant::Standard(m) => m.get_kv_len() } }
    pub fn save_kv_to_disk(&mut self, p: &Path) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.save_kv_cache(p, false, 1024), ModelVariant::QuantizedText(m) => m.save_kv_cache(p, false, 1024), ModelVariant::Standard(m) => m.save_kv_cache(p) } }
    pub fn load_kv_from_disk(&mut self, p: &Path) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.load_kv_cache(p, &self.text_device, 0, 128), ModelVariant::QuantizedText(m) => m.load_kv_cache(p, &self.text_device, 0, 128), ModelVariant::Standard(m) => m.load_kv_cache(p, &self.text_device) } }
    pub fn clear_kv_cache(&mut self) { self.qwen3_vl.clear_kv_cache(); }
}

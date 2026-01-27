use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, IndexOp};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::fs;
use std::path::Path;

use crate::models::qwen3vl::config::Qwen3VLConfig;
use crate::models::qwen3vl::quantized_model::{QuantizedQwen3VLModel, QuantizedQwen3TextModel};
use crate::openai_types::ChatCompletionParameters;

#[derive(Clone)]
pub enum ModelVariant {
    Standard(Box<crate::models::qwen3vl::model::Qwen3VLModel>),
    QuantizedVL(QuantizedQwen3VLModel),
    QuantizedText(QuantizedQwen3TextModel),
}

impl ModelVariant {
    pub fn forward(&mut self, input_ids: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, video_pixel_values: Option<&Tensor>, video_grid_thw: Option<&Tensor>, cache_position: Option<&Tensor>, seqlen_offset: usize) -> Result<Tensor> {
        match self {
            Self::Standard(m) => m.forward(input_ids, pixel_values, image_grid_thw, video_pixel_values, video_grid_thw, cache_position, seqlen_offset),
            Self::QuantizedVL(m) => m.forward(input_ids, pixel_values, image_grid_thw, video_pixel_values, video_grid_thw, cache_position, seqlen_offset),
            Self::QuantizedText(m) => m.forward(input_ids, cache_position, seqlen_offset),
        }
    }
}

pub struct Qwen3VLGenerateModel {
    pub chat_template: crate::chat_template::ChatTemplate,
    pub tokenizer: crate::tokenizer::Qwen3Tokenizer,
    pub pre_processor: crate::parsing::Qwen3VLPreProcessor,
    pub qwen3_vl: ModelVariant, // Restore original name
    pub text_device: Device,
    pub vision_device: Device,
    pub eos_token_id1: u32,
    pub eos_token_id2: u32,
    pub generation_config: crate::models::qwen3vl::config::Qwen3VLGenerationConfig,
    pub model_name: String,
    pub hard_token_limit: usize,
}

impl Qwen3VLGenerateModel {
    pub fn init_with_config(
        path: &str,
        mmproj_path: Option<&str>,
        config_path: Option<&str>,
        text_device: Option<&Device>,
        text_device_id: usize,
        vision_device: Option<&Device>,
        vision_device_id: usize,
        dtype: Option<DType>,
        max_tokens: Option<usize>,
    ) -> Result<Self> {
        let text_dev = text_device.cloned().unwrap_or(Device::Cpu);
        let vision_dev = vision_device.cloned().unwrap_or(Device::Cpu);
        let dtype = dtype.unwrap_or(DType::F32);
        
        let model_path = Path::new(path);
        let config_file = config_path.map(Path::new).unwrap_or(&model_path.join("config.json")).to_path_buf();
        let config_str = fs::read_to_string(&config_file)?;
        let cfg: Qwen3VLConfig = serde_json::from_str(&config_str)?;
        
        let tokenizer_file = model_path.join("tokenizer.json");
        let tokenizer = crate::tokenizer::Qwen3Tokenizer::new(&tokenizer_file.to_string_lossy())?;
        
        let chat_template_file = model_path.join("chat_template.json");
        let chat_template = crate::chat_template::ChatTemplate::new(&chat_template_file.to_string_lossy())?;
        
        let pre_processor = crate::parsing::Qwen3VLPreProcessor::new(&cfg)?;
        
        let eos_token_id1 = cfg.text_config.as_ref().and_then(|c| c.eos_token_id).unwrap_or(151643) as u32;
        let eos_token_id2 = 151645;
        
        let generation_config = crate::models::qwen3vl::config::Qwen3VLGenerationConfig {
            max_new_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            eos_token_id: vec![eos_token_id1, eos_token_id2],
        };

        let model_name = path.to_string();
        let hard_token_limit = max_tokens.unwrap_or(32768);
        let kv_reserve = 100_000_000;

        let qwen3_vl = if path.ends_with(".gguf") {
            let file = std::fs::File::open(path)?;
            let mut reader = std::io::BufReader::new(file);
            let ct = candle_core::quantized::gguf_file::Content::read(&mut reader)?;
            
            if let Some(mm_path) = mmproj_path {
                let mm_file = std::fs::File::open(mm_path)?;
                let mut mm_reader = std::io::BufReader::new(mm_file);
                let ct_mm = candle_core::quantized::gguf_file::Content::read(&mut mm_reader)?;
                let m = QuantizedQwen3VLModel::new(&cfg, &ct, &mut reader, &ct_mm, &mut mm_reader, &text_dev, text_device_id, &vision_dev, vision_device_id, dtype, kv_reserve)?;
                ModelVariant::QuantizedVL(m)
            } else {
                let is_06b = path.contains("0.6B");
                let baking_only = is_06b;
                let single_layer_mode = is_06b;
                let m = QuantizedQwen3TextModel::new(&cfg, &ct, &mut reader, &text_dev, text_device_id, dtype, kv_reserve, baking_only, single_layer_mode)?;
                ModelVariant::QuantizedText(m)
            }
        } else {
            let file = std::fs::File::open(path)?;
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let mut cursor = std::io::Cursor::new(&mmap[..]);
            let content = candle_core::quantized::gguf_file::Content::read(&mut cursor)?;
            let mmap_arc = Arc::new(mmap);
            
            if let Some(mm_path) = mmproj_path {
                let mm_file = std::fs::File::open(mm_path)?;
                let mm_mmap = unsafe { memmap2::MmapOptions::new().map(&mm_file)? };
                let mut mm_cursor = std::io::Cursor::new(&mm_mmap[..]);
                let mm_content = candle_core::quantized::gguf_file::Content::read(&mut mm_cursor)?;
                let mm_mmap_arc = Arc::new(mm_mmap);
                
                let model = QuantizedQwen3VLModel::new_with_mmap(
                    &cfg, &content, Some(mmap_arc), &mm_content, Some(mm_mmap_arc), &text_dev, text_device_id, &vision_dev, vision_device_id, dtype, kv_reserve
                )?;
                ModelVariant::QuantizedVL(model)
            } else {
                let is_06b = path.contains("0.6B");
                let baking_only = is_06b;
                let single_layer_mode = is_06b;
                let model = QuantizedQwen3TextModel::new_with_mmap(
                    &cfg, &content, Some(mmap_arc), &text_dev, text_device_id, dtype, kv_reserve, baking_only, single_layer_mode
                )?;
                ModelVariant::QuantizedText(model)
            }
        };

        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor,
            qwen3_vl,
            text_device: text_dev,
            vision_device: vision_dev,
            eos_token_id1,
            eos_token_id2,
            generation_config,
            model_name,
            hard_token_limit,
        })
    }

    pub fn prefill_text_only(
        &mut self,
        text: &str,
        cancel_token: Option<Arc<AtomicBool>>,
        mut relay_target: Option<&mut Qwen3VLGenerateModel>,
    ) -> Result<()> {
        let tokens = self.tokenizer.encode(text, true).map_err(|e| anyhow!(e))?;
        let token_ids = tokens.get_ids();
        let total_tokens = token_ids.len();
        let chunk_size = 512;
        let mut current_pos = 0;

        println!("[PREFILL-TEXT] Starting raw relay: {} tokens", total_tokens);

        while current_pos < total_tokens {
            if let Some(token) = &cancel_token {
                if token.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); }
            }

            let end = (current_pos + chunk_size).min(total_tokens);
            let chunk = &token_ids[current_pos..end];
            let chunk_ids = Tensor::new(chunk, &self.text_device)?.unsqueeze(0)?;
            let chunk_pos = Tensor::arange(current_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?;

            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos)?;

            if let Some(ref mut target) = relay_target {
                let (ks, vs) = self.get_current_kv();
                let mut new_ks_i8 = Vec::with_capacity(ks.len());
                let mut new_vs_i8 = Vec::with_capacity(vs.len());
                let mut k_scales = Vec::with_capacity(ks.len());
                let mut v_scales = Vec::with_capacity(vs.len());

                for (k, v) in ks.iter().zip(vs.iter()) {
                    let seq_len = k.dim(candle_core::D::Minus2)?; 
                    let start = seq_len.saturating_sub(chunk.len());
                    let k_new = k.narrow(candle_core::D::Minus2, start, chunk.len())?;
                    let v_new = v.narrow(candle_core::D::Minus2, start, chunk.len())?;
                    
                    let k_max = k_new.abs()?.max_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
                    let v_max = v_new.abs()?.max_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
                    let k_s = k_max / 127.0;
                    let v_s = v_max / 127.0;
                    
                    let k_i8 = if k_s > 0.0 { (k_new.to_dtype(DType::F32)? / k_s as f64)?.round()?.to_dtype(DType::U8)? } else { k_new.to_dtype(DType::U8)? };
                    let v_i8 = if v_s > 0.0 { (v_new.to_dtype(DType::F32)? / v_s as f64)?.round()?.to_dtype(DType::U8)? } else { v_new.to_dtype(DType::U8)? };
                    
                    new_ks_i8.push(k_i8);
                    new_vs_i8.push(v_i8);
                    k_scales.push(k_s);
                    v_scales.push(v_s);
                }
                target.inject_kv_quantized(&new_ks_i8, &new_vs_i8, &k_scales, &v_scales)?;
            }
            current_pos = end;
        }
        Ok(())
    }

    pub fn prefill_only(
        &mut self,
        params: ChatCompletionParameters,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        mut relay_target: Option<&mut Qwen3VLGenerateModel>,
    ) -> Result<usize> {
        let mes = params.messages;
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let full_input_ids_vec = self.tokenizer.text_encode_vec(input.replace_text.clone(), true)?;
        let total_tokens = full_input_ids_vec.len();

        let mut current_pos = self.get_kv_len();
        let chunk_size = 512;

        println!("[PREFILL] Starting relay: {} tokens", total_tokens);

        while current_pos < total_tokens {
            if let Some(token) = &cancel_token {
                if token.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); }
            }

            let end = (current_pos + chunk_size).min(total_tokens);
            let end = if end == total_tokens { total_tokens - 1 } else { end };
            if end <= current_pos { break; }

            let chunk = &full_input_ids_vec[current_pos..end];
            let chunk_ids = Tensor::from_vec(chunk.to_vec(), (1, end - current_pos), &self.text_device)?;
            let chunk_pos = Tensor::arange(current_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?;

            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos)?;

            if let Some(ref mut target) = relay_target {
                let (ks, vs) = self.get_current_kv();
                let mut new_ks_i8 = Vec::with_capacity(ks.len());
                let mut new_vs_i8 = Vec::with_capacity(vs.len());
                let mut k_scales = Vec::with_capacity(ks.len());
                let mut v_scales = Vec::with_capacity(vs.len());

                for (k, v) in ks.iter().zip(vs.iter()) {
                    let s_len = k.dim(candle_core::D::Minus2)?; 
                    let start = s_len.saturating_sub(chunk.len());
                    let k_new = k.narrow(candle_core::D::Minus2, start, chunk.len())?;
                    let v_new = v.narrow(candle_core::D::Minus2, start, chunk.len())?;
                    
                    let k_max = k_new.abs()?.max_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
                    let v_max = v_new.abs()?.max_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
                    let k_s = k_max / 127.0;
                    let v_s = v_max / 127.0;
                    
                    let k_i8 = if k_s > 0.0 { (k_new.to_dtype(DType::F32)? / k_s as f64)?.round()?.to_dtype(DType::U8)? } else { k_new.to_dtype(DType::U8)? };
                    let v_i8 = if v_s > 0.0 { (v_new.to_dtype(DType::F32)? / v_s as f64)?.round()?.to_dtype(DType::U8)? } else { v_new.to_dtype(DType::U8)? };
                    
                    new_ks_i8.push(k_i8);
                    new_vs_i8.push(v_i8);
                    k_scales.push(k_s);
                    v_scales.push(v_s);
                }
                target.inject_kv_quantized(&new_ks_i8, &new_vs_i8, &k_scales, &v_scales)?;
            }
            current_pos = end;
        }

        if let Some(sid) = session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(format!("{}.safetensors", sid));
            if !path.exists() { let _ = fs::create_dir_all(&path); }
            self.save_kv_to_disk(&path)?;
            let token_path = path.join("tokens.json");
            if let Ok(file) = fs::File::create(&token_path) {
                let _ = serde_json::to_writer(file, &full_input_ids_vec);
            }
        }

        Ok(current_pos)
    }

    pub fn generate(
        &mut self,
        params: ChatCompletionParameters,
        cancel_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
    ) -> Result<String> {
        let mes = params.messages;
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let full_input_ids_vec = self.tokenizer.text_encode_vec(input.replace_text.clone(), true)?;
        
        let mut current_pos = self.get_kv_len();
        
        if let Some(sid) = &session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(format!("{}.safetensors", sid));
            if path.exists() {
                let token_path = path.join("tokens.json");
                if let Ok(file) = fs::File::open(token_path) {
                    let cached_tokens: Vec<u32> = serde_json::from_reader(file)?;
                    let mut match_len = 0;
                    for (c, f) in cached_tokens.iter().zip(full_input_ids_vec.iter()) {
                        if c == f { match_len += 1; } else { break; }
                    }
                    if match_len > 0 {
                        println!("[KV-BRIDGE] Partial Match ({} tokens). Prefilling remaining.", match_len);
                        self.load_kv_from_disk(&path)?;
                        current_pos = self.get_kv_len();
                    }
                }
            }
        }

        let mut all_ids = full_input_ids_vec.clone();
        let mut generated_text = String::new();
        let max_new_tokens = params.max_tokens.unwrap_or(2048);

        for i in 0..max_new_tokens {
            if let Some(token) = &cancel_token {
                if token.load(Ordering::Relaxed) { break; }
            }

            let input_ids = if i == 0 {
                let start = current_pos;
                let end = full_input_ids_vec.len();
                if start >= end {
                    let last = *full_input_ids_vec.last().unwrap();
                    Tensor::new(vec![last], &self.text_device)?.unsqueeze(0)?
                } else {
                    Tensor::new(&full_input_ids_vec[start..end], &self.text_device)?.unsqueeze(0)?
                }
            } else {
                let last = *all_ids.last().unwrap();
                Tensor::new(vec![last], &self.text_device)?.unsqueeze(0)?
            };

            let seq_len = input_ids.dim(1)?;
            let chunk_pos = Tensor::arange(current_pos as u32, (current_pos + seq_len) as u32, &self.text_device)?.unsqueeze(0)?;
            
            let logits = self.qwen3_vl.forward(&input_ids, None, None, None, None, Some(&chunk_pos), current_pos)?;
            let logits = logits.squeeze(0)?.i(logits.dim(1)? - 1)?;
            let next_id = logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
            
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            
            all_ids.push(next_id);
            let next_token = self.tokenizer.decode(&[next_id], true).map_err(|e| anyhow!(e))?;
            generated_text.push_str(&next_token);
            current_pos += seq_len;
        }

        Ok(generated_text)
    }

    pub fn get_kv_len(&self) -> usize {
        match &self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.language_model.get_kv_len(),
            ModelVariant::QuantizedText(m) => m.language_model.get_kv_len(),
            _ => 0,
        }
    }

    pub fn get_current_kv(&self) -> (Vec<Tensor>, Vec<Tensor>) {
        let mut ks = vec![];
        let mut vs = vec![];
        match &self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => {
                for layer in &m.language_model.layers {
                    if let Some((k, v)) = &layer.self_attn.kv_cache {
                        ks.push(k.clone());
                        vs.push(v.clone());
                    }
                }
            },
            ModelVariant::QuantizedText(m) => {
                for layer in &m.language_model.layers {
                    if let Some((k, v)) = &layer.self_attn.kv_cache {
                        ks.push(k.clone());
                        vs.push(v.clone());
                    }
                }
            },
            _ => {}
        }
        (ks, vs)
    }

    pub fn inject_kv_quantized(&mut self, ks: &[Tensor], vs: &[Tensor], k_scales: &[f32], v_scales: &[f32]) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.language_model.inject_live_kv_quantized(ks, vs, k_scales, v_scales),
            ModelVariant::QuantizedText(m) => m.language_model.inject_live_kv_quantized(ks, vs, k_scales, v_scales),
            _ => Ok(()),
        }
    }

    pub fn save_kv_to_disk(&mut self, path: &Path) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.language_model.save_kv_cache(path, false, 1024),
            ModelVariant::QuantizedText(m) => m.language_model.save_kv_cache(path, false, 1024),
            _ => Ok(()),
        }
    }

    pub fn load_kv_from_disk(&mut self, path: &Path) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.language_model.load_kv_cache(path, &self.text_device, 0, 1),
            ModelVariant::QuantizedText(m) => m.language_model.load_kv_cache(path, &self.text_device, 0, 1),
            _ => Ok(()),
        }
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.to_device(device)?,
            ModelVariant::QuantizedText(m) => m.to_device(device)?,
            _ => {},
        }
        self.text_device = device.clone();
        self.vision_device = device.clone();
        Ok(())
    }

    pub fn clear_kv_cache(&mut self) {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.language_model.clear_kv_cache(),
            ModelVariant::QuantizedText(m) => m.language_model.clear_kv_cache(),
            _ => {},
        }
    }
}

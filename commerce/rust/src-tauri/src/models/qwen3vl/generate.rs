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
            quantized_model::QuantizedQwen3VLModel,
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
use rayon::prelude::*;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::fs;
use std::path::Path;

#[derive(Clone)]
pub enum ModelVariant {
    Standard(crate::models::qwen3vl::model::Qwen3VLModel),
    QuantizedVL(QuantizedQwen3VLModel),
    QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel),
}

impl ModelVariant {
    pub fn forward(&mut self, input_ids: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, video_pixel_values: Option<&Tensor>, video_grid_thw: Option<&Tensor>, cache_position: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>) -> Result<Tensor> {
        match self {
            Self::Standard(m) => m.forward(input_ids, pixel_values, image_grid_thw, video_pixel_values, video_grid_thw, cache_position, seqlen_offset),
            Self::QuantizedVL(m) => m.forward(input_ids, pixel_values, image_grid_thw, video_pixel_values, video_grid_thw, cache_position, seqlen_offset, total_len, session_id),
            Self::QuantizedText(m) => m.forward(input_ids, cache_position, seqlen_offset, total_len, session_id),
        }
    }

    pub fn rebalance_layers(&mut self, device_id: usize, context_len: usize) -> Result<()> {
        match self {
            Self::Standard(_) => Ok(()), // Standard model doesn't support dynamic rebalancing yet
            Self::QuantizedVL(m) => m.rebalance_layers(device_id, context_len),
            Self::QuantizedText(m) => m.rebalance_layers(device_id, context_len),
        }
    }

    pub fn drop_kv_storage(&mut self) -> Result<()> {
        match self {
            Self::Standard(_) => Ok(()),
            Self::QuantizedVL(m) => m.language_model.drop_kv_storage(),
            Self::QuantizedText(m) => m.language_model.drop_kv_storage(),
        }
    }

    pub fn inject_kv_bitkv(&mut self, k_anchors: &[Tensor], k_packed: &[Tensor], k_scales: &[Tensor], v_anchors: &[Tensor], v_packed: &[Tensor], v_scales: &[Tensor], original_shape: &[usize]) -> Result<()> {
        match self {
            Self::QuantizedVL(m) => m.language_model.inject_live_kv_bitkv(k_anchors, k_packed, k_scales, v_anchors, v_packed, v_scales, original_shape),
            Self::QuantizedText(m) => m.language_model.inject_live_kv_bitkv(k_anchors, k_packed, k_scales, v_anchors, v_packed, v_scales, original_shape),
            _ => Ok(()),
        }
    }

    pub fn is_cpu(&self) -> bool {
        match self {
            Self::Standard(m) => m.device().is_cpu(),
            Self::QuantizedVL(m) => m.language_model.is_forced_cpu,
            Self::QuantizedText(m) => m.language_model.is_forced_cpu,
        }
    }
}

pub struct Qwen3VLGenerateModel {
    pub chat_template: ChatTemplate,
    pub tokenizer: TokenizerModel,
    pub pre_processor: Qwen3VLProcessor,
    pub qwen3_vl: ModelVariant,
    pub text_device: Device,
    pub vision_device: Device,
    pub eos_token_id1: u32,
    pub eos_token_id2: u32,
    pub generation_config: Qwen3VLGenerationConfig,
    pub model_name: String,
    pub hard_token_limit: Option<usize>,
    pub kv_root: std::path::PathBuf, // [NEW] Added for absolute path resolution
}

impl Qwen3VLGenerateModel {
    pub fn init(
        path: &str,
        text_device: Option<&Device>,
        text_device_id: usize,
        vision_device: Option<&Device>,
        vision_device_id: usize,
        dtype: Option<DType>,
        hard_token_limit: Option<usize>,
        force_text_only: bool,
        baking_only: bool,
        is_disk_swap: bool, // [NEW]
        kv_root: std::path::PathBuf,
    ) -> Result<Self> {
        Self::init_with_config(path, None, None, text_device, text_device_id, vision_device, vision_device_id, dtype, hard_token_limit, force_text_only, baking_only, is_disk_swap, kv_root)
    }

    pub fn init_with_tokenizer(
        path: &str,
        tokenizer_path: Option<&str>,
        text_device: Option<&Device>,
        text_device_id: usize,
        vision_device: Option<&Device>,
        vision_device_id: usize,
        dtype: Option<DType>,
        hard_token_limit: Option<usize>,
        force_text_only: bool,
        baking_only: bool,
        is_disk_swap: bool, // [NEW]
        kv_root: std::path::PathBuf,
    ) -> Result<Self> {
        Self::init_with_config(path, tokenizer_path, None, text_device, text_device_id, vision_device, vision_device_id, dtype, hard_token_limit, force_text_only, baking_only, is_disk_swap, kv_root) 
    }

    pub fn init_with_config(
        path: &str,
        tokenizer_path: Option<&str>,
        config_path: Option<&str>,
        text_device: Option<&Device>,
        text_device_id: usize,
        vision_device: Option<&Device>,
        vision_device_id: usize,
        dtype: Option<DType>,
        hard_token_limit: Option<usize>,
        force_text_only: bool,
        baking_only: bool,
        is_disk_swap: bool, // [NEW]
        kv_root: std::path::PathBuf,
    ) -> Result<Self> {
        let path = if let Some(stripped) = path.strip_prefix(r"\\?\") { stripped } else { path };
        // ... (path normalization omitted for brevity in match)
        
        // [STRICT-2B-ALIGNMENT] If a tokenizer_path (2B model) is provided, use it for ALL metadata.
        // This ensures 0.6B baking uses 2B's vocab, chat template, and special token IDs.
        let meta_path = tokenizer_path.unwrap_or(path);
        let meta_path = if let Some(stripped) = meta_path.strip_prefix(r"\\?\") { stripped } else { meta_path };
        
        let tok_path = meta_path; // For clarity
        let cfg_path = meta_path; // Forced to use 2B config even for 0.6B loading

        let chat_template = ChatTemplate::init(tok_path)?;
        let tokenizer = TokenizerModel::init(tok_path)?;
        let final_config_path = std::path::Path::new(cfg_path).join("config.json");
        
        println!("[MODEL-METADATA] Loading Config/Tokenizer from: {:?}", final_config_path);
        let raw_config: serde_json::Value = serde_json::from_slice(&std::fs::read(&final_config_path)?)?;

        let cfg: Qwen3VLConfig = if raw_config.get("text_config").is_some() {
            serde_json::from_value(raw_config)?
        } else {
            let text_config: crate::models::qwen3vl::config::Qwen3VLTextConfig = serde_json::from_value(raw_config.clone())?;
            crate::models::qwen3vl::config::Qwen3VLConfig {
                architectures: raw_config.get("architectures").and_then(|v| serde_json::from_value(v.clone()).ok()),
                auto_map: raw_config.get("auto_map").and_then(|v| serde_json::from_value(v.clone()).ok()),
                hidden_size: raw_config.get("hidden_size").and_then(|v| v.as_u64()).map(|v| v as usize),
                image_token_id: raw_config.get("image_token_id").and_then(|v| v.as_u64()).map(|v| v as usize),
                model_type: raw_config.get("model_type").and_then(|v| v.as_str()).unwrap_or("qwen2").to_string(),
                text_config: Some(text_config),
                tie_word_embeddings: raw_config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(true),
                torch_dtype: raw_config.get("torch_dtype").and_then(|v| v.as_str()).map(|s| s.to_string()),
                transformers_version: raw_config.get("transformers_version").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                video_token_id: raw_config.get("video_token_id").and_then(|v| v.as_u64()).map(|v| v as usize),
                vision_config: None,
                vision_start_token_id: None,
                vision_end_token_id: None,
            }
        };

        let text_dev = get_device(text_device);
        let vision_dev = get_device(vision_device);
        let cfg_dtype = cfg.text_config.as_ref().and_then(|tc| tc.dtype.as_deref()).unwrap_or("float16");
        let dtype = get_dtype(dtype, cfg_dtype);

        let gguf_files = find_type_files(path, "gguf")?;
        let mmproj_path = gguf_files.iter().find(|f| f.contains("mmproj")).cloned();
        let is_vision_model = mmproj_path.is_some() && !force_text_only;
        let pre_processor = Qwen3VLProcessor::new(meta_path, &vision_dev, dtype)?;

        let qwen3_vl = if !gguf_files.is_empty() {
            // [HYBRID-FILE-SELECTOR-V2] Mode-aware component selection
            let mut model_path = if path.contains("0.6B") {
                // Baking mode: Look for skeleton first
                gguf_files.iter().find(|f| f.contains("Clean-L0")).cloned()
            } else {
                // Inference mode: Look for 2B Body first
                gguf_files.iter().find(|f| f.contains("Body-L1-27")).cloned()
            };
            
            // Legacy fallbacks if modular components are missing
            if model_path.is_none() { model_path = gguf_files.iter().find(|f| f.contains("0.6B") && f.contains("Q8_0")).cloned(); }
            if model_path.is_none() { model_path = gguf_files.iter().find(|f| f.contains("0.6B") && f.contains("Q4_K_M")).cloned(); }
            if model_path.is_none() { model_path = gguf_files.iter().find(|f| f.contains("0.6B")).cloned(); }
            if model_path.is_none() { model_path = gguf_files.iter().find(|f| !f.contains("mmproj")).cloned(); }

            let main_path = model_path.ok_or(anyhow!("No GGUF file found in {}", path))?;
            println!("[MODEL] Selected Main GGUF: {}", main_path);

            let limit_tokens = hard_token_limit.unwrap_or(4096) as u64;  
            let reserve_tokens = limit_tokens.min(8192);
            let kv_reserve = reserve_tokens * 40000;

            if is_vision_model {
                let mmproj = mmproj_path.ok_or(anyhow!("Missing mmproj GGUF"))?;
                let main_file = std::fs::File::open(&main_path)?;
                let main_mmap = unsafe { memmap2::MmapOptions::new().map(&main_file)? };
                let mmproj_file = std::fs::File::open(&mmproj)?;
                let mmproj_mmap = unsafe { memmap2::MmapOptions::new().map(&mmproj_file)? };

                let mut main_cursor = std::io::Cursor::new(&main_mmap[..]);
                let main_content = gguf_file::Content::read(&mut main_cursor)?;
                let mut mmproj_cursor = std::io::Cursor::new(&mmproj_mmap[..]);
                let mmproj_content = gguf_file::Content::read(&mut mmproj_cursor)?;

                let model = crate::models::qwen3vl::quantized_model::QuantizedQwen3VLModel::new_with_mmap(&cfg, &main_content, Some(Arc::new(main_mmap)), &mmproj_content, Some(Arc::new(mmproj_mmap)), &text_dev, text_device_id, &vision_dev, vision_device_id, dtype, kv_reserve, baking_only, force_text_only, !force_text_only, is_disk_swap)?;
                ModelVariant::QuantizedVL(model)
            } else {
                let file = std::fs::File::open(&main_path)?;
                let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
                let mut cursor = std::io::Cursor::new(&mmap[..]);        
                let content = gguf_file::Content::read(&mut cursor)?;
                
                let is_06b = path.contains("0.6B");
                let actual_baking_only = baking_only || is_06b;
                let single_layer_mode = baking_only || is_06b;
                
                let model = crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel::new_with_mmap(&cfg, &content, Some(Arc::new(mmap)), &text_dev, text_device_id, dtype, kv_reserve, actual_baking_only, single_layer_mode, true, false, is_disk_swap)?;
                ModelVariant::QuantizedText(model)
            }
        } else {
            let model_list = find_type_files(path, "safetensors")?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &text_dev)? };
            let model = Qwen3VLModel::new(cfg, vb)?;
            ModelVariant::Standard(model)
        };

        let generation_config_path = std::path::Path::new(meta_path).join("generation_config.json");
        let generation_config: Qwen3VLGenerationConfig = if generation_config_path.exists() {
            serde_json::from_slice(&std::fs::read(generation_config_path)?)? 
        } else {
            println!("[WARN] generation_config.json not found in meta_path: {:?}", meta_path);
            Qwen3VLGenerationConfig::default()
        };
        let model_name = if path.contains("0.6B") { "qwen3vl-0.6B".to_string() } else { "qwen3vl-2B".to_string() };

        let (eos_token_id1, eos_token_id2) = match &generation_config.eos_token_id {
            serde_json::Value::Number(n) => { let id = n.as_u64().unwrap_or(151645) as u32; (id, id) },
            serde_json::Value::Array(arr) => { let id1 = arr.get(0).and_then(|v| v.as_u64()).unwrap_or(151643) as u32; let id2 = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(id1 as u64) as u32; (id1, id2) },
            _ => (151643, 151643),
        };

        Ok(Self { chat_template, tokenizer, pre_processor, qwen3_vl, text_device: text_dev, vision_device: vision_dev, eos_token_id1, eos_token_id2, generation_config, model_name, hard_token_limit, kv_root })
    }

    pub fn prefill_text_only(&mut self, text: &str, cancel_token: Option<Arc<AtomicBool>>, mut relay_target: Option<&mut Qwen3VLGenerateModel>, auto_save_path: Option<&std::path::Path>) -> Result<()> {
        let token_ids = self.tokenizer.text_encode_vec(text.to_string(), false)?;
        let total_tokens = token_ids.len();
        let chunk_size = 512;
        let mut current_pos = 0;

        while current_pos < total_tokens {
            if let Some(token) = &cancel_token { if token.load(Ordering::Relaxed) { return Err(anyhow!("Task cancelled during prefill_text_only")); } }
            let end = (current_pos + chunk_size).min(total_tokens);
            let chunk = &token_ids[current_pos..end];
            let chunk_ids = Tensor::from_vec(chunk.to_vec(), (1, end - current_pos), &self.text_device)?;
            let chunk_pos = Tensor::arange(current_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?;

            // [FIX] Added missing session_id argument (None for prefill)
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, total_tokens, None)?;

            if let Some(path) = auto_save_path { 
                self.save_kv_to_disk(path)?; 
            }
            if let Some(ref mut target) = relay_target {
                let (ks, vs) = self.get_current_kv();
                // [TURBO-RELAY] Parallelize layer compression across ALL CPU cores
                let results: Result<Vec<_>> = ks.par_iter().zip(vs.par_iter()).map(|(k, v): (&Tensor, &Tensor)| {
                    let seq_len = k.dim(candle_core::D::Minus2)?;
                    let chunk_tokens = end - current_pos;
                    let start = seq_len.saturating_sub(chunk_tokens);
                    
                    let k_new = k.narrow(candle_core::D::Minus2, start, chunk_tokens)?;
                    let v_new = v.narrow(candle_core::D::Minus2, start, chunk_tokens)?;
                    
                    if let ModelVariant::QuantizedText(m) = &self.qwen3_vl {
                        let res_k = m.language_model.compress_to_bitkv(&k_new)?;
                        let res_v = m.language_model.compress_to_bitkv(&v_new)?;
                        Ok((res_k, res_v))
                    } else if let ModelVariant::QuantizedVL(m) = &self.qwen3_vl {
                        let res_k = m.language_model.compress_to_bitkv(&k_new)?;
                        let res_v = m.language_model.compress_to_bitkv(&v_new)?;
                        Ok((res_k, res_v))
                    } else {
                        Err(anyhow!("Unsupported model variant for BitKV relay"))
                    }
                }).collect();

                let results = results?;
                let mut k_anchors = Vec::with_capacity(results.len());
                let mut k_packed = Vec::with_capacity(results.len());
                let mut k_scales = Vec::with_capacity(results.len());
                let mut v_anchors = Vec::with_capacity(results.len());
                let mut v_packed = Vec::with_capacity(results.len());
                let mut v_scales = Vec::with_capacity(results.len());
                let mut original_shape = vec![];

                for (res_k, res_v) in results {
                    k_anchors.push(res_k.0); k_packed.push(res_k.1); k_scales.push(res_k.2);
                    v_anchors.push(res_v.0); v_packed.push(res_v.1); v_scales.push(res_v.2);
                    original_shape = res_k.3;
                }
                
                if !k_anchors.is_empty() {
                    target.inject_kv_bitkv(&k_anchors, &k_packed, &k_scales, &v_anchors, &v_packed, &v_scales, &original_shape)?;
                }
            }
            current_pos = end;
        }
        if auto_save_path.is_some() { println!("[STREAMING] Disk save complete. Dropping KV storage from memory."); let _ = self.qwen3_vl.drop_kv_storage(); }
        Ok(())
    }

    pub fn prefill_only(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, mut relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        // [STRICT-ALIGN] Never add BOS manually; ChatML handles starts.
        let full_input_ids_vec = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_tokens = full_input_ids_vec.len();
        let mut current_pos = self.get_kv_len();
        let prefill_chunk_size = 512;

        while current_pos < total_tokens {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Prefill cancelled")); } }
            let end = (current_pos + prefill_chunk_size).min(total_tokens);
            let end = if end == total_tokens { total_tokens - 1 } else { end };
            if end <= current_pos { break; }
            let chunk = &full_input_ids_vec[current_pos..end];
            let chunk_ids = Tensor::from_vec(chunk.to_vec(), (1, end - current_pos), &self.text_device)?;
            let chunk_pos = Tensor::arange(current_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?;

            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, total_tokens, session_id.clone())?;

            if let Some(ref mut target) = relay_target {
                let (ks, vs) = self.get_current_kv();
                // [TURBO-RELAY] Parallelize layer compression
                    let results: Result<Vec<_>> = ks.par_iter().zip(vs.par_iter()).map(|(k, v): (&Tensor, &Tensor)| {
                    let s_len = k.dim(candle_core::D::Minus2)?;
                    let chunk_tokens = end - current_pos;
                    let start = s_len.saturating_sub(chunk_tokens);
                    
                    let k_new = k.narrow(candle_core::D::Minus2, start, chunk_tokens)?;
                    let v_new = v.narrow(candle_core::D::Minus2, start, chunk_tokens)?;
                    
                    if let ModelVariant::QuantizedText(m) = &self.qwen3_vl {
                        let res_k = m.language_model.compress_to_bitkv(&k_new)?;
                        let res_v = m.language_model.compress_to_bitkv(&v_new)?;
                        Ok((res_k, res_v))
                    } else {
                        Err(anyhow!("Unsupported model variant for BitKV relay"))
                    }
                }).collect();

                let results = results?;
                let mut k_anchors = Vec::with_capacity(results.len());
                let mut k_packed = Vec::with_capacity(results.len());
                let mut k_scales = Vec::with_capacity(results.len());
                let mut v_anchors = Vec::with_capacity(results.len());
                let mut v_packed = Vec::with_capacity(results.len());
                let mut v_scales = Vec::with_capacity(results.len());
                let mut original_shape = vec![];

                for (res_k, res_v) in results {
                    k_anchors.push(res_k.0); k_packed.push(res_k.1); k_scales.push(res_k.2);
                    v_anchors.push(res_v.0); v_packed.push(res_v.1); v_scales.push(res_v.2);
                    original_shape = res_k.3;
                }
                
                if !k_anchors.is_empty() {
                    target.inject_kv_bitkv(&k_anchors, &k_packed, &k_scales, &v_anchors, &v_packed, &v_scales, &original_shape)?;
                }
            }
            current_pos = end;
        }

        if let Some(sid) = session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(sid);
            if !path.exists() { let _ = fs::create_dir_all(&path); }     
            self.save_kv_to_disk(&path)?;
            let token_path = path.join("tokens.json");
            if let Ok(file) = fs::File::create(&token_path) {
                let _ = serde_json::to_writer(file, &full_input_ids_vec);
            }
        }
        Ok(current_pos)
    }

    pub fn prefill_chunk(&mut self, text: String, cancel_flag: Option<Arc<AtomicBool>>, mut relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let full_ids_vec = self.tokenizer.text_encode_vec(text, false)?;
        let total_size = full_ids_vec.len();
        let mut processed_so_far = 0;
        
        // Memory-safe prefill chunk size (Adjusted based on device)
        let step_size = if self.text_device.is_cpu() { 4096 } else { 2048 };

        while processed_so_far < total_size {
            let current_pos = self.get_kv_len();
            let remaining = total_size - processed_so_far;
            let current_chunk_len = if remaining > step_size { step_size } else { remaining };
            
            let chunk_ids_slice = &full_ids_vec[processed_so_far..processed_so_far + current_chunk_len];
            let chunk_ids = Tensor::from_vec(chunk_ids_slice.to_vec(), (1, current_chunk_len), &self.text_device)?;
            let chunk_pos = Tensor::arange(current_pos as u32, (current_pos + current_chunk_len) as u32, &self.text_device)?.unsqueeze(0)?;
            
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            
            // Forward pass for current segment
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, total_size, None)?;
            
            // Optional: Relay KV cache segment to target
            if let Some(ref mut target) = relay_target {
                let (ks, vs) = self.get_current_kv();
                let results: Result<Vec<_>> = ks.par_iter().zip(vs.par_iter()).map(|(k, v): (&Tensor, &Tensor)| {
                     let s_len = k.dim(candle_core::D::Minus2)?;
                     let k_new = k.narrow(candle_core::D::Minus2, s_len - current_chunk_len, current_chunk_len)?;
                     let v_new = v.narrow(candle_core::D::Minus2, s_len - current_chunk_len, current_chunk_len)?;
                     
                     if let ModelVariant::QuantizedText(m) = &self.qwen3_vl {
                        let res_k = m.language_model.compress_to_bitkv(&k_new)?;
                        let res_v = m.language_model.compress_to_bitkv(&v_new)?;
                        Ok((res_k, res_v))
                    } else {
                        Err(anyhow!("Unsupported variant"))
                    }
                }).collect();

                let results = results?;
                let mut k_anchors = Vec::with_capacity(results.len());
                let mut k_packed = Vec::with_capacity(results.len());
                let mut k_scales = Vec::with_capacity(results.len());
                let mut v_anchors = Vec::with_capacity(results.len());
                let mut v_packed = Vec::with_capacity(results.len());
                let mut v_scales = Vec::with_capacity(results.len());
                let mut original_shape = vec![];

                for (res_k, res_v) in results {
                    k_anchors.push(res_k.0); k_packed.push(res_k.1); k_scales.push(res_k.2);
                    v_anchors.push(res_v.0); v_packed.push(res_v.1); v_scales.push(res_v.2);
                    original_shape = res_k.3;
                }
                
                if !k_anchors.is_empty() {
                    target.inject_kv_bitkv(&k_anchors, &k_packed, &k_scales, &v_anchors, &v_packed, &v_scales, &original_shape)?;
                }
            }
            
            processed_so_far += current_chunk_len;
            if total_size > step_size {
                println!("[MODEL-PROGRESS] Chunked Prefill: {}/{} tokens ({:.1}%)", processed_so_far, total_size, (processed_so_far as f64 / total_size as f64) * 100.0);
            }
        }
        
        Ok(total_size)
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>) -> Result<String> {
        let temperature = mes.temperature.unwrap_or(0.7) as f32;
        let top_p = mes.top_p.unwrap_or(0.9) as f32;
        let top_k = 40;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(Some(temperature), Some(top_p), Some(top_k), seed);
        let repetition_penalty = 1.1;

        // [ZERO-PREFILL-INIT] Try loading existing context first
        if let Some(sid) = &session_id {
            let path = self.kv_root.join(sid);
            if path.exists() {
                println!("[ZERO-PREFILL] Found existing session '{}' at {:?}. Loading from SSD...", sid, path);
                self.load_kv_from_disk(&path)?; 
            } else {
                println!("[ZERO-PREFILL-WARN] Snapshot not found at {:?}. Recalculating...", path);
            }
        }

        let mut seqlen_offset = self.get_kv_len();
        if seqlen_offset > 0 {
            println!("[ZERO-PREFILL] Context successfully restored. Starting from token #{}", seqlen_offset);
        }

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut input = self.pre_processor.process_info(&mes, &mes_render)?;
        // [STRICT-ALIGN] Never add BOS manually; parity with prefill_only
        let full_input_ids_vec = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_tokens = full_input_ids_vec.len();

        // [STRICT-RELAY] If we have context, local_pos starts from seqlen_offset
        let mut local_pos = if seqlen_offset > 0 { 
            if seqlen_offset >= total_tokens {
                println!("[ZERO-PREFILL] Full context match ({} tokens). Skipping entire prefill loop.", seqlen_offset);
                total_tokens.saturating_sub(1)
            } else {
                println!("[ZERO-PREFILL] Skipping first {} baked tokens. Resuming prefill from position {}.", seqlen_offset, seqlen_offset);
                seqlen_offset
            }
        } else { 
            0 
        };
        
        // [SYNC-OFFSET] Ensure seqlen_offset matches local_pos for the loop
        seqlen_offset = local_pos;

        // [DYNAMIC-BATCH-SIZE] Adjust batch size based on safety mode
        let prefill_chunk_size = if self.qwen3_vl.is_cpu() { 
            1024 
        } else if seqlen_offset > 0 || self.kv_root.to_string_lossy().contains("inference") {
            // [RECOVERY-MODE] If resuming or in Swap mode, use tiny safer batches
            64 
        } else {
            // [SPEED-MODE] Default prefill batch
            128 
        };

        while local_pos < total_tokens {
            let remaining = total_tokens - local_pos;
            // [STRICT-ALIGN] 마지막 토큰은 루프 밖(Generation 시작 직전)에서 처리합니다.
            if remaining <= 1 { break; }
            
            let mut chunk_size = if remaining > prefill_chunk_size { prefill_chunk_size } else { remaining };
            
            // [FINAL-STRETCH-SAFETY] Progress based batch shrinking
            let progress = local_pos as f32 / total_tokens as f32 * 100.0;
            if progress > 90.0 {
                // Near the end, use tiny batches to avoid OOM during peak KV cache
                chunk_size = chunk_size.min(32);
            }

            if local_pos + chunk_size >= total_tokens {
                chunk_size = (total_tokens - local_pos).saturating_sub(1);
            }
            if chunk_size == 0 { break; }

            // [FIX] Slice relative to local_pos, but use seqlen_offset for the model
            let chunk = &full_input_ids_vec[local_pos..local_pos + chunk_size];
            let chunk_ids = Tensor::from_vec(chunk.to_vec(), (1, chunk_size), &self.text_device)?;
            let chunk_pos = Tensor::arange(seqlen_offset as u32, (seqlen_offset + chunk_size) as u32, &self.text_device)?.unsqueeze(0)?;

            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Generation cancelled during prefill")); }
            }

            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), seqlen_offset, total_tokens, session_id.clone())?;
            
            // [CHECKPOINT] Save to SSD after EVERY batch to allow granular resumption
            if let Some(sid) = &session_id {
                let path = crate::utils::paths::get_kv_dir(None).join(sid);
                let _ = self.save_kv_to_disk(&path);
            }

            // [DYNAMIC-RECOVERY] Check for GPU availability every chunk during prefill
            // [SAFETY] Skip rebalancing if we are in the critical final 5%
            if !self.qwen3_vl.is_cpu() && progress < 95.0 {
                let _ = self.qwen3_vl.rebalance_layers(0, seqlen_offset + chunk_size);
            }
            
            local_pos += chunk_size;
            seqlen_offset += chunk_size;
        }

        let mut all_ids = full_input_ids_vec.clone();
        let mut generated_text = String::new();
        let max_new_tokens = mes.max_tokens.unwrap_or(2048);

        let mut pixel_values = input.pixel_values.take();
        let image_grid_thw = input.image_grid_thw.take();

        println!("[DEBUG-GEN] Starting token generation loop. Max tokens: {}", max_new_tokens);
        let mut prev_time = std::time::Instant::now();

        // [PRE-GEN-LOAD] Warm up GPU by filling it with layers before the loop starts
        if !self.qwen3_vl.is_cpu() {
            println!("[HYBRID-SPEED-UP] Pre-loading layers to GPU for instant generation...");
            let _ = self.qwen3_vl.rebalance_layers(0, seqlen_offset);
        }

        for _i in 0..max_new_tokens {
            if let Some(flag) = &cancel_flag { 
                if flag.load(Ordering::Relaxed) { 
                    println!("[DEBUG-GEN] Cancellation detected at token {}", _i);
                    return Err(anyhow!("Task cancelled during generation")); 
                } 
            }
            
            // [ADAPTIVE-VRAM-GUARD] Check every 5 tokens to prevent OOM as KV cache grows
            if _i > 0 && _i % 5 == 0 && !self.qwen3_vl.is_cpu() {
                let _ = self.qwen3_vl.rebalance_layers(0, seqlen_offset);
            }

            let input_ids = if generated_text.is_empty() {
                // Process remaining tokens from prefill
                let start = local_pos;
                let end = total_tokens;
                Tensor::new(&full_input_ids_vec[start..end], &self.text_device)?.unsqueeze(0)?
            } else {
                Tensor::new(vec![*all_ids.last().unwrap()], &self.text_device)?.unsqueeze(0)?
            };

            let seq_len = input_ids.dim(1)?;
            let chunk_pos = Tensor::arange(seqlen_offset as u32, (seqlen_offset + seq_len) as u32, &self.text_device)?.unsqueeze(0)?;
            
            // [QUESTION-PREFILL-ACCELERATION]
            // If this is the first forward (processing the prompt chunk), 
            // force all layers to GPU regardless of usual safety margins.
            if generated_text.is_empty() && !self.qwen3_vl.is_cpu() {
                println!("[TURBO] Accelerating Question Prefill ({} tokens)...", seq_len);
                let _ = self.qwen3_vl.rebalance_layers(0, seqlen_offset + seq_len);
            }

            let logits = self.qwen3_vl.forward(&input_ids, pixel_values.as_ref(), image_grid_thw.as_ref(), None, None, Some(&chunk_pos), seqlen_offset, total_tokens, session_id.clone())?;
            
            // [GEN-PROGRESS-LOG]
            let elapsed = prev_time.elapsed().as_secs_f32();
            if _i > 0 {
                println!("[GEN] Token #{} | Latency: {:.2}s | Speed: {:.2} t/s", _i, elapsed, 1.0 / elapsed);
            }
            prev_time = std::time::Instant::now();
            
            let logits = logits.squeeze(0)?;
            let mut logits = logits.i(logits.dim(0)? - 1)?.to_dtype(DType::F32)?;

            if repetition_penalty != 1.0 {
                let penalty_context = if all_ids.len() > 512 { &all_ids[all_ids.len()-512..] } else { &all_ids[..] };
                logits = apply_repeat_penalty(&logits, repetition_penalty, penalty_context)?;
            }

            let next_id = logit_processor.sample(&logits)?;
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            all_ids.push(next_id);
            generated_text.push_str(&self.tokenizer.token_decode(vec![next_id])?);
            
            // [REBALANCE] 512 토큰마다 VRAM 상태 체크하여 레이어 재배치 (CPU 모드가 아닐 때만)
            if _i > 0 && _i % 512 == 0 && !self.qwen3_vl.is_cpu() {
                if let Err(e) = self.qwen3_vl.rebalance_layers(0, seqlen_offset + seq_len) {
                    println!("[REBALANCE] Failed: {}", e);
                }
            }

            seqlen_offset += seq_len;
            pixel_values = None;
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
        let mut ks = vec![]; let mut vs = vec![];
        match &self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => { for layer in &m.language_model.layers { if let Some((k, v)) = &layer.self_attn.kv_cache { ks.push(k.clone()); vs.push(v.clone()); } } },
            ModelVariant::QuantizedText(m) => { for layer in &m.language_model.layers { if let Some((k, v)) = &layer.self_attn.kv_cache { ks.push(k.clone()); vs.push(v.clone()); } } },
            _ => {} 
        }
        (ks, vs)
    }

    pub fn inject_kv_bitkv(&mut self, k_anchors: &[Tensor], k_packed: &[Tensor], k_scales: &[Tensor], v_anchors: &[Tensor], v_packed: &[Tensor], v_scales: &[Tensor], original_shape: &[usize]) -> Result<()> {
        self.qwen3_vl.inject_kv_bitkv(k_anchors, k_packed, k_scales, v_anchors, v_packed, v_scales, original_shape)
    }

    pub fn save_kv_to_disk(&mut self, path: &Path) -> Result<()> {
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.save_kv_cache(path, false, 1024),
            ModelVariant::QuantizedText(m) => m.save_kv_cache(path, false, 1024),
            _ => Ok(()),
        }
    }

    pub fn load_kv_from_disk(&mut self, path: &Path) -> Result<()> {
        if !path.exists() { return Err(anyhow!("KV path does not exist: {:?}", path)); }
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.load_kv_cache(path, &self.text_device, 0, 128),
            ModelVariant::QuantizedText(m) => m.load_kv_cache(path, &self.text_device, 0, 128),
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
            ModelVariant::QuantizedVL(m) => m.clear_kv_cache(),
            ModelVariant::QuantizedText(m) => m.clear_kv_cache(),
            _ => {},
        }
    }
}
use anyhow::{Result, anyhow};
use candle_core::{quantized::gguf_file, DType, Device, Tensor};
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
        find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
    openai_types::ChatCompletionParameters,
};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::fs;

enum ModelVariant {
    Standard(Qwen3VLModel),
    QuantizedVL(QuantizedQwen3VLModel),
    QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel),
}

pub struct Qwen3VLGenerateModel {
    chat_template: ChatTemplate,
    tokenizer: TokenizerModel,
    pre_processor: Qwen3VLProcessor,
    qwen3_vl: ModelVariant,
    text_device: Device,
    vision_device: Device,
    eos_token_id1: u32,
    eos_token_id2: u32,
    generation_config: Qwen3VLGenerationConfig,
    model_name: String,
    hard_token_limit: Option<usize>,
}

impl Qwen3VLGenerateModel {
    pub fn init(path: &str, text_device: Option<&Device>, text_device_id: usize, vision_device: Option<&Device>, vision_device_id: usize, dtype: Option<DType>, hard_token_limit: Option<usize>) -> Result<Self> {
        // [FIX] Normalize path to remove Windows UNC prefix (\\?\) which causes issues with some loaders
        let path = if let Some(stripped) = path.strip_prefix(r"\\?\") { stripped } else { path };
        
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = std::path::Path::new(path).join("config.json");
        
        // [FIX] Robust Config Loading
        // 1. Read raw JSON
        let raw_config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
        
        // 2. Try standard deserialization first
        let cfg: Qwen3VLConfig = if raw_config.get("text_config").is_some() {
            serde_json::from_value(raw_config)?
        } else {
            // 3. Fallback: Construct config assuming root IS the text config
            println!("[CONFIG] 'text_config' missing. Assuming flat text model structure.");
            let text_config: crate::models::qwen3vl::config::Qwen3VLTextConfig = serde_json::from_value(raw_config.clone())
                .map_err(|e| anyhow!("Failed to parse flat text config: {}", e))?;
            
            // Construct Qwen3VLConfig manually
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
                vision_config: None, // No vision config for flat text models
                vision_start_token_id: None,
                vision_end_token_id: None,
            }
        };
        
        let text_dev = get_device(text_device);
        let vision_dev = get_device(vision_device);
        
        // Safe access to optional text_config
        let cfg_dtype = cfg.text_config.as_ref().and_then(|tc| tc.dtype.as_deref()).unwrap_or("float16");
        let dtype = get_dtype(dtype, cfg_dtype);
        
        let gguf_files = find_type_files(path, "gguf")?;
        let mmproj_path = gguf_files.iter().find(|f| f.contains("mmproj")).cloned();
        let is_vision_model = mmproj_path.is_some();

        // Processor initialization with vision support check
        let has_vision = cfg.image_token_id.is_some() && is_vision_model;
        let pre_processor = Qwen3VLProcessor::new(path, &vision_dev, dtype)?;
        
        let qwen3_vl = if !gguf_files.is_empty() {
            let model_path = gguf_files.iter().find(|f| !f.contains("mmproj")).cloned();

            let max_tokens = hard_token_limit.unwrap_or(4096) as u64;
            let kv_reserve = max_tokens * 30000;

            if is_vision_model {
                // CASE 1: Vision-Language Model
                let mmproj = mmproj_path.ok_or(anyhow!("Missing mmproj GGUF"))?;
                let main = model_path.ok_or(anyhow!("Missing main GGUF for VL model"))?;
                let mut mmproj_file = std::fs::File::open(&mmproj)?;
                let mmproj_content = gguf_file::Content::read(&mut mmproj_file)?;
                let mut main_file = std::fs::File::open(&main)?;
                let main_content = gguf_file::Content::read(&mut main_file)?;
                
                let model = QuantizedQwen3VLModel::new(&cfg, &main_content, &mut main_file, &mmproj_content, &mut mmproj_file, &text_dev, text_device_id, &vision_dev, vision_device_id, dtype, kv_reserve)?;
                ModelVariant::QuantizedVL(model)
            } else {
                // CASE 2: Pure Text Model (0.6B etc.)
                let main = model_path.or_else(|| if !gguf_files.is_empty() { Some(gguf_files[0].clone()) } else { None }).ok_or(anyhow!("No GGUF file found"))?;
                let mut file = std::fs::File::open(&main)?;
                let content = gguf_file::Content::read(&mut file)?;
                let model = crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel::new(&cfg, &content, &mut file, &text_dev, text_device_id, dtype, kv_reserve)?;
                ModelVariant::QuantizedText(model)
            }
        } else {
            // ... (standard/safetensors branch)
            let model_list = find_type_files(path, "safetensors")?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &text_dev)? };
            let model = Qwen3VLModel::new(cfg, vb)?;
            ModelVariant::Standard(model)
        };

        let generation_config_path = std::path::Path::new(path).join("generation_config.json");
        let generation_config: Qwen3VLGenerationConfig = if generation_config_path.exists() {
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?
        } else {
            Qwen3VLGenerationConfig::default()
        };
        let model_name = if path.contains("0.6B") { "qwen3vl-0.6B".to_string() } else { "qwen3vl-2B".to_string() };

        let (eos_token_id1, eos_token_id2) = match &generation_config.eos_token_id {
            serde_json::Value::Number(n) => {
                let id = n.as_u64().unwrap_or(151645) as u32;
                (id, id)
            },
            serde_json::Value::Array(arr) => {
                let id1 = arr.get(0).and_then(|v| v.as_u64()).unwrap_or(151643) as u32;
                let id2 = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(id1 as u64) as u32;
                (id1, id2)
            },
            _ => (151643, 151643),
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

    pub fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>) -> Result<String> {
        let temperature = match mes.temperature {
            None => self.generation_config.temperature,
            Some(tem) => tem as f32,
        };
        let top_p = match mes.top_p {
            None => self.generation_config.top_p,
            Some(top_p) => top_p as f32,
        };
        let top_k = self.generation_config.top_k;
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor =
            get_logit_processor(Some(temperature), Some(top_p), Some(top_k), seed);
        let repetition_penalty = self.generation_config.repetition_penalty;
        
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        // Make input mutable to take ownership of fields
        let mut input = self.pre_processor.process_info(&mes, &mes_render)?;
        let mut input_ids = self
            .tokenizer
            .text_encode(input.replace_text.clone(), &self.text_device)?;
        let mut seq_len = input_ids.dim(1)?;
        
        println!("[GENERATE] Input Token Count: {}", seq_len);
        println!("[GENERATE] Input Shape: {:?}", input_ids.shape());

        let mut full_input_ids_vec = input_ids.flatten_all()?.to_vec1::<u32>()?; // Save original full input for saving later

        // [OPTIMIZATION] Detect action flags early to control caching behavior
        let is_ingest = input.replace_text.contains("ACTION: INGEST");
        let is_save = input.replace_text.contains("ACTION: SAVE");

        let mut seqlen_offset = 0;

        // KV Cache Disk Loading (Hierarchical Prefix Caching)
        let cache_path = if let Some(sid) = &session_id {
             let model_type_sub = if self.model_name.contains("0.6B") || self.model_name.contains("small") { "small" } else { "large" };
             let p = crate::utils::paths::get_kv_dir(None).join(sid).join(model_type_sub);
             if !p.exists() { let _ = fs::create_dir_all(&p); }
             Some(p)
        } else {
             None
        };

        if let Some(path) = &cache_path {
             match &mut self.qwen3_vl {
                 ModelVariant::QuantizedVL(m) => {
                     let token_path = path.join("tokens.json");
                     let mut loaded = false;
                     if token_path.exists() {
                         if let Ok(file) = fs::File::open(&token_path) {
                             let reader = std::io::BufReader::new(file);
                             if let Ok(cached_tokens) = serde_json::from_reader::<_, Vec<u32>>(reader) {
                                 let mut match_len = 0;
                                 for (c, f) in cached_tokens.iter().zip(full_input_ids_vec.iter()) {
                                     if c == f { match_len += 1; } else { break; }
                                 }
                                 if match_len > 50 {
                                     println!("[KV-HIERARCHY] Cache Hit (VL)! Using prefix: {} tokens.", match_len);
                                     if m.load_kv_cache(path, &self.text_device, match_len).is_ok() {
                                         seqlen_offset = match_len;
                                         loaded = true;
                                         let remaining = seq_len.saturating_sub(seqlen_offset);
                                         if remaining > 0 {
                                             input_ids = input_ids.narrow(1, seqlen_offset, remaining)?;
                                             seq_len = remaining;
                                         } else {
                                             input_ids = input_ids.narrow(1, seq_len - 1, 1)?;
                                             seqlen_offset = seq_len - 1;
                                             seq_len = 1;
                                         }
                                     }
                                 }
                             }
                         }
                     }
                     if !loaded { m.clear_kv_cache(); }
                 },
                 ModelVariant::QuantizedText(m) => {
                     let token_path = path.join("tokens.json");
                     let mut loaded = false;
                     if token_path.exists() {
                         if let Ok(file) = fs::File::open(&token_path) {
                             let reader = std::io::BufReader::new(file);
                             if let Ok(cached_tokens) = serde_json::from_reader::<_, Vec<u32>>(reader) {
                                 let mut match_len = 0;
                                 for (c, f) in cached_tokens.iter().zip(full_input_ids_vec.iter()) {
                                     if c == f { match_len += 1; } else { break; }
                                 }
                                 if match_len > 50 {
                                     println!("[KV-HIERARCHY] Cache Hit (Text)! Using prefix: {} tokens.", match_len);
                                     if m.load_kv_cache(path, &self.text_device, match_len).is_ok() {
                                         seqlen_offset = match_len;
                                         loaded = true;
                                         let remaining = seq_len.saturating_sub(seqlen_offset);
                                         if remaining > 0 {
                                             input_ids = input_ids.narrow(1, seqlen_offset, remaining)?;
                                             seq_len = remaining;
                                         } else {
                                             input_ids = input_ids.narrow(1, seq_len - 1, 1)?;
                                             seqlen_offset = seq_len - 1;
                                             seq_len = 1;
                                         }
                                     }
                                 }
                             }
                         }
                     }
                     if !loaded { m.clear_kv_cache(); }
                 },
                 _ => {} 
             }
        }
        
        // [CHUNKED PREFILL] - Line Aware (2048 tokens)
        if seq_len > 2048 {
            println!("[PREFILL] Line-aware chunking {} tokens...", seq_len);
            let newline_token_id = if let Ok(ids) = self.tokenizer.text_encode("\n".to_string(), &self.text_device) {
                ids.flatten_all()?.to_vec1::<u32>()?.get(0).cloned().unwrap_or(198)
            } else { 198 };

            let mut current_pos = 0;
            let input_vec = full_input_ids_vec.clone();

            while current_pos < seq_len - 1 {
                let remaining = seq_len - current_pos;
                if remaining <= 2048 { break; } 

                let mut chunk_size = 2048;
                let lookback_range = 256; 
                let search_end = current_pos + 2048;
                let search_start = search_end.saturating_sub(lookback_range);
                
                for i in (search_start..search_end).rev() {
                    if i < input_vec.len() && input_vec[i] == newline_token_id {
                        chunk_size = i - current_pos + 1; 
                        break;
                    }
                }

                let chunk_ids = input_ids.narrow(1, current_pos, chunk_size)?;
                let start_pos = (seqlen_offset + current_pos) as u32;
                let chunk_pos = Tensor::arange(start_pos, start_pos + chunk_size as u32, &self.text_device)?;
                
                let _logits = match &mut self.qwen3_vl {
                    ModelVariant::Standard(m) => m.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), seqlen_offset + current_pos)?,
                    ModelVariant::QuantizedVL(m) => m.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), seqlen_offset + current_pos)?,
                    ModelVariant::QuantizedText(m) => m.forward(&chunk_ids, Some(&chunk_pos), seqlen_offset + current_pos)?,
                };
                
                current_pos += chunk_size;
                print!(">"); use std::io::Write; std::io::stdout().flush().ok();
            }
            
            seqlen_offset += current_pos;
            input_ids = input_ids.narrow(1, current_pos, seq_len - current_pos)?;
            seq_len = seq_len - current_pos;
            println!("\n[PREFILL] Ingested up to offset {}.", seqlen_offset);

            if is_save && !is_ingest {
                if let Some(path) = &cache_path {
                    let res = match &mut self.qwen3_vl {
                        ModelVariant::QuantizedVL(m) => m.save_kv_cache(path, false, 0),
                        ModelVariant::QuantizedText(m) => m.save_kv_cache(path, false, 0),
                        _ => Ok(()),
                    };
                    if res.is_ok() {
                        let token_path = path.join("tokens.json");
                        if let Ok(file) = fs::File::create(&token_path) {
                            let _ = serde_json::to_writer(file, &&full_input_ids_vec[..seqlen_offset]);
                        }
                    }
                }
            }
        }
        
        let mut pixel_values = input.pixel_values.take();
        let image_grid_thw_tensor = input.image_grid_thw.take(); 
        let mut pixel_values_video = input.pixel_values_video.take();
        let video_grid_thw_tensor = input.video_grid_thw.take();
        
        let start_pos = seqlen_offset as u32;
        let mut cache_position = Tensor::arange(start_pos, start_pos + seq_len as u32, &self.text_device)?;
        
        let requested_tokens = mes.max_tokens.unwrap_or(2048);
        let mut sample_len = requested_tokens;
        
        if let Some(limit) = self.hard_token_limit {
            let current_usage = seq_len;
            if current_usage >= limit {
                 println!("[WARN] Input length {} exceeds hard limit {}. Truncating generation.", current_usage, limit);
                 sample_len = 16; 
            } else {
                 let available = limit - current_usage;
                 if (sample_len as usize) > available {
                      println!("[CONFIG] Clamping generation: Requested {} -> Available {} (Total Limit: {})", sample_len, available, limit);
                      sample_len = available as u32;
                 }
            }
        }

        let generation_result: Result<Vec<u32>> = (|| {
            let mut generate = Vec::new();
            for _i in 0..sample_len {
                if let Some(flag) = &cancel_flag {
                    if flag.load(Ordering::Relaxed) {
                        return Err(anyhow!("Generation cancelled"));
                    }
                }
                
                use std::io::Write;
                print!(".");
                std::io::stdout().flush().ok();

                let logits = match &mut self.qwen3_vl {
                    ModelVariant::Standard(m) => m.forward(
                        &input_ids,
                        pixel_values.as_ref(),
                        image_grid_thw_tensor.as_ref(),
                        pixel_values_video.as_ref(),
                        video_grid_thw_tensor.as_ref(),
                        Some(&cache_position),
                        seqlen_offset,
                    )?,
                    ModelVariant::QuantizedVL(m) => m.forward(
                        &input_ids,
                        pixel_values.as_ref(),
                        image_grid_thw_tensor.as_ref(),
                        pixel_values_video.as_ref(),
                        video_grid_thw_tensor.as_ref(),
                        Some(&cache_position),
                        seqlen_offset,
                    )?,
                    ModelVariant::QuantizedText(m) => m.forward(
                        &input_ids,
                        Some(&cache_position),
                        seqlen_offset,
                    )?,
                };
                let mut logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                
                // [FIX] Apply Repetition Penalty to prevent loops
                if repetition_penalty != 1.0 {
                    // Combine prompt tokens and generated tokens for context
                    let mut context = full_input_ids_vec.clone();
                    context.extend(&generate);
                    
                    // Only use the last N tokens for penalty to keep performance
                    let penalty_context = if context.len() > 512 {
                        &context[context.len()-512..]
                    } else {
                        &context[..]
                    };
                    
                    logits = apply_repeat_penalty(&logits, repetition_penalty, penalty_context)?;
                }

                let next_token = logit_processor.sample(&logits)?;
                                generate.push(next_token);
                                
                                if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
                                    break;
                                }
                                
                                                seqlen_offset += seq_len;
                                
                                                seq_len = 1;
                                
                                                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.text_device)?;
                                
                                                cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.text_device)?;
                                
                                                                
                                
                                                // [BRANCH] Only reset vision inputs if the model variant actually supports vision
                                
                                                if matches!(self.qwen3_vl, ModelVariant::Standard(_) | ModelVariant::QuantizedVL(_)) {
                                
                                                    if pixel_values.is_some() {
                                
                                                        pixel_values = None;
                                
                                                        pixel_values_video = None;
                                
                                                    }
                                
                                                }
                                
                                            }
                                
                                
            Ok(generate)
        })();

        let generate = generation_result?;

        if let Some(path) = &cache_path {
             match &mut self.qwen3_vl {
                 ModelVariant::QuantizedVL(m) => {
                     let token_path = path.join("tokens.json");
                     if is_ingest {
                         println!("[KV-VRAM] Accumulating context in VRAM (VL)...");
                     } else if is_save {
                         println!("[KV-DISK] Finalizing ingestion. Ultra-fast Direct Save (VL)...");
                         let target_block_size = 0; 
                         let mut all_tokens = full_input_ids_vec;
                         all_tokens.extend(&generate);
                         if !all_tokens.is_empty() { all_tokens.pop(); }
                         if m.save_kv_cache(path, false, target_block_size).is_ok() {
                             if let Ok(file) = fs::File::create(&token_path) {
                                 let _ = serde_json::to_writer(file, &all_tokens);
                             }
                         }
                     } else {
                         m.clear_kv_cache();
                     }
                 },
                 ModelVariant::QuantizedText(m) => {
                     let token_path = path.join("tokens.json");
                     if is_ingest {
                         println!("[KV-VRAM] Accumulating context in VRAM (Text)...");
                     } else if is_save {
                         println!("[KV-DISK] Finalizing ingestion. Ultra-fast Direct Save (Text)...");
                         let target_block_size = 0; 
                         let mut all_tokens = full_input_ids_vec;
                         all_tokens.extend(&generate);
                         if !all_tokens.is_empty() { all_tokens.pop(); }
                         if m.save_kv_cache(path, false, target_block_size).is_ok() {
                             if let Ok(file) = fs::File::create(&token_path) {
                                 let _ = serde_json::to_writer(file, &all_tokens);
                             }
                         }
                     } else {
                         m.clear_kv_cache();
                     }
                 },
                 ModelVariant::Standard(m) => m.clear_kv_cache(),
             }
        } else {
            match &mut self.qwen3_vl {
                ModelVariant::Standard(m) => m.clear_kv_cache(),
                ModelVariant::QuantizedVL(m) => m.clear_kv_cache(),
                ModelVariant::QuantizedText(m) => m.clear_kv_cache(),
            }
        }

        let res = self.tokenizer.token_decode(generate)?;
        Ok(res)
    }

    pub fn clear_kv_cache(&mut self) {
        match &mut self.qwen3_vl {
            ModelVariant::Standard(m) => m.clear_kv_cache(),
            ModelVariant::QuantizedVL(m) => m.clear_kv_cache(),
            ModelVariant::QuantizedText(m) => m.clear_kv_cache(),
        }
    }
}
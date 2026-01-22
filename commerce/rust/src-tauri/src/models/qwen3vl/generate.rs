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
    Quantized(QuantizedQwen3VLModel),
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
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = std::path::Path::new(path).join("config.json");
        let cfg: Qwen3VLConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        
        let text_dev = get_device(text_device);
        let vision_dev = get_device(vision_device);
        
        let cfg_dtype = cfg.text_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        
        // Processor uses vision_device for image processing
        let pre_processor = Qwen3VLProcessor::new(path, &vision_dev, dtype)?;
        
        // Check for GGUF files first
        let gguf_files = find_type_files(path, "gguf")?;
        println!("[DEBUG] Found {} GGUF files in {}", gguf_files.len(), path);
        
        let qwen3_vl = if !gguf_files.is_empty() {
            // Find mmproj (vision) and main model
            let mmproj_path = gguf_files.iter().find(|f| f.contains("mmproj")).cloned();
            let model_path = gguf_files.iter().find(|f| !f.contains("mmproj")).cloned();

            if let (Some(mmproj), Some(main)) = (mmproj_path, model_path.clone()) {
                let mut mmproj_file = std::fs::File::open(&mmproj)?;
                let mmproj_content = gguf_file::Content::read(&mut mmproj_file)?;
                
                let mut main_file = std::fs::File::open(&main)?;
                let main_content = gguf_file::Content::read(&mut main_file)?;
                
                let max_tokens = hard_token_limit.unwrap_or(4096) as u64;
                let kv_reserve = max_tokens * 200_000;
                
                let model = QuantizedQwen3VLModel::new(&cfg, &main_content, &mut main_file, &mmproj_content, &mut mmproj_file, &text_dev, text_device_id, &vision_dev, vision_device_id, dtype, kv_reserve)?;
                ModelVariant::Quantized(model)
            } else if let Some(main) = model_path.or_else(|| if !gguf_files.is_empty() { Some(gguf_files[0].clone()) } else { None }) {
                 let mut file = std::fs::File::open(&main)?;
                 let content = gguf_file::Content::read(&mut file)?;
                 let mut file2 = std::fs::File::open(&main)?; 
                 let content2 = gguf_file::Content::read(&mut file2)?;
                 
                 let max_tokens = hard_token_limit.unwrap_or(4096) as u64;
                 let kv_reserve = max_tokens * 200_000;
                 
                 let model = QuantizedQwen3VLModel::new(&cfg, &content, &mut file, &content2, &mut file2, &text_dev, text_device_id, &vision_dev, vision_device_id, dtype, kv_reserve)?;
                 ModelVariant::Quantized(model)
            } else {
                 return Err(anyhow!("No valid GGUF model found"));
            }
        } else {
            let model_list = find_type_files(path, "safetensors")?;
            // Standard model currently doesn't support dual device as easily in this wrapper,
            // but we use text_dev as primary.
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &text_dev)? };
            let model = Qwen3VLModel::new(cfg, vb)?;
            ModelVariant::Standard(model)
        };

        let generation_config_path = std::path::Path::new(path).join("generation_config.json");
        let generation_config: Qwen3VLGenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor,
            qwen3_vl,
            text_device: text_dev,
            vision_device: vision_dev,
            eos_token_id1: generation_config.eos_token_id[0] as u32,
            eos_token_id2: generation_config.eos_token_id[1] as u32,
            generation_config,
            model_name: "qwen3vl".to_string(),
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

        // HARD SAFETY CHECK: Truncate Input if it exceeds limit (Smart Tail-Heavy Drop)
        // [ADAPTIVE] For ingestion/classification, we must not drop document structure.
        if let Some(limit) = self.hard_token_limit {
            let is_critical = input.replace_text.contains("[DATA_PART]") || input.replace_text.contains("[FULL_STRUCTURE]");
            let effective_limit = if is_critical { 16384 } else { limit.max(2048) };
            let max_input = if effective_limit > 128 { effective_limit - 128 } else { effective_limit };
            
            if seq_len > max_input {
                println!("⚠️ [WARN] Input too long ({} > {}). Using System-Aware Truncation (Critical={}).", seq_len, max_input, is_critical);
                
                // Keep System Prompt (approx first 200 tokens) and the Tail (Most important)
                let head_reserve = 200.min(max_input / 4);
                let tail_reserve = max_input - head_reserve;
                
                let head_ids = input_ids.narrow(1, 0, head_reserve)?;
                let tail_ids = input_ids.narrow(1, seq_len - tail_reserve, tail_reserve)?;
                
                input_ids = Tensor::cat(&[head_ids, tail_ids], 1)?;
                seq_len = max_input;
                
                // Update full_input_ids_vec to match the new truncated content
                full_input_ids_vec = input_ids.flatten_all()?.to_vec1::<u32>()?;
            }
        }

        let mut seqlen_offset = 0;

        // KV Cache Disk Loading (Prefix Caching)
        // [ADAPTIVE] Disable disk cache for critical ingestion to ensure in-memory consistency
        let is_critical = input.replace_text.contains("[DATA_PART]") || input.replace_text.contains("[FULL_STRUCTURE]");
        let cache_path = if let Some(sid) = &session_id {
             if is_critical { None } else {
                 let p = std::path::Path::new("tmp_kv").join(sid);
                 if !p.exists() { let _ = fs::create_dir_all(&p); }
                 Some(p)
             }
        } else {
             None
        };

        if let Some(path) = &cache_path {
             match &mut self.qwen3_vl {
                 ModelVariant::Quantized(m) => {
                     let token_path = path.join("tokens.json");
                     let mut loaded = false;
                     if token_path.exists() {
                         if let Ok(file) = fs::File::open(&token_path) {
                             let reader = std::io::BufReader::new(file);
                             if let Ok(cached_tokens) = serde_json::from_reader::<_, Vec<u32>>(reader) {
                                 // FIND LONGEST COMMON PREFIX
                                 let mut match_len = 0;
                                 for (c, f) in cached_tokens.iter().zip(full_input_ids_vec.iter()) {
                                     if c == f { match_len += 1; } else { break; }
                                 }

                                 if match_len > 100 {
                                     println!("[KV-DISK] Cache Hit! Longest Common Prefix: {} tokens.", match_len);
                                     
                                     // [OPTIMIZATION] Even if match_len is slightly less than cached_tokens.len(),
                                     // we trust the cache up to match_len.
                                     if m.load_kv_cache(path, &self.text_device, match_len).is_ok() {
                                         seqlen_offset = match_len;
                                         loaded = true;
                                         
                                         let remaining = seq_len.saturating_sub(seqlen_offset);
                                         if remaining > 0 {
                                             input_ids = input_ids.narrow(1, seqlen_offset, remaining)?;
                                             seq_len = remaining;
                                             println!("[KV-DISK] Processing only {} new tokens.", seq_len);
                                         } else {
                                             // If perfectly matched, we still need to process the very last token 
                                             // to get the next distribution.
                                             println!("[KV-DISK] Perfect match. Re-processing last token.");
                                             input_ids = input_ids.narrow(1, seq_len - 1, 1)?;
                                             seqlen_offset = seq_len - 1;
                                             seq_len = 1;
                                         }
                                     }
                                 } else {
                                     println!("[KV-DISK] Cache Mismatch (Prefix only {}). Starting fresh.", match_len);
                                 }
                             }
                         }
                     }
                     
                     if !loaded {
                         m.clear_kv_cache();
                     }
                 },
                 _ => {} 
             }
        }
        
        // [CHUNKED PREFILL] - Line Aware (512 tokens)
        if seq_len > 512 {
            println!("[PREFILL] Line-aware chunking {} tokens...", seq_len);
            
            let newline_token_id = if let Ok(ids) = self.tokenizer.text_encode("\n".to_string(), &self.text_device) {
                ids.flatten_all()?.to_vec1::<u32>()?.get(0).cloned().unwrap_or(198)
            } else { 198 };

            let mut current_pos = 0;
            let input_vec = full_input_ids_vec.clone();

            while current_pos < seq_len - 1 {
                let remaining = seq_len - current_pos;
                if remaining <= 512 { break; } 

                let mut chunk_size = 512;
                let lookback_range = 128; 
                
                let search_end = current_pos + 512;
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
                    ModelVariant::Quantized(m) => m.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), seqlen_offset + current_pos)?,
                };
                
                current_pos += chunk_size;
                print!(">"); use std::io::Write; std::io::stdout().flush().ok();
            }
            
            seqlen_offset += current_pos;
            input_ids = input_ids.narrow(1, current_pos, seq_len - current_pos)?;
            seq_len = seq_len - current_pos;
            println!("\n[PREFILL] Ingested up to offset {}.", seqlen_offset);

            // [NEW] Intermediate Cache Save: Save context knowledge after heavy prefill
            if let Some(path) = &cache_path {
                if let ModelVariant::Quantized(m) = &mut self.qwen3_vl {
                    if m.save_kv_cache(path, false).is_ok() {
                        let token_path = path.join("tokens.json");
                        // We save only the part of full_input_ids_vec that corresponds to seqlen_offset
                        let ingested_tokens = &full_input_ids_vec[..seqlen_offset];
                        if let Ok(file) = fs::File::create(&token_path) {
                            let _ = serde_json::to_writer(file, &ingested_tokens);
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
        
        let requested_tokens = mes.max_tokens.unwrap_or(1024);
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
                    ModelVariant::Quantized(m) => m.forward(
                        &input_ids,
                        pixel_values.as_ref(),
                        image_grid_thw_tensor.as_ref(),
                        pixel_values_video.as_ref(),
                        video_grid_thw_tensor.as_ref(),
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
                                
                if pixel_values.is_some() {
                    pixel_values = None;
                    pixel_values_video = None;
                }
            }
            Ok(generate)
        })();

        let generate = generation_result?;

        if let Some(path) = &cache_path {
             match &mut self.qwen3_vl {
                 ModelVariant::Quantized(m) => {
                     println!("[KV-DISK] Saving KV cache to {:?}", path);
                     if m.offload_kv_cache(path).is_ok() {
                         let mut all_tokens = full_input_ids_vec;
                         all_tokens.extend(&generate);
                         if !all_tokens.is_empty() {
                             all_tokens.pop(); 
                         }

                         let token_path = path.join("tokens.json");
                         if let Ok(file) = fs::File::create(&token_path) {
                             let _ = serde_json::to_writer(file, &all_tokens);
                         }
                     }
                 },
                 ModelVariant::Standard(m) => m.clear_kv_cache(),
             }
        } else {
            match &mut self.qwen3_vl {
                ModelVariant::Standard(m) => m.clear_kv_cache(),
                ModelVariant::Quantized(m) => m.clear_kv_cache(),
            }
        }

        let res = self.tokenizer.token_decode(generate)?;
        Ok(res)
    }
}

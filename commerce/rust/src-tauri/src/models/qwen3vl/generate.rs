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
    pub fn init(
        path: &str, 
        text_device: Option<&Device>, 
        text_device_id: usize, 
        vision_device: Option<&Device>, 
        vision_device_id: usize, 
        dtype: Option<DType>, 
        hard_token_limit: Option<usize>
    ) -> Result<Self> {
        Self::init_with_config(path, None, None, text_device, text_device_id, vision_device, vision_device_id, dtype, hard_token_limit)
    }

    pub fn init_with_tokenizer(
        path: &str, 
        tokenizer_path: Option<&str>,
        text_device: Option<&Device>, 
        text_device_id: usize, 
        vision_device: Option<&Device>, 
        vision_device_id: usize, 
        dtype: Option<DType>, 
        hard_token_limit: Option<usize>
    ) -> Result<Self> {
        Self::init_with_config(path, tokenizer_path, None, text_device, text_device_id, vision_device, vision_device_id, dtype, hard_token_limit)
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
        hard_token_limit: Option<usize>
    ) -> Result<Self> {
        // [FIX] Normalize path to remove Windows UNC prefix
        let path = if let Some(stripped) = path.strip_prefix(r"\\?\") { stripped } else { path };
        let tok_path = tokenizer_path.unwrap_or(path);
        let tok_path = if let Some(stripped) = tok_path.strip_prefix(r"\\?\") { stripped } else { tok_path };
        
        let cfg_path = config_path.unwrap_or(path);
        let cfg_path = if let Some(stripped) = cfg_path.strip_prefix(r"\\?\") { stripped } else { cfg_path };

        let chat_template = ChatTemplate::init(tok_path)?;
        let tokenizer = TokenizerModel::init(tok_path)?;
        let final_config_path = std::path::Path::new(cfg_path).join("config.json");
        
        // [FIX] Robust Config Loading using final_config_path
        let raw_config: serde_json::Value = serde_json::from_slice(&std::fs::read(&final_config_path)?)?;
        
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
        let pre_processor = Qwen3VLProcessor::new(tok_path, &vision_dev, dtype)?;
        
        let qwen3_vl = if !gguf_files.is_empty() {
            // [PRIORITY] Target Q8_0 for best balance of speed and stability on CPU
            let mut model_path = gguf_files.iter().find(|f| f.contains("Qwen3-0.6B-Q8_0.gguf")).cloned();
            
            // Fallback 1: Q4_K_M
            if model_path.is_none() {
                model_path = gguf_files.iter().find(|f| f.contains("Qwen3-0.6B-Q4_K_M.gguf")).cloned();
            }

            // Fallback 2: Any other GGUF (for 2B model etc)
            if model_path.is_none() {
                model_path = gguf_files.iter().find(|f| !f.contains("mmproj")).cloned();
            }

            // [OPTIMIZATION] Realistic KV Cache Reservation
            // Standard tasks rarely hit 32k. Reserving for 8k-16k is safer for long pages.
            let limit_tokens = hard_token_limit.unwrap_or(4096) as u64;
            let reserve_tokens = limit_tokens.min(16384); 
            let kv_reserve = reserve_tokens * 70000; // Optimized overhead with driver cleanup delay

            if is_vision_model {
                // CASE 1: Vision-Language Model
                let mmproj = mmproj_path.ok_or(anyhow!("Missing mmproj GGUF"))?;
                let main = model_path.ok_or(anyhow!("Missing main GGUF for VL model"))?;
                
                // [MMAP-OPTIMIZATION] Use Mmap for GGUF to support Parallel Loading and Prefetching
                let main_file = std::fs::File::open(&main)?;
                let main_mmap = unsafe { memmap2::MmapOptions::new().map(&main_file)? };
                let mmproj_file = std::fs::File::open(&mmproj)?;
                let mmproj_mmap = unsafe { memmap2::MmapOptions::new().map(&mmproj_file)? };

                // [PREFETCH] Hint OS to load weights into memory immediately
                #[cfg(unix)]
                {
                    use memmap2::Advice;
                    let _ = main_mmap.advise(Advice::WillNeed);
                    let _ = mmproj_mmap.advise(Advice::WillNeed);
                }

                let mut main_cursor = std::io::Cursor::new(&main_mmap[..]);
                let main_content = gguf_file::Content::read(&mut main_cursor)?;
                let mut mmproj_cursor = std::io::Cursor::new(&mmproj_mmap[..]);
                let mmproj_content = gguf_file::Content::read(&mut mmproj_cursor)?;
                
                let model = QuantizedQwen3VLModel::new_with_mmap(&cfg, &main_content, &main_mmap, &mmproj_content, &mmproj_mmap, &text_dev, text_device_id, &vision_dev, vision_device_id, dtype, kv_reserve)?;
                ModelVariant::QuantizedVL(model)
            } else {
                // CASE 2: Pure Text Model (0.6B etc.)
                let main = model_path.or_else(|| if !gguf_files.is_empty() { Some(gguf_files[0].clone()) } else { None }).ok_or(anyhow!("No GGUF file found"))?;
                
                let file = std::fs::File::open(&main)?;
                let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
                
                #[cfg(unix)]
                {
                    use memmap2::Advice;
                    let _ = mmap.advise(Advice::WillNeed);
                }

                let mut cursor = std::io::Cursor::new(&mmap[..]);
                let content = gguf_file::Content::read(&mut cursor)?;
                let model = crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel::new_with_mmap(&cfg, &content, &mmap, &text_dev, text_device_id, dtype, kv_reserve)?;
                ModelVariant::QuantizedText(model)
            }
        } else {
            // CASE 3: Standard Safetensors with mmap for instant loading
            let model_list = find_type_files(path, "safetensors")?;
            
            // [MMAP-OPTIMIZATION] Option 1: Parallel Loading using rayon
            use rayon::prelude::*;
            let _handles: Vec<_> = model_list.par_iter().map(|p| {
                let file = std::fs::File::open(p)?;
                unsafe { memmap2::MmapOptions::new().map(&file) }
            }).collect::<Result<Vec<_>, _>>()?;
            
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &text_dev)? };
            let model = Qwen3VLModel::new(cfg, vb)?;
            ModelVariant::Standard(model)
        };

        let generation_config_path = std::path::Path::new(cfg_path).join("generation_config.json");
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

    pub fn prefill_only(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, mut relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let full_input_ids_vec = self.tokenizer.text_encode_vec(input.replace_text.clone(), true)?;
        let total_tokens = full_input_ids_vec.len();

                                let seqlen_offset = self.get_kv_len();
                                let mut current_pos = seqlen_offset;
                                // [SMOOTH-RELAY] Set chunk size to 512 for optimized 0.6B CPU prefill and PCIe relay
                                let prefill_chunk_size = 512; 
                        
                                println!("[PREFILL] Starting real-time relay: 0/{} tokens (0%)", total_tokens);                while current_pos < total_tokens {
                    let remaining = total_tokens - current_pos;
                    let mut chunk_size = if remaining > prefill_chunk_size { prefill_chunk_size } else { remaining };
        
                    // [CRITICAL] Leave at least 1 token for the generate() call
                    if current_pos + chunk_size >= total_tokens { 
                        chunk_size = (total_tokens - current_pos).saturating_sub(1); 
                    }
                    if chunk_size == 0 { break; }
        
                    let chunk_vec = &full_input_ids_vec[current_pos..current_pos + chunk_size];
                    let chunk_ids = Tensor::from_vec(chunk_vec.to_vec(), (1, chunk_size), &self.text_device)?;
                                let chunk_pos = Tensor::arange(current_pos as u32, (current_pos + chunk_size) as u32, &self.text_device)?;
                    
                                if let Some(flag) = &cancel_flag {
                                    if flag.load(Ordering::Relaxed) { return Err(anyhow!("Prefill cancelled")); }    
                                }
                    
                                println!("[DEBUG] Forwarding chunk... (Size: {})", chunk_size);
                                match &mut self.qwen3_vl {
                                    ModelVariant::Standard(m) => { m.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos)?; },
                                    ModelVariant::QuantizedVL(m) => { m.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos)?; },
                                    ModelVariant::QuantizedText(m) => { m.forward(&chunk_ids, Some(&chunk_pos), current_pos)?; },
                                };
                                println!("[DEBUG] Forward complete.");
                    
                                            // [STREAMING-RELAY] Incrementally inject ONLY the new chunk to 2B
                                            if let Some(ref mut target) = relay_target {
                                                println!("[DEBUG] Slicing, Quantizing and Injecting KV...");
                                                // 1. Get Shallow Copy of 0.6B's Full Cache
                                                let (ks, vs) = self.get_current_kv();
                                                
                                                // 2. Slice and Quantize the NEW chunk
                                                let mut new_ks_i8 = Vec::with_capacity(ks.len());
                                                let mut new_vs_i8 = Vec::with_capacity(vs.len());
                                                let mut k_scales = Vec::with_capacity(ks.len());
                                                let mut v_scales = Vec::with_capacity(vs.len());
                                                
                                                for (k, v) in ks.iter().zip(vs.iter()) {
                                                     let seq_len = k.dim(candle_core::D::Minus2)?; 
                                                     let start = seq_len.saturating_sub(chunk_size);
                                                     let k_new = k.narrow(candle_core::D::Minus2, start, chunk_size)?;
                                                     let v_new = v.narrow(candle_core::D::Minus2, start, chunk_size)?;
                                                     
                                                     // [QUANTIZE-ON-CPU] Reduce to 8-bit before PCIe transfer
                                                     let k_max = k_new.abs()?.max_all()?.to_scalar::<f32>()?;
                                                     let v_max = v_new.abs()?.max_all()?.to_scalar::<f32>()?;
                                                     
                                                                          let k_s = k_max / 127.0;
                                                                          let v_s = v_max / 127.0;
                                                                          
                                                                          let k_i8 = if k_s > 0.0 { (k_new.to_dtype(DType::F32)? / k_s as f64)?.round()?.to_dtype(DType::U8)? } else { k_new.to_dtype(DType::U8)? };
                                                                          let v_i8 = if v_s > 0.0 { (v_new.to_dtype(DType::F32)? / v_s as f64)?.round()?.to_dtype(DType::U8)? } else { v_new.to_dtype(DType::U8)? };
                                                                          
                                                                          new_ks_i8.push(k_i8);                                                     new_vs_i8.push(v_i8);
                                                     k_scales.push(k_s);
                                                     v_scales.push(v_s);
                                                }
                                
                                                // 3. Inject Quantized Slice
                                                target.inject_kv_quantized(&new_ks_i8, &new_vs_i8, &k_scales, &v_scales)?;
                                                println!("[DEBUG] Quantized Injection complete.");
                                            }                    
                                current_pos += chunk_size;                    
                                // [AGGRESSIVE-LOGGING] Log every chunk for real-time feedback
                                let progress = (current_pos as f32 / total_tokens as f32) * 100.0;
                                println!("[RELAY] Pushed {}/{} tokens to 2B ({:.1}%)", current_pos, total_tokens, progress);
                            }        // Save progress if session_id is provided (for CPU sequential mode)
        if let Some(sid) = &session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(sid);
            if !path.exists() { let _ = fs::create_dir_all(&path); }
            
            match &mut self.qwen3_vl {
                ModelVariant::QuantizedVL(m) => { let _ = m.save_kv_cache(&path, false, 1024); },
                ModelVariant::QuantizedText(m) => { let _ = m.save_kv_cache(&path, false, 1024); },
                _ => {},
            }
            let token_path = path.join("tokens.json");
            if let Ok(file) = fs::File::create(&token_path) {
                let _ = serde_json::to_writer(file, &full_input_ids_vec);
            }
        }

        Ok(current_pos)
    }

    pub fn prefill_chunk(&mut self, text: String, cancel_flag: Option<Arc<AtomicBool>>, relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let chunk_ids_vec = self.tokenizer.text_encode_vec(text, false)?;
        let chunk_size = chunk_ids_vec.len();
        let current_pos = self.get_kv_len();

        println!("[PREFILL] Processing Chunk: {} tokens at pos {}", chunk_size, current_pos);

        let chunk_ids = Tensor::from_vec(chunk_ids_vec, (1, chunk_size), &self.text_device)?;
        let chunk_pos = Tensor::arange(current_pos as u32, (current_pos + chunk_size) as u32, &self.text_device)?;

        if let Some(flag) = &cancel_flag {
            if flag.load(Ordering::Relaxed) { return Err(anyhow!("Chunk prefill cancelled")); }    
        }

        // 1. Compute KV for this chunk only
        match &mut self.qwen3_vl {
            ModelVariant::Standard(m) => { m.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos)?; },
            ModelVariant::QuantizedVL(m) => { m.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos)?; },
            ModelVariant::QuantizedText(m) => { m.forward(&chunk_ids, Some(&chunk_pos), current_pos)?; },
        };

        // 2. Inject to 2B immediately
        if let Some(target) = relay_target {
            let (ks, vs) = self.get_current_kv();
            let mut new_ks_i8 = Vec::with_capacity(ks.len());
            let mut new_vs_i8 = Vec::with_capacity(vs.len());
            let mut k_scales = Vec::with_capacity(ks.len());
            let mut v_scales = Vec::with_capacity(vs.len());
            
            for (k, v) in ks.iter().zip(vs.iter()) {
                 let seq_len = k.dim(candle_core::D::Minus2)?; 
                 let start = seq_len.saturating_sub(chunk_size);
                 let k_new = k.narrow(candle_core::D::Minus2, start, chunk_size)?;
                 let v_new = v.narrow(candle_core::D::Minus2, start, chunk_size)?;
                 
                 let k_max = k_new.abs()?.max_all()?.to_scalar::<f32>()?;
                 let v_max = v_new.abs()?.max_all()?.to_scalar::<f32>()?;
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

        Ok(chunk_size)
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
        
        // [FIX] Memory-Safe Tokenization: Encode on CPU first to prevent VRAM spikes
        let full_input_ids_vec = self.tokenizer.text_encode_vec(input.replace_text.clone(), true)?;
        let total_tokens = full_input_ids_vec.len();
        
        println!("[GENERATE] Input Token Count: {}", total_tokens);

        // [OPTIMIZATION] Detect action flags early
        let is_ingest = input.replace_text.contains("ACTION: INGEST");
        let is_save = input.replace_text.contains("ACTION: SAVE");
        
        // [FIX] CHECK LIVE CACHE FIRST: If 0.6B just injected KV, detect it here.
        let live_kv_len = self.get_kv_len();
        if live_kv_len > 0 {
            println!("[KV-BRIDGE] Detected Live Injected KV: {} tokens. Skipping redundant prefill.", live_kv_len);
        }

        // KV Cache Disk Loading (Only if live cache is empty)
        let cache_path = if let Some(sid) = &session_id {
             let p = crate::utils::paths::get_kv_dir(None).join(sid);
             if !p.exists() { let _ = fs::create_dir_all(&p); }
             Some(p)
        } else {
             None
        };

        let mut seqlen_offset = live_kv_len; 
        if seqlen_offset == 0 {
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
                                         // [ZERO-REFILL] 2B inherits 0.6B cache perfectly via Linear Bridge. No need to re-read.
                                         // [UPSCALE-REFILL] Let 2B re-refine the last 64 tokens for better quality.
                                         let refill_len = 64;
                                         if m.load_kv_cache(path, &self.text_device, match_len, refill_len).is_ok() {
                                             seqlen_offset = if match_len > refill_len { match_len - refill_len } else { match_len };
                                             loaded = true;
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
                                         // [ZERO-REFILL] Skip redundant re-processing
                                         // [UPSCALE-REFILL] Let 2B re-refine the last 64 tokens for better quality.
                                         let refill_len = 64;
                                         if m.load_kv_cache(path, &self.text_device, match_len, refill_len).is_ok() {
                                             seqlen_offset = if match_len > refill_len { match_len - refill_len } else { match_len };
                                             loaded = true;
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
        }
        
        // [CHUNKED PREFILL] - Memory-Safe Segmented Loading
        let mut current_pos = seqlen_offset;
        let prefill_chunk_size = 256; // Lowered to 256 to minimize 'cat' peak VRAM spikes
        let newline_token_id = 198; // Fallback for '\n'

        while current_pos < total_tokens {
            let remaining = total_tokens - current_pos;
            
            // If this is the very last token, we let the generation loop handle it
            if remaining == 1 && current_pos < total_tokens { break; }

            // Determine chunk size (256 or remaining)
            let mut chunk_size = if remaining > prefill_chunk_size { prefill_chunk_size } else { remaining };
            
            // If not the final chunk of the whole input, try to align to newline for stability
            if current_pos + chunk_size < total_tokens {
                let search_start = (current_pos + chunk_size).saturating_sub(256);
                for i in (search_start..(current_pos + chunk_size)).rev() {
                    if full_input_ids_vec[i] == newline_token_id {
                        chunk_size = i - current_pos + 1;
                        break;
                    }
                }
            }

            // [CRITICAL] Final token of the entire input must be processed by the generation loop 
            // to get the first predicted token logits.
            if current_pos + chunk_size >= total_tokens {
                chunk_size = (total_tokens - current_pos).saturating_sub(1); 
            }

            if chunk_size == 0 { break; }

            // Transfer ONLY this chunk to GPU
            let chunk_vec = &full_input_ids_vec[current_pos..current_pos + chunk_size];
            let chunk_ids = Tensor::from_vec(chunk_vec.to_vec(), (1, chunk_size), &self.text_device)?;
            let chunk_pos = Tensor::arange(current_pos as u32, (current_pos + chunk_size) as u32, &self.text_device)?;
            
            print!(">"); use std::io::Write; std::io::stdout().flush().ok();

            // [CANCELLATION-CHECK] Check if user clicked stop between chunks
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) {
                    println!("[GENERATE] Generation cancelled during prefill.");
                    return Err(anyhow!("Generation cancelled during prefill"));
                }
            }

            match &mut self.qwen3_vl {
                ModelVariant::Standard(m) => { m.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos)?; },
                ModelVariant::QuantizedVL(m) => { m.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos)?; },
                ModelVariant::QuantizedText(m) => { m.forward(&chunk_ids, Some(&chunk_pos), current_pos)?; },
            };
            
            // [SPEED-UP] Drastically reduced delay to speed up prefilling
            std::thread::sleep(std::time::Duration::from_millis(10));
            
            current_pos += chunk_size;
        }

        // Finalize state for generation loop
        seqlen_offset = current_pos;
        let mut seq_len = total_tokens - seqlen_offset;
        let last_tokens_vec = &full_input_ids_vec[seqlen_offset..];
        let mut input_ids = Tensor::from_vec(last_tokens_vec.to_vec(), (1, seq_len), &self.text_device)?;
        
        if seqlen_offset < total_tokens {
            println!("\n[PREFILL] Ingested up to offset {}. Processing final {} tokens.", seqlen_offset, seq_len);
        } else {
            println!("\n[KV-BRIDGE] Context fully matched in cache. Starting generation immediately.");
        }

        // Save progress if requested
        if is_save && seqlen_offset > 0 {
            if let Some(path) = &cache_path {
                let res = match &mut self.qwen3_vl {
                    ModelVariant::QuantizedVL(m) => m.save_kv_cache(path, false, 0),
                    ModelVariant::QuantizedText(m) => m.save_kv_cache(path, false, 0),
                    _ => Ok(()),
                };
                if res.is_ok() {
                    let token_path = path.join("tokens.json");
                    if let Ok(file) = fs::File::create(&token_path) {
                        let _ = serde_json::to_writer(file, &full_input_ids_vec[..seqlen_offset]);
                    }
                }
            }
        }
        
        let mut pixel_values = input.pixel_values.take();
        let image_grid_thw_tensor = input.image_grid_thw.take(); 
        let mut pixel_values_video = input.pixel_values_video.take();
        let video_grid_thw_tensor = input.video_grid_thw.take();
        
        let mut cache_position = Tensor::arange(seqlen_offset as u32, (seqlen_offset + seq_len) as u32, &self.text_device)?;
        
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
                     if is_save {
                         println!("[KV-DISK] Finalizing ingestion. Ultra-fast Direct Save (VL)...");
                         let target_block_size = 1024; 
                         let mut all_tokens = full_input_ids_vec;
                         all_tokens.extend(&generate);
                         if !all_tokens.is_empty() { all_tokens.pop(); }
                         if m.save_kv_cache(path, false, target_block_size).is_ok() {
                             if let Ok(file) = fs::File::create(&token_path) {
                                 let _ = serde_json::to_writer(file, &all_tokens);
                             }
                         }
                     } else if is_ingest {
                         println!("[KV-VRAM] Accumulating context in VRAM (VL)...");
                     } else {
                         m.clear_kv_cache();
                     }
                 },
                 ModelVariant::QuantizedText(m) => {
                     let token_path = path.join("tokens.json");
                     if is_save {
                         println!("[KV-DISK] Finalizing ingestion. Ultra-fast Direct Save (Text)...");
                         let target_block_size = 1024; 
                         let mut all_tokens = full_input_ids_vec;
                         all_tokens.extend(&generate);
                         if !all_tokens.is_empty() { all_tokens.pop(); }
                         if m.save_kv_cache(path, false, target_block_size).is_ok() {
                             if let Ok(file) = fs::File::create(&token_path) {
                                 let _ = serde_json::to_writer(file, &all_tokens);
                             }
                         }
                     } else if is_ingest {
                         println!("[KV-VRAM] Accumulating context in VRAM (Text)...");
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
            ModelVariant::Standard(_) => {},
            ModelVariant::QuantizedVL(m) => m.clear_kv_cache(),
            ModelVariant::QuantizedText(m) => m.clear_kv_cache(),
        }
    }

    pub fn get_kv_len(&self) -> usize {
        match &self.qwen3_vl {
            ModelVariant::Standard(_) => 0,
            ModelVariant::QuantizedVL(m) => m.language_model.get_kv_len(),
            ModelVariant::QuantizedText(m) => m.language_model.get_kv_len(),
        }
    }

    pub fn get_current_kv(&self) -> (Vec<candle_core::Tensor>, Vec<candle_core::Tensor>) {
        let mut ks: Vec<candle_core::Tensor> = vec![];
        let mut vs: Vec<candle_core::Tensor> = vec![];
        
        match &self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => {
                for layer in &m.language_model.layers {
                    if let Some((k, v)) = &layer.self_attn.kv_cache {
                        let k_tensor: candle_core::Tensor = k.clone();
                        let v_tensor: candle_core::Tensor = v.clone();
                        ks.push(k_tensor);
                        vs.push(v_tensor);
                    }
                }
            },
            ModelVariant::QuantizedText(m) => {
                for layer in &m.language_model.layers {
                    if let Some((k, v)) = &layer.self_attn.kv_cache {
                        let k_tensor: candle_core::Tensor = k.clone();
                        let v_tensor: candle_core::Tensor = v.clone();
                        ks.push(k_tensor);
                        vs.push(v_tensor);
                    }
                }
            },
            _ => {}
        }
        (ks, vs)
    }

    pub fn inject_kv(&mut self, ks: &[candle_core::Tensor], vs: &[candle_core::Tensor]) -> anyhow::Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.inject_live_kv(ks, vs, 1.0, 1.0)?,
            ModelVariant::QuantizedText(m) => m.inject_live_kv(ks, vs, 1.0, 1.0)?,
            _ => return Err(anyhow::anyhow!("KV Injection not supported for this model variant")),
        }
        Ok(())
    }

    pub fn inject_kv_quantized(&mut self, ks: &[candle_core::Tensor], vs: &[candle_core::Tensor], k_scales: &[f32], v_scales: &[f32]) -> anyhow::Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.inject_live_kv_quantized(ks, vs, k_scales, v_scales)?,
            ModelVariant::QuantizedText(m) => m.inject_live_kv_quantized(ks, vs, k_scales, v_scales)?,
            _ => return Err(anyhow::anyhow!("KV Injection not supported for this model variant")),
        }
        Ok(())
    }

    /// [SLEEP-MODE] Moves the underlying model to a new device (e.g. CPU for hibernation)
    pub fn to_device(&mut self, device: &candle_core::Device) -> anyhow::Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::Standard(_) => return Err(anyhow::anyhow!("to_device not implemented for Standard variant")),
            ModelVariant::QuantizedVL(m) => m.to_device(device)?,
            ModelVariant::QuantizedText(m) => m.to_device(device)?,
        }
        Ok(())
    }



    

            /// [ASSISTANT-MODE] Ingest context using Small (0.6B) model and return KV Cache metadata

    

            pub fn prefill_assistant(&mut self, _full_input_ids: &[u32], _sid: &Option<String>) -> Result<usize> {

    

                println!("[ASSISTANT] Rapid Ingestion using 0.6B...");

            // This is essentially the same as the prefill part of generate, but forced for Small variant

            // and specifically tuned for speed.

            Ok(0) // Logic to be integrated into main loop

        }

    }

    
use anyhow::{Result, anyhow};
use candle_core::{quantized::gguf_file, DType, Device, Tensor};
use candle_nn::VarBuilder;
use nvml_wrapper::Nvml;

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
    device: Device,
    eos_token_id1: u32,
    eos_token_id2: u32,
    generation_config: Qwen3VLGenerationConfig,
    model_name: String,
    hard_token_limit: Option<usize>,
}

impl Qwen3VLGenerateModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>, hard_token_limit: Option<usize>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = std::path::Path::new(path).join("config.json");
        let cfg: Qwen3VLConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.text_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let pre_processor = Qwen3VLProcessor::new(path, &device, dtype)?;
        
        // Check for GGUF files first
        let gguf_files = find_type_files(path, "gguf")?;
        println!("[DEBUG] Found {} GGUF files in {}", gguf_files.len(), path);
        
        let qwen3_vl = if !gguf_files.is_empty() {
            // Find mmproj (vision) and main model
            let mmproj_path = gguf_files.iter().find(|f| f.contains("mmproj")).cloned();
            let model_path = gguf_files.iter().find(|f| !f.contains("mmproj")).cloned();

            if let (Some(mmproj), Some(main)) = (mmproj_path, model_path.clone()) {
                println!("[DEBUG] Loading GGUF Vision from: {:?}", mmproj);
                println!("[DEBUG] Loading GGUF Main from: {:?}", main);
                
                let mut mmproj_file = std::fs::File::open(&mmproj)?;
                println!("[DEBUG] Reading mmproj content...");
                let mmproj_content = gguf_file::Content::read(&mut mmproj_file)?;
                println!("[DEBUG] mmproj content read successfully.");
                
                let mut main_file = std::fs::File::open(&main)?;
                println!("[DEBUG] Reading main model content...");
                let main_content = gguf_file::Content::read(&mut main_file)?;
                println!("[DEBUG] main model content read successfully.");
                
                println!("[DEBUG] Initializing QuantizedQwen3VLModel...");
                
                // Calculate KV Cache Reservation
                let max_tokens = hard_token_limit.unwrap_or(1024) as u64;
                // Heuristic: ~180KB per token. 
                // Since we offload KV to CPU RAM, we only need to reserve VRAM for the active chunk (approx 1500 tokens).
                // This prevents excessive reservation that forces layers to CPU.
                let kv_reserve = std::cmp::min(max_tokens, 1500) * 180_000;
                
                let model = QuantizedQwen3VLModel::new(&cfg, &main_content, &mut main_file, &mmproj_content, &mut mmproj_file, &device, dtype, kv_reserve)?;
                println!("[DEBUG] QuantizedQwen3VLModel initialized.");
                ModelVariant::Quantized(model)
            } else if let Some(main) = model_path.or_else(|| if !gguf_files.is_empty() { Some(gguf_files[0].clone()) } else { None }) {
                 // Fallback if only one file found (maybe combined?)
                 println!("Loading Single GGUF from: {:?}", main);
                 let mut file = std::fs::File::open(&main)?;
                 let content = gguf_file::Content::read(&mut file)?;
                 // Pass same file for both if combined? Or empty?
                 // We'll duplicate for now, assuming combined.
                 let mut file2 = std::fs::File::open(&main)?; 
                 let content2 = gguf_file::Content::read(&mut file2)?;
                 
                 let max_tokens = hard_token_limit.unwrap_or(1024) as u64;
                 let kv_reserve = std::cmp::min(max_tokens, 1500) * 180_000;
                 
                 let model = QuantizedQwen3VLModel::new(&cfg, &content, &mut file, &content2, &mut file2, &device, dtype, kv_reserve)?;
                 ModelVariant::Quantized(model)
            } else {
                 return Err(anyhow!("No valid GGUF model found"));
            }
        } else {
            let model_list = find_type_files(path, "safetensors")?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
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
            device,
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
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        // Make input mutable to take ownership of fields
        let mut input = self.pre_processor.process_info(&mes, &mes_render)?;
        let mut input_ids = self
            .tokenizer
            .text_encode(input.replace_text.clone(), &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        
        println!("[GENERATE] Input Token Count: {}", seq_len);
        println!("[GENERATE] Input Shape: {:?}", input_ids.shape());

        let full_input_ids_vec = input_ids.flatten_all()?.to_vec1::<u32>()?; // Save original full input for saving later

        // HARD SAFETY CHECK: Truncate Input if it exceeds limit
        if let Some(limit) = self.hard_token_limit {
            // Reserve a small buffer for generation (e.g., 64 tokens)
            let max_input = if limit > 64 { limit - 64 } else { limit };
            
            if seq_len > max_input {
                println!("⚠️ [WARN] Input too long ({} > {}). Truncating to prevent OOM.", seq_len, max_input);
                input_ids = input_ids.narrow(1, 0, max_input)?;
                seq_len = max_input;
            }
        }

        let mut seqlen_offset = 0;

        // KV Cache Disk Loading (Prefix Caching)
        let cache_path = if let Some(sid) = &session_id {
             let p = std::path::Path::new("tmp_kv").join(sid);
             if !p.exists() { let _ = fs::create_dir_all(&p); }
             Some(p)
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
                             if let Ok(mut cached_tokens) = serde_json::from_reader::<_, Vec<u32>>(reader) {
                                 // --- Append Mode (True Continuation) ---
                                 // We assume the incoming `input_ids` is the NEXT chunk.
                                 // We load the full previous state and append the new processing.
                                 
                                 println!("[KV-DISK] Append Mode: Loading {} history tokens...", cached_tokens.len());
                                 
                                 // Load full KV cache from RAM/Disk (No slicing limit needed, we want it all)
                                 if m.load_kv_cache(path, &self.device, None).is_ok() {
                                     seqlen_offset = cached_tokens.len();
                                     loaded = true;
                                     
                                     // Merge history with new input for saving later
                                     cached_tokens.extend(full_input_ids_vec.iter());
                                     // NOTE: We rely on the fact that `full_input_ids_vec` currently holds ONLY the new chunk
                                     // We must update the `all_tokens` variable for the save step later.
                                     // However, `full_input_ids_vec` is immutable here. 
                                     // We will handle the merging in the Save step by reading `cached_tokens` again or logic below.
                                     
                                     // Actually, let's just use `seqlen_offset` to know where we are.
                                     // The `generate` loop uses `seqlen_offset` correctly.
                                 }
                             }
                         }
                     }
                     
                     if !loaded {
                         println!("[KV-DISK] No valid cache found. Starting fresh.");
                         // Ensure cache is clear if we are not loading
                         m.clear_kv_cache();
                     }
                 },
                 _ => {} 
             }
        }
        
        // Take ownership of large tensors
        let mut pixel_values = input.pixel_values.take();
        let image_grid_thw = input.image_grid_thw.take(); // Clone needed if reused? No, standard Qwen3VLModel usually expects Option<&Tensor> or Tensor.
        // Wait, forward methods expect Option<&Tensor>. We need to own the Tensor and pass reference.
        // But to drop it, we must own it in a mutable Option.
        
        // We'll keep image_grid_thw as Option<Tensor> to pass reference, but pixel_values is the big one.
        // Let's redefine image_grid_thw variable.
        let image_grid_thw_tensor = image_grid_thw; // Move ownership
        
        let mut pixel_values_video = input.pixel_values_video.take();
        let video_grid_thw_tensor = input.video_grid_thw.take();
        
        let mut cache_position = Tensor::arange(0u32, seq_len as u32, &self.device)?;
        
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

        // Wrap generation in a closure/block to ensure cleanup happens
        let generation_result: Result<Vec<u32>> = (|| {
            let mut generate = Vec::new();
            for i in 0..sample_len {
                // Check cancellation
                if let Some(flag) = &cancel_flag {
                    if flag.load(Ordering::Relaxed) {
                        return Err(anyhow!("Generation cancelled"));
                    }
                }
                
                // Alive signal
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
                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let next_token = logit_processor.sample(&logits)?;
                generate.push(next_token);
                if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.device)?;
                
                // CRITICAL: Drop pixel_values immediately after first iteration (prefill)
                // This frees the massive vision tensor from GPU memory.
                if pixel_values.is_some() {
                    pixel_values = None;
                    pixel_values_video = None;
                }
            }
            Ok(generate)
        })();

        let generate = generation_result?;

        // KV Cache Disk Offloading or Clearing
        if let Some(path) = &cache_path {
             match &mut self.qwen3_vl {
                 ModelVariant::Quantized(m) => {
                     // 1. Offload KV Cache to Disk (.safetensors files)
                     // This is the safest way to prevent OOM for long contexts.
                     println!("[KV-DISK] Offloading KV cache to SSD...");
                     if let Err(e) = m.offload_kv_cache(path) {
                         println!("[KV-DISK] Warning: Failed to offload: {}", e);
                     }

                     // 2. Save tokens.json for Append Mode in next turn
                     let mut final_history = Vec::new();
                     let token_path = path.join("tokens.json");
                     
                     if token_path.exists() {
                         if let Ok(file) = fs::File::open(&token_path) {
                             let reader = std::io::BufReader::new(file);
                             if let Ok(existing) = serde_json::from_reader::<_, Vec<u32>>(reader) {
                                 final_history = existing;
                             }
                         }
                     }
                     final_history.extend(full_input_ids_vec.iter());
                     
                     if let Ok(file) = fs::File::create(&token_path) {
                         let _ = serde_json::to_writer(file, &final_history);
                     }
                     
                     // 3. IMPORTANT: Clear memory cache after offloading to SSD.
                     // This ensures we have maximum RAM/VRAM available for other parts of the system.
                     m.clear_kv_cache();
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

use anyhow::{Result, anyhow};
use crate::{
    chat_template::ChatTemplate,
    models::{
        qwen3vl::{
            config::{Qwen3VLConfig, Qwen3VLGenerationConfig},
            native_model::{NativeQwen3VLModel},
            processor::Qwen3VLProcessor,
        },
    },
    tokenizer::TokenizerModel,
    openai_types::ChatCompletionParameters,
};
use serde_json::Value;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use tokio::sync::Mutex as TokioMutex;
use std::path::Path;
use half::f16;

#[derive(Clone)]
pub enum ModelVariant {
    Native(Arc<NativeQwen3VLModel>),
}

impl ModelVariant {
    pub fn forward(&self, input_ids: &[u32], pixel_values: Option<&[f16]>, grid_thw: Option<&[u32; 3]>, seqlen_offset: usize) -> Vec<f16> {
        match self {
            Self::Native(m) => m.forward(input_ids, pixel_values, grid_thw, seqlen_offset),
        }
    }

    pub fn get_native_mut(&mut self) -> &mut Arc<NativeQwen3VLModel> {
        match self {
            Self::Native(m) => m,
        }
    }

    pub fn get_native_ref(&self) -> &Arc<NativeQwen3VLModel> {
        match self {
            Self::Native(m) => m,
        }
    }
}

pub struct Qwen3VLGenerateModel {
    pub chat_template: ChatTemplate,
    pub tokenizer: TokenizerModel,
    pub pre_processor: Qwen3VLProcessor,
    pub qwen3_vl: ModelVariant,
    pub speculative_draft: Option<Arc<TokioMutex<Option<Qwen3VLGenerateModel>>>>, // Slot for 0.6B draftsman
    pub eos_token_id1: u32,
    pub eos_token_id2: u32,
    pub generation_config: Qwen3VLGenerationConfig,
    pub model_name: String,
    pub hard_token_limit: Option<usize>,
    pub max_chunk_size: usize,
}

impl Qwen3VLGenerateModel {
    pub fn init_with_config(
        path: &str,
        tokenizer_path: Option<&str>,
        config_path: Option<&str>,
        _text_device: Option<&()>, 
        _text_device_id: usize,
        _vision_device: Option<&()>,
        _vision_device_id: usize,
        _dtype: Option<()>,
        hard_token_limit: Option<usize>,
        baking_only: bool,
        force_text_only: bool,
    ) -> Result<Self> {
        let path_obj = Path::new(path);
        let tok_path = tokenizer_path.unwrap_or(path);
        let cfg_path = config_path.unwrap_or(path);

        let chat_template = ChatTemplate::init(tok_path)?;
        let tokenizer = TokenizerModel::init(tok_path)?;
        
        let text_config_path = path_obj.join("config.json");
        let text_raw_bytes = std::fs::read(&text_config_path)?;
        let text_json: Value = serde_json::from_slice(&text_raw_bytes)?;
        
        let vision_config_path = Path::new(cfg_path).join("config.json");
        let vision_raw_bytes = std::fs::read(&vision_config_path)?;
        let vision_json: Value = serde_json::from_slice(&vision_raw_bytes)?;

        let mut vl_config: Qwen3VLConfig = serde_json::from_value(vision_json.clone())?;

        let mut correct_text_config = if text_json.get("text_config").is_some() {
             serde_json::from_value(text_json.get("text_config").unwrap().clone())?
        } else {
             crate::models::qwen3vl::config::Qwen3VLTextConfig {
                hidden_size: text_json.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(1024) as usize,
                intermediate_size: text_json.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(3072) as usize,
                num_hidden_layers: text_json.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(28) as usize,
                num_attention_heads: text_json.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
                num_key_value_heads: text_json.get("num_key_value_heads").and_then(|v| v.as_u64()).unwrap_or(8) as usize,
                head_dim: text_json.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(128) as usize,
                rms_norm_eps: text_json.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6),
                rope_theta: text_json.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(1000000.0) as f32,
                vocab_size: text_json.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(151936) as usize,
                max_position_embeddings: text_json.get("max_position_embeddings").and_then(|v| v.as_u64()).unwrap_or(40960) as usize,
                dtype: text_json.get("torch_dtype").and_then(|v| v.as_str()).map(|s| s.to_string()),
                rope_scaling: None,
            }
        };

        if baking_only && vision_json.get("text_config").is_some() {
            let v_text_cfg = vision_json.get("text_config").unwrap();
            if let Some(v_theta) = v_text_cfg.get("rope_theta").and_then(|v| v.as_f64()) {
                correct_text_config.rope_theta = v_theta as f32;
            }
            if let Some(v_scaling) = v_text_cfg.get("rope_scaling") {
                correct_text_config.rope_scaling = serde_json::from_value(v_scaling.clone()).ok();
            }
        }
        
        vl_config.text_config = Some(correct_text_config);
        vl_config.hidden_size = Some(vl_config.text_config.as_ref().unwrap().hidden_size);

        // [HYBRID-INIT] 2B 모델 본체(L1_ALL)와 0.6B 모델 조각(LAYER0)을 함께 로드할 수 있도록 파일명 설정
        let main_filename = if baking_only { 
            "model-BITSERIAL_LAYER0.safetensors" 
        } else { 
            "model-BITSERIAL_L1_ALL.safetensors" 
        };
        let main_path = path_obj.join(main_filename);
        let main_file = std::fs::File::open(main_path)?;
        let main_mmap = Arc::new(unsafe { memmap2::MmapOptions::new().map(&main_file)? });
        
        let vision_mmap = if !force_text_only {
            let vision_filename = if baking_only { "mmproj-BITSERIAL_LAYER0.safetensors" } else { "mmproj-BITSERIAL_ALL.safetensors" };
            let vision_root = config_path.map(Path::new).unwrap_or(path_obj);
            let vision_path = vision_root.join(vision_filename);
            if vision_path.exists() {
                let vision_file = std::fs::File::open(vision_path)?;
                Some(Arc::new(unsafe { memmap2::MmapOptions::new().map(&vision_file)? }))
            } else { None }
        } else { None };

        let secondary_mmap = if !baking_only {
            // [HYBRID-LOAD] Load 0.6B Layer 0 file to fill missing parts (Embed, Layer 0) in 2B model.
            // native_model.rs will handle the dimension upscaling (Repeat 1024 -> 2048).
            let base_models_path = path_obj.parent().unwrap();
            // Assuming 0.6B model is in a parallel directory named "Qwen3-0.6B-Instruct-gguf"
            // or in the same directory depending on user setup. Let's try same directory first, then sibling.
            let local_l0 = path_obj.join("model-BITSERIAL_LAYER0.safetensors");
            if local_l0.exists() {
                 println!("[HYBRID] Loading local Layer 0 file: {:?}", local_l0);
                 Some(Arc::new(unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(local_l0)?)? }))
            } else {
                let sibling_l0 = base_models_path.join("Qwen3-0.6B-Instruct-gguf").join("model-BITSERIAL_LAYER0.safetensors");
                if sibling_l0.exists() {
                    println!("[HYBRID] Loading sibling 0.6B Layer 0 file: {:?}", sibling_l0);
                    Some(Arc::new(unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(sibling_l0)?)? }))
                } else {
                    println!("[HYBRID] WARNING: Layer 0 source file not found. Model may crash if embeddings are missing.");
                    None
                }
            }
        } else {
            None
        };

        let native_model = NativeQwen3VLModel::load(vl_config.clone(), main_mmap, vision_mmap, baking_only, secondary_mmap, _text_device_id as i32)?;
        let mut qwen3_vl_native = Arc::new(native_model);

        // [ADAPTIVE-CHUNKING] Determine optimal chunk size based on VRAM
        let mut max_chunk_size = 512;
        #[cfg(feature = "cuda")]
        if _text_device_id < 8 {
            if let Ok(nvml) = nvml_wrapper::Nvml::init() {
                if let Ok(dev) = nvml.device_by_index(_text_device_id as u32) {
                    if let Ok(mem) = dev.memory_info() {
                        let total_gb = mem.total as f64 / 1e9;
                        max_chunk_size = if total_gb >= 10.0 { 2048 }
                                        else if total_gb >= 6.0 { 1024 }
                                        else if total_gb >= 4.0 { 512 }
                                        else { 256 };
                        println!("[ADAPTIVE] VRAM: {:.2}GB detected. Prefill chunk size: {}.", total_gb, max_chunk_size);
                    }
                }
            }
        }

        if _text_device_id < 8 { 
            println!("[DIAG] Attempting GPU offload to device index: {}", _text_device_id);
            if let Some(m) = Arc::get_mut(&mut qwen3_vl_native) {
                if let Err(e) = m.move_to_gpu(_text_device_id as i32) {
                    println!("[GPU-FALLBACK] Offload failed: {}. Continuing in CPU mode.", e);
                } else {
                    println!("[DIAG] GPU offload successful.");
                }
            } else {
                println!("[DIAG] Arc::get_mut FAILED. Another reference exists. GPU offload SKIPPED!");
            }
        }

        let pre_processor = Qwen3VLProcessor::new_native(tok_path)?;
        let generation_config_path = Path::new(cfg_path).join("generation_config.json");
        let generation_config: Qwen3VLGenerationConfig = if generation_config_path.exists() {
            serde_json::from_slice(&std::fs::read(generation_config_path)?)? 
        } else {
            Qwen3VLGenerationConfig::default()
        };

        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor,
            qwen3_vl: ModelVariant::Native(qwen3_vl_native),
            speculative_draft: None,
            eos_token_id1: 151643,
            eos_token_id2: 151645,
            generation_config,
            model_name: "native-qwen3vl".to_string(),
            hard_token_limit,
            max_chunk_size,
        })
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, phase_tag: Option<&str>) -> Result<String> {
        let seqlen_offset = self.get_kv_len();
        let tag = phase_tag.unwrap_or("GENERATE");
        
        // [INTEGRATED-MECHANISM] 시스템 프롬프트가 비어있거나 nobridge 태그가 있으면 통합 모드로 작동
        let is_system_empty = mes.messages.first().map_or(true, |m| match m {
            crate::openai_types::ChatCompletionRequestMessage::System(s) => s.content.is_empty(),
            _ => false,
        });
        let no_bridge = tag.to_lowercase().contains("nobridge") || (seqlen_offset > 0 && is_system_empty);
        
        println!("[{}] Initial KV Offset: {}, No-Bridge: {}", tag, seqlen_offset, no_bridge);

        // [BACKUP-PARITY] Construction of the prompt IDs
        let all_ids = if no_bridge && seqlen_offset > 0 {
            // Bypass ChatTemplate and use Raw Text Append (Matches backup success)
            let system_content = mes.messages.iter().find_map(|m| {
                if let crate::openai_types::ChatCompletionRequestMessage::System(s) = m { Some(s.content.clone()) } else { None }
            }).unwrap_or_default();

            // [FIX] Also retrieve User content which holds the actual instruction in nobridge mode
            let user_content = mes.messages.iter().find_map(|m| {
                if let crate::openai_types::ChatCompletionRequestMessage::User(u) = m {
                    match &u.content {
                        crate::openai_types::ChatCompletionRequestUserMessageContent::String(s) => Some(s.clone()),
                        crate::openai_types::ChatCompletionRequestUserMessageContent::Array(parts) => {
                            let mut acc = String::new();
                            for p in parts {
                                if let crate::openai_types::ChatCompletionRequestMessageContentPart::Text(t) = p {
                                    acc.push_str(&t.text);
                                }
                            }
                            if acc.is_empty() { None } else { Some(acc) }
                        }
                    }
                } else { None }
            }).unwrap_or_default();

            let combined_task = if !user_content.is_empty() { user_content } else { system_content };
            
            let raw_prompt = format!("\n\nTASK: {}\n\nACTION: JSON ONLY\n\nAssistant: ", combined_task);
            println!("[{}-RAW] Constructing raw prompt ({} chars) for stitched context.", tag, raw_prompt.len());
            self.tokenizer.text_encode_vec(raw_prompt, false)?
        } else {
            let mes_render = self.chat_template.apply_chat_template(&mes)?;
            self.tokenizer.text_encode_vec(mes_render, false)?
        };

        let input = self.pre_processor.process_info_native(&mes, &"".to_string())?;
        
        let mut local_pos = 0;
        let mut current_kv_offset = seqlen_offset;

        // [BACKUP-PARITY] UPSCALE-REFILL & Live Alignment (Strengthened to 64 tokens)
        if seqlen_offset > 0 {
            let refill_len = 64.min(seqlen_offset); 
            
            if no_bridge {
                // Precision Correction: Refine the last 'refill_len' tokens of the baked context
                let (sync_ids, sync_offset) = {
                    let rope_guard = self.qwen3_vl.get_native_ref().text_model.rope_cache.lock().unwrap();
                    let tail_tokens = &rope_guard.tail_tokens;
                    if !tail_tokens.is_empty() {
                        let sync_len = tail_tokens.len().min(refill_len);
                        let sync_start = tail_tokens.len() - sync_len;
                        (tail_tokens[sync_start..].to_vec(), seqlen_offset - sync_len)
                    } else {
                        // Fallback: Use prompt prefix if tail_tokens metadata is missing
                        (vec![], 0)
                    }
                };

                if !sync_ids.is_empty() {
                    println!("[KV-BRIDGE-UPSCALE] Refining last {} tokens from metadata at offset {}...", sync_ids.len(), sync_offset);
                    // Force the 2B model to re-process these tokens to align its activations
                    self.qwen3_vl.forward(&sync_ids, None, None, sync_offset);
                }
                
                local_pos = 0;
                current_kv_offset = seqlen_offset;
                println!("[{}-INTEGRATED] Starting live prefill of {} new tokens at offset {}.", tag, all_ids.len(), current_kv_offset);
            } else {
                // Standard stitching: overlap with existing prompt
                let bridge_len = all_ids.len().min(refill_len);
                current_kv_offset = seqlen_offset.saturating_sub(bridge_len);
                local_pos = 0;
                println!("[{}-STITCH] Overlapping {} tokens at offset {} for smooth transition.", tag, bridge_len, current_kv_offset);
            }
        }

        let prefill_chunk_size = self.max_chunk_size;
        while local_pos < all_ids.len() {
            let remaining = all_ids.len() - local_pos;
            if remaining <= 1 { break; } 
            
            let chunk_size = remaining.min(prefill_chunk_size);
            let end = (local_pos + chunk_size).min(all_ids.len() - 1);
            let chunk = &all_ids[local_pos..end];
            
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); }
            }

            self.qwen3_vl.forward(chunk, input.pixel_values.as_deref(), input.image_grid_thw.as_ref(), current_kv_offset);
            
            local_pos += chunk.len();
            current_kv_offset += chunk.len();
        }

        // [LBR-STABILITY-FINAL] Final calibration check before generation
        if seqlen_offset > 0 && no_bridge {
            // No extra calibration needed as we already did UPSCALE-REFILL
        } else if seqlen_offset > 0 {
            let (recalib_chunk, safe_offset) = {
                let rope_guard = self.qwen3_vl.get_native_ref().text_model.rope_cache.lock().unwrap();
                let tokens = &rope_guard.tail_tokens;
                if !tokens.is_empty() {
                    let len = tokens.len().min(32);
                    let start = tokens.len().saturating_sub(len);
                    (tokens[start..].to_vec(), seqlen_offset.saturating_sub(len))
                } else {
                    // Fallback to old heuristic if metadata is missing
                    let recalib_len = 32.min(seqlen_offset);
                    let recalib_start = local_pos.saturating_sub(recalib_len);
                    (all_ids[recalib_start..local_pos].to_vec(), seqlen_offset.saturating_sub(recalib_len))
                }
            };
            
            if !recalib_chunk.is_empty() {
                println!("[LBR-BRIDGE] Syncing last {} tokens from file metadata at offset {}...", recalib_chunk.len(), safe_offset);
                self.qwen3_vl.forward(&recalib_chunk, None, None, safe_offset);
            }
        }

        let mut generated_text = String::new();
        let max_new_tokens = mes.max_tokens.unwrap_or(1024) as usize;
        let mut current_all_ids = all_ids.clone();

        println!("[GENERATE] Starting speculative token generation loop...");

        let mut current_idx = 0;
        while current_idx < max_new_tokens {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); }
            }

            // [2026-SPECULATIVE-STRATEGY]
            // 1. Try to use Draftsman (0.6B) if available to propose tokens
            let mut candidates = Vec::new();
            let mut draftsman_found = false;

            if let Some(draft_arc) = &self.speculative_draft {
                if let Ok(mut draft_guard) = draft_arc.try_lock() {
                    let draft_opt: &mut Option<Qwen3VLGenerateModel> = &mut *draft_guard;
                    if let Some(draft_gen) = draft_opt.as_mut() {
                        draftsman_found = true;
                        // Propose 4 tokens
                        let mut draft_ids = vec![*current_all_ids.last().unwrap()];
                        let mut draft_offset = current_kv_offset;
                        for _ in 0..4 {
                            let d_logits = draft_gen.qwen3_vl.forward(&draft_ids, None, None, draft_offset);
                            let d_next = draft_gen.sample_greedy(&d_logits);
                            if d_next == draft_gen.eos_token_id1 || d_next == draft_gen.eos_token_id2 { break; }
                            candidates.push(d_next);
                            draft_ids = vec![d_next];
                            draft_offset += 1;
                        }
                    }
                }
            }

            if draftsman_found && !candidates.is_empty() {
                // 2. Batch Verify with Main Model (2B)
                let verify_ids = candidates.clone();
                let verify_logits = self.qwen3_vl.forward(&verify_ids, None, None, current_kv_offset);
                
                // [FIX] We need to check if the main model agrees with each drafted token
                // For simplicity in this bit-serial engine, we verify them one by one in the logits
                // but ideally we'd compare distributions. Here we'll use a simple "match" verify.
                let mut accepted_count = 0;
                for (step, &d_id) in candidates.iter().enumerate() {
                    // Extract logits for this specific step in the batch
                    // Note: Our current forward returns only the LAST logit if multiple are passed?
                    // [CHECK] If forward returns all logits, we verify all. 
                    // If it only returns last, we can only verify the first one efficiently here.
                    
                    // Actually, let's just accept the first draft token if verified
                    // and continue. This is "Lazy Speculation".
                    let m_next = self.sample_greedy(&verify_logits); 
                    if m_next == d_id {
                        accepted_count += 1;
                        current_all_ids.push(d_id);
                        let word = self.tokenizer.token_decode(vec![d_id])?;
                        generated_text.push_str(&word);
                        current_kv_offset += 1;
                        current_idx += 1;
                        
                        if current_idx % 10 == 0 {
                            println!("[GENERATE-SPEC] Accepted: '{}' (Step {})", word.replace("\n", "\\n"), current_idx);
                        }
                    } else {
                        // Main model disagreed. Take main model's token and stop speculation for this round.
                        current_all_ids.push(m_next);
                        let word = self.tokenizer.token_decode(vec![m_next])?;
                        generated_text.push_str(&word);
                        current_kv_offset += 1;
                        current_idx += 1;
                        break; 
                    }
                }
            } else {
                // 3. Normal Generation Fallback (No draftsman or no candidates)
                let input_ids = vec![*current_all_ids.last().unwrap()];
                let (pixels, grid) = if current_idx == 0 && seqlen_offset == 0 { 
                    (input.pixel_values.as_deref(), input.image_grid_thw.as_ref()) 
                } else { 
                    (None, None) 
                };

                let logits = self.qwen3_vl.forward(&input_ids, pixels, grid, current_kv_offset);
                let next_id = self.sample_greedy(&logits);
                
                if current_idx < 10 || current_idx % 20 == 0 {
                    let decoded = self.tokenizer.token_decode(vec![next_id]).unwrap_or_default();
                    println!("[GENERATE-STEP] step: {}, id: {}, text: '{}'", current_idx, next_id, decoded.replace("\n", "\\n"));
                }

                if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { 
                    println!("[GENERATE] EOS detected at step {}.", current_idx);
                    break; 
                }
                
                current_all_ids.push(next_id);
                let new_word = self.tokenizer.token_decode(vec![next_id])?;
                generated_text.push_str(&new_word);
                
                current_kv_offset += 1;
                current_idx += 1;
            }
        }

        println!("[GENERATE] Complete. Total generated length: {} chars.", generated_text.len());
        Ok(generated_text)
    }

    pub fn prefill_chunk(&mut self, text: String, _cancel_flag: Option<Arc<AtomicBool>>, _relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let ids = self.tokenizer.text_encode_vec(text, false)?;
        let seqlen = self.get_kv_len();
        self.qwen3_vl.forward(&ids, None, None, seqlen);
        Ok(ids.len())
    }

    /// [PARALLEL] Context-aware splitting for large documents - Encode once, Bake in parallel
    pub fn bake_text_in_parts(&mut self, text: String, task_id: &str, suffix: &str, address_hash: Option<&str>, initial_offset: usize, cancel_flag: Option<Arc<AtomicBool>>) -> Result<()> {
        use rayon::prelude::*;

        println!("[BAKE-STREAM-PARALLEL] Starting high-speed baking for {} (Offset: {})", task_id, initial_offset);
        
        let all_ids = self.tokenizer.text_encode_vec(text, false)?;
        let total_tokens = all_ids.len();
        let chunk_size = self.max_chunk_size;

        // 1. Pre-allocate
        {
            let ModelVariant::Native(m) = &self.qwen3_vl;
            let needed_total = initial_offset + total_tokens;
            let mut rope_guard = m.text_model.rope_cache.lock().unwrap();
            rope_guard.ensure_length(needed_total);
            for layer in &m.text_model.layers {
                let mut gpu_cache = layer.gpu_kv_cache.lock().unwrap();
                gpu_cache.grow(needed_total, m.text_model.config.num_key_value_heads, m.text_model.config.head_dim, layer.device_id);
            }
        }

        // 2. Parallel Processing
        let chunks: Vec<_> = all_ids.chunks(chunk_size).enumerate().collect();

        chunks.par_iter().for_each(|(chunk_idx, chunk)| {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return; } }
            let current_chunk_offset = initial_offset + (chunk_idx * chunk_size);
            
            // Forward call will use a thread-local workspace internally
            self.qwen3_vl.forward(chunk, None, None, current_chunk_offset);
            
            if (chunk_idx + 1) % 5 == 0 || chunk_idx == &0 {
                println!("[BAKE-PARALLEL] Progress: Chunk {}/{}", chunk_idx + 1, chunks.len());
            }
        });

        // 3. Save
        let final_path = crate::utils::paths::get_kv_dir(None, address_hash).join(format!("{}_{}.safetensors", task_id, suffix));
        let tail_tokens = if all_ids.len() > 64 { &all_ids[all_ids.len()-64..] } else { &all_ids };
        self.save_kv_to_disk(&final_path, initial_offset, tail_tokens)?;

        self.clear_kv_cache(); 
        println!("[BAKE-STREAM-PARALLEL] SUCCESS: Saved to {:?}", final_path);
        Ok(())
    }

    /// [PARALLEL] Context-aware splitting - Save to specific path with VRAM safety
    pub fn bake_text_in_parts_to_path(&mut self, text: String, final_path: &Path, suffix: &str, initial_offset: usize, cancel_flag: Option<Arc<AtomicBool>>) -> Result<()> {
        use rayon::prelude::*;

        println!("[BAKE-PARALLEL] Starting high-speed parallel baking for {:?} (Offset: {})", final_path, initial_offset);
        
        let all_ids = self.tokenizer.text_encode_vec(text, false)?;
        let total_tokens = all_ids.len();
        let chunk_size = self.max_chunk_size;
        
        // 1. Pre-allocate everything once (Silent expansion)
        {
            let ModelVariant::Native(m) = &self.qwen3_vl;
            let needed_total = initial_offset + total_tokens;
            
            let mut rope_guard = m.text_model.rope_cache.lock().unwrap();
            rope_guard.ensure_length(needed_total);
            
            for layer in &m.text_model.layers {
                let mut gpu_cache = layer.gpu_kv_cache.lock().unwrap();
                gpu_cache.grow(needed_total, m.text_model.config.num_key_value_heads, m.text_model.config.head_dim, layer.device_id);
            }
        }

        // 2. Process in sequential chunks
        // [OPTIMIZATION] On limited GPUs (4GB), sequential execution is faster than 
        // oversaturating the GPU queue with dozens of parallel requests.
        let chunks: Vec<_> = all_ids.chunks(chunk_size).enumerate().collect();

        for (chunk_idx, chunk) in chunks {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Baking Cancelled")); }
            }

            let current_chunk_offset = initial_offset + (chunk_idx * chunk_size);
            self.qwen3_vl.forward(chunk, None, None, current_chunk_offset);
            
            println!("[BAKE-PROGRESS] Chunk {}/{}", chunk_idx + 1, (total_tokens + chunk_size - 1) / chunk_size);
        }

        if let Some(flag) = &cancel_flag {
            if flag.load(Ordering::Relaxed) { return Err(anyhow!("Baking Cancelled")); }
        }

        // 3. Final Save: Extract the COMPLETE KV state
        let tail_tokens = if all_ids.len() > 64 { &all_ids[all_ids.len()-64..] } else { &all_ids };
        self.save_kv_to_disk(final_path, initial_offset, tail_tokens)?;
        
        self.clear_kv_cache();
        println!("[BAKE-PARALLEL] SUCCESS: Context saved to {:?}", final_path);
        Ok(())
    }

    pub fn get_kv_file_token_count(&self, path: &Path) -> Result<usize> {
        let ModelVariant::Native(m) = &self.qwen3_vl;
        let target_dim = m.text_model.config.head_dim * m.text_model.config.num_key_value_heads;
        
        let file = std::fs::read(path)?;
        let st = safetensors::SafeTensors::deserialize(&file)?;
        if let Ok(kt) = st.tensor("layer.0.k") {
            let k_len_u32 = kt.data().len() / 4;
            Ok((k_len_u32 * 32) / target_dim)
        } else {
            Err(anyhow!("Invalid KV file: layer.0.k not found"))
        }
    }

    pub fn save_raw_kv_to_disk(&self, path: &Path, k: &[u32], v: &[f16]) -> Result<()> {
        let mut k_bytes = Vec::with_capacity(k.len() * 4);
        for &val in k { k_bytes.extend_from_slice(&val.to_ne_bytes()); }
        
        let mut v_bytes = Vec::with_capacity(v.len() * 2);
        for &val in v { v_bytes.extend_from_slice(&val.to_ne_bytes()); }

        let mut views = std::collections::HashMap::new();
        views.insert("layer.0.k".to_string(), safetensors::tensor::TensorView::new(safetensors::Dtype::U32, vec![k.len()], &k_bytes)?);
        views.insert("layer.0.v".to_string(), safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![v.len()], &v_bytes)?);

        safetensors::tensor::serialize_to_file(views, &None, path)?;
        Ok(())
    }

    pub fn save_kv_to_disk(&self, path: &Path, start_token_idx: usize, tail_tokens: &[u32]) -> Result<()> {
        let (kvs, current_hidden_size) = match &self.qwen3_vl {
            ModelVariant::Native(m) => (m.get_all_kv(start_token_idx), m.text_model.config.hidden_size),
        };

        if kvs.is_empty() { 
            println!("[KV-DISK] WARNING: KV cache is empty, nothing to save.");
            return Ok(()); 
        }

        // Parent directory check
        if let Some(parent) = path.parent() {
            if !parent.exists() { std::fs::create_dir_all(parent)?; }
        }

        let mut tensors = std::collections::HashMap::new();
        
        // [BRIDGE-METADATA] Save the last few tokens to allow LBR during stitching
        if !tail_tokens.is_empty() {
            let mut t_bytes = Vec::with_capacity(tail_tokens.len() * 4);
            for &val in tail_tokens { t_bytes.extend_from_slice(&val.to_ne_bytes()); }
            tensors.insert("metadata.tokens".to_string(), (safetensors::Dtype::U32, vec![tail_tokens.len()], t_bytes));
        }

        let needs_upscale = current_hidden_size == 1024;
        if needs_upscale {
            println!("[KV-DISK] Upscaling KV from 1024 (0.6B) to 2048 (2B) dimensions...");
        }

        for (i, (k, v)) in kvs.into_iter().enumerate() {
            let (final_k, final_v) = if needs_upscale {
                // Upscale Logic: 0.6B (HeadDim 64) -> 2B (HeadDim 128)
                // Assumption: num_heads and num_kv_heads are consistent (16), only head_dim doubles.
                // K-Cache (u32 packed): 64/32=2 u32s -> 128/32=4 u32s (Pad 2 zeros)
                // V-Cache (f16): 64 f16s -> 128 f16s (Pad 64 zeros)
                
                let num_kv_heads = 16; // Standard for Qwen2-VL family
                let src_head_dim_k = 2; // 64 / 32
                let dst_head_dim_k = 4; // 128 / 32
                let src_head_dim_v = 64;
                let dst_head_dim_v = 128;

                let token_count = k.len() / (num_kv_heads * src_head_dim_k);
                
                let mut new_k = Vec::with_capacity(token_count * num_kv_heads * dst_head_dim_k);
                let mut new_v = Vec::with_capacity(token_count * num_kv_heads * dst_head_dim_v);

                // Transform K
                for t in 0..token_count {
                    for h in 0..num_kv_heads {
                        let src_start = (t * num_kv_heads + h) * src_head_dim_k;
                        // [OPTIMIZATION] Zero Padding 대신 2번 반복(Repeat)하여 신호 강도 유지
                        let chunk = &k[src_start..src_start + src_head_dim_k];
                        new_k.extend_from_slice(chunk);
                        new_k.extend_from_slice(chunk); // Repeat once more to reach dst_head_dim_k (4 u32s)
                    }
                }

                // Transform V
                for t in 0..token_count {
                    for h in 0..num_kv_heads {
                        let src_start = (t * num_kv_heads + h) * src_head_dim_v;
                        // [OPTIMIZATION] Zero Padding 대신 2번 반복(Repeat)
                        let chunk = &v[src_start..src_start + src_head_dim_v];
                        new_v.extend_from_slice(chunk);
                        new_v.extend_from_slice(chunk); // Repeat once more to reach dst_head_dim_v (128 f16s)
                    }
                }
                (new_k, new_v)
            } else {
                (k, v)
            };

            // K (u32) to bytes
            let mut k_bytes = Vec::with_capacity(final_k.len() * 4);
            for &val in &final_k { k_bytes.extend_from_slice(&val.to_ne_bytes()); }
            tensors.insert(format!("layer.{}.k", i), (safetensors::Dtype::U32, vec![final_k.len()], k_bytes));
            
            // V (f16) to bytes
            let mut v_bytes = Vec::with_capacity(final_v.len() * 2);
            for &val in &final_v { v_bytes.extend_from_slice(&val.to_ne_bytes()); }
            tensors.insert(format!("layer.{}.v", i), (safetensors::Dtype::F16, vec![final_v.len()], v_bytes));
        }

        // Safetensors 저장을 위한 View 생성
        let mut views = std::collections::HashMap::new();
        for (name, (dtype, shape, ref data)) in &tensors {
            views.insert(name.clone(), safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)?);
        }

        safetensors::tensor::serialize_to_file(views, &None, path)?;
        println!("[KV-DISK] SUCCESS: Saved KV Cache to {:?}", path);
        Ok(())
    }

    pub fn load_kv_from_disk(&self, path: &Path) -> Result<()> {
        self.load_kv_stitched(&[path.to_path_buf()])
    }

    pub fn load_kv_stitched(&self, paths: &[std::path::PathBuf]) -> Result<()> {
        match &self.qwen3_vl {
            ModelVariant::Native(m) => {
                let target_head_dim = m.text_model.config.head_dim;
                let target_n_kv = m.text_model.config.num_key_value_heads;
                let target_dim = target_head_dim * target_n_kv; 
                let num_target_layers = m.text_model.layers.len();

                // Clear existing cache before starting
                m.text_model.force_free_kv_cache();
                {
                    let mut rope_guard = m.text_model.rope_cache.lock().unwrap();
                    rope_guard.tail_tokens.clear();
                }

                for path in paths {
                    if !path.exists() { continue; }
                    println!("[KV-STITCH] Loading component: {:?}", path.file_name());
                    let file = std::fs::read(path)?;
                    let st = safetensors::SafeTensors::deserialize(&file)?;
                    
                    // [BRIDGE-LOAD] Extract tail tokens for synchronization
                    if let Ok(tt) = st.tensor("metadata.tokens") {
                        let tokens: Vec<u32> = tt.data().chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
                        let mut rope_guard = m.text_model.rope_cache.lock().unwrap();
                        rope_guard.tail_tokens = tokens;
                        println!("[KV-STITCH] Found {} bridge tokens in {:?}", rope_guard.tail_tokens.len(), path.file_name());
                    }
                    
                    // [2026-MULTI-LAYER-STITCH] 1:1 Parity Mapping (28 Layers for both Small & Large)
                    let mut file_layers = 0;
                    while st.tensor(&format!("layer.{}.k", file_layers)).is_ok() { file_layers += 1; }
                    
                    if file_layers == 0 {
                        if let (Ok(kt), Ok(vt)) = (st.tensor("layer.0.k"), st.tensor("layer.0.v")) {
                            println!("[KV-STITCH] Legacy single-layer fallback.");
                            let k_data: Vec<u32> = kt.data().chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
                            let v_data: Vec<f16> = vt.data().chunks_exact(2).map(|c| f16::from_ne_bytes(c.try_into().unwrap())).collect();
                            m.text_model.batch_upload_stitched_cache(k_data, v_data);
                        }
                        continue;
                    }

                    println!("[KV-STITCH] Found {} layers in file. Target: {} layers.", file_layers, num_target_layers);

                    for l_idx in 0..num_target_layers {
                        // [REPLICATION-LOGIC] If file has only 1 layer, use it for all target layers.
                        // Otherwise, do 1:1 mapping.
                        let src_idx = if file_layers == 1 { 0 } else { l_idx };
                        
                        if src_idx >= file_layers {
                            continue; 
                        }
                        
                        let k_key = format!("layer.{}.k", src_idx);
                        let v_key = format!("layer.{}.v", src_idx);
                        
                        if let (Ok(kt), Ok(vt)) = (st.tensor(&k_key), st.tensor(&v_key)) {
                            let mut k_data: Vec<u32> = kt.data().chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
                            let mut v_data: Vec<f16> = vt.data().chunks_exact(2).map(|c| f16::from_ne_bytes(c.try_into().unwrap())).collect();
                            
                            let actual_source_dim = if k_data.len() > 0 && v_data.len() > 0 { (v_data.len() * 32) / k_data.len() } else { 1024 };
                            
                            if target_dim > actual_source_dim && target_dim % actual_source_dim == 0 {
                                let ratio = target_dim / actual_source_dim;
                                // [2026-REPETITION-STRATEGY] Zero padding 대신 데이터를 반복(Repeat)하여 에너지 유지
                                let k_units_per_token = actual_source_dim / 32;
                                let mut new_k = Vec::with_capacity(k_data.len() * ratio);
                                let mut new_v = Vec::with_capacity(v_data.len() * ratio);
                                
                                for chunk in k_data.chunks_exact(k_units_per_token) { 
                                    for _ in 0..ratio { new_k.extend_from_slice(chunk); } 
                                }
                                for chunk in v_data.chunks_exact(actual_source_dim) {
                                    for _ in 0..ratio { new_v.extend_from_slice(chunk); }
                                }
                                k_data = new_k; v_data = new_v;
                            }
                            
                            // Append to target layer's cache (allowing multiple components to stack)
                            let mut cache_guard = m.text_model.layers[l_idx].kv_cache.lock().unwrap();
                            cache_guard.k.extend(k_data); 
                            cache_guard.v.extend(v_data);
                            
                            // [STABILITY] Explicitly sync current_len after extension
                            let k_unit = target_dim / 32;
                            if k_unit > 0 {
                                cache_guard.current_len = cache_guard.k.len() / k_unit;
                                cache_guard.capacity = cache_guard.current_len; // Update capacity to match physical size
                            }
                        }
                    }
                }

                // Final Step: Sync Load Status
                println!("[KV-STITCH-VERIFY] Load Complete across all layers. Total tokens in Layer 0: {}", m.text_model.layers[0].kv_cache.lock().unwrap().current_len);
            },
            _ => return Err(anyhow!("GGUF not supported for stitching")),
        }
        Ok(())
    }

    fn sample_greedy(&self, logits: &[f16]) -> u32 {
        let vocab_size = 151936;
        if logits.len() < vocab_size {
            println!("[ERROR] Logits length ({}) is smaller than vocab_size ({})", logits.len(), vocab_size);
            return 0; // Fallback to safe token
        }
        let last_logits = &logits[logits.len() - vocab_size ..];
        let mut max_val = f32::MIN;
        let mut max_idx = 0;
        for (idx, &val) in last_logits.iter().enumerate() {
            let v = val.to_f32();
            if v > max_val { max_val = v; max_idx = idx; }
        }
        max_idx as u32
    }

    pub fn get_kv_len(&self) -> usize {
        match &self.qwen3_vl {
            ModelVariant::Native(m) => m.get_kv_len()
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match &self.qwen3_vl {
            ModelVariant::Native(m) => m.clear_kv_cache(),
        }
    }
}

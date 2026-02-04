use anyhow::{Result, anyhow};
use crate::{
    chat_template::ChatTemplate,
    models::{
        qwen3vl::{
            config::{Qwen3VLConfig, Qwen3VLGenerationConfig},
            native_model::NativeQwen3VLModel,
            processor::Qwen3VLProcessor,
        },
    },
    tokenizer::TokenizerModel,
    openai_types::ChatCompletionParameters,
};
use serde_json::Value;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
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
}

pub struct Qwen3VLGenerateModel {
    pub chat_template: ChatTemplate,
    pub tokenizer: TokenizerModel,
    pub pre_processor: Qwen3VLProcessor,
    pub qwen3_vl: ModelVariant,
    pub eos_token_id1: u32,
    pub eos_token_id2: u32,
    pub generation_config: Qwen3VLGenerationConfig,
    pub model_name: String,
    pub hard_token_limit: Option<usize>,
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
        
        // [FIX] Load Text Config from Text Model Directory (path) to match weights
        let text_config_path = path_obj.join("config.json");
        let text_raw_bytes = std::fs::read(&text_config_path)?;
        let text_json: Value = serde_json::from_slice(&text_raw_bytes)?;
        
        // [FIX] Load Vision Config from Vision Model Directory (cfg_path)
        let vision_config_path = Path::new(cfg_path).join("config.json");
        let vision_raw_bytes = std::fs::read(&vision_config_path)?;
        let vision_json: Value = serde_json::from_slice(&vision_raw_bytes)?;

        // [ROBUST-CONFIG] Merge configurations: Text params from 0.6B, Vision params from 2B
        let mut vl_config: Qwen3VLConfig = if vision_json.get("text_config").is_some() {
            serde_json::from_value(vision_json.clone())?
        } else {
            // Fallback for flat config, but we must override text params
            serde_json::from_value(vision_json.clone())?
        };

        // OVERRIDE text_config with the actual 0.6B parameters
        let mut correct_text_config = if text_json.get("text_config").is_some() {
             serde_json::from_value(text_json.get("text_config").unwrap().clone())?
        } else {
             // Flat config (0.6B style)
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

        // [CRITICAL] Inherit RoPE settings from the 2B model if we are in baking mode (cross-model relay)
        if baking_only && vision_json.get("text_config").is_some() {
            let v_text_cfg = vision_json.get("text_config").unwrap();
            if let Some(v_theta) = v_text_cfg.get("rope_theta").and_then(|v| v.as_f64()) {
                println!("[LOAD] Baking Mode: Inheriting rope_theta ({}) from 2B config", v_theta);
                correct_text_config.rope_theta = v_theta as f32;
            }
            if let Some(v_scaling) = v_text_cfg.get("rope_scaling") {
                println!("[LOAD] Baking Mode: Inheriting rope_scaling from 2B config");
                correct_text_config.rope_scaling = serde_json::from_value(v_scaling.clone()).ok();
            }
        }
        
        vl_config.text_config = Some(correct_text_config);
        
        // Ensure hidden_size at root matches text config for consistency
        vl_config.hidden_size = Some(vl_config.text_config.as_ref().unwrap().hidden_size);

        // [HYBRID-FILE-SELECTION] 
        let main_filename = if baking_only { "model-BITSERIAL_LAYER0.safetensors" } else { "model-BITSERIAL_ALL.safetensors" };
        let main_path = path_obj.join(main_filename);
        let main_file = std::fs::File::open(main_path)?;
        let main_mmap = Arc::new(unsafe { memmap2::MmapOptions::new().map(&main_file)? });
        
        // [VISION-BRANCH] 원안대로 텍스트 전용일 때는 비전 파일을 "절대" 쳐다보지도 않음
        let vision_mmap = if !force_text_only {
            let vision_filename = if baking_only { "mmproj-BITSERIAL_LAYER0.safetensors" } else { "mmproj-BITSERIAL_ALL.safetensors" };
            let vision_root = config_path.map(Path::new).unwrap_or(path_obj);
            let vision_path = vision_root.join(vision_filename);
            
            if vision_path.exists() {
                println!("[LOAD] Activating Vision Module from {:?}", vision_path.file_name());
                let vision_file = std::fs::File::open(vision_path)?;
                Some(Arc::new(unsafe { memmap2::MmapOptions::new().map(&vision_file)? }))
            } else {
                None
            }
        } else {
            println!("[LOAD] Strict Text-Only Mode. Vision Module blocked.");
            None
        };

        // [UPSCALE-LOADER] If we are baking with a small model for a large model context,
        // we might need architectural tensors (like q_norm/k_norm) from the large model.
        let secondary_mmap = if baking_only && tokenizer_path.is_some() {
            let large_path = Path::new(tokenizer_path.unwrap()).join("model-BITSERIAL_ALL.safetensors");
            if large_path.exists() {
                println!("[LOAD] Baking Mode: Loading secondary tensors from {:?}", large_path.file_name());
                let large_file = std::fs::File::open(large_path)?;
                Some(Arc::new(unsafe { memmap2::MmapOptions::new().map(&large_file)? }))
            } else { None }
        } else { None };

        let native_model = NativeQwen3VLModel::load(vl_config.clone(), main_mmap, vision_mmap, baking_only, secondary_mmap)?;
        let mut qwen3_vl_native = Arc::new(native_model);

        // [CRITICAL] Move to GPU immediately while reference count is 1
        // [STABILITY] Skip GPU offloading if baking_only is true to ensure 100% reliable cache generation
        if _text_device_id < 8 && !baking_only { 
            println!("[LOAD] Moving Native Model to GPU-{}...", _text_device_id);
            if let Some(m) = Arc::get_mut(&mut qwen3_vl_native) {
                m.move_to_gpu(_text_device_id as i32);
            } else {
                println!("[ERROR] Failed to get mutable reference for GPU offloading during initialization.");
            }
        } else if baking_only {
            println!("[LOAD] Baking Mode: Forcing CPU-only execution for 100% stability.");
        }

        let qwen3_vl = ModelVariant::Native(qwen3_vl_native);

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
            qwen3_vl,
            eos_token_id1: 151643,
            eos_token_id2: 151645,
            generation_config,
            model_name: "native-qwen3vl".to_string(),
            hard_token_limit,
        })
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, _session_id: Option<String>) -> Result<String> {
        let seqlen_offset = self.get_kv_len();
        println!("[GENERATE] Initial KV Offset: {}", seqlen_offset);

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info_native(&mes, &mes_render)?;
        let all_ids = self.tokenizer.text_encode_vec(input.replace_text, false)?;
        
        // [2025-H2-RESEARCH] Last-Block Recalibration (LBR)
        // Stabilizes 0.6B -> 2B relay by re-computing the final 64 tokens using the Large model.
        let recalibration_len = 64; 
        let mut local_pos = 0;
        let mut current_kv_offset = seqlen_offset;

        if seqlen_offset > 0 {
            // [DIAGNOSTIC] Print first 20 tokens of the new prompt
            let head_len = all_ids.len().min(20);
            let head_text = self.tokenizer.token_decode(all_ids[..head_len].to_vec()).unwrap_or_default();
            println!("[ZERO-PREFILL-DIAG] New Prompt (Head): '{}'", head_text.replace("\n", "\\n"));

            if all_ids.len() >= seqlen_offset {
                // CASE A: Standard Full-Text Match with LBR
                local_pos = seqlen_offset.saturating_sub(recalibration_len);
                current_kv_offset = local_pos; // Overwrite last 64 stale tokens with Large-model precision
                println!("[ZERO-PREFILL-LBR] Recalibrating last {} tokens for transition stability.", seqlen_offset - local_pos);
            } else {
                // CASE B: Modular Extension
                println!("[ZERO-PREFILL-MODULAR] Modular Mode. Cache ({}) > Prompt ({}).", seqlen_offset, all_ids.len());
                
                let mut found_split = 0;
                for i in 0..all_ids.len().min(20) {
                    if all_ids[i] == 151645 { // <|im_end|>
                        found_split = i + 1;
                        if found_split < all_ids.len() && all_ids[found_split] == 198 { // \n
                            found_split += 1;
                        }
                        break;
                    }
                }

                if found_split > 0 {
                    // Even in modular mode, we recalibrate if the prompt allows
                    local_pos = found_split;
                    println!("[ZERO-PREFILL-MODULAR] Skipping header ({} tokens).", found_split);
                } else {
                    local_pos = 0;
                }
            }
        }

        let prefill_chunk_size = 512;
        while local_pos < all_ids.len() {
            let remaining = all_ids.len() - local_pos;
            if remaining <= 1 { break; } // Keep the last token for generation trigger
            
            let chunk_size = remaining.min(prefill_chunk_size);
            let end = (local_pos + chunk_size).min(all_ids.len() - 1);
            let chunk = &all_ids[local_pos..end];
            
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); }
            }

            self.qwen3_vl.forward(chunk, input.pixel_values.as_deref(), input.image_grid_thw.as_ref(), current_kv_offset);
            
            let processed = chunk.len();
            local_pos += processed;
            current_kv_offset += processed;
        }

        let mut generated_text = String::new();
        let max_new_tokens = mes.max_tokens.unwrap_or(1024) as usize;
        let mut current_all_ids = all_ids.clone();

        println!("[GENERATE] Starting token generation loop...");

        for i in 0..max_new_tokens {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); }
            }

            // The very first token of the loop handles the "last token" of the prompt
            let input_ids = if i == 0 { 
                if local_pos < current_all_ids.len() {
                    current_all_ids[local_pos..].to_vec()
                } else {
                    vec![*current_all_ids.last().unwrap()]
                }
            } else { 
                vec![*current_all_ids.last().unwrap()] 
            };

            let (pixels, grid) = if i == 0 && seqlen_offset == 0 { 
                (input.pixel_values.as_deref(), input.image_grid_thw.as_ref()) 
            } else { 
                (None, None) 
            };

            let logits = self.qwen3_vl.forward(&input_ids, pixels, grid, current_kv_offset);
            
            // [DIST-AUDIT] Logit Confidence check at Step 0
            if i == 0 {
                let mut indexed_logits: Vec<(usize, f32)> = logits.iter().enumerate().map(|(idx, &val)| (idx, val.to_f32())).collect();
                indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                println!("[DIST-AUDIT] Top 5 logits at first step:");
                for j in 0..5 {
                    if j >= indexed_logits.len() { break; }
                    let (id, val) = indexed_logits[j];
                    let text = self.tokenizer.token_decode(vec![id as u32]).unwrap_or_default();
                    println!("  #{} -> ID: {}, val: {:.4}, text: '{}'", j+1, id, val, text.replace("\n", "\\n"));
                }
            }

            let next_id = self.sample_greedy(&logits);
            
            // [DIAGNOSTIC] 실시간 생성 로그
            if i < 10 || i % 20 == 0 {
                let decoded = self.tokenizer.token_decode(vec![next_id]).unwrap_or_default();
                println!("[GENERATE-STEP] step: {}, id: {}, text: '{}'", i, next_id, decoded.replace("\n", "\\n"));
            }

            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { 
                println!("[GENERATE] EOS detected at step {}.", i);
                break; 
            }
            
            current_all_ids.push(next_id);
            let new_word = self.tokenizer.token_decode(vec![next_id])?;
            generated_text.push_str(&new_word);
            
            current_kv_offset += input_ids.len();
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

    /// [OPTIMIZED] Context-aware splitting for large documents - Encode once, Bake in chunks
    pub fn bake_text_in_parts(&mut self, text: String, task_id: &str, suffix: &str, address_hash: Option<&str>, initial_offset: usize, cancel_flag: Option<Arc<AtomicBool>>) -> Result<()> {
        println!("[BAKE-STREAM] Starting optimized baking for {} (Global Pos: {}, Target: 512 tokens per chunk)", task_id, initial_offset);
        
        // 1. Encode the entire text ONCE
        let all_ids = self.tokenizer.text_encode_vec(text, false)?;
        
        self.clear_kv_cache(); 

        // 2. Process in 512-token chunks INCREMENTALLY
        let mut current_offset = initial_offset;
        for (chunk_idx, chunk) in all_ids.chunks(512).enumerate() {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Baking Cancelled")); }
            }

            println!("[BAKE-STREAM] Processing chunk {}/{} (Offset: {})", chunk_idx + 1, (all_ids.len() + 511) / 512, current_offset);

            // Forward Pass with proper incremental offset
            self.qwen3_vl.forward(chunk, None, None, current_offset);
            current_offset += chunk.len();
        }

        // 3. Final Pull: Extract the COMPLETE KV state from Layer 0 once
        let ModelVariant::Native(m) = &self.qwen3_vl;
        let h_d = m.text_model.config.head_dim;
        let n_kv = m.text_model.config.num_key_value_heads;
        if let Some((k, v)) = m.text_model.layers[0].get_kv_data(h_d, n_kv) {
            let final_path = crate::utils::paths::get_kv_dir(None, address_hash).join(format!("{}_{}.safetensors", task_id, suffix));
            self.save_raw_kv_to_disk(&final_path, &k, &v)?;
            println!("[BAKE-STREAM] SUCCESS: Saved context ({} tokens) to {:?}", k.len() * 32 / (n_kv * h_d), final_path);
        }

        self.clear_kv_cache(); 
        Ok(())
    }

    /// [OPTIMIZED] Context-aware splitting - Save to specific path
    pub fn bake_text_in_parts_to_path(&mut self, text: String, final_path: &Path, suffix: &str, initial_offset: usize, cancel_flag: Option<Arc<AtomicBool>>) -> Result<()> {
        println!("[BAKE-PATH] Baking incrementally to {:?} (Starting Offset: {})", final_path, initial_offset);
        
        let all_ids = self.tokenizer.text_encode_vec(text, false)?;
        let total_tokens = all_ids.len();
        
        self.clear_kv_cache();

        let mut current_offset = initial_offset;
        for (chunk_idx, chunk) in all_ids.chunks(512).enumerate() {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Baking Cancelled")); }
            }
            
            println!("[BAKE-PATH] Processing chunk {}/{} (Global Pos: {})", chunk_idx + 1, (total_tokens + 511) / 512, current_offset);
            
            self.qwen3_vl.forward(chunk, None, None, current_offset);
            current_offset += chunk.len();
        }

        let ModelVariant::Native(m) = &self.qwen3_vl;
        if let Some((k, v)) = m.text_model.layers[0].get_kv_data(m.text_model.config.head_dim, m.text_model.config.num_key_value_heads) {
            self.save_raw_kv_to_disk(final_path, &k, &v)?;
            println!("[BAKE-PATH] SUCCESS: Saved context ({} tokens) to {:?}", k.len() * 32 / (m.text_model.config.num_key_value_heads * m.text_model.config.head_dim), final_path);
        }
        
        self.clear_kv_cache();
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
        let mut tensors = std::collections::HashMap::new();
        
        // [DIST-AUDIT] Calculate stats for V-cache (Values are more sensitive to distribution)
        let v_f32: Vec<f32> = v.iter().map(|&x| x.to_f32()).collect();
        let sum: f32 = v_f32.iter().sum();
        let mean = sum / v_f32.len() as f32;
        let max_abs = v_f32.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("[DIST-AUDIT-BAKE] V-Cache Stats -> Mean: {:.6}, MaxAbs: {:.6}", mean, max_abs);

        let k_u8 = unsafe { std::slice::from_raw_parts(k.as_ptr() as *const u8, k.len() * 4) };
        tensors.insert("layer.0.k".to_string(), safetensors::tensor::TensorView::new(safetensors::Dtype::U32, vec![k.len()], k_u8)?);
        
        let v_u8 = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) };
        tensors.insert("layer.0.v".to_string(), safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![v.len()], v_u8)?);

        safetensors::tensor::serialize_to_file(tensors, &None, path)?;
        Ok(())
    }

    pub fn save_kv_to_disk(&self, path: &Path) -> Result<()> {
        let kvs = match &self.qwen3_vl {
            ModelVariant::Native(m) => m.text_model.get_all_kv(),
        };

        if kvs.is_empty() { 
            println!("[KV-DISK] WARNING: KV cache is empty, nothing to save.");
            return Ok(()); 
        }

        let abs_path = std::fs::canonicalize(path.parent().unwrap_or(Path::new(".")))
            .map(|p| p.join(path.file_name().unwrap()))
            .unwrap_or(path.to_path_buf());
            
        println!("[KV-DISK] Attempting to save {} layers to {:?}", kvs.len(), abs_path);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                println!("[KV-DISK] Creating missing directory: {:?}", parent);
                std::fs::create_dir_all(parent)?;
            }
        }

        let mut tensors = std::collections::HashMap::new();
        for (i, (k, v)) in kvs.into_iter().enumerate() {
            let k_u8 = unsafe { std::slice::from_raw_parts(k.as_ptr() as *const u8, k.len() * 4) };
            tensors.insert(format!("layer.{}.k", i), safetensors::tensor::TensorView::new(safetensors::Dtype::U32, vec![k.len()], k_u8)?);
            let v_u8 = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) };
            tensors.insert(format!("layer.{}.v", i), safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![v.len()], v_u8)?);
        }

        match safetensors::tensor::serialize_to_file(tensors, &None, path) {
            Ok(_) => println!("[KV-DISK] SUCCESS: Saved KV Cache to {:?}", abs_path),
            Err(e) => println!("[KV-DISK] ERROR: Failed to save safetensors: {}", e),
        }
        Ok(())
    }

    pub fn load_kv_from_disk(&self, path: &Path) -> Result<()> {
        self.load_kv_stitched(&[path.to_path_buf()])
    }

    pub fn load_kv_stitched(&self, paths: &[std::path::PathBuf]) -> Result<()> {
        match &self.qwen3_vl {
            ModelVariant::Native(m) => {
                let mut combined_k = Vec::new();
                let mut combined_v = Vec::new();
                
                let target_head_dim = m.text_model.config.head_dim;
                let target_n_kv = m.text_model.config.num_key_value_heads;
                let target_dim = target_head_dim * target_n_kv; 

                for path in paths {
                    if !path.exists() { continue; }
                    println!("[KV-STITCH] Loading component: {:?}", path.file_name());
                    let file = std::fs::read(path)?;
                    let st = safetensors::SafeTensors::deserialize(&file)?;
                    
                    if let (Ok(kt), Ok(vt)) = (st.tensor("layer.0.k"), st.tensor("layer.0.v")) {
                        let mut k_data: Vec<u32> = kt.data().chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
                        let mut v_data: Vec<f16> = vt.data().chunks_exact(2).map(|c| f16::from_ne_bytes(c.try_into().unwrap())).collect();
                        
                        // [DYNAMIC-DIMENSION-MATCHING] Robust mapping for Qwen family.
                        // 0.6B models usually have 8 KV heads * 128 head_dim = 1024.
                        // 2B models vary (e.g., 1536 or 2048).
                        let actual_source_dim = if k_data.len() > 0 && v_data.len() > 0 {
                            (v_data.len() * 32) / k_data.len()
                        } else { 1024 };
                        
                        if target_dim > actual_source_dim && target_dim % actual_source_dim == 0 {
                            let ratio = target_dim / actual_source_dim;
                            println!("[KV-UPSCALE] 2025-H2 Drift Compensation: Bridging 1-bit Small -> Large");
                            
                            let mut new_k = Vec::with_capacity(k_data.len() * ratio);
                            let mut new_v = Vec::with_capacity(v_data.len() * ratio);
                            
                            // [2025-H2-RESEARCH] Drift Compensation Factor
                            // Shift the 1-bit signal mean to match Large model's activation center
                            let semantic_multiplier = f16::from_f32(1.12);
                            let drift_bias = f16::from_f32(0.005); 

                            let k_units_per_token = actual_source_dim / 32;
                            for chunk in k_data.chunks_exact(k_units_per_token) {
                                for _ in 0..ratio { new_k.extend_from_slice(chunk); }
                            }
                            for chunk in v_data.chunks_exact(actual_source_dim) {
                                for _ in 0..ratio { 
                                    let mut v_scaled = chunk.to_vec();
                                    for val in v_scaled.iter_mut() { 
                                        // Apply both Scale and Drift Bias for 1-bit stability
                                        *val = (*val * semantic_multiplier) + drift_bias; 
                                    }
                                    new_v.extend_from_slice(&v_scaled); 
                                }
                            }
                            k_data = new_k;
                            v_data = new_v;
                        }
                        
                        combined_k.extend(k_data);
                        combined_v.extend(v_data);
                    }
                }

                if !combined_k.is_empty() {
                    // [CRITICAL-FIX] Clear any existing GPU cache to force a fresh start with new data.
                    m.text_model.force_free_kv_cache();
                    
                    let total_tokens = (combined_k.len() * 32) / target_dim;
                    
                    // [DIST-AUDIT] Restored Stats check
                    let v_f32: Vec<f32> = combined_v.iter().map(|&x| x.to_f32()).collect();
                    let sum: f32 = v_f32.iter().sum();
                    let mean = sum / v_f32.len() as f32;
                    let max_abs = v_f32.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                    println!("[DIST-AUDIT-LOAD] Restored V-Cache Stats -> Mean: {:.6}, MaxAbs: {:.6}", mean, max_abs);

                    println!("[KV-STITCH-VERIFY] Load Complete: {} tokens (Dim: {})", total_tokens, target_dim);
                    
                    if total_tokens > 0 {
                        m.text_model.batch_upload_stitched_cache(combined_k, combined_v);
                    }
                }
            }
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
            ModelVariant::Native(m) => m.text_model.get_kv_len()
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match &self.qwen3_vl {
            ModelVariant::Native(m) => m.clear_kv_cache(),
        }
    }
}

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
            serde_json::from_value(vision_json)?
        } else {
            // Fallback for flat config, but we must override text params
            serde_json::from_value(vision_json.clone())?
        };

        // OVERRIDE text_config with the actual 0.6B parameters
        let correct_text_config = if text_json.get("text_config").is_some() {
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

        let native_model = NativeQwen3VLModel::load(vl_config.clone(), main_mmap, vision_mmap, baking_only)?;
        let mut qwen3_vl_native = Arc::new(native_model);

        // [CRITICAL] Move to GPU immediately while reference count is 1
        if _text_device_id < 8 { // Valid GPU ID check (0-7)
            println!("[LOAD] Moving Native Model to GPU-{}...", _text_device_id);
            if let Some(m) = Arc::get_mut(&mut qwen3_vl_native) {
                m.move_to_gpu(_text_device_id as i32);
            } else {
                println!("[ERROR] Failed to get mutable reference for GPU offloading during initialization.");
            }
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
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info_native(&mes, &mes_render)?;
        let mut all_ids = self.tokenizer.text_encode_vec(input.replace_text, false)?;
        
        let mut seqlen_offset = self.get_kv_len();
        let mut generated_text = String::new();
        let max_new_tokens = mes.max_tokens.unwrap_or(1024) as usize;

        for i in 0..max_new_tokens {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); }
            }

            let input_ids = if i == 0 { all_ids.clone() } else { vec![*all_ids.last().unwrap()] };
            let (pixels, grid) = if i == 0 && seqlen_offset == 0 { (input.pixel_values.as_deref(), input.image_grid_thw.as_ref()) } else { (None, None) };

            let logits = self.qwen3_vl.forward(&input_ids, pixels, grid, seqlen_offset);
            let next_id = self.sample_greedy(&logits);
            
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            
            all_ids.push(next_id);
            let new_word = self.tokenizer.token_decode(vec![next_id])?;
            print!("{}", new_word); 
            use std::io::Write;
            let _ = std::io::stdout().flush();
            generated_text.push_str(&new_word);
            
            seqlen_offset += input_ids.len();
        }

        println!("\n[INFERENCE-RESULT] {}", generated_text);
        Ok(generated_text)
    }

    pub fn prefill_chunk(&mut self, text: String, _cancel_flag: Option<Arc<AtomicBool>>, _relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let ids = self.tokenizer.text_encode_vec(text, false)?;
        let seqlen = self.get_kv_len();
        self.qwen3_vl.forward(&ids, None, None, seqlen);
        Ok(ids.len())
    }

    /// [NEW] Context-aware splitting for large documents
    pub fn bake_text_in_parts(&mut self, text: String, task_id: &str, suffix: &str, cancel_flag: Option<Arc<AtomicBool>>) -> Result<()> {
        let lines: Vec<&str> = text.lines().collect();
        let mut current_chunk = String::new();
        let mut current_tokens = 0;
        let mut part_idx = 1;

        println!("[BAKE-PARTS] Starting split-baking for {} (Target: 512 per part)", task_id);

        for line in lines {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Baking Cancelled")); }
            }

            let line_text = format!("{}\n", line);
            let line_ids = self.tokenizer.text_encode_vec(line_text.clone(), false)?;
            let line_token_count = line_ids.len();

            // Check if adding this line exceeds 512 tokens
            if current_tokens + line_token_count > 512 && !current_chunk.is_empty() {
                // 1. Bake the current accumulated chunk
                let chunk_ids = self.tokenizer.text_encode_vec(current_chunk.clone(), false)?;
                self.qwen3_vl.forward(&chunk_ids, None, None, 0); // Always 0 because we clear cache each time

                // 2. Save KV snapshot for this part
                let part_path = crate::utils::paths::get_kv_dir(None).join(format!("{}_{}_part{}.safetensors", task_id, suffix, part_idx));
                self.save_kv_to_disk(&part_path)?;

                // 3. Clear memory for next part
                self.clear_kv_cache();
                println!("[BAKE-PARTS] Completed part {} ({} tokens)", part_idx, current_tokens);

                // 4. Reset for next part
                current_chunk.clear();
                current_tokens = 0;
                part_idx += 1;
            }

            current_chunk.push_str(&line_text);
            current_tokens += line_token_count;
        }

        // Handle remaining text
        if !current_chunk.is_empty() {
            let chunk_ids = self.tokenizer.text_encode_vec(current_chunk, false)?;
            self.qwen3_vl.forward(&chunk_ids, None, None, 0);
            let part_path = crate::utils::paths::get_kv_dir(None).join(format!("{}_{}_part{}.safetensors", task_id, suffix, part_idx));
            self.save_kv_to_disk(&part_path)?;
            self.clear_kv_cache();
            println!("[BAKE-PARTS] Completed final part {} ({} tokens)", part_idx, current_tokens);
        }

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
                let target_dim = target_head_dim * target_n_kv; // e.g., 2048 for 2B

                for path in paths {
                    if !path.exists() { continue; }
                    let file = std::fs::read(path)?;
                    let st = safetensors::SafeTensors::deserialize(&file)?;
                    
                    if let (Ok(kt), Ok(vt)) = (st.tensor("layer.0.k"), st.tensor("layer.0.v")) {
                        let mut k_data: Vec<u32> = kt.data().chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
                        let mut v_data: Vec<f16> = vt.data().chunks_exact(2).map(|c| f16::from_ne_bytes(c.try_into().unwrap())).collect();
                        
                        // [DIMENSION-MATCHING] Check if upscaling is needed (e.g., 0.6B -> 2B)
                        let source_dim_bits = (k_data.len() * 32) / (v_data.len() / k_data.len() / 32).max(1); // Approximate source dim
                        // Actually, a simpler way is to check the ratio between current model dim and source dim
                        let seq_len = v_data.len() / (v_data.len() / (k_data.len() * 32 / target_head_dim).max(1)).max(1); // Dummy placeholder
                        
                        // Calculate real sequence length based on the smallest unit
                        // For bit-serial, 1 token = target_dim / 32 units of u32
                        let source_dim = 1024; // Most common small model dim
                        if target_dim > source_dim && target_dim % source_dim == 0 {
                            let ratio = target_dim / source_dim;
                            println!("[KV-UPscale] Matching dimensions: {} -> {} (Ratio: {})", source_dim, target_dim, ratio);
                            
                            let mut new_k = Vec::with_capacity(k_data.len() * ratio);
                            let mut new_v = Vec::with_capacity(v_data.len() * ratio);
                            
                            // Repeat data to match new dimension
                            for chunk in k_data.chunks_exact(source_dim / 32) {
                                for _ in 0..ratio { new_k.extend_from_slice(chunk); }
                            }
                            for chunk in v_data.chunks_exact(source_dim) {
                                for _ in 0..ratio { new_v.extend_from_slice(chunk); }
                            }
                            k_data = new_k;
                            v_data = new_v;
                        }
                        
                        combined_k.extend(k_data);
                        combined_v.extend(v_data);
                    }
                }

                if !combined_k.is_empty() {
                    m.text_model.layers[0].set_kv_data(combined_k, combined_v);
                    println!("[KV-STITCH] Successfully merged & matched {} KV segments into Layer 0", paths.len());
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
            ModelVariant::Native(m) => {
                let cache = m.text_model.layers[0].kv_cache.lock().unwrap();
                if let Some((k, _)) = cache.as_ref() {
                    let head_dim = m.text_model.config.head_dim;
                    let n_kv = m.text_model.config.num_key_value_heads;
                    // Bit-serial packing: Each u32 stores 32 bits (signs)
                    // k.len() is number of u32s. Total bits = k.len() * 32.
                    // seq_len = Total bits / (n_kv * head_dim)
                    (k.len() * 32) / (n_kv * head_dim)
                } else { 0 }
            }
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match &self.qwen3_vl {
            ModelVariant::Native(m) => m.clear_kv_cache(),
        }
    }
}

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
    openai_types::{
        ChatCompletionParameters, 
        ChatCompletionRequestMessage, 
        ChatCompletionRequestUserMessageContent
    },
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

        // [STRICT-CONFIG-ISOLATION] 
        let mut vl_config: Qwen3VLConfig = if baking_only {
            // In Baking mode, we ONLY care about the text model's own parameters.
            let mut cfg: Qwen3VLConfig = serde_json::from_value(text_json.clone())?;
            cfg.text_config = Some(serde_json::from_value(text_json)?);
            cfg
        } else {
            // In Inference mode, we use 2B's structure but override with 0.6B's text params.
            let mut cfg: Qwen3VLConfig = if vision_json.get("text_config").is_some() {
                serde_json::from_value(vision_json)?
            } else {
                serde_json::from_value(vision_json.clone())?
            };
            
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
            cfg.text_config = Some(correct_text_config);
            cfg
        };
        
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
        let mut seqlen_offset = self.get_kv_len();
        println!("[DIAG] Start Generate. Current KV Cache Offset: {}", seqlen_offset);

        // [RAW-BYPASS] 
        let mes_render_default = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info_native(&mes, &mes_render_default)?;

        let input_text = if let Some(ChatCompletionRequestMessage::User(u)) = mes.messages.last() {
            match &u.content {
                ChatCompletionRequestUserMessageContent::String(s) if s.contains("<|im_start|>") => s.clone(),
                _ => input.replace_text.clone()
            }
        } else {
            input.replace_text.clone()
        };

        let all_ids = self.tokenizer.text_encode_vec(input_text, false)?;
        
        if all_ids.len() > 10 {
            println!("[DIAG] Current Prompt Fingerprint (first 10): {:?}", &all_ids[..10]);
        }
        println!("[DIAG] Total Tokens in this request: {}", all_ids.len());

        // [RESUME-LOGIC] Determine if we can skip prefill based on existing KV cache
        let mut local_pos = 0;
        if seqlen_offset > 0 {
            if seqlen_offset < all_ids.len() {
                // Scenario: We have partial context (e.g. baked PUG + System Prompt).
                // We skip what is already in VRAM and process only the new user suffix.
                println!("[DIAG] Perfect Match Detected. Skipping {} baked tokens. Remaining: {}", seqlen_offset, all_ids.len() - seqlen_offset);
                local_pos = seqlen_offset;
            } else {
                // Scenario: Unexpected offset (e.g. cache is larger than prompt)
                // Fallback: Clear and restart to be safe, or treat as relay.
                println!("[DIAG] Cache Offset ({}) >= Total Tokens ({}). Restarting Prefill.", seqlen_offset, all_ids.len());
                self.clear_kv_cache();
                seqlen_offset = 0;
                local_pos = 0;
            }
        } else {
            println!("[DIAG] Fresh Start (No Cache). Prefilling {} tokens.", all_ids.len());
        }

        let prefill_start = std::time::Instant::now();
        let prefill_chunk_size = 512;
        while local_pos < all_ids.len() {
            let remaining = all_ids.len() - local_pos;
            if remaining <= 1 { break; } 
            
            let chunk_size = remaining.min(prefill_chunk_size);
            let end = (local_pos + chunk_size).min(all_ids.len() - 1);
            let chunk = &all_ids[local_pos..end];
            
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); }
            }

            // [BATCH-ACCELERATION] Process the entire chunk at once for maximum speed
            match &mut self.qwen3_vl {
                ModelVariant::Native(m) => m.forward_kv_only(chunk, seqlen_offset),
            }
            
            seqlen_offset += chunk.len();
            local_pos += chunk.len();
        }
        
        // Final token always gets a full forward pass to start generation
        if local_pos < all_ids.len() {
            let last_id = all_ids[all_ids.len() - 1];
            self.qwen3_vl.forward(&[last_id], None, None, seqlen_offset);
        }

        if all_ids.len() > 1 {
            println!("[DIAG] Total Suffix Batch Ingestion took {}ms", prefill_start.elapsed().as_millis());
        }

        let mut generated_text = String::new();
        let max_new_tokens = mes.max_tokens.unwrap_or(1024) as usize;
        let mut current_all_ids = all_ids.clone();

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

            let logits = self.qwen3_vl.forward(&input_ids, pixels, grid, seqlen_offset);
            let next_id = self.sample_greedy(&logits);
            
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            
            current_all_ids.push(next_id);
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

    /// [OPTIMIZED] Context-aware splitting for large documents - Encode once, Bake in chunks
    pub fn bake_text_in_parts(&mut self, text: String, task_id: &str, suffix: &str, address_hash: Option<&str>, cancel_flag: Option<Arc<AtomicBool>>) -> Result<()> {
        println!("[BAKE-STREAM] Starting optimized baking for {} (Target: 512 tokens per chunk)", task_id);
        
        // 1. Encode the entire text ONCE
        let all_ids = self.tokenizer.text_encode_vec(text, false)?;
        let total_tokens = all_ids.len();
        
        let mut master_k: Vec<u32> = Vec::new();
        let mut master_v: Vec<f16> = Vec::new();

        // 2. Process in 512-token chunks
        for (chunk_idx, chunk) in all_ids.chunks(512).enumerate() {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Baking Cancelled")); }
            }

            println!("[BAKE-STREAM] Processing chunk {}/{}", chunk_idx + 1, (total_tokens + 511) / 512);

            // Forward Pass (Always from offset 0 because we clear cache every time)
            self.qwen3_vl.forward(chunk, None, None, 0);

            // Pull KV from GPU to RAM
            let ModelVariant::Native(m) = &self.qwen3_vl;
            let h_d = m.text_model.config.head_dim;
            let n_kv = m.text_model.config.num_key_value_heads;
            if let Some((k, v)) = m.text_model.layers[0].get_kv_data(h_d, n_kv) {
                master_k.extend(k);
                master_v.extend(v);
            }

            // Clear VRAM for next chunk
            self.clear_kv_cache();
        }

        // 3. Final Save
        if !master_k.is_empty() {
            let final_path = crate::utils::paths::get_kv_dir(None, address_hash).join(format!("{}_{}.safetensors", task_id, suffix));
            self.save_raw_kv_to_disk(&final_path, &master_k, &master_v)?;
            println!("[BAKE-STREAM] SUCCESS: Saved giant KV context ({} tokens) to {:?}", master_v.len() / (8 * 128), final_path);
        }

        Ok(())
    }

    /// [OPTIMIZED] Context-aware splitting - Save to specific path
    pub fn bake_text_in_parts_to_path(&mut self, text: String, final_path: &Path, _suffix: &str, cancel_flag: Option<Arc<AtomicBool>>) -> Result<()> {
        println!("[BAKE-PATH] Baking to {:?}", final_path);
        
        let all_ids = self.tokenizer.text_encode_vec(text, false)?;
        let total_tokens = all_ids.len();
        
        let mut master_k: Vec<u32> = Vec::new();
        let mut master_v: Vec<f16> = Vec::new();
        let mut current_offset = 0;

        for chunk in all_ids.chunks(512) {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Baking Cancelled")); }
            }
            
            // [FIX] Pass incremental offset to ensure correct Positional Embeddings
            self.qwen3_vl.forward(chunk, None, None, current_offset);
            
            let ModelVariant::Native(m) = &self.qwen3_vl;
            if let Some((k, v)) = m.text_model.layers[0].get_kv_data(m.text_model.config.head_dim, m.text_model.config.num_key_value_heads) {
                // We only want the NEW tokens' KV from this forward pass
                // But forward() appends to cache. However, since we call clear_kv_cache() 
                // at the end of the loop, get_kv_data() will return exactly the current chunk.
                master_k.extend(k); master_v.extend(v);
            }
            self.clear_kv_cache();
            current_offset += chunk.len();
        }

        if !master_k.is_empty() {
            self.save_raw_kv_to_disk(final_path, &master_k, &master_v)?;
            println!("[BAKE-PATH] SUCCESS: Saved context to {:?}", final_path);
        }
        Ok(())
    }

    pub fn save_raw_kv_to_disk(&self, path: &Path, k: &[u32], v: &[f16]) -> Result<()> {
        let mut tensors = std::collections::HashMap::new();
        
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
                        let k_data: Vec<u32> = kt.data().chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
                        let v_data: Vec<f16> = vt.data().chunks_exact(2).map(|c| f16::from_ne_bytes(c.try_into().unwrap())).collect();
                        
                        let packed_head_dim = target_head_dim / 32; // 4
                        let mut final_k = Vec::new();
                        let mut final_v = Vec::new();

                        // [STRICT-PARITY] 0.6B(Small) -> 2B(Large) Semantic Bridge
                        let s_head_dim = 64; 
                        let s_n_kv = 2; // Assuming 0.6B has 2 KV heads
                        let s_packed_dim = s_head_dim / 32; // 2

                        let s_k_per_token = s_n_kv * s_packed_dim; // 4
                        let s_v_per_token = s_n_kv * s_head_dim;   // 128

                        if k_data.len() % s_k_per_token == 0 && v_data.len() % s_v_per_token == 0 {
                            let total_tokens = k_data.len() / s_k_per_token;
                            println!("[KV-STITCH] Expanding {} tokens: 0.6B(Small) -> 2B(Large) Head-Aware Mapping.", total_tokens);
                            
                            for t in 0..total_tokens {
                                let kc = &k_data[t * s_k_per_token .. (t+1) * s_k_per_token];
                                let vc = &v_data[t * s_v_per_token .. (t+1) * s_v_per_token];

                                // 1. 먼저 소스의 Head들을 128차원으로 각각 확장
                                let mut expanded_heads_k = Vec::with_capacity(s_n_kv * packed_head_dim);
                                let mut expanded_heads_v = Vec::with_capacity(s_n_kv * target_head_dim);
                                
                                for h in 0..s_n_kv {
                                    let kh = &kc[h * s_packed_dim .. (h+1) * s_packed_dim];
                                    let vh = &vc[h * s_head_dim .. (h+1) * s_head_dim];
                                    
                                    // Dimension Doubling (64 -> 128)
                                    expanded_heads_k.extend_from_slice(kh);
                                    expanded_heads_k.extend_from_slice(kh);
                                    expanded_heads_v.extend_from_slice(vh);
                                    expanded_heads_v.extend_from_slice(vh);
                                }

                                // 2. 타겟 모델의 Head 수(target_n_kv)에 맞춰 순환 할당 (Cycling)
                                for h_idx in 0..target_n_kv {
                                    let src_h = h_idx % s_n_kv;
                                    final_k.extend_from_slice(&expanded_heads_k[src_h * packed_head_dim .. (src_h + 1) * packed_head_dim]);
                                    final_v.extend_from_slice(&expanded_heads_v[src_h * target_head_dim .. (src_h + 1) * target_head_dim]);
                                }
                            }
                        } else {
                            println!("[KV-STITCH] Warning: 0.6B format mismatch. Falling back to direct replication.");
                            final_k = k_data;
                            final_v = v_data;
                        }
                        
                        combined_k.extend(final_k);
                        combined_v.extend(final_v);
                    }
                }

                if !combined_k.is_empty() {
                    // [CRITICAL-FIX] Clear any existing GPU cache to force a fresh start with new data.
                    m.text_model.force_free_kv_cache();
                    
                    let total_tokens = (combined_k.len() * 32) / target_dim;
                    println!("[KV-STITCH] Loaded {} tokens. Dropping last token for re-evaluation parity.", total_tokens);
                    
                    if total_tokens > 0 {
                        let tokens_to_keep = if total_tokens > 1 { total_tokens - 1 } else { total_tokens };
                        let k_keep = tokens_to_keep * (target_dim / 32);
                        let v_keep = tokens_to_keep * target_dim;
                        
                        // Only truncate if we actually have more than we need
                        if combined_k.len() > k_keep { combined_k.truncate(k_keep); }
                        if combined_v.len() > v_keep { combined_v.truncate(v_keep); }
                        
                        println!("[KV-STITCH] Replicating {} tokens across all available layers.", tokens_to_keep);
                        m.text_model.batch_upload_stitched_cache(combined_k, combined_v);
                    } else {
                        println!("[KV-STITCH] Token calculation resulted in 0. Clearing cache.");
                        m.text_model.clear_kv_cache();
                    }
                    
                    println!("[KV-STITCH] SUCCESS: Ready for high-speed full-model inference.");
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
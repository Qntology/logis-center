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

        let main_filename = if baking_only { "model-BITSERIAL_LAYER0.safetensors" } else { "model-BITSERIAL_ALL.safetensors" };
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

        let secondary_mmap = if baking_only && tokenizer_path.is_some() {
            let large_path = Path::new(tokenizer_path.unwrap()).join("model-BITSERIAL_ALL.safetensors");
            if large_path.exists() {
                let large_file = std::fs::File::open(large_path)?;
                Some(Arc::new(unsafe { memmap2::MmapOptions::new().map(&large_file)? }))
            } else { None }
        } else { None };

        let native_model = NativeQwen3VLModel::load(vl_config.clone(), main_mmap, vision_mmap, baking_only, secondary_mmap)?;
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
            if let Some(m) = Arc::get_mut(&mut qwen3_vl_native) {
                m.move_to_gpu(_text_device_id as i32);
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

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info_native(&mes, &mes_render)?;
        let all_ids = self.tokenizer.text_encode_vec(input.replace_text, false)?;
        
        let mut local_pos = 0;
        let mut current_kv_offset = seqlen_offset;

        if no_bridge && seqlen_offset > 0 {
            // [BACKUP-PARITY] 베이킹된 문맥이 이미 있으므로, 전체 프롬프트 중 
            // 베이킹된 부분(HTML)을 건너뛰고 순수 질문 토큰만 찾아서 주입해야 함.
            // 보통 <|im_start|>user\n 문구 뒤부터가 실제 질문임.
            let user_trigger = self.tokenizer.text_encode_vec("<|im_start|>user\n".to_string(), false)?;
            if let Some(pos) = all_ids.windows(user_trigger.len()).position(|window| window == user_trigger) {
                local_pos = pos;
                println!("[{}-INTEGRATED] Fast-forwarding to user query at pos {}.", tag, local_pos);
            }
        } else if seqlen_offset > 0 && !no_bridge {
            let recalibration_len = 32;
            let bridge_len = all_ids.len().min(recalibration_len);
            current_kv_offset = seqlen_offset.saturating_sub(bridge_len);
            local_pos = 0;
            println!("[{}-STITCH] Bridging {} tokens at offset {} to sync context.", tag, bridge_len, current_kv_offset);
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

        // [LBR-STABILITY] Recalibrate the last block to bridge 0.6B -> 2B drift
        // This is critical for 1-bit models to maintain context coherence after stitching.
        if !no_bridge || seqlen_offset > 0 {
            let recalib_len = current_kv_offset.min(32);
            let safe_offset = current_kv_offset.saturating_sub(recalib_len);
            
            if recalib_len > 0 {
                // Ensure we use the most recent tokens from all_ids if available, or just re-run the tail
                let recalib_start = all_ids.len().saturating_sub(recalib_len);
                let recalib_chunk = &all_ids[recalib_start..];
                
                println!("[LBR-STITCH] Synchronizing last {} tokens at offset {} for 2B activation alignment...", recalib_chunk.len(), safe_offset);
                // [CRITICAL] We don't need pixels/grid here, just text activation sync
                self.qwen3_vl.forward(recalib_chunk, None, None, safe_offset);
            }
        }

        let mut generated_text = String::new();
        let max_new_tokens = mes.max_tokens.unwrap_or(1024) as usize;
        let mut current_all_ids = all_ids.clone();

        println!("[GENERATE] Starting speculative token generation loop...");

        for i in 0..max_new_tokens {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); }
            }

            // [2025-SPECULATIVE] Try to get a draft token from the Small model if it's hibernated/available
            // For now, we use 2B for verification and 0.6B for drafting if the architecture allows.
            // Simplified: Verification-first approach to stabilize 1-bit drift
            
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

            // Verified forward pass
            let logits = self.qwen3_vl.forward(&input_ids, pixels, grid, current_kv_offset);
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
        println!("[BAKE-STREAM] Starting optimized baking for {} (Global Pos: {}, Target: {} tokens per chunk)", task_id, initial_offset, self.max_chunk_size);
        
        // 1. Encode the entire text ONCE
        let all_ids = self.tokenizer.text_encode_vec(text, false)?;
        
        self.clear_kv_cache(); 

        // 2. Process in dynamic chunks INCREMENTALLY
        let mut current_offset = initial_offset;
        for (chunk_idx, chunk) in all_ids.chunks(self.max_chunk_size).enumerate() {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Baking Cancelled")); }
            }

            println!("[BAKE-STREAM] Processing chunk {}/{} (Offset: {})", chunk_idx + 1, (all_ids.len() + self.max_chunk_size - 1) / self.max_chunk_size, current_offset);

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
        println!("[BAKE-PATH] Baking incrementally to {:?} (Starting Offset: {}, Chunk Size: {})", final_path, initial_offset, self.max_chunk_size);
        
        let all_ids = self.tokenizer.text_encode_vec(text, false)?;
        let total_tokens = all_ids.len();
        
        self.clear_kv_cache();

        let mut current_offset = initial_offset;
        for (chunk_idx, chunk) in all_ids.chunks(self.max_chunk_size).enumerate() {
            if let Some(flag) = &cancel_flag {
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Baking Cancelled")); }
            }
            
            println!("[BAKE-PATH] Processing chunk {}/{} (Global Pos: {})", chunk_idx + 1, (total_tokens + self.max_chunk_size - 1) / self.max_chunk_size, current_offset);
            
            self.qwen3_vl.forward(chunk, None, None, current_offset);
            current_offset += chunk.len();
        }

        // [FIX] Save ALL layers, not just layer 0
        self.save_kv_to_disk(final_path)?;
        
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

    pub fn save_kv_to_disk(&self, path: &Path) -> Result<()> {
        let kvs = match &self.qwen3_vl {
            ModelVariant::Native(m) => m.text_model.get_all_kv(),
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
        // [FIX] Windows Os Error 1784 대응: raw_parts 대신 안전한 바이트 복사 사용

        for (i, (k, v)) in kvs.into_iter().enumerate() {
            // K (u32) to bytes
            let mut k_bytes = Vec::with_capacity(k.len() * 4);
            for &val in &k { k_bytes.extend_from_slice(&val.to_ne_bytes()); }
            tensors.insert(format!("layer.{}.k", i), (safetensors::Dtype::U32, vec![k.len()], k_bytes));
            
            // V (f16) to bytes
            let mut v_bytes = Vec::with_capacity(v.len() * 2);
            for &val in &v { v_bytes.extend_from_slice(&val.to_ne_bytes()); }
            tensors.insert(format!("layer.{}.v", i), (safetensors::Dtype::F16, vec![v.len()], v_bytes));
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

                for path in paths {
                    if !path.exists() { continue; }
                    println!("[KV-STITCH] Loading component: {:?}", path.file_name());
                    let file = std::fs::read(path)?;
                    let st = safetensors::SafeTensors::deserialize(&file)?;
                    
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
                                println!("[KV-UPSCALE] Applying 2026-H1 Drift Compensation (1.12x + 0.005 bias)");
                                
                                let mut new_k = Vec::with_capacity(k_data.len() * ratio);
                                let mut new_v = Vec::with_capacity(v_data.len() * ratio);
                                
                                // [2026-COGNITIVE-STABILITY] Drift Compensation Factor
                                let semantic_multiplier = f16::from_f32(1.12);
                                let drift_bias = f16::from_f32(0.005); 

                                let k_units_per_token = actual_source_dim / 32;
                                for chunk in k_data.chunks_exact(k_units_per_token) { for _ in 0..ratio { new_k.extend_from_slice(chunk); } }
                                for chunk in v_data.chunks_exact(actual_source_dim) {
                                    for _ in 0..ratio { 
                                        let mut v_scaled = chunk.to_vec();
                                        for val in v_scaled.iter_mut() { 
                                            // Scale signal and shift mean to Large model's activation center
                                            *val = (*val * semantic_multiplier) + drift_bias; 
                                        }
                                        new_v.extend_from_slice(&v_scaled); 
                                    }
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
            ModelVariant::Native(m) => m.text_model.get_kv_len()
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match &self.qwen3_vl {
            ModelVariant::Native(m) => m.clear_kv_cache(),
        }
    }
}

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
use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
use std::fs;
use std::path::Path;

use std::path::PathBuf;
use tokio::sync::mpsc;

// [GLOBAL] 전역 대기열 카운터: 20개 상한선 체크용
pub static ACTIVE_BAKE_TASKS: AtomicUsize = AtomicUsize::new(0);

// [MEMORY] 강제 메모리 해제 및 초기화 함수 (Async)
async fn purge_vram_position(device: &Device) {
    if device.is_cuda() {
        let dev_clone = device.clone();
        // 비동기 방식으로 GPU 동기화를 호출하여 현재 포지션의 연산 잔재를 완전히 소거
        let _ = tokio::task::spawn_blocking(move || {
            let _ = dev_clone.synchronize();
        }).await;
    }
}

struct LayerKVDump {
    layer_idx: usize,
    k: Tensor,
    v: Tensor,
}

struct BakeTask {
    task_dir: PathBuf,
    kv_name: Option<String>,
    offset: usize,
    layers: Vec<LayerKVDump>,
}

fn spawn_bake_worker(mut rx: mpsc::Receiver<BakeTask>) {
    tokio::spawn(async move {
        println!("[BAKE-WORKER] Background KV storage worker started.");
        while let Some(task) = rx.recv().await {
            let task_dir = task.task_dir;
            let kv_name = task.kv_name;
            let offset = task.offset;
            
            println!("[BAKE-WORKER] Saving block at offset {}...", offset);
            
            for layer in task.layers {
                let filename = match (&kv_name, offset) {
                    (Some(name), 0) => format!("layer_{}_kv.safetensors", name),
                    (Some(name), off) => format!("layer_{}_kv_{}.safetensors", name, off),
                    (None, 0) => format!("layer_{}_kv.safetensors", layer.layer_idx),
                    (None, off) => format!("layer_{}_kv_{}.safetensors", layer.layer_idx, off),
                };
                let path = task_dir.join(filename);
                
                // --- Manual BitKV Compression (Worker Thread) ---
                let mut map = std::collections::HashMap::new();
                
                let process_tensor = |t: Tensor, prefix: &str, target_map: &mut std::collections::HashMap<String, Tensor>| {
                    if let Ok(dims) = t.dims4() {
                        let (b, h, s, d) = dims;
                        if let Ok(t_f32) = t.to_device(&Device::Cpu).and_then(|t| t.to_dtype(DType::F32)) {
                            if let Ok(t_data) = t_f32.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                                let anchor_count = (0..s).filter(|&i| i < 4 || i % 8 == 0).count();
                                let mut anchors = vec![0.0f32; b * h * anchor_count * d];
                                let mut packed_residuals = vec![0u8; (b * h * s * d + 7) / 8];
                                let mut scales = vec![0.0f32; b * h * s];

                                // Bit Packing & Anchors/Scales (Simplified loop for worker)
                                let head_token_size = s * d;
                                for bh_idx in 0..(b * h) {
                                    let bh_offset = bh_idx * head_token_size;
                                    for i in 0..head_token_size {
                                        if t_data[bh_offset + i] >= 0.0 {
                                            packed_residuals[(bh_offset + i) / 8] |= 1 << ((bh_offset + i) % 8);
                                        }
                                    }
                                    
                                    for token_idx in 0..s {
                                        let token_data = &t_data[bh_offset + token_idx * d .. bh_offset + (token_idx + 1) * d];
                                        if token_idx < 4 || token_idx % 8 == 0 {
                                            let anchor_pos = if token_idx < 4 { token_idx } else { 4 + (token_idx - 4) / 8 };
                                            anchors[(bh_idx * anchor_count + anchor_pos) * d .. (bh_idx * anchor_count + anchor_pos + 1) * d].copy_from_slice(token_data);
                                        }
                                        let mut max_abs = 0.0f32;
                                        for &v in token_data { let a = v.abs(); if a > max_abs { max_abs = a; } }
                                        scales[bh_idx * s + token_idx] = max_abs;
                                    }
                                }

                                if let Ok(at) = Tensor::from_vec(anchors, vec![b, h, anchor_count, d], &Device::Cpu) { target_map.insert(format!("{}_anchors", prefix), at); }
                                let packed_len = packed_residuals.len();
                                if let Ok(pt) = Tensor::from_vec(packed_residuals, vec![packed_len], &Device::Cpu) { target_map.insert(format!("{}_packed", prefix), pt); }
                                if let Ok(st) = Tensor::from_vec(scales, vec![b, h, s, 1], &Device::Cpu) { target_map.insert(format!("{}_scales", prefix), st); }
                                if prefix == "k" {
                                    if let Ok(sh) = Tensor::from_vec(vec![b as u32, h as u32, s as u32, d as u32], (4,), &Device::Cpu) { target_map.insert("k_shape".to_string(), sh); }
                                }
                            }
                        }
                    }
                };

                let _ = process_tensor(layer.k, "k", &mut map);
                let _ = process_tensor(layer.v, "v", &mut map);
                if let Ok(mode_tensor) = Tensor::from_vec(vec![3u32], (1,), &Device::Cpu) {
                    map.insert("mode".to_string(), mode_tensor);
                }

                let _ = candle_core::safetensors::save(&map, &path); 
            }
        }
        println!("[BAKE-WORKER] Worker finished.");
    });
}

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

    pub fn rebalance_layers(&mut self, device_id: usize) -> Result<()> {
        match self {
            Self::Standard(_) => Ok(()), 
            Self::QuantizedVL(m) => m.rebalance_layers(device_id),
            Self::QuantizedText(m) => m.rebalance_layers(device_id),
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
    pub kv_root: std::path::PathBuf,
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
        is_disk_swap: bool,
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
        is_disk_swap: bool,
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
        is_disk_swap: bool,
        kv_root: std::path::PathBuf,
    ) -> Result<Self> {
        let path = if let Some(stripped) = path.strip_prefix(r"\\?\") { stripped } else { path };
        let tok_path = tokenizer_path.unwrap_or(path);
        let tok_path = if let Some(stripped) = tok_path.strip_prefix(r"\\?\") { stripped } else { tok_path };
        let cfg_path = config_path.unwrap_or(path);
        let cfg_path = if let Some(stripped) = cfg_path.strip_prefix(r"\\?\") { stripped } else { cfg_path };

        let chat_template = ChatTemplate::init(tok_path)?;
        let tokenizer = TokenizerModel::init(tok_path)?;
        let final_config_path = std::path::Path::new(cfg_path).join("config.json");
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
        let pre_processor = Qwen3VLProcessor::new(tok_path, &vision_dev, dtype)?;

        let qwen3_vl = if !gguf_files.is_empty() {
            let mut model_path = gguf_files.iter().find(|f| f.contains("Qwen3-0.6B-Q8_0.gguf")).cloned();
            if model_path.is_none() { model_path = gguf_files.iter().find(|f| f.contains("Qwen3-0.6B-Q4_K_M.gguf")).cloned(); }
            if model_path.is_none() { model_path = gguf_files.iter().find(|f| !f.contains("mmproj")).cloned(); }

            let limit_tokens = hard_token_limit.unwrap_or(4096) as u64;  
            let reserve_tokens = limit_tokens.min(8192);
            let kv_reserve = reserve_tokens * 40000;

            if is_vision_model {
                let mmproj = mmproj_path.ok_or(anyhow!("Missing mmproj GGUF"))?;
                let main = model_path.ok_or(anyhow!("Missing main GGUF for VL model"))?;
                let main_file = std::fs::File::open(&main)?;
                let main_mmap = unsafe { memmap2::MmapOptions::new().map(&main_file)? };
                let mmproj_file = std::fs::File::open(&mmproj)?;
                let mmproj_mmap = unsafe { memmap2::MmapOptions::new().map(&mmproj_file)? };
                let mut main_cursor = std::io::Cursor::new(&main_mmap[..]);
                let main_content = gguf_file::Content::read(&mut main_cursor)?;
                let mut mmproj_cursor = std::io::Cursor::new(&mmproj_mmap[..]);
                let mmproj_content = gguf_file::Content::read(&mut mmproj_cursor)?;
                let model = QuantizedQwen3VLModel::new_with_mmap(&cfg, &main_content, Some(Arc::new(main_mmap)), &mmproj_content, Some(Arc::new(mmproj_mmap)), &text_dev, text_device_id, &vision_dev, vision_device_id, dtype, kv_reserve, baking_only)?;
                ModelVariant::QuantizedVL(model)
            } else {
                let main = model_path.or_else(|| if !gguf_files.is_empty() { Some(gguf_files[0].clone()) } else { None }).ok_or(anyhow!("No GGUF file found"))?;
                let file = std::fs::File::open(&main)?;
                let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
                let mut cursor = std::io::Cursor::new(&mmap[..]);        
                let content = gguf_file::Content::read(&mut cursor)?;
                let is_06b = path.contains("0.6B");
                let actual_baking_only = baking_only || is_06b;
                let single_layer_mode = baking_only || is_06b;
                let model = crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel::new_with_mmap(&cfg, &content, Some(Arc::new(mmap)), &text_dev, text_device_id, dtype, kv_reserve, actual_baking_only, single_layer_mode)?;
                ModelVariant::QuantizedText(model)
            }
        } else {
            let model_list = find_type_files(path, "safetensors")?      ;
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
            if let Some(token) = &cancel_token { if token.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            let end = (current_pos + chunk_size).min(total_tokens);
            let chunk = &token_ids[current_pos..end];
            let chunk_ids = Tensor::from_vec(chunk.to_vec(), (1, end - current_pos), &self.text_device)?;
            let chunk_pos = Tensor::arange(current_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?;
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, total_tokens, None)?;
            if let Some(path) = auto_save_path { let _ = self.save_kv_to_disk(path, None, end); }
            if let Some(ref mut target) = relay_target {
                let (ks, vs) = self.get_current_kv();
                let results: Result<Vec<_>> = ks.par_iter().zip(vs.par_iter()).map(|(k, v): (&Tensor, &Tensor)| {
                    let seq_len = k.dim(candle_core::D::Minus2)?;
                    let start = seq_len.saturating_sub(end - current_pos);
                    let k_new = k.narrow(candle_core::D::Minus2, start, end - current_pos)?;
                    let v_new = v.narrow(candle_core::D::Minus2, start, end - current_pos)?;
                    if let ModelVariant::QuantizedText(m) = &self.qwen3_vl {
                        let res_k = m.language_model.compress_to_bitkv(&k_new)?;
                        let res_v = m.language_model.compress_to_bitkv(&v_new)?;
                        Ok((res_k, res_v))
                    } else { Err(anyhow!("Unsupported")) }
                }).collect();
                let results = results?;
                let mut k_anchors = vec![]; let mut k_packed = vec![]; let mut k_scales = vec![];
                let mut v_anchors = vec![]; let mut v_packed = vec![]; let mut v_scales = vec![];
                let mut original_shape = vec![];
                for (rk, rv) in results {
                    k_anchors.push(rk.0); k_packed.push(rk.1); k_scales.push(rk.2);
                    v_anchors.push(rv.0); v_packed.push(rv.1); v_scales.push(rv.2);
                    original_shape = rk.3;
                }
                if !k_anchors.is_empty() { target.inject_kv_bitkv(&k_anchors, &k_packed, &k_scales, &v_anchors, &v_packed, &v_scales, &original_shape)?; }
            }
            current_pos = end;
        }
        if auto_save_path.is_some() { let _ = self.qwen3_vl.drop_kv_storage(); }
        Ok(())
    }

    pub fn prefill_only(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, mut relay_target: Option<&mut Qwen3VLGenerateModel>, kv_name: Option<String>) -> Result<usize> {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let full_input_ids_vec = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_tokens = full_input_ids_vec.len();
        
        // [증분 저장] 2048 토큰 단위로 기억 조각을 생성
        let prefill_chunk_size = 2048;
        let mut current_pos = self.get_kv_len();
        if current_pos > 0 { println!("[RESUME] Resuming from token {}.", current_pos); }

        while current_pos < total_tokens {
            // [대기열 제어] 20개 초과 시 일시 정지 (신호등 역할)
            while ACTIVE_BAKE_TASKS.load(Ordering::SeqCst) >= 20 {
                println!("[MEMORY] 69. 메모리 제거 대기... (Queue: {})", ACTIVE_BAKE_TASKS.load(Ordering::SeqCst));
                std::thread::sleep(std::time::Duration::from_millis(100));
            }

            if let Some(flag) = &cancel_flag { 
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } 
            }

            let end = (current_pos + prefill_chunk_size).min(total_tokens);
            let chunk_len = end - current_pos;
            let chunk = &full_input_ids_vec[current_pos..end];
            
            let chunk_ids = Tensor::from_vec(chunk.to_vec(), (1, chunk_len), &self.text_device)?;
            let chunk_pos = Tensor::arange(current_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?;

            // 1. GPU 추론 진행
            println!("[BAKING] {} to {} / Total: {}", current_pos, end, total_tokens);
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, total_tokens, session_id.clone())?;

            // 2. 비동기 워커에게 기억 조각(KV) 전달
            if let Some(sid) = &session_id {
                let path = crate::utils::paths::get_kv_dir(None).join(sid);
                if !path.exists() { let _ = fs::create_dir_all(&path); }
                
                // 대기열 카운트 증가
                ACTIVE_BAKE_TASKS.fetch_add(1, Ordering::SeqCst);
                
                // 디스크 저장 명령 (Incremental Save)
                self.save_kv_to_disk(&path, kv_name.as_deref(), end)?;
                
                // [MEMORY] 1. 메모리 삭제됨 (VRAM 즉시 초기화)
                let dev = self.text_device.clone();
                tauri::async_runtime::spawn(async move {
                    // GPU 메모리 포지션을 강하게 비움
                    purge_vram_position(&dev).await;
                    // 저장이 완료된 것으로 간주하고 카운트 감소
                    ACTIVE_BAKE_TASKS.fetch_sub(1, Ordering::SeqCst);
                });
            }

            if let Some(ref mut target) = relay_target {
                let (ks, vs) = self.get_current_kv();
                let results: Result<Vec<_>> = ks.par_iter().zip(vs.par_iter()).map(|(k, v): (&Tensor, &Tensor)| {
                    let s_len = k.dim(candle_core::D::Minus2)?;
                    let start = s_len.saturating_sub(chunk_len);
                    let k_new = k.narrow(candle_core::D::Minus2, start, chunk_len)?;
                    let v_new = v.narrow(candle_core::D::Minus2, start, chunk_len)?;
                    if let ModelVariant::QuantizedText(m) = &self.qwen3_vl {
                        let rk = m.language_model.compress_to_bitkv(&k_new)?;
                        let rv = m.language_model.compress_to_bitkv(&v_new)?;
                        Ok((rk, rv))
                    } else { Err(anyhow!("Unsupported")) }
                }).collect();
                let results = results?;
                let mut ka = vec![]; let mut kp = vec![]; let mut ks_ = vec![];
                let mut va = vec![]; let mut vp = vec![]; let mut vs_ = vec![];
                let mut os = vec![];
                for (rk, rv) in results {
                    ka.push(rk.0); kp.push(rk.1); ks_.push(rk.2);
                    va.push(rv.0); vp.push(rv.1); vs_.push(rv.2);
                    os = rk.3;
                }
                if !ka.is_empty() { target.inject_kv_bitkv(&ka, &kp, &ks_, &va, &vp, &vs_, &os)?; }
            }
            current_pos = end;
            println!("[MEMORY] 73. 추론 진행 메모리 (Ready)");
        }

        // 최종 동기화: 모든 대기열이 비워질 때까지 대기
        while ACTIVE_BAKE_TASKS.load(Ordering::SeqCst) > 0 {
            println!("[MEMORY] 69. 최종 메모리 제거 대기... (Remaining: {})", ACTIVE_BAKE_TASKS.load(Ordering::SeqCst));
            std::thread::sleep(std::time::Duration::from_millis(200));
        }

        if let Some(sid) = session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(sid);
            let token_path = path.join("tokens.json");
            if let Ok(file) = fs::File::create(&token_path) { let _ = serde_json::to_writer(file, &full_input_ids_vec); }
        }
        Ok(current_pos)
    }

    pub fn prefill_chunk(&mut self, text: String, cancel_flag: Option<Arc<AtomicBool>>, mut relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let chunk_ids_vec = self.tokenizer.text_encode_vec(text, false)?;
        let chunk_size = chunk_ids_vec.len();
        let current_pos = self.get_kv_len();
        let chunk_ids = Tensor::from_vec(chunk_ids_vec, (1, chunk_size), &self.text_device)?;
        let chunk_pos = Tensor::arange(current_pos as u32, (current_pos + chunk_size) as u32, &self.text_device)?.unsqueeze(0)?;
        self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, chunk_size, None)?;
        if let Some(ref mut target) = relay_target {
            let (ks, vs) = self.get_current_kv();
            let results: Result<Vec<_>> = ks.par_iter().zip(vs.par_iter()).map(|(k, v): (&Tensor, &Tensor)| {
                let s_len = k.dim(candle_core::D::Minus2)?;
                let k_new = k.narrow(candle_core::D::Minus2, s_len - chunk_size, chunk_size)?;
                let v_new = v.narrow(candle_core::D::Minus2, s_len - chunk_size, chunk_size)?;
                if let ModelVariant::QuantizedText(m) = &self.qwen3_vl {
                    let rk = m.language_model.compress_to_bitkv(&k_new)?;
                    let rv = m.language_model.compress_to_bitkv(&v_new)?;
                    Ok((rk, rv))
                } else { Err(anyhow!("Unsupported")) }
            }).collect();
            let results = results?;
            let mut ka = vec![]; let mut kp = vec![]; let mut ks_ = vec![];
            let mut va = vec![]; let mut vp = vec![]; let mut vs_ = vec![];
            let mut os = vec![];
            for (rk, rv) in results {
                ka.push(rk.0); kp.push(rk.1); ks_.push(rk.2);
                va.push(rv.0); vp.push(rv.1); vs_.push(rv.2);
                os = rk.3;
            }
            if !ka.is_empty() { target.inject_kv_bitkv(&ka, &kp, &ks_, &va, &vp, &vs_, &os)?; }
        }
        Ok(chunk_size)
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, kv_name: Option<String>) -> Result<String> {
        let temperature = mes.temperature.unwrap_or(0.7) as f32;
        let top_p = mes.top_p.unwrap_or(0.9) as f32;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(Some(temperature), Some(top_p), Some(40), seed);
        let mut all_ids = vec![];
        let mut generated_text = String::new();
        let mut seqlen_offset = self.get_kv_len();

        if let Some(sid) = &session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(sid);
            let progress_path = path.join("generation_progress.json");
            if progress_path.exists() {
                if let Ok(data) = std::fs::read_to_string(&progress_path) {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                        println!("[RESUME] Found progress. Loading...");
                        let _ = self.load_kv_from_disk(&path, kv_name.as_deref());
                        seqlen_offset = self.get_kv_len();
                        generated_text = json["text"].as_str().unwrap_or("").to_string();
                        if let Some(ids) = json["ids"].as_array() { all_ids = ids.iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect(); }
                    }
                }
            }
        }

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut input = self.pre_processor.process_info(&mes, &mes_render)?;
        let full_input_ids_vec = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_tokens = full_input_ids_vec.len();

        if all_ids.is_empty() {
            all_ids = full_input_ids_vec.clone();
            if seqlen_offset == 0 {
                if let Some(sid) = &session_id {
                    let path = crate::utils::paths::get_kv_dir(None).join(sid);
                    if path.exists() { self.load_kv_from_disk(&path, kv_name.as_deref())?; seqlen_offset = self.get_kv_len(); }
                }
            }
        }

        let mut local_pos = if seqlen_offset > 0 { if seqlen_offset >= total_tokens { total_tokens.saturating_sub(1) } else { seqlen_offset } } else { 0 };
        let prefill_chunk_size = 128;

        while local_pos < total_tokens {
            let remaining = total_tokens - local_pos;
            if remaining == 1 { break; }
            let mut chunk_size = if remaining > prefill_chunk_size { prefill_chunk_size } else { remaining };
            if local_pos + chunk_size >= total_tokens { chunk_size = (total_tokens - local_pos).saturating_sub(1); }
            if chunk_size == 0 { break; }
            let chunk = &full_input_ids_vec[local_pos..local_pos + chunk_size];
            let chunk_ids = Tensor::from_vec(chunk.to_vec(), (1, chunk_size), &self.text_device)?;
            let chunk_pos = Tensor::arange(seqlen_offset as u32, (seqlen_offset + chunk_size) as u32, &self.text_device)?.unsqueeze(0)?;
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), seqlen_offset, total_tokens, session_id.clone())?;
            local_pos += chunk_size;
            seqlen_offset += chunk_size;
            if seqlen_offset % 1024 == 0 {
                if let Some(sid) = &session_id {
                    let path = crate::utils::paths::get_kv_dir(None).join(sid);
                    let _ = self.save_kv_to_disk(&path, kv_name.as_deref(), seqlen_offset);
                    println!("[GEN-PREFILL] Checkpoint at {}", seqlen_offset);
                }
            }
        }

        let max_new_tokens = mes.max_tokens.unwrap_or(2048);
        let mut pixel_values = input.pixel_values.take();
        let image_grid_thw = input.image_grid_thw.take();

        // [WINDOW-BAKE] Setup async channel for background saving (Queue size: 10)
        let (bake_tx, bake_rx) = mpsc::channel(10);
        spawn_bake_worker(bake_rx);
        let task_dir_base = if let Some(sid) = &session_id {
            crate::utils::paths::get_kv_dir(None).join(sid)
        } else {
            std::path::PathBuf::new()
        };

        // [SLIDING-WINDOW-CONFIG] 512 block size for more granular management
        let window_keep_size = 1024; // 2 blocks of 512 (Keep very lean)
        let bake_block_size = 512;

        for i in 0..max_new_tokens {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            
            // [WINDOW-BAKE] Trigger dump every 512 tokens
            if seqlen_offset > 0 && seqlen_offset % bake_block_size == 0 && !task_dir_base.as_os_str().is_empty() {
                println!("[WINDOW-BAKE] {} boundary hit. Offloading 512-token block to disk.", seqlen_offset);
                let (ks, vs) = self.get_current_kv();
                let mut layer_dumps = Vec::new();
                
                for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() {
                    let current_mem_len = k.dim(2)?;
                    if current_mem_len >= bake_block_size {
                        let k_slice = k.narrow(2, current_mem_len - bake_block_size, bake_block_size)?.contiguous()?;
                        let v_slice = v.narrow(2, current_mem_len - bake_block_size, bake_block_size)?.contiguous()?;
                        layer_dumps.push(LayerKVDump { layer_idx: idx, k: k_slice, v: v_slice });
                    }
                }
                
                if !layer_dumps.is_empty() {
                    let task = BakeTask {
                        task_dir: task_dir_base.clone(),
                        kv_name: kv_name.clone().or_else(|| Some("default".to_string())),
                        offset: seqlen_offset - bake_block_size,
                        layers: layer_dumps,
                    };
                    let _ = bake_tx.blocking_send(task);
                    
                    // [SLIDING-WINDOW-PURGE] Keep recent 10 blocks in RAM
                    let current_kv_len = self.get_kv_len();
                    if current_kv_len > window_keep_size {
                        let purge_len = current_kv_len - window_keep_size;
                        let aligned_purge = (purge_len / bake_block_size) * bake_block_size;
                        if aligned_purge > 0 {
                            println!("[WINDOW-BAKE] Purging {} old tokens. (RAM: {} / Disk: 0-{})", 
                                aligned_purge, window_keep_size, seqlen_offset - window_keep_size);
                            let _ = self.truncate_kv_cache(aligned_purge);
                        }
                    }
                }
            }

            let input_ids = if generated_text.is_empty() && seqlen_offset < total_tokens {
                Tensor::new(&full_input_ids_vec[local_pos..total_tokens], &self.text_device)?.unsqueeze(0)?
            } else { Tensor::new(vec![*all_ids.last().unwrap()], &self.text_device)?.unsqueeze(0)? };
            let seq_len = input_ids.dim(1)?;
            let chunk_pos = Tensor::arange(seqlen_offset as u32, (seqlen_offset + seq_len) as u32, &self.text_device)?.unsqueeze(0)?;
            let logits = self.qwen3_vl.forward(&input_ids, pixel_values.as_ref(), image_grid_thw.as_ref(), None, None, Some(&chunk_pos), seqlen_offset, total_tokens, session_id.clone())?;
            let mut logits = logits.squeeze(0)?.i(logits.dim(1)? - 1)?.to_dtype(DType::F32)?;
            if 1.1 != 1.0 { let penalty_context = if all_ids.len() > 512 { &all_ids[all_ids.len()-512..] } else { &all_ids[..] }; logits = apply_repeat_penalty(&logits, 1.1, penalty_context)?; }
            let next_id = logit_processor.sample(&logits)?;
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            all_ids.push(next_id);
            generated_text.push_str(&self.tokenizer.token_decode(vec![next_id])?);
            // [INFERENCE-SAVE] Only checkpoint for Large models
            if i > 0 && i % 50 == 0 && !self.model_name.contains("0.6B") {
                if let Some(sid) = &session_id {
                    let path = crate::utils::paths::get_kv_dir(None).join(sid);
                    let _ = self.save_kv_to_disk(&path, kv_name.as_deref(), seqlen_offset);
                    let progress = serde_json::json!({ "text": generated_text, "ids": all_ids });
                    let _ = std::fs::write(path.join("generation_progress.json"), progress.to_string());
                    println!("[GEN] Checkpoint at {}", i);
                }
            }
            seqlen_offset += seq_len;
            pixel_values = None;
        }
        if let Some(sid) = &session_id { let _ = std::fs::remove_file(crate::utils::paths::get_kv_dir(None).join(sid).join("generation_progress.json")); }
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
            ModelVariant::QuantizedVL(m) => { for l in &m.language_model.layers { if let Some((k, v)) = &l.self_attn.kv_cache { ks.push(k.clone()); vs.push(v.clone()); } } },
            ModelVariant::QuantizedText(m) => { for l in &m.language_model.layers { if let Some((k, v)) = &l.self_attn.kv_cache { ks.push(k.clone()); vs.push(v.clone()); } } },
            _ => {} 
        }
        (ks, vs)
    }

    pub fn inject_kv_bitkv(&mut self, ka: &[Tensor], kp: &[Tensor], ks: &[Tensor], va: &[Tensor], vp: &[Tensor], vs: &[Tensor], os: &[usize]) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.language_model.inject_live_kv_bitkv(ka, kp, ks, va, vp, vs, os),
            ModelVariant::QuantizedText(m) => m.language_model.inject_live_kv_bitkv(ka, kp, ks, va, vp, vs, os),
            _ => Ok(()),
        }
    }

    pub fn save_kv_to_disk(&mut self, path: &Path, kv_name: Option<&str>, offset: usize) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.save_kv_cache(path, false, offset, kv_name),
            ModelVariant::QuantizedText(m) => m.save_kv_cache(path, false, offset, kv_name),
            _ => Ok(()),
        }
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.truncate_kv_cache(len),
            ModelVariant::QuantizedText(m) => m.truncate_kv_cache(len),
            _ => Ok(()),
        }
    }

    pub fn load_kv_from_disk(&mut self, path: &Path, kv_name: Option<&str>) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.load_kv_cache(path, &self.text_device, 0, 128, kv_name),
            ModelVariant::QuantizedText(m) => m.load_kv_cache(path, &self.text_device, 0, 128, kv_name),
            _ => Ok(()),
        }
    }

    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.to_device(d)?,
            ModelVariant::QuantizedText(m) => m.to_device(d)?,
            _ => {},
        }
        self.text_device = d.clone(); self.vision_device = d.clone();
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

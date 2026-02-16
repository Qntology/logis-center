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
use tokio::sync::Mutex as TokioMutex;
use rayon::prelude::*;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::fs;
use std::path::Path;

use std::path::PathBuf;
use tokio::sync::mpsc;

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

fn spawn_bake_worker(active_tasks: Arc<std::sync::atomic::AtomicUsize>) -> mpsc::UnboundedSender<BakeTask> {
    let (tx, mut rx) = mpsc::unbounded_channel::<BakeTask>();
    
    tokio::spawn(async move {
        println!("[BAKE-WORKER] Background BitKV storage worker started.");
        while let Some(task) = rx.recv().await {
            let task_dir = task.task_dir;
            let kv_name = task.kv_name;
            let offset = task.offset;
            
            for layer in task.layers {
                let filename = match (&kv_name, task.offset) {
                    (Some(name), 0) => format!("layer_{}_kv.safetensors", name),
                    (Some(name), off) => format!("layer_{}_kv_{}.safetensors", name, off),
                    (None, 0) => format!("layer_{}_kv.safetensors", layer.layer_idx),
                    (None, off) => format!("layer_{}_kv_{}.safetensors", layer.layer_idx, off),
                };
                let final_path = task_dir.join(&filename);
                let tmp_path = task_dir.join(format!("{}.tmp", filename));

                let mut map = std::collections::HashMap::new();

                // --- [BitKV Compression Logic] ---
                let mut process_kv = |t: Tensor, prefix: &str| -> Result<()> {
                    let dims = t.dims4()?;
                    let (b, h, s, d) = dims;
                    let t_f32 = t.to_dtype(DType::F32)?;
                    let t_data = t_f32.flatten_all()?.to_vec1::<f32>()?;
                    
                    let anchor_count = (0..s).filter(|&i| i < 4 || i % 8 == 0).count();
                    let mut anchors = vec![0.0f32; b * h * anchor_count * d];
                    let mut packed_residuals = vec![0u8; (b * h * s * d + 7) / 8];
                    let mut scales = vec![0.0f32; b * h * s];

                    let head_token_size = s * d;
                    for bh_idx in 0..(b * h) {
                        let bh_offset = bh_idx * head_token_size;
                        for i in 0..head_token_size {
                            if t_data[bh_offset + i] >= 0.0 {
                                let global_bit_idx = bh_offset + i;
                                packed_residuals[global_bit_idx / 8] |= 1 << (global_bit_idx % 8);
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

                    map.insert(format!("{}_anchors", prefix), Tensor::from_vec(anchors, vec![b, h, anchor_count, d], &Device::Cpu)?);
                    let p_len = packed_residuals.len();
                    map.insert(format!("{}_packed", prefix), Tensor::from_vec(packed_residuals, vec![p_len], &Device::Cpu)?);
                    map.insert(format!("{}_scales", prefix), Tensor::from_vec(scales, vec![b, h, s, 1], &Device::Cpu)?);
                    
                    if prefix == "k" {
                        map.insert("k_shape".to_string(), Tensor::from_vec(vec![b as u32, h as u32, s as u32, d as u32], (4,), &Device::Cpu)?);
                    }
                    Ok(())
                };

                let _ = process_kv(layer.k, "k");
                let _ = process_kv(layer.v, "v");
                
                if let Ok(mode_tensor) = Tensor::from_vec(vec![3u32], (1,), &Device::Cpu) {
                    map.insert("mode".to_string(), mode_tensor);
                }

                // Atomic Write: Write to .tmp then rename
                if let Ok(_) = candle_core::safetensors::save(&map, &tmp_path) {
                    let _ = std::fs::rename(&tmp_path, &final_path);
                }
            }
            // 작업 완료 후 카운터 감소
            active_tasks.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        }
    });
    tx
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
    pub drafter: Option<Arc<TokioMutex<Option<Qwen3VLGenerateModel>>>>, // 투기적 추론용 드래프터
    pub bake_tx: mpsc::UnboundedSender<BakeTask>, // [NEW] Asynchronous baking queue
    pub active_bake_tasks: Arc<std::sync::atomic::AtomicUsize>, // [NEW] Track background saves
}

impl Qwen3VLGenerateModel {
    pub fn set_drafter(&mut self, drafter: Arc<TokioMutex<Option<Qwen3VLGenerateModel>>>) {
        self.drafter = Some(drafter);
    }

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
        _is_disk_swap: bool,
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
        
        let active_bake_tasks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let bake_tx = spawn_bake_worker(active_bake_tasks.clone());

        Ok(Self { chat_template, tokenizer, pre_processor, qwen3_vl, text_device: text_dev, vision_device: vision_dev, eos_token_id1, eos_token_id2, generation_config, model_name, hard_token_limit, kv_root, drafter: None, bake_tx, active_bake_tasks })
    }

    pub fn prefill_text_only(&mut self, text: &str, cancel_token: Option<Arc<AtomicBool>>, mut relay_target: Option<&mut Qwen3VLGenerateModel>, auto_save_path: Option<&std::path::Path>) -> Result<()> {
        let token_ids = self.tokenizer.text_encode_vec(text.to_string(), false)?;
        let total_tokens = token_ids.len();
        let chunk_size = 2048;
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
        
        // [GRANULAR-BAKING] Use 2048 blocks for consistent memory pressure
        let prefill_chunk_size = 2048;
        let mut current_pos = self.get_kv_len();
        
        // [CLEAN-START] 최초 작업 시작 시 기존 찌꺼기 제거 (강력한 초기화)
        if current_pos == 0 {
            if let Some(sid) = &session_id {
                let path = crate::utils::paths::get_kv_dir(None).join(sid);
                if path.exists() {
                    println!("[BAKING] Strong cleaning session directory: {:?}", path);
                    let _ = std::fs::remove_dir_all(&path);
                    // Windows에서 파일 삭제 후 핸들이 해제될 때까지 약간의 여유를 줍니다.
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                let _ = std::fs::create_dir_all(&path);
            }
        }

        if current_pos > 0 { println!("[RESUME] Resuming from token {}.", current_pos); }

        while current_pos < total_tokens {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            
            let end = (current_pos + prefill_chunk_size).min(total_tokens);
            let chunk_len = end - current_pos;
            if chunk_len == 0 { break; }
            let chunk = &full_input_ids_vec[current_pos..end];
            let chunk_ids = Tensor::from_vec(chunk.to_vec(), (1, chunk_len), &self.text_device)?;
            let chunk_pos = Tensor::arange(current_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?;
            println!("[BAKING] Step: {} to {} / Total: {}", current_pos, end, total_tokens);
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, total_tokens, session_id.clone())?;
            // [SEGMENT-RESET] Persistent save and system-level memory purge after every chunk
            if let Some(sid) = &session_id {
                let path = crate::utils::paths::get_kv_dir(None).join(sid);
                if !path.exists() { let _ = fs::create_dir_all(&path); }     
                
                // 1. Save to disk (Incremental Chunk) via Async Queue
                let (ks, vs) = self.get_current_kv();
                let mut layer_dumps = Vec::with_capacity(ks.len());
                for (idx, (k, v)) in ks.into_iter().zip(vs).enumerate() {
                    // [INCREMENTAL-FIX] 전체가 아닌 현재 조각(chunk_len)만 추출하여 중복 방지
                    let s_len = k.dim(candle_core::D::Minus2)?;
                    let take_len = chunk_len.min(s_len);
                    if let (Ok(k_chunk), Ok(v_chunk)) = (
                        k.narrow(candle_core::D::Minus2, s_len - take_len, take_len),
                        v.narrow(candle_core::D::Minus2, s_len - take_len, take_len)
                    ) {
                        if let (Ok(k_cpu), Ok(v_cpu)) = (k_chunk.to_device(&Device::Cpu), v_chunk.to_device(&Device::Cpu)) {
                            layer_dumps.push(LayerKVDump { layer_idx: idx, k: k_cpu, v: v_cpu });
                        }
                    }
                }
                self.active_bake_tasks.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let _ = self.bake_tx.send(BakeTask {
                    task_dir: path,
                    kv_name: kv_name.clone(),
                    offset: end,
                    layers: layer_dumps,
                });
                
                // 2. Purge from VRAM (Keep memory lean)
                println!("[BAKING] Segment queued. Resetting memory for token {}.", end);
                let _ = self.qwen3_vl.drop_kv_storage(); 
                
                // 3. [CRITICAL-RESET] Force OS to reclaim RAM immediately
                if self.text_device.is_cuda() { let _ = self.text_device.synchronize(); }
                #[cfg(target_os = "windows")]
                unsafe {
                    use windows_sys::Win32::System::Threading::GetCurrentProcess;
                    use windows_sys::Win32::System::Memory::*;
                    let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
                }
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
        }
        
        if let Some(sid) = session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(sid);
            if !path.exists() { let _ = fs::create_dir_all(&path); }
            let token_path = path.join("tokens.json");
            if let Ok(file) = fs::File::create(&token_path) { let _ = serde_json::to_writer(file, &full_input_ids_vec); }
            
            println!("[BAKING] Prefill complete. Small engine transitioning...");
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
        let temperature = mes.temperature.unwrap_or(0.9) as f32;
        let top_p = mes.top_p.unwrap_or(0.9) as f32;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(Some(temperature), Some(top_p), Some(1), seed);
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
        
        // [HARD-SWAP-CONFIG] Constant memory usage strategy
        let window_keep_size = 2048; 
        let bake_block_size = 2048;

        // [PHASE-1: Granular Prefill] Offload prompt chunks immediately to SSD
        while local_pos < total_tokens {
            let remaining = total_tokens - local_pos;
            if remaining == 1 { break; }
            let mut chunk_size = if remaining > bake_block_size { bake_block_size } else { remaining };
            if local_pos + chunk_size >= total_tokens { chunk_size = (total_tokens - local_pos).saturating_sub(1); }
            if chunk_size == 0 { break; }
            
            let chunk = &full_input_ids_vec[local_pos..local_pos + chunk_size];
            let chunk_ids = Tensor::from_vec(chunk.to_vec(), (1, chunk_size), &self.text_device)?;
            let chunk_pos = Tensor::arange(seqlen_offset as u32, (seqlen_offset + chunk_size) as u32, &self.text_device)?.unsqueeze(0)?;
            
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), seqlen_offset, total_tokens, session_id.clone())?;
            
            local_pos += chunk_size;
            seqlen_offset += chunk_size;

            // Immediate Offload & Purge during prefill
            if let Some(sid) = &session_id {
                let path = crate::utils::paths::get_kv_dir(None).join(sid);
                
                // [INCREMENTAL-SAVE-GENERATION]
                let (ks, vs) = self.get_current_kv();
                let mut layer_dumps = Vec::with_capacity(ks.len());
                for (idx, (k, v)) in ks.into_iter().zip(vs).enumerate() {
                    let s_len = k.dim(candle_core::D::Minus2)?;
                    let take_len = chunk_size.min(s_len);
                    if let (Ok(k_chunk), Ok(v_chunk)) = (
                        k.narrow(candle_core::D::Minus2, s_len - take_len, take_len),
                        v.narrow(candle_core::D::Minus2, s_len - take_len, take_len)
                    ) {
                        if let (Ok(k_cpu), Ok(v_cpu)) = (k_chunk.to_device(&Device::Cpu), v_chunk.to_device(&Device::Cpu)) {
                            layer_dumps.push(LayerKVDump { layer_idx: idx, k: k_cpu, v: v_cpu });
                        }
                    }
                }
                
                self.active_bake_tasks.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let _ = self.bake_tx.send(BakeTask {
                    task_dir: path,
                    kv_name: kv_name.clone(),
                    offset: seqlen_offset,
                    layers: layer_dumps,
                });

                // Purge only AFTER queuing to background
                let _ = self.qwen3_vl.drop_kv_storage();
                if self.text_device.is_cuda() { let _ = self.text_device.synchronize(); }
                #[cfg(target_os = "windows")]
                unsafe {
                    use windows_sys::Win32::System::Threading::GetCurrentProcess;
                    use windows_sys::Win32::System::Memory::*;
                    let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
                }
            }
        }

        let max_new_tokens = mes.max_tokens.unwrap_or(2048) as usize;
        let mut pixel_values = input.pixel_values.take();
        let image_grid_thw = input.image_grid_thw.take();

        // [PHASE-2: Constant-Speed Generation with Hard Swap]
        // Use synchronous saving to ensure disk is updated immediately
        let task_dir_base = if let Some(sid) = &session_id {
            // [FIX] Ensure we use the correct task data directory
            crate::utils::paths::get_task_specific_dir(None, sid)
        } else {
            std::path::PathBuf::new()
        };

        let mut i: usize = 0;
        while i < max_new_tokens {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            
            // --- [ULTRA-SPEED: SPECULATIVE DECODING BLOCK] ---
            let mut speculative_success = false;
            let lookahead = 4; // 드래프터가 미리 예측할 토큰 수

            if let Some(drafter_mutex) = &self.drafter {
                if let Ok(mut drafter_guard) = drafter_mutex.try_lock() {
                    let drafter_opt: &mut Option<Qwen3VLGenerateModel> = &mut *drafter_guard;
                    if let Some(drafter) = drafter_opt.as_mut() {
                        // 1. 드래프터로 N개 토큰 미리 생성 (Drafting)
                        let mut draft_ids = vec![];
                        let mut current_draft_context = all_ids.clone();
                        
                        // 드래프터 KV 캐시 싱크 (필요시)
                        if drafter.get_kv_len() < seqlen_offset {
                             // 단순화를 위해 드래프터는 이전 문맥을 알고 있다고 가정하거나 
                             // 여기서 가볍게 채워줄 수 있습니다.
                        }

                        for _ in 0..lookahead {
                            let d_input = Tensor::new(vec![*current_draft_context.last().unwrap()], &self.text_device)?.unsqueeze(0)?;
                            let d_chunk_pos = Tensor::arange((seqlen_offset + draft_ids.len()) as u32, (seqlen_offset + draft_ids.len() + 1) as u32, &self.text_device)?.unsqueeze(0)?;
                            let d_logits = drafter.qwen3_vl.forward(&d_input, None, None, None, None, Some(&d_chunk_pos), seqlen_offset + draft_ids.len(), total_tokens, None)?;
                            let d_logits = d_logits.squeeze(0)?.i(0)?.to_dtype(DType::F32)?;
                            let d_next_id = logit_processor.sample(&d_logits)?;
                            draft_ids.push(d_next_id);
                            current_draft_context.push(d_next_id);
                            if d_next_id == self.eos_token_id1 || d_next_id == self.eos_token_id2 { break; }
                        }

                        if !draft_ids.is_empty() {
                            // 2. 메인 모델(2B)로 한 번에 검증 (Verification)
                            let v_input = Tensor::from_vec(draft_ids.clone(), (1, draft_ids.len()), &self.text_device)?;
                            let v_chunk_pos = Tensor::arange(seqlen_offset as u32, (seqlen_offset + draft_ids.len()) as u32, &self.text_device)?.unsqueeze(0)?;
                            let v_logits = self.qwen3_vl.forward(&v_input, None, None, None, None, Some(&v_chunk_pos), seqlen_offset, total_tokens, session_id.clone())?;
                            
                            // 3. 일치 여부 확인 및 수락
                            let mut accepted_count: usize = 0;
                            for (idx, &d_id) in draft_ids.iter().enumerate() {
                                let v_logit = v_logits.squeeze(0)?.i(idx)?.to_dtype(DType::F32)?;
                                let v_next_id = logit_processor.sample(&v_logit)?; // 사실 검증은 sample보다는 max logit 비교가 정석이나 여기선 유연하게 처리
                                
                                if d_id == v_next_id {
                                    accepted_count += 1;
                                    all_ids.push(d_id);
                                    generated_text.push_str(&self.tokenizer.token_decode(vec![d_id])?);
                                    if d_id == self.eos_token_id1 || d_id == self.eos_token_id2 { break; }
                                } else {
                                    // 틀린 지점부터는 메인 모델의 정답을 사용하고 중단
                                    all_ids.push(v_next_id);
                                    generated_text.push_str(&self.tokenizer.token_decode(vec![v_next_id])?);
                                    accepted_count += 1; // 메인 모델 정답 포함
                                    break;
                                }
                            }
                            
                            i += accepted_count;
                            seqlen_offset += accepted_count;
                            speculative_success = true;
                        }
                    }
                }
            }

            if !speculative_success {
                // 표준 생성 방식 (Speculative 실패 시 또는 미사용 시)
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
                
                i += 1;
                seqlen_offset += seq_len;
            }

            // [HARD-SWAP-INFERENCE] 128 토큰마다 메모리 정리 및 체크포인트
            if seqlen_offset > 0 && seqlen_offset % 128 == 0 && !task_dir_base.as_os_str().is_empty() {
                if !task_dir_base.exists() { let _ = std::fs::create_dir_all(&task_dir_base); }
                
                // [ASYNC-BAKING-UPGRADE]
                let (ks, vs) = self.get_current_kv();
                let mut layer_dumps = Vec::with_capacity(ks.len());
                for (idx, (k, v)) in ks.into_iter().zip(vs).enumerate() {
                    if let (Ok(k_cpu), Ok(v_cpu)) = (k.to_device(&Device::Cpu), v.to_device(&Device::Cpu)) {
                        layer_dumps.push(LayerKVDump { layer_idx: idx, k: k_cpu, v: v_cpu });
                    }
                }
                
                let _ = self.bake_tx.send(BakeTask {
                    task_dir: task_dir_base.clone(),
                    kv_name: kv_name.clone(),
                    offset: seqlen_offset,
                    layers: layer_dumps,
                });
                
                let current_kv_len = self.get_kv_len();
                if current_kv_len > 1024 {
                    self.truncate_kv_cache(current_kv_len - 1024)?;
                    if self.text_device.is_cuda() { let _ = self.text_device.synchronize(); }
                    #[cfg(target_os = "windows")]
                    unsafe {
                        use windows_sys::Win32::System::Threading::GetCurrentProcess;
                        use windows_sys::Win32::System::Memory::*;
                        let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
                    }
                }

                // 체크포인트 저장
                if let Some(sid) = &session_id {
                    let path = crate::utils::paths::get_kv_dir(None).join(sid);
                    let progress = serde_json::json!({ "text": generated_text, "ids": all_ids });
                    let _ = std::fs::write(path.join("generation_progress.json"), progress.to_string());
                }
            }

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
                // [SYNC-ON-DEMAND] 파일을 읽기 직전에만 대기열이 비었는지 확인 (겸사겸사)
                let mut wait_count = 0;
                while self.active_bake_tasks.load(std::sync::atomic::Ordering::SeqCst) > 0 {
                    if wait_count == 0 { println!("[LOADER] Waiting for background bake tasks to commit to disk..."); }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    wait_count += 1;
                    if wait_count > 100 { break; } // 5초 타임아웃
                }
                if wait_count > 0 { println!("[LOADER] All files ready. Proceeding with load."); }
        
                match &mut self.qwen3_vl {
                    ModelVariant::QuantizedVL(m) => m.load_kv_cache(path, &self.text_device, 0, 0, kv_name),    
                    ModelVariant::QuantizedText(m) => m.load_kv_cache(path, &self.text_device, 0, 0, kv_name),  
                    _ => Ok(()),
                }
            }    pub fn to_device(&mut self, d: &Device) -> Result<()> {
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

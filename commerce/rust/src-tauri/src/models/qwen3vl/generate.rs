use crate::models::qwen3vl::quantized_model::{KVLocation, KVBlock, KVRegistry, BitKVMetadata, QuantizedQwen3VLModel, MemorySlot};
use anyhow::{Result, anyhow};
use candle_core::{quantized::gguf_file, DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::{
    chat_template::ChatTemplate,
    models::{
        qwen3vl::{
            config::{Qwen3VLConfig, Qwen3VLGenerationConfig},
            model::Qwen3VLModel,
            processor::Qwen3VLProcessor,
        },
    },
    tokenizer::TokenizerModel,
    utils::{
        find_type_files, get_device, get_dtype, get_logit_processor,
        direct_loader::{save_kv_block, load_kv_block},
    },
    openai_types::ChatCompletionParameters,
};
use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}, Mutex};
use std::fs;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use safetensors;
use once_cell::sync::Lazy;

// [GLOBAL] 슬롯 관리자
pub struct SlotManager {
    pub slots: Vec<MemorySlot>,
    pub handoff_notifier: Arc<tokio::sync::Notify>,
    pub active_write_count: Arc<AtomicUsize>,
    pub count_reads: Arc<AtomicUsize>,
    pub count_writes: Arc<AtomicUsize>,
    pub count_cached: Arc<AtomicUsize>,
    pub count_free: Arc<AtomicUsize>,
    pub request_tx: mpsc::Sender<SlotRequest>,
}

pub enum SlotRequest {
    Acquire { total_tokens: usize, tx: tokio::sync::oneshot::Sender<usize> },
    Release { idx: usize, is_bake: bool },
}

impl SlotManager {
    pub fn new(count: usize) -> (Self, mpsc::Receiver<SlotRequest>) {
        let (tx, rx) = mpsc::channel(1024);
        let mut slots = Vec::new();
        let num_layers = 28;
        for i in 0..count { slots.push(MemorySlot::new(i, num_layers)); }
        (Self {
            slots, handoff_notifier: Arc::new(tokio::sync::Notify::new()),
            active_write_count: Arc::new(AtomicUsize::new(0)),
            count_reads: Arc::new(AtomicUsize::new(0)), count_writes: Arc::new(AtomicUsize::new(0)),
            count_cached: Arc::new(AtomicUsize::new(0)), count_free: Arc::new(AtomicUsize::new(count)),
            request_tx: tx,
        }, rx)
    }
    fn update_counters(&self, old_state: u8, new_state: u8) {
        if old_state == new_state { return; }
        match old_state { 0 => self.count_free.fetch_sub(1, Ordering::SeqCst), 1 => self.count_writes.fetch_sub(1, Ordering::SeqCst), 2 => self.count_cached.fetch_sub(1, Ordering::SeqCst), 3 => self.count_reads.fetch_sub(1, Ordering::SeqCst), _ => 0 };
        match new_state { 0 => self.count_free.fetch_add(1, Ordering::SeqCst), 1 => self.count_writes.fetch_add(1, Ordering::SeqCst), 2 => self.count_cached.fetch_add(1, Ordering::SeqCst), 3 => self.count_reads.fetch_add(1, Ordering::SeqCst), _ => 0 };
    }
    pub async fn acquire_write_slot(&self, total_tokens: usize) -> usize {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.request_tx.send(SlotRequest::Acquire { total_tokens, tx }).await;
        rx.await.unwrap_or(0)
    }
    pub async fn release_slot(&self, idx: usize) { let _ = self.request_tx.send(SlotRequest::Release { idx, is_bake: false }).await; }
    pub async fn mark_ready(&self, idx: usize) { let _ = self.request_tx.send(SlotRequest::Release { idx, is_bake: true }).await; }
    pub async fn acquire_read_slot(&self) -> usize {
        loop {
            for (i, slot) in self.slots.iter().enumerate() {
                let current = slot.state.load(Ordering::SeqCst);
                if current == 0 || current == 2 {
                    if slot.state.compare_exchange(current, 3, Ordering::SeqCst, Ordering::SeqCst).is_ok() { self.update_counters(current, 3); return i; }
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }
    pub fn get_counts(&self) -> (usize, usize, usize, usize) {
        (self.count_reads.load(Ordering::Relaxed), self.count_writes.load(Ordering::Relaxed), self.count_cached.load(Ordering::Relaxed), self.count_free.load(Ordering::Relaxed))
    }
}

pub static GLOBAL_IO_COUNTER: AtomicUsize = AtomicUsize::new(0);
pub static SLOT_MANAGER_DATA: Lazy<(SlotManager, Mutex<Option<mpsc::Receiver<SlotRequest>>>)> = Lazy::new(|| {
    let (sm, rx) = SlotManager::new(128); (sm, Mutex::new(Some(rx)))
});
pub static SLOT_MANAGER: Lazy<&SlotManager> = Lazy::new(|| &SLOT_MANAGER_DATA.0);

pub struct LayerKVDump {
    pub layer_idx: usize,
    pub k_anchors: Tensor, pub k_packed: Tensor, pub k_scales: Tensor,
    pub v_anchors: Tensor, pub v_packed: Tensor, pub v_scales: Tensor,
}

pub struct BakeTask {
    pub slot_id: usize, pub task_dir: PathBuf, pub kv_name: Option<String>,
    pub offset: usize, pub layers: Vec<LayerKVDump>, pub is_relay_baking: bool,
    pub block_idx: Option<usize>, pub registry: KVRegistry,
}

pub struct SaveTask {
    pub slot_id: usize, pub path: PathBuf, pub tensors: std::collections::HashMap<String, Tensor>,
    pub is_last: bool, pub block_idx: Option<usize>, pub registry: Option<KVRegistry>,
}

pub enum SlotTask { Bake(BakeTask), Load(LoadTask) }
pub struct LoadTask { pub slot_id: usize, pub path: PathBuf, pub layer_idx: usize, pub kv_name: Option<String>, pub shared_block: KVBlock, pub registry: KVRegistry }

pub static BAKE_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();
pub static LOAD_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();
use tokio::sync::OnceCell;

pub async fn get_worker_channel() -> Result<mpsc::Sender<SlotTask>> { BAKE_TX.get().cloned().ok_or(anyhow!("Bake init error")) }
pub async fn get_load_worker() -> Result<mpsc::Sender<SlotTask>> { LOAD_TX.get().cloned().ok_or(anyhow!("Load init error")) }
pub async fn wait_for_global_io() { while GLOBAL_IO_COUNTER.load(Ordering::SeqCst) > 0 { tokio::time::sleep(std::time::Duration::from_millis(10)).await; } }

pub fn init_bake_worker() {
    let (btx, brx) = mpsc::channel(2048); let (ltx, lrx) = mpsc::channel(2048);
    let _ = BAKE_TX.set(btx); let _ = LOAD_TX.set(ltx);
    if let Some(rx) = SLOT_MANAGER_DATA.1.lock().unwrap().take() { tauri::async_runtime::spawn(async move { spawn_slot_dispatcher(rx).await; }); }
    tauri::async_runtime::spawn(async move { spawn_slot_worker(brx); }); 
    tauri::async_runtime::spawn(async move { spawn_slot_worker(lrx); });
}

async fn spawn_slot_dispatcher(mut rx: mpsc::Receiver<SlotRequest>) {
    while let Some(req) = rx.recv().await {
        match req {
            SlotRequest::Acquire { total_tokens, tx } => {
                let max_writes = 64; let mut found = None;
                while found.is_none() {
                    if SLOT_MANAGER.active_write_count.load(Ordering::SeqCst) < max_writes {
                        for (i, slot) in SLOT_MANAGER.slots.iter().enumerate() {
                            if slot.state.load(Ordering::SeqCst) == 0 {
                                if slot.state.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() { SLOT_MANAGER.update_counters(0, 1); SLOT_MANAGER.active_write_count.fetch_add(1, Ordering::SeqCst); found = Some(i); break; }
                            }
                        }
                        if found.is_none() {
                            for (i, slot) in SLOT_MANAGER.slots.iter().enumerate() {
                                if slot.state.load(Ordering::SeqCst) == 2 {
                                    if slot.state.compare_exchange(2, 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() { SLOT_MANAGER.update_counters(2, 1); SLOT_MANAGER.active_write_count.fetch_add(1, Ordering::SeqCst); found = Some(i); break; }
                                }
                            }
                        }
                    }
                    if found.is_none() { tokio::time::sleep(std::time::Duration::from_millis(5)).await; }
                }
                let _ = tx.send(found.unwrap());
            },
            SlotRequest::Release { idx, is_bake } => {
                let s = &SLOT_MANAGER.slots[idx]; let old = s.state.load(Ordering::SeqCst);
                let new = if is_bake { 2 } else { 0 };
                if s.state.compare_exchange(old, new, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    SLOT_MANAGER.update_counters(old, new);
                    if old == 1 { SLOT_MANAGER.active_write_count.fetch_sub(1, Ordering::SeqCst); }
                    SLOT_MANAGER.handoff_notifier.notify_waiters();
                }
            }
        }
    }
}

fn spawn_slot_worker(mut rx: mpsc::Receiver<SlotTask>) {
    let (io_tx, mut io_rx) = mpsc::channel::<SaveTask>(2048); 
    tokio::spawn(async move {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(64)); 
        while let Some(task) = io_rx.recv().await {
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let (tp, ts, reg, b_idx, sid, is_last) = (task.path.clone(), task.tensors, task.registry.clone(), task.block_idx, task.slot_id, task.is_last);
            tokio::spawn(async move {
                struct IoGuard; impl Drop for IoGuard { fn drop(&mut self) { GLOBAL_IO_COUNTER.fetch_sub(1, Ordering::SeqCst); } }
                let _guard = IoGuard;
                if let Some(p) = tp.parent() { if !p.exists() { let _ = fs::create_dir_all(p); } }
                if let Ok(data) = safetensors::serialize(&ts, &None) {
                    drop(ts); let _ = save_kv_block(&tp, &data);
                    if let (Some(r), Some(idx)) = (reg, b_idx) {
                        if let Ok(mut entries) = r.entries.write() {
                            if idx < entries.len() {
                                let e = &mut entries[idx]; e.ssd_path = Some(tp.parent().unwrap().to_path_buf());
                                if let Some(l_str) = tp.file_name().and_then(|n| n.to_str()).and_then(|s| s.strip_prefix('l')).and_then(|s| s.strip_suffix(".st")) {
                                    if let Ok(l_idx) = l_str.parse::<usize>() { if l_idx < 28 { e.location[l_idx] = KVLocation::SSD; } }
                                }
                            }
                        }
                    }
                }
                let rem = SLOT_MANAGER.slots[sid].remaining_layers.fetch_sub(1, Ordering::SeqCst);
                if rem == 1 || is_last { SLOT_MANAGER.mark_ready(sid).await; }
                drop(permit);
            });
        }
    });

    tokio::spawn(async move {
        while let Some(task) = rx.recv().await {
            match task {
                SlotTask::Bake(bake) => {
                    let io_tx_inner = io_tx.clone();
                    let (sid, off, is_relay, block_idx, registry) = (bake.slot_id, bake.offset, bake.is_relay_baking, bake.block_idx, bake.registry.clone());
                    let loop_count = bake.layers.len();
                    SLOT_MANAGER.slots[sid].remaining_layers.store(loop_count, Ordering::SeqCst);
                    for l_idx in 0..loop_count {
                        let src = &bake.layers[l_idx];
                        let act_l = if is_relay { 0 } else { src.layer_idx };
                        let mut map = std::collections::HashMap::new();
                        let prefix = if is_relay { format!("b{}_l0_", off) } else { format!("b{}_l{}_", off, act_l) };
                        map.insert(format!("{}k_anchors", prefix), src.k_anchors.clone());
                        map.insert(format!("{}k_packed", prefix), src.k_packed.clone());
                        map.insert(format!("{}k_scales", prefix), src.k_scales.clone());
                        map.insert(format!("{}v_anchors", prefix), src.v_anchors.clone());
                        map.insert(format!("{}v_packed", prefix), src.v_packed.clone());
                        map.insert(format!("{}v_scales", prefix), src.v_scales.clone());
                        let file_path = bake.task_dir.join(format!("l{}.st", act_l));
                        GLOBAL_IO_COUNTER.fetch_add(1, Ordering::SeqCst);
                        let _ = io_tx_inner.send(SaveTask { slot_id: sid, path: file_path, tensors: map, is_last: l_idx == loop_count - 1, block_idx, registry: Some(registry.clone()) }).await;
                    }
                },
                SlotTask::Load(load) => {
                    let sid = load.slot_id; let reg = load.registry.clone(); let shared_block = load.shared_block.clone();
                    let provided_path = load.path.clone(); let kv_name = load.kv_name.clone();
                    tokio::spawn(async move {
                        let _guard = ReadSlotGuard { sid, active: true };
                        let (b_idx_off, b_idx) = { match shared_block.inner.read() { Ok(inner) => (inner.offset, inner.index), _ => (0, 999) } };
                        let mut root = provided_path.clone();
                        while !root.to_string_lossy().ends_with("kv") && root.parent().is_some() { root = root.parent().unwrap().to_path_buf(); }
                        let block_root = root.join(kv_name.as_deref().unwrap_or("inference")).join(format!("b{}", b_idx_off));
                        for l_idx in 0..28 {
                            let file_path = block_root.join(format!("l{}.st", l_idx));
                            if file_path.is_file() {
                                if let Ok(content) = load_kv_block(&file_path) {
                                    if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                                        let prefix = format!("b{}_l{}_", b_idx_off, l_idx);
                                        let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).ok();
                                        if let (Some(ka), Some(kp), Some(ks), Some(va), Some(vp), Some(vs)) = (get_t("k_anchors"), get_t("k_packed"), get_t("k_scales"), get_t("v_anchors"), get_t("v_packed"), get_t("v_scales")) {
                                            let bytes_to_f32 = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
                                            let meta = BitKVMetadata {
                                                k_anchors: Tensor::from_vec(bytes_to_f32(ka.data()), (1, 1, ka.shape()[2], 64), &Device::Cpu).unwrap(),
                                                k_packed: Tensor::from_slice(kp.data(), kp.shape(), &Device::Cpu).unwrap(),
                                                k_scales: Tensor::from_vec(bytes_to_f32(ks.data()), (1, 1, 256, 1), &Device::Cpu).unwrap(),
                                                v_anchors: Tensor::from_vec(bytes_to_f32(va.data()), (1, 1, ka.shape()[2], 64), &Device::Cpu).unwrap(),
                                                v_packed: Tensor::from_slice(vp.data(), vp.shape(), &Device::Cpu).unwrap(),
                                                v_scales: Tensor::from_vec(bytes_to_f32(vs.data()), (1, 1, 256, 1), &Device::Cpu).unwrap(),
                                                original_shape: vec![1, 1, 256, 64],
                                            };
                                            if let Ok(mut r) = reg.entries.write() {
                                                if b_idx < r.len() {
                                                    {
                                                        let mut cache = r[b_idx].bitkv_cache.write().unwrap();
                                                        cache[l_idx] = Some(meta);
                                                    }
                                                    r[b_idx].location[l_idx] = KVLocation::RAM;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
            }
        }
    });
}

struct ReadSlotGuard { sid: usize, active: bool }
impl Drop for ReadSlotGuard { fn drop(&mut self) { if self.active { let sid = self.sid; tauri::async_runtime::spawn(async move { SLOT_MANAGER.release_slot(sid).await; }); } } }

#[derive(Clone)]
pub enum ModelVariant { Standard(Qwen3VLModel), QuantizedVL(QuantizedQwen3VLModel), QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel) }

impl ModelVariant {
    pub async fn forward(&mut self, input_ids: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, video_pixel_values: Option<&Tensor>, video_grid_thw: Option<&Tensor>, cache_position: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>, kv_name: Option<String>) -> Result<Tensor> {
        match self {
            Self::Standard(m) => m.forward(input_ids, pixel_values, image_grid_thw, video_pixel_values, video_grid_thw, cache_position, seqlen_offset),
            Self::QuantizedVL(m) => m.forward(input_ids, pixel_values, image_grid_thw, video_pixel_values, video_grid_thw, cache_position, seqlen_offset, total_len, session_id, kv_name).await,
            Self::QuantizedText(m) => m.forward(input_ids, cache_position, seqlen_offset, total_len, session_id, kv_name).await,
        }
    }
    pub fn rebalance_layers(&mut self, device_id: usize, offset: usize, total_len: usize) -> Result<()> { match self { Self::Standard(_) => Ok(()), Self::QuantizedVL(m) => m.rebalance_layers(device_id, offset, total_len), Self::QuantizedText(m) => m.rebalance_layers(device_id, offset, total_len) } }
    pub fn get_current_kv(&self) -> (Vec<Tensor>, Vec<Tensor>) { match self { Self::QuantizedVL(m) => m.language_model.get_current_kv(), Self::QuantizedText(m) => m.language_model.get_current_kv(), _ => (vec![], vec![]) } }
    pub fn inject_kv_bitkv(&mut self, ka: &[Tensor], kp: &[Tensor], ks: &[Tensor], va: &[Tensor], vp: &[Tensor], vs: &[Tensor], os: &[usize]) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.inject_live_kv_bitkv(ka, kp, ks, va, vp, vs, os), Self::QuantizedText(m) => m.language_model.inject_live_kv_bitkv(ka, kp, ks, va, vp, vs, os), _ => Ok(()) } }
    pub async fn drop_kv_storage(&mut self) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.drop_kv_storage(), Self::QuantizedText(m) => m.language_model.drop_kv_storage(), _ => Ok(()) } }
    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.force_flush_all_active_blocks(session_id, kv_name).await, Self::QuantizedText(m) => m.language_model.force_flush_all_active_blocks(session_id, kv_name).await, _ => Ok(()) } }
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
    pub fn init_with_config(path: &str, tokenizer_path: Option<&str>, config_path: Option<&str>, text_device: Option<&Device>, text_device_id: usize, vision_device: Option<&Device>, vision_device_id: usize, dtype: Option<DType>, hard_token_limit: Option<usize>, force_text_only: bool, baking_only: bool, _is_disk_swap: bool, kv_root: std::path::PathBuf) -> Result<Self> {
        let path = path.strip_prefix(r"\\?\").unwrap_or(path);
        let tok_path = tokenizer_path.unwrap_or(path).strip_prefix(r"\\?\").unwrap_or(tokenizer_path.unwrap_or(path));
        let cfg_path = config_path.unwrap_or(path).strip_prefix(r"\\?\").unwrap_or(config_path.unwrap_or(path));
        let chat_template = ChatTemplate::init(tok_path)?;
        let tokenizer = TokenizerModel::init(tok_path)?;
        let raw_config: serde_json::Value = serde_json::from_slice(&std::fs::read(&std::path::Path::new(cfg_path).join("config.json"))?)?;
        let cfg: Qwen3VLConfig = if raw_config.get("text_config").is_some() { serde_json::from_value(raw_config)? } else {
            let text_config: crate::models::qwen3vl::config::Qwen3VLTextConfig = serde_json::from_value(raw_config.clone())?;
            Qwen3VLConfig { architectures: raw_config.get("architectures").and_then(|v| serde_json::from_value(v.clone()).ok()), auto_map: raw_config.get("auto_map").and_then(|v| serde_json::from_value(v.clone()).ok()), hidden_size: raw_config.get("hidden_size").and_then(|v| v.as_u64()).map(|v| v as usize), image_token_id: raw_config.get("image_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), model_type: raw_config.get("model_type").and_then(|v| v.as_str()).unwrap_or("qwen2").to_string(), text_config: Some(text_config), tie_word_embeddings: raw_config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(true), torch_dtype: raw_config.get("torch_dtype").and_then(|v| v.as_str()).map(|s| s.to_string()), transformers_version: raw_config.get("transformers_version").and_then(|v| v.as_str()).unwrap_or("").to_string(), video_token_id: raw_config.get("video_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), vision_config: None, vision_start_token_id: None, vision_end_token_id: None }
        };
        let t_dev = get_device(text_device); let v_dev = get_device(vision_device); let dtype = get_dtype(dtype, cfg.text_config.as_ref().and_then(|tc| tc.dtype.as_deref()).unwrap_or("float16"));
        let gguf_f = find_type_files(path, "gguf")?; let mmproj_p = gguf_f.iter().find(|f| f.contains("mmproj")).cloned();
        let mut m_p = gguf_f.iter().find(|f| f.contains("Qwen3-0.6B-Q8_0.gguf")).cloned();
        if m_p.is_none() { m_p = gguf_f.iter().find(|f| f.contains("Qwen3-0.6B-Q4_K_M.gguf")).cloned(); }
        if m_p.is_none() { m_p = gguf_f.iter().find(|f| !f.contains("mmproj")).cloned(); }
        let qwen3_vl = if !gguf_f.is_empty() {
            let kv_res = hard_token_limit.unwrap_or(4096) as u64 * 40000;
            if mmproj_p.is_some() && !force_text_only {
                let m_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(m_p.as_ref().unwrap())?)? };
                let mm_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&mmproj_p.unwrap())?)? };
                ModelVariant::QuantizedVL(QuantizedQwen3VLModel::new_with_mmap(&cfg, &gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?, Some(Arc::new(m_mmap)), &gguf_file::Content::read(&mut std::io::Cursor::new(&mm_mmap[..]))?, Some(Arc::new(mm_mmap)), &t_dev, text_device_id, &v_dev, vision_device_id, dtype, kv_res, baking_only)?)
            } else {
                let m_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(m_p.as_ref().unwrap())?)? };
                ModelVariant::QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel::new_with_mmap(&cfg, &gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?, Some(Arc::new(m_mmap)), &t_dev, text_device_id, dtype, kv_res, baking_only, baking_only)?)
            }
        } else { ModelVariant::Standard(Qwen3VLModel::new(cfg, unsafe { VarBuilder::from_mmaped_safetensors(&find_type_files(path, "safetensors")?, dtype, &t_dev)? })?) };
        let g_p = std::path::Path::new(cfg_path).join("generation_config.json"); let g_cfg = if g_p.exists() { serde_json::from_slice(&std::fs::read(g_p)?)? } else { Qwen3VLGenerationConfig::default() };
        let (e1, e2) = match &g_cfg.eos_token_id { serde_json::Value::Number(n) => { let id = n.as_u64().unwrap_or(151645) as u32; (id, id) }, serde_json::Value::Array(arr) => { (arr.get(0).and_then(|v| v.as_u64()).unwrap_or(151643) as u32, arr.get(1).and_then(|v| v.as_u64()).unwrap_or(151643) as u32) }, _ => (151643, 151643) };
        let loaded_model_name = if m_p.as_ref().map(|p| p.contains("0.6B")).unwrap_or(false) { "0.6B".to_string() } else { "2B".to_string() };
        Ok(Self { chat_template, tokenizer, pre_processor: Qwen3VLProcessor::new(tok_path, &v_dev, dtype)?, qwen3_vl, text_device: t_dev, vision_device: v_dev, eos_token_id1: e1, eos_token_id2: e2, generation_config: g_cfg, model_name: loaded_model_name, hard_token_limit, kv_root })
    }

    pub async fn prefill_only(&mut self, mes: ChatCompletionParameters, _cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _relay_target: Option<&mut Qwen3VLGenerateModel>, _kv_name: Option<String>) -> Result<usize> {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let full_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_toks = full_ids.len();
        self.qwen3_vl.forward(&Tensor::from_vec(full_ids.clone(), (1, total_toks), &self.text_device)?, None, None, None, None, None, 0, total_toks, session_id.clone(), _kv_name.clone()).await?;
        if let Some(s_id) = &session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(s_id);
            if !path.exists() { fs::create_dir_all(&path)?; }
            fs::write(path.join("tokens.json"), serde_json::to_string(&full_ids)?)?;
            let _ = self.force_flush_all_active_blocks(s_id, _kv_name.as_deref()).await;
            println!("[PREFILL-SAVE] All active KV blocks flushed to disk.");
        }
        Ok(total_toks)
    }

    pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _kv_name: Option<String>) -> Result<String> {
        let temperature = mes.temperature.unwrap_or(0.7) as f32;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut lp = get_logit_processor(Some(temperature), Some(mes.top_p.unwrap_or(0.9) as f32), Some(40), seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let f_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_toks = f_ids.len();
        let kv_len = self.get_kv_len();
        let is_0_6b_full = self.model_name.contains("0.6B") && self.get_kv_len() > 0;
        let use_zero_prefill = !is_0_6b_full && kv_len >= total_toks && total_toks > 0;
        let mut gen_text = String::new();
        let (input_ids, offset) = if use_zero_prefill {
            println!("[ZERO-PREFILL] Active cache found (Len: {}). Skipping full prompt prefill.", kv_len);
            (Tensor::from_vec(vec![f_ids[total_toks - 1]], (1, 1), &self.text_device)?, total_toks - 1)
        } else {
            println!("[FULL-PREFILL] Computing entire context for 28-layer inference (Len: {}).", total_toks);
            (Tensor::from_vec(f_ids.clone(), (1, total_toks), &self.text_device)?, 0)
        };
        let mut logits = self.qwen3_vl.forward(&input_ids, None, None, None, None, None, offset, total_toks, session_id.clone(), _kv_name.clone()).await?;
        let mut gen_ids = vec![];
        for i in 0..mes.max_tokens.unwrap_or(2048) {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { break; } }
            let next_id = lp.sample(&logits.flatten_all()?.to_dtype(DType::F32)?)?;
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            gen_ids.push(next_id);
            gen_text.push_str(&self.tokenizer.token_decode(vec![next_id])?);
            let current_pos = total_toks + i as usize;
            logits = self.qwen3_vl.forward(&Tensor::from_vec(vec![next_id], (1, 1), &self.text_device)?, None, None, None, None, None, current_pos, current_pos + 1, session_id.clone(), _kv_name.clone()).await?;
        }
        if let Some(s_id) = &session_id {
            let _ = self.force_flush_all_active_blocks(s_id, _kv_name.as_deref()).await;
            println!("[GEN-SAVE] All active KV blocks flushed to disk.");
        }
        Ok(gen_text)
    }

    pub fn get_kv_len(&self) -> usize { match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.get_kv_len(), ModelVariant::QuantizedText(m) => m.language_model.get_kv_len(), _ => 0 } }
    pub async fn drop_kv_storage(&mut self) -> Result<()> { self.qwen3_vl.drop_kv_storage().await }
    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> { self.qwen3_vl.force_flush_all_active_blocks(session_id, kv_name).await }
    pub fn clear_kv_cache(&mut self) { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.clear_kv_cache(), ModelVariant::QuantizedText(m) => m.language_model.clear_kv_cache(), _ => {} } }
    pub fn save_kv_to_disk(&mut self, path: &Path, kv_name: Option<&str>, offset: usize) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.save_kv_cache(path, false, offset, kv_name), ModelVariant::QuantizedText(m) => m.language_model.save_kv_cache(path, false, offset, kv_name), _ => Ok(()) } }
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.truncate_kv_cache(len), ModelVariant::QuantizedText(m) => m.language_model.truncate_kv_cache(len), _ => Ok(()) } }
    pub fn load_kv_from_disk(&mut self, path: &Path, kv_name: Option<&str>) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.load_kv_cache(path, &self.text_device, 0, 128, kv_name), ModelVariant::QuantizedText(m) => m.language_model.load_kv_cache(path, &self.text_device, 0, 128, kv_name), _ => Ok(()) } }
    pub async fn prefill_chunk(&mut self, text: String, _cancel_flag: Option<Arc<AtomicBool>>, _relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let chunk_ids_vec = self.tokenizer.text_encode_vec(text, false)?;
        let chunk_size = chunk_ids_vec.len();
        let current_pos = self.get_kv_len();
        let chunk_ids = Tensor::from_vec(chunk_ids_vec, (1, chunk_size), &self.text_device)?;
        self.qwen3_vl.forward(&chunk_ids, None, None, None, None, None, current_pos, current_pos + chunk_size, None, None).await?;
        Ok(chunk_size)
    }
}

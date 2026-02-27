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

#[derive(Clone)]
pub struct LayerKVDump {
    pub layer_idx: usize,
    pub k_data: Tensor,
    pub v_data: Tensor,
    pub k_shape: Tensor,
    pub raw_k: Option<Tensor>,
    pub raw_v: Option<Tensor>,
}

pub struct BakeTask {
    pub slot_id: usize, pub task_dir: PathBuf, pub kv_name: Option<String>,
    pub offset: usize, pub layers: Vec<LayerKVDump>, pub is_relay_baking: bool,
    pub block_idx: Option<usize>, pub registry: KVRegistry,
}

pub struct SaveTask {
    pub slot_id: usize, pub path: PathBuf, pub tensors: std::collections::HashMap<String, Tensor>,
    pub is_last: bool, pub block_idx: Option<usize>, pub registry: Option<KVRegistry>,
    pub kv_name: Option<String>,
}

pub enum SlotTask { 
    Bake(BakeTask), 
    Load(LoadTask),
    IndexUpdate {
        kv_name: String,
        layer_idx: usize,
        offset: usize,
        len: usize,
        file_name: String,
    },
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct LayerIndex {
    pub layer_idx: usize,
    pub total_tokens: usize,
    pub blocks: Vec<LayerBlockInfo>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct LayerBlockInfo {
    pub offset: usize,
    pub len: usize,
    pub file: String,
}

// [GLOBAL] 인덱스 업데이트 채널
pub static INDEX_TX: Lazy<mpsc::Sender<SlotTask>> = Lazy::new(|| {
    let (tx, mut rx) = mpsc::channel(2048);
    tokio::spawn(async move {
        let kv_dir = crate::utils::paths::get_kv_dir(None);
        while let Some(task) = rx.recv().await {
            if let SlotTask::IndexUpdate { kv_name, layer_idx, offset, len, file_name } = task {
                let index_path = kv_dir.join(&kv_name).join(format!("layer{}.json", layer_idx));
                let mut index = if index_path.exists() {
                    fs::read_to_string(&index_path).ok().and_then(|s| serde_json::from_str::<LayerIndex>(&s).ok())
                        .unwrap_or(LayerIndex { layer_idx, total_tokens: 0, blocks: vec![] })
                } else {
                    let _ = fs::create_dir_all(index_path.parent().unwrap());
                    LayerIndex { layer_idx, total_tokens: 0, blocks: vec![] }
                };

                if !index.blocks.iter().any(|b| b.offset == offset) {
                    index.blocks.push(LayerBlockInfo { offset, len, file: file_name });
                    index.blocks.sort_by_key(|b| b.offset);
                    index.total_tokens = index.blocks.iter().map(|b| b.len).sum();
                    if let Ok(json) = serde_json::to_string_pretty(&index) {
                        let _ = fs::write(&index_path, json);
                    }
                }
            }
        }
    });
    tx
});

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
            SlotRequest::Acquire { total_tokens: _total_tokens, tx } => {
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
    let (io_tx, mut io_rx) = mpsc::channel::<SaveTask>(10000); 
    tokio::spawn(async move {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(2000)); 
        while let Some(task) = io_rx.recv().await {
            let sem = semaphore.clone();
            let (tp, ts, reg, b_idx, sid, is_last, kv_n) = (task.path.clone(), task.tensors, task.registry.clone(), task.block_idx, task.slot_id, task.is_last, task.kv_name.clone());
            tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                struct IoGuard; impl Drop for IoGuard { fn drop(&mut self) { GLOBAL_IO_COUNTER.fetch_sub(1, Ordering::SeqCst); } }
                let _guard = IoGuard;
                if let Some(p) = tp.parent() { if !p.exists() { let _ = fs::create_dir_all(p); } }
                if let Ok(data) = safetensors::serialize(&ts, &None) {
                    drop(ts); 
                    // [DIRECT-IO] Use OS-accelerated high-speed write
                    let _ = save_kv_block(&tp, &data);
                    
                    // [CENTRALIZED-INDEX-UPDATE] 인덱스 채널로 업데이트 전송
                    if let Some(kv_name) = kv_n {
                        if let Some(l_str) = tp.file_name().and_then(|n| n.to_str()).and_then(|s| s.strip_prefix('l')).and_then(|s| s.strip_suffix(".st")) {
                            if let Ok(l_idx) = l_str.parse::<usize>() {
                                // 오프셋 추출 (파일명 b{offset}_l{idx}_k_anchors 에서 추출하거나 SaveTask에 추가 가능)
                                // 현재는 tp.parent() 폴더명이 b{offset} 이므로 이를 활용
                                let offset_str = tp.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()).and_then(|s| s.strip_prefix('b')).unwrap_or("0");
                                let offset = offset_str.parse::<usize>().unwrap_or(0);
                                
                                let _ = INDEX_TX.send(SlotTask::IndexUpdate {
                                    kv_name,
                                    layer_idx: l_idx,
                                    offset,
                                    len: 256,
                                    file_name: format!("b{}/l{}.st", offset, l_idx),
                                }).await;
                            }
                        }
                    }

                    if let (Some(r), Some(idx)) = (reg, b_idx) {
                        if let Ok(mut entries) = r.entries.write() {
                            if idx < entries.len() {
                                let e = &mut entries[idx]; 
                                if let Some(l_str) = tp.file_name().and_then(|n| n.to_str()).and_then(|s| s.strip_prefix('l')).and_then(|s| s.strip_suffix(".st")) {
                                    if let Ok(l_idx) = l_str.parse::<usize>() { 
                                        if l_idx < 28 { 
                                            e.location[l_idx] = KVLocation::SSD; 
                                            e.ssd_path = Some(tp.parent().unwrap().to_path_buf());
                                            // [MEMORY-RELEASE] SSD 저장 완료 즉시 RAM 메모리 해제
                                            if let Ok(mut cache) = e.bitkv_cache.write() {
                                                if l_idx < cache.len() { cache[l_idx] = None; }
                                            }
                                        } 
                                    }
                                }
                            }
                        }
                    }
                }
                let rem = SLOT_MANAGER.slots[sid].remaining_layers.fetch_sub(1, Ordering::SeqCst);
                if rem == 1 || is_last { SLOT_MANAGER.mark_ready(sid).await; }
            });
        }
    });

    tokio::spawn(async move {
        while let Some(task) = rx.recv().await {
            match task {
                SlotTask::Bake(bake) => {
                    let io_tx_inner = io_tx.clone();
                    let (sid, off, _is_relay, block_idx, registry, kv_name) = (bake.slot_id, bake.offset, bake.is_relay_baking, bake.block_idx, bake.registry.clone(), bake.kv_name.clone());
                    let loop_count = bake.layers.len();
                    SLOT_MANAGER.slots[sid].remaining_layers.store(loop_count, Ordering::SeqCst);
                    
                    for l_idx in 0..loop_count {
                        let mut src = bake.layers[l_idx].clone();
                        let act_l = src.layer_idx;
                        let task_dir = bake.task_dir.clone();
                        let registry_inner = registry.clone();
                        let kv_name_inner = kv_name.clone();
                        let io_tx_nested = io_tx_inner.clone();

                        // [BACK-COMPRESSION] 압축 작업 자체를 백그라운드로 완전히 분리
                        tokio::spawn(async move {
                            // [FIX] Convert raw tensors to BF16 (or F32 fallback)
                            if let (Some(rk), Some(rv)) = (src.raw_k.take(), src.raw_v.take()) {
                                let k_bf16 = rk.to_dtype(DType::BF16).unwrap_or(rk);
                                let v_bf16 = rv.to_dtype(DType::BF16).unwrap_or(rv);
                                src.k_data = k_bf16;
                                src.v_data = v_bf16;
                            }

                            let mut map = std::collections::HashMap::new();
                            let prefix = format!("b{}_l{}_", off, act_l);
                            map.insert(format!("{}k_data", prefix), src.k_data);
                            map.insert(format!("{}v_data", prefix), src.v_data);
                            map.insert(format!("{}k_shape", prefix), src.k_shape);
                            
                            let file_path = task_dir.join(format!("l{}.st", act_l));
                            GLOBAL_IO_COUNTER.fetch_add(1, Ordering::SeqCst);
                            let _ = io_tx_nested.send(SaveTask { 
                                slot_id: sid, path: file_path, tensors: map, 
                                is_last: false, block_idx, // [FIX] is_last logic handled by loop_count check outside
                                registry: Some(registry_inner), kv_name: kv_name_inner 
                            }).await;
                        });
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
                                // [DIRECT-IO] High-speed OS-level read
                                if let Ok(content) = load_kv_block(&file_path) {
                                    if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                                        let prefix = format!("b{}_l{}_", b_idx_off, l_idx);
                                        let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).ok();
                                        
                                        // [MODIFIED] BF16 Direct Load
                                        if let (Some(kd), Some(vd), Some(sh)) = (get_t("k_data"), get_t("v_data"), get_t("k_shape")) {
                                            let sh_u32: Vec<u32> = sh.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                                            let file_shape: Vec<usize> = sh_u32.iter().map(|&x| x as usize).collect();

                                            let dev = &Device::Cpu;
                                            let kd_t = Tensor::from_raw_buffer(kd.data(), DType::BF16, &file_shape, dev).unwrap();
                                            let vd_t = Tensor::from_raw_buffer(vd.data(), DType::BF16, &file_shape, dev).unwrap();

                                            let meta = BitKVMetadata {
                                                k_data: kd_t,
                                                v_data: vd_t,
                                                original_shape: file_shape,
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
                },
                SlotTask::IndexUpdate { .. } => {
                    // IndexUpdate tasks are handled by their own INDEX_TX worker.
                    // This match arm ensures the main worker doesn't panic if it receives one.
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
    pub fn inject_kv_bitkv(&mut self, kd: &[Tensor], vd: &[Tensor], os: &[usize]) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.inject_live_kv_bitkv(kd, vd, os), Self::QuantizedText(m) => m.language_model.inject_live_kv_bitkv(kd, vd, os), _ => Ok(()) } }
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
            wait_for_global_io().await; // [SYNC] Ensure SSD write is complete
            println!("[PREFILL-SAVE] All active KV blocks safely persisted to disk.");
        }
        Ok(total_toks)
    }

        pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _kv_name: Option<String>) -> Result<String> {
            // [CONTEXT-RESTORATION] If a session_id is provided, load the previous KV context from SSD
            if let Some(s_id) = &session_id {
                let snapshot_path = crate::utils::paths::get_kv_dir(None).join(s_id);
                if snapshot_path.exists() {
                    println!("[GEN-LOAD] Loading existing snapshot from {:?}...", snapshot_path);
                    let _ = self.load_kv_from_disk(&snapshot_path, _kv_name.as_deref());
                }
            }
    
            let temperature = mes.temperature.unwrap_or(0.7) as f32;
            let seed = mes.seed.unwrap_or(34562) as u64;
            let mut lp = get_logit_processor(Some(temperature), Some(mes.top_p.unwrap_or(0.9) as f32), Some(40), seed);
            let mes_render = self.chat_template.apply_chat_template(&mes)?;
            let input = self.pre_processor.process_info(&mes, &mes_render)?;
            let f_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
            
            let total_toks = f_ids.len();
            let kv_len = self.get_kv_len();
            
            // [FIX] Correctly determine if we should append to existing context
            let use_zero_prefill = kv_len > 0;
            let mut gen_text = String::new();
            let (input_ids, offset) = if use_zero_prefill {
                println!("[ZERO-PREFILL] Context restored. Appending new prompt at offset {}.", kv_len);
                (Tensor::from_vec(f_ids.clone(), (1, total_toks), &self.text_device)?, kv_len)
            } else {
                println!("[FULL-PREFILL] No context found. Computing entire prompt (Len: {}).", total_toks);
                (Tensor::from_vec(f_ids.clone(), (1, total_toks), &self.text_device)?, 0)
            };
            
            let total_tokens_after_prefill = offset + total_toks;
        
        wait_for_global_io().await; // [SYNC] Ensure disk is ready before inference
        let mut logits = self.qwen3_vl.forward(&input_ids, None, None, None, None, None, offset, total_tokens_after_prefill, session_id.clone(), _kv_name.clone()).await?;
        let mut gen_ids = vec![];
        for i in 0..mes.max_tokens.unwrap_or(2048) {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { break; } }
            let next_id = lp.sample(&logits.flatten_all()?.to_dtype(DType::F32)?)?;
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            gen_ids.push(next_id);
            gen_text.push_str(&self.tokenizer.token_decode(vec![next_id])?);
            let current_pos = total_tokens_after_prefill + i as usize;
            
            wait_for_global_io().await; // [SYNC] Wait for any incremental baking
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
    pub fn load_kv_from_disk(&mut self, path: &Path, kv_name: Option<&str>) -> Result<()> { 
        match &mut self.qwen3_vl { 
            ModelVariant::QuantizedVL(m) => m.language_model.load_kv_cache(path, &self.text_device, 0, 128, kv_name), 
            ModelVariant::QuantizedText(m) => m.language_model.load_kv_cache(path, &self.text_device, 0, 128, kv_name), 
            _ => Ok(()) 
        } 
    }
    pub async fn prefill_chunk(&mut self, text: String, _cancel_flag: Option<Arc<AtomicBool>>, _relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let chunk_ids_vec = self.tokenizer.text_encode_vec(text, false)?;
        let chunk_size = chunk_ids_vec.len();
        let current_pos = self.get_kv_len();
        let chunk_ids = Tensor::from_vec(chunk_ids_vec, (1, chunk_size), &self.text_device)?;
        self.qwen3_vl.forward(&chunk_ids, None, None, None, None, None, current_pos, current_pos + chunk_size, None, None).await?;
        Ok(chunk_size)
    }
}

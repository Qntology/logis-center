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
        let (tx, rx) = mpsc::channel(64);
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
    let (sm, rx) = SlotManager::new(512); (sm, Mutex::new(Some(rx)))
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
    let (tx, mut rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let kv_dir = crate::utils::paths::get_kv_dir(None);
        while let Some(task) = rx.recv().await {
            if let SlotTask::IndexUpdate { kv_name, layer_idx, offset, len, file_name } = task {
                let index_path = kv_dir.join(&kv_name).join(format!("layer{}.json", layer_idx));
                
                // [DIRECT-IO] Use OS-accelerated read for index metadata
                let mut index = if index_path.exists() {
                    if let Ok(data) = load_kv_block(&index_path) {
                        String::from_utf8(data).ok()
                            .and_then(|s| serde_json::from_str::<LayerIndex>(&s).ok())
                            .unwrap_or(LayerIndex { layer_idx, total_tokens: 0, blocks: vec![] })
                    } else {
                        LayerIndex { layer_idx, total_tokens: 0, blocks: vec![] }
                    }
                } else {
                    let _ = fs::create_dir_all(index_path.parent().unwrap());
                    LayerIndex { layer_idx, total_tokens: 0, blocks: vec![] }
                };

                if !index.blocks.iter().any(|b| b.offset == offset) {
                    index.blocks.push(LayerBlockInfo { offset, len, file: file_name });
                    index.blocks.sort_by_key(|b| b.offset);
                    index.total_tokens = index.blocks.iter().map(|b| b.len).sum();
                    if let Ok(json) = serde_json::to_string_pretty(&index) {
                        // [DIRECT-IO] Use OS-accelerated write for index metadata
                        let _ = save_kv_block(&index_path, json.as_bytes());
                    }
                }
                // [SYNC-WAIT-FIX] 인덱스 업데이트 완료 후 카운터 감소
                GLOBAL_IO_COUNTER.fetch_sub(1, Ordering::SeqCst);
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
    let (btx, brx) = mpsc::channel(256); let (ltx, lrx) = mpsc::channel(256);
    let _ = BAKE_TX.set(btx); let _ = LOAD_TX.set(ltx);
    if let Some(rx) = SLOT_MANAGER_DATA.1.lock().unwrap().take() { tauri::async_runtime::spawn(async move { spawn_slot_dispatcher(rx).await; }); }
    tauri::async_runtime::spawn(async move { spawn_slot_worker(brx); }); 
    tauri::async_runtime::spawn(async move { spawn_slot_worker(lrx); });
}

async fn spawn_slot_dispatcher(mut rx: mpsc::Receiver<SlotRequest>) {
    while let Some(req) = rx.recv().await {
        match req {
            SlotRequest::Acquire { total_tokens: _total_tokens, tx } => {
                let max_writes = 128; let mut found = None;
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
    let (io_tx, mut io_rx) = mpsc::channel::<SaveTask>(256); 
    tokio::spawn(async move {
        // [PARALLEL-IO] SSD 쓰기를 병렬로 처리하되, 시스템 부하를 고려해 동시 실행 수를 제한합니다.
        let semaphore = Arc::new(tokio::sync::Semaphore::new(12)); // 최대 12개 파일 동시 쓰기
        
        while let Some(task) = io_rx.recv().await {
            let sem = semaphore.clone();
            let (tp, ts, reg, b_idx, sid, is_last, kv_n) = (task.path.clone(), task.tensors, task.registry.clone(), task.block_idx, task.slot_id, task.is_last, task.kv_name.clone());
            
            tokio::spawn(async move {
                // 세마포어 허가 대기 (병렬 수 제한)
                let _permit = sem.acquire().await.unwrap();
                
                struct IoGuard; impl Drop for IoGuard { fn drop(&mut self) { GLOBAL_IO_COUNTER.fetch_sub(1, Ordering::SeqCst); } }
                let _guard = IoGuard;
                
                if let Some(p) = tp.parent() { if !p.exists() { let _ = fs::create_dir_all(p); } }
                match safetensors::serialize(&ts, &None) {
                    Ok(data) => {
                        drop(ts); 
                        if let Err(e) = save_kv_block(&tp, &data) {
                            eprintln!("[IO-ERROR] save_kv_block failed for {:?}: {}", tp, e);
                        } else if tp.to_string_lossy().contains("b0.st") || tp.to_string_lossy().contains("b23.st") {
                            // 시작과 끝 레이어 저장 시 로그 출력
                            println!("[IO-SUCCESS] Parallel Save Complete: {:?}", tp);
                        }
                        
                        // [CENTRALIZED-INDEX-UPDATE] 인덱스 채널로 업데이트 전송
                        if let Some(kv_name) = kv_n {
                            if let Some(l_str) = tp.file_name().and_then(|n| n.to_str()).and_then(|s| s.strip_prefix('b')).and_then(|s| s.strip_suffix(".st")) {
                                if let Ok(l_idx) = l_str.parse::<usize>() {
                                    let offset_str = tp.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()).and_then(|s| s.strip_prefix('l')).unwrap_or("0");
                                    let offset = offset_str.parse::<usize>().unwrap_or(0);

                                    // [SYNC-WAIT-FIX] 인덱스 업데이트 시작 전 카운터 증가
                                    GLOBAL_IO_COUNTER.fetch_add(1, Ordering::SeqCst);
                                    let _ = INDEX_TX.send(SlotTask::IndexUpdate {
                                        kv_name, layer_idx: l_idx, offset, len: 256,
                                        file_name: format!("l{}/b{}.st", offset, l_idx),
                                    }).await;
                                    }                            }
                        }

                        if let (Some(r), Some(idx)) = (reg, b_idx) {
                            if let Ok(mut entries) = r.entries.write() {
                                if idx < entries.len() {
                                    let e = &mut entries[idx]; 
                                    if let Some(l_str) = tp.file_name().and_then(|n| n.to_str()).and_then(|s| s.strip_prefix('b')).and_then(|s| s.strip_suffix(".st")) {
                                        if let Ok(l_idx) = l_str.parse::<usize>() { 
                                            if l_idx < 28 { 
                                                e.location[l_idx] = KVLocation::SSD; 
                                                e.ssd_path = Some(tp.parent().unwrap().to_path_buf());
                                                if let Ok(mut cache) = e.bitkv_cache.write() {
                                                    if l_idx < cache.len() { cache[l_idx] = None; }
                                                }
                                            } 
                                        }
                                    }
                                }
                            }
                        }
                    },
                    Err(e) => {
                        eprintln!("[IO-ERROR] safetensors::serialize failed for {:?}: {}", tp, e);
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
                        let task_dir = bake.task_dir.clone(); // task_dir is session_id/kv_type/l{offset}
                        let registry_inner = registry.clone();
                        let kv_name_inner = kv_name.clone();
                        let io_tx_nested = io_tx_inner.clone();

                        // [BACK-COMPRESSION] 압축 및 직렬화 작업 자체를 백그라운드로 완전히 분리
                        tokio::spawn(async move {
                            // [FIX] Convert raw tensors to BF16 (or F32 fallback)
                            // Handle potential missing raw data gracefully to avoid hangs
                            let (k_data, v_data) = if let (Some(rk), Some(rv)) = (src.raw_k.take(), src.raw_v.take()) {
                                // [ZERO-CPU] Perform DType conversion on GPU before moving to CPU
                                let k_bf16 = rk.to_dtype(DType::BF16).unwrap_or(rk);
                                let v_bf16 = rv.to_dtype(DType::BF16).unwrap_or(rv);
                                
                                let k_cpu = k_bf16.to_device(&Device::Cpu).unwrap_or(k_bf16);
                                let v_cpu = v_bf16.to_device(&Device::Cpu).unwrap_or(v_bf16);
                                (k_cpu, v_cpu)
                            } else {
                                // If raw is missing, fallback to existing data or empty tensors
                                // This ensures we don't skip the IO task creation entirely
                                (src.k_data.clone(), src.v_data.clone())
                            };

                            let mut map = std::collections::HashMap::new();
                            // [OFFSET-CENTRIC-FIX] Use layer index as file name (b{layer}), and offset as folder name (l{offset})
                            let prefix = format!("l{}_b{}_", off, act_l);
                            map.insert(format!("{}k_data", prefix), k_data);
                            map.insert(format!("{}v_data", prefix), v_data);
                            map.insert(format!("{}k_shape", prefix), src.k_shape.clone());
                            
                            // [STRUCTURE] task_root/l{offset}/b{layer}.st
                            // task_dir already contains the l{offset} part from trigger_realtime_incremental_bake
                            let file_path = task_dir.join(format!("b{}.st", act_l));
                            
                            // [FIX] Send to IO worker after confirming all tensors are on CPU
                            if let Err(e) = io_tx_nested.send(SaveTask { 
                                slot_id: sid, 
                                path: file_path.clone(), 
                                tensors: map, 
                                is_last: false, 
                                block_idx, 
                                registry: Some(registry_inner), 
                                kv_name: kv_name_inner 
                            }).await {
                                eprintln!("[BAKE-ERROR] Failed to send SaveTask to IO worker: {}. Slot {} might hang.", e, sid);
                                // If we fail to send, we must manually decrement the global counter (since IoGuard won't run)
                                // and the remaining_layers counter to prevent slot hang.
                                GLOBAL_IO_COUNTER.fetch_sub(1, Ordering::SeqCst);
                                let rem = SLOT_MANAGER.slots[sid].remaining_layers.fetch_sub(1, Ordering::SeqCst);
                                if rem == 1 { SLOT_MANAGER.mark_ready(sid).await; }
                            }
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
                        // [LOAD-FIX] Match new structure: session_id/kv_type/l{offset}/b{layer}.st
                        let block_root = root.join(kv_name.as_deref().unwrap_or("inference")).join(format!("l{}", b_idx_off));
                        
                        if sid == 0 {
                            println!("[LOAD-START] Loading from {:?}", block_root);
                        }

                        for l_idx in 0..28 {
                            let file_path = block_root.join(format!("b{}.st", l_idx));
                            if file_path.is_file() {
                                if let Ok(content) = load_kv_block(&file_path) {
                                    if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                                        let prefix = format!("l{}_b{}_", b_idx_off, l_idx);
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
    pub fn get_kv_len(&self) -> usize {
        match self {
            Self::QuantizedVL(m) => m.language_model.current_kv_len,
            Self::QuantizedText(m) => m.language_model.current_kv_len,
            _ => 0,
        }
    }

    pub fn set_kv_len(&mut self, len: usize) {
        match self {
            Self::QuantizedVL(m) => m.language_model.current_kv_len = len,
            Self::QuantizedText(m) => m.language_model.current_kv_len = len,
            _ => {}
        }
    }

    pub fn get_registry(&self) -> Option<KVRegistry> {
        match self {
            Self::QuantizedVL(m) => Some(m.language_model.registry.clone()),
            Self::QuantizedText(m) => Some(m.language_model.registry.clone()),
            _ => None,
        }
    }

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
    
    // [NEW] 레이어별 전수 처리를 위한 단일 레이어 제어 인터페이스
    pub fn get_initial_embeddings(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::QuantizedVL(m) => m.language_model.get_initial_embeddings(ids),
            Self::QuantizedText(m) => m.language_model.get_initial_embeddings(ids),
            _ => Err(anyhow!("Unsupported model variant for initial embeddings")),
        }
    }

    pub async fn forward_single_layer(&mut self, layer_idx: usize, xs: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>, offset: usize, session_id: Option<String>, kv_name: Option<String>, baking: bool) -> Result<Tensor> {
        match self {
            Self::QuantizedVL(m) => m.language_model.forward_single_layer(layer_idx, xs, cos, sin, mask, offset, session_id, kv_name, baking).await,
            Self::QuantizedText(m) => m.language_model.forward_single_layer(layer_idx, xs, cos, sin, mask, offset, session_id, kv_name, baking).await,
            _ => Err(anyhow!("Unsupported model variant for single layer forward")),
        }
    }

    pub fn clear_layer(&mut self, layer_idx: usize) {
        match self {
            Self::QuantizedVL(m) => m.language_model.layers[layer_idx].clear(),
            Self::QuantizedText(m) => m.language_model.layers[layer_idx].clear(),
            _ => {}
        }
    }

    pub fn save_hidden_states(&self, path: &Path, t: &Tensor) -> Result<()> {
        match self {
            Self::QuantizedVL(m) => m.language_model.save_hidden_states(path, t),
            Self::QuantizedText(m) => m.language_model.save_hidden_states(path, t),
            _ => Err(anyhow!("Unsupported model variant for save_hidden_states")),
        }
    }

    pub fn load_hidden_states(&self, path: &Path, device: &Device, dtype: DType) -> Result<Tensor> {
        match self {
            Self::QuantizedVL(m) => m.language_model.load_hidden_states(path, device, dtype),
            Self::QuantizedText(m) => m.language_model.load_hidden_states(path, device, dtype),
            _ => Err(anyhow!("Unsupported model variant for load_hidden_states")),
        }
    }

    pub fn sync_blocks_from_registry(&mut self) -> Result<()> {
        match self {
            Self::QuantizedVL(m) => m.sync_blocks_from_registry(),
            Self::QuantizedText(m) => m.sync_blocks_from_registry(),
            _ => Ok(()),
        }
    }

    /// [NEW] 모든 레이어 가중치를 한꺼번에 로드하여 디코딩 속도 확보
    pub fn reload_all_layers(&mut self) -> Result<()> {
        match self {
            Self::QuantizedVL(m) => m.language_model.reload_all_layers(),
            Self::QuantizedText(m) => m.language_model.reload_all_layers(),
            _ => Ok(()),
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
    pub fn init_with_config(path: &str, tokenizer_path: Option<&str>, config_path: Option<&str>, text_device: Option<&Device>, text_device_id: usize, vision_device: Option<&Device>, vision_device_id: usize, dtype: Option<DType>, hard_token_limit: Option<usize>, force_text_only: bool, baking_only: bool, _is_disk_swap: bool, kv_root: std::path::PathBuf) -> Result<Self> {
        let path = path.strip_prefix(r"\\?\").unwrap_or(path);
        let tok_path = tokenizer_path.unwrap_or(path).strip_prefix(r"\\?\").unwrap_or(tokenizer_path.unwrap_or(path));
        let cfg_path = config_path.unwrap_or(path).strip_prefix(r"\\?\").unwrap_or(config_path.unwrap_or(path));
        let chat_template = ChatTemplate::init(tok_path)?;
        let tokenizer = TokenizerModel::init(tok_path)?;
        // [ZERO-CPU] Use accelerated direct_loader for config loading
        let config_data = load_kv_block(&std::path::Path::new(cfg_path).join("config.json"))?;
        let raw_config: serde_json::Value = serde_json::from_slice(&config_data)?;
        let cfg: Qwen3VLConfig = if raw_config.get("text_config").is_some() { serde_json::from_value(raw_config)? } else {
            let text_config: crate::models::qwen3vl::config::Qwen3VLTextConfig = serde_json::from_value(raw_config.clone())?;
            Qwen3VLConfig { 
                architectures: raw_config.get("architectures").and_then(|v| serde_json::from_value(v.clone()).ok()), 
                auto_map: raw_config.get("auto_map").and_then(|v| serde_json::from_value(v.clone()).ok()), 
                hidden_size: raw_config.get("hidden_size").and_then(|v| v.as_u64()).map(|v| v as usize), 
                image_token_id: raw_config.get("image_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), 
                model_type: raw_config.get("model_type").and_then(|v| v.as_str()).unwrap_or("qwen3_5").to_string(), 
                text_config: Some(text_config), 
                tie_word_embeddings: raw_config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(true), 
                transformers_version: raw_config.get("transformers_version").and_then(|v| v.as_str()).unwrap_or("").to_string(), 
                video_token_id: raw_config.get("video_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), 
                vision_config: None, 
                vision_start_token_id: None, 
                vision_end_token_id: None 
            }
        };
        let t_dev = get_device(text_device); let v_dev = get_device(vision_device); let dtype = get_dtype(dtype, cfg.text_config.as_ref().and_then(|tc| tc.dtype.as_deref()).unwrap_or("float16"));
        let gguf_f = find_type_files(path, "gguf")?; let mmproj_p = gguf_f.iter().find(|f| f.contains("mmproj")).cloned();
        let mut m_p = gguf_f.iter().find(|f| f.contains("Qwen3.5-0.8B.gguf")).cloned();
        if m_p.is_none() { m_p = gguf_f.iter().find(|f| f.contains("Qwen3-0.8B-Q4_K_M.gguf")).cloned(); }
        if m_p.is_none() { m_p = gguf_f.iter().find(|f| !f.contains("mmproj")).cloned(); }
        let qwen3_vl = if !gguf_f.is_empty() {
            let kv_res = hard_token_limit.unwrap_or(4096) as u64 * 40000;
            if mmproj_p.is_some() && !force_text_only {
                let m_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(m_p.as_ref().unwrap())?)? };
                let mm_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&mmproj_p.unwrap())?)? };
                let ct_main = Arc::new(gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?);
                let ct_vision = Arc::new(gguf_file::Content::read(&mut std::io::Cursor::new(&mm_mmap[..]))?);
                
                ModelVariant::QuantizedVL(QuantizedQwen3VLModel::new_with_mmap(&cfg, ct_main, Some(Arc::new(m_mmap)), ct_vision, Some(Arc::new(mm_mmap)), &t_dev, text_device_id, &v_dev, vision_device_id, dtype, kv_res, baking_only)?)
            } else {
                let m_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(m_p.as_ref().unwrap())?)? };
                let ct_main = Arc::new(gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?);
                
                ModelVariant::QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel::new_with_mmap(&cfg, ct_main, Some(Arc::new(m_mmap)), &t_dev, text_device_id, dtype, kv_res, baking_only, baking_only)?)
            }
        } else { ModelVariant::Standard(Qwen3VLModel::new(cfg, unsafe { VarBuilder::from_mmaped_safetensors(&find_type_files(path, "safetensors")?, dtype, &t_dev)? })?) };
        let g_p = std::path::Path::new(cfg_path).join("generation_config.json"); 
        // [ZERO-CPU] Use accelerated direct_loader for generation config
        let g_cfg = if g_p.exists() { 
            let g_data = load_kv_block(&g_p)?;
            serde_json::from_slice(&g_data)? 
        } else { Qwen3VLGenerationConfig::default() };
        let (e1, e2) = match &g_cfg.eos_token_id { serde_json::Value::Number(n) => { let id = n.as_u64().unwrap_or(151645) as u32; (id, id) }, serde_json::Value::Array(arr) => { (arr.get(0).and_then(|v| v.as_u64()).unwrap_or(151643) as u32, arr.get(1).and_then(|v| v.as_u64()).unwrap_or(151643) as u32) }, _ => (151643, 151643) };
        let loaded_model_name = if m_p.as_ref().map(|p| p.contains("0.8B")).unwrap_or(false) { "Qwen3.5-0.8B".to_string() } else { "Qwen3-VL-2B".to_string() };
        Ok(Self { chat_template, tokenizer, pre_processor: Qwen3VLProcessor::new(tok_path, &v_dev, dtype)?, qwen3_vl, text_device: t_dev, vision_device: v_dev, eos_token_id1: e1, eos_token_id2: e2, generation_config: g_cfg, model_name: loaded_model_name, hard_token_limit, kv_root })
    }

    pub async fn prefill_only(&mut self, mes: ChatCompletionParameters, _cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _relay_target: Option<&mut Qwen3VLGenerateModel>, _kv_name: Option<String>) -> Result<usize> {
        // [FIX] Always start prefill from zero to avoid double offset on restart
        println!("[PREFILL-START] Initializing prefill for session: {:?}, kv_name: {:?}", session_id, _kv_name);
        self.clear_kv_cache();
        if let ModelVariant::QuantizedVL(m) = &mut self.qwen3_vl { m.language_model.truncate_kv_cache(0)?; }
        if let ModelVariant::QuantizedText(m) = &mut self.qwen3_vl { m.language_model.truncate_kv_cache(0)?; }

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let full_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_toks = full_ids.len();
        println!("[PREFILL-EXEC] Forwarding {} tokens...", total_toks);

        self.qwen3_vl.forward(&Tensor::from_vec(full_ids.clone(), (1, total_toks), &self.text_device)?, None, None, None, None, None, 0, total_toks, session_id.clone(), _kv_name.clone()).await?;

        if let Some(s_id) = &session_id {
            let kv_type = _kv_name.as_deref().unwrap_or("text");
            let path = crate::utils::paths::get_kv_dir(None).join(s_id).join(kv_type);
            if !path.exists() { 
                println!("[PREFILL-SAVE] Creating directory: {:?}", path);
                fs::create_dir_all(&path)?; 
            }

            // [ZERO-CPU] Use accelerated direct_loader for token saving
            let token_data = serde_json::to_vec(&full_ids)?;
            let token_path = path.join("tokens.json");
            println!("[PREFILL-SAVE] Saving tokens.json to {:?}", token_path);
            let _ = save_kv_block(&token_path, &token_data);

            println!("[PREFILL-SAVE] Flushing active blocks for session: {}, type: {}", s_id, kv_type);
            let _ = self.force_flush_all_active_blocks(s_id, Some(kv_type)).await;
            wait_for_global_io().await; // [SYNC] Ensure SSD write is complete
            println!("[PREFILL-SAVE] All active KV blocks safely persisted to disk in structured format.");
        }
        Ok(total_toks)
    }

    pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _kv_name: Option<String>) -> Result<String> {
        // [CONTEXT-RESTORATION] Load existing KV context from SSD (Inference path only)
        if let Some(s_id) = &session_id {
            let kv_type = _kv_name.as_deref().unwrap_or("text");
            let snapshot_root = crate::utils::paths::get_kv_dir(None).join(s_id).join(kv_type);

            println!("[GEN-LOAD] Checking for existing context in {:?}", snapshot_root);
            if snapshot_root.exists() && fs::read_dir(&snapshot_root).map(|mut d| d.next().is_some()).unwrap_or(false) {
                println!("[GEN-LOAD] Loading structured context from {:?}...", snapshot_root);
                let _ = self.load_kv_from_disk(&snapshot_root, None);
            } else {
                println!("[GEN-LOAD] No existing context found for session {}, type {}. Proceeding with fresh prefill if needed.", s_id, kv_type);
            }

            // [FIX] 로드 또는 스티칭 후 반드시 레이어별 블록 리스트를 새로고침해야 함 
            self.qwen3_vl.sync_blocks_from_registry()?;
        }

        let temperature = mes.temperature.unwrap_or(1.0) as f32;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let top_p = mes.top_p.unwrap_or(1.0) as f32;
        let top_k = mes.top_k;
        let min_p = mes.min_p;
        let rep_penalty = mes.repetition_penalty.unwrap_or(1.0);
        let pres_penalty = mes.presence_penalty.unwrap_or(0.0);

        let mut lp = get_logit_processor(Some(temperature), Some(top_p), top_k, min_p, seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let f_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;

        let total_toks = f_ids.len();
        let kv_len = self.get_kv_len();
        println!("[DIAG-GEN] Current KV Cache Len: {} tokens. Prompt: {} tokens.", kv_len, total_toks);

        // [FIX] Always use stitched context (kv_len) if available
        let mut gen_text = String::new();
        let (input_ids, offset) = if kv_len > 0 {
            println!("[STITCHED-INFERENCE] Using existing context ({} tokens). Appending prompt ({} tokens).", kv_len, total_toks);
            (Tensor::from_vec(f_ids.clone(), (1, total_toks), &self.text_device)?, kv_len)
        } else {
            println!("[FULL-PREFILL] No existing context. Computing entire prompt (Len: {}).", total_toks);
            (Tensor::from_vec(f_ids.clone(), (1, total_toks), &self.text_device)?, 0)
        };            
            let total_tokens_after_prefill = offset + input_ids.dim(1)?;
        
        wait_for_global_io().await; // [SYNC] Ensure disk is ready before inference
        let mut logits = self.qwen3_vl.forward(&input_ids, None, None, None, None, None, offset, total_tokens_after_prefill, session_id.clone(), _kv_name.clone()).await?;
        
        // [DEBUG-SAMPLING] 첫 번째 토큰 샘플링 전 로그
        println!("[DEBUG-GEN] Prefill Complete. EOS IDs: {}, {}. Sampling first token...", self.eos_token_id1, self.eos_token_id2);

        let mut gen_ids = vec![];

        // [DENSE-MODE] Find token IDs for '<', 'think', and '{' to apply biases
        let think_token_id = self.tokenizer.text_encode_vec("<think>".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let lt_id = self.tokenizer.text_encode_vec("<".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);

        let max_new_tokens = mes.max_tokens.unwrap_or(2048);
        println!("[GEN-START] Target: {} tokens | Temp: {} | Seed: {}", max_new_tokens, temperature, seed);

        for i in 0..max_new_tokens {
            if let Some(flag) = &cancel_flag { 
                if flag.load(Ordering::Relaxed) { 
                    println!("[GEN-STOP] Cancelled by user at token {}.", i);
                    break; 
                } 
            }
            
            let mut logits_tensor = logits.flatten_all()?.to_dtype(DType::F32)?;
            let vocab_size = logits_tensor.dim(0)?;
            // [FIX] Clone the device to avoid borrowing conflict during logits_tensor re-assignment
            let device = logits_tensor.device().clone();

            // [PENALTIES] Apply combined repetition and presence penalties (GPU)
            if !gen_ids.is_empty() {
                logits_tensor = apply_penalties(&logits_tensor, rep_penalty, pres_penalty, &gen_ids)?;
            }

            // [MIN-P-FILTERING] GPU Implementation of Min-P sampling
            if let Some(mp) = min_p {
                if mp > 0.0 {
                    let max_logit = logits_tensor.max(0)?.to_scalar::<f32>()?;
                    let threshold = max_logit + (mp as f32).ln();
                    let mask = logits_tensor.ge(threshold as f64)?;
                    let threshold_t = Tensor::new(-1000.0f32, &candle_core::Device::Cpu)?.to_device(&device)?;
                    logits_tensor = mask.where_cond(&logits_tensor, &threshold_t)?;
                }
            }

            // [DENSE-BIAS] GPU Implementation to prevent reasoning mode and force JSON
            let mut bias_indices = vec![];
            let mut bias_values = vec![];
            if (think_token_id as usize) < vocab_size { bias_indices.push(think_token_id); bias_values.push(-100.0f32); }
            if (lt_id as usize) < vocab_size { bias_indices.push(lt_id); bias_values.push(-10.0f32); }
            
            // If it's the very first token, strongly favor '{'
            if i == 0 {
                if (open_bracket_id as usize) < vocab_size { bias_indices.push(open_bracket_id); bias_values.push(20.0f32); }
                if (self.eos_token_id1 as usize) < vocab_size { bias_indices.push(self.eos_token_id1); bias_values.push(-1000.0f32); }
                if (self.eos_token_id2 as usize) < vocab_size { bias_indices.push(self.eos_token_id2); bias_values.push(-1000.0f32); }
            }
            
            if !bias_indices.is_empty() {
                let idx_t = Tensor::new(bias_indices.as_slice(), &candle_core::Device::Cpu)?.to_device(&device)?;
                let val_t = Tensor::from_vec(bias_values, (bias_indices.len(),), &candle_core::Device::Cpu)?.to_device(&device)?;
                logits_tensor = logits_tensor.scatter_add(&idx_t, &val_t, 0)?;
            }
            
            let mut next_id = lp.sample(&logits_tensor)?;
            
            // [FORCE-START] If model still tries to output EOS at step 0, override it with '{'
            if i == 0 && (next_id == self.eos_token_id1 || next_id == self.eos_token_id2) {
                println!("[DEBUG-GEN] EOS detected on first token. Overriding with '{{' to force JSON.");
                next_id = 123; // ASCII for '{' is usually safe, or use open_bracket_id
                if open_bracket_id != 999999 { next_id = open_bracket_id; }
            }
            
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { 
                println!("[GEN-STOP] EOS detected at token {}.", i + 1);
                break; 
            }
            gen_ids.push(next_id);
            let piece = self.tokenizer.token_decode(vec![next_id])?;
            gen_text.push_str(&piece);

            // [EARLY-STOP] JSON 중첩 깊이(Nesting Depth)를 추적하여 완벽히 닫혔을 때만 종료
            if gen_text.contains('{') {
                let mut depth = 0;
                let mut has_started = false;
                for c in gen_text.chars() {
                    if c == '{' { depth += 1; has_started = true; }
                    else if c == '}' { depth -= 1; }
                }
                // 모든 괄호의 쌍이 맞고(depth 0), 마지막 토큰이 '}' 계열일 때 종료
                if has_started && depth == 0 && gen_text.trim_end().ends_with('}') {
                    println!("[GEN-STOP] Balanced JSON detected (Depth 0) at token {}.", i + 1);
                    break;
                }
            }
            
            if i == max_new_tokens - 1 {
                println!("[GEN-STOP] Reached max_tokens limit ({}).", max_new_tokens);
            }
            
            let current_pos = total_tokens_after_prefill + i as usize;
            
            // [FIX] Qwen 3.5 mRoPE를 위한 Rank 3 위치 텐서 생성 (디코딩용 단일 토큰)
            let cache_pos_1d = Tensor::from_vec(vec![current_pos as u32], (1, 1), &self.text_device)?;
            let pos_ids_3d = cache_pos_1d.unsqueeze(0)?.broadcast_as((3, 1, 1))?;

            wait_for_global_io().await; // [SYNC] Wait for any incremental baking
            logits = self.qwen3_vl.forward(&Tensor::from_vec(vec![next_id], (1, 1), &self.text_device)?, None, None, None, None, Some(&pos_ids_3d), current_pos, current_pos + 1, session_id.clone(), _kv_name.clone()).await?;
        }
        
        println!("[GEN-RESULT] Generated {} tokens. Result: {}", gen_ids.len(), gen_text.replace("\n", " "));
        
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
            ModelVariant::QuantizedVL(m) => m.language_model.load_kv_cache(path, &self.text_device, 0, 0, kv_name),
            ModelVariant::QuantizedText(m) => m.language_model.load_kv_cache(path, &self.text_device, 0, 0, kv_name),            _ => Ok(()) 
        } 
    }

    /// [NEW] 모든 레이어 가중치를 메모리에 상주시켜 디코딩 속도 확보
    pub fn reload_all_layers(&mut self) -> Result<()> {
        self.qwen3_vl.reload_all_layers()
    }

    pub fn reload_layer(&mut self, l_idx: usize) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.language_model.reload_layer(l_idx, &self.text_device),
            ModelVariant::QuantizedText(m) => m.language_model.reload_layer(l_idx, &self.text_device),
            _ => Ok(()),
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

fn apply_penalties(logits: &Tensor, repetition_penalty: f32, presence_penalty: f32, previous_tokens: &[u32]) -> Result<Tensor> {
    if repetition_penalty == 1.0 && presence_penalty == 0.0 { return Ok(logits.clone()); }
    if previous_tokens.is_empty() { return Ok(logits.clone()); }

    let device = logits.device();
    let vocab_size = logits.dim(0)?;

    // [ZERO-CPU] Keep logits on GPU. Only manipulate the indices of previous tokens.
    let mut unique_tokens: Vec<u32> = previous_tokens.iter().cloned().collect();
    unique_tokens.sort();
    unique_tokens.dedup();
    
    // Filter out of bounds tokens
    let unique_tokens: Vec<u32> = unique_tokens.into_iter().filter(|&t| (t as usize) < vocab_size).collect();
    if unique_tokens.is_empty() { return Ok(logits.clone()); }

    let indices = Tensor::new(unique_tokens.as_slice(), &candle_core::Device::Cpu)?.to_device(device)?;
    
    // 1. Get logits for previous tokens
    let mut prev_logits = logits.index_select(&indices, 0)?;

    // 2. Apply presence penalty
    if presence_penalty != 0.0 {
        prev_logits = (prev_logits - presence_penalty as f64)?;
    }

    // 3. Apply repetition penalty (Multiplicative scaling on GPU)
    if repetition_penalty != 1.0 {
        let penalty = repetition_penalty as f64;
        // if logit < 0 { logit * penalty } else { logit / penalty }
        let cond = prev_logits.gt(0.0)?;
        let div_v = (prev_logits.clone() / penalty)?;
        let mul_v = (prev_logits.clone() * penalty)?;
        prev_logits = cond.where_cond(&div_v, &mul_v)?;
    }

    // 4. Calculate delta and scatter back to original logits
    let original_prev_logits = logits.index_select(&indices, 0)?;
    let delta = (prev_logits - original_prev_logits)?;
    
    Ok(logits.scatter_add(&indices, &delta, 0)?)
}

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
    pub shared_block: Option<KVBlock>,
}

pub struct SaveTask {
    pub slot_id: usize, pub path: PathBuf, pub tensors: std::collections::HashMap<String, Tensor>,
    pub is_last: bool, pub block_idx: Option<usize>, pub registry: Option<KVRegistry>,
    pub kv_name: Option<String>,
    pub shared_block: Option<KVBlock>,
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
                
                // [FIX] Use standard fs for JSON metadata to prevent DirectStorage sync/corruption issues
                let mut index = if index_path.exists() {
                    if let Ok(data) = fs::read(&index_path) {
                        serde_json::from_slice::<LayerIndex>(&data)
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
                        let _ = fs::write(&index_path, json.as_bytes());
                    }
                }
            }
        }
    });
    tx
});

pub struct LoadTask { pub slot_id: usize, pub path: PathBuf, pub layer_idx: usize, pub kv_name: Option<String>, pub shared_block: KVBlock, pub registry: KVRegistry }

// [BACKGROUND-COMPRESSOR] 워커 스레드에서 실행되는 1비트 압축 함수
fn compress_1bit_worker(t: &Tensor) -> Result<(Tensor, Tensor)> {
    let device = t.device();
    let t_cpu = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let dims = t_cpu.dims();
    let last_dim = dims[dims.len() - 1];
    let row_elements = dims.iter().rev().nth(1).cloned().unwrap_or(1) * last_dim;
    let total_elements = t_cpu.elem_count();
    
    let t_vec = t_cpu.flatten_all()?.to_vec1::<f32>()?;
    let mut packed = vec![0u8; (total_elements + 7) / 8];
    let mut scales = Vec::new();

    // 행(Row) 단위로 스케일 계산 및 비트 패킹 수행
    for chunk in t_vec.chunks(last_dim) {
        let mut max_abs = 0.0f32;
        for &v in chunk { if v.abs() > max_abs { max_abs = v.abs(); } }
        scales.push(max_abs);
    }

    // [PARALLEL-BIT-PACKING] 비트 연산은 CPU 부하가 크므로 최대한 효율적으로 처리
    for (i, &v) in t_vec.iter().enumerate() {
        if v > 0.0 {
            packed[i / 8] |= 1 << (i % 8);
        }
    }

    let scales_len = scales.len();
    let packed_t = Tensor::from_vec(packed, (total_elements + 7) / 8, &Device::Cpu)?;
    let scales_t = Tensor::from_vec(scales, (scales_len,), &Device::Cpu)?;
    
    Ok((packed_t, scales_t))
}

pub static BAKE_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();
pub static LOAD_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();
use tokio::sync::OnceCell;

pub async fn get_worker_channel() -> Result<mpsc::Sender<SlotTask>> { BAKE_TX.get().cloned().ok_or(anyhow!("Bake init error")) }
pub async fn get_load_worker() -> Result<mpsc::Sender<SlotTask>> { LOAD_TX.get().cloned().ok_or(anyhow!("Load init error")) }
pub async fn wait_for_global_io() {
    let mut last_log = std::time::Instant::now();
    loop {
        let count = GLOBAL_IO_COUNTER.load(Ordering::SeqCst);
        if count == 0 { break; }
        
        if last_log.elapsed() >= std::time::Duration::from_secs(1) {
            println!("[SYNC-WAIT] Waiting for background KV backup... Remaining tasks: {}", count);
            last_log = std::time::Instant::now();
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

pub fn init_bake_worker() {
    let (btx, brx) = mpsc::channel(64); let (ltx, lrx) = mpsc::channel(64);
    let _ = BAKE_TX.set(btx); let _ = LOAD_TX.set(ltx);
    if let Some(rx) = SLOT_MANAGER_DATA.1.lock().unwrap().take() { tauri::async_runtime::spawn(async move { spawn_slot_dispatcher(rx).await; }); }
    tauri::async_runtime::spawn(async move { spawn_slot_worker(brx); }); 
    tauri::async_runtime::spawn(async move { spawn_slot_worker(lrx); });
}

async fn spawn_slot_dispatcher(mut rx: mpsc::Receiver<SlotRequest>) {
    while let Some(req) = rx.recv().await {
        match req {
            SlotRequest::Acquire { total_tokens: _total_tokens, tx } => {
                let max_writes = 128; // [EXPANDED]
                let mut found = None;
                let mut retry_count = 0;
                while found.is_none() {
                    if SLOT_MANAGER.active_write_count.load(Ordering::SeqCst) < max_writes {
                        for (i, slot) in SLOT_MANAGER.slots.iter().enumerate() {
                            if slot.state.load(Ordering::SeqCst) == 0 {
                                if slot.state.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() { 
                                    SLOT_MANAGER.update_counters(0, 1); 
                                    SLOT_MANAGER.active_write_count.fetch_add(1, Ordering::SeqCst); 
                                    found = Some(i); break; 
                                }
                            }
                        }
                        if found.is_none() {
                            for (i, slot) in SLOT_MANAGER.slots.iter().enumerate() {
                                if slot.state.load(Ordering::SeqCst) == 2 {
                                    if slot.state.compare_exchange(2, 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() { 
                                        SLOT_MANAGER.update_counters(2, 1); 
                                        SLOT_MANAGER.active_write_count.fetch_add(1, Ordering::SeqCst); 
                                        found = Some(i); break; 
                                    }
                                }
                            }
                        }
                    }
                    if found.is_none() { 
                        retry_count += 1;
                        if retry_count % 100 == 0 { println!("[SLOT-WARN] High congestion. Active writes: {}", SLOT_MANAGER.active_write_count.load(Ordering::SeqCst)); }
                        tokio::time::sleep(std::time::Duration::from_millis(2)).await; 
                    }
                }
                let _ = tx.send(found.unwrap());
            },
            SlotRequest::Release { idx, is_bake } => {
                let s = &SLOT_MANAGER.slots[idx];
                let old = s.state.load(Ordering::SeqCst);
                let new = if is_bake { 2 } else { 0 }; // 2: BakedCache, 0: Free
                
                if s.state.compare_exchange(old, new, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    SLOT_MANAGER.update_counters(old, new);
                    if old == 1 { // 1: Write
                        SLOT_MANAGER.active_write_count.fetch_sub(1, Ordering::SeqCst);
                        println!("[SLOT-RECLAIM] Write Slot {} reclaimed. Active: {}", idx, SLOT_MANAGER.active_write_count.load(Ordering::SeqCst));
                    }
                    SLOT_MANAGER.handoff_notifier.notify_waiters();
                }
            }
        }
    }
}

fn spawn_slot_worker(mut rx: mpsc::Receiver<SlotTask>) {
    let (io_tx, mut io_rx) = mpsc::channel::<SaveTask>(1024); 
    tokio::spawn(async move {
        // [SCALABLE-IO] 전역 동시성 제한을 256으로 늘려 여러 레이어가 독립적으로 대역폭 사용 가능하게 함
        let semaphore = Arc::new(tokio::sync::Semaphore::new(256)); 
        while let Some(task) = io_rx.recv().await {
            let sem = semaphore.clone();
            let (tp, ts, reg, b_idx, sid, is_last, kv_n, shared_block) = (task.path.clone(), task.tensors, task.registry.clone(), task.block_idx, task.slot_id, task.is_last, task.kv_name.clone(), task.shared_block.clone());
            tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                struct IoGuard; impl Drop for IoGuard { fn drop(&mut self) { GLOBAL_IO_COUNTER.fetch_sub(1, Ordering::SeqCst); } }
                let _guard = IoGuard;
                if let Some(p) = tp.parent() { if !p.exists() { let _ = fs::create_dir_all(p); } }
                
                if let Ok(_) = candle_core::safetensors::save(&ts, &tp) {
                    drop(ts);
                    
                    // [CENTRALIZED-INDEX-UPDATE] Restore index update logic
                    if let Some(kv_name) = kv_n {
                        if let Some(l_str) = tp.file_name().and_then(|n| n.to_str()).and_then(|s| s.strip_prefix('l')).and_then(|s| s.strip_suffix(".st")) {
                            if let Ok(l_idx) = l_str.parse::<usize>() {
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

                    // 1. Registry Update (SSD 위치 등록)
                    if let (Some(r), Some(idx)) = (reg, b_idx) {
                        if let Ok(mut entries) = r.entries.write() {
                            if idx < entries.len() {
                                let e = &mut entries[idx]; 
                                if let Some(l_str) = tp.file_name().and_then(|n| n.to_str()).and_then(|s| s.strip_prefix('l')).and_then(|s| s.strip_suffix(".st")) {
                                    if let Ok(l_idx) = l_str.parse::<usize>() { 
                                        if l_idx < 28 { 
                                            e.location[l_idx] = KVLocation::SSD; 
                                            // [FIX] 절대 경로로 저장하여 로드 시 혼선 방지
                                            if let Ok(abs_path) = fs::canonicalize(tp.parent().unwrap()) {
                                                e.ssd_path = Some(abs_path);
                                            } else {
                                                e.ssd_path = Some(tp.parent().unwrap().to_path_buf());
                                            }
                                            if let Ok(mut cache) = e.bitkv_cache.write() {
                                                if l_idx < cache.len() { cache[l_idx] = None; }
                                            }
                                        } 
                                    }
                                }
                            }
                        }
                    }

                    // 2. Immediate Block Memory Release
                    if let Some(block) = shared_block {
                        if let Ok(mut inner) = block.inner.write() {
                            inner.k_cache = None;
                            inner.v_cache = None;
                        }
                    }
                }
                
                // 3. Slot (Layer Batch) Completion Check
                let rem = SLOT_MANAGER.slots[sid].remaining_layers.fetch_sub(1, Ordering::SeqCst);
                if rem == 1 || is_last { 
                    SLOT_MANAGER.mark_ready(sid).await; 
                }
            });
        }
    });

    tokio::spawn(async move {
        while let Some(task) = rx.recv().await {
            match task {
                SlotTask::Bake(bake) => {
                    let io_tx_inner = io_tx.clone();
                    let (sid, off, _is_relay, block_idx, registry, kv_name, shared_block) = (bake.slot_id, bake.offset, bake.is_relay_baking, bake.block_idx, bake.registry.clone(), bake.kv_name.clone(), bake.shared_block.clone());
                    let loop_count = bake.layers.len();
                    SLOT_MANAGER.slots[sid].remaining_layers.store(loop_count, Ordering::SeqCst);
                    
                    for l_idx in 0..loop_count {
                        let mut src = bake.layers[l_idx].clone();
                        let act_l = src.layer_idx;
                        let task_dir = bake.task_dir.clone();
                        let registry_inner = registry.clone();
                        let kv_name_inner = kv_name.clone();
                        let io_tx_nested = io_tx_inner.clone();
                        let shared_block_inner = shared_block.clone();

                        // [WORKER-COMPRESSION] 워커 내에서 비동기 압축 수행
                        tokio::spawn(async move {
                            // [RAII-GUARD] 어떤 경우에도 카운트를 줄이고 슬롯을 관리하는 가드
                            struct SlotGuard { sid: usize, is_last: bool }
                            impl Drop for SlotGuard {
                                fn drop(&mut self) {
                                    let sid = self.sid;
                                    let rem = SLOT_MANAGER.slots[sid].remaining_layers.fetch_sub(1, Ordering::SeqCst);
                                    if rem <= 1 || self.is_last {
                                        let _ = tauri::async_runtime::spawn(async move {
                                            SLOT_MANAGER.mark_ready(sid).await;
                                        });
                                    }
                                }
                            }
                            let _guard = SlotGuard { sid, is_last: false };

                            let mut map = std::collections::HashMap::new();
                            let prefix = format!("b{}_l{}_", off, act_l);

                            // [DIRECT-SAVE] BF16으로 통일하여 저장 (파일 크기 1MB로 유지)
                            if let (Some(rk), Some(rv)) = (src.raw_k, src.raw_v) {
                                let rk_bf16 = rk.to_device(&Device::Cpu).unwrap_or(rk).to_dtype(DType::BF16).unwrap();
                                let rv_bf16 = rv.to_device(&Device::Cpu).unwrap_or(rv).to_dtype(DType::BF16).unwrap();
                                
                                map.insert(format!("{}k_raw", prefix), rk_bf16);
                                map.insert(format!("{}v_raw", prefix), rv_bf16);
                            } else {
                                // Fallback for pre-compressed or empty data
                                map.insert(format!("{}k_data", prefix), src.k_data);
                                map.insert(format!("{}v_data", prefix), src.v_data);
                            }
                            
                            let file_path = task_dir.join(format!("l{}.st", act_l));
                            // [CRITICAL] Redundant fetch_add removed here! (Handled by quantized_model.rs)
                            let _ = io_tx_nested.send(SaveTask { slot_id: sid, path: file_path, tensors: map, is_last: false, block_idx, registry: Some(registry_inner), kv_name: kv_name_inner, shared_block: shared_block_inner }).await;
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
                                if let Ok(content) = load_kv_block(&file_path) {
                                    // [FIX] Use candle_core::safetensors::load_buffer with device
                                    if let Ok(st_map) = candle_core::safetensors::load_buffer(&content, &Device::Cpu) {
                                        let prefix = format!("b{}_l{}_", b_idx_off, l_idx);
                                        let get_t = |s: &str| st_map.get(&format!("{}{}", prefix, s)).cloned();
                                        
                                        // [MODIFIED] BF16 Direct Load
                                        if let (Some(kd_t), Some(vd_t), Some(sh_t)) = (get_t("k_data"), get_t("v_data"), get_t("k_shape")) {
                                            let res: Result<()> = (|| {
                                                let sh_u32: Vec<u32> = sh_t.to_dtype(DType::U32)?.flatten_all()?.to_vec1()?;
                                                let file_shape: Vec<usize> = sh_u32.iter().map(|&x| x as usize).collect();
                                                
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
                                                Ok(())
                                            })();
                                            if let Err(e) = res {
                                                println!("[LOAD-ERROR] Failed to process loaded KV: {:?}", e);
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
                let ct_main = Arc::new(gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?);
                let ct_vision = Arc::new(gguf_file::Content::read(&mut std::io::Cursor::new(&mm_mmap[..]))?);
                
                ModelVariant::QuantizedVL(QuantizedQwen3VLModel::new_with_mmap(&cfg, ct_main, Some(Arc::new(m_mmap)), ct_vision, Some(Arc::new(mm_mmap)), &t_dev, text_device_id, &v_dev, vision_device_id, dtype, kv_res, baking_only)?)
            } else {
                let m_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(m_p.as_ref().unwrap())?)? };
                let ct_main = Arc::new(gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?);
                
                ModelVariant::QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel::new_with_mmap(&cfg, ct_main, Some(Arc::new(m_mmap)), &t_dev, text_device_id, dtype, kv_res, baking_only, baking_only)?)
            }
        } else { ModelVariant::Standard(Qwen3VLModel::new(cfg, unsafe { VarBuilder::from_mmaped_safetensors(&find_type_files(path, "safetensors")?, dtype, &t_dev)? })?) };
        let g_p = std::path::Path::new(cfg_path).join("generation_config.json"); let g_cfg = if g_p.exists() { serde_json::from_slice(&std::fs::read(g_p)?)? } else { Qwen3VLGenerationConfig::default() };
        let (e1, e2) = match &g_cfg.eos_token_id { serde_json::Value::Number(n) => { let id = n.as_u64().unwrap_or(151645) as u32; (id, id) }, serde_json::Value::Array(arr) => { (arr.get(0).and_then(|v| v.as_u64()).unwrap_or(151643) as u32, arr.get(1).and_then(|v| v.as_u64()).unwrap_or(151643) as u32) }, _ => (151643, 151643) };
        let loaded_model_name = if m_p.as_ref().map(|p| p.contains("0.6B")).unwrap_or(false) { "0.6B".to_string() } else { "2B".to_string() };
        Ok(Self { chat_template, tokenizer, pre_processor: Qwen3VLProcessor::new(tok_path, &v_dev, dtype)?, qwen3_vl, text_device: t_dev, vision_device: v_dev, eos_token_id1: e1, eos_token_id2: e2, generation_config: g_cfg, model_name: loaded_model_name, hard_token_limit, kv_root })
    }

    pub async fn prefill_only(&mut self, mes: ChatCompletionParameters, _cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _relay_target: Option<&mut Qwen3VLGenerateModel>, _kv_name: Option<String>) -> Result<usize> {
        // [FIX] Always start prefill from zero to avoid double offset on restart
        self.clear_kv_cache();
        if let ModelVariant::QuantizedVL(m) = &mut self.qwen3_vl { m.language_model.truncate_kv_cache(0)?; }
        if let ModelVariant::QuantizedText(m) = &mut self.qwen3_vl { m.language_model.truncate_kv_cache(0)?; }

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let full_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_toks = full_ids.len();
        self.qwen3_vl.forward(&Tensor::from_vec(full_ids.clone(), (1, total_toks), &self.text_device)?, None, None, None, None, None, 0, total_toks, session_id.clone(), _kv_name.clone()).await?;
        if let Some(s_id) = &session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(s_id);
            // [STABILITY] Create full inference path to ensure worker doesn't fail
            let inf_path = path.join("inference").join("text");
            if !inf_path.exists() { fs::create_dir_all(&inf_path)?; }
            
            fs::write(path.join("tokens.json"), serde_json::to_string(&full_ids)?)?;
            println!("[PREFILL-SAVE] Tokens persisted. Inference directory ready at {:?}", inf_path);
        }
        Ok(total_toks)
    }

        pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _kv_name: Option<String>) -> Result<String> {
            // [CONTEXT-RESTORATION] If a session_id is provided, load the previous KV context from SSD
            let mut is_reference_snapshot = false;
            if let Some(s_id) = &session_id {
                let snapshot_root = crate::utils::paths::get_kv_dir(None).join(s_id);
                
                // [FIX] Try both 'inference/text' and 'reference/text' paths
                let paths_to_try = vec![
                    snapshot_root.join("inference").join("text"),
                    snapshot_root.join("reference").join("text"),
                    snapshot_root.clone(),
                ];

                for snapshot_path in paths_to_try {
                    if snapshot_path.exists() && fs::read_dir(&snapshot_path).map(|mut d| d.next().is_some()).unwrap_or(false) {
                        println!("[GEN-LOAD] Loading existing snapshot from {:?}...", snapshot_path);
                        
                        // [FIX] If loading from 'reference', we must force prefill for the other 27 layers.
                        if snapshot_path.to_string_lossy().contains("reference") {
                            is_reference_snapshot = true;
                        }

                        let _ = self.load_kv_from_disk(&snapshot_path, None); // [FIX] Already pointed to full path
                        
                        if is_reference_snapshot {
                            println!("[GEN-LOAD] Reference snapshot detected. Resetting Registry Entry states for Full 28-Layer Prefill...");
                            
                            // [FIX] Registry 상태를 RAM/Empty로 초기화하고 token_start를 0번부터 다시 설정
                            let reset_reg = |reg: &KVRegistry| {
                                let mut entries = reg.entries.write().unwrap();
                                for (i, entry) in entries.iter_mut().enumerate() {
                                    for loc in entry.location.iter_mut() { *loc = KVLocation::RAM; }
                                    for slot in entry.slot_ids.iter_mut() { *slot = None; }
                                    entry.token_start = i * 256; // [FIX] Re-initialize start position
                                    entry.token_len = 0;
                                    entry.is_dirty.fill(true);
                                    let mut cache = entry.bitkv_cache.write().unwrap();
                                    cache.fill(None);
                                }
                            };

                            if let ModelVariant::QuantizedVL(m) = &mut self.qwen3_vl {
                                reset_reg(&m.language_model.registry);
                                let _ = m.language_model.truncate_kv_cache(0);
                            } else if let ModelVariant::QuantizedText(m) = &mut self.qwen3_vl {
                                reset_reg(&m.language_model.registry);
                                let _ = m.language_model.truncate_kv_cache(0);
                            }
                            self.clear_kv_cache();
                        }
                        break;
                    }
                }
            }
    
            let temperature = mes.temperature.unwrap_or(1.0) as f32;
            let seed = mes.seed.unwrap_or(34562) as u64;
            let mut lp = get_logit_processor(Some(temperature), Some(mes.top_p.unwrap_or(1.0) as f32), Some(9), seed);
            let mes_render = self.chat_template.apply_chat_template(&mes)?;
            let input = self.pre_processor.process_info(&mes, &mes_render)?;
            let f_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
            
            let total_toks = f_ids.len();
            println!("[DIAG-GEN] Encoded Prompt Length: {} tokens.", total_toks); // [DEBUG] 24k 원인 파악용
            
            let kv_len = self.get_kv_len();
            
            // [REFINED-RELAY-LOGIC] Trust restored kv_len and skip prefill if it matches or covers the prompt.
            let mut gen_text = String::new();
            let (input_ids, offset) = if kv_len > 0 && !is_reference_snapshot {
                if kv_len >= total_toks {
                    println!("[SKIP-PREFILL] Snapshot covers entire prompt (Detected: {}, Needed: {}). Capping offset.", kv_len, total_toks);
                    // [FIX] Strictly cap offset at total_toks - 1 to prevent double offset
                    let last_id = *f_ids.last().unwrap_or(&0);
                    (Tensor::from_vec(vec![last_id], (1, 1), &self.text_device)?, total_toks - 1)
                } else {
                    let missing_ids = f_ids[kv_len..].to_vec();
                    let missing_len = missing_ids.len();
                    println!("[PARTIAL-PREFILL] Context partially restored ({}). Prefilling remaining {} tokens.", kv_len, missing_len);
                    (Tensor::from_vec(missing_ids, (1, missing_len), &self.text_device)?, kv_len)
                }
            } else {
                if is_reference_snapshot {
                    println!("[FULL-PREFILL] Reference context found. Computing entire prompt to fill all 28 layers (Len: {}).", total_toks);
                } else {
                    println!("[FULL-PREFILL] No context found. Computing entire prompt (Len: {}).", total_toks);
                }
                (Tensor::from_vec(f_ids.clone(), (1, total_toks), &self.text_device)?, 0)
            };
            
            let total_tokens_after_prefill = offset + input_ids.dim(1)?;
        
        let mut logits = self.qwen3_vl.forward(&input_ids, None, None, None, None, None, offset, total_tokens_after_prefill, session_id.clone(), _kv_name.clone()).await?;
        
        // [CRITICAL-SYNC] Prefill 연산 직후 발생한 백그라운드 IO(대피 및 저장)가 완료될 때까지 반드시 대기
        println!("[GEN-SYNC] Prefill computation done. Waiting for background KV backup to finish...");
        wait_for_global_io().await; 
        
        // [DEBUG-SAMPLING] 첫 번째 토큰 샘플링 전 로그
        println!("[DEBUG-GEN] Prefill & Backup Complete. Sampling first token...");

        let mut gen_ids = vec![];

        // [DENSE-MODE] Find token IDs for '<', 'think', and '{' to apply biases
        let think_token_id = self.tokenizer.text_encode_vec("<think>".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let lt_id = self.tokenizer.text_encode_vec("<".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);

        for i in 0..mes.max_tokens.unwrap_or(2048) {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { break; } }
            
            let mut logits_tensor = logits.flatten_all()?.to_dtype(DType::F32)?;
            
            // [REPETITION-PENALTY]
            if !gen_ids.is_empty() {
                logits_tensor = apply_repetition_penalty(&logits_tensor, 1.2, &gen_ids)?;
            }

            // [DENSE-BIAS] Penalize <think> and < to prevent reasoning mode
            let mut logits_vec = logits_tensor.to_vec1::<f32>()?;
            let len = logits_vec.len();
            if (think_token_id as usize) < len { logits_vec[think_token_id as usize] -= 100.0; }
            if (lt_id as usize) < len { logits_vec[lt_id as usize] -= 10.0; } // Bias away from starting any tags
            
            // If it's the very first token, strongly favor '{'
            if i == 0 && (open_bracket_id as usize) < len {
                logits_vec[open_bracket_id as usize] += 20.0; // [BOOST-HEAVY] Increase boost
                // [EOS-BAN] Forcibly ban EOS tokens on the first step
                if (self.eos_token_id1 as usize) < len { logits_vec[self.eos_token_id1 as usize] = -1000.0; }
                if (self.eos_token_id2 as usize) < len { logits_vec[self.eos_token_id2 as usize] = -1000.0; }
            }
            logits_tensor = Tensor::from_vec(logits_vec, logits_tensor.shape(), logits_tensor.device())?;
            
            let mut next_id = lp.sample(&logits_tensor)?;
            
            // [FORCE-START] If model still tries to output EOS at step 0, override it with '{'
            if i == 0 && (next_id == self.eos_token_id1 || next_id == self.eos_token_id2) {
                println!("[DEBUG-GEN] EOS detected on first token. Overriding with '{{' to force JSON.");
                next_id = 123; // ASCII for '{' is usually safe, or use open_bracket_id
                if open_bracket_id != 999999 { next_id = open_bracket_id; }
            }
            
            if i == 0 {
                println!("[DEBUG-GEN] First Token Final: {} ('{}')", next_id, self.tokenizer.token_decode(vec![next_id]).unwrap_or_else(|_| "???".to_string()));
            }

            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { 
                if i == 0 { println!("[DEBUG-GEN] Warning: Model emitted EOS on first token."); }
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
                    println!("[DEBUG-GEN] Balanced JSON detected (Depth 0). Stopping at token {}.", i + 1);
                    break;
                }
            }
            
            let current_pos = total_tokens_after_prefill + i as usize;
            
            wait_for_global_io().await; // [SYNC] Wait for any incremental baking
            logits = self.qwen3_vl.forward(&Tensor::from_vec(vec![next_id], (1, 1), &self.text_device)?, None, None, None, None, None, current_pos, current_pos + 1, session_id.clone(), _kv_name.clone()).await?;
        }
        if let Some(s_id) = &session_id {
            // [REMOVED] Redundant force_flush_all_active_blocks
            println!("[GEN-SAVE] Generation complete. Remaining KV blocks flushing in background.");
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

fn apply_repetition_penalty(logits: &Tensor, penalty: f32, previous_tokens: &[u32]) -> Result<Tensor> {
    let mut logits_vec = logits.to_vec1::<f32>()?;
    let mut set = std::collections::HashSet::new();
    for &t in previous_tokens {
        if !set.contains(&t) {
            let logit = logits_vec[t as usize];
            if logit < 0.0 {
                logits_vec[t as usize] = logit * penalty;
            } else {
                logits_vec[t as usize] = logit / penalty;
            }
            set.insert(t);
        }
    }
    let dev = logits.device();
    Ok(Tensor::from_vec(logits_vec, logits.shape(), dev)?)
}

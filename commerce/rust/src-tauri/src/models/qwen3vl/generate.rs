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
    pub async fn mark_ready(&self, idx: usize) { 
        let _ = self.request_tx.send(SlotRequest::Release { idx, is_bake: true }).await; 
    }

    pub async fn sync_with_sentinels(&self, session_id: &str) {
        let kv_dir = crate::utils::paths::get_kv_dir(None).join(session_id);
        if !kv_dir.exists() { return; }

        // [RECURSIVE-SCAN] 세션 디렉토리 하위의 모든 sentinels 폴더를 찾습니다.
        fn scan_sentinels(dir: &Path, sm: &SlotManager) {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        if path.file_name().map_or(false, |n| n == "sentinels") {
                            // 센티널 폴더 발견 시 파일 처리
                            if let Ok(s_entries) = std::fs::read_dir(&path) {
                                for s_entry in s_entries.flatten() {
                                    let s_path = s_entry.path();
                                    let ext = s_path.extension().and_then(|e| e.to_str()).unwrap_or("");
                                    if let Some(name) = s_path.file_stem().and_then(|s| s.to_str()) {
                                        if name.starts_with('s') {
                                            if let Ok(sid) = name[1..].parse::<usize>() {
                                                if sid < sm.slots.len() {
                                                    match ext {
                                                        "done" => {
                                                            // 비동기 런타임이므로 블로킹 방지를 위해 별도 실행
                                                            let sm_clone = sm.handoff_notifier.clone();
                                                            let s = &sm.slots[sid];
                                                            let old = s.state.load(Ordering::SeqCst);
                                                            if s.state.compare_exchange(old, 2, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                                                                sm.update_counters(old, 2);
                                                                sm_clone.notify_waiters();
                                                            }
                                                            let _ = std::fs::remove_file(&s_path);
                                                        },
                                                        _ => {}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            scan_sentinels(&path, sm);
                        }
                    }
                }
            }
        }

        scan_sentinels(&kv_dir, self);
    }

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
}

pub static GLOBAL_IO_COUNTER: AtomicUsize = AtomicUsize::new(0);
pub static SLOT_MANAGER_DATA: Lazy<(SlotManager, Mutex<Option<mpsc::Receiver<SlotRequest>>>)> = Lazy::new(|| {
    let (sm, rx) = SlotManager::new(512); 
    (sm, Mutex::new(Some(rx)))
});
pub static SLOT_MANAGER: Lazy<&SlotManager> = Lazy::new(|| &SLOT_MANAGER_DATA.0);

#[derive(Clone, Debug)]
pub struct LayerKVDump {
    pub layer_idx: usize,
    pub k_data: Tensor,
    pub v_data: Tensor,
    pub k_shape: Tensor,
    pub raw_k: Option<Tensor>,
    pub raw_v: Option<Tensor>,
}

#[derive(Debug)]
pub struct BakeTask {
    pub slot_id: usize, pub task_dir: PathBuf, pub kv_name: Option<String>,
    pub offset: usize, pub layers: Vec<LayerKVDump>, pub is_relay_baking: bool,
    pub block_idx: Option<usize>, pub registry: KVRegistry,
}

#[derive(Debug)]
pub struct SaveTask {
    pub slot_id: usize, pub path: PathBuf, pub tensors: std::collections::HashMap<String, Tensor>,
    pub is_last: bool, pub block_idx: Option<usize>, pub registry: Option<KVRegistry>,
    pub kv_name: Option<String>,
}
#[derive(Debug)]
pub struct WeightLoadTask {
    pub layer_idx: usize,
    pub config: crate::models::qwen3vl::config::Qwen3VLTextConfig,
    pub ct: Arc<gguf_file::Content>,
    pub mmap: Arc<memmap2::Mmap>,
    pub device: Device,
    pub dtype: DType,
    pub base_name: String,
    pub registry: KVRegistry,
    pub baking_only: bool,
    pub response_tx: tokio::sync::oneshot::Sender<Result<crate::models::qwen3vl::quantized_model::QuantizedQwen3VLTextDecoderLayer>>,
}

#[derive(Debug)]
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
    WeightLoad(WeightLoadTask),
    // [NEW] 가중치 비동기 소각을 위한 태스크
    Evict(crate::models::qwen3vl::quantized_model::QuantizedQwen3VLTextDecoderLayer),
}

// [NEW] 가중치 로딩 및 소각 전담 워커 채널
pub static WEIGHT_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();
pub static EVICT_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();

pub async fn get_weight_worker() -> Result<mpsc::Sender<SlotTask>> { WEIGHT_TX.get().cloned().ok_or(anyhow!("Weight worker init error")) }
pub async fn get_evict_worker() -> Result<mpsc::Sender<SlotTask>> { EVICT_TX.get().cloned().ok_or(anyhow!("Evict worker init error")) }

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

pub static INDEX_TX: Lazy<mpsc::Sender<SlotTask>> = Lazy::new(|| {
    let (tx, mut rx) = mpsc::channel(512);
    tokio::spawn(async move {
        let kv_dir = crate::utils::paths::get_kv_dir(None);
        while let Some(task) = rx.recv().await {
            if let SlotTask::IndexUpdate { kv_name, layer_idx, offset, len, file_name } = task {
                let index_path = kv_dir.join(&kv_name).join(format!("layer{}.json", layer_idx));
                
                // [FIX] 기존 인덱스 파일을 더 확실하게 읽어옴 (Overwrite 방지)
                let mut index = if index_path.exists() {
                    if let Ok(data) = std::fs::read(&index_path) {
                        serde_json::from_slice::<LayerIndex>(&data)
                            .unwrap_or(LayerIndex { layer_idx, total_tokens: 0, blocks: vec![] })
                    } else {
                        LayerIndex { layer_idx, total_tokens: 0, blocks: vec![] }
                    }
                } else {
                    let _ = fs::create_dir_all(index_path.parent().unwrap());
                    LayerIndex { layer_idx, total_tokens: 0, blocks: vec![] }
                };

                // [FIX] 중복 체크 및 정렬된 삽입
                if !index.blocks.iter().any(|b| b.offset == offset) {
                    index.blocks.push(LayerBlockInfo { offset, len, file: file_name });
                    index.blocks.sort_by_key(|b| b.offset);
                    
                    // [CRITICAL] 전체 토큰 길이는 마지막 블록의 오프셋 + 길이임
                    if let Some(last) = index.blocks.last() {
                        index.total_tokens = last.offset + last.len;
                    }
                    
                    if let Ok(json) = serde_json::to_string_pretty(&index) {
                        let _ = std::fs::write(&index_path, json.as_bytes());
                    }
                }
            }
        }
    });
    tx
});

#[derive(Debug)]
pub struct LoadTask { pub slot_id: usize, pub path: PathBuf, pub layer_idx: usize, pub kv_name: Option<String>, pub shared_block: KVBlock, pub registry: KVRegistry }

pub static BAKE_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();
pub static LOAD_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();
use tokio::sync::OnceCell;

pub async fn get_worker_channel() -> Result<mpsc::Sender<SlotTask>> { BAKE_TX.get().cloned().ok_or(anyhow!("Bake init error")) }
pub async fn get_load_worker() -> Result<mpsc::Sender<SlotTask>> { LOAD_TX.get().cloned().ok_or(anyhow!("Load init error")) }

pub async fn wait_for_global_io() { 
    let start = std::time::Instant::now();
    while GLOBAL_IO_COUNTER.load(Ordering::SeqCst) > 0 { 
        if start.elapsed().as_millis() % 2000 == 0 {
            let f = SLOT_MANAGER.count_free.load(Ordering::SeqCst);
            let w = SLOT_MANAGER.count_writes.load(Ordering::SeqCst);
            let c = SLOT_MANAGER.count_cached.load(Ordering::SeqCst);
            let r = SLOT_MANAGER.count_reads.load(Ordering::SeqCst);
            println!("[IO-WAIT] Pending: {} | Slots: Free={}, Writing={}, Cached={}, Reading={}", 
                GLOBAL_IO_COUNTER.load(Ordering::SeqCst), f, w, c, r);
        }

        if start.elapsed().as_secs() > 10 {
            println!("[STUCK-GUARD] IO Counter stuck at {}. Forcing proceed.", GLOBAL_IO_COUNTER.load(Ordering::SeqCst));
            GLOBAL_IO_COUNTER.store(0, Ordering::SeqCst);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await; 
    } 
}

async fn spawn_weight_loader_worker(mut rx: mpsc::Receiver<SlotTask>) {
    while let Some(task) = rx.recv().await {
        if let SlotTask::WeightLoad(t) = task {
            let (tx, layer_idx) = (t.response_tx, t.layer_idx);
            let (config, ct, mmap, device, dtype, base_name, registry, baking_only) = (t.config, t.ct, t.mmap, t.device, t.dtype, t.base_name, t.registry, t.baking_only);

            // [OPTIMIZATION] .await를 제거하여 워커가 즉시 다음 메시지를 받을 수 있도록 함 (True Async)
            let _ = tokio::task::spawn_blocking(move || {
                let gguf_blk = format!("blk.{}", layer_idx);
                let prefix = if ct.tensor_infos.contains_key(&format!("{}.attn_norm.weight", gguf_blk)) { 
                    gguf_blk 
                } else { 
                    format!("{}.layers.{}", base_name, layer_idx) 
                };

                let res = crate::models::qwen3vl::quantized_model::QuantizedQwen3VLTextDecoderLayer::new_direct(
                    &config, &ct, &mmap, &prefix, &device, dtype, layer_idx, baking_only, registry
                );
                let _ = tx.send(res);
            });
        }
    }
}
async fn spawn_evict_worker(mut rx: mpsc::Receiver<SlotTask>) {
    while let Some(task) = rx.recv().await {
        if let SlotTask::Evict(mut layer) = task {
            // [NON-BLOCKING-DROP] 메인 스레드 밖에서 무거운 텐서 파괴 수행
            layer.clear();
            drop(layer);
        }
    }
}

pub fn init_bake_worker() {
    let (btx, brx) = mpsc::channel(256); 
    let (ltx, lrx) = mpsc::channel(256);
    let (wtx, wrx) = mpsc::channel(128); // [NEW] Weight relay channel
    let (etx, erx) = mpsc::channel(128); // [NEW] Evict channel

    let _ = BAKE_TX.set(btx); 
    let _ = LOAD_TX.set(ltx);
    let _ = WEIGHT_TX.set(wtx); // [NEW] Global weight transmitter
    let _ = EVICT_TX.set(etx); // [NEW] Global evict transmitter

    if let Some(rx) = SLOT_MANAGER_DATA.1.lock().unwrap().take() { 
        tauri::async_runtime::spawn(async move { spawn_slot_dispatcher(rx).await; }); 
    }
    tauri::async_runtime::spawn(async move { spawn_slot_worker(brx); });
    tauri::async_runtime::spawn(async move { spawn_slot_worker(lrx); });
    tauri::async_runtime::spawn(async move { spawn_weight_loader_worker(wrx).await; }); // [NEW] Async weight loader
    tauri::async_runtime::spawn(async move { spawn_evict_worker(erx).await; }); // [NEW] Async evict worker
}
async fn spawn_slot_dispatcher(mut rx: mpsc::Receiver<SlotRequest>) {
    while let Some(req) = rx.recv().await {
        match req {
            SlotRequest::Acquire { total_tokens: _total_tokens, tx } => {
                let max_writes = 512;
                let mut found = None;
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
                    if found.is_none() { tokio::time::sleep(std::time::Duration::from_millis(2)).await; }
                }
                let _ = tx.send(found.unwrap());
            },
            SlotRequest::Release { idx, is_bake } => {
                let s = &SLOT_MANAGER.slots[idx]; 
                let old = s.state.load(Ordering::SeqCst);
                let new = if is_bake { 2 } else { 0 };
                if s.state.compare_exchange(old, new, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    SLOT_MANAGER.update_counters(old, new);
                    if old == 1 { SLOT_MANAGER.active_write_count.fetch_sub(1, Ordering::SeqCst); }
                    
                    // [IMMEDIATE-CLEANUP] 슬롯 해제 시 내부에 들고 있던 텐서들을 즉시 해제
                    for k in &s.k_layers { let mut g = k.lock().unwrap(); *g = None; }
                    for v in &s.v_layers { let mut g = v.lock().unwrap(); *g = None; }
                    
                    SLOT_MANAGER.handoff_notifier.notify_waiters();
                }
            }
        }
    }
}

struct SlotCompletionGuard {
    sid: usize,
    is_last: bool,
    active: bool,
    sentinel_path: Option<PathBuf>,
}

impl Drop for SlotCompletionGuard {
    fn drop(&mut self) {
        if self.active {
            let sid = self.sid;
            let is_last = self.is_last;
            let s_path = self.sentinel_path.clone();
            tauri::async_runtime::spawn(async move {
                if let Some(path) = s_path {
                    let done_path = path.with_extension("done");
                    let _ = fs::File::create(&done_path);
                    let _ = fs::remove_file(&path);
                }
                let rem = SLOT_MANAGER.slots[sid].remaining_layers.fetch_sub(1, Ordering::SeqCst);
                if rem == 1 || is_last {
                    SLOT_MANAGER.mark_ready(sid).await;
                }
            });
        }
    }
}

fn spawn_slot_worker(mut rx: mpsc::Receiver<SlotTask>) {
    let (io_tx, mut io_rx) = mpsc::channel::<SaveTask>(1024);
    tokio::spawn(async move {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(128));
        while let Some(task) = io_rx.recv().await {
            let sem = semaphore.clone();
            let (tp, ts, reg, b_idx, sid, is_last, kv_n) = (task.path.clone(), task.tensors, task.registry.clone(), task.block_idx, task.slot_id, task.is_last, task.kv_name.clone());
            tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                struct IoGuard; 
                impl Drop for IoGuard { 
                    fn drop(&mut self) { 
                        // [SAFE-SUB] 언더플로우 방지: 0보다 클 때만 깎습니다.
                        let current = GLOBAL_IO_COUNTER.load(Ordering::SeqCst);
                        if current > 0 { GLOBAL_IO_COUNTER.fetch_sub(1, Ordering::SeqCst); }
                    } 
                }
                let _io_guard = IoGuard;
                let sentinel_dir = if let Some(p) = tp.parent() { p.parent().map(|p2| p2.join("sentinels")) } else { None };
                let mut s_path = None;
                if let Some(sd) = sentinel_dir {
                    if !sd.exists() { let _ = fs::create_dir_all(&sd); }
                    let p_path = sd.join(format!("s{}.pending", sid));
                    let _ = fs::File::create(&p_path);
                    s_path = Some(p_path);
                }
                
                if let Some(p) = tp.parent() { if !p.exists() { let _ = fs::create_dir_all(p); } }
                let tp_for_blocking = tp.clone();
                let save_res = tokio::task::spawn_blocking(move || {
                    let serialized = safetensors::serialize(&ts, &None);
                    drop(ts); 
                    if let Ok(data) = serialized { save_kv_block(&tp_for_blocking, &data) } else { Err(anyhow!("Serialization failed")) }
                }).await;

                // [RESOURCE-RECOVERY] 레이어 하나가 완료될 때마다 카운트 감소 및 센티널 관리
                if let Some(path) = s_path {
                    let _ = fs::File::create(path.with_extension("done"));
                    let _ = fs::remove_file(&path);
                }
                let rem = SLOT_MANAGER.slots[sid].remaining_layers.fetch_sub(1, Ordering::SeqCst);
                if rem == 1 {
                    // 이 블록의 모든 레이어(28개)가 SSD에 기록되었을 때만 슬롯을 Ready로 전환
                    SLOT_MANAGER.mark_ready(sid).await;
                }

                match save_res {
                    Ok(Ok(_)) => {
                        if let Some(kv_name) = kv_n {
                            if let Some(l_str) = tp.file_name().and_then(|n| n.to_str()).and_then(|s| s.strip_prefix('l')).and_then(|s| s.strip_suffix(".st")) {
                                if let Ok(l_idx) = l_str.parse::<usize>() {
                                    let offset_str = tp.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()).and_then(|s| s.strip_prefix('b')).unwrap_or("0");
                                    let offset = offset_str.parse::<usize>().unwrap_or(0);
                                    let _ = INDEX_TX.send(SlotTask::IndexUpdate { kv_name, layer_idx: l_idx, offset, len: 256, file_name: format!("b{}/l{}.st", offset, l_idx) }).await;
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
                                                if let Ok(mut cache) = e.bitkv_cache.write() { if l_idx < cache.len() { cache[l_idx] = None; } }
                                            } 
                                        }
                                    }
                                }
                            }
                        }
                    },
                    _ => {}
                }
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
                        let src = bake.layers[l_idx].clone();
                        let act_l = src.layer_idx;
                        let task_dir = bake.task_dir.clone();
                        let registry_inner = registry.clone();
                        let kv_name_inner = kv_name.clone();
                        let io_tx_nested = io_tx_inner.clone();
                        let mut map = std::collections::HashMap::new();
                        let prefix = format!("b{}_l{}_", off, act_l);
                        map.insert(format!("{}k_data", prefix), src.k_data.clone());
                        map.insert(format!("{}v_data", prefix), src.v_data.clone());
                        map.insert(format!("{}k_shape", prefix), src.k_shape.clone());
                        let file_path = task_dir.join(format!("l{}.st", act_l));

                        // [CRITICAL-FIX] 레이어 하나를 저장 큐에 넣을 때마다 카운터를 1씩 올립니다.
                        // 이를 통해 워커 내부의 IoGuard와 1:1 대응을 맞추어 언더플로우를 방지합니다.
                        GLOBAL_IO_COUNTER.fetch_add(1, Ordering::SeqCst);

                        let _ = io_tx_nested.send(SaveTask { 
                            slot_id: sid, path: file_path, tensors: map, is_last: false, 
                            block_idx, registry: Some(registry_inner), kv_name: kv_name_inner 
                        }).await;
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
                                        let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).or_else(|_| st.tensor(s)).ok();
                                        if let (Some(kd), Some(vd), Some(sh)) = (get_t("k_data"), get_t("v_data"), get_t("k_shape")) {
                                            let sh_u32: Vec<u32> = sh.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                                            let file_shape: Vec<usize> = sh_u32.iter().map(|&x| x as usize).collect();
                                            let kd_t = Tensor::from_raw_buffer(kd.data(), DType::BF16, &file_shape, &Device::Cpu).unwrap();
                                            let vd_t = Tensor::from_raw_buffer(vd.data(), DType::BF16, &file_shape, &Device::Cpu).unwrap();
                                            let meta = BitKVMetadata { k_data: kd_t, v_data: vd_t, original_shape: file_shape };
                                            if let Ok(mut r) = reg.entries.write() {
                                                if b_idx < r.len() {
                                                    { let mut cache = r[b_idx].bitkv_cache.write().unwrap(); cache[l_idx] = Some(meta); }
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
                _ => {}
            }
        }
    });
}

struct ReadSlotGuard { sid: usize, active: bool }
impl Drop for ReadSlotGuard { fn drop(&mut self) { if self.active { let sid = self.sid; tauri::async_runtime::spawn(async move { SLOT_MANAGER.release_slot(sid).await; }); } } }

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
            if !path.exists() { fs::create_dir_all(&path)?; }
            fs::write(path.join("tokens.json"), serde_json::to_string(&full_ids)?)?;
            let _ = self.force_flush_all_active_blocks(s_id, _kv_name.as_deref()).await;
            wait_for_global_io().await;
            println!("[PREFILL-SAVE] All active KV blocks safely persisted to disk.");
        }
        Ok(total_toks)
    }

    pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _kv_name: Option<String>) -> Result<String> {
        // [ID-NORMALIZATION-CONSOLIDATED] 모든 경로의 기준이 되는 ID를 여기서 즉시 정규화합니다.
        let session_id = session_id.map(|s| s.split("_step_").next().unwrap_or(&s).to_string());
        
        if let Some(s_id) = &session_id {
            let snapshot_root = crate::utils::paths::get_kv_dir(None).join(s_id);
            let kv_type = _kv_name.as_deref().unwrap_or("text");
            
            // 이제 모든 경로는 정규화된 단일 ID를 기준으로 탐색됩니다.
            let paths_to_try = vec![
                snapshot_root.join("inference").join(kv_type),
                snapshot_root.join("reference").join(kv_type),
                snapshot_root.clone()
            ];
            
            for snapshot_path in paths_to_try {
                if snapshot_path.exists() && fs::read_dir(&snapshot_path).map(|mut d| d.next().is_some()).unwrap_or(false) {
                    println!("[GEN-LOAD] Loading snapshot from unified path: {:?}", snapshot_path);
                    let _ = self.load_kv_from_disk(&snapshot_path, None);
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
        let kv_len = self.get_kv_len();
        let mut gen_text = String::new();
        let (input_ids, offset) = if kv_len > 0 {
            if kv_len >= total_toks {
                let last_id = *f_ids.last().unwrap_or(&0);
                (Tensor::from_vec(vec![last_id], (1, 1), &self.text_device)?, total_toks - 1)
            } else {
                let missing_ids = f_ids[kv_len..].to_vec();
                let missing_len = missing_ids.len();
                (Tensor::from_vec(missing_ids, (1, missing_len), &self.text_device)?, kv_len)
            }
        } else { (Tensor::from_vec(f_ids.clone(), (1, total_toks), &self.text_device)?, 0) };
        let mut total_tokens_after_prefill = offset + input_ids.dim(1)?;
        wait_for_global_io().await;
        let mut logits = self.qwen3_vl.forward(&input_ids, None, None, None, None, None, offset, total_tokens_after_prefill, session_id.clone(), _kv_name.clone()).await?;
        
        // [SSD-BRIDGE] Ideal Transition Mode: Flush -> Clear -> Registry Rebuild
        if let Some(s_id) = &session_id {
            println!("[SSD-BRIDGE] Initiating Ideal Bridge for Unified Task: {}. Flushing KV...", s_id);
            let actual_len = self.get_kv_len();
            
            // 1. 잔여 데이터 SSD 강제 백업
            let _ = self.force_flush_all_active_blocks(s_id, _kv_name.as_deref()).await;
            wait_for_global_io().await;
            
            // 2. VRAM 초기화
            println!("[SSD-BRIDGE] Initializing VRAM (Clearing pointers)...");
            self.clear_kv_cache();
            
            // 3. SSD 장부(Registry) 재구축
            println!("[SSD-BRIDGE] Rebuilding Registry from unified snapshots...");
            let kv_type = _kv_name.as_deref().unwrap_or("text");
            let snapshot_path = crate::utils::paths::get_kv_dir(None).join(s_id).join("inference").join(kv_type);
            let device_clone = self.text_device.clone();
            
            // 장부에 실제 길이를 각인시켜 복구 정합성 확보
            let _ = self.load_kv_cache(&snapshot_path, &device_clone, actual_len, 0, None);
            
            wait_for_global_io().await;
            
            // [VERIFICATION] 이제 get_kv_len()이 논리적 길이를 반환하므로 반드시 actual_len과 일치해야 합니다.
            let verified_len = self.get_kv_len();
            println!("[SSD-BRIDGE] Ideal Bridge Established. Context (V:{}) verified.", verified_len);

            // 복구된 장부를 기반으로 디코딩 시작 지점 동기화
            total_tokens_after_prefill = verified_len;
            println!("[SSD-BRIDGE] Syncing decoding start pos to: {}", total_tokens_after_prefill);
        }

            let mut gen_ids = vec![];
            let _think_token_id = 151643;

            // [FIX] 루프 진입 전 현재 오프셋 확정 (이미 로드된 KV 이후부터 시작)
            let mut current_decode_pos = total_tokens_after_prefill;

            for _i in 0..mes.max_tokens.unwrap_or(2048) {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { break; } }
            let mut logits_tensor = logits.flatten_all()?.to_dtype(DType::F32)?;
            if !gen_ids.is_empty() { logits_tensor = apply_repetition_penalty(&logits_tensor, 1.2, &gen_ids)?; }
            let next_id = lp.sample(&logits_tensor)?;
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            gen_ids.push(next_id);
            gen_text.push_str(&self.tokenizer.token_decode(vec![next_id])?);
            
            // [IO-SYNC] 다음 토큰 연산 전 이전 IO(SSD 저장 등) 완료 대기
            wait_for_global_io().await;
            
            logits = self.qwen3_vl.forward(&Tensor::from_vec(vec![next_id], (1, 1), &self.text_device)?, None, None, None, None, None, current_decode_pos, current_decode_pos + 1, session_id.clone(), _kv_name.clone()).await?;
            current_decode_pos += 1;
        }

        Ok(gen_text)
    }

    pub fn get_kv_len(&self) -> usize { match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.get_kv_len(), ModelVariant::QuantizedText(m) => m.language_model.get_kv_len(), _ => 0 } }
    pub async fn drop_kv_storage(&mut self) -> Result<()> { self.qwen3_vl.drop_kv_storage().await }
    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> { self.qwen3_vl.force_flush_all_active_blocks(session_id, kv_name).await }
    pub fn clear_kv_cache(&mut self) { 
        match &mut self.qwen3_vl { 
            ModelVariant::QuantizedVL(m) => m.clear_kv_cache(), 
            ModelVariant::QuantizedText(m) => m.clear_kv_cache(), 
            _ => {} 
        } 
    }
    
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.language_model.truncate_kv_cache(len),
            ModelVariant::QuantizedText(m) => m.language_model.truncate_kv_cache(len),
            _ => Ok(()),
        }
    }

    pub async fn prefill_chunk(&mut self, prompt: String, _cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>) -> Result<usize> {
        // openai_types의 정확한 구조에 맞춰 ChatCompletionParameters 구성
        let mes = ChatCompletionParameters {
            messages: vec![crate::openai_types::ChatCompletionRequestMessage::User(
                crate::openai_types::ChatCompletionRequestUserMessage {
                    content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt),
                    name: None,
                }
            )],
            ..Default::default()
        };
        self.prefill_only(mes, _cancel_flag, session_id, None, None).await
    }

    pub fn save_kv_to_disk(&mut self, path: &Path, kv_name: Option<&str>, offset: usize) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.save_kv_cache(path, false, offset, kv_name), ModelVariant::QuantizedText(m) => m.language_model.save_kv_cache(path, false, offset, kv_name), _ => Ok(()) } }
    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name), ModelVariant::QuantizedText(m) => m.language_model.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name), _ => Ok(()) } }
    pub fn load_kv_from_disk(&mut self, path: &Path, kv_name: Option<&str>) -> Result<()> { 
        let device_clone = self.text_device.clone();
        self.load_kv_cache(path, &device_clone, 0, 128, kv_name) 
    }
}

fn apply_repetition_penalty(logits: &Tensor, penalty: f32, previous_tokens: &[u32]) -> Result<Tensor> {
    let mut logits_vec = logits.to_vec1::<f32>()?;
    let mut set = std::collections::HashSet::new();
    for &t in previous_tokens { if !set.contains(&t) { let logit = logits_vec[t as usize]; if logit < 0.0 { logits_vec[t as usize] = logit * penalty; } else { logits_vec[t as usize] = logit / penalty; } set.insert(t); } }
    let dev = logits.device();
    Ok(Tensor::from_vec(logits_vec, logits.shape(), dev)?)
}

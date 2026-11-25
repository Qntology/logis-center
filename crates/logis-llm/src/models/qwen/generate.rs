use crate::models::qwen::quantized_model::{KVLocation, KVBlock, KVRegistry, BitKVMetadata, QuantizedQwenVLModel, MemorySlot};
use anyhow::{Result, anyhow};
use candle_core::{quantized::gguf_file, DType, Device, Tensor};
use crate::utils::GpuDeviceExt;
use candle_nn::{VarBuilder, Module}; // 🌟 [CRITICAL FIX] Module 트레이트를 추가하여 embed_tokens.forward()를 활성화합니다.
use std::io::Write;

use crate::{
    chat_template::ChatTemplate,
    models::{
        qwen::{
            config::{QwenVLConfig, QwenVLGenerationConfig},
            model::QwenVLModel,
            processor::QwenVLProcessor,
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
    let (sm, rx) = SlotManager::new(128); // 32 -> 128로 증가
    (sm, Mutex::new(Some(rx)))
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
    /// 🌟 [TURBO-KV-BLOCK] Some 이면 raw_k / raw_v 가 TurboQuant 압축 u32 행입니다.
    ///   SSD 에는 형 변환 없이 그대로, `tq_meta` 텐서와 함께 저장합니다.
    pub tq_meta: Option<crate::models::common::kv_turbo::TqMeta>,
}

pub struct BakeTask {
    pub slot_id: usize, pub task_dir: PathBuf, pub kv_name: Option<String>,
    pub offset: usize, pub layers: Vec<LayerKVDump>, pub is_relay_baking: bool,
    pub block_idx: Option<usize>, pub registry: KVRegistry,
}

pub struct SaveTask {
    pub slot_id: usize, 
    pub path: PathBuf, 
    pub tensors: std::collections::HashMap<String, Tensor>,
    pub is_last: bool, 
    pub block_idx: Option<usize>, 
    pub registry: Option<KVRegistry>,
    pub kv_name: Option<String>,
    pub block_len: usize, 
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
        let kv_dir = crate::utils::paths::get_kv_dir();
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
            }
        }
    });
    tx
});

pub struct LoadTask { pub slot_id: usize, pub path: PathBuf, pub layer_idx: usize, pub kv_name: Option<String>, pub shared_block: KVBlock, pub registry: KVRegistry, pub is_cpu: bool }

pub static BAKE_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();
pub static LOAD_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();
use tokio::sync::OnceCell;

pub async fn get_worker_channel() -> Result<mpsc::Sender<SlotTask>> { BAKE_TX.get().cloned().ok_or(anyhow!("Bake init error")) }
pub async fn get_load_worker() -> Result<mpsc::Sender<SlotTask>> { LOAD_TX.get().cloned().ok_or(anyhow!("Load init error")) }
/// I/O 큐 상한선(백프레셔 워터마크). 이 값을 넘을 때만 대기합니다.
pub const IO_BACKPRESSURE_WATERMARK: usize = 64;

/// [BACKPRESSURE] 큐를 0까지 비우지 않고 상한선까지만 기다립니다.
/// 디코딩 루프가 매 토큰 SSD 쓰기 전량 완료를 대기하던 직렬화를 제거합니다.
pub async fn wait_for_global_io() {
    while GLOBAL_IO_COUNTER.load(Ordering::SeqCst) > IO_BACKPRESSURE_WATERMARK {
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
}

/// [DRAIN] 스냅샷 확정처럼 "전부 디스크에 안착했는지"를 보장해야 하는 지점 전용입니다.
pub async fn drain_global_io() {
    while GLOBAL_IO_COUNTER.load(Ordering::SeqCst) > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
}

pub fn init_bake_worker() {
    let (btx, brx) = mpsc::channel(64); let (ltx, lrx) = mpsc::channel(64);
    let _ = BAKE_TX.set(btx); let _ = LOAD_TX.set(ltx);
    if let Some(rx) = SLOT_MANAGER_DATA.1.lock().unwrap().take() { crate::rt::spawn(async move { spawn_slot_dispatcher(rx).await; }); }
    crate::rt::spawn(async move { spawn_slot_worker(brx); }); 
    crate::rt::spawn(async move { spawn_slot_worker(lrx); });
}

async fn spawn_slot_dispatcher(mut rx: mpsc::Receiver<SlotRequest>) {
    // [CRITICAL FIX] 대기표(Queue) 시스템 도입: 슬롯이 부족하다고 매니저가 멈춰서(Block) 기다리는 데드락 원천 차단!
    let mut waiting_acquires = std::collections::VecDeque::new();
    
    while let Some(req) = rx.recv().await {
        match req {
            SlotRequest::Acquire { total_tokens: _total_tokens, tx } => {
                // 슬롯을 당장 안 주고 대기열에 줄부터 세웁니다.
                waiting_acquires.push_back(tx);
            },
            SlotRequest::Release { idx, is_bake } => {
                // 반납 신호가 오면 즉각적으로 슬롯을 비워줍니다.
                let s = &SLOT_MANAGER.slots[idx]; let old = s.state.load(Ordering::SeqCst);
                let new = if is_bake { 2 } else { 0 };
                if s.state.compare_exchange(old, new, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    SLOT_MANAGER.update_counters(old, new);
                    if old == 1 { SLOT_MANAGER.active_write_count.fetch_sub(1, Ordering::SeqCst); }
                    SLOT_MANAGER.handoff_notifier.notify_waiters();
                }
            }
        }

        // [핵심 로직] 여유 슬롯이 있고 대기표를 뽑은 사람이 있다면, 하나씩 슬롯을 나눠줍니다.
        let max_writes = 64;
        while !waiting_acquires.is_empty() && SLOT_MANAGER.active_write_count.load(Ordering::SeqCst) < max_writes {
            let mut found = None;
            
            // 1. 완전히 비어있는 슬롯(0) 탐색
            for (i, slot) in SLOT_MANAGER.slots.iter().enumerate() {
                if slot.state.load(Ordering::SeqCst) == 0 {
                    if slot.state.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() { 
                        SLOT_MANAGER.update_counters(0, 1); 
                        SLOT_MANAGER.active_write_count.fetch_add(1, Ordering::SeqCst); 
                        found = Some(i); break; 
                    }
                }
            }
            
            // 2. 캐시된 슬롯(2) 덮어쓰기 탐색
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

            if let Some(idx) = found {
                // 슬롯을 찾았으면 대기표 1번에게 전달
                let tx = waiting_acquires.pop_front().unwrap();
                let _ = tx.send(idx);
            } else {
                // 슬롯이 없으면 미련 없이 루프를 탈출하여 다음 'Release' 신호를 받을 준비를 합니다. (데드락 회피)
                break; 
            }
        }
    }
}

fn spawn_slot_worker(mut rx: mpsc::Receiver<SlotTask>) {
    let (io_tx, mut io_rx) = mpsc::channel::<SaveTask>(100); 
    tokio::spawn(async move {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(16)); 
        while let Some(task) = io_rx.recv().await {
            let sem = semaphore.clone();
            let (tp, ts, reg, b_idx, sid, is_last, kv_n) = (task.path.clone(), task.tensors, task.registry.clone(), task.block_idx, task.slot_id, task.is_last, task.kv_name.clone());
            let block_len_for_index = task.block_len; 

            tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                struct IoGuard; impl Drop for IoGuard { fn drop(&mut self) { GLOBAL_IO_COUNTER.fetch_sub(1, Ordering::SeqCst); } }
                let _guard = IoGuard;
                
                if let Some(p) = tp.parent() { if !p.exists() { let _ = fs::create_dir_all(p); } }
                
                let tp_clone = tp.clone();
                let serialize_result = tokio::task::spawn_blocking(move || {
                    let tmp_path = tp_clone.with_extension("tmp");
                    candle_core::safetensors::save(&ts, &tmp_path)?;
                    
                    std::fs::rename(&tmp_path, &tp_clone)?;
                    
                    Ok::<_, anyhow::Error>(())
                }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("Thread error: {}", e)));

                match serialize_result {
                    Ok(_) => {
                        let parsed_layer_idx = tp.file_name()
                            .and_then(|n| n.to_str())
                            .and_then(|s| s.strip_prefix('l'))
                            .and_then(|s| s.strip_suffix(".st"))
                            .and_then(|s| s.parse::<usize>().ok());

                        if let (Some(r), Some(idx), Some(layer_num)) = (reg, b_idx, parsed_layer_idx) {
                            if let Ok(mut entries) = r.entries.write() {
                                if idx < entries.len() {
                                    let e = &mut entries[idx]; 
                                    e.location[layer_num] = KVLocation::SSD; 
                                    e.ssd_path = Some(tp.parent().unwrap().to_path_buf());
                                }
                            }
                        }
                        
                        if let (Some(kv_name), Some(l_num)) = (kv_n, parsed_layer_idx) {
                            let offset_str = tp.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()).and_then(|s| s.strip_prefix('b')).unwrap_or("0");
                            let offset = offset_str.parse::<usize>().unwrap_or(0);
                            let _ = INDEX_TX.send(SlotTask::IndexUpdate {
                                kv_name, layer_idx: l_num, offset, len: block_len_for_index, file_name: format!("b{}/l{}.st", offset, l_num),
                            }).await;
                        }
                    },
                    Err(e) => {
                        println!("[CRITICAL-IO] Failed to serialize tensor for path: {:?}. Error: {}", tp, e);
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
                    let loop_count = bake.layers.len();

                    GLOBAL_IO_COUNTER.fetch_add(loop_count, std::sync::atomic::Ordering::SeqCst);
                    GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

                    let io_tx_inner = io_tx.clone();
                    let (sid, off, _is_relay, block_idx, registry, kv_name) = (bake.slot_id, bake.offset, bake.is_relay_baking, bake.block_idx, bake.registry.clone(), bake.kv_name.clone());
                    
                    SLOT_MANAGER.slots[sid].remaining_layers.store(loop_count, Ordering::SeqCst);
                    
                    for l_idx in 0..loop_count {
                        let mut src = bake.layers[l_idx].clone();
                        let act_l = src.layer_idx;
                        let task_dir = bake.task_dir.clone();
                        let registry_inner = registry.clone();
                        let kv_name_inner = kv_name.clone();
                        let io_tx_nested = io_tx_inner.clone();

                        tokio::spawn(async move {
                            let tq_meta = src.tq_meta;
                            if let (Some(rk), Some(rv), Some(_)) = (src.raw_k.as_ref(), src.raw_v.as_ref(), tq_meta) {
                                // 🌟 [TURBO-KV-BLOCK] TurboQuant 압축 블록은 u32 행을 그대로 저장합니다.
                                //    (F32 변환이나 헤드/차원 정렬을 하면 압축 데이터가 깨집니다)
                                let k_cpu = rk.to_device(&Device::Cpu).unwrap_or_else(|_| rk.clone());
                                let v_cpu = rv.to_device(&Device::Cpu).unwrap_or_else(|_| rv.clone());
                                src.k_data = k_cpu.contiguous().unwrap_or(k_cpu);
                                src.v_data = v_cpu.contiguous().unwrap_or(v_cpu);
                                src.raw_k = None;
                                src.raw_v = None;
                            }
                            if let (Some(rk), Some(rv)) = (src.raw_k.take(), src.raw_v.take()) {
                                let k_cpu = rk.to_device(&Device::Cpu).unwrap_or(rk);
                                let v_cpu = rv.to_device(&Device::Cpu).unwrap_or(rv);

                                let mut k_aligned = k_cpu.clone();
                                let mut v_aligned = v_cpu.clone();

                                let (_b, h, _s, d) = k_cpu.dims4().unwrap_or((1, 0, 0, 0));
                                let k_shape_u32 = src.k_shape.to_vec1::<u32>().unwrap_or_default();
                                if k_shape_u32.len() == 4 {
                                    let target_heads = k_shape_u32[1] as usize;
                                    let target_dim = k_shape_u32[3] as usize;

                                    if d > 0 && d < target_dim {
                                        k_aligned = Tensor::cat(&[&k_cpu, &k_cpu], candle_core::D::Minus1).unwrap_or(k_cpu.clone());
                                        v_aligned = Tensor::cat(&[&v_cpu, &v_cpu], candle_core::D::Minus1).unwrap_or(v_cpu.clone());
                                    }
                                    if h > 0 && h != target_heads {
                                        let mut k_list = Vec::with_capacity(target_heads);
                                        let mut v_list = Vec::with_capacity(target_heads);
                                        for i in 0..target_heads {
                                            let src_idx = i % h;
                                            k_list.push(k_aligned.narrow(1, src_idx, 1).unwrap());
                                            v_list.push(v_aligned.narrow(1, src_idx, 1).unwrap());
                                        }
                                        k_aligned = Tensor::cat(&k_list, 1).unwrap_or(k_aligned);
                                        v_aligned = Tensor::cat(&v_list, 1).unwrap_or(v_aligned);
                                    }
                                }

                                let k_contig = k_aligned.contiguous().unwrap_or(k_aligned);
                                let v_contig = v_aligned.contiguous().unwrap_or(v_aligned);

                                // /qwen/ 은 F32 로직, /qwen3/, /qwen3_5/ 은 FP4 로직으로 동적 분기
                                let target_dtype = if k_contig.dtype() == DType::F4 { DType::F4 } else { DType::F32 };
                                src.k_data = k_contig.to_dtype(target_dtype).unwrap_or_else(|_| k_contig.clone());
                                src.v_data = v_contig.to_dtype(target_dtype).unwrap_or_else(|_| v_contig.clone());
                            }
                            let mut map = std::collections::HashMap::new();
                            let prefix = format!("b{}_l{}_", off, act_l);
                            map.insert(format!("{}k_data", prefix), src.k_data.clone());
                            map.insert(format!("{}v_data", prefix), src.v_data.clone());
                            map.insert(format!("{}k_shape", prefix), src.k_shape.clone());
                            // 🌟 [TURBO-KV-BLOCK] 압축 형식(형식·비트·회전·시드)을 함께 기록해 로드 시 코덱을 다시 만듭니다.
                            //    이 텐서가 없는 파일은 기존 FP8/F32 블록으로 읽힙니다.
                            if let Some(meta) = tq_meta {
                                if let Ok(t) = meta.to_tensor() {
                                    map.insert(format!("{}tq_meta", prefix), t);
                                }
                            }

                            let file_path = task_dir.join(format!("l{}.st", act_l));
                            
                            
                            let b_len = src.k_shape.to_vec1::<u32>().unwrap_or_default().get(2).cloned().unwrap_or(256) as usize;
                            
                            if io_tx_nested.send(SaveTask { 
                                slot_id: sid, path: file_path.clone(), tensors: map, is_last: false, block_idx, registry: Some(registry_inner), kv_name: kv_name_inner,
                                block_len: b_len 
                            }).await.is_err() {
                                GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                            }
                        });
                    }
                },
                SlotTask::Load(load) => {
                    let sid = load.slot_id;
                    let reg = load.registry.clone(); 
                    let shared_block = load.shared_block.clone();
                    let provided_path = load.path.clone(); 
                    let target_layer = load.layer_idx; 

                    tokio::spawn(async move {
                        let _guard = ReadSlotGuard { sid, active: true };
                        let (b_idx_off, b_idx) = { match shared_block.inner.read() { Ok(inner) => (inner.offset, inner.index), _ => (0, 999) } };
                        
                        let file_path = provided_path.join(format!("l{}.st", target_layer));
                        if file_path.is_file() {
                            if let Ok(content) = load_kv_block(&file_path) {
                                if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                                    let prefix = format!("b{}_l{}_", b_idx_off, target_layer);
                                    let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).ok();
                                
                                    // 🌟 [TURBO-KV-BLOCK] TurboQuant 압축 블록(Qwen3.5 전용 형식)은 이 경로의 소비자
                                    //    (QuantizedQwen 어텐션)가 풀 수 없으므로 RAM 캐시에 올리지 않습니다.
                                    if get_t("tq_meta").is_some() {
                                        println!("[TURBO-KV-BLOCK] {:?} 는 TurboQuant 압축 블록이라 이 로더에서는 건너뜁니다.", file_path);
                                        return;
                                    }
                                    if let (Some(kd), Some(vd), Some(sh)) = (get_t("k_data"), get_t("v_data"), get_t("k_shape")) {
                                        let sh_u32: Vec<u32> = sh.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                                        let file_shape: Vec<usize> = sh_u32.iter().map(|&x| x as usize).collect();

                                        let dev = &Device::Cpu;
                                        // 저장된 파일의 DType을 감지하여 FP4, F32 모두 호환되도록 로드합니다.
                                        let saved_dtype = match kd.dtype() {
                                            safetensors::Dtype::F64 => DType::F4,
                                            safetensors::Dtype::F32 => DType::F32,
                                            safetensors::Dtype::F16 => DType::F16,
                                            _ => DType::BF16,
                                        };
                                        
                                        let mut kd_t = Tensor::from_raw_buffer(kd.data(), saved_dtype, &file_shape, dev).unwrap_or_else(|_| Tensor::zeros(file_shape.clone(), saved_dtype, dev).unwrap());
                                        let mut vd_t = Tensor::from_raw_buffer(vd.data(), saved_dtype, &file_shape, dev).unwrap_or_else(|_| Tensor::zeros(file_shape.clone(), saved_dtype, dev).unwrap());
                                        
                                        if load.is_cpu {
                                            kd_t = kd_t.to_dtype(DType::F32).unwrap_or(kd_t);
                                            vd_t = vd_t.to_dtype(DType::F32).unwrap_or(vd_t);
                                        }
                                        let meta = BitKVMetadata { k_data: kd_t, v_data: vd_t, original_shape: file_shape, tq: None };
                                        
                                        if let Ok(mut r) = reg.entries.write() {
                                            if b_idx < r.len() {
                                                {
                                                    let mut cache = r[b_idx].bitkv_cache.write().unwrap();
                                                    cache[target_layer] = Some(meta);
                                                }
                                                r[b_idx].location[target_layer] = KVLocation::RAM;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });
                },
                SlotTask::IndexUpdate { .. } => {}
            }
        }
    });
}

struct ReadSlotGuard { sid: usize, active: bool }
impl Drop for ReadSlotGuard { fn drop(&mut self) { if self.active { let sid = self.sid; crate::rt::spawn(async move { SLOT_MANAGER.release_slot(sid).await; }); } } }

fn hash_str_u64(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn build_json_protected_ids(tokenizer: &TokenizerModel) -> std::collections::HashSet<u32> {
    let mut protected = std::collections::HashSet::new();
    for (tok_str, &id) in tokenizer.tokenizer.get_vocab(true).iter() {
        let maybe_punct = tok_str.len() <= 16
            && (tok_str.starts_with("<0x")
                || tok_str
                    .bytes()
                    .any(|b| matches!(b, b'"' | b'{' | b'[' | b'}' | b']' | b',' | b':')));
        if !maybe_punct {
            continue;
        }
        if let Ok(piece) = tokenizer.token_decode(vec![id]) {
            let p = piece.trim();
            if matches!(p, "\"" | "{" | "[" | "}" | "]" | "," | ":") {
                protected.insert(id);
            }
        }
    }
    protected
}

fn build_json_starter_ids(tokenizer: &TokenizerModel) -> std::collections::HashSet<u32> {
    let mut starters = std::collections::HashSet::new();
    for (tok_str, &id) in tokenizer.tokenizer.get_vocab(true).iter() {
        if tok_str.contains('{')
            || tok_str.contains('[')
            || tok_str.starts_with("<0x")
            || tok_str.starts_with('\u{0120}')
        {
            if let Ok(piece) = tokenizer.token_decode(vec![id]) {
                let trimmed = piece.trim_start_matches(|c: char| c == ' ' || c == '\t');
                if (trimmed.starts_with('{') || trimmed.starts_with('[')) && !trimmed.contains("//") {
                    starters.insert(id);
                }
            }
        }
    }
    if let Ok(v) = tokenizer.text_encode_vec("{".to_string(), false) {
        if let Some(&id) = v.first() {
            starters.insert(id);
        }
    }
    starters
}

fn backend_label_for(dev: &Device) -> &'static str {
    if dev.is_rocm() {
        "rocm"
    } else if dev.is_cuda() {
        "cuda"
    } else if dev.is_vulkan() {
        "vulkan"
    } else if dev.is_metal() {
        "metal"
    } else {
        "cpu"
    }
}

#[derive(Clone)]
pub enum ModelVariant { Standard(QwenVLModel), QuantizedVL(QuantizedQwenVLModel), QuantizedText(crate::models::qwen::quantized_model::QuantizedQwenTextModel) }

impl ModelVariant {
    // 🌟 [추가] Semantic Bias 연산을 위해 전체 단어장의 벡터(Weight)를 그대로 반환합니다.
    pub fn get_embed_tokens(&self) -> Result<Tensor> {
        match self {
            Self::Standard(m) => Ok(m.language_model.embed_tokens.embeddings().clone()),
            Self::QuantizedVL(m) => Ok(m.language_model.embed_tokens.embeddings().clone()),
            Self::QuantizedText(m) => Ok(m.language_model.embed_tokens.embeddings().clone()),
        }
    }

    pub fn embedding_token_id(&self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(m) => Ok(m.language_model.embed_tokens.forward(input_ids)?),
            Self::QuantizedVL(m) => Ok(m.language_model.embed_tokens.forward(input_ids)?),
            Self::QuantizedText(m) => Ok(m.language_model.embed_tokens.forward(input_ids)?),
        }
    }

    pub fn trim_head(&self) -> Option<&crate::models::common::vocab_trim::TrimmedHead> {
        match self {
            Self::QuantizedVL(m) => m.trim_head(),
            Self::QuantizedText(m) => m.trim_head(),
            Self::Standard(_) => None,
        }
    }

    pub fn head_src_ref(&self) -> Option<&crate::models::common::vocab_trim::TableSource> {
        match self {
            Self::QuantizedVL(m) => m.head_src_ref(),
            Self::QuantizedText(m) => m.head_src_ref(),
            Self::Standard(_) => None,
        }
    }

    pub fn embed_vocab(&self) -> Result<usize> {
        match self {
            Self::Standard(m) => Ok(m.language_model.embed_tokens.embeddings().dim(0)?),
            Self::QuantizedVL(m) => Ok(m.language_model.embed_tokens.embeddings().dim(0)?),
            Self::QuantizedText(m) => Ok(m.language_model.embed_tokens.embeddings().dim(0)?),
        }
    }

    pub fn embed_kept_rows_f32(&self, off: usize, take: usize, device: &Device) -> Result<Tensor> {
        if let Some(head) = self.trim_head() {
            Ok(head.kept_rows_f32(off, take, device)?)
        } else {
            Ok(self
                .get_embed_tokens()?
                .narrow(0, off, take)?
                .to_dtype(DType::F32)?
                .to_device(device)?)
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::QuantizedVL(_) | Self::QuantizedText(_))
    }

    pub fn resident_vram_bytes(&self) -> u64 {
        let head_bytes = self.trim_head().map(|h| h.head_bytes() as u64).unwrap_or(0);
        match self {
            Self::QuantizedVL(m) => {
                let emb = m.language_model.embed_tokens.embeddings();
                let h = emb.dim(1).unwrap_or(1024) as u64;
                let layers = m.language_model.layers.len() as u64;
                let emb_b = if m.head_src.is_some() {
                    head_bytes
                } else {
                    (emb.elem_count() * emb.dtype().size_in_bytes()) as u64
                };
                emb_b + layers * h * h * 12
            }
            Self::QuantizedText(m) => {
                let emb = m.language_model.embed_tokens.embeddings();
                let h = emb.dim(1).unwrap_or(1024) as u64;
                let layers = m.language_model.layers.len() as u64;
                let emb_b = if m.head_src.is_some() {
                    head_bytes
                } else {
                    (emb.elem_count() * emb.dtype().size_in_bytes()) as u64
                };
                emb_b + layers * h * h * 12
            }
            Self::Standard(m) => {
                let emb = m.language_model.embed_tokens.embeddings();
                let h = emb.dim(1).unwrap_or(1024) as u64;
                let layers = m
                    .config
                    .text_config
                    .as_ref()
                    .map(|tc| tc.num_hidden_layers)
                    .unwrap_or(28) as u64;
                (emb.elem_count() * emb.dtype().size_in_bytes()) as u64 + layers * h * h * 24
            }
        }
    }

    pub async fn forward_last_hidden(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        video_pixel_values: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        cache_position: Option<&Tensor>,
        seqlen_offset: usize,
        total_len: usize,
        session_id: Option<String>,
        kv_name: Option<String>,
    ) -> Result<Tensor> {
        match self {
            Self::QuantizedVL(m) => {
                m.forward_last_hidden(
                    input_ids,
                    pixel_values,
                    image_grid_thw,
                    video_pixel_values,
                    video_grid_thw,
                    cache_position,
                    seqlen_offset,
                    total_len,
                    session_id,
                    kv_name,
                )
                .await
            }
            Self::QuantizedText(m) => {
                m.forward_last_hidden(
                    input_ids,
                    cache_position,
                    seqlen_offset,
                    total_len,
                    session_id,
                    kv_name,
                )
                .await
            }
            Self::Standard(_) => {
                Err(anyhow!("Standard QwenVLModel does not support forward_last_hidden"))
            }
        }
    }

    pub fn logits_from_hidden(&self, hidden: &Tensor) -> Result<Tensor> {
        match self {
            Self::QuantizedVL(m) => m.logits_from_hidden(hidden),
            Self::QuantizedText(m) => m.logits_from_hidden(hidden),
            Self::Standard(_) => {
                Err(anyhow!("Standard QwenVLModel does not support logits_from_hidden"))
            }
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

    /// 🌟 [VISION-JIT] mmproj 재구성 소스가 등록되어 JIT 이 가능한 변형인지 확인합니다.
    /// QuantizedText / Standard 는 애초에 mmproj 개념이 없으므로 항상 false 입니다.
    pub fn is_vision_jit_capable(&self) -> bool {
        match self {
            Self::QuantizedVL(m) => m.is_vision_jit_capable(),
            _ => false,
        }
    }

    /// 🌟 [VISION-JIT] 현재 비전 가중치가 메모리에 상주 중인지 확인합니다.
    pub fn is_vision_resident(&self) -> bool {
        match self {
            Self::QuantizedVL(m) => m.is_vision_resident(),
            _ => false,
        }
    }

    /// 🌟 [VISION-JIT] 텍스트 모델은 그대로 둔 채 비전 가중치만 붙였다 뗍니다.
    pub fn set_vision_active(&mut self, active: bool) -> Result<()> {
        match self {
            Self::QuantizedVL(m) => m.set_vision_active(active),
            _ => Ok(()),
        }
    }
    pub fn get_current_kv(&self) -> (Vec<Tensor>, Vec<Tensor>) { match self { Self::QuantizedVL(m) => m.language_model.get_current_kv(), Self::QuantizedText(m) => m.language_model.get_current_kv(), _ => (vec![], vec![]) } }
    pub fn inject_kv_bitkv(&mut self, kd: &[Tensor], vd: &[Tensor], os: &[usize]) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.inject_live_kv_bitkv(kd, vd, os), Self::QuantizedText(m) => m.language_model.inject_live_kv_bitkv(kd, vd, os), _ => Ok(()) } }
    pub async fn drop_kv_storage(&mut self) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.drop_kv_storage(), Self::QuantizedText(m) => m.language_model.drop_kv_storage(), _ => Ok(()) } }
    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.force_flush_all_active_blocks(session_id, kv_name).await, Self::QuantizedText(m) => m.language_model.force_flush_all_active_blocks(session_id, kv_name).await, _ => Ok(()) } }
}

pub struct QwenVLGenerateModel {
    pub chat_template: ChatTemplate,
    pub tokenizer: TokenizerModel,
    pub pre_processor: QwenVLProcessor,
    pub qwen: ModelVariant,
    pub text_device: Device,
    pub vision_device: Device,
    pub eos_token_id1: u32,
    pub eos_token_id2: u32,
    pub generation_config: QwenVLGenerationConfig,
    pub model_name: String,
    pub hard_token_limit: Option<usize>,
    pub kv_root: std::path::PathBuf,
    pub json_protected_ids: std::collections::HashSet<u32>,
    pub json_starter_ids: std::collections::HashSet<u32>,
    pub recipe: String,
}

impl QwenVLGenerateModel {
    /// 🌟 [VOCAB TRIM] 지금의 문서 언어(`vocab_trim::DocLangScope`)로 출력 투영의 남긴 행을 맞춥니다. 생성 · 미리 읽기마다 처음에 부릅니다.
    pub fn refresh_vocab(&mut self) -> Result<()> {
        self.refresh_vocab_for("", false)
    }

    /// [`Self::refresh_vocab`] + 이번 생성의 프롬프트 글에 나온 문자 체계. 샘플링(온도 > 0)이면 전체 행 (`vocab_trim::refresh_for`)
    pub fn refresh_vocab_for(&mut self, prompt: &str, sampling: bool) -> Result<()> {
        let tok = &self.tokenizer.tokenizer;
        let model_hint = Some(self.model_name.as_str());
        match &mut self.qwen {
            ModelVariant::QuantizedVL(m) => {
                if let Some(src) = m.head_src.clone() {
                    crate::models::common::vocab_trim::refresh_for_ids(
                        &mut m.trim_head,
                        &src,
                        tok,
                        &m.text_device,
                        prompt,
                        &[],
                        model_hint,
                        sampling,
                    )?;
                }
            }
            ModelVariant::QuantizedText(m) => {
                if let Some(src) = m.head_src.clone() {
                    crate::models::common::vocab_trim::refresh_for_ids(
                        &mut m.trim_head,
                        &src,
                        tok,
                        &m.text_device,
                        prompt,
                        &[],
                        model_hint,
                        sampling,
                    )?;
                }
            }
            ModelVariant::Standard(_) => {}
        }
        Ok(())
    }

    fn refresh_vocab_from_payload_ids(
        &mut self,
        payload_text: &str,
        payload_ids: &[u32],
        sampling: bool,
    ) -> Result<()> {
        let tok = &self.tokenizer.tokenizer;
        let model_hint = Some(self.model_name.as_str());
        match &mut self.qwen {
            ModelVariant::QuantizedVL(m) => {
                if let Some(src) = m.head_src.clone() {
                    crate::models::common::vocab_trim::refresh_for_ids(
                        &mut m.trim_head,
                        &src,
                        tok,
                        &m.text_device,
                        payload_text,
                        payload_ids,
                        model_hint,
                        sampling,
                    )?;
                }
            }
            ModelVariant::QuantizedText(m) => {
                if let Some(src) = m.head_src.clone() {
                    crate::models::common::vocab_trim::refresh_for_ids(
                        &mut m.trim_head,
                        &src,
                        tok,
                        &m.text_device,
                        payload_text,
                        payload_ids,
                        model_hint,
                        sampling,
                    )?;
                }
            }
            ModelVariant::Standard(_) => {}
        }
        Ok(())
    }

    fn encode_with_spans(&self, text: &str) -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
        let enc = self
            .tokenizer
            .tokenizer
            .encode(text, false)
            .map_err(anyhow::Error::msg)?;
        Ok((enc.get_ids().to_vec(), enc.get_offsets().to_vec()))
    }

    fn collect_payload_text_and_ids(
        &self,
        mes_render: &str,
        user_text: &str,
        ids_vec: &[u32],
        spans: &[(usize, usize)],
    ) -> (String, Vec<u32>) {
        let ranges = crate::models::common::vocab_trim::extract_data_payload_ranges(mes_render);
        if ranges.is_empty() {
            return (user_text.to_string(), ids_vec.to_vec());
        }
        let mut payload_text = String::new();
        let mut payload_ids = Vec::new();
        for &(rs, re) in &ranges {
            let chunk = mes_render[rs..re].trim();
            if !chunk.is_empty() {
                if !payload_text.is_empty() {
                    payload_text.push('\n');
                }
                payload_text.push_str(chunk);
            }
            for (&tid, &(ts, te)) in ids_vec.iter().zip(spans.iter()) {
                if te > rs && ts < re {
                    payload_ids.push(tid);
                }
            }
        }
        if payload_text.is_empty() {
            (user_text.to_string(), ids_vec.to_vec())
        } else {
            (payload_text, payload_ids)
        }
    }

    fn slot_label(&self) -> &'static str {
        if self.model_name == "0.6B" {
            "qwen_0_6b"
        } else {
            "qwen_vl"
        }
    }

    fn record_load_reflection(
        &self,
        load_ms: f32,
        vram_before: Option<crate::utils::device_utils::DeviceVramSnapshot>,
        vram_after: Option<crate::utils::device_utils::DeviceVramSnapshot>,
    ) {
        if logis_schema::reflection::ReflectionLogMode::from_env()
            == logis_schema::reflection::ReflectionLogMode::Off
        {
            return;
        }
        let vocab_v = self.qwen.embed_vocab().unwrap_or(0) as u32;
        let active_k = self
            .qwen
            .trim_head()
            .map(|h| h.kept.len() as u32)
            .unwrap_or(vocab_v);
        let (table_mode, resident_table_mb, mapped_table_mb) = match self.qwen.head_src_ref() {
            Some(src) => {
                let row_bytes = (src.cols / src.dtype.block_size().max(1)) * src.dtype.type_size();
                let total_bytes = (src.rows * row_bytes) as f32;
                let head_bytes = self
                    .qwen
                    .trim_head()
                    .map_or(0.0, |h| h.head_bytes() as f32);
                ("mapped".to_string(), head_bytes / 1e6, total_bytes / 1e6)
            }
            None => (
                if self.qwen.is_quantized() {
                    "quant".to_string()
                } else {
                    "resident".to_string()
                },
                self.qwen.resident_vram_bytes() as f32 / 1e6,
                0.0,
            ),
        };
        let rec = logis_schema::reflection::ReflectionRecord {
            ts_ms: logis_schema::reflection::current_ts_ms(),
            run_id: logis_schema::reflection::current_run_id(),
            task_id: self.model_name.clone(),
            git_rev: option_env!("LOGIS_GIT_REV").unwrap_or("dev").into(),
            build_profile: logis_schema::reflection::build_profile_label(),
            features: Vec::new(),
            backend: backend_label_for(&self.text_device).into(),
            device_name: format!("{:?}", self.text_device),
            artifact: logis_schema::reflection::ArtifactReflection {
                slot: self.slot_label().into(),
                bytes: self.qwen.resident_vram_bytes(),
                sha256_16: String::new(),
                recipe: self.recipe.clone(),
                vocab_v,
                artifact_k: active_k,
                plan_sha256: None,
                space_tag: None,
            },
            load: Some(logis_schema::reflection::LoadReflection {
                load_ms,
                vram_before_mb: vram_before.map_or(0.0, |s| s.resident_used_mb_f32()),
                vram_after_mb: vram_after.map_or(0.0, |s| s.resident_used_mb_f32()),
                table_mode,
                resident_table_mb,
                mapped_table_mb,
                env_overrides: Vec::new(),
                accel_fallbacks: crate::models::common::accel_check::recorded_fallbacks(
                    &self.text_device,
                ),
                route_block_reason: None,
            }),
            exec: None,
        };
        logis_schema::reflection::record(&rec);
    }

    fn prejudice_cached_active(
        &self,
        target_text: &str,
    ) -> Result<crate::models::common::prejudice::CachedPrejudice> {
        let target_ids = self.tokenizer.text_encode_vec(target_text.to_string(), false)?;
        if target_ids.is_empty() {
            return Err(anyhow!("empty target tokens"));
        }
        let target_tensor =
            Tensor::from_vec(target_ids.clone(), (1, target_ids.len()), &self.text_device)?;
        let target_emb = self
            .qwen
            .embedding_token_id(&target_tensor)?
            .to_dtype(DType::F32)?;
        let target_emb_sum = target_emb.sum_keepdim(1)?;
        let len_tensor = Tensor::new(target_ids.len() as f32, &self.text_device)?;
        let target_vec = target_emb_sum
            .broadcast_div(&len_tensor)?
            .squeeze(0)?
            .squeeze(0)?;

        let vocab_label = self
            .qwen
            .trim_head()
            .map(|h| format!("{}:{:016x}", h.label, h.kept_hash))
            .unwrap_or_else(|| "full".into());
        let num_rows = self
            .qwen
            .trim_head()
            .map(|h| h.kept.len())
            .unwrap_or(self.qwen.embed_vocab()?);
        let cache_key = format!("qwenvl:k:{:016x}:{vocab_label}", hash_str_u64(target_text));
        crate::models::common::prejudice::get_or_build(
            &cache_key,
            &target_vec,
            num_rows,
            &self.text_device,
            |off, take| self.qwen.embed_kept_rows_f32(off, take, &self.text_device),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn select_token_from_hidden(
        &mut self,
        hidden: &Tensor,
        step_idx: usize,
        is_strict_json: bool,
        temperature: f64,
        penalty: f32,
        logit_processor: &mut candle_transformers::generation::LogitsProcessor,
        penalized_history: &[u32],
        gen_text: &str,
        enter_id: u32,
        think_token_id: u32,
        open_bracket_id: u32,
        lt_id: u32,
        slash_id: u32,
        double_slash_id: u32,
        space_double_slash_id: u32,
        active_prej: Option<&crate::models::common::prejudice::CachedPrejudice>,
        full_prej: Option<&Tensor>,
        scratch: &mut crate::models::common::vocab_trim::DecodeStepScratch,
        guard_stats: &mut crate::models::common::vocab_trim::StepGuardStats,
    ) -> Result<u32> {
        scratch.banned_ids.clear();
        scratch.banned_ids.push(think_token_id);
        if is_strict_json {
            let is_url_single = gen_text.ends_with("http:/") || gen_text.ends_with("https:/");
            let is_url_double = gen_text.ends_with("http:") || gen_text.ends_with("https:");
            if !is_url_single && gen_text.ends_with('/') {
                scratch.banned_ids.push(slash_id);
            }
            if !is_url_double {
                scratch.banned_ids.push(double_slash_id);
                scratch.banned_ids.push(space_double_slash_id);
            }
        }
        if step_idx == 0 {
            scratch.banned_ids.push(self.eos_token_id1);
            scratch.banned_ids.push(self.eos_token_id2);
            scratch.banned_ids.push(enter_id);
            if is_strict_json {
                if let Some(head) = self.qwen.trim_head() {
                    for &vid in head.kept.iter() {
                        if !self.json_starter_ids.contains(&vid) {
                            scratch.banned_ids.push(vid);
                        }
                    }
                }
            }
        }

        let recent_start = penalized_history.len().saturating_sub(64);
        let seen_tokens = &penalized_history[recent_start..];
        let fallback_eos = if is_strict_json && step_idx == 0 {
            open_bracket_id
        } else {
            self.eos_token_id1
        };

        if let Some(head) = self.qwen.trim_head() {
            let banned_ids = std::mem::take(&mut scratch.banned_ids);
            let pp = crate::models::common::vocab_trim::StepPostProcess {
                seen_tokens,
                repeat_penalty: penalty,
                banned_ids: &banned_ids,
                prejudice_kept: active_prej.map(|cp| &cp.values),
                target_norm_cpu: active_prej.map(|cp| cp.target_norm_cpu.as_slice()),
                fallback_eos_id: fallback_eos,
            };
            let mut cand = if temperature <= 1e-7 {
                let (tok, st) = head.greedy_single_pass_k(hidden, &pp, scratch)?;
                guard_stats.merge(&st);
                tok
            } else {
                guard_stats.steps += 1;
                head.sample_single_pass_k(hidden, logit_processor, &pp, scratch)?
            };
            scratch.banned_ids = banned_ids;
            if step_idx == 0 {
                if is_strict_json && !self.json_starter_ids.contains(&cand) {
                    cand = open_bracket_id;
                } else if !is_strict_json
                    && (cand == self.eos_token_id1 || cand == self.eos_token_id2)
                {
                    cand = open_bracket_id;
                }
            }
            return Ok(cand);
        }

        let logits = self.qwen.logits_from_hidden(hidden)?;
        self.select_token_from_logits(
            &logits,
            step_idx,
            is_strict_json,
            logit_processor,
            penalized_history,
            gen_text,
            enter_id,
            think_token_id,
            open_bracket_id,
            lt_id,
            slash_id,
            double_slash_id,
            space_double_slash_id,
            full_prej,
            guard_stats,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn select_token_from_logits(
        &self,
        logits: &Tensor,
        step_idx: usize,
        is_strict_json: bool,
        logit_processor: &mut candle_transformers::generation::LogitsProcessor,
        penalized_history: &[u32],
        gen_text: &str,
        enter_id: u32,
        think_token_id: u32,
        open_bracket_id: u32,
        lt_id: u32,
        slash_id: u32,
        double_slash_id: u32,
        space_double_slash_id: u32,
        full_prej: Option<&Tensor>,
        guard_stats: &mut crate::models::common::vocab_trim::StepGuardStats,
    ) -> Result<u32> {
        let logits_squeezed = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits_steered = if let Some(prej) = full_prej {
            logits_squeezed.broadcast_sub(prej)?
        } else {
            logits_squeezed
        };
        let mut logits_vec = logits_steered.flatten_all()?.to_vec1::<f32>()?;
        let len = logits_vec.len();

        let penalty = 1.2f32;
        if !penalized_history.is_empty() {
            let mut set = std::collections::HashSet::new();
            let start_at = penalized_history.len().saturating_sub(64);
            for &t in &penalized_history[start_at..] {
                if !set.contains(&t) && (t as usize) < len {
                    let logit = logits_vec[t as usize];
                    logits_vec[t as usize] = if logit < 0.0 {
                        logit * penalty
                    } else {
                        logit / penalty
                    };
                    set.insert(t);
                }
            }
        }

        if (think_token_id as usize) < len {
            logits_vec[think_token_id as usize] -= 1000.0;
        }
        if (lt_id as usize) < len {
            logits_vec[lt_id as usize] -= 10.0;
        }
        if is_strict_json {
            let is_url_single = gen_text.ends_with("http:/") || gen_text.ends_with("https:/");
            let is_url_double = gen_text.ends_with("http:") || gen_text.ends_with("https:");
            if !is_url_single && gen_text.ends_with('/') {
                if (slash_id as usize) < len {
                    logits_vec[slash_id as usize] -= 10000.0;
                }
            }
            if !is_url_double {
                if (double_slash_id as usize) < len {
                    logits_vec[double_slash_id as usize] -= 10000.0;
                }
                if (space_double_slash_id as usize) < len {
                    logits_vec[space_double_slash_id as usize] -= 10000.0;
                }
            }
        }

        if step_idx == 0 {
            if (self.eos_token_id1 as usize) < len {
                logits_vec[self.eos_token_id1 as usize] = -10000.0;
            }
            if (self.eos_token_id2 as usize) < len {
                logits_vec[self.eos_token_id2 as usize] = -10000.0;
            }
            if (enter_id as usize) < len {
                logits_vec[enter_id as usize] -= 50.0;
            }
            if is_strict_json {
                for (j, slot) in logits_vec.iter_mut().enumerate() {
                    if !self.json_starter_ids.contains(&(j as u32)) {
                        *slot = f32::NEG_INFINITY;
                    }
                }
            } else if (open_bracket_id as usize) < len {
                logits_vec[open_bracket_id as usize] += 20.0;
            }
        }

        let logits_tensor = Tensor::from_vec(logits_vec, (len,), &Device::Cpu)?;
        let mut next_token = logit_processor.sample(&logits_tensor)?;
        guard_stats.steps += 1;

        if step_idx == 0 {
            if is_strict_json && !self.json_starter_ids.contains(&next_token) {
                next_token = open_bracket_id;
            } else if !is_strict_json
                && (next_token == self.eos_token_id1 || next_token == self.eos_token_id2)
            {
                next_token = open_bracket_id;
            }
        }

        Ok(next_token)
    }

    /// config.json에서 로드된 모델의 최대 토큰 한계치를 반환합니다.
    pub fn get_max_tokens(&self) -> usize {
        match &self.qwen {
            ModelVariant::QuantizedVL(m) => m.language_model.config.max_position_embeddings,
            ModelVariant::QuantizedText(m) => m.language_model.config.max_position_embeddings,
            ModelVariant::Standard(m) => m.config.text_config.as_ref().map(|tc| tc.max_position_embeddings).unwrap_or(32768),
        }
    }

    /// 🌟 [VISION-JIT] mmproj 를 mmap 에서 다시 구성할 수 있는 상태인지 확인합니다.
    pub fn is_vision_jit_capable(&self) -> bool {
        self.qwen.is_vision_jit_capable()
    }

    /// 🌟 [VISION-JIT] 현재 비전 가중치가 메모리에 상주 중인지 확인합니다.
    pub fn vision_resident(&self) -> bool {
        self.qwen.is_vision_resident()
    }

    /// 🌟 [VISION-JIT] 0.6B 텍스트 모델은 유지한 채 비전 가중치만 탈부착합니다.
    pub fn set_vision_active(&mut self, active: bool) -> Result<()> {
        self.qwen.set_vision_active(active)?;

        if !active {
            if self.text_device.is_cuda_or_rocm() { let _ = self.text_device.synchronize(); }
            crate::utils::legacy_working_set_trim(); // 🌟 [RAM-SAWTOOTH] 기본은 작업 집합을 내쫓지 않습니다 (LOGIS_MEM_TRIM=legacy)
        }
        Ok(())
    }

    pub fn init_with_config(path: &str, tokenizer_path: Option<&str>, config_path: Option<&str>, text_device: Option<&Device>, text_device_id: usize, vision_device: Option<&Device>, vision_device_id: usize, dtype: Option<DType>, hard_token_limit: Option<usize>, force_text_only: bool, baking_only: bool, _is_disk_swap: bool, kv_root: std::path::PathBuf) -> Result<Self> {
        let load_t0 = std::time::Instant::now();
        let path = path.strip_prefix(r"\\?\").unwrap_or(path);
        let tok_path = tokenizer_path.unwrap_or(path).strip_prefix(r"\\?\").unwrap_or(tokenizer_path.unwrap_or(path));
        let cfg_path = config_path.unwrap_or(path).strip_prefix(r"\\?\").unwrap_or(config_path.unwrap_or(path));
        let chat_template = ChatTemplate::init(tok_path)?;
        let tokenizer = TokenizerModel::init(tok_path)?;
        let raw_config: serde_json::Value = serde_json::from_slice(&std::fs::read(&std::path::Path::new(cfg_path).join("config.json"))?).map(logis_schema::json_order::sorted)?;
        let cfg: QwenVLConfig = if raw_config.get("text_config").is_some() { serde_json::from_value(raw_config)? } else {
            let text_config: crate::models::qwen::config::QwenVLTextConfig = serde_json::from_value(raw_config.clone())?;
            QwenVLConfig { architectures: raw_config.get("architectures").and_then(|v| serde_json::from_value(v.clone()).ok()), auto_map: raw_config.get("auto_map").and_then(|v| serde_json::from_value(v.clone()).ok()), hidden_size: raw_config.get("hidden_size").and_then(|v| v.as_u64()).map(|v| v as usize), image_token_id: raw_config.get("image_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), model_type: raw_config.get("model_type").and_then(|v| v.as_str()).unwrap_or("qwen2").to_string(), text_config: Some(text_config), tie_word_embeddings: raw_config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(true), torch_dtype: raw_config.get("torch_dtype").and_then(|v| v.as_str()).map(|s| s.to_string()), transformers_version: raw_config.get("transformers_version").and_then(|v| v.as_str()).unwrap_or("").to_string(), video_token_id: raw_config.get("video_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), vision_config: None, vision_start_token_id: None, vision_end_token_id: None }
        };
        let t_dev = get_device(text_device); let v_dev = get_device(vision_device); 
        let vram_before = crate::utils::device_utils::device_vram_snapshot(&t_dev);
        let parsed_dtype = get_dtype(dtype, cfg.text_config.as_ref().and_then(|tc| tc.dtype.as_deref()).unwrap_or("float16"));
        
        // [CRITICAL FIX] CPU 모드일 경우 데이터 타입을 설정 파일 무시하고 F32로 완전히 통일
        let effective_dtype = if t_dev.is_cpu() { DType::F32 } else { parsed_dtype };

        let gguf_f = find_type_files(path, "gguf")?; let mmproj_p = gguf_f.iter().find(|f| f.contains("mmproj")).cloned();
        let mut m_p = gguf_f.iter().find(|f| f.contains("Qwen3-0.6B-Q8_0.gguf")).cloned();
        if m_p.is_none() { m_p = gguf_f.iter().find(|f| f.contains("Qwen3-0.6B-Q4_K_M.gguf")).cloned(); }
        if m_p.is_none() { m_p = gguf_f.iter().find(|f| !f.contains("mmproj")).cloned(); }
        let recipe = if !gguf_f.is_empty() {
            if m_p.as_deref().is_some_and(|p| p.contains("Q8_0")) {
                "Q8_0".to_string()
            } else if m_p.as_deref().is_some_and(|p| p.contains("Q4_K")) {
                "Q4_K_M".to_string()
            } else {
                "GGUF".to_string()
            }
        } else {
            format!("{effective_dtype:?}")
        };
        let qwen = if !gguf_f.is_empty() {
            let kv_res = hard_token_limit.unwrap_or(4096) as u64 * 40000;
            if mmproj_p.is_some() && !force_text_only {
                let m_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(m_p.as_ref().unwrap())?)? };
                let mm_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&mmproj_p.unwrap())?)? };
                let ct_main = Arc::new(gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?);
                let ct_vision = Arc::new(gguf_file::Content::read(&mut std::io::Cursor::new(&mm_mmap[..]))?);
                
                ModelVariant::QuantizedVL(QuantizedQwenVLModel::new_with_mmap(&cfg, ct_main, Some(Arc::new(m_mmap)), ct_vision, Some(Arc::new(mm_mmap)), &t_dev, text_device_id, &v_dev, vision_device_id, effective_dtype, kv_res, baking_only)?)
            } else {
                let m_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(m_p.as_ref().unwrap())?)? };
                let ct_main = Arc::new(gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?);
                
                ModelVariant::QuantizedText(crate::models::qwen::quantized_model::QuantizedQwenTextModel::new_with_mmap(&cfg, ct_main, Some(Arc::new(m_mmap)), &t_dev, text_device_id, effective_dtype, kv_res, baking_only, baking_only)?)
            }
        } else { ModelVariant::Standard(QwenVLModel::new(cfg, unsafe { VarBuilder::from_mmaped_safetensors(&find_type_files(path, "safetensors")?, effective_dtype, &t_dev)? })?) };
        let g_p = std::path::Path::new(cfg_path).join("generation_config.json"); let g_cfg = if g_p.exists() { serde_json::from_slice(&std::fs::read(g_p)?)? } else { QwenVLGenerationConfig::default() };
        let (e1, e2) = match &g_cfg.eos_token_id { serde_json::Value::Number(n) => { let id = n.as_u64().unwrap_or(151645) as u32; (id, id) }, serde_json::Value::Array(arr) => { (arr.get(0).and_then(|v| v.as_u64()).unwrap_or(151643) as u32, arr.get(1).and_then(|v| v.as_u64()).unwrap_or(151643) as u32) }, _ => (151643, 151643) };
        let loaded_model_name = if m_p.as_ref().map(|p| p.contains("0.6B")).unwrap_or(false) { "0.6B".to_string() } else { "2B".to_string() };
        let json_protected_ids = build_json_protected_ids(&tokenizer);
        let json_starter_ids = build_json_starter_ids(&tokenizer);
        
        // pre_processor 에도 effective_dtype를 전달하여 생성 단계부터 F32 보장
        let gen = Self { chat_template, tokenizer, pre_processor: QwenVLProcessor::new(tok_path, &v_dev, effective_dtype)?, qwen, text_device: t_dev, vision_device: v_dev, eos_token_id1: e1, eos_token_id2: e2, generation_config: g_cfg, model_name: loaded_model_name, hard_token_limit, kv_root, json_protected_ids, json_starter_ids, recipe };
        let vram_after = crate::utils::device_utils::device_vram_snapshot(&gen.text_device);
        gen.record_load_reflection(
            load_t0.elapsed().as_secs_f64() as f32 * 1e3,
            vram_before,
            vram_after,
        );
        Ok(gen)
    }

    pub async fn prefill_only(&mut self, raw_text: String, _cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _relay_target: Option<&mut QwenVLGenerateModel>, _kv_name: Option<String>) -> Result<usize> {
        self.clear_kv_cache();
        if let ModelVariant::QuantizedVL(m) = &mut self.qwen { m.language_model.truncate_kv_cache(0)?; }
        if let ModelVariant::QuantizedText(m) = &mut self.qwen { m.language_model.truncate_kv_cache(0)?; }

        let (full_ids, spans) = self.encode_with_spans(&raw_text)?;
        let (payload_text, payload_ids) =
            self.collect_payload_text_and_ids(&raw_text, &raw_text, &full_ids, &spans);
        self.refresh_vocab_from_payload_ids(&payload_text, &payload_ids, false)?;
        let total_toks = full_ids.len();
        self.qwen.forward(&Tensor::from_vec(full_ids.clone(), (1, total_toks), &self.text_device)?, None, None, None, None, None, 0, total_toks, session_id.clone(), _kv_name.clone()).await?;
        if let Some(s_id) = &session_id {
            let path = crate::utils::paths::get_kv_dir().join(s_id);
            if !path.exists() { fs::create_dir_all(&path)?; }
            fs::write(path.join("tokens.json"), serde_json::to_string(&full_ids)?)?;
            let _ = self.force_flush_all_active_blocks(s_id, _kv_name.as_deref()).await;
        
            tokio::time::sleep(std::time::Duration::from_millis(50)).await; 
            
            println!("[PREFILL-WAIT] Waiting for SSD write to complete...");
            drain_global_io().await; 
            println!("[PREFILL-SAVE] Confirm: Block 0 to 42 are safely on SSD.");
        }

        // 🌟 [종료 직후 메모리 즉각 강제 반환]
        if self.text_device.is_cuda_or_rocm() { let _ = self.text_device.synchronize(); }
        crate::utils::legacy_working_set_trim(); // 🌟 [RAM-SAWTOOTH] 기본은 작업 집합을 내쫓지 않습니다 (LOGIS_MEM_TRIM=legacy)

        Ok(total_toks)
    }

    // 🌟 [추가] semantic_prejudice 파라미터를 받습니다.
    pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _kv_name: Option<String>, semantic_prejudice: Option<&str>) -> Result<String> {
        let mut is_reference_snapshot = false;
        if let Some(s_id) = &session_id {
            let snapshot_root = crate::utils::paths::get_kv_dir().join(s_id);
            
            let paths_to_try = vec![
                snapshot_root.join("inference").join("text"),
                snapshot_root.join("reference").join("text"),
                snapshot_root.clone(),
            ];

            for snapshot_path in paths_to_try {
                if snapshot_path.exists() && std::fs::read_dir(&snapshot_path).map(|mut d| d.next().is_some()).unwrap_or(false) {
                    println!("[GEN-LOAD] Loading existing snapshot from {:?}...", snapshot_path);
                    
                    if snapshot_path.to_string_lossy().contains("reference") {
                        is_reference_snapshot = true;
                    }

                    let _ = self.load_kv_from_disk(&snapshot_path, None); 
                    
                    if is_reference_snapshot {
                        println!("[GEN-LOAD] Reference snapshot detected. Resetting Registry Entry states for Full 28-Layer Prefill...");
                        
                        let reset_reg = |reg: &crate::models::qwen::quantized_model::KVRegistry| {
                            let mut entries = reg.entries.write().unwrap();
                            for (i, entry) in entries.iter_mut().enumerate() {
                                for loc in entry.location.iter_mut() { *loc = crate::models::qwen::quantized_model::KVLocation::RAM; }
                                for slot in entry.slot_ids.iter_mut() { *slot = None; }
                                entry.token_start = i * 1024; 
                                entry.token_len = 0;
                                entry.is_dirty.fill(true);
                                let mut cache = entry.bitkv_cache.write().unwrap();
                                cache.fill(None);
                            }
                        };

                        if let crate::models::qwen::generate::ModelVariant::QuantizedVL(m) = &mut self.qwen {
                            reset_reg(&m.language_model.registry);
                            let _ = m.language_model.truncate_kv_cache(0);
                        } else if let crate::models::qwen::generate::ModelVariant::QuantizedText(m) = &mut self.qwen {
                            reset_reg(&m.language_model.registry);
                            let _ = m.language_model.truncate_kv_cache(0);
                        }
                        self.clear_kv_cache();
                    }
                    break;
                }
            }
        }

        let temperature = mes.temperature.unwrap_or(0.0) as f32;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut lp = get_logit_processor(Some(temperature), Some(mes.top_p.unwrap_or(0.95) as f32), Some(40), seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let (f_ids, spans) = self.encode_with_spans(&input.replace_text)?;
        let user_text = crate::models::common::vocab_trim::prompt_text(&mes);
        let (payload_text, payload_ids) =
            self.collect_payload_text_and_ids(&input.replace_text, &user_text, &f_ids, &spans);
        self.refresh_vocab_from_payload_ids(&payload_text, &payload_ids, temperature > 1e-7)?;
        
        let total_toks = f_ids.len();
        println!("[DIAG-GEN] Encoded Prompt Length: {} tokens.", total_toks); 
        
        let kv_len = self.get_kv_len();
        
        let mut gen_text = String::new();
        let (input_ids, offset) = if kv_len > 0 && !is_reference_snapshot {
            if kv_len >= total_toks {
                println!("[SKIP-PREFILL] Snapshot covers entire prompt (Detected: {}, Needed: {}). Capping offset.", kv_len, total_toks);
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
        
        let prompt_seq_len = input_ids.dim(1)?;
        let total_tokens_after_prefill = offset + prompt_seq_len;

        // 🌟 [Contrastive Semantic Steering] 오답 레이블 진영 밀어내기 (Prejudice)
        let mut active_prej: Option<crate::models::common::prejudice::CachedPrejudice> = None;
        let mut semantic_prejudice_tensor: Option<Tensor> = None;
        if let Some(target_text) = semantic_prejudice {
            if self.qwen.trim_head().is_some() {
                match self.prejudice_cached_active(target_text) {
                    Ok(cp) => {
                        active_prej = Some(cp);
                        println!("[SEMANTIC-PREJUDICE] Cached Active-K Vector Prejudice for target: '{}'", target_text);
                    }
                    Err(e) => println!("[SEMANTIC-PREJUDICE] Failed to calculate active prejudice: {}", e),
                }
            } else if let Ok(target_ids) = self.tokenizer.text_encode_vec(target_text.to_string(), false) {
                if !target_ids.is_empty() {
                    let calc_prej = || -> Result<Tensor> {
                        let target_tensor = Tensor::from_vec(target_ids.clone(), (1, target_ids.len()), &self.text_device)?;
                        let target_emb = self.qwen.embedding_token_id(&target_tensor)?.to_dtype(DType::F32)?;
                        let target_emb_sum = target_emb.sum_keepdim(1)?;
                        let len_tensor = Tensor::new(target_ids.len() as f32, &self.text_device)?;
                        let target_emb_avg = target_emb_sum.broadcast_div(&len_tensor)?;
                        let target_vec = target_emb_avg.squeeze(0)?.squeeze(0)?;

                        // Phase 3-1 D3: chunked (16,384 rows) + 16-slot LRU PrejudiceCache (eliminates full-F32 OOM).
                        let all_embs = self.qwen.get_embed_tokens()?;
                        let vocab = all_embs.dim(0)?;
                        let key = format!("qwenvl:v{vocab}:{target_text}");
                        let cached = crate::models::common::prejudice::get_or_build(
                            &key,
                            &target_vec,
                            vocab,
                            &self.text_device,
                            |off, take| Ok(all_embs.narrow(0, off, take)?),
                        )?;
                        Ok(cached.values)
                    };
                    match calc_prej() {
                        Ok(prej) => {
                            semantic_prejudice_tensor = Some(prej);
                            println!("[SEMANTIC-PREJUDICE] Generated Vector Prejudice for target: '{}'", target_text);
                        }
                        Err(e) => println!("[SEMANTIC-PREJUDICE] Failed to calculate prejudice: {}", e),
                    }
                }
            }
        }

        // 🌟 [CRITICAL FIX] qwen3의 청크 루프 찌꺼기를 지우고 원래 0.6B의 정상적인 단일 Forward 연산으로 복구합니다!
        let prefill_t0 = std::time::Instant::now();
        let mut last_hidden: Option<Tensor> = None;
        let mut standard_logits: Option<Tensor> = None;
        crate::models::qwen::generate::wait_for_global_io().await; 
        if self.qwen.is_quantized() {
            last_hidden = Some(
                self.qwen
                    .forward_last_hidden(
                        &input_ids,
                        None,
                        None,
                        None,
                        None,
                        None,
                        offset,
                        total_tokens_after_prefill,
                        session_id.clone(),
                        _kv_name.clone(),
                    )
                    .await?,
            );
        } else {
            standard_logits = Some(
                self.qwen
                    .forward(
                        &input_ids,
                        None,
                        None,
                        None,
                        None,
                        None,
                        offset,
                        total_tokens_after_prefill,
                        session_id.clone(),
                        _kv_name.clone(),
                    )
                    .await?,
            );
        }
        let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;
        
        println!("[DEBUG-GEN] Prefill Complete. Sampling first token...");

        let mut gen_ids = vec![];
        let mut penalized_history: Vec<u32> = Vec::new();
        let mut scratch = crate::models::common::vocab_trim::DecodeStepScratch::default();
        let mut guard_stats = crate::models::common::vocab_trim::StepGuardStats::default();

        let think_token_id = self.tokenizer.text_encode_vec("<think>".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(123);
        let lt_id = self.tokenizer.text_encode_vec("<".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let enter_id = self.tokenizer.text_encode_vec("\n".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);

        let slash_id = self.tokenizer.text_encode_vec("/".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let double_slash_id = self.tokenizer.text_encode_vec("//".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);
        let space_double_slash_id = self.tokenizer.text_encode_vec(" //".to_string(), false).unwrap_or_default().into_iter().next().unwrap_or(999999);

        let is_strict_json = input.replace_text.contains("/no_think") || input.replace_text.contains("RETURN JSON ONLY");

        let decode_t0 = std::time::Instant::now();
        let mut decode_steps = 0usize;
        for i in 0..mes.max_tokens.unwrap_or(2048) {
            if let Some(flag) = &cancel_flag { if flag.load(std::sync::atomic::Ordering::Relaxed) { break; } }

            let next_token = if let Some(ref hidden) = last_hidden {
                self.select_token_from_hidden(
                    hidden,
                    i as usize,
                    is_strict_json,
                    temperature as f64,
                    1.2,
                    &mut lp,
                    &penalized_history,
                    &gen_text,
                    enter_id,
                    think_token_id,
                    open_bracket_id,
                    lt_id,
                    slash_id,
                    double_slash_id,
                    space_double_slash_id,
                    active_prej.as_ref(),
                    semantic_prejudice_tensor.as_ref(),
                    &mut scratch,
                    &mut guard_stats,
                )?
            } else {
                let logits = standard_logits
                    .as_ref()
                    .ok_or_else(|| anyhow!("missing logits during generation"))?;
                self.select_token_from_logits(
                    logits,
                    i as usize,
                    is_strict_json,
                    &mut lp,
                    &penalized_history,
                    &gen_text,
                    enter_id,
                    think_token_id,
                    open_bracket_id,
                    lt_id,
                    slash_id,
                    double_slash_id,
                    space_double_slash_id,
                    semantic_prejudice_tensor.as_ref(),
                    &mut guard_stats,
                )?
            };

            let is_eos = next_token == self.eos_token_id1 || next_token == self.eos_token_id2;
            gen_ids.push(next_token);
            if !(is_strict_json && self.json_protected_ids.contains(&next_token)) {
                penalized_history.push(next_token);
            }
            if let Ok(piece) = self.tokenizer.token_decode(vec![next_token]) {
                gen_text.push_str(&piece);
            }

            let mut is_json_finished = false;
            if gen_text.contains('{') {
                let mut depth = 0;
                let mut has_started = false;
                for c in gen_text.chars() {
                    if c == '{' { depth += 1; has_started = true; }
                    else if c == '}' { depth -= 1; }
                }
                if has_started && depth == 0 && gen_text.trim_end().ends_with('}') {
                    is_json_finished = true; 
                }
            } else if is_strict_json && gen_text.trim_start().starts_with('[') {
                let mut depth = 0;
                let mut has_started = false;
                for c in gen_text.chars() {
                    if c == '[' { depth += 1; has_started = true; }
                    else if c == ']' { depth -= 1; }
                }
                if has_started && depth == 0 && gen_text.trim_end().ends_with(']') {
                    is_json_finished = true;
                }
            }
            
            let current_pos = total_tokens_after_prefill + i as usize;
            
            if true {
                let pct = if is_json_finished || is_eos {
                    100
                } else {
                    15 + (84.0 * (1.0 - (-0.012 * (i as f32)).exp())) as i32
                };

                if let Some(tx) = crate::progress::PROGRESS_TX.get() {
                    if let Some(sid) = &session_id {
                        let task_id = if sid.starts_with("task_") || sid.starts_with("img_") || sid.starts_with("search_") {
                            let p: Vec<&str> = sid.split('_').collect();
                            if p.len() >= 2 { format!("{}_{}", p[0], p[1]) } else { sid.clone() }
                        } else { sid.clone() };
                        
                        let current_cat = crate::progress::CURRENT_UI_CATEGORY.read().unwrap().clone();

                        let summary_msg = if current_cat.contains("Classification") {
                            format!("Analyzing structure ({}%)...", pct)
                        } else if task_id.starts_with("search_") {
                            format!("Generating insights ({}%)...", pct)
                        } else {
                            format!("Extracting data ({}%)...", pct)
                        };

                        let _ = tx.send(serde_json::json!({
                            "task_id": task_id,
                            "category": format!("{} (Decoding)", current_cat), 
                            "summary": summary_msg,
                            "spinner": "⠧"
                        }));
                    }
                }
                
                if !is_strict_json {
                    print!("\r[DECODING] {} tokens generated (Context: {})    ", i + 1, current_pos + 1);
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
            }

            if is_json_finished || is_eos { break; }

            decode_steps += 1;
            let next_in = Tensor::from_vec(vec![next_token], (1, 1), &self.text_device)?;
            crate::models::qwen::generate::wait_for_global_io().await;
            if self.qwen.is_quantized() {
                last_hidden = Some(
                    self.qwen
                        .forward_last_hidden(
                            &next_in,
                            None,
                            None,
                            None,
                            None,
                            None,
                            current_pos,
                            current_pos + 1,
                            session_id.clone(),
                            _kv_name.clone(),
                        )
                        .await?,
                );
            } else {
                standard_logits = Some(
                    self.qwen
                        .forward(
                            &next_in,
                            None,
                            None,
                            None,
                            None,
                            None,
                            current_pos,
                            current_pos + 1,
                            session_id.clone(),
                            _kv_name.clone(),
                        )
                        .await?,
                );
            }

            if i > 0 && i % 30 == 0 {
                crate::utils::legacy_working_set_trim(); // 🌟 [RAM-SAWTOOTH] 기본은 작업 집합을 내쫓지 않습니다 (LOGIS_MEM_TRIM=legacy)
            }
        }
        let decode_ms = decode_t0.elapsed().as_secs_f64() * 1e3;

        if logis_schema::reflection::ReflectionLogMode::from_env()
            != logis_schema::reflection::ReflectionLogMode::Off
        {
            let head = self.qwen.trim_head();
            let vocab_v = self.qwen.embed_vocab().unwrap_or(0) as u32;
            let active_k = head.map(|h| h.kept.len() as u32).unwrap_or(vocab_v);
            let scope_label = head.map(|h| h.label.clone()).unwrap_or_else(|| "full".into());
            let head_path = if guard_stats.quant_mode_a {
                "quant_mode_a".into()
            } else if head.is_some() {
                "trimmed_head".into()
            } else {
                "linear_proj".into()
            };
            let ctx = crate::models::common::vocab_trim::current_lang_context();
            let effective_langs =
                ctx.langs
                    .union(crate::models::common::vocab_trim::Langs::present(
                        &payload_text,
                    ));
            let rec = logis_schema::reflection::ReflectionRecord {
                ts_ms: logis_schema::reflection::current_ts_ms(),
                run_id: logis_schema::reflection::current_run_id(),
                task_id: self.model_name.clone(),
                git_rev: option_env!("LOGIS_GIT_REV").unwrap_or("dev").into(),
                build_profile: logis_schema::reflection::build_profile_label(),
                features: Vec::new(),
                backend: backend_label_for(&self.text_device).into(),
                device_name: format!("{:?}", self.text_device),
                artifact: logis_schema::reflection::ArtifactReflection {
                    slot: self.slot_label().into(),
                    bytes: self.qwen.resident_vram_bytes(),
                    sha256_16: String::new(),
                    recipe: self.recipe.clone(),
                    vocab_v,
                    artifact_k: active_k,
                    plan_sha256: None,
                    space_tag: None,
                },
                load: None,
                exec: Some(logis_schema::reflection::ExecReflection {
                    step: "generate".into(),
                    langs: [ctx.doc_lang, ctx.ui_lang, effective_langs.label()],
                    scope_label,
                    vocab_k: [vocab_v, active_k],
                    head_path,
                    guard_kind: head
                        .map(|h| h.guard_kind_label().into())
                        .unwrap_or_else(|| "none".into()),
                    tokens: [prompt_seq_len as u32, decode_steps as u32],
                    timing_ms: [
                        prefill_ms as f32,
                        decode_ms as f32,
                        guard_stats.rescue_ms() as f32,
                    ],
                    steps: [
                        guard_stats.fast_steps as u32,
                        0,
                        guard_stats.constrained_steps as u32,
                        guard_stats.flagged as u32,
                        guard_stats.rest_wins as u32,
                    ],
                    sampling: temperature > 1e-7,
                    trim_mode: head
                        .map(|h| h.exec_mode.as_str().into())
                        .unwrap_or_else(|| "full_untrimmed".into()),
                    clusters_checked: guard_stats.clusters_checked as u32,
                    rows_scored: guard_stats.rows_scored as u64,
                    ..Default::default()
                }),
            };
            logis_schema::reflection::record(&rec);
        }
        
        if let Some(s_id) = &session_id {
            let _ = self.force_flush_all_active_blocks(s_id, _kv_name.as_deref()).await;
        }

        // 🌟 [종료 직후 메모리 즉각 강제 반환] 
        // 80% -> 65% 로 브라우저를 켤 때만 메모리가 OS로 반환되는 지연(Lazy Free) 현상을 타파합니다.
        // 생성이 종료된 직후에 GPU 동기화 및 OS 메모리 반환을 찔러 넣어 사용자가 즉각 65%를 볼 수 있게 합니다.
        if self.text_device.is_cuda_or_rocm() { let _ = self.text_device.synchronize(); }
        crate::utils::legacy_working_set_trim(); // 🌟 [RAM-SAWTOOTH] 기본은 작업 집합을 내쫓지 않습니다 (LOGIS_MEM_TRIM=legacy)

        Ok(gen_text)
    }

    pub fn get_kv_len(&self) -> usize { match &self.qwen { ModelVariant::QuantizedVL(m) => m.language_model.get_kv_len(), ModelVariant::QuantizedText(m) => m.language_model.get_kv_len(), _ => 0 } }
    pub async fn drop_kv_storage(&mut self) -> Result<()> { self.qwen.drop_kv_storage().await }
    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> { self.qwen.force_flush_all_active_blocks(session_id, kv_name).await }
    pub fn clear_kv_cache(&mut self) { match &mut self.qwen { ModelVariant::QuantizedVL(m) => m.language_model.clear_kv_cache(), ModelVariant::QuantizedText(m) => m.language_model.clear_kv_cache(), _ => {} } }
    pub fn clear_all_caches(&mut self) { self.clear_kv_cache(); }
    pub fn save_kv_to_disk(&mut self, path: &Path, kv_name: Option<&str>, offset: usize) -> Result<()> { match &mut self.qwen { ModelVariant::QuantizedVL(m) => m.language_model.save_kv_cache(path, false, offset, kv_name), ModelVariant::QuantizedText(m) => m.language_model.save_kv_cache(path, false, offset, kv_name), _ => Ok(()) } }
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> { match &mut self.qwen { ModelVariant::QuantizedVL(m) => m.language_model.truncate_kv_cache(len), ModelVariant::QuantizedText(m) => m.language_model.truncate_kv_cache(len), _ => Ok(()) } }
    pub fn load_kv_from_disk(&mut self, path: &Path, kv_name: Option<&str>) -> Result<()> { 
        match &mut self.qwen { 
            ModelVariant::QuantizedVL(m) => m.language_model.load_kv_cache(path, &self.text_device, 0, 128, kv_name), 
            ModelVariant::QuantizedText(m) => m.language_model.load_kv_cache(path, &self.text_device, 0, 128, kv_name), 
            _ => Ok(()) 
        } 
    }
    pub async fn prefill_chunk(&mut self, text: String, _cancel_flag: Option<Arc<AtomicBool>>, _relay_target: Option<&mut QwenVLGenerateModel>) -> Result<usize> {
        let (chunk_ids_vec, spans) = self.encode_with_spans(&text)?;
        let (payload_text, payload_ids) =
            self.collect_payload_text_and_ids(&text, &text, &chunk_ids_vec, &spans);
        self.refresh_vocab_from_payload_ids(&payload_text, &payload_ids, false)?;
        let chunk_size = chunk_ids_vec.len();
        let current_pos = self.get_kv_len();
        let chunk_ids = Tensor::from_vec(chunk_ids_vec, (1, chunk_size), &self.text_device)?;
        self.qwen.forward(&chunk_ids, None, None, None, None, None, current_pos, current_pos + chunk_size, None, None).await?;
        Ok(chunk_size)
    }
}
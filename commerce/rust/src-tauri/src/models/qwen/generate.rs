use crate::models::qwen::quantized_model::{KVLocation, KVBlock, KVRegistry, BitKVMetadata, QuantizedQwenVLModel, MemorySlot};
use anyhow::{Result, anyhow};
use candle_core::{quantized::gguf_file, DType, Device, Tensor};
use candle_nn::VarBuilder;
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
pub async fn wait_for_global_io() { while GLOBAL_IO_COUNTER.load(Ordering::SeqCst) > 0 { tokio::time::sleep(std::time::Duration::from_millis(10)).await; } }

pub fn init_bake_worker() {
    let (btx, brx) = mpsc::channel(64); let (ltx, lrx) = mpsc::channel(64);
    let _ = BAKE_TX.set(btx); let _ = LOAD_TX.set(ltx);
    if let Some(rx) = SLOT_MANAGER_DATA.1.lock().unwrap().take() { tauri::async_runtime::spawn(async move { spawn_slot_dispatcher(rx).await; }); }
    tauri::async_runtime::spawn(async move { spawn_slot_worker(brx); }); 
    tauri::async_runtime::spawn(async move { spawn_slot_worker(lrx); });
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
                    
                    let plain_data = std::fs::read(&tmp_path)?;
                    let encrypted_data = crate::utils::crypto::encrypt_data(&plain_data)?;
                    
                    std::fs::write(&tp_clone, encrypted_data)?;
                    let _ = std::fs::remove_file(tmp_path);
                    
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

                                // /qwen/ 은 F32 로직, /qwen3/, /qwen3_5/ 은 FP8 로직으로 동적 분기
                                let target_dtype = if k_contig.dtype() == DType::F8E4M3 { DType::F8E4M3 } else { DType::F32 };
                                src.k_data = k_contig.to_dtype(target_dtype).unwrap_or_else(|_| k_contig.clone());
                                src.v_data = v_contig.to_dtype(target_dtype).unwrap_or_else(|_| v_contig.clone());
                            }
                            let mut map = std::collections::HashMap::new();
                            let prefix = format!("b{}_l{}_", off, act_l);
                            map.insert(format!("{}k_data", prefix), src.k_data.clone());
                            map.insert(format!("{}v_data", prefix), src.v_data.clone());
                            map.insert(format!("{}k_shape", prefix), src.k_shape.clone());
                            
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
                            if let Ok(encrypted_content) = load_kv_block(&file_path) {
                                if let Ok(content) = crate::utils::crypto::decrypt_data(&encrypted_content) {
                                    if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                                        let prefix = format!("b{}_l{}_", b_idx_off, target_layer);
                                        let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).ok();
                                    
                                        if let (Some(kd), Some(vd), Some(sh)) = (get_t("k_data"), get_t("v_data"), get_t("k_shape")) {
                                            let sh_u32: Vec<u32> = sh.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                                            let file_shape: Vec<usize> = sh_u32.iter().map(|&x| x as usize).collect();

                                            let dev = &Device::Cpu;
                                            // 저장된 파일의 DType을 감지하여 FP8, F32 모두 호환되도록 로드합니다.
                                            let saved_dtype = match kd.dtype() {
                                                safetensors::Dtype::F8_E4M3 => DType::F8E4M3,
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
                                            let meta = BitKVMetadata { k_data: kd_t, v_data: vd_t, original_shape: file_shape };
                                            
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
                        }
                    });
                },
                SlotTask::IndexUpdate { .. } => {}
            }
        }
    });
}

struct ReadSlotGuard { sid: usize, active: bool }
impl Drop for ReadSlotGuard { fn drop(&mut self) { if self.active { let sid = self.sid; tauri::async_runtime::spawn(async move { SLOT_MANAGER.release_slot(sid).await; }); } } }

#[derive(Clone)]
pub enum ModelVariant { Standard(QwenVLModel), QuantizedVL(QuantizedQwenVLModel), QuantizedText(crate::models::qwen::quantized_model::QuantizedQwenTextModel) }

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
}

impl QwenVLGenerateModel {
    /// config.json에서 로드된 모델의 최대 토큰 한계치를 반환합니다.
    pub fn get_max_tokens(&self) -> usize {
        match &self.qwen {
            ModelVariant::QuantizedVL(m) => m.language_model.config.max_position_embeddings,
            ModelVariant::QuantizedText(m) => m.language_model.config.max_position_embeddings,
            ModelVariant::Standard(m) => m.config.text_config.as_ref().map(|tc| tc.max_position_embeddings).unwrap_or(32768),
        }
    }

    pub fn init_with_config(path: &str, tokenizer_path: Option<&str>, config_path: Option<&str>, text_device: Option<&Device>, text_device_id: usize, vision_device: Option<&Device>, vision_device_id: usize, dtype: Option<DType>, hard_token_limit: Option<usize>, force_text_only: bool, baking_only: bool, _is_disk_swap: bool, kv_root: std::path::PathBuf) -> Result<Self> {
        let path = path.strip_prefix(r"\\?\").unwrap_or(path);
        let tok_path = tokenizer_path.unwrap_or(path).strip_prefix(r"\\?\").unwrap_or(tokenizer_path.unwrap_or(path));
        let cfg_path = config_path.unwrap_or(path).strip_prefix(r"\\?\").unwrap_or(config_path.unwrap_or(path));
        let chat_template = ChatTemplate::init(tok_path)?;
        let tokenizer = TokenizerModel::init(tok_path)?;
        let raw_config: serde_json::Value = serde_json::from_slice(&std::fs::read(&std::path::Path::new(cfg_path).join("config.json"))?)?;
        let cfg: QwenVLConfig = if raw_config.get("text_config").is_some() { serde_json::from_value(raw_config)? } else {
            let text_config: crate::models::qwen::config::QwenVLTextConfig = serde_json::from_value(raw_config.clone())?;
            QwenVLConfig { architectures: raw_config.get("architectures").and_then(|v| serde_json::from_value(v.clone()).ok()), auto_map: raw_config.get("auto_map").and_then(|v| serde_json::from_value(v.clone()).ok()), hidden_size: raw_config.get("hidden_size").and_then(|v| v.as_u64()).map(|v| v as usize), image_token_id: raw_config.get("image_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), model_type: raw_config.get("model_type").and_then(|v| v.as_str()).unwrap_or("qwen2").to_string(), text_config: Some(text_config), tie_word_embeddings: raw_config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(true), torch_dtype: raw_config.get("torch_dtype").and_then(|v| v.as_str()).map(|s| s.to_string()), transformers_version: raw_config.get("transformers_version").and_then(|v| v.as_str()).unwrap_or("").to_string(), video_token_id: raw_config.get("video_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), vision_config: None, vision_start_token_id: None, vision_end_token_id: None }
        };
        let t_dev = get_device(text_device); let v_dev = get_device(vision_device); 
        let parsed_dtype = get_dtype(dtype, cfg.text_config.as_ref().and_then(|tc| tc.dtype.as_deref()).unwrap_or("float16"));
        
        // [CRITICAL FIX] CPU 모드일 경우 데이터 타입을 설정 파일 무시하고 F32로 완전히 통일
        let effective_dtype = if t_dev.is_cpu() { DType::F32 } else { parsed_dtype };

        let gguf_f = find_type_files(path, "gguf")?; let mmproj_p = gguf_f.iter().find(|f| f.contains("mmproj")).cloned();
        let mut m_p = gguf_f.iter().find(|f| f.contains("Qwen3-0.6B-Q8_0.gguf")).cloned();
        if m_p.is_none() { m_p = gguf_f.iter().find(|f| f.contains("Qwen3-0.6B-Q4_K_M.gguf")).cloned(); }
        if m_p.is_none() { m_p = gguf_f.iter().find(|f| !f.contains("mmproj")).cloned(); }
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
        
        // pre_processor 에도 effective_dtype를 전달하여 생성 단계부터 F32 보장
        Ok(Self { chat_template, tokenizer, pre_processor: QwenVLProcessor::new(tok_path, &v_dev, effective_dtype)?, qwen, text_device: t_dev, vision_device: v_dev, eos_token_id1: e1, eos_token_id2: e2, generation_config: g_cfg, model_name: loaded_model_name, hard_token_limit, kv_root })
    }

    pub async fn prefill_only(&mut self, raw_text: String, _cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _relay_target: Option<&mut QwenVLGenerateModel>, _kv_name: Option<String>) -> Result<usize> {
        self.clear_kv_cache();
        if let ModelVariant::QuantizedVL(m) = &mut self.qwen { m.language_model.truncate_kv_cache(0)?; }
        if let ModelVariant::QuantizedText(m) = &mut self.qwen { m.language_model.truncate_kv_cache(0)?; }

        let full_ids = self.tokenizer.text_encode_vec(raw_text, false)?;
        let total_toks = full_ids.len();
        self.qwen.forward(&Tensor::from_vec(full_ids.clone(), (1, total_toks), &self.text_device)?, None, None, None, None, None, 0, total_toks, session_id.clone(), _kv_name.clone()).await?;
        if let Some(s_id) = &session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(s_id);
            if !path.exists() { fs::create_dir_all(&path)?; }
            fs::write(path.join("tokens.json"), serde_json::to_string(&full_ids)?)?;
            let _ = self.force_flush_all_active_blocks(s_id, _kv_name.as_deref()).await;
        
            tokio::time::sleep(std::time::Duration::from_millis(50)).await; 
            
            println!("[PREFILL-WAIT] Waiting for SSD write to complete...");
            wait_for_global_io().await; 
            println!("[PREFILL-SAVE] Confirm: Block 0 to 42 are safely on SSD.");
        }
        Ok(total_toks)
    }

    pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _kv_name: Option<String>) -> Result<String> {
        let mut is_reference_snapshot = false;
        if let Some(s_id) = &session_id {
            let snapshot_root = crate::utils::paths::get_kv_dir(None).join(s_id);
            
            let paths_to_try = vec![
                snapshot_root.join("inference").join("text"),
                snapshot_root.join("reference").join("text"),
                snapshot_root.clone(),
            ];

            for snapshot_path in paths_to_try {
                if snapshot_path.exists() && fs::read_dir(&snapshot_path).map(|mut d| d.next().is_some()).unwrap_or(false) {
                    println!("[GEN-LOAD] Loading existing snapshot from {:?}...", snapshot_path);
                    
                    if snapshot_path.to_string_lossy().contains("reference") {
                        is_reference_snapshot = true;
                    }

                    let _ = self.load_kv_from_disk(&snapshot_path, None); 
                    
                    if is_reference_snapshot {
                        println!("[GEN-LOAD] Reference snapshot detected. Resetting Registry Entry states for Full 28-Layer Prefill...");
                        
                        let reset_reg = |reg: &KVRegistry| {
                            let mut entries = reg.entries.write().unwrap();
                            for (i, entry) in entries.iter_mut().enumerate() {
                                for loc in entry.location.iter_mut() { *loc = KVLocation::RAM; }
                                for slot in entry.slot_ids.iter_mut() { *slot = None; }
                                entry.token_start = i * 1024; 
                                entry.token_len = 0;
                                entry.is_dirty.fill(true);
                                let mut cache = entry.bitkv_cache.write().unwrap();
                                cache.fill(None);
                            }
                        };

                        if let ModelVariant::QuantizedVL(m) = &mut self.qwen {
                            reset_reg(&m.language_model.registry);
                            let _ = m.language_model.truncate_kv_cache(0);
                        } else if let ModelVariant::QuantizedText(m) = &mut self.qwen {
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
        let f_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        
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
        
        let total_tokens_after_prefill = offset + input_ids.dim(1)?;
    
        wait_for_global_io().await; 
        let mut logits = self.qwen.forward(&input_ids, None, None, None, None, None, offset, total_tokens_after_prefill, session_id.clone(), _kv_name.clone()).await?;
        
        println!("[DEBUG-GEN] Prefill Complete. Sampling first token...");

        let mut gen_ids = vec![];

        let think_token_id = self.tokenizer.text_encode_vec("<think>".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let open_bracket_id = self.tokenizer.text_encode_vec("{".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(123);
        let lt_id = self.tokenizer.text_encode_vec("<".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let enter_id = self.tokenizer.text_encode_vec("\n".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);

        let slash_id = self.tokenizer.text_encode_vec("/".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let double_slash_id = self.tokenizer.text_encode_vec("//".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);
        let space_double_slash_id = self.tokenizer.text_encode_vec(" //".to_string(), false).ok().and_then(|v: Vec<u32>| v.first().cloned()).unwrap_or(999999);

        let is_strict_json = input.replace_text.contains("/no_think") || input.replace_text.contains("RETURN JSON ONLY");

        for i in 0..mes.max_tokens.unwrap_or(2048) {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { break; } }
            
            let mut logits_vec = logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let len = logits_vec.len();

            if !gen_ids.is_empty() {
                let penalty = 1.2;
                let mut set = std::collections::HashSet::new();
                for &t in &gen_ids {
                    if !set.contains(&t) && (t as usize) < len {
                        let logit = logits_vec[t as usize];
                        logits_vec[t as usize] = if logit < 0.0 { logit * penalty } else { logit / penalty };
                        set.insert(t);
                    }
                }
            }

            if (think_token_id as usize) < len { logits_vec[think_token_id as usize] -= 1000.0; }
            if (lt_id as usize) < len { logits_vec[lt_id as usize] -= 10.0; }

            if is_strict_json {
                let is_url_single = gen_text.ends_with("http:/") || gen_text.ends_with("https:/");
                let is_url_double = gen_text.ends_with("http:") || gen_text.ends_with("https:");
                
                if !is_url_single && gen_text.ends_with('/') {
                    if (slash_id as usize) < len { logits_vec[slash_id as usize] -= 10000.0; }
                }
                if !is_url_double {
                    if (double_slash_id as usize) < len { logits_vec[double_slash_id as usize] -= 10000.0; }
                    if (space_double_slash_id as usize) < len { logits_vec[space_double_slash_id as usize] -= 10000.0; }
                }
            }
            
            if i == 0 {
                if (self.eos_token_id1 as usize) < len { logits_vec[self.eos_token_id1 as usize] = -10000.0; }
                if (self.eos_token_id2 as usize) < len { logits_vec[self.eos_token_id2 as usize] = -10000.0; }
                if (enter_id as usize) < len { logits_vec[enter_id as usize] -= 50.0; }
                
                if (open_bracket_id as usize) < len {
                    let boost = if is_strict_json { 10000.0 } else { 20.0 };
                    logits_vec[open_bracket_id as usize] += boost;
                }
            }

            let logits_tensor = Tensor::from_vec(logits_vec, (len,), &Device::Cpu)?;
            let mut next_id = lp.sample(&logits_tensor)?;
            
            if i == 0 {
                if is_strict_json {
                    next_id = open_bracket_id;
                } else if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 {
                    println!("[DEBUG-GEN] EOS detected on first token. Overriding with '{{' to force JSON.");
                    next_id = open_bracket_id;
                }
            }

            let is_eos = next_id == self.eos_token_id1 || next_id == self.eos_token_id2;

            gen_ids.push(next_id);
            if let Ok(piece) = self.tokenizer.token_decode(vec![next_id]) {
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
                    println!("[DEBUG-GEN] Balanced JSON detected (Depth 0). Stopping at token {}.", i + 1);
                    is_json_finished = true; 
                }
            }
            
            let current_pos = total_tokens_after_prefill + i as usize;
            
            
            if true {
                let pct = if is_json_finished || is_eos {
                    100
                } else {
                    
                    // 기존 가속도(-0.05)가 너무 빨라 20토큰 만에 69%에 도달한 뒤 100%로 튀는 현상을 방지합니다.
                    // 일반적인 JSON 응답 길이(50~200토큰)에 맞춰 -0.012로 조정하여 부드럽고 정확하게 증가하도록 개선했습니다.
                    // (예: 10토큰=25%, 50토큰=53%, 100토큰=73%, 200토큰=91%)
                    15 + (84.0 * (1.0 - (-0.012 * (i as f32)).exp())) as i32
                };

                if let Some(tx) = crate::scheduler::PROGRESS_TX.get() {
                    if let Some(sid) = &session_id {
                        let task_id = if sid.starts_with("task_") || sid.starts_with("img_") || sid.starts_with("search_") {
                            let p: Vec<&str> = sid.split('_').collect();
                            if p.len() >= 2 { format!("{}_{}", p[0], p[1]) } else { sid.clone() }
                        } else { sid.clone() };
                        
                        let current_cat = crate::CURRENT_UI_CATEGORY.read().unwrap().clone();

                        
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
                    let _ = std::io::stdout().flush();
                }
            }

            if is_json_finished || is_eos { 
                break; 
            }

            wait_for_global_io().await;
            logits = self.qwen.forward(&Tensor::from_vec(vec![next_id], (1, 1), &self.text_device)?, None, None, None, None, None, current_pos, current_pos + 1, session_id.clone(), _kv_name.clone()).await?;

            if i > 0 && i % 30 == 0 {
                #[cfg(target_os = "windows")]
                unsafe {
                    use windows_sys::Win32::System::Threading::GetCurrentProcess;
                    use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
                    let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
                }
                #[cfg(target_os = "linux")]
                unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
                #[cfg(target_os = "macos")]
                unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }
            }
        }

        if let Some(s_id) = &session_id {
            let _ = self.force_flush_all_active_blocks(s_id, _kv_name.as_deref()).await;
        }
        Ok(gen_text)
    }

    pub fn get_kv_len(&self) -> usize { match &self.qwen { ModelVariant::QuantizedVL(m) => m.language_model.get_kv_len(), ModelVariant::QuantizedText(m) => m.language_model.get_kv_len(), _ => 0 } }
    pub async fn drop_kv_storage(&mut self) -> Result<()> { self.qwen.drop_kv_storage().await }
    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> { self.qwen.force_flush_all_active_blocks(session_id, kv_name).await }
    pub fn clear_kv_cache(&mut self) { match &mut self.qwen { ModelVariant::QuantizedVL(m) => m.language_model.clear_kv_cache(), ModelVariant::QuantizedText(m) => m.language_model.clear_kv_cache(), _ => {} } }
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
        let chunk_ids_vec = self.tokenizer.text_encode_vec(text, false)?;
        let chunk_size = chunk_ids_vec.len();
        let current_pos = self.get_kv_len();
        let chunk_ids = Tensor::from_vec(chunk_ids_vec, (1, chunk_size), &self.text_device)?;
        self.qwen.forward(&chunk_ids, None, None, None, None, None, current_pos, current_pos + chunk_size, None, None).await?;
        Ok(chunk_size)
    }
}
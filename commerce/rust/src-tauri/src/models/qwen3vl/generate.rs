use crate::models::qwen3vl::quantized_model::KVLocation;
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
use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
use std::fs;
use std::path::Path;

use std::path::PathBuf;
use tokio::sync::mpsc;
use std::collections::VecDeque;
use tokio::sync::oneshot;
use rayon::prelude::*;

pub enum SlotRequest {
    AcquireRead { response: oneshot::Sender<usize> },
    AcquireWrite { response: oneshot::Sender<usize>, total_tokens: usize },
    Release { idx: usize, state_at_release: u8 }, 
    MarkReady { idx: usize, bytes: usize },
    Flush { response: oneshot::Sender<()> },
}

// [GLOBAL] 슬롯 관리자: 중앙 디스패처 기반 관리
pub struct SlotManager {
    pub slots: Vec<crate::models::qwen3vl::quantized_model::MemorySlot>,
    pub request_tx: mpsc::Sender<SlotRequest>,
    pub count_reads: Arc<AtomicUsize>,
    pub count_writes: Arc<AtomicUsize>,
    pub count_cached: Arc<AtomicUsize>,
    pub count_free: Arc<AtomicUsize>,
}

impl SlotManager {
    pub fn new(count: usize) -> Self {
        let mut slots = Vec::new();
        let num_layers = 28;
        for i in 0..count {
            slots.push(crate::models::qwen3vl::quantized_model::MemorySlot::new(i, num_layers));
        }

        let (tx, rx) = mpsc::channel(1000); 
        let count_reads = Arc::new(AtomicUsize::new(0));
        let count_writes = Arc::new(AtomicUsize::new(0));
        let count_cached = Arc::new(AtomicUsize::new(0));
        let count_free = Arc::new(AtomicUsize::new(count));

        let count_r = count_reads.clone();
        let count_w = count_writes.clone();
        let count_c = count_cached.clone();
        let count_f = count_free.clone();
        
        let mut free_pool: VecDeque<usize> = (0..count).collect();
        let ready_pool: VecDeque<usize> = VecDeque::new();
        let max_pool_size = count;
        
        tauri::async_runtime::spawn(async move {
            Self::slot_dispatcher(rx, free_pool, ready_pool, count_r, count_w, count_c, count_f, max_pool_size).await;
        });

        Self {
            slots,
            request_tx: tx,
            count_reads,
            count_writes,
            count_cached,
            count_free,
        }
    }

    async fn slot_dispatcher(
        mut rx: mpsc::Receiver<SlotRequest>,
        mut free_pool: VecDeque<usize>,
        mut ready_pool: VecDeque<usize>,
        count_r: Arc<AtomicUsize>,
        count_w: Arc<AtomicUsize>,
        count_c: Arc<AtomicUsize>,
        count_f: Arc<AtomicUsize>,
        max_pool_size: usize,
    ) {
        let mut pending_writes: VecDeque<(oneshot::Sender<usize>, usize)> = VecDeque::new();
        let mut pending_reads: VecDeque<oneshot::Sender<usize>> = VecDeque::new();
        let mut flush_waiters: Vec<oneshot::Sender<()>> = Vec::new();
        let mut active_write_count: usize = 0;
        let mut active_read_count: usize = 0;
        let mut current_ram_usage: usize = 0;
        let ram_threshold: usize = 100 * 1024 * 1024;
        let mut sys = sysinfo::System::new_all();

        println!("[SLOT-DISPATCHER] Robust Service Running.");

        while let Some(request) = rx.recv().await {
            match request {
                SlotRequest::Flush { response } => {
                    if active_write_count == 0 && active_read_count == 0 { let _ = response.send(()); } 
                    else { flush_waiters.push(response); }
                }
                SlotRequest::AcquireWrite { response, total_tokens } => {
                    sys.refresh_memory();
                    let max_concurrent = Self::calculate_dynamic_budget(&sys, total_tokens, max_pool_size);
                    if active_write_count < max_concurrent {
                        if let Some(idx) = free_pool.pop_front() {
                            let _ = response.send(idx); active_write_count += 1;
                            count_f.fetch_sub(1, Ordering::SeqCst); count_w.fetch_add(1, Ordering::SeqCst);
                            continue;
                        } else if let Some(idx) = ready_pool.pop_front() {
                            let _ = response.send(idx); active_write_count += 1;
                            count_c.fetch_sub(1, Ordering::SeqCst); count_w.fetch_add(1, Ordering::SeqCst);
                            continue;
                        }
                    }
                    pending_writes.push_back((response, total_tokens));
                }
                SlotRequest::AcquireRead { response } => {
                    if let Some(idx) = free_pool.pop_front() {
                        let _ = response.send(idx); active_read_count += 1;
                        count_f.fetch_sub(1, Ordering::SeqCst); count_r.fetch_add(1, Ordering::SeqCst);
                    } else if let Some(idx) = ready_pool.pop_front() {
                        let _ = response.send(idx); active_read_count += 1;
                        count_c.fetch_sub(1, Ordering::SeqCst); count_r.fetch_add(1, Ordering::SeqCst);
                    } else { pending_reads.push_back(response); }
                }
                SlotRequest::Release { idx, state_at_release } => {
                    free_pool.push_back(idx);
                    count_f.fetch_add(1, Ordering::SeqCst);
                    if state_at_release == 1 { active_write_count = active_write_count.saturating_sub(1); count_w.fetch_sub(1, Ordering::SeqCst); } 
                    else if state_at_release == 3 || state_at_release == 2 { active_read_count = active_read_count.saturating_sub(1); count_r.fetch_sub(1, Ordering::SeqCst); }
                    
                    if active_write_count == 0 && active_read_count == 0 { while let Some(w) = flush_waiters.pop() { let _ = w.send(()); } }
                    sys.refresh_memory();
                    Self::process_queues(&sys, &mut free_pool, &mut ready_pool, &mut pending_writes, &mut pending_reads, &mut active_write_count, &mut active_read_count, &count_f, &count_c, &count_w, &count_r, max_pool_size);
                }
                SlotRequest::MarkReady { idx, bytes } => {
                    ready_pool.push_back(idx);
                    current_ram_usage += bytes;
                    count_c.fetch_add(1, Ordering::SeqCst);
                    active_write_count = active_write_count.saturating_sub(1);
                    count_w.fetch_sub(1, Ordering::SeqCst);

                    if active_write_count == 0 && active_read_count == 0 { while let Some(w) = flush_waiters.pop() { let _ = w.send(()); } }

                    if current_ram_usage > ram_threshold {
                        println!("[SWAP-OUT] RAM Usage ({:.2} MB) > 100MB. Evicting oldest slots...", current_ram_usage as f64 / 1e6);
                        while current_ram_usage > (ram_threshold / 2) && !ready_pool.is_empty() {
                            if let Some(evict_idx) = ready_pool.pop_front() {
                                let slot = &crate::models::qwen3vl::generate::SLOT_MANAGER.slots[evict_idx];
                                for l_k in &slot.k_layers { if let Ok(mut g) = l_k.try_lock() { *g = None; } }
                                for l_v in &slot.v_layers { if let Ok(mut g) = l_v.try_lock() { *g = None; } }
                                slot.state.store(0, Ordering::SeqCst);
                                current_ram_usage = current_ram_usage.saturating_sub(20 * 1024 * 1024);
                                count_c.fetch_sub(1, Ordering::SeqCst); free_pool.push_back(evict_idx); count_f.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                    }
                    sys.refresh_memory();
                    Self::process_queues(&sys, &mut free_pool, &mut ready_pool, &mut pending_writes, &mut pending_reads, &mut active_write_count, &mut active_read_count, &count_f, &count_c, &count_w, &count_r, max_pool_size);
                }
            }
        }
    }

    fn calculate_dynamic_budget(sys: &sysinfo::System, total_tokens: usize, max_pool_size: usize) -> usize {
        let available_ram_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        let usable_ram = (available_ram_gb - 1.5).max(0.0);
        let ram_budget = (usable_ram / 0.02) as usize;
        let token_budget = (total_tokens / 1024).saturating_add(2);
        ram_budget.min(token_budget).min(max_pool_size).max(4).min(max_pool_size)
    }

    fn process_queues(
        sys: &sysinfo::System,
        free_pool: &mut VecDeque<usize>,
        ready_pool: &mut VecDeque<usize>,
        pending_writes: &mut VecDeque<(oneshot::Sender<usize>, usize)>,
        pending_reads: &mut VecDeque<oneshot::Sender<usize>>,
        active_write_count: &mut usize,
        active_read_count: &mut usize,
        count_f: &Arc<AtomicUsize>,
        count_c: &Arc<AtomicUsize>,
        count_w: &Arc<AtomicUsize>,
        count_r: &Arc<AtomicUsize>,
        max_pool_size: usize,
    ) {
        while !pending_writes.is_empty() {
            let (_, total_tokens) = pending_writes.front().unwrap();
            let max_concurrent = Self::calculate_dynamic_budget(sys, *total_tokens, max_pool_size);
            if *active_write_count < max_concurrent {
                if let Some(idx) = free_pool.pop_front() {
                    let (res, _) = pending_writes.pop_front().unwrap(); let _ = res.send(idx);
                    *active_write_count += 1; count_f.fetch_sub(1, Ordering::SeqCst); count_w.fetch_add(1, Ordering::SeqCst);
                    continue;
                } else if let Some(idx) = ready_pool.pop_front() {
                    let (res, _) = pending_writes.pop_front().unwrap(); let _ = res.send(idx);
                    *active_write_count += 1; count_c.fetch_sub(1, Ordering::SeqCst); count_w.fetch_add(1, Ordering::SeqCst);
                    continue;
                }
            }
            break;
        }
        while !pending_reads.is_empty() {
            if let Some(idx) = free_pool.pop_front() {
                let res = pending_reads.pop_front().unwrap(); let _ = res.send(idx);
                *active_read_count += 1; count_f.fetch_sub(1, Ordering::SeqCst); count_r.fetch_add(1, Ordering::SeqCst);
            } else if let Some(idx) = ready_pool.pop_front() {
                let res = pending_reads.pop_front().unwrap(); let _ = res.send(idx);
                *active_read_count += 1; count_c.fetch_sub(1, Ordering::SeqCst); count_r.fetch_add(1, Ordering::SeqCst);
            } else { break; }
        }
    }

    pub async fn reset_all_slots(&self) {
        for i in 0..self.slots.len() { self.release_slot(i).await; }
    }

    pub async fn wait_for_all_tasks(&self) {
        let (tx, rx) = oneshot::channel();
        let _ = self.request_tx.send(SlotRequest::Flush { response: tx }).await;
        let _ = rx.await;
        println!("[SLOT-MANAGER] All background tasks flushed.");
    }

    pub async fn acquire_read_slot(&self) -> usize {
        let (tx, rx) = oneshot::channel();
        let _ = self.request_tx.send(SlotRequest::AcquireRead { response: tx }).await;
        rx.await.unwrap_or(0)
    }

    pub async fn acquire_write_slot(&self, total_tokens: usize) -> usize {
        let (tx, rx) = oneshot::channel();
        let _ = self.request_tx.send(SlotRequest::AcquireWrite { response: tx, total_tokens }).await;
        rx.await.unwrap_or(0)
    }

    pub async fn release_slot(&self, id: usize) {
        if id < self.slots.len() {
            let state_at_release = self.slots[id].state.load(Ordering::SeqCst);
            for layer_k in &self.slots[id].k_layers { *layer_k.lock().await = None; }
            for layer_v in &self.slots[id].v_layers { *layer_v.lock().await = None; }
            self.slots[id].state.store(0, Ordering::SeqCst);
            let _ = self.request_tx.send(SlotRequest::Release { idx: id, state_at_release }).await;
        }
    }

    pub async fn mark_ready(&self, id: usize, bytes: usize) {
        if id < self.slots.len() {
            self.slots[id].state.store(2, Ordering::SeqCst);
            let _ = self.request_tx.send(SlotRequest::MarkReady { idx: id, bytes }).await;
        }
    }

    pub fn get_counts(&self) -> (usize, usize, usize, usize) {
        (self.count_reads.load(Ordering::Relaxed), self.count_writes.load(Ordering::Relaxed), self.count_cached.load(Ordering::Relaxed), self.count_free.load(Ordering::Relaxed))
    }
}

pub static ACTIVE_BAKE_TASKS: AtomicUsize = AtomicUsize::new(0);
pub static SLOT_MANAGER: once_cell::sync::Lazy<SlotManager> = once_cell::sync::Lazy::new(|| SlotManager::new(65));

struct LayerKVDump {
    layer_idx: usize,
    k_data: Vec<f32>,
    v_data: Vec<f32>,
    shape: Vec<usize>,
}

struct BakeTask {
    slot_id: usize,
    task_dir: PathBuf,
    kv_name: Option<String>,
    offset: usize,
    layers: Vec<LayerKVDump>,
}

struct SaveTask {
    slot_id: usize,
    path: PathBuf,
    tensors: std::collections::HashMap<String, Tensor>,
    is_last: bool, 
    total_bytes_hint: usize, 
}

pub enum SlotTask {
    Bake(BakeTask),
    Load(LoadTask),
}

pub struct LoadTask {
    pub slot_id: usize,
    pub path: PathBuf,
    pub layer_idx: usize,
    pub kv_name: Option<String>,
    pub shared_block: crate::models::qwen3vl::quantized_model::KVBlock,
    pub registry: crate::models::qwen3vl::quantized_model::KVRegistry, 
}

use tokio::sync::OnceCell;
pub static SLOT_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();

pub async fn get_worker_channel() -> Result<mpsc::Sender<SlotTask>> {
    for _ in 0..50 {
        if let Some(tx) = SLOT_TX.get() { return Ok(tx.clone()); }
        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    Err(anyhow!("Slot worker channel initialization timed out"))
}

pub fn init_bake_worker() {
    let (tx, rx) = mpsc::channel(100);
    tauri::async_runtime::spawn(async move { spawn_slot_worker(rx); });
    let _ = SLOT_TX.set(tx);
}

fn spawn_slot_worker(mut rx: mpsc::Receiver<SlotTask>) {
    let (io_tx, mut io_rx) = mpsc::channel::<SaveTask>(1000); 
    
    // Phase B: 디스크 I/O 전담 워커
    tokio::spawn(async move {
        while let Some(task) = io_rx.recv().await {
            if let Some(parent) = task.path.parent() { if !parent.exists() { let _ = std::fs::create_dir_all(parent); } }
            let _ = candle_core::safetensors::save(&task.tensors, &task.path);
            let slot = &SLOT_MANAGER.slots[task.slot_id];
            let remaining = slot.remaining_layers.fetch_sub(1, Ordering::SeqCst);
            if remaining == 1 || task.is_last {
                SLOT_MANAGER.mark_ready(task.slot_id, task.total_bytes_hint).await;
            }
        }
    });

    // Phase A: CPU 작업 전담 워커 (Async-Safe)
    tokio::spawn(async move {
        while let Some(task) = rx.recv().await {
            match task {
                SlotTask::Bake(bake) => {
                    let io_tx_clone = io_tx.clone();
                    // [FIX] 블로킹 연산(압축)만 spawn_blocking으로 처리
                    let _ = tokio::task::spawn_blocking(move || {
                        let slot_id = bake.slot_id;
                        let task_dir = bake.task_dir;
                        let kv_name = bake.kv_name;
                        let offset = bake.offset;
                        if bake.layers.is_empty() { return; }
                        
                        let slot = &SLOT_MANAGER.slots[slot_id];
                        let total_layers = bake.layers.len();
                        slot.remaining_layers.store(total_layers, Ordering::SeqCst);
                        let total_bytes_hint = total_layers * (20 * 1024 * 1024 / 28);

                        // Rayon 병렬 압축은 별도 스레드풀이므로 런타임을 블로킹하지 않음
                        bake.layers.into_par_iter().enumerate().for_each(|(idx, layer)| {
                            let is_last_layer = idx == total_layers - 1;
                            let filename = match (&kv_name, offset) {
                                (Some(name), 0) => format!("layer_{}_kv.safetensors", name),
                                (Some(name), off) => format!("layer_{}_kv_{}.safetensors", name, off),
                                (None, 0) => format!("layer_{}_kv.safetensors", layer.layer_idx),
                                (None, off) => format!("layer_{}_kv_{}.safetensors", layer.layer_idx, off),
                            };
                            let path = task_dir.join(filename);
                            let mut map = std::collections::HashMap::new();
                            let (b, h, s, d) = (layer.shape[0], layer.shape[1], layer.shape[2], layer.shape[3]);
                            let anchor_count = (0..s).filter(|&i| i < 4 || i % 8 == 0).count();
                            let head_token_size = s * d;

                            // K 압축
                            let mut k_anchors = vec![0.0f32; b * h * anchor_count * d];
                            let mut k_packed = vec![0u8; (b * h * s * d + 7) / 8];
                            let mut k_scales = vec![0.0f32; b * h * s];
                            for bh_idx in 0..(b * h) {
                                let bh_offset = bh_idx * head_token_size;
                                for i in 0..head_token_size { if layer.k_data[bh_offset + i] >= 0.0 { k_packed[(bh_offset + i) / 8] |= 1 << ((bh_offset + i) % 8); } }
                                for token_idx in 0..s {
                                    let token_data = &layer.k_data[bh_offset + token_idx * d .. bh_offset + (token_idx + 1) * d];
                                    if token_idx < 4 || token_idx % 8 == 0 {
                                        let a_pos = if token_idx < 4 { token_idx } else { 4 + (token_idx - 4) / 8 };
                                        k_anchors[(bh_idx * anchor_count + a_pos) * d .. (bh_idx * anchor_count + a_pos + 1) * d].copy_from_slice(token_data);
                                    }
                                    let mut max_abs = 0.0f32;
                                    for &v in token_data { let a = v.abs(); if a > max_abs { max_abs = a; } }
                                    k_scales[bh_idx * s + token_idx] = max_abs;
                                }
                            }
                            map.insert("k_anchors".to_string(), Tensor::from_vec(k_anchors, vec![b, h, anchor_count, d], &Device::Cpu).unwrap());
                            map.insert("k_packed".to_string(), Tensor::from_vec(k_packed, vec![(b * h * s * d + 7) / 8], &Device::Cpu).unwrap());
                            map.insert("k_scales".to_string(), Tensor::from_vec(k_scales, vec![b, h, s, 1], &Device::Cpu).unwrap());
                            map.insert("k_shape".to_string(), Tensor::from_vec(vec![b as u32, h as u32, s as u32, d as u32], (4,), &Device::Cpu).unwrap());

                            // V 압축
                            let mut v_anchors = vec![0.0f32; b * h * anchor_count * d];
                            let mut v_packed = vec![0u8; (b * h * s * d + 7) / 8];
                            let mut v_scales = vec![0.0f32; b * h * s];
                            for bh_idx in 0..(b * h) {
                                let bh_offset = bh_idx * head_token_size;
                                for i in 0..head_token_size { if layer.v_data[bh_offset + i] >= 0.0 { v_packed[(bh_offset + i) / 8] |= 1 << ((bh_offset + i) % 8); } }
                                for token_idx in 0..s {
                                    let token_data = &layer.v_data[bh_offset + token_idx * d .. bh_offset + (token_idx + 1) * d];
                                    if token_idx < 4 || token_idx % 8 == 0 {
                                        let a_pos = if token_idx < 4 { token_idx } else { 4 + (token_idx - 4) / 8 };
                                        v_anchors[(bh_idx * anchor_count + a_pos) * d .. (bh_idx * anchor_count + a_pos + 1) * d].copy_from_slice(token_data);
                                    }
                                    let mut max_abs = 0.0f32;
                                    for &v in token_data { let a = v.abs(); if a > max_abs { max_abs = a; } }
                                    v_scales[bh_idx * s + token_idx] = max_abs;
                                }
                            }
                            map.insert("v_anchors".to_string(), Tensor::from_vec(v_anchors, vec![b, h, anchor_count, d], &Device::Cpu).unwrap());
                            map.insert("v_packed".to_string(), Tensor::from_vec(v_packed, vec![(b * h * s * d + 7) / 8], &Device::Cpu).unwrap());
                            map.insert("v_scales".to_string(), Tensor::from_vec(v_scales, vec![b, h, s, 1], &Device::Cpu).unwrap());
                            map.insert("mode".to_string(), Tensor::from_vec(vec![3u32], (1,), &Device::Cpu).unwrap());
                            
                            // [FIX] 비동기 런타임 밖이므로 blocking_send가 안전함
                            let _ = io_tx_clone.blocking_send(SaveTask { slot_id, path, tensors: map, is_last: is_last_layer, total_bytes_hint });
                        });
                    }).await;
                }
                SlotTask::Load(load) => {
                    let (offset, index) = { let inner = load.shared_block.inner.read().unwrap(); (inner.offset, inner.index) };
                    let layer_idx = load.layer_idx;
                    let filename = if offset == 0 { format!("layer_{}_kv.safetensors", layer_idx) } else { format!("layer_{}_kv_{}.safetensors", layer_idx, offset) };
                    let path = load.path.join(filename);
                    let mut success = false;
                    let mut total_bytes = 0;
                    if path.exists() {
                        if let Ok(content) = std::fs::read(&path) {
                            if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                                let ex_vec = |name: &str| -> Option<Vec<f32>> { st.tensor(name).ok().map(|v| unsafe { std::slice::from_raw_parts(v.data().as_ptr() as *const f32, v.data().len() / 4).to_vec() }) };
                                if let (Some(ka), Some(kp), Some(ks), Some(va), Some(vp), Some(vs)) = (ex_vec("k_anchors"), st.tensor("k_packed").ok(), ex_vec("k_scales"), ex_vec("v_anchors"), st.tensor("v_packed").ok(), ex_vec("v_scales")) {
                                    let o_shape = if let Ok(view) = st.tensor("k_shape") {
                                        let s_u32: &[u32] = unsafe { std::slice::from_raw_parts(view.data().as_ptr() as *const u32, view.data().len() / 4) };
                                        s_u32.iter().map(|&x| x as usize).collect()
                                    } else { vec![1, 8, 1024, 128] };
                                    let mut inner = load.shared_block.inner.write().unwrap();
                                    inner.bitkv_metadata = Some(crate::models::qwen3vl::quantized_model::BitKVMetadata {
                                        k_anchors: Tensor::from_vec(ka, (o_shape[0], o_shape[1], (o_shape[2] + 7) / 8 + 4, o_shape[3]), &Device::Cpu).unwrap(),
                                        k_packed: Tensor::from_slice(kp.data(), kp.shape(), &Device::Cpu).unwrap(),
                                        k_scales: Tensor::from_vec(ks, (o_shape[0], o_shape[1], o_shape[2], 1), &Device::Cpu).unwrap(),
                                        v_anchors: Tensor::from_vec(va, (o_shape[0], o_shape[1], (o_shape[2] + 7) / 8 + 4, o_shape[3]), &Device::Cpu).unwrap(),
                                        v_packed: Tensor::from_slice(vp.data(), vp.shape(), &Device::Cpu).unwrap(),
                                        v_scales: Tensor::from_vec(vs, (o_shape[0], o_shape[1], o_shape[2], 1), &Device::Cpu).unwrap(),
                                        original_shape: o_shape,
                                    });
                                    total_bytes = content.len(); success = true;
                                }
                            }
                        }
                    }
                    if success {
                        { let mut reg = load.registry.entries.write().unwrap(); if index < reg.len() { reg[index].location[layer_idx] = KVLocation::RAM; reg[index].slot_ids[layer_idx] = Some(load.slot_id); } }
                        SLOT_MANAGER.mark_ready(load.slot_id, total_bytes).await;
                    } else {
                        { let mut reg = load.registry.entries.write().unwrap(); if index < reg.len() { reg[index].location[layer_idx] = KVLocation::SSD; } }
                        SLOT_MANAGER.release_slot(load.slot_id).await;
                    }
                }
            }
        }
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
        match self { Self::Standard(_) => Ok(()), Self::QuantizedVL(m) => m.rebalance_layers(device_id), Self::QuantizedText(m) => m.rebalance_layers(device_id), }
    }

    pub fn drop_kv_storage(&mut self) -> Result<()> {
        match self { Self::Standard(_) => Ok(()), Self::QuantizedVL(m) => m.language_model.drop_kv_storage(), Self::QuantizedText(m) => m.language_model.drop_kv_storage(), }
    }

    pub fn inject_kv_bitkv(&mut self, k_anchors: &[Tensor], k_packed: &[Tensor], k_scales: &[Tensor], v_anchors: &[Tensor], v_packed: &[Tensor], v_scales: &[Tensor], original_shape: &[usize]) -> Result<()> {
        match self { Self::QuantizedVL(m) => m.language_model.inject_live_kv_bitkv(k_anchors, k_packed, k_scales, v_anchors, v_packed, v_scales, original_shape), Self::QuantizedText(m) => m.language_model.inject_live_kv_bitkv(k_anchors, k_packed, k_scales, v_anchors, v_packed, v_scales, original_shape), _ => Ok(()), }
    }

    pub fn is_cpu(&self) -> bool {
        match self { Self::Standard(m) => m.device().is_cpu(), Self::QuantizedVL(m) => m.language_model.is_forced_cpu, Self::QuantizedText(m) => m.language_model.is_forced_cpu, }
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

        let cfg: Qwen3VLConfig = if raw_config.get("text_config").is_some() { serde_json::from_value(raw_config)? } 
        else {
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
                vision_config: None, vision_start_token_id: None, vision_end_token_id: None,
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
            let mut m_path = gguf_files.iter().find(|f| f.contains("Qwen3-0.6B-Q8_0.gguf")).cloned();
            if m_path.is_none() { m_path = gguf_files.iter().find(|f| f.contains("Qwen3-0.6B-Q4_K_M.gguf")).cloned(); }
            if m_path.is_none() { m_path = gguf_files.iter().find(|f| !f.contains("mmproj")).cloned(); }
            let l_toks = hard_token_limit.unwrap_or(4096) as u64;  
            let r_toks = l_toks.min(8192);
            let kv_res = r_toks * 40000;

            if is_vision_model {
                let mm = mmproj_path.ok_or(anyhow!("Missing mmproj GGUF"))?;
                let main = m_path.ok_or(anyhow!("Missing main GGUF for VL model"))?;
                let main_file = std::fs::File::open(&main)?;
                let main_mmap = unsafe { memmap2::MmapOptions::new().map(&main_file)? };
                let mmproj_file = std::fs::File::open(&mm)?;
                let mmproj_mmap = unsafe { memmap2::MmapOptions::new().map(&mmproj_file)? };
                let mut m_cursor = std::io::Cursor::new(&main_mmap[..]);
                let m_content = gguf_file::Content::read(&mut m_cursor)?;
                let mut mm_cursor = std::io::Cursor::new(&mmproj_mmap[..]);
                let mm_content = gguf_file::Content::read(&mut mm_cursor)?;
                let model = QuantizedQwen3VLModel::new_with_mmap(&cfg, &m_content, Some(Arc::new(main_mmap)), &mm_content, Some(Arc::new(mmproj_mmap)), &text_dev, text_device_id, &vision_dev, vision_device_id, dtype, kv_res, baking_only)?;
                ModelVariant::QuantizedVL(model)
            } else {
                let main = m_path.or_else(|| if !gguf_files.is_empty() { Some(gguf_files[0].clone()) } else { None }).ok_or(anyhow!("No GGUF file found"))?;
                let file = std::fs::File::open(&main)?;
                let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
                let mut cursor = std::io::Cursor::new(&mmap[..]);        
                let content = gguf_file::Content::read(&mut cursor)?;
                let is_06b = path.contains("0.6B");
                let model = crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel::new_with_mmap(&cfg, &content, Some(Arc::new(mmap)), &text_dev, text_device_id, dtype, kv_res, baking_only || is_06b, baking_only || is_06b)?;
                ModelVariant::QuantizedText(model)
            }
        } else {
            let model_list = find_type_files(path, "safetensors")?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &text_dev)? };
            ModelVariant::Standard(Qwen3VLModel::new(cfg, vb)?)
        };

        let g_path = std::path::Path::new(cfg_path).join("generation_config.json");
        let g_cfg: Qwen3VLGenerationConfig = if g_path.exists() { serde_json::from_slice(&std::fs::read(g_path)?)? } else { Qwen3VLGenerationConfig::default() };
        let m_name = if path.contains("0.6B") { "qwen3vl-0.6B".to_string() } else { "qwen3vl-2B".to_string() };
        let (eos1, eos2) = match &g_cfg.eos_token_id {
            serde_json::Value::Number(n) => { let id = n.as_u64().unwrap_or(151645) as u32; (id, id) },
            serde_json::Value::Array(arr) => { let id1 = arr.get(0).and_then(|v| v.as_u64()).unwrap_or(151643) as u32; let id2 = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(id1 as u64) as u32; (id1, id2) },
            _ => (151643, 151643),
        };
        Ok(Self { chat_template, tokenizer, pre_processor, qwen3_vl, text_device: text_dev, vision_device: vision_dev, eos_token_id1: eos1, eos_token_id2: eos2, generation_config: g_cfg, model_name: m_name, hard_token_limit, kv_root })
    }

    pub fn prefill_text_only(&mut self, text: &str, _cancel_token: Option<Arc<AtomicBool>>, mut relay_target: Option<&mut Qwen3VLGenerateModel>, auto_save_path: Option<&std::path::Path>) -> Result<()> {
        let t_ids = self.tokenizer.text_encode_vec(text.to_string(), false)?;
        let t_toks = t_ids.len();
        let c_size = 512;
        let mut c_pos = 0;

        while c_pos < t_toks {
            let end = (c_pos + c_size).min(t_toks);
            let chunk = &t_ids[c_pos..end];
            let c_ids = Tensor::from_vec(chunk.to_vec(), (1, end - c_pos), &self.text_device)?;
            let c_pos_t = Tensor::arange(c_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?;
            self.qwen3_vl.forward(&c_ids, None, None, None, None, Some(&c_pos_t), c_pos, t_toks, None)?;
            if let Some(path) = auto_save_path { let _ = self.save_kv_to_disk(path, None, end); }
            if let Some(ref mut target) = relay_target {
                let (ks, vs) = self.get_current_kv();
                let res: Result<Vec<_>> = ks.par_iter().zip(vs.par_iter()).map(|(k, v): (&Tensor, &Tensor)| {
                    let s_len = k.dim(candle_core::D::Minus2)?;
                    let k_vec = k.narrow(candle_core::D::Minus2, s_len.saturating_sub(end - c_pos), end - c_pos)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let v_vec = v.narrow(candle_core::D::Minus2, s_len.saturating_sub(end - c_pos), end - c_pos)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    if let ModelVariant::QuantizedText(m) = &self.qwen3_vl {
                        let rk = m.language_model.compress_to_bitkv(&Tensor::from_vec(k_vec, (1, 8, end-c_pos, 128), &Device::Cpu)?)?;
                        let rv = m.language_model.compress_to_bitkv(&Tensor::from_vec(v_vec, (1, 8, end-c_pos, 128), &Device::Cpu)?)?;
                        Ok((rk, rv))
                    } else { Err(anyhow!("Unsupported")) }
                }).collect();
                let results = res?;
                let (mut ka, mut kp, mut kst, mut va, mut vp, mut vst, mut os) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
                for (rk, rv) in results { ka.push(rk.0); kp.push(rk.1); kst.push(rk.2); va.push(rv.0); vp.push(rv.1); vst.push(rv.2); os = rk.3; }
                if !ka.is_empty() { target.inject_kv_bitkv(&ka, &kp, &kst, &va, &vp, &vst, &os)?; }
            }
            c_pos = end;
        }
        if auto_save_path.is_some() { let _ = self.qwen3_vl.drop_kv_storage(); }
        Ok(())
    }

    pub async fn prefill_chunk(&mut self, text: String, _cancel_flag: Option<Arc<AtomicBool>>, mut relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let ids = self.tokenizer.text_encode_vec(text, false)?;
        let size = ids.len();
        let pos = self.get_kv_len();
        let c_ids = Tensor::from_vec(ids, (1, size), &self.text_device)?;
        let c_pos = Tensor::arange(pos as u32, (pos + size) as u32, &self.text_device)?.unsqueeze(0)?;
        self.qwen3_vl.forward(&c_ids, None, None, None, None, Some(&c_pos), pos, size, None)?;
        if let Some(ref mut target) = relay_target {
            let (ks, vs) = self.get_current_kv();
            let res: Result<Vec<_>> = ks.par_iter().zip(vs.par_iter()).map(|(k, v): (&Tensor, &Tensor)| {
                let s_len = k.dim(candle_core::D::Minus2)?;
                let k_vec = k.narrow(2, s_len.saturating_sub(size), size)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let v_vec = v.narrow(2, s_len.saturating_sub(size), size)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                if let ModelVariant::QuantizedText(m) = &self.qwen3_vl {
                    let rk = m.language_model.compress_to_bitkv(&Tensor::from_vec(k_vec, (1, 8, size, 128), &Device::Cpu)?)?;
                    let rv = m.language_model.compress_to_bitkv(&Tensor::from_vec(v_vec, (1, 8, size, 128), &Device::Cpu)?)?;
                    Ok((rk, rv))
                } else { Err(anyhow!("Unsupported")) }
            }).collect();
            let results = res?;
            let (mut ka, mut kp, mut kst, mut va, mut vp, mut vst, mut os) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
            for (rk, rv) in results { ka.push(rk.0); kp.push(rk.1); kst.push(rk.2); va.push(rv.0); vp.push(rv.1); vst.push(rv.2); os = rk.3; }
            if !ka.is_empty() { target.inject_kv_bitkv(&ka, &kp, &kst, &va, &vp, &vst, &os)?; }
        }
        Ok(size)
    }

    pub async fn prefill_only(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _relay_target: Option<&mut Qwen3VLGenerateModel>, kv_name: Option<String>) -> Result<usize> {
        SLOT_MANAGER.reset_all_slots().await;
        let m_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &m_render)?;
        let full_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let t_toks = full_ids.len();
        let c_size = 2048;
        let mut c_pos = self.get_kv_len();

        while c_pos < t_toks {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            let end = (c_pos + c_size).min(t_toks);
            let c_len = end - c_pos;
            let c_ids = Tensor::from_vec(full_ids[c_pos..end].to_vec(), (1, c_len), &self.text_device)?;
            let c_pos_t = Tensor::arange(c_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?;

            println!("[BAKING] {} to {} / Total: {}", c_pos, end, t_toks);
            self.qwen3_vl.forward(&c_ids, None, None, None, None, Some(&c_pos_t), c_pos, t_toks, session_id.clone())?;

            if let Some(sid) = &session_id {
                let slot_id = SLOT_MANAGER.acquire_write_slot(t_toks).await;
                let path = crate::utils::paths::get_kv_dir(None).join(sid);
                if !path.exists() { let _ = fs::create_dir_all(&path); }

                let (ks, vs) = self.get_current_kv();
                if ks.is_empty() { return Err(anyhow!("No KV data")); }

                let mut dumps = Vec::new();
                for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() {
                    let s_len = k.dim(2)?;
                    let k_vec = k.narrow(2, s_len.saturating_sub(c_len), c_len)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let v_vec = v.narrow(2, s_len.saturating_sub(c_len), c_len)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    dumps.push(LayerKVDump { layer_idx: idx, k_data: k_vec, v_data: v_vec, shape: vec![1, 8, c_len, 128] });
                }

                if let Some(reg_obj) = match self.qwen3_vl { ModelVariant::QuantizedText(ref m) => Some(m.language_model.registry.clone()), ModelVariant::QuantizedVL(ref m) => Some(m.language_model.registry.clone()), _ => None } {
                    let mut reg = reg_obj.entries.write().unwrap();
                    for entry in reg.iter_mut() { for l_idx in 0..28 { if entry.location[l_idx] == KVLocation::VRAM { entry.location[l_idx] = KVLocation::SSD; entry.ssd_path = Some(path.clone()); } } }
                }
                
                self.clear_temporal_kv_caches();
                let dev = self.text_device.clone();
                if dev.is_cuda() { let _ = dev.synchronize(); }

                if let Ok(tx) = get_worker_channel().await {
                    let _ = tx.send(SlotTask::Bake(BakeTask { slot_id, task_dir: path, kv_name: kv_name.clone(), offset: end, layers: dumps })).await;
                }
                self.clear_temporal_kv_caches();
            }
            c_pos = end;
        }
        Ok(c_pos)
    }

    pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, kv_name: Option<String>) -> Result<String> {
        SLOT_MANAGER.reset_all_slots().await;
        let mut l_proc = get_logit_processor(Some(mes.temperature.unwrap_or(0.7) as f32), Some(mes.top_p.unwrap_or(0.9) as f32), Some(40), mes.seed.unwrap_or(34562) as u64);
        let m_render = self.chat_template.apply_chat_template(&mes)?;
        let mut input = self.pre_processor.process_info(&mes, &m_render)?;
        let full_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let t_toks = full_ids.len();
        let mut all_ids = full_ids.clone();
        let mut gen_text = String::new();
        let mut s_off = self.get_kv_len();
        let mut l_pos = s_off;
        let c_size = 128;

        while l_pos < t_toks {
            let chunk_size = (t_toks - l_pos).min(c_size);
            if chunk_size == 0 { break; }
            let c_ids = Tensor::from_vec(full_ids[l_pos..l_pos+chunk_size].to_vec(), (1, chunk_size), &self.text_device)?;
            let c_pos_t = Tensor::arange(s_off as u32, (s_off + chunk_size) as u32, &self.text_device)?.unsqueeze(0)?;
            self.qwen3_vl.forward(&c_ids, None, None, None, None, Some(&c_pos_t), s_off, t_toks, session_id.clone())?;
            l_pos += chunk_size; s_off += chunk_size;

            if let Some(sid) = &session_id {
                let slot_id = SLOT_MANAGER.acquire_write_slot(t_toks).await;
                let path = crate::utils::paths::get_kv_dir(None).join(sid);
                let (ks, vs) = self.get_current_kv();
                if !ks.is_empty() {
                    let mut dumps = Vec::new();
                    for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() {
                        let k_vec = k.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        let v_vec = v.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        dumps.push(LayerKVDump { layer_idx: idx, k_data: k_vec, v_data: v_vec, shape: vec![1, 8, chunk_size, 128] });
                    }
                    if let Some(reg_obj) = match self.qwen3_vl { ModelVariant::QuantizedText(ref m) => Some(m.language_model.registry.clone()), ModelVariant::QuantizedVL(ref m) => Some(m.language_model.registry.clone()), _ => None } {
                        let mut reg = reg_obj.entries.write().unwrap();
                        for entry in reg.iter_mut() { for l_idx in 0..28 { if entry.location[l_idx] == KVLocation::VRAM { entry.location[l_idx] = KVLocation::SSD; entry.ssd_path = Some(path.clone()); } } }
                    }
                    let dev = self.text_device.clone();
                    if dev.is_cuda() { let _ = dev.synchronize(); }
                    if let Ok(tx) = get_worker_channel().await { let _ = tx.send(SlotTask::Bake(BakeTask { slot_id, task_dir: path, kv_name: kv_name.clone(), offset: s_off, layers: dumps })).await; }
                } else { SLOT_MANAGER.release_slot(slot_id).await; }
            }
            self.clear_temporal_kv_caches();
        }

        let m_new = mes.max_tokens.unwrap_or(2048);
        let mut p_vals = input.pixel_values.take();
        let i_grid = input.image_grid_thw.take();

        for _ in 0..m_new {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            let input_ids = Tensor::new(vec![*all_ids.last().unwrap()], &self.text_device)?.unsqueeze(0)?;
            let chunk_pos = Tensor::arange(s_off as u32, (s_off + 1) as u32, &self.text_device)?.unsqueeze(0)?;
            let logits = self.qwen3_vl.forward(&input_ids, p_vals.as_ref(), i_grid.as_ref(), None, None, Some(&chunk_pos), s_off, t_toks, session_id.clone())?;
            let mut l_vec = logits.squeeze(0)?.i(logits.dim(1)? - 1)?.to_dtype(DType::F32)?;
            l_vec = apply_repeat_penalty(&l_vec, 1.1, if all_ids.len() > 512 { &all_ids[all_ids.len()-512..] } else { &all_ids[..] })?;
            let next_id = l_proc.sample(&l_vec)?;
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            all_ids.push(next_id);
            gen_text.push_str(&self.tokenizer.token_decode(vec![next_id])?);
            s_off += 1; p_vals = None;
            self.clear_temporal_kv_caches();
        }
        Ok(gen_text)
    }

    pub fn clear_temporal_kv_caches(&mut self) {
        match self.qwen3_vl {
            ModelVariant::QuantizedText(ref mut m) => {
                let reg_obj = m.language_model.registry.clone();
                let reg = reg_obj.entries.read().unwrap();
                for (layer_idx, layer) in m.language_model.layers.iter_mut().enumerate() {
                    for block in &mut layer.self_attn.kv_blocks {
                        let mut inner = block.inner.write().unwrap();
                        if inner.location == KVLocation::VRAM {
                            let reg_loc = if inner.index < reg.len() { reg[inner.index].location[layer_idx] } else { KVLocation::VRAM };
                            if reg_loc != KVLocation::VRAM { inner.k_cache = None; inner.v_cache = None; inner.location = reg_loc; }
                        }
                    }
                }
            },
            ModelVariant::QuantizedVL(ref mut m) => {
                let reg_obj = m.language_model.registry.clone();
                let reg = reg_obj.entries.read().unwrap();
                for (layer_idx, layer) in m.language_model.layers.iter_mut().enumerate() {
                    for block in &mut layer.self_attn.kv_blocks {
                        let mut inner = block.inner.write().unwrap();
                        if inner.location == KVLocation::VRAM {
                            let reg_loc = if inner.index < reg.len() { reg[inner.index].location[layer_idx] } else { KVLocation::VRAM };
                            if reg_loc != KVLocation::VRAM { inner.k_cache = None; inner.v_cache = None; inner.location = reg_loc; }
                        }
                    }
                }
            },
            _ => {}
        }
    }

    pub fn get_kv_len(&self) -> usize {
        match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.get_kv_len(), ModelVariant::QuantizedText(m) => m.language_model.get_kv_len(), _ => 0 }
    }

    pub fn get_current_kv(&self) -> (Vec<Tensor>, Vec<Tensor>) {
        let mut ks = vec![]; let mut vs = vec![];
        match &self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => { 
                for l in &m.language_model.layers { 
                    let (mut lk, mut lv) = (Vec::new(), Vec::new());
                    for b in &l.self_attn.kv_blocks { let inner = b.inner.read().unwrap(); if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) { lk.push(k.clone()); lv.push(v.clone()); } }
                    if !lk.is_empty() { if let (Ok(k), Ok(v)) = (Tensor::cat(&lk, 2), Tensor::cat(&lv, 2)) { ks.push(k); vs.push(v); } }
                } 
            },
            ModelVariant::QuantizedText(m) => { 
                for l in &m.language_model.layers { 
                    let (mut lk, mut lv) = (Vec::new(), Vec::new());
                    for b in &l.self_attn.kv_blocks { let inner = b.inner.read().unwrap(); if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) { lk.push(k.clone()); lv.push(v.clone()); } }
                    if !lk.is_empty() { if let (Ok(k), Ok(v)) = (Tensor::cat(&lk, 2), Tensor::cat(&lv, 2)) { ks.push(k); vs.push(v); } }
                } 
            },
            _ => {} 
        }
        (ks, vs)
    }

    pub fn inject_kv_bitkv(&mut self, ka: &[Tensor], kp: &[Tensor], ks: &[Tensor], va: &[Tensor], vp: &[Tensor], vs: &[Tensor], os: &[usize]) -> Result<()> {
        match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.inject_live_kv_bitkv(ka, kp, ks, va, vp, vs, os), ModelVariant::QuantizedText(m) => m.language_model.inject_live_kv_bitkv(ka, kp, ks, va, vp, vs, os), _ => Ok(()) }
    }

    pub fn save_kv_to_disk(&mut self, path: &Path, kv_name: Option<&str>, offset: usize) -> Result<()> {
        match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.save_kv_cache(path, false, offset, kv_name), ModelVariant::QuantizedText(m) => m.save_kv_cache(path, false, offset, kv_name), _ => Ok(()) }
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.truncate_kv_cache(len), ModelVariant::QuantizedText(m) => m.truncate_kv_cache(len), _ => Ok(()) }
    }

    pub fn load_kv_from_disk(&mut self, path: &Path, kv_name: Option<&str>) -> Result<()> {
        match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.load_kv_cache(path, &self.text_device, 0, 128, kv_name), ModelVariant::QuantizedText(m) => m.load_kv_cache(path, &self.text_device, 0, 128, kv_name), _ => Ok(()) }
    }

    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.to_device(d)?, ModelVariant::QuantizedText(m) => m.to_device(d)?, _ => {} }
        self.text_device = d.clone(); self.vision_device = d.clone(); Ok(())
    }

    pub fn clear_kv_cache(&mut self) {
        match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.clear_kv_cache(), ModelVariant::QuantizedText(m) => m.clear_kv_cache(), _ => {} }
    }
}

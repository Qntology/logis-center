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
    Release { idx: usize }, 
    MarkReady { idx: usize, bytes: usize },
    Flush { response: oneshot::Sender<()> },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum InternalState { Free, Writing, Ready, Reading }

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
        for i in 0..count { slots.push(crate::models::qwen3vl::quantized_model::MemorySlot::new(i, 28)); }
        let (tx, rx) = mpsc::channel(1000); 
        let count_reads = Arc::new(AtomicUsize::new(0));
        let count_writes = Arc::new(AtomicUsize::new(0));
        let count_cached = Arc::new(AtomicUsize::new(0));
        let count_free = Arc::new(AtomicUsize::new(count));
        let (cr, cw, cc, cf) = (count_reads.clone(), count_writes.clone(), count_cached.clone(), count_free.clone());
        tauri::async_runtime::spawn(async move { Self::slot_dispatcher(rx, cr, cw, cc, cf, count).await; });
        Self { slots, request_tx: tx, count_reads, count_writes, count_cached, count_free }
    }

    async fn slot_dispatcher(mut rx: mpsc::Receiver<SlotRequest>, count_r: Arc<AtomicUsize>, count_w: Arc<AtomicUsize>, count_c: Arc<AtomicUsize>, count_f: Arc<AtomicUsize>, max_slots: usize) {
        let mut slot_states = vec![InternalState::Free; max_slots];
        let mut free_pool: VecDeque<usize> = (0..max_slots).collect();
        let mut ready_pool: VecDeque<usize> = VecDeque::new();
        let mut pending_writes: VecDeque<(oneshot::Sender<usize>, usize)> = VecDeque::new();
        let mut pending_reads: VecDeque<oneshot::Sender<usize>> = VecDeque::new();
        let mut flush_waiters: Vec<oneshot::Sender<()>> = Vec::new();
        let mut current_ram_usage: usize = 0;
        let ram_threshold = 100 * 1024 * 1024;
        let mut sys = sysinfo::System::new_all();

        while let Some(request) = rx.recv().await {
            match request {
                SlotRequest::AcquireWrite { response, total_tokens } => {
                    sys.refresh_memory();
                    let max_c = Self::calculate_dynamic_budget(&sys, total_tokens, max_slots);
                    if count_w.load(Ordering::SeqCst) < max_c {
                        if let Some(idx) = free_pool.pop_front().or_else(|| ready_pool.pop_front()) {
                            let old = slot_states[idx];
                            slot_states[idx] = InternalState::Writing;
                            if old == InternalState::Free { count_f.fetch_sub(1, Ordering::SeqCst); } else { count_c.fetch_sub(1, Ordering::SeqCst); }
                            count_w.fetch_add(1, Ordering::SeqCst); let _ = response.send(idx); continue;
                        }
                    }
                    pending_writes.push_back((response, total_tokens));
                }
                SlotRequest::AcquireRead { response } => {
                    if let Some(idx) = free_pool.pop_front().or_else(|| ready_pool.pop_front()) {
                        let old = slot_states[idx];
                        slot_states[idx] = InternalState::Reading;
                        if old == InternalState::Free { count_f.fetch_sub(1, Ordering::SeqCst); } else { count_c.fetch_sub(1, Ordering::SeqCst); }
                        count_r.fetch_add(1, Ordering::SeqCst); let _ = response.send(idx);
                    } else { pending_reads.push_back(response); }
                }
                SlotRequest::Release { idx } => {
                    if idx < max_slots {
                        let old = slot_states[idx];
                        if old != InternalState::Free {
                            slot_states[idx] = InternalState::Free; free_pool.push_back(idx); count_f.fetch_add(1, Ordering::SeqCst);
                            match old { InternalState::Writing => { count_w.fetch_sub(1, Ordering::SeqCst); }, InternalState::Ready => { count_c.fetch_sub(1, Ordering::SeqCst); }, InternalState::Reading => { count_r.fetch_sub(1, Ordering::SeqCst); }, _ => {} }
                        }
                    }
                    Self::check_flush(&count_w, &count_r, &mut flush_waiters);
                    Self::process_queues_robust(&sys, &mut free_pool, &mut ready_pool, &mut pending_writes, &mut pending_reads, &mut slot_states, &count_f, &count_c, &count_w, &count_r, max_slots);
                }
                SlotRequest::MarkReady { idx, bytes } => {
                    if idx < max_slots {
                        let old = slot_states[idx];
                        if old == InternalState::Writing || old == InternalState::Reading {
                            slot_states[idx] = InternalState::Ready; ready_pool.push_back(idx); count_c.fetch_add(1, Ordering::SeqCst); current_ram_usage += bytes;
                            if old == InternalState::Writing { count_w.fetch_sub(1, Ordering::SeqCst); } else { count_r.fetch_sub(1, Ordering::SeqCst); }
                            if current_ram_usage > ram_threshold {
                                while current_ram_usage > (ram_threshold / 2) && !ready_pool.is_empty() {
                                    if let Some(ev_idx) = ready_pool.pop_front() {
                                        if slot_states[ev_idx] == InternalState::Ready {
                                            slot_states[ev_idx] = InternalState::Free; free_pool.push_back(ev_idx); count_f.fetch_add(1, Ordering::SeqCst); count_c.fetch_sub(1, Ordering::SeqCst);
                                            current_ram_usage = current_ram_usage.saturating_sub(20 * 1024 * 1024);
                                            let slot = &crate::models::qwen3vl::generate::SLOT_MANAGER.slots[ev_idx];
                                            for l in &slot.k_layers { if let Ok(mut g) = l.try_lock() { *g = None; } }
                                            for l in &slot.v_layers { if let Ok(mut g) = l.try_lock() { *g = None; } }
                                            slot.state.store(0, Ordering::SeqCst);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Self::check_flush(&count_w, &count_r, &mut flush_waiters);
                    Self::process_queues_robust(&sys, &mut free_pool, &mut ready_pool, &mut pending_writes, &mut pending_reads, &mut slot_states, &count_f, &count_c, &count_w, &count_r, max_slots);
                }
                SlotRequest::Flush { response } => {
                    if count_w.load(Ordering::SeqCst) == 0 && count_r.load(Ordering::SeqCst) == 0 { let _ = response.send(()); } 
                    else { flush_waiters.push(response); }
                }
            }
        }
    }

    fn check_flush(cw: &Arc<AtomicUsize>, cr: &Arc<AtomicUsize>, waiters: &mut Vec<oneshot::Sender<()>>) {
        if cw.load(Ordering::SeqCst) == 0 && cr.load(Ordering::SeqCst) == 0 { while let Some(w) = waiters.pop() { let _ = w.send(()); } }
    }

    fn process_queues_robust(sys: &sysinfo::System, free_pool: &mut VecDeque<usize>, ready_pool: &mut VecDeque<usize>, pending_writes: &mut VecDeque<(oneshot::Sender<usize>, usize)>, pending_reads: &mut VecDeque<oneshot::Sender<usize>>, slot_states: &mut [InternalState], count_f: &Arc<AtomicUsize>, count_c: &Arc<AtomicUsize>, count_w: &Arc<AtomicUsize>, count_r: &Arc<AtomicUsize>, max_slots: usize) {
        while !pending_writes.is_empty() {
            let max_c = Self::calculate_dynamic_budget(sys, pending_writes.front().unwrap().1, max_slots);
            if count_w.load(Ordering::SeqCst) < max_c {
                if let Some(idx) = free_pool.pop_front().or_else(|| ready_pool.pop_front()) {
                    let (res, _) = pending_writes.pop_front().unwrap(); let old = slot_states[idx]; slot_states[idx] = InternalState::Writing;
                    if old == InternalState::Free { count_f.fetch_sub(1, Ordering::SeqCst); } else { count_c.fetch_sub(1, Ordering::SeqCst); }
                    count_w.fetch_add(1, Ordering::SeqCst); let _ = res.send(idx); continue;
                }
            }
            break;
        }
        while !pending_reads.is_empty() {
            if let Some(idx) = free_pool.pop_front().or_else(|| ready_pool.pop_front()) {
                let res = pending_reads.pop_front().unwrap(); let old = slot_states[idx]; slot_states[idx] = InternalState::Reading;
                if old == InternalState::Free { count_f.fetch_sub(1, Ordering::SeqCst); } else { count_c.fetch_sub(1, Ordering::SeqCst); }
                count_r.fetch_add(1, Ordering::SeqCst); let _ = res.send(idx); continue;
            }
            break;
        }
    }

    fn calculate_dynamic_budget(sys: &sysinfo::System, total_tokens: usize, max_slots: usize) -> usize {
        let avail = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        let budget = ((avail - 1.5).max(0.0) / 0.02) as usize;
        budget.min((total_tokens / 1024).saturating_add(2)).min(max_slots).max(4).min(max_slots)
    }

    pub async fn wait_for_all_tasks(&self) {
        let (tx, rx) = oneshot::channel();
        if self.request_tx.send(SlotRequest::Flush { response: tx }).await.is_ok() { let _ = rx.await; }
    }

    pub async fn reset_all_slots(&self) { for i in 0..self.slots.len() { self.release_slot(i).await; } }
    pub async fn acquire_read_slot(&self) -> usize { let (tx, rx) = oneshot::channel(); let _ = self.request_tx.send(SlotRequest::AcquireRead { response: tx }).await; rx.await.unwrap_or(0) }
    pub async fn acquire_write_slot(&self, total_tokens: usize) -> usize { let (tx, rx) = oneshot::channel(); let _ = self.request_tx.send(SlotRequest::AcquireWrite { response: tx, total_tokens }).await; rx.await.unwrap_or(0) }
    pub async fn release_slot(&self, id: usize) {
        if id < self.slots.len() {
            for l in &self.slots[id].k_layers { if let Ok(mut g) = l.try_lock() { *g = None; } }
            for l in &self.slots[id].v_layers { if let Ok(mut g) = l.try_lock() { *g = None; } }
            self.slots[id].state.store(0, Ordering::SeqCst); let _ = self.request_tx.send(SlotRequest::Release { idx: id }).await;
        }
    }
    pub async fn mark_ready(&self, id: usize, bytes: usize) { if id < self.slots.len() { self.slots[id].state.store(2, Ordering::SeqCst); let _ = self.request_tx.send(SlotRequest::MarkReady { idx: id, bytes }).await; } }
    pub fn get_counts(&self) -> (usize, usize, usize, usize) { (self.count_reads.load(Ordering::Relaxed), self.count_writes.load(Ordering::Relaxed), self.count_cached.load(Ordering::Relaxed), self.count_free.load(Ordering::Relaxed)) }
}

pub static SLOT_MANAGER: once_cell::sync::Lazy<SlotManager> = once_cell::sync::Lazy::new(|| SlotManager::new(65));

struct LayerKVDump { layer_idx: usize, k_data: Vec<f32>, v_data: Vec<f32>, shape: Vec<usize> }
struct BakeTask { slot_id: usize, task_dir: PathBuf, kv_name: Option<String>, offset: usize, layers: Vec<LayerKVDump>, is_relay_baking: bool }
struct SaveTask { slot_id: usize, path: PathBuf, tensors: std::collections::HashMap<String, Tensor>, is_last: bool, total_bytes_hint: usize }
pub enum SlotTask { Bake(BakeTask), Load(LoadTask) }
pub struct LoadTask { pub slot_id: usize, pub path: PathBuf, pub layer_idx: usize, pub kv_name: Option<String>, pub shared_block: crate::models::qwen3vl::quantized_model::KVBlock, pub registry: crate::models::qwen3vl::quantized_model::KVRegistry }

use tokio::sync::OnceCell;
pub static SLOT_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();
pub async fn get_worker_channel() -> Result<mpsc::Sender<SlotTask>> {
    for _ in 0..50 { if let Some(tx) = SLOT_TX.get() { return Ok(tx.clone()); } tokio::task::yield_now().await; tokio::time::sleep(std::time::Duration::from_millis(100)).await; }
    Err(anyhow!("Slot worker channel initialization timed out"))
}
pub fn init_bake_worker() { let (tx, rx) = mpsc::channel(100); tauri::async_runtime::spawn(async move { spawn_slot_worker(rx); }); let _ = SLOT_TX.set(tx); }

fn spawn_slot_worker(mut rx: mpsc::Receiver<SlotTask>) {
    let (io_tx, mut io_rx) = mpsc::channel::<SaveTask>(1000); 
    tokio::spawn(async move {
        while let Some(task) = io_rx.recv().await {
            if let Some(parent) = task.path.parent() { if !parent.exists() { let _ = std::fs::create_dir_all(parent); } }
            let _ = candle_core::safetensors::save(&task.tensors, &task.path);
            let slot = &SLOT_MANAGER.slots[task.slot_id];
            if slot.remaining_layers.fetch_sub(1, Ordering::SeqCst) == 1 || task.is_last {
                SLOT_MANAGER.mark_ready(task.slot_id, task.total_bytes_hint).await;
            }
        }
    });
    tokio::spawn(async move {
        while let Some(task) = rx.recv().await {
            match task {
                SlotTask::Bake(bake) => {
                    let io_tx_inner = io_tx.clone();
                    tokio::task::spawn_blocking(move || {
                        let (sid, t_dir, kv_n, off, is_relay) = (bake.slot_id, bake.task_dir, bake.kv_name, bake.offset, bake.is_relay_baking);
                        if bake.layers.is_empty() { let _ = SLOT_MANAGER.request_tx.blocking_send(SlotRequest::Release { idx: sid }); return; }
                        let loop_count = if is_relay { 1 } else { bake.layers.len() };
                        let slot = &SLOT_MANAGER.slots[sid];
                        slot.remaining_layers.store(loop_count, Ordering::SeqCst);
                        let hint = bake.layers.len() * (20 * 1024 * 1024 / 28);
                        for l_idx in 0..loop_count {
                            let source = &bake.layers[l_idx];
                            let fname = if is_relay { if off == 0 { "layer_relay_kv.safetensors".into() } else { format!("layer_relay_kv_{}.safetensors", off) } } 
                            else { match (&kv_n, off) { (Some(n), 0) => format!("layer_{}_kv.safetensors", n), (Some(n), o) => format!("layer_{}_kv_{}.safetensors", n, o), (None, 0) => format!("layer_{}_kv.safetensors", l_idx), (None, o) => format!("layer_{}_kv_{}.safetensors", l_idx, o) } };
                            let mut map = std::collections::HashMap::new();
                            let (b, h, s, d) = (source.shape[0], source.shape[1], source.shape[2], source.shape[3]);
                            let a_count = (0..s).filter(|&i| i < 4 || i % 8 == 0).count();
                            let head_size = s * d;
                            let mut ka = vec![0.0f32; b * h * a_count * d]; let mut kp = vec![0u8; (b * h * s * d + 7) / 8]; let mut ks = vec![0.0f32; b * h * s];
                            for bh in 0..(b * h) {
                                let bho = bh * head_size;
                                for i in 0..head_size { if source.k_data[bho + i] >= 0.0 { kp[(bho + i) / 8] |= 1 << ((bho + i) % 8); } }
                                for ti in 0..s {
                                    let td = &source.k_data[bho + ti * d .. bho + (ti + 1) * d];
                                    if ti < 4 || ti % 8 == 0 { let ap = if ti < 4 { ti } else { 4 + (ti - 4) / 8 }; ka[(bh * a_count + ap) * d .. (bh * a_count + ap + 1) * d].copy_from_slice(td); }
                                    let mut m = 0.0f32; for &v in td { let a = v.abs(); if a > m { m = a; } } ks[bh * s + ti] = m;
                                }
                            }
                            map.insert("k_anchors".to_string(), Tensor::from_vec(ka, vec![b, h, a_count, d], &Device::Cpu).unwrap());
                            map.insert("k_packed".to_string(), Tensor::from_vec(kp, vec![(b * h * s * d + 7) / 8], &Device::Cpu).unwrap());
                            map.insert("k_scales".to_string(), Tensor::from_vec(ks, vec![b, h, s, 1], &Device::Cpu).unwrap());
                            map.insert("k_shape".to_string(), Tensor::from_vec(vec![b as u32, h as u32, s as u32, d as u32], (4,), &Device::Cpu).unwrap());
                            let mut va = vec![0.0f32; b * h * a_count * d]; let mut vp = vec![0u8; (b * h * s * d + 7) / 8]; let mut vs = vec![0.0f32; b * h * s];
                            for bh in 0..(b * h) {
                                let bho = bh * head_size;
                                for i in 0..head_size { if source.v_data[bho + i] >= 0.0 { vp[(bho + i) / 8] |= 1 << ((bho + i) % 8); } }
                                for ti in 0..s {
                                    let td = &source.v_data[bho + ti * d .. bho + (ti + 1) * d];
                                    if ti < 4 || ti % 8 == 0 { let ap = if ti < 4 { ti } else { 4 + (ti - 4) / 8 }; va[(bh * a_count + ap) * d .. (bh * a_count + ap + 1) * d].copy_from_slice(td); }
                                    let mut m = 0.0f32; for &v in td { let a = v.abs(); if a > m { m = a; } } vs[bh * s + ti] = m;
                                }
                            }
                            map.insert("v_anchors".to_string(), Tensor::from_vec(va, vec![b, h, a_count, d], &Device::Cpu).unwrap());
                            map.insert("v_packed".to_string(), Tensor::from_vec(vp, vec![(b * h * s * d + 7) / 8], &Device::Cpu).unwrap());
                            map.insert("v_scales".to_string(), Tensor::from_vec(vs, vec![b, h, s, 1], &Device::Cpu).unwrap());
                            map.insert("mode".to_string(), Tensor::from_vec(vec![3u32], (1,), &Device::Cpu).unwrap());
                            let _ = io_tx_inner.blocking_send(SaveTask { slot_id: sid, path: t_dir.join(fname), tensors: map, is_last: l_idx == loop_count - 1, total_bytes_hint: hint });
                        }
                    }).await.ok();
                }
                SlotTask::Load(load) => {
                    let (off, idx) = { let inner = load.shared_block.inner.read().unwrap(); (inner.offset, inner.index) };
                    let mut path = load.path.join(if off == 0 { format!("layer_{}_kv.safetensors", load.layer_idx) } else { format!("layer_{}_kv_{}.safetensors", load.layer_idx, off) });
                    if !path.exists() { let relay = load.path.join(if off == 0 { "layer_relay_kv.safetensors".into() } else { format!("layer_relay_kv_{}.safetensors", off) }); if relay.exists() { path = relay; } }
                    let mut success = false;
                    if let Ok(content) = std::fs::read(&path) {
                        if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                            let ex_vec = |name: &str| -> Option<Vec<f32>> { st.tensor(name).ok().map(|v| unsafe { std::slice::from_raw_parts(v.data().as_ptr() as *const f32, v.data().len() / 4).to_vec() }) };
                            if let (Some(ka), Some(kp), Some(ks), Some(va), Some(vp), Some(vs)) = (ex_vec("k_anchors"), st.tensor("k_packed").ok(), ex_vec("k_scales"), ex_vec("v_anchors"), st.tensor("v_packed").ok(), ex_vec("v_scales")) {
                                let o_shape = if let Ok(view) = st.tensor("k_shape") { let s_u32: &[u32] = unsafe { std::slice::from_raw_parts(view.data().as_ptr() as *const u32, view.data().len() / 4) }; s_u32.iter().map(|&x| x as usize).collect() } else { vec![1, 8, 1024, 128] };
                                let mut inner = load.shared_block.inner.write().unwrap();
                                inner.bitkv_metadata = Some(crate::models::qwen3vl::quantized_model::BitKVMetadata { k_anchors: Tensor::from_vec(ka, (o_shape[0], o_shape[1], (o_shape[2] + 7) / 8 + 4, o_shape[3]), &Device::Cpu).unwrap(), k_packed: Tensor::from_slice(kp.data(), kp.shape(), &Device::Cpu).unwrap(), k_scales: Tensor::from_vec(ks, (o_shape[0], o_shape[1], o_shape[2], 1), &Device::Cpu).unwrap(), v_anchors: Tensor::from_vec(va, (o_shape[0], o_shape[1], (o_shape[2] + 7) / 8 + 4, o_shape[3]), &Device::Cpu).unwrap(), v_packed: Tensor::from_slice(vp.data(), vp.shape(), &Device::Cpu).unwrap(), v_scales: Tensor::from_vec(vs, (o_shape[0], o_shape[1], o_shape[2], 1), &Device::Cpu).unwrap(), original_shape: o_shape });
                                success = true;
                            }
                        }
                    }
                    if success { { let mut reg = load.registry.entries.write().unwrap(); if idx < reg.len() { reg[idx].location[load.layer_idx] = KVLocation::RAM; reg[idx].slot_ids[load.layer_idx] = Some(load.slot_id); } } SLOT_MANAGER.mark_ready(load.slot_id, 20 * 1024 * 1024).await; } 
                    else { { let mut reg = load.registry.entries.write().unwrap(); if idx < reg.len() { reg[idx].location[load.layer_idx] = KVLocation::SSD; } } SLOT_MANAGER.release_slot(load.slot_id).await; }
                }
            }
        }
    });
}

#[derive(Clone)]
pub enum ModelVariant { Standard(crate::models::qwen3vl::model::Qwen3VLModel), QuantizedVL(QuantizedQwen3VLModel), QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel) }
impl ModelVariant {
    pub fn forward(&mut self, input_ids: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, video_pixel_values: Option<&Tensor>, video_grid_thw: Option<&Tensor>, cache_position: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>) -> Result<Tensor> {
        match self { Self::Standard(m) => m.forward(input_ids, pixel_values, image_grid_thw, video_pixel_values, video_grid_thw, cache_position, seqlen_offset), Self::QuantizedVL(m) => m.forward(input_ids, pixel_values, image_grid_thw, video_pixel_values, video_grid_thw, cache_position, seqlen_offset, total_len, session_id), Self::QuantizedText(m) => m.forward(input_ids, cache_position, seqlen_offset, total_len, session_id) }
    }
    pub fn prefetch_all_slots(&mut self, session_id: &str) -> Result<()> {
        let registry = match self { Self::QuantizedVL(m) => Some(&m.language_model.registry), Self::QuantizedText(m) => Some(&m.language_model.registry), _ => None };
        if let Some(reg_obj) = registry {
            let entries = reg_obj.entries.read().unwrap();
            for (idx, entry) in entries.iter().enumerate() {
                for layer_idx in 0..28 {
                    if entry.location[layer_idx] == KVLocation::SSD {
                        let path = crate::utils::paths::get_kv_dir(None).join(session_id);
                        let _ = reg_obj.request_load(idx, layer_idx, &path);
                    }
                }
            }
        }
        Ok(())
    }
}

pub struct Qwen3VLGenerateModel { pub chat_template: ChatTemplate, pub tokenizer: TokenizerModel, pub pre_processor: Qwen3VLProcessor, pub qwen3_vl: ModelVariant, pub text_device: Device, pub vision_device: Device, pub eos_token_id1: u32, pub eos_token_id2: u32, pub generation_config: Qwen3VLGenerationConfig, pub model_name: String, pub hard_token_limit: Option<usize>, pub kv_root: std::path::PathBuf }
impl Qwen3VLGenerateModel {
    pub fn init_with_config(path: &str, tokenizer_path: Option<&str>, config_path: Option<&str>, text_device: Option<&Device>, text_device_id: usize, vision_device: Option<&Device>, vision_device_id: usize, dtype: Option<DType>, hard_token_limit: Option<usize>, force_text_only: bool, baking_only: bool, _is_disk_swap: bool, kv_root: std::path::PathBuf) -> Result<Self> {
        let path = if let Some(stripped) = path.strip_prefix(r"\\?\") { stripped } else { path };
        let tok_path = tokenizer_path.unwrap_or(path); let tok_path = if let Some(stripped) = tok_path.strip_prefix(r"\\?\") { stripped } else { tok_path };
        let cfg_path = config_path.unwrap_or(path); let cfg_path = if let Some(stripped) = cfg_path.strip_prefix(r"\\?\") { stripped } else { cfg_path };
        let tokenizer = TokenizerModel::init(tok_path)?;
        let raw_config: serde_json::Value = serde_json::from_slice(&std::fs::read(std::path::Path::new(cfg_path).join("config.json"))?)?;
        let cfg: Qwen3VLConfig = if raw_config.get("text_config").is_some() { serde_json::from_value(raw_config)? } else {
            let text_config: crate::models::qwen3vl::config::Qwen3VLTextConfig = serde_json::from_value(raw_config.clone())?;
            crate::models::qwen3vl::config::Qwen3VLConfig { architectures: raw_config.get("architectures").and_then(|v| serde_json::from_value(v.clone()).ok()), auto_map: raw_config.get("auto_map").and_then(|v| serde_json::from_value(v.clone()).ok()), hidden_size: raw_config.get("hidden_size").and_then(|v| v.as_u64()).map(|v| v as usize), image_token_id: raw_config.get("image_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), model_type: raw_config.get("model_type").and_then(|v| v.as_str()).unwrap_or("qwen2").to_string(), text_config: Some(text_config), tie_word_embeddings: raw_config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(true), torch_dtype: raw_config.get("torch_dtype").and_then(|v| v.as_str()).map(|s| s.to_string()), transformers_version: raw_config.get("transformers_version").and_then(|v| v.as_str()).unwrap_or("").to_string(), video_token_id: raw_config.get("video_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), vision_config: None, vision_start_token_id: None, vision_end_token_id: None }
        };
        let (text_dev, vision_dev) = (get_device(text_device), get_device(vision_device));
        let dtype = get_dtype(dtype, cfg.text_config.as_ref().and_then(|tc| tc.dtype.as_deref()).unwrap_or("float16"));
        let gguf_files = find_type_files(path, "gguf")?; let mmproj = gguf_files.iter().find(|f| f.contains("mmproj")).cloned();
        let qwen3_vl = if !gguf_files.is_empty() {
            let mut m_path = gguf_files.iter().find(|f| f.contains("Qwen3-0.6B-Q8_0.gguf")).cloned(); if m_path.is_none() { m_path = gguf_files.iter().find(|f| f.contains("Qwen3-0.6B-Q4_K_M.gguf")).cloned(); } if m_path.is_none() { m_path = gguf_files.iter().find(|f| !f.contains("mmproj")).cloned(); }
            let kv_res = hard_token_limit.unwrap_or(4096) as u64 * 40000;
            if mm.is_some() && !force_text_only {
                let main_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&m_path.unwrap())?)? };
                let mmproj_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&mm.unwrap())?)? };
                ModelVariant::QuantizedVL(QuantizedQwen3VLModel::new_with_mmap(&cfg, &gguf_file::Content::read(&mut std::io::Cursor::new(&main_mmap[..]))?, Some(Arc::new(main_mmap)), &gguf_file::Content::read(&mut std::io::Cursor::new(&mmproj_mmap[..]))?, Some(Arc::new(mmproj_mmap)), &text_dev, text_device_id, &vision_dev, vision_device_id, dtype, kv_res, baking_only)?)
            } else {
                let mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&m_path.unwrap())?)? };
                ModelVariant::QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel::new_with_mmap(&cfg, &gguf_file::Content::read(&mut std::io::Cursor::new(&mmap[..]))?, Some(Arc::new(mmap)), &text_dev, text_device_id, dtype, kv_res, baking_only || path.contains("0.6B"), baking_only || path.contains("0.6B"))?)
            }
        } else {
            let model_list = find_type_files(path, "safetensors")?;
            ModelVariant::Standard(Qwen3VLModel::new(cfg, unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &text_dev)? })?)
        };
        let g_path = std::path::Path::new(cfg_path).join("generation_config.json"); let g_cfg: Qwen3VLGenerationConfig = if g_path.exists() { serde_json::from_slice(&std::fs::read(g_path)?)? } else { Qwen3VLGenerationConfig::default() };
        let (eos1, eos2) = match &g_cfg.eos_token_id { serde_json::Value::Number(n) => { let id = n.as_u64().unwrap_or(151645) as u32; (id, id) }, serde_json::Value::Array(arr) => { let id1 = arr.get(0).and_then(|v| v.as_u64()).unwrap_or(151643) as u32; let id2 = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(id1 as u64) as u32; (id1, id2) }, _ => (151643, 151643) };
        Ok(Self { chat_template: ChatTemplate::init(tok_path)?, tokenizer, pre_processor: Qwen3VLProcessor::new(tok_path, &vision_dev, dtype)?, qwen3_vl, text_device: text_dev, vision_device: vision_dev, eos_token_id1: eos1, eos_token_id2: eos2, generation_config: g_cfg, model_name: if path.contains("0.6B") { "qwen3vl-0.6B".into() } else { "qwen3vl-2B".into() }, hard_token_limit, kv_root })
    }

    pub async fn prefill_only(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _relay_target: Option<&mut Qwen3VLGenerateModel>, kv_name: Option<String>) -> Result<usize> {
        if session_id.is_none() { SLOT_MANAGER.reset_all_slots().await; }
        let full_ids = self.tokenizer.text_encode_vec(self.pre_processor.process_info(&mes, &self.chat_template.apply_chat_template(&mes)?)?.replace_text, false)?;
        let (t_toks, c_size, is_06b) = (full_ids.len(), 2048, self.model_name.contains("0.6B"));
        let mut c_pos = self.get_kv_len();
        while c_pos < t_toks {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            let end = (c_pos + c_size).min(t_toks); let c_len = end - c_pos;
            println!("[BAKING] {} to {} / Total: {}", c_pos, end, t_toks);
            self.qwen3_vl.forward(&Tensor::from_vec(full_ids[c_pos..end].to_vec(), (1, c_len), &self.text_device)?, None, None, None, None, Some(&Tensor::arange(c_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?), c_pos, t_toks, session_id.clone())?;
            if let Some(sid) = &session_id {
                let slot_id = SLOT_MANAGER.acquire_write_slot(t_toks).await;
                let path = crate::utils::paths::get_kv_dir(None).join(sid); if !path.exists() { let _ = fs::create_dir_all(&path); }
                let (ks, vs) = self.get_current_kv();
                if !ks.is_empty() {
                    let mut dumps = Vec::new();
                    for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() {
                        let s_len = k.dim(2)?;
                        // [FIX] contiguous()를 호출하여 메모리 불연속성으로 인한 Panic 방지
                        let k_vec = k.narrow(2, s_len.saturating_sub(c_len), c_len)?.contiguous()?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        let v_vec = v.narrow(2, s_len.saturating_sub(c_len), c_len)?.contiguous()?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        dumps.push(LayerKVDump { layer_idx: idx, k_data: k_vec, v_data: v_vec, shape: vec![1, 8, c_len, 128] });
                    }
                    if let Some(reg_obj) = match self.qwen3_vl { ModelVariant::QuantizedText(ref m) => Some(&m.language_model.registry), ModelVariant::QuantizedVL(ref m) => Some(&m.language_model.registry), _ => None } {
                        let mut reg = reg_obj.entries.write().unwrap();
                        for entry in reg.iter_mut() { for l in 0..28 { if entry.location[l] == KVLocation::VRAM { entry.location[l] = KVLocation::SSD; entry.ssd_path = Some(path.clone()); } } }
                    }
                    self.clear_temporal_kv_caches(); let dev = self.text_device.clone(); if dev.is_cuda() { let _ = dev.synchronize(); }
                    if let Ok(tx) = get_worker_channel().await { let _ = tx.send(SlotTask::Bake(BakeTask { slot_id, task_dir: path, kv_name: kv_name.clone(), offset: end, layers: dumps, is_relay_baking: is_06b })).await; }
                } else { SLOT_MANAGER.release_slot(slot_id).await; }
                self.clear_temporal_kv_caches();
            }
            c_pos = end;
        }
        Ok(c_pos)
    }

    pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, kv_name: Option<String>) -> Result<String> {
        if session_id.is_some() { let _ = self.qwen3_vl.prefetch_all_slots(session_id.as_ref().unwrap()); }
        let mut l_proc = get_logit_processor(Some(mes.temperature.unwrap_or(0.7) as f32), Some(mes.top_p.unwrap_or(0.9) as f32), Some(40), mes.seed.unwrap_or(34562) as u64);
        let mut input = self.pre_processor.process_info(&mes, &self.chat_template.apply_chat_template(&mes)?)?;
        let full_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?; let mut all_ids = full_ids.clone();
        let (t_toks, mut s_off, c_size) = (full_ids.len(), self.get_kv_len(), 128);
        let mut l_pos = s_off;
        while l_pos < t_toks {
            let chunk_size = (t_toks - l_pos).min(c_size); if chunk_size == 0 { break; }
            self.qwen3_vl.forward(&Tensor::from_vec(full_ids[l_pos..l_pos+chunk_size].to_vec(), (1, chunk_size), &self.text_device)?, None, None, None, None, Some(&Tensor::arange(s_off as u32, (s_off + chunk_size) as u32, &self.text_device)?.unsqueeze(0)?), s_off, t_toks, session_id.clone())?;
            l_pos += chunk_size; s_off += chunk_size;
            if let Some(sid) = &session_id {
                let slot_id = SLOT_MANAGER.acquire_write_slot(t_toks).await;
                let (ks, vs) = self.get_current_kv();
                if !ks.is_empty() {
                    let mut dumps = Vec::new();
                    for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() {
                        let k_v = k.contiguous()?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        let v_v = v.contiguous()?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        dumps.push(LayerKVDump { layer_idx: idx, k_data: k_v, v_data: v_v, shape: vec![1, 8, chunk_size, 128] });
                    }
                    if let Some(reg_obj) = match self.qwen3_vl { ModelVariant::QuantizedText(ref m) => Some(&m.language_model.registry), ModelVariant::QuantizedVL(ref m) => Some(&m.language_model.registry), _ => None } {
                        let mut reg = reg_obj.entries.write().unwrap();
                        for entry in reg.iter_mut() { for l in 0..28 { if entry.location[l] == KVLocation::VRAM { entry.location[l] = KVLocation::SSD; entry.ssd_path = Some(crate::utils::paths::get_kv_dir(None).join(sid)); } } }
                    }
                    if let Ok(tx) = get_worker_channel().await { let _ = tx.send(SlotTask::Bake(BakeTask { slot_id, task_dir: crate::utils::paths::get_kv_dir(None).join(sid), kv_name: kv_name.clone(), offset: s_off, layers: dumps, is_relay_baking: false })).await; }
                } else { SLOT_MANAGER.release_slot(slot_id).await; }
            }
            self.clear_temporal_kv_caches();
        }
        let (mut p_vals, i_grid, mut gen_text) = (input.pixel_values.take(), input.image_grid_thw.take(), String::new());
        for _ in 0..mes.max_tokens.unwrap_or(2048) {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            let logits = self.qwen3_vl.forward(&Tensor::new(vec![*all_ids.last().unwrap()], &self.text_device)?.unsqueeze(0)?, p_vals.as_ref(), i_grid.as_ref(), None, None, Some(&Tensor::arange(s_off as u32, (s_off + 1) as u32, &self.text_device)?.unsqueeze(0)?), s_off, t_toks, session_id.clone())?;
            let mut l_vec = logits.squeeze(0)?.i(logits.dim(1)? - 1)?.to_dtype(DType::F32)?;
            l_vec = apply_repeat_penalty(&l_vec, 1.1, if all_ids.len() > 512 { &all_ids[all_ids.len()-512..] } else { &all_ids[..] })?;
            let next_id = l_proc.sample(&l_vec)?; if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            all_ids.push(next_id); gen_text.push_str(&self.tokenizer.token_decode(vec![next_id])?);
            s_off += 1; p_vals = None; self.clear_temporal_kv_caches();
        }
        Ok(gen_text)
    }

    pub fn clear_temporal_kv_caches(&mut self) {
        let registry = match self.qwen3_vl { ModelVariant::QuantizedText(ref m) => Some(&m.language_model.registry), ModelVariant::QuantizedVL(ref m) => Some(&m.language_model.registry), _ => None };
        if let Some(reg_obj) = registry {
            let reg = reg_obj.entries.read().unwrap();
            let layers = match self.qwen3_vl { ModelVariant::QuantizedText(ref mut m) => &mut m.language_model.layers, ModelVariant::QuantizedVL(ref mut m) => &mut m.language_model.layers, _ => unreachable!() };
            for (l_idx, layer) in layers.iter_mut().enumerate() {
                for block in &mut layer.self_attn.kv_blocks {
                    let mut inner = block.inner.write().unwrap();
                    if inner.location == KVLocation::VRAM {
                        let reg_loc = if inner.index < reg.len() { reg[inner.index].location[l_idx] } else { KVLocation::VRAM };
                        if reg_loc != KVLocation::VRAM { inner.k_cache = None; inner.v_cache = None; inner.location = reg_loc; }
                    }
                }
            }
        }
    }

    pub fn get_kv_len(&self) -> usize { match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.get_kv_len(), ModelVariant::QuantizedText(m) => m.language_model.get_kv_len(), _ => 0 } }
    pub fn get_current_kv(&self) -> (Vec<Tensor>, Vec<Tensor>) {
        let mut ks = vec![]; let mut vs = vec![];
        let layers = match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => Some(&m.language_model.layers), ModelVariant::QuantizedText(m) => Some(&m.language_model.layers), _ => None };
        if let Some(layers) = layers {
            for l in layers {
                let (mut lk, mut lv) = (vec![], vec![]);
                for b in &l.self_attn.kv_blocks { let inner = b.inner.read().unwrap(); if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) { lk.push(k.clone()); lv.push(v.clone()); } }
                if !lk.is_empty() { if let (Ok(k), Ok(v)) = (Tensor::cat(&lk, 2), Tensor::cat(&lv, 2)) { ks.push(k); vs.push(v); } }
            }
        }
        (ks, vs)
    }
    pub fn save_kv_to_disk(&mut self, path: &Path, kv_name: Option<&str>, offset: usize) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.save_kv_cache(path, false, offset, kv_name), ModelVariant::QuantizedText(m) => m.save_kv_cache(path, false, offset, kv_name), _ => Ok(()) } }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.to_device(d)?, ModelVariant::QuantizedText(m) => m.to_device(d)?, _ => {} } self.text_device = d.clone(); self.vision_device = d.clone(); Ok(()) }
    pub fn drop_kv_storage(&mut self) -> Result<()> { self.qwen3_vl.drop_kv_storage() }
}

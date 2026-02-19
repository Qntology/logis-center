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
    Release { idx: usize, success: bool }, 
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
    pub is_flushing: Arc<AtomicBool>,
}

impl SlotManager {
    pub fn new(count: usize) -> Self {
        let mut slots = Vec::new();
        for i in 0..count { slots.push(crate::models::qwen3vl::quantized_model::MemorySlot::new(i, 28)); }
        let (tx, rx) = mpsc::channel(1000); 
        let cr = Arc::new(AtomicUsize::new(0));
        let cw = Arc::new(AtomicUsize::new(0));
        let cc = Arc::new(AtomicUsize::new(0));
        let cf = Arc::new(AtomicUsize::new(count));
        let ff = Arc::new(AtomicBool::new(false));
        let (cr_c, cw_c, cc_c, cf_c, ff_c) = (cr.clone(), cw.clone(), cc.clone(), cf.clone(), ff.clone());
        tauri::async_runtime::spawn(async move { Self::slot_dispatcher(rx, cr_c, cw_c, cc_c, cf_c, ff_c, count).await; });
        Self { slots, request_tx: tx, count_reads: cr, count_writes: cw, count_cached: cc, count_free: cf, is_flushing: ff }
    }

    async fn slot_dispatcher(mut rx: mpsc::Receiver<SlotRequest>, cr: Arc<AtomicUsize>, cw: Arc<AtomicUsize>, cc: Arc<AtomicUsize>, cf: Arc<AtomicUsize>, is_flushing: Arc<AtomicBool>, max_slots: usize) {
        let mut states = vec![InternalState::Free; max_slots];
        let mut free_p: VecDeque<usize> = (0..max_slots).collect();
        let mut ready_p: VecDeque<usize> = VecDeque::new();
        let mut p_writes: VecDeque<(oneshot::Sender<usize>, usize)> = VecDeque::new();
        let mut p_reads: VecDeque<oneshot::Sender<usize>> = VecDeque::new();
        let mut flushers: Vec<oneshot::Sender<()>> = Vec::new();
        let mut ram_bytes = 0usize;
        let mut sys = sysinfo::System::new_all();

        println!("[SLOT-DISPATCHER] High-Speed Sticky Cache Mode 가동.");

        while let Some(req) = rx.recv().await {
            match req {
                SlotRequest::AcquireWrite { response, total_tokens } => {
                    if is_flushing.load(Ordering::SeqCst) { continue; }
                    sys.refresh_memory();
                    let max_c = Self::calculate_dynamic_budget(&sys, total_tokens, max_slots);
                    
                    // 1순위: Free 슬롯 사용
                    if !free_p.is_empty() && cw.load(Ordering::SeqCst) < max_c {
                        let idx = free_p.pop_front().unwrap();
                        states[idx] = InternalState::Writing;
                        cf.fetch_sub(1, Ordering::SeqCst); cw.fetch_add(1, Ordering::SeqCst);
                        let _ = response.send(idx);
                    } 
                    // 2순위: Ready 슬롯(캐시) 강제 회수하여 사용
                    else if !ready_p.is_empty() && cw.load(Ordering::SeqCst) < max_c {
                        let idx = ready_p.pop_front().unwrap();
                        states[idx] = InternalState::Writing;
                        cc.fetch_sub(1, Ordering::SeqCst); cw.fetch_add(1, Ordering::SeqCst);
                        let _ = response.send(idx);
                    }
                    else { p_writes.push_back((response, total_tokens)); }
                }
                SlotRequest::AcquireRead { response } => {
                    if is_flushing.load(Ordering::SeqCst) { continue; }
                    if !free_p.is_empty() {
                        let idx = free_p.pop_front().unwrap();
                        states[idx] = InternalState::Reading;
                        cf.fetch_sub(1, Ordering::SeqCst); cr.fetch_add(1, Ordering::SeqCst);
                        let _ = response.send(idx);
                    } else if !ready_p.is_empty() {
                        let idx = ready_p.pop_front().unwrap();
                        states[idx] = InternalState::Reading;
                        cc.fetch_sub(1, Ordering::SeqCst); cr.fetch_add(1, Ordering::SeqCst);
                        let _ = response.send(idx);
                    } else { p_reads.push_back(response); }
                }
                SlotRequest::MarkReady { idx, bytes } => {
                    if idx < max_slots {
                        let old = states[idx];
                        if old == InternalState::Writing || old == InternalState::Reading {
                            states[idx] = InternalState::Ready;
                            ready_p.push_back(idx);
                            cc.fetch_add(1, Ordering::SeqCst);
                            ram_bytes += bytes;
                            if old == InternalState::Writing { cw.fetch_sub(1, Ordering::SeqCst); }
                            else { cr.fetch_sub(1, Ordering::SeqCst); }
                            
                            // 100MB 예산 초과 시 오래된 캐시부터 완전 삭제
                            while ram_bytes > 100 * 1024 * 1024 && !ready_p.is_empty() {
                                let ev = ready_p.pop_front().unwrap();
                                if states[ev] == InternalState::Ready {
                                    states[ev] = InternalState::Free;
                                    free_p.push_back(ev);
                                    cf.fetch_add(1, Ordering::SeqCst);
                                    cc.fetch_sub(1, Ordering::SeqCst);
                                    ram_bytes = ram_bytes.saturating_sub(2 * 1024 * 1024); // 근사치
                                }
                            }
                        }
                    }
                    if cw.load(Ordering::SeqCst) == 0 && cr.load(Ordering::SeqCst) == 0 { while let Some(w) = flushers.pop() { let _ = w.send(()); } }
                    Self::process_queues_robust(&sys, &mut free_p, &mut ready_p, &mut p_writes, &mut p_reads, &mut states, &cf, &cc, &cw, &cr, max_slots);
                }
                SlotRequest::Release { idx, success: _ } => {
                    if idx < max_slots && states[idx] != InternalState::Free {
                        let old = states[idx];
                        states[idx] = InternalState::Free;
                        free_p.push_back(idx);
                        cf.fetch_add(1, Ordering::SeqCst);
                        if old == InternalState::Writing { cw.fetch_sub(1, Ordering::SeqCst); }
                        else if old == InternalState::Ready { cc.fetch_sub(1, Ordering::SeqCst); }
                        else { cr.fetch_sub(1, Ordering::SeqCst); }
                    }
                    if cw.load(Ordering::SeqCst) == 0 && cr.load(Ordering::SeqCst) == 0 { while let Some(w) = flushers.pop() { let _ = w.send(()); } }
                    Self::process_queues_robust(&sys, &mut free_p, &mut ready_p, &mut p_writes, &mut p_reads, &mut states, &cf, &cc, &cw, &cr, max_slots);
                }
                SlotRequest::Flush { response } => {
                    is_flushing.store(true, Ordering::SeqCst);
                    if cw.load(Ordering::SeqCst) == 0 && cr.load(Ordering::SeqCst) == 0 { let _ = response.send(()); } 
                    else { flushers.push(response); }
                }
            }
        }
    }

    fn process_queues_robust(sys: &sysinfo::System, free_p: &mut VecDeque<usize>, ready_p: &mut VecDeque<usize>, p_writes: &mut VecDeque<(oneshot::Sender<usize>, usize)>, p_reads: &mut VecDeque<oneshot::Sender<usize>>, states: &mut [InternalState], cf: &Arc<AtomicUsize>, cc: &Arc<AtomicUsize>, cw: &Arc<AtomicUsize>, cr: &Arc<AtomicUsize>, max_slots: usize) {
        while (!free_p.is_empty() || !ready_p.is_empty()) && !p_writes.is_empty() {
            let max_c = Self::calculate_dynamic_budget(sys, p_writes.front().unwrap().1, max_slots);
            if cw.load(Ordering::SeqCst) < max_c {
                let idx = if !free_p.is_empty() { free_p.pop_front().unwrap() } else { cc.fetch_sub(1, Ordering::SeqCst); ready_p.pop_front().unwrap() };
                let (res, _) = p_writes.pop_front().unwrap();
                states[idx] = InternalState::Writing;
                if idx < max_slots { cw.fetch_add(1, Ordering::SeqCst); } // Only if idx was Free
                let _ = res.send(idx);
            } else { break; }
        }
        while (!free_p.is_empty() || !ready_p.is_empty()) && !p_reads.is_empty() {
            let idx = if !free_p.is_empty() { free_p.pop_front().unwrap() } else { cc.fetch_sub(1, Ordering::SeqCst); ready_p.pop_front().unwrap() };
            let res = p_reads.pop_front().unwrap();
            states[idx] = InternalState::Reading;
            cr.fetch_add(1, Ordering::SeqCst);
            let _ = res.send(idx);
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
        self.is_flushing.store(false, Ordering::SeqCst);
    }

    pub async fn reset_all_slots(&self) { for i in 0..self.slots.len() { self.release_slot(i, true).await; } }
    pub async fn acquire_read_slot(&self) -> usize { let (tx, rx) = oneshot::channel(); let _ = self.request_tx.send(SlotRequest::AcquireRead { response: tx }).await; rx.await.unwrap_or(0) }
    pub async fn acquire_write_slot(&self, total_tokens: usize) -> usize { let (tx, rx) = oneshot::channel(); let _ = self.request_tx.send(SlotRequest::AcquireWrite { response: tx, total_tokens }).await; rx.await.unwrap_or(0) }
    
    pub async fn release_slot(&self, id: usize, success: bool) {
        if id < self.slots.len() {
            for l in &self.slots[id].k_layers { if let Ok(mut g) = l.try_lock() { *g = None; } }
            for l in &self.slots[id].v_layers { if let Ok(mut g) = l.try_lock() { *g = None; } }
            self.slots[id].state.store(0, Ordering::SeqCst);
            let _ = self.request_tx.send(SlotRequest::Release { idx: id, success }).await;
        }
    }
    pub async fn mark_ready(&self, id: usize, bytes: usize) {
        if id < self.slots.len() {
            self.slots[id].state.store(2, Ordering::SeqCst);
            let _ = self.request_tx.send(SlotRequest::MarkReady { idx: id, bytes }).await;
        }
    }
    pub fn get_counts(&self) -> (usize, usize, usize, usize) { (self.count_reads.load(Ordering::Relaxed), self.count_writes.load(Ordering::Relaxed), self.count_cached.load(Ordering::Relaxed), self.count_free.load(Ordering::Relaxed)) }
}

pub static SLOT_MANAGER: once_cell::sync::Lazy<SlotManager> = once_cell::sync::Lazy::new(|| SlotManager::new(65));

struct LayerKVDump { layer_idx: usize, k_data: Vec<f32>, v_data: Vec<f32>, shape: Vec<usize> }
struct BakeTask { slot_id: usize, task_dir: PathBuf, kv_name: Option<String>, offset: usize, layers: Vec<LayerKVDump>, is_relay_baking: bool }
struct SaveTask { slot_id: usize, path: PathBuf, tensors: std::collections::HashMap<String, Tensor>, is_last: bool }
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
            let res = candle_core::safetensors::save(&task.tensors, &task.path);
            let slot = &SLOT_MANAGER.slots[task.slot_id];
            if slot.remaining_layers.fetch_sub(1, Ordering::SeqCst) == 1 || task.is_last {
                // 쓰기 완료 후 'Ready'로 표시하여 재사용 가능하게 함
                SLOT_MANAGER.mark_ready(task.slot_id, 2 * 1024 * 1024).await;
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
                        if bake.layers.is_empty() { let _ = SLOT_MANAGER.request_tx.blocking_send(SlotRequest::Release { idx: sid, success: false }); return; }
                        let loop_count = if is_relay { 1 } else { bake.layers.len() };
                        let slot = &SLOT_MANAGER.slots[sid];
                        slot.remaining_layers.store(loop_count, Ordering::SeqCst);
                        for l_idx in 0..loop_count {
                            let source = &bake.layers[l_idx];
                            let fname = if is_relay { if off == 0 { "layer_relay_kv.safetensors".to_string() } else { format!("layer_relay_kv_{}.safetensors", off) } }
                            else { match (&kv_n, off) { (Some(n), 0) => format!("layer_{}_kv.safetensors", n), (Some(n), o) => format!("layer_{}_kv_{}.safetensors", n, o), (None, 0) => format!("layer_{}_kv.safetensors", source.layer_idx), (None, o) => format!("layer_{}_kv_{}.safetensors", source.layer_idx, o) } };
                            let mut map = std::collections::HashMap::new();
                            let (b, h, s, d) = (source.shape[0], source.shape[1], source.shape[2], source.shape[3]);
                            let head_size = s * d;
                            let a_count = (0..s).filter(|&i| i < 4 || i % 8 == 0).count();
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
                            let _ = io_tx_inner.blocking_send(SaveTask { slot_id: sid, path: t_dir.join(fname), tensors: map, is_last: l_idx == loop_count - 1 });
                        }
                    }).await.ok();
                }
                SlotTask::Load(load) => {
                    let (off, idx) = { let inner = load.shared_block.inner.read().unwrap(); (inner.offset, inner.index) };
                    let fname = if off == 0 { format!("layer_{}_kv.safetensors", load.layer_idx) } else { format!("layer_{}_kv_{}.safetensors", load.layer_idx, off) };
                    let r_name = if off == 0 { "layer_relay_kv.safetensors".to_string() } else { format!("layer_relay_kv_{}.safetensors", off) };
                    let mut path = load.path.join(fname);
                    if !path.exists() { let r_path = load.path.join(r_name); if r_path.exists() { path = r_path; } }
                    let mut success = false;
                    if let Ok(content) = std::fs::read(&path) {
                        if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                            let ex_v = |name: &str| -> Option<Vec<f32>> { st.tensor(name).ok().map(|v| unsafe { std::slice::from_raw_parts(v.data().as_ptr() as *const f32, v.data().len() / 4).to_vec() }) };
                            if let (Some(ka), Some(kp), Some(ks), Some(va), Some(vp), Some(vs)) = (ex_v("k_anchors"), st.tensor("k_packed").ok(), ex_v("k_scales"), ex_v("v_anchors"), st.tensor("v_packed").ok(), ex_v("v_scales")) {
                                let o_s = if let Ok(view) = st.tensor("k_shape") { let s_u32: &[u32] = unsafe { std::slice::from_raw_parts(view.data().as_ptr() as *const u32, view.data().len() / 4) }; s_u32.iter().map(|&x| x as usize).collect() } else { vec![1, 8, 128, 128] };
                                let mut inner = load.shared_block.inner.write().unwrap();
                                inner.bitkv_metadata = Some(crate::models::qwen3vl::quantized_model::BitKVMetadata { k_anchors: Tensor::from_vec(ka, (o_s[0], o_s[1], (o_s[2] + 7) / 8 + 4, o_s[3]), &Device::Cpu).unwrap(), k_packed: Tensor::from_slice(kp.data(), kp.shape(), &Device::Cpu).unwrap(), k_scales: Tensor::from_vec(ks, (o_s[0], o_s[1], o_s[2], 1), &Device::Cpu).unwrap(), v_anchors: Tensor::from_vec(va, (o_s[0], o_s[1], (o_s[2] + 7) / 8 + 4, o_s[3]), &Device::Cpu).unwrap(), v_packed: Tensor::from_slice(vp.data(), vp.shape(), &Device::Cpu).unwrap(), v_scales: Tensor::from_vec(vs, (o_s[0], o_s[1], o_s[2], 1), &Device::Cpu).unwrap(), original_shape: o_s });
                                success = true;
                            }
                        }
                    }
                    if success { 
                        { let mut reg = load.registry.entries.write().unwrap(); if idx < reg.len() { reg[idx].location[load.layer_idx] = KVLocation::RAM; reg[idx].slot_ids[load.layer_idx] = Some(load.slot_id); } } 
                        SLOT_MANAGER.mark_ready(load.slot_id, 2 * 1024 * 1024).await;
                    } 
                    else { 
                        { let mut reg = load.registry.entries.write().unwrap(); if idx < reg.len() { reg[idx].location[load.layer_idx] = KVLocation::SSD; } } 
                        SLOT_MANAGER.release_slot(load.slot_id, false).await;
                    }
                }
            }
        }
    });
}

#[derive(Clone)]
pub enum ModelVariant { Standard(crate::models::qwen3vl::model::Qwen3VLModel), QuantizedVL(QuantizedQwen3VLModel), QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel) }
impl ModelVariant {
    pub fn forward(&mut self, i: &Tensor, pv: Option<&Tensor>, ithw: Option<&Tensor>, vpv: Option<&Tensor>, vthw: Option<&Tensor>, cp: Option<&Tensor>, off: usize, tl: usize, sid: Option<String>) -> Result<Tensor> {
        match self { Self::Standard(m) => m.forward(i, pv, ithw, vpv, vthw, cp, off), Self::QuantizedVL(m) => m.forward(i, pv, ithw, vpv, vthw, cp, off, tl, sid), Self::QuantizedText(m) => m.forward(i, cp, off, tl, sid) }
    }
    pub fn rebalance_layers(&mut self, d: usize) -> Result<()> { match self { Self::Standard(_) => Ok(()), Self::QuantizedVL(m) => m.rebalance_layers(d), Self::QuantizedText(m) => m.rebalance_layers(d) } }
    pub fn drop_kv_storage(&mut self) -> Result<()> { match self { Self::Standard(_) => Ok(()), Self::QuantizedVL(m) => m.language_model.drop_kv_storage(), Self::QuantizedText(m) => m.language_model.drop_kv_storage() } }
    pub fn inject_kv_bitkv(&mut self, ka: &[Tensor], kp: &[Tensor], ks: &[Tensor], va: &[Tensor], vp: &[Tensor], vs: &[Tensor], os: &[usize]) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.inject_live_kv_bitkv(ka, kp, ks, va, vp, vs, os), Self::QuantizedText(m) => m.language_model.inject_live_kv_bitkv(ka, kp, ks, va, vp, vs, os), _ => Ok(()) } }
    pub fn is_cpu(&self) -> bool { match self { Self::Standard(m) => m.device().is_cpu(), Self::QuantizedVL(m) => m.language_model.is_forced_cpu, Self::QuantizedText(m) => m.language_model.is_forced_cpu } }
}

pub struct Qwen3VLGenerateModel { pub chat_template: ChatTemplate, pub tokenizer: TokenizerModel, pub pre_processor: Qwen3VLProcessor, pub qwen3_vl: ModelVariant, pub text_device: Device, pub vision_device: Device, pub eos_token_id1: u32, pub eos_token_id2: u32, pub generation_config: Qwen3VLGenerationConfig, pub model_name: String, pub hard_token_limit: Option<usize>, pub kv_root: std::path::PathBuf }
impl Qwen3VLGenerateModel {
    pub fn init_with_config(path: &str, tokenizer_path: Option<&str>, config_path: Option<&str>, text_device: Option<&Device>, text_device_id: usize, vision_device: Option<&Device>, vision_device_id: usize, dtype: Option<DType>, hard_token_limit: Option<usize>, force_text_only: bool, baking_only: bool, _is_disk_swap: bool, kv_root: std::path::PathBuf) -> Result<Self> {
        let path = if let Some(s) = path.strip_prefix(r"\\?\") { s } else { path };
        let tok_p = if let Some(s) = tokenizer_path.unwrap_or(path).strip_prefix(r"\\?\") { s } else { tokenizer_path.unwrap_or(path) };
        let cfg_p = if let Some(s) = config_path.unwrap_or(path).strip_prefix(r"\\?\") { s } else { config_path.unwrap_or(path) };
        let tokenizer = TokenizerModel::init(tok_p)?;
        let raw_c: serde_json::Value = serde_json::from_slice(&std::fs::read(std::path::Path::new(cfg_p).join("config.json"))?)?;
        let cfg: Qwen3VLConfig = if raw_c.get("text_config").is_some() { serde_json::from_value(raw_c)? } else {
            let text_config: crate::models::qwen3vl::config::Qwen3VLTextConfig = serde_json::from_value(raw_c.clone())?;
            crate::models::qwen3vl::config::Qwen3VLConfig { architectures: raw_c.get("architectures").and_then(|v| serde_json::from_value(v.clone()).ok()), auto_map: raw_c.get("auto_map").and_then(|v| serde_json::from_value(v.clone()).ok()), hidden_size: raw_c.get("hidden_size").and_then(|v| v.as_u64()).map(|v| v as usize), image_token_id: raw_c.get("image_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), model_type: raw_c.get("model_type").and_then(|v| v.as_str()).unwrap_or("qwen2").to_string(), text_config: Some(text_config), tie_word_embeddings: raw_c.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(true), torch_dtype: raw_c.get("torch_dtype").and_then(|v| v.as_str()).map(|s| s.to_string()), transformers_version: raw_c.get("transformers_version").and_then(|v| v.as_str()).unwrap_or("").to_string(), video_token_id: raw_c.get("video_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), vision_config: None, vision_start_token_id: None, vision_end_token_id: None }
        };
        let (t_dev, v_dev) = (get_device(text_device), get_device(vision_device));
        let dtype = get_dtype(dtype, cfg.text_config.as_ref().and_then(|tc| tc.dtype.as_deref()).unwrap_or("float16"));
        let gguf_f = find_type_files(path, "gguf")?; let mmproj_p = gguf_f.iter().find(|f| f.contains("mmproj")).cloned();
        let qwen3_vl = if !gguf_f.is_empty() {
            let mut m_p = gguf_f.iter().find(|f| f.contains("Qwen3-0.6B-Q8_0.gguf")).cloned(); if m_p.is_none() { m_p = gguf_f.iter().find(|f| f.contains("Qwen3-0.6B-Q4_K_M.gguf")).cloned(); } if m_p.is_none() { m_p = gguf_f.iter().find(|f| !f.contains("mmproj")).cloned(); }
            let kv_res = hard_token_limit.unwrap_or(4096) as u64 * 40000;
            if mmproj_p.is_some() && !force_text_only {
                let m_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&m_p.unwrap())?)? };
                let mm_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&mmproj_p.unwrap())?)? };
                ModelVariant::QuantizedVL(QuantizedQwen3VLModel::new_with_mmap(&cfg, &gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?, Some(Arc::new(m_mmap)), &gguf_file::Content::read(&mut std::io::Cursor::new(&mm_mmap[..]))?, Some(Arc::new(mm_mmap)), &t_dev, text_device_id, &v_dev, vision_device_id, dtype, kv_res, baking_only)?)
            } else {
                let m_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&m_p.unwrap())?)? };
                ModelVariant::QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel::new_with_mmap(&cfg, &gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?, Some(Arc::new(m_mmap)), &t_dev, text_device_id, dtype, kv_res, baking_only || path.contains("0.6B"), baking_only || path.contains("0.6B"))?)
            }
        } else {
            ModelVariant::Standard(Qwen3VLModel::new(cfg, unsafe { VarBuilder::from_mmaped_safetensors(&find_type_files(path, "safetensors")?, dtype, &t_dev)? })?)
        };
        let g_p = std::path::Path::new(cfg_p).join("generation_config.json"); let g_cfg: Qwen3VLGenerationConfig = if g_p.exists() { serde_json::from_slice(&std::fs::read(g_p)?)? } else { Qwen3VLGenerationConfig::default() };
        let (e1, e2) = match &g_cfg.eos_token_id { serde_json::Value::Number(n) => { let id = n.as_u64().unwrap_or(151645) as u32; (id, id) }, serde_json::Value::Array(arr) => { let id1 = arr.get(0).and_then(|v| v.as_u64()).unwrap_or(151643) as u32; let id2 = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(id1 as u64) as u32; (id1, id2) }, _ => (151643, 151643) };
        Ok(Self { chat_template: ChatTemplate::init(tok_p)?, tokenizer, pre_processor: Qwen3VLProcessor::new(tok_p, &v_dev, dtype)?, qwen3_vl, text_device: t_dev, vision_device: v_dev, eos_token_id1: e1, eos_token_id2: e2, generation_config: g_cfg, model_name: if path.contains("0.6B") { "0.6B".into() } else { "2B".into() }, hard_token_limit, kv_root })
    }

    pub async fn prefill_chunk(&mut self, text: String, _cancel: Option<Arc<AtomicBool>>, _relay: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let ids = self.tokenizer.text_encode_vec(text, false)?; let size = ids.len(); let pos = self.get_kv_len();
        self.qwen3_vl.forward(&Tensor::from_vec(ids, (1, size), &self.text_device)?, None, None, None, None, Some(&Tensor::arange(pos as u32, (pos + size) as u32, &self.text_device)?.unsqueeze(0)?), pos, size, None)?;
        Ok(size)
    }

    pub async fn prefill_only(&mut self, mes: ChatCompletionParameters, cancel: Option<Arc<AtomicBool>>, sid: Option<String>, _relay: Option<&mut Qwen3VLGenerateModel>, kv_n: Option<String>) -> Result<usize> {
        if sid.is_none() { SLOT_MANAGER.reset_all_slots().await; }
        let f_ids = self.tokenizer.text_encode_vec(self.pre_processor.process_info(&mes, &self.chat_template.apply_chat_template(&mes)?)?.replace_text, false)?;
        let (t_toks, c_size, is_06b) = (f_ids.len(), 128, self.model_name.contains("0.6B"));
        let mut c_pos = self.get_kv_len();
        while c_pos < t_toks {
            if let Some(flag) = &cancel { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            let end = (c_pos + c_size).min(t_toks); let c_len = end - c_pos;
            println!("[BAKING] {} to {} / Total: {}", c_pos, end, t_toks);
            self.qwen3_vl.forward(&Tensor::from_vec(f_ids[c_pos..end].to_vec(), (1, c_len), &self.text_device)?, None, None, None, None, Some(&Tensor::arange(c_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?), c_pos, t_toks, sid.clone())?;
            if let Some(s_id) = &sid {
                let slot_id = SLOT_MANAGER.acquire_write_slot(t_toks).await;
                let path = crate::utils::paths::get_kv_dir(None).join(s_id); if !path.exists() { let _ = fs::create_dir_all(&path); }
                let (ks, vs) = self.get_current_kv();
                if !ks.is_empty() {
                    let mut dumps = Vec::new();
                    for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() {
                        let dims = k.dims(); let heads = dims[1]; let h_dim = dims[3]; let s_len = dims[2];
                        let k_v = k.narrow(2, s_len.saturating_sub(c_len), c_len)?.contiguous()?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        let v_v = v.narrow(2, s_len.saturating_sub(c_len), c_len)?.contiguous()?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        dumps.push(LayerKVDump { layer_idx: idx, k_data: k_v, v_data: v_v, shape: vec![1, heads, c_len, h_dim] });
                    }
                    self.clear_temporal_kv_caches(); let dev = self.text_device.clone(); if dev.is_cuda() { let _ = dev.synchronize(); }
                    if let Ok(tx) = get_worker_channel().await { let _ = tx.send(SlotTask::Bake(BakeTask { slot_id, task_dir: path, kv_name: kv_n.clone(), offset: end, layers: dumps, is_relay_baking: is_06b })).await; }
                } else { SLOT_MANAGER.release_slot(slot_id, false).await; }
            }
            c_pos = end;
        }
        Ok(c_pos)
    }

    pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel: Option<Arc<AtomicBool>>, sid: Option<String>, kv_n: Option<String>) -> Result<String> {
        if sid.is_none() { SLOT_MANAGER.reset_all_slots().await; }
        let mut l_proc = get_logit_processor(Some(mes.temperature.unwrap_or(0.7) as f32), Some(mes.top_p.unwrap_or(0.9) as f32), Some(40), mes.seed.unwrap_or(34562) as u64);
        let m_render = self.chat_template.apply_chat_template(&mes)?; let mut input = self.pre_processor.process_info(&mes, &m_render)?;
        let f_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?; let mut a_ids = f_ids.clone();
        let (t_toks, mut s_off, c_size) = (f_ids.len(), self.get_kv_len(), 128);
        let mut l_pos = s_off;
        while l_pos < t_toks {
            let chunk_size = (t_toks - l_pos).min(c_size); if chunk_size == 0 { break; }
            self.qwen3_vl.forward(&Tensor::from_vec(f_ids[l_pos..l_pos+chunk_size].to_vec(), (1, chunk_size), &self.text_device)?, None, None, None, None, Some(&Tensor::arange(s_off as u32, (s_off + chunk_size) as u32, &self.text_device)?.unsqueeze(0)?), s_off, t_toks, sid.clone())?;
            l_pos += chunk_size; s_off += chunk_size;
            if let Some(s_id) = &sid {
                let slot_id = SLOT_MANAGER.acquire_write_slot(t_toks).await;
                let (ks, vs) = self.get_current_kv();
                if !ks.is_empty() {
                    let mut dumps = Vec::new();
                    for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() {
                        let dims = k.dims(); let heads = dims[1]; let h_dim = dims[3];
                        let k_vec = k.contiguous()?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        let v_vec = v.contiguous()?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        dumps.push(LayerKVDump { layer_idx: idx, k_data: k_vec, v_data: v_vec, shape: vec![1, heads, chunk_size, h_dim] });
                    }
                    if let Ok(tx) = get_worker_channel().await { let _ = tx.send(SlotTask::Bake(BakeTask { slot_id, task_dir: crate::utils::paths::get_kv_dir(None).join(sid.as_ref().unwrap()), kv_name: kv_n.clone(), offset: s_off, layers: dumps, is_relay_baking: false })).await; }
                } else { SLOT_MANAGER.release_slot(slot_id, false).await; }
            }
        }
        self.clear_temporal_kv_caches();
        let (mut p_vals, i_grid, mut g_text) = (input.pixel_values.take(), input.image_grid_thw.take(), String::new());
        for _ in 0..mes.max_tokens.unwrap_or(2048) {
            if let Some(flag) = &cancel { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            let logits = self.qwen3_vl.forward(&Tensor::new(vec![*a_ids.last().unwrap()], &self.text_device)?.unsqueeze(0)?, p_vals.as_ref(), i_grid.as_ref(), None, None, Some(&Tensor::arange(s_off as u32, (s_off + 1) as u32, &self.text_device)?.unsqueeze(0)?), s_off, t_toks, sid.clone())?;
            let l_vec = apply_repeat_penalty(&logits.squeeze(0)?.i(logits.dim(1)? - 1)?.to_dtype(DType::F32)?, 1.1, if a_ids.len() > 512 { &a_ids[a_ids.len()-512..] } else { &a_ids[..] })?;
            let next_id = l_proc.sample(&l_vec)?; if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            a_ids.push(next_id); g_text.push_str(&self.tokenizer.token_decode(vec![next_id])?);
            s_off += 1; p_vals = None; self.clear_temporal_kv_caches();
        }
        Ok(g_text)
    }

    pub fn clear_temporal_kv_caches(&mut self) {
        let reg_obj = match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => Some(m.language_model.registry.clone()), ModelVariant::QuantizedText(m) => Some(m.language_model.registry.clone()), _ => None };
        if let Some(reg_obj) = reg_obj {
            let reg = reg_obj.entries.read().unwrap();
            let layers = match self.qwen3_vl { ModelVariant::QuantizedVL(ref mut m) => &mut m.language_model.layers, ModelVariant::QuantizedText(ref mut m) => &mut m.language_model.layers, _ => unreachable!() };
            for (l_idx, l) in layers.iter_mut().enumerate() {
                for b in &mut l.self_attn.kv_blocks {
                    let mut inner = b.inner.write().unwrap();
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
        let (mut ks, mut vs) = (vec![], vec![]);
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
    pub fn save_kv_to_disk(&mut self, p: &Path, n: Option<&str>, o: usize) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.save_kv_cache(p, false, o, n), ModelVariant::QuantizedText(m) => m.save_kv_cache(p, false, o, n), _ => Ok(()) } }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.to_device(d)?, ModelVariant::QuantizedText(m) => m.to_device(d)?, _ => {} } self.text_device = d.clone(); self.vision_device = d.clone(); Ok(()) }
    pub fn drop_kv_storage(&mut self) -> Result<()> { self.qwen3_vl.drop_kv_storage() }
    pub fn clear_kv_cache(&mut self) { self.clear_temporal_kv_caches(); }
    pub fn truncate_kv_cache(&mut self, l: usize) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.truncate_kv_cache(l), ModelVariant::QuantizedText(m) => m.truncate_kv_cache(l), _ => Ok(()) } }
    pub fn load_kv_from_disk(&mut self, p: &Path, n: Option<&str>) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.load_kv_cache(p, &self.text_device, 0, 128, n), ModelVariant::QuantizedText(m) => m.load_kv_cache(p, &self.text_device, 0, 128, n), _ => Ok(()) } }
}

use crate::models::qwen3vl::quantized_model::KVLocation;
use anyhow::{Result, anyhow};
use candle_core::{quantized::gguf_file, DType, Device, Tensor, IndexOp, Module};
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
use std::time::Duration;
use tokio::sync::mpsc;
use std::collections::HashMap;
use std::collections::VecDeque;
use tokio::sync::oneshot;
use rayon::prelude::*;

pub enum SlotRequest {
    AcquireRead { response: oneshot::Sender<usize> },
    AcquireWrite { response: oneshot::Sender<usize>, total_tokens: usize },
    Release { 
        idx: usize, 
        task_id: Option<String>, 
        block_index: Option<usize>,
        is_bake: bool 
    }, 
    Flush { response: oneshot::Sender<()> },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum InternalState { Free, Writing, Reading }

pub struct SlotManager {
    pub slots: Vec<crate::models::qwen3vl::quantized_model::MemorySlot>,
    pub request_tx: mpsc::Sender<SlotRequest>,
    pub count_reads: Arc<AtomicUsize>,
    pub count_writes: Arc<AtomicUsize>,
    pub count_free: Arc<AtomicUsize>,
}

impl SlotManager {
    pub fn new(count: usize) -> Self {
        println!("[INIT] SlotManager initialized with {} slots (1024-token units).", count);
        let mut slots = Vec::new();
        for i in 0..count { slots.push(crate::models::qwen3vl::quantized_model::MemorySlot::new(i, 28)); }
        let (tx, rx) = mpsc::channel(1000); 
        let cr = Arc::new(AtomicUsize::new(0));
        let cw = Arc::new(AtomicUsize::new(0));
        let cf = Arc::new(AtomicUsize::new(count));
        let (cr_c, cw_c, cf_c) = (cr.clone(), cw.clone(), cf.clone());
        tauri::async_runtime::spawn(async move { Self::slot_dispatcher(rx, cr_c, cw_c, cf_c, count).await; });
        Self { slots, request_tx: tx, count_reads: cr, count_writes: cw, count_free: cf }
    }

    async fn slot_dispatcher(mut rx: mpsc::Receiver<SlotRequest>, cr: Arc<AtomicUsize>, cw: Arc<AtomicUsize>, cf: Arc<AtomicUsize>, max_slots: usize) {
        let mut states = vec![InternalState::Free; max_slots];
        let mut free_p: VecDeque<usize> = (0..max_slots).collect();
        let mut p_writes: VecDeque<(oneshot::Sender<usize>, usize)> = VecDeque::new();
        let mut p_reads: VecDeque<oneshot::Sender<usize>> = VecDeque::new();
        let mut flushers: Vec<oneshot::Sender<()>> = Vec::new();
        let mut sys = sysinfo::System::new_all();

        while let Some(req) = rx.recv().await {
            match req {
                SlotRequest::AcquireWrite { response, total_tokens } => {
                    sys.refresh_memory();
                    let max_c = Self::calculate_dynamic_budget(&sys, total_tokens, max_slots);
                    if cw.load(Ordering::SeqCst) < max_c && !free_p.is_empty() {
                        let idx = free_p.pop_front().unwrap();
                        states[idx] = InternalState::Writing;
                        cf.fetch_sub(1, Ordering::SeqCst); cw.fetch_add(1, Ordering::SeqCst);
                        let _ = response.send(idx);
                    } else { p_writes.push_back((response, total_tokens)); }
                }
                SlotRequest::AcquireRead { response } => {
                    if !free_p.is_empty() {
                        let idx = free_p.pop_front().unwrap();
                        states[idx] = InternalState::Reading;
                        cf.fetch_sub(1, Ordering::SeqCst); cr.fetch_add(1, Ordering::SeqCst);
                        let _ = response.send(idx);
                    } else { p_reads.push_back(response); }
                }
                SlotRequest::Release { idx, .. } => {
                    if idx < max_slots && states[idx] != InternalState::Free {
                        let old = states[idx]; states[idx] = InternalState::Free; free_p.push_back(idx); cf.fetch_add(1, Ordering::SeqCst);
                        if old == InternalState::Writing { cw.fetch_sub(1, Ordering::SeqCst); } else { cr.fetch_sub(1, Ordering::SeqCst); }
                    }
                    if cw.load(Ordering::SeqCst) == 0 && cr.load(Ordering::SeqCst) == 0 { while let Some(w) = flushers.pop() { let _ = w.send(()); } }
                    Self::process_queues_robust(&sys, &mut free_p, &mut p_writes, &mut p_reads, &mut states, &cf, &cw, &cr, max_slots);
                }
                SlotRequest::Flush { response } => {
                    if cw.load(Ordering::SeqCst) == 0 && cr.load(Ordering::SeqCst) == 0 { let _ = response.send(()); } 
                    else { flushers.push(response); }
                }
            }
        }
    }

    fn process_queues_robust(sys: &sysinfo::System, free_p: &mut VecDeque<usize>, p_writes: &mut VecDeque<(oneshot::Sender<usize>, usize)>, p_reads: &mut VecDeque<oneshot::Sender<usize>>, states: &mut [InternalState], cf: &Arc<AtomicUsize>, cw: &Arc<AtomicUsize>, cr: &Arc<AtomicUsize>, max_slots: usize) {
        while !free_p.is_empty() && !p_writes.is_empty() {
            let max_c = Self::calculate_dynamic_budget(sys, p_writes.front().unwrap().1, max_slots);
            if cw.load(Ordering::SeqCst) < max_c {
                let idx = free_p.pop_front().unwrap(); let (res, _) = p_writes.pop_front().unwrap();
                states[idx] = InternalState::Writing; cf.fetch_sub(1, Ordering::SeqCst); cw.fetch_add(1, Ordering::SeqCst);
                let _ = res.send(idx);
            } else { break; }
        }
        while !free_p.is_empty() && !p_reads.is_empty() {
            let idx = free_p.pop_front().unwrap(); let res = p_reads.pop_front().unwrap();
            states[idx] = InternalState::Reading; cf.fetch_sub(1, Ordering::SeqCst); cr.fetch_add(1, Ordering::SeqCst);
            let _ = res.send(idx);
        }
    }

    fn calculate_dynamic_budget(sys: &sysinfo::System, _total_tokens: usize, max_slots: usize) -> usize {
        let avail = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        let base_min = 32; let max_cap = 48; 
        let budget = if avail < 2.0 { base_min } else { ((avail - 2.0) / 0.05) as usize + base_min };
        budget.min(max_cap).min(max_slots)
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
            self.slots[id].state.store(0, Ordering::SeqCst);
            let _ = self.request_tx.send(SlotRequest::Release { idx: id, task_id: None, block_index: None, is_bake: false }).await;
        }
    }
    pub fn get_counts(&self) -> (usize, usize, usize) { (self.count_reads.load(Ordering::Relaxed), self.count_writes.load(Ordering::Relaxed), self.count_free.load(Ordering::Relaxed)) }
}

pub static SLOT_MANAGER: once_cell::sync::Lazy<SlotManager> = once_cell::sync::Lazy::new(|| SlotManager::new(128));
pub static GLOBAL_IO_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub async fn wait_for_global_io() {
    let _ = SLOT_MANAGER.wait_for_all_tasks().await;
    let mut attempts = 0;
    while GLOBAL_IO_COUNTER.load(Ordering::SeqCst) > 0 && attempts < 200 {
        tokio::time::sleep(Duration::from_millis(100)).await; attempts += 1;
    }
}

struct LayerKVDump { layer_idx: usize, k_tensor: Tensor, v_tensor: Tensor }
struct BakeTask { slot_id: usize, task_dir: PathBuf, kv_name: Option<String>, offset: usize, layers: Vec<LayerKVDump>, is_relay_baking: bool, block_idx: Option<usize>, registry: Option<crate::models::qwen3vl::quantized_model::KVRegistry> }
struct SaveTask { slot_id: usize, path: PathBuf, tensors: std::collections::HashMap<String, Tensor>, is_last: bool, block_idx: Option<usize>, registry: Option<crate::models::qwen3vl::quantized_model::KVRegistry> }
pub enum SlotTask { Bake(BakeTask), Load(LoadTask), ChunkedLoad(ChunkedLoadTask) }
pub struct LoadTask { pub slot_id: usize, pub path: PathBuf, pub layer_idx: usize, pub kv_name: Option<String>, pub shared_block: crate::models::qwen3vl::quantized_model::KVBlock, pub registry: crate::models::qwen3vl::quantized_model::KVRegistry }
pub struct ChunkedLoadTask { pub slot_ids: Vec<usize>, pub path: PathBuf, pub layer_indices: Vec<usize>, pub shared_blocks: Vec<crate::models::qwen3vl::quantized_model::KVBlock>, pub registry: crate::models::qwen3vl::quantized_model::KVRegistry }

use tokio::sync::OnceCell;
pub static BAKE_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();
pub static LOAD_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();

pub async fn get_bake_worker() -> Result<mpsc::Sender<SlotTask>> {
    for _ in 0..50 { if let Some(tx) = BAKE_TX.get() { return Ok(tx.clone()); } tokio::time::sleep(std::time::Duration::from_millis(100)).await; }
    Err(anyhow!("Bake worker timeout"))
}

pub async fn get_load_worker() -> Result<mpsc::Sender<SlotTask>> {
    for _ in 0..50 { if let Some(tx) = LOAD_TX.get() { return Ok(tx.clone()); } tokio::time::sleep(std::time::Duration::from_millis(100)).await; }
    Err(anyhow!("Load worker timeout"))
}

pub fn init_bake_worker() {
    let (btx, brx) = mpsc::channel(1000); let (ltx, lrx) = mpsc::channel(1000);
    tauri::async_runtime::spawn(async move { spawn_slot_worker(brx); }); 
    tauri::async_runtime::spawn(async move { spawn_slot_worker(lrx); });
    let _ = BAKE_TX.set(btx); let _ = LOAD_TX.set(ltx);
}

fn spawn_slot_worker(mut rx: mpsc::Receiver<SlotTask>) {
    let (io_tx, mut io_rx) = mpsc::channel::<SaveTask>(1000); 
    tokio::spawn(async move {
        while let Some(task) = io_rx.recv().await {
            let tp = task.path.clone(); let ts = task.tensors; let reg = task.registry.clone(); let b_idx = task.block_idx; let sid = task.slot_id; let is_last = task.is_last;
            tokio::spawn(async move {
                if let Some(p) = tp.parent() { if !p.exists() { let _ = fs::create_dir_all(p); } }
                let tmp = tp.with_extension("tmp");
                if candle_core::safetensors::save(&ts, &tmp).is_ok() {
                    if fs::rename(&tmp, &tp).is_ok() {
                        if let (Some(r), Some(idx)) = (reg, b_idx) {
                            if let Ok(mut entries) = r.entries.write() {
                                if idx < entries.len() {
                                    let e = &mut entries[idx]; e.ssd_path = Some(tp.clone());
                                    if tp.file_name().map(|n| n == "l0.st").unwrap_or(false) {
                                        for i in 0..28 { e.location[i] = crate::models::qwen3vl::quantized_model::KVLocation::SSD; }
                                    } else if let Some(l) = tp.file_name().and_then(|n| n.to_str()).and_then(|s| s.strip_prefix('l')).and_then(|s| s.strip_suffix(".st")).and_then(|s| s.parse::<usize>().ok()) {
                                        if l < 28 { e.location[l] = crate::models::qwen3vl::quantized_model::KVLocation::SSD; }
                                    }
                                }
                            }
                        }
                    }
                }
                GLOBAL_IO_COUNTER.fetch_sub(1, Ordering::SeqCst);
                let slot = &SLOT_MANAGER.slots[sid];
                if slot.remaining_layers.fetch_sub(1, Ordering::SeqCst) == 1 || is_last {
                    let _ = SLOT_MANAGER.request_tx.send(SlotRequest::Release { idx: sid, task_id: None, block_index: None, is_bake: true }).await;
                }
            });
        }
    });
    tokio::spawn(async move {
        while let Some(task) = rx.recv().await {
            match task {
                SlotTask::Bake(bake) => {
                    let io_tx_inner = io_tx.clone();
                    tokio::task::spawn_blocking(move || {
                        let sid = bake.slot_id;
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let (t_dir, off, is_relay) = (bake.task_dir, bake.offset, bake.is_relay_baking);
                            if bake.layers.is_empty() { let _ = SLOT_MANAGER.request_tx.blocking_send(SlotRequest::Release { idx: sid, task_id: None, block_index: None, is_bake: false }); return; }
                            let loop_count = bake.layers.len(); let slot = &SLOT_MANAGER.slots[sid]; slot.remaining_layers.store(loop_count, Ordering::SeqCst); 
                            let block_dir = t_dir.join(format!("b{}", off)); let block_prefix = format!("b{}_", off);
                            for l_idx in 0..loop_count {
                                let mut layer_map = HashMap::new(); let src = &bake.layers[l_idx];
                                let act_l = if is_relay && loop_count == 1 { 0 } else { src.layer_idx };
                                let prefix = format!("{}l{}_", block_prefix, act_l);
                                let k_res = src.k_tensor.to_device(&Device::Cpu).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>());
                                let v_res = src.v_tensor.to_device(&Device::Cpu).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>());
                                let (kd, vd) = match (k_res, v_res) { (Ok(k), Ok(v)) => (k, v), _ => { slot.remaining_layers.fetch_sub(1, Ordering::SeqCst); continue; } };
                                let ks = src.k_tensor.dims(); let (b, h, s, d) = (ks[0], ks[1], ks[2], ks[3]);
                                let hs = s * d; let ac = (0..s).filter(|&i| i < 4 || i % 8 == 0).count();
                                let mut ka = vec![0.0f32; b*h*ac*d]; let mut kp = vec![0u8; (b*h*s*d+7)/8]; let mut ksc = vec![0.0f32; b*h*s];
                                for bh in 0..(b*h) {
                                    let bho = bh*hs;
                                    for i in 0..hs { if kd[bho+i] >= 0.0 { kp[(bho+i)/8] |= 1 << ((bho+i)%8); } }
                                    for ti in 0..s {
                                        let td = &kd[bho+ti*d .. bho+(ti+1)*d];
                                        if ti < 4 || ti % 8 == 0 { let ap = if ti < 4 { ti } else { 4+(ti-4)/8 }; ka[(bh*ac+ap)*d .. (bh*ac+ap+1)*d].copy_from_slice(td); }
                                        let mut m = 0.0f32; for &v in td { let a = v.abs(); if a > m { m = a; } } ksc[bh*s+ti] = m;
                                    }
                                }
                                layer_map.insert(format!("{}k_anchors", prefix), Tensor::from_vec(ka, vec![b,h,ac,d], &Device::Cpu).unwrap());
                                layer_map.insert(format!("{}k_packed", prefix), Tensor::from_vec(kp, vec![(b*h*s*d+7)/8], &Device::Cpu).unwrap());
                                layer_map.insert(format!("{}k_scales", prefix), Tensor::from_vec(ksc, vec![b,h,s,1], &Device::Cpu).unwrap());
                                layer_map.insert(format!("{}k_shape", prefix), Tensor::from_vec(vec![b as u32, h as u32, s as u32, d as u32], (4,), &Device::Cpu).unwrap());
                                
                                let mut va = vec![0.0f32; b*h*ac*d]; let mut vp = vec![0u8; (b*h*s*d+7)/8]; let mut vsc = vec![0.0f32; b*h*s];
                                for bh in 0..(b*h) {
                                    let bho = bh*hs;
                                    for i in 0..hs { if vd[bho+i] >= 0.0 { vp[(bho+i)/8] |= 1 << ((bho+i)%8); } }
                                    for ti in 0..s {
                                        let td = &vd[bho+ti*d .. bho+(ti+1)*d];
                                        if ti < 4 || ti % 8 == 0 { let ap = if ti < 4 { ti } else { 4+(ti-4)/8 }; va[(bh*ac+ap)*d .. (bh*ac+ap+1)*d].copy_from_slice(td); }
                                        let mut m = 0.0f32; for &v in td { let a = v.abs(); if a > m { m = a; } } vsc[bh*s+ti] = m;
                                    }
                                }
                                layer_map.insert(format!("{}v_anchors", prefix), Tensor::from_vec(va, vec![b,h,ac,d], &Device::Cpu).unwrap());
                                layer_map.insert(format!("{}v_packed", prefix), Tensor::from_vec(vp, vec![(b*h*s*d+7)/8], &Device::Cpu).unwrap());
                                layer_map.insert(format!("{}v_scales", prefix), Tensor::from_vec(vsc, vec![b,h,s,1], &Device::Cpu).unwrap());
                                GLOBAL_IO_COUNTER.fetch_add(1, Ordering::SeqCst);
                                let _ = io_tx_inner.blocking_send(SaveTask { slot_id: sid, path: block_dir.join(format!("l{}.st", act_l)), tensors: layer_map, is_last: l_idx == loop_count - 1, block_idx: bake.block_idx, registry: bake.registry.clone() });
                            }
                        }));
                        if let Err(p) = result { println!("[WORKER-PANIC] !! Error in BakeTask: {:?}", p); let _ = SLOT_MANAGER.request_tx.blocking_send(SlotRequest::Release { idx: sid, task_id: None, block_index: None, is_bake: true }); }
                    }).await.ok();
                }
                SlotTask::Load(load) => {
                    let (b_idx_off, b_idx, sid, tp) = { match load.shared_block.inner.read() { Ok(inner) => { let p = inner.ssd_path.clone().or_else(|| { if let Ok(reg) = load.registry.entries.read() { if inner.index < reg.len() { reg[inner.index].ssd_path.clone() } else { None } } else { None } }); (inner.offset, inner.index, load.slot_id, p) }, _ => (0, 999, load.slot_id, None) } };
                    if let Some(path) = tp {
                        let l_idx = load.layer_idx; let reg = load.registry.clone(); let b_off = b_idx_off;
                        tokio::spawn(async move {
                            let act_p = if path.is_dir() { let lp = path.join(format!("l{}.st", l_idx)); if lp.exists() { lp } else { let fallback = path.join("l0.st"); if fallback.exists() { fallback } else { lp } } } else { path.clone() };
                            if let Ok(st) = candle_core::safetensors::load(&act_p, &Device::Cpu) {
                                let prefix = format!("b{}_l{}_", b_off, if act_p.to_string_lossy().contains("l0.st") { 0 } else { l_idx });
                                if let (Some(kh), Some(ka), Some(kp), Some(ks), Some(va), Some(vp), Some(vs)) = (st.get(&format!("{}k_shape", prefix)), st.get(&format!("{}k_anchors", prefix)), st.get(&format!("{}k_packed", prefix)), st.get(&format!("{}k_scales", prefix)), st.get(&format!("{}v_anchors", prefix)), st.get(&format!("{}v_packed", prefix)), st.get(&format!("{}v_scales", prefix)) ) {
                                    let v_u32 = kh.to_vec1::<u32>().unwrap_or_default();
                                    let os = vec![v_u32[0] as usize, v_u32[1] as usize, v_u32[2] as usize, v_u32[3] as usize];
                                    let m = crate::models::qwen3vl::quantized_model::BitKVMetadata { k_anchors: ka.clone(), k_packed: kp.clone(), k_scales: ks.clone(), v_anchors: va.clone(), v_packed: vp.clone(), v_scales: vs.clone(), original_shape: os };
                                    if let Ok(mut r) = reg.entries.write() { if b_idx < r.len() { let e = &mut r[b_idx]; let mut cache = e.bitkv_cache.write().unwrap(); cache[l_idx] = Some(m); e.location[l_idx] = crate::models::qwen3vl::quantized_model::KVLocation::RAM; } }
                                }
                            }
                            SLOT_MANAGER.release_slot(sid).await;
                        });
                    } else { SLOT_MANAGER.release_slot(sid).await; }
                }
                _ => {}
            }
        }
    });
}

#[derive(Clone)]
pub enum ModelVariant { Standard(crate::models::qwen3vl::model::Qwen3VLModel), QuantizedVL(QuantizedQwen3VLModel), QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel) }

impl ModelVariant {
    pub fn forward(&mut self, i: &Tensor, pv: Option<&Tensor>, ithw: Option<&Tensor>, vpv: Option<&Tensor>, vthw: Option<&Tensor>, cp: Option<&Tensor>, off: usize, tl: usize, sid: Option<String>, sl: Option<usize>, nl: Option<usize>) -> Result<Tensor> {
        match self { 
            Self::Standard(m) => m.forward(i, pv, ithw, vpv, vthw, cp, off), 
            Self::QuantizedVL(m) => m.forward(i, pv, ithw, vpv, vthw, cp, off, tl, sid, sl, nl), 
            Self::QuantizedText(m) => m.forward(i, cp, off, tl, sid, sl, nl) 
        }
    }

    pub fn forward_relay(&mut self, embeds: &Tensor, off: usize, tl: usize, sid: Option<String>, sl: Option<usize>, nl: Option<usize>) -> Result<Tensor> {
        match self {
            Self::QuantizedText(m) => m.language_model.forward(embeds, off, tl, None, None, None, sid, sl, nl),
            Self::QuantizedVL(m) => m.language_model.forward(embeds, off, tl, None, None, None, sid, sl, nl),
            _ => Err(anyhow!("Relay not supported for this model variant")),
        }
    }
    pub fn rebalance_layers(&mut self, d: usize, target_idx: usize) -> Result<()> { match self { Self::Standard(_) => Ok(()), Self::QuantizedVL(m) => m.rebalance_layers(d, target_idx), Self::QuantizedText(m) => m.rebalance_layers(d, target_idx) } }
    pub fn drop_kv_storage(&mut self) -> Result<()> { match self { Self::Standard(_) => Ok(()), Self::QuantizedVL(m) => m.language_model.drop_kv_storage(), Self::QuantizedText(m) => m.language_model.drop_kv_storage() } }
    pub fn save_metadata_to_file(&self, path: &Path) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.registry.save_to_file(path), Self::QuantizedText(m) => m.language_model.registry.save_to_file(path), _ => Ok(()) } }
    pub fn load_metadata_from_file(&self, path: &Path) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.registry.load_from_file(path), Self::QuantizedText(m) => m.language_model.registry.load_from_file(path), _ => Ok(()) } }
    pub fn inject_kv_bitkv(&mut self, ka: &[Tensor], kp: &[Tensor], ks: &[Tensor], va: &[Tensor], vp: &[Tensor], vs: &[Tensor], os: &[usize]) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.inject_live_kv_bitkv(ka, kp, ks, va, vp, vs, os), Self::QuantizedText(m) => m.language_model.inject_live_kv_bitkv(ka, kp, ks, va, vp, vs, os), _ => Ok(()) } }
    pub fn is_cpu(&self) -> bool { match self { Self::Standard(m) => m.device().is_cpu(), Self::QuantizedVL(m) => m.language_model.is_forced_cpu, Self::QuantizedText(m) => m.language_model.is_forced_cpu } }
    pub fn get_registry(&self) -> Option<crate::models::qwen3vl::quantized_model::KVRegistry> { match self { Self::QuantizedText(m) => Some(m.language_model.registry.clone()), Self::QuantizedVL(m) => Some(m.language_model.registry.clone()), _ => None } }
    pub fn set_kv_len(&mut self, len: usize) { match self { Self::QuantizedText(m) => m.language_model.current_kv_len = len, Self::QuantizedVL(m) => m.language_model.current_kv_len = len, _ => {}, } }
}

pub struct Qwen3VLGenerateModel { pub chat_template: ChatTemplate, pub tokenizer: TokenizerModel, pub pre_processor: Qwen3VLProcessor, pub qwen3_vl: ModelVariant, pub text_device: Device, pub vision_device: Device, pub eos_token_id1: u32, pub eos_token_id2: u32, pub generation_config: Qwen3VLGenerationConfig, pub model_name: String, pub hard_token_limit: Option<usize>, pub kv_root: std::path::PathBuf }

impl Qwen3VLGenerateModel {
    pub fn init_with_config(path: &str, tokenizer_path: Option<&str>, config_path: Option<&str>, text_device: Option<&Device>, text_device_id: usize, vision_device: Option<&Device>, vision_device_id: usize, dtype: Option<DType>, hard_token_limit: Option<usize>, force_text_only: bool, baking_only: bool, is_disk_swap: bool, kv_root: std::path::PathBuf) -> Result<Self> {
        let path = if let Some(s) = path.strip_prefix(r"\\?\") { s } else { path };
        let tok_p = if let Some(s) = tokenizer_path.unwrap_or(path).strip_prefix(r"\\?\") { s } else { tokenizer_path.unwrap_or(path) };
        let cfg_p = if let Some(s) = config_path.unwrap_or(path).strip_prefix(r"\\?\") { s } else { config_path.unwrap_or(path) };
        let tokenizer = TokenizerModel::init(tok_p)?;
        let raw_c: serde_json::Value = serde_json::from_slice(&std::fs::read(std::path::Path::new(cfg_p).join("config.json"))?)?;
        let cfg: Qwen3VLConfig = if raw_c.get("text_config").is_some() { serde_json::from_value(raw_c)? } else { let text_config: crate::models::qwen3vl::config::Qwen3VLTextConfig = serde_json::from_value(raw_c.clone())?; crate::models::qwen3vl::config::Qwen3VLConfig { architectures: raw_c.get("architectures").and_then(|v| serde_json::from_value(v.clone()).ok()), auto_map: raw_c.get("auto_map").and_then(|v| serde_json::from_value(v.clone()).ok()), hidden_size: raw_c.get("hidden_size").and_then(|v| v.as_u64()).map(|v| v as usize), image_token_id: raw_c.get("image_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), model_type: raw_c.get("model_type").and_then(|v| v.as_str()).unwrap_or("qwen2").to_string(), text_config: Some(text_config), tie_word_embeddings: raw_c.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(true), torch_dtype: raw_c.get("torch_dtype").and_then(|v| v.as_str()).map(|s| s.to_string()), transformers_version: raw_c.get("transformers_version").and_then(|v| v.as_str()).unwrap_or("").to_string(), video_token_id: raw_c.get("video_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), vision_config: None, vision_start_token_id: None, vision_end_token_id: None } };
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
                ModelVariant::QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel::new_with_mmap(&cfg, &gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?, Some(Arc::new(m_mmap)), &t_dev, text_device_id, dtype, kv_res, baking_only, baking_only)?)
            }
        } else { ModelVariant::Standard(Qwen3VLModel::new(cfg, unsafe { VarBuilder::from_mmaped_safetensors(&find_type_files(path, "safetensors")?, dtype, &t_dev)? })?) };
        let g_p = std::path::Path::new(cfg_p).join("generation_config.json"); let g_cfg: Qwen3VLGenerationConfig = if g_p.exists() { serde_json::from_slice(&std::fs::read(g_p)?)? } else { Qwen3VLGenerationConfig::default() };
        let (e1, e2) = match &g_cfg.eos_token_id { serde_json::Value::Number(n) => { let id = n.as_u64().unwrap_or(151645) as u32; (id, id) }, serde_json::Value::Array(arr) => { let id1 = arr.get(0).and_then(|v| v.as_u64()).unwrap_or(151643) as u32; let id2 = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(id1 as u64) as u32; (id1, id2) }, _ => (151643, 151643) };
        Ok(Self { chat_template: ChatTemplate::init(tok_p)?, tokenizer, pre_processor: Qwen3VLProcessor::new(tok_p, &v_dev, dtype)?, qwen3_vl, text_device: t_dev, vision_device: v_dev, eos_token_id1: e1, eos_token_id2: e2, generation_config: g_cfg, model_name: if baking_only { "Small (Single-Layer)".into() } else if path.contains("0.6B") { "Small (Full-Layer)".into() } else { "2B (Full-Layer)".into() }, hard_token_limit, kv_root })
    }

    pub async fn prefill_chunk(&mut self, text: String, _cancel: Option<Arc<AtomicBool>>, _relay: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let ids = self.tokenizer.text_encode_vec(text, false)?; let size = ids.len(); let pos = self.get_kv_len();
        self.qwen3_vl.forward(&Tensor::from_vec(ids, (1, size), &self.text_device)?, None, None, None, None, Some(&Tensor::arange(pos as u32, (pos + size) as u32, &self.text_device)?.unsqueeze(0)?), pos, size, None, None, None)?;
        Ok(size)
    }

    pub async fn prefill_only(&mut self, mes: ChatCompletionParameters, cancel: Option<Arc<AtomicBool>>, sid: Option<String>, _relay: Option<&mut Qwen3VLGenerateModel>, kv_n: Option<String>) -> Result<usize> {
        let start_prep = std::time::Instant::now(); if sid.is_none() { SLOT_MANAGER.reset_all_slots().await; }
        let m_render = self.chat_template.apply_chat_template(&mes)?; let input = self.pre_processor.process_info(&mes, &m_render)?;
        let f_ids = self.tokenizer.text_encode_vec(input.replace_text, false)?; let t_toks = f_ids.len();
        if let Some(flag) = &cancel { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
        println!("[BAKING] Starting Horizontal Pass for {} tokens...", t_toks);
        self.qwen3_vl.forward(&Tensor::from_vec(f_ids.clone(), (1, t_toks), &self.text_device)?, None, None, None, None, Some(&Tensor::arange(0u32, t_toks as u32, &self.text_device)?.unsqueeze(0)?), 0, t_toks, sid.clone(), None, None)?;
        if let Some(s_id) = &sid {
            let unbaked = self.get_all_unbaked_kv_blocks();
            for (ks, vs, off) in unbaked {
                let slot_id = SLOT_MANAGER.acquire_write_slot(t_toks).await;
                let path = crate::utils::paths::get_kv_dir(None).join(s_id); if !path.exists() { let _ = fs::create_dir_all(&path); }
                let mut dumps = Vec::new(); for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() { dumps.push(LayerKVDump { layer_idx: idx, k_tensor: k, v_tensor: v }); }
                if let Ok(tx) = get_bake_worker().await {
                    let rr = match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => Some(m.language_model.registry.clone()), ModelVariant::QuantizedText(m) => Some(m.language_model.registry.clone()), _ => None };
                    let is_baking_mode = match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.baking_only, ModelVariant::QuantizedText(m) => m.language_model.baking_only, _ => false };
                    let _ = tx.send(SlotTask::Bake(BakeTask { slot_id, task_dir: path, kv_name: kv_n.clone(), offset: off, layers: dumps, is_relay_baking: is_baking_mode, block_idx: Some(off / 256), registry: rr })).await;
                }
            }
            SLOT_MANAGER.wait_for_all_tasks().await;
            let path = crate::utils::paths::get_kv_dir(None).join(s_id); if !path.exists() { let _ = fs::create_dir_all(&path); }
            let _ = self.qwen3_vl.save_metadata_to_file(&path); self.clear_temporal_kv_caches();
        }
        Ok(t_toks)
    }

    pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel: Option<Arc<AtomicBool>>, sid: Option<String>, kv_n: Option<String>, pre_draft: Option<Vec<u32>>) -> Result<String> {
        let start_prep = std::time::Instant::now(); if sid.is_none() { SLOT_MANAGER.reset_all_slots().await; }
        let mut l_proc = get_logit_processor(Some(mes.temperature.unwrap_or(0.7) as f32), Some(mes.top_p.unwrap_or(0.9) as f32), Some(40), mes.seed.unwrap_or(34562) as u64);
        let m_render = self.chat_template.apply_chat_template(&mes)?; let mut input = self.pre_processor.process_info(&mes, &m_render)?;
        let f_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?; let t_toks = f_ids.len();
        let mut a_ids = f_ids.clone(); let mut curr_s_off = self.get_kv_len();
        let (mut p_vals, i_grid, mut g_text) = (input.pixel_values.take(), input.image_grid_thw.take(), String::new());
        let max_gen = mes.max_tokens.unwrap_or(2048) as usize; let mut gen_count = 0;
        let mut active_draft = pre_draft.unwrap_or_default(); let is_small = self.model_name.contains("Small");

        if t_toks > curr_s_off {
            let prefill_len = t_toks - curr_s_off;
            self.qwen3_vl.forward(&Tensor::from_vec(f_ids[curr_s_off..].to_vec(), (1, prefill_len), &self.text_device)?, None, None, None, None, Some(&Tensor::arange(curr_s_off as u32, t_toks as u32, &self.text_device)?.unsqueeze(0)?), curr_s_off, prefill_len, sid.clone(), None, None)?;
            curr_s_off = t_toks;
        }

        if is_small {
            println!("[DRAFTING] Small model starting HIGH-SPEED RELAY (0..{})", t_toks + 24);
            let mut current_chunk_ids = Vec::new();
            let mut temp_off = curr_s_off;

            for _ in 0..24 {
                let last_c = *a_ids.last().unwrap_or(&0);
                let logits = self.qwen3_vl.forward(&Tensor::from_vec(vec![last_c], (1, 1), &self.text_device)?, None, None, None, None, None, temp_off, 1, sid.clone(), Some(0), Some(1))?;
                let next_id = l_proc.sample(&logits.flatten_all()?)?;
                a_ids.push(next_id);
                current_chunk_ids.push(next_id);
                temp_off += 1;
                if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            }

            if !a_ids.is_empty() {
                self.clear_kv_cache(); 
                println!("[RELAY-START] High-Speed Hybrid Relay (Prompt: 32-Chunk / Draft: 1-Token)");
                
                let prompt_chunk_size = 32; 
                for chunk_start in (0..curr_s_off).step_by(prompt_chunk_size) {
                    let chunk_end = (chunk_start + prompt_chunk_size).min(curr_s_off);
                    let chunk_ids = &a_ids[chunk_start..chunk_end];
                    if chunk_ids.is_empty() { continue; }

                    let mut current_h = match &mut self.qwen3_vl {
                        ModelVariant::QuantizedText(m) => m.language_model.embed_tokens.forward(&Tensor::from_vec(chunk_ids.to_vec(), (1, chunk_ids.len()), &self.text_device)?)?,
                        ModelVariant::QuantizedVL(m) => m.language_model.embed_tokens.forward(&Tensor::from_vec(chunk_ids.to_vec(), (1, chunk_ids.len()), &self.text_device)?)?,
                        _ => unreachable!(),
                    };

                    for l_idx in 0..28 {
                        current_h = self.qwen3_vl.forward_relay(&current_h, chunk_start, a_ids.len(), sid.clone(), Some(l_idx), Some(1))?;
                    }
                    if chunk_start % 2048 == 0 { println!("[PROMPT-SPEED] Tokens {}/{} synced.", chunk_start, curr_s_off); }
                }

                for t_idx in curr_s_off..a_ids.len() {
                    let token_id = a_ids[t_idx];
                    let mut current_h = match &mut self.qwen3_vl {
                        ModelVariant::QuantizedText(m) => m.language_model.embed_tokens.forward(&Tensor::from_vec(vec![token_id], (1, 1), &self.text_device)?)?,
                        ModelVariant::QuantizedVL(m) => m.language_model.embed_tokens.forward(&Tensor::from_vec(vec![token_id], (1, 1), &self.text_device)?)?,
                        _ => unreachable!(),
                    };

                    for l_idx in 0..28 {
                        current_h = self.qwen3_vl.forward_relay(&current_h, t_idx, a_ids.len(), sid.clone(), Some(l_idx), Some(1))?;
                    }
                }
                println!("[RELAY-COMPLETE] All layers synchronized up to token {}.", temp_off - 1);
            }

            if let Some(s_id) = &sid {
                self.bundle_draft_fragments(s_id, 0, temp_off).await?;
                self.qwen3_vl.set_kv_len(temp_off);
            }

            let final_text = self.tokenizer.token_decode(current_chunk_ids)?;
            println!("[BURST-RESULT] {}", final_text);
            g_text = final_text;
        } else {
            while gen_count < max_gen {
                if let Some(flag) = &cancel { if flag.load(Ordering::Relaxed) { break; } }
                let last_c = *a_ids.last().unwrap_or(&0);
                let mut v_batch = vec![last_c]; let is_verifying_macro_draft = !active_draft.is_empty(); v_batch.extend(&active_draft);
                let logits = self.qwen3_vl.forward(&Tensor::from_vec(v_batch.clone(), (1, v_batch.len()), &self.text_device)?, p_vals.as_ref(), i_grid.as_ref(), None, None, None, curr_s_off, v_batch.len(), sid.clone(), None, None)?;
                let mut acc_this = 0; let mut confirmed = Vec::new();
                for i in 0..v_batch.len() - 1 {
                    let target_l = logits.narrow(1, i, 1)?.flatten_all()?;
                    let l_v = apply_repeat_penalty(&target_l.to_dtype(DType::F32)?, 1.1, if a_ids.len() > 512 { &a_ids[a_ids.len()-512..] } else { &a_ids[..] })?;
                    let acc_id = l_proc.sample(&l_v)?;
                    if i < active_draft.len() && acc_id == active_draft[i] { confirmed.push(acc_id); acc_this += 1; if acc_id == self.eos_token_id1 || acc_id == self.eos_token_id2 { break; } }
                    else { confirmed.push(acc_id); acc_this += 1; break; }
                }
                if acc_this == 0 { let target_l = logits.narrow(1, 0, 1)?.flatten_all()?; let next_id = l_proc.sample(&target_l)?; confirmed.push(next_id); acc_this = 1; }
                let chunk_txt = self.tokenizer.token_decode(confirmed.clone())?; print!("{}", chunk_txt);
                if !active_draft.is_empty() { print!(" [SPEC] Accepted: {}/{}", confirmed.len().min(active_draft.len()), active_draft.len()); }
                use std::io::Write; let _ = std::io::stdout().flush();
                g_text.push_str(&chunk_txt); a_ids.extend(confirmed); curr_s_off += acc_this; gen_count += acc_this; active_draft.clear(); p_vals = None;
                if is_verifying_macro_draft { break; }
                if *a_ids.last().unwrap() == self.eos_token_id1 || *a_ids.last().unwrap() == self.eos_token_id2 { break; }
                self.clear_temporal_kv_caches();
            }
        }
        if let Some(s_id) = &sid { let path = crate::utils::paths::get_kv_dir(None).join(s_id); if !path.exists() { let _ = fs::create_dir_all(&path); } let _ = self.qwen3_vl.save_metadata_to_file(&path); }
        Ok(g_text)
    }

    pub async fn bundle_draft_fragments(&mut self, session_id: &str, start_off: usize, end_off: usize) -> Result<()> {
        let task_dir = crate::utils::paths::get_task_specific_dir(None, session_id);
        let frag_dir = task_dir.join("fragments");
        let meta_path = frag_dir.join("metadata.json");
        if !meta_path.exists() { return Ok(()); }
        
        let content = std::fs::read_to_string(&meta_path)?;
        let mut fragments: Vec<serde_json::Value> = Vec::new();
        for line in content.lines() { if let Ok(v) = serde_json::from_str(line) { fragments.push(v); } }

        let block_size = 256;
        let start_block = (start_off / block_size) * block_size;
        let end_block = ((end_off + block_size - 1) / block_size) * block_size;

        for b_off in (start_block..end_block).step_by(block_size) {
            let b_end = b_off + block_size;
            let block_dir = crate::utils::paths::get_kv_dir(None).join(session_id).join(format!("b{}", b_off));
            if !block_dir.exists() { let _ = fs::create_dir_all(&block_dir); }

            for l_idx in 0..28 {
                let mut layer_frags = fragments.iter()
                    .filter(|f| f["layer"].as_u64().unwrap_or(999) == l_idx as u64)
                    .filter(|f| {
                        let off = f["off"].as_u64().unwrap_or(0) as usize;
                        off >= b_off && off < b_end
                    })
                    .collect::<Vec<_>>();
                
                if layer_frags.is_empty() { continue; }
                layer_frags.sort_by_key(|f| f["off"].as_u64().unwrap_or(0));

                let mut ks = Vec::new();
                let mut vs = Vec::new();
                for frag in layer_frags {
                    let path_str = frag["path"].as_str().unwrap_or("");
                    if let Ok(st_map) = candle_core::safetensors::load(Path::new(path_str), &Device::Cpu) {
                        if let (Some(k), Some(v)) = (st_map.get("k"), st_map.get("v")) {
                            ks.push(k.clone());
                            vs.push(v.clone());
                        }
                    }
                }

                if !ks.is_empty() {
                    let k_combined = Tensor::cat(&ks, 2)?;
                    let v_combined = Tensor::cat(&vs, 2)?;
                    let mut map = HashMap::new();
                    let prefix = format!("b{}_l{}_", b_off, l_idx);
                    map.insert(format!("{}k_anchors", prefix), k_combined.clone());
                    map.insert(format!("{}v_anchors", prefix), v_combined.clone());
                    map.insert(format!("{}k_shape", prefix), Tensor::from_vec(vec![1u32, 16, k_combined.dim(2)? as u32, 128], (4,), &Device::Cpu)?);
                    let save_path = block_dir.join(format!("l{}.st", l_idx));
                    let _ = candle_core::safetensors::save(&map, &save_path);

                    if let Some(reg_obj) = self.qwen3_vl.get_registry() {
                        let mut reg = reg_obj.entries.write().unwrap();
                        let block_idx = b_off / block_size;
                        if block_idx < reg.len() {
                            let entry = &mut reg[block_idx];
                            entry.ssd_path = Some(block_dir.clone());
                            entry.location[l_idx] = KVLocation::SSD;
                            entry.token_len = k_combined.dim(2)?;
                        }
                    }
                }
            }
            println!("[BUNDLER] Block b{} consolidation complete.", b_off);
        }
        Ok(())
    }

    pub fn clear_temporal_kv_caches(&mut self) {
        let reg_obj = match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => Some(m.language_model.registry.clone()), ModelVariant::QuantizedText(m) => Some(m.language_model.registry.clone()), _ => None };
        if let Some(reg_obj) = reg_obj {
            let reg = reg_obj.entries.read().unwrap();
            let layers = match self.qwen3_vl { ModelVariant::QuantizedVL(ref mut m) => &mut m.language_model.layers, ModelVariant::QuantizedText(ref mut m) => &mut m.language_model.layers, _ => unreachable!() };
            for (l_idx, l) in layers.iter_mut().enumerate() {
                for b in &mut l.self_attn.kv_blocks {
                    let mut inner = b.inner.write().unwrap();
                    let reg_loc = if inner.index < reg.len() { reg[inner.index].location[l_idx] } else { inner.location };
                    if reg_loc == KVLocation::SSD || (inner.location != KVLocation::VRAM && reg_loc != KVLocation::VRAM) {
                        if inner.k_cache.is_some() || inner.v_cache.is_some() { inner.k_cache = None; inner.v_cache = None; inner.location = reg_loc; }
                    }
                }
            }
        }
    }

    pub fn get_blocks_by_location(&self, target_loc: KVLocation) -> Result<Vec<(Vec<Tensor>, Vec<Tensor>, usize, usize)>> {
        let mut results = Vec::new();
        let layers = match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => Some(&m.language_model.layers), ModelVariant::QuantizedText(m) => Some(&m.language_model.layers), _ => None };
        if let Some(layers) = layers {
            if layers.is_empty() { return Ok(vec![]); }
            let num_blocks = layers[0].self_attn.kv_blocks.len();
            for b_idx in 0..num_blocks {
                let mut ks = Vec::new(); let mut vs = Vec::new(); let mut match_found = false; let mut offset = 0;
                for l in layers {
                    if b_idx < l.self_attn.kv_blocks.len() {
                        let inner = l.self_attn.kv_blocks[b_idx].inner.read().unwrap();
                        if inner.location == target_loc || (target_loc == KVLocation::SSD_PENDING && inner.location == KVLocation::RAM && inner.ssd_path.is_none()) {
                            if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                                let k_cpu = if k.device().is_cpu() { k.clone() } else { k.to_device(&Device::Cpu)? };
                                let v_cpu = if v.device().is_cpu() { v.clone() } else { v.to_device(&Device::Cpu)? };
                                ks.push(k_cpu); vs.push(v_cpu); offset = inner.offset + inner.len; match_found = true;
                            }
                        }
                    }
                }
                if match_found { results.push((ks, vs, offset, b_idx)); }
            }
        }
        Ok(results)
    }
    pub fn mark_block_location(&mut self, block_idx: usize, new_loc: KVLocation) {
        let layers = match self.qwen3_vl { ModelVariant::QuantizedVL(ref mut m) => &mut m.language_model.layers, ModelVariant::QuantizedText(ref mut m) => &mut m.language_model.layers, _ => return };
        for l in layers { if block_idx < l.self_attn.kv_blocks.len() { let mut inner = l.self_attn.kv_blocks[block_idx].inner.write().unwrap(); inner.location = new_loc; } }
    }
    pub fn get_kv_len(&self) -> usize { match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.get_kv_len(), ModelVariant::QuantizedText(m) => m.language_model.get_kv_len(), _ => 0 } }
    pub fn get_all_unbaked_kv_blocks(&self) -> Vec<(Vec<Tensor>, Vec<Tensor>, usize)> {
        let mut all_blocks: Vec<(Vec<Tensor>, Vec<Tensor>, usize)> = Vec::new();
        let layers: Option<&Vec<crate::models::qwen3vl::quantized_model::QuantizedQwen3VLTextDecoderLayer>> = match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => Some(&m.language_model.layers), ModelVariant::QuantizedText(m) => Some(&m.language_model.layers), _ => None };
        if let Some(layers) = layers {
            if layers.is_empty() { return vec![]; }
            let num_blocks = layers[0].self_attn.kv_blocks.len();
            for b_idx in 0..num_blocks {
                let mut ks: Vec<Tensor> = Vec::new(); let mut vs: Vec<Tensor> = Vec::new(); let mut has_data = false; let mut block_start_offset = 0;
                for l in layers { if b_idx < l.self_attn.kv_blocks.len() { let inner = l.self_attn.kv_blocks[b_idx].inner.read().unwrap(); if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) { ks.push(k.clone()); vs.push(v.clone()); block_start_offset = inner.offset; has_data = true; } } }
                if has_data { all_blocks.push((ks, vs, block_start_offset)); }
            }
        }
        all_blocks
    }
    pub fn save_kv_to_disk(&mut self, p: &Path, n: Option<&str>, o: usize) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.save_kv_cache(p, false, o, n), ModelVariant::QuantizedText(m) => m.save_kv_cache(p, false, o, n), _ => Ok(()) } }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.to_device(d)?, ModelVariant::QuantizedText(m) => m.to_device(d)?, _ => {} } self.text_device = d.clone(); self.vision_device = d.clone(); Ok(()) }
    pub fn drop_kv_storage(&mut self) -> Result<()> { self.qwen3_vl.drop_kv_storage() }
    pub fn clear_kv_cache(&mut self) { 
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.language_model.clear_kv_cache(),
            ModelVariant::QuantizedText(m) => m.language_model.clear_kv_cache(),
            _ => {},
        }
    }
    pub fn truncate_kv_cache(&mut self, l: usize) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.truncate_kv_cache(l), ModelVariant::QuantizedText(m) => m.truncate_kv_cache(l), _ => Ok(()) } }
    pub fn load_kv_from_disk(&mut self, p: &Path, n: Option<&str>) -> Result<()> { 
        let _ = self.qwen3_vl.load_metadata_from_file(p);
        let mut restored_len = 0;
        let reg_ref = match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => Some(m.language_model.registry.clone()), ModelVariant::QuantizedText(m) => Some(m.language_model.registry.clone()), _ => None };
        if let Some(reg) = reg_ref { if let Ok(entries) = reg.entries.read() { if let Some(last_valid) = entries.iter().rev().find(|e| e.token_len > 0) { restored_len = last_valid.token_start + last_valid.token_len; } } }
        if restored_len > 0 { println!("[SSD-LOAD] Metadata restored. Exact KV Length: {}", restored_len); }
        match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.load_kv_cache(p, &self.text_device, restored_len, 128, n), ModelVariant::QuantizedText(m) => m.load_kv_cache(p, &self.text_device, restored_len, 128, n), _ => Ok(()) } 
    }
}

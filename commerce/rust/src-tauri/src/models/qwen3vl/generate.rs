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
                        cf.fetch_sub(1, Ordering::SeqCst);
                        cw.fetch_add(1, Ordering::SeqCst);
                        let _ = response.send(idx);
                    } else { p_writes.push_back((response, total_tokens)); }
                }
                SlotRequest::AcquireRead { response } => {
                    if !free_p.is_empty() {
                        let idx = free_p.pop_front().unwrap();
                        states[idx] = InternalState::Reading;
                        cf.fetch_sub(1, Ordering::SeqCst);
                        cr.fetch_add(1, Ordering::SeqCst);
                        let _ = response.send(idx);
                    } else { p_reads.push_back(response); }
                }
                SlotRequest::Release { idx, .. } => {
                    if idx < max_slots && states[idx] != InternalState::Free {
                        let old = states[idx];
                        states[idx] = InternalState::Free;
                        free_p.push_back(idx);
                        cf.fetch_add(1, Ordering::SeqCst);
                        if old == InternalState::Writing { cw.fetch_sub(1, Ordering::SeqCst); }
                        else { cr.fetch_sub(1, Ordering::SeqCst); }
                        println!("[CENTRAL-CONTROL] << Received RELEASE notification for Slot {}. (Free slots: {})", idx, cf.load(Ordering::SeqCst));
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
                let idx = free_p.pop_front().unwrap();
                let (res, _) = p_writes.pop_front().unwrap();
                states[idx] = InternalState::Writing;
                cf.fetch_sub(1, Ordering::SeqCst); cw.fetch_add(1, Ordering::SeqCst);
                let _ = res.send(idx);
            } else { break; }
        }
        while !free_p.is_empty() && !p_reads.is_empty() {
            let idx = free_p.pop_front().unwrap();
            let res = p_reads.pop_front().unwrap();
            states[idx] = InternalState::Reading;
            cf.fetch_sub(1, Ordering::SeqCst); cr.fetch_add(1, Ordering::SeqCst);
            let _ = res.send(idx);
        }
    }

    fn calculate_dynamic_budget(sys: &sysinfo::System, _total_tokens: usize, max_slots: usize) -> usize {
        let avail = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        let base_min = 56;
        let budget = ((avail - 1.5).max(0.0) / 0.02) as usize;
        budget.max(base_min).min(max_slots)
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
            let _ = self.request_tx.send(SlotRequest::Release { 
                idx: id, 
                task_id: None, 
                block_index: None, 
                is_bake: false 
            }).await;
        }
    }
    pub fn get_counts(&self) -> (usize, usize, usize) { (self.count_reads.load(Ordering::Relaxed), self.count_writes.load(Ordering::Relaxed), self.count_free.load(Ordering::Relaxed)) }
}

pub static SLOT_MANAGER: once_cell::sync::Lazy<SlotManager> = once_cell::sync::Lazy::new(|| SlotManager::new(256));

struct LayerKVDump { layer_idx: usize, k_tensor: Tensor, v_tensor: Tensor }
struct BakeTask { 
    slot_id: usize, 
    task_dir: PathBuf, 
    kv_name: Option<String>, 
    offset: usize, 
    layers: Vec<LayerKVDump>, 
    is_relay_baking: bool,
    block_idx: Option<usize>, // [NEW] 추적용
    registry: Option<crate::models::qwen3vl::quantized_model::KVRegistry> // [NEW] 장부 업데이트용
}
struct SaveTask { 
    slot_id: usize, 
    path: PathBuf, 
    tensors: std::collections::HashMap<String, Tensor>, 
    is_last: bool,
    block_idx: Option<usize>,
    registry: Option<crate::models::qwen3vl::quantized_model::KVRegistry>
}
pub struct ChunkedLoadTask {
    pub slot_ids: Vec<usize>,
    pub path: PathBuf,
    pub layer_indices: Vec<usize>,
    pub shared_blocks: Vec<crate::models::qwen3vl::quantized_model::KVBlock>,
    pub registry: crate::models::qwen3vl::quantized_model::KVRegistry,
}

pub enum SlotTask { Bake(BakeTask), Load(LoadTask), ChunkedLoad(ChunkedLoadTask) }
pub struct LoadTask { pub slot_id: usize, pub path: PathBuf, pub layer_idx: usize, pub kv_name: Option<String>, pub shared_block: crate::models::qwen3vl::quantized_model::KVBlock, pub registry: crate::models::qwen3vl::quantized_model::KVRegistry }

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
    let (btx, brx) = mpsc::channel(1000);
    let (ltx, lrx) = mpsc::channel(1000);
    tauri::async_runtime::spawn(async move { spawn_slot_worker(brx); }); 
    tauri::async_runtime::spawn(async move { spawn_slot_worker(lrx); });
    let _ = BAKE_TX.set(btx);
    let _ = LOAD_TX.set(ltx);
}

fn spawn_slot_worker(mut rx: mpsc::Receiver<SlotTask>) {
    let (io_tx, mut io_rx) = mpsc::channel::<SaveTask>(1000); 
    tokio::spawn(async move {
        while let Some(task) = io_rx.recv().await {
            if let Some(parent) = task.path.parent() { if !parent.exists() { let _ = std::fs::create_dir_all(parent); } }
            
            let b_idx_str = task.block_idx.map(|i| i.to_string()).unwrap_or("?".into());
            // println!("[WORKER-IO] << [RECEIVED] Slot {} starting save for Block {} to {:?}", task.slot_id, b_idx_str, task.path.file_name());

            // [LOG] 3단계: 결과 기록
            match candle_core::safetensors::save(&task.tensors, &task.path) {
                Ok(_) => {
                    println!("[WORKER-IO] [SUCCESS] Block {} saved. Updating Registry.", b_idx_str);
                    
                    // [REGISTRY-UPDATE] 장부에 저장 경로 기록
                    if let (Some(reg), Some(idx)) = (task.registry, task.block_idx) {
                        if let Ok(mut entries) = reg.entries.write() {
                            if idx < entries.len() {
                                entries[idx].ssd_path = Some(task.path.clone());
                                // [SAFE-UPDATE] 이미 RAM에 로드된 레이어는 건드리지 않고, 
                                // SSD_PENDING이나 VRAM 상태인 것들만 SSD로 확정합니다.
                                for loc in entries[idx].location.iter_mut() {
                                    if *loc == crate::models::qwen3vl::quantized_model::KVLocation::SSD_PENDING || *loc == crate::models::qwen3vl::quantized_model::KVLocation::VRAM {
                                        *loc = crate::models::qwen3vl::quantized_model::KVLocation::SSD;
                                    }
                                }
                            }
                        }
                    }
                },
                Err(e) => {
                    println!("[WORKER-IO] !! [ERROR] Block {} save failed! Path: {:?}, Cause: {:?}", b_idx_str, task.path, e);
                }
            }

            let slot = &SLOT_MANAGER.slots[task.slot_id];
            if slot.remaining_layers.fetch_sub(1, Ordering::SeqCst) == 1 || task.is_last {
                // [NEW] 상세 정보와 함께 슬롯 해제 보고
                let _ = SLOT_MANAGER.request_tx.send(SlotRequest::Release { 
                    idx: task.slot_id,
                    task_id: None, 
                    block_index: None,
                    is_bake: true 
                }).await;
            }
        }
    });
    tokio::spawn(async move {
        while let Some(task) = rx.recv().await {
            match task {
                SlotTask::Bake(bake) => {
                    println!("[SLOT-WORKER] >> Received instruction: BAKE for Slot {}. TaskDir: {:?}, Offset: {}", bake.slot_id, bake.task_dir, bake.offset);
                    let io_tx_inner = io_tx.clone();
                    tokio::task::spawn_blocking(move || {
                        let sid = bake.slot_id;
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let (t_dir, kv_n, off, is_relay) = (bake.task_dir, bake.kv_name, bake.offset, bake.is_relay_baking);
                            if bake.layers.is_empty() { 
                                println!("[SLOT-WORKER] !! Bake layers empty for Slot {}. Releasing.", sid);
                                let _ = SLOT_MANAGER.request_tx.blocking_send(SlotRequest::Release { idx: sid, task_id: None, block_index: None, is_bake: false }); 
                                return; 
                            }
                            let loop_count = bake.layers.len();
                            let slot = &SLOT_MANAGER.slots[sid];
                            slot.remaining_layers.store(1, Ordering::SeqCst); // Only 1 bundle task now
                            
                            let mut bundle_map = std::collections::HashMap::new();
                            let bundle_fname = if is_relay {
                                if off == 0 { format!("bundle_relay_kv.safetensors") }
                                else { format!("bundle_relay_kv_{}.safetensors", off) }
                            } else {
                                if off == 0 { format!("bundle_inference_kv.safetensors") }
                                else { format!("bundle_inference_kv_{}.safetensors", off) }
                            };

                            for l_idx in 0..loop_count {
                                let source = &bake.layers[l_idx];
                                let prefix = format!("l{}_", source.layer_idx);
                                
                                let k_res = source.k_tensor.to_device(&Device::Cpu).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>());
                                let v_res = source.v_tensor.to_device(&Device::Cpu).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>());
                                
                                let (k_data, v_data) = match (k_res, v_res) { 
                                    (Ok(k), Ok(v)) => (k, v), 
                                    _ => continue 
                                };

                                let k_shape = source.k_tensor.dims(); 
                                let (b, h, s, d) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
                                let head_size = s * d; 
                                let a_count = (0..s).filter(|&i| i < 4 || i % 8 == 0).count();
                                
                                let mut ka = vec![0.0f32; b * h * a_count * d]; 
                                let mut kp = vec![0u8; (b * h * s * d + 7) / 8]; 
                                let mut ks = vec![0.0f32; b * h * s];
                                
                                for bh in 0..(b * h) {
                                    let bho = bh * head_size;
                                    for i in 0..head_size { if k_data[bho + i] >= 0.0 { kp[(bho + i) / 8] |= 1 << ((bho + i) % 8); } }
                                    for ti in 0..s {
                                        let td = &k_data[bho + ti * d .. bho + (ti + 1) * d];
                                        if ti < 4 || ti % 8 == 0 { let ap = if ti < 4 { ti } else { 4 + (ti - 4) / 8 }; ka[(bh * a_count + ap) * d .. (bh * a_count + ap + 1) * d].copy_from_slice(td); }
                                        let mut m = 0.0f32; for &v in td { let a = v.abs(); if a > m { m = a; } } ks[bh * s + ti] = m;
                                    }
                                }
                                
                                bundle_map.insert(format!("{}k_anchors", prefix), Tensor::from_vec(ka, vec![b, h, a_count, d], &Device::Cpu).unwrap());
                                bundle_map.insert(format!("{}k_packed", prefix), Tensor::from_vec(kp, vec![(b * h * s * d + 7) / 8], &Device::Cpu).unwrap());
                                bundle_map.insert(format!("{}k_scales", prefix), Tensor::from_vec(ks, vec![b, h, s, 1], &Device::Cpu).unwrap());
                                bundle_map.insert(format!("{}k_shape", prefix), Tensor::from_vec(vec![b as u32, h as u32, s as u32, d as u32], (4,), &Device::Cpu).unwrap());
                                
                                let mut va = vec![0.0f32; b * h * a_count * d]; 
                                let mut vp = vec![0u8; (b * h * s * d + 7) / 8]; 
                                let mut vs = vec![0.0f32; b * h * s];
                                
                                for bh in 0..(b * h) {
                                    let bho = bh * head_size;
                                    for i in 0..head_size { if v_data[bho + i] >= 0.0 { vp[(bho + i) / 8] |= 1 << ((bho + i) % 8); } }
                                    for ti in 0..s {
                                        let td = &v_data[bho + ti * d .. bho + (ti + 1) * d];
                                        if ti < 4 || ti % 8 == 0 { let ap = if ti < 4 { ti } else { 4 + (ti - 4) / 8 }; va[(bh * a_count + ap) * d .. (bh * a_count + ap + 1) * d].copy_from_slice(td); }
                                        let mut m = 0.0f32; for &v in td { let a = v.abs(); if a > m { m = a; } } vs[bh * s + ti] = m;
                                    }
                                }
                                
                                bundle_map.insert(format!("{}v_anchors", prefix), Tensor::from_vec(va, vec![b, h, a_count, d], &Device::Cpu).unwrap());
                                bundle_map.insert(format!("{}v_packed", prefix), Tensor::from_vec(vp, vec![(b * h * s * d + 7) / 8], &Device::Cpu).unwrap());
                                bundle_map.insert(format!("{}v_scales", prefix), Tensor::from_vec(vs, vec![b, h, s, 1], &Device::Cpu).unwrap());
                            }
                            
                            bundle_map.insert("mode".to_string(), Tensor::from_vec(vec![3u32], (1,), &Device::Cpu).unwrap());
                            
                            let _ = io_tx_inner.blocking_send(SaveTask { 
                                slot_id: sid, 
                                path: t_dir.join(bundle_fname), 
                                tensors: bundle_map, 
                                is_last: true,
                                block_idx: bake.block_idx,
                                registry: bake.registry.clone()
                            });
                                
                        }));
                        
                        if let Err(panic_info) = result {
                            let msg = if let Some(s) = panic_info.downcast_ref::<&str>() { s.to_string() }
                                      else if let Some(s) = panic_info.downcast_ref::<String>() { s.clone() }
                                      else { "Unknown Panic Payload".to_string() };
                            println!("[WORKER-BAKE] !! [PANIC] Slot {} CRASHED! Cause: {}", sid, msg);
                            let _ = SLOT_MANAGER.request_tx.blocking_send(SlotRequest::Release { 
                                idx: sid, task_id: None, block_index: None, is_bake: true 
                            });
                        }
                    }).await.ok();
                }
                                SlotTask::Load(load) => {
                                    let (off, b_idx, s_id, target_path) = {
                                        match load.shared_block.inner.read() {
                                            Ok(inner) => {
                                                let path = inner.ssd_path.clone().or_else(|| {
                                                    if let Ok(reg) = load.registry.entries.read() {
                                                        if inner.index < reg.len() { reg[inner.index].ssd_path.clone() } else { None }
                                                    } else { None }
                                                });
                                                (inner.offset, inner.index, load.slot_id, path)
                                            },
                                            Err(e) => {
                                                println!("[WORKER-LOAD] !! Block Lock Poisoned: {:?}", e);
                                                (0, 999, load.slot_id, None)
                                            }
                                        }
                                    };
                
                                    if target_path.is_none() {
                                        println!("[WORKER-LOAD] !! Target path is None for Block {}. Releasing Slot {}.", b_idx, s_id);
                                        let _ = SLOT_MANAGER.request_tx.try_send(SlotRequest::Release { idx: s_id, task_id: None, block_index: None, is_bake: false });
                                        continue;
                                    }
                                    
                                    let target_path = target_path.unwrap();
                                    let l_idx = load.layer_idx;
                                    let registry = load.registry.clone();
                                    let shared_block = load.shared_block.clone();
                                    
                                    println!("[WORKER-LOAD] >> [START] Layer {} Block {} (Slot {}) from {:?}", l_idx, b_idx, s_id, target_path);
                                    
                                    tokio::spawn(async move {
                                        let res = async {
                                            let content = tokio::fs::read(&target_path).await
                                                .map_err(|e| format!("IO Error: {:?}", e))?;
                                            
                                            let st = safetensors::SafeTensors::deserialize(&content)
                                                .map_err(|e| format!("Deserialization Error: {:?}", e))?;
                
                                            // [FIX] DYNAMIC SHAPE DETECTION
                                                                                            // [BUNDLE-EXTRACT]
                                                                                            let prefix = format!("l{}_", l_idx);
                                                                                            let ka_name = format!("{}k_anchors", prefix);
                                                                                            let kh_name = format!("{}k_shape", prefix);
                                                                                            let kp_name = format!("{}k_packed", prefix);
                                                                                            let ks_name = format!("{}k_scales", prefix);
                                                                                            let va_name = format!("{}v_anchors", prefix);
                                                                                            let vp_name = format!("{}v_packed", prefix);
                                                                                            let vs_name = format!("{}v_scales", prefix);
                                            
                                                                                            let (ka_data, o_s, a_cnt) = if let Ok(t_view) = st.tensor(&ka_name) {
                                                                                                let dims = t_view.shape();
                                                                                                let data = unsafe { std::slice::from_raw_parts(t_view.data().as_ptr() as *const f32, t_view.data().len() / 4).to_vec() };
                                                                                                let s_len = if let Ok(v) = st.tensor(&kh_name) {
                                                                                                    let v_u32: &[u32] = unsafe { std::slice::from_raw_parts(v.data().as_ptr() as *const u32, v.data().len() / 4) };
                                                                                                    v_u32[2] as usize
                                                                                                } else { 256 };
                                                                                                (Some(data), vec![dims[0], dims[1], s_len, dims[3]], dims[2])
                                                                                            } else { (None, vec![1, 16, 256, 128], 36) };
                                                            
                                                                                            let ex_v = |name: &str| -> Option<Vec<f32>> { 
                                                                                                st.tensor(name).ok().map(|v| unsafe { std::slice::from_raw_parts(v.data().as_ptr() as *const f32, v.data().len() / 4).to_vec() }) 
                                                                                            };
                                                                                            
                                                                                            if let (Some(ka), Some(kp), Some(ks), Some(va), Some(vp), Some(vs)) = (ka_data, st.tensor(&kp_name).ok(), ex_v(&ks_name), ex_v(&va_name), st.tensor(&vp_name).ok(), ex_v(&vs_name)) {
                                                                                                let metadata = crate::models::qwen3vl::quantized_model::BitKVMetadata { 
                                                                                                    k_anchors: Tensor::from_vec(ka, (o_s[0], o_s[1], a_cnt, o_s[3]), &Device::Cpu).unwrap(), 
                                                                                                    k_packed: Tensor::from_slice(kp.data(), kp.shape(), &Device::Cpu).unwrap(), 
                                                                                                    k_scales: Tensor::from_vec(ks, (o_s[0], o_s[1], o_s[2], 1), &Device::Cpu).unwrap(), 
                                                                                                    v_anchors: Tensor::from_vec(va, (o_s[0], o_s[1], a_cnt, o_s[3]), &Device::Cpu).unwrap(), 
                                                                                                    v_packed: Tensor::from_slice(vp.data(), vp.shape(), &Device::Cpu).unwrap(), 
                                                                                                    v_scales: Tensor::from_vec(vs, (o_s[0], o_s[1], o_s[2], 1), &Device::Cpu).unwrap(), 
                                                                                                    original_shape: o_s 
                                                                                                };
                                                                                            if let Ok(mut inner) = shared_block.inner.write() {
                                                    inner.bitkv_metadata = Some(metadata.clone());
                                                    
                                                                                                        if let Ok(mut reg) = registry.entries.write() {
                                                                                                            if b_idx < reg.len() { 
                                                                                                                let is_relay = target_path.file_name().map(|n| n.to_string_lossy().contains("bundle_relay_kv")).unwrap_or(false);
                                                                                                                
                                                                                                                // [ONE-READ-ALL-LAYERS]
                                                                                                                let entry = &mut reg[b_idx];
                                                                                                                for target_l in 0..28 {
                                                                                                                    let prefix = format!("l{}_", target_l);
                                                                                                                    let ka_name = format!("{}k_anchors", prefix);
                                                                                                                    let kh_name = format!("{}k_shape", prefix);
                                                                                                                    let kp_name = format!("{}k_packed", prefix);
                                                                                                                    let ks_name = format!("{}k_scales", prefix);
                                                                                                                    let va_name = format!("{}v_anchors", prefix);
                                                                                                                    let vp_name = format!("{}v_packed", prefix);
                                                                                                                    let vs_name = format!("{}v_scales", prefix);
                                                    
                                                                                                                    if let (Ok(ka), Ok(kh), Ok(kp), Ok(ks), Ok(va), Ok(vp), Ok(vs)) = (
                                                                                                                        st.tensor(&ka_name), st.tensor(&kh_name), st.tensor(&kp_name), 
                                                                                                                        st.tensor(&ks_name), st.tensor(&va_name), st.tensor(&vp_name), st.tensor(&vs_name)
                                                                                                                    ) {
                                                                                                                        let dims = ka.shape();
                                                                                                                        let a_cnt = dims[2];
                                                                                                                        let v_u32: &[u32] = unsafe { std::slice::from_raw_parts(kh.data().as_ptr() as *const u32, kh.data().len() / 4) };
                                                                                                                        let o_s = vec![v_u32[0] as usize, v_u32[1] as usize, v_u32[2] as usize, v_u32[3] as usize];
                                                                                                                        
                                                                                                                        let metadata = crate::models::qwen3vl::quantized_model::BitKVMetadata { 
                                                                                                                            k_anchors: Tensor::from_slice(unsafe { std::slice::from_raw_parts(ka.data().as_ptr() as *const f32, ka.data().len() / 4) }, (o_s[0], o_s[1], a_cnt, o_s[3]), &Device::Cpu).unwrap(), 
                                                                                                                            k_packed: Tensor::from_slice(kp.data(), kp.shape(), &Device::Cpu).unwrap(), 
                                                                                                                            k_scales: Tensor::from_slice(unsafe { std::slice::from_raw_parts(ks.data().as_ptr() as *const f32, ks.data().len() / 4) }, (o_s[0], o_s[1], o_s[2], 1), &Device::Cpu).unwrap(), 
                                                                                                                            v_anchors: Tensor::from_slice(unsafe { std::slice::from_raw_parts(va.data().as_ptr() as *const f32, va.data().len() / 4) }, (o_s[0], o_s[1], a_cnt, o_s[3]), &Device::Cpu).unwrap(), 
                                                                                                                            v_packed: Tensor::from_slice(vp.data(), vp.shape(), &Device::Cpu).unwrap(), 
                                                                                                                            v_scales: Tensor::from_slice(unsafe { std::slice::from_raw_parts(vs.data().as_ptr() as *const f32, vs.data().len() / 4) }, (o_s[0], o_s[1], o_s[2], 1), &Device::Cpu).unwrap(), 
                                                                                                                            original_shape: o_s 
                                                                                                                        };
                                                                                                                        
                                                                                                                        {
                                                                                                                            let mut cache = entry.bitkv_cache.write().unwrap();
                                                                                                                            cache[target_l] = Some(metadata);
                                                                                                                        }
                                                                                                                        
                                                                                                                        if entry.location[target_l] == crate::models::qwen3vl::quantized_model::KVLocation::SSD {
                                                                                                                            entry.location[target_l] = crate::models::qwen3vl::quantized_model::KVLocation::RAM;
                                                                                                                        }
                                                                                                                    }
                                                                                                                }
                                                                                                                
                                                                                                                entry.slot_ids[l_idx] = Some(s_id);
                                                                                                                if is_relay {
                                                                                                                    println!("[WORKER-LOAD] << [SUCCESS] RELAY Bundle Block {} fully cached (Slot {}).", b_idx, s_id);
                                                                                                                } else {
                                                                                                                    println!("[WORKER-LOAD] << [SUCCESS] Bundle Block {} fully cached for ALL layers (Requested L{} in Slot {}).", b_idx, l_idx, s_id);
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                    
                                                }
                                                Ok(())
                                            } else {
                                                Err("Missing tensors in safetensors file".to_string())
                                            }
                                        }.await;
                
                                        if let Err(e) = res {
                                            println!("[WORKER-LOAD] !! [ERROR] Layer {} Block {} failed: {} (Slot {})", l_idx, b_idx, e, s_id);
                                            SLOT_MANAGER.release_slot(s_id).await;
                                        }
                                        // SLOT_MANAGER.release_slot(s_id).await; // Removed for Reservation
                                    });
                                }
                                SlotTask::ChunkedLoad(chunk) => {
                                    let registry = chunk.registry.clone();
                                    let num_blocks = chunk.layer_indices.len();
                                    println!("[WORKER-CHUNK] >> [START] CHUNKED LOAD for {} blocks.", num_blocks);
                                    
                                    for i in 0..num_blocks {
                                        let (l_idx, s_id, block) = (chunk.layer_indices[i], chunk.slot_ids[i], chunk.shared_blocks[i].clone());
                                        let (b_idx, target_path) = {
                                            match block.inner.read() {
                                                Ok(inner) => {
                                                    let path = inner.ssd_path.clone().or_else(|| {
                                                        if let Ok(reg) = registry.entries.read() {
                                                            if inner.index < reg.len() { reg[inner.index].ssd_path.clone() } else { None }
                                                        } else { None }
                                                    });
                                                    (inner.index, path)
                                                },
                                                Err(e) => {
                                                    println!("[WORKER-CHUNK] !! Block {} Lock Poisoned: {:?}", i, e);
                                                    (999, None)
                                                }
                                            }
                                        };
                
                                        if target_path.is_none() {
                                            println!("[WORKER-CHUNK] !! No path for Block {} Index {}. Releasing Slot {}.", i, b_idx, s_id);
                                            SLOT_MANAGER.release_slot(s_id).await;
                                            continue;
                                        }
                                        
                                        let target_path = target_path.unwrap();
                                        let reg_inner = registry.clone();
                
                                        tokio::spawn(async move {
                                            let res = async {
                                                let content = tokio::fs::read(&target_path).await
                                                    .map_err(|e| format!("IO Error: {:?}", e))?;
                                                
                                                let st = safetensors::SafeTensors::deserialize(&content)
                                                    .map_err(|e| format!("Deserialization Error: {:?}", e))?;
                
                                                // [BUNDLE-EXTRACT]
                                                let prefix = format!("l{}_", l_idx);
                                                let ka_name = format!("{}k_anchors", prefix);
                                                let kh_name = format!("{}k_shape", prefix);
                                                let kp_name = format!("{}k_packed", prefix);
                                                let ks_name = format!("{}k_scales", prefix);
                                                let va_name = format!("{}v_anchors", prefix);
                                                let vp_name = format!("{}v_packed", prefix);
                                                let vs_name = format!("{}v_scales", prefix);

                                                let (ka_data, o_s, a_cnt) = if let Ok(t_view) = st.tensor(&ka_name) {
                                                    let dims = t_view.shape();
                                                    let data = unsafe { std::slice::from_raw_parts(t_view.data().as_ptr() as *const f32, t_view.data().len() / 4).to_vec() };
                                                    let s_len = if let Ok(v) = st.tensor(&kh_name) {
                                                        let v_u32: &[u32] = unsafe { std::slice::from_raw_parts(v.data().as_ptr() as *const u32, v.data().len() / 4) };
                                                        v_u32[2] as usize
                                                    } else { 256 };
                                                    (Some(data), vec![dims[0], dims[1], s_len, dims[3]], dims[2])
                                                } else { (None, vec![1, 16, 256, 128], 36) };
                
                                                let ex_v = |name: &str| -> Option<Vec<f32>> { 
                                                    st.tensor(name).ok().map(|v| unsafe { std::slice::from_raw_parts(v.data().as_ptr() as *const f32, v.data().len() / 4).to_vec() }) 
                                                };
                                                
                                                if let (Some(ka), Some(kp), Some(ks), Some(va), Some(vp), Some(vs)) = (ka_data, st.tensor(&kp_name).ok(), ex_v(&ks_name), ex_v(&va_name), st.tensor(&vp_name).ok(), ex_v(&vs_name)) {
                                                    let metadata = crate::models::qwen3vl::quantized_model::BitKVMetadata { 
                                                        k_anchors: Tensor::from_vec(ka, (o_s[0], o_s[1], a_cnt, o_s[3]), &Device::Cpu).unwrap(), 
                                                        k_packed: Tensor::from_slice(kp.data(), kp.shape(), &Device::Cpu).unwrap(), 
                                                        k_scales: Tensor::from_vec(ks, (o_s[0], o_s[1], o_s[2], 1), &Device::Cpu).unwrap(), 
                                                        v_anchors: Tensor::from_vec(va, (o_s[0], o_s[1], a_cnt, o_s[3]), &Device::Cpu).unwrap(), 
                                                        v_packed: Tensor::from_slice(vp.data(), vp.shape(), &Device::Cpu).unwrap(), 
                                                        v_scales: Tensor::from_vec(vs, (o_s[0], o_s[1], o_s[2], 1), &Device::Cpu).unwrap(), 
                                                        original_shape: o_s 
                                                    };
                
                                                    if let Ok(mut inner) = block.inner.write() {
                                                        inner.bitkv_metadata = Some(metadata.clone());
                                                                if let Ok(mut reg) = reg_inner.entries.write() {
                                                            if b_idx < reg.len() { 
                                                                let is_relay = target_path.file_name().map(|n| n.to_string_lossy().contains("bundle_relay_kv")).unwrap_or(false);
                                                                
                                                                // [ONE-READ-ALL-LAYERS]
                                                                let entry = &mut reg[b_idx];
                                                                for target_l in 0..28 {
                                                                    let prefix = format!("l{}_", target_l);
                                                                    let ka_name = format!("{}k_anchors", prefix);
                                                                    let kh_name = format!("{}k_shape", prefix);
                                                                    let kp_name = format!("{}k_packed", prefix);
                                                                    let ks_name = format!("{}k_scales", prefix);
                                                                    let va_name = format!("{}v_anchors", prefix);
                                                                    let vp_name = format!("{}v_packed", prefix);
                                                                    let vs_name = format!("{}v_scales", prefix);

                                                                    if let (Ok(ka), Ok(kh), Ok(kp), Ok(ks), Ok(va), Ok(vp), Ok(vs)) = (
                                                                        st.tensor(&ka_name), st.tensor(&kh_name), st.tensor(&kp_name), 
                                                                        st.tensor(&ks_name), st.tensor(&va_name), st.tensor(&vp_name), st.tensor(&vs_name)
                                                                    ) {
                                                                        let dims = ka.shape();
                                                                        let a_cnt = dims[2];
                                                                        let v_u32: &[u32] = unsafe { std::slice::from_raw_parts(kh.data().as_ptr() as *const u32, kh.data().len() / 4) };
                                                                        let o_s = vec![v_u32[0] as usize, v_u32[1] as usize, v_u32[2] as usize, v_u32[3] as usize];
                                                                        
                                                                        let metadata = crate::models::qwen3vl::quantized_model::BitKVMetadata { 
                                                                            k_anchors: Tensor::from_slice(unsafe { std::slice::from_raw_parts(ka.data().as_ptr() as *const f32, ka.data().len() / 4) }, (o_s[0], o_s[1], a_cnt, o_s[3]), &Device::Cpu).unwrap(), 
                                                                            k_packed: Tensor::from_slice(kp.data(), kp.shape(), &Device::Cpu).unwrap(), 
                                                                            k_scales: Tensor::from_slice(unsafe { std::slice::from_raw_parts(ks.data().as_ptr() as *const f32, ks.data().len() / 4) }, (o_s[0], o_s[1], o_s[2], 1), &Device::Cpu).unwrap(), 
                                                                            v_anchors: Tensor::from_slice(unsafe { std::slice::from_raw_parts(va.data().as_ptr() as *const f32, va.data().len() / 4) }, (o_s[0], o_s[1], a_cnt, o_s[3]), &Device::Cpu).unwrap(), 
                                                                            v_packed: Tensor::from_slice(vp.data(), vp.shape(), &Device::Cpu).unwrap(), 
                                                                            v_scales: Tensor::from_slice(unsafe { std::slice::from_raw_parts(vs.data().as_ptr() as *const f32, vs.data().len() / 4) }, (o_s[0], o_s[1], o_s[2], 1), &Device::Cpu).unwrap(), 
                                                                            original_shape: o_s 
                                                                        };
                                                                        
                                                                        {
                                                                            let mut cache = entry.bitkv_cache.write().unwrap();
                                                                            cache[target_l] = Some(metadata);
                                                                        }
                                                                        
                                                                        if entry.location[target_l] == crate::models::qwen3vl::quantized_model::KVLocation::SSD {
                                                                            entry.location[target_l] = crate::models::qwen3vl::quantized_model::KVLocation::RAM;
                                                                        }
                                                                    }
                                                                }
                                                                
                                                                if is_relay {
                                                                    println!("[WORKER-CHUNK] << [SUCCESS] RELAY Bundle Block {} fully cached (Slot {}).", b_idx, s_id);
                                                                } else {
                                                                    println!("[WORKER-CHUNK] << [SUCCESS] Bundle Block {} fully cached into RAM for ALL layers (Requested L{} in Slot {}).", b_idx, l_idx, s_id);
                                                                }
                                                            }
                                                        }
                                                    }
                                                    Ok(())
                                                } else {
                                                    Err(format!("Missing Layer {} tensors in bundle", l_idx))
                                                }
                                            }.await;
                
                                            if let Err(e) = res {
                                                println!("[WORKER-CHUNK] !! [ERROR] Layer {} Block {} failed: {} (Slot {})", l_idx, b_idx, e, s_id);
                                                SLOT_MANAGER.release_slot(s_id).await;
                                            }
                                            // SLOT_MANAGER.release_slot(s_id).await; // Removed for Reservation
                                        });
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
        match self { 
            Self::Standard(m) => m.forward(i, pv, ithw, vpv, vthw, cp, off), 
            Self::QuantizedVL(m) => m.forward(i, pv, ithw, vpv, vthw, cp, off, tl, sid), 
            Self::QuantizedText(m) => m.forward(i, cp, off, tl, sid) 
        }
    }
    pub fn rebalance_layers(&mut self, d: usize, target_idx: usize) -> Result<()> { 
        match self { 
            Self::Standard(_) => Ok(()), 
            Self::QuantizedVL(m) => m.rebalance_layers(d, target_idx), 
            Self::QuantizedText(m) => m.rebalance_layers(d, target_idx) 
        } 
    }
    pub fn drop_kv_storage(&mut self) -> Result<()> { match self { Self::Standard(_) => Ok(()), Self::QuantizedVL(m) => m.language_model.drop_kv_storage(), Self::QuantizedText(m) => m.language_model.drop_kv_storage() } }
    pub fn save_metadata_to_file(&self, path: &Path) -> Result<()> {
        match self {
            Self::QuantizedVL(m) => m.language_model.registry.save_to_file(path),
            Self::QuantizedText(m) => m.language_model.registry.save_to_file(path),
            _ => Ok(())
        }
    }
    pub fn load_metadata_from_file(&self, path: &Path) -> Result<()> {
        match self {
            Self::QuantizedVL(m) => m.language_model.registry.load_from_file(path),
            Self::QuantizedText(m) => m.language_model.registry.load_from_file(path),
            _ => Ok(())
        }
    }
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
        let start_prep = std::time::Instant::now();
        if sid.is_none() { SLOT_MANAGER.reset_all_slots().await; }
        
        // [LOG] Start of processing
        let raw_prompt_len = mes.messages.iter().map(|m| match m {
            crate::openai_types::ChatCompletionRequestMessage::User(u) => match &u.content {
                crate::openai_types::ChatCompletionRequestUserMessageContent::Text(t) => t.len(),
                _ => 0,
            },
            crate::openai_types::ChatCompletionRequestMessage::System(s) => s.content.len(),
            _ => 0,
        }).sum::<usize>();
        println!("[GENERATE] >> [START] Preparing input (Raw chars: {})...", raw_prompt_len);

        let m_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &m_render)?;
        let f_ids = self.tokenizer.text_encode_vec(input.replace_text, false)?;
        
        let t_toks = f_ids.len();
        println!("[TIMER] Tokenization & Pre-processing: {:.2}s for {} tokens", start_prep.elapsed().as_secs_f32(), t_toks);
        
        let is_06b = self.model_name.contains("0.6B");

        if let Some(flag) = &cancel { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }

        // [HORIZONTAL-CALL] Entire sequence is now handled layer-by-layer internally
        println!("[BAKING] Starting Horizontal Pass for {} tokens...", t_toks);
        self.qwen3_vl.forward(
            &Tensor::from_vec(f_ids.clone(), (1, t_toks), &self.text_device)?, 
            None, None, None, None, 
            Some(&Tensor::arange(0u32, t_toks as u32, &self.text_device)?.unsqueeze(0)?), 
            0, t_toks, sid.clone()
        )?;

        if let Some(s_id) = &sid {
            let unbaked_blocks = self.get_all_unbaked_kv_blocks();
            for (ks, vs, block_offset) in unbaked_blocks {
                let slot_id = SLOT_MANAGER.acquire_write_slot(t_toks).await;
                let path = crate::utils::paths::get_kv_dir(None).join(s_id);
                if !path.exists() { let _ = fs::create_dir_all(&path); }
                
                let mut dumps = Vec::new();
                for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() {
                    let k: Tensor = k; let v: Tensor = v;
                    dumps.push(LayerKVDump { layer_idx: idx, k_tensor: k, v_tensor: v });
                }
                if let Ok(tx) = get_bake_worker().await {
                    let reg_ref = match &self.qwen3_vl {
                        ModelVariant::QuantizedVL(m) => Some(m.language_model.registry.clone()),
                        ModelVariant::QuantizedText(m) => Some(m.language_model.registry.clone()),
                        _ => None
                    };
                    let _ = tx.send(SlotTask::Bake(BakeTask { 
                        slot_id, task_dir: path, kv_name: kv_n.clone(), offset: block_offset, 
                        layers: dumps, is_relay_baking: is_06b,
                        block_idx: None,
                        registry: reg_ref
                    })).await;
                }
            }
            // [STRICT-FLUSH] Clear the SSD road before 2B starts reading
            println!("[BAKING] All blocks queued. Waiting for SSD Flush (Crucial for 2B Performance)...");
            SLOT_MANAGER.wait_for_all_tasks().await;
            
            // [METADATA-PERSISTENCE] Save registry to file for cross-layer/cross-session reliability
            let path = crate::utils::paths::get_kv_dir(None).join(s_id);
            if let Err(e) = self.qwen3_vl.save_metadata_to_file(&path) {
                println!("[BAKING] !! Metadata Save Failed: {:?}", e);
            } else {
                println!("[BAKING] Metadata persisted to {:?}", path.join("metadata.json"));
            }

            println!("[BAKING] SSD Flush Complete. Road cleared.");
            self.clear_temporal_kv_caches();
        }
        Ok(t_toks)
    }

    pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel: Option<Arc<AtomicBool>>, sid: Option<String>, kv_n: Option<String>) -> Result<String> {
        let start_prep = std::time::Instant::now();
        if sid.is_none() { SLOT_MANAGER.reset_all_slots().await; }
        
        // [RESTORED] Logit processor for sampling next tokens
        let mut l_proc = get_logit_processor(Some(mes.temperature.unwrap_or(0.7) as f32), Some(mes.top_p.unwrap_or(0.9) as f32), Some(40), mes.seed.unwrap_or(34562) as u64);

        let m_render = self.chat_template.apply_chat_template(&mes)?; 
        let mut input = self.pre_processor.process_info(&mes, &m_render)?;
        let f_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?; 
        
        let t_toks = f_ids.len();
        println!("[TIMER] Input Rendering & Tokenization: {:.2}s for {} tokens", start_prep.elapsed().as_secs_f32(), t_toks);
        
        let mut a_ids = f_ids.clone();
        let s_off = self.get_kv_len();

        if t_toks > s_off {
            let prefill_len = t_toks - s_off;
            println!("[GENERATE] Starting Horizontal Prefill for {} tokens...", prefill_len);
            self.qwen3_vl.forward(
                &Tensor::from_vec(f_ids[s_off..].to_vec(), (1, prefill_len), &self.text_device)?, 
                None, None, None, None, 
                Some(&Tensor::arange(s_off as u32, t_toks as u32, &self.text_device)?.unsqueeze(0)?), 
                s_off, t_toks, sid.clone()
            )?;

            if let Some(s_id) = &sid {
                let unbaked_blocks = self.get_all_unbaked_kv_blocks();
                for (ks, vs, block_offset) in unbaked_blocks {
                    let slot_id = SLOT_MANAGER.acquire_write_slot(t_toks).await;
                    let mut dumps = Vec::new();
                    for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() {
                        let k: Tensor = k; let v: Tensor = v;
                        dumps.push(LayerKVDump { layer_idx: idx, k_tensor: k, v_tensor: v });
                    }
                    if let Ok(tx) = get_bake_worker().await {
                        let _ = tx.send(SlotTask::Bake(BakeTask { 
                            slot_id, 
                            task_dir: crate::utils::paths::get_kv_dir(None).join(s_id), 
                            kv_name: kv_n.clone(), 
                            offset: block_offset, 
                            layers: dumps, 
                            is_relay_baking: false,
                            block_idx: None,
                            registry: None
                        })).await;
                    }
                }
                SLOT_MANAGER.wait_for_all_tasks().await;

                // [METADATA-PERSISTENCE] Save registry to file for cross-layer/cross-session reliability
                let path = crate::utils::paths::get_kv_dir(None).join(s_id);
                if let Err(e) = self.qwen3_vl.save_metadata_to_file(&path) {
                    println!("[GENERATE] !! Metadata Save Failed: {:?}", e);
                } else {
                    println!("[GENERATE] Metadata persisted to {:?}", path.join("metadata.json"));
                }

                self.clear_temporal_kv_caches();
            }
        }
        
        let mut curr_s_off = t_toks;
        let (mut p_vals, i_grid, mut g_text) = (input.pixel_values.take(), input.image_grid_thw.take(), String::new());
        let mut tokens_since_last_bake = 0;

        for _ in 0..mes.max_tokens.unwrap_or(2048) {
            if let Some(flag) = &cancel { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            
            let logits = self.qwen3_vl.forward(&Tensor::new(vec![*a_ids.last().unwrap()], &self.text_device)?.unsqueeze(0)?, p_vals.as_ref(), i_grid.as_ref(), None, None, Some(&Tensor::arange(curr_s_off as u32, (curr_s_off + 1) as u32, &self.text_device)?.unsqueeze(0)?), curr_s_off, t_toks, sid.clone())?;
            let l_vec = apply_repeat_penalty(&logits.squeeze(0)?.i(logits.dim(1)? - 1)?.to_dtype(DType::F32)?, 1.1, if a_ids.len() > 512 { &a_ids[a_ids.len()-512..] } else { &a_ids[..] })?;
            let next_id = l_proc.sample(&l_vec)?; if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            a_ids.push(next_id); g_text.push_str(&self.tokenizer.token_decode(vec![next_id])?);
            
            curr_s_off += 1; 
            p_vals = None; 

            // [RELAY-BAKING] 릴레이 시스템에 의해 SSD_PENDING으로 분류된 블록들을 찾아 비동기 저장
            if let Some(s_id) = &sid {
                let pending_blocks = self.get_blocks_by_location(KVLocation::SSD_PENDING)?;
                if !pending_blocks.is_empty() {
                    let reg_ref = match &self.qwen3_vl {
                        ModelVariant::QuantizedVL(m) => Some(m.language_model.registry.clone()),
                        ModelVariant::QuantizedText(m) => Some(m.language_model.registry.clone()),
                        _ => None
                    };

                    for (ks, vs, block_offset, block_idx) in pending_blocks {
                        // [LOG] 1단계: 요청 기록
                        println!("[GENERATE] >> [REQUEST] Sending Block {} to SSD Baker (Offset: {})", block_idx, block_offset);
                        
                        let slot_id = SLOT_MANAGER.acquire_write_slot(curr_s_off).await;
                        let mut dumps = Vec::new();
                        for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() {
                            dumps.push(LayerKVDump { layer_idx: idx, k_tensor: k, v_tensor: v });
                        }
                        if let Ok(tx) = get_bake_worker().await {
                            let path = crate::utils::paths::get_kv_dir(None).join(s_id);
                            let reg_ref = match &self.qwen3_vl {
                                ModelVariant::QuantizedVL(m) => Some(m.language_model.registry.clone()),
                                ModelVariant::QuantizedText(m) => Some(m.language_model.registry.clone()),
                                _ => None
                            };
                            let _ = tx.send(SlotTask::Bake(BakeTask { 
                                slot_id, task_dir: path, kv_name: kv_n.clone(), offset: block_offset, 
                                layers: dumps, is_relay_baking: false,
                                block_idx: Some(block_idx),
                                registry: reg_ref
                            })).await;
                        }
                        
                        // 저장 요청을 보냈으므로 상태를 업데이트하여 중복 요청 방지
                        self.mark_block_location(block_idx, KVLocation::Streaming); 
                    }
                }
            }

            self.clear_temporal_kv_caches();
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

            pub fn get_blocks_by_location(&self, target_loc: KVLocation) -> Result<Vec<(Vec<Tensor>, Vec<Tensor>, usize, usize)>> {
                let mut results = Vec::new();
                let layers = match &self.qwen3_vl {
                    ModelVariant::QuantizedVL(m) => Some(&m.language_model.layers),
                    ModelVariant::QuantizedText(m) => Some(&m.language_model.layers),
                    _ => None
                };
                if let Some(layers) = layers {
                    if layers.is_empty() { return Ok(vec![]); }
                    let num_blocks = layers[0].self_attn.kv_blocks.len();
                    for b_idx in 0..num_blocks {
                        let mut ks = Vec::new();
                        let mut vs = Vec::new();
                        let mut match_found = false;
                        let mut offset = 0;
                                        for l in layers {
                                            let inner = l.self_attn.kv_blocks[b_idx].inner.read().unwrap();
                                            // target_loc이거나, 혹은 이미 RAM으로 핸드오프된 데이터를 가져옵니다.
                                            if inner.location == target_loc || (target_loc == KVLocation::SSD_PENDING && inner.location == KVLocation::RAM && inner.ssd_path.is_none()) {
                                                if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                                                    // 이미 CPU에 있다면 그대로, GPU에 있다면 CPU로 즉시 복사하여 Worker에게 전달
                                                    let k_cpu = if k.device().is_cpu() { k.clone() } else { k.to_device(&Device::Cpu)? };
                                                    let v_cpu = if v.device().is_cpu() { v.clone() } else { v.to_device(&Device::Cpu)? };
                                                    ks.push(k_cpu); 
                                                    vs.push(v_cpu);
                                                    offset = inner.offset + inner.len;
                                                    match_found = true;
                                                }
                                            }
                                        }
                        
                        if match_found { results.push((ks, vs, offset, b_idx)); }
                    }
                }
                Ok(results)
            }
        
    
        pub fn mark_block_location(&mut self, block_idx: usize, new_loc: KVLocation) {
            let layers = match self.qwen3_vl {
                ModelVariant::QuantizedVL(ref mut m) => &mut m.language_model.layers,
                ModelVariant::QuantizedText(ref mut m) => &mut m.language_model.layers,
                _ => return
            };
            for l in layers {
                if block_idx < l.self_attn.kv_blocks.len() {
                    let mut inner = l.self_attn.kv_blocks[block_idx].inner.write().unwrap();
                    inner.location = new_loc;
                }
            }
        }
    
        pub fn get_kv_len(&self) -> usize { 
     match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.get_kv_len(), ModelVariant::QuantizedText(m) => m.language_model.get_kv_len(), _ => 0 } }
    pub fn get_all_unbaked_kv_blocks(&self) -> Vec<(Vec<Tensor>, Vec<Tensor>, usize)> {
        let mut all_blocks: Vec<(Vec<Tensor>, Vec<Tensor>, usize)> = Vec::new();
        let layers: Option<&Vec<crate::models::qwen3vl::quantized_model::QuantizedQwen3VLTextDecoderLayer>> = match &self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => Some(&m.language_model.layers),
            ModelVariant::QuantizedText(m) => Some(&m.language_model.layers),
            _ => None
        };
        if let Some(layers) = layers {
            if layers.is_empty() { return vec![]; }
            let num_blocks = layers[0].self_attn.kv_blocks.len();
            for b_idx in 0..num_blocks {
                let mut ks: Vec<Tensor> = Vec::new();
                let mut vs: Vec<Tensor> = Vec::new();
                let mut has_data = false;
                let mut end_offset = 0;
                for l in layers {
                    if b_idx < l.self_attn.kv_blocks.len() {
                        let inner = l.self_attn.kv_blocks[b_idx].inner.read().unwrap();
                        if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                            ks.push(k.clone()); vs.push(v.clone());
                            end_offset = inner.offset + inner.len;
                            has_data = true;
                        }
                    }
                }
                if has_data { all_blocks.push((ks, vs, end_offset)); }
            }
        }
        all_blocks
    }
    pub fn save_kv_to_disk(&mut self, p: &Path, n: Option<&str>, o: usize) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.save_kv_cache(p, false, o, n), ModelVariant::QuantizedText(m) => m.save_kv_cache(p, false, o, n), _ => Ok(()) } }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.to_device(d)?, ModelVariant::QuantizedText(m) => m.to_device(d)?, _ => {} } self.text_device = d.clone(); self.vision_device = d.clone(); Ok(()) }
    pub fn drop_kv_storage(&mut self) -> Result<()> { self.qwen3_vl.drop_kv_storage() }
    pub fn clear_kv_cache(&mut self) { self.clear_temporal_kv_caches(); }
    pub fn truncate_kv_cache(&mut self, l: usize) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.truncate_kv_cache(l), ModelVariant::QuantizedText(m) => m.truncate_kv_cache(l), _ => Ok(()) } }
    pub fn load_kv_from_disk(&mut self, p: &Path, n: Option<&str>) -> Result<()> { 
        let _ = self.qwen3_vl.load_metadata_from_file(p);
        match &mut self.qwen3_vl { 
            ModelVariant::QuantizedVL(m) => m.load_kv_cache(p, &self.text_device, 0, 128, n), 
            ModelVariant::QuantizedText(m) => m.load_kv_cache(p, &self.text_device, 0, 128, n), 
            _ => Ok(()) 
        } 
    }
}

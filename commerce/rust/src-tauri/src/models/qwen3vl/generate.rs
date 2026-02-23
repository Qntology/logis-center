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
        
        // [OPTIMIZATION] sysinfo 초기화는 한 번만 수행하여 오버헤드 제거
        let mut sys = sysinfo::System::new(); 
        sys.refresh_memory();
        
        println!("[INIT] SlotManager Dispatcher is now online.");

        while let Some(req) = rx.recv().await {
            match req {
                SlotRequest::AcquireWrite { response, total_tokens } => {
                    // 주기적으로만 메모리 정보 갱신 (매번 하면 느려짐)
                    if cf.load(Ordering::SeqCst) % 8 == 0 { sys.refresh_memory(); }
                    
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
                        
                        let (rc, wc) = (cr.load(Ordering::SeqCst), cw.load(Ordering::SeqCst));
                        if wc % 5 == 0 || wc < 5 {
                            println!("[SLOTS] Released slot {}. Active: (W:{}, R:{})", idx, wc, rc);
                        }
                    }
                    // [FIX] 카운트 업데이트 직후 즉시 Flush 대기자들을 확인
                    if cw.load(Ordering::SeqCst) == 0 && cr.load(Ordering::SeqCst) == 0 { 
                        while let Some(w) = flushers.pop() { let _ = w.send(()); } 
                    }
                    Self::process_queues_robust(&sys, &mut free_p, &mut p_writes, &mut p_reads, &mut states, &cf, &cw, &cr, max_slots);
                }
                SlotRequest::Flush { response } => {
                    let (rc, wc) = (cr.load(Ordering::SeqCst), cw.load(Ordering::SeqCst));
                    // [FIX] 요청을 받는 즉시 카운트가 0이면 대기 없이 응답
                    if wc == 0 && rc == 0 { 
                        let _ = response.send(()); 
                    } else { 
                        println!("[SLOTS] Flush requested. Waiting for (W:{}, R:{}) tasks...", wc, rc);
                        flushers.push(response); 
                    }
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

    fn calculate_dynamic_budget(_sys: &sysinfo::System, _total_tokens: usize, max_slots: usize) -> usize {
        // [VRAM-AWARE-BUDGET] 시스템 RAM 대신 VRAM 여유 공간을 기준으로 슬롯 개수 제한
        use nvml_wrapper::Nvml;
        let mut free_vram_gb = 1.0; // 기본값
        
        if let Ok(nvml) = Nvml::init() {
            if let Ok(dev) = nvml.device_by_index(0) { // GPU 0번 기준
                if let Ok(mem) = dev.memory_info() {
                    free_vram_gb = mem.free as f64 / 1024.0 / 1024.0 / 1024.0;
                }
            }
        }

        // 0.6B 모델 기준 1슬롯(1024토큰)은 약 0.22GB 점유
        // 여유 VRAM의 절반 정도만 슬롯에 할당하여 연산 버퍼 확보
        let safe_vram = (free_vram_gb - 0.5).max(0.0); // 안전 마진 500MB
        let budget = (safe_vram / 0.22) as usize;
        
        // 4GB GPU 기준 보통 8~12개 슬롯으로 제한됨
        let final_budget = budget.clamp(4, 16).min(max_slots);
        
        if _total_tokens % 1024 == 0 {
            println!("[SLOTS] Dynamic Budget updated: {} slots based on {:.2}GB free VRAM", final_budget, free_vram_gb);
        }
        
        final_budget
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
            // [FIX] std::sync::Mutex용 동기 락 사용
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
    // 1. 모든 슬롯 작업(압축/큐잉) 완료 대기
    let _ = SLOT_MANAGER.wait_for_all_tasks().await;
    
    // 2. 모든 물리적 디스크 쓰기 완료 대기
    let mut attempts = 0;
    while GLOBAL_IO_COUNTER.load(Ordering::SeqCst) > 0 && attempts < 500 {
        if attempts % 50 == 0 { println!("[IO-WAIT] Waiting for {} pending disk writes...", GLOBAL_IO_COUNTER.load(Ordering::SeqCst)); }
        tokio::time::sleep(Duration::from_millis(50)).await; 
        attempts += 1;
    }
    println!("[IO-WAIT] Global IO Sync Complete.");
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
    // [ROBUST-WAIT] 워커가 준비될 때까지 최대 10초간 대기 (초기 로딩 지연 대응)
    for _ in 0..100 { 
        if let Some(tx) = BAKE_TX.get() { return Ok(tx.clone()); } 
        tokio::time::sleep(std::time::Duration::from_millis(100)).await; 
    }
    Err(anyhow!("Bake worker timeout - Check if init_bake_worker was called"))
}

pub async fn get_load_worker() -> Result<mpsc::Sender<SlotTask>> {
    for _ in 0..100 { 
        if let Some(tx) = LOAD_TX.get() { return Ok(tx.clone()); } 
        tokio::time::sleep(std::time::Duration::from_millis(100)).await; 
    }
    Err(anyhow!("Load worker timeout"))
}

pub fn init_bake_worker() {
    let (btx, brx) = mpsc::channel(1000); let (ltx, lrx) = mpsc::channel(1000);
    
    // [CRITICAL] 채널을 먼저 설정하여 게터가 즉시 유효한 송신기를 반환할 수 있게 함
    let _ = BAKE_TX.set(btx); 
    let _ = LOAD_TX.set(ltx);
    
    println!("[INIT] Bake/Load Workers initialized and ready.");
    
    tauri::async_runtime::spawn(async move { spawn_slot_worker(brx); }); 
    tauri::async_runtime::spawn(async move { spawn_slot_worker(lrx); });
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
                    let (sid, t_dir, off, is_relay) = (bake.slot_id, bake.task_dir, bake.offset, bake.is_relay_baking);
                    
                    tokio::task::spawn_blocking(move || {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            if bake.layers.is_empty() { 
                                let _ = SLOT_MANAGER.request_tx.blocking_send(SlotRequest::Release { idx: sid, task_id: None, block_index: None, is_bake: true }); 
                                return; 
                            }
                            let loop_count = bake.layers.len(); 
                            let slot = &SLOT_MANAGER.slots[sid]; 
                            slot.remaining_layers.store(loop_count, Ordering::SeqCst); 
                            
                            let block_dir = t_dir.join(format!("b{}", off)); 
                            let block_prefix = format!("b{}_", off);
                            
                            for l_idx in 0..loop_count {
                                let src = &bake.layers[l_idx];
                                let act_l = if is_relay && loop_count == 1 { 0 } else { src.layer_idx };
                                let prefix = format!("{}l{}_", block_prefix, act_l);
                                
                                let k_res = src.k_tensor.to_device(&Device::Cpu).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>());
                                let v_res = src.v_tensor.to_device(&Device::Cpu).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>());
                                
                                let (kd, vd) = match (k_res, v_res) { 
                                    (Ok(k), Ok(v)) => (k, v), 
                                    (e1, e2) => {
                                        println!("[WORKER-ERR] Block {} Tensor transfer failed! K_err: {:?}, V_err: {:?}", off/256, e1.err(), e2.err());
                                        // [CRITICAL] 실패 시에도 카운트를 깎고 마지막이면 슬롯 해제
                                        if slot.remaining_layers.fetch_sub(1, Ordering::SeqCst) == 1 {
                                            let _ = SLOT_MANAGER.request_tx.blocking_send(SlotRequest::Release { idx: sid, task_id: None, block_index: None, is_bake: true });
                                        }
                                        continue; 
                                    }
                                };

                                let ks = src.k_tensor.dims(); 
                                let (b, h, s, d) = (ks[0], ks[1], ks[2], ks[3]);
                                if s == 0 {
                                    println!("[WORKER-SKIP] Block {} has 0 tokens.", off/256);
                                    if slot.remaining_layers.fetch_sub(1, Ordering::SeqCst) == 1 {
                                        let _ = SLOT_MANAGER.request_tx.blocking_send(SlotRequest::Release { idx: sid, task_id: None, block_index: None, is_bake: true });
                                    }
                                    continue;
                                }

                                let mut layer_map = HashMap::new();
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
                                let save_path = block_dir.join(format!("l{}.st", act_l));
                                let _ = io_tx_inner.blocking_send(SaveTask { slot_id: sid, path: save_path.clone(), tensors: layer_map, is_last: l_idx == loop_count - 1, block_idx: bake.block_idx, registry: bake.registry.clone() });
                                println!("[SSD-BAKE-QUEUED] Block {}, Layer {}", off / 256, act_l);
                            }
                        }));
                        if let Err(p) = result { 
                            println!("[WORKER-PANIC] !! Error in BakeTask: {:?}", p); 
                            let _ = SLOT_MANAGER.request_tx.blocking_send(SlotRequest::Release { idx: sid, task_id: None, block_index: None, is_bake: true }); 
                        }
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
                SlotTask::ChunkedLoad(load) => {
                    let reg = load.registry.clone();
                    let sids = load.slot_ids;
                    let l_indices = load.layer_indices;
                    let blocks = load.shared_blocks;
                    let path = load.path;
                    
                    tokio::spawn(async move {
                        for i in 0..sids.len() {
                            let sid = sids[i];
                            let l_idx = l_indices[i];
                            let block = &blocks[i];
                            
                            let (b_idx_off, b_idx) = { 
                                let inner = block.inner.read().unwrap();
                                (inner.offset, inner.index)
                            };
                            
                            let act_p = if path.is_dir() { 
                                let lp = path.join(format!("l{}.st", l_idx)); 
                                if lp.exists() { lp } else { path.join("l0.st") }
                            } else { path.clone() };

                            if let Ok(st) = candle_core::safetensors::load(&act_p, &Device::Cpu) {
                                let prefix = format!("b{}_l{}_", b_idx_off, if act_p.to_string_lossy().contains("l0.st") { 0 } else { l_idx });
                                if let (Some(kh), Some(ka), Some(kp), Some(ks), Some(va), Some(vp), Some(vs)) = (st.get(&format!("{}k_shape", prefix)), st.get(&format!("{}k_anchors", prefix)), st.get(&format!("{}k_packed", prefix)), st.get(&format!("{}k_scales", prefix)), st.get(&format!("{}v_anchors", prefix)), st.get(&format!("{}v_packed", prefix)), st.get(&format!("{}v_scales", prefix)) ) {
                                    let v_u32 = kh.to_vec1::<u32>().unwrap_or_default();
                                    let os = vec![v_u32[0] as usize, v_u32[1] as usize, v_u32[2] as usize, v_u32[3] as usize];
                                    let m = crate::models::qwen3vl::quantized_model::BitKVMetadata { k_anchors: ka.clone(), k_packed: kp.clone(), k_scales: ks.clone(), v_anchors: va.clone(), v_packed: vp.clone(), v_scales: vs.clone(), original_shape: os };
                                    if let Ok(mut r) = reg.entries.write() { if b_idx < r.len() { let e = &mut r[b_idx]; let mut cache = e.bitkv_cache.write().unwrap(); cache[l_idx] = Some(m); e.location[l_idx] = crate::models::qwen3vl::quantized_model::KVLocation::RAM; } }
                                }
                            }
                            SLOT_MANAGER.release_slot(sid).await;
                        }
                    });
                }
                SlotTask::ChunkedLoad(load) => {
                    let reg = load.registry.clone();
                    let sids = load.slot_ids;
                    let l_indices = load.layer_indices;
                    let blocks = load.shared_blocks;
                    let path = load.path;
                    
                    tokio::spawn(async move {
                        for i in 0..sids.len() {
                            let sid = sids[i];
                            let l_idx = l_indices[i];
                            let block = &blocks[i];
                            
                            let (b_idx_off, b_idx) = { 
                                let inner = block.inner.read().unwrap();
                                (inner.offset, inner.index)
                            };
                            
                            let act_p = if path.is_dir() { 
                                let lp = path.join(format!("l{}.st", l_idx)); 
                                if lp.exists() { lp } else { path.join("l0.st") }
                            } else { path.clone() };

                            if let Ok(st) = candle_core::safetensors::load(&act_p, &Device::Cpu) {
                                let prefix = format!("b{}_l{}_", b_idx_off, if act_p.to_string_lossy().contains("l0.st") { 0 } else { l_idx });
                                if let (Some(kh), Some(ka), Some(kp), Some(ks), Some(va), Some(vp), Some(vs)) = (st.get(&format!("{}k_shape", prefix)), st.get(&format!("{}k_anchors", prefix)), st.get(&format!("{}k_packed", prefix)), st.get(&format!("{}k_scales", prefix)), st.get(&format!("{}v_anchors", prefix)), st.get(&format!("{}v_packed", prefix)), st.get(&format!("{}v_scales", prefix)) ) {
                                    let v_u32 = kh.to_vec1::<u32>().unwrap_or_default();
                                    let os = vec![v_u32[0] as usize, v_u32[1] as usize, v_u32[2] as usize, v_u32[3] as usize];
                                    let m = crate::models::qwen3vl::quantized_model::BitKVMetadata { k_anchors: ka.clone(), k_packed: kp.clone(), k_scales: ks.clone(), v_anchors: va.clone(), v_packed: vp.clone(), v_scales: vs.clone(), original_shape: os };
                                    if let Ok(mut r) = reg.entries.write() { if b_idx < r.len() { let e = &mut r[b_idx]; let mut cache = e.bitkv_cache.write().unwrap(); cache[l_idx] = Some(m); e.location[l_idx] = crate::models::qwen3vl::quantized_model::KVLocation::RAM; } }
                                }
                            }
                            SLOT_MANAGER.release_slot(sid).await;
                        }
                    });
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

    pub fn forward_relay(&mut self, embeds: &Tensor, off: usize, tl: usize, sid: Option<String>, sl: Option<usize>, nl: Option<usize>) -> Result<Tensor> {
        match self {
            Self::QuantizedText(m) => m.language_model.forward(embeds, off, tl, None, None, None),
            Self::QuantizedVL(m) => m.language_model.forward(embeds, off, tl, None, None, None),
            _ => Err(anyhow!("Relay not supported for this model variant")),
        }
    }

    pub fn rebalance_layers(&mut self, device_id: usize, offset: usize, total_len: usize) -> Result<()> {
        match self {
            Self::Standard(_) => Ok(()), 
            Self::QuantizedVL(m) => m.rebalance_layers(device_id, offset, total_len),
            Self::QuantizedText(m) => m.rebalance_layers(device_id, offset, total_len),
        }
    }

    pub fn drop_kv_storage(&mut self) -> Result<()> {
        match self {
            Self::Standard(_) => Ok(()),
            Self::QuantizedVL(m) => m.language_model.drop_kv_storage(),
            Self::QuantizedText(m) => m.language_model.drop_kv_storage(),
        }
    }

    pub fn load_kv_cache(&mut self, path: &Path, device: &Device, expected_len: usize, upscale_refill_len: usize, kv_name: Option<&str>, fragments: &[(usize, std::path::PathBuf)], current_kv_len: usize) -> Result<()> {
        match self {
            Self::QuantizedVL(m) => m.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name, fragments, current_kv_len),
            Self::QuantizedText(m) => m.load_kv_cache(path, device, expected_len, upscale_refill_len, kv_name, fragments, current_kv_len),
            _ => Ok(())
        }
    }

    pub fn save_metadata_to_file(&self, path: &Path) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.registry.save_to_file(path), Self::QuantizedText(m) => m.language_model.registry.save_to_file(path), _ => Ok(()) } }
    pub fn load_metadata_from_file(&self, path: &Path) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.registry.load_from_file(path), Self::QuantizedText(m) => m.language_model.registry.load_from_file(path), _ => Ok(()) } }
    pub fn save_kv_to_disk(&mut self, p: &Path, n: Option<&str>, o: usize) -> Result<()> { match self { Self::QuantizedVL(m) => m.save_kv_cache(p, false, o, n), Self::QuantizedText(m) => m.save_kv_cache(p, false, o, n), _ => Ok(()) } }
    
    pub fn inject_kv_bitkv(&mut self, k_anchors: &[Tensor], k_packed: &[Tensor], k_scales: &[Tensor], v_anchors: &[Tensor], v_packed: &[Tensor], v_scales: &[Tensor], original_shape: &[usize]) -> Result<()> {
        match self {
            Self::QuantizedVL(m) => m.language_model.inject_live_kv_bitkv(k_anchors, k_packed, k_scales, v_anchors, v_packed, v_scales, original_shape),
            Self::QuantizedText(m) => m.language_model.inject_live_kv_bitkv(k_anchors, k_packed, k_scales, v_anchors, v_packed, v_scales, original_shape),
            _ => Ok(()),
        }
    }

    pub fn is_cpu(&self) -> bool {
        match self {
            Self::Standard(m) => m.device().is_cpu(),
            Self::QuantizedVL(m) => m.language_model.is_forced_cpu,
            Self::QuantizedText(m) => m.language_model.is_forced_cpu,
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
    pub fn init(
        path: &str,
        text_device: Option<&Device>,
        text_device_id: usize,
        vision_device: Option<&Device>,
        vision_device_id: usize,
        dtype: Option<DType>,
        hard_token_limit: Option<usize>,
        force_text_only: bool,
        baking_only: bool,
        is_disk_swap: bool,
        kv_root: std::path::PathBuf,
    ) -> Result<Self> {
        let path = if let Some(s) = path.strip_prefix(r"\\?\") { s } else { path };
        Self::init_with_config(path, None, None, text_device, text_device_id, vision_device, vision_device_id, dtype, hard_token_limit, force_text_only, baking_only, is_disk_swap, kv_root)
    }

    pub fn init_with_tokenizer(
        path: &str,
        tokenizer_path: Option<&str>,
        text_device: Option<&Device>,
        text_device_id: usize,
        vision_device: Option<&Device>,
        vision_device_id: usize,
        dtype: Option<DType>,
        hard_token_limit: Option<usize>,
        force_text_only: bool,
        baking_only: bool,
        is_disk_swap: bool,
        kv_root: std::path::PathBuf,
    ) -> Result<Self> {
        let path = if let Some(s) = path.strip_prefix(r"\\?\") { s } else { path };
        Self::init_with_config(path, tokenizer_path, None, text_device, text_device_id, vision_device, vision_device_id, dtype, hard_token_limit, force_text_only, baking_only, is_disk_swap, kv_root) 
    }

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
        let path = if let Some(s) = path.strip_prefix(r"\\?\") { s } else { path };
        let tok_p = if let Some(s) = tokenizer_path.unwrap_or(path).strip_prefix(r"\\?\") { s } else { tokenizer_path.unwrap_or(path) };
        let cfg_p = if let Some(s) = config_path.unwrap_or(path).strip_prefix(r"\\?\") { s } else { config_path.unwrap_or(path) };

        let chat_template = ChatTemplate::init(tok_p)?;
        let tokenizer = TokenizerModel::init(tok_p)?;
        let raw_c: serde_json::Value = serde_json::from_slice(&std::fs::read(std::path::Path::new(cfg_p).join("config.json"))?)?;

        let cfg: Qwen3VLConfig = if raw_c.get("text_config").is_some() {
            serde_json::from_value(raw_c)?
        } else {
            let text_config: crate::models::qwen3vl::config::Qwen3VLTextConfig = serde_json::from_value(raw_c.clone())?;
            crate::models::qwen3vl::config::Qwen3VLConfig {
                architectures: raw_c.get("architectures").and_then(|v| serde_json::from_value(v.clone()).ok()),
                auto_map: raw_c.get("auto_map").and_then(|v| serde_json::from_value(v.clone()).ok()),
                hidden_size: raw_c.get("hidden_size").and_then(|v| v.as_u64()).map(|v| v as usize),
                image_token_id: raw_c.get("image_token_id").and_then(|v| v.as_u64()).map(|v| v as usize),
                model_type: raw_c.get("model_type").and_then(|v| v.as_str()).unwrap_or("qwen2").to_string(),
                text_config: Some(text_config),
                tie_word_embeddings: raw_c.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(true),
                torch_dtype: raw_c.get("torch_dtype").and_then(|v| v.as_str()).map(|s| s.to_string()),
                transformers_version: raw_c.get("transformers_version").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                video_token_id: raw_c.get("video_token_id").and_then(|v| v.as_u64()).map(|v| v as usize),
                vision_config: None,
                vision_start_token_id: None,
                vision_end_token_id: None,
            }
        };

        let t_dev = get_device(text_device);
        let v_dev = get_device(vision_device);
        let dtype = get_dtype(dtype, cfg.text_config.as_ref().and_then(|tc| tc.dtype.as_deref()).unwrap_or("float16"));

        let gguf_f = find_type_files(path, "gguf")?;
        let mmproj_p = gguf_f.iter().find(|f| f.contains("mmproj")).cloned();
        let is_v = mmproj_p.is_some() && !force_text_only;

        let qwen3_vl = if !gguf_f.is_empty() {
            let mut m_p = gguf_f.iter().find(|f| f.contains("Qwen3-0.6B-Q8_0.gguf")).cloned();
            if m_p.is_none() { m_p = gguf_f.iter().find(|f| f.contains("Qwen3-0.6B-Q4_K_M.gguf")).cloned(); }
            if m_p.is_none() { m_p = gguf_f.iter().find(|f| !f.contains("mmproj")).cloned(); }

            let kv_res = hard_token_limit.unwrap_or(4096) as u64 * 40000;

            if is_v {
                let m_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&m_p.unwrap())?)? };
                let mm_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&mmproj_p.unwrap())?)? };
                ModelVariant::QuantizedVL(QuantizedQwen3VLModel::new_with_mmap(&cfg, &gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?, Some(Arc::new(m_mmap)), &gguf_file::Content::read(&mut std::io::Cursor::new(&mm_mmap[..]))?, Some(Arc::new(mm_mmap)), &t_dev, text_device_id, &v_dev, vision_device_id, dtype, kv_res, baking_only)?)
            } else {
                let m_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&m_p.unwrap())?)? };
                ModelVariant::QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel::new_with_mmap(&cfg, &gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?, Some(Arc::new(m_mmap)), &t_dev, text_device_id, dtype, kv_res, baking_only, baking_only)?)
            }
        } else {
            ModelVariant::Standard(Qwen3VLModel::new(cfg, unsafe { VarBuilder::from_mmaped_safetensors(&find_type_files(path, "safetensors")?, dtype, &t_dev)? })?)
        };

        let g_p = std::path::Path::new(cfg_p).join("generation_config.json");
        let g_cfg = if g_p.exists() { serde_json::from_slice(&std::fs::read(g_p)?)? } else { Qwen3VLGenerationConfig::default() };
        let (e1, e2) = match &g_cfg.eos_token_id {
            serde_json::Value::Number(n) => { let id = n.as_u64().unwrap_or(151645) as u32; (id, id) },
            serde_json::Value::Array(arr) => { let id1 = arr.get(0).and_then(|v| v.as_u64()).unwrap_or(151643) as u32; let id2 = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(id1 as u64) as u32; (id1, id2) },
            _ => (151643, 151643),
        };

        Ok(Self { chat_template, tokenizer, pre_processor: Qwen3VLProcessor::new(tok_p, &v_dev, dtype)?, qwen3_vl, text_device: t_dev, vision_device: v_dev, eos_token_id1: e1, eos_token_id2: e2, generation_config: g_cfg, model_name: if baking_only { "Small".into() } else { "2B".into() }, hard_token_limit, kv_root })
    }

    pub fn prefill_text_only(&mut self, text: &str, cancel_token: Option<Arc<AtomicBool>>, mut relay_target: Option<&mut Qwen3VLGenerateModel>, auto_save_path: Option<&std::path::Path>) -> Result<()> {
        let token_ids = self.tokenizer.text_encode_vec(text.to_string(), false)?;
        let total_tokens = token_ids.len();
        let chunk_size = 512;
        let mut current_pos = 0;

        while current_pos < total_tokens {
            if let Some(token) = &cancel_token { if token.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            let end = (current_pos + chunk_size).min(total_tokens);
            let chunk = &token_ids[current_pos..end];
            let chunk_ids = Tensor::from_vec(chunk.to_vec(), (1, end - current_pos), &self.text_device)?;
            let chunk_pos = Tensor::arange(current_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?;
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, total_tokens, None)?;
            if let Some(path) = auto_save_path { let _ = self.save_kv_to_disk(path, None, end); }
            if let Some(ref mut target) = relay_target {
                let (ks, vs) = self.get_current_kv();
                let results: Result<Vec<_>> = ks.par_iter().zip(vs.par_iter()).map(|(k, v): (&Tensor, &Tensor)| {
                    let s_len = k.dim(candle_core::D::Minus2)?;
                    let start = s_len.saturating_sub(end - current_pos);
                    let k_new = k.narrow(candle_core::D::Minus2, start, end - current_pos)?;
                    let v_new = v.narrow(candle_core::D::Minus2, start, end - current_pos)?;
                    if let ModelVariant::QuantizedText(m) = &self.qwen3_vl {
                        let res_k = m.language_model.compress_to_bitkv(&k_new)?;
                        let res_v = m.language_model.compress_to_bitkv(&v_new)?;
                        Ok((res_k, res_v))
                    } else { Err(anyhow!("Unsupported")) }
                }).collect();
                let results = results?;
                let mut k_anchors = vec![]; let mut k_packed = vec![]; let mut k_scales = vec![];
                let mut v_anchors = vec![]; let mut v_packed = vec![]; let mut v_scales = vec![];
                let mut original_shape = vec![];
                for (rk, rv) in results {
                    k_anchors.push(rk.0); k_packed.push(rk.1); k_scales.push(rk.2);
                    v_anchors.push(rv.0); v_packed.push(rv.1); v_scales.push(rv.2);
                    original_shape = rk.3;
                }
                if !k_anchors.is_empty() { target.qwen3_vl.inject_kv_bitkv(&k_anchors, &k_packed, &k_scales, &v_anchors, &v_packed, &v_scales, &original_shape)?; }
            }
            current_pos = end;
        }
        if auto_save_path.is_some() { let _ = self.qwen3_vl.drop_kv_storage(); }
        Ok(())
    }

    pub async fn prefill_only(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, mut _relay_target: Option<&mut Qwen3VLGenerateModel>, kv_name: Option<String>) -> Result<usize> {
        let _start_prep = std::time::Instant::now(); if session_id.is_none() { SLOT_MANAGER.reset_all_slots().await; }
        let m_render = self.chat_template.apply_chat_template(&mes)?; let input = self.pre_processor.process_info(&mes, &m_render)?;
        let f_ids = self.tokenizer.text_encode_vec(input.replace_text, false)?; let t_toks = f_ids.len();
        if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
        println!("[BAKING] Starting Horizontal Pass for {} tokens...", t_toks);
        
        self.qwen3_vl.forward(&Tensor::from_vec(f_ids.clone(), (1, t_toks), &self.text_device)?, None, None, None, None, Some(&Tensor::arange(0u32, t_toks as u32, &self.text_device)?.unsqueeze(0)?), 0, t_toks, session_id.clone())?;
        
        if let Some(s_id) = &session_id {
            let unbaked = self.get_all_unbaked_kv_blocks();
            let mut tasks_sent = 0;
            
            println!("[BAKING] Processing {} unbaked blocks...", unbaked.len());
            
            for (ks, vs, off) in unbaked {
                let slot_id = SLOT_MANAGER.acquire_write_slot(t_toks).await;
                let path = crate::utils::paths::get_kv_dir(None).join(s_id); 
                if !path.exists() { let _ = fs::create_dir_all(&path); }
                
                // [CRITICAL-FIX] 워커에 던지기 전에 '메인 스레드'에서 즉시 CPU로 복사
                let mut dumps = Vec::new(); 
                for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() { 
                    let k_cpu = k.to_device(&Device::Cpu)?;
                    let v_cpu = v.to_device(&Device::Cpu)?;
                    dumps.push(LayerKVDump { layer_idx: idx, k_tensor: k_cpu, v_tensor: v_cpu }); 
                }
                
                if let Ok(tx) = get_bake_worker().await {
                    let rr = match &self.qwen3_vl { 
                        ModelVariant::QuantizedVL(m) => Some(m.language_model.registry.clone()), 
                        ModelVariant::QuantizedText(m) => Some(m.language_model.registry.clone()), 
                        _ => None 
                    };
                    let is_baking_mode = match &self.qwen3_vl { 
                        ModelVariant::QuantizedVL(m) => m.language_model.baking_only, 
                        ModelVariant::QuantizedText(m) => m.language_model.baking_only, 
                        _ => false 
                    };
                    
                    if tx.send(SlotTask::Bake(BakeTask { 
                        slot_id, task_dir: path, kv_name: kv_name.clone(), 
                        offset: off, layers: dumps, is_relay_baking: is_baking_mode, 
                        block_idx: Some(off / 256), registry: rr 
                    })).await.is_ok() {
                        tasks_sent += 1;
                    } else {
                        SLOT_MANAGER.release_slot(slot_id).await;
                    }
                } else {
                    SLOT_MANAGER.release_slot(slot_id).await;
                }
            }
            
            // [WAIT] 모든 블록이 SSD에 써질 때까지 '절대' 리턴하지 않음
            if tasks_sent > 0 {
                println!("[BAKING] Syncing {} blocks to SSD...", tasks_sent);
                wait_for_global_io().await;
            }
            
            let path = crate::utils::paths::get_kv_dir(None).join(s_id);
            self.clear_temporal_kv_caches();
        }
        Ok(t_toks)
    }

    pub async fn prefill_chunk(&mut self, text: String, _cancel_flag: Option<Arc<AtomicBool>>, mut _relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let chunk_ids_vec = self.tokenizer.text_encode_vec(text, false)?;
        let chunk_size = chunk_ids_vec.len();
        let current_pos = self.get_kv_len();
        let chunk_ids = Tensor::from_vec(chunk_ids_vec, (1, chunk_size), &self.text_device)?;
        let chunk_pos = Tensor::arange(current_pos as u32, (current_pos + chunk_size) as u32, &self.text_device)?.unsqueeze(0)?;
        self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, chunk_size, None)?;
        Ok(chunk_size)
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

            // [INFERENCE-STORAGE] 추론 시에는 각 레이어의 지능이 다르므로 28개 전체를 저장합니다.
            // 베이킹(Baking) 단계에서만 0번 레이어 릴레이를 사용합니다.
            for l_idx in 0..28 {
                let mut layer_frags = fragments.iter().filter(|f| f["layer"].as_u64().unwrap_or(999) == l_idx as u64).filter(|f| {
                    let off = f["off"].as_u64().unwrap_or(0) as usize;
                    off >= b_off && off < b_end
                }).collect::<Vec<_>>();
                if layer_frags.is_empty() { continue; }
                layer_frags.sort_by_key(|f| f["off"].as_u64().unwrap_or(0));

                let mut ks = Vec::new(); let mut vs = Vec::new();
                for frag in layer_frags {
                    let path_str = frag["path"].as_str().unwrap_or("");
                    if let Ok(st_map) = candle_core::safetensors::load(Path::new(path_str), &Device::Cpu) {
                        if let (Some(k), Some(v)) = (st_map.get("k"), st_map.get("v")) { ks.push(k.clone()); vs.push(v.clone()); }
                    }
                }
                if !ks.is_empty() {
                    let k_combined = Tensor::cat(&ks, 2)?; let v_combined = Tensor::cat(&vs, 2)?;
                    let mut map = HashMap::new();
                    let prefix = format!("b{}_l{}_", b_off, l_idx);
                    map.insert(format!("{}k_anchors", prefix), k_combined.clone());
                    map.insert(format!("{}v_anchors", prefix), v_combined.clone());
                    map.insert(format!("{}k_shape", prefix), Tensor::from_vec(vec![1u32, 16, k_combined.dim(2)? as u32, 128], (4,), &Device::Cpu)?);
                    let save_path = block_dir.join(format!("l{}.st", l_idx));
                    let _ = candle_core::safetensors::save(&map, &save_path);
                }
            }
        }
        Ok(())
    }

    pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, kv_name: Option<String>) -> Result<String> {
        SLOT_MANAGER.reset_all_slots().await;
        let mut logit_processor = get_logit_processor(Some(mes.temperature.unwrap_or(0.7) as f32), Some(mes.top_p.unwrap_or(0.9) as f32), Some(40), mes.seed.unwrap_or(34562) as u64);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut input = self.pre_processor.process_info(&mes, &mes_render)?;
        let f_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let t_toks = f_ids.len();
        let mut a_ids = f_ids.clone(); let mut curr_s_off = self.get_kv_len();
        let (mut p_vals, i_grid, mut g_text) = (input.pixel_values.take(), input.image_grid_thw.take(), String::new());
        let max_gen = mes.max_tokens.unwrap_or(2048) as usize;

        if t_toks > curr_s_off {
            let prefill_len = t_toks - curr_s_off;
            self.qwen3_vl.forward(&Tensor::from_vec(f_ids[curr_s_off..].to_vec(), (1, prefill_len), &self.text_device)?, None, None, None, None, Some(&Tensor::arange(curr_s_off as u32, t_toks as u32, &self.text_device)?.unsqueeze(0)?), curr_s_off, prefill_len, session_id.clone())?;
            curr_s_off = t_toks;
        }

        // [UNIFIED-HIGH-SPEED-LOOP] 0.6B와 2B 모두 고속 수직 연산 경로를 사용합니다.
        let mut gen_count = 0;
        while g_text.len() < max_gen {
            let last_c = *a_ids.last().unwrap_or(&0);
            // [LOG-FIX] total_len을 현재 누적 길이(curr_s_off + 1)로 전달하여 로그 가독성 개선
            let logits = self.qwen3_vl.forward(&Tensor::from_vec(vec![last_c], (1, 1), &self.text_device)?, p_vals.as_ref(), i_grid.as_ref(), None, None, None, curr_s_off, curr_s_off + 1, session_id.clone())?;
            let next_id = logit_processor.sample(&logits.flatten_all()?)?;
            let chunk_txt = self.tokenizer.token_decode(vec![next_id])?;
            g_text.push_str(&chunk_txt); a_ids.push(next_id); curr_s_off += 1; p_vals = None; gen_count += 1;
            
            // [INCREMENTAL-SAVE] 32단어마다 중간 저장 (충돌 방지 및 진행 상황 확인용)
            if gen_count % 32 == 0 && session_id.is_some() {
                let path = crate::utils::paths::get_kv_dir(None).join(session_id.as_ref().unwrap());
                let _ = self.save_kv_to_disk(&path, kv_name.as_deref(), curr_s_off);
                let _ = self.qwen3_vl.save_metadata_to_file(&path);
                println!("[SSD-CHECKPOINT] Intermediate backup at token {}", curr_s_off);
            }

            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
        }

        // [FINAL-SAVE] 추론 완료 후 최종 백업
        if let Some(s_id) = &session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(s_id);
            if !path.exists() { let _ = fs::create_dir_all(&path); }
            let _ = self.save_kv_to_disk(&path, kv_name.as_deref(), curr_s_off);
            
            // [REGISTRY-SAVE] 메타데이터 목차 파일 저장 (다음 스텝에서 로드할 때 필수)
            let _ = self.qwen3_vl.save_metadata_to_file(&path);
            println!("[SSD-AUTO-SAVE] KV Cache and Metadata backed up to {:?}", path);
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
                    let reg_loc = if inner.index < reg.len() { reg[inner.index].location[l_idx] } else { inner.location };
                    if reg_loc == KVLocation::SSD || (inner.location != KVLocation::VRAM && reg_loc != KVLocation::VRAM) {
                        if inner.k_cache.is_some() || inner.v_cache.is_some() { inner.k_cache = None; inner.v_cache = None; inner.location = reg_loc; }
                    }
                }
            }
        }
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

    pub fn get_current_kv(&self) -> (Vec<Tensor>, Vec<Tensor>) {
        let mut ks = vec![]; let mut vs = vec![];
        let layers = match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => Some(&m.language_model.layers), ModelVariant::QuantizedText(m) => Some(&m.language_model.layers), _ => None };
        if let Some(layers) = layers {
            for l in layers {
                let mut lk = Vec::new(); let mut lv = Vec::new();
                for b in &l.self_attn.kv_blocks {
                    let inner = b.inner.read().unwrap();
                    if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) { lk.push(k.clone()); lv.push(v.clone()); }
                }
                if !lk.is_empty() { ks.push(Tensor::cat(&lk, 2).unwrap()); vs.push(Tensor::cat(&lv, 2).unwrap()); }
            }
        }
        (ks, vs)
    }

    pub fn save_kv_to_disk(&mut self, p: &Path, n: Option<&str>, o: usize) -> Result<()> { self.qwen3_vl.save_kv_to_disk(p, n, o) }
    pub fn load_kv_from_disk(&mut self, p: &Path, n: Option<&str>) -> Result<()> { 
        let _ = self.qwen3_vl.load_metadata_from_file(p);
        let mut restored_len = 0;
        let reg_ref = match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => Some(m.language_model.registry.clone()), ModelVariant::QuantizedText(m) => Some(m.language_model.registry.clone()), _ => None };
        if let Some(reg) = reg_ref { if let Ok(entries) = reg.entries.read() { if let Some(last_valid) = entries.iter().rev().find(|e| e.token_len > 0) { restored_len = last_valid.token_start + last_valid.token_len; } } }
        if restored_len > 0 { println!("[SSD-LOAD] Metadata restored. Exact KV Length: {}", restored_len); }
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.load_kv_cache(p, &self.text_device, restored_len, 128, n, &[], restored_len),
            ModelVariant::QuantizedText(m) => m.load_kv_cache(p, &self.text_device, restored_len, 128, n, &[], restored_len),
            _ => Ok(())
        }
    }
    pub fn truncate_kv_cache(&mut self, l: usize) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.truncate_kv_cache(l), ModelVariant::QuantizedText(m) => m.truncate_kv_cache(l), _ => Ok(()) } }
    pub fn to_device(&mut self, d: &Device) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.to_device(d)?, ModelVariant::QuantizedText(m) => m.to_device(d)?, _ => {} } self.text_device = d.clone(); self.vision_device = d.clone(); Ok(()) }
    pub fn drop_kv_storage(&mut self) -> Result<()> { self.qwen3_vl.drop_kv_storage() }
    pub fn clear_kv_cache(&mut self) { self.clear_temporal_kv_caches(); }
}

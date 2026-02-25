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
use rayon::prelude::*;
use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
use std::fs;
use std::path::Path;

use std::path::PathBuf;
use tokio::sync::mpsc;

// [GLOBAL] 슬롯 관리자
pub struct SlotManager {
    pub slots: Vec<crate::models::qwen3vl::quantized_model::MemorySlot>,
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
    Release { idx: usize, task_id: Option<String>, block_index: Option<usize>, is_bake: bool },
}

impl SlotManager {
    pub fn new(count: usize) -> (Self, mpsc::Receiver<SlotRequest>) {
        let (tx, rx) = mpsc::channel(100);
        let mut slots = Vec::new();
        let num_layers = 28;
        for i in 0..count {
            slots.push(crate::models::qwen3vl::quantized_model::MemorySlot::new(i, num_layers));
        }
        (Self {
            slots,
            handoff_notifier: Arc::new(tokio::sync::Notify::new()),
            active_write_count: Arc::new(AtomicUsize::new(0)),
            count_reads: Arc::new(AtomicUsize::new(0)),
            count_writes: Arc::new(AtomicUsize::new(0)),
            count_cached: Arc::new(AtomicUsize::new(0)),
            count_free: Arc::new(AtomicUsize::new(count)),
            request_tx: tx,
        }, rx)
    }

    fn update_counters(&self, old_state: u8, new_state: u8) {
        if old_state == new_state { return; }
        match old_state {
            0 => { self.count_free.fetch_sub(1, Ordering::SeqCst); },
            1 => { self.count_writes.fetch_sub(1, Ordering::SeqCst); },
            2 => { self.count_cached.fetch_sub(1, Ordering::SeqCst); },
            3 => { self.count_reads.fetch_sub(1, Ordering::SeqCst); },
            _ => {},
        };
        match new_state {
            0 => { self.count_free.fetch_add(1, Ordering::SeqCst); },
            1 => { self.count_writes.fetch_add(1, Ordering::SeqCst); },
            2 => { self.count_cached.fetch_add(1, Ordering::SeqCst); },
            3 => { self.count_reads.fetch_add(1, Ordering::SeqCst); },
            _ => {},
        };
    }

    pub async fn acquire_write_slot(&self, total_tokens: usize) -> usize {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.request_tx.send(SlotRequest::Acquire { total_tokens, tx }).await;
        rx.await.unwrap_or(0)
    }

    pub async fn acquire_read_slot(&self) -> usize {
        loop {
            for (i, slot) in self.slots.iter().enumerate() {
                let current = slot.state.load(Ordering::SeqCst);
                if current == 0 || current == 2 {
                    if slot.state.compare_exchange(current, 3, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                        self.update_counters(current, 3);
                        return i;
                    }
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    pub async fn release_slot(&self, id: usize) {
        let _ = self.request_tx.send(SlotRequest::Release { idx: id, task_id: None, block_index: None, is_bake: true }).await;
    }

    pub async fn mark_ready(&self, id: usize) {
        if id < self.slots.len() {
            let slot = &self.slots[id];
            let current = slot.state.load(Ordering::SeqCst);
            if (current == 1 || current == 3) && slot.state.compare_exchange(current, 2, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                self.update_counters(current, 2);
                if current == 1 { self.active_write_count.fetch_sub(1, Ordering::SeqCst); }
            }
        }
        self.handoff_notifier.notify_waiters();
    }

    pub fn get_counts(&self) -> (usize, usize, usize, usize) {
        (self.count_reads.load(Ordering::Relaxed), self.count_writes.load(Ordering::Relaxed), self.count_cached.load(Ordering::Relaxed), self.count_free.load(Ordering::Relaxed))
    }
}

pub static ACTIVE_BAKE_TASKS: AtomicUsize = AtomicUsize::new(0);
pub static GLOBAL_IO_COUNTER: AtomicUsize = AtomicUsize::new(0);

use once_cell::sync::Lazy;
pub static SLOT_MANAGER_DATA: Lazy<(SlotManager, std::sync::Mutex<Option<mpsc::Receiver<SlotRequest>>>)> = Lazy::new(|| {
    let (sm, rx) = SlotManager::new(128);
    (sm, std::sync::Mutex::new(Some(rx)))
});
pub static SLOT_MANAGER: Lazy<&SlotManager> = Lazy::new(|| &SLOT_MANAGER_DATA.0);

pub struct LayerKVDump {
    pub layer_idx: usize,
    pub k_tensor: Tensor,
    pub v_tensor: Tensor,
}

pub struct BakeTask {
    pub slot_id: usize,
    pub task_dir: PathBuf,
    pub kv_name: Option<String>,
    pub offset: usize,
    pub layers: Vec<LayerKVDump>,
    pub is_relay_baking: bool,
    pub block_idx: Option<usize>,
    pub registry: crate::models::qwen3vl::quantized_model::KVRegistry,
}

pub struct SaveTask {
    pub slot_id: usize,
    pub path: PathBuf,
    pub tensors: std::collections::HashMap<String, Tensor>,
    pub is_last: bool,
    pub block_idx: Option<usize>,
    pub registry: Option<crate::models::qwen3vl::quantized_model::KVRegistry>,
}

pub enum SlotTask { Bake(BakeTask), Load(LoadTask) }

pub struct LoadTask {
    pub slot_id: usize,
    pub path: PathBuf,
    pub layer_idx: usize,
    pub kv_name: Option<String>,
    pub shared_block: crate::models::qwen3vl::quantized_model::KVBlock,
    pub registry: crate::models::qwen3vl::quantized_model::KVRegistry,
}

use tokio::sync::OnceCell;
pub static BAKE_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();
pub static LOAD_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();

pub async fn get_worker_channel() -> Result<mpsc::Sender<SlotTask>> { BAKE_TX.get().cloned().ok_or(anyhow!("Bake worker not initialized")) }
pub async fn get_load_worker() -> Result<mpsc::Sender<SlotTask>> { LOAD_TX.get().cloned().ok_or(anyhow!("Load worker not initialized")) }
pub async fn wait_for_global_io() { while GLOBAL_IO_COUNTER.load(Ordering::SeqCst) > 0 { tokio::time::sleep(std::time::Duration::from_millis(10)).await; } }

pub fn init_bake_worker() {
    let (btx, brx) = mpsc::channel(1000);
    let (ltx, lrx) = mpsc::channel(1000);
    let _ = BAKE_TX.set(btx); let _ = LOAD_TX.set(ltx);
    if let Some(rx) = SLOT_MANAGER_DATA.1.lock().unwrap().take() { tauri::async_runtime::spawn(async move { spawn_slot_dispatcher(rx).await; }); }
    tauri::async_runtime::spawn(async move { spawn_slot_worker(brx); }); 
    tauri::async_runtime::spawn(async move { spawn_slot_worker(lrx); });
}

async fn spawn_slot_dispatcher(mut rx: mpsc::Receiver<SlotRequest>) {
    while let Some(req) = rx.recv().await {
        match req {
            SlotRequest::Acquire { total_tokens, tx } => {
                let max_writes = if total_tokens < 4096 { 12 } else if total_tokens < 8192 { 8 } else { 4 };
                let mut found = None;
                while found.is_none() {
                    if SLOT_MANAGER.active_write_count.load(Ordering::SeqCst) < max_writes {
                        for (i, slot) in SLOT_MANAGER.slots.iter().enumerate() {
                            if slot.state.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() { SLOT_MANAGER.update_counters(0, 1); SLOT_MANAGER.active_write_count.fetch_add(1, Ordering::SeqCst); found = Some(i); break; }
                        }
                        if found.is_none() {
                            for (i, slot) in SLOT_MANAGER.slots.iter().enumerate() {
                                if slot.state.compare_exchange(2, 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() { SLOT_MANAGER.update_counters(2, 1); SLOT_MANAGER.active_write_count.fetch_add(1, Ordering::SeqCst); found = Some(i); break; }
                            }
                        }
                    }
                    if found.is_none() { tokio::time::sleep(std::time::Duration::from_millis(10)).await; }
                }
                let _ = tx.send(found.unwrap());
            },
            SlotRequest::Release { idx, .. } => {
                let slot = &SLOT_MANAGER.slots[idx];
                let prev = slot.state.swap(0, Ordering::SeqCst);
                if prev != 0 {
                    SLOT_MANAGER.update_counters(prev, 0);
                    if prev == 1 { SLOT_MANAGER.active_write_count.fetch_sub(1, Ordering::SeqCst); }
                    for lk in &slot.k_layers { if let Ok(mut g) = lk.try_lock() { *g = None; } }
                    for lv in &slot.v_layers { if let Ok(mut g) = lv.try_lock() { *g = None; } }
                }
                SLOT_MANAGER.handoff_notifier.notify_waiters();
            }
        }
    }
}

fn spawn_slot_worker(mut rx: mpsc::Receiver<SlotTask>) {
    let (io_tx, mut io_rx) = mpsc::channel::<SaveTask>(1000); 
    tokio::spawn(async move {
        while let Some(task) = io_rx.recv().await {
            let (tp, ts, reg, b_idx, sid, is_last) = (task.path.clone(), task.tensors, task.registry.clone(), task.block_idx, task.slot_id, task.is_last);
            if let Some(p) = tp.parent() { if !p.exists() { let _ = fs::create_dir_all(p); } }
            
            // [IO-LOG]
            if let Ok(_) = candle_core::safetensors::save(&ts, &tp) {
                println!("[BAKE-SAVE] Saved KV: {:?}", tp.file_name().unwrap_or_default());
            } else {
                println!("[BAKE-ERR] Failed to save KV: {:?}", tp);
            }

            if let (Some(r), Some(idx)) = (reg, b_idx) {
                if let Ok(mut entries) = r.entries.write() {
                    if idx < entries.len() {
                        let e = &mut entries[idx]; e.ssd_path = Some(tp.parent().unwrap().to_path_buf());
                        if let Some(l_str) = tp.file_name().and_then(|n| n.to_str()).and_then(|s| s.strip_prefix('l')).and_then(|s| s.strip_suffix(".st")) {
                            if let Ok(l_idx) = l_str.parse::<usize>() { if l_idx < 28 { e.location[l_idx] = crate::models::qwen3vl::quantized_model::KVLocation::SSD; } }
                        }
                    }
                }
            }
            GLOBAL_IO_COUNTER.fetch_sub(1, Ordering::SeqCst);
            let rem = SLOT_MANAGER.slots[sid].remaining_layers.fetch_sub(1, Ordering::SeqCst);
            if rem <= 1 || is_last { SLOT_MANAGER.mark_ready(sid).await; }
        }
    });

    tokio::spawn(async move {
        while let Some(task) = rx.recv().await {
            match task {
                SlotTask::Bake(bake) => {
                    let io_tx_inner = io_tx.clone();
                    let (sid, off, is_relay) = (bake.slot_id, bake.offset, bake.is_relay_baking);
                    let loop_count = bake.layers.len();
                    SLOT_MANAGER.slots[sid].remaining_layers.store(loop_count, Ordering::SeqCst);
                    for l_idx in 0..loop_count {
                        let src = &bake.layers[l_idx];
                        let act_l = if is_relay && loop_count == 1 { 0 } else { src.layer_idx };
                        let mut map = std::collections::HashMap::new();
                        let prefix = if is_relay { format!("b{}_l0_", off) } else { format!("b{}_l{}_", off, act_l) };
                        if let (Ok(ks_dims), Ok(vs_dims)) = (src.k_tensor.dims4(), src.v_tensor.dims4()) {
                            let (b, h, s, d) = (ks_dims.0, ks_dims.1, ks_dims.2, ks_dims.3);
                            let total_elements = b * h * s * d;
                            let ac = (0..s).filter(|&i| i < 4 || i % 8 == 0).count();
                            
                            // 1. Process K
                            if let Ok(t_f32) = src.k_tensor.to_device(&Device::Cpu).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.flatten_all()) {
                                if let Ok(t_data) = t_f32.to_vec1::<f32>() {
                                    let mut ka = vec![0.0f32; b * h * ac * d];
                                    let mut kp = vec![0u8; (total_elements * 4 + 7) / 8];
                                    let mut ksc = vec![0.0f32; b * h * s];
                                    for bh in 0..(b*h) {
                                        let bho = bh * (s * d);
                                        for ti in 0..s {
                                            let td = &t_data[bho + ti * d .. bho + (ti + 1) * d];
                                            if ti < 4 || ti % 8 == 0 {
                                                let anchor_pos = if ti < 4 { ti } else { 4 + (ti - 4) / 8 };
                                                ka[(bh * ac + anchor_pos) * d .. (bh * ac + anchor_pos + 1) * d].copy_from_slice(td);
                                            }
                                            
                                            // [STABLE-REVERT] Max-Absolute Scaling
                                            let mut max_abs = 0.0f32;
                                            for &v in td { let a = v.abs(); if a > max_abs { max_abs = a; } }
                                            let base_s = max_abs;
                                            ksc[bh * s + ti] = base_s;

                                            let mut res = td.to_vec();
                                            for bit_l in 0..4 {
                                                let cur_s = base_s / (2.0f32.powi(bit_l as i32));
                                                for i in 0..d {
                                                    let bit_idx = (bho + ti * d + i) + bit_l * total_elements;
                                                    if res[i] >= 0.0 { kp[bit_idx / 8] |= 1 << (bit_idx % 8); res[i] -= cur_s; }
                                                    else { res[i] += cur_s; }
                                                }
                                            }
                                        }
                                    }
                                    let kpl = kp.len();
                                    if let Ok(t) = Tensor::from_vec(ka, vec![b, h, ac, d], &Device::Cpu) { map.insert(format!("{}k_anchors", prefix), t); }
                                    if let Ok(t) = Tensor::from_vec(kp, vec![kpl], &Device::Cpu) { map.insert(format!("{}k_packed", prefix), t); }
                                    if let Ok(t) = Tensor::from_vec(ksc, vec![b, h, s, 1], &Device::Cpu) { map.insert(format!("{}k_scales", prefix), t); }
                                }
                            }
                            
                            // 2. Process V
                            if let Ok(t_f32) = src.v_tensor.to_device(&Device::Cpu).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.flatten_all()) {
                                if let Ok(t_data) = t_f32.to_vec1::<f32>() {
                                    let mut va = vec![0.0f32; b * h * ac * d];
                                    let mut vp = vec![0u8; (total_elements * 4 + 7) / 8];
                                    let mut vsc = vec![0.0f32; b * h * s];
                                    for bh in 0..(b*h) {
                                        let bho = bh * (s * d);
                                        for ti in 0..s {
                                            let td = &t_data[bho + ti * d .. bho + (ti + 1) * d];
                                            if ti < 4 || ti % 8 == 0 {
                                                let anchor_pos = if ti < 4 { ti } else { 4 + (ti - 4) / 8 };
                                                va[(bh * ac + anchor_pos) * d .. (bh * ac + anchor_pos + 1) * d].copy_from_slice(td);
                                            }
                                            
                                            // [STABLE-REVERT] Max-Absolute Scaling
                                            let mut max_abs = 0.0f32;
                                            for &v in td { let a = v.abs(); if a > max_abs { max_abs = a; } }
                                            let base_s = max_abs;
                                            vsc[bh * s + ti] = base_s;

                                            let mut res = td.to_vec();
                                            for bit_l in 0..4 {
                                                let cur_s = base_s / (2.0f32.powi(bit_l as i32));
                                                for i in 0..d {
                                                    let bit_idx = (bho + ti * d + i) + bit_l * total_elements;
                                                    if res[i] >= 0.0 { vp[bit_idx / 8] |= 1 << (bit_idx % 8); res[i] -= cur_s; }
                                                    else { res[i] += cur_s; }
                                                }
                                            }
                                        }
                                    }
                                    let vpl = vp.len();
                                    if let Ok(t) = Tensor::from_vec(va, vec![b, h, ac, d], &Device::Cpu) { map.insert(format!("{}v_anchors", prefix), t); }
                                    if let Ok(t) = Tensor::from_vec(vp, vec![vpl], &Device::Cpu) { map.insert(format!("{}v_packed", prefix), t); }
                                    if let Ok(t) = Tensor::from_vec(vsc, vec![b, h, s, 1], &Device::Cpu) { map.insert(format!("{}v_scales", prefix), t); }
                                }
                            }
                            
                            if let Ok(t) = Tensor::from_vec(vec![b as u32, h as u32, s as u32, d as u32], (4,), &Device::Cpu) { map.insert(format!("{}k_shape", prefix), t); }
                            if let Ok(t) = Tensor::from_vec(vec![3u32], (1,), &Device::Cpu) { map.insert(format!("{}mode", prefix), t); }
                        }
                        GLOBAL_IO_COUNTER.fetch_add(1, Ordering::SeqCst);
                        let _ = io_tx_inner.send(SaveTask { slot_id: sid, path: bake.task_dir.join(format!("l{}.st", act_l)), tensors: map, is_last: l_idx == loop_count - 1, block_idx: bake.block_idx, registry: Some(bake.registry.clone()) }).await;
                    }
                },
                SlotTask::Load(load) => {
                    let sid = load.slot_id; let reg = load.registry.clone(); let l_idx = load.layer_idx; let shared_block = load.shared_block.clone();
                    let provided_path = load.path.clone(); 
                    tokio::spawn(async move {
                        let _guard = ReadSlotGuard { sid, active: true };
                        let (b_idx_off, b_idx, recorded_path) = { match shared_block.inner.read() { Ok(inner) => (inner.offset, inner.index, inner.ssd_path.clone()), _ => (0, 999, None) } };
                        let mut root = provided_path.clone();
                        while root.to_string_lossy().contains("inference") || root.to_string_lossy().contains("reference") || root.to_string_lossy().contains("b") {
                            if let Some(parent) = root.parent() { root = parent.to_path_buf(); if root.to_string_lossy().ends_with("kv") { break; } } else { break; }
                        }
                        let filename = format!("l{}.st", l_idx);
                        let b_str = format!("b{}", b_idx_off);
                        let act_p = if let Some(path) = recorded_path { if path.is_file() { path } else { path.join(&filename) } } else { root.join("inference").join(&b_str).join(&filename) };
                        let ref_p = root.join("reference").join(&b_str).join("l0.st");
                        let final_path = if act_p.is_file() { Some(act_p) } else if ref_p.is_file() { Some(ref_p) } else { None };
                        if let Some(p) = final_path {
                            if let Ok(st) = candle_core::safetensors::load(&p, &Device::Cpu) {
                                let is_relay = p.to_string_lossy().contains("l0.st");
                                let prefix = if is_relay { format!("b{}_l0_", b_idx_off) } else { format!("b{}_l{}_", b_idx_off, l_idx) };
                                if let (Some(kh), Some(ka), Some(kp), Some(ks), Some(va), Some(vp), Some(vs)) = (st.get(&format!("{}k_shape", prefix)), st.get(&format!("{}k_anchors", prefix)), st.get(&format!("{}k_packed", prefix)), st.get(&format!("{}k_scales", prefix)), st.get(&format!("{}v_anchors", prefix)), st.get(&format!("{}v_packed", prefix)), st.get(&format!("{}v_scales", prefix)) ) {
                                    let v_u32 = kh.to_vec1::<u32>().unwrap_or_default();
                                    let os = vec![v_u32[0] as usize, v_u32[1] as usize, v_u32[2] as usize, v_u32[3] as usize];
                                    let m = crate::models::qwen3vl::quantized_model::BitKVMetadata { k_anchors: ka.clone(), k_packed: kp.clone(), k_scales: ks.clone(), v_anchors: va.clone(), v_packed: vp.clone(), v_scales: vs.clone(), original_shape: os };
                                    if let Ok(mut r) = reg.entries.write() { if b_idx < r.len() { let e = &mut r[b_idx]; let mut cache = e.bitkv_cache.write().unwrap(); cache[l_idx] = Some(m); e.location[l_idx] = crate::models::qwen3vl::quantized_model::KVLocation::RAM; } }
                                }
                            }
                        }
                    });
                }
            }
        }
    });
}

struct ReadSlotGuard { sid: usize, active: bool }
impl Drop for ReadSlotGuard { fn drop(&mut self) { if self.active { let sid = self.sid; tauri::async_runtime::spawn(async move { SLOT_MANAGER.release_slot(sid).await; }); } } }

#[derive(Clone)]
pub enum ModelVariant { Standard(crate::models::qwen3vl::model::Qwen3VLModel), QuantizedVL(QuantizedQwen3VLModel), QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel) }

impl ModelVariant {
    pub async fn forward(&mut self, input_ids: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, video_pixel_values: Option<&Tensor>, video_grid_thw: Option<&Tensor>, cache_position: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>) -> Result<Tensor> {
        match self {
            Self::Standard(m) => m.forward(input_ids, pixel_values, image_grid_thw, video_pixel_values, video_grid_thw, cache_position, seqlen_offset),
            Self::QuantizedVL(m) => m.forward(input_ids, pixel_values, image_grid_thw, video_pixel_values, video_grid_thw, cache_position, seqlen_offset, total_len, session_id).await,
            Self::QuantizedText(m) => m.forward(input_ids, cache_position, seqlen_offset, total_len, session_id).await,
        }
    }
    pub fn rebalance_layers(&mut self, device_id: usize, offset: usize, total_len: usize) -> Result<()> { match self { Self::Standard(_) => Ok(()), Self::QuantizedVL(m) => m.rebalance_layers(device_id, offset, total_len), Self::QuantizedText(m) => m.rebalance_layers(device_id, offset, total_len) } }
    pub fn get_current_kv(&self) -> (Vec<Tensor>, Vec<Tensor>) { match self { Self::QuantizedVL(m) => m.language_model.get_current_kv(), Self::QuantizedText(m) => m.language_model.get_current_kv(), _ => (vec![], vec![]) } }
    pub fn inject_kv_bitkv(&mut self, ka: &[Tensor], kp: &[Tensor], ks: &[Tensor], va: &[Tensor], vp: &[Tensor], vs: &[Tensor], os: &[usize]) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.inject_live_kv_bitkv(ka, kp, ks, va, vp, vs, os), Self::QuantizedText(m) => m.language_model.inject_live_kv_bitkv(ka, kp, ks, va, vp, vs, os), _ => Ok(()) } }
    pub async fn drop_kv_storage(&mut self) -> Result<()> { match self { Self::QuantizedVL(m) => m.language_model.drop_kv_storage(), Self::QuantizedText(m) => m.language_model.drop_kv_storage(), _ => Ok(()) } }
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
            crate::models::qwen3vl::config::Qwen3VLConfig { architectures: raw_config.get("architectures").and_then(|v| serde_json::from_value(v.clone()).ok()), auto_map: raw_config.get("auto_map").and_then(|v| serde_json::from_value(v.clone()).ok()), hidden_size: raw_config.get("hidden_size").and_then(|v| v.as_u64()).map(|v| v as usize), image_token_id: raw_config.get("image_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), model_type: raw_config.get("model_type").and_then(|v| v.as_str()).unwrap_or("qwen2").to_string(), text_config: Some(text_config), tie_word_embeddings: raw_config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(true), torch_dtype: raw_config.get("torch_dtype").and_then(|v| v.as_str()).map(|s| s.to_string()), transformers_version: raw_config.get("transformers_version").and_then(|v| v.as_str()).unwrap_or("").to_string(), video_token_id: raw_config.get("video_token_id").and_then(|v| v.as_u64()).map(|v| v as usize), vision_config: None, vision_start_token_id: None, vision_end_token_id: None }
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
                ModelVariant::QuantizedVL(QuantizedQwen3VLModel::new_with_mmap(&cfg, &gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?, Some(Arc::new(m_mmap)), &gguf_file::Content::read(&mut std::io::Cursor::new(&mm_mmap[..]))?, Some(Arc::new(mm_mmap)), &t_dev, text_device_id, &v_dev, vision_device_id, dtype, kv_res, baking_only)?)
            } else {
                let m_mmap = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(m_p.as_ref().unwrap())?)? };
                // [FIX] Single-layer mode ONLY for Baking. Inference (Step A) needs ALL layers.
                let use_sl = baking_only;
                ModelVariant::QuantizedText(crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel::new_with_mmap(&cfg, &gguf_file::Content::read(&mut std::io::Cursor::new(&m_mmap[..]))?, Some(Arc::new(m_mmap)), &t_dev, text_device_id, dtype, kv_res, baking_only, use_sl)?)
            }
        } else { ModelVariant::Standard(Qwen3VLModel::new(cfg, unsafe { VarBuilder::from_mmaped_safetensors(&find_type_files(path, "safetensors")?, dtype, &t_dev)? })?) };
        let g_p = std::path::Path::new(cfg_path).join("generation_config.json"); let g_cfg = if g_p.exists() { serde_json::from_slice(&std::fs::read(g_p)?)? } else { Qwen3VLGenerationConfig::default() };
        let (e1, e2) = match &g_cfg.eos_token_id { serde_json::Value::Number(n) => { let id = n.as_u64().unwrap_or(151645) as u32; (id, id) }, serde_json::Value::Array(arr) => { (arr.get(0).and_then(|v| v.as_u64()).unwrap_or(151643) as u32, arr.get(1).and_then(|v| v.as_u64()).unwrap_or(151643) as u32) }, _ => (151643, 151643) };
        let loaded_model_name = if m_p.as_ref().map(|p| p.contains("0.6B")).unwrap_or(false) { "0.6B".to_string() } else { "2B".to_string() };
        Ok(Self { chat_template, tokenizer, pre_processor: Qwen3VLProcessor::new(tok_path, &v_dev, dtype)?, qwen3_vl, text_device: t_dev, vision_device: v_dev, eos_token_id1: e1, eos_token_id2: e2, generation_config: g_cfg, model_name: loaded_model_name, hard_token_limit, kv_root })
    }

    pub async fn prefill_only(&mut self, mes: ChatCompletionParameters, _cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _relay_target: Option<&mut Qwen3VLGenerateModel>, _kv_name: Option<String>) -> Result<usize> {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let full_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_toks = full_ids.len();
        self.qwen3_vl.forward(&Tensor::from_vec(full_ids.clone(), (1, total_toks), &self.text_device)?, None, None, None, None, Some(&Tensor::arange(0u32, total_toks as u32, &self.text_device)?.unsqueeze(0)?), 0, total_toks, session_id.clone()).await?;
        if let Some(s_id) = &session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(s_id);
            if !path.exists() { fs::create_dir_all(&path)?; }
            fs::write(path.join("tokens.json"), serde_json::to_string(&full_ids)?)?;
        }
        Ok(total_toks)
    }

    pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _kv_name: Option<String>) -> Result<String> {
        let temperature = mes.temperature.unwrap_or(0.7) as f32;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut lp = get_logit_processor(Some(temperature), Some(mes.top_p.unwrap_or(0.9) as f32), Some(40), seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let f_ids = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_toks = f_ids.len();
        let mut gen_text = String::new();
        let mut logits = self.qwen3_vl.forward(&Tensor::from_vec(f_ids.clone(), (1, total_toks), &self.text_device)?, None, None, None, None, Some(&Tensor::arange(0u32, total_toks as u32, &self.text_device)?.unsqueeze(0)?), 0, total_toks, session_id.clone()).await?;
        let mut gen_ids = vec![];
        for i in 0..mes.max_tokens.unwrap_or(2048) {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { break; } }
            let next_id = lp.sample(&logits.flatten_all()?.to_dtype(DType::F32)?)?;
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            gen_ids.push(next_id);
            gen_text.push_str(&self.tokenizer.token_decode(vec![next_id])?);
            logits = self.qwen3_vl.forward(&Tensor::from_vec(vec![next_id], (1, 1), &self.text_device)?, None, None, None, None, None, (total_toks as usize + i as usize), (total_toks as usize + i as usize + 1), session_id.clone()).await?;
        }
        Ok(gen_text)
    }

    pub fn get_kv_len(&self) -> usize { match &self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.get_kv_len(), ModelVariant::QuantizedText(m) => m.language_model.get_kv_len(), _ => 0 } }
    pub async fn drop_kv_storage(&mut self) -> Result<()> { self.qwen3_vl.drop_kv_storage().await }
    pub fn clear_kv_cache(&mut self) { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.clear_kv_cache(), ModelVariant::QuantizedText(m) => m.language_model.clear_kv_cache(), _ => {} } }
    pub fn save_kv_to_disk(&mut self, path: &Path, kv_name: Option<&str>, offset: usize) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.save_kv_cache(path, false, offset, kv_name), ModelVariant::QuantizedText(m) => m.language_model.save_kv_cache(path, false, offset, kv_name), _ => Ok(()) } }
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.truncate_kv_cache(len), ModelVariant::QuantizedText(m) => m.language_model.truncate_kv_cache(len), _ => Ok(()) } }
    pub fn load_kv_from_disk(&mut self, path: &Path, kv_name: Option<&str>) -> Result<()> { match &mut self.qwen3_vl { ModelVariant::QuantizedVL(m) => m.language_model.load_kv_cache(path, &self.text_device, 0, 128, kv_name), ModelVariant::QuantizedText(m) => m.language_model.load_kv_cache(path, &self.text_device, 0, 128, kv_name), _ => Ok(()) } }
    pub async fn prefill_chunk(&mut self, text: String, cancel_flag: Option<Arc<AtomicBool>>, _relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let chunk_ids_vec = self.tokenizer.text_encode_vec(text, false)?;
        let chunk_size = chunk_ids_vec.len();
        let current_pos = self.get_kv_len();
        let chunk_ids = Tensor::from_vec(chunk_ids_vec, (1, chunk_size), &self.text_device)?;
        let chunk_pos = Tensor::arange(current_pos as u32, (current_pos + chunk_size) as u32, &self.text_device)?.unsqueeze(0)?;
        self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, current_pos + chunk_size, None).await?;
        Ok(chunk_size)
    }
}

use crate::models::qwen3vl::quantized_model::{KVLocation, RegistryEntry};
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

// [GLOBAL] 슬롯 관리자: Base(28) + Sub(14) 하이브리드 관리
pub struct SlotManager {
    pub base_slots: Vec<crate::models::qwen3vl::quantized_model::MemorySlot>, // Layer 0-27 Dedicated
    pub sub_slots: Vec<crate::models::qwen3vl::quantized_model::MemorySlot>,  // I/O Shuttle Pool
    pub handoff_notifier: Arc<tokio::sync::Notify>,
    pub purge_signal: Arc<tokio::sync::Notify>, // [NEW] Added for memory cleanup sync
    pub sub_slot_idx: Arc<AtomicUsize>, // Round-robin index for sub slots
    
    // Counters for Sub Slots (since Base Slots are static)
    pub sub_active_count: Arc<AtomicUsize>,
}

impl SlotManager {
    pub fn new() -> Self {
        // ... (existing initialization)
        let mut base_slots = Vec::new();
        for i in 0..28 {
            base_slots.push(crate::models::qwen3vl::quantized_model::MemorySlot::new(i, 1));
        }

        let mut sub_slots = Vec::new();
        for i in 0..14 {
            sub_slots.push(crate::models::qwen3vl::quantized_model::MemorySlot::new(28 + i, 28));
        }

        Self {
            base_slots,
            sub_slots,
            handoff_notifier: Arc::new(tokio::sync::Notify::new()),
            purge_signal: Arc::new(tokio::sync::Notify::new()),
            sub_slot_idx: Arc::new(AtomicUsize::new(0)),
            sub_active_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub async fn reset_all_slots(&self) {
        println!("[SLOT-MANAGER] Resetting all slots...");
        for slot in &self.base_slots {
            slot.state.store(0, Ordering::SeqCst);
            for k in &slot.k_layers { *k.lock().await = None; }
            for v in &slot.v_layers { *v.lock().await = None; }
        }
        for slot in &self.sub_slots {
            self.release_sub_slot(slot.id).await;
        }
    }

    // Base Slot Access (Direct)
    pub fn get_base_slot(&self, layer_idx: usize) -> &crate::models::qwen3vl::quantized_model::MemorySlot {
        &self.base_slots[layer_idx]
    }

    // Sub Slot Acquisition (Round-Robin)
    pub async fn acquire_sub_slot(&self) -> usize {
        let count = self.sub_slots.len();
        loop {
            let start_idx = self.sub_slot_idx.load(Ordering::Relaxed);
            for i in 0..count {
                let idx = (start_idx + i) % count;
                let slot = &self.sub_slots[idx];
                
                // Try to grab a free slot (State 0)
                if slot.state.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    self.sub_active_count.fetch_add(1, Ordering::SeqCst);
                    self.sub_slot_idx.store((idx + 1) % count, Ordering::Relaxed);
                    return slot.id; // Returns global ID (28+)
                }
            }
            // Wait if all full
            tokio::time::timeout(std::time::Duration::from_millis(10), self.handoff_notifier.notified()).await.ok();
        }
    }

    pub async fn release_sub_slot(&self, global_id: usize) {
        if global_id >= 28 {
            let idx = global_id - 28;
            if idx < self.sub_slots.len() {
                let slot = &self.sub_slots[idx];
                let prev = slot.state.swap(0, Ordering::SeqCst);
                if prev != 0 {
                    self.sub_active_count.fetch_sub(1, Ordering::SeqCst);
                    // Clear data
                    for k in &slot.k_layers { *k.lock().await = None; }
                    for v in &slot.v_layers { *v.lock().await = None; }
                    slot.ready_signal.notify_waiters();
                    
                    // Signal for sync blocking wait
                    let _guard = slot.lock.lock().unwrap();
                    slot.condvar.notify_all();
                }
            }
            self.handoff_notifier.notify_waiters();
        }
    }
    
    pub async fn release_slot(&self, id: usize) {
        if id >= 28 {
            self.release_sub_slot(id).await;
        } else {
            // Base Slot: Just reset state to 0 (Free)
            // Note: We generally don't "release" base slots in the same way, but for compatibility:
            if id < self.base_slots.len() {
                let slot = &self.base_slots[id];
                slot.state.store(0, Ordering::SeqCst);
                slot.ready_signal.notify_waiters();
                
                let _guard = slot.lock.lock().unwrap();
                slot.condvar.notify_all();
            }
        }
    }

    pub async fn mark_ready(&self, id: usize) {
        if id >= 28 {
            // Sub Slot
            let idx = id - 28;
            if idx < self.sub_slots.len() {
                let slot = &self.sub_slots[idx];
                let current = slot.state.load(Ordering::SeqCst);
                // Transition 1 (Writing) -> 2 (Ready)
                if current == 1 && slot.state.compare_exchange(1, 2, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    slot.ready_signal.notify_waiters();
                    
                    let _guard = slot.lock.lock().unwrap();
                    slot.condvar.notify_all();
                }
            }
        } else {
            // Base Slot
            if id < self.base_slots.len() {
                let slot = &self.base_slots[id];
                slot.state.store(2, Ordering::SeqCst); // 2 = Ready
                slot.ready_signal.notify_waiters();
                
                let _guard = slot.lock.lock().unwrap();
                slot.condvar.notify_all();
            }
        }
        self.handoff_notifier.notify_waiters();
    }

    // Helper to get any slot by global ID
    pub fn get_slot(&self, global_id: usize) -> Option<&crate::models::qwen3vl::quantized_model::MemorySlot> {
        if global_id < 28 {
            self.base_slots.get(global_id)
        } else {
            self.sub_slots.get(global_id - 28)
        }
    }

    pub fn debug_stats(&self) {
        let active_sub = self.sub_active_count.load(Ordering::Relaxed);
        let mut active_base_count = 0;
        let mut base_info = Vec::new();
        for (i, s) in self.base_slots.iter().enumerate() {
            let state = s.state.load(Ordering::Relaxed);
            if state != 0 { 
                active_base_count += 1; 
                // 상세 분석을 위해 처음 몇 개 레이어의 정보만 수집
                if i < 3 { base_info.push(format!("L{}:S{}", i, state)); }
            }
        }
        println!("[SLOT-STATS] Base: {}/28 ({:?}) | Sub: {}/14 Active", active_base_count, base_info, active_sub);
    }

    pub async fn wait_all_sub_slots(&self) {
        let mut last_print = std::time::Instant::now();
        loop {
            let active = self.sub_active_count.load(Ordering::Relaxed);
            if active == 0 { break; }
            
            if last_print.elapsed().as_secs() >= 1 {
                println!("[SLOT-MANAGER] Waiting for {} sub-slots to finish IO...", active);
                last_print = std::time::Instant::now();
            }
            
            tokio::time::timeout(std::time::Duration::from_millis(100), self.handoff_notifier.notified()).await.ok();
        }
        println!("[SLOT-MANAGER] All sub-slots cleared.");
    }

    pub fn get_counts(&self) -> (usize, usize) {
        let active = self.sub_active_count.load(Ordering::Relaxed);
        (active, 14 - active)
    }
}

pub static ACTIVE_BAKE_TASKS: AtomicUsize = AtomicUsize::new(0);
pub static SLOT_MANAGER: once_cell::sync::Lazy<SlotManager> = once_cell::sync::Lazy::new(|| SlotManager::new()); // 1 Central + 13 Workers

// [MEMORY] 강제 메모리 해제 및 초기화 함수 (Async)
async fn purge_vram_position(device: &Device) {
    if device.is_cuda() {
        let dev_clone = device.clone();
        // 비동기 방식으로 GPU 동기화를 호출하여 현재 포지션의 연산 잔재를 완전히 소거
        let _ = tokio::task::spawn_blocking(move || {
            let _ = dev_clone.synchronize();
        }).await;
    }
}

pub struct LayerKVDump {
    pub layer_idx: usize,
    pub k: Tensor,
    pub v: Tensor,
}

pub struct BakeTask {
    pub slot_id: usize,
    pub task_dir: PathBuf,
    pub kv_name: Option<String>,
    pub offset: usize,
    pub layers: Vec<LayerKVDump>,
    pub registry: Option<crate::models::qwen3vl::quantized_model::KVRegistry>,
}

pub struct SaveTask {
    pub slot_id: usize,
    pub layer_idx: usize,
    pub path: PathBuf,
    pub tensors: std::collections::HashMap<String, Tensor>,
    pub registry: Option<crate::models::qwen3vl::quantized_model::KVRegistry>,
    pub offset: usize,
}

// [GLOBAL] 통합 슬롯 작업 채널
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
    pub registry: crate::models::qwen3vl::quantized_model::KVRegistry, // [NEW]
}

use tokio::sync::OnceCell;

pub static SLOT_TX: OnceCell<mpsc::Sender<SlotTask>> = OnceCell::const_new();

/// 워커 채널을 안전하게 가져오며, 준비될 때까지 비동기적으로 대기합니다.
pub async fn get_worker_channel() -> Result<mpsc::Sender<SlotTask>> {
    // 준비될 때까지 최대 5초간 대기 (폴링 대신 yield 활용)
    for _ in 0..50 {
        if let Some(tx) = SLOT_TX.get() {
            return Ok(tx.clone());
        }
        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    Err(anyhow!("Slot worker channel initialization timed out"))
}

pub fn init_bake_worker() {
    let (tx, rx) = mpsc::channel(100);
    tauri::async_runtime::spawn(async move {
        spawn_slot_worker(rx);
    });
    let _ = SLOT_TX.set(tx);
    println!("[INIT] Slot worker channel initialized and registered.");
}

fn spawn_slot_worker(mut rx: mpsc::Receiver<SlotTask>) {
    let (io_tx, mut io_rx) = mpsc::channel::<SaveTask>(500); // Increased capacity for multi-layer tasks
    
    // Phase B: 디스크 I/O 전담 워커 (IO 채널)
    tokio::spawn(async move {
        println!("[IO-WORKER] Disk writer started.");
        while let Some(task) = io_rx.recv().await {
            let _start = std::time::Instant::now();
            
            if let Some(parent) = task.path.parent() {
                if !parent.exists() { let _ = std::fs::create_dir_all(parent); }
            }

            let save_result = candle_core::safetensors::save(&task.tensors, &task.path);
            if let Err(e) = save_result {
                println!("[IO-ERROR] Failed to save SSD chunk to {:?}: {}", task.path, e);
            } else {
                // [REGISTRY-UPDATE] 명시적인 layer_idx를 사용하여 정확하게 업데이트
                if let Some(reg_obj) = &task.registry {
                    let mut reg = reg_obj.entries.write().unwrap();
                    let block_index = task.offset / 1024;
                    if block_index < reg.len() && task.layer_idx < 28 {
                        reg[block_index].location[task.layer_idx] = KVLocation::SSD;
                        reg[block_index].slot_ids[task.layer_idx] = None;
                    }
                }
            }
            
            let slot = SLOT_MANAGER.get_slot(task.slot_id).unwrap();
            let remaining = slot.remaining_layers.fetch_sub(1, Ordering::SeqCst);
            
            if remaining == 1 {
                SLOT_MANAGER.release_slot(task.slot_id).await;
            }
        }
    });

    // Phase A: CPU 작업 전담 워커 (압축 및 로드)
    tokio::spawn(async move {
        println!("[SLOT-WORKER] Started.");
        while let Some(task) = rx.recv().await {
            match task {
                SlotTask::Bake(bake) => {
                    let slot_id = bake.slot_id;
                    let task_dir = bake.task_dir;
                    let kv_name = bake.kv_name;
                    let offset = bake.offset;
                    let registry = bake.registry;
                    
                    if bake.layers.is_empty() {
                        SLOT_MANAGER.release_slot(slot_id).await;
                        continue;
                    }

                    let slot = SLOT_MANAGER.get_slot(slot_id).unwrap();
                    let total_layers = bake.layers.len();
                    slot.remaining_layers.store(total_layers, Ordering::SeqCst);
                    
                    for layer in bake.layers {
                        // ... existing layer processing ...
                        {
                            if let Ok(mut k_guard) = slot.k_layers[layer.layer_idx].try_lock() { *k_guard = Some(layer.k.clone()); }
                            if let Ok(mut v_guard) = slot.v_layers[layer.layer_idx].try_lock() { *v_guard = Some(layer.v.clone()); }
                        }

                        let filename = match (&kv_name, offset) {
                            (Some(name), 0) => format!("layer_{}_{}_kv.safetensors", name, layer.layer_idx),
                            (Some(name), off) => format!("layer_{}_{}_kv_{}.safetensors", name, layer.layer_idx, off),
                            (None, 0) => format!("layer_{}_kv.safetensors", layer.layer_idx),
                            (None, off) => format!("layer_{}_kv_{}.safetensors", layer.layer_idx, off),
                        };
                        let path = task_dir.join(filename);
                        println!("[IO-WRITE] Slot {} saving Layer {} to {:?}", slot_id, layer.layer_idx, path);
                        
                        let mut map = std::collections::HashMap::new();
                        // ... tensor processing ...
                        if let Ok(dims) = layer.k.dims4() {
                            let (b, h, s, d) = dims;
                            if let Ok(t_f32) = layer.k.to_device(&Device::Cpu).and_then(|t| t.to_dtype(DType::F32)) {
                                if let Ok(t_data) = t_f32.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                                    let anchor_count = (0..s).filter(|&i| i < 4 || i % 8 == 0).count();
                                    let mut anchors = vec![0.0f32; b * h * anchor_count * d];
                                    let mut packed_residuals = vec![0u8; (b * h * s * d + 7) / 8];
                                    let mut scales = vec![0.0f32; b * h * s];
                                    let head_token_size = s * d;
                                    for bh_idx in 0..(b * h) {
                                        let bh_offset = bh_idx * head_token_size;
                                        for i in 0..head_token_size { if t_data[bh_offset + i] >= 0.0 { packed_residuals[(bh_offset + i) / 8] |= 1 << ((bh_offset + i) % 8); } }
                                        for token_idx in 0..s {
                                            let token_data = &t_data[bh_offset + token_idx * d .. bh_offset + (token_idx + 1) * d];
                                            if token_idx < 4 || token_idx % 8 == 0 {
                                                let anchor_pos = if token_idx < 4 { token_idx } else { 4 + (token_idx - 4) / 8 };
                                                anchors[(bh_idx * anchor_count + anchor_pos) * d .. (bh_idx * anchor_count + anchor_pos + 1) * d].copy_from_slice(token_data);
                                            }
                                            let mut max_abs = 0.0f32;
                                            for &v in token_data { let a = v.abs(); if a > max_abs { max_abs = a; } }
                                            scales[bh_idx * s + token_idx] = max_abs;
                                        }
                                    }
                                    if let Ok(at) = Tensor::from_vec(anchors, vec![b, h, anchor_count, d], &Device::Cpu) { map.insert("k_anchors".to_string(), at); }
                                    let packed_len = packed_residuals.len();
                                    if let Ok(pt) = Tensor::from_vec(packed_residuals, vec![packed_len], &Device::Cpu) { map.insert("k_packed".to_string(), pt); }
                                    if let Ok(st) = Tensor::from_vec(scales, vec![b, h, s, 1], &Device::Cpu) { map.insert("k_scales".to_string(), st); }
                                    if let Ok(sh) = Tensor::from_vec(vec![b as u32, h as u32, s as u32, d as u32], (4,), &Device::Cpu) { map.insert("k_shape".to_string(), sh); }
                                }
                            }
                        }

                        if let Ok(dims) = layer.v.dims4() {
                            let (b, h, s, d) = dims;
                            if let Ok(t_f32) = layer.v.to_device(&Device::Cpu).and_then(|t| t.to_dtype(DType::F32)) {
                                if let Ok(t_data) = t_f32.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                                    let anchor_count = (0..s).filter(|&i| i < 4 || i % 8 == 0).count();
                                    let mut anchors = vec![0.0f32; b * h * anchor_count * d];
                                    let mut packed_residuals = vec![0u8; (b * h * s * d + 7) / 8];
                                    let mut scales = vec![0.0f32; b * h * s];
                                    let head_token_size = s * d;
                                    for bh_idx in 0..(b * h) {
                                        let bh_offset = bh_idx * head_token_size;
                                        for i in 0..head_token_size { if t_data[bh_offset + i] >= 0.0 { packed_residuals[(bh_offset + i) / 8] |= 1 << ((bh_offset + i) % 8); } }
                                        for token_idx in 0..s {
                                            let token_data = &t_data[bh_offset + token_idx * d .. bh_offset + (token_idx + 1) * d];
                                            if token_idx < 4 || token_idx % 8 == 0 {
                                                let anchor_pos = if token_idx < 4 { token_idx } else { 4 + (token_idx - 4) / 8 };
                                                anchors[(bh_idx * anchor_count + anchor_pos) * d .. (bh_idx * anchor_count + anchor_pos + 1) * d].copy_from_slice(token_data);
                                            }
                                            let mut max_abs = 0.0f32;
                                            for &v in token_data { let a = v.abs(); if a > max_abs { max_abs = a; } }
                                            scales[bh_idx * s + token_idx] = max_abs;
                                        }
                                    }
                                    if let Ok(at) = Tensor::from_vec(anchors, vec![b, h, anchor_count, d], &Device::Cpu) { map.insert("v_anchors".to_string(), at); }
                                    let packed_len = packed_residuals.len();
                                    if let Ok(pt) = Tensor::from_vec(packed_residuals, vec![packed_len], &Device::Cpu) { map.insert("v_packed".to_string(), pt); }
                                    if let Ok(st) = Tensor::from_vec(scales, vec![b, h, s, 1], &Device::Cpu) { map.insert("v_scales".to_string(), st); }
                                }
                            }
                        }

                        if let Ok(mode_tensor) = Tensor::from_vec(vec![3u32], (1,), &Device::Cpu) { map.insert("mode".to_string(), mode_tensor); }

                        if let Err(e) = io_tx.send(SaveTask { slot_id, layer_idx: layer.layer_idx, path, tensors: map, registry: registry.clone(), offset }).await {
                            println!("[SLOT-WORKER] Fatal error: io_tx channel closed: {}", e);
                        }
                    }
                }
                SlotTask::Load(load) => {
                    let (offset, index) = {
                        let inner = load.shared_block.inner.read().unwrap();
                        (inner.offset, inner.index)
                    };
                    let layer_idx = load.layer_idx;
                    
                                        // [FIX] 우선순위 기반 파일명 후보 생성
                    
                                        let mut candidates = Vec::new();
                    
                                        if let Some(name) = &load.kv_name {
                    
                                            candidates.push(if offset == 0 { format!("layer_{}_{}_kv.safetensors", name, layer_idx) } 
                    
                                                             else { format!("layer_{}_{}_kv_{}.safetensors", name, layer_idx, offset) });
                    
                                            candidates.push(if offset == 0 { format!("layer_{}_0_kv.safetensors", name) } 
                    
                                                             else { format!("layer_{}_0_kv_{}.safetensors", name, offset) });
                    
                                        }
                    
                                        candidates.push(if offset == 0 { format!("layer_{}_kv.safetensors", layer_idx) } 
                    
                                                         else { format!("layer_{}_kv_{}.safetensors", layer_idx, offset) });
                    
                                        candidates.push(if offset == 0 { "layer_0_kv.safetensors".to_string() } 
                    
                                                         else { format!("layer_0_kv_{}.safetensors", offset) });
                    
                    
                    
                                                                                                    let mut path = load.path.clone();
                    
                    
                    
                                                                                                    let mut found = false;
                    
                    
                    
                                                                                                    let mut target_name = String::new();
                    
                    
                    
                                                                                                    
                    
                    
                    
                                                                                                    println!("[IO-READ-DEBUG] Slot {} Layer {} Offset {} checking candidates in {:?}", load.slot_id, layer_idx, offset, load.path);
                    
                    
                    
                                                                                                    for (i, cand) in candidates.iter().enumerate() {
                    
                    
                    
                                                                                                        if i == 0 { target_name = cand.clone(); }
                    
                    
                    
                                                                                                        let p = load.path.join(cand);
                    
                    
                    
                                                                                                        let exists = p.exists();
                    
                    
                    
                                                                                                        println!("  > Candidate: {} (Exists: {})", cand, exists);
                    
                    
                    
                                                                                                        if exists { path = p; found = true; break; }
                    
                    
                    
                                                                                                    }
                    
                    
                    
                                        
                    
                    
                    
                                                            if !found {
                    
                    
                    
                                                                println!("[SLOT-WORKER-FAIL] Failed to locate KV file. Target: {:?}, Layer: {}, Offset: {}", target_name, layer_idx, offset);
                    
                    
                    
                                                            }

                    let mut success = false;
                    if found {
                        match std::fs::read(&path) {
                            Ok(content) => {
                                if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                                    let mode = if let Ok(view) = st.tensor("mode") { u32::from_le_bytes(view.data()[0..4].try_into().unwrap()) } else { 1 };

                                    let dequant_res: Result<(Tensor, Tensor)> = (|| {
                                        if mode == 3 {
                                            let k_raw = crate::models::qwen3vl::quantized_model::decompress_bitkv_cpu(
                                                st.tensor("k_anchors")?, st.tensor("k_packed")?, st.tensor("k_scales")?, 
                                                &if let Ok(v) = st.tensor("k_shape") {
                                                    let s: &[u32] = unsafe { std::slice::from_raw_parts(v.data().as_ptr() as *const u32, v.data().len() / 4) };
                                                    s.iter().map(|&x| x as usize).collect()
                                                } else { vec![1, 8, 1024, 128] }
                                            )?;
                                            
                                            let v_raw = crate::models::qwen3vl::quantized_model::decompress_bitkv_cpu(
                                                st.tensor("v_anchors")?, st.tensor("v_packed")?, st.tensor("v_scales")?, 
                                                &if let Ok(v) = st.tensor("k_shape") {
                                                    let s: &[u32] = unsafe { std::slice::from_raw_parts(v.data().as_ptr() as *const u32, v.data().len() / 4) };
                                                    s.iter().map(|&x| x as usize).collect()
                                                } else { vec![1, 8, 1024, 128] }
                                            )?;

                                            // [LINEAR-BRIDGE] 0.6B(8 heads) -> 2B(16 heads) 자동 변환
                                            let (mut k, mut v) = (k_raw, v_raw);
                                            let h = k.dim(1)?;
                                            if h == 8 {
                                                // 2B 모델을 위해 헤드를 8 -> 16으로 확장 (단순 복제 방식)
                                                k = Tensor::cat(&[&k, &k], 1)?;
                                                v = Tensor::cat(&[&v, &v], 1)?;
                                            }
                                            
                                            Ok((k, v))
                                        } else {
                                            Err(anyhow!("Legacy mode not supported in worker"))
                                        }
                                    })();

                                    if let Ok((k, v)) = dequant_res {
                                        let slot = SLOT_MANAGER.get_slot(load.slot_id).unwrap();
                                        {
                                            let mut k_guard = slot.k_layers[layer_idx].lock().await;
                                            *k_guard = Some(k);
                                        }
                                        {
                                            let mut v_guard = slot.v_layers[layer_idx].lock().await;
                                            *v_guard = Some(v);
                                        }
                                        success = true;
                                    } else if let Err(e) = dequant_res {
                                        println!("[SLOT-WORKER-ERROR] Dequant failed: {}", e);
                                    }
                                } else {
                                    println!("[SLOT-WORKER-ERROR] Failed to deserialize SafeTensors from {:?}", path);
                                }
                            }
                            Err(e) => {
                                println!("[SLOT-WORKER-ERROR] Failed to read file {:?}: {}", path, e);
                            }
                        }
                    }
                    
                    if success {
                        {
                            let mut reg = load.registry.entries.write().unwrap();
                            if index < reg.len() {
                                reg[index].location[layer_idx] = KVLocation::RAM;
                                reg[index].slot_ids[layer_idx] = Some(load.slot_id);
                            }
                        }
                        // [FIX] 블록 자체의 상태도 업데이트하여 Polling 루프 즉시 탈출 지원
                        {
                            let mut inner = load.shared_block.inner.write().unwrap();
                            inner.location = KVLocation::RAM;
                        }
                        SLOT_MANAGER.mark_ready(load.slot_id).await;
                    } else {
                        println!("[SLOT-WORKER-FAIL] Could not find file for Layer {} at {:?}", layer_idx, path);
                        {
                            let mut reg = load.registry.entries.write().unwrap();
                            if index < reg.len() { reg[index].location[layer_idx] = KVLocation::SSD; }
                        }
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
    pub async fn forward(&mut self, input_ids: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, video_pixel_values: Option<&Tensor>, video_grid_thw: Option<&Tensor>, cache_position: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>) -> Result<Tensor> {
        match self {
            Self::Standard(m) => m.forward(input_ids, pixel_values, image_grid_thw, video_pixel_values, video_grid_thw, cache_position, seqlen_offset),
            Self::QuantizedVL(m) => m.forward(input_ids, pixel_values, image_grid_thw, video_pixel_values, video_grid_thw, cache_position, seqlen_offset, total_len, session_id).await,
            Self::QuantizedText(m) => m.forward(input_ids, cache_position, seqlen_offset, total_len, session_id).await,
        }
    }

    pub fn rebalance_layers(&mut self, device_id: usize) -> Result<()> {
        match self {
            Self::Standard(_) => Ok(()), 
            Self::QuantizedVL(m) => m.rebalance_layers(device_id),
            Self::QuantizedText(m) => m.rebalance_layers(device_id),
        }
    }

    pub fn drop_kv_storage(&mut self) -> Result<()> {
        match self {
            Self::Standard(_) => Ok(()),
            Self::QuantizedVL(m) => m.language_model.drop_kv_storage(),
            Self::QuantizedText(m) => m.language_model.drop_kv_storage(),
        }
    }

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
        let path = if let Some(stripped) = path.strip_prefix(r"\\?\") { stripped } else { path };
        let tok_path = tokenizer_path.unwrap_or(path);
        let tok_path = if let Some(stripped) = tok_path.strip_prefix(r"\\?\") { stripped } else { tok_path };
        let cfg_path = config_path.unwrap_or(path);
        let cfg_path = if let Some(stripped) = cfg_path.strip_prefix(r"\\?\") { stripped } else { cfg_path };

        let chat_template = ChatTemplate::init(tok_path)?;
        let tokenizer = TokenizerModel::init(tok_path)?;
        let final_config_path = std::path::Path::new(cfg_path).join("config.json");
        let raw_config: serde_json::Value = serde_json::from_slice(&std::fs::read(&final_config_path)?)?;

        let cfg: Qwen3VLConfig = if raw_config.get("text_config").is_some() {
            serde_json::from_value(raw_config)?
        } else {
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
                vision_config: None,
                vision_start_token_id: None,
                vision_end_token_id: None,
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
            let mut model_path = gguf_files.iter().find(|f| f.contains("Qwen3-0.6B-Q8_0.gguf")).cloned();
            if model_path.is_none() { model_path = gguf_files.iter().find(|f| f.contains("Qwen3-0.6B-Q4_K_M.gguf")).cloned(); }
            if model_path.is_none() { model_path = gguf_files.iter().find(|f| !f.contains("mmproj")).cloned(); }

            let limit_tokens = hard_token_limit.unwrap_or(4096) as u64;  
            let reserve_tokens = limit_tokens.min(8192);
            let kv_reserve = reserve_tokens * 40000;

            if is_vision_model {
                let mmproj = mmproj_path.ok_or(anyhow!("Missing mmproj GGUF"))?;
                let main = model_path.ok_or(anyhow!("Missing main GGUF for VL model"))?;
                let main_file = std::fs::File::open(&main)?;
                let main_mmap = unsafe { memmap2::MmapOptions::new().map(&main_file)? };
                let mmproj_file = std::fs::File::open(&mmproj)?;
                let mmproj_mmap = unsafe { memmap2::MmapOptions::new().map(&mmproj_file)? };
                let mut main_cursor = std::io::Cursor::new(&main_mmap[..]);
                let main_content = gguf_file::Content::read(&mut main_cursor)?;
                let mut mmproj_cursor = std::io::Cursor::new(&mmproj_mmap[..]);
                let mmproj_content = gguf_file::Content::read(&mut mmproj_cursor)?;
                let model = QuantizedQwen3VLModel::new_with_mmap(&cfg, &main_content, Some(Arc::new(main_mmap)), &mmproj_content, Some(Arc::new(mmproj_mmap)), &text_dev, text_device_id, &vision_dev, vision_device_id, dtype, kv_reserve, baking_only)?;
                ModelVariant::QuantizedVL(model)
            } else {
                let main = model_path.or_else(|| if !gguf_files.is_empty() { Some(gguf_files[0].clone()) } else { None }).ok_or(anyhow!("No GGUF file found"))?;
                let file = std::fs::File::open(&main)?;
                let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
                let mut cursor = std::io::Cursor::new(&mmap[..]);        
                let content = gguf_file::Content::read(&mut cursor)?;
                let is_06b = path.contains("0.6B");
                let actual_baking_only = baking_only || is_06b;
                let single_layer_mode = baking_only || is_06b;
                let model = crate::models::qwen3vl::quantized_model::QuantizedQwen3TextModel::new_with_mmap(&cfg, &content, Some(Arc::new(mmap)), &text_dev, text_device_id, dtype, kv_reserve, actual_baking_only, single_layer_mode)?;
                ModelVariant::QuantizedText(model)
            }
        } else {
            let model_list = find_type_files(path, "safetensors")?      ;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &text_dev)? };
            let model = Qwen3VLModel::new(cfg, vb)?;
            ModelVariant::Standard(model)
        };

        let generation_config_path = std::path::Path::new(cfg_path).join("generation_config.json");
        let generation_config: Qwen3VLGenerationConfig = if generation_config_path.exists() {
            serde_json::from_slice(&std::fs::read(generation_config_path)?)? 
        } else {
            Qwen3VLGenerationConfig::default()
        };
        let model_name = if path.contains("0.6B") { "qwen3vl-0.6B".to_string() } else { "qwen3vl-2B".to_string() };
        let (eos_token_id1, eos_token_id2) = match &generation_config.eos_token_id {
            serde_json::Value::Number(n) => { let id = n.as_u64().unwrap_or(151645) as u32; (id, id) },
            serde_json::Value::Array(arr) => { let id1 = arr.get(0).and_then(|v| v.as_u64()).unwrap_or(151643) as u32; let id2 = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(id1 as u64) as u32; (id1, id2) },
            _ => (151643, 151643),
        };

        Ok(Self { chat_template, tokenizer, pre_processor, qwen3_vl, text_device: text_dev, vision_device: vision_dev, eos_token_id1, eos_token_id2, generation_config, model_name, hard_token_limit, kv_root })
    }

    pub async fn prefill_text_only(&mut self, text: &str, cancel_token: Option<Arc<AtomicBool>>, mut relay_target: Option<&mut Qwen3VLGenerateModel>, auto_save_path: Option<&std::path::Path>) -> Result<()> {
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
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, total_tokens, None).await?;
            if let Some(path) = auto_save_path { let _ = self.save_kv_to_disk(path, None, end); }
            if let Some(ref mut target) = relay_target {
                let (ks, vs) = self.get_current_kv();
                let results: Result<Vec<_>> = ks.par_iter().zip(vs.par_iter()).map(|(k, v): (&Tensor, &Tensor)| {
                    let seq_len = k.dim(candle_core::D::Minus2)?;
                    let start = seq_len.saturating_sub(end - current_pos);
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
                if !k_anchors.is_empty() { target.inject_kv_bitkv(&k_anchors, &k_packed, &k_scales, &v_anchors, &v_packed, &v_scales, &original_shape)?; }
            }
            current_pos = end;
        }
        if auto_save_path.is_some() { let _ = self.qwen3_vl.drop_kv_storage(); }
        Ok(())
    }

    pub async fn prefill_only(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, _relay_target: Option<&mut Qwen3VLGenerateModel>, kv_name: Option<String>) -> Result<usize> {
        // [STAGE-RESET] 작업 시작 전 슬롯 초기화
        SLOT_MANAGER.reset_all_slots().await;

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let full_input_ids_vec = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_tokens = full_input_ids_vec.len();
        
        // [증분 저장] 2048 토큰 단위로 기억 조각을 생성
        let prefill_chunk_size = 2048;
        let mut current_pos = self.get_kv_len();
        if current_pos > 0 { println!("[RESUME] Resuming from token {}.", current_pos); }

        while current_pos < total_tokens {
            if let Some(flag) = &cancel_flag { 
                if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } 
            }

            let end = (current_pos + prefill_chunk_size).min(total_tokens);
            let chunk_len = end - current_pos;
            let chunk = &full_input_ids_vec[current_pos..end];
            
            let chunk_ids = Tensor::from_vec(chunk.to_vec(), (1, chunk_len), &self.text_device)?;
            let chunk_pos = Tensor::arange(current_pos as u32, end as u32, &self.text_device)?.unsqueeze(0)?;

            // 1. GPU 추론 진행
            println!("[BAKING] {} to {} / Total: {}", current_pos, end, total_tokens);
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, total_tokens, session_id.clone()).await?;

            // [LOG-ENHANCED] 상세 슬롯 상태 출력
            SLOT_MANAGER.debug_stats();

            // 2. [HANDOFF] 슬롯 확보 및 VRAM -> RAM 전송
            if let Some(sid) = &session_id {
                let result = async {
                    let path = crate::utils::paths::get_kv_dir(None).join(sid);
                    if !path.exists() { let _ = fs::create_dir_all(&path); }

                    // KV 캐시 추출
                    let (ks, vs) = self.get_current_kv();
                    if ks.is_empty() {
                        return Err(anyhow!("No KV data to offload."));
                    }

                    // [MULTI-BLOCK-FIX] 청크 내의 모든 블록을 순회하며 등록
                    let num_blocks_in_chunk = (chunk_len + 1023) / 1024;
                    
                    // [REGISTRY-SYNC] Get Registry
                    let registry = match self.qwen3_vl {
                        ModelVariant::QuantizedText(ref m) => Some(m.language_model.registry.clone()),
                        ModelVariant::QuantizedVL(ref m) => Some(m.language_model.registry.clone()),
                        _ => None,
                    };

                    for b_idx in 0..num_blocks_in_chunk {
                        let block_offset_in_chunk = b_idx * 1024;
                        let current_block_len = (chunk_len - block_offset_in_chunk).min(1024);
                        let global_block_index = (current_pos + block_offset_in_chunk) / 1024;

                        // [BASE-SLOT-UPDATE] 해당 블록의 데이터를 Base Slot에 저장
                        // (현재 설계상 Base Slot은 레이어당 1개이므로 가장 최신 블록만 RAM 상주)
                        for (l_idx, (k, v)) in ks.iter().zip(vs.iter()).enumerate() {
                            if l_idx < 28 {
                                let s_len = k.dim(2)?;
                                // 청크 내에서 해당 블록의 위치를 정확히 잘라냄
                                let k_slice = k.narrow(2, s_len - chunk_len + block_offset_in_chunk, current_block_len)?.contiguous()?;
                                let v_slice = v.narrow(2, s_len - chunk_len + block_offset_in_chunk, current_block_len)?.contiguous()?;
                                
                                let base_slot = SLOT_MANAGER.get_base_slot(l_idx);
                                *base_slot.k_layers[0].lock().await = Some(k_slice.clone());
                                *base_slot.v_layers[0].lock().await = Some(v_slice.clone());
                                base_slot.state.store(2, Ordering::SeqCst);
                                base_slot.ready_signal.notify_waiters();
                            }
                        }

                        if let Some(reg_obj) = registry.as_ref() {
                            let mut reg = reg_obj.entries.write().unwrap();
                            let entry = RegistryEntry {
                                location: vec![KVLocation::RAM; 28],
                                slot_ids: (0..28).map(|i| Some(i)).collect(),
                                token_start: current_pos + block_offset_in_chunk,
                                token_len: current_block_len,
                                ssd_path: Some(path.clone()),
                            };

                            if reg.len() <= global_block_index {
                                reg.push(entry);
                            } else {
                                reg[global_block_index] = entry;
                            }
                        }

                        // [SHUTTLE-START] Sub Slot을 확보하여 SSD로 전송 (각 블록별로 개별 전송)
                        let sub_slot_id = SLOT_MANAGER.acquire_sub_slot().await;
                        let mut layer_dumps = Vec::new();
                        for (l_idx, (k, v)) in ks.iter().zip(vs.iter()).enumerate() {
                            let s_len = k.dim(2)?;
                            let k_slice = k.narrow(2, s_len - chunk_len + block_offset_in_chunk, current_block_len)?.contiguous()?;
                            let v_slice = v.narrow(2, s_len - chunk_len + block_offset_in_chunk, current_block_len)?.contiguous()?;
                            layer_dumps.push(LayerKVDump { layer_idx: l_idx, k: k_slice, v: v_slice });
                        }

                        match get_worker_channel().await {
                            Ok(tx) => {
                                tx.send(SlotTask::Bake(BakeTask {
                                    slot_id: sub_slot_id,
                                    task_dir: path.clone(),
                                    kv_name: kv_name.clone(),
                                    offset: current_pos + block_offset_in_chunk,
                                    layers: layer_dumps,
                                    registry: registry.clone(),
                                })).await.map_err(|e| anyhow!("Failed to send to shuttle worker: {}", e))?;
                            },
                            Err(e) => return Err(e)
                        }
                    }
                    Ok(())
                }.await;

                if let Err(e) = result {
                    println!("[BAKE-ERROR] Failed to handoff to shuttle: {}", e);
                }

                self.clear_temporal_kv_caches();
            }

            current_pos = end;
            println!("[MEMORY] 73. 추론 진행 메모리 (Ready)");
        }

        if let Some(sid) = session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(sid);
            let token_path = path.join("tokens.json");
            if let Ok(file) = fs::File::create(&token_path) { let _ = serde_json::to_writer(file, &full_input_ids_vec); }
        }

        // [CRITICAL] SSD 저장이 완료될 때까지 대기하여 다음 단계(Purge/Load)와의 충돌 방지
        SLOT_MANAGER.wait_all_sub_slots().await;

        Ok(current_pos)
    }

    pub async fn prefill_chunk(&mut self, text: String, _cancel_flag: Option<Arc<AtomicBool>>, mut relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let chunk_ids_vec = self.tokenizer.text_encode_vec(text, false)?;
        let chunk_size = chunk_ids_vec.len();
        let current_pos = self.get_kv_len();
        let chunk_ids = Tensor::from_vec(chunk_ids_vec, (1, chunk_size), &self.text_device)?;
        let chunk_pos = Tensor::arange(current_pos as u32, (current_pos + chunk_size) as u32, &self.text_device)?.unsqueeze(0)?;
        self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, chunk_size, None).await?;
        if let Some(ref mut target) = relay_target {
            let (ks, vs) = self.get_current_kv();
            let results: Result<Vec<_>> = ks.par_iter().zip(vs.par_iter()).map(|(k, v): (&Tensor, &Tensor)| {
                let s_len = k.dim(candle_core::D::Minus2)?;
                let k_new = k.narrow(candle_core::D::Minus2, s_len - chunk_size, chunk_size)?;
                let v_new = v.narrow(candle_core::D::Minus2, s_len - chunk_size, chunk_size)?;
                if let ModelVariant::QuantizedText(m) = &self.qwen3_vl {
                    let rk = m.language_model.compress_to_bitkv(&k_new)?;
                    let rv = m.language_model.compress_to_bitkv(&v_new)?;
                    Ok((rk, rv))
                } else { Err(anyhow!("Unsupported")) }
            }).collect();
            let results = results?;
            let mut ka = vec![]; let mut kp = vec![]; let mut ks_ = vec![];
            let mut va = vec![]; let mut vp = vec![]; let mut vs_ = vec![];
            let mut os = vec![];
            for (rk, rv) in results {
                ka.push(rk.0); kp.push(rk.1); ks_.push(rk.2);
                va.push(rv.0); vp.push(rv.1); vs_.push(rv.2);
                os = rk.3;
            }
            if !ka.is_empty() { target.inject_kv_bitkv(&ka, &kp, &ks_, &va, &vp, &vs_, &os)?; }
        }
        Ok(chunk_size)
    }

    pub async fn generate(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, kv_name: Option<String>) -> Result<String> {
        // [STAGE-RESET] 작업 시작 전 슬롯 초기화
        SLOT_MANAGER.reset_all_slots().await;

        let temperature = mes.temperature.unwrap_or(0.7) as f32;
        let top_p = mes.top_p.unwrap_or(0.9) as f32;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(Some(temperature), Some(top_p), Some(40), seed);
        let mut all_ids = vec![];
        let mut generated_text = String::new();
        let mut seqlen_offset = self.get_kv_len();

        if let Some(sid) = &session_id {
            let path = crate::utils::paths::get_kv_dir(None).join(sid);
            let progress_path = path.join("generation_progress.json");
            if progress_path.exists() {
                if let Ok(data) = std::fs::read_to_string(&progress_path) {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                        println!("[RESUME] Found progress. Loading...");
                        let _ = self.load_kv_from_disk(&path, kv_name.as_deref());
                        seqlen_offset = self.get_kv_len();
                        generated_text = json["text"].as_str().unwrap_or("").to_string();
                        if let Some(ids) = json["ids"].as_array() { all_ids = ids.iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect(); }
                    }
                }
            }
        }

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut input = self.pre_processor.process_info(&mes, &mes_render)?;
        let full_input_ids_vec = self.tokenizer.text_encode_vec(input.replace_text.clone(), false)?;
        let total_tokens = full_input_ids_vec.len();

        if all_ids.is_empty() {
            all_ids = full_input_ids_vec.clone();
            if seqlen_offset == 0 {
                if let Some(sid) = &session_id {
                    let path = crate::utils::paths::get_kv_dir(None).join(sid);
                    if path.exists() { self.load_kv_from_disk(&path, kv_name.as_deref())?; seqlen_offset = self.get_kv_len(); }
                }
            }
        }

        let mut local_pos = if seqlen_offset > 0 { if seqlen_offset >= total_tokens { total_tokens.saturating_sub(1) } else { seqlen_offset } } else { 0 };
        let prefill_chunk_size = 128;

        while local_pos < total_tokens {
            let remaining = total_tokens - local_pos;
            if remaining == 1 { break; }
            let mut chunk_size = if remaining > prefill_chunk_size { prefill_chunk_size } else { remaining };
            if local_pos + chunk_size >= total_tokens { chunk_size = (total_tokens - local_pos).saturating_sub(1); }
            if chunk_size == 0 { break; }
            let chunk = &full_input_ids_vec[local_pos..local_pos + chunk_size];
            let chunk_ids = Tensor::from_vec(chunk.to_vec(), (1, chunk_size), &self.text_device)?;
            let chunk_pos = Tensor::arange(seqlen_offset as u32, (seqlen_offset + chunk_size) as u32, &self.text_device)?.unsqueeze(0)?;
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), seqlen_offset, total_tokens, session_id.clone()).await?;
            local_pos += chunk_size;
            seqlen_offset += chunk_size;
            if seqlen_offset % 1024 == 0 && seqlen_offset > 0 {
                println!("[DEBUG-BAKE] Prefill Trigger at {}", seqlen_offset);
                if let Some(sid) = &session_id {
                    let slot_id = SLOT_MANAGER.acquire_sub_slot().await;
                    let path = crate::utils::paths::get_kv_dir(None).join(sid);
                    
                    let result = async {
                        if !path.exists() { let _ = fs::create_dir_all(&path); }

                        let (ks, vs) = self.get_current_kv();
                        if ks.is_empty() {
                            return Err(anyhow!("No KV data to checkpoint."));
                        }

                        let mut layer_dumps = Vec::new();
                        for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() {
                            layer_dumps.push(LayerKVDump { layer_idx: idx, k: k.clone(), v: v.clone() });
                        }
                        
                        match get_worker_channel().await {
                            Ok(tx) => {
                                tx.send(SlotTask::Bake(BakeTask {
                                    slot_id, task_dir: path, kv_name: kv_name.clone(), offset: seqlen_offset, layers: layer_dumps, registry: None
                                })).await.map_err(|e| anyhow!("Send failed: {}", e))?;
                                Ok(())
                            },
                            Err(e) => Err(e)
                        }
                    }.await;

                    if let Err(e) = result {
                        println!("[GEN-PREFILL] Checkpoint skipped: {}. Releasing slot {}.", e, slot_id);
                        SLOT_MANAGER.release_slot(slot_id).await;
                    } else {
                        println!("[GEN-PREFILL] Checkpoint Slot {} at {}", slot_id, seqlen_offset);
                    }
                }
            }
        }

        let max_new_tokens = mes.max_tokens.unwrap_or(2048);
        let mut pixel_values = input.pixel_values.take();
        let image_grid_thw = input.image_grid_thw.take();

        let task_dir_base = if let Some(sid) = &session_id {
            crate::utils::paths::get_kv_dir(None).join(sid)
        } else {
            std::path::PathBuf::new()
        };

        // [SLIDING-WINDOW-CONFIG] 512 block size for more granular management
        let window_keep_size = 1024; 
        let bake_block_size = 512;

        for i in 0..max_new_tokens {
            if let Some(flag) = &cancel_flag { if flag.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); } }
            
            // [WINDOW-BAKE] Trigger dump every 512 tokens
            if seqlen_offset > 0 && seqlen_offset % bake_block_size == 0 && !task_dir_base.as_os_str().is_empty() {
                let slot_id = SLOT_MANAGER.acquire_sub_slot().await;
                
                let result = async {
                    let (ks, vs) = self.get_current_kv();
                    if ks.is_empty() { return Err(anyhow!("No KV data for window bake.")); }

                    let mut layer_dumps = Vec::new();
                    for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() {
                        let current_mem_len = k.dim(2)?;
                        if current_mem_len >= bake_block_size {
                            let k_slice = k.narrow(2, current_mem_len - bake_block_size, bake_block_size)?.contiguous()?;
                            let v_slice = v.narrow(2, current_mem_len - bake_block_size, bake_block_size)?.contiguous()?;
                            layer_dumps.push(LayerKVDump { layer_idx: idx, k: k_slice, v: v_slice });
                        }
                    }
                    
                    if layer_dumps.is_empty() { return Err(anyhow!("Layer dumps empty for window bake.")); }

                    match get_worker_channel().await {
                        Ok(tx) => {
                            tx.send(SlotTask::Bake(BakeTask {
                                slot_id,
                                task_dir: task_dir_base.clone(),
                                kv_name: kv_name.clone().or_else(|| Some("default".to_string())),
                                offset: seqlen_offset - bake_block_size,
                                layers: layer_dumps,
                                registry: None,
                            })).await.map_err(|e| anyhow!("Send failed: {}", e))?;
                            Ok(())
                        },
                        Err(e) => Err(e)
                    }
                }.await;

                if let Err(e) = result {
                    println!("[WINDOW-BAKE] Skipped: {}. Releasing slot {}.", e, slot_id);
                    SLOT_MANAGER.release_slot(slot_id).await;
                } else {
                    println!("[WINDOW-BAKE] {} boundary hit. Offloaded to Slot {}.", seqlen_offset, slot_id);
                    // [SLIDING-WINDOW-PURGE] 
                    let current_kv_len = self.get_kv_len();
                    if current_kv_len > window_keep_size {
                        let purge_len = current_kv_len - window_keep_size;
                        let _ = self.truncate_kv_cache(purge_len);
                    }
                }
            }

            let input_ids = if generated_text.is_empty() && seqlen_offset < total_tokens {
                Tensor::new(&full_input_ids_vec[local_pos..total_tokens], &self.text_device)?.unsqueeze(0)?
            } else { Tensor::new(vec![*all_ids.last().unwrap()], &self.text_device)?.unsqueeze(0)? };
            let seq_len = input_ids.dim(1)?;
            let chunk_pos = Tensor::arange(seqlen_offset as u32, (seqlen_offset + seq_len) as u32, &self.text_device)?.unsqueeze(0)?;
            let logits = self.qwen3_vl.forward(&input_ids, pixel_values.as_ref(), image_grid_thw.as_ref(), None, None, Some(&chunk_pos), seqlen_offset, total_tokens, session_id.clone()).await?;
            let mut logits = logits.squeeze(0)?.i(logits.dim(1)? - 1)?.to_dtype(DType::F32)?;
            if 1.1 != 1.0 { let penalty_context = if all_ids.len() > 512 { &all_ids[all_ids.len()-512..] } else { &all_ids[..] }; logits = apply_repeat_penalty(&logits, 1.1, penalty_context)?; }
            let next_id = logit_processor.sample(&logits)?;
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            all_ids.push(next_id);
            generated_text.push_str(&self.tokenizer.token_decode(vec![next_id])?);
            
            // [INFERENCE-SAVE] Periodic Checkpointing
            if i > 0 && i % 50 == 0 && !self.model_name.contains("0.6B") {
                if let Some(sid) = &session_id {
                    let slot_id = SLOT_MANAGER.acquire_sub_slot().await;
                    let path = crate::utils::paths::get_kv_dir(None).join(sid);
                    
                    let result = async {
                        let (ks, vs) = self.get_current_kv();
                        if ks.is_empty() { return Err(anyhow!("No KV data for inference checkpoint.")); }

                        let mut layer_dumps = Vec::new();
                        for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() {
                            layer_dumps.push(LayerKVDump { layer_idx: idx, k: k.clone(), v: v.clone() });
                        }
                        
                        match get_worker_channel().await {
                            Ok(tx) => {
                                tx.send(SlotTask::Bake(BakeTask {
                                    slot_id, task_dir: path.clone(), kv_name: kv_name.clone(), offset: seqlen_offset, layers: layer_dumps, registry: None
                                })).await.map_err(|e| anyhow!("Send failed: {}", e))?;
                                Ok(())
                            },
                            Err(e) => Err(e)
                        }
                    }.await;

                    if let Err(e) = result {
                        println!("[INFERENCE-SAVE] Checkpoint skipped: {}. Releasing slot {}.", e, slot_id);
                        SLOT_MANAGER.release_slot(slot_id).await;
                    } else {
                        let progress = serde_json::json!({ "text": generated_text, "ids": all_ids });
                        let _ = std::fs::write(path.join("generation_progress.json"), progress.to_string());
                    }
                }
            }
            seqlen_offset += seq_len;
            pixel_values = None;

            // [CLEANUP] Clear temporal VRAM caches for blocks that are already backed up
            self.clear_temporal_kv_caches();
        }
        if let Some(sid) = &session_id { let _ = std::fs::remove_file(crate::utils::paths::get_kv_dir(None).join(sid).join("generation_progress.json")); }
        Ok(generated_text)
    }

    pub fn clear_temporal_kv_caches(&mut self) {
        match self.qwen3_vl {
            ModelVariant::QuantizedText(ref mut m) => {
                let reg_obj = m.language_model.registry.clone();
                let reg = reg_obj.entries.read().unwrap();
                for (layer_idx, layer) in m.language_model.layers.iter_mut().enumerate() {
                    for block in &mut layer.self_attn.kv_blocks {
                        let mut inner = block.inner.write().unwrap();
                        // [OPTIMIZATION] VRAM만 비우고 RAM(Slot)에 있는 데이터는 유지
                        // 이를 통해 'C' 숫자가 쌓이고 SSD 재읽기가 방지됨
                        if inner.location == KVLocation::VRAM {
                            // 이미 SSD나 RAM에 백업된 정보가 있을 때만 VRAM을 비움
                            let reg_loc = if inner.index < reg.len() { 
                                reg[inner.index].location[layer_idx] 
                            } else { 
                                KVLocation::VRAM 
                            };
                            
                            if reg_loc != KVLocation::VRAM {
                                inner.k_cache = None;
                                inner.v_cache = None;
                                inner.location = reg_loc;
                            }
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
                            let reg_loc = if inner.index < reg.len() { 
                                reg[inner.index].location[layer_idx] 
                            } else { 
                                KVLocation::VRAM 
                            };
                            
                            if reg_loc != KVLocation::VRAM {
                                inner.k_cache = None;
                                inner.v_cache = None;
                                inner.location = reg_loc;
                            }
                        }
                    }
                }
            },
            _ => {}
        }
    }

    pub fn get_kv_len(&self) -> usize {
        match &self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.language_model.get_kv_len(),
            ModelVariant::QuantizedText(m) => m.language_model.get_kv_len(),
            _ => 0,
        }
    }

    pub fn get_current_kv(&self) -> (Vec<Tensor>, Vec<Tensor>) {
        let mut ks = vec![]; let mut vs = vec![];
        match &self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => { 
                for l in &m.language_model.layers { 
                    let mut layer_k = Vec::new();
                    let mut layer_v = Vec::new();
                    for b in &l.self_attn.kv_blocks {
                        let inner = b.inner.read().unwrap();
                        if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                            layer_k.push(k.clone());
                            layer_v.push(v.clone());
                        }
                    }
                    if !layer_k.is_empty() {
                        if let (Ok(k), Ok(v)) = (Tensor::cat(&layer_k, 2), Tensor::cat(&layer_v, 2)) {
                            ks.push(k); vs.push(v);
                        }
                    }
                } 
            },
            ModelVariant::QuantizedText(m) => { 
                for l in &m.language_model.layers { 
                    let mut layer_k = Vec::new();
                    let mut layer_v = Vec::new();
                    for b in &l.self_attn.kv_blocks {
                        let inner = b.inner.read().unwrap();
                        if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                            layer_k.push(k.clone());
                            layer_v.push(v.clone());
                        }
                    }
                    if !layer_k.is_empty() {
                        if let (Ok(k), Ok(v)) = (Tensor::cat(&layer_k, 2), Tensor::cat(&layer_v, 2)) {
                            ks.push(k); vs.push(v);
                        }
                    }
                } 
            },
            _ => {} 
        }
        (ks, vs)
    }

    pub fn inject_kv_bitkv(&mut self, ka: &[Tensor], kp: &[Tensor], ks: &[Tensor], va: &[Tensor], vp: &[Tensor], vs: &[Tensor], os: &[usize]) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.language_model.inject_live_kv_bitkv(ka, kp, ks, va, vp, vs, os),
            ModelVariant::QuantizedText(m) => m.language_model.inject_live_kv_bitkv(ka, kp, ks, va, vp, vs, os),
            _ => Ok(()),
        }
    }

    pub fn save_kv_to_disk(&mut self, path: &Path, kv_name: Option<&str>, offset: usize) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.save_kv_cache(path, false, offset, kv_name),
            ModelVariant::QuantizedText(m) => m.save_kv_cache(path, false, offset, kv_name),
            _ => Ok(()),
        }
    }

    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.truncate_kv_cache(len),
            ModelVariant::QuantizedText(m) => m.truncate_kv_cache(len),
            _ => Ok(()),
        }
    }

    pub fn load_kv_from_disk(&mut self, path: &Path, kv_name: Option<&str>) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.load_kv_cache(path, &self.text_device, 0, 128, kv_name),
            ModelVariant::QuantizedText(m) => m.load_kv_cache(path, &self.text_device, 0, 128, kv_name),
            _ => Ok(()),
        }
    }

    pub fn to_device(&mut self, d: &Device) -> Result<()> {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.to_device(d)?,
            ModelVariant::QuantizedText(m) => m.to_device(d)?,
            _ => {},
        }
        self.text_device = d.clone(); self.vision_device = d.clone();
        Ok(())
    }

    pub fn clear_kv_cache(&mut self) {
        match &mut self.qwen3_vl {
            ModelVariant::QuantizedVL(m) => m.clear_kv_cache(),
            ModelVariant::QuantizedText(m) => m.clear_kv_cache(),
            _ => {},
        }
    }
}

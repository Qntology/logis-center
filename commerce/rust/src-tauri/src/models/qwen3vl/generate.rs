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

// [GLOBAL] 슬롯 관리자: 풀(Pool) 기반 유기적 관리
pub struct SlotManager {
    pub slots: Vec<crate::models::qwen3vl::quantized_model::MemorySlot>,
    pub handoff_notifier: Arc<tokio::sync::Notify>,
    pub active_write_count: Arc<AtomicUsize>,
    pub last_picked_idx: Arc<AtomicUsize>, // [NEW] 로테이션을 위한 인덱스
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
        Self {
            slots,
            handoff_notifier: Arc::new(tokio::sync::Notify::new()),
            active_write_count: Arc::new(AtomicUsize::new(0)),
            last_picked_idx: Arc::new(AtomicUsize::new(0)),
            count_reads: Arc::new(AtomicUsize::new(0)),
            count_writes: Arc::new(AtomicUsize::new(0)),
            count_cached: Arc::new(AtomicUsize::new(0)),
            count_free: Arc::new(AtomicUsize::new(count)),
        }
    }

    // 상태 변경 시 카운터 업데이트 유틸리티
    fn update_counters(&self, old_state: u8, new_state: u8) {
        if old_state == new_state { return; }
        let dec = |c: &Arc<AtomicUsize>| { c.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some(v.saturating_sub(1))).ok(); };
        match old_state {
            0 => dec(&self.count_free),
            1 => dec(&self.count_writes),
            2 => dec(&self.count_cached),
            3 => dec(&self.count_reads),
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

    pub async fn reset_all_slots(&self) {
        println!("[SLOT-MANAGER] Force resetting all slots for new stage...");
        for i in 0..self.slots.len() {
            self.release_slot(i).await;
        }
    }

    pub async fn acquire_read_slot(&self) -> usize {
        let count = self.slots.len();
        loop {
            let start_idx = self.last_picked_idx.load(Ordering::Relaxed);
            
            // 1순위: 로테이션하며 완전히 비어있는(Free) 슬롯 찾기
            for i in 0..count {
                let idx = (start_idx + i) % count;
                let slot = &self.slots[idx];
                if slot.state.compare_exchange(0, 3, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    self.update_counters(0, 3);
                    self.last_picked_idx.store((idx + 1) % count, Ordering::Relaxed);
                    return idx;
                }
            }

            // 2순위: 비어있는 게 없다면 이미 캐시된(Ready) 슬롯 중 로테이션하며 찾기
            for i in 0..count {
                let idx = (start_idx + i) % count;
                let slot = &self.slots[idx];
                if slot.state.compare_exchange(2, 3, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    self.prepare_for_reuse(idx).await;
                    self.update_counters(2, 3);
                    self.last_picked_idx.store((idx + 1) % count, Ordering::Relaxed);
                    return idx;
                }
            }

            tokio::time::timeout(std::time::Duration::from_millis(50), self.handoff_notifier.notified()).await.ok();
        }
    }

    pub async fn acquire_write_slot(&self, total_tokens: usize) -> usize {
        let max_writes = self.get_max_concurrent_writes(total_tokens);
        let count = self.slots.len();
        loop {
            if self.active_write_count.load(Ordering::SeqCst) < max_writes {
                let start_idx = self.last_picked_idx.load(Ordering::Relaxed);
                for i in 0..count {
                    let idx = (start_idx + i) % count;
                    if self.slots[idx].state.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                        self.update_counters(0, 1);
                        self.active_write_count.fetch_add(1, Ordering::SeqCst);
                        self.last_picked_idx.store((idx + 1) % count, Ordering::Relaxed);
                        println!("[SLOT-ACQUIRE] Write Slot {} (Rotation)", idx);
                        return idx;
                    }
                }
            }
            tokio::time::timeout(std::time::Duration::from_millis(50), self.handoff_notifier.notified()).await.ok();
        }
    }

    pub async fn release_slot(&self, id: usize) {
        if id < self.slots.len() {
            let prev = self.slots[id].state.swap(0, Ordering::SeqCst);
            if prev != 0 {
                self.update_counters(prev, 0);
                if prev == 1 { self.active_write_count.fetch_sub(1, Ordering::SeqCst); }
                for layer_k in &self.slots[id].k_layers { *layer_k.lock().await = None; }
                for layer_v in &self.slots[id].v_layers { *layer_v.lock().await = None; }
            }
        }
        self.handoff_notifier.notify_waiters();
    }

    // 컨텍스트 크기에 따른 "동시 쓰기 작업 허용 수" 제한
    fn get_max_concurrent_writes(&self, total_tokens: usize) -> usize {
        if total_tokens < 4096 { 12 }
        else if total_tokens < 8192 { 8 }
        else { 4 }
    }

    // 슬롯 재사용 전 초기화 (내부용)
    async fn prepare_for_reuse(&self, id: usize) {
        for layer_k in &self.slots[id].k_layers { *layer_k.lock().await = None; }
        for layer_v in &self.slots[id].v_layers { *layer_v.lock().await = None; }
    }

    pub async fn mark_ready(&self, id: usize) {
        if id < self.slots.len() {
            let current = self.slots[id].state.load(Ordering::SeqCst);
            if (current == 1 || current == 3) && self.slots[id].state.compare_exchange(current, 2, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
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
pub static SLOT_MANAGER: once_cell::sync::Lazy<SlotManager> = once_cell::sync::Lazy::new(|| SlotManager::new(15)); // 1 Central + 14 Workers

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

struct LayerKVDump {
    layer_idx: usize,
    k: Tensor,
    v: Tensor,
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
            let start = std::time::Instant::now();
            
            // [FIX] 경로 확인 및 생성 (tmp 폴더 실패 방지)
            if let Some(parent) = task.path.parent() {
                if !parent.exists() { let _ = std::fs::create_dir_all(parent); }
            }

            // [CRITICAL] 쓰기 시도 및 에러 로깅
            let save_result = candle_core::safetensors::save(&task.tensors, &task.path);
            if let Err(e) = save_result {
                println!("[IO-ERROR] Failed to save SSD chunk to {:?}: {}", task.path, e);
            }
            
            // [FIX] Decrement remaining layers and release ONLY when 0
            let slot = &SLOT_MANAGER.slots[task.slot_id];
            let remaining = slot.remaining_layers.fetch_sub(1, Ordering::SeqCst);
            
            if remaining % 10 == 0 || remaining <= 1 {
                println!("[IO-WORKER] Processed {:?}. Remaining: {}. (Time: {:.2?})", 
                    task.path.file_name().unwrap_or_default(), remaining - 1, start.elapsed());
            }

            if remaining == 1 {
                SLOT_MANAGER.mark_ready(task.slot_id).await;
                println!("[IO-WORKER] Slot {} marked as READY. (WriteCount decreased)", task.slot_id);
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
                    
                    if bake.layers.is_empty() {
                        println!("[SLOT-WORKER] BakeTask has no layers. Releasing slot {}.", slot_id);
                        SLOT_MANAGER.release_slot(slot_id).await;
                        continue;
                    }

                    // [FIX] Initialize layer counter before starting Phase B tasks
                    let slot = &SLOT_MANAGER.slots[slot_id];
                    let total_layers = bake.layers.len();
                    slot.remaining_layers.store(total_layers, Ordering::SeqCst);
                    
                    for layer in bake.layers {
                        // [RAM-STORAGE] Direct Access를 위해 RAM 슬롯에 텐서 저장
                        {
                            if let Ok(mut k_guard) = slot.k_layers[layer.layer_idx].try_lock() {
                                *k_guard = Some(layer.k.clone());
                            }
                            if let Ok(mut v_guard) = slot.v_layers[layer.layer_idx].try_lock() {
                                *v_guard = Some(layer.v.clone());
                            }
                        }

                        let filename = match (&kv_name, offset) {
                            (Some(name), 0) => format!("layer_{}_kv.safetensors", name),
                            (Some(name), off) => format!("layer_{}_kv_{}.safetensors", name, off),
                            (None, 0) => format!("layer_{}_kv.safetensors", layer.layer_idx),
                            (None, off) => format!("layer_{}_kv_{}.safetensors", layer.layer_idx, off),
                        };
                        let path = task_dir.join(filename);
                        
                        let mut map = std::collections::HashMap::new();
                        
                        // [REFACTORED] Move process_tensor logic into a standalone non-capturing way
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
                                        for i in 0..head_token_size {
                                            if t_data[bh_offset + i] >= 0.0 { packed_residuals[(bh_offset + i) / 8] |= 1 << ((bh_offset + i) % 8); }
                                        }
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
                                        for i in 0..head_token_size {
                                            if t_data[bh_offset + i] >= 0.0 { packed_residuals[(bh_offset + i) / 8] |= 1 << ((bh_offset + i) % 8); }
                                        }
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

                        if let Ok(mode_tensor) = Tensor::from_vec(vec![3u32], (1,), &Device::Cpu) {
                            map.insert("mode".to_string(), mode_tensor);
                        }

                        // [CRITICAL] 텐서 처리가 실패하여 map이 비었더라도, 반드시 io_tx로 작업을 보내 카운트를 감소시켜야 합니다.
                        if let Err(e) = io_tx.send(SaveTask { slot_id, path, tensors: map }).await {
                            println!("[SLOT-WORKER] Fatal error: io_tx channel closed: {}", e);
                        }
                    }
                }
                SlotTask::Load(load) => {
                    // [LAYER-LOAD] Load specific layer for this master block
                    let (offset, index) = {
                        let inner = load.shared_block.inner.read().unwrap();
                        (inner.offset, inner.index)
                    };
                    
                    let layer_idx = load.layer_idx;

                    let filename = if offset == 0 { 
                        format!("layer_{}_kv.safetensors", layer_idx) 
                    } else { 
                        format!("layer_{}_kv_{}.safetensors", layer_idx, offset) 
                    };
                    
                    let path = load.path.join(filename);
                    let mut success = false;

                    if path.exists() {
                        if let Ok(content) = std::fs::read(&path) {
                            if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                                let extract_vec = |name: &str| -> Option<Vec<f32>> {
                                    st.tensor(name).ok().map(|v| unsafe {
                                        std::slice::from_raw_parts(v.data().as_ptr() as *const f32, v.data().len() / 4).to_vec()
                                    })
                                };

                                if let (Some(ka), Some(kp), Some(ks)) = (extract_vec("k_anchors"), st.tensor("k_packed").ok(), extract_vec("k_scales")) {
                                    if let (Some(va), Some(vp), Some(vs)) = (extract_vec("v_anchors"), st.tensor("v_packed").ok(), extract_vec("v_scales")) {
                                        
                                        let original_shape = if let Ok(view) = st.tensor("k_shape") {
                                            let shape_u32: &[u32] = unsafe { std::slice::from_raw_parts(view.data().as_ptr() as *const u32, view.data().len() / 4) };
                                            shape_u32.iter().map(|&x| x as usize).collect()
                                        } else { vec![1, 8, 1024, 128] };

                                        // Store metadata only for the requested layer
                                        let mut inner = load.shared_block.inner.write().unwrap();
                                        inner.bitkv_metadata = Some(crate::models::qwen3vl::quantized_model::BitKVMetadata {
                                            k_anchors: Tensor::from_vec(ka, (original_shape[0], original_shape[1], (original_shape[2] + 7) / 8 + 4, original_shape[3]), &Device::Cpu).unwrap(),
                                            k_packed: Tensor::from_slice(kp.data(), kp.shape(), &Device::Cpu).unwrap(),
                                            k_scales: Tensor::from_vec(ks, (original_shape[0], original_shape[1], original_shape[2], 1), &Device::Cpu).unwrap(),
                                            v_anchors: Tensor::from_vec(va, (original_shape[0], original_shape[1], (original_shape[2] + 7) / 8 + 4, original_shape[3]), &Device::Cpu).unwrap(),
                                            v_packed: Tensor::from_slice(vp.data(), vp.shape(), &Device::Cpu).unwrap(),
                                            v_scales: Tensor::from_vec(vs, (original_shape[0], original_shape[1], original_shape[2], 1), &Device::Cpu).unwrap(),
                                            original_shape,
                                        });
                                        success = true;
                                    }
                                }
                            }
                        }
                    }
                    
                    if success {
                        {
                            let mut reg = load.registry.entries.write().unwrap();
                            if index < reg.len() {
                                // [2D-UPDATE] 해당 블록의 특정 레이어만 RAM 상태로 변경
                                reg[index].location[layer_idx] = KVLocation::RAM;
                                reg[index].slot_ids[layer_idx] = Some(load.slot_id);
                            }
                        }
                        // Note: For layer-wise distribution, we don't hold the slot globally for all layers.
                        // However, to satisfy SLOT_MANAGER stats, we mark it Ready.
                        SLOT_MANAGER.mark_ready(load.slot_id).await;
                    } else {
                        {
                            let mut reg = load.registry.entries.write().unwrap();
                            if index < reg.len() {
                                reg[index].location[layer_idx] = KVLocation::SSD;
                            }
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
    pub fn forward(&mut self, input_ids: &Tensor, pixel_values: Option<&Tensor>, image_grid_thw: Option<&Tensor>, video_pixel_values: Option<&Tensor>, video_grid_thw: Option<&Tensor>, cache_position: Option<&Tensor>, seqlen_offset: usize, total_len: usize, session_id: Option<String>) -> Result<Tensor> {
        match self {
            Self::Standard(m) => m.forward(input_ids, pixel_values, image_grid_thw, video_pixel_values, video_grid_thw, cache_position, seqlen_offset),
            Self::QuantizedVL(m) => m.forward(input_ids, pixel_values, image_grid_thw, video_pixel_values, video_grid_thw, cache_position, seqlen_offset, total_len, session_id),
            Self::QuantizedText(m) => m.forward(input_ids, cache_position, seqlen_offset, total_len, session_id),
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

    pub async fn prefill_only(&mut self, mes: ChatCompletionParameters, cancel_flag: Option<Arc<AtomicBool>>, session_id: Option<String>, mut relay_target: Option<&mut Qwen3VLGenerateModel>, kv_name: Option<String>) -> Result<usize> {
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
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, total_tokens, session_id.clone())?;

            // 2. [HANDOFF] 슬롯 확보 및 VRAM -> RAM 전송
            if let Some(sid) = &session_id {
                let slot_id = SLOT_MANAGER.acquire_write_slot(total_tokens).await;
                
                let result = async {
                    let path = crate::utils::paths::get_kv_dir(None).join(sid);
                    if !path.exists() { let _ = fs::create_dir_all(&path); }

                    // KV 캐시 추출
                    let (ks, vs) = self.get_current_kv();
                    if ks.is_empty() {
                        return Err(anyhow!("No KV data to offload. Possibly auto-purged by forward pass."));
                    }

                    let mut layer_dumps = Vec::new();
                    for (idx, (k, v)) in ks.into_iter().zip(vs.into_iter()).enumerate() {
                        let s_len = k.dim(2)?;
                        if s_len < chunk_len {
                            return Err(anyhow!("KV length mismatch: {} vs chunk {}", s_len, chunk_len));
                        }
                        // 현재 청크에 해당하는 부분만 추출
                        let k_slice = k.narrow(2, s_len - chunk_len, chunk_len)?.contiguous()?;
                        let v_slice = v.narrow(2, s_len - chunk_len, chunk_len)?.contiguous()?;
                        layer_dumps.push(LayerKVDump { layer_idx: idx, k: k_slice, v: v_slice });
                    }

                                    // [HANDOFF] VRAM -> SSD Transition
                                    // Sync ALL layers via central registry so every layer knows where its context is stored
                                    let registry = match self.qwen3_vl {
                                        ModelVariant::QuantizedText(ref m) => Some(m.language_model.registry.clone()),
                                        ModelVariant::QuantizedVL(ref m) => Some(m.language_model.registry.clone()),
                                        _ => None,
                                    };
                    
                                    if let Some(reg_obj) = registry {
                                        let mut reg = reg_obj.entries.write().unwrap();
                                        let block_index = self.get_kv_len() / 1024; // Approximate
                                        if block_index < reg.len() {
                                            for l_idx in 0..28 {
                                                if reg[block_index].location[l_idx] == KVLocation::VRAM {
                                                    reg[block_index].location[l_idx] = KVLocation::SSD;
                                                    reg[block_index].ssd_path = Some(path.clone());
                                                }
                                            }
                                        }
                                    }
                    // Also clear physical caches in the blocks
                    self.clear_temporal_kv_caches();

                    let dev = self.text_device.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        let _ = dev.synchronize();
                    }).await;

                    // 워커가 준비될 때까지 명시적으로 대기 (헬퍼 함수 활용)
                    match get_worker_channel().await {
                        Ok(tx) => {
                            tx.send(SlotTask::Bake(BakeTask {
                                slot_id,
                                task_dir: path,
                                kv_name: kv_name.clone(),
                                offset: end,
                                layers: layer_dumps,
                            })).await.map_err(|e| anyhow!("Failed to send to slot worker: {}", e))?;
                            Ok(())
                        },
                        Err(e) => Err(e)
                    }
                }.await;

                if let Err(e) = result {
                    println!("[SLOT-ERROR] Failed to prepare offload: {}. Releasing slot {}.", e, slot_id);
                    SLOT_MANAGER.release_slot(slot_id).await;
                    // Auto-purged data is not a fatal error for the prefill loop, it just means we don't save this chunk.
                    if e.to_string().contains("No KV data") {
                        // Continue
                    } else {
                        return Err(e);
                    }
                }

                // [CLEANUP] Ensure caches are clear for the next chunk
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
        Ok(current_pos)
    }

    pub async fn prefill_chunk(&mut self, text: String, cancel_flag: Option<Arc<AtomicBool>>, mut relay_target: Option<&mut Qwen3VLGenerateModel>) -> Result<usize> {
        let chunk_ids_vec = self.tokenizer.text_encode_vec(text, false)?;
        let chunk_size = chunk_ids_vec.len();
        let current_pos = self.get_kv_len();
        let chunk_ids = Tensor::from_vec(chunk_ids_vec, (1, chunk_size), &self.text_device)?;
        let chunk_pos = Tensor::arange(current_pos as u32, (current_pos + chunk_size) as u32, &self.text_device)?.unsqueeze(0)?;
        self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), current_pos, chunk_size, None)?;
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
            self.qwen3_vl.forward(&chunk_ids, None, None, None, None, Some(&chunk_pos), seqlen_offset, total_tokens, session_id.clone())?;
            local_pos += chunk_size;
            seqlen_offset += chunk_size;
            if seqlen_offset % 1024 == 0 && seqlen_offset > 0 {
                if let Some(sid) = &session_id {
                    let slot_id = SLOT_MANAGER.acquire_write_slot(total_tokens).await;
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
                                    slot_id, task_dir: path, kv_name: kv_name.clone(), offset: seqlen_offset, layers: layer_dumps
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
                let slot_id = SLOT_MANAGER.acquire_write_slot(total_tokens).await;
                
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
            let logits = self.qwen3_vl.forward(&input_ids, pixel_values.as_ref(), image_grid_thw.as_ref(), None, None, Some(&chunk_pos), seqlen_offset, total_tokens, session_id.clone())?;
            let mut logits = logits.squeeze(0)?.i(logits.dim(1)? - 1)?.to_dtype(DType::F32)?;
            if 1.1 != 1.0 { let penalty_context = if all_ids.len() > 512 { &all_ids[all_ids.len()-512..] } else { &all_ids[..] }; logits = apply_repeat_penalty(&logits, 1.1, penalty_context)?; }
            let next_id = logit_processor.sample(&logits)?;
            if next_id == self.eos_token_id1 || next_id == self.eos_token_id2 { break; }
            all_ids.push(next_id);
            generated_text.push_str(&self.tokenizer.token_decode(vec![next_id])?);
            
            // [INFERENCE-SAVE] Periodic Checkpointing
            if i > 0 && i % 50 == 0 && !self.model_name.contains("0.6B") {
                if let Some(sid) = &session_id {
                    let slot_id = SLOT_MANAGER.acquire_write_slot(total_tokens).await;
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
                                    slot_id, task_dir: path.clone(), kv_name: kv_name.clone(), offset: seqlen_offset, layers: layer_dumps
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

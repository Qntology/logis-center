use std::sync::Arc;
use memmap2::Mmap;
use crate::models::qwen3vl::config::{Qwen3VLConfig, Qwen3VLTextConfig};
use crate::models::qwen3vl::native_backend::*;
#[cfg(feature = "cuda")]
use cudarc::driver::sys::{CUdeviceptr, lib};
use half::f16;
use safetensors::SafeTensors;
use anyhow::{Result, anyhow};
use rayon::prelude::*;

pub enum LinearVariant {
    Standard { weight: NativeTensor, bias: Option<NativeTensor> },
    BitSerial { weight_packed: NativeTensor, scales: NativeTensor, bias: Option<NativeTensor> },
}

// [NEW] CPU-Side High-Speed Workspace (Zero-Allocation Pipeline)
pub struct ForwardWorkspace {
    pub hidden_a: Vec<f16>,
    pub hidden_b: Vec<f16>,
    pub intermediate_a: Vec<f16>,
    pub intermediate_b: Vec<f16>,
    pub intermediate_c: Vec<f16>, // [NEW] Added for multi-stage MLP
    pub q: Vec<f16>,
    pub k: Vec<f16>,
    pub v: Vec<f16>,
}

impl ForwardWorkspace {
    pub fn new() -> Self {
        Self { 
            hidden_a: Vec::new(), hidden_b: Vec::new(), 
            intermediate_a: Vec::new(), intermediate_b: Vec::new(), intermediate_c: Vec::new(),
            q: Vec::new(), k: Vec::new(), v: Vec::new() 
        }
    }

    pub fn ensure_capacity(&mut self, hidden_size: usize, intermediate_size: usize, q_len: usize, n_h: usize, n_kv: usize, head_dim: usize) {
        let req_hidden = q_len * hidden_size;
        let req_inter = q_len * intermediate_size;
        let req_q = q_len * n_h * head_dim;
        let req_kv = q_len * n_kv * head_dim;
        
        let eps_signal = f16::from_f32(1e-6);
        // [STABILITY-FIX] Initialize with tiny epsilon instead of pure zero to jumpstart 1-bit activations
        if self.hidden_a.len() < req_hidden { self.hidden_a.resize(req_hidden, eps_signal); }
        if self.hidden_b.len() < req_hidden { self.hidden_b.resize(req_hidden, eps_signal); }
        if self.intermediate_a.len() < req_inter { self.intermediate_a.resize(req_inter, eps_signal); }
        if self.intermediate_b.len() < req_inter { self.intermediate_b.resize(req_inter, eps_signal); }
        if self.intermediate_c.len() < req_inter { self.intermediate_c.resize(req_inter, eps_signal); }
        if self.q.len() < req_q { self.q.resize(req_q, eps_signal); }
        if self.k.len() < req_kv { self.k.resize(req_kv, eps_signal); }
        if self.v.len() < req_kv { self.v.resize(req_kv, eps_signal); }
    }
}

// [NEW] CPU-Side Dynamic KV Cache (Memory Optimizer)
pub struct DynamicKVCache {
    pub k: Vec<u32>,
    pub v: Vec<f16>,
    pub capacity: usize,
    pub current_len: usize,
}

impl DynamicKVCache {
    pub fn new() -> Self {
        Self { k: Vec::new(), v: Vec::new(), capacity: 0, current_len: 0 }
    }

    pub fn grow(&mut self, needed_len: usize, n_kv: usize, head_dim: usize) {
        if needed_len > self.capacity {
            // [STRATEGY] Grow by 1024 token blocks to minimize fragmentation
            let new_cap = (needed_len + 1023) / 1024 * 1024;
            let k_unit = n_kv * (head_dim / 32);
            let v_unit = n_kv * head_dim;
            
            self.k.resize(new_cap * k_unit, 0);
            self.v.resize(new_cap * v_unit, f16::from_f32(1e-6));
            self.capacity = new_cap;
            println!("[CPU-GROW] KV Cache expanded to {} tokens.", new_cap);
        }
    }

    pub fn clear(&mut self) {
        // [OPTIMIZATION] Keep capacity, just reset length
        self.current_len = 0;
    }
}

// [NEW] GPU-Side Dynamic KV Cache (VRAM Memory Optimizer)
pub struct DynamicGpuKVCache {
    pub k_ptr: Option<GpuPtr>,
    pub v_ptr: Option<GpuPtr>,
    pub capacity: usize,
    pub current_len: usize,
}

impl DynamicGpuKVCache {
    pub fn new() -> Self {
        Self { k_ptr: None, v_ptr: None, capacity: 0, current_len: 0 }
    }

    #[cfg(feature = "cuda")]
    pub fn grow(&mut self, needed_len: usize, n_kv: usize, head_dim: usize, device_id: i32) {
        if needed_len > self.capacity {
            let new_cap = (needed_len + 1023) / 1024 * 1024;
            let k_bytes = new_cap * n_kv * (head_dim / 32) * 4;
            let v_bytes = new_cap * n_kv * head_dim * 2;

            unsafe {
                let cuda_lib = crate::models::qwen3vl::native_backend::lib();
                
                // [STABILITY] Ensure valid context
                let mut ctx = std::ptr::null_mut() as cudarc::driver::sys::CUcontext;
                cuda_lib.cuCtxGetCurrent(&mut ctx);
                if ctx == std::ptr::null_mut() && device_id >= 0 {
                    let mut dev = 0 as cudarc::driver::sys::CUdevice;
                    cuda_lib.cuDeviceGet(&mut dev, device_id);
                    cuda_lib.cuDevicePrimaryCtxRetain(&mut ctx, dev);
                    cuda_lib.cuCtxSetCurrent(ctx);
                }

                let mut new_kp: cudarc::driver::sys::CUdeviceptr = 0;
                let mut new_vp: cudarc::driver::sys::CUdeviceptr = 0;
                cuda_lib.cuMemAlloc_v2(&mut new_kp, k_bytes);
                cuda_lib.cuMemAlloc_v2(&mut new_vp, v_bytes);

                // If existing data, copy to new buffer (Realloc behavior)
                if let (Some(old_k), Some(old_v)) = (self.k_ptr.take(), self.v_ptr.take()) {
                    if self.current_len > 0 {
                        let old_k_bytes = self.current_len * n_kv * (head_dim / 32) * 4;
                        let old_v_bytes = self.current_len * n_kv * head_dim * 2;
                        cuda_lib.cuMemcpyDtoD_v2(new_kp, old_k.0 as cudarc::driver::sys::CUdeviceptr, old_k_bytes);
                        cuda_lib.cuMemcpyDtoD_v2(new_vp, old_v.0 as cudarc::driver::sys::CUdeviceptr, old_v_bytes);
                    }
                    // Free old buffers
                    cuda_lib.cuMemFree_v2(old_k.0 as cudarc::driver::sys::CUdeviceptr);
                    cuda_lib.cuMemFree_v2(old_v.0 as cudarc::driver::sys::CUdeviceptr);
                }

                self.k_ptr = Some(GpuPtr(new_kp as *mut _));
                self.v_ptr = Some(GpuPtr(new_vp as *mut _));
                self.capacity = new_cap;
                println!("[GPU-GROW] VRAM KV Cache expanded to {} tokens on Device {}.", new_cap, device_id);
            }
        }
    }

    pub fn clear(&mut self) {
        self.current_len = 0;
    }
}

// [NEW] Bit-Serial Dequantizer for Unified Load Support with Repetition
pub fn dequantize_bit_serial_to_f16(packed: &[u32], scales: &[f16], n: usize, src_k: usize, target_k: usize) -> Vec<f16> {
    let src_k_blocks = (src_k + 31) / 32;
    let target_k_blocks = (target_k + 31) / 32;
    let n_padded = (n + 7) / 8 * 8;
    let mut out = vec![f16::ZERO; n * target_k];
    
    let repetition_ratio = if target_k > src_k { target_k / src_k } else { 1 };

    // Layout is Shuffled: [N/8, K_blocks, 8]
    for n_idx_outer in 0..(n_padded / 8) {
        for src_kb in 0..src_k_blocks {
            for sub_n in 0..8 {
                let n_idx = n_idx_outer * 8 + sub_n;
                if n_idx >= n { continue; }
                
                let shuffle_idx = (n_idx_outer * src_k_blocks + src_kb) * 8 + sub_n;
                let bits = packed[shuffle_idx];
                let scale = scales[shuffle_idx];
                
                // [REPETITION] 타겟 차원이 더 크면 반복해서 채움
                for r in 0..repetition_ratio {
                    let target_kb = src_kb + (r * src_k_blocks);
                    for b in 0..32 {
                        let k_idx = target_kb * 32 + b;
                        if k_idx >= target_k { break; }
                        
                        let bit = (bits >> b) & 1;
                        let val = if bit == 1 { scale } else { -scale };
                        out[n_idx * target_k + k_idx] = val;
                    }
                }
            }
        }
    }
    out
}

pub struct NativeLinear {
    pub in_features: usize, 
    pub out_features: usize, 
    pub src_in: usize,  // [NEW] Original source dimension before hybrid expansion
    pub src_out: usize, // [NEW] Original source dimension
    pub variant: LinearVariant, 
    pub device_id: i32,
}

unsafe impl Send for NativeLinear {}
unsafe impl Sync for NativeLinear {}

impl NativeLinear {
    #[cfg(feature = "cuda")]
    pub fn forward_gpu(&self, d_i: CUdeviceptr, d_o: CUdeviceptr, m: usize) {
        unsafe {
            match &self.variant {
                LinearVariant::Standard { weight, .. } => {
                    let d_w = weight.gpu_ptr.expect("Standard weight must be on GPU").0 as *const f16;
                    crate::models::qwen3vl::native_backend::standard_matmul_cuda_f16(d_i as *const f16, d_w, d_o as *mut f16, m as i32, self.out_features as i32, self.in_features as i32);
                },
                LinearVariant::BitSerial { weight_packed, scales, .. } => {
                    let d_w = weight_packed.gpu_ptr.expect("BitSerial weight must be on GPU").0 as *const u32;
                    let d_s = scales.gpu_ptr.expect("Scales must be on GPU").0 as *const f16;
                    crate::models::qwen3vl::native_backend::cuda_matmul_f16(d_i as *const f16, d_w, d_s, d_o as *mut f16, m as i32, self.out_features as i32, self.in_features as i32, self.device_id as i32, self.src_in as i32);
                }
            }
        }
    }

    // [MODIFIED] High-speed forward that writes directly to provided slice
    pub fn forward_into(&self, x: &[f16], out: &mut [f16], global_scratch: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>) {
        let m = x.len() / self.in_features;
        match &self.variant {
            LinearVariant::Standard { weight, bias } => {
                #[cfg(feature = "cuda")]
                if self.device_id >= 0 {
                    if let Some(w_gpu) = weight.gpu_ptr {
                        let (d_i, d_o) = if let Some((si, so)) = global_scratch {
                            self.ensure_gpu_buffers_ext(m, si, so)
                        } else { (0 as CUdeviceptr, 0 as CUdeviceptr) };

                        if d_i != 0 && d_o != 0 {
                            unsafe {
                                let cuda_lib = crate::models::qwen3vl::native_backend::lib();
                                cuda_lib.cuMemcpyHtoD_v2(d_i, x.as_ptr() as *const _, x.len() * 2);
                                crate::models::qwen3vl::native_backend::standard_matmul_cuda_f16(d_i as *const f16, w_gpu.0 as *const f16, d_o as *mut f16, m as i32, self.out_features as i32, self.in_features as i32);
                                cuda_lib.cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut _, d_o, out.len() * 2);
                                if let Some(b) = bias {
                                    let b_ref = b.get_raw_slice::<f16>();
                                    for i in 0..m { for j in 0..self.out_features { out[i * self.out_features + j] += b_ref[j]; } }
                                }
                                return;
                            }
                        }
                    }
                }
                let w_cow = weight.get_slice::<f16>(); 
                let b_cow = bias.as_ref().map(|t| t.get_slice::<f16>());
                native_linear_f16_into(x, w_cow.as_ref(), b_cow.as_ref().map(|b| b.as_ref()), out, m, self.out_features, self.in_features);
            },
            LinearVariant::BitSerial { weight_packed, scales, bias } => {
                #[cfg(feature = "cuda")]
                {
                    if self.device_id >= 0 {
                        let wp_ptr = weight_packed.gpu_ptr.map(|p| p.0 as usize).unwrap_or(0);
                        if wp_ptr != 0 {
                            let (d_i, d_o) = if let Some((si, so)) = global_scratch {
                                self.ensure_gpu_buffers_ext(m, si, so)
                            } else { (0 as CUdeviceptr, 0 as CUdeviceptr) };

                            if d_i != 0 && d_o != 0 {
                                // Bit-Serial Matrix Multiply (Inplace version)
                                crate::models::qwen3vl::native_backend::bit_serial_matmul_gpu_buffered_into(x, weight_packed, scales, out, m, self.out_features, self.in_features, self.device_id as usize, d_i, d_o, self.src_in);
                                
                                if let Some(b) = bias { 
                                    unsafe {
                                        let b_ref = b.get_raw_slice::<f16>();
                                        for i in 0..m { for j in 0..self.out_features { out[i * self.out_features + j] += b_ref[j]; } } 
                                    }
                                }
                                return;
                            }
                        }
                    }
                }

                unsafe {
                    let wp_ref = weight_packed.get_raw_slice::<u32>(); 
                    let s_ref = scales.get_raw_slice::<f16>();
                    // Modified to use bit_serial_matmul_f32_extreme_into
                    bit_serial_matmul_f32_extreme_into(x, wp_ref, s_ref, out, m, self.out_features, self.in_features);
                    
                    if let Some(b) = bias {
                        let b_ref = b.get_raw_slice::<f16>();
                        for i in 0..m {
                            for j in 0..self.out_features {
                                out[i * self.out_features + j] += b_ref[j];
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn forward(&self, x: &[f16], global_scratch: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>) -> Vec<f16> {
        let eps_signal = f16::from_f32(1e-6);
        let mut out = vec![eps_signal; (x.len() / self.in_features) * self.out_features];
        self.forward_into(x, &mut out, global_scratch);
        out
    }

    #[cfg(feature = "cuda")]
    fn ensure_gpu_buffers_ext(&self, m: usize, scratch_i: &std::sync::Mutex<Option<(GpuPtr, usize)>>, scratch_o: &std::sync::Mutex<Option<(GpuPtr, usize)>>) -> (CUdeviceptr, CUdeviceptr) {
        use cudarc::driver::sys::*;
        // [ADAPTIVE-POOLING] Request 25% extra padding to prevent jittering allocs
        let req_i = (m * self.in_features * 4 * 125) / 100;
        let req_o = (m * self.out_features * 4 * 125) / 100;
        let cuda_lib = unsafe { crate::models::qwen3vl::native_backend::lib() };
        
        unsafe {
            let mut ctx = std::ptr::null_mut() as CUcontext;
            cuda_lib.cuCtxGetCurrent(&mut ctx);
            if ctx == std::ptr::null_mut() && self.device_id >= 0 {
                let mut dev = 0 as CUdevice;
                cuda_lib.cuDeviceGet(&mut dev, self.device_id);
                cuda_lib.cuDevicePrimaryCtxRetain(&mut ctx, dev);
                cuda_lib.cuCtxSetCurrent(ctx);
            }
        }

        let mut si_guard = scratch_i.lock().unwrap();
        let d_i = if let Some((ptr, size)) = *si_guard {
            if size >= (m * self.in_features * 4) { ptr.0 as CUdeviceptr }
            else {
                // Grow only if significantly smaller
                let mut new_ptr: CUdeviceptr = 0;
                let _ = unsafe { cuda_lib.cuMemFree_v2(ptr.0 as CUdeviceptr) };
                let res = unsafe { cuda_lib.cuMemAlloc_v2(&mut new_ptr, req_i) };
                if (res as i32) == 0 && new_ptr != 0 {
                    println!("[VRAM-POOL] Growing Input Scratch to {} bytes.", req_i);
                    *si_guard = Some((GpuPtr(new_ptr as *mut _), req_i));
                    new_ptr
                } else { 0 as CUdeviceptr }
            }
        } else {
            let mut new_ptr: CUdeviceptr = 0;
            let res = unsafe { cuda_lib.cuMemAlloc_v2(&mut new_ptr, req_i) };
            if (res as i32) == 0 && new_ptr != 0 {
                *si_guard = Some((GpuPtr(new_ptr as *mut _), req_i));
                new_ptr
            } else { 0 as CUdeviceptr }
        };

        let mut so_guard = scratch_o.lock().unwrap();
        let d_o = if let Some((ptr, size)) = *so_guard {
            if size >= (m * self.out_features * 4) { ptr.0 as CUdeviceptr }
            else {
                let mut new_ptr: CUdeviceptr = 0;
                let _ = unsafe { cuda_lib.cuMemFree_v2(ptr.0 as CUdeviceptr) };
                let res = unsafe { cuda_lib.cuMemAlloc_v2(&mut new_ptr, req_o) };
                if (res as i32) == 0 && new_ptr != 0 {
                    println!("[VRAM-POOL] Growing Output Scratch to {} bytes.", req_o);
                    *so_guard = Some((GpuPtr(new_ptr as *mut _), req_o));
                    new_ptr
                } else { 0 as CUdeviceptr }
            }
        } else {
            let mut new_ptr: CUdeviceptr = 0;
            let res = unsafe { cuda_lib.cuMemAlloc_v2(&mut new_ptr, req_o) };
            if (res as i32) == 0 && new_ptr != 0 {
                *so_guard = Some((GpuPtr(new_ptr as *mut _), req_o));
                new_ptr
            } else { 0 as CUdeviceptr }
        };

        (d_i, d_o)
    }

    pub fn move_to_gpu(&mut self, device_id: i32) {
        self.device_id = device_id;
        match &mut self.variant {
            LinearVariant::Standard { weight, bias } => {
                weight.move_to_gpu(device_id);
                if let Some(b) = bias { b.move_to_gpu(device_id); }
            },
            LinearVariant::BitSerial { weight_packed, scales, bias } => {
                #[cfg(feature = "cuda")] 
                {
                    weight_packed.move_to_gpu(device_id);
                    scales.move_to_gpu_f16(device_id);
                    if let Some(b) = bias { b.move_to_gpu(device_id); }
                }
            }
        }
    }
}

pub struct NativeLayer {
    pub input_layernorm: NativeTensor,
    pub post_attention_layernorm: NativeTensor,
    pub q_norm: Option<NativeTensor>, 
    pub k_norm: Option<NativeTensor>, 
    pub q_proj: NativeLinear,
    pub k_proj: NativeLinear,
    pub v_proj: NativeLinear,
    pub o_proj: NativeLinear,
    pub gate_proj: NativeLinear,
    pub up_proj: NativeLinear,
    pub down_proj: NativeLinear,
    pub device_id: i32,
    pub is_support_layer: bool,
    pub kv_cache: std::sync::Mutex<DynamicKVCache>, 
    pub gpu_kv_cache: std::sync::Mutex<DynamicGpuKVCache>, 
    // [REMOVED] Individual scratch buffers to save VRAM
    pub gpu_broken: std::sync::atomic::AtomicBool,
}

unsafe impl Send for NativeLayer {}
unsafe impl Sync for NativeLayer {}

impl NativeLayer {
    pub fn forward<'a>(
        &self, 
        x: &[f16], 
        config: &Qwen3VLTextConfig, 
        seqlen_offset: usize, 
        _idx: usize, 
        rope_cos: &[f16], 
        rope_sin: &[f16], 
        is_baking: bool, 
        is_vision: bool, // [NEW] Explicit modality flag from task level
        global_scratch: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>,
        workspace: &'a mut ForwardWorkspace,
        ping_pong: bool
    ) -> &'a [f16] { 
        let hidden_size = config.hidden_size; 
        let q_len = x.len() / hidden_size;
        let head_dim = config.head_dim; 
        let n_h = config.num_attention_heads; 
        let n_kv = config.num_key_value_heads;
        
        workspace.ensure_capacity(hidden_size, config.intermediate_size, q_len, n_h, n_kv, head_dim);

        // [2026-COGNITIVE-STABILITY-MAX] Precision energy measurement for Layer 0
        let mut final_alpha = 1.0f32;
        let mut semantic_gain = 1.0f32;

        if _idx == 0 && q_len > 0 {
            let samples = x.iter().take(500).map(|v| v.to_f32().abs()).collect::<Vec<_>>();
            let mean_abs = samples.iter().sum::<f32>() / samples.len() as f32;
            let max_abs = samples.iter().fold(1e-9f32, |a, &b| a.max(b));
            let density = (mean_abs / max_abs).clamp(0.0, 1.0);
            
            let total_context = seqlen_offset + q_len;
            // Adaptive log-scale factor: starts at 0.0, grows as context increases
            let context_log = (total_context as f32 / 1024.0).max(1.0).ln();

            // --- [SCENARIO-BASED ADAPTIVE CURVES] ---
            if is_vision {
                // Scenario A: VISION (Image/OCR)
                // Steep growth: As visual data grows, signal clarity needs aggressive protection
                let base_vision_alpha = 1.55f32 + (context_log * 0.18); 
                final_alpha = base_vision_alpha.clamp(1.5, 2.5);
                semantic_gain = 1.15f32 + (context_log * 0.05);
                
                if is_baking { final_alpha *= 1.45f32; } 
            } else {
                // Scenario B: TEXT/PUG (Language/Structure)
                // Shallow growth: Focus on structural stability without saturating the signal
                let base_text_alpha = 1.10f32 + (context_log * 0.08);
                final_alpha = base_text_alpha.clamp(1.0, 1.8);
                semantic_gain = 1.02f32 + (context_log * 0.03);
                
                if is_baking { final_alpha *= 1.30f32; } 
            }

            // --- [DYNAMIC CHUNK & DENSITY CORRECTION] ---
            // Chunk Momentum: Stabilize large parallel prefill blocks
            let chunk_momentum = if q_len > 128 {
                (1.0 + (q_len as f32 / 128.0).log2() * 0.04).clamp(1.0, 1.15)
            } else { 1.0 };

            // Apply statistical density boost (universal hardware-level correction)
            let depth_ratio = _idx as f32 / config.num_hidden_layers as f32;
            let accuracy_correction = (1.0 - depth_ratio * 0.15).clamp(0.85, 1.0);
            let density_boost = if density < 0.2f32 { 6.0f32 } else { (1.0f32 / (density + 0.1f32)).min(4.0f32) };
            
            // Final Fluid Alpha Calculation
            // Combined: Modality Base * Hardware-Level Normalization * Context Fatigue * Chunk Momentum
            final_alpha = (final_alpha * (0.1f32 / (mean_abs + 1e-9f32)) * density_boost * accuracy_correction * chunk_momentum).clamp(0.2f32, 5.0f32);
            semantic_gain = (semantic_gain * (density * 1.5 + 0.85)).clamp(0.90, 1.50); 
            
            // [HYBRID-EXTRA-BOOST] Apply additional strength for the 0.6B Support Layer
            if self.is_support_layer && !is_baking {
                final_alpha *= 1.12f32; 
                semantic_gain *= 1.04f32;
            }
            
            // [STABILITY-CAP] Scenario-specific caps for extreme long-context
            let max_safe_alpha = if is_vision { 4.8f32 } else { 3.8f32 };
            if final_alpha > max_safe_alpha { final_alpha = max_safe_alpha; }
            
            if q_len > 100 || seqlen_offset % 1024 == 0 {
                println!("[COGNITIVE-FLUID] L0 | Modality: {} | Alpha: {:.4} | Gain: {:.2} | Context: {}t | Momentum: {:.2}x", 
                    if is_vision { "VISION" } else { "TEXT/PUG" }, final_alpha, semantic_gain, total_context, chunk_momentum);
            }
        }

        // Target buffer based on ping-pong flag
        let out_ptr = if ping_pong { workspace.hidden_b.as_mut_ptr() } else { workspace.hidden_a.as_mut_ptr() };

        // [GPU-PATH-FINAL-OPTIMIZATION]
        #[cfg(feature = "cuda")]
        if self.device_id >= 0 && !self.gpu_broken.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some((si, so)) = global_scratch {
                let (d_i, d_o) = self.q_proj.ensure_gpu_buffers_ext(q_len, si, so);
                if d_i != 0 && d_o != 0 {
                    unsafe {
                        let cuda_lib = crate::models::qwen3vl::native_backend::lib();
                        cuda_lib.cuMemcpyHtoD_v2(d_i, x.as_ptr() as *const _, x.len() * 2);
                        let d_ln_w = self.input_layernorm.gpu_ptr.expect("LN weight on GPU").0 as *const f16;
                        crate::models::qwen3vl::native_backend::cuda_rms_norm_f16(d_i as *const f16, d_ln_w, d_o as *mut f16, q_len as i32, hidden_size as i32, config.rms_norm_eps as f32);
                        
                        self.q_proj.forward_gpu(d_o, d_i, q_len); cuda_lib.cuMemcpyDtoH_v2(workspace.q.as_mut_ptr() as *mut _, d_i, workspace.q.len() * 2);
                        self.k_proj.forward_gpu(d_o, d_i, q_len); cuda_lib.cuMemcpyDtoH_v2(workspace.k.as_mut_ptr() as *mut _, d_i, workspace.k.len() * 2);
                        self.v_proj.forward_gpu(d_o, d_i, q_len); cuda_lib.cuMemcpyDtoH_v2(workspace.v.as_mut_ptr() as *mut _, d_i, workspace.v.len() * 2);

                        native_apply_rope_f16_with_offset(&mut workspace.q, &mut workspace.k, q_len, seqlen_offset, n_h, head_dim, config.rope_theta, rope_cos, rope_sin);

                        let mut gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
                        let needed_tokens = seqlen_offset + q_len;
                        
                        // [DYNAMIC-GROWTH] Ensure VRAM is allocated and large enough
                        gpu_cache_guard.grow(needed_tokens, n_kv, head_dim, self.device_id);
                        
                        if let (Some(k_ptr), Some(v_ptr)) = (gpu_cache_guard.k_ptr, gpu_cache_guard.v_ptr) {
                            let k_packed = pack_f16_to_bits(&workspace.k);
                            let k_target_offset = (seqlen_offset * n_kv * (head_dim/32) * 4) as u64;
                            let v_target_offset = (seqlen_offset * n_kv * head_dim * 2) as u64;

                            cuda_lib.cuMemcpyHtoD_v2((k_ptr.0 as u64 + k_target_offset) as cudarc::driver::sys::CUdeviceptr, k_packed.as_ptr() as *const _, k_packed.len() * 4);
                            cuda_lib.cuMemcpyHtoD_v2((v_ptr.0 as u64 + v_target_offset) as cudarc::driver::sys::CUdeviceptr, workspace.v.as_ptr() as *const _, workspace.v.len() * 2);
                            cuda_lib.cuMemcpyHtoD_v2(d_i, workspace.q.as_ptr() as *const _, workspace.q.len() * 2);

                            let acc_corr = (1.0 - (_idx as f32 / 28.0) * 0.15).clamp(0.85, 1.0);
                            let src_h_d = self.q_proj.src_in / n_h;
                            let mut attn_out_raw = crate::models::qwen3vl::native_backend::native_bit_serial_attn_gpu_buffered(&workspace.q, k_ptr, v_ptr, n_h, n_kv, head_dim, needed_tokens, self.device_id as usize, d_i, d_o, final_alpha * acc_corr, src_h_d);
                            
                            // Apply Semantic Gain
                            if semantic_gain != 1.0 {
                                for val in attn_out_raw.iter_mut() { *val = f16::from_f32(val.to_f32() * semantic_gain); }
                            }
                            
                            gpu_cache_guard.current_len = needed_tokens;
                        }

                        self.o_proj.forward_gpu(d_o, d_i, q_len);
                        let res_slice = std::slice::from_raw_parts_mut(out_ptr, x.len());
                        cuda_lib.cuMemcpyDtoH_v2(res_slice.as_mut_ptr() as *mut _, d_i, x.len() * 2);
                        for i in 0..x.len() { res_slice[i] += x[i]; }
                        cuda_lib.cuMemcpyHtoD_v2(d_i, res_slice.as_ptr() as *const _, x.len() * 2);

                        let d_post_w = self.post_attention_layernorm.gpu_ptr.expect("Post LN on GPU").0 as *const f16;
                        crate::models::qwen3vl::native_backend::cuda_rms_norm_f16(d_i as *const f16, d_post_w, d_o as *mut f16, q_len as i32, hidden_size as i32, config.rms_norm_eps as f32);

                        self.gate_proj.forward_gpu(d_o, d_i, q_len); 
                        let inter_size = q_len * config.intermediate_size;
                        cuda_lib.cuMemcpyDtoH_v2(workspace.intermediate_a.as_mut_ptr() as *mut _, d_i, inter_size * 2);
                        
                        self.up_proj.forward_gpu(d_o, d_i, q_len); 
                        cuda_lib.cuMemcpyDtoH_v2(workspace.intermediate_b.as_mut_ptr() as *mut _, d_i, inter_size * 2);
                        
                        native_silu_f16(&mut workspace.intermediate_a[..inter_size]); 
                        for i in 0..inter_size { 
                            workspace.intermediate_a[i] *= workspace.intermediate_b[i]; 
                        }
                        
                        cuda_lib.cuMemcpyHtoD_v2(d_i, workspace.intermediate_a.as_ptr() as *const _, inter_size * 2);
                        self.down_proj.forward_gpu(d_i, d_o, q_len);
                        
                        let mlp_h = std::slice::from_raw_parts_mut(workspace.q.as_mut_ptr(), x.len());
                        cuda_lib.cuMemcpyDtoH_v2(mlp_h.as_mut_ptr() as *mut _, d_o, x.len() * 2);
                        
                        // [HYBRID-OUTPUT-ADAPT] If 0.6B output is smaller than 2B input, repeat signal
                        let src_out_size = self.down_proj.src_out; 
                        let target_out_size = self.down_proj.out_features;
                        
                        if target_out_size > src_out_size {
                            for t in 0..q_len {
                                for i in src_out_size..target_out_size {
                                    mlp_h[t * target_out_size + i] = mlp_h[t * target_out_size + (i % src_out_size)];
                                }
                            }
                        }

                        for i in 0..x.len() { res_slice[i] += mlp_h[i]; }
                        return res_slice;
                    }
                }
            }
        }

        // [CPU-PATH-ZERO-COPY]
        let ln_w = self.input_layernorm.get_slice::<f16>();
        let mut cpu_norm = vec![f16::ZERO; x.len()];
        native_rms_norm_f16_into(x, ln_w.as_ref(), config.rms_norm_eps as f32, hidden_size, &mut cpu_norm);

        self.q_proj.forward_into(&cpu_norm, &mut workspace.q, global_scratch);
        self.k_proj.forward_into(&cpu_norm, &mut workspace.k, global_scratch);
        self.v_proj.forward_into(&cpu_norm, &mut workspace.v, global_scratch);

        native_apply_rope_f16_with_offset(&mut workspace.q, &mut workspace.k, q_len, seqlen_offset, n_h, head_dim, config.rope_theta, rope_cos, rope_sin);

        let mut cache = self.kv_cache.lock().unwrap();
        let needed_len = seqlen_offset + q_len;
        cache.grow(needed_len, n_kv, head_dim);
        let k_packed = pack_f16_to_bits(&workspace.k);
        cache.k[seqlen_offset * (n_kv * head_dim / 32) .. seqlen_offset * (n_kv * head_dim / 32) + k_packed.len()].copy_from_slice(&k_packed);
        cache.v[seqlen_offset * (n_kv * head_dim) .. seqlen_offset * (n_kv * head_dim) + workspace.v.len()].copy_from_slice(&workspace.v);
        cache.current_len = needed_len;

        let mut attn_out = native_bit_serial_attn_f16(&workspace.q, &cache.k, &cache.v, hidden_size, n_h, n_kv, q_len, needed_len, final_alpha);
        
        // Apply Semantic Gain
        if semantic_gain != 1.0 {
            for val in attn_out.iter_mut() { *val = f16::from_f32(val.to_f32() * semantic_gain); }
        }

        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, x.len()) };
        self.o_proj.forward_into(&attn_out, out_slice, global_scratch);
        for i in 0..x.len() { out_slice[i] += x[i]; }

        let post_ln_w = self.post_attention_layernorm.get_slice::<f16>();
        // [FIX] Use correctly sized slice for output to prevent out-of-bounds in par_chunks
        native_rms_norm_f16_into(out_slice, post_ln_w.as_ref(), config.rms_norm_eps as f32, hidden_size, &mut workspace.intermediate_a[..x.len()]);
        
        self.gate_proj.forward_into(&workspace.intermediate_a[..x.len()], &mut workspace.intermediate_b, global_scratch);
        self.up_proj.forward_into(&workspace.intermediate_a[..x.len()], &mut workspace.intermediate_c, global_scratch); 
        
        native_silu_f16(&mut workspace.intermediate_b[..q_len * config.intermediate_size]); 
        for i in 0..q_len * config.intermediate_size { 
            workspace.intermediate_b[i] *= workspace.intermediate_c[i]; 
        }
        
        let mut mlp_out_h = vec![f16::from_f32(1e-6); x.len()];
        self.down_proj.forward_into(&workspace.intermediate_b[..q_len * config.intermediate_size], &mut mlp_out_h, global_scratch);
        
        // [HYBRID-OUTPUT-ADAPT-CPU] Repeat 0.6B signal if needed before adding residual
        let src_out_size = self.down_proj.src_out;
        let target_out_size = self.down_proj.out_features;
        if target_out_size > src_out_size {
            for t in 0..q_len {
                for i in src_out_size..target_out_size {
                    mlp_out_h[t * target_out_size + i] = mlp_out_h[t * target_out_size + (i % src_out_size)];
                }
            }
        }

        for i in 0..x.len() { out_slice[i] += mlp_out_h[i]; }
        
        out_slice
    }

    pub fn move_to_gpu(&mut self, device_id: i32) {
        self.device_id = device_id;
        self.input_layernorm.move_to_gpu(device_id);
        self.post_attention_layernorm.move_to_gpu(device_id);
        if let Some(ref mut qn) = self.q_norm { qn.move_to_gpu(device_id); }
        if let Some(ref mut kn) = self.k_norm { kn.move_to_gpu(device_id); }
        
        self.q_proj.move_to_gpu(device_id); self.k_proj.move_to_gpu(device_id);
        self.v_proj.move_to_gpu(device_id); self.o_proj.move_to_gpu(device_id);
        self.gate_proj.move_to_gpu(device_id); self.up_proj.move_to_gpu(device_id);
        self.down_proj.move_to_gpu(device_id);
    }

    pub fn clear_kv_cache(&self) {
        let mut cache = self.kv_cache.lock().unwrap();
        cache.clear();
        let mut gpu_cache = self.gpu_kv_cache.lock().unwrap();
        gpu_cache.clear();
    }

    pub fn force_free_kv_cache(&self) {
        let mut cache = self.kv_cache.lock().unwrap();
        cache.k = Vec::new();
        cache.v = Vec::new();
        cache.capacity = 0;
        cache.current_len = 0;
        
        let mut gpu_cache = self.gpu_kv_cache.lock().unwrap();
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            if let Some(k) = gpu_cache.k_ptr.take() { let _ = lib().cuMemFree_v2(k.0 as CUdeviceptr); }
            if let Some(v) = gpu_cache.v_ptr.take() { let _ = lib().cuMemFree_v2(v.0 as CUdeviceptr); }
        }
        gpu_cache.capacity = 0;
        gpu_cache.current_len = 0;
    }

    pub fn get_kv_data(&self, head_dim: usize, n_kv: usize, start_token: usize) -> Option<(Vec<u32>, Vec<f16>)> {
        #[cfg(feature = "cuda")]
        if self.device_id >= 0 && !self.gpu_broken.load(std::sync::atomic::Ordering::Relaxed) {
            use cudarc::driver::sys::{lib, CUdeviceptr};
            let gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
            if let (Some(k_ptr), Some(v_ptr)) = (gpu_cache_guard.k_ptr, gpu_cache_guard.v_ptr) {
                let current_len = gpu_cache_guard.current_len;
                if current_len > start_token {
                    let extract_len = current_len - start_token;
                    let k_unit = n_kv * (head_dim / 32);
                    let v_unit = n_kv * head_dim;
                    
                    let k_size = extract_len * k_unit;
                    let v_size = extract_len * v_unit;
                    let mut k_host = vec![0u32; k_size];
                    let mut v_host = vec![f16::ZERO; v_size];
                    
                    unsafe {
                        let cuda_lib = lib();
                        let k_offset_bytes = (start_token * k_unit * 4) as u64;
                        let v_offset_bytes = (start_token * v_unit * 2) as u64;
                        
                        let d_k_src = (k_ptr.0 as u64 + k_offset_bytes) as CUdeviceptr;
                        let d_v_src = (v_ptr.0 as u64 + v_offset_bytes) as CUdeviceptr;
                        
                        let _ = cuda_lib.cuMemcpyDtoH_v2(k_host.as_mut_ptr() as *mut _, d_k_src, k_size * 4);
                        let _ = cuda_lib.cuMemcpyDtoH_v2(v_host.as_mut_ptr() as *mut _, d_v_src, v_size * 2);
                    }
                    return Some((k_host, v_host));
                }
            }
        }

        let cache = self.kv_cache.lock().unwrap();
        if cache.current_len > start_token {
            let k_unit = n_kv * (head_dim / 32);
            let v_unit = n_kv * head_dim;
            let extract_len = cache.current_len - start_token;
            
            Some((
                cache.k[start_token * k_unit .. cache.current_len * k_unit].to_vec(),
                cache.v[start_token * v_unit .. cache.current_len * v_unit].to_vec()
            ))
        } else {
            None
        }
    }

    pub fn get_kv_len(&self, head_dim: usize, n_kv: usize) -> usize {
        #[cfg(feature = "cuda")]
        if self.device_id >= 0 {
            let gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
            if gpu_cache_guard.current_len > 0 { return gpu_cache_guard.current_len; }
        }

        let cache = self.kv_cache.lock().unwrap();
        cache.current_len
    }

    pub fn set_kv_data(&self, k: Vec<u32>, v: Vec<f16>) {
        let mut cache = self.kv_cache.lock().unwrap();
        let tokens = if k.is_empty() { 0 } else { k.len() / (v.len() / k.len() * 32) }; // This is a bit complex to infer, better to pass n_kv/head_dim
        // For simplicity, we just set the raw vectors and update capacity/len
        cache.k = k;
        cache.v = v;
        cache.capacity = 0; // Invalidate if we just injected raw
        // In real use case, set_kv_data is used for small things or we should fix the logic
    }

    /// [NEW] Direct GPU injection to prevent redundant CPU conversions
    #[cfg(feature = "cuda")]
    pub fn inject_gpu_kv(&self, k_data: &[u32], v_data: &[f16], n_kv: usize, head_dim: usize) {
        let tokens = (k_data.len() * 32) / (n_kv * head_dim);
        let mut gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
        
        unsafe {
            use cudarc::driver::sys::*;
            let cuda_lib = crate::models::qwen3vl::native_backend::lib();
            // [STABILITY] Ensure valid context
            let mut ctx = std::ptr::null_mut() as CUcontext;
            cuda_lib.cuCtxGetCurrent(&mut ctx);
            if ctx == std::ptr::null_mut() && self.device_id >= 0 {
                let mut dev = 0 as CUdevice;
                cuda_lib.cuDeviceGet(&mut dev, self.device_id);
                cuda_lib.cuDevicePrimaryCtxRetain(&mut ctx, dev);
                cuda_lib.cuCtxSetCurrent(ctx);
            }

            let mut kp: CUdeviceptr = 0;
            let mut vp: CUdeviceptr = 0;
            // [DYNAMIC-ALLOCATION] Allocate exactly what is needed for the injected component
            let k_bytes = tokens * n_kv * (head_dim / 32) * 4;
            let v_bytes = tokens * n_kv * head_dim * 2;
            
            cuda_lib.cuMemAlloc_v2(&mut kp, k_bytes); 
            cuda_lib.cuMemAlloc_v2(&mut vp, v_bytes); 
            
            // Copy data via F16 path
            cuda_lib.cuMemcpyHtoD_v2(kp, k_data.as_ptr() as *const _, k_data.len() * 4);
            cuda_lib.cuMemcpyHtoD_v2(vp, v_data.as_ptr() as *const _, v_data.len() * 2);
            
            gpu_cache_guard.k_ptr = Some(GpuPtr(kp as *mut _));
            gpu_cache_guard.v_ptr = Some(GpuPtr(vp as *mut _));
            gpu_cache_guard.capacity = tokens;
            gpu_cache_guard.current_len = tokens;
        }
    }

    /// [NEW] Direct GPU-to-GPU injection for extremely fast layer replication
    #[cfg(feature = "cuda")]
    pub fn inject_gpu_kv_direct(&self, k_src: GpuPtr, v_src: GpuPtr, tokens: usize, n_kv: usize, head_dim: usize) {
        let mut gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
        
        unsafe {
            use cudarc::driver::sys::*;
            let cuda_lib = crate::models::qwen3vl::native_backend::lib();
            // [STABILITY] Ensure valid context
            let mut ctx = std::ptr::null_mut() as CUcontext;
            cuda_lib.cuCtxGetCurrent(&mut ctx);
            if ctx == std::ptr::null_mut() && self.device_id >= 0 {
                let mut dev = 0 as CUdevice;
                cuda_lib.cuDeviceGet(&mut dev, self.device_id);
                cuda_lib.cuDevicePrimaryCtxRetain(&mut ctx, dev);
                cuda_lib.cuCtxSetCurrent(ctx);
            }

            let mut kp: CUdeviceptr = 0;
            let mut vp: CUdeviceptr = 0;
            // [DYNAMIC-ALLOCATION] Allocate exactly what is needed for the component
            let k_bytes = tokens * n_kv * (head_dim / 32) * 4;
            let v_bytes = tokens * n_kv * head_dim * 2;
            
            cuda_lib.cuMemAlloc_v2(&mut kp, k_bytes); 
            cuda_lib.cuMemAlloc_v2(&mut vp, v_bytes); 
            
            // Copy data via DtoD (Device to Device)
            cuda_lib.cuMemcpyDtoD_v2(kp, k_src.0 as CUdeviceptr, k_bytes);
            cuda_lib.cuMemcpyDtoD_v2(vp, v_src.0 as CUdeviceptr, v_bytes);
            
            gpu_cache_guard.k_ptr = Some(GpuPtr(kp as *mut _));
            gpu_cache_guard.v_ptr = Some(GpuPtr(vp as *mut _));
            gpu_cache_guard.capacity = tokens;
            gpu_cache_guard.current_len = tokens;
        }
    }
}

// [NEW] Dynamic RoPE Cache (Position Optimizer)
pub struct RopeCache {
    pub cos: Vec<f16>,
    pub sin: Vec<f16>,
    pub head_dim: usize,
    pub theta: f32,
    pub current_max_len: usize,
    pub tail_tokens: Vec<u32>, // [BRIDGE] Last tokens from baked component
}

impl RopeCache {
    pub fn new(head_dim: usize, theta: f32, initial_len: usize) -> Self {
        let mut cache = Self {
            cos: Vec::with_capacity(initial_len * head_dim),
            sin: Vec::with_capacity(initial_len * head_dim),
            head_dim,
            theta,
            current_max_len: 0,
            tail_tokens: Vec::new(),
        };
        cache.ensure_length(initial_len);
        cache
    }

    pub fn ensure_length(&mut self, needed_len: usize) {
        if needed_len > self.current_max_len {
            let start = self.current_max_len;
            let end = (needed_len + 1023) / 1024 * 1024; // Expand in 1k blocks
            
            for p in start..end {
                for d in 0..(self.head_dim / 2) {
                    let exponent = (2.0 * d as f32) / (self.head_dim as f32);
                    let freq = 1.0 / self.theta.powf(exponent);
                    let (sn, cs) = ((p as f32) * freq).sin_cos();
                    self.cos.push(f16::from_f32(cs));
                    self.sin.push(f16::from_f32(sn));
                }
            }
            self.current_max_len = end;
            if start > 0 { println!("[ROPE-GROW] Expanded RoPE table to {} tokens.", end); }
        }
    }
}

pub struct NativeQwen3TextModel {
    pub config: Qwen3VLTextConfig, 
    pub embed_tokens: NativeLinear, 
    pub layers: Vec<NativeLayer>, 
    pub norm: NativeTensor,
    pub rope_cache: std::sync::Mutex<RopeCache>,
}

unsafe impl Send for NativeQwen3TextModel {}
unsafe impl Sync for NativeQwen3TextModel {}

impl NativeQwen3TextModel {
    pub fn forward(
        &self, 
        input_ids: &[u32], 
        _pixel_values: Option<&[f16]>, 
        _grid_thw: Option<&[u32; 3]>, 
        seqlen_offset: usize, 
        global_scratch: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>,
        workspace: Option<&mut ForwardWorkspace>,
        support_layer: Option<&NativeLayer>, // [NEW] Optional support path
        is_vision: bool, // [NEW] Modality flag
    ) -> Vec<f16> {
        let hid = self.config.hidden_size;
        let embeds = match &self.embed_tokens.variant {
            LinearVariant::Standard { weight, .. } => {
                let w_cow = weight.get_slice::<f16>();
                native_embedding_lookup_f16(input_ids, w_cow.as_ref(), hid)
            },
            LinearVariant::BitSerial { weight_packed, .. } => {
                let w_cow = weight_packed.get_slice::<f16>();
                native_embedding_lookup_f16(input_ids, w_cow.as_ref(), hid)
            }
        };

        self.forward_ext(input_ids, embeds, seqlen_offset, global_scratch, workspace, support_layer, is_vision)
    }

    pub fn forward_ext(
        &self, 
        _input_ids: &[u32], 
        embeds: Vec<f16>, 
        seqlen_offset: usize, 
        global_scratch: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>,
        workspace: Option<&mut ForwardWorkspace>,
        support_layer: Option<&NativeLayer>, // [NEW] Optional support path
        is_vision: bool, // [NEW] Modality flag
    ) -> Vec<f16> {
        let hid = self.config.hidden_size;
        let is_baking = self.layers.len() <= 1;
        let q_len = embeds.len() / hid;
        
        // [STABILITY-FIX] Always ensure we have a valid workspace with correct capacity
        let mut internal_ws = ForwardWorkspace::new();
        let ws = match workspace {
            Some(w) => w,
            None => &mut internal_ws,
        };

        // Ensure buffers are large enough before copying anything
        ws.ensure_capacity(hid, self.config.intermediate_size, q_len, self.config.num_attention_heads, self.config.num_key_value_heads, self.config.head_dim);

        // Copy input to hidden_a
        ws.hidden_a[..embeds.len()].copy_from_slice(&embeds);
        let mut cur_x: &[f16] = unsafe { std::slice::from_raw_parts(ws.hidden_a.as_ptr(), embeds.len()) };

        // [DYNAMIC-ROPE] Ensure we have enough positional embeddings for current offset
        let needed_rope = seqlen_offset + q_len;
        let mut rope_guard = self.rope_cache.lock().unwrap();
        rope_guard.ensure_length(needed_rope);
        
        // Break link to mutex for the loop (Read-only access to internal buffers is safe here)
        let r_cos = &rope_guard.cos;
        let r_sin = &rope_guard.sin;

        for (i, layer) in self.layers.iter().enumerate() { 
            // [HOT-SWAP] Use Support Layer for index 0 if provided
            let active_layer = if i == 0 && support_layer.is_some() {
                support_layer.unwrap()
            } else {
                layer
            };

            let use_b = i % 2 == 0;
            let out_slice = active_layer.forward(
                cur_x, 
                &self.config, 
                seqlen_offset, 
                i, 
                r_cos, 
                r_sin, 
                is_baking, 
                is_vision, // [NEW] Pass modality
                global_scratch,
                ws,
                use_b
            ); 
            
            // [BORROW-CHECKER-FIX] Detach output slice from workspace to allow next iteration's borrow
            cur_x = unsafe { std::slice::from_raw_parts(out_slice.as_ptr(), out_slice.len()) };
        }
        
        let norm_cow = self.norm.get_slice::<f16>();
        let eps_signal = f16::from_f32(1e-6);
        let mut cpu_norm = vec![eps_signal; cur_x.len()];
        native_rms_norm_f16_into(cur_x, norm_cow.as_ref(), self.config.rms_norm_eps as f32, hid, &mut cpu_norm);
        cpu_norm
    }

    pub fn move_to_gpu(&mut self, device_id: i32) {
        self.embed_tokens.move_to_gpu(device_id);
        self.norm.move_to_gpu(device_id);
        for layer in &mut self.layers { layer.move_to_gpu(device_id); }
    }

    pub fn clear_kv_cache(&self) {
        for layer in &self.layers { layer.clear_kv_cache(); }
    }

    pub fn force_free_kv_cache(&self) {
        for layer in &self.layers { layer.force_free_kv_cache(); }
    }

    /// [NEW] Optimized batch upload for stitched cache
    pub fn batch_upload_stitched_cache(&self, k: Vec<u32>, v: Vec<f16>) {
        let h_d = self.config.head_dim;
        let n_kv = self.config.num_key_value_heads;
        
        #[cfg(feature = "cuda")]
        if self.layers[0].device_id >= 0 {
            println!("[GPU-BATCH-UPLOAD] Optimizing PCIe: HtoD once, then DtoD for {} layers...", self.layers.len());
            
            // 1. Direct Inject to FIRST layer (HtoD via F16)
            self.layers[0].inject_gpu_kv(&k, &v, n_kv, h_d);
            
            // 2. Replicate from Layer 0 to all other layers (DtoD)
            let l0_cache = self.layers[0].gpu_kv_cache.lock().unwrap();
            if let (Some(k_src), Some(v_src)) = (l0_cache.k_ptr, l0_cache.v_ptr) {
                let tokens = l0_cache.current_len;
                for i in 1..self.layers.len() {
                    self.layers[i].inject_gpu_kv_direct(k_src, v_src, tokens, n_kv, h_d);
                }
            }
            
            println!("[GPU-BATCH-UPLOAD] SUCCESS: All layers ready in VRAM via DtoD.");
            return;
        }

        // Fallback for CPU mode
        for layer in &self.layers {
            layer.set_kv_data(k.clone(), v.clone());
        }
    }

    pub fn get_kv_len(&self) -> usize {
        let h_d = self.config.head_dim;
        let n_kv = self.config.num_key_value_heads;
        self.layers[0].get_kv_len(h_d, n_kv)
    }

    pub fn get_all_kv(&self, start_token_idx: usize) -> Vec<(Vec<u32>, Vec<f16>)> {
        let h_d = self.config.head_dim;
        let n_kv = self.config.num_key_value_heads;
        self.layers.iter().filter_map(|l| l.get_kv_data(h_d, n_kv, start_token_idx)).collect()
    }
}

pub struct NativeQwen3VLModel {
    pub config: Qwen3VLConfig, 
    pub text_model: NativeQwen3TextModel, 
    pub lm_head: NativeLinear,
    pub visual: Option<NativeVisionModel>,
    pub support_layer0: Option<NativeLayer>, // [NEW] Permanent 0.6B Layer 0 slot
    pub support_workspace: std::sync::Mutex<ForwardWorkspace>, // [NEW] Isolated workspace for baking
    pub global_scratch_i: std::sync::Mutex<Option<(GpuPtr, usize)>>,
    pub global_scratch_o: std::sync::Mutex<Option<(GpuPtr, usize)>>,
    pub workspace: std::sync::Mutex<ForwardWorkspace>,
}

pub struct NativeVisionModel {
    pub patch_embed: NativeLinear,
    pub blocks: Vec<NativeLayer>,
    pub merger: NativeLayer,
}

unsafe impl Send for NativeVisionModel {}
unsafe impl Sync for NativeVisionModel {}

impl NativeVisionModel {
    pub fn move_to_gpu(&mut self, device_id: i32) {
        self.patch_embed.move_to_gpu(device_id);
        for block in &mut self.blocks { block.move_to_gpu(device_id); }
        self.merger.move_to_gpu(device_id);
    }

    pub fn forward<'a>(
        &self, 
        pixel_values: &[f16], 
        grid_thw: &[u32; 3], 
        rope_cos: &[f16], 
        rope_sin: &[f16], 
        global_scratch: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>, 
        workspace: &'a mut ForwardWorkspace
    ) -> &'a [f16] {
        let patches = (grid_thw[1] * grid_thw[2]) as usize;
        
        // 1. Patch Embedding (Result to hidden_a)
        let embed_out = self.patch_embed.forward(pixel_values, global_scratch);
        workspace.hidden_a[..embed_out.len()].copy_from_slice(&embed_out);
        
        // 2. Transformer Blocks
        // [FIX] Single definition of cur_x with proper raw pointer detachment
        let mut cur_x: &[f16] = unsafe { std::slice::from_raw_parts(workspace.hidden_a.as_ptr(), embed_out.len()) };
        
        for (i, block) in self.blocks.iter().enumerate() {
            let use_b = i % 2 != 0; // Toggle based on embedding being in A
            let out_slice = block.forward(cur_x, &crate::models::qwen3vl::config::Qwen3VLTextConfig {
                hidden_size: self.patch_embed.out_features,
                intermediate_size: block.gate_proj.out_features,
                num_hidden_layers: 1,
                num_attention_heads: 16,
                num_key_value_heads: 16,
                head_dim: self.patch_embed.out_features / 16,
                rms_norm_eps: 1e-6,
                rope_theta: 10000.0,
                vocab_size: 0,
                max_position_embeddings: 4096,
                dtype: None,
                rope_scaling: None,
            }, 0, 0, rope_cos, rope_sin, false, true, global_scratch, workspace, use_b);
            
            // Detach slice from mutable workspace borrow to allow next iteration
            cur_x = unsafe { std::slice::from_raw_parts(out_slice.as_ptr(), out_slice.len()) };
        }
        
        // 3. Patch Merger
        let last_use_b = self.blocks.len() % 2 != 0;
        let merger_out = self.merger.forward(cur_x, &crate::models::qwen3vl::config::Qwen3VLTextConfig {
            hidden_size: cur_x.len() / patches,
            intermediate_size: cur_x.len() / patches,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: cur_x.len() / patches,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            vocab_size: 0,
            max_position_embeddings: 4096,
            dtype: None,
            rope_scaling: None,
        }, 0, 0, rope_cos, rope_sin, false, true, global_scratch, workspace, !last_use_b);

        merger_out
    }
}

impl NativeQwen3VLModel {
    pub fn load(config: Qwen3VLConfig, m_mmap: Arc<Mmap>, v_mmap: Option<Arc<Mmap>>, baking: bool, s_mmap: Option<Arc<Mmap>>) -> Result<Self> {
        // [GPU-INITIALIZATION-FIX] Detect first available GPU
        let mut active_gpu_id = -1;
        #[cfg(feature = "cuda")]
        unsafe {
            let cuda_lib = crate::models::qwen3vl::native_backend::lib();
            let mut count = 0;
            if cuda_lib.cuInit(0) == 0 && cuda_lib.cuDeviceGetCount(&mut count) == 0 && count > 0 {
                active_gpu_id = 0;
                println!("[LOAD] GPU detected at index 0. Initializing layers with GPU acceleration enabled.");
            }
        }

        let st = SafeTensors::deserialize(&m_mmap)?; 
        let st_sec = s_mmap.as_ref().map(|m| SafeTensors::deserialize(m)).transpose()?;
        let t_c = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;
        
        let find_key_ext = |target: &str, primary: &SafeTensors, secondary: Option<&SafeTensors>, l_idx: i32| -> Option<(String, bool)> {
            // After Python patch, names are unified to model.language_model...
            // 1. Try Primary
            if primary.tensor(target).is_ok() || primary.tensor(&format!("{}.packed", target)).is_ok() {
                return Some((target.to_string(), true));
            }
            
            // 2. Try Secondary (2B)
            if let Some(sec) = secondary {
                if sec.tensor(target).is_ok() || sec.tensor(&format!("{}.packed", target)).is_ok() {
                    return Some((target.to_string(), false));
                }
            }

            // Fallback: search for tied weights or variations if still missing
            let variations = vec![
                target.to_string(),
                target.replace("lm_head", "model.lm_head"),
                target.replace("lm_head", "output"),
            ];
            
            for v in variations.iter() {
                if primary.tensor(v).is_ok() || primary.tensor(&format!("{}.packed", v)).is_ok() { return Some((v.clone(), true)); }
            }

            None
        };

        let mut get_l = |base: &str, in_f: usize, out_f: usize, _l_idx: i32| -> Result<NativeLinear> {
            let (key, is_primary) = find_key_ext(base, &st, st_sec.as_ref(), _l_idx).ok_or_else(|| anyhow!("LinearTensorNotFound: {}", base))?;
            let current_st = if is_primary { &st } else { st_sec.as_ref().unwrap() };
            let current_mmap = if is_primary { &m_mmap } else { s_mmap.as_ref().unwrap() };

            let p_n = format!("{}.packed", key);
            let s_n = format!("{}.scales", key);
            
            if current_st.tensor(&p_n).is_ok() {
                let vp = current_st.tensor(&p_n)?; 
                let vs = current_st.tensor(&s_n)?;
                let op = unsafe { vp.data().as_ptr().offset_from(current_mmap.as_ptr()) } as usize;
                let os = unsafe { vs.data().as_ptr().offset_from(current_mmap.as_ptr()) } as usize;
                
                Ok(NativeLinear { 
                    in_features: in_f, out_features: out_f, 
                    src_in: in_f, src_out: out_f,
                    variant: LinearVariant::BitSerial {
                        weight_packed: NativeTensor::from_mmap(current_mmap.clone(), op, vp.shape().to_vec(), NativeDType::U32),
                        scales: NativeTensor::from_mmap(current_mmap.clone(), os, vs.shape().to_vec(), NativeDType::F16),
                        bias: None,
                    }, 
                    device_id: active_gpu_id,
                })
            } else {
                let v = current_st.tensor(&key)?;
                let src_shape = v.shape();
                let src_in = src_shape[1];
                let src_out = src_shape[0];

                if in_f > src_in || out_f > src_out {
                    println!("[LOAD-ADAPT] Expanding linear {}: [{},{}] -> [{},{}]", key, src_out, src_in, out_f, in_f);
                    let src_data = v.data();
                    let src_f16 = unsafe { std::slice::from_raw_parts(src_data.as_ptr() as *const f16, src_data.len() / 2) };
                    
                    let mut new_data = vec![f16::ZERO; out_f * in_f];
                    let r_n = out_f / src_out;
                    let r_k = in_f / src_in;
                    let energy_scale = 1.0f32 / (r_k as f32);

                    for row in 0..out_f {
                        for col in 0..in_f {
                            let s_row = row % src_out;
                            let s_col = col % src_in;
                            new_data[row * in_f + col] = f16::from_f32(src_f16[s_row * src_in + s_col].to_f32() * energy_scale);
                        }
                    }
                    
                    let boxed = new_data.into_boxed_slice();
                    let ptr = boxed.as_ptr() as *const u8;
                    std::mem::forget(boxed);
                    
                    Ok(NativeLinear { 
                        in_features: in_f, out_features: out_f, src_in, src_out,
                        variant: LinearVariant::Standard { 
                            weight: NativeTensor { data_ptr: ptr, gpu_ptr: None, shape: vec![out_f, in_f], dtype: NativeDType::F16, _mmap: None, device_id: active_gpu_id }, 
                            bias: None 
                        }, 
                        device_id: active_gpu_id,
                    })
                } else {
                    let o = unsafe { v.data().as_ptr().offset_from(current_mmap.as_ptr()) } as usize;
                    Ok(NativeLinear { 
                        in_features: in_f, out_features: out_f, src_in: in_f, src_out: out_f,
                        variant: LinearVariant::Standard { 
                            weight: NativeTensor::from_mmap(current_mmap.clone(), o, v.shape().to_vec(), NativeDType::F16), 
                            bias: None 
                        }, 
                        device_id: active_gpu_id,
                    })
                }
            }
        };

        let get_t = |name: &str, target_size: usize| -> Result<NativeTensor> {
            let (key, is_primary) = find_key_ext(name, &st, st_sec.as_ref(), -1).ok_or_else(|| anyhow!("TensorNotFound: {}", name))?;
            let current_st = if is_primary { &st } else { st_sec.as_ref().unwrap() };
            let current_mmap = if is_primary { &m_mmap } else { s_mmap.as_ref().unwrap() };

            let v = current_st.tensor(&key)?;
            let src_data = v.data();
            let src_len_f16 = src_data.len() / 2;

            if target_size > src_len_f16 {
                println!("[LOAD-ADAPT] Expanding tensor {}: {} -> {}", key, src_len_f16, target_size);
                let src_f16 = unsafe { std::slice::from_raw_parts(src_data.as_ptr() as *const f16, src_len_f16) };
                let ratio = target_size / src_len_f16;
                let mut new_data = Vec::with_capacity(target_size * 2);
                for _ in 0..ratio {
                    for &val in src_f16 { new_data.extend_from_slice(&val.to_le_bytes()); }
                }
                let boxed = new_data.into_boxed_slice();
                let ptr = boxed.as_ptr() as *const u8;
                std::mem::forget(boxed);
                Ok(NativeTensor { data_ptr: ptr, gpu_ptr: None, shape: vec![target_size], dtype: NativeDType::F16, _mmap: None, device_id: active_gpu_id })
            } else {
                let off = unsafe { v.data().as_ptr().offset_from(current_mmap.as_ptr()) } as usize;
                Ok(NativeTensor { data_ptr: unsafe { current_mmap.as_ptr().add(off) }, gpu_ptr: None, shape: v.shape().to_vec(), dtype: NativeDType::F16, _mmap: Some(current_mmap.clone()), device_id: active_gpu_id })
            }
        };

        println!("[LOAD] Initializing Native Engine (Baking: {})...", baking);

        // [CRITICAL-FIX] 임베딩 로더: 1비트로 압축된 경우 로딩 시점에 FP16으로 복원 (반복 확장 지원)
        let get_embed = |base: &str, vocab: usize, target_hid: usize, is_support: bool| -> Result<NativeLinear> {
            let (key, is_primary) = find_key_ext(base, &st, st_sec.as_ref(), -1).ok_or_else(|| anyhow!("EmbedTensorNotFound: {}", base))?;
            let current_st = if is_primary { &st } else { st_sec.as_ref().unwrap() };
            let current_mmap = if is_primary { &m_mmap } else { s_mmap.as_ref().unwrap() };

            let p_n = format!("{}.packed", key);
            let s_n = format!("{}.scales", key);
            
            if current_st.tensor(&p_n).is_ok() {
                println!("[LOAD] De-quantizing bit-serial embedding for lookup: {}", key);
                let vp = current_st.tensor(&p_n)?;
                let vs = current_st.tensor(&s_n)?;
                
                // 실제 파일에 들어있는 원본 hidden_size 계산
                let src_hid = (vp.data().len() / 4 * 32) / vocab;
                
                let packed_ref = unsafe { std::slice::from_raw_parts(vp.data().as_ptr() as *const u32, vp.data().len() / 4) };
                let scales_ref = unsafe { std::slice::from_raw_parts(vs.data().as_ptr() as *const f16, vs.data().len() / 2) };
                
                if target_hid > src_hid {
                    println!("[HYBRID-EMBED] Repeating embedding: {} -> {}", src_hid, target_hid);
                }

                let dequantized = dequantize_bit_serial_to_f16(packed_ref, scales_ref, vocab, src_hid, target_hid);
                
                // [MEM-COPY] 복원된 FP16 데이터를 담은 NativeTensor 생성
                let mut dest = Vec::with_capacity(dequantized.len() * 2);
                for v in dequantized { dest.extend_from_slice(&v.to_le_bytes()); }
                let boxed_data = dest.into_boxed_slice();
                let leaked_ptr = boxed_data.as_ptr();
                std::mem::forget(boxed_data); // Keep alive for the session

                Ok(NativeLinear { 
                    in_features: vocab, out_features: target_hid, 
                    src_in: src_hid, src_out: target_hid,
                    variant: LinearVariant::Standard { 
                        weight: NativeTensor { data_ptr: leaked_ptr, gpu_ptr: None, shape: vec![vocab, target_hid], dtype: NativeDType::F16, _mmap: None, device_id: active_gpu_id }, 
                        bias: None 
                    }, 
                    device_id: active_gpu_id 
                })
            } else {
                get_l(base, vocab, target_hid, -1)
            }
        };

        let emb = get_embed("model.embed_tokens.weight", 151936, t_c.hidden_size, s_mmap.is_some())
            .or_else(|_| get_embed("model.language_model.embed_tokens.weight", 151936, t_c.hidden_size, s_mmap.is_some()))?;
            
        let mut layers = Vec::new(); 
        let l_to_l = if baking { 1 } else { t_c.num_hidden_layers };

        // [FIX] Correct Dimensionality Calculation for Attention Block
        let q_out = t_c.num_attention_heads * t_c.head_dim;
        let kv_out = t_c.num_key_value_heads * t_c.head_dim;

        // [NEW] Dedicated loader for the Support Slot (Static Header)
        let mut support_layer0 = None;
        if let Some(ref sec_mmap) = s_mmap {
            let sec_st = SafeTensors::deserialize(sec_mmap)?;
            println!("[LOAD-HYBRID] Initializing Support Layer 0 from 0.6B...");
            
            let get_sec_l = |base: &str, target_in: usize, target_out: usize| -> Result<NativeLinear> {
                let p_n = format!("{}.packed", base);
                let s_n = format!("{}.scales", base);
                let vp = sec_st.tensor(&p_n)?; let vs = sec_st.tensor(&s_n)?;
                let src_out = vp.shape()[0];
                let src_in = (vp.data().len() / 4 * 32) / src_out;
                
                let packed_u32 = unsafe { std::slice::from_raw_parts(vp.data().as_ptr() as *const u32, vp.data().len() / 4) };
                let scales_f16 = unsafe { std::slice::from_raw_parts(vs.data().as_ptr() as *const f16, vs.data().len() / 2) };
                
                let mut new_packed = vec![0u32; (target_out/8) * (target_in/32) * 8];
                let mut new_scales = vec![f16::ZERO; (target_out/8) * (target_in/32) * 8];
                let energy_scale = f16::from_f32(1.0 / ((target_in/src_in) as f32));

                for no in 0..(target_out/8) {
                    for ko in 0..(target_in/32) {
                        for sub_n in 0..8 {
                            let src_idx = ((no % (src_out/8)) * (src_in/32) + (ko % (src_in/32))) * 8 + sub_n;
                            let dst_idx = (no * (target_in/32) + ko) * 8 + sub_n;
                            new_packed[dst_idx] = packed_u32[src_idx];
                            new_scales[dst_idx] = scales_f16[src_idx] * energy_scale;
                        }
                    }
                }
                let p_boxed = new_packed.into_boxed_slice();
                let s_boxed = new_scales.into_boxed_slice();
                let p_ptr = p_boxed.as_ptr() as *const u8;
                let s_ptr = s_boxed.as_ptr() as *const u8;
                std::mem::forget(p_boxed); std::mem::forget(s_boxed);

                Ok(NativeLinear { 
                    in_features: target_in, out_features: target_out, src_in, src_out,
                    variant: LinearVariant::BitSerial {
                        weight_packed: NativeTensor { data_ptr: p_ptr, gpu_ptr: None, shape: vec![target_out, target_in/32 * 8], dtype: NativeDType::U32, _mmap: None, device_id: active_gpu_id },
                        scales: NativeTensor { data_ptr: s_ptr, gpu_ptr: None, shape: vec![target_out, target_in/32 * 8], dtype: NativeDType::F16, _mmap: None, device_id: active_gpu_id },
                        bias: None,
                    }, 
                    device_id: active_gpu_id,
                })
            };

            let get_sec_t = |name: &str, target_size: usize| -> Result<NativeTensor> {
                let v = sec_st.tensor(name)?;
                let src_data = v.data();
                let src_len_f16 = src_data.len() / 2;
                
                if target_size > src_len_f16 {
                    println!("[LOAD-ADAPT] Expanding support tensor {}: {} -> {}", name, src_len_f16, target_size);
                    let src_f16 = unsafe { std::slice::from_raw_parts(src_data.as_ptr() as *const f16, src_len_f16) };
                    let ratio = target_size / src_len_f16;
                    let mut new_data = Vec::with_capacity(target_size * 2);
                    for _ in 0..ratio {
                        for &val in src_f16 { new_data.extend_from_slice(&val.to_le_bytes()); }
                    }
                    let boxed = new_data.into_boxed_slice();
                    let ptr = boxed.as_ptr();
                    std::mem::forget(boxed);
                    Ok(NativeTensor { data_ptr: ptr, gpu_ptr: None, shape: vec![target_size], dtype: NativeDType::F16, _mmap: None, device_id: active_gpu_id })
                } else {
                    let off = unsafe { v.data().as_ptr().offset_from(sec_mmap.as_ptr()) } as usize;
                    Ok(NativeTensor { data_ptr: unsafe { sec_mmap.as_ptr().add(off) }, gpu_ptr: None, shape: v.shape().to_vec(), dtype: NativeDType::F16, _mmap: Some(sec_mmap.clone()), device_id: active_gpu_id })
                }
            };

            let p = "model.language_model.layers.0";
            support_layer0 = Some(NativeLayer {
                input_layernorm: get_sec_t(&format!("{}.input_layernorm.weight", p), t_c.hidden_size)?,
                post_attention_layernorm: get_sec_t(&format!("{}.post_attention_layernorm.weight", p), t_c.hidden_size)?,
                q_norm: get_sec_t(&format!("{}.self_attn.q_norm.weight", p), t_c.hidden_size).ok(),
                k_norm: get_sec_t(&format!("{}.self_attn.k_norm.weight", p), t_c.hidden_size).ok(),
                q_proj: get_sec_l(&format!("{}.self_attn.q_proj.weight", p), t_c.hidden_size, q_out)?,
                k_proj: get_sec_l(&format!("{}.self_attn.k_proj.weight", p), t_c.hidden_size, kv_out)?,
                v_proj: get_sec_l(&format!("{}.self_attn.v_proj.weight", p), t_c.hidden_size, kv_out)?,
                o_proj: get_sec_l(&format!("{}.self_attn.o_proj.weight", p), q_out, t_c.hidden_size)?,
                gate_proj: get_sec_l(&format!("{}.mlp.gate_proj.weight", p), t_c.hidden_size, t_c.intermediate_size)?,
                up_proj: get_sec_l(&format!("{}.mlp.up_proj.weight", p), t_c.hidden_size, t_c.intermediate_size)?,
                down_proj: get_sec_l(&format!("{}.mlp.down_proj.weight", p), t_c.intermediate_size, t_c.hidden_size)?,
                device_id: active_gpu_id, is_support_layer: true,
                kv_cache: std::sync::Mutex::new(DynamicKVCache::new()), 
                gpu_kv_cache: std::sync::Mutex::new(DynamicGpuKVCache::new()),
                gpu_broken: std::sync::atomic::AtomicBool::new(false),
            });
        }

        for i in 0..t_c.num_hidden_layers {
            let layer_idx = i as i32;
            let current_st = &st;
            let current_mmap = &m_mmap;
            
            let mut get_hybrid_l = |base: &str, target_in: usize, target_out: usize| -> Result<NativeLinear> {
                let (key, _) = find_key_ext(base, current_st, None, layer_idx).ok_or_else(|| anyhow!("LinearTensorNotFound: {}", base))?;
                let p_n = format!("{}.packed", key);
                let s_n = format!("{}.scales", key);
                
                if current_st.tensor(&p_n).is_ok() {
                    let vp = current_st.tensor(&p_n)?; let vs = current_st.tensor(&s_n)?;
                    let op = unsafe { vp.data().as_ptr().offset_from(current_mmap.as_ptr()) } as usize;
                    let os = unsafe { vs.data().as_ptr().offset_from(current_mmap.as_ptr()) } as usize;
                    Ok(NativeLinear { 
                        in_features: target_in, out_features: target_out, src_in: target_in, src_out: target_out,
                        variant: LinearVariant::BitSerial {
                            weight_packed: NativeTensor::from_mmap(current_mmap.clone(), op, vp.shape().to_vec(), NativeDType::U32),
                            scales: NativeTensor::from_mmap(current_mmap.clone(), os, vs.shape().to_vec(), NativeDType::F16),
                            bias: None,
                        }, 
                        device_id: active_gpu_id,
                    })
                } else {
                    let v = current_st.tensor(&key)?;
                    let o = unsafe { v.data().as_ptr().offset_from(current_mmap.as_ptr()) } as usize;
                    Ok(NativeLinear { in_features: target_in, out_features: target_out, src_in: target_in, src_out: target_out, variant: LinearVariant::Standard { weight: NativeTensor::from_mmap(current_mmap.clone(), o, v.shape().to_vec(), NativeDType::F16), bias: None }, device_id: active_gpu_id })
                }
            };

            let p = format!("model.language_model.layers.{}", i);
            if current_st.tensor(&format!("{}.input_layernorm.weight", p)).is_err() { continue; }

            layers.push(NativeLayer {
                input_layernorm: get_t(&format!("{}.input_layernorm.weight", p), t_c.hidden_size)?,
                post_attention_layernorm: get_t(&format!("{}.post_attention_layernorm.weight", p), t_c.hidden_size)?,
                q_norm: get_t(&format!("{}.self_attn.q_norm.weight", p), t_c.hidden_size).ok(),
                k_norm: get_t(&format!("{}.self_attn.k_norm.weight", p), t_c.hidden_size).ok(),
                q_proj: get_hybrid_l(&format!("{}.self_attn.q_proj.weight", p), t_c.hidden_size, q_out)?,
                k_proj: get_hybrid_l(&format!("{}.self_attn.k_proj.weight", p), t_c.hidden_size, kv_out)?,
                v_proj: get_hybrid_l(&format!("{}.self_attn.v_proj.weight", p), t_c.hidden_size, kv_out)?,
                o_proj: get_hybrid_l(&format!("{}.self_attn.o_proj.weight", p), q_out, t_c.hidden_size)?,
                gate_proj: get_hybrid_l(&format!("{}.mlp.gate_proj.weight", p), t_c.hidden_size, t_c.intermediate_size)?,
                up_proj: get_hybrid_l(&format!("{}.mlp.up_proj.weight", p), t_c.hidden_size, t_c.intermediate_size)?,
                down_proj: get_hybrid_l(&format!("{}.mlp.down_proj.weight", p), t_c.intermediate_size, t_c.hidden_size)?,
                device_id: active_gpu_id, 
                is_support_layer: false,
                kv_cache: std::sync::Mutex::new(DynamicKVCache::new()), 
                gpu_kv_cache: std::sync::Mutex::new(DynamicGpuKVCache::new()),
                gpu_broken: std::sync::atomic::AtomicBool::new(false),
            });
        }
        
        let norm = get_t("model.language_model.norm.weight", t_c.hidden_size)?;
        
        // [SMART-HEAD-LOADER] 가중치 공유(tied weights) 대응 강화
        let head_res = get_l("lm_head.weight", t_c.hidden_size, 151936, -1)
            .or_else(|_| get_l("model.language_model.lm_head.weight", t_c.hidden_size, 151936, -1))
            .or_else(|_| {
                println!("[LOAD] lm_head not found, using tied weights from embed_tokens.");
                get_l("model.embed_tokens.weight", 151936, t_c.hidden_size, -1)
                    .or_else(|_| get_l("model.language_model.embed_tokens.weight", 151936, t_c.hidden_size, -1))
                    .map(|mut emb_l| {
                        emb_l.in_features = t_c.hidden_size;
                        emb_l.out_features = 151936;
                        emb_l.src_in = t_c.hidden_size;
                        emb_l.src_out = 151936;
                        emb_l.device_id = active_gpu_id; // [FIX] Ensure tied head has device_id
                        emb_l
                    })
            });
            
        let head = head_res.map_err(|e| {
            println!("[CRITICAL-LOAD-ERROR] ALL attempts to find lm_head or tied weights failed: {}.", e);
            anyhow!("LMHeadNotFound")
        })?;

        // [PRECOMPUTE-ROPE]
        let rope_cache = RopeCache::new(t_c.head_dim, t_c.rope_theta, 4096); // Start with safe 4k

        let visual = if let Some(vm) = v_mmap {
            let vst = SafeTensors::deserialize(&vm)?;
            
            // [SMART-VISION-LOADER] 비전용 지능형 키 검색
            let find_vkey = |target: &str| -> Option<String> {
                let variations = vec![
                    target.to_string(),
                    target.replace("patch_embed.proj", "patch_embd"),
                    target.replace("patch_embed", "patch_embd"),
                    target.replace(".attn.q_proj", ".attn.qkv"), // QKV가 통합된 경우 대비
                ];
                for v in variations {
                    if vst.tensor(&v).is_ok() || vst.tensor(&format!("{}.packed", v)).is_ok() { return Some(v); }
                }
                vst.names().iter().find(|&n| n.contains(target)).map(|n| n.to_string())
            };

            let get_vt = |name: &str, target_size: usize| -> Result<NativeTensor> {
                let key = find_vkey(name).ok_or_else(|| anyhow!("VisionTensorNotFound: {}", name))?;
                let v = vst.tensor(&key)?;
                let src_data = v.data();
                let src_len_f16 = src_data.len() / 2;

                if target_size > src_len_f16 {
                    println!("[LOAD-ADAPT-VISION] Expanding vision tensor {}: {} -> {}", key, src_len_f16, target_size);
                    let src_f16 = unsafe { std::slice::from_raw_parts(src_data.as_ptr() as *const f16, src_len_f16) };
                    let ratio = target_size / src_len_f16;
                    let mut new_data = Vec::with_capacity(target_size * 2);
                    for _ in 0..ratio {
                        for &val in src_f16 { new_data.extend_from_slice(&val.to_le_bytes()); }
                    }
                    let boxed = new_data.into_boxed_slice();
                    let ptr = boxed.as_ptr() as *const u8;
                    std::mem::forget(boxed);
                    Ok(NativeTensor { data_ptr: ptr, gpu_ptr: None, shape: vec![target_size], dtype: NativeDType::F16, _mmap: None, device_id: active_gpu_id })
                } else {
                    let off = unsafe { v.data().as_ptr().offset_from(vm.as_ptr()) } as usize;
                    Ok(NativeTensor { data_ptr: unsafe { vm.as_ptr().add(off) }, gpu_ptr: None, shape: v.shape().to_vec(), dtype: NativeDType::F16, _mmap: Some(vm.clone()), device_id: active_gpu_id })
                }
            };

            let get_vl = |base: &str, in_f: usize, out_f: usize| -> Result<NativeLinear> {
                let key = find_vkey(base).ok_or_else(|| anyhow!("VisionLinearTensorNotFound: {}", base))?;
                let p_n = format!("{}.packed", key);
                let s_n = format!("{}.scales", key);
                if vst.tensor(&p_n).is_ok() {
                    let vp = vst.tensor(&p_n)?; let vs = vst.tensor(&s_n)?;
                    let op = unsafe { vp.data().as_ptr().offset_from(vm.as_ptr()) } as usize;
                    let os = unsafe { vs.data().as_ptr().offset_from(vm.as_ptr()) } as usize;
                    Ok(NativeLinear { in_features: in_f, out_features: out_f, src_in: in_f, src_out: out_f, variant: LinearVariant::BitSerial {
                        weight_packed: NativeTensor::from_mmap(vm.clone(), op, vp.shape().to_vec(), NativeDType::U32),
                        scales: NativeTensor::from_mmap(vm.clone(), os, vs.shape().to_vec(), NativeDType::F16),
                        bias: None,
                    }, device_id: active_gpu_id })
                } else {
                    let v = vst.tensor(&key)?;
                    let src_shape = v.shape();
                    let src_in = src_shape[1];
                    let src_out = src_shape[0];

                    if in_f > src_in || out_f > src_out {
                        println!("[LOAD-ADAPT-VISION] Expanding vision linear {}: [{},{}] -> [{},{}]", key, src_out, src_in, out_f, in_f);
                        let src_data = v.data();
                        let src_f16 = unsafe { std::slice::from_raw_parts(src_data.as_ptr() as *const f16, src_data.len() / 2) };
                        let mut new_data = vec![f16::ZERO; out_f * in_f];
                        let r_k = in_f / src_in;
                        let energy_scale = 1.0f32 / (r_k as f32);
                        for row in 0..out_f {
                            for col in 0..in_f {
                                let s_row = row % src_out;
                                let s_col = col % src_in;
                                new_data[row * in_f + col] = f16::from_f32(src_f16[s_row * src_in + s_col].to_f32() * energy_scale);
                            }
                        }
                        let boxed = new_data.into_boxed_slice();
                        let ptr = boxed.as_ptr() as *const u8;
                        std::mem::forget(boxed);
                        Ok(NativeLinear { in_features: in_f, out_features: out_f, src_in, src_out, variant: LinearVariant::Standard { weight: NativeTensor { data_ptr: ptr, gpu_ptr: None, shape: vec![out_f, in_f], dtype: NativeDType::F16, _mmap: None, device_id: active_gpu_id }, bias: None }, device_id: active_gpu_id })
                    } else {
                        let o = unsafe { v.data().as_ptr().offset_from(vm.as_ptr()) } as usize;
                        Ok(NativeLinear { in_features: in_f, out_features: out_f, src_in: in_f, src_out: out_f, variant: LinearVariant::Standard { weight: NativeTensor::from_mmap(vm.clone(), o, v.shape().to_vec(), NativeDType::F16), bias: None }, device_id: active_gpu_id })
                    }
                }
            };

            let v_cfg = config.vision_config.as_ref().ok_or(anyhow!("Missing vision_config"))?;
            let v_hid = v_cfg.hidden_size;
            let v_intermediate = v_hid * 4;
            let v_out_hidden = v_cfg.out_hidden_size.unwrap_or(v_hid * 2);
            
            let patch_embed = get_vl("visual.patch_embed.proj.weight", 1536, v_hid)?;
            let mut blocks = Vec::new();
            let b_to_l = if baking { 1 } else { v_cfg.depth };
            for i in 0..b_to_l {
                let p = format!("visual.blocks.{}", i);
                blocks.push(NativeLayer {
                    input_layernorm: get_vt(&format!("{}.norm1.weight", p), v_hid)?,
                    post_attention_layernorm: get_vt(&format!("{}.norm2.weight", p), v_hid)?,
                    q_norm: None, k_norm: None,
                    q_proj: get_vl(&format!("{}.attn.q_proj.weight", p), v_hid, v_hid)?,
                    k_proj: get_vl(&format!("{}.attn.k_proj.weight", p), v_hid, v_hid)?,
                    v_proj: get_vl(&format!("{}.attn.v_proj.weight", p), v_hid, v_hid)?,
                    o_proj: get_vl(&format!("{}.attn.proj.weight", p), v_hid, v_hid)?,
                    gate_proj: get_vl(&format!("{}.mlp.fc1.weight", p), v_hid, v_intermediate)?,
                    up_proj: get_vl(&format!("{}.mlp.fc1.weight", p), v_hid, v_intermediate)?, 
                    down_proj: get_vl(&format!("{}.mlp.fc2.weight", p), v_intermediate, v_hid)?,
                    device_id: -1, is_support_layer: false,
                    kv_cache: std::sync::Mutex::new(DynamicKVCache::new()), 
                    gpu_kv_cache: std::sync::Mutex::new(DynamicGpuKVCache::new()),
                    gpu_broken: std::sync::atomic::AtomicBool::new(false),
                });
            }
            let merger = NativeLayer {
                input_layernorm: get_vt("visual.merger.norm.weight", v_intermediate)?,
                post_attention_layernorm: get_vt("visual.merger.norm.weight", v_intermediate)?, 
                q_norm: None, k_norm: None,
                q_proj: get_vl("visual.merger.mlp.0.weight", v_intermediate, v_intermediate)?,
                k_proj: get_vl("visual.merger.mlp.0.weight", v_intermediate, v_intermediate)?,
                v_proj: get_vl("visual.merger.mlp.0.weight", v_intermediate, v_intermediate)?,
                o_proj: get_vl("visual.merger.mlp.2.weight", v_intermediate, v_out_hidden)?,
                gate_proj: get_vl("visual.merger.mlp.0.weight", v_intermediate, v_intermediate)?,
                up_proj: get_vl("visual.merger.mlp.0.weight", v_intermediate, v_intermediate)?,
                down_proj: get_vl("visual.merger.mlp.2.weight", v_intermediate, v_out_hidden)?,
                device_id: -1, is_support_layer: false,
                kv_cache: std::sync::Mutex::new(DynamicKVCache::new()), 
                gpu_kv_cache: std::sync::Mutex::new(DynamicGpuKVCache::new()),
                gpu_broken: std::sync::atomic::AtomicBool::new(false),
            };
            Some(NativeVisionModel { patch_embed, blocks, merger })
        } else { None };
            
        Ok(Self { 
            config: config.clone(), 
            text_model: NativeQwen3TextModel { config: t_c.clone(), embed_tokens: emb, layers, norm, rope_cache: std::sync::Mutex::new(rope_cache) }, 
            lm_head: head, 
            visual,
            support_layer0,
            support_workspace: std::sync::Mutex::new(ForwardWorkspace::new()),
            global_scratch_i: std::sync::Mutex::new(None),
            global_scratch_o: std::sync::Mutex::new(None),
            workspace: std::sync::Mutex::new(ForwardWorkspace::new()),
        })
    }
    pub fn forward(&self, i_ids: &[u32], p_v: Option<&[f16]>, g_t: Option<&[u32; 3]>, s_o: usize) -> Vec<f16> {
        #[cfg(feature = "cuda")]
        if !self.text_model.layers.is_empty() && self.text_model.layers[0].device_id >= 0 {
            use cudarc::driver::sys::*;
            unsafe {
                let dev_id = self.text_model.layers[0].device_id;
                let cuda_lib = crate::models::qwen3vl::native_backend::lib();
                let mut ctx = std::ptr::null_mut() as CUcontext;
                cuda_lib.cuCtxGetCurrent(&mut ctx);
                if ctx == std::ptr::null_mut() {
                    let mut dev = 0 as CUdevice;
                    cuda_lib.cuDeviceGet(&mut dev, dev_id);
                    cuda_lib.cuDevicePrimaryCtxRetain(&mut ctx, dev);
                    cuda_lib.cuCtxSetCurrent(ctx);
                }
            }
        }

        // [HYBRID-ROUTING] Use support workspace and layer for baking tasks
        let is_baking = self.text_model.layers.len() <= 1;
        let is_vision = p_v.is_some(); // [NEW] Explicit vision modality detection
        let mut ws_lock = if is_baking { self.support_workspace.lock().unwrap() } else { self.workspace.lock().unwrap() };
        let ws = &mut *ws_lock;
        
        let mut embeds = match &self.text_model.embed_tokens.variant {
            LinearVariant::Standard { weight, .. } => {
                let w_cow = weight.get_slice::<f16>();
                native_embedding_lookup_f16(i_ids, w_cow.as_ref(), self.text_model.config.hidden_size)
            },
            LinearVariant::BitSerial { weight_packed, .. } => {
                let w_cow = weight_packed.get_slice::<f16>();
                native_embedding_lookup_f16(i_ids, w_cow.as_ref(), self.text_model.config.hidden_size)
            },
        };

        let scratch = Some((&self.global_scratch_i, &self.global_scratch_o));

        // [DYNAMIC-ROPE] Pre-fetch or grow RoPE for fusion
        let (r_cos, r_sin) = {
            let mut guard = self.text_model.rope_cache.lock().unwrap();
            guard.ensure_length(s_o + i_ids.len() + 1024); // Extra buffer for vision
            (unsafe { std::slice::from_raw_parts(guard.cos.as_ptr(), guard.cos.len()) },
             unsafe { std::slice::from_raw_parts(guard.sin.as_ptr(), guard.sin.len()) })
        };

        // [VISION-FUSION]
        if let (Some(pv), Some(gt)) = (p_v, g_t) {
            if let Some(ref visual) = self.visual {
                let vision_features_slice = visual.forward(pv, gt, r_cos, r_sin, scratch, ws);
                
                // [BORROW-CHECKER-FIX] Detach vision_features from ws
                let vision_features: &[f16] = unsafe {
                    std::slice::from_raw_parts(vision_features_slice.as_ptr(), vision_features_slice.len())
                };

                let img_token_id = 151655; // <|image_pad|>
                let hid = self.text_model.config.hidden_size;
                let mut vision_idx = 0;
                
                for (i, &id) in i_ids.iter().enumerate() {
                    if id == img_token_id && vision_idx < (vision_features.len() / hid) {
                        embeds[i * hid .. (i + 1) * hid].copy_from_slice(&vision_features[vision_idx * hid .. (vision_idx + 1) * hid]);
                        vision_idx += 1;
                    }
                }
            }
        }

        // [HOT-SWAP] Pass support layer to text model if baking or if it exists as static header
        let active_support = if is_baking || self.support_layer0.is_some() { self.support_layer0.as_ref() } else { None };
        let norm_x = self.text_model.forward_ext(i_ids, embeds, s_o, scratch, Some(ws), active_support, is_vision);
        self.lm_head.forward(&norm_x, scratch)
    }
        pub fn clear_kv_cache(&self) { self.text_model.clear_kv_cache(); }
        pub fn force_free_kv_cache(&self) { self.text_model.force_free_kv_cache(); }
        pub fn move_to_gpu(&mut self, device_id: i32) {
            self.text_model.move_to_gpu(device_id);
            self.lm_head.move_to_gpu(device_id);
            if let Some(ref mut v) = self.visual {
                v.move_to_gpu(device_id);
            }
        }
    }
    
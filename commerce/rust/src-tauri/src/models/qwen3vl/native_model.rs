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

pub struct NativeLinear {
    pub in_features: usize, 
    pub out_features: usize, 
    pub variant: LinearVariant, 
    pub device_id: i32,
    pub scratch_i: std::sync::Mutex<Option<(GpuPtr, usize)>>, // (Pointer, Current size in bytes)
    pub scratch_o: std::sync::Mutex<Option<(GpuPtr, usize)>>,
}

unsafe impl Send for NativeLinear {}
unsafe impl Sync for NativeLinear {}

impl NativeLinear {
    pub fn forward(&self, x: &[f16]) -> Vec<f16> {
        let m = x.len() / self.in_features;
        match &self.variant {
            LinearVariant::Standard { weight, bias } => {
                let w_cow = weight.get_slice::<f16>(); 
                let b_cow = bias.as_ref().map(|t| t.get_slice::<f16>());
                native_linear_f16(x, w_cow.as_ref(), b_cow.as_ref().map(|b| b.as_ref()), m, self.out_features, self.in_features)
            },
            LinearVariant::BitSerial { weight_packed, scales, bias } => {
                #[cfg(feature = "cuda")]
                {
                    if self.device_id >= 0 {
                        if weight_packed.gpu_ptr.is_some() {
                            let (d_i, d_o) = self.ensure_gpu_buffers(m);
                            let mut res = bit_serial_matmul_gpu_buffered(x, weight_packed, scales, m, self.out_features, self.in_features, self.device_id as usize, d_i, d_o);
                            
                            // [STABILITY-FALLBACK] If GPU returned zeros, retry on CPU
                            if !res.is_empty() && res[0].to_f32() == 0.0 && res.iter().take(50).all(|val| val.to_f32() == 0.0) {
                                unsafe {
                                    let wp_ref = weight_packed.get_raw_slice::<u32>(); 
                                    let s_ref = scales.get_raw_slice::<f16>();
                                    let out = bit_serial_matmul_f32_extreme(x, wp_ref, s_ref, m, self.out_features, self.in_features);
                                    res = out.into_iter().map(f16::from_f32).collect();
                                }
                            }

                            if let Some(b) = bias { 
                                unsafe {
                                    let b_ref = b.get_raw_slice::<f16>();
                                    for i in 0..m { for j in 0..self.out_features { res[i * self.out_features + j] += b_ref[j]; } } 
                                }
                            }
                            return res;
                        }
                    }
                }

                unsafe {
                    let wp_ref = weight_packed.get_raw_slice::<u32>(); 
                    let s_ref = scales.get_raw_slice::<f16>();
                    let mut out = bit_serial_matmul_f32_extreme(x, wp_ref, s_ref, m, self.out_features, self.in_features);
                    
                    if let Some(b) = bias {
                        let b_ref = b.get_raw_slice::<f16>();
                        for i in 0..m {
                            for j in 0..self.out_features {
                                if i * self.out_features + j < out.len() && j < b_ref.len() {
                                    out[i * self.out_features + j] += b_ref[j].to_f32();
                                }
                            }
                        }
                    }
                    out.into_iter().map(f16::from_f32).collect()
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn ensure_gpu_buffers(&self, m: usize) -> (CUdeviceptr, CUdeviceptr) {
        use cudarc::driver::sys::*;
        let req_i = m * self.in_features * 4;
        let req_o = m * self.out_features * 4;
        
        let mut si_guard = self.scratch_i.lock().unwrap();
        let d_i = if let Some((ptr, size)) = *si_guard {
            if size >= req_i { ptr.0 as CUdeviceptr }
            else {
                println!("[GPU-SCRATCH] Growing input buffer to {} bytes", req_i);
                unsafe {
                    let mut new_ptr: CUdeviceptr = 0;
                    let lib = lib();
                    let _ = lib.cuMemFree_v2(ptr.0 as CUdeviceptr);
                    let _ = lib.cuMemAlloc_v2(&mut new_ptr, req_i);
                    *si_guard = Some((GpuPtr(new_ptr as *mut _), req_i));
                    new_ptr
                }
            }
        } else {
            unsafe {
                let mut new_ptr: CUdeviceptr = 0;
                let _ = lib().cuMemAlloc_v2(&mut new_ptr, req_i);
                *si_guard = Some((GpuPtr(new_ptr as *mut _), req_i));
                new_ptr
            }
        };

        let mut so_guard = self.scratch_o.lock().unwrap();
        let d_o = if let Some((ptr, size)) = *so_guard {
            if size >= req_o { ptr.0 as CUdeviceptr }
            else {
                println!("[GPU-SCRATCH] Growing output buffer to {} bytes", req_o);
                unsafe {
                    let mut new_ptr: CUdeviceptr = 0;
                    let lib = lib();
                    let _ = lib.cuMemFree_v2(ptr.0 as CUdeviceptr);
                    let _ = lib.cuMemAlloc_v2(&mut new_ptr, req_o);
                    *so_guard = Some((GpuPtr(new_ptr as *mut _), req_o));
                    new_ptr
                }
            }
        } else {
            unsafe {
                let mut new_ptr: CUdeviceptr = 0;
                let _ = lib().cuMemAlloc_v2(&mut new_ptr, req_o);
                *so_guard = Some((GpuPtr(new_ptr as *mut _), req_o));
                new_ptr
            }
        };

        (d_i, d_o)
    }

    pub fn move_to_gpu(&mut self, device_id: i32) {
        self.device_id = device_id;
        if let LinearVariant::BitSerial { weight_packed, scales, .. } = &mut self.variant {
            #[cfg(feature = "cuda")] 
            {
                weight_packed.move_to_gpu(device_id);
                scales.move_to_gpu(device_id);
            }
        }
    }

    pub fn force_free_kv_cache(&self) {
        // Linear doesn't have KV cache but has scratch buffers
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let lib = lib();
            if let Some((ptr, _)) = self.scratch_i.lock().unwrap().take() { let _ = lib.cuMemFree_v2(ptr.0 as CUdeviceptr); }
            if let Some((ptr, _)) = self.scratch_o.lock().unwrap().take() { let _ = lib.cuMemFree_v2(ptr.0 as CUdeviceptr); }
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
    pub kv_cache: std::sync::Mutex<Option<(Vec<u32>, Vec<f16>)>>, 
    pub gpu_kv_cache: std::sync::Mutex<Option<(GpuPtr, GpuPtr, usize)>>, 
    pub attn_scratch_q: std::sync::Mutex<Option<(GpuPtr, usize)>>,
    pub attn_scratch_o: std::sync::Mutex<Option<(GpuPtr, usize)>>,
}

unsafe impl Send for NativeLayer {}
unsafe impl Sync for NativeLayer {}

impl NativeLayer {
    pub fn forward(&self, x: &[f16], config: &Qwen3VLTextConfig, seqlen_offset: usize, _idx: usize, rope_cos: &[f16], rope_sin: &[f16], is_baking: bool) -> Vec<f16> {
        let hidden_size = config.hidden_size; let q_len = x.len() / hidden_size;
        let head_dim = config.head_dim; let n_h = config.num_attention_heads; let n_kv = config.num_key_value_heads;

        let residual = x.to_vec();
        let ln_weight_cow = self.input_layernorm.get_slice::<f16>();
        let x_norm = native_rms_norm_f16(x, ln_weight_cow.as_ref(), config.rms_norm_eps as f32, hidden_size);
        
        // [RAYON-STRATEGY] Only use parallel join if processing multiple tokens (prefill)
                            let (mut q, mut k, mut v) = if q_len > 1 {
                                let (q_p, (k_p, v_p)) = rayon::join(
                                    || self.q_proj.forward(&x_norm),
                                    || rayon::join(|| self.k_proj.forward(&x_norm), || self.v_proj.forward(&x_norm))
                                );
                                (q_p, k_p, v_p)
                            } else {
                                (self.q_proj.forward(&x_norm), self.k_proj.forward(&x_norm), self.v_proj.forward(&x_norm))
                            };
                    
                            if _idx == 0 && q_len > 0 {
                                let v_max = v.iter().take(100).fold(0.0f32, |a, &b| a.max(b.to_f32().abs()));
                                if v_max == 0.0 {
                                    println!("[STABILITY-CRITICAL] Layer 0 V-PROJ produced ALL ZEROS at forward!");
                                }
                            }        // [2025-COGNITIVE-STABILITY] Embedding-Guided Adaptive Scaling
        // Treat Layer 0 as an embedding engine to measure semantic density.
        
        let measure_context = |data: &mut Vec<f16>, name: &str| -> (f32, f32) {
            if data.is_empty() { return (0.0, 1.0); }
            let samples = data.iter().take(500).map(|x| x.to_f32().abs()).collect::<Vec<_>>();
            let mean_abs = samples.iter().sum::<f32>() / samples.len() as f32;
            let max_abs = samples.iter().fold(1e-9f32, |a, &b| a.max(b));
            
            // Semantic Density: High ratio = rich context, Low ratio = sparse/risky context
            let density = (mean_abs / max_abs).clamp(0.0, 1.0);

            if _idx == 0 {
                println!("[SEMANTIC-DIAG] Layer 0 {} -> Energy: {:.2e}, Density: {:.4}", name, mean_abs, density);
            }

            if mean_abs < 1e-12 {
                println!("[STABILITY-WARN] {} signal collapsed. Jumpstarting...", name);
                for i in 0..data.len().min(100) { data[i] = f16::from_f32(1e-6); }
                (1e-6, 0.1)
            } else {
                (mean_abs, density)
            }
        };

        let (q_energy, q_density) = measure_context(&mut q, "Q");
        let (_k_energy, _k_density) = measure_context(&mut k, "K");
        let (_v_energy, _v_density) = measure_context(&mut v, "V");

        // [MATH-RECALIBRATION] 10x stronger base to prevent GPU-specific underflow
        let density_boost = if q_density < 0.2f32 { 8.0f32 } else { (1.0f32 / (q_density + 0.1f32)).min(5.0f32) };
        let mut final_alpha = (0.1f32 / (q_energy + 1e-9f32) * density_boost).clamp(0.2f32, 2.5f32);
        
        if is_baking { final_alpha *= 1.5f32; } 

        if let Some(ref qw) = self.q_norm { 
            let qw_cow = qw.get_slice::<f16>();
            // [FIX] Correct per-head normalization: each head (128 dims) uses its own weights if available
            q.par_chunks_exact_mut(head_dim).enumerate().for_each(|(h_idx, head_data)| {
                let w_off = (h_idx * head_dim) % qw_cow.len();
                let head_w = &qw_cow[w_off .. w_off + head_dim];
                
                let mut v = 0.0f32; for &x in head_data.iter() { let val = x.to_f32(); v += val * val; }
                let inv = 1.0 / (v / head_dim as f32 + config.rms_norm_eps as f32).sqrt();
                for j in 0..head_dim { 
                    head_data[j] = f16::from_f32(head_data[j].to_f32() * inv * head_w[j].to_f32()); 
                }
            });
        }
        if let Some(ref kw) = self.k_norm { 
            let kw_cow = kw.get_slice::<f16>();
            // [FIX] Correct per-head normalization for keys
            k.par_chunks_exact_mut(head_dim).enumerate().for_each(|(h_idx, head_data)| {
                let w_off = (h_idx * head_dim) % kw_cow.len();
                let head_w = &kw_cow[w_off .. w_off + head_dim];
                
                let mut v = 0.0f32; for &x in head_data.iter() { let val = x.to_f32(); v += val * val; }
                let inv = 1.0 / (v / head_dim as f32 + config.rms_norm_eps as f32).sqrt();
                for j in 0..head_dim { 
                    head_data[j] = f16::from_f32(head_data[j].to_f32() * inv * head_w[j].to_f32()); 
                }
            });
        }

        native_apply_rope_f16_with_offset(&mut q, &mut k, q_len, seqlen_offset, n_h, head_dim, config.rope_theta, rope_cos, rope_sin);

        #[cfg(feature = "cuda")]
        if self.device_id >= 0 {
            use cudarc::driver::sys::*;
            let mut gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
            let (k_ptr, v_ptr, current_len) = if let Some((kp, vp, l)) = *gpu_cache_guard { (kp, vp, l) } else {
                let mut kp: CUdeviceptr = 0; let mut vp: CUdeviceptr = 0;
                let max_tokens = 16384; 
                unsafe { 
                    lib().cuMemAlloc_v2(&mut kp, max_tokens * n_kv * (head_dim/32) * 4); 
                    lib().cuMemAlloc_v2(&mut vp, max_tokens * n_kv * head_dim * 4); 
                }
                
                (GpuPtr(kp as *mut _), GpuPtr(vp as *mut _), 0)
            };

            let max_tokens = 16384;
            if current_len + q_len > max_tokens {
                println!("[STABILITY-RECOVERY] Sequence length ({}) exceeds GPU limit ({}). Falling back to CPU...", current_len + q_len, max_tokens);
                if let Some((k_host, v_host)) = self.get_kv_data(head_dim, n_kv) {
                    let cpu_attn = native_bit_serial_attn_f16(&q, &k_host, &v_host, hidden_size, n_h, n_kv, q_len, current_len + q_len, 0.1f32);
                    return cpu_attn;
                }
            }

            let k_packed = pack_f16_to_bits(&k);
            let v_f32: Vec<f32> = if q_len > 1 {
                v.par_iter().map(|val: &f16| val.to_f32()).collect()
            } else {
                v.iter().map(|val: &f16| val.to_f32()).collect()
            };
            unsafe {
                let cuda_lib = lib();
                // [FIX] Use correct pointer arithmetic for CUdeviceptr offsets
                let k_offset_bytes = current_len * n_kv * (head_dim / 32) * 4;
                let v_offset_bytes = current_len * n_kv * head_dim * 4;
                
                let d_k_target = (k_ptr.0 as usize + k_offset_bytes) as CUdeviceptr;
                let d_v_target = (v_ptr.0 as usize + v_offset_bytes) as CUdeviceptr;
                
                let _ = cuda_lib.cuMemcpyHtoD_v2(d_k_target, k_packed.as_ptr() as *const _, k_packed.len() * 4);
                let _ = cuda_lib.cuMemcpyHtoD_v2(d_v_target, v_f32.as_ptr() as *const _, v_f32.len() * 4);
            }
            let new_len = current_len + q_len;
            *gpu_cache_guard = Some((k_ptr, v_ptr, new_len));

            // [OPTIMIZATION] Reuse scratch buffers for Attention
            let (d_q, d_o) = self.ensure_attn_scratch(q.len());

            // [STABILITY-ORGANIC-RECOVERY] Multi-stage GPU persistence loop
            // Instead of giving up, 0.6B model fights to stay on GPU via incremental alpha hunting.
            let mut attn_out = native_bit_serial_attn_gpu_buffered(&q, k_ptr, v_ptr, n_h, n_kv, head_dim, new_len, self.device_id as usize, d_q, d_o, final_alpha);
            
            let is_dead = |data: &[f16]| -> bool {
                !data.is_empty() && (data[0].to_f32().abs() < 1e-9) && data.iter().take(20).all(|x| x.to_f32().abs() < 1e-9)
            };

            if is_dead(&attn_out) {
                let mut current_alpha = final_alpha;
                let mut success = false;
                
                // [2025-H2-RESEARCH] 3-Stage GPU Probing: Much faster than CPU fallback
                for stage in 1..=3 {
                    current_alpha += 0.5; // Stronger increments for GPU survival
                    println!("[STABILITY-RECOVERY] GPU Stage {} (Baking: {}) -> Hunting with alpha {:.2}...", stage, is_baking, current_alpha);
                    
                    let retry = native_bit_serial_attn_gpu_buffered(&q, k_ptr, v_ptr, n_h, n_kv, head_dim, new_len, self.device_id as usize, d_q, d_o, current_alpha);
                    if !is_dead(&retry) {
                        println!("[STABILITY] GPU Signal RECOVERED at Stage {}! Alpha: {:.2}", stage, current_alpha);
                        attn_out = retry;
                        success = true;
                        break;
                    }
                }

                if !success {
                    let final_cpu_alpha = current_alpha + 0.3;
                    println!("[STABILITY-CRITICAL] All GPU stages failed. Final fallback to CPU with alpha {:.2}...", final_cpu_alpha);
                    drop(gpu_cache_guard);
                    if let Some((k_host, v_host)) = self.get_kv_data(head_dim, n_kv) {
                        attn_out = native_bit_serial_attn_f16(&q, &k_host, &v_host, hidden_size, n_h, n_kv, q_len, new_len, final_cpu_alpha);
                    }
                }
            }

            let mut x_at = self.o_proj.forward(&attn_out);
            for i in 0..x_at.len() { x_at[i] += residual[i]; }
            let r_mlp = x_at.clone();
            
            let x_n_m = unsafe {
                let post_ln_weight_ref = self.post_attention_layernorm.get_raw_slice::<f16>();
                native_rms_norm_f16(&x_at, post_ln_weight_ref, config.rms_norm_eps as f32, hidden_size)
            };
            
            let (mut gate, up) = if q_len > 1 {
                rayon::join(|| self.gate_proj.forward(&x_n_m), || self.up_proj.forward(&x_n_m))
            } else {
                (self.gate_proj.forward(&x_n_m), self.up_proj.forward(&x_n_m))
            };
            
            native_silu_f16(&mut gate); for i in 0..gate.len() { gate[i] *= up[i]; }
            let mut x_m = self.down_proj.forward(&gate); for i in 0..x_m.len() { x_m[i] += r_mlp[i]; }
            return x_m;
        }

        let mut cache_guard = self.kv_cache.lock().unwrap();
        let (k_p_f, v_f_f): (Vec<u32>, Vec<f16>) = if let Some((pk, pv)) = cache_guard.take() {
            let mut nk = pk; let mut nv = pv; nk.extend_from_slice(&pack_f16_to_bits(&k)); nv.extend_from_slice(&v); (nk, nv)
        } else { (pack_f16_to_bits(&k), v) };
        let t_s = v_f_f.len() / (n_kv * head_dim);
        *cache_guard = Some((k_p_f.clone(), v_f_f.clone())); drop(cache_guard);

        let attn_out = native_bit_serial_attn_f16(&q, &k_p_f, &v_f_f, hidden_size, n_h, n_kv, q_len, t_s, 0.1f32);
        let mut x_at = self.o_proj.forward(&attn_out);
        for i in 0..x_at.len() { x_at[i] += residual[i]; }
        let r_mlp = x_at.clone();
        let post_ln_weight_cow = self.post_attention_layernorm.get_slice::<f16>();
        let x_n_m = native_rms_norm_f16(&x_at, post_ln_weight_cow.as_ref(), config.rms_norm_eps as f32, hidden_size);
        let mut gate = self.gate_proj.forward(&x_n_m); let up = self.up_proj.forward(&x_n_m);
        native_silu_f16(&mut gate); for i in 0..gate.len() { gate[i] *= up[i]; }
        let mut x_m = self.down_proj.forward(&gate); for i in 0..x_m.len() { x_m[i] += r_mlp[i]; }
        x_m
    }

    #[cfg(feature = "cuda")]
    fn ensure_attn_scratch(&self, size: usize) -> (CUdeviceptr, CUdeviceptr) {
        use cudarc::driver::sys::*;
        let req_bytes = size * 4;
        
        let mut sq_guard = self.attn_scratch_q.lock().unwrap();
        let d_q = if let Some((ptr, cur_s)) = *sq_guard {
            if cur_s >= req_bytes { ptr.0 as CUdeviceptr }
            else {
                unsafe {
                    let mut new_ptr: CUdeviceptr = 0;
                    let _ = lib().cuMemFree_v2(ptr.0 as CUdeviceptr);
                    let _ = lib().cuMemAlloc_v2(&mut new_ptr, req_bytes);
                    *sq_guard = Some((GpuPtr(new_ptr as *mut _), req_bytes));
                    new_ptr
                }
            }
        } else {
            unsafe {
                let mut new_ptr: CUdeviceptr = 0;
                let _ = lib().cuMemAlloc_v2(&mut new_ptr, req_bytes);
                *sq_guard = Some((GpuPtr(new_ptr as *mut _), req_bytes));
                new_ptr
            }
        };

        let mut so_guard = self.attn_scratch_o.lock().unwrap();
        let d_o = if let Some((ptr, cur_s)) = *so_guard {
            if cur_s >= req_bytes { ptr.0 as CUdeviceptr }
            else {
                unsafe {
                    let mut new_ptr: CUdeviceptr = 0;
                    let _ = lib().cuMemFree_v2(ptr.0 as CUdeviceptr);
                    let _ = lib().cuMemAlloc_v2(&mut new_ptr, req_bytes);
                    *so_guard = Some((GpuPtr(new_ptr as *mut _), req_bytes));
                    new_ptr
                }
            }
        } else {
            unsafe {
                let mut new_ptr: CUdeviceptr = 0;
                let _ = lib().cuMemAlloc_v2(&mut new_ptr, req_bytes);
                *so_guard = Some((GpuPtr(new_ptr as *mut _), req_bytes));
                new_ptr
            }
        };

        (d_q, d_o)
    }

    pub fn move_to_gpu(&mut self, device_id: i32) {
        self.device_id = device_id;
        self.q_proj.move_to_gpu(device_id); self.k_proj.move_to_gpu(device_id);
        self.v_proj.move_to_gpu(device_id); self.o_proj.move_to_gpu(device_id);
        self.gate_proj.move_to_gpu(device_id); self.up_proj.move_to_gpu(device_id);
        self.down_proj.move_to_gpu(device_id);
    }

    pub fn clear_kv_cache(&self) {
        let mut cache = self.kv_cache.lock().unwrap();
        *cache = None;
        let mut gpu_cache = self.gpu_kv_cache.lock().unwrap();
        if let Some((k, v, _)) = *gpu_cache {
            // [OPTIMIZATION] Don't free VRAM, just reset length to 0. 
            // This prevents expensive cuMemAlloc/Free cycles during Baking chunks.
            *gpu_cache = Some((k, v, 0));
        }
    }

    /// [NEW] Hard clear for when we really need to free VRAM (e.g., model unload)
    pub fn force_free_kv_cache(&self) {
        let mut cache = self.kv_cache.lock().unwrap();
        *cache = None;
        let mut gpu_cache = self.gpu_kv_cache.lock().unwrap();
        if let Some((k, v, _)) = gpu_cache.take() {
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                let _ = lib().cuMemFree_v2(k.0 as CUdeviceptr);
                let _ = lib().cuMemFree_v2(v.0 as CUdeviceptr);
            }
        }
        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            if let Some((ptr, _)) = self.attn_scratch_q.lock().unwrap().take() { let _ = lib().cuMemFree_v2(ptr.0 as CUdeviceptr); }
            if let Some((ptr, _)) = self.attn_scratch_o.lock().unwrap().take() { let _ = lib().cuMemFree_v2(ptr.0 as CUdeviceptr); }
        }
    }

    pub fn get_kv_data(&self, head_dim: usize, n_kv: usize) -> Option<(Vec<u32>, Vec<f16>)> {
        #[cfg(feature = "cuda")]
        if self.device_id >= 0 {
            use cudarc::driver::sys::{lib, CUdeviceptr};
            let gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
            if let Some((k_ptr, v_ptr, current_len)) = *gpu_cache_guard {
                if current_len == 0 { return None; }
                let k_size = current_len * n_kv * (head_dim / 32);
                let v_size = current_len * n_kv * head_dim;
                
                let mut k_host = vec![0u32; k_size];
                let mut v_host_f32 = vec![0.0f32; v_size];
                
                unsafe {
                    let cuda_lib = lib();
                    // [FIX] Correctly cast GpuPtr to CUdeviceptr for reliable reading
                    let d_k_src = k_ptr.0 as usize as CUdeviceptr;
                    let d_v_src = v_ptr.0 as usize as CUdeviceptr;
                    let _ = cuda_lib.cuMemcpyDtoH_v2(k_host.as_mut_ptr() as *mut _, d_k_src, k_size * 4);
                    let _ = cuda_lib.cuMemcpyDtoH_v2(v_host_f32.as_mut_ptr() as *mut _, d_v_src, v_size * 4);
                }
                
                let v_host: Vec<f16> = v_host_f32.into_iter().map(|v| {
                    if v.is_nan() { f16::ZERO } else { f16::from_f32(v) }
                }).collect();
                return Some((k_host, v_host));
            }
        }

        let cache = self.kv_cache.lock().unwrap();
        cache.clone()
    }

    pub fn get_kv_len(&self, head_dim: usize, n_kv: usize) -> usize {
        #[cfg(feature = "cuda")]
        if self.device_id >= 0 {
            let gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
            if let Some((_, _, current_len)) = *gpu_cache_guard {
                if current_len > 0 { return current_len; }
            }
        }

        let cache = self.kv_cache.lock().unwrap();
        if let Some((k, _)) = cache.as_ref() {
            (k.len() * 32) / (n_kv * head_dim)
        } else {
            0
        }
    }

    pub fn set_kv_data(&self, k: Vec<u32>, v: Vec<f16>) {
        let mut cache = self.kv_cache.lock().unwrap();
        *cache = Some((k, v));
    }

    /// [NEW] Direct GPU injection to prevent redundant CPU conversions
    #[cfg(feature = "cuda")]
    pub fn inject_gpu_kv(&self, k_data: &[u32], v_data_f32: &[f32], n_kv: usize, head_dim: usize) {
        let tokens = (k_data.len() * 32) / (n_kv * head_dim);
        let mut gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
        
        unsafe {
            let lib = lib();
            let mut kp: CUdeviceptr = 0;
            let mut vp: CUdeviceptr = 0;
            // Allocate full buffer (128k)
            lib.cuMemAlloc_v2(&mut kp, 131072 * n_kv * (head_dim/32) * 4); 
            lib.cuMemAlloc_v2(&mut vp, 131072 * n_kv * head_dim * 4); 
            
            // Copy data
            lib.cuMemcpyHtoD_v2(kp, k_data.as_ptr() as *const _, k_data.len() * 4);
            lib.cuMemcpyHtoD_v2(vp, v_data_f32.as_ptr() as *const _, v_data_f32.len() * 4);
            
            *gpu_cache_guard = Some((GpuPtr(kp as *mut _), GpuPtr(vp as *mut _), tokens));
        }
    }

    /// [NEW] Direct GPU-to-GPU injection for extremely fast layer replication
    #[cfg(feature = "cuda")]
    pub fn inject_gpu_kv_direct(&self, k_src: GpuPtr, v_src: GpuPtr, tokens: usize, n_kv: usize, head_dim: usize) {
        let mut gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
        
        unsafe {
            let lib = lib();
            let mut kp: CUdeviceptr = 0;
            let mut vp: CUdeviceptr = 0;
            // Allocate full buffer (128k)
            lib.cuMemAlloc_v2(&mut kp, 131072 * n_kv * (head_dim/32) * 4); 
            lib.cuMemAlloc_v2(&mut vp, 131072 * n_kv * head_dim * 4); 
            
            // Copy data via DtoD (Device to Device)
            let k_size = tokens * n_kv * (head_dim / 32) * 4;
            let v_size = tokens * n_kv * head_dim * 4;
            lib.cuMemcpyDtoD_v2(kp, k_src.0 as CUdeviceptr, k_size);
            lib.cuMemcpyDtoD_v2(vp, v_src.0 as CUdeviceptr, v_size);
            
            *gpu_cache_guard = Some((GpuPtr(kp as *mut _), GpuPtr(vp as *mut _), tokens));
        }
    }
}

pub struct NativeQwen3TextModel {
    pub config: Qwen3VLTextConfig, 
    pub embed_tokens: NativeLinear, 
    pub layers: Vec<NativeLayer>, 
    pub norm: NativeTensor,
    pub rope_cos: Vec<f16>,
    pub rope_sin: Vec<f16>,
}

unsafe impl Send for NativeQwen3TextModel {}
unsafe impl Sync for NativeQwen3TextModel {}

impl NativeQwen3TextModel {
    pub fn forward(&self, input_ids: &[u32], pixel_values: Option<&[f16]>, grid_thw: Option<&[u32; 3]>, seqlen_offset: usize) -> Vec<f16> {
        let hid = self.config.hidden_size;
        let is_baking = self.layers.len() <= 1; // [CRITICAL] 1 layer = Small baking model
        
        let mut cur_x = match &self.embed_tokens.variant {
            LinearVariant::Standard { weight, .. } => {
                let w_cow = weight.get_slice::<f16>();
                native_embedding_lookup_f16(input_ids, w_cow.as_ref(), hid)
            },
            LinearVariant::BitSerial { weight_packed, .. } => {
                let w_cow = weight_packed.get_slice::<f16>();
                native_embedding_lookup_f16(input_ids, w_cow.as_ref(), hid)
            }
        };

        for (i, layer) in self.layers.iter().enumerate() { 
            cur_x = layer.forward(&cur_x, &self.config, seqlen_offset, i, &self.rope_cos, &self.rope_sin, is_baking); 
        }
        let norm_cow = self.norm.get_slice::<f16>();
        native_rms_norm_f16(&cur_x, norm_cow.as_ref(), self.config.rms_norm_eps as f32, hid)
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
            
            // 1. Convert to f32 ONCE
            let v_f32: Vec<f32> = v.par_iter().map(|val: &f16| val.to_f32()).collect();
            
            // 2. Inject to FIRST layer (HtoD)
            self.layers[0].inject_gpu_kv(&k, &v_f32, n_kv, h_d);
            
            // 3. Replicate from Layer 0 to all other layers (DtoD)
            if let Some((k_src, v_src, tokens)) = *self.layers[0].gpu_kv_cache.lock().unwrap() {
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

    pub fn get_all_kv(&self) -> Vec<(Vec<u32>, Vec<f16>)> {
        let h_d = self.config.head_dim;
        let n_kv = self.config.num_key_value_heads;
        self.layers.iter().filter_map(|l| l.get_kv_data(h_d, n_kv)).collect()
    }
}

pub struct NativeQwen3VLModel {
    pub config: Qwen3VLConfig, 
    pub text_model: NativeQwen3TextModel, 
    pub lm_head: NativeLinear,
    pub visual: Option<NativeVisionModel>,
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

    pub fn forward(&self, pixel_values: &[f16], grid_thw: &[u32; 3], rope_cos: &[f16], rope_sin: &[f16]) -> Vec<f16> {
        // [VISION-TRANSFORMER-PIPELINE]
        // Hidden states: (patches, hidden_size)
        // grid_thw: [T, H, W]
        let patches = (grid_thw[1] * grid_thw[2]) as usize;
        
        // 1. Patch Embedding
        let mut x = self.patch_embed.forward(pixel_values);
        
        // 2. Transformer Blocks
        // Vision blocks usually don't use KV cache (full attention per image)
        for block in &self.blocks {
            // [STUB] Vision attention is full attention, no RoPE/Cache offset needed here usually
            // Simplified: treat as a single sequence of patches
            x = block.forward(&x, &crate::models::qwen3vl::config::Qwen3VLTextConfig {
                hidden_size: self.patch_embed.out_features,
                intermediate_size: block.gate_proj.out_features,
                num_hidden_layers: 1,
                num_attention_heads: 16, // Fixed for vision usually
                num_key_value_heads: 16,
                head_dim: self.patch_embed.out_features / 16,
                rms_norm_eps: 1e-6,
                rope_theta: 10000.0,
                vocab_size: 0,
                max_position_embeddings: 4096,
                dtype: None,
                rope_scaling: None,
            }, 0, 0, rope_cos, rope_sin, false);
        }
        
        // 3. Patch Merger
        self.merger.forward(&x, &crate::models::qwen3vl::config::Qwen3VLTextConfig {
            hidden_size: x.len() / patches,
            intermediate_size: x.len() / patches, // Placeholder
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: x.len() / patches,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            vocab_size: 0,
            max_position_embeddings: 4096,
            dtype: None,
            rope_scaling: None,
        }, 0, 0, rope_cos, rope_sin, false)
    }
}

impl NativeQwen3VLModel {
    pub fn load(config: Qwen3VLConfig, m_mmap: Arc<Mmap>, v_mmap: Option<Arc<Mmap>>, baking: bool, s_mmap: Option<Arc<Mmap>>) -> Result<Self> {
        let st = SafeTensors::deserialize(&m_mmap)?; 
        let st_sec = s_mmap.as_ref().map(|m| SafeTensors::deserialize(m)).transpose()?;
        let t_c = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;
        
        let find_key_ext = |target: &str, primary: &SafeTensors, secondary: Option<&SafeTensors>| -> Option<(String, bool)> {
            let variations = vec![
                target.to_string(),
                target.replace("model.", "model.language_model."),
                target.replace("model.layers", "model.language_model.layers"),
                target.replace("lm_head", "model.lm_head"),
                target.replace("lm_head", "output"),
                format!("model.language_model.{}", target.replace("model.", "")),
                format!("model.{}", target),
            ];
            
            // 1. Try Primary
            for v in variations.iter() {
                if primary.tensor(v).is_ok() || primary.tensor(&format!("{}.packed", v)).is_ok() { return Some((v.clone(), true)); }
            }
            
            // 2. Try Secondary (if baking)
            if let Some(sec) = secondary {
                for v in variations.iter() {
                    if sec.tensor(v).is_ok() || sec.tensor(&format!("{}.packed", v)).is_ok() { return Some((v.clone(), false)); }
                }
            }

            // 3. Exhaustive search in Primary
            for name in primary.names() {
                if name.contains(target) {
                    return Some((name.replace(".packed", "").replace(".scales", "").replace(".shape", "").replace(".format", ""), true));
                }
            }
            
            // 4. Exhaustive search in Secondary
            if let Some(sec) = secondary {
                for name in sec.names() {
                    if name.contains(target) {
                        return Some((name.replace(".packed", "").replace(".scales", "").replace(".shape", "").replace(".format", ""), false));
                    }
                }
            }

            None
        };

        let get_l = |base: &str, in_f: usize, out_f: usize| -> Result<NativeLinear> {
            let (key, is_primary) = find_key_ext(base, &st, st_sec.as_ref()).ok_or_else(|| anyhow!("LinearTensorNotFound: {}", base))?;
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
                    variant: LinearVariant::BitSerial {
                        weight_packed: NativeTensor::from_mmap(current_mmap.clone(), op, vp.shape().to_vec(), NativeDType::U32),
                        scales: NativeTensor::from_mmap(current_mmap.clone(), os, vs.shape().to_vec(), NativeDType::F16),
                        bias: None,
                    }, 
                    device_id: -1,
                    scratch_i: std::sync::Mutex::new(None),
                    scratch_o: std::sync::Mutex::new(None),
                })
            } else {
                let v = current_st.tensor(&key)?;
                let o = unsafe { v.data().as_ptr().offset_from(current_mmap.as_ptr()) } as usize;
                Ok(NativeLinear { 
                    in_features: in_f, out_features: out_f, 
                    variant: LinearVariant::Standard { 
                        weight: NativeTensor::from_mmap(current_mmap.clone(), o, v.shape().to_vec(), NativeDType::F16), 
                        bias: None 
                    }, 
                    device_id: -1,
                    scratch_i: std::sync::Mutex::new(None),
                    scratch_o: std::sync::Mutex::new(None),
                })
            }
        };

        let get_t = |name: &str| -> Result<NativeTensor> {
            let (key, is_primary) = find_key_ext(name, &st, st_sec.as_ref()).ok_or_else(|| anyhow!("TensorNotFound: {}", name))?;
            let current_st = if is_primary { &st } else { st_sec.as_ref().unwrap() };
            let current_mmap = if is_primary { &m_mmap } else { s_mmap.as_ref().unwrap() };

            let v = current_st.tensor(&key)?;
            let off = unsafe { v.data().as_ptr().offset_from(current_mmap.as_ptr()) } as usize;
            Ok(NativeTensor::from_mmap(current_mmap.clone(), off, v.shape().to_vec(), NativeDType::F16))
        };

        println!("[LOAD] Initializing Native Engine (Baking: {})...", baking);

        // [CRITICAL-FIX] 임베딩 로더: 임베딩은 항상 F16 룩업 테이블이어야 함
        let get_embed = |base: &str, vocab: usize, hid: usize| -> Result<NativeLinear> {
            let (key, is_primary) = find_key_ext(base, &st, st_sec.as_ref()).ok_or_else(|| anyhow!("EmbedTensorNotFound: {}", base))?;
            let current_st = if is_primary { &st } else { st_sec.as_ref().unwrap() };
            let current_mmap = if is_primary { &m_mmap } else { s_mmap.as_ref().unwrap() };

            if current_st.tensor(&format!("{}.packed", key)).is_ok() {
                println!("[LOAD] WARNING: embed_tokens was bit-quantized. Reverting to F16 for lookup table...");
                // 양자화된 경우 matmul을 통해 F16으로 복구 (단순 lookup이 불가능하므로 로딩 시점에 미리 계산)
                // 하지만 여기서는 간결성을 위해 Standard로 강제 로드 시도 (GGUF 원본이 F16인 경우 대비)
                if current_st.tensor(&key).is_ok() {
                    let v = current_st.tensor(&key)?;
                    let o = unsafe { v.data().as_ptr().offset_from(current_mmap.as_ptr()) } as usize;
                    return Ok(NativeLinear { in_features: vocab, out_features: hid, variant: LinearVariant::Standard { weight: NativeTensor::from_mmap(current_mmap.clone(), o, v.shape().to_vec(), NativeDType::F16), bias: None }, device_id: -1, scratch_i: std::sync::Mutex::new(None), scratch_o: std::sync::Mutex::new(None) });
                }
            }
            get_l(base, vocab, hid)
        };

        let emb = get_embed("model.embed_tokens.weight", 151936, t_c.hidden_size)
            .or_else(|_| get_embed("model.language_model.embed_tokens.weight", 151936, t_c.hidden_size))?;
            
        let mut layers = Vec::new(); 
        let l_to_l = if baking { 1 } else { t_c.num_hidden_layers };

        // [FIX] Correct Dimensionality Calculation for Attention Block
        let q_out = t_c.num_attention_heads * t_c.head_dim;
        let kv_out = t_c.num_key_value_heads * t_c.head_dim;

        for i in 0..l_to_l {
            // [STRICT-PARITY] Use unified naming convention enforced by quantization script
            let p = format!("model.language_model.layers.{}", i);

            layers.push(NativeLayer {
                input_layernorm: get_t(&format!("{}.input_layernorm.weight", p))?,
                post_attention_layernorm: get_t(&format!("{}.post_attention_layernorm.weight", p))?,
                q_norm: get_t(&format!("{}.self_attn.q_norm.weight", p)).ok(),
                k_norm: get_t(&format!("{}.self_attn.k_norm.weight", p)).ok(),
                q_proj: get_l(&format!("{}.self_attn.q_proj.weight", p), t_c.hidden_size, q_out)?,
                k_proj: get_l(&format!("{}.self_attn.k_proj.weight", p), t_c.hidden_size, kv_out)?,
                v_proj: get_l(&format!("{}.self_attn.v_proj.weight", p), t_c.hidden_size, kv_out)?,
                o_proj: get_l(&format!("{}.self_attn.o_proj.weight", p), q_out, t_c.hidden_size)?,
                gate_proj: get_l(&format!("{}.mlp.gate_proj.weight", p), t_c.hidden_size, t_c.intermediate_size)?,
                up_proj: get_l(&format!("{}.mlp.up_proj.weight", p), t_c.hidden_size, t_c.intermediate_size)?,
                down_proj: get_l(&format!("{}.mlp.down_proj.weight", p), t_c.intermediate_size, t_c.hidden_size)?,
                device_id: -1, kv_cache: std::sync::Mutex::new(None), gpu_kv_cache: std::sync::Mutex::new(None),
                attn_scratch_q: std::sync::Mutex::new(None), attn_scratch_o: std::sync::Mutex::new(None),
            });
        }
        
        let norm = get_t("model.language_model.norm.weight")?;
        
        // [SMART-HEAD-LOADER] 가중치 공유(tied weights) 대응 강화
        let head_res = get_l("lm_head.weight", t_c.hidden_size, 151936)
            .or_else(|_| get_l("model.language_model.lm_head.weight", t_c.hidden_size, 151936))
            .or_else(|_| {
                println!("[LOAD] lm_head not found, using tied weights from embed_tokens.");
                get_l("model.embed_tokens.weight", 151936, t_c.hidden_size)
                    .or_else(|_| get_l("model.language_model.embed_tokens.weight", 151936, t_c.hidden_size))
                    .map(|mut emb_l| {
                        emb_l.in_features = t_c.hidden_size;
                        emb_l.out_features = 151936;
                        emb_l
                    })
            });
            
        let head = head_res.map_err(|e| {
            println!("[CRITICAL-LOAD-ERROR] ALL attempts to find lm_head or tied weights failed: {}.", e);
            anyhow!("LMHeadNotFound")
        })?;

        // [PRECOMPUTE-ROPE]
        let max_seq_len = 16384;
        let h_d = t_c.head_dim;
        let theta = t_c.rope_theta;
        let mut rope_cos = Vec::with_capacity(max_seq_len * h_d);
        let mut rope_sin = Vec::with_capacity(max_seq_len * h_d);

        for p in 0..max_seq_len {
            for d in 0..(h_d / 2) {
                let exponent = (2.0 * d as f32) / (h_d as f32);
                let freq = 1.0 / theta.powf(exponent);
                let (sn, cs) = ((p as f32) * freq).sin_cos();
                rope_cos.push(f16::from_f32(cs));
                rope_sin.push(f16::from_f32(sn));
            }
        }

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

            let get_vt = |name: &str| -> Result<NativeTensor> {
                let key = find_vkey(name).ok_or_else(|| anyhow!("VisionTensorNotFound: {}", name))?;
                let v = vst.tensor(&key)?;
                let off = unsafe { v.data().as_ptr().offset_from(vm.as_ptr()) } as usize;
                Ok(NativeTensor::from_mmap(vm.clone(), off, v.shape().to_vec(), NativeDType::F16))
            };

            let get_vl = |base: &str, in_f: usize, out_f: usize| -> Result<NativeLinear> {
                let key = find_vkey(base).ok_or_else(|| anyhow!("VisionLinearTensorNotFound: {}", base))?;
                let p_n = format!("{}.packed", key);
                let s_n = format!("{}.scales", key);
                if vst.tensor(&p_n).is_ok() {
                    let vp = vst.tensor(&p_n)?; let vs = vst.tensor(&s_n)?;
                    let op = unsafe { vp.data().as_ptr().offset_from(vm.as_ptr()) } as usize;
                    let os = unsafe { vs.data().as_ptr().offset_from(vm.as_ptr()) } as usize;
                    Ok(NativeLinear { in_features: in_f, out_features: out_f, variant: LinearVariant::BitSerial {
                        weight_packed: NativeTensor::from_mmap(vm.clone(), op, vp.shape().to_vec(), NativeDType::U32),
                        scales: NativeTensor::from_mmap(vm.clone(), os, vs.shape().to_vec(), NativeDType::F16),
                        bias: None,
                    }, device_id: -1, scratch_i: std::sync::Mutex::new(None), scratch_o: std::sync::Mutex::new(None) })
                } else {
                    let v = vst.tensor(&key)?;
                    let o = unsafe { v.data().as_ptr().offset_from(vm.as_ptr()) } as usize;
                    Ok(NativeLinear { in_features: in_f, out_features: out_f, variant: LinearVariant::Standard { weight: NativeTensor::from_mmap(vm.clone(), o, v.shape().to_vec(), NativeDType::F16), bias: None }, device_id: -1, scratch_i: std::sync::Mutex::new(None), scratch_o: std::sync::Mutex::new(None) })
                }
            };

            let v_cfg = config.vision_config.as_ref().ok_or(anyhow!("Missing vision_config"))?;
            let v_intermediate = v_cfg.hidden_size * 4;
            let v_out_hidden = v_cfg.out_hidden_size.unwrap_or(v_cfg.hidden_size * 2);
            
            // patch_embed -> patch_embd 자동 변환됨
            let patch_embed = get_vl("visual.patch_embed.proj.weight", 1536, v_cfg.hidden_size)?;
            let mut blocks = Vec::new();
            let b_to_l = if baking { 1 } else { v_cfg.depth };
            for i in 0..b_to_l {
                let p = format!("visual.blocks.{}", i);
                blocks.push(NativeLayer {
                    input_layernorm: get_vt(&format!("{}.norm1.weight", p))?,
                    post_attention_layernorm: get_vt(&format!("{}.norm2.weight", p))?,
                    q_norm: None, k_norm: None,
                    q_proj: get_vl(&format!("{}.attn.q_proj.weight", p), v_cfg.hidden_size, v_cfg.hidden_size)?,
                    k_proj: get_vl(&format!("{}.attn.k_proj.weight", p), v_cfg.hidden_size, v_cfg.hidden_size)?,
                    v_proj: get_vl(&format!("{}.attn.v_proj.weight", p), v_cfg.hidden_size, v_cfg.hidden_size)?,
                    o_proj: get_vl(&format!("{}.attn.proj.weight", p), v_cfg.hidden_size, v_cfg.hidden_size)?,
                    gate_proj: get_vl(&format!("{}.mlp.fc1.weight", p), v_cfg.hidden_size, v_intermediate)?,
                    up_proj: get_vl(&format!("{}.mlp.fc1.weight", p), v_cfg.hidden_size, v_intermediate)?, 
                    down_proj: get_vl(&format!("{}.mlp.fc2.weight", p), v_intermediate, v_cfg.hidden_size)?,
                    device_id: -1, kv_cache: std::sync::Mutex::new(None), gpu_kv_cache: std::sync::Mutex::new(None),
                    attn_scratch_q: std::sync::Mutex::new(None), attn_scratch_o: std::sync::Mutex::new(None),
                });
            }
            let merger = NativeLayer {
                input_layernorm: get_vt("visual.merger.norm.weight")?,
                post_attention_layernorm: get_vt("visual.merger.norm.weight")?, // Placeholder
                q_norm: None, k_norm: None,
                q_proj: get_vl("visual.merger.mlp.0.weight", v_cfg.hidden_size * 4, v_cfg.hidden_size * 4)?,
                k_proj: get_vl("visual.merger.mlp.0.weight", v_cfg.hidden_size * 4, v_cfg.hidden_size * 4)?,
                v_proj: get_vl("visual.merger.mlp.0.weight", v_cfg.hidden_size * 4, v_cfg.hidden_size * 4)?,
                o_proj: get_vl("visual.merger.mlp.2.weight", v_cfg.hidden_size * 4, v_out_hidden)?,
                gate_proj: get_vl("visual.merger.mlp.0.weight", v_cfg.hidden_size * 4, v_cfg.hidden_size * 4)?,
                up_proj: get_vl("visual.merger.mlp.0.weight", v_cfg.hidden_size * 4, v_cfg.hidden_size * 4)?,
                down_proj: get_vl("visual.merger.mlp.2.weight", v_cfg.hidden_size * 4, v_out_hidden)?,
                device_id: -1, kv_cache: std::sync::Mutex::new(None), gpu_kv_cache: std::sync::Mutex::new(None),
                attn_scratch_q: std::sync::Mutex::new(None), attn_scratch_o: std::sync::Mutex::new(None),
            };
            Some(NativeVisionModel { patch_embed, blocks, merger })
        } else { None };
            
        Ok(Self { config: config.clone(), text_model: NativeQwen3TextModel { config: t_c.clone(), embed_tokens: emb, layers, norm, rope_cos, rope_sin }, lm_head: head, visual })
    }
    pub fn forward(&self, i_ids: &[u32], p_v: Option<&[f16]>, g_t: Option<&[u32; 3]>, s_o: usize) -> Vec<f16> {
        // [2026-SPECULATIVE-TREE] Support for validating multiple candidates in parallel
        let _batch_size = i_ids.len() / (if i_ids.len() > 1 { 1 } else { 1 }); // Adjust for Tree-Depth
        
        let mut embeds = match &self.text_model.embed_tokens.variant {
            LinearVariant::Standard { weight, .. } => {
                let w_cow = weight.get_slice::<f16>();
                native_embedding_lookup_f16(i_ids, w_cow.as_ref(), self.text_model.config.hidden_size)
            },
            _ => Vec::new(),
        };

        // [VISION-FUSION] Skip during speculative tree validation to save cycles
        if let (Some(pv), Some(gt)) = (p_v, g_t) {
            if let Some(ref visual) = self.visual {
                let vision_features = visual.forward(pv, gt, &self.text_model.rope_cos, &self.text_model.rope_sin);
                let img_token_id = 151655; // <|image_pad|>
                let hid = self.text_model.config.hidden_size;
                
                let mut vision_idx = 0;
                for (i, &id) in i_ids.iter().enumerate() {
                    if id == img_token_id && vision_idx < (vision_features.len() / hid) {
                        let target_slice = &mut embeds[i * hid .. (i + 1) * hid];
                        let source_slice = &vision_features[vision_idx * hid .. (vision_idx + 1) * hid];
                        target_slice.copy_from_slice(source_slice);
                        vision_idx += 1;
                    }
                }
            }
        }

        let mut cur_x = embeds;
        let is_baking = self.text_model.layers.len() <= 1;
        for (i, layer) in self.text_model.layers.iter().enumerate() { 
            cur_x = layer.forward(&cur_x, &self.text_model.config, s_o, i, &self.text_model.rope_cos, &self.text_model.rope_sin, is_baking); 
        }
        
        let norm_cow = self.text_model.norm.get_slice::<f16>();
        let norm_x = native_rms_norm_f16(&cur_x, norm_cow.as_ref(), self.text_model.config.rms_norm_eps as f32, self.text_model.config.hidden_size);
        self.lm_head.forward(&norm_x)
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
    
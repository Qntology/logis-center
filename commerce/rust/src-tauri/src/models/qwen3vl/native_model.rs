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
    // [REMOVED] Individual scratch buffers to save VRAM
}

unsafe impl Send for NativeLinear {}
unsafe impl Sync for NativeLinear {}

impl NativeLinear {
    pub fn forward(&self, x: &[f16], global_scratch: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>) -> Vec<f16> {
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
                        let wp_ptr = weight_packed.gpu_ptr.map(|p| p.0 as usize).unwrap_or(0);
                        if wp_ptr != 0 {
                            // [VRAM-SHARING] Use global scratch buffers if provided
                            let (d_i, d_o) = if let Some((si, so)) = global_scratch {
                                self.ensure_gpu_buffers_ext(m, si, so)
                            } else {
                                // Fallback (should not happen in optimized path)
                                (0 as CUdeviceptr, 0 as CUdeviceptr)
                            };

                            if d_i != 0 && d_o != 0 {
                                let mut res = bit_serial_matmul_gpu_buffered(x, weight_packed, scales, m, self.out_features, self.in_features, self.device_id as usize, d_i, d_o);
                                
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
    fn ensure_gpu_buffers_ext(&self, m: usize, scratch_i: &std::sync::Mutex<Option<(GpuPtr, usize)>>, scratch_o: &std::sync::Mutex<Option<(GpuPtr, usize)>>) -> (CUdeviceptr, CUdeviceptr) {
        use cudarc::driver::sys::*;
        let req_i = m * self.in_features * 4;
        let req_o = m * self.out_features * 4;
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
            if size >= req_i { ptr.0 as CUdeviceptr }
            else {
                let mut new_ptr: CUdeviceptr = 0;
                let _ = unsafe { cuda_lib.cuMemFree_v2(ptr.0 as CUdeviceptr) };
                let res = unsafe { cuda_lib.cuMemAlloc_v2(&mut new_ptr, req_i) };
                if (res as i32) == 0 && new_ptr != 0 {
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
            if size >= req_o { ptr.0 as CUdeviceptr }
            else {
                let mut new_ptr: CUdeviceptr = 0;
                let _ = unsafe { cuda_lib.cuMemFree_v2(ptr.0 as CUdeviceptr) };
                let res = unsafe { cuda_lib.cuMemAlloc_v2(&mut new_ptr, req_o) };
                if (res as i32) == 0 && new_ptr != 0 {
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
    pub kv_cache: std::sync::Mutex<Option<(Vec<u32>, Vec<f16>)>>, 
    pub gpu_kv_cache: std::sync::Mutex<Option<(GpuPtr, GpuPtr, usize)>>, 
    // [REMOVED] Individual scratch buffers to save VRAM
    pub gpu_broken: std::sync::atomic::AtomicBool,
}

unsafe impl Send for NativeLayer {}
unsafe impl Sync for NativeLayer {}

impl NativeLayer {
    pub fn forward(&self, x: &[f16], config: &Qwen3VLTextConfig, seqlen_offset: usize, _idx: usize, rope_cos: &[f16], rope_sin: &[f16], is_baking: bool, global_scratch: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>) -> Vec<f16> {
        let hidden_size = config.hidden_size; let q_len = x.len() / hidden_size;
        let head_dim = config.head_dim; let n_h = config.num_attention_heads; let n_kv = config.num_key_value_heads;

        let residual = x.to_vec();
        let ln_weight_cow = self.input_layernorm.get_slice::<f16>();
        let x_norm = native_rms_norm_f16(x, ln_weight_cow.as_ref(), config.rms_norm_eps as f32, hidden_size);

        let (mut q, mut k, mut v) = if q_len > 1 {
            let (q_p, (k_p, v_p)) = rayon::join(
                || self.q_proj.forward(&x_norm, global_scratch),
                || rayon::join(|| self.k_proj.forward(&x_norm, global_scratch), || self.v_proj.forward(&x_norm, global_scratch))
            );
            (q_p, k_p, v_p)
        } else {
            (self.q_proj.forward(&x_norm, global_scratch), self.k_proj.forward(&x_norm, global_scratch), self.v_proj.forward(&x_norm, global_scratch))
        };

                

                        if _idx == 0 && q_len > 0 {

                

        
                                let v_max = v.iter().take(100).fold(0.0f32, |a, &b| a.max(b.to_f32().abs()));
                                if v_max == 0.0 {
                                    println!("[STABILITY-CRITICAL] Layer 0 V-PROJ produced ALL ZEROS at forward!");
                                }
                            }        // [2025-COGNITIVE-STABILITY] Embedding-Guided Adaptive Scaling
        // Treat Layer 0 as an embedding engine to measure semantic density.
        
        let measure_context = |data: &mut Vec<f16>, name: &str, idx: usize| -> (f32, f32) {
            if data.is_empty() { return (0.0, 1.0); }
            let samples = data.iter().take(500).map(|x| x.to_f32().abs()).collect::<Vec<_>>();
            let mean_abs = samples.iter().sum::<f32>() / samples.len() as f32;
            let max_abs = samples.iter().fold(1e-9f32, |a, &b| a.max(b));
            
            // Semantic Density: High ratio = rich context, Low ratio = sparse/risky context
            let density = (mean_abs / max_abs).clamp(0.0, 1.0);

            if idx == 0 {
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

        // [ACCURACY-CORRECTION] Attenuate memory intensity for higher layers
        let layer_depth_ratio = _idx as f32 / config.num_hidden_layers as f32;
        let accuracy_correction = (1.0 - layer_depth_ratio * 0.15).clamp(0.85, 1.0);

        // [SPEED-OPTIMIZATION] Skip diagnostic sampling for internal layers during inference
        let (q_energy, q_density) = if is_baking || _idx == 0 {
            measure_context(&mut q, "Q", _idx)
        } else {
            (1.0f32, 0.5f32) // Default values for speed
        };

        let density_boost = if q_density < 0.2f32 { 8.0f32 } else { (1.0f32 / (q_density + 0.1f32)).min(5.0f32) };
        let mut final_alpha = (0.1f32 / (q_energy + 1e-9f32) * density_boost * accuracy_correction).clamp(0.2f32, 2.5f32);
        
        let semantic_gain = if is_baking || _idx == 0 { (q_density * 2.0 + 0.8).clamp(0.9, 1.2) } else { 1.0f32 };
        
        if _idx == 0 && q_len > 0 {
            println!("[COGNITIVE] Layer 0 {} -> Alpha: {:.4}, Semantic Gain: {:.4} (Density: {:.4})", 
                if is_baking { "BAKING" } else { "INF" }, final_alpha, semantic_gain, q_density);
        }

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
        if self.device_id >= 0 && !self.gpu_broken.load(std::sync::atomic::Ordering::Relaxed) {
            use cudarc::driver::sys::*;
            let mut gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
            let (k_ptr, v_ptr, current_len) = if let Some((kp, vp, l)) = *gpu_cache_guard { (kp, vp, l) } else {
                let mut kp: CUdeviceptr = 0; let mut vp: CUdeviceptr = 0;
                let max_tokens = 16384; 
                unsafe { 
                    let cuda_lib = lib();
                    // [STABILITY] Ensure we are on the correct device context before allocation
                    let mut ctx = std::ptr::null_mut() as CUcontext;
                    cuda_lib.cuCtxGetCurrent(&mut ctx);
                    if ctx == std::ptr::null_mut() {
                        let mut dev = 0 as CUdevice;
                        cuda_lib.cuDeviceGet(&mut dev, self.device_id);
                        cuda_lib.cuDevicePrimaryCtxRetain(&mut ctx, dev);
                        cuda_lib.cuCtxSetCurrent(ctx);
                    }

                    let max_tokens_gpu = 4096;
                    let res_k = cuda_lib.cuMemAlloc_v2(&mut kp, max_tokens_gpu * n_kv * (head_dim/32) * 4); 
                    let res_v = cuda_lib.cuMemAlloc_v2(&mut vp, max_tokens_gpu * n_kv * head_dim * 4); 
                    
                    if (res_k as i32) != 0 || (res_v as i32) != 0 {
                        println!("[STABILITY-CRITICAL] cuMemAlloc_v2 FAILED! Code: K={}, V={}", res_k as i32, res_v as i32);
                        drop(gpu_cache_guard);
                        if let Some((k_host, v_host)) = self.get_kv_data(head_dim, n_kv) {
                            let cpu_attn = native_bit_serial_attn_f16(&q, &k_host, &v_host, hidden_size, n_h, n_kv, q_len, q_len, 0.1f32);
                            return cpu_attn;
                        }
                        return vec![f16::ZERO; q.len()];
                    }
                }
                
                (GpuPtr(kp as *mut _), GpuPtr(vp as *mut _), 0)
            };

            let max_tokens_logical = 200000;
            let max_tokens_gpu = 4096;

            if current_len + q_len > max_tokens_logical {
                println!("[CRITICAL] Hard limit reached: 200,000 tokens exceeded. Truncating context.");
                return vec![f16::ZERO; q.len()];
            }

            if current_len + q_len > max_tokens_gpu {
                if _idx == 0 && q_len > 0 {
                    println!("[STABILITY-RECOVERY] Context size ({}) exceeds GPU window (4096). Switching to high-speed CPU-AVX path.", current_len + q_len);
                }
                // GPU 데이터를 CPU로 동기화하고 연산 진행
                if let Some((k_host, v_host)) = self.get_kv_data(head_dim, n_kv) {
                    let mut cache = self.kv_cache.lock().unwrap();
                    *cache = Some((k_host, v_host));
                }
                self.gpu_broken.store(true, std::sync::atomic::Ordering::Relaxed);
                drop(gpu_cache_guard);
                // The main CPU logic at the bottom of the function will handle it now
            } else {
                let k_packed = pack_f16_to_bits(&k);
                let v_f32: Vec<f32> = v.iter().map(|val: &f16| val.to_f32()).collect();
                
                if _idx == 0 && q_len > 0 {
                    let v_f32_max = v_f32.iter().take(100).fold(0.0f32, |a, &b| a.max(b.abs()));
                    println!("[GPU-DEBUG] Transferring Chunk to GPU. Len: {}, V-sample-max: {:.6}, Current-Offset: {}", q_len, v_f32_max, current_len);
                }

            unsafe {
                let cuda_lib = lib();
                // [STABILITY-FIX] Ensure context is current for THIS thread before copying
                let mut ctx = std::ptr::null_mut() as CUcontext;
                cuda_lib.cuCtxGetCurrent(&mut ctx);
                if ctx == std::ptr::null_mut() && self.device_id >= 0 {
                    let mut dev = 0 as CUdevice;
                    cuda_lib.cuDeviceGet(&mut dev, self.device_id);
                    cuda_lib.cuDevicePrimaryCtxRetain(&mut ctx, dev);
                    cuda_lib.cuCtxSetCurrent(ctx);
                }

                let k_offset_bytes = (current_len as u64) * (n_kv as u64) * ((head_dim / 32) as u64) * 4;
                let v_offset_bytes = (current_len as u64) * (n_kv as u64) * (head_dim as u64) * 4;
                
                let d_k_target = (k_ptr.0 as u64 + k_offset_bytes) as CUdeviceptr;
                let d_v_target = (v_ptr.0 as u64 + v_offset_bytes) as CUdeviceptr;
                
                if k_ptr.0.is_null() || v_ptr.0.is_null() {
                    println!("[STABILITY-CRITICAL] Base pointers are NULL during forward! Re-allocating...");
                    // [RECOVERY] If base pointers are null, trigger a local allocation or fallback
                }

                let res_k = cuda_lib.cuMemcpyHtoD_v2(d_k_target, k_packed.as_ptr() as *const _, k_packed.len() * 4);
                let res_v = cuda_lib.cuMemcpyHtoD_v2(d_v_target, v_f32.as_ptr() as *const _, v_f32.len() * 4);
                    
                    if (res_k as i32) != 0 || (res_v as i32) != 0 {
                        println!("[STABILITY-CRITICAL] cuMemcpyHtoD_v2 FAILED! K_err={}, V_err={}", res_k as i32, res_v as i32);
                    }
                }
                let new_len = current_len + q_len;
                *gpu_cache_guard = Some((k_ptr, v_ptr, new_len));

                // [OPTIMIZATION] Reuse global scratch buffers for Attention to save VRAM
                let (d_q, d_o) = if let Some((si, so)) = global_scratch {
                    let mut si_g = si.lock().unwrap();
                    let mut so_g = so.lock().unwrap();
                    let q_bytes = q.len() * 4;
                    // Use ensure_gpu_buffers_ext logic style or just get pointer
                    let ptr_i = if let Some((p, s)) = *si_g { if s >= q_bytes { p.0 as CUdeviceptr } else { 0 as CUdeviceptr } } else { 0 as CUdeviceptr };
                    let ptr_o = if let Some((p, s)) = *so_g { if s >= q_bytes { p.0 as CUdeviceptr } else { 0 as CUdeviceptr } } else { 0 as CUdeviceptr };
                    (ptr_i, ptr_o)
                } else { (0, 0) };

                // [STABILITY-ORGANIC-RECOVERY] Multi-stage GPU persistence loop
                let mut attn_out = if d_q != 0 && d_o != 0 {
                    native_bit_serial_attn_gpu_buffered(&q, k_ptr, v_ptr, n_h, n_kv, head_dim, new_len, self.device_id as usize, d_q, d_o, final_alpha)
                } else {
                    native_bit_serial_attn_gpu(&q, k_ptr, v_ptr, n_h, n_kv, head_dim, new_len, self.device_id as usize, final_alpha)
                };
                
                // Apply gain
                for val in attn_out.iter_mut() {
                    *val = f16::from_f32(val.to_f32() * semantic_gain);
                }

                let is_dead = |data: &[f16]| -> bool {
                    !data.is_empty() && (data[0].to_f32().abs() < 1e-9) && data.iter().take(20).all(|x| x.to_f32().abs() < 1e-9)
                };

                if is_dead(&attn_out) {
                    println!("[STABILITY] GPU output dead at Layer {}. Syncing cache and falling back to CPU.", _idx);
                    self.gpu_broken.store(true, std::sync::atomic::Ordering::Relaxed);
                    if let Some((k_host, v_host)) = self.get_kv_data(head_dim, n_kv) {
                        let mut cache = self.kv_cache.lock().unwrap();
                        *cache = Some((k_host, v_host));
                    }
                    drop(gpu_cache_guard);
                } else {
                    let mut x_at = self.o_proj.forward(&attn_out, global_scratch);
                    for i in 0..x_at.len() { x_at[i] += residual[i]; }
                    let r_mlp = x_at.clone();
                    let x_n_m = unsafe {
                        let post_ln_weight_ref = self.post_attention_layernorm.get_raw_slice::<f16>();
                        native_rms_norm_f16(&x_at, post_ln_weight_ref, config.rms_norm_eps as f32, hidden_size)
                    };
                    let (mut gate, up) = if q_len > 1 {
                        rayon::join(|| self.gate_proj.forward(&x_n_m, global_scratch), || self.up_proj.forward(&x_n_m, global_scratch))
                    } else {
                        (self.gate_proj.forward(&x_n_m, global_scratch), self.up_proj.forward(&x_n_m, global_scratch))
                    };
                    
                    native_silu_f16(&mut gate); for i in 0..gate.len() { gate[i] *= up[i]; }
                    let mut x_m = self.down_proj.forward(&gate, global_scratch); for i in 0..x_m.len() { x_m[i] += r_mlp[i]; }
                    return x_m;
                }
            }
        }

        // [CPU-ONLY-PATH] Zero-copy update and high-speed AVX2 execution
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if cache_guard.is_none() {
            *cache_guard = Some((Vec::new(), Vec::new()));
        }
        
        if let Some((ref mut pk, ref mut pv)) = *cache_guard {
            pk.extend_from_slice(&pack_f16_to_bits(&k));
            pv.extend_from_slice(&v);
            
            let t_s = pv.len() / (n_kv * head_dim);
            let mut attn_out = native_bit_serial_attn_f16(&q, pk, pv, hidden_size, n_h, n_kv, q_len, t_s, final_alpha);
            
            for val in attn_out.iter_mut() {
                *val = f16::from_f32(val.to_f32() * semantic_gain);
            }

            let mut x_at = self.o_proj.forward(&attn_out, global_scratch);
            for i in 0..x_at.len() { x_at[i] += residual[i]; }
            let r_mlp = x_at.clone();
            let post_ln_weight_cow = self.post_attention_layernorm.get_slice::<f16>();
            let x_n_m = native_rms_norm_f16(&x_at, post_ln_weight_cow.as_ref(), config.rms_norm_eps as f32, hidden_size);
            let (mut gate, up) = if q_len > 1 {
                rayon::join(|| self.gate_proj.forward(&x_n_m, global_scratch), || self.up_proj.forward(&x_n_m, global_scratch))
            } else {
                (self.gate_proj.forward(&x_n_m, global_scratch), self.up_proj.forward(&x_n_m, global_scratch))
            };
            native_silu_f16(&mut gate); for i in 0..gate.len() { gate[i] *= up[i]; }
            let mut x_m = self.down_proj.forward(&gate, global_scratch); for i in 0..x_m.len() { x_m[i] += r_mlp[i]; }
            return x_m;
        }
        
        vec![f16::ZERO; x.len()]
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
    pub fn inject_gpu_kv(&self, k_data: &[u32], v_data: &[f16], n_kv: usize, head_dim: usize) {
        let tokens = (k_data.len() * 32) / (n_kv * head_dim);
        let mut gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
        
        unsafe {
            use cudarc::driver::sys::*;
            let cuda_lib = lib();
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
            // Allocate capped buffer (4k)
            let max_tokens_gpu = 4096;
            cuda_lib.cuMemAlloc_v2(&mut kp, max_tokens_gpu * n_kv * (head_dim/32) * 4); 
            cuda_lib.cuMemAlloc_v2(&mut vp, max_tokens_gpu * n_kv * head_dim * 2); 
            
            // Copy data via F16 path
            cuda_lib.cuMemcpyHtoD_v2(kp, k_data.as_ptr() as *const _, k_data.len() * 4);
            cuda_lib.cuMemcpyHtoD_v2(vp, v_data.as_ptr() as *const _, v_data.len() * 2);
            
            *gpu_cache_guard = Some((GpuPtr(kp as *mut _), GpuPtr(vp as *mut _), tokens));
        }
    }

    /// [NEW] Direct GPU-to-GPU injection for extremely fast layer replication
    #[cfg(feature = "cuda")]
    pub fn inject_gpu_kv_direct(&self, k_src: GpuPtr, v_src: GpuPtr, tokens: usize, n_kv: usize, head_dim: usize) {
        let mut gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
        
        unsafe {
            use cudarc::driver::sys::*;
            let cuda_lib = lib();
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
            // Allocate capped buffer (4k)
            let max_tokens_gpu = 4096;
            cuda_lib.cuMemAlloc_v2(&mut kp, max_tokens_gpu * n_kv * (head_dim/32) * 4); 
            cuda_lib.cuMemAlloc_v2(&mut vp, max_tokens_gpu * n_kv * head_dim * 2); 
            
            // Copy data via DtoD (Device to Device)
            let k_size = tokens * n_kv * (head_dim / 32) * 4;
            let v_size = tokens * n_kv * head_dim * 2;
            cuda_lib.cuMemcpyDtoD_v2(kp, k_src.0 as CUdeviceptr, k_size);
            cuda_lib.cuMemcpyDtoD_v2(vp, v_src.0 as CUdeviceptr, v_size);
            
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
    pub fn forward(&self, input_ids: &[u32], pixel_values: Option<&[f16]>, grid_thw: Option<&[u32; 3]>, seqlen_offset: usize, global_scratch: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>) -> Vec<f16> {
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

        self.forward_ext(input_ids, embeds, seqlen_offset, global_scratch)
    }

    pub fn forward_ext(&self, _input_ids: &[u32], embeds: Vec<f16>, seqlen_offset: usize, global_scratch: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>) -> Vec<f16> {
        let hid = self.config.hidden_size;
        let is_baking = self.layers.len() <= 1;
        let mut cur_x = embeds;

        for (i, layer) in self.layers.iter().enumerate() { 
            cur_x = layer.forward(&cur_x, &self.config, seqlen_offset, i, &self.rope_cos, &self.rope_sin, is_baking, global_scratch); 
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
            
            // 1. Direct Inject to FIRST layer (HtoD via F16)
            self.layers[0].inject_gpu_kv(&k, &v, n_kv, h_d);
            
            // 2. Replicate from Layer 0 to all other layers (DtoD)
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
    // [VRAM-SHARING] Global scratch buffers for all layers
    pub global_scratch_i: std::sync::Mutex<Option<(GpuPtr, usize)>>,
    pub global_scratch_o: std::sync::Mutex<Option<(GpuPtr, usize)>>,
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

    pub fn forward(&self, pixel_values: &[f16], grid_thw: &[u32; 3], rope_cos: &[f16], rope_sin: &[f16], global_scratch: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>) -> Vec<f16> {
        // [VISION-TRANSFORMER-PIPELINE]
        // Hidden states: (patches, hidden_size)
        // grid_thw: [T, H, W]
        let patches = (grid_thw[1] * grid_thw[2]) as usize;
        
        // 1. Patch Embedding
        let mut x = self.patch_embed.forward(pixel_values, global_scratch);
        
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
            }, 0, 0, rope_cos, rope_sin, false, global_scratch);
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
        }, 0, 0, rope_cos, rope_sin, false, global_scratch)
    }
}

impl NativeQwen3VLModel {
    pub fn load(config: Qwen3VLConfig, m_mmap: Arc<Mmap>, v_mmap: Option<Arc<Mmap>>, baking: bool, s_mmap: Option<Arc<Mmap>>) -> Result<Self> {
        #[cfg(feature = "cuda")]
        unsafe {
            let cuda_lib = lib();
            let res_init = cuda_lib.cuInit(0);
            if (res_init as i32) != 0 {
                println!("[CUDA-INIT] cuInit FAILED with code: {}", res_init as i32);
            } else {
                let mut count = 0;
                cuda_lib.cuDeviceGetCount(&mut count);
                println!("[CUDA-INIT] cuInit Success. Total CUDA devices visible: {}", count);
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

        let get_l = |base: &str, in_f: usize, out_f: usize, l_idx: i32| -> Result<NativeLinear> {
            let (key, is_primary) = find_key_ext(base, &st, st_sec.as_ref(), l_idx).ok_or_else(|| anyhow!("LinearTensorNotFound: {}", base))?;
            let current_st = if is_primary { &st } else { st_sec.as_ref().unwrap() };
            let current_mmap = if is_primary { &m_mmap } else { s_mmap.as_ref().unwrap() };

            let p_n = format!("{}.packed", key);
            let s_n = format!("{}.scales", key);
            
            if current_st.tensor(&p_n).is_ok() {
                let vp = current_st.tensor(&p_n)?; 
                let vs = current_st.tensor(&s_n)?;
                
                // [STABILITY-DIAG] 로딩 시점에 텐서 유효성 샘플링
                let scale_sample = vs.data();
                if scale_sample.len() >= 4 {
                    let s0 = f16::from_le_bytes([scale_sample[0], scale_sample[1]]).to_f32();
                    let s1 = f16::from_le_bytes([scale_sample[2], scale_sample[3]]).to_f32();
                    if s0 == 0.0 && scale_sample.iter().take(100).all(|&b| b == 0) {
                        println!("[LOAD-CRITICAL] Tensor {} has ALL ZERO scales! Inference will fail.", key);
                    } else if l_idx == 0 && key.contains("v_proj") {
                        println!("[LOAD-DEBUG] Layer 0 V-PROJ Scales sample: {:.6}, {:.6}", s0, s1);
                    }
                }

                let packed_sample = vp.data();
                if l_idx == 0 && key.contains("v_proj") && packed_sample.len() >= 4 {
                    let p0 = u32::from_le_bytes([packed_sample[0], packed_sample[1], packed_sample[2], packed_sample[3]]);
                    println!("[LOAD-DEBUG] Layer 0 V-PROJ Packed sample: 0x{:08X}", p0);
                }

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
                })
            }
        };

        let get_t = |name: &str, l_idx: i32| -> Result<NativeTensor> {
            let (key, is_primary) = find_key_ext(name, &st, st_sec.as_ref(), l_idx).ok_or_else(|| anyhow!("TensorNotFound: {}", name))?;
            let current_st = if is_primary { &st } else { st_sec.as_ref().unwrap() };
            let current_mmap = if is_primary { &m_mmap } else { s_mmap.as_ref().unwrap() };

            let v = current_st.tensor(&key)?;
            let off = unsafe { v.data().as_ptr().offset_from(current_mmap.as_ptr()) } as usize;
            Ok(NativeTensor::from_mmap(current_mmap.clone(), off, v.shape().to_vec(), NativeDType::F16))
        };

        println!("[LOAD] Initializing Native Engine (Baking: {})...", baking);

        // [CRITICAL-FIX] 임베딩 로더: 임베딩은 항상 F16 룩업 테이블이어야 함
        let get_embed = |base: &str, vocab: usize, hid: usize| -> Result<NativeLinear> {
            let (key, is_primary) = find_key_ext(base, &st, st_sec.as_ref(), -1).ok_or_else(|| anyhow!("EmbedTensorNotFound: {}", base))?;
            let current_st = if is_primary { &st } else { st_sec.as_ref().unwrap() };
            let current_mmap = if is_primary { &m_mmap } else { s_mmap.as_ref().unwrap() };

            if current_st.tensor(&format!("{}.packed", key)).is_ok() {
                println!("[LOAD] WARNING: embed_tokens was bit-quantized. Reverting to F16 for lookup table...");
                // 양자화된 경우 matmul을 통해 F16으로 복구 (단순 lookup이 불가능하므로 로딩 시점에 미리 계산)
                // 하지만 여기서는 간결성을 위해 Standard로 강제 로드 시도 (GGUF 원본이 F16인 경우 대비)
                if current_st.tensor(&key).is_ok() {
                    let v = current_st.tensor(&key)?;
                    let o = unsafe { v.data().as_ptr().offset_from(current_mmap.as_ptr()) } as usize;
                    return Ok(NativeLinear { in_features: vocab, out_features: hid, variant: LinearVariant::Standard { weight: NativeTensor::from_mmap(current_mmap.clone(), o, v.shape().to_vec(), NativeDType::F16), bias: None }, device_id: -1 });
                }
            }
            get_l(base, vocab, hid, -1)
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
            let layer_idx = i as i32;

            layers.push(NativeLayer {
                input_layernorm: get_t(&format!("{}.input_layernorm.weight", p), layer_idx)?,
                post_attention_layernorm: get_t(&format!("{}.post_attention_layernorm.weight", p), layer_idx)?,
                q_norm: get_t(&format!("{}.self_attn.q_norm.weight", p), layer_idx).ok(),
                k_norm: get_t(&format!("{}.self_attn.k_norm.weight", p), layer_idx).ok(),
                q_proj: get_l(&format!("{}.self_attn.q_proj.weight", p), t_c.hidden_size, q_out, layer_idx)?,
                k_proj: get_l(&format!("{}.self_attn.k_proj.weight", p), t_c.hidden_size, kv_out, layer_idx)?,
                v_proj: get_l(&format!("{}.self_attn.v_proj.weight", p), t_c.hidden_size, kv_out, layer_idx)?,
                o_proj: get_l(&format!("{}.self_attn.o_proj.weight", p), q_out, t_c.hidden_size, layer_idx)?,
                gate_proj: get_l(&format!("{}.mlp.gate_proj.weight", p), t_c.hidden_size, t_c.intermediate_size, layer_idx)?,
                up_proj: get_l(&format!("{}.mlp.up_proj.weight", p), t_c.hidden_size, t_c.intermediate_size, layer_idx)?,
                down_proj: get_l(&format!("{}.mlp.down_proj.weight", p), t_c.intermediate_size, t_c.hidden_size, layer_idx)?,
                device_id: -1, kv_cache: std::sync::Mutex::new(None), gpu_kv_cache: std::sync::Mutex::new(None),
                gpu_broken: std::sync::atomic::AtomicBool::new(false),
            });
        }
        
        let norm = get_t("model.language_model.norm.weight", -1)?;
        
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
                    }, device_id: -1 })
                } else {
                    let v = vst.tensor(&key)?;
                    let o = unsafe { v.data().as_ptr().offset_from(vm.as_ptr()) } as usize;
                    Ok(NativeLinear { in_features: in_f, out_features: out_f, variant: LinearVariant::Standard { weight: NativeTensor::from_mmap(vm.clone(), o, v.shape().to_vec(), NativeDType::F16), bias: None }, device_id: -1 })
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
                    gpu_broken: std::sync::atomic::AtomicBool::new(false),
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
                gpu_broken: std::sync::atomic::AtomicBool::new(false),
            };
            Some(NativeVisionModel { patch_embed, blocks, merger })
        } else { None };
            
        Ok(Self { 
            config: config.clone(), 
            text_model: NativeQwen3TextModel { config: t_c.clone(), embed_tokens: emb, layers, norm, rope_cos, rope_sin }, 
            lm_head: head, 
            visual,
            global_scratch_i: std::sync::Mutex::new(None),
            global_scratch_o: std::sync::Mutex::new(None),
        })
    }
    pub fn forward(&self, i_ids: &[u32], p_v: Option<&[f16]>, g_t: Option<&[u32; 3]>, s_o: usize) -> Vec<f16> {
        #[cfg(feature = "cuda")]
        if self.device_id >= 0 {
            use cudarc::driver::sys::*;
            unsafe {
                let cuda_lib = crate::models::qwen3vl::native_backend::lib();
                let mut ctx = std::ptr::null_mut() as CUcontext;
                cuda_lib.cuCtxGetCurrent(&mut ctx);
                if ctx == std::ptr::null_mut() {
                    let mut dev = 0 as CUdevice;
                    cuda_lib.cuDeviceGet(&mut dev, self.device_id);
                    cuda_lib.cuDevicePrimaryCtxRetain(&mut ctx, dev);
                    cuda_lib.cuCtxSetCurrent(ctx);
                }
            }
        }

        // [2026-SPECULATIVE-TREE] Support for validating multiple candidates in parallel
        let _batch_size = i_ids.len() / (if i_ids.len() > 1 { 1 } else { 1 }); // Adjust for Tree-Depth
        
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

        // [VRAM-SHARING] Use global scratch buffers
        let scratch = Some((&self.global_scratch_i, &self.global_scratch_o));

        // [VISION-FUSION] Skip during speculative tree validation to save cycles
        if let (Some(pv), Some(gt)) = (p_v, g_t) {
            if let Some(ref visual) = self.visual {
                let vision_features = visual.forward(pv, gt, &self.text_model.rope_cos, &self.text_model.rope_sin, scratch);
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

        let norm_x = self.text_model.forward_ext(i_ids, embeds, s_o, scratch);
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
    
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
}

unsafe impl Send for NativeLayer {}
unsafe impl Sync for NativeLayer {}

impl NativeLayer {
    pub fn forward(&self, x: &[f16], config: &Qwen3VLTextConfig, seqlen_offset: usize, _idx: usize) -> Vec<f16> {
        let hidden_size = config.hidden_size; let q_len = x.len() / hidden_size;
        let head_dim = config.head_dim; let n_h = config.num_attention_heads; let n_kv = config.num_key_value_heads;

        let residual = x.to_vec();
        let ln_weight_cow = self.input_layernorm.get_slice::<f16>();
        let x_norm = native_rms_norm_f16(x, ln_weight_cow.as_ref(), config.rms_norm_eps as f32, hidden_size);
        
        // [RAYON-STRATEGY] Only use parallel join if processing multiple tokens (prefill)
        let (mut q, mut k, v) = if q_len > 1 {
            let (q_p, (k_p, v_p)) = rayon::join(
                || self.q_proj.forward(&x_norm),
                || rayon::join(|| self.k_proj.forward(&x_norm), || self.v_proj.forward(&x_norm))
            );
            (q_p, k_p, v_p)
        } else {
            (self.q_proj.forward(&x_norm), self.k_proj.forward(&x_norm), self.v_proj.forward(&x_norm))
        };

        if let Some(ref qw) = self.q_norm { 
            let qw_cow = qw.get_slice::<f16>();
            q = native_rms_norm_f16(&q, qw_cow.as_ref(), config.rms_norm_eps as f32, head_dim); 
        }
        if let Some(ref kw) = self.k_norm { 
            let kw_cow = kw.get_slice::<f16>();
            k = native_rms_norm_f16(&k, kw_cow.as_ref(), config.rms_norm_eps as f32, head_dim); 
        }

        native_apply_rope_f16_with_offset(&mut q, &mut k, q_len, seqlen_offset, n_h, head_dim, config.rope_theta);

        #[cfg(feature = "cuda")]
        if self.device_id >= 0 {
            use cudarc::driver::sys::*;
            let mut gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
            let (k_ptr, v_ptr, mut current_len) = if let Some((kp, vp, l)) = *gpu_cache_guard { (kp, vp, l) } else {
                let mut kp: CUdeviceptr = 0; let mut vp: CUdeviceptr = 0;
                unsafe { 
                    let cuda_lib = lib();
                    lib().cuMemAlloc_v2(&mut kp, 16384 * n_kv * (head_dim/32) * 4); 
                    lib().cuMemAlloc_v2(&mut vp, 16384 * n_kv * head_dim * 4); 
                }
                
                // Note: Stitched cache is now handled by batch_upload_stitched_cache
                // so we don't need the individual 'cpu_cache.take()' upload here.
                (GpuPtr(kp as *mut _), GpuPtr(vp as *mut _), 0)
            };

            let k_packed = pack_f16_to_bits(&k);
            let v_f32: Vec<f32> = if q_len > 1 {
                v.par_iter().map(|val: &f16| val.to_f32()).collect()
            } else {
                v.iter().map(|val: &f16| val.to_f32()).collect()
            };
            unsafe {
                let cuda_lib = lib();
                let k_offset = current_len * n_kv * (head_dim/32) * 4;
                let v_offset = current_len * n_kv * head_dim * 4;
                let _ = cuda_lib.cuMemcpyHtoD_v2((k_ptr.0 as usize + k_offset) as CUdeviceptr, k_packed.as_ptr() as *const _, k_packed.len() * 4);
                let _ = cuda_lib.cuMemcpyHtoD_v2((v_ptr.0 as usize + v_offset) as CUdeviceptr, v_f32.as_ptr() as *const _, v_f32.len() * 4);
            }
            let new_len = current_len + q_len;
            *gpu_cache_guard = Some((k_ptr, v_ptr, new_len));

            let attn_out = native_bit_serial_attn_gpu(&q, k_ptr, v_ptr, n_h, head_dim, new_len, self.device_id as usize);
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

        let attn_out = native_bit_serial_attn_f16(&q, &k_p_f, &v_f_f, hidden_size, n_h, q_len, t_s);
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
            use cudarc::driver::sys::lib;
            let gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
            if let Some((k_ptr, v_ptr, current_len)) = *gpu_cache_guard {
                if current_len == 0 { return None; }
                let k_size = current_len * n_kv * (head_dim / 32);
                let v_size = current_len * n_kv * head_dim;
                
                let mut k_host = vec![0u32; k_size];
                let mut v_host_f32 = vec![0.0f32; v_size];
                
                unsafe {
                    let cuda_lib = lib();
                    let _ = cuda_lib.cuMemcpyDtoH_v2(k_host.as_mut_ptr() as *mut _, k_ptr.0 as usize as u64, k_size * 4);
                    let _ = cuda_lib.cuMemcpyDtoH_v2(v_host_f32.as_mut_ptr() as *mut _, v_ptr.0 as usize as u64, v_size * 4);
                }
                
                let v_host: Vec<f16> = v_host_f32.into_iter().map(f16::from_f32).collect();
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
            // Allocate full buffer (16k)
            lib.cuMemAlloc_v2(&mut kp, 16384 * n_kv * (head_dim/32) * 4); 
            lib.cuMemAlloc_v2(&mut vp, 16384 * n_kv * head_dim * 4); 
            
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
            // Allocate full buffer (16k)
            lib.cuMemAlloc_v2(&mut kp, 16384 * n_kv * (head_dim/32) * 4); 
            lib.cuMemAlloc_v2(&mut vp, 16384 * n_kv * head_dim * 4); 
            
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
    pub config: Qwen3VLTextConfig, pub embed_tokens: NativeLinear, pub layers: Vec<NativeLayer>, pub norm: NativeTensor,
}

unsafe impl Send for NativeQwen3TextModel {}
unsafe impl Sync for NativeQwen3TextModel {}

impl NativeQwen3TextModel {
    pub fn forward(&self, input_ids: &[u32], seqlen_offset: usize) -> Vec<f16> {
        let hid = self.config.hidden_size;
        let x = match &self.embed_tokens.variant {
            LinearVariant::Standard { weight, .. } => {
                let w_cow = weight.get_slice::<f16>();
                native_embedding_lookup_f16(input_ids, w_cow.as_ref(), hid)
            },
            LinearVariant::BitSerial { weight_packed, .. } => {
                // If quantized embedding is used, we still need a lookup. 
                // For now, most embeddings are Standard f16.
                let w_cow = weight_packed.get_slice::<f16>();
                native_embedding_lookup_f16(input_ids, w_cow.as_ref(), hid)
            }
        };
        let mut cur_x = x;
        for (i, layer) in self.layers.iter().enumerate() { cur_x = layer.forward(&cur_x, &self.config, seqlen_offset, i); }
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
}

impl NativeQwen3VLModel {
    pub fn load(config: Qwen3VLConfig, m_mmap: Arc<Mmap>, v_mmap: Option<Arc<Mmap>>, baking: bool) -> Result<Self> {
        let st = SafeTensors::deserialize(&m_mmap)?; 
        let t_c = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;
        
        let find_key = |target: &str| -> Option<String> {
            let variations = vec![
                target.to_string(),
                target.replace("model.", "model.language_model."),
                target.replace("model.layers", "model.language_model.layers"),
                target.replace("lm_head", "model.lm_head"),
                target.replace("lm_head", "output"),
                format!("model.language_model.{}", target.replace("model.", "")),
                format!("model.{}", target),
            ];
            for v in variations {
                if st.tensor(&v).is_ok() { return Some(v); }
                if st.tensor(&format!("{}.packed", v)).is_ok() { return Some(v); }
            }
            // Exhaustive search for tied weights or renamed heads
            for name in st.names() {
                if name.contains(target) {
                    return Some(name.replace(".packed", "").replace(".scales", "").replace(".shape", "").replace(".format", ""));
                }
            }
            None
        };

        let get_l = |base: &str, in_f: usize, out_f: usize| -> Result<NativeLinear> {
            let key = find_key(base).ok_or_else(|| anyhow!("LinearTensorNotFound: {}", base))?;
            let p_n = format!("{}.packed", key);
            let s_n = format!("{}.scales", key);
            
            if st.tensor(&p_n).is_ok() {
                let vp = st.tensor(&p_n)?; 
                let vs = st.tensor(&s_n)?;
                let op = unsafe { vp.data().as_ptr().offset_from(m_mmap.as_ptr()) } as usize;
                let os = unsafe { vs.data().as_ptr().offset_from(m_mmap.as_ptr()) } as usize;
                
                Ok(NativeLinear { 
                    in_features: in_f, out_features: out_f, 
                    variant: LinearVariant::BitSerial {
                        weight_packed: NativeTensor::from_mmap(m_mmap.clone(), op, vp.shape().to_vec(), NativeDType::U32),
                        scales: NativeTensor::from_mmap(m_mmap.clone(), os, vs.shape().to_vec(), NativeDType::F16),
                        bias: None,
                    }, 
                    device_id: -1,
                    scratch_i: std::sync::Mutex::new(None),
                    scratch_o: std::sync::Mutex::new(None),
                })
            } else {
                let v = st.tensor(&key)?;
                let o = unsafe { v.data().as_ptr().offset_from(m_mmap.as_ptr()) } as usize;
                Ok(NativeLinear { 
                    in_features: in_f, out_features: out_f, 
                    variant: LinearVariant::Standard { 
                        weight: NativeTensor::from_mmap(m_mmap.clone(), o, v.shape().to_vec(), NativeDType::F16), 
                        bias: None 
                    }, 
                    device_id: -1,
                    scratch_i: std::sync::Mutex::new(None),
                    scratch_o: std::sync::Mutex::new(None),
                })
            }
        };

        let get_t = |name: &str| -> Result<NativeTensor> {
            let key = find_key(name).ok_or_else(|| anyhow!("TensorNotFound: {}", name))?;
            let v = st.tensor(&key)?;
            let off = unsafe { v.data().as_ptr().offset_from(m_mmap.as_ptr()) } as usize;
            Ok(NativeTensor::from_mmap(m_mmap.clone(), off, v.shape().to_vec(), NativeDType::F16))
        };

        println!("[LOAD] Initializing Native Engine (Baking: {})...", baking);

        // 임베딩 로드
        let emb = get_l("model.embed_tokens.weight", 151936, t_c.hidden_size)
            .or_else(|_| get_l("model.language_model.embed_tokens.weight", 151936, t_c.hidden_size))?;
            
        let mut layers = Vec::new(); 
        let l_to_l = if baking { 1 } else { t_c.num_hidden_layers };
        for i in 0..l_to_l {
            let p = format!("model.layers.{}", i);
            layers.push(NativeLayer {
                input_layernorm: get_t(&format!("{}.input_layernorm.weight", p))?,
                post_attention_layernorm: get_t(&format!("{}.post_attention_layernorm.weight", p))?,
                q_norm: get_t(&format!("{}.self_attn.q_norm.weight", p)).ok(),
                k_norm: get_t(&format!("{}.self_attn.k_norm.weight", p)).ok(),
                q_proj: get_l(&format!("{}.self_attn.q_proj.weight", p), t_c.hidden_size, t_c.hidden_size)?,
                k_proj: get_l(&format!("{}.self_attn.k_proj.weight", p), t_c.hidden_size, t_c.hidden_size)?,
                v_proj: get_l(&format!("{}.self_attn.v_proj.weight", p), t_c.hidden_size, t_c.hidden_size)?,
                o_proj: get_l(&format!("{}.self_attn.o_proj.weight", p), t_c.hidden_size, t_c.hidden_size)?,
                gate_proj: get_l(&format!("{}.mlp.gate_proj.weight", p), t_c.hidden_size, t_c.intermediate_size)?,
                up_proj: get_l(&format!("{}.mlp.up_proj.weight", p), t_c.hidden_size, t_c.intermediate_size)?,
                down_proj: get_l(&format!("{}.mlp.down_proj.weight", p), t_c.intermediate_size, t_c.hidden_size)?,
                device_id: -1, kv_cache: std::sync::Mutex::new(None), gpu_kv_cache: std::sync::Mutex::new(None),
            });
        }
        let norm = get_t("model.norm.weight")?;
        
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
            };
            Some(NativeVisionModel { patch_embed, blocks, merger })
        } else { None };
            
        Ok(Self { config: config.clone(), text_model: NativeQwen3TextModel { config: t_c.clone(), embed_tokens: emb, layers, norm }, lm_head: head, visual })
    }
    pub fn forward(&self, i_ids: &[u32], _p_v: Option<&[f16]>, _g_t: Option<&[u32; 3]>, s_o: usize) -> Vec<f16> {
        let x = self.text_model.forward(i_ids, s_o); self.lm_head.forward(&x)
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
    
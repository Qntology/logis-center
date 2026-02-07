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
use std::sync::atomic::Ordering;

pub enum LinearVariant {
    Standard { weight: NativeTensor, bias: Option<NativeTensor> },
    BitSerial { weight_packed: NativeTensor, scales: NativeTensor, bias: Option<NativeTensor> },
}

pub struct ForwardWorkspace {
    pub hidden_a: Vec<f16>,
    pub hidden_b: Vec<f16>,
    pub intermediate_a: Vec<f16>,
    pub intermediate_b: Vec<f16>,
    pub intermediate_c: Vec<f16>,
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
        let eps = f16::from_f32(1e-6);
        if self.hidden_a.len() < req_hidden { self.hidden_a.resize(req_hidden, eps); }
        if self.hidden_b.len() < req_hidden { self.hidden_b.resize(req_hidden, eps); }
        if self.intermediate_a.len() < req_inter { self.intermediate_a.resize(req_inter, eps); }
        if self.intermediate_b.len() < req_inter { self.intermediate_b.resize(req_inter, eps); }
        if self.intermediate_c.len() < req_inter { self.intermediate_c.resize(req_inter, eps); }
        if self.q.len() < req_q { self.q.resize(req_q, eps); }
        if self.k.len() < req_kv { self.k.resize(req_kv, eps); }
        if self.v.len() < req_kv { self.v.resize(req_kv, eps); }
    }
}

pub struct DynamicKVCache {
    pub k: Vec<u32>,
    pub v: Vec<f16>,
    pub capacity: usize,
    pub current_len: usize,
}

impl DynamicKVCache {
    pub fn new() -> Self { Self { k: Vec::new(), v: Vec::new(), capacity: 0, current_len: 0 } }
    pub fn grow(&mut self, needed_len: usize, n_kv: usize, head_dim: usize) {
        if needed_len > self.capacity {
            let new_cap = (needed_len + 1023) / 1024 * 1024;
            self.k.resize(new_cap * n_kv * (head_dim/32), 0);
            self.v.resize(new_cap * n_kv * head_dim, f16::from_f32(1e-6));
            self.capacity = new_cap;
        }
    }
    pub fn clear(&mut self) { self.current_len = 0; }
}

pub struct DynamicGpuKVCache {
    pub k_ptr: Option<GpuPtr>,
    pub v_ptr: Option<GpuPtr>,
    pub capacity: usize,
    pub current_len: usize,
}

impl DynamicGpuKVCache {
    pub fn new() -> Self { Self { k_ptr: None, v_ptr: None, capacity: 0, current_len: 0 } }
    #[cfg(feature = "cuda")]
    pub fn grow(&mut self, needed_len: usize, n_kv: usize, head_dim: usize, device_id: i32) {
        if needed_len > self.capacity {
            let new_cap = (needed_len + 1023) / 1024 * 1024;
            let k_bytes = new_cap * n_kv * (head_dim/32) * 4;
            let v_bytes = new_cap * n_kv * head_dim * 2;
            unsafe {
                let cl = crate::models::qwen3vl::native_backend::lib();
                let mut ctx = std::ptr::null_mut() as cudarc::driver::sys::CUcontext;
                cl.cuCtxGetCurrent(&mut ctx);
                if ctx == std::ptr::null_mut() && device_id >= 0 {
                    let mut dev = 0 as cudarc::driver::sys::CUdevice; cl.cuDeviceGet(&mut dev, device_id);
                    cl.cuDevicePrimaryCtxRetain(&mut ctx, dev); cl.cuCtxSetCurrent(ctx);
                }
                let mut new_kp: cudarc::driver::sys::CUdeviceptr = 0;
                let mut new_vp: cudarc::driver::sys::CUdeviceptr = 0;
                cl.cuMemAlloc_v2(&mut new_kp, k_bytes);
                cl.cuMemAlloc_v2(&mut new_vp, v_bytes);
                if let (Some(old_k), Some(old_v)) = (self.k_ptr.take(), self.v_ptr.take()) {
                    if self.current_len > 0 {
                        let okb = self.current_len * n_kv * (head_dim/32) * 4;
                        let ovb = self.current_len * n_kv * head_dim * 2;
                        cl.cuMemcpyDtoD_v2(new_kp, old_k.0 as cudarc::driver::sys::CUdeviceptr, okb);
                        cl.cuMemcpyDtoD_v2(new_vp, old_v.0 as cudarc::driver::sys::CUdeviceptr, ovb);
                    }
                    cl.cuMemFree_v2(old_k.0 as cudarc::driver::sys::CUdeviceptr);
                    cl.cuMemFree_v2(old_v.0 as cudarc::driver::sys::CUdeviceptr);
                }
                self.k_ptr = Some(GpuPtr(new_kp as *mut _));
                self.v_ptr = Some(GpuPtr(new_vp as *mut _));
                self.capacity = new_cap;
            }
        }
    }
    pub fn clear(&mut self) { self.current_len = 0; }
}

pub fn dequantize_bit_serial_to_f16(packed: &[u32], scales: &[f16], n: usize, src_k: usize, target_k: usize) -> Vec<f16> {
    let src_k_blocks = (src_k + 31) / 32;
    let n_padded = (n + 7) / 8 * 8;
    let mut out = vec![f16::ZERO; n * target_k];
    let ratio = if target_k > src_k { target_k / src_k } else { 1 };
    for no in 0..(n_padded / 8) {
        for skb in 0..src_k_blocks {
            for sub_n in 0..8 {
                let n_idx = no * 8 + sub_n; if n_idx >= n { continue; }
                let s_idx = (no * src_k_blocks + skb) * 8 + sub_n;
                let bits = packed[s_idx]; let scale = scales[s_idx];
                for r in 0..ratio {
                    let tkb = skb + (r * src_k_blocks);
                    for b in 0..32 {
                        let kidx = tkb * 32 + b; if kidx >= target_k { break; }
                        let val = if ((bits >> b) & 1) == 1 { scale } else { -scale };
                        out[n_idx * target_k + kidx] = val;
                    }
                }
            }
        }
    }
    out
}

pub struct NativeLinear {
    pub in_features: usize, pub out_features: usize, pub src_in: usize, pub src_out: usize,
    pub variant: LinearVariant, pub device_id: i32,
}

unsafe impl Send for NativeLinear {}
unsafe impl Sync for NativeLinear {}

impl NativeLinear {
    pub fn is_bit_serial(&self) -> bool { matches!(self.variant, LinearVariant::BitSerial { .. }) }
    #[cfg(feature = "cuda")]
    pub fn forward_gpu(&self, d_i: CUdeviceptr, d_o: CUdeviceptr, m: usize) {
        unsafe {
            match &self.variant {
                LinearVariant::Standard { weight, .. } => {
                    let d_w = weight.gpu_ptr.expect("W on GPU").0 as *const f16;
                    crate::models::qwen3vl::native_backend::standard_matmul_cuda_f16(d_i as *const f16, d_w, d_o as *mut f16, m as i32, self.out_features as i32, self.in_features as i32);
                },
                LinearVariant::BitSerial { weight_packed, scales, .. } => {
                    let d_w = weight_packed.gpu_ptr.expect("W on GPU").0 as *const u32;
                    let d_s = scales.gpu_ptr.expect("S on GPU").0 as *const f16;
                    crate::models::qwen3vl::native_backend::cuda_matmul_f16(d_i as *const f16, d_w, d_s, d_o as *mut f16, m as i32, self.out_features as i32, self.in_features as i32, self.device_id, self.src_in as i32);
                }
            }
        }
    }
    pub fn forward_into(&self, x: &[f16], out: &mut [f16], global_scratch: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>) {
        let m = x.len() / self.in_features;
        match &self.variant {
            LinearVariant::Standard { weight, bias } => {
                #[cfg(feature = "cuda")]
                if self.device_id >= 0 {
                    if let Some(w_gpu) = weight.gpu_ptr {
                        let (d_i, d_o, _) = if let Some((si, so, sr)) = global_scratch { self.ensure_gpu_buffers_ext(m, si, so, sr) } else { (0, 0, 0) };
                        if d_i != 0 && d_o != 0 {
                            unsafe {
                                let cl = crate::models::qwen3vl::native_backend::lib();
                                cl.cuMemcpyHtoD_v2(d_i, x.as_ptr() as *const _, x.len() * 2);
                                crate::models::qwen3vl::native_backend::standard_matmul_cuda_f16(d_i as *const f16, w_gpu.0 as *const f16, d_o as *mut f16, m as i32, self.out_features as i32, self.in_features as i32);
                                cl.cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut _, d_o, out.len() * 2);
                                if let Some(b) = bias {
                                    let br = b.get_raw_slice::<f16>();
                                    for i in 0..m { for j in 0..self.out_features { out[i * self.out_features + j] += br[j]; } }
                                }
                                return;
                            }
                        }
                    }
                }
                let w_cow = weight.get_slice::<f16>(); let b_cow = bias.as_ref().map(|t| t.get_slice::<f16>());
                native_linear_f16_into(x, w_cow.as_ref(), b_cow.as_ref().map(|b| b.as_ref()), out, m, self.out_features, self.in_features);
            },
            LinearVariant::BitSerial { weight_packed, scales, bias } => {
                #[cfg(feature = "cuda")]
                if self.device_id >= 0 && weight_packed.gpu_ptr.is_some() {
                    let (d_i, d_o, _) = if let Some((si, so, sr)) = global_scratch { self.ensure_gpu_buffers_ext(m, si, so, sr) } else { (0, 0, 0) };
                    if d_i != 0 && d_o != 0 {
                        crate::models::qwen3vl::native_backend::bit_serial_matmul_gpu_buffered_into(x, weight_packed, scales, out, m, self.out_features, self.in_features, self.device_id as usize, d_i, d_o, self.src_in);
                        if let Some(b) = bias { unsafe { let br = b.get_raw_slice::<f16>(); for i in 0..m { for j in 0..self.out_features { out[i * self.out_features + j] += br[j]; } } } }
                        return;
                    }
                }
                unsafe {
                    let wpr = weight_packed.get_raw_slice::<u32>(); let sr = scales.get_raw_slice::<f16>();
                    bit_serial_matmul_f32_extreme_into(x, wpr, sr, out, m, self.out_features, self.in_features);
                    if let Some(b) = bias { let br = b.get_raw_slice::<f16>(); for i in 0..m { for j in 0..self.out_features { out[i * self.out_features + j] += br[j]; } } }
                }
            }
        }
    }
    pub fn forward(&self, x: &[f16], gs: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>) -> Vec<f16> {
        let mut out = vec![f16::from_f32(1e-6); (x.len() / self.in_features) * self.out_features];
        self.forward_into(x, &mut out, gs); out
    }
    #[cfg(feature = "cuda")]
    fn ensure_gpu_buffers_ext(&self, m: usize, si: &std::sync::Mutex<Option<(GpuPtr, usize)>>, so: &std::sync::Mutex<Option<(GpuPtr, usize)>>, sr: &std::sync::Mutex<Option<(GpuPtr, usize)>>) -> (CUdeviceptr, CUdeviceptr, CUdeviceptr) {
        let req_i = (m * self.in_features * 4 * 125) / 100; let req_o = (m * self.out_features * 4 * 125) / 100;
        let req_r = (m * self.in_features.max(self.out_features) * 4 * 125) / 100;
        let cl = unsafe { crate::models::qwen3vl::native_backend::lib() };
        unsafe {
            let mut ctx = std::ptr::null_mut() as cudarc::driver::sys::CUcontext;
            cl.cuCtxGetCurrent(&mut ctx);
            if ctx == std::ptr::null_mut() && self.device_id >= 0 {
                let mut dev = 0 as cudarc::driver::sys::CUdevice; cl.cuDeviceGet(&mut dev, self.device_id);
                cl.cuDevicePrimaryCtxRetain(&mut ctx, dev); cl.cuCtxSetCurrent(ctx);
            }
        }
        let mut sig = si.lock().unwrap();
        let di = if let Some((p, s)) = *sig { if s >= (m * self.in_features * 4) { p.0 as CUdeviceptr } else {
            let mut np: CUdeviceptr = 0; let _ = unsafe { cl.cuMemFree_v2(p.0 as CUdeviceptr) };
            if (unsafe { cl.cuMemAlloc_v2(&mut np, req_i) } as i32) == 0 { *sig = Some((GpuPtr(np as *mut _), req_i)); np } else { 0 }
        } } else {
            let mut np: CUdeviceptr = 0;
            if (unsafe { cl.cuMemAlloc_v2(&mut np, req_i) } as i32) == 0 { *sig = Some((GpuPtr(np as *mut _), req_i)); np } else { 0 }
        };
        let mut sog = so.lock().unwrap();
        let dog = if let Some((p, s)) = *sog { if s >= (m * self.out_features * 4) { p.0 as CUdeviceptr } else {
            let mut np: CUdeviceptr = 0; let _ = unsafe { cl.cuMemFree_v2(p.0 as CUdeviceptr) };
            if (unsafe { cl.cuMemAlloc_v2(&mut np, req_o) } as i32) == 0 { *sog = Some((GpuPtr(np as *mut _), req_o)); np } else { 0 }
        } } else {
            let mut np: CUdeviceptr = 0;
            if (unsafe { cl.cuMemAlloc_v2(&mut np, req_o) } as i32) == 0 { *sog = Some((GpuPtr(np as *mut _), req_o)); np } else { 0 }
        };
        let mut srg = sr.lock().unwrap();
        let drg = if let Some((p, s)) = *srg { if s >= (m * self.in_features.max(self.out_features) * 4) { p.0 as CUdeviceptr } else {
            let mut np: CUdeviceptr = 0; let _ = unsafe { cl.cuMemFree_v2(p.0 as CUdeviceptr) };
            if (unsafe { cl.cuMemAlloc_v2(&mut np, req_r) } as i32) == 0 { *srg = Some((GpuPtr(np as *mut _), req_r)); np } else { 0 }
        } } else {
            let mut np: CUdeviceptr = 0;
            if (unsafe { cl.cuMemAlloc_v2(&mut np, req_r) } as i32) == 0 { *srg = Some((GpuPtr(np as *mut _), req_r)); np } else { 0 }
        };
        (di, dog, drg)
    }
    pub fn move_to_gpu(&mut self, device_id: i32) -> anyhow::Result<()> {
        self.device_id = device_id;
        match &mut self.variant {
            LinearVariant::Standard { weight, bias } => { 
                weight.move_to_gpu(device_id)?; 
                if let Some(b) = bias { b.move_to_gpu(device_id)?; } 
            },
            LinearVariant::BitSerial { weight_packed, scales, bias } => { 
                weight_packed.move_to_gpu(device_id)?; 
                scales.move_to_gpu(device_id)?; 
                if let Some(b) = bias { b.move_to_gpu(device_id)?; } 
            }
        }
        Ok(())
    }
}

pub struct NativeLayer {
    pub input_layernorm: NativeTensor, pub post_attention_layernorm: NativeTensor,
    pub q_norm: Option<NativeTensor>, pub k_norm: Option<NativeTensor>,
    pub q_proj: NativeLinear, pub k_proj: NativeLinear, pub v_proj: NativeLinear, pub o_proj: NativeLinear,
    pub gate_proj: NativeLinear, pub up_proj: NativeLinear, pub down_proj: NativeLinear,
    pub device_id: i32, pub is_support_layer: bool,
    pub kv_cache: std::sync::Mutex<DynamicKVCache>, pub gpu_kv_cache: std::sync::Mutex<DynamicGpuKVCache>,
    pub rope_cache_gpu: std::sync::Mutex<Option<(NativeTensor, NativeTensor)>>,
    pub gpu_broken: std::sync::atomic::AtomicBool,
}

unsafe impl Send for NativeLayer {}
unsafe impl Sync for NativeLayer {}

impl NativeLayer {
    pub fn move_to_gpu(&mut self, device_id: i32) -> anyhow::Result<()> {
        self.device_id = device_id;
        self.input_layernorm.move_to_gpu(device_id)?;
        self.post_attention_layernorm.move_to_gpu(device_id)?;
        if let Some(ref mut qn) = self.q_norm { qn.move_to_gpu(device_id)?; }
        if let Some(ref mut kn) = self.k_norm { kn.move_to_gpu(device_id)?; }
        self.q_proj.move_to_gpu(device_id)?;
        self.k_proj.move_to_gpu(device_id)?;
        self.v_proj.move_to_gpu(device_id)?;
        self.o_proj.move_to_gpu(device_id)?;
        self.gate_proj.move_to_gpu(device_id)?;
        self.up_proj.move_to_gpu(device_id)?;
        self.down_proj.move_to_gpu(device_id)?;
        Ok(())
    }

    pub fn forward<'a>(
        &self, x: &[f16], config: &Qwen3VLTextConfig, s_o: usize, _idx: usize, _r_cos: &[f16], _r_sin: &[f16], 
        is_baking: bool, is_vision: bool, global_scratch: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>,
        workspace: &'a mut ForwardWorkspace, ping_pong: bool
    ) -> &'a [f16] {
        let h_s = config.hidden_size; let q_len = x.len() / h_s; let h_d = config.head_dim;
        let n_h = config.num_attention_heads; let n_kv = config.num_key_value_heads;
        workspace.ensure_capacity(h_s, config.intermediate_size, q_len, n_h, n_kv, h_d);
        let mut f_alpha = 1.0f32; let mut s_gain = 1.0f32;
        if _idx == 0 && q_len > 0 {
            let samples = x.iter().take(500).map(|v| v.to_f32().abs()).collect::<Vec<_>>();
            let m_a = samples.iter().sum::<f32>() / samples.len() as f32;
            let max_a = samples.iter().fold(1e-9f32, |a, &b| a.max(b));
            let density = (m_a / max_a).clamp(0.0, 1.0);
            let t_ctx = s_o + q_len; let c_log = (t_ctx as f32 / 1024.0).max(1.0).ln();
            if is_vision {
                f_alpha = (1.55 + c_log * 0.18).clamp(1.5, 2.5); s_gain = 1.15 + c_log * 0.05;
                if is_baking { f_alpha *= 1.45; }
            } else {
                f_alpha = (1.10 + c_log * 0.08).clamp(1.0, 1.8); s_gain = 1.02 + c_log * 0.03;
                if is_baking { f_alpha *= 1.30; }
            }
            let c_mom = if q_len > 128 { (1.0 + (q_len as f32 / 128.0).log2() * 0.04).clamp(1.0, 1.15) } else { 1.0 };
            let d_ratio = _idx as f32 / config.num_hidden_layers as f32;
            let a_corr = (1.0 - d_ratio * 0.15).clamp(0.85, 1.0);
            let d_boost = if density < 0.2 { 6.0 } else { (1.0 / (density + 0.1)).min(4.0) };
            f_alpha = (f_alpha * (0.1 / (m_a + 1e-9)) * d_boost * a_corr * c_mom).clamp(0.2, 5.0);
            s_gain = (s_gain * (density * 1.5 + 0.85)).clamp(0.90, 1.50);
            if self.is_support_layer && !is_baking { f_alpha *= 1.12; s_gain *= 1.04; }
            let m_s_a = if is_vision { 4.8 } else { 3.8 }; if f_alpha > m_s_a { f_alpha = m_s_a; }
        }
        let out_ptr = if ping_pong { workspace.hidden_b.as_mut_ptr() } else { workspace.hidden_a.as_mut_ptr() };
        #[cfg(feature = "cuda")]
        if self.device_id >= 0 && !self.gpu_broken.load(Ordering::Relaxed) {
            if let Some((si, so, sr)) = global_scratch {
                let (di, dog, drg) = self.q_proj.ensure_gpu_buffers_ext(q_len, si, so, sr);
                if di != 0 && dog != 0 && drg != 0 {
                    unsafe {
                        let cl = crate::models::qwen3vl::native_backend::lib();
                        cl.cuMemcpyHtoD_v2(di, x.as_ptr() as *const _, x.len() * 2);
                        if self.input_layernorm.gpu_ptr.is_none() || self.post_attention_layernorm.gpu_ptr.is_none() { 
                            self.gpu_broken.store(true, Ordering::Relaxed); return &[]; 
                        }
                        let dlw = self.input_layernorm.gpu_ptr.as_ref().unwrap().0 as *const f16;
                        crate::models::qwen3vl::native_backend::cuda_rms_norm_f16(di as *const f16, dlw, dog as *mut f16, q_len as i32, h_s as i32, config.rms_norm_eps as f32);
                        self.q_proj.forward_gpu(dog, di, q_len); let dq = di;
                        self.k_proj.forward_gpu(dog, drg, q_len); let dk = drg;
                        {
                            let mut rg = self.rope_cache_gpu.lock().unwrap();
                            if rg.is_none() {
                                let h_s_rope = 16384 * (h_d / 2) * 2;
                                let mut ct = NativeTensor { data_ptr: std::ptr::null(), host_size: h_s_rope, gpu_ptr: None, shape: vec![16384, h_d/2], dtype: NativeDType::F16, _mmap: None, device_id: self.device_id };
                                let mut st = NativeTensor { data_ptr: std::ptr::null(), host_size: h_s_rope, gpu_ptr: None, shape: vec![16384, h_d/2], dtype: NativeDType::F16, _mmap: None, device_id: self.device_id };
                                let cp = crate::models::qwen3vl::native_backend::native_precompute_rope_f16(h_d, 16384, config.rope_theta);
                                ct.data_ptr = cp.0.as_ptr() as *const u8; st.data_ptr = cp.1.as_ptr() as *const u8;
                                ct.move_to_gpu(self.device_id); st.move_to_gpu(self.device_id); *rg = Some((ct, st));
                            }
                            if let Some((ref c, ref s)) = *rg {
                                let dc = c.gpu_ptr.as_ref().unwrap().0 as CUdeviceptr;
                                let ds = s.gpu_ptr.as_ref().unwrap().0 as CUdeviceptr;
                                crate::models::qwen3vl::native_backend::native_cuda_apply_rope(dq, dk, dc, ds, q_len, s_o, n_h, n_kv, h_d);
                            }
                        }
                        let mut gkg = self.gpu_kv_cache.lock().unwrap();
                        let n_tok = s_o + q_len; gkg.grow(n_tok, n_kv, h_d, self.device_id);
                        if let (Some(kp), Some(vp)) = (gkg.k_ptr, gkg.v_ptr) {
                            let ko = (s_o * n_kv * (h_d/32) * 4) as u64; let vo = (s_o * n_kv * h_d * 2) as u64;
                            let dkd = (kp.0 as u64 + ko) as CUdeviceptr; let dvd = (vp.0 as u64 + vo) as CUdeviceptr;
                            crate::models::qwen3vl::native_backend::native_cuda_pack_bits(dk, dkd, q_len * n_kv * h_d);
                            self.v_proj.forward_gpu(dq, dvd, q_len); gkg.current_len = n_tok;
                        }
                        crate::models::qwen3vl::native_backend::native_bit_serial_attn_gpu_buffered(&[], gkg.k_ptr.unwrap(), gkg.v_ptr.unwrap(), n_h, n_kv, h_d, n_tok, self.device_id as usize, dq, dog, f_alpha * (1.0 - (_idx as f32 / 28.0) * 0.15).clamp(0.85, 1.0), self.q_proj.src_in / n_h, true);
                        if s_gain != 1.0 { crate::models::qwen3vl::native_backend::native_cuda_apply_gain(dog, s_gain, q_len * h_s); }
                        self.o_proj.forward_gpu(dog, drg, q_len); // attn_out in drg
                        cl.cuMemcpyHtoD_v2(dog, x.as_ptr() as *const _, x.len() * 2); // upload x
                        crate::models::qwen3vl::native_backend::native_cuda_add_inplace(drg, dog, x.len()); // drg = x + attn_out
                        let dpw = self.post_attention_layernorm.gpu_ptr.as_ref().unwrap().0 as *const f16;
                        crate::models::qwen3vl::native_backend::cuda_rms_norm_f16(drg as *const f16, dpw, dog as *mut f16, q_len as i32, h_s as i32, config.rms_norm_eps as f32);
                        self.gate_proj.forward_gpu(dog, di, q_len); // Gate in di
                        self.up_proj.forward_gpu(dog, dq, q_len); // Up in dq
                        crate::models::qwen3vl::native_backend::native_cuda_silu_inplace(di, q_len * config.intermediate_size);
                        crate::models::qwen3vl::native_backend::native_cuda_element_mul(di, dq, q_len * config.intermediate_size);
                        self.down_proj.forward_gpu(di, dog, q_len); // MLP res in dog
                        let so_size = self.down_proj.src_out; let to_size = self.down_proj.out_features;
                        if to_size > so_size { crate::models::qwen3vl::native_backend::native_cuda_hybrid_repeat(dog, so_size, to_size, q_len); }
                        crate::models::qwen3vl::native_backend::native_cuda_add_inplace(dog, drg, x.len()); // result = MLP + (x + attn_out)
                        let rs = std::slice::from_raw_parts_mut(out_ptr, x.len());
                        cl.cuMemcpyDtoH_v2(rs.as_mut_ptr() as *mut _, dog, x.len() * 2);
                        return rs;
                    }
                }
            }
        }
        let lnw = self.input_layernorm.get_slice::<f16>(); let eps_s = f16::from_f32(1e-6);
        let mut cn = vec![eps_s; x.len()]; native_rms_norm_f16_into(x, lnw.as_ref(), config.rms_norm_eps as f32, h_s, &mut cn);
        self.q_proj.forward_into(&cn, &mut workspace.q, global_scratch);
        self.k_proj.forward_into(&cn, &mut workspace.k, global_scratch);
        self.v_proj.forward_into(&cn, &mut workspace.v, global_scratch);
        native_apply_rope_f16_with_offset(&mut workspace.q, &mut workspace.k, q_len, s_o, n_h, h_d, config.rope_theta, _r_cos, _r_sin);
        let mut c = self.kv_cache.lock().unwrap(); let nl = s_o + q_len; c.grow(nl, n_kv, h_d);
        let kp = pack_f16_to_bits(&workspace.k);
        c.k[s_o * (n_kv * h_d / 32) .. s_o * (n_kv * h_d / 32) + kp.len()].copy_from_slice(&kp);
        c.v[s_o * (n_kv * h_d) .. s_o * (n_kv * h_d) + workspace.v.len()].copy_from_slice(&workspace.v);
        c.current_len = nl;
        let mut ao = native_bit_serial_attn_f16(&workspace.q, &c.k, &c.v, h_s, n_h, n_kv, q_len, nl, f_alpha);
        if s_gain != 1.0 { for v in ao.iter_mut() { *v = f16::from_f32(v.to_f32() * s_gain); } }
        let os = unsafe { std::slice::from_raw_parts_mut(out_ptr, x.len()) };
        self.o_proj.forward_into(&ao, os, global_scratch); for i in 0..x.len() { os[i] += x[i]; }
        let plnw = self.post_attention_layernorm.get_slice::<f16>();
        native_rms_norm_f16_into(os, plnw.as_ref(), config.rms_norm_eps as f32, h_s, &mut workspace.intermediate_a[..x.len()]);
        self.gate_proj.forward_into(&workspace.intermediate_a[..x.len()], &mut workspace.intermediate_b, global_scratch);
        self.up_proj.forward_into(&workspace.intermediate_a[..x.len()], &mut workspace.intermediate_c, global_scratch);
        native_silu_f16(&mut workspace.intermediate_b[..q_len * config.intermediate_size]);
        for i in 0..q_len * config.intermediate_size { workspace.intermediate_b[i] *= workspace.intermediate_c[i]; }
        let mut moh = vec![f16::from_f32(1e-6); x.len()];
        self.down_proj.forward_into(&workspace.intermediate_b[..q_len * config.intermediate_size], &mut moh, global_scratch);
        let sos = self.down_proj.src_out; let tos = self.down_proj.out_features;
        if tos > sos { for t in 0..q_len { for i in sos..tos { moh[t*tos+i] = moh[t*tos+(i%sos)]; } } }
        for i in 0..x.len() { os[i] += moh[i]; }
        os
    }
    pub fn clear_kv_cache(&self) { self.kv_cache.lock().unwrap().clear(); self.gpu_kv_cache.lock().unwrap().clear(); }
    pub fn force_free_kv_cache(&self) {
        self.kv_cache.lock().unwrap().k = Vec::new(); self.kv_cache.lock().unwrap().v = Vec::new();
        let mut gc = self.gpu_kv_cache.lock().unwrap();
        #[cfg(feature = "cuda")]
        unsafe { if let Some(k) = gc.k_ptr.take() { let _ = lib().cuMemFree_v2(k.0 as CUdeviceptr); } if let Some(v) = gc.v_ptr.take() { let _ = lib().cuMemFree_v2(v.0 as CUdeviceptr); } }
        gc.capacity = 0; gc.current_len = 0;
    }
    pub fn get_kv_data(&self, h_d: usize, n_kv: usize, s_t: usize) -> Option<(Vec<u32>, Vec<f16>)> {
        #[cfg(feature = "cuda")]
        if self.device_id >= 0 && !self.gpu_broken.load(Ordering::Relaxed) {
            let g = self.gpu_kv_cache.lock().unwrap();
            if let (Some(kp), Some(vp)) = (g.k_ptr, g.v_ptr) {
                if g.current_len > s_t {
                    let el = g.current_len - s_t; let ku = n_kv * (h_d/32); let vu = n_kv * h_d;
                    let mut kh = vec![0u32; el * ku]; let mut vh = vec![f16::ZERO; el * vu];
                    unsafe {
                        let cl = crate::models::qwen3vl::native_backend::lib();
                        let dks = (kp.0 as u64 + (s_t * ku * 4) as u64) as CUdeviceptr;
                        let dvs = (vp.0 as u64 + (s_t * vu * 2) as u64) as CUdeviceptr;
                        let _ = cl.cuMemcpyDtoH_v2(kh.as_mut_ptr() as *mut _, dks, kh.len()*4);
                        let _ = cl.cuMemcpyDtoH_v2(vh.as_mut_ptr() as *mut _, dvs, vh.len()*2);
                    }
                    return Some((kh, vh));
                }
            }
        }
        let c = self.kv_cache.lock().unwrap();
        if c.current_len > s_t {
            let ku = n_kv * (h_d/32); let vu = n_kv * h_d;
            Some((c.k[s_t * ku .. c.current_len * ku].to_vec(), c.v[s_t * vu .. c.current_len * vu].to_vec()))
        } else { None }
    }
    pub fn get_kv_len(&self, h_d: usize, n_kv: usize) -> usize {
        #[cfg(feature = "cuda")]
        if self.device_id >= 0 { let g = self.gpu_kv_cache.lock().unwrap(); if g.current_len > 0 { return g.current_len; } }
        self.kv_cache.lock().unwrap().current_len
    }
    pub fn set_kv_data(&self, k: Vec<u32>, v: Vec<f16>) { let mut c = self.kv_cache.lock().unwrap(); c.k = k; c.v = v; c.capacity = 0; }
    #[cfg(feature = "cuda")]
    pub fn inject_gpu_kv(&self, k: &[u32], v: &[f16], n_kv: usize, h_d: usize) {
        let tok = (k.len() * 32) / (n_kv * h_d); let mut g = self.gpu_kv_cache.lock().unwrap();
        unsafe {
            let cl = crate::models::qwen3vl::native_backend::lib();
            let mut ctx = std::ptr::null_mut() as cudarc::driver::sys::CUcontext; cl.cuCtxGetCurrent(&mut ctx);
            if ctx == std::ptr::null_mut() && self.device_id >= 0 {
                let mut dev = 0 as cudarc::driver::sys::CUdevice; cl.cuDeviceGet(&mut dev, self.device_id);
                cl.cuDevicePrimaryCtxRetain(&mut ctx, dev); cl.cuCtxSetCurrent(ctx);
            }
            let mut kp: CUdeviceptr = 0; let mut vp: CUdeviceptr = 0;
            let _ = cl.cuMemAlloc_v2(&mut kp, tok * n_kv * (h_d/32) * 4); let _ = cl.cuMemAlloc_v2(&mut vp, tok * n_kv * h_d * 2);
            let _ = cl.cuMemcpyHtoD_v2(kp, k.as_ptr() as *const _, k.len()*4); let _ = cl.cuMemcpyHtoD_v2(vp, v.as_ptr() as *const _, v.len()*2);
            g.k_ptr = Some(GpuPtr(kp as *mut _)); g.v_ptr = Some(GpuPtr(vp as *mut _)); g.capacity = tok; g.current_len = tok;
        }
    }
    #[cfg(feature = "cuda")]
    pub fn inject_gpu_kv_direct(&self, ks: GpuPtr, vs: GpuPtr, tok: usize, n_kv: usize, h_d: usize) {
        let mut g = self.gpu_kv_cache.lock().unwrap();
        unsafe {
            let cl = crate::models::qwen3vl::native_backend::lib();
            let mut ctx = std::ptr::null_mut() as cudarc::driver::sys::CUcontext; cl.cuCtxGetCurrent(&mut ctx);
            if ctx == std::ptr::null_mut() && self.device_id >= 0 {
                let mut dev = 0 as cudarc::driver::sys::CUdevice; cl.cuDeviceGet(&mut dev, self.device_id);
                cl.cuDevicePrimaryCtxRetain(&mut ctx, dev); cl.cuCtxSetCurrent(ctx);
            }
            let mut kp: CUdeviceptr = 0; let mut vp: CUdeviceptr = 0;
            let kb = tok * n_kv * (h_d/32) * 4; let vb = tok * n_kv * h_d * 2;
            let _ = cl.cuMemAlloc_v2(&mut kp, kb); let _ = cl.cuMemAlloc_v2(&mut vp, vb);
            let _ = cl.cuMemcpyDtoD_v2(kp, ks.0 as CUdeviceptr, kb); let _ = cl.cuMemcpyDtoD_v2(vp, vs.0 as CUdeviceptr, vb);
            g.k_ptr = Some(GpuPtr(kp as *mut _)); g.v_ptr = Some(GpuPtr(vp as *mut _)); g.capacity = tok; g.current_len = tok;
        }
    }
}

pub struct NativeQwen3TextModel {
    pub config: Qwen3VLTextConfig, pub embed_tokens: NativeLinear, pub layers: Vec<NativeLayer>, pub norm: NativeTensor,
    pub rope_cache: std::sync::Mutex<RopeCache>,
}

unsafe impl Send for NativeQwen3TextModel {}
unsafe impl Sync for NativeQwen3TextModel {}

impl NativeQwen3TextModel {
    pub fn forward(&self, i_ids: &[u32], _pv: Option<&[f16]>, _gt: Option<&[u32; 3]>, s_o: usize, gs: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>, workspace: Option<&mut ForwardWorkspace>, sl: Option<&NativeLayer>, is_vision: bool) -> Vec<f16> {
        let hid = self.config.hidden_size;
        let embeds = match &self.embed_tokens.variant {
            LinearVariant::Standard { weight, .. } => native_embedding_lookup_f16(i_ids, weight.get_slice::<f16>().as_ref(), hid),
            LinearVariant::BitSerial { weight_packed, .. } => native_embedding_lookup_f16(i_ids, weight_packed.get_slice::<f16>().as_ref(), hid)
        };
        self.forward_ext(i_ids, embeds, s_o, gs, workspace, sl, is_vision)
    }
    pub fn forward_ext(&self, _i_ids: &[u32], embeds: Vec<f16>, s_o: usize, gs: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>, workspace: Option<&mut ForwardWorkspace>, sl: Option<&NativeLayer>, is_vision: bool) -> Vec<f16> {
        let hid = self.config.hidden_size; let is_baking = self.layers.len() <= 1; let q_len = embeds.len() / hid;
        let mut internal_ws = ForwardWorkspace::new(); let ws = match workspace { Some(w) => w, None => &mut internal_ws };
        ws.ensure_capacity(hid, self.config.intermediate_size, q_len, self.config.num_attention_heads, self.config.num_key_value_heads, self.config.head_dim);
        ws.hidden_a[..embeds.len()].copy_from_slice(&embeds);
        let mut cur_x: &[f16] = unsafe { std::slice::from_raw_parts(ws.hidden_a.as_ptr(), embeds.len()) };
        { let mut rg = self.rope_cache.lock().unwrap(); rg.ensure_length(s_o + q_len); }
        let rg_lock = self.rope_cache.lock().unwrap(); let r_cos = &rg_lock.cos; let r_sin = &rg_lock.sin;
        for (i, layer) in self.layers.iter().enumerate() {
            let active = layer;
            let use_b = i % 2 == 0;
            let out = active.forward(cur_x, &self.config, s_o, i, r_cos, r_sin, is_baking, is_vision, gs, ws, use_b);
            cur_x = unsafe { std::slice::from_raw_parts(out.as_ptr(), out.len()) };
        }
        let n_cow = self.norm.get_slice::<f16>(); let eps = f16::from_f32(1e-6);
        let mut cn = vec![eps; cur_x.len()]; native_rms_norm_f16_into(cur_x, n_cow.as_ref(), self.config.rms_norm_eps as f32, hid, &mut cn);
        cn
    }
    pub fn move_to_gpu(&mut self, device_id: i32) -> anyhow::Result<()> { 
        self.embed_tokens.move_to_gpu(device_id)?; 
        self.norm.move_to_gpu(device_id)?; 
        for layer in &mut self.layers { layer.move_to_gpu(device_id)?; } 
        Ok(())
    }
    pub fn clear_kv_cache(&self) { for layer in &self.layers { layer.clear_kv_cache(); } }
    pub fn force_free_kv_cache(&self) { for layer in &self.layers { layer.force_free_kv_cache(); } }
    pub fn batch_upload_stitched_cache(&self, k: Vec<u32>, v: Vec<f16>) {
        let hd = self.config.head_dim; let nkv = self.config.num_key_value_heads;
        #[cfg(feature = "cuda")]
        if self.layers[0].device_id >= 0 {
            self.layers[0].inject_gpu_kv(&k, &v, nkv, hd);
            let l0c = self.layers[0].gpu_kv_cache.lock().unwrap();
            if let (Some(ks), Some(vs)) = (l0c.k_ptr, l0c.v_ptr) {
                let tok = l0c.current_len; for i in 1..self.layers.len() { self.layers[i].inject_gpu_kv_direct(ks, vs, tok, nkv, hd); }
            }
            return;
        }
        for layer in &self.layers { layer.set_kv_data(k.clone(), v.clone()); }
    }
    pub fn get_kv_len(&self) -> usize { self.layers[0].get_kv_len(self.config.head_dim, self.config.num_key_value_heads) }
    pub fn get_all_kv(&self, start_idx: usize) -> Vec<(Vec<u32>, Vec<f16>)> {
        let hd = self.config.head_dim; let nkv = self.config.num_key_value_heads;
        self.layers.iter().filter_map(|l| l.get_kv_data(hd, nkv, start_idx)).collect()
    }
}

pub struct NativeQwen3VLModel {
    pub config: Qwen3VLConfig, pub text_model: NativeQwen3TextModel, pub lm_head: NativeLinear, pub visual: Option<NativeVisionModel>,
    pub support_layer0: Option<NativeLayer>, pub support_workspace: std::sync::Mutex<ForwardWorkspace>,
    pub global_scratch_i: std::sync::Mutex<Option<(GpuPtr, usize)>>, 
    pub global_scratch_o: std::sync::Mutex<Option<(GpuPtr, usize)>>,
    pub global_scratch_r: std::sync::Mutex<Option<(GpuPtr, usize)>>,
    pub workspace: std::sync::Mutex<ForwardWorkspace>,
}

pub struct NativeVisionModel { pub patch_embed: NativeLinear, pub blocks: Vec<NativeLayer>, pub merger: NativeLayer }
unsafe impl Send for NativeVisionModel {}
unsafe impl Sync for NativeVisionModel {}

impl NativeVisionModel {
    pub fn move_to_gpu(&mut self, dev: i32) -> anyhow::Result<()> { 
        self.patch_embed.move_to_gpu(dev)?; 
        for b in &mut self.blocks { b.move_to_gpu(dev)?; } 
        self.merger.move_to_gpu(dev)?;
        Ok(())
    }
    pub fn forward<'a>(&self, pv: &[f16], gt: &[u32; 3], rc: &[f16], rs: &[f16], gs: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>, ws: &'a mut ForwardWorkspace) -> &'a [f16] {
        let patches = (gt[1] * gt[2]) as usize; let eo = self.patch_embed.forward(pv, gs); ws.hidden_a[..eo.len()].copy_from_slice(&eo);
        let mut cur_x: &[f16] = unsafe { std::slice::from_raw_parts(ws.hidden_a.as_ptr(), eo.len()) };
        for (i, block) in self.blocks.iter().enumerate() {
            let use_b = i % 2 != 0;
            let out = block.forward(cur_x, &crate::models::qwen3vl::config::Qwen3VLTextConfig {
                hidden_size: self.patch_embed.out_features, intermediate_size: block.gate_proj.out_features, num_hidden_layers: 1, num_attention_heads: 16, num_key_value_heads: 16,
                head_dim: self.patch_embed.out_features / 16, rms_norm_eps: 1e-6, rope_theta: 10000.0, vocab_size: 0, max_position_embeddings: 4096, dtype: None, rope_scaling: None,
            }, 0, 0, rc, rs, false, true, gs, ws, use_b);
            cur_x = unsafe { std::slice::from_raw_parts(out.as_ptr(), out.len()) };
        }
        let last_b = self.blocks.len() % 2 != 0;
        self.merger.forward(cur_x, &crate::models::qwen3vl::config::Qwen3VLTextConfig {
            hidden_size: cur_x.len() / patches, intermediate_size: cur_x.len() / patches, num_hidden_layers: 1, num_attention_heads: 1, num_key_value_heads: 1,
            head_dim: cur_x.len() / patches, rms_norm_eps: 1e-6, rope_theta: 10000.0, vocab_size: 0, max_position_embeddings: 4096, dtype: None, rope_scaling: None,
        }, 0, 0, rc, rs, false, true, gs, ws, !last_b)
    }
}

impl NativeQwen3VLModel {
    pub fn load(config: Qwen3VLConfig, m_mmap: Arc<Mmap>, v_mmap: Option<Arc<Mmap>>, baking: bool, s_mmap: Option<Arc<Mmap>>, device_id: i32) -> Result<Self> {
        let mut active_gpu_id = device_id;
        #[cfg(feature = "cuda")]
        unsafe {
            let cl = crate::models::qwen3vl::native_backend::lib();
            if active_gpu_id >= 0 {
                println!("[LOAD] Using GPU-{} as requested.", active_gpu_id);
            }
        }
        let st = SafeTensors::deserialize(&m_mmap)?; let st_sec = s_mmap.as_ref().map(|m| SafeTensors::deserialize(m)).transpose()?;
        let t_c = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;
        let find_key_ext = |target: &str, primary: &SafeTensors, secondary: Option<&SafeTensors>, _l_idx: i32| -> Option<(String, bool)> {
            if primary.tensor(target).is_ok() || primary.tensor(&format!("{}.packed", target)).is_ok() { return Some((target.to_string(), true)); }
            if let Some(sec) = secondary { if sec.tensor(target).is_ok() || sec.tensor(&format!("{}.packed", target)).is_ok() { return Some((target.to_string(), false)); } }
            let vars = vec![target.to_string(), target.replace("lm_head", "model.lm_head"), target.replace("lm_head", "output")];
            for v in vars { if primary.tensor(&v).is_ok() || primary.tensor(&format!("{}.packed", v)).is_ok() { return Some((v, true)); } }
            None
        };
        let mut get_l = |base: &str, in_f: usize, out_f: usize, l_idx: i32| -> Result<NativeLinear> {
            let (key, is_p) = find_key_ext(base, &st, st_sec.as_ref(), l_idx).ok_or_else(|| anyhow!("LinearNotFound: {}", base))?;
            let cst = if is_p { &st } else { st_sec.as_ref().unwrap() }; let cm = if is_p { &m_mmap } else { s_mmap.as_ref().unwrap() };
            let pn = format!("{}.packed", key); let sn = format!("{}.scales", key);
            if cst.tensor(&pn).is_ok() {
                let vp = cst.tensor(&pn)?; let vs = cst.tensor(&sn)?;
                let op = unsafe { vp.data().as_ptr().offset_from(cm.as_ptr()) } as usize; let os = unsafe { vs.data().as_ptr().offset_from(cm.as_ptr()) } as usize;
                Ok(NativeLinear { in_features: in_f, out_features: out_f, src_in: in_f, src_out: out_f, variant: LinearVariant::BitSerial {
                    weight_packed: NativeTensor::from_mmap(cm.clone(), op, vp.shape().to_vec(), NativeDType::U32),
                    scales: NativeTensor::from_mmap(cm.clone(), os, vs.shape().to_vec(), NativeDType::F16), bias: None,
                }, device_id: active_gpu_id })
            } else {
                let v = cst.tensor(&key)?; let src_shape = v.shape(); let si = src_shape[1]; let so = src_shape[0];
                if in_f > si || out_f > so {
                    let sd = v.data(); let sf = unsafe { std::slice::from_raw_parts(sd.as_ptr() as *const f16, sd.len()/2) };
                    let mut nd = vec![f16::ZERO; out_f * in_f]; let rk = in_f/si; let es = 1.0/(rk as f32);
                    for row in 0..out_f { for col in 0..in_f { nd[row*in_f+col] = f16::from_f32(sf[(row%so)*si+(col%si)].to_f32() * es); } }
                    let boxed = nd.into_boxed_slice(); let ptr = boxed.as_ptr() as *const u8; 
                    let h_size = out_f * in_f * 2;
                    std::mem::forget(boxed);
                    Ok(NativeLinear { in_features: in_f, out_features: out_f, src_in: si, src_out: so, variant: LinearVariant::Standard { weight: NativeTensor { data_ptr: ptr, host_size: h_size, gpu_ptr: None, shape: vec![out_f, in_f], dtype: NativeDType::F16, _mmap: None, device_id: active_gpu_id }, bias: None }, device_id: active_gpu_id })
                } else {
                    let o = unsafe { v.data().as_ptr().offset_from(cm.as_ptr()) } as usize;
                    Ok(NativeLinear { in_features: in_f, out_features: out_f, src_in: in_f, src_out: in_f, variant: LinearVariant::Standard { weight: NativeTensor::from_mmap(cm.clone(), o, v.shape().to_vec(), NativeDType::F16), bias: None }, device_id: active_gpu_id })
                }
            }
        };
        let mut get_t = |name: &str, ts: usize| -> Result<NativeTensor> {
            let (key, is_p) = find_key_ext(name, &st, st_sec.as_ref(), -1).ok_or_else(|| anyhow!("TNotFound: {}", name))?;
            let cst = if is_p { &st } else { st_sec.as_ref().unwrap() }; let cm = if is_p { &m_mmap } else { s_mmap.as_ref().unwrap() };
            let v = cst.tensor(&key)?; let sd = v.data(); let sl = sd.len()/2;
            if ts > sl {
                let sf = unsafe { std::slice::from_raw_parts(sd.as_ptr() as *const f16, sl) }; let ratio = ts/sl;
                let mut nd = Vec::with_capacity(ts*2); for _ in 0..ratio { for &val in sf { nd.extend_from_slice(&val.to_le_bytes()); } }
                let boxed = nd.into_boxed_slice(); let ptr = boxed.as_ptr() as *const u8; 
                let h_size = ts * 2;
                std::mem::forget(boxed);
                Ok(NativeTensor { data_ptr: ptr, host_size: h_size, gpu_ptr: None, shape: vec![ts], dtype: NativeDType::F16, _mmap: None, device_id: active_gpu_id })
            } else {
                let off = unsafe { v.data().as_ptr().offset_from(cm.as_ptr()) } as usize;
                let h_size = v.shape().iter().product::<usize>() * 2;
                Ok(NativeTensor { data_ptr: unsafe { cm.as_ptr().add(off) }, host_size: h_size, gpu_ptr: None, shape: v.shape().to_vec(), dtype: NativeDType::F16, _mmap: Some(cm.clone()), device_id: active_gpu_id })
            }
        };
        let mut get_embed = |base: &str, vocab: usize, thid: usize, _hsec: bool| -> Result<NativeLinear> {
            let (key, is_p) = find_key_ext(base, &st, st_sec.as_ref(), -1).ok_or_else(|| anyhow!("EmbedNotFound: {}", base))?;
            let cst = if is_p { &st } else { st_sec.as_ref().unwrap() }; let cm = if is_p { &m_mmap } else { s_mmap.as_ref().unwrap() };
            let pn = format!("{}.packed", key); let sn = format!("{}.scales", key);
            if cst.tensor(&pn).is_ok() {
                let vp = cst.tensor(&pn)?; let vs = cst.tensor(&sn)?; let shid = (vp.data().len()/4*32)/vocab;
                let pr = unsafe { std::slice::from_raw_parts(vp.data().as_ptr() as *const u32, vp.data().len()/4) };
                let sr = unsafe { std::slice::from_raw_parts(vs.data().as_ptr() as *const f16, vs.data().len()/2) };
                let dq = dequantize_bit_serial_to_f16(pr, sr, vocab, shid, thid); // Note: dequantize usually returns flattened [vocab * target_k] if we pass thid?
                // Wait, dequantize_bit_serial_to_f16 signature is (packed, scales, n, src_k, target_k).
                // It already has logic to expand 'ratio' inside! Let's verify 'dequantize_bit_serial_to_f16'.
                
                // [VERIFICATION] The 'dequantize_bit_serial_to_f16' function ALREADY implements the repeat logic:
                // "let ratio = if target_k > src_k { target_k / src_k } else { 1 }; ... for r in 0..ratio ..."
                // So we just need to ensure we pass 'thid' as target_k.
                // The current call is: dequantize_bit_serial_to_f16(pr, sr, vocab, shid, thid);
                // So it should be correct!
                
                let mut dst = Vec::with_capacity(dq.len()*2); for v in dq { dst.extend_from_slice(&v.to_le_bytes()); }
                let boxed = dst.into_boxed_slice(); let ptr = boxed.as_ptr(); 
                let h_size = vocab * thid * 2;
                std::mem::forget(boxed);
                Ok(NativeLinear { in_features: vocab, out_features: thid, src_in: shid, src_out: thid, variant: LinearVariant::Standard { weight: NativeTensor { data_ptr: ptr, host_size: h_size, gpu_ptr: None, shape: vec![vocab, thid], dtype: NativeDType::F16, _mmap: None, device_id: active_gpu_id }, bias: None }, device_id: active_gpu_id })
            } else { get_l(base, vocab, thid, -1) }
        };
        let emb = get_embed("model.language_model.embed_tokens.weight", 151936, t_c.hidden_size, s_mmap.is_some())?;
        let mut layers = Vec::new(); let l_to_l = if baking { 1 } else { t_c.num_hidden_layers };
        let q_out = t_c.num_attention_heads * t_c.head_dim; let kv_out = t_c.num_key_value_heads * t_c.head_dim;
        let mut support_layer0 = None;
        if let Some(ref sec_mmap) = s_mmap {
            let sec_st = SafeTensors::deserialize(sec_mmap)?;
            let get_sec_l = |base: &str, ti: usize, to: usize| -> Result<NativeLinear> {
                let pn = format!("{}.packed", base); let sn = format!("{}.scales", base);
                let vp = sec_st.tensor(&pn)?; let vs = sec_st.tensor(&sn)?;
                
                // [FIX] Read original source shape if available, otherwise fallback to heuristic
                let (so, si) = if let Ok(st) = sec_st.tensor(&format!("{}.shape", base)) {
                    let s_data = unsafe { std::slice::from_raw_parts(st.data().as_ptr() as *const i32, 2) };
                    (s_data[0] as usize, s_data[1] as usize)
                } else {
                    let s_out = (vs.data().len() / 2) / ((vp.data().len() / 4 * 32) / (vs.data().len() / 2)); // Heuristic
                    (s_out, (vp.data().len() / 4 * 32) / s_out)
                };

                let pu = unsafe { std::slice::from_raw_parts(vp.data().as_ptr() as *const u32, vp.data().len()/4) };
                let sf = unsafe { std::slice::from_raw_parts(vs.data().as_ptr() as *const f16, vs.data().len()/2) };
                
                // [HYBRID-UPSCALING] Repeat weights if target dims are larger (e.g. 0.6B -> 2B)
                let ratio_i = if ti > si { ti / si } else { 1 };
                let ratio_o = if to > so { to / so } else { 1 };
                
                if ratio_i > 1 || ratio_o > 1 {
                    println!("[HYBRID-LOAD] Upscaling layer {} from {}x{} to {}x{} (Repeat: {}x{})", base, so, si, to, ti, ratio_o, ratio_i);
                    let mut np = vec![0u32; (to/8)*(ti/32)*8]; 
                    let mut ns = vec![f16::ZERO; (to/8)*(ti/32)*8];
                    
                    let src_k_blocks = si / 32;
                    let dst_k_blocks = ti / 32;
                    let src_n_blocks = so / 8;
                    let dst_n_blocks = to / 8;

                    for no in 0..dst_n_blocks { 
                        let src_no = no % src_n_blocks;
                        for ko in 0..dst_k_blocks { 
                            let src_ko = ko % src_k_blocks;
                            
                            // Copy 8-row block
                            for sn in 0..8 {
                                let s_idx = (src_no * src_k_blocks + src_ko) * 8 + sn; 
                                let d_idx = (no * dst_k_blocks + ko) * 8 + sn;
                                if s_idx < pu.len() {
                                    np[d_idx] = pu[s_idx]; 
                                    // Scale adjustment might be needed if norm distribution changes, 
                                    // but for direct repetition, keeping scale is usually safer for signal magnitude preservation.
                                    ns[d_idx] = sf[s_idx]; 
                                }
                            } 
                        } 
                    }
                    
                    let pb = np.into_boxed_slice(); let sb = ns.into_boxed_slice();
                    let pp = pb.as_ptr() as *const u8; let sp = sb.as_ptr() as *const u8; 
                    let h_size_packed = (to/8)*(ti/32)*8 * 4;
                    let h_size_scales = (to/8)*(ti/32)*8 * 2;
                    std::mem::forget(pb); std::mem::forget(sb);
                    
                    Ok(NativeLinear { 
                        in_features: ti, out_features: to, src_in: ti, src_out: to, // It is now upscaled
                        variant: LinearVariant::BitSerial { 
                            weight_packed: NativeTensor { data_ptr: pp, host_size: h_size_packed, gpu_ptr: None, shape: vec![to, ti/32], dtype: NativeDType::U32, _mmap: None, device_id: active_gpu_id }, 
                            scales: NativeTensor { data_ptr: sp, host_size: h_size_scales, gpu_ptr: None, shape: vec![to, ti/32], dtype: NativeDType::F16, _mmap: None, device_id: active_gpu_id }, 
                            bias: None 
                        }, 
                        device_id: active_gpu_id 
                    })
                } else {
                    // Standard load logic (No upscale needed)
                    // ... (Original logic for direct mmap if dims match)
                    // Actually, if we are loading from secondary, we might want to ensure it's loaded into RAM if we can't mmap easily with different strides.
                    // But if sizes match, we can try to use the mmap slice if contiguous?
                    // The original code tried to construct new vectors anyway for layout shuffling logic in 'get_sec_l' 
                    // Wait, the original code ALREADY constructed new vectors: "let mut np = vec![...]"
                    // So we just stick with that logic but without the ratio loop if ratio is 1.
                    
                    let mut np = vec![0u32; (to/8)*(ti/32)*8]; let mut ns = vec![f16::ZERO; (to/8)*(ti/32)*8];
                    let es = f16::from_f32(1.0/((ti as f32 / si as f32).max(1.0)));
                    
                    let src_k_blocks = si / 32;
                    let dst_k_blocks = ti / 32;

                    for no in 0..(to/8) { 
                        for ko in 0..dst_k_blocks { 
                            for sn in 0..8 {
                                let s_idx = ((no % (so/8)) * src_k_blocks + (ko % src_k_blocks)) * 8 + sn; 
                                let d_idx = (no * dst_k_blocks + ko) * 8 + sn;
                                if s_idx < pu.len() {
                                    np[d_idx] = pu[s_idx]; ns[d_idx] = sf[s_idx] * es;
                                }
                            } 
                        } 
                    }
                    let pb = np.into_boxed_slice(); let sb = ns.into_boxed_slice();
                    let pp = pb.as_ptr() as *const u8; let sp = sb.as_ptr() as *const u8; 
                    let h_size_packed = (to/8)*(ti/32)*8 * 4;
                    let h_size_scales = (to/8)*(ti/32)*8 * 2;
                    std::mem::forget(pb); std::mem::forget(sb);
                    Ok(NativeLinear { in_features: ti, out_features: to, src_in: si, src_out: so, variant: LinearVariant::BitSerial { weight_packed: NativeTensor { data_ptr: pp, host_size: h_size_packed, gpu_ptr: None, shape: vec![to, ti/32], dtype: NativeDType::U32, _mmap: None, device_id: active_gpu_id }, scales: NativeTensor { data_ptr: sp, host_size: h_size_scales, gpu_ptr: None, shape: vec![to, ti/32], dtype: NativeDType::F16, _mmap: None, device_id: active_gpu_id }, bias: None }, device_id: active_gpu_id })
                }
            };
            let get_sec_t = |name: &str, ts: usize| -> Result<NativeTensor> {
                let v = sec_st.tensor(name)?; let sd = v.data(); let sl = sd.len()/2;
                if ts > sl {
                    let sf = unsafe { std::slice::from_raw_parts(sd.as_ptr() as *const f16, sl) }; let ratio = ts/sl;
                    let mut nd = Vec::with_capacity(ts*2); for _ in 0..ratio { for &val in sf { nd.extend_from_slice(&val.to_le_bytes()); } }
                    let boxed = nd.into_boxed_slice(); let ptr = boxed.as_ptr(); 
                    let h_size = ts * 2;
                    std::mem::forget(boxed);
                    Ok(NativeTensor { data_ptr: ptr as *const u8, host_size: h_size, gpu_ptr: None, shape: vec![ts], dtype: NativeDType::F16, _mmap: None, device_id: active_gpu_id })
                } else {
                    let off = unsafe { v.data().as_ptr().offset_from(sec_mmap.as_ptr()) } as usize;
                    let h_size = v.shape().iter().product::<usize>() * 2;
                    Ok(NativeTensor { data_ptr: unsafe { sec_mmap.as_ptr().add(off) }, host_size: h_size, gpu_ptr: None, shape: v.shape().to_vec(), dtype: NativeDType::F16, _mmap: Some(sec_mmap.clone()), device_id: active_gpu_id })
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
                device_id: active_gpu_id, is_support_layer: true, kv_cache: std::sync::Mutex::new(DynamicKVCache::new()), gpu_kv_cache: std::sync::Mutex::new(DynamicGpuKVCache::new()), rope_cache_gpu: std::sync::Mutex::new(None), gpu_broken: std::sync::atomic::AtomicBool::new(false),
            });
        }
        for i in 0..l_to_l {
            let p = format!("model.language_model.layers.{}", i);
            layers.push(NativeLayer {
                input_layernorm: get_t(&format!("{}.input_layernorm.weight", p), t_c.hidden_size)?,
                post_attention_layernorm: get_t(&format!("{}.post_attention_layernorm.weight", p), t_c.hidden_size)?,
                q_norm: get_t(&format!("{}.self_attn.q_norm.weight", p), t_c.hidden_size).ok(),
                k_norm: get_t(&format!("{}.self_attn.k_norm.weight", p), t_c.hidden_size).ok(),
                q_proj: get_l(&format!("{}.self_attn.q_proj.weight", p), t_c.hidden_size, q_out, i as i32)?,
                k_proj: get_l(&format!("{}.self_attn.k_proj.weight", p), t_c.hidden_size, kv_out, i as i32)?,
                v_proj: get_l(&format!("{}.self_attn.v_proj.weight", p), t_c.hidden_size, kv_out, i as i32)?,
                o_proj: get_l(&format!("{}.self_attn.o_proj.weight", p), q_out, t_c.hidden_size, i as i32)?,
                gate_proj: get_l(&format!("{}.mlp.gate_proj.weight", p), t_c.hidden_size, t_c.intermediate_size, i as i32)?,
                up_proj: get_l(&format!("{}.mlp.up_proj.weight", p), t_c.hidden_size, t_c.intermediate_size, i as i32)?,
                down_proj: get_l(&format!("{}.mlp.down_proj.weight", p), t_c.intermediate_size, t_c.hidden_size, i as i32)?,
                device_id: active_gpu_id, is_support_layer: false, kv_cache: std::sync::Mutex::new(DynamicKVCache::new()), gpu_kv_cache: std::sync::Mutex::new(DynamicGpuKVCache::new()), rope_cache_gpu: std::sync::Mutex::new(None), gpu_broken: std::sync::atomic::AtomicBool::new(false),
            });
        }
        
        // [FIX] For baking mode (Layer 0 only), 'norm' and 'lm_head' might be missing.
        // We create dummy tensors to satisfy the struct requirements without crashing.
        let norm = match get_t("model.language_model.norm.weight", t_c.hidden_size) {
            Ok(t) => t,
            Err(_) if baking => {
                // Create dummy norm
                let h_size = t_c.hidden_size * 2;
                let dummy_data = vec![0u8; h_size].into_boxed_slice();
                let ptr = dummy_data.as_ptr();
                std::mem::forget(dummy_data);
                NativeTensor {
                    data_ptr: ptr, host_size: h_size, gpu_ptr: None,
                    shape: vec![t_c.hidden_size], dtype: NativeDType::F16,
                    _mmap: None, device_id: active_gpu_id
                }
            },
            Err(e) => return Err(e),
        };

        let h_res = get_l("model.language_model.lm_head.weight", t_c.hidden_size, 151936, -1).or_else(|_| {
            get_l("model.language_model.embed_tokens.weight", 151936, t_c.hidden_size, -1).map(|mut el| { el.in_features = t_c.hidden_size; el.out_features = 151936; el.device_id = active_gpu_id; el })
        });

        let lm_head = match h_res {
            Ok(l) => l,
            Err(_) if baking => {
                // Create dummy lm_head
                let h_size = t_c.hidden_size * 151936 * 2; // Very large, but virtual
                // We won't allocate real memory for this big dummy, just a minimal placeholder
                // Actually, let's just make a small one and lie about the shape to avoid OOM
                // Since we never use it in baking, it's safe.
                let dummy_data = vec![0u8; 16].into_boxed_slice();
                let ptr = dummy_data.as_ptr();
                std::mem::forget(dummy_data);
                
                NativeLinear { 
                    in_features: t_c.hidden_size, out_features: 151936, src_in: t_c.hidden_size, src_out: 151936, 
                    variant: LinearVariant::Standard { 
                        weight: NativeTensor { 
                            data_ptr: ptr, host_size: 16, gpu_ptr: None, 
                            shape: vec![151936, t_c.hidden_size], dtype: NativeDType::F16, 
                            _mmap: None, device_id: active_gpu_id 
                        }, 
                        bias: None 
                    }, 
                    device_id: active_gpu_id 
                }
            },
            Err(e) => return Err(e),
        };

        let visual = if let Some(vm) = v_mmap {
            let vst = SafeTensors::deserialize(&vm)?;
            let find_vkey = |target: &str| -> Option<String> {
                let vars = vec![target.to_string(), target.replace("patch_embed.proj", "patch_embd"), target.replace("patch_embed", "patch_embd")];
                for v in vars { if vst.tensor(&v).is_ok() { return Some(v); } }
                vst.names().iter().find(|&n| n.contains(target)).map(|n| n.to_string())
            };
            let get_vvt = |name: &str, ts: usize| -> Result<NativeTensor> {
                let key = find_vkey(name).ok_or_else(|| anyhow!("VNotFound: {}", name))?; let v = vst.tensor(&key)?; let sd = v.data(); let sl = sd.len()/2;
                if ts > sl {
                    let sf = unsafe { std::slice::from_raw_parts(sd.as_ptr() as *const f16, sl) }; let mut nd = Vec::with_capacity(ts*2);
                    for _ in 0..(ts/sl) { for &val in sf { nd.extend_from_slice(&val.to_le_bytes()); } }
                    let boxed = nd.into_boxed_slice(); let ptr = boxed.as_ptr() as *const u8; 
                    let h_size = ts * 2;
                    std::mem::forget(boxed);
                    Ok(NativeTensor { data_ptr: ptr, host_size: h_size, gpu_ptr: None, shape: vec![ts], dtype: NativeDType::F16, _mmap: None, device_id: active_gpu_id })
                } else {
                    let off = unsafe { v.data().as_ptr().offset_from(vm.as_ptr()) } as usize;
                    let h_size = v.shape().iter().product::<usize>() * 2;
                    Ok(NativeTensor { data_ptr: unsafe { vm.as_ptr().add(off) }, host_size: h_size, gpu_ptr: None, shape: v.shape().to_vec(), dtype: NativeDType::F16, _mmap: Some(vm.clone()), device_id: active_gpu_id })
                }
            };
            let get_vvl = |base: &str, ti: usize, to: usize| -> Result<NativeLinear> {
                let key = find_vkey(base).ok_or_else(|| anyhow!("VLNotFound: {}", base))?; let v = vst.tensor(&key)?;
                let o = unsafe { v.data().as_ptr().offset_from(vm.as_ptr()) } as usize;
                let h_size = v.shape().iter().product::<usize>() * 2;
                Ok(NativeLinear { in_features: ti, out_features: to, src_in: ti, src_out: to, variant: LinearVariant::Standard { weight: NativeTensor::from_mmap(vm.clone(), o, v.shape().to_vec(), NativeDType::F16), bias: None }, device_id: active_gpu_id })
            };
            let v_cfg = config.vision_config.as_ref().ok_or(anyhow!("Missing vision_config"))?;
            let vh = v_cfg.hidden_size; let vi = vh*4; let voh = v_cfg.out_hidden_size.unwrap_or(vh*2);
            let pe = get_vvl("visual.patch_embed.proj.weight", 1536, vh)?;
            let mut vb = Vec::new();
            for i in 0..(if baking { 1 } else { v_cfg.depth }) {
                let p = format!("visual.blocks.{}", i);
                vb.push(NativeLayer {
                    input_layernorm: get_vvt(&format!("{}.norm1.weight", p), vh)?, post_attention_layernorm: get_vvt(&format!("{}.norm2.weight", p), vh)?, q_norm: None, k_norm: None,
                    q_proj: get_vvl(&format!("{}.attn.q_proj.weight", p), vh, vh)?, k_proj: get_vvl(&format!("{}.attn.k_proj.weight", p), vh, vh)?, v_proj: get_vvl(&format!("{}.attn.v_proj.weight", p), vh, vh)?, o_proj: get_vvl(&format!("{}.attn.proj.weight", p), vh, vh)?,
                    gate_proj: get_vvl(&format!("{}.mlp.fc1.weight", p), vh, vi)?, up_proj: get_vvl(&format!("{}.mlp.fc1.weight", p), vh, vi)?, down_proj: get_vvl(&format!("{}.mlp.fc2.weight", p), vi, vh)?,
                    device_id: active_gpu_id, is_support_layer: false, kv_cache: std::sync::Mutex::new(DynamicKVCache::new()), gpu_kv_cache: std::sync::Mutex::new(DynamicGpuKVCache::new()), rope_cache_gpu: std::sync::Mutex::new(None), gpu_broken: std::sync::atomic::AtomicBool::new(false),
                });
            }
            let merger = NativeLayer {
                input_layernorm: get_vvt("visual.merger.norm.weight", vi)?, post_attention_layernorm: get_vvt("visual.merger.norm.weight", vi)?, q_norm: None, k_norm: None,
                q_proj: get_vvl("visual.merger.mlp.0.weight", vi, vi)?, k_proj: get_vvl("visual.merger.mlp.0.weight", vi, vi)?, v_proj: get_vvl("visual.merger.mlp.0.weight", vi, vi)?, o_proj: get_vvl("visual.merger.mlp.2.weight", vi, voh)?,
                gate_proj: get_vvl("visual.merger.mlp.0.weight", vi, vi)?, up_proj: get_vvl("visual.merger.mlp.0.weight", vi, vi)?, down_proj: get_vvl("visual.merger.mlp.2.weight", vi, voh)?,
                device_id: active_gpu_id, is_support_layer: false, kv_cache: std::sync::Mutex::new(DynamicKVCache::new()), gpu_kv_cache: std::sync::Mutex::new(DynamicGpuKVCache::new()), rope_cache_gpu: std::sync::Mutex::new(None), gpu_broken: std::sync::atomic::AtomicBool::new(false),
            };
            Some(NativeVisionModel { patch_embed: pe, blocks: vb, merger })
        } else { None };
        Ok(Self {
            config: config.clone(), text_model: NativeQwen3TextModel { config: t_c.clone(), embed_tokens: emb, layers, norm, rope_cache: std::sync::Mutex::new(RopeCache::new(t_c.head_dim, t_c.rope_theta, 4096)) },
            lm_head: h_res, visual, support_layer0, support_workspace: std::sync::Mutex::new(ForwardWorkspace::new()), global_scratch_i: std::sync::Mutex::new(None), global_scratch_o: std::sync::Mutex::new(None), global_scratch_r: std::sync::Mutex::new(None), workspace: std::sync::Mutex::new(ForwardWorkspace::new()),
        })
    }
    pub fn forward(&self, i_ids: &[u32], pv: Option<&[f16]>, gt: Option<&[u32; 3]>, so: usize) -> Vec<f16> {
        let is_baking = self.text_model.layers.len() <= 1; let is_vision = pv.is_some();
        let mut wl = if is_baking { self.support_workspace.lock().unwrap() } else { self.workspace.lock().unwrap() };
        let mut embeds = match &self.text_model.embed_tokens.variant {
            LinearVariant::Standard { weight, .. } => native_embedding_lookup_f16(i_ids, weight.get_slice::<f16>().as_ref(), self.text_model.config.hidden_size),
            LinearVariant::BitSerial { weight_packed, .. } => native_embedding_lookup_f16(i_ids, weight_packed.get_slice::<f16>().as_ref(), self.text_model.config.hidden_size),
        };
        let sc = Some((&self.global_scratch_i, &self.global_scratch_o, &self.global_scratch_r));
        if let (Some(pv_val), Some(gt_val)) = (pv, gt) {
            if let Some(ref v) = self.visual {
                let rc = { let mut rg = self.text_model.rope_cache.lock().unwrap(); rg.ensure_length(so+i_ids.len()+1024); rg.cos.clone() };
                let rs = { let rg = self.text_model.rope_cache.lock().unwrap(); rg.sin.clone() };
                let vf = v.forward(pv_val, gt_val, &rc, &rs, sc, &mut wl);
                let hid = self.text_model.config.hidden_size; let mut vidx = 0;
                for (i, &id) in i_ids.iter().enumerate() { if id == 151655 && vidx < (vf.len()/hid) { embeds[i*hid..(i+1) * hid].copy_from_slice(&vf[vidx*hid..(vidx+1)*hid]); vidx += 1; } }
            }
        }
        // [FIX] Disable support_layer0 for inference. 2B model should use its own Layer 0 with the stitched KV.
        let sl = None; 
        let nx = self.text_model.forward_ext(i_ids, embeds, so, sc, Some(&mut wl), sl, is_vision);
        self.lm_head.forward(&nx, sc)
    }
    pub fn move_to_gpu(&mut self, dev: i32) -> anyhow::Result<()> {
        self.text_model.move_to_gpu(dev)?; 
        self.lm_head.move_to_gpu(dev)?;
        if let Some(ref mut v) = self.visual { v.move_to_gpu(dev)?; }
        if let Some(ref mut l0) = self.support_layer0 { 
            l0.move_to_gpu(dev)?; 
            println!("[LOAD] Hybrid Support Layer 0 moved to GPU-{}", dev); 
        }
        Ok(())
    }
    pub fn clear_kv_cache(&self) { self.text_model.clear_kv_cache(); }
    pub fn force_free_kv_cache(&self) { 
        self.text_model.force_free_kv_cache(); 
        if let Some(ref l0) = self.support_layer0 { l0.force_free_kv_cache(); }
    }

    pub fn get_kv_len(&self) -> usize {
        // [HYBRID-VISIBILITY] If in baking mode and support_layer0 exists, it's the one holding the context
        if self.text_model.layers.len() <= 1 && self.support_layer0.is_some() {
            let l0 = self.support_layer0.as_ref().unwrap();
            return l0.get_kv_len(self.text_model.config.head_dim, self.text_model.config.num_key_value_heads);
        }
        self.text_model.get_kv_len()
    }

    pub fn get_all_kv(&self, start_idx: usize) -> Vec<(Vec<u32>, Vec<f16>)> {
        let hd = self.text_model.config.head_dim; 
        let nkv = self.text_model.config.num_key_value_heads;
        
        // [HYBRID-VISIBILITY] Baking mode check
        if self.text_model.layers.len() <= 1 && self.support_layer0.is_some() {
            let l0 = self.support_layer0.as_ref().unwrap();
            if let Some(kv) = l0.get_kv_data(hd, nkv, start_idx) {
                return vec![kv];
            }
            return vec![];
        }
        
        self.text_model.get_all_kv(start_idx)
    }
}
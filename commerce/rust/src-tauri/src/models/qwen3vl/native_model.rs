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
use regex::Regex;

pub enum LinearVariant {
    Standard { weight: NativeTensor, bias: Option<NativeTensor> },
    BitSerial { weight_packed: NativeTensor, scales: NativeTensor, bias: Option<NativeTensor> },
    BitSliced4 { weight_packed: NativeTensor, scales: NativeTensor, bias: Option<NativeTensor> },
}

pub struct ForwardWorkspace {
    pub hidden_a: Vec<f16>, pub hidden_b: Vec<f16>, pub intermediate_a: Vec<f16>, pub intermediate_b: Vec<f16>, pub intermediate_c: Vec<f16>,
    pub q: Vec<f16>, pub k: Vec<f16>, pub v: Vec<f16>,
}

impl ForwardWorkspace {
    pub fn new() -> Self { Self { hidden_a: Vec::new(), hidden_b: Vec::new(), intermediate_a: Vec::new(), intermediate_b: Vec::new(), intermediate_c: Vec::new(), q: Vec::new(), k: Vec::new(), v: Vec::new() } }
    pub fn ensure_capacity(&mut self, hidden_size: usize, intermediate_size: usize, q_len: usize, n_h: usize, n_kv: usize, head_dim: usize) {
        let req_h = q_len * hidden_size; let req_i = q_len * intermediate_size;
        let eps = f16::from_f32(1e-6);
        if self.hidden_a.len() < req_h { self.hidden_a.resize(req_h, eps); } if self.hidden_b.len() < req_h { self.hidden_b.resize(req_h, eps); }
        if self.intermediate_a.len() < req_i { self.intermediate_a.resize(req_i, eps); } if self.intermediate_b.len() < req_i { self.intermediate_b.resize(req_i, eps); } if self.intermediate_c.len() < req_i { self.intermediate_c.resize(req_i, eps); }
        if self.q.len() < q_len * n_h * head_dim { self.q.resize(q_len * n_h * head_dim, eps); }
        if self.k.len() < q_len * n_kv * head_dim { self.k.resize(q_len * n_kv * head_dim, eps); }
        if self.v.len() < q_len * n_kv * head_dim { self.v.resize(q_len * n_kv * head_dim, eps); }
    }
}

pub struct DynamicKVCache { pub k: Vec<u32>, pub v: Vec<f16>, pub capacity: usize, pub current_len: usize }
impl DynamicKVCache {
    pub fn new() -> Self { Self { k: Vec::new(), v: Vec::new(), capacity: 0, current_len: 0 } }
    pub fn grow(&mut self, needed: usize, n_kv: usize, h_d: usize) {
        if needed > self.capacity {
            let nc = (needed + 1023) / 1024 * 1024;
            self.k.resize(nc * n_kv * (h_d/32), 0); self.v.resize(nc * n_kv * h_d, f16::ZERO);
            self.capacity = nc;
        }
    }
    pub fn clear(&mut self) { self.current_len = 0; }
}

pub struct DynamicGpuKVCache { pub k_ptr: Option<GpuPtr>, pub v_ptr: Option<GpuPtr>, pub capacity: usize, pub current_len: usize }
impl DynamicGpuKVCache {
    pub fn new() -> Self { Self { k_ptr: None, v_ptr: None, capacity: 0, current_len: 0 } }
    #[cfg(feature = "cuda")]
    pub fn grow(&mut self, needed: usize, n_kv: usize, h_d: usize, dev: i32) {
        if needed > self.capacity {
            let nc = (needed + 127) / 128 * 128;
            let kb = nc * n_kv * (h_d/32) * 4; let vb = nc * n_kv * h_d * 2;
            unsafe {
                let cl = crate::models::qwen3vl::native_backend::lib();
                let mut ctx = std::ptr::null_mut() as cudarc::driver::sys::CUcontext; cl.cuCtxGetCurrent(&mut ctx);
                if ctx == std::ptr::null_mut() && dev >= 0 && dev < 8 {
                    let mut d = 0 as cudarc::driver::sys::CUdevice; cl.cuDeviceGet(&mut d, dev);
                    cl.cuDevicePrimaryCtxRetain(&mut ctx, d); cl.cuCtxSetCurrent(ctx);
                }
                let mut nkp: cudarc::driver::sys::CUdeviceptr = 0; let mut nvp: cudarc::driver::sys::CUdeviceptr = 0;
                // [FIX] Correct return type comparison
                if (cl.cuMemAlloc_v2(&mut nkp, kb) as i32) != 0 || (cl.cuMemAlloc_v2(&mut nvp, vb) as i32) != 0 { return; }
                if let (Some(ok), Some(ov)) = (self.k_ptr.take(), self.v_ptr.take()) {
                    if self.current_len > 0 {
                        cl.cuMemcpyDtoD_v2(nkp, ok.0 as CUdeviceptr, self.current_len * n_kv * (h_d/32) * 4);
                        cl.cuMemcpyDtoD_v2(nvp, ov.0 as CUdeviceptr, self.current_len * n_kv * h_d * 2);
                    }
                    cl.cuMemFree_v2(ok.0 as CUdeviceptr); cl.cuMemFree_v2(ov.0 as CUdeviceptr);
                }
                self.k_ptr = Some(GpuPtr(nkp as *mut _)); self.v_ptr = Some(GpuPtr(nvp as *mut _));
                self.capacity = nc; cl.cuStreamSynchronize(std::ptr::null_mut());
            }
        }
    }
    pub fn clear(&mut self) { self.current_len = 0; }
}

fn log_tensor_health(name: &str, data: &[f16], shape: &[usize]) {
    if data.is_empty() { return; }
    let (mut min, mut max, mut sum) = (f32::MAX, f32::MIN, 0.0f32);
    for &v in data { let f = v.to_f32(); if f < min { min = f; } if f > max { max = f; } sum += f.abs(); }
    println!("[HEALTH] {:<40} | Shape: {:<12?} | Min: {:>8.4} | Max: {:>8.4} | AbsAvg: {:>8.4}", name, shape, min, max, sum / data.len() as f32);
}

pub fn dequantize_bit_serial_to_f16(p: &[u32], s: &[f16], n: usize, sk: usize, tk: usize, sc: usize) -> Vec<f16> {
    let skb = (sk + 31) / 32; let np = (n + 7) / 8 * 8; let mut out = vec![f16::ZERO; n * tk];
    let ratio = tk / sk; let ss = (np / 8) * skb * 8;
    for no in 0..(np / 8) {
        for skb_idx in 0..skb {
            for sub_n in 0..8 {
                let n_idx = no * 8 + sub_n; if n_idx >= n { continue; }
                for b in 0..32 {
                    let k_base = skb_idx * 32 + b; if k_base >= sk { break; }
                    let mut b_sum = 0.0f32;
                    for sl in 0..sc { if ((p[sl * ss + (no * skb + skb_idx) * 8 + sub_n] >> b) & 1) == 1 { b_sum += (1 << sl) as f32; } }
                    let f_val = if sc == 4 { (b_sum - 8.0) * s[n_idx].to_f32() } else { (if b_sum >= 1.0 { 1.0 } else { -1.0 }) * s[(no * skb + skb_idx) * 8 + sub_n].to_f32() };
                    for r in 0..ratio { out[n_idx * tk + k_base + (r * sk)] = f16::from_f32(f_val); }
                }
            }
        }
    }
    out
}

pub struct NativeLinear { pub in_features: usize, pub out_features: usize, pub src_in: usize, pub src_out: usize, pub variant: LinearVariant, pub device_id: i32 }
unsafe impl Send for NativeLinear {}
unsafe impl Sync for NativeLinear {}

impl NativeLinear {
    pub fn is_bit_serial(&self) -> bool { matches!(self.variant, LinearVariant::BitSerial { .. }) }
    #[cfg(feature = "cuda")]
    pub fn forward_gpu(&self, di: CUdeviceptr, dog: CUdeviceptr, m: usize) {
        unsafe { match &self.variant {
            LinearVariant::Standard { weight, .. } => crate::models::qwen3vl::native_backend::standard_matmul_cuda_f16(di as *const f16, weight.gpu_ptr.unwrap().0 as *const f16, dog as *mut f16, m as i32, self.out_features as i32, self.in_features as i32),
            LinearVariant::BitSerial { weight_packed, scales, .. } => crate::models::qwen3vl::native_backend::cuda_matmul_f16(di as *const f16, weight_packed.gpu_ptr.unwrap().0 as *const u32, scales.gpu_ptr.unwrap().0 as *const f16, dog as *mut f16, m as i32, self.out_features as i32, self.in_features as i32, self.device_id, self.src_in as i32),
            LinearVariant::BitSliced4 { weight_packed, scales, .. } => crate::models::qwen3vl::native_backend::cuda_matmul_4bit_f16(di as *const f16, weight_packed.gpu_ptr.unwrap().0 as *const u32, scales.gpu_ptr.unwrap().0 as *const f16, dog as *mut f16, m as i32, self.out_features as i32, self.in_features as i32, self.device_id, self.src_in as i32),
        } }
    }
    pub fn forward_into(&self, x: &[f16], out: &mut [f16], gs: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>) {
        let m = x.len() / self.in_features;
        match &self.variant {
            LinearVariant::Standard { weight, bias } => {
                #[cfg(feature = "cuda")]
                if self.device_id >= 0 {
                    if let Some(w_gpu) = weight.gpu_ptr {
                        let di = gs.and_then(|(si, _, _)| si.lock().unwrap().as_ref().map(|(p, _)| p.0 as CUdeviceptr)).unwrap_or(0);
                        let dog = gs.and_then(|(_, so, _)| so.lock().unwrap().as_ref().map(|(p, _)| p.0 as CUdeviceptr)).unwrap_or(0);
                        if di != 0 && dog != 0 {
                            unsafe {
                                let cl = crate::models::qwen3vl::native_backend::lib(); cl.cuMemcpyHtoD_v2(di, x.as_ptr() as *const _, x.len() * 2);
                                crate::models::qwen3vl::native_backend::standard_matmul_cuda_f16(di as *const f16, w_gpu.0 as *const f16, dog as *mut f16, m as i32, self.out_features as i32, self.in_features as i32);
                                cl.cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut _, dog, out.len() * 2);
                                if let Some(b) = bias { let br = unsafe { b.get_raw_slice::<f16>() }; for i in 0..m { for j in 0..self.out_features { out[i * self.out_features + j] += br[j]; } } }
                                return;
                            }
                        }
                    }
                }
                native_linear_f16_into(x, weight.get_slice::<f16>().as_ref(), bias.as_ref().map(|b| b.get_slice::<f16>()).as_deref().map(|b| b.as_ref()), out, m, self.out_features, self.in_features);
            },
            LinearVariant::BitSerial { weight_packed, scales, bias } => {
                let wpr = weight_packed.get_slice::<u32>(); let sr = scales.get_slice::<f16>();
                bit_serial_matmul_f32_extreme_into(x, wpr.as_ref(), sr.as_ref(), out, m, self.out_features, self.in_features);
                if let Some(b) = bias { let br = unsafe { b.get_raw_slice::<f16>() }; for i in 0..m { for j in 0..self.out_features { out[i * self.out_features + j] += br[j]; } } }
            },
            LinearVariant::BitSliced4 { weight_packed, scales, bias } => {
                let wpr = weight_packed.get_slice::<u32>(); let sr = scales.get_slice::<f16>();
                let dq = dequantize_bit_serial_to_f16(wpr.as_ref(), sr.as_ref(), self.out_features, self.src_in, self.in_features, 4);
                native_linear_f16_into(x, &dq, bias.as_ref().map(|b| b.get_slice::<f16>()).as_deref().map(|b| b.as_ref()), out, m, self.out_features, self.in_features);
            }
        }
    }
    pub fn forward(&self, x: &[f16], gs: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>) -> Vec<f16> {
        let mut out = vec![f16::ZERO; (x.len() / self.in_features) * self.out_features]; self.forward_into(x, &mut out, gs); out
    }
    pub fn move_to_gpu(&mut self, dev: i32) -> anyhow::Result<()> {
        self.device_id = dev; match &mut self.variant {
            LinearVariant::Standard { weight, bias } => { weight.move_to_gpu(dev)?; if let Some(b) = bias { b.move_to_gpu(dev)?; } },
            LinearVariant::BitSerial { weight_packed, scales, bias } => { weight_packed.move_to_gpu(dev)?; scales.move_to_gpu(dev)?; if let Some(b) = bias { b.move_to_gpu(dev)?; } },
            LinearVariant::BitSliced4 { weight_packed, scales, bias } => { weight_packed.move_to_gpu(dev)?; scales.move_to_gpu(dev)?; if let Some(b) = bias { b.move_to_gpu(dev)?; } },
        }
        Ok(())
    }
}

pub struct NativeLayer {
    pub input_layernorm: NativeTensor, pub post_attention_layernorm: NativeTensor, pub q_norm: Option<NativeTensor>, pub k_norm: Option<NativeTensor>,
    pub q_proj: NativeLinear, pub k_proj: NativeLinear, pub v_proj: NativeLinear, pub o_proj: NativeLinear, pub gate_proj: NativeLinear, pub up_proj: NativeLinear, pub down_proj: NativeLinear,
    pub device_id: i32, pub is_support_layer: bool, pub kv_cache: std::sync::Mutex<DynamicKVCache>, pub gpu_kv_cache: std::sync::Mutex<DynamicGpuKVCache>, pub gpu_broken: std::sync::atomic::AtomicBool,
}

impl NativeLayer {
    pub fn move_to_gpu(&mut self, dev: i32) -> anyhow::Result<()> {
        self.device_id = dev; self.input_layernorm.move_to_gpu(dev)?; self.post_attention_layernorm.move_to_gpu(dev)?;
        if let Some(ref mut qn) = self.q_norm { qn.move_to_gpu(dev)?; } if let Some(ref mut kn) = self.k_norm { kn.move_to_gpu(dev)?; }
        self.q_proj.move_to_gpu(dev)?; self.k_proj.move_to_gpu(dev)?; self.v_proj.move_to_gpu(dev)?; self.o_proj.move_to_gpu(dev)?;
        self.gate_proj.move_to_gpu(dev)?; self.up_proj.move_to_gpu(dev)?; self.down_proj.move_to_gpu(dev)?; Ok(())
    }
    pub fn forward<'a>(&self, x: &[f16], config: &Qwen3VLTextConfig, s_o: usize, _idx: usize, rc: &[f16], rs: &[f16], _is_baking: bool, is_vision: bool, gs: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>, ws: &'a mut ForwardWorkspace, use_b: bool, rope_gpu: &std::sync::Mutex<Option<(NativeTensor, NativeTensor)>>) -> &'a [f16] {
        let h_s = config.hidden_size; let q_len = x.len() / h_s; let h_d = config.head_dim; let n_h = config.num_attention_heads; let n_kv = config.num_key_value_heads;
        let out_ptr = if use_b { ws.hidden_b.as_mut_ptr() } else { ws.hidden_a.as_mut_ptr() };
        let f_alpha = if is_vision { 2.0f32 } else { 1.2f32 }; let s_gain = if is_vision { 1.15f32 } else { 1.02f32 };
        #[cfg(feature = "cuda")]
        if self.device_id >= 0 && self.device_id < 8 && !self.gpu_broken.load(Ordering::Relaxed) {
            if let Some((si, so, sr)) = gs {
                let rb = q_len * h_s.max(self.gate_proj.out_features) * 4;
                let di = si.lock().unwrap().as_ref().map(|(p, s)| if *s >= rb { p.0 as CUdeviceptr } else { 0 }).unwrap_or(0);
                let dog = so.lock().unwrap().as_ref().map(|(p, s)| if *s >= rb { p.0 as CUdeviceptr } else { 0 }).unwrap_or(0);
                let drg = sr.lock().unwrap().as_ref().map(|(p, s)| if *s >= rb { p.0 as CUdeviceptr } else { 0 }).unwrap_or(0);
                if di != 0 && dog != 0 && drg != 0 {
                    unsafe {
                        let cl = crate::models::qwen3vl::native_backend::lib(); cl.cuMemcpyHtoD_v2(di, x.as_ptr() as *const _, x.len() * 2);
                        let dlw = self.input_layernorm.gpu_ptr.as_ref().unwrap().0 as *const f16;
                        crate::models::qwen3vl::native_backend::cuda_rms_norm_f16(di as *const f16, dlw, dog as *mut f16, q_len as i32, h_s as i32, config.rms_norm_eps as f32);
                        self.q_proj.forward_gpu(dog, di, q_len); let dq = di; self.k_proj.forward_gpu(dog, drg, q_len); let dk = drg;
                        { let mut rgl = rope_gpu.lock().unwrap(); if rgl.is_none() {
                            let cp = crate::models::qwen3vl::native_backend::native_precompute_rope_f16(h_d, 32768, config.rope_theta);
                            let mut ct = NativeTensor { data_ptr: cp.0.as_ptr() as *const u8, host_size: cp.0.len()*2, gpu_ptr: None, shape: vec![32768, h_d/2], dtype: NativeDType::F16, _mmap: None, device_id: self.device_id };
                            let mut st = NativeTensor { data_ptr: cp.1.as_ptr() as *const u8, host_size: cp.1.len()*2, gpu_ptr: None, shape: vec![32768, h_d/2], dtype: NativeDType::F16, _mmap: None, device_id: self.device_id };
                            let _ = ct.move_to_gpu(self.device_id); let _ = st.move_to_gpu(self.device_id); *rgl = Some((ct, st));
                        } if let Some((ref c, ref s)) = *rgl { crate::models::qwen3vl::native_backend::native_cuda_apply_rope(dq, dk, c.gpu_ptr.unwrap().0 as CUdeviceptr, s.gpu_ptr.unwrap().0 as CUdeviceptr, q_len, s_o, n_h, n_kv, h_d); } }
                        let mut gkg = self.gpu_kv_cache.lock().unwrap(); let n_tok = s_o + q_len; gkg.grow(n_tok, n_kv, h_d, self.device_id);
                        if let (Some(kp), Some(vp)) = (gkg.k_ptr, gkg.v_ptr) {
                            let dkd = (kp.0 as u64 + (s_o * n_kv * (h_d/32) * 4) as u64) as CUdeviceptr;
                            let dvd = (vp.0 as u64 + (s_o * n_kv * h_d * 2) as u64) as CUdeviceptr;
                            crate::models::qwen3vl::native_backend::native_cuda_pack_bits(dk, dkd, q_len * n_kv * h_d);
                            self.v_proj.forward_gpu(dq, dvd, q_len); gkg.current_len = n_tok;
                        }
                        crate::models::qwen3vl::native_backend::native_bit_serial_attn_gpu_buffered(&[], gkg.k_ptr.unwrap(), gkg.v_ptr.unwrap(), n_h, n_kv, h_d, n_tok, self.device_id as usize, dq, dog, f_alpha, self.q_proj.src_in / n_h, true, q_len);
                        if s_gain != 1.0 { crate::models::qwen3vl::native_backend::native_cuda_apply_gain(dog, s_gain, q_len * h_s); }
                        self.o_proj.forward_gpu(dog, drg, q_len); cl.cuMemcpyHtoD_v2(dog, x.as_ptr() as *const _, x.len() * 2);
                        crate::models::qwen3vl::native_backend::native_cuda_add_inplace(drg, dog, x.len());
                        let dpw = self.post_attention_layernorm.gpu_ptr.as_ref().unwrap().0 as *const f16;
                        crate::models::qwen3vl::native_backend::cuda_rms_norm_f16(drg as *const f16, dpw, dog as *mut f16, q_len as i32, h_s as i32, config.rms_norm_eps as f32);
                        self.gate_proj.forward_gpu(dog, di, q_len); self.up_proj.forward_gpu(dog, dq, q_len);
                        crate::models::qwen3vl::native_backend::native_cuda_silu_inplace(di, q_len * config.intermediate_size);
                        crate::models::qwen3vl::native_backend::native_cuda_element_mul(di, dq, q_len * config.intermediate_size);
                        self.down_proj.forward_gpu(di, dog, q_len); if self.down_proj.out_features > self.down_proj.src_out { crate::models::qwen3vl::native_backend::native_cuda_hybrid_repeat(dog, self.down_proj.src_out, self.down_proj.out_features, q_len); }
                        crate::models::qwen3vl::native_backend::native_cuda_add_inplace(dog, drg, x.len());
                        let rs_sl = std::slice::from_raw_parts_mut(out_ptr, x.len()); cl.cuMemcpyDtoH_v2(rs_sl.as_mut_ptr() as *mut _, dog, x.len() * 2); return rs_sl;
                    }
                }
            }
        }
        let lnw = self.input_layernorm.get_slice::<f16>(); let mut cn = vec![f16::ZERO; x.len()];
        native_rms_norm_f16_into(x, lnw.as_ref(), config.rms_norm_eps as f32, h_s, &mut cn);
        self.q_proj.forward_into(&cn, &mut ws.q, gs); self.k_proj.forward_into(&cn, &mut ws.k, gs); self.v_proj.forward_into(&cn, &mut ws.v, gs);
        native_apply_rope_f16_with_offset(&mut ws.q, &mut ws.k, q_len, s_o, n_h, h_d, config.rope_theta, rc, rs);
        let mut cache = self.kv_cache.lock().unwrap(); let nl = s_o + q_len; cache.grow(nl, n_kv, h_d);
        let kp = pack_f16_to_bits(&ws.k); cache.k[s_o*(n_kv*h_d/32)..s_o*(n_kv*h_d/32)+kp.len()].copy_from_slice(&kp);
        cache.v[s_o*(n_kv*h_d)..s_o*(n_kv*h_d)+ws.v.len()].copy_from_slice(&ws.v); cache.current_len = nl;
        let mut ao = native_bit_serial_attn_f16(&ws.q, &cache.k, &cache.v, h_s, n_h, n_kv, q_len, nl, f_alpha);
        if s_gain != 1.0 { for v in ao.iter_mut() { *v = f16::from_f32(v.to_f32() * s_gain); } }
        let os = unsafe { std::slice::from_raw_parts_mut(out_ptr, x.len()) }; self.o_proj.forward_into(&ao, os, gs);
        for i in 0..x.len() { os[i] += x[i]; } let plnw = self.post_attention_layernorm.get_slice::<f16>();
        native_rms_norm_f16_into(os, plnw.as_ref(), config.rms_norm_eps as f32, h_s, &mut ws.intermediate_a[..x.len()]);
        self.gate_proj.forward_into(&ws.intermediate_a[..x.len()], &mut ws.intermediate_b, gs); self.up_proj.forward_into(&ws.intermediate_a[..x.len()], &mut ws.intermediate_c, gs);
        native_silu_f16(&mut ws.intermediate_b[..q_len * config.intermediate_size]);
        for i in 0..q_len * config.intermediate_size { ws.intermediate_b[i] *= ws.intermediate_c[i]; }
        let mut moh = vec![f16::ZERO; x.len()]; self.down_proj.forward_into(&ws.intermediate_b[..q_len * config.intermediate_size], &mut moh, gs);
        if self.down_proj.out_features > self.down_proj.src_out { for t in 0..q_len { for i in self.down_proj.src_out..self.down_proj.out_features { moh[t*self.down_proj.out_features+i] = moh[t*self.down_proj.out_features+(i%self.down_proj.src_out)]; } } }
        for i in 0..x.len() { os[i] += moh[i]; } os
    }
    pub fn clear_kv_cache(&self) { self.kv_cache.lock().unwrap().clear(); self.gpu_kv_cache.lock().unwrap().clear(); }
    pub fn force_free_kv_cache(&self) { self.kv_cache.lock().unwrap().k = Vec::new(); self.kv_cache.lock().unwrap().v = Vec::new(); let mut gc = self.gpu_kv_cache.lock().unwrap(); #[cfg(feature = "cuda")] unsafe { if let Some(k) = gc.k_ptr.take() { let _ = lib().cuMemFree_v2(k.0 as CUdeviceptr); } if let Some(v) = gc.v_ptr.take() { let _ = lib().cuMemFree_v2(v.0 as CUdeviceptr); } } gc.capacity = 0; gc.current_len = 0; }
    pub fn get_kv_data(&self, hd: usize, nkv: usize, st: usize) -> Option<(Vec<u32>, Vec<f16>)> {
        #[cfg(feature = "cuda")] if self.device_id >= 0 && !self.gpu_broken.load(Ordering::Relaxed) {
            let g = self.gpu_kv_cache.lock().unwrap(); if let (Some(kp), Some(vp)) = (g.k_ptr, g.v_ptr) { if g.current_len > st {
                let el = g.current_len - st; let ku = nkv * (hd/32); let vu = nkv * hd; let mut kh = vec![0u32; el * ku]; let mut vh = vec![f16::ZERO; el * vu];
                unsafe { let cl = crate::models::qwen3vl::native_backend::lib(); let dks = (kp.0 as u64 + (st * ku * 4) as u64) as CUdeviceptr; let dvs = (vp.0 as u64 + (st * vu * 2) as u64) as CUdeviceptr; let _ = cl.cuMemcpyDtoH_v2(kh.as_mut_ptr() as *mut _, dks, kh.len()*4); let _ = cl.cuMemcpyDtoH_v2(vh.as_mut_ptr() as *mut _, dvs, vh.len()*2); }
                return Some((kh, vh));
            } }
        }
        let c = self.kv_cache.lock().unwrap(); if c.current_len > st { let ku = nkv * (hd/32); let vu = nkv * hd; Some((c.k[st * ku .. c.current_len * ku].to_vec(), c.v[st * vu .. c.current_len * vu].to_vec())) } else { None }
    }
    pub fn get_kv_len(&self, hd: usize, nkv: usize) -> usize { #[cfg(feature = "cuda")] if self.device_id >= 0 { let g = self.gpu_kv_cache.lock().unwrap(); if g.current_len > 0 { return g.current_len; } } self.kv_cache.lock().unwrap().current_len }
    pub fn set_kv_data(&self, k: Vec<u32>, v: Vec<f16>) { let mut c = self.kv_cache.lock().unwrap(); c.k = k; c.v = v; c.capacity = 0; }
    #[cfg(feature = "cuda")]
    pub fn inject_gpu_kv(&self, k: &[u32], v: &[f16], nkv: usize, hd: usize) {
        let tok = (k.len() * 32) / (nkv * hd); let mut g = self.gpu_kv_cache.lock().unwrap();
        unsafe { let cl = crate::models::qwen3vl::native_backend::lib(); let mut ctx = std::ptr::null_mut() as cudarc::driver::sys::CUcontext; cl.cuCtxGetCurrent(&mut ctx);
            if ctx == std::ptr::null_mut() && self.device_id >= 0 { let mut d = 0 as cudarc::driver::sys::CUdevice; cl.cuDeviceGet(&mut d, self.device_id); cl.cuDevicePrimaryCtxRetain(&mut ctx, d); cl.cuCtxSetCurrent(ctx); }
            let mut kp: CUdeviceptr = 0; let mut vp: CUdeviceptr = 0; let _ = cl.cuMemAlloc_v2(&mut kp, tok * nkv * (hd/32) * 4); let _ = cl.cuMemAlloc_v2(&mut vp, tok * nkv * hd * 2);
            let _ = cl.cuMemcpyHtoD_v2(kp, k.as_ptr() as *const _, k.len()*4); let _ = cl.cuMemcpyHtoD_v2(vp, v.as_ptr() as *const _, v.len()*2);
            g.k_ptr = Some(GpuPtr(kp as *mut _)); g.v_ptr = Some(GpuPtr(vp as *mut _)); g.capacity = tok; g.current_len = tok;
        }
    }
    #[cfg(feature = "cuda")]
    pub fn inject_gpu_kv_direct(&self, ks: GpuPtr, vs: GpuPtr, tok: usize, nkv: usize, hd: usize) {
        let mut g = self.gpu_kv_cache.lock().unwrap(); unsafe { let cl = crate::models::qwen3vl::native_backend::lib(); let mut ctx = std::ptr::null_mut() as cudarc::driver::sys::CUcontext; cl.cuCtxGetCurrent(&mut ctx);
            if ctx == std::ptr::null_mut() && self.device_id >= 0 { let mut d = 0 as cudarc::driver::sys::CUdevice; cl.cuDeviceGet(&mut d, self.device_id); cl.cuDevicePrimaryCtxRetain(&mut ctx, d); cl.cuCtxSetCurrent(ctx); }
            let mut kp: CUdeviceptr = 0; let mut vp: CUdeviceptr = 0; let kb = tok * nkv * (hd/32) * 4; let vb = tok * nkv * hd * 2;
            if (cl.cuMemAlloc_v2(&mut kp, kb) as i32) == 0 && (cl.cuMemAlloc_v2(&mut vp, vb) as i32) == 0 { cl.cuMemcpyDtoD_v2(kp, ks.0 as CUdeviceptr, kb); cl.cuMemcpyDtoD_v2(vp, vs.0 as CUdeviceptr, vb); g.k_ptr = Some(GpuPtr(kp as *mut _)); g.v_ptr = Some(GpuPtr(vp as *mut _)); g.capacity = tok; g.current_len = tok; }
        }
    }
}

pub struct NativeQwen3TextModel {
    pub config: Qwen3VLTextConfig, pub embed_tokens: NativeLinear, pub layers: Vec<NativeLayer>, pub norm: NativeTensor,
    pub rope_cache: std::sync::Mutex<RopeCache>, pub rope_cache_gpu: std::sync::Mutex<Option<(NativeTensor, NativeTensor)>>,
}

impl NativeQwen3TextModel {
    pub fn forward(&self, i_ids: &[u32], _pv: Option<&[f16]>, _gt: Option<&[u32; 3]>, s_o: usize, gs: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>, workspace: Option<&mut ForwardWorkspace>, sl: Option<&NativeLayer>, is_vision: bool) -> Vec<f16> {
        let hid = self.config.hidden_size; let embeds = match &self.embed_tokens.variant { LinearVariant::Standard { weight, .. } => native_embedding_lookup_f16(i_ids, weight.get_slice::<f16>().as_ref(), hid), _ => vec![f16::ZERO; i_ids.len() * hid] };
        self.forward_ext(i_ids, embeds, s_o, gs, workspace, sl, is_vision)
    }
    pub fn forward_ext(&self, _i_ids: &[u32], embeds: Vec<f16>, s_o: usize, gs: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>, workspace: Option<&mut ForwardWorkspace>, _sl: Option<&NativeLayer>, is_vision: bool) -> Vec<f16> {
        let hid = self.config.hidden_size; let is_baking = self.layers.len() <= 1; let q_len = embeds.len() / hid;
        let mut internal_ws = ForwardWorkspace::new(); let ws = match workspace { Some(w) => w, None => &mut internal_ws };
        ws.ensure_capacity(hid, self.config.intermediate_size, q_len, self.config.num_attention_heads, self.config.num_key_value_heads, self.config.head_dim);
        ws.hidden_a[..embeds.len()].copy_from_slice(&embeds); let mut cur_x: &[f16] = unsafe { std::slice::from_raw_parts(ws.hidden_a.as_ptr(), embeds.len()) };
        if !cur_x.is_empty() { log_tensor_health("Embeddings", cur_x, &[q_len, hid]); }
        { let mut rg = self.rope_cache.lock().unwrap(); rg.ensure_length(s_o + q_len); }
        let rg_lock = self.rope_cache.lock().unwrap(); let r_cos = &rg_lock.cos; let r_sin = &rg_lock.sin;
        for (i, layer) in self.layers.iter().enumerate() {
            let out = layer.forward(cur_x, &self.config, s_o, i, r_cos, r_sin, is_baking, is_vision, gs, ws, i % 2 == 0, &self.rope_cache_gpu);
            if !out.is_empty() && (i == 0 || i == self.layers.len() - 1) { log_tensor_health(&format!("Layer {} Output", i), out, &[q_len, hid]); }
            if out.is_empty() { cur_x = &[]; } else { cur_x = unsafe { std::slice::from_raw_parts(out.as_ptr(), out.len()) }; }
        }
        if cur_x.is_empty() { return vec![]; }
        let n_cow = self.norm.get_slice::<f16>(); let mut cn = vec![f16::ZERO; cur_x.len()]; native_rms_norm_f16_into(cur_x, n_cow.as_ref(), self.config.rms_norm_eps as f32, hid, &mut cn);
        log_tensor_health("Final Norm Output", &cn, &[q_len, hid]); cn
    }
    pub fn move_to_gpu(&mut self, dev: i32) -> anyhow::Result<()> { self.embed_tokens.move_to_gpu(dev)?; self.norm.move_to_gpu(dev)?; for layer in &mut self.layers { layer.move_to_gpu(dev)?; } Ok(()) }
    pub fn clear_kv_cache(&self) { for layer in &self.layers { layer.clear_kv_cache(); } }
    pub fn force_free_kv_cache(&self) { for layer in &self.layers { layer.force_free_kv_cache(); } let mut rg = self.rope_cache_gpu.lock().unwrap(); *rg = None; }
    pub fn batch_upload_stitched_cache(&self, k: Vec<u32>, v: Vec<f16>) {
        let hd = self.config.head_dim; let nkv = self.config.num_key_value_heads;
        #[cfg(feature = "cuda")] if self.layers[0].device_id >= 0 && self.layers[0].device_id < 8 {
            self.layers[0].inject_gpu_kv(&k, &v, nkv, hd); let l0c = self.layers[0].gpu_kv_cache.lock().unwrap();
            if let (Some(ks), Some(vs)) = (l0c.k_ptr, l0c.v_ptr) { let tok = l0c.current_len; for i in 1..self.layers.len() { self.layers[i].inject_gpu_kv_direct(ks, vs, tok, nkv, hd); } }
            return;
        }
        for layer in &self.layers { layer.set_kv_data(k.clone(), v.clone()); }
    }
    pub fn get_kv_len(&self) -> usize { self.layers[0].get_kv_len(self.config.head_dim, self.config.num_key_value_heads) }
    pub fn get_all_kv(&self, st: usize) -> Vec<(Vec<u32>, Vec<f16>)> { let hd = self.config.head_dim; let nkv = self.config.num_key_value_heads; self.layers.iter().filter_map(|l| l.get_kv_data(hd, nkv, st)).collect() }
}

pub struct NativeQwen3VLModel {
    pub config: Qwen3VLConfig, pub text_model: NativeQwen3TextModel, pub lm_head: NativeLinear, pub visual: Option<NativeVisionModel>,
    pub support_layer0: Option<NativeLayer>, pub support_workspace: std::sync::Mutex<ForwardWorkspace>,
    pub global_scratch_i: std::sync::Mutex<Option<(GpuPtr, usize)>>, pub global_scratch_o: std::sync::Mutex<Option<(GpuPtr, usize)>>, pub global_scratch_r: std::sync::Mutex<Option<(GpuPtr, usize)>>,
    pub workspace: std::sync::Mutex<ForwardWorkspace>,
}

pub struct NativeVisionModel { pub patch_embed: NativeLinear, pub blocks: Vec<NativeLayer>, pub merger: NativeLayer }
impl NativeVisionModel {
    pub fn move_to_gpu(&mut self, dev: i32) -> anyhow::Result<()> { self.patch_embed.move_to_gpu(dev)?; for b in &mut self.blocks { b.move_to_gpu(dev)?; } self.merger.move_to_gpu(dev)?; Ok(()) }
    pub fn forward<'a>(&self, pv: &[f16], gt: &[u32; 3], rc: &[f16], rs: &[f16], gs: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>, ws: &'a mut ForwardWorkspace, rope_gpu: &std::sync::Mutex<Option<(NativeTensor, NativeTensor)>>) -> &'a [f16] {
        let eo = self.patch_embed.forward(pv, gs); ws.hidden_a[..eo.len()].copy_from_slice(&eo); let mut cur_x: &[f16] = unsafe { std::slice::from_raw_parts(ws.hidden_a.as_ptr(), eo.len()) };
        for (i, block) in self.blocks.iter().enumerate() { let out = block.forward(cur_x, &crate::models::qwen3vl::config::Qwen3VLTextConfig { hidden_size: self.patch_embed.out_features, intermediate_size: block.gate_proj.out_features, num_hidden_layers: 1, num_attention_heads: 16, num_key_value_heads: 16, head_dim: self.patch_embed.out_features / 16, rms_norm_eps: 1e-6, rope_theta: 10000.0, vocab_size: 0, max_position_embeddings: 32768, dtype: None, rope_scaling: None }, 0, 0, rc, rs, false, true, gs, ws, i % 2 != 0, rope_gpu); cur_x = unsafe { std::slice::from_raw_parts(out.as_ptr(), out.len()) }; }
        self.merger.forward(cur_x, &crate::models::qwen3vl::config::Qwen3VLTextConfig { hidden_size: cur_x.len() / (gt[1] * gt[2]) as usize, intermediate_size: cur_x.len() / (gt[1] * gt[2]) as usize, num_hidden_layers: 1, num_attention_heads: 1, num_key_value_heads: 1, head_dim: cur_x.len() / (gt[1] * gt[2]) as usize, rms_norm_eps: 1e-6, rope_theta: 10000.0, vocab_size: 0, max_position_embeddings: 32768, dtype: None, rope_scaling: None }, 0, 0, rc, rs, false, true, gs, ws, self.blocks.len() % 2 == 0, rope_gpu)
    }
}

impl NativeQwen3VLModel {
    pub fn load(config: Qwen3VLConfig, m_mmap: Arc<Mmap>, v_mmap: Option<Arc<Mmap>>, baking: bool, s_mmap: Option<Arc<Mmap>>, dev_id: i32) -> Result<Self> {
        let st = SafeTensors::deserialize(&m_mmap)?; let st_sec = s_mmap.as_ref().map(|m| SafeTensors::deserialize(m)).transpose()?;
        let t_c = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;
        let find_key = |target: &str, p: &SafeTensors, s: Option<&SafeTensors>| -> Option<(String, bool)> {
            if p.tensor(target).is_ok() || p.tensor(&format!("{}.packed", target)).is_ok() { return Some((target.to_string(), true)); }
            if let Some(sec) = s { if sec.tensor(target).is_ok() || sec.tensor(&format!("{}.packed", target)).is_ok() { return Some((target.to_string(), false)); } }
            None
        };
        let get_l = |base: &str, inf: usize, outf: usize| -> Result<NativeLinear> {
            let (key, is_p) = find_key(base, &st, st_sec.as_ref()).ok_or_else(|| anyhow!("LinearNotFound: {}", base))?;
            let cst = if is_p { &st } else { st_sec.as_ref().unwrap() }; let cm = if is_p { &m_mmap } else { s_mmap.as_ref().unwrap() };
            if cst.tensor(&format!("{}.packed", key)).is_ok() {
                let vp = cst.tensor(&format!("{}.packed", key))?; let vs = cst.tensor(&format!("{}.scales", key))?; let format = cst.tensor(&format!("{}.format", key)).map(|t| t.data()[0] as i8).unwrap_or(1);
                let (so, si) = if let Ok(sh) = cst.tensor(&format!("{}.shape", key)) { let sd = unsafe { std::slice::from_raw_parts(sh.data().as_ptr() as *const i32, 2) }; (sd[0] as usize, sd[1] as usize) } else { (outf, inf) };
                if inf > si || outf > so {
                    let dq = dequantize_bit_serial_to_f16(unsafe { std::slice::from_raw_parts(vp.data().as_ptr() as *const u32, vp.data().len()/4) }, unsafe { std::slice::from_raw_parts(vs.data().as_ptr() as *const f16, vs.data().len()/2) }, so, si, inf, if format == 4 { 4 } else { 1 });
                    let boxed = dq.into_boxed_slice(); let ptr = boxed.as_ptr(); std::mem::forget(boxed);
                    return Ok(NativeLinear { in_features: inf, out_features: outf, src_in: si, src_out: so, variant: LinearVariant::Standard { weight: NativeTensor { data_ptr: ptr as *const u8, host_size: outf*inf*2, gpu_ptr: None, shape: vec![outf, inf], dtype: NativeDType::F16, _mmap: None, device_id: dev_id }, bias: None }, device_id: dev_id });
                }
                Ok(NativeLinear { in_features: inf, out_features: outf, src_in: inf, src_out: outf, variant: if format == 4 { LinearVariant::BitSliced4 { weight_packed: NativeTensor::from_mmap(cm.clone(), unsafe { vp.data().as_ptr().offset_from(cm.as_ptr()) } as usize, vp.shape().to_vec(), NativeDType::U32), scales: NativeTensor::from_mmap(cm.clone(), unsafe { vs.data().as_ptr().offset_from(cm.as_ptr()) } as usize, vs.shape().to_vec(), NativeDType::F16), bias: None } } else { LinearVariant::BitSerial { weight_packed: NativeTensor::from_mmap(cm.clone(), unsafe { vp.data().as_ptr().offset_from(cm.as_ptr()) } as usize, vp.shape().to_vec(), NativeDType::U32), scales: NativeTensor::from_mmap(cm.clone(), unsafe { vs.data().as_ptr().offset_from(cm.as_ptr()) } as usize, vs.shape().to_vec(), NativeDType::F16), bias: None } }, device_id: dev_id })
            } else {
                let v = cst.tensor(&key)?; let o = unsafe { v.data().as_ptr().offset_from(cm.as_ptr()) } as usize;
                Ok(NativeLinear { in_features: inf, out_features: outf, src_in: inf, src_out: outf, variant: LinearVariant::Standard { weight: NativeTensor::from_mmap(cm.clone(), o, v.shape().to_vec(), NativeDType::F16), bias: None }, device_id: dev_id })
            }
        };
        let get_t = |name: &str, ts: usize| -> Result<NativeTensor> {
            let (key, is_p) = find_key(name, &st, st_sec.as_ref()).ok_or_else(|| anyhow!("TNotFound: {}", name))?;
            let cst = if is_p { &st } else { st_sec.as_ref().unwrap() }; let cm = if is_p { &m_mmap } else { s_mmap.as_ref().unwrap() };
            let v = cst.tensor(&key)?; if ts > v.data().len()/2 { let mut nd = Vec::with_capacity(ts*2); let sf = unsafe { std::slice::from_raw_parts(v.data().as_ptr() as *const f16, v.data().len()/2) }; for _ in 0..(ts/(v.data().len()/2)) { for &val in sf { nd.extend_from_slice(&val.to_le_bytes()); } } let boxed = nd.into_boxed_slice(); let ptr = boxed.as_ptr(); std::mem::forget(boxed); Ok(NativeTensor { data_ptr: ptr as *const u8, host_size: ts*2, gpu_ptr: None, shape: vec![ts], dtype: NativeDType::F16, _mmap: None, device_id: dev_id }) }
            else { Ok(NativeTensor { data_ptr: unsafe { cm.as_ptr().add(unsafe { v.data().as_ptr().offset_from(cm.as_ptr()) } as usize) }, host_size: v.data().len(), gpu_ptr: None, shape: v.shape().to_vec(), dtype: NativeDType::F16, _mmap: Some(cm.clone()), device_id: dev_id }) }
        };
        let get_embed = |base: &str, vocab: usize, thid: usize| -> Result<NativeLinear> {
            let (key, is_p) = find_key(base, &st, st_sec.as_ref()).ok_or_else(|| anyhow!("EmbedNotFound: {}", base))?;
            let cst = if is_p { &st } else { st_sec.as_ref().unwrap() }; let cm = if is_p { &m_mmap } else { s_mmap.as_ref().unwrap() };
            if cst.tensor(&format!("{}.packed", key)).is_ok() {
                let vp = cst.tensor(&format!("{}.packed", key))?; let vs = cst.tensor(&format!("{}.scales", key))?; let format = cst.tensor(&format!("{}.format", key)).map(|t| t.data()[0] as i8).unwrap_or(1);
                let dq = dequantize_bit_serial_to_f16(unsafe { std::slice::from_raw_parts(vp.data().as_ptr() as *const u32, vp.data().len()/4) }, unsafe { std::slice::from_raw_parts(vs.data().as_ptr() as *const f16, vs.data().len()/2) }, vocab, (vp.data().len()/4*32)/(vocab*(if format == 4 { 4 } else { 1 })), thid, if format == 4 { 4 } else { 1 });
                let boxed = dq.into_boxed_slice(); let ptr = boxed.as_ptr(); std::mem::forget(boxed);
                Ok(NativeLinear { in_features: vocab, out_features: thid, src_in: thid, src_out: thid, variant: LinearVariant::Standard { weight: NativeTensor { data_ptr: ptr as *const u8, host_size: vocab*thid*2, gpu_ptr: None, shape: vec![vocab, thid], dtype: NativeDType::F16, _mmap: None, device_id: dev_id }, bias: None }, device_id: dev_id })
            } else { get_l(base, vocab, thid) }
        };
        let emb = get_embed("model.language_model.embed_tokens.weight", 151936, t_c.hidden_size)?;
        let mut layers = Vec::new(); for i in 0..(if baking { 1 } else { t_c.num_hidden_layers }) {
            let p = format!("model.language_model.layers.{}", i);
            layers.push(NativeLayer {
                input_layernorm: get_t(&format!("{}.input_layernorm.weight", p), t_c.hidden_size)?, post_attention_layernorm: get_t(&format!("{}.post_attention_layernorm.weight", p), t_c.hidden_size)?, q_norm: get_t(&format!("{}.self_attn.q_norm.weight", p), t_c.hidden_size).ok(), k_norm: get_t(&format!("{}.self_attn.k_norm.weight", p), t_c.hidden_size).ok(),
                q_proj: get_l(&format!("{}.self_attn.q_proj.weight", p), t_c.hidden_size, t_c.num_attention_heads*t_c.head_dim)?, k_proj: get_l(&format!("{}.self_attn.k_proj.weight", p), t_c.hidden_size, t_c.num_key_value_heads*t_c.head_dim)?, v_proj: get_l(&format!("{}.self_attn.v_proj.weight", p), t_c.hidden_size, t_c.num_key_value_heads*t_c.head_dim)?, o_proj: get_l(&format!("{}.self_attn.o_proj.weight", p), t_c.num_attention_heads*t_c.head_dim, t_c.hidden_size)?,
                gate_proj: get_l(&format!("{}.mlp.gate_proj.weight", p), t_c.hidden_size, t_c.intermediate_size)?, up_proj: get_l(&format!("{}.mlp.up_proj.weight", p), t_c.hidden_size, t_c.intermediate_size)?, down_proj: get_l(&format!("{}.mlp.down_proj.weight", p), t_c.intermediate_size, t_c.hidden_size)?,
                device_id: dev_id, is_support_layer: false, kv_cache: std::sync::Mutex::new(DynamicKVCache::new()), gpu_kv_cache: std::sync::Mutex::new(DynamicGpuKVCache::new()), gpu_broken: std::sync::atomic::AtomicBool::new(false),
            });
        }
        let norm = match get_t("model.language_model.norm.weight", t_c.hidden_size) { Ok(t) => t, Err(e) => if baking { let h = t_c.hidden_size*2; let d = vec![0u8; h]; let b = d.into_boxed_slice(); let p = b.as_ptr(); std::mem::forget(b); NativeTensor { data_ptr: p, host_size: h, gpu_ptr: None, shape: vec![t_c.hidden_size], dtype: NativeDType::F16, _mmap: None, device_id: dev_id } } else { return Err(e); } };
        let head = match get_l("model.language_model.lm_head.weight", t_c.hidden_size, 151936).or_else(|_| get_l("model.language_model.embed_tokens.weight", t_c.hidden_size, 151936)) { Ok(l) => l, Err(e) => if baking { let d = vec![0u8; 16]; let b = d.into_boxed_slice(); let p = b.as_ptr(); std::mem::forget(b); NativeLinear { in_features: t_c.hidden_size, out_features: 151936, src_in: t_c.hidden_size, src_out: 151936, variant: LinearVariant::Standard { weight: NativeTensor { data_ptr: p, host_size: 16, gpu_ptr: None, shape: vec![151936, t_c.hidden_size], dtype: NativeDType::F16, _mmap: None, device_id: dev_id }, bias: None }, device_id: dev_id } } else { return Err(e); } };
        let mut model = Self {
            config: config.clone(), text_model: NativeQwen3TextModel { config: t_c.clone(), embed_tokens: emb, layers, norm, rope_cache: std::sync::Mutex::new(RopeCache::new(t_c.head_dim, t_c.rope_theta, 32768)), rope_cache_gpu: std::sync::Mutex::new(None) },
            lm_head: head, visual: None, support_layer0: None, support_workspace: std::sync::Mutex::new(ForwardWorkspace::new()), global_scratch_i: std::sync::Mutex::new(None), global_scratch_o: std::sync::Mutex::new(None), global_scratch_r: std::sync::Mutex::new(None), workspace: std::sync::Mutex::new(ForwardWorkspace::new()),
        };
        #[cfg(feature = "cuda")] if dev_id >= 0 { let pb = 2048 * t_c.hidden_size.max(t_c.intermediate_size) * 4; unsafe { let cl = crate::models::qwen3vl::native_backend::lib(); let (mut p1, mut p2, mut p3) = (0, 0, 0); if (cl.cuMemAlloc_v2(&mut p1, pb) as i32) == 0 { *model.global_scratch_i.get_mut().unwrap() = Some((GpuPtr(p1 as *mut _), pb)); } if (cl.cuMemAlloc_v2(&mut p2, pb) as i32) == 0 { *model.global_scratch_o.get_mut().unwrap() = Some((GpuPtr(p2 as *mut _), pb)); } if (cl.cuMemAlloc_v2(&mut p3, pb) as i32) == 0 { *model.global_scratch_r.get_mut().unwrap() = Some((GpuPtr(p3 as *mut _), pb)); } } }
        Ok(model)
    }
    pub fn load_kv_stitched(&self, paths: &[std::path::PathBuf]) -> Result<()> {
        let n_kv = self.text_model.config.num_key_value_heads; let h_d = self.text_model.config.head_dim; self.text_model.force_free_kv_cache();
        let mut expanded = Vec::new(); for p in paths { if p.exists() && p.is_file() { expanded.push(p.clone()); } else { let dir = p.parent().unwrap_or(std::path::Path::new(".")); let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or(""); if let Ok(entries) = std::fs::read_dir(dir) { let mut shards = Vec::new(); for entry in entries.flatten() { let path = entry.path(); let name = path.file_name().and_then(|s| s.to_str()).unwrap_or(""); if name.starts_with(stem) && name.contains("_shard") && name.ends_with(".safetensors") { shards.push(path); } } let re = Regex::new(r"_shard(\d+)").unwrap(); shards.sort_by(|a, b| { let get_n = |p: &std::path::Path| { let name = p.file_name().and_then(|s| s.to_str()).unwrap_or(""); re.captures(name).and_then(|c| c[1].parse::<usize>().ok()).unwrap_or(0) }; get_n(a).cmp(&get_n(b)) }); expanded.extend(shards); } } }
        if expanded.is_empty() { return Ok(()); }
        let mut total_t = 0; let mut metadatas = Vec::new(); let mut mmaps = Vec::new();
        for path in &expanded { let file = std::fs::File::open(path)?; let mmap = Arc::new(unsafe { memmap2::MmapOptions::new().map(&file)? }); let tokens = { let st = SafeTensors::deserialize(&mmap)?; let t = if let Ok(meta) = st.tensor("metadata.tokens") { meta.shape()[0] } else if let Ok(kt) = st.tensor("layer.0.k") { let u32s = kt.data().len() / 4; let upt = n_kv * (h_d / 32); if upt > 0 { u32s / upt } else { 0 } } else { 0 }; if t > 0 { if let Ok(tt) = st.tensor("metadata.tokens") { let tu32: Vec<u32> = tt.data().chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect(); self.text_model.rope_cache.lock().unwrap().tail_tokens = tu32; } } t }; if tokens > 0 { total_t += tokens; metadatas.push(tokens); mmaps.push(mmap); } }
        if total_t == 0 { return Ok(()); }
        for l_idx in 0..self.text_model.layers.len() { let layer = &self.text_model.layers[l_idx]; let mut gpu_kv = layer.gpu_kv_cache.lock().unwrap(); #[cfg(feature = "cuda")] if self.lm_head.device_id >= 0 { gpu_kv.grow(total_t, n_kv, h_d, self.lm_head.device_id); }
            let mut curr_o = 0; for (p_idx, mmap) in mmaps.iter().enumerate() { let st = SafeTensors::deserialize(mmap)?; let tip = metadatas[p_idx]; let mut fl = 0; while st.tensor(&format!("layer.{}.k", fl)).is_ok() { fl += 1; } let si = if fl == 1 { 0 } else { l_idx }; if si >= fl { curr_o += tip; continue; }
                if let (Ok(kt), Ok(vt)) = (st.tensor(&format!("layer.{}.k", si)), st.tensor(&format!("layer.{}.v", si))) { #[cfg(feature = "cuda")] if self.lm_head.device_id >= 0 { unsafe { let cl = crate::models::qwen3vl::native_backend::lib(); if let Some(kp) = gpu_kv.k_ptr { cl.cuMemcpyHtoD_v2((kp.0 as u64 + (curr_o * n_kv * (h_d/32) * 4) as u64) as CUdeviceptr, kt.data().as_ptr() as *const _, kt.data().len()); } if let Some(vp) = gpu_kv.v_ptr { cl.cuMemcpyHtoD_v2((vp.0 as u64 + (curr_o * n_kv * h_d * 2) as u64) as CUdeviceptr, vt.data().as_ptr() as *const _, vt.data().len()); } } } else { let mut kvh = layer.kv_cache.lock().unwrap(); kvh.grow(total_t, n_kv, h_d); let ku = n_kv * (h_d/32); let vu = n_kv * h_d; let kd: Vec<u32> = kt.data().chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect(); let vd: Vec<f16> = vt.data().chunks_exact(2).map(|c| f16::from_ne_bytes(c.try_into().unwrap())).collect(); kvh.k[curr_o*ku..(curr_o+tip)*ku].copy_from_slice(&kd); kvh.v[curr_o*vu..(curr_o+tip)*vu].copy_from_slice(&vd); } }
                curr_o += tip; } gpu_kv.current_len = total_t; if self.lm_head.device_id < 0 { layer.kv_cache.lock().unwrap().current_len = total_t; } } Ok(())
    }
    pub fn forward(&self, i_ids: &[u32], pv: Option<&[f16]>, gt: Option<&[u32; 3]>, so: usize) -> Vec<f16> {
        let is_b = self.text_model.layers.len() <= 1; let mut wl = if is_b { self.support_workspace.lock().unwrap() } else { self.workspace.lock().unwrap() };
        let embeds = match &self.text_model.embed_tokens.variant { LinearVariant::Standard { weight, .. } => native_embedding_lookup_f16(i_ids, weight.get_slice::<f16>().as_ref(), self.text_model.config.hidden_size), _ => vec![f16::ZERO; i_ids.len() * self.text_model.config.hidden_size] };
        let sc = Some((&self.global_scratch_i, &self.global_scratch_o, &self.global_scratch_r)); let mut fe = embeds;
        if let (Some(pvv), Some(gtv)) = (pv, gt) { if let Some(ref v) = self.visual { let rc = { let mut rg = self.text_model.rope_cache.lock().unwrap(); rg.ensure_length(so+i_ids.len()+1024); rg.cos.clone() }; let rs = { let rg = self.text_model.rope_cache.lock().unwrap(); rg.sin.clone() }; let vf = v.forward(pvv, gtv, &rc, &rs, sc, &mut wl, &self.text_model.rope_cache_gpu); let hid = self.text_model.config.hidden_size; let mut vidx = 0; for (i, &id) in i_ids.iter().enumerate() { if id == 151655 && vidx < (vf.len()/hid) { fe[i*hid..(i+1)*hid].copy_from_slice(&vf[vidx*hid..(vidx+1)*hid]); vidx += 1; } } } }
        let nx = self.text_model.forward_ext(i_ids, fe, so, sc, Some(&mut wl), None, pv.is_some()); if is_b { return vec![]; } self.lm_head.forward(&nx, sc)
    }
    pub fn move_to_gpu(&mut self, dev: i32) -> anyhow::Result<()> { self.text_model.move_to_gpu(dev)?; self.lm_head.move_to_gpu(dev)?; if let Some(ref mut v) = self.visual { v.move_to_gpu(dev)?; } if let Some(ref mut l0) = self.support_layer0 { l0.move_to_gpu(dev)?; } Ok(()) }
    pub fn clear_kv_cache(&self) { self.text_model.clear_kv_cache(); }
    pub fn force_free_kv_cache(&self) { self.text_model.force_free_kv_cache(); if let Some(ref l0) = self.support_layer0 { l0.force_free_kv_cache(); } }
    pub fn get_kv_len(&self) -> usize { if self.text_model.layers.len() <= 1 && self.support_layer0.is_some() { return self.support_layer0.as_ref().unwrap().get_kv_len(self.text_model.config.head_dim, self.text_model.config.num_key_value_heads); } self.text_model.get_kv_len() }
    pub fn get_all_kv(&self, st: usize) -> Vec<(Vec<u32>, Vec<f16>)> { let hd = self.text_model.config.head_dim; let nkv = self.text_model.config.num_key_value_heads; if self.text_model.layers.len() <= 1 && self.support_layer0.is_some() { if let Some(kv) = self.support_layer0.as_ref().unwrap().get_kv_data(hd, nkv, st) { return vec![kv]; } } self.text_model.get_all_kv(st) }
}

pub struct RopeCache { pub cos: Vec<f16>, pub sin: Vec<f16>, pub head_dim: usize, pub theta: f32, pub tail_tokens: Vec<u32> }
impl RopeCache {
    pub fn new(head_dim: usize, theta: f32, max_len: usize) -> Self { let (cos, sin) = crate::models::qwen3vl::native_backend::native_precompute_rope_f16(head_dim, max_len, theta); Self { cos, sin, head_dim, theta, tail_tokens: Vec::new() } }
    pub fn ensure_length(&mut self, len: usize) { if len * (self.head_dim / 2) > self.cos.len() { let (cos, sin) = crate::models::qwen3vl::native_backend::native_precompute_rope_f16(self.head_dim, (len + 1023) / 1024 * 1024, self.theta); self.cos = cos; self.sin = sin; } }
}
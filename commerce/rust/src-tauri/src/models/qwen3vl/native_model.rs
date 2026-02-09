use std::sync::Arc;
use memmap2::Mmap;
use crate::models::qwen3vl::config::{Qwen3VLConfig, Qwen3VLTextConfig};
use crate::models::qwen3vl::native_backend::*;
#[cfg(feature = "cuda")]
use cudarc::driver::sys::{CUdeviceptr, lib};
use half::f16;
use safetensors::SafeTensors;
use anyhow::{Result, anyhow};
use std::sync::atomic::{Ordering, AtomicU64};
use regex::Regex;
use std::fs::OpenOptions;
use std::io::Write;

pub enum LinearVariant {
    Standard { weight: NativeTensor, bias: Option<NativeTensor> },
    BitSerial { weight_packed: NativeTensor, scales: NativeTensor, bias: Option<NativeTensor> },
}

static ACTIVE_GPU_BYTES: AtomicU64 = AtomicU64::new(0);

fn log_vram_state(stage: &str) {
    #[cfg(feature = "cuda")]
    unsafe {
        let cl = crate::models::qwen3vl::native_backend::lib();
        let mut free: usize = 0;
        let mut total: usize = 0;
        if (cl.cuMemGetInfo_v2(&mut free, &mut total) as i32) == 0 {
            let sys_used = (total - free) / (1024 * 1024);
            let engine_actual = ACTIVE_GPU_BYTES.load(Ordering::Relaxed) / (1024 * 1024);
            let free_mb = free / (1024 * 1024);
            let line = format!("[{:<15}] Sys-Occupied: {:>4} MB | Engine-Data: {:>4} MB | Free: {:>4} MB\n", stage, sys_used, engine_actual, free_mb);
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("vram_diagnostics.log") { let _ = file.write_all(line.as_bytes()); }
            println!("{}", line.trim());
        }
    }
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
                let mut ctx = std::ptr::null_mut() as CUcontext; cl.cuCtxGetCurrent(&mut ctx);
                if ctx == std::ptr::null_mut() && dev >= 0 && dev < 8 {
                    let mut d = 0 as CUdevice; cl.cuDeviceGet(&mut d, dev);
                    cl.cuDevicePrimaryCtxRetain(&mut ctx, d); cl.cuCtxSetCurrent(ctx);
                }
                let mut nkp: CUdeviceptr = 0; let mut nvp: CUdeviceptr = 0;
                if (cl.cuMemAlloc_v2(&mut nkp, kb) as i32) != 0 || (cl.cuMemAlloc_v2(&mut nvp, vb) as i32) != 0 { return; }
                if let (Some(ok), Some(ov)) = (self.k_ptr.take(), self.v_ptr.take()) {
                    if self.current_len > 0 {
                        let _ = cl.cuMemcpyDtoD_v2(nkp, ok.0 as CUdeviceptr, self.current_len * n_kv * (h_d/32) * 4);
                        let _ = cl.cuMemcpyDtoD_v2(nvp, ov.0 as CUdeviceptr, self.current_len * n_kv * h_d * 2);
                    }
                    let _ = cl.cuMemFree_v2(ok.0 as CUdeviceptr); let _ = cl.cuMemFree_v2(ov.0 as CUdeviceptr);
                    let old_sz = self.capacity * (n_kv * (h_d/32) * 4 + n_kv * h_d * 2);
                    ACTIVE_GPU_BYTES.fetch_sub(old_sz as u64, Ordering::Relaxed);
                }
                self.k_ptr = Some(GpuPtr(nkp as *mut _)); self.v_ptr = Some(GpuPtr(nvp as *mut _));
                self.capacity = nc; 
                let new_sz = nc * (n_kv * (h_d/32) * 4 + n_kv * h_d * 2);
                ACTIVE_GPU_BYTES.fetch_add(new_sz as u64, Ordering::Relaxed);
                let _ = cl.cuStreamSynchronize(std::ptr::null_mut());
            }
        }
    }
}

fn log_tensor_health(name: &str, data: &[f16], shape: &[usize]) {
    if data.is_empty() { return; }
    let (mut min, mut max, mut sum) = (3.4e38f32, -3.4e38f32, 0.0f32);
    let mut nan_count = 0;
    for &v in data { 
        let f = v.to_f32(); if f.is_nan() || f.is_infinite() { nan_count += 1; continue; }
        if f < min { min = f; } if f > max { max = f; } sum += f.abs(); 
    }
    let nan_str = if nan_count > 0 { format!(" | NaNs: {}", nan_count) } else { "".to_string() };
    println!("[HEALTH] {:<30} | Shape: {:<12?} | Min: {:>8.4} | Max: {:>8.4} | AbsAvg: {:>8.4}{}", name, shape, if min > 3e38 { 0.0 } else { min }, if max < -3e38 { 0.0 } else { max }, sum / data.len() as f32, nan_str);
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
                    let f_val = if sc == 4 { (b_sum - 8.0f32) * s[n_idx].to_f32() } else { (if b_sum >= 1.0 { 1.0f32 } else { -1.0f32 }) * s[(no * skb + skb_idx) * 8 + sub_n].to_f32() };
                    for r in 0..ratio { let target_idx = n_idx * tk + k_base + (r * sk); if target_idx < out.len() { out[target_idx] = f16::from_f32(f_val); } }
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
    pub fn forward(&self, x: &[f16], gs: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>) -> Vec<f16> {
        let mut out = vec![f16::ZERO; (x.len() / self.in_features) * self.out_features]; self.forward_into(x, &mut out, gs); out
    }
    #[cfg(feature = "cuda")]
    pub fn forward_gpu(&self, di: CUdeviceptr, dog: CUdeviceptr, m: usize) {
        unsafe { match &self.variant {
            LinearVariant::Standard { weight, .. } => crate::models::qwen3vl::native_backend::standard_matmul_cuda_f16(di as *const f16, weight.gpu_ptr.expect("W on GPU").0 as *const f16, dog as *mut f16, m as i32, self.out_features as i32, self.in_features as i32),
            LinearVariant::BitSerial { weight_packed, scales, .. } => crate::models::qwen3vl::native_backend::cuda_matmul_f16(di as *const f16, weight_packed.gpu_ptr.expect("PW on GPU").0 as *const u32, scales.gpu_ptr.expect("S on GPU").0 as *const f16, dog as *mut f16, m as i32, self.out_features as i32, self.in_features as i32, self.device_id, self.src_in as i32),
        } }
    }
    pub fn forward_into(&self, x: &[f16], out: &mut [f16], gs: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>) {
        let m = x.len() / self.in_features;
        match &self.variant {
            LinearVariant::Standard { weight, bias } => {
                #[cfg(feature = "cuda")] if self.device_id >= 0 {
                    if let (Some(w_gpu), Some((si, so, _))) = (weight.gpu_ptr, gs) {
                        let di = si.lock().unwrap().as_ref().map(|(p, _)| p.0 as CUdeviceptr).unwrap_or(0);
                        let dog = so.lock().unwrap().as_ref().map(|(p, _)| p.0 as CUdeviceptr).unwrap_or(0);
                        if di != 0 && dog != 0 { unsafe {
                            let cl = crate::models::qwen3vl::native_backend::lib(); let _ = cl.cuMemcpyHtoD_v2(di, x.as_ptr() as *const _, x.len() * 2);
                            crate::models::qwen3vl::native_backend::standard_matmul_cuda_f16(di as *const f16, w_gpu.0 as *const f16, dog as *mut f16, m as i32, self.out_features as i32, self.in_features as i32);
                            let _ = cl.cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut _, dog, out.len() * 2);
                            if let Some(b) = bias { let br = unsafe { b.get_raw_slice::<f16>() }; for i in 0..m { for j in 0..self.out_features { out[i * self.out_features + j] += br[j]; } } }
                            return;
                        } }
                    }
                }
                native_linear_f16_into(x, weight.get_slice::<f16>().as_ref(), bias.as_ref().map(|b| b.get_slice::<f16>()).as_deref().map(|b| b.as_ref()), out, m, self.out_features, self.in_features);
            },
            LinearVariant::BitSerial { weight_packed, scales, bias } => {
                bit_serial_matmul_f32_extreme_into(x, weight_packed.get_slice::<u32>().as_ref(), scales.get_slice::<f16>().as_ref(), out, m, self.out_features, self.in_features);
                if let Some(b) = bias { let br = unsafe { b.get_raw_slice::<f16>() }; for i in 0..m { for j in 0..self.out_features { out[i * self.out_features + j] += br[j]; } } }
            }
        }
    }
    pub fn move_to_gpu(&mut self, dev: i32) -> anyhow::Result<()> {
        self.device_id = dev; match &mut self.variant {
            LinearVariant::Standard { weight, bias } => { weight.move_to_gpu(dev)?; if let Some(b) = bias { b.move_to_gpu(dev)?; } },
            LinearVariant::BitSerial { weight_packed, scales, bias } => { weight_packed.move_to_gpu(dev)?; scales.move_to_gpu(dev)?; if let Some(b) = bias { b.move_to_gpu(dev)?; } },
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
        let f_alpha = if is_vision { 2.0f32 } else { 1.2f32 };
        #[cfg(feature = "cuda")] if self.device_id >= 0 && self.device_id < 8 && !self.gpu_broken.load(Ordering::Relaxed) {
            if let Some((si, so, sr)) = gs {
                let rb = q_len * h_s.max(self.gate_proj.out_features) * 4;
                let di = si.lock().unwrap().as_ref().map(|(p, s)| if *s >= rb { p.0 as CUdeviceptr } else { 0 }).unwrap_or(0);
                let dog = so.lock().unwrap().as_ref().map(|(p, s)| if *s >= rb { p.0 as CUdeviceptr } else { 0 }).unwrap_or(0);
                let drg = sr.lock().unwrap().as_ref().map(|(p, s)| if *s >= rb { p.0 as CUdeviceptr } else { 0 }).unwrap_or(0);
                if di != 0 && dog != 0 && drg != 0 { unsafe {
                    let cl = crate::models::qwen3vl::native_backend::lib(); let _ = cl.cuMemcpyHtoD_v2(di, x.as_ptr() as *const _, x.len() * 2);
                    let dlw = self.input_layernorm.gpu_ptr.as_ref().unwrap().0 as *const f16;
                    crate::models::qwen3vl::native_backend::cuda_rms_norm_f16(di as *const f16, dlw, dog as *mut f16, q_len as i32, h_s as i32, config.rms_norm_eps as f32);
                    self.q_proj.forward_gpu(dog, di, q_len); let dq = di; self.k_proj.forward_gpu(dog, drg, q_len); let dk = drg;
                    { let mut rgl = rope_gpu.lock().unwrap(); if rgl.is_none() {
                        let cp = crate::models::qwen3vl::native_backend::native_precompute_rope_f16(h_d, 32768, config.rope_theta);
                        let mut ct = NativeTensor::from_raw(cp.0.as_ptr() as *const u8, cp.0.len()*2, vec![32768, h_d/2], NativeDType::F16, self.device_id);
                        let mut st = NativeTensor::from_raw(cp.1.as_ptr() as *const u8, cp.1.len()*2, vec![32768, h_d/2], NativeDType::F16, self.device_id);
                        let _ = ct.move_to_gpu(self.device_id); let _ = st.move_to_gpu(self.device_id); *rgl = Some((ct, st));
                    } if let Some((ref c, ref s)) = *rgl { crate::models::qwen3vl::native_backend::native_cuda_apply_rope(dq, dk, c.gpu_ptr.unwrap().0 as CUdeviceptr, s.gpu_ptr.unwrap().0 as CUdeviceptr, q_len, s_o, n_h, n_kv, h_d); } }
                    let mut gkg = self.gpu_kv_cache.lock().unwrap(); let n_tok = s_o + q_len; gkg.grow(n_tok, n_kv, h_d, self.device_id);
                    if let (Some(kp), Some(vp)) = (gkg.k_ptr, gkg.v_ptr) {
                        let dkd = (kp.0 as u64 + (s_o * n_kv * (h_d/32) * 4) as u64) as CUdeviceptr;
                        let dvd = (vp.0 as u64 + (s_o * n_kv * h_d * 2) as u64) as CUdeviceptr;
                        crate::models::qwen3vl::native_backend::native_cuda_pack_bits(dk, dkd, q_len * n_kv * h_d);
                        self.v_proj.forward_gpu(dog, dvd, q_len); gkg.current_len = n_tok;
                    }
                    crate::models::qwen3vl::native_backend::native_bit_serial_attn_gpu_buffered(&[], gkg.k_ptr.unwrap(), gkg.v_ptr.unwrap(), n_h, n_kv, h_d, n_tok, self.device_id as usize, dq, dog, f_alpha, self.q_proj.src_in / n_h, true, q_len);
                    let _ = cl.cuMemcpyHtoD_v2(di, x.as_ptr() as *const _, x.len() * 2);
                    crate::models::qwen3vl::native_backend::native_cuda_add_inplace(dog, di, x.len());
                    let dpw = self.post_attention_layernorm.gpu_ptr.as_ref().unwrap().0 as *const f16;
                    crate::models::qwen3vl::native_backend::cuda_rms_norm_f16(dog as *const f16, dpw, di as *mut f16, q_len as i32, h_s as i32, config.rms_norm_eps as f32);
                    let d_res = dog; let d_up = (dog as u64 + (q_len * h_s * 2) as u64) as CUdeviceptr;
                    self.gate_proj.forward_gpu(di, drg, q_len); self.up_proj.forward_gpu(di, d_up, q_len);
                    crate::models::qwen3vl::native_backend::native_cuda_silu_inplace(drg, q_len * config.intermediate_size);
                    crate::models::qwen3vl::native_backend::native_cuda_element_mul(drg, d_up, q_len * config.intermediate_size);
                    self.down_proj.forward_gpu(drg, di, q_len);
                    if self.down_proj.out_features > self.down_proj.src_out { 
                        crate::models::qwen3vl::native_backend::native_cuda_hybrid_repeat(di, self.down_proj.src_out, self.down_proj.out_features, q_len);
                        let gain = (self.down_proj.src_out as f32 / self.down_proj.out_features as f32).sqrt();
                        crate::models::qwen3vl::native_backend::native_cuda_apply_gain(di, gain, q_len * self.down_proj.out_features);
                    }
                    crate::models::qwen3vl::native_backend::native_cuda_add_inplace(di, d_res, x.len());
                    let rs = std::slice::from_raw_parts_mut(out_ptr, x.len()); let _ = cl.cuMemcpyDtoH_v2(rs.as_mut_ptr() as *mut _, di, x.len() * 2); return rs;
                } }
            }
        }
        let lnw = self.input_layernorm.get_slice::<f16>(); let mut cn = vec![f16::ZERO; x.len()];
        native_rms_norm_f16_into(x, lnw.as_ref(), config.rms_norm_eps as f32, h_s, &mut cn);
        self.q_proj.forward_into(&cn, &mut ws.q, gs); self.k_proj.forward_into(&cn, &mut ws.k, gs); self.v_proj.forward_into(&cn, &mut ws.v, gs);
        native_apply_rope_f16_with_offset(&mut ws.q, &mut ws.k, q_len, s_o, n_h, h_d, config.rope_theta, rc, rs);
        let mut cache = self.kv_cache.lock().unwrap(); let nl = s_o + q_len; cache.grow(nl, n_kv, h_d);
        let kp = pack_f16_to_bits(&ws.k); cache.k[s_o*(n_kv*h_d/32)..s_o*(n_kv*h_d/32)+kp.len()].copy_from_slice(&kp);
        cache.v[s_o*(n_kv*h_d)..s_o*(n_kv*h_d)+ws.v.len()].copy_from_slice(&ws.v);
        let ao = native_bit_serial_attn_f16(&ws.q, &cache.k, &cache.v, h_s, n_h, n_kv, q_len, nl, f_alpha);
        let os = unsafe { std::slice::from_raw_parts_mut(out_ptr, x.len()) }; self.o_proj.forward_into(&ao, os, gs);
        for i in 0..x.len() { os[i] += x[i]; } let plnw = self.post_attention_layernorm.get_slice::<f16>();
        native_rms_norm_f16_into(os, plnw.as_ref(), config.rms_norm_eps as f32, h_s, &mut ws.intermediate_a[..x.len()]);
        self.gate_proj.forward_into(&ws.intermediate_a[..x.len()], &mut ws.intermediate_b, gs); self.up_proj.forward_into(&ws.intermediate_a[..x.len()], &mut ws.intermediate_c, gs);
        native_silu_f16(&mut ws.intermediate_b[..q_len * config.intermediate_size]);
        for i in 0..q_len * config.intermediate_size { ws.intermediate_b[i] *= ws.intermediate_c[i]; }
        let mut moh = vec![f16::ZERO; x.len()]; self.down_proj.forward_into(&ws.intermediate_b[..q_len * config.intermediate_size], &mut moh, gs);
        if self.down_proj.out_features > self.down_proj.src_out { 
            for t in 0..q_len { for i in self.down_proj.src_out..self.down_proj.out_features { moh[t*self.down_proj.out_features+i] = moh[t*self.down_proj.out_features+(i%self.down_proj.src_out)]; } }
            let gain = (self.down_proj.src_out as f32 / self.down_proj.out_features as f32).sqrt();
            for v in moh.iter_mut() { *v = f16::from_f32(v.to_f32() * gain); }
        }
        for i in 0..x.len() { os[i] += moh[i]; } os
    }
    pub fn clear_kv_cache(&self) { self.kv_cache.lock().unwrap().current_len = 0; self.gpu_kv_cache.lock().unwrap().current_len = 0; }
    pub fn get_kv_data(&self, hd: usize, nkv: usize, st: usize) -> Option<(Vec<u32>, Vec<f16>)> {
        #[cfg(feature = "cuda")] if self.device_id >= 0 {
            let g = self.gpu_kv_cache.lock().unwrap(); if let (Some(kp), Some(vp)) = (g.k_ptr, g.v_ptr) { if g.current_len > st {
                let el = g.current_len - st; let ku = nkv * (hd/32); let vu = nkv * hd; let mut kh = vec![0u32; el * ku]; let mut vh = vec![f16::ZERO; el * vu];
                unsafe { let cl = crate::models::qwen3vl::native_backend::lib(); let _ = cl.cuMemcpyDtoH_v2(kh.as_mut_ptr() as *mut _, (kp.0 as u64 + (st * ku * 4) as u64) as CUdeviceptr, kh.len()*4); let _ = cl.cuMemcpyDtoH_v2(vh.as_mut_ptr() as *mut _, (vp.0 as u64 + (st * vu * 2) as u64) as CUdeviceptr, vh.len()*2); }
                return Some((kh, vh));
            } }
        }
        let c = self.kv_cache.lock().unwrap(); if c.current_len > st { 
            let ku = nkv * (hd/32); let vu = nkv * hd; 
            Some((c.k[st * ku .. c.current_len * ku].to_vec(), c.v[st * vu .. c.current_len * vu].to_vec())) 
        } else { None }
    }
}

pub struct NativeVisionModel { pub patch_embed: NativeLinear, pub blocks: Vec<NativeLayer>, pub merger: NativeLayer }
impl NativeVisionModel {
    pub fn move_to_gpu(&mut self, dev: i32) -> anyhow::Result<()> { self.patch_embed.move_to_gpu(dev)?; for b in &mut self.blocks { b.move_to_gpu(dev)?; } self.merger.move_to_gpu(dev)?; Ok(()) }
    pub fn forward<'a>(&self, pv: &[f16], gt: &[u32; 3], rc: &[f16], rs: &[f16], gs: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>, ws: &'a mut ForwardWorkspace, rope_gpu: &std::sync::Mutex<Option<(NativeTensor, NativeTensor)>>) -> &'a [f16] {
        let eo = self.patch_embed.forward(pv, gs); ws.hidden_a[..eo.len()].copy_from_slice(&eo); let mut cur_x: &[f16] = unsafe { std::slice::from_raw_parts(ws.hidden_a.as_ptr(), eo.len()) };
        for (i, block) in self.blocks.iter().enumerate() { let out = block.forward(cur_x, &Qwen3VLTextConfig { hidden_size: self.patch_embed.out_features, intermediate_size: block.gate_proj.out_features, num_hidden_layers: 1, num_attention_heads: 16, num_key_value_heads: 16, head_dim: self.patch_embed.out_features / 16, rms_norm_eps: 1e-6, rope_theta: 10000.0, vocab_size: 0, max_position_embeddings: 32768, dtype: None, rope_scaling: None }, 0, 0, rc, rs, false, true, gs, ws, i % 2 != 0, rope_gpu); cur_x = unsafe { std::slice::from_raw_parts(out.as_ptr(), out.len()) }; }
        self.merger.forward(cur_x, &Qwen3VLTextConfig { hidden_size: cur_x.len() / (gt[1] * gt[2]) as usize, intermediate_size: cur_x.len() / (gt[1] * gt[2]) as usize, num_hidden_layers: 1, num_attention_heads: 1, num_key_value_heads: 1, head_dim: cur_x.len() / (gt[1] * gt[2]) as usize, rms_norm_eps: 1e-6, rope_theta: 10000.0, vocab_size: 0, max_position_embeddings: 32768, dtype: None, rope_scaling: None }, 0, 0, rc, rs, false, true, gs, ws, self.blocks.len() % 2 == 0, rope_gpu)
    }
}

pub struct NativeQwen3TextModel {
    pub config: Qwen3VLTextConfig, pub embed_tokens: NativeLinear, pub layers: Vec<NativeLayer>, pub norm: NativeTensor,
    pub rope_cache: std::sync::Mutex<RopeCache>, pub rope_cache_gpu: std::sync::Mutex<Option<(NativeTensor, NativeTensor)>>,
}

impl NativeQwen3TextModel {
    pub fn forward_ext(&self, _i_ids: &[u32], embeds: Vec<f16>, s_o: usize, gs: Option<(&std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>, &std::sync::Mutex<Option<(GpuPtr, usize)>>)>, workspace: Option<&mut ForwardWorkspace>, _sl: Option<&NativeLayer>, is_vision: bool) -> Vec<f16> {
        let hid = self.config.hidden_size; let q_len = embeds.len() / hid;
        let mut int_ws = ForwardWorkspace::new(); let ws = workspace.unwrap_or(&mut int_ws);
        ws.ensure_capacity(hid, self.config.intermediate_size, q_len, self.config.num_attention_heads, self.config.num_key_value_heads, self.config.head_dim);
        ws.hidden_a[..embeds.len()].copy_from_slice(&embeds); let mut cur_x: &[f16] = unsafe { std::slice::from_raw_parts(ws.hidden_a.as_ptr(), embeds.len()) };
        
        // [LOG-DIET] Only log health during prefill (q_len > 1) to keep the output clean
        if q_len > 1 { log_tensor_health("Embeddings", cur_x, &[q_len, hid]); }
        
        { let mut rg = self.rope_cache.lock().unwrap(); rg.ensure_length(s_o + q_len); }
        let rg = self.rope_cache.lock().unwrap();
        for (i, layer) in self.layers.iter().enumerate() {
            let out = layer.forward(cur_x, &self.config, s_o, i, &rg.cos, &rg.sin, false, is_vision, gs, ws, i % 2 == 0, &self.rope_cache_gpu);
            
            // [2026-STABILITY-DAM-V2] Refined Adaptive Signal Control
            let mut stabilized_out = out.to_vec();
            let num_layers = self.layers.len();
            
            if i == 0 && num_layers > 1 {
                // Layer 0 (4-bit) -> Layer 1 (1-bit): Use 0.45x Dampen for better signal
                for v in stabilized_out.iter_mut() {
                    let mut f = v.to_f32() * 0.45;
                    if f > 16.0 { f = 16.0; } if f < -16.0 { f = -16.0; }
                    *v = f16::from_f32(f);
                }
            } else if i > 0 {
                // Progressive Clamping: Allow more variance in deeper layers
                let clamp_range = 16.0 + (i as f32 / num_layers as f32) * 8.0; 
                for v in stabilized_out.iter_mut() {
                    let mut f = v.to_f32();
                    if f > clamp_range { f = clamp_range; } if f < -clamp_range { f = -clamp_range; }
                    *v = f16::from_f32(f);
                }
            }

            if q_len > 1 && (i == 0 || i == num_layers - 1) { 
                log_tensor_health(&format!("Layer {} Output (Stabilized-V2)", i), &stabilized_out, &[q_len, hid]); 
            }
            
            // Since we created a new Vec for stabilization, we need to manage its lifetime.
            // For efficiency, we copy it back to the workspace buffer.
            let target_ptr = if i % 2 == 0 { ws.hidden_b.as_mut_ptr() } else { ws.hidden_a.as_mut_ptr() };
            unsafe { std::ptr::copy_nonoverlapping(stabilized_out.as_ptr(), target_ptr, stabilized_out.len()); }
            cur_x = unsafe { std::slice::from_raw_parts(target_ptr, stabilized_out.len()) };
        }
        let n_w = self.norm.get_slice::<f16>(); let mut cn = vec![f16::ZERO; cur_x.len()]; 
        native_rms_norm_f16_into(cur_x, n_w.as_ref(), self.config.rms_norm_eps as f32, hid, &mut cn);
        
        // [2026-FINAL-ATTENUATION-V2] 0.05 was too quiet, 0.18 is the 'sweet spot'.
        // This allows semantic signals to survive while keeping 1-bit noise under control.
        for v in cn.iter_mut() {
            let mut f = v.to_f32() * 0.18;
            if f > 4.0 { f = 4.0; } if f < -4.0 { f = -4.0; } // Sane range for 4-bit LM Head
            *v = f16::from_f32(f);
        }

        if q_len > 1 { log_tensor_health("Final Norm Output (Corrected)", &cn, &[q_len, hid]); }
        cn
    }
    pub fn move_to_gpu(&mut self, dev: i32) -> anyhow::Result<()> { self.embed_tokens.move_to_gpu(dev)?; self.norm.move_to_gpu(dev)?; for l in &mut self.layers { l.move_to_gpu(dev)?; } Ok(()) }
    pub fn force_free_kv_cache(&self) { for l in &self.layers { l.kv_cache.lock().unwrap().k = Vec::new(); l.kv_cache.lock().unwrap().v = Vec::new(); } }
    pub fn batch_upload_stitched_cache(&self, k: Vec<u32>, v: Vec<f16>) { for l in &self.layers { l.kv_cache.lock().unwrap().k = k.clone(); l.kv_cache.lock().unwrap().v = v.clone(); } }
}

pub struct NativeQwen3VLModel {
    pub config: Qwen3VLConfig, pub text_model: NativeQwen3TextModel, pub lm_head: NativeLinear, pub visual: Option<NativeVisionModel>,
    pub support_layer0: Option<NativeLayer>, pub support_workspace: std::sync::Mutex<ForwardWorkspace>,
    pub global_scratch_i: std::sync::Mutex<Option<(GpuPtr, usize)>>, pub global_scratch_o: std::sync::Mutex<Option<(GpuPtr, usize)>>, pub global_scratch_r: std::sync::Mutex<Option<(GpuPtr, usize)>>,
    pub workspace: std::sync::Mutex<ForwardWorkspace>,
    pub align_matrix: Option<NativeLinear>, 
}

impl NativeQwen3VLModel {
    pub fn get_kv_len(&self) -> usize { 
        let gpu_len = self.text_model.layers[0].gpu_kv_cache.lock().unwrap().current_len;
        if gpu_len > 0 { return gpu_len; }
        self.text_model.layers[0].kv_cache.lock().unwrap().current_len
    }
    pub fn clear_kv_cache(&self) { self.text_model.layers.iter().for_each(|l| l.clear_kv_cache()); }
    pub fn get_all_kv(&self, st: usize) -> Vec<(Vec<u32>, Vec<f16>)> { 
        let hd = self.text_model.config.head_dim; let nkv = self.text_model.config.num_key_value_heads;
        self.text_model.layers.iter().filter_map(|l| l.get_kv_data(hd, nkv, st)).collect()
    }
    pub fn load(config: Qwen3VLConfig, m_mmap: Arc<Mmap>, v_mmap: Option<Arc<Mmap>>, baking: bool, s_mmap: Option<Arc<Mmap>>, dev_id: i32) -> Result<Self> {
        let align_path = std::path::Path::new("src-tauri/models/align_matrix.safetensors");
        let align_matrix = if align_path.exists() {
            println!("[MODEL] Loading alignment matrix...");
            if let Ok(data) = std::fs::read(align_path) {
                if let Ok(st_align) = SafeTensors::deserialize(&data) {
                    if let Ok(weight) = st_align.tensor("weight") {
                        let shape = weight.shape().to_vec();
                        Some(NativeLinear {
                            in_features: shape[0], out_features: shape[1], src_in: shape[0], src_out: shape[1],
                            variant: LinearVariant::Standard {
                                weight: NativeTensor::from_raw(weight.data().as_ptr(), weight.data().len(), shape, NativeDType::F16, dev_id),
                                bias: None
                            },
                            device_id: dev_id
                        })
                    } else { None }
                } else { None }
            } else { None }
        } else { None };

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
                if format == 4 || inf > si || outf > so {
                    let dq = dequantize_bit_serial_to_f16(unsafe { std::slice::from_raw_parts(vp.data().as_ptr() as *const u32, vp.data().len()/4) }, unsafe { std::slice::from_raw_parts(vs.data().as_ptr() as *const f16, vs.data().len()/2) }, so, si, inf, if format == 4 { 4 } else { 1 });
                    let b = dq.into_boxed_slice(); let p = b.as_ptr(); std::mem::forget(b);
                    return Ok(NativeLinear { in_features: inf, out_features: outf, src_in: si, src_out: so, variant: LinearVariant::Standard { weight: NativeTensor::from_raw(p as *const u8, outf*inf*2, vec![outf, inf], NativeDType::F16, dev_id), bias: None }, device_id: dev_id });
                }
                Ok(NativeLinear { in_features: inf, out_features: outf, src_in: inf, src_out: outf, variant: LinearVariant::BitSerial { weight_packed: NativeTensor::from_mmap(cm.clone(), unsafe { vp.data().as_ptr().offset_from(cm.as_ptr()) } as usize, vp.shape().to_vec(), NativeDType::U32), scales: NativeTensor::from_mmap(cm.clone(), unsafe { vs.data().as_ptr().offset_from(cm.as_ptr()) } as usize, vs.shape().to_vec(), NativeDType::F16), bias: None }, device_id: dev_id })
            } else {
                let v = cst.tensor(&key)?; let o = unsafe { v.data().as_ptr().offset_from(cm.as_ptr()) } as usize;
                Ok(NativeLinear { in_features: inf, out_features: outf, src_in: inf, src_out: outf, variant: LinearVariant::Standard { weight: NativeTensor::from_mmap(cm.clone(), o, v.shape().to_vec(), NativeDType::F16), bias: None }, device_id: dev_id })
            }
        };
        let get_t = |name: &str, ts: usize| -> Result<NativeTensor> {
            let (key, is_p) = find_key(name, &st, st_sec.as_ref()).ok_or_else(|| anyhow!("TNotFound: {}", name))?;
            let cst = if is_p { &st } else { st_sec.as_ref().unwrap() }; let cm = if is_p { &m_mmap } else { s_mmap.as_ref().unwrap() };
            let v = cst.tensor(&key)?; if ts > v.data().len()/2 { let mut nd = Vec::with_capacity(ts*2); let sf = unsafe { std::slice::from_raw_parts(v.data().as_ptr() as *const f16, v.data().len()/2) }; for _ in 0..(ts/(v.data().len()/2)) { for &val in sf { nd.extend_from_slice(&val.to_le_bytes()); } } let b = nd.into_boxed_slice(); let p = b.as_ptr(); std::mem::forget(b); Ok(NativeTensor::from_raw(p as *const u8, ts*2, vec![ts], NativeDType::F16, dev_id)) }
            else { Ok(NativeTensor::from_mmap(cm.clone(), unsafe { v.data().as_ptr().offset_from(cm.as_ptr()) } as usize, v.shape().to_vec(), NativeDType::F16)) }
        };
        let get_embed = |base: &str, vocab: usize, thid: usize| -> Result<NativeLinear> {
            let (key, is_p) = find_key(base, &st, st_sec.as_ref()).ok_or_else(|| anyhow!("EmbedNotFound: {}", base))?;
            let cst = if is_p { &st } else { st_sec.as_ref().unwrap() }; let cm = if is_p { &m_mmap } else { s_mmap.as_ref().unwrap() };
            if cst.tensor(&format!("{}.packed", key)).is_ok() {
                let vp = cst.tensor(&format!("{}.packed", key))?;
                let vs = cst.tensor(&format!("{}.scales", key))?;
                let format = cst.tensor(&format!("{}.format", key)).map(|t| t.data()[0] as i8).unwrap_or(1);
                let (so, si) = (vocab, thid);
                let dq = dequantize_bit_serial_to_f16(
                    unsafe { std::slice::from_raw_parts(vp.data().as_ptr() as *const u32, vp.data().len()/4) },
                    unsafe { std::slice::from_raw_parts(vs.data().as_ptr() as *const f16, vs.data().len()/2) },
                    so, si, thid, if format == 4 { 4 } else { 1 }
                );
                let b = dq.into_boxed_slice(); let p = b.as_ptr(); std::mem::forget(b);
                return Ok(NativeLinear { in_features: vocab, out_features: thid, src_in: thid, src_out: thid, variant: LinearVariant::Standard { weight: NativeTensor::from_raw(p as *const u8, vocab*thid*2, vec![vocab, thid], NativeDType::F16, dev_id), bias: None }, device_id: dev_id });
            }
            get_l(base, vocab, thid)
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
        let norm = match get_t("model.language_model.norm.weight", t_c.hidden_size) { Ok(t) => t, Err(_) => if baking { NativeTensor::from_raw(vec![0u8; t_c.hidden_size*2].as_ptr(), t_c.hidden_size*2, vec![t_c.hidden_size], NativeDType::F16, dev_id) } else { return Err(anyhow!("NormNotFound")); } };
        let head = get_l("model.language_model.lm_head.weight", t_c.hidden_size, 151936).or_else(|_| get_l("model.language_model.embed_tokens.weight", t_c.hidden_size, 151936))?;
        let mut model = Self {
            config: config.clone(), text_model: NativeQwen3TextModel { config: t_c.clone(), embed_tokens: emb, layers, norm, 
                rope_cache: std::sync::Mutex::new(RopeCache::new(t_c.head_dim, t_c.rope_theta, 32768)), 
                rope_cache_gpu: std::sync::Mutex::new(None) 
            },
            lm_head: head, visual: None, support_layer0: None, support_workspace: std::sync::Mutex::new(ForwardWorkspace::new()), global_scratch_i: std::sync::Mutex::new(None), global_scratch_o: std::sync::Mutex::new(None), global_scratch_r: std::sync::Mutex::new(None), workspace: std::sync::Mutex::new(ForwardWorkspace::new()),
            align_matrix,
        };
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
                if format == 4 || inf > si || outf > so {
                    let dq = dequantize_bit_serial_to_f16(unsafe { std::slice::from_raw_parts(vp.data().as_ptr() as *const u32, vp.data().len()/4) }, unsafe { std::slice::from_raw_parts(vs.data().as_ptr() as *const f16, vs.data().len()/2) }, so, si, inf, if format == 4 { 4 } else { 1 });
                    let b = dq.into_boxed_slice(); let p = b.as_ptr(); std::mem::forget(b);
                    return Ok(NativeLinear { in_features: inf, out_features: outf, src_in: si, src_out: so, variant: LinearVariant::Standard { weight: NativeTensor::from_raw(p as *const u8, outf*inf*2, vec![outf, inf], NativeDType::F16, dev_id), bias: None }, device_id: dev_id });
                }
                Ok(NativeLinear { in_features: inf, out_features: outf, src_in: inf, src_out: outf, variant: LinearVariant::BitSerial { weight_packed: NativeTensor::from_mmap(cm.clone(), unsafe { vp.data().as_ptr().offset_from(cm.as_ptr()) } as usize, vp.shape().to_vec(), NativeDType::U32), scales: NativeTensor::from_mmap(cm.clone(), unsafe { vs.data().as_ptr().offset_from(cm.as_ptr()) } as usize, vs.shape().to_vec(), NativeDType::F16), bias: None }, device_id: dev_id })
            } else {
                let v = cst.tensor(&key)?; let o = unsafe { v.data().as_ptr().offset_from(cm.as_ptr()) } as usize;
                Ok(NativeLinear { in_features: inf, out_features: outf, src_in: inf, src_out: outf, variant: LinearVariant::Standard { weight: NativeTensor::from_mmap(cm.clone(), o, v.shape().to_vec(), NativeDType::F16), bias: None }, device_id: dev_id })
            }
        };
        let get_t = |name: &str, ts: usize| -> Result<NativeTensor> {
            let (key, is_p) = find_key(name, &st, st_sec.as_ref()).ok_or_else(|| anyhow!("TNotFound: {}", name))?;
            let cst = if is_p { &st } else { st_sec.as_ref().unwrap() }; let cm = if is_p { &m_mmap } else { s_mmap.as_ref().unwrap() };
            let v = cst.tensor(&key)?; if ts > v.data().len()/2 { let mut nd = Vec::with_capacity(ts*2); let sf = unsafe { std::slice::from_raw_parts(v.data().as_ptr() as *const f16, v.data().len()/2) }; for _ in 0..(ts/(v.data().len()/2)) { for &val in sf { nd.extend_from_slice(&val.to_le_bytes()); } } let b = nd.into_boxed_slice(); let p = b.as_ptr(); std::mem::forget(b); Ok(NativeTensor::from_raw(p as *const u8, ts*2, vec![ts], NativeDType::F16, dev_id)) }
            else { Ok(NativeTensor::from_mmap(cm.clone(), unsafe { v.data().as_ptr().offset_from(cm.as_ptr()) } as usize, v.shape().to_vec(), NativeDType::F16)) }
        };
        let get_embed = |base: &str, vocab: usize, thid: usize| -> Result<NativeLinear> {
            let (key, is_p) = find_key(base, &st, st_sec.as_ref()).ok_or_else(|| anyhow!("EmbedNotFound: {}", base))?;
            let cst = if is_p { &st } else { st_sec.as_ref().unwrap() }; let cm = if is_p { &m_mmap } else { s_mmap.as_ref().unwrap() };
            
            // [FIX] Ensure we don't calculate insane sizes for embeddings
            if cst.tensor(&format!("{}.packed", key)).is_ok() {
                let vp = cst.tensor(&format!("{}.packed", key))?;
                let vs = cst.tensor(&format!("{}.scales", key))?;
                let format = cst.tensor(&format!("{}.format", key)).map(|t| t.data()[0] as i8).unwrap_or(1);
                
                // Safe dimension extraction
                let (so, si) = (vocab, thid);
                let dq = dequantize_bit_serial_to_f16(
                    unsafe { std::slice::from_raw_parts(vp.data().as_ptr() as *const u32, vp.data().len()/4) },
                    unsafe { std::slice::from_raw_parts(vs.data().as_ptr() as *const f16, vs.data().len()/2) },
                    so, si, thid, if format == 4 { 4 } else { 1 }
                );
                let b = dq.into_boxed_slice(); let p = b.as_ptr(); std::mem::forget(b);
                return Ok(NativeLinear { in_features: vocab, out_features: thid, src_in: thid, src_out: thid, variant: LinearVariant::Standard { weight: NativeTensor::from_raw(p as *const u8, vocab*thid*2, vec![vocab, thid], NativeDType::F16, dev_id), bias: None }, device_id: dev_id });
            }
            get_l(base, vocab, thid)
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
        let norm = match get_t("model.language_model.norm.weight", t_c.hidden_size) { Ok(t) => t, Err(_) => if baking { NativeTensor::from_raw(vec![0u8; t_c.hidden_size*2].as_ptr(), t_c.hidden_size*2, vec![t_c.hidden_size], NativeDType::F16, dev_id) } else { return Err(anyhow!("NormNotFound")); } };
        let head = get_l("model.language_model.lm_head.weight", t_c.hidden_size, 151936).or_else(|_| get_l("model.language_model.embed_tokens.weight", t_c.hidden_size, 151936))?;
        let mut model = Self {
            config: config.clone(), text_model: NativeQwen3TextModel { config: t_c.clone(), embed_tokens: emb, layers, norm, 
                // [2026-ROPE-RESTORE] Revert to native 5,000,000 for 2B-VL's long-context intelligence.
                rope_cache: std::sync::Mutex::new(RopeCache::new(t_c.head_dim, 5000000.0, 32768)), 
                rope_cache_gpu: std::sync::Mutex::new(None) 
            },
            lm_head: head, visual: None, support_layer0: None, support_workspace: std::sync::Mutex::new(ForwardWorkspace::new()), global_scratch_i: std::sync::Mutex::new(None), global_scratch_o: std::sync::Mutex::new(None), global_scratch_r: std::sync::Mutex::new(None), workspace: std::sync::Mutex::new(ForwardWorkspace::new()),
            align_matrix,
        };
        #[cfg(feature = "cuda")] if dev_id >= 0 { 
            log_vram_state("BEFORE-OFFLOAD"); let _ = model.move_to_gpu(dev_id); log_vram_state("AFTER-OFFLOAD");
            let pb = 2048 * t_c.hidden_size.max(t_c.intermediate_size) * 4; 
            unsafe { let cl = crate::models::qwen3vl::native_backend::lib(); let (mut p1, mut p2, mut p3) = (0, 0, 0); 
                if (cl.cuMemAlloc_v2(&mut p1, pb) as i32) == 0 { *model.global_scratch_i.get_mut().unwrap() = Some((GpuPtr(p1 as *mut _), pb)); ACTIVE_GPU_BYTES.fetch_add(pb as u64, Ordering::Relaxed); } 
                if (cl.cuMemAlloc_v2(&mut p2, pb) as i32) == 0 { *model.global_scratch_o.get_mut().unwrap() = Some((GpuPtr(p2 as *mut _), pb)); ACTIVE_GPU_BYTES.fetch_add(pb as u64, Ordering::Relaxed); } 
                if (cl.cuMemAlloc_v2(&mut p3, pb) as i32) == 0 { *model.global_scratch_r.get_mut().unwrap() = Some((GpuPtr(p3 as *mut _), pb)); ACTIVE_GPU_BYTES.fetch_add(pb as u64, Ordering::Relaxed); } 
            } 
        }
        Ok(model)
    }
    pub fn load_kv_stitched(&self, paths: &[std::path::PathBuf]) -> Result<usize> {
        let n_kv = self.text_model.config.num_key_value_heads; let h_d = self.text_model.config.head_dim;
        let mut expanded = Vec::new();
        
        // [SHARD-EXPANSION] Resolve base patterns into individual shard files
        for p in paths {
            if p.exists() && p.is_file() {
                expanded.push(p.clone());
            } else {
                let dir = p.parent().unwrap_or(std::path::Path::new("."));
                let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                if let Ok(entries) = std::fs::read_dir(dir) {
                    let mut shards = Vec::new();
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                        // Match both exact name OR shard pattern
                        if name.starts_with(stem) && name.ends_with(".safetensors") {
                            shards.push(path);
                        }
                    }
                    // Crucial: Sort numerically (shard0, shard1, ..., shard11)
                    let re = Regex::new(r"shard(\d+)").unwrap();
                    shards.sort_by_key(|p| {
                        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                        re.captures(name).and_then(|c| c[1].parse::<usize>().ok()).unwrap_or(0)
                    });
                    expanded.extend(shards);
                }
            }
        }
        
        // Remove duplicates just in case
        expanded.dedup();
        
        let mut total_t = 0; let mut metadatas = Vec::new(); let mut mmaps = Vec::new();
        for path in &expanded {
            println!("[DIAG-KV] Attempting to open memory shard: {:?}", path);
            let file = std::fs::File::open(path)?; 
            let mmap = Arc::new(unsafe { memmap2::MmapOptions::new().map(&file)? });
            let tokens = { 
                let st = SafeTensors::deserialize(&mmap)?; 
                if let Ok(kt) = st.tensor("layer.0.k") {
                    let t = kt.data().len() / (n_kv * (h_d/32) * 4);
                    println!("[DIAG-KV] Shard loaded: {} tokens", t);
                    t
                } else { 0 }
            };
            if tokens > 0 { total_t += tokens; metadatas.push(tokens); mmaps.push(mmap); }
        }
        
        // [QUALITY-GUARD] Enforce 500 token minimum for meaningful analysis
        if total_t < 500 {
            println!("[KV-ERROR] Context too small ({} tokens). Minimum 500 required.", total_t);
            return Err(anyhow!("Insufficient context tokens: {}. Expected at least 500.", total_t));
        }

        for l_idx in 0..self.text_model.layers.len() {
            let layer = &self.text_model.layers[l_idx]; let mut gpu_kv = layer.gpu_kv_cache.lock().unwrap();
            let target_dim = self.text_model.config.hidden_size;
            
            #[cfg(feature = "cuda")] if self.lm_head.device_id >= 0 { gpu_kv.grow(total_t, n_kv, h_d, self.lm_head.device_id); }
            let mut curr_o = 0;
            for (p_idx, mmap) in mmaps.iter().enumerate() {
                let st = SafeTensors::deserialize(mmap)?; let tip = metadatas[p_idx];
                let mut fl = 0; while st.tensor(&format!("layer.{}.k", fl)).is_ok() { fl += 1; }
                let si = if fl == 1 { 0 } else { l_idx }; if si >= fl { curr_o += tip; continue; }
                if let (Ok(kt), Ok(vt)) = (st.tensor(&format!("layer.{}.k", si)), st.tensor(&format!("layer.{}.v", si))) {
                    let mut k_data: Vec<u32> = kt.data().chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
                    let mut v_data: Vec<f16> = vt.data().chunks_exact(2).map(|c| f16::from_ne_bytes(c.try_into().unwrap())).collect();
                    
                    let actual_source_dim = if k_data.len() > 0 && v_data.len() > 0 { (v_data.len() * 32) / k_data.len() } else { 1024 };
                    
                    // [2026-SEMANTIC-ALIGNMENT] Project 0.6B memories into 2B space
                    if let Some(ref align) = self.align_matrix {
                        if target_dim > actual_source_dim && actual_source_dim == align.in_features {
                            v_data = align.forward(&v_data, None);
                            let mut k_f16 = Vec::with_capacity(k_data.len() * 32);
                            for &packed in &k_data {
                                for b in 0..32 {
                                    k_f16.push(if (packed >> b) & 1 == 1 { f16::from_f32(1.0) } else { f16::from_f32(-1.0) });
                                }
                            }
                            let k_projected = align.forward(&k_f16, None);
                            k_data = pack_f16_to_bits(&k_projected);
                        }
                    }

                    #[cfg(feature = "cuda")] if self.lm_head.device_id >= 0 { unsafe {
                        let cl = crate::models::qwen3vl::native_backend::lib();
                        if let Some(kp) = gpu_kv.k_ptr { let _ = cl.cuMemcpyHtoD_v2((kp.0 as u64 + (curr_o * n_kv * (h_d/32) * 4) as u64) as CUdeviceptr, k_data.as_ptr() as *const _, k_data.len() * 4); }
                        if let Some(vp) = gpu_kv.v_ptr { let _ = cl.cuMemcpyHtoD_v2((vp.0 as u64 + (curr_o * n_kv * h_d * 2) as u64) as CUdeviceptr, v_data.as_ptr() as *const _, v_data.len() * 2); }
                    } }
                }
                curr_o += tip;
            }
            gpu_kv.current_len = total_t;
            
            // [STABILITY] Gain Correction (0.50x) to neutralize energy boost from 0.6B->2B dimension expansion
            #[cfg(feature = "cuda")] if self.lm_head.device_id >= 0 {
                unsafe {
                    if let Some(kp) = gpu_kv.k_ptr { 
                        crate::models::qwen3vl::native_backend::native_cuda_apply_gain(kp.0 as CUdeviceptr, 0.50, (total_t * n_kv * (h_d/32)) as usize); 
                    }
                    if let Some(vp) = gpu_kv.v_ptr { 
                        crate::models::qwen3vl::native_backend::native_cuda_apply_gain(vp.0 as CUdeviceptr, 0.50, (total_t * n_kv * h_d) as usize); 
                    }
                }
            }
        }
        Ok(total_t)
    }
    pub fn forward(&self, i_ids: &[u32], pv: Option<&[f16]>, gt: Option<&[u32; 3]>, so: usize) -> Vec<f16> {
        let is_b = self.text_model.layers.len() <= 1; let mut wl = if is_b { self.support_workspace.lock().unwrap() } else { self.workspace.lock().unwrap() };
        let embeds = match &self.text_model.embed_tokens.variant { LinearVariant::Standard { weight, .. } => native_embedding_lookup_f16(i_ids, weight.get_slice::<f16>().as_ref(), self.text_model.config.hidden_size), _ => vec![f16::ZERO; i_ids.len() * self.text_model.config.hidden_size] };
        let sc = Some((&self.global_scratch_i, &self.global_scratch_o, &self.global_scratch_r)); let mut fe = embeds;
        if let (Some(pvv), Some(gtv)) = (pv, gt) { if let Some(ref v) = self.visual {
            let rc = { let mut rg = self.text_model.rope_cache.lock().unwrap(); rg.ensure_length(so+i_ids.len()+1024); rg.cos.clone() };
            let rs = { let rg = self.text_model.rope_cache.lock().unwrap(); rg.sin.clone() };
            let vf = v.forward(pvv, gtv, &rc, &rs, sc, &mut wl, &self.text_model.rope_cache_gpu);
            let hid = self.text_model.config.hidden_size; let mut vidx = 0;
            for (i, &id) in i_ids.iter().enumerate() { if id == 151655 && vidx < (vf.len()/hid) { fe[i*hid..(i+1)*hid].copy_from_slice(&vf[vidx*hid..(vidx+1)*hid]); vidx += 1; } }
        } }
        log_vram_state("IN-FORWARD");
        let nx = self.text_model.forward_ext(i_ids, fe, so, sc, Some(&mut wl), None, pv.is_some());
        if is_b { return vec![]; }
        self.lm_head.forward(&nx, sc)
    }
    pub fn move_to_gpu(&mut self, dev: i32) -> anyhow::Result<()> { self.text_model.move_to_gpu(dev)?; self.lm_head.move_to_gpu(dev)?; if let Some(ref mut v) = self.visual { v.move_to_gpu(dev)?; } Ok(()) }
}

pub struct RopeCache { pub cos: Vec<f16>, pub sin: Vec<f16>, pub head_dim: usize, pub theta: f32, pub tail_tokens: Vec<u32> }
impl RopeCache {
    pub fn new(hd: usize, th: f32, l: usize) -> Self { let (c, s) = crate::models::qwen3vl::native_backend::native_precompute_rope_f16(hd, l, th); Self { cos: c, sin: s, head_dim: hd, theta: th, tail_tokens: Vec::new() } }
    pub fn ensure_length(&mut self, needed: usize) { if needed * (self.head_dim / 2) > self.cos.len() { let (c, s) = crate::models::qwen3vl::native_backend::native_precompute_rope_f16(self.head_dim, (needed + 1023) / 1024 * 1024, self.theta); self.cos = c; self.sin = s; } }
}

impl NativeTensor {
    pub fn from_raw(p: *const u8, sz: usize, sh: Vec<usize>, dt: NativeDType, dev: i32) -> Self { Self { data_ptr: p, host_size: sz, gpu_ptr: None, shape: sh, dtype: dt, _mmap: None, device_id: dev } }
}
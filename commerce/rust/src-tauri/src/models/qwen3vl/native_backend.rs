use std::sync::Arc;
use memmap2::Mmap;
use rayon::prelude::*;
use half::f16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(feature = "cuda")]
pub use cudarc::driver::sys::{lib, CUdeviceptr, CUcontext, CUdevice, CUresult};

pub static GPU_PANIC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn is_gpu_panicked() -> bool {
    GPU_PANIC.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn trigger_gpu_panic() {
    if !is_gpu_panicked() {
        println!("[GPU-PANIC] Critical error detected. Disabling GPU for all future operations.");
        GPU_PANIC.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

// --- [CRITICAL] EXTERN C DECLARATIONS WITH LINK_NAME ---
#[cfg(feature = "cuda")]
extern "C" {
    #[link_name = "bit_serial_matmul_cuda_direct"]
    fn cuda_matmul_f32(d_i: *const f32, d_w: *const u32, d_s: *const f32, d_o: *mut f32, m: i32, n: i32, k: i32, dev: i32);
    
    #[link_name = "bit_serial_matmul_cuda_f16"]
    pub fn cuda_matmul_f16(d_i: *const f16, d_w: *const u32, d_s: *const f16, d_o: *mut f16, m: i32, n: i32, k: i32, dev: i32, src_k: i32);
    
    #[link_name = "bit_serial_attn_cuda_direct"]
    fn cuda_attn_f32(d_q: *const f32, d_k: *const u32, d_v: *const f32, d_o: *mut f32, n_h: i32, n_kv: i32, h_d: i32, t_s: i32, scale: f32, dev: i32, q_len: i32, alpha: f32);
    
    #[link_name = "bit_serial_attn_cuda_f16"]
    fn cuda_attn_f16(d_q: *const f16, d_k: *const u32, d_v: *const f16, d_o: *mut f16, n_h: i32, n_kv: i32, h_d: i32, t_s: i32, scale: f32, dev: i32, q_len: i32, alpha: f32, src_h_d: i32);

    #[link_name = "standard_matmul_cuda_f16"]
    pub fn standard_matmul_cuda_f16(d_i: *const f16, d_w: *const f16, d_o: *mut f16, m: i32, n: i32, k: i32);

    #[link_name = "cuda_rms_norm_f16"]
    pub fn cuda_rms_norm_f16(d_i: *const f16, d_w: *const f16, d_o: *mut f16, m: i32, hid: i32, eps: f32);

    #[link_name = "cuda_silu_mul_f16"]
    pub fn cuda_silu_mul_f16(d_gate: *mut f16, d_up: *const f16, size: i32);

    #[link_name = "cuda_apply_rope_f16"]
    pub fn cuda_apply_rope_f16(d_q: *mut f16, d_k: *mut f16, d_cos: *const f16, d_sin: *const f16, q_len: i32, s_o: i32, n_h: i32, n_kv: i32, h_d: i32);

    #[link_name = "cuda_pack_bits_f16"]
    pub fn cuda_pack_bits_f16(d_src: *const f16, d_dst: *mut u32, elements: i32);

    #[link_name = "cuda_apply_gain_f16"]
    pub fn cuda_apply_gain_f16(d_data: *mut f16, gain: f32, elements: i32);

    #[link_name = "cuda_add_inplace_f16"]
    pub fn cuda_add_inplace_f16(d_dst: *mut f16, d_src: *const f16, size: i32);

    #[link_name = "cuda_hybrid_repeat_f16"]
    pub fn cuda_hybrid_repeat_f16(d_data: *mut f16, src_size: i32, target_size: i32, q_len: i32);

    #[link_name = "cuda_silu_inplace_f16"]
    pub fn cuda_silu_inplace_f16(d_data: *mut f16, size: i32);

    #[link_name = "cuda_element_mul_f16"]
    pub fn cuda_element_mul_f16(d_dst: *mut f16, d_src: *const f16, size: i32);
}

#[cfg(feature = "cuda")]
pub fn native_cuda_element_mul(d_dst: CUdeviceptr, d_src: CUdeviceptr, size: usize) {
    unsafe { cuda_element_mul_f16(d_dst as *mut f16, d_src as *const f16, size as i32); }
}

#[cfg(feature = "cuda")]
pub fn native_cuda_apply_gain(d_data: CUdeviceptr, gain: f32, elements: usize) {
    unsafe { cuda_apply_gain_f16(d_data as *mut f16, gain, elements as i32); }
}

#[cfg(feature = "cuda")]
pub fn native_cuda_add_inplace(d_dst: CUdeviceptr, d_src: CUdeviceptr, size: usize) {
    unsafe { cuda_add_inplace_f16(d_dst as *mut f16, d_src as *const f16, size as i32); }
}

#[cfg(feature = "cuda")]
pub fn native_cuda_hybrid_repeat(d_data: CUdeviceptr, src_size: usize, target_size: usize, q_len: usize) {
    unsafe { cuda_hybrid_repeat_f16(d_data as *mut f16, src_size as i32, target_size as i32, q_len as i32); }
}

#[cfg(feature = "cuda")]
pub fn native_cuda_silu_inplace(d_data: CUdeviceptr, size: usize) {
    unsafe { cuda_silu_inplace_f16(d_data as *mut f16, size as i32); }
}

#[cfg(feature = "cuda")]
pub fn native_cuda_apply_rope(d_q: CUdeviceptr, d_k: CUdeviceptr, d_cos: CUdeviceptr, d_sin: CUdeviceptr, q_l: usize, s_o: usize, n_h: usize, n_kv: usize, h_d: usize) {
    unsafe { cuda_apply_rope_f16(d_q as *mut f16, d_k as *mut f16, d_cos as *const f16, d_sin as *const f16, q_l as i32, s_o as i32, n_h as i32, n_kv as i32, h_d as i32); }
}

#[cfg(feature = "cuda")]
pub fn native_cuda_pack_bits(d_src: CUdeviceptr, d_dst: CUdeviceptr, elements: usize) {
    unsafe { cuda_pack_bits_f16(d_src as *const f16, d_dst as *mut u32, elements as i32); }
}

#[cfg(feature = "cuda")]
pub fn native_bit_serial_attn_gpu_buffered(q: &[f16], k_p: GpuPtr, v_p: GpuPtr, n_h: usize, n_kv: usize, h_d: usize, t_s: usize, dev: usize, d_q: CUdeviceptr, d_o: CUdeviceptr, alpha: f32, src_h_d: usize, q_on_gpu: bool) {
    if d_q == 0 || d_o == 0 || k_p.0.is_null() || v_p.0.is_null() { return; }
    unsafe {
        if !q_on_gpu { let _ = lib().cuMemcpyHtoD_v2(d_q, q.as_ptr() as *const _, q.len() * 2); }
        let actual_q_len = if q.is_empty() { 1 } else { q.len() / (n_h * h_d) };
        cuda_attn_f16(d_q as *const f16, k_p.0 as *const u32, v_p.0 as *const f16, d_o as *mut f16, n_h as i32, n_kv as i32, h_d as i32, t_s as i32, 1.0/(h_d as f32).sqrt(), dev as i32, actual_q_len as i32, alpha, src_h_d as i32);
    }
}

#[cfg(feature = "cuda")]
pub fn native_bit_serial_attn_gpu(q: &[f16], k_p: GpuPtr, v_p: GpuPtr, n_h: usize, n_kv: usize, h_d: usize, t_s: usize, dev: usize, alpha: f32, src_h_d: usize) -> Vec<f16> {
    let eps_signal = f16::from_f32(1e-6);
    unsafe {
        let mut d_q: CUdeviceptr = 0; let mut d_o: CUdeviceptr = 0;
        let _ = lib().cuMemAlloc_v2(&mut d_q, q.len() * 2); 
        let _ = lib().cuMemAlloc_v2(&mut d_o, q.len() * 2);
        native_bit_serial_attn_gpu_buffered(q, k_p, v_p, n_h, n_kv, h_d, t_s, dev, d_q, d_o, alpha, src_h_d, false);
        let mut o_f = vec![eps_signal; q.len()]; 
        let _ = lib().cuMemcpyDtoH_v2(o_f.as_mut_ptr() as *mut _, d_o, q.len() * 2);
        let _ = lib().cuMemFree_v2(d_q); let _ = lib().cuMemFree_v2(d_o);
        o_f
    }
}

#[cfg(feature = "cuda")]
pub fn bit_serial_matmul_gpu_buffered_into(i: &[f16], w: &NativeTensor, s: &NativeTensor, out: &mut [f16], m: usize, n: usize, k: usize, dev: usize, d_i: CUdeviceptr, d_o: CUdeviceptr, src_k: usize) {
    if d_i == 0 || d_o == 0 { return; }
    unsafe {
        let _ = lib().cuMemcpyHtoD_v2(d_i, i.as_ptr() as *const _, m * k * 2);
        let d_w = w.gpu_ptr.expect("W on GPU").0 as *const u32;
        let d_s = s.gpu_ptr.expect("S on GPU").0 as *const f16;
        cuda_matmul_f16(d_i as *const f16, d_w, d_s, d_o as *mut f16, m as i32, n as i32, k as i32, dev as i32, src_k as i32);
        let _ = lib().cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut _, d_o, m * n * 2);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GpuPtr(pub *mut std::ffi::c_void);
unsafe impl Send for GpuPtr {}
unsafe impl Sync for GpuPtr {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NativeDType { F32, F16, BF16, U32 }

#[derive(Clone)]
pub struct NativeTensor {
    pub data_ptr: *const u8,
    pub gpu_ptr: Option<GpuPtr>, 
    pub shape: Vec<usize>,
    pub dtype: NativeDType,
    pub _mmap: Option<Arc<Mmap>>,
    pub device_id: i32,
}

unsafe impl Send for NativeTensor {}
unsafe impl Sync for NativeTensor {}

impl NativeTensor {
    pub fn from_mmap(mmap: Arc<Mmap>, offset: usize, shape: Vec<usize>, dtype: NativeDType) -> Self {
        unsafe { let data_ptr = mmap.as_ptr().add(offset); Self { data_ptr, gpu_ptr: None, shape, dtype, _mmap: Some(mmap), device_id: -1 } }
    }

    pub fn get_slice<T: Clone>(&self) -> std::borrow::Cow<'_, [T]> {
        let size = self.shape.iter().product::<usize>();
        if size == 0 { return std::borrow::Cow::Owned(Vec::new()); }
        let ptr = self.data_ptr as *const T;
        let alignment = std::mem::align_of::<T>();
        if (ptr as usize) % alignment == 0 {
            unsafe { std::borrow::Cow::Borrowed(std::slice::from_raw_parts(ptr, size)) }
        } else {
            unsafe {
                let mut dest = Vec::with_capacity(size);
                std::ptr::copy_nonoverlapping(self.data_ptr, dest.as_mut_ptr() as *mut u8, size * std::mem::size_of::<T>());
                dest.set_len(size);
                std::borrow::Cow::Owned(dest)
            }
        }
    }

    pub unsafe fn get_raw_slice<T>(&self) -> &[T] {
        let size = self.shape.iter().product::<usize>();
        std::slice::from_raw_parts(self.data_ptr as *const T, size)
    }

    #[cfg(feature = "cuda")]
    pub fn move_to_gpu(&mut self, device_id: i32) {
        if self.gpu_ptr.is_some() { return; }
        unsafe {
            let cuda_lib = lib();
            let mut ctx = std::ptr::null_mut() as CUcontext;
            cuda_lib.cuCtxGetCurrent(&mut ctx);
            if ctx == std::ptr::null_mut() {
                let mut dev = 0 as CUdevice;
                cuda_lib.cuDeviceGet(&mut dev, device_id);
                cuda_lib.cuDevicePrimaryCtxRetain(&mut ctx, dev);
                cuda_lib.cuCtxSetCurrent(ctx);
            }
            let size_raw = self.shape.iter().product::<usize>();
            let size_bytes = size_raw * if self.dtype == NativeDType::U32 || self.dtype == NativeDType::F32 { 4 } else { 2 };
            // println!("[GPU-ALLOC] Allocating {} bytes for tensor (Ptr: {:p})", size_bytes, self.data_ptr);
            let mut ptr: CUdeviceptr = 0;
            if (cuda_lib.cuMemAlloc_v2(&mut ptr, size_bytes) as i32) == 0 && ptr != 0 {
                if (cuda_lib.cuMemcpyHtoD_v2(ptr, self.data_ptr as *const _, size_bytes) as i32) == 0 {
                    self.gpu_ptr = Some(GpuPtr(ptr as *mut _));
                    self.device_id = device_id;
                } else { 
                    println!("[GPU-ERROR] Memcpy failed for tensor (Ptr: {:p}, Size: {})", self.data_ptr, size_bytes);
                    let _ = cuda_lib.cuMemFree_v2(ptr); 
                }
            } else {
                println!("[GPU-ERROR] MemAlloc failed for size {}", size_bytes);
            }
        }
    }
}

pub fn pack_f16_to_bits(src: &[f16]) -> Vec<u32> {
    let k_blocks = (src.len() + 31) / 32;
    let mut dst = vec![0u32; k_blocks];
    for kb in 0..k_blocks {
        let mut bits = 0u32;
        for b in 0..32 { if kb * 32 + b < src.len() && src[kb * 32 + b].to_f32() >= 0.0 { bits |= 1 << b; } }
        dst[kb] = bits;
    }
    dst
}

pub fn bit_serial_matmul_f32_shuffled(i: &[f16], w: &[u32], s: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; m * n];
    let k_b = (k + 31) / 32;
    let n_groups = (n + 7) / 8;
    let mut ip = vec![0u32; m * k_b];
    ip.par_chunks_mut(k_b).enumerate().for_each(|(idx, row)| {
        let start = idx * k; let end = (start + k).min(i.len());
        let src = &i[start..end];
        for kb in 0..k_b {
            let mut b = 0u32;
            for bt in 0..32 { if kb * 32 + bt < src.len() && src[kb * 32 + bt].to_f32() >= 0.0 { b |= 1 << bt; } }
            row[kb] = b;
        }
    });
    o.par_chunks_mut(n).enumerate().for_each(|(m_idx, row_o)| {
        let ir_ptr = unsafe { ip.as_ptr().add(m_idx * k_b) };
        for g in 0..n_groups {
            let n_base = g * 8; let w_off = g * k_b * 8;
            if w_off + k_b * 8 > w.len() { continue; }
            let w_ptr = unsafe { w.as_ptr().add(w_off) };
            let s_ptr = unsafe { s.as_ptr().add(w_off) };
            for kb in 0..k_b {
                let input_bits = unsafe { *ir_ptr.add(kb) };
                for i_ch in 0..8 {
                    if n_base + i_ch < n {
                        let weight_bits = unsafe { *w_ptr.add(kb * 8 + i_ch) };
                        let scale = unsafe { *s_ptr.add(kb * 8 + i_ch) };
                        let dot = 32 - 2 * (input_bits ^ weight_bits).count_ones() as i32;
                        row_o[n_base + i_ch] += (dot as f32) * scale.to_f32();
                    }
                }
            }
        }
    });
    o
}

pub fn bit_serial_matmul_f32_extreme_into(i: &[f16], w: &[u32], s: &[f16], out: &mut [f16], m: usize, n: usize, k: usize) {
    let res = bit_serial_matmul_f32_shuffled(i, w, s, m, n, k);
    for (j, &val) in res.iter().enumerate() { if j < out.len() { out[j] = f16::from_f32(val); } }
}

pub fn native_rms_norm_f16_into(x: &[f16], w: &[f16], eps: f32, hid: usize, out: &mut [f16]) {
    out.par_chunks_mut(hid).enumerate().for_each(|(i, out_row)| {
        let x_start = i * hid;
        if x_start < x.len() {
            let x_end = (x_start + hid).min(x.len());
            let x_row = &x[x_start..x_end];
            let mut v = 0.0f32;
            for &val in x_row { let f = val.to_f32(); v += f * f; }
            let inv = 1.0 / (v / hid as f32 + eps).sqrt();
            for (j, out_val) in out_row.iter_mut().enumerate() {
                let x_idx = j % x_row.len(); let w_idx = j % w.len();
                *out_val = f16::from_f32(x_row[x_idx].to_f32() * inv * w[w_idx].to_f32());
            }
        }
    });
}

pub fn native_linear_f16_into(x: &[f16], w: &[f16], b: Option<&[f16]>, out: &mut [f16], _m: usize, out_f: usize, in_f: usize) {
    out.par_chunks_mut(out_f).enumerate().for_each(|(i, out_row)| {
        let x_start = i * in_f;
        if x_start < x.len() {
            let x_end = (x_start + in_f).min(x.len());
            let x_row = &x[x_start..x_end];
            for j in 0..out_f {
                let mut acc = 0.0f32;
                let w_row_start = j * in_f;
                if w_row_start < w.len() {
                    let w_row_end = (w_row_start + in_f).min(w.len());
                    let w_row = &w[w_row_start..w_row_end];
                    for k in 0..in_f {
                        let x_val = x_row.get(k % x_row.len()).map(|v| v.to_f32()).unwrap_or(0.0);
                        let w_val = w_row.get(k % w_row.len()).map(|v| v.to_f32()).unwrap_or(0.0);
                        acc += x_val * w_val;
                    }
                }
                if let Some(bias) = b { acc += bias.get(j % bias.len()).map(|v| v.to_f32()).unwrap_or(0.0); }
                if let Some(out_ptr) = out_row.get_mut(j) { *out_ptr = f16::from_f32(acc); }
            }
        }
    });
}

pub fn native_bit_serial_attn_f16(q: &[f16], k_p: &[u32], v_f: &[f16], hid: usize, n_h: usize, n_kv: usize, q_l: usize, t_s: usize, alpha: f32) -> Vec<f16> {
    let h_d = hid / n_h; let k_b = (h_d + 31) / 32; let h_kv_ratio = n_h / n_kv;
    let mut q_bits = vec![0u32; q_l * n_h * k_b];
    q_bits.par_chunks_mut(k_b).enumerate().for_each(|(idx, row)| {
        let h_idx = idx % n_h; let q_idx = idx / n_h;
        let start = (q_idx * n_h * h_d) + (h_idx * h_d);
        if start + h_d <= q.len() {
            let src = &q[start..start + h_d];
            for kb in 0..k_b {
                let mut b = 0u32;
                for bt in 0..32 { if kb * 32 + bt < h_d && src[kb * 32 + bt].to_f32() >= 0.0 { b |= 1 << bt; } }
                row[kb] = b;
            }
        }
    });
    let eps_signal = f16::from_f32(1e-6);
    let mut o = vec![eps_signal; q_l * hid];
    o.par_chunks_mut(hid).enumerate().for_each(|(q_idx, row_o)| {
        for h in 0..n_h {
            let h_kv = h / h_kv_ratio; let q_b_ptr = unsafe { q_bits.as_ptr().add((q_idx * n_h + h) * k_b) };
            let mut running_max = -1e20f32; let mut running_sum = 0.0f32;
            let mut local_o = vec![0.0f32; h_d];
            for j in 0..t_s {
                let mut dot = 0; let k_p_idx = (j * n_kv + h_kv) * k_b;
                if k_p_idx + k_b <= k_p.len() {
                    for kb in 0..k_b { dot += 32 - 2 * (unsafe { *q_b_ptr.add(kb) ^ k_p[k_p_idx + kb] }).count_ones() as i32; }
                }
                let score = ((dot as f32) + alpha) * (1.0 / (h_d as f32).sqrt());
                let n_max = running_max.max(score); let e_scale = (running_max - n_max).exp(); let e_score = (score - n_max).exp();
                running_sum = running_sum * e_scale + e_score; running_max = n_max;
                let v_ptr_idx = (j * n_kv + h_kv) * h_d;
                if v_ptr_idx + h_d <= v_f.len() {
                    for d in 0..h_d { local_o[d] = local_o[d] * e_scale + e_score * v_f[v_ptr_idx + d].to_f32(); }
                }
            }
            for d in 0..h_d { if h * h_d + d < row_o.len() { row_o[h * h_d + d] = f16::from_f32(local_o[d] / (running_sum + 1e-9)); } }
        }
    });
    o
}

pub fn native_rms_norm_f16(x: &[f16], w: &[f16], eps: f32, hid: usize) -> Vec<f16> {
    let mut out = vec![f16::from_f32(1e-6); x.len()];
    native_rms_norm_f16_into(x, w, eps, hid, &mut out);
    out
}

pub fn native_silu_f16(x: &mut [f16]) {
    x.par_iter_mut().for_each(|val| { let f = val.to_f32(); *val = f16::from_f32(f * (1.0 / (1.0 + (-f).exp()))); });
}

pub fn native_embedding_lookup_f16(ids: &[u32], table: &[f16], hid: usize) -> Vec<f16> {
    let eps_signal = f16::from_f32(1e-6);
    let mut out = vec![eps_signal; ids.len() * hid];
    for (i, &id) in ids.iter().enumerate() {
        let start = id as usize * hid;
        if start + hid <= table.len() { out[i * hid..(i + 1) * hid].copy_from_slice(&table[start..start + hid]); }
    }
    out
}

pub fn native_precompute_rope_f16(h_d: usize, max_len: usize, theta: f32) -> (Vec<f16>, Vec<f16>) {
    let mut cos = Vec::with_capacity(max_len * (h_d / 2));
    let mut sin = Vec::with_capacity(max_len * (h_d / 2));
    for p in 0..max_len {
        for d in 0..(h_d / 2) {
            let freq = 1.0 / theta.powf((2.0 * d as f32) / (h_d as f32));
            let (sn, cs) = ((p as f32) * freq).sin_cos();
            cos.push(f16::from_f32(cs)); sin.push(f16::from_f32(sn));
        }
    }
    (cos, sin)
}

pub fn native_apply_rope_f16_with_offset(q: &mut [f16], k: &mut [f16], q_len: usize, s_o: usize, n_h: usize, h_d: usize, _theta: f32, cos: &[f16], sin: &[f16]) {
    let n_kv = k.len() / (q_len * h_d);
    for t in 0..q_len {
        let p = t + s_o;
        for h in 0..n_h {
            let q_ptr = &mut q[t * n_h * h_d + h * h_d .. t * n_h * h_d + (h + 1) * h_d];
            for d in 0..(h_d / 2) {
                let (c, s) = (cos[p * (h_d / 2) + d].to_f32(), sin[p * (h_d / 2) + d].to_f32());
                let (v0, v1) = (q_ptr[d].to_f32(), q_ptr[d + h_d / 2].to_f32());
                q_ptr[d] = f16::from_f32(v0 * c - v1 * s); q_ptr[d + h_d / 2] = f16::from_f32(v0 * s + v1 * c);
            }
        }
        for h in 0..n_kv {
            let k_ptr = &mut k[t * n_kv * h_d + h * h_d .. t * n_kv * h_d + (h + 1) * h_d];
            for d in 0..(h_d / 2) {
                let (c, s) = (cos[p * (h_d / 2) + d].to_f32(), sin[p * (h_d / 2) + d].to_f32());
                let (v0, v1) = (k_ptr[d].to_f32(), k_ptr[d + h_d / 2].to_f32());
                k_ptr[d] = f16::from_f32(v0 * c - v1 * s); k_ptr[d + h_d / 2] = f16::from_f32(v0 * s + v1 * c);
            }
        }
    }
}

pub struct RopeCache { pub cos: Vec<f16>, pub sin: Vec<f16>, pub head_dim: usize, pub theta: f32, pub tail_tokens: Vec<u32> }
impl RopeCache {
    pub fn new(h_d: usize, theta: f32, initial_len: usize) -> Self {
        let (cos, sin) = native_precompute_rope_f16(h_d, initial_len, theta);
        Self { cos, sin, head_dim: h_d, theta, tail_tokens: Vec::new() }
    }
    pub fn ensure_length(&mut self, needed: usize) {
        let current = self.cos.len() / (self.head_dim / 2);
        if needed > current {
            let (c, s) = native_precompute_rope_f16(self.head_dim, needed, self.theta);
            self.cos = c; self.sin = s;
        }
    }
}
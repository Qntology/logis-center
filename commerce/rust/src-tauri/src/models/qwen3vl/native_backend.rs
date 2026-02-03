use std::sync::Arc;
use memmap2::Mmap;
use rayon::prelude::*;
use half::f16;

#[cfg(feature = "cuda")]
extern "C" {
    fn bit_serial_matmul_cuda_direct(d_i: *const f32, d_w: *const u32, d_s: *const f32, d_o: *mut f32, m: i32, n: i32, k: i32, dev: i32);
    fn bit_serial_attn_cuda_direct(d_q: *const f32, d_k: *const u32, d_v: *const f32, d_o: *mut f32, n_h: i32, h_d: i32, t_s: i32, scale: f32, dev: i32);
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NativeDType { F32, F16, BF16, U32 }

#[derive(Clone)]
pub struct NativeTensor {
    pub data_ptr: *const u8,
    pub gpu_ptr: Option<*mut std::ffi::c_void>, 
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
    pub fn get_slice<T>(&self) -> &[T] {
        let size = self.shape.iter().product::<usize>();
        unsafe { std::slice::from_raw_parts(self.data_ptr as *const T, size) }
    }
    #[cfg(feature = "cuda")]
    pub fn move_to_gpu(&mut self, device_id: i32) {
        if self.gpu_ptr.is_some() { return; }
        let size = self.shape.iter().product::<usize>() * if self.dtype == NativeDType::U32 || self.dtype == NativeDType::F32 { 4 } else { 2 };
        unsafe {
            use cudarc::driver::sys::*;
            let mut ptr: CUdeviceptr = 0;
            cuMemAlloc_v2(&mut ptr, size);
            cuMemcpyHtoD_v2(ptr, self.data_ptr as *const _, size);
            self.gpu_ptr = Some(ptr as *mut _); self.device_id = device_id;
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

#[cfg(feature = "cuda")]
pub fn native_bit_serial_attn_gpu(
    q: &[f16], k_packed_ptr: *mut std::ffi::c_void, v_f16_ptr: *mut std::ffi::c_void,
    n_h: usize, h_d: usize, t_s: usize, dev: usize
) -> Vec<f16> {
    unsafe {
        use cudarc::driver::sys::*;
        let mut d_q: CUdeviceptr = 0; let mut d_o: CUdeviceptr = 0;
        let q_f32: Vec<f32> = q.iter().map(|v| v.to_f32()).collect();
        cuMemAlloc_v2(&mut d_q, q.len() * 4);
        cuMemcpyHtoD_v2(d_q, q_f32.as_ptr() as *const _, q.len() * 4);
        cuMemAlloc_v2(&mut d_o, q.len() * 4);

        bit_serial_attn_cuda_direct(d_q as *const f32, k_packed_ptr as *const u32, v_f16_ptr as *const f32, d_o as *mut f32, n_h as i32, h_d as i32, t_s as i32, 1.0/(h_d as f32).sqrt(), dev as i32);

        let mut o_f32 = vec![0.0f32; q.len()];
        cuMemcpyDtoH_v2(o_f32.as_mut_ptr() as *mut _, d_o, q.len() * 4);
        cuMemFree_v2(d_q); cuMemFree_v2(d_o);
        o_f32.into_iter().map(f16::from_f32).collect()
    }
}

pub fn native_bit_serial_attn_f16(q: &[f16], k_p: &[u32], v_f: &[f16], hid: usize, n_h: usize, q_l: usize, t_s: usize) -> Vec<f16> {
    let h_d = hid / n_h; let k_b = (h_d + 31) / 32; let mut o = vec![f16::ZERO; q_l * hid]; let sc = 1.0 / (h_d as f32).sqrt();
    o.par_chunks_exact_mut(hid).enumerate().for_each(|(i, out)| {
        for h in 0..n_h {
            let mut s = vec![0.0f32; t_s]; let qi = &q[(i * n_h + h) * h_d .. (i * n_h + h + 1) * h_d];
            let mut qp = [0u32; 8]; for kb in 0..k_b { let mut bts = 0u32; for b in 0..32 { if qi[kb * 32 + b].to_f32() >= 0.0 { bts |= 1 << b; } } qp[kb] = bts; }
            for j in 0..t_s {
                let kj = &k_p[(j * n_h + h) * k_b .. (j * n_h + h + 1) * k_b];
                let mut dot = 0i32; for kb in 0..k_b { dot += 32 - 2 * (qp[kb] ^ kj[kb]).count_ones() as i32; }
                s[j] = (dot as f32) * sc;
            }
            let max_s = s.iter().fold(f32::MIN, |a, &b| a.max(b));
            let mut sum_e = 0.0f32; for x in s.iter_mut() { *x = (*x - max_s).exp(); sum_e += *x; }
            for j in 0..t_s {
                let vj = &v_f[(j * n_h + h) * h_d .. (j * n_h + h + 1) * h_d]; let sc_final = s[j] / sum_e;
                for d in 0..h_d { out[h * h_d + d] = f16::from_f32(out[h * h_d + d].to_f32() + sc_final * vj[d].to_f32()); }
            }
        }
    });
    o
}

pub fn bit_serial_matmul_f32_extreme(i: &[f16], w: &[u32], s: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; m * n]; let k_b = k / 32; let mut ip = vec![0u32; m * k_b];
    ip.par_chunks_mut(k_b).enumerate().for_each(|(idx, row)| {
        let r = &i[idx * k .. (idx + 1) * k];
        for kb in 0..k_b { let mut b = 0u32; for bt in 0..32 { if r[kb * 32 + bt].to_f32() >= 0.0 { b |= 1 << bt; } } row[kb] = b; }
    });
    o.par_chunks_mut(n).enumerate().for_each(|(idx, row)| {
        let ir = &ip[idx * k_b .. (idx + 1) * k_b];
        for j in 0..n {
            let mut d = 0i32; let w_row = &w[j * k_b .. (j + 1) * k_b];
            for kb in 0..k_b { d += (ir[kb] ^ w_row[kb]).count_ones() as i32; }
            row[j] = ((k_b as i32 * 32 - 2 * d) as f32) * s[j].to_f32();
        }
    });
    o
}

#[cfg(feature = "cuda")]
pub fn bit_serial_matmul_gpu(i: &[f16], w: &NativeTensor, s: &NativeTensor, m: usize, n: usize, k: usize, dev: usize) -> Vec<f16> {
    unsafe {
        use cudarc::driver::sys::*;
        let mut d_i: CUdeviceptr = 0; let mut d_o: CUdeviceptr = 0; let mut d_s: CUdeviceptr = 0;
        let i_f32: Vec<f32> = i.iter().map(|v| v.to_f32()).collect();
        cuMemAlloc_v2(&mut d_i, m * k * 4); cuMemcpyHtoD_v2(d_i, i_f32.as_ptr() as *const _, m * k * 4);
        cuMemAlloc_v2(&mut d_o, m * n * 4);
        let s_f32: Vec<f32> = s.get_slice::<f16>().iter().map(|v| v.to_f32()).collect();
        cuMemAlloc_v2(&mut d_s, n * 4); cuMemcpyHtoD_v2(d_s, s_f32.as_ptr() as *const _, n * 4);
        bit_serial_matmul_cuda_direct(d_i as *const f32, w.gpu_ptr.unwrap() as *const u32, d_s as *const f32, d_o as *mut f32, m as i32, n as i32, k as i32, dev as i32);
        let mut o_f = vec![0.0f32; m * n]; cuMemcpyDtoH_v2(o_f.as_mut_ptr() as *mut _, d_o, m * n * 4);
        cuMemFree_v2(d_i); cuMemFree_v2(d_o); cuMemFree_v2(d_s);
        o_f.into_iter().map(f16::from_f32).collect()
    }
}

pub fn native_silu_f16(i: &mut [f16]) { i.par_iter_mut().for_each(|x| { let v = x.to_f32(); *x = f16::from_f32(v / (1.0 + (-v).exp())); }); }
pub fn native_rms_norm_f16(i: &[f16], w: &[f16], e: f32, hid: usize) -> Vec<f16> {
    let mut o = vec![f16::ZERO; i.len()];
    i.par_chunks_exact(hid).zip(o.par_chunks_exact_mut(hid)).for_each(|(r_i, r_o)| {
        let mut v = 0.0f32; for &x in r_i { let val = x.to_f32(); v += val * val; }
        let inv = 1.0 / (v / hid as f32 + e).sqrt();
        for j in 0..hid { if j < w.len() { r_o[j] = f16::from_f32(r_i[j].to_f32() * inv * w[j].to_f32()); } }
    });
    o
}
pub fn native_embedding_lookup_f16(ids: &[u32], t: &[f16], hid: usize) -> Vec<f16> {
    let mut o = vec![f16::ZERO; ids.len() * hid];
    o.par_chunks_exact_mut(hid).enumerate().for_each(|(i, r)| { let s = ids[i] as usize * hid; r.copy_from_slice(&t[s..s+hid]); });
    o
}
pub fn native_apply_rope_f16_with_offset(q: &mut [f16], k: &mut [f16], _ql: usize, off: usize, _n_h: usize, h_d: usize, th: f32) {
    let h_d_2 = h_d / 2;
    let mut apply = |data: &mut [f16]| {
        data.par_chunks_exact_mut(h_d).enumerate().for_each(|(i, h)| {
            let p = (off + i) as f32;
            for d in 0..h_d_2 {
                let (sn, cs) = (p / th.powf(2.0 * d as f32 / h_d as f32)).sin_cos();
                let (v0, v1) = (h[d].to_f32(), h[d + h_d_2].to_f32());
                h[d] = f16::from_f32(v0 * cs - v1 * sn); h[d+h_d_2] = f16::from_f32(v0 * sn + v1 * cs);
            }
        });
    };
    apply(q); apply(k);
}
pub fn native_linear_f16(i: &[f16], w: &[f16], b: Option<&[f16]>, m: usize, n: usize, k: usize) -> Vec<f16> {
    let mut o = vec![f16::ZERO; m * n];
    o.par_chunks_exact_mut(n).enumerate().for_each(|(idx, ro)| {
        let ri = &i[idx * k .. (idx + 1) * k];
        for j in 0..n {
            let mut s = 0.0f32; let wr = &w[j * k .. (j + 1) * k];
            for l in 0..k { s += ri[l].to_f32() * wr[l].to_f32(); }
            if let Some(bv) = b { s += bv[j].to_f32(); }
            ro[j] = f16::from_f32(s);
        }
    });
    o
}

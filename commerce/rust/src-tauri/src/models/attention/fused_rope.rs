use candle_core::{DType, Result, Tensor};
#[cfg(feature = "cuda")]
use kernels::ffi;

#[derive(Clone, Copy)]
enum RopeLayout {
    BatchMajor { q_bh: u32, k_bh: u32, seq_len: u32, d: u32 },
    TokenMajor { num_tokens: u32, q_heads: u32, k_heads: u32, d: u32 },
}

impl RopeLayout {
    fn positions_len(self) -> usize {
        match self {
            Self::BatchMajor { seq_len, .. } => seq_len as usize,
            Self::TokenMajor { num_tokens, .. } => num_tokens as usize,
        }
    }
    fn q_bh(self) -> u32 { match self { Self::BatchMajor { q_bh, .. } => q_bh, Self::TokenMajor { q_heads, .. } => q_heads } }
    fn k_bh(self) -> u32 { match self { Self::BatchMajor { k_bh, .. } => k_bh, Self::TokenMajor { k_heads, .. } => k_heads } }
    fn seq_len(self) -> u32 { match self { Self::BatchMajor { seq_len, .. } => seq_len, Self::TokenMajor { num_tokens, .. } => num_tokens } }
    fn d(self) -> u32 { match self { Self::BatchMajor { d, .. } => d, Self::TokenMajor { d, .. } => d } }
}

fn resolve_rope_layout(q: &Tensor, k: &Tensor) -> Result<RopeLayout> {
    match (q.dims().len(), k.dims().len()) {
        (4, 4) => {
            let (b, q_h, seq_len, d) = q.dims4()?;
            let (kb, k_h, k_seq_len, kd) = k.dims4()?;
            if b != kb || seq_len != k_seq_len || d != kd { candle_core::bail!("Shape mismatch"); }
            Ok(RopeLayout::BatchMajor { q_bh: (b * q_h) as u32, k_bh: (b * k_h) as u32, seq_len: seq_len as u32, d: d as u32 })
        }
        (3, 3) => {
            let (num_tokens, q_heads, d) = q.dims3()?;
            let (k_num_tokens, k_heads, kd) = k.dims3()?;
            if num_tokens != k_num_tokens || d != kd { candle_core::bail!("Shape mismatch"); }
            Ok(RopeLayout::TokenMajor { num_tokens: num_tokens as u32, q_heads: q_heads as u32, k_heads: k_heads as u32, d: d as u32 })
        }
        _ => candle_core::bail!("FusedRope expects 3D or 4D inputs")
    }
}

#[cfg(feature = "cuda")]
fn launch_fused_rope(
    q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor, positions: &Tensor, is_interleaved: bool,
) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;
    use crate::models::attention::cuda_utils::get_raw_stream;

    let layout = resolve_rope_layout(q, k)?;
    let positions = if positions.dtype() != DType::I64 { positions.to_dtype(DType::I64)? } else { positions.clone() };
    if !q.is_contiguous() || !k.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() || !positions.is_contiguous() {
        candle_core::bail!("Tensors must be contiguous");
    }

    let dev = q.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let stream_ptr = get_raw_stream(dev);

    let (q_s, _) = q.storage_and_layout();
    let (k_s, _) = k.storage_and_layout();
    let (cos_s, _) = cos.storage_and_layout();
    let (sin_s, _) = sin.storage_and_layout();
    let (pos_s, _) = positions.storage_and_layout();

    let q_c = match &*q_s { candle_core::Storage::Cuda(s) => s, _ => candle_core::bail!("CUDA only") };
    let k_c = match &*k_s { candle_core::Storage::Cuda(s) => s, _ => candle_core::bail!("CUDA only") };
    let cos_c = match &*cos_s { candle_core::Storage::Cuda(s) => s, _ => candle_core::bail!("CUDA only") };
    let sin_c = match &*sin_s { candle_core::Storage::Cuda(s) => s, _ => candle_core::bail!("CUDA only") };
    let pos_c = match &*pos_s { candle_core::Storage::Cuda(s) => s, _ => candle_core::bail!("CUDA only") };

    let pos_ptr = match &pos_c.slice { CudaStorageSlice::I64(s) => s.device_ptr(&stream).0 as *const i64, _ => candle_core::bail!("I64 only") };

    match q.dtype() {
        DType::F32 => {
            let q_ptr = match &q_c.slice { CudaStorageSlice::F32(s) => s.device_ptr(&stream).0 as *mut f32, _ => unreachable!() };
            let k_ptr = match &k_c.slice { CudaStorageSlice::F32(s) => s.device_ptr(&stream).0 as *mut f32, _ => unreachable!() };
            let cos_ptr = match &cos_c.slice { CudaStorageSlice::F32(s) => s.device_ptr(&stream).0 as *const f32, _ => unreachable!() };
            let sin_ptr = match &sin_c.slice { CudaStorageSlice::F32(s) => s.device_ptr(&stream).0 as *const f32, _ => unreachable!() };
            unsafe {
                match layout {
                    RopeLayout::BatchMajor { q_bh, k_bh, seq_len, d } => {
                        if is_interleaved { ffi::fused_rope_i_f32(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, q_bh, k_bh, seq_len, d, stream_ptr); }
                        else { ffi::fused_rope_f32(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, q_bh, k_bh, seq_len, d, stream_ptr); }
                    }
                    RopeLayout::TokenMajor { num_tokens, q_heads, k_heads, d } => {
                        if is_interleaved { ffi::fused_rope_i_tok_major_f32(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens, q_heads, k_heads, d, stream_ptr); }
                        else { ffi::fused_rope_tok_major_f32(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens, q_heads, k_heads, d, stream_ptr); }
                    }
                }
            }
        }
        DType::F16 => {
            let q_ptr = match &q_c.slice { CudaStorageSlice::F16(s) => s.device_ptr(&stream).0 as *mut core::ffi::c_void, _ => unreachable!() };
            let k_ptr = match &k_c.slice { CudaStorageSlice::F16(s) => s.device_ptr(&stream).0 as *mut core::ffi::c_void, _ => unreachable!() };
            let cos_ptr = match &cos_c.slice { CudaStorageSlice::F16(s) => s.device_ptr(&stream).0 as *const core::ffi::c_void, _ => unreachable!() };
            let sin_ptr = match &sin_c.slice { CudaStorageSlice::F16(s) => s.device_ptr(&stream).0 as *const core::ffi::c_void, _ => unreachable!() };
            unsafe {
                match layout {
                    RopeLayout::BatchMajor { q_bh, k_bh, seq_len, d } => {
                        if is_interleaved { ffi::fused_rope_i_f16(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, q_bh, k_bh, seq_len, d, stream_ptr); }
                        else { ffi::fused_rope_f16(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, q_bh, k_bh, seq_len, d, stream_ptr); }
                    }
                    RopeLayout::TokenMajor { num_tokens, q_heads, k_heads, d } => {
                        if is_interleaved { ffi::fused_rope_i_tok_major_f16(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens, q_heads, k_heads, d, stream_ptr); }
                        else { ffi::fused_rope_tok_major_f16(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens, q_heads, k_heads, d, stream_ptr); }
                    }
                }
            }
        }
        DType::BF16 => {
            let q_ptr = match &q_c.slice { CudaStorageSlice::BF16(s) => s.device_ptr(&stream).0 as *mut core::ffi::c_void, _ => unreachable!() };
            let k_ptr = match &k_c.slice { CudaStorageSlice::BF16(s) => s.device_ptr(&stream).0 as *mut core::ffi::c_void, _ => unreachable!() };
            let cos_ptr = match &cos_c.slice { CudaStorageSlice::BF16(s) => s.device_ptr(&stream).0 as *const core::ffi::c_void, _ => unreachable!() };
            let sin_ptr = match &sin_c.slice { CudaStorageSlice::BF16(s) => s.device_ptr(&stream).0 as *const core::ffi::c_void, _ => unreachable!() };
            unsafe {
                match layout {
                    RopeLayout::BatchMajor { q_bh, k_bh, seq_len, d } => {
                        if is_interleaved { ffi::fused_rope_i_bf16(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, q_bh, k_bh, seq_len, d, stream_ptr); }
                        else { ffi::fused_rope_bf16(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, q_bh, k_bh, seq_len, d, stream_ptr); }
                    }
                    RopeLayout::TokenMajor { num_tokens, q_heads, k_heads, d } => {
                        if is_interleaved { ffi::fused_rope_i_tok_major_bf16(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens, q_heads, k_heads, d, stream_ptr); }
                        else { ffi::fused_rope_tok_major_bf16(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens, q_heads, k_heads, d, stream_ptr); }
                    }
                }
            }
        }
        _ => candle_core::bail!("Unsupported dtype")
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn launch_fused_rope_partial_token_major(
    q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor, positions: &Tensor, is_interleaved: bool, rotary_dim: usize,
) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::CudaStorageSlice;
    use crate::models::attention::cuda_utils::get_raw_stream;

    let (num_tokens, q_heads, full_d) = q.dims3()?;
    let k_heads = k.dim(1)? as u32;
    let dev = q.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let stream_ptr = get_raw_stream(dev);

    let (q_s, _) = q.storage_and_layout();
    let (k_s, _) = k.storage_and_layout();
    let (cos_s, _) = cos.storage_and_layout();
    let (sin_s, _) = sin.storage_and_layout();
    let (pos_s, _) = positions.storage_and_layout();

    let q_c = match &*q_s { candle_core::Storage::Cuda(s) => s, _ => candle_core::bail!("CUDA only") };
    let k_c = match &*k_s { candle_core::Storage::Cuda(s) => s, _ => candle_core::bail!("CUDA only") };
    let cos_c = match &*cos_s { candle_core::Storage::Cuda(s) => s, _ => candle_core::bail!("CUDA only") };
    let sin_c = match &*sin_s { candle_core::Storage::Cuda(s) => s, _ => candle_core::bail!("CUDA only") };
    let pos_c = match &*pos_s { candle_core::Storage::Cuda(s) => s, _ => candle_core::bail!("CUDA only") };

    let pos_ptr = match &pos_c.slice { CudaStorageSlice::I64(s) => s.device_ptr(&stream).0 as *const i64, _ => unreachable!() };

    match q.dtype() {
        DType::F32 => {
            let q_ptr = match &q_c.slice { CudaStorageSlice::F32(s) => s.device_ptr(&stream).0 as *mut f32, _ => unreachable!() };
            let k_ptr = match &k_c.slice { CudaStorageSlice::F32(s) => s.device_ptr(&stream).0 as *mut f32, _ => unreachable!() };
            let cos_ptr = match &cos_c.slice { CudaStorageSlice::F32(s) => s.device_ptr(&stream).0 as *const f32, _ => unreachable!() };
            let sin_ptr = match &sin_c.slice { CudaStorageSlice::F32(s) => s.device_ptr(&stream).0 as *const f32, _ => unreachable!() };
            unsafe {
                if is_interleaved { ffi::fused_rope_i_partial_tok_major_f32(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens as u32, q_heads as u32, k_heads, rotary_dim as u32, full_d as u32, stream_ptr); }
                else { ffi::fused_rope_partial_tok_major_f32(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens as u32, q_heads as u32, k_heads, rotary_dim as u32, full_d as u32, stream_ptr); }
            }
        }
        DType::F16 => {
            let q_ptr = match &q_c.slice { CudaStorageSlice::F16(s) => s.device_ptr(&stream).0 as *mut core::ffi::c_void, _ => unreachable!() };
            let k_ptr = match &k_c.slice { CudaStorageSlice::F16(s) => s.device_ptr(&stream).0 as *mut core::ffi::c_void, _ => unreachable!() };
            let cos_ptr = match &cos_c.slice { CudaStorageSlice::F16(s) => s.device_ptr(&stream).0 as *const core::ffi::c_void, _ => unreachable!() };
            let sin_ptr = match &sin_c.slice { CudaStorageSlice::F16(s) => s.device_ptr(&stream).0 as *const core::ffi::c_void, _ => unreachable!() };
            unsafe {
                if is_interleaved { ffi::fused_rope_i_partial_tok_major_f16(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens as u32, q_heads as u32, k_heads, rotary_dim as u32, full_d as u32, stream_ptr); }
                else { ffi::fused_rope_partial_tok_major_f16(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens as u32, q_heads as u32, k_heads, rotary_dim as u32, full_d as u32, stream_ptr); }
            }
        }
        DType::BF16 => {
            let q_ptr = match &q_c.slice { CudaStorageSlice::BF16(s) => s.device_ptr(&stream).0 as *mut core::ffi::c_void, _ => unreachable!() };
            let k_ptr = match &k_c.slice { CudaStorageSlice::BF16(s) => s.device_ptr(&stream).0 as *mut core::ffi::c_void, _ => unreachable!() };
            let cos_ptr = match &cos_c.slice { CudaStorageSlice::BF16(s) => s.device_ptr(&stream).0 as *const core::ffi::c_void, _ => unreachable!() };
            let sin_ptr = match &sin_c.slice { CudaStorageSlice::BF16(s) => s.device_ptr(&stream).0 as *const core::ffi::c_void, _ => unreachable!() };
            unsafe {
                if is_interleaved { ffi::fused_rope_i_partial_tok_major_bf16(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens as u32, q_heads as u32, k_heads, rotary_dim as u32, full_d as u32, stream_ptr); }
                else { ffi::fused_rope_partial_tok_major_bf16(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, num_tokens as u32, q_heads as u32, k_heads, rotary_dim as u32, full_d as u32, stream_ptr); }
            }
        }
        _ => candle_core::bail!("Unsupported dtype")
    }
    Ok(())
}

pub struct FusedRope;
impl FusedRope {
    #[cfg(feature = "cuda")]
    pub fn apply(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor, positions: &Tensor, is_interleaved: bool) -> Result<(Tensor, Tensor)> {
        launch_fused_rope(q, k, cos, sin, positions, is_interleaved)?;
        Ok((q.to_owned(), k.to_owned()))
    }
    #[cfg(feature = "cuda")]
    pub fn apply_inplace(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor, positions: &Tensor, is_interleaved: bool) -> Result<()> {
        launch_fused_rope(q, k, cos, sin, positions, is_interleaved)
    }
    #[cfg(feature = "cuda")]
    pub fn apply_inplace_partial(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor, positions: &Tensor, is_interleaved: bool, rotary_dim: usize) -> Result<()> {
        launch_fused_rope_partial_token_major(q, k, cos, sin, positions, is_interleaved, rotary_dim)
    }
}

use candle_core::{DType, Result, Tensor};
#[cfg(feature = "cuda")]
use crate::kernels::ffi;

#[derive(Debug, Clone)]
struct CausalMask {
    sliding_window: i32,
}

impl candle_core::InplaceOp1 for CausalMask {
    fn name(&self) -> &'static str { "causal_mask" }

    fn cpu_fwd(&self, _: &mut candle_core::CpuStorage, _: &candle_core::Layout) -> Result<()> {
        candle_core::bail!("causal_mask is CUDA only in this impl")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, input: &mut candle_core::CudaStorage, input_layout: &candle_core::Layout) -> Result<()> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::CudaStorageSlice;
        use crate::cuda_utils::get_raw_stream;
        use std::ffi::c_void;

        let dev = input.device();
        let (tgt_len, tgt_len1) = input_layout.shape().dims2()?;
        assert!(tgt_len == tgt_len1);

        let stream_ptr = get_raw_stream(dev);

        let src_ptr = match &input.slice {
            CudaStorageSlice::F32(inp) => *inp.device_ptr() as *mut c_void,
            CudaStorageSlice::F16(inp) => *inp.device_ptr() as *mut c_void,
            CudaStorageSlice::BF16(inp) => *inp.device_ptr() as *mut c_void,
            _ => candle_core::bail!("Unsupported dtype for causal mask"),
        };


        unsafe {
            match input.dtype() {
                DType::F32 => ffi::causal_mask_f32(src_ptr, tgt_len as i32, self.sliding_window, stream_ptr),
                DType::F16 => ffi::causal_mask_f16(src_ptr, tgt_len as i32, self.sliding_window, stream_ptr),
                DType::BF16 => ffi::causal_mask_bf16(src_ptr, tgt_len as i32, self.sliding_window, stream_ptr),
                _ => unreachable!(),
            }
        }
        Ok(())
    }
}

pub fn causal_mask(mask: &Tensor, sliding_window: Option<usize>) -> Result<()> {
    let op = CausalMask { sliding_window: sliding_window.unwrap_or(0) as i32 };
    mask.inplace_op1(&op)
}

#[cfg(feature = "cuda")]
pub fn update_mask_cuda(mask: &Tensor, _: Option<&Tensor>) -> Result<()> {
    causal_mask(mask, None)
}

#[cfg(not(feature = "cuda"))]
pub fn update_mask_cuda(_: &Tensor, _: Option<&Tensor>) -> Result<()> {
    Ok(())
}

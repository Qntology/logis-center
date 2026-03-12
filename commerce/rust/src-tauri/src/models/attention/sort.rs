use candle::{Result, Tensor};
use candle_core as candle;
#[cfg(feature = "cuda")]
use kernels::ffi;

#[derive(Debug, Clone)]
struct ArgSort {
    asc: bool,
    last_dim: usize,
    inplace: bool,
}

impl candle::CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "argsort"
    }

    fn cpu_fwd(
        &self,
        _: &candle::CpuStorage,
        _: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle_core::bail!("ArgSort is CUDA only")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::CudaStorageSlice;
        use crate::models::attention::cuda_utils::{WrapErr, get_raw_stream};
        
        let dev = storage.device();
        let elem_count = layout.shape().elem_count();
        let ncols = self.last_dim as i32;
        let nrows = elem_count as i32 / ncols;
        
        let mut dst = unsafe { dev.alloc::<u32>(elem_count) }.w()?;
        let stream_ptr = get_raw_stream(dev);

        use std::ffi::c_void;

        let src_ptr = match &storage.slice {
            CudaStorageSlice::U8(inp) => inp.device_ptr().0 as *const c_void,
            CudaStorageSlice::U32(inp) => inp.device_ptr().0 as *const c_void,
            CudaStorageSlice::I64(inp) => inp.device_ptr().0 as *const c_void,
            CudaStorageSlice::BF16(inp) => inp.device_ptr().0 as *const c_void,
            CudaStorageSlice::F16(inp) => inp.device_ptr().0 as *const c_void,
            CudaStorageSlice::F32(inp) => inp.device_ptr().0 as *const c_void,
            CudaStorageSlice::F64(inp) => inp.device_ptr().0 as *const c_void,
            _ => candle_core::bail!("Unsupported dtype for sort"),
        };
        
        let dst_ptr = dst.device_ptr().0 as *mut c_void;
        
        unsafe {
            if self.asc {
                match storage.dtype() {
                    candle::DType::U8 => ffi::asort_asc_u8(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream_ptr),
                    candle::DType::U32 => ffi::asort_asc_u32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream_ptr),
                    candle::DType::I64 => ffi::asort_asc_i64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream_ptr),
                    candle::DType::BF16 => ffi::asort_asc_bf16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream_ptr),
                    candle::DType::F16 => ffi::asort_asc_f16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream_ptr),
                    candle::DType::F32 => ffi::asort_asc_f32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream_ptr),
                    candle::DType::F64 => ffi::asort_asc_f64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream_ptr),
                    _ => candle_core::bail!("Unsupported dtype for ArgSort"),
                }
            } else {
                match storage.dtype() {
                    candle::DType::U8 => ffi::asort_desc_u8(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream_ptr),
                    candle::DType::U32 => ffi::asort_desc_u32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream_ptr),
                    candle::DType::I64 => ffi::asort_desc_i64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream_ptr),
                    candle::DType::BF16 => ffi::asort_desc_bf16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream_ptr),
                    candle::DType::F16 => ffi::asort_desc_f16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream_ptr),
                    candle::DType::F32 => ffi::asort_desc_f32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream_ptr),
                    candle::DType::F64 => ffi::asort_desc_f64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream_ptr),
                    _ => candle_core::bail!("Unsupported dtype for ArgSort"),
                }
            }
        }
        
        let dst_ret = candle::cuda_backend::CudaStorage::new(CudaStorageSlice::U32(dst), dev.clone());
        Ok((dst_ret, layout.shape().clone()))
    }
}

pub trait ArgSortOp {
    fn arg_sort(&self, asc: bool) -> Result<Tensor>;
    fn sort(&self, asc: bool) -> Result<(Tensor, Tensor)>;
}

impl ArgSortOp for Tensor {
    fn arg_sort(&self, asc: bool) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "arg_sort" });
        }
        let last_dim = match self.dims().last() {
            Some(last_dim) => *last_dim,
            None => candle_core::bail!("empty last-dim in arg-sort"),
        };
        self.apply_op1_no_bwd(&ArgSort { asc, last_dim, inplace: false })
    }

    fn sort(&self, asc: bool) -> Result<(Tensor, Tensor)> {
        if !self.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "sort" });
        }
        let last_dim = match self.dims().last() {
            Some(last_dim) => *last_dim,
            None => candle_core::bail!("empty last-dim in sort"),
        };
        let sorted = self.copy()?;
        let asort = sorted.apply_op1_no_bwd(&ArgSort { asc, last_dim, inplace: true })?;
        Ok((sorted, asort))
    }
}

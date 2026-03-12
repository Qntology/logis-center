// Original implementation: https://github.com/EricLBuehler/candle-vllm/blob/master/src/backend/gptq.rs
#[cfg(feature = "cuda")]
use crate::models::attention::kernels::ffi::{
    awq_repack, gemm_half_q_half_alt, gptq_repack, marlin_4bit_bf16, marlin_4bit_f16,
    marlin_awq_4bit_bf16, marlin_awq_4bit_f16,
};
#[allow(unused_imports)]
use candle::backend::BackendStorage;
#[cfg(feature = "cuda")]
use candle::CudaStorage;
#[allow(unused_imports)]
use candle::{CpuStorage, DType, Layout, Result, Shape, Storage, Tensor};
use candle_core as candle;

#[allow(unused)]
struct GPTQMatMul {
    qzeros: Option<Tensor>,
    g_idx: Option<Tensor>,
    workspace: Option<Tensor>,
    bits: i32,
    group_size: i32,
    is_awq: bool,
}

impl GPTQMatMul {
    #[cfg(feature = "cuda")]
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        qweight: &CudaStorage,
        qweight_l: &Layout,
        scale: &CudaStorage,
        scale_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use std::ffi::c_void;
        let dev = qweight.device();
        let stream = dev.cu_stream();
        let x_shape = x_l.dims();
        let weight_shape = qweight_l.dims();

        let pack_factor: usize = 32 / self.bits as usize;
        let marlin_format = self.workspace.is_some();
        let size_k = weight_shape[0] * pack_factor * if marlin_format { 2 } else { 1 };
        let size_n = weight_shape[1] / if marlin_format { 2 } else { 1 };

        let mut out_shape: Vec<usize> = x_shape.to_vec();
        out_shape[x_shape.len() - 1] = size_n;
        let oshape: Shape = out_shape.into();

        let input = x.as_cuda_slice::<T>()?;
        let qw = qweight.as_cuda_slice::<u32>()?;
        let qs = scale.as_cuda_slice::<T>()?;

        let input = input.slice(x_l.start_offset()..);
        let qw = qw.slice(qweight_l.start_offset()..);
        let qs = qs.slice(scale_l.start_offset()..);

        let elem_count = oshape.elem_count();
        let out = unsafe { dev.alloc::<T>(elem_count) }.map_err(candle::Error::wrap)?;

        let out_ptr = out.device_ptr().0 as *mut c_void;
        let in_ptr = input.device_ptr().0 as *const c_void;
        let qw_ptr = qw.device_ptr().0 as *const c_void;
        let qs_ptr = qs.device_ptr().0 as *const c_void;

        let qzeros_ptr = if self.qzeros.is_some() {
            let (qzeros, qzeros_l) = self.qzeros.as_ref().unwrap().storage_and_layout();
            let qzeros = match &*qzeros {
                Storage::Cuda(p) => p,
                _ => candle::bail!("qzeros must be a cuda tensor"),
            };
            let qzeros_ = qzeros.as_cuda_slice::<u32>()?;
            let qzeros_ = qzeros_.slice(qzeros_l.start_offset()..);
            qzeros_.device_ptr().0 as *const c_void
        } else {
            std::ptr::null()
        };

        let g_idx_ptr = if self.g_idx.is_some() {
            let (g_idx, g_idx_l) = self.g_idx.as_ref().unwrap().storage_and_layout();
            let g_idx = match &*g_idx {
                Storage::Cuda(p) => p,
                _ => candle::bail!("g_idx must be a cuda tensor"),
            };
            let g_idx_ = g_idx.as_cuda_slice::<u32>()?;
            let g_idx_ = g_idx_.slice(g_idx_l.start_offset()..);
            g_idx_.device_ptr().0 as *const c_void
        } else {
            std::ptr::null()
        };

        unsafe {
            let stream_ptr = *stream as *const _ as i64;
            if marlin_format {
                let workspace_ptr = if self.workspace.is_some() {
                    let (workspace, workspace_l) =
                        self.workspace.as_ref().unwrap().storage_and_layout();
                    let workspace = match &*workspace {
                        Storage::Cuda(p) => p,
                        _ => candle::bail!("workspace must be a cuda tensor"),
                    };
                    let workspace_ = workspace.as_cuda_slice::<u32>()?;
                    let workspace_ = workspace_.slice(workspace_l.start_offset()..);
                    workspace_.device_ptr().0 as *const c_void
                } else {
                    candle::bail!("workspace is required for marlin matmul!")
                };

                if x.dtype() == DType::F16 {
                    if self.is_awq {
                        marlin_awq_4bit_f16(
                            in_ptr,
                            qw_ptr as *const i32,
                            qs_ptr,
                            qzeros_ptr,
                            g_idx_ptr,
                            out_ptr,
                            (x_shape[0] * x_shape[1]) as i32,
                            size_k as i32,
                            size_n as i32,
                            workspace_ptr,
                            self.group_size,
                            stream_ptr,
                        );
                    } else {
                        marlin_4bit_f16(
                            in_ptr,
                            qw_ptr as *const i32,
                            qs_ptr,
                            qzeros_ptr,
                            g_idx_ptr,
                            out_ptr,
                            (x_shape[0] * x_shape[1]) as i32,
                            size_k as i32,
                            size_n as i32,
                            workspace_ptr,
                            self.group_size,
                            stream_ptr,
                        );
                    }
                } else if x.dtype() == DType::BF16 {
                    if self.is_awq {
                        marlin_awq_4bit_bf16(
                            in_ptr,
                            qw_ptr as *const i32,
                            qs_ptr,
                            qzeros_ptr,
                            g_idx_ptr,
                            out_ptr,
                            (x_shape[0] * x_shape[1]) as i32,
                            size_k as i32,
                            size_n as i32,
                            workspace_ptr,
                            self.group_size,
                            stream_ptr,
                        );
                    } else {
                        marlin_4bit_bf16(
                            in_ptr,
                            qw_ptr as *const i32,
                            qs_ptr,
                            qzeros_ptr,
                            g_idx_ptr,
                            out_ptr,
                            (x_shape[0] * x_shape[1]) as i32,
                            size_k as i32,
                            size_n as i32,
                            workspace_ptr,
                            self.group_size,
                            stream_ptr,
                        );
                    }
                }
            } else {
                if x.dtype() == DType::F16 {
                    gemm_half_q_half_alt(
                        in_ptr,
                        qw_ptr as *const u32,
                        qzeros_ptr as *const u32,
                        qs_ptr,
                        g_idx_ptr as *const i32,
                        out_ptr,
                        (x_shape[0] * x_shape[1]) as i32,
                        size_n as i32,
                        size_k as i32,
                        self.bits,
                        stream_ptr,
                    )
                } else {
                    candle::bail!("GPTQMatMul is only supported for f16 non-marlin matmul.");
                }
            }
        }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, oshape))
    }
}

impl candle::CustomOp3 for GPTQMatMul {
    fn name(&self) -> &'static str {
        "GPTQMatMul"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for GPTQMatMul")
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        qweight: &CudaStorage,
        qweight_l: &Layout,
        scale: &CudaStorage,
        scale_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match x.dtype() {
            DType::F16 => self.cuda_fwd_t::<half::f16>(x, x_l, qweight, qweight_l, scale, scale_l),
            DType::BF16 => {
                self.cuda_fwd_t::<half::bf16>(x, x_l, qweight, qweight_l, scale, scale_l)
            }
            dt => candle::bail!("GPTQMatMul is only supported for f16 and bf16 ({dt:?})"),
        }
    }
}

pub fn gptq_matmul(
    x: &Tensor,
    qweight: &Tensor,
    scale: &Tensor,
    qzeros: &Option<Tensor>,
    g_idx: &Option<Tensor>,
    workspace: &Option<Tensor>,
    bits: i32,
    group_size: i32,
    is_awq: bool,
) -> Result<Tensor> {
    let op = GPTQMatMul {
        qzeros: qzeros.to_owned(),
        g_idx: g_idx.to_owned(),
        workspace: workspace.to_owned(),
        bits,
        group_size,
        is_awq,
    };
    x.apply_op3(qweight, scale, op)
}

#[allow(dead_code)]
struct MarlinRepack {
    bits: i32,
    is_awq: bool,
}

impl MarlinRepack {
    #[cfg(feature = "cuda")]
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        qweight: &CudaStorage,
        qweight_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        let dev = qweight.device();
        let stream = dev.cu_stream();
        let q_shape = qweight_l.dims();
        let mut out_shape: Vec<usize> = q_shape.to_vec();
        let pack_factor = (32 / self.bits) as usize;
        if self.is_awq {
            out_shape[0] = q_shape[0] / pack_factor / 2;
            out_shape[1] = q_shape[1] * pack_factor * 2;
        } else {
            out_shape[0] = q_shape[0] / 2;
            out_shape[1] = q_shape[1] * 2;
        }

        let oshape: Shape = out_shape.into();
        let q = qweight.as_cuda_slice::<u32>()?;
        let q = q.slice(qweight_l.start_offset()..);
        let elem_count = oshape.elem_count();
        let out = unsafe { dev.alloc::<u32>(elem_count) }.map_err(candle::Error::wrap)?;

        let out_ptr = out.device_ptr().0 as *const core::ffi::c_void;
        let q_ptr = q.device_ptr().0 as *const core::ffi::c_void;
        let stream_ptr = *stream as *const _ as i64;

        unsafe {
            if self.is_awq {
                awq_repack(
                    q_ptr,
                    out_ptr,
                    q_shape[0] as i32,
                    q_shape[1] as i32,
                    self.bits,
                    stream_ptr,
                )
            } else {
                gptq_repack(
                    q_ptr,
                    out_ptr,
                    q_shape[0] as i32,
                    q_shape[1] as i32,
                    stream_ptr,
                )
            }
        }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, oshape))
    }
}

impl candle::CustomOp1 for MarlinRepack {
    fn name(&self) -> &'static str {
        "MarlinRepack"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for MarlinRepack")
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, qweight: &CudaStorage, qweight_l: &Layout) -> Result<(CudaStorage, Shape)> {
        match qweight.dtype() {
            DType::U32 => self.cuda_fwd_t::<u32>(qweight, qweight_l),
            dt => candle::bail!("MarlinRepack is only supported for i32/u32 weight ({dt:?})"),
        }
    }
}

pub fn marlin_weight_repack(qweight: &Tensor, bits: i32, is_awq: bool) -> Result<Tensor> {
    let op = MarlinRepack { bits, is_awq };
    qweight.apply_op1(op)
}

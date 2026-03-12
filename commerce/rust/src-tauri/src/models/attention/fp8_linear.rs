#[cfg(feature = "cuda")]
use crate::models::attention::cuda_utils;
#[cfg(feature = "cuda")]
use crate::models::attention::kernels::ffi;
#[cfg(feature = "metal")]
use crate::models::attention::metal_kernels;
#[cfg(all(feature = "cuda", feature = "flashinfer"))]
use candle_core::cuda_backend::cudarc::driver::CudaSlice;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, Result, Tensor};
#[cfg(all(feature = "cuda", feature = "flashinfer"))]
use std::cell::RefCell;

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
struct FlashInferFp8Workspace {
    buffer: CudaSlice<u8>,
    size: usize,
    device_ordinal: usize,
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
thread_local! {
    static FLASHINFER_FP8_WORKSPACE: RefCell<Option<FlashInferFp8Workspace>> = const { RefCell::new(None) };
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn get_or_init_flashinfer_fp8_workspace(
    dev: &candle_core::cuda_backend::CudaDevice,
    required_size: usize,
) -> Result<(*mut std::ffi::c_void, usize)> {
    FLASHINFER_FP8_WORKSPACE.with(|cell| {
        let mut slot = cell.borrow_mut();
        let ordinal = dev.ordinal();

        let needs_init = match slot.as_ref() {
            None => true,
            Some(existing) => existing.device_ordinal != ordinal || existing.size < required_size,
        };

        if needs_init {
            let alloc_size = required_size.max(1);
            let buffer = unsafe { dev.alloc::<u8>(alloc_size) }.map_err(candle_core::Error::wrap)?;
            *slot = Some(FlashInferFp8Workspace {
                buffer,
                size: alloc_size,
                device_ordinal: ordinal,
            });
        }

        let ws = slot.as_ref().unwrap();
        let _stream = dev.cu_stream();
        Ok((*ws.buffer.device_ptr() as *mut std::ffi::c_void, ws.size))
    })
}

#[cfg(feature = "cuda")]
fn get_cuda_slice<
    T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType,
>(
    tensor: &Tensor,
    dev: &candle_core::cuda_backend::CudaDevice,
) -> Result<u64> {
    let (storage, _) = tensor.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(c) => {
            let slice = c.as_cuda_slice::<T>()?;
            let _stream = dev.cu_stream();
            Ok(*slice.device_ptr())
        }
        _ => candle_core::bail!("expecting cuda tensor"),
    }
}

/// FP8 Matrix Multiplication: C = A * B^T (conventional path).
#[allow(unused)]
pub fn fp8_matmul(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    block_size: &[usize],
) -> Result<Tensor> {
    let (m, k) = input.dims2()?;
    let (n, k_w) = weight.dims2()?;

    if k != k_w {
        candle_core::bail!("Shape mismatch in fp8_matmul");
    }

    let dev = input.device();
    let dtype = input.dtype();
    let scale_row_stride = (k_w + block_size[1] - 1) / block_size[1];
    let output = Tensor::zeros((m, n), dtype, dev)?;

    match (dev, dtype) {
        #[cfg(feature = "cuda")]
        (Device::Cuda(cu_dev), DType::F16) => {
            let stream = cu_dev.cu_stream();
            let stream_ptr = *stream as i64;

            let (input_storage, _) = input.storage_and_layout();
            let input_ptr = match &*input_storage {
                candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<half::f16>()?.device_ptr() as *const core::ffi::c_void,
                _ => candle_core::bail!("input must be a cuda tensor"),
            };

            let (weight_storage, _) = weight.storage_and_layout();
            let weight_ptr = match &*weight_storage {
                candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr() as *const u8,
                _ => candle_core::bail!("weight must be a cuda tensor"),
            };

            let (scale_storage, _) = weight_scale.storage_and_layout();
            let weight_scale_ptr = match &*scale_storage {
                candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr() as *const f32,
                _ => candle_core::bail!("weight_scale must be a cuda tensor"),
            };

            let (output_storage, _) = output.storage_and_layout();
            let output_ptr = match &*output_storage {
                candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<half::f16>()?.device_ptr() as *mut core::ffi::c_void,
                _ => candle_core::bail!("output allocation failed"),
            };

            unsafe {
                ffi::fp8_matmul_f16(
                    input_ptr, weight_ptr, weight_scale_ptr, output_ptr,
                    m as i32, n as i32, k as i32, scale_row_stride as i32,
                    block_size[0] as i32, block_size[1] as i32, stream_ptr,
                )
            }
        }
        #[cfg(feature = "cuda")]
        (Device::Cuda(cu_dev), DType::BF16) => {
            let stream = cu_dev.cu_stream();
            let stream_ptr = *stream as i64;

            let (input_storage, _) = input.storage_and_layout();
            let input_ptr = match &*input_storage {
                candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<half::bf16>()?.device_ptr() as *const core::ffi::c_void,
                _ => candle_core::bail!("input must be a cuda tensor"),
            };

            let (weight_storage, _) = weight.storage_and_layout();
            let weight_ptr = match &*weight_storage {
                candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr() as *const u8,
                _ => candle_core::bail!("weight must be a cuda tensor"),
            };

            let (scale_storage, _) = weight_scale.storage_and_layout();
            let weight_scale_ptr = match &*scale_storage {
                candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr() as *const f32,
                _ => candle_core::bail!("weight_scale must be a cuda tensor"),
            };

            let (output_storage, _) = output.storage_and_layout();
            let output_ptr = match &*output_storage {
                candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<half::bf16>()?.device_ptr() as *mut core::ffi::c_void,
                _ => candle_core::bail!("output allocation failed"),
            };

            unsafe {
                ffi::fp8_matmul_bf16(
                    input_ptr, weight_ptr, weight_scale_ptr, output_ptr,
                    m as i32, n as i32, k as i32, scale_row_stride as i32,
                    block_size[0] as i32, block_size[1] as i32, stream_ptr,
                )
            }
        }
        #[cfg(feature = "metal")]
        (Device::Metal(dev), _) => {
            let (input_storage, input_layout) = input.storage_and_layout();
            let input_slice = match &*input_storage { candle_core::Storage::Metal(c) => c, _ => candle_core::bail!("input must be a metal tensor") };
            let input_offset = input_layout.start_offset() * input.dtype().size_in_bytes();
            let (weight_storage, weight_layout) = weight.storage_and_layout();
            let weight_slice = match &*weight_storage { candle_core::Storage::Metal(c) => c, _ => candle_core::bail!("weight must be a metal tensor") };
            let weight_offset = weight_layout.start_offset() * weight.dtype().size_in_bytes();
            let (scale_storage, scale_layout) = weight_scale.storage_and_layout();
            let scale_slice = match &*scale_storage { candle_core::Storage::Metal(c) => c, _ => candle_core::bail!("weight_scale must be a metal tensor") };
            let scale_offset = scale_layout.start_offset() * weight_scale.dtype().size_in_bytes();
            let (output_storage, output_layout) = output.storage_and_layout();
            let output_slice = match &*output_storage { candle_core::Storage::Metal(c) => c, _ => candle_core::bail!("output allocation failed") };
            let output_offset = output_layout.start_offset() * output.dtype().size_in_bytes();

            let command_buffer = dev.command_buffer()?;
            metal_kernels::call_fp8_matmul(
                dev.device(), &command_buffer, metal_kernels::Kernels::default(), dtype,
                input_slice.buffer(), input_offset, weight_slice.buffer(), weight_offset,
                scale_slice.buffer(), scale_offset, output_slice.buffer(), output_offset,
                m as i32, n as i32, k as i32, scale_row_stride as i32, block_size[0] as i32, block_size[1] as i32,
            ).map_err(candle_core::Error::wrap)?;
        }
        _ => candle_core::bail!("fp8_matmul unsupported configuration"),
    }
    Ok(output)
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
pub fn fp8_matmul_flashinfer(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
) -> Result<Tensor> {
    let (m, k) = input.dims2()?;
    let (n, k_w) = weight.dims2()?;
    if k != k_w { candle_core::bail!("Shape mismatch in fp8_matmul_flashinfer"); }

    let dev = input.device();
    let cu_dev = dev.as_cuda_device()?;
    let stream = cu_dev.cu_stream();
    let stream_ptr = *stream as *const _ as i64;

    let out = Tensor::zeros((m, n), DType::BF16, dev)?;
    let k_over_128 = k / 128;
    let input_q = Tensor::zeros((m, k), DType::U8, dev)?;
    let m_padded = (m + 4 - 1) / 4 * 4;
    let input_scale = Tensor::zeros((k_over_128, m_padded), DType::F32, dev)?;
    let scale_stride = input_scale.stride()[0] as i32;

    let q_ptr = get_cuda_slice::<u8>(&input_q, cu_dev)? as *mut std::ffi::c_void;
    let s_ptr = get_cuda_slice::<f32>(&input_scale, cu_dev)? as *mut f32;
    let inp_ptr = get_cuda_slice::<half::bf16>(input, cu_dev)? as *const std::ffi::c_void;

    unsafe {
        ffi::fp8_quantize_per_token_group_launch(
            inp_ptr, q_ptr, s_ptr, (m * k_over_128) as i32, 128, k_over_128 as i32, scale_stride, false, true, stream_ptr,
        );
    }

    let required_ws = unsafe { ffi::flashinfer_fp8_blockscale_workspace_size_fp8(m as i32, n as i32, k as i32) };
    let (workspace_ptr, workspace_size) = get_or_init_flashinfer_fp8_workspace(cu_dev, required_ws)?;

    let weight_ptr = get_cuda_slice::<u8>(weight, cu_dev)? as *const std::ffi::c_void;
    let weight_scale_ptr = get_cuda_slice::<f32>(weight_scale, cu_dev)? as *const f32;
    let out_ptr = get_cuda_slice::<half::bf16>(&out, cu_dev)? as *mut std::ffi::c_void;

    let status = unsafe {
        ffi::flashinfer_fp8_blockscale_fp8(
            q_ptr, s_ptr, weight_ptr, weight_scale_ptr, out_ptr,
            m as i32, n as i32, k as i32, workspace_ptr, workspace_size, stream_ptr,
        )
    };
    if status != 0 { candle_core::bail!("flashinfer fp8 blockscale gemm failed: {status}"); }
    Ok(out)
}

#[cfg(all(feature = "cuda", feature = "cutlass"))]
#[allow(unused)]
pub fn fp8_matmul_cutlass(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    block_size: &[usize],
) -> Result<Tensor> {
    let (m, k) = input.dims2()?;
    let (k_b, n) = weight.dims2()?;
    let dev = input.device();
    let cu_dev = dev.as_cuda_device()?;
    let stream = cu_dev.cu_stream();
    let stream_ptr = *stream as *const _ as i64;
    let dtype = input.dtype();
    let scale_row_stride = (k + block_size[1] - 1) / block_size[1];
    let sm_version = cuda_utils::sm_version(cu_dev).unwrap_or(0) as i32;

    let w_ptr = get_cuda_slice::<u8>(weight, cu_dev)?;
    let ws_ptr = get_cuda_slice::<f32>(weight_scale, cu_dev)?;

    let m_padded = (m + 3) / 4 * 4;
    let input_padded = if m_padded > m { input.pad_with_zeros(0, 0, m_padded - m)? } else { input.clone() };
    let output = Tensor::zeros((m_padded, n), dtype, dev)?;

    let k_over_128 = (k + 127) / 128;
    let input_q = Tensor::zeros((m_padded, k), DType::U8, dev)?;
    let input_scale = Tensor::zeros((m_padded, k_over_128), DType::F32, dev)?;
    let scale_stride = input_scale.stride()[0] as i32;

    let q_ptr = get_cuda_slice::<u8>(&input_q, cu_dev)? as *mut std::ffi::c_void;
    let s_ptr = get_cuda_slice::<f32>(&input_scale, cu_dev)? as *mut f32;
    let inp_ptr = if dtype == DType::F16 { get_cuda_slice::<half::f16>(&input_padded, cu_dev)? } else { get_cuda_slice::<half::bf16>(&input_padded, cu_dev)? };

    unsafe {
        ffi::fp8_quantize_per_token_group_launch(
            inp_ptr as *const std::ffi::c_void, q_ptr, s_ptr, (m_padded * k_over_128) as i32, 128, k_over_128 as i32, scale_stride, dtype == DType::F16, true, stream_ptr,
        );
    }

    let out_ptr = if dtype == DType::F16 { get_cuda_slice::<half::f16>(&output, cu_dev)? } else { get_cuda_slice::<half::bf16>(&output, cu_dev)? };

    unsafe {
        if dtype == DType::F16 {
            ffi::fp8_matmul_f16_cutlass(q_ptr as *const u8, s_ptr, w_ptr as *const u8, ws_ptr, out_ptr as *mut _, m_padded as i32, n as i32, k as i32, scale_row_stride as i32, block_size[0] as i32, block_size[1] as i32, sm_version, stream_ptr)
        } else {
            ffi::fp8_matmul_bf16_cutlass(q_ptr as *const u8, s_ptr, w_ptr as *const u8, ws_ptr, out_ptr as *mut _, m_padded as i32, n as i32, k as i32, scale_row_stride as i32, block_size[0] as i32, block_size[1] as i32, sm_version, stream_ptr)
        }
    }

    if m_padded > m { Ok(output.narrow(0, 0, m)?.contiguous()?) } else { Ok(output) }
}

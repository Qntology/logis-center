// GDN (Gated Delta Net) operations module
// Provides Rust interfaces for GDN CUDA kernels used in Qwen3.5's linear attention layers.

#[cfg(feature = "cuda")]
use candle_core as candle;
use candle_core::{DType, IndexOp, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_core::{Device, Storage};
#[cfg(feature = "cuda")]
use half::{bf16, f16};
#[cfg(feature = "cuda")]
use crate::models::attention::kernels::ffi;
#[cfg(feature = "cuda")]
use std::ffi::{c_int, c_void};

#[cfg(feature = "cuda")]
fn get_cuda_const_ptr(t: &Tensor, dev: &candle::CudaDevice) -> Result<*const c_void> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let (storage, layout) = t.storage_and_layout();
    let offset = layout.start_offset();
    match (&*storage, t.dtype()) {
        (Storage::Cuda(s), DType::F16) => {
            Ok(*s.as_cuda_slice::<f16>()?.slice(offset..).device_ptr() as *const c_void)
        }
        (Storage::Cuda(s), DType::BF16) => {
            Ok(*s.as_cuda_slice::<bf16>()?.slice(offset..).device_ptr() as *const c_void)
        }
        (Storage::Cuda(s), DType::F32) => {
            Ok(*s.as_cuda_slice::<f32>()?.slice(offset..).device_ptr() as *const c_void)
        }
        _ => candle_core::bail!("Expected CUDA tensor with f16/bf16/f32 dtype"),
    }
}

#[cfg(feature = "cuda")]
fn get_cuda_const_ptr_u32(t: &Tensor, dev: &candle::CudaDevice) -> Result<*const u32> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let (storage, layout) = t.storage_and_layout();
    let offset = layout.start_offset();
    match &*storage {
        Storage::Cuda(s) => {
            Ok(*s.as_cuda_slice::<u32>()?.slice(offset..).device_ptr() as *const u32)
        }
        _ => candle_core::bail!("Expected CUDA u32 tensor"),
    }
}

#[cfg(feature = "cuda")]
fn get_cuda_const_ptr_i64(t: &Tensor, dev: &candle::CudaDevice) -> Result<*const i64> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let (storage, layout) = t.storage_and_layout();
    let offset = layout.start_offset();
    match &*storage {
        Storage::Cuda(s) => {
            Ok(*s.as_cuda_slice::<i64>()?.slice(offset..).device_ptr() as *const i64)
        }
        _ => candle_core::bail!("Expected CUDA i64 tensor"),
    }
}

#[cfg(feature = "cuda")]
fn get_cuda_mut_ptr(t: &Tensor, dev: &candle::CudaDevice) -> Result<*mut c_void> {
    Ok(get_cuda_const_ptr(t, dev)? as *mut c_void)
}

#[cfg(feature = "cuda")]
fn ensure_contiguous(t: &Tensor) -> Result<Tensor> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

/// Causal conv1d forward pass for variable-length sequences (prefill mode).
#[cfg(feature = "cuda")]
pub fn causal_conv1d_fwd(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    cu_seqlens: Option<&Tensor>,
    activation_silu: bool,
) -> Result<Tensor> {
    match (x.device(), x.dtype(), cu_seqlens) {
        (Device::Cuda(dev), DType::F16 | DType::BF16 | DType::F32, Some(cu)) => {
            let (total_tokens, d_conv) = x.dims2()?;
            let kernel_size = weight.dim(2)?;
            if kernel_size > 16 {
                return causal_conv1d_fwd_naive_with_state(
                    x,
                    weight,
                    bias,
                    conv_state,
                    Some(cu),
                    activation_silu,
                );
            }
            let batch = conv_state.dim(0)?;
            let out = Tensor::zeros((total_tokens, d_conv), x.dtype(), x.device())?;
            let cu_u32 = if cu.dtype() == DType::U32 {
                cu.clone()
            } else {
                cu.to_dtype(DType::U32)?
            };

            let x_ptr = get_cuda_const_ptr(x, dev)?;
            let weight_ptr = get_cuda_const_ptr(weight, dev)?;
            let bias_ptr = if let Some(b) = bias {
                get_cuda_const_ptr(b, dev)?
            } else {
                std::ptr::null()
            };
            let state_ptr = get_cuda_mut_ptr(conv_state, dev)?;
            let out_ptr = get_cuda_mut_ptr(&out, dev)?;
            let cu_ptr = get_cuda_const_ptr_u32(&cu_u32, dev)?;
            let stream = dev.cu_stream();
            let stream_ptr = *stream as *const _ as i64;

            unsafe {
                match x.dtype() {
                    DType::F16 => ffi::causal_conv1d_fwd_f16(
                        x_ptr,
                        weight_ptr,
                        bias_ptr,
                        state_ptr,
                        out_ptr,
                        cu_ptr,
                        batch as c_int,
                        d_conv as c_int,
                        kernel_size as c_int,
                        activation_silu,
                        stream_ptr,
                    ),
                    DType::BF16 => ffi::causal_conv1d_fwd_bf16(
                        x_ptr,
                        weight_ptr,
                        bias_ptr,
                        state_ptr,
                        out_ptr,
                        cu_ptr,
                        batch as c_int,
                        d_conv as c_int,
                        kernel_size as c_int,
                        activation_silu,
                        stream_ptr,
                    ),
                    DType::F32 => ffi::causal_conv1d_fwd_f32(
                        x_ptr as *const f32,
                        weight_ptr as *const f32,
                        bias_ptr as *const f32,
                        state_ptr as *mut f32,
                        out_ptr as *mut f32,
                        cu_ptr,
                        batch as c_int,
                        d_conv as c_int,
                        kernel_size as c_int,
                        activation_silu,
                        stream_ptr,
                    ),
                    _ => unreachable!(),
                }
            }
            Ok(out)
        }
        _ => causal_conv1d_fwd_naive_with_state(
            x,
            weight,
            bias,
            conv_state,
            cu_seqlens,
            activation_silu,
        ),
    }
}

/// Causal conv1d single-step update for decode mode.
#[cfg(feature = "cuda")]
pub fn causal_conv1d_update(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    activation_silu: bool,
) -> Result<Tensor> {
    match (x.device(), x.dtype()) {
        (Device::Cuda(dev), DType::F16 | DType::BF16 | DType::F32) => {
            let (batch, d_conv) = x.dims2()?;
            let kernel_size = weight.dim(2)?;
            let out = Tensor::zeros((batch, d_conv), x.dtype(), x.device())?;

            let x_ptr = get_cuda_const_ptr(x, dev)?;
            let weight_ptr = get_cuda_const_ptr(weight, dev)?;
            let bias_ptr = if let Some(b) = bias {
                get_cuda_const_ptr(b, dev)?
            } else {
                std::ptr::null()
            };
            let state_ptr = get_cuda_mut_ptr(conv_state, dev)?;
            let out_ptr = get_cuda_mut_ptr(&out, dev)?;
            let stream = dev.cu_stream();
            let stream_ptr = *stream as *const _ as i64;

            unsafe {
                match x.dtype() {
                    DType::F16 => ffi::causal_conv1d_update_f16(
                        x_ptr,
                        weight_ptr,
                        bias_ptr,
                        state_ptr,
                        out_ptr,
                        batch as c_int,
                        d_conv as c_int,
                        kernel_size as c_int,
                        activation_silu,
                        stream_ptr,
                    ),
                    DType::BF16 => ffi::causal_conv1d_update_bf16(
                        x_ptr,
                        weight_ptr,
                        bias_ptr,
                        state_ptr,
                        out_ptr,
                        batch as c_int,
                        d_conv as c_int,
                        kernel_size as c_int,
                        activation_silu,
                        stream_ptr,
                    ),
                    DType::F32 => ffi::causal_conv1d_update_f32(
                        x_ptr as *const f32,
                        weight_ptr as *const f32,
                        bias_ptr as *const f32,
                        state_ptr as *mut f32,
                        out_ptr as *mut f32,
                        batch as c_int,
                        d_conv as c_int,
                        kernel_size as c_int,
                        activation_silu,
                        stream_ptr,
                    ),
                    _ => unreachable!(),
                }
            }
            Ok(out)
        }
        _ => causal_conv1d_update_naive(x, weight, bias, conv_state, activation_silu),
    }
}

/// Causal conv1d single-step update with slot-indexed global state.
#[cfg(feature = "cuda")]
pub fn causal_conv1d_update_slots(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    slots: &Tensor,
    activation_silu: bool,
) -> Result<Tensor> {
    match (x.device(), x.dtype()) {
        (Device::Cuda(dev), DType::F16 | DType::BF16 | DType::F32) => {
            let x_c = x.contiguous()?;
            let weight_c = weight.contiguous()?;
            let bias_c = if let Some(b) = bias {
                Some(b.contiguous()?)
            } else {
                None
            };

            let (batch, d_conv) = x_c.dims2()?;
            let kernel_size = weight_c.dim(2)?;
            if slots.dtype() != DType::I64 || slots.dim(0)? != batch {
                candle_core::bail!(
                    "causal_conv1d_update_slots expects slots [batch] I64, got {:?} {:?}",
                    slots.shape(),
                    slots.dtype()
                );
            }
            let out = Tensor::zeros((batch, d_conv), x.dtype(), x.device())?;

            let x_ptr = get_cuda_const_ptr(&x_c, dev)?;
            let weight_ptr = get_cuda_const_ptr(&weight_c, dev)?;
            let bias_ptr = if let Some(ref b) = bias_c {
                get_cuda_const_ptr(b, dev)?
            } else {
                std::ptr::null()
            };
            let state_ptr = get_cuda_mut_ptr(conv_state, dev)?;
            let slots_ptr = get_cuda_const_ptr_i64(slots, dev)?;
            let out_ptr = get_cuda_mut_ptr(&out, dev)?;
            let stream = dev.cu_stream();
            let stream_ptr = *stream as *const _ as i64;

            unsafe {
                match x.dtype() {
                    DType::F16 => {
                        let slots_ptr_i32 = slots_ptr as *const i32;
                        ffi::causal_conv1d_update_slots_f16(
                            x_ptr,
                            weight_ptr,
                            bias_ptr,
                            state_ptr,
                            slots_ptr_i32,
                            out_ptr,
                            batch as c_int,
                            d_conv as c_int,
                            kernel_size as c_int,
                            activation_silu,
                            stream_ptr,
                        )
                    }
                    DType::BF16 => {
                        let slots_ptr_i32 = slots_ptr as *const i32;
                        ffi::causal_conv1d_update_slots_bf16(
                            x_ptr,
                            weight_ptr,
                            bias_ptr,
                            state_ptr,
                            slots_ptr_i32,
                            out_ptr,
                            batch as c_int,
                            d_conv as c_int,
                            kernel_size as c_int,
                            activation_silu,
                            stream_ptr,
                        )
                    }
                    DType::F32 => {
                        let slots_ptr_i32 = slots_ptr as *const i32;
                        ffi::causal_conv1d_update_slots_f32(
                            x_ptr as *const f32,
                            weight_ptr as *const f32,
                            bias_ptr as *const f32,
                            state_ptr as *mut f32,
                            slots_ptr_i32,
                            out_ptr as *mut f32,
                            batch as c_int,
                            d_conv as c_int,
                            kernel_size as c_int,
                            activation_silu,
                            stream_ptr,
                        )
                    }
                    _ => unreachable!(),
                }
            }
            Ok(out)
        }
        _ => {
            // Non-CUDA fallback
            let slots_vec = if slots.dtype() == DType::I64 {
                slots.to_vec1::<i64>()?
            } else {
                candle_core::bail!("causal_conv1d_update_slots fallback expects I64 slots");
            };
            if slots_vec.is_empty() {
                candle_core::bail!("causal_conv1d_update_slots got empty slots");
            }
            let mut gathered = Vec::with_capacity(slots_vec.len());
            for &s in &slots_vec {
                gathered.push(conv_state.i(s as usize)?);
            }
            let gathered_refs = gathered.iter().collect::<Vec<_>>();
            let mut batch_state = Tensor::stack(&gathered_refs, 0)?;
            let out =
                causal_conv1d_update_naive(x, weight, bias, &mut batch_state, activation_silu)?;
            for (i, &s) in slots_vec.iter().enumerate() {
                *conv_state = conv_state.slice_assign(
                    &[
                        s as usize..s as usize + 1,
                        0..conv_state.dim(1)?,
                        0..conv_state.dim(2)?,
                    ],
                    &batch_state.narrow(0, i, 1)?,
                )?;
            }
            Ok(out)
        }
    }
}

/// Fused GDN gating computation.
#[cfg(feature = "cuda")]
pub fn fused_gdn_gating(
    a_log: &Tensor,
    a: &Tensor,
    b: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    match (a.device(), a.dtype()) {
        (Device::Cuda(dev), DType::F16 | DType::BF16 | DType::F32) => {
            let (batch, seq_len, heads) = a.dims3()?;
            let g = Tensor::zeros(a.shape(), a.dtype(), a.device())?;
            let beta = Tensor::zeros(a.shape(), a.dtype(), a.device())?;

            let al_ptr = get_cuda_const_ptr(a_log, dev)?;
            let a_ptr = get_cuda_const_ptr(a, dev)?;
            let b_ptr = get_cuda_const_ptr(b, dev)?;
            let dt_ptr = get_cuda_const_ptr(dt_bias, dev)?;
            let g_ptr = get_cuda_mut_ptr(&g, dev)?;
            let beta_ptr = get_cuda_mut_ptr(&beta, dev)?;
            let stream = dev.cu_stream();
            let stream_ptr = *stream as *const _ as i64;

            unsafe {
                match a.dtype() {
                    DType::F16 => {
                        if a_log.dtype() == DType::F32 {
                            ffi::fused_gdn_gating_f16_alog_f32(
                                al_ptr as *const f32,
                                a_ptr,
                                b_ptr,
                                dt_ptr,
                                g_ptr,
                                beta_ptr,
                                batch as c_int,
                                seq_len as c_int,
                                heads as c_int,
                                stream_ptr,
                            )
                        } else {
                            ffi::fused_gdn_gating_f16(
                                al_ptr,
                                a_ptr,
                                b_ptr,
                                dt_ptr,
                                g_ptr,
                                beta_ptr,
                                batch as c_int,
                                seq_len as c_int,
                                heads as c_int,
                                stream_ptr,
                            )
                        }
                    }
                    DType::BF16 => {
                        if a_log.dtype() == DType::F32 {
                            ffi::fused_gdn_gating_bf16_alog_f32(
                                al_ptr as *const f32,
                                a_ptr,
                                b_ptr,
                                dt_ptr,
                                g_ptr,
                                beta_ptr,
                                batch as c_int,
                                seq_len as c_int,
                                heads as c_int,
                                stream_ptr,
                            )
                        } else {
                            ffi::fused_gdn_gating_bf16(
                                al_ptr,
                                a_ptr,
                                b_ptr,
                                dt_ptr,
                                g_ptr,
                                beta_ptr,
                                batch as c_int,
                                seq_len as c_int,
                                heads as c_int,
                                stream_ptr,
                            )
                        }
                    }
                    DType::F32 => ffi::fused_gdn_gating_f32(
                        al_ptr as *const f32,
                        a_ptr as *const f32,
                        b_ptr as *const f32,
                        dt_ptr as *const f32,
                        g_ptr as *mut f32,
                        beta_ptr as *mut f32,
                        batch as c_int,
                        seq_len as c_int,
                        heads as c_int,
                        stream_ptr,
                    ),
                    _ => unreachable!(),
                }
            }
            Ok((g, beta))
        }
        _ => fused_gdn_gating_naive(a_log, a, b, dt_bias),
    }
}

/// Fused gated RMSNorm.
#[cfg(feature = "cuda")]
pub fn gated_rmsnorm_silu_mul(
    x: &Tensor,
    z: &Tensor,
    norm_weight: &Tensor,
    norm_bias: Option<&Tensor>,
    eps: f64,
    group_size: usize,
) -> Result<Tensor> {
    match (x.device(), x.dtype()) {
        (Device::Cuda(dev), DType::F16 | DType::BF16 | DType::F32) => {
            let x_c = x.contiguous()?;
            let (rows, value_dim) = x_c.dims2()?;
            let z_c = if z.dtype() == x.dtype() {
                z.contiguous()?
            } else {
                z.to_dtype(x.dtype())?.contiguous()?
            };
            let (z_rows, z_dim) = z_c.dims2()?;
            if z_rows != rows || z_dim != value_dim {
                candle_core::bail!("gated_rmsnorm_silu_mul shape mismatch");
            }
            let out = Tensor::zeros((rows, value_dim), x.dtype(), x.device())?;

            let x_ptr = get_cuda_const_ptr(&x_c, dev)?;
            let z_ptr = get_cuda_const_ptr(&z_c, dev)?;
            let w_ptr = get_cuda_const_ptr(&norm_weight, dev)?;
            let b_ptr = if let Some(b) = norm_bias {
                get_cuda_const_ptr(b, dev)?
            } else {
                std::ptr::null()
            };
            let out_ptr = get_cuda_mut_ptr(&out, dev)?;
            let stream = dev.cu_stream();
            let stream_ptr = *stream as *const _ as i64;
            let eps = eps as f32;

            let per_group_weights = norm_weight.dim(0)? == group_size;

            unsafe {
                match x.dtype() {
                    DType::F16 => {
                        if norm_weight.dtype() == DType::F32 {
                            ffi::gdn_gated_rmsnorm_silu_mul_f16_wf32(
                                x_ptr,
                                z_ptr,
                                w_ptr as *const f32,
                                b_ptr as *const f32,
                                out_ptr,
                                rows as c_int,
                                value_dim as c_int,
                                group_size as c_int,
                                eps,
                                per_group_weights,
                                norm_bias.is_some(),
                                stream_ptr,
                            )
                        } else {
                            ffi::gdn_gated_rmsnorm_silu_mul_f16(
                                x_ptr,
                                z_ptr,
                                w_ptr,
                                b_ptr,
                                out_ptr,
                                rows as c_int,
                                value_dim as c_int,
                                group_size as c_int,
                                eps,
                                per_group_weights,
                                norm_bias.is_some(),
                                stream_ptr,
                            )
                        }
                    }
                    DType::BF16 => {
                        if norm_weight.dtype() == DType::F32 {
                            ffi::gdn_gated_rmsnorm_silu_mul_bf16_wf32(
                                x_ptr,
                                z_ptr,
                                w_ptr as *const f32,
                                b_ptr as *const f32,
                                out_ptr,
                                rows as c_int,
                                value_dim as c_int,
                                group_size as c_int,
                                eps,
                                per_group_weights,
                                norm_bias.is_some(),
                                stream_ptr,
                            )
                        } else {
                            ffi::gdn_gated_rmsnorm_silu_mul_bf16(
                                x_ptr,
                                z_ptr,
                                w_ptr,
                                b_ptr,
                                out_ptr,
                                rows as c_int,
                                value_dim as c_int,
                                group_size as c_int,
                                eps,
                                per_group_weights,
                                norm_bias.is_some(),
                                stream_ptr,
                            )
                        }
                    }
                    DType::F32 => ffi::gdn_gated_rmsnorm_silu_mul_f32(
                        x_ptr as *const f32,
                        z_ptr as *const f32,
                        w_ptr as *const f32,
                        b_ptr as *const f32,
                        out_ptr as *mut f32,
                        rows as c_int,
                        value_dim as c_int,
                        group_size as c_int,
                        eps,
                        per_group_weights,
                        norm_bias.is_some(),
                        stream_ptr,
                    ),
                    _ => unreachable!(),
                }
            }
            Ok(out)
        }
        _ => gated_rmsnorm_silu_mul_naive(x, z, norm_weight, norm_bias, eps, group_size),
    }
}

/// DeltaNet recurrent update.
#[cfg(feature = "cuda")]
pub fn gated_delta_rule_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    match q.device() {
        Device::Cuda(dev) => {
            let (bh, seq_len, k_dim) = q.dims3()?;
            let v_dim = v.dim(2)?;
            let q_c = ensure_contiguous(q)?;
            let k_c = ensure_contiguous(k)?;
            let v_c = ensure_contiguous(v)?;
            let out_dtype = q_c.dtype();

            let g_f32 = if g.dtype() == DType::F32 {
                ensure_contiguous(g)?
            } else {
                g.to_dtype(DType::F32)?.contiguous()?
            };
            let beta_f32 = if beta.dtype() == DType::F32 {
                ensure_contiguous(beta)?
            } else {
                beta.to_dtype(DType::F32)?.contiguous()?
            };

            let state_ptr = get_cuda_mut_ptr(state, dev)? as *mut f32;
            let out = Tensor::zeros((bh, seq_len, v_dim), DType::F32, q_c.device())?;

            let q_ptr = get_cuda_const_ptr(&q_c, dev)?;
            let k_ptr = get_cuda_const_ptr(&k_c, dev)?;
            let v_ptr = get_cuda_const_ptr(&v_c, dev)?;
            let g_ptr = get_cuda_const_ptr(&g_f32, dev)? as *const f32;
            let beta_ptr = get_cuda_const_ptr(&beta_f32, dev)? as *const f32;
            let out_ptr = get_cuda_mut_ptr(&out, dev)? as *mut f32;
            let stream = dev.cu_stream();
            let stream_ptr = *stream as *const _ as i64;

            unsafe {
                match out_dtype {
                    DType::F32 => ffi::gated_delta_rule_recurrence(
                        q_ptr as *const f32,
                        k_ptr as *const f32,
                        v_ptr as *const f32,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        out_ptr,
                        bh as c_int,
                        seq_len as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        stream_ptr,
                    ),
                    DType::F16 => ffi::gated_delta_rule_recurrence_f16(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        out_ptr,
                        bh as c_int,
                        seq_len as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        stream_ptr,
                    ),
                    DType::BF16 => ffi::gated_delta_rule_recurrence_bf16(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        out_ptr,
                        bh as c_int,
                        seq_len as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        stream_ptr,
                    ),
                    dt => candle_core::bail!("gated_delta_rule_recurrence unsupported dtype: {:?}", dt),
                }
            }
            out.to_dtype(out_dtype)
        }
        _ => gated_delta_rule_recurrence_naive(q, k, v, g, beta, state),
    }
}

/// One-step decode recurrence with slots.
#[cfg(feature = "cuda")]
pub fn gated_delta_rule_decode_slots(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    slots: &Tensor,
) -> Result<Tensor> {
    match q.device() {
        Device::Cuda(dev) => {
            let q_c = ensure_contiguous(q)?;
            let k_c = ensure_contiguous(k)?;
            let v_c = ensure_contiguous(v)?;
            let g_c = ensure_contiguous(g)?;
            let beta_c = ensure_contiguous(beta)?;

            let (batch, heads, k_dim) = q.dims3()?;
            let v_dim = v.dim(2)?;

            let slots_ptr = get_cuda_const_ptr_i64(slots, dev)?;
            let stream = dev.cu_stream();
            let stream_ptr = *stream as *const _ as i64;

            if q.dtype() == DType::F32 {
                let out = Tensor::zeros((batch, heads, v_dim), DType::F32, q.device())?;
                let q_ptr = get_cuda_const_ptr(&q_c, dev)? as *const f32;
                let k_ptr = get_cuda_const_ptr(&k_c, dev)? as *const f32;
                let v_ptr = get_cuda_const_ptr(&v_c, dev)? as *const f32;
                let g_ptr = get_cuda_const_ptr(&g_c, dev)? as *const f32;
                let beta_ptr = get_cuda_const_ptr(&beta_c, dev)? as *const f32;
                let state_ptr = get_cuda_mut_ptr(state, dev)? as *mut f32;
                let out_ptr = get_cuda_mut_ptr(&out, dev)? as *mut f32;

                unsafe {
                    ffi::gated_delta_rule_decode_slots_f32(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        slots_ptr,
                        out_ptr,
                        batch as c_int,
                        heads as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        stream_ptr,
                    )
                }
                Ok(out)
            } else {
                let out = Tensor::zeros((batch, heads, v_dim), q.dtype(), q.device())?;
                let q_ptr = get_cuda_const_ptr(&q_c, dev)?;
                let k_ptr = get_cuda_const_ptr(&k_c, dev)?;
                let v_ptr = get_cuda_const_ptr(&v_c, dev)?;
                let g_ptr = get_cuda_const_ptr(&g_c, dev)?;
                let beta_ptr = get_cuda_const_ptr(&beta_c, dev)?;
                let state_ptr = get_cuda_mut_ptr(state, dev)? as *mut f32;
                let out_ptr = get_cuda_mut_ptr(&out, dev)?;

                match q.dtype() {
                    DType::F16 => unsafe {
                        ffi::gated_delta_rule_decode_slots_f16_state_f32(
                            q_ptr,
                            k_ptr,
                            v_ptr,
                            g_ptr,
                            beta_ptr,
                            state_ptr,
                            slots_ptr,
                            out_ptr as *mut c_void,
                            batch as c_int,
                            heads as c_int,
                            k_dim as c_int,
                            v_dim as c_int,
                            stream_ptr,
                        )
                    },
                    DType::BF16 => unsafe {
                        ffi::gated_delta_rule_decode_slots_bf16_state_f32(
                            q_ptr,
                            k_ptr,
                            v_ptr,
                            g_ptr,
                            beta_ptr,
                            state_ptr,
                            slots_ptr,
                            out_ptr as *mut c_void,
                            batch as c_int,
                            heads as c_int,
                            k_dim as c_int,
                            v_dim as c_int,
                            stream_ptr,
                        )
                    },
                    dt => candle_core::bail!("gated_delta_rule_decode_slots unsupported dtype: {:?}", dt),
                }
                Ok(out)
            }
        }
        _ => {
            // Naive fallback
            let slots_vec = slots.to_vec1::<i64>()?;
            let mut outs = Vec::with_capacity(slots_vec.len());
            for (b, &slot) in slots_vec.iter().enumerate() {
                let q_b = q.i(b)?;
                let k_b = k.i(b)?;
                let v_b = v.i(b)?;
                let g_b = g.i(b)?;
                let beta_b = beta.i(b)?;
                let mut state_b = state.i(slot as usize)?;

                let out_b = gated_delta_rule_recurrence_naive(
                    &q_b.unsqueeze(1)?,
                    &k_b.unsqueeze(1)?,
                    &v_b.unsqueeze(1)?,
                    &g_b.unsqueeze(1)?,
                    &beta_b.unsqueeze(1)?,
                    &mut state_b,
                )?.squeeze(1)?;

                *state = state.slice_assign(
                    &[slot as usize..slot as usize + 1, 0..state.dim(1)?, 0..state.dim(2)?, 0..state.dim(3)?],
                    &state_b.unsqueeze(0)?,
                )?;
                outs.push(out_b);
            }
            let refs = outs.iter().collect::<Vec<_>>();
            Tensor::stack(&refs, 0)
        }
    }
}

pub fn l2_norm_last_dim(input: &Tensor, eps: f64) -> Result<Tensor> {
    match input.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => {
            let input_c = ensure_contiguous(input)?;
            let shape = input_c.shape();
            let dim = shape.dims()[shape.rank() - 1];
            let rows = shape.elem_count() / dim;
            let output = Tensor::zeros(shape, input.dtype(), input.device())?;
            let in_ptr = get_cuda_const_ptr(&input_c, dev)?;
            let out_ptr = get_cuda_mut_ptr(&output, dev)?;
            let stream = dev.cu_stream();
            let stream_ptr = *stream as *const _ as i64;

            match input.dtype() {
                DType::F32 => unsafe {
                    ffi::l2_norm_last_dim_f32(in_ptr as *const f32, out_ptr as *mut f32, rows as c_int, dim as c_int, eps as f32, stream_ptr)
                },
                DType::F16 => unsafe {
                    ffi::l2_norm_last_dim_f16(in_ptr, out_ptr as *mut c_void, rows as c_int, dim as c_int, eps as f32, stream_ptr)
                },
                DType::BF16 => unsafe {
                    ffi::l2_norm_last_dim_bf16(in_ptr, out_ptr as *mut c_void, rows as c_int, dim as c_int, eps as f32, stream_ptr)
                },
                dt => candle_core::bail!("l2_norm_last_dim unsupported dtype {:?}", dt),
            }
            Ok(output)
        }
        _ => {
            let sumsq = input.sqr()?.sum_keepdim(input.rank() - 1)?;
            let norm = (sumsq + eps)?.sqrt()?;
            input.broadcast_div(&norm)
        }
    }
}

pub fn gated_delta_rule_recurrence_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    slots: &Tensor,
    cu_seqlens: &Tensor,
) -> Result<Tensor> {
    match q.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => {
            let q_c = ensure_contiguous(q)?;
            let k_c = ensure_contiguous(k)?;
            let v_c = ensure_contiguous(v)?;
            let g_c = ensure_contiguous(g)?;
            let beta_c = ensure_contiguous(beta)?;

            let (total_tokens, num_heads, k_dim) = q_c.dims3()?;
            let v_dim = v_c.dim(2)?;
            let batch = slots.dim(0)?;

            let out = Tensor::zeros((total_tokens, num_heads, v_dim), q.dtype(), q.device())?;

            let q_ptr = get_cuda_const_ptr(&q_c, dev)?;
            let k_ptr = get_cuda_const_ptr(&k_c, dev)?;
            let v_ptr = get_cuda_const_ptr(&v_c, dev)?;
            let g_ptr = get_cuda_const_ptr(&g_c, dev)?;
            let beta_ptr = get_cuda_const_ptr(&beta_c, dev)?;
            let state_ptr = get_cuda_mut_ptr(state, dev)? as *mut f32;
            let slots_ptr = get_cuda_const_ptr_i64(slots, dev)?;
            let cu_ptr = get_cuda_const_ptr_u32(cu_seqlens, dev)?;
            let out_ptr = get_cuda_mut_ptr(&out, dev)?;
            let stream = dev.cu_stream();
            let stream_ptr = *stream as *const _ as i64;

            match q.dtype() {
                DType::F32 => unsafe {
                    ffi::gated_delta_rule_recurrence_varlen_f32(
                        q_ptr as *const f32, k_ptr as *const f32, v_ptr as *const f32, g_ptr as *const f32, beta_ptr as *const f32,
                        state_ptr, slots_ptr, out_ptr as *mut f32, cu_ptr, batch as c_int, num_heads as c_int, k_dim as c_int, v_dim as c_int, stream_ptr
                    )
                },
                DType::F16 => unsafe {
                    ffi::gated_delta_rule_recurrence_varlen_f16(
                        q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, state_ptr, slots_ptr, out_ptr as *mut c_void, cu_ptr, batch as c_int, num_heads as c_int, k_dim as c_int, v_dim as c_int, stream_ptr
                    )
                },
                DType::BF16 => unsafe {
                    ffi::gated_delta_rule_recurrence_varlen_bf16(
                        q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, state_ptr, slots_ptr, out_ptr as *mut c_void, cu_ptr, batch as c_int, num_heads as c_int, k_dim as c_int, v_dim as c_int, stream_ptr
                    )
                },
                dt => candle_core::bail!("gated_delta_rule_recurrence_varlen unsupported dtype {:?}", dt),
            }
            Ok(out)
        }
        _ => gated_delta_rule_recurrence_varlen_naive(q, k, v, g, beta, state, slots, cu_seqlens),
    }
}

fn gated_delta_rule_recurrence_varlen_naive(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    slots: &Tensor,
    cu_seqlens: &Tensor,
) -> Result<Tensor> {
    let cu = cu_seqlens.to_vec1::<u32>()?;
    let slots_vec = slots.to_vec1::<i64>()?;
    let batch = slots_vec.len();
    let num_heads = q.dim(1)?;
    let v_dim = v.dim(2)?;
    let mut outputs = Vec::with_capacity(batch);

    for b in 0..batch {
        let start = cu[b] as usize;
        let end = cu[b+1] as usize;
        if start >= end { continue; }
        let seq_len = end - start;

        let q_b = q.narrow(0, start, seq_len)?.transpose(0, 1)?;
        let k_b = k.narrow(0, start, seq_len)?.transpose(0, 1)?;
        let v_b = v.narrow(0, start, seq_len)?.transpose(0, 1)?;
        let g_b = g.narrow(0, start, seq_len)?.transpose(0, 1)?;
        let beta_b = beta.narrow(0, start, seq_len)?.transpose(0, 1)?;

        let slot = slots_vec[b] as usize;
        let mut state_b = state.i(slot)?;

        let out_b = gated_delta_rule_recurrence_naive(&q_b, &k_b, &v_b, &g_b, &beta_b, &mut state_b)?;

        *state = state.slice_assign(
            &[slot..slot+1, 0..state.dim(1)?, 0..state.dim(2)?, 0..state.dim(3)?],
            &state_b.unsqueeze(0)?,
        )?;
        outputs.push(out_b.transpose(0, 1)?);
    }

    if outputs.is_empty() {
        return Tensor::zeros(q.shape(), q.dtype(), q.device());
    }
    let refs = outputs.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 0)
}

fn gated_delta_rule_recurrence_naive(
    q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor, state: &mut Tensor,
) -> Result<Tensor> {
    let (bh, seq_len, _k_dim) = q.dims3()?;
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let g = g.to_dtype(DType::F32)?;
    let beta = beta.to_dtype(DType::F32)?;
    let mut s = state.to_dtype(DType::F32)?;

    let mut outputs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let q_t = q.narrow(1, t, 1)?.squeeze(1)?;
        let k_t = k.narrow(1, t, 1)?.squeeze(1)?;
        let v_t = v.narrow(1, t, 1)?.squeeze(1)?;
        let g_t = g.narrow(1, t, 1)?.squeeze(1)?;
        let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?;

        let decay = g_t.exp()?.unsqueeze(1)?.unsqueeze(2)?;
        s = s.broadcast_mul(&decay)?;

        let k_exp = k_t.unsqueeze(2)?;
        let kv_mem = s.broadcast_mul(&k_exp)?.sum(1)?;
        let delta = v_t.broadcast_sub(&kv_mem)?.broadcast_mul(&beta_t.unsqueeze(1)?)?;
        s = (s + k_exp.broadcast_mul(&delta.unsqueeze(1)?)?)?;

        let y_t = s.broadcast_mul(&q_t.unsqueeze(2)?)?.sum(1)?;
        outputs.push(y_t.unsqueeze(1)?);
    }
    *state = s.to_dtype(state.dtype())?;
    let output_refs = outputs.iter().collect::<Vec<_>>();
    Tensor::cat(&output_refs, 1)?.to_dtype(q.dtype())
}

#[cfg(not(feature = "cuda"))]
pub fn causal_conv1d_fwd(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, conv_state: &mut Tensor, cu_seqlens: Option<&Tensor>, activation_silu: bool) -> Result<Tensor> {
    causal_conv1d_fwd_naive_with_state(x, weight, bias, conv_state, cu_seqlens, activation_silu)
}

#[cfg(not(feature = "cuda"))]
pub fn causal_conv1d_update(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, conv_state: &mut Tensor, activation_silu: bool) -> Result<Tensor> {
    causal_conv1d_update_naive(x, weight, bias, conv_state, activation_silu)
}

#[cfg(not(feature = "cuda"))]
pub fn fused_gdn_gating(a_log: &Tensor, a: &Tensor, b: &Tensor, dt_bias: &Tensor) -> Result<(Tensor, Tensor)> {
    fused_gdn_gating_naive(a_log, a, b, dt_bias)
}

#[cfg(not(feature = "cuda"))]
pub fn gated_rmsnorm_silu_mul(x: &Tensor, z: &Tensor, norm_weight: &Tensor, norm_bias: Option<&Tensor>, eps: f64, group_size: usize) -> Result<Tensor> {
    gated_rmsnorm_silu_mul_naive(x, z, norm_weight, norm_bias, eps, group_size)
}

#[cfg(not(feature = "cuda"))]
pub fn causal_conv1d_update_slots(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, conv_state: &mut Tensor, slots: &Tensor, activation_silu: bool) -> Result<Tensor> {
    let slots_vec = slots.to_vec1::<i64>()?;
    let mut gathered = Vec::with_capacity(slots_vec.len());
    for &s in &slots_vec { gathered.push(conv_state.i(s as usize)?); }
    let mut batch_state = Tensor::stack(&gathered.iter().collect::<Vec<_>>(), 0)?;
    let out = causal_conv1d_update_naive(x, weight, bias, &mut batch_state, activation_silu)?;
    for (i, &s) in slots_vec.iter().enumerate() {
        *conv_state = conv_state.slice_assign(&[s as usize..s as usize + 1, 0..conv_state.dim(1)?, 0..conv_state.dim(2)?], &batch_state.narrow(0, i, 1)?)?;
    }
    Ok(out)
}

#[cfg(not(feature = "cuda"))]
pub fn gated_delta_rule_recurrence(q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor, state: &mut Tensor) -> Result<Tensor> {
    gated_delta_rule_recurrence_naive(q, k, v, g, beta, state)
}

#[cfg(not(feature = "cuda"))]
pub fn gated_delta_rule_decode_slots(q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor, state: &mut Tensor, slots: &Tensor) -> Result<Tensor> {
    gated_delta_rule_decode_slots_naive(q, k, v, g, beta, state, slots)
}

fn causal_conv1d_fwd_naive_with_state(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, conv_state: &mut Tensor, cu_seqlens: Option<&Tensor>, activation_silu: bool) -> Result<Tensor> {
    if cu_seqlens.is_none() { return causal_conv1d_naive(x, weight, bias, activation_silu); }
    let weight_2d = weight.squeeze(1)?;
    let kernel_size = weight_2d.dim(1)?;
    let d_conv = weight_2d.dim(0)?;
    let batch_size = conv_state.dim(0)?;
    let cu = cu_seqlens.unwrap().to_vec1::<u32>()?;
    let mut outputs = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let start = cu[b] as usize;
        let end = cu[b+1] as usize;
        let seq_len = end.saturating_sub(start);
        let seq_x = x.narrow(0, start, seq_len)?;
        let history = conv_state.i(b)?.transpose(0, 1)?;
        let x_padded = Tensor::cat(&[&history, &seq_x], 0)?;
        let mut seq_out = x_padded.narrow(0, 0, seq_len)?.broadcast_mul(&weight_2d.i((.., 0))?)?;
        for k in 1..kernel_size { seq_out = (seq_out + x_padded.narrow(0, k, seq_len)?.broadcast_mul(&weight_2d.i((.., k))?)?)?; }
        if let Some(bias) = bias { seq_out = seq_out.broadcast_add(bias)?; }
        if activation_silu { seq_out = candle_nn::ops::silu(&seq_out)?; }
        outputs.push(seq_out);
        let next_history = x_padded.narrow(0, seq_len, kernel_size - 1)?.transpose(0, 1)?;
        *conv_state = conv_state.slice_assign(&[b..b + 1, 0..d_conv, 0..kernel_size - 1], &next_history.unsqueeze(0)?)?;
    }
    Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 0)
}

pub fn causal_conv1d_naive(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, activation_silu: bool) -> Result<Tensor> {
    let weight_2d = weight.squeeze(1)?;
    let kernel_size = weight_2d.dim(1)?;
    let d_conv = weight_2d.dim(0)?;
    let seq_len = x.dim(0)?;
    let padding = Tensor::zeros((kernel_size - 1, d_conv), x.dtype(), x.device())?;
    let x_padded = Tensor::cat(&[&padding, x], 0)?;
    let mut output = x_padded.narrow(0, 0, seq_len)?.broadcast_mul(&weight_2d.i((.., 0))?)?;
    for k in 1..kernel_size { output = (output + x_padded.narrow(0, k, seq_len)?.broadcast_mul(&weight_2d.i((.., k))?)?)?; }
    if let Some(bias) = bias { output = output.broadcast_add(bias)?; }
    if activation_silu { output = candle_nn::ops::silu(&output)?; }
    Ok(output)
}

pub fn causal_conv1d_update_naive(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, conv_state: &mut Tensor, activation_silu: bool) -> Result<Tensor> {
    let weight_2d = weight.squeeze(1)?;
    let kernel_size = weight_2d.dim(1)?;
    let x_expanded = x.unsqueeze(2)?;
    let full_window = Tensor::cat(&[&*conv_state, &x_expanded], 2)?;
    *conv_state = full_window.narrow(2, 1, kernel_size - 1)?;
    let mut output = full_window.broadcast_mul(&weight_2d.unsqueeze(0)?)?.sum(2)?;
    if let Some(bias) = bias { output = output.broadcast_add(bias)?; }
    if activation_silu { output = candle_nn::ops::silu(&output)?; }
    Ok(output)
}

pub fn fused_gdn_gating_naive(a_log: &Tensor, a: &Tensor, b: &Tensor, dt_bias: &Tensor) -> Result<(Tensor, Tensor)> {
    let a_dt = a.broadcast_add(dt_bias)?;
    let g = softplus(&a_dt)?.broadcast_mul(&a_log.exp()?.neg()?)?;
    let beta = candle_nn::ops::sigmoid(b)?;
    Ok((g, beta))
}

pub fn gated_rmsnorm_silu_mul_naive(x: &Tensor, z: &Tensor, norm_weight: &Tensor, norm_bias: Option<&Tensor>, eps: f64, group_size: usize) -> Result<Tensor> {
    let (rows, value_dim) = x.dims2()?;
    let groups = value_dim / group_size;
    let x_grouped = x.reshape((rows, groups, group_size))?;
    let variance = (&x_grouped * &x_grouped)?.mean_keepdim(2)?;
    let mut y = x_grouped.broadcast_div(&(variance + eps)?.sqrt()?)?;
    if norm_weight.dim(0)? == group_size {
        y = y.broadcast_mul(&norm_weight.reshape((1, 1, group_size))?)?;
        if let Some(b) = norm_bias { y = y.broadcast_add(&b.reshape((1, 1, group_size))?)?; }
    } else {
        y = y.broadcast_mul(&norm_weight.reshape((1, groups, group_size))?)?;
        if let Some(b) = norm_bias { y = y.broadcast_add(&b.reshape((1, groups, group_size))?)?; }
    }
    let y = y.reshape((rows, value_dim))?;
    (y * candle_nn::ops::silu(z)?)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    (x.exp()? + 1.0)?.log()
}

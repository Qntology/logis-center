//! Low-level CUDA kernel launchers for the full-layer attention pipeline.
//!
//! Each function wraps a single `CudaFunction` from [`CudaAttnModules`] with
//! the correct grid/block configuration and argument ordering. These are the
//! building blocks used by `encode_attn_phase` and friends in the parent
//! module.
//!
//! The cfg gate on the parent module (`native-cuda` + Linux/Windows) applies
//! here via module inclusion, so no additional `#[cfg(...)]` is needed.

use cudarc::driver::{CudaSlice, CudaView, LaunchConfig, PushKernelArg};

use super::super::cuda_graph::{CudaGraph, CudaGraphError};
use super::CudaAttnModules;

/// Launch `fused_qk_norm`.
///
/// Grid `(nq + nkv, 1, 1)`, block `(256, 1, 1)`.
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) unsafe fn launch_fused_qk_norm(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_q_in: &CudaSlice<f32>,
    d_k_in: &CudaSlice<f32>,
    d_q_out: &mut CudaSlice<f32>,
    d_k_out: &mut CudaSlice<f32>,
    d_q_weight: &CudaSlice<f32>,
    d_k_weight: &CudaSlice<f32>,
    nq: u32,
    nkv: u32,
    head_dim: u32,
    eps: f32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (nq + nkv, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_qk_norm)
        .arg(d_q_in)
        .arg(d_k_in)
        .arg(d_q_out)
        .arg(d_k_out)
        .arg(d_q_weight)
        .arg(d_k_weight)
        .arg(&nq)
        .arg(&nkv)
        .arg(&head_dim)
        .arg(&eps)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("fused_qk_norm launch: {e}")))
}

/// Launch `fused_qk_rope`.
///
/// Grid `(ceil(half_dim/64), nq + nkv, 1)`, block `(64, 1, 1)`.
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) unsafe fn launch_fused_qk_rope(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_q_in: &CudaSlice<f32>,
    d_k_in: &CudaSlice<f32>,
    d_q_out: &mut CudaSlice<f32>,
    d_k_out: &mut CudaSlice<f32>,
    d_cos: &CudaSlice<f32>,
    d_sin: &CudaSlice<f32>,
    nq: u32,
    nkv: u32,
    half_dim: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = half_dim.div_ceil(64);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, nq + nkv, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_qk_rope)
        .arg(d_q_in)
        .arg(d_k_in)
        .arg(d_q_out)
        .arg(d_k_out)
        .arg(d_cos)
        .arg(d_sin)
        .arg(&nq)
        .arg(&nkv)
        .arg(&half_dim)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("fused_qk_rope launch: {e}")))
}

/// Launch `fused_qk_norm_rope`.
///
/// Grid `(nq + nkv, 1, 1)`, block `(256, 1, 1)`.
///
/// `d_k_in_view` is a `CudaView` pointing at the K section of the QKV buffer.
///
/// # Safety
/// All slices/views must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_fused_qk_norm_rope(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_q_in: &CudaSlice<f32>,
    d_k_in_view: &CudaView<'_, f32>,
    d_q_out: &mut CudaSlice<f32>,
    d_k_out: &mut CudaSlice<f32>,
    d_q_weight: &CudaSlice<f32>,
    d_k_weight: &CudaSlice<f32>,
    d_cos: &CudaSlice<f32>,
    d_sin: &CudaSlice<f32>,
    nq: u32,
    nkv: u32,
    head_dim: u32,
    eps: f32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (nq + nkv, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_qk_norm_rope)
        .arg(d_q_in)
        .arg(d_k_in_view)
        .arg(d_q_out)
        .arg(d_k_out)
        .arg(d_q_weight)
        .arg(d_k_weight)
        .arg(d_cos)
        .arg(d_sin)
        .arg(&nq)
        .arg(&nkv)
        .arg(&head_dim)
        .arg(&eps)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("fused_qk_norm_rope launch: {e}")))
}

/// Launch `fused_kv_store`.
///
/// Grid `(ceil(head_dim/64), nkv, 1)`, block `(64, 1, 1)`.
///
/// `d_pos_seqlen[0]` = current position (read by the kernel from device memory).
///
/// # Safety
/// All slices/views must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_fused_kv_store(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_k_data: &CudaSlice<f32>,
    d_v_data_view: &CudaView<'_, f32>,
    d_k_cache: &mut CudaSlice<u16>,
    d_v_cache: &mut CudaSlice<u16>,
    head_dim: u32,
    nkv: u32,
    max_seq: u32,
    d_pos_seqlen: &CudaSlice<u32>,
    layer_offset: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = head_dim.div_ceil(64);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, nkv, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_kv_store)
        .arg(d_k_data)
        .arg(d_v_data_view)
        .arg(d_k_cache)
        .arg(d_v_cache)
        .arg(&head_dim)
        .arg(&nkv)
        .arg(&max_seq)
        .arg(d_pos_seqlen)
        .arg(&layer_offset)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("fused_kv_store launch: {e}")))
}

/// Launch `batched_attn_scores_v2`.
///
/// Grid `(n_q, max_seq / BATCH_STRIDE, 1)`, block `(128, 1, 1)`.
///
/// The grid Y dimension is fixed at `max_seq / BATCH_STRIDE` (not `seq_len`) so the
/// kernel sequence can be captured as a CUDA driver graph once and replayed for any
/// position.  Blocks with `pos_start >= seq_len` (read from `d_pos_seqlen[1]`) exit
/// immediately via the existing loop condition, adding only negligible overhead.
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_batched_attn_scores_v2(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_queries: &CudaSlice<f32>,
    d_k_cache: &CudaSlice<u16>,
    d_scores: &mut CudaSlice<f32>,
    head_dim: u32,
    n_q: u32,
    n_kv: u32,
    heads_per_group: u32,
    max_seq: u32,
    d_pos_seqlen: &CudaSlice<u32>,
    inv_sqrt_hd: f32,
    cache_layer_offset: u32,
) -> Result<(), CudaGraphError> {
    const BATCH_STRIDE: u32 = 4;
    // Fixed grid Y = max_seq / BATCH_STRIDE — constant across all decode positions,
    // allowing the kernel sequence to be captured as a replayable CUDA graph.
    let grid_y = max_seq.div_ceil(BATCH_STRIDE);
    let cfg = LaunchConfig {
        grid_dim: (n_q, grid_y, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.batched_attn_scores_v2)
        .arg(d_queries)
        .arg(d_k_cache)
        .arg(d_scores)
        .arg(&head_dim)
        .arg(&n_q)
        .arg(&n_kv)
        .arg(&heads_per_group)
        .arg(&max_seq)
        .arg(d_pos_seqlen)
        .arg(&inv_sqrt_hd)
        .arg(&cache_layer_offset)
        .arg(&BATCH_STRIDE)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("batched_attn_scores_v2 launch: {e}")))
}

/// Launch `batched_softmax`.
///
/// Grid `(n_q, 1, 1)`, block `(256, 1, 1)`.
///
/// `d_pos_seqlen[1]` = seq_len (read by the kernel from device memory).
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
pub(super) unsafe fn launch_batched_softmax(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_scores: &mut CudaSlice<f32>,
    n_q: u32,
    max_seq: u32,
    d_pos_seqlen: &CudaSlice<u32>,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_q, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.batched_softmax)
        .arg(d_scores)
        .arg(&n_q)
        .arg(&max_seq)
        .arg(d_pos_seqlen)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("batched_softmax launch: {e}")))
}

/// Launch `batched_attn_weighted_sum`.
///
/// Grid `(ceil(head_dim/64), n_q, 1)`, block `(64, 1, 1)`.
///
/// `d_pos_seqlen[1]` = seq_len (read by the kernel from device memory).
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_batched_attn_weighted_sum(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_scores: &CudaSlice<f32>,
    d_v_cache: &CudaSlice<u16>,
    d_attn_out: &mut CudaSlice<f32>,
    head_dim: u32,
    n_q: u32,
    n_kv: u32,
    heads_per_group: u32,
    max_seq: u32,
    d_pos_seqlen: &CudaSlice<u32>,
    cache_layer_offset: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = head_dim.div_ceil(64);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, n_q, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.batched_attn_weighted_sum)
        .arg(d_scores)
        .arg(d_v_cache)
        .arg(d_attn_out)
        .arg(&head_dim)
        .arg(&n_q)
        .arg(&n_kv)
        .arg(&heads_per_group)
        .arg(&max_seq)
        .arg(d_pos_seqlen)
        .arg(&cache_layer_offset)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("batched_attn_weighted_sum launch: {e}")))
}

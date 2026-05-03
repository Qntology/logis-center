//! Prefill (batch) GPU dispatch for OxiBonsai.
//!
//! Extracted from `metal_full_layer.rs` for maintainability.
//! Handles batch processing of multiple tokens during prompt prefill.

use metal::{Buffer, MTLResourceOptions, MTLSize};
use std::sync::Arc;

use super::metal_full_layer::{FullForwardLayerParams, GpuKvCache};
use super::metal_graph::{
    alloc_buf, div_ceil, download_f32, set_scalar, upload_f32, MetalGraph, MetalGraphError,
    MetalWeightHandle,
};

// ═══════════════════════════════════════════════════════════════════════════
// Prefill buffers (batch processing)
// ═══════════════════════════════════════════════════════════════════════════

/// Intermediate buffers for batch prefill (processing multiple tokens at once).
///
/// Batched buffers use column-major layout: `buf[col * dim + element]` where
/// `col` is the batch (token) index and `dim` is the vector dimension.
///
/// Single-token buffers are used inside the sequential attention loop where
/// we process one query position at a time with the existing attention kernels.
pub(crate) struct PrefillBuffers {
    /// Batched hidden states: `[batch_size × hidden_size]` f32 (column-major).
    pub hidden_buf: Buffer,
    /// Batched normed hidden: `[batch_size × hidden_size]` f32 (column-major).
    pub normed_buf: Buffer,
    /// Batched QKV projection output: `[batch_size × qkv_dim]` f32 (column-major).
    pub qkv_buf: Buffer,
    /// Batched attention output: `[batch_size × nq*head_dim]` f32 (column-major).
    pub attn_out_buf: Buffer,
    /// Batched SwiGLU output: `[batch_size × intermediate_size]` f32 (column-major).
    pub swiglu_buf: Buffer,
    /// Single-token Q after norm: `[nq * head_dim]` f32.
    pub q_normed_buf: Buffer,
    /// Single-token K after norm: `[nkv * head_dim]` f32.
    pub k_normed_buf: Buffer,
    /// Single-token Q after RoPE: `[nq * head_dim]` f32.
    pub q_rope_buf: Buffer,
    /// Single-token K after RoPE: `[nkv * head_dim]` f32.
    pub k_rope_buf: Buffer,
    /// Single-token attention scores: `[nq × max_seq]` f32.
    pub scores_buf: Buffer,
    /// RoPE cos table: `[batch_size × half_dim]` f32.
    pub cos_buf: Buffer,
    /// RoPE sin table: `[batch_size × half_dim]` f32.
    pub sin_buf: Buffer,
    /// Cached dimensions.
    batch_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    max_seq: usize,
}

impl PrefillBuffers {
    /// Allocate all prefill buffers for the given dimensions and batch size.
    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        device: &metal::Device,
        batch_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<Self, MetalGraphError> {
        let f = std::mem::size_of::<f32>();
        let shared = MTLResourceOptions::StorageModeShared;
        let private = MTLResourceOptions::StorageModePrivate;
        let qkv_dim = nq * head_dim + 2 * nkv * head_dim;

        Ok(Self {
            hidden_buf: alloc_buf(device, (batch_size * hidden_size * f) as u64, shared)?, // CPU upload
            normed_buf: alloc_buf(device, (batch_size * hidden_size * f) as u64, private)?, // GPU-only
            qkv_buf: alloc_buf(device, (batch_size * qkv_dim * f) as u64, private)?, // GPU-only
            attn_out_buf: alloc_buf(device, (batch_size * nq * head_dim * f) as u64, private)?, // GPU-only
            swiglu_buf: alloc_buf(device, (batch_size * intermediate_size * f) as u64, private)?, // GPU-only
            q_normed_buf: alloc_buf(device, (nq * head_dim * f) as u64, private)?, // GPU-only
            k_normed_buf: alloc_buf(device, (nkv * head_dim * f) as u64, private)?, // GPU-only
            q_rope_buf: alloc_buf(device, (nq * head_dim * f) as u64, private)?,   // GPU-only
            k_rope_buf: alloc_buf(device, (nkv * head_dim * f) as u64, private)?,  // GPU-only
            scores_buf: alloc_buf(device, (nq * max_seq * f) as u64, private)?,    // GPU-only
            cos_buf: alloc_buf(device, (batch_size * (head_dim / 2) * f) as u64, shared)?, // CPU upload
            sin_buf: alloc_buf(device, (batch_size * (head_dim / 2) * f) as u64, shared)?, // CPU upload
            batch_size,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq,
        })
    }

    /// Check whether existing buffers match the requested dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn matches(
        &self,
        batch_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> bool {
        self.batch_size == batch_size
            && self.hidden_size == hidden_size
            && self.intermediate_size == intermediate_size
            && self.nq == nq
            && self.nkv == nkv
            && self.head_dim == head_dim
            && self.max_seq == max_seq
    }
}

/// References to a single layer's weight buffers for prefill encoding.
pub(crate) struct LayerWeightRefs<'a> {
    pub attn_norm: &'a Buffer,
    pub qkv: &'a Buffer,
    pub q_norm: &'a Buffer,
    pub k_norm: &'a Buffer,
    pub output_proj: &'a Buffer,
    pub ffn_norm: &'a Buffer,
    pub gate_up: &'a Buffer,
    pub down: &'a Buffer,
}

/// Model configuration for a single transformer layer.
pub(crate) struct LayerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub max_seq_len: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// MetalGraph extensions for prefill dispatch
// ═══════════════════════════════════════════════════════════════════════════

impl MetalGraph {
    /// Acquire the prefill buffer set, allocating if needed.
    #[allow(clippy::too_many_arguments)]
    fn acquire_prefill_buffers(
        &self,
        batch_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<PrefillBuffers>>, MetalGraphError> {
        let mut guard = self.prefill_buffers.lock().map_err(|_| {
            MetalGraphError::ExecutionFailed("prefill_buffers lock poisoned".into())
        })?;

        let needs_alloc = match guard.as_ref() {
            Some(b) => !b.matches(
                batch_size,
                hidden_size,
                intermediate_size,
                nq,
                nkv,
                head_dim,
                max_seq,
            ),
            None => true,
        };

        if needs_alloc {
            *guard = Some(PrefillBuffers::allocate(
                &self.device,
                batch_size,
                hidden_size,
                intermediate_size,
                nq,
                nkv,
                head_dim,
                max_seq,
            )?);
        }

        Ok(guard)
    }

    /// Encode ALL transformer layers for batch prefill in a SINGLE command buffer.
    ///
    /// Like `encode_full_forward`, except:
    /// - `hidden_batch` is `batch_size × hidden_size` (all prompt tokens)
    /// - `cos_table`/`sin_table` are `batch_size × half_dim` (all positions' RoPE)
    /// - Each layer uses GEMM (not GEMV) for batched projections
    /// - Sequential attention per token within each layer
    /// - After all layers, only the LAST token feeds into final norm + LM head
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn encode_full_forward_prefill(
        &self,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[(
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
        )],
        cos_table: &[f32],
        sin_table: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        logits_out: Option<&mut Vec<f32>>,
        greedy_token_id_out: Option<&mut u32>,
    ) -> Result<(), MetalGraphError> {
        let half_dim = head_dim / 2;
        let f = std::mem::size_of::<f32>();

        // ── Validate inputs ──────────────────────────────────────────────
        if hidden_batch.len() < batch_size * hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "hidden_batch too short: need {}, got {}",
                batch_size * hidden_size,
                hidden_batch.len()
            )));
        }
        if cos_table.len() < batch_size * half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "cos_table too short: need {}, got {}",
                batch_size * half_dim,
                cos_table.len()
            )));
        }
        if sin_table.len() < batch_size * half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "sin_table too short: need {}, got {}",
                batch_size * half_dim,
                sin_table.len()
            )));
        }
        if layer_weights.len() != n_layers {
            return Err(MetalGraphError::EncodingFailed(format!(
                "layer_weights length mismatch: need {n_layers}, got {}",
                layer_weights.len()
            )));
        }

        // ── Acquire buffers ──────────────────────────────────────────────
        let pb_guard = self.acquire_prefill_buffers(
            batch_size,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq_len,
        )?;
        let bufs = pb_guard.as_ref().ok_or_else(|| {
            MetalGraphError::ExecutionFailed("prefill_buffers not allocated".into())
        })?;

        let kv_guard = self.acquire_kv_cache(n_layers, nkv, max_seq_len, head_dim)?;
        let kv = kv_guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("kv_cache not allocated".into()))?;

        // ── Upload host data ─────────────────────────────────────────────
        unsafe {
            upload_f32(&bufs.hidden_buf, &hidden_batch[..batch_size * hidden_size]);
            upload_f32(&bufs.cos_buf, &cos_table[..batch_size * half_dim]);
            upload_f32(&bufs.sin_buf, &sin_table[..batch_size * half_dim]);
        }

        let config = LayerConfig {
            hidden_size,
            intermediate_size,
            n_q_heads: nq,
            n_kv_heads: nkv,
            head_dim,
            eps,
            max_seq_len,
        };

        // ── Create command buffer + single compute encoder ───────────────
        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        // ── Encode all layers ────────────────────────────────────────────
        for (layer_idx, weights) in layer_weights.iter().enumerate() {
            let layer_refs = LayerWeightRefs {
                attn_norm: &weights.0.buffer,
                qkv: &weights.1.buffer,
                q_norm: &weights.2.buffer,
                k_norm: &weights.3.buffer,
                output_proj: &weights.4.buffer,
                ffn_norm: &weights.5.buffer,
                gate_up: &weights.6.buffer,
                down: &weights.7.buffer,
            };
            self.encode_layer_prefill(
                encoder,
                bufs,
                kv,
                &layer_refs,
                layer_idx,
                batch_size,
                pos_start,
                &config,
            )?;
        }

        // ── Tail: final norm on last token + LM head ─────────────────────
        match (final_norm_w, lm_head_w) {
            (Some(fnorm_w), Some(lm_w)) if lm_head_out_features > 0 => {
                let h = hidden_size as u32;
                let last_token_offset = ((batch_size - 1) * hidden_size * f) as u64;

                // Final RMSNorm: read last token from hidden_buf → normed_buf[0..h]
                encoder.set_compute_pipeline_state(&self.pipelines.rmsnorm_weighted_v2);
                encoder.set_buffer(0, Some(&bufs.hidden_buf), last_token_offset);
                encoder.set_buffer(1, Some(&fnorm_w.buffer), 0);
                encoder.set_buffer(2, Some(&bufs.normed_buf), 0);
                unsafe {
                    set_scalar(encoder, 3, &final_norm_eps);
                    set_scalar(encoder, 4, &h);
                }
                encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));

                // Ensure logits buffer is allocated
                let mut lg = self.logits_buf.lock().map_err(|_| {
                    MetalGraphError::ExecutionFailed("logits_buf lock poisoned".into())
                })?;
                let needed_bytes = (lm_head_out_features * f) as u64;
                if lg.as_ref().is_none_or(|b| b.length() < needed_bytes) {
                    *lg = Some(alloc_buf(
                        &self.device,
                        needed_bytes,
                        MTLResourceOptions::StorageModeShared,
                    )?);
                }
                let logits_buf = lg.as_ref().ok_or(MetalGraphError::BufferCreationFailed)?;

                // LM head GEMV (single token, reads from normed_buf[0..h])
                self.dispatch_gemv_q1(
                    encoder,
                    &lm_w.buffer,
                    &bufs.normed_buf,
                    logits_buf,
                    lm_head_out_features as u32,
                    h,
                );

                // Greedy decoding: argmax on GPU
                if greedy_token_id_out.is_some() {
                    let mut tid_guard = self.token_id_buf.lock().map_err(|_| {
                        MetalGraphError::ExecutionFailed("token_id_buf lock poisoned".into())
                    })?;
                    let needed = std::mem::size_of::<u32>() as u64;
                    if tid_guard.as_ref().is_none_or(|b| b.length() < needed) {
                        *tid_guard = Some(alloc_buf(
                            &self.device,
                            needed,
                            MTLResourceOptions::StorageModeShared,
                        )?);
                    }
                    let token_id_buf_ref = tid_guard
                        .as_ref()
                        .ok_or(MetalGraphError::BufferCreationFailed)?;

                    self.dispatch_argmax(
                        encoder,
                        logits_buf,
                        token_id_buf_ref,
                        lm_head_out_features as u32,
                    );

                    encoder.end_encoding();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();

                    let token_id = unsafe { *(token_id_buf_ref.contents() as *const u32) };
                    if let Some(out) = greedy_token_id_out {
                        *out = token_id;
                    }
                } else {
                    encoder.end_encoding();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();

                    if let Some(out) = logits_out {
                        out.resize(lm_head_out_features, 0.0);
                        unsafe { download_f32(logits_buf, out) };
                    }
                }
            }
            _ => {
                encoder.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
            }
        }

        Ok(())
    }

    /// Encode full-forward prefill for **verification** (speculative decoding).
    ///
    /// Identical to `encode_full_forward_prefill` for the layer loop, but the
    /// tail runs final RMSNorm + LM-head GEMM on **all** batch positions and
    /// returns per-position argmax token IDs instead of single-token logits.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn encode_full_forward_prefill_verify(
        &self,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[(
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
            &Arc<MetalWeightHandle>,
        )],
        cos_table: &[f32],
        sin_table: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        batch_token_ids_out: &mut Vec<u32>,
    ) -> Result<(), MetalGraphError> {
        let half_dim = head_dim / 2;
        let f = std::mem::size_of::<f32>();

        // ── Validate inputs ──────────────────────────────────────────────
        if hidden_batch.len() < batch_size * hidden_size {
            return Err(MetalGraphError::EncodingFailed(format!(
                "hidden_batch too short: need {}, got {}",
                batch_size * hidden_size,
                hidden_batch.len()
            )));
        }
        if cos_table.len() < batch_size * half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "cos_table too short: need {}, got {}",
                batch_size * half_dim,
                cos_table.len()
            )));
        }
        if sin_table.len() < batch_size * half_dim {
            return Err(MetalGraphError::EncodingFailed(format!(
                "sin_table too short: need {}, got {}",
                batch_size * half_dim,
                sin_table.len()
            )));
        }
        if layer_weights.len() != n_layers {
            return Err(MetalGraphError::EncodingFailed(format!(
                "layer_weights length mismatch: need {n_layers}, got {}",
                layer_weights.len()
            )));
        }

        // ── Acquire buffers ──────────────────────────────────────────────
        let pb_guard = self.acquire_prefill_buffers(
            batch_size,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq_len,
        )?;
        let bufs = pb_guard.as_ref().ok_or_else(|| {
            MetalGraphError::ExecutionFailed("prefill_buffers not allocated".into())
        })?;

        let kv_guard = self.acquire_kv_cache(n_layers, nkv, max_seq_len, head_dim)?;
        let kv = kv_guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("kv_cache not allocated".into()))?;

        // ── Upload host data ─────────────────────────────────────────────
        unsafe {
            upload_f32(&bufs.hidden_buf, &hidden_batch[..batch_size * hidden_size]);
            upload_f32(&bufs.cos_buf, &cos_table[..batch_size * half_dim]);
            upload_f32(&bufs.sin_buf, &sin_table[..batch_size * half_dim]);
        }

        let config = LayerConfig {
            hidden_size,
            intermediate_size,
            n_q_heads: nq,
            n_kv_heads: nkv,
            head_dim,
            eps,
            max_seq_len,
        };

        // ── Create command buffer + single compute encoder ───────────────
        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        // ── Encode all layers ────────────────────────────────────────────
        for (layer_idx, weights) in layer_weights.iter().enumerate() {
            let layer_refs = LayerWeightRefs {
                attn_norm: &weights.0.buffer,
                qkv: &weights.1.buffer,
                q_norm: &weights.2.buffer,
                k_norm: &weights.3.buffer,
                output_proj: &weights.4.buffer,
                ffn_norm: &weights.5.buffer,
                gate_up: &weights.6.buffer,
                down: &weights.7.buffer,
            };
            self.encode_layer_prefill(
                encoder,
                bufs,
                kv,
                &layer_refs,
                layer_idx,
                batch_size,
                pos_start,
                &config,
            )?;
        }

        // ── Tail: final norm on ALL tokens + LM head + batch argmax ─────
        match (final_norm_w, lm_head_w) {
            (Some(fnorm_w), Some(lm_w)) if lm_head_out_features > 0 => {
                let h = hidden_size as u32;

                // Batched RMSNorm on all positions:
                //   hidden_buf[0..batch_size*h] → normed_buf[0..batch_size*h]
                self.dispatch_batched_rmsnorm(
                    encoder,
                    &bufs.hidden_buf,
                    &fnorm_w.buffer,
                    &bufs.normed_buf,
                    final_norm_eps,
                    h,
                    batch_size as u32,
                );

                // Ensure logits buffer is allocated for ALL positions
                let mut lg = self.logits_buf.lock().map_err(|_| {
                    MetalGraphError::ExecutionFailed("logits_buf lock poisoned".into())
                })?;
                let needed_bytes = (batch_size * lm_head_out_features * f) as u64;
                if lg.as_ref().is_none_or(|b| b.length() < needed_bytes) {
                    *lg = Some(alloc_buf(
                        &self.device,
                        needed_bytes,
                        MTLResourceOptions::StorageModeShared,
                    )?);
                }
                let logits_buf = lg.as_ref().ok_or(MetalGraphError::BufferCreationFailed)?;

                // LM head GEMM: all positions at once
                //   normed_buf → logits_buf  (batch_size columns)
                self.dispatch_gemm_q1_v7(
                    encoder,
                    &lm_w.buffer,
                    &bufs.normed_buf,
                    logits_buf,
                    lm_head_out_features as u32,
                    h,
                    batch_size as u32,
                );

                // Allocate token_ids buffer for batch_size u32s
                let mut tid_guard = self.token_id_buf.lock().map_err(|_| {
                    MetalGraphError::ExecutionFailed("token_id_buf lock poisoned".into())
                })?;
                let needed = (batch_size * std::mem::size_of::<u32>()) as u64;
                if tid_guard.as_ref().is_none_or(|b| b.length() < needed) {
                    *tid_guard = Some(alloc_buf(
                        &self.device,
                        needed,
                        MTLResourceOptions::StorageModeShared,
                    )?);
                }
                let token_id_buf_ref = tid_guard
                    .as_ref()
                    .ok_or(MetalGraphError::BufferCreationFailed)?;

                // Per-position argmax: dispatch K separate argmax kernels
                // Each reads from logits_buf at offset col * vocab * sizeof(f32)
                // and writes to token_id_buf at offset col * sizeof(u32)
                let vocab = lm_head_out_features as u32;
                let f32_size = std::mem::size_of::<f32>() as u64;
                let u32_size = std::mem::size_of::<u32>() as u64;
                for col in 0..batch_size {
                    let logit_offset = col as u64 * vocab as u64 * f32_size;
                    let tid_offset = col as u64 * u32_size;

                    encoder.set_compute_pipeline_state(&self.pipelines.argmax);
                    encoder.set_buffer(0, Some(logits_buf), logit_offset);
                    encoder.set_buffer(1, Some(token_id_buf_ref), tid_offset);
                    unsafe {
                        set_scalar(encoder, 2, &vocab);
                    }
                    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
                }

                encoder.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();

                // Download all token IDs
                batch_token_ids_out.clear();
                batch_token_ids_out.reserve(batch_size);
                unsafe {
                    let ptr = token_id_buf_ref.contents() as *const u32;
                    for col in 0..batch_size {
                        batch_token_ids_out.push(*ptr.add(col));
                    }
                }
            }
            _ => {
                encoder.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
            }
        }

        Ok(())
    }

    /// Encode a single transformer layer for batch prefill (multiple tokens).
    ///
    /// Non-attention operations (RMSNorm, QKV projection, FFN) use batched GEMM
    /// kernels to process all tokens in parallel. Attention is processed
    /// sequentially per token using existing single-token kernels, since each
    /// query position needs access to all prior KV entries up to its position.
    ///
    /// The hidden state is read from and written to `bufs.hidden_buf` in-place
    /// via residual-add GEMM variants.
    #[allow(clippy::too_many_arguments)]
    fn encode_layer_prefill(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        bufs: &PrefillBuffers,
        kv: &GpuKvCache,
        layer_weights: &LayerWeightRefs<'_>,
        layer_idx: usize,
        batch_size: usize,
        pos_start: usize,
        config: &LayerConfig,
    ) -> Result<(), MetalGraphError> {
        let h = config.hidden_size;
        let nq = config.n_q_heads;
        let nkv = config.n_kv_heads;
        let hd = config.head_dim;
        let inter = config.intermediate_size;
        let qkv_out = (nq + 2 * nkv) * hd;
        let half_dim = hd / 2;
        let bs = batch_size as u32;

        let inv_sqrt_hd = 1.0f32 / (hd as f32).sqrt();
        let heads_per_group = (nq / nkv) as u32;
        let cache_layer_offset = kv.layer_offset_elements(layer_idx);

        // ══════════════════════════════════════════════════════════════
        // 1. Batched RMSNorm (all tokens)
        // ══════════════════════════════════════════════════════════════
        self.dispatch_batched_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            layer_weights.attn_norm,
            &bufs.normed_buf,
            config.eps,
            h as u32,
            bs,
        );

        // ══════════════════════════════════════════════════════════════
        // 2. QKV GEMM for all tokens at once
        // ══════════════════════════════════════════════════════════════
        self.dispatch_gemm_q1_v7(
            encoder,
            layer_weights.qkv,
            &bufs.normed_buf,
            &bufs.qkv_buf,
            qkv_out as u32,
            h as u32,
            bs,
        );

        // ══════════════════════════════════════════════════════════════
        // 3. Sequential attention for each token
        // ══════════════════════════════════════════════════════════════
        //
        // For each token t at position (pos_start + t), we:
        //   a) Extract Q, K from the batched qkv_buf (column-major offsets)
        //   b) Apply QK norm
        //   c) Apply RoPE
        //   d) Store K, V into the KV cache
        //   e) Compute attention scores, softmax, weighted sum
        //
        // The existing single-token attention kernels read from buffer(0)
        // at offset 0. We use set_buffer with byte offsets to point into
        // the correct column of the batched qkv_buf / attn_out_buf.
        let f = std::mem::size_of::<f32>();

        for t in 0..batch_size {
            let pos = pos_start + t;
            let seq_len = (pos + 1) as u32;

            // Column-major offset for token t's QKV data
            let qkv_col_byte_offset = (t * qkv_out * f) as u64;
            let q_byte_offset = qkv_col_byte_offset;
            let k_byte_offset = qkv_col_byte_offset + (nq * hd * f) as u64;
            let v_byte_offset = qkv_col_byte_offset + ((nq + nkv) * hd * f) as u64;

            // Fused QK-norm: normalise Q and K heads for this token
            self.dispatch_fused_qk_norm(
                encoder,
                &bufs.qkv_buf,
                q_byte_offset,
                &bufs.qkv_buf,
                k_byte_offset,
                &bufs.q_normed_buf,
                &bufs.k_normed_buf,
                layer_weights.q_norm,
                layer_weights.k_norm,
                nq as u32,
                nkv as u32,
                hd as u32,
                config.eps,
            );

            // We need cos/sin for this token's position.
            // The cos_buf/sin_buf in PrefillBuffers contain pre-computed RoPE
            // values for all positions. Each token t maps to byte offset
            // t * half_dim * sizeof(f32).
            let rope_byte_offset = (t * half_dim * f) as u64;

            // Fused QK-RoPE for this token (inline dispatch with buffer offset)
            {
                encoder.set_compute_pipeline_state(&self.pipelines.fused_qk_rope);
                encoder.set_buffer(0, Some(&bufs.q_normed_buf), 0);
                encoder.set_buffer(1, Some(&bufs.k_normed_buf), 0);
                encoder.set_buffer(2, Some(&bufs.q_rope_buf), 0);
                encoder.set_buffer(3, Some(&bufs.k_rope_buf), 0);
                encoder.set_buffer(4, Some(&bufs.cos_buf), rope_byte_offset);
                encoder.set_buffer(5, Some(&bufs.sin_buf), rope_byte_offset);
                unsafe {
                    set_scalar(encoder, 6, &(nq as u32));
                    set_scalar(encoder, 7, &(nkv as u32));
                    set_scalar(encoder, 8, &(half_dim as u32));
                }
                let tg_x = div_ceil(half_dim, 64) as u64;
                encoder.dispatch_thread_groups(
                    MTLSize::new(tg_x, (nq + nkv) as u64, 1),
                    MTLSize::new(64, 1, 1),
                );
            }

            // Fused KV-Store: copy K and V into the cache at position `pos`
            self.dispatch_fused_kv_store(
                encoder,
                &bufs.k_rope_buf,
                &bufs.qkv_buf,
                v_byte_offset,
                &kv.k_cache,
                &kv.v_cache,
                nkv as u32,
                hd as u32,
                config.max_seq_len as u32,
                pos as u32,
                cache_layer_offset,
            );

            // Attention scores V2: 128-thread TGs with position batching
            {
                self.dispatch_attention_scores_v2(
                    encoder,
                    &bufs.q_rope_buf,
                    &kv.k_cache,
                    &bufs.scores_buf,
                    hd as u32,
                    nq as u32,
                    nkv as u32,
                    heads_per_group,
                    config.max_seq_len as u32,
                    seq_len,
                    inv_sqrt_hd,
                    cache_layer_offset,
                );
            }

            // Attention softmax
            {
                encoder.set_compute_pipeline_state(&self.pipelines.batched_softmax);
                encoder.set_buffer(0, Some(&bufs.scores_buf), 0);
                unsafe {
                    set_scalar(encoder, 1, &(nq as u32));
                    set_scalar(encoder, 2, &(config.max_seq_len as u32));
                    set_scalar(encoder, 3, &seq_len);
                }
                encoder
                    .dispatch_thread_groups(MTLSize::new(nq as u64, 1, 1), MTLSize::new(256, 1, 1));
            }

            // Attention weighted sum → write to the correct column of attn_out_buf
            {
                let attn_col_byte_offset = (t * nq * hd * f) as u64;
                encoder.set_compute_pipeline_state(&self.pipelines.batched_attention_weighted_sum);
                encoder.set_buffer(0, Some(&bufs.scores_buf), 0);
                encoder.set_buffer(1, Some(&kv.v_cache), 0);
                encoder.set_buffer(2, Some(&bufs.attn_out_buf), attn_col_byte_offset);
                unsafe {
                    set_scalar(encoder, 3, &(hd as u32));
                    set_scalar(encoder, 4, &(nq as u32));
                    set_scalar(encoder, 5, &(nkv as u32));
                    set_scalar(encoder, 6, &heads_per_group);
                    set_scalar(encoder, 7, &(config.max_seq_len as u32));
                    set_scalar(encoder, 8, &seq_len);
                    set_scalar(encoder, 9, &cache_layer_offset);
                }
                let tg_x = div_ceil(hd, 64) as u64;
                encoder.dispatch_thread_groups(
                    MTLSize::new(tg_x, nq as u64, 1),
                    MTLSize::new(64, 1, 1),
                );
            }
        }

        // ══════════════════════════════════════════════════════════════
        // 4. Output proj GEMM + residual for all tokens
        // ══════════════════════════════════════════════════════════════
        self.dispatch_gemm_q1_v7_residual(
            encoder,
            layer_weights.output_proj,
            &bufs.attn_out_buf,
            &bufs.hidden_buf,
            h as u32,
            (nq * hd) as u32,
            bs,
            &bufs.hidden_buf,
        );

        // ══════════════════════════════════════════════════════════════
        // 5. Batched FFN RMSNorm
        // ══════════════════════════════════════════════════════════════
        self.dispatch_batched_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            layer_weights.ffn_norm,
            &bufs.normed_buf,
            config.eps,
            h as u32,
            bs,
        );

        // ══════════════════════════════════════════════════════════════
        // 6. Fused gate+up+SwiGLU GEMM (all tokens)
        // ══════════════════════════════════════════════════════════════
        self.dispatch_fused_gate_up_swiglu_gemm(
            encoder,
            layer_weights.gate_up,
            &bufs.normed_buf,
            &bufs.swiglu_buf,
            inter as u32,
            h as u32,
            bs,
        );

        // ══════════════════════════════════════════════════════════════
        // 7. Down GEMM + residual (all tokens)
        // ══════════════════════════════════════════════════════════════
        self.dispatch_gemm_q1_v7_residual(
            encoder,
            layer_weights.down,
            &bufs.swiglu_buf,
            &bufs.hidden_buf,
            h as u32,
            inter as u32,
            bs,
            &bufs.hidden_buf,
        );

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public prefill entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Attempt to run batch prefill (ALL transformer layers + LM head) in a
/// single Metal command buffer for multiple prompt tokens.
///
/// Like `try_metal_full_forward`, but processes `batch_size` tokens at once
/// using GEMM instead of GEMV for projections, with sequential per-token
/// attention within each layer. Only the last token's logits are returned.
///
/// Returns `Ok(())` on success. Returns `Err(...)` if Metal is unavailable
/// or any dispatch step fails.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_prefill(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[FullForwardLayerParams<'_>],
    cos_table: &[f32],
    sin_table: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    eps: f32,
    max_seq_len: usize,
    final_norm_handle: Option<u64>,
    final_norm_bytes: Option<&[f32]>,
    final_norm_eps: f32,
    lm_head_handle: Option<u64>,
    lm_head_bytes: Option<&[u8]>,
    lm_head_out_features: usize,
    logits_out: Option<&mut Vec<f32>>,
    greedy_token_id_out: Option<&mut u32>,
) -> Result<(), MetalGraphError> {
    if layer_params.len() != n_layers {
        return Err(MetalGraphError::EncodingFailed(format!(
            "layer_params length mismatch: need {n_layers}, got {}",
            layer_params.len()
        )));
    }

    let graph = MetalGraph::global()?;

    // Upload/cache all per-layer weights (same as try_metal_full_forward)
    #[allow(clippy::type_complexity)]
    let mut layer_weights: Vec<(
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
    )> = Vec::with_capacity(n_layers);

    for lp in layer_params {
        let attn_norm_w =
            graph.get_or_upload_f32_weight(lp.attn_norm_handle, lp.attn_norm_bytes)?;
        let q_norm_w = graph.get_or_upload_f32_weight(lp.q_norm_handle, lp.q_norm_bytes)?;
        let k_norm_w = graph.get_or_upload_f32_weight(lp.k_norm_handle, lp.k_norm_bytes)?;
        let ffn_norm_w = graph.get_or_upload_f32_weight(lp.ffn_norm_handle, lp.ffn_norm_bytes)?;

        let fused_qkv_w =
            graph.get_or_upload_q1_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)?;
        let attn_proj_w =
            graph.get_or_upload_q1_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)?;

        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_w = graph.get_or_upload_q1_weight_soa_lazy(lp.gate_up_handle, || {
            let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            fused.extend_from_slice(gate_bytes);
            fused.extend_from_slice(up_bytes);
            fused
        })?;

        let down_w = graph.get_or_upload_q1_weight_soa(lp.down_handle, lp.down_bytes)?;

        layer_weights.push((
            attn_norm_w,
            fused_qkv_w,
            q_norm_w,
            k_norm_w,
            attn_proj_w,
            ffn_norm_w,
            gate_up_w,
            down_w,
        ));
    }

    let weight_refs: Vec<_> = layer_weights
        .iter()
        .map(|(a, b, c, d, e, f, g, h)| (a, b, c, d, e, f, g, h))
        .collect();

    let final_norm_cached = match (final_norm_handle, final_norm_bytes) {
        (Some(handle), Some(bytes)) => Some(graph.get_or_upload_f32_weight(handle, bytes)?),
        _ => None,
    };

    let lm_head_cached = match (lm_head_handle, lm_head_bytes) {
        (Some(handle), Some(bytes)) => Some(graph.get_or_upload_q1_weight_soa(handle, bytes)?),
        _ => None,
    };

    graph.encode_full_forward_prefill(
        hidden_batch,
        pos_start,
        batch_size,
        n_layers,
        &weight_refs,
        cos_table,
        sin_table,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        eps,
        max_seq_len,
        final_norm_cached.as_ref(),
        final_norm_eps,
        lm_head_cached.as_ref(),
        lm_head_out_features,
        logits_out,
        greedy_token_id_out,
    )
}

/// Full-forward prefill for **verification** (speculative decoding).
///
/// Runs all transformer layers then final-norm + LM-head on **every** batch
/// position and returns per-position argmax token IDs.
#[allow(clippy::too_many_arguments)]
pub fn try_metal_full_forward_prefill_verify(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[FullForwardLayerParams<'_>],
    cos_table: &[f32],
    sin_table: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    eps: f32,
    max_seq_len: usize,
    final_norm_handle: Option<u64>,
    final_norm_bytes: Option<&[f32]>,
    final_norm_eps: f32,
    lm_head_handle: Option<u64>,
    lm_head_bytes: Option<&[u8]>,
    lm_head_out_features: usize,
    batch_token_ids_out: &mut Vec<u32>,
) -> Result<(), MetalGraphError> {
    if layer_params.len() != n_layers {
        return Err(MetalGraphError::EncodingFailed(format!(
            "layer_params length mismatch: need {n_layers}, got {}",
            layer_params.len()
        )));
    }

    let graph = MetalGraph::global()?;

    // Upload/cache all per-layer weights (same as try_metal_full_forward_prefill)
    #[allow(clippy::type_complexity)]
    let mut layer_weights: Vec<(
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
        Arc<MetalWeightHandle>,
    )> = Vec::with_capacity(n_layers);

    for lp in layer_params {
        let attn_norm_w =
            graph.get_or_upload_f32_weight(lp.attn_norm_handle, lp.attn_norm_bytes)?;
        let q_norm_w = graph.get_or_upload_f32_weight(lp.q_norm_handle, lp.q_norm_bytes)?;
        let k_norm_w = graph.get_or_upload_f32_weight(lp.k_norm_handle, lp.k_norm_bytes)?;
        let ffn_norm_w = graph.get_or_upload_f32_weight(lp.ffn_norm_handle, lp.ffn_norm_bytes)?;

        let fused_qkv_w =
            graph.get_or_upload_q1_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)?;
        let attn_proj_w =
            graph.get_or_upload_q1_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)?;

        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_w = graph.get_or_upload_q1_weight_soa_lazy(lp.gate_up_handle, || {
            let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            fused.extend_from_slice(gate_bytes);
            fused.extend_from_slice(up_bytes);
            fused
        })?;

        let down_w = graph.get_or_upload_q1_weight_soa(lp.down_handle, lp.down_bytes)?;

        layer_weights.push((
            attn_norm_w,
            fused_qkv_w,
            q_norm_w,
            k_norm_w,
            attn_proj_w,
            ffn_norm_w,
            gate_up_w,
            down_w,
        ));
    }

    let weight_refs: Vec<_> = layer_weights
        .iter()
        .map(|(a, b, c, d, e, f, g, h)| (a, b, c, d, e, f, g, h))
        .collect();

    let final_norm_cached = match (final_norm_handle, final_norm_bytes) {
        (Some(handle), Some(bytes)) => Some(graph.get_or_upload_f32_weight(handle, bytes)?),
        _ => None,
    };

    let lm_head_cached = match (lm_head_handle, lm_head_bytes) {
        (Some(handle), Some(bytes)) => Some(graph.get_or_upload_q1_weight_soa(handle, bytes)?),
        _ => None,
    };

    graph.encode_full_forward_prefill_verify(
        hidden_batch,
        pos_start,
        batch_size,
        n_layers,
        &weight_refs,
        cos_table,
        sin_table,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        eps,
        max_seq_len,
        final_norm_cached.as_ref(),
        final_norm_eps,
        lm_head_cached.as_ref(),
        lm_head_out_features,
        batch_token_ids_out,
    )
}

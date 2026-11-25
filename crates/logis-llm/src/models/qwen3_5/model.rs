use std::io::{Read, Seek};

use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor, quantized::QMatMul};
use crate::utils::GpuDeviceExt;
use candle_nn::{
    Conv1d, Embedding, Linear, Module, VarBuilder, embedding, linear_b, linear_no_bias,
    ops::sigmoid,
};

use crate::models::qwen::quantized_model::{KVBlock, KVLocation, KVRegistry};

#[derive(Clone)]
pub enum EmbedTable {
    Dense(Embedding),
    Quant {
        q: std::sync::Arc<candle_core::quantized::QTensor>,
        hidden: usize,
    },
    /// 🌟 [VOCAB TRIM] 표를 장치에 올리지 않고 매핑된 GGUF 에서 쓰는 행만 읽어 풀어 올립니다 (입력 임베딩 VRAM 0).
    /// 출력 투영은 남긴 행만의 [`crate::models::common::vocab_trim::TrimmedHead`] 입니다.
    Mapped {
        src: crate::models::common::vocab_trim::TableSource,
        device: Device,
    },
}

/// Host-side token ID bounds validation before device upload or embedding lookup.
pub fn validate_host_token_ids(ids: &[u32], vocab_size: usize, slot: &str) -> Result<()> {
    if let Some(&bad) = ids.iter().find(|&&id| (id as usize) >= vocab_size) {
        return Err(anyhow!(
            "[{slot}] token ID {bad} out of bounds for vocab size {vocab_size}"
        ));
    }
    Ok(())
}

/// Deep copy a tensor onto `dev`. Uses `Tensor::copy` when source and destination share the same
/// device because `Tensor::to_device` performs a shallow clone on same-device transfers.
fn clone_tensor_to_device(t: &Tensor, dev: &Device) -> Result<Tensor> {
    let c = t.contiguous()?;
    if c.device().same_device(dev) {
        Ok(c.copy()?)
    } else {
        Ok(c.to_device(dev)?)
    }
}

impl EmbedTable {
    /// Bounds guard ensuring every token ID in `ids` is strictly `< vocab` (Phase 2-2 D3).
    pub fn validate_ids_slice(&self, ids: &[u32]) -> candle_core::Result<()> {
        let v = self.vocab()?;
        validate_host_token_ids(ids, v, "EmbedTable")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    /// Tensor-level bounds guard (checks `max_id < vocab`).
    pub fn validate_ids(&self, ids: &Tensor) -> candle_core::Result<()> {
        if ids.elem_count() == 0 {
            return Ok(());
        }
        let flat = ids.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?;
        self.validate_ids_slice(&flat)
    }

    pub fn forward(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(e) => e.forward(ids),
            Self::Quant { q, .. } => q.embedding(ids),
            Self::Mapped { src, device } => {
                let flat = ids.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?;
                self.validate_ids_slice(&flat)?;
                let mut dims = ids.dims().to_vec();
                dims.push(src.cols);
                Self::mapped_rows(src, &flat, device)?.reshape(dims)
            }
        }
    }

    /// [`Self::rows_f32`] + 매핑된 표면 어휘 블록(의미 선입견의 코사인)을 남긴 행의 출력 투영(`head`, 장치)에서 꺼냅니다.
    /// 잘린 행 자리는 남긴 행 하나(0 번)로 채웁니다: 코사인은 행마다 따로 계산되어(블록 모양도 그대로) 남긴 행의 값은 같고, 0 으로 채우면
    /// 노름 0 이라 NaN 이 됩니다. 채운 자리의 선입견은 생성기가 `vocab_trim::exclude_dropped` 로 +∞(빼면 −∞)로 바꿔 쓰이지 않습니다. 매핑에서 어휘 전체를 읽으면
    /// 생성마다 표 페이지를 다시 들여와 0.6B 목록 추출에서 호출당 약 59 ms 가 더 들었습니다.
    pub fn rows_f32_with_head(&self, off: usize, take: usize, head: Option<&crate::models::common::vocab_trim::TrimmedHead>) -> candle_core::Result<Tensor> {
        let (Self::Mapped { device, .. }, Some(head)) = (self, head) else {
            return self.rows_f32(off, take);
        };
        let rows: Vec<u32> = (off..off + take).map(|id| head.row_of(id as u32).unwrap_or(0)).collect();
        head.embed_rows(&rows, device)
    }

    /// [`Self::forward`] + 매핑된 표면 남긴 행의 출력 투영(`head`, 장치)에 있는 토큰은 거기서 꺼내고(`TrimmedHead::embed_rows`),
    /// 남긴 행 밖의 토큰(HTML 의 NBSP 같은 기호 등)만 매핑에서 읽어 원래 순서로 합칩니다. 둘 다 원래 표의 행이라 값은 같습니다.
    /// 투영이 아직 없으면 매핑에서 읽습니다.
    pub fn forward_with_head(&self, ids: &Tensor, head: Option<&crate::models::common::vocab_trim::TrimmedHead>) -> candle_core::Result<Tensor> {
        let (Self::Mapped { src, device }, Some(head)) = (self, head) else {
            return self.forward(ids);
        };
        let flat = ids.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?;
        self.validate_ids_slice(&flat)?;
        let mut dims = ids.dims().to_vec();
        dims.push(src.cols);
        let (mut kept, mut miss, mut order) = (Vec::new(), Vec::new(), Vec::with_capacity(flat.len()));
        for &id in &flat {
            match head.row_of(id) {
                Some(j) => {
                    order.push((false, kept.len()));
                    kept.push(j);
                }
                None => {
                    order.push((true, miss.len()));
                    miss.push(id);
                }
            }
        }
        let x = if miss.is_empty() {
            head.embed_rows(&kept, device)?
        } else if kept.is_empty() {
            Self::mapped_rows(src, &flat, device)?
        } else {
            let nk = kept.len();
            let both = Tensor::cat(&[&head.embed_rows(&kept, device)?, &Self::mapped_rows(src, &miss, device)?], 0)?;
            let perm: Vec<u32> = order.iter().map(|&(m, i)| if m { (nk + i) as u32 } else { i as u32 }).collect();
            both.index_select(&Tensor::from_vec(perm, flat.len(), device)?, 0)?
        };
        x.reshape(dims)
    }

    /// 매핑된 표의 `ids` 행을 F32 로 풀어 `device` 에. Q8_0 은 장치에서 풀어도 CPU 와 비트까지 같으므로(f16 척도 × int8 이 f32 에서 정확)
    /// 양자화 행만 올리고 장치에서 풉니다 (CPU 에서 F32 로 풀어 올리면 올리는 양이 3.8 배이고 CPU 풀기가 더해집니다).
    fn mapped_rows(src: &crate::models::common::vocab_trim::TableSource, ids: &[u32], device: &Device) -> candle_core::Result<Tensor> {
        let on = if src.dtype == candle_core::quantized::GgmlDType::Q8_0 { device.clone() } else { Device::Cpu };
        let q = src.gather(ids, &on).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        q.dequantize(&on)?.to_device(device)
    }

    pub fn vocab(&self) -> candle_core::Result<usize> {
        match self {
            Self::Dense(e) => e.embeddings().dim(0),
            Self::Quant { q, .. } => Ok(q.shape().dims()[0]),
            Self::Mapped { src, .. } => Ok(src.rows),
        }
    }

    pub fn rows_f32(&self, off: usize, take: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(e) => e.embeddings().narrow(0, off, take)?.to_dtype(DType::F32),
            Self::Quant { q, .. } => {
                let ids = Tensor::arange(off as u32, (off + take) as u32, &q.device())?;
                q.embedding(&ids)
            }
            Self::Mapped { src, device } => {
                let ids: Vec<u32> = (off as u32..(off + take) as u32).collect();
                Self::mapped_rows(src, &ids, device)
            }
        }
    }

    /// 매핑된 표의 자리 (잘라 내기가 켜진 경우)
    pub fn mapped_source(&self) -> Option<&crate::models::common::vocab_trim::TableSource> {
        match self {
            Self::Mapped { src, .. } => Some(src),
            _ => None,
        }
    }

    pub fn dense(&self) -> Option<&Tensor> {
        match self {
            Self::Dense(e) => Some(e.embeddings()),
            Self::Quant { .. } | Self::Mapped { .. } => None,
        }
    }

    pub fn hidden(&self) -> usize {
        match self {
            Self::Dense(e) => e.embeddings().dim(1).unwrap_or(0),
            Self::Quant { hidden, .. } => *hidden,
            Self::Mapped { src, .. } => src.cols,
        }
    }
}

fn embed_q8_env() -> Option<bool> {
    std::env::var("LOGIS_EMBED_Q8").ok().map(|v| {
        let v = v.trim().to_ascii_lowercase();
        !(v.is_empty() || v == "0" || v == "false" || v == "off" || v == "no")
    })
}

/// 🌟 [PYRAMID-QUANT / LM HEAD] 묶인 입력 임베딩 · 출력 헤드를 GGUF 의 Q8_0 블록 그대로 둘지.
///
/// F16 표는 Q8_0 값(정수 × F16 척도)을 F16 으로 다시 반올림한 것이라 값이 더 정확하지 않고, 크기는 두 배입니다.
/// ROCm(RX 6600) 실측 (tests/it/p2b_models.rs `p2b_qwen35_2b_lm_head_q8` · `p2b_lm_head_cost`):
///   2B 출력 헤드 한 토큰 F16 19.4 ms → Q8_0 8.4 ms, 디코드 45.4 → 35.4 ms/토큰, 상주 VRAM −508 MB,
///   teacher-forcing top-1 100 % · KLD 7e-6, 탐욕 생성 같음.
/// CUDA(RTX 3050 sm_86)에서도 `get_rows_q8_0` 임베딩 + W8A16 `forward_exact` / Quantized Mode A 를 실기 검증했으며,
/// 로드 시 [`quant_embed_self_check`] 로 자동 폴백 가드를 유지합니다 (`LOGIS_EMBED_Q8=0` 으로 끌 수 있음).
fn use_quant_embed(device: &Device) -> bool {
    embed_q8_env().unwrap_or_else(|| device.is_vulkan() || device.is_rocm() || device.is_cuda())
}

fn quant_embed_ok(dtype: candle_core::quantized::GgmlDType) -> bool {
    embed_q8_env() == Some(true) || dtype == candle_core::quantized::GgmlDType::Q8_0
}

/// CUDA 등에서 양자화 임베딩/헤드를 켤 때 소형 자가 검증(첫 4행 역양자화 + `q.embedding` + 1행 `forward_exact`)이
/// CPU 기준과 일치하는지 확인하고 불일치 시 Dense 표로 안전 폴백합니다 (`P4-F12`).
pub fn quant_embed_self_check(device: &Device, q: &candle_core::quantized::QTensor) -> bool {
    if !device.is_cuda() {
        return true;
    }
    let check = (|| -> Result<bool> {
        let dims = q.shape().dims();
        if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
            return Ok(false);
        }
        let rows = dims[0].min(4);
        let cols = dims[1];
        let gpu_rows = q.dequantize_rows_to(0, rows, DType::F32, &Device::Cpu)?;
        let vals = gpu_rows.flatten_all()?.to_vec1::<f32>()?;
        if !vals.iter().all(|v| v.is_finite()) {
            return Ok(false);
        }
        // Verify `QTensor::embedding` (`get_rows_*` CUDA kernel) matches dequantized rows.
        let ids = Tensor::arange(0u32, rows as u32, device)?;
        let emb = q.embedding(&ids)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let emb_vals = emb.flatten_all()?.to_vec1::<f32>()?;
        if emb_vals.len() != vals.len() {
            return Ok(false);
        }
        for (a, b) in emb_vals.iter().zip(vals.iter()) {
            if !a.is_finite() || (a - b).abs() > 1e-4 * b.abs().max(1.0) {
                return Ok(false);
            }
        }
        let probe_x = Tensor::ones((1, cols), DType::F32, device)?;
        let gpu_y = q.forward_exact(&probe_x)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let gy = gpu_y.flatten_all()?.to_vec1::<f32>()?;
        if gy.len() != dims[0] || !gy.iter().all(|v| v.is_finite()) {
            return Ok(false);
        }
        // Check first `rows` dot products against row sums of dequantized rows.
        for r in 0..rows {
            let want: f32 = vals[r * cols..(r + 1) * cols].iter().sum();
            let diff = (gy[r] - want).abs();
            let tol = 1e-2 * want.abs().max(1.0);
            if diff > tol {
                return Ok(false);
            }
        }
        Ok(true)
    })();
    match check {
        Ok(true) => true,
        Ok(false) => {
            println!("[MODEL] ⚠️ CUDA Q8 embed self-check mismatch → falling back to Dense embed table.");
            false
        }
        Err(e) => {
            println!("[MODEL] ⚠️ CUDA Q8 embed self-check error ({e}) → falling back to Dense embed table.");
            false
        }
    }
}

use crate::{
    models::{
        common::{
            conv1d_depthwise, get_conv1d,
            gguf::{GateUpDownMLPGguf, Gguf, ProjKind, QuantizedLinear},
            softplus,
        },
        qwen3_5::config::{Qwen3_5Config, Qwen3_5TextConfig},
        qwen3vl::model::Qwen3VLVisionModel,
    },
    position_embed::rope::{Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb},
    utils::tensor_utils::{
        l2_normalize, masked_scatter_dim0,
        prepare_causal_attention_mask, repeat_interleave, split_tensor,
    },
};

#[derive(Clone)]
pub struct Qwen3_5RMSNorm {
    eps: f64,
    weight: Tensor,
}

impl Qwen3_5RMSNorm {
    pub fn new(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let weight = weight.to_dtype(candle_core::DType::F32)?.affine(1.0, 1.0)?;
        Ok(Self { eps, weight })
    }

    pub fn from_weight(weight: Tensor, eps: f64) -> Result<Self> {
        Ok(Self { eps, weight })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let d = *xs.dims().last().unwrap_or(&0);
        if crate::models::common::accel_check::fused_op_ok(
            xs.device(),
            crate::models::common::accel_check::FusedOpKind::ResidualRmsNorm,
            xs.dtype(),
            d,
        ) {
            return Ok(candle_core::decoder_ops::rmsnorm_fused(
                xs,
                &self.weight,
                self.eps as f32,
                false,
            )?);
        }
        // F32로 저장된 가중치를 xs(BF16) 타입에 맞춰준 뒤 네이티브 커널 실행!
        let w = self.weight.to_dtype(xs.dtype())?;
        Ok(candle_nn::ops::rms_norm(xs, &w, self.eps as f32)?)
    }

    pub fn weight(&self) -> &Tensor { &self.weight }

    pub fn clear(&mut self) {
        self.weight = Tensor::zeros((1,), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
    }
    pub fn eps(&self) -> f64 { self.eps }
    
    // 현재 가중치가 1바이트 껍데기인지(비워져 있는지) 확인
    pub fn is_cleared(&self) -> bool { self.weight.elem_count() <= 1 }
}

pub struct Qwen3_5RMSNormGated {
    weight: Tensor,
    eps: f64,
    dtype: DType,
}

impl Qwen3_5RMSNormGated {
    pub fn new(vb: VarBuilder, hidden_size: usize, eps: f64) -> Result<Self> {
        let dtype = vb.dtype();
        let weight = vb.get(hidden_size, "weight")?;
        Ok(Self { weight, eps, dtype })
    }

    pub fn from_weight(weight: Tensor, eps: f64) -> Result<Self> {
        let dtype = weight.dtype();
        Ok(Self { weight, eps, dtype })
    }

    pub fn forward(&self, xs: &Tensor, gate: Option<&Tensor>) -> Result<Tensor> {
        let w = self.weight.to_dtype(xs.dtype())?;
        let mut out = candle_nn::ops::rms_norm(xs, &w, self.eps as f32)?;
        if let Some(gate) = gate {
            
            let gate_val = gate.to_dtype(candle_core::DType::F32)?.silu()?.to_dtype(xs.dtype())?;
            out = out.broadcast_mul(&gate_val)?;
        }
        Ok(out)
    }

    pub fn clear(&mut self) {
        self.weight = Tensor::zeros((1,), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
    }
    pub fn eps(&self) -> f64 { self.eps }
}

#[macro_export]
macro_rules! transmute_tensors {
    ($($tensor:expr),*) => {
        ($(
            $tensor.transpose(1, 2)?.contiguous()?.to_dtype(candle_core::DType::F32)?,
        )*)
    };
}
#[macro_export]
macro_rules! right_pad_zero_tensor {
    ($dim:expr, $pad_size:expr, $($tensor:expr),+) => {
        ($(
            $tensor.pad_with_zeros($dim, 0, $pad_size)?.contiguous()?,
        )+)
    };
}

#[macro_export]
macro_rules! reshape_chunk_tensor {
    ($chunk_size:expr, $($tensor:expr),*) => {
        ($(
            {
                let (bs, head, _, dim) = $tensor.dims4()?;
                $tensor.reshape((bs, head, (), $chunk_size, dim))?.contiguous()?
            },
        )*)
    };
}

pub struct Qwen3_5GatedDeltaNet {
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv1d: Conv1d,
    dt_bias: Tensor,
    a_log: Tensor,
    norm: Qwen3_5RMSNormGated,
    out_proj: ProjKind,
    in_proj_qkv: ProjKind,
    in_proj_z: ProjKind,
    in_proj_b: ProjKind,
    in_proj_a: ProjKind,
    conv_state_cache: Option<Tensor>,
    recurrent_state_cache: Option<Tensor>,
    pub is_state_dirty: bool,
    // 🌟 [GDN-FUSED] v-head 가 읽는 k-head 대응 방식 (GGUF alphaedge.v_head_order, 기본 Grouped = repeat_interleave)
    v_head_order: candle_core::delta_rule::GdnHeadOrder,
}

/// 🌟 [GDN-FUSED] GGUF 메타데이터 `logis.v_head_order` 우선, 구형 `alphaedge.v_head_order` 폴백 ("grouped" / "tiled").
fn gguf_v_head_order<R: Read + Seek>(gguf: &Gguf<R>) -> candle_core::delta_rule::GdnHeadOrder {
    let v = gguf
        .get_matedata("logis.v_head_order")
        .or_else(|_| gguf.get_matedata("alphaedge.v_head_order"))
        .ok()
        .and_then(|v| v.to_string().ok().cloned())
        .unwrap_or_default();
    if v.trim().eq_ignore_ascii_case("tiled") {
        candle_core::delta_rule::GdnHeadOrder::Tiled
    } else {
        candle_core::delta_rule::GdnHeadOrder::Grouped
    }
}

impl Qwen3_5GatedDeltaNet {
    pub fn new_from_vb(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let hidden_size = config.hidden_size; 
        let num_v_heads = config.linear_num_value_heads; 
        let num_k_heads = config.linear_num_key_heads; 
        let head_k_dim = config.linear_key_head_dim; 
        let head_v_dim = config.linear_value_head_dim; 
        let key_dim = head_k_dim * num_k_heads; 
        let value_dim = head_v_dim * num_v_heads; 
        let conv_kernel_size = config.linear_conv_kernel_dim; 
        let layer_norm_epsilon = config.rms_norm_eps;
        let conv_dim = key_dim * 2 + value_dim; 
        
        
        let conv1d = get_conv1d(vb.pp("ssm_conv1d"), conv_dim, conv_dim, conv_kernel_size, 0, 1, 1, conv_dim, false)?;
        let dt_bias = vb.get(num_v_heads, "ssm_dt.bias")?;
        let a_log = vb.get(num_v_heads, "ssm_a")?;
        let norm = Qwen3_5RMSNormGated::new(vb.pp("ssm_norm"), head_v_dim, layer_norm_epsilon)?;

        let out_proj = linear_no_bias(value_dim, hidden_size, vb.pp("out_proj"))?;
        let in_proj_qkv = linear_no_bias(hidden_size, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_no_bias(hidden_size, value_dim, vb.pp("in_proj_z"))?;
        let in_proj_b = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_a"))?;

        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv1d,
            dt_bias,
            a_log,
            norm,
            out_proj: ProjKind::LinearProj(out_proj),
            in_proj_qkv: ProjKind::LinearProj(in_proj_qkv),
            in_proj_z: ProjKind::LinearProj(in_proj_z),
            in_proj_b: ProjKind::LinearProj(in_proj_b),
            in_proj_a: ProjKind::LinearProj(in_proj_a),
            conv_state_cache: None,
            recurrent_state_cache: None,
            is_state_dirty: false,
            v_head_order: candle_core::delta_rule::GdnHeadOrder::Grouped,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(
        gguf: &mut Gguf<R>,
        prefix: &str,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        let num_k_heads = gguf.get_matedata("qwen35.ssm.group_count")?.to_u32()? as usize;
        let num_v_heads = gguf.get_matedata("qwen35.ssm.time_step_rank")?.to_u32()? as usize;
        let conv_kernel_size = gguf.get_matedata("qwen35.ssm.conv_kernel")?.to_u32()? as usize;
        let head_k_dim = gguf.get_matedata("qwen35.ssm.state_size")?.to_u32()? as usize;
        let head_v_dim = head_k_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let conv1d = gguf.conv1d(
            &format!("{prefix}.ssm_conv1d"),
            0, 
            1,
            1,
            conv_dim,
            false,
        )?;
        
        
        let dt_bias = gguf.get_dequantized_f16(&format!("{prefix}.ssm_dt.bias"))?.to_dtype(DType::F32)?;
        let a_log = gguf.get_dequantized_f16(&format!("{prefix}.ssm_a"))?.to_dtype(DType::F32)?;
        let norm_weight = gguf.get_dequantized_f16(&format!("{prefix}.ssm_norm.weight"))?.to_dtype(DType::F32)?;
        let norm = Qwen3_5RMSNormGated::from_weight(norm_weight, rms_norm_eps)?;
        
        let out_proj = gguf.quantize_linear(&format!("{prefix}.ssm_out"), false)?;
        let in_proj_qkv = gguf.quantize_linear(&format!("{prefix}.attn_qkv"), false)?;
        let in_proj_z = gguf.quantize_linear(&format!("{prefix}.attn_gate"), false)?;
        let in_proj_b = gguf.quantize_linear(&format!("{prefix}.ssm_beta"), false)?;
        let in_proj_a = gguf.quantize_linear(&format!("{prefix}.ssm_alpha"), false)?;

        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv1d,
            dt_bias,
            a_log,
            norm,
            out_proj: ProjKind::QuantizedProj(out_proj),
            in_proj_qkv: ProjKind::QuantizedProj(in_proj_qkv),
            in_proj_z: ProjKind::QuantizedProj(in_proj_z),
            in_proj_b: ProjKind::QuantizedProj(in_proj_b),
            in_proj_a: ProjKind::QuantizedProj(in_proj_a),
            conv_state_cache: None,
            recurrent_state_cache: None,
            is_state_dirty: false,
            v_head_order: gguf_v_head_order(gguf),
        })
    }

    fn torch_causal_conv1d_update(&mut self, xs: &Tensor) -> Result<Tensor> {
        let conv_state = self.conv_state_cache.as_ref().unwrap();
        let seq_len = xs.dim(2)?; 
        let state_len = conv_state.dim(D::Minus1)?;
        let take_len = self.conv_kernel_size - 1; 
        
        let state_to_use = if state_len > take_len {
            conv_state.narrow(D::Minus1, state_len - take_len, take_len)?
        } else if state_len < take_len {
            conv_state.pad_with_zeros(D::Minus1, take_len - state_len, 0)?
        } else {
            conv_state.clone()
        };
        
        let conv_state_new = Tensor::cat(&[&state_to_use, xs], D::Minus1)?;
        
        let next_cache_len = conv_state_new.dim(D::Minus1)?;
        // 🌟 [GDN-02] narrow 뷰는 [B,C,K] 버퍼 전체를 붙잡으므로 마지막 K-1 열만 연속 복사해 둡니다. (값은 비트 단위로 같음)
        let conv_update = conv_state_new.narrow(D::Minus1, next_cache_len - take_len, take_len)?.contiguous()?;
        self.conv_state_cache = Some(conv_update);
        
        let out = conv1d_depthwise(&conv_state_new, self.conv1d.weight(), self.conv1d.bias())?;
        
        let out_len = out.dim(D::Minus1)?;
        let final_out = if out_len > seq_len {
            out.narrow(D::Minus1, out_len - seq_len, seq_len)?
        } else {
            out
        };
        
        Ok(final_out.silu()?)
    }

    fn torch_chunk_gated_delta_rule(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        use_qk_l2norm_in_kernel: bool,
        chunk_size: usize,
    ) -> Result<Tensor> {
        let (query, key) = if use_qk_l2norm_in_kernel {
            (l2_normalize(query, 3)?, l2_normalize(key, 3)?)
        } else {
            (query.clone(), key.clone())
        };
        let initial_dtype = query.dtype();
        let (query, key, value, beta, g) = transmute_tensors!(query, key, value, beta, g);
        let (batch_size, num_heads, sequence_length, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
        let (query, key, value, beta, g) =
            right_pad_zero_tensor!(2, pad_size, query, key, value, beta, g);
        let total_sequence_length = sequence_length + pad_size;
        let scale = 1.0 / (query.dim(D::Minus1)? as f64).sqrt();
        let query = query.affine(scale, 0.0)?;
        let v_beta = value.broadcast_mul(&beta.unsqueeze(D::Minus1)?.contiguous()?)?;
        let k_beta = key.broadcast_mul(&beta.unsqueeze(D::Minus1)?.contiguous()?)?;
        let (query, key, k_beta, v_beta) =
            reshape_chunk_tensor!(chunk_size, query, key, k_beta, v_beta);
        let g = g.reshape((g.dim(0)?, g.dim(1)?, (), chunk_size))?;
        let g = g.cumsum(D::Minus1)?;
        let decay_mask = g
            .unsqueeze(D::Minus1)?
            .broadcast_sub(&g.unsqueeze(D::Minus2)?)?
            .exp()?
            .to_dtype(candle_core::DType::F32)?;
            
        
        let tril_mask = Tensor::tril2(chunk_size, candle_core::DType::F32, query.device())?
            .to_dtype(candle_core::DType::U8)?
            .broadcast_as(decay_mask.shape())?;
        
        let on_false = decay_mask.zeros_like()?;
        let decay_mask = tril_mask.where_cond(&decay_mask, &on_false)?.contiguous()?;
        
        let mut attn = k_beta.squeeze(0)?.contiguous()?.matmul(&key.squeeze(0)?.transpose(D::Minus1, D::Minus2)?.contiguous()?)?.unsqueeze(0)?.mul(&decay_mask)?.affine(-1.0, 0.0)?;
        
        
        let mask = Tensor::triu2(chunk_size, candle_core::DType::F32, query.device())?
            .to_dtype(candle_core::DType::U8)?
            .broadcast_as(decay_mask.shape())?;
        
        attn = mask.where_cond(&on_false, &attn)?;
        if attn.device().is_vulkan() {
            attn = candle_core::delta_rule::chunk_tril_recurrence(&attn)?;
        } else {
            let (d0, d1, d2, _, _) = attn.dims5()?;
            for i in 1..chunk_size {
                let row = attn.i((.., .., .., i, ..i))?.contiguous()?;
                let sub = attn.i((.., .., .., ..i, ..i))?.contiguous()?;
                let attn_i = row
                    .unsqueeze(D::Minus1)?
                    .broadcast_mul(&sub)?
                    .sum(D::Minus2)?
                    .add(&row)?
                    .unsqueeze(D::Minus2)?;
                attn = attn.slice_assign(&[(0..d0), (0..d1), (0..d2), (i..i + 1), (0..i)], &attn_i)?;
            }
        }
        let attn = attn
            .broadcast_add(&Tensor::eye(chunk_size, attn.dtype(), attn.device())?)?
            .contiguous()?;
        
        let value = attn.squeeze(0)?.matmul(&v_beta.squeeze(0)?)?.unsqueeze(0)?;
        let k_cumdecay = attn
            .squeeze(0)?
            .matmul(
                &k_beta
                    .broadcast_mul(&g.exp()?.unsqueeze(D::Minus1)?)?
                    .squeeze(0)?,
            )?
            .unsqueeze(0)?;
        let mut last_recurrent_state = if let Some(recurrent) = self.recurrent_state_cache.as_ref()
        {
            recurrent.clone()
        } else {
            Tensor::zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                candle_core::DType::F32,
                value.device(),
            )?
        };

        let mut core_attn_out = value.zeros_like()?;
        
        let tril_mask = Tensor::tril2(chunk_size, candle_core::DType::F32, query.device())?
            .to_dtype(candle_core::DType::U8)?
            .broadcast_as((batch_size, num_heads, chunk_size, chunk_size))?;
        let on_false = tril_mask.zeros_like()?.to_dtype(candle_core::DType::F32)?;
        let last_dim = core_attn_out.dim(D::Minus1)?;
        let vk_cat = value.device().is_vulkan();
        let mut out_chunks: Vec<Tensor> = Vec::new();
        for i in 0..total_sequence_length / chunk_size {
            let q_i = query.i((.., .., i))?.contiguous()?;
            let k_i = key.i((.., .., i))?.contiguous()?;
            let v_i = value.i((.., .., i))?.contiguous()?;
            let g_i = g.i((.., .., i))?.contiguous()?;
            let attn = q_i
                .matmul(&k_i.transpose(D::Minus1, D::Minus2)?.contiguous()?)?
                .mul(&decay_mask.i((.., .., i))?)?;
            let attn = tril_mask.where_cond(&attn, &on_false)?.contiguous()?;
            let v_prime = k_cumdecay.i((.., .., i))?.matmul(&last_recurrent_state)?;
            let v_new = v_i.sub(&v_prime)?;
            let attn_inter = q_i
                .broadcast_mul(&g_i.unsqueeze(D::Minus1)?.exp()?)?
                .matmul(&last_recurrent_state)?;
            let out_i = attn_inter.add(&attn.matmul(&v_new)?)?.unsqueeze(2)?;
            if vk_cat {
                out_chunks.push(out_i);
            } else {
                core_attn_out = core_attn_out.slice_assign(
                    &[
                        (0..batch_size),
                        (0..num_heads),
                        (i..i + 1),
                        (0..chunk_size),
                        (0..last_dim),
                    ],
                    &out_i,
                )?;
            }
            let g_i_last_dim = g_i.dim(D::Minus1)?;
            last_recurrent_state = last_recurrent_state
                .broadcast_mul(
                    &g_i.narrow(D::Minus1, g_i_last_dim - 1, 1)?
                        .unsqueeze(D::Minus1)?
                        .exp()?,
                )?
                .add(
                    &k_i.broadcast_mul(
                        &g_i.narrow(D::Minus1, g_i_last_dim - 1, 1)?
                            .broadcast_sub(&g_i)?
                            .exp()?
                            .unsqueeze(D::Minus1)?,
                    )?
                    .transpose(D::Minus1, D::Minus2)?
                    .squeeze(0)?
                    .matmul(&v_new.squeeze(0)?)?
                    .unsqueeze(0)?,
                )?;
        }
        self.recurrent_state_cache = Some(last_recurrent_state);
        if vk_cat && !out_chunks.is_empty() {
            core_attn_out = Tensor::cat(&out_chunks, 2)?;
        }
        core_attn_out =
            core_attn_out.reshape((batch_size, num_heads, (), core_attn_out.dim(D::Minus1)?))?;
        core_attn_out = core_attn_out.narrow(2, 0, sequence_length)?;
        core_attn_out = core_attn_out
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(initial_dtype)?;

        Ok(core_attn_out)
    }

    fn torch_recurrent_gated_delta_rule(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        use_qk_l2norm_in_kernel: bool,
    ) -> Result<Tensor> {
        let (query, key) = if use_qk_l2norm_in_kernel {
            (l2_normalize(query, 3)?, l2_normalize(key, 3)?)
        } else {
            (query.clone(), key.clone())
        };
        let initial_dtype = query.dtype();
        
        let (batch_size, sequence_length, num_heads, k_head_dim) = query.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let scale = 1.0 / (k_head_dim as f64).sqrt();
        let query = query.affine(scale, 0.0)?;
        
        let mut last_recurrent_state = if let Some(recurrent) = self.recurrent_state_cache.as_ref() {
            recurrent.clone()
        } else {
            Tensor::zeros((batch_size, num_heads, k_head_dim, v_head_dim), candle_core::DType::F32, value.device())?
        };

        if sequence_length == 1 {
            
            let q_i = query.squeeze(1)?.contiguous()?.to_dtype(candle_core::DType::F32)?;
            let k_i = key.squeeze(1)?.contiguous()?.to_dtype(candle_core::DType::F32)?;
            let v_i = value.squeeze(1)?.contiguous()?.to_dtype(candle_core::DType::F32)?;
            
            let g_i = g.squeeze(1)?.contiguous()?.to_dtype(candle_core::DType::F32)?.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?.contiguous()?;
            let beta_i = beta.squeeze(1)?.contiguous()?.to_dtype(candle_core::DType::F32)?.unsqueeze(D::Minus1)?.contiguous()?;
            
            // println!("[DEBUG-CONTIG] SSM Fast-Path Q: {}, K: {}, V: {}", q_i.is_contiguous(), k_i.is_contiguous(), v_i.is_contiguous());

            last_recurrent_state = last_recurrent_state.broadcast_mul(&g_i)?;
            let vk_mm = last_recurrent_state.device().is_vulkan();
            let kv_mem = if vk_mm {
                k_i.unsqueeze(D::Minus2)?.contiguous()?.matmul(&last_recurrent_state)?.squeeze(D::Minus2)?
            } else {
                last_recurrent_state.broadcast_mul(&k_i.unsqueeze(D::Minus1)?.contiguous()?)?.sum(D::Minus2)?
            };
            let delta = v_i.broadcast_sub(&kv_mem)?.broadcast_mul(&beta_i)?;
            last_recurrent_state = last_recurrent_state.broadcast_add(
                &k_i.unsqueeze(D::Minus1)?.contiguous()?.broadcast_mul(&delta.unsqueeze(D::Minus2)?.contiguous()?)?,
            )?;
            let out_i = if vk_mm {
                q_i.unsqueeze(D::Minus2)?.contiguous()?.matmul(&last_recurrent_state)?
            } else {
                last_recurrent_state.broadcast_mul(&q_i.unsqueeze(D::Minus1)?.contiguous()?)?.sum_keepdim(D::Minus2)?
            };
            
            self.recurrent_state_cache = Some(last_recurrent_state);
            
            
            return Ok(out_i.transpose(1, 2)?.contiguous()?.to_dtype(initial_dtype)?); 
        }

        let (query, key, value, beta, g) = transmute_tensors!(query, key, value, beta, g);
        let mut core_attn_out = Tensor::zeros(
            (batch_size, num_heads, sequence_length, v_head_dim),
            candle_core::DType::F32,
            value.device(),
        )?;
        for i in 0..sequence_length {
            let q_i = query.i((.., .., i))?;
            let k_i = key.i((.., .., i))?;
            let v_i = value.i((.., .., i))?;
            let g_i = g.i((.., .., i))?.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
            let beta_i = beta.i((.., .., i))?.unsqueeze(D::Minus1)?;
            
            last_recurrent_state = last_recurrent_state.broadcast_mul(&g_i)?;
            let kv_mem = last_recurrent_state.broadcast_mul(&k_i.unsqueeze(D::Minus1)?)?.sum(D::Minus2)?;
            let delta = v_i.broadcast_sub(&kv_mem)?.broadcast_mul(&beta_i)?;
            last_recurrent_state = last_recurrent_state.broadcast_add(&k_i.unsqueeze(D::Minus1)?.broadcast_mul(&delta.unsqueeze(D::Minus2)?)?)?;
            let out_i = last_recurrent_state.broadcast_mul(&q_i.unsqueeze(D::Minus1)?)?.sum_keepdim(D::Minus2)?;
            
            core_attn_out = core_attn_out.slice_assign(&[(0..batch_size), (0..num_heads), (i..i + 1), (0..v_head_dim)], &out_i)?;
        }
        self.recurrent_state_cache = Some(last_recurrent_state);
        core_attn_out = core_attn_out.transpose(1, 2)?.contiguous()?.to_dtype(initial_dtype)?;

        Ok(core_attn_out)
    }

    pub fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let dev = xs.device();
        let dtype = xs.dtype();
        
        if let Some(conv) = self.conv_state_cache.take() {
            self.conv_state_cache = Some(conv.to_device(dev)?.to_dtype(dtype)?);
        }
        if let Some(rec) = self.recurrent_state_cache.take() {
            self.recurrent_state_cache = Some(rec.to_device(dev)?.to_dtype(candle_core::DType::F32)?);
        }

        // 🌟 [GDN-FUSED] 포크의 융합 커널이 있으면 혼합기 전체(conv·SiLU·L2 norm·게이트·스캔)를 한 번에 계산합니다.
        let gdn_cfg = self.gdn_config();
        if crate::models::common::gdn_fused_ready(dev, dtype, &gdn_cfg) {
            return self.forward_fused(xs, &gdn_cfg);
        }

        let (bs, seq_len, _) = xs.dims3()?;


        let mut mixed_qkv = self.in_proj_qkv.forward(xs)?
            .to_dtype(dtype)? // 👈 SSM 입력 타입 보정
            .transpose(1, 2)?
            .contiguous()?; 
        
        let z = self.in_proj_z.forward(xs)?
            .to_dtype(dtype)? // 👈 SSM 게이트 타입 보정
            .reshape((bs, seq_len, (), self.head_v_dim))?;
        let b = self.in_proj_b.forward(xs)?.to_dtype(dtype)?;
        let a = self.in_proj_a.forward(xs)?.to_dtype(dtype)?;
        let use_precomputed_states =
            self.conv_state_cache.is_some() && self.recurrent_state_cache.is_some() && seq_len == 1;
        if use_precomputed_states {
            mixed_qkv = self.torch_causal_conv1d_update(&mixed_qkv)?;
        } else {
            let take_len = self.conv_kernel_size - 1;
            let (bs, dim, seq_len) = mixed_qkv.dims3()?;

            let prev_causal = if let Some(prev_state) = &self.conv_state_cache {
                let prev_len = prev_state.dim(D::Minus1)?;
                if prev_len >= take_len {
                    prev_state.narrow(D::Minus1, prev_len - take_len, take_len)?
                } else {
                    prev_state.pad_with_zeros(D::Minus1, take_len - prev_len, 0)?
                }
            } else {
                Tensor::zeros((bs, dim, take_len), mixed_qkv.dtype(), mixed_qkv.device())?
            };

            let cat_u = Tensor::cat(&[&prev_causal, &mixed_qkv], D::Minus1)?;
            let cat_len = cat_u.dim(D::Minus1)?;
            // 🌟 [GDN-02] narrow 뷰를 그대로 두면 [B,C,T] 프리필 버퍼 전체가 상태로 살아남아(RAM 대피 시 통째로 복사)
            //    마지막 K-1 열만 연속 복사합니다. prev_causal + mixed_qkv 연결 뒤 잘라 suffix seq_len < K-1 이어도 앞 문맥을 유지합니다.
            let next_conv_state = if cat_len >= take_len {
                cat_u.narrow(D::Minus1, cat_len - take_len, take_len)?.contiguous()?
            } else {
                cat_u.pad_with_zeros(D::Minus1, take_len - cat_len, 0)?
            };
            self.conv_state_cache = Some(next_conv_state);
            mixed_qkv = conv1d_depthwise(&cat_u, self.conv1d.weight(), self.conv1d.bias())?;
            
            let out_len = mixed_qkv.dim(D::Minus1)?;
            mixed_qkv = if out_len > seq_len {
                mixed_qkv.narrow(D::Minus1, out_len - seq_len, seq_len)?
            } else {
                mixed_qkv
            };
            
            mixed_qkv = mixed_qkv.silu()?;
        }
        let mixed_qkv = mixed_qkv.transpose(1, 2)?;
        let qkv_split = split_tensor(
            &mixed_qkv,
            &[self.key_dim, self.key_dim, self.value_dim],
            D::Minus1,
        )?;
        
        
        let mut query = qkv_split[0].contiguous()?.reshape((bs, seq_len, (), self.head_k_dim))?;
        let mut key = qkv_split[1].contiguous()?.reshape((bs, seq_len, (), self.head_k_dim))?;
        let value = qkv_split[2].contiguous()?.reshape((bs, seq_len, (), self.head_v_dim))?;

        
        let beta = sigmoid(&b)?.to_dtype(dtype)?; 
        
        
        let a_plus_bias = softplus(
            &a.to_dtype(candle_core::DType::F32)?.broadcast_add(&self.dt_bias.to_dtype(candle_core::DType::F32)?)?,
        )?.to_dtype(dtype)?;
        let g = self.a_log.to_dtype(dtype)?.broadcast_mul(&a_plus_bias)?;

        if self.num_v_heads / self.num_k_heads > 1 {
            query = repeat_interleave(&query, self.num_v_heads / self.num_k_heads, 2)?;
            key = repeat_interleave(&key, self.num_v_heads / self.num_k_heads, 2)?;
        }
        let core_attn_out = if !use_precomputed_states {
            self.torch_chunk_gated_delta_rule(&query, &key, &value, &g, &beta, true, 64)?
        } else {
            self.torch_recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, true)?
        };
        
        
        // println!("[DEBUG-CONTIG] SSM Output: {}, Z: {}", core_attn_out.is_contiguous(), z.is_contiguous());
        let core_attn_out = core_attn_out.contiguous()?.reshape(((), self.head_v_dim))?;
        let z = z.contiguous()?.reshape(((), self.head_v_dim))?;
        
        let core_attn_out = self.norm.forward(&core_attn_out, Some(&z))?.to_dtype(dtype)?;
        
        let core_attn_out = core_attn_out.contiguous()?.reshape((bs, seq_len, ()))?; 
        
        let output = self.out_proj.forward(&core_attn_out)?.to_dtype(dtype)?; 

        self.is_state_dirty = true; 
        Ok(output)
    }

    /// 🌟 [GDN-FUSED] 포크 융합 커널에 넘길 층 형상.
    fn gdn_config(&self) -> candle_core::delta_rule::GdnConfig {
        let mut cfg = candle_core::delta_rule::GdnConfig::new(
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        );
        cfg.conv_kernel = self.conv_kernel_size;
        cfg.v_head_order = self.v_head_order;
        cfg
    }

    /// 🌟 [GDN-FUSED] 융합 경로: 프로젝션 → `fused_gated_delta_rule`(conv+SiLU+L2norm+게이트+스캔, F32 연산)
    /// → `fused_gated_rms_norm` → out_proj.
    ///
    /// 상태 형식은 기존 경로와 같습니다: conv 상태 `[B, C, K-1]` (모델 dtype, conv 이전 입력의 마지막 K-1 열),
    /// 재귀 상태 `[B, Hv, Dk, Dv]` F32. 커널이 새 상태를 따로 만들어 돌려주므로 프리필 버퍼를 붙잡지 않습니다.
    /// 기존 경로와 달리 conv·L2 norm·g·beta 를 F16 모델(2B)에서도 F32 로 계산합니다(기준 구현에 더 가까움).
    fn forward_fused(&mut self, xs: &Tensor, cfg: &candle_core::delta_rule::GdnConfig) -> Result<Tensor> {
        let dtype = xs.dtype();
        let (bs, seq_len, _) = xs.dims3()?;
        // [B, T, C] 그대로 넘깁니다 (전치·연속 복사 없음).
        let mixed_qkv = self.in_proj_qkv.forward(xs)?.to_dtype(dtype)?;
        let z = self
            .in_proj_z
            .forward(xs)?
            .to_dtype(dtype)?
            .reshape((bs, seq_len, self.num_v_heads, self.head_v_dim))?;
        let b = self.in_proj_b.forward(xs)?.to_dtype(dtype)?;
        let a = self.in_proj_a.forward(xs)?.to_dtype(dtype)?;

        // 스냅샷 등에서 온 conv 상태 길이가 K-1 과 다르면 기존 경로와 같은 규칙(뒤쪽 K-1 열 / 앞쪽 0 채움)으로 맞춥니다.
        let take_len = self.conv_kernel_size - 1;
        // (커널이 실패해도 문맥을 잃지 않도록 기존 상태는 성공한 뒤에만 교체합니다)
        let conv_state = match self.conv_state_cache.as_ref() {
            Some(s) => {
                let len = s.dim(D::Minus1)?;
                let s = if len > take_len {
                    s.narrow(D::Minus1, len - take_len, take_len)?
                } else if len < take_len {
                    s.pad_with_zeros(D::Minus1, take_len - len, 0)?
                } else {
                    s.clone()
                };
                Some(s.contiguous()?)
            }
            None => None,
        };
        let rec_state = self.recurrent_state_cache.clone();

        let out = candle_core::delta_rule::fused_gated_delta_rule(
            &mixed_qkv,
            self.conv1d.weight(),
            self.conv1d.bias(),
            conv_state.as_ref(),
            &a,
            &b,
            &self.a_log,
            &self.dt_bias,
            rec_state.as_ref(),
            cfg,
        )?;
        drop(mixed_qkv);
        self.conv_state_cache = Some(out.conv_state);
        self.recurrent_state_cache = Some(out.recurrent_state);

        // 게이트 RMSNorm: x / sqrt(mean(x²)+eps) · w · silu(z), F32 연산 후 모델 dtype.
        let core = candle_core::delta_rule::fused_gated_rms_norm(&out.out, &z, &self.norm.weight, self.norm.eps())?;
        let core = core.reshape((bs, seq_len, ()))?;
        let output = self.out_proj.forward(&core)?.to_dtype(dtype)?;
        self.is_state_dirty = true;
        Ok(output)
    }

    pub fn clear_cache(&mut self) {
        self.conv_state_cache = None;
        self.recurrent_state_cache = None;
        self.is_state_dirty = false;
    }

    pub fn get_states(&self) -> (Option<Tensor>, Option<Tensor>) {
        (self.conv_state_cache.clone(), self.recurrent_state_cache.clone())
    }

    pub fn set_states(&mut self, conv: Option<Tensor>, recurrent: Option<Tensor>) {
        self.conv_state_cache = conv;
        self.recurrent_state_cache = recurrent;
        self.is_state_dirty = false; 
    }

    pub fn capture_state(&self, target_dev: &Device) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let conv = match &self.conv_state_cache {
            Some(t) => Some(clone_tensor_to_device(t, target_dev)?),
            None => None,
        };
        let rec = match &self.recurrent_state_cache {
            Some(t) => Some(clone_tensor_to_device(t, target_dev)?),
            None => None,
        };
        Ok((conv, rec))
    }

    pub fn restore_state_cow(
        &mut self,
        conv: Option<&Tensor>,
        recurrent: Option<&Tensor>,
        compute_dev: &Device,
    ) -> Result<()> {
        self.conv_state_cache = match conv {
            Some(t) => Some(clone_tensor_to_device(t, compute_dev)?),
            None => None,
        };
        self.recurrent_state_cache = match recurrent {
            Some(t) => Some(clone_tensor_to_device(t, compute_dev)?),
            None => None,
        };
        self.is_state_dirty = false;
        Ok(())
    }

    pub fn clear_weights(&mut self) {
        let dummy_p = crate::models::common::gguf::dummy_proj(&Device::Cpu);
        self.in_proj_qkv = dummy_p.clone();
        self.in_proj_z = dummy_p.clone();
        self.in_proj_b = dummy_p.clone();
        self.in_proj_a = dummy_p.clone();
        self.out_proj = dummy_p;
        
        let dummy_t = Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap();
        self.dt_bias = dummy_t.clone();
        self.a_log = dummy_t.clone();
        self.norm.clear();
        self.conv1d = candle_nn::Conv1d::new(dummy_t.clone(), None, candle_nn::Conv1dConfig::default());
    }

    pub fn load_weights_inplace(&mut self, ct: &candle_core::quantized::gguf_file::Content, data: &[u8], prefix: &str, device: &Device) -> Result<()> {
        let conv_dim = self.key_dim * 2 + self.value_dim;
        
        
        let t_conv1d = crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.ssm_conv1d.weight"), device)?;
        let conv1d_weight_raw = t_conv1d.dequantize_f16(device).or_else(|_| t_conv1d.dequantize(device))?.to_dtype(DType::F32)?.contiguous()?;
        
        let conv1d_weight = conv1d_weight_raw.reshape((conv_dim, 1, self.conv_kernel_size))?;
        
        self.conv1d = candle_nn::Conv1d::new(conv1d_weight, None, candle_nn::Conv1dConfig { 
            padding: 0, stride: 1, dilation: 1, groups: conv_dim, cudnn_fwd_algo: None 
        });
        
        
        let t_dt = crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.ssm_dt.bias"), device)?;
        let dt_bias_raw = t_dt.dequantize_f16(device).or_else(|_| t_dt.dequantize(device))?;
        self.dt_bias = dt_bias_raw.to_dtype(DType::F32)?; 
        
        
        let t_a = crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.ssm_a"), device)?;
        let a_log_raw = t_a.dequantize_f16(device).or_else(|_| t_a.dequantize(device))?;
        self.a_log = a_log_raw.to_dtype(DType::F32)?;  
        
        
        let t_norm = crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.ssm_norm.weight"), device)?;
        let norm_weight = t_norm.dequantize_f16(device).or_else(|_| t_norm.dequantize(device))?.to_dtype(DType::F32)?;
        self.norm = Qwen3_5RMSNormGated::from_weight(norm_weight, 1e-6)?;
        
        self.out_proj = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.ssm_out.weight"), device)?)?, None));
        self.in_proj_qkv = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.attn_qkv.weight"), device)?)?, None));
        self.in_proj_z = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.attn_gate.weight"), device)?)?, None));
        self.in_proj_b = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.ssm_beta.weight"), device)?)?, None));
        self.in_proj_a = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.ssm_alpha.weight"), device)?)?, None));
        Ok(())
    }
}

/// 🌟 [OFFLOAD-ATTN] 블록을 한 번에 모아 계산할 때 허용하는 최대 크기(복원 기준 K+V 바이트).
/// 넘으면 블록을 하나씩 올려 계산하는 기존 경로로 메모리를 묶어 둡니다. (4B 기준 약 12만 토큰)
pub const OFFLOAD_ATTN_GATHER_BUDGET: usize = 512 * 1024 * 1024;

/// 🌟 [OFFLOAD-ATTN] fused flash-attn 을 쓸 수 없을 때(빌드·설정·자가 검증) 비융합 경로가 만들 점수 행렬의 상한.
/// 넘으면 프리필을 블록 순회 경로로 계산합니다.
pub const OFFLOAD_ATTN_UNFUSED_BUDGET: usize = 256 * 1024 * 1024;

/// `LOGIS_KV_BLOCK_ATTN=legacy` (또는 `block`, `off`) 이면 기존 블록 순회 경로만 씁니다. (A/B 비교용)
fn offload_attn_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let v = std::env::var("LOGIS_KV_BLOCK_ATTN").unwrap_or_default().trim().to_ascii_lowercase();
        let on = !matches!(v.as_str(), "legacy" | "block" | "off" | "0" | "false");
        if !on {
            println!("[OFFLOAD-ATTN] LOGIS_KV_BLOCK_ATTN={v} → Qwen3.5 어텐션을 기존 블록 순회 경로로 계산합니다.");
        }
        on
    })
}

pub struct Qwen3_5Attention {
    q_proj: ProjKind,
    k_proj: ProjKind,
    v_proj: ProjKind,
    o_proj: ProjKind,
    q_norm: Qwen3_5RMSNorm,
    k_norm: Qwen3_5RMSNorm,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    kv_blocks: Vec<KVBlock>,
    registry: KVRegistry,
    layer_idx: usize,
    pub active_session_id: Option<String>,
    pub active_kv_name: Option<String>,
    // 🌟 [KV RESIDENCY] 디코딩 진입 시 1회 결정된 KV 블록 배치 위치.
    //    기본값 Ssd 는 기존 동작(VRAM→RAM→SSD 연속 강등)과 100% 동일합니다.
    pub kv_residency: crate::utils::resources::KvResidency,
    /// 🌟 [P1-2 / P4-O01] RAM 대피 KV 블록을 계산 스트림과 중첩 업로드하는 비동기 프리페처
    pub prefetcher: Option<crate::utils::prefetch::OffloadPrefetcher>,
    /// 다음 레이어 진입 전에 선제 업로드를 시작한 블록 (블록 인덱스, 전송 핸들, TurboQuant 메타)
    pub prefetched_block: Option<(usize, crate::utils::prefetch::PrefetchedKv, Option<crate::models::common::kv_turbo::TqMeta>)>,
    // 🌟 [KV-PRECISION] true 면 새 블록을 압축(TurboQuant·FP8)하지 않고 모델 dtype 으로 둡니다.
    //    문맥 시작 시 VRAM 여유로 정합니다. RAM/SSD 로 내릴 때는 기존처럼 FP8 로 줄여 내립니다.
    pub kv_exact: bool,
    /// 🌟 [KV-HALF] 원본 KV 를 반정밀도(BF16 지원 GPU 는 BF16, 그 외 F16/CPU 는 넘치면 BF16)로 보관합니다.
    pub kv_half: bool,
}

/// 🌟 [KV-HALF] 원본 KV 조각을 반정밀도로 바꿉니다.
/// - `prev` 가 있으면(기존 블록에 이어 붙임) 그 dtype 을 따릅니다. F16 블록에 붙는 값은 F16 범위로 잘라 inf 를 막습니다.
/// - 새 블록(`prev == None`):
///   - BF16 지원 GPU(CUDA/ROCm): `to_scalar` 호스트 동기화와 ±60000 포화 클램핑 없이 곧바로 BF16 으로 보관합니다 (`P4-F04`).
///   - 그 외 GPU(Vulkan/Metal): 안전 범위 클램프 후 F16 으로 보관합니다.
///   - CPU: `|x|` 최댓값을 재어 F16 범위 이내면 F16, 넘으면 BF16 으로 보관합니다.
pub fn half_kv_piece(x: &Tensor, prev: Option<DType>) -> Result<Tensor> {
    let lim = crate::models::common::kv_turbo::F16_SAFE_MAX as f64;
    match prev {
        Some(dt) if x.dtype() == dt => Ok(x.clone()),
        Some(DType::BF16) => Ok(x.to_dtype(DType::BF16)?),
        Some(_) => Ok(x.clamp(-lim, lim)?.to_dtype(DType::F16)?),
        None => {
            if matches!(x.dtype(), DType::F16 | DType::BF16) {
                return Ok(x.clone());
            }
            if x.device().is_cuda_or_rocm() {
                // F32 -> BF16: F32 와 동일한 지수 범위(±3.4e38)라 클램프나 호스트 동기화 없이 오버플로 안전합니다.
                return Ok(x.to_dtype(DType::BF16)?);
            }
            if !x.device().is_cpu() {
                return Ok(x.clamp(-lim, lim)?.to_dtype(DType::F16)?);
            }
            let max_abs = x.abs()?.flatten_all()?.max(0)?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
            if max_abs.is_finite() && (max_abs as f64) <= lim {
                Ok(x.to_dtype(DType::F16)?)
            } else {
                Ok(x.to_dtype(DType::BF16)?)
            }
        }
    }
}

impl Qwen3_5Attention {
    pub fn new_from_vb(vb: VarBuilder, config: &Qwen3_5TextConfig, layer_idx: usize, registry: KVRegistry) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let num_key_value_heads = config.num_key_value_heads;
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);
        let q_proj = linear_b(hidden_size, num_attention_heads * head_dim * 2, config.attention_bias, vb.pp("q_proj"))?;
        let k_proj = linear_b(hidden_size, num_key_value_heads * head_dim, config.attention_bias, vb.pp("k_proj"))?;
        let v_proj = linear_b(hidden_size, num_key_value_heads * head_dim, config.attention_bias, vb.pp("v_proj"))?;
        let o_proj = linear_b(num_attention_heads * head_dim, hidden_size, config.attention_bias, vb.pp("o_proj"))?;
        let q_norm = Qwen3_5RMSNorm::new(vb.pp("q_norm"), head_dim, config.rms_norm_eps)?;
        let k_norm = Qwen3_5RMSNorm::new(vb.pp("k_norm"), head_dim, config.rms_norm_eps)?;

        Ok(Self {
            q_proj: ProjKind::LinearProj(q_proj),
            k_proj: ProjKind::LinearProj(k_proj),
            v_proj: ProjKind::LinearProj(v_proj),
            o_proj: ProjKind::LinearProj(o_proj),
            q_norm, k_norm, num_attention_heads, num_key_value_heads, num_kv_groups, head_dim, scaling,
            kv_blocks: Vec::new(),
            registry,
            layer_idx,
            active_session_id: None,
            active_kv_name: None,
            kv_residency: crate::utils::resources::KvResidency::Ssd,
            prefetcher: None,
            prefetched_block: None,
            kv_exact: false,
            kv_half: false,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(gguf: &mut Gguf<R>, prefix: &str, rms_norm_eps: f64, layer_idx: usize, registry: KVRegistry) -> Result<Self> {
        let num_attention_heads = gguf.get_matedata("qwen35.attention.head_count")?.to_u32()? as usize;
        let num_key_value_heads = gguf.get_matedata("qwen35.attention.head_count_kv")?.to_u32()? as usize;
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let head_dim = gguf.get_matedata("qwen35.attention.key_length")?.to_u32()? as usize;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);
        let q_proj = gguf.quantize_linear(&format!("{prefix}.attn_q"), false)?;
        let k_proj = gguf.quantize_linear(&format!("{prefix}.attn_k"), false)?;
        let v_proj = gguf.quantize_linear(&format!("{prefix}.attn_v"), false)?;
        let o_proj = gguf.quantize_linear(&format!("{prefix}.attn_output"), false)?;
        let q_norm_weight = gguf.get_dequantized(&format!("{prefix}.attn_q_norm.weight"))?;
        let q_norm = Qwen3_5RMSNorm::from_weight(q_norm_weight, rms_norm_eps)?;
        let k_norm_weight = gguf.get_dequantized(&format!("{prefix}.attn_k_norm.weight"))?;
        let k_norm = Qwen3_5RMSNorm::from_weight(k_norm_weight, rms_norm_eps)?;

        Ok(Self {
            q_proj: ProjKind::QuantizedProj(q_proj),
            k_proj: ProjKind::QuantizedProj(k_proj),
            v_proj: ProjKind::QuantizedProj(v_proj),
            o_proj: ProjKind::QuantizedProj(o_proj),
            q_norm, k_norm, num_attention_heads, num_key_value_heads, num_kv_groups, head_dim, scaling,
            kv_blocks: Vec::new(),
            registry,
            layer_idx,
            active_session_id: None,
            active_kv_name: None,
            kv_residency: crate::utils::resources::KvResidency::Ssd,
            prefetcher: None,
            prefetched_block: None,
            kv_exact: false,
            kv_half: false,
        })
    }

    pub fn forward(&mut self, xs: &Tensor, cos: &Tensor, sin: &Tensor, attention_mask: Option<&Tensor>, seqlen_offset: usize) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let target_dtype = xs.dtype();
        
        let query_chunk = self.q_proj.forward(xs)?
            .to_dtype(target_dtype)? // 👈 타입 불일치 방어 1
            .reshape((b_sz, q_len, self.num_attention_heads, self.head_dim * 2))?
            .chunk(2, D::Minus1)?;
            
        
        let q_raw = query_chunk[0].contiguous()?.reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?;
        let gate = query_chunk[1].contiguous()?.reshape((b_sz, q_len, ()))?;
        let k_raw = self.k_proj.forward(xs)?
            .to_dtype(target_dtype)?
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?;
        let v_raw = self.v_proj.forward(xs)?
            .to_dtype(target_dtype)?
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?;

        let (query_states, key_states, value_states) = if crate::models::common::accel_check::fused_op_ok(
            q_raw.device(),
            crate::models::common::accel_check::FusedOpKind::RopeKvWrite,
            q_raw.dtype(),
            self.head_dim,
        ) {
            let cos_h = crate::models::common::rope_half_table(cos)?;
            let sin_h = crate::models::common::rope_half_table(sin)?;
            let k_out = Tensor::zeros(
                (b_sz, self.num_key_value_heads, q_len, self.head_dim),
                target_dtype,
                q_raw.device(),
            )?;
            let v_out = Tensor::zeros(
                (b_sz, self.num_key_value_heads, q_len, self.head_dim),
                target_dtype,
                q_raw.device(),
            )?;
            // Qwen3_5RMSNorm.weight 는 이미 (1 + w) 로 사전 가산되어 있으므로 norm_plus_one = false.
            let q_out = candle_core::decoder_ops::rope_kv_write(
                &q_raw,
                &k_raw,
                &v_raw,
                candle_core::accel::decoder_ops::HeadLayout::Bshd,
                candle_core::accel::decoder_ops::HeadLayout::Bhsd,
                Some(self.q_norm.weight()),
                Some(self.k_norm.weight()),
                &cos_h,
                &sin_h,
                &k_out,
                &v_out,
                0,
                self.q_norm.eps() as f32,
                candle_core::accel::decoder_ops::RopeMode::NeoX,
                false,
            )?;
            (q_out, k_out, v_out)
        } else {
            let query_states = self.q_norm.forward(&q_raw)?.transpose(1, 2)?.contiguous()?;
            let key_states = self.k_norm.forward(&k_raw)?.transpose(1, 2)?.contiguous()?;
            let value_states = v_raw.transpose(1, 2)?.contiguous()?;

            // 전체 차원(256) 중 RoPE 차원(64)만 잘라내서 회전시킨 뒤, 나머지 차원과 다시 결합합니다!
            let rot_dim = cos.dim(D::Minus1)?;
            let (query_states, key_states) = if rot_dim < self.head_dim {
                let q_rot = query_states.narrow(D::Minus1, 0, rot_dim)?;
                let q_pass = query_states.narrow(D::Minus1, rot_dim, self.head_dim - rot_dim)?;
                let k_rot = key_states.narrow(D::Minus1, 0, rot_dim)?;
                let k_pass = key_states.narrow(D::Minus1, rot_dim, self.head_dim - rot_dim)?;

                let (q_rot, k_rot) = apply_rotary_pos_emb(&q_rot, &k_rot, cos, sin, false)?;

                (Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?, Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?)
            } else {
                apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?
            };
            (query_states, key_states, value_states)
        };
        
        let dev = xs.device();
        let target_dtype = xs.dtype();

        // 🌟 [TURBO-KV-BLOCK] GPU + turbo-quant 빌드에서는 새 블록을 TurboQuant 압축 u32 행으로 보관합니다.
        //    기본은 벡터별 스케일 FP8(정확도 우선, 기존 FP8 블록과 거의 같은 크기)이고,
        //    CANDLE_TURBO_QUANT_BITS=4 등으로 4bit 를 고르면 FP8 대비 약 56% 크기입니다. (kv_turbo::kv_quant_options)
        //    None 이면(CPU, CANDLE_TURBO_QUANT=off, turbo-quant 미빌드, GPU 자가 검증 실패) 기존 FP8 경로 그대로입니다.
        //    새 블록을 만들 때만 필요하므로 처음 필요할 때 한 번 정합니다.
        let head_dim = self.head_dim;
        let mut tq_codec: Option<Option<std::sync::Arc<crate::models::common::kv_turbo::TqBlockCodec>>> = None;

        let mut tokens_to_process = q_len;
        let mut chunk_offset = 0;
        while tokens_to_process > 0 {
            let mut appended = false;
            if let Some(last_block) = self.kv_blocks.last_mut() {
                let mut inner = last_block.inner.write().unwrap();
                let free_space = 1024usize.saturating_sub(inner.len);
                if inner.location == KVLocation::VRAM && free_space > 0 {
                    let take = tokens_to_process.min(free_space);
                    
                    
                    let k_piece = key_states.narrow(2, chunk_offset, take)?.contiguous()?;
                    let v_piece = value_states.narrow(2, chunk_offset, take)?.contiguous()?;

                    if let (Some(pk), Some(pv)) = (inner.k_cache.take(), inner.v_cache.take()) {
                        let pk = if !pk.device().same_device(dev) { pk.to_device(dev)? } else { pk };
                        let pv = if !pv.device().same_device(dev) { pv.to_device(dev)? } else { pv };

                        if let Some(meta) = inner.tq {
                            // 🌟 [TURBO-KV-BLOCK] 압축 블록: 새 토큰만 같은 코덱으로 압축해 뒤에 붙입니다.
                            //    (FP8 경로처럼 블록 전체를 풀었다가 다시 압축하지 않습니다)
                            let codec = crate::models::common::kv_turbo::codec_for(&meta, dev)?;
                            let nk = codec.pack_k(&k_piece)?;
                            let nv = codec.pack_v(&v_piece)?;
                            inner.k_cache = Some(Tensor::cat(&[&pk, &nk], 2)?.contiguous()?);
                            inner.v_cache = Some(Tensor::cat(&[&pv, &nv], 2)?.contiguous()?);
                        } else if self.kv_exact && self.kv_half && matches!(pk.dtype(), DType::F16 | DType::BF16) {
                            // 🌟 [KV-HALF] 보관 dtype 그대로 이어 붙입니다 (블록 전체를 F32 로 풀었다 다시 줄이지 않습니다).
                            let nk = half_kv_piece(&k_piece, Some(pk.dtype()))?;
                            let nv = half_kv_piece(&v_piece, Some(pv.dtype()))?;
                            inner.k_cache = Some(Tensor::cat(&[&pk, &nk], 2)?.contiguous()?);
                            inner.v_cache = Some(Tensor::cat(&[&pv, &nv], 2)?.contiguous()?);
                        } else {
                            let pk_f = pk.to_dtype(target_dtype).unwrap_or_else(|_| pk.clone());
                            let pv_f = pv.to_dtype(target_dtype).unwrap_or_else(|_| pv.clone());

                            let cat_k = Tensor::cat(&[&pk_f, &k_piece], 2)?.contiguous()?;
                            let cat_v = Tensor::cat(&[&pv_f, &v_piece], 2)?.contiguous()?;

                            // 🌟 [KV-PRECISION / PRC-04] 원본 KV 모드이거나 TurboQuant 부재 폴백일 때 스케일 없는 FP8 로 조용히 강등하지 않습니다.
                            let fp8 = dev.is_cuda_or_rocm() && !self.kv_exact && raw_fp8_kv_enabled();
                            if fp8 {
                                inner.k_cache = Some(cat_k.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| cat_k.clone()));
                                inner.v_cache = Some(cat_v.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| cat_v.clone()));
                            } else if self.kv_half && matches!(cat_k.dtype(), DType::F32) {
                                inner.k_cache = Some(half_kv_piece(&cat_k, None)?);
                                inner.v_cache = Some(half_kv_piece(&cat_v, None)?);
                            } else {
                                inner.k_cache = Some(cat_k);
                                inner.v_cache = Some(cat_v);
                            }
                        }
                        inner.len += take; tokens_to_process -= take; chunk_offset += take;
                        appended = true;
                        
                        let mut reg = self.registry.entries.write().unwrap();
                        if inner.index < reg.len() {
                            reg[inner.index].token_len = inner.len;
                            if self.layer_idx < reg[inner.index].is_dirty.len() { reg[inner.index].is_dirty[self.layer_idx] = true; }
                        }
                    }
                }
            }
            if !appended {
                let take = tokens_to_process.min(1024);
                
                
                let k_piece = key_states.narrow(2, chunk_offset, take)?.contiguous()?;
                let v_piece = value_states.narrow(2, chunk_offset, take)?.contiguous()?;
                let index = self.kv_blocks.len();
                let current_total = seqlen_offset + chunk_offset;
                
                let new_block = KVBlock::new(KVLocation::VRAM, index, take, current_total);
                {
                    let mut inner = new_block.inner.write().unwrap();
                    // 🌟 [KV-PRECISION] 원본 KV 모드면 압축 코덱을 쓰지 않습니다.
                    let codec = if self.kv_exact {
                        None
                    } else {
                        tq_codec
                            .get_or_insert_with(|| crate::models::common::kv_turbo::block_codec(head_dim, dev))
                            .clone()
                    };
                    match &codec {
                        Some(codec) => {
                            // 🌟 [TURBO-KV-BLOCK] 새 블록은 처음부터 TurboQuant 압축 u32 행으로 보관합니다.
                            inner.k_cache = Some(codec.pack_k(&k_piece)?);
                            inner.v_cache = Some(codec.pack_v(&v_piece)?);
                            inner.tq = Some(codec.meta);
                        }
                        None if self.kv_exact && self.kv_half => {
                            // 🌟 [KV-HALF] 원본 KV 를 F16 으로 보관합니다 (|x| 가 F16 범위를 넘으면 BF16).
                            inner.k_cache = Some(half_kv_piece(&k_piece, None)?);
                            inner.v_cache = Some(half_kv_piece(&v_piece, None)?);
                        }
                        None => {
                            let fp8 = dev.is_cuda_or_rocm() && !self.kv_exact && raw_fp8_kv_enabled();
                            if fp8 {
                                inner.k_cache = Some(k_piece.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| k_piece.clone()));
                                inner.v_cache = Some(v_piece.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| v_piece.clone()));
                            } else if self.kv_half && matches!(k_piece.dtype(), DType::F32) {
                                inner.k_cache = Some(half_kv_piece(&k_piece, None)?);
                                inner.v_cache = Some(half_kv_piece(&v_piece, None)?);
                            } else {
                                inner.k_cache = Some(k_piece);
                                inner.v_cache = Some(v_piece);
                            }
                        }
                    }
                }
                
                let mut reg = self.registry.entries.write().unwrap();
                if index < reg.len() {
                    reg[index].token_start = current_total;
                    reg[index].token_len = take;
                    if self.layer_idx < reg[index].is_dirty.len() { reg[index].is_dirty[self.layer_idx] = true; }
                    reg[index].location[self.layer_idx] = KVLocation::VRAM;
                }
                self.kv_blocks.push(new_block);
                tokens_to_process -= take; chunk_offset += take;
            }
        }

        let total_tokens_now = seqlen_offset + q_len;

        
        // 🌟 [OFFLOAD-ATTN] 블록을 먼저 계산 장치로 모은 뒤(필요하면 RAM/SSD 에서 읽어 오고) 한 번에 계산합니다.
        //   - 디코딩: 모든 블록이 같은 TurboQuant 형식이면 압축 행을 이어 붙여 fused 커널 한 번으로 계산합니다.
        //   - 프리필(과 그 밖의 경우): 과거 이력은 복원하고, 이번 청크는 압축 전 원본 K/V 로 이어 붙여 flash-attn 으로 계산합니다.
        //     기존 블록 경로는 방금 압축한 이번 청크를 다시 풀어 썼기 때문에 프롬프트 전체가 압축 오차를 안고 계산되었습니다.
        //   - 디코딩 배치 계획이 Vram 인데 RAM 에 내려가 있던 블록(프리필 중 RAM 으로 대피한 블록, SSD 에서 읽은 블록)은
        //     한 번 올린 뒤 VRAM 에 둡니다. 토큰마다 같은 블록을 다시 PCIe 로 올리지 않습니다.
        //   - 모은 크기가 예산을 넘거나 계산이 실패하면 기존 블록 순회(온라인 소프트맥스) 경로로 계산합니다.
        let promote_to_vram =
            self.kv_residency == crate::utils::resources::KvResidency::Vram && !dev.is_cpu();
        let mut parts: Vec<crate::models::common::kv_turbo::KvPart> = Vec::with_capacity(self.kv_blocks.len());

        let num_blocks = self.kv_blocks.len();
        for bi in 0..num_blocks {
            // 🌟 [P1-2 / P4-O01] 이 블록을 소비하기 전에 먼저 프리페치 핸들을 꺼내고, 다음 RAM 블록을 비동기 선제 업로드합니다.
            let prefetched_here = if self.prefetched_block.as_ref().is_some_and(|(idx, _, _)| *idx == bi) {
                self.prefetched_block.take().and_then(|(_, pf, tq)| {
                    self.prefetcher
                        .as_mut()
                        .and_then(|p| pf.wait_and_take(p).ok())
                        .map(|(kd, vd)| (kd, vd, tq))
                })
            } else {
                None
            };
            if bi + 1 < num_blocks {
                self.prefetch_ram_block(bi + 1, dev);
            }
            let block = &self.kv_blocks[bi];
            let (index, b_off, b_len) = {
                let inner = block.inner.read().unwrap();
                (inner.index, inner.offset, inner.len)
            };
            if b_off >= total_tokens_now { continue; }

            
            // (k, v, tq): tq 가 Some 이면 k / v 는 TurboQuant 압축 u32 행이고 복원은 누산기가 합니다.
            let (k_block, v_block, block_tq) = {
                let mut inner = block.inner.write().unwrap(); 
                if let Some((k_raw, v_raw, loaded_tq)) = prefetched_here {
                    inner.tq = loaded_tq;
                    let (k_gpu, v_gpu) = if loaded_tq.is_some() {
                        (k_raw.clone(), v_raw.clone())
                    } else {
                        (k_raw.to_dtype(target_dtype)?, v_raw.to_dtype(target_dtype)?)
                    };
                    if promote_to_vram && inner.location != KVLocation::VRAM {
                        inner.k_cache = Some(k_raw);
                        inner.v_cache = Some(v_raw);
                        inner.location = KVLocation::VRAM;
                        let mut reg = self.registry.entries.write().unwrap();
                        if index < reg.len() && self.layer_idx < reg[index].location.len() {
                            reg[index].location[self.layer_idx] = KVLocation::VRAM;
                        }
                    }
                    (k_gpu, v_gpu, loaded_tq)
                } else if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                    if inner.tq.is_some() {
                        // 🌟 [TURBO-KV-BLOCK] 압축 블록은 형 변환 없이 장치만 맞춥니다. (RAM 블록이면 압축된 채로 PCIe 전송)
                        let (kd, vd) = (k.to_device(dev)?, v.to_device(dev)?);
                        if promote_to_vram && inner.location != KVLocation::VRAM {
                            inner.k_cache = Some(kd.clone());
                            inner.v_cache = Some(vd.clone());
                            inner.location = KVLocation::VRAM;
                            let mut reg = self.registry.entries.write().unwrap();
                            if index < reg.len() && self.layer_idx < reg[index].location.len() {
                                reg[index].location[self.layer_idx] = KVLocation::VRAM;
                            }
                        }
                        (kd, vd, inner.tq)
                    } else if inner.location == KVLocation::VRAM {
                        (k.to_dtype(target_dtype).unwrap_or_else(|_| k.clone()), v.to_dtype(target_dtype).unwrap_or_else(|_| v.clone()), None) // VRAM에 있다면 복원해서 씀
                    } else {
                        // RAM에 있다면 GPU로 복사해서 씀 (배치 계획이 Vram 이면 저장 형식 그대로 VRAM 에 둡니다)
                        let (kd, vd) = (k.to_device(dev)?, v.to_device(dev)?);
                        if promote_to_vram {
                            inner.k_cache = Some(kd.clone());
                            inner.v_cache = Some(vd.clone());
                            inner.location = KVLocation::VRAM;
                            let mut reg = self.registry.entries.write().unwrap();
                            if index < reg.len() && self.layer_idx < reg[index].location.len() {
                                reg[index].location[self.layer_idx] = KVLocation::VRAM;
                            }
                        }
                        (kd.to_dtype(target_dtype)?, vd.to_dtype(target_dtype)?, None)
                    }
                } else {
                    let mut k_cpu = None;
                    let mut v_cpu = None;
                    let mut loaded_tq: Option<crate::models::common::kv_turbo::TqMeta> = None;
                    
                    {
                        let reg = self.registry.entries.read().unwrap();
                        let cache = reg[index].bitkv_cache.read().unwrap();
                        if let Some(m) = &cache[self.layer_idx] {
                            if m.tq.is_some() {
                                // 🌟 [TURBO-KV-BLOCK] RAM 캐시의 압축 블록은 그대로 사용합니다.
                                k_cpu = Some(m.k_data.clone()); v_cpu = Some(m.v_data.clone());
                                loaded_tq = m.tq;
                            } else {
                                let kd_t = if dev.is_cpu() { m.k_data.to_dtype(DType::F32)? } else { m.k_data.clone() };
                                let vd_t = if dev.is_cpu() { m.v_data.to_dtype(DType::F32)? } else { m.v_data.clone() };
                                k_cpu = Some(kd_t); v_cpu = Some(vd_t);
                            }
                        }
                    }

                    // ... SSD 읽어오는 로직 생략 (기존과 동일하게 작동함) ...
                    if k_cpu.is_none() {
                        let kv_dir = crate::utils::paths::get_kv_dir();
                        let sid = self.active_session_id.as_deref().unwrap_or("default_session");
                        let kv_name_raw = self.active_kv_name.as_deref().unwrap_or("text");
                        let kv_type = kv_name_raw.split('/').last().unwrap_or("text");
                        
                        let mut path_candidates = Vec::new();
                        
                        
                        // 이 경로가 누락되어 Base 스냅샷을 놔두고 빈 껍데기(0.0)를 읽어오는 환각이 발생했습니다.
                        if let Some(p) = {
                            let reg = self.registry.entries.read().unwrap();
                            if index < reg.len() { reg[index].ssd_path.clone() } else { None }
                        } {
                            path_candidates.push(p);
                        }

                        path_candidates.push(kv_dir.join(format!("{}/inference/{}/b{}", sid, kv_type, b_off)));
                        path_candidates.push(kv_dir.join(format!("{}/reference/{}/b{}", sid, kv_type, b_off)));

                        for full_path in path_candidates {
                            let block_file = full_path.join(format!("l{}.st", self.layer_idx));
                            for _retry in 0..3 {
                                if block_file.exists() {
                                    if let Ok(raw_content) = crate::utils::direct_loader::load_kv_block(&block_file) {
                                        // [CRYPTO-COMPAT] 저장측(spawn_slot_worker)은 평문 safetensors로 기록합니다.
                                        // decrypt_data 는 평문에도 Ok(쓰레기)를 돌려주므로(무결성 검증 없음),
                                        // 평문으로 파싱되는지 먼저 확인하고 암호문일 때만 복호화합니다. (zeros 폴백 방지)
                                        let content = crate::utils::crypto::kv_block_plaintext(raw_content);
                                        {
                                            if let Ok(st) = safetensors::SafeTensors::deserialize(&content) {
                                                let prefix = format!("b{}_l{}_", b_off, self.layer_idx);
                                                let get_t = |s: &str| st.tensor(&format!("{}{}", prefix, s)).or_else(|_| st.tensor(s)).ok();
                                                
                                                if let (Some(kd), Some(vd), Some(sh)) = (get_t("k_data"), get_t("v_data"), get_t("k_shape")) {
                                                    let sh_u32: Vec<u32> = sh.data().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                                                    let meta_os: Vec<usize> = sh_u32.iter().map(|&x| x as usize).collect();

                                                    // 🌟 [TURBO-KV-BLOCK] `tq_meta` 텐서가 있으면 TurboQuant 압축 블록입니다.
                                                    //    u32 행을 각자의 모양 그대로 읽고, 형식은 파일의 메타데이터를 따릅니다.
                                                    let file_tq = get_t("tq_meta").and_then(|t| crate::models::common::kv_turbo::TqMeta::from_le_bytes(t.data()));
                                                    if let Some(tq) = file_tq {
                                                        let packed = if kd.dtype() == safetensors::Dtype::U32 && vd.dtype() == safetensors::Dtype::U32 {
                                                            match (
                                                                Tensor::from_raw_buffer(kd.data(), DType::U32, kd.shape(), &Device::Cpu),
                                                                Tensor::from_raw_buffer(vd.data(), DType::U32, vd.shape(), &Device::Cpu),
                                                            ) {
                                                                (Ok(kd_t), Ok(vd_t)) => Some((kd_t, vd_t)),
                                                                _ => None,
                                                            }
                                                        } else {
                                                            None
                                                        };
                                                        if let Some((kd_t, vd_t)) = packed {
                                                            k_cpu = Some(kd_t.clone());
                                                            v_cpu = Some(vd_t.clone());
                                                            loaded_tq = Some(tq);

                                                            let mut reg = self.registry.entries.write().unwrap();
                                                            if index < reg.len() {
                                                                reg[index].ssd_path = Some(full_path.clone());
                                                                reg[index].location[self.layer_idx] = KVLocation::RAM;
                                                                let mut cache = reg[index].bitkv_cache.write().unwrap();
                                                                cache[self.layer_idx] = Some(crate::models::qwen::quantized_model::BitKVMetadata { k_data: kd_t, v_data: vd_t, original_shape: meta_os, tq: Some(tq) });
                                                            }
                                                            break;
                                                        }
                                                        println!("[TURBO-KV-BLOCK] 압축 블록 {:?} 를 읽지 못했습니다 (k {:?} v {:?}).", block_file, kd.dtype(), vd.dtype());
                                                    } else {
                                                    
                                                    let saved_dtype = match kd.dtype() {
                                                        safetensors::Dtype::F64 => DType::F8E4M3,
                                                        safetensors::Dtype::F32 => DType::F32,
                                                        safetensors::Dtype::F16 => DType::F16,
                                                        _ => DType::BF16,
                                                    };
                                                    // 🌟 [버그 수정] FP8/F32로 저장된 텐서를 무조건 BF16으로 읽어와서 파괴되는 현상을 고쳤습니다.
                                                    let mut kd_t = Tensor::from_raw_buffer(kd.data(), saved_dtype, &meta_os, &Device::Cpu).unwrap();
                                                    let mut vd_t = Tensor::from_raw_buffer(vd.data(), saved_dtype, &meta_os, &Device::Cpu).unwrap();
                                                    
                                                    if dev.is_cpu() {
                                                        kd_t = kd_t.to_dtype(DType::F32)?;
                                                        vd_t = vd_t.to_dtype(DType::F32)?;
                                                    }

                                                    k_cpu = Some(kd_t.clone());
                                                    v_cpu = Some(vd_t.clone());
                                                    
                                                    let mut reg = self.registry.entries.write().unwrap();
                                                    if index < reg.len() { 
                                                        reg[index].ssd_path = Some(full_path.clone());
                                                        reg[index].location[self.layer_idx] = KVLocation::RAM; 
                                                        let mut cache = reg[index].bitkv_cache.write().unwrap();
                                                        cache[self.layer_idx] = Some(crate::models::qwen::quantized_model::BitKVMetadata { k_data: kd_t, v_data: vd_t, original_shape: meta_os, tq: None });
                                                    }
                                                    break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    // [NO-BLOCKING-WAIT] 파일 자체가 없으면 재시도해도 생기지 않습니다.
                                    // 비동기 런타임 워커를 5ms씩 묶어두던 블로킹 sleep을 제거합니다.
                                    break;
                                }
                                std::thread::sleep(std::time::Duration::from_millis(5));
                            }
                            if k_cpu.is_some() { break; }
                        }
                    }
                    
                    // 🌟 [OFFLOAD-ATTN] RAM 캐시에도 SSD 에도 블록이 없으면 0 으로 채운 K/V 를 어텐션에 넣지 않습니다.
                    //    (0 키는 점수 0 으로 소프트맥스에 끼어들어 실제 문맥을 희석합니다. 빈 블록을 캐시에 남기지도 않아서
                    //     비동기 SSD 기록이 끝난 뒤의 토큰에서는 다시 읽어 옵니다)
                    let (Some(k_safe), Some(v_safe)) = (k_cpu, v_cpu) else {
                        static MISSING_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                        if !MISSING_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            println!(
                                "[OFFLOAD-ATTN] ⚠️ 레이어 {} 블록 {} (토큰 {}..{}) 를 RAM 캐시와 SSD 어디에서도 찾지 못해 이번 계산에서 제외합니다. (0 으로 채워 넣던 기존 동작은 문맥을 희석했습니다. 이후 같은 경고는 생략)",
                                self.layer_idx, index, b_off, b_off + b_len
                            );
                        }
                        continue;
                    };

                    // 🌟 [TURBO-KV-BLOCK] 압축 블록은 압축된 채로 장치에 올립니다. (복원은 계산 경로가 합니다)
                    let (k_raw, v_raw) = (k_safe.to_device(dev)?, v_safe.to_device(dev)?);
                    let (k_gpu, v_gpu) = if loaded_tq.is_some() {
                        (k_raw.clone(), v_raw.clone())
                    } else {
                        (k_raw.to_dtype(target_dtype)?, v_raw.to_dtype(target_dtype)?)
                    };

                    inner.tq = loaded_tq;
                    if promote_to_vram {
                        // 🌟 [OFFLOAD-ATTN] 배치 계획이 Vram 이면 저장 형식 그대로 VRAM 에 두고, RAM 사본은 버립니다.
                        inner.k_cache = Some(k_raw);
                        inner.v_cache = Some(v_raw);
                        inner.location = KVLocation::VRAM;
                        let reg = self.registry.entries.read().unwrap();
                        if index < reg.len() {
                            if let Ok(mut cache) = reg[index].bitkv_cache.write() {
                                if self.layer_idx < cache.len() {
                                    cache[self.layer_idx] = None;
                                }
                            }
                        }
                        drop(reg);
                        let mut reg = self.registry.entries.write().unwrap();
                        if index < reg.len() && self.layer_idx < reg[index].location.len() {
                            reg[index].location[self.layer_idx] = KVLocation::VRAM;
                        }
                    } else {
                        inner.k_cache = Some(k_safe);
                        inner.v_cache = Some(v_safe);
                        inner.location = KVLocation::RAM;
                    }

                    (k_gpu, v_gpu, loaded_tq)
                }
            };

            // 블록 텐서의 실제 행 수를 넘지 않게 합니다. (SSD 파일이 장부보다 짧은 경우 대비)
            let rows = k_block.dim(2).unwrap_or(0).min(v_block.dim(2).unwrap_or(0));
            let len = b_len.min(rows);
            if len == 0 {
                continue;
            }
            parts.push(crate::models::common::kv_turbo::KvPart {
                k: k_block,
                v: v_block,
                tq: block_tq,
                off: b_off,
                len,
            });
        }

        let attn_output = self.attend_parts(
            &query_states,
            &key_states,
            &value_states,
            &parts,
            attention_mask,
            seqlen_offset,
            total_tokens_now,
        )?;
        drop(parts);
        let attn_output = attn_output.to_dtype(target_dtype)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?.contiguous()?;
        
        // 💡 gate(F32/BF16)를 확실히 target_dtype으로 sigmoid 취한 뒤 다시 target_dtype으로 고정
        let gate_final = candle_nn::ops::sigmoid(&gate.to_dtype(target_dtype)?)?.to_dtype(target_dtype)?; 
        
        
        let attn_output = attn_output.mul(&gate_final)?;
        
        Ok(attn_output.apply(&self.o_proj)?.to_dtype(target_dtype)?)
    }

    /// 🌟 [OFFLOAD-ATTN] 계산 장치로 모은 블록(`parts`)으로 어텐션을 계산합니다.
    /// 출력: `(b, q_heads, q_len, d)` (dtype 은 호출부가 맞춥니다)
    ///
    /// 1. 디코딩이고 모든 블록이 같은 TurboQuant 형식이면: 압축 행을 이어 붙여 fused 커널 한 번.
    /// 2. 그 밖에는: 과거 이력(복원) + 이번 스텝의 원본 K/V → 타일 flash-attn 한 번.
    /// 3. 모을 크기가 [`OFFLOAD_ATTN_GATHER_BUDGET`] 를 넘거나, 1·2 가 실패하거나,
    ///    `LOGIS_KV_BLOCK_ATTN=legacy` 이면: 기존 블록 순회(온라인 소프트맥스).
    #[allow(clippy::too_many_arguments)]
    fn attend_parts(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        parts: &[crate::models::common::kv_turbo::KvPart],
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        total_tokens: usize,
    ) -> Result<Tensor> {
        use crate::models::common::{accel_check, kv_turbo};
        let q_len = query_states.dim(2)?;
        let dev = query_states.device();
        if offload_attn_enabled() {
            let dense_bytes = total_tokens
                * self.num_key_value_heads
                * self.head_dim
                * 2
                * query_states.dtype().size_in_bytes();
            // 이 dtype·head_dim 의 flash-attn 자가 검증. fused 커널을 못 쓰면 candle 의 비융합 경로가
            // 점수 행렬(q_len × 전체 길이) 전체를 만들므로, 긴 프리필은 블록 순회 경로로 메모리를 묶어 둡니다.
            let fa_fused = accel_check::flash_attn_ok(dev, query_states.dtype(), self.head_dim);
            let score_bytes = query_states.dim(0)? * self.num_attention_heads * q_len * total_tokens * 4 * 3;
            let unfused_too_big = !fa_fused && q_len > 1 && score_bytes > OFFLOAD_ATTN_UNFUSED_BUDGET;
            if dense_bytes <= OFFLOAD_ATTN_GATHER_BUDGET && !unfused_too_big {
                let fast = (|| -> Result<Tensor> {
                    if q_len <= 4 {
                        if let Some(o) = kv_turbo::fused_decode_attention(query_states, parts, self.scaling)? {
                            // 🌟 [PRC-11] Qwen3.5 TurboQuant 압축 블록 실측 감시 연결 (장치당 최초 PROBE_LIMIT 회)
                            if accel_check::probe_wanted(dev) && parts.iter().any(|p| p.tq.is_some()) {
                                if let Ok(ref_o) = kv_turbo::flash_history_attention(
                                    query_states,
                                    key_states,
                                    value_states,
                                    parts,
                                    seqlen_offset,
                                    self.scaling,
                                ) {
                                    if let Ok(rel) = accel_check::rel_err(&o, &ref_o) {
                                        let label = format!("qwen3.5-tq {}", kv_turbo::kv_formats_label(dev));
                                        if !accel_check::record_probe(dev, rel, &label) {
                                            return Ok(ref_o);
                                        }
                                    }
                                }
                            }
                            return Ok(o);
                        }
                    }
                    kv_turbo::flash_history_attention(
                        query_states,
                        key_states,
                        value_states,
                        parts,
                        seqlen_offset,
                        self.scaling,
                    )
                })();
                match fast {
                    Ok(o) => return Ok(o),
                    Err(e) => {
                        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            println!(
                                "[OFFLOAD-ATTN] ⚠️ 레이어 {} 를 한 번에 계산하지 못해 블록 순회 경로로 계산합니다: {} (이후 같은 경고는 생략)",
                                self.layer_idx, e
                            );
                        }
                    }
                }
            }
        }

        // 기존 블록 순회 경로 (블록마다 계산 후 온라인 소프트맥스로 병합)
        let q_aligned = query_states.affine(self.scaling, 0.0)?.contiguous()?;
        let mut block_attn = kv_turbo::BlockAttention::new(
            &q_aligned,
            self.num_key_value_heads,
            self.num_kv_groups,
        )?;
        for p in parts {
            let k = p.k.narrow(2, 0, p.len)?;
            let v = p.v.narrow(2, 0, p.len)?;
            match p.tq {
                Some(meta) => match kv_turbo::codec_for(&meta, dev) {
                    Ok(codec) => block_attn.add_packed(&k, &v, &codec, attention_mask, p.off)?,
                    Err(e) => {
                        // 이 빌드가 압축 블록을 풀 수 없으면(turbo-quant 미빌드 등) 그 블록만 건너뜁니다.
                        println!(
                            "[TURBO-KV-BLOCK] 레이어 {} 블록(토큰 {}..) 를 복원할 수 없어 건너뜁니다: {}",
                            self.layer_idx, p.off, e
                        );
                    }
                },
                None => block_attn.add_plain(&k, &v, attention_mask, p.off)?,
            }
        }
        // 👈 F32 정밀도 연산 결과 (호출부가 모델 dtype 으로 변환)
        block_attn.finish()?.ok_or_else(|| anyhow!("No KV data processed"))
    }

    /// 🌟 [P1-2 / P4-O01] 이 어텐션 층의 지정 블록(또는 첫 RAM 상주 블록)을 계산 스트림과 겹쳐
    /// 비동기 DMA(`OffloadPrefetcher`)로 미리 GPU 에 올립니다.
    pub fn prefetch_ram_block(&mut self, block_idx: usize, dev: &Device) {
        if dev.is_cpu() {
            return;
        }
        if self.prefetched_block.as_ref().is_some_and(|(idx, _, _)| *idx == block_idx) {
            return;
        }
        let Some(block) = self.kv_blocks.get(block_idx) else {
            return;
        };
        let (host_kv, tq) = {
            let Ok(inner) = block.inner.read() else { return };
            if inner.location == KVLocation::VRAM {
                return;
            }
            if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                if k.device().is_cpu() && v.device().is_cpu() {
                    (Some((k.clone(), v.clone())), inner.tq)
                } else {
                    (None, inner.tq)
                }
            } else {
                let idx = inner.index;
                drop(inner);
                let Ok(reg) = self.registry.entries.read() else { return };
                if idx >= reg.len() {
                    return;
                }
                let Ok(cache) = reg[idx].bitkv_cache.read() else { return };
                match cache.get(self.layer_idx).and_then(|o| o.as_ref()) {
                    Some(m) if m.k_data.device().is_cpu() && m.v_data.device().is_cpu() => {
                        (Some((m.k_data.clone(), m.v_data.clone())), m.tq)
                    }
                    _ => (None, None),
                }
            }
        };
        let Some((kh, vh)) = host_kv else {
            return;
        };
        let need_bytes = (kh.elem_count() * kh.dtype().size_in_bytes())
            .max(vh.elem_count() * vh.dtype().size_in_bytes())
            .max(crate::utils::prefetch::ASYNC_DMA_MIN_BYTES);
        let need_new = self
            .prefetcher
            .as_ref()
            .is_none_or(|p| p.is_async_capable() && p.capacity_bytes() < need_bytes);
        if need_new {
            if let Ok(p) = crate::utils::prefetch::OffloadPrefetcher::new(dev, need_bytes) {
                self.prefetcher = Some(p);
            }
        }
        if let Some(p) = self.prefetcher.as_mut() {
            if let Ok(pf) = p.prefetch_kv_pair(&kh, &vh) {
                self.prefetched_block = Some((block_idx, pf, tq));
            }
        }
    }

    /// 이 층의 첫 번째 RAM 상주 블록을 선제 업로드합니다 (직전 레이어 계산과 DMA 중첩).
    pub fn prefetch_first_ram_block(&mut self, dev: &Device) {
        if dev.is_cpu() || self.kv_blocks.is_empty() {
            return;
        }
        for bi in 0..self.kv_blocks.len() {
            let is_ram = self.kv_blocks[bi]
                .inner
                .read()
                .map(|g| g.location != KVLocation::VRAM)
                .unwrap_or(false);
            if is_ram {
                self.prefetch_ram_block(bi, dev);
                break;
            }
        }
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_blocks.clear();
        self.prefetched_block = None;
    }

    pub fn capture_kv_prefix(&self, target_dev: &Device) -> Result<Option<FullAttnKvSnapshot>> {
        let mut blocks = Vec::with_capacity(self.kv_blocks.len());
        let mut all_tq = !self.kv_blocks.is_empty();
        let mut all_fp8 = !self.kv_blocks.is_empty();
        for block in &self.kv_blocks {
            let inner = block.inner.read().unwrap();
            if inner.location != KVLocation::VRAM {
                return Ok(None);
            }
            let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) else {
                return Ok(None);
            };
            if inner.tq.is_none() {
                all_tq = false;
            }
            if inner.tq.is_some() || k.dtype() != DType::F8E4M3 {
                all_fp8 = false;
            }
            blocks.push(SavedKvBlock {
                index: inner.index,
                len: inner.len,
                offset: inner.offset,
                kv_exact: self.kv_exact,
                kv_half: self.kv_half,
                k: clone_tensor_to_device(k, target_dev)?,
                v: clone_tensor_to_device(v, target_dev)?,
                tq: inner.tq,
            });
        }
        Ok(Some(if all_tq {
            FullAttnKvSnapshot::TurboRaw(blocks)
        } else if all_fp8 {
            FullAttnKvSnapshot::Fp8Blocks(blocks)
        } else {
            FullAttnKvSnapshot::DenseBlocks(blocks)
        }))
    }

    pub fn restore_kv_prefix_cow(&mut self, snap: &FullAttnKvSnapshot, compute_dev: &Device) -> Result<usize> {
        let blocks = match snap {
            FullAttnKvSnapshot::DenseBlocks(b)
            | FullAttnKvSnapshot::Fp8Blocks(b)
            | FullAttnKvSnapshot::TurboRaw(b) => b,
        };
        self.kv_blocks.clear();
        self.prefetched_block = None;
        let mut restored_tokens = 0usize;
        for b in blocks {
            let block = KVBlock::new(KVLocation::VRAM, b.index, b.len, b.offset);
            {
                let mut inner = block.inner.write().unwrap();
                inner.k_cache = Some(clone_tensor_to_device(&b.k, compute_dev)?);
                inner.v_cache = Some(clone_tensor_to_device(&b.v, compute_dev)?);
                inner.tq = b.tq;
            }
            if let Ok(mut reg) = self.registry.entries.write() {
                if b.index < reg.len() {
                    reg[b.index].token_start = b.offset;
                    reg[b.index].token_len = b.len;
                    if self.layer_idx < reg[b.index].is_dirty.len() {
                        reg[b.index].is_dirty[self.layer_idx] = true;
                    }
                    if self.layer_idx < reg[b.index].location.len() {
                        reg[b.index].location[self.layer_idx] = KVLocation::VRAM;
                    }
                }
            }
            self.kv_exact = b.kv_exact;
            self.kv_half = b.kv_half;
            restored_tokens += b.len;
            self.kv_blocks.push(block);
        }
        Ok(restored_tokens)
    }

    
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        let mut _current_total = 0;
        let mut to_remove = Vec::new();
        let total_blocks = self.kv_blocks.len();
        
        for i in 0..total_blocks {
            let block = &mut self.kv_blocks[i];
            let mut inner = block.inner.write().unwrap();
            
            if _current_total + inner.len <= len {
                _current_total += inner.len;
            } else {
                let keep_in_this_block = len - _current_total;
                if keep_in_this_block > 0 {
                    if inner.location == KVLocation::VRAM {
                        let (new_k, new_v) = if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                            (Some(k.narrow(2, 0, keep_in_this_block)?), Some(v.narrow(2, 0, keep_in_this_block)?))
                        } else { (None, None) };
                        inner.k_cache = new_k;
                        inner.v_cache = new_v;
                    }
                    inner.len = keep_in_this_block;
                    _current_total += keep_in_this_block;
                    for j in (i + 1)..total_blocks { to_remove.push(j); }
                } else {
                    for j in i..total_blocks { to_remove.push(j); }
                }
                break;
            }
        }
        
        to_remove.sort_by(|a, b| b.cmp(a));
        for idx in to_remove { self.kv_blocks.remove(idx); }
        self.prefetched_block = None;

        Ok(())
    }

    pub fn clear_weights(&mut self) {
        let dummy = crate::models::common::gguf::dummy_proj(&Device::Cpu);
        self.q_proj = dummy.clone();
        self.k_proj = dummy.clone();
        self.v_proj = dummy.clone();
        self.o_proj = dummy;
        self.q_norm.clear();
        self.k_norm.clear();
        self.prefetched_block = None;
    }

    pub fn load_weights_inplace(&mut self, ct: &candle_core::quantized::gguf_file::Content, data: &[u8], prefix: &str, device: &Device) -> Result<()> {
        self.q_proj = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.attn_q.weight"), device)?)?, None));
        self.k_proj = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.attn_k.weight"), device)?)?, None));
        self.v_proj = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.attn_v.weight"), device)?)?, None));
        self.o_proj = ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.attn_output.weight"), device)?)?, None));
        
        
        let t_q_norm = crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.attn_q_norm.weight"), device)?;
        let q_norm_w = t_q_norm.dequantize_f16(device).or_else(|_| t_q_norm.dequantize(device))?.to_dtype(DType::F32)?;
        self.q_norm = Qwen3_5RMSNorm::from_weight(q_norm_w, self.q_norm.eps())?;
        
        
        let t_k_norm = crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.attn_k_norm.weight"), device)?;
        let k_norm_w = t_k_norm.dequantize_f16(device).or_else(|_| t_k_norm.dequantize(device))?.to_dtype(DType::F32)?;
        self.k_norm = Qwen3_5RMSNorm::from_weight(k_norm_w, self.k_norm.eps())?;
        Ok(())
    }
}

enum AttnKind {
    LinearAttn(Qwen3_5GatedDeltaNet),
    SelfAttn(Qwen3_5Attention),
}

impl AttnKind {
    fn forward(&mut self, xs: &Tensor, cos: Option<&Tensor>, sin: Option<&Tensor>, attention_mask: Option<&Tensor>, seqlen_offset: usize) -> Result<Tensor> {
        match self {
            AttnKind::LinearAttn(attn) => attn.forward(xs, attention_mask),
            AttnKind::SelfAttn(attn) => {
                if let (Some(c), Some(s)) = (cos, sin) {
                    attn.forward(xs, c, s, attention_mask, seqlen_offset)
                } else {
                    Err(anyhow!("Qwen3_5 self attn cos and sin is all need"))
                }
            }
        }
    }

    pub fn get_ssm_states(&self) -> (Option<Tensor>, Option<Tensor>) {
        if let AttnKind::LinearAttn(attn) = self {
            attn.get_states()
        } else {
            (None, None)
        }
    }

    pub fn set_ssm_states(&mut self, conv: Option<Tensor>, recurrent: Option<Tensor>) {
        if let AttnKind::LinearAttn(attn) = self {
            attn.set_states(conv, recurrent);
        }
    }

    pub fn is_ssm_dirty(&self) -> bool {
        if let AttnKind::LinearAttn(attn) = self {
            attn.is_state_dirty
        } else {
            false
        }
    }

    pub fn set_ssm_dirty(&mut self, dirty: bool) {
        if let AttnKind::LinearAttn(attn) = self {
            attn.is_state_dirty = dirty;
        }
    }

    // 🌟 [KV RESIDENCY] full_attention 레이어에만 배치 위치를 전파합니다.
    //    linear_attention(SSM) 레이어는 고정 크기 state 라 블록 오프로딩 대상이 아닙니다.
    pub fn set_kv_residency(&mut self, residency: crate::utils::resources::KvResidency) {
        if let AttnKind::SelfAttn(attn) = self {
            attn.kv_residency = residency;
        }
    }

    pub fn capture_hybrid_state(&self, target_dev: &Device) -> Result<Option<HybridLayerState>> {
        match self {
            AttnKind::LinearAttn(attn) => {
                let (conv_state, recurrent_state) = attn.capture_state(target_dev)?;
                Ok(Some(HybridLayerState::Linear {
                    conv_state,
                    recurrent_state,
                }))
            }
            AttnKind::SelfAttn(attn) => match attn.capture_kv_prefix(target_dev)? {
                Some(kv) => Ok(Some(HybridLayerState::Full(kv))),
                None => Ok(None),
            },
        }
    }

    pub fn restore_hybrid_state_cow(&mut self, state: &HybridLayerState, compute_dev: &Device) -> Result<()> {
        match (self, state) {
            (
                AttnKind::LinearAttn(attn),
                HybridLayerState::Linear {
                    conv_state,
                    recurrent_state,
                },
            ) => attn.restore_state_cow(conv_state.as_ref(), recurrent_state.as_ref(), compute_dev),
            (AttnKind::SelfAttn(attn), HybridLayerState::Full(kv)) => {
                attn.restore_kv_prefix_cow(kv, compute_dev)?;
                Ok(())
            }
            _ => Err(anyhow!("HybridLayerState variant mismatch during restore")),
        }
    }
}

pub struct Qwen3_5DecoderLayer {
    pub layer_type: String,
    attn: AttnKind,
    mlp: crate::models::common::gguf::GateUpDownMLPGguf,
    input_layernorm: Qwen3_5RMSNorm,
    post_attention_layernorm: Qwen3_5RMSNorm,
}

impl Qwen3_5DecoderLayer {
    pub fn new_from_vb(
        vb: VarBuilder,
        config: &Qwen3_5TextConfig,
        layer_idx: usize,
        registry: KVRegistry, 
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let layer_type = config.layer_types[layer_idx].clone();
        let attn = if layer_type.eq("linear_attention") {
            let attn = Qwen3_5GatedDeltaNet::new_from_vb(vb.pp("linear_attn"), config)?;
            AttnKind::LinearAttn(attn)
        } else {
            let attn = Qwen3_5Attention::new_from_vb(vb.pp("self_attn"), config, layer_idx, registry)?;
            AttnKind::SelfAttn(attn)
        };
        let mlp = GateUpDownMLPGguf::new_from_vb(
            vb.pp("mlp"),
            hidden_size,
            config.intermediate_size,
            false,
            None,
            None,
            None,
            Some(config.hidden_act),
        )?;
        let input_layernorm =
            Qwen3_5RMSNorm::new(vb.pp("input_layernorm"), hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = Qwen3_5RMSNorm::new(
            vb.pp("post_attention_layernorm"),
            hidden_size,
            config.rms_norm_eps,
        )?;
        Ok(Self {
            layer_type,
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    
    pub fn new_skeleton<R: Read + Seek>(
        gguf: &mut Gguf<R>,
        layer_type: &str,
        rms_norm_eps: f64,
        layer_idx: usize,
        registry: KVRegistry,
    ) -> Result<Self> {
        let dummy_p = crate::models::common::gguf::dummy_proj(&Device::Cpu);
        let dummy_t = Tensor::zeros((1,), DType::F32, &Device::Cpu)?;
        let dummy_norm = Qwen3_5RMSNorm { eps: rms_norm_eps, weight: dummy_t.clone() };

        let attn = if layer_type == "linear_attention" {
            let num_k_heads = gguf.get_matedata("qwen35.ssm.group_count")?.to_u32()? as usize;
            let num_v_heads = gguf.get_matedata("qwen35.ssm.time_step_rank")?.to_u32()? as usize;
            let conv_kernel_size = gguf.get_matedata("qwen35.ssm.conv_kernel")?.to_u32()? as usize;
            let head_k_dim = gguf.get_matedata("qwen35.ssm.state_size")?.to_u32()? as usize;
            let key_dim = head_k_dim * num_k_heads;
            let value_dim = head_k_dim * num_v_heads;
            
            AttnKind::LinearAttn(Qwen3_5GatedDeltaNet {
                num_v_heads, num_k_heads, head_k_dim, head_v_dim: head_k_dim, key_dim, value_dim, conv_kernel_size,
                conv1d: candle_nn::Conv1d::new(dummy_t.clone(), None, candle_nn::Conv1dConfig::default()),
                dt_bias: dummy_t.clone(), a_log: dummy_t.clone(),
                norm: Qwen3_5RMSNormGated { weight: dummy_t.clone(), eps: rms_norm_eps, dtype: DType::F32 },
                out_proj: dummy_p.clone(), in_proj_qkv: dummy_p.clone(), in_proj_z: dummy_p.clone(),
                in_proj_b: dummy_p.clone(), in_proj_a: dummy_p.clone(),
                conv_state_cache: None, recurrent_state_cache: None, is_state_dirty: false,
                v_head_order: gguf_v_head_order(gguf),
            })
        } else {
            let num_attention_heads = gguf.get_matedata("qwen35.attention.head_count")?.to_u32()? as usize;
            let num_key_value_heads = gguf.get_matedata("qwen35.attention.head_count_kv")?.to_u32()? as usize;
            let num_kv_groups = num_attention_heads / num_key_value_heads;
            let head_dim = gguf.get_matedata("qwen35.attention.key_length")?.to_u32()? as usize;
            let scaling = 1f64 / f64::sqrt(head_dim as f64);

            AttnKind::SelfAttn(Qwen3_5Attention {
                q_proj: dummy_p.clone(), k_proj: dummy_p.clone(), v_proj: dummy_p.clone(), o_proj: dummy_p.clone(),
                q_norm: dummy_norm.clone(), k_norm: dummy_norm.clone(),
                num_attention_heads, num_key_value_heads, num_kv_groups, head_dim, scaling,
                kv_blocks: Vec::new(), registry, layer_idx, active_session_id: None, active_kv_name: None,
                kv_residency: crate::utils::resources::KvResidency::Ssd,
                prefetcher: None, prefetched_block: None,
                kv_exact: false, kv_half: false,
            })
        };

        
        let mlp = crate::models::common::gguf::GateUpDownMLPGguf::new_dummy(&candle_core::Device::Cpu);

        Ok(Self {
            layer_type: layer_type.to_string(),
            attn, mlp, input_layernorm: dummy_norm.clone(), post_attention_layernorm: dummy_norm,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(
        gguf: &mut Gguf<R>,
        prefix: &str,
        layer_type: &str,
        rms_norm_eps: f64,
        layer_idx: usize, 
        registry: KVRegistry, 
    ) -> Result<Self> {
        let attn = if layer_type.eq("linear_attention") {
            let attn = Qwen3_5GatedDeltaNet::new_from_gguf(gguf, prefix, rms_norm_eps)?;
            AttnKind::LinearAttn(attn)
        } else {
            let attn = Qwen3_5Attention::new_from_gguf(gguf, prefix, rms_norm_eps, layer_idx, registry)?;
            AttnKind::SelfAttn(attn)
        };
        let mlp = GateUpDownMLPGguf::new_from_gguf(
            gguf,
            prefix,
            false,
            None,
            None,
            None,
            Some(candle_nn::Activation::Silu),
        )?;
        
        let input_norm_weight = gguf.get_dequantized_f16(&format!("{prefix}.attn_norm.weight"))?.to_dtype(candle_core::DType::BF16)?;
        let input_layernorm = Qwen3_5RMSNorm::from_weight(input_norm_weight, rms_norm_eps)?;
        let post_norm_weight = gguf.get_dequantized_f16(&format!("{prefix}.post_attention_norm.weight"))?.to_dtype(candle_core::DType::BF16)?;
        let post_attention_layernorm = Qwen3_5RMSNorm::from_weight(post_norm_weight, rms_norm_eps)?;
        Ok(Self {
            layer_type: layer_type.to_string(),
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(&mut self, xs: &Tensor, cos: Option<&Tensor>, sin: Option<&Tensor>, attention_mask: Option<&Tensor>, seqlen_offset: usize) -> Result<Tensor> {
        let residual = xs.clone(); // xs는 BF16
        let mut xs = self.input_layernorm.forward(xs)?;
        xs = self.attn.forward(&xs, cos, sin, attention_mask, seqlen_offset)?;

        let hidden_dim = *residual.dims().last().unwrap_or(&0);
        let (residual, normed_post) = if crate::models::common::accel_check::fused_op_ok(
            residual.device(),
            crate::models::common::accel_check::FusedOpKind::ResidualRmsNorm,
            residual.dtype(),
            hidden_dim,
        ) {
            let delta = if xs.dtype() != residual.dtype() {
                xs.to_dtype(residual.dtype())?
            } else {
                xs
            };
            candle_core::decoder_ops::residual_rmsnorm_inplace(
                residual,
                Some(&delta),
                self.post_attention_layernorm.weight(),
                self.post_attention_layernorm.eps() as f32,
                false,
            )?
        } else {
            // 💡 attn 출력이 F32로 오염되었을 경우를 대비해 BF16으로 내린 뒤 더함
            let r = xs.to_dtype(residual.dtype())?.add(&residual)?;
            let n = self.post_attention_layernorm.forward(&r)?;
            (r, n)
        };

        let mlp_out = self.mlp.forward(&normed_post)?;
        // 💡 mlp 출력(F32 가능성 높음)을 BF16으로 내린 뒤 최종 더하기
        let xs = mlp_out.to_dtype(residual.dtype())?.add(&residual)?;
        Ok(xs)
    }

    pub fn get_ssm_states(&self) -> (Option<Tensor>, Option<Tensor>) {
        self.attn.get_ssm_states()
    }

    pub fn set_ssm_states(&mut self, conv: Option<Tensor>, recurrent: Option<Tensor>) {
        self.attn.set_ssm_states(conv, recurrent);
    }

    pub fn is_ssm_dirty(&self) -> bool {
        self.attn.is_ssm_dirty()
    }

    pub fn set_ssm_dirty(&mut self, dirty: bool) {
        self.attn.set_ssm_dirty(dirty);
    }

    // 🌟 [KV RESIDENCY] 레이어 단위 배치 위치 전파 위임
    pub fn set_kv_residency(&mut self, residency: crate::utils::resources::KvResidency) {
        self.attn.set_kv_residency(residency);
    }

    pub fn capture_hybrid_state(&self, target_dev: &Device) -> Result<Option<HybridLayerState>> {
        self.attn.capture_hybrid_state(target_dev)
    }

    pub fn restore_hybrid_state_cow(&mut self, state: &HybridLayerState, compute_dev: &Device) -> Result<()> {
        self.attn.restore_hybrid_state_cow(state, compute_dev)
    }

    pub fn clear_cache(&mut self) {
        match &mut self.attn {
            AttnKind::LinearAttn(attn) => { attn.clear_cache(); }
            AttnKind::SelfAttn(attn) => { attn.clear_kv_cache(); }
        }
    }

    
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        if let AttnKind::SelfAttn(attn) = &mut self.attn {
            attn.truncate_kv_cache(len)?;
        }
        // SSM (LinearAttn)은 토큰 단위 롤백이 불가하므로 무시합니다.
        Ok(())
    }

    pub fn clear_weights(&mut self) {
        match &mut self.attn {
            AttnKind::LinearAttn(attn) => attn.clear_weights(),
            AttnKind::SelfAttn(attn) => attn.clear_weights(),
        }
        self.mlp.clear_weights();
        self.input_layernorm.clear();
        self.post_attention_layernorm.clear();
    }

    pub fn is_cleared(&self) -> bool { self.input_layernorm.is_cleared() }

    pub fn load_weights_inplace(&mut self, ct: &candle_core::quantized::gguf_file::Content, data: &[u8], prefix: &str, device: &Device) -> Result<()> {
        match &mut self.attn {
            AttnKind::LinearAttn(attn) => attn.load_weights_inplace(ct, data, prefix, device)?,
            AttnKind::SelfAttn(attn) => attn.load_weights_inplace(ct, data, prefix, device)?,
        }
        self.mlp.load_weights_inplace(ct, data, prefix, device)?;
        
        
        let t_in = crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.attn_norm.weight"), device)?;
        let in_norm_w = t_in.dequantize_f16(device).or_else(|_| t_in.dequantize(device))?.to_dtype(DType::F32)?;
        self.input_layernorm = Qwen3_5RMSNorm::from_weight(in_norm_w, self.input_layernorm.eps())?;
        
        
        let t_post = crate::models::common::gguf::qtensor_from_bytes(ct, data, &format!("{prefix}.post_attention_norm.weight"), device)?;
        let post_norm_w = t_post.dequantize_f16(device).or_else(|_| t_post.dequantize(device))?.to_dtype(DType::F32)?;
        self.post_attention_layernorm = Qwen3_5RMSNorm::from_weight(post_norm_w, self.post_attention_layernorm.eps())?;
        Ok(())
    }
}

// ----------------------------------------------------------------------------------
// [Qwen3_5TextModel] SSD 캐싱 관리자 연결
// ----------------------------------------------------------------------------------

/// 🌟 [KV RESIDENCY] 디코딩 진입 시 앞으로 생성할 토큰 수를 알 수 없으므로,
/// 보수적으로 이만큼 더 자란다고 가정하고 VRAM 상주 가능 여부를 판정합니다.
/// (scheduler.rs 의 max_tokens 는 128~512, qwen3_5 generate 기본값은 1024)
pub const QWEN35_DECODE_HEADROOM_TOKENS: usize = 2048;

pub const DECODE_RESIDENT_VRAM_MARGIN_BYTES: u64 = 640 * 1024 * 1024;

/// 🌟 [PRC-04] 스케일 없는 raw FP8 KV 강등은 명시적 옵트인(`LOGIS_RAW_FP8_KV=1`)일 때만 허용합니다.
fn raw_fp8_kv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("LOGIS_RAW_FP8_KV")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

/// 🌟 [SPD-05] 가중치 상주 / 프리필 상주 / keep_warm 판정용 비례 VRAM 여유.
///  기본(v2): `kv_vram_margin_bytes(DECODE_TRANSIENT_BYTES) + DECODE_TRANSIENT_BYTES` (= 128 + 64 = 192 MiB).
///  `LOGIS_KV_PLAN=legacy`: 기존 고정 640 MiB.
pub fn weight_resident_margin_bytes() -> u64 {
    if crate::utils::resources::kv_plan_legacy() {
        DECODE_RESIDENT_VRAM_MARGIN_BYTES
    } else {
        crate::utils::resources::kv_vram_margin_bytes(DECODE_TRANSIENT_BYTES) + DECODE_TRANSIENT_BYTES
    }
}

/// 🌟 [WEIGHT-RESIDENT] `weight_bytes` 는 층 가중치 전체, `missing_bytes` 는 그중 아직 VRAM 에 없는(새로 읽어야 하는) 양입니다.
/// 이미 올라와 있는 층은 VRAM free 에서 빠져 있으므로 다시 요구하지 않습니다.
fn decode_resident_gate(dev: &Device, weight_bytes: u64, missing_bytes: u64, kv_bytes: u64) -> (bool, Option<u64>) {
    if !dev.is_cuda_or_rocm() {
        return ((crate::utils::resources::free_ram_bytes() as u64) > 6_000_000_000, None);
    }
    let free = crate::utils::device_utils::device_vram_snapshot(dev)
        .map(|s| s.effective_free_bytes());
    let margin = weight_resident_margin_bytes();
    match free {
        Some(f) => (
            weight_bytes > 0 && f >= missing_bytes + kv_bytes + margin,
            Some(f),
        ),
        None => (false, None),
    }
}

/// 🌟 [WEIGHT-RESIDENT] 상주 가중치를 둔 채 계산할 때 최소한 남겨 둘 VRAM (KV·활성값 순간치 외 여유).
/// 이보다 모자라면 상주 층을 내려놓고 기존 스트리밍 경로로 돌아갑니다.
pub const RESIDENT_MIN_RESERVE_BYTES: u64 = 256 * 1024 * 1024;

/// 디코딩 한 토큰의 활성값 순간치(대략). 프리필은 [`Qwen3_5TextModel::prefill_transient_bytes`] 로 계산합니다.
pub const DECODE_TRANSIENT_BYTES: u64 = 64 * 1024 * 1024;

/// 🌟 [KV-PLAN v2] Qwen3.5 메모리 계획 상태. `clear_cache` 뒤에도 직전 KV 판정(히스테리시스)은 남고,
/// 생성기 인스턴스가 버려지면(퍼지·작업 종료) 함께 사라집니다.
#[derive(Debug, Clone, Default)]
pub struct Qwen35MemState {
    /// 이번 생성의 최대 새 토큰 수 (generate 가 알려 줌). 0 이면 모름 → [`QWEN35_DECODE_HEADROOM_TOKENS`].
    pub planned_decode_tokens: usize,
    /// 직전 호출의 KV 계층 판정 (계층이 이웃 호출 사이에 뒤집히지 않게 합니다).
    pub last_kv_plan: Option<crate::utils::resources::KvResidency>,
    /// 이번 호출 동안 GDN conv/recurrent 상태를 VRAM 에 둘지. None = 아직 판정 전(둡니다).
    pub ssm_in_vram: Option<bool>,
    /// 원본 KV 를 F16 으로 보관하는지 (F32 모델).
    pub kv_half: bool,
    /// SSD 계획에서 RAM 에 둘 수 있는 KV 바이트 상한 / 지금 RAM 에 있는 KV 바이트.
    pub ram_kv_budget: u64,
    pub ram_kv_bytes: u64,
}

/// `LOGIS_SSM_RESIDENCY=follow_kv` 이면 예전처럼 KV 계획이 Vram 일 때만 GDN 상태를 VRAM 에 두고,
/// 아니면 층마다 CPU 로 내렸다가 다음 토큰에 다시 올립니다.
fn ssm_follows_kv_plan() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LOGIS_SSM_RESIDENCY")
            .map(|v| v.trim().eq_ignore_ascii_case("follow_kv"))
            .unwrap_or(false)
    })
}

/// `LOGIS_KV_DEMOTE=fp8` 이면 원본 KV 블록을 RAM 으로 내릴 때 예전처럼 스케일 없는 FP8 로 바꿉니다.
/// 기본은 TurboQuant 벡터별 스케일 FP8 코덱으로 압축합니다 (정확도가 같거나 높습니다).
fn kv_demote_plain_fp8() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LOGIS_KV_DEMOTE")
            .map(|v| v.trim().eq_ignore_ascii_case("fp8"))
            .unwrap_or(false)
    })
}

/// KV 블록의 K+V 바이트 (텐서가 없으면 0).
fn kv_block_bytes(k: Option<&Tensor>, v: Option<&Tensor>) -> u64 {
    let b = |t: Option<&Tensor>| t.map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64).unwrap_or(0);
    b(k) + b(v)
}

fn env_switch(var: &str, default: bool) -> bool {
    match std::env::var(var) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            if v.is_empty() {
                default
            } else {
                !matches!(v.as_str(), "0" | "false" | "off" | "no")
            }
        }
        Err(_) => default,
    }
}

/// `LOGIS_PREFILL_RESIDENT=0` 이면 프리필은 예전처럼 층을 읽고 바로 비웁니다.
fn prefill_resident_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_switch("LOGIS_PREFILL_RESIDENT", true))
}

/// `LOGIS_KEEP_WEIGHTS_WARM=0` 이면 호출이 끝날 때마다 층 가중치를 내려놓습니다(예전 동작).
fn keep_weights_warm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_switch("LOGIS_KEEP_WEIGHTS_WARM", true))
}

pub struct Qwen3_5TextModel {
    embed_tokens: EmbedTable,
    pub layers: Vec<Qwen3_5DecoderLayer>,
    norm: Qwen3_5RMSNorm,
    rotary_emb: Qwen3VLTextRotaryEmbedding,
    mrope_section: Vec<usize>,
    dtype: DType,
    pub registry: KVRegistry,
    pub current_kv_len: usize,
    pub active_session_id: Option<String>,
    pub active_kv_name: Option<String>,
    // 🌟 [KV RESIDENCY] 디코딩 루프 진입 시 1회만 계산되는 배치 계획.
    //    프리필 구간에서는 None 으로 무효화되어 기존 SSD 오프로딩이 그대로 유지됩니다.
    pub kv_plan: Option<crate::utils::resources::KvResidency>,
    // 🌟 [DECODE-RESIDENT] 가중치 상주 여부도 같은 시점에 1회만 판정하여
    //    토큰마다 반복되던 sysinfo 호출을 제거합니다.
    pub keep_weights_resident: bool,
    pub resident_prefix: usize,
    pub release_streamed_pages: bool,
    pub mmap: Option<std::sync::Arc<memmap2::Mmap>>,
    pub ct: Option<std::sync::Arc<candle_core::quantized::gguf_file::Content>>,
    pub base_name: String,
    /// 🌟 [WEIGHT-RESIDENT] 호출이 끝나도 VRAM 에 올린 층 가중치를 그대로 둘 수 있는 인스턴스인지.
    /// 텍스트 전용 생성기(4B 등)만 켭니다. 비전 타워를 붙였다 떼는 2B 는 예전처럼 호출마다 내려놓습니다.
    /// 생성기 슬롯이 비워지면(퍼지·다른 모델 적재) 인스턴스와 함께 해제됩니다.
    pub keep_warm: bool,
    /// 🌟 [KV-PRECISION] 이번 문맥의 KV 를 원본 정밀도로 두는지 (문맥 첫 프리필에서 정함)
    pub kv_exact: bool,
    /// 🌟 [KV-PRECISION] 바깥 모델이 알려 주는 이번 프롬프트 전체 길이 (청크로 나눠 들어와도 전체 기준으로 판정)
    pub planned_prompt_len: usize,
    /// 🌟 [KV-PLAN v2] 메모리 계획 상태 (최대 새 토큰 수 · 직전 판정 · GDN 상태 상주 · 반정밀도 KV · SSD 계획의 RAM 예산)
    pub mem: Qwen35MemState,
    /// 🌟 [SPD-12] `true` 이면 세션 ID 가 있을 때 프리필 청크마다 디스크 KV 플러시를 수행합니다 (`prefill_only`/`generate_part`/테스트 기본값).
    /// 일회성 `generate()` 호출은 `false` 로 두어 불필요한 SSD 플러시/VRAM→RAM 강등을 생략합니다 (`KvResidency::Ssd` 는 예외).
    pub persist_kv: bool,
}

impl Qwen3_5TextModel {
    pub fn new_from_vb(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let embed_tokens = EmbedTable::Dense(embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?);
        let registry = KVRegistry::new(); // 장부 초기화
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            let mut layer = Qwen3_5DecoderLayer::new_from_vb(vb_layers.pp(i), config, i, registry.clone())?;
            layer.clear_weights(); // 초기화 즉시 1바이트 껍데기로 만들어버림
            layers.push(layer);
        }
        let norm = Qwen3_5RMSNorm::new(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;
        let rope_dim = (config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize;
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(rope_dim, config.rope_parameters.rope_theta);
        Ok(Self {
            embed_tokens, layers, norm, rotary_emb, mrope_section: config.rope_parameters.mrope_section.clone(), dtype: vb.dtype(),
            registry, current_kv_len: 0, active_session_id: None, active_kv_name: None,
            kv_plan: None, keep_weights_resident: false, resident_prefix: 0, release_streamed_pages: false,
            mmap: None, ct: None, base_name: "model".to_string(), keep_warm: false, kv_exact: false, planned_prompt_len: 0, mem: Qwen35MemState::default(),
            persist_kv: true,
        })
    }

    pub fn new_from_gguf<R: Read + Seek>(
        gguf: &mut Gguf<R>, 
        device: &Device,
        mmap_handle: Option<std::sync::Arc<memmap2::Mmap>>,
        ct_handle: Option<std::sync::Arc<candle_core::quantized::gguf_file::Content>>
    ) -> Result<Self> {
        let dtype = match gguf.get_matedata("general.dtype") {
            Ok(v) => match v.to_u32() as Result<u32, candle_core::Error> { Ok(0) => DType::F32, Ok(1) => DType::F16, _ => DType::F16 },
            Err(_) => DType::F16,
        };
        let dtype = if device.is_vulkan()
            && std::env::var("LOGIS_VK_DTYPE").map(|v| !v.trim().eq_ignore_ascii_case("f16")).unwrap_or(true)
        {
            DType::F32
        } else {
            dtype
        };
        let num_layers = gguf.get_matedata("qwen35.block_count")?.to_u32()? as usize;
        let full_attention_interval = gguf.get_matedata("qwen35.full_attention_interval")?.to_u32()? as usize;
        let rope_freq_base = gguf.get_matedata("qwen35.rope.freq_base")?.to_f32()?;
        let rope_dimension_count = gguf.get_matedata("qwen35.rope.dimension_count")?.to_u32()? as usize;
        let mut mrope_section = gguf.get_matedata("qwen35.rope.dimension_sections")?.to_vec()?.iter().map(|v: &candle_core::quantized::gguf_file::Value| v.to_i32().map(|x| x as usize)).collect::<Result<Vec<usize>, candle_core::Error>>()?;
        let _ = mrope_section.pop();
        let rms_norm_eps = gguf.get_matedata("qwen35.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let hidden_size = gguf.get_matedata("qwen35.embedding_length")?.to_u32()? as usize;
        
        // 🌟 [EMBED SCOPE] 양자화 원본과 역양자화본이 동시에 사는 구간을 최소화합니다.
        //
        //  ── 실측 사고 ──
        //   구버전은 embed_tensor(quantized, GPU 약 540MB)를 함수 스코프 끝까지
        //   붙들고 있었습니다. 그래서 그 뒤의 레이어 스켈레톤 생성과
        //   output_norm 로드가 전부 540MB + 1,017MB = 1,557MB 를 깔고 진행됐습니다.
        //   블록으로 감싸 역양자화 직후 원본을 떨어뜨리면
        //   이후 구간의 기준선이 1,017MB 로 내려갑니다.
        //
        //  ⚠️ CPU 역양자화로 바꾸지 않은 이유:
        //     이 환경의 실측 free RAM 은 592MB~1.3GB 입니다(로그 KV-PLAN 참조).
        //     CPU 경로는 호스트에 540MB + 1,017MB 를 동시에 요구하므로
        //     VRAM 을 아끼려다 시스템 RAM 을 터뜨립니다. GPU 역양자화가 옳습니다.
        // 🌟 [VOCAB TRIM] 잘라 내기가 켜지면(기본 full) 입력 임베딩은 매핑에서 행만 읽고 출력 투영은 남긴 행만 장치에 둡니다.
        let mapped = crate::models::common::vocab_trim::mapped_table(device, mmap_handle.as_ref(), ct_handle.as_deref());
        let quant_embed = if mapped.is_some() {
            None
        } else if use_quant_embed(device) {
            let q = gguf.tensor("token_embd.weight")?;
            if quant_embed_ok(q.dtype()) && quant_embed_self_check(device, &q) {
                Some(q)
            } else {
                None
            }
        } else {
            None
        };
        let embed_tokens = if let Some(src) = mapped {
            println!("[VOCAB-TRIM] 입력 임베딩 {} × {} 을 장치에 올리지 않고 매핑에서 행만 읽습니다.", src.rows, src.cols);
            EmbedTable::Mapped { src, device: device.clone() }
        } else if let Some(q) = quant_embed {
            EmbedTable::Quant { q: std::sync::Arc::new(q), hidden: hidden_size }
        } else {
            let embed_dtype = if device.is_cpu() { DType::F32 } else { DType::F16 };
            let embed_weight = {
                let embed_tensor = gguf.tensor("token_embd.weight")?;
                let w = embed_tensor
                    .dequantize_f16(device)
                    .or_else(|_| embed_tensor.dequantize(device))?
                    .to_dtype(embed_dtype)?;
                drop(embed_tensor);
                w
            };
            if device.is_cuda_or_rocm() {
                // 방금 떨어뜨린 양자화 원본을 할당자가 실제로 반환하도록 경계를 만듭니다.
                let _ = device.synchronize();
            }
            println!(
                "[MODEL] embed_tokens materialized: {:?} {:?} ({:.0}MB). Quantized source released.",
                embed_weight.shape().dims(),
                embed_weight.dtype(),
                (embed_weight.elem_count() as f64) * 2.0 / 1e6
            );
            EmbedTable::Dense(Embedding::new(embed_weight, hidden_size))
        };
        
        
        crate::models::common::legacy_working_set_trim();

        let registry = KVRegistry {
            entries: std::sync::Arc::new(std::sync::RwLock::new(
                (0..128)
                    .map(|i| crate::models::qwen::quantized_model::RegistryEntry::new(i * 1024, 0, num_layers.max(28)))
                    .collect(),
            )),
        };
        let mut layers = vec![];
        for i in 0..num_layers {
            let layer_type = if (i + 1) % full_attention_interval == 0 { "full_attention".to_string() } else { "linear_attention".to_string() };
            
            let layer = Qwen3_5DecoderLayer::new_skeleton(gguf, &layer_type, rms_norm_eps, i, registry.clone())?;
            layers.push(layer);
        }
        
        
        let norm_weight = gguf.get_dequantized_f16("output_norm.weight").or_else(|_| gguf.get_dequantized("output_norm.weight"))?.to_dtype(candle_core::DType::F32)?;
        let norm = Qwen3_5RMSNorm::from_weight(norm_weight, rms_norm_eps)?;
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(rope_dimension_count, rope_freq_base);

        Ok(Self {
            embed_tokens, layers, norm, rotary_emb, mrope_section, dtype,
            registry, current_kv_len: 0, active_session_id: None, active_kv_name: None,
            kv_plan: None, keep_weights_resident: false, resident_prefix: 0, release_streamed_pages: false,
            mmap: mmap_handle, ct: ct_handle, base_name: "model".to_string(), keep_warm: false, kv_exact: false, planned_prompt_len: 0, mem: Qwen35MemState::default(),
            persist_kv: true,
        })
    }

    // 찰나의 순간에 디스크에서 가중치를 퍼 올리는 JIT 로더
    pub fn reload_layer(&mut self, layer_idx: usize, device: &Device) -> Result<()> {
        if let (Some(mmap), Some(ct)) = (self.mmap.as_ref(), self.ct.as_ref()) {
            // Qwen 3.5 gguf 블록 접두어
            let prefix = format!("blk.{layer_idx}");
            // 🌟 [MMAP-LOAD] 맵의 바이트를 그대로 넘겨 텐서마다 힙에 복사하지 않고 장치로 올립니다.
            self.layers[layer_idx].load_weights_inplace(ct, &mmap[..], &prefix, device)?;
        }
        Ok(())
    }

    // 메모리는 유지하되 PCIe & SSD 병목은 완벽 차단한 궁극의 파이프라인
    pub async fn forward(&mut self, inputs_embeds: &Tensor, position_ids: &Tensor, seqlen_offset: usize, session_id: Option<String>, kv_name: Option<String>) -> Result<Tensor> {
        self.active_session_id = session_id.clone();
        self.active_kv_name = kv_name.clone();

        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let is_decoding = seq_len <= 1; // 현재 AI가 대답 중인지 판별

        let (cos, sin) = self.rotary_emb.forward(position_ids, self.dtype, self.mrope_section.clone())?;
        
        let mut xs = if !inputs_embeds.device().same_device(&cos.device()) { 
            inputs_embeds.to_device(&cos.device())? 
        } else { 
            inputs_embeds.clone() 
        };
        xs = xs.to_dtype(self.dtype)?; 

        let attention_mask: Option<Tensor> = { 
            if seq_len <= 1 { None } 
            else { Some(prepare_causal_attention_mask(b_size, seq_len, seqlen_offset, xs.device())?) } 
        };
        let attention_mask = if let Some(m) = attention_mask { Some(m.to_dtype(self.dtype)?) } else { None };

        let total_layers = self.layers.len();

        let mut resident_decided = false;
        let kv_residency = if is_decoding {
            if self.kv_plan.is_none() {
                let (full_attn_layers, kv_heads, kv_head_dim) = self.kv_plan_geometry();
                let legacy_plan = crate::utils::resources::kv_plan_legacy();
                // 🌟 [KV-PLAN v2] 여유 토큰은 이번 생성의 실제 max_tokens(상한 2048), 이미 VRAM 에 만든 프롬프트 KV 는 다시 요구하지 않습니다.
                let headroom = if legacy_plan { QWEN35_DECODE_HEADROOM_TOKENS } else { self.decode_headroom_tokens() };
                let planned = seqlen_offset + seq_len + headroom;
                let already = if legacy_plan { 0 } else { self.vram_kv_tokens().min(seqlen_offset) };
                let label = if crate::lang_resident::resident_variant().is_some() { "Qwen3.5-4B" } else { "Qwen3.5(2B)" };
                let mut per_token = if legacy_plan {
                    (full_attn_layers * kv_heads * kv_head_dim * 2 * self.kv_elem_bytes()) as u64
                } else {
                    self.kv_bytes_per_token_now(xs.device())
                };
                // 🌟 [WEIGHT-RESIDENT] 동기화로 메모리 풀이 쥐고 있던 프리필 순간치를 돌려준 뒤 VRAM 을 잽니다.
                //    (동기화 없이 재면 이미 끝난 프리필 활성값까지 '사용 중' 으로 잡혀 판정이 실제보다 비관적입니다)
                //    상주 층 때문에 디코딩 여유조차 모자라면 여기서 상주 층을 내려놓습니다.
                let mut kv_need = per_token * planned.saturating_sub(already) as u64;
                let _ = self.ensure_resident_reserve(xs.device(), kv_need + DECODE_TRANSIENT_BYTES);
                let plan_input = |bpe: usize| crate::utils::resources::KvPlanInput {
                    gpu_id: crate::utils::resources::primary_gpu_id(),
                    is_cpu_mode: xs.device().is_cpu(),
                    num_kv_layers: full_attn_layers,
                    num_kv_heads: kv_heads,
                    head_dim: kv_head_dim,
                    // 🌟 [KV-PRECISION] 원본 KV 면 보관 dtype 크기, 압축이면 FP8(1바이트) 기준 (예전 판정용)
                    bytes_per_elem: bpe,
                    planned_tokens: planned,
                    label,
                };
                let prev_plan = self.mem.last_kv_plan;
                let extra = |bpt: u64| crate::utils::resources::KvPlanExtra {
                    already_tokens: already,
                    decode_transient_bytes: DECODE_TRANSIENT_BYTES,
                    bytes_per_token: Some(bpt),
                    prev: prev_plan,
                };
                let mut plan = crate::utils::resources::plan_kv_residency_ex(&plan_input(self.kv_elem_bytes()), &extra(per_token));
                // 🌟 [KV-PLAN v2] 원본 KV 가 VRAM 에 안 들어가면 RAM 으로 내리기 전에 VRAM 안에서 벡터별 FP8 로 압축해 봅니다.
                //    (예전에는 RAM 으로 내리면서 스케일 없는 FP8 로 캐스팅했으므로 정확도가 같거나 높습니다)
                if !legacy_plan && self.kv_exact && plan != crate::utils::resources::KvResidency::Vram && xs.device().is_cuda_or_rocm() {
                    let packed = self.recompress_vram_kv(xs.device());
                    if packed > 0 {
                        per_token = self.kv_bytes_per_token_now(xs.device());
                        kv_need = per_token * planned.saturating_sub(already) as u64;
                        plan = crate::utils::resources::plan_kv_residency_ex(&plan_input(self.kv_elem_bytes()), &extra(per_token));
                        println!(
                            "[KV-PLAN] {} | 원본 KV 블록 {}개를 VRAM 안에서 벡터별 FP8 로 압축했습니다 → {:?}",
                            label, packed, plan
                        );
                    }
                }
                self.mem.last_kv_plan = Some(plan);
                // 🌟 [SSM-RESIDENT] GDN 상태는 프리필에서 이미 VRAM 에 있습니다. KV 계층과 무관하게 그대로 두고,
                //    디코딩 순간치조차 못 남길 만큼 VRAM 이 모자랄 때만 CPU 로 내립니다.
                if !ssm_follows_kv_plan() && xs.device().is_cuda_or_rocm() {
                    let free = crate::utils::device_utils::device_vram_snapshot(xs.device())
                        .map(|s| s.effective_free_bytes());
                    let keep_ssm = free.map_or(true, |f| f >= crate::utils::resources::KV_VRAM_MARGIN_MIN_BYTES);
                    if self.mem.ssm_in_vram != Some(keep_ssm) {
                        println!(
                            "[SSM-RESIDENT] {} | GDN 상태 {:.0} MB → {} (VRAM free {})",
                            label,
                            self.ssm_state_bytes() as f64 / 1e6,
                            if keep_ssm { "VRAM 유지 (KV 계층과 무관, 토큰마다 CPU 왕복 없음)" } else { "CPU 대피 (VRAM 부족)" },
                            free.map(|f| format!("{:.0} MB", f as f64 / 1e6)).unwrap_or_else(|| "?".to_string())
                        );
                    }
                    self.mem.ssm_in_vram = Some(keep_ssm);
                }
                // 🌟 [KV-SSD-LRU] SSD 계획이어도 RAM 예산 안의 블록은 RAM 에 두고, 넘칠 때만 오래된 블록부터 SSD 로 내립니다.
                if plan == crate::utils::resources::KvResidency::Ssd {
                    let ram_free = crate::utils::resources::free_ram_bytes();
                    let margin = crate::utils::resources::kv_ram_margin_bytes(crate::utils::resources::total_ram_bytes());
                    self.mem.ram_kv_budget = ram_free.saturating_sub(margin);
                    self.mem.ram_kv_bytes = self.ram_kv_bytes_now();
                }
                let weight_bytes = self.decode_resident_bytes();
                let kv_bytes = kv_need;
                // 🌟 [WEIGHT-RESIDENT] 이미 VRAM 에 있는 층(프리필에서 올린 층, 직전 호출에서 남긴 층)은 다시 요구하지 않습니다.
                let missing_bytes = self.missing_layer_bytes();
                let res_margin = weight_resident_margin_bytes();
                let (mut keep, vram_free) =
                    decode_resident_gate(xs.device(), weight_bytes, missing_bytes, kv_bytes);
                let (mut prefix, _prefix_load_bytes) = match vram_free {
                    Some(free) if !keep => self.decode_resident_prefix(
                        free.saturating_sub(kv_bytes + res_margin + self.max_layer_bytes()),
                    ),
                    _ => (0, 0),
                };
                // 🌟 [DECODE-RESIDENT] 앞쪽 상주 층 수가 전체와 같으면 '전부 상주' 입니다 (예전 로그: Partial 32/32).
                if !keep && prefix >= total_layers && total_layers > 0 {
                    keep = true;
                    prefix = 0;
                }
                let prefix_bytes: u64 = (0..prefix).map(|i| self.layer_bytes(i)).sum();
                let streamed_bytes = if keep { 0 } else { weight_bytes.saturating_sub(prefix_bytes) };
                let ram_free = crate::utils::resources::free_ram_bytes() as u64;
                let release_streamed =
                    streamed_bytes > 0 && ram_free < crate::utils::resources::RAM_PRESSURE_MARGIN_BYTES + streamed_bytes;
                self.keep_weights_resident = keep;
                self.resident_prefix = prefix;
                self.release_streamed_pages = release_streamed;
                resident_decided = keep || prefix > 0;
                let variant = if label == "Qwen3.5(2B)" { "q35_2b" } else { "lang4b" };
                crate::utils::score_dynamics::record_baseline(
                    &format!("model.decode_resident_ratio.{}", variant),
                    if keep { 1.0 } else { prefix as f32 / total_layers.max(1) as f32 },
                );
                crate::utils::score_dynamics::record_baseline(
                    &format!("model.decode_entry_ram_free_mb.{}", variant),
                    (ram_free / 1_000_000) as f32,
                );
                let page_mode = if release_streamed { "켬" } else { "끔" };
                match vram_free {
                    Some(free) => println!(
                        "[DECODE-RESIDENT] {} | 층 가중치 {:.0} MB (이미 VRAM 상주 {:.0} MB · 새로 읽을 {:.0} MB) + KV {:.0} MB + margin {:.0} MB vs VRAM free {:.0} MB → {}",
                        label,
                        weight_bytes as f64 / 1e6,
                        weight_bytes.saturating_sub(missing_bytes) as f64 / 1e6,
                        missing_bytes as f64 / 1e6,
                        kv_bytes as f64 / 1e6,
                        res_margin as f64 / 1e6,
                        free as f64 / 1e6,
                        if keep {
                            "Vram (디코딩 동안 층 가중치를 VRAM 에 두고 토큰마다 mmap 에서 다시 읽지 않습니다)".to_string()
                        } else if prefix > 0 {
                            format!(
                                "Partial (층 {}/{}개 {:.0} MB 를 VRAM 에 두고 나머지 {}개 층 {:.0} MB 만 토큰마다 mmap 에서 읽습니다 · 읽은 층의 파일 페이지 반환 {} · RAM 여유 {:.0} MB)",
                                prefix,
                                total_layers,
                                prefix_bytes as f64 / 1e6,
                                total_layers - prefix,
                                streamed_bytes as f64 / 1e6,
                                page_mode,
                                ram_free as f64 / 1e6
                            )
                        } else {
                            format!(
                                "Stream (토큰마다 mmap 에서 층을 다시 읽는 기존 경로 · 읽은 층의 파일 페이지 반환 {} · RAM 여유 {:.0} MB)",
                                page_mode,
                                ram_free as f64 / 1e6
                            )
                        }
                    ),
                    None => println!(
                        "[DECODE-RESIDENT] {} | {} → {} (읽은 층의 파일 페이지 반환 {} · RAM 여유 {:.0} MB)",
                        label,
                        if xs.device().is_cuda_or_rocm() { "VRAM 조회 실패" } else { "CUDA 밖 장치는 기존 기준(여유 RAM 6GB 초과)" },
                        if keep { "Resident" } else { "Stream" },
                        page_mode,
                        ram_free as f64 / 1e6
                    ),
                }
                self.kv_plan = Some(plan);
                for layer in self.layers.iter_mut() {
                    layer.set_kv_residency(plan);
                }
            }
            self.kv_plan.unwrap_or(crate::utils::resources::KvResidency::Ssd)
        } else {
            self.kv_plan = None;
            // 🌟 [WEIGHT-RESIDENT] 프리필에서 읽은 층을 계산 뒤에도 VRAM 에 둘지 디코딩과 같은 기준으로 정합니다.
            //    통과하면 디코딩 첫 토큰에서 같은 층을 mmap 에서 다시 읽지 않습니다(호출당 층 가중치 전송 2회 → 1회,
            //    이전 호출에서 남긴 층이 있으면 0회). 통과하지 못하면 예전처럼 읽고 바로 비웁니다.
            let (keep, prefix) = self.plan_prefill_residency(xs.device(), seq_len, seqlen_offset);
            self.keep_weights_resident = keep;
            self.resident_prefix = prefix;
            resident_decided = keep || prefix > 0;
            if seqlen_offset == 0 {
                self.plan_kv_precision(xs.device(), seq_len, keep, prefix);
            }
            let streamed_bytes: u64 = if keep {
                0
            } else {
                (prefix..total_layers).map(|i| self.layer_bytes(i)).sum()
            };
            self.release_streamed_pages = streamed_bytes > 0
                && (crate::utils::resources::free_ram_bytes() as u64)
                    < crate::utils::resources::RAM_PRESSURE_MARGIN_BYTES + streamed_bytes;
            for layer in self.layers.iter_mut() {
                layer.set_kv_residency(crate::utils::resources::KvResidency::Ssd);
            }
            crate::utils::resources::KvResidency::Ssd
        };

        let mut keep_resident = self.keep_weights_resident;
        let mut resident_prefix = self.resident_prefix;
        let release_streamed = self.release_streamed_pages;
        // 🌟 [WEIGHT-RESIDENT] 이번 forward 에서 새로 올려 상주시킨 층 수 (이미 상주하던 층은 세지 않습니다)
        let mut newly_resident = 0usize;
        for l_idx in 0..total_layers {
            if self.layers[l_idx].is_cleared() {
                if keep_resident || l_idx < resident_prefix {
                    newly_resident += 1;
                }
                if let Err(e) = self.reload_layer(l_idx, xs.device()) {
                    if !keep_resident && resident_prefix == 0 {
                        return Err(e);
                    }
                    for j in 0..total_layers {
                        if j != l_idx {
                            self.layers[j].clear_weights();
                        }
                    }
                    keep_resident = false;
                    resident_prefix = 0;
                    resident_decided = false;
                    self.keep_weights_resident = false;
                    self.resident_prefix = 0;
                    println!(
                        "[DECODE-RESIDENT] 층 {} 를 VRAM 에 상주시키다 실패했습니다 ({}). 이미 올린 층을 비우고 이번 생성의 남은 토큰은 토큰마다 mmap 에서 다시 읽는 기존 경로로 진행합니다.",
                        l_idx, e
                    );
                    if xs.device().is_cuda_or_rocm() {
                        let _ = xs.device().synchronize();
                    }
                    self.reload_layer(l_idx, xs.device())?;
                }
                if release_streamed || keep_resident || l_idx < resident_prefix {
                    self.release_layer_pages(l_idx);
                }
            }
            // 🌟 [PREFETCH] 다음 층을 다시 읽어야 하면(스트리밍) 이 층을 계산하는 동안 OS 가 그 파일 페이지를 미리 읽게 합니다.
            if l_idx + 1 < total_layers && self.layers[l_idx + 1].is_cleared() {
                self.prefetch_layer_pages(l_idx + 1);
            }
            // 🌟 [P1-2 / P4-O01] 다음 층이 SelfAttn 이고 RAM 상주 KV 블록을 가지고 있으면 이 층 계산과 중첩해 비동기 H2D 프리페치
            if l_idx + 1 < total_layers {
                if let AttnKind::SelfAttn(next_attn) = &mut self.layers[l_idx + 1].attn {
                    next_attn.prefetch_first_ram_block(xs.device());
                }
            }
            let layer = &mut self.layers[l_idx];
            if let AttnKind::SelfAttn(attn) = &mut layer.attn {
                attn.active_session_id = session_id.clone();
                attn.active_kv_name = kv_name.clone();
            }
            if layer.layer_type == "linear_attention" && seqlen_offset > 0 {
                let (conv_opt, rec_opt) = layer.get_ssm_states();
                if conv_opt.is_none() || rec_opt.is_none() {
                    if let Some(sid) = &session_id {
                        let kv_dir = crate::utils::paths::get_kv_dir();
                        
                        let kv_name_raw = kv_name.as_deref().unwrap_or("text");
                        let mut safe_kv_type = kv_name_raw.split('/').last().unwrap_or("text");
                        if safe_kv_type == "inference" || safe_kv_type == "reference" || safe_kv_type.is_empty() {
                            safe_kv_type = "text";
                        }
                        
                        let path_candidates = vec![
                            kv_dir.join(format!("{}/inference/{}/ssm", sid, safe_kv_type)),
                            kv_dir.join(format!("{}/reference/{}/ssm", sid, safe_kv_type)),
                            kv_dir.join(format!("{}/inference/text/ssm", sid)), // 최후의 폴백
                        ];

                        let mut found_file = false;
                        
                        for ssm_dir in path_candidates {
                            let st_path = ssm_dir.join(format!("l{}.st", l_idx));
                            if st_path.exists() {
                                found_file = true;
                                let mut loaded = false;
                                for _retry in 0..3 {
                                    if let Ok(encrypted_data) = std::fs::read(&st_path) {
                                        if let Ok(plain_data) = crate::utils::crypto::decrypt_data(&encrypted_data) {
                                            // Device 강제 할당 제거: CPU에서 풀어서 안전하게 GPU로 넘깁니다.
                                            if let Ok(tensors) = candle_core::safetensors::load_buffer(&plain_data, &candle_core::Device::Cpu) {
                                                let loaded_conv = tensors.get("conv_state").map(|t| t.to_device(xs.device()).unwrap_or(t.clone()));
                                                let loaded_rec = tensors.get("recurrent_state").map(|t| t.to_device(xs.device()).unwrap_or(t.clone()));
                                                if loaded_conv.is_some() && loaded_rec.is_some() {
                                                    layer.set_ssm_states(loaded_conv, loaded_rec);
                                                    loaded = true;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    std::thread::sleep(std::time::Duration::from_millis(5));
                                }
                                if loaded { break; }
                            }
                        }
                        if !found_file {
                            println!("[WARNING] SSM state file completely missing for Layer {} (Session: {})! Model context destroyed.", l_idx, sid);
                        }
                    }
                }
            }

            let layer_mask = if layer.layer_type.ne("linear_attention") || (seq_len != 1 && b_size != 1) { attention_mask.clone() } else { None };
            
            xs = layer.forward(&xs, Some(&cos), Some(&sin), layer_mask.as_ref(), seqlen_offset)?;

            
            // 🌟 [KV RESIDENCY - SSM] SSM state 는 토큰 수와 무관한 고정 크기(레이어 합계 약 50~100MB)입니다.
            //    VRAM 상주가 확정된 디코딩 구간에서는 CPU 대피를 건너뛰어
            //    매 토큰 27개 레이어 × CPU↔GPU 왕복을 통째로 제거합니다.
            //    🌟 [SSM-RESIDENT] 기본은 KV 계층과 무관하게 이번 호출 동안 VRAM 에 둡니다
            //    (`LOGIS_SSM_RESIDENCY=follow_kv` 이면 예전 규칙).
            let ssm_keep_in_vram = if ssm_follows_kv_plan() {
                is_decoding && kv_residency == crate::utils::resources::KvResidency::Vram
            } else {
                !xs.device().is_cpu() && self.mem.ssm_in_vram.unwrap_or(true)
            };

            if layer.layer_type == "linear_attention" && !ssm_keep_in_vram {
                let is_dirty_backup = layer.is_ssm_dirty(); 
                
                let (conv_opt, rec_opt) = layer.get_ssm_states();
                if let (Some(conv), Some(rec)) = (conv_opt, rec_opt) {
                    let safe_conv = conv.to_device(&candle_core::Device::Cpu).unwrap_or(conv);
                    let safe_rec = rec.to_device(&candle_core::Device::Cpu).unwrap_or(rec);
                    layer.set_ssm_states(Some(safe_conv), Some(safe_rec));
                }

                layer.set_ssm_dirty(is_dirty_backup); 
            }
            // [DECODE-RESIDENT] 디코딩 시 매 토큰 mmap→heap 가중치 복사(레이어 수 × 토큰 수)를 제거합니다.
            if self.ct.is_some() && self.mmap.is_some() && !keep_resident && l_idx >= resident_prefix {
                layer.clear_weights();
            }

            
            // 🌟 [KV RESIDENCY] 디코딩 진입 시 1회 결정된 배치 위치에 따라 과거 블록을 처리합니다.
            //   - Vram : 아무것도 하지 않고 과거 블록을 VRAM 에 그대로 상주시킵니다. (SSD/RAM 왕복 0회)
            //   - Ram  : VRAM → RAM 대피까지만 수행하고 SSD 강등은 하지 않습니다.
            //            Qwen3_5Attention::forward 가 KVLocation::RAM 블록을 to_device(dev) 로
            //            잠깐 올려 쓰는 경로를 이미 갖고 있으므로 추가 구현 없이 동작합니다.
            //   - Ssd  : 기존과 100% 동일하게 VRAM → RAM → SSD 로 연속 강등합니다.
            if is_decoding && kv_residency != crate::utils::resources::KvResidency::Vram {
                let demote_to_ssd = kv_residency == crate::utils::resources::KvResidency::Ssd;

                if let AttnKind::SelfAttn(attn) = &mut layer.attn {
                    let mut reg = attn.registry.entries.write().unwrap();
                    let total_blocks = attn.kv_blocks.len();
                    let demote_head_dim = attn.head_dim;
                    // 🌟 [KV-DEMOTE] 원본 블록을 내릴 때 쓸 벡터별 FP8 코덱 (처음 필요할 때 한 번 정합니다)
                    let mut demote_codec: Option<Option<std::sync::Arc<crate::models::common::kv_turbo::TqBlockCodec>>> = None;
                    let legacy_ssd = crate::utils::resources::kv_plan_legacy();
                    
                    let mut garbage_bin = Vec::new();
                    
                    for (idx, block) in attn.kv_blocks.iter_mut().enumerate() {
                        if idx + 1 >= total_blocks { continue; } 
                        
                        let mut inner = block.inner.write().unwrap();

                        // 🌟 [VRAM 즉시 반환 (Early Offloading) - Qwen 3.5]
                        // 과거 블록이 VRAM에 남아 OS에 80%~85% 메모리를 점유하는 현상을 끊어냅니다.
                        // 브라우저 켤 때 65%로 떨어지던 그 메모리 반환 효과를 디코딩 루프마다 선제적으로 트리거합니다.
                        if inner.location == KVLocation::VRAM {
                            let k = inner.k_cache.take();
                            let v = inner.v_cache.take();
                            if let (Some(k_t), Some(v_t)) = (k, v) {
                                if inner.tq.is_some() {
                                    // 🌟 [TURBO-KV-BLOCK] 압축 블록은 형 변환 없이 그대로 RAM 으로 내립니다.
                                    inner.k_cache = Some(k_t.to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| k_t.clone()));
                                    inner.v_cache = Some(v_t.to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| v_t.clone()));
                                } else {
                                // 🌟 [KV-DEMOTE] 원본 블록은 TurboQuant 벡터별 FP8 로 압축해 내립니다 (스케일 없는 FP8 캐스팅보다 정확).
                                //    코덱이 없거나(CPU·미빌드·4bit 정책) `LOGIS_KV_DEMOTE=fp8` 이면 예전 경로입니다.
                                let codec = if kv_demote_plain_fp8() || !k_t.device().is_cuda_or_rocm() || k_t.dtype() == candle_core::DType::F8E4M3 {
                                    None
                                } else {
                                    demote_codec
                                        .get_or_insert_with(|| {
                                            if crate::models::common::kv_turbo::kv_is_compact(k_t.device()) {
                                                None
                                            } else {
                                                crate::models::common::kv_turbo::block_codec(demote_head_dim, k_t.device())
                                            }
                                        })
                                        .clone()
                                };
                                let packed = codec.as_ref().and_then(|c| match (c.pack_k(&k_t), c.pack_v(&v_t)) {
                                    (Ok(pk), Ok(pv)) => match (pk.to_device(&candle_core::Device::Cpu), pv.to_device(&candle_core::Device::Cpu)) {
                                        (Ok(pk), Ok(pv)) => Some((pk, pv, c.meta)),
                                        _ => None,
                                    },
                                    _ => None,
                                });
                                if let Some((pk, pv, meta)) = packed {
                                    inner.k_cache = Some(pk);
                                    inner.v_cache = Some(pv);
                                    inner.tq = Some(meta);
                                } else {
                                let target_dtype = if k_t.device().is_cuda_or_rocm() || k_t.dtype() == candle_core::DType::F8E4M3 { candle_core::DType::F8E4M3 } else { candle_core::DType::F32 };
                                inner.k_cache = Some(k_t.to_dtype(target_dtype).unwrap_or_else(|_| k_t.clone()).to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| k_t.clone()));
                                inner.v_cache = Some(v_t.to_dtype(target_dtype).unwrap_or_else(|_| v_t.clone()).to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| v_t.clone()));
                                }
                                }
                                self.mem.ram_kv_bytes += kv_block_bytes(inner.k_cache.as_ref(), inner.v_cache.as_ref());
                                inner.location = KVLocation::RAM;
                                if idx < reg.len() { reg[idx].location[l_idx] = KVLocation::RAM; }
                            }
                        }

                        // 🌟 [RAM 상주 유지] RAM 여유가 확인된 경우에는 여기서 멈춥니다.
                        //    기존 코드는 RAM 을 경유지로만 쓰고 즉시 SSD 로 버렸기 때문에
                        //    다음 토큰마다 SSD 재읽기 경로를 반드시 다시 타야 했습니다.
                        // 🌟 [KV-REGISTRY] SSD 경로가 있어도 이 블록이 아직 기록되지 않았으면(is_dirty) 버리지 않습니다.
                        //    (이전 문맥이 남긴 경로·늦게 끝난 기록 작업이 경로를 채운 경우 다른 문맥의 파일을 읽게 됩니다)
                        //    🌟 [KV-SSD-LRU] RAM 예산 안이면 다음 토큰이 또 읽을 블록이므로 버리지 않습니다
                        //    (예전에는 매 토큰 버리고 매 토큰 디스크에서 다시 읽었습니다). `LOGIS_KV_PLAN=legacy` 이면 예전 동작.
                        let over_ram_budget = legacy_ssd || self.mem.ram_kv_bytes > self.mem.ram_kv_budget;
                        if demote_to_ssd
                            && over_ram_budget
                            && inner.location == KVLocation::RAM
                            && idx < reg.len()
                            && reg[idx].ssd_path.is_some()
                            && !reg[idx].is_dirty.get(l_idx).copied().unwrap_or(true)
                        {
                            self.mem.ram_kv_bytes = self
                                .mem
                                .ram_kv_bytes
                                .saturating_sub(kv_block_bytes(inner.k_cache.as_ref(), inner.v_cache.as_ref()));
                            garbage_bin.push((inner.k_cache.take(), inner.v_cache.take()));
                            inner.location = KVLocation::SSD;
                            // 🌟 [TURBO-KV-BLOCK] 다시 읽을 때 SSD 파일의 tq_meta 로 형식을 정합니다.
                            inner.tq = None;
                            
                            reg[idx].location[l_idx] = KVLocation::SSD;
                            let mut cache = reg[idx].bitkv_cache.write().unwrap();
                            cache[l_idx] = None;
                        }
                    }
                    
                    if !garbage_bin.is_empty() {
                        tokio::task::spawn_blocking(move || {
                            drop(garbage_bin);
                        });
                    }
                }
            }

            
            if let Some(sid) = &session_id {
                let kv_dir = crate::utils::paths::get_kv_dir();
                let kv_name_raw = kv_name.as_deref().unwrap_or("text");
                let last_part = kv_name_raw.split('/').last().unwrap_or("text");
                let kv_type = if last_part == "inference" || last_part == "reference" || last_part.is_empty() { "text" } else { last_part };
                let sub_path = format!("{}/inference/{}", sid, kv_type);
                let base_dir = kv_dir.join(&sub_path);

                match &mut layer.attn {
                    AttnKind::SelfAttn(attn) => {
                        let mut dumps = Vec::new();
                        for block in attn.kv_blocks.iter_mut() {
                            let inner = block.inner.write().unwrap(); 
                            let is_full = inner.len == 1024;
                            
                            
                            // 오직 꽉 찬 블록만 '복사본'을 디스크로 보낼 뿐, VRAM에서 절대 삭제하지 않습니다!
                            let should_evacuate = is_full; 
                            
                            if should_evacuate && inner.k_cache.is_some() && inner.location == crate::models::qwen::quantized_model::KVLocation::VRAM {
                                let is_dirty = {
                                    let reg = attn.registry.entries.read().unwrap();
                                    if inner.index < reg.len() && l_idx < reg[inner.index].is_dirty.len() { reg[inner.index].is_dirty[l_idx] } else { true }
                                };

                                if is_dirty {
                                    let k = inner.k_cache.as_ref().unwrap();
                                    let v = inner.v_cache.as_ref().unwrap();
                                    let k_cpu = k.to_device(&candle_core::Device::Cpu).unwrap_or(k.clone());
                                    let v_cpu = v.to_device(&candle_core::Device::Cpu).unwrap_or(v.clone());
                                    let k_shape_u32: Vec<u32> = k_cpu.shape().dims().iter().map(|&x| x as u32).collect();
                                    
                                    dumps.push((
                                        crate::models::qwen::generate::LayerKVDump {
                                            layer_idx: l_idx,
                                            k_data: candle_core::Tensor::zeros((1,), candle_core::DType::U8, &candle_core::Device::Cpu).unwrap(),
                                            v_data: candle_core::Tensor::zeros((1,), candle_core::DType::U8, &candle_core::Device::Cpu).unwrap(),
                                            k_shape: candle_core::Tensor::from_vec(k_shape_u32, (k_cpu.shape().dims().len(),), &candle_core::Device::Cpu).unwrap(),
                                            raw_k: Some(k_cpu.contiguous().unwrap_or(k_cpu.clone())),
                                            raw_v: Some(v_cpu.contiguous().unwrap_or(v_cpu.clone())),
                                            // 🌟 [TURBO-KV-BLOCK] 압축 블록이면 형식과 함께 압축된 채로 SSD 에 기록됩니다.
                                            tq_meta: inner.tq,
                                        },
                                        inner.offset,
                                        inner.index
                                    ));
                                    
                                    let mut reg = attn.registry.entries.write().unwrap();
                                    if inner.index < reg.len() {
                                        reg[inner.index].is_dirty[l_idx] = false;
                                    }
                                }

                                
                                // VRAM 클리어 및 메모리 청소는 모든 문맥 파악이 끝난 뒤 `force_flush_all_active_blocks` 함수가 알아서 담당합니다.
                            }
                        }

                        if !dumps.is_empty() {
                            if let Some(tx) = crate::models::qwen::generate::BAKE_TX.get() {
                                let reg_clone = attn.registry.clone();
                                let sub_path_clone = sub_path.clone();
                                crate::rt::spawn(async move {
                                    for (dump, off, b_idx) in dumps {
                                        let sid = crate::models::qwen::generate::SLOT_MANAGER.acquire_write_slot(1024).await;
                                        let block_dir = base_dir.join(format!("b{}", off));
                                        if !block_dir.exists() { let _ = std::fs::create_dir_all(&block_dir); }
                                        
                                        crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                        if tx.send(crate::models::qwen::generate::SlotTask::Bake(crate::models::qwen::generate::BakeTask {
                                            slot_id: sid, task_dir: block_dir, kv_name: Some(sub_path_clone.clone()), offset: off, layers: vec![dump],
                                            is_relay_baking: false, block_idx: Some(b_idx), registry: reg_clone.clone(),
                                        })).await.is_err() {
                                            crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                                            crate::models::qwen::generate::SLOT_MANAGER.release_slot(sid).await;
                                        }
                                    }
                                });
                            }
                        }
                    },
                    AttnKind::LinearAttn(_attn) => {
                        // 디코딩 중 SSM 디스크 I/O 스팸 방지
                    }
                }
            }
        }
        
        // 🌟 [WEIGHT-RESIDENT] 이번 forward 에서 실제로 새로 올린 층이 있을 때만 알립니다.
        //    (프리필에서 이미 올렸거나 직전 호출에서 남긴 층은 다시 읽지 않았으므로 조용히 넘어갑니다)
        let phase = if is_decoding { "첫 디코딩 토큰" } else { "프리필" };
        if resident_decided && keep_resident && newly_resident > 0 {
            let free_prev = crate::utils::resources::free_ram_bytes() as u64;
            crate::utils::resources::release_os_working_set();
            let (free_now, total) = crate::utils::resources::ram_snapshot();
            println!(
                "[DECODE-RESIDENT] {}에서 층 {}개를 VRAM 에 올렸습니다 (전체 {}개 상주). 남은 계산은 mmap 을 읽지 않으므로 작업 집합을 반환합니다 | RAM 여유 {}MB → {}MB | {}",
                phase,
                newly_resident,
                total_layers,
                free_prev / 1_000_000,
                free_now / 1_000_000,
                crate::utils::resources::ram_pressure_line(free_now, total)
            );
        } else if resident_decided && resident_prefix > 0 && newly_resident > 0 {
            let (free_now, total) = crate::utils::resources::ram_snapshot();
            println!(
                "[DECODE-RESIDENT] {}에서 층 {}/{}개를 VRAM 에 올렸고 그 층의 파일 페이지는 바로 작업 집합에서 내려놓았습니다. 나머지 {}개 층은 매번 mmap 에서 읽고 {} | {}",
                phase,
                resident_prefix,
                total_layers,
                total_layers - resident_prefix,
                if release_streamed { "읽은 직후 그 층의 파일 페이지도 내려놓습니다" } else { "RAM 여유가 기준 위라 그 층의 파일 페이지는 작업 집합에 둡니다" },
                crate::utils::resources::ram_pressure_line(free_now, total)
            );
        }
        tokio::task::yield_now().await;
        xs = self.norm.forward(&xs)?;
        self.current_kv_len = seqlen_offset + seq_len;
        Ok(xs)
    }

    /// 🌟 [KV RESIDENCY] KV 블록을 실제로 보유하는 full_attention 레이어의
    /// (레이어 수, KV 헤드 수, head_dim) 을 반환합니다.
    /// linear_attention(SSM) 레이어는 토큰 수와 무관한 고정 크기이므로 집계에서 제외합니다.
    fn kv_plan_geometry(&self) -> (usize, usize, usize) {
        let mut layers = 0usize;
        let mut heads = 0usize;
        let mut dim = 0usize;
        for layer in self.layers.iter() {
            if let AttnKind::SelfAttn(attn) = &layer.attn {
                layers += 1;
                heads = attn.num_key_value_heads;
                dim = attn.head_dim;
            }
        }
        (layers, heads, dim)
    }

    /// 🌟 [KV-PLAN v2] generate 가 이번 생성의 최대 새 토큰 수(max_tokens)를 알려 줍니다. KV 계획의 여유 토큰이 됩니다.
    pub fn set_planned_decode_tokens(&mut self, max_new_tokens: usize) {
        self.mem.planned_decode_tokens = max_new_tokens;
    }

    /// KV 계획에 쓰는 디코딩 여유 토큰 수 = min(이번 max_tokens, 2048). 모르면 2048.
    pub fn decode_headroom_tokens(&self) -> usize {
        match self.mem.planned_decode_tokens {
            0 => QWEN35_DECODE_HEADROOM_TOKENS,
            n => n.min(QWEN35_DECODE_HEADROOM_TOKENS),
        }
    }

    /// 지금 VRAM 에 만들어져 있는 KV 토큰 수 (첫 full_attention 층 기준). 이 KV 는 VRAM free 에서 이미 빠져 있습니다.
    pub fn vram_kv_tokens(&self) -> usize {
        for layer in self.layers.iter() {
            if let AttnKind::SelfAttn(attn) = &layer.attn {
                return attn
                    .kv_blocks
                    .iter()
                    .map(|b| {
                        let i = b.inner.read().unwrap();
                        if i.location == KVLocation::VRAM && i.k_cache.is_some() { i.len } else { 0 }
                    })
                    .sum();
            }
        }
        0
    }

    /// 지금 RAM 에 있는 KV 바이트 (모든 full_attention 층).
    fn ram_kv_bytes_now(&self) -> u64 {
        let mut total = 0u64;
        for layer in self.layers.iter() {
            if let AttnKind::SelfAttn(attn) = &layer.attn {
                for b in attn.kv_blocks.iter() {
                    let i = b.inner.read().unwrap();
                    if i.location == KVLocation::RAM {
                        total += kv_block_bytes(i.k_cache.as_ref(), i.v_cache.as_ref());
                    }
                }
            }
        }
        total
    }

    /// GDN(linear_attention) 층들의 conv + recurrent 상태 바이트.
    pub fn ssm_state_bytes(&self) -> u64 {
        self.layers
            .iter()
            .filter(|l| l.layer_type == "linear_attention")
            .map(|l| {
                let (c, r) = l.get_ssm_states();
                kv_block_bytes(c.as_ref(), r.as_ref())
            })
            .sum()
    }

    /// 🌟 [KV-PLAN v2] 지금 정밀도의 토큰당 KV 바이트 (전 full_attention 층 K+V).
    /// 원본이면 보관 dtype 크기(F16 보관이면 2), 압축이면 TurboQuant 행 머리까지 포함한 실제 행 크기.
    pub fn kv_bytes_per_token_now(&self, dev: &Device) -> u64 {
        let (layers, heads, dim) = self.kv_plan_geometry();
        if self.kv_exact {
            let elem = if self.mem.kv_half { 2 } else { self.dtype.size_in_bytes() };
            return crate::utils::resources::kv_bytes_per_token_exact(layers, heads, dim, elem);
        }
        match crate::models::common::kv_turbo::default_meta(dim, dev) {
            Ok(m) if dev.is_cuda_or_rocm() => (m.bytes_per_token_head() * layers * heads) as u64,
            _ => crate::utils::resources::kv_bytes_per_token_compressed(layers, heads, dim),
        }
    }

    /// 🌟 [KV-PLAN v2] VRAM 에 있는 원본 KV 블록을 그 자리에서 TurboQuant 벡터별 FP8 로 압축합니다. 압축한 블록 수.
    /// 전부 성공하면 이후 새 토큰도 압축 블록으로 쌓습니다. 코덱이 없으면(4bit 정책·미빌드 등) 아무것도 하지 않습니다.
    fn recompress_vram_kv(&mut self, dev: &Device) -> usize {
        if kv_demote_plain_fp8() || crate::models::common::kv_turbo::kv_is_compact(dev) {
            return 0;
        }
        let (_, _, dim) = self.kv_plan_geometry();
        let Some(codec) = crate::models::common::kv_turbo::block_codec(dim, dev) else {
            return 0;
        };
        let mut packed = 0usize;
        let mut failed = 0usize;
        for layer in self.layers.iter_mut() {
            if let AttnKind::SelfAttn(attn) = &mut layer.attn {
                for block in attn.kv_blocks.iter() {
                    let mut inner = block.inner.write().unwrap();
                    if inner.location != KVLocation::VRAM || inner.tq.is_some() {
                        continue;
                    }
                    let packed_pair = match (inner.k_cache.as_ref(), inner.v_cache.as_ref()) {
                        (Some(k), Some(v)) if k.dtype() != DType::F8E4M3 => match (codec.pack_k(k), codec.pack_v(v)) {
                            (Ok(pk), Ok(pv)) => Some((pk, pv)),
                            _ => None,
                        },
                        _ => continue,
                    };
                    match packed_pair {
                        Some((pk, pv)) => {
                            inner.k_cache = Some(pk);
                            inner.v_cache = Some(pv);
                            inner.tq = Some(codec.meta);
                            packed += 1;
                        }
                        None => failed += 1,
                    }
                }
            }
        }
        if failed == 0 {
            for layer in self.layers.iter_mut() {
                if let AttnKind::SelfAttn(attn) = &mut layer.attn {
                    attn.kv_exact = false;
                    attn.kv_half = false;
                }
            }
            self.kv_exact = false;
            self.mem.kv_half = false;
        }
        let _ = dev.synchronize();
        packed
    }

    pub fn clear_cache(&mut self) {
        let was_resident = self.keep_weights_resident
            || self.resident_prefix > 0
            || self.layers.iter().any(|l| !l.is_cleared());
        // 🌟 [SPD-05] KV 캐시를 먼저 비워 VRAM 을 회수한 뒤 `keep_warm_ok()` 를 판정합니다.
        //    (기존에는 KV 를 비우기 전에 VRAM free 를 읽어 상주 가중치를 불필요하게 내려놓았습니다.)
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
            // 🌟 [KV-PRECISION] 다음 문맥의 첫 프리필에서 다시 정합니다. (그 전까지는 압축이 기본)
            if let AttnKind::SelfAttn(attn) = &mut layer.attn {
                attn.kv_exact = false;
                attn.kv_half = false;
            }
        }
        // 🌟 [KV-REGISTRY] 이전 문맥의 RAM 사본·SSD 경로가 다음 문맥에 섞이지 않게 장부도 비웁니다.
        self.reset_registry_runtime();
        let keep_warm = was_resident && self.keep_warm_ok();
        if self.ct.is_some() && self.mmap.is_some() && was_resident && !keep_warm {
            for layer in self.layers.iter_mut() {
                layer.clear_weights();
            }
        }
        self.kv_exact = false;
        self.planned_prompt_len = 0;
        // 🌟 [KV-PLAN v2] 직전 KV 판정(last_kv_plan)은 히스테리시스용으로 남기고 나머지는 다음 호출에서 다시 정합니다.
        self.mem.planned_decode_tokens = 0;
        self.mem.ssm_in_vram = None;
        self.mem.kv_half = false;
        self.mem.ram_kv_budget = 0;
        self.mem.ram_kv_bytes = 0;
        self.current_kv_len = 0;
        self.kv_plan = None;
        self.keep_weights_resident = false;
        self.resident_prefix = 0;
        self.release_streamed_pages = false;
    }

    fn decode_resident_bytes(&self) -> u64 {
        let ct = match self.ct.as_ref() {
            Some(c) => c,
            None => return 0,
        };
        ct.tensor_infos
            .iter()
            .filter(|(name, _)| name.starts_with("blk."))
            .map(|(_, info)| {
                let block = info.ggml_dtype.block_size().max(1);
                (info.shape.elem_count() / block * info.ggml_dtype.type_size()) as u64
            })
            .sum()
    }

    fn layer_file_ranges(&self, layer_idx: usize) -> Vec<(usize, usize)> {
        let ct = match self.ct.as_ref() {
            Some(c) => c,
            None => return Vec::new(),
        };
        let prefix = format!("blk.{}.", layer_idx);
        ct.tensor_infos
            .iter()
            .filter(|(name, _)| name.starts_with(&prefix))
            .map(|(_, info)| {
                let block = info.ggml_dtype.block_size().max(1);
                (
                    (ct.tensor_data_offset + info.offset) as usize,
                    info.shape.elem_count() / block * info.ggml_dtype.type_size(),
                )
            })
            .collect()
    }

    fn layer_bytes(&self, layer_idx: usize) -> u64 {
        self.layer_file_ranges(layer_idx).iter().map(|(_, len)| *len as u64).sum()
    }

    fn max_layer_bytes(&self) -> u64 {
        (0..self.layers.len()).map(|i| self.layer_bytes(i)).max().unwrap_or(0)
    }

    /// 앞에서부터 VRAM 에 둘 수 있는 층 수와 그러려고 새로 읽어야 하는 바이트.
    /// 🌟 [WEIGHT-RESIDENT] 이미 올라와 있는 층은 추가 비용이 0 입니다.
    fn decode_resident_prefix(&self, budget: u64) -> (usize, u64) {
        let mut count = 0usize;
        let mut bytes = 0u64;
        for i in 0..self.layers.len() {
            let b = self.layer_bytes(i);
            let cost = if self.layers[i].is_cleared() { b } else { 0 };
            if b == 0 || bytes + cost > budget {
                break;
            }
            bytes += cost;
            count += 1;
        }
        (count, bytes)
    }

    /// 🌟 [WEIGHT-RESIDENT] 아직 VRAM 에 없어 다시 읽어야 하는 층 가중치 바이트
    fn missing_layer_bytes(&self) -> u64 {
        (0..self.layers.len())
            .filter(|&i| self.layers[i].is_cleared())
            .map(|i| self.layer_bytes(i))
            .sum()
    }

    /// 🌟 [WEIGHT-RESIDENT] VRAM 에 올라와 있는 층 가중치 바이트
    pub fn resident_layer_bytes(&self) -> u64 {
        (0..self.layers.len())
            .filter(|&i| !self.layers[i].is_cleared())
            .map(|i| self.layer_bytes(i))
            .sum()
    }

    /// 🌟 [WEIGHT-RESIDENT] 프리필 한 번(청크 `seq_len` 토큰)에 순간적으로 필요한 활성값 VRAM 의 보수적 추정치.
    /// (은닉 상태·MLP 중간값·SSM 청크 행렬·양자화 matmul 의 역양자화 버퍼)
    fn prefill_transient_bytes(&self, seq_len: usize) -> u64 {
        let hidden = self.embed_tokens.hidden().max(1) as u64;
        let inter = self
            .ct
            .as_ref()
            .and_then(|ct| ct.metadata.get("qwen35.feed_forward_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as u64)
            .unwrap_or(hidden * 4);
        let per_token = hidden * 4 * 8 + inter * 4 * 3;
        seq_len as u64 * per_token + 128 * 1024 * 1024
    }

    /// 🌟 [WEIGHT-RESIDENT] 상주 층을 둔 채로 `need` 바이트(KV·활성값 순간치)를 쓸 수 있는지 확인합니다.
    /// 동기화로 메모리 풀이 쥔 여분을 돌려준 뒤 재고, [`RESIDENT_MIN_RESERVE_BYTES`] 까지 더해도 모자라면
    /// 상주 층을 모두 내려놓습니다(예전 스트리밍 경로). 반환값은 그 뒤의 VRAM free (CUDA/ROCm 밖이면 None).
    fn ensure_resident_reserve(&mut self, dev: &Device, need: u64) -> Option<u64> {
        if !dev.is_cuda_or_rocm() {
            return None;
        }
        let _ = dev.synchronize();
        let free = crate::utils::device_utils::device_vram_snapshot(dev)
            .map(|s| s.effective_free_bytes())?;
        let resident = self.resident_layer_bytes();
        if resident > 0 && free < need + RESIDENT_MIN_RESERVE_BYTES {
            let held = self.layers.iter().filter(|l| !l.is_cleared()).count();
            for layer in self.layers.iter_mut() {
                layer.clear_weights();
            }
            self.keep_weights_resident = false;
            self.resident_prefix = 0;
            let _ = dev.synchronize();
            let after = crate::utils::device_utils::device_vram_snapshot(dev)
                .map(|s| s.effective_free_bytes());
            println!(
                "[WEIGHT-RESIDENT] VRAM free {:.0} MB 가 필요량 {:.0} MB + 여유 {:.0} MB 보다 작아 상주하던 층 {}개 {:.0} MB 를 내려놓습니다 → free {:.0} MB (이번 계산은 층을 읽고 바로 비우는 경로부터 다시 판정합니다)",
                free as f64 / 1e6,
                need as f64 / 1e6,
                RESIDENT_MIN_RESERVE_BYTES as f64 / 1e6,
                held,
                resident as f64 / 1e6,
                after.unwrap_or(free) as f64 / 1e6
            );
            return after;
        }
        Some(free)
    }

    /// 🌟 [WEIGHT-RESIDENT] 프리필에서 읽은 층을 계산 뒤에도 VRAM 에 둘지 정합니다. `(전부 상주, 앞쪽 상주 층 수)`
    /// 디코딩 진입 판정과 같은 기준(새로 읽을 가중치 + KV + margin ≤ VRAM free)에 프리필 활성값 순간치를 더해 판정합니다.
    fn plan_prefill_residency(&mut self, dev: &Device, seq_len: usize, seqlen_offset: usize) -> (bool, usize) {
        if !dev.is_cuda_or_rocm() || !prefill_resident_enabled() || self.ct.is_none() {
            return (false, 0);
        }
        let weight_bytes = self.decode_resident_bytes();
        if weight_bytes == 0 {
            return (false, 0);
        }
        let (full_attn_layers, kv_heads, kv_head_dim) = self.kv_plan_geometry();
        // 🌟 [KV-PLAN v2] 이미 VRAM 에 만든 KV(seqlen_offset 토큰)는 다시 요구하지 않고, 여유 토큰은 실제 max_tokens 입니다.
        let kv_bytes = if crate::utils::resources::kv_plan_legacy() {
            let planned = seqlen_offset + seq_len + QWEN35_DECODE_HEADROOM_TOKENS;
            (full_attn_layers * kv_heads * kv_head_dim * 2 * planned) as u64
        } else {
            let new_tokens = seq_len + self.decode_headroom_tokens();
            self.kv_bytes_per_token_now(dev) * new_tokens as u64
        };
        let transient = self.prefill_transient_bytes(seq_len);
        let Some(free) = self.ensure_resident_reserve(dev, kv_bytes + transient) else {
            return (false, 0);
        };
        let missing = self.missing_layer_bytes();
        let n = self.layers.len();
        let res_margin = weight_resident_margin_bytes();
        let mut keep = free >= missing + kv_bytes + transient + res_margin;
        let mut prefix = if keep {
            0
        } else {
            self.decode_resident_prefix(free.saturating_sub(
                kv_bytes + transient + res_margin + self.max_layer_bytes(),
            ))
            .0
        };
        let already_all = !keep && prefix >= n && n > 0;
        if already_all {
            // 🌟 [PREFILL-RESIDENT] 모든 층이 이미 상주 → '전부 상주' 로 둡니다 (디코딩 판정·로그와 일치).
            keep = true;
            prefix = 0;
        }
        if seqlen_offset == 0 {
            let verdict = if already_all {
                format!("이미 올라와 있는 층 {n}개를 그대로 씁니다 (새로 읽지 않습니다)")
            } else if keep {
                format!("전부 상주 (층 {n}개 · 디코딩에서 다시 읽지 않습니다)")
            } else if prefix >= n {
                format!("이미 올라와 있는 층 {n}개를 그대로 씁니다 (새로 읽지 않습니다)")
            } else if prefix > 0 {
                format!("앞쪽 층 {prefix}/{n}개 상주, 나머지는 읽고 바로 비움")
            } else {
                "상주 없음 (예전처럼 층을 읽고 바로 비웁니다)".to_string()
            };
            println!(
                "[PREFILL-RESIDENT] {} 토큰 | 층 가중치 {:.0} MB (이미 상주 {:.0} MB · 새로 읽을 {:.0} MB) + KV {:.0} MB + 프리필 순간치 {:.0} MB + margin {:.0} MB vs VRAM free {:.0} MB → {}",
                seq_len,
                weight_bytes as f64 / 1e6,
                weight_bytes.saturating_sub(missing) as f64 / 1e6,
                missing as f64 / 1e6,
                kv_bytes as f64 / 1e6,
                transient as f64 / 1e6,
                res_margin as f64 / 1e6,
                free as f64 / 1e6,
                verdict
            );
        }
        (keep, prefix)
    }

    /// 🌟 [KV-PRECISION] KV 원소 하나의 바이트 수 (원본이면 모델 dtype, 압축이면 FP8 기준 1)
    fn kv_elem_bytes(&self) -> usize {
        if self.kv_exact {
            if self.mem.kv_half { 2 } else { self.dtype.size_in_bytes() }
        } else {
            1
        }
    }

    /// 🌟 [KV-PRECISION] 문맥 첫 프리필에서 이번 문맥의 KV 를 원본 정밀도로 둘지 정합니다.
    /// 프리필이 상주시키기로 한 층 가중치·프리필 순간치를 먼저 빼고, 남는 VRAM 에 원본 KV(+ 디코딩 여유 토큰)와
    /// 안전 여유가 들어가면 원본으로 둡니다(kv_turbo::exact_kv_affordable). 들어가지 않으면 TurboQuant 압축 블록입니다.
    fn plan_kv_precision(&mut self, dev: &Device, seq_len: usize, keep: bool, prefix: usize) {
        use crate::models::common::kv_turbo::{self as kvt, KvHalfMode};
        let (full_attn_layers, kv_heads, kv_head_dim) = self.kv_plan_geometry();
        let legacy = crate::utils::resources::kv_plan_legacy();
        let prompt = self.planned_prompt_len.max(seq_len);
        let headroom = if legacy { QWEN35_DECODE_HEADROOM_TOKENS } else { self.decode_headroom_tokens() };
        let planned = prompt + headroom;
        let exact_bytes =
            (full_attn_layers * kv_heads * kv_head_dim * 2 * planned * self.dtype.size_in_bytes()) as u64;
        // 프리필이 새로 올려 둘 층 가중치 + 프리필 순간치 (층을 스트리밍할 때만 스트리밍 버퍼 1층분을 더합니다)
        let n = self.layers.len();
        let weights_to_load: u64 = if keep {
            self.missing_layer_bytes()
        } else {
            let streaming = legacy || prefix < n;
            (0..prefix.min(n))
                .filter(|&i| self.layers[i].is_cleared())
                .map(|i| self.layer_bytes(i))
                .sum::<u64>()
                + if streaming { self.max_layer_bytes() } else { 0 }
        };
        let reserve = weights_to_load + self.prefill_transient_bytes(seq_len);
        // 🌟 [KV-PLAN v2] KV 안전 여유는 비례값 max(96MB, 2 × 디코딩 순간치) (예전 고정 640MB 는 LOGIS_KV_PLAN=legacy)
        let margin = if legacy {
            crate::utils::resources::KV_VRAM_SAFETY_MARGIN_BYTES
        } else {
            crate::utils::resources::kv_vram_margin_bytes(DECODE_TRANSIENT_BYTES)
        };
        let (mut exact, mut free) = kvt::exact_kv_affordable_with_margin(dev, exact_bytes, reserve, margin);
        // 🌟 [KV-HALF] F32 모델: F32 원본 KV 가 안 들어가면 F16 원본(절반 크기)을 먼저 시도합니다 (FP8 압축보다 정확).
        let mut half = false;
        if self.dtype == DType::F32 && dev.is_cuda_or_rocm() {
            let half_bytes = exact_bytes / 2;
            let half_fits = || {
                crate::utils::device_utils::device_vram_snapshot(dev)
                    .map(|s| {
                        let f = s.effective_free_bytes();
                        (f >= half_bytes + margin + reserve, f)
                    })
            };
            match kvt::kv_half_mode() {
                KvHalfMode::Off => {}
                KvHalfMode::Prefer => {
                    if exact {
                        half = true;
                    } else if let Some((ok, f)) = half_fits() {
                        free = Some(f);
                        exact = ok;
                        half = ok;
                    }
                }
                KvHalfMode::Auto => {
                    if !exact && kvt::kv_precision_policy() == kvt::KvPrecisionPolicy::Auto {
                        if let Some((ok, f)) = half_fits() {
                            free = Some(f);
                            exact = ok;
                            half = ok;
                        }
                    }
                }
            }
        }
        self.kv_exact = exact;
        self.mem.kv_half = half;
        for layer in self.layers.iter_mut() {
            if let AttnKind::SelfAttn(attn) = &mut layer.attn {
                attn.kv_exact = exact;
                attn.kv_half = half;
            }
        }
        if dev.is_cuda_or_rocm() || dev.is_vulkan() {
            println!(
                "[KV-PRECISION] Qwen3.5 | 프롬프트 {} + 여유 {} 토큰 · 원본 KV {:.0} MB (F16 보관 시 {:.0} MB) + 가중치·순간치 {:.0} MB + 여유 {:.0} MB vs VRAM free {} → {}",
                prompt,
                headroom,
                exact_bytes as f64 / 1e6,
                (exact_bytes / 2) as f64 / 1e6,
                reserve as f64 / 1e6,
                margin as f64 / 1e6,
                free.map(|f| format!("{:.0} MB", f as f64 / 1e6)).unwrap_or_else(|| "?".to_string()),
                if exact && half {
                    "원본 KV · F16 보관 (압축하지 않음 · F32 계산)"
                } else if exact {
                    "원본 KV (압축하지 않음 · 정확도 우선)"
                } else {
                    "TurboQuant 압축 KV (메모리 우선)"
                }
            );
        }
    }

    /// 🌟 [WEIGHT-RESIDENT] 호출이 끝난 뒤에도 층 가중치를 VRAM 에 남겨 둘지.
    /// 텍스트 전용 생성기([`Self::keep_warm`])이고, CUDA/ROCm 에서 KV 를 비운 뒤의 VRAM free 가
    /// margin 이상일 때만 남깁니다. (`LOGIS_KEEP_WEIGHTS_WARM=0` 이면 예전처럼 매번 내려놓습니다)
    fn keep_warm_ok(&self) -> bool {
        if !self.keep_warm || !keep_weights_warm_enabled() {
            return false;
        }
        let dev = self.norm.weight.device().clone();
        if !dev.is_cuda_or_rocm() {
            return false;
        }
        let _ = dev.synchronize();
        let res_margin = weight_resident_margin_bytes();
        match crate::utils::device_utils::device_vram_snapshot(&dev) {
            Some(s) => s.effective_free_bytes() >= res_margin,
            None => false,
        }
    }

    /// 🌟 [WEIGHT-RESIDENT] VRAM 에 상주시킨 층 가중치를 모두 내려놓습니다.
    pub fn release_resident_weights(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_weights();
        }
        self.keep_weights_resident = false;
        self.resident_prefix = 0;
    }

    /// 🌟 [KV-REGISTRY] 블록 장부의 실행 중 상태(RAM 사본, SSD 경로, 변경 표시)를 비웁니다.
    /// 장부는 블록 번호로만 찾기 때문에, 비우지 않으면 다음 문맥의 같은 번호 블록이 이전 문맥의
    /// RAM 사본·SSD 파일을 읽을 수 있습니다. (RAM 사본은 프로세스 메모리도 계속 붙잡습니다)
    pub fn reset_registry_runtime(&self) {
        if let Ok(mut reg) = self.registry.entries.write() {
            for e in reg.iter_mut() {
                if let Ok(mut cache) = e.bitkv_cache.write() {
                    for slot in cache.iter_mut() {
                        *slot = None;
                    }
                }
                e.ssd_path = None;
                e.is_dirty.fill(true);
            }
        }
    }

    /// 🌟 [PREFETCH] 곧 다시 읽을 층의 파일 페이지를 OS 가 미리(비동기로) 읽어 두게 합니다.
    /// 층을 스트리밍하는 동안 지금 층을 계산하는 시간에 다음 층의 디스크 읽기가 겹치므로,
    /// RAM 압박으로 페이지가 내려간 상태(SSD 재읽기)에서 층마다 기다리던 시간이 줄어듭니다.
    /// 이미 메모리에 있는 페이지면 아무 일도 하지 않습니다. 계산 결과에는 영향이 없습니다.
    fn prefetch_layer_pages(&self, layer_idx: usize) {
        let mmap = match self.mmap.as_ref() {
            Some(m) => m,
            None => return,
        };
        let ranges: Vec<(usize, usize)> = self
            .layer_file_ranges(layer_idx)
            .into_iter()
            .filter(|&(start, len)| len > 0 && start < mmap.len())
            .map(|(start, len)| (start, len.min(mmap.len() - start)))
            .collect();
        if ranges.is_empty() {
            return;
        }
        #[cfg(target_os = "windows")]
        unsafe {
            use windows_sys::Win32::System::Memory::{PrefetchVirtualMemory, WIN32_MEMORY_RANGE_ENTRY};
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            let entries: Vec<WIN32_MEMORY_RANGE_ENTRY> = ranges
                .iter()
                .map(|&(start, len)| WIN32_MEMORY_RANGE_ENTRY {
                    VirtualAddress: mmap.as_ptr().add(start) as *mut std::ffi::c_void,
                    NumberOfBytes: len,
                })
                .collect();
            let _ = PrefetchVirtualMemory(GetCurrentProcess(), entries.len(), entries.as_ptr(), 0);
        }
        #[cfg(unix)]
        {
            for (start, len) in ranges {
                let _ = mmap.advise_range(memmap2::Advice::WillNeed, start, len);
            }
        }
    }

    fn release_layer_pages(&self, layer_idx: usize) {
        let mmap = match self.mmap.as_ref() {
            Some(m) => m,
            None => return,
        };
        for (start, len) in self.layer_file_ranges(layer_idx) {
            if len == 0 || start >= mmap.len() {
                continue;
            }
            let len = len.min(mmap.len() - start);
            #[cfg(target_os = "windows")]
            unsafe {
                let _ = windows_sys::Win32::System::Memory::VirtualUnlock(
                    mmap.as_ptr().add(start) as *const std::ffi::c_void,
                    len,
                );
            }
            #[cfg(unix)]
            unsafe {
                let _ = mmap.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, start, len);
            }
        }
    }

    
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.truncate_kv_cache(len)?;
        }
        self.current_kv_len = len;
        Ok(())
    }

    pub fn restore_kv_registry(&mut self, kv_name: &str) -> Result<()> {
        use crate::models::qwen::generate::LayerIndex;
        let kv_dir = crate::utils::paths::get_kv_dir();
        let index_path = kv_dir.join(kv_name).join("layer0.json");
        
        if !index_path.exists() { return Ok(()); }
        
        let index_json = String::from_utf8(crate::utils::direct_loader::load_kv_block(&index_path).unwrap_or_default()).unwrap_or_default();
        let index: LayerIndex = serde_json::from_str(&index_json).unwrap_or_else(|_| LayerIndex { layer_idx: 0, total_tokens: 0, blocks: vec![] });
        let total_tokens = index.total_tokens;

        // 🌟 [KV-REGISTRY] 다른 스냅샷에서 읽어 둔 RAM 사본이 남아 있으면 새 스냅샷의 같은 번호 블록 대신 그것을 읽습니다.
        //    복원할 때마다 장부의 RAM 사본과 경로를 비우고 아래에서 이 스냅샷의 경로로 다시 채웁니다.
        self.reset_registry_runtime();
        {
            let mut reg = self.registry.entries.write().unwrap();
            let needed_blocks = (total_tokens + 1023) / 1024;
            while reg.len() < needed_blocks {
                let off = reg.len() * 1024;
                reg.push(crate::models::qwen::quantized_model::RegistryEntry::new(off, 0, self.layers.len().max(28)));
            }
        }

        for layer in self.layers.iter_mut() {
            if let AttnKind::SelfAttn(attn) = &mut layer.attn {
                let reg_len = self.registry.entries.read().unwrap().len();
                while attn.kv_blocks.len() < reg_len {
                    let idx = attn.kv_blocks.len();
                    let off = idx * 1024;
                    attn.kv_blocks.push(crate::models::qwen::quantized_model::KVBlock::new(
                        crate::models::qwen::quantized_model::KVLocation::SSD, idx, 0, off
                    ));
                }
            }
        }

        {
            let mut reg = self.registry.entries.write().unwrap();
            for (idx, entry) in reg.iter_mut().enumerate() {
                let off = idx * 1024;
                let b_len = if off + 1024 <= total_tokens { 1024 } else { total_tokens.saturating_sub(off) };
                entry.token_len = b_len;
                
                
                // 이 처리가 없으면 스냅샷을 로드하자마자 전체 문맥을 다시 디스크에 덮어쓰는 I/O 스팸이 발생합니다.
                entry.is_dirty.fill(false);
                entry.ssd_path = Some(kv_dir.join(kv_name).join(format!("b{}", off)));
                
                for layer in self.layers.iter_mut() {
                    if let AttnKind::SelfAttn(attn) = &mut layer.attn {
                        if let Some(block) = attn.kv_blocks.get(idx) {
                            let mut inner = block.inner.write().unwrap();
                            inner.len = b_len;
                            inner.ssd_path = entry.ssd_path.clone();
                        }
                    }
                }
            }
        }
        self.current_kv_len = total_tokens;
        Ok(())
    }

    
    pub async fn force_flush_all_active_blocks(&mut self, session_id: &str, kv_name: Option<&str>) -> Result<()> {
        use crate::models::qwen::generate::{SLOT_MANAGER, SlotTask, BakeTask, BAKE_TX, LayerKVDump};
        
        let mut block_groups: std::collections::HashMap<(usize, usize), Vec<LayerKVDump>> = std::collections::HashMap::new();
        let mut ssm_dumps = Vec::new();

        for (l_idx, layer) in self.layers.iter_mut().enumerate() {
            if let AttnKind::SelfAttn(attn) = &mut layer.attn {
                attn.prefetched_block = None;

                let mut gpu_k_list = Vec::new();
                let mut gpu_v_list = Vec::new();
                let mut target_indices = Vec::new();
                // 🌟 [TURBO-KV-BLOCK] TurboQuant 압축 블록은 BF16 변환·병합 없이 따로 내립니다.
                let mut packed_indices = Vec::new();

                // 1. 읽기 락만 짧게 잡고 넘길 대상들을 수집 및 VRAM 선행 처리
                for (idx, block) in attn.kv_blocks.iter().enumerate() {
                    let inner = block.inner.read().unwrap(); 
                    let is_full = inner.len == 1024;
                    
                    if is_full && inner.k_cache.is_some() && inner.location == crate::models::qwen::quantized_model::KVLocation::VRAM {
                        if inner.tq.is_some() {
                            packed_indices.push(idx);
                            continue;
                        }
                        let k = inner.k_cache.as_ref().unwrap();
                        let v = inner.v_cache.as_ref().unwrap();
                        
                        
                        let k_gpu = k.to_dtype(candle_core::DType::BF16).unwrap_or_else(|_| k.clone()).contiguous().unwrap_or_else(|_| k.clone());
                        let v_gpu = v.to_dtype(candle_core::DType::BF16).unwrap_or_else(|_| v.clone()).contiguous().unwrap_or_else(|_| v.clone());
                        
                        gpu_k_list.push(k_gpu);
                        gpu_v_list.push(v_gpu);
                        target_indices.push(idx);
                    }
                }

                // 2. 모인 VRAM 텐서가 있다면 통째로 합쳐서 한 방에 보냄
                if !gpu_k_list.is_empty() {
                    
                    let merged_k_gpu = candle_core::Tensor::cat(&gpu_k_list, 2).unwrap_or_else(|_| gpu_k_list[0].clone());
                    let merged_v_gpu = candle_core::Tensor::cat(&gpu_v_list, 2).unwrap_or_else(|_| gpu_v_list[0].clone());

                    // 🌟 [PRC-04] 스케일 없는 raw FP8 강등은 명시적 옵트인(`LOGIS_RAW_FP8_KV=1`)이면서 비원본 모드일 때만 허용합니다.
                    let target_dtype = if !self.kv_exact
                        && raw_fp8_kv_enabled()
                        && (merged_k_gpu.device().is_cuda_or_rocm() || merged_k_gpu.dtype() == candle_core::DType::F8E4M3)
                    {
                        candle_core::DType::F8E4M3
                    } else if merged_k_gpu.device().is_cuda_or_rocm() {
                        candle_core::DType::BF16
                    } else {
                        candle_core::DType::F32
                    };
                    let merged_k_cpu = merged_k_gpu.to_dtype(target_dtype).unwrap_or_else(|_| merged_k_gpu.clone()).to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| merged_k_gpu.clone());
                    let merged_v_cpu = merged_v_gpu.to_dtype(target_dtype).unwrap_or_else(|_| merged_v_gpu.clone()).to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| merged_v_gpu.clone());

                    // 3. CPU(RAM)로 무사히 넘어온 거대 텐서를 다시 원래 블록 크기대로 썰어서 캐시에 재할당
                    let mut current_offset = 0;
                    for (i, &idx) in target_indices.iter().enumerate() {
                        let chunk_len = gpu_k_list[i].dim(2).unwrap_or(1024);
                        
                        let k_cpu = merged_k_cpu.narrow(2, current_offset, chunk_len).unwrap_or_else(|_| merged_k_cpu.clone()).contiguous().unwrap_or_else(|_| merged_k_cpu.clone());
                        let v_cpu = merged_v_cpu.narrow(2, current_offset, chunk_len).unwrap_or_else(|_| merged_v_cpu.clone()).contiguous().unwrap_or_else(|_| merged_v_cpu.clone());
                        current_offset += chunk_len;

                        let mut inner = attn.kv_blocks[idx].inner.write().unwrap();
                        let is_dirty = {
                            let reg = attn.registry.entries.read().unwrap();
                            if inner.index < reg.len() && l_idx < reg[inner.index].is_dirty.len() { 
                                reg[inner.index].is_dirty[l_idx] 
                            } else { true }
                        };

                        if is_dirty {
                            let k_shape_u32: Vec<u32> = k_cpu.shape().dims().iter().map(|&x| x as u32).collect();
                            
                            block_groups.entry((inner.offset, inner.index)).or_default().push(LayerKVDump {
                                layer_idx: l_idx,
                                k_data: candle_core::Tensor::zeros((1,), candle_core::DType::U8, &candle_core::Device::Cpu).unwrap(),
                                v_data: candle_core::Tensor::zeros((1,), candle_core::DType::U8, &candle_core::Device::Cpu).unwrap(),
                                k_shape: candle_core::Tensor::from_vec(k_shape_u32, (k_cpu.shape().dims().len(),), &candle_core::Device::Cpu).unwrap(),
                                raw_k: Some(k_cpu.clone()),
                                raw_v: Some(v_cpu.clone()),
                                tq_meta: None,
                            });
                            
                            let mut reg = attn.registry.entries.write().unwrap();
                            if inner.index < reg.len() {
                                reg[inner.index].is_dirty[l_idx] = false;
                            }
                        }

                        inner.k_cache = Some(k_cpu);
                        inner.v_cache = Some(v_cpu);
                        inner.location = crate::models::qwen::quantized_model::KVLocation::RAM;
                        
                        let mut reg = attn.registry.entries.write().unwrap();
                        if inner.index < reg.len() {
                            reg[inner.index].location[l_idx] = crate::models::qwen::quantized_model::KVLocation::RAM;
                        }
                    }
                }

                // 🌟 [TURBO-KV-BLOCK] 압축 블록: u32 행을 그대로 RAM 으로 내리고, 변경된 블록은
                //    압축된 채로(tq_meta 포함) SSD 에 기록합니다. FP8 경로 대비 전송량과 파일 크기가 작습니다.
                for &idx in packed_indices.iter() {
                    let mut inner = attn.kv_blocks[idx].inner.write().unwrap();
                    let (k_cpu, v_cpu) = match (inner.k_cache.as_ref(), inner.v_cache.as_ref()) {
                        (Some(k), Some(v)) => (
                            k.to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| k.clone()),
                            v.to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| v.clone()),
                        ),
                        _ => continue,
                    };
                    let is_dirty = {
                        let reg = attn.registry.entries.read().unwrap();
                        if inner.index < reg.len() && l_idx < reg[inner.index].is_dirty.len() {
                            reg[inner.index].is_dirty[l_idx]
                        } else { true }
                    };

                    if is_dirty {
                        let k_shape_u32: Vec<u32> = k_cpu.shape().dims().iter().map(|&x| x as u32).collect();
                        block_groups.entry((inner.offset, inner.index)).or_default().push(LayerKVDump {
                            layer_idx: l_idx,
                            k_data: candle_core::Tensor::zeros((1,), candle_core::DType::U8, &candle_core::Device::Cpu).unwrap(),
                            v_data: candle_core::Tensor::zeros((1,), candle_core::DType::U8, &candle_core::Device::Cpu).unwrap(),
                            k_shape: candle_core::Tensor::from_vec(k_shape_u32, (k_cpu.shape().dims().len(),), &candle_core::Device::Cpu).unwrap(),
                            raw_k: Some(k_cpu.clone()),
                            raw_v: Some(v_cpu.clone()),
                            tq_meta: inner.tq,
                        });

                        let mut reg = attn.registry.entries.write().unwrap();
                        if inner.index < reg.len() {
                            reg[inner.index].is_dirty[l_idx] = false;
                        }
                    }

                    inner.k_cache = Some(k_cpu);
                    inner.v_cache = Some(v_cpu);
                    inner.location = crate::models::qwen::quantized_model::KVLocation::RAM;

                    let mut reg = attn.registry.entries.write().unwrap();
                    if inner.index < reg.len() {
                        reg[inner.index].location[l_idx] = crate::models::qwen::quantized_model::KVLocation::RAM;
                    }
                }
            }

            if layer.is_ssm_dirty() {
                let (conv_opt, rec_opt) = layer.get_ssm_states();
                if let (Some(conv), Some(rec)) = (conv_opt, rec_opt) {
                    let conv_cpu = conv.to_device(&candle_core::Device::Cpu).unwrap_or(conv);
                    let rec_cpu = rec.to_device(&candle_core::Device::Cpu).unwrap_or(rec);
                    ssm_dumps.push((l_idx, conv_cpu, rec_cpu));
                    layer.set_ssm_dirty(false);
                }
            }
        }

        let kv_dir = crate::utils::paths::get_kv_dir();
        let mode = false; 
        
        
        let kv_name_raw = kv_name.unwrap_or("text");
        let last_part = kv_name_raw.split('/').last().unwrap_or("text");
        let kv_type = if last_part == "inference" || last_part == "reference" || last_part.is_empty() { "text" } else { last_part };
        let sub_path = format!("{}/inference/{}", session_id, kv_type);
        let base_dir = kv_dir.join(&sub_path);

        if !ssm_dumps.is_empty() {
            let ssm_dir = base_dir.join("ssm");
            if !ssm_dir.exists() { let _ = std::fs::create_dir_all(&ssm_dir); }

            let dump_count = ssm_dumps.len();
            crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_add(dump_count, std::sync::atomic::Ordering::SeqCst);

            tokio::task::spawn_blocking(move || {
                for (l_idx, conv, rec) in ssm_dumps {
                    let mut map = std::collections::HashMap::new();
                    map.insert("conv_state".to_string(), conv);
                    map.insert("recurrent_state".to_string(), rec);

                    let st_path = ssm_dir.join(format!("l{}.st", l_idx));
                    let tmp_path = st_path.with_extension("tmp");
                    
                    if candle_core::safetensors::save(&map, &tmp_path).is_ok() {
                        if let Ok(plain_data) = std::fs::read(&tmp_path) {
                            if let Ok(encrypted_data) = crate::utils::crypto::encrypt_data(&plain_data) {
                                let _ = std::fs::write(&st_path, encrypted_data);
                            }
                        }
                        let _ = std::fs::remove_file(tmp_path);
                    }
                    crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                }
            });
        }

        if block_groups.is_empty() { return Ok(()); }

        if let Some(tx) = BAKE_TX.get() {
            for ((off, idx), layers) in block_groups {
                let sid = SLOT_MANAGER.acquire_write_slot(1024).await;
                let block_dir = base_dir.join(format!("b{}", off));
                if !block_dir.exists() { let _ = std::fs::create_dir_all(&block_dir); }

                crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                if tx.send(SlotTask::Bake(BakeTask {
                    slot_id: sid, task_dir: block_dir, kv_name: Some(sub_path.clone()), offset: off, layers,
                    is_relay_baking: mode, block_idx: Some(idx), registry: self.registry.clone(),
                })).await.is_err() {
                    crate::models::qwen::generate::GLOBAL_IO_COUNTER.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    SLOT_MANAGER.release_slot(sid).await;
                }
            }
        }
        Ok(())
    }
}

/// 🌟 [P2B-TEST] 실모델 대조 테스트(tests/it/p2b_models.rs)는 앱의 작업 중지 신호 파일(EXTRACTION_STOPPED)과
/// 무관하게 프리필해야 합니다. 앱 상태 파일을 건드리지 않도록 테스트 프로세스 안에서만 이 검사를 끕니다.
/// 기본값 false = 기존 동작 (앱 코드는 이 값을 바꾸지 않습니다).
pub static IGNORE_STOP_SIGNAL_FOR_TESTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Minimum prefix tokens required before capturing a reusable hybrid prompt snapshot.
pub const MIN_PREFIX_SNAPSHOT_TOKENS: usize = 64;
/// Minimum free VRAM required to hold the active GPU prefix snapshot without inducing KV offloading.
pub const HYBRID_SNAPSHOT_VRAM_MIN_FREE_BYTES: u64 = 192 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheAccessMode {
    ReadWrite,
    ReadOnly,
}

#[derive(Clone)]
pub struct SavedKvBlock {
    pub index: usize,
    pub len: usize,
    pub offset: usize,
    pub kv_exact: bool,
    pub kv_half: bool,
    pub k: Tensor,
    pub v: Tensor,
    pub tq: Option<crate::models::common::kv_turbo::TqMeta>,
}

#[derive(Clone)]
pub enum FullAttnKvSnapshot {
    DenseBlocks(Vec<SavedKvBlock>),
    Fp8Blocks(Vec<SavedKvBlock>),
    TurboRaw(Vec<SavedKvBlock>),
}

#[derive(Clone)]
pub enum HybridLayerState {
    Linear {
        conv_state: Option<Tensor>,
        recurrent_state: Option<Tensor>,
    },
    Full(FullAttnKvSnapshot),
}

#[derive(Clone)]
pub struct HybridPromptSnapshot {
    pub tokens: Vec<u32>,
    pub layers: Vec<HybridLayerState>,
    pub last_hidden: Tensor,
    pub rope_deltas_cpu: Option<Vec<i64>>,
    pub kv_exact: bool,
    pub kv_half: bool,
    pub on_vram: bool,
}

#[derive(Default)]
pub struct HybridSnapshotCache {
    pub system_host_slot: Option<HybridPromptSnapshot>,
    pub active_vram_slot: Option<HybridPromptSnapshot>,
    pub prev_prompt_tokens: Vec<u32>,
}

impl HybridSnapshotCache {
    pub fn longest_prefix_match(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }

    pub fn lookup_best(&self, tokens: &[u32]) -> Option<(&HybridPromptSnapshot, usize)> {
        let mut best: Option<(&HybridPromptSnapshot, usize, bool)> = None;
        for cand in [self.active_vram_slot.as_ref(), self.system_host_slot.as_ref()]
            .into_iter()
            .flatten()
        {
            let clen = cand.tokens.len();
            if clen < MIN_PREFIX_SNAPSHOT_TOKENS || clen > tokens.len() {
                continue;
            }
            if &tokens[..clen] == cand.tokens.as_slice() {
                let replace = match best {
                    None => true,
                    Some((_, best_len, best_vram)) => {
                        clen > best_len || (clen == best_len && cand.on_vram && !best_vram)
                    }
                };
                if replace {
                    best = Some((cand, clen, cand.on_vram));
                }
            }
        }
        best.map(|(snap, len, _)| (snap, len))
    }

    pub fn store_system_host(&mut self, mut snap: HybridPromptSnapshot) {
        snap.on_vram = false;
        self.system_host_slot = Some(snap);
    }

    pub fn store_active(&mut self, snap: HybridPromptSnapshot) {
        self.prev_prompt_tokens = snap.tokens.clone();
        self.active_vram_slot = Some(snap);
    }

    pub fn clear(&mut self) {
        self.system_host_slot = None;
        self.active_vram_slot = None;
        self.prev_prompt_tokens.clear();
    }
}

// ----------------------------------------------------------------------------------
// [Qwen3_5Model] 최상위 모델 구조 수정 및 인자 전달
// ----------------------------------------------------------------------------------
pub struct Qwen3_5Model {
    spatial_merge_size: usize,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    visual: Option<Qwen3VLVisionModel>,
    pub language_model: Qwen3_5TextModel,
    lm_head: ProjKind,
    /// 🌟 [VOCAB TRIM] 남긴 행만의 출력 투영 (있으면 `lm_head` 대신). 생성기가 문서 언어로 맞춥니다.
    pub trim_head: Option<crate::models::common::vocab_trim::TrimmedHead>,
    rope_deltas: Option<Tensor>,
    rope_deltas_cpu: Option<Vec<i64>>,
}

impl Qwen3_5Model {
    pub fn new_from_vb(vb: VarBuilder, config: Qwen3_5Config) -> Result<Self> {
        let vb_m = vb.pp("model");
        let visual = Qwen3VLVisionModel::new(config.vision_config.clone(), vb_m.pp("visual"))?;
        let language_model =
            Qwen3_5TextModel::new_from_vb(vb_m.pp("language_model"), &config.text_config)?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(
                language_model
                    .embed_tokens
                    .dense()
                    .expect("safetensors path always builds a dense embedding table")
                    .clone(),
                None,
            )
        } else {
            linear_no_bias(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                vb.pp("lm_head"),
            )?
        };
        Ok(Self {
            spatial_merge_size: config.vision_config.spatial_merge_size,
            image_token_id: config.image_token_id,
            video_token_id: config.video_token_id,
            vision_start_token_id: config.vision_start_token_id,
            vision_end_token_id: config.vision_end_token_id,
            visual: Some(visual),
            language_model,
            lm_head: ProjKind::LinearProj(lm_head),
            trim_head: None,
            rope_deltas: None,
            rope_deltas_cpu: None,
        })
    }

    pub fn new_from_gguf<R1: Read + Seek, R2: Read + Seek>(
        gguf: &mut Gguf<R1>,
        mmproj_gguf: Option<&mut Gguf<R2>>,
        device: &Device,
        mmap_handle: Option<std::sync::Arc<memmap2::Mmap>>,
        ct_handle: Option<std::sync::Arc<candle_core::quantized::gguf_file::Content>>
    ) -> Result<Self> {
        let spatial_merge_size = 2usize;
        let lookup_gguf_tok = |gguf: &Gguf<R1>, meta_key: &str, tok_str: &str, default_id: u32| -> u32 {
            if let Ok(v) = gguf.get_matedata(meta_key) {
                if let Ok(id) = v.to_u32() {
                    return id;
                }
            }
            if let Ok(arr) = gguf.get_matedata("tokenizer.ggml.tokens").and_then(|v| Ok(v.to_vec()?.clone())) {
                if let Some(pos) = arr.iter().position(|x| x.to_string().ok().is_some_and(|s| s == tok_str)) {
                    return pos as u32;
                }
            }
            default_id
        };
        let vision_start_token_id = lookup_gguf_tok(gguf, "qwen35.vision_start_token_id", "<|vision_start|>", 248053);
        let vision_end_token_id = lookup_gguf_tok(gguf, "qwen35.vision_end_token_id", "<|vision_end|>", 248054);
        let image_token_id = lookup_gguf_tok(gguf, "qwen35.image_token_id", "<|image_pad|>", 248056);
        let video_token_id = lookup_gguf_tok(gguf, "qwen35.video_token_id", "<|video_pad|>", 248057);

        // 🌟 [LOAD ORDER] 언어 모델을 먼저 세웁니다.
        //    레이어는 new_skeleton 이라 VRAM 을 쓰지 않고, 여기서 올라가는 것은
        //    embed_tokens(F16 1,017MB) 와 output_norm 뿐입니다.
        //    비전을 먼저 세우던 구버전은 비전 1.44GB 위에 이 1GB 를 얹어
        //    피크를 그대로 합산시켰습니다.
        let language_model = Qwen3_5TextModel::new_from_gguf(gguf, device, mmap_handle, ct_handle)?;

        // 🌟 [TIED LM HEAD] output.weight 가 없으면 embed_tokens 와 '같은 가중치' 입니다.
        //
        //  ── 실측 사고 ──
        //   구버전은 output.weight 가 없을 때 token_embd.weight 를 한 번 더 읽었습니다.
        //     · embed_tokens : F16 역양자화본  248,320 × 2,048 × 2B = 1,017MB
        //     · lm_head      : 같은 텐서의 quantized 본           ≈   540MB
        //   즉 동일 가중치가 VRAM 에 두 벌 존재했고, 정상 구간 소비 1.81GB 중
        //   540MB 가 순수 중복이었습니다.
        //
        //  ── 왜 TensorF16 으로 공유해도 되는가 ──
        //   candle 의 QMatMul::TensorF16 은 forward 에서 입력을 F16 으로 캐스팅하고
        //   matmul 후 원래 dtype 으로 되돌립니다. 아래 forward 는 hidden 을
        //   F32 로 넘기므로(hidden_state_aligned) 이 경로가 정확히 맞습니다.
        //   Tensor::clone 은 스토리지 Arc 참조 복사라 새 할당이 없습니다.
        let lm_head = match gguf.tensor("output.weight") {
            Ok(tensor) => {
                println!("[MODEL] lm_head: dedicated 'output.weight' loaded (untied).");
                ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_qtensor(tensor)?, None))
            }
            Err(_) => match &language_model.embed_tokens {
                EmbedTable::Quant { q, .. } => {
                    ProjKind::QuantizedProj(QuantizedLinear::new(QMatMul::from_arc(q.clone())?, None))
                }
                // 매핑된 표: 출력 투영은 남긴 행만의 `trim_head` 로 곱합니다 (여기 자리는 쓰지 않는 1 행).
                EmbedTable::Mapped { src, .. } => {
                    ProjKind::LinearProj(Linear::new(Tensor::zeros((1, src.cols), DType::F32, device)?, None))
                }
                EmbedTable::Dense(e) => {
                    let shared = e.embeddings().clone();
                    let qm = if shared.dtype() == DType::F16 {
                        QMatMul::TensorF16(shared)
                    } else {
                        QMatMul::Tensor(shared)
                    };
                    println!(
                        "[MODEL] lm_head: 'output.weight' absent → tied to embed_tokens. Duplicate {:.0}MB avoided.",
                        (e.embeddings().elem_count() as f64) * 2.0 / 1e6
                    );
                    ProjKind::QuantizedProj(QuantizedLinear::new(qm, None))
                }
            },
        };

        // 🌟 [VISION-SKELETON] 여기서는 비전을 만들지 않습니다.
        //    호출부가 attach_vision_skeleton() 으로 '가중치 0바이트' 구조만 붙입니다.
        //    (하위 호환: mmproj_gguf 가 주어지면 구버전대로 전량 적재합니다)
        let visual = if let Some(mmproj) = mmproj_gguf {
            println!("[MODEL] ⚠️ Vision tower loaded EAGERLY (legacy path). Peak VRAM will include full mmproj.");
            Some(Qwen3VLVisionModel::new_from_gguf(mmproj)?)
        } else {
            None
        };

        Ok(Self {
            spatial_merge_size,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            vision_end_token_id,
            visual,
            language_model,

            lm_head,
            trim_head: None,
            rope_deltas: None,
            rope_deltas_cpu: None,
        })
    }

    /// 🌟 [VISION-SKELETON] mmproj 의 메타데이터만 읽어 비전 구조를 붙입니다.
    ///    이 시점의 VRAM 증가는 0 이며, 실제 가중치는
    ///    forward 가 캐시 MISS 를 확인한 뒤 블록 단위로 읽어 옵니다.
    pub fn attach_vision_skeleton(&mut self, mmproj_path: &str, device: &Device) -> Result<()> {
        self.visual = Some(Qwen3VLVisionModel::new_skeleton_from_mmproj(mmproj_path, device)?);
        Ok(())
    }

    
    /// 🌟 [VISION-JIT] mmproj GGUF 실제 파일 경로를 비전 모델에 등록합니다.
    /// 이 호출이 없으면 unload/reload 가 전부 no-op 이 되어 기존 거동이 100% 보존됩니다.
    pub fn set_mmproj_path(&mut self, path: Option<String>) {
        if let Some(v) = self.visual.as_mut() {
            v.set_mmproj_path(path);
        }
    }

    /// 🌟 [VISION-JIT] mmproj 재로드 소스가 등록되어 JIT 이 가능한 상태인지 확인합니다.
    pub fn is_vision_jit_capable(&self) -> bool {
        self.visual.as_ref().map(|v| v.is_jit_capable()).unwrap_or(false)
    }

    /// 🌟 [VISION-JIT] 현재 비전 가중치가 메모리에 상주 중인지 확인합니다.
    pub fn is_vision_resident(&self) -> bool {
        self.visual.as_ref().map(|v| v.is_weights_loaded).unwrap_or(false)
    }

    /// 🌟 [VISION-JIT] 비전 가중치만 붙였다 뗍니다. 텍스트 모델은 그대로 상주합니다.
    /// active=false 는 텍스트 전용 추론 진입 시, true 는 이미지/비디오 입력 감지 시 호출합니다.
    ///
    /// 🌟 [LAZY ATTACH] active=true 는 더 이상 즉시 로드하지 않습니다.
    ///
    ///  ── 실측 사고 ──
    ///   ensure_qwen3_5(true) → set_vision_active(true) → reload_weights() 가
    ///   '비전 캐시 조회보다 먼저' 실행됐습니다. 로그가 그대로 보여줍니다.
    ///     [VISION-JIT] mmproj weights reloaded (streaming: true) ...
    ///     [VISION-CACHE] HIT key=... ViT 27-block attention skipped.
    ///     [VISION-JIT] mmproj weights unloaded.
    ///   스트리밍이라 블록은 안 올라가지만 load_shared_weights()
    ///   (patch_embed + pos_embed + merger fc1/fc2 ≈ 75MB) 는 매번 왕복했습니다.
    ///   크롭 9개 기준 약 1.4GB 의 무의미한 PCIe 전송입니다.
    ///
    ///  ── 왜 지연해도 안전한가 ──
    ///   Qwen3_5Model::forward 는 캐시 MISS 를 확인한 직후
    ///     if !visual.is_weights_loaded && visual.is_jit_capable() { visual.reload_weights()?; }
    ///   를 스스로 수행합니다. 비디오 경로도 동일한 가드를 갖고 있습니다.
    ///   JIT 불가(safetensors 경로)인 인스턴스는 is_weights_loaded 가 true 로 고정이라
    ///   unload 도 no-op 이므로 가중치가 사라지지 않습니다.
    pub fn set_vision_active(&mut self, active: bool) -> Result<()> {
        if let Some(v) = self.visual.as_mut() {
            if active {
                // 로드하지 않습니다. forward 가 실제 필요 시점에 읽습니다.
                if !v.is_weights_loaded && v.is_jit_capable() {
                    println!("[VISION-JIT] Attach requested — deferred until a cache MISS actually needs the ViT.");
                }
            } else {
                v.unload_weights();
            }
        }
        Ok(())
    }

    /// 🌟 [VISION-STREAM] 비전 27블록을 mmap에서 하나씩 읽고 즉시 폐기하는 스트리밍을 켭니다.
    ///   상주 가중치 337MB → 약 12MB 로 감소. 이미지당 PCIe 전송 27회 추가(~135ms).
    ///   OCR은 이미지 1장당 1회이므로 체감 영향 미미합니다.
    ///   mmproj source 미등록 시 내부에서 안전하게 무시됩니다.
    pub fn set_block_streaming(&mut self, enabled: bool) {
        if let Some(v) = self.visual.as_mut() {
            v.set_block_streaming(enabled);
        }
    }

    pub fn compute_and_set_rope_deltas(
        &mut self,
        full_input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
    ) -> Result<()> {
        let (_, deltas, deltas_cpu) = self.get_rope_index(full_input_ids, image_grid_thw, video_grid_thw, None)?;
        self.rope_deltas = Some(deltas);
        self.rope_deltas_cpu = Some(deltas_cpu);
        Ok(())
    }

    pub fn embed_vocab(&self) -> candle_core::Result<usize> {
        self.language_model.embed_tokens.vocab()
    }

    pub fn embed_rows_f32(&self, off: usize, take: usize) -> candle_core::Result<Tensor> {
        self.language_model.embed_tokens.rows_f32_with_head(off, take, self.trim_head.as_ref())
    }

    pub fn embedding_token_id(&self, input_ids: &Tensor) -> Result<Tensor> {
        Ok(self.language_model.embed_tokens.forward_with_head(input_ids, self.trim_head.as_ref())?)
    }

    fn get_rope_index(
        &self,
        input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        _video_grid_thw: Option<&Tensor>,
        _mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Vec<i64>)> {
        let spatial_merge_size = self.spatial_merge_size;
        let image_token_id = self.image_token_id;
        let vision_start_token_id = self.vision_start_token_id;
        
        let (b_sz, seq_len) = input_ids.dims2()?;
        let mut mrope_position_deltas = Vec::new();

        let input_ids_vec = input_ids.to_vec2::<u32>()?;
        let mut image_idx = 0;

        let image_thw_cpu = if let Some(thw) = image_grid_thw {
            Some(thw.to_device(&Device::Cpu)?.to_vec2::<u32>()?)
        } else { None };

        let mut flat_pos_ids: Vec<u32> = Vec::with_capacity(3 * b_sz * seq_len);

        for b in 0..b_sz {
            let ids = &input_ids_vec[b];
            let mut curr_pos = 0u32;
            let mut llm_pos_ids = vec![vec![0u32; seq_len]; 3];
            let mut i = 0;
            
            while i < seq_len {
                // 💡 vision_start를 믿지 않고, 확실한 image_token_id만으로 3D 격자를 발동시킵니다.
                if ids[i] == image_token_id {
                    if let Some(thw_cpu_array) = &image_thw_cpu {
                        if image_idx < thw_cpu_array.len() {
                            let thw = &thw_cpu_array[image_idx];
                            image_idx += 1;
                            let (t, h, w) = (thw[0], thw[1] / spatial_merge_size as u32, thw[2] / spatial_merge_size as u32);
                            
                            let img_len = (t * h * w) as usize;
                            for tt in 0..t {
                                for hh in 0..h {
                                    for ww in 0..w {
                                        let idx = i + (tt * h * w + hh * w + ww) as usize;
                                        if idx < seq_len {
                                            llm_pos_ids[0][idx] = curr_pos + tt;
                                            llm_pos_ids[1][idx] = curr_pos + hh;
                                            llm_pos_ids[2][idx] = curr_pos + ww;
                                        }
                                    }
                                }
                            }
                            i += img_len;
                            curr_pos += t.max(h).max(w); 
                            continue; // 이미지 블록 처리가 끝났으므로 다음 루프로 직행
                        }
                    }
                }
                
                // 일반 텍스트 토큰 및 쪼개진 특수 토큰 처리
                for d in 0..3 { llm_pos_ids[d][i] = curr_pos; }
                i += 1;
                curr_pos += 1;
            }
            
            for d in 0..3 {
                flat_pos_ids.extend_from_slice(&llm_pos_ids[d]);
            }
            mrope_position_deltas.push(curr_pos as i64 - seq_len as i64);
        } 

        
        let position_ids = Tensor::from_vec(flat_pos_ids, (3, b_sz, seq_len), &Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_device(input_ids.device())?;
            
        let target_dtype = if input_ids.device().is_cuda_or_rocm() { DType::BF16 } else { DType::F32 };
        let deltas = Tensor::from_vec(mrope_position_deltas.clone(), (b_sz, 1), &Device::Cpu)?
            .to_dtype(DType::F32)? 
            .to_device(input_ids.device())? 
            .to_dtype(target_dtype)?; 
        
        Ok((position_ids.contiguous()?, deltas, mrope_position_deltas))
    }

    fn compute_3d_position_ids(
        &mut self,
        input_ids: &Tensor,
        inputs_embeds: &Tensor,
        image_grid_thw: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = inputs_embeds.dims3()?;

        
        // 디코딩 중에는 절대 재계산하지 않고, 저장해둔 오프셋(delta)을 유지하여 위치 연속성을 100% 보장합니다.
        if seqlen_offset == 0 {
            let (pos_ids, rope_deltas, deltas_cpu) = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, None)?;
            self.rope_deltas = Some(rope_deltas);
            self.rope_deltas_cpu = Some(deltas_cpu);
            return Ok(pos_ids.to_device(inputs_embeds.device())?);
        }

        // 이후 1글자씩 뱉어낼 때는 과거로 돌아가지 않고 정상적으로 +1씩 전진합니다.
        if self.rope_deltas_cpu.is_none() {
            self.rope_deltas_cpu = Some(vec![0i64; bs]);
        }

        let deltas_cpu = self.rope_deltas_cpu.as_ref().unwrap();
        let mut p_ids_vec = Vec::with_capacity(bs);
        for b in 0..bs {
            let delta = deltas_cpu[b];
            let real_start = (seqlen_offset as i64 + delta) as f32;
            let p_id = if seq_len == 1 {
                // arange 대신 단일 스칼라 텐서를 즉시 생성
                Tensor::new(&[real_start], &Device::Cpu)?
                    .reshape((1, 1, 1))?
                    .broadcast_as((3, 1, 1))?
            } else {
                // Suffix prefill after prefix cache restore (seq_len > 1, seqlen_offset > 0)
                Tensor::arange(0f32, seq_len as f32, &Device::Cpu)?
                    .affine(1.0, real_start as f64)?
                    .reshape((1, 1, seq_len))?
                    .broadcast_as((3, 1, seq_len))?
            };
            p_ids_vec.push(p_id);
        }
        
        
        // GPU로 넘기기 전에 무조건 contiguous()로 물리적 메모리 연속성을 100% 보장합니다!
        let position_ids = Tensor::cat(&p_ids_vec, 1)?.contiguous()?.to_device(inputs_embeds.device())?;
        Ok(position_ids)
    }

    pub async fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        pixel_values_video: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        cache_position: Option<&Tensor>, 
        seqlen_offset: usize,
        session_id: Option<String>,
        kv_name: Option<String>,
    ) -> Result<Tensor> {
        let hidden_state_aligned = self
            .forward_last_hidden(
                input_ids,
                pixel_values,
                image_grid_thw,
                pixel_values_video,
                video_grid_thw,
                cache_position,
                seqlen_offset,
                session_id,
                kv_name,
            )
            .await?;
        self.logits_from_hidden(&hidden_state_aligned)
    }

    /// 마지막 토큰 위치의 F32 정규화 은닉 벡터 `[B, 1, H]` 만 반환합니다 (`lm_head` 생략).
    pub async fn forward_last_hidden(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        pixel_values_video: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        cache_position: Option<&Tensor>,
        seqlen_offset: usize,
        session_id: Option<String>,
        kv_name: Option<String>,
    ) -> Result<Tensor> {
        // 🌟 [SPD-06] 순수 텍스트 디코딩 스텝마다 input_ids 를 CPU 로 D2H 복사하지 않습니다.
        //    비전 마스킹 또는 프리필(total_len > 1, offset == 0) 비전 태깅이 필요할 때만 지연 복사합니다.
        let need_cpu_ids = pixel_values.is_some()
            || pixel_values_video.is_some()
            || (input_ids.dim(1)? > 1 && seqlen_offset == 0);
        let input_ids_cpu = if need_cpu_ids {
            Some(input_ids.to_device(&Device::Cpu)?)
        } else {
            None
        };

        // 🌟 [VISION-JIT + CACHE] 여기서 무조건 mmproj 를 붙이면
        //    캐시 히트로 ViT 를 건너뛸 때도 600MB 를 헛되이 올리게 됩니다.
        //    reload 는 실제로 ViT 를 호출하기 직전(캐시 MISS 확정 후)으로 미룹니다.
        //    비디오는 캐시 대상이 아니므로 여기서 즉시 붙입니다.
        if pixel_values_video.is_some() {
            if let Some(visual) = self.visual.as_mut() {
                if !visual.is_weights_loaded && visual.is_jit_capable() {
                    visual.reload_weights()?;
                }
            }
        }
        
        let mut inputs_embeds = self.language_model.embed_tokens.forward_with_head(input_ids, self.trim_head.as_ref())?;
        // 🌟 [PRC-13] 모델의 실제 활성 dtype(`self.language_model.dtype`)으로 맞춥니다.
        //    (기존 BF16 고정 캐스트는 2B F16(10-bit) 과 4B F32(23-bit) 를 7-bit BF16 으로 절단했습니다.)
        let target_dtype = self.language_model.dtype;
        if inputs_embeds.dtype() != target_dtype {
            inputs_embeds = inputs_embeds.to_dtype(target_dtype)?;
        }
        
        if let Some(pixel_values) = pixel_values {
            if let Some(image_grid_thw) = image_grid_thw {
                // 외부 유틸리티 대신 CPU에서 직접 마스크를 만들어 CUDA U32/U8 충돌 원천 차단
                let ids_cpu = input_ids_cpu.as_ref().expect("input_ids_cpu initialized for vision");
                let img_token_t = Tensor::new(vec![self.image_token_id as f32], &Device::Cpu)?;
                let mask_cpu = ids_cpu.to_dtype(DType::F32)?.broadcast_eq(&img_token_t)?.to_dtype(DType::U8)?;
                let vision_mask = mask_cpu.to_device(input_ids.device())?;

                // 🌟 [VISION-CACHE] 프롬프트가 요구하는 이미지 토큰 수를 먼저 확정합니다.
                //    캐시본이 이 수와 다르면 masked_scatter_dim0 이 붕괴하므로 반드시 검증합니다.
                let expected_tokens = vision_mask
                    .to_dtype(candle_core::DType::F32)?
                    .sum_all()?
                    .to_scalar::<f32>()? as usize;

                // 🌟 [VISION-CACHE] ViT 는 결정론적 feed-forward 이므로
                //    동일 (pixel_values, grid_thw) 조합의 결과를 디스크에서 그대로 재사용합니다.
                //    pixel_values 는 이미 리사이즈/정규화가 끝난 상태라
                //    Part 6 의 적응형 해상도 변화가 해시에 자동 반영됩니다.
                let cache_key = crate::models::vision_cache::VisionEmbedCache::compute_key(
                    pixel_values, image_grid_thw
                ).ok();

                let mut image_embeds: Option<Tensor> = None;

                if let Some(key) = cache_key {
                    if let Some((cached, _cached_ds)) = crate::models::vision_cache::VISION_CACHE
                        .try_load(key, inputs_embeds.device(), inputs_embeds.dtype())
                    {
                        match crate::models::vision_cache::validate_embed_shape(&cached, expected_tokens) {
                            Ok(_) => { image_embeds = Some(cached); }
                            Err(e) => {
                                println!("[VISION-CACHE] Discarding stale cache entry: {}. Falling back to ViT.", e);
                            }
                        }
                    }
                }

                if image_embeds.is_none() {
                    // 🌟 [VISION-JIT] 캐시 MISS 가 확정된 이 순간에만 mmproj 를 mmap 에서 붙입니다.
                    //    캐시 히트 경로에서는 600MB 를 아예 올리지 않습니다.
                    // 🌟 [VISION-STREAM] forward 가 블록을 하나씩 읽고 버리므로 &mut 借用 이 필요합니다.
                    //    reload 와 forward 를 하나의 as_mut() 借用 으로 묶습니다.
                    if let Some(visual) = self.visual.as_mut() {
                        if !visual.is_weights_loaded && visual.is_jit_capable() {
                            visual.reload_weights()?;
                        }

                        let (computed, deepstacks): (Tensor, Vec<Tensor>) =
                            visual.forward(pixel_values, image_grid_thw)?;

                        if let Some(key) = cache_key {
                            if let Err(e) = crate::models::vision_cache::VISION_CACHE
                                .save(key, &computed, &deepstacks)
                            {
                                println!("[VISION-CACHE] Save failed (non-fatal): {}", e);
                            }
                        }
                        image_embeds = Some(computed);
                    }
                }

                if let Some(image_embeds) = image_embeds {
                    let image_embeds = image_embeds.to_dtype(inputs_embeds.dtype())?;
                    inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?;
                }
            }
        }
        if let Some(pixel_values_video) = pixel_values_video {
            if let Some(video_grid_thw) = video_grid_thw {
                // 🌟 [VISION-STREAM] forward 가 &mut self 로 바뀌었으므로 as_mut() 로 받습니다.
                //    (embeds 를 먼저 확보한 뒤 inputs_embeds 를 만지면 借用 충돌이 없습니다)
                let video_embeds_opt: Option<Tensor> = if let Some(visual) = self.visual.as_mut() {
                    let (video_embeds, _): (Tensor, _) = visual.forward(pixel_values_video, video_grid_thw)?;
                    Some(video_embeds)
                } else {
                    None
                };

                if let Some(video_embeds) = video_embeds_opt {
                    let ids_cpu = input_ids_cpu.as_ref().expect("input_ids_cpu initialized for video");
                    let vid_token_t = Tensor::new(vec![self.video_token_id as f32], &Device::Cpu)?;
                    let mask_cpu = ids_cpu.to_dtype(DType::F32)?.broadcast_eq(&vid_token_t)?.to_dtype(DType::U8)?;
                    let vision_mask = mask_cpu.to_device(input_ids.device())?;
                    
                    let video_embeds = video_embeds.to_dtype(inputs_embeds.dtype())?;
                    inputs_embeds = masked_scatter_dim0(&inputs_embeds, &video_embeds, &vision_mask)?;
                }
            }
        }

        // 🌟 [VISION-JIT] ViT 는 상태 없는 1회성 feed-forward 이고, 그 출력은 이미 inputs_embeds 에
        //    병합이 끝났습니다. 프리필(chunk_size 2048)이 시작되기 전에 mmproj 를 반환해야
        //    그 VRAM 을 어텐션/KV 가 그대로 쓸 수 있습니다.
        if pixel_values.is_some() || pixel_values_video.is_some() {
            if let Some(visual) = self.visual.as_mut() {
                visual.unload_weights();
            }
        }

        let position_ids;
        if cache_position.is_some() && seqlen_offset > 0 {
            let (bs, seq_len, _) = inputs_embeds.dims3()?;
            
            
            let delta = cache_position.unwrap().i(0)?
                .to_dtype(candle_core::DType::F32)? 
                .to_device(&Device::Cpu)? 
                .broadcast_add(&Tensor::new(seqlen_offset as f32, &Device::Cpu)?)?;
            
            position_ids = Tensor::arange(0f32, seq_len as f32, &Device::Cpu)?
                .unsqueeze(0)?
                .broadcast_as((bs, seq_len))?
                .broadcast_add(&delta)?
                .unsqueeze(0)?
                .broadcast_as((3, bs, seq_len))?
                .contiguous()?
                .to_device(input_ids.device())?;
        } else {
            position_ids = self.compute_3d_position_ids(input_ids, &inputs_embeds, image_grid_thw, video_grid_thw, seqlen_offset)?;
        }
        
        let total_len = inputs_embeds.dim(1)?;
        
        let chunk_size = 2048; 
        let mut final_hidden_state = None;

        // 🌟 [VISION TAG] 프리필 진입 직전, KV 블록별 비전 토큰 비율을 장부에 기록합니다.
        //    KV 텐서에는 어떤 손실 변형도 가하지 않는 순수 메타데이터입니다.
        //    (인접 토큰 평균 풀링은 K 의 RoPE 회전 위상을 파괴하므로 채택하지 않았습니다)
        if total_len > 1 && seqlen_offset == 0 {
            if let Some(ids_cpu) = input_ids_cpu.as_ref() {
                if let Ok(ids_2d) = ids_cpu.to_vec2::<u32>() {
                    if let Some(row) = ids_2d.first() {
                        self.language_model.registry.tag_vision_blocks(
                            row,
                            &[self.image_token_id, self.video_token_id],
                        );
                    }
                }
            }
        }

        if total_len > 1 {
            let mut processed = 0;
            let mut current_offset = seqlen_offset;
            // 🌟 [KV-PRECISION] 프리필이 청크로 나뉘어도 KV 정밀도는 프롬프트 전체 길이로 판정합니다.
            if seqlen_offset == 0 && self.language_model.planned_prompt_len < total_len {
                self.language_model.planned_prompt_len = total_len;
            }
            
            while processed < total_len {
                
                if !IGNORE_STOP_SIGNAL_FOR_TESTS.load(std::sync::atomic::Ordering::Relaxed)
                    && crate::utils::is_extraction_stopped()
                {
                    return Err(anyhow::anyhow!("Task cancelled"));
                }

                let take = (total_len - processed).min(chunk_size);
                let chunk_embeds = inputs_embeds.narrow(1, processed, take)?;
                let chunk_pos_ids = position_ids.narrow(2, processed, take)?;

                let outputs = self.language_model.forward(&chunk_embeds, &chunk_pos_ids, current_offset, session_id.clone(), kv_name.clone()).await?;

                if processed + take == total_len {
                    let seq_len = outputs.dim(1)?;
                    
                    final_hidden_state = Some(outputs.narrow(1, seq_len - 1, 1)?.contiguous()?);
                }

                if let Some(sid) = &session_id {
                    let want_flush = self.language_model.persist_kv
                        || self.language_model.kv_plan == Some(crate::utils::resources::KvResidency::Ssd);
                    if want_flush {
                        let _ = self.language_model.force_flush_all_active_blocks(sid, kv_name.as_deref()).await;
                    }
                }

                processed += take;
                current_offset += take;
                
                
                let pct = ((processed as f32 / total_len as f32) * 100.0) as i32;
                if let Some(tx) = crate::progress::PROGRESS_TX.get() {
                    if let Some(sid) = &session_id {
                        let task_id = if sid.starts_with("task_") || sid.starts_with("search_") || sid.starts_with("img_") {
                            let p: Vec<&str> = sid.split('_').collect();
                            if p.len() >= 2 { format!("{}_{}", p[0], p[1]) } else { sid.clone() }
                        } else { sid.clone() };
                        
                        
                        let current_cat = crate::progress::CURRENT_UI_CATEGORY.read().unwrap().clone();

                        let summary_msg = format!("Reading context ({}%)...", pct);

                        let _ = tx.send(serde_json::json!({
                            "task_id": task_id,
                            "category": format!("{} (Prefill)", current_cat),
                            "summary": summary_msg,
                            "spinner": "⠹"
                        }));
                    }
                }

                use std::io::Write;
                print!("\r[Qwen3.5-PREFILL] {} / {} tokens processed", processed, total_len);
                let _ = std::io::stdout().flush();
            }
            println!("\n[Qwen3.5-PREFILL] Complete. Starting Generation...");
        } else {
            let outputs = self.language_model.forward(&inputs_embeds, &position_ids, seqlen_offset, session_id.clone(), kv_name.clone()).await?;
            let seq_len = outputs.dim(1)?;
            
            final_hidden_state = Some(outputs.narrow(1, seq_len - 1, 1)?.contiguous()?);
        }

        let hidden_state = final_hidden_state.unwrap();
        Ok(hidden_state.contiguous()?.to_dtype(DType::F32)?)
    }

    /// 정규화된 마지막 은닉 벡터 `[B, 1, H]` → 전체 어휘 로짓 `[B, 1, V]` (CPU).
    pub fn logits_from_hidden(&mut self, hidden_state_aligned: &Tensor) -> Result<Tensor> {
        // 🌟 [VOCAB TRIM] 남긴 행만 곱하고 장치에서 전체 어휘 길이로 펼칩니다 (잘린 행 −∞). 전체 어휘 경로처럼 CPU 로 돌려줍니다.
        if self.trim_head.is_none() {
            if let Some(src) = self.language_model.embed_tokens.mapped_source().cloned() {
                // 생성기가 언어를 정하기 전(refresh_vocab 없이) 불린 계산은 전체 행으로 곱합니다 (off 와 같은 로짓)
                self.trim_head = Some(crate::models::common::vocab_trim::all_rows(&src, hidden_state_aligned.device())?);
            }
        }
        if let Some(head) = &self.trim_head {
            return Ok(head.logits(hidden_state_aligned)?.to_device(&Device::Cpu)?);
        }
        let logits_gpu = self.lm_head.forward(hidden_state_aligned)?;
        Ok(logits_gpu.to_device(&Device::Cpu)?)
    }

    /// Part 5 (Pack A·C) + Phase 3-1 D2/D4: 100% 무손실 Greedy 토큰 선택 + Guard 통계 반환.
    /// `TrimmedHead`, `ProjKind::Exact`/`QuantizedProj` (Mode A `fused_quant_argmax_cs`), `LinearProj` 모두 지원.
    pub fn greedy_from_hidden_with_stats(
        &mut self,
        hidden_state_aligned: &Tensor,
    ) -> Result<(u32, crate::models::common::vocab_trim::StepGuardStats)> {
        use crate::models::common::vocab_trim::StepGuardStats;
        if self.trim_head.is_none() {
            if let Some(src) = self.language_model.embed_tokens.mapped_source().cloned() {
                self.trim_head = Some(crate::models::common::vocab_trim::all_rows(&src, hidden_state_aligned.device())?);
            }
        }
        if let Some(head) = &self.trim_head {
            let (ids, st) = head.greedy_next_token_detailed(hidden_state_aligned)?;
            return Ok((ids[0], st));
        }
        let dev = hidden_state_aligned.device();
        let dt = hidden_state_aligned.dtype();
        let h_dim = *hidden_state_aligned.dims().last().unwrap_or(&0);
        let fused_ok = crate::models::common::accel_check::fused_op_ok(
            dev,
            crate::models::common::accel_check::FusedOpKind::LmHeadArgmaxCs,
            dt,
            h_dim,
        );
        if fused_ok && (dev.is_cuda() || dev.is_rocm()) {
            let q_and_bias = match &self.lm_head {
                ProjKind::Exact(ex) => Some((ex.weight(), None)),
                ProjKind::QuantizedProj(ql) => ql.qtensor().map(|q| (q, ql.bias())),
                _ => None,
            };
            if let Some((qw, bias)) = q_and_bias {
                if h_dim % qw.dtype().block_size().max(1) == 0
                    && matches!(
                        qw.dtype(),
                        candle_core::quantized::GgmlDType::Q8_0
                            | candle_core::quantized::GgmlDType::Q4_0
                            | candle_core::quantized::GgmlDType::Q4K
                            | candle_core::quantized::GgmlDType::Q5K
                            | candle_core::quantized::GgmlDType::Q6K
                    )
                {
                    let m = hidden_state_aligned.elem_count() / h_dim.max(1);
                    let h_2d = hidden_state_aligned.reshape((m, h_dim))?.contiguous()?;
                    let out = candle_core::lm_head::fused_quant_argmax_cs(&h_2d, qw, bias, None, None)?;
                    let (ids, _) = out.download_ids_and_mask()?;
                    return Ok((
                        ids[0],
                        StepGuardStats {
                            steps: 1,
                            fast_steps: 1,
                            quant_mode_a: true,
                            ..Default::default()
                        },
                    ));
                }
            }
        }
        if let ProjKind::LinearProj(lin) = &self.lm_head {
            if fused_ok {
                let out = candle_core::lm_head::fused_linear_argmax_cs(
                    hidden_state_aligned,
                    lin.weight(),
                    lin.bias(),
                    None,
                    None,
                )?;
                let id = out.ids.to_device(&Device::Cpu)?.to_vec1::<u32>()?[0];
                return Ok((
                    id,
                    StepGuardStats {
                        steps: 1,
                        fast_steps: 1,
                        ..Default::default()
                    },
                ));
            }
        }
        let logits = self.logits_from_hidden(hidden_state_aligned)?;
        let id = logits
            .to_dtype(DType::F32)?
            .argmax(D::Minus1)?
            .flatten_all()?
            .to_vec1::<u32>()?[0];
        Ok((
            id,
            StepGuardStats {
                steps: 1,
                ..Default::default()
            },
        ))
    }

    /// Part 5 (Pack A·C): 100% 무손실 Greedy 토큰 선택 (Mode B `TrimmedHead::greedy_next_token` 또는 Mode A `fused_linear_argmax_cs`).
    pub fn greedy_from_hidden(&mut self, hidden_state_aligned: &Tensor) -> Result<u32> {
        self.greedy_from_hidden_with_stats(hidden_state_aligned).map(|(id, _)| id)
    }

    /// 매핑된 출력 투영 표 (잘라 내기가 켜진 경우)
    pub fn vocab_table(&self) -> Option<crate::models::common::vocab_trim::TableSource> {
        self.language_model.embed_tokens.mapped_source().cloned()
    }

    /// Actual resident GPU weight + active LM head bytes (Phase 3-2 D2: reflects `EmbedTable::Mapped` 0 B table).
    pub fn resident_vram_bytes(&self) -> u64 {
        let mut total = 0u64;
        if let Some(ct) = &self.language_model.ct {
            let resident_layers = if self.language_model.keep_weights_resident {
                self.language_model.layers.len()
            } else {
                self.language_model.resident_prefix
            };
            for (name, info) in ct.tensor_infos.iter() {
                if name == "token_embd.weight" || name == "output.weight" {
                    continue;
                }
                if let Some(rest) = name.strip_prefix("blk.") {
                    if let Some(idx_str) = rest.split('.').next() {
                        if let Ok(idx) = idx_str.parse::<usize>() {
                            if idx >= resident_layers {
                                continue;
                            }
                        }
                    }
                }
                let bs = info.ggml_dtype.block_size().max(1);
                let bytes = (info.shape.elem_count() / bs) * info.ggml_dtype.type_size();
                total += bytes as u64;
            }
        }
        match &self.language_model.embed_tokens {
            EmbedTable::Mapped { .. } => {
                // Input table VRAM = 0; only active trim_head is resident on GPU.
                if let Some(head) = &self.trim_head {
                    total += head.head_bytes() as u64;
                }
            }
            EmbedTable::Quant { q, .. } => {
                let bs = q.dtype().block_size().max(1);
                total += ((q.shape().elem_count() / bs) * q.dtype().type_size()) as u64;
            }
            EmbedTable::Dense(e) => {
                total += (e.embeddings().elem_count() * e.embeddings().dtype().size_in_bytes()) as u64;
            }
        }
        total
    }

    /// Active-row embedding lookup for `PrejudiceCache`.
    pub fn embed_kept_rows_f32(&self, off: usize, take: usize, device: &Device) -> Result<Tensor> {
        if let Some(head) = &self.trim_head {
            Ok(head.kept_rows_f32(off, take, device)?)
        } else {
            Ok(self.language_model.embed_tokens.rows_f32(off, take)?)
        }
    }

    pub fn last_rope_deltas(&self) -> Option<Vec<i64>> {
        self.rope_deltas_cpu.clone()
    }

    pub fn capture_hybrid_snapshot(
        &self,
        tokens: &[u32],
        last_hidden: &Tensor,
        target_dev: &Device,
    ) -> Result<Option<HybridPromptSnapshot>> {
        if tokens.len() < MIN_PREFIX_SNAPSHOT_TOKENS {
            return Ok(None);
        }
        let mut layers = Vec::with_capacity(self.language_model.layers.len());
        for layer in &self.language_model.layers {
            match layer.capture_hybrid_state(target_dev)? {
                Some(st) => layers.push(st),
                None => return Ok(None),
            }
        }
        let hidden_clone = clone_tensor_to_device(&last_hidden.to_dtype(DType::F32)?, target_dev)?;
        Ok(Some(HybridPromptSnapshot {
            tokens: tokens.to_vec(),
            layers,
            last_hidden: hidden_clone,
            rope_deltas_cpu: self.rope_deltas_cpu.clone(),
            kv_exact: self.language_model.kv_exact,
            kv_half: self.language_model.mem.kv_half,
            on_vram: !target_dev.is_cpu(),
        }))
    }

    pub fn restore_hybrid_snapshot(
        &mut self,
        snap: &HybridPromptSnapshot,
        compute_dev: &Device,
    ) -> Result<Tensor> {
        if self.language_model.layers.len() != snap.layers.len() {
            return Err(anyhow!(
                "HybridPromptSnapshot layer count mismatch: {} vs {}",
                snap.layers.len(),
                self.language_model.layers.len()
            ));
        }
        self.language_model.clear_cache();
        for (layer, state) in self.language_model.layers.iter_mut().zip(snap.layers.iter()) {
            layer.restore_hybrid_state_cow(state, compute_dev)?;
        }
        self.language_model.current_kv_len = snap.tokens.len();
        self.language_model.kv_exact = snap.kv_exact;
        self.language_model.mem.kv_half = snap.kv_half;
        self.language_model.kv_plan = None;
        self.rope_deltas_cpu = snap.rope_deltas_cpu.clone();
        self.rope_deltas = snap.rope_deltas_cpu.as_ref().and_then(|deltas| {
            let bs = deltas.len();
            Tensor::from_vec(deltas.clone(), (bs, 1), compute_dev).ok()
        });
        clone_tensor_to_device(&snap.last_hidden.to_dtype(DType::F32)?, compute_dev)
    }

    pub async fn forward_tokens_segmented<F>(
        &mut self,
        all_tokens: &[u32],
        start_offset: usize,
        checkpoints: &[usize],
        device: &Device,
        mut on_checkpoint: F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Self, usize, &Tensor),
    {
        let total = all_tokens.len();
        if start_offset >= total {
            return Err(anyhow!(
                "forward_tokens_segmented: start_offset ({start_offset}) >= total ({total})"
            ));
        }
        self.validate_input_ids(&all_tokens[start_offset..])?;
        let mut bounds: Vec<usize> = checkpoints
            .iter()
            .copied()
            .filter(|&cp| cp > start_offset && cp <= total)
            .collect();
        bounds.sort_unstable();
        bounds.dedup();
        if bounds.last().copied() != Some(total) {
            bounds.push(total);
        }
        self.language_model.planned_prompt_len = total;
        let mut pos = start_offset;
        let mut final_hidden: Option<Tensor> = None;
        for end in bounds {
            let seg = &all_tokens[pos..end];
            let seg_ids = Tensor::from_vec(seg.to_vec(), (1, seg.len()), device)?;
            let hidden = self
                .forward_last_hidden(&seg_ids, None, None, None, None, None, pos, None, None)
                .await?;
            if checkpoints.contains(&end) {
                on_checkpoint(self, end, &hidden);
            }
            final_hidden = Some(hidden);
            pos = end;
        }
        final_hidden.ok_or_else(|| anyhow!("forward_tokens_segmented produced no hidden state"))
    }

    pub fn validate_input_ids(&self, ids: &[u32]) -> Result<()> {
        let vocab = self.language_model.embed_tokens.vocab()?;
        validate_host_token_ids(ids, vocab, "Qwen3_5Model::validate_input_ids")
    }

    pub fn clear_cache(&mut self) {
        self.language_model.clear_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::Activation;
    use crate::models::qwen3_5::config::RopeParameters;

    struct DetBackend;

    impl candle_nn::var_builder::SimpleBackend for DetBackend {
        fn get(
            &self,
            s: candle_core::Shape,
            name: &str,
            _h: candle_nn::Init,
            dtype: DType,
            dev: &Device,
        ) -> candle_core::Result<Tensor> {
            let n = s.elem_count();
            let mut hash = 0x9e37_79b9_7f4a_7c15u64;
            for b in name.as_bytes() {
                hash = hash.wrapping_mul(6364136223846793005).wrapping_add(*b as u64);
            }
            let is_norm = s.dims().len() == 1
                && (name.contains("norm") || name.contains("bias"))
                && !name.contains("A_log");
            let vals: Vec<f32> = (0..n)
                .map(|i| {
                    if is_norm {
                        0.92 + ((i % 7) as f32) * 0.02
                    } else if name.contains("A_log") {
                        -1.5 + ((i % 5) as f32) * 0.1
                    } else {
                        let x = ((hash.wrapping_add(i as u64 * 2654435761) >> 16) % 101) as f32;
                        (x - 50.0) * 0.0025
                    }
                })
                .collect();
            Tensor::from_vec(vals, s, dev)?.to_dtype(dtype)
        }

        fn get_unchecked(&self, _name: &str, _dtype: DType, _dev: &Device) -> candle_core::Result<Tensor> {
            Err(candle_core::Error::Msg("unused".into()))
        }

        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    fn tiny_test_text_config() -> Qwen3_5TextConfig {
        Qwen3_5TextConfig {
            attention_bias: false,
            attention_dropout: 0.0,
            attn_output_gate: true,
            dtype: "float32".to_string(),
            eos_token_id: 1,
            full_attention_interval: 2,
            head_dim: 16,
            hidden_act: Activation::Silu,
            hidden_size: 32,
            initializer_range: 0.02,
            intermediate_size: 64,
            layer_types: vec!["linear_attention".to_string(), "full_attention".to_string()],
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 16,
            linear_num_key_heads: 2,
            linear_num_value_heads: 2,
            linear_value_head_dim: 16,
            max_position_embeddings: 512,
            mlp_only_layers: Vec::new(),
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            num_attention_heads: 2,
            num_hidden_layers: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: Some(true),
            use_cache: true,
            vocab_size: 64,
            mamba_ssm_dtype: "float32".to_string(),
            rope_parameters: RopeParameters {
                mrope_interleaved: false,
                mrope_section: vec![2, 1, 1],
                rope_type: "default".to_string(),
                rope_theta: 10_000.0,
                partial_rotary_factor: 0.5,
            },
        }
    }

    fn build_tiny_loaded_model() -> Result<Qwen3_5Model> {
        let cfg = tiny_test_text_config();
        let vb = VarBuilder::from_backend(Box::new(DetBackend), DType::F32, Device::Cpu);
        let embed_tokens = EmbedTable::Dense(embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?);
        let registry = KVRegistry::new();
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3_5DecoderLayer::new_from_vb(vb.pp("layers").pp(i), &cfg, i, registry.clone())?);
        }
        let norm = Qwen3_5RMSNorm::new(vb.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let rope_dim = (cfg.head_dim as f32 * cfg.rope_parameters.partial_rotary_factor) as usize;
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(rope_dim, cfg.rope_parameters.rope_theta);
        let language_model = Qwen3_5TextModel {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            mrope_section: cfg.rope_parameters.mrope_section.clone(),
            dtype: DType::F32,
            registry,
            current_kv_len: 0,
            active_session_id: None,
            active_kv_name: None,
            kv_plan: None,
            keep_weights_resident: false,
            resident_prefix: 0,
            release_streamed_pages: false,
            mmap: None,
            ct: None,
            base_name: "test".to_string(),
            keep_warm: false,
            kv_exact: true,
            planned_prompt_len: 0,
            mem: Qwen35MemState::default(),
            persist_kv: false,
        };
        let lm_head = ProjKind::LinearProj(Linear::new(
            language_model.embed_tokens.dense().unwrap().clone(),
            None,
        ));
        Ok(Qwen3_5Model {
            spatial_merge_size: 2,
            image_token_id: 2,
            video_token_id: 3,
            vision_start_token_id: 4,
            vision_end_token_id: 5,
            visual: None,
            language_model,
            lm_head,
            trim_head: None,
            rope_deltas: None,
            rope_deltas_cpu: None,
        })
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        let diff = a
            .to_dtype(DType::F32)?
            .sub(&b.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(diff.into_iter().fold(0.0f32, f32::max))
    }

    #[test]
    fn gdn_suffix_conv_len_lt_kernel_matches_full_pass() -> Result<()> {
        let cfg = tiny_test_text_config();
        let vb = VarBuilder::from_backend(Box::new(DetBackend), DType::F32, Device::Cpu);
        let mut gdn_full = Qwen3_5GatedDeltaNet::new_from_vb(vb.pp("gdn"), &cfg)?;
        let mut gdn_split = Qwen3_5GatedDeltaNet::new_from_vb(vb.pp("gdn"), &cfg)?;

        let input_data: Vec<f32> = (0..(9 * cfg.hidden_size))
            .map(|i| ((i * 13 % 37) as f32 - 18.0) * 0.05)
            .collect();
        let all_x = Tensor::from_vec(input_data, (1, 9, cfg.hidden_size), &Device::Cpu)?;

        let prefix_x = all_x.narrow(1, 0, 8)?;
        let step_x = all_x.narrow(1, 8, 1)?;
        let _ = gdn_full.forward(&prefix_x, None)?;
        let full_step = gdn_full.forward(&step_x, None)?;
        let (full_conv, full_rec) = gdn_full.get_states();

        let p0 = all_x.narrow(1, 0, 6)?;
        let s1 = all_x.narrow(1, 6, 2)?;
        let _ = gdn_split.forward(&p0, None)?;
        let _ = gdn_split.forward(&s1, None)?;
        let split_step = gdn_split.forward(&step_x, None)?;
        let (split_conv, split_rec) = gdn_split.get_states();

        let conv_err = max_abs_diff(full_conv.as_ref().unwrap(), split_conv.as_ref().unwrap())?;
        let rec_err = max_abs_diff(full_rec.as_ref().unwrap(), split_rec.as_ref().unwrap())?;
        let step_err = max_abs_diff(&full_step, &split_step)?;
        assert!(conv_err < 1e-5, "conv state diverged: {conv_err}");
        assert!(rec_err < 1e-4, "recurrent state diverged: {rec_err}");
        assert!(step_err < 1e-4, "step output diverged: {step_err}");
        Ok(())
    }

    #[test]
    fn hybrid_snapshot_cache_exact_and_prefix_cow_parity() -> Result<()> {
        let rt = tokio::runtime::Builder::new_current_thread().build()?;
        let mut model = build_tiny_loaded_model()?;
        let tokens_96: Vec<u32> = (0..96u32).map(|i| (i * 7 + 10) % 64).collect();
        let next_tok = Tensor::from_vec(vec![19u32], (1, 1), &Device::Cpu)?;

        let ref_h96 = rt.block_on(model.forward_tokens_segmented(
            &tokens_96,
            0,
            &[],
            &Device::Cpu,
            |_, _, _| {},
        ))?;
        let ref_h97 = rt.block_on(model.forward_last_hidden(
            &next_tok,
            None,
            None,
            None,
            None,
            None,
            96,
            None,
            None,
        ))?;

        model.clear_cache();
        let mut snap_sys: Option<HybridPromptSnapshot> = None;
        let cap_h96 = rt.block_on(model.forward_tokens_segmented(
            &tokens_96,
            0,
            &[64],
            &Device::Cpu,
            |m, cp, h| {
                if cp == 64 {
                    snap_sys = m
                        .capture_hybrid_snapshot(&tokens_96[..64], h, &Device::Cpu)
                        .expect("capture sys");
                }
            },
        ))?;
        assert!(max_abs_diff(&ref_h96, &cap_h96)? < 1e-5);
        let snap_full = model
            .capture_hybrid_snapshot(&tokens_96, &cap_h96, &Device::Cpu)?
            .expect("capture full");

        let mut cache = HybridSnapshotCache::default();
        cache.store_system_host(snap_sys.expect("sys snap present"));
        cache.store_active(snap_full);

        let (exact_snap, matched_full) = cache.lookup_best(&tokens_96).expect("exact hit");
        assert_eq!(matched_full, 96);
        let exact_snap_clone = exact_snap.clone();
        let h_restored = model.restore_hybrid_snapshot(&exact_snap_clone, &Device::Cpu)?;
        assert!(max_abs_diff(&ref_h96, &h_restored)? < 1e-5);
        let h97_run1 = rt.block_on(model.forward_last_hidden(
            &next_tok,
            None,
            None,
            None,
            None,
            None,
            96,
            None,
            None,
        ))?;
        assert!(max_abs_diff(&ref_h97, &h97_run1)? < 1e-5);

        // CoW isolation check
        let h_restored_2 = model.restore_hybrid_snapshot(&exact_snap_clone, &Device::Cpu)?;
        assert!(max_abs_diff(&ref_h96, &h_restored_2)? < 1e-5);
        let h97_run2 = rt.block_on(model.forward_last_hidden(
            &next_tok,
            None,
            None,
            None,
            None,
            None,
            96,
            None,
            None,
        ))?;
        assert!(max_abs_diff(&ref_h97, &h97_run2)? < 1e-5);

        // Prefix hit from system_host_slot (64) with tiny suffix length 2 (< conv_kernel-1)
        let mut alt_tokens = tokens_96[..64].to_vec();
        alt_tokens.extend_from_slice(&[31u32, 42u32]);

        model.clear_cache();
        let ref_alt = rt.block_on(model.forward_tokens_segmented(
            &alt_tokens,
            0,
            &[],
            &Device::Cpu,
            |_, _, _| {},
        ))?;

        let (sys_hit, matched_sys) = cache.lookup_best(&alt_tokens).expect("sys hit");
        assert_eq!(matched_sys, 64);
        let sys_clone = sys_hit.clone();
        let _ = model.restore_hybrid_snapshot(&sys_clone, &Device::Cpu)?;
        let res_alt = rt.block_on(model.forward_tokens_segmented(
            &alt_tokens,
            matched_sys,
            &[],
            &Device::Cpu,
            |_, _, _| {},
        ))?;
        assert!(max_abs_diff(&ref_alt, &res_alt)? < 1e-4);
        Ok(())
    }
}
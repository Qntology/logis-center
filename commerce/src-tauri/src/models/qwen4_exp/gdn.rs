//! Gated DeltaNet (linear attention) — `Qwen4ExpTextGatedDeltaNet` (qwen4-spec §6).
//!
//! 핵심 연산은 [`gdn_forward`] 한 함수 경계 뒤에 있습니다. 입력/출력은 DESIGN (II) 의
//! `candle_core::delta_rule::fused_gated_delta_rule(mixed_qkv, conv_weight, conv_bias, conv_state, a, b, neg_a,
//! dt_bias, recurrent_state, &GdnConfig) -> GdnOutput` 계약과 똑같이 맞춰 두었으므로, 포크의 융합 커널이 들어오면
//! 이 함수 본문만 그 호출로 바꾸면 됩니다 (지금은 candle 텐서 연산 기반 재귀 공식 — 정확도 기준 구현).
//!
//! 수식 (HF 와 동일):
//!  - `qkv = silu(causal_dwconv([conv_state ; mixed_qkv]))` → q | k | v
//!  - `beta = sigmoid(b)`, `g = neg_a · softplus(a + dt_bias)` (F32, neg_a = -exp(A_log))
//!  - q, k 를 v-head 수에 맞춰 `repeat_interleave` (Grouped) → L2 norm(eps 1e-6, F32) → `q *= dk^-0.5`
//!  - `S = S·exp(g); S += k ⊗ beta·(v - Sᵀk); o = Sᵀq` (상태 `[B, Hv, Dk, Dv]` F32)

use anyhow::{ensure, Result};
use candle_core::{DType, Module, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use super::config::Qwen4ExpTextConfig;
use super::layers::{linear_no_bias, sigmoid, softplus, GatedRmsNorm};

/// v-head 와 k-head 의 대응 방식. Qwen3.5 / Qwen4-Exp 는 `Grouped` (torch `repeat_interleave`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnHeadOrder {
    /// v-head h 는 k-head `h / (Hv/Hk)` 를 씁니다.
    Grouped,
    /// v-head h 는 k-head `h % Hk` 를 씁니다.
    Tiled,
}

/// DESIGN (II) `candle_core::delta_rule::GdnConfig` 와 같은 필드.
#[derive(Debug, Clone, Copy)]
pub struct GdnConfig {
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel: usize,
    pub use_qk_l2norm: bool,
    pub l2norm_eps: f32,
    pub v_head_order: GdnHeadOrder,
}

/// DESIGN (II) `candle_core::delta_rule::GdnOutput` 와 같은 필드.
#[derive(Debug, Clone)]
pub struct GdnOutput {
    /// `[B, T, Hv, Dv]`, mixed_qkv 의 dtype
    pub out: Tensor,
    /// `[B, C, K-1]`, mixed_qkv 의 dtype (다음 호출의 conv_state)
    pub conv_state: Tensor,
    /// `[B, Hv, Dk, Dv]` F32
    pub recurrent_state: Tensor,
}

/// 융합 GDN 의 텐서 연산 기준 구현 (CPU/CUDA/ROCm/Vulkan 공통).
///
/// - `mixed_qkv`: `[B, T, C]` (C = 2·Hk·Dk + Hv·Dv), conv 이전 프로젝션
/// - `conv_weight`: `[C, K]` 또는 `[C, 1, K]`, `conv_bias`: `[C]`
/// - `conv_state`: `[B, C, K-1]` 직전 입력 (None = 0)
/// - `a`, `b`: `[B, T, Hv]` in_proj_a / in_proj_b 원출력
/// - `neg_a`: `[Hv]` = -exp(A_log), `dt_bias`: `[Hv]`
/// - `recurrent_state`: `[B, Hv, Dk, Dv]` F32 (None = 0)
#[allow(clippy::too_many_arguments)]
pub fn gdn_forward(
    mixed_qkv: &Tensor,
    conv_weight: &Tensor,
    conv_bias: Option<&Tensor>,
    conv_state: Option<&Tensor>,
    a: &Tensor,
    b: &Tensor,
    neg_a: &Tensor,
    dt_bias: &Tensor,
    recurrent_state: Option<&Tensor>,
    cfg: &GdnConfig,
) -> Result<GdnOutput> {
    let (bsz, t, c) = mixed_qkv.dims3()?;
    let (hk, hv, dk, dv, k) = (cfg.num_k_heads, cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim, cfg.conv_kernel);
    ensure!(c == 2 * hk * dk + hv * dv, "[QWEN4-EXP] GDN 채널 {c} != 2·Hk·Dk + Hv·Dv");
    ensure!(k >= 1 && hv % hk == 0, "[QWEN4-EXP] GDN 설정 오류 (K={k}, Hv={hv}, Hk={hk})");
    let in_dt = mixed_qkv.dtype();
    let dev = mixed_qkv.device();

    // ── 0. 🌟 [GDN-FUSED] 포크 융합 커널 (같은 계약, F32 연산). LOGIS_GDN_FUSED=0 이거나 커널이 없는 장치면 아래 텐서 경로 ──
    let fused_cfg = candle_core::delta_rule::GdnConfig {
        num_k_heads: hk,
        num_v_heads: hv,
        head_k_dim: dk,
        head_v_dim: dv,
        conv_kernel: k,
        use_qk_l2norm: cfg.use_qk_l2norm,
        l2norm_eps: cfg.l2norm_eps,
        v_head_order: match cfg.v_head_order {
            GdnHeadOrder::Grouped => candle_core::delta_rule::GdnHeadOrder::Grouped,
            GdnHeadOrder::Tiled => candle_core::delta_rule::GdnHeadOrder::Tiled,
        },
    };
    if crate::models::common::gdn_fused_enabled(dev, &fused_cfg) {
        let o = candle_core::delta_rule::fused_gated_delta_rule(
            mixed_qkv,
            conv_weight,
            conv_bias,
            conv_state,
            a,
            b,
            neg_a,
            dt_bias,
            recurrent_state,
            &fused_cfg,
        )?;
        return Ok(GdnOutput { out: o.out, conv_state: o.conv_state, recurrent_state: o.recurrent_state });
    }

    // ── 1. causal depthwise conv + SiLU (F32 연산) ──
    let w = conv_weight.to_dtype(DType::F32)?.reshape((c, k))?;
    let x = mixed_qkv.transpose(1, 2)?.contiguous()?; // [B, C, T]
    let prev = match conv_state {
        Some(s) => s.to_dtype(in_dt)?,
        None => Tensor::zeros((bsz, c, k.saturating_sub(1)), in_dt, dev)?,
    };
    let padded = if k > 1 { Tensor::cat(&[&prev, &x], 2)? } else { x.clone() }; // [B, C, T+K-1]
    let new_conv_state = if k > 1 {
        padded.narrow(2, t, k - 1)?.contiguous()?
    } else {
        Tensor::zeros((bsz, c, 0), in_dt, dev)?
    };
    let pf = padded.to_dtype(DType::F32)?;
    let mut conv = pf.narrow(2, 0, t)?.broadcast_mul(&w.narrow(1, 0, 1)?.reshape((1, c, 1))?)?;
    for j in 1..k {
        let tap = pf.narrow(2, j, t)?.broadcast_mul(&w.narrow(1, j, 1)?.reshape((1, c, 1))?)?;
        conv = (conv + tap)?;
    }
    if let Some(bias) = conv_bias {
        conv = conv.broadcast_add(&bias.to_dtype(DType::F32)?.reshape((1, c, 1))?)?;
    }
    // HF: conv 은 가중치 dtype 에서 계산 후 입력 dtype 으로 캐스팅. 여기서는 F32 로 계산 후 같은 dtype 으로 내립니다.
    let qkv = conv.silu()?.to_dtype(in_dt)?.transpose(1, 2)?; // [B, T, C]

    let q = qkv.narrow(2, 0, hk * dk)?.reshape((bsz, t, hk, dk))?;
    let kk = qkv.narrow(2, hk * dk, hk * dk)?.reshape((bsz, t, hk, dk))?;
    let v = qkv.narrow(2, 2 * hk * dk, hv * dv)?.reshape((bsz, t, hv, dv))?;

    // ── 2. 게이트 ──
    let beta = sigmoid(&b.to_dtype(DType::F32)?)?; // [B, T, Hv]
    let a_dt = a.to_dtype(DType::F32)?.broadcast_add(&dt_bias.to_dtype(DType::F32)?)?;
    let g = softplus(&a_dt)?.broadcast_mul(&neg_a.to_dtype(DType::F32)?)?; // [B, T, Hv]

    // ── 3. head 확장 + L2 norm (F32) ──
    let rep = hv / hk;
    let expand = |x: &Tensor| -> Result<Tensor> {
        if rep == 1 {
            return Ok(x.clone());
        }
        Ok(match cfg.v_head_order {
            GdnHeadOrder::Grouped => x
                .unsqueeze(3)?
                .broadcast_as((bsz, t, hk, rep, dk))?
                .reshape((bsz, t, hv, dk))?,
            GdnHeadOrder::Tiled => x
                .unsqueeze(2)?
                .broadcast_as((bsz, t, rep, hk, dk))?
                .reshape((bsz, t, hv, dk))?,
        })
    };
    let mut qf = expand(&q.to_dtype(DType::F32)?)?;
    let mut kf = expand(&kk.to_dtype(DType::F32)?)?;
    if cfg.use_qk_l2norm {
        let l2 = |x: &Tensor| -> Result<Tensor> {
            let inv = (x.sqr()?.sum_keepdim(D::Minus1)? + cfg.l2norm_eps as f64)?.sqrt()?.recip()?;
            Ok(x.broadcast_mul(&inv)?)
        };
        qf = l2(&qf)?;
        kf = l2(&kf)?;
    }
    let qf = (qf * (1.0 / (dk as f64).sqrt()))?;
    let vf = v.to_dtype(DType::F32)?;

    // [B, Hv, T, *] 로 바꿔 시간 축을 순회합니다.
    let qf = qf.transpose(1, 2)?.contiguous()?; // [B, Hv, T, Dk]
    let kf = kf.transpose(1, 2)?.contiguous()?;
    let vf = vf.transpose(1, 2)?.contiguous()?; // [B, Hv, T, Dv]
    let beta = beta.transpose(1, 2)?.contiguous()?; // [B, Hv, T]
    let decay = g.transpose(1, 2)?.contiguous()?.exp()?; // [B, Hv, T]

    // ── 4. 재귀 delta rule ──
    let mut state = match recurrent_state {
        Some(s) => s.to_dtype(DType::F32)?,
        None => Tensor::zeros((bsz, hv, dk, dv), DType::F32, dev)?,
    };
    let mut outs = Vec::with_capacity(t);
    for i in 0..t {
        let q_t = qf.narrow(2, i, 1)?; // [B, Hv, 1, Dk]
        let k_t = kf.narrow(2, i, 1)?;
        let v_t = vf.narrow(2, i, 1)?; // [B, Hv, 1, Dv]
        let b_t = beta.narrow(2, i, 1)?.unsqueeze(3)?; // [B, Hv, 1, 1]
        let d_t = decay.narrow(2, i, 1)?.unsqueeze(3)?; // [B, Hv, 1, 1]
        state = state.broadcast_mul(&d_t)?;
        let kv_mem = k_t.matmul(&state)?; // [B, Hv, 1, Dv]
        let delta = (v_t - kv_mem)?.broadcast_mul(&b_t)?;
        state = (state + k_t.transpose(2, 3)?.matmul(&delta)?)?; // k ⊗ delta
        outs.push(q_t.matmul(&state)?); // [B, Hv, 1, Dv]
    }
    let out = if t == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 2)? }; // [B, Hv, T, Dv]
    let out = out.transpose(1, 2)?.contiguous()?.to_dtype(in_dt)?; // [B, T, Hv, Dv]

    Ok(GdnOutput { out, conv_state: new_conv_state, recurrent_state: state })
}

/// 한 linear-attention 층 (`linear_attn.*`).
pub struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    conv_weight: Tensor, // [C, K]
    neg_a: Tensor,       // [Hv] F32
    dt_bias: Tensor,     // [Hv]
    norm: GatedRmsNorm,
    out_proj: Linear,
    cfg: GdnConfig,
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
}

impl GatedDeltaNet {
    pub fn new(vb: VarBuilder, c: &Qwen4ExpTextConfig) -> Result<Self> {
        let h = c.hidden_size;
        let (hk, hv, dk, dv, k) = (
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
            c.linear_conv_kernel_dim,
        );
        let conv_dim = 2 * hk * dk + hv * dv;
        let in_proj_qkv = linear_no_bias(vb.clone(), h, conv_dim, "in_proj_qkv")?;
        let in_proj_z = linear_no_bias(vb.clone(), h, hv * dv, "in_proj_z")?;
        let in_proj_b = linear_no_bias(vb.clone(), h, hv, "in_proj_b")?;
        let in_proj_a = linear_no_bias(vb.clone(), h, hv, "in_proj_a")?;
        let conv_weight = vb.get((conv_dim, 1, k), "conv1d.weight")?.reshape((conv_dim, k))?;
        // HF: `-self.A_log.float().exp()` — F32 로 읽어야 F16 에서 -inf 가 되지 않습니다.
        let a_log = vb.get(hv, "A_log")?.to_dtype(DType::F32)?;
        let neg_a = a_log.exp()?.neg()?;
        let dt_bias = vb.get(hv, "dt_bias")?;
        let norm = GatedRmsNorm::new(vb.pp("norm"), dv, c.rms_norm_eps, c.gate_act()?)?;
        let out_proj = linear_no_bias(vb.clone(), hv * dv, h, "out_proj")?;
        let cfg = GdnConfig {
            num_k_heads: hk,
            num_v_heads: hv,
            head_k_dim: dk,
            head_v_dim: dv,
            conv_kernel: k,
            use_qk_l2norm: true,
            l2norm_eps: 1e-6,
            v_head_order: GdnHeadOrder::Grouped,
        };
        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv_weight,
            neg_a,
            dt_bias,
            norm,
            out_proj,
            cfg,
            conv_state: None,
            recurrent_state: None,
        })
    }

    /// `m: [B, T, H]` → `[B, T, H]`. conv/recurrent 상태를 이어 받아 갱신합니다 (프리필 청크·디코드 공통).
    pub fn forward(&mut self, m: &Tensor) -> Result<Tensor> {
        let (bsz, t, _) = m.dims3()?;
        let mixed = self.in_proj_qkv.forward(m)?;
        let z = self.in_proj_z.forward(m)?;
        let b = self.in_proj_b.forward(m)?;
        let a = self.in_proj_a.forward(m)?;
        let o = gdn_forward(
            &mixed,
            &self.conv_weight,
            None,
            self.conv_state.as_ref(),
            &a,
            &b,
            &self.neg_a,
            &self.dt_bias,
            self.recurrent_state.as_ref(),
            &self.cfg,
        )?;
        self.conv_state = Some(o.conv_state);
        self.recurrent_state = Some(o.recurrent_state);
        let dv = self.cfg.head_v_dim;
        let core = o.out.reshape(((), dv))?;
        let z = z.reshape(((), dv))?;
        let y = self.norm.forward(&core, &z)?.reshape((bsz, t, self.cfg.num_v_heads * dv))?;
        Ok(self.out_proj.forward(&y)?)
    }

    pub fn clear_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }

    pub fn states(&self) -> (Option<&Tensor>, Option<&Tensor>) {
        (self.conv_state.as_ref(), self.recurrent_state.as_ref())
    }
}

use anyhow::Result;
use candle_core::{D, IndexOp, Tensor};
use crate::utils::GpuDeviceExt;
use candle_nn::{
    Activation, BatchNorm, BatchNormConfig, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig,
    ConvTranspose1d, ConvTranspose1dConfig, Embedding, LayerNorm, LayerNormConfig, Linear, Module,
    ModuleT, RmsNorm, VarBuilder, batch_norm, conv1d, conv1d_no_bias, conv2d, conv2d_no_bias,
    embedding, layer_norm, linear_b, linear_no_bias, ops::sigmoid, rms_norm,
};

pub mod gguf;
// 🌟 [TURBO-KV-BLOCK] 블록 단위 KV(Qwen3.5)용 TurboQuant 코덱과 블록 어텐션 누산기, KV 압축 형식 정책
pub mod kv_turbo;
// 🌟 [ACCEL-CHECK] GPU 가속 커널(FlashAttention / TurboQuant) 자가 검증과 실측 정확도 감시
pub mod accel_check;

use crate::{
    position_embed::rope::{RoPE, apply_rotary_pos_emb, apply_rotary_pos_emb_roformer},
    utils::tensor_utils::{prepare_causal_attention_mask, repeat_kv},
};

#[cfg(feature = "turbo-quant")]
use candle_nn::{
    flash_attn::{FlashAttnConfig, flash_attn_bhsd},
    turbo_quant::TurboKvCache,
};

/// 🌟 [TURBO-KV] TurboKvCache 압축 버퍼가 한 번에 늘어나는 토큰 수입니다.
#[cfg(feature = "turbo-quant")]
pub const TURBO_KV_GROW_BY: usize = 1024;

/// 🌟 [TURBO-KV] KV 캐시를 TurboQuant 로 압축 보관할 수 있는 빌드·장치·설정인지.
/// `turbo-quant` feature 로 빌드되었고, GPU(CUDA / ROCm / Vulkan) 장치이며,
/// `CANDLE_TURBO_QUANT=off` 로 끄지 않았을 때만 true 입니다.
/// false 이면 각 모델은 기존 KV 경로(FP8 압축, RAM 대피 등)를 그대로 사용합니다.
pub fn turbo_kv_active(device: &candle_core::Device) -> bool {
    #[cfg(feature = "turbo-quant")]
    {
        return matches!(
            device,
            candle_core::Device::Cuda(_) | candle_core::Device::Rocm(_) | candle_core::Device::Vulkan(_)
        ) && candle_core::accel::config().turbo_quant.enabled;
    }
    #[cfg(not(feature = "turbo-quant"))]
    {
        let _ = device;
        false
    }
}

/// 🌟 [TURBO-KV] [`turbo_kv_active`] 이면서, 이 장치·head_dim 에서 GPU 자가 검증([`accel_check`])을
/// 통과했고 실측 감시가 TurboQuant 를 끄지 않았는지. 모델들은 이 함수로 압축 캐시 사용 여부를 정합니다.
pub fn turbo_kv_ready(device: &candle_core::Device, head_dim: usize) -> bool {
    turbo_kv_active(device) && accel_check::turbo_quant_ok(device, head_dim)
}

/// 🌟 [TURBO-KV] 이번 forward 에 압축 캐시 경로를 쓸지 (Qwen3 / QwenVL).
/// - 기본 FP8 형식: 첫 프리필 청크부터 씁니다. (과거 이력은 압축본, 현재 청크는 원본으로 계산)
/// - 4bit 등 저비트 형식: 프리필은 기존 경로로 계산하고, 디코딩(q_len == 1)부터 이력을 한 번에
///   압축 캐시로 옮깁니다. 프롬프트 표현이 저비트 오차를 레이어마다 누적하지 않도록 하기 위해서입니다.
/// - 이미 압축 캐시에 이력이 있으면 계속 씁니다.
pub fn turbo_kv_use(
    device: &candle_core::Device,
    head_dim: usize,
    q_len: usize,
    has_turbo_history: bool,
) -> bool {
    turbo_kv_ready(device, head_dim)
        && (q_len == 1 || has_turbo_history || !kv_turbo::kv_is_compact(device))
}

/// 🌟 [TURBO-KV] [`turbo_kv_step`] 결과
#[cfg(feature = "turbo-quant")]
pub struct TurboStep {
    /// 어텐션 출력 (batch, q_heads, q_len, head_dim)
    pub out: Tensor,
    /// 실측 감시가 압축 캐시를 버린 경우, 호출부가 평문 캐시로 이어 쓸 원본 전체 K/V
    pub plain_kv: Option<(Tensor, Tensor)>,
}

/// 🌟 [TURBO-KV] 한 스텝 분량의 K/V 를 TurboQuant 압축 캐시에 붙이고 어텐션을 계산합니다.
///
/// - `q`: (batch, q_heads, q_len, head_dim), `k`/`v`: (batch, kv_heads, q_len, head_dim)
/// - 출력: (batch, q_heads, q_len, head_dim)
/// - 마스크는 causal(bottom-right 정렬)로 고정됩니다. 청크 프리필(seqlen_offset > 0)도 그대로 맞습니다.
/// - 원본 전체 K/V 가 손에 있을 때(빈 캐시의 첫 채움, 또는 `prev` 평문 이력 흡수)는 원본으로
///   정확하게 계산합니다. `prev` 는 스냅샷 복원본·RAM 대피본·4bit 형식의 프리필 결과처럼 그것이
///   전체 이력인 평문 캐시이므로, 압축 캐시를 비우고 먼저 흡수한 뒤 새 토큰을 붙입니다.
/// - 이력이 압축 캐시에만 있는 청크 프리필(q_len > 1)은 압축된 이력을 풀어 원본 현재 청크와 이어 붙이고
///   타일 flash-attn 커널로 계산합니다. (기존 "FP8 이력 + 원본 현재 청크" 경로와 같은 구성)
/// - 디코딩(q_len == 1)은 압축 캐시 위에서 바로(fused) 계산합니다.
/// - [`accel_check`] 실측 감시: 원본으로 계산하는 스텝에서 같은 질의를 압축 캐시로도 계산해
///   오차를 기록하고, 비정상이면 압축 캐시를 버리고 원본 K/V 를 `plain_kv` 로 돌려줍니다.
#[cfg(feature = "turbo-quant")]
pub fn turbo_kv_step(
    cache: &mut Option<TurboKvCache>,
    prev: Option<(Tensor, Tensor)>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    scaling: f64,
) -> Result<TurboStep> {
    let dev = k.device().clone();
    if cache.is_none() {
        *cache = Some(TurboKvCache::with_options(
            &kv_turbo::kv_quant_options(),
            head_dim,
            TURBO_KV_GROW_BY,
            &dev,
        )?);
    }
    let Some(c) = cache.as_mut() else {
        anyhow::bail!("turbo-kv: 캐시 초기화 실패");
    };
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let q_len = q.dim(2)?;
    let cfg = FlashAttnConfig::new(scaling as f32).causal();
    // 🌟 [ACCEL-CHECK] 아래에서 flash-attn 을 쓰므로 이 dtype·head_dim 의 자가 검증을 먼저 끝냅니다.
    //    (실패하면 이 백엔드의 fused flash-attn 이 꺼지고 candle 의 비융합 경로로 계산됩니다)
    let _ = accel_check::flash_attn_ok(&dev, q.dtype(), head_dim);

    // 원본 전체 K/V 가 손에 있는 경우: 평문 이력 흡수(prev) 또는 빈 캐시의 첫 채움
    let exact_full = match prev {
        Some((pk, pv)) => {
            c.reset();
            let pk = pk.to_device(&dev)?.to_dtype(k.dtype())?.contiguous()?;
            let pv = pv.to_device(&dev)?.to_dtype(v.dtype())?.contiguous()?;
            c.append(&pk, &pv)?;
            c.append(&k, &v)?;
            Some((Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?))
        }
        None if c.seq_len() == 0 => {
            c.append(&k, &v)?;
            Some((k.clone(), v.clone()))
        }
        None => None,
    };

    if let Some((full_k, full_v)) = exact_full {
        let out = flash_attn_bhsd(&q, &full_k, &full_v, &cfg)?;
        // 🌟 [ACCEL-CHECK] 실측 감시: 같은 질의를 압축 캐시로도 계산해 실제 오차를 기록합니다.
        if accel_check::probe_wanted(&dev) {
            let label = match c.formats() {
                Some((kf, vf)) => format!("K {kf} / V {vf}"),
                None => "plain".to_string(),
            };
            let packed = c.attention(&q, &cfg)?;
            let rel = accel_check::rel_err(&packed, &out)?;
            if !accel_check::record_probe(&dev, rel, &label) {
                *cache = None;
                return Ok(TurboStep { out, plain_kv: Some((full_k, full_v)) });
            }
        }
        return Ok(TurboStep { out, plain_kv: None });
    }

    if q_len > 1 {
        // 청크 프리필: 과거 이력은 압축본을 풀어서, 이번 청크는 원본 그대로 계산합니다.
        let hist = c.keys_values()?;
        c.append(&k, &v)?;
        let out = match hist {
            Some((hk, hv)) => {
                let full_k = Tensor::cat(&[&hk.to_dtype(k.dtype())?, &k], 2)?;
                let full_v = Tensor::cat(&[&hv.to_dtype(v.dtype())?, &v], 2)?;
                flash_attn_bhsd(&q, &full_k, &full_v, &cfg)?
            }
            None => flash_attn_bhsd(&q, &k, &v, &cfg)?,
        };
        return Ok(TurboStep { out, plain_kv: None });
    }

    // 디코딩: 압축 캐시 위에서 바로(fused) 계산합니다.
    c.append(&k, &v)?;
    let out = c.attention(&q, &cfg)?;
    Ok(TurboStep { out, plain_kv: None })
}

#[derive(Debug, Clone)]
pub struct GateUpDownMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl GateUpDownMLP {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        act_fn: Activation,
        bias: bool,
        gate_pp_name: Option<&str>,
        up_pp_name: Option<&str>,
        down_pp_name: Option<&str>,
    ) -> Result<Self> {
        let gate_pp_name = gate_pp_name.unwrap_or("gate_proj");
        let up_pp_name = up_pp_name.unwrap_or("up_proj");
        let down_pp_name = down_pp_name.unwrap_or("down_proj");
        let gate_proj = linear_b(hidden_size, intermediate_size, bias, vb.pp(gate_pp_name))?;
        let up_proj = linear_b(hidden_size, intermediate_size, bias, vb.pp(up_pp_name))?;
        let down_proj = linear_b(intermediate_size, hidden_size, bias, vb.pp(down_pp_name))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }
}

impl Module for GateUpDownMLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

pub struct TwoLinearMLP {
    linear1: Linear,
    linear2: Linear,
    act: Activation,
}

impl TwoLinearMLP {
    pub fn new(
        vb: VarBuilder,
        // embedding_dim: usize,
        // mlp_dim: usize,
        in_dim: usize,
        middle_dim: usize,
        out_dim: usize,
        act: Activation,
        bias: bool,
        linear1_pp_name: &str,
        linear2_pp_name: &str,
    ) -> Result<Self> {
        let linear1 = linear_b(in_dim, middle_dim, bias, vb.pp(linear1_pp_name))?;
        let linear2 = linear_b(middle_dim, out_dim, bias, vb.pp(linear2_pp_name))?;

        Ok(Self {
            linear1,
            linear2,
            act,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs
            .apply(&self.linear1)?
            .apply(&self.act)?
            .apply(&self.linear2)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct NaiveAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    middle_size: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl NaiveAttention {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: Option<usize>,
        bias: bool,
        q_proj_pp_name: Option<&str>,
        k_proj_pp_name: Option<&str>,
        v_proj_pp_name: Option<&str>,
        o_proj_pp_name: Option<&str>,
    ) -> Result<Self> {
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let head_dim = match head_dim {
            None => hidden_size / num_attention_heads,
            Some(dim) => dim,
        };
        let q_proj_pp_name = q_proj_pp_name.unwrap_or("q_proj");
        let k_proj_pp_name = k_proj_pp_name.unwrap_or("k_proj");
        let v_proj_pp_name = v_proj_pp_name.unwrap_or("v_proj");
        let o_proj_pp_name = o_proj_pp_name.unwrap_or("o_proj");
        let q_proj = linear_b(
            hidden_size,
            num_attention_heads * head_dim,
            bias,
            vb.pp(q_proj_pp_name),
        )?;
        let k_proj = linear_b(
            hidden_size,
            num_key_value_heads * head_dim,
            bias,
            vb.pp(k_proj_pp_name),
        )?;
        let v_proj = linear_b(
            hidden_size,
            num_key_value_heads * head_dim,
            bias,
            vb.pp(v_proj_pp_name),
        )?;
        let o_proj = linear_b(
            num_attention_heads * head_dim,
            hidden_size,
            bias,
            vb.pp(o_proj_pp_name),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: num_attention_heads,
            num_kv_heads: num_key_value_heads,
            num_kv_groups,
            head_dim,
            middle_size: num_attention_heads * head_dim,
            kv_cache: None,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        tof32: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) = if let Some(cos) = cos {
            if let Some(sin) = sin {
                apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?
            } else {
                (query_states, key_states)
            }
        } else {
            (query_states, key_states)
        };

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.middle_size))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn forward_with_cache(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        tof32: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?;
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };

        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.middle_size))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
pub struct QKVCatAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    middle_size: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl QKVCatAttention {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        head_dim: Option<usize>,
        bias: bool,
        qkv_proj_pp_name: Option<&str>,
        o_proj_pp_name: Option<&str>,
    ) -> Result<Self> {
        let head_dim = match head_dim {
            None => hidden_size / num_attention_heads,
            Some(dim) => dim,
        };
        let qkv_proj_pp_name = qkv_proj_pp_name.unwrap_or("wqkv");
        let o_proj_pp_name = o_proj_pp_name.unwrap_or("o_proj");
        let qkv_proj = linear_b(
            hidden_size,
            3 * num_attention_heads * head_dim,
            bias,
            vb.pp(qkv_proj_pp_name),
        )?;
        let o_proj = linear_b(
            num_attention_heads * head_dim,
            hidden_size,
            bias,
            vb.pp(o_proj_pp_name),
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads: num_attention_heads,
            head_dim,
            middle_size: num_attention_heads * head_dim,
            kv_cache: None,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        tof32: bool,
        use_roformer: bool,
    ) -> Result<Tensor> {
        let (b, q_len, _) = xs.dims3()?;
        // (3, B, n_head, seq_len, head_dim)
        let qkv = self
            .qkv_proj
            .forward(xs)?
            .reshape((b, q_len, 3, self.num_heads, ()))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let query_states = qkv.i(0)?.contiguous()?;
        let key_states = qkv.i(1)?.contiguous()?;
        let value_states = qkv.i(2)?.contiguous()?;
        let (query_states, key_states) = if let Some(cos) = cos {
            if let Some(sin) = sin {
                if use_roformer {
                    apply_rotary_pos_emb_roformer(&query_states, &key_states, cos, sin, tof32)?
                } else {
                    apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?
                }
            } else {
                (query_states, key_states)
            }
        } else {
            (query_states, key_states)
        };

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            None,
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b, q_len, self.middle_size))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn forward_with_cache(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        tof32: bool,
        use_roformer: bool,
    ) -> Result<Tensor> {
        let (b, q_len, _) = xs.dims3()?;
        let qkv = self
            .qkv_proj
            .forward(xs)?
            .reshape((b, q_len, 3, self.num_heads, ()))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let query_states = qkv.i(0)?.contiguous()?;
        let key_states = qkv.i(1)?.contiguous()?;
        let value_states = qkv.i(2)?.contiguous()?;
        let (query_states, key_states) = if use_roformer {
            apply_rotary_pos_emb_roformer(&query_states, &key_states, cos, sin, tof32)?
        } else {
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?
        };
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };

        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            None,
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b, q_len, self.middle_size))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

pub struct QKNormAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    pub kv_cache: Option<(Tensor, Tensor)>,
    // 🌟 [TURBO-KV] TurboQuant 로 압축된 KV 캐시. 이 경로가 켜져 있으면 kv_cache(평문)는 비어 있습니다.
    #[cfg(feature = "turbo-quant")]
    turbo_kv: Option<TurboKvCache>,
    // 🌟 [TURBO-KV] 레이어가 TurboQuant 경로를 써도 되는지. KV 를 RAM 으로 대피시키는 동안은 false.
    turbo_kv_allowed: bool,
}

impl QKNormAttention {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        head_dim: Option<usize>,
        num_key_value_heads: Option<usize>,
        attention_bias: bool,
        rms_norm_eps: f64,
        q_proj_pp_name: Option<&str>,
        k_proj_pp_name: Option<&str>,
        v_proj_pp_name: Option<&str>,
        o_proj_pp_name: Option<&str>,
        q_norm_pp_name: Option<&str>,
        k_norm_pp_name: Option<&str>,
    ) -> Result<Self> {
        let head_dim = head_dim.unwrap_or(hidden_size / num_attention_heads);
        let num_key_value_heads = num_key_value_heads.unwrap_or(num_attention_heads);
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);
        let q_proj_pp_name = q_proj_pp_name.unwrap_or("q_proj");
        let k_proj_pp_name = k_proj_pp_name.unwrap_or("k_proj");
        let v_proj_pp_name = v_proj_pp_name.unwrap_or("v_proj");
        let o_proj_pp_name = o_proj_pp_name.unwrap_or("o_proj");
        let q_norm_pp_name = q_norm_pp_name.unwrap_or("q_norm");
        let k_norm_pp_name = k_norm_pp_name.unwrap_or("k_norm");
        let q_proj = linear_b(
            hidden_size,
            num_attention_heads * head_dim,
            attention_bias,
            vb.pp(q_proj_pp_name),
        )?;
        let k_proj = linear_b(
            hidden_size,
            num_key_value_heads * head_dim,
            attention_bias,
            vb.pp(k_proj_pp_name),
        )?;
        let v_proj = linear_b(
            hidden_size,
            num_key_value_heads * head_dim,
            attention_bias,
            vb.pp(v_proj_pp_name),
        )?;
        let o_proj = linear_b(
            num_attention_heads * head_dim,
            hidden_size,
            attention_bias,
            vb.pp(o_proj_pp_name),
        )?;
        let q_norm = rms_norm(head_dim, rms_norm_eps, vb.pp(q_norm_pp_name))?;
        let k_norm = rms_norm(head_dim, rms_norm_eps, vb.pp(k_norm_pp_name))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            num_kv_groups,
            head_dim,
            scaling,
            kv_cache: None,
            #[cfg(feature = "turbo-quant")]
            turbo_kv: None,
            turbo_kv_allowed: true,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_attention_heads,
            self.head_dim,
        ))?;
        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?;
        let key_states = self.k_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;
        let value_states = self.v_proj.forward(xs)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;

        // 🌟 [TURBO-KV] GPU + turbo-quant 빌드에서는 새 K/V 를 TurboQuant 압축 캐시에 붙이고
        //    압축된 캐시로 어텐션을 계산합니다. (FP8 재압축/전체 복원 왕복이 사라집니다)
        //    attention_mask 는 Qwen3 의 표준 causal 마스크이므로 causal 설정으로 대체됩니다.
        //    언제 압축 캐시를 쓰는지는 turbo_kv_use (형식 정책·GPU 자가 검증·실측 감시) 가 정합니다.
        #[cfg(feature = "turbo-quant")]
        {
            let dev = query_states.device().clone();
            if self.turbo_kv_allowed
                && turbo_kv_use(&dev, self.head_dim, q_len, self.has_turbo_kv())
            {
                let prev = self.kv_cache.take();
                let step = turbo_kv_step(
                    &mut self.turbo_kv,
                    prev,
                    &query_states,
                    &key_states,
                    &value_states,
                    self.head_dim,
                    self.scaling,
                )?;
                if let Some(kv) = step.plain_kv {
                    // 실측 감시가 압축 캐시를 버렸습니다. 원본 K/V 로 기존 경로를 이어 갑니다.
                    self.kv_cache = Some(kv);
                }
                let attn_output = step
                    .out
                    .transpose(1, 2)?
                    .contiguous()?
                    .reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
                return Ok(attn_output.apply(&self.o_proj)?);
            }
            // 압축 캐시를 더 쓰지 않는데 이력이 남아 있으면(자가 검증·실측 감시가 TurboQuant 를 끈 경우 등)
            // 평문 캐시로 풀어 기존 경로로 이어 갑니다. 풀지 못하면 문맥 없이 계속하지 않도록 에러를 냅니다.
            if self.kv_cache.is_none() && self.has_turbo_kv() {
                self.kv_cache = self.take_turbo_kv()?;
            }
        }

        // 🌟 [CRITICAL FIX] `&self.kv_cache`로 참조(Borrow)하면 이전 캐시와 새 캐시가 VRAM에 동시에 존재하여 2배 폭발이 일어납니다!
        // `take()`로 소유권을 완전히 빼앗아 연산 직후 이전 캐시가 VRAM에서 즉시 소멸되도록 수정합니다.
        let (key_states, value_states) = match self.kv_cache.take() {
            None => (key_states.contiguous()?, value_states.contiguous()?),
            Some((prev_k, prev_v)) => {
                let dev = key_states.device();
                // CPU RAM에 대피해 있던 이전 캐시를 잠시 VRAM으로 가져와 병합 연산을 수행합니다.
                let prev_k = if !prev_k.device().same_device(dev) { prev_k.to_device(dev)? } else { prev_k };
                let prev_v = if !prev_v.device().same_device(dev) { prev_v.to_device(dev)? } else { prev_v };
                let k = Tensor::cat(&[&prev_k, &key_states], 2)?.contiguous()?;
                let v = Tensor::cat(&[&prev_v, &value_states], 2)?.contiguous()?;
                (k, v)
            }
        };
        
        // 🌟 [JIT CPU Offload 방어] 여기서 즉시 CPU로 내리면 이후 FP8 압축이 CPU 연산으로 돌아가 엄청난 병목이 생깁니다.
        // VRAM에 잠시 남겨두고, 레이어(Layer) 레벨에서 즉각 FP8 압축 후 대피시키도록 위임합니다.
        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            self.scaling,
        )?;
        let attn_output =
            attn_output.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
        self.reset_turbo_kv();
    }

    /// 🌟 [TURBO-KV] TurboQuant 경로 허용 여부를 바꿉니다. (KV 를 RAM/SSD 로 대피시킬 때 false)
    pub fn set_turbo_kv_allowed(&mut self, allowed: bool) {
        self.turbo_kv_allowed = allowed;
    }

    /// 🌟 [TURBO-KV] 압축 캐시를 비웁니다. (평문 kv_cache 는 건드리지 않습니다)
    pub fn reset_turbo_kv(&mut self) {
        #[cfg(feature = "turbo-quant")]
        {
            self.turbo_kv = None;
        }
    }

    /// 🌟 [TURBO-KV] 압축 캐시에 토큰이 들어 있는지.
    pub fn has_turbo_kv(&self) -> bool {
        #[cfg(feature = "turbo-quant")]
        {
            return self.turbo_kv.as_ref().map_or(false, |c| c.seq_len() > 0);
        }
        #[cfg(not(feature = "turbo-quant"))]
        {
            false
        }
    }

    /// 🌟 [TURBO-KV] 압축 캐시를 원래 dtype 의 (k, v) 로 풀어 돌려줍니다. 캐시는 그대로 유지됩니다. (스냅샷 저장용)
    /// 형태: (batch, kv_heads, seq, head_dim)
    pub fn turbo_kv_snapshot(&self) -> Result<Option<(Tensor, Tensor)>> {
        #[cfg(feature = "turbo-quant")]
        {
            if let Some(c) = &self.turbo_kv {
                if c.seq_len() > 0 {
                    return Ok(c.keys_values()?);
                }
            }
        }
        Ok(None)
    }

    /// 🌟 [TURBO-KV] 압축 캐시를 풀어서 꺼내고 비웁니다. (VRAM → RAM 대피 전환용)
    /// 압축 해제에 실패하면 캐시를 그대로 둔 채 에러를 돌려줍니다.
    pub fn take_turbo_kv(&mut self) -> Result<Option<(Tensor, Tensor)>> {
        let kv = self.turbo_kv_snapshot()?;
        self.reset_turbo_kv();
        Ok(kv)
    }
}

pub struct NaiveAttnTwoLinearMLPBlock {
    self_attn: NaiveAttention,
    mlp: TwoLinearMLP,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl NaiveAttnTwoLinearMLPBlock {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: Option<usize>,
        head_dim: Option<usize>,
        attn_bias: bool,
        attn_pp_name: &str,
        o_proj_pp_name: Option<&str>,
        intermediate_size: usize,
        hidden_act: Activation,
        mlp_bias: bool,
        mlp_pp_name: &str,
        linear1_pp_name: &str,
        linear2_pp_name: &str,
        norm_eps: f64,
        input_norm_pp_name: &str,
        post_norm_pp_name: &str,
    ) -> Result<Self> {
        let num_key_value_heads = match num_key_value_heads {
            Some(heads) => heads,
            None => num_attention_heads,
        };
        let self_attn = NaiveAttention::new(
            vb.pp(attn_pp_name),
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            attn_bias,
            None,
            None,
            None,
            o_proj_pp_name,
        )?;
        let mlp = TwoLinearMLP::new(
            vb.pp(mlp_pp_name),
            hidden_size,
            intermediate_size,
            hidden_size,
            hidden_act,
            mlp_bias,
            linear1_pp_name,
            linear2_pp_name,
        )?;

        let input_layernorm =
            get_layer_norm(vb.pp(input_norm_pp_name), norm_eps, hidden_size, true)?;
        let post_attention_layernorm =
            get_layer_norm(vb.pp(post_norm_pp_name), norm_eps, hidden_size, true)?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        tof32: bool,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, cos, sin, attention_mask, tof32)?;
        let residual = residual.add(&xs)?;
        let xs = self.post_attention_layernorm.forward(&residual)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }
}

pub struct NaiveAttnGateUpDownMLPBlock {
    self_attn: NaiveAttention,
    mlp: GateUpDownMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl NaiveAttnGateUpDownMLPBlock {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: Option<usize>,
        head_dim: Option<usize>,
        attn_bias: bool,
        attn_pp_name: &str,
        o_proj_pp_name: Option<&str>,
        intermediate_size: usize,
        hidden_act: Activation,
        mlp_bias: bool,
        mlp_pp_name: &str,
        norm_eps: f64,
        input_norm_pp_name: &str,
        post_norm_pp_name: &str,
    ) -> Result<Self> {
        let num_key_value_heads = match num_key_value_heads {
            Some(heads) => heads,
            None => num_attention_heads,
        };
        let self_attn = NaiveAttention::new(
            vb.pp(attn_pp_name),
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            attn_bias,
            None,
            None,
            None,
            o_proj_pp_name,
        )?;
        let mlp = GateUpDownMLP::new(
            vb.pp(mlp_pp_name),
            hidden_size,
            intermediate_size,
            hidden_act,
            mlp_bias,
            None,
            None,
            None,
        )?;
        let input_layernorm = rms_norm(hidden_size, norm_eps, vb.pp(input_norm_pp_name))?;
        let post_attention_layernorm = rms_norm(hidden_size, norm_eps, vb.pp(post_norm_pp_name))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_with_cache(&xs, cos, sin, attention_mask, false)?;
        let residual = residual.add(&xs)?;
        let xs = self.post_attention_layernorm.forward(&residual)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }
    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

/// 🌟 [VISION-TILE] eager_attention_forward 의 블록 분할은 KV 축만 자릅니다.
/// 쿼리 축은 seq 전체로 남아 있어 전이 버퍼가 (q_len × min(kv_len, 4096) × heads) 로 커집니다.
/// ViT 는 이미지 1장이 곧 chunk 1개라 q_len == kv_len == N 이 되어
/// N=5720 기준 attn_folded 750MB + broadcast_sub 750MB + exp 750MB = 약 2.2GB 가
/// 동시에 살아 있게 됩니다.
///
/// 소프트맥스는 KV 축에 대해 정규화되므로 쿼리 행은 서로 완전히 독립입니다.
/// 따라서 쿼리를 타일로 잘라 계산한 뒤 이어 붙여도 결과가 완전히 동일합니다.
/// (근사·손실이 전혀 없는 정확한 분해입니다)
///
/// 이 함수는 가용 VRAM 예산 안에 전이 버퍼를 가두는 쿼리 타일 크기를 계산합니다.
/// 0 을 반환하면 타일링을 하지 않는다는 뜻입니다.
pub fn vision_query_tile_size(num_heads: usize, kv_len: usize, dtype_bytes: usize) -> usize {
    // eager_attention_forward 내부 블록 상한
    const ATTN_BLOCK: usize = 4096;
    // attn_folded / broadcast_sub 임시 / exp 결과가 동시에 존재하는 최악 계수
    const LIVE_COPIES: usize = 3;
    // ViT 가중치 + CUDA 컨텍스트 + 파편화 여유
    const RESERVE: u64 = 400_000_000;
    // 어텐션 이외의 전이 텐서(pos_embed 보간, MLP intermediate 등)를 위해 절반만 씁니다.
    const BUDGET_SHARE: u64 = 2;

    const TILE_MIN: usize = 256;
    const TILE_MAX: usize = 4096;

    if num_heads == 0 || kv_len == 0 || dtype_bytes == 0 {
        return 0;
    }

    let span = kv_len.min(ATTN_BLOCK);
    let bytes_per_q_row = LIVE_COPIES
        .saturating_mul(span)
        .saturating_mul(num_heads)
        .saturating_mul(dtype_bytes);
    if bytes_per_q_row == 0 {
        return 0;
    }

    let free_vram = match crate::utils::gpu_mem_info(0) {
        Some((free, _)) => free,
        None => return TILE_MAX.min(kv_len),
    };

    let usable = free_vram.saturating_sub(RESERVE);
    let budget = usable / BUDGET_SHARE;
    if budget == 0 {
        return TILE_MIN;
    }

    let tile = (budget / bytes_per_q_row as u64) as usize;
    let tile = tile.clamp(TILE_MIN, TILE_MAX);
    tile
}

pub fn eager_attention_forward(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_key_value_groups: Option<usize>,
    attention_mask: Option<&Tensor>,
    scaling: f64,
) -> Result<Tensor> {
    let dev = query_states.device();
    let orig_dtype = query_states.dtype();
    
    // 🌟 [VRAM 최적화] F32 연산은 VRAM을 2배로 폭식하므로, Attention 연산 직전에만 BF16으로 다운캐스팅하여 OOM을 원천 차단합니다.
    let target_dtype = if dev.is_cuda_or_rocm() { candle_core::DType::BF16 } else { candle_core::DType::F32 };

    #[cfg(feature = "flash-attn")]
    {
        // 🌟 [ACCEL-CHECK] fused flash-attn 이 꺼져 있거나(CANDLE_FLASH_ATTN=off 등) 이 GPU 의
        //    자가 검증을 통과하지 못하면 아래의 기존 블록 어텐션(GQA 접기 + 블록 온라인 소프트맥스)으로 계산합니다.
        //    (candle 의 비융합 대체 경로는 점수 행렬 전체를 F32 로 만들기 때문에 긴 문맥에서 VRAM 을 많이 씁니다)
        if accel_check::flash_attn_ok(dev, target_dtype, query_states.dim(D::Minus1)?) {
            // Flash Attention은 GQA를 네이티브로 지원하며 BF16/F16을 요구합니다.
            let q = query_states.to_dtype(target_dtype)?.transpose(1, 2)?.contiguous()?;
            let k = key_states.to_dtype(target_dtype)?.transpose(1, 2)?.contiguous()?;
            let v = value_states.to_dtype(target_dtype)?.transpose(1, 2)?.contiguous()?;

            // flash_attn 은 (batch, seq, heads, head_dim) 을 그대로 돌려줍니다.
            // 아래 비-flash 경로와 같은 (b, q_len, h, d) 레이아웃이므로 다시 transpose 하지 않습니다.
            // (예전에는 여기서 transpose(1, 2) 를 한 번 더 해서 호출부의 reshape 가 헤드/토큰을 뒤섞었습니다)
            let attn_output = candle_flash_attn::flash_attn(
                &q, &k, &v,
                scaling as f32,
                attention_mask.is_some(),
            )?;

            return Ok(attn_output.to_dtype(orig_dtype)?.contiguous()?);
        }
    }

    {
        let kv_seq_len = key_states.dim(2)?;
        let block_size = 4096;

        // 마스크가 현재 청크 길이만큼만 있으면(이전 이력 없이 만든 causal 마스크) 이력 부분을 0(모두 보임)으로
        // 채워 오른쪽(bottom-right)에 맞춥니다. flash-attn 경로의 causal 과 같은 의미입니다.
        let padded_mask = match attention_mask {
            Some(m) if m.dim(D::Minus1)? < kv_seq_len => {
                let mut pad_shape = m.dims().to_vec();
                let last = pad_shape.len() - 1;
                pad_shape[last] = kv_seq_len - m.dim(D::Minus1)?;
                let pad = Tensor::zeros(pad_shape, m.dtype(), m.device())?;
                Some(Tensor::cat(&[&pad, m], D::Minus1)?)
            }
            _ => None,
        };
        let attention_mask = padded_mask.as_ref().or(attention_mask);

        // [GQA-FOLD] repeat_kv는 K/V 전체를 groups배로 물리 복제합니다.
        // Q의 head 축을 접으면 동일 결과를 복제 없이 얻으므로, VRAM 피크와 대역폭을 함께 절감합니다.
        // 그룹 수를 넘겨받지 못했으면 헤드 수(q 헤드 = kv 헤드 × groups)에서 구합니다.
        let groups = match num_key_value_groups {
            Some(g) => g.max(1),
            None => {
                let (q_heads, kv_heads) = (query_states.dim(1)?, key_states.dim(1)?);
                if kv_heads > 0 && q_heads % kv_heads == 0 { q_heads / kv_heads } else { 1 }
            }
        };

        let q_aligned = query_states.to_dtype(target_dtype)?.contiguous()?;
        let k_aligned = key_states.to_dtype(target_dtype)?.contiguous()?;
        let v_aligned = value_states.to_dtype(target_dtype)?.contiguous()?;

        let (q_b, q_h, q_l, q_d) = q_aligned.dims4()?;
        let kv_h = k_aligned.dim(1)?;
        let q_work = if groups > 1 {
            q_aligned.reshape((q_b, kv_h, groups * q_l, q_d))?
        } else {
            q_aligned.clone()
        };

        // 짧은 문맥이거나 이미지가 작을 경우 기존 방식 사용
        if kv_seq_len <= block_size {
            // [GQA-FOLD] q_work(접힌 Q)로 matmul하여 KV 헤드 수 불일치를 해소합니다.
            // q_work: [b, kv_h, groups*q_l, d], k_aligned^T: [b, kv_h, d, kv_len]
            let attn_folded = q_work.matmul(&k_aligned.transpose(D::Minus2, D::Minus1)?)?;
            let attn_folded = (attn_folded * scaling)?;
            // [UNFOLD-SCORES] [b, kv_h, groups*q_l, kv_len] → [b, q_h, q_l, kv_len]
            let attn_weights = if groups > 1 {
                attn_folded.reshape((q_b, q_h, q_l, kv_seq_len))?
            } else {
                attn_folded
            };
            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => {
                    let mask_aligned = mask.to_dtype(target_dtype)?;
                    attn_weights.broadcast_add(&mask_aligned)?
                }
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            // [GQA-FOLD] P @ V: softmax 결과를 다시 접어서 V와 곱합니다.
            let p_folded = if groups > 1 {
                attn_weights.reshape((q_b, kv_h, groups * q_l, kv_seq_len))?
            } else {
                attn_weights
            };
            let out_folded = p_folded.matmul(&v_aligned)?;
            // [UNFOLD-OUT] [b, kv_h, groups*q_l, d] → [b, q_h, q_l, d]
            let attn_output = if groups > 1 {
                out_folded.reshape((q_b, q_h, q_l, q_d))?
            } else {
                out_folded
            };
            return Ok(attn_output.to_dtype(orig_dtype)?.transpose(1, 2)?.contiguous()?);
        }

        // [FLASH-DECODING] 5만 토큰 이상의 초장문 문맥을 위한 블록 단위 어텐션 최적화
        let num_blocks = (kv_seq_len + block_size - 1) / block_size;
        let mut block_outputs = Vec::with_capacity(num_blocks); 
        let mut block_lse = Vec::with_capacity(num_blocks); 

        for i in 0..num_blocks {
            let start = i * block_size;
            let end = (start + block_size).min(kv_seq_len);
            let block_len = end - start;
            let k_block = k_aligned.narrow(2, start, block_len)?;
            let v_block = v_aligned.narrow(2, start, block_len)?;
            // [GQA-FOLD] q_work로 matmul → [b, kv_h, groups*q_l, block_len]
            let attn_folded = (q_work.matmul(&k_block.transpose(2, 3)?)? * scaling)?;
            // [UNFOLD-SCORES] 마스크 적용을 위해 원래 헤드 배열로 복원
            let attn_weights = if groups > 1 {
                attn_folded.reshape((q_b, q_h, q_l, block_len))?
            } else {
                attn_folded
            };
            let attn_weights = if let Some(mask) = attention_mask {
                let m_len = mask.dim(D::Minus1)?;
                if start < m_len {
                    let m_end = end.min(m_len);
                    let sub_mask = mask.narrow(D::Minus1, start, m_end - start)?;
                    attn_weights.broadcast_add(&sub_mask.to_dtype(target_dtype)?)?
                } else {
                    attn_weights
                }
            } else {
                attn_weights
            };
            let max_logits = attn_weights.max_keepdim(D::Minus1)?;
            let safe_floor = Tensor::new(-10000.0_f32, max_logits.device())?.to_dtype(max_logits.dtype())?.broadcast_as(max_logits.shape())?;
            let max_logits_safe = max_logits.maximum(&safe_floor)?;
            let exp_weights = attn_weights.broadcast_sub(&max_logits_safe)?.exp()?;
            let sum_exp = exp_weights.sum_keepdim(D::Minus1)?;
            // [GQA-FOLD] P @ V: exp_weights를 접어서 V와 곱한 뒤 다시 펴ます.
            let p_folded = if groups > 1 {
                exp_weights.reshape((q_b, kv_h, groups * q_l, block_len))?
            } else {
                exp_weights
            };
            let out_folded = p_folded.matmul(&v_block)?;
            let out_block = if groups > 1 {
                out_folded.reshape((q_b, q_h, q_l, q_d))?
            } else {
                out_folded
            };
            block_outputs.push(out_block);
            block_lse.push((sum_exp, max_logits));
        }

        let (mut current_exp_sum, mut current_max_logit) = block_lse[0].clone();
        let mut final_output = block_outputs[0].clone();

        for i in 1..num_blocks {
            let (next_exp_sum, next_max_logit) = &block_lse[i];
            let out_i = &block_outputs[i];

            let new_max_logit = current_max_logit.broadcast_maximum(next_max_logit)?;
            let exp_a = current_max_logit.broadcast_sub(&new_max_logit)?.exp()?;
            let exp_b = next_max_logit.broadcast_sub(&new_max_logit)?.exp()?;

            final_output = (final_output.broadcast_mul(&exp_a)? + out_i.broadcast_mul(&exp_b)?)?;
            current_exp_sum = (current_exp_sum.broadcast_mul(&exp_a)? + next_exp_sum.broadcast_mul(&exp_b)?)?;
            current_max_logit = new_max_logit;
        }

        let final_output = final_output.broadcast_div(&current_exp_sum)?;
        Ok(final_output.to_dtype(orig_dtype)?.transpose(1, 2)?.contiguous()?)
    }
}

pub fn get_conv2d(
    vb: VarBuilder,
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
) -> Result<Conv2d> {
    let cfg = Conv2dConfig {
        padding,
        stride,
        dilation,
        groups,
        cudnn_fwd_algo: None,
    };
    let conv2d = if bias {
        conv2d(in_c, out_c, kernel_size, cfg, vb)?
    } else {
        conv2d_no_bias(in_c, out_c, kernel_size, cfg, vb)?
    };
    Ok(conv2d)
}

pub fn get_conv1d(
    vb: VarBuilder,
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
) -> Result<Conv1d> {
    let cfg = Conv1dConfig {
        padding,
        stride,
        dilation,
        groups,
        cudnn_fwd_algo: None,
    };
    let conv1d = if bias {
        conv1d(in_c, out_c, kernel_size, cfg, vb)?
    } else {
        conv1d_no_bias(in_c, out_c, kernel_size, cfg, vb)?
    };
    Ok(conv1d)
}

pub fn get_layer_norm(vb: VarBuilder, eps: f64, dim: usize, affine: bool) -> Result<LayerNorm> {
    let ln_config = LayerNormConfig {
        eps,
        remove_mean: true, // true for layernorm, false for RMSNorm
        affine,            // true for with bias, false for without bias
    };
    let norm = layer_norm(dim, ln_config, vb)?;
    Ok(norm)
}

pub fn get_layer_norm_without_weight(vb: VarBuilder, eps: f64, dim: usize) -> Result<LayerNorm> {
    let weight = Tensor::ones(dim, vb.dtype(), vb.device())?;
    let bias = Tensor::zeros(dim, vb.dtype(), vb.device())?;
    Ok(LayerNorm::new(weight, bias, eps))
}

pub fn get_batch_norm(vb: VarBuilder, eps: f64, dim: usize, affine: bool) -> Result<BatchNorm> {
    let bn_config = BatchNormConfig {
        eps,
        remove_mean: true,
        affine,
        momentum: 0.1,
    };
    let norm = batch_norm(dim, bn_config, vb)?;
    Ok(norm)
}

pub fn deform_conv2d_kernel(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    offset: &Tensor,
    mask: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    // 不考虑空洞卷积, bs = 1
    let (_, in_c, in_h, in_w) = input.dims4()?;
    let (out_channel, _, ker_h, ker_w) = weight.dims4()?;
    let out_h = ((in_h + 2 * padding - ker_h) / stride) + 1;
    let out_w = ((in_w + 2 * padding - ker_w) / stride) + 1;

    let num_kernels = in_c * out_h * out_w;
    let mask_vec = if let Some(mask) = mask {
        Some(mask.squeeze(0)?.to_vec3::<f32>()?)
    } else {
        None
    };
    let offset_vec = offset.squeeze(0)?.to_vec3::<f32>()?;
    let input_vec = input.squeeze(0)?.to_vec3::<f32>()?;
    let mut columns_vec = vec![vec![0.0f32; out_h * out_w]; in_c * ker_h * ker_w];
    for index in 0..num_kernels {
        let out_x = index % out_w;
        let out_y = (index / out_w) % out_h;
        let in_c = index / (out_w * out_h);
        let out_c = in_c * ker_h * ker_w;

        for i in 0..ker_h {
            for j in 0..ker_w {
                let mask_idx = i * ker_w + j;
                let offset_idx = 2 * mask_idx;
                let mask_value = if mask.is_some() {
                    mask_vec.as_ref().unwrap()[mask_idx][out_y][out_x]
                } else {
                    1.0
                };
                let offset_h = offset_vec[offset_idx][out_y][out_x];
                let offset_w = offset_vec[offset_idx + 1][out_y][out_x];
                let y = ((out_y * stride - padding) + i) as f32 + offset_h;
                let x = ((out_x * stride - padding) + j) as f32 + offset_w;
                let val = if y <= -1.0 || in_h as f32 <= y || x <= -1.0 || in_w as f32 <= x {
                    0.0
                } else {
                    let h_low = y.floor();
                    let w_low = x.floor();
                    let h_high = h_low + 1.0;
                    let w_high = w_low + 1.0;
                    let lh = y - h_low;
                    let lw = x - w_low;
                    let hh = 1.0 - lh;
                    let hw = 1.0 - lw;
                    let w1 = hh * hw;
                    let w2 = hh * lw;
                    let w3 = lh * hw;
                    let w4 = lh * lw;
                    let v1 = if h_low >= 0.0 && w_low >= 0.0 {
                        input_vec[in_c][h_low as usize][w_low as usize]
                    } else {
                        0.0
                    };
                    let v2 = if h_low >= 0.0 && w_high <= (in_w - 1) as f32 {
                        input_vec[in_c][h_low as usize][w_high as usize]
                    } else {
                        0.0
                    };
                    let v3 = if h_high <= (in_h - 1) as f32 && w_low >= 0.0 {
                        input_vec[in_c][h_high as usize][w_low as usize]
                    } else {
                        0.0
                    };
                    let v4 = if h_high <= (in_h - 1) as f32 && w_high <= (in_w - 1) as f32 {
                        input_vec[in_c][h_high as usize][w_high as usize]
                    } else {
                        0.0
                    };
                    w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
                };
                columns_vec[out_c + i * ker_w + j][out_y * out_w + out_x] = mask_value * val;
            }
        }
    }

    let columns = Tensor::new(columns_vec, weight.device())?;
    let mut out =
        weight
            .flatten_from(1)?
            .matmul(&columns)?
            .reshape((1, out_channel, out_h, out_w))?;
    if let Some(bias) = bias {
        out = out.broadcast_add(bias)?;
    }
    Ok(out)
}

pub struct LlamaModel {
    pub embed_tokens: Embedding,
    layers: Vec<NaiveAttnGateUpDownMLPBlock>,
    norm: RmsNorm,
    rotary_emb: RoPE,
}

impl LlamaModel {
    pub fn new(
        vb: VarBuilder,
        vocab_size: usize,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: Option<usize>,
        head_dim: Option<usize>,
        attn_bias: bool,
        attn_pp_name: &str,
        o_proj_pp_name: Option<&str>,
        intermediate_size: usize,
        hidden_act: Activation,
        mlp_bias: bool,
        mlp_pp_name: &str,
        norm_eps: f64,
        input_norm_pp_name: &str,
        post_norm_pp_name: &str,
        rope_theta_base: f32,
    ) -> Result<Self> {
        let embed_tokens = embedding(vocab_size, hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        for i in 0..num_hidden_layers {
            let layers_i = NaiveAttnGateUpDownMLPBlock::new(
                vb_layers.pp(i),
                hidden_size,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                attn_bias,
                attn_pp_name,
                o_proj_pp_name,
                intermediate_size,
                hidden_act,
                mlp_bias,
                mlp_pp_name,
                norm_eps,
                input_norm_pp_name,
                post_norm_pp_name,
            )?;
            layers.push(layers_i);
        }
        let norm = rms_norm(hidden_size, norm_eps, vb.pp("norm"))?;
        let head_dim = head_dim.unwrap_or(hidden_size / num_attention_heads);
        let rotary_emb = RoPE::new(head_dim, rope_theta_base, vb.device())?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
        })
    }

    pub fn forward(&mut self, inputs_embeds: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;

        let (cos, sin) = self
            .rotary_emb
            .forward(seqlen_offset, seq_len, inputs_embeds.device())?;
        let mut xs = inputs_embeds.clone();
        let attention_mask: Option<Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                Some(prepare_causal_attention_mask(
                    b_size,
                    seq_len,
                    0,
                    xs.device(),
                )?)
            }
        };
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, &cos, &sin, attention_mask.as_ref())?;
        }
        let xs = xs.apply(&self.norm)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

pub struct LlamaForCausalLM {
    pub model: LlamaModel,
    lm_head: Linear,
}

impl LlamaForCausalLM {
    pub fn new(
        vb: VarBuilder,
        vocab_size: usize,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: Option<usize>,
        head_dim: Option<usize>,
        attn_bias: bool,
        attn_pp_name: &str,
        o_proj_pp_name: Option<&str>,
        intermediate_size: usize,
        hidden_act: Activation,
        mlp_bias: bool,
        mlp_pp_name: &str,
        norm_eps: f64,
        input_norm_pp_name: &str,
        post_norm_pp_name: &str,
        rope_theta_base: f32,
    ) -> Result<Self> {
        let model = LlamaModel::new(
            vb.pp("model"),
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            attn_bias,
            attn_pp_name,
            o_proj_pp_name,
            intermediate_size,
            hidden_act,
            mlp_bias,
            mlp_pp_name,
            norm_eps,
            input_norm_pp_name,
            post_norm_pp_name,
            rope_theta_base,
        )?;
        let lm_head = linear_no_bias(hidden_size, vocab_size, vb.pp("lm_head"))?;
        Ok(Self { model, lm_head })
    }

    pub fn forward(&mut self, inputs_embeds: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let outputs = self.model.forward(inputs_embeds, seqlen_offset)?;
        let seq_len = outputs.dim(1)?;
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }
    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

pub struct GLU {
    dim: usize,
}

impl GLU {
    pub fn new(dim: usize) -> Result<Self> {
        Ok(Self { dim })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x_ = xs.chunk(2, self.dim)?;
        let x_1 = sigmoid(x_[1].as_ref())?;
        let xs = x_1.mul(x_[0].as_ref())?;
        Ok(xs)
    }
}

pub struct GEGLU {
    dim: usize,
}

impl GEGLU {
    pub fn new(dim: usize) -> Result<Self> {
        Ok(Self { dim })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x_ = xs.chunk(2, self.dim)?;
        let x_1 = x_[1].as_ref().gelu()?;
        let xs = x_1.mul(x_[0].as_ref())?;
        Ok(xs)
    }
}

pub struct WNConv1d {
    conv: Conv1d,
}
impl WNConv1d {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        stride: usize,
        bias: bool,
    ) -> Result<Self> {
        let in_c = in_c / groups;
        let weight_g = vb.get((out_c, 1, 1), "weight_g")?;
        let weight_v = vb.get((out_c, in_c, kernel_size), "weight_v")?;
        // let bias = vb.get(out_c, "bias").ok();
        let bias = if bias {
            vb.get(out_c, "bias").ok()
        } else {
            None
        };
        let weight_norm = weight_v.sqr()?.sum_keepdim(1)?.sum_keepdim(2)?.sqrt()?;
        let normalized_weight = weight_v.broadcast_div(&weight_norm)?;
        let scaled_weight = normalized_weight.broadcast_mul(&weight_g)?;
        let cfg = Conv1dConfig {
            padding,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };
        let conv = Conv1d::new(scaled_weight, bias, cfg);
        Ok(Self { conv })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        Ok(x)
    }
}

pub struct WNConvTranspose1d {
    conv_transpose: ConvTranspose1d,
}

impl WNConvTranspose1d {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        dilation: usize,
        kernel_size: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
        stride: usize,
    ) -> Result<Self> {
        let in_c = in_c / groups;
        let weight_g = vb.get((in_c, 1, 1), "weight_g")?;
        let weight_v = vb.get((in_c, out_c, kernel_size), "weight_v")?;
        let bias = vb.get(out_c, "bias").ok();
        let weight_norm = weight_v.sqr()?.sum_keepdim(1)?.sum_keepdim(2)?.sqrt()?;
        let normalized_weight = weight_v.broadcast_div(&weight_norm)?;
        let scaled_weight = normalized_weight.broadcast_mul(&weight_g)?;
        let config = ConvTranspose1dConfig {
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        };
        let conv_transpose = ConvTranspose1d::new(scaled_weight, bias, config);
        Ok(Self { conv_transpose })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_transpose.forward(x)?;
        Ok(x)
    }
}

pub struct Conv2dWithBN {
    conv_0: Conv2d,
    bn_1: BatchNorm,
    bn_with_relu: bool,
}

impl Conv2dWithBN {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        ks: usize,
        padding: usize,
        stride: usize,
        bias: bool,
        bn_with_relu: bool,
    ) -> Result<Self> {
        let conv_0 = get_conv2d(vb.pp("0"), in_c, out_c, ks, padding, stride, 1, 1, bias)?;
        let bn_1 = get_batch_norm(vb.pp("1"), 1e-5, out_c, true)?;
        Ok(Self {
            conv_0,
            bn_1,
            bn_with_relu,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_0.forward(x)?;
        let mut x = self.bn_1.forward_t(&x, false)?;
        if self.bn_with_relu {
            x = x.relu()?;
        }
        Ok(x)
    }
}

pub struct WNLinear {
    linear: Linear,
}
impl WNLinear {
    pub fn new(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Self> {
        let weight_g = vb.get((out_dim, 1), "weight_g")?;
        let weight_v = vb.get((out_dim, in_dim), "weight_v")?;

        let bias = if bias {
            vb.get(out_dim, "bias").ok()
        } else {
            None
        };
        let weight_norm = weight_v.sqr()?.sum_keepdim(0)?.sqrt()?.affine(1.0, 1e-8)?;
        let normalized_weight = weight_v.broadcast_div(&weight_norm)?;
        let scaled_weight = normalized_weight.broadcast_mul(&weight_g)?;
        let linear = Linear::new(scaled_weight, bias);
        Ok(Self { linear })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear.forward(x)?;
        Ok(x)
    }
}

pub fn mish(xs: &Tensor) -> Result<Tensor> {
    let tanh = xs.exp()?.affine(1.0, 1.0)?.log()?.tanh()?;
    let xs = xs.mul(&tanh)?;
    Ok(xs)
}

pub fn softplus(xs: &Tensor) -> Result<Tensor> {
    // ln(1 + exp(x))
    Ok((xs.exp()? + 1.0)?.log()?)
}

pub fn softplus_stable(xs: &Tensor) -> Result<Tensor> {
    // max(x, 0) + ln(1 + exp(-abs(x)))
    let zero = Tensor::zeros_like(xs)?;
    let x_max_0 = xs.maximum(&zero)?;
    Ok((xs.abs()?.neg()?.exp()? + 1.0)?.log()?.add(&x_max_0)?)
}

pub fn conv1d_depthwise(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let len_in = input.dim(2)?;
    let weight = weight.squeeze(1)?.to_dtype(input.dtype())?;
    let kernel_size = weight.dim(1)?;

    let len_out = len_in - kernel_size + 1;

    let out = if len_out == 1 {
        // 디코딩 초고속 패스
        input.broadcast_mul(&weight.unsqueeze(0)?)?.sum_keepdim(2)?
    } else {
        // 프리필(Prefill) 처리용 기존 패스
        let mut out = input
            .narrow(2, 0, len_out)?
            .broadcast_mul(&weight.narrow(1, 0, 1)?.unsqueeze(0)?)?;
        for k in 1..kernel_size {
            
            let piece = input.narrow(2, k, len_out)?.broadcast_mul(&weight.narrow(1, k, 1)?.unsqueeze(0)?)?;
            out = out.add(&piece)?;
        }
        out
    };

    match bias {
        None => Ok(out),
        Some(bias) => {
            let b = bias.dims1()?;
            
            let bias = bias.reshape((1, b, 1))?.to_dtype(input.dtype())?;
            Ok(out.broadcast_add(&bias)?)
        }
    }
}
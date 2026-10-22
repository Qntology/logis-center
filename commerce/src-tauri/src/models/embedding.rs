use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, Module};
use crate::utils::GpuDeviceExt;
use candle_nn::{VarBuilder, linear_no_bias as linear};
use serde::Deserialize;
use std::sync::Arc;
use tokenizers::Tokenizer;
use std::path::Path;

pub mod text_cache;
pub mod cpu_par;
mod batched;
pub(crate) mod cpu_packed;
pub use batched::{BatchOpts, BatchRunStats};
#[cfg(feature = "vulkan")]
pub(crate) use batched::vulkan_exec_mode;

/// 임베딩 차원 (granite-embedding-97m-multilingual-r2 hidden_size).
pub const EMBED_DIM: usize = 384;
/// 한 번의 순전파에 넣는 최대 토큰 수 (기존 embed() 와 동일한 512 청크).
pub const EMBED_CHUNK_TOKENS: usize = 512;

// =====================================================================
// 🌟 [EMBED-BATCHED] 배치 순전파 사용 여부
// ---------------------------------------------------------------------
//   CUDA · ROCm(AMD) · CPU · Vulkan · Metal 에서 기본 ON 입니다. LOGIS_EMBED_BATCHED=0 이면 기존
//   per-text 경로(embed_batch 의 embed())로 돌아갑니다.
//   ROCm 은 AMD RX 6600(gfx1032) 실기기에서 검증했습니다(tests/it/app_embed.rs GPU 테스트):
//   배치 구성과 무관하게 비트 동일, per-text 대비 최소 코사인 0.99976(≤64토큰)/1.0(>64토큰),
//   F32 기준 평균 코사인은 두 경로 동일. 문제가 보이면 LOGIS_EMBED_BATCHED=0 으로 즉시 되돌릴 수 있습니다.
//   CPU · Vulkan · Metal 은 고정 모양 대신 동적 행 수로 묶습니다(batched.rs ⑥, BatchOpts::for_device).
//   🌟 [MAC-VERIFY] Metal (GitHub macos-15, Apple Paravirtual device, docs/mac-verify.md): 동적 모드가
//   per-text 와 320/320 비트 동일 · 구성 무관(단건 12/12 · 표본 12/12 · 역순 절반 160/160) · 전체 2.7배 ·
//   짧은 구 4.6배 · 단건 13.4 vs 13.1 ms. 고정 모양은 0.4배(단건 346.6 vs 14.6 ms)였습니다. candle 의 Metal
//   f32 GEMM(MLX 포트)은 M ≥ 2 에서 원소마다 k 를 순서대로 누산해 행 결과가 M 과 무관하고(포크
//   matmul_m_invariance_tests: M 2..300 불일치 0, 배치 attention 불일치 0), M = 1(gemv)만 다른데
//   임베딩은 [CLS]·[SEP] 로 최소 3토큰이라 M = 1 이 생기지 않습니다.
//   값은 프로세스 안에서 한 번만 읽습니다. 캐시 공간(순전파 태그)이
//   이 판정에 묶여 있어, 도중에 바뀌면 같은 텍스트가 두 공간에 섞이기 때문입니다.
// =====================================================================
/// None = 미설정, Some(false) = "0", Some(true) = 그 밖의 값.
static EMBED_BATCHED_ENV: once_cell::sync::Lazy<Option<bool>> = once_cell::sync::Lazy::new(|| {
    let v = std::env::var("LOGIS_EMBED_BATCHED").ok().map(|v| v.trim() != "0");
    println!(
        "[EMBED-BATCHED] LOGIS_EMBED_BATCHED → {} (CUDA/ROCm/CPU/Vulkan/Metal 기본 ON)",
        match v { None => "미설정", Some(true) => "배치 순전파 ON", Some(false) => "per-text 순전파" }
    );
    v
});

/// 이 장치에서 배치 순전파(embed_many)를 쓰는지 여부.
pub fn batched_enabled(device: &Device) -> bool {
    if device.is_cuda() || device.is_rocm() || device.is_cpu() || device.is_vulkan() || device.is_metal() {
        EMBED_BATCHED_ENV.unwrap_or(true)
    } else {
        false
    }
}

// =====================================================================
// 🌟 [EMBED-FORWARD] granite-embedding-97m-multilingual-r2 순전파 판본
// ---------------------------------------------------------------------
//  ── 무엇이 다른가 ──
//   모델 config.json 은 ModernBertModel 입니다: LayerNorm(norm_bias=false), 전역 층(3층마다)
//   RoPE θ 150000 · 지역 층 θ 160000 + ±64 슬라이딩 창(local_attention 128), 공식 문장 임베딩은
//   CLS 풀링(1_Pooling: pooling_mode_cls_token) + 정규화입니다.
//   이 파일의 기존 순전파(Legacy)는 RmsNorm · 모든 층 θ 150000 · 전역 어텐션 · 평균 풀링이고,
//   GPU 에서는 RoPE 표를 BF16 으로 만들어 256 을 넘는 위치가 짝수로 뭉개집니다.
//   두 판본 벡터의 코사인은 0.79~0.89 입니다(로컬 가중치 재현).
//
//  ── 왜 기본값을 바꾸지 않는가 ──
//   라벨 뱅크 · z 점수 바닥 · SDS 기준선 · LanceDB 에 이미 저장된 벡터가 전부 Legacy 공간입니다.
//   같은 CI 문서 라벨 19개 라우팅 재현(대문자 접기 포함): Legacy 19/19, Exact(CLS) 17/19
//   (INVOICE NUMBER→amount · TOTAL WEIGHT→item_gross_weight), Exact+평균 풀링 18/19.
//   Exact 는 재색인 · 임계 재보정과 함께 옮겨야 하므로 LOGIS_EMBED_FORWARD 로만 켭니다.
//   판본마다 캐시 공간(순전파 태그)이 달라 같은 텍스트의 두 벡터가 섞이지 않습니다.
// =====================================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardMode {
    /// 기존 순전파 (RmsNorm · 단일 θ · 전역 어텐션 · 평균 풀링). 기본값.
    Legacy,
    /// ModernBERT 기준 순전파 + CLS 풀링 (공식 문장 임베딩).
    Exact,
    /// ModernBERT 기준 순전파 + 평균 풀링 (이관 실험용).
    ExactMean,
}

impl ForwardMode {
    pub fn from_env_value(v: Option<&str>) -> Self {
        match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            Some("exact") | Some("exact-cls") | Some("modernbert") => ForwardMode::Exact,
            Some("exact-mean") => ForwardMode::ExactMean,
            _ => ForwardMode::Legacy,
        }
    }

    /// 캐시 공간을 나누는 순전파 태그 접미사. Legacy 는 예전 태그 그대로입니다.
    pub fn tag_suffix(&self) -> Option<&'static str> {
        match self {
            ForwardMode::Legacy => None,
            ForwardMode::Exact => Some("+modernbert-cls/v1"),
            ForwardMode::ExactMean => Some("+modernbert-mean/v1"),
        }
    }

    pub fn is_exact(&self) -> bool {
        !matches!(self, ForwardMode::Legacy)
    }

    fn cls_pooling(&self) -> bool {
        matches!(self, ForwardMode::Exact)
    }
}

/// 값은 프로세스 안에서 한 번만 읽습니다(캐시 공간이 이 판정에 묶여 있습니다).
static EMBED_FORWARD_ENV: once_cell::sync::Lazy<ForwardMode> = once_cell::sync::Lazy::new(|| {
    let raw = std::env::var("LOGIS_EMBED_FORWARD").ok();
    let mode = ForwardMode::from_env_value(raw.as_deref());
    println!(
        "[EMBED-FORWARD] LOGIS_EMBED_FORWARD → {} ({})",
        raw.as_deref().unwrap_or("미설정"),
        match mode {
            ForwardMode::Legacy => "기존 순전파 · 평균 풀링 (기본)",
            ForwardMode::Exact => "ModernBERT 기준 순전파 · CLS 풀링",
            ForwardMode::ExactMean => "ModernBERT 기준 순전파 · 평균 풀링",
        }
    );
    if mode.is_exact() {
        println!(
            "[EMBED-FORWARD] ⚠️ LanceDB 에 이미 저장된 아이템·청크 벡터와 bias.json 뱅크 임계값은 기존 순전파 기준입니다. 이 판본으로 운영하려면 재색인과 임계 재보정이 함께 필요합니다 (캐시는 판본별 공간이라 섞이지 않습니다)."
        );
    }
    mode
});

/// 이 프로세스의 임베딩 순전파 판본.
pub fn forward_mode() -> ForwardMode {
    *EMBED_FORWARD_ENV
}

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub norm_eps: f64,
    pub vocab_size: usize,
    pub pad_token_id: Option<usize>,
    #[serde(default)]
    pub global_rope_theta: Option<f64>,
    #[serde(default)]
    pub local_rope_theta: Option<f64>,
    #[serde(default)]
    pub local_attention: Option<usize>,
    #[serde(default)]
    pub global_attn_every_n_layers: Option<usize>,
}

// --- Common Layers ---

/// ModernBERT 의 LayerNorm(bias=False). 평균·분산은 F32 로 계산합니다.
struct LayerNormNoBias {
    weight_f32: Tensor,
    eps: f64,
}

impl LayerNormNoBias {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight_f32 = vb.get(dim, "weight")?.to_dtype(DType::F32)?;
        Ok(Self { weight_f32, eps })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let y = xc.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        y.broadcast_mul(&self.weight_f32)?.to_dtype(x_dtype)
    }
}

/// 층 정규화: Legacy 는 RmsNorm, Exact 는 LayerNorm.
enum Norm {
    Rms(RmsNorm),
    Layer(LayerNormNoBias),
}

impl Norm {
    fn new(mode: ForwardMode, dim: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        if mode.is_exact() {
            Ok(Norm::Layer(LayerNormNoBias::new(dim, eps, vb)?))
        } else {
            Ok(Norm::Rms(RmsNorm::new(dim, eps, vb)?))
        }
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Norm::Rms(n) => n.forward(x),
            Norm::Layer(n) => n.forward(x),
        }
    }

    fn rms(&self) -> Option<&RmsNorm> {
        match self {
            Norm::Rms(n) => Some(n),
            Norm::Layer(_) => None,
        }
    }
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = if x_dtype == DType::BF16 { DType::BF16 } else { DType::F32 };
        let x = x.to_dtype(internal_dtype)?;
        // [EMBED-CPU-PAR] CPU 는 같은 식을 여러 코어로 (같은 비트)
        let variance = cpu_par::powf2(&x)?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x_dtype)?;
        x_normed.broadcast_mul(&self.weight)
    }
}

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> candle_core::Result<Self> {
        let dtype = if device.is_cuda_or_rocm() { DType::BF16 } else { DType::F32 };
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (theta.powf(i as f64 / dim as f64) as f32))
            .collect();
        // VRAM 메모리 절약을 위해 초기화 시점부터 디바이스의 데이터 타입(GPU=BF16, CPU=F32)으로 캐스팅하여 저장합니다.
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(dtype)?.unsqueeze(1)?;
        let freqs = t.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin })
    }

    /// 🌟 [EMBED-FORWARD] F32 RoPE 표. BF16 표는 256 을 넘는 위치(정수)를 정확히 담지 못합니다.
    fn new_f32(dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> candle_core::Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (theta.powf(i as f64 / dim as f64) as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(DType::F32)?.unsqueeze(1)?;
        let freqs = t.matmul(&inv_freq.unsqueeze(0)?)?;
        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()? })
    }

    // 🌟 [ROPE-HOIST] 기존에는 12개 층이 매번 cos/sin 을 narrow → cat → unsqueeze →
    //    contiguous 로 다시 만들었고(q·k 각각), 순전파당 ~48개의 복사 커널이 쓰였습니다.
    //    값은 seq_len 에만 의존하므로 순전파당 한 번 만들고 모든 층이 공유합니다.
    //    같은 텐서 연산(같은 narrow·cat·contiguous)이라 값은 비트 단위로 동일합니다.
    /// [1, 1, seq_len, head_dim] 모양의 (cos, sin).
    fn cos_sin(&self, seq_len: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let cos = Tensor::cat(&[&cos, &cos], candle_core::D::Minus1)?;
        let sin = Tensor::cat(&[&sin, &sin], candle_core::D::Minus1)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?.contiguous()?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?.contiguous()?;
        Ok((cos, sin))
    }

    /// x * cos + rotate_half(x) * sin (기존 apply_rotary 와 같은 연산 순서).
    /// 표가 x 와 다른 dtype(Exact 의 F32 표)이면 표의 dtype 에서 회전한 뒤 x 의 dtype 으로 돌려줍니다.
    fn apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        if x.dtype() != cos.dtype() {
            return Self::apply(&x.to_dtype(cos.dtype())?, cos, sin)?.to_dtype(x.dtype());
        }
        let last_dim = x.dim(candle_core::D::Minus1)?;
        // narrow 이후 메모리 연속성 보장
        let x1 = x.narrow(candle_core::D::Minus1, 0, last_dim / 2)?.contiguous()?;
        let x2 = x.narrow(candle_core::D::Minus1, last_dim / 2, last_dim / 2)?.contiguous()?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)?;
        Ok((x.contiguous()?.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?)
    }
}

// (ModernBERT 아키텍처로 넘어가면서 repeat_kv 함수가 제거되었습니다.)

// --- Float Model ---

struct Mlp {
    wi: candle_nn::Linear,
    wo: candle_nn::Linear,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        let wi = linear(hidden_size, intermediate_size * 2, vb.pp("Wi"))?;
        let wo = linear(intermediate_size, hidden_size, vb.pp("Wo"))?;
        Ok(Self { wi, wo })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let wi_out = self.wi.forward(x)?;
        // silu(gate) * up — CPU 는 같은 식을 여러 코어로, gate/up 복사 없이 (cpu_par, 같은 비트)
        let act = cpu_par::silu_mul_halves(&wi_out)?;
        Ok(self.wo.forward(&act)?)
    }
}

struct Attention {
    wqkv: candle_nn::Linear,
    wo: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let wqkv = linear(cfg.hidden_size, 3 * cfg.hidden_size, vb.pp("Wqkv"))?;
        let wo = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("Wo"))?;
        Ok(Self { wqkv, wo, num_heads: cfg.num_attention_heads, head_dim })
    }

    /// `mask` = [B, 1, 1, L] 가산 마스크(유효 0 / 패딩 -inf). None 이면 기존과 같은 무마스크 경로.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> candle_core::Result<Tensor> {
        let qkv = self.wqkv.forward(x)?;
        let y = self.attend(&qkv, cos, sin, mask)?;
        Ok(self.wo.forward(&y)?)
    }

    /// QKV 투영 뒤 ~ Wo 앞의 attention 본체 (qkv [B, L, 3H] → [B, L, H]). CPU 패킹 경로도 이 연산을 그대로 씁니다.
    fn attend(&self, qkv: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> candle_core::Result<Tensor> {
        let (batch_size, seq_len, _) = qkv.dims3()?;

        // [B, L, 3H] 를 [B, L, 3, heads, hd] 로 보고(복사 없음) q / k / v 를 각각 한 번만 [B, heads, L, hd] 로
        // 복사합니다. 예전(narrow → contiguous → reshape → transpose → contiguous)과 같은 원소 배치입니다.
        let qkv5 = qkv.reshape((batch_size, seq_len, 3, self.num_heads, self.head_dim))?;
        let split = |i: usize| -> candle_core::Result<Tensor> { qkv5.narrow(2, i, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous() };
        let (q, k, v) = (split(0)?, split(1)?, split(2)?);

        let q = RotaryEmbedding::apply(&q, cos, sin)?;
        let k = RotaryEmbedding::apply(&k, cos, sin)?;

        // matmul 진입 전 q와 k.transpose의 연속성 완벽 보장
        let q_cont = q.contiguous()?;
        let k_trans_cont = k.transpose(2, 3)?.contiguous()?;

        let att = (q_cont.matmul(&k_trans_cont)? / (self.head_dim as f64).sqrt())?;
        // 🌟 [EMBED-BATCHED] 패딩 버킷에서만 가산 마스크를 더합니다(exp(-inf)=0).
        let att = match mask {
            Some(m) => att.broadcast_add(m)?,
            None => att,
        };
        let att = cpu_par::softmax_last(&att)?;

        // 두 번째 matmul 진입 전 att와 v의 연속성 보장, 그리고 최종 reshape 전 transpose 결과의 연속성 보장
        att.contiguous()?.matmul(&v.contiguous()?)?.transpose(1, 2)?.contiguous()?.reshape((batch_size, seq_len, self.num_heads * self.head_dim))
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    attn_norm: Option<Norm>,
    mlp_norm: Norm,
}

impl DecoderLayer {
    fn new(cfg: &Config, mode: ForwardMode, vb: VarBuilder) -> candle_core::Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        // 🌟 [CRITICAL FIX] Granite(ModernBERT)는 layers.0에서 attn_norm을 생략하므로 Option(.ok())으로 안전하게 받습니다.
        let attn_norm = Norm::new(mode, cfg.hidden_size, cfg.norm_eps, vb.pp("attn_norm")).ok();
        let mlp_norm = Norm::new(mode, cfg.hidden_size, cfg.norm_eps, vb.pp("mlp_norm"))?;
        Ok(Self { self_attn, mlp, attn_norm, mlp_norm })
    }
}

/// 🌟 [EMBED-FORWARD] Exact 판본의 층별 RoPE · 지역 창.
struct ExactAttention {
    /// 전역 층(layer % every_n == 0) RoPE (θ = global_rope_theta, F32 표)
    global: RotaryEmbedding,
    /// 지역 층 RoPE (θ = local_rope_theta, F32 표)
    local: RotaryEmbedding,
    every_n: usize,
    /// 지역 층이 볼 수 있는 거리 (local_attention / 2)
    window_half: usize,
}

impl ExactAttention {
    /// 지역 층의 슬라이딩 창 가산 마스크 [1, 1, L, L]. 창이 전 길이를 덮으면 None.
    /// HF 와 같이 창 밖은 -inf 대신 큰 음수로 둡니다(패딩 질의 행이 전부 -inf 가 되어 NaN 이 생기지 않게).
    fn window_mask(&self, seq_len: usize, device: &Device, dtype: DType) -> candle_core::Result<Option<Tensor>> {
        if seq_len <= self.window_half + 1 {
            return Ok(None);
        }
        let mut m = vec![0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if i.abs_diff(j) > self.window_half {
                    m[i * seq_len + j] = -1e30;
                }
            }
        }
        Ok(Some(Tensor::from_vec(m, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)?))
    }
}

pub struct FloatModel {
    tok_embeddings: candle_nn::Embedding,
    embeddings_norm: Norm,
    layers: Vec<DecoderLayer>,
    final_norm: Norm,
    rotary: Arc<RotaryEmbedding>,
    /// 🌟 [EMBED-FORWARD] Exact 판본에서만 Some.
    exact: Option<ExactAttention>,
    mode: ForwardMode,
    /// 🌟 [EMBED-CPU-PACKED] CPU + LOGIS_EMBED_CPU_KERNEL=packed 일 때만 (embedding/cpu_packed.rs).
    packed: Option<cpu_packed::PackedModel>,
}

impl FloatModel {
    pub fn new(cfg: &Config, vb: VarBuilder, device: &Device) -> candle_core::Result<Self> {
        Self::new_with_mode(cfg, vb, device, ForwardMode::Legacy)
    }

    pub fn new_with_mode(cfg: &Config, vb: VarBuilder, device: &Device, mode: ForwardMode) -> candle_core::Result<Self> {
        // CPU 강제 할당을 제거하고 전달받은 디바이스(VRAM)와 데이터 타입(BF16)을 그대로 사용합니다.
        let tok_embeddings = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embeddings.tok_embeddings"))?;
        let embeddings_norm = Norm::new(mode, cfg.hidden_size, cfg.norm_eps, vb.pp("embeddings.norm"))?;

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary = Arc::new(RotaryEmbedding::new(head_dim, 32768, 150000.0, device)?);
        let exact = if mode.is_exact() {
            let global_theta = cfg.global_rope_theta.unwrap_or(150000.0);
            let local_theta = cfg.local_rope_theta.unwrap_or(global_theta);
            Some(ExactAttention {
                global: RotaryEmbedding::new_f32(head_dim, 8192, global_theta, device)?,
                local: RotaryEmbedding::new_f32(head_dim, 8192, local_theta, device)?,
                every_n: cfg.global_attn_every_n_layers.unwrap_or(1).max(1),
                window_half: cfg.local_attention.map(|w| w / 2).unwrap_or(usize::MAX / 2),
            })
        } else {
            None
        };

        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, mode, vb.pp(format!("layers.{}", i)))?);
        }
        let final_norm = Norm::new(mode, cfg.hidden_size, cfg.norm_eps, vb.pp("final_norm"))?;
        let mut model = Self { tok_embeddings, embeddings_norm, layers, final_norm, rotary, exact, mode, packed: None };
        if device.is_cpu() && cpu_packed::enabled() && !mode.is_exact() {
            model.packed = model.build_packed(cfg)?;
        }
        Ok(model)
    }

    /// Linear 가중치를 NR 열 패널로 한 번 다시 배치합니다. 모양이 NR 배수가 아니면 None(기존 경로).
    /// 패킹 커널은 Legacy(RmsNorm) 판본만 구현합니다.
    fn build_packed(&self, cfg: &Config) -> candle_core::Result<Option<cpu_packed::PackedModel>> {
        use cpu_packed::{PackedW, NR};
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        if h % NR != 0 || i % NR != 0 {
            println!("[EMBED-CPU-PACKED] hidden {h} / intermediate {i} 가 {NR} 의 배수가 아니라 기존 경로를 씁니다.");
            return Ok(None);
        }
        let (emb_norm, final_norm) = match (self.embeddings_norm.rms(), self.final_norm.rms()) {
            (Some(e), Some(f)) => (e, f),
            _ => return Ok(None),
        };
        let v = |t: &Tensor| -> candle_core::Result<Vec<f32>> { t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>() };
        let mut layers = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            let mlp_norm = match l.mlp_norm.rms() { Some(n) => n, None => return Ok(None) };
            layers.push(cpu_packed::PackedLayer {
                wqkv: PackedW::pack(&v(l.self_attn.wqkv.weight())?, 3 * h, h),
                attn_wo: PackedW::pack(&v(l.self_attn.wo.weight())?, h, h),
                wi: PackedW::pack_glu(&v(l.mlp.wi.weight())?, i, h),
                mlp_wo: PackedW::pack(&v(l.mlp.wo.weight())?, h, i),
                attn_norm: match l.attn_norm.as_ref().map(|n| n.rms()) {
                    Some(Some(n)) => Some(v(&n.weight)?),
                    Some(None) => return Ok(None),
                    None => None,
                },
                mlp_norm: v(&mlp_norm.weight)?,
            });
        }
        Ok(Some(cpu_packed::PackedModel {
            layers,
            emb_norm: v(&emb_norm.weight)?,
            final_norm: v(&final_norm.weight)?,
            eps: cfg.norm_eps as f32,
        }))
    }

    /// 🌟 [EMBED-CPU-PACKED] forward_hidden 의 CPU 패킹 판본. attention 본체(attend)는 같은 candle 연산입니다.
    fn forward_hidden_packed(&self, p: &cpu_packed::PackedModel, x: Tensor, seq_len: usize, mask: Option<&Tensor>) -> candle_core::Result<Tensor> {
        use cpu_packed::{gemm, rms_norm, Blocking, Epilogue};
        let (b, l, h) = x.dims3()?;
        debug_assert_eq!(l, seq_len);
        let m = b * l;
        let i_sz = p.layers.first().map(|pl| pl.mlp_wo.k).unwrap_or(0);
        let threads = rayon::current_num_threads();
        let blk = Blocking::default();
        let mut xs: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut xn = vec![0f32; m * h];
        rms_norm(&xs, h, &p.emb_norm, p.eps, &mut xn);
        std::mem::swap(&mut xs, &mut xn);
        let (cos, sin) = self.rotary.cos_sin(seq_len)?;
        let mut qkv = vec![0f32; m * 3 * h];
        let mut o = vec![0f32; m * h];
        let mut act = vec![0f32; m * i_sz];
        for (layer, pl) in self.layers.iter().zip(&p.layers) {
            let src: &[f32] = match &pl.attn_norm {
                Some(w) => {
                    rms_norm(&xs, h, w, p.eps, &mut xn);
                    &xn
                }
                None => &xs,
            };
            gemm(src, m, &pl.wqkv, &mut qkv, blk, Epilogue::Store, threads);
            let qkv_t = Tensor::from_slice(&qkv, (b, l, 3 * h), &Device::Cpu)?;
            let y = layer.self_attn.attend(&qkv_t, &cos, &sin, mask)?.flatten_all()?.to_vec1::<f32>()?;
            gemm(&y, m, &pl.attn_wo, &mut o, blk, Epilogue::Store, threads);
            xs.iter_mut().zip(&o).for_each(|(a, d)| *a += d);
            rms_norm(&xs, h, &pl.mlp_norm, p.eps, &mut xn);
            gemm(&xn, m, &pl.wi, &mut act, blk, Epilogue::SiluMul, threads);
            gemm(&act, m, &pl.mlp_wo, &mut o, blk, Epilogue::Store, threads);
            xs.iter_mut().zip(&o).for_each(|(a, d)| *a += d);
        }
        rms_norm(&xs, h, &p.final_norm, p.eps, &mut xn);
        Tensor::from_vec(xn, (b, l, h), &Device::Cpu)
    }

    pub fn forward(&self, input_ids: &Tensor, _device: &Device) -> candle_core::Result<Tensor> {
        let seq_len = input_ids.dim(0)?;
        // 임베딩 텐서가 이미 VRAM에 올라가 있으므로 to_device() 복사 오버헤드를 제거합니다.
        let x = self.tok_embeddings.forward(input_ids)?;
        let x = x.reshape((1, seq_len, ()))?;
        self.forward_hidden(x, seq_len, None)
    }

    /// 🌟 [EMBED-BATCHED] `ids` = [B*L] (행 우선) → [B, L, H].
    /// B=1 · mask=None 이면 `forward` 와 완전히 같은 연산열입니다.
    pub(crate) fn forward_batch(&self, ids: &Tensor, batch: usize, seq_len: usize, mask: Option<&Tensor>) -> candle_core::Result<Tensor> {
        let x = self.tok_embeddings.forward(ids)?;
        let x = x.reshape((batch, seq_len, ()))?;
        self.forward_hidden(x, seq_len, mask)
    }

    fn forward_hidden(&self, x: Tensor, seq_len: usize, mask: Option<&Tensor>) -> candle_core::Result<Tensor> {
        if let Some(p) = &self.packed {
            return self.forward_hidden_packed(p, x, seq_len, mask);
        }
        let mut x = self.embeddings_norm.forward(&x)?;
        let (cos, sin) = self.rotary.cos_sin(seq_len)?;
        // 🌟 [EMBED-FORWARD] Exact: 전역 층과 지역 층이 서로 다른 RoPE 표를 쓰고, 지역 층은 ±창 밖을 보지 않습니다.
        let exact_tables = match &self.exact {
            Some(ex) => {
                let (cg, sg) = ex.global.cos_sin(seq_len)?;
                let (cl, sl) = ex.local.cos_sin(seq_len)?;
                let window = ex.window_mask(seq_len, x.device(), x.dtype())?;
                let local_mask = match (mask, window) {
                    (Some(m), Some(w)) => Some(m.broadcast_add(&w)?),
                    (None, Some(w)) => Some(w),
                    (m, None) => m.cloned(),
                };
                Some((cg, sg, cl, sl, local_mask, ex.every_n))
            }
            None => None,
        };

        for (li, layer) in self.layers.iter().enumerate() {
            let (c, s, m) = match &exact_tables {
                None => (&cos, &sin, mask),
                Some((cg, sg, cl, sl, local_mask, every_n)) => {
                    if li % every_n == 0 { (cg, sg, mask) } else { (cl, sl, local_mask.as_ref()) }
                }
            };
            let residual = &x;
            // 🌟 [CRITICAL FIX] attn_norm이 없는 레이어(layers.0)는 직전의 x를 그대로 사용합니다.
            let x_norm = match &layer.attn_norm {
                Some(norm) => norm.forward(&x)?,
                None => x.clone(),
            };
            let attn_out = layer.self_attn.forward(&x_norm, c, s, m)?;
            x = (residual + attn_out)?;

            let residual = &x;
            let x_norm = layer.mlp_norm.forward(&x)?;
            let mlp_out = layer.mlp.forward(&x_norm)?;
            x = (residual + mlp_out)?;
        }
        self.final_norm.forward(&x)
    }
}

// =====================================================================
// 🌟 [EMBED-TOKENIZER] 토크나이저 전역 공유
// ---------------------------------------------------------------------
//   tokenizer.json 은 27MB 이고 파싱에 0.75~0.85s 가 걸립니다(실측).
//   퍼지마다 EmbeddingModel 이 다시 만들어지면서 매번 다시 파싱했습니다.
//   (경로, 크기, 수정시각) 이 같으면 같은 Arc<Tokenizer> 를 재사용합니다.
//   Tokenizer::encode 는 &self 이고 Send + Sync 이므로 공유해도 안전합니다.
// =====================================================================
type TokKey = (std::path::PathBuf, u64, u128);
static EMBED_TOKENIZERS: once_cell::sync::Lazy<std::sync::Mutex<Vec<(TokKey, Arc<Tokenizer>)>>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(Vec::new()));

fn shared_tokenizer(path: &Path) -> Result<Arc<Tokenizer>> {
    let meta = std::fs::metadata(path)?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let key: TokKey = (path.to_path_buf(), meta.len(), mtime);
    let mut g = EMBED_TOKENIZERS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, t)) = g.iter().find(|(k, _)| *k == key) {
        println!("[EMBED-TOKENIZER] ♻️ 토크나이저 재사용 (재파싱 생략)");
        return Ok(t.clone());
    }
    let started = std::time::Instant::now();
    let tok = Arc::new(Tokenizer::from_file(path).map_err(anyhow::Error::msg)?);
    // 같은 경로의 옛 판본은 버립니다(파일이 교체된 경우).
    g.retain(|(k, _)| k.0 != key.0);
    g.push((key, tok.clone()));
    println!("[EMBED-TOKENIZER] 📖 토크나이저 최초 파싱 {:.0}ms (이후 재로드는 재사용)", started.elapsed().as_secs_f64() * 1000.0);
    Ok(tok)
}

/// 청크 벡터들을 텍스트 하나의 벡터로 합칩니다.
/// 기존 embed() 의 f32 산술(청크 순서대로 누적 → 개수로 나눔 → 재정규화)을 그대로 옮긴 것입니다.
pub(crate) fn aggregate_chunk_vectors(chunks: &[&[f32]]) -> Vec<f32> {
    let mut accumulated_vector = vec![0.0f32; EMBED_DIM];
    let mut total_chunks = 0.0f32;
    for vec in chunks {
        for (i, val) in vec.iter().enumerate() {
            accumulated_vector[i] += val;
        }
        total_chunks += 1.0;
    }
    if total_chunks > 0.0 {
        for val in accumulated_vector.iter_mut() {
            *val /= total_chunks;
        }
        let sum_sq: f32 = accumulated_vector.iter().map(|v| v * v).sum();
        let norm = sum_sq.sqrt();
        if norm > 1e-6 {
            for val in accumulated_vector.iter_mut() {
                *val /= norm;
            }
        }
    }
    accumulated_vector
}

// --- Wrapper ---

pub struct EmbeddingModel {
    model: FloatModel,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    dtype: DType,
    pad_id: u32,
}

impl EmbeddingModel {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::new_with_device(model_path, &Device::Cpu)
    }

    pub fn new_with_device<P: AsRef<Path>>(model_path: P, device: &Device) -> Result<Self> {
        Self::new_with_device_and_mode(model_path, device, forward_mode())
    }

    /// 🌟 [EMBED-FORWARD] 순전파 판본을 직접 고릅니다(검증 · 이관 실험용). 앱 경로는 new_with_device 를 씁니다.
    pub fn new_with_device_and_mode<P: AsRef<Path>>(model_path: P, device: &Device, mode: ForwardMode) -> Result<Self> {
        let model_path = model_path.as_ref();
        let config_path = model_path.join("config.json");
        let tokenizer_path = model_path.join("tokenizer.json");
        let weights_path = model_path.join("model.safetensors");

        println!("[EmbeddingModel] Loading on {:?}...", device);

        let config_str = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let tokenizer = shared_tokenizer(&tokenizer_path)?;

        let dtype = if device.is_cuda_or_rocm() { DType::BF16 } else { DType::F32 };

        let model = if weights_path.exists() {
             println!("[EmbeddingModel] Loading Safetensors model from {:?}", weights_path);
             let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], dtype, device).map_err(anyhow::Error::msg)? };
             FloatModel::new_with_mode(&config, vb, device, mode).map_err(anyhow::Error::msg)?
        } else {
            return Err(anyhow!("No valid model found (looked for model.safetensors)"));
        };

        // 🌟 [EMBED-CACHE] 모델 파일이 바뀌었으면 캐시 공간을 새 지문으로 넘깁니다.
        text_cache::refresh_fingerprint(model_path, text_cache::device_class(device));

        let pad_id = config.pad_token_id.unwrap_or(0) as u32;
        Ok(Self { model, tokenizer, device: device.clone(), dtype, pad_id })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 이 인스턴스의 순전파 판본.
    pub fn forward_mode(&self) -> ForwardMode {
        self.model.mode
    }

    /// 풀링할 토큰 범위: CLS 풀링이면 첫 토큰([CLS]) 하나, 아니면 실제 길이 전체.
    fn pooled_len(&self, len: usize) -> usize {
        if self.model.mode.cls_pooling() { len.min(1) } else { len }
    }

    /// 이 모델 인스턴스가 배치 순전파를 쓰는지 여부.
    pub fn batched_enabled(&self) -> bool {
        batched_enabled(&self.device)
    }

    /// 이 장치의 배치 순전파 정책 (CUDA/ROCm 고정 모양 · CPU/Vulkan 동적 행 수).
    pub fn batch_opts(&self) -> BatchOpts {
        BatchOpts::for_device(&self.device)
    }

    /// 특수 토큰 포함 토큰 수 (embed() 와 같은 encode(text, true)). 진단·검증용.
    pub fn token_len(&self, text: &str) -> Result<usize> {
        Ok(self.tokenizer.encode(text, true).map_err(anyhow::Error::msg)?.get_ids().len())
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true).map_err(anyhow::Error::msg)?;
        let token_ids = tokens.get_ids();

        if token_ids.is_empty() { return Ok(vec![0.0; EMBED_DIM]); }

        let mut chunk_vectors: Vec<Vec<f32>> = Vec::new();
        for chunk in token_ids.chunks(EMBED_CHUNK_TOKENS) {
            // 입력 텐서를 처음부터 타겟 디바이스(VRAM)로 올립니다.
            let input_tensor = Tensor::new(chunk, &self.device).map_err(anyhow::Error::msg)?;
            let hidden_states = self.model.forward(&input_tensor, &self.device).map_err(anyhow::Error::msg)?;
            let take = self.pooled_len(chunk.len());
            let hidden_states = if take == chunk.len() {
                hidden_states
            } else {
                hidden_states.narrow(1, 0, take).map_err(anyhow::Error::msg)?
            };
            let normalized = Self::pool_single(&hidden_states)?;

            // 🌟 [CRITICAL FIX] VRAM(GPU)에 존재하는 최종 연산 결과 텐서를 시스템 메모리(CPU)로 안전하게 명시적 복사한 뒤, F32 배열로 캐스팅합니다.
            let normalized_cpu = normalized.to_device(&Device::Cpu).map_err(anyhow::Error::msg)?;
            let normalized_f32 = normalized_cpu.to_dtype(candle_core::DType::F32).map_err(anyhow::Error::msg)?;
            let vec: Vec<f32> = normalized_f32.flatten_all().map_err(anyhow::Error::msg)?.to_vec1().map_err(anyhow::Error::msg)?;
            chunk_vectors.push(vec);
        }

        let refs: Vec<&[f32]> = chunk_vectors.iter().map(|v| v.as_slice()).collect();
        Ok(aggregate_chunk_vectors(&refs))
    }

    /// [1, s, H] → 평균 풀링 + L2 정규화 [1, H] (기존 embed() 의 GPU 측 연산 그대로).
    fn pool_single(hidden_states: &Tensor) -> Result<Tensor> {
        let (_b, s, _h) = hidden_states.dims3().map_err(anyhow::Error::msg)?;
        let sum = hidden_states.sum(1).map_err(anyhow::Error::msg)?;
        let mean = (sum / (s as f64)).map_err(anyhow::Error::msg)?;
        let norm = mean.sqr().map_err(anyhow::Error::msg)?.sum_all().map_err(anyhow::Error::msg)?.sqrt().map_err(anyhow::Error::msg)?;
        mean.broadcast_div(&norm).map_err(anyhow::Error::msg)
    }

    // =====================================================================
    // 🌟 [CROSSOVER] 동시 순전파 스레드 수를 여유에 맞춰 조절합니다.
    // ---------------------------------------------------------------------
    //  ── 왜 여기인가 ──
    //   embed_batch 는 이름과 달리 '배치 연산' 이 아닙니다.
    //   내부에서 self.embed(text) 를 1건씩 순회하므로,
    //   texts.len() 을 100 으로 주든 1000 으로 주든 순전파는 항상 1건 단위입니다.
    //   시퀀스 길이도 embed() 내부에서 512 로 이미 고정되어 있습니다.
    //   즉 activation 의 유일한 변수는 '동시에 순전파하는 스레드 수' 입니다.
    //   (CUDA/ROCm 의 기본 경로는 이제 embed_many 이며, 이 함수는
    //    LOGIS_EMBED_BATCHED=0 · Vulkan · CPU 경로에서만 쓰입니다)
    //
    //  ── 3 이 왜 위험한가 ──
    //   Attention::forward 는 .contiguous() 를 12회 호출하며 매번 새 텐서를
    //   할당합니다. 그 순간 점유가 스레드 수만큼 배가됩니다.
    //   생성 모델(Qwen3 0.6B / Qwen3.5 2B)이 함께 상주 중일 때
    //   이 3배가 곧 OOM 경계선을 넘는 지점입니다.
    //
    //  ── 판정 근거 ──
    //   '한 스레드가 쓰는 순간 점유' 를 알면 몇 개를 띄울 수 있는지 나옵니다.
    //   그 값은 관측으로 학습되며(embed_activation_unit_mb), 관측 전에는
    //   기존 동작(3)을 그대로 유지해 회귀가 없습니다.
    //   3 은 여전히 상한이고, 줄이는 방향으로만 작동합니다.
    // =====================================================================
    /// 한 스레드가 순전파 중 쓰는 VRAM(MB) 관측치. 0 이면 미관측.
    fn observed_unit_mb() -> u64 {
        EMBED_UNIT_ACTIVATION_MB.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn free_vram_mb(&self) -> u64 {
        if self.device.is_cpu() { return u64::MAX; }
        crate::utils::gpu_mem_info(0)
            .map(|(free, _)| free / (1024 * 1024))
            .unwrap_or(0)
    }

    /// 지금 띄워도 되는 동시 순전파 스레드 수. 상한은 기존과 같은 3 입니다.
    fn adaptive_thread_count(&self, len: usize) -> usize {
        const MAX_THREADS: usize = 3;
        if self.device.is_cpu() { return 1; }
        // 🌟 [EMBED-THREADS] CUDA 는 1스레드 (E9).
        //    CudaDevice 는 스레드마다 다른 per-thread 기본 스트림에 커널을 넣지만 cuBLAS 핸들(작업 공간)은
        //    하나를 공유합니다. 실측(320건): 3스레드 embed_batch 결과 중 2건이 같은 텍스트의 순차 embed() 와
        //    비트 단위로 달랐습니다(동시 GEMM 의 작업 공간 공유가 의심됨). 3스레드 이득은 1.15배뿐이었고,
        //    CUDA 기본 경로는 이제 배치 순전파(embed_many)이므로 per-text 경로는 결정론을 우선합니다.
        //    LOGIS_EMBED_PERTEXT_THREADS=3 이면 기존 동작으로 돌아갑니다.
        if self.device.is_cuda() {
            let forced = std::env::var("LOGIS_EMBED_PERTEXT_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok());
            match forced {
                Some(n) if n > 1 => return n.min(MAX_THREADS).min(len).max(1),
                _ => return 1,
            }
        }
        let cap = MAX_THREADS.min(len).max(1);
        let unit = Self::observed_unit_mb();
        if unit == 0 {
            // 미관측 : 기존 동작을 그대로 유지합니다.
            return cap;
        }
        let free = self.free_vram_mb();
        // 관측 단위의 2배를 안전 계수로 둡니다.
        // (관측은 '피크 이후 잔여' 라 과소평가 방향이므로 여유를 둡니다)
        let affordable = (free / unit.max(1) / 2).max(1) as usize;
        let picked = affordable.min(cap);
        if picked < cap {
            println!(
                "[EmbeddingModel] 🧵 [CROSSOVER] 자유 {}MB / 스레드당 관측 {}MB → 동시 순전파 {} → {} 로 축소합니다.",
                free, unit, cap, picked
            );
        }
        picked
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(Vec::new()); }

        let mut results = vec![vec![0.0; EMBED_DIM]; texts.len()];

        // 🌟 [CRITICAL FIX] 장치(Device) 타입에 따른 스레드 동적 분기 및 VRAM 최적화
        // GPU(CUDA)일 경우 워프 스케줄링 한계와 PCIe 대역폭 병목을 방지하기 위해 가장 이상적인 3개로 제한합니다.
        // CPU일 경우 스레드 경합(Thread Contention)과 L3 캐시 오염을 막기 위해 무조건 1개(직렬)로 강제 고정합니다.
        //
        // 🌟 [CROSSOVER] 상한 3 은 유지하되, 생성 모델과 동시 상주 중이라
        //    여유가 얇으면 1~2 로 줄입니다. activation 이 스레드 수에 비례하므로
        //    이것이 이 코드베이스에서 순간 피크를 줄이는 유일한 실효 축입니다.
        let num_threads = self.adaptive_thread_count(texts.len());

        let chunk_size = (texts.len() + num_threads - 1) / num_threads;

        std::thread::scope(|s| {
            let mut handles = Vec::new();

            for (chunk_idx, chunk) in texts.chunks(chunk_size).enumerate() {
                let start_idx = chunk_idx * chunk_size;
                handles.push(s.spawn(move || {
                    let mut local_res = Vec::with_capacity(chunk.len());
                    for text in chunk {
                        // 실패는 0벡터로 자리만 지킵니다. 호출부(get_embedding_batch)는
                        // 0벡터를 캐시에 넣지 않으므로 다음 호출에서 다시 계산됩니다.
                        local_res.push(self.embed(text).unwrap_or(vec![0.0; EMBED_DIM]));
                    }
                    (start_idx, local_res)
                }));
            }

            for handle in handles {
                if let Ok((start_idx, local_res)) = handle.join() {
                    for (i, vector) in local_res.into_iter().enumerate() {
                        if start_idx + i < results.len() {
                            results[start_idx + i] = vector;
                        }
                    }
                }
            }
        });

        // 🌟 [MEMORY CLEAR] 임베딩 배치 연산 종료 후 즉각적인 VRAM 동기화 및 가비지 텐서 회수
        if !self.device.is_cpu() {
            let _ = self.device.synchronize();
        }

        Ok(results)
    }

    /// 🌟 [CROSSOVER] 스레드 하나가 쓰는 순간 점유를 관측해 기록합니다.
    ///
    ///  ── 왜 embed_batch 안에서 재지 않는가 ──
    ///   std::thread::scope 안에서는 모든 스레드가 동시에 돌고 있어
    ///   '한 스레드분' 을 분리해 잴 수 없습니다.
    ///   호출부가 (연산 전 free, 연산 후 free, 사용한 스레드 수)를 넘겨 주면
    ///   총 점유를 스레드 수로 나눠 단위값을 얻습니다.
    ///
    ///  ── 누적 최대값을 쓰는 이유 ──
    ///   짧은 텍스트만 들어온 배치는 과소평가됩니다. 그 값으로 스레드 수를
    ///   정하면 다음에 긴 텍스트가 들어올 때 OOM 이 납니다.
    ///   최대값을 유지하면 판정이 항상 보수적인 방향입니다.
    pub fn record_unit_activation(free_before_mb: u64, free_after_mb: u64, threads: usize) {
        if threads == 0 { return; }
        if free_before_mb <= free_after_mb { return; }
        let total = free_before_mb - free_after_mb;
        let unit = (total / threads as u64).max(1);
        let prev = EMBED_UNIT_ACTIVATION_MB.load(std::sync::atomic::Ordering::SeqCst);
        if unit > prev {
            EMBED_UNIT_ACTIVATION_MB.store(unit, std::sync::atomic::Ordering::SeqCst);
            println!(
                "[EmbeddingModel] 📏 [CROSSOVER] 스레드당 순간 점유 관측 갱신: {}MB (총 {}MB / {}스레드)",
                unit, total, threads
            );
        }
    }

    /// 마지막 배치가 실제로 사용한 스레드 수. 호출부의 관측에 필요합니다.
    pub fn last_thread_count(&self, len: usize) -> usize {
        self.adaptive_thread_count(len)
    }
}

/// 🌟 [CROSSOVER] 스레드 하나가 순전파 중 쓰는 VRAM(MB) 누적 최대 관측치.
///
///  ── 왜 전역 static 인가 ──
///   EmbeddingModel 은 unload/reload 로 인스턴스가 계속 교체됩니다.
///   구조체 필드에 두면 재로드마다 관측이 초기화되어
///   매번 처음부터 3스레드로 시작하는 위험한 상태로 되돌아갑니다.
///   프로세스 전역에 임베딩 모델은 하나뿐이므로 static 이 적절하며,
///   model.rs 의 크로스오버 원장이 이미 같은 선례입니다.
static EMBED_UNIT_ACTIVATION_MB: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
mod forward_parity_tests {
    //! 🌟 [EMBED-FORWARD] 순전파 판본별 벡터를 HF transformers 기준 벡터와 대조합니다.
    //! 기준 벡터: scratch 스크립트(granite_ref.py)가 ModernBertModel(F32 · eager)로 만든 JSON
    //!   [{text, n_tokens, cls, mean}] (모두 L2 정규화, 512 토큰 이하).
    //! 실행: LOGIS_EMBED_PARITY_JSON=<json> cargo test --lib -- embed_forward_parity --ignored --nocapture
    //! 모델 폴더는 LOGIS_EMBED_MODEL_DIR, 없으면 앱 데이터 폴더의 models/granite-embedding-97m-multilingual-r2.
    use super::*;

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na * nb) }
    }

    fn summary(label: &str, sims: &[(usize, f32)]) -> (f32, f32) {
        let bucket = |lo: usize, hi: usize| -> String {
            let v: Vec<f32> = sims.iter().filter(|(n, _)| *n >= lo && *n <= hi).map(|(_, s)| *s).collect();
            if v.is_empty() {
                return "-".to_string();
            }
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
            format!("평균 {:.5} 최소 {:.5} ({}건)", mean, min, v.len())
        };
        let all: Vec<f32> = sims.iter().map(|(_, s)| *s).collect();
        let mean = all.iter().sum::<f32>() / all.len().max(1) as f32;
        let min = all.iter().cloned().fold(f32::INFINITY, f32::min);
        println!(
            "[EMBED-PARITY] {label} | 전체 평균 {:.5} 최소 {:.5} | ≤64토큰 {} | 65~256 {} | >256 {}",
            mean,
            min,
            bucket(0, 64),
            bucket(65, 256),
            bucket(257, usize::MAX)
        );
        (mean, min)
    }

    #[test]
    #[ignore = "실모델 대조: 로컬 granite 가중치 + HF 기준 벡터 JSON(LOGIS_EMBED_PARITY_JSON) 필요"]
    fn embed_forward_parity_against_reference() {
        let json_path = match std::env::var("LOGIS_EMBED_PARITY_JSON") {
            Ok(p) => p,
            Err(_) => {
                println!("LOGIS_EMBED_PARITY_JSON 이 없어 건너뜁니다.");
                return;
            }
        };
        let model_dir = std::env::var("LOGIS_EMBED_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| crate::utils::get_app_dir().join("models").join("granite-embedding-97m-multilingual-r2"));
        let refs: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(&json_path).expect("기준 JSON 읽기")).expect("기준 JSON 파싱");
        let texts: Vec<String> = refs.iter().map(|r| r["text"].as_str().unwrap().to_string()).collect();
        let lens: Vec<usize> = refs.iter().map(|r| r["n_tokens"].as_u64().unwrap() as usize).collect();
        let vec_of = |r: &serde_json::Value, k: &str| -> Vec<f32> {
            r[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
        };

        let mut devices: Vec<(&str, Device)> = vec![("cpu-f32", Device::Cpu)];
        if let Ok(d) = Device::new_cuda(0) {
            devices.push(("cuda-bf16", d));
        }
        for (dname, dev) in devices.iter() {
            for mode in [ForwardMode::Legacy, ForwardMode::Exact, ForwardMode::ExactMean] {
                let m = EmbeddingModel::new_with_device_and_mode(&model_dir, dev, mode).expect("모델 적재");
                let key = if mode == ForwardMode::Exact { "cls" } else { "mean" };
                let per_text: Vec<Vec<f32>> = texts.iter().map(|t| m.embed(t).expect("embed")).collect();
                let batched = m.embed_many(&texts, &m.batch_opts()).expect("embed_many");
                for (tok_len, t) in lens.iter().zip(texts.iter()) {
                    assert_eq!(m.token_len(t).unwrap(), *tok_len, "HF 와 같은 토큰화여야 합니다: {t}");
                }
                let sims_pt: Vec<(usize, f32)> =
                    refs.iter().zip(per_text.iter()).zip(lens.iter()).map(|((r, v), n)| (*n, cos(v, &vec_of(r, key)))).collect();
                let sims_b: Vec<(usize, f32)> =
                    refs.iter().zip(batched.iter()).zip(lens.iter()).map(|((r, v), n)| (*n, cos(v, &vec_of(r, key)))).collect();
                let (_, min_pt) = summary(&format!("{dname} {:?} per-text vs HF({key})", mode), &sims_pt);
                let (_, min_b) = summary(&format!("{dname} {:?} batched vs HF({key})", mode), &sims_b);
                if mode.is_exact() && *dname == "cpu-f32" {
                    assert!(min_pt > 0.9999 && min_b > 0.9999, "{dname} {:?}: F32 기준 판본은 HF 와 같아야 합니다", mode);
                }
                if mode.is_exact() && *dname == "cuda-bf16" {
                    assert!(min_pt > 0.995 && min_b > 0.995, "{dname} {:?}: BF16 기준 판본이 HF 와 크게 다릅니다", mode);
                }
            }
        }
    }
}

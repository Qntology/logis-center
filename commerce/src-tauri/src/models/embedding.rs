use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, Module};
use crate::utils::GpuDeviceExt;
use candle_nn::{VarBuilder, linear_no_bias as linear};
use serde::Deserialize;
use std::sync::Arc;
use tokenizers::Tokenizer;
use std::path::Path;

pub mod text_cache;
mod batched;
pub(crate) mod cpu_packed;
pub use batched::{BatchOpts, BatchRunStats};

/// 임베딩 차원 (granite-embedding-97m-multilingual-r2 hidden_size).
pub const EMBED_DIM: usize = 384;
/// 한 번의 순전파에 넣는 최대 토큰 수 (기존 embed() 와 동일한 512 청크).
pub const EMBED_CHUNK_TOKENS: usize = 512;

// =====================================================================
// 🌟 [EMBED-BATCHED] 배치 순전파 사용 여부
// ---------------------------------------------------------------------
//   CUDA · ROCm(AMD) · CPU · Vulkan 에서 기본 ON 입니다. LOGIS_EMBED_BATCHED=0 이면 기존
//   per-text 경로(embed_batch 의 embed())로 돌아갑니다.
//   ROCm 은 AMD RX 6600(gfx1032) 실기기에서 검증했습니다(tests/it/app_embed.rs GPU 테스트):
//   배치 구성과 무관하게 비트 동일, per-text 대비 최소 코사인 0.99976(≤64토큰)/1.0(>64토큰),
//   F32 기준 평균 코사인은 두 경로 동일. 문제가 보이면 LOGIS_EMBED_BATCHED=0 으로 즉시 되돌릴 수 있습니다.
//   CPU · Vulkan 은 고정 모양 대신 동적 행 수로 묶습니다(batched.rs ⑥, BatchOpts::for_device).
//   Metal 은 실기기 검증 전이라 기본은 기존 per-text 경로이고, LOGIS_EMBED_BATCHED=1 이면 켭니다(opt-in).
//   켜면 CUDA/ROCm 과 같은 고정 모양입니다(BatchOpts::for_device): candle 의 Metal GEMM(MLX 포트)은
//   M==1·N==1 이면 gemv, M<16 이면 다른 타일, b·M·N ≥ 2^20 이면 큰 타일을 골라 같은 행도 M 에 따라
//   k 누산 순서가 바뀔 수 있으므로, 행 수를 길이 L 만의 함수로 고정해야 구성 무관이 됩니다.
//   (docs/mac-verify.md · 포크 tests/matmul_m_invariance_tests.rs 의 metal_ 측정)
//   값은 프로세스 안에서 한 번만 읽습니다. 캐시 공간(순전파 태그)이
//   이 판정에 묶여 있어, 도중에 바뀌면 같은 텍스트가 두 공간에 섞이기 때문입니다.
// =====================================================================
/// None = 미설정, Some(false) = "0", Some(true) = 그 밖의 값.
static EMBED_BATCHED_ENV: once_cell::sync::Lazy<Option<bool>> = once_cell::sync::Lazy::new(|| {
    let v = std::env::var("LOGIS_EMBED_BATCHED").ok().map(|v| v.trim() != "0");
    println!(
        "[EMBED-BATCHED] LOGIS_EMBED_BATCHED → {} (CUDA/ROCm/CPU/Vulkan 기본 ON · Metal 은 =1 일 때만)",
        match v { None => "미설정", Some(true) => "배치 순전파 ON", Some(false) => "per-text 순전파" }
    );
    v
});

/// 이 장치에서 배치 순전파(embed_many)를 쓰는지 여부.
pub fn batched_enabled(device: &Device) -> bool {
    if device.is_cuda() || device.is_rocm() || device.is_cpu() || device.is_vulkan() {
        EMBED_BATCHED_ENV.unwrap_or(true)
    } else if device.is_metal() {
        // 🌟 [MAC-VERIFY] Metal 은 opt-in (실행 검증 전). 고정 모양 정책은 BatchOpts::for_device.
        EMBED_BATCHED_ENV.unwrap_or(false)
    } else {
        false
    }
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
}

// --- Common Layers ---

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
        let variance = x.powf(2.0)?.mean_keepdim(candle_core::D::Minus1)?;
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
    fn apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
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
        let last_dim = wi_out.dim(candle_core::D::Minus1)?;
        // narrow 이후 메모리 연속성 보장
        let gate = wi_out.narrow(candle_core::D::Minus1, 0, last_dim / 2)?.contiguous()?;
        let up = wi_out.narrow(candle_core::D::Minus1, last_dim / 2, last_dim / 2)?.contiguous()?;
        // 연산 결과에 대해 Linear 계층(matmul) 진입 전 연속성 보장
        let act = (candle_nn::ops::silu(&gate)? * up)?.contiguous()?;
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

        let hidden_size = self.num_heads * self.head_dim;
        // narrow 이후 메모리 연속성 보장
        let q_all = qkv.narrow(2, 0, hidden_size)?.contiguous()?;
        let k_all = qkv.narrow(2, hidden_size, hidden_size)?.contiguous()?;
        let v_all = qkv.narrow(2, 2 * hidden_size, hidden_size)?.contiguous()?;

        // reshape 및 transpose 이후 메모리 연속성 보장
        let q = q_all.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k_all.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v_all.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

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
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;

        // 두 번째 matmul 진입 전 att와 v의 연속성 보장, 그리고 최종 reshape 전 transpose 결과의 연속성 보장
        att.contiguous()?.matmul(&v.contiguous()?)?.transpose(1, 2)?.contiguous()?.reshape((batch_size, seq_len, self.num_heads * self.head_dim))
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    attn_norm: Option<RmsNorm>,
    mlp_norm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        // 🌟 [CRITICAL FIX] Granite(ModernBERT)는 layers.0에서 attn_norm을 생략하므로 Option(.ok())으로 안전하게 받습니다.
        let attn_norm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("attn_norm")).ok();
        let mlp_norm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("mlp_norm"))?;
        Ok(Self { self_attn, mlp, attn_norm, mlp_norm })
    }
}

pub struct FloatModel {
    tok_embeddings: candle_nn::Embedding,
    embeddings_norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    rotary: Arc<RotaryEmbedding>,
    /// 🌟 [EMBED-CPU-PACKED] CPU + LOGIS_EMBED_CPU_KERNEL=packed 일 때만 (embedding/cpu_packed.rs).
    packed: Option<cpu_packed::PackedModel>,
}

impl FloatModel {
    pub fn new(cfg: &Config, vb: VarBuilder, device: &Device) -> candle_core::Result<Self> {
        // CPU 강제 할당을 제거하고 전달받은 디바이스(VRAM)와 데이터 타입(BF16)을 그대로 사용합니다.
        let tok_embeddings = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embeddings.tok_embeddings"))?;
        let embeddings_norm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("embeddings.norm"))?;

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary = Arc::new(RotaryEmbedding::new(head_dim, 32768, 150000.0, device)?);

        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, vb.pp(format!("layers.{}", i)))?);
        }
        let final_norm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("final_norm"))?;
        let mut model = Self { tok_embeddings, embeddings_norm, layers, final_norm, rotary, packed: None };
        if device.is_cpu() && cpu_packed::enabled() {
            model.packed = model.build_packed(cfg)?;
        }
        Ok(model)
    }

    /// Linear 가중치를 NR 열 패널로 한 번 다시 배치합니다. 모양이 NR 배수가 아니면 None(기존 경로).
    fn build_packed(&self, cfg: &Config) -> candle_core::Result<Option<cpu_packed::PackedModel>> {
        use cpu_packed::{PackedW, NR};
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        if h % NR != 0 || i % NR != 0 {
            println!("[EMBED-CPU-PACKED] hidden {h} / intermediate {i} 가 {NR} 의 배수가 아니라 기존 경로를 씁니다.");
            return Ok(None);
        }
        let v = |t: &Tensor| -> candle_core::Result<Vec<f32>> { t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>() };
        let mut layers = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            layers.push(cpu_packed::PackedLayer {
                wqkv: PackedW::pack(&v(l.self_attn.wqkv.weight())?, 3 * h, h),
                attn_wo: PackedW::pack(&v(l.self_attn.wo.weight())?, h, h),
                wi: PackedW::pack_glu(&v(l.mlp.wi.weight())?, i, h),
                mlp_wo: PackedW::pack(&v(l.mlp.wo.weight())?, h, i),
                attn_norm: match &l.attn_norm {
                    Some(n) => Some(v(&n.weight)?),
                    None => None,
                },
                mlp_norm: v(&l.mlp_norm.weight)?,
            });
        }
        Ok(Some(cpu_packed::PackedModel {
            layers,
            emb_norm: v(&self.embeddings_norm.weight)?,
            final_norm: v(&self.final_norm.weight)?,
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

        for layer in &self.layers {
            let residual = &x;
            // 🌟 [CRITICAL FIX] attn_norm이 없는 레이어(layers.0)는 직전의 x를 그대로 사용합니다.
            let x_norm = match &layer.attn_norm {
                Some(norm) => norm.forward(&x)?,
                None => x.clone(),
            };
            let attn_out = layer.self_attn.forward(&x_norm, &cos, &sin, mask)?;
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
             FloatModel::new(&config, vb, device).map_err(anyhow::Error::msg)?
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

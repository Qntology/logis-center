//! 🌟 [PYRAMID-QUANT] SigLIP2 가중치 정밀도 계획.
//!
//! 지금 앱은 GPU 에서 BF16(`runtime-bf16/` GGUF), CPU · Vulkan 에서 원본 F32 로 올립니다. 이 모듈은 그 밖의 정밀도를
//! 텐서 묶음 단위로 고르는 계획을 둡니다. 묶음은 지시문의 4계층 표를 따릅니다:
//!
//!   sv.head          비전 MapHead 전체 (in_proj · out_proj · mlp)          st.head        텍스트 투영 헤드
//!   sv.attn.edge     비전 층 0 · 1 · 23 ~ 26 의 q/k/v/out                  st.attn.edge   텍스트 층 0 · 26
//!   sv.mlp.edge      같은 층의 fc1 · fc2                                    st.mlp.edge
//!   sv.attn.mid      비전 층 2 ~ 22 의 q/k/v/out                            st.attn.mid    텍스트 층 1 ~ 25
//!   sv.fc1.mid       같은 층의 fc1 (4304 × 1152)                            st.fc1.mid
//!   sv.fc2.mid       같은 층의 fc2 (1152 × 4304)                            st.fc2.mid
//!
//! 정규화 · 바이어스 · 위치 임베딩 · 패치 투영 · probe · logit_scale/bias · 토큰 표는 어느 계획에서도 원본 F32 입니다(1계층).
//!
//! 블록 형식이 나누지 못하는 입력 폭은 0 으로 채워 양자화합니다. so400m 의 1152 는 Q8_0(32) 로 나뉘지만 K 형식(256)으로는
//! 안 나뉘고, 4304 는 32 로도 안 나뉩니다(4304 = 32 × 134 + 16). 채운 열에 곱해지는 활성은 0 이라 값은 같습니다.
//!
//! `LOGIS_SIGLIP2_QUANT` 로 고릅니다: `bf16` · `f32` · `q8` · `pyramid` · `<기준>;<묶음>=<형식>,…`
//! (묶음 앞부분만 써도 됩니다: `sv` 는 비전 전부, `st.fc1` 은 텍스트 fc1 전부). 형식은 f32 · q8_0 · q6k · q5k · q4k.
//! 정하지 않으면 ROCm 은 `q8`, CUDA 는 `bf16`, CPU · Vulkan 은 언제나 `f32` 입니다.
//!
//!  ── 왜 ROCm 기본이 q8 인가 (RX 6600 실측, SyntheticData/eval/README.md · score_vision_quant.py) ──
//!   영어 무역 서류 88쪽, 원본 F32 기준:      ΔAP          문서번호 회수   쪽 전체 시간   VRAM 최대(11쪽)
//!     bf16 (예전 기본)                      −0.0008       44/45           602 s          2,088 MB
//!     q8 (전부 Q8_0)                        +0.0006       45/45           538 s ¹        1,438 MB
//!     pyramid (지시문 4계층)                −0.0006       43/45           —              —
//!     f32                                    0             45/45           461 s          3,954 MB
//!   ¹ 타워 GGUF 를 처음 한 번 만드는 49 s 제외
//!   q8 은 bf16 보다 원본에 가깝고 빠르며 650 MB 작습니다. 이 GPU 는 BF16 GEMM 이 F32 보다 느리고(텍스트 구 1,230개 134 s vs
//!   69 s), 양자화 행렬곱은 9행 이상이면 블록을 F32 타일로 풀어 F32 GEMM 을 합니다. 4계층은 중간층 fc1 Q4_K 등이 문서번호
//!   크롭 두 개를 놓쳐 기각했습니다(1152 · 4304 폭이라 K 형식은 0 채움도 필요).
//!
//! 양자화는 원본 F32 safetensors 에서 한 번 합니다([`ggml_kquant`], 가중 최소제곱 척도). 앱은 그 블록을 타워 GGUF 로 한 번
//! 써 두고(`gguf_quant.rs`, 지문 = 계획 꼬리표) 그대로 곱합니다. `LOGIS_SIGLIP2_QUANT_SIM=1` 이면 블록을 F32 로 풀어 올리는
//! 측정 경로([`dequantized_tower`])를 씁니다 — 계획끼리 가중치 값만 다르고 연산은 같아 민감도를 연산 차이 없이 잽니다.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::RwLock;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use ggml_kquant::Format;
use once_cell::sync::Lazy;

use crate::model::lang_gguf::StFile;

pub const QUANT_ENV: &str = "LOGIS_SIGLIP2_QUANT";

/// 가중치 하나의 형식
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QType {
    F32,
    Q8_0,
    Q6K,
    Q5K,
    Q4K,
}

impl QType {
    pub const ALL: [QType; 5] = [QType::F32, QType::Q8_0, QType::Q6K, QType::Q5K, QType::Q4K];

    pub fn name(self) -> &'static str {
        match self {
            QType::F32 => "f32",
            QType::Q8_0 => "q8_0",
            QType::Q6K => "q6k",
            QType::Q5K => "q5k",
            QType::Q4K => "q4k",
        }
    }

    pub fn parse(s: &str) -> Option<QType> {
        let s = s.trim().to_ascii_lowercase().replace('_', "");
        QType::ALL.into_iter().find(|q| q.name().replace('_', "") == s)
    }

    pub fn format(self) -> Format {
        match self {
            QType::F32 => Format::F32,
            QType::Q8_0 => Format::Q8_0,
            QType::Q6K => Format::Q6K,
            QType::Q5K => Format::Q5K,
            QType::Q4K => Format::Q4K,
        }
    }

    pub fn ggml(self) -> GgmlDType {
        match self {
            QType::F32 => GgmlDType::F32,
            QType::Q8_0 => GgmlDType::Q8_0,
            QType::Q6K => GgmlDType::Q6K,
            QType::Q5K => GgmlDType::Q5K,
            QType::Q4K => GgmlDType::Q4K,
        }
    }
}

/// 비전 타워에서 양 끝으로 보는 층 (지시문: 첫 2층과 23 ~ 26)
fn vision_edge(layer: usize) -> bool {
    layer < 2 || layer >= 23
}

/// 텍스트 타워에서 양 끝으로 보는 층 (첫 층과 마지막 층)
fn text_edge(layer: usize) -> bool {
    layer == 0 || layer >= 26
}

/// SigLIP2 so400m 가중치 이름의 묶음. 양자화하지 않는 텐서는 `None`.
pub fn group(name: &str) -> Option<&'static str> {
    let vision = name.starts_with("vision_model.");
    if !vision && !name.starts_with("text_model.") {
        return None;
    }
    if !name.ends_with(".weight") && !name.ends_with("in_proj_weight") {
        return None;
    }
    if name.contains(".head.") || name.ends_with(".head.weight") {
        return Some(if vision { "sv.head" } else { "st.head" });
    }
    let layer: usize = name.split(".layers.").nth(1)?.split('.').next()?.parse().ok()?;
    let edge = if vision { vision_edge(layer) } else { text_edge(layer) };
    let part = if name.contains(".self_attn.") {
        "attn"
    } else if name.contains(".mlp.fc1.") {
        "fc1"
    } else if name.contains(".mlp.fc2.") {
        "fc2"
    } else {
        return None;
    };
    Some(match (vision, edge, part) {
        (true, true, "attn") => "sv.attn.edge",
        (true, true, _) => "sv.mlp.edge",
        (true, false, "attn") => "sv.attn.mid",
        (true, false, "fc1") => "sv.fc1.mid",
        (true, false, _) => "sv.fc2.mid",
        (false, true, "attn") => "st.attn.edge",
        (false, true, _) => "st.mlp.edge",
        (false, false, "attn") => "st.attn.mid",
        (false, false, "fc1") => "st.fc1.mid",
        (false, false, _) => "st.fc2.mid",
    })
}

/// 지시문 §2.2 의 SigLIP2 배정
pub const PYRAMID: [(&str, QType); 12] = [
    ("sv.head", QType::Q8_0),
    ("sv.attn.edge", QType::Q8_0),
    ("sv.mlp.edge", QType::Q8_0),
    ("sv.attn.mid", QType::Q6K),
    ("sv.fc2.mid", QType::Q6K),
    ("sv.fc1.mid", QType::Q4K),
    ("st.head", QType::Q8_0),
    ("st.attn.edge", QType::Q8_0),
    ("st.mlp.edge", QType::Q8_0),
    ("st.attn.mid", QType::Q6K),
    ("st.fc2.mid", QType::Q6K),
    ("st.fc1.mid", QType::Q4K),
];

/// 정밀도 계획: 양자화하지 않는 가중치의 dtype 과 묶음별 형식
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightPlan {
    /// 지문 · 로그에 쓰는 이름 (`bf16`, `f32`, `q8`, `pyr`, `custom-<해시>`)
    pub tag: String,
    /// 양자화하지 않는 가중치와 활성의 dtype. 묶음이 하나라도 있으면 F32 입니다.
    pub dense: DType,
    pub groups: BTreeMap<String, QType>,
}

impl WeightPlan {
    /// 지금 앱의 GPU 기본: 타워별 BF16 GGUF
    pub fn bf16() -> WeightPlan {
        WeightPlan { tag: "bf16".into(), dense: DType::BF16, groups: BTreeMap::new() }
    }

    /// 원본 그대로 (기준선)
    pub fn f32() -> WeightPlan {
        WeightPlan { tag: "f32".into(), dense: DType::F32, groups: BTreeMap::new() }
    }

    /// 양자화하는 묶음 전부 Q8_0
    pub fn q8() -> WeightPlan {
        WeightPlan { tag: "q8".into(), dense: DType::F32, groups: PYRAMID.iter().map(|(g, _)| (g.to_string(), QType::Q8_0)).collect() }
    }

    /// 지시문의 4계층
    pub fn pyramid() -> WeightPlan {
        WeightPlan { tag: "pyr".into(), dense: DType::F32, groups: PYRAMID.iter().map(|(g, q)| (g.to_string(), *q)).collect() }
    }

    /// `bf16` · `f32` · `q8` · `pyramid`, 뒤에 `;묶음=형식,…` 을 붙일 수 있습니다.
    pub fn parse(spec: &str) -> Result<WeightPlan> {
        let mut parts = spec.split(';');
        let base = parts.next().unwrap_or("").trim().to_ascii_lowercase();
        let mut plan = match base.as_str() {
            "" | "bf16" | "default" => WeightPlan::bf16(),
            "f32" | "off" | "none" => WeightPlan::f32(),
            "q8" | "q8_0" => WeightPlan::q8(),
            "pyramid" | "pyr" => WeightPlan::pyramid(),
            other => bail!("{QUANT_ENV}: 모르는 계획 {other:?} (bf16, f32, q8, pyramid)"),
        };
        let mut changed = false;
        for part in parts {
            for item in part.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let (g, t) = item.split_once('=').ok_or_else(|| anyhow!("{QUANT_ENV}: {item:?} 는 묶음=형식 이 아닙니다"))?;
                let q = QType::parse(t).ok_or_else(|| anyhow!("{QUANT_ENV}: 모르는 형식 {t:?} (f32, q8_0, q6k, q5k, q4k)"))?;
                let g = g.trim();
                let known: Vec<&str> = PYRAMID.iter().map(|(n, _)| *n).filter(|n| *n == g || n.starts_with(&format!("{g}."))).collect();
                if known.is_empty() {
                    bail!("{QUANT_ENV}: {g:?} 로 시작하는 묶음이 없습니다");
                }
                for n in known {
                    plan.groups.insert(n.to_string(), q);
                }
                changed = true;
            }
        }
        plan.groups.retain(|_, q| *q != QType::F32);
        if !plan.groups.is_empty() {
            plan.dense = DType::F32;
        }
        if changed {
            plan.tag = if plan.groups.is_empty() {
                if plan.dense == DType::BF16 { "bf16".into() } else { "f32".into() }
            } else {
                format!("custom-{:08x}", plan.hash() as u32)
            };
        }
        Ok(plan)
    }

    pub fn is_quantized(&self) -> bool {
        !self.groups.is_empty()
    }

    /// 배정의 안정적인 해시 (지문 · 이름)
    pub fn hash(&self) -> u64 {
        let mut h = fnv(FNV_OFFSET, b"siglip2-weights-v1");
        h = fnv(h, format!("{:?}", self.dense).as_bytes());
        for (g, q) in &self.groups {
            h = fnv(h, g.as_bytes());
            h = fnv(h, q.name().as_bytes());
        }
        h
    }

    /// 가중치 `name` 의 형식
    pub fn qtype(&self, name: &str) -> QType {
        group(name).and_then(|g| self.groups.get(g).copied()).unwrap_or(QType::F32)
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// `LOGIS_SIGLIP2_QUANT` 로 고른 계획. 없으면 장치별 기본([`gpu_default`]).
static ACTIVE: Lazy<RwLock<Option<WeightPlan>>> = Lazy::new(|| {
    let plan = std::env::var(QUANT_ENV).ok().and_then(|v| match WeightPlan::parse(&v) {
        Ok(p) => Some(p),
        Err(e) => {
            println!("[SigLIP2-QUANT] {e} → 장치 기본 계획을 씁니다.");
            None
        }
    });
    RwLock::new(plan)
});

/// GPU 의 기본 계획: ROCm 은 q8 (모듈 설명의 실측), 그 밖(CUDA · Metal)은 재지 않아 예전 그대로 bf16.
fn gpu_default(rocm: bool) -> WeightPlan {
    if rocm {
        WeightPlan::q8()
    } else {
        WeightPlan::bf16()
    }
}

/// 이 빌드의 GPU 에 적용될 계획 (장치를 모를 때: ROCm 빌드면 ROCm 으로 봅니다)
pub fn active() -> WeightPlan {
    let chosen = ACTIVE.read().unwrap_or_else(|e| e.into_inner()).clone();
    chosen.unwrap_or_else(|| gpu_default(cfg!(feature = "rocm") && !cfg!(feature = "cuda")))
}

/// 계획을 바꿉니다 (시험용). `None` 이면 장치 기본으로 돌아갑니다. 이미 올라간 타워는 바뀌지 않습니다.
pub fn set_active(plan: Option<WeightPlan>) {
    *ACTIVE.write().unwrap_or_else(|e| e.into_inner()) = plan;
}

/// 이 장치에서 실제로 쓰는 계획. CPU · Vulkan 은 언제나 원본 F32 입니다(앱의 기존 규칙).
pub fn plan_for(device: &Device) -> WeightPlan {
    if device.is_cpu() || device.is_vulkan() {
        return WeightPlan::f32();
    }
    let chosen = ACTIVE.read().unwrap_or_else(|e| e.into_inner()).clone();
    chosen.unwrap_or_else(|| gpu_default(device.is_rocm()))
}

/// 이 장치에서 SigLIP2 를 올리는 dtype
pub fn dtype_for(device: &Device) -> DType {
    plan_for(device).dense
}

/// 🌟 [WEIGHTS FINGERPRINT] 이 장치의 SigLIP2 벡터가 어떤 가중치 · 정밀도에서 나왔는지 (`bf16`, `f32`, `q8-<해시>` …).
pub fn weights_tag(device: &Device) -> String {
    let p = plan_for(device);
    if p.is_quantized() {
        format!("{}-{:08x}", p.tag, p.hash() as u32)
    } else {
        p.tag
    }
}

/// 예전 키를 쓰는 정밀도인가 (이 변경 전의 장치 기본: GPU BF16 · CPU 와 Vulkan F32). 그 캐시 · SDS 키는 그대로 둡니다.
fn is_legacy_precision(device: &Device) -> bool {
    let p = plan_for(device);
    !p.is_quantized() && p.dense == if device.is_cpu() || device.is_vulkan() { DType::F32 } else { DType::BF16 }
}

/// 마지막으로 올린 SigLIP2 타워의 꼬리표. 예전 정밀도면 `None`.
static LOADED_SUFFIX: Lazy<RwLock<Option<Option<String>>>> = Lazy::new(|| RwLock::new(None));

/// 타워를 올릴 때 부릅니다. 이후 캐시 · SDS 키가 이 정밀도를 따릅니다.
pub fn note_loaded(device: &Device) {
    let suffix = if is_legacy_precision(device) { None } else { Some(weights_tag(device)) };
    *LOADED_SUFFIX.write().unwrap_or_else(|e| e.into_inner()) = Some(suffix);
}

/// 🌟 [WEIGHTS FINGERPRINT] 구 캐시 · 레이아웃 캐시 · SDS 비전 스코프에 붙일 꼬리표.
///
/// 예전 정밀도(GPU BF16 · CPU F32)는 `None` 이라 그 캐시와 원장이 그대로 이어집니다. 다른 계획(ROCm 기본 q8 · F32 강제 ·
/// 양자화)은 계획 이름과 해시를 붙여, 다른 정밀도에서 쌓은 벡터 · 템플릿 · 적응 임계치와 섞이지 않습니다 — q8 로 처음
/// 올리면 구 벡터를 한 번 다시 인코딩하고, 레이아웃 템플릿과 SDS 비전 기준선은 새로 쌓습니다.
/// 아직 아무 타워도 올리지 않았으면 이 빌드의 GPU 에 적용될 계획으로 봅니다.
pub fn fingerprint_suffix() -> Option<String> {
    if let Some(s) = LOADED_SUFFIX.read().unwrap_or_else(|e| e.into_inner()).clone() {
        return s;
    }
    let p = active();
    if !p.is_quantized() && p.dense == DType::BF16 {
        None
    } else if p.is_quantized() {
        Some(format!("{}-{:08x}", p.tag, p.hash() as u32))
    } else {
        Some(p.tag)
    }
}

/// 🌟 [PYRAMID-QUANT] 선형층: 밀집(F32 · BF16) 또는 블록 형식 그대로(장치의 양자화 행렬곱, F32 누산 `forward_exact`).
pub enum SLinear {
    Dense(candle_nn::Linear),
    /// `cols` 는 저장한 입력 폭(0 으로 채운 폭). 활성이 더 좁으면 0 을 채워 곱합니다.
    Quant { w: candle_core::quantized::QMatMul, b: Option<Tensor>, cols: usize },
}

impl candle_core::Module for SLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            SLinear::Dense(l) => l.forward(x),
            SLinear::Quant { w, b, cols } => {
                let k = x.dim(candle_core::D::Minus1)?;
                let padded;
                let x = if k < *cols {
                    padded = x.pad_with_zeros(candle_core::D::Minus1, 0, cols - k)?;
                    &padded
                } else {
                    x
                };
                let y = w.forward_exact(x)?;
                match b {
                    Some(b) => y.broadcast_add(b),
                    None => Ok(y),
                }
            }
        }
    }
}

/// `<vb 접두>.weight` · `.bias` 선형층. 양자화 타워 GGUF(`q`)에 블록 형식으로 있으면 그것을 올리고, F32 로 있으면 저장 형상
/// (0 으로 채운 형상일 수 있음)대로 읽습니다. GGUF 가 없으면 예전과 같은 `candle_nn::linear` 입니다.
pub fn linear(vb: &VarBuilder, q: Option<&super::gguf::MappedGguf>, in_dim: usize, out_dim: usize) -> candle_core::Result<SLinear> {
    if let Some(file) = q {
        let name = if vb.prefix().is_empty() { "weight".to_string() } else { format!("{}.weight", vb.prefix()) };
        if let Some((dt, dims)) = file.info(&name) {
            if let [rows, cols] = dims[..] {
                let b = vb.get(rows, "bias")?;
                if dt.block_size() > 1 {
                    let w = file.qtensor(&name, vb.device()).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    return Ok(SLinear::Quant { w: candle_core::quantized::QMatMul::from_qtensor(w)?, b: Some(b), cols });
                }
                let w = vb.get((rows, cols), "weight")?;
                return Ok(SLinear::Dense(candle_nn::Linear::new(w, Some(b))));
            }
        }
    }
    Ok(SLinear::Dense(candle_nn::linear(in_dim, out_dim, vb.clone())?))
}

/// 한 타워를 계획대로 양자화했다가 F32 로 풀어 올립니다 (§ 모듈 설명: 민감도 측정 경로).
///
/// `prefix` 는 `vision_model.` 또는 `text_model.`, `skip` 의 텐서는 올리지 않습니다(텍스트 토큰 표).
pub fn dequantized_tower(model_dir: &Path, prefix: &str, skip: &[&str], device: &Device, plan: &WeightPlan) -> Result<VarBuilder<'static>> {
    let t0 = Instant::now();
    let st = StFile::open(&model_dir.join("model.safetensors"))?;
    let mut names: Vec<&String> = st
        .entries
        .keys()
        .filter(|n| (n.starts_with(prefix) || n.as_str() == "logit_scale" || n.as_str() == "logit_bias") && !skip.contains(&n.as_str()))
        .collect();
    names.sort();
    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(names.len());
    let mut by_type: BTreeMap<QType, (usize, u64, u64)> = BTreeMap::new();
    let mut err_sum: BTreeMap<QType, (f64, usize)> = BTreeMap::new();
    for name in names {
        let e = &st.entries[name];
        let raw = st.bytes(name)?;
        let values: Vec<f32> = match e.dtype {
            DType::F32 => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            DType::BF16 => raw.chunks_exact(2).map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
            DType::F16 => raw.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
            other => bail!("{name} 의 dtype {other:?} 는 지원하지 않습니다"),
        };
        let q = if e.shape.len() == 2 { plan.qtype(name) } else { QType::F32 };
        let values = if q == QType::F32 {
            values
        } else {
            let (rows, cols) = (e.shape[0], e.shape[1]);
            let back = round_trip(&values, rows, cols, q).with_context(|| format!("{name} 양자화"))?;
            let err = ggml_kquant::rel(&back, &values) as f64;
            let s = err_sum.entry(q).or_default();
            s.0 += err;
            s.1 += 1;
            let padded = cols.next_multiple_of(q.format().block_size());
            let slot = by_type.entry(q).or_default();
            slot.0 += 1;
            slot.1 += (rows * cols * 4) as u64;
            slot.2 += (rows * padded / q.format().block_size() * q.format().type_size()) as u64;
            back
        };
        tensors.insert(name.clone(), Tensor::from_vec(values, e.shape.clone(), device)?);
    }
    let summary: Vec<String> = by_type
        .iter()
        .map(|(q, (n, f, b))| {
            let (es, en) = err_sum.get(q).copied().unwrap_or((0.0, 1));
            format!("{} {}개 {:.0}→{:.0} MB (상대 오차 평균 {:.4})", q.name(), n, *f as f64 / 1e6, *b as f64 / 1e6, es / en.max(1) as f64)
        })
        .collect();
    println!(
        "[SigLIP2-QUANT] {} 계획 '{}' — {} ({:.1} s, 블록을 F32 로 풀어 올림)",
        prefix.trim_end_matches('.'),
        plan.tag,
        if summary.is_empty() { "양자화 없음".to_string() } else { summary.join(" · ") },
        t0.elapsed().as_secs_f64()
    );
    Ok(VarBuilder::from_tensors(tensors, DType::F32, device))
}

/// `rows × cols` 를 `q` 로 양자화했다가 풉니다. 블록이 나누지 못하는 폭은 0 을 채워 양자화하고 채운 열은 버립니다.
pub fn round_trip(values: &[f32], rows: usize, cols: usize, q: QType) -> Result<Vec<f32>> {
    let bs = q.format().block_size();
    let padded = cols.next_multiple_of(bs);
    let src: std::borrow::Cow<[f32]> = if padded == cols {
        std::borrow::Cow::Borrowed(values)
    } else {
        let mut v = vec![0f32; rows * padded];
        for r in 0..rows {
            v[r * padded..r * padded + cols].copy_from_slice(&values[r * cols..(r + 1) * cols]);
        }
        std::borrow::Cow::Owned(v)
    };
    let bytes = ggml_kquant::encode(&src, padded, q.format(), None).map_err(|e| anyhow!(e))?;
    let back = decode(q.ggml(), &bytes, rows, padded)?;
    if padded == cols {
        return Ok(back);
    }
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        out.extend_from_slice(&back[r * padded..r * padded + cols]);
    }
    Ok(out)
}

/// GGML 바이트를 candle 의 역양자화기로 F32 로 풉니다.
pub fn decode(dtype: GgmlDType, bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    // qtensor_from_ggml 은 바이트를 블록 구조체로 읽으므로 정렬된 버퍼에 옮깁니다.
    let mut words = vec![0u64; bytes.len().div_ceil(8)];
    let (_, raw, _) = unsafe { words.align_to_mut::<u8>() };
    raw[..bytes.len()].copy_from_slice(bytes);
    let q = qtensor_from_ggml(dtype, &raw[..bytes.len()], vec![rows, cols], &Device::Cpu)?;
    Ok(q.dequantize(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_follow_the_tier_table() {
        assert_eq!(group("vision_model.encoder.layers.5.mlp.fc1.weight"), Some("sv.fc1.mid"));
        assert_eq!(group("vision_model.encoder.layers.23.self_attn.q_proj.weight"), Some("sv.attn.edge"));
        assert_eq!(group("vision_model.encoder.layers.22.self_attn.q_proj.weight"), Some("sv.attn.mid"));
        assert_eq!(group("vision_model.encoder.layers.1.mlp.fc2.weight"), Some("sv.mlp.edge"));
        assert_eq!(group("vision_model.head.attention.in_proj_weight"), Some("sv.head"));
        assert_eq!(group("vision_model.head.mlp.fc1.weight"), Some("sv.head"));
        assert_eq!(group("text_model.head.weight"), Some("st.head"));
        assert_eq!(group("text_model.encoder.layers.26.mlp.fc1.weight"), Some("st.mlp.edge"));
        assert_eq!(group("text_model.encoder.layers.25.self_attn.out_proj.weight"), Some("st.attn.mid"));
        for never in [
            "vision_model.encoder.layers.5.mlp.fc1.bias",
            "vision_model.encoder.layers.5.layer_norm1.weight",
            "vision_model.embeddings.patch_embedding.weight",
            "vision_model.embeddings.position_embedding.weight",
            "vision_model.head.probe",
            "vision_model.post_layernorm.weight",
            "text_model.embeddings.token_embedding.weight",
            "logit_scale",
        ] {
            assert_eq!(group(never), None, "{never}");
        }
    }

    #[test]
    fn plans_parse_and_tag_apart() {
        assert_eq!(WeightPlan::parse("").unwrap(), WeightPlan::bf16());
        assert_eq!(WeightPlan::parse("f32").unwrap().dense, DType::F32);
        let q8 = WeightPlan::parse("q8").unwrap();
        assert!(q8.groups.values().all(|q| *q == QType::Q8_0) && q8.groups.len() == 12);
        assert_eq!(q8.qtype("vision_model.encoder.layers.5.mlp.fc1.weight"), QType::Q8_0);
        assert_eq!(q8.qtype("vision_model.post_layernorm.weight"), QType::F32);
        let p = WeightPlan::parse("pyramid").unwrap();
        assert_eq!(p.qtype("vision_model.encoder.layers.5.mlp.fc1.weight"), QType::Q4K);
        assert_eq!(p.qtype("text_model.encoder.layers.0.mlp.fc1.weight"), QType::Q8_0);
        let c = WeightPlan::parse("pyramid;sv.fc1=q8_0").unwrap();
        assert_eq!(c.qtype("vision_model.encoder.layers.5.mlp.fc1.weight"), QType::Q8_0);
        assert!(c.tag.starts_with("custom-") && c.tag != p.tag);
        let only_text = WeightPlan::parse("f32;st=q8_0").unwrap();
        assert_eq!(only_text.qtype("vision_model.encoder.layers.5.mlp.fc1.weight"), QType::F32);
        assert_eq!(only_text.dense, DType::F32);
        assert!(WeightPlan::parse("q3").is_err() && WeightPlan::parse("q8;zz=q8_0").is_err() && WeightPlan::parse("q8;sv=q3k").is_err());
        assert_ne!(WeightPlan::q8().hash(), WeightPlan::pyramid().hash());
    }

    #[test]
    fn widths_the_block_does_not_divide_round_trip_through_zero_padding() -> Result<()> {
        // so400m 의 fc2 입력 4304 = 32 × 134 + 16 과 1152 = 256 × 4 + 128 을 줄인 꼴.
        let (rows, cols) = (3, 300);
        let v: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.731).sin() * 0.02).collect();
        for q in [QType::Q8_0, QType::Q6K, QType::Q5K, QType::Q4K] {
            let back = round_trip(&v, rows, cols, q)?;
            assert_eq!(back.len(), v.len(), "{q:?}");
            let err = ggml_kquant::rel(&back, &v);
            assert!(err < 0.12, "{q:?}: {err}");
        }
        Ok(())
    }
}

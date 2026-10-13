//! Qwen4-Exp (transformers `qwen4_exp` / `qwen4_exp_text`) 텍스트 설정.
//!
//! 기준: transformers 5.17 `configuration_qwen4_exp.py` (`Qwen4ExpTextConfig` 기본값 · `__post_init__` · `validate_architecture`).
//! 모든 필드는 config.json 에 없을 수 있으므로 기본값을 갖습니다. 주의할 함정(qwen4-spec Q4-08):
//!  - RoPE 기본값은 Qwen3.5 와 달리 `partial_rotary_factor = 1.0`, `rope_theta = 10000`
//!    (`rope_parameters` → 최상위 레거시 키 → 기본값 순으로 읽습니다).
//!  - 실제 체크포인트의 `layer_types` 는 `"full_attention"` 을 쓰지만 런타임에서는 QSA(`qwen_sparse_attention`) 로 바뀝니다.
//!  - `ple_layer_ids` 는 1-기반입니다. PLE 의 eos 는 토크나이저가 아니라 config 의 `eos_token_id`(리스트면 첫 원소) 입니다.

use anyhow::{anyhow, bail, Result};
use serde::Deserialize;
use serde_json::Value;

/// `eos_token_id` 는 정수 또는 정수 리스트입니다.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum EosId {
    One(u32),
    Many(Vec<u32>),
}

impl EosId {
    pub fn first(&self) -> Option<u32> {
        match self {
            EosId::One(v) => Some(*v),
            EosId::Many(v) => v.first().copied(),
        }
    }

    pub fn all(&self) -> Vec<u32> {
        match self {
            EosId::One(v) => vec![*v],
            EosId::Many(v) => v.clone(),
        }
    }
}

/// 디코더 층 종류. `"full_attention"` 은 런타임에서 QSA 로 재매핑됩니다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet (linear attention)
    Linear,
    /// Qwen Sparse Attention (gated full attention + 블록 압축 인덱서)
    Qsa,
}

/// GDN gated RMSNorm 활성 함수 (`output_gate_type or hidden_act`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateAct {
    Silu,
    Sigmoid,
}

fn d_vocab() -> usize { 248_320 }
fn d_hidden() -> usize { 2048 }
fn d_layers() -> usize { 40 }
fn d_heads() -> usize { 16 }
fn d_kv_heads() -> usize { 2 }
fn d_head_dim() -> Option<usize> { Some(256) }
fn d_act() -> String { "silu".to_string() }
fn d_max_pos() -> usize { 32_768 }
fn d_eps() -> f64 { 1e-6 }
fn d_conv_kernel() -> usize { 4 }
fn d_lin_dim() -> usize { 128 }
fn d_lin_k_heads() -> usize { 16 }
fn d_lin_v_heads() -> usize { 32 }
fn d_moe_inter() -> usize { 512 }
fn d_experts_per_tok() -> usize { 10 }
fn d_experts() -> usize { 512 }
fn d_true() -> bool { true }
fn d_hc_count() -> usize { 4 }
fn d_hc_lowrank() -> usize { 320 }
fn d_ple_kernel() -> usize { 4 }
fn d_ngram() -> usize { 3 }
fn d_heads_per_ngram() -> usize { 8 }
fn d_ngram_base() -> u64 { 20_000_000 }
fn d_ngram_div() -> u64 { 128 }
fn d_seed() -> u64 { 1234 }
fn d_split() -> usize { 512 }

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen4ExpTextConfig {
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    #[serde(default = "d_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_kv_heads")]
    pub num_key_value_heads: usize,
    /// 기본 256. 명시적 null 이면 hidden_size / num_attention_heads (HF `getattr(..) or` 와 같음).
    #[serde(default = "d_head_dim")]
    pub head_dim: Option<usize>,
    #[serde(default = "d_act")]
    pub hidden_act: String,
    #[serde(default = "d_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_parameters: Option<Value>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub full_attention_interval: Option<usize>,
    #[serde(default = "d_conv_kernel")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "d_lin_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "d_lin_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "d_lin_k_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "d_lin_v_heads")]
    pub linear_num_value_heads: usize,
    #[serde(default = "d_moe_inter")]
    pub moe_intermediate_size: usize,
    #[serde(default = "d_moe_inter")]
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "d_experts")]
    pub num_experts: usize,
    #[serde(default = "d_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "d_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "d_hc_count")]
    pub hc_count: usize,
    #[serde(default = "d_hc_lowrank")]
    pub hc_lowrank: usize,
    #[serde(default)]
    pub ple_layer_ids: Option<Vec<usize>>,
    #[serde(default)]
    pub ple_embed_dim: Option<usize>,
    #[serde(default = "d_ple_kernel")]
    pub ple_conv_kernel_size: usize,
    #[serde(default = "d_ngram")]
    pub ngram_size: usize,
    #[serde(default = "d_heads_per_ngram")]
    pub heads_per_ngram: usize,
    #[serde(default = "d_ngram_base")]
    pub ngram_vocab_size_base: u64,
    #[serde(default = "d_ngram_div")]
    pub make_ngram_vocab_size_divisible_by: u64,
    #[serde(default = "d_seed")]
    pub seed: u64,
    #[serde(default = "d_split")]
    pub split_ngram_parts: usize,
    #[serde(default)]
    pub indexer_n_heads: Option<usize>,
    #[serde(default)]
    pub indexer_kv_heads: Option<usize>,
    #[serde(default)]
    pub indexer_head_dim: Option<usize>,
    #[serde(default)]
    pub indexer_budget: Option<usize>,
    #[serde(default)]
    pub indexer_compress_ratio: Option<usize>,
    #[serde(default)]
    pub output_gate_type: Option<String>,
    #[serde(default)]
    pub eos_token_id: Option<EosId>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    /// `"bfloat16"` / `"float32"` … (구버전 키 `torch_dtype` 도 허용)
    #[serde(default, alias = "torch_dtype")]
    pub dtype: Option<String>,
    #[serde(default)]
    pub model_type: Option<String>,
}

impl Qwen4ExpTextConfig {
    /// config.json 값(최상위 `{model_type:"qwen4_exp", text_config:{..}}` 래퍼 또는 텍스트 설정 자체)을 읽어
    /// `finalize()` + `validate()` 까지 마친 설정을 돌려줍니다.
    pub fn from_value(v: &Value) -> Result<Self> {
        let top_tie = v.get("tie_word_embeddings").and_then(|x| x.as_bool());
        let text = match v.get("text_config") {
            Some(t) if t.is_object() => t.clone(),
            _ => v.clone(),
        };
        let mut cfg: Qwen4ExpTextConfig = serde_json::from_value(text)
            .map_err(|e| anyhow!("[QWEN4-EXP] config.json 해석 실패: {e}"))?;
        if cfg.tie_word_embeddings.is_none() {
            cfg.tie_word_embeddings = top_tie;
        }
        cfg.finalize();
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let raw = std::fs::read(path).map_err(|e| anyhow!("[QWEN4-EXP] {:?} 를 읽을 수 없습니다: {e}", path))?;
        let v: Value = serde_json::from_slice(&raw)?;
        Self::from_value(&v)
    }

    /// HF `__post_init__` 재현: ple_layer_ids 정렬·중복 제거, ple_embed_dim 기본값, layer_types 기본/재매핑.
    pub fn finalize(&mut self) {
        let mut ids = self.ple_layer_ids.take().unwrap_or_default();
        ids.sort_unstable();
        ids.dedup();
        self.ple_layer_ids = Some(ids);
        if self.ple_embed_dim.is_none() {
            self.ple_embed_dim = Some(self.hidden_size);
        }
        if self.head_dim.is_none() {
            self.head_dim = Some(self.hidden_size / self.num_attention_heads.max(1));
        }
        let types = match self.layer_types.take() {
            None => {
                let interval = self.full_attention_interval.unwrap_or(4).max(1);
                (0..self.num_hidden_layers)
                    .map(|i| {
                        if (i + 1) % interval != 0 {
                            "linear_attention".to_string()
                        } else {
                            "qwen_sparse_attention".to_string()
                        }
                    })
                    .collect()
            }
            Some(v) => v
                .into_iter()
                .map(|t| if t == "full_attention" { "qwen_sparse_attention".to_string() } else { t })
                .collect(),
        };
        self.layer_types = Some(types);
    }

    /// HF `validate_architecture` 재현 (+ 포트가 가정하는 정수 관계 검사).
    pub fn validate(&self) -> Result<()> {
        let types = self.layer_types();
        if types.len() < self.num_hidden_layers {
            bail!(
                "[QWEN4-EXP] layer_types 길이 {} < num_hidden_layers {}",
                types.len(),
                self.num_hidden_layers
            );
        }
        for t in types {
            if t != "linear_attention" && t != "qwen_sparse_attention" {
                bail!("[QWEN4-EXP] 지원하지 않는 layer type: {t}");
            }
        }
        self.gate_act()?;
        if self.hc_count <= 1 {
            bail!("[QWEN4-EXP] hc_count > 1 이어야 합니다 (현재 {})", self.hc_count);
        }
        if self.num_experts == 0 || self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
            bail!(
                "[QWEN4-EXP] num_experts_per_tok({}) 는 [1, num_experts({})] 범위여야 합니다",
                self.num_experts_per_tok,
                self.num_experts
            );
        }
        if self.moe_intermediate_size == 0 || self.shared_expert_intermediate_size == 0 {
            bail!("[QWEN4-EXP] moe_intermediate_size / shared_expert_intermediate_size 는 0 보다 커야 합니다");
        }
        if self.linear_num_key_heads == 0 || self.linear_num_value_heads % self.linear_num_key_heads != 0 {
            bail!(
                "[QWEN4-EXP] linear_num_value_heads({}) 는 linear_num_key_heads({}) 의 배수여야 합니다",
                self.linear_num_value_heads,
                self.linear_num_key_heads
            );
        }
        if self.num_key_value_heads == 0 || self.num_attention_heads % self.num_key_value_heads != 0 {
            bail!("[QWEN4-EXP] num_attention_heads 는 num_key_value_heads 의 배수여야 합니다");
        }
        if self.rotary_dim() % 2 != 0 || self.rotary_dim() > self.head_dim() {
            bail!("[QWEN4-EXP] rotary_dim {} 은 짝수이고 head_dim {} 이하여야 합니다", self.rotary_dim(), self.head_dim());
        }
        // QSA: 모든 어텐션 층이 인덱서를 만들므로 사실상 필수입니다.
        let has_qsa = types.iter().take(self.num_hidden_layers).any(|t| t == "qwen_sparse_attention");
        let q = [
            self.indexer_n_heads,
            self.indexer_kv_heads,
            self.indexer_head_dim,
            self.indexer_budget,
            self.indexer_compress_ratio,
        ];
        if q.iter().any(|v| v.is_some()) || has_qsa {
            if q.iter().any(|v| v.is_none()) {
                bail!("[QWEN4-EXP] QSA 설정(indexer_n_heads/kv_heads/head_dim/budget/compress_ratio) 이 빠졌습니다");
            }
            if q.iter().any(|v| v == &Some(0)) {
                bail!("[QWEN4-EXP] QSA 설정 값은 양수여야 합니다");
            }
            if self.indexer_kv_heads != Some(1) {
                bail!("[QWEN4-EXP] QSA 는 indexer_kv_heads=1 만 지원합니다");
            }
            if self.indexer_budget.unwrap_or(0) % self.indexer_compress_ratio.unwrap_or(1) != 0 {
                bail!("[QWEN4-EXP] indexer_budget 은 indexer_compress_ratio 의 배수여야 합니다");
            }
            if self.rotary_dim() > self.indexer_head_dim.unwrap_or(0) {
                bail!(
                    "[QWEN4-EXP] rotary_dim {} > indexer_head_dim {:?}",
                    self.rotary_dim(),
                    self.indexer_head_dim
                );
            }
        }
        let ple = self.ple_ids();
        if !ple.is_empty() {
            let g = self.ngram_heads();
            let e = self.ple_dim();
            if g == 0 || e == 0 || e % g != 0 {
                bail!("[QWEN4-EXP] ple_embed_dim({e}) 은 n-gram head 수({g}) 의 배수여야 합니다");
            }
            for &id in ple {
                if id < 1 || id > self.num_hidden_layers {
                    bail!("[QWEN4-EXP] ple_layer_ids 는 1-기반 [1, {}] 이어야 합니다 (받은 값 {id})", self.num_hidden_layers);
                }
                if types[id - 1] != "linear_attention" {
                    bail!("[QWEN4-EXP] PLE 는 linear_attention 층에만 붙을 수 있습니다 (층 {id})");
                }
            }
            if self.eos_token_id.as_ref().and_then(|e| e.first()).is_none() {
                bail!("[QWEN4-EXP] PLE 사용 시 eos_token_id 가 필요합니다");
            }
        }
        Ok(())
    }

    pub fn layer_types(&self) -> &[String] {
        self.layer_types.as_deref().unwrap_or(&[])
    }

    pub fn layer_kind(&self, i: usize) -> LayerKind {
        match self.layer_types().get(i).map(|s| s.as_str()) {
            Some("linear_attention") => LayerKind::Linear,
            _ => LayerKind::Qsa,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads.max(1))
    }

    fn rope_param(&self, key: &str) -> Option<f64> {
        self.rope_parameters.as_ref().and_then(|v| v.get(key)).and_then(|v| v.as_f64())
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_param("rope_theta").or(self.rope_theta).unwrap_or(10_000.0)
    }

    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_param("partial_rotary_factor")
            .or(self.partial_rotary_factor)
            .unwrap_or(1.0)
    }

    /// `int(head_dim * partial_rotary_factor)` (HF 와 같은 내림)
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim() as f64 * self.partial_rotary_factor()) as usize
    }

    pub fn rope_type(&self) -> String {
        self.rope_parameters
            .as_ref()
            .and_then(|v| v.get("rope_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string()
    }

    pub fn gate_act(&self) -> Result<GateAct> {
        let a = self.output_gate_type.clone().unwrap_or_else(|| self.hidden_act.clone());
        match a.as_str() {
            "silu" | "swish" => Ok(GateAct::Silu),
            "sigmoid" => Ok(GateAct::Sigmoid),
            other => bail!("[QWEN4-EXP] 지원하지 않는 output gate 활성 함수: {other}"),
        }
    }

    pub fn ple_ids(&self) -> &[usize] {
        self.ple_layer_ids.as_deref().unwrap_or(&[])
    }

    /// 디코더 층 i(0-기반)의 PLE 인덱스 (ple_layer_ids 안의 위치)
    pub fn ple_index(&self, i: usize) -> Option<usize> {
        self.ple_ids().iter().position(|&id| id == i + 1)
    }

    pub fn ple_dim(&self) -> usize {
        self.ple_embed_dim.unwrap_or(self.hidden_size)
    }

    pub fn ngram_heads(&self) -> usize {
        self.ngram_size.saturating_sub(1) * self.heads_per_ngram
    }

    /// PLE n-gram 경계 토큰 (config.eos_token_id, 리스트면 첫 원소)
    pub fn ple_eos(&self) -> u32 {
        self.eos_token_id.as_ref().and_then(|e| e.first()).unwrap_or(0)
    }

    pub fn eos_ids(&self) -> Vec<u32> {
        self.eos_token_id.as_ref().map(|e| e.all()).unwrap_or_default()
    }

    pub fn tie_embeddings(&self) -> bool {
        self.tie_word_embeddings.unwrap_or(false)
    }
}

//! Qwen4-Exp 텍스트 백본 + LM head (qwen4-spec §2-§3, §10).
//!
//! ```text
//! x0 = repeat(embed(ids), hc)                  # [B, T, hc*H], 스트림-우선
//! for layer:
//!     x += PLE(x, ids)                          # PLE 층만
//!     (m, inj) = attn_hyper_connection(x);  x = x + inj ⊗ (GDN | QSA)(m)
//!     (m, inj) = mlp_hyper_connection(x);   x = x + inj ⊗ MoE(m)
//! h = hyper_connection_mixer(x)                 # inject 없는 GatedResidual → [B, T, H]
//! logits = lm_head(h)                           # model.norm / input_layernorm 없음
//! ```
//! 증분 디코드 상태: QSA 층 = K/V + 인덱서 원시/압축 키, linear 층 = conv + recurrent 상태,
//! PLE 층 = 팽창 conv 상태 + n-gram 문맥. 위치 카운터 = 지금까지 넣은 토큰 수 (`seqlen_offset`).
//! 프리필을 여러 청크로 나눠도 모든 상태가 이어지므로 한 번에 넣은 것과 같습니다.
//! MTP(`mtp.*`) 와 비전(`model.visual.*`) 가중치는 읽지 않습니다 (HF 도 텍스트 추론에서 무시).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

use super::config::{LayerKind, Qwen4ExpTextConfig};
use super::gdn::GatedDeltaNet;
use super::hyper::GatedResidual;
use super::layers::RopeTable;
use super::moe::SparseMoe;
use super::ple::PleLayer;
use super::qsa::QsaAttention;

pub enum Mixer {
    Linear(GatedDeltaNet),
    Qsa(QsaAttention),
}

pub struct DecoderLayer {
    pub mixer: Mixer,
    pub mlp: SparseMoe,
    pub ple: Option<PleLayer>,
    hc_attn: GatedResidual,
    hc_mlp: GatedResidual,
}

impl DecoderLayer {
    fn new(
        vb: VarBuilder,
        c: &Qwen4ExpTextConfig,
        idx: usize,
        st: Option<&Arc<MmapedSafetensors>>,
        prefix: &str,
        opts: &LoadOptions,
    ) -> Result<Self> {
        let mixer = match c.layer_kind(idx) {
            LayerKind::Linear => Mixer::Linear(GatedDeltaNet::new(vb.pp("linear_attn"), c)?),
            LayerKind::Qsa => Mixer::Qsa(QsaAttention::new(vb.pp("self_attn"), c)?),
        };
        let mlp = SparseMoe::new(vb.pp("mlp"), c)?;
        let ple = match c.ple_index(idx) {
            Some(p) => Some(PleLayer::new(vb.pp("ple"), c, p, st, &format!("{prefix}.ple"), opts.ple_dense_limit_bytes)?),
            None => None,
        };
        let (hc, h, r, eps) = (c.hc_count, c.hidden_size, c.hc_lowrank, c.rms_norm_eps);
        Ok(Self {
            mixer,
            mlp,
            ple,
            hc_attn: GatedResidual::new(vb.pp("attn_hyper_connection"), hc, h, r, eps, true)?,
            hc_mlp: GatedResidual::new(vb.pp("mlp_hyper_connection"), hc, h, r, eps, true)?,
        })
    }

    fn forward(&mut self, x: &Tensor, ids: &[Vec<u32>], rope: &RopeTable, offset: usize) -> Result<Tensor> {
        let x = match self.ple.as_mut() {
            Some(p) => (x + p.forward(x, ids)?)?,
            None => x.clone(),
        };
        let (m, inj) = self.hc_attn.forward(&x)?;
        let y = match &mut self.mixer {
            Mixer::Linear(g) => g.forward(&m)?,
            Mixer::Qsa(a) => a.forward(&m, rope, offset)?,
        };
        let x = GatedResidual::inject(&x, &y, inj.as_ref().expect("inject"))?;
        let (m, inj) = self.hc_mlp.forward(&x)?;
        let y = self.mlp.forward(&m)?;
        GatedResidual::inject(&x, &y, inj.as_ref().expect("inject"))
    }

    fn clear_cache(&mut self) {
        match &mut self.mixer {
            Mixer::Linear(g) => g.clear_cache(),
            Mixer::Qsa(a) => a.clear_cache(),
        }
        if let Some(p) = self.ple.as_mut() {
            p.clear_cache();
        }
    }
}

/// `Qwen4ExpTextModel` (+ `lm_head`).
pub struct Qwen4ExpForCausalLM {
    pub cfg: Qwen4ExpTextConfig,
    embed_tokens: Embedding,
    pub layers: Vec<DecoderLayer>,
    mixer: GatedResidual,
    lm_head: Linear,
    rope: RopeTable,
    dtype: DType,
    device: Device,
    seqlen: usize,
}

/// 체크포인트 안의 safetensors 파일 목록 (단일 파일 또는 `model.safetensors.index.json` 샤드).
pub fn safetensors_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&index)?)?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| anyhow!("[QWEN4-EXP] {:?} 에 weight_map 이 없습니다", index))?;
        let mut files: Vec<String> = map.values().filter_map(|x| x.as_str().map(|s| s.to_string())).collect();
        files.sort();
        files.dedup();
        return Ok(files.into_iter().map(|f| dir.join(f)).collect());
    }
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    files.sort();
    ensure!(!files.is_empty(), "[QWEN4-EXP] {:?} 에 safetensors 가중치가 없습니다", dir);
    Ok(files)
}

/// 로드 옵션. 기본값은 환경 변수(`LOGIS_QWEN4_PLE_DENSE_MB`, `LOGIS_QWEN4_MAX_RESIDENT_GB`) 를 읽습니다.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// 원본 크기가 이 바이트를 넘는 PLE n-gram 표는 장치에 올리지 않고 mmap 에서 필요한 행만 모읍니다 (기본 512 MB).
    /// 0 이면 모든 표가 mmap 행 수집 경로를 탑니다 (테스트로 그 경로를 강제할 때).
    pub ple_dense_limit_bytes: usize,
    /// 장치에 통째로 올릴 가중치 추정치 상한 (기본 6 GB). 넘으면 OOM 전에 명확한 오류로 거부합니다.
    pub max_resident_bytes: usize,
}

impl LoadOptions {
    pub fn from_env() -> Self {
        let gb = std::env::var("LOGIS_QWEN4_MAX_RESIDENT_GB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(6.0);
        Self { ple_dense_limit_bytes: super::ple::dense_table_limit_from_env(), max_resident_bytes: (gb * 1e9) as usize }
    }
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self::from_env()
    }
}

/// 장치에 통째로 올릴 가중치 바이트 추정 (MTP/비전 제외, `ple_dense_limit` 을 넘어 mmap 행 수집으로 돌릴 큰 n-gram 표 제외).
pub fn resident_weight_bytes(st: &MmapedSafetensors, dtype: DType, ple_dense_limit: usize) -> usize {
    let mut table = 0usize;
    let mut table_dst = 0usize;
    let mut rest = 0usize;
    for (name, view) in st.tensors() {
        if name.starts_with("mtp.") || name.contains("visual.") {
            continue;
        }
        let elems: usize = view.shape().iter().product();
        let src = match DType::try_from(view.dtype()) {
            Ok(d) if d.is_float() => d,
            _ => continue, // int64 PLE 버퍼 등
        };
        if name.contains("ngram_embedding") {
            // 통째로 올릴지는 원본 크기로 정하고 (ple.rs 와 같은 기준), 올리면 모델 dtype 으로 변환됩니다.
            table += elems * src.size_in_bytes();
            table_dst += elems * dtype.size_in_bytes();
        } else {
            rest += elems * dtype.size_in_bytes();
        }
    }
    // ple.rs 와 같은 기준: 층별 표 크기가 아니라 합계로 보수적으로 판정합니다 (표가 여럿이면 과대 추정 쪽).
    if table <= ple_dense_limit {
        rest += table_dst;
    }
    rest
}

impl Qwen4ExpForCausalLM {
    /// HF 디렉터리(config.json + safetensors) 를 mmap VarBuilder 로 읽습니다 (옵션은 환경 변수 기본값).
    pub fn load(dir: &Path, dtype: DType, device: &Device) -> Result<Self> {
        Self::load_with(dir, dtype, device, &LoadOptions::from_env())
    }

    pub fn load_with(dir: &Path, dtype: DType, device: &Device, opts: &LoadOptions) -> Result<Self> {
        let cfg = Qwen4ExpTextConfig::from_file(&dir.join("config.json"))?;
        let files = safetensors_files(dir)?;
        let st = Arc::new(unsafe { MmapedSafetensors::multi(&files)? });
        let need = resident_weight_bytes(&st, dtype, opts.ple_dense_limit_bytes);
        let cap = opts.max_resident_bytes;
        if need > cap {
            bail!(
                "[QWEN4-EXP] 상주 가중치 추정 {:.1} GB > 한도 {:.1} GB (LOGIS_QWEN4_MAX_RESIDENT_GB). \
                 이 포트는 아직 전문가를 통째로 올립니다 — 전체 크기 Qwen4-Exp 는 선택 전문가 스트리밍(미구현)이 필요합니다.",
                need as f64 / 1e9,
                cap as f64 / 1e9
            );
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? };
        Self::new_with(vb, cfg, Some(st), opts)
    }

    /// `vb` 는 체크포인트 루트. 텍스트 전용(`model.*`) 과 VLM(`model.language_model.*`) 접두사를 모두 받습니다.
    pub fn new(vb: VarBuilder, cfg: Qwen4ExpTextConfig, st: Option<Arc<MmapedSafetensors>>) -> Result<Self> {
        Self::new_with(vb, cfg, st, &LoadOptions::from_env())
    }

    pub fn new_with(
        vb: VarBuilder,
        cfg: Qwen4ExpTextConfig,
        st: Option<Arc<MmapedSafetensors>>,
        opts: &LoadOptions,
    ) -> Result<Self> {
        let prefix = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            "model.language_model"
        } else if vb.contains_tensor("model.embed_tokens.weight") {
            "model"
        } else {
            bail!("[QWEN4-EXP] embed_tokens 가중치를 찾을 수 없습니다 (model.* / model.language_model.*)");
        };
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let vm = vb.pp(prefix);
        let (h, v) = (cfg.hidden_size, cfg.vocab_size);
        let emb_w = vm.get((v, h), "embed_tokens.weight")?;
        let embed_tokens = Embedding::new(emb_w.clone(), h);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("{prefix}.layers.{i}");
            layers.push(DecoderLayer::new(vm.pp(format!("layers.{i}")), &cfg, i, st.as_ref(), &lp, opts)?);
        }
        let mixer = GatedResidual::new(
            vm.pp("hyper_connection_mixer"),
            cfg.hc_count,
            h,
            cfg.hc_lowrank,
            cfg.rms_norm_eps,
            false,
        )?;
        // 묶인(tied) 경우에만 embed_tokens 를 LM head 로 씁니다. 묶이지 않았는데 lm_head.weight 가 없으면
        // 잘린/접두사가 틀린 체크포인트이므로 엉뚱한 출력 투영으로 생성하지 않고 거부합니다.
        let lm_head = if cfg.tie_embeddings() {
            Linear::new(emb_w, None)
        } else if vb.contains_tensor("lm_head.weight") {
            Linear::new(vb.get((v, h), "lm_head.weight")?, None)
        } else {
            bail!("[QWEN4-EXP] lm_head.weight 없음 (tie_word_embeddings=false). 체크포인트가 잘렸거나 접두사가 다릅니다.");
        };
        let rope = RopeTable::new(cfg.rope_theta(), cfg.rotary_dim(), dtype, &device);
        println!(
            "[QWEN4-EXP] 📦 로드 완료 | 층 {} (linear {} / QSA {}) · PLE {:?} · 전문가 {} top-{} · hc {} · {:?} on {:?}",
            cfg.num_hidden_layers,
            (0..cfg.num_hidden_layers).filter(|&i| cfg.layer_kind(i) == LayerKind::Linear).count(),
            (0..cfg.num_hidden_layers).filter(|&i| cfg.layer_kind(i) == LayerKind::Qsa).count(),
            cfg.ple_ids(),
            cfg.num_experts,
            cfg.num_experts_per_tok,
            cfg.hc_count,
            dtype,
            device.location()
        );
        Ok(Self { cfg, embed_tokens, layers, mixer, lm_head, rope, dtype, device, seqlen: 0 })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 지금까지 캐시에 들어간 토큰 수.
    pub fn seqlen(&self) -> usize {
        self.seqlen
    }

    /// 백본: `ids: [B, T]` (u32) → 최종 mixer 출력 `[B, T, H]`. `seqlen_offset` 는 캐시 길이와 같아야 합니다.
    pub fn forward_hidden(&mut self, ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward_hidden_traced(ids, seqlen_offset, None)
    }

    /// 층별 출력(`[B, T, hc*H]`) 을 `trace` 에 모으는 디버그용 forward.
    pub fn forward_hidden_traced(
        &mut self,
        ids: &Tensor,
        seqlen_offset: usize,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let (_b, t) = ids.dims2()?;
        ensure!(
            seqlen_offset == self.seqlen,
            "[QWEN4-EXP] seqlen_offset {seqlen_offset} != 캐시 길이 {} (clear_cache 후 0 부터 넣어야 합니다)",
            self.seqlen
        );
        let ids_host: Vec<Vec<u32>> = ids.to_device(&Device::Cpu)?.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let ids_dev = ids.to_dtype(DType::U32)?.to_device(&self.device)?;
        self.rope.ensure(seqlen_offset + t)?;
        let e = self.embed_tokens.forward(&ids_dev)?; // [B, T, H]
        let mut x = e.repeat((1, 1, self.cfg.hc_count))?;
        for layer in self.layers.iter_mut() {
            x = layer.forward(&x, &ids_host, &self.rope, seqlen_offset)?;
            if let Some(tr) = trace.as_deref_mut() {
                tr.push(x.clone());
            }
        }
        let (h, _) = self.mixer.forward(&x)?;
        self.seqlen += t;
        Ok(h)
    }

    /// 전체 위치의 logits `[B, T, V]`.
    pub fn forward(&mut self, ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let h = self.forward_hidden(ids, seqlen_offset)?;
        Ok(self.lm_head.forward(&h)?)
    }

    /// mixer 출력 `[.., H]` → logits `[.., V]`.
    pub fn lm_head_forward(&self, h: &Tensor) -> Result<Tensor> {
        Ok(self.lm_head.forward(h)?)
    }

    /// 마지막 위치의 logits `[B, V]` (생성용).
    pub fn forward_last(&mut self, ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let h = self.forward_hidden(ids, seqlen_offset)?;
        let t = h.dim(1)?;
        let last = h.narrow(1, t - 1, 1)?.squeeze(1)?;
        Ok(self.lm_head.forward(&last)?)
    }

    pub fn clear_cache(&mut self) {
        for l in self.layers.iter_mut() {
            l.clear_cache();
        }
        self.seqlen = 0;
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.cfg.max_position_embeddings
    }
}

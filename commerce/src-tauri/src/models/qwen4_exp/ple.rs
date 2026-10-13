//! Per-Layer Embedding (PLE): 해시 n-gram 임베딩 + 게이트 + 팽창(dilated) depthwise conv (qwen4-spec §8 / Q4-04).
//!
//! 해시는 정수 연산이라 한 비트라도 다르면 완전히 다른 행을 읽습니다. HF 와 똑같이:
//!  - head g (PLE 인덱스 p) 의 크기 `P[g]` = `base-1` 보다 큰 `(p·G + g + 1)` 번째 소수 (trial division),
//!    `OFF` = 누적합 (PLE 층마다 0 부터), 표 행 수 = `ceil(ΣP / div) · div`
//!  - 승수 `m_i = 2·(splitmix64(seed + 10007·p + GAMMA·(i+1)) % half) + 1`, `half = max(1, ((2^63-1)/vocab)/2)`
//!  - 이력 `[N-1 개 문맥 ; 토큰]` (처음 문맥은 eos), `s_k` = `[t-k, t-1]` 안에 eos 가 있거나 범위 밖이면 eos, 아니면 `tok[t-k]`
//!  - n = 2..N: `mix = XOR_{j<n} s_j·m_j` (u64, 오버플로 없음), `id = mix % P + OFF` (2-gram head 먼저)
//! 체크포인트의 `layer_multipliers / ngram_heads_vocab_sizes / ngram_heads_offsets` 버퍼가 있으면 로드 시 대조합니다.
//!
//! PLE 층: `key = GRMS_k(key_proj(emb))`, `val = value_proj(emb)`, `qry = GRMS_q(x)`,
//! `gate = Σ_H(key·qry)/sqrt(H)` → `sign·sqrt(max(|gate|, 1e-6))`, `gv = sigmoid(gate)·val` (스트림별),
//! `out = flat(gv) + silu(dwconv_dil(GRMS_c(flat(gv)), dil=N, causal pad (Kp-1)·N))`.
//! 상태: conv 입력 마지막 `(Kp-1)·N` 열 (처음엔 0), n-gram 문맥 마지막 `N-1` 개 id (처음엔 eos).

use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use super::config::Qwen4ExpTextConfig;
use super::layers::{linear_no_bias, sigmoid, RmsNorm1p};

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const M1: u64 = 0xBF58_476D_1CE4_E5B9;
const M2: u64 = 0x94D0_49BB_1331_11EB;
const PRIME_1: u64 = 10_007;

/// 이 크기(바이트)를 넘는 n-gram 표는 통째로 올리지 않고 mmap 에서 행 단위로 모읍니다.
pub(crate) fn dense_table_limit() -> usize {
    std::env::var("LOGIS_QWEN4_PLE_DENSE_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512)
        * 1024
        * 1024
}

pub fn splitmix64(v: u64) -> u64 {
    let mut v = v.wrapping_add(GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(M1);
    v = (v ^ (v >> 27)).wrapping_mul(M2);
    v ^ (v >> 31)
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v % 2 == 0 {
        return v == 2;
    }
    let mut d = 3u64;
    while d * d <= v {
        if v % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// n-gram 해시기 (PLE 층 하나).
#[derive(Debug, Clone)]
pub struct NgramHasher {
    pub multipliers: Vec<u64>,
    pub head_sizes: Vec<u64>,
    pub head_offsets: Vec<u64>,
    pub padded_rows: usize,
    pub eos: u32,
    pub n: usize,
    pub heads_per_ngram: usize,
}

impl NgramHasher {
    #[allow(clippy::too_many_arguments)]
    pub fn new(vocab: usize, n: usize, hpn: usize, base: u64, divisor: u64, seed: u64, ple_idx: usize, eos: u32) -> Self {
        let g = (n - 1) * hpn;
        // (ple_idx·G + 1) .. (ple_idx·G + G) 번째 소수 (base-1 초과)
        let mut head_sizes = Vec::with_capacity(g);
        let mut prime = base.saturating_sub(1);
        let skip = ple_idx * g;
        for i in 0..skip + g {
            prime += 1;
            while !is_prime(prime) {
                prime += 1;
            }
            if i >= skip {
                head_sizes.push(prime);
            }
        }
        let mut head_offsets = Vec::with_capacity(g);
        let mut total = 0u64;
        for &p in &head_sizes {
            head_offsets.push(total);
            total += p;
        }
        let div = divisor.max(1);
        let padded_rows = (total.div_ceil(div) * div) as usize;

        let max_long = i64::MAX as u64;
        let half = ((max_long / (vocab.max(1) as u64)) / 2).max(1);
        let base_seed = seed.wrapping_add(PRIME_1.wrapping_mul(ple_idx as u64));
        let multipliers = (0..n)
            .map(|i| {
                let v = base_seed.wrapping_add(GAMMA.wrapping_mul(i as u64 + 1));
                2 * (splitmix64(v) % half) + 1
            })
            .collect();
        Self { multipliers, head_sizes, head_offsets, padded_rows, eos, n, heads_per_ngram: hpn }
    }

    pub fn num_heads(&self) -> usize {
        self.head_sizes.len()
    }

    /// `ctx` (N-1 개 직전 id) 와 새 토큰들로 `[T, G]` n-gram 행 번호를 만듭니다 (행 우선으로 평탄화).
    pub fn ids(&self, ctx: &[u32], toks: &[u32]) -> Vec<u32> {
        let c = self.n - 1;
        debug_assert_eq!(ctx.len(), c);
        let hist: Vec<u32> = ctx.iter().chain(toks.iter()).copied().collect();
        let g = self.num_heads();
        let mut out = Vec::with_capacity(toks.len() * g);
        // 직전 eos 위치(엄격히 이전) 를 따라가며 계산합니다.
        let mut prev_eos: Option<usize> = None;
        for (pos, &tok) in hist.iter().enumerate() {
            if pos >= c {
                let mut s = vec![0u64; self.n];
                s[0] = tok as u64;
                for (k, sk) in s.iter_mut().enumerate().skip(1) {
                    // valid ⇔ pos-k ≥ 0 이고 [pos-k, pos-1] 안에 eos 가 없음 ⇔ prev_eos < pos-k
                    let valid = pos >= k && prev_eos.map_or(true, |e| e < pos - k);
                    *sk = if valid { hist[pos - k] as u64 } else { self.eos as u64 };
                }
                for ng in 2..=self.n {
                    let mut mix = s[0].wrapping_mul(self.multipliers[0]);
                    for j in 1..ng {
                        mix ^= s[j].wrapping_mul(self.multipliers[j]);
                    }
                    for h in 0..self.heads_per_ngram {
                        let gi = (ng - 2) * self.heads_per_ngram + h;
                        out.push((mix % self.head_sizes[gi] + self.head_offsets[gi]) as u32);
                    }
                }
            }
            if tok == self.eos {
                prev_eos = Some(pos);
            }
        }
        out
    }
}

/// n-gram 임베딩 표.
pub enum NgramTable {
    /// 장치에 통째로 올린 `[rows, Dh]`
    Dense(Tensor),
    /// mmap 된 safetensors 샤드에서 필요한 행만 모읍니다 (기본 설정 표는 층당 ~82 GB).
    Mmap { st: Arc<MmapedSafetensors>, shards: Vec<(String, usize)>, dh: usize },
}

impl NgramTable {
    /// `ids` → `[n, Dh]` (dtype, device)
    pub fn gather(&self, ids: &[u32], dtype: DType, device: &Device) -> Result<Tensor> {
        match self {
            NgramTable::Dense(t) => {
                let idx = Tensor::from_slice(ids, ids.len(), t.device())?;
                Ok(t.index_select(&idx, 0)?.to_dtype(dtype)?)
            }
            NgramTable::Mmap { st, shards, dh } => {
                let mut rows: Vec<Tensor> = Vec::with_capacity(ids.len());
                for &id in ids {
                    let mut r = id as usize;
                    let mut found = None;
                    for (name, n) in shards {
                        if r < *n {
                            found = Some(name);
                            break;
                        }
                        r -= *n;
                    }
                    let name = found.ok_or_else(|| anyhow!("[QWEN4-EXP] n-gram 행 {id} 이 표 범위를 벗어났습니다"))?;
                    let view = st.get(name)?;
                    let st_dt: DType = view.dtype().try_into()?;
                    let es = st_dt.size_in_bytes();
                    let bytes = &view.data()[r * dh * es..(r + 1) * dh * es];
                    rows.push(Tensor::from_raw_buffer(bytes, st_dt, &[1, *dh], &Device::Cpu)?.to_dtype(dtype)?);
                }
                Ok(Tensor::cat(&rows, 0)?.to_device(device)?)
            }
        }
    }
}

/// PLE 층 (`layers.{i}.ple.*`).
pub struct PleLayer {
    pub hasher: NgramHasher,
    table: NgramTable,
    key_proj: Linear,
    value_proj: Linear,
    norm_key: RmsNorm1p,
    norm_query: RmsNorm1p,
    norm_conv: RmsNorm1p,
    conv_w: Tensor, // [hc*H, Kp]
    hc: usize,
    hidden: usize,
    kernel: usize,
    dil: usize,
    conv_state: Option<Tensor>, // [B, hc*H, (Kp-1)*N]
    ngram_ctx: Vec<Vec<u32>>,   // [B][N-1]
}

impl PleLayer {
    /// `prefix`: 이 층 `ple` 모듈의 전체 이름 (mmap 표에서 샤드 이름을 찾을 때 씁니다).
    pub fn new(
        vb: VarBuilder,
        c: &Qwen4ExpTextConfig,
        ple_idx: usize,
        st: Option<&Arc<MmapedSafetensors>>,
        prefix: &str,
    ) -> Result<Self> {
        let h = c.hidden_size;
        let hc = c.hc_count;
        let e = c.ple_dim();
        let hasher = NgramHasher::new(
            c.vocab_size,
            c.ngram_size,
            c.heads_per_ngram,
            c.ngram_vocab_size_base,
            c.make_ngram_vocab_size_divisible_by,
            c.seed,
            ple_idx,
            c.ple_eos(),
        );
        let g = hasher.num_heads();
        ensure!(g > 0 && e % g == 0, "[QWEN4-EXP] ple_embed_dim {e} % n-gram heads {g} != 0");
        let dh = e / g;
        let emb_prefix = format!("{prefix}.ple_embedding");

        // ── 체크포인트 버퍼 대조 (정수 해시는 전부 아니면 전무) ──
        if let Some(st) = st {
            let check = |name: &str, want: &[u64]| -> Result<()> {
                let full = format!("{emb_prefix}.{name}");
                if st.get(&full).is_err() {
                    return Ok(());
                }
                let got = st.load(&full, &Device::Cpu)?.to_dtype(DType::I64)?.to_vec1::<i64>()?;
                let got: Vec<u64> = got.into_iter().map(|v| v as u64).collect();
                if got != want {
                    bail!("[QWEN4-EXP] {full} 불일치: 체크포인트 {got:?} vs 계산 {want:?}");
                }
                Ok(())
            };
            check("layer_multipliers", &hasher.multipliers)?;
            check("ngram_heads_vocab_sizes", &hasher.head_sizes)?;
            check("ngram_heads_offsets", &hasher.head_offsets)?;
        }

        // ── n-gram 표: 단일 weight 또는 shard_{k}.weight 를 dim 0 으로 이어 붙인 것 ──
        let single = format!("{emb_prefix}.ngram_embedding.weight");
        let mut shards: Vec<(String, usize)> = Vec::new();
        let mut table_dtype_size = 4usize;
        if let Some(st) = st {
            if let Ok(v) = st.get(&single) {
                shards.push((single.clone(), v.shape()[0]));
                table_dtype_size = DType::try_from(v.dtype()).map(|d| d.size_in_bytes()).unwrap_or(4);
            } else {
                for k in 0..c.split_ngram_parts.max(1) {
                    let name = format!("{emb_prefix}.ngram_embedding.shard_{k}.weight");
                    match st.get(&name) {
                        Ok(v) => {
                            table_dtype_size = DType::try_from(v.dtype()).map(|d| d.size_in_bytes()).unwrap_or(4);
                            shards.push((name, v.shape()[0]));
                        }
                        Err(_) => break,
                    }
                }
            }
        }
        let rows: usize = shards.iter().map(|(_, n)| *n).sum();
        let vbe = vb.pp("ple_embedding").pp("ngram_embedding");
        let table = if st.is_some() && !shards.is_empty() && rows * dh * table_dtype_size > dense_table_limit() {
            ensure!(rows >= hasher.padded_rows, "[QWEN4-EXP] n-gram 표 행 {rows} < 필요 {}", hasher.padded_rows);
            println!(
                "[QWEN4-EXP] 🧮 PLE#{ple_idx} n-gram 표 {rows}행 × {dh} ({} MB) 는 mmap 행 수집으로 씁니다.",
                rows * dh * table_dtype_size / (1024 * 1024)
            );
            NgramTable::Mmap { st: st.unwrap().clone(), shards, dh }
        } else if vbe.contains_tensor("weight") {
            NgramTable::Dense(vbe.get((hasher.padded_rows, dh), "weight")?)
        } else {
            let mut parts = Vec::new();
            let mut k = 0;
            while vbe.contains_tensor(&format!("shard_{k}.weight")) {
                let n = shards.get(k).map(|s| s.1).unwrap_or(hasher.padded_rows / c.split_ngram_parts.max(1));
                parts.push(vbe.get((n, dh), &format!("shard_{k}.weight"))?);
                k += 1;
            }
            ensure!(!parts.is_empty(), "[QWEN4-EXP] PLE#{ple_idx} n-gram 표를 찾을 수 없습니다 ({emb_prefix})");
            let t = Tensor::cat(&parts, 0)?;
            ensure!(
                t.dim(0)? == hasher.padded_rows,
                "[QWEN4-EXP] n-gram 표 행 {} != 계산 {}",
                t.dim(0)?,
                hasher.padded_rows
            );
            NgramTable::Dense(t)
        };

        let hch = hc * h;
        let key_proj = linear_no_bias(vb.clone(), e, hch, "key_proj")?;
        let value_proj = linear_no_bias(vb.clone(), e, h, "value_proj")?;
        let norm_key = RmsNorm1p::new(vb.pp("norm_key"), hch, Some(h), c.rms_norm_eps)?;
        let norm_query = RmsNorm1p::new(vb.pp("norm_query"), hch, Some(h), c.rms_norm_eps)?;
        let norm_conv = RmsNorm1p::new(vb.pp("norm_conv"), hch, Some(h), c.rms_norm_eps)?;
        let kp = c.ple_conv_kernel_size;
        let conv_w = vb.get((hch, 1, kp), "conv1d.weight")?.reshape((hch, kp))?;
        Ok(Self {
            hasher,
            table,
            key_proj,
            value_proj,
            norm_key,
            norm_query,
            norm_conv,
            conv_w,
            hc,
            hidden: h,
            kernel: kp,
            dil: c.ngram_size,
            conv_state: None,
            ngram_ctx: Vec::new(),
        })
    }

    pub fn clear_cache(&mut self) {
        self.conv_state = None;
        self.ngram_ctx.clear();
    }

    fn state_len(&self) -> usize {
        (self.kernel - 1) * self.dil
    }

    /// `x: [B, T, hc*H]`, `ids[b]`: 이번 청크의 토큰 id (길이 T) → PLE 출력 `[B, T, hc*H]` (x 에 더할 값)
    pub fn forward(&mut self, x: &Tensor, ids: &[Vec<u32>]) -> Result<Tensor> {
        let (bsz, t, f) = x.dims3()?;
        let dt = x.dtype();
        let dev = x.device().clone();
        ensure!(ids.len() == bsz, "[QWEN4-EXP] PLE ids 배치 {} != {bsz}", ids.len());
        if self.ngram_ctx.len() != bsz {
            self.ngram_ctx = vec![vec![self.hasher.eos; self.hasher.n - 1]; bsz];
        }
        let mut all = Vec::with_capacity(bsz * t * self.hasher.num_heads());
        for (b, toks) in ids.iter().enumerate() {
            ensure!(toks.len() == t, "[QWEN4-EXP] PLE ids 길이 {} != T {t}", toks.len());
            all.extend(self.hasher.ids(&self.ngram_ctx[b], toks));
            let c = self.hasher.n - 1;
            let mut hist: Vec<u32> = self.ngram_ctx[b].iter().chain(toks.iter()).copied().collect();
            let keep = hist.split_off(hist.len() - c);
            self.ngram_ctx[b] = keep;
        }
        let emb = self.table.gather(&all, dt, &dev)?.reshape((bsz, t, ()))?; // [B, T, E]

        let (h, hc) = (self.hidden, self.hc);
        let key = self.norm_key.forward(&self.key_proj.forward(&emb)?)?.reshape((bsz, t, hc, h))?;
        let value = self.value_proj.forward(&emb)?; // [B, T, H]
        let query = self.norm_query.forward(x)?.reshape((bsz, t, hc, h))?;
        let gate = ((key * query)?.sum_keepdim(D::Minus1)? / (h as f64).sqrt())?; // [B, T, hc, 1]
        let sign = (gate.gt(0.0)?.to_dtype(dt)? - gate.lt(0.0)?.to_dtype(dt)?)?;
        let mag = gate.abs()?.maximum(1e-6)?.sqrt()?;
        let gate = (mag * sign)?;
        let gv = sigmoid(&gate)?.broadcast_mul(&value.unsqueeze(2)?)?; // [B, T, hc, H]
        let gv = gv.reshape((bsz, t, f))?;
        let gvn = self.norm_conv.forward(&gv)?;

        // 팽창 depthwise conv: out[c, t] = Σ_k w[c, k] · padded[c, t + k·dil]
        let s = self.state_len();
        let xin = gvn.transpose(1, 2)?.contiguous()?; // [B, C, T]
        let prev = match &self.conv_state {
            Some(st) => st.clone(),
            None => Tensor::zeros((bsz, f, s), dt, &dev)?,
        };
        let padded = Tensor::cat(&[&prev, &xin], 2)?; // [B, C, S+T]
        self.conv_state = Some(padded.narrow(2, t, s)?.contiguous()?);
        let mut conv: Option<Tensor> = None;
        for k in 0..self.kernel {
            let wk = self.conv_w.narrow(1, k, 1)?.reshape((1, f, 1))?;
            let tap = padded.narrow(2, k * self.dil, t)?.broadcast_mul(&wk)?;
            conv = Some(match conv {
                None => tap,
                Some(acc) => (acc + tap)?,
            });
        }
        let conv = conv.expect("kernel >= 1").silu()?.transpose(1, 2)?; // [B, T, C]
        Ok((gv + conv)?)
    }
}

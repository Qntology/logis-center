//! Qwen Sparse Attention (QSA) — gated full attention + 블록 압축 키 인덱서 (qwen4-spec §7 / Q4-03).
//!
//! 본 어텐션은 Qwen3.5 gated attention 과 같습니다: `q_proj → [q | gate]`(head 별), q/k (+1) RMSNorm,
//! 부분 RoPE(NeoX, 앞 rd 차원), GQA, `scale = hd^-0.5`, `o_proj(attn · sigmoid(gate))`.
//! 차이는 마스크뿐이며 `causal AND selected` 입니다. 인덱서는 절대 위치 p 의 쿼리마다
//!  - `nb = (p+1) / C` 개의 완성 블록 키 `kb = RoPE_at(bC)(k_norm(mean_f32(raw_k[bC..bC+C]).to(dt)))`
//!  - `score_b = Σ_h relu(q_h · kb) / sqrt(dI)` (F32, head 합 이전에 ReLU)
//!  - `topk(min(budget/C, nb))` 블록 ∪ 꼬리 `[nb·C, p]` (불완전 마지막 블록은 항상 포함)
//! 을 고릅니다. `p+1 < budget + C` 이면 모든 블록이 선택되어 밀집 어텐션과 같습니다.
//! `(p+1) % C == 0` 이고 자기 블록이 top-k 에 없으면 쿼리가 자기 자신을 보지 않는데, HF 동작 그대로입니다.
//!
//! 캐시: K/V (norm·RoPE 이후), 인덱서 원시 키 `[B, L, dI]` (norm·RoPE 이전), 완성 블록 압축 키 `[B, L/C, dI]`
//! (norm·RoPE 이후, 한 번 완성되면 변하지 않으므로 증분 저장이 정확합니다).
//! top-k 는 CPU 에서 (점수 내림차순, 같은 점수는 낮은 블록 번호 우선) 결정적으로 고릅니다 (Q4-12).

use anyhow::{ensure, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::kv_cache::{Cache, KvCache};
use candle_nn::{Linear, VarBuilder};

use super::config::Qwen4ExpTextConfig;
use super::layers::{apply_rope, linear_b, linear_no_bias, sigmoid, RmsNorm1p, RopeTable};

/// 프리필 어텐션을 쿼리 행 단위로 나눠 `[B, H, rows, KV]` 순간 메모리를 묶어 둡니다.
const PREFILL_ROW_TILE: usize = 256;
const KV_GROW: usize = 512;
/// 인덱서 점수 타일의 원소 수 상한 (`rows · nI · NB`, F32 기준 약 128 MB).
const SCORE_TILE_ELEMS: usize = 32 * 1024 * 1024;

/// 쿼리 행 하나의 선택 결과.
#[derive(Debug, Clone, PartialEq)]
pub enum RowSelection {
    /// 인과 범위 `[0, p]` 전체
    Dense,
    /// 선택된 토큰 위치 (오름차순)
    Sparse(Vec<u32>),
}

/// 인덱서 (`self_attn.indexer.*`).
pub struct QsaIndexer {
    qk_proj: Linear,
    q_norm: RmsNorm1p,
    k_norm: RmsNorm1p,
    n_heads: usize,
    dim: usize,
    budget: usize,
    ratio: usize,
    topk_blocks: usize,
    raw_keys: Cache,  // [B, L, dI]
    comp_keys: Cache, // [B, NB, dI]
}

impl QsaIndexer {
    pub fn new(vb: VarBuilder, c: &Qwen4ExpTextConfig) -> Result<Self> {
        let n_heads = c.indexer_n_heads.unwrap_or(0);
        let kv_heads = c.indexer_kv_heads.unwrap_or(1);
        let dim = c.indexer_head_dim.unwrap_or(0);
        let budget = c.indexer_budget.unwrap_or(0);
        let ratio = c.indexer_compress_ratio.unwrap_or(1).max(1);
        let qk_proj = linear_no_bias(vb.clone(), c.hidden_size, (n_heads + kv_heads) * dim, "index_qk_proj")?;
        let q_norm = RmsNorm1p::new(vb.pp("q_layernorm"), dim, None, c.rms_norm_eps)?;
        let k_norm = RmsNorm1p::new(vb.pp("k_layernorm"), dim, None, c.rms_norm_eps)?;
        Ok(Self {
            qk_proj,
            q_norm,
            k_norm,
            n_heads,
            dim,
            budget,
            ratio,
            topk_blocks: budget / ratio,
            raw_keys: Cache::new(1, KV_GROW),
            comp_keys: Cache::new(1, KV_GROW / ratio.max(1) + 1),
        })
    }

    pub fn clear_cache(&mut self) {
        self.raw_keys.reset();
        self.comp_keys.reset();
    }

    pub fn cached_len(&self) -> usize {
        self.raw_keys.current_seq_len()
    }

    /// `m: [B, T, H]`, 쿼리 절대 위치 `offset..offset+T`. 반환: `sel[b][t]`.
    pub fn select(&mut self, m: &Tensor, rope: &RopeTable, offset: usize) -> Result<Vec<Vec<RowSelection>>> {
        let (bsz, t, _) = m.dims3()?;
        let dt = m.dtype();
        let qk = self.qk_proj.forward(m)?;
        let q = qk.narrow(2, 0, self.n_heads * self.dim)?.reshape((bsz, t, self.n_heads, self.dim))?;
        let raw_k = qk.narrow(2, self.n_heads * self.dim, self.dim)?.contiguous()?; // [B, T, dI]
        let q = self.q_norm.forward(&q)?;
        let (cos, sin) = rope.slice(offset, t)?;
        let rd = cos.dim(1)?;
        let q = apply_rope(&q, &cos.reshape((1, t, 1, rd))?, &sin.reshape((1, t, 1, rd))?)?;

        ensure!(
            self.raw_keys.current_seq_len() == offset,
            "[QWEN4-EXP] QSA 인덱서 캐시 길이 {} != seqlen_offset {}",
            self.raw_keys.current_seq_len(),
            offset
        );
        self.raw_keys.append(&raw_k)?;
        let total = offset + t;

        // 새로 완성된 블록의 압축 키를 덧붙입니다.
        let nb_old = self.comp_keys.current_seq_len();
        let nb_new = total / self.ratio;
        if nb_new > nb_old {
            let all_raw = self.raw_keys.current_data()?.expect("raw keys appended");
            let n = nb_new - nb_old;
            let groups = all_raw
                .narrow(1, nb_old * self.ratio, n * self.ratio)?
                .contiguous()?
                .reshape((bsz, n, self.ratio, self.dim))?;
            let pooled = groups.to_dtype(DType::F32)?.mean(2)?.to_dtype(dt)?; // [B, n, dI]
            let pooled = self.k_norm.forward(&pooled)?;
            let starts: Vec<u32> = (nb_old..nb_new).map(|b| (b * self.ratio) as u32).collect();
            let (bc, bs) = rope.gather(&starts)?;
            let kb = apply_rope(&pooled, &bc.reshape((1, n, rd))?, &bs.reshape((1, n, rd))?)?;
            self.comp_keys.append(&kb.contiguous()?)?;
        }

        if total < self.budget + self.ratio {
            return Ok(vec![vec![RowSelection::Dense; t]; bsz]);
        }

        // 점수: [B, T, nI, dI] · [B, NB, dI]ᵀ → relu → Σ_h / sqrt(dI)
        // 긴 프리필에서 [B, T·nI, NB] 가 커지지 않도록 쿼리 행을 나눠 계산합니다.
        let comp = self.comp_keys.current_data()?.expect("blocks exist").contiguous()?.to_dtype(DType::F32)?;
        let nb_all = comp.dim(1)?;
        let comp_t = comp.transpose(1, 2)?.contiguous()?; // [B, dI, NB]
        let qf = q.to_dtype(DType::F32)?.contiguous()?; // [B, T, nI, dI]
        let tile = (SCORE_TILE_ELEMS / (self.n_heads * nb_all).max(1)).clamp(1, t);
        let mut scores: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(t); bsz];
        let mut r0 = 0;
        while r0 < t {
            let n = (t - r0).min(tile);
            let qt = qf.narrow(1, r0, n)?.reshape((bsz, n * self.n_heads, self.dim))?;
            let st = qt.matmul(&comp_t)?.reshape((bsz, n, self.n_heads, nb_all))?.relu()?.sum(2)?;
            let st = (st / (self.dim as f64).sqrt())?.to_device(&Device::Cpu)?.to_vec3::<f32>()?;
            for (b, rows) in st.into_iter().enumerate() {
                scores[b].extend(rows);
            }
            r0 += n;
        }

        let mut out = Vec::with_capacity(bsz);
        for sb in scores.iter() {
            let mut rows = Vec::with_capacity(t);
            for (ti, row) in sb.iter().enumerate() {
                let p = offset + ti;
                let nb = (p + 1) / self.ratio;
                if nb <= self.topk_blocks {
                    rows.push(RowSelection::Dense);
                    continue;
                }
                let mut order: Vec<usize> = (0..nb).collect();
                order.sort_by(|&x, &y| row[y].partial_cmp(&row[x]).unwrap_or(std::cmp::Ordering::Equal).then(x.cmp(&y)));
                let mut blocks: Vec<usize> = order[..self.topk_blocks].to_vec();
                blocks.sort_unstable();
                let mut toks = Vec::with_capacity(self.topk_blocks * self.ratio + self.ratio);
                for blk in blocks {
                    for j in 0..self.ratio {
                        toks.push((blk * self.ratio + j) as u32);
                    }
                }
                for j in nb * self.ratio..=p {
                    toks.push(j as u32);
                }
                rows.push(RowSelection::Sparse(toks));
            }
            out.push(rows);
        }
        Ok(out)
    }

    pub fn budget(&self) -> usize {
        self.budget
    }
}

/// QSA 어텐션 층 (`self_attn.*`).
pub struct QsaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm1p,
    k_norm: RmsNorm1p,
    pub indexer: QsaIndexer,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    scale: f64,
    kv: KvCache, // [B, nKV, L, hd]
    /// 마지막 forward 의 선택 결과 (검증/진단용)
    pub last_selection: Vec<Vec<RowSelection>>,
}

impl QsaAttention {
    pub fn new(vb: VarBuilder, c: &Qwen4ExpTextConfig) -> Result<Self> {
        let h = c.hidden_size;
        let hd = c.head_dim();
        let (nh, nkv) = (c.num_attention_heads, c.num_key_value_heads);
        let bias = c.attention_bias;
        Ok(Self {
            q_proj: linear_b(vb.clone(), h, nh * hd * 2, "q_proj", bias)?,
            k_proj: linear_b(vb.clone(), h, nkv * hd, "k_proj", bias)?,
            v_proj: linear_b(vb.clone(), h, nkv * hd, "v_proj", bias)?,
            o_proj: linear_b(vb.clone(), nh * hd, h, "o_proj", bias)?,
            q_norm: RmsNorm1p::new(vb.pp("q_norm"), hd, None, c.rms_norm_eps)?,
            k_norm: RmsNorm1p::new(vb.pp("k_norm"), hd, None, c.rms_norm_eps)?,
            indexer: QsaIndexer::new(vb.pp("indexer"), c)?,
            n_heads: nh,
            n_kv: nkv,
            hd,
            scale: (hd as f64).powf(-0.5),
            kv: KvCache::new(2, KV_GROW),
            last_selection: Vec::new(),
        })
    }

    pub fn clear_cache(&mut self) {
        self.kv.reset();
        self.indexer.clear_cache();
        self.last_selection.clear();
    }

    pub fn kv_len(&self) -> usize {
        self.kv.current_seq_len()
    }

    /// `m: [B, T, H]` (절대 위치 offset..offset+T) → `[B, T, H]`
    pub fn forward(&mut self, m: &Tensor, rope: &RopeTable, offset: usize) -> Result<Tensor> {
        let (bsz, t, _) = m.dims3()?;
        let sel = self.indexer.select(m, rope, offset)?;

        let qg = self.q_proj.forward(m)?.reshape((bsz, t, self.n_heads, 2 * self.hd))?;
        let q = qg.narrow(3, 0, self.hd)?;
        let gate = qg.narrow(3, self.hd, self.hd)?.reshape((bsz, t, self.n_heads * self.hd))?;
        let q = self.q_norm.forward(&q.contiguous()?)?.transpose(1, 2)?; // [B, nH, T, hd]
        let k = self
            .k_norm
            .forward(&self.k_proj.forward(m)?.reshape((bsz, t, self.n_kv, self.hd))?)?
            .transpose(1, 2)?;
        let v = self.v_proj.forward(m)?.reshape((bsz, t, self.n_kv, self.hd))?.transpose(1, 2)?;

        let (cos, sin) = rope.slice(offset, t)?;
        let rd = cos.dim(1)?;
        let (c4, s4) = (cos.reshape((1, 1, t, rd))?, sin.reshape((1, 1, t, rd))?);
        let q = apply_rope(&q, &c4, &s4)?.contiguous()?;
        let k = apply_rope(&k, &c4, &s4)?.contiguous()?;
        let v = v.contiguous()?;

        ensure!(
            self.kv.current_seq_len() == offset,
            "[QWEN4-EXP] QSA KV 캐시 길이 {} != seqlen_offset {}",
            self.kv.current_seq_len(),
            offset
        );
        let (k_all, v_all) = self.kv.append(&k, &v)?; // [B, nKV, L, hd]
        let total = offset + t;

        let mut outs = Vec::with_capacity(bsz);
        for bi in 0..bsz {
            let qb = q.narrow(0, bi, 1)?;
            let kb = k_all.narrow(0, bi, 1)?.contiguous()?;
            let vb = v_all.narrow(0, bi, 1)?.contiguous()?;
            let rows = &sel[bi];
            let o = if t == 1 {
                // 디코드: 선택 위치만 모아 마스크 없이 softmax (마스크 softmax 와 정확히 같음).
                match &rows[0] {
                    RowSelection::Dense => self.attend(&qb, &kb, &vb, None)?,
                    RowSelection::Sparse(idx) => {
                        let ids = Tensor::from_slice(idx.as_slice(), idx.len(), m.device())?;
                        let ks = kb.index_select(&ids, 2)?;
                        let vs = vb.index_select(&ids, 2)?;
                        self.attend(&qb, &ks, &vs, None)?
                    }
                }
            } else {
                let mut tiles = Vec::new();
                let mut r0 = 0;
                while r0 < t {
                    let n = (t - r0).min(PREFILL_ROW_TILE);
                    let mask = build_mask(&rows[r0..r0 + n], offset + r0, total, m.device())?;
                    let qt = qb.narrow(2, r0, n)?.contiguous()?;
                    tiles.push(self.attend(&qt, &kb, &vb, Some(&mask))?);
                    r0 += n;
                }
                if tiles.len() == 1 { tiles.pop().unwrap() } else { Tensor::cat(&tiles, 2)? }
            };
            outs.push(o);
        }
        self.last_selection = sel;
        let attn = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 0)? }; // [B, nH, T, hd]
        let attn = attn.transpose(1, 2)?.reshape((bsz, t, self.n_heads * self.hd))?;
        let attn = (attn * sigmoid(&gate)?)?;
        Ok(self.o_proj.forward(&attn)?)
    }

    /// HF eager: `softmax_f32(q kᵀ · scale + mask).to(dt) · v`, GQA 는 q 를 `[nKV, rep·T]` 로 접어 처리.
    /// q: `[1, nH, T, hd]`, k/v: `[1, nKV, S, hd]`, mask: `[T, S]` 가산(F32, 0 / -inf).
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, nh, t, hd) = q.dims4()?;
        let s = k.dim(2)?;
        let rep = nh / self.n_kv;
        let dt = q.dtype();
        let qg = q.reshape((b, self.n_kv, rep * t, hd))?;
        let att = (qg.matmul(&k.transpose(2, 3)?)? * self.scale)?; // [b, nKV, rep*T, S]
        let att = att.to_dtype(DType::F32)?;
        let att = match mask {
            Some(m) => att.reshape((b, self.n_kv, rep, t, s))?.broadcast_add(m)?.reshape((b, self.n_kv, rep * t, s))?,
            None => att,
        };
        let p = candle_nn::ops::softmax_last_dim(&att)?.to_dtype(dt)?;
        let o = p.matmul(v)?; // [b, nKV, rep*T, hd]
        Ok(o.reshape((b, nh, t, hd))?)
    }
}

/// 쿼리 행들(절대 위치 `p0..`)의 가산 마스크 `[rows, total]` (허용 0, 금지 -inf).
fn build_mask(rows: &[RowSelection], p0: usize, total: usize, dev: &Device) -> Result<Tensor> {
    let n = rows.len();
    let mut m = vec![f32::NEG_INFINITY; n * total];
    for (r, sel) in rows.iter().enumerate() {
        let p = p0 + r;
        let row = &mut m[r * total..(r + 1) * total];
        match sel {
            RowSelection::Dense => {
                for x in row.iter_mut().take(p + 1) {
                    *x = 0.0;
                }
            }
            RowSelection::Sparse(idx) => {
                for &j in idx {
                    if (j as usize) <= p {
                        row[j as usize] = 0.0;
                    }
                }
            }
        }
    }
    Ok(Tensor::from_vec(m, (n, total), dev)?)
}

/// 선택 결과를 `[T, total]` 0/1 마스크로 펼칩니다 (검증용).
pub fn selection_mask(rows: &[RowSelection], offset: usize, total: usize) -> Vec<Vec<u8>> {
    rows.iter()
        .enumerate()
        .map(|(r, sel)| {
            let p = offset + r;
            let mut v = vec![0u8; total];
            match sel {
                RowSelection::Dense => v.iter_mut().take(p + 1).for_each(|x| *x = 1),
                RowSelection::Sparse(idx) => idx.iter().for_each(|&j| v[j as usize] = 1),
            }
            v
        })
        .collect()
}

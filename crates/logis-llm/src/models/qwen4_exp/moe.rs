//! Sparse MoE 블록 (`Qwen4ExpTextSparseMoeBlock`, qwen4-spec §9 / Q4-05).
//!
//!  - 라우터: `logits = x Wrᵀ`(모델 dtype), `p = softmax_f32(전체 전문가)`, top-k, `norm_topk_prob` 이면 재정규화,
//!    가중치를 모델 dtype 으로 캐스팅.
//!  - 전문가 e: `[gate | up] = x gate_up[e]ᵀ` (gate 가 앞 I 행), `y_e = down[e](silu(gate)·up) · w`.
//!  - 공유 전문가: `sigmoid(x Wsgᵀ) · down(silu(gate(x))·up(x))`.
//! top-k 는 CPU 에서 결정적으로 고르고 (확률 내림차순, 같으면 낮은 번호 우선), 토큰을 전문가별로 묶어
//! 실제로 선택된 전문가만 계산합니다 (전문가 오름차순으로 index_add — HF 누적 순서와 같음).
//!
//! 가중치 형식: 체크포인트 `experts.{e}.{gate,up,down}_proj.weight` 와 런타임 융합
//! `experts.gate_up_proj [E, 2I, H]` / `experts.down_proj [E, H, I]` 둘 다 읽습니다.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use super::config::Qwen4ExpTextConfig;
use super::layers::{linear_no_bias, sigmoid};

pub struct SparseMoe {
    router: Tensor,          // [E, H]
    gate_up: Vec<Tensor>,    // E × [2I, H]
    down: Vec<Tensor>,       // E × [H, I]
    shared_gate_proj: Linear,
    shared_up_proj: Linear,
    shared_down_proj: Linear,
    shared_expert_gate: Linear,
    top_k: usize,
    norm_topk: bool,
    inter: usize,
}

impl SparseMoe {
    pub fn new(vb: VarBuilder, c: &Qwen4ExpTextConfig) -> Result<Self> {
        let (h, i, e) = (c.hidden_size, c.moe_intermediate_size, c.num_experts);
        let router = vb.get((e, h), "gate.weight")?;
        let ve = vb.pp("experts");
        let (gate_up, down) = if ve.contains_tensor("gate_up_proj") {
            let gu = ve.get((e, 2 * i, h), "gate_up_proj")?;
            let dn = ve.get((e, h, i), "down_proj")?;
            let mut a = Vec::with_capacity(e);
            let mut b = Vec::with_capacity(e);
            for x in 0..e {
                a.push(gu.get(x)?);
                b.push(dn.get(x)?);
            }
            (a, b)
        } else {
            let mut a = Vec::with_capacity(e);
            let mut b = Vec::with_capacity(e);
            for x in 0..e {
                let vx = ve.pp(x.to_string());
                let g = vx.get((i, h), "gate_proj.weight")?;
                let u = vx.get((i, h), "up_proj.weight")?;
                a.push(Tensor::cat(&[&g, &u], 0)?);
                b.push(vx.get((h, i), "down_proj.weight")?);
            }
            (a, b)
        };
        let vs = vb.pp("shared_expert");
        let is = c.shared_expert_intermediate_size;
        Ok(Self {
            router,
            gate_up,
            down,
            shared_gate_proj: linear_no_bias(vs.clone(), h, is, "gate_proj")?,
            shared_up_proj: linear_no_bias(vs.clone(), h, is, "up_proj")?,
            shared_down_proj: linear_no_bias(vs.clone(), is, h, "down_proj")?,
            shared_expert_gate: linear_no_bias(vb.clone(), h, 1, "shared_expert_gate")?,
            top_k: c.num_experts_per_tok,
            norm_topk: c.norm_topk_prob,
            inter: i,
        })
    }

    /// 토큰별 (전문가, 가중치 f32) top-k 목록.
    pub fn route(&self, x2: &Tensor) -> Result<Vec<Vec<(usize, f32)>>> {
        let logits = x2.matmul(&self.router.t()?)?; // [N, E] 모델 dtype
        let probs = candle_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
        let probs = probs.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        Ok(probs
            .into_iter()
            .map(|row| {
                let mut idx: Vec<usize> = (0..row.len()).collect();
                idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
                let mut sel: Vec<(usize, f32)> = idx[..self.top_k].iter().map(|&e| (e, row[e])).collect();
                if self.norm_topk {
                    let s: f32 = sel.iter().map(|(_, w)| *w).sum();
                    for (_, w) in sel.iter_mut() {
                        *w /= s;
                    }
                }
                sel
            })
            .collect())
    }

    /// `x: [B, T, H]` → `[B, T, H]`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, h) = x.dims3()?;
        let dt = x.dtype();
        let dev = x.device();
        let x2 = x.reshape((b * t, h))?;
        let routes = self.route(&x2)?;

        // 전문가별 (토큰, 가중치) 묶음
        let ne = self.gate_up.len();
        let mut per: Vec<(Vec<u32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); ne];
        for (tok, sel) in routes.iter().enumerate() {
            for &(e, w) in sel {
                per[e].0.push(tok as u32);
                per[e].1.push(w);
            }
        }
        let mut out = Tensor::zeros((b * t, h), dt, dev)?;
        for (e, (toks, ws)) in per.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let n = toks.len();
            let idx = Tensor::from_slice(toks.as_slice(), n, dev)?;
            let xe = x2.index_select(&idx, 0)?; // [n, H]
            let gu = xe.matmul(&self.gate_up[e].t()?)?; // [n, 2I]
            let g = gu.narrow(D::Minus1, 0, self.inter)?;
            let u = gu.narrow(D::Minus1, self.inter, self.inter)?;
            let hdn = (g.silu()? * u)?;
            let ye = hdn.matmul(&self.down[e].t()?)?; // [n, H]
            // HF: routing weight 를 모델 dtype 으로 캐스팅한 뒤 곱합니다.
            let w = Tensor::from_slice(ws.as_slice(), (n, 1), dev)?.to_dtype(dt)?;
            let ye = ye.broadcast_mul(&w)?;
            out = out.index_add(&idx, &ye, 0)?;
        }

        let shared = self
            .shared_down_proj
            .forward(&(self.shared_gate_proj.forward(&x2)?.silu()? * self.shared_up_proj.forward(&x2)?)?)?;
        let sg = sigmoid(&self.shared_expert_gate.forward(&x2)?)?; // [N, 1]
        let y = (out + shared.broadcast_mul(&sg)?)?;
        Ok(y.reshape((b, t, h))?)
    }
}

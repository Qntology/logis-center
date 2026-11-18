//! GatedResidual 하이퍼커넥션 (`Qwen4ExpTextGatedResidual`, qwen4-spec §4 / Q4-02).
//!
//! 잔차 스트림은 `hc` 개 복사본을 스트림-우선 배치로 이어 붙인 `[B, T, hc*H]` 입니다.
//!  - `xn = GroupedRMS(x)` (H 단위 그룹, F32, (1+w))
//!  - `a = silu(down(xn) / hc)`, `g = sigmoid(up(a))` → `[hc, H]`
//!  - `mixed = mean_s(g_s ⊙ xn_s)`
//!  - `inj = 2·sigmoid(inject(xn) / hc)` → `[hc]` (최종 mixer 에는 없음)
//! 층 출력 y 는 `x + concat_s(inj_s · y)` 로 각 스트림에 주입됩니다. RMSNorm/최종 norm 은 따로 없습니다.

use anyhow::Result;
use candle_core::{Module, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use super::layers::{linear_no_bias, sigmoid, RmsNorm1p};

#[derive(Debug, Clone)]
pub struct GatedResidual {
    hc: usize,
    hidden: usize,
    norm: RmsNorm1p,
    down: Linear,
    up: Linear,
    inject: Option<Linear>,
}

impl GatedResidual {
    pub fn new(vb: VarBuilder, hc: usize, hidden: usize, lowrank: usize, eps: f64, with_inject: bool) -> Result<Self> {
        let hch = hc * hidden;
        let norm = RmsNorm1p::new(vb.pp("hc_norm"), hch, Some(hidden), eps)?;
        let down = linear_no_bias(vb.clone(), hch, lowrank, "input_mix_weight_down")?;
        let up = linear_no_bias(vb.clone(), lowrank, hch, "input_mix_weight_up")?;
        let inject = if with_inject {
            Some(linear_no_bias(vb.clone(), hch, hc, "block_inject_weight")?)
        } else {
            None
        };
        Ok(Self { hc, hidden, norm, down, up, inject })
    }

    /// `x: [B, T, hc*H]` → `(mixed [B, T, H], inj [B, T, hc] | None)`
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let (b, t, f) = x.dims3()?;
        anyhow::ensure!(
            f == self.hc * self.hidden,
            "[QWEN4-EXP] 하이퍼커넥션 입력 폭 {f} != hc*H {}",
            self.hc * self.hidden
        );
        let hc = self.hc as f64;
        let xn = self.norm.forward(x)?;
        let a = (self.down.forward(&xn)? / hc)?.silu()?;
        let g = sigmoid(&self.up.forward(&a)?)?.reshape((b, t, self.hc, self.hidden))?;
        let mixed = (g * xn.reshape((b, t, self.hc, self.hidden))?)?.sum(D::Minus2)?;
        let mixed = (mixed / hc)?;
        let inj = match &self.inject {
            Some(w) => Some((sigmoid(&(w.forward(&xn)? / hc)?)? * 2.0)?),
            None => None,
        };
        Ok((mixed, inj))
    }

    /// `x + flatten(y[.., None, :] * inj[.., :, None])`
    pub fn inject(x: &Tensor, y: &Tensor, inj: &Tensor) -> Result<Tensor> {
        let (b, t, f) = x.dims3()?;
        let hc = inj.dim(D::Minus1)?;
        let h = y.dim(D::Minus1)?;
        let add = y.unsqueeze(D::Minus2)?.broadcast_mul(&inj.unsqueeze(D::Minus1)?)?;
        debug_assert_eq!(hc * h, f);
        Ok((x + add.reshape((b, t, f))?)?)
    }
}

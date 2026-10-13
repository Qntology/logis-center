//! Qwen4-Exp 공용 레이어: (+1) RMSNorm / 그룹 RMSNorm, GDN gated RMSNorm, 부분 RoPE 표, 활성 함수.
//!
//! HF 캐스팅 순서를 그대로 따릅니다 (qwen4-spec §5, key_apis B):
//!  - `Qwen4ExpTextRMSNorm`: `(x_f32 * rsqrt(mean(x²)+eps)) * (1 + w_f32)` → 입력 dtype. `group_size` 가 있으면
//!    마지막 축을 `[.., G, group]` 으로 나눠 그룹마다 정규화합니다 (hc_norm, PLE norm_*).
//!  - `Qwen4ExpTextRMSNormGated`: `w * (x_f32 * rsqrt(var+eps)).to(dtype) * ACT(z_f32)` → dtype (w 에 +1 없음).

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;

use super::config::GateAct;

/// `Qwen4ExpTextRMSNorm` (+1 가중치, 선택적 그룹). 가중치는 미리 `1 + w` (F32) 로 저장합니다.
#[derive(Debug, Clone)]
pub struct RmsNorm1p {
    w1p: Tensor,
    group: Option<usize>,
    eps: f64,
}

impl RmsNorm1p {
    pub fn new(vb: VarBuilder, dim: usize, group: Option<usize>, eps: f64) -> Result<Self> {
        let w = vb.get(dim, "weight")?;
        Self::from_weight(&w, group, eps)
    }

    pub fn from_weight(w: &Tensor, group: Option<usize>, eps: f64) -> Result<Self> {
        let w1p = (w.to_dtype(DType::F32)? + 1.0)?;
        Ok(Self { w1p, group, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let normed = match self.group {
            Some(g) if g != x.dim(D::Minus1)? => {
                let dims = xf.dims().to_vec();
                let last = dims[dims.len() - 1];
                let mut gshape = dims[..dims.len() - 1].to_vec();
                gshape.push(last / g);
                gshape.push(g);
                let xg = xf.reshape(gshape)?;
                let n = rms_core(&xg, self.eps)?;
                n.reshape(dims)?
            }
            _ => rms_core(&xf, self.eps)?,
        };
        Ok(normed.broadcast_mul(&self.w1p)?.to_dtype(dt)?)
    }
}

/// `x * rsqrt(mean(x², -1) + eps)` (F32 입력 가정)
fn rms_core(x: &Tensor, eps: f64) -> Result<Tensor> {
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let inv = (var + eps)?.sqrt()?.recip()?;
    Ok(x.broadcast_mul(&inv)?)
}

/// GDN 출력용 gated RMSNorm (`Qwen4ExpTextRMSNormGated`).
#[derive(Debug, Clone)]
pub struct GatedRmsNorm {
    w: Tensor,
    eps: f64,
    act: GateAct,
}

impl GatedRmsNorm {
    pub fn new(vb: VarBuilder, dim: usize, eps: f64, act: GateAct) -> Result<Self> {
        Ok(Self { w: vb.get(dim, "weight")?, eps, act })
    }

    /// x, z: `[.., Dv]`
    pub fn forward(&self, x: &Tensor, z: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let n = rms_core(&x.to_dtype(DType::F32)?, self.eps)?.to_dtype(dt)?;
        let n = n.broadcast_mul(&self.w.to_dtype(dt)?)?;
        let zf = z.to_dtype(DType::F32)?;
        let g = match self.act {
            GateAct::Silu => zf.silu()?,
            GateAct::Sigmoid => candle_nn::ops::sigmoid(&zf)?,
        };
        Ok((n.to_dtype(DType::F32)? * g)?.to_dtype(dt)?)
    }
}

/// 텍스트 전용 1-D RoPE 표 (`[cap, rotary_dim]`, cos = cat(cos f, cos f)).
///
/// 텍스트만 넣으면 mRoPE 의 T/H/W 위치가 같으므로 interleaved mRoPE 와 수학적으로 동일합니다 (qwen4-spec Q4-11).
/// HF 와 같은 F32 연산 순서로 만든 뒤 모델 dtype 으로 내립니다. 인덱서가 과거 블록 위치의 cos/sin 을 쓰므로
/// 절대 위치로 색인합니다. 필요하면 두 배씩 늘립니다.
pub struct RopeTable {
    inv_freq: Vec<f32>,
    rot_dim: usize,
    cap: usize,
    cos: Option<Tensor>,
    sin: Option<Tensor>,
    dtype: DType,
    device: Device,
}

impl RopeTable {
    pub fn new(theta: f64, rot_dim: usize, dtype: DType, device: &Device) -> Self {
        let base = theta as f32;
        let inv_freq = (0..rot_dim / 2)
            .map(|i| 1.0f32 / base.powf((2 * i) as f32 / rot_dim as f32))
            .collect();
        Self { inv_freq, rot_dim, cap: 0, cos: None, sin: None, dtype, device: device.clone() }
    }

    pub fn rot_dim(&self) -> usize {
        self.rot_dim
    }

    /// 위치 `[0, need)` 를 덮도록 표를 늘립니다.
    pub fn ensure(&mut self, need: usize) -> Result<()> {
        if need <= self.cap && self.cos.is_some() {
            return Ok(());
        }
        let cap = need.max(self.cap * 2).max(256);
        let half = self.rot_dim / 2;
        let mut c = Vec::with_capacity(cap * self.rot_dim);
        let mut s = Vec::with_capacity(cap * self.rot_dim);
        for p in 0..cap {
            let pf = p as f32;
            let row_c: Vec<f32> = self.inv_freq.iter().map(|f| (pf * f).cos()).collect();
            let row_s: Vec<f32> = self.inv_freq.iter().map(|f| (pf * f).sin()).collect();
            debug_assert_eq!(row_c.len(), half);
            c.extend_from_slice(&row_c);
            c.extend_from_slice(&row_c);
            s.extend_from_slice(&row_s);
            s.extend_from_slice(&row_s);
        }
        let cos = Tensor::from_vec(c, (cap, self.rot_dim), &Device::Cpu)?
            .to_dtype(self.dtype)?
            .to_device(&self.device)?;
        let sin = Tensor::from_vec(s, (cap, self.rot_dim), &Device::Cpu)?
            .to_dtype(self.dtype)?
            .to_device(&self.device)?;
        self.cos = Some(cos);
        self.sin = Some(sin);
        self.cap = cap;
        Ok(())
    }

    /// 연속 위치 `[start, start+len)` 의 `(cos, sin)`: `[len, rot_dim]`
    pub fn slice(&self, start: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let (c, s) = self.tables()?;
        Ok((c.narrow(0, start, len)?, s.narrow(0, start, len)?))
    }

    /// 임의 위치 목록의 `(cos, sin)`: `[n, rot_dim]`
    pub fn gather(&self, positions: &[u32]) -> Result<(Tensor, Tensor)> {
        let (c, s) = self.tables()?;
        let idx = Tensor::from_slice(positions, positions.len(), &self.device)?;
        Ok((c.index_select(&idx, 0)?, s.index_select(&idx, 0)?))
    }

    fn tables(&self) -> Result<(&Tensor, &Tensor)> {
        match (&self.cos, &self.sin) {
            (Some(c), Some(s)) => Ok((c, s)),
            _ => anyhow::bail!("[QWEN4-EXP] RoPE 표가 아직 준비되지 않았습니다"),
        }
    }
}

/// NeoX `rotate_half` 부분 RoPE: 마지막 축의 앞 `rd` 차원만 회전하고 나머지는 그대로 둡니다.
///
/// `x`: `[.., T, (H,) D]` 꼴, `cos`/`sin`: x 의 축에 브로드캐스트되도록 미리 reshape 된 `[.., rd]`.
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let rd = cos.dim(D::Minus1)?;
    let xr = x.narrow(D::Minus1, 0, rd)?;
    let half = rd / 2;
    let x1 = xr.narrow(D::Minus1, 0, half)?;
    let x2 = xr.narrow(D::Minus1, half, half)?;
    let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    let out = (xr.broadcast_mul(cos)? + rot.broadcast_mul(sin)?)?;
    if rd < d {
        Ok(Tensor::cat(&[&out, &x.narrow(D::Minus1, rd, d - rd)?], D::Minus1)?)
    } else {
        Ok(out)
    }
}

/// torch `F.softplus` (threshold 20) 와 같은 안정형: `max(x,0) + log(1 + exp(-|x|))`
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    let pos = x.relu()?;
    let tail = (x.abs()?.neg()?.exp()? + 1.0)?.log()?;
    Ok((pos + tail)?)
}

pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::sigmoid(x)?)
}

/// 바이어스 없는 Linear 를 HF 이름(`{name}.weight`)으로 읽습니다.
pub fn linear_no_bias(vb: VarBuilder, in_dim: usize, out_dim: usize, name: &str) -> Result<candle_nn::Linear> {
    let w = vb.get((out_dim, in_dim), &format!("{name}.weight"))?;
    Ok(candle_nn::Linear::new(w, None))
}

/// 바이어스 선택 Linear.
pub fn linear_b(vb: VarBuilder, in_dim: usize, out_dim: usize, name: &str, bias: bool) -> Result<candle_nn::Linear> {
    let w = vb.get((out_dim, in_dim), &format!("{name}.weight"))?;
    let b = if bias { Some(vb.get(out_dim, &format!("{name}.bias"))?) } else { None };
    Ok(candle_nn::Linear::new(w, b))
}

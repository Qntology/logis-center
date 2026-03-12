use crate::models::layers::distributed::Shard;
use crate::models::layers::others::should_skip_fp8_for_module;
use crate::models::layers::VarBuilderX;
use crate::utils::config::QuantConfig;
use candle_core::quantized::QMatMul;
use candle_core::{DType, Module, Result, Tensor};

#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.matmul(&self.weight.t()?)?;
        if let Some(bias) = &self.bias {
            x.broadcast_add(bias)
        } else {
            Ok(x)
        }
    }
}

pub fn linear(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::var_builder::ShardedVarBuilder,
    shards: Shard,
    dtype: DType,
) -> Result<Linear> {
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", shards)?;
    let bias = if vb.contains_tensor("bias") {
        Some(vb.get_with_hints((out_dim,), "bias", shards)?)
    } else {
        None
    };
    Ok(Linear::new(weight.to_dtype(dtype)?, bias.map(|b| b.to_dtype(dtype)).transpose()?))
}

pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::var_builder::ShardedVarBuilder,
    shards: Shard,
    dtype: DType,
) -> Result<Linear> {
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", shards)?;
    Ok(Linear::new(weight.to_dtype(dtype)?, None))
}

pub fn linear_no_bias_merged(
    num_experts: usize,
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::var_builder::ShardedVarBuilder,
    shards: Shard,
    dtype: DType,
) -> Result<Linear> {
    let weight = vb.get_with_hints((num_experts, out_dim, in_dim), "weight", shards)?;
    Ok(Linear::new(weight.to_dtype(dtype)?, None))
}

#[derive(Debug, Clone)]
pub struct QLinear {
    pub inner: Option<QMatMul>,
    pub wna16: Option<WNA16>,
    pub bias: Option<Tensor>,
    pub dtype: DType,
}

#[derive(Debug, Clone)]
pub struct WNA16 {
    // Dummy for now, implementation depends on specific requirements
}

impl WNA16 {
    pub fn new(
        _in_dim: usize,
        _out_dim: usize,
        _vb: candle_nn::var_builder::ShardedVarBuilder,
        _shards: Shard,
        _quant_cfg: &Option<QuantConfig>,
        _bias: bool,
        _dtype: DType,
        _training: bool,
    ) -> Result<Self> {
        Ok(Self {})
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

impl QLinear {
    pub fn new(
        _in_dim: usize,
        _out_dim: usize,
        _vb: candle_nn::var_builder::ShardedVarBuilder,
        _shards: Shard,
        _dtype: DType,
    ) -> Result<Self> {
        candle_core::bail!("QLinear::new from ShardedVarBuilder not implemented")
    }

    pub fn new_fused(
        _num_experts: usize,
        _in_dim: usize,
        _out_dim: usize,
        _vb: candle_nn::var_builder::ShardedVarBuilder,
        _shards: Shard,
        _dtype: DType,
    ) -> Result<Self> {
        candle_core::bail!("QLinear::new_fused not implemented")
    }

    pub fn from_linear_x(ln: Linear, _quant_type: String, dtype: DType) -> Result<Self> {
        // Implementation for converting Linear to QLinear (placeholder)
        Ok(Self {
            inner: None,
            wna16: None,
            bias: ln.bias,
            dtype,
        })
    }

    pub fn dequantize(&self) -> Result<Tensor> {
        candle_core::bail!("Dequantize not implemented")
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(wna16) = &self.wna16 {
            return wna16.forward(x);
        }
        if let Some(inner) = &self.inner {
            let xs = if x.dtype() != DType::F32 {
                x.to_dtype(DType::F32)?
            } else {
                x.to_owned()
            };
            let xs = QMatMul::forward(inner, &xs)?;

            if let Some(bias) = &self.bias {
                xs.broadcast_add(bias)
            } else {
                Ok(xs)
            }
        } else {
            candle_core::bail!("Invalid quantization type!")
        }
    }
}

impl QLinear {
    pub fn indexed_moe_forward(&self, _x: &Tensor, _ids: &Tensor) -> Result<Tensor> {
        candle_core::bail!("indexed_moe_forward not implemented for QLinear in this fork")
    }
}

#[derive(Debug, Clone)]
pub enum LinearX {
    Linear(Linear),
    QLinear(QLinear),
    LnFp8(LnFp8),
}

impl Module for LinearX {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(ln) => ln.forward(x),
            Self::QLinear(ln) => ln.forward(x),
            Self::LnFp8(ln) => ln.forward(x),
        }
    }
}

impl LinearX {
    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(_) => {
                candle_core::bail!("Linear does not support indexed_moe_forward")
            }
            Self::QLinear(ln) => ln.indexed_moe_forward(x, ids),
            Self::LnFp8(_) => candle_core::bail!("LnFp8 does not support indexed_moe_forward yet"),
        }
    }
}

pub fn linear_x(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilderX,
    shards: Shard,
    quant_cfg: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    let module_path = vb.module_path().to_string();
    let vb_inner = vb.0.clone();
    
    if let Some(cfg) = quant_cfg {
        if cfg.quant_method == "fp8" {
            if should_skip_fp8_for_module(&module_path, cfg) {
                let ln = linear(in_dim, out_dim, vb_inner, shards, dtype)?;
                return Ok(LinearX::Linear(ln));
            }

            let has_fp8_scale = vb_inner.contains_tensor("weight_scale")
                || vb_inner.contains_tensor("weight_scale_inv");
            if !has_fp8_scale {
                let weight = vb_inner.get_with_hints((out_dim, in_dim), "weight", shards)?;
                if matches!(
                    weight.dtype(),
                    DType::BF16 | DType::F16 | DType::F32 | DType::F64
                ) {
                    let ln = linear(in_dim, out_dim, vb_inner, shards, dtype)?;
                    return Ok(LinearX::Linear(ln));
                }
            }

            match load_ln_fp8_with_hints(in_dim, out_dim, vb_inner, shards, cfg, true) {
                Ok(ln) => return Ok(LinearX::LnFp8(ln)),
                Err(err) => return Err(err),
            }
        }

        let wna16 = WNA16::new(
            in_dim,
            out_dim,
            vb_inner,
            shards,
            quant_cfg,
            true,
            dtype,
            true,
        )?;
        let ln = QLinear {
            inner: None,
            wna16: Some(wna16),
            bias: None,
            dtype,
        };
        Ok(LinearX::QLinear(ln))
    } else {
        let ln = linear(in_dim, out_dim, vb_inner, shards, dtype)?;
        if let Some(quantized_type) = quant {
            Ok(LinearX::QLinear(QLinear::from_linear_x(
                ln,
                quantized_type.clone(),
                dtype,
            )?))
        } else {
            Ok(LinearX::Linear(ln))
        }
    }
}

pub fn linear_no_bias_x(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilderX,
    shards: Shard,
    quant_cfg: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    let module_path = vb.module_path().to_string();
    let vb_inner = vb.0.clone();

    if let Some(cfg) = quant_cfg {
        if cfg.quant_method == "fp8" {
            if should_skip_fp8_for_module(&module_path, cfg) {
                let ln = linear_no_bias(in_dim, out_dim, vb_inner, shards, dtype)?;
                return Ok(LinearX::Linear(ln));
            }

            let has_fp8_scale = vb_inner.contains_tensor("weight_scale")
                || vb_inner.contains_tensor("weight_scale_inv");
            if !has_fp8_scale {
                let weight = vb_inner.get_with_hints((out_dim, in_dim), "weight", shards)?;
                if matches!(
                    weight.dtype(),
                    DType::BF16 | DType::F16 | DType::F32 | DType::F64
                ) {
                    let ln = linear_no_bias(in_dim, out_dim, vb_inner, shards, dtype)?;
                    return Ok(LinearX::Linear(ln));
                }
            }

            match load_ln_fp8_with_hints(in_dim, out_dim, vb_inner, shards, cfg, false) {
                Ok(ln) => return Ok(LinearX::LnFp8(ln)),
                Err(err) => return Err(err),
            }
        }

        let wna16 = WNA16::new(
            in_dim,
            out_dim,
            vb_inner,
            shards,
            quant_cfg,
            false,
            dtype,
            true,
        )?;
        let ln = QLinear {
            inner: None,
            wna16: Some(wna16),
            bias: None,
            dtype,
        };
        Ok(LinearX::QLinear(ln))
    } else {
        let ln = linear_no_bias(in_dim, out_dim, vb_inner, shards, dtype)?;
        if let Some(quantized_type) = quant {
            Ok(LinearX::QLinear(QLinear::from_linear_x(
                ln,
                quantized_type.clone(),
                dtype,
            )?))
        } else {
            Ok(LinearX::Linear(ln))
        }
    }
}

pub fn linear_no_bias_merged_x(
    num_experts: usize,
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilderX,
    shards: Shard,
    _: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    let vb_inner = vb.0.clone();
    let ln = linear_no_bias_merged(num_experts, in_dim, out_dim, vb_inner, shards, dtype)?;
    if let Some(quantized_type) = quant {
        Ok(LinearX::QLinear(QLinear::from_linear_x(
            ln,
            quantized_type.clone(),
            dtype,
        )?))
    } else {
        Ok(LinearX::Linear(ln))
    }
}

pub fn linear_b_x(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilderX,
    shard: Shard,
    quant_cfg: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    if bias {
        linear_x(in_dim, out_dim, vb, shard, quant_cfg, quant, dtype)
    } else {
        linear_no_bias_x(in_dim, out_dim, vb, shard, quant_cfg, quant, dtype)
    }
}

#[derive(Debug, Clone)]
pub struct LnFp8 {
    pub weight: Tensor,
    pub weight_scale: Tensor,
    pub weight_scale_cutlass: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub weight_block_size: Vec<usize>,
    pub sm_version: usize,
}

impl Module for LnFp8 {
    fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        candle_core::bail!("LnFp8::forward not implemented")
    }
}

impl LnFp8 {
    pub fn load_with_hints(
        _in_dim: usize,
        _out_dim: usize,
        _vb: VarBuilderX,
        _shards: Shard,
        _cfg: &QuantConfig,
        _dtype: DType,
    ) -> Result<Self> {
        candle_core::bail!("LnFp8::load_with_hints not implemented")
    }
}

pub fn load_ln_fp8_with_hints(
    _in_dim: usize,
    _out_dim: usize,
    _vb: candle_nn::var_builder::ShardedVarBuilder,
    _shards: Shard,
    _cfg: &QuantConfig,
    _bias: bool,
) -> Result<LnFp8> {
    candle_core::bail!("load_ln_fp8_with_hints not implemented")
}
